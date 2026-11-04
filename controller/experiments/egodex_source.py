"""EgoDex (Apple Vision Pro / ARKit) -> MABEL operator-pose source.

Turns real Vision Pro recordings into the exact payload the live headset
streams, so the retargeting stack can be exercised against *human* motion
instead of the synthetic sweeps in `exp_front_workspace.py`.

    dataset: EgoDex, Hoque et al. 2025, arXiv:2505.11709 (Apple).
             829 h of egocentric Vision Pro recordings, 30 Hz, ARKit pose
             annotations for head + both hands + upper body, 194 tasks.
    licence: CC-by-NC-ND. NON-COMMERCIAL, NO-DERIVATIVES. The raw parquet
             cache under `data/egodex/` is gitignored and MUST NOT be
             redistributed with this repo; re-download it with
             `exp_egodex_retarget.py --fetch` instead of copying it around.
    mirror:  hf://datasets/D1ckyTwist3r/EgoDex (LeRobot v3.0 re-encode of
             Apple's HDF5 release; the columns below are that mirror's).

WHY THIS MATTERS
----------------
The volumetric grid in `exp_front_workspace.py` is an *adversarial* test: it
commands axis-aligned palm rolls at every cell of a box, including poses a
human would never ask for. It is the right test for coverage, and a poor
test for realism -- it cannot tell you whether the retargeter feels natural,
because none of its input is natural. EgoDex is the complement: every frame
is a real person's wrist, moving the way wrists actually move.

CONVENTIONS -- ALL FOUR WERE MEASURED, NOT ASSUMED
--------------------------------------------------
The mirror's column names are only half the story, and two of them are
actively misleading. Each convention below was pinned by a discriminating
test against the data (numbers are from `pour`, 11 717 frames):

1. `*_orient_world` and the per-joint rotations are **row-major** 3x3.
   Test: express the wrist->middle-metacarpal vector in the wrist frame.
   The metacarpal is nearly rigid to the wrist, so the correct reading is
   the one with no scatter. Row-major puts it at local +X, 86 mm, sd 4.1 mm;
   column-major scatters it over 25 mm in a direction that drifts. Row-major
   also reproduces the ARKit hand-anchor convention (+X along the fingers).

2. `extrinsics_w2c` really is **world->camera** (invert it for the head pose)
   AND its camera frame is **OpenCV** (X right, Y **down**, Z **forward**),
   not the ARKit/OpenGL frame MABEL expects (X right, Y up, -Z forward).
   Test: which axis of which reading points at the hands? Inverted +Z wins
   at 37.6 deg mean; every other candidate is >= 43 deg, and the -Z that the
   ARKit convention would imply is 142 deg -- i.e. behind the operator.
   Corroboration: the winning reading is pitched 20.4 deg BELOW horizontal
   and its +Y axis points 69 deg down, which is exactly self-consistent for
   a seated tabletop task viewed through an OpenCV-convention camera.
   Hence `head_mat = inv(extrinsics_w2c) @ diag(1, -1, -1, 1)`.

3. The world frame is already ARKit's: **Y up**, gravity-aligned, origin on
   the floor at session start. Head y sits at 0.99-1.35 m and wrists at
   0.71-1.12 m, which is a seated person, so no world rotation is needed --
   `AVP_TO_ROBOT` consumes this frame directly.

4. `*_finger_poses` is **25 joints x 12 floats** (3 translation + 9 rotation),
   in ARKit `HandSkeleton.JointName` order MINUS the wrist (which lives in
   the separate `*_transl_world` / `*_orient_world` columns), PLUS the
   forearm at the end. Test: distance-to-wrist is a sawtooth of one 4-chain
   then four 5-chains then a lone 265 mm joint; chain lengths rank
   161/158/154/141 mm (middle > index > ring > little, the human ordering)
   and adjacent knuckles sit 17-19 mm apart across a 53 mm palm.
   Index 24 at 265 mm is the forearm's elbow end -- which is a gift, because
   MABEL's arm IK has an elbow-direction objective that the headset path
   feeds from exactly this joint (`wire._FOREARM_ARM`).

MABEL wants 21 landmarks and ARKit gives 26, so the five metacarpals are
dropped (`_NAMES_21` has knuckle/base/tip/tip per finger, no metacarpal).

Usage
-----
    eps = load_task("controller/experiments/data/egodex/pour")
    for p in eps[0].payloads():
        pose = OperatorPose.from_payload(p)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# EgoDex 25-joint layout -> MABEL's 21 wire names (see _NAMES_21 in the engine).
# Verified empirically; see convention 4 in the module docstring.
# ---------------------------------------------------------------------------
FINGER_INDEX = {
    "thumb_knuckle": 0, "thumb_intermediate_base": 1,
    "thumb_intermediate_tip": 2, "thumb_tip": 3,
    # metacarpals (4, 9, 14, 19) are deliberately dropped -- MABEL has no slot.
    "index_knuckle": 5, "index_intermediate_base": 6,
    "index_intermediate_tip": 7, "index_tip": 8,
    "middle_knuckle": 10, "middle_intermediate_base": 11,
    "middle_intermediate_tip": 12, "middle_tip": 13,
    "ring_knuckle": 15, "ring_intermediate_base": 16,
    "ring_intermediate_tip": 17, "ring_tip": 18,
    "little_knuckle": 20, "little_intermediate_base": 21,
    "little_intermediate_tip": 22, "little_tip": 23,
}
FOREARM_INDEX = 24                      # elbow end of the forearm
FOREARM_NAME = "forearm_arm"            # the name mabel.wire looks for

# _NAMES_21 order, mirrored here so this module imports nothing from the engine
# (it must stay usable without MuJoCo). Guarded by test_egodex_source.py.
NAMES_21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]

# OpenCV camera (X right, Y down, Z fwd) -> ARKit camera (X right, Y up, -Z fwd).
# Right-multiplied, so it relabels the camera's OWN axes and leaves the world
# frame and the head position untouched. See convention 2.
OPENCV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])

FPS = 30.0


def _mat4(t, r9):
    """(3,) translation + (9,) row-major rotation -> 4x4 rigid transform."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(r9, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, float)
    return T


@dataclass
class Episode:
    """One EgoDex recording, converted to MABEL operator payloads on demand."""
    task: str
    index: int                  # episode_index within the task
    instruction: str
    head: np.ndarray            # (N,4,4) ARKit-convention head pose, ARKit world
    wrist: dict                 # {"left"/"right": (N,4,4)}
    fingers: dict               # {"left"/"right": (N,25,4,4)}

    @property
    def n(self):
        return len(self.head)

    @property
    def duration(self):
        return self.n / FPS

    def payload(self, i, t0=0.0):
        """Frame i as the exact dict the Vision Pro publisher sends.

        Matches `mabel.sources.avp.AVPFrame` / `OperatorPose.from_payload`:
        head_mat, {side}_hand_mats (21, [0]=wrist), {side}_joints (named,
        including forearm_arm so the elbow objective is fed), {side}_elbow,
        {side}_pinch, timestamp.
        """
        p = {"timestamp": float(t0 + i / FPS),
             "head_mat": self.head[i].flatten().tolist()}
        for side in ("left", "right"):
            W = self.wrist[side][i]
            F = self.fingers[side][i]
            named = {"wrist": W}
            for name, idx in FINGER_INDEX.items():
                named[name] = F[idx]
            named[FOREARM_NAME] = F[FOREARM_INDEX]
            p[f"{side}_hand_mats"] = [named[n].flatten().tolist() for n in NAMES_21]
            p[f"{side}_joints"] = {k: v.flatten().tolist() for k, v in named.items()}
            p[f"{side}_elbow"] = F[FOREARM_INDEX].flatten().tolist()
            # thumb-index tip gap drives the clutch / grasp scalar
            p[f"{side}_pinch"] = float(np.linalg.norm(F[3, :3, 3] - F[8, :3, 3]))
        return p

    def payloads(self, t0=0.0):
        for i in range(self.n):
            yield self.payload(i, t0)

    # --- motion descriptors, used to pick and to label clips ----------------
    def wrist_speed(self, side):
        """Per-frame wrist linear speed [m/s]."""
        p = self.wrist[side][:, :3, 3]
        return np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS

    def wrist_omega(self, side):
        """Per-frame wrist angular speed [deg/s], from the rotation increment."""
        R = self.wrist[side][:, :3, :3]
        dR = np.einsum('nij,nkj->nik', R[1:], R[:-1])          # R_{t+1} R_t^T
        tr = np.clip((np.trace(dR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(tr)) * FPS

    def head_omega(self):
        """Per-frame head angular speed [deg/s]."""
        R = self.head[:, :3, :3]
        dR = np.einsum('nij,nkj->nik', R[1:], R[:-1])
        tr = np.clip((np.trace(dR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(tr)) * FPS

    def gaze(self):
        """Head gaze direction in the ARKit world frame (unit, (N,3)).

        MABEL's convention: the head looks along its own -Z (frames.GAZE_LOCAL).
        """
        g = -self.head[:, :3, 2]
        return g / np.linalg.norm(g, axis=1, keepdims=True)

    def palm_normal(self, side):
        """Palm normal in the ARKit world frame (unit, (N,3)).

        In the ARKit hand-anchor frame the finger axis is local +X for the
        LEFT hand and -X for the right (measured: the index knuckle sits at
        local x = +86 mm on the left, -86 mm on the right -- the two hands are
        mirrored, not translated), and the palm faces along the anchor's +Y on
        the left / -Y on the right.

        The sign is physical, not assumed: for these four tabletop tasks the
        outward palm normal must point mostly DOWNWARD (you work over a
        table). This reading does, on every task and both hands -- left
        83-98 % of frames palm-down, right 87-97 % -- while the opposite sign
        would have the operator working with both palms turned to the ceiling.
        """
        s = -1.0 if side == "right" else +1.0
        n = s * self.wrist[side][:, :3, 1]
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def summary(self):
        d = {"task": self.task, "episode": int(self.index), "frames": self.n,
             "duration_s": round(self.duration, 2), "instruction": self.instruction}
        for side in ("left", "right"):
            d[f"{side}_speed_p95_mps"] = round(float(np.percentile(self.wrist_speed(side), 95)), 3)
            d[f"{side}_omega_p95_dps"] = round(float(np.percentile(self.wrist_omega(side), 95)), 1)
            d[f"{side}_roll_range_deg"] = round(float(self.palm_roll_range(side)), 1)
        d["head_omega_p95_dps"] = round(float(np.percentile(self.head_omega(), 95)), 1)
        d["gaze_sweep_deg"] = round(float(self.gaze_sweep()), 1)
        return d

    def palm_roll_range(self, side):
        """Angular spread of the palm normal over the clip [deg] -- how much
        the wrist actually re-orients. This is the number that says whether a
        clip exercises the palm-flip case at all."""
        n = self.palm_normal(side)
        m = n.mean(0)
        m /= np.linalg.norm(m) + 1e-12
        return np.degrees(np.arccos(np.clip(n @ m, -1, 1))).max() * 2.0

    def gaze_sweep(self):
        """Angular spread of the gaze over the clip [deg]."""
        g = self.gaze()
        m = g.mean(0)
        m /= np.linalg.norm(m) + 1e-12
        return np.degrees(np.arccos(np.clip(g @ m, -1, 1))).max() * 2.0


def load_task(task_dir, max_episodes=None):
    """Load every episode in one EgoDex task directory (LeRobot layout)."""
    import pyarrow.parquet as pq

    task = os.path.basename(os.path.normpath(task_dir))
    files = []
    data_root = os.path.join(task_dir, "data")
    for root, _, names in os.walk(data_root):
        files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"no parquet under {data_root} -- run exp_egodex_retarget.py --fetch")

    eps = []
    for f in files:
        d = pq.ParquetFile(f).read().to_pandas()
        for ei in sorted(d["episode_index"].unique()):
            if max_episodes is not None and len(eps) >= max_episodes:
                return eps
            g = d[d["episode_index"] == ei]
            E = np.stack([np.asarray(v, float).reshape(4, 4) for v in g["extrinsics_w2c"]])
            head = np.linalg.inv(E) @ OPENCV_TO_ARKIT          # conventions 2 + 3
            wrist, fingers = {}, {}
            for side in ("left", "right"):
                wrist[side] = np.stack([
                    _mat4(t, r) for t, r in zip(g[f"{side}_transl_world"],
                                                g[f"{side}_orient_world"])])
                fp = np.stack([np.asarray(v, float).reshape(25, 12)
                               for v in g[f"{side}_finger_poses"]])
                fingers[side] = np.stack([[_mat4(j[:3], j[3:]) for j in fr] for fr in fp])
            eps.append(Episode(task=task, index=int(ei),
                               instruction=str(g["instruction"].iloc[0]),
                               head=head, wrist=wrist, fingers=fingers))
    return eps


def save_clip(ep, path, note=""):
    """Write one episode as a self-describing JSON clip of VP wire payloads.

    The clip is a DERIVED, reformatted excerpt of a CC-by-NC-ND dataset: keep
    it for local regression testing, do not redistribute it.
    """
    obj = {
        "source": "EgoDex (Apple), arXiv:2505.11709, CC-by-NC-ND -- do not redistribute",
        "task": ep.task,
        "episode_index": int(ep.index),
        "instruction": ep.instruction,
        "note": note,
        "fps": FPS,
        "frames": ep.n,
        "duration_s": round(ep.duration, 2),
        "format": "MABEL Vision Pro wire payload (mabel.sources.avp / OperatorPose.from_payload)",
        "frame_convention": "ARKit world (Y up); head_mat looks along its own -Z",
        "metrics": ep.summary(),
        "payloads": list(ep.payloads()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # GZIP BY DEFAULT at scale. A clip is ~2.5 MB of JSON and compresses ~8x;
    # at 100 clips that is 250 MB against 30 MB, and this machine's disk is
    # already at 100 % use. `load_clip` sniffs the extension, so callers do not
    # care which form is on disk.
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "wt") as fh:
            json.dump(obj, fh)
    else:
        with open(path, "w") as fh:
            json.dump(obj, fh)
    return os.path.getsize(path)


def load_clip(path):
    """Read back a saved clip -> (metadata dict, list of payloads).

    Accepts the plain and gzipped forms interchangeably, and falls back to the
    other extension if the exact path is absent, so a caller that hard-coded
    `.json` still finds a `.json.gz` written later.
    """
    import gzip
    if not os.path.exists(path):
        alt = path[:-3] if path.endswith(".gz") else path + ".gz"
        if os.path.exists(alt):
            path = alt
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        obj = json.load(fh)
    return obj, obj["payloads"]
