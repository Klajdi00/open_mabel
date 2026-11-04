"""
curation_recorder.py — bridge-side episode recorder in the DATA CURATION
STUDIO's exact input format.

Drop-in replacement for ``mabel.recorder.EpisodeRecorder`` in the teleop bridge
(same duck-type: set_task / status / count_episodes / list_tasks / start /
_write_frame / stop / .active / .fps / .task), but every episode is written as

    <out_dir>/<folder>/episode_XXXX.h5     (+ sibling dataset_meta.json)

by ``robot_learning/imitation_learning/data_collection/recorder.EpisodeRecorder`` —
the SAME writer the IL collection pipeline uses, so what the app records is
byte-compatible with what the Data Curation studio (robot_learning/data_curation)
ingests: attrs {schema_json, num_frames, episode_index, created, task}; keys
time/sim, time/wall, obs/cam/<name>/rgb (vlen JPEG), obs/cam/<name>/depth
(uint16 mm), obs/lidar, obs/joint_pos|joint_vel|joint_torque (actuator-order),
obs/qpos|qvel, obs/pose/{chassis,head,left_palm,right_palm}, obs/base_twist,
action/ctrl (+ action/qpos_des when the bridge streams desired joints).

The app's "recorded folder" == the curation studio's DATASET: point the studio
at this root to browse everything recorded from the headset:

    MABEL_DATASET_ROOT=<repo>/dataset/simulation/vision_os \
        python3 -m server  (robot_learning/data_curation)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("recorder")

_HERE = os.path.dirname(os.path.abspath(__file__))
_IL = os.path.normpath(os.path.join(_HERE, "..", "robot_learning", "imitation_learning"))


def _load(name, path):
    """Load an IL-pipeline module by FILE PATH — never via sys.path: the name
    `data_collection` is taken by the bridge's own server/data_collection.py
    (BagRecorder), and putting robot_learning/imitation_learning on sys.path shadows
    it (the 2026-07-11 bridge-crash-on-boot)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DC = _load("_mabel_dc_config",
            os.path.join(_IL, "data_collection", "config.py"))
_H5Writer = _load("_mabel_dc_recorder",
                  os.path.join(_IL, "data_collection", "recorder.py")).EpisodeRecorder

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                                   # pragma: no cover
    _HAVE_CV2 = False


# ── the operator's control state, per frame (2026-10-16) ─────────────────────
# Every frame records WHICH body parts were driving the robot, so a policy (or
# an ablation analysis) can condition on / filter by it — toggles can change
# mid-episode, so this is per frame, never only per episode. Numeric codes, not
# strings (h5py cannot store numpy unicode arrays); the legend rides the
# episode's schema_json as "control_encoding".
CONTROL_TRACKING = ("neck", "arms", "hands", "torso", "lift")
CONTROL_METHODS = ("navigation", "joint", "wrist", "retarget")
CONTROL_LEGEND = {
    "obs/control/tracking": {"dtype": "int8", "order": list(CONTROL_TRACKING),
                             "meaning": "1 = that operator body part drives its joints; "
                                        "0 = frozen (parked at home: neck level/forward, "
                                        "torso 0 rad, lift at home height). -1 = unknown"},
    "obs/control/scope": {"dtype": "int8", "values": {"0": "local (UPPER BODY: base out of loop)",
                                                     "1": "global (WHOLE BODY: base follows)",
                                                     "-1": "unknown"}},
    "obs/control/method": {"dtype": "int8", "values": {str(i): m for i, m in enumerate(CONTROL_METHODS)}
                           | {"-1": "other/unknown"}},
    "obs/control/flags": {"dtype": "int8", "order": ["gravity_comp", "whole_body_retarget", "estop"]},
    "obs/control/engaged": {"dtype": "int8",
                            "meaning": "1 = retargeter engaged (robot following the operator)"},
}


def encode_control(c):
    """Control-spec dict (the bridge's `_control_snapshot`) → obs/control/* arrays.
    Missing input → nothing (old bridges keep writing their old schema)."""
    if not isinstance(c, dict):
        return {}
    tr = c.get("tracking") or {}
    tracking = np.array([(-1 if tr.get(k) is None else int(bool(tr.get(k))))
                         for k in CONTROL_TRACKING], dtype=np.int8)
    sc = str(c.get("scope") or "").lower()
    scope = np.int8(1 if sc == "global" else 0 if sc == "local" else -1)
    m = str(c.get("method") or "")
    method = np.int8(CONTROL_METHODS.index(m) if m in CONTROL_METHODS else -1)
    flags = np.array([int(bool(c.get("gravity_comp", True))),
                      int(bool(c.get("whole_body_retarget", False))),
                      int(bool(c.get("estop", False)))], dtype=np.int8)
    engaged = np.int8(1 if c.get("engaged") else 0)
    return {"obs/control/tracking": tracking, "obs/control/scope": scope,
            "obs/control/method": method, "obs/control/flags": flags,
            "obs/control/engaged": engaged}


class CurationRecorder:
    """Buffers RecordPump samples into curation-format HDF5 episodes."""

    #: `?client=` platform tag → sibling dataset root. Vision Pro and iPhone
    #: recordings stay separated for downstream curation; anything else uses
    #: the configured root as-is.
    _CLIENT_SUBDIR = {"visionos": "vision_os", "ios": "ios", "ipados": "ios"}

    def __init__(self, out_dir: str, fps: float = 10.0, model=None):
        self.out_dir = os.path.abspath(os.path.expanduser(out_dir))
        os.makedirs(self.out_dir, exist_ok=True)
        self._root_parent = os.path.dirname(self.out_dir)
        self.fps = max(1.0, float(fps))
        self.model = model
        self.active = False
        self.task = "default"
        self.ep_dir = None                 # last saved episode path (API parity)
        self._writer = None
        self._n = 0
        self._last_t = -1e9
        # actuator → qpos/qvel address maps (per-actuator joint state, aligned
        # with action/ctrl — the ordering downstream training consumes).
        self._qadr, self._vadr = [], []
        if model is not None:
            import mujoco
            for i in range(model.nu):
                jid = -1
                if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                    jid = int(model.actuator_trnid[i, 0])
                elif model.actuator(i).name == "Lift":
                    try:
                        jid = int(model.joint(_DC.LIFT_JOINT).id)
                    except Exception:
                        jid = -1
                self._qadr.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
                self._vadr.append(int(model.jnt_dofadr[jid]) if jid >= 0 else -1)

    # ── task folders / status (same wire shape the apps already parse) ────────
    @staticmethod
    def _safe(name, fallback="default"):
        s = "".join(c for c in (name or "") if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        return s or fallback

    def set_task(self, task):
        self.task = self._safe(task)

    def set_source_client(self, tag):
        """Route the dataset root by the recording CLIENT (the `?client=`
        platform tag): Vision Pro → <parent>/vision_os, iPhone → <parent>/ios.
        Unknown/empty tags keep the current root. Never switches mid-episode."""
        if self.active:
            return
        sub = self._CLIENT_SUBDIR.get((tag or "").strip().lower())
        if not sub:
            return
        root = os.path.join(self._root_parent, sub)
        if root != self.out_dir:
            self.out_dir = root
            os.makedirs(self.out_dir, exist_ok=True)

    def count_episodes(self, task=None):
        d = os.path.join(self.out_dir, self._safe(task or self.task))
        if not os.path.isdir(d):
            return 0
        return sum(1 for n in os.listdir(d)
                   if n.startswith("episode_") and n.endswith(".h5"))

    def list_tasks(self):
        if not os.path.isdir(self.out_dir):
            return []
        return sorted(n for n in os.listdir(self.out_dir)
                      if os.path.isdir(os.path.join(self.out_dir, n)) and not n.startswith("."))

    def episode_index(self):
        """Index of the episode being recorded (or the next one) in the current
        folder. Reads the cached writer's counter while it exists, because the
        files on disk lag it: finish() only QUEUES the H5 save, so counting
        files right after a stop can return the index that was just used."""
        if self._writer is not None and getattr(self._writer, "dir", None) == \
                os.path.join(self.out_dir, self.task):
            return int(self._writer.ep_count)
        return self.count_episodes()

    def status(self):
        tasks = self.list_tasks()
        counts = {t: self.count_episodes(t) for t in tasks}
        return {"task": self.task, "episodeCount": self.count_episodes(),
                "isRecording": self.active, "tasks": tasks,
                "taskCounts": counts, "total": sum(counts.values()),
                "dir": self.out_dir,
                # Additive: the 0-based index of the episode in progress, so a
                # client can label "Episode N" without guessing from counts.
                "episodeIndex": self.episode_index() if self.active else None}

    # ── lifecycle (driven by RecordPump on its worker thread) ─────────────────
    def start(self, task, meta):
        if self.active:
            self.stop()
        self.set_task(task)
        folder = os.path.join(self.out_dir, self.task)
        os.makedirs(folder, exist_ok=True)
        # One cached writer per folder: its background save worker stays alive
        # across episodes (finish() only QUEUES the H5 flush). Re-create only on
        # a folder switch — the join then happens on the pump worker thread,
        # never the sim thread.
        if self._writer is not None and getattr(self._writer, "dir", None) != folder:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._writer is None:
            self._writer = _H5Writer(folder, self._schema(meta))
        self._writer.start()
        self._n = 0
        self._last_t = -1e9
        self.active = True
        log.info("REC ▶ %s → %s  (%.0f fps, curation h5)", self.task, folder, self.fps)

    def _schema(self, meta):
        """schema_json in the collection pipeline's exact shape."""
        rgb = meta.get("rgb_cameras", []) or []
        dep = set(meta.get("depth_cameras", []) or [])
        w, h = (meta.get("rgb_res") or [424, 240])[:2]
        groups = {}
        for g, attr in (("base", "BASE_ACTUATORS"), ("lift_torso", "LIFT_TORSO"),
                        ("neck", "NECK"), ("left_arm", "LEFT_ARM"),
                        ("left_hand", "LEFT_HAND"), ("right_arm", "RIGHT_ARM"),
                        ("right_hand", "RIGHT_HAND")):
            if hasattr(_DC, attr):
                groups[g] = getattr(_DC, attr)
        return {
            "source": "sim",
            "recorded_by": "vision_os_teleop",
            "scene": meta.get("scene", ""),
            "actuator_names": meta.get("actuator_names", []),
            "actuator_groups": groups,
            "cameras": [{"name": c, "rgb": True, "depth": c in dep, "w": int(w), "h": int(h)}
                        for c in rgb],
            "lidar": {"n_rays": int(meta.get("lidar_rays", 0)),
                      "angle_res_deg": getattr(_DC, "LIDAR_ANGLE_RES_DEG", 1.0),
                      "units": "metres", "cutoff_is_no_hit": True},
            "depth_encoding": {"dtype": "uint16", "scale": _DC.DEPTH_SCALE,
                               "max_m": _DC.DEPTH_MAX_M, "units": "millimetres"},
            "nq": int(meta.get("nq", 0)), "nv": int(meta.get("nv", 0)),
            "nu": int(meta.get("nu", 0)),
            # How to decode obs/control/* (see encode_control).
            "control_encoding": CONTROL_LEGEND,
        }

    def maybe_record(self, t, sample_fn):
        if not self.active or (t - self._last_t) < (1.0 / self.fps):
            return
        self._last_t = t
        try:
            self._write_frame(t, sample_fn())
        except Exception as e:              # never let recording crash teleop
            log.warning("REC frame %d skipped: %s", self._n, e)

    def _write_frame(self, t, s):
        obs = {}
        for cam, img in (s.get("rgb") or {}).items():
            if not _HAVE_CV2:
                continue
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), _DC.JPEG_QUALITY])
            if ok:
                obs[f"obs/cam/{cam}/rgb"] = jpg.flatten()
        for cam, dep in (s.get("depth") or {}).items():
            mm = np.clip(dep, 0.0, _DC.DEPTH_MAX_M) * _DC.DEPTH_SCALE
            obs[f"obs/cam/{cam}/depth"] = mm.astype(np.uint16)
        if s.get("lidar") is not None:
            obs["obs/lidar"] = np.asarray(s["lidar"], dtype=np.float32)
        qpos, qvel = s.get("qpos"), s.get("qvel")
        if qpos is not None and self._qadr:
            obs["obs/joint_pos"] = np.array(
                [qpos[a] if a >= 0 else np.nan for a in self._qadr], dtype=np.float32)
        if qvel is not None and self._vadr:
            obs["obs/joint_vel"] = np.array(
                [qvel[a] if a >= 0 else np.nan for a in self._vadr], dtype=np.float32)
        if s.get("act_force") is not None:
            obs["obs/joint_torque"] = np.asarray(s["act_force"], dtype=np.float32)
        if qpos is not None:
            obs["obs/qpos"] = np.asarray(qpos, dtype=np.float32)
        if qvel is not None:
            obs["obs/qvel"] = np.asarray(qvel, dtype=np.float32)
        for key, pose in (s.get("poses") or {}).items():
            obs[f"obs/pose/{key}"] = np.asarray(pose, dtype=np.float32)
        if s.get("base_twist") is not None:
            obs["obs/base_twist"] = np.asarray(s["base_twist"], dtype=np.float32)
        obs.update(encode_control(s.get("control")))
        act = {}
        if s.get("ctrl") is not None:
            act["action/ctrl"] = np.asarray(s["ctrl"], dtype=np.float32)
        if s.get("qpos_des") is not None:
            act["action/qpos_des"] = np.asarray(s["qpos_des"], dtype=np.float32)
        self._writer.append(obs, act, sim_time=float(t), wall_time=time.time())
        self._n += 1

    def stop(self):
        """Finalise → returns the saved episode path (None if nothing recorded)."""
        if not self.active:
            return None
        self.active = False
        # finish() only QUEUES the save on the writer's background worker —
        # never join here (this can run on the sim thread via pump.stop()).
        path = self._writer.finish(task=self.task) if self._writer else None
        self.ep_dir = path
        if path:
            log.info("REC ■ saved %s (%d frames)", path, self._n)
        else:
            log.info("REC ■ discarded (0 frames)")
        return path
