"""Per-operator body & hand calibration for MABEL VR teleop.

A one-time, session-start calibration measures the operator's anthropometry from
the SAME sparse signals the headset already streams (head pose + two wrists + the
27-joint ARKit hand skeleton) and condenses them into an :class:`OperatorProfile`.
That profile then parameterises the retargeting scale, the shoulder anchor, and the
per-finger hand retargeter so teleop maps 1:1 across operators of different size.

Why this exists
---------------
The legacy mapping used a single fixed workspace gain (``config.RETARGET_SCALE``)
and re-estimated the operator's finger bone lengths live every solve. A tall and a
short operator therefore covered MABEL's workspace differently, and the fingers were
noisy. Here we *measure* the operator once and freeze it.

Method (3 guided poses, ~10 s)
------------------------------
All geometry is expressed in a **yaw-only, gravity-aligned operator body frame**
built from the live head pose (:func:`body_frame`): axes ``[right, up, forward]``,
so head pitch/roll never contaminate a "lateral" or "vertical" measurement.

1. **T-pose** (arms horizontal to the sides) -> per-side wrist body coords. In a
   T-pose the shoulder shares the wrist's height and forward offset (the arm is
   purely lateral), so the wrist's up/forward coords localise the shoulder, and the
   lateral wrist distance gives the arm length once the shoulder half-width is known.
2. **Arms-forward** (both arms straight ahead, parallel) -> the lateral separation
   of the two forward-pointing wrists is the biacromial **shoulder width** (the
   user-observed trick), plus a reach measurement.
3. **Open hand + pinch** -> per-finger bone lengths (canonical-normalised, reusing
   :func:`hand_ik_retarget.human_keypoints`), palm length / hand-size scale, and the
   thumb-index pinch open/close span.

Anthropometric fallbacks (e.g. shoulder width from stature) use the canonical
Winter / Drillis-Contini segment ratios in ``config`` (``ANTHRO_*``). Per the
NASA-STD-3000 percentile caveat, every directly *measurable* quantity (arm length,
shoulder width) is measured; ratios are only a fallback for untracked segments.

The inverse problem (frames -> dimensions) lives here as pure functions
(:func:`measure_tpose`, :func:`measure_arms_forward`, :func:`measure_hands`,
:func:`build_profile`) so it is unit-testable offline; :class:`BodyCalibrator`
wraps them in a streaming hold-detection state machine. A matching forward model
(:func:`synthesize_pose`) generates synthetic operators for the tests/experiments.

References: EgoPoser per-user T-pose ratios (Jiang et al., ECCV 2024,
arXiv:2308.06493); ACE workspace scaling (Yang et al., CoRL 2024, arXiv:2408.11805);
AnyTeleop hand-size alpha (Qin et al., RSS 2023, arXiv:2307.04577); Winter,
"Biomechanics and Motor Control of Human Movement".
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .. import config as C
from ..hand_retargeting.keypoint_ik import human_keypoints, _ALL_FINGERS, _WIRE_FINGER

log = logging.getLogger("mabel.calibration")

# ARKit / visionOS world frame is gravity-aligned with +Y up (see types.py).
ARKIT_UP = np.array([0.0, 1.0, 0.0])

# Pose identifiers for the guided capture sequence.
POSE_TPOSE = "tpose"
POSE_FORWARD = "arms_forward"
POSE_HANDS = "hands"
POSE_SEQUENCE = (POSE_TPOSE, POSE_FORWARD, POSE_HANDS)

# Finger metric-length chains on the ARKit wire skeleton (knuckle -> ... -> tip).
_FINGER_CHAIN = {
    f: [f"{_WIRE_FINGER.get(f, f)}_knuckle",
        f"{_WIRE_FINGER.get(f, f)}_intermediate_base",
        f"{_WIRE_FINGER.get(f, f)}_intermediate_tip",
        f"{_WIRE_FINGER.get(f, f)}_tip"]
    for f in ("index", "middle", "ring", "pinky")
}
_FINGER_CHAIN["thumb"] = ["thumb_knuckle", "thumb_intermediate_base",
                          "thumb_intermediate_tip", "thumb_tip"]


# ── operator body frame (single source of truth, shared with the retargeter) ──

def body_frame(head):
    """Yaw-only, gravity-aligned operator frame from a 4x4 head pose.

    Returns ``(origin, B)`` where ``origin`` is the head position and ``B`` has the
    world-space ``[right, up, forward]`` axes as columns. Body coords of a world
    point ``p`` are ``B.T @ (p - origin)`` -> ``[right, up, forward]``. Built from
    the head's heading flattened to horizontal, so head pitch/roll do not tilt the
    frame (a "lateral" measurement stays lateral even if the operator looks down)."""
    head = np.asarray(head, float)
    origin = head[:3, 3].copy()
    R = head[:3, :3]
    fwd = -R[:, 2]                                  # ARKit camera looks along -Z
    fwd = fwd - ARKIT_UP * float(fwd @ ARKIT_UP)    # flatten to horizontal
    n = np.linalg.norm(fwd)
    if n < 1e-6:                                     # looking straight up/down
        fwd = -R[:, 1]
        fwd = fwd - ARKIT_UP * float(fwd @ ARKIT_UP)
        n = np.linalg.norm(fwd)
    fwd = fwd / (n + 1e-9)
    right = np.cross(fwd, ARKIT_UP)
    right = right / (np.linalg.norm(right) + 1e-9)
    B = np.column_stack([right, ARKIT_UP, fwd])
    return origin, B


def to_body(p, origin, B):
    """World point -> operator body coords [right, up, forward]."""
    return B.T @ (np.asarray(p, float) - origin)


def _wrist_world(hand_mats):
    """Wrist world position from a 9-element hand transform list ([0] = wrist)."""
    if hand_mats is None or len(hand_mats) == 0:
        return None
    return np.asarray(hand_mats[0], float)[:3, 3]


# ── the profile ───────────────────────────────────────────────────────────────

@dataclass
class OperatorProfile:
    """Measured per-operator anthropometry. All lengths in metres; body-frame
    offsets are ``[right, up, forward]`` relative to the live head pose."""
    user: str = "default"

    # Body
    height_m: float = 0.0               # primary stature (arm-derived, frame-free)
    height_hmd_m: float = 0.0           # secondary stature (HMD eye height; needs floor)
    arm_length_m: float = 0.0           # mean shoulder->wrist length
    arm_length_lr: tuple = (0.0, 0.0)   # (left, right) shoulder->wrist
    upper_arm_m: float = 0.0
    forearm_m: float = 0.0
    shoulder_width_m: float = 0.0       # biacromial
    neck_height_m: float = 0.0          # head (eye) above the shoulder line
    reach_m: float = 0.0                # max measured wrist reach radius from shoulder
    # Shoulder centres as fixed offsets in the operator body frame (rebuilt live
    # from the current head pose by the retargeter for the shoulder anchor).
    shoulder_offset_body: dict = field(default_factory=lambda: {"l": [0, 0, 0],
                                                                 "r": [0, 0, 0]})

    # Hands (per side: "left" / "right")
    finger_len_norm: dict = field(default_factory=dict)   # canonical-normalised bone len
    finger_len_m: dict = field(default_factory=dict)      # metric per-finger len
    palm_len_m: dict = field(default_factory=dict)        # wrist->mid-knuckle (norm scale)
    pinch_open_m: dict = field(default_factory=dict)      # thumb-index gap, open
    pinch_close_m: dict = field(default_factory=dict)     # thumb-index gap, pinched

    # Provenance
    created: float = 0.0
    quality: dict = field(default_factory=dict)           # per-pose sample counts / flags

    # ── derived robot-coupling helpers (need the robot side) ──────────────────
    def gamma(self, robot_arm_reach):
        """Per-user workspace scale gamma = robot_arm_reach / human_arm_reach,
        clamped to the safe band. Falls back to 1.0 if the arm length is unmeasured."""
        if self.arm_length_m <= 1e-3:
            return 1.0
        g = float(robot_arm_reach) / self.arm_length_m
        return float(np.clip(g, C.RETARGET_GAMMA_MIN, C.RETARGET_GAMMA_MAX))

    def shoulder_world(self, side, head):
        """World shoulder centre for ``side`` ('l'/'r') at the live head pose."""
        origin, B = body_frame(head)
        off = np.asarray(self.shoulder_offset_body.get(side, [0, 0, 0]), float)
        return origin + B @ off

    # ── persistence ───────────────────────────────────────────────────────────
    def to_json(self):
        d = asdict(self)
        d["arm_length_lr"] = list(self.arm_length_lr)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        d["arm_length_lr"] = tuple(d.get("arm_length_lr", (0.0, 0.0)))
        return cls(**d)

    def save(self, directory=None):
        directory = directory or C.CALIB_PROFILE_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.user}.json")
        with open(path, "w") as fh:
            fh.write(self.to_json())
        return path

    @classmethod
    def load(cls, user, directory=None):
        directory = directory or C.CALIB_PROFILE_DIR
        path = os.path.join(directory, f"{user}.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return cls.from_json(fh.read())


# ── pure measurement functions (frames -> raw measurements) ───────────────────

def _avg_wrist_body(poses, side):
    """Mean wrist body coords over a window of poses for one side, or None."""
    rows = []
    for p in poses:
        head = getattr(p, "head", None)
        mats = p.right if side == "r" else p.left
        w = _wrist_world(mats)
        if head is None or w is None:
            continue
        origin, B = body_frame(head)
        rows.append(to_body(w, origin, B))
    if not rows:
        return None
    return np.mean(rows, axis=0)


def measure_tpose(poses):
    """T-pose window -> per-side mean wrist body coords + arm span.

    Returns ``{'l': [r,u,f], 'r': [...], 'arm_span': m, 'n': k}`` (keys present only
    for sides that were tracked)."""
    out = {}
    for side in ("l", "r"):
        b = _avg_wrist_body(poses, side)
        if b is not None:
            out[side] = b
    # Arm span from the mean world wrist separation (frame-independent).
    spans = []
    for p in poses:
        wl, wr = _wrist_world(p.left), _wrist_world(p.right)
        if wl is not None and wr is not None:
            spans.append(float(np.linalg.norm(wl - wr)))
    if spans:
        out["arm_span"] = float(np.mean(spans))
    out["n"] = len(poses)
    return out


def measure_arms_forward(poses):
    """Arms-forward window -> shoulder width + per-side wrist body coords + reach.

    The lateral (body-right) separation of the two forward-pointing wrists is the
    biacromial shoulder width; the forward coord is the reach proxy."""
    out = {}
    seps = []
    for p in poses:
        head = getattr(p, "head", None)
        wl, wr = _wrist_world(p.left), _wrist_world(p.right)
        if head is None or wl is None or wr is None:
            continue
        origin, B = body_frame(head)
        bl, br = to_body(wl, origin, B), to_body(wr, origin, B)
        seps.append(abs(bl[0] - br[0]))          # |right_L - right_R|
    if seps:
        out["shoulder_width"] = float(np.mean(seps))
    for side in ("l", "r"):
        b = _avg_wrist_body(poses, side)
        if b is not None:
            out[side] = b
    out["n"] = len(poses)
    return out


def _palm_and_fingers(joints):
    """One hand's named skeleton -> (palm_len_m, {finger: metric_len},
    {finger: normalised_len}), or None if too sparse."""
    def P(name):
        T = joints.get(name)
        return None if T is None else np.asarray(T, float)[:3, 3]

    wrist = P("wrist")
    i_kn = P("index_knuckle")
    p_kn = P("little_knuckle")
    if wrist is None or i_kn is None or p_kn is None:
        return None
    palm_len = float(np.linalg.norm((i_kn + p_kn) / 2.0 - wrist))
    if palm_len < 1e-4:
        return None

    metric = {}
    for f, chain in _FINGER_CHAIN.items():
        pts = [P(n) for n in chain]
        segs = [np.linalg.norm(pts[k + 1] - pts[k])
                for k in range(len(pts) - 1)
                if pts[k] is not None and pts[k + 1] is not None]
        if segs:
            metric[f] = float(np.sum(segs))

    # Canonical-normalised bone length (kn->pip + pip->tip), the quantity the
    # keypoint-IK calibrator consumes -- reuse the exact same construction.
    norm = {}
    kp = human_keypoints(joints, None)
    if kp is not None:
        for f in _ALL_FINGERS:
            kn, pip, tip = kp.get((f, "kn")), kp.get((f, "pip")), kp.get((f, "tip"))
            if kn is not None and pip is not None and tip is not None:
                norm[f] = float(np.linalg.norm(pip - kn) + np.linalg.norm(tip - pip))
    return palm_len, metric, norm


def measure_hands(poses):
    """Open-hand window -> per-side palm length, finger lengths (metric + norm),
    and pinch open/close span. Averaged over the window to denoise."""
    out = {"left": {}, "right": {}}
    for side in ("left", "right"):
        palm, metric, norm = [], {}, {}
        pinch = []
        jkey = f"{side}_joints"
        pkey = f"{side}_pinch"
        for p in poses:
            joints = getattr(p, jkey, None)
            if joints:
                res = _palm_and_fingers(joints)
                if res is not None:
                    pl, mt, nm = res
                    palm.append(pl)
                    for f, v in mt.items():
                        metric.setdefault(f, []).append(v)
                    for f, v in nm.items():
                        norm.setdefault(f, []).append(v)
            g = getattr(p, pkey, None)
            if g is not None:
                pinch.append(float(g))
        d = out[side]
        if palm:
            d["palm_len_m"] = float(np.mean(palm))
        if metric:
            d["finger_len_m"] = {f: float(np.mean(v)) for f, v in metric.items()}
        if norm:
            d["finger_len_norm"] = {f: float(np.mean(v)) for f, v in norm.items()}
        if pinch:
            d["pinch_open_m"] = float(np.max(pinch))
            d["pinch_close_m"] = float(np.min(pinch))
    return out


def build_profile(tpose, forward, hands, user="default", created=None):
    """Combine the three pose measurements into an OperatorProfile.

    tpose/forward: dicts from measure_tpose / measure_arms_forward.
    hands: dict from measure_hands. Missing measurements fall back to the Winter
    anthropometric ratios (and are flagged in ``quality``)."""
    prof = OperatorProfile(user=user, created=created if created is not None else 0.0)
    q = prof.quality

    # Shoulder width: measured (arms-forward) preferred; else ratio of a provisional
    # stature derived from the raw arm span (span ~= stature).
    w_sh = forward.get("shoulder_width")
    if w_sh is None and "arm_span" in tpose:
        w_sh = C.ANTHRO_BIACROMIAL * tpose["arm_span"]
        q["shoulder_width"] = "fallback_ratio"
    w_sh = float(w_sh or 0.0)
    prof.shoulder_width_m = w_sh

    # Per-side arm length = lateral distance from the shoulder (+-w_sh/2) to the
    # T-pose wrist (arm is purely lateral, so the lateral coord carries the length).
    arms = {}
    for side in ("l", "r"):
        b = tpose.get(side)
        if b is None:
            continue
        L = abs(float(b[0])) - w_sh / 2.0
        if L > 0.05:
            arms[side] = L
    if not arms and "arm_span" in tpose:
        L = max(0.0, (tpose["arm_span"] - w_sh) / 2.0)
        arms = {"l": L, "r": L}
        q["arm_length"] = "from_span"
    L_l = arms.get("l", arms.get("r", 0.0))
    L_r = arms.get("r", arms.get("l", 0.0))
    prof.arm_length_lr = (float(L_l), float(L_r))
    prof.arm_length_m = float(np.mean([L_l, L_r])) if arms else 0.0
    prof.upper_arm_m = C.ANTHRO_UPPER_ARM_FRAC * prof.arm_length_m
    prof.forearm_m = C.ANTHRO_FOREARM_FRAC * prof.arm_length_m

    # Shoulder centres in the body frame. In a T-pose the shoulder shares the
    # wrist's forward coord and (averaged with arms-forward) the wrist's height; only
    # the lateral coord is pulled in to +-w_sh/2.
    for side, sign in (("l", -1.0), ("r", +1.0)):
        bt = tpose.get(side)
        if bt is None:
            continue
        up_samples = [bt[1]]
        bf = forward.get(side)
        if bf is not None:
            up_samples.append(bf[1])
        prof.shoulder_offset_body[side] = [sign * w_sh / 2.0,
                                           float(np.mean(up_samples)),
                                           float(bt[2])]

    # Neck height: head (eye) above the shoulder line = -(mean shoulder up coord).
    ups = [prof.shoulder_offset_body[s][1] for s in ("l", "r") if s in tpose]
    if ups:
        prof.neck_height_m = float(-np.mean(ups))

    # Reach: max measured wrist distance from its shoulder over the forward window;
    # defaults to the arm length.
    reach = prof.arm_length_m
    for side in ("l", "r"):
        bf = forward.get(side)
        so = prof.shoulder_offset_body.get(side)
        if bf is not None and so is not None:
            reach = max(reach, float(np.linalg.norm(np.asarray(bf) - np.asarray(so))))
    prof.reach_m = reach

    # Stature: primary from the (frame-free) arm length via the Winter ratio;
    # secondary from the HMD eye height (needs a floor-referenced origin).
    if prof.arm_length_m > 1e-3:
        prof.height_m = prof.arm_length_m / C.ANTHRO_ARM_RATIO
    eye = forward.get("eye_height", tpose.get("eye_height"))
    if eye:
        prof.height_hmd_m = float(eye) / C.ANTHRO_EYE_HEIGHT

    # Hands.
    for side in ("left", "right"):
        h = hands.get(side, {})
        if "palm_len_m" in h:
            prof.palm_len_m[side] = h["palm_len_m"]
        if "finger_len_m" in h:
            prof.finger_len_m[side] = h["finger_len_m"]
        if "finger_len_norm" in h:
            prof.finger_len_norm[side] = h["finger_len_norm"]
        if "pinch_open_m" in h:
            prof.pinch_open_m[side] = h["pinch_open_m"]
        if "pinch_close_m" in h:
            prof.pinch_close_m[side] = h["pinch_close_m"]

    q["n_tpose"] = tpose.get("n", 0)
    q["n_forward"] = forward.get("n", 0)
    return prof


# ── streaming hold-detection state machine ────────────────────────────────────

def _key_points(pose):
    """Flat array of world points used for the still-hold test: the head + both
    wrists. Fingertips are deliberately EXCLUDED so the hands pose (open hand ->
    fist -> pinch) is captured while the WRISTS hold still -- the finger bone lengths
    are sums of rigid phalanx segments, hence curl-invariant, so averaging across the
    open/closed sweep is correct and also yields the pinch open/close span."""
    pts = []
    head = getattr(pose, "head", None)
    if head is not None:
        pts.append(np.asarray(head, float)[:3, 3])
    for mats in (pose.left, pose.right):
        w = _wrist_world(mats)
        if w is not None:
            pts.append(w)
    return np.array(pts) if pts else None


class BodyCalibrator:
    """Drives the 3-pose capture from a stream of OperatorPose frames.

    Usage (in the facade / controller)::

        cal = BodyCalibrator(user="operator")
        while cal.active:
            cal.observe(pose, now)         # one call per frame
        profile = cal.profile              # populated when cal.done

    Each pose auto-captures once the tracked points hold still
    (``CALIB_HOLD_SPEED`` for ``CALIB_HOLD_TIME``), then averages a trailing window
    (``CALIB_AVG_WINDOW``). Stateless math is delegated to the measure_* functions so
    the same logic is exercised by the offline tests."""

    def __init__(self, user="default"):
        self.user = user
        self.pose_idx = 0
        self.done = False
        self.profile = None
        self._buf = []                  # (t, pose, keypoints) for the current pose
        self._still_since = None
        self._need_motion = False       # re-arm: require a NEW distinct pose between captures
        self._last_captured_kp = None   # mean key-points of the last captured pose
        self._measurements = {}

    @property
    def active(self):
        return not self.done

    @property
    def current_pose(self):
        if self.pose_idx >= len(POSE_SEQUENCE):
            return None
        return POSE_SEQUENCE[self.pose_idx]

    def reset(self):
        self.__init__(self.user)

    def observe(self, pose, now=None):
        """Feed one frame. Returns the pose name just captured (or None)."""
        if self.done or pose is None or getattr(pose, "head", None) is None:
            return None
        now = time.monotonic() if now is None else now
        kp = _key_points(pose)
        if kp is None:
            return None
        self._buf.append((now, pose, kp))
        # Trim to a little over the averaging window.
        cutoff = now - max(C.CALIB_AVG_WINDOW, C.CALIB_HOLD_TIME) - 0.2
        self._buf = [s for s in self._buf if s[0] >= cutoff]

        still = self._is_still(now)        # True (still) | False (moving) | None (unknown)
        # Re-arm gate: after a capture, the next capture only fires once a NEW, distinct
        # still pose is held -- judged by comparing the current still configuration to
        # the just-captured one, NOT by observing the motion transient. This is robust
        # to both continuous streams and discrete frame jumps (where the transient is
        # never seen), and still stops leftover frames of the captured pose from being
        # re-captured (they are not novel).
        if self._need_motion:
            if still is True and self._pose_is_novel(now):
                self._need_motion = False
                self._still_since = now         # time the new pose's hold from here
            return None
        if still is not True:
            self._still_since = None
            return None
        if self._still_since is None:
            self._still_since = now
        if now - self._still_since < C.CALIB_HOLD_TIME:
            return None
        return self._capture(now)

    def _pose_is_novel(self, now):
        """True if the current still configuration differs from the last captured pose
        by more than CALIB_POSE_NOVELTY (or the tracked-point set changed)."""
        if self._last_captured_kp is None:
            return True
        recent = [k for (t, _, k) in self._buf
                  if t >= now - C.CALIB_STILL_WINDOW
                  and k.shape == self._last_captured_kp.shape]
        if not recent:
            return True                          # shape changed -> a different pose
        cur = np.mean(recent, axis=0)
        return float(np.max(np.linalg.norm(cur - self._last_captured_kp, axis=1))) \
            > C.CALIB_POSE_NOVELTY

    def _is_still(self, now):
        """Positional SPREAD of the tracked points over the recent window vs the hold
        threshold. Spread (max deviation from the window mean) is jitter-tolerant where
        frame-to-frame speed is not. Returns True (holding still), False (moving), or
        None (not enough / inconsistent data yet)."""
        recent = [s for s in self._buf if s[0] >= now - C.CALIB_STILL_WINDOW]
        if len(recent) < 3:
            return None
        shapes = {k.shape for _, _, k in recent}
        if len(shapes) != 1:                       # a hand appeared/dropped mid-window
            return None
        K = np.stack([k for _, _, k in recent])    # (T, P, 3)
        dev = float(np.max(np.linalg.norm(K - K.mean(axis=0), axis=2)))
        return bool(dev < C.CALIB_HOLD_JITTER)

    def _capture(self, now):
        # Average only the still-hold frames (since _still_since), bounded by the window,
        # so transition frames before the hold never contaminate the measurement.
        lo = now - C.CALIB_AVG_WINDOW
        if self._still_since is not None:
            lo = max(lo, self._still_since)
        rows = [(p, k) for (t, p, k) in self._buf if t >= lo]
        window = [p for p, _ in rows]
        if len(window) < C.CALIB_MIN_SAMPLES:
            return None
        # Remember this pose's mean key-points so the re-arm gate can tell the next
        # (distinct) pose apart from leftover frames of this one.
        kps = [k for _, k in rows]
        if kps and all(k.shape == kps[0].shape for k in kps):
            self._last_captured_kp = np.mean(kps, axis=0)
        name = self.current_pose
        if name == POSE_TPOSE:
            self._measurements["tpose"] = measure_tpose(window)
        elif name == POSE_FORWARD:
            self._measurements["forward"] = measure_arms_forward(window)
        elif name == POSE_HANDS:
            self._measurements["hands"] = measure_hands(window)
        log.info("calibration pose '%s' captured (%d samples)", name, len(window))
        self.pose_idx += 1
        self._buf = []
        self._still_since = None
        self._need_motion = True              # require motion before the next pose
        if self.pose_idx >= len(POSE_SEQUENCE):
            self._finish(now)
        return name

    def _finish(self, now):
        self.profile = build_profile(
            self._measurements.get("tpose", {}),
            self._measurements.get("forward", {}),
            self._measurements.get("hands", {}),
            user=self.user, created=now)
        self.done = True
        log.info("calibration complete for '%s': arm=%.3f m, shoulder=%.3f m, H~%.2f m",
                 self.user, self.profile.arm_length_m,
                 self.profile.shoulder_width_m, self.profile.height_m)


# ── forward model: synthesise calibration frames for a known operator ─────────
# Used by the unit tests and the experiments harness to validate the inverse
# problem against ground truth. Kept next to the inverse so the two stay in sync.

def _pose(head_pos, head_R, wl, wr, l_joints=None, r_joints=None,
          l_pinch=None, r_pinch=None):
    from ..types import OperatorPose
    def T(pos):
        m = np.eye(4)
        m[:3, 3] = pos
        return m
    head = np.eye(4)
    head[:3, :3] = head_R
    head[:3, 3] = head_pos
    return OperatorPose(
        head=head,
        left_hand=([T(wl)] + [T(wl)] * 8) if wl is not None else None,
        right_hand=([T(wr)] + [T(wr)] * 8) if wr is not None else None,
        left_joints=l_joints, right_joints=r_joints,
        left_pinch=l_pinch, right_pinch=r_pinch)


def synthesize_pose(pose_name, *, height=1.75, arm_length=None, shoulder_width=None,
                    neck_height=None, head_yaw=0.0, head_pos=(0.0, 1.6, 0.0),
                    hand=None, n=12, noise=0.0, seed=0):
    """Generate a list of ``n`` OperatorPose frames for ``pose_name`` for an operator
    of the given dimensions, in the ARKit world frame (+Y up, -Z forward).

    Defaults derive unmeasured dimensions from ``height`` via the Winter ratios, so a
    single ``height`` yields a self-consistent operator. ``noise`` (m, std) adds
    isotropic Gaussian jitter to every point. ``hand`` (a dict of metric finger
    lengths + palm length) drives the synthetic skeleton for the hands pose."""
    rng = np.random.default_rng(seed)
    L = C.ANTHRO_ARM_RATIO * height if arm_length is None else arm_length
    w_sh = C.ANTHRO_BIACROMIAL * height if shoulder_width is None else shoulder_width
    h_neck = C.ANTHRO_ACROMION_BELOW_EYE * height if neck_height is None else neck_height
    head_pos = np.asarray(head_pos, float)

    cy, sy = np.cos(head_yaw), np.sin(head_yaw)
    head_R = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    _, B = body_frame(np.block([[head_R, head_pos[:, None]], [0, 0, 0, 1]]))
    right, up, fwd = B[:, 0], B[:, 1], B[:, 2]
    # Shoulder world centres (body offset [+-w_sh/2, -h_neck, 0]).
    sh_l = head_pos + B @ np.array([-w_sh / 2.0, -h_neck, 0.0])
    sh_r = head_pos + B @ np.array([+w_sh / 2.0, -h_neck, 0.0])

    def jit(p):
        return p + (rng.normal(0, noise, 3) if noise > 0 else 0.0)

    out = []
    p_open = (hand or {}).get("pinch_open", 0.09)
    p_close = (hand or {}).get("pinch_close", 0.02)
    for k in range(n):
        if pose_name == POSE_TPOSE:                 # wrists straight out to the sides
            wl = sh_l - right * L
            wr = sh_r + right * L
            out.append(_pose(jit(head_pos), head_R, jit(wl), jit(wr)))
        elif pose_name == POSE_FORWARD:             # wrists straight ahead, parallel
            wl = sh_l + fwd * L
            wr = sh_r + fwd * L
            out.append(_pose(jit(head_pos), head_R, jit(wl), jit(wr)))
        elif pose_name == POSE_HANDS:               # hands raised in front, open->pinch
            # Natural "present/inspect hands" pose: wrists raised in front of the face
            # (distinct from the arms-forward pose, so the operator visibly moves into
            # it). Finger measurements are all wrist-relative, so the exact wrist
            # placement does not affect the recovered bone lengths.
            wl_h = head_pos + fwd * 0.30 + up * (-0.10) + right * (-0.10)
            wr_h = head_pos + fwd * 0.30 + up * (-0.10) + right * (+0.10)
            lj = synthesize_hand_skeleton(wl_h, B, hand, "left", rng, noise)
            rj = synthesize_hand_skeleton(wr_h, B, hand, "right", rng, noise)
            # Pinch scalar ramps open->close monotonically across the window so both
            # extremes appear exactly (the wrists stay put, so the hold detector fires).
            pinch = p_open + (p_close - p_open) * (k / max(n - 1, 1))
            out.append(_pose(jit(head_pos), head_R, jit(wl_h), jit(wr_h),
                             l_joints=lj, r_joints=rj, l_pinch=pinch, r_pinch=pinch))
    return out


def synthesize_hand_skeleton(wrist, B, hand, side, rng, noise=0.0):
    """A flat open-hand ARKit skeleton at ``wrist`` for the given metric finger/palm
    lengths, laid out in the operator body frame ``B`` (fingers along +forward, splayed
    along +-right). Returns ``{wire_name: 4x4 world}``."""
    hand = hand or {}
    palm = hand.get("palm_len", 0.09)
    flen = hand.get("fingers", {"thumb": 0.06, "index": 0.08, "middle": 0.09,
                                "ring": 0.085, "pinky": 0.07})
    right, up, fwd = B[:, 0], B[:, 1], B[:, 2]
    def T(pos):
        m = np.eye(4)
        m[:3, 3] = pos + (rng.normal(0, noise, 3) if noise > 0 else 0.0)
        return m
    out = {"wrist": T(wrist)}
    # Knuckle row at the palm, splayed laterally; finger bones along +forward.
    splay = {"thumb": -0.045, "index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        w = _WIRE_FINGER.get(f, f)
        kn = wrist + fwd * palm + right * splay[f]
        L = flen.get(f, 0.08)
        chain = _FINGER_CHAIN[f]
        pts = [kn, kn + fwd * 0.45 * L, kn + fwd * 0.75 * L, kn + fwd * L]
        for name, p in zip(chain, pts):
            out[name] = T(p)
    return out
