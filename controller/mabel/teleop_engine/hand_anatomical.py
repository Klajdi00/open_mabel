"""Webcam-parity anatomical hand retargeting (the control-center GUI's algorithm).

This is a faithful Python port of the browser-side webcam hand retargeting the
control-center GUI runs (control_center/web/js/bodyteleop.js `palmFrame` /
`decomposeHand` + dexterity.js `retargetLandmarks` + handmodel.js calibrated
ranges), so the Vision Pro / iOS / web app path drives the ORCA fingers with the
SAME mapping as the webcam teleop:

  * palm frame [x̂ ŷ ẑ]: ŷ = wrist→middle knuckle, ẑ = dorsal palm normal
    (chirality-signed so fingers always flex toward −ẑ), x̂ = ŷ×ẑ
  * per finger: abduction from the palm-plane projection of the proximal phalanx
    (faded near MCP≈90°), MCP/PIP/DIP flexion = signed angles between successive
    bone segments about the finger's flexion axis
  * thumb: metacarpal DIRECTION kept as measured (2-DOF opposition), MP/IP
    flexion about the thumb-plane normal; opposition = asin(−dir·ẑ)
  * ORCA mapping: PIP tendon-couples the anatomical PIP+DIP pair (pip + ½·max(0,dip)),
    abduction pinned to mid-range, thumb URDF remap (opposition→thumb_mcp,
    MP→thumb_dip, IP→thumb_pip), each angle clamped into the calibrated joint range.

Differences from the JS (deliberate, documented):
  * input is the TRUE ARKit skeleton (the 21-landmark world-transform list the
    engine builds), so the MediaPipe camera-coordinate negation is not applied —
    the math is coordinate-free and consumes hand-centered points directly;
  * a degenerate palm HOLDS the last solved pose instead of snapping to the JS
    "safe neutral" (open hand) — dropouts on the headset are transient;
  * a light EMA (`HAND_ANAT_SMOOTH`, same 0.45 the shipped HandRetargeter uses)
    filters ARKit jitter per joint.
"""
import time

import numpy as np

from .hyperparams import (HAND_ANAT_SMOOTH, HAND_ONEEURO, HAND_ONEEURO_MIN_CUTOFF,
                          HAND_ONEEURO_BETA, HAND_ONEEURO_DCUTOFF, HAND_TRACK_ABDUCTION,
                          HAND_ABD_GAIN, HAND_THUMB_FLEX_OFFSET, HAND_THUMB_FLEX_GAIN)

# MediaPipe finger base indices (knuckle); chain is base, +1 PIP, +2 DIP, +3 tip.
_FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "little": 17}
# Anatomical clamp ranges [rad] for the decomposed angles (bodyteleop.FLEX_LIMITS).
_FLEX = {"mcp": (-0.35, 1.90), "pip": (-0.30, 2.05), "dip": (-0.30, 1.65),
         "abd": (-0.50, 0.50), "tmp": (-0.45, 1.30), "tip": (-0.45, 1.70)}

# Calibrated ORCA joint ranges [rad] (handmodel.HAND_JOINTS; URDF wire names —
# thumb_mcp = ORCA CMC, thumb_dip = ORCA MP, thumb_pip = ORCA IP). The webcam GUI
# maps rad→percent→rad through these, i.e. a clamp into the calibrated range.
HAND_RANGES = {
    "thumb_mcp":  (-0.8727, 0.8727),
    "thumb_abd":  (-0.3491, 0.7330),
    "thumb_pip":  (-0.3491, 1.9548),
    "thumb_dip":  (-0.2094, 1.8150),
    "index_abd":  (-0.6458, 0.6458),
    "index_mcp":  (-0.3491, 1.6581),
    "index_pip":  (-0.3491, 1.8850),
    "middle_abd": (-0.6458, 0.6458),
    "middle_mcp": (-0.3491, 1.5882),
    "middle_pip": (-0.3491, 1.8675),
    "ring_abd":   (-0.6458, 0.6358),
    "ring_mcp":   (-0.3491, 1.5882),
    "ring_pip":   (-0.3491, 1.8675),
    "pinky_abd":  (-0.6458, 0.6458),
    "pinky_mcp":  (-0.3491, 1.7104),
    "pinky_pip":  (-0.3491, 1.8850),
}


def _unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-12 else v / n


def _signed_angle(u, v, n):
    """Signed angle from u to v about unit axis n̂ (u, v need not be unit)."""
    return float(np.arctan2(np.dot(np.cross(u, v), n), np.dot(u, v)))


def _rot_axis(v, a, th):
    """Rodrigues rotation of v about unit axis â by θ."""
    c, s = np.cos(th), np.sin(th)
    return v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1.0 - c)


def _clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


def palm_frame(shape, chirality):
    """Palm frame (columns [x̂, ŷ, ẑ]) from a hand-centered 21-point shape.
    ŷ = wrist→middle knuckle, ẑ = DORSAL palm normal (χ makes the same formula
    chirality-correct: fingers always flex toward −ẑ).

    χ SIGN vs the JS: the webcam pipeline labels handedness on the MediaPipe
    (selfie-mirrored) convention, so dexterity.js uses χ=+1 for its 'right'. TRUE
    ARKit skeletons carry real chirality — on them (index−w)×(little−w) points to
    the PALM side of a right hand — so the sign is flipped here (verified on a
    synthetic anatomical hand: thumb medial, curl toward the palm)."""
    w = shape[0]
    chi = 1.0 if chirality == "left" else -1.0
    y = _unit(shape[9] - w)
    zr = chi * np.cross(shape[5] - w, shape[17] - w)
    if y is None or np.linalg.norm(zr) < 1e-12:
        return None
    z = _unit(zr - y * np.dot(zr, y))
    if z is None:
        return None
    return np.column_stack([np.cross(y, z), y, z])


def decompose_hand(shape, chirality):
    """Decompose a hand-centered 21-point shape into anatomical angles.
    Returns {"fingers": {f: {abd, mcp, pip, dip}}, "thumb": {dir, mp, ip}} [rad],
    or None when the palm is degenerate. Port of bodyteleop.decomposeHand."""
    B = palm_frame(shape, chirality)
    if B is None:
        return None
    w, z = shape[0], B[:, 2]
    fingers = {}
    for f, base in _FINGER_BASE.items():
        kn = shape[base] - w
        d1 = shape[base + 1] - shape[base]
        d2 = shape[base + 2] - shape[base + 1]
        d3 = shape[base + 3] - shape[base + 2]
        m_pl = _unit(kn - z * np.dot(kn, z))
        if m_pl is None:
            return None
        # Abduction from the palm-plane projection of the proximal phalanx, faded
        # with the projection magnitude (smooth, singular-free near MCP ≈ 90°).
        d1pl = d1 - z * np.dot(d1, z)
        w_abd = _clamp(np.linalg.norm(d1pl) / (0.35 * (np.linalg.norm(d1) + 1e-9)), 0.0, 1.0)
        d1pl_u = _unit(d1pl)
        abd = w_abd * _signed_angle(m_pl, d1pl_u, z) if d1pl_u is not None else 0.0
        # Flexion axis â ⊥ the finger's sagittal plane; flexion > 0 curls to −ẑ.
        r = _rot_axis(m_pl, z, abd)
        a = _unit(np.cross(z, r))
        if a is None:
            a = np.cross(z, m_pl)
        u1, u2, u3 = _unit(d1), _unit(d2), _unit(d3)
        if u1 is None or u2 is None or u3 is None:
            return None
        fingers[f] = {
            "abd": _clamp(abd, *_FLEX["abd"]),
            "mcp": _clamp(_signed_angle(r, u1, a), *_FLEX["mcp"]),
            "pip": _clamp(_signed_angle(u1, u2, a), *_FLEX["pip"]),
            "dip": _clamp(_signed_angle(u2, u3, a), *_FLEX["dip"]),
        }
    # Thumb: metacarpal DIRECTION kept as measured (2-DOF opposition), then MP/IP
    # flexion about the thumb-plane normal.
    t = shape[2] - shape[1]
    p_mp = shape[3] - shape[2]
    p_ip = shape[4] - shape[3]
    t_u, mp_u, ip_u = _unit(t), _unit(p_mp), _unit(p_ip)
    if t_u is None or mp_u is None or ip_u is None:
        return None
    axis = np.cross(t_u, mp_u)
    if np.linalg.norm(axis) < 0.15:
        axis = np.cross(t_u, ip_u)
    if np.linalg.norm(axis) < 0.15:
        axis = np.cross(z, t_u)                    # straight thumb: angles ≈ 0
    a_t = _unit(axis)
    if a_t is None:
        return None
    thumb = {
        "dir": B.T @ t_u,                          # metacarpal direction, palm coords
        "mp": _clamp(_signed_angle(t_u, mp_u, a_t), *_FLEX["tmp"]),
        "ip": _clamp(_signed_angle(mp_u, ip_u, a_t), *_FLEX["tip"]),
    }
    return {"fingers": fingers, "thumb": thumb}


def retarget_hand(pts21, chirality):
    """21 hand landmark POSITIONS (ARKit world, MediaPipe layout) -> {suffix: rad}
    ORCA joint targets — the dexterity.retargetLandmarks mapping. Returns None on a
    degenerate palm (caller holds the last pose)."""
    pts = np.asarray(pts21, float)
    shape = pts - pts[0]
    dec = decompose_hand(shape, chirality)
    if dec is None:
        return None
    pose = {}
    finger_key = {"index": "index", "middle": "middle", "ring": "ring", "little": "pinky"}
    for f, k in finger_key.items():
        A = dec["fingers"][f]
        pose[f"{k}_mcp"] = _clamp(A["mcp"], *HAND_RANGES[f"{k}_mcp"])
        # ORCA PIP tendon-couples the anatomical PIP+DIP pair: take the dominant
        # PIP and credit half the DIP so a distal-only curl still registers.
        pose[f"{k}_pip"] = _clamp(A["pip"] + 0.5 * max(0.0, A["dip"]), *HAND_RANGES[f"{k}_pip"])
        # SPLAY: track the MEASURED abduction (was hard-pinned to neutral, i.e. untracked).
        # The one-euro filter smooths it downstream like every other joint. Set
        # MABEL_HAND_ABD_GAIN=-1 if a spread maps the wrong way; =0 restores neutral splay.
        pose[f"{k}_abd"] = (_clamp(HAND_ABD_GAIN * A["abd"], *HAND_RANGES[f"{k}_abd"])
                            if HAND_TRACK_ABDUCTION else 0.5 * sum(HAND_RANGES[f"{k}_abd"]))
    # Thumb URDF naming remap: thumb_mcp = ORCA CMC (opposition), thumb_dip = ORCA MP,
    # thumb_pip = ORCA IP. Opposition = the metacarpal's dive below the palm plane
    # (−ẑ dorsal): the palm-frame z component of the unit direction.
    T = dec["thumb"]
    opp = float(np.arcsin(_clamp(-T["dir"][2], -1.0, 1.0)))
    pose["thumb_mcp"] = _clamp(opp, *HAND_RANGES["thumb_mcp"])
    # Subtract the resting-thumb bend before mapping flexion so a RELAXED thumb reads as
    # OPEN, not already-closing (the "thumb closes too early" report). Tunable per operator
    # via MABEL_HAND_THUMB_OFFSET (rad) / MABEL_HAND_THUMB_GAIN.
    pose["thumb_dip"] = _clamp(HAND_THUMB_FLEX_GAIN * (T["mp"] - HAND_THUMB_FLEX_OFFSET),
                               *HAND_RANGES["thumb_dip"])
    pose["thumb_pip"] = _clamp(HAND_THUMB_FLEX_GAIN * (T["ip"] - HAND_THUMB_FLEX_OFFSET),
                               *HAND_RANGES["thumb_pip"])
    pose["thumb_abd"] = 0.5 * sum(HAND_RANGES["thumb_abd"])
    return pose


class AnatomicalHandRetargeter:
    """Drop-in finger solver for the TeleopEngine: consumes the engine's 21-mat
    hand list, runs the webcam anatomical decomposition, and returns joint targets
    aligned with the SAME `joint_names` order HandOptimizer used (so the engine's
    finger actuator write is unchanged). EMA-smoothed; holds the last pose on a
    degenerate frame."""

    # suffix -> MuJoCo joint/actuator name fragment ("<Side> <fragment>")
    _SUFFIX_TO_NAME = {
        "thumb_mcp": "Thumb MCP", "thumb_abd": "Thumb Abd",
        "thumb_pip": "Thumb PIP", "thumb_dip": "Thumb DIP",
        "index_abd": "Index Abd", "index_mcp": "Index MCP", "index_pip": "Index PIP",
        "middle_abd": "Middle Abd", "middle_mcp": "Middle MCP", "middle_pip": "Middle PIP",
        "ring_abd": "Ring Abd", "ring_mcp": "Ring MCP", "ring_pip": "Ring PIP",
        "pinky_abd": "Pinky Abd", "pinky_mcp": "Pinky MCP", "pinky_pip": "Pinky PIP",
    }

    def __init__(self, model, side, smooth=HAND_ANAT_SMOOTH):
        self.side = side
        self.smooth = float(smooth)
        S = side.capitalize()
        # Same order as HandOptimizer.joint_names: thumb MCP/Abd/PIP/DIP, then
        # Abd/MCP/PIP per finger.
        suffixes = ["thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip"]
        for f in ("index", "middle", "ring", "pinky"):
            suffixes += [f"{f}_abd", f"{f}_mcp", f"{f}_pip"]
        self.suffixes = suffixes
        self.joint_names = [f"{S} {self._SUFFIX_TO_NAME[s]}" for s in suffixes]
        # Clamp bounds: calibrated webcam range ∩ the model's actuator ctrlrange.
        self.bounds = []
        for s, n in zip(suffixes, self.joint_names):
            lo, hi = HAND_RANGES[s]
            try:
                alo, ahi = model.actuator(n).ctrlrange
                lo, hi = max(lo, float(alo)), min(hi, float(ahi))
            except KeyError:
                pass
            self.bounds.append((lo, hi))
        self.last_q = None
        self._dq = None            # one-euro: filtered per-joint velocity estimate
        self._t_prev = None

    def reset(self):
        self.last_q = None
        self._dq = None
        self._t_prev = None

    def solve(self, h_mats, joints=None):
        """h_mats: 21-element list of flattened/4x4 world transforms (MediaPipe
        layout). ``joints`` (the named skeleton) is accepted for signature parity
        with the keypoint solver and ignored — the decomposition is positional.
        Returns the joint target array aligned with self.joint_names."""
        pts = [np.asarray(m, float).reshape(4, 4)[:3, 3] for m in h_mats]
        pose = retarget_hand(pts, self.side)
        if pose is None:                               # degenerate: hold last
            return self.last_q
        q = np.array([_clamp(pose[s], *b) for s, b in zip(self.suffixes, self.bounds)])
        return self._filter(q)

    def _filter(self, q):
        """Smooth the raw retargeted joints. One-euro (the SOTA for noisy interactive
        tracking): strong smoothing at low joint speed kills the measured ~0.7 rad/frame
        finger-tracking spikes, while the cutoff rises with |q̇| so a fast open/close is
        NOT laggy. Falls back to the fixed EMA when disabled or on the first/stalled
        frame. Per-joint alpha; the arm already carries its own SLSQP smoothness term."""
        now = time.monotonic()
        if self.last_q is None or self._t_prev is None:
            self.last_q = q.copy()
            self._dq = np.zeros_like(q)
            self._t_prev = now
            return self.last_q
        dt = now - self._t_prev
        self._t_prev = now
        if not HAND_ONEEURO or dt <= 0.0 or dt > 0.5:      # legacy EMA / first / gap frame
            a = self.smooth if self.smooth < 1.0 else 1.0
            self.last_q = self.last_q + a * (q - self.last_q)
            return self.last_q
        two_pi = 2.0 * np.pi
        a_d = 1.0 / (1.0 + (1.0 / (two_pi * HAND_ONEEURO_DCUTOFF)) / dt)
        dq_raw = (q - self.last_q) / dt
        self._dq = self._dq + a_d * (dq_raw - self._dq)    # low-pass the velocity
        cutoff = HAND_ONEEURO_MIN_CUTOFF + HAND_ONEEURO_BETA * np.abs(self._dq)
        a = 1.0 / (1.0 + (1.0 / (two_pi * cutoff)) / dt)   # per-joint adaptive alpha
        self.last_q = self.last_q + a * (q - self.last_q)
        return self.last_q
