"""
hand_retarget.py — per-finger Vision Pro → ORCA hand retargeting.

Replaces the pinch→grasp stand-in (hands.apply_grasp) with a full anatomical
joint-angle mapping: each of the operator's fingers drives its ORCA-hand
counterpart, decoupled from the arm/wrist SE(3) retargeting (retarget.py),
which keeps owning the palm pose. Together they form the decoupled scheme:

    wrist/arm : clutch-based relative SE(3) mapping  → WholeBodyIK (8 DOF/arm)
    fingers   : anatomical angle extraction          → 16 ORCA actuators/hand

Angle extraction (frame-invariant — works in any world frame)
--------------------------------------------------------------
ARKit reports the 27 hand-skeleton joints. For each finger we take the world
positions of the 4-point chain (metacarpal → knuckle → intermediate_base →
intermediate_tip → tip) and measure:

  * flexion at a joint = the interior angle between the two adjacent bone
    segments, θ = atan2(|u×v|, u·v)  ∈ [0, π]. atan2 is used instead of
    acos for numerical stability near 0.
  * abduction (finger splay) = the SIGNED angle between the proximal phalanx
    and the finger's own metacarpal direction, both projected into the palm
    plane; the sign comes from the palm normal, so it is chirality-correct.

The palm frame is built from the wrist and the index/little metacarpals:
  x̂ = index_mcp − wrist (forward along the palm),
  n̂ = (little_mcp − wrist) × x̂  (palm normal; negated for the left hand so
                                  the same formula serves both chiralities).

Mapping to ORCA
---------------
Human ranges differ from the ORCA's ctrlrange, so each DOF gets an affine map
angle_orca = GAIN · (θ − OFFSET), clamped to the actuator ctrlrange by the
applier. Flexions map near 1:1 (the ORCA hand is anthropomorphic); thumb
abduction maps into its negative-only range. Per-DOF constants live in
`_FLEX_MAP` / `_ABD_GAIN` below — tune there, nowhere else.

An EMA (`SMOOTH`, default 0.45) filters ARKit jitter per actuator; at the
90 Hz hand-tracking rate this adds ≈ one frame (11 ms) of lag.

Consumers: TeleopController (live teleop) calls `HandRetargeter.apply()`
per tick; `finger_angles()` is pure NumPy (no MuJoCo) for offline tests.
"""
import numpy as np

# ── wire joint names (HandJoint rawValues from the visionOS app) ─────────────
_CHAINS = {
    # finger: (metacarpal, knuckle, intermediate_base, intermediate_tip, tip)
    "index":  ("index_metacarpal", "index_knuckle", "index_intermediate_base",
               "index_intermediate_tip", "index_tip"),
    "middle": ("middle_metacarpal", "middle_knuckle", "middle_intermediate_base",
               "middle_intermediate_tip", "middle_tip"),
    "ring":   ("ring_metacarpal", "ring_knuckle", "ring_intermediate_base",
               "ring_intermediate_tip", "ring_tip"),
    "pinky":  ("little_metacarpal", "little_knuckle", "little_intermediate_base",
               "little_intermediate_tip", "little_tip"),
}
# thumb chain: CMC (knuckle) → MP (intermediate_base) → IP (intermediate_tip) → tip
_THUMB = ("thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip")

# ── tuning constants (single source of truth) ────────────────────────────────
SMOOTH = 0.45            # EMA per actuator; 1 = no smoothing

# flexion DOFs: orca_suffix -> (gain, offset_rad)
# ARKit's straight finger measures ~0 interior angle; ORCA 0 = extended.
_FLEX_MAP = {
    "MCP": (1.00, 0.10),     # small offset: relaxed hands read ~6° at MCP
    "PIP": (1.00, 0.05),
    "DIP": (1.00, 0.05),     # thumb only (finger DIPs are tendon-coupled to PIP)
}
_ABD_GAIN = 1.0              # finger splay gain
_THUMB_ABD_GAIN = 1.0        # thumb opposition gain


def _ang(u, v):
    """Interior angle between two 3-vectors, stable near 0 and π."""
    c = np.cross(u, v)
    return float(np.arctan2(np.linalg.norm(c), float(np.dot(u, v))))


def _signed_ang_in_plane(u, v, n):
    """Signed angle from u to v about plane normal n (all 3-vectors)."""
    un = u - n * np.dot(u, n)
    vn = v - n * np.dot(v, n)
    s = float(np.dot(np.cross(un, vn), n))
    c = float(np.dot(un, vn))
    return float(np.arctan2(s, c))


def finger_angles(joints, chirality="right"):
    """Extract anatomical joint angles from a dict of world transforms.

    joints   : {wire_joint_name: 4x4 row-major np.ndarray (world)}
    chirality: "left" | "right" — fixes the abduction sign convention.

    Returns {orca_suffix: angle_rad} with suffixes like "index_mcp",
    "index_pip", "index_abd", "thumb_mcp", "thumb_abd", "thumb_pip",
    "thumb_dip" — or None if the minimal joints are missing.
    """
    def P(name):
        T = joints.get(name)
        return None if T is None else np.asarray(T, dtype=float)[:3, 3]

    wrist = P("wrist")
    i_mcp = P("index_metacarpal")
    l_mcp = P("little_metacarpal")
    if wrist is None or i_mcp is None or l_mcp is None:
        return None

    # Palm frame: forward + normal (normal flips with chirality, which keeps
    # the signed splay correct for both hands with the SAME formula).
    fwd = i_mcp - wrist
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    # n̂ = (little→) × (index→): positive splay = toward the thumb side on a
    # right hand. Validated against synthetic ground-truth sweeps in
    # docs/tools/exp_hand_retarget.py (E1); the left hand mirrors.
    n = np.cross(l_mcp - wrist, fwd)
    n /= (np.linalg.norm(n) + 1e-9)
    if chirality == "left":
        n = -n

    out = {}

    for finger, (mc, kn, ib, it, tip) in _CHAINS.items():
        p_mc, p_kn, p_ib, p_it = P(mc), P(kn), P(ib), P(it)
        if any(v is None for v in (p_mc, p_kn, p_ib, p_it)):
            continue
        meta = p_kn - p_mc           # metacarpal bone
        prox = p_ib - p_kn           # proximal phalanx
        mid = p_it - p_ib            # middle phalanx
        g, o = _FLEX_MAP["MCP"]
        out[f"{finger}_mcp"] = g * (_ang(meta, prox) - o)
        g, o = _FLEX_MAP["PIP"]
        out[f"{finger}_pip"] = g * (_ang(prox, mid) - o)
        out[f"{finger}_abd"] = _ABD_GAIN * _signed_ang_in_plane(meta, prox, n)

    # Thumb: CMC abduction (opposition) + three flexions down the chain.
    t_cmc, t_mp, t_ip, t_tip = (P(_THUMB[0]), P(_THUMB[1]), P(_THUMB[2]), P(_THUMB[3]))
    if all(v is not None for v in (t_cmc, t_mp, t_ip, t_tip)):
        cmc_seg = t_mp - t_cmc
        prox = t_ip - t_mp
        dist = t_tip - t_ip
        g, o = _FLEX_MAP["MCP"]
        out["thumb_mcp"] = g * (_ang(t_cmc - wrist, cmc_seg) - o)
        g, o = _FLEX_MAP["PIP"]
        out["thumb_pip"] = g * (_ang(cmc_seg, prox) - o)
        g, o = _FLEX_MAP["DIP"]
        out["thumb_dip"] = g * (_ang(prox, dist) - o)
        # Opposition: thumb metacarpal splay off the palm-forward direction.
        # ORCA's thumb_abd ctrlrange is negative-only (0 = against the palm).
        abd = _signed_ang_in_plane(fwd, cmc_seg, n)
        out["thumb_abd"] = -abs(abd) * _THUMB_ABD_GAIN

    return out or None


# ── MuJoCo applier ───────────────────────────────────────────────────────────

_FINGER_NAME = {"thumb": "Thumb", "index": "Index", "middle": "Middle",
                "ring": "Ring", "pinky": "Pinky"}
_SEG_NAME = {"mcp": "MCP", "pip": "PIP", "dip": "DIP", "abd": "Abd"}


class HandRetargeter:
    """Streams per-finger angles onto the ORCA hand actuators, with EMA
    smoothing and ctrlrange clamping. One instance serves both hands."""

    def __init__(self, model, data, smooth=SMOOTH):
        self.model = model
        self.data = data
        self.smooth = float(smooth)
        self._ema = {}            # actuator name -> filtered angle
        # Cache actuator ids + ranges so apply() does no name lookups.
        self._act = {}
        for side in ("Left", "Right"):
            for finger, F in _FINGER_NAME.items():
                for seg, S in _SEG_NAME.items():
                    name = f"{side} {F} {S}"
                    try:
                        aid = model.actuator(name).id
                    except Exception:
                        continue
                    lo, hi = model.actuator_ctrlrange[aid]
                    self._act[(side, f"{finger}_{seg}")] = (name, float(lo), float(hi))

    def apply(self, side, joints, chirality=None):
        """side: 'Left' | 'Right'; joints: {wire_name: 4x4 world transform}.
        Returns the dict of commanded angles (post-clamp) or None."""
        chir = chirality or side.lower()
        angles = finger_angles(joints, chir)
        if not angles:
            return None
        out = {}
        a = self.smooth
        for suffix, theta in angles.items():
            entry = self._act.get((side, suffix))
            if entry is None:
                continue
            name, lo, hi = entry
            prev = self._ema.get(name)
            sm = theta if prev is None else prev + a * (theta - prev)
            self._ema[name] = sm
            cmd = float(min(max(sm, lo), hi))
            self.data.actuator(name).ctrl[0] = cmd
            out[suffix] = cmd
        return out
