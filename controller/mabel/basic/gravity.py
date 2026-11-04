"""Model-based gravity compensation. numpy only — no simulator, sim or real.

    tau_g,i  =  - SUM_{j >= i}  ( z_i x (c_j - p_i) ) . (m_j * g)

That is the whole thing: the generalised force gravity exerts on joint `i` is the
weight of every link distal to it, moment-armed about that joint's axis, and the
compensation is its negative. `z_i`, `p_i` and `c_j` come from forward kinematics
on the MEASURED joint angles.

WHY THIS AND NOT `qfrc_bias`
----------------------------
The first version read MuJoCo's `qfrc_bias` off a scratch `MjData`. It is correct
and it is one line, but it is privileged: it needs a live simulator, and on the
physical arm there is none. It also silently folds in whatever else the simulator
knows (contacts, external wrenches), which is *not* what a feed-forward should
carry — you want the model's prediction of gravity, not the truth of the moment,
or the term stops being a feed-forward and starts fighting the PD.

This version consumes exactly two things:
  1. a-priori model constants (`arm_params.py`, generated from the MJCF the same
     way a URDF is generated — link transforms, masses, COM offsets), and
  2. measured joint angles.
Both exist on hardware, so the same code runs in both places. Validated against
MuJoCo's own `qfrc_bias` over random poses in `exp_gravity_model.py` / the test
suite — agreement there is what licenses the claim that this is correct.

BASE ORIENTATION
----------------
Gravity enters through the shoulder's orientation, so a pitched torso changes the
load. `torque()` therefore takes the world<-base rotation. On the robot that comes
from the torso/lift encoders or an IMU; it defaults to the rest-pose value baked
by the extractor rather than to identity, because assuming "upright" for a robot
with a pitching torso is a silent 10-20% error on the shoulder joints.
"""
from __future__ import annotations

import numpy as np

from .arm_params import ARM, GRAVITY


def _quat_to_R(q):
    """MuJoCo/URDF scalar-first (w, x, y, z) -> 3x3."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ArmGravity:
    """Gravity torque for one 7-DOF arm, from measured joint angles.

    >>> g = ArmGravity("right")
    >>> tau = g.torque([0.0] * 7)            # {joint_name: N*m}
    """

    def __init__(self, side="right", joint_names=None, gravity=GRAVITY):
        if side not in ARM:
            raise ValueError(f"side must be one of {sorted(ARM)}, got {side!r}")
        self.side = side
        spec = ARM[side]
        self.links = spec["links"]
        self.n = len(self.links)
        self.base_R0 = np.array(spec["base_R"], dtype=float).reshape(3, 3)
        self.g = np.array(gravity, dtype=float)
        self.total_mass = float(spec["total_mass"])
        self.joint_names = list(joint_names) if joint_names else [
            f"{side.capitalize()} Arm {i}" for i in range(1, self.n + 1)]
        # constants hoisted out of the hot loop
        self._R_fixed = [_quat_to_R(L["quat"]) for L in self.links]
        self._p_fixed = [np.array(L["pos"], dtype=float) for L in self.links]
        self._m = np.array([L["mass"] for L in self.links], dtype=float)
        self._com = [np.array(L["com"], dtype=float) for L in self.links]

    # ── forward kinematics ───────────────────────────────────────────────────
    def frames(self, q, base_R=None):
        """(axes, anchors, coms) in the base frame, for measured angles `q`.

        axes[i]   : joint i's rotation axis (unit, world-aligned via base_R)
        anchors[i]: a point on joint i's axis
        coms[j]   : link j's centre of mass
        """
        q = np.asarray(q, dtype=float).ravel()
        if q.size != self.n:
            raise ValueError(f"expected {self.n} joint angles, got {q.size}")
        R = self.base_R0 if base_R is None else np.asarray(base_R, float).reshape(3, 3)
        p = np.zeros(3)
        axes, anchors, coms = [], [], []
        for i in range(self.n):
            # pre-joint frame: parent, then the link's fixed offset
            p = p + R @ self._p_fixed[i]
            R = R @ self._R_fixed[i]
            axes.append(R[:, 2].copy())     # every arm joint turns about local +Z
            anchors.append(p.copy())
            R = R @ _rot_z(q[i])            # then the joint angle itself
            coms.append(p + R @ self._com[i])
        return axes, anchors, coms

    # ── the law ──────────────────────────────────────────────────────────────
    def torque(self, q, base_R=None):
        """{joint: N*m} to hold the arm against gravity at pose `q`."""
        axes, anchors, coms = self.frames(q, base_R)
        out = {}
        for i in range(self.n):
            t = 0.0
            for j in range(i, self.n):      # only links DISTAL to joint i load it
                if self._m[j] <= 0.0:
                    continue
                r = coms[j] - anchors[i]
                F = self._m[j] * self.g
                t += float(np.dot(axes[i], np.cross(r, F)))
            out[self.joint_names[i]] = -t   # hold it, don't join it
        return out

    def torque_array(self, q, base_R=None):
        t = self.torque(q, base_R)
        return np.array([t[n] for n in self.joint_names], dtype=float)
