"""Task-space wrist tracking on top of the pure PD.

    wrist target  ->  DLS IK  ->  q_des  ->  PD torque

Three stages, each independently measurable, which is the point of the rewrite:
when the arm misbehaves you can ask WHICH stage — did the IK fail to find a
solution (`ik_err` large), or did the PD fail to reach one it found (`joint_err`
large)? A controller that answers that question is debuggable; one that reports
a single "it moved wrong" is not.

`gravity=False` (the default) is the pure-PD baseline: the arm holds where
`kp * e == gravity torque`, so it settles BELOW its target by however much load
it carries — a real, quantified droop, not a mystery. `gravity=True` adds the
feed-forward that cancels it. Both are measured side by side in
`exp_basic_controller.py` (E1-E3, E6), so the choice is made on numbers.
"""
from __future__ import annotations

import numpy as np

from .pd import JointPD
from .wrist_ik import WristIK

# The 7-DOF chain per side. The ORCA wrist is deliberately NOT in the IK chain:
# it carries the hand and belongs to grasp orientation, not to reaching, and
# letting a position-only solver spend it is what twists the hand while the palm
# sits perfectly on target.
ARM_JOINTS = {
    "left":  [f"Left Arm {i}" for i in range(1, 8)],
    "right": [f"Right Arm {i}" for i in range(1, 8)],
}
PALM_BODY = {"left": "left_palm", "right": "right_palm"}


class TaskSpaceArm:
    """One arm: hold a joint pose, or track a wrist target in world space."""

    def __init__(self, model, data, side="right", kp=100.0, kd=20.0,
                 lambda_=0.08, rest=None, gravity=False):
        if side not in ARM_JOINTS:
            raise ValueError(f"side must be left|right, got {side!r}")
        self.side = side
        self.joints = ARM_JOINTS[side]
        self.pd = JointPD(model, data, self.joints, kp=kp, kd=kd,
                          gravity=gravity)
        self.ik = WristIK(model, data, self.joints, PALM_BODY[side],
                          lambda_=lambda_, rest=rest)
        self.target = None          # world xyz, or None = joint-hold only
        self.ik_err = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def attach(self):
        self.pd.attach().hold_here()
        return self

    def detach(self):
        self.pd.detach()
        return self

    # ── commands ─────────────────────────────────────────────────────────────
    def set_base_orientation(self, base_R):
        """World<-base rotation for gravity (torso lean). Forwarded to the PD;
        None = the baked rest-pose lean. On the robot: torso encoders / IMU."""
        self.pd.set_base_orientation(base_R)
        return self

    def hold_here(self):
        """Freeze at the measured pose and drop any wrist target."""
        self.target = None
        self.pd.hold_here()
        return self

    def set_wrist_target(self, xyz):
        """Track a world-space palm position. None returns to joint-hold."""
        self.target = None if xyz is None else np.asarray(xyz, dtype=float)
        return self

    def set_joint_targets(self, q_des):
        self.target = None
        self.pd.set_targets(q_des)
        return self

    # ── one tick ─────────────────────────────────────────────────────────────
    def step(self):
        """Re-solve (if tracking) and apply torque. Call before `mj_step`."""
        if self.target is not None:
            q, self.ik_err = self.ik.solve(self.target)
            self.pd.set_targets(dict(zip(self.joints, q)))
        self.pd.step()
        return self

    # ── measurement surface ──────────────────────────────────────────────────
    def palm(self):
        return self.ik.palm_pos()

    def task_error(self):
        """Metres between the palm and its target (nan when not tracking)."""
        if self.target is None:
            return float("nan")
        return float(np.linalg.norm(self.target - self.palm()))

    def joint_error(self):
        """Worst |q_des - q| over the chain, in radians."""
        e = self.pd.error()
        return max((abs(v) for v in e.values()), default=0.0)
