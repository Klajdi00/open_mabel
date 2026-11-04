"""arms.py — one bimanual arm as a typed subsystem.

A :class:`~mabel_hw.subsystems.joint_group.JointGroupSubsystem` (all the
calibration, model-frame targets, and lease-gated writes come from there) plus
the one thing that is specific to the arms: the **commandable range** each joint
exposes to a teleop client.

Everything this class emits is MODEL-frame radians — the MJCF/URDF joint
convention — so ROS, the server, and the GUI consume it directly and never see a
raw motor angle, an encoder turn, or a calibration window.

The control law is NOT here. Gravity compensation and the impedance/compliance
modes live in ``controller/mabel/arm_control`` and arrive as ``velocity``/``kp``/
``kd``/``effort`` on :meth:`~mabel_hw.subsystems.joint_group.JointGroupSubsystem.command`.
This layer's whole job is: calibrate, read, write.
"""
from __future__ import annotations

from typing import Dict, List

from .joint_group import JointGroupSubsystem

__all__ = ["ArmsSubsystem"]


class ArmsSubsystem(JointGroupSubsystem):
    """One arm (7 DoF + gripper) over any JointGroupDriver backend."""

    def commandable_ranges(self) -> Dict[str, List[float]]:
        """``{joint: [lo, hi]}`` — the calibrated MODEL-frame range each joint may
        be commanded to, in radians, for a teleop slider to clamp against.

        This is the single place the calibration story resolves into a clean
        human-facing bound: the raw frame, the multi-turn window, and the anchor
        math all stay hidden, and the consumer gets only ``[lo, hi]`` model
        radians. Joints with no calibration entry are absent (they are
        uncalibrated pass-through and have no meaningful limit).

        Pulled in on BOTH ends by the subsystem's `end_stop_margin`, so this is the
        range a slider may actually command — not the mechanical stop. The two must
        agree: the calibrated window is the MEASURED hard stop, and a client that
        clamps to the stop itself asks the motor to drive into it (that is the
        l_j4 stall). A joint whose range is narrower than the margin is reported
        at its midpoint rather than inverted."""
        with self._lock:
            calib = dict(self._calib)
            margin = self._margin
        out = {}
        for name, jc in calib.items():
            if name not in self.names:
                continue
            lo, hi = float(jc.model_min), float(jc.model_max)
            if (hi - lo) > 2.0 * margin:
                lo, hi = lo + margin, hi - margin
            else:
                lo = hi = 0.5 * (lo + hi)
            out[name] = [lo, hi]
        return out
