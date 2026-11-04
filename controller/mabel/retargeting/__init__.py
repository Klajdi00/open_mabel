"""mabel.retargeting — two-mode body retargeting (LOCAL / GLOBAL).

A from-the-basics implementation of §V-B of the MABEL whole-body control and
retargeting paper. The operator's head + two hands drive MABEL's upper body in
BOTH modes; the modes differ in which measurements they are allowed to use, and
therefore in whether the mobile base moves at all:

    LOCAL   S_L = {T_dh_L, T_dh_R, skeleton}
            Device-frame optical measurements only. The torso frame is fitted
            directly in the device frame (eq. 14), so nothing depends on the
            headset's world estimate, and nu_b^cmd = 0: the base stays put.
            This is what makes teleop work from a MOVING VEHICLE, where the
            cabin is visually static while the IMU integrates the vehicle's yaw
            and tilts its gravity reference.

    GLOBAL  S_G = S_L u {T_ad}
            Adds the device pose in the ARKit anchor frame, so heading can be
            integrated in F_a (eq. 13) and the operator's body motion commands
            the base.

The modes meet at exactly two quantities — one rotation R_bo and one base
channel nu_b^cmd (eq. 15) — so everything downstream (wrist targets, swivel,
gaze, fingers) is mode-invariant by construction, and the handoff between them
is EXACT rather than merely smooth (see mode.py's alignment latch).

Typical use, from the engine:

    rt = BodyRetargeter(mode="local")
    rt.apply_profile(operator_profile)          # optional, measured anthropometry
    dev = device_measurements(T_ad, hands, skeletons)   # T_dh = T_ad^-1 T_ah
    out = rt.update(dev, WorldMeasurements(T_ad), robot_frames, dt)
    #  out.wrist_targets -> the arm IK;  out.base_twist -> safe_base_step;
    #  out.neck / out.swivel / out.lift_rate / out.torso -> the body DoFs

Doc: docs/retargeting_modes.tex -> .pdf. Experiments:
experiments/exp_retarget_modes.py. Tests: tests/test_retarget_modes.py.
"""
from .types import (DeviceMeasurements, WorldMeasurements, BodyFrameEstimate,
                    RobotFrames, RetargetOutput)                    # noqa: F401
from .frames import (device_measurements, to_device_frame, rigid_inverse,
                     AVP_TO_ROBOT)                                  # noqa: F401
from .operator_model import OperatorModel                           # noqa: F401
from .local_frame import LocalTorsoFit, project_cervical            # noqa: F401
from .global_frame import GlobalBodyFrame                           # noqa: F401
from .mode import ModeArbiter, RetargetMode                         # noqa: F401
from .wrist_targets import wrist_targets, wrist_target              # noqa: F401
from .body_dofs import swivel_target, lift_and_torso, hand_centroid  # noqa: F401
from .retargeter import BodyRetargeter                              # noqa: F401

__all__ = [
    "BodyRetargeter", "RetargetMode", "ModeArbiter",
    "LocalTorsoFit", "GlobalBodyFrame", "OperatorModel",
    "DeviceMeasurements", "WorldMeasurements", "BodyFrameEstimate",
    "RobotFrames", "RetargetOutput",
    "device_measurements", "to_device_frame", "rigid_inverse", "AVP_TO_ROBOT",
    "wrist_targets", "wrist_target", "swivel_target", "lift_and_torso",
    "hand_centroid", "project_cervical",
]
