"""
mabel.whole_body_retargeting — operator body pose -> robot body targets.

Maps the operator's head + wrist poses (from ANY source: Vision Pro ARKit,
webcam skeleton via the web GUI studios — mabel.sources normalizes both into
OperatorPose) onto MABEL's body. Every piece is decoupled and independently
usable:

    wrist_clutch  Retargeter — clutch-relative wrist SE(3) targets (UMI-style
                  relative mapping with per-operator scale), feeding the DLS
                  WholeBodyIK. The desktop-viewer/legacy path.
    arm_cosine    ArmRetargeter — cosine-directional contact-point arm IK
                  (SLSQP), scale-independent; parity with the apps' engine.
    neck          NeckController — 3-DOF gaze look-at. DECOUPLED from the arm
                  retargeting: it reads only the head pose/gaze and drives only
                  the neck actuators, so it composes with any arm/hand mode
                  (and is skipped entirely when a client drives the neck in
                  joint space).
    followers     LazyBase / LazyLift / LazyTorso — reach-assist followers
                  that move the big DOFs only when the hand centroid leaves a
                  comfort box (MABEL's root-translation handling).
    calibration   OperatorProfile / BodyCalibrator — per-operator
                  anthropometry (3-pose capture) -> retarget scale, shoulder
                  anchor, per-finger sizing.

Finger retargeting is intentionally NOT here: hands are position-controlled
and decoupled (mabel.hand_retargeting).
"""
from .wrist_clutch import Retargeter                          # noqa: F401
from .arm_cosine import ArmRetargeter, robot_palm_frame       # noqa: F401
from .neck import NeckController, look_at_point               # noqa: F401
from .followers import LazyBase, LazyLift, LazyTorso          # noqa: F401
from .calibration import OperatorProfile, BodyCalibrator      # noqa: F401

__all__ = ["Retargeter", "ArmRetargeter", "robot_palm_frame",
           "NeckController", "look_at_point",
           "LazyBase", "LazyLift", "LazyTorso",
           "OperatorProfile", "BodyCalibrator"]
