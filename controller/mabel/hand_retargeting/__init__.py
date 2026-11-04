"""
mabel.hand_retargeting — operator finger skeleton -> ORCA hand joint targets.

Hand control is POSITION-controlled and fully DECOUPLED from the rest of the
body: whatever the arms/torso/base are doing (kinematic IK, impedance,
compliance), the fingers always track through here. The input is the same
device-agnostic hand skeleton for every source — Vision Pro ARKit joints or a
webcam hand-landmark skeleton (mabel.sources converts both into OperatorPose
hand matrices) — so the retargeters never know which device produced it.

Three interchangeable retargeters (benchmarked in docs/mabel_controller.pdf):

    keypoint_ik  HandIKRetargeter — DexPilot-style keypoint IK on the ORCA
                 kinematics (SLSQP reference cost or fast damped Gauss-Newton;
                 config.HAND_SOLVER). The default (config.FINGER_RETARGET).
    vector       HandVectorRetargeter — AnyTeleop-style scale-invariant vector
                 optimization with the per-user hand-size alpha.
    anatomical   HandRetargeter — direct per-joint anatomical angle mapping
                 (no optimization; cheapest, least accurate pinches).
    grasp        apply_grasp — a single [0,1] grip value curls all fingers
                 (the app slider / pinch-to-grasp path).

`make_hand_retargeter(model, data)` builds the configured one.
"""
from .keypoint_ik import HandIKRetargeter, human_keypoints   # noqa: F401
from .vector import HandVectorRetargeter, make_hand_retargeter  # noqa: F401
from .anatomical import HandRetargeter, finger_angles        # noqa: F401
from .grasp import apply_grasp                               # noqa: F401

__all__ = ["HandIKRetargeter", "HandVectorRetargeter", "HandRetargeter",
           "make_hand_retargeter", "apply_grasp", "human_keypoints",
           "finger_angles"]
