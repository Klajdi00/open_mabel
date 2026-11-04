"""Optional sibling-subpackage dependencies (mabel.*), with graceful fallback.

This package LIVES inside `mabel` (controller/mabel/teleop_engine — moved from
simulation/mabel_mujoco/scripts/teleop on 2026-07-22 so sim and the real robot
share one installable retargeting/control stack), so no sys.path wiring is
needed; the try/except fallbacks remain so a partial install degrades feature-
by-feature instead of failing to import."""
try:
    from mabel.swerve import swerve_ik
    from mabel import config as MC          # tip-over motion model (lift-dependent)
    _HAVE_SWERVE = True
except Exception:
    _HAVE_SWERVE = False
    MC = None
    swerve_ik = None
try:
    from mabel.self_collision import SelfCollisionGuard, solve_qp
    _HAVE_SELFCOL = True
except Exception:
    _HAVE_SELFCOL = False
    SelfCollisionGuard = None
    solve_qp = None
try:
    # analytic base-frame estimator: robust operator base yaw from head + two
    # wrists (rejects look-around / reach / gaze-fixed that fool the elbow line)
    from mabel.whole_body_retargeting.body_frame_estimator import BodyFrameEstimator
    _HAVE_BODYFRAME = True
except Exception:
    _HAVE_BODYFRAME = False
    BodyFrameEstimator = None
try:
    # dexterous_hand-planner finger IK (keypoint_ik: the planner_avp /
    # planner_per_finger port) — the engine's default finger solver; the
    # anatomical decomposition is the fallback when this is absent.
    from mabel.hand_retargeting.keypoint_ik import (
        HandIKRetargeter, human_keypoints as kp_human_keypoints)
    _HAVE_KEYPOINT = True
except Exception:
    _HAVE_KEYPOINT = False
    HandIKRetargeter = None
    kp_human_keypoints = None
