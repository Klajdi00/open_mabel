"""Backward-compatibility contract: every import line used by an external
consumer of the `mabel` package (server bridge, ROS nodes, web studios,
learning tools, hand-studio GUI, simulation scripts) must keep working after
the package reorganisation. This test IS that contract — extend it when a new
consumer imports a new symbol."""
import sys


def test_config_is_mujoco_free():
    # The ROS nav node imports mabel.config on the Jetson without MuJoCo/scipy.
    for m in [m for m in list(sys.modules) if m == "mabel" or m.startswith("mabel.")]:
        del sys.modules[m]
    had_mj = "mujoco" in sys.modules
    import mabel.config as C
    if not had_mj:
        assert "mujoco" not in sys.modules, "mabel.config must not import mujoco"
    C.safe_base_step(0, 0, 0, 0.3, 0, 0, 0.01, 0.2)
    assert C.COM_XY.shape == (2,)
    assert C.LIFT_RANGE == (0.0, 0.635)


def test_legacy_consumer_imports():
    # server/sim_teleop_bridge.py
    from mabel import config as C                              # noqa: F401
    from mabel import MabelTeleop, OperatorPose, app_joystick  # noqa: F401
    from mabel.recorder import EpisodeRecorder                 # noqa: F401
    from mabel.wbc_runtime import _IOS_TO_MJ                   # noqa: F401
    from mabel import mock_avp                                 # noqa: F401
    # server/_test_teleop_client.py
    from mabel.calibration import synthesize_pose, POSE_HANDS  # noqa: F401
    # server/tools + vision_os_apps/docs/tools
    from mabel.teleop_controller import TeleopController       # noqa: F401
    from mabel.hand_ik_retarget import HandIKRetargeter, human_keypoints  # noqa: F401
    # web_gui / unity control centers
    from mabel.wire import operator_pose_from_teleop_frame     # noqa: F401
    # ros (ROS): keyboard_teleop, mujoco_node, navigation
    from mabel.wbc import WholeBodyIK                          # noqa: F401
    from mabel.lazy_base import LazyBase                       # noqa: F401
    from mabel.wbc_runtime import WBCEngine                    # noqa: F401
    from mabel import base_motion as BM                        # noqa: F401
    # simulation scripts (compliance.py, keyboard_teleop.py, tools/*)
    from mabel.swerve import swerve_ik                         # noqa: F401
    from mabel.avp_receiver import AVPReceiver                 # noqa: F401
    from mabel.dashboard import Dashboard                      # noqa: F401
    from mabel.admittance import (external_base_wrench, BaseAdmittance,   # noqa: F401
                                  boundary_gate, hand_centroid_base)
    from mabel.force_estimation import ContactEstimator        # noqa: F401
    from mabel.dynamics import (UpperBodyDynamics, SimPlant,   # noqa: F401
                                ComplianceController,
                                COMPLIANT_ARM_JOINTS, LAZY_JOINTS)
    from mabel.self_collision import SelfCollisionGuard, solve_qp, _seg_seg  # noqa: F401
    # hardware_bridge / firmware hand-studio GUI
    from mabel.hand_vector_retarget import make_hand_retargeter  # noqa: F401
    # vision_os_apps experiment tools
    from mabel.hand_retarget import finger_angles, _FLEX_MAP, SMOOTH  # noqa: F401
    # simulation tools/test_retargeting.py
    from mabel.retarget import Retargeter                      # noqa: F401
    from mabel.facade import MabelTeleop as _MT                # noqa: F401
    from mabel.types import OperatorPose as _OP                # noqa: F401
    # simulation/mabel_mujoco/scripts/teleop/deps.py
    from mabel.self_collision import SelfCollisionGuard as _G  # noqa: F401
    # simulation_bridge/engine.py
    from mabel import swerve as _swerve                        # noqa: F401
    # com_support users (wbc_stability experiment)
    from mabel.com_support import robot_com, support_margin, CoMBaseSupport  # noqa: F401
    from mabel.head import NeckController                      # noqa: F401
    from mabel.hands import apply_grasp                        # noqa: F401
    from mabel.arm_retarget import ArmRetargeter, _natural_frame  # noqa: F401


def test_new_taxonomy_imports():
    from mabel.motion_model import safe_base_step, tip_usage, max_speed_dir  # noqa: F401
    from mabel.motion_model.tip_over import COM_XY, TIP_LEVER_FWD            # noqa: F401
    from mabel.dynamic_models import (MujocoDynamics, MomentumObserver,      # noqa: F401
                                      ArmsDynamicModel, UpperBodyDynamicModel,
                                      WholeBodyDynamicModel, region_joints)
    from mabel.whole_body_controller import (WholeBodyIK, WBCEngine,         # noqa: F401
                                             JointPositionController,
                                             GravityCompController,
                                             SelfCollisionGuard,
                                             ComplianceController, swerve_ik)
    from mabel.whole_body_controller.admittance import (ContactEstimator,    # noqa: F401
                                                        WrenchEstimator,
                                                        BaseAdmittance,
                                                        localize_contact)
    from mabel.hand_retargeting import (HandIKRetargeter, HandVectorRetargeter,  # noqa: F401
                                        HandRetargeter, make_hand_retargeter,
                                        apply_grasp)
    from mabel.whole_body_retargeting import (Retargeter, ArmRetargeter,     # noqa: F401
                                              NeckController, LazyBase,
                                              OperatorProfile, BodyCalibrator)
    from mabel.sources import mock_frame, AVPReceiver, operator_pose_from_teleop_frame  # noqa: F401
    from mabel.plant import Plant, SimPlant                                  # noqa: F401


def test_top_level_lazy_api_complete():
    import mabel
    for sym in mabel.__all__:
        assert getattr(mabel, sym) is not None, sym


def test_config_and_motion_model_are_same_objects():
    # config re-exports the motion model: same function objects, one source.
    import mabel.config as C
    from mabel.motion_model import tip_over as T
    assert C.safe_base_step is T.safe_base_step
    assert C.tip_usage is T.tip_usage
    assert C.COM_XY is T.COM_XY
