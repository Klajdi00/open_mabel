"""The tip model's lift input must be CASCADE-TOTAL metres, from either naming.

`safe_base_step`'s z_lift is documented as "metres of cascaded lift travel",
`LIFT_RANGE == (0.0, 0.635)` -- the whole column. The lift reaches this node under
two different namings, and getting either wrong silently erodes tip safety:

  * sim / URDF  : `lift_lower` + `lift_upper`, each limited to 0..0.3175. This node
    used to read ONLY `lift_lower`, i.e. HALF of z_lift. At full lift the model then
    computed CoM h = 0.512 m instead of 0.655 m and permitted ~28% more decel and
    speed than is tip-safe -- most generous exactly when the robot is least stable.
  * real / bridge : ONE joint (mabel_hw LIFT_JOINT, default `lift`) already in total
    metres. `lift_lower` never appears, so the old lookup never matched and z_lift
    sat at 0.0 (= fully lowered = the most generous caps) no matter how high the
    lift actually was.

Failing OPEN on a missing sensor is the through-line of both bugs, so the last test
pins that an unseen lift does not silently read as "lowered and therefore safe".

Skips cleanly without rclpy / the ROS overlay.
"""
import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

from sensor_msgs.msg import JointState  # noqa: E402

from mabel_navigation.cmd_vel_bridge import CmdVelBridge  # noqa: E402

# Per-stage limit (URDF lift_lower/lift_upper) and the cascade total the model wants.
STAGE_MAX = 0.3175
TOTAL_MAX = 0.635


@pytest.fixture
def node():
    rclpy.init()
    n = CmdVelBridge()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def _js(**joints):
    m = JointState()
    m.name = list(joints)
    m.position = [float(v) for v in joints.values()]
    return m


def test_sim_naming_sums_both_cascade_stages(node):
    """lift_lower + lift_upper -> the TOTAL, not just the lower stage."""
    node._on_js(_js(lift_lower=STAGE_MAX, lift_upper=STAGE_MAX))
    assert node._lift == pytest.approx(TOTAL_MAX), (
        "z_lift must be the cascade TOTAL; reading only lift_lower halves it and "
        "makes the tip model think the CoM is ~14 cm lower than it is")


def test_sim_naming_ignores_unrelated_joints(node):
    node._on_js(_js(left_arm_1=1.2, lift_lower=0.1, lift_upper=0.1, neck_yaw=0.3))
    assert node._lift == pytest.approx(0.2)


def test_real_bridge_single_joint_is_already_total(node):
    """The bridge publishes one `lift` joint in total metres — take it as-is."""
    node._on_js(_js(lift=TOTAL_MAX))
    assert node._lift == pytest.approx(TOTAL_MAX), (
        "the bridge's LIFT_JOINT carries the full column height already; summing or "
        "halving it would misreport the CoM")


def test_missing_lift_keeps_last_value_and_is_flagged(node):
    """An absent lift must not silently reset to 0.0 (= most generous caps)."""
    assert node._lift_seen is False, "must start 'never seen', not 'seen at 0.0'"

    node._on_js(_js(lift=0.5))
    assert node._lift_seen is True
    assert node._lift == pytest.approx(0.5)

    # A later message with no lift in it (e.g. an arms-only source) must not
    # clobber the known height back to "fully lowered".
    node._on_js(_js(left_arm_1=0.4))
    assert node._lift == pytest.approx(0.5), (
        "a lift-less JointState must keep the last known height; resetting to 0.0 "
        "would silently restore the most generous tip caps")


def test_half_scale_regression_is_measurable(node):
    """Guard the actual safety number, not just the plumbing."""
    mm = pytest.importorskip("mabel_navigation.motion_model")
    node._on_js(_js(lift_lower=STAGE_MAX, lift_upper=STAGE_MAX))
    correct = mm.com_height(node._lift)
    halved = mm.com_height(STAGE_MAX)          # what the old code fed the model
    assert correct > halved, "sanity: total lift must read as a higher CoM"
    assert correct - halved > 0.1, (
        f"the half-scale bug understated CoM height by {correct - halved:.3f} m; "
        f"this test exists so it cannot come back unnoticed")
