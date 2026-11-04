"""Regression guard: exactly ONE publisher may drive the base.

`/mabel_cmd` is the canonical base command topic. The gateway's plant (`ROSPlant`
on the real robot, `SimRosPlant` in sim) is its ONLY publisher. Nav2's twist takes
the long way round on purpose:

    Nav2 /cmd_vel -> cmd_vel_bridge -> /nav/mabel_cmd -> plant.nav_twist()
                  -> gateway injects it through the controller facade (tip model,
                     operator takeover) -> plant publishes /mabel_cmd -> base

The bug these tests exist to prevent: `cmd_vel_bridge` publishing `/mabel_cmd`
directly, which put TWO publishers on the base (teleop via the plant + Nav2), with
nothing arbitrating them. `cmd_vel_bridge` masked it by going silent when idle --
a mitigation, not arbitration. The old code admitted it in `ros_plant.py`:
"Driving the real base from Nav2 needs a cmd remap and is left for the hardware
bring-up".

Consequence worth stating: routing nav through the facade IS the estop > teleop >
nav arbitration -- the gateway marks nav inactive on human stick deflection, so
these two topics being distinct is a safety property, not plumbing taste.

Skips cleanly without rclpy / the ROS overlay (lint-only CI).
"""
import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

from mabel_navigation.cmd_vel_bridge import CmdVelBridge  # noqa: E402

# The canonical base topic, and the deliberately non-canonical nav topic.
CANONICAL = "/mabel_cmd"
NAV_TOPIC = "/nav/mabel_cmd"


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _published_topics(node):
    return [t for t, _ in node.get_publisher_names_and_types_by_node(node.get_name(), "")]


def test_cmd_vel_bridge_does_not_publish_the_canonical_base_topic(ros):
    """Nav2's bridge must never be a second publisher on /mabel_cmd."""
    node = CmdVelBridge()
    try:
        topics = _published_topics(node)
        assert CANONICAL not in topics, (
            f"cmd_vel_bridge publishes {CANONICAL} again -- that is a SECOND publisher "
            f"on the base (the plant is the first), with no arbiter between them. "
            f"Publish to {NAV_TOPIC} and let the plant re-inject it through the "
            f"controller facade. Topics: {topics}")
    finally:
        node.destroy_node()


def test_cmd_vel_bridge_publishes_the_nav_topic(ros):
    """The twist must still reach the plant -- via the nav topic."""
    node = CmdVelBridge()
    try:
        topics = _published_topics(node)
        assert NAV_TOPIC in topics, (
            f"cmd_vel_bridge must publish {NAV_TOPIC} (the plant subscribes it). "
            f"Topics: {topics}")
    finally:
        node.destroy_node()


def test_cmd_out_topic_is_overridable(ros):
    """The topic is a parameter so the plant and this node stay in lockstep."""
    node = CmdVelBridge()
    try:
        assert node.get_parameter("cmd_out_topic").value == NAV_TOPIC
    finally:
        node.destroy_node()


def test_motion_model_is_on_by_default(ros):
    """`use_motion_model:=false` makes the tip gate a runtime no-op -- never default."""
    node = CmdVelBridge()
    try:
        assert node.get_parameter("use_motion_model").value is True, (
            "the tip-safe motion model must default ON: it is the gate that stops "
            "Nav2 commanding a motion that tips the robot")
    finally:
        node.destroy_node()
