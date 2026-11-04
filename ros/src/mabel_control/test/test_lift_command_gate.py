"""Regression guard: the position /lift command gate on the real plant.

On real bring-up the gateway seeds MuJoCo's `Lift` ctrl from MEASURED /lift
feedback before any command may flow. The old one-shot seed gave up after 5 s and
the plant then streamed the unseeded ctrl (0 = the floor, or a stale target)
verbatim as /mabel_cmd.lift_pos → /lift/command — driving the worm-gear column
down the moment the Pico appeared ("lift dives on real_run start").

The fix pinned here: `ROSPlant.set_lift_command_enabled(False)` closes the lift
command channel — write() streams lift_pos=NaN, the contract's "no lift command"
sentinel (dropped by swerve_node, ignored by the HAL's lift_node) — until the
gateway's seed succeeds and reopens it. The gate must default OPEN so every other
ROSPlant user (mabel_gateway, sim) is byte-identical to before.

Skips cleanly without rclpy / mujoco / the ROS overlay.
"""
import math
import time
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

from pathlib import Path  # noqa: E402

from rclpy.node import Node  # noqa: E402
from mabel_interfaces.msg import MabelSwerveCmd  # noqa: E402

from mabel_control.ros_plant import ROSPlant  # noqa: E402

# The canonical MJCF — the single source of truth for the robot model.
MJCF = (Path(__file__).resolve().parents[4]
        / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml")


@pytest.fixture
def plant():
    if not MJCF.exists():
        pytest.skip(f"canonical MJCF not found at {MJCF}")
    rclpy.init()
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    node = Node("test_lift_gate")
    yield ROSPlant(node, model, mujoco.MjData(model)), node
    node.destroy_node()
    rclpy.shutdown()


def _lift_pos_after_write(p, node, sub_node):
    """One write() → the lift_pos the /mabel_cmd subscriber actually receives."""
    got = []
    sub = sub_node.create_subscription(
        MabelSwerveCmd, "/mabel_cmd", lambda m: got.append(float(m.lift_pos)), 10)
    try:
        cmd = SimpleNamespace(base=(0.0, 0.0, 0.0), engaged=1.0)
        deadline = time.time() + 10.0
        while time.time() < deadline and not got:
            p.write(cmd)
            rclpy.spin_once(sub_node, timeout_sec=0.05)
            rclpy.spin_once(node, timeout_sec=0.01)
        assert got, "/mabel_cmd never arrived"
        return got[-1]
    finally:
        sub_node.destroy_subscription(sub)


def test_gate_defaults_open_and_streams_finite(plant):
    """Untouched, the plant behaves exactly as before: a finite lift_pos streams."""
    p, node = plant
    sub_node = Node("test_lift_gate_sub_open")
    try:
        assert math.isfinite(_lift_pos_after_write(p, node, sub_node)), (
            "the lift gate must DEFAULT OPEN — closing it by default would silence "
            "the lift for every existing ROSPlant user")
    finally:
        sub_node.destroy_node()


def test_lift_feedback_age_tracks_only_lift_arrivals(plant):
    """The joint cache never forgets, so lift_feedback_age() is the gateway's only
    honest lift-liveness signal: huge at boot, unaffected by non-lift joints, fresh
    once a lift joint actually streams. The gateway closes its lift command gate on
    staleness and re-seeds from measured on return (a disconnected lift's desired
    height must never change)."""
    from sensor_msgs.msg import JointState  # noqa: PLC0415

    p, node = plant
    assert p.lift_feedback_age() > 60.0, "age must start ancient, not fresh"
    pub_node = Node("test_lift_fb_pub")
    try:
        pub = pub_node.create_publisher(JointState, "/joint_states", 10)
        # a non-lift joint must NOT freshen the lift stamp
        arm = JointState()
        arm.name, arm.position = ["left_arm_2"], [0.1]
        deadline = time.time() + 3.0
        while time.time() < deadline:
            pub.publish(arm)
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(pub_node, timeout_sec=0.01)
        assert p.lift_feedback_age() > 60.0, "a non-lift joint freshened the stamp"
        # a lift joint DOES
        msg = JointState()
        msg.name, msg.position = ["lift_lower"], [0.1]
        deadline = time.time() + 10.0
        while time.time() < deadline and p.lift_feedback_age() > 1.0:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(pub_node, timeout_sec=0.01)
        assert p.lift_feedback_age() < 1.0, "lift joint arrival never stamped"
    finally:
        pub_node.destroy_node()


def test_gate_closed_streams_nan_then_reopens(plant):
    """Closed gate -> lift_pos=NaN on the wire (the 'no lift command' sentinel);
    reopening restores the finite stream. This is the bring-up ordering the
    gateway relies on: closed until the measured-height seed lands."""
    p, node = plant
    sub_node = Node("test_lift_gate_sub_gated")
    try:
        p.set_lift_command_enabled(False)
        assert math.isnan(_lift_pos_after_write(p, node, sub_node)), (
            "a CLOSED lift gate must stream NaN — anything finite here is an "
            "unseeded target the column would dive to")
        p.set_lift_command_enabled(True)
        assert math.isfinite(_lift_pos_after_write(p, node, sub_node)), \
            "reopening the gate must restore the finite lift command stream"
    finally:
        sub_node.destroy_node()
