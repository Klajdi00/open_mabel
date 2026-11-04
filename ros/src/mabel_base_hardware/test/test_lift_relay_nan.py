"""Regression guard: the /mabel_cmd → /lift/command relay drops the NaN sentinel.

The real gateway streams /mabel_cmd.lift_pos = NaN while its measured-height lift
seed hasn't grounded (see mabel_control.ros_plant.set_lift_command_enabled).
BaseBridgeClient must treat NaN as "no lift command" and forward NOTHING — a
forwarded unseeded value is how the worm-gear column used to dive to the floor /
a stale target the moment the Pico appeared on bring-up. Finite values must keep
relaying exactly as before.

Skips cleanly without rclpy / the ROS overlay.
"""
import math
import time

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

from mabel_interfaces.msg import MabelSwerveCmd  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

from mabel_base_hardware.swerve_node import BaseBridgeClient  # noqa: E402


@pytest.fixture
def graph():
    rclpy.init()
    relay = BaseBridgeClient()
    probe = Node("test_lift_relay_probe")
    got = []
    probe.create_subscription(Float64, "/lift/command",
                              lambda m: got.append(float(m.data)), 10)
    pub = probe.create_publisher(MabelSwerveCmd, "/mabel_cmd", 10)
    yield relay, probe, pub, got
    probe.destroy_node()
    relay.destroy_node()
    rclpy.shutdown()


def _pump(relay, probe, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        rclpy.spin_once(relay, timeout_sec=0.02)
        rclpy.spin_once(probe, timeout_sec=0.02)


def test_finite_lift_pos_is_relayed_and_nan_is_dropped(graph):
    relay, probe, pub, got = graph

    # NaN = "no lift command": nothing may reach /lift/command.
    msg = MabelSwerveCmd()
    msg.lift_pos = math.nan
    deadline = time.time() + 2.0
    while time.time() < deadline:
        pub.publish(msg)
        _pump(relay, probe, 0.05)
    assert not got, (f"NaN lift_pos leaked through the relay as {got[:3]} — an "
                     "unseeded gateway value the column would dive to")

    # A finite value keeps relaying exactly as before.
    msg.lift_pos = 0.31
    deadline = time.time() + 10.0
    while time.time() < deadline and not got:
        pub.publish(msg)
        _pump(relay, probe, 0.05)
    # lift_pos is a float32 field — compare at float32 precision.
    assert got and abs(got[-1] - 0.31) < 1e-6, "finite lift_pos was not relayed"
