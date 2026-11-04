"""Regression guard: the REAL plant consumes Nav2's twist through the controller.

The gateway's nav logic is plant-agnostic -- it gates on `hasattr(plant, "nav_twist")`
(server/sim_teleop_bridge.py). For a long time only `SimRosPlant` had that method, so:

  * in SIM  Nav2's twist was injected through the controller facade (tip model +
    operator takeover) -- correct;
  * on REAL `hasattr` was False, so the gateway skipped it and Nav2 instead drove
    the base by publishing the canonical /mabel_cmd itself -- ungated by the
    controller, and a second publisher racing the plant.

These tests pin the fix: `ROSPlant` exposes `nav_twist()` and feeds it from the
non-canonical /nav/mabel_cmd, so sim and real take the SAME path.

Skips cleanly without rclpy / mujoco / the ROS overlay.
"""
import time

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

NAV_TOPIC = "/nav/mabel_cmd"
# The gateway ignores a twist older than this, so a dead Nav2 ages out instead of
# latching its last command onto the base.
FRESH_S = 0.5


@pytest.fixture
def plant():
    if not MJCF.exists():
        pytest.skip(f"canonical MJCF not found at {MJCF}")
    rclpy.init()
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    node = Node("test_ros_plant_nav")
    yield ROSPlant(node, model, mujoco.MjData(model)), node
    node.destroy_node()
    rclpy.shutdown()


def test_rosplant_exposes_nav_twist(plant):
    """The gateway probes `hasattr(plant, "nav_twist")` — without it, real skips nav."""
    p, _ = plant
    assert hasattr(p, "nav_twist"), (
        "ROSPlant must expose nav_twist(): the gateway gates the whole nav-through-"
        "controller path on hasattr(plant, 'nav_twist'). Without it the real robot "
        "silently falls back to Nav2 driving the base ungated.")


def test_nav_twist_starts_stale(plant):
    """Before Nav2 says anything the twist must read as ancient, not as zero-fresh."""
    p, _ = plant
    _, _, _, age = p.nav_twist()
    assert age > FRESH_S, (
        f"nav_twist() age must start > {FRESH_S}s so the gateway ignores it until "
        f"Nav2 actually commands; got {age}")


def test_nav_twist_receives_the_nav_topic(plant):
    """The twist must arrive from /nav/mabel_cmd, fresh and uncorrupted."""
    p, node = plant
    pub_node = Node("test_nav_pub")
    pub = pub_node.create_publisher(MabelSwerveCmd, NAV_TOPIC, 10)
    msg = MabelSwerveCmd()
    msg.vx, msg.vy, msg.omega = 0.42, -0.13, 0.77

    try:
        deadline = time.time() + 10.0
        got = False
        while time.time() < deadline and not got:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(pub_node, timeout_sec=0.01)
            vx, _, _, age = p.nav_twist()
            got = age < FRESH_S and abs(vx - 0.42) < 1e-6

        vx, vy, w, age = p.nav_twist()
        assert got, f"nav_twist() never saw {NAV_TOPIC} (age={age})"
        assert abs(vy + 0.13) < 1e-6 and abs(w - 0.77) < 1e-6, (
            f"twist corrupted in transit: got ({vx}, {vy}, {w})")
    finally:
        pub_node.destroy_node()
