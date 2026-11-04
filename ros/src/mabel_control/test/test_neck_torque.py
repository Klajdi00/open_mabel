"""Regression guard: the REAL plant energises the neck on /body/torque_enable.

The neck (serial direct-drive Dynamixel, owned by the hardware bridge) boots
torque-OFF. Nothing energised it before this fix: the gateway drove the arms +
both hands through the e-stop torque edge but never the neck, so /body/command
targets were inert (limp neck) AND no e-stop ever released the neck's torque.
These bugs surfaced as "the control studio can't disable torque / neck commands
do nothing / e-stop doesn't reach the neck".

The gateway's torque edge is plant-agnostic — it gates on
``getattr(plant, "set_neck_torque", None)`` and ``getattr(plant,
"neck_torque_state", None)`` (server/sim_teleop_bridge.py). So the fix is:
``ROSPlant`` exposes ``set_neck_torque()`` (publishes Bool on
``/body/torque_enable``, the topic ``hardware_bridge/mabel_hw/ros/body_node.py``
subscribes) and ``neck_torque_state()`` (the latched /body/torque_state cache),
exactly like the arms. These tests pin that contract.

Skips cleanly without rclpy / mujoco / the ROS overlay.
"""
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

from rclpy.node import Node          # noqa: E402
from std_msgs.msg import Bool        # noqa: E402

from mabel_control.ros_plant import ROSPlant  # noqa: E402

MJCF = (Path(__file__).resolve().parents[4]
        / "simulation" / "mabel_mujoco" / "models" / "mabel_full.xml")

TORQUE_TOPIC = "/body/torque_enable"


@pytest.fixture
def plant_and_sink():
    if not MJCF.exists():
        pytest.skip(f"canonical MJCF not found at {MJCF}")
    rclpy.init()
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    node = Node("test_neck_torque")
    plant = ROSPlant(node, model, mujoco.MjData(model))
    # A sink node capturing whatever ROSPlant publishes on /body/torque_enable.
    sink = Node("test_neck_torque_sink")
    got = []
    sink.create_subscription(Bool, TORQUE_TOPIC, lambda m: got.append(bool(m.data)), 10)
    # Let discovery match the new pub/sub pair before any publish.
    t0 = time.time()
    while time.time() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.02)
        rclpy.spin_once(sink, timeout_sec=0.02)
    yield plant, node, sink, got
    node.destroy_node()
    sink.destroy_node()
    rclpy.shutdown()


def _pump(node, sink, got, want_len, budget=2.0):
    t0 = time.time()
    while len(got) < want_len and time.time() - t0 < budget:
        rclpy.spin_once(node, timeout_sec=0.02)
        rclpy.spin_once(sink, timeout_sec=0.02)


def test_rosplant_exposes_neck_torque_api(plant_and_sink):
    """The gateway probes getattr(plant, "set_neck_torque"/"neck_torque_state").
    Without them the whole neck stays limp and e-stop never touches it."""
    plant, _, _, _ = plant_and_sink
    assert callable(getattr(plant, "set_neck_torque", None)), (
        "ROSPlant must expose set_neck_torque(): the gateway energises the neck "
        "only through getattr(plant, 'set_neck_torque'). Without it /body/command "
        "is inert (limp neck) and no e-stop releases the neck.")
    assert callable(getattr(plant, "neck_torque_state", None)), (
        "ROSPlant must expose neck_torque_state() so the gateway can reconcile "
        "torque intent across a bridge restart (the neck boots torque-OFF).")


def test_neck_torque_state_starts_none(plant_and_sink):
    """Latched cache is None until body_node reports — so the reconcile does not
    fire on a fabricated 'False' before the bridge has spoken."""
    plant, _, _, _ = plant_and_sink
    assert plant.neck_torque_state() is None


def test_set_neck_torque_publishes_true(plant_and_sink):
    """Energise → a single Bool(True) reaches /body/torque_enable (the body_node
    topic). This is the message that was never sent before the fix."""
    plant, node, sink, got = plant_and_sink
    plant.set_neck_torque(True)
    _pump(node, sink, got, 1)
    assert got and got[-1] is True, (
        f"set_neck_torque(True) must publish Bool(True) on {TORQUE_TOPIC}; got {got}")


def test_set_neck_torque_publishes_false(plant_and_sink):
    """Limp / e-stop → Bool(False) reaches /body/torque_enable (the neck goes
    limp). This is the e-stop-on-neck path the studio could not drive before."""
    plant, node, sink, got = plant_and_sink
    plant.set_neck_torque(False)
    _pump(node, sink, got, 1)
    assert got and got[-1] is False, (
        f"set_neck_torque(False) must publish Bool(False) on {TORQUE_TOPIC}; got {got}")
