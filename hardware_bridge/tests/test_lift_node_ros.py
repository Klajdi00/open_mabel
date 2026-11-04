"""The lift DDS adapter against a real rclpy graph, with the Pico faked.

Marked `ros`: auto-skipped where rclpy is absent (the Mac), runs for real on
thor. Complements test_ros_nodes_static.py (module SHAPE) and test_lift_driver.py
(the driver in isolation) by exercising the whole get/send path a client binds
to — over a LIVE DDS graph — the parts that rot silently without one:

  • /lift/command      (Float64 m)   -> the mock column tracks the target
  • /lift/joint_states (JointState)  -> position + velocity stream back out
  • /lift/calibrate    (Bool true)   -> stall-seek runs, `calibrated` flips
  • /lift/limits       (String)      -> the min/max envelope, LATCHED for a late join

The Pico is faked with MockLiftDriver (via MabelHardware(mock=True)), so this
needs no hardware — it is the automated stand-in for the physical bring-up
checklist while the real column is off the bus.
"""
import json
import threading
import time

import pytest

pytest.importorskip("rclpy")
pytestmark = pytest.mark.ros

import rclpy                                                        # noqa: E402
from rclpy.executors import MultiThreadedExecutor                   # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile                  # noqa: E402
from std_msgs.msg import Bool, Float64, String                      # noqa: E402
from sensor_msgs.msg import JointState                              # noqa: E402

from mabel_hw import config, MabelHardware                          # noqa: E402
from mabel_hw.ros.lift_node import LiftNode                         # noqa: E402


class Probe(Node):
    """Stands in for EVERY /lift consumer at once: the swerve_node relay's
    readback, the server module-status reader, and RViz's joint source."""

    def __init__(self, cfg):
        super().__init__("lift_probe")
        self.js, self.limits = [], []
        self.create_subscription(
            JointState, cfg["topic_joint_states"],
            lambda m: self.js.append((list(m.name), list(m.position),
                                      list(m.velocity))), 10)
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, cfg["topic_limits"],
            lambda m: self.limits.append(json.loads(m.data)), latched)
        self.cmd = self.create_publisher(Float64, cfg["topic_command"], 10)
        self.calib = self.create_publisher(Bool, cfg["topic_calibrate"], 10)


@pytest.fixture()
def graph():
    """A live lift node + probe on their own context, spinning in a thread.
    The lift is backed by the mock driver so the Pico need not be present."""
    cfg = dict(config.LIFT)
    if not rclpy.ok():
        rclpy.init()
    hw = MabelHardware(mock=True, subsystems=["lift"])
    hw.open()                                   # build the mock subsystems (else the
                                                # node falls back to its real-serial path)
    node = LiftNode(cfg, hw=hw)
    probe = Probe(cfg)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    time.sleep(0.5)
    yield node, probe, hw, cfg
    try:
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        probe.destroy_node()
        hw.close()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def _until(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_topics_are_advertised_with_the_contract_types(graph):
    node, _, _, _ = graph
    topics = dict(node.get_topic_names_and_types())
    for tp, ty in (("/lift/command", "std_msgs/msg/Float64"),
                   ("/lift/calibrate", "std_msgs/msg/Bool"),
                   ("/lift/joint_states", "sensor_msgs/msg/JointState"),
                   ("/lift/limits", "std_msgs/msg/String")):
        assert tp in topics, f"{tp} not advertised"
        assert ty in topics[tp], f"{tp} is {topics[tp]}, want {ty}"


def test_state_streams_out(graph):
    """The bridge PUBLISHES the lift joint (height + velocity) — the signal the
    server mirror and RViz both consume."""
    _, probe, _, cfg = graph
    assert _until(lambda: len(probe.js) > 0), "no /lift/joint_states ever published"
    names, pos, vel = probe.js[-1]
    assert names == [cfg["joint_name"]]
    assert len(pos) == 1 and len(vel) == 1


def test_command_moves_the_column(graph):
    """A /lift/command target reaches the (mock) Pico and the readback follows —
    proves the bridge RECEIVES commands and reflects them in state."""
    _, probe, _, _ = graph
    assert _until(lambda: len(probe.js) > 0)
    probe.cmd.publish(Float64(data=0.42))
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - 0.42) < 1e-6), \
        "the lift never tracked the commanded height"


def test_stroke_absolute_conversion_with_a_real_datum(graph):
    """The /lift/* contract is STROKE (0 == floor); the firmware is ABSOLUTE (floor
    at min_pos). With a real calibrated datum the node must convert BOTH ways:
    a stroke command lands at floor+stroke on the device, and the device's absolute
    height streams back out as stroke. This is the "lift stalls from the apps/web but
    works from the setup studio" fix — the setup studio drives absolute directly."""
    node, probe, _, _ = graph
    # Model a calibrated column: floor 0.3175 m, ceiling 0.9525 m (0.635 m stroke).
    drv = node.core._driver
    drv.min_pos, drv.max_pos = 0.3175, 0.9525
    drv._h = 0.3175                                       # sitting on the floor
    # Wait for the node to pick up the new floor datum (via _publish_state), which
    # also means the readback is now streaming in STROKE (floor -> 0).
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - 0.0) < 1e-3), \
        "floor should read as stroke 0, not the absolute datum"
    # Limits are reported in the stroke frame: [0, span].
    assert _until(lambda: probe.limits and abs(probe.limits[-1]["max"] - 0.635) < 1e-3
                  and probe.limits[-1]["min"] == 0.0), "limits not in the stroke frame"
    # Command 0.20 m of STROKE -> the device must move to 0.5175 m ABSOLUTE...
    probe.cmd.publish(Float64(data=0.20))
    assert _until(lambda: abs(drv._h - 0.5175) < 1e-3), \
        "stroke command was not lifted to floor+stroke on the device"
    # ...and stream back out as 0.20 m of stroke.
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - 0.20) < 1e-3), \
        "absolute height did not convert back to stroke on the way out"


def test_move_energizes_then_releases_and_dedups(graph):
    """The lift's control contract: HOLD == torque off (the worm gear holds), a CHANGED
    target energizes → moves → releases torque on arrival, and an UNCHANGED (held) target
    is a no-op (not re-streamed to the Pico). This is the setup-studio behaviour and the
    fix for "the column never moves from teleop" (a 50 Hz held setpoint kept re-planning
    the profile / re-energizing a resting column)."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    # boots into the HOLD state: no torque
    assert _until(lambda: drv._torque is False), "lift should boot released (no torque)"
    # a fresh target energizes, moves, and releases torque again on arrival
    probe.cmd.publish(Float64(data=0.30))
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - 0.30) < 2e-3), \
        "changed target did not move the column"
    assert _until(lambda: drv._torque is False), "torque not released after arrival"
    # a repeat of the SAME target is a no-op — it must NOT re-energize a resting column
    arrived = drv._h
    probe.cmd.publish(Float64(data=0.30))
    time.sleep(0.3)
    assert drv._torque is False, "an unchanged held target re-energized the motor"
    assert abs(drv._h - arrived) < 1e-9


def test_joystick_jog_moves_while_held_and_stops_on_release(graph):
    """The drive-stick JOG contract: a velocity on /lift/velocity energizes + drives the
    firmware velocity mode while the stick is held, and a return to centre (0) STOPS the
    motion and releases torque (worm gear holds). This is the "up/down while pressed, stop
    on release" the operator asked for; the position /lift/command stream is ignored while
    a jog owns the lift."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    vel = probe.create_publisher(Float64, "/lift/velocity", 10)
    # stick held UP -> energize + velocity mode at the commanded speed
    vel.publish(Float64(data=0.05))
    assert _until(lambda: drv._torque is True and abs(drv._vel - 0.05) < 1e-6), \
        "jog did not energize + command the firmware velocity"
    # a position setpoint arriving mid-jog is IGNORED (jog owns the lift)
    held = drv._h
    probe.cmd.publish(Float64(data=0.30))
    time.sleep(0.3)
    assert abs(drv._h - held) < 1e-9, "position setpoint moved the column during a jog"
    # stick released (0) -> stop + release torque
    vel.publish(Float64(data=0.0))
    assert _until(lambda: drv._torque is False and drv._vel == 0.0), \
        "jog release did not stop + release torque"


def test_limits_latched_for_a_late_subscriber(graph):
    """/lift/limits is TRANSIENT_LOCAL, so a dashboard/RViz that joins late still
    learns the travel envelope (the torque_state latching scar)."""
    _, probe, _, _ = graph
    assert _until(lambda: len(probe.limits) > 0), "limits never delivered"
    lim = probe.limits[-1]
    assert set(lim) == {"min", "max", "calibrated"}
    assert lim["calibrated"] is False          # not calibrated until commanded


def test_calibrate_runs_and_updates_the_envelope(graph):
    """A /lift/calibrate=true triggers the on-Pico stall-seek; `calibrated`
    flips true and the latched limits re-publish with the measured max."""
    _, probe, _, _ = graph
    assert _until(lambda: len(probe.limits) > 0)
    probe.calib.publish(Bool(data=True))
    assert _until(lambda: probe.limits[-1]["calibrated"] is True), \
        "calibrate never marked the lift calibrated"
    assert probe.limits[-1]["max"] > 0.0


def test_calibrate_false_is_ignored(graph):
    """A Bool(false) is not a calibrate trigger — only an explicit true is."""
    node, probe, _, _ = graph
    assert _until(lambda: len(probe.limits) > 0)
    probe.calib.publish(Bool(data=False))
    time.sleep(0.4)
    assert probe.limits[-1]["calibrated"] is False


def test_nan_position_command_is_ignored_and_never_latched(graph):
    """NaN is the gateway's "no lift command" sentinel (its measured-height seed
    hasn't grounded yet). It must be dropped ENTIRELY: no motion, no torque, and —
    critically — never latched into _target, where it would poison the dedup."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    h0 = drv._h
    probe.cmd.publish(Float64(data=float("nan")))
    time.sleep(0.3)
    assert node._target is None, "a NaN command must never latch a target"
    assert drv._torque is False and abs(drv._h - h0) < 1e-9


def test_nan_jog_velocity_is_ignored(graph):
    """A NaN on /lift/velocity must not energize or start a jog."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    vel = probe.create_publisher(Float64, "/lift/velocity", 10)
    vel.publish(Float64(data=float("nan")))
    time.sleep(0.3)
    assert drv._torque is False and not node._jogging


def test_seed_matching_measured_adopts_without_energizing(graph):
    """The gateway's bring-up seed streams the MEASURED height. The first grounded
    setpoint that already matches where the column is must be adopted as a HOLD —
    no torque, no zero-length move — and a later CHANGED target must still move."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    here = probe.js[-1][1][0]                     # measured stroke right now
    probe.cmd.publish(Float64(data=here))
    assert _until(lambda: node._target is not None), "seed target never adopted"
    assert drv._torque is False, "a hold-in-place seed must not energize the motor"
    probe.cmd.publish(Float64(data=here + 0.25))
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - (here + 0.25)) < 2e-3), \
        "a genuinely changed target must still move the column"


def test_connection_gap_regrounds_and_forgets_the_target(graph):
    """A Pico disconnect must (a) drop the grounding gate so commands received while
    the column is away are ignored WITHOUT latching, and (b) forget the dead
    session's target — so on rejoin the first setpoint is judged against reality,
    never against stale memory. This is the "lift dives to the last target on
    bring-up" scar."""
    node, probe, _, _ = graph
    drv = node.core._driver
    assert _until(lambda: len(probe.js) > 0)
    probe.cmd.publish(Float64(data=0.30))
    assert _until(lambda: node._target is not None and abs(node._target - 0.30) < 1e-6)
    # the Pico leaves; the node must re-ground and forget the setpoint
    drv._connected = False
    assert _until(lambda: node._have_fb is False and node._target is None), \
        "a connection gap must clear the grounding gate and the latched target"
    # a (stale) stream arriving while the column is away must be ignored, unlatched
    h_gone = drv._h
    probe.cmd.publish(Float64(data=0.05))
    time.sleep(0.3)
    assert node._target is None and abs(drv._h - h_gone) < 1e-9, \
        "a command received while disconnected leaked into the target/hardware"
    # rejoin: telemetry re-grounds, and a fresh setpoint works again
    drv._connected = True
    assert _until(lambda: node._have_fb is True), "telemetry did not re-ground"
    probe.cmd.publish(Float64(data=0.12))
    assert _until(lambda: probe.js and abs(probe.js[-1][1][0] - 0.12) < 2e-3), \
        "the lift did not accept a fresh grounded command after rejoin"
