#!/usr/bin/env python3
"""Arms bridge CLIENT — links the ROS graph to the hardware_bridge arms subsystem.

The bimanual arms are now owned by the standalone ``~/open_mabel/hardware_bridge`` (the
single HAL): it owns the arms Teensy ethernet link (or the legacy socketcan path) AND
the calibration store (``hardware_bridge/calibration_data/mabel.json``, slice
``arm/openarm-dual`` —
zero/sign/limits + the joint<->motor map). This node NO LONGER touches CAN — it is the
thin translator/monitor between the MABEL teleop contract and the bridge, exactly like
``mabel_base_hardware/swerve_node.py`` is for the base:

    /arms/command        (JointState)  teleop/ros_plant -> bridge  (pass-through: the
                                       bridge subscribes it directly — no relay hop)
    /mabel_cmd           (MabelSwerveCmd.hardware_state)  e-stop mirror:
        engage  -> /arms/torque_enable false   (free-drive / limp)
        release -> /arms/torque_enable true    (bridge holds-in-place first: no lurch)
    bridge -> /arms/joint_states   pos/vel/effort   (joint_state_publisher source_list
                                       merges it into /joint_states for RViz/TF)
    bridge -> /arms/motor_health   {joint: {torqueNm, tempC}} @2 Hz (apps' telemetry)

The e-stop mirror is TRANSITION-only (it never spams a latched state), so manual
torque control (GUI / CLI) still works between e-stop events; disable it with
``follow_estop:=false``. The node also logs bridge-arms liveness transitions so a
bring-up shell shows arms online/offline like the other watchdog rows.

The old direct-CAN driver this file used to be lives in git history and in the
bridge's "can" transport (hardware_bridge/subsystems/damiao.py) — one MIT framing,
one owner.
"""
import time

import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

try:
    from mabel_interfaces.msg import MabelSwerveCmd
    _HAVE_CMD = True
except Exception:                                     # bench without the interfaces pkg
    MabelSwerveCmd = None
    _HAVE_CMD = False


class ArmsBridgeClient(Node):
    def __init__(self):
        super().__init__("arms_hardware_node")

        # Bridge contract topics (must match hardware_bridge/config.py ARMS).
        self.declare_parameter("joint_states_topic", "/arms/joint_states")
        self.declare_parameter("motor_health_topic", "/arms/motor_health")
        self.declare_parameter("torque_enable_topic", "/arms/torque_enable")
        # OFF by default during hardware bring-up: with follow_estop false this node
        # NEVER publishes /arms/torque_enable — the arms can only be energised by an
        # explicit manual publish. Opt in with follow_estop:=true once the arms are
        # calibrated (configs/*_arm_calibration.yaml are still placeholders).
        self.declare_parameter("follow_estop", False)
        self.declare_parameter("liveness_period", 2.0)   # s without feedback = offline

        js_topic = self.get_parameter("joint_states_topic").value
        health_topic = self.get_parameter("motor_health_topic").value
        torque_topic = self.get_parameter("torque_enable_topic").value
        self._follow_estop = bool(self.get_parameter("follow_estop").value)
        self._liveness = float(self.get_parameter("liveness_period").value)

        self._last_js_t = None
        self._online = False
        self._estop = None                               # unknown until first /mabel_cmd
        self.create_subscription(JointState, js_topic, self._on_js, 10)
        self.create_subscription(String, health_topic, lambda m: None, 5)

        self._pub_torque = self.create_publisher(Bool, torque_topic, 10)
        if self._follow_estop and _HAVE_CMD:
            self.create_subscription(MabelSwerveCmd, "/mabel_cmd", self._on_cmd, 10)
        elif self._follow_estop:
            self.get_logger().warn(
                "mabel_interfaces unavailable — e-stop mirror OFF; drive torque via "
                "/arms/torque_enable manually.")

        self.create_timer(1.0, self._check_liveness)
        self.get_logger().info(
            f"arms bridge client up: monitoring {js_topic} (+{health_topic}); "
            f"e-stop mirror {'ON (/mabel_cmd -> ' + torque_topic + ')' if self._follow_estop and _HAVE_CMD else 'off'}. "
            f"Arm hardware + calibration owned by ~/open_mabel/hardware_bridge.")

    # --- liveness --------------------------------------------------------------
    def _on_js(self, _msg: JointState):
        self._last_js_t = time.monotonic()

    def _check_liveness(self):
        online = (self._last_js_t is not None and
                  time.monotonic() - self._last_js_t < self._liveness)
        if online and not self._online:
            self.get_logger().info("bridge arms ONLINE (/arms/joint_states streaming).")
        elif not online and self._online:
            self.get_logger().warn(
                "bridge arms OFFLINE (no /arms/joint_states) — is the hardware "
                "bridge up with MABEL_HW_ARMS and the Teensy lane carrying? "
                "(hardware_bridge/install/setup_arms_lane.sh)")
        self._online = online

    # --- e-stop mirror (transitions only) ---------------------------------------
    def _on_cmd(self, msg):
        estop = (msg.hardware_state == 1)
        if estop == self._estop:
            return
        if self._estop is None and not estop:
            # First sample with e-stop already released: just latch. Energising 14
            # arm motors must take a REAL engage->release transition (or a manual
            # /arms/torque_enable), never mere node startup while teleop streams.
            self._estop = estop
            return
        self._estop = estop
        # Engage -> limp; release -> energise (the bridge holds-in-place first, so
        # re-energising never lurches). Same rule the hands follow.
        self._pub_torque.publish(Bool(data=not estop))
        self.get_logger().info(
            f"e-stop {'ENGAGED -> arms free-drive' if estop else 'released -> arms energised (hold-in-place)'}.")


def main(args=None):
    rclpy.init(args=args)
    node = ArmsBridgeClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
