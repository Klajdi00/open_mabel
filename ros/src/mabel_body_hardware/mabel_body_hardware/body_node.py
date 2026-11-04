#!/usr/bin/env python3
"""Body/neck bridge CLIENT — links the ROS graph to the hardware_bridge neck subsystem.

The neck (serial direct-drive: yaw -> pitch -> roll, camera on the roll link) is now
owned by the standalone ``~/open_mabel/hardware_bridge`` (the single HAL): it owns the neck's
Dynamixel bus (XC330 servos on /dev/ttyUSB0, Protocol 2.0) AND the calibration store
(``hardware_bridge/calibration_data/mabel.json``, slice ``neck/serial`` — neutral-anchored
level-head zero). This node NO LONGER opens the serial port — it is the thin
translator/monitor between the MABEL teleop contract and the bridge, exactly like
``mabel_arms_hardware/arms_node.py`` is for the arms:

    /body/command        (JointState, task-space rad {neck_yaw, neck_pitch, neck_roll})
                                       teleop/ros_plant -> bridge  (pass-through: the HAL
                                       body_node subscribes it directly — no relay hop)
    /mabel_cmd           (MabelSwerveCmd.hardware_state)  e-stop mirror:
        engage  -> /body/torque_enable false   (limp)
        release -> /body/torque_enable true     (bridge holds-in-place first: no lurch)
    bridge -> /body/joint_states   pos/vel/effort   (joint_state_publisher source_list
                                       merges it into /joint_states for RViz/TF; the URDF
                                       already carries neck_yaw/neck_pitch/neck_roll)
    bridge -> /body/motor_health   {joint: {torqueNm, tempC, currentmA, voltV}} @2 Hz

The e-stop mirror is TRANSITION-only (it never spams a latched state), so manual torque
control (setup studio / CLI) still works between e-stop events; disable it with
``follow_estop:=false`` (the default). This node also logs bridge-neck liveness
transitions so a bring-up shell shows the neck online/offline like the other watchdog rows.

The old direct-Dynamixel driver this file used to be (``firmware/head/dynamixel_neck``,
which also ENERGIZED the neck on connect) lives in git history and in the HAL's
``dynamixel`` backend (``hardware_bridge/mabel_hw/drivers/dynamixel.py``) — one Protocol-2.0
framing, one owner. Calibrate the neck OFFLINE in the setup studio (it stops the HAL and
owns the bus); the HAL hot-reloads the master mabel.json the moment the studio saves.
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


class BodyBridgeClient(Node):
    def __init__(self):
        super().__init__("body_hardware_node")

        # Bridge contract topics (must match hardware_bridge/config.py BODY).
        self.declare_parameter("joint_states_topic", "/body/joint_states")
        self.declare_parameter("motor_health_topic", "/body/motor_health")
        self.declare_parameter("torque_enable_topic", "/body/torque_enable")
        # OFF by default during bring-up: with follow_estop false this node NEVER
        # publishes /body/torque_enable — the neck can only be energised by an explicit
        # manual publish. Opt in with follow_estop:=true once the neck is calibrated
        # (setup studio -> neck -> Set Level writes calibration_data/mabel.json neck/serial).
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
                "mabel_interfaces unavailable — e-stop mirror OFF; drive neck torque via "
                "/body/torque_enable manually.")

        self.create_timer(1.0, self._check_liveness)
        self.get_logger().info(
            f"body/neck bridge client up: monitoring {js_topic} (+{health_topic}); "
            f"e-stop mirror {'ON (/mabel_cmd -> ' + torque_topic + ')' if self._follow_estop and _HAVE_CMD else 'off'}. "
            f"Neck hardware + calibration owned by ~/open_mabel/hardware_bridge (MABEL_HW_NECK).")

    # --- liveness --------------------------------------------------------------
    def _on_js(self, _msg: JointState):
        self._last_js_t = time.monotonic()

    def _check_liveness(self):
        online = (self._last_js_t is not None and
                  time.monotonic() - self._last_js_t < self._liveness)
        if online and not self._online:
            self.get_logger().info("bridge neck ONLINE (/body/joint_states streaming).")
        elif not online and self._online:
            self.get_logger().warn(
                "bridge neck OFFLINE (no /body/joint_states) — is the hardware bridge up "
                "with MABEL_HW_NECK + MABEL_HW_BODY, and are the Dynamixels on the bus? "
                "(hardware_bridge/experiments/exp_neck.py)")
        self._online = online

    # --- e-stop mirror (transitions only) ---------------------------------------
    def _on_cmd(self, msg):
        estop = (msg.hardware_state == 1)
        if estop == self._estop:
            return
        if self._estop is None and not estop:
            # First sample with e-stop already released: just latch. Energising the neck
            # must take a REAL engage->release transition (or a manual /body/torque_enable),
            # never mere node startup while teleop streams.
            self._estop = estop
            return
        self._estop = estop
        # Engage -> limp; release -> energise (the bridge holds-in-place first, so
        # re-energising never lurches). Same rule the arms + hands follow.
        self._pub_torque.publish(Bool(data=not estop))
        self.get_logger().info(
            f"e-stop {'ENGAGED -> neck limp' if estop else 'released -> neck energised (hold-in-place)'}.")


def main(args=None):
    rclpy.init(args=args)
    node = BodyBridgeClient()
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
