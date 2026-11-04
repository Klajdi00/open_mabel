#!/usr/bin/env python3
"""ORCA hand bringup node (one instance per hand).

A single node drives ONE physical ORCA hand on ONE Feetech serial bus. Run two
of them (under the ``left_hand`` and ``right_hand`` namespaces) to drive both
hands -- see ``both_hands.launch.py``.

This node is a thin ROS wrapper around the standalone ``orcahand`` package
(``firmware/orca_hand/feetech``) -- the single hardware-abstraction layer also
used directly by controllers and policies. All the joint <-> tick conversion,
calibration clamping, and serial framing live in ``orcahand.OrcaHand``; this
file only does ROS plumbing.

Interfaces (all relative -> live under the node's namespace):

  Subscribed:
    command            (sensor_msgs/JointState)  target joint angles [rad]
                       names may be the bare joint ("index_mcp") or the
                       side-prefixed joint ("left_index_mcp").

  Subscribed:
    torque_enable      (std_msgs/Bool)  True = engage (hold), False = torque OFF
                       (free-drive: motors de-energised, joints move by hand).

  Published (at ~publish_rate Hz):
    joint_states       (sensor_msgs/JointState)  position[rad] / velocity[rad/s]
                       / effort[normalised Present Load -1..1]
    temperatures       (std_msgs/Float64MultiArray)  per-joint deg C
    voltages           (std_msgs/Float64MultiArray)  per-joint volts
    torque_state       (std_msgs/Bool, latched)  True = engaged, False = free-drive

The joint <-> actuator-ID map and per-joint angle ranges come from the YAML
config (config/<side>_hand.yaml); raw tick limits come from the calibration
JSON it references (config/<side>_calibration.json). Both are the same files
the firmware's calibrate_limits.py produces.
"""

import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, MultiArrayLayout

from orcahand import OrcaHand


class HandNode(Node):

    def __init__(self):
        super().__init__("hand_node")

        # ---- parameters ----------------------------------------------------
        self.declare_parameter("side", "left")
        self.declare_parameter("port", "")                 # "" => auto-detect
        self.declare_parameter("baudrate", 1_000_000)
        self.declare_parameter("config_file", "")          # "" => share/config/<side>_hand.yaml
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("torque_limit", 500)
        self.declare_parameter("goal_speed", 2000)
        self.declare_parameter("goal_acc", 40)
        self.declare_parameter("auto_enable_torque", True)

        side = self.get_parameter("side").value
        port = self.get_parameter("port").value or None
        baud = int(self.get_parameter("baudrate").value)
        rate = float(self.get_parameter("publish_rate").value)

        cfg_path = self.get_parameter("config_file").value
        if not cfg_path:
            share = get_package_share_directory("mabel_hand_hardware")
            cfg_path = os.path.join(share, "config", f"{side}_hand.yaml")

        self.side = side
        self.frame_id = f"{side}_hand"
        self._prefix = f"{side}_"
        self.hand = None

        # ---- ROS interfaces (always advertised, even with no hardware) -----
        # so the topics exist and the status dashboard shows the hand WAITING
        # rather than the whole bring-up dying when this hand is unplugged.
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_temp = self.create_publisher(Float64MultiArray, "temperatures", 10)
        self.pub_volt = self.create_publisher(Float64MultiArray, "voltages", 10)
        self.sub_cmd = self.create_subscription(
            JointState, "command", self._on_command, 10)

        # Torque control: a command in (toggle motors on/off for free-drive) and
        # a LATCHED state out, so a dashboard/teleop that starts later still sees
        # the current engaged/free-drive mode without waiting for a change.
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_torque = self.create_publisher(Bool, "torque_state", latched)
        self.sub_torque = self.create_subscription(
            Bool, "torque_enable", self._on_torque, 10)

        # Explicitly-disabled side: port 'none' means "do not touch hardware,
        # stay idle". Used by the full bring-up to keep the second hand OFF by
        # default so it never auto-detects (and steals) the one plugged-in
        # adapter -- pass a real port to enable that side.
        if isinstance(port, str) and port.lower() in ("none", "off", "disabled"):
            self.get_logger().info(
                f"[{side}] disabled (port='{port}'); idle -- pass a real "
                f"{side}_port to enable this hand.")
            self._publish_torque_state()
            return

        # ---- build + connect the hand (the shared hardware layer) ----------
        # NEVER fatal: a missing adapter / unpowered hand must leave this node
        # alive and idle (no feedback timer) so the rest of the bring-up runs.
        auto_torque = bool(self.get_parameter("auto_enable_torque").value)
        try:
            self.hand = OrcaHand(
                config_path=cfg_path,
                port=port,
                baudrate=baud,
                torque_limit=int(self.get_parameter("torque_limit").value),
                goal_speed=int(self.get_parameter("goal_speed").value),
                goal_acc=int(self.get_parameter("goal_acc").value),
            )
            self.hand.connect(enable_torque=auto_torque, scan=True)
        except Exception as e:
            self.hand = None
            self.get_logger().warn(
                f"[{side}] hardware not available ({e}); node idle -- topics "
                f"advertised but no feedback until the hand is connected.")
            self._publish_torque_state()
            return

        self.side = self.hand.side
        self.frame_id = f"{self.side}_hand"
        self._prefix = f"{self.side}_"
        self.get_logger().info(f"[{self.side}] bus open on {self.hand.bus.port} @ {baud}")
        if self.hand.missing_joints:
            self.get_logger().warn(
                f"[{self.side}] NOT responding: {self.hand.missing_joints}")
        if auto_torque:
            self.get_logger().info(f"[{self.side}] torque ENABLED -- hand is live")
        else:
            self.get_logger().info(f"[{self.side}] torque OFF -- free-drive (move by hand)")
        self._publish_torque_state()

        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._publish_feedback)
        self.get_logger().info(
            f"[{self.side}] ready: {len(self.hand.joint_names)} joints, "
            f"feedback @ {rate} Hz")

    # ------------------------------------------------------------ torque mode
    def _on_torque(self, msg: Bool):
        """Engage motors (hold position) or release them for true free-drive.

        Toggled live (e.g. Tab in the status dashboard): with torque OFF the
        motors are fully de-energised so the joints move freely by hand, while
        joint_states keeps streaming position/velocity feedback.
        """
        if self.hand is None:
            return
        want = bool(msg.data)
        if want == bool(self.hand.is_torque_enabled):
            return
        try:
            self.hand.enable_torque() if want else self.hand.disable_torque()
        except Exception as e:
            self.get_logger().warn(f"[{self.side}] torque toggle failed: {e}")
            return
        self.get_logger().info(
            f"[{self.side}] torque {'ENGAGED -- holding' if want else 'OFF -- free-drive'}")
        self._publish_torque_state()

    def _publish_torque_state(self):
        on = bool(self.hand.is_torque_enabled) if self.hand is not None else False
        self.pub_torque.publish(Bool(data=on))

    # -------------------------------------------------------- command intake
    def _on_command(self, msg: JointState):
        if self.hand is None or not msg.name or not msg.position:
            return
        # In free-drive (torque off) ignore position targets so the hand stays
        # backdrivable -- a streaming controller can't silently re-grip it.
        if not self.hand.is_torque_enabled:
            return
        targets = {}
        for name, angle in zip(msg.name, msg.position):
            bare = name[len(self._prefix):] if name.startswith(self._prefix) else name
            targets[bare] = angle
        self.hand.set_joint_positions(targets)

    # ---------------------------------------------------------- feedback loop
    def _publish_feedback(self):
        now = self.get_clock().now().to_msg()
        state = self.hand.get_state()

        names, pos, vel, eff, temps, volts = [], [], [], [], [], []
        for jn in self.hand.joint_names:
            # Motor not responding / read timed out this cycle: get_state() omits
            # it from the snapshot, so OMIT the joint here too rather than reporting
            # 0.0 rad. robot_state_publisher (and the relay) retain the last good
            # angle for absent joints, so a dropped packet no longer snaps the
            # finger to straight -- the model stays synced with the real hand.
            if jn not in state.position:
                continue
            names.append(self._prefix + jn)
            pos.append(state.position[jn])
            vel.append(state.velocity.get(jn, 0.0))
            eff.append(state.effort.get(jn, 0.0))
            temps.append(state.temperature.get(jn, 0.0))
            volts.append(state.voltage.get(jn, 0.0))

        if not names:
            return

        js = JointState()
        js.header.stamp = now
        js.header.frame_id = self.frame_id
        js.name = names
        js.position = pos
        js.velocity = vel
        js.effort = eff
        self.pub_js.publish(js)

        self.pub_temp.publish(self._labelled_array(names, temps))
        self.pub_volt.publish(self._labelled_array(names, volts))

    @staticmethod
    def _labelled_array(names, values):
        m = Float64MultiArray()
        m.layout = MultiArrayLayout()
        m.layout.dim = [
            MultiArrayDimension(label=n, size=1, stride=1) for n in names
        ]
        m.data = [float(v) for v in values]
        return m

    # ----------------------------------------------------------------- close
    def shutdown(self):
        if self.hand is None:
            return
        try:
            self.hand.disconnect()
            self.get_logger().info(f"[{self.side}] torque disabled, bus closed")
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HandNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
