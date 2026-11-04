#!/usr/bin/env python3
"""Base bridge CLIENT — adapts /mabel_cmd to the hardware_bridge DDS contract.

The mobile base is now owned by the standalone ``~/open_mabel/hardware_bridge`` (the single
HAL): it owns the Teensy link AND the validated ps5_teleop inverse kinematics +
calibration. This node NO LONGER touches the Teensy — it is a thin translator between
the MABEL teleop/arbiter contract and the bridge's generic, mabel_interfaces-free topics:

    /mabel_cmd (MabelSwerveCmd, SIM frame)                    [teleop / arbiter]
        │  vx -> -vx, vy -> -vy, omega -> omega   (sim -> real 180-deg yaw flip)
        ▼
    /mabel/hw/base/cmd_vel (geometry_msgs/Twist, REAL frame)  -> hardware_bridge -> motors
    /mabel/hw/base/estop   (std_msgs/Bool)                    -> hardware_bridge
    /lift/command          (std_msgs/Float64, metres)         -> hardware_bridge lift

    hardware_bridge -> /mabel/hw/base/joint_states (JointState) ──┐ republish
                                                                  ▼
    /joint_states  ──>  swerve_odometry (forward kinematics -> /odom + odom->base_link)

    hardware_bridge -> /lift/joint_states (JointState, joint "lift", metres of stroke)
        │  split the single column joint into the URDF's two prismatic stages
        ▼
    /lift/joint_states_urdf (JointState, lift_lower + lift_upper, each = stroke/2)
        ──>  joint_state_publisher source_list  ──>  robot_state_publisher -> TF -> RViz

The 180-deg flip lives HERE (not in the bridge) so the bridge stays in the one canonical
real-robot frame the validated ps5 path uses. ``swerve_odometry`` is unchanged.

Lift-state note: the bridge publishes ONE joint "lift" the URDF doesn't have; the URDF
models the column as two equal prismatic stages lift_lower + lift_upper (each 0..0.3175 m,
0..0.635 m total). We split the incoming stroke in half across the two stages — the SAME
total-stroke convention cmd_vel_bridge already uses when it SUMS the two stages back. With
no lift on the bridge the topic is silent and joint_state_publisher holds the home default
(today's behaviour), so this never destabilises a lift-less bring-up.
"""
import math

import rclpy
import rclpy.logging
from rclpy.node import Node

from mabel_interfaces.msg import MabelSwerveCmd
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64


class BaseBridgeClient(Node):
    def __init__(self):
        super().__init__('base_bridge_client')

        # Bridge contract topics (must match hardware_bridge/config.py BASE).
        self.declare_parameter('cmd_vel_topic', '/mabel/hw/base/cmd_vel')
        self.declare_parameter('estop_topic', '/mabel/hw/base/estop')
        self.declare_parameter('bridge_joint_states', '/mabel/hw/base/joint_states')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('lift_topic', '/lift/command')
        self.declare_parameter('forward_lift', True)
        # Lift STATE readback -> URDF-native two-stage joints for RViz.
        self.declare_parameter('lift_state_topic', '/lift/joint_states')
        self.declare_parameter('lift_state_urdf_topic', '/lift/joint_states_urdf')
        self.declare_parameter('lift_stage_joints', ['lift_lower', 'lift_upper'])
        self.declare_parameter('lift_stroke', 0.635)         # m, URDF total travel

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        bridge_js = self.get_parameter('bridge_joint_states').value
        js_topic = self.get_parameter('joint_states_topic').value
        self._lift_topic = self.get_parameter('lift_topic').value
        self._forward_lift = bool(self.get_parameter('forward_lift').value)
        self._lift_stage_joints = list(self.get_parameter('lift_stage_joints').value)
        self._lift_stroke = float(self.get_parameter('lift_stroke').value)

        # /mabel_cmd -> bridge.
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self._estop_pub = self.create_publisher(Bool, estop_topic, 10)
        self._lift_pub = (self.create_publisher(Float64, self._lift_topic, 10)
                          if self._forward_lift else None)
        self.create_subscription(MabelSwerveCmd, '/mabel_cmd', self._on_cmd, 10)

        # bridge joint_states -> /joint_states (the input swerve_odometry consumes).
        self._js_pub = self.create_publisher(JointState, js_topic, 10)
        self.create_subscription(JointState, bridge_js, self._on_js, 10)

        # bridge /lift/joint_states (joint "lift", metres) -> the URDF's two stages,
        # so a moving column animates in RViz. Silent lift -> no publish -> home default.
        self._lift_state_urdf_pub = self.create_publisher(
            JointState, self.get_parameter('lift_state_urdf_topic').value, 10)
        self.create_subscription(
            JointState, self.get_parameter('lift_state_topic').value,
            self._on_lift_state, 10)

        self.get_logger().info(
            f"base bridge client up: /mabel_cmd -> {cmd_vel_topic} (+estop, lift), "
            f"{bridge_js} -> {js_topic}. Base hardware owned by ~/open_mabel/hardware_bridge.")

    def _on_cmd(self, msg: MabelSwerveCmd):
        estop = (msg.hardware_state == 1)
        self._estop_pub.publish(Bool(data=estop))

        twist = Twist()
        if not estop:
            # /mabel_cmd is the SIM body frame (forward = vx < 0); the real /server frame
            # is +x forward / +y left — a 180-deg yaw flip. Identical keypresses then move
            # sim and real the same physical way (verify forward/left on the robot once).
            twist.linear.x = -float(msg.vx)
            twist.linear.y = -float(msg.vy)
            twist.angular.z = float(msg.omega)
        self._cmd_pub.publish(twist)

        # NaN = the gateway's "no lift command" sentinel (its /lift seed hasn't
        # grounded yet) — drop it, never forward. Forwarding an unseeded value is
        # how the column used to dive to the floor / a stale target on bring-up.
        if self._lift_pub is not None and math.isfinite(msg.lift_pos):
            self._lift_pub.publish(Float64(data=float(msg.lift_pos)))

    def _on_js(self, msg: JointState):
        # Republish the bridge's encoder readback as /joint_states (optionally remapped
        # under a full bring-up so a joint_state_publisher can merge all subsystems).
        self._js_pub.publish(msg)

    def _on_lift_state(self, msg: JointState):
        # The bridge sends ONE joint "lift" (metres of column stroke). The URDF has two
        # equal prismatic stages, so each carries half the stroke. Clamp to the URDF
        # travel so a units mismatch can never fling the render past its stops.
        try:
            i = list(msg.name).index('lift')
        except ValueError:
            return                                  # not the lift joint we split
        h = float(msg.position[i]) if i < len(msg.position) else 0.0
        h = max(0.0, min(self._lift_stroke, h))
        half = h / 2.0
        out = JointState()
        out.header = msg.header
        out.name = list(self._lift_stage_joints)
        out.position = [half] * len(self._lift_stage_joints)
        self._lift_state_urdf_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BaseBridgeClient()
    except Exception as e:                          # never crash the bring-up
        rclpy.logging.get_logger("base_bridge_client").warn(
            f"client failed to start ({e}).")
        rclpy.try_shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
