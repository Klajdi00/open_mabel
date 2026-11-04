"""mock_control_publisher.py — inject SYNTHETIC teleop command targets onto the
robot's command topics so the collector logs a non-trivial `action/ctrl`, WITHOUT
moving the robot.

The arm/hand subsystems are brought up disabled (or torque-off), so nothing
consumes these commands into motion — they exist purely as the supervision signal
recorded alongside the real sensor streams. This is the "mock control input" for
the data-collection dry run.

Publishes slow, smooth, per-joint sinusoids (bounded, small amplitude) as
sensor_msgs/JointState on:  /arms/command, /left_hand/command,
/right_hand/command, /body/command  — joint names match the URDF /joint_states.
"""
from __future__ import annotations
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARMS = ([f"left_arm_{i}" for i in range(1, 8)] + ["left_wrist"] +
        [f"right_arm_{i}" for i in range(1, 8)] + ["right_wrist"])
_FINGERS = ["thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
            "index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp",
            "middle_pip", "ring_abd", "ring_mcp", "ring_pip",
            "pinky_abd", "pinky_mcp", "pinky_pip"]
LHAND = [f"left_{f}" for f in _FINGERS]
RHAND = [f"right_{f}" for f in _FINGERS]
BODY = ["torso", "neck_yaw", "neck_pitch", "neck_roll"]


class MockControl(Node):
    def __init__(self):
        super().__init__("mock_control_publisher")
        self.pubs = {
            "/arms/command": (self.create_publisher(JointState, "/arms/command", 10), ARMS, 0.15),
            "/left_hand/command": (self.create_publisher(JointState, "/left_hand/command", 10), LHAND, 0.30),
            "/right_hand/command": (self.create_publisher(JointState, "/right_hand/command", 10), RHAND, 0.30),
            "/body/command": (self.create_publisher(JointState, "/body/command", 10), BODY, 0.10),
        }
        self.t0 = time.time()
        self.create_timer(0.05, self._tick)   # 20 Hz
        self.get_logger().info("mock control publishing synthetic targets (robot NOT commanded to move)")

    def _tick(self):
        t = time.time() - self.t0
        for topic, (pub, names, amp) in self.pubs.items():
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = names
            # each joint a slow sinusoid, phase-spread so the vector is not uniform
            msg.position = [amp * math.sin(0.5 * t + 0.37 * i) for i in range(len(names))]
            pub.publish(msg)


def main():
    rclpy.init()
    node = MockControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
