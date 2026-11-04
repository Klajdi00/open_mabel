"""Command-based odometry for MABEL, using the calibrated base motion model.

Integrates the velocity command you SEND to the base (/mabel_cmd) through the
sim-identified forward model (controller/mabel/base_motion.BaseMotionModel: 0.989
slip gain + the swerve alignment ramp) to dead-reckon a pose, published on
``/odom_cmd``.

WHY this helps localization: unlike wheel-encoder odometry, this never depends on
wheel-ground contact, so it is IMMUNE to wheel SPIN/slip -- in free space it tracks
ground truth to ~2% (benchmarked). It is a clean, high-rate PROCESS model to fuse
with cuVSLAM (down-weighted, like the wheels), and a reference for SLIP/STUCK
detection: if cuVSLAM moves far less than this predicts, the base is BLOCKED; if the
wheel encoders read far more, the wheels are SLIPPING. It is NOT an absolute
reference (it still over-reads when the robot is physically stuck) -- cuVSLAM is.
"""
import math
import os
import sys
import types

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from mabel_interfaces.msg import MabelSwerveCmd

# Reuse the controller's calibrated motion model (the SAME code the server runs),
# importing only base_motion without the heavy package __init__.
_CTRL = os.path.expanduser(os.environ.get("MABEL_CONTROLLER", "~/open_mabel/controller"))
if "mabel" not in sys.modules:
    _pkg = types.ModuleType("mabel")
    _pkg.__path__ = [os.path.join(_CTRL, "mabel")]
    sys.modules["mabel"] = _pkg
from mabel import base_motion as BM   # noqa: E402


class CommandOdometry(Node):
    def __init__(self):
        super().__init__("command_odometry")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", False)   # EKF input by default, not TF owner
        # Loose covariance -- this over-reads when the robot is physically stuck, so
        # the fusion must never let it dominate cuVSLAM (same role as the wheels).
        self.declare_parameter("twist_cov_vxy", 0.05)
        self.declare_parameter("twist_cov_vyaw", 0.02)
        self.rate = float(self.get_parameter("rate").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.cov_vxy = float(self.get_parameter("twist_cov_vxy").value)
        self.cov_vyaw = float(self.get_parameter("twist_cov_vyaw").value)

        self.model = BM.BaseMotionModel()
        self.cmd = (0.0, 0.0, 0.0)
        self.pub = self.create_publisher(Odometry, "/odom_cmd", 10)
        self.tf = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(MabelSwerveCmd, "/mabel_cmd", self._on_cmd, 10)
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(
            f"command odometry up: /mabel_cmd -> /odom_cmd (BaseMotionModel, "
            f"gain={BM.SLIP_GAIN}, tau={BM.TAU}s)")

    def _on_cmd(self, m: MabelSwerveCmd):
        self.cmd = (float(m.vx), float(m.vy), float(m.omega))

    def _tick(self):
        dt = 1.0 / self.rate
        vx, vy, w = self.cmd
        x, y, th = self.model.step(vx, vy, w, dt)
        now = self.get_clock().now().to_msg()

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.orientation.z = math.sin(th / 2.0)
        od.pose.pose.orientation.w = math.cos(th / 2.0)
        # ramped actual body twist (what the model believes the base is doing now)
        od.twist.twist.linear.x = self.model.vx
        od.twist.twist.linear.y = self.model.vy
        od.twist.twist.angular.z = self.model.w
        pc = [0.0] * 36
        for i in (0, 7, 14, 21, 28, 35):
            pc[i] = 1e6                       # pose not fused (dead-reckoned, drifts)
        od.pose.covariance = pc
        tc = [0.0] * 36
        tc[0] = self.cov_vxy; tc[7] = self.cov_vxy; tc[14] = 1e6
        tc[21] = 1e6; tc[28] = 1e6; tc[35] = self.cov_vyaw
        od.twist.covariance = tc
        self.pub.publish(od)

        if self.tf is not None:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation.z = od.pose.pose.orientation.z
            t.transform.rotation.w = od.pose.pose.orientation.w
            self.tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CommandOdometry()
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
