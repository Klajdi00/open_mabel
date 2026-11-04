"""Swerve wheel odometry for MABEL (forward kinematics from joint encoders).

This is the robot's MOTION MODEL used for state estimation: it integrates the
3-module swerve forward kinematics from ``/joint_states`` into a dead-reckoned
pose, publishing ``/odom`` + the ``odom -> base_link`` TF.

IMPORTANT -- runs on the REAL robot, no simulator privilege:
  * The ONLY input is ``/joint_states`` -- the steer-column angles (joint
    position) and drive-wheel speeds (joint velocity), which are real encoder
    feedback. It never reads MuJoCo ground truth.
  * In sim, run ``mujoco_node`` with ``publish_odom:=false`` so this node owns
    ``/odom`` + the TF (the sim then only emits ground truth on
    ``/odom_ground_truth`` for evaluation). cuVSLAM corrects ``map -> odom`` on
    top (publish_odom_to_base_tf:=false), so the drift this produces is exactly
    what visual SLAM is there to fix.

Forward kinematics: each module i has global wheel heading
``phi_i = steer_i + yaw_offset_i`` and ground speed ``s_i = drive_i * R``.  Its
contact velocity in the body frame is ``[vx - w*ry_i, vy + w*rx_i] = s_i*[cos,
sin]``; stacking the 3 modules gives an over-determined 6x3 system solved by
least squares for the body twist (vx, vy, w), which is then integrated.

The module geometry MUST match the base (interface_config.SWERVE_MODULES /
controller config). ``wheel_radius`` is a parameter: default 0.0378 m matches
the MuJoCo model; set 0.0381 m (3in REV MAXSwerve) on the real robot.
"""
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# Module [x, y] offsets (m) and steering-column mount yaw (rad). Must match
# mabel_sim.interface_config.SWERVE_MODULES and controller/mabel/config.py.
HALF_PI = math.pi / 2.0
SWERVE_MODULES = {
    "fl": {"steer": "fl_steer", "drive": "fl_drive",
           "pos": (-0.17, -0.17), "yaw": 1.570795 + HALF_PI},
    "fr": {"steer": "fr_steer", "drive": "fr_drive",
           "pos": (-0.17,  0.17), "yaw": 0.0 + HALF_PI},
    "b":  {"steer": "b_steer",  "drive": "b_drive",
           "pos": (0.15,  0.0),  "yaw": 3.9269926 + HALF_PI},
}


class SwerveOdometry(Node):
    def __init__(self):
        super().__init__("swerve_odometry")

        # Rolling radius (matches the sim drive assumption / real REV MAXSwerve).
        # NOTE: in the current sim the wheel COLLISION is the drive-motor mesh (no
        # round wheel) + a velocity drive actuator, so the contact slips ~40-50%
        # nonlinearly (worst at cold-start) and wheel odometry over-reads -- no
        # radius constant fixes that. Tuning models/swerve_module.xml (cylinder
        # wheel collision + torque/motor drive) is the real fix. On the real robot
        # (which grips) this 0.0381 is correct.
        self.declare_parameter("wheel_radius", 0.0381)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        # Velocity (twist) covariance the EKF uses to WEIGHT the wheels. The sim base
        # slips ~3x (benchmarked: wheel path 3.0x ground truth), and a real swerve
        # slips on turns too, so the wheel VELOCITY is only loosely trusted. Larger =
        # less weight on the wheels in the fused estimate. Pose covariance is set huge
        # because only the wheel VELOCITY is fused (cuVSLAM owns absolute pose).
        self.declare_parameter("twist_cov_vxy", 0.05)   # (m/s)^2 on vx, vy
        self.declare_parameter("twist_cov_vyaw", 0.02)  # (rad/s)^2 on yaw rate
        self.R = float(self.get_parameter("wheel_radius").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.cov_vxy = float(self.get_parameter("twist_cov_vxy").value)
        self.cov_vyaw = float(self.get_parameter("twist_cov_vyaw").value)

        # Precompute the least-squares geometry: rows [1,0,-ry; 0,1,rx] per module.
        self._mods = list(SWERVE_MODULES.values())
        rows = []
        for m in self._mods:
            rx, ry = m["pos"]
            rows.append([1.0, 0.0, -ry])
            rows.append([0.0, 1.0, rx])
        self._A = np.array(rows)                          # 6x3
        self._Apinv = np.linalg.pinv(self._A)             # 3x6

        # integrated pose (odom frame)
        self.x = self.y = self.th = 0.0
        self._last_t = None

        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, qos_profile_sensor_data)
        self.get_logger().info(
            f"swerve odometry up: /joint_states -> /odom"
            f"{' + odom->base_link TF' if self.publish_tf else ''} "
            f"(wheel_radius={self.R} m)")

    def _on_js(self, msg: JointState):
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        # need all 6 swerve joints present
        try:
            b = np.empty(6)
            for i, m in enumerate(self._mods):
                phi = pos[m["steer"]] + m["yaw"]          # global wheel heading
                s = vel.get(m["drive"], 0.0) * self.R     # signed ground speed
                b[2 * i] = s * math.cos(phi)
                b[2 * i + 1] = s * math.sin(phi)
        except KeyError:
            return                                        # swerve joints not (yet) present

        vx, vy, w = self._Apinv @ b                       # body-frame twist

        # timestamp / dt
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_t is None:
            self._last_t = t
            self._publish(msg.header.stamp, vx, vy, w)
            return
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0.0 or dt > 1.0:                         # skip bad/huge gaps
            self._publish(msg.header.stamp, vx, vy, w)
            return

        # integrate body twist into the odom frame (midpoint heading)
        thm = self.th + 0.5 * w * dt
        self.x += (vx * math.cos(thm) - vy * math.sin(thm)) * dt
        self.y += (vx * math.sin(thm) + vy * math.cos(thm)) * dt
        self.th = math.atan2(math.sin(self.th + w * dt), math.cos(self.th + w * dt))
        self._publish(msg.header.stamp, vx, vy, w)

    def _publish(self, stamp, vx, vy, w):
        qz = math.sin(self.th / 2.0)
        qw = math.cos(self.th / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.angular.z = float(w)
        # Pose covariance huge (wheel absolute pose NOT fused -- it drifts); twist
        # covariance from params so the EKF down-weights the slip-prone wheels.
        pc = [0.0]*36
        for i in (0, 7, 14, 21, 28, 35):
            pc[i] = 1e6
        odom.pose.covariance = pc
        tc = [0.0]*36
        tc[0] = self.cov_vxy; tc[7] = self.cov_vxy; tc[14] = 1e6
        tc[21] = 1e6; tc[28] = 1e6; tc[35] = self.cov_vyaw
        odom.twist.covariance = tc
        self.pub.publish(odom)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = SwerveOdometry()
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
