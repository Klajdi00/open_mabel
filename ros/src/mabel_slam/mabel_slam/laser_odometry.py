#!/usr/bin/env python3
"""laser_odometry.py — 2D LiDAR odometry: /scan → odom->base_link TF + /odom_laser.

The LiDAR-mode odometry source for the real robot (real_run.sh spawns this whenever
the RPLIDAR /scan is live): the robot's odom is DERIVED FROM THE LIDAR by
scan-to-keyframe ICP, so slam_toolbox gets the odom->base_link TF it requires with
no wheel odometry in the loop (the wheel /odom, when the base is on, remains purely
the twist feedback Nav2's MPPI reads via cmd_vel_bridge's /odom_nav — it never owns
TF in this mode).

  in:   /scan           sensor_msgs/LaserScan (frame lidar_link, ~13 Hz A2M12)
        TF base_link -> lidar_link   (static URDF chain, looked up once)
  out:  TF odom -> base_link         (this node is the SOLE owner in lidar mode)
        /odom_laser     nav_msgs/Odometry (pose in odom + finite-difference twist)

Method: point-to-point ICP against a KEYFRAME scan (not scan-to-scan — matching the
last keyframe kills the per-scan drift accumulation), seeded with the previous pose;
a new keyframe is adopted after 0.25 m / 0.25 rad of travel. 720-point scans at
13 Hz cost ~2-4 ms/frame with scipy cKDTree — negligible. Degenerate scans (too few
valid returns, ICP divergence) hold the last pose so TF never goes stale.

Run it in the conda mabel env (rclpy + scipy + CycloneDDS domain 0, same graph as
the bridge): python3 ros/src/mabel_slam/scripts/laser_odometry.py
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from scipy.spatial import cKDTree

KEYFRAME_TRANS = 0.25          # m of travel before adopting a new keyframe
KEYFRAME_ROT = 0.25            # rad
ICP_ITERS = 12
ICP_PAIR_MAX = 0.5             # m: reject NN pairs farther than this
MIN_POINTS = 60                # degenerate-scan gate


def _se2(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def _se2_params(T):
    return float(T[0, 2]), float(T[1, 2]), math.atan2(T[1, 0], T[0, 0])


def _apply(T, pts):
    return pts @ T[:2, :2].T + T[:2, 2]


def _icp(src, kd_dst, dst, T0):
    """Point-to-point 2D ICP of src onto dst (prebuilt cKDTree), seeded with T0."""
    T = T0.copy()
    for _ in range(ICP_ITERS):
        moved = _apply(T, src)
        d, idx = kd_dst.query(moved, k=1)
        # Trimmed inliers: cap the pair gate at 2.5x the median residual so a
        # DYNAMIC object (a person walking past the lidar) contributes outlier
        # pairs that get rejected instead of dragging the pose — the static
        # majority of the room anchors the estimate.
        gate = min(ICP_PAIR_MAX, 2.5 * float(np.median(d)) + 1e-3)
        keep = d < gate
        if keep.sum() < MIN_POINTS // 2:
            return None
        a = moved[keep]
        b = dst[idx[keep]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        H = (a - ca).T @ (b - cb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        t = cb - R @ ca
        dT = np.eye(3)
        dT[:2, :2] = R
        dT[:2, 2] = t
        T = dT @ T
        if abs(t[0]) + abs(t[1]) < 1e-5 and abs(math.atan2(R[1, 0], R[0, 0])) < 1e-5:
            break
    return T


class LaserOdometry(Node):
    def __init__(self):
        super().__init__("laser_odometry")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value

        self._tfbuf = Buffer()
        self._tfl = TransformListener(self._tfbuf, self)
        self._tfb = TransformBroadcaster(self)
        self._pub = self.create_publisher(Odometry, "/odom_laser", 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan,
                                 qos_profile_sensor_data)

        self.T_bl = None               # base_link -> lidar (from URDF, static)
        self.T_ok = np.eye(3)          # odom -> keyframe LIDAR pose
        self.T_kl = np.eye(3)          # keyframe -> current LIDAR pose
        self.key_pts = None
        self.key_kd = None
        self._prev = None              # (t, x, y, th) for twist
        self._scan_frame = None
        self._last_scan_rx = 0.0
        self._hold_warned = False
        # TF KEEP-ALIVE: if the lidar drops mid-session (unplugged / brownout),
        # scans stop and a silent node would let odom->base_link EXPIRE — the
        # whole nav TF tree dies and RViz freezes every display that needs a
        # transform (observed 2026-07-09: an unplugged lidar read as "cameras not
        # showing"). Instead HOLD the last pose at 5 Hz with fresh stamps until
        # scans resume (the bridge auto-reconnects the lidar on replug).
        self.create_timer(0.2, self._keepalive)
        self.get_logger().info("laser odometry: /scan -> odom->base_link + /odom_laser")

    # ---- static lidar mount -------------------------------------------------
    def _lidar_mount(self, frame):
        """base_link -> <scan frame> from TF (URDF static chain), cached."""
        if self.T_bl is not None:
            return self.T_bl
        try:
            tr = self._tfbuf.lookup_transform(self.base_frame, frame,
                                              rclpy.time.Time())
            q = tr.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.T_bl = _se2(tr.transform.translation.x,
                             tr.transform.translation.y, yaw)
            self.get_logger().info(
                f"lidar mount {self.base_frame}->{frame}: "
                f"x={tr.transform.translation.x:.3f} y={tr.transform.translation.y:.3f} "
                f"yaw={yaw:.3f}")
        except Exception:
            return None                # URDF TF not up yet; retry next scan
        return self.T_bl

    def _keepalive(self):
        """Re-broadcast the last odom->base_link with a FRESH stamp while scans
        are absent (>0.5 s), so the TF tree survives a lidar dropout and the
        stack resumes seamlessly when the bridge reconnects the device."""
        import time as _time
        if self.T_bl is None or self._last_scan_rx == 0.0:
            return                       # never had a scan yet: nothing to hold
        if _time.monotonic() - self._last_scan_rx < 0.5:
            return                       # scans flowing: they own the broadcast
        if not self._hold_warned:
            self._hold_warned = True
            self.get_logger().warn(
                "no /scan for >0.5 s (lidar unplugged?) — HOLDING the last "
                "odom->base_link so the TF tree stays alive; resumes on replug.")
        self._broadcast(self.get_clock().now().to_msg())

    # ---- per-scan -----------------------------------------------------------
    def _on_scan(self, msg):
        import time as _time
        self._last_scan_rx = _time.monotonic()
        if self._hold_warned:
            self._hold_warned = False
            self.get_logger().info("/scan resumed — live LiDAR odometry again.")
        T_bl = self._lidar_mount(msg.header.frame_id)
        if T_bl is None:
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        n = len(r)
        th = msg.angle_min + msg.angle_increment * np.arange(n)
        good = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        if good.sum() < MIN_POINTS:
            self._broadcast(msg.header.stamp)      # hold pose; never stale TF
            return
        pts = np.stack([r[good] * np.cos(th[good]),
                        r[good] * np.sin(th[good])], axis=1).astype(np.float64)

        if self.key_pts is None:
            self.key_pts, self.key_kd = pts, cKDTree(pts)
        else:
            T = _icp(pts, self.key_kd, self.key_pts, self.T_kl)
            if T is not None:
                self.T_kl = T
            dx, dy, dth = _se2_params(self.T_kl)
            if math.hypot(dx, dy) > KEYFRAME_TRANS or abs(dth) > KEYFRAME_ROT:
                # promote: fold keyframe->current into odom->keyframe
                self.T_ok = self.T_ok @ self.T_kl
                self.T_kl = np.eye(3)
                self.key_pts, self.key_kd = pts, cKDTree(pts)

        self._broadcast(msg.header.stamp)

    def _broadcast(self, stamp):
        # odom->base_link = odom->lidar ∘ (base->lidar)^-1
        T_ol = self.T_ok @ self.T_kl
        T_ob = T_ol @ np.linalg.inv(self.T_bl)
        x, y, yaw = _se2_params(T_ob)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self._tfb.sendTransform(t)

        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x = x
        o.pose.pose.position.y = y
        o.pose.pose.orientation.z = t.transform.rotation.z
        o.pose.pose.orientation.w = t.transform.rotation.w
        now = stamp.sec + stamp.nanosec * 1e-9
        if self._prev is not None and now > self._prev[0]:
            dt = now - self._prev[0]
            if dt < 1.0:
                c, s = math.cos(-yaw), math.sin(-yaw)   # world delta -> body twist
                wx = (x - self._prev[1]) / dt
                wy = (y - self._prev[2]) / dt
                o.twist.twist.linear.x = c * wx - s * wy
                o.twist.twist.linear.y = s * wx + c * wy
                dyaw = (yaw - self._prev[3] + math.pi) % (2 * math.pi) - math.pi
                o.twist.twist.angular.z = dyaw / dt
        self._prev = (now, x, y, yaw)
        self._pub.publish(o)


def main():
    rclpy.init()
    node = LaserOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
