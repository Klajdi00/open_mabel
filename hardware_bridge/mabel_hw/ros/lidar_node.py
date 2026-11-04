"""lidar_node — RPLIDAR DDS adapter.

The CAPTURE (serial ownership, motor-stall ladder, brownout detection) lives in
:class:`mabel_hw.subsystems.lidar.LidarSubsystem`; this node only assembles and
publishes LaserScan from its REP-103 revolutions. Contract (byte-identical):

  Published:  <topic>   sensor_msgs/LaserScan  (default /scan, frame lidar_link,
                                                sensor QoS — same contract as the
                                                sim's mujoco_sensors_node)
  Static TF:  lidar -> lidar_link              (identity; sim parity)

Scan assembly: one LaserScan per revolution, binned to a fixed `bins`-wide
uniform grid (min range per bin, inf where nothing landed). The subsystem's
angles are ALREADY REP-103 CCW with the mount yaw folded in.
"""

import math
import queue

import numpy as np

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from ..subsystems.lidar import LidarSubsystem as CoreLidar


def build(cfg, hw=None, **_):
    return LidarNode(cfg, hw=hw)


class LidarNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_lidar")
        self.cfg = cfg

        if hw is not None and hw._lidar is not None:
            self.core = hw._lidar
            self._owns_core = False
        else:
            self.core = CoreLidar(cfg,
                                  logger=lambda m: self.get_logger().warn(m))
            self._owns_core = True

        self.pub = self.create_publisher(LaserScan, cfg["topic"],
                                         qos_profile_sensor_data)

        if cfg.get("urdf_frame"):
            from geometry_msgs.msg import TransformStamped
            from tf2_ros import StaticTransformBroadcaster
            self._tf = StaticTransformBroadcaster(self)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = cfg["urdf_frame"]
            t.child_frame_id = cfg["frame"]
            t.transform.rotation.w = 1.0
            self._tf.sendTransform([t])

        # Capture pushes revolutions from ITS thread; publish from the executor
        # thread via a 1-deep latest-wins queue (rclpy publishers are not
        # guaranteed thread-safe across contexts).
        self._q = queue.Queue(maxsize=1)
        self._unsub = self.core.on_scan(self._enqueue)
        self.create_timer(0.02, self._drain)

        self.get_logger().info(
            f"lidar online: RPLIDAR on {cfg['port']} -> {cfg['topic']} "
            f"(frame {cfg['frame']}, {cfg['bins']} bins).")

    def _enqueue(self, scan):
        try:
            self._q.put_nowait(scan)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(scan)
            except queue.Full:
                pass

    def _drain(self):
        try:
            scan = self._q.get_nowait()
        except queue.Empty:
            return
        self._publish(scan)

    # ---- LaserScan assembly ----------------------------------------------------
    def _publish(self, scan):
        c = self.cfg
        n = int(c["bins"])
        theta = np.asarray(scan.angles)          # REP-103 CCW, yaw folded in
        dists = np.asarray(scan.ranges)
        qual = np.asarray(scan.quality)
        idx = np.minimum((theta / (2.0 * math.pi) * n).astype(int), n - 1)
        ranges = np.full(n, np.inf, dtype=np.float32)
        intens = np.zeros(n, dtype=np.float32)
        valid = (dists >= float(c["range_min"])) & (dists <= float(c["range_max"]))
        # min range per bin (conservative for obstacles); numpy minimum-at is atomic.
        np.minimum.at(ranges, idx[valid], dists[valid].astype(np.float32))
        np.maximum.at(intens, idx[valid], qual[valid].astype(np.float32))
        scan_time = float(getattr(scan, "_scan_time", 0.0))

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = c["frame"]
        msg.angle_min = 0.0
        msg.angle_increment = 2.0 * math.pi / n
        msg.angle_max = msg.angle_increment * (n - 1)
        msg.scan_time = scan_time
        msg.time_increment = scan_time / n
        msg.range_min = float(c["range_min"])
        msg.range_max = float(c["range_max"])
        msg.ranges = ranges.tolist()
        msg.intensities = intens.tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self._unsub()
            if self._owns_core:
                self.core.close()
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
LidarSubsystem = LidarNode
