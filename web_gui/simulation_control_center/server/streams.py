"""
Selectable sensor streams for the 3D viewer, beyond the bridge's single map
plane: per-camera depth CLOUDS (head / front base, TF-transformed into odom)
and the 2D occupancy GRID from either SLAM stack (lidar slam_toolbox /map or
the visual/nvblox distance-map slice). Encoded with the SAME binary records
as server/docs/data_streams.md (0x04 GRID / 0x05 CLOUD) so the web decoder is
shared. Also: reset_slam() — calls every known SLAM reset/clear service.
"""
import math
import os
import struct
import threading
import time

from . import config as CFG

CLOUD_TOPICS = {
    "head": os.environ.get("MABEL_CC_HEAD_CLOUD", "/camera/zed/depth/points"),
    "front": os.environ.get("MABEL_CC_FRONT_CLOUD", "/camera/front/depth/points"),
}

# Pure-sim (Mac, no ROS) per-camera depth clouds. Each selectable source maps to
# a MuJoCo camera in the scene; the SimMirror renders depth+RGB for the sources a
# /ws/stream client currently wants and stores them here as cloud:<source>, so
# the SAME frontend per-camera toggles work with or without ROS. (ROS path above
# takes priority when a live node exists.)
SIM_CLOUD_CAMERAS = {
    "head": "zed_left",
    "front": "body_front_cam",
    "wrist_left": "l_wrist_cam",
    "wrist_right": "r_wrist_cam",
}
SIM_DEMAND_TTL = 1.5           # a source is "wanted" for this long after ensure_cloud
GRID_TOPICS = {
    "lidar": "/map",
    "visual": "/nvblox_node/static_map_slice",
}
MAX_CLOUD_POINTS = 60000
ODOM_FRAME = "odom"


class StreamHub:
    """Latest encoded record per source + a revision counter, fed by rclpy
    subscriptions created on demand (created only when a client asks)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.records = {}          # key -> (rev, bytes)
        self._rev = 0
        self._subs = {}
        self._node = None
        self._tf = None
        self._sim_demand = {}      # sim source -> last-wanted monotonic time

    def attach(self, node):
        """Called from the telemetry ROS thread once the node exists."""
        self._node = node
        try:
            import tf2_ros
            self._tf = tf2_ros.Buffer()
            self._listener = tf2_ros.TransformListener(self._tf, node,
                                                       spin_thread=False)
        except Exception:
            self._tf = None

    def _store(self, key, blob):
        with self._lock:
            self._rev += 1
            self.records[key] = (self._rev, blob)

    def latest(self, keys, after_rev):
        """Records newer than after_rev among keys → (new_rev, [(key, bytes)])."""
        out = []
        rev = after_rev
        with self._lock:
            for k in keys:
                r = self.records.get(k)
                if r and r[0] > after_rev:
                    out.append((k, r[1]))
                    rev = max(rev, r[0])
        return rev, out

    # ── subscriptions (idempotent) ──────────────────────────────────────────
    def ensure_cloud(self, source):
        # Pure-sim source: no ROS sub — just refresh demand so the SimMirror
        # renders it. Called every /ws/stream poll (~0.15 s), so demand stays
        # fresh while a socket is open and goes stale (→ render stops) on close.
        if source in SIM_CLOUD_CAMERAS and not (
                self._node and source in CLOUD_TOPICS):
            self._sim_demand[source] = time.time()
            return
        key = "cloud:" + source
        if key in self._subs or not self._node or source not in CLOUD_TOPICS:
            return
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data

        def cb(msg):
            blob = self._encode_cloud(msg)
            if blob:
                self._store(key, blob)

        self._subs[key] = self._node.create_subscription(
            PointCloud2, CLOUD_TOPICS[source], cb, qos_profile_sensor_data)

    # ── pure-sim producer hooks (fed by SimMirror) ──────────────────────────
    def sim_wanted_now(self, ttl=SIM_DEMAND_TTL):
        """Sim cloud sources a /ws/stream client has asked for within `ttl`."""
        now = time.time()
        return [s for s, t in list(self._sim_demand.items()) if now - t <= ttl]

    def sim_store(self, source, blob):
        self._store("cloud:" + source, blob)

    def ensure_grid(self, source):
        key = "grid:" + source
        if key in self._subs or not self._node or source not in GRID_TOPICS:
            return
        if source == "lidar":
            from nav_msgs.msg import OccupancyGrid

            def cb(msg):
                self._store(key, encode_grid(
                    msg.info.width, msg.info.height, msg.info.resolution,
                    msg.info.origin.position.x, msg.info.origin.position.y,
                    msg.info.origin.position.z,
                    _quat_yaw(msg.info.origin.orientation),
                    bytes((v if 0 <= v <= 100 else 255) & 0xFF
                          for v in msg.data)))

            self._subs[key] = self._node.create_subscription(
                OccupancyGrid, GRID_TOPICS[source], cb, 5)
        else:
            try:
                from nvblox_msgs.msg import DistanceMapSlice
            except Exception:
                return

            def cb(msg):
                # distance slice → occupancy: near-zero ESDF distance = occupied
                cells = bytearray(len(msg.data))
                unk = msg.unknown_value
                for i, d in enumerate(msg.data):
                    if d == unk:
                        cells[i] = 255
                    else:
                        cells[i] = 100 if d < 0.15 else 0
                self._store(key, encode_grid(
                    msg.width, msg.height, msg.resolution,
                    msg.origin.x, msg.origin.y, 0.0, 0.0, bytes(cells)))

            self._subs[key] = self._node.create_subscription(
                DistanceMapSlice, GRID_TOPICS[source], cb, 5)

    def _encode_cloud(self, msg):
        """PointCloud2 → 0x05 record in the odom frame (TF-transformed)."""
        try:
            import numpy as np
            from sensor_msgs_py import point_cloud2
            pts = point_cloud2.read_points_numpy(
                msg, field_names=("x", "y", "z"), skip_nans=True)
            if pts.size == 0:
                return None
            if len(pts) > MAX_CLOUD_POINTS:
                pts = pts[:: max(1, len(pts) // MAX_CLOUD_POINTS)]
            # camera/sensor frame → odom
            if self._tf is not None:
                try:
                    t = self._tf.lookup_transform(
                        ODOM_FRAME, msg.header.frame_id, __import__(
                            "rclpy").time.Time())
                    q = t.transform.rotation
                    R = _quat_mat(q.w, q.x, q.y, q.z)
                    o = t.transform.translation
                    pts = pts @ np.asarray(R, dtype=np.float32).T
                    pts = pts + np.array([o.x, o.y, o.z], dtype=np.float32)
                except Exception:
                    return None                # no TF yet → skip frame
            n = len(pts)
            buf = bytearray(8 + n * 16)
            struct.pack_into("<BBH", buf, 0, 0x05, 1, 0)
            struct.pack_into("<I", buf, 4, n)
            rec = np.zeros((n, 4), dtype=np.float32)
            rec[:, :3] = pts
            # constant light-gray color, alpha 255
            col = np.full((n, 4), 200, dtype=np.uint8)
            col[:, 3] = 255
            body = np.frombuffer(rec.tobytes(), dtype=np.uint8).reshape(n, 16).copy()
            body[:, 12:16] = col
            buf[8:] = body.tobytes()
            return bytes(buf)
        except Exception:
            return None


def _quat_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def _quat_mat(w, x, y, z):
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def encode_grid(w, h, res, ox, oy, oz, yaw, cells):
    """0x04 GRID record (server/docs/data_streams.md layout)."""
    buf = bytearray(32 + len(cells))
    struct.pack_into("<BBH", buf, 0, 0x04, 0, 0)
    struct.pack_into("<II", buf, 4, w, h)
    struct.pack_into("<fffff", buf, 12, res, ox, oy, oz, yaw)
    buf[32:] = cells
    return bytes(buf)


HUB = StreamHub()


def pause_slam(timeout=2.0):
    """Pause slam_toolbox measurement intake (best-effort; nvblox has no
    pause — the GUI freezes its view instead)."""
    import rclpy
    from rclpy.node import Node

    from rclpy.executors import SingleThreadedExecutor
    node = Node("mabel_cc_slam_pause")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        try:
            from slam_toolbox.srv import Pause
        except Exception:
            return {"/slam_toolbox/pause_new_measurements": False}
        cli = node.create_client(Pause, "/slam_toolbox/pause_new_measurements")
        if not cli.wait_for_service(timeout_sec=1.0):
            return {"/slam_toolbox/pause_new_measurements": False}
        fut = cli.call_async(Pause.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            ex.spin_once(timeout_sec=0.1)
        return {"/slam_toolbox/pause_new_measurements":
                fut.done() and fut.exception() is None}
    finally:
        node.destroy_node()


def reset_mapping(clear_s=3.0):
    """Clear the MAP (nvblox 3D mesh → which also feeds the 2D /map slice)
    while KEEPING localization: cuVSLAM is never touched. Uses nvblox's
    radius-clearing parameters as a transient wipe — set a tiny keep-radius +
    a fast clear rate, let a few ticks deallocate every block, then restore.
    slam_toolbox (lidar mode) gets its reset service when present."""
    import rclpy
    from rclpy.node import Node
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

    from rclpy.executors import SingleThreadedExecutor
    results = {}
    node = Node("mabel_cc_map_reset")
    ex = SingleThreadedExecutor()
    ex.add_node(node)

    def set_params(pairs, timeout=8.0):
        cli = node.create_client(SetParameters, "/nvblox_node/set_parameters")
        if not cli.wait_for_service(timeout_sec=4.0):
            node.destroy_client(cli)
            return False
        req = SetParameters.Request()
        for name, val in pairs:
            pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                double_value=float(val))
            req.parameters.append(Parameter(name=name, value=pv))
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            ex.spin_once(timeout_sec=0.1)
        ok = fut.done() and fut.result() is not None and \
            all(r.successful for r in fut.result().results)
        node.destroy_client(cli)
        return ok

    try:
        # 1. nvblox transient wipe (keeps a 0.5 m bubble around the robot)
        armed = set_params([("map_clearing_radius_m", 0.5),
                            ("clear_map_outside_radius_rate_hz", 4.0)])
        if armed:
            t0 = time.time()
            while time.time() - t0 < clear_s:
                ex.spin_once(timeout_sec=0.2)
        restored = set_params([("clear_map_outside_radius_rate_hz", 0.0),
                               ("map_clearing_radius_m", -1.0)])
        results["/nvblox_node (radius wipe)"] = bool(armed and restored)

        # 2. slam_toolbox reset (only in lidar-SLAM mode)
        try:
            from slam_toolbox.srv import Reset as StReset
            cli = node.create_client(StReset, "/slam_toolbox/reset")
            if cli.wait_for_service(timeout_sec=1.0):
                fut = cli.call_async(StReset.Request())
                t0 = time.time()
                while not fut.done() and time.time() - t0 < 5.0:
                    ex.spin_once(timeout_sec=0.1)
                results["/slam_toolbox/reset"] = fut.done() and fut.exception() is None
            node.destroy_client(cli)
        except Exception:
            pass
    finally:
        ex.remove_node(node)
        node.destroy_node()
    return results


def reset_slam(timeout=3.0):
    """Fire every known SLAM reset/clear service; returns {service: ok}."""
    import rclpy
    from rclpy.node import Node

    from rclpy.executors import SingleThreadedExecutor
    results = {}
    node = Node("mabel_cc_slam_reset")
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        candidates = []
        # cuVSLAM (this stack's live SLAM): re-anchors the odom TF — the fix
        # for accumulated pose drift. Proper Isaac srv type, NOT std_srvs.
        try:
            from isaac_ros_visual_slam_interfaces.srv import Reset as VsReset
            candidates.append(("/visual_slam/reset", VsReset, VsReset.Request()))
        except Exception:
            pass
        try:
            from slam_toolbox.srv import Reset as StReset
            candidates.append(("/slam_toolbox/reset", StReset, StReset.Request()))
        except Exception:
            pass
        try:
            from std_srvs.srv import Empty
            # only some nvblox builds expose a clear service — harmless probe
            candidates.append(("/nvblox_node/clear_map", Empty, Empty.Request()))
        except Exception:
            pass

        for name, srv, req in candidates:
            cli = node.create_client(srv, name)
            if not cli.wait_for_service(timeout_sec=2.0):
                results[name] = False
                node.destroy_client(cli)
                continue
            fut = cli.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < max(timeout, 8.0):
                ex.spin_once(timeout_sec=0.1)
            results[name] = fut.done() and fut.exception() is None
            node.destroy_client(cli)
    finally:
        node.destroy_node()
    return results
