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
        self._want_clouds = set()  # sources clients asked for — survives a tap
        self._want_grids = set()   # respawn so subscriptions come back
        self._odom_pose = None     # (t 3-vec, R 3x3) from the /odom TOPIC

    def attach(self, node):
        """Called from the telemetry ROS thread each time its node (re)spawns.
        Old subscriptions died with the previous node — drop them and re-create
        every source a client has asked for."""
        self._node = node
        self._subs.clear()
        try:
            import tf2_ros
            self._tf = tf2_ros.Buffer()
            self._listener = tf2_ros.TransformListener(self._tf, node,
                                                       spin_thread=False)
        except Exception:
            self._tf = None
        for src in list(self._want_clouds):
            self.ensure_cloud(src)
        for src in list(self._want_grids):
            self.ensure_grid(src)

    def set_odom_pose(self, px, py, pz, qx, qy, qz, qw):
        """Latest odom→base_link pose from the /odom TOPIC (fed by the tap) —
        the fallback used while SLAM publishes no `odom` TF frame."""
        import numpy as np
        self._odom_pose = (np.array([px, py, pz], dtype=np.float32),
                           np.asarray(_quat_mat(qw, qx, qy, qz),
                                      dtype=np.float32))

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
        if source not in CLOUD_TOPICS:
            return
        self._want_clouds.add(source)      # remembered even while the tap is
        key = "cloud:" + source            # down — attach() rewires it
        if key in self._subs or not self._node:
            return
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data

        def cb(msg):
            blob = self._encode_cloud(msg)
            if blob:
                self._store(key, blob)

        self._subs[key] = self._node.create_subscription(
            PointCloud2, CLOUD_TOPICS[source], cb, qos_profile_sensor_data)

    def ensure_grid(self, source):
        if source not in GRID_TOPICS:
            return
        self._want_grids.add(source)
        key = "grid:" + source
        if key in self._subs or not self._node:
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
            # Carry the sensor's real per-point colour when the cloud has one
            # (packed float rgb slot, 0x00RRGGBB little-endian — same layout
            # the HAL's ros_msgs.pointcloud2_msg writes); gray only as fallback.
            names = {f.name for f in msg.fields}
            cfield = "rgb" if "rgb" in names else (
                "rgba" if "rgba" in names else None)
            fields = ("x", "y", "z") + ((cfield,) if cfield else ())
            arr = point_cloud2.read_points_numpy(
                msg, field_names=fields, skip_nans=False)
            if arr.size == 0:
                return None
            pts = np.ascontiguousarray(arr[:, :3], dtype=np.float32)
            good = np.isfinite(pts).all(axis=1)
            pts = pts[good]
            if pts.size == 0:
                return None
            if cfield:
                packed = np.ascontiguousarray(
                    arr[good, 3], dtype=np.float32).view(np.uint32)
                col = np.empty((len(pts), 4), dtype=np.uint8)
                col[:, 0] = (packed >> 16) & 0xFF
                col[:, 1] = (packed >> 8) & 0xFF
                col[:, 2] = packed & 0xFF
                col[:, 3] = 255
            else:
                col = np.full((len(pts), 4), 200, dtype=np.uint8)
                col[:, 3] = 255
            if len(pts) > MAX_CLOUD_POINTS:
                step = max(1, len(pts) // MAX_CLOUD_POINTS)
                pts, col = pts[::step], col[::step]
            # camera/sensor frame → odom. Same fallback chain as the gateway's
            # CloudFramer (map_stream.py): the dynamic odom→base_link TF is
            # ABSENT while SLAM is down/initialising, and requiring it dropped
            # every frame (silent dead cloud, 2026-07-23). base_link←sensor is
            # the static URDF chain robot_state_publisher always publishes;
            # compose it with the last /odom topic pose when we have one.
            if self._tf is None:
                return None                    # never ship optical-frame points
            from rclpy.time import Time

            def _apply(t, p):
                q, o = t.transform.rotation, t.transform.translation
                R = np.asarray(_quat_mat(q.w, q.x, q.y, q.z), dtype=np.float32)
                return p @ R.T + np.array([o.x, o.y, o.z], dtype=np.float32)

            try:
                pts = _apply(self._tf.lookup_transform(
                    ODOM_FRAME, msg.header.frame_id, Time()), pts)
            except Exception:
                try:
                    pts = _apply(self._tf.lookup_transform(
                        "base_link", msg.header.frame_id, Time()), pts)
                    if self._odom_pose is not None:
                        t, R = self._odom_pose
                        pts = pts @ R.T + t
                except Exception:
                    return None                # no usable transform → skip frame
            n = len(pts)
            buf = bytearray(8 + n * 16)
            struct.pack_into("<BBH", buf, 0, 0x05, 1, 0)
            struct.pack_into("<I", buf, 4, n)
            rec = np.zeros((n, 4), dtype=np.float32)
            rec[:, :3] = pts
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
