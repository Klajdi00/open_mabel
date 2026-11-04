#!/usr/bin/env python3
"""bag_to_rerun.py — convert a MABEL rosbag2 (MCAP) session into a Rerun recording.

The recorder (`data_collection.py`) writes the canonical dataset as an MCAP bag;
this tool replays it into Rerun so the WHOLE session — the tf tree, every camera
+ depth + point cloud, the nvblox mesh, the 2-D occupancy map, the LiDAR scan,
the planned path, and all joint / command / IMU time-series — is visualized in
one coherent 3-D scene with synchronized timelines.

    # one-time:  pip install rerun-sdk  (already in the `mabel` conda env)
    python3 bag_to_rerun.py recordings/20260624_180000_session            # → <bag>/rerun.rrd
    python3 bag_to_rerun.py recordings/2026...session --spawn             # open the viewer live
    rerun recordings/20260624_180000_session/rerun.rrd                    # view a saved .rrd

Run it in the SAME environment as the recording (the `mabel` conda env), so that
ros message typesupport is importable for deserialization. Message types whose
Python bindings are not installed (e.g. nvblox_msgs/Mesh) are skipped with a
count; everything else converts. The nvblox mesh still comes through via its
`mesh_marker` (triangle list) and `mesh_pointcloud` topics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    sys.exit("rerun not found — `pip install rerun-sdk==0.26.2` in this env (the `mabel` conda env).")

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# ── tf tree → Rerun entity paths ─────────────────────────────────────────────
class TfTree:
    """Maps ROS frame_ids to nested Rerun entity paths so all data is spatially
    coherent. Rerun composes transforms down the entity path, and tf gives each
    child's pose in its parent — so logging Transform3D at `<parent_path>/<child>`
    reproduces the kinematic tree exactly."""

    def __init__(self, root: str = "world"):
        self.root = root
        self.parent: dict[str, str] = {}
        self._path_cache: dict[str, str] = {}

    def set_parent(self, child: str, parent: str) -> None:
        if self.parent.get(child) != parent:
            self.parent[child] = parent
            self._path_cache.clear()   # tree changed → invalidate

    def path(self, frame: str) -> str:
        frame = frame.lstrip("/")
        if frame in self._path_cache:
            return self._path_cache[frame]
        chain, seen, f = [], set(), frame
        while f and f not in seen:
            seen.add(f)
            chain.append(f)
            f = self.parent.get(f)
        p = self.root + "/" + "/".join(reversed(chain))
        self._path_cache[frame] = p
        return p


# ── small helpers ────────────────────────────────────────────────────────────
def safe(part: str) -> str:
    """Rerun entity-path parts can't contain whitespace; joint names ('Left Arm 1') do."""
    return "_".join(str(part).split())


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                                  # URDF rpy = Rz·Ry·Rx


def load_urdf_visuals(urdf_path: Path):
    """[(link_name, vertices Nx3 in link frame, faces Mx3)] from a URDF's <visual>
    meshes. Vertices are pre-transformed by the visual origin + scale so each can be
    logged at its link's entity path and posed by tf. Needs trimesh."""
    import xml.etree.ElementTree as ET
    import trimesh
    pkg_root = urdf_path.parent.parent          # …/mabel_description/urdf/x.urdf → …/mabel_description
    root = ET.parse(urdf_path).getroot()
    out, errs = [], 0
    for link in root.findall("link"):
        vis = link.find("visual")
        geo = vis.find("geometry") if vis is not None else None
        mesh = geo.find("mesh") if geo is not None else None
        if mesh is None:
            continue
        fn = mesh.get("filename", "")
        fn = fn.replace("package://mabel_description/", str(pkg_root) + "/")
        try:
            m = trimesh.load(fn, force="mesh", process=False)
            v = np.asarray(m.vertices, dtype=np.float64)
            f = np.asarray(m.faces, dtype=np.uint32)
        except Exception:
            errs += 1
            continue
        scale = mesh.get("scale")
        if scale:
            v = v * np.array([float(x) for x in scale.split()])
        org = vis.find("origin")
        if org is not None:
            xyz = np.array([float(x) for x in (org.get("xyz") or "0 0 0").split()])
            rpy = [float(x) for x in (org.get("rpy") or "0 0 0").split()]
            v = v @ _rpy_matrix(*rpy).T + xyz
        out.append((link.get("name"), v.astype(np.float32), f))
    return out, errs


def log_robot_meshes(tf: TfTree, urdf_path: Path, color=(184, 196, 208)) -> int:
    """Log each URDF link's visual mesh (static) at its tf entity path, so the robot
    BODY appears in the 3-D view, posed live by the recorded transforms. Returns the
    number of links drawn (0 if trimesh is missing or the URDF can't be read)."""
    try:
        visuals, errs = load_urdf_visuals(urdf_path)
    except ImportError:
        print("  (robot body skipped: `pip install trimesh` to draw it)")
        return 0
    except (OSError, SyntaxError) as e:
        print(f"  (robot body skipped: {e})")
        return 0
    for name, verts, faces in visuals:
        rr.log(tf.path(name) + "/visual",
               rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                         albedo_factor=color), static=True)
    if errs:
        print(f"  ({errs} link meshes failed to load)")
    return len(visuals)


def stamp_to_sec(stamp) -> float | None:
    try:
        return stamp.sec + stamp.nanosec * 1e-9
    except AttributeError:
        return None


_PCL_DTYPE = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
              5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pointcloud2_to_xyzrgb(msg, max_points: int = 400_000, max_range: float = 0.0):
    """(Nx3 float32 xyz, Nx3 uint8 rgb-or-None) from a PointCloud2, dropping NaNs.

    `max_range` (m, 0 = off): drop points farther than this from the sensor. Depth
    cameras spit out a handful of far-range fliers (here ~60/frame out to ~18 m while
    95%% of points are inside ~4 m), and their COUNT jitters frame-to-frame, so the
    scene's bounding box pulses and the 3-D view's auto-camera keeps refitting — which
    looks like the cloud shaking/vanishing. Clipping the fliers stabilizes the box."""
    fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
    if not all(k in fields for k in ("x", "y", "z")):
        return None, None
    n = msg.width * msg.height
    if n == 0:
        return None, None
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)

    def col(name):
        off, dt = fields[name]
        np_dt = _PCL_DTYPE.get(dt, np.float32)
        width = np.dtype(np_dt).itemsize
        return raw[:, off:off + width].copy().view(np_dt).reshape(-1)

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    if max_range and max_range > 0:
        keep &= (xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2) <= max_range * max_range
    xyz = xyz[keep]

    rgb = None
    rgb_field = "rgb" if "rgb" in fields else ("rgba" if "rgba" in fields else None)
    if rgb_field is not None:
        off, _ = fields[rgb_field]
        packed = raw[:, off:off + 4][keep]          # bytes: B,G,R,(A) little-endian
        rgb = packed[:, [2, 1, 0]].astype(np.uint8)  # → R,G,B

    if xyz.shape[0] > max_points:                    # uniform subsample huge clouds
        idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(int)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
    return xyz, rgb


def image_to_array(msg):
    """(numpy array, is_depth, meter). Handles the encodings MABEL publishes."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("32FC1",):
        arr = buf.view(np.float32).reshape(h, msg.step // 4)[:, :w]
        return arr.copy(), True, 1.0                 # metres
    if enc in ("16UC1", "mono16"):
        arr = buf.view(np.uint16).reshape(h, msg.step // 2)[:, :w]
        return arr.copy(), True, 1000.0              # millimetres → 1000 units / m
    if enc == "mono8":
        return buf.reshape(h, msg.step)[:, :w].copy(), False, None
    ch = 4 if enc in ("rgba8", "bgra8") else 3
    arr = buf.reshape(h, msg.step)[:, :w * ch].reshape(h, w, ch)
    if enc in ("bgr8", "bgra8"):
        arr = arr[:, :, [2, 1, 0] + ([3] if ch == 4 else [])]
    return arr.copy(), False, None


def numeric_fields(msg, prefix=""):
    """Flatten scalar float/int fields of a small message for time-series logging."""
    out = {}
    for slot in getattr(msg, "__slots__", []):
        name = slot.lstrip("_")
        val = getattr(msg, slot)
        if isinstance(val, bool):
            out[prefix + name] = float(val)
        elif isinstance(val, (int, float)):
            out[prefix + name] = float(val)
    return out


# ── per-message-type handlers ────────────────────────────────────────────────
class Converter:
    def __init__(self, tf: TfTree, frustum: float = 0.25, log_depth: bool = True,
                 point_radius: float = -1.5, max_range: float = 8.0):
        self.tf = tf
        self.frustum = frustum          # fixed camera image-plane distance (m)
        self.log_depth = log_depth      # whether to log DepthImage (+ its back-projection)
        self.max_range = max_range      # clip cloud points beyond this many m (stabilizes bbox)
        # NEGATIVE = UI points (screen-space): clouds stay visible at any zoom. A small
        # world-space radius (e.g. 0.01 m) goes sub-pixel when the view fits the whole
        # scene, which makes the cloud look "missing".
        self.point_radius = point_radius
        self.seen_pinhole: set[str] = set()
        self.cam_paths: set[str] = set()
        self.skipped: dict[str, int] = {}

    def handle(self, topic, type_str, msg, bag_t):
        short = type_str.split("/")[-1]
        fn = getattr(self, "_" + short, None)
        if fn is None:
            self.skipped[type_str] = self.skipped.get(type_str, 0) + 1
            return
        # Per-message time: prefer the header stamp, else the bag receive time.
        t = None
        hdr = getattr(msg, "header", None)
        if hdr is not None:
            t = stamp_to_sec(hdr.stamp)
        rr.set_time("ros_time", timestamp=(t if t else bag_t))
        fn(topic, msg)

    # transforms ----------------------------------------------------------------
    def _TFMessage(self, topic, msg):
        is_static = "static" in topic                  # /tf_static → time-independent
        for tr in msg.transforms:
            child = tr.child_frame_id.lstrip("/")
            parent = tr.header.frame_id.lstrip("/")
            self.tf.set_parent(child, parent)
            tt = tr.transform.translation
            q = tr.transform.rotation
            if not is_static:
                t = stamp_to_sec(tr.header.stamp)
                if t:
                    rr.set_time("ros_time", timestamp=t)
            rr.log(self.tf.path(child),
                   rr.Transform3D(translation=[tt.x, tt.y, tt.z],
                                  quaternion=[q.x, q.y, q.z, q.w]),
                   static=is_static)

    # cameras -------------------------------------------------------------------
    def _CameraInfo(self, topic, msg):
        path = self.tf.path(msg.header.frame_id)
        if path in self.seen_pinhole:
            return
        self.seen_pinhole.add(path)
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        # A FIXED image_plane_distance stops the frustum from auto-rescaling (and
        # visibly pulsing) as the 3-D scene's bounding box changes each frame.
        rr.log(path, rr.Pinhole(image_from_camera=K, resolution=[msg.width, msg.height],
                                camera_xyz=rr.ViewCoordinates.RDF,
                                image_plane_distance=self.frustum), static=True)

    def _Image(self, topic, msg):
        path = self.tf.path(msg.header.frame_id)
        self.cam_paths.add(path)
        try:
            arr, is_depth, meter = image_to_array(msg)
        except (ValueError, AttributeError):
            return
        if is_depth:
            if self.log_depth:
                rr.log(path + "/depth", rr.DepthImage(arr, meter=meter))
        else:
            rr.log(path + "/image", rr.Image(arr))

    # geometry / clouds ---------------------------------------------------------
    def _PointCloud2(self, topic, msg):
        # don't range-clip the nvblox mesh cloud (it's the whole map, legitimately large)
        mr = 0.0 if "mesh" in topic else self.max_range
        xyz, rgb = pointcloud2_to_xyzrgb(msg, max_range=mr)
        if xyz is None or xyz.shape[0] == 0:
            return
        leaf = "mesh" if "mesh" in topic else "points"
        rr.log(self.tf.path(msg.header.frame_id) + f"/{leaf}",
               rr.Points3D(xyz, colors=rgb, radii=self.point_radius))

    def _LaserScan(self, topic, msg):
        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(r.shape[0], dtype=np.float32) * msg.angle_increment
        good = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        pts = np.stack([r[good] * np.cos(ang[good]),
                        r[good] * np.sin(ang[good]),
                        np.zeros(int(good.sum()), np.float32)], axis=1)
        rr.log(self.tf.path(msg.header.frame_id) + "/scan",
               rr.Points3D(pts, colors=[255, 80, 80], radii=self.point_radius))

    def _Marker(self, topic, msg):
        # nvblox mesh_marker → TRIANGLE_LIST (msg.type == 11): vertices in triples.
        if not msg.points:
            return
        v = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        cols = None
        if msg.colors and len(msg.colors) == len(msg.points):
            cols = np.array([[c.r, c.g, c.b] for c in msg.colors], dtype=np.float32)
            cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        frame = msg.header.frame_id or "map"
        rr.log(self.tf.path(frame) + "/nvblox_mesh",
               rr.Mesh3D(vertex_positions=v, vertex_colors=cols))

    def _OccupancyGrid(self, topic, msg):
        name = topic.strip("/").replace("/", "_")
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)   # int16: 100*255 > int8
        # 2-D image view: unknown(-1)→128 grey, free(0)→255, occupied(100)→0.
        img = np.where(grid < 0, 128, (255 - grid * 255 // 100)).astype(np.uint8)
        rr.log(f"maps2d/{name}", rr.Image(img))
        # 3-D overlay: occupied cells as points on the floor, in the map frame.
        res, ox, oy = msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y
        rows, cols = np.where(grid >= 50)
        if rows.size:
            pts = np.stack([ox + (cols + 0.5) * res,
                            oy + (rows + 0.5) * res,
                            np.zeros(rows.size, np.float32)], axis=1).astype(np.float32)
            rr.log(self.tf.path(msg.header.frame_id) + f"/{name}",
                   rr.Points3D(pts, colors=[60, 60, 60], radii=res * 0.5))

    def _Path(self, topic, msg):
        if not msg.poses:
            return
        pts = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in msg.poses], dtype=np.float32)
        frame = msg.header.frame_id or "map"
        base = self.tf.path(frame)
        rr.log(base + "/plan", rr.LineStrips3D([pts], colors=[40, 200, 90], radii=0.02))

    def _PoseStamped(self, topic, msg):
        p, o = msg.pose.position, msg.pose.orientation
        rr.log(self.tf.path(msg.header.frame_id) + "/goal",
               rr.Transform3D(translation=[p.x, p.y, p.z],
                              quaternion=[o.x, o.y, o.z, o.w], axis_length=0.4))

    # proprioception / commands / status ---------------------------------------
    def _JointState(self, topic, msg):
        group = topic.strip("/").replace("/", "_") or "joint_states"
        for name, pos in zip(msg.name, msg.position):
            rr.log(f"joints/{group}/{safe(name)}", rr.Scalars(float(pos)))

    def _Imu(self, topic, msg):
        a, g = msg.linear_acceleration, msg.angular_velocity
        for k, v in {"accel/x": a.x, "accel/y": a.y, "accel/z": a.z,
                     "gyro/x": g.x, "gyro/y": g.y, "gyro/z": g.z}.items():
            rr.log(f"imu/{k}", rr.Scalars(float(v)))

    def _Odometry(self, topic, msg):
        base = topic.strip("/").replace("/", "_")
        t = msg.twist.twist
        for k, v in {"vx": t.linear.x, "vy": t.linear.y, "wz": t.angular.z}.items():
            rr.log(f"odom/{base}/{k}", rr.Scalars(float(v)))

    def _Twist(self, topic, msg):
        base = topic.strip("/").replace("/", "_")
        for k, v in {"vx": msg.linear.x, "vy": msg.linear.y, "wz": msg.angular.z}.items():
            rr.log(f"cmd/{base}/{k}", rr.Scalars(float(v)))

    def _String(self, topic, msg):
        rr.log(topic.strip("/").replace("/", "_"), rr.TextLog(msg.data))

    def _Bool(self, topic, msg):
        rr.log(topic.strip("/").replace("/", "_"), rr.Scalars(float(msg.data)))

    def _MabelSwerveCmd(self, topic, msg):
        for k, v in numeric_fields(msg).items():
            rr.log(f"mabel_cmd/{k}", rr.Scalars(v))


# ── blueprint: a useful default layout ───────────────────────────────────────
def build_blueprint(cam_paths: list[str]):
    cam_views = [rrb.Spatial2DView(origin=p, name=p.split("/")[-2] if "/" in p else p)
                 for p in sorted(cam_paths)]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D scene"),
            rrb.Vertical(
                rrb.Tabs(*cam_views, name="cameras") if cam_views else rrb.TextLogView(origin="mabel_control_mode"),
                rrb.Tabs(
                    rrb.TimeSeriesView(origin="joints", name="joints"),
                    rrb.TimeSeriesView(origin="cmd", name="base cmd"),
                    rrb.TimeSeriesView(origin="mabel_cmd", name="swerve cmd"),
                    rrb.TimeSeriesView(origin="imu", name="imu"),
                    rrb.Spatial2DView(origin="maps2d", name="2D map"),
                ),
                row_shares=[2, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="collapsed"),
    )


def open_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    uri = str(bag_path)
    if bag_path.is_file() and bag_path.suffix == ".mcap":
        uri = str(bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    return reader


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a MABEL rosbag2 (MCAP) session to Rerun.")
    ap.add_argument("bag", type=Path, help="recording dir (contains metadata.yaml) or a .mcap file")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output .rrd (default <bag>/rerun.rrd)")
    ap.add_argument("--spawn", action="store_true", help="open the Rerun viewer live instead of saving")
    ap.add_argument("--max-msgs", type=int, default=0, help="stop after N messages (debug)")
    ap.add_argument("--exclude", nargs="*", default=[], help="topic substrings to skip")
    ap.add_argument("--urdf", type=Path, default=None,
                    help="URDF for the robot-body overlay (default: repo mabel_full.urdf)")
    ap.add_argument("--no-robot", action="store_true", help="don't draw the robot body meshes")
    ap.add_argument("--frustum", type=float, default=0.25,
                    help="camera frustum image-plane distance in metres (fixed; stops pulsing)")
    ap.add_argument("--no-depth", action="store_true",
                    help="don't log depth images (keeps the explicit point clouds)")
    ap.add_argument("--point-size", type=float, default=1.5,
                    help="cloud/scan point size in UI points (screen-space; visible at any zoom)")
    ap.add_argument("--max-range", type=float, default=8.0,
                    help="clip depth-cloud points beyond this many m (0 = keep all); "
                         "drops jittering far fliers that destabilize the auto-camera")
    args = ap.parse_args()

    bag = args.bag
    if not bag.exists():
        sys.exit(f"no such bag: {bag}")

    type_of = {t.name: t.type for t in open_reader(bag).get_all_topics_and_types()}

    # PASS 1 — build the COMPLETE tf tree before logging anything. Entity paths are
    # derived from the tf hierarchy; if we discovered it incrementally, a frame seen
    # before its full parent chain (e.g. /tf_static at t=0, before odom→base_link)
    # would be logged at a SHORT path that never matches the final one, leaving that
    # subtree (and the robot meshes) stranded at the origin. One cheap tf-only sweep
    # fixes every path up front.
    tf = TfTree("world")
    first_tf: dict[str, tuple] = {}     # child → (translation, quaternion) first seen
    pre = open_reader(bag)
    while pre.has_next():
        topic, data, _ = pre.read_next()
        if topic in ("/tf", "/tf_static"):
            try:
                m = deserialize_message(data, get_message(type_of[topic]))
                for tr in m.transforms:
                    child = tr.child_frame_id.lstrip("/")
                    tf.set_parent(child, tr.header.frame_id.lstrip("/"))
                    if child not in first_tf:
                        t, q = tr.transform.translation, tr.transform.rotation
                        first_tf[child] = ([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            except Exception:
                pass
    print(f"tf tree: {len(tf.parent)} frames resolved")

    rr.init("mabel_dataset", spawn=args.spawn)
    if not args.spawn:
        out = args.output or (bag if bag.is_dir() else bag.parent) / "rerun.rrd"
        rr.save(str(out))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Seed every frame with its first transform STATICALLY, so the robot is fully
    # posed from the very first timeline instant. Without this, a dynamic frame has
    # no transform until its first /tf message arrives, so at t=0 it snaps from the
    # origin into place — the "initial shake". Temporal /tf updates override this.
    for child, (trans, quat) in first_tf.items():
        rr.log(tf.path(child), rr.Transform3D(translation=trans, quaternion=quat), static=True)

    # PASS 2 — stream everything; tf.path() is now stable.
    reader = open_reader(bag)
    conv = Converter(tf, frustum=args.frustum, log_depth=not args.no_depth,
                     point_radius=-abs(args.point_size), max_range=args.max_range)
    n = 0
    msg_classes: dict[str, object] = {}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if any(x in topic for x in args.exclude):
            continue
        type_str = type_of.get(topic)
        if not type_str:
            continue
        cls = msg_classes.get(type_str)
        if cls is None:
            try:
                cls = get_message(type_str)
            except (ModuleNotFoundError, ValueError, AttributeError):
                cls = False
            msg_classes[type_str] = cls
        if cls is False:
            conv.skipped[type_str] = conv.skipped.get(type_str, 0) + 1
            continue
        try:
            msg = deserialize_message(data, cls)
            conv.handle(topic, type_str, msg, t_ns * 1e-9)
        except Exception as e:                       # never let one bad msg abort the run
            conv.skipped[type_str + f" (err: {type(e).__name__})"] = \
                conv.skipped.get(type_str, 0) + 1
        n += 1
        if n % 5000 == 0:
            print(f"  …{n} messages")
        if args.max_msgs and n >= args.max_msgs:
            break

    # Robot body overlay: log each URDF link's mesh at its (now-known) tf path, so
    # the robot appears in the 3-D view, posed by the recorded transforms.
    if not args.no_robot:
        urdf = args.urdf or (Path(__file__).resolve().parent.parent
                             / "ros/src/mabel_description/urdf/mabel_full.urdf")
        if urdf.exists():
            drew = log_robot_meshes(tf, urdf)
            if drew:
                print(f"robot body: drew {drew} link meshes from {urdf.name}")
        else:
            print(f"  (robot body skipped: no URDF at {urdf})")

    rr.send_blueprint(build_blueprint(list(conv.cam_paths)))

    print(f"\nconverted {n} messages.")
    if conv.skipped:
        print("skipped (no python typesupport or undecodable):")
        for k, v in sorted(conv.skipped.items()):
            print(f"  {v:6d}  {k}")
    if not args.spawn:
        print(f"\n✓ wrote {out}\n  view with:  rerun {out}")


if __name__ == "__main__":
    main()
