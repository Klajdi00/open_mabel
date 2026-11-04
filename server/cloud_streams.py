"""
cloud_streams.py — per-camera live pointcloud planes (:9601-9604).

One dedicated WebSocket port per camera, each speaking the CLOUD 0x05 record
from docs/data_streams.md (16-byte vertex: x,y,z f32 + r,g,b,a u8, chunked)
— the exact format the existing /map clients already decode, so consuming a
per-camera cloud is a port change, not a new decoder.

  9601 head (zed_left)     9602 wrist_left     9603 wrist_right     9604 front

Two interchangeable sources, same wire either way:
  * SimDepthCloudSource — pure-Mac sim: ONE thread (one GL context) renders
    depth+RGB per subscribed camera at low res, back-projects through the
    camera intrinsics, and publishes world-frame clouds. Depth never leaves
    the process — only the derived cloud is streamed.
  * RosCloudSource (real robot, P4) — subscribes the RealSense/ZED
    PointCloud2 topics and re-encodes them onto the same ports.

Demand + budget contract: nothing renders for a camera whose hub has no
subscriber; per-camera rate is capped (default 2 Hz) because macOS off-screen
GL is one serialized readback slot shared with the video renderers. Points are
capped (default 20k/frame) and clouds are snapshots — a slow client gets a
fresher cloud, never a backlog (small per-socket queues + drop-oldest).
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

import data_streams
from map_stream import MapStreamHub, encode_cloud_chunks

__all__ = ["CloudHubSet", "SimDepthCloudSource", "intrinsics_from_fovy",
           "CLOUD_CAMERAS"]

# Dedicated-plane stream key → MuJoCo camera name (sim source).
CLOUD_CAMERAS = {"head": "zed_left", "wrist_left": "l_wrist_cam",
                 "wrist_right": "r_wrist_cam", "front": "body_front_cam"}

# PRIORITY (2026-07-13): video + robot_state + control are REAL-TIME and must never
# be starved. The point clouds are a background overlay, so they are deliberately
# CHEAP: fewer points + lower rate. Each cloud plane deserialises a PointCloud2 and
# does tf/numpy work on the GIL, so both knobs directly buy back gateway headroom for
# the state stream. Was 2 Hz / 20k pts/cam (4 cams = 80k pts, ~8 Hz of GIL-heavy
# decode); now 1 Hz / 6k pts. Raise per-camera via the envs when a client needs a
# denser cloud and the box has headroom.
CLOUD_HZ = float(os.getenv("MABEL_CLOUD_HZ", "1.0"))            # per camera
CLOUD_MAX_POINTS = int(os.getenv("MABEL_CLOUD_MAX_POINTS", "6000"))
CLOUD_MAX_RANGE = float(os.getenv("MABEL_CLOUD_MAX_RANGE", "8.0"))   # metres
CLOUD_RES = os.getenv("MABEL_CLOUD_RES", "320x240")
_CLOUD_IDLE_S = 0.5     # unwatched cloud plane: idle-poll period (see _run_one)


class CloudHub(MapStreamHub):
    # Clouds are SNAPSHOTS: a subscriber that can't drain fast enough must get a
    # fresher cloud, not a deep backlog of stale ones. The parent's 256-deep
    # queue is sized for the multiplexed mesh stream; a handful of frames is
    # plenty here and bounds worst-case buffering per slow client.
    _QUEUE_MAX = 8


class CloudHubSet:
    """The four cloud hubs + their dedicated-port servers (one asyncio loop)."""

    def __init__(self, available: dict[str, bool], log, host: str = "0.0.0.0"):
        self.log = log
        self.host = host
        self.ports = dict(data_streams.CLOUD_PORTS)
        self.hubs: dict[str, CloudHub] = {}
        self._notices: dict[str, str] = {}
        for key in self.ports:
            if available.get(key, False):
                self.hubs[key] = CloudHub()
            else:
                reason = available.get(f"{key}_reason")
                self._notices[key] = reason if isinstance(reason, str) else \
                    "no cloud source for this camera on this rig"

    def wants(self, key: str) -> bool:
        hub = self.hubs.get(key)
        return bool(hub is not None and hub.enabled())

    def any_wanted(self) -> bool:
        return any(h.enabled() for h in self.hubs.values())

    def publish(self, key: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        hub = self.hubs.get(key)
        if hub is not None:
            hub.publish_cloud(encode_cloud_chunks(xyz, rgb,
                                                  max_points=CLOUD_MAX_POINTS))

    def start(self) -> bool:
        """Serve every port on a dedicated thread + asyncio loop."""
        if not data_streams.streams_enabled():
            return False
        import asyncio
        ready = threading.Event()
        holder: dict = {}

        def _thread():
            async def _serve():
                loop = asyncio.get_running_loop()
                for key, port in self.ports.items():
                    hub = self.hubs.get(key)
                    if hub is not None:
                        hub.attach_loop(loop)
                        src = hub
                    else:
                        src = data_streams.NoticeSource(self._notices[key], port)
                    try:
                        await data_streams.serve_hub_ws(
                            self.host, port, src, f"cloud:{key}", self.log)
                    except OSError as e:
                        self.log.warning("cloud port :%d unavailable (%s) — skipped",
                                         port, e)
                ready.set()
                await asyncio.Future()

            try:
                asyncio.run(_serve())
            except Exception as e:                                 # noqa: BLE001
                self.log.error("cloud plane failed: %s", e)
                holder["failed"] = True
                ready.set()

        threading.Thread(target=_thread, name="cloud-plane", daemon=True).start()
        ready.wait(timeout=10.0)
        return not holder.get("failed", False)


def intrinsics_from_fovy(model, cam_name: str, w: int, h: int):
    """(f, cx, cy) pinhole intrinsics of a MuJoCo camera at a render size —
    MuJoCo cameras are defined by vertical fovy alone; fx = fy = f."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        return None
    fovy = math.radians(float(model.cam_fovy[cid]))
    f = (h / 2.0) / math.tan(fovy / 2.0)
    return f, (w - 1) / 2.0, (h - 1) / 2.0, cid


class RosCloudSource:
    """Real robot: re-encode the RealSense/ZED PointCloud2 topics onto the
    :9601-9604 planes. One thread + PRIVATE rclpy Context PER camera — the
    map-stream lesson: a 30 Hz × ~0.5 MB depth firehose sharing a DDS socket
    with anything reliable livelocks it, so each firehose gets its own UDP
    socket. Subscriptions are demand-driven: no client on a port → the topic
    is not even delivered to this process.

    Clouds ALWAYS leave in the `odom` frame, via the shared CloudFramer chain
    (odom TF → base_link TF ∘ last odom → drop). Shipping the raw sensor cloud
    is not an option: the RealSense/ZED topics are in the camera OPTICAL frame
    (+z = forward), and every client maps ROS→viewer as (x, z, -y), so an
    untransformed cloud puts the camera's forward axis on the viewer's UP axis
    — the head/front clouds stand straight up out of the robot (the bug this
    class shipped until 2026-07-13, hit whenever SLAM is down: thor with
    cuvslam:=false publishes no `odom` TF frame at all). A dropped frame is
    always better than a mis-oriented one."""

    TOPICS = {
        "head":        os.getenv("MABEL_CLOUD_TOPIC_HEAD", "/camera/zed/depth/points"),
        "wrist_left":  os.getenv("MABEL_CLOUD_TOPIC_WRIST_LEFT", "/camera/wrist_left/depth/points"),
        "wrist_right": os.getenv("MABEL_CLOUD_TOPIC_WRIST_RIGHT", "/camera/wrist_right/depth/points"),
        "front":       os.getenv("MABEL_CLOUD_TOPIC_FRONT", "/camera/front/depth/points"),
    }
    TARGET_FRAME = os.getenv("MABEL_CLOUD_FRAME", "odom")
    # The odom TOPIC (always published, even with the odom TF absent) — the
    # base_link fallback composes the static URDF chain with this pose.
    ODOM_TOPIC = os.getenv("MABEL_MAP_ODOM_TOPIC", "/odom")

    def __init__(self, hubset: CloudHubSet, log):
        self.hubs = hubset
        self.log = log
        self._stop = threading.Event()

    def start(self) -> "RosCloudSource | None":
        try:
            import rclpy  # noqa: F401  (probe before spawning anything)
        except Exception as e:
            self.log.warning("ros clouds: rclpy unavailable (%s) — cloud ports idle", e)
            return None
        n = 0
        for key, topic in self.TOPICS.items():
            if key not in self.hubs.hubs:
                continue                     # bound as a notice — no source expected
            threading.Thread(target=self._run_one, args=(key, topic),
                             name=f"cloud-ros-{key}", daemon=True).start()
            n += 1
        self.log.info("ros clouds: %d per-camera PointCloud2 sources (own DDS "
                      "participant each, demand-driven)", n)
        return self if n else None

    def stop(self) -> None:
        self._stop.set()

    def _run_one(self, key: str, topic: str) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from nav_msgs.msg import Odometry
        from map_stream import CloudFramer, _extract_xyzrgb

        ctx = Context()
        rclpy.init(args=None, context=ctx)
        node = rclpy.create_node(f"mabel_cloud_{key}", context=ctx)
        framer = CloudFramer(self.TARGET_FRAME, f"cloud:{key}")
        try:
            from tf2_ros import Buffer, TransformListener
            framer.tf_buffer = Buffer()
            TransformListener(framer.tf_buffer, node, spin_thread=False)
        except Exception as e:                                     # noqa: BLE001
            self.log.warning("ros clouds: %s has no tf2 (%s) — clouds will be "
                             "dropped rather than mis-oriented", key, e)
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE
        hub = self.hubs.hubs[key]
        last = [0.0]
        period = 1.0 / max(CLOUD_HZ, 0.1)

        def _on_odom(msg) -> None:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            framer.set_odom_pose(p.x, p.y, p.z, q.x, q.y, q.z, q.w)

        # The odom pose feeds the base_link fallback ONLY — cheap (one small msg
        # per tick), and it keeps the plane honest while the odom TF is missing.
        node.create_subscription(Odometry, self.ODOM_TOPIC, _on_odom, qos)

        def _on_cloud(msg) -> None:
            now = time.monotonic()
            if now - last[0] < period:
                return
            last[0] = now
            xyz, rgb = _extract_xyzrgb(msg)
            if xyz is None or xyz.shape[0] == 0:
                return
            frame = getattr(msg.header, "frame_id", "") or ""
            xyz = framer.to_target(xyz, frame)
            if xyz is None:            # no usable transform — never ship a camera-
                return                 # optical-frame cloud ("pointing up" on every
                                       # client); drop the frame instead.
            self.hubs.publish(key, xyz, rgb)

        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        sub = None
        try:
            while not self._stop.is_set() and ctx.ok():
                want = hub.enabled()
                if want and sub is None:
                    sub = node.create_subscription(PointCloud2, topic, _on_cloud, qos)
                    self.log.info("ros clouds: %s subscribed (%s → :%d)",
                                  key, topic, self.hubs.ports[key])
                elif not want and sub is not None:
                    node.destroy_subscription(sub)
                    sub = None
                if sub is None:
                    # NO VIEWER: do not spin the executor hot. spin_once wakes on a
                    # 0.2 s timeout regardless, and with 4 cloud planes + map + slam
                    # that is a constant GIL drumbeat competing with the real-time
                    # state stream. Idle-poll slowly instead — a client that opens a
                    # cloud is picked up within HALF_IDLE.
                    self._stop.wait(_CLOUD_IDLE_S)
                    continue
                ex.spin_once(timeout_sec=0.2)
        except Exception as e:                                     # noqa: BLE001
            self.log.warning("ros clouds: %s source stopped: %s", key, e)
        finally:
            ex.remove_node(node)
            node.destroy_node()
            try:
                rclpy.shutdown(context=ctx)
            except Exception:
                pass


class SimDepthCloudSource:
    """Renders depth+RGB for each SUBSCRIBED cloud camera and publishes
    world-frame clouds. One thread = one GL context (the proven
    sim_sensor_source pattern); renderers are created lazily inside the thread.

    The depth image is internal only — the wire carries the derived cloud."""

    def __init__(self, model, state, nq: int, hubset: CloudHubSet, log,
                 gl_guard=None):
        self.model, self.state, self.nq = model, state, nq
        self.hubs = hubset
        self.log = log
        self._gl_guard = gl_guard        # EGL serialization (no-op on macOS glfw)
        try:
            w, h = (int(v) for v in CLOUD_RES.lower().split("x"))
        except ValueError:
            w, h = 320, 240
        self.res = (w, h)
        # Pixel-grid subsample stride so a full frame lands under the point cap
        # BEFORE masking (encode_cloud_chunks re-caps after range masking).
        px = w * h
        self.stride = max(1, int(math.ceil(math.sqrt(px / max(CLOUD_MAX_POINTS, 1)))))

    def start(self) -> None:
        threading.Thread(target=self._run, name="cloud-render", daemon=True).start()

    def _run(self) -> None:
        import contextlib
        import mujoco
        guard = self._gl_guard if self._gl_guard is not None else contextlib.nullcontext()
        w, h = self.res
        data_l = mujoco.MjData(self.model)
        with guard:
            rgb_r = mujoco.Renderer(self.model, height=h, width=w)
            depth_r = mujoco.Renderer(self.model, height=h, width=w)
            depth_r.enable_depth_rendering()
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False   # rays poison depth
        vopt.sitegroup[3] = 0
        # Resolve intrinsics + camera ids once; drop cameras absent from the scene.
        cams: dict[str, tuple] = {}
        for key, cam in CLOUD_CAMERAS.items():
            if key not in self.hubs.hubs:
                continue
            intr = intrinsics_from_fovy(self.model, cam, w, h)
            if intr is None:
                self.log.warning("cloud camera '%s' missing from scene — %s idle", cam, key)
                continue
            cams[key] = (cam, *intr)
        if not cams:
            return
        # Precompute the subsampled pixel grid (u right, v down).
        vs, us = np.mgrid[0:h:self.stride, 0:w:self.stride]
        us = us.ravel().astype(np.float32)
        vs = vs.ravel().astype(np.float32)
        period = 1.0 / max(CLOUD_HZ, 0.1)
        last: dict[str, float] = {k: 0.0 for k in cams}
        last_t: dict[str, float] = {k: -1.0 for k in cams}
        self.log.info("cloud render  %d cams @ ≤%.1f Hz, %dx%d stride %d (≤%d pts)",
                      len(cams), CLOUD_HZ, w, h, self.stride, CLOUD_MAX_POINTS)
        while True:
            now = time.time()
            did = False
            for key, (cam, f, cx, cy, cid) in cams.items():
                if not self.hubs.wants(key):
                    last_t[key] = -1.0             # next subscriber → fresh first frame
                    continue
                if now - last[key] < period:
                    continue
                t = float(self.state[0])
                if t == last_t[key]:               # robot still + frame already out
                    continue                       # (late joiners get the cached snapshot)
                last[key] = now
                last_t[key] = t
                data_l.qpos[:] = self.state[1:1 + self.nq]
                data_l.time = t
                mujoco.mj_kinematics(self.model, data_l)
                mujoco.mj_comPos(self.model, data_l)
                mujoco.mj_camlight(self.model, data_l)
                with guard:
                    depth_r.update_scene(data_l, camera=cam, scene_option=vopt)
                    depth = depth_r.render()
                    rgb_r.update_scene(data_l, camera=cam, scene_option=vopt)
                    rgb = rgb_r.render()
                z = depth[::self.stride, ::self.stride].ravel()
                good = (z > 0.05) & (z < CLOUD_MAX_RANGE) & np.isfinite(z)
                if not good.any():
                    self.hubs.publish(key, np.empty((0, 3), np.float32),
                                      np.empty((0, 3), np.uint8))
                    did = True
                    continue
                zg, ug, vg = z[good], us[good], vs[good]
                # Camera frame (MuJoCo/OpenGL): +x right, +y up, looking down -z;
                # depth is planar (distance along the view axis).
                pc = np.empty((zg.size, 3), np.float32)
                pc[:, 0] = (ug - cx) / f * zg
                pc[:, 1] = -(vg - cy) / f * zg
                pc[:, 2] = -zg
                R = np.asarray(data_l.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
                p = np.asarray(data_l.cam_xpos[cid], dtype=np.float64)
                world = (pc @ R.T.astype(np.float32)) + p.astype(np.float32)
                colors = rgb[::self.stride, ::self.stride].reshape(-1, 3)[good]
                self.hubs.publish(key, world, np.ascontiguousarray(colors, np.uint8))
                did = True
            if not did:
                time.sleep(0.05)                    # nobody watching → zero GPU
            else:
                time.sleep(0.005)
