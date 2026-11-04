#!/usr/bin/env python3
"""
sim_camera_server.py — serve MABEL's simulated cameras as MJPEG, like hardware.

This is the robot's "camera box": a standalone process that renders the MuJoCo
head (ZED) and wrist cameras and serves each as a plain HTTP MJPEG stream on
port 8080 — exactly the multipart/x-mixed-replace format every client (Vision
Pro, iOS app, web browser) pulls directly (no custom protocol). It mirrors the
robot sim_teleop_bridge.py is driving by reading the qpos snapshot the bridge
publishes to a shared memory-mapped file; it never runs physics itself.

Why a separate process: on macOS the off-screen GL render is a ~40 ms fixed
GPU-readback stall. Running it here, on its own core, keeps it from starving the
bridge's real-time physics loop. Physics and video then run in parallel, exactly
like a real robot where the cameras are independent hardware.

Workflow
--------
  # terminal 1 — physics + teleop (publishes robot state, no rendering):
  python sim_teleop_bridge.py --scene ../simulation/mabel_mujoco/scenes/mabel_home.xml

  # terminal 2 — the cameras (this process):
  python sim_camera_server.py

Clients pull http://<mac>:8080/camera/<name>/stream.mjpg as before.
Scene is auto-read from <state-file>.scene; override with --scene to force a match.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import asyncio
import logging
import sys
import threading
import time

import contextlib

import cv2
import numpy as np
import mujoco

from config import CameraSource
from mjpeg_server import MJPEGServer, PushCapture
from sim_cameras import CAMERAS, HEAD_W, HEAD_H, camera_plan, weighted_schedule
from dedicated_video import DedicatedVideoPlane
from cloud_streams import CloudHubSet, SimDepthCloudSource, CLOUD_CAMERAS
import data_streams

import console as con
log = con.setup("cameras")

# Multi-thread GL race fix (matches mabel_sim's sensor node). EGL — the headless
# Linux/GPU backend — is NOT safe to drive from several threads at once: creating
# / making-current per-thread contexts concurrently races and the render threads
# deadlock after one frame (the "video froze at 1 frame" bug). Serialize GL ops
# behind one lock when on EGL. On glfw (macOS) keep it a no-op so the ~41 ms
# readback stalls still OVERLAP across threads (the macOS throughput win).
_GL_LOCK = threading.Lock() if os.environ.get("MUJOCO_GL", "").lower() == "egl" else None


def _gl_guard():
    return _GL_LOCK if _GL_LOCK is not None else contextlib.nullcontext()


class _RosImagePublisher:
    """Optional ROS fan-out: publish each rendered camera as sensor_msgs/Image so
    the SAME MuJoCo render feeds BOTH the MJPEG server (iOS) and ROS. One render,
    two sinks — no second renderer in sim, and on the real robot no second open of
    a device that allows only one reader. rclpy is imported lazily so the plain
    MJPEG path never needs ROS."""

    def __init__(self, prefix: str, cams: list[str]) -> None:
        import rclpy
        from sensor_msgs.msg import Image
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        self._Image = Image
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("mabel_sim_camera_server")
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT     # video = latest-wins, low latency
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self._pubs = {c: self._node.create_publisher(Image, f"{prefix}/{c}/image_raw", qos)
                      for c in cams}
        log.info("ROS image publish ON → %s/<cam>/image_raw  (%d cams)", prefix, len(cams))

    def publish(self, cam: str, rgb: np.ndarray, sim_t: float) -> None:
        pub = self._pubs.get(cam)
        if pub is None:
            return
        h, w = rgb.shape[0], rgb.shape[1]
        m = self._Image()
        sec = int(sim_t)
        m.header.stamp.sec = sec
        m.header.stamp.nanosec = int((sim_t - sec) * 1e9)
        m.header.frame_id = cam
        m.height = h
        m.width = w
        m.encoding = "rgb8"
        m.is_bigendian = 0
        m.step = w * 3
        m.data = rgb.tobytes()
        pub.publish(m)


def _resolve_scene(state_file: str, scene_arg: str | None) -> str:
    if scene_arg:
        return scene_arg if os.path.isabs(scene_arg) else os.path.abspath(scene_arg)
    sidecar = state_file + ".scene"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            p = f.readline().strip()
        if p:
            return p
    sys.exit(f"No scene recorded at {sidecar}. Start sim_teleop_bridge.py first, "
             f"or pass --scene.")


def _start_http_thread(host, port, path_caps, ready, loop_holder, fanout=None):
    """Run the aiohttp MJPEG server on its own asyncio loop (this thread)."""
    from aiohttp import web

    async def _serve():
        server = MJPEGServer.from_captures(path_caps)
        server.set_h264_hook(fanout)             # H.264-over-WebSocket (browsers)
        # Same-origin operator console: http://<host>:<port>/console (avoids the
        # https→ws:// mixed-content block that silently kills LAN bridges).
        console = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "website"))
        if server.add_console(console):
            log.info("web console  http://%s:%d/console", con.lan_ip(), port)
        runner = web.AppRunner(server.app())
        await runner.setup()
        # Bind the port FIRST; only once it succeeds do we hand the loop to the
        # PushCaptures. Binding the loop before the bind would leave the render
        # thread pushing into a loop that gets closed when the bind fails.
        await web.TCPSite(runner, host, port).start()
        loop = asyncio.get_running_loop()
        server.start_captures(loop)              # binds the loop to each PushCapture
        loop_holder["loop"] = loop
        log.info("cameras up   http://%s:%d", con.lan_ip(), port)
        for path, cap in path_caps.items():
            log.info(con.kv(cap.source.name, f"→ {path}"))
        ready.set()
        try:
            await asyncio.Future()
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_serve())
    except OSError as e:
        if getattr(e, "errno", None) in (48, 98):
            log.error("Port %d already in use — another sim_camera_server.py is "
                      "probably running. Free it: lsof -ti :%d | xargs kill -9",
                      port, port)
        else:
            log.error("HTTP thread error: %s", e)
        loop_holder["failed"] = True
        ready.set()


def main() -> None:
    ap = argparse.ArgumentParser(description="MABEL simulated MJPEG camera server")
    ap.add_argument("--state-file", default="/tmp/mabel_teleop_state.f64",
                    help="memmap path written by sim_teleop_bridge.py")
    ap.add_argument("--scene", default=None,
                    help="force a scene (default: the one recorded by the bridge)")
    ap.add_argument("--head-res", default=f"{HEAD_W}x{HEAD_H}",
                    help=f"render resolution WxH for the head cameras (default {HEAD_W}x{HEAD_H})")
    ap.add_argument("--head-weight", type=int, default=None,
                    help="render-slot share for the head vs wrists (default 4; higher = "
                         "faster head, slower wrists)")
    ap.add_argument("--head-only", action="store_true",
                    help="render only the head camera (max head fps, no wrist thumbnails)")
    ap.add_argument("--stereo", action="store_true",
                    help="render both head eyes (zed_left+zed_right) for the per-eye stereo path")
    ap.add_argument("--shadows", action="store_true",
                    help="render shadows in the camera feeds (prettier, but the shadow "
                         "pass adds ~32 ms PER FRAME on macOS — ~73 ms/frame vs ~41 ms "
                         "without — nearly halving the total frame budget. Off by default "
                         "so the head AND wrist streams stay low-latency.")
    ap.add_argument("--head-fov", type=float,
                    default=(float(os.getenv("MABEL_HEAD_FOV")) if os.getenv("MABEL_HEAD_FOV") else None),
                    help="override the head-camera vertical FOV in degrees (match the bridge)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--max-fps", type=float, default=0.0,
                    help="optional per-camera FPS cap (0 = uncapped: render every "
                         "new pose as fast as the GPU allows, ~24 fps/cam). Set e.g. "
                         "30 to leave GPU headroom under heavy multi-camera load.")
    ap.add_argument("--secondary-fps", type=float,
                    default=float(os.getenv("MABEL_SIM_SECONDARY_FPS", "6.0")),
                    help="FPS cap for NON-primary cameras (wrists, front base, the "
                         "unused stereo eye). macOS off-screen render() is one "
                         "serialized ~54 ms GPU-readback slot, so N cameras rendering "
                         "flat-out share it N-ways (~2.6 fps each at 6 cams). Capping "
                         "the thumbnails hands that slot back to the head POV, which "
                         "stays ~16 fps instead of collapsing to ~2.6. 0 = uncapped "
                         "(old behaviour). The single highest-weight camera (the head) "
                         "is always uncapped.")
    ap.add_argument("--udp-video-port", type=int,
                    default=int(os.getenv("MABEL_UDP_VIDEO_PORT", "9091")),
                    help="UDP port for the latest-frame-wins JPEG video path "
                         "(udp_video.py; 0 disables). The headset subscribes here "
                         "for minimum-latency video and falls back to MJPEG/HTTP.")
    ap.add_argument("--h264-port", type=int,
                    default=int(os.getenv("MABEL_SIM_H264_PORT", "9092")),
                    help="UDP port for the H.264 video path (sim_h264.py; same 'MV' "
                         "protocol as JPEG, matches the real robot's :9092). 0 disables. "
                         "The SAME encoder also serves /camera/<name>/h264 over WebSocket "
                         "(browser WebCodecs).")
    ap.add_argument("--no-h264", action="store_true",
                    help="disable H.264 entirely (also: MABEL_SIM_H264=0). Clients then "
                         "use UDP-JPEG / MJPEG.")
    ap.add_argument("--h264-fps", type=int, default=int(os.getenv("MABEL_SIM_H264_FPS", "30")),
                    help="H.264 encoder cadence (re-sends the latest frame to hold it).")
    ap.add_argument("--h264-ws-width", type=int,
                    default=int(os.getenv("MABEL_SIM_H264_WS_WIDTH", "640")),
                    help="downscale long edge for the WebSocket H.264 profile "
                         "(LAN/UDP stays match-source). 0 = no downscale.")
    ap.add_argument("--h264-lan-bitrate", default=os.getenv("MABEL_SIM_H264_LAN_BITRATE", "auto"),
                    help="LAN/UDP H.264 bitrate ('auto' scales with resolution, or e.g. 6M).")
    ap.add_argument("--h264-ws-bitrate", default=os.getenv("MABEL_SIM_H264_WS_BITRATE", "2M"),
                    help="WebSocket H.264 bitrate (low + tight GOP for low latency).")
    ap.add_argument("--http-host", default=os.getenv("MABEL_HTTP_HOST", "0.0.0.0"))
    ap.add_argument("--http-port", type=int, default=int(os.getenv("MABEL_HTTP_PORT", "8080")))
    ap.add_argument("--ros", action="store_true",
                    help="ALSO publish each rendered camera as sensor_msgs/Image on ROS "
                         "(<ros-prefix>/<cam>/image_raw), so the ONE MuJoCo render feeds both "
                         "the MJPEG server (iOS) and ROS — no second renderer. Also enabled "
                         "with MABEL_CAM_ROS=1.")
    ap.add_argument("--ros-prefix", default=os.getenv("MABEL_CAM_ROS_PREFIX", "/sim_cam"),
                    help="ROS topic prefix for --ros image topics (default /sim_cam).")
    args = ap.parse_args()
    if os.getenv("MABEL_CAM_ROS", "0").lower() in ("1", "true", "yes", "on"):
        args.ros = True
    try:
        head_w, head_h = (int(v) for v in args.head_res.lower().split("x"))
    except ValueError:
        ap.error("--head-res must look like 960x540")

    scene = _resolve_scene(args.state_file, args.scene)
    log.info("Scene: %s", scene)
    model = mujoco.MjModel.from_xml_path(scene)
    if args.head_fov is not None:
        for cam in ("zed_left", "zed_right"):
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cid >= 0:
                model.cam_fovy[cid] = float(args.head_fov)
        log.info("head camera fovy overridden → %.1f°", args.head_fov)
    data = mujoco.MjData(model)
    nq = model.nq

    have = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)}

    # Wait for the bridge's published state.
    expected = (1 + nq) * 8
    log.info("waiting for robot state %s … (start sim_teleop_bridge.py)", args.state_file)
    for _ in range(240):
        if os.path.exists(args.state_file) and os.path.getsize(args.state_file) >= expected:
            break
        time.sleep(0.5)
    if not (os.path.exists(args.state_file) and os.path.getsize(args.state_file) >= expected):
        sys.exit("[cameras] state file never appeared / wrong size — scene mismatch?")
    state = np.memmap(args.state_file, dtype=np.float64, mode="r", shape=(1 + nq,))

    # Collapse the path-level roster into a per-camera render plan. The head gets
    # the largest weight (share of the ~14 render slots/sec the machine can make),
    # the wrists a small share, the unused stereo eye none — see sim_cameras.py.
    weights, cam_to_paths, cam_to_res = camera_plan(
        head_weight=args.head_weight, head_only=args.head_only, stereo=args.stereo)
    missing = [c for c, w in weights.items() if w > 0 and c not in have]
    if missing:
        log.warning("Scene missing cameras %s — disabling them. Present: %s",
                    sorted(missing), sorted(have))
        for c in missing:
            weights[c] = 0
    # 962x dedicated-video plane: the hi-res head SBS stream needs BOTH eyes
    # renderable even in --mono runs, so give the right eye a render thread when
    # the plane is on (demand-gated — it costs zero GPU until a :9620 client
    # attaches, exactly like --stereo with nobody watching).
    if data_streams.streams_enabled() and "zed_right" in have:
        weights["zed_right"] = max(weights.get("zed_right", 0), 1)
    schedule = weighted_schedule(weights)
    if not schedule:
        sys.exit(f"[cameras] no enabled camera exists in the scene; have {sorted(have)}")

    # One PushCapture per served path. Only build paths for cameras we render.
    path_caps: dict[str, PushCapture] = {}
    label_of: dict[str, str] = {}              # label of each camera's heaviest entry
    best_w: dict[str, int] = {}
    for c in CAMERAS:
        if c.name not in best_w or c.weight > best_w[c.name]:
            best_w[c.name] = c.weight
            label_of[c.name] = c.label
    for c in CAMERAS:
        if weights.get(c.name, 0) > 0:
            path_caps[c.path] = PushCapture(
                CameraSource(name=c.label, device="sim", path=c.path))

    # HEAD STEREO (the headset's source): zed_left|zed_right rendered from ONE
    # state snapshot by the left-eye worker and composited side by side, so the
    # two eyes are always the SAME physics instant. Two separate eye streams
    # drift by whole frames (each worker reads the state on its own clock, and
    # the client pairs whatever arrived last) — measured as visible left/right
    # mismatch and discomfort in the headset. A first-class path: MJPEG, UDP and
    # H.264 pick it up like any other camera.
    STEREO_PATH = "/camera/head_stereo/stream.mjpg"
    if "zed_left" in have and "zed_right" in have and weights.get("zed_left", 0) > 0:
        path_caps[STEREO_PATH] = PushCapture(
            CameraSource(name="Head Stereo (L|R, synced)", device="sim", path=STEREO_PATH))

    # 962x dedicated-port video plane (docs/data_streams.md): head SBS hi-res on
    # :9620 (composited from the two eye renders), wrist/front standard-res
    # tees on :9621-9623. A camera absent from the scene binds as a 503 notice.
    dedi = None
    if data_streams.streams_enabled():
        _dedi_avail = {
            "head_sbs": "zed_left" in have and "zed_right" in have,
            "head_sbs_reason": "scene has no stereo head cameras",
            "wrist_left": weights.get("l_wrist_cam", 0) > 0,
            "wrist_left_reason": "scene has no left wrist camera",
            "wrist_right": weights.get("r_wrist_cam", 0) > 0,
            "wrist_right_reason": "scene has no right wrist camera",
            "front": weights.get("body_front_cam", 0) > 0,
            "front_reason": "scene has no front base camera",
        }
        dedi = DedicatedVideoPlane(_dedi_avail, log, host=args.http_host)
        if not dedi.start():
            dedi = None

    # Pointcloud planes :9601-9604 (CLOUD 0x05 records): depth is rendered
    # INTERNALLY on one dedicated thread and only the derived world-frame cloud
    # is streamed. Cameras absent from the scene bind as WS notices.
    clouds = None
    if data_streams.streams_enabled():
        _cloud_avail: dict = {}
        for _k, _cam in CLOUD_CAMERAS.items():
            _cloud_avail[_k] = _cam in have
            _cloud_avail[f"{_k}_reason"] = f"scene has no '{_cam}' camera"
        clouds = CloudHubSet(_cloud_avail, log, host=args.http_host)
        if clouds.start():
            SimDepthCloudSource(model, state, nq, clouds, log,
                                gl_guard=_GL_LOCK).start()
        else:
            clouds = None
    # MuJoCo camera name → its dedicated-plane role: the eyes feed the SBS
    # compositor; wrist/front just tee their existing JPEGs.
    _DEDI_EYE = {"zed_left": "left", "zed_right": "right"}
    _DEDI_TEE = {"l_wrist_cam": "wrist_left", "r_wrist_cam": "wrist_right",
                 "body_front_cam": "front"}

    # Low-latency UDP video tee — every JPEG pushed to a PushCapture is also
    # chunked onto UDP for subscribed headsets (MJPEG/HTTP stays as fallback).
    udp_video = None
    from udp_video import stream_name            # path → wire stream key (for viewer checks)
    if args.udp_video_port:
        try:
            from udp_video import UDPVideoSender
            udp_video = UDPVideoSender(port=args.udp_video_port)
            udp_video.start()
            udp_video.attach(path_caps)
        except OSError as e:
            log.warning("UDP video disabled — port %d unavailable (%s). "
                        "Clients will use the MJPEG fallback.",
                        args.udp_video_port, e)

    # H.264 video is OFF BY DEFAULT IN THE SIM. Measured:
    # the Mac's ffmpeg h264_videotoolbox encoder adds ~400 ms of FIXED pipeline latency
    # (bandwidth-independent), so the POV lagged badly even on wifi, while MJPEG drop-to-
    # latest is ~3–8 ms. The sim therefore serves only MJPEG (per-client transcoded, drop-
    # to-latest) + UDP-JPEG; every client (iOS already, web + Vision Pro via their MJPEG
    # fallback) renders MJPEG. This is SIM-ONLY — the real robot encodes with NVENC
    # (hardware_bridge), which is genuinely low-latency, and is untouched. Opt back in for
    # an experiment with MABEL_SIM_H264=1.
    h264_on = (args.h264_port and not args.no_h264
               and os.getenv("MABEL_SIM_H264", "0").lower() not in ("0", "false", "no", "off"))
    fanout = None
    if h264_on:
        try:
            from sim_h264 import SimH264Fanout
            h264_udp = UDPVideoSender(port=args.h264_port)
            h264_udp.start()
            fanout = SimH264Fanout(
                h264_udp, fps=args.h264_fps, lan_bitrate=args.h264_lan_bitrate,
                ws_width=args.h264_ws_width, ws_bitrate=args.h264_ws_bitrate,
                log=log)
            if not fanout._enabled:                # no working ffmpeg/encoder → drop it
                fanout = None
                h264_udp.close()
        except OSError as e:
            log.warning("H.264 UDP disabled — port %d unavailable (%s). "
                        "Clients will use UDP-JPEG / MJPEG.", args.h264_port, e)

    # Start the HTTP server on its own thread.
    ready = threading.Event()
    loop_holder: dict = {}
    threading.Thread(target=_start_http_thread,
                     args=(args.http_host, args.http_port, path_caps, ready,
                           loop_holder, fanout),
                     daemon=True).start()
    ready.wait(timeout=10.0)
    if loop_holder.get("failed") or "loop" not in loop_holder:
        sys.exit(f"[cameras] HTTP server failed to start on "
                 f"{args.http_host}:{args.http_port} — see the error above. Aborting.")

    encode = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    # Shadows are the single biggest render cost: on macOS the extra shadow PASS
    # (rendering the scene from the sun's POV) adds ~32 ms to every frame —
    # ~73 ms vs ~41 ms — independent of camera resolution, which is why a wrist
    # thumbnail was as slow to produce as the head. Off by default so the whole
    # camera budget roughly DOUBLES (~14 → ~24 frames/s) and every stream stays
    # responsive; `--shadows` restores them. (The desktop viewer is unaffected.)
    shadows_on = bool(args.shadows)

    enabled = sorted({c for c in schedule})
    log.info("render plan  %d cameras (one thread each) │ shadows %s │ GL=%s%s │ %s",
             len(enabled),
             "on" if shadows_on else "off (low-latency)",
             os.environ.get("MUJOCO_GL", "?"),
             " +lock" if _GL_LOCK is not None else "",
             " · ".join(f"{label_of.get(c, c)}" for c in enabled))

    # Optional ROS fan-out: the same render also goes out as ROS Image topics.
    ros_pub = None
    if args.ros:
        try:
            ros_pub = _RosImagePublisher(args.ros_prefix, enabled)
        except Exception as e:
            log.warning("ROS image publish unavailable (%s) — MJPEG only.", e)

    # PARALLEL per-camera rendering. macOS' off-screen render() is a FIXED ~41 ms
    # GPU-readback stall regardless of resolution, but it RELEASES the GIL and the
    # stalls OVERLAP across threads (measured: 2 threads ≈ 1.7× throughput). The old
    # single-thread weighted round-robin served all cameras off one ~24 fps budget
    # (head ~12, wrists ~6); one render thread PER camera instead gives every stream
    # ~full single-camera fps and stops the wrists from starving the head.
    #
    # Each thread owns its own MjData + Renderer (GL contexts are per-thread). To
    # mirror the robot it runs only the POSITION pipeline (kinematics+comPos+camlight
    # ≈ 0.02 ms) — pixel-identical to mj_forward but ~340× cheaper (no mass-matrix /
    # collision / constraint solve, none of which rendering needs). It renders at the
    # stream's NATIVE resolution (cost is resolution-independent, so this just avoids
    # a downscale and shrinks the JPEG → lower encode + wire latency), and only when
    # the pose actually changes, so a still robot costs no GPU.
    min_interval = (1.0 / args.max_fps) if getattr(args, "max_fps", 0) else 0.0

    # Per-camera FPS cap. macOS off-screen render() is ONE serialized ~54 ms slot, so
    # every camera rendering flat-out shares it (6 cams ≈ 2.6 fps each — measured
    # WORSE total throughput than 1 cam, the "overlap" only holds to ~2 threads). We
    # protect the head POV: the single highest-weight camera is uncapped; every other
    # camera (wrists, front, unused eye) is capped at --secondary-fps so its renders
    # don't monopolise the slot. Capping the thumbnails hands the budget back to the
    # head (2.6 → ~16 fps at 6 active cams).
    _max_w = max(weights.values()) if weights else 0
    _sec_int = (1.0 / args.secondary_fps) if getattr(args, "secondary_fps", 0) else 0.0
    cam_min_interval = {
        c: (min_interval if weights.get(c, 0) >= _max_w
            else max(min_interval, _sec_int))
        for c in enabled
    }

    def _dedi_key(cam: str) -> str | None:
        """The dedicated-plane stream this MuJoCo camera feeds (if any)."""
        if cam in _DEDI_EYE:
            return "head_sbs"
        return _DEDI_TEE.get(cam)

    def has_viewers(paths: list[str], cam: str | None = None) -> bool:
        """Is ANY client (MJPEG or UDP) currently watching one of these paths?
        Drives demand-driven rendering: a camera nobody is looking at costs zero
        GPU. The single biggest head-fps win — a browser/iPhone watching only the
        mono `main` eye no longer pays to render the headset's right eye, and idle
        wrists don't steal the head's GPU-readback budget (measured ~2x head fps,
        tools/exp_video_latency.py). A dedicated-port (962x) client counts as a
        viewer of the camera that feeds its stream."""
        for p in paths:
            if path_caps[p].has_clients:
                return True
            if udp_video is not None and udp_video.has_subscriber(stream_name(p)):
                return True
            if fanout is not None and fanout.wants(stream_name(p)):   # UDP- or WS-H.264
                return True
        if dedi is not None and cam is not None:
            key = _dedi_key(cam)
            if key is not None and dedi.wants(key):
                return True
        return False

    def render_worker(cam: str) -> None:
        data_l = mujoco.MjData(model)
        res = cam_to_res.get(cam)
        w, h = res if res is not None else (head_w, head_h)
        my_interval = cam_min_interval.get(cam, min_interval)
        # Hide the LiDAR rangefinder rays: the default option draws the 360 laser
        # beams as decorative geoms, which otherwise streak across the camera feed
        # (and, on the depth path, poison the pointcloud). Off for all renders.
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        vopt.sitegroup[3] = 0
        with _gl_guard():                    # context CREATION is the main EGL race point
            r = mujoco.Renderer(model, height=h, width=w)
        paths = [p for p in cam_to_paths[cam] if p in path_caps]
        # The LEFT eye's worker also owns the synced stereo pair (see STEREO_PATH).
        stereo_paths = ([STEREO_PATH] if cam == "zed_left" and STEREO_PATH in path_caps
                        else [])
        dedi_key = _dedi_key(cam) if dedi is not None else None
        last_t = -1.0
        last_push = 0.0
        served = False                       # have we pushed a frame to the CURRENT viewers?
        while True:
            want_stereo = bool(stereo_paths) and has_viewers(stereo_paths)
            if not has_viewers(paths, cam) and not want_stereo:
                served = False               # next viewer forces a fresh first frame
                time.sleep(0.05)             # nobody watching → zero GPU
                continue
            t = float(state[0])
            now = time.time()
            # While the hi-res SBS stream is watched, BOTH eyes run at the head
            # cadence (uncapped) — a secondary-fps-capped right eye would give
            # the stereo pair mismatched timestamps (nauseating in a headset).
            eff_interval = (min_interval
                            if (dedi_key == "head_sbs" and dedi.wants("head_sbs"))
                            else my_interval)
            # Render on a pose change OR to hand the first frame to a new viewer
            # (so a still robot isn't a black panel). Then honour the fps cap.
            if (t == last_t and served) or (eff_interval and now - last_push < eff_interval):
                time.sleep(0.001)            # robot still / fps-capped → idle (no GPU)
                continue
            data_l.qpos[:] = state[1:1 + nq]
            data_l.time = t
            mujoco.mj_kinematics(model, data_l)
            mujoco.mj_comPos(model, data_l)
            mujoco.mj_camlight(model, data_l)
            with _gl_guard():                # serialize GL on EGL (no-op on glfw)
                r.update_scene(data_l, camera=cam, scene_option=vopt)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows_on else 0
                rgb = r.render().copy()      # copy: render() reuses its internal buffer
            rgb_r = None
            if want_stereo:
                # SAME data_l (same qpos, same t) → the right eye is the same
                # instant as the left by construction. zed_right has no res
                # override, so it renders at the head size like zed_left.
                with _gl_guard():
                    r.update_scene(data_l, camera="zed_right", scene_option=vopt)
                    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows_on else 0
                    rgb_r = r.render().copy()
            last_t = t
            last_push = now
            if rgb_r is not None:
                sbs = cv2.cvtColor(np.hstack((rgb, rgb_r)), cv2.COLOR_RGB2BGR)
                if fanout is not None and fanout.wants("head_stereo"):
                    fanout.feed("head_stereo", sbs)
                ok_s, buf_s = cv2.imencode(".jpg", sbs, encode)
                if ok_s:
                    path_caps[STEREO_PATH].push(buf_s.tobytes())
            # Same render → ROS (raw RGB) AND the MJPEG/UDP server (JPEG).
            if ros_pub is not None:
                ros_pub.publish(cam, rgb, t)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # H.264 tap: feed the RAW BGR (pre-JPEG, native resolution) to the encoder
            # — no extra decode, no generational JPEG artifacts, "match source" is free.
            # Demand-gated, so an unwatched stream never spawns ffmpeg.
            if fanout is not None:
                for p in paths:
                    name = stream_name(p)
                    if fanout.wants(name):
                        fanout.feed(name, bgr)
            # Dedicated-plane taps (962x). Eyes feed the SBS compositor with the
            # RAW BGR (it encodes the composite once, at hi-res quality); wrist/
            # front just tee the JPEG encoded below. Both demand-gated.
            if dedi is not None and dedi_key == "head_sbs" and dedi.wants("head_sbs"):
                dedi.push_eye(_DEDI_EYE[cam], bgr)
            ok, buf = cv2.imencode(".jpg", bgr, encode)
            if ok:
                jpeg = buf.tobytes()
                for p in paths:
                    path_caps[p].push(jpeg)
                if (dedi is not None and dedi_key is not None
                        and dedi_key != "head_sbs" and dedi.wants(dedi_key)):
                    dedi.push(dedi_key, jpeg)
                served = True                # this viewer set now has a frame

    workers = [threading.Thread(target=render_worker, args=(c,),
                                name=f"render-{c}", daemon=True) for c in enabled]
    for th in workers:
        th.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
