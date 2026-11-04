"""ros_camera_node.py -- the SINGLE camera-hardware node for MABEL.

ONE process owns the USB depth cameras (Intel RealSense D435i + Stereolabs
ZED-M) through the ROS-free `camera_hal`, then fans every frame out to BOTH:

  1. ROS 2  -- publishes the sim camera contract (/camera/front/*, /camera/zed/*)
               so RViz / nvblox / cuVSLAM consume real pixels over DDS, exactly
               like the sim. ROS NEVER opens the hardware itself.
  2. SERVER -- pushes JPEG frames into the server's MJPEG/UDP machinery
               (server/mjpeg_server.PushCapture), so the Vision Pro / web / iOS
               clients get the same /camera/*/stream.mjpg paths over the network.

This is the "hardware interface" layer the ROS hardware bring-up connects to via
DDS: the cameras live here, behind one owner, and everyone else subscribes.

    # ROS only (what the bring-up launches):
    python ros_camera_node.py --ros
    # ROS + server MJPEG on :8080:
    python ros_camera_node.py --ros --serve --http-port 8080
    # server only (no ROS):
    python ros_camera_node.py --serve

Run it in sensors/.venv (has pyrealsense2 V4L2 + pyzed + rclpy). The wrapper
scripts/run_camera_node.sh pins CycloneDDS/domain 0 so it shares the conda ROS
graph (the same apt<->conda DDS bridge the SLAM stack uses).
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_SERVER_DIR = os.path.normpath(os.path.join(_HERE, '..', 'server'))   # udp_video
sys.path.insert(0, _HERE)                       # camera_hal
from camera_hal import (CameraManager, RealSenseCamera, UvcCamera,  # noqa: E402
                        ZedCamera)
from camera_hal.frame import CameraFrame                          # noqa: E402

# Camera extrinsics derived from the CANONICAL MJCF (simulation/mabel_mujoco/
# models/mabel_full.xml) so the published optical frames -- and the pointclouds in
# them -- line up with the robot in RViz EXACTLY as the sim does. Each entry: the
# parent URDF link, the translation (m) in that link, and the optical-frame
# orientation as a quaternion (x,y,z,w). Optical frame is REP-145: x-right,
# y-down, z-forward (the view direction).
#
# Derivation (MuJoCo camera: looks down -Z, image right=+X, up=+Y; so optical
# x=cam_x, y=-cam_y, z=-cam_z):
#  * front  body_front_cam @ chassis: pos (-0.25,0,0.13), xyaxes "0 1 0  0 0 1"
#    -> cam_x=(0,1,0) cam_y=(0,0,1) cam_z=(1,0,0); optical x=(0,1,0) y=(0,0,-1)
#    z=(-1,0,0) (looks down robot -X). quat = (-0.5,-0.5,0.5,0.5). base_link ==
#    mabel_chassis (identity base_joint), so parent = base_link.
#  * zed_left @ head: pos (-0.0317,0.006,-0.1089), xyaxes ~ identity -> optical is
#    head rotated 180 deg about X. quat = (1,0,0,0). PARENT = head so the cloud
#    rides the neck (parenting to base_link is what pinned it at the origin).
_CAM_EXTRINSICS = {
    'front': {'parent': 'base_link', 'xyz': (-0.25, 0.0, 0.13),
              'quat': (-0.5, -0.5, 0.5, 0.5), 'img_ns': 'color',
              'color_frame': 'front_color_optical_frame',
              'depth_frame': 'front_depth_optical_frame'},
    'zed':   {'parent': 'head', 'xyz': (-0.0317, 0.006, -0.1089),
              'quat': (1.0, 0.0, 0.0, 0.0), 'img_ns': 'left',
              'color_frame': 'zed_left_optical_frame',
              'depth_frame': 'zed_depth_optical_frame'},
}


# --- per-camera ROS wiring (topic + frame contract, matching mabel_sim) -------
class _CamPub:
    """Publishers + static TF for one camera, on the /camera/<key>/* contract."""

    def __init__(self, node, key: str):
        from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.key = key
        ext = _CAM_EXTRINSICS[key]
        # left/color image lives on .../left for the ZED (stereo) and .../color
        # for the RealSense (RGB-D) -- matches sensor_config.py + hardware.rviz.
        img_ns = ext['img_ns']
        self.color_frame = ext['color_frame']
        self.depth_frame = ext['depth_frame']
        q = qos_profile_sensor_data
        self.pub_color = node.create_publisher(Image, f'/camera/{key}/{img_ns}/image_raw', q)
        self.pub_color_info = node.create_publisher(CameraInfo, f'/camera/{key}/{img_ns}/camera_info', q)
        self.pub_depth = node.create_publisher(Image, f'/camera/{key}/depth/image_rect_raw', q)
        self.pub_depth_info = node.create_publisher(CameraInfo, f'/camera/{key}/depth/camera_info', q)
        self.pub_cloud = node.create_publisher(PointCloud2, f'/camera/{key}/depth/points', q)
        self.pub_imu = node.create_publisher(Imu, f'/camera/{key}/imu', q)
        # Rectified IR stereo pair -> the cuVSLAM input (visual_slam navigation env).
        # Only the front RealSense exposes one; advertise the topics so the SLAM
        # stack binds them, and fill them in once IR frames arrive.
        self.with_ir = (key == 'front')
        self.ir1_frame = f'{key}_infra1_optical_frame'
        self.ir2_frame = f'{key}_infra2_optical_frame'
        self._ir_tf_done = False
        if self.with_ir:
            self.pub_ir1 = node.create_publisher(Image, f'/camera/{key}/infra1/image_rect_raw', q)
            self.pub_ir1_info = node.create_publisher(CameraInfo, f'/camera/{key}/infra1/camera_info', q)
            self.pub_ir2 = node.create_publisher(Image, f'/camera/{key}/infra2/image_rect_raw', q)
            self.pub_ir2_info = node.create_publisher(CameraInfo, f'/camera/{key}/infra2/camera_info', q)
        self._last_stamp = -1.0
        self._publish_static_tf(ext)

    def _publish_static_tf(self, ext):
        # <parent URDF link> -> <cam>_{color,depth}_optical_frame, from the MJCF
        # (the auto-generated URDF carries no camera frames). Parent is base_link
        # for the front cam and `head` for the ZED, so each cloud renders at the
        # real mount and the ZED tracks the neck. Latched (StaticTransform).
        from tf2_ros import StaticTransformBroadcaster
        from geometry_msgs.msg import TransformStamped
        self._statictf = StaticTransformBroadcaster(self.node)
        tfs = []
        for child in (self.color_frame, self.depth_frame):
            t = TransformStamped()
            t.header.stamp = self.node.get_clock().now().to_msg()
            t.header.frame_id = ext['parent']
            t.child_frame_id = child
            (t.transform.translation.x, t.transform.translation.y,
             t.transform.translation.z) = (float(v) for v in ext['xyz'])
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = ext['quat']
            tfs.append(t)
        self._statictf.sendTransform(tfs)

    def publish(self, fr: CameraFrame):
        if fr is None or fr.stamp == self._last_stamp:
            return False
        self._last_stamp = fr.stamp
        stamp = self.node.get_clock().now().to_msg()
        if fr.rgb is not None:
            # camera_hal hands back BGR (OpenCV order); publish RGB to MATCH the sim
            # contract (mujoco_sensors_node publishes rgb8) -- and because nvblox
            # rejects bgr8 ("Invalid color image encoding") for its colour integration.
            rgb8 = np.ascontiguousarray(fr.rgb[:, :, ::-1])
            self.pub_color.publish(_image_msg(rgb8, 'rgb8', stamp, self.color_frame))
            if fr.intrinsics:
                self.pub_color_info.publish(_caminfo_msg(fr.intrinsics, stamp, self.color_frame))
        if fr.depth_m is not None:
            self.pub_depth.publish(_image_msg(
                fr.depth_m.astype(np.float32), '32FC1', stamp, self.depth_frame))
            if fr.intrinsics:
                self.pub_depth_info.publish(_caminfo_msg(fr.intrinsics, stamp, self.depth_frame))
        if fr.has_cloud:
            self.pub_cloud.publish(_cloud_msg(
                fr.points_xyz, fr.points_rgb, stamp, self.depth_frame))
        if fr.imu is not None:
            self.pub_imu.publish(_imu_msg(fr.imu, stamp, self.color_frame))
        if self.with_ir and fr.has_ir_stereo and fr.ir_intrinsics is not None:
            base = float(fr.ir_baseline_m or 0.0)
            if not self._ir_tf_done:
                self._publish_ir_tf(base)
                self._ir_tf_done = True
            self.pub_ir1.publish(_image_msg(fr.ir_left, 'mono8', stamp, self.ir1_frame))
            self.pub_ir2.publish(_image_msg(fr.ir_right, 'mono8', stamp, self.ir2_frame))
            self.pub_ir1_info.publish(
                _caminfo_msg(fr.ir_intrinsics, stamp, self.ir1_frame))
            # The RIGHT imager carries the stereo baseline in P[3] = -fx*baseline,
            # the rectified-stereo convention cuVSLAM triangulates with.
            self.pub_ir2_info.publish(
                _caminfo_msg(fr.ir_intrinsics, stamp, self.ir2_frame,
                             tx=-fr.ir_intrinsics.fx * base))
        return True

    def _publish_ir_tf(self, baseline_m: float):
        # IR optical frames, published once the baseline is known (first IR frame).
        # The LEFT imager (infra1) sits at the front cam mount -- reuse the cam's
        # base_link extrinsic. The RIGHT imager (infra2) is its child, offset by the
        # baseline along the optical +X axis (REP-145: x-right), identity rotation --
        # an exact rectified-stereo rig matching the camera_info P matrices.
        from tf2_ros import StaticTransformBroadcaster
        from geometry_msgs.msg import TransformStamped
        ext = _CAM_EXTRINSICS[self.key]
        self._ir_statictf = StaticTransformBroadcaster(self.node)
        now = self.node.get_clock().now().to_msg()
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = ext['parent']
        t1.child_frame_id = self.ir1_frame
        (t1.transform.translation.x, t1.transform.translation.y,
         t1.transform.translation.z) = (float(v) for v in ext['xyz'])
        (t1.transform.rotation.x, t1.transform.rotation.y,
         t1.transform.rotation.z, t1.transform.rotation.w) = ext['quat']
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = self.ir1_frame
        t2.child_frame_id = self.ir2_frame
        t2.transform.translation.x = float(baseline_m)
        t2.transform.rotation.w = 1.0
        self._ir_statictf.sendTransform([t1, t2])


# --- wrist UVC camera ROS wiring (colour-only) --------------------------------
class _WristPub:
    """Publishers for ONE wrist UVC cam, on the SAME contract the hardware_bridge
    wrist driver uses so RViz / consumers see one stream name either way:
        /camera/wrist_<side>/image_raw    (rgb8)
        /camera/wrist_<side>/camera_info
    Colour only -- no depth / cloud / IMU. No static TF: the wrist moves with the
    arm and its kinematics aren't owned here; the 2-D Image display needs none."""

    def __init__(self, node, key: str):
        from sensor_msgs.msg import Image, CameraInfo
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.key = key                               # 'wrist_left' | 'wrist_right'
        self.frame_id = f'{key}_camera_optical_frame'
        q = qos_profile_sensor_data
        self.pub_color = node.create_publisher(Image, f'/camera/{key}/image_raw', q)
        self.pub_color_info = node.create_publisher(
            CameraInfo, f'/camera/{key}/camera_info', q)
        self._last_stamp = -1.0

    def publish(self, fr: CameraFrame) -> bool:
        if fr is None or fr.rgb is None or fr.stamp == self._last_stamp:
            return False
        self._last_stamp = fr.stamp
        stamp = self.node.get_clock().now().to_msg()
        # camera_hal hands back BGR (OpenCV order); publish rgb8 like the depth cams.
        rgb8 = np.ascontiguousarray(fr.rgb[:, :, ::-1])
        self.pub_color.publish(_image_msg(rgb8, 'rgb8', stamp, self.frame_id))
        if fr.intrinsics:
            self.pub_color_info.publish(_caminfo_msg(fr.intrinsics, stamp, self.frame_id))
        return True


# --- numpy -> ROS message builders (no cv_bridge; plain rclpy) ----------------
def _image_msg(arr: np.ndarray, encoding: str, stamp, frame_id):
    from sensor_msgs.msg import Image
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = int(arr.shape[0]), int(arr.shape[1])
    m.encoding = encoding
    m.is_bigendian = 0
    m.step = int(arr.strides[0])
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


def _caminfo_msg(intr, stamp, frame_id, tx=0.0):
    """CameraInfo from pinhole intrinsics. `tx` sets P[0,3] = -fx*baseline for the
    RIGHT camera of a rectified stereo pair (0 for mono / the left camera)."""
    from sensor_msgs.msg import CameraInfo
    m = CameraInfo()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.width, m.height = int(intr.width), int(intr.height)
    m.distortion_model = 'plumb_bob'
    m.d = [float(c) for c in (intr.coeffs or (0.0, 0.0, 0.0, 0.0, 0.0))]
    m.k = [intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [intr.fx, 0.0, intr.cx, float(tx), 0.0, intr.fy, intr.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


def _cloud_msg(xyz: np.ndarray, rgb: np.ndarray | None, stamp, frame_id):
    from sensor_msgs.msg import PointCloud2, PointField
    n = int(xyz.shape[0])
    arr = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
    arr['x'] = xyz[:, 0]; arr['y'] = xyz[:, 1]; arr['z'] = xyz[:, 2]
    if rgb is not None and len(rgb) == n:
        b = rgb[:, 0].astype(np.uint32); g = rgb[:, 1].astype(np.uint32); r = rgb[:, 2].astype(np.uint32)
        packed = (r << 16) | (g << 8) | b               # 0x00RRGGBB
        arr['rgb'] = packed.view(np.float32)
    m = PointCloud2()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height = 1
    m.width = n
    m.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    m.is_bigendian = False
    m.point_step = 16
    m.row_step = 16 * n
    m.is_dense = False
    m.data = arr.tobytes()
    return m


def _imu_msg(imu, stamp, frame_id):
    from sensor_msgs.msg import Imu
    m = Imu()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = imu.accel
    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = imu.gyro
    if imu.orientation is not None:
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = imu.orientation
    else:
        m.orientation_covariance[0] = -1.0          # "no orientation" per REP-145
    return m


# Stream-name contract the web/iOS/Vision Pro clients already pull. The website
# shows "main" (head) + the two wrists; we map the head ZED -> main + head_left,
# the front D435i -> front, and each wrist UVC cam -> its own wrist_<side> stream
# (the SAME names the hardware_bridge video server uses). Each camera key -> the
# wire stream names it feeds.
_STREAMS = {'zed': ['main', 'head_left'], 'front': ['front'],
            'wrist_left': ['wrist_left'], 'wrist_right': ['wrist_right']}


# --- H.264 fan-out: per-camera ffmpeg encoder -> chunked UDP (access units) ----
class _H264Fanout:
    """Low-latency H.264 for iOS/Vision Pro (a future VideoToolbox decoder). One
    ffmpeg per camera (libopenh264, baseline, IDR every ~1 s) reads raw BGR on a
    pipe and emits AUD-delimited access units, which we chunk over UDP reusing the
    proven udp_video transport (port 9092). Demand-driven: ffmpeg is only FED when
    a UDP subscriber is present, so it idles at ~0 CPU otherwise."""

    AUD = b'\x00\x00\x01\x09'                          # H.264 access-unit delimiter

    def __init__(self, streams: dict, port: int, fps: int, bitrate: str = '3M'):
        sys.path.insert(0, _SERVER_DIR)
        import udp_video
        self._udp = udp_video.UDPVideoSender(port=port)
        self._udp.start()
        self._streams = streams
        self._fps = fps
        self._bitrate = bitrate
        self._ffmpeg = shutil.which('ffmpeg') or 'ffmpeg'
        self._raw: dict[str, np.ndarray] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def _wanted(self, key: str) -> bool:
        return any(self._udp.has_subscriber(n) for n in self._streams.get(key, []))

    def push(self, key: str, bgr: np.ndarray):
        if key not in self._streams:
            return
        with self._lock:
            self._raw[key] = bgr
            if key not in self._procs:
                self._start(key, int(bgr.shape[1]), int(bgr.shape[0]))

    def _start(self, key, w, h):
        cmd = [self._ffmpeg, '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-framerate', str(self._fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libopenh264',
               '-rc_mode', 'bitrate', '-b:v', self._bitrate, '-g', str(self._fps),
               '-bsf:v', 'h264_metadata=aud=insert', '-f', 'h264', 'pipe:1']
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as e:                          # noqa: BLE001
            print(f'[h264] ffmpeg start failed ({e}); H.264 disabled for {key}', flush=True)
            return
        self._procs[key] = p
        threading.Thread(target=self._feed, args=(key, p), daemon=True).start()
        threading.Thread(target=self._read, args=(key, p), daemon=True).start()

    def _feed(self, key, p):
        period = 1.0 / self._fps
        while p.poll() is None:
            if not self._wanted(key):                   # demand-driven: idle ffmpeg
                time.sleep(0.1)
                continue
            with self._lock:
                bgr = self._raw.get(key)
            if bgr is not None:
                try:
                    p.stdin.write(np.ascontiguousarray(bgr).tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    break
            time.sleep(period)

    def _read(self, key, p):
        names = self._streams.get(key, [])
        buf = b''
        while p.poll() is None:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:                                 # split on AUD boundaries
                i = buf.find(self.AUD)
                if i < 0:
                    break
                j = buf.find(self.AUD, i + 4)
                if j < 0:
                    break
                au = buf[i:j]
                for n in names:
                    self._udp.send(n, au)
                buf = buf[j:]


# --- per-client bitrate control (slow links) ---------------------------------
# A client on a slow link (MJPEG rides TCP, where a full-res frame stalls the whole
# stream AND starves the SLAM mesh) asks for a
# smaller frame via ?q=<10-95>&scale=<0.1-1.0>. We decode the freshest JPEG and
# re-encode it downscaled, PER CLIENT, so LAN viewers (no params) keep the crisp
# frame untouched while a remote headset gets a light one that fits the link. This
# mirrors server/mjpeg_server.py, but PIL-only (no cv2 → no numpy ABI clash), to keep
# _VideoFanout self-contained.
def _clamp_int(v, lo, hi):
    try:
        return max(lo, min(hi, int(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clamp_float(v, lo, hi):
    try:
        return max(lo, min(hi, float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def _transcode_jpeg_pil(jpeg: bytes, q: int, scale: float | None) -> bytes:
    """Decode → (optional area downscale) → re-encode at quality `q`, PIL-only.
    Returns the original bytes on any failure so the stream never breaks."""
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)); img.load()
        if scale is not None and scale < 0.999:
            w, h = img.size
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.BILINEAR)
        b = io.BytesIO()
        img.convert('RGB').save(b, 'JPEG', quality=int(q))
        return b.getvalue()
    except Exception:
        return jpeg


# --- video fan-out: low-latency MJPEG (HTTP) + JPEG-over-UDP + H.264-over-UDP --
class _VideoFanout:
    """The node's video server -- the SAME contract the website / iOS / Vision Pro
    pull, so they work unchanged. Self-contained (Pillow + aiohttp, no cv2 -> no
    numpy ABI clash). Three transports, all demand-driven:
      * MJPEG  http://<host>:<port>/camera/<name>/stream.mjpg   (browser <img>, iOS)
      * JPEG   udp://<host>:9091   (Vision Pro low-latency, udp_video protocol)
      * H.264  udp://<host>:9092   (future VideoToolbox, low-latency teleop)

    LOW LATENCY: encoding runs in per-stream worker threads (off the capture/ROS
    loop) and ONLY when a client is watching; MJPEG delivery is event-driven with a
    maxsize=1 latest-wins queue per client + capped aiohttp write buffers (the
    dominant lag if left unbounded). push(key,bgr) just stows the latest frame."""

    def __init__(self, host, port, quality, fps, h264=True,
                 udp_port=9091, h264_port=9092):
        from PIL import Image                            # noqa: F401  (probe import)
        self._quality = quality
        self._fps = fps
        self._streams = dict(_STREAMS)
        self._name_key = {n: k for k, ns in self._streams.items() for n in ns}
        self._raw: dict[str, np.ndarray] = {}
        self._evt: dict[str, threading.Event] = {}
        self._latest_jpeg: dict[str, bytes] = {}
        self._clients: dict[str, set] = {}              # name -> set(asyncio.Queue)
        self._lock = threading.Lock()
        self._loop = None
        self._host, self._port = host, port
        sys.path.insert(0, _SERVER_DIR)
        import udp_video
        self._udp = udp_video.UDPVideoSender(port=udp_port)
        self._udp.start()
        self._h264 = _H264Fanout(self._streams, h264_port, fps) if h264 else None
        for key in self._streams:
            self._evt[key] = threading.Event()
            threading.Thread(target=self._encoder, args=(key,), daemon=True).start()
        threading.Thread(target=self._run_http, daemon=True).start()

    # -- producer side (capture/main loop): fast, non-blocking -----------------
    def push(self, key, bgr):
        if key not in self._streams:
            return
        with self._lock:
            self._raw[key] = bgr
        self._evt[key].set()
        if self._h264 is not None:
            self._h264.push(key, bgr)

    def _wanted(self, key):
        for n in self._streams[key]:
            with self._lock:
                if self._clients.get(n):
                    return True
            if self._udp.has_subscriber(n):
                return True
        return False

    # -- per-stream JPEG encoder thread: demand-driven, off the ROS loop --------
    def _encoder(self, key):
        import io
        from PIL import Image
        names = self._streams[key]
        while True:
            self._evt[key].wait()
            self._evt[key].clear()
            if not self._wanted(key):
                continue
            with self._lock:
                bgr = self._raw.get(key)
            if bgr is None:
                continue
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])  # camera_hal BGR -> RGB
            b = io.BytesIO()
            Image.fromarray(rgb).save(b, 'JPEG', quality=self._quality)
            jpeg = b.getvalue()
            for n in names:
                self._latest_jpeg[n] = jpeg
                self._distribute(n, jpeg)               # MJPEG clients (latest-wins)
                self._udp.send(n, jpeg)                 # UDP-JPEG subscribers

    def _distribute(self, name, jpeg):
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            queues = list(self._clients.get(name, ()))
        for q in queues:
            loop.call_soon_threadsafe(self._put_latest, q, jpeg)

    @staticmethod
    def _put_latest(q, jpeg):
        if q.full():
            try:
                q.get_nowait()
            except Exception:
                pass
        try:
            q.put_nowait(jpeg)
        except Exception:
            pass

    # -- HTTP MJPEG server (event-driven, write-buffer-capped) -----------------
    def _run_http(self):
        import asyncio
        from aiohttp import web

        async def serve():
            self._loop = asyncio.get_running_loop()
            app = web.Application()
            for n in self._name_key:
                app.router.add_get(f'/camera/{n}/stream.mjpg', self._handler)
            app.router.add_get('/', self._index)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, self._host, self._port).start()
            await asyncio.Future()
        try:
            asyncio.run(serve())
        except Exception as e:                           # noqa: BLE001
            print(f'[server] MJPEG server stopped: {e}', flush=True)

    async def _index(self, request):
        from aiohttp import web
        links = ''.join(f'<li><a href="/camera/{n}/stream.mjpg">{n}</a></li>'
                        for n in self._name_key)
        return web.Response(content_type='text/html',
                            text=f'<h1>MABEL cameras</h1><ul>{links}</ul>')

    async def _handler(self, request):
        import asyncio
        from aiohttp import web
        name = request.match_info.get('name') or request.path.split('/')[2]
        # Per-client bitrate (slow links): ?q=<10-95>&scale=<0.1-1.0> → transcode below.
        req_q = _clamp_int(request.query.get('q'), 10, 95)
        req_scale = _clamp_float(request.query.get('scale'), 0.1, 1.0)
        transcode = req_q is not None or (req_scale is not None and req_scale < 0.999)
        enc_q = req_q if req_q is not None else 70
        resp = web.StreamResponse(status=200, headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache', 'Connection': 'close'})
        await resp.prepare(request)
        loop = asyncio.get_event_loop()
        # LOW LATENCY: cap aiohttp's write buffer (unbounded -> ~1.5 s lag) and the
        # kernel send buffer to ~1 frame, so a slow link drops frames instead of
        # backlogging stale ones.
        try:
            request.transport.set_write_buffer_limits(high=131072, low=32768)
            sock = request.transport.get_extra_info('socket')
            if sock is not None:
                import socket as _s
                sock.setsockopt(_s.SOL_SOCKET, _s.SO_SNDBUF, 128 * 1024)
        except Exception:
            pass
        q = asyncio.Queue(maxsize=1)
        with self._lock:
            self._clients.setdefault(name, set()).add(q)
            self._evt_wake(name)
        j = self._latest_jpeg.get(name)                  # warm start
        if j:
            try:
                q.put_nowait(j)
            except Exception:
                pass
        try:
            while True:
                jpeg = await q.get()
                if transcode:
                    # Off the event loop: decode+resize+re-encode (~3-5 ms) would
                    # otherwise stall every other client's writes.
                    jpeg = await loop.run_in_executor(
                        None, _transcode_jpeg_pil, jpeg, enc_q, req_scale)
                await resp.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                    + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
            pass
        finally:
            with self._lock:
                self._clients.get(name, set()).discard(q)
        return resp

    def _evt_wake(self, name):
        # a new client just subscribed -> wake that camera's encoder immediately
        key = self._name_key.get(name)
        if key in self._evt:
            self._evt[key].set()


# Wrist UVC cams: udev-pinned symlink + by-path fallback (the SAME convention as
# hardware_bridge/config.py -- hub port 1-4.1.2 = LEFT, 1-4.1.1 = RIGHT).
_WRIST_DEFAULTS = {
    'wrist_left':  ('/dev/wrist_left',
                    '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2:1.0-video-index0'),
    'wrist_right': ('/dev/wrist_right',
                    '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.1:1.0-video-index0'),
}


def _build_cameras(which: str, args):
    cams = []
    if which in ('all', 'realsense'):
        cams.append(('front', RealSenseCamera(width=args.width, height=args.height,
                                              fps=args.fps, enable_ir=args.ir)))
    if which in ('all', 'zed'):
        cams.append(('zed', ZedCamera(resolution=args.zed_res, fps=args.fps)))
    if args.wrist:
        # Two colour-only Arducam Global-Shutter cams. Each degrades gracefully:
        # absent -> the manager just keeps retrying (never fatal), like the depth cams.
        devs = {'wrist_left': (args.wrist_left_dev, _WRIST_DEFAULTS['wrist_left'][1]),
                'wrist_right': (args.wrist_right_dev, _WRIST_DEFAULTS['wrist_right'][1])}
        for key, (dev, fallback) in devs.items():
            cams.append((key, UvcCamera(
                name=key, device=dev, fallback=fallback,
                model=f'Arducam Global-Shutter ({key})',
                width=args.wrist_width, height=args.wrist_height, fps=args.wrist_fps,
                auto_exposure=args.wrist_auto_exposure, exposure=args.wrist_exposure,
                gain=args.wrist_gain)))
    return cams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cameras', choices=['all', 'realsense', 'zed', 'none'], default='all',
                    help="which DEPTH cameras to own; 'none' = wrists only (use when the "
                         "hardware_bridge owns front+ZED -- avoids slow failing opens that "
                         "starve the wrist publish loop).")
    ap.add_argument('--ros', action='store_true', help='publish the /camera/* ROS contract')
    ap.add_argument('--serve', action='store_true', help='serve MJPEG to the network')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--zed-res', default='HD720')
    # Wrist UVC cams (Arducam Global-Shutter) -> /camera/wrist_<side>/image_raw.
    ap.add_argument('--wrist', dest='wrist', action='store_true', default=True,
                    help='publish both wrist UVC cams /camera/wrist_{left,right}/* (default).')
    ap.add_argument('--no-wrist', dest='wrist', action='store_false',
                    help='skip the wrist cams (depth cameras only).')
    ap.add_argument('--wrist-left-dev',
                    default=os.getenv('CAM_WRIST_LEFT_DEV', _WRIST_DEFAULTS['wrist_left'][0]))
    ap.add_argument('--wrist-right-dev',
                    default=os.getenv('CAM_WRIST_RIGHT_DEV', _WRIST_DEFAULTS['wrist_right'][0]))
    ap.add_argument('--wrist-width', type=int, default=int(os.getenv('CAM_WRIST_WIDTH', '640')))
    ap.add_argument('--wrist-height', type=int, default=int(os.getenv('CAM_WRIST_HEIGHT', '480')))
    ap.add_argument('--wrist-fps', type=int, default=int(os.getenv('CAM_WRIST_FPS', '30')))
    # Manual exposure (matches hardware_bridge CAM_WRIST_* env) -- auto-exposure
    # collapses these cams to ~2 fps in dim light. 1=manual (default), 3=auto.
    ap.add_argument('--wrist-auto-exposure', type=int,
                    default=int(os.getenv('CAM_WRIST_AUTO_EXPOSURE', '1')))
    ap.add_argument('--wrist-exposure', type=int,
                    default=int(os.getenv('CAM_WRIST_EXPOSURE', '120')))   # x100us -> 12 ms
    ap.add_argument('--wrist-gain', type=int,
                    default=int(os.getenv('CAM_WRIST_GAIN', '180')))
    ap.add_argument('--ir', dest='ir', action='store_true', default=True,
                    help='publish the front D435i IR stereo pair for cuVSLAM (default).')
    ap.add_argument('--no-ir', dest='ir', action='store_false',
                    help='skip IR stereo (MJPEG/RGB-D only, no visual SLAM).')
    ap.add_argument('--http-host', default=os.getenv('MABEL_HTTP_HOST', '0.0.0.0'))
    ap.add_argument('--http-port', type=int, default=int(os.getenv('MABEL_HTTP_PORT', '8080')))
    ap.add_argument('--udp-video-port', type=int,
                    default=int(os.getenv('UDP_VIDEO_PORT', '9091')),
                    help='UDP JPEG port (Vision Pro low-latency).')
    ap.add_argument('--h264-port', type=int, default=int(os.getenv('H264_VIDEO_PORT', '9092')),
                    help='UDP H.264 port (future VideoToolbox client).')
    ap.add_argument('--no-h264', action='store_true', help='disable the H.264 UDP encoder.')
    ap.add_argument('--jpeg-quality', type=int,
                    default=int(os.getenv('JPEG_QUALITY', '80')))
    ap.add_argument('--rate', type=float, default=60.0,
                    help='capture->fan-out poll rate (Hz). Higher = lower latency.')
    args = ap.parse_args()
    if not args.ros and not args.serve:
        args.ros = True                                 # default: ROS bring-up mode

    cam_defs = _build_cameras(args.cameras, args)
    manager = CameraManager([c for _, c in cam_defs])
    manager.start()
    print(f'[camera_node] opening {[k for k,_ in cam_defs]} '
          f'(ros={args.ros} serve={args.serve}) ...', flush=True)

    node = None
    pubs: dict[str, _CamPub] = {}
    if args.ros:
        import rclpy
        rclpy.init()
        node = rclpy.create_node('camera_hardware')
        for key, _ in cam_defs:
            # Wrist UVC cams are colour-only (no depth/cloud/IMU/TF) -> lighter pub.
            pubs[key] = (_WristPub(node, key) if key.startswith('wrist_')
                         else _CamPub(node, key))

    server = None
    if args.serve:
        # Video fan-out (MJPEG + UDP-JPEG + H.264-UDP). NON-FATAL: if it can't
        # start, ROS publishing must still run.
        try:
            # H.264-over-UDP (:9092) is a "future VideoToolbox" path that NOTHING
            # currently consumes (iOS = MJPEG :8080, Vision Pro = UDP-JPEG :9091), yet
            # push() fed every frame to 3 ffmpeg subprocesses unconditionally → ~1.3
            # CPU cores of pure waste that STARVED the MJPEG encoder (30fps capture →
            # 18fps served, jittery). Make it opt-in (MABEL_VIDEO_H264=1) so the default
            # path is lean and low-latency. Re-enable only when a VideoToolbox client exists.
            server = _VideoFanout(args.http_host, args.http_port,
                                  quality=args.jpeg_quality, fps=int(args.fps),
                                  h264=(os.getenv('MABEL_VIDEO_H264', '0') == '1'
                                        and not args.no_h264),
                                  udp_port=args.udp_video_port,
                                  h264_port=args.h264_port)
            print(f'[camera_node] video: MJPEG http://{args.http_host}:{args.http_port}'
                  f'/camera/<name>/stream.mjpg | JPEG udp:{args.udp_video_port}'
                  f"{'' if args.no_h264 else f' | H.264 udp:{args.h264_port}'}", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f'[camera_node] WARNING: video fan-out disabled ({type(e).__name__}: '
                  f'{str(e)[:80]}). ROS publishing continues.', flush=True)
            server = None

    period = 1.0 / max(1.0, args.rate)
    nframes = {k: 0 for k, _ in cam_defs}
    t_log = time.monotonic()
    try:
        while True:
            if node is not None:
                import rclpy
                rclpy.spin_once(node, timeout_sec=0.0)
            for key, cam in cam_defs:
                # look up by the camera_hal device name ('realsense'/'zedm'), NOT
                # our contract key ('front'/'zed') -- CameraManager keys on cam.name.
                fr = manager.latest(cam.name)
                if fr is None:
                    continue
                if key in pubs and pubs[key].publish(fr):
                    nframes[key] += 1
                if server is not None and fr.rgb is not None:
                    server.push(key, fr.rgb)
            now = time.monotonic()
            if now - t_log >= 5.0:
                st = ', '.join(f'{k}:{s.fps:.1f}fps/{s.state}' for k, s
                               in zip([k for k, _ in cam_defs], manager.statuses()))
                pub = ', '.join(f'{k}:{nframes[k]}pub' for k, _ in cam_defs)
                print(f'[camera_node] {st} | {pub}', flush=True)
                t_log = now
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        if node is not None:
            import rclpy
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
