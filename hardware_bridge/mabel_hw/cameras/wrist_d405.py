"""wrist_d405.py — native pyrealsense2 driver for an Intel RealSense D405 wrist cam.

Publishes the SAME per-side colour contract as the UVC wrist driver (wrist_cams.py),
so RViz / teleop / the apps see one stream name regardless of which camera model is
bolted to the wrist:
    /camera/wrist_<side>/image_raw              (+ .../camera_info)        rgb8
    /camera/wrist_<side>/depth/image_rect_raw   (+ .../depth/camera_info)  16UC1 (mm)
    /camera/wrist_<side>/depth/points           sensor_msgs/PointCloud2 (XYZRGB)

and fans the colour frames out to the bridge ``VideoServer`` (``wrist_<side>``
MJPEG/UDP stream) exactly like every other bridge camera — no ROS hop on the video
path. Depth is aligned to colour and the depth image + cloud publish only while
someone subscribes (RViz / a grasp node) — teleop never pays for them. The manager
parents ``wrist_<side>_camera_optical_frame`` at the URDF ``l_wrist``/``r_wrist``
link, so the cloud rides the arm in RViz.

Device selection: the D405 shares the pyrealsense2 SDK with the front D435, and a
serial-less ``rs.config`` grabs WHICHEVER RealSense enumerates first — plugging the
D405 in actually hijacked the front pipeline (observed: /camera/front/* streaming
with the D405's 18 mm baseline / 0.1 mm depth scale). So this driver claims ONLY a
device whose name contains "D405" (optionally pinned to a configured serial for the
future two-D405 rig), and realsense_d435.py symmetrically refuses D405s.

CameraInfo carries the REAL SDK intrinsics (the D405 is factory calibrated), unlike
the UVC wrist driver's pinhole placeholder. Missing camera => idle + periodic
reconnect, never fatal — same contract as the other bridge camera drivers.
"""

import threading
import time

import numpy as np

try:
    import pyrealsense2 as rs
    _RS_OK, _RS_ERR = True, None
except Exception as _e:                              # pragma: no cover
    rs = None
    _RS_OK, _RS_ERR = False, _e

from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from . import ros_msgs, rs_util

_CLOUD_STRIDE = 2     # subsample the cloud (1/stride^2 points) to bound bandwidth
_CLOUD_HZ = 6.0       # viewing overlay, not a mapping input: full-rate clouds
                      # (~12 MB/s) overflowed the 1 MB kernel UDP buffer and the
                      # cloud FLICKERED in RViz; 6 Hz renders solid.
_FIRST_FRAME_S = 8.0  # a started pipeline that yields no FIRST frame is not really
                      # ours (see the frame watchdog in _capture_loop) — re-claim it.
_STALL_S = 5.0        # mid-stream gap that means the link died (USB stall/unplug).


def _usb_hint(side):
    """Diagnostic tail for a stalled D405: is a RealSense still bound to uvcvideo?

    The RSUSB backend needs the device on usbfs (libusb detaches uvcvideo when it
    claims it). A D405 left on uvcvideo starts but delivers nothing, so SAY SO in
    the log instead of leaving a bare timeout for someone to reverse-engineer."""
    try:
        import glob
        import os
        held = []
        for dev in glob.glob("/sys/bus/usb/devices/*/product"):
            with open(dev) as f:
                if "405" not in f.read():
                    continue
            base = os.path.dirname(dev)
            drv = os.path.join(base + ":1.0", "driver")
            if os.path.islink(drv) and os.path.basename(os.readlink(drv)) == "uvcvideo":
                held.append(os.path.basename(base))
        if held:
            return (f" — a D405 is still bound to uvcvideo ({', '.join(held)}); the "
                    f"RSUSB backend needs it on usbfs. Retrying re-claims it.")
    except Exception:                                  # diagnostics must never throw
        pass
    return ""


def find_d405_serial(want_serial="", exclude=()):
    """Serial of the first known D405 (or the configured one, used verbatim).

    ``exclude`` = serials pinned to the OTHER wrist: an unpinned side must never
    resolve (or keep, see the claim check in _capture_loop) its sibling's camera,
    else a worker-restart race swaps the sides. Live scans (rs_util.scan) throw
    WHOLESALE while any RealSense in this process is busy streaming (RSUSB
    backend), so fall back to the registry cached from earlier successful scans.
    None => serial unknowable — the caller may still blind-start if sysfs shows
    a D405 present (mirror of realsense_d435)."""
    if want_serial:
        return want_serial
    try:
        devices = rs_util.scan()
    except Exception:
        devices = rs_util.registry()
    for serial, name in devices.items():
        if "D405" in name and serial not in exclude:
            return serial
    return None


class WristD405:
    """One D405 wrist camera: capture thread + graceful (re)connect, publishing the
    wrist_<side> colour contract to ROS and the bridge VideoServer."""

    def __init__(self, node, side, cfg, video=None):
        self.node = node
        self.side = side
        self.cfg = cfg
        self.video = video                           # bridge VideoServer (or None)
        self.stream = f"wrist_{side}"                # /camera/wrist_<side>/stream.mjpg
        self.log = node.get_logger()
        self.frame_id = f"wrist_{side}_camera_optical_frame"
        self.serial = str(cfg.get(f"{side}_serial", "") or "")
        other = "left" if side == "right" else "right"
        self.exclude = frozenset(
            s for s in (str(cfg.get(f"{other}_serial", "") or ""),) if s)
        self._stop = threading.Event()
        self._stamp_warned = False
        self._last_cloud = 0.0

        ns = f"/camera/wrist_{side}"
        self.pub_img = node.create_publisher(Image, f"{ns}/image_raw", 5)
        self.pub_info = node.create_publisher(CameraInfo, f"{ns}/camera_info", 5)
        self.pub_depth = node.create_publisher(Image, f"{ns}/depth/image_rect_raw", 5)
        self.pub_depth_info = node.create_publisher(
            CameraInfo, f"{ns}/depth/camera_info", 5)
        self.pub_cloud = node.create_publisher(PointCloud2, f"{ns}/depth/points", 5)

        if not _RS_OK:
            self.log.warn(f"pyrealsense2 unavailable ({_RS_ERR}); wrist_{side} (D405) IDLE.")
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._capture_loop()
            except Exception as e:                    # camera unplugged / SDK error
                self.log.warn(f"wrist_{self.side} D405 error ({e}); retrying in 3 s.",
                              throttle_duration_sec=10.0)
                time.sleep(3.0)

    def _capture_loop(self):
        serial = find_d405_serial(self.serial, self.exclude)
        if serial is None:
            # Serial unknowable (hot-plugged while another camera streams blocks
            # live scans). sysfs product names stay readable: if a D405 is
            # physically present, claim it with a serial-less start — resolution
            # skips busy devices — and VERIFY the claim below (never keep the
            # front D435). Otherwise idle-poll.
            if not any("D405" in p for p in rs_util.sysfs_products()):
                time.sleep(2.0)                      # camera absent; keep polling
                return
        c = self.cfg
        pipeline = rs.pipeline(rs_util.context())
        config = rs.config()
        if serial is not None:
            config.enable_device(serial)             # NEVER grab the front D435
        config.enable_stream(rs.stream.color, c["width"], c["height"],
                             rs.format.rgb8, c["fps"])
        with_depth = bool(c.get("d405_depth", True))
        if with_depth:
            config.enable_stream(rs.stream.depth, c["width"], c["height"],
                                 rs.format.z16, c["fps"])
        try:
            profile = pipeline.start(config)
        except Exception as e:
            # TWO-D405 RIG (2026-07-10, both wrists are D405s): with no serial
            # pin, BOTH per-process wrist workers resolve the same
            # first-D405-by-name — the loser's pinned start fails "busy"
            # forever. Fall back ONCE to a serial-less start: resolution skips
            # busy devices, so it lands on the FREE D405, and the name check
            # below still rejects a non-D405. Pin CAM_WRIST_{L,R}_SERIAL for
            # deterministic sides.
            if not self.serial and serial is not None:
                self.log.warn(f"wrist_{self.side} D405 {serial} busy/unusable "
                              f"({e}); blind-starting on the next free D405.")
                serial = None
                config = rs.config()
                config.enable_stream(rs.stream.color, c["width"], c["height"],
                                     rs.format.rgb8, c["fps"])
                if with_depth:
                    config.enable_stream(rs.stream.depth, c["width"], c["height"],
                                         rs.format.z16, c["fps"])
                profile = pipeline.start(config)
            elif with_depth:
                self.log.warn(f"wrist_{self.side} D405 refused color+depth ({e}); "
                              "retrying color-only.")
                with_depth = False
                config = rs.config()
                if serial is not None:
                    config.enable_device(serial)
                config.enable_stream(rs.stream.color, c["width"], c["height"],
                                     rs.format.rgb8, c["fps"])
                profile = pipeline.start(config)
            else:
                raise
        dev = profile.get_device()
        dev_name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        rs_util.remember(serial, dev_name)
        if "D405" not in dev_name or serial in self.exclude:
            # Blind start grabbed a non-D405 (the front cam was idle) or the
            # OTHER wrist's pinned D405 — release it for its rightful owner
            # (realsense_d435.py / the sibling wrist worker) and keep waiting.
            pipeline.stop()
            time.sleep(2.0)
            return
        depth_scale = 0.0
        align = None
        if with_depth:
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)        # depth in the colour frame
        self.log.info(
            f"wrist_{self.side} D405 (serial {serial}) streaming "
            f"{c['width']}x{c['height']}@{c['fps']} "
            f"(depth={'on scale=%.5f' % depth_scale if with_depth else 'OFF'}).")
        # FRAME WATCHDOG (2026-07-13). pipeline.start() can SUCCEED on a device the
        # kernel's uvcvideo still owns — RSUSB/libusb only detaches uvcvideo when it
        # really claims the device — and the handle then delivers NOTHING while the
        # blocking wait_for_frames() never returns. Hit live: the LEFT D405 sat on
        # uvcvideo (the right one on usbfs), so its worker logged "streaming", idled
        # at ~4% CPU, published ZERO frames, and raised no error — the bring-up panel
        # just read "waiting" forever with nothing in the logs to explain it.
        # So NEVER wait unbounded: poll with a deadline and raise on a stall. The
        # outer _run() retry then tears the pipeline down and re-starts it, which
        # re-claims the device (uvcvideo -> usbfs) and heals the camera unattended.
        got_first = False
        last = time.time()
        try:
            while not self._stop.is_set():
                ok, frames = pipeline.try_wait_for_frames(500)   # ms; never blocks forever
                if not ok:
                    limit = _FIRST_FRAME_S if not got_first else _STALL_S
                    if time.time() - last > limit:
                        raise RuntimeError(
                            f"no {'first ' if not got_first else ''}frame for "
                            f"{limit:.0f}s{_usb_hint(self.side)}")
                    continue
                got_first, last = True, time.time()
                if align is not None:
                    frames = align.process(frames)
                color = frames.get_color_frame()
                if not color:
                    continue
                depth = frames.get_depth_frame() if with_depth else None
                self._publish(color, depth, depth_scale, self._frame_stamp(color))
        finally:
            pipeline.stop()

    def _publish(self, color, depth, depth_scale, stamp):
        intr = color.profile.as_video_stream_profile().intrinsics
        K = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        color_np = np.asanyarray(color.get_data())               # (H,W,3) rgb8

        # ── sink 1: network video (direct, no ROS) — same as every bridge cam ──
        if self.video is not None and self.video.wants(self.stream):
            self.video.push(self.stream, np.ascontiguousarray(color_np[:, :, ::-1]))

        # ── sink 2: ROS image topics ────────────────────────────────────────────
        self.pub_img.publish(ros_msgs.image_msg(color_np, "rgb8", self.frame_id, stamp))
        self.pub_info.publish(ros_msgs.camera_info_msg(
            intr.width, intr.height, K, self.frame_id, stamp))

        # Depth image + cloud only while subscribed (RViz / a grasp node); teleop
        # never pays for them.
        if depth is None:
            return
        depth_np = None
        if (self.pub_depth.get_subscription_count() > 0
                or self.pub_depth_info.get_subscription_count() > 0):
            depth_np = np.asanyarray(depth.get_data())           # (H,W) uint16
            depth_mm = (depth_np.astype(np.float32) * depth_scale * 1000.0
                        ).astype(np.uint16)
            self.pub_depth.publish(ros_msgs.image_msg(
                depth_mm, "16UC1", self.frame_id, stamp))
            self.pub_depth_info.publish(ros_msgs.camera_info_msg(
                intr.width, intr.height, K, self.frame_id, stamp))
        if (self.pub_cloud.get_subscription_count() > 0
                and (time.monotonic() - self._last_cloud) >= 1.0 / _CLOUD_HZ):
            self._last_cloud = time.monotonic()
            if depth_np is None:
                depth_np = np.asanyarray(depth.get_data())
            self._publish_cloud(color_np, depth_np, depth_scale, K, stamp)

    def _publish_cloud(self, color_np, depth_np, depth_scale, K, stamp):
        fx, fy, cx, cy = K
        s = _CLOUD_STRIDE
        z = depth_np[::s, ::s].astype(np.float32) * depth_scale     # metres
        h, w = z.shape
        us, vs = np.meshgrid(np.arange(w) * s, np.arange(h) * s)
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        valid = z > 0.0
        xyz = np.stack((x[valid], y[valid], z[valid]), axis=-1)
        rgb = color_np[::s, ::s][valid]
        self.pub_cloud.publish(
            ros_msgs.pointcloud2_msg(xyz, self.frame_id, stamp, rgb=rgb))

    def _frame_stamp(self, frame):
        """SENSOR-side (host-clock) stamp when available, else node clock — same
        rationale as realsense_d435.py: publish-time stamps turn GIL stalls into
        timestamp jitter."""
        try:
            if frame.get_frame_timestamp_domain() in (
                    rs.timestamp_domain.global_time,
                    rs.timestamp_domain.system_time):
                t_ms = frame.get_timestamp()
                if abs(t_ms / 1000.0 - time.time()) < 1.0:
                    sec = int(t_ms // 1000)
                    return TimeMsg(sec=sec, nanosec=int((t_ms - sec * 1000.0) * 1e6))
        except Exception:
            pass
        if not self._stamp_warned:
            self._stamp_warned = True
            self.log.warn(f"wrist_{self.side} D405 global-time stamps unavailable; "
                          "using node-clock stamps.")
        return self.node.get_clock().now().to_msg()

    def stop(self):
        self._stop.set()
