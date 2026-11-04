"""wrist_cams.py — V4L2/MJPEG driver for the two Arducam Global-Shutter wrist cams.

Publishes a per-side colour contract for teleop / hand-eye viewing:
    /camera/wrist_left/image_raw    (+ .../wrist_left/camera_info)    rgb8
    /camera/wrist_right/image_raw   (+ .../wrist_right/camera_info)   rgb8

The two cameras are ELECTRONICALLY IDENTICAL (same USB VID:PID 32e4:0144 AND the same
serial 01.00.00), so they cannot be distinguished by /dev/videoN or by-id — only by the
physical USB hub PORT. The convention (decided) is hub port 1-4.1.1 = RIGHT, 1-4.1.2 =
LEFT; the 99-mabel-wrist-cams.udev rule pins those ports to /dev/wrist_{right,left}. Each
side opens its configured device (falling back to the by-path node if the symlink is
absent), and degrades gracefully: a missing camera => idle + periodic reconnect, never
fatal — exactly like the D435 / ZED drivers.

Intrinsics are a rough pinhole placeholder (these cams are wide-angle/fisheye and are NOT
yet calibrated); camera_info is published so image_transport/RViz are happy, but do not
use it for metric work until a real calibration is run.
"""

import threading
import time

import numpy as np

try:
    import cv2
    _CV_OK, _CV_ERR = True, None
except Exception as _e:                              # pragma: no cover
    cv2 = None
    _CV_OK, _CV_ERR = False, _e

from sensor_msgs.msg import Image, CameraInfo

from . import ros_msgs


class _WristSide:
    """One wrist camera: its own capture thread + graceful (re)connect.

    Each captured frame fans out to TWO sinks — the ROS image topic (DDS graph) and
    the bridge's MJPEG ``VideoServer`` (network), the latter straight as JPEG with
    no ROS hop so the video path stays fast (see video_server.py)."""

    def __init__(self, node, side, device, fallback, cfg, video=None):
        self.node = node
        self.side = side
        self.device = device
        self.fallback = fallback
        self.cfg = cfg
        self.video = video                           # bridge VideoServer (or None)
        self.stream = f"wrist_{side}"                # /camera/wrist_<side>/stream.mjpg
        self.log = node.get_logger()
        self.frame_id = f"wrist_{side}_camera_optical_frame"
        self._stop = threading.Event()

        ns = f"/camera/wrist_{side}"
        self.pub_img = node.create_publisher(Image, f"{ns}/image_raw", 5)
        self.pub_info = node.create_publisher(CameraInfo, f"{ns}/camera_info", 5)

        if not _CV_OK:
            self.log.warn(f"opencv unavailable ({_CV_ERR}); wrist_{side} IDLE.")
            return
        threading.Thread(target=self._run, daemon=True).start()

    # ---- device resolution --------------------------------------------------
    def _resolve(self):
        """Prefer the udev symlink; fall back to the by-path node if absent."""
        import os
        for path in (self.device, self.fallback):
            if path and os.path.exists(path):
                return path
        return None

    def _open(self, path):
        c = self.cfg
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        # MJPG @ requested geometry — YUYV on these cams caps at 10 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        cap.set(cv2.CAP_PROP_FPS, c["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # always the freshest frame (low latency)
        # Exposure: these cams default to AUTO (aperture-priority). In dim indoor light
        # auto-exposure stretches the exposure time to ~0.5 s, collapsing the framerate
        # to ~2 fps even though the cam can do 30. Force MANUAL exposure with a short
        # time (+ gain to keep the image bright) so the framerate holds. Order matters:
        # auto_exposure MUST go to manual BEFORE exposure_time_absolute is writable.
        ae = int(c.get("auto_exposure", 1))         # 1=manual, 3=aperture-priority auto
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(ae))
        if ae == 1:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(c.get("exposure", 120)))  # x100us -> 12 ms
            cap.set(cv2.CAP_PROP_GAIN, float(c.get("gain", 180)))          # 16..255
        return cap

    # ---- capture loop -------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                self._capture_loop()
            except Exception as e:
                self.log.warn(f"wrist_{self.side} error ({e}); retrying in 3 s.",
                              throttle_duration_sec=10.0)
                time.sleep(3.0)

    def _capture_loop(self):
        path = self._resolve()
        if path is None:
            time.sleep(2.0)                          # camera absent; let service retry
            return
        cap = self._open(path)
        if cap is None:
            self.log.warn(f"wrist_{self.side}: cannot open {path}; retry in 3 s.",
                          throttle_duration_sec=10.0)
            time.sleep(3.0)
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.cfg["width"]
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.cfg["height"]
        K = (float(w), float(w), w / 2.0, h / 2.0)   # rough pinhole placeholder (uncalibrated)
        self.log.info(f"wrist_{self.side} streaming {w}x{h}@{self.cfg['fps']} on {path}.")
        misses = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 30:                  # camera dropped: reopen
                        self.log.warn(f"wrist_{self.side}: stream stalled; reopening.")
                        return
                    time.sleep(0.01)
                    continue
                misses = 0
                # ── sink 1: network video (direct, no ROS) ──────────────────
                # Push raw BGR; the VideoServer downscales + fans out to MJPEG /
                # UDP-JPEG / UDP-H.264, only when a client is watching.
                if self.video is not None and self.video.wants(self.stream):
                    self.video.push(self.stream, frame)
                # ── sink 2: ROS image topic ─────────────────────────────────
                stamp = self.node.get_clock().now().to_msg()
                rgb = np.ascontiguousarray(frame[:, :, ::-1])   # BGR (cv2) -> RGB
                self.pub_img.publish(ros_msgs.image_msg(rgb, "rgb8", self.frame_id, stamp))
                self.pub_info.publish(
                    ros_msgs.camera_info_msg(w, h, K, self.frame_id, stamp))
        finally:
            cap.release()

    def stop(self):
        self._stop.set()


class WristCameras:
    """Owns BOTH wrist cameras (left + right). One capture thread per side; each
    fans out to ROS + the bridge ``VideoServer`` (network MJPEG).

    Per-side driver dispatch on cfg['<side>_kind']: "uvc" = the Arducam V4L2 driver
    above; "d405" = the Intel RealSense D405 (wrist_d405.py, pyrealsense2 — colour +
    depth + cloud on the same /camera/wrist_<side>/* contract)."""

    def __init__(self, node, cfg, video=None):
        self.sides = []
        for side in ("left", "right"):
            kind = str(cfg.get(f"{side}_kind", "uvc")).lower()
            if kind == "d405":
                from .wrist_d405 import WristD405
                self.sides.append(WristD405(node, side, cfg, video=video))
            else:
                self.sides.append(
                    _WristSide(node, side, cfg[f"{side}_device"],
                               cfg[f"{side}_fallback"], cfg, video=video))

    def stop(self):
        for s in self.sides:
            s.stop()
