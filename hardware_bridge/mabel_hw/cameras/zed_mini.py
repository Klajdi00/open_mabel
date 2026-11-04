"""zed_mini.py — native pyzed driver for the ZED-M head camera.

Publishes the existing sim/ROS contract so SLAM/nvblox runs unchanged:
    /camera/zed/left/image_raw   (+ .../left/camera_info)    rgb8
    /camera/zed/right/image_raw  (+ .../right/camera_info)   rgb8
    /camera/zed/depth/image_rect_raw                          32FC1 (metres)
    /camera/zed/depth/points                                  PointCloud2 (XYZRGB)

Left/depth/cloud share ``zed_left_camera_optical_frame`` (the manager publishes
head -> zed_camera_link -> zed_*_camera_optical_frame). pos-tracking/odometry stay OFF —
the robot owns odometry. Runs its own capture thread; absent camera => idle + reconnect.
"""

import glob
import threading
import time

import numpy as np

try:
    import pyzed.sl as sl
    _ZED_OK, _ZED_ERR = True, None
except Exception as _e:                              # pragma: no cover
    sl = None
    _ZED_OK, _ZED_ERR = False, _e

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from . import ros_msgs

LEFT_FRAME = "zed_left_camera_optical_frame"
RIGHT_FRAME = "zed_right_camera_optical_frame"
# Cloud sizing: the RViz cloud is a VIEWING overlay, not a mapping input. At the
# old stride 2 / full frame rate the HD720 cloud was ~3.7 MB × 30 Hz ≈ 110 MB/s —
# a single DDS sample larger than the kernel's 1 MB rmem_max, so fragments were
# dropped constantly and the cloud FLICKERED in RViz. Stride 4 @ 5 Hz ≈ 4.6 MB/s
# with every sample under the buffer: solid rendering.
_CLOUD_STRIDE = 4
_CLOUD_HZ = 5.0
_ZED_USB_VID = "2b03"     # Stereolabs vendor id (ZED / ZED-M / ZED 2 family)


_ZED_USB_PID_VIDEO = "f682"   # the ZED-M VIDEO device. ⚠ Do NOT gate on the vendor
                              # id alone: the HID interface (f681) stays enumerated
                              # when the video device is gone (half-seated USB-C /
                              # mid-replug), and opening against that state yields a
                              # camera whose grabs fail forever.


def _zed_on_usb():
    """Cheap sysfs scan: is the ZED VIDEO device physically enumerated on USB?

    Guard for the reconnect loop: ``sl.Camera.open()`` on an ABSENT camera is a
    ~0.5 s NATIVE call that holds the GIL; and open against a HALF-present ZED
    (HID up, video down) succeeds-then-never-grabs. Reading sysfs is pure-Python,
    microseconds.
    """
    for p in glob.glob("/sys/bus/usb/devices/*/idProduct"):
        try:
            with open(p) as f:
                if f.read().strip().lower() == _ZED_USB_PID_VIDEO:
                    return True
        except OSError:
            continue
    return False


class ZedMini:
    def __init__(self, node, cfg, video=None):
        self.node = node
        self.cfg = cfg
        self.video = video                # bridge VideoServer (or None)
        # head tile: served under BOTH names the GUI/apps look for (main, head_left),
        # mirroring the sensors node — same physical ZED-M left eye.
        self.streams = ("main", "head_left")
        self.log = node.get_logger()
        self._stop = threading.Event()
        self.connected = False            # True while the ZED is open + grabbing
                                          # (drives the video 'main'→front fallback)

        self.pub_left = node.create_publisher(Image, "/camera/zed/left/image_raw", 5)
        self.pub_right = node.create_publisher(Image, "/camera/zed/right/image_raw", 5)
        self.pub_left_info = node.create_publisher(
            CameraInfo, "/camera/zed/left/camera_info", 5)
        self.pub_right_info = node.create_publisher(
            CameraInfo, "/camera/zed/right/camera_info", 5)
        self.pub_depth = node.create_publisher(
            Image, "/camera/zed/depth/image_rect_raw", 5)
        self.pub_cloud = node.create_publisher(PointCloud2, "/camera/zed/depth/points", 5)

        if not _ZED_OK:
            self.log.warn(f"pyzed unavailable ({_ZED_ERR}); head ZED IDLE.")
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 3.0
        while not self._stop.is_set():
            # Never call the (GIL-holding, ~0.5 s) native zed.open unless a
            # Stereolabs device is actually enumerated on USB — an absent/loose
            # ZED otherwise stalls the D435 IR stream ~550 ms every retry and
            # wrecks cuVSLAM tracking. sysfs poll is free; hot-plug still works.
            if not _zed_on_usb():
                self.log.warn("ZED not on USB; head cam idle (polling).",
                              throttle_duration_sec=60.0)
                time.sleep(3.0)
                backoff = 3.0
                continue
            try:
                self._capture_loop()
                backoff = 3.0                     # streamed OK; quick reconnect
            except Exception as e:
                # RECYCLE THE WHOLE WORKER PROCESS: after a failed session (grabs
                # died / open failed with the video device present) the ZED SDK's
                # in-process state (USB handles, CUDA context) is suspect — an
                # in-process reopen was observed to succeed and then never grab.
                # The camera manager respawns this worker fresh in ~3 s; a clean
                # SDK re-registers a replugged camera reliably.
                self.log.warn(f"ZED session failed ({e}) — recycling the zed "
                              "worker process for a clean SDK re-open.")
                import os
                os._exit(23)

    def _capture_loop(self):
        c = self.cfg
        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, c["resolution"], sl.RESOLUTION.HD720)
        init.camera_fps = int(c["fps"])
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = (sl.DEPTH_MODE.PERFORMANCE if c["depth"]
                           else sl.DEPTH_MODE.NONE)
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"zed.open -> {status}")

        K_left, K_right, w, h = self._intrinsics(zed)
        runtime = sl.RuntimeParameters()
        left, right, depth, cloud = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
        self.log.info(f"ZED-M streaming {c['resolution']}@{c['fps']} (depth={c['depth']}).")
        self.connected = True
        try:
            misses = 0
            last_cloud = 0.0
            while not self._stop.is_set():
                # Depth is the ZED's frame-rate ceiling (~13 Hz with, ~30 without):
                # compute it ONLY while someone actually subscribes to the depth
                # image or the cloud (same demand-gating as the D405 wrist cam) —
                # video-only consumers (apps' head view) get the full 30 Hz.
                want_depth = c["depth"] and self.pub_depth.get_subscription_count() > 0
                want_cloud = c["depth"] and self.pub_cloud.get_subscription_count() > 0
                try:
                    runtime.enable_depth = bool(want_depth or want_cloud)
                except AttributeError:
                    pass                             # older SDK: depth always on
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    # A dead camera (unplugged mid-stream) fails EVERY grab — and
                    # each failed grab BLOCKS ~9 s inside the SDK, so bail after
                    # TWO misses (~18 s worst case, usually much less) and let
                    # _run() recycle the process. A bare `continue` here once spun
                    # forever on the stale handle ("stale" after replug).
                    misses += 1
                    if misses >= 2:
                        raise RuntimeError(
                            "grab failing repeatedly (unplugged?) — recycling")
                    time.sleep(0.1)
                    continue
                misses = 0
                stamp = self.node.get_clock().now().to_msg()
                zed.retrieve_image(left, sl.VIEW.LEFT)
                zed.retrieve_image(right, sl.VIEW.RIGHT)
                # network sink (no ROS hop) — push raw BGR once; the VideoServer
                # downscales + fans out (one shared JPEG / H.264 for both head stream
                # names), only if a client is watching either.
                if self.video is not None:
                    want_main = any(self.video.wants(s) for s in self.streams)
                    want_sbs = self.video.wants("head_sbs")   # :9620 dedicated hi-res
                    want_stereo = self.video.wants("head_stereo")
                    if want_main or want_sbs or want_stereo:
                        bgr = np.ascontiguousarray(left.get_data()[:, :, :3])  # BGRA→BGR
                        if want_main:
                            self.video.push(self.streams, bgr)
                        if want_sbs or want_stereo:
                            # Side-by-side stereo (left|right) from THIS grab: the
                            # ZED SDK's VIEW.LEFT/RIGHT are RECTIFIED and from the
                            # same exposure, so the two halves can never be out of
                            # step — the headset's comfort depends on exactly that
                            # (two separate eye streams drift by whole frames).
                            # 'head_stereo' (the headset's source) goes out at 2×
                            # the net width so each eye keeps full net resolution;
                            # one encode serves both names.
                            rbgr = np.ascontiguousarray(right.get_data()[:, :, :3])
                            sbs = np.hstack((bgr, rbgr))
                            names = tuple(n for n, w in (("head_stereo", want_stereo),
                                                         ("head_sbs", want_sbs)) if w)
                            net = getattr(self.video, "net_width", 0) or getattr(
                                self.video, "_net_width", 640)
                            self.video.push(names, sbs, width=2 * int(net))
                self._pub_image(self.pub_left, left, LEFT_FRAME, stamp)
                self._pub_image(self.pub_right, right, RIGHT_FRAME, stamp)
                if K_left:
                    self.pub_left_info.publish(
                        ros_msgs.camera_info_msg(w, h, K_left, LEFT_FRAME, stamp))
                if K_right:
                    self.pub_right_info.publish(
                        ros_msgs.camera_info_msg(w, h, K_right, RIGHT_FRAME, stamp))
                if want_depth or want_cloud:
                    if want_depth:
                        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                        self._pub_depth(depth, stamp)
                    if want_cloud and (time.monotonic() - last_cloud) >= 1.0 / _CLOUD_HZ:
                        last_cloud = time.monotonic()
                        zed.retrieve_measure(cloud, sl.MEASURE.XYZRGBA)
                        self._pub_cloud(cloud, stamp)
        finally:
            self.connected = False
            zed.close()

    def _intrinsics(self, zed):
        """Best-effort left/right pinhole intrinsics across SDK versions."""
        try:
            info = zed.get_camera_information()
            calib = info.camera_configuration.calibration_parameters
            res = info.camera_configuration.resolution
            w, h = int(res.width), int(res.height)
            lc, rc = calib.left_cam, calib.right_cam
            return ((lc.fx, lc.fy, lc.cx, lc.cy),
                    (rc.fx, rc.fy, rc.cx, rc.cy), w, h)
        except Exception as e:                        # pragma: no cover
            self.log.warn(f"ZED intrinsics unavailable ({e}); camera_info omitted.")
            return None, None, 0, 0

    def _pub_image(self, pub, mat, frame, stamp):
        bgra = mat.get_data()                         # (H,W,4) uint8 BGRA
        rgb = np.ascontiguousarray(bgra[:, :, [2, 1, 0]])
        pub.publish(ros_msgs.image_msg(rgb, "rgb8", frame, stamp))

    def _pub_depth(self, mat, stamp):
        depth = np.ascontiguousarray(mat.get_data().astype(np.float32))   # metres, may be NaN
        self.pub_depth.publish(ros_msgs.image_msg(depth, "32FC1", LEFT_FRAME, stamp))

    def _pub_cloud(self, mat, stamp):
        data = mat.get_data()                         # (H,W,4) float32: x,y,z,rgba
        s = _CLOUD_STRIDE
        sub = data[::s, ::s].reshape(-1, 4)
        xyz = sub[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        # MEASURE.XYZRGBA packs the colour float's bytes as r,g,b,a (only the
        # IMAGE views are BGRA) — Stereolabs' own ROS wrapper switches to
        # XYZBGRA for PointCloud2 for exactly this reason. A [2,1,0] swap here
        # ships BGR and every client renders yellow as blue.
        rgba_u8 = sub[finite, 3].view(np.uint8).reshape(-1, 4)
        rgb = rgba_u8[:, :3]
        self.pub_cloud.publish(ros_msgs.pointcloud2_msg(xyz, LEFT_FRAME, stamp, rgb=rgb))

    def stop(self):
        self._stop.set()
