"""Stereolabs ZED-M backend (ZED SDK Python API, pyzed.sl).

Owns the device over USB and returns the left colour image + metric depth + a
coloured XYZ point cloud (GPU-computed by the SDK) + the ZED-M's IMU
(accel + gyro + fused orientation). No ROS.

HARD REQUIREMENT: the ZED-M must be on a USB-3 SuperSpeed link. On USB-2 it
enumerates HID-only (no video) and `open()` fails -- we surface a clear hint to
move it to a SuperSpeed port. Run `sensors/usb_check.sh` to verify the link.
"""
from __future__ import annotations

import os
import time
import urllib.request

import numpy as np

from .base import CameraBase, CameraError
from .frame import CameraFrame, CameraIntrinsics, ImuSample

try:
    import pyzed.sl as sl
except Exception:                     # pragma: no cover - guarded for nice error
    sl = None

_USB2_HINT = ("ZED on USB-2 (HID-only, no video) or not detected -- plug it into "
              "a USB-3 SuperSpeed port with the ZED's micro-B SS cable, then run "
              "sensors/usb_check.sh")


class ZedCamera(CameraBase):
    name = "zedm"
    model = "Stereolabs ZED-M"

    # resolution name -> sl.RESOLUTION (resolved lazily; sl may be None)
    _RES = {"HD2K": "HD2K", "HD1080": "HD1080", "HD720": "HD720", "VGA": "VGA"}

    def __init__(self, resolution: str = "HD720", fps: int = 30,
                 depth_mode: str = "ULTRA", cloud_stride: int = 2,
                 min_depth_m: float = 0.2, settings_path: str | None = None):
        super().__init__()
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.cloud_stride = max(1, cloud_stride)
        self.min_depth_m = min_depth_m
        # user-writable dir for the per-camera factory calibration .conf (the
        # SDK's default /usr/local/zed/settings is root-owned). Auto-populated.
        self.settings_path = settings_path or os.path.expanduser(
            "~/.cache/mabel/zed_settings")
        self._zed = None
        self._runtime = None
        self._mat_img = self._mat_depth = self._mat_cloud = None
        self._sensors = None

    # -- presence ----------------------------------------------------------
    def is_present(self) -> bool:
        """True only if the SDK enumerates a streamable ZED (i.e. on USB-3)."""
        if sl is None:
            return False
        try:
            return len(sl.Camera.get_device_list()) > 0
        except Exception:
            return False

    def _ensure_calibration(self) -> None:
        """Make sure this camera's factory calibration .conf is on disk.

        The SDK normally downloads it to the root-owned /usr/local/zed/settings;
        here we drop it in a user-writable dir and point the SDK at it via
        optional_settings_path, so no sudo is needed on first run.
        """
        try:
            devs = sl.Camera.get_device_list()
        except Exception:
            devs = []
        if not devs:
            return                                 # let open() raise the USB hint
        serial = devs[0].serial_number
        os.makedirs(self.settings_path, mode=0o755, exist_ok=True)
        conf = os.path.join(self.settings_path, f"SN{serial}.conf")
        if os.path.exists(conf) and os.path.getsize(conf) > 0:
            return
        url = f"https://calib.stereolabs.com/?SN={serial}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:   # follows redirects
                data = r.read()
            if data.lstrip().startswith(b"[") and len(data) > 200:
                with open(conf, "wb") as f:
                    f.write(data)
        except Exception:
            pass                                   # open() surfaces a clear hint below

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if sl is None:
            raise CameraError("pyzed not installed",
                              hint="run sensors/install_deps.sh (needs the ZED SDK)")

        self._ensure_calibration()
        self._zed = sl.Camera()
        init = sl.InitParameters()
        init.optional_settings_path = self.settings_path
        init.camera_resolution = getattr(sl.RESOLUTION, self._RES.get(self.resolution, "HD720"))
        init.camera_fps = self.fps
        init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode, sl.DEPTH_MODE.NEURAL)
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE   # +X right, +Y down, +Z fwd
        init.depth_minimum_distance = self.min_depth_m
        init.sdk_verbose = 0                                  # quiet -> clean dashboard

        status = self._zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self._zed = None
            hint = _USB2_HINT
            if "CALIBRATION" in str(status).upper():
                hint = (f"factory calibration missing -- fetch it with: curl -sSL "
                        f"'https://calib.stereolabs.com/?SN=<serial>' -o "
                        f"{self.settings_path}/SN<serial>.conf  (needs internet)")
            raise CameraError(f"ZED open failed: {status}", hint=hint)

        self._runtime = sl.RuntimeParameters()
        self._mat_img = sl.Mat()
        self._mat_depth = sl.Mat()
        self._mat_cloud = sl.Mat()
        self._sensors = sl.SensorsData()

        info = self._zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters
        left = calib.left_cam
        try:
            baseline = float(calib.get_camera_baseline())
            # SDK returns the baseline in the configured coordinate unit (METER);
            # guard the odd build that hands back millimetres.
            baseline_m = baseline / 1000.0 if baseline > 1.0 else baseline
        except Exception:
            baseline_m = None
        self._intrinsics = CameraIntrinsics(
            width=int(left.image_size.width), height=int(left.image_size.height),
            fx=float(left.fx), fy=float(left.fy), cx=float(left.cx), cy=float(left.cy),
            model="pinhole", coeffs=tuple(float(x) for x in left.disto),
            baseline_m=baseline_m)
        self._open = True

    def _read_imu(self) -> ImuSample | None:
        try:
            if self._zed.get_sensors_data(self._sensors,
                                          sl.TIME_REFERENCE.IMAGE) != sl.ERROR_CODE.SUCCESS:
                return None
            imu = self._sensors.get_imu_data()
            a = imu.get_linear_acceleration()
            g = imu.get_angular_velocity()             # deg/s -> rad/s
            ori = imu.get_pose().get_orientation().get()
            return ImuSample(
                stamp=time.monotonic(),
                accel=(float(a[0]), float(a[1]), float(a[2])),
                gyro=tuple(np.deg2rad([g[0], g[1], g[2]]).astype(float)),
                orientation=(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])))
        except Exception:
            return None

    def read(self) -> CameraFrame | None:
        if not self._open:
            raise CameraError("read() before open()")
        if self._zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        self._zed.retrieve_image(self._mat_img, sl.VIEW.LEFT)
        bgra = self._mat_img.get_data()                 # HxWx4 uint8 BGRA
        rgb = np.ascontiguousarray(bgra[:, :, :3])      # -> BGR

        self._zed.retrieve_measure(self._mat_depth, sl.MEASURE.DEPTH)
        depth = np.asarray(self._mat_depth.get_data(), dtype=np.float32)  # metres, NaN=invalid

        self._zed.retrieve_measure(self._mat_cloud, sl.MEASURE.XYZRGBA)
        cloud = np.asarray(self._mat_cloud.get_data())  # HxWx4 float32 (x,y,z, packed rgba)
        s = self.cloud_stride
        cloud = cloud[::s, ::s]
        xyz = cloud[:, :, :3].reshape(-1, 3)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite].astype(np.float32)
        # 4th channel is RGBA packed into a float32 -> view bytes as BGR for OpenCV
        rgba = cloud[:, :, 3].reshape(-1)[finite].copy().view(np.uint8).reshape(-1, 4)
        prgb = rgba[:, [2, 1, 0]].copy()                # RGBA -> BGR

        return CameraFrame(
            name=self.name, stamp=time.monotonic(), rgb=rgb, depth_m=depth,
            points_xyz=xyz, points_rgb=prgb, intrinsics=self._intrinsics,
            imu=self._read_imu())

    def close(self) -> None:
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception:
                pass
        self._zed = None
        self._open = False
