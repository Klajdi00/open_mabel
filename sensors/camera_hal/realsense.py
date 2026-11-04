"""Intel RealSense D435i backend (pyrealsense2).

Owns the device over USB and returns colour + metric depth + an organized
XYZRGB point cloud + the D435i's built-in IMU (accel + gyro). No ROS.

The cloud is deprojected straight from the depth frame after it's aligned to
colour, so depth pixel (i,j) maps to colour pixel (i,j) -- giving exact per-point
colour for free without RealSense's texcoord dance.
"""
from __future__ import annotations

import time

import numpy as np

from .base import CameraBase, CameraError
from .frame import CameraFrame, CameraIntrinsics, ImuSample

try:
    import pyrealsense2 as rs
except Exception:                     # pragma: no cover - import guarded for nice error
    rs = None


class RealSenseCamera(CameraBase):
    name = "realsense"
    model = "Intel RealSense D435i"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_imu: bool = True, serial: str | None = None,
                 cloud_stride: int = 2, enable_ir: bool = True):
        """`cloud_stride` subsamples the cloud (2 -> 1/4 the points) for live viz.

        `enable_ir` also streams the two depth-module IR imagers (infra 1 & 2) as a
        rectified mono stereo pair -- the input cuVSLAM tracks on (visual_slam
        navigation env). It costs only two extra mono8 streams over the same USB3
        link; turn it off (--no-ir) for an MJPEG-only bring-up that never runs SLAM.
        """
        super().__init__()
        self.width, self.height, self.fps = width, height, fps
        self.enable_imu = enable_imu
        self.enable_ir = enable_ir
        self.serial = serial
        self.cloud_stride = max(1, cloud_stride)
        self._pipe = None
        self._align = None
        self._depth_scale = 1.0
        self._uv = None               # cached pixel grid for deprojection
        self._last_imu = ImuSample(stamp=0.0)
        self._ir_intrinsics = None    # shared rectified IR pinhole (set in open())
        self._ir_baseline_m = None    # IR stereo baseline left->right (m)

    # -- presence ----------------------------------------------------------
    @staticmethod
    def list_serials() -> list[str]:
        if rs is None:
            return []
        try:
            return [d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()]
        except Exception:
            return []

    def is_present(self) -> bool:
        return bool(self.list_serials())

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if rs is None:
            raise CameraError("pyrealsense2 not installed",
                              hint="run sensors/install_deps.sh")
        if not self.is_present():
            raise CameraError("no RealSense device found",
                              hint="check USB / `rs-enumerate-devices`")

        self._pipe = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if self.enable_ir:
            # The two depth-module imagers as mono8, SAME geometry as depth so the
            # pair is the factory-rectified stereo cuVSLAM expects. infrared,1 is the
            # LEFT imager (== depth origin); infrared,2 the right.
            cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
            cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
        if self.enable_imu:
            # D435i IMU: accel + gyro are separate motion streams at their own rates.
            try:
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
            except Exception:
                self.enable_imu = False   # unit without IMU -> just skip it

        try:
            profile = self._pipe.start(cfg)
        except Exception as e:
            # Most commonly: IMU enabled on a unit that reports "HID Motion Sensor
            # Failure". Retry once without the IMU rather than failing the camera.
            if self.enable_imu:
                self.enable_imu = False
                cfg2 = rs.config()
                if self.serial:
                    cfg2.enable_device(self.serial)
                cfg2.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                cfg2.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                if self.enable_ir:
                    cfg2.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
                    cfg2.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
                profile = self._pipe.start(cfg2)
            else:
                raise CameraError(f"RealSense start failed: {e}")

        self._align = rs.align(rs.stream.color)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # intrinsics from the (colour) stream the cloud is expressed in
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics = CameraIntrinsics(
            width=intr.width, height=intr.height,
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
            model=str(intr.model).split('.')[-1], coeffs=tuple(intr.coeffs),
            depth_scale_m=self._depth_scale)

        # pixel grid for vectorized deprojection (u-cx)/fx etc.
        u = np.arange(intr.width, dtype=np.float32)
        v = np.arange(intr.height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        self._uv = ((uu - intr.ppx) / intr.fx, (vv - intr.ppy) / intr.fy)

        # IR stereo calibration straight from the device. The factory-rectified IR
        # pair shares ONE rectified pinhole (no distortion -> coeffs all zero, what
        # cuVSLAM assumes with rectified_images:=true); the only stereo unknown is
        # the baseline, the |x| translation from the left imager to the right.
        if self.enable_ir:
            try:
                ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
                ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
                k = ir1.get_intrinsics()
                ext = ir1.get_extrinsics_to(ir2)         # left -> right imager
                self._ir_baseline_m = float(abs(ext.translation[0]))
                self._ir_intrinsics = CameraIntrinsics(
                    width=k.width, height=k.height,
                    fx=k.fx, fy=k.fy, cx=k.ppx, cy=k.ppy,
                    model='pinhole', coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
                    baseline_m=self._ir_baseline_m)
            except Exception:
                # Some firmwares/units don't expose the IR pair -> degrade to no IR
                # rather than failing the whole camera (color+depth still stream).
                self.enable_ir = False
                self._ir_intrinsics = None
                self._ir_baseline_m = None
        self._open = True

    def _pump_imu(self, frames) -> None:
        if not self.enable_imu:
            return
        # IMU is best-effort: never let a motion-frame quirk kill the image stream.
        try:
            accel, gyro = self._last_imu.accel, self._last_imu.gyro
            a = frames.first_or_default(rs.stream.accel)
            g = frames.first_or_default(rs.stream.gyro)
            if a:
                d = a.as_motion_frame().get_motion_data(); accel = (d.x, d.y, d.z)
            if g:
                d = g.as_motion_frame().get_motion_data(); gyro = (d.x, d.y, d.z)
            self._last_imu = ImuSample(stamp=time.monotonic(), accel=accel, gyro=gyro)
        except Exception:
            pass

    def read(self) -> CameraFrame | None:
        if not self._open:
            raise CameraError("read() before open()")
        try:
            frames = self._pipe.wait_for_frames(2000)
        except Exception:
            return None
        self._pump_imu(frames)

        # Pull the IR pair from the RAW frameset, BEFORE align (align reprojects
        # depth into the colour frame; the IR imagers must stay in their own
        # rectified frame so the stereo geometry is preserved for cuVSLAM).
        ir_left = ir_right = None
        if self.enable_ir:
            try:
                il = frames.get_infrared_frame(1)
                ir = frames.get_infrared_frame(2)
                if il and ir:
                    ir_left = np.asanyarray(il.get_data())        # HxW uint8
                    ir_right = np.asanyarray(ir.get_data())       # HxW uint8
            except Exception:
                ir_left = ir_right = None

        frames = self._align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            return None

        rgb = np.asanyarray(c.get_data())                         # HxWx3 BGR
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale

        # organized cloud: deproject aligned depth, colour from the aligned image
        s = self.cloud_stride
        z = depth[::s, ::s]
        x = self._uv[0][::s, ::s] * z
        y = self._uv[1][::s, ::s] * z
        mask = (z > 0) & np.isfinite(z)
        xyz = np.stack((x[mask], y[mask], z[mask]), axis=-1).astype(np.float32)
        prgb = rgb[::s, ::s][mask]

        return CameraFrame(
            name=self.name, stamp=time.monotonic(), rgb=rgb, depth_m=depth,
            points_xyz=xyz, points_rgb=prgb, intrinsics=self._intrinsics,
            imu=self._last_imu if self.enable_imu else None,
            ir_left=ir_left, ir_right=ir_right,
            ir_intrinsics=self._ir_intrinsics if ir_left is not None else None,
            ir_baseline_m=self._ir_baseline_m if ir_left is not None else None)

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
        self._pipe = None
        self._open = False
