"""Hardware-agnostic data containers shared by every camera backend.

These types are the *only* thing the rest of `sensors/` sees -- a RealSense, a
ZED, or any future camera all hand back the same `CameraFrame`, so the viewer
(and a future ROS bridge) never needs to know which SDK produced the pixels.
Nothing here imports a vendor SDK or ROS; it is plain numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics + distortion for one (rectified) image stream.

    Everything SLAM/localization needs later: the K matrix, the distortion
    model + coefficients, and -- for stereo rigs like the ZED -- the baseline.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "pinhole"            # "pinhole" | "brown_conrady" | "kannala_brandt" ...
    coeffs: tuple[float, ...] = ()    # distortion coefficients (k1,k2,p1,p2,k3,...)
    depth_scale_m: float = 1.0        # raw depth unit -> metres (RealSense Z16)
    baseline_m: float | None = None   # stereo baseline (ZED), None for mono/RGB-D

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def pretty(self) -> str:
        b = f"  baseline={self.baseline_m * 1000:.1f}mm" if self.baseline_m else ""
        return (f"{self.width}x{self.height}  f=({self.fx:.1f},{self.fy:.1f})  "
                f"c=({self.cx:.1f},{self.cy:.1f})  {self.model}{b}")


@dataclass
class ImuSample:
    """A single inertial reading. Units: accel m/s^2, gyro rad/s."""

    stamp: float                                  # seconds (host monotonic clock)
    accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] | None = None  # quat (x,y,z,w), ZED only
    temperature_c: float | None = None

    @property
    def accel_mag(self) -> float:
        return float(np.linalg.norm(self.accel))


@dataclass
class CameraFrame:
    """One synchronized capture from a camera: colour + depth + cloud + IMU."""

    name: str
    stamp: float                                   # host monotonic seconds
    rgb: np.ndarray | None = None                  # HxWx3 uint8, BGR (OpenCV order)
    depth_m: np.ndarray | None = None              # HxW float32, metres (0 = invalid)
    points_xyz: np.ndarray | None = None           # Nx3 float32, metres, camera frame
    points_rgb: np.ndarray | None = None           # Nx3 uint8, BGR, matches points_xyz
    intrinsics: CameraIntrinsics | None = None
    imu: ImuSample | None = None                   # latest inertial sample, if any
    # Rectified IR stereo pair (D435i): the depth module's two imagers, mono8,
    # epipolar-aligned by the ASIC. This is what cuVSLAM tracks on (visual_slam
    # navigation env). ir_left is the LEFT imager (== depth origin); ir_right the
    # right. ir_intrinsics is the SHARED rectified pinhole (both imagers); the
    # horizontal baseline (left->right) is ir_baseline_m. None on cameras without
    # an exposed IR pair (e.g. the ZED, which hands back its own left/right RGB).
    ir_left: np.ndarray | None = None              # HxW uint8, rectified left IR
    ir_right: np.ndarray | None = None             # HxW uint8, rectified right IR
    ir_intrinsics: CameraIntrinsics | None = None  # shared rectified intrinsics
    ir_baseline_m: float | None = None             # stereo baseline (left->right), m

    @property
    def has_cloud(self) -> bool:
        return self.points_xyz is not None and len(self.points_xyz) > 0

    @property
    def has_ir_stereo(self) -> bool:
        return self.ir_left is not None and self.ir_right is not None

    @property
    def depth_range_m(self) -> tuple[float, float]:
        if self.depth_m is None:
            return (0.0, 0.0)
        valid = self.depth_m[(self.depth_m > 0) & np.isfinite(self.depth_m)]
        if valid.size == 0:
            return (0.0, 0.0)
        return (float(valid.min()), float(valid.max()))
