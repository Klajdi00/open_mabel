"""camera_hal -- a tiny, ROS-free hardware abstraction for MABEL's depth cameras.

    from camera_hal import RealSenseCamera, ZedCamera, CameraManager

Each backend owns one USB device through its vendor SDK and returns a uniform
`CameraFrame` (RGB + metric depth + XYZRGB cloud + intrinsics + IMU). The viewer
in `bringup.py` consumes only `CameraFrame`, so a future ROS bridge can reuse
this layer unchanged.
"""
from .base import CameraBase, CameraError
from .frame import CameraFrame, CameraIntrinsics, ImuSample
from .manager import CameraManager, CameraStatus
from .realsense import RealSenseCamera
from .uvc import UvcCamera
from .zed import ZedCamera

__all__ = [
    "CameraBase", "CameraError",
    "CameraFrame", "CameraIntrinsics", "ImuSample",
    "CameraManager", "CameraStatus",
    "RealSenseCamera", "UvcCamera", "ZedCamera",
]
