"""Generic UVC / V4L2 colour-camera backend (OpenCV VideoCapture).

For the plain USB webcams MABEL carries that are NOT depth cameras -- the two
Arducam Global-Shutter WRIST cams. They speak MJPG over V4L2 and hand back a
colour frame only (no depth / cloud / IMU), so this backend fills just
`CameraFrame.rgb` (+ a rough placeholder pinhole `intrinsics`).

Each instance owns ONE /dev/videoN node. The two wrist cams are electronically
identical (same VID:PID *and* serial), so they are told apart ONLY by the physical
USB hub port; the 99-mabel-wrist-cams.udev rule pins those ports to
/dev/wrist_{left,right}. Pass that symlink as `device` and the by-path node as
`fallback` (used automatically when the symlink is absent), mirroring the proven
hardware_bridge wrist driver.

Like every backend it degrades gracefully: a missing device just reports
is_present()==False so the CameraManager keeps retrying -- never fatal.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .base import CameraBase, CameraError
from .frame import CameraFrame, CameraIntrinsics

try:
    import cv2
except Exception:                     # pragma: no cover - import guarded for nice error
    cv2 = None


class UvcCamera(CameraBase):
    """A single MJPG UVC camera over V4L2 (e.g. a wrist cam)."""

    def __init__(self, name: str, device: str, fallback: str | None = None,
                 model: str | None = None, width: int = 1280, height: int = 720,
                 fps: int = 30, auto_exposure: int = 1, exposure: int = 120,
                 gain: int = 180):
        super().__init__()
        self.name = name
        self.model = model or f"UVC camera ({name})"
        self.device = device
        self.fallback = fallback
        self.width, self.height, self.fps = width, height, fps
        # Exposure: AUTO (aperture-priority) collapses to ~2 fps in dim light (the cam
        # stretches exposure to ~0.5 s). MANUAL (auto_exposure=1) + a short exposure holds
        # the framerate; gain compensates brightness. Mirrors hardware_bridge wrist_cams.py.
        self.auto_exposure, self.exposure, self.gain = auto_exposure, exposure, gain
        self._cap = None
        self._path = None

    # -- device resolution -------------------------------------------------
    def _resolve(self) -> str | None:
        """Prefer the udev symlink; fall back to the by-path node if absent."""
        for path in (self.device, self.fallback):
            if path and os.path.exists(path):
                return path
        return None

    # -- presence ----------------------------------------------------------
    def is_present(self) -> bool:
        return self._resolve() is not None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        if cv2 is None:
            raise CameraError("opencv (cv2) not installed",
                              hint="pip install opencv-python in the camera env")
        path = self._resolve()
        if path is None:
            raise CameraError(f"{self.name}: no device at {self.device}",
                              hint="check the wrist cam USB / udev symlink")
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"{self.name}: cannot open {path}",
                              hint="another process may hold the camera")
        # MJPG @ requested geometry -- YUYV on these cams caps at ~10 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always the freshest frame (low latency)
        # Force manual exposure (order matters: auto_exposure->manual BEFORE exposure is
        # writable) so dim light can't crater the framerate. See class docstring/__init__.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(self.auto_exposure))
        if self.auto_exposure == 1:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
            cap.set(cv2.CAP_PROP_GAIN, float(self.gain))
        self._cap = cap
        self._path = path
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        # Rough pinhole placeholder -- these wide-angle cams are NOT calibrated;
        # camera_info keeps RViz / image_transport happy, do not use for metric work.
        self._intrinsics = CameraIntrinsics(
            width=w, height=h, fx=float(w), fy=float(w),
            cx=w / 2.0, cy=h / 2.0, model="pinhole")
        self._open = True

    def read(self) -> CameraFrame | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None                          # transient miss -> manager retries
        # camera_hal convention: `rgb` carries BGR (OpenCV order); the ROS bridge
        # flips it to rgb8 on publish, exactly like the RealSense backend.
        return CameraFrame(name=self.name, stamp=time.monotonic(),
                           rgb=frame, intrinsics=self._intrinsics)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:                    # noqa: BLE001
                pass
        self._cap = None
        self._open = False
