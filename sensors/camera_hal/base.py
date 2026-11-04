"""Common interface every camera backend implements.

A backend owns exactly one physical device over USB (via its vendor SDK) and
hands back `CameraFrame`s. It knows nothing about visualization or ROS.
"""
from __future__ import annotations

import abc

from .frame import CameraFrame, CameraIntrinsics


class CameraError(RuntimeError):
    """Raised when a device cannot be opened or read.

    `hint` carries an operator-facing remedy (e.g. "plug into a USB-3 port")
    that the dashboard surfaces verbatim.
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class CameraBase(abc.ABC):
    """Abstract USB camera. Subclasses: RealSenseCamera, ZedCamera."""

    #: short, stable id used in logs / window titles (e.g. "realsense", "zedm")
    name: str = "camera"
    #: human label for the dashboard (e.g. "Intel RealSense D435i")
    model: str = "camera"

    def __init__(self) -> None:
        self._open = False
        self._intrinsics: CameraIntrinsics | None = None

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the device and start streaming. Raises CameraError on failure."""

    @abc.abstractmethod
    def read(self) -> CameraFrame | None:
        """Block for the next frame. Returns None on a transient miss."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device. Safe to call more than once."""

    # -- introspection -----------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @abc.abstractmethod
    def is_present(self) -> bool:
        """True if the hardware is plugged in and usable RIGHT NOW (cheap check).

        Lets the manager skip an absent camera without a noisy open() failure.
        """

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
