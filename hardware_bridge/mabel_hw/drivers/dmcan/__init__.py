"""ctypes binding for DaMiao's ``libdm_device`` — the DM-USB2FDCAN dongle backend.

The DM-USB2FDCAN (USB ``34b7:6877``) is NOT the HDSC serial dongle: it has no
serial data path (its CDC port is inert) and is driven over libusb vendor
interfaces by DaMiao's closed-source backend, vendored under ``lib/`` (see
README.md for provenance). This module is import-light — the shared library is
loaded lazily by :func:`load`, so importing costs nothing on hosts without the
dongle or on platforms we don't vendor a binary for.

Struct layouts are transcribed from the SDK's ``dmcan.h`` / ``dmcan_def.py``.
Only the surface the MABEL drivers need is bound (find/open/enable-channel/
send/recv-callback/close); the SDK's queue/burst helpers are deliberately out.
"""
from __future__ import annotations

import ctypes as C
import os
import platform
import sys
from pathlib import Path

__all__ = ["DmcanError", "load", "usb_rx_frame", "DEV_RECV_CALLBACK",
           "dmcan_channel_can_info", "DEVICE_TYPE_USB2CANFD"]

DEVICE_TYPE_USB2CANFD = 0                     # dmcan_device_type_t


class DmcanError(RuntimeError):
    """The dmcan backend is unavailable or a device call failed."""


class dmcan_channel_can_info(C.Structure):
    _pack_ = 1
    _fields_ = [("channel", C.c_uint8), ("canfd", C.c_bool),
                ("can_baudrate", C.c_uint32), ("canfd_baudrate", C.c_uint32),
                ("can_sp", C.c_float), ("canfd_sp", C.c_float)]


class usb_rx_frame_head(C.Structure):
    _pack_ = 1
    _fields_ = [("can_id", C.c_uint32, 29), ("esi", C.c_uint32, 1),
                ("ext", C.c_uint32, 1), ("rtr", C.c_uint32, 1),
                ("timestamp", C.c_uint64), ("channel", C.c_uint8),
                ("canfd", C.c_uint8, 1), ("dir", C.c_uint8, 1),
                ("brs", C.c_uint8, 1), ("ack", C.c_uint8, 1),
                ("dlc", C.c_uint8, 4), ("reserved", C.c_uint16)]


class usb_rx_frame(C.Structure):
    _pack_ = 1
    _fields_ = [("head", usb_rx_frame_head), ("payload", C.c_uint8 * 64)]


class _dmcan_context(C.Structure):
    pass


class _dmcan_device_handle(C.Structure):
    pass


DEV_RECV_CALLBACK = C.CFUNCTYPE(None, C.POINTER(_dmcan_device_handle),
                                C.POINTER(usb_rx_frame))

_LIB = None


_LIB_SUFFIX = {"darwin": ".dylib", "win32": ".dll"}       # everything else: ELF .so


def _lib_path() -> Path:
    env = os.environ.get("MABEL_DMCAN_LIB")
    if env:
        return Path(env)
    plat = f"{sys.platform}-{platform.machine()}"          # e.g. linux-aarch64
    name = "libdm_device" + _LIB_SUFFIX.get(sys.platform, ".so")
    return Path(__file__).resolve().parent / "lib" / plat / name


def load() -> C.CDLL:
    """Load (once) and return the prototyped backend. Raises DmcanError with a
    actionable message when the binary is absent for this platform."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _lib_path()
    if not path.is_file():
        raise DmcanError(
            f"libdm_device not vendored for this platform ({path}) — fetch it from "
            "github.com/dmBots/dm-device-sdk or set $MABEL_DMCAN_LIB")
    try:
        lib = C.CDLL(str(path))
    except OSError as e:
        raise DmcanError(f"could not load {path}: {e} (libusb-1.0 installed?)") from e

    hp, cp = C.POINTER(_dmcan_device_handle), C.POINTER(_dmcan_context)
    proto = [
        ("dmcan_context_create", None, [C.POINTER(cp)]),
        ("dmcan_context_destroy", None, [cp]),
        ("dmcan_find_devices", C.c_int, [cp]),
        ("dmcan_find_devices_with_type", C.c_int, [cp, C.c_int]),
        ("dmcan_device_get", C.c_bool, [cp, C.POINTER(hp), C.c_int]),
        ("dmcan_device_open", C.c_bool, [hp]),
        ("dmcan_device_close", None, [hp]),
        ("dmcan_device_enable_channel", C.c_bool, [hp, C.c_uint8]),
        ("dmcan_device_disable_channel", C.c_bool, [hp, C.c_uint8]),
        ("dmcan_device_get_channel_baudrate", C.c_bool,
         [hp, C.c_uint8, C.POINTER(dmcan_channel_can_info)]),
        ("dmcan_device_hook_recv_callback", None, [hp, DEV_RECV_CALLBACK]),
        ("dmcan_device_send_can", C.c_bool,
         [hp, C.c_uint8, C.c_uint32, C.c_bool, C.c_bool, C.c_bool, C.c_bool,
          C.c_uint8, C.POINTER(C.c_uint8)]),
    ]
    for name, res, args in proto:
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    _LIB = lib
    return lib
