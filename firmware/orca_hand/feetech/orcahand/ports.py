"""USB-serial port discovery for the Feetech adapter.

One canonical ``detect_port`` for the whole stack -- the firmware scripts, the
GUI, and the ROS node all historically rolled their own slightly-different
versions. This is the single implementation.
"""

from __future__ import annotations

import serial.tools.list_ports

from .constants import USB_VENDOR_IDS


def list_ports() -> list:
    """Return all serial ports, USB-vendor matches first (best candidates)."""
    ports = list(serial.tools.list_ports.comports())
    return sorted(ports, key=lambda p: 0 if p.vid in USB_VENDOR_IDS else 1)


def detect_port() -> str:
    """Auto-detect the Feetech USB adapter.

    Resolution order:
      1. a port whose USB vendor id is a known Feetech adapter (CH340/CP210x),
      2. a port whose description names a known adapter chip,
      3. the only port present, if exactly one (ignoring Bluetooth).

    Raises
    ------
    RuntimeError
        if no plausible adapter can be found. Pass an explicit ``port`` to
        ``OrcaHand`` / ``FeetechBus`` to bypass detection.
    """
    ports = list(serial.tools.list_ports.comports())

    for vid in USB_VENDOR_IDS:
        for p in ports:
            if p.vid == vid:
                return p.device

    # macOS/Linux pseudo-ports that are never a Feetech adapter — must never be
    # auto-selected (debug-console in particular looks like "the only real port"
    # once a flaky adapter drops off the bus, and the bus would open it with 0 motors).
    _SKIP = ("bluetooth", "debug-console", "debug console", "wlan-debug")

    def _is_skip(dev: str) -> bool:
        return any(s in dev for s in _SKIP)

    for p in ports:
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        if _is_skip(dev):
            continue
        if any(k in desc for k in ("ch340", "cp210", "ftdi", "usb serial")):
            return p.device

    # Last resort: a single, genuine USB serial port (usbserial / usbmodem), never a
    # pseudo-port. If none qualifies, raise so the caller fails clearly instead of
    # silently opening the wrong device.
    real = [p for p in ports if not _is_skip((p.device or "").lower())
            and any(k in (p.device or "").lower() for k in ("usbserial", "usbmodem",
                                                            "ttyusb", "ttyacm"))]
    if len(real) == 1:
        return real[0].device

    raise RuntimeError(
        "No Feetech USB serial adapter found. Plug it in or pass an explicit "
        "port=... (candidates: " + ", ".join(p.device for p in ports) + ")"
    )
