"""
serial_ports.py
===============

Cross-platform discovery of the **DM-USB2FDCAN serial adapter** so the same
calibration code runs on Linux *and* macOS.

The original auto-detect globbed ``/dev/ttyACM*`` / ``/dev/ttyUSB*`` — Linux-only
names.  On macOS the adapter enumerates as a USB-CDC modem under a completely
different name (``/dev/cu.usbmodemXXXX``), so that glob found nothing and the
serial path was effectively Linux-only.

This module uses :mod:`serial.tools.list_ports` (which works on every platform
pyserial supports) and ranks the candidates so the real motor adapter is picked
ahead of always-present junk ports (Bluetooth, the macOS debug console, etc.).

Platform device names this recognises:

* **macOS**  ``/dev/cu.usbmodem*`` (CDC-ACM — what the DM-USB2FDCAN shows up as),
  ``/dev/cu.usbserial*``, ``/dev/cu.wchusbserial*`` (CH340),
  ``/dev/cu.SLAB_USBtoUART`` (CP210x).  We use ``cu.`` (call-out), never
  ``tty.`` (the dial-in variant blocks on carrier-detect).
* **Linux**  ``/dev/ttyACM*`` (CDC-ACM), ``/dev/ttyUSB*`` (FTDI/CP210x/CH340).
* **Windows** ``COM*``.
"""
from __future__ import annotations

import glob
import sys
from dataclasses import dataclass
from typing import List, Optional

# STM32 Virtual COM Port — the MCU inside most DaMiao USB-CAN adapters.  Used only
# as a *ranking hint*; detection never requires a matching VID so a relabelled or
# clone adapter still works.
_STM32_VID = 0x0483

# Substrings of ports that are present on every Mac and are never our adapter.
_MAC_NOISE = ("bluetooth", "debug-console", "wlan-debug")


@dataclass(frozen=True)
class PortInfo:
    device: str            # e.g. /dev/cu.usbmodem1101
    description: str       # human string from the OS
    score: int             # higher = more likely to be the DM-USB2FDCAN

    @property
    def label(self) -> str:
        d = self.description or "serial port"
        return f"{self.device}  —  {d}" if d != self.device else self.device


def _glob_fallback() -> List[str]:
    """Last-resort device-name glob when pyserial's enumeration is empty.

    Covers the common adapter names on each platform; ``cu.`` (not ``tty.``) on
    macOS so we open the non-blocking call-out device.
    """
    if sys.platform == "darwin":
        pats = ["/dev/cu.usbmodem*", "/dev/cu.usbserial*",
                "/dev/cu.wchusbserial*", "/dev/cu.SLAB_USBtoUART*"]
    elif sys.platform.startswith("linux"):
        pats = ["/dev/ttyACM*", "/dev/ttyUSB*"]
    else:
        pats = []
    out: List[str] = []
    for p in pats:
        out += glob.glob(p)
    return sorted(set(out))


def _score(device: str, description: str, vid: Optional[int]) -> int:
    """Rank a candidate; negative means "almost certainly not the adapter"."""
    dev = device.lower()
    desc = (description or "").lower()

    if sys.platform == "darwin" and any(n in dev for n in _MAC_NOISE):
        return -100
    # On macOS only the call-out node should ever be offered.
    if sys.platform == "darwin" and dev.startswith("/dev/tty."):
        return -50

    s = 0
    if vid == _STM32_VID:
        s += 50                                   # the DaMiao MCU
    if "usbmodem" in dev or "/dev/ttyacm" in dev:
        s += 40                                   # CDC-ACM = the DM-USB2FDCAN
    if "usbserial" in dev or "/dev/ttyusb" in dev or "wchusbserial" in dev:
        s += 20                                   # FTDI/CH340 USB-serial
    if any(k in desc for k in ("damiao", "dm-usb", "usb2fdcan", "stm32",
                               "cdc", "can")):
        s += 15
    if dev.startswith("/dev/cu.usb") or dev.startswith("/dev/ttyacm") \
            or dev.startswith("/dev/ttyusb") or dev.startswith("com"):
        s += 5
    return s


def list_ports(include_unlikely: bool = False) -> List[PortInfo]:
    """All serial ports, best candidate first.

    ``include_unlikely`` keeps obviously-wrong ports (Bluetooth, ``tty.`` on
    macOS) in the list — useful so a GUI dropdown can still show *something* if
    auto-detect guessed wrong, but they sort to the bottom.
    """
    ports: List[PortInfo] = []
    try:
        from serial.tools import list_ports as _lp
        for p in _lp.comports():
            sc = _score(p.device, p.description or "", getattr(p, "vid", None))
            if sc < 0 and not include_unlikely:
                continue
            ports.append(PortInfo(p.device, p.description or "", sc))
    except Exception:
        pass

    if not ports:                                  # pyserial gave us nothing
        for dev in _glob_fallback():
            ports.append(PortInfo(dev, "", _score(dev, "", None)))

    ports.sort(key=lambda pi: (-pi.score, pi.device))
    return ports


def auto_port() -> str:
    """Return the single most-likely adapter device, or raise if none found."""
    cands = list_ports()
    if not cands:
        raise FileNotFoundError(
            "no serial adapter found. Plug in the DM-USB2FDCAN and check it "
            "enumerates: macOS `ls /dev/cu.usbmodem*`, Linux `ls /dev/ttyACM*`. "
            "(Make sure no slcand/other process is holding the port.)")
    return cands[0].device
