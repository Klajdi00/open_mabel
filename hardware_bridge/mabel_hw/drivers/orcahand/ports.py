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


def detect_side_port(side: str, serial_map: dict, log=None) -> "str | None":
    """SIDE-AWARE auto-detect: resolve which USB adapter is the ``side`` hand.

    The two ORCA hand adapters are IDENTICAL CH340s — the USB serial number is the
    only stable discriminator, so first-come-first-served detection (``detect_port``)
    can hand the left adapter to the right hand. This resolves through a registry
    (``serial_map``: side -> adapter serial number, e.g. from config.HANDS
    ``adapter_serials``) instead:

      1. an adapter whose USB serial number IS registered for this side wins;
      2. a side with NO registered serial may claim the single candidate adapter
         that is not registered to another side (logged loudly with the serial to
         pin) — ambiguous cases (0 or 2+ unclaimed) resolve to None;
      3. otherwise None: the caller stays idle and retries (hot-plug support).

    Returns the device path, or None when no adapter for this side is present.
    Never raises — absence of a hand must not kill the bridge.
    """
    log = log or (lambda *a, **k: None)
    want = (serial_map.get(side) or "").strip()
    other_serials = {(v or "").strip() for s, v in serial_map.items()
                     if s != side and (v or "").strip()}
    candidates = [p for p in serial.tools.list_ports.comports()
                  if p.vid in USB_VENDOR_IDS]

    if want:
        for p in candidates:
            if (p.serial_number or "") == want:
                return p.device
        # Registered but absent. If an UNREGISTERED adapter is present, hint how to
        # re-pin (e.g. the hand was rewired onto a new adapter) — but never claim it:
        # an unknown adapter may be the other hand.
        unknown = [p for p in candidates
                   if (p.serial_number or "") not in ({want} | other_serials)]
        for p in unknown:
            log(f"[{side}] registered adapter (serial {want}) not present, but an "
                f"UNREGISTERED adapter is: {p.device} (serial {p.serial_number}). "
                f"If this is the {side} hand, pin it: "
                f"MABEL_HW_HANDS_{side.upper()}_SERIAL={p.serial_number}")
        return None

    # No registered serial for this side: claim the single unclaimed candidate.
    unclaimed = [p for p in candidates if (p.serial_number or "") not in other_serials]
    if len(unclaimed) == 1:
        p = unclaimed[0]
        log(f"[{side}] no registered adapter serial — claiming the only unclaimed "
            f"adapter {p.device} (serial {p.serial_number}). Pin it so this can "
            f"never mis-side: MABEL_HW_HANDS_{side.upper()}_SERIAL={p.serial_number}")
        return p.device
    if len(unclaimed) > 1:
        log(f"[{side}] {len(unclaimed)} unregistered adapters present — AMBIGUOUS, "
            f"staying idle. Pin serials (ls -l /dev/serial/by-id/): "
            + ", ".join(f"{p.device}={p.serial_number}" for p in unclaimed))
    return None


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
