"""
transports.py
=============

Physical CAN transports for the DaMiao actuators, **ported from the proven
firmware driver** (``firmware/openarm/scripts/teleop_gui.py``) so the calibration
tooling talks to the motor exactly the way the working teleop GUI does.

The DaMiao CAN payload + feedback format is identical across adapters; only the
physical link differs.  Every transport exposes the same tiny interface so the
protocol layer (:mod:`damiao`) is shared:

    send(arb_id:int, data:bytes)          transmit one standard (11-bit) CAN frame
    poll(timeout)->(arb_id, data)|None    receive one frame (arb id + 8 payload bytes)
    close()                               release the device

Three backends:

* ``serial``    — DaMiao USB-to-CAN *serial* adapter, official DM_Control_Python
  framing (pyserial, 921600 baud, ``0x55 0xAA`` header).  Most reliable on macOS
  when you have that adapter.
* ``gs_usb``    — CANable / candleLight clones (e.g. the **Jhoinrch USB-CAN**,
  ``1d50:606f`` "canable2 gs_usb") over USB via python-can + libusb/pyusb.  This
  is the macOS path for those adapters — there is no SocketCAN on a Mac.
* ``socketcan`` — Linux kernel CAN (``can0``).

The hard-won macOS fixes from the firmware are kept verbatim: hunting Homebrew
``libusb`` paths that ``find_library`` misses on Apple Silicon, and a retry-open
loop around the gs_usb bring-up (python-can issues ``libusb_reset_device`` which
briefly re-enumerates the clone adapter, so the first open can race and fail).
"""
from __future__ import annotations

import glob
import sys
import time

_DARWIN = sys.platform == "darwin"


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def find_serial_port():
    """First likely DaMiao USB-to-CAN *serial* adapter on macOS/Linux, else None."""
    pats = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
            "/dev/cu.SLAB_USBtoUART*", "/dev/ttyACM*", "/dev/ttyUSB*")
    for pat in pats:
        hits = sorted(g for g in glob.glob(pat) if "debug-console" not in g)
        if hits:
            return hits[0]
    return None


def _ensure_libusb_backend():
    """Return a pyusb libusb1 backend, hunting Homebrew paths that
    ``ctypes.util.find_library`` misses on Apple Silicon.  Returns None on Linux
    or if pyusb isn't needed — gs_usb falls back to its own discovery then."""
    try:
        import usb.backend.libusb1
    except Exception:
        return None
    be = usb.backend.libusb1.get_backend()
    if be is not None:
        return be
    import os
    for cand in ("/opt/homebrew/lib/libusb-1.0.dylib",   # Apple Silicon brew
                 "/usr/local/lib/libusb-1.0.dylib",       # Intel brew
                 "/opt/local/lib/libusb-1.0.dylib"):      # MacPorts
        if os.path.exists(cand):
            be = usb.backend.libusb1.get_backend(find_library=lambda _: cand)
            if be is not None:
                return be
    return None


def gs_usb_present() -> bool:
    """True if a CANable/candleLight (gs_usb) adapter is on the USB bus.

    Used for backend auto-selection so the GUI can default to gs_usb when that
    adapter is plugged in but no serial adapter is.  Best-effort; never raises.
    """
    try:
        import usb.core
        be = _ensure_libusb_backend()
        kw = {"find_all": True}
        if be is not None:
            kw["backend"] = be
        for dev in usb.core.find(**kw):
            # candleLight/CANable family: VID 0x1D50, PID 0x606F (+ clones at 0x1209)
            if dev.idVendor in (0x1D50, 0x1209):
                return True
    except Exception:
        pass
    return False


# --------------------------------------------------------------------------- #
#  transports                                                                 #
# --------------------------------------------------------------------------- #
class PyCanTransport:
    """SocketCAN (Linux) or gs_usb (CANable/candleLight over USB) via python-can."""

    def __init__(self, interface, channel, bitrate):
        import can
        self._can = can
        self.interface = interface
        if interface == "gs_usb":
            _ensure_libusb_backend()   # nudge pyusb toward Homebrew libusb on macOS
            # python-can's gs_usb open issues libusb_reset_device(), which the CAN
            # engine needs to bring its channel up — but the reset briefly
            # re-enumerates the adapter, so the first open can race and fail with
            # "No such device"/"Access denied".  Retry a few times to ride it out.
            last = None
            for _ in range(5):
                try:
                    self._bus = can.interface.Bus(
                        interface="gs_usb", channel=channel, index=0, bitrate=bitrate)
                    break
                except Exception as e:
                    last = e
                    time.sleep(0.8)
            else:
                raise last
        else:
            self._bus = can.interface.Bus(channel=channel, interface="socketcan")

    def send(self, arb_id, data):
        self._bus.send(self._can.Message(
            arbitration_id=arb_id, data=bytes(data), is_extended_id=False))

    def poll(self, timeout=0.5):
        m = self._bus.recv(timeout=timeout)
        if m is None or not m.is_rx:
            return None
        return (m.arbitration_id, bytes(m.data))

    def close(self):
        self._bus.shutdown()


class SerialDamiaoTransport:
    """DaMiao USB-to-CAN serial adapter — the official DM_Control_Python protocol.

    Each CAN frame is wrapped in a 30-byte serial packet (``0x55 0xAA`` header, CAN
    id at [13:15], 8 data bytes at [21:29]); replies arrive as 16-byte packets
    framed by ``0xAA … 0x55`` with the CAN id little-endian at [3:7] and payload at
    [7:15].  This is the path the official library uses to run cleanly on macOS.
    """
    interface = "serial"
    # Byte template lifted verbatim from DM_CAN.MotorControl.send_data_frame.
    _TX = bytes([0x55, 0xAA, 0x1e, 0x03, 0x01, 0, 0, 0, 0x0a, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def __init__(self, port, baud=921600):
        import serial   # pyserial (lazy import — only needed for this backend)
        self._ser = serial.Serial(port, baud, timeout=0)
        self._buf = bytearray()
        self.port = port

    def send(self, arb_id, data):
        f = bytearray(self._TX)
        f[13] = arb_id & 0xFF
        f[14] = (arb_id >> 8) & 0xFF
        f[21:29] = bytes(data)[:8].ljust(8, b"\x00")
        self._ser.write(bytes(f))

    def poll(self, timeout=0.5):
        frame = self._extract()
        if frame is not None:
            return frame
        self._ser.timeout = timeout
        chunk = self._ser.read(64)            # blocks up to `timeout` for the 1st byte
        if chunk:
            self._buf.extend(chunk)
        n = self._ser.in_waiting               # then drain whatever else arrived
        if n:
            self._buf.extend(self._ser.read(n))
        return self._extract()

    def _extract(self):
        b = self._buf
        while len(b) >= 16:
            if b[0] == 0xAA and b[15] == 0x55:
                fr = bytes(b[:16]); del b[:16]
                arb = fr[3] | (fr[4] << 8) | (fr[5] << 16) | (fr[6] << 24)
                return (arb, fr[7:15])
            del b[:1]                          # resync: drop one byte, try again
        return None

    def close(self):
        self._ser.close()


def make_transport(interface: str, *, channel: str = "can0",
                   port: str | None = None, bitrate: int = 1_000_000):
    """Open the transport for ``interface`` (``serial`` | ``gs_usb`` | ``socketcan``)."""
    if interface == "serial":
        port = port or find_serial_port()
        if not port:
            raise RuntimeError("no serial adapter found (pass port=<device>)")
        return SerialDamiaoTransport(port)
    if interface in ("gs_usb", "socketcan"):
        return PyCanTransport(interface, channel, bitrate)
    raise ValueError(f"unknown interface '{interface}'")


def auto_interface() -> str:
    """Pick the best backend for this machine, mirroring the firmware GUI.

    macOS: ``serial`` if a DaMiao serial adapter is present, else ``gs_usb``
    (CANable/candleLight).  Linux: ``socketcan``.
    """
    if not _DARWIN:
        return "socketcan"
    if find_serial_port():
        return "serial"
    return "gs_usb"
