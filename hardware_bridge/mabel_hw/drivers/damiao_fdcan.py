"""damiao_fdcan.py — DaMiao motor over the DM-USB2FDCAN dongle (USB ``34b7:6877``).

The DM-USB2FDCAN is DaMiao's newer USB-to-CANFD adapter. Unlike the HDSC serial
dongle (``damiao_serial.py``) it has NO serial data path — its CDC port ignores the
0x55AA framing entirely (verified live 2026-07-26: byte-identical probes that the
HDSC dongle answers get pure silence). The factory firmware is driven over libusb
vendor interfaces through DaMiao's closed-source backend, vendored + bound in
``drivers/dmcan``. Today it carries MABEL's neck-yaw DM-J4310.

On the CAN side the *motor* protocol is identical to every other DaMiao: MIT
frames to the motor's ESC id (packing law shared with ``damiao_can``), feedback on
its MST id, and the 0x7FF param bus (0x33 read / 0x55 write / 0xCC refresh /
0xAA save) — verified against cmjang/DM_Control_Python and dmBots/dm-device-sdk.

The public surface mirrors ``DamiaoSerialDriver`` exactly (drop-in for the studio
and any future HAL backend), and reuses its ``DamiaoSerialState``. Same safety
contract:
  • connect() NEVER energizes — open device, enable the CAN channel, read the
    motor's scale registers (RIDs 21/22/23 — the wire truth) and ctrl mode.
  • set_torque(True) is a SAFE HOLD — present pose is polled first and commanded
    as the MIT target in the same breath as enable (0xFC): no snap, ever.
  • set_torque(False) (0xFD) frees the joint — the E-stop / hand-calibration path.
  • read() polls 0xCC — the official non-energizing refresh.

Extra (config-only, used to commission a fresh motor): ``write_param`` (0x55) and
``save_params`` (0xAA — motor must be disabled; we disable first). These write
REGISTERS, never a motion target.
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Optional

from ..errors import SubsystemUnavailable
from ..types import monotonic
from .damiao_can import f2u, u2f
from .damiao_serial import DamiaoSerialState

__all__ = ["DamiaoFdcanDriver", "dongle_present"]

_USB_VID, _USB_PID = 0x34B7, 0x6877

_PARAM_BUS_ID = 0x7FF
_CMD_ENABLE, _CMD_DISABLE = 0xFC, 0xFD
_CMD_REFRESH, _CMD_READ, _CMD_WRITE, _CMD_SAVE = 0xCC, 0x33, 0x55, 0xAA
_RID_CTRL_MODE, _RID_PMAX, _RID_VMAX, _RID_TMAX = 10, 21, 22, 23
_INT_RIDS = frozenset(range(7, 11)) | frozenset(range(13, 17)) | {35, 36}

# Channel bring-up retries — see connect(): a stale channel from a previous
# close makes every OTHER open fail to enable, so 2 attempts is the measured
# floor and 4 leaves margin for a busy bus.
_OPEN_ATTEMPTS, _OPEN_RETRY_S = 4, 0.15


def dongle_present() -> bool:
    """A DM-USB2FDCAN is enumerated on USB.

    Used only to sharpen connect()'s failure hint (absent vs present-but-unopenable),
    so it must never raise. On Linux this is a sysfs scan needing no device access, so
    it still answers before a udev rule grants libusb permissions. macOS has no sysfs;
    there we ask pyserial for the dongle's CDC node, which enumerates alongside the
    vendor interfaces — that node is INERT for data (see the module docstring) but its
    presence is a perfectly good existence check, and needs no device access either.
    """
    import sys                                                 # noqa: PLC0415
    if sys.platform == "darwin":
        try:
            from serial.tools import list_ports                # noqa: PLC0415
        except Exception:
            return False
        return any(p.vid == _USB_VID and p.pid == _USB_PID
                   for p in list_ports.comports())
    from pathlib import Path                                    # noqa: PLC0415
    for dev in Path("/sys/bus/usb/devices").glob("*"):
        try:
            vid = (dev / "idVendor").read_text().strip()
            pid = (dev / "idProduct").read_text().strip()
        except OSError:
            continue
        if vid == f"{_USB_VID:04x}" and pid == f"{_USB_PID:04x}":
            return True
    return False


def param_frame(can_id: int, cmd: int, rid: int = 0, value: bytes = b"") -> bytes:
    """One 8-byte 0x7FF param-bus payload: [id_lo, id_hi, cmd, rid, value·4]."""
    v = bytes(value)[:4].ljust(4, b"\x00") if cmd == _CMD_WRITE else b"\x00" * 4
    return bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, cmd & 0xFF, rid & 0xFF]) + v


def mit_frame(p: float, v: float, kp: float, kd: float, t: float,
              p_max: float, v_max: float, t_max: float) -> bytes:
    """The 8-byte MIT command payload — same packing law as damiao_can/serial."""
    pi = f2u(p, -p_max, p_max, 16)
    vi = f2u(v, -v_max, v_max, 12)
    ki = f2u(kp, 0.0, 500.0, 12)
    di = f2u(kd, 0.0, 5.0, 12)
    ti = f2u(t, -t_max, t_max, 12)
    return bytes([
        (pi >> 8) & 0xFF, pi & 0xFF,
        (vi >> 4) & 0xFF, ((vi & 0xF) << 4) | ((ki >> 8) & 0xF), ki & 0xFF,
        (di >> 4) & 0xFF, ((di & 0xF) << 4) | ((ti >> 8) & 0xF), ti & 0xFF])


class DamiaoFdcanDriver:
    """Owns the DM-USB2FDCAN dongle for ONE motor on channel 0. Thread-safe;
    request/response — every TX waits for its matched reply (the SDK's recv
    callback feeds a small ring the waiter drains), so no shared-bus races."""

    def __init__(self, can_id: int = 2, kp: float = 20.0, kd: float = 1.0,
                 p_max: float = 12.5, v_max: float = 30.0, t_max: float = 10.0,
                 channel: int = 0, device_index: int = 0, logger=None):
        self.can_id, self.channel = int(can_id), int(channel)
        self._index = int(device_index)
        self.kp, self.kd = float(kp), float(kd)
        # Config seeds the fixed-point MIT scales; connect() overwrites them from
        # the motor's own registers whenever the reads succeed (the wire truth).
        self.p_max, self.v_max, self.t_max = float(p_max), float(v_max), float(t_max)
        self.scales_from_motor = False
        self.ctrl_mode: Optional[int] = None
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._rxlock = threading.Condition()
        self._rx: list = []                       # (can_id, bytes) ring, newest last
        self._lib = None
        self._ctx = None
        self._dev = None
        self._cb = None                           # keep the CFUNCTYPE alive (GC trap)
        self._state = DamiaoSerialState()
        self._torque: Optional[bool] = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        """Open the dongle + CAN channel and read the motor's scale registers.
        Never energizes. Raises SubsystemUnavailable when the dongle is absent
        or unopenable (udev permissions, another owner)."""
        with self._lock:
            if self._dev is not None:
                return
            import ctypes as ct                                # noqa: PLC0415
            from . import dmcan                                # noqa: PLC0415
            try:
                lib = dmcan.load()
            except dmcan.DmcanError as e:
                raise SubsystemUnavailable(str(e)) from e

            # The SDK's close() does not fully release the CAN channel, so the
            # FIRST open after any close reliably fails to enable it and the next
            # one reliably succeeds — measured on macOS as a perfect alternation
            # (8/8 bring-ups, every one on cycle 2). Retrying the *enable* within
            # one open NEVER recovers (0/5 attempts); only a fresh open does. So
            # the retry has to wrap the whole find→get→open→enable cycle.
            ctx = dev = None
            why = None
            for _ in range(_OPEN_ATTEMPTS):
                ctx = ct.POINTER(dmcan._dmcan_context)()
                lib.dmcan_context_create(ct.byref(ctx))
                n = lib.dmcan_find_devices_with_type(ctx, dmcan.DEVICE_TYPE_USB2CANFD)
                if n <= self._index:
                    lib.dmcan_context_destroy(ctx)
                    hint = ("permissions? (34b7:6877 must be user-accessible)"
                            if dongle_present() else "is it plugged in?")
                    raise SubsystemUnavailable(
                        f"no DM-USB2FDCAN dongle found via libdm_device — {hint}")
                dev = ct.POINTER(dmcan._dmcan_device_handle)()
                if not lib.dmcan_device_get(ctx, ct.byref(dev), self._index) \
                        or not lib.dmcan_device_open(dev):
                    lib.dmcan_context_destroy(ctx)
                    ctx = dev = None
                    why = ("could not open the DM-USB2FDCAN dongle — held by "
                           "another process, or the udev rule is missing")
                    time.sleep(_OPEN_RETRY_S)
                    continue
                if lib.dmcan_device_enable_channel(dev, self.channel):
                    break
                lib.dmcan_device_close(dev)
                lib.dmcan_context_destroy(ctx)
                ctx = dev = None
                why = f"could not enable CAN channel {self.channel} on the dongle"
                time.sleep(_OPEN_RETRY_S)
            else:
                raise SubsystemUnavailable(
                    f"{why} (after {_OPEN_ATTEMPTS} open attempts)")

            def _on_recv(_h, frame):
                try:
                    f = frame.contents
                    if f.head.channel != self.channel or f.head.dlc > 8:
                        return
                    d = bytes(f.payload[:8])
                    with self._rxlock:
                        self._rx.append((int(f.head.can_id), d))
                        del self._rx[:-64]                     # bounded ring
                        self._rxlock.notify_all()
                except Exception:                              # never raise into C
                    pass

            self._cb = dmcan.DEV_RECV_CALLBACK(_on_recv)
            lib.dmcan_device_hook_recv_callback(dev, self._cb)
            self._lib, self._ctx, self._dev = lib, ctx, dev

            info = dmcan.dmcan_channel_can_info()
            baud = None
            if lib.dmcan_device_get_channel_baudrate(dev, self.channel, ct.byref(info)):
                baud = int(info.can_baudrate)
        self._read_scales()
        self._log(f"[damiao-fdcan] dongle open, ch {self.channel} "
                  f"({baud if baud else '?'} bit/s), motor id {self.can_id}, "
                  f"ranges p±{self.p_max} v±{self.v_max} t±{self.t_max} "
                  f"({'motor registers' if self.scales_from_motor else 'config defaults'})")

    def connected(self) -> bool:
        return self._dev is not None

    @property
    def resolved_port(self) -> Optional[str]:
        return "usb:34b7:6877" if self._dev is not None else None

    @property
    def is_torque_enabled(self) -> Optional[bool]:
        return self._torque

    def close(self) -> None:
        with self._lock:
            if self._dev is not None:
                try:
                    # Leaving a motor energized with nobody watching is never safe.
                    self._send(self.can_id, [0xFF] * 7 + [_CMD_DISABLE])
                except Exception:
                    pass
                try:
                    self._lib.dmcan_device_close(self._dev)
                    self._lib.dmcan_context_destroy(self._ctx)
                except Exception:
                    pass
                self._dev = self._ctx = self._cb = None
                self._torque = None

    # ── wire primitives (call with lock held) ────────────────────────────────
    def _send(self, arb_id: int, data8) -> None:
        if self._dev is None:
            raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
        import ctypes as ct                                    # noqa: PLC0415
        buf = (ct.c_uint8 * 8)(*bytes(data8))
        if not self._lib.dmcan_device_send_can(
                self._dev, self.channel, arb_id, False, False, False, False, 8, buf):
            # A reset/replugged dongle re-enumerates and this handle goes stale
            # forever. DROP it so the next connect() re-finds the device (the
            # serial driver's hot-replug pattern). Deliberately no device_close():
            # libusb teardown on a dead handle can abort() the process — a leaked
            # handle beats a crashed studio.
            self._dev = self._ctx = self._cb = None
            self._torque = None
            raise SubsystemUnavailable(
                "DM-USB2FDCAN send failed — dongle reset/replugged; reconnecting "
                "on the next poll")

    def _rx_until(self, match, timeout: float = 0.25):
        """Wait for a frame where ``match(arb, data)`` returns non-None."""
        deadline = time.monotonic() + timeout
        seen = 0
        with self._rxlock:
            while True:
                for arb, d in self._rx[seen:]:
                    got = match(arb, d)
                    if got is not None:
                        return got
                seen = len(self._rx)
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._rxlock.wait(left)

    def _drain_rx(self) -> None:
        with self._rxlock:
            self._rx.clear()

    def _ingest_feedback(self, d: bytes) -> DamiaoSerialState:
        st = DamiaoSerialState(
            live=True,
            position=u2f((d[1] << 8) | d[2], -self.p_max, self.p_max, 16),
            velocity=u2f((d[3] << 4) | (d[4] >> 4), -self.v_max, self.v_max, 12),
            torque=u2f(((d[4] & 0xF) << 8) | d[5], -self.t_max, self.t_max, 12),
            t_mos=int(d[6]), t_rotor=int(d[7]), err=(d[0] >> 4) & 0xF,
            stamp=monotonic())
        self._state = st
        return st

    def _feedback_match(self, arb: int, d: bytes):
        # MIT feedback rides the motor's MST id; the payload's byte-0 low nibble
        # carries the slave id — match on that (works whatever MST_ID is set to).
        if arb != _PARAM_BUS_ID and len(d) == 8 and (d[0] & 0x0F) == (self.can_id & 0x0F):
            return d
        return None

    def _poll_once(self, timeout: float = 0.25) -> Optional[DamiaoSerialState]:
        """0xCC refresh → one feedback frame. Non-energizing. Lock held by caller."""
        self._drain_rx()
        self._send(_PARAM_BUS_ID, param_frame(self.can_id, _CMD_REFRESH))
        d = self._rx_until(self._feedback_match, timeout)
        return self._ingest_feedback(d) if d is not None else None

    # ── registers ────────────────────────────────────────────────────────────
    def _param_match(self, cmd: int, rid: Optional[int]):
        def match(_arb, d):
            if len(d) == 8 and d[2] == cmd and (d[0] | (d[1] << 8)) == self.can_id \
                    and (rid is None or d[3] == rid):
                return d
            return None
        return match

    def read_param(self, rid: int, timeout: float = 0.3) -> Optional[float]:
        """Read one motor register over the 0x7FF param bus (0x33 — config-only,
        never moves anything). uint32 for the int RIDs, float32 otherwise."""
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            self._drain_rx()
            self._send(_PARAM_BUS_ID, param_frame(self.can_id, _CMD_READ, rid))
            d = self._rx_until(self._param_match(_CMD_READ, int(rid)), timeout)
        if d is None:
            return None
        if int(rid) in _INT_RIDS:
            return float(int.from_bytes(d[4:8], "little"))
        return struct.unpack("<f", d[4:8])[0]

    def write_param(self, rid: int, value, timeout: float = 0.3) -> bool:
        """Write one motor register (0x55 — config-only: ids, limits, mode; NEVER
        a motion target). Volatile until ``save_params``. True when the motor
        acked the write."""
        raw = (int(value).to_bytes(4, "little") if int(rid) in _INT_RIDS
               else struct.pack("<f", float(value)))
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            self._drain_rx()
            self._send(_PARAM_BUS_ID, param_frame(self.can_id, _CMD_WRITE, rid, raw))
            d = self._rx_until(self._param_match(_CMD_WRITE, int(rid)), timeout)
        return d is not None

    def save_params(self) -> None:
        """Persist the motor's registers to its flash (0xAA). The motor must be
        de-energized to save — we disable first (a freed joint is the safe state
        for commissioning) and leave it disabled."""
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            self._send(self.can_id, [0xFF] * 7 + [_CMD_DISABLE])
            self._torque = False
            time.sleep(0.05)
            self._send(_PARAM_BUS_ID, param_frame(self.can_id, _CMD_SAVE))
            time.sleep(0.2)                       # flash write settles before more traffic

    def _read_scales(self) -> None:
        """Pull (p_max, v_max, t_max) + ctrl_mode from the motor; keep config
        defaults for any register that does not answer."""
        got = {}
        for name, rid in (("p_max", _RID_PMAX), ("v_max", _RID_VMAX),
                          ("t_max", _RID_TMAX)):
            try:
                v = self.read_param(rid)
            except SubsystemUnavailable:
                return
            if v is not None and v > 0.0:
                got[name] = float(v)
        if len(got) == 3:
            self.p_max, self.v_max, self.t_max = got["p_max"], got["v_max"], got["t_max"]
            self.scales_from_motor = True
        try:
            mode = self.read_param(_RID_CTRL_MODE)
            self.ctrl_mode = None if mode is None else int(mode)
        except SubsystemUnavailable:
            pass

    # ── telemetry ────────────────────────────────────────────────────────────
    def read(self) -> DamiaoSerialState:
        """Poll the motor (0xCC, non-energizing). ``live=False`` (with the last-
        known pose) when it does not answer — dead 24 V reads as not-live."""
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            st = self._poll_once()
        if st is None:
            old = self._state
            st = DamiaoSerialState(live=False, position=old.position,
                                   velocity=0.0, torque=0.0, t_mos=old.t_mos,
                                   t_rotor=old.t_rotor, err=old.err, stamp=old.stamp)
            self._state = st
        return st

    # ── commands ─────────────────────────────────────────────────────────────
    def _mit(self, p_des: float, v_des: float, kp: float, kd: float,
             t_ff: float) -> Optional[DamiaoSerialState]:
        p = max(-self.p_max, min(self.p_max, float(p_des)))
        self._drain_rx()
        self._send(self.can_id, mit_frame(p, v_des, kp, kd, t_ff,
                                          self.p_max, self.v_max, self.t_max))
        d = self._rx_until(self._feedback_match, 0.2)
        return self._ingest_feedback(d) if d is not None else None

    def set_torque(self, on: bool) -> None:
        """Energize (SAFE HOLD at the measured pose — no lurch) or free the motor."""
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            if on:
                st = self._poll_once()
                if st is None:
                    raise SubsystemUnavailable(
                        "motor did not answer the pre-enable poll — check 24 V "
                        "power / CAN wiring before energizing")
                self._send(self.can_id, [0xFF] * 7 + [_CMD_ENABLE])
                # Command "stay here" in the same breath: an enabled MIT motor
                # with no target produces no holding torque, and a stale target
                # would snap.
                self._mit(st.position, 0.0, self.kp, self.kd, 0.0)
            else:
                self._send(self.can_id, [0xFF] * 7 + [_CMD_DISABLE])
            self._torque = bool(on)

    def set_position(self, raw_rad: float) -> DamiaoSerialState:
        """MIT position command (raw motor radians). The caller owns range
        clamping to the CALIBRATED window; this clamps only to ±p_max packing.
        Requires torque ON."""
        return self.set_target(raw_rad, 0.0)

    def set_target(self, raw_rad: float, v_ff: float = 0.0) -> DamiaoSerialState:
        """MIT position command WITH velocity feedforward — the smooth-tracking
        variant for streamed trajectories. The MIT law is a true PD,
        ``tau = kp·(p_des − p) + kd·(v_des − v)``: with v_des = 0 a moving target
        makes kd fight the motion (stair-step jitter); passing the trajectory's
        own slew rate as ``v_ff`` lets kd damp AROUND the motion instead (the
        same fix the arms needed for teleop tracking). Requires torque ON."""
        with self._lock:
            if self._dev is None:
                raise SubsystemUnavailable("DM-USB2FDCAN dongle not connected")
            if self._torque is not True:
                raise SubsystemUnavailable("torque is OFF — enable before commanding")
            st = self._mit(float(raw_rad), float(v_ff), self.kp, self.kd, 0.0)
            return st if st is not None else self._state

    def stop(self) -> None:
        """Halt NOW: re-target the present position (torque on) — the MIT loop
        holds it. A de-energized motor is already free; nothing to do."""
        with self._lock:
            if self._dev is None or self._torque is not True:
                return
            st = self._poll_once()
            if st is not None:
                self._mit(st.position, 0.0, self.kp, self.kd, 0.0)
