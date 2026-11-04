"""damiao_serial.py — DaMiao motor over the DaMiao USB-to-CAN *serial* dongle.

The DaMiao dongle (HDSC MCU, USB vendor 0x2E88) is NOT a socketcan device — it wraps
each CAN frame in DaMiao's own binary serial framing (verified against
cmjang/DM_Control_Python, the firmware ``dm_configurator.py``, and live hardware):

  host → dongle : 30 bytes — ``55 AA 1E …`` with the arbitration id LE at [13:17],
                  DLC 8 at [18] and the 8 CAN payload bytes at [21:29]
  dongle → host : 16 bytes — ``AA <cmd> <dlc> <canid LE ×4> <8 data> 55``

``damiao_can.CanLink`` (socketcan) stays the arms/body bench backend; this driver is
for the single dongle-attached motor — today MABEL's torso (DM-J10422P, CAN id 1,
MST id 0x11). The setup studio talks to it directly (HAL stopped) for the torso's
manual calibration + slider control, the same direct-bus pattern as the serial neck
and the lift Pico.

MIT-mode framing is byte-identical to ``damiao_can.DamiaoBus`` (one packing law,
``f2u``/``u2f`` imported from there). The fixed-point (p_max, v_max, t_max) scales are
READ FROM THE MOTOR's own registers (RIDs 21/22/23) at connect — a mismatch there
silently rescales every sample and every command, so we never trust config defaults
when the motor can tell us itself (calibration/README.md's rule).

Safety contract (mirrors the neck/lift studio drivers):
  • connect() NEVER energizes — it opens the serial and reads registers only.
  • set_torque(True) is a SAFE HOLD: present position is read first and commanded as
    the MIT target in the same breath as enable, so the motor holds where it is and
    never snaps to a stale setpoint.
  • set_torque(False) (0xFD) is the free direction — E-stop path, and how the operator
    drags the joint by hand during manual calibration.
  • read() polls with 0xCC — the official non-energizing refresh; it can never move
    or energize the motor.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from ..errors import SubsystemUnavailable
from ..types import monotonic
from .damiao_can import f2u, u2f

__all__ = ["DamiaoSerialDriver", "DamiaoSerialState", "resolve_port"]

_AUTO = "auto"

# The DaMiao dongle enumerates as "HDSC CDC Device", USB vendor 0x2E88. ttyACM
# numbering is arrival-order (ACM0 is often the base Teensy, 0x16C0), so we locate
# the dongle by vendor id exactly like the lift finds its Pico (0x2E8A) — a replug
# that renumbers the port is re-resolved on the next connect.
_DM_VID = 0x2E88

_PARAM_BUS_ID = 0x7FF               # register read/write frames are addressed here
_CMD_ENABLE, _CMD_DISABLE = 0xFC, 0xFD
_CMD_REFRESH = 0xCC                 # non-energizing "report your status"
_RID_CTRL_MODE, _RID_PMAX, _RID_VMAX, _RID_TMAX = 10, 21, 22, 23

# 30-byte host->dongle template; build_frame patches [13:17] and [21:29].
_TX_TEMPLATE = bytes([
    0x55, 0xAA, 0x1E, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00,
    0x00, 0, 0, 0, 0, 0x00, 0x08, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x00,
])
_RX_LEN, _RX_HEAD, _RX_TAIL, _RX_DATA_CMD = 16, 0xAA, 0x55, 0x11


def build_frame(arb_id: int, data8) -> bytes:
    """One CAN frame (id + 8 payload bytes) in the dongle's 30-byte serial wrap."""
    f = bytearray(_TX_TEMPLATE)
    f[13] = arb_id & 0xFF
    f[14] = (arb_id >> 8) & 0xFF
    f[15] = (arb_id >> 16) & 0xFF
    f[16] = (arb_id >> 24) & 0xFF
    f[21:29] = bytes(data8)
    return bytes(f)


def parse_frames(buf: bytes):
    """Yield (can_id, data8) for every well-formed 16-byte rx frame in ``buf``."""
    i, n = 0, len(buf)
    while i <= n - _RX_LEN:
        if buf[i] == _RX_HEAD and buf[i + _RX_LEN - 1] == _RX_TAIL:
            p = buf[i:i + _RX_LEN]
            arb = p[3] | (p[4] << 8) | (p[5] << 16) | (p[6] << 24)
            yield arb, bytes(p[7:15])
            i += _RX_LEN
        else:
            i += 1


def resolve_port(port: str = _AUTO) -> Optional[str]:
    """An explicit path wins verbatim; ``auto`` finds the dongle by USB vendor id."""
    if port and port != _AUTO:
        return port
    try:
        from serial.tools import list_ports as lp             # noqa: PLC0415
    except Exception:
        return None
    cands = sorted((c for c in lp.comports() if c.vid == _DM_VID),
                   key=lambda c: c.device)
    return cands[0].device if cands else None


class DamiaoSerialState:
    """One motor's latest feedback. ``live`` is per-read: the motor answered THIS poll.

    ``err`` is the raw feedback byte-0 high nibble, which is the motor's STATE, not
    purely an error code (measured on the DM-J10422): 0 = disabled, 1 = enabled (MIT),
    and only ``0x8``+ are faults (8 over-V, 9 under-V, A over-I, B MOS over-T, C coil
    over-T, D comm loss, E overload). ``enabled``/``fault`` pre-split it so no caller
    ever renders "fault 0x1" on a healthy energized motor again."""
    __slots__ = ("live", "position", "velocity", "torque", "t_mos", "t_rotor",
                 "err", "stamp")

    def __init__(self, live=False, position=0.0, velocity=0.0, torque=0.0,
                 t_mos=0, t_rotor=0, err=0, stamp=0.0):
        self.live, self.position, self.velocity, self.torque = live, position, velocity, torque
        self.t_mos, self.t_rotor, self.err, self.stamp = t_mos, t_rotor, err, stamp

    @property
    def enabled(self) -> bool:
        """Torque is ON per the MOTOR's own feedback (state nibble == 1)."""
        return self.err == 1

    @property
    def fault(self) -> int:
        """The fault code (0x8..0xE), or 0 when the state nibble is a healthy state."""
        return self.err if self.err >= 0x8 else 0


class DamiaoSerialDriver:
    """Owns the dongle serial link for ONE motor. Thread-safe; request/response —
    every TX waits for its reply, so no RX thread and no shared-bus arbitration."""

    def __init__(self, port: str = _AUTO, baud: int = 921600, can_id: int = 1,
                 kp: float = 30.0, kd: float = 1.0, p_max: float = 12.5,
                 v_max: float = 30.0, t_max: float = 10.0, logger=None):
        self._port, self._baud, self.can_id = port, int(baud), int(can_id)
        self.kp, self.kd = float(kp), float(kd)
        # Fixed-point MIT scales: config seeds them; connect() overwrites from the
        # motor's own registers (the wire truth) whenever the reads succeed.
        self.p_max, self.v_max, self.t_max = float(p_max), float(v_max), float(t_max)
        self.scales_from_motor = False
        self.ctrl_mode: Optional[int] = None       # 1 = MIT (RID 10), read at connect
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._ser = None
        self._resolved: Optional[str] = None
        self._state = DamiaoSerialState()
        self._torque: Optional[bool] = None        # last commanded state (None = never)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        """Open the dongle and read the motor's scale registers. Never energizes."""
        with self._lock:
            if self._ser is not None:
                return
            import serial                                      # noqa: PLC0415
            dev = resolve_port(self._port)
            if not dev:
                raise SubsystemUnavailable(
                    f"no DaMiao USB-CAN dongle found (no USB vendor 0x{_DM_VID:04X} "
                    "serial device; is it plugged in?)")
            try:
                self._ser = serial.Serial(dev, self._baud, timeout=0.05)
            except Exception as e:
                raise SubsystemUnavailable(
                    f"could not open DaMiao dongle {dev}: {e} — is the HAL (or another "
                    "studio) holding it?") from e
            self._resolved = dev
            time.sleep(0.2)
            self._ser.reset_input_buffer()
        self._read_scales()
        self._log(f"[damiao-serial] {dev} @ {self._baud}, motor id {self.can_id}, "
                  f"ranges p±{self.p_max} v±{self.v_max} t±{self.t_max} "
                  f"({'motor registers' if self.scales_from_motor else 'config defaults'})")

    def connected(self) -> bool:
        return self._ser is not None

    @property
    def resolved_port(self) -> Optional[str]:
        return self._resolved

    @property
    def is_torque_enabled(self) -> Optional[bool]:
        return self._torque

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    # Leaving a motor energized with nobody watching is never safe.
                    self._ser.write(build_frame(self.can_id, [0xFF] * 7 + [_CMD_DISABLE]))
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                self._torque = None

    # ── wire primitives (call with lock held) ────────────────────────────────
    def _tx(self, arb_id: int, data8) -> None:
        if self._ser is None:
            raise SubsystemUnavailable("DaMiao dongle not connected")
        try:
            self._ser.write(build_frame(arb_id, data8))
        except Exception as e:
            # A yanked dongle re-enumerates on replug; drop the socket so the next
            # connect() re-resolves the port by VID (lift.py's hot-replug pattern).
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            self._torque = None
            raise SubsystemUnavailable(f"DaMiao dongle write failed: {e}") from e

    def _rx_until(self, match, timeout: float = 0.25):
        """Read frames until ``match(arb, data)`` returns non-None, or timeout."""
        deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < deadline:
            chunk = self._ser.read(256)
            if chunk:
                buf += chunk
                for arb, d in parse_frames(buf):
                    got = match(arb, d)
                    if got is not None:
                        return got
                # keep any trailing partial frame
                buf = buf[-(_RX_LEN - 1):] if len(buf) >= _RX_LEN else buf
        return None

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
        # MIT feedback rides the motor's MST id; the payload carries the slave id in
        # byte0's low nibble. Match on the payload (works whatever MST_ID is set to)
        # and ignore the 0x7FF param bus.
        if arb != _PARAM_BUS_ID and len(d) == 8 and (d[0] & 0x0F) == (self.can_id & 0x0F):
            return d
        return None

    def _poll_once(self, timeout: float = 0.25) -> Optional[DamiaoSerialState]:
        """0xCC refresh → one feedback frame. Non-energizing. Lock held by caller."""
        self._ser.reset_input_buffer()
        self._tx(_PARAM_BUS_ID, [self.can_id & 0xFF, (self.can_id >> 8) & 0xFF,
                                 _CMD_REFRESH, 0, 0, 0, 0, 0])
        d = self._rx_until(self._feedback_match, timeout)
        return self._ingest_feedback(d) if d is not None else None

    # ── registers ────────────────────────────────────────────────────────────
    def read_param(self, rid: int, timeout: float = 0.3) -> Optional[float]:
        """Read one motor register over the 0x7FF param bus (0x33 — config-only,
        never moves anything). uint32 for the int RIDs, float32 otherwise."""
        import struct                                          # noqa: PLC0415
        with self._lock:
            if self._ser is None:
                raise SubsystemUnavailable("DaMiao dongle not connected")
            self._ser.reset_input_buffer()
            self._tx(_PARAM_BUS_ID, [self.can_id & 0xFF, (self.can_id >> 8) & 0xFF,
                                     0x33, int(rid), 0, 0, 0, 0])

            def match(arb, d):
                if len(d) == 8 and d[2] == 0x33 and d[3] == int(rid) and \
                        (d[0] | (d[1] << 8)) == self.can_id:
                    return d
                return None

            d = self._rx_until(match, timeout)
        if d is None:
            return None
        if (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36):
            return float(int.from_bytes(d[4:8], "little"))
        return struct.unpack("<f", d[4:8])[0]

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
        """Poll the motor (0xCC, non-energizing). ``live=False`` (with the last-known
        pose) when it does not answer — dead 24 V power reads as not-live, not a crash."""
        with self._lock:
            if self._ser is None:
                raise SubsystemUnavailable("DaMiao dongle not connected")
            st = self._poll_once()
            if st is not None:
                return st
            stale = self._state
            return DamiaoSerialState(live=False, position=stale.position,
                                     velocity=0.0, torque=0.0, t_mos=stale.t_mos,
                                     t_rotor=stale.t_rotor, err=stale.err,
                                     stamp=stale.stamp)

    # ── control ──────────────────────────────────────────────────────────────
    def _mit(self, p_des: float, v_des: float, kp: float, kd: float,
             t_ff: float) -> Optional[DamiaoSerialState]:
        """One MIT frame (packing law == damiao_can.send_mit); returns the feedback
        the motor answers with. Lock held by caller."""
        pi = f2u(p_des, -self.p_max, self.p_max, 16)
        vi = f2u(v_des, -self.v_max, self.v_max, 12)
        ki = f2u(kp, 0.0, 500.0, 12)
        di = f2u(kd, 0.0, 5.0, 12)
        ti = f2u(t_ff, -self.t_max, self.t_max, 12)
        self._ser.reset_input_buffer()
        self._tx(self.can_id, [
            (pi >> 8) & 0xFF, pi & 0xFF,
            (vi >> 4) & 0xFF, ((vi & 0xF) << 4) | ((ki >> 8) & 0xF), ki & 0xFF,
            (di >> 4) & 0xFF, ((di & 0xF) << 4) | ((ti >> 8) & 0xF), ti & 0xFF])
        d = self._rx_until(self._feedback_match, 0.2)
        return self._ingest_feedback(d) if d is not None else None

    def set_torque(self, on: bool, hold_t_ff: float = 0.0) -> None:
        """Energize (SAFE HOLD at the measured pose — no lurch) or free the motor.

        ``hold_t_ff`` rides the hold frame's feed-forward field: a gravity-loaded joint
        (the torso) enables straight into its gravity torque instead of sagging one
        PD-error's worth before holding."""
        with self._lock:
            if self._ser is None:
                raise SubsystemUnavailable("DaMiao dongle not connected")
            if on:
                st = self._poll_once()
                if st is None:
                    raise SubsystemUnavailable(
                        "motor did not answer the pre-enable poll — check 24 V power / "
                        "CAN wiring before energizing")
                self._tx(self.can_id, [0xFF] * 7 + [_CMD_ENABLE])
                # Command "stay here" in the same breath: an enabled MIT motor with no
                # target yet produces no holding torque, and a stale target would snap.
                self._mit(st.position, 0.0, self.kp, self.kd, float(hold_t_ff))
            else:
                self._tx(self.can_id, [0xFF] * 7 + [_CMD_DISABLE])
            self._torque = bool(on)

    def set_position(self, raw_rad: float, t_ff: float = 0.0) -> DamiaoSerialState:
        """MIT position command (raw motor radians) with optional torque feed-forward
        (N·m, raw-motor frame — the full MIT law τ = kp·e + kd·ė + t_ff). The caller
        owns range clamping to the CALIBRATED window and the gravity model behind
        ``t_ff`` (controller/mabel/torso_control for the torso); this clamps only to
        the motor's ±p_max/±t_max packing ranges. Requires torque ON."""
        with self._lock:
            if self._ser is None:
                raise SubsystemUnavailable("DaMiao dongle not connected")
            if self._torque is not True:
                raise SubsystemUnavailable("torque is OFF — enable before commanding")
            st = self._mit(float(raw_rad), 0.0, self.kp, self.kd, float(t_ff))
            return st if st is not None else self._state

    def set_target(self, raw_rad: float, v_ff: float = 0.0,
                   t_ff: float = 0.0) -> DamiaoSerialState:
        """MIT position command WITH velocity feedforward (and optional torque
        feed-forward) — the smooth-tracking variant for streamed trajectories;
        see damiao_fdcan.set_target for the PD rationale. Same surface as the
        FDCAN driver so seam adapters (drivers/solo.py) treat both dongles
        identically. Requires torque ON."""
        with self._lock:
            if self._ser is None:
                raise SubsystemUnavailable("DaMiao dongle not connected")
            if self._torque is not True:
                raise SubsystemUnavailable("torque is OFF — enable before commanding")
            st = self._mit(float(raw_rad), float(v_ff), self.kp, self.kd, float(t_ff))
            return st if st is not None else self._state

    def stop(self, t_ff: float = 0.0) -> None:
        """Halt NOW: re-target the present position (torque on) — the MIT loop then
        holds it there (with ``t_ff`` so a gravity-loaded joint holds without sag).
        A de-energized motor is already free; nothing to do."""
        with self._lock:
            if self._ser is None or self._torque is not True:
                return
            st = self._poll_once()
            if st is not None:
                self._mit(st.position, 0.0, self.kp, self.kd, float(t_ff))
