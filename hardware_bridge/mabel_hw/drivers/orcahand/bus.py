"""Feetech STS/SCS serial transport -- the only place that speaks the wire.

A ``FeetechBus`` owns one serial port and talks to every motor hanging off it.
It is deliberately *dumb*: it deals in raw hardware units (0-4095 ticks, raw
speed/load, deg C, volts) and motor IDs. All joint-name <-> motor-ID mapping
and radian conversion live one layer up in :class:`orcahand.hand.OrcaHand`.

This is a dependency-free implementation of the Feetech 1.0 packet protocol
over :mod:`pyserial` -- no vendor SDK required, so it drops cleanly into a ROS
node or a bare control script alike. It is the single, canonical transport that
replaces the ~5 ad-hoc ``Bus`` classes that used to live across the firmware.

Packet framing (every packet)::

    0xFF 0xFF  ID  LEN  INST  [PARAMS...]  CHECKSUM
    LEN      = number of bytes after LEN (INST + params + checksum)
    CHECKSUM = ~(ID + LEN + INST + params) & 0xFF
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterable, List, Optional

import serial

from . import constants as C
from .ports import detect_port


class FeetechBus:
    """Half-duplex Feetech bus over a single serial port.

    Thread-safe: every transaction is guarded by an internal lock, so a
    feedback-polling thread and a command thread can share one bus.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = C.DEFAULT_BAUDRATE,
        timeout: float = 0.01,
    ):
        self.port = port or detect_port()
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        # Latched after ONE failed extended (15-byte) feedback read: an older
        # servo/firmware without Present_Current then keeps the classic 8-byte
        # path at full rate instead of eating a timeout every cycle.
        self._no_ext = False

    # ------------------------------------------------------------- lifecycle
    def open(self) -> "FeetechBus":
        if self._ser is None or not self._ser.is_open:
            # exclusive=True takes a POSIX advisory lock (TIOCEXCL) so a SECOND
            # process can't open the same adapter and trample the bus -- e.g. the
            # left + right hand nodes both auto-detecting the one plugged-in
            # adapter. The loser raises here and its node idles cleanly instead of
            # both corrupting each other's reads.
            self._ser = serial.Serial(self.port, self.baudrate,
                                      timeout=self.timeout, exclusive=True)
        return self

    def close(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def __enter__(self) -> "FeetechBus":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- framing
    @staticmethod
    def _checksum(body: List[int]) -> int:
        """``body`` is every byte after the 0xFF 0xFF header, minus checksum."""
        return (~sum(body)) & 0xFF

    def _write_packet(self, body: List[int]) -> None:
        pkt = bytearray([0xFF, 0xFF] + body + [self._checksum(body)])
        self._ser.write(pkt)

    def _require_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("FeetechBus is not open; call .open() first.")

    # ----------------------------------------------------------------- writes
    def write_register(self, motor_id: int, address: int, data: Iterable[int]) -> None:
        """WRITE instruction to one id (or ``BROADCAST_ID`` for all)."""
        data = list(data)
        with self._lock:
            self._require_open()
            body = [motor_id, len(data) + 3, C.INST_WRITE, address] + data
            self._write_packet(body)

    def ofs_cal(self, motor_id: int, position: int = 2048) -> None:
        """Offset-calibration instruction (0x0B): redefine the motor's CURRENT
        physical pose to read as ``position`` (default 2048 = centre).

        The servo computes the encoder offset itself and burns it to EEPROM --
        no register write or EEPROM unlock/lock needed. Disable torque first
        (the offset cannot be burned safely while the motor is driving). This is
        the dependency-free equivalent of the vendor SDK's ``reOfsCal``; the
        wire packet is ``FF FF ID 04 0x0B pos_lo pos_hi CHK``.
        """
        pos = int(position) & 0xFFFF
        body = [motor_id, 4, C.INST_OFSCAL, pos & 0xFF, (pos >> 8) & 0xFF]
        with self._lock:
            self._require_open()
            self._write_packet(body)
            time.sleep(0.05)            # let the servo write EEPROM
            self._ser.reset_input_buffer()   # drop the OFSCAL status reply

    # ------------------------------------------------------------ generic read
    def read_register(self, motor_id: int, address: int, length: int = 1,
                      timeout: float = 0.02) -> Optional[int]:
        """READ ``length`` bytes from a control-table ``address`` of one motor.

        Returns the little-endian unsigned value, or ``None`` on timeout / bad
        packet. Used by the GUI's register editor and EEPROM-config tools; the
        hot feedback path uses :meth:`read_feedback` / :meth:`read_positions`.
        """
        with self._lock:
            self._require_open()
            self._ser.reset_input_buffer()
            self._write_packet([motor_id, 4, C.INST_READ, address, length])
            want = 6 + length  # FF FF ID LEN ERR D0..D(length-1) CHK
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self._ser.in_waiting >= want:
                    resp = self._ser.read(want)
                    if resp[0] != 0xFF or resp[1] != 0xFF or resp[2] != motor_id:
                        return None
                    data = resp[5:5 + length]
                    return int.from_bytes(data, "little")
            return None

    def sync_write_positions(self, id_to_raw: Dict[int, int]) -> None:
        """SYNC_WRITE 2-byte goal positions to many motors in one packet.

        This is the hot path -- one round-trip-free packet commands the whole
        hand. Speed/accel/torque come from the per-motor registers set by
        :meth:`set_goal_speed` / :meth:`set_goal_acc` / :meth:`set_torque_limit`.
        """
        if not id_to_raw:
            return
        per = 2  # goal position is 2 bytes
        ids = list(id_to_raw.keys())
        length = (per + 1) * len(ids) + 4
        body = [C.BROADCAST_ID, length, C.INST_SYNC_WRITE, C.ADDR_GOAL_POSITION, per]
        for sid in ids:
            raw = int(id_to_raw[sid]) & 0xFFFF
            body += [sid, raw & 0xFF, (raw >> 8) & 0xFF]
        with self._lock:
            self._require_open()
            self._write_packet(body)

    # ------------------------------------------------------------ bulk config
    def set_torque_enable(self, enable: bool, motor_id: int = C.BROADCAST_ID) -> None:
        self.write_register(motor_id, C.ADDR_TORQUE_ENABLE, [1 if enable else 0])

    def set_torque_limit(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        self.write_register(motor_id, C.ADDR_TORQUE_LIMIT,
                            [value & 0xFF, (value >> 8) & 0xFF])

    def set_goal_speed(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        self.write_register(motor_id, C.ADDR_GOAL_SPEED,
                            [value & 0xFF, (value >> 8) & 0xFF])

    def set_goal_acc(self, value: int, motor_id: int = C.BROADCAST_ID) -> None:
        self.write_register(motor_id, C.ADDR_GOAL_ACC, [value & 0xFF])

    # ------------------------------------------------------------------ reads
    def ping(self, motor_id: int, timeout: float = 0.02) -> bool:
        """True if ``motor_id`` answers a PING within ``timeout`` seconds."""
        with self._lock:
            self._require_open()
            self._ser.reset_input_buffer()
            self._write_packet([motor_id, 2, C.INST_PING])
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self._ser.in_waiting >= 6:
                    resp = self._ser.read(6)
                    if resp[0] == 0xFF and resp[1] == 0xFF and resp[2] == motor_id:
                        return True
            return False

    def scan(self, id_range: Iterable[int] = range(1, 21)) -> List[int]:
        """Return the sorted list of motor IDs that answer a PING."""
        return [i for i in id_range if self.ping(i)]

    def read_feedback(self, motor_id: int, timeout: float = 0.01) -> Optional[dict]:
        """Read one motor's feedback block.

        Reads the EXTENDED block (0x38..0x46) so `current` — the only measured
        torque signal the servo has — comes back in the same transaction as the
        classic 8 bytes. A servo/firmware that will not serve 15 bytes falls back
        to the 8-byte block ONCE and is remembered, so a hand that predates the
        current register keeps working exactly as before (just without torque).

        Returns decoded *raw* hardware units, or ``None`` on timeout / bad packet::

            position    : 0-4095 ticks
            speed       : signed ticks/s
            load        : signed -1000..1000   (PWM duty ‰ — NOT a torque)
            voltage     : volts
            temperature : deg C
            current     : signed amps          (absent if the servo has no register)
        """
        if not self._no_ext:
            fb = self._read_block(motor_id, C.FEEDBACK_EXT_LEN, timeout)
            if fb is not None:
                return fb
            # One clean fallback: don't pay the extended timeout every cycle.
            self._no_ext = True
        return self._read_block(motor_id, C.FEEDBACK_LEN, timeout)

    def _read_block(self, motor_id: int, rd_len: int,
                    timeout: float) -> Optional[dict]:
        with self._lock:
            self._require_open()
            self._ser.reset_input_buffer()
            self._write_packet(
                [motor_id, 4, C.INST_READ, C.ADDR_PRESENT_POSITION, rd_len])
            want = 6 + rd_len          # FF FF ID LEN ERR D0..Dn CHK
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self._ser.in_waiting >= want:
                    resp = self._ser.read(want)
                    if resp[0] != 0xFF or resp[1] != 0xFF or resp[2] != motor_id:
                        return None
                    return self._decode_feedback(resp[5:5 + rd_len])
            return None

    def read_positions(self, motor_ids: List[int], timeout: float = 0.01) -> Dict[int, int]:
        """Fast position-only read of many motors via SYNC_READ (0x82).

        Falls back to per-motor reads if the bus/adapter does not return a
        clean concatenated sync-read reply. Missing motors are simply absent
        from the returned dict.
        """
        if not motor_ids:
            return {}
        try:
            result = self._sync_read_positions(motor_ids, timeout)
            if result:
                return result
        except Exception:
            pass
        # Fallback: individual feedback reads (slower but rock solid).
        out: Dict[int, int] = {}
        for mid in motor_ids:
            fb = self.read_feedback(mid, timeout)
            if fb is not None:
                out[mid] = fb["position"]
        return out

    def _sync_read_positions(self, motor_ids: List[int], timeout: float) -> Dict[int, int]:
        rd_len = 2
        with self._lock:
            self._require_open()
            self._ser.reset_input_buffer()
            length = 4 + len(motor_ids)
            body = ([C.BROADCAST_ID, length, C.INST_SYNC_READ,
                     C.ADDR_PRESENT_POSITION, rd_len] + list(motor_ids))
            self._write_packet(body)

            # Each motor replies FF FF ID LEN ERR D0 D1 CHK -> 8 bytes.
            per = 6 + rd_len
            want = per * len(motor_ids)
            buf = bytearray()
            t0 = time.time()
            while time.time() - t0 < timeout * len(motor_ids):
                if self._ser.in_waiting:
                    buf += self._ser.read(self._ser.in_waiting)
                    if len(buf) >= want:
                        break
        return self._parse_sync_read(bytes(buf), set(motor_ids), rd_len)

    @staticmethod
    def _parse_sync_read(buf: bytes, ids: set, rd_len: int) -> Dict[int, int]:
        out: Dict[int, int] = {}
        i = 0
        n = len(buf)
        while i + 6 + rd_len <= n:
            if buf[i] != 0xFF or buf[i + 1] != 0xFF:
                i += 1
                continue
            mid = buf[i + 2]
            d = buf[i + 5:i + 5 + rd_len]
            if mid in ids and len(d) == rd_len:
                pos = d[0] | (d[1] << 8)
                # Drop physically-impossible reads: a single-turn STS encoder
                # only ever reports 0-4095. Anything larger is a garbled / mis-
                # aligned packet (e.g. a dropped byte) and must not be trusted --
                # such reads otherwise poison end-stop calibration (see
                # _decode_feedback).
                if C.RAW_MIN <= pos <= C.RAW_MAX:
                    out[mid] = pos
            i += 6 + rd_len
        return out

    @staticmethod
    def _decode_feedback(d: bytes) -> Optional[dict]:
        pos = d[0] | (d[1] << 8)
        # A single-turn STS/SCS encoder only reports 0-4095. A position outside
        # that window is a corrupt packet (dropped/extra byte, bus glitch), not a
        # real reading -- treat the whole frame as a miss rather than letting a
        # bogus tick like 18433 / 51199 propagate into telemetry or calibration.
        if not (C.RAW_MIN <= pos <= C.RAW_MAX):
            return None

        speed = d[2] | (d[3] << 8)
        if speed & 0x8000:                 # sign-magnitude, bit15 = direction
            speed = -(speed & 0x7FFF)

        load = d[4] | (d[5] << 8)
        if load & 0x0400:                  # sign-magnitude, bit10 = direction
            load = -(load & 0x03FF)
        else:
            load = load & 0x03FF

        out = {
            "position": pos,
            "speed": speed,
            "load": load,                  # PWM duty ‰ — a drive effort, NOT torque
            "voltage": d[6] / 10.0,        # 0.1 V units
            "temperature": float(d[7]),    # deg C
        }
        # Present_Current (0x45) rides at the tail of the extended block. This is
        # the MEASURED signal: torque = Kt·I. Absent on a short (8-byte) read.
        if len(d) >= C.FEEDBACK_EXT_LEN:
            cur = d[13] | (d[14] << 8)
            if cur & 0x8000:               # sign-magnitude, bit15 = direction
                cur = -(cur & 0x7FFF)
            out["current"] = cur * C.CURRENT_LSB_A      # amps
        return out
