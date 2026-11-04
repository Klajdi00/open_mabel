"""
damiao.py
=========

Host-side driver for DaMiao actuators in **MIT mode**, used by the calibration
collection.  Rebuilt on the proven transport layer from the firmware teleop GUI
(:mod:`transports`) so it speaks to the motor exactly the way the working
``firmware/openarm`` driver does — over **serial**, **gs_usb** (CANable /
candleLight, the macOS path for those adapters), or **socketcan**.

MIT-mode control law (what the motor runs internally):

    tau = kp * (p_des - p) + kd * (v_des - v) + t_ff

so the calibration drives:
  * **velocity hold**  -> kp = 0, kd > 0, v_des = level        (staircase)
  * **position track** -> kp > 0, kd > 0, p_des/v_des = ref    (sine / chirp)
  * **pure torque**    -> kp = kd = 0, t_ff = tau              (open loop)

Telemetry follows the firmware/official-library design: a motor's pos/vel/tau
stream from its feedback frames, which a motor emits both for MIT control frames
*and* for the non-energizing ``0xCC`` refresh poll.  So ``refresh_status`` keeps a
motor live on the bus without enabling torque; ``enable`` (``0xFC``) is purely for
torque control.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from .motors import MotorProfile, get as get_motor
from .transports import make_transport, auto_interface

GLOBAL_ID = 0x7FF      # status/refresh frames are addressed here (per DM_CAN)


# ----------------------------------------------------------------- packing
def _float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    x = min(max(x, lo), hi)
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))


def _uint_to_float(i: int, lo: float, hi: float, bits: int) -> float:
    return i * (hi - lo) / ((1 << bits) - 1) + lo


class MotorFeedback:
    __slots__ = ("t", "pos", "vel", "tau", "t_mos", "t_rotor", "err")

    def __init__(self):
        self.t = 0.0
        self.pos = self.vel = self.tau = 0.0
        self.t_mos = self.t_rotor = 0.0
        self.err = 0


class DamiaoBus:
    """One CAN bus carrying one or more DaMiao motors in MIT mode.

    Backend-agnostic: ``interface`` is ``serial`` | ``gs_usb`` | ``socketcan``;
    ``None`` auto-selects per platform (macOS serial-adapter-or-gs_usb, Linux
    socketcan), matching the firmware teleop GUI.
    """

    def __init__(self, channel: str = "can0", interface: Optional[str] = None,
                 bitrate: int = 1_000_000, port: Optional[str] = None):
        self.interface = interface or auto_interface()
        self._channel = channel
        self._bitrate = bitrate
        self._port = port
        self._recovering = False
        self._tx_err = 0

        self._transport = make_transport(self.interface, channel=channel,
                                         port=port, bitrate=bitrate)
        self.profiles: Dict[int, MotorProfile] = {}
        self.state: Dict[int, MotorFeedback] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._rx = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx.start()

    # ------------------------------------------------------------ registry
    def add_motor(self, can_id: int, profile: MotorProfile | str) -> None:
        prof = get_motor(profile) if isinstance(profile, str) else profile
        with self._lock:
            self.profiles[can_id] = prof
            self.state.setdefault(can_id, MotorFeedback())

    # --------------------------------------------------------- raw control
    def _send(self, arb_id: int, data: list[int]) -> None:
        """Transmit one frame, recovering the bus after repeated TX errors.

        On gs_usb/serial recovery simply reopens the USB device; on socketcan it
        would need an `ip link` bounce (left to the caller / firmware GUI).
        """
        try:
            self._transport.send(arb_id, data)
            self._tx_err = 0
        except Exception:
            self._tx_err += 1
            if self._tx_err >= 3:
                self._tx_err = 0
                self._recover()
            raise

    def _recover(self) -> None:
        if self._recovering:
            return
        self._recovering = True
        try:
            try:
                self._transport.close()
            except Exception:
                pass
            time.sleep(0.6)
            self._transport = make_transport(self.interface, channel=self._channel,
                                             port=self._port, bitrate=self._bitrate)
        except Exception:
            pass
        finally:
            self._recovering = False

    def _cmd(self, can_id: int, byte: int) -> None:
        """Non-MIT control command (enable/disable/clear/zero): FF…FF <byte>."""
        self._send(can_id, [0xFF] * 7 + [byte])

    def enable(self, can_id: int) -> None:
        """Enter MIT mode (0xFC) so the motor accepts torque. Telemetry already
        streams via refresh_status / control frames — enable is only for torque."""
        self._cmd(can_id, 0xFC)

    def disable(self, can_id: int) -> None:
        """Exit MIT mode (0xFD): torque off, motor free; still answers refresh."""
        self._cmd(can_id, 0xFD)

    def clear_faults(self, can_id: int) -> None:
        self._cmd(can_id, 0xFB)

    def set_zero(self, can_id: int) -> None:
        self._cmd(can_id, 0xFE)

    def refresh_status(self, can_id: int) -> None:
        """Non-energizing telemetry poll (0xCC, the official refresh_motor_status):
        makes the motor emit one feedback frame without entering MIT mode."""
        self._send(GLOBAL_ID,
                   [can_id & 0xFF, (can_id >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])

    def scan(self, ids=range(1, 16), bursts: int = 3, settle: float = 0.02
             ) -> list[int]:
        """Probe ``ids`` with the non-energizing 0xCC query and return those that
        answer.  Repeats the sweep ``bursts`` times and waits for the rx loop to
        catch the replies, so a momentarily-busy bus doesn't read as empty."""
        ids = list(ids)
        t0 = time.perf_counter()
        for _ in range(max(1, bursts)):
            for mid in ids:
                try:
                    self._send(GLOBAL_ID,
                               [mid & 0xFF, (mid >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])
                except Exception:
                    pass
                time.sleep(settle)
        time.sleep(0.2)        # let the rx loop parse the last replies
        with self._lock:
            return sorted(cid for cid, fb in self.state.items() if fb.t >= t0)

    def mit(self, can_id: int, p_des: float, v_des: float,
            kp: float, kd: float, t_ff: float) -> None:
        """Send one MIT-mode command frame."""
        prof = self.profiles[can_id]
        p = _float_to_uint(p_des, -prof.p_max, prof.p_max, 16)
        v = _float_to_uint(v_des, -prof.v_max, prof.v_max, 12)
        kpi = _float_to_uint(kp, 0.0, prof.kp_max, 12)
        kdi = _float_to_uint(kd, 0.0, prof.kd_max, 12)
        t = _float_to_uint(t_ff, -prof.t_max, prof.t_max, 12)
        self._send(can_id, [
            (p >> 8) & 0xFF, p & 0xFF,
            (v >> 4) & 0xFF, ((v & 0xF) << 4) | ((kpi >> 8) & 0xF), kpi & 0xFF,
            (kdi >> 4) & 0xFF, ((kdi & 0xF) << 4) | ((t >> 8) & 0xF), t & 0xFF,
        ])

    # convenience wrappers --------------------------------------------------
    def hold_velocity(self, can_id: int, w: float, kd: float) -> None:
        self.mit(can_id, 0.0, w, 0.0, kd, 0.0)

    def track_position(self, can_id: int, q: float, w: float,
                       kp: float, kd: float) -> None:
        self.mit(can_id, q, w, kp, kd, 0.0)

    def torque(self, can_id: int, t_ff: float) -> None:
        self.mit(can_id, 0.0, 0.0, 0.0, 0.0, t_ff)

    # ----------------------------------------------------------- feedback
    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            if self._recovering:
                time.sleep(0.1)
                continue
            try:
                frame = self._transport.poll(timeout=0.2)
            except Exception:
                time.sleep(0.2)
                continue
            if frame is None:
                continue
            self._parse(frame[1])

    def _parse(self, d: bytes) -> None:
        if len(d) < 6:
            return
        # canonical DaMiao MIT feedback: byte0 low nibble = id, high = error
        can_id = d[0] & 0x0F
        with self._lock:
            prof = self.profiles.get(can_id)
            if prof is None:
                return
            p_int = (d[1] << 8) | d[2]
            v_int = (d[3] << 4) | (d[4] >> 4)
            t_int = ((d[4] & 0xF) << 8) | d[5]
            fb = self.state.setdefault(can_id, MotorFeedback())
            fb.t = time.perf_counter()
            fb.err = d[0] >> 4
            fb.pos = _uint_to_float(p_int, -prof.p_max, prof.p_max, 16)
            fb.vel = _uint_to_float(v_int, -prof.v_max, prof.v_max, 12)
            fb.tau = _uint_to_float(t_int, -prof.t_max, prof.t_max, 12)
            if len(d) >= 8:
                fb.t_mos, fb.t_rotor = float(d[6]), float(d[7])

    def get(self, can_id: int) -> MotorFeedback:
        with self._lock:
            fb = self.state.setdefault(can_id, MotorFeedback())
            out = MotorFeedback()
            for s in MotorFeedback.__slots__:
                setattr(out, s, getattr(fb, s))
            return out

    # ------------------------------------------------------------- cleanup
    def shutdown(self, disable_all: bool = True) -> None:
        if disable_all:
            for cid in list(self.profiles):
                try:
                    self.disable(cid)
                except Exception:
                    pass
        self._stop.set()
        try:
            self._rx.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._transport.close()
        except Exception:
            pass

    def __enter__(self) -> "DamiaoBus":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
