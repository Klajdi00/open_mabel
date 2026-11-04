"""Hardware abstraction over the REV SPARK Flex CAN controllers.

The GUI never touches sparkcan_py directly — it talks to a SwerveBus, which
owns one SparkFlex per CAN ID and runs three daemon threads:

    * heartbeat  — keeps every controller enabled (SPARKs disable without it)
    * send       — pushes the latest setpoints for *enabled* motors
    * poll       — reads telemetry (pos / vel / current / temp) into a cache

Telemetry getters beyond GetVelocity / GetAbsoluteEncoderPosition are not yet
exposed by the pybind bindings.  The bus probes each instance with ``hasattr``
at construction, so the moment the extension is rebuilt with the extra getters
(see sparkcan_py/bindings/pybind11/sparkcan_bindings.cpp) the temperature and
current readouts light up automatically — no GUI change required.

If the SparkFlex extension or the CAN interface is unavailable, the bus falls
back to a self-contained simulation so the interface is still fully reviewable.
"""

import math
import random
import threading
import time

from . import config as C


class MotorState:
    """Live command + telemetry for a single controller."""

    __slots__ = ("can_id", "role", "enabled", "setpoint",
                 "pos", "vel", "current", "temp", "voltage", "online",
                 "pos_fresh", "vel_fresh")

    def __init__(self, can_id, role):
        self.can_id = can_id
        self.role = role
        self.enabled = False
        self.setpoint = 0.0
        self.pos = float("nan")
        self.vel = float("nan")
        self.current = float("nan")
        self.temp = float("nan")
        self.voltage = float("nan")
        self.online = False
        self.pos_fresh = False   # is the encoder/position frame actually updating?
        self.vel_fresh = False   # is the velocity frame actually updating?


class SwerveBus:
    """Thread-safe controller for all swerve motors on one CAN interface."""

    def __init__(self, iface=C.CAN_IFACE, mock=False):
        self.iface = iface
        self.mock = mock
        self.roles = C.motor_roles()
        self.offsets = C.steer_offsets()
        self.ids = C.all_motor_ids()
        self.states = {mid: MotorState(mid, self.roles[mid]) for mid in self.ids}

        self._motors = {}        # can_id -> SparkFlex (real mode only)
        self._caps = {}          # capability flags discovered via hasattr
        self._applied = {}       # can_id -> last duty actually sent (for slew ramp)
        # Staleness tracking: {can_id: {"pos": [last_val, last_change_t], "vel": [...]}}
        self._fresh = {mid: {"pos": [None, 0.0], "vel": [None, 0.0]} for mid in self.ids}
        self._lock = threading.Lock()
        self._running = True
        self.error = None        # human-readable reason we fell into mock mode

        if not self.mock:
            self._connect()

        self._threads = [
            threading.Thread(target=self._heartbeat_loop, daemon=True),
            threading.Thread(target=self._send_loop, daemon=True),
            threading.Thread(target=self._poll_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    # ── Connection ────────────────────────────────────────────────────────────
    def _connect(self):
        try:
            from sparkcan_py import SparkFlex
        except Exception as e:                      # extension not built
            self.mock = True
            self.error = f"sparkcan_py unavailable ({e}); running in SIM mode."
            return
        try:
            for mid in self.ids:
                self._motors[mid] = SparkFlex(self.iface, mid)
                time.sleep(0.003)                   # spacing, mirrors motortest2.py
        except Exception as e:                      # CAN down / iface missing
            self._motors.clear()
            self.mock = True
            self.error = f"Cannot open {self.iface} ({e}); running in SIM mode."
            return
        # Probe which methods this build of the extension exposes.
        sample = next(iter(self._motors.values()))
        for attr in ("GetVelocity", "GetAbsoluteEncoderPosition",
                     "GetAbsoluteEncoderVelocity", "GetPosition",
                     "GetTemperature", "GetCurrent", "GetVoltage",
                     "SetDutyCycle", "ClearStickyFaults", "SetCtrlType",
                     "SetIdleMode", "SetPeriodicStatus0Period",
                     "SetPeriodicStatus2Period", "SetPeriodicStatus5Period"):
            self._caps[attr] = hasattr(sample, attr)

        # Configure closed-loop control type + idle mode, like the YOR robot.
        # Without the right ctrl type a controller can ignore our setpoints
        # (e.g. SetVelocity is dropped if the flashed kCtrlType is duty-cycle),
        # and steer motors need kPosition so they servo the absolute encoder.
        if C.APPLY_CTRL_TYPE and self._caps.get("SetCtrlType"):
            try:
                from sparkcan_py import CtrlType, IdleMode
                drive_ct = CtrlType.kDutyCycle if C.DRIVE_USE_DUTY else CtrlType.kVelocity
                for mid, m in self._motors.items():
                    m.SetCtrlType(drive_ct if self.roles[mid] == C.DRIVE else CtrlType.kPosition)
                    if C.SET_IDLE_COAST and self._caps.get("SetIdleMode"):
                        m.SetIdleMode(IdleMode.kCoast)
            except Exception:
                pass

        # Tell every controller to actually broadcast the telemetry frames we
        # read. Without this, frame 2 (velocity) and frame 5 (steer absolute
        # encoder) never arrive and those readouts stay blank.
        for m in self._motors.values():
            try:
                if self._caps.get("SetPeriodicStatus0Period"):
                    m.SetPeriodicStatus0Period(C.STATUS0_PERIOD_MS)
                if self._caps.get("SetPeriodicStatus2Period"):
                    m.SetPeriodicStatus2Period(C.STATUS2_PERIOD_MS)
                if self._caps.get("SetPeriodicStatus5Period"):
                    m.SetPeriodicStatus5Period(C.STATUS5_PERIOD_MS)
            except Exception:
                pass

        # ── Safe startup ──────────────────────────────────────────────────────
        # CRITICAL: the heartbeat thread has NOT started yet, so no controller is
        # enabled. Before it ever can be, overwrite whatever stale closed-loop
        # setpoint each controller is still holding from a previous session with
        # a harmless open-loop zero, and clear sticky faults. This guarantees the
        # motors stay still the instant heartbeats begin — no startup lurch.
        for m in self._motors.values():
            try:
                if self._caps.get("ClearStickyFaults"):
                    m.ClearStickyFaults()
                self._idle(m)
            except Exception:
                pass

    def _idle(self, motor):
        """Command a single controller to a limp, motionless state.

        Prefers open-loop duty-cycle 0 (true coast, no PID activity). Falls back
        to velocity 0 only if an older extension build lacks SetDutyCycle."""
        if self._caps.get("SetDutyCycle"):
            motor.SetDutyCycle(0.0)
        else:
            motor.SetVelocity(0.0)

    @property
    def connected(self) -> bool:
        return not self.mock

    def capability(self, attr: str) -> bool:
        return bool(self._caps.get(attr, False))

    # ── Commands (called from the UI thread) ───────────────────────────────────
    def set_enabled(self, can_id: int, on: bool):
        with self._lock:
            st = self.states[can_id]
            st.enabled = on
            if not on:
                st.setpoint = 0.0

    def set_setpoint(self, can_id: int, value: float):
        with self._lock:
            self.states[can_id].setpoint = float(value)

    def enable_all(self, on: bool):
        for mid in self.ids:
            self.set_enabled(mid, on)

    def stop_all(self):
        """E-STOP: disable everything and command every controller limp NOW.

        Disabling clears the enabled flags (so the heartbeat thread stops
        energizing the bus), and we immediately push an open-loop zero to every
        controller so any in-flight motion is cut without waiting for the loop."""
        with self._lock:
            for st in self.states.values():
                st.enabled = False
                st.setpoint = 0.0
        self._applied = {mid: 0.0 for mid in self.ids}
        if not self.mock:
            for m in self._motors.values():
                try:
                    self._idle(m)
                except Exception:
                    pass

    def snapshot(self):
        """Return a shallow copy of every MotorState for the UI to render."""
        with self._lock:
            out = {}
            for mid, st in self.states.items():
                copy = MotorState(mid, st.role)
                for slot in MotorState.__slots__:
                    setattr(copy, slot, getattr(st, slot))
                out[mid] = copy
            return out

    # ── Background loops ───────────────────────────────────────────────────────
    def _heartbeat_loop(self):
        # Heartbeat continuously. The SPARK controllers only broadcast the
        # position/velocity (frame 2) and absolute-encoder (frame 5) status
        # frames *while they are being heartbeated* — so without a steady
        # heartbeat the steering-angle and velocity readouts never update, even
        # when you're only monitoring. (This is exactly what swerve_teleop.py
        # does to get live feedback.)
        #
        # Energizing every controller is safe here because: (a) on connect we
        # overwrite any stale setpoint with SetDutyCycle(0) before the first
        # heartbeat, and (b) the send loop holds every *disabled* motor at
        # open-loop duty-0 (limp) every tick. Heartbeat enables the output stage
        # but the commanded value is zero → no motion, exactly like the teleop
        # warmup.
        dt = 1.0 / C.HEARTBEAT_HZ
        while self._running:
            if not self.mock:
                for m in self._motors.values():
                    try:
                        m.Heartbeat()
                    except Exception:
                        pass
            time.sleep(dt)

    def _send_loop(self):
        dt = 1.0 / C.SEND_HZ
        while self._running:
            with self._lock:
                cmds = [(mid, st.role, st.enabled, st.setpoint)
                        for mid, st in self.states.items()]
            step = C.DRIVE_SLEW / C.SEND_HZ      # max duty change this tick
            if not self.mock:
                for mid, role, enabled, sp in cmds:
                    m = self._motors.get(mid)
                    if m is None:
                        continue
                    try:
                        if not enabled:
                            self._applied[mid] = 0.0
                            self._idle(m)         # limp/coast, no PID to oscillate
                        elif role == C.DRIVE:
                            self._drive_command(m, mid, sp, step)
                        else:
                            m.SetPosition(sp + self.offsets.get(mid, 0.0))
                    except Exception:
                        pass
            time.sleep(dt)

    def _drive_command(self, motor, mid, target, step):
        """Send a drive command, ramped toward ``target`` to avoid jerk/shake.

        Open-loop duty-cycle by default (no velocity PID, so nothing can
        oscillate). Falls back to closed-loop SetVelocity only if configured
        off duty mode or if the extension lacks SetDutyCycle."""
        if C.DRIVE_USE_DUTY and self._caps.get("SetDutyCycle"):
            cur = self._applied.get(mid, 0.0)
            if target > cur:
                cur = min(target, cur + step)
            else:
                cur = max(target, cur - step)
            self._applied[mid] = cur
            motor.SetDutyCycle(cur)
        else:
            motor.SetVelocity(target)

    def _poll_loop(self):
        dt = 1.0 / C.POLL_HZ
        while self._running:
            if self.mock:
                self._poll_mock()
            else:
                self._poll_real()
            time.sleep(dt)

    def _mark_fresh(self, mid, field, value):
        """Track whether a feedback value is actually updating. Returns True if
        it changed within TELEMETRY_STALE_SEC (i.e. the frame is broadcasting)."""
        if value != value:                       # NaN → no data at all
            return False
        rec = self._fresh[mid][field]
        now = time.time()
        if rec[0] is None or abs(value - rec[0]) > 1e-9:
            rec[0] = value
            rec[1] = now
        return (now - rec[1]) < C.TELEMETRY_STALE_SEC

    def _poll_real(self):
        nan = float("nan")
        for mid, m in self._motors.items():
            st = self.states[mid]
            try:
                if self.roles[mid] == C.STEER:
                    # Steer angle + rate come from the absolute encoder (frame 5).
                    pos = m.GetAbsoluteEncoderPosition() if self._caps.get("GetAbsoluteEncoderPosition") else nan
                    vel = m.GetAbsoluteEncoderVelocity() if self._caps.get("GetAbsoluteEncoderVelocity") else nan
                else:
                    # Drive position + wheel velocity come from the primary
                    # encoder (frame 2).
                    pos = m.GetPosition() if self._caps.get("GetPosition") else nan
                    vel = m.GetVelocity() if self._caps.get("GetVelocity") else nan
                cur = m.GetCurrent() if self._caps.get("GetCurrent") else nan
                tmp = m.GetTemperature() if self._caps.get("GetTemperature") else nan
                volt = m.GetVoltage() if self._caps.get("GetVoltage") else nan
                pos_fresh = self._mark_fresh(mid, "pos", pos)
                vel_fresh = self._mark_fresh(mid, "vel", vel)
                with self._lock:
                    st.pos, st.vel, st.current = pos, vel, cur
                    st.temp, st.voltage, st.online = tmp, volt, True
                    st.pos_fresh, st.vel_fresh = pos_fresh, vel_fresh
            except Exception:
                with self._lock:
                    st.online = False
                    st.pos_fresh = st.vel_fresh = False

    def _poll_mock(self):
        t = time.time()
        with self._lock:
            for mid, st in self.states.items():
                st.online = True
                if st.role == C.DRIVE:
                    target = st.setpoint if st.enabled else 0.0
                    st.vel = target * 1000.0 + random.uniform(-15, 15)
                    st.current = abs(target) * 9.0 + random.uniform(0, 1.5)
                    st.pos = (t * target * 0.2) % 1.0
                else:
                    target = (st.setpoint if st.enabled else 0.5)
                    st.pos = target + 0.01 * math.sin(t * 2 + mid)
                    st.vel = random.uniform(-5, 5)
                    st.current = random.uniform(0, 2.0) + (4 if st.enabled else 0)
                st.temp = 27.0 + abs(st.current) * 1.4 + math.sin(t * 0.3 + mid)
                st.voltage = 12.4 + random.uniform(-0.1, 0.1)
                st.pos_fresh = st.vel_fresh = True   # simulated frames always live

    def close(self):
        # Cut motion first (while threads are still alive), then stop the loops.
        self.stop_all()
        time.sleep(0.05)
        self._running = False
        time.sleep(0.05)
        for m in self._motors.values():
            try:
                self._idle(m)
            except Exception:
                pass
