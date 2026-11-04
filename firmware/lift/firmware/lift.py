# lift.py - High-level lift state machine.
#
# Owns the motor, encoder, and cascaded controller, and arbitrates between
# operating modes, soft limits, stall-based calibration, and homing. The
# rest of the system talks to a Lift instance and calls update() every tick.

import utime
import json

import config
import storage
from motor import Motor
from encoder import Encoder
from controller import VelocityLoop, MotionProfile


def emit_log(msg):
    """Send a human-readable event to the host as a JSON {"log": ...} line so
    it shows up in the GUI event log (plain prints get filtered out)."""
    try:
        print(json.dumps({"log": msg}))
    except Exception:
        pass

# Operating modes.
MODE_IDLE       = 0
MODE_MANUAL     = 1   # open-loop fixed effort
MODE_VELOCITY   = 2   # closed-loop velocity hold
MODE_POSITION   = 3   # trapezoidal move to a height
MODE_CALIB_DOWN = 4
MODE_CALIB_UP   = 5
MODE_HOME       = 6

MODE_NAMES = {
    MODE_IDLE: "IDLE", MODE_MANUAL: "MANUAL", MODE_VELOCITY: "VELOCITY",
    MODE_POSITION: "POSITION", MODE_CALIB_DOWN: "CALIB_DOWN",
    MODE_CALIB_UP: "CALIB_UP", MODE_HOME: "HOME",
}

# Status codes (broad health/limit state, distinct from mode).
STATUS_READY, STATUS_MAX, STATUS_MIN, STATUS_MOVING, STATUS_CONTACT = 0, 1, 2, 3, 4
STATUS_NAMES = {0: "READY", 1: "MAX", 2: "MIN", 3: "MOVING", 4: "CONTACT"}


class Lift:
    def __init__(self):
        self.motor = Motor()
        self.encoder = Encoder()
        self.vloop = VelocityLoop()
        self.profile = MotionProfile()

        self.mode = MODE_IDLE
        self.status = STATUS_READY

        self._manual_effort = 0.0
        self._move_start_ms = utime.ticks_ms()
        self._move_start_count = 0
        self._armed = False         # True once the encoder confirms real motion

        # A latched fault. Set when a commanded move produced no encoder ticks at
        # all; while it is set, a move in the SAME direction is refused. Without this
        # a host that streams a latched setpoint re-commands the move the instant the
        # firmware coasts, so the column is driven into the stall again every couple
        # of seconds at full effort — indefinitely. Cleared by stop(), a torque
        # cycle, home/calibrate/sethome, or a target the other way.
        self.fault_dir = 0            # 0 none | +1 up faulted | -1 down faulted
        self.is_calibrated = False
        self.min_height_mm = config.LIFT_MIN_HEIGHT_MM
        self.max_height_mm = config.LIFT_MAX_HEIGHT_MM
        self.park_mm = config.DEFAULT_PARK_MM
        self.home_height_mm = config.HOME_HEIGHT_MM
        self.torque_on = True       # H-bridge enabled (motor can drive/hold)

        self._calib_bottom_count = 0   # encoder count recorded at the floor
        self._seek_effort = 0.0        # open-loop drive for calibrate/home
        self._last_cmd_dir = 0         # sign of the last commanded effort
        self._free_seen = False        # has this move proven it can run free?
        self._slow_since = 0           # ticks_ms when the speed first went low
        self.contact_mm = None         # height where contact was last detected
        self._arrived = False          # inside the stop band; hysteresis latch

        # Restore persisted scale/limits/position so the lift remembers where it
        # is across a power cycle (assumes it wasn't back-driven while off).
        self._restore()

        # The desired height starts AT the measured height, never at 0 and never at
        # whatever was last commanded. A host that reads 'target' on connect and
        # echoes it back therefore holds station instead of driving the column to a
        # stale setpoint — the MABEL failure where a fresh server bring-up sent the
        # default 0 and the lift ran all the way to the floor.
        self.profile.set_target(self.position_mm, None, 0.0)

    def _restore(self):
        st = storage.load()
        if not st:
            return
        self.encoder.set_scale(st.get("mm_per_count", config.MM_PER_COUNT))
        self.min_height_mm = st.get("min_height", config.LIFT_MIN_HEIGHT_MM)
        self.max_height_mm = st.get("max_height", config.LIFT_MAX_HEIGHT_MM)
        self.is_calibrated = bool(st.get("calibrated", False))
        # Map the current (boot) count to the last-known height.
        self.encoder.set_position_mm(st.get("height", config.LIFT_MIN_HEIGHT_MM))

    def _save(self):
        try:
            storage.save(self.encoder.mm_per_count, self.min_height_mm,
                         self.max_height_mm, self.position_mm, self.is_calibrated)
        except Exception:
            pass     # never let a flash write disrupt the control loop

    # ------------------------------------------------------------------ utils
    @property
    def position_mm(self):
        return self.encoder.position_mm

    @property
    def velocity_mm_s(self):
        return self.encoder.velocity_mm_s

    @property
    def target_mm(self):
        """The height the lift is currently trying to reach.

        Only meaningful while driving to a position; when idle it reads back the
        height the column is resting at, so 'target' is always a safe value for a
        host to adopt as its own setpoint on connect.
        """
        if self.mode == MODE_POSITION:
            return self.profile.target_mm
        return self.position_mm

    def _begin_move(self):
        self._move_start_ms = utime.ticks_ms()
        self.encoder.last_tick_ms = self._move_start_ms
        self._move_start_count = self.encoder.count
        self._armed = False
        self._free_seen = False
        self._slow_since = 0

    def _enforce_soft_limit(self, effort):
        """Zero out effort that would drive past a calibrated end stop."""
        if not (self.is_calibrated and config.LIMIT_ENABLED):
            return effort
        pos = self.position_mm
        if effort > 0 and pos >= self.max_height_mm:
            return 0.0
        if effort < 0 and pos <= self.min_height_mm:
            return 0.0
        return effort

    # --------------------------------------------------------------- commands
    def stop(self):
        self.fault_dir = 0            # an explicit stop clears the latch
        was_moving = self.status == STATUS_MOVING
        self.motor.coast()
        self.vloop.reset()
        self.mode = MODE_IDLE
        if self.status == STATUS_MOVING:
            self.status = STATUS_READY
        if was_moving and self.is_calibrated:
            self._save()       # remember where we came to rest

    def set_torque(self, on):
        """Enable/disable the H-bridge. Off => motor free, ignores motion."""
        if on:
            self.motor.enable()
            self.torque_on = True
            self.fault_dir = 0        # a torque cycle clears the latch
        else:
            self.stop()
            self.motor.disable()
            self.torque_on = False

    def set_manual(self, effort):
        """Open-loop drive at a fixed signed effort (-1..1)."""
        self.mode = MODE_MANUAL
        self._manual_effort = effort
        self.status = STATUS_MOVING
        self._begin_move()

    def set_velocity(self, mm_s):
        """Closed-loop velocity hold at `mm_s` (signed: +up / -down)."""
        self.mode = MODE_VELOCITY
        self.vloop.reset()
        self.vloop.setpoint = mm_s
        self.status = STATUS_MOVING
        self._begin_move()

    def set_position(self, target_mm, v_max=None):
        """Move to an absolute height in mm. ALWAYS adopts the newest target.

        Two callers, one path. A one-shot 'pos 700' starts a fresh trapezoid; a
        policy or animation streaming setpoints at 50 Hz retargets the move that
        is already in flight, so the column always chases the most recent desired
        position and never has to arrive before it can be redirected.

        The old code compared the new target against the CURRENT profile target
        and ignored anything within POSITION_TOL_MM (0.5 mm) of it, to avoid
        re-planning on a held setpoint. That silently broke streaming: an
        animation moving slower than 0.5 mm per frame had most of its frames
        dropped, and every frame that WAS accepted called vloop.reset() +
        _begin_move(), which wiped the velocity integral and restarted the stall
        timer ~25 times a second — so end-stop detection could never fire.

        Now a repeat is cheap (retarget only, no reset), so nothing is dropped and
        nothing is restarted. The stall timers keep running across a retarget and
        are only rearmed when the direction of travel actually flips.
        """
        if self.is_calibrated and config.LIMIT_ENABLED:
            if target_mm < self.min_height_mm:
                target_mm = self.min_height_mm
            elif target_mm > self.max_height_mm:
                target_mm = self.max_height_mm

        want_dir = 1 if target_mm > self.position_mm else -1
        if self.fault_dir != 0 and want_dir == self.fault_dir:
            # Refused: this direction already proved it cannot move. Re-driving is
            # what turned one stall into an endless series of them.
            return
        if self.fault_dir != 0 and want_dir != self.fault_dir:
            self.fault_dir = 0        # the other way is worth a try
            emit_log("Fault cleared by a reversed target")

        if self.mode == MODE_POSITION:
            # Already driving to a height: move the goalposts, keep the motion.
            was_going_up = (self.profile.target_mm - self.position_mm) > 0
            now_going_up = (target_mm - self.position_mm) > 0
            self.profile.retarget(target_mm, v_max)
            if was_going_up != now_going_up:
                # A reversal passes through zero speed, so the encoder legitimately
                # stops ticking for a moment. Rearm, or the no-tick cut fires on our
                # own turnaround and reports a phantom end stop.
                self._begin_move()
            return

        self.mode = MODE_POSITION
        self.vloop.reset()
        self.profile.set_target(target_mm, v_max, self.velocity_mm_s)
        self.status = STATUS_MOVING
        self._arrived = False
        self._begin_move()

    def park(self):
        """Go to the resting pose (mid-stroke by default)."""
        if not self.is_calibrated:
            return False
        self.set_position(self.park_mm, config.MAX_VELOCITY_MM_S * 0.8)
        return True

    def calibrate(self):
        self.fault_dir = 0
        """Two-stage stall calibration: seek the floor, then the ceiling.

        The physical stroke (LIFT_TRAVEL_MM) is known, so the measured encoder
        count between the two ends gives the exact counts->mm scale. Result is
        persisted to flash.
        """
        self.is_calibrated = False
        self.mode = MODE_CALIB_DOWN
        self._seek_effort = -config.CALIB_DOWN_EFFORT   # open-loop, drive down
        self.status = STATUS_MOVING
        self._begin_move()
        emit_log("Calibrate: seeking floor (down @ {:.0f}%)".format(
            config.CALIB_DOWN_EFFORT * 100))

    def home(self):
        """Stall-seek the floor and define it as the bottom datum."""
        self.fault_dir = 0
        self.mode = MODE_HOME
        self._seek_effort = -config.CALIB_DOWN_EFFORT   # open-loop, drive down
        self.status = STATUS_MOVING
        self._begin_move()
        emit_log("Home: seeking floor (down)")

    def set_home(self):
        """Treat the CURRENT position as the home/bottom datum (no motion).

        Defines here as LIFT_MIN_HEIGHT and the top as min + fixed travel,
        then marks the lift calibrated.
        """
        self.stop()
        self.encoder.set_position_mm(config.LIFT_MIN_HEIGHT_MM)
        self.min_height_mm = config.LIFT_MIN_HEIGHT_MM
        self.max_height_mm = config.LIFT_MIN_HEIGHT_MM + config.LIFT_TRAVEL_MM
        self.is_calibrated = True
        self.status = STATUS_MIN
        self._save()

    # ------------------------------------------------------------- main tick
    def update(self, now_ms, dt):
        self.encoder.update(now_ms)
        self.motor.sample_current()

        if not self.torque_on:
            self.motor.coast()
            self.encoder.set_direction(0)
            return

        if self.mode == MODE_IDLE:
            self.motor.coast()
            self.encoder.set_direction(0)
            return

        if self.mode == MODE_MANUAL:
            effort = self._enforce_soft_limit(self._manual_effort)
            self.motor.set_effort(effort)

        elif self.mode in (MODE_CALIB_DOWN, MODE_CALIB_UP, MODE_HOME):
            # Open-loop seek at a fixed effort — robust, no velocity-loop
            # dependence, and bypasses soft limits (we WANT to reach the stops).
            self.motor.set_effort(self._seek_effort)

        elif self.mode == MODE_VELOCITY:
            effort = self.vloop.compute(self.vloop.setpoint,
                                        self.velocity_mm_s, dt)
            effort = self._enforce_soft_limit(effort)
            self.motor.set_effort(effort)

        elif self.mode == MODE_POSITION:
            distance = self.profile.target_mm - self.position_mm
            if config.BANG_BANG:
                # Full duty toward the target, TAPERED near it, with hysteresis on the
                # stop band. See config.BANG_BANG for why each part is needed.
                ad = abs(distance)
                if ad <= config.STOP_BAND_MM:
                    self._arrived = True
                if self._arrived and ad <= config.RESTART_BAND_MM:
                    # Hold. Re-driving inside the restart band is the limit cycle.
                    self.motor.coast()
                    self.vloop.reset()
                    self.mode = MODE_IDLE
                    self.status = STATUS_READY
                    if self.is_calibrated:
                        self._save()
                    return
                self._arrived = False
                effort = config.MAX_EFFORT
                if ad < config.TAPER_MM:
                    span = config.TAPER_MM - config.STOP_BAND_MM
                    frac = (ad - config.STOP_BAND_MM) / span if span > 0 else 1.0
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    effort = (config.TAPER_MIN_EFFORT
                              + (config.MAX_EFFORT - config.TAPER_MIN_EFFORT) * frac)
                if distance < 0:
                    effort = -effort
            else:
                if self.profile.at_target(self.position_mm, self.velocity_mm_s):
                    self.motor.coast()
                    self.vloop.reset()
                    self.mode = MODE_IDLE
                    self.status = STATUS_READY
                    if self.is_calibrated:
                        self._save()   # remember the arrival position
                    return
                v_set = self.profile.compute(self.position_mm, dt)
                effort = self.vloop.compute(v_set, self.velocity_mm_s, dt)
            effort = self._enforce_soft_limit(effort)
            self.motor.set_effort(effort)

        # Tell the encoder which way we're driving so it counts correctly.
        e = self.motor.effort
        if e != 0.0:
            self._last_cmd_dir = 1 if e > 0 else -1
        self.encoder.set_direction(1 if e > 0 else (-1 if e < 0 else 0))

        self._check_contact(now_ms)
        self._check_stall(now_ms)

    def _check_contact(self, now_ms):
        """External load from a speed deficit at known duty.

        No force sensor is needed: the free-running speed at a given duty is measured
        (config.CONTACT_V_SLOPE/OFFSET), so a column that WAS running free and then
        slows is being resisted. This sees a partial obstruction, which the no-tick
        cut cannot — that needs the column stopped outright.

        Deliberately requires proof of free running first. Acceleration from rest and
        the stiction regime are both legitimately slow, and treating those as contact
        is exactly the kind of false trip that would break continuous commanding.
        """
        if not config.CONTACT_DETECT:
            return
        eff = abs(self.motor.effort)
        if eff < config.CONTACT_MIN_EFFORT or not self._armed:
            self._slow_since = 0
            return
        if self.mode in (MODE_CALIB_DOWN, MODE_CALIB_UP, MODE_HOME):
            return                      # these drive INTO the stops on purpose
        v_free = config.CONTACT_V_SLOPE * eff + config.CONTACT_V_OFFSET
        if v_free <= 1.0:
            return
        v = abs(self.velocity_mm_s)
        if v >= config.CONTACT_ARM_FRAC * v_free:
            self._free_seen = True      # proven: it can run free at this duty
            self._slow_since = 0
            return
        if not self._free_seen:
            return                      # still getting up to speed
        if v >= config.CONTACT_TRIP_FRAC * v_free:
            self._slow_since = 0
            return
        if self._slow_since == 0:
            self._slow_since = now_ms
            return
        if utime.ticks_diff(now_ms, self._slow_since) < config.CONTACT_TRIP_MS:
            return
        # Held slow while running at real duty, having already run free: contact.
        # REPORT ONLY unless config.CONTACT_STOP — the robot may be carrying a load,
        # and extra force is information, not a reason to abandon the move.
        self.contact_mm = self.position_mm
        self._slow_since = 0
        la = self.motor.load_amps
        emit_log("CONTACT at {:.1f} mm: {:.1f} mm/s at duty {:.2f}, expected {:.1f}{}"
                 .format(self.position_mm, v, eff, v_free,
                         "" if la is None else ", extra load {:+.2f} A".format(la)))
        if config.CONTACT_STOP:
            self.motor.coast()
            self.vloop.reset()
            self.mode = MODE_IDLE
            self.status = STATUS_CONTACT
            self._free_seen = False
            if self.is_calibrated:
                self._save()

    # ---------------------------------------------------------- stall / limits
    def _check_stall(self, now_ms):
        if self.motor.effort == 0.0:
            return
        # Calibration/homing MUST always detect the end stops, regardless of the
        # soft-limit toggle (the toggle only governs normal-move protection).
        calibrating = self.mode in (MODE_CALIB_DOWN, MODE_CALIB_UP, MODE_HOME)
        if not calibrating and not config.LIMIT_ENABLED:
            return

        # Arm only once the encoder proves the lift is really moving. Until
        # then we never declare an end-stop, so a slow start can't false-trip
        # and chop the motion short.
        if not self._armed:
            # The jog gets a longer window than a closed-loop move, but it is NOT
            # exempt any more: an open-loop 'up' with a dead encoder used to drive
            # at effort 1.0 indefinitely.
            no_motion_ms = (config.MANUAL_NO_MOTION_FAULT_MS
                            if self.mode == MODE_MANUAL
                            else config.NO_MOTION_FAULT_MS)
            if self.encoder.count != self._move_start_count:
                self._armed = True
            elif (utime.ticks_diff(now_ms, self._move_start_ms)
                  >= no_motion_ms):
                if self.mode in (MODE_CALIB_DOWN, MODE_HOME):
                    # Seeking the floor but didn't move: we're already on it.
                    # Treat it as the end stop and continue the sequence.
                    emit_log("Already at floor (no motion); continuing")
                    self._on_stall(now_ms)
                else:
                    # Jog, closed-loop move or up-sweep with zero feedback => real
                    # fault: either the encoder is not wired/powered, or the column
                    # cannot move (no H-bridge supply, jammed, already on the stop).
                    self.motor.coast()
                    self.vloop.reset()
                    failed = MODE_NAMES.get(self.mode, "?")
                    # LATCH the direction so a streaming host cannot immediately
                    # re-command the same move and grind into the stall again.
                    e = self.motor.effort
                    self.fault_dir = 1 if self._last_cmd_dir > 0 else -1
                    self.mode = MODE_IDLE
                    self.status = STATUS_READY
                    emit_log("FAULT in {}: no encoder ticks (count={}). "
                             "Check encoder wiring A=GP27 B=GP28, or that the "
                             "lift can move that way.".format(
                                 failed, self.encoder.count))
            return

        # Armed: a sustained loss of ticks means we've hit a physical end stop.
        # Calibration/homing sweeps use a forgiving window (a slow patch must
        # not false-trip); normal moves use the fast jam cut.
        if self.mode in (MODE_CALIB_DOWN, MODE_CALIB_UP, MODE_HOME):
            grace, no_tick = config.CALIB_GRACE_MS, config.CALIB_NO_TICK_LIMIT_MS
        else:
            grace, no_tick = config.GRACE_MS, config.NO_TICK_LIMIT_MS
        since_move = utime.ticks_diff(now_ms, self._move_start_ms)
        since_tick = utime.ticks_diff(now_ms, self.encoder.last_tick_ms)
        if since_move < grace or since_tick < no_tick:
            return
        self._on_stall(now_ms)

    def _on_stall(self, now_ms):
        going_down = self.motor.effort < 0
        self.motor.coast()
        self.vloop.reset()

        if self.mode == MODE_CALIB_DOWN:
            # Floor found: record the count here, then seek the ceiling.
            self._calib_bottom_count = self.encoder.count
            emit_log("Floor found (count={}); seeking ceiling (up @ {:.0f}%)"
                     .format(self._calib_bottom_count,
                             config.CALIB_UP_EFFORT * 100))
            utime.sleep_ms(400)            # settle off the dead-stop
            self.mode = MODE_CALIB_UP
            self._seek_effort = config.CALIB_UP_EFFORT   # open-loop, drive up
            self.status = STATUS_MOVING
            self._begin_move()

        elif self.mode == MODE_CALIB_UP:
            # Ceiling found. The count spanned between the two hard stops maps
            # to the known physical stroke -> exact counts->mm scale.
            span = self.encoder.count - self._calib_bottom_count
            if abs(span) > 10:
                self.encoder.set_scale(config.LIFT_TRAVEL_MM / abs(span))
            # Pin the absolute frame: the recorded bottom count == LIFT_MIN.
            self.encoder.set_reference(self._calib_bottom_count,
                                       config.LIFT_MIN_HEIGHT_MM)
            self.min_height_mm = config.LIFT_MIN_HEIGHT_MM
            self.max_height_mm = config.LIFT_MIN_HEIGHT_MM + config.LIFT_TRAVEL_MM
            self.is_calibrated = True
            self.mode = MODE_IDLE
            self.status = STATUS_MAX
            self._save()
            emit_log("CALIBRATED: span={} counts, {:.5f} mm/count, "
                     "range {:.0f}-{:.0f} mm".format(
                         span, self.encoder.mm_per_count,
                         self.min_height_mm, self.max_height_mm))

        elif self.mode == MODE_HOME:
            self.encoder.set_position_mm(self.home_height_mm)
            self.mode = MODE_IDLE
            self.status = STATUS_MIN
            self._save()
            emit_log("Homed at {:.0f} mm".format(self.position_mm))

        else:
            # MANUAL / VELOCITY / POSITION hit a jam or an end stop: STOP ONLY.
            # Never redefine the origin here — a mid-travel jam (or a fast
            # protective cut) must not corrupt the absolute encoder position.
            # The datum is re-established only by home / calibrate / sethome.
            if going_down and self.position_mm <= self.min_height_mm + 5.0:
                self.status = STATUS_MIN
            elif (not going_down) and self.position_mm >= self.max_height_mm - 5.0:
                self.status = STATUS_MAX
            else:
                self.status = STATUS_READY
            self.mode = MODE_IDLE
            if self.is_calibrated:
                self._save()
