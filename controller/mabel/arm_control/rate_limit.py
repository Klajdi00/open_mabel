"""rate_limit.py — joint-level max speed. The gate that keeps a slider jump from
becoming a torque spike.

WHY THIS EXISTS. The firmware closes the loop itself:

    T = kp*(pos - q) + kd*(vel - dq) + t_ff

`pos` is what we send; `q` is where the joint IS. So the torque the motor develops
is proportional to the POSITION ERROR we hand it, and the host cannot clamp `T`
directly — only the error. Send a target far from `q` (an operator dragging a
slider to an extreme, a stale seed, a joint that drifted while limp) and the motor
immediately pulls kp*error: at kp=30 a 0.8 rad jump is 24 N.m, which parks the
joint against whatever stops it and cooks the driver. The DaMiao's own over-current
trip is a LATCH that needs a physical power cycle.

So never command a joint more than `max_speed * dt` away from where it actually is.
The error — and therefore the torque — is then BOUNDED by kp * max_speed * dt, no
matter how far away the target is. The joint SLEWS to an extreme target at a known
speed instead of lunging at it, and an unreachable target just leaves it resting at
that bounded torque forever rather than stalling at full authority.

This is the ONLY speed limit in the stack: the HAL (hardware_bridge) deliberately
carries no slew/stall/thermal guard — it clamps t_ff to the motor's rated t_max and
the position to the calibrated end-stops, and nothing else. Bounding kp*error is
this library's job, per the repo invariant that `controller/` is the sole gate on
anything that moves the robot.

    limiter = JointRateLimiter(max_speed=1.5)          # rad/s
    safe = limiter.step(targets, measured)             # -> command these

Use `measured` (real encoders), NOT the previous command: limiting against the last
command lets the pair drift arbitrarily far from the joint while each STEP looks
small — the error, and the torque, grow unbounded anyway. Limiting against measured
is what actually bounds kp*error.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Mapping, Optional

__all__ = ["JointRateLimiter", "DEFAULT_MAX_SPEED"]

#: rad/s. The bound this buys is  kp * max_speed * dt  — so at kp=30 and a 50 Hz
#: command stream, 4.0 rad/s caps the position term at 30 * 4.0 * 0.02 = 2.4 N.m.
#: That is still an order of magnitude under a DM4310's 10 N.m rating (and ~20x under
#: a DM8009's 54), while crossing a 90 deg range in ~0.4 s — fast enough to teleop.
#: 1.5 rad/s (86 deg/s) was the first cut and the operator found it far too slow.
#: RAISING THIS IS SAFE UNTIL kp*max_speed*dt approaches a joint's t_max; past that
#: you are back to commanding stall-level torque on a far target. Override per joint
#: (a DM8009 shoulder can take more than a DM4310 wrist) or via MABEL_ARM_MAX_SPEED.
DEFAULT_MAX_SPEED = 4.0


class JointRateLimiter:
    """Bound every joint command to `max_speed` rad/s away from the MEASURED angle.

    Stateless w.r.t. the target (no accumulation to drift or wind up): each call is
    a pure function of (target, measured, elapsed). Per-joint overrides let a heavy
    shoulder move slower than a wrist.
    """

    def __init__(self, max_speed: float = DEFAULT_MAX_SPEED, *,
                 per_joint: Optional[Mapping[str, float]] = None,
                 max_dt: float = 0.1):
        self.max_speed = float(max_speed)
        self.per_joint = dict(per_joint or {})
        # A long stall (GC pause, a blocked publish, a paused debugger) must not
        # cash in as one huge allowed step — that is exactly the lunge we prevent.
        self.max_dt = float(max_dt)
        self._last_t: Optional[float] = None

    def speed_for(self, joint: str) -> float:
        return float(self.per_joint.get(joint, self.max_speed))

    def reset(self) -> None:
        """Forget the clock (call on resume/e-stop release so the first step after
        a pause is sized by a fresh dt, not by however long we were stopped)."""
        self._last_t = None

    def tick(self, now: Optional[float] = None) -> float:
        """Advance the clock ONCE and return the elapsed dt (clamped to max_dt).

        ⚠ Call this once per command PASS, then hand the result to every `step()`
        in that pass. Letting step() take its own dt per joint means the first
        joint consumes the whole interval and every other joint sees dt~=0 and
        freezes — a 14-joint arm then moves one joint and stalls the rest.
        """
        t = time.monotonic() if now is None else now
        dt = self.max_dt if self._last_t is None else (t - self._last_t)
        self._last_t = t
        return max(0.0, min(float(dt), self.max_dt))

    def step(self, targets: Mapping[str, float],
             measured: Mapping[str, float],
             dt: Optional[float] = None,
             now: Optional[float] = None) -> Dict[str, float]:
        """Return commandable targets: each within `max_speed * dt` of `measured`.

        `targets`/`measured` are MODEL-frame radians keyed by joint name. A joint
        absent from `measured` passes through unlimited — we cannot bound an error
        we cannot see, and silently dropping the command would be worse (the caller
        should not be commanding an unread joint anyway).
        """
        t = time.monotonic() if now is None else now
        if dt is None:
            dt = self.max_dt if self._last_t is None else (t - self._last_t)
        self._last_t = t
        dt = max(0.0, min(float(dt), self.max_dt))

        out: Dict[str, float] = {}
        for name, tgt in targets.items():
            tgt = float(tgt)
            cur = measured.get(name)
            if cur is None:
                out[name] = tgt
                continue
            step = self.speed_for(name) * dt
            delta = tgt - float(cur)
            if delta > step:
                out[name] = float(cur) + step
            elif delta < -step:
                out[name] = float(cur) - step
            else:
                out[name] = tgt                # already within reach: no clamp
        return out


# ── the glide limiter (2026-07-29) ───────────────────────────────────────────
#: rad/s — the fleet-wide "commanded blend" budget for LARGE jumps (green-ball
#: task moves, plan playback). Matches the iOS ArmPlan.safeBlendVel and the web
#: studio's motions.js SAFE_BLEND_VEL: a 2-3 rad task move takes 2-3 s.
DEFAULT_BLEND_SPEED = 1.0
#: rad/s^2 — velocity shaping. The glide reaches blend speed in ~0.17 s and
#: decelerates into the target, so d(command)/dt is continuous — the velocity
#: feedforward downstream differentiates the command, and step/kinked commands
#: were reaching the MIT kd term as spikes.
DEFAULT_MAX_ACCEL = 6.0
#: rad — the HARD error ceiling. History: 0.15 DRAGGED the held setpoint with a
#: pushed arm (plastic yield — the "torque drop after a certain region" report);
#: 0.45 made it a pure-spring ceiling; 2026-07-30 the operator asked for a much
#: higher one → 0.8 (kp·err ≈ 20 N·m at the kp-25 shoulders — a strong spring
#: through the whole realistic push range; the motor's own T_MAX register clamps
#: the instantaneous peak, and the thermal yield below guards sustained holds).
#: None/<=0 disables the ceiling entirely; MABEL_ARM_ERR_MAX overrides.
DEFAULT_ERR_MAX = 0.8
#: The THERMAL YIELD — sustained-stall protection replacing the old always-drag.
#: kp*error is firmware-side and unclampable, so error parked at the ceiling
#: means sustained near-rating torque (heat). If the error stays above
#: `yield_err` for `yield_after` seconds, the setpoint deliberately gives way at
#: `yield_rate` down to `yield_rest` (kp*0.12 ~= 3 N.m — BELOW the old 0.15
#: bound) and LATCHES there until the arm catches back up. Quick interactions
#: never see it; leaning on the arm for seconds feels the robot politely yield.
DEFAULT_YIELD_ERR = 0.25    # rad — sustained error above this arms the yield timer
DEFAULT_YIELD_AFTER = 2.0   # s   — how long before the yield engages
DEFAULT_YIELD_RATE = 0.4    # rad/s — how fast the setpoint gives way
DEFAULT_YIELD_REST = 0.12   # rad — the latched gentle-hold error ceiling
_BLEND_ENGAGE = 0.30      # rad — a target jump beyond this glides at blend speed
_BLEND_RELEASE = 0.02     # rad — hysteresis: release the blend latch on arrival
_STALE_S = 0.5            # s — a pass gap beyond this reseeds from measured


class JointGlideLimiter(JointRateLimiter):
    """Smooth, stateful joint glide with the same measured-error torque bound.

    WHY (the "green ball jitter", 2026-07-29): JointRateLimiter re-anchors the
    command to the MEASURED angle every tick (cmd = q_meas ± v·dt). During a
    multi-second task-space move that feeds encoder noise and the arm's own
    oscillation straight back into the command — a self-exciting chatter loop,
    amplified by the velocity feedforward differentiating the command. Small
    slider nudges finish within one step, which is why joint mode never showed
    it.

    THE FIX: integrate a persistent per-joint setpoint from the PREVIOUS
    COMMAND (smooth by construction — the measurement is out of the loop), with
    accel-limited, decelerate-into-target velocity shaping; and keep the safety
    guarantee as an explicit clamp: the setpoint may never sit more than
    `err_max` rad from the measured angle, so kp*error stays bounded exactly as
    before. A far target (>0.30 rad — a green-ball Execute) glides at
    `blend_speed` (1 rad/s → 2-3 s for a typical task move); near-tracking
    (sliders, eased app bursts) runs at `max_speed` and stays snappy.
    """

    def __init__(self, max_speed: float = DEFAULT_MAX_SPEED, *,
                 blend_speed: float = DEFAULT_BLEND_SPEED,
                 max_accel: float = DEFAULT_MAX_ACCEL,
                 err_max: "float | None" = DEFAULT_ERR_MAX,
                 yield_err: float = DEFAULT_YIELD_ERR,
                 yield_after: float = DEFAULT_YIELD_AFTER,
                 yield_rate: float = DEFAULT_YIELD_RATE,
                 yield_rest: float = DEFAULT_YIELD_REST,
                 per_joint: Optional[Mapping[str, float]] = None,
                 per_joint_err: Optional[Mapping[str, float]] = None,
                 no_yield: Optional[Iterable[str]] = None,
                 max_dt: float = 0.1):
        super().__init__(max_speed, per_joint=per_joint, max_dt=max_dt)
        self.blend_speed = float(blend_speed)
        self.max_accel = float(max_accel)
        # None / <=0 → no ceiling (unlimited spring; the yield + the glide +
        # the motor's own T_MAX register are the remaining, sufficient guards).
        self.err_max = (float(err_max)
                        if err_max is not None and float(err_max) > 0
                        else float("inf"))
        self.yield_err = float(yield_err)
        self.yield_after = float(yield_after)
        self.yield_rate = float(yield_rate)
        self.yield_rest = float(yield_rest)
        self.per_joint_err = dict(per_joint_err or {})
        # JOINTS THE THERMAL YIELD MUST NOT LATCH (2026-08-07). The yield assumes
        # the error is an OPERATOR PUSH: back off, and the joint catches up once
        # they let go. That premise is false for a GRAVITY-LOADED joint like the
        # torso, where a large sustained error is the normal condition while
        # raising the trunk and the load never "lets go" — so the latch, whose
        # release test is `meas_err <= 0.5*yield_rest`, can never clear.
        # Observed: torso slumped on its −60° forward stop with Home commanding
        # +15°, the setpoint pinned at measured + 0.120 rad (== yield_rest, the
        # tell) forever, so ⌂ Home could not bring the torso back up at all.
        # These joints keep the SPRING (err_max still bounds kp·err) and lose
        # only the decaying ratchet.
        self.no_yield = set(no_yield or ())
        self._s: Dict[str, float] = {}      # persistent commanded setpoints
        self._v: Dict[str, float] = {}      # setpoint velocities
        self._blend: set = set()            # joints latched into blend speed
        self._strain_s: Dict[str, float] = {}   # accumulated seconds of |err|>yield_err
        self._ycap: Dict[str, float] = {}       # decaying yield ceiling per joint
        self._yielded: set = set()          # joints latched into the gentle hold

    def err_for(self, joint: str) -> float:
        return float(self.per_joint_err.get(joint, self.err_max))

    def reset(self) -> None:
        """Forget the clock AND the setpoints — the next step seeds from
        measured (e-stop release / resume must hold where the arm IS)."""
        super().reset()
        self._s.clear(); self._v.clear(); self._blend.clear()
        self._strain_s.clear(); self._yielded.clear(); self._ycap.clear()

    def tick(self, now: Optional[float] = None) -> float:
        t = time.monotonic() if now is None else now
        if self._last_t is not None and (t - self._last_t) > _STALE_S:
            # long stall (pause, e-stop, mode dwell): the arm may have moved —
            # reseed from measured rather than glide from a stale setpoint.
            self._s.clear(); self._v.clear(); self._blend.clear()
        return super().tick(t)

    def step(self, targets: Mapping[str, float],
             measured: Mapping[str, float],
             dt: Optional[float] = None,
             now: Optional[float] = None) -> Dict[str, float]:
        import math
        t = time.monotonic() if now is None else now
        if dt is None:
            dt = self.max_dt if self._last_t is None else (t - self._last_t)
            self._last_t = t
        dt = max(1e-4, min(float(dt), self.max_dt))

        out: Dict[str, float] = {}
        for name, tgt in targets.items():
            tgt = float(tgt)
            cur = measured.get(name)
            if cur is None:                 # unreadable joint: pass through
                out[name] = tgt
                continue
            cur = float(cur)
            s = self._s.get(name)
            v = self._v.get(name, 0.0)
            if s is None:                   # first sight: hold where the arm IS
                s, v = cur, 0.0
            err = tgt - s
            if abs(err) > _BLEND_ENGAGE:
                self._blend.add(name)
            elif abs(err) < _BLEND_RELEASE:
                self._blend.discard(name)
            vmax = (min(self.speed_for(name), self.blend_speed)
                    if name in self._blend else self.speed_for(name))
            # decelerate-into-target profile, accel-limited
            if err:
                stop_v = math.sqrt(2.0 * self.max_accel * abs(err))
                v_des = math.copysign(min(vmax, stop_v, abs(err) / dt), err)
            else:
                v_des = 0.0
            dv = self.max_accel * dt
            v = min(max(v_des, v - dv), v + dv)
            s_prev = s
            s = s + v * dt
            if (err > 0 and s > tgt) or (err < 0 and s < tgt):
                s = tgt                     # never overshoot the target
            # THE SPRING (2026-07-30). Within `err_max` the setpoint is NOT
            # dragged toward the arm: a push meets a true spring-damper whose
            # resistance GROWS with displacement all the way to kp·err_max, and
            # release springs the arm back (the old 0.15 drag-clamp saturated
            # at ~3.75 N·m then yielded plastically — "torque drop after a
            # certain region"). err_max is the instantaneous HARD ceiling.
            e_hard = self.err_for(name)
            meas_err = abs(s - cur)
            # THE THERMAL YIELD: kp·error is firmware-side and unclampable, so
            # error parked near the ceiling means sustained near-rating torque.
            # If it stays above `yield_err` for `yield_after` s, give way at
            # `yield_rate` down to `yield_rest` and LATCH the gentle hold until
            # the arm catches back up. Quick pushes never see this.
            if name in self.no_yield:
                pass                        # gravity-loaded: never latch (see __init__)
            elif meas_err > self.yield_err:
                # accumulate in DT-seconds (the dynamics' own time base — a
                # wall-clock timer here desynchronizes from simulated/replayed dt)
                self._strain_s[name] = self._strain_s.get(name, 0.0) + dt
                if self._strain_s[name] >= self.yield_after:
                    self._yielded.add(name)
            else:
                self._strain_s.pop(name, None)
            if name in self._yielded:
                if meas_err <= 0.5 * self.yield_rest:
                    self._yielded.discard(name)     # arm caught up: spring restored
                    self._ycap.pop(name, None)
                    self._strain_s.pop(name, None)
                else:
                    # DECAYING RATCHET: the ceiling starts where the error was
                    # when the yield engaged and walks down at yield_rate to
                    # yield_rest. (Recomputing it from the pre-clamp error each
                    # tick let the glide's pull toward the target reset it — the
                    # setpoint never actually gave way.)
                    prev = self._ycap.get(name, min(self.err_for(name), meas_err))
                    cap = max(self.yield_rest, prev - self.yield_rate * dt)
                    self._ycap[name] = cap
                    e_hard = min(e_hard, cap)
            lo, hi = cur - e_hard, cur + e_hard
            s = min(max(s, lo), hi)
            # book-keep the REALIZED velocity: zeroing v at a clamp made the
            # next ticks re-ramp from 0 (a sawtooth while the bound binds); the
            # realized slope keeps the accel limit meaningful through clamps.
            v = (s - s_prev) / dt
            self._s[name] = s
            self._v[name] = v
            out[name] = s
        return out


# ── the TORQUE-REFERENCED error ceiling (2026-08-01) ─────────────────────────
#
# WHY ONE err_max IS THE WRONG SHAPE. The ceiling exists to bound the firmware's
# position term, which is kp*err — but `err_max` is a fleet-wide constant while
# `kp` spans 8..32 N.m/rad along the arm and the motors span 10..54 N.m. At the
# deployed err_max = 0.8 rad the PERMITTED position torque, as a multiple of each
# motor's CONTINUOUS rating (the DaMiao datasheet convention, peak/3 — and the
# quantity its over-current latch actually integrates), is:
#
#   j1 DM8009 kp 32 -> 25.6 N.m = 1.4x cont     j5 DM4310 kp  8 ->  6.4 = 1.9x
#   j2 DM8009 kp 28 -> 22.4 N.m = 1.2x cont     j6 DM4310 kp 12 ->  9.6 = 2.9x
#   j3 DM4340 kp 20 -> 16.0 N.m = 1.8x cont     j7 DM4310 kp 10 ->  8.0 = 2.4x
#   j4 DM4340 kp 28 -> 22.4 N.m = 2.5x cont
#
# Every joint is allowed to sit ABOVE its continuous rating, and the allowance
# differs by 2.4x across one arm — so which joint trips first is an accident of
# where in the arm the operator's mismatch lands, which is exactly the reported
# "sometimes the motor overloads". Deriving the ceiling from a TORQUE budget
# instead gives every joint the same headroom by construction and keeps the
# spring feel wherever the motor can afford it.
#
# The budget is a fraction of the CONTINUOUS rating: at 1.0 a joint parked at the
# ceiling sits exactly at continuous — warm, not latching — and the glide's
# thermal yield still walks it down after `yield_after` seconds.
DEFAULT_TORQUE_BUDGET = 1.0

#: Peak rated torque [N.m] per DaMiao model (firmware manifest:
#: firmware/openarm_dual/config/robot.json).
MOTOR_PEAK_NM = {"DM8009": 54.0, "DM4340": 27.0, "DM4310": 10.0}
#: Continuous / peak. DaMiao datasheet convention.
CONTINUOUS_FRAC = 1.0 / 3.0
#: Which motor sits on arm joint j1..j7 (same on both sides).
ARM_MOTOR_BY_INDEX = ("DM8009", "DM8009", "DM4340", "DM4340",
                      "DM4310", "DM4310", "DM4310")


def torque_error_ceiling(kp, motor="DM4310", budget=DEFAULT_TORQUE_BUDGET,
                         *, lo=0.08, hi=DEFAULT_ERR_MAX):
    """Commanded-error ceiling [rad] that keeps kp*err within `budget` x the
    motor's CONTINUOUS rating.

    Clamped into [lo, hi]: `lo` keeps a very stiff joint from becoming a
    zero-compliance wall the operator cannot push at all, and `hi` never
    RAISES a ceiling above the fleet default — this may only tighten.
    """
    kp = float(kp)
    if kp <= 0.0:
        return float(hi)
    cont = MOTOR_PEAK_NM.get(motor, 10.0) * CONTINUOUS_FRAC
    return float(min(max(budget * cont / kp, lo), hi))


def arm_error_ceilings(gains, budget=DEFAULT_TORQUE_BUDGET, *,
                       name_fn=None):
    """{joint name: err_max} for the arm, from {fw_id: (kp, kd)} or
    {fw_id: {"kp": ...}}.

    `name_fn(fw_id)` renames the keys into whatever frame the limiter is keyed
    in — the deployed limiter is keyed by MJCF ACTUATOR name ("Left Arm 2"),
    the registry by firmware id ("l_j2"). Defaults to identity.
    """
    out = {}
    for fw, g in (gains or {}).items():
        kp = g.get("kp") if isinstance(g, dict) else (g[0] if g else None)
        if kp is None:
            continue
        try:
            idx = int(str(fw).rsplit("j", 1)[1]) - 1
        except (ValueError, IndexError):
            continue
        if not 0 <= idx < len(ARM_MOTOR_BY_INDEX):
            continue
        key = name_fn(fw) if name_fn else fw
        if key:
            out[key] = torque_error_ceiling(kp, ARM_MOTOR_BY_INDEX[idx], budget)
    return out


def arm_velocity_ff_ceilings(gains, budget=DEFAULT_TORQUE_BUDGET, *,
                             name_fn=None, lo=1.0, hi=8.0):
    """{joint: v_ff_max [rad/s]} keeping the MIT law's `kd*vel` term within
    `budget` x the motor's CONTINUOUS rating.

    THE SECOND HALF OF THE TORQUE BOUND, and the one that was missing. The host
    supplies a target VELOCITY on /arms/command so the firmware's kd term can
    track a moving target — and `kd * vel` is a torque exactly as `kp * err` is,
    but it was clamped to a single fleet-wide +-8 rad/s (`ros_plant.py`
    `_arm_target_velocity`). At the deployed kd that permits, as a multiple of
    each motor's continuous rating:

        j1 kd 4.0 -> 32 N.m = 1.8x     j5 kd 1.0 ->  8.0 = 2.4x
        j2 kd 5.0 -> 40 N.m = 2.2x     j6 kd 1.4 -> 11.2 = 3.4x
        j3 kd 2.9 -> 23 N.m = 2.6x     j7 kd 1.3 -> 10.4 = 3.1x
        j4 kd 3.7 -> 33 N.m = 3.7x

    i.e. the velocity feed-forward alone could demand up to 3.7x continuous, and
    it is reached by ordinary means: the command's own rate limit is 4 rad/s, but
    the glide limiter's error clamp is a POSITION clamp with no rate bound, so a
    joint moving fast drags the setpoint along faster than that and whoever
    differentiates the published command sees it. Bounding the FF (rather than
    the clamp) fixes the harm at its source: the position command tracking the
    joint is benign — it REDUCES kp*err — and only its derivative was dangerous.

    The resulting ceilings (2.4-4.5 rad/s) sit above the gate's own 4 rad/s
    tracking cap for the joints that can afford it and below it for the ones
    that cannot, so real teleop tracking is untouched.
    """
    out = {}
    for fw, g in (gains or {}).items():
        kd = g.get("kd") if isinstance(g, dict) else (g[1] if len(g) > 1 else None)
        if not kd:
            continue
        try:
            idx = int(str(fw).rsplit("j", 1)[1]) - 1
        except (ValueError, IndexError):
            continue
        if not 0 <= idx < len(ARM_MOTOR_BY_INDEX):
            continue
        cont = MOTOR_PEAK_NM[ARM_MOTOR_BY_INDEX[idx]] * CONTINUOUS_FRAC
        key = name_fn(fw) if name_fn else fw
        if key:
            out[key] = float(min(max(budget * cont / float(kd), lo), hi))
    return out


__all__ = ["JointRateLimiter", "JointGlideLimiter", "DEFAULT_MAX_SPEED",
           "DEFAULT_BLEND_SPEED", "DEFAULT_MAX_ACCEL", "DEFAULT_ERR_MAX",
           "DEFAULT_TORQUE_BUDGET", "MOTOR_PEAK_NM", "CONTINUOUS_FRAC",
           "ARM_MOTOR_BY_INDEX", "torque_error_ceiling", "arm_error_ceilings",
           "arm_velocity_ff_ceilings"]
