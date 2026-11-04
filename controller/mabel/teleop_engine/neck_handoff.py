"""Neck-yaw -> base-yaw HANDOFF: the robot's view follows the operator past the neck.

THE REQUIREMENT (operator, 2026-10-23): "I need to have 360 view as I move my head —
the robot head cam needs to project anywhere". The Vision Pro shows the head camera on
a head-locked window, so whatever the robot's camera sees is drawn straight ahead of the
operator's eyes. The neck yaw stops at +-90 deg (MJCF ctrlrange), so past that the
window kept showing the view stuck at the limit (measured by
`controller/experiments/exp_head_gaze_coverage.py`: 0 deg error to +-90, then the error
grows 1:1 to 90 deg at a look behind).

THE LAW. `neck_angles()` already removes the MEASURED chassis turn from the head
attitude, so the neck yaw it returns is "head heading minus chassis heading". When that
exceeds `handoff` (80 deg, 10 deg inside the joint limit so the neck still has room
to track a fast head while the base catches up), the excess becomes a base yaw RATE:

    e  = yaw - clip(yaw, -handoff, +handoff)
    wz = clip(gain * e, -wmax, +wmax)

As the chassis turns, the measured chassis yaw grows, `neck_angles` returns a smaller
yaw, `e` -> 0 and the base stops: a closed loop on MEASURED yaw (encoders/odom are
mirrored into the model on the real robot, `server/plant.py` ROSPlant.read_state), with
the neck itself never leaving its range. The rate is ADDED to the base target at the
point of consumption and then passes through `motion_model.step` (the tip-over-safe
shaping every base twist must take) like the stick.

THE STALL GUARD. The loop closes only if the chassis yaw actually updates. If odometry
is stale (or the base is blocked) the error never shrinks and a pure rate law would spin
the base forever. So the guard integrates what it COMMANDED and compares it to what was
MEASURED since the handoff engaged; once commanded exceeds measured by `stall`, the
handoff latches OFF (logged) until `reset()` — an engage / clutch / recalibrate.
"""
import math


class NeckYawHandoff:
    def __init__(self, handoff, gain, wmax, stall, enabled=True):
        self.handoff = float(handoff)
        self.gain = float(gain)
        self.wmax = float(wmax)
        self.stall = float(stall)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        """Forget the episode: re-arms after a stall latch."""
        self.error = 0.0
        self.stalled = False
        self._active = False
        self._cmd_int = 0.0
        self._moved = 0.0
        self._yaw_prev = None

    def observe(self, neck_yaw):
        """Per retarget frame: the UNCLIPPED neck yaw neck_angles() asked for (rad),
        or None when the neck is not being tracked (frozen / no head)."""
        if neck_yaw is None or not math.isfinite(neck_yaw):
            self.error = 0.0
            return
        h = self.handoff
        self.error = float(neck_yaw - min(max(neck_yaw, -h), h))

    def rate(self, chassis_yaw, dt, live=True):
        """Per control tick: base yaw rate (rad/s, + = CCW, the same sense as the
        neck's + yaw) to add to the base target. `chassis_yaw` = MEASURED chassis yaw
        (rad, any datum); `live` = engaged and streaming."""
        if not (self.enabled and live) or self.stalled or self.error == 0.0:
            self._active = False
            return 0.0
        if not self._active:                     # a new handoff episode
            self._active = True
            self._cmd_int = 0.0
            self._moved = 0.0
            self._yaw_prev = float(chassis_yaw)
        # Unwrapped: a long turn may pass +-180 deg.
        self._moved += _wrap(float(chassis_yaw) - self._yaw_prev)
        self._yaw_prev = float(chassis_yaw)
        wz = max(-self.wmax, min(self.wmax, self.gain * self.error))
        self._cmd_int += wz * float(dt)
        # Only the shortfall in the COMMANDED direction counts as a stall.
        lag = abs(self._cmd_int) - max(0.0, math.copysign(1.0, self._cmd_int) * self._moved)
        if lag > self.stall:
            self.stalled = True
            self._active = False
            return 0.0
        return wz


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
