# controller.py - Cascaded motion control.
#
#   position cmd --> [trapezoidal profile] --> velocity setpoint
#                                                   |
#                                     measured vel -+
#                                                   v
#                                            [PI velocity loop] --> effort
#
# A velocity command bypasses the profile and feeds the inner loop directly.
# Both paths share the same well-damped velocity regulator, so position moves
# inherit smooth, speed-limited motion for free.

import math

import config


class VelocityLoop:
    """Inner loop: PI + feedforward regulator from target mm/s to motor effort."""

    def __init__(self):
        self._integral = 0.0
        self.setpoint = 0.0

    def reset(self):
        self._integral = 0.0
        self.setpoint = 0.0

    def compute(self, v_target, v_measured, dt):
        self.setpoint = v_target
        error = v_target - v_measured

        # Conditional integration for anti-windup: accumulate, then clamp the
        # integral *term* (Ki * integral) to a bounded effort contribution.
        self._integral += error * dt
        i_term = config.VEL_KI * self._integral
        if i_term > config.VEL_I_LIMIT:
            i_term = config.VEL_I_LIMIT
            self._integral = i_term / config.VEL_KI
        elif i_term < -config.VEL_I_LIMIT:
            i_term = -config.VEL_I_LIMIT
            self._integral = i_term / config.VEL_KI

        effort = (config.VEL_FF * v_target) + (config.VEL_KP * error) + i_term

        # Static-friction deadband: tiny non-zero commands won't move the lift,
        # so either kick past stiction or command a true stop.
        if abs(v_target) < 1e-3 and abs(error) < 1e-3:
            return 0.0
        if 0.0 < abs(effort) < config.MIN_EFFORT:
            effort = math.copysign(config.MIN_EFFORT, effort)
        return effort


class MotionProfile:
    """Outer loop: trapezoidal velocity profile toward a position target.

    The velocity setpoint is the smaller of the cruise limit and the
    "braking" speed sqrt(2*a*distance) that decelerates exactly onto the
    target, and it is slew-rate limited by MAX_ACCEL on every tick.
    """

    def __init__(self):
        self.target_mm = 0.0
        self.v_cmd = 0.0          # current ramped velocity setpoint
        self.v_max = config.MAX_VELOCITY_MM_S

    def reset(self, current_vel=0.0):
        self.v_cmd = current_vel

    def set_target(self, target_mm, v_max=None, current_vel=0.0):
        self.target_mm = target_mm
        self.v_max = config.MAX_VELOCITY_MM_S if v_max is None else v_max
        self.v_cmd = current_vel

    def retarget(self, target_mm, v_max=None):
        """Adopt a NEW target mid-flight, keeping the ramp.

        For a streamed setpoint (a policy or an animation at 50 Hz): only the
        goal moves. self.v_cmd — the velocity the column is currently ramped to
        — is deliberately left alone, so the profile carries on from the speed
        it is actually doing instead of restarting the trapezoid on every frame.
        compute() re-derives the braking speed from the new distance on the very
        next tick, so the response is immediate without a discontinuity.
        """
        self.target_mm = target_mm
        if v_max is not None:
            self.v_max = v_max

    def at_target(self, position_mm, velocity_mm_s):
        return (abs(self.target_mm - position_mm) <= config.POSITION_TOL_MM and
                abs(velocity_mm_s) <= config.ARRIVE_VEL_MM_S)

    def compute(self, position_mm, dt):
        distance = self.target_mm - position_mm
        direction = 1.0 if distance >= 0 else -1.0

        # Speed that decelerates to a stop right at the target.
        brake_v = math.sqrt(2.0 * config.MAX_ACCEL_MM_S2 * abs(distance))
        v_desired = direction * min(self.v_max, brake_v)

        # Slew-rate (acceleration) limit the setpoint.
        max_step = config.MAX_ACCEL_MM_S2 * dt
        dv = v_desired - self.v_cmd
        if dv > max_step:
            dv = max_step
        elif dv < -max_step:
            dv = -max_step
        self.v_cmd += dv
        return self.v_cmd
