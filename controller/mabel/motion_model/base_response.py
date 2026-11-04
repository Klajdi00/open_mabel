"""Calibrated forward motion model for the MABEL 3-module swerve base.

System-identified in sim (controller/experiments/motion_id.py: the full robot on a
bare floor, driven by this package's own swerve IK, mabel.swerve.swerve_ik). The
result is strikingly simple and clean -- within the operating envelope the base
reproduces a commanded body twist (vx, vy, omega) almost exactly:

    actual_twist  ~=  SLIP_GAIN * clamp(commanded_twist, envelope)

with three measured properties (all verified across a full per-axis sweep):
  * SLIP_GAIN = 0.989  -- a single uniform ~1.1% slip, identical for vx, vy, omega.
  * FULLY DECOUPLED     -- combined commands track each axis independently; the
                           cross-terms (e.g. omega leaking into vx) are < 0.5%.
  * LINEAR, NO DEADBAND -- exact down to 0.05 m/s; NO saturation inside the envelope
                           (omega is perfect to +-3 rad/s).
Beyond the envelope the swerve control destabilises (reverse vx >= 1.0 m/s and
vy >= 1.5 m/s lock up or spin out), so the model clamps to the reliable region --
which is exactly the config speed caps (MAX_LIN_SPEED, MAX_ANG_SPEED).

This SUPERSEDES the implicit "wheel_speed * r = ground speed" no-slip assumption: it
is the same kinematics plus the measured slip gain and an explicit validity bound.

Localization use:
  * PROCESS MODEL -- integrate the command you SENT through this model to dead-reckon.
    In free space it matches ground truth to ~1%, and unlike wheel encoders it does
    NOT depend on wheel-ground contact, so it is immune to wheel SPIN/slip.
  * SLIP / STUCK detection -- if the wheel encoders or the camera disagree with this
    prediction, the base is slipping (encoders >> model) or blocked (vision << model).
The big localization failures we benchmarked (~3x wheel over-read) were the robot
getting STUCK in clutter (wheels spinning while pinned), NOT a kinematics error --
this model makes that detectable.
"""
import math

# --- measured constants (controller/experiments/motion_id.py, mabel_bare.xml) -----
SLIP_GAIN = 0.989          # commanded -> actual body-twist gain (1.1% slip), all axes
V_MAX = 0.8                # m/s   reliable linear envelope (= config.MAX_LIN_SPEED)
W_MAX = 2.0                # rad/s reliable angular envelope (= config.MAX_ANG_SPEED)
# When the command CHANGES, the swerve re-steers with drive choked, so the body
# velocity ramps to the new target with a first-order time constant (measured step
# response: 0 -> 63% in ~0.13 s, settled by ~0.3 s). Ignoring this makes a forward
# model OVER-shoot at every command change; the stateful BaseMotionModel includes it.
TAU = 0.13                 # s   velocity-ramp time constant (alignment transient)


def predict_twist(vx, vy, omega):
    """Commanded body twist -> predicted ACTUAL body twist.

    Decoupled + linear: scale by the slip gain after clamping to the reliable
    envelope (translation magnitude and yaw rate). Returns (vx, vy, omega)."""
    v = math.hypot(vx, vy)
    if v > V_MAX:
        f = V_MAX / v
        vx *= f
        vy *= f
    if omega > W_MAX:
        omega = W_MAX
    elif omega < -W_MAX:
        omega = -W_MAX
    return SLIP_GAIN * vx, SLIP_GAIN * vy, SLIP_GAIN * omega


def integrate(x, y, theta, vx, vy, omega, dt):
    """STEADY-STATE forward step (no transient): advance an odom-frame pose by the
    calibrated actual twist with exact midpoint integration. Good when the command
    is held; use BaseMotionModel for command sequences (it adds the ramp)."""
    avx, avy, aw = predict_twist(vx, vy, omega)
    thm = theta + 0.5 * aw * dt
    x += (avx * math.cos(thm) - avy * math.sin(thm)) * dt
    y += (avx * math.sin(thm) + avy * math.cos(thm)) * dt
    theta = math.atan2(math.sin(theta + aw * dt), math.cos(theta + aw * dt))
    return x, y, theta


class BaseMotionModel:
    """Stateful forward model: feed it the control inputs (vx, vy, omega) you send to
    the base and dt; it returns how far the robot actually travels, including the
    first-order velocity ramp at every command change (the swerve alignment
    transient). This is the accurate "how far would the robot go" predictor and a
    drop-in localization PROCESS model.

        m = BaseMotionModel()
        for (vx, vy, w), dt in commands:
            x, y, theta = m.step(vx, vy, w, dt)   # integrated pose in the odom frame
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.vx = self.vy = self.w = 0.0          # current (ramped) actual twist

    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.vx = self.vy = self.w = 0.0

    def step(self, vx, vy, omega, dt):
        """Advance one tick under the commanded twist; return (x, y, theta)."""
        tvx, tvy, tw = predict_twist(vx, vy, omega)        # steady-state target
        a = 1.0 - math.exp(-dt / TAU)                      # first-order ramp toward it
        self.vx += a * (tvx - self.vx)
        self.vy += a * (tvy - self.vy)
        self.w += a * (tw - self.w)
        thm = self.theta + 0.5 * self.w * dt
        self.x += (self.vx * math.cos(thm) - self.vy * math.sin(thm)) * dt
        self.y += (self.vx * math.sin(thm) + self.vy * math.cos(thm)) * dt
        self.theta = math.atan2(math.sin(self.theta + self.w * dt),
                                math.cos(self.theta + self.w * dt))
        return self.x, self.y, self.theta
