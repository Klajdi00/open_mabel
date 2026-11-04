"""LAYER 1 base dynamics: tip-over motion model + lift-aware reverse guard."""
import numpy as np

from .hyperparams import *
from .deps import MC


class BaseMotionModel:
    """LAYER 1 of 3. Turns a *desired* body velocity (vx, vy, omega) into a
    physically realizable one by ramping the CURRENT velocity toward it through the
    base's dynamic envelope — whose max speed / acceleration shrink ~1/CoM-height,
    so a raised lift cannot brake or turn hard enough to tip the robot.

    Default backend = MABEL's tip-over / ZMP model (mabel.config.safe_base_step,
    measured from the chassis support polygon). To swap the dynamics (different
    chassis, a learned model, extra constraints) subclass and override step() — the
    WholeBodyController treats this purely through the step() interface."""

    def __init__(self, backend=MC):
        self.backend = backend
        self.available = backend is not None

    def step(self, vel, tgt, dt, z_lift):
        """vel, tgt = (vx, vy, omega) body frame; returns the new tip-safe velocity."""
        if self.backend is None:
            return tuple(float(t) for t in tgt)               # no model -> pass through
        return self.backend.safe_base_step(*vel, *tgt, dt, float(z_lift))

    def max_speed(self, z_lift):
        """(fwd, back, lat) tip-safe speed caps at this lift height, for telemetry."""
        if self.backend is None:
            return None
        return self.backend.max_speed_dir(float(z_lift))


class LiftAwareReverseGuard:
    """A control-barrier-style base_filter (the WholeBodyController hook). MABEL's
    triangular base has a SINGLE REAR wheel and a short front lever, so reversing
    (+vx, toward the front edge) while the lift is RAISED throws the CoM over that
    short lever and tips it — a mode the rigid point-mass tip model under-protects
    because the tall, compliant lift+arm chain whips forward under servo lag. This
    filter linearly scales BACKWARD velocity to zero as the lift rises (forward /
    lateral / turn are physically safe and pass through untouched). Swap or remove
    it by changing WholeBodyController(base_filter=...) — it is fully decoupled."""

    def __init__(self, lift_act, lift_max=LIFT_RANGE[1]):
        self.lift_act = lift_act
        self.lift_max = max(lift_max, 1e-6)

    def __call__(self, data, vel):
        vx, vy, omega = vel
        if vx > 0.0:                                       # +vx = reverse (front-edge tip)
            z = float(data.ctrl[self.lift_act])
            frac = float(np.clip(1.0 - z / self.lift_max, 0.0, 1.0))   # 1 low -> 0 raised
            vx *= frac
        return (vx, vy, omega)
