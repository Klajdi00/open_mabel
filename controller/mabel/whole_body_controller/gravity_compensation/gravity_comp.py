"""
Gravity-compensation feedforward controller (region-selectable).

Applies exactly the model torque that cancels gravity (optionally the full
bias, gravity + Coriolis) on a region's joints, plus a small viscous damping
for numerical stability:

    tau_j = g_j(q)  (or b_j(q, qd))  -  D_j qd_j

With no other input the region floats: it holds pose against gravity and is
back-drivable by hand. This is the K=0 special case of the impedance law and
the building block of every torque mode — the compliance controller IS this
law plus target-tracking bookkeeping, and the Day-1 PD becomes gravity-aware
by taking `gravity_ff` as its feedforward hook.

Region selection comes from the dynamic models (mabel.dynamic_models):
'arms' | 'upper_body' | 'whole_body' — the lift (cascaded tendon) is handled
through its cascade joint automatically when the region includes it.
"""
from ...dynamic_models.model import _dof
from ...dynamic_models.regions import RegionDynamicModel

__all__ = ["GravityCompController"]


class GravityCompController:
    """Pure gravity(/bias)-feedforward + damping on a region's joints."""

    def __init__(self, region_model: RegionDynamicModel, plant,
                 damping=0.6, lazy_damping=12.0, use_bias=True,
                 neutralize_servos=True, guard=None):
        self.region = region_model
        self.plant = plant
        self.damping = float(damping)            # arm joints
        self.lazy_damping = float(lazy_damping)  # torso / lift (move deliberately)
        self.use_bias = bool(use_bias)
        # Optional self-collision "soft wall" (SelfCollisionGuard.barrier_force).
        self.guard = guard
        if neutralize_servos:
            names = list(self.region.joints)
            if self.region.includes_lift:
                names.append("Lift")             # the tendon actuator
            for a in names:
                try:
                    self.plant.disable_servo(a)
                except KeyError:
                    pass

    def _damping_for(self, joint):
        from ...dynamic_models.regions import COMPLIANT_ARM_JOINTS
        return self.damping if joint in COMPLIANT_ARM_JOINTS else self.lazy_damping

    def update(self):
        """One tick: apply the compensation efforts. Returns {joint: tau}."""
        ff = self.region.bias_ff() if self.use_bias else self.region.gravity_ff()
        m = self.region.model
        tau_sc = (self.guard.barrier_force(self.region.data.qvel)
                  if self.guard is not None else None)
        applied = {}
        self.plant.begin()
        for j, tau_g in ff.items():
            tau = tau_g - self._damping_for(j) * self.plant.qd(j)
            if tau_sc is not None:
                tau += tau_sc[_dof(m, j)]
            self.plant.apply_effort(j, tau)
            applied[j] = tau
        self.plant.commit()
        return applied
