"""
Compliance control — the robot is fully back-drivable and a human can move it.

Gravity-compensating a joint set leaves each joint behaving as pure inertia:
it floats, holds pose against gravity, and can be dragged by hand ("dance with
the robot"). A small viscous damping prevents numerical drift; heavy damping on
the 'lazy' joints (torso, lift) makes the big DOFs move deliberately.

Pick the compliant REGION (arms only / upper body / whole body) either by
passing joint lists (legacy constructor, unchanged) or with
`ComplianceController.for_region(...)`. Whole-body compliance — where the
mobile base also admits the operator's push and glides — is the
compliance x whole_body pairing of the runtime engine
(mabel.whole_body_controller.WBCEngine), which composes this law with the
base admittance and the tip-safe motion model.
"""
from ... import config as C
from ...dynamic_models.model import MujocoDynamics, _dof
from ...dynamic_models.regions import (COMPLIANT_ARM_JOINTS, LAZY_JOINTS,
                                       LAZY_TENDON_ACTUATORS, region_joints,
                                       region_includes_lift)
from ...plant import Plant


class ComplianceController:
    """Gravity-compensates a set of joints so they are back-drivable, with extra
    viscous damping on the 'lazy' ones. Cancelling the full bias (gravity +
    Coriolis) leaves each joint behaving as pure inertia -> it floats and can be
    dragged; the small compliant_damping prevents numerical drift."""

    def __init__(self, dyn: MujocoDynamics, plant: Plant,
                 compliant=COMPLIANT_ARM_JOINTS, lazy=LAZY_JOINTS,
                 compliant_damping=0.6, lazy_damping=12.0, guard=None,
                 manage_lift=True):
        self.dyn, self.plant = dyn, plant
        self.compliant = list(compliant)
        self.lazy = list(lazy)
        self.compliant_damping = compliant_damping
        self.lazy_damping = lazy_damping
        # Legacy default: the standalone compliance stack always frees the lift
        # (manage_lift=True). Region-faithful construction (for_region) keeps
        # the lift on its position servo in the 'arms' region instead.
        self.manage_lift = bool(manage_lift)
        # Optional self-collision "soft wall": a SelfCollisionGuard whose
        # barrier_force is added to the managed-joint efforts so a dragged arm
        # stops at the safety margin. None (default) -> no avoidance (plain comp).
        self.guard = guard
        # Neutralize the stiff servos on every managed joint (and the lift tendon).
        managed = self.compliant + self.lazy
        if self.manage_lift:
            managed += list(LAZY_TENDON_ACTUATORS)
        for a in managed:
            try:
                self.plant.disable_servo(a)
            except KeyError:
                pass

    @classmethod
    def for_region(cls, model, data, plant, region="arms", **kw):
        """Region-selectable construction, matching the runtime engine's
        semantics: 'arms' -> only the arms are compliant (torso + lift hold on
        their servos); 'upper_body'/'whole_body' -> the torso joins as a lazy
        (heavily damped) compliant joint and the lift is freed +
        gravity-compensated. For the base-gliding whole-body behaviour use
        WBCEngine(compliance, whole_body) instead — the base needs the
        admittance + tip-safe motion model, not joint torques."""
        joints = region_joints(region)
        compliant = [j for j in joints if j in COMPLIANT_ARM_JOINTS]
        lazy = [j for j in joints if j not in COMPLIANT_ARM_JOINTS]
        kw.setdefault("manage_lift", region_includes_lift(region))
        return cls(MujocoDynamics(model, data), plant,
                   compliant=compliant, lazy=lazy, **kw)

    def update(self):
        """Compute and apply the compensation efforts for one tick."""
        b = self.dyn.bias()
        m = self.dyn.model
        tau_sc = self.guard.barrier_force(self.dyn.data.qvel) if self.guard is not None else None
        self.plant.begin()
        for j in self.compliant:
            tau = b[_dof(m, j)] - self.compliant_damping * self.plant.qd(j)
            if tau_sc is not None:
                tau += tau_sc[_dof(m, j)]
            self.plant.apply_effort(j, tau)
        for j in self.lazy:
            tau = b[_dof(m, j)] - self.lazy_damping * self.plant.qd(j)
            if tau_sc is not None:
                tau += tau_sc[_dof(m, j)]
            self.plant.apply_effort(j, tau)
        # Lift (cascaded tendon): gravity-comp its joint DOF + heavy damping so the
        # raised body neither sags nor moves quickly when dragged. Skipped when
        # the region keeps the lift on its position servo (for_region('arms')).
        if self.manage_lift:
            lj = C.LIFT_JOINT
            tau = b[_dof(m, lj)] - self.lazy_damping * self.plant.qd(lj)
            self.plant.apply_effort(lj, tau)
        self.plant.commit()
