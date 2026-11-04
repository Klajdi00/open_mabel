"""
Control REGIONS — which joints each dynamic model / torque law owns.

MABEL's torque-domain control is parameterised by a region, growing outward
from the arms:

    arms        the 16 arm joints (7 arm + 1 ORCA wrist per side). The hands'
                masses are part of the MuJoCo model, so arm gravity/bias terms
                already carry the hands; the ORCA finger joints themselves stay
                position-controlled (hand control is decoupled by design).
    upper_body  + Torso hinge + the cascaded Lift (tendon actuator; its qpos
                is read on the LIFT_JOINT = "Lift Lower" cascade joint).
    whole_body  same joint set as upper_body + the mobile base: the base is
                not a torque-controlled joint — it is driven by the admittance
                law (whole_body_controller/admittance) through the tip-safe
                motion model and swerve IK.

`RegionDynamicModel` binds a region's joint set to the full-model
`MujocoDynamics` oracle and answers the two questions every torque controller
asks: "what feedforward holds this region against gravity?" and "what external
torque is someone applying to it?" (via the momentum observer).
"""
import numpy as np

from .model import MujocoDynamics, _dof
from .observers import MomentumObserver

# Joints made back-drivable ("float" under gravity comp) and the lazy ones.
COMPLIANT_ARM_JOINTS = (
    [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
    + [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"]
)
LAZY_JOINTS = ["Torso"]            # heavy viscous damping -> moves slowly
LAZY_TENDON_ACTUATORS = ["Lift"]   # lift is tendon-driven; handled separately

REGIONS = ("arms", "upper_body", "whole_body")
# Legacy spelling used by the wire protocol / apps ("arm" == "arms").
_REGION_ALIASES = {"arm": "arms"}


def region_joints(region):
    """Torque-managed joint names for a region (excludes the tendon Lift
    actuator, which is neutralised/driven separately — see the engine)."""
    region = _REGION_ALIASES.get(region, region)
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")
    joints = list(COMPLIANT_ARM_JOINTS)
    if region in ("upper_body", "whole_body"):
        joints += list(LAZY_JOINTS)
    return joints


def region_includes_lift(region):
    region = _REGION_ALIASES.get(region, region)
    return region in ("upper_body", "whole_body")


def region_includes_base(region):
    region = _REGION_ALIASES.get(region, region)
    return region == "whole_body"


class RegionDynamicModel:
    """A control region bound to the MuJoCo dynamics oracle.

    Answers, for the region's joints only:
      * gravity_ff()   {joint: tau} gravity feedforward g(q)
      * bias_ff()      {joint: tau} gravity + Coriolis feedforward b(q, qd)
      * mass_matrix()  the region block of M(q)
      * external_torques(tau_cmd, dt)   momentum-observer residual on the
                       region's DOFs — the external (operator) joint torque
    The full-model MujocoDynamics is shared: region selection is pure indexing,
    so the physics never diverges between regions.
    """

    region = None      # set by subclasses / constructor

    def __init__(self, model, data, region=None, observer_gain=25.0):
        self.model, self.data = model, data
        self.region = _REGION_ALIASES.get(region or self.region, region or self.region)
        self.dyn = MujocoDynamics(model, data)
        self.joints = region_joints(self.region)
        self.dofs = [_dof(model, j) for j in self.joints]
        self.includes_lift = region_includes_lift(self.region)
        self.includes_base = region_includes_base(self.region)
        if self.includes_lift:
            from .. import config as C
            self.lift_joint = C.LIFT_JOINT
        else:
            self.lift_joint = None
        self._observer = None
        self._observer_gain = float(observer_gain)

    # feedforward ----------------------------------------------------------------
    def gravity_ff(self):
        """Gravity feedforward per managed joint: {joint: tau}."""
        g = self.dyn.gravity()
        out = {j: float(g[d]) for j, d in zip(self.joints, self.dofs)}
        if self.lift_joint is not None:
            out[self.lift_joint] = float(g[_dof(self.model, self.lift_joint)])
        return out

    def bias_ff(self):
        """Gravity + Coriolis/centrifugal feedforward per managed joint."""
        b = self.dyn.bias()
        out = {j: float(b[d]) for j, d in zip(self.joints, self.dofs)}
        if self.lift_joint is not None:
            out[self.lift_joint] = float(b[_dof(self.model, self.lift_joint)])
        return out

    def mass_matrix(self):
        """Region block of the joint-space mass matrix M(q)."""
        M = self.dyn.mass_matrix()
        idx = np.asarray(self.dofs)
        return M[np.ix_(idx, idx)]

    # external-torque estimation ---------------------------------------------------
    @property
    def observer(self):
        """Lazily-built full-model momentum observer (shared state across calls)."""
        if self._observer is None:
            self._observer = MomentumObserver(self.model, self.data,
                                              gain=self._observer_gain)
        return self._observer

    def external_torques(self, tau_cmd, dt):
        """Momentum-observer external-torque estimate on the region's joints:
        {joint: tau_ext}. `tau_cmd` is the FULL nv commanded torque vector
        (gravity comp + actuators), exactly what the plant applied this tick."""
        r = self.observer.update(tau_cmd, dt)
        out = {j: float(r[d]) for j, d in zip(self.joints, self.dofs)}
        if self.lift_joint is not None:
            out[self.lift_joint] = float(r[_dof(self.model, self.lift_joint)])
        return out
