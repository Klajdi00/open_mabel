"""
mabel.dynamic_models — the robot's dynamics, by control region.

One physics oracle (`MujocoDynamics`, model.py) computes gravity g(q), bias
C·qd + g, the mass matrix M(q), and full inverse dynamics from the canonical
MuJoCo model — in sim AND on hardware (fed measured q, qd). Region selection
is pure joint indexing on top (regions.py), so the physics never diverges
between regions:

    arms        ArmsDynamicModel       arm + hand dynamics; body held
    upper_body  UpperBodyDynamicModel  + torso + cascaded lift
    whole_body  WholeBodyDynamicModel  + mobile base (force-controlled via the
                                       admittance; adds base_wrench())

Each region answers: what feedforward cancels gravity here (gravity_ff /
bias_ff — the gravity-compensation torque), and what external force/torque is
someone applying (external_torques via the generalized-momentum observer,
observers.py; Cartesian wrench + contact localization live in
whole_body_controller/admittance/wrench_estimation.py).
"""
from .model import MujocoDynamics, UpperBodyDynamics, _dof            # noqa: F401
from .observers import MomentumObserver                               # noqa: F401
from .regions import (RegionDynamicModel, REGIONS, region_joints,     # noqa: F401
                      region_includes_lift, region_includes_base,
                      COMPLIANT_ARM_JOINTS, LAZY_JOINTS, LAZY_TENDON_ACTUATORS)
from .arms import ArmsDynamicModel                                    # noqa: F401
from .upper_body import UpperBodyDynamicModel                        # noqa: F401
from .whole_body import WholeBodyDynamicModel                         # noqa: F401

__all__ = ["MujocoDynamics", "UpperBodyDynamics", "MomentumObserver",
           "RegionDynamicModel", "ArmsDynamicModel", "UpperBodyDynamicModel",
           "WholeBodyDynamicModel", "REGIONS", "region_joints",
           "region_includes_lift", "region_includes_base",
           "COMPLIANT_ARM_JOINTS", "LAZY_JOINTS", "LAZY_TENDON_ACTUATORS"]
