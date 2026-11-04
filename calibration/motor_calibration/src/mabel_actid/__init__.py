"""
mabel_actid
===========

Actuator-level system identification for the MABEL arm's Damiao FOC servos.

Where ``mabel_sysid`` identifies the *link* inertial parameters (mass, COM) of
the rigid-body model, this package identifies what lives **inside each joint**:
the friction and damping of the brushless-DC pancake motor + planetary gearbox,
plus the reflected rotor inertia.  These are exactly the terms a rigid-body
inverse-dynamics / gravity-compensation model leaves out, and the usual reason a
feed-forward torque that is "correct" on paper still fails to move (or overshoots)
the real joint.

The identified numbers map one-to-one onto MuJoCo joint attributes:

    Coulomb (dry) friction   F_c   <->   joint ``frictionloss``   (N*m)
    viscous damping          F_v   <->   joint ``damping``        (N*m*s/rad)
    reflected rotor inertia  J     <->   joint ``armature``       (kg*m^2)

so calibrating an actuator and pasting three numbers into ``mabel_full.xml``
closes the dominant sim-to-real gap for that joint.

Public surface
--------------
``MOTORS``                  database of Damiao motor profiles (limits, gear ratio)
``DamiaoBus`` / ``Actuator``host-side python-can driver (MIT mode)
``FrictionParams``          the friction + inertia model
``identify_*``              the fitting routines
``SyntheticActuator``       a ground-truth simulator to validate the pipeline
"""
from __future__ import annotations

from .motors import MOTORS, MotorProfile
from .friction_model import FrictionParams, friction_torque, dynamic_torque
from .trajectories import velocity_staircase, sine_profile, chirp_profile
from .data_logging import CalibrationLog
from .simulate import SyntheticActuator
from . import identify, metrics

__all__ = [
    "MOTORS",
    "MotorProfile",
    "FrictionParams",
    "friction_torque",
    "dynamic_torque",
    "velocity_staircase",
    "sine_profile",
    "chirp_profile",
    "CalibrationLog",
    "SyntheticActuator",
    "identify",
    "metrics",
]
