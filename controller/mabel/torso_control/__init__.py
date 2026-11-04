"""mabel.torso_control — the torso's control law (gravity feed-forward + frame map).

Import-light on purpose (pure math, no MuJoCo): the setup studio imports this on its
serial control path. Constants are MJCF-derived and baked; re-derive with
``controller/experiments/torso_gravity_derive.py`` after any MJCF mass/geometry change.
"""
from .gravity import (GRAV_A, GRAV_PHI, TORSO_RANGE, TORSO_URDF_MID, ff_trim,
                      hold_tau, model_to_urdf, posture_tau, urdf_to_model)

__all__ = ["TORSO_RANGE", "TORSO_URDF_MID", "GRAV_A", "GRAV_PHI",
           "hold_tau", "posture_tau", "ff_trim", "model_to_urdf", "urdf_to_model"]
