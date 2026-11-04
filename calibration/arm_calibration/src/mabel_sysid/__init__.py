"""
mabel_sysid
===========

System identification of the MABEL right-arm inertial parameters (link mass and
centre of mass) for inverse dynamics and gravity compensation.

The package is organised around four ideas:

* :mod:`mabel_sysid.arm_model`  -- a thin wrapper over a MuJoCo model that lets
  us read/write each link's mass and COM and evaluate the (Lagrangian) inverse
  dynamics torque.
* :mod:`mabel_sysid.trajectory` -- generators for static calibration poses and
  smooth exciting trajectories.
* :mod:`mabel_sysid.data_generation` -- turns trajectories into a measured
  dataset ``(q, qdot, qddot, tau)`` with optional sensor noise.
* :mod:`mabel_sysid.regressor` / :mod:`mabel_sysid.identify` -- the two
  identification engines (analytic linear least-squares and MuJoCo-in-the-loop
  nonlinear optimisation) plus :mod:`mabel_sysid.metrics` for reporting.
"""

from .arm_model import ArmModel, LinkParams  # noqa: F401

__all__ = ["ArmModel", "LinkParams"]
