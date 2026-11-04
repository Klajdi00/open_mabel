"""
trajectory.py
=============

Excitation trajectories for system identification.

Two regimes are provided:

* :func:`static_poses` -- a cloud of random standstill configurations.  At a
  standstill the only torque is gravity ``g(q)``, which depends only on the
  *first moments* ``m_i c_i``.  These poses are all you need for **gravity
  compensation**.

* :func:`fourier_trajectory` -- a smooth, periodic "finite Fourier series"
  trajectory (the standard excitation in robot identification).  Because it has
  non-zero velocity and acceleration, the measured torque now contains inertial
  terms ``M(q) qddot`` which depend on the link *masses* themselves -- this is
  what makes mass (not just first moment) observable.

A Fourier-series joint trajectory is

    q_j(t) = q0_j + sum_{l=1..L} [ a_jl/(w l) sin(w l t) - b_jl/(w l) cos(w l t) ]

with base frequency ``w = 2 pi f``.  Velocity and acceleration are available in
closed form, which avoids numerical-differentiation noise in the "ground-truth"
states.
"""
from __future__ import annotations

import numpy as np


def static_poses(arm, n: int, rng: np.random.Generator,
                 margin: float = 0.05) -> np.ndarray:
    """Random standstill configurations, shape ``(n, nv)``."""
    return arm.sample_configs(n, rng, margin=margin)


def fourier_trajectory(arm, *, duration: float = 20.0, dt: float = 0.01,
                       n_harmonics: int = 5, base_freq: float = 0.35,
                       rng: np.random.Generator | None = None):
    """Generate a smooth exciting trajectory.

    Returns ``(t, Q, QD, QDD)`` where each ``Q*`` has shape ``(T, nv)``.  Joint
    angles are clipped to stay inside the limits (with a small margin).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    nv = arm.nv
    w = 2.0 * np.pi * base_freq
    t = np.arange(0.0, duration, dt)
    T = t.size

    lo = arm.q_lower + 0.10
    hi = arm.q_upper - 0.10
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)

    Q = np.zeros((T, nv))
    QD = np.zeros((T, nv))
    QDD = np.zeros((T, nv))

    for j in range(nv):
        a = rng.uniform(-1, 1, n_harmonics)
        b = rng.uniform(-1, 1, n_harmonics)
        q = np.zeros(T)
        qd = np.zeros(T)
        qdd = np.zeros(T)
        for l in range(1, n_harmonics + 1):
            wl = w * l
            q += a[l - 1] / wl * np.sin(wl * t) - b[l - 1] / wl * np.cos(wl * t)
            qd += a[l - 1] * np.cos(wl * t) + b[l - 1] * np.sin(wl * t)
            qdd += -a[l - 1] * wl * np.sin(wl * t) + b[l - 1] * wl * np.cos(wl * t)
        # normalise the *position* envelope into the joint range, scale the
        # derivatives by the same factor so they stay kinematically consistent.
        amp = np.max(np.abs(q)) + 1e-9
        scale = half[j] / amp
        Q[:, j] = mid[j] + scale * q
        QD[:, j] = scale * qd
        QDD[:, j] = scale * qdd

    # safety clip (envelope scaling already keeps us in range, but guard stops)
    Q = np.clip(Q, arm.q_lower + 1e-3, arm.q_upper - 1e-3)
    return t, Q, QD, QDD
