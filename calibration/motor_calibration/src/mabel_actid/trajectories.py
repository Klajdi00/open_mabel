"""
trajectories.py
===============

Excitation profiles for actuator identification.  Two jobs:

1. **Friction curve** -- a *velocity staircase*: hold a sequence of constant
   velocities (both directions), dwell at each so the joint reaches steady state,
   and average the reported torque over the steady part.  Constant velocity =>
   zero acceleration => the measured torque is friction alone.  This traces out
   ``tau_f(w)`` directly and is the most robust friction measurement there is.

2. **Inertia + dynamic check** -- a *sine* or *chirp* sweep that deliberately
   excites acceleration so the reflected inertia ``J`` becomes observable, and
   so the friction fit can be cross-validated on data it never saw.

All generators return arrays sampled at ``rate`` Hz; the staircase also returns a
boolean ``steady`` mask flagging the samples to use for the constant-velocity fit
(transients at each step edge are excluded).
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class StaircaseProfile:
    t: np.ndarray          # (N,) time, s
    w_cmd: np.ndarray      # (N,) commanded velocity, rad/s
    steady: np.ndarray     # (N,) bool: True on the settled part of each step
    levels: np.ndarray     # (L,) the distinct velocity levels, rad/s


def velocity_staircase(v_max: float, n_levels: int = 12,
                       dwell: float = 1.5, settle: float = 0.6,
                       rate: float = 500.0, both_directions: bool = True,
                       v_min: float = 0.02, log_spaced: bool = True
                       ) -> StaircaseProfile:
    """A symmetric velocity staircase from ``v_min`` up to ``v_max`` (and negative).

    ``v_max``      use a *safe* fraction of the motor's rated speed.
    ``n_levels``   distinct speeds per direction.
    ``dwell``      seconds held at each level.
    ``settle``     seconds at the start of each dwell to discard (transient).
    ``v_min``      smallest level, rad/s (keep > 0 but small to see the
                   Stribeck dip; ~0.01-0.05 rad/s).
    ``log_spaced`` log-space the levels so low speeds -- where break-away /
                   Stribeck lives -- are densely sampled.  Strongly recommended.
    """
    lo = max(v_min, 1e-4)
    if log_spaced:
        pos = np.geomspace(lo, v_max, n_levels)
    else:
        pos = np.linspace(lo, v_max, n_levels)
    if both_directions:
        # interleave +/- so consecutive steps share a position band (helps the
        # later gravity-subtraction trick): +v1,-v1,+v2,-v2,...
        levels = np.empty(2 * n_levels)
        levels[0::2] = pos
        levels[1::2] = -pos
    else:
        levels = pos

    dt = 1.0 / rate
    n_dwell = int(round(dwell * rate))
    n_settle = int(round(settle * rate))

    w_cmd, steady = [], []
    for v in levels:
        w_cmd.append(np.full(n_dwell, v))
        m = np.zeros(n_dwell, dtype=bool)
        m[n_settle:] = True
        steady.append(m)
    w_cmd = np.concatenate(w_cmd)
    steady = np.concatenate(steady)
    t = np.arange(w_cmd.size) * dt
    return StaircaseProfile(t=t, w_cmd=w_cmd, steady=steady, levels=levels)


def sine_profile(amp: float, freq: float, duration: float,
                 rate: float = 500.0):
    """Position sine ``amp*sin(2*pi*f*t)`` with analytic vel/acc.

    Returns ``(t, q, w, a)``.
    """
    dt = 1.0 / rate
    t = np.arange(0.0, duration, dt)
    wfreq = 2 * np.pi * freq
    q = amp * np.sin(wfreq * t)
    w = amp * wfreq * np.cos(wfreq * t)
    a = -amp * wfreq ** 2 * np.sin(wfreq * t)
    return t, q, w, a


def chirp_profile(amp: float, f0: float, f1: float, duration: float,
                  rate: float = 500.0):
    """Linear-frequency position chirp from ``f0`` to ``f1`` Hz.

    Sweeping frequency excites a broad acceleration spectrum, which is what makes
    the reflected inertia ``J`` well-conditioned.  Returns ``(t, q, w, a)``.
    """
    dt = 1.0 / rate
    t = np.arange(0.0, duration, dt)
    k = (f1 - f0) / duration
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    inst_w = 2 * np.pi * (f0 + k * t)            # d(phase)/dt
    inst_a = 2 * np.pi * k                        # d(inst_w)/dt
    q = amp * np.sin(phase)
    w = amp * inst_w * np.cos(phase)
    a = amp * (inst_a * np.cos(phase) - inst_w ** 2 * np.sin(phase))
    return t, q, w, a
