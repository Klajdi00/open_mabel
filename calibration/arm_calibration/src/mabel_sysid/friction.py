"""
friction.py
===========

**Separating gravity from friction with bidirectional averaging.**

At a static hold the motor reports

    tau_meas(q, d) = tau_gravity(q)  +  tau_coulomb * d  +  noise

where ``d = +/-1`` is the direction the joint was travelling when it stopped.
The friction term does not vanish just because the joint is stationary -- it
keeps whatever sign opposed the last motion.  On the DM-J4340 that offset is
0.43 N*m and on the DM-J4310 it is 0.083 N*m; the wrist joints' entire gravity
sinusoid is 0.02-0.12 N*m.  Identify from one direction only and you are fitting
a sinusoid that has been shifted bodily by more than its own amplitude.

Visit every angle **twice** -- once arriving from below (d = +1), once from
above (d = -1) -- and the two measurements separate cleanly:

    gravity  :  tau_g(q)   = 0.5 * ( tau_+(q) + tau_-(q) )      <- friction cancels
    friction :  tau_c      = 0.5 * ( tau_+(q) - tau_-(q) )      <- gravity cancels

This is the oldest trick in the identification literature (it is why Khalil &
Dombre recommend running every trajectory in both directions) and it is the
single most important thing this package does for the real robot.  It costs one
extra pass -- nothing compared to the bias it removes.

The difference channel is a bonus: it *measures the Coulomb friction of every
joint in situ*, on the assembled arm, including harness drag and bearing
preload that a bench test of the bare motor cannot see.  For joints 1-2
(DM-J8009P) this is currently the ONLY friction number we have.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class BidirectionalSplit:
    angles: np.ndarray        # (K,) the angles visited in both directions
    Q: np.ndarray             # (K, nv) configuration at each angle
    tau_gravity: np.ndarray   # (K, nv) friction-free gravity torque
    tau_coulomb: np.ndarray   # (K, nv) half-difference = Coulomb friction
    coulomb_mean: np.ndarray  # (nv,)   mean Coulomb over the sweep
    coulomb_std: np.ndarray   # (nv,)   spread (a check: it should be ~constant)
    n_pairs: int


def split_bidirectional(sweep: dict, *, atol: float = 1e-3) -> BidirectionalSplit:
    """Pair up the +1 and -1 visits to each angle and split gravity / friction.

    ``sweep`` is the dict returned by :func:`mabel_sysid.plant.run_sweep` (or by
    the hardware collector -- they emit the same schema): arrays ``angle``,
    ``direction``, ``q``, ``tau``.
    """
    ang = np.asarray(sweep["angle"], float)
    dirn = np.asarray(sweep["direction"], int)
    Qa = np.asarray(sweep["q"], float)
    TAU = np.asarray(sweep["tau"], float)

    up = np.where(dirn > 0)[0]
    dn = np.where(dirn < 0)[0]
    if up.size == 0 or dn.size == 0:
        raise ValueError(
            "bidirectional split needs holds in BOTH directions -- run the sweep "
            "with bidirectional=True, or the Coulomb friction will bias the fit "
            "by up to its full magnitude (0.43 N*m on a DM-J4340).")

    angles, Qs, taug, tauc = [], [], [], []
    for i in up:
        # nearest down-visit to the same angle
        k = int(np.argmin(np.abs(ang[dn] - ang[i])))
        jdx = dn[k]
        if abs(ang[jdx] - ang[i]) > max(atol, 1e-6):
            continue
        angles.append(0.5 * (ang[i] + ang[jdx]))
        Qs.append(0.5 * (Qa[i] + Qa[jdx]))
        taug.append(0.5 * (TAU[i] + TAU[jdx]))
        tauc.append(0.5 * (TAU[i] - TAU[jdx]))

    if not angles:
        raise ValueError("no +/- pairs matched; check the sweep's angle grid")

    angles = np.asarray(angles)
    Qs = np.asarray(Qs)
    taug = np.asarray(taug)
    tauc = np.asarray(tauc)
    return BidirectionalSplit(
        angles=angles, Q=Qs, tau_gravity=taug, tau_coulomb=tauc,
        coulomb_mean=tauc.mean(axis=0), coulomb_std=tauc.std(axis=0),
        n_pairs=len(angles))


# --------------------------------------------------------------------------- #
#  Per-joint sinusoid fit (the 2 parameters a joint can reveal)                #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class SinusoidFit:
    joint: int
    a: float                  # tau = a cos(q) + b sin(q)
    b: float
    amplitude: float          # sqrt(a^2+b^2) -- the gravity swing
    phase: float              # atan2(b, a)
    r2: float
    residual_rms: float       # N*m
    sigma_a: float            # 1-sigma confidence
    sigma_b: float
    n: int
    snr: float                # amplitude / residual_rms


def fit_sinusoid(angles: np.ndarray, tau_j: np.ndarray, joint: int) -> SinusoidFit:
    """Least-squares fit of ``tau_j = a cos(q) + b sin(q)`` -- the exact model.

    No constant term: the theory says the offset is identically zero (verified to
    1e-14 N*m against MuJoCo).  If your data wants a constant, something is
    wrong -- residual friction, a torque-sensor bias, or a joint that moved when
    it should not have.  :func:`fit_sinusoid_with_offset` reports that offset so
    you can catch it.
    """
    angles = np.asarray(angles, float)
    tau_j = np.asarray(tau_j, float)
    A = np.column_stack([np.cos(angles), np.sin(angles)])
    coef, *_ = np.linalg.lstsq(A, tau_j, rcond=None)
    pred = A @ coef
    resid = tau_j - pred
    n = angles.size
    dof = max(n - 2, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    ss_tot = float(np.sum((tau_j - tau_j.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / max(ss_tot, 1e-18)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    amp = float(np.hypot(*coef))
    return SinusoidFit(
        joint=joint, a=float(coef[0]), b=float(coef[1]), amplitude=amp,
        phase=float(np.arctan2(coef[1], coef[0])), r2=r2, residual_rms=rms,
        sigma_a=float(np.sqrt(max(cov[0, 0], 0))),
        sigma_b=float(np.sqrt(max(cov[1, 1], 0))),
        n=n, snr=amp / max(rms, 1e-12))


def fit_sinusoid_with_offset(angles: np.ndarray, tau_j: np.ndarray) -> float:
    """Return the fitted constant term.  Should be ~0; a large value is a red flag."""
    A = np.column_stack([np.cos(angles), np.sin(angles), np.ones_like(angles)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(tau_j, float), rcond=None)
    return float(coef[2])
