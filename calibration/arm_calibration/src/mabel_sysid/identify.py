"""
identify.py
===========

Two identification engines, both grounded in the Lagrangian dynamics model.

Method B -- analytic linear least squares (``identify_gravity_linear``)
    Uses the closed-form gravity regressor ``g(q) = Y_g(q) theta``.  Stacking the
    static poses gives an over-determined linear system ``TAU = Y theta`` solved
    by (ridge-regularised) least squares.  Fast, exact, and it exposes the
    *identifiability* of the problem through the regressor's singular values.
    Solves for the first moments ``phi_i = m_i c_i`` and masses directly.

Method A -- MuJoCo-in-the-loop nonlinear least squares (``identify_nonlinear``)
    Parameterises the unknowns as (mass, COM) per link, writes them into a
    MuJoCo model, and minimises the torque residual
        r(x) = InverseDynamics(q, qdot, qddot ; x) - tau_meas
    with :func:`scipy.optimize.least_squares`.  Because it calls MuJoCo's full
    inverse dynamics it naturally uses velocity/acceleration data, applies box
    bounds (mass > 0) and a prior (CAD) regulariser, and identifies mass and COM
    *separately*.

Both return the recovered (mass, COM) per link.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from typing import Optional

import numpy as np
from scipy import optimize

from .arm_model import ArmModel
from .data_generation import Dataset
from . import regressor as R


# --------------------------------------------------------------------------- #
#  Method B : analytic linear gravity least squares                           #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class LinearResult:
    theta: np.ndarray          # [m, phi] per link
    masses: np.ndarray         # (n_links,)
    coms: np.ndarray           # (n_links, 3)
    cond_number: float         # condition number of the stacked regressor
    singular_values: np.ndarray
    residual_rms: float        # N*m, fit residual on the training torques


def identify_gravity_linear(arm: ArmModel, data: Dataset,
                            ridge: float = 0.0,
                            prior_theta: Optional[np.ndarray] = None
                            ) -> LinearResult:
    """Solve ``TAU = Y_g(q) theta`` over the static samples (qdot=qddot=0).

    ``ridge`` adds Tikhonov regularisation ``ridge * ||theta - prior||^2`` which
    is needed because some parameters (e.g. the proximal link masses) are weakly
    or not observable from gravity alone.
    """
    static = np.all(np.abs(data.QD) < 1e-9, axis=1) & \
             np.all(np.abs(data.QDD) < 1e-9, axis=1)
    Q = data.Q[static]
    TAU = data.TAU[static]
    if Q.shape[0] == 0:
        raise ValueError("linear gravity ID needs static (qdot=qddot=0) samples")

    Y = R.stack_gravity_regressor(arm, Q)          # (Ns*nv, 4*n_links)
    tau = TAU.reshape(-1)                          # (Ns*nv,)

    sv = np.linalg.svd(Y, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    if ridge > 0.0:
        if prior_theta is None:
            prior_theta = np.zeros(Y.shape[1])
        A = np.vstack([Y, np.sqrt(ridge) * np.eye(Y.shape[1])])
        bvec = np.concatenate([tau, np.sqrt(ridge) * prior_theta])
        theta, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    else:
        theta, *_ = np.linalg.lstsq(Y, tau, rcond=None)

    resid = Y @ theta - tau
    masses, coms = R.params_to_mass_com(theta, arm.n_links)
    return LinearResult(theta=theta, masses=masses, coms=coms,
                        cond_number=cond, singular_values=sv,
                        residual_rms=float(np.sqrt(np.mean(resid ** 2))))


# --------------------------------------------------------------------------- #
#  Method A : MuJoCo-in-the-loop nonlinear least squares                      #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class NonlinearResult:
    x: np.ndarray              # flat [m, cx, cy, cz] per link
    masses: np.ndarray
    coms: np.ndarray
    success: bool
    cost: float
    n_eval: int
    residual_rms: float


def _unpack(x: np.ndarray, n_links: int):
    x = x.reshape(n_links, 4)
    return x[:, 0], x[:, 1:]


def identify_nonlinear(arm: ArmModel, data: Dataset, x0: np.ndarray, *,
                       prior_x: Optional[np.ndarray] = None,
                       reg: float = 1e-3,
                       mass_bounds=(0.02, 5.0),
                       com_bound: float = 0.30,
                       max_samples: int = 600,
                       max_nfev: int = 200,
                       rng: Optional[np.random.Generator] = None,
                       progress: bool = True,
                       verbose: int = 0) -> NonlinearResult:
    """Fit (mass, COM) by minimising the inverse-dynamics torque residual.

    ``arm`` is used as a scratch model -- its parameters are overwritten during
    optimisation and left at the solution on return.
    ``reg`` weights a soft prior pulling the solution toward ``prior_x`` (CAD),
    which stabilises directions the data does not constrain.
    ``max_samples`` caps how many measured states the optimiser sees (each
    residual evaluation runs that many MuJoCo inverse-dynamics calls, so this is
    the main speed knob).  ``progress`` prints a live heartbeat so a long solve
    never looks frozen.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_links = arm.n_links

    # optionally subsample to keep the optimiser snappy
    N = len(data)
    if N > max_samples:
        sel = rng.choice(N, max_samples, replace=False)
    else:
        sel = np.arange(N)
    Q, QD, QDD = data.Q[sel], data.QD[sel], data.QDD[sel]
    TAU = data.TAU[sel]
    tau_flat = TAU.reshape(-1)
    n_resid_data = tau_flat.size

    if prior_x is None:
        prior_x = x0.copy()
    reg_scale = np.sqrt(reg) * np.sqrt(max(n_resid_data, 1))

    _prog = {"n": 0, "t0": time.time(), "last": 0.0}

    def residual(x):
        arm.set_param_vector(x)
        pred = arm.batch_torque(Q, QD, QDD).reshape(-1)
        data_res = pred - tau_flat
        # soft prior on the parameter deviation (scaled so it is comparable to
        # the data term regardless of sample count)
        prior_res = reg_scale * (x - prior_x)
        _prog["n"] += 1
        if progress:
            now = time.time()
            if now - _prog["last"] > 0.4:
                rms = float(np.sqrt(np.mean(data_res ** 2)))
                sys.stdout.write(
                    f"\r  optimising... eval {_prog['n']:4d} | "
                    f"torque-residual RMS {rms:.4f} N*m | "
                    f"{now - _prog['t0']:5.1f}s elapsed")
                sys.stdout.flush()
                _prog["last"] = now
        return np.concatenate([data_res, prior_res])

    # bounds: mass within mass_bounds, COM within +/- com_bound box
    lo = np.empty_like(x0)
    hi = np.empty_like(x0)
    for k in range(n_links):
        lo[4 * k] = mass_bounds[0]
        hi[4 * k] = mass_bounds[1]
        lo[4 * k + 1 : 4 * k + 4] = -com_bound
        hi[4 * k + 1 : 4 * k + 4] = com_bound

    if progress:
        print(f"  fitting {x0.size} params to {n_resid_data} torque samples "
              f"(this runs MuJoCo inverse dynamics in a loop)...")
    sol = optimize.least_squares(
        residual, x0, bounds=(lo, hi), method="trf", x_scale="jac",
        xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=max_nfev, verbose=verbose)
    if progress:
        sys.stdout.write("\r" + " " * 78 + "\r")  # clear the heartbeat line
        sys.stdout.flush()

    arm.set_param_vector(sol.x)
    masses, coms = _unpack(sol.x, n_links)
    data_res = residual(sol.x)[:n_resid_data]
    return NonlinearResult(
        x=sol.x, masses=masses, coms=coms, success=bool(sol.success),
        cost=float(sol.cost), n_eval=int(sol.nfev),
        residual_rms=float(np.sqrt(np.mean(data_res ** 2))))


def perturb_params(arm: ArmModel, rng: np.random.Generator, *,
                   mass_frac: float = 0.30, com_shift: float = 0.02
                   ) -> np.ndarray:
    """Create a scrambled initial guess: +/- ``mass_frac`` on each mass and a
    random ``com_shift`` (m) offset on each COM component.  Returns a flat vec.
    """
    x = arm.get_param_vector().copy()
    n_links = arm.n_links
    for k in range(n_links):
        x[4 * k] *= 1.0 + rng.uniform(-mass_frac, mass_frac)
        x[4 * k + 1 : 4 * k + 4] += rng.uniform(-com_shift, com_shift, 3)
    return x
