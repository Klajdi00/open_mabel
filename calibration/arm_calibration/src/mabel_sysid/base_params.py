"""
base_params.py
==============

**What gravity can and cannot tell you.**

The static (gravity) torque is linear in the per-link inertial parameters

    tau_g(q) = Y_g(q) . theta,     theta_i = [ m_i, phi_i ] = [ m_i, m_i c_i ]

(4 numbers per link, see :mod:`mabel_sysid.regressor`).  But ``Y_g`` is
**structurally rank-deficient**: for this 7-DOF arm it has 28 columns and rank
15, of which only **14 are numerically well conditioned** (the 15th singular
value sits ~3 decades below the 14th).  Fourteen = **2 per joint**, and that is
not a coincidence:

    A revolute joint j only feels the component of the gravity wrench
    perpendicular to its own axis z_j.  Everything distal to joint j enters
    solely through the *subtree first moment* s_j = sum_{i in subtree(j)} m_i
    (p_ci - p_j), and only the two components of s_j perpendicular to z_j
    produce torque about z_j.  Hence 2 identifiable numbers per joint.

Consequences you must accept (they are laws, not limitations of the code):

* **Per-link mass is NOT identifiable from gravity.**  Moving mass from link i's
  frame origin into the parent's first moment leaves every static torque
  unchanged.  Any code that prints "identified mass = 0.83 kg" from static data
  is reporting the *prior*, or noise, dressed up as a measurement.
* **The COM component along a joint's own axis is not identifiable** either.
* What IS identifiable -- exactly, and with excellent conditioning -- is the
  14-dimensional set of subtree-first-moment combinations, **and that is
  precisely and completely what gravity compensation needs**.

This module therefore does three things:

1. :func:`identifiable_subspace` -- numerically extract the identifiable
   parameter subspace (SVD of the stacked regressor over a dense pose cloud),
   with the rank chosen against a noise-aware tolerance.
2. :func:`fit_theta` -- fit the parameters by a **minimum-norm update of a CAD
   prior**: the data determines the identifiable directions, and the CAD /
   official-URDF prior supplies the null space it cannot see.  This is the only
   honest way to end up with a full (m, c) set to paste into an MJCF.
3. :func:`base_param_report` -- per-direction standard deviations (Gautier's
   relative-sigma criterion) so you can see which combinations the data actually
   pinned down.

References
----------
Gautier & Khalil, "Direct calculation of minimum set of inertial parameters of
serial robots", IEEE T-RA 6(3), 1990.
Khalil & Dombre, *Modeling, Identification and Control of Robots*, ch. 12.
Gautier, "Numerical calculation of the base inertial parameters of robots",
J. Robotic Systems 8(4), 1991.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .arm_model import ArmModel
from . import regressor as R


# --------------------------------------------------------------------------- #
#  Identifiable subspace                                                       #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Subspace:
    """The identifiable / unidentifiable split of the parameter space."""

    V_id: np.ndarray        # (n_params, r)   orthonormal basis of identifiable dirs
    V_null: np.ndarray      # (n_params, n-r) orthonormal basis of the blind dirs
    sing: np.ndarray        # (n_params,)     singular values of the pose-cloud regressor
    rank: int               # r
    n_params: int
    tol: float              # the threshold that produced `rank`

    @property
    def n_blind(self) -> int:
        return self.n_params - self.rank


def identifiable_subspace(arm: ArmModel, *, n_poses: int = 600,
                          rank_tol: float = 1e-3,
                          rng: Optional[np.random.Generator] = None) -> Subspace:
    """Split the parameter space into what gravity torques can and cannot see.

    ``rank_tol`` is a *relative* singular-value threshold (sigma_k / sigma_max).
    The default 1e-3 deliberately discards the near-singular 15th direction of
    this arm: keeping it would amplify torque noise by ~1/7e-4 ~ 1400x while
    contributing almost nothing to the predicted torque.  Set it smaller only if
    your torque measurements are extraordinarily clean.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Q = arm.sample_configs(n_poses, rng)
    Y = R.stack_gravity_regressor(arm, Q)

    _, s, Vt = np.linalg.svd(Y, full_matrices=True)
    n_params = Y.shape[1]
    # pad the spectrum if the SVD returned fewer values than columns
    sing = np.zeros(n_params)
    sing[: s.size] = s

    thresh = rank_tol * sing[0]
    rank = int(np.sum(sing > thresh))
    return Subspace(V_id=Vt[:rank].T, V_null=Vt[rank:].T, sing=sing,
                    rank=rank, n_params=n_params, tol=float(thresh))


# --------------------------------------------------------------------------- #
#  Fit: minimum-norm update of a prior                                         #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GravityFit:
    theta: np.ndarray            # (4*n_links,) full parameter vector
    masses: np.ndarray           # (n_links,)
    coms: np.ndarray             # (n_links, 3)
    theta_prior: np.ndarray
    delta_id: np.ndarray         # the correction applied, in the identifiable subspace
    rank: int
    cond: float                  # condition number of the regressor RESTRICTED to V_id
    residual_rms: float          # N*m, on the training torques
    sigma_beta: np.ndarray       # (rank,) std-dev of each identified direction
    sigma_beta_rel: np.ndarray   # (rank,) relative std-dev [%] -- Gautier's criterion
    noise_std: float             # estimated torque noise std from the residual


def fit_theta(arm: ArmModel, Q: np.ndarray, TAU: np.ndarray,
              theta_prior: np.ndarray, sub: Subspace, *,
              weights: Optional[np.ndarray] = None) -> GravityFit:
    """Fit the gravity parameters from static poses, anchored on a prior.

    We solve, over the **identifiable subspace only**,

        min_d  || W ( Y (theta_prior + V_id d) - tau_meas ) ||^2

    and return ``theta = theta_prior + V_id d``.  The null-space content of
    ``theta`` is therefore *exactly* the prior's -- the data never had an opinion
    about it, so we do not pretend it did.  This is a minimum-norm (in the
    subspace metric) correction of CAD, and it is what keeps the resulting
    per-link masses physically sane while making the *predicted gravity torque*
    optimal.

    Parameters
    ----------
    Q     : (N, nv) static configurations (must be standstill measurements)
    TAU   : (N, nv) measured gravity torques (Coulomb friction already removed,
            e.g. by bidirectional averaging -- see :mod:`mabel_sysid.friction`)
    theta_prior : (4*n_links,) CAD / official-URDF parameters
    weights : (N, nv) optional per-measurement weights (1/sigma_j).  Use these
            when joints have very different torque scales: the shoulder swings
            10 N*m while the wrist swings 0.05 N*m, and an unweighted fit would
            let the shoulder's noise drown the wrist entirely.
    """
    Y = R.stack_gravity_regressor(arm, Q)          # (N*nv, n_params)
    tau = np.asarray(TAU).reshape(-1)
    resid_prior = tau - Y @ theta_prior            # what the prior fails to explain

    A = Y @ sub.V_id                               # (N*nv, r)
    if weights is not None:
        w = np.asarray(weights).reshape(-1)
        Aw = A * w[:, None]
        bw = resid_prior * w
    else:
        w = None
        Aw, bw = A, resid_prior

    sv = np.linalg.svd(Aw, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    d, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    theta = theta_prior + sub.V_id @ d

    resid = Y @ theta - tau
    n, r = A.shape
    dof = max(n - r, 1)
    sigma2 = float(resid @ resid) / dof            # noise variance estimate
    # parameter covariance of the identified directions:  sigma^2 (A^T A)^-1
    AtA_inv = np.linalg.pinv(Aw.T @ Aw)
    var_beta = sigma2 * np.clip(np.diag(AtA_inv), 0.0, None)
    sigma_beta = np.sqrt(var_beta)
    # beta = coordinates of theta in the identifiable basis
    beta = sub.V_id.T @ theta
    sigma_rel = 100.0 * sigma_beta / np.maximum(np.abs(beta), 1e-12)

    masses, coms = R.params_to_mass_com(theta, arm.n_links)
    return GravityFit(
        theta=theta, masses=masses, coms=coms, theta_prior=theta_prior,
        delta_id=d, rank=sub.rank, cond=cond,
        residual_rms=float(np.sqrt(np.mean(resid ** 2))),
        sigma_beta=sigma_beta, sigma_beta_rel=sigma_rel,
        noise_std=float(np.sqrt(sigma2)))


# --------------------------------------------------------------------------- #
#  Physical-consistency projection                                             #
# --------------------------------------------------------------------------- #
def enforce_physical(arm: ArmModel, theta: np.ndarray, theta_prior: np.ndarray,
                     sub: Subspace, *,
                     mass_lo: float = 0.02, mass_hi: float = 5.0,
                     com_box: float = 0.25) -> np.ndarray:
    """Pull an identified ``theta`` back into the physically feasible set.

    A minimum-norm update of a physically sane prior is almost always already
    feasible, but a noisy fit can push a small link's mass negative or throw its
    COM outside the part.  We re-solve the fit's *null-space* freedom (which the
    data does not constrain anyway) to restore feasibility, leaving the
    identifiable directions -- and hence the predicted gravity torque -- intact.

    Concretely: keep ``V_id^T theta`` fixed, and choose the null-space component
    ``V_null^T theta`` to satisfy the box constraints as closely as possible,
    starting from the prior's.  Solved as a small bounded least-squares problem.
    """
    from scipy import optimize

    beta_fixed = sub.V_id.T @ theta          # must be preserved exactly
    n_links = arm.n_links

    lo = np.empty(sub.n_params)
    hi = np.empty(sub.n_params)
    for k in range(n_links):
        lo[4 * k] = mass_lo
        hi[4 * k] = mass_hi
        # first moment phi = m*c, so |phi| <= mass_hi * com_box
        lo[4 * k + 1: 4 * k + 4] = -mass_hi * com_box
        hi[4 * k + 1: 4 * k + 4] = mass_hi * com_box

    if np.all(theta >= lo) and np.all(theta <= hi):
        return theta                          # already feasible -- nothing to do

    # variables: the null-space coordinates n; theta(n) = V_id beta_fixed + V_null n
    n0 = sub.V_null.T @ theta_prior
    base = sub.V_id @ beta_fixed

    def resid(nvec):
        th = base + sub.V_null @ nvec
        viol = np.concatenate([np.clip(lo - th, 0, None), np.clip(th - hi, 0, None)])
        # 1e3 weight: feasibility dominates, prior-closeness breaks the tie
        return np.concatenate([1e3 * viol, 0.1 * (nvec - n0)])

    sol = optimize.least_squares(resid, n0, method="lm", xtol=1e-12, ftol=1e-12)
    return base + sub.V_null @ sol.x


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def base_param_report(fit: GravityFit, *, sigma_warn: float = 20.0) -> str:
    """Human-readable table of the identified directions and their confidence.

    Gautier's rule of thumb: a base parameter whose relative standard deviation
    exceeds ~20% was not really identified -- the data is consistent with a wide
    range of values, and you should either excite it better or leave it at CAD.
    """
    lines = [
        f"identified {fit.rank} base directions "
        f"(cond = {fit.cond:.1f}, residual RMS = {fit.residual_rms:.4f} N*m, "
        f"est. noise = {fit.noise_std:.4f} N*m)",
        "",
        f"  {'dir':>4} {'sigma':>12} {'rel. sigma':>12}   status",
    ]
    n_bad = 0
    for i in range(fit.rank):
        rel = fit.sigma_beta_rel[i]
        bad = rel > sigma_warn
        n_bad += bad
        lines.append(f"  {i+1:>4} {fit.sigma_beta[i]:>12.5f} {rel:>11.1f}%   "
                     f"{'POORLY EXCITED' if bad else 'ok'}")
    if n_bad:
        lines.append("")
        lines.append(f"  !! {n_bad} direction(s) above the {sigma_warn:.0f}% "
                     f"relative-sigma threshold -- add poses that excite them, or "
                     f"they stay at their CAD value.")
    return "\n".join(lines)
