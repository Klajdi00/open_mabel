"""
regressor.py
============

Analytic **gravity regressor** for the arm.

Key fact (the heart of the Lagrangian method): the static gravity torque is a
*linear* function of each link's inertial parameters

    theta_i = [ m_i,  phi_i ] = [ m_i,  m_i * c_i ]            (4 numbers/link)

where ``m_i`` is the link mass and ``phi_i = m_i c_i`` is the *first moment of
mass* (mass times centre of mass).  Stacking the parameters of all moving links
into one vector ``theta`` we can write

    g(q) = Y_g(q) @ theta                                    (per configuration)

``Y_g(q)`` is the gravity regressor below.  Derivation:

    potential energy   P(q) = - sum_i  m_i * g_vec . p_ci(q)
    gravity torque     g(q) = dP/dq = - sum_i  J_ci(q)^T (m_i g_vec)

with the COM world position  p_ci = p_i + R_i c_i  (link-frame origin + rotated
local COM).  Splitting the Jacobian gives, for joint row j,

    g_j = - sum_i [ m_i * (g_vec . dp_i/dq_j)
                    + (g_vec x omega_j) . (R_i phi_i) ]

so the column multiplying ``m_i``     is  -(J_pi^T g_vec)            and the
columns multiplying ``phi_i`` (3) are  -(g_vec x omega_j)^T R_i.

Here ``J_pi`` is the translational Jacobian of the link-frame origin and
``omega_j`` is the world-frame axis of ancestor joint j (both obtained directly
from MuJoCo, so no kinematics are re-derived by hand).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .arm_model import ArmModel


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def gravity_regressor_single(arm: ArmModel, q: np.ndarray) -> np.ndarray:
    """Gravity regressor ``Y_g(q)`` of shape ``(nv, 4 * n_links)``.

    Column block layout per link: ``[m_i, phi_ix, phi_iy, phi_iz]``.
    """
    m = arm.model
    d = arm.data
    d.qpos[:] = q
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_forward(m, d)

    g = m.opt.gravity.copy()  # (3,)
    nv = arm.nv
    Y = np.zeros((nv, 4 * arm.n_links))

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    for k, b in enumerate(arm.moving_body_ids):
        # Jacobian of the body-FRAME ORIGIN (not the COM) and rotational Jac.
        mujoco.mj_jacBody(m, d, jacp, jacr, b)
        R = d.xmat[b].reshape(3, 3)  # body orientation, world <- body

        # mass column:  -(J_pi^T g)
        Y[:, 4 * k] = -(jacp.T @ g)

        # first-moment columns:  row j = -(g x omega_j)^T R
        # vectorised over joints:  -(skew(g) @ jacr)^T @ R
        gx_omega = _skew(g) @ jacr          # (3, nv), columns g x omega_j
        Y[:, 4 * k + 1 : 4 * k + 4] = -(gx_omega.T @ R)
    return Y


def true_param_vector(arm: ArmModel) -> np.ndarray:
    """Ground-truth ``theta`` = [m_i, m_i*cx, m_i*cy, m_i*cz] per link."""
    vec = []
    for b in arm.moving_body_ids:
        m_i = float(arm.model.body_mass[b])
        c_i = arm.model.body_ipos[b]
        vec.extend([m_i, m_i * c_i[0], m_i * c_i[1], m_i * c_i[2]])
    return np.asarray(vec)


def stack_gravity_regressor(arm: ArmModel, Q: np.ndarray):
    """Stack the per-config gravity regressors and torques.

    Returns ``(Y_stacked, )`` with shape ``(N*nv, 4*n_links)``.
    """
    rows = [gravity_regressor_single(arm, q) for q in Q]
    return np.vstack(rows)


def params_to_mass_com(theta: np.ndarray, n_links: int):
    """Convert a ``theta`` vector of [m, phi] blocks to ``(masses, coms)``.

    ``com_i = phi_i / m_i``.  Guards against division by ~0 mass.
    """
    theta = np.asarray(theta).reshape(n_links, 4)
    masses = theta[:, 0]
    phis = theta[:, 1:]
    coms = np.zeros_like(phis)
    safe = np.abs(masses) > 1e-9
    coms[safe] = phis[safe] / masses[safe, None]
    return masses, coms
