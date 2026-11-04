"""
arm_model.py
============

A thin convenience wrapper around the standalone MuJoCo right-arm model.

It exposes exactly the operations the identification code needs:

* enumerate the *moving* links (the ones whose mass / COM we identify),
* read and write each link's mass and centre-of-mass offset,
* sample random joint configurations inside the joint limits,
* evaluate the inverse-dynamics joint torque  tau = M(q) qddot + C(q,qdot) qdot
  + g(q)  for a batch of states (this is MuJoCo's Lagrangian model), and
* build the analytic gravity regressor (see :mod:`mabel_sysid.regressor`).

MuJoCo stores a link's inertial data as:
    body_mass[i]       scalar mass m_i
    body_ipos[i]       COM expressed in the body frame   c_i  (3,)
    body_inertia[i]    principal moments of inertia about the COM (3,)
    body_iquat[i]      orientation of the principal axes  (4,)
Changing ``body_mass`` / ``body_ipos`` and calling ``mj_forward`` / ``mj_inverse``
re-evaluates the dynamics with the new values -- this is what makes the
MuJoCo-in-the-loop optimiser possible.
"""
from __future__ import annotations

import dataclasses
from typing import List, Sequence

import mujoco
import numpy as np


@dataclasses.dataclass
class LinkParams:
    """Inertial parameters of a single link that we identify."""

    name: str
    body_id: int
    mass: float
    com: np.ndarray  # (3,) centre of mass in the body frame


class ArmModel:
    """Wrapper around the standalone right-arm MuJoCo model."""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Moving links = every body that carries a joint (skip the welded base
        # ``r_arm_mount`` and the ``world`` body, whose inertia is irrelevant
        # because they do not move).
        self.moving_body_ids: List[int] = []
        for j in range(self.model.njnt):
            bid = int(self.model.jnt_bodyid[j])
            if bid not in self.moving_body_ids:
                self.moving_body_ids.append(bid)
        self.moving_body_ids.sort()
        self.n_links = len(self.moving_body_ids)
        self.nv = self.model.nv

        # joint limits in DOF order
        self.q_lower = self.model.jnt_range[:, 0].copy()
        self.q_upper = self.model.jnt_range[:, 1].copy()

    # ------------------------------------------------------------------ names
    def body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def link_names(self) -> List[str]:
        return [self.body_name(b) for b in self.moving_body_ids]

    # ------------------------------------------------------------- parameters
    def get_link_params(self) -> List[LinkParams]:
        out = []
        for b in self.moving_body_ids:
            out.append(
                LinkParams(
                    name=self.body_name(b),
                    body_id=b,
                    mass=float(self.model.body_mass[b]),
                    com=self.model.body_ipos[b].copy(),
                )
            )
        return out

    def get_param_vector(self) -> np.ndarray:
        """Flatten params as [m_0, cx_0, cy_0, cz_0, m_1, ...] (4 per link)."""
        vec = []
        for b in self.moving_body_ids:
            vec.append(self.model.body_mass[b])
            vec.extend(self.model.body_ipos[b])
        return np.asarray(vec, dtype=float)

    def set_param_vector(self, vec: Sequence[float]) -> None:
        vec = np.asarray(vec, dtype=float)
        for k, b in enumerate(self.moving_body_ids):
            self.model.body_mass[b] = vec[4 * k]
            self.model.body_ipos[b] = vec[4 * k + 1 : 4 * k + 4]

    def get_masses(self) -> np.ndarray:
        return np.array([self.model.body_mass[b] for b in self.moving_body_ids])

    def get_coms(self) -> np.ndarray:
        return np.array([self.model.body_ipos[b].copy() for b in self.moving_body_ids])

    # -------------------------------------------------------------- sampling
    def sample_configs(self, n: int, rng: np.random.Generator,
                       margin: float = 0.05) -> np.ndarray:
        """Sample ``n`` joint configurations uniformly within the limits.

        ``margin`` (rad) keeps samples away from the hard stops.
        """
        lo = self.q_lower + margin
        hi = self.q_upper - margin
        return rng.uniform(lo, hi, size=(n, self.nv))

    # ------------------------------------------------------------- dynamics
    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray,
                         qdd: np.ndarray) -> np.ndarray:
        """Joint torque from MuJoCo inverse dynamics for one state.

        Returns tau = M(q) qdd + C(q, qd) qd + g(q)  (plus armature*qdd), i.e.
        the generalised force the actuators must supply.  This is exactly the
        Euler-Lagrange model evaluated by MuJoCo.
        """
        d = self.data
        d.qpos[:] = q
        d.qvel[:] = qd
        d.qacc[:] = qdd
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Static holding torque g(q) (qvel = qacc = 0)."""
        d = self.data
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(self.model, d)
        return d.qfrc_bias.copy()

    def batch_torque(self, Q: np.ndarray, QD: np.ndarray | None = None,
                     QDD: np.ndarray | None = None) -> np.ndarray:
        """Inverse-dynamics torque for a batch of states ``(N, nv)``."""
        Q = np.atleast_2d(Q)
        N = Q.shape[0]
        if QD is None:
            QD = np.zeros_like(Q)
        if QDD is None:
            QDD = np.zeros_like(Q)
        out = np.empty((N, self.nv))
        for k in range(N):
            out[k] = self.inverse_dynamics(Q[k], QD[k], QDD[k])
        return out
