"""
data_generation.py
===================

Turn excitation trajectories into a *measured* dataset.

On the real robot the dataset is recorded from encoders + joint-torque sensors
(or motor-current * torque-constant).  In simulation we synthesise it from the
**ground-truth** model with MuJoCo inverse dynamics and then add Gaussian noise
to emulate sensing imperfections:

    tau_meas = InverseDynamics_truth(q, qdot, qddot) + noise

The returned :class:`Dataset` bundles the states and the noisy torques.  The
states ``(q, qdot, qddot)`` are assumed measured/known; only the *parameters*
are unknown to the identifier.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .arm_model import ArmModel
from . import trajectory as traj


@dataclasses.dataclass
class Dataset:
    Q: np.ndarray      # (N, nv)
    QD: np.ndarray     # (N, nv)
    QDD: np.ndarray    # (N, nv)
    TAU: np.ndarray    # (N, nv) measured (noisy) torque
    TAU_CLEAN: np.ndarray  # (N, nv) noise-free torque (for diagnostics)

    def __len__(self):
        return self.Q.shape[0]

    def split(self, frac_train: float, rng: np.random.Generator):
        """Shuffle and split into (train, test) datasets."""
        N = len(self)
        idx = rng.permutation(N)
        ntr = int(round(frac_train * N))
        tr, te = idx[:ntr], idx[ntr:]
        return self._take(tr), self._take(te)

    def _take(self, idx):
        return Dataset(self.Q[idx], self.QD[idx], self.QDD[idx],
                       self.TAU[idx], self.TAU_CLEAN[idx])


def make_dataset(truth: ArmModel, *, n_static: int = 200,
                 dynamic_duration: float = 30.0, dynamic_dt: float = 0.02,
                 torque_noise_std: float = 0.05,
                 rng: np.random.Generator | None = None) -> Dataset:
    """Build a combined static + dynamic dataset from the ground-truth model.

    Parameters
    ----------
    n_static : number of random standstill poses (gravity-only samples).
    dynamic_duration / dynamic_dt : exciting trajectory length / sampling.
    torque_noise_std : std-dev (N*m) of additive Gaussian torque noise.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # --- static poses (qdot = qddot = 0) ---
    Qs = traj.static_poses(truth, n_static, rng)
    QDs = np.zeros_like(Qs)
    QDDs = np.zeros_like(Qs)

    # --- dynamic excitation ---
    _, Qd, QDd, QDDd = traj.fourier_trajectory(
        truth, duration=dynamic_duration, dt=dynamic_dt, rng=rng)

    Q = np.vstack([Qs, Qd])
    QD = np.vstack([QDs, QDd])
    QDD = np.vstack([QDDs, QDDd])

    TAU_CLEAN = truth.batch_torque(Q, QD, QDD)
    noise = rng.normal(0.0, torque_noise_std, size=TAU_CLEAN.shape)
    TAU = TAU_CLEAN + noise
    return Dataset(Q, QD, QDD, TAU, TAU_CLEAN)


def make_static_dataset(truth: ArmModel, *, n_static: int = 300,
                        torque_noise_std: float = 0.05,
                        rng: np.random.Generator | None = None) -> Dataset:
    """Gravity-only dataset (all standstill).  Enough for gravity compensation."""
    if rng is None:
        rng = np.random.default_rng(0)
    Q = traj.static_poses(truth, n_static, rng)
    QD = np.zeros_like(Q)
    QDD = np.zeros_like(Q)
    TAU_CLEAN = truth.batch_torque(Q, QD, QDD)
    TAU = TAU_CLEAN + rng.normal(0.0, torque_noise_std, size=TAU_CLEAN.shape)
    return Dataset(Q, QD, QDD, TAU, TAU_CLEAN)
