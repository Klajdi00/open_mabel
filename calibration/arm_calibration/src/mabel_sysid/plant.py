"""
plant.py
========

**A simulated stand-in for the real arm -- with all the things that ruin a
system identification.**

The original sim test in this package generated its "measurements" with MuJoCo
inverse dynamics on the *same* model it then identified.  That test cannot fail:
the only error source is the Gaussian noise you add yourself, so the identifier
is graded on a problem with no model mismatch, no friction, no quantisation and
no gain error.  It always passes, and it tells you nothing about the robot.

This module builds the adversarial version.  ``MeasuredPlant`` wraps a
*ground-truth* MuJoCo model whose parameters differ from the model the
identifier is given, and corrupts every torque the way the real DaMiao stack
does:

* **Coulomb + viscous friction** at the joint, ``tau_f = tau_c sign(qdot) + b qdot``.
  This is the big one.  At a static hold the joint is not moving, so ``sign(qdot)``
  is set by *how the joint arrived* -- friction acts as a sign-dependent offset of
  up to +/-0.43 N*m (DM-J4340) or +/-0.083 N*m (DM-J4310).  Sweep a joint at a
  badly chosen posture and that offset exceeds its entire gravity signal: you
  would be measuring friction and calling it mass.  (The excitation module picks
  postures that keep the signal 14-23x above the friction, but the friction is
  still a first-order term and must be cancelled, not ignored.)
* **Torque quantisation**: the MIT protocol packs torque into 12 bits over
  +/-t_max, so the finest torque the motor can report is 2*t_max/4095 -- 4.9 mN*m
  on a DM-J4310 (t_max=10), 13 mN*m on a DM-J4340, 20 mN*m on a DM-J8009P.
* **Torque-constant / gear-efficiency error**: reported torque is
  ``kt * current * N * eta``.  A few percent error in kt, and an efficiency that
  differs between driving and back-driving, both scale the measurement.
* **Encoder zero offset**: a mis-set joint zero rotates the whole sinusoid.
* Additive Gaussian sensor noise.

If the identification survives *this* plant, it has a chance on the real robot.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .arm_model import ArmModel
from .excitation import JOINT_COULOMB, JOINT_TAU_CONT


# t_max of the MIT encoding per joint (the 12-bit torque field spans +/- t_max).
# DM-J8009P: 40, DM-J4340: 27, DM-J4310: 10.  Source: mabel_actid/motors.py.
JOINT_T_MAX = np.array([40.0, 40.0, 27.0, 27.0, 10.0, 10.0, 10.0])
TORQUE_LSB = 2.0 * JOINT_T_MAX / 4095.0     # N*m per bit -> [0.0195, .., 0.0049]


@dataclasses.dataclass
class PlantErrors:
    """Every way the real robot lies to you.  Defaults reflect the measured stack."""

    # friction (N*m).  None -> use the measured JOINT_COULOMB values.
    coulomb: Optional[np.ndarray] = None
    viscous: Optional[np.ndarray] = None        # N*m/(rad/s)
    # multiplicative torque gain error per joint (1.0 = perfect kt & gearbox)
    kt_scale: Optional[np.ndarray] = None
    # extra gain applied only when back-driving (gearbox efficiency asymmetry)
    eta_asym: float = 0.0                       # 0 = symmetric; 0.05 = 5% asymmetry
    # encoder zero offset (rad) per joint
    q_offset: Optional[np.ndarray] = None
    # additive Gaussian torque noise (N*m)
    noise_std: float = 0.01
    # quantise the reported torque to the MIT 12-bit grid
    quantise: bool = True

    def resolved(self, nv: int, rng: np.random.Generator) -> "PlantErrors":
        """Fill in the Nones with the measured / typical values."""
        return PlantErrors(
            coulomb=JOINT_COULOMB[:nv] if self.coulomb is None else np.asarray(self.coulomb),
            viscous=(np.array([0.30, 0.30, 0.2115, 0.2115, 0.0198, 0.0198, 0.0198])[:nv]
                     if self.viscous is None else np.asarray(self.viscous)),
            kt_scale=np.ones(nv) if self.kt_scale is None else np.asarray(self.kt_scale),
            eta_asym=self.eta_asym,
            q_offset=np.zeros(nv) if self.q_offset is None else np.asarray(self.q_offset),
            noise_std=self.noise_std, quantise=self.quantise)


class MeasuredPlant:
    """Ground-truth arm + the measurement chain's imperfections.

    ``truth`` is a MuJoCo model carrying the REAL (unknown to the identifier)
    mass/COM.  Every call returns what the motor's CAN feedback would report.
    """

    def __init__(self, truth: ArmModel, errors: PlantErrors,
                 rng: Optional[np.random.Generator] = None):
        self.truth = truth
        self.rng = rng or np.random.default_rng(0)
        self.err = errors.resolved(truth.nv, self.rng)

    # ------------------------------------------------------------------ torque
    def measure(self, Q: np.ndarray, QD: np.ndarray, QDD: np.ndarray,
                direction: np.ndarray) -> np.ndarray:
        """Reported joint torque for a batch of states.

        ``direction`` is (N, nv) in {-1, 0, +1}: the sign of the joint's motion,
        which sets the sign of the Coulomb term.  At a *static hold* the joint
        has stopped, but the friction force does not vanish -- it holds whatever
        sign opposed the last motion.  This is exactly the effect the
        bidirectional protocol exploits, so we model it explicitly rather than
        pretending a stationary joint is frictionless.
        """
        Q = np.atleast_2d(Q); QD = np.atleast_2d(QD); QDD = np.atleast_2d(QDD)
        direction = np.atleast_2d(direction)
        e = self.err

        # the arm's own dynamics, at the TRUE parameters
        tau = self.truth.batch_torque(Q, QD, QDD)

        # friction opposes motion: for a moving joint sign(qdot); for a joint at
        # rest, the sign of the motion that brought it here.
        sgn = np.where(np.abs(QD) > 1e-6, np.sign(QD), direction)
        tau = tau + e.coulomb * sgn + e.viscous * QD

        # gearbox efficiency asymmetry: back-driving (tau and qdot opposite) is
        # lossier than driving.
        if e.eta_asym:
            backdriving = (tau * QD) < 0
            tau = tau * np.where(backdriving, 1.0 + e.eta_asym, 1.0)

        # torque-constant error
        tau = tau * e.kt_scale

        # sensor noise, then the 12-bit MIT quantiser
        tau = tau + self.rng.normal(0.0, e.noise_std, size=tau.shape)
        if e.quantise:
            lsb = TORQUE_LSB[: self.truth.nv]
            tau = np.round(tau / lsb) * lsb
        return tau

    # ---------------------------------------------------------------- position
    def report_q(self, Q: np.ndarray) -> np.ndarray:
        """Encoder-reported angle = true angle + zero offset."""
        return np.atleast_2d(Q) + self.err.q_offset


def run_sweep(plant: MeasuredPlant, plan) -> dict:
    """Execute a :class:`~mabel_sysid.excitation.SweepPlan` against the plant.

    Returns the per-hold *settled* measurements, which is all the identification
    consumes:  ``{'angle', 'direction', 'q', 'tau', 'tau_std'}``, one row per
    hold, with the torque averaged over the settled part of the dwell.
    """
    # direction of travel at every sample: sign of the joint's velocity, held
    # through the dwell that follows it.
    j = plan.joint
    dirsig = np.zeros_like(plan.QD)
    running = np.zeros(plan.Q.shape[1])
    for i in range(plan.Q.shape[0]):
        v = plan.QD[i]
        moving = np.abs(v) > 1e-6
        running = np.where(moving, np.sign(v), running)
        dirsig[i] = running

    TAU = plant.measure(plan.Q, plan.QD, plan.QDD, dirsig)
    Qrep = plant.report_q(plan.Q)

    rows = {"angle": [], "direction": [], "q": [], "tau": [], "tau_std": []}
    settle = plan.limits.settle
    for h in plan.holds:
        m = (plan.t >= h.t_start + settle) & (plan.t <= h.t_end)
        if not m.any():
            continue
        rows["angle"].append(float(Qrep[m][:, j].mean()))
        rows["direction"].append(h.direction)
        rows["q"].append(Qrep[m].mean(axis=0))
        rows["tau"].append(TAU[m].mean(axis=0))
        rows["tau_std"].append(TAU[m].std(axis=0))
    return {k: np.asarray(v) for k, v in rows.items()}
