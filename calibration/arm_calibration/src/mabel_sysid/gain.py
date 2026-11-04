"""
gain.py
=======

**Calibrating the torque scale -- the error that identification cannot touch.**

Every torque this pipeline consumes is the DaMiao motor's own estimate, derived
from winding current through a torque constant and a gearbox:

    tau_reported = k_j * tau_true ,      k_j = kt_error * gear_efficiency * ...

and ``mabel_actid.motors`` records ``kt_out = 0.0`` -- *unknown* -- for every
motor in the arm.  Nobody has measured it.

Why this matters more than anything else in the package: a scale error on the
torque is **mathematically indistinguishable from a heavier arm**.  If joint 2
reports 3% high, the least squares concludes the arm is 3% heavier, fits
beautifully (residual unchanged), and hands you a model that over-compensates
gravity by 3% forever.  The simulation study (experiments E10) measures the
damage: a +3% torque-constant error costs 0.112 N*m of gravity error, versus
0.0002 N*m with a clean chain.  It turns a 2000x improvement over CAD into a 4x
one.  Noise averages away; **this does not**.

The fix: a known payload
------------------------
Gravity identification cannot see the scale, but a *difference* measurement can.
Hold the arm at a fixed pose, record the torques, bolt on a **known** mass at a
**known** point on the hand, and record again.  Subtract:

    delta_tau_reported(q)  =  k_j * [ Y_g(q) . delta_theta ]

``delta_theta`` is not estimated -- it is *known*, because you know the mass you
just hung and where you hung it.  So each joint's gain falls straight out:

    k_j  =  sum_q  delta_tau_reported_j(q) * delta_tau_predicted_j(q)
            / sum_q  delta_tau_predicted_j(q)^2                        (least squares)

Everything else cancels: the arm's own unknown parameters, the friction (if you
keep the bidirectional averaging), the model error, the encoder offsets.  What is
left is exactly the multiplicative gain on that joint's torque channel.

Use a payload heavy enough to move the needle -- a 0.5-1.0 kg dumbbell weight
gives a several-N*m swing at the shoulder and a few tenths at the wrist -- and
measure it on a kitchen scale.  Its mass is then known to a fraction of a
percent, which is an order of magnitude better than the gain error you are
chasing.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .arm_model import ArmModel
from . import regressor as R


@dataclasses.dataclass
class PayloadSpec:
    """A known mass rigidly attached to a known link at a known place."""
    mass: float                 # kg, WEIGH IT -- do not trust the label
    link: int                   # index into arm.moving_body_ids (usually the last)
    pos: np.ndarray             # (3,) attachment point in that link's body frame, m

    def delta_theta(self, arm: ArmModel) -> np.ndarray:
        """The exact change to the parameter vector this payload represents."""
        d = np.zeros(4 * arm.n_links)
        k = self.link
        d[4 * k] = self.mass                              # + mass
        d[4 * k + 1: 4 * k + 4] = self.mass * np.asarray(self.pos, float)  # + m*c
        return d


@dataclasses.dataclass
class GainFit:
    k: np.ndarray               # (nv,) per-joint torque gain
    sigma_k: np.ndarray         # (nv,) 1-sigma
    r2: np.ndarray              # (nv,) fit quality per joint
    swing: np.ndarray           # (nv,) payload-induced torque swing (N*m)
    trusted: np.ndarray         # (nv,) bool: swing big enough to believe the gain

    def report(self) -> str:
        lines = [f"  {'joint':>6} {'gain k':>9} {'+/-':>8} {'R^2':>7} "
                 f"{'swing':>9}   status"]
        for j in range(self.k.size):
            st = ("ok" if self.trusted[j] else
                  "payload swing too small -- gain NOT measured, left at 1.0")
            lines.append(f"  {'J'+str(j+1):>6} {self.k[j]:>9.4f} "
                         f"{self.sigma_k[j]:>8.4f} {self.r2[j]:>7.4f} "
                         f"{self.swing[j]:>8.3f}N   {st}")
        bad = np.abs(self.k[self.trusted] - 1.0)
        if bad.size:
            lines.append("")
            lines.append(f"  worst measured gain error: {100*bad.max():.1f}%  "
                         f"(this maps ~1:1 onto your gravity-comp error)")
        return "\n".join(lines)


def calibrate_gain(arm: ArmModel, Q: np.ndarray,
                   tau_bare: np.ndarray, tau_loaded: np.ndarray,
                   payload: PayloadSpec, *,
                   min_swing: float = 0.05) -> GainFit:
    """Per-joint torque gain from a bare/loaded pair of static measurements.

    ``Q`` are the poses (the SAME poses for both runs -- do not re-plan them),
    ``tau_bare`` / ``tau_loaded`` the friction-free (bidirectionally averaged)
    torques with and without the payload.

    ``min_swing`` (N*m) is the smallest payload-induced torque swing we are
    willing to divide by.  A joint the payload barely loads -- a wrist roll whose
    axis points at the payload, say -- gives a near-zero denominator and a wild
    gain estimate.  We refuse to report those and leave their gain at 1.0 rather
    than inventing a number.
    """
    Q = np.atleast_2d(Q)
    dtau_meas = np.asarray(tau_loaded, float) - np.asarray(tau_bare, float)

    # what the payload SHOULD do, from geometry we know exactly
    Y = R.stack_gravity_regressor(arm, Q)                  # (N*nv, 4*n_links)
    dth = payload.delta_theta(arm)
    dtau_pred = (Y @ dth).reshape(Q.shape[0], arm.nv)

    nv = arm.nv
    k = np.ones(nv)
    sk = np.zeros(nv)
    r2 = np.zeros(nv)
    swing = np.zeros(nv)
    trusted = np.zeros(nv, dtype=bool)

    for j in range(nv):
        p = dtau_pred[:, j]
        y = dtau_meas[:, j]
        swing[j] = float(np.max(p) - np.min(p)) if p.size else 0.0
        denom = float(p @ p)
        if denom < 1e-12 or swing[j] < min_swing:
            continue                                        # leave k=1, untrusted
        kj = float(p @ y) / denom
        resid = y - kj * p
        dof = max(y.size - 1, 1)
        var = float(resid @ resid) / dof
        k[j] = kj
        sk[j] = float(np.sqrt(var / denom))
        ss = float(np.sum((y - y.mean()) ** 2))
        r2[j] = 1.0 - float(resid @ resid) / max(ss, 1e-18)
        trusted[j] = True

    return GainFit(k=k, sigma_k=sk, r2=r2, swing=swing, trusted=trusted)


def apply_gain(TAU: np.ndarray, fit: GainFit) -> np.ndarray:
    """Undo the measured torque gain, so the identification sees true torque."""
    k = np.where(fit.trusted, fit.k, 1.0)
    return np.asarray(TAU, float) / k
