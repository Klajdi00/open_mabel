"""
frame.py
========

**The frame map -- and why a wrong one is the most dangerous bug in this package.**

The MJCF model and the motor encoders do not agree on what a joint angle means:

    q_motor = s_j * q_mjcf + o_j ,        tau_motor = s_j * tau_mjcf ,   s_j = +/-1

The offset ``o_j`` comes from homing (park the arm in a known reference posture
and read the encoders).  The **sign** ``s_j`` comes from how the motor was
mounted and how its CAN id was flashed, and MABEL has no recorded map for the
arm -- ``hardware_bridge``'s DaMiao driver explicitly leaves the wiring to
robot-specific config, and the firmware's ``teleop_gui.py`` lists joint limits
whose signs *disagree* with the MJCF for joints 1, 2 and 4.

Why this is worse than an ordinary bug: a flipped sign does not crash and does
not look wrong.  The measured torque is negated, the fitted gravity sinusoid is
mirrored, the least-squares still converges, the residual is still small, and you
paste a confidently wrong first moment into the MJCF.  Gravity compensation then
pushes the arm the way gravity is already pulling it.  **You cannot detect this
from the sweep data alone** -- both hypotheses fit a sinusoid equally well.

You can detect it by asking the *model* which hypothesis explains the data.  For
each joint we sweep, we hold both hypotheses up against the CAD prediction and
score them by correlation, not by absolute error -- the CAD masses may be 30%
off, but the *shape and phase* of the gravity sinusoid is set by the geometry,
which we trust.  The right sign correlates strongly; the wrong sign does not.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .arm_model import ArmModel


@dataclasses.dataclass
class SignVerdict:
    joint: int
    sign: int             # the sign the data supports
    r_plus: float         # correlation of the data with the s=+1 hypothesis
    r_minus: float        # ... with s=-1
    margin: float         # |r_plus - r_minus| -- how decisive the call is
    amplitude: float      # measured gravity swing (N*m)
    confident: bool       # margin big enough AND the winner is a good fit

    def explain(self) -> str:
        if not self.confident:
            return (f"J{self.joint+1}: INCONCLUSIVE (r+={self.r_plus:+.2f}, "
                    f"r-={self.r_minus:+.2f}, swing {self.amplitude:.3f} N*m). "
                    f"Too little gravity signal to call the sign. Re-run this "
                    f"joint at a posture with a bigger swing, or set it by hand "
                    f"after watching which way it physically moves.")
        return (f"J{self.joint+1}: sign = {self.sign:+d}  "
                f"(r+={self.r_plus:+.2f}, r-={self.r_minus:+.2f}, "
                f"margin {self.margin:.2f}, swing {self.amplitude:.3f} N*m)")


def verify_sign(cad: ArmModel, j: int, q_motor: np.ndarray, tau_motor: np.ndarray,
                offset_j: float, posture_mjcf: np.ndarray, *,
                min_margin: float = 0.5, min_r: float = 0.7,
                min_amplitude: float = 0.02) -> SignVerdict:
    """Decide the sign of joint ``j`` from a bidirectionally-averaged sweep.

    Parameters
    ----------
    q_motor, tau_motor : the encoder angle and (friction-free) torque of joint j
        at each hold, in MOTOR coordinates.
    offset_j : the homing offset already established for this joint.
    posture_mjcf : the MJCF posture the other joints were held at.
    """
    q_motor = np.asarray(q_motor, float)
    tau_motor = np.asarray(tau_motor, float)
    amp = 0.5 * (np.max(tau_motor) - np.min(tau_motor))

    scores = {}
    for s in (+1, -1):
        q_mjcf = (q_motor - offset_j) / s
        pred_mjcf = np.empty_like(q_mjcf)
        q = np.asarray(posture_mjcf, float).copy()
        for i, qq in enumerate(q_mjcf):
            q[j] = qq
            pred_mjcf[i] = cad.gravity_torque(q)[j]
        pred_motor = s * pred_mjcf
        # correlation: robust to a wrong overall mass scale, sensitive to phase
        a = tau_motor - tau_motor.mean()
        b = pred_motor - pred_motor.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores[s] = float(a @ b / denom) if denom > 1e-12 else 0.0

    best = max(scores, key=lambda s: scores[s])
    margin = abs(scores[+1] - scores[-1])
    confident = (margin >= min_margin and scores[best] >= min_r
                 and amp >= min_amplitude)
    return SignVerdict(joint=j, sign=int(best), r_plus=scores[+1],
                       r_minus=scores[-1], margin=float(margin),
                       amplitude=float(amp), confident=bool(confident))


def consistency_check(cad: ArmModel, Q: np.ndarray, TAU: np.ndarray) -> dict:
    """A blunt guard to run on the WHOLE dataset before identifying anything.

    Correlates the measured gravity torque against the CAD prediction, per joint.
    A correct frame map gives a strongly positive correlation on every joint that
    has real gravity signal.  A negative correlation means that joint's sign is
    flipped -- stop, do not identify.
    """
    pred = np.array([cad.gravity_torque(q) for q in Q])
    out = {"r": [], "flipped": [], "weak": []}
    for j in range(cad.nv):
        a = TAU[:, j] - TAU[:, j].mean()
        b = pred[:, j] - pred[:, j].mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        r = float(a @ b / d) if d > 1e-12 else 0.0
        swing = float(np.max(TAU[:, j]) - np.min(TAU[:, j]))
        out["r"].append(r)
        out["flipped"].append(bool(r < -0.3))
        out["weak"].append(bool(abs(r) < 0.3 and swing < 0.05))
    out["any_flipped"] = bool(any(out["flipped"]))
    return out
