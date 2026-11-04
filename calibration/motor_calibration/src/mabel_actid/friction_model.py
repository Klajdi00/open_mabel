"""
friction_model.py
=================

The joint-side actuator model we identify.

At constant velocity the rigid-body inertia term vanishes, and (on a bench with
no external load) the torque the motor must supply is *only* friction:

    tau_f(w) =  [ F_c+ + (F_s+ - F_c+) * exp(-|w/v_s|^delta) ] * 1(w>0)      Coulomb + Stribeck, forward
              - [ F_c- + (F_s- - F_c-) * exp(-|w/v_s|^delta) ] * 1(w<0)      Coulomb + Stribeck, reverse
              + F_v * w                                                       viscous

This is the classic Coulomb + viscous + Stribeck curve, with **direction
asymmetry** (a planetary gearbox is rarely symmetric: backlash, helical thrust
and seal drag differ between drive directions).  Special cases:

* ``F_s = F_c`` and symmetric  -> plain Coulomb + viscous (the MuJoCo model).
* ``F_v = 0``                  -> dry friction only.

Adding the rigid-body term gives the **full dynamic model** used when the data
contains acceleration (sine / chirp excitation):

    tau_motor =  J * a  +  tau_f(w)  +  tau_g(theta)

where ``J`` is the reflected inertia seen at the joint (rotor inertia * N^2 +
gearbox + any fixed bench load) and ``tau_g`` is an optional gravity/load term.

The fields map onto MuJoCo joint attributes:
    F_c (mean of +/-) -> frictionloss      F_v -> damping      J -> armature
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class FrictionParams:
    """Identified actuator parameters, joint/output side (SI units)."""

    Fc_pos: float = 0.0    # N*m   Coulomb (dry) friction, forward
    Fc_neg: float = 0.0    # N*m   Coulomb (dry) friction, reverse
    Fv: float = 0.0        # N*m*s/rad  viscous damping
    Fs_pos: float = 0.0    # N*m   break-away / static, forward (>= Fc_pos)
    Fs_neg: float = 0.0    # N*m   break-away / static, reverse (>= Fc_neg)
    v_s: float = 0.1       # rad/s Stribeck characteristic velocity
    delta: float = 2.0     # -     Stribeck shape exponent (2 = Gaussian)
    J: float = 0.0         # kg*m^2 reflected inertia at the joint (armature)

    # ---- convenience views -------------------------------------------------
    @property
    def Fc(self) -> float:
        """Direction-averaged Coulomb friction -> MuJoCo ``frictionloss``."""
        return 0.5 * (self.Fc_pos + self.Fc_neg)

    @property
    def Fs(self) -> float:
        return 0.5 * (self.Fs_pos + self.Fs_neg)

    def mujoco_attrs(self) -> dict:
        """The three numbers to paste into a MuJoCo <joint .../> element."""
        return {
            "frictionloss": round(self.Fc, 6),
            "damping": round(self.Fv, 6),
            "armature": round(self.J, 8),
        }

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrictionParams":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclasses.dataclass
class DCMotorParams:
    """The electrical torque–speed envelope, for Isaac Lab's ``DCMotor`` model.

    A DC/FOC actuator can deliver at most a velocity-dependent torque: flat at
    its current limit near zero speed, then dropping linearly as back-EMF eats
    the available voltage, reaching zero at the no-load speed.  Isaac Lab models
    this exactly (``DCMotorCfg``):

        tau_max(w) = clip( tau_stall * (1 - w / w_noload),  -inf, tau_cont )

    Fields are joint/output side (SI):
      * ``tau_stall``  N*m   stall torque  -> ``saturation_effort`` (line at w=0)
      * ``w_noload``   rad/s no-load speed -> ``velocity_limit``    (line at tau=0)
      * ``tau_cont``   N*m   continuous torque -> ``effort_limit``  (thermal clip)
      * ``tau_peak``   N*m   peak torque actually OBSERVED near zero speed
    """
    tau_stall: float = 0.0      # saturation_effort  (extrapolated line intercept)
    w_noload: float = 0.0       # velocity_limit     (extrapolated zero-torque speed)
    tau_cont: float = 0.0       # effort_limit       (datasheet; not free-bench measurable)
    tau_peak: float = 0.0       # max torque seen near w=0 (measured lower bound on stall)
    r2: float = 0.0             # quality of the linear envelope fit
    observed: bool = False      # True if a real torque droop was seen (else extrapolated)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
#  model evaluation                                                           #
# --------------------------------------------------------------------------- #
def friction_torque(p: FrictionParams, w: np.ndarray,
                    w_eps: float = 1e-3) -> np.ndarray:
    """Friction torque ``tau_f(w)`` for an array of joint velocities ``w``.

    ``w_eps`` smooths the sign discontinuity at zero with a ``tanh`` so the
    model stays differentiable for the optimiser; physically it is the width of
    the velocity band over which dry friction reverses.
    """
    w = np.asarray(w, dtype=float)
    s = np.tanh(w / w_eps)                 # smooth sign in [-1, 1]
    fwd = w >= 0.0

    Fc = np.where(fwd, p.Fc_pos, p.Fc_neg)
    Fs = np.where(fwd, p.Fs_pos, p.Fs_neg)
    stribeck = np.exp(-np.abs(w / max(p.v_s, 1e-9)) ** p.delta)
    coulomb = (Fc + (Fs - Fc) * stribeck) * s
    return coulomb + p.Fv * w


def dynamic_torque(p: FrictionParams, w: np.ndarray, a: np.ndarray,
                   tau_g: np.ndarray | float = 0.0,
                   w_eps: float = 1e-3) -> np.ndarray:
    """Full motor torque ``J*a + tau_f(w) + tau_g`` for time-series data."""
    return p.J * np.asarray(a, dtype=float) \
        + friction_torque(p, w, w_eps) + np.asarray(tau_g)


def torque_speed_envelope(dc: DCMotorParams, w: np.ndarray) -> np.ndarray:
    """Max deliverable torque vs speed (Isaac ``DCMotor`` upper envelope).

    ``clip(tau_stall * (1 - w/w_noload), -tau_cont, tau_cont)`` — the linear
    torque–speed line capped by the continuous-torque limit.
    """
    w = np.asarray(w, dtype=float)
    if dc.w_noload <= 0:
        return np.full_like(w, dc.tau_cont or dc.tau_stall)
    line = dc.tau_stall * (1.0 - w / dc.w_noload)
    cont = dc.tau_cont if dc.tau_cont > 0 else dc.tau_stall
    return np.clip(line, -cont, cont)
