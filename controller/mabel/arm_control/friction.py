"""friction.py — Coulomb friction feed-forward for the arm joints.

WHY THIS EXISTS.  Once the gravity feed-forward is correct, the arm's remaining
steady-state error is not gravity and not a tuning oversight — it is **dry friction**,
and it has a closed form.  The DaMiao arms run the MIT law in firmware,

    tau = kp (q* - q) + kd (qd* - qd) + t_ff

and a joint with Coulomb friction ``tau_c`` does not converge to its setpoint: it
sticks anywhere the net actuator torque fails to break friction, i.e. anywhere inside

    |kp (q* - q)| < tau_c        =>        |e_ss| < tau_c / kp                    (1)

That is a DEADBAND, not a transient.  Waiting longer does not shrink it, and it is
invisible to a gravity model, which is why it survives a feed-forward that is exact.

Measured on the MABEL plant (simulated path-following study, 2026-08-06, gravity FF exact
to 0.000 N.m): the per-joint RMS tracking error along a slow 3-D path came out at
0.62-1.09x of ``tau_c/kp`` on every one of the fourteen arm joints.  Equation (1) is
therefore not a plausible story about the error — it *is* the error.

TWO WAYS OUT, and this module is the second one:

  1. Raise ``kp``.  Error falls as 1/kp, and the ceiling is damping, not the protocol:
     the MIT packing caps ``kd`` at 5, so a joint of inertia ``J`` can only be driven
     to ``kp <= (b + kd)^2 / (4 zeta^2 J)`` before it rings.  Cheap, but bounded.
  2. Cancel ``tau_c`` directly.  Feed it forward and equation (1)'s deadband shrinks by
     whatever fraction of the friction the model gets right.  Unbounded in principle,
     and it costs no stability margin — which is the whole attraction.

THE LAW.

    t_fric(qd*) = scale * tau_c * tanh(qd* / v_eps)                               (2)

Three deliberate choices, each of which is the difference between this helping and
this oscillating:

  * **COMMANDED velocity, never measured.**  ``sign(qd_meas)`` closes a positive
    feedback loop through the velocity estimate: at rest the estimate dithers about
    zero, the feed-forward slams +-tau_c with it, and the joint buzzes.  The commanded
    velocity is noise-free and known one tick early, which is what a feed-forward is
    for.
  * **tanh, not sign.**  A hard sign() steps 2*tau_c N.m across zero crossings — twice
    per revolution on a circular path — which excites exactly the modes the gain
    ceiling was protecting.  ``v_eps`` sets the width of the blend; it should be a few
    times the velocity resolution, NOT so small that tanh becomes sign again.
  * **scale < 1.**  Over-compensating dry friction is unstable: if the feed-forward
    exceeds the true friction, the excess drives the joint past the setpoint, the sign
    flips, and the joint limit-cycles.  Under-compensating merely leaves a smaller
    deadband.  The asymmetry is total, so the default is deliberately timid.

NOT MODELLED, on purpose: viscous friction (the MJCF/joint damping term already
carries it and the kd term handles the rest), Stribeck velocity-weakening, and
position-dependent friction.  Each needs identification data this robot does not have
yet; adding an unidentified term to a feed-forward is how you get a confident model of
the wrong thing.

STATUS AND SAFETY.  **Off by default.**  This law injects torque as a function of the
COMMAND, so a bad ``tau_c`` estimate injects torque into a stationary arm.  Enable it
per joint, from measured friction, and ramp ``scale`` up from small — the same
discipline the gravity feed-forward got.  ``t_max`` clamps the contribution to a hard
per-joint ceiling regardless of what the model asks for.

The simulated evidence (conditions
``gff_tuned`` and ``gff_tuned_fric``).  It has NOT been validated on hardware.
"""
from __future__ import annotations

import math

__all__ = ["FRIC_SCALE_DEFAULT", "FRIC_VEPS_DEFAULT", "FrictionFF",
           "friction_deadband"]

# ── defaults, chosen from a measured trade-off, not from taste ────────────────
# path-following friction sweep, helix path, tuned gains,
# reporting (de-biased fingertip error [mm] / oscillation index):
#
#     scale \ v_eps      0.02          0.06          0.15
#        0.40         7.96 / 0.151  9.10 / 0.128  10.08 / 0.112
#        0.60         6.55 / 0.206  8.22 / 0.151   9.63 / 0.121
#        0.80         5.38 / 0.285  7.42 / 0.179   9.20 / 0.130
#
# The trade is monotone in both knobs and there is no free-lunch corner: every
# millimetre of accuracy is bought with chatter.  (0.80, 0.02) is the most ACCURATE
# corner and also the CHATTIEST — a bad default for a law someone might switch on
# without reading this file, and doubly so on hardware, where the chatter lands in
# real gearboxes.  The defaults below sit mid-grid: ~2/3 of the available accuracy
# gain for ~half the chatter of the aggressive corner, and only ~1.5x the oscillation
# of running the tuned gains with no friction feed-forward at all.
#
# There is no single right point on this curve; pick from the table for the task.

#: Fraction of the modelled friction actually fed forward.  Below 1 by design — see
#: "scale < 1" above; over-compensation limit-cycles, under-compensation just leaves a
#: smaller deadband, and the two failure modes are not symmetric.
FRIC_SCALE_DEFAULT = 0.60

#: Blend half-width [rad/s] of the tanh that replaces sign().  Wider than the sharpest
#: swept value on purpose: sharper buys ~1.7 mm and costs ~35 % more chatter.
FRIC_VEPS_DEFAULT = 0.06


def friction_deadband(tau_c: float, kp: float) -> float:
    """The steady-state error a PD joint cannot remove: ``tau_c / kp`` [rad].

    Exposed because it is the number that decides whether a gain change or a friction
    feed-forward is the right fix, and it is worth being able to print it without
    re-deriving equation (1) from memory.
    """
    kp = float(kp)
    return abs(float(tau_c)) / kp if kp > 0 else float("inf")


class FrictionFF:
    """Per-joint Coulomb friction feed-forward, driven by the COMMANDED velocity.

    Parameters
    ----------
    tau_c : {joint: N.m}
        Modelled Coulomb friction per joint.  On the MJCF plant this is the model's
        own ``frictionloss``; on hardware it must be IDENTIFIED (the breakaway torque
        from a slow bidirectional sweep) — the two are not interchangeable, and the
        real arms are already known to run above the model on J1.
    scale : float
        Global under-compensation factor, ``FRIC_SCALE_DEFAULT`` by default.
    v_eps : float
        tanh blend half-width [rad/s].
    t_max : {joint: N.m} or None
        Hard per-joint ceiling on the contribution.  Defaults to ``tau_c`` itself,
        which means the law can never ask for more than one friction's worth of
        torque however the velocity is scaled.
    enabled : bool
        Off by default.  See the module docstring.
    """

    def __init__(self, tau_c: dict, *, scale=FRIC_SCALE_DEFAULT,
                 v_eps: float = FRIC_VEPS_DEFAULT, t_max: dict = None,
                 enabled: bool = False):
        self.tau_c = {k: abs(float(v)) for k, v in (tau_c or {}).items()}
        # `scale` may be PER JOINT.  The accuracy/chatter trade is not the same at
        # every joint: chatter comes from the tanh switching through a reversal, and a
        # shoulder on a slow reach reverses far less often than a wrist, which rolls
        # continuously.  Measured on the tuned plant, the residual after the global
        # 0.6 sat at 9-14 mrad on J2/J3/J4 and 1.6-3.1 mrad on J5-J7 — so the proximal
        # joints are where the remaining error is AND where over-compensation is
        # cheapest.  A scalar applies to every joint, as before.
        self.scale = (dict(scale) if isinstance(scale, dict) else float(scale))
        self.v_eps = max(1e-6, float(v_eps))
        self.t_max = dict(t_max) if t_max else dict(self.tau_c)
        self.enabled = bool(enabled)

    def snapshot(self) -> dict:
        """The block that goes into an experiment's archived configuration."""
        return {"law": "t_fric = scale * tau_c * tanh(qd_cmd / v_eps)",
                "source": "controller/mabel/arm_control/friction.py",
                "enabled": self.enabled,
                "scale": (dict(self.scale) if isinstance(self.scale, dict)
                          else self.scale),
                "scale_per_joint": isinstance(self.scale, dict),
                "v_eps_rad_s": self.v_eps,
                "tau_c_nm": dict(self.tau_c), "t_max_nm": dict(self.t_max),
                "velocity": "COMMANDED (never measured — see module docstring)"}

    def scale_of(self, joint: str) -> float:
        """This joint's under-compensation factor (per-joint table or the scalar)."""
        if isinstance(self.scale, dict):
            return float(self.scale.get(joint, FRIC_SCALE_DEFAULT))
        return float(self.scale)

    def tff(self, cmd_vel: dict) -> dict:
        """{joint: N.m} for the commanded joint velocities.  ``{}`` when disabled, so
        a caller can add it unconditionally and the off state costs nothing."""
        if not self.enabled:
            return {}
        out = {}
        for j, tc in self.tau_c.items():
            sc = self.scale_of(j)
            if sc == 0.0:
                continue
            v = float(cmd_vel.get(j, 0.0))
            if not math.isfinite(v):
                continue
            t = sc * tc * math.tanh(v / self.v_eps)
            cap = float(self.t_max.get(j, tc))
            out[j] = max(-cap, min(cap, t))
        return out

    def deadbands(self, gains: dict) -> dict:
        """{joint: rad} residual deadband ``(1 - scale) * tau_c / kp`` under this law.

        The prediction the experiment checks: with ``scale = 0.8`` the deadband should
        fall to a fifth of its uncompensated value, and no further.
        """
        out = {}
        for j, tc in self.tau_c.items():
            g = gains.get(j)
            kp = float(g[0] if isinstance(g, (tuple, list)) else g or 0.0)
            resid = tc * ((1.0 - self.scale_of(j)) if self.enabled else 1.0)
            out[j] = friction_deadband(resid, kp)
        return out
