"""admittance.py — FLOATING soft mode: joint-space arm admittance on the real contract.

WHAT THIS IS. Before this module, real-robot SOFT (compliance) froze the arm's
position targets at mode entry: the operator could drag the arm only by winding up
kp·error against the compliant gains, and it rubber-banded back on release. This
law makes the TARGET follow the drag instead — the arm floats:

    τ_ext = τ_g_model(q_meas) − τ_meas            (quasi-static observer)
    M_v q̈* + D_v q̇* = τ_ext                       (virtual mass–damper per joint)
    q* ← q* + q̇*·dt                               (streamed on the position wire)

At rest the motor's reported torque equals the gravity hold (τ_meas ≈ τ_g), so the
estimate is ~0 inside a per-joint DEADBAND that absorbs the model residual +
stiction band — the arm holds drift-free. A push unbalances the motor torque
(kp·error + kd·q̇ appear in τ_meas), the estimate turns on, and the virtual damper
integrates it into target motion: the target chases the hand, kp·error collapses,
and the arm coasts weightlessly; release → the estimate dies → the damper coasts
the target to rest and the arm PARKS where it was left (no spring-back).

SIGN. τ_meas is the MOTOR's applied torque (model frame — joint_group dir-corrects
the wire). Quasi-statics: 0 = τ_m + τ_ext − τ_g  ⇒  τ_ext = τ_g − τ_m. A downward
push drives q below target, kp pushes up, τ_m rises above τ_g, τ_ext < 0 = down. ✓

STABILITY. The estimate closes a loop through the servo: τ_ext ≈ kp·(q − q*) − …,
so q̇* ≈ (kp/D_v)(q − q*) — first-order, time constant D_v/kp. With the deployed
compliant gains (kp 8–25) and D_v ≥ ~1.5 the pole stays well below the 50 Hz tick.

SAFETY RAILS (each one is load-bearing):
  · deadband  — below it the estimate is ZERO: model error can never walk the arm;
  · err_max   — |q* − q_meas| is clamped (the glide-limiter lesson): the target can
    never run away from a stalled joint, so kp·error — hence torque — is bounded;
  · vmax      — target speed cap (a τ spike cannot fling the target);
  · joint-range margin — never park the target on a stop;
  · stale/missing feedback (None τ or q) → that joint HOLDS (no estimate, no motion).

The math lives here (controller/); the gateway (server/plant.py ROSPlant) only
hosts it — same layering as the gravity feedforward. Tunables are LIVE in
control_param.json ``arm.soft_float`` (control_params.arm_soft_float): per-joint
deadbands calibrated to the measured no-hands sysid residual band. Hands are NOT
mounted (2026-07) — the gravity model must be the hands=NONE compensator; re-tune
the deadbands when the ORCA hands are refitted.
"""
from __future__ import annotations

import math

__all__ = ["SoftArmAdmittance", "FLOAT_DEFAULTS"]

# Defaults, overridden by control_param.json arm.soft_float (LIVE). Deadbands are
# per-joint-CLASS: the shoulders carry the largest gravity-model residual (the
# 2026-07-28 no-hands sysid trims l_j1 0.40 / l_j2 0.72 show the model error the
# trims absorb on AVERAGE — pose-to-pose the residual band is ~1.5–2 N·m there,
# and J1 adds friction "far above model"), the distal joints are clean.
FLOAT_DEFAULTS = {
    # Deadbands cover the residual VARIATION around the auto-tare bias (below),
    # not the absolute model error — which is why they can stay small enough to
    # feel floaty even though the real robot's mass/CoM sysid is imperfect.
    # Sized against the emulated-real verification (exp_soft_admittance.py):
    # joint stiction eats ~1 N·m of a hand drag before the motor ever sees it,
    # so shoulder deadbands above ~1 N·m made a firm 10 N drag numb.
    "deadband_nm": {"j1": 0.7, "j2": 0.7, "j3": 0.5, "j4": 0.5,
                    "j5": 0.25, "j6": 0.25, "j7": 0.2},
    "virtual_damping": {"j1": 2.5, "j2": 2.5, "j3": 1.8, "j4": 1.8,
                        "j5": 0.8, "j6": 0.8, "j7": 0.6},   # N·m·s/rad
    "virtual_mass": {"j1": 0.35, "j2": 0.35, "j3": 0.15, "j4": 0.15,
                     "j5": 0.05, "j6": 0.05, "j7": 0.04},   # kg·m²
    # Post-breakaway sensitivity: drive = gain·(|est|−deadband). The quasi-static
    # observer only ever sees ext − stiction − local model error, so the visible
    # signal above the deadband is a fraction of the hand's force; the gain maps
    # that thin margin back into a light, floaty response once motion starts.
    "drive_gain": 2.5,
    "vmax": 1.2,          # rad/s target speed cap
    "err_max": 0.45,      # rad  |target − measured| clamp (bounds kp·error ≤ ~11 N·m at the kp-25 shoulders)
    "range_margin": 0.06,  # rad  never park a target on a joint stop
    "tau_lp": 0.15,       # s    low-pass on the τ_ext estimate (sensor grain)
    "engage_ramp_s": 0.4,  # s    fade the estimate in after engage (no entry kick)
    # AUTO-TARE bias estimator — the model-error defense (operator directive:
    # "the real robot system ID is not perfect … make sure floating still works").
    # At engage, and whenever the joint is QUIET (virtual velocity ~0, measured
    # velocity ~0), the observer's output IS the local gravity-model residual —
    # so it is learned as a per-joint bias and subtracted. The residual is a
    # smooth function of pose, so after any drag the next rest pose re-tares in
    # ~a second. Rate-limited (a genuine slow push is eaten only at bias_rate —
    # push harder than the deadband and the arm always answers) and clamped
    # (bias_max bounds how much error the tare may ever hide).
    # bias_rate MUST stay well below the rate a human push develops (~2 N·m/s
    # through the τ low-pass): at 1.0 the tare RACED the rising estimate inside
    # the deadband and ate a slow firm drag entirely (measured in the emulated-
    # real harness — the arm went numb after ~1.5 s of pushing). 0.15 re-tares a
    # new rest pose in a few seconds and can never keep up with a real push.
    "bias_rate": 0.15,    # N·m/s adaptation rate while quiet
    "bias_max": 5.0,      # N·m  |bias| clamp
    "bias_holdoff_s": 1.0,  # s  no adaptation this soon after any drive/motion
    "still_qd": 0.06,     # rad/s measured-velocity gate for "quiet"
    "still_v": 0.03,      # rad/s virtual-velocity gate for "quiet"
    # FLOAT BUDGET — the residual-creep bound. A gravity-model residual above
    # the deadband is INDISTINGUISHABLE from an operator push at this sensing
    # level (both are plant-side torques — the lead/lag discriminator we tried
    # provably cannot separate them), so instead the law bounds how far any
    # single out-of-band episode may float: the target may travel this many rad
    # per joint while the estimate stays continuously outside the deadband;
    # the counter RESETS the moment the estimate re-enters the band (a human
    # easing their grip for an instant — natural in any real drag). A
    # residual-driven glide never eases, so it freezes at the budget.
    "episode_budget_rad": 1.5,
}


def _cls(fw: str) -> str:
    """'l_j4' → 'j4' (per-class parameter lookup)."""
    return fw.split("_", 1)[1] if "_" in fw else fw


class SoftArmAdmittance:
    """Per-joint floating admittance for the 14 arm joints, model frame.

    Parameters
    ----------
    compensator : ArmGravityCompensator
        THE shared gravity model (hands per its HandsConfig — NONE today). Must be
        the same instance/config the feedforward uses, so the observer's residual
        is exactly what the deployed FF leaves uncompensated.
    joints : iterable of fw ids ('l_j1'..'r_j7')
    params : dict — control_params.arm_soft_float() overrides on FLOAT_DEFAULTS.
    """

    def __init__(self, compensator, joints, params=None, log=None):
        self._g = compensator
        self.joints = tuple(joints)
        self._log = log or (lambda *a, **k: None)
        p = dict(FLOAT_DEFAULTS)
        for k, v in (params or {}).items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k] = {**p[k], **v}
            elif not str(k).startswith("_"):
                p[k] = v
        self.p = p
        self._q_tgt: dict = {}        # fw -> target (rad, model frame)
        self._v: dict = {}            # fw -> virtual velocity (rad/s)
        self._tau_f: dict = {}        # fw -> low-passed τ_ext estimate
        self._bias: dict = {}         # fw -> auto-tared model residual (N·m)
        self._tared: set = set()      # joints whose ENTRY tare has been taken
        self._active_t: dict = {}     # fw -> last time the joint was "active"
        self._ep: dict = {}           # fw -> out-of-band episode travel (rad)
        self._engaged_t: float | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @property
    def engaged(self) -> bool:
        return self._engaged_t is not None

    def engage(self, q_meas: dict, now: float) -> None:
        """Seed targets = MEASURED pose (float from wherever the arm is; a stale
        target would be an entry lurch) and restart the ramp."""
        self._q_tgt = {j: float(q_meas[j]) for j in self.joints if j in q_meas
                       and q_meas[j] is not None and math.isfinite(q_meas[j])}
        self._v = {j: 0.0 for j in self._q_tgt}
        self._tau_f = {j: 0.0 for j in self._q_tgt}
        self._bias = {}
        self._tared = set()           # entry tare taken on each joint's first sample
        self._active_t = {}
        self._ep = {}
        self._engaged_t = float(now)
        self._log(f"soft-float ENGAGED: {len(self._q_tgt)} joints seeded from measured")

    def disengage(self) -> None:
        if self._engaged_t is not None:
            self._log("soft-float disengaged")
        self._engaged_t = None
        self._q_tgt.clear()
        self._v.clear()
        self._tau_f.clear()
        self._bias.clear()
        self._tared.clear()
        self._active_t.clear()
        self._ep.clear()

    # ── the per-tick law ──────────────────────────────────────────────────────
    def update(self, q_meas: dict, tau_meas: dict, dt: float, now: float,
               limits: dict | None = None, qd_meas: dict | None = None) -> dict:
        """One admittance tick → {fw: target rad} for the joints that can float
        this tick (fresh q AND τ; others are omitted = hold last command).

        q_meas / tau_meas : {fw: value|None}, MODEL frame.
        qd_meas           : {fw: rad/s|None} measured velocity — gates the
                            auto-tare ("quiet" detection); None disables gating
                            by measured velocity (virtual velocity still gates).
        limits            : {fw: (lo, hi)} joint range (rad) for the stop margin.
        """
        if self._engaged_t is None or dt <= 0.0:
            return {}
        dt = min(dt, 0.1)                       # a stalled loop must not integrate a leap
        ramp = 1.0
        if self.p["engage_ramp_s"] > 0:
            ramp = min(1.0, (now - self._engaged_t) / self.p["engage_ramp_s"])

        # gravity hold torque at the MEASURED pose (model frame, trimmed — the
        # same evaluation the FF streams, so the residual is the FF's residual)
        pose = {j: q_meas[j] for j in self.joints
                if q_meas.get(j) is not None and math.isfinite(q_meas[j])}
        tau_g = self._g.tff(pose, frame="model") if pose else {}

        out = {}
        a = dt / max(self.p["tau_lp"], dt)      # τ low-pass blend
        for j in self.joints:
            q = q_meas.get(j)
            tm = tau_meas.get(j)
            if (q is None or tm is None or not math.isfinite(q)
                    or not math.isfinite(tm) or j not in tau_g):
                # no honest feedback → this joint holds (absent from output)
                self._v[j] = 0.0
                continue
            if j not in self._q_tgt:            # hot-joined joint: seed, hold this tick
                self._q_tgt[j] = float(q)
                self._v[j] = 0.0
                self._tau_f[j] = 0.0
                continue
            c = _cls(j)
            # quasi-static external-torque observer, low-passed
            tau_ext = float(tau_g[j]) - float(tm)
            if j not in self._tared:
                # ENTRY TARE: the first sample after engage IS the local
                # gravity-model residual (nobody is pulling the instant the mode
                # flips). Tare on the RAW sample and SEED the filter with it —
                # taring the low-passed value would capture only dt/tau_lp of
                # the residual and leak the rest as a transient drive.
                self._tau_f[j] = tau_ext
                self._bias[j] = tau_ext
                self._tared.add(j)
            f = self._tau_f[j] = (1 - a) * self._tau_f.get(j, 0.0) + a * tau_ext
            # AUTO-TARE bias: the model-error defense. Beyond the entry tare,
            # adapt only while QUIET (virtual AND measured velocity ~0),
            # rate-limited and clamped, so the residual at each new rest pose is
            # re-learned in ~a second while a genuine push (which makes the
            # joint move) is never eaten.
            b = self._bias.get(j, 0.0)
            db = float(self.p["deadband_nm"].get(c, 1.0))
            qd = None if qd_meas is None else qd_meas.get(j)
            # QUIET = no motion, the de-biased estimate INSIDE the deadband, and
            # a HOLD-OFF since the last activity. Each condition closed a real
            # hole (measured in the emulated-real harness):
            #   · in-band  — a deadband-absorbed push produces no motion, so a
            #     motion-only gate let the tare eat it (numb-arm deadlock);
            #   · hold-off — right after a drag the estimate is still settling
            #     through the low-pass; taring there bakes the tail in.
            # The out-of-band threshold is 2·db, not db: between db and 2·db sits
            # a BOUNDARY ZONE where the drive is too weak to trip the creep-kill
            # (the virtual velocity hovers right at still_v) yet strong enough to
            # micro-creep forever if no tare path owns it — measured as a
            # perpetual ~0.03 rad/s walk. The slow tare owns that zone; a genuine
            # push sits far above 2·db and is still never eaten.
            active = (abs(f - b) > 2.0 * db
                      or abs(self._v.get(j, 0.0)) >= float(self.p["still_v"])
                      or (qd is not None and math.isfinite(qd)
                          and abs(qd) >= float(self.p["still_qd"])))
            if active:
                self._active_t[j] = now
            quiet = (not active
                     and now - self._active_t.get(j, -1e9)
                     >= float(self.p["bias_holdoff_s"]))
            if quiet:
                step_b = float(self.p["bias_rate"]) * dt
                b += max(-step_b, min(step_b, f - b))
            bmax = float(self.p["bias_max"])
            b = self._bias[j] = max(-bmax, min(bmax, b))
            # smooth deadband on the DE-BIASED estimate (continuous at the edge)
            eb = f - b
            mag = abs(eb)
            gain = float(self.p.get("drive_gain", 1.0))
            drive = 0.0 if mag <= db else math.copysign((mag - db) * gain, eb) * ramp
            # FLOAT BUDGET: reset while in-band; freeze the drive once an
            # uninterrupted out-of-band episode has travelled its budget.
            if drive == 0.0:
                self._ep[j] = 0.0
            elif self._ep.get(j, 0.0) >= float(self.p["episode_budget_rad"]):
                drive = 0.0
            # virtual mass–damper:  M v̇ = drive − D v
            M = float(self.p["virtual_mass"].get(c, 0.1))
            D = float(self.p["virtual_damping"].get(c, 1.5))
            v = self._v.get(j, 0.0)
            v += (drive - D * v) / max(M, 1e-4) * dt
            vmax = float(self.p["vmax"])
            v = max(-vmax, min(vmax, v))
            self._v[j] = v
            # integrate the target, then the rails
            t = self._q_tgt[j] + v * dt
            self._ep[j] = self._ep.get(j, 0.0) + abs(v) * dt
            em = float(self.p["err_max"])
            if t > q + em:
                t = q + em
                self._v[j] = 0.0               # anti-windup: a clamped target stops
            elif t < q - em:
                t = q - em
                self._v[j] = 0.0
            if limits and j in limits:
                lo, hi = limits[j]
                m = float(self.p["range_margin"])
                t = max(lo + m, min(hi - m, t))
            self._q_tgt[j] = t
            out[j] = t
        return out

    # telemetry / tests
    def state(self) -> dict:
        return {"targets": dict(self._q_tgt), "vel": dict(self._v),
                "tau_ext": dict(self._tau_f), "bias": dict(self._bias),
                "engaged": self.engaged}
