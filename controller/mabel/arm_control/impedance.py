"""impedance.py — the real-arm joint control POLICY, as three explicit modes.

The openarm arms run the DaMiao MIT law ON THE MOTOR:  T = kp·(pos − q) + kd·(vel − q̇)
+ t_ff. So unlike the sim (which owns the torque loop through the Plant seam and
:class:`JointPositionController`), the real arm is commanded by the five MIT parameters
(pos, vel, kp, kd, t_ff) and the firmware closes the loop. The three controllers the
operator asked for are therefore three PARAMETERISATIONS of that one command:

    ArmControlMode.PURE_PD    kp = full,  t_ff = 0        baseline; droops under gravity
    ArmControlMode.GRAVITY_PD kp = full,  t_ff = τ_g(q)   gravity cancelled → tracks
    ArmControlMode.COMPLIANT  kp > 0 + CLUTCH, t_ff = τ_g drag it; the spring holds it

This module owns ONLY the control policy (which kp/kd/t_ff, and which position setpoint).
The frame conversion (model↔raw motor), the 2π multi-turn fold, and the end-stop clamp stay
in the transport layer (the hardware bridge / a Plant), so the policy is identical for sim
and real and is trivially unit-testable. Gravity τ_g comes from
:class:`mabel.arm_control.gravity.ArmGravityCompensator` (already in the raw motor frame).

WHY COMPLIANCE IS kp>0 + A CLUTCH, not kp≈0. Gravity comp is never perfect — system ID has
error (measured live: the J1 feed-forward over-compensates ~3°), leaving a RESIDUAL torque
τ_res = τ_g_model − τ_g_true. With kp = 0 nothing opposes τ_res, so the arm slowly DRIFTS
(or falls) — kp = 0 only holds if gravity comp is exact, which it is not. So keep a real kp
and hold a position SETPOINT the spring can pull toward: at rest, kp·(q_des − q) develops
exactly the τ_res the model got wrong, and the joint settles at a small bounded offset
q_des − τ_res/kp instead of drifting away.

But a fixed setpoint would fight the operator's push. So the setpoint CLUTCHES: while the
joint is MOVING (|q̇| > v_move — the human is dragging it) q_des tracks the measured q, so
kp·(q_des − q) ≈ 0 and the arm is free (only kd damping resists); the moment the joint goes
STILL (the human let go) q_des FREEZES at the current pose and kp holds it there against the
gravity residual. "Push it anywhere and it stays" — exactly, and robust to imperfect τ_g.
The clutch state lives on the ArmImpedance instance (per joint), so this class is stateful
in COMPLIANT mode; reset it (set_mode / reset_clutch) when re-engaging.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ArmControlMode(enum.Enum):
    PURE_PD = "pure_pd"          # kp full, t_ff = 0        — Day-1 baseline (droops)
    GRAVITY_PD = "gravity_pd"    # kp full, t_ff = τ_g      — Day-2 tracking
    COMPLIANT = "compliant"      # kp ≈ 0, t_ff = τ_g, kd   — drag-and-hold (admittance-like)

    @classmethod
    def coerce(cls, m) -> "ArmControlMode":
        if isinstance(m, cls):
            return m
        s = str(m).strip().lower()
        for v in cls:
            if s in (v.value, v.name.lower()):
                return v
        # friendly aliases from the UI vocabulary
        if s in ("stiff", "impedance", "position"):
            return cls.GRAVITY_PD
        if s in ("soft", "compliance", "admittance", "backdrive", "free"):
            return cls.COMPLIANT
        if s in ("pd", "baseline", "raw"):
            return cls.PURE_PD
        raise ValueError(f"unknown arm control mode {m!r}")


@dataclass(frozen=True)
class JointCommand:
    """The per-joint outcome of the policy. The transport turns this into the MIT frame:
    ``kp/kd/t_ff`` go on the wire as-is. The position command is picked by:
      * ``hold_pos`` not None → command EXACTLY this raw setpoint (the COMPLIANT clutch owns
        it); the transport still end-stop-clamps but does no 2π fold (it is already raw).
      * else ``track`` True  → the app/target angle (clamped, 2π-folded by the transport).
      * else                 → the joint's MEASURED / latched-hold angle."""
    kp: float
    kd: float
    t_ff: float
    track: bool                       # True: command the target; False: command held pose
    hold_pos: float = None            # explicit raw setpoint owned by the policy (clutch)


@dataclass(frozen=True)
class ComplianceTuning:
    """The tunables for COMPLIANT mode — the ones the operator dials in on the hand-drag.

    kp_scale : fraction of the joint's normal kp kept as the HOLDING spring. NON-zero (that
               was the point — gravity comp is imperfect, so kp must hold the residual). Too
               low → the joint settles far below the setpoint under the gravity residual (and
               feels loose); too high → a stiffer detent when you let go. ~0.3 is a good start.
    kd_scale : fraction of the joint's normal kd kept as viscous damping. Main FEEL knob when
               dragging: too low and a nudged joint coasts/oscillates; too high and it feels
               sticky. Start ~0.6.
    kd_floor : absolute minimum kd (N·m·s/rad) so a joint with tiny model kd is not twitchy.
    v_move   : joint speed (rad/s) above which the CLUTCH treats the joint as being pushed and
               lets q_des track q (free); below it, q_des freezes and kp holds. Too low →
               chatters/creeps on sensor noise; too high → a slow push meets the spring. """
    kp_scale: float = 0.3
    kd_scale: float = 0.6
    kd_floor: float = 0.4
    v_move: float = 0.08


class ArmImpedance:
    """Maps a control MODE + per-joint base gains + gravity FF → per-joint MIT parameters.

    Parameters
    ----------
    base_gains : {fw_joint: (kp, kd)}  the joint's normal (stiff) position gains, already
                 clamped to the motor's kp_max/kd_max by the caller.
    mode       : ArmControlMode (or a coercible string). Default GRAVITY_PD.
    gain_scale : global multiplier on kp only (detune a shaking loop); never touches kd.
    compliance : ComplianceTuning for COMPLIANT mode.
    """

    def __init__(self, base_gains, mode=ArmControlMode.GRAVITY_PD, *,
                 gain_scale=1.0, compliance: ComplianceTuning = None):
        self.base_gains = dict(base_gains or {})
        self.mode = ArmControlMode.coerce(mode)
        self.gain_scale = float(gain_scale)
        self.compliance = compliance or ComplianceTuning()
        self._q_des: dict = {}        # fw -> frozen raw setpoint (the COMPLIANT clutch)

    def set_mode(self, mode) -> None:
        new = ArmControlMode.coerce(mode)
        if new is not self.mode:
            self.reset_clutch()       # re-anchor on any mode change (no stale setpoint)
        self.mode = new

    def reset_clutch(self) -> None:
        """Forget every frozen COMPLIANT setpoint; each joint re-anchors to its measured
        pose on the next command. Call on (re)engage so the arm never snaps to an old hold."""
        self._q_des.clear()

    def seed_clutch(self, fw_joint: str, q_des: float) -> None:
        """Force the clutch's frozen setpoint for one joint (raw frame). Used for a
        TORQUE-CONTINUOUS handoff into COMPLIANT: the transport seeds
        q_des = q + (kp_high/kp_low)·(target − q) so the low-kp spring initially exerts the
        SAME torque the high-kp mode was holding — the arm does not jump when the stiffness
        drops. Thereafter the normal clutch (freeze-when-still, track-when-pushed) takes over."""
        self._q_des[fw_joint] = float(q_des)

    @property
    def compliant_kp_ratio(self) -> float:
        """kp_high / kp_low for the current compliance tuning (∞-safe: 0 kp_scale → 0)."""
        s = self.compliance.kp_scale
        return (1.0 / s) if s > 1e-6 else 0.0

    @property
    def gravity_on(self) -> bool:
        """Does this mode feed gravity forward? (PURE_PD is the only one that does not.)"""
        return self.mode is not ArmControlMode.PURE_PD

    def command(self, fw_joint: str, gravity_tff: float = 0.0,
                q_meas: float = None, qd_meas: float = None) -> JointCommand:
        """The MIT parameters for one joint under the current mode.

        ``gravity_tff`` is τ_g for this joint in the RAW MOTOR frame (from
        ArmGravityCompensator.tff); applied for every mode except PURE_PD.
        ``q_meas`` / ``qd_meas`` are the joint's measured RAW angle and speed — required by
        the COMPLIANT clutch, ignored by the other modes."""
        kp0, kd0 = self.base_gains.get(fw_joint, (0.0, 0.0))
        kp0 *= self.gain_scale
        ff = 0.0 if self.mode is ArmControlMode.PURE_PD else float(gravity_tff)

        if self.mode is ArmControlMode.COMPLIANT:
            c = self.compliance
            kp = kp0 * c.kp_scale             # NON-zero: the spring that holds the residual
            kd = max(kd0 * c.kd_scale, c.kd_floor)
            hold = self._clutch(fw_joint, q_meas, qd_meas)
            return JointCommand(kp=kp, kd=kd, t_ff=ff, track=False, hold_pos=hold)

        # PURE_PD / GRAVITY_PD: full stiffness, track the commanded target.
        return JointCommand(kp=kp0, kd=kd0, t_ff=ff, track=True)

    def _clutch(self, fw: str, q_meas, qd_meas):
        """The COMPLIANT setpoint: track the measured pose WHILE the joint is moving (the
        human is dragging → q_des follows q → the spring exerts ~0, so it is free), FREEZE it
        when the joint goes still (human let go → kp holds against the gravity residual).
        Returns the raw setpoint, or None if we have no measurement yet (transport then
        holds at the measured/latched pose)."""
        if q_meas is None:
            return self._q_des.get(fw)        # keep the last setpoint if we briefly lose q
        moving = (qd_meas is None) or (abs(qd_meas) > self.compliance.v_move)
        if moving or fw not in self._q_des:
            self._q_des[fw] = float(q_meas)   # clutch engaged: re-anchor to here
        return self._q_des[fw]
