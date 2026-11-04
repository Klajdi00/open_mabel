"""gravity.py — torso gravity feed-forward + the torso's frame constants.

WHERE THIS LIVES AND WHY. Same rule as ``arm_control/gravity.py``: the control law
belongs under ``controller/`` — the hardware layers (mabel_hw driver, the setup studio)
only HOST the torso motor; the math is here so sim, studio, and the future HAL torso
subsystem all feed forward the identical torque.

THE LAW. The torso is ONE DaMiao DM-J10422P (22:1) running the MIT law
``τ = kp·(q*−q) + kd·(q̇*−q̇) + t_ff``. Measured on the joint (2026-07-26): ~7-10 N·m of
stiction and up to ~19 N·m of gravity load. kp alone must build a large position error
to produce that torque, so from the gravity-loaded forward stop a step-capped PD simply
stalls ("cannot move back up"). Gravity MUST be fed forward:  t_ff = τ_hold(θ)  deletes
the gravity term; PD then only fights friction and tracks.

THE MODEL (derived from the canonical MJCF — single source of truth). Everything above
the torso joint (torso link, neck/head, BOTH OpenArm arms with hands, ~20.27 kg subtree)
rotates rigidly about the joint at the nominal MJCF pose, so the holding torque is exactly
sinusoidal in the torso angle:

    τ_hold(θ) = A · sin(PHI − θ)        [N·m, MJCF/URDF frame]

Constants are BAKED (this module must import in the studio without MuJoCo); re-derive them
with ``controller/experiments/torso_gravity_derive.py`` whenever the MJCF masses, the
arms, or the torso geometry change (the tip_over.py rule). Validated against hardware
2026-07-26: predicted +19.0 N·m hold at θ=−35°, measured +21.1 N·m (incl. friction).

THE FRAME (the ±35°-slider / leaning-rig bug this module also fixes). The MJCF/URDF
torso joint is  θ ∈ [−1.0472, +0.2618] rad = [−60°(forward), +15°(back)],  0 = upright.
The studio's manual calibration is CENTER-anchored (model 0 = mid-travel), and the
measured raw span (1.31038 rad) matches the URDF span (1.3090) to 0.08° — so the two
frames differ by ONE constant:  θ_urdf = model + URDF_MID.  Gravity direction confirms
the orientation: the joint falls FORWARD (toward raw_min) under gravity, matching
τ_hold > 0 across the whole range.

POSTURE-AWARE FF (2026-07-29). The sinusoid above is exact only at the arm pose it
was fitted in (arms parked, no hands): the torso subtree CARRIES both arms, so an
extended arm moves the subtree CoM forward and the true hold torque grows by several
N·m that the sinusoid cannot see — the operator report was "torso too stiff", because
kp had to stay high (150) to cover the arm-pose error. The posture-aware path
evaluates the SAME MJCF the arm feedforward uses (one shared mj_forward in
``arm_control.ArmGravityCompensator`` — pass ``"torso"`` in the pose dict and the
torso's gravity torque comes back arm-pose-aware, hands honored via HandsConfig).
``posture_tau()`` below then applies the live-measured trim (scale·τ+offset, fitted
by ``hardware_bridge/experiments/exp_torso_gravity_sysid.py`` against the real
motor's holding torque and stored in control_param.json ``torso.gravity_feedforward``
— the ONE editable surface). ``hold_tau`` stays as the arm-pose-blind FALLBACK for
runs where the arms are not streaming (body-only bring-up) or MuJoCo is absent.
"""
from __future__ import annotations

import math

__all__ = ["TORSO_RANGE", "TORSO_URDF_MID", "GRAV_A", "GRAV_PHI",
           "hold_tau", "posture_tau", "ff_trim", "model_to_urdf", "urdf_to_model"]

# MJCF/URDF joint range (rad): -60 deg forward lean … +15 deg backward, 0 = upright.
TORSO_RANGE = (-1.0472, 0.2618)

# model -> URDF(0 = upright) offset, i.e. the URDF angle at model 0.
# 2026-07-27: ZERO. The torso calibration (torso/damiao in mabel.json) is now
# UPRIGHT-anchored — its model_mid is set to raw_mid so model == raw == θ_urdf
# (the motor was zeroed at upright; the raw window matches TORSO_RANGE to 0.08°).
# Before that the calib was mid-travel-anchored and this constant carried the
# mid-range shift (-0.3927), which is also why every viz leaned ~22.5° back at
# model 0. If the calibration is ever re-anchored, recompute this as
# "θ_urdf when the streamed model reads 0".
TORSO_URDF_MID = 0.0

# τ_hold(θ) = GRAV_A · sin(GRAV_PHI − θ).
#
# MEASURED ON THE REAL ROBOT 2026-07-27 (supersedes the MJCF derivation, whose
# nominal arm pose put the subtree CoM 7.5 cm forward of the hinge ⇒ a bogus
# 14.95 N·m at upright — the real machine BALANCES there, operator-confirmed:
# a limp torso rests at +0.6°). Sweep: hold at 7 angles across [−49°, +6°],
# down AND back up (so static friction cancels in the fit), reading the motor's
# own torque feedback from /body/joint_states. Least-squares fit of
# τ = A·sin(PHI−θ): rms 1.65 N·m ≈ the friction band (pass hysteresis ~±2 N·m,
# consistent with the ~7-10 N·m breakaway stiction being mostly held by the
# worm-ish 22:1 gear at rest). Curve: −60°→18.3, −30°→11.2, 0°→+1.1,
# +10°→−2.5 N·m; sign flips through the +3° balance point (hold always points
# back toward upright). Re-measure with
# controller/experiments/torso_gravity_measure.sh (gateway stopped, bridge up)
# — or re-derive from the MJCF only after its arm pose matches the real arms.
GRAV_A = 20.57             # N·m
GRAV_PHI = 0.0516          # rad (+3.0° — the measured balance point)


def hold_tau(theta_urdf: float) -> float:
    """Gravity holding torque (N·m, MJCF/URDF frame, 0 = upright) the actuator must
    supply to keep the torso static at ``theta_urdf``. Positive = toward backward
    (+θ). Zero at upright (balanced), positive when leaned forward (−θ), negative
    when leaned back — the hold always points back toward upright."""
    return GRAV_A * math.sin(GRAV_PHI - float(theta_urdf))


def model_to_urdf(model: float) -> float:
    """Studio center-anchored calibrated angle (0 = mid-travel) -> URDF angle (0 = upright)."""
    return float(model) + TORSO_URDF_MID


def urdf_to_model(theta_urdf: float) -> float:
    return float(theta_urdf) - TORSO_URDF_MID


def ff_trim() -> tuple:
    """(scale, offset_nm) for the posture-aware feedforward, LIVE from
    control_param.json ``torso.gravity_feedforward`` (the master control-law
    registry) — fitted by exp_torso_gravity_sysid.py against the real motor's
    measured holding torque. Missing file/block keeps the identity trim (the raw
    MJCF prediction), so nothing breaks without it."""
    try:
        from ..control_params import torso_gravity_ff
        d = torso_gravity_ff() or {}
        return float(d.get("scale", 1.0)), float(d.get("offset_nm", 0.0))
    except Exception:
        return 1.0, 0.0


def posture_tau(tau_model: float) -> float:
    """The deployed torso feedforward: the ARM-POSE-AWARE MJCF gravity torque
    (``ArmGravityCompensator.tff`` with ``"torso"`` in the pose, model frame)
    calibrated onto the real joint by the live-measured trim. Falls back to the
    identity trim when the registry is absent."""
    s, o = ff_trim()
    return s * float(tau_model) + o
