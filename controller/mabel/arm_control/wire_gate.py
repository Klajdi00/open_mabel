"""wire_gate.py — build THE deployed command gate, one definition, one place.

`facade._glide_wire_gate` runs a single `JointGlideLimiter` over the FINAL ctrl
of every rate-limited actuator, whoever wrote it — the retarget engine, the
engagement supervisor's ease, a slider, the web task-space IK. That limiter is
the only thing on this host that bounds the firmware's `kp*(pos-q)` term, so how
it is CONFIGURED is a safety property, and a safety property should not be
assembled inline at the call site (it was: `server/sim_teleop_bridge.py` built it
from three env vars, and nothing else could reproduce the same object).

`deployed_glide_limiter()` is that assembly, importable by the bridge, by the
offline chain harnesses and by tests, so all three drive an identical gate.

Two things it fixes over the inline construction:

1. **The error ceiling is torque-referenced and per joint.** One `err_max` for
   the whole arm permits `kp_j * err_max`, and kp spans 8..32 N.m/rad while the
   motors span 10..54 N.m — so the same ceiling is 1.2x one joint's continuous
   rating and 2.9x another's (see `rate_limit.arm_error_ceilings`). Derived from
   a torque budget instead, every joint gets the same headroom.

2. **The neck is included.** `_RATE_LIMITED` used to be ("Arm", "Torso",
   "Wrist"), so retargeted neck commands reached the Dynamixels through no rate
   bound at all — a 40 deg head turn at engage is a step, and a Dynamixel in
   position mode answers a step by slamming at profile velocity. The neck is a
   position servo, so it gets a RATE bound and NO error ceiling (bounding the
   error of a servo that closes its own loop only adds lag).

`MABEL_ARM_TORQUE_GATE=0` restores the flat pre-2026-08 ceiling for A/B runs.
"""
from __future__ import annotations

import os

from .rate_limit import (DEFAULT_BLEND_SPEED, DEFAULT_ERR_MAX,
                         DEFAULT_MAX_SPEED, DEFAULT_TORQUE_BUDGET,
                         JointGlideLimiter, arm_error_ceilings)

__all__ = ["deployed_glide_limiter", "NECK_ACTUATORS", "NECK_MAX_SPEED",
           "NO_YIELD_ACTUATORS", "TORSO_ERR_MAX",
           "torque_gate_enabled", "err_ceilings_enabled",
           "velocity_ff_ceilings", "DEFAULT_VEL_FF_MAX"]

#: MJCF actuator names for the neck (position servos, not MIT joints).
NECK_ACTUATORS = ("Neck Yaw", "Neck Pitch", "Neck Roll")
#: Joints exempt from the glide's THERMAL YIELD. The yield reads a sustained
#: error as an operator PUSH and ratchets the setpoint down to `yield_rest`,
#: latching until the joint "catches back up". The TORSO carries the robot's own
#: trunk against gravity: a large sustained error is its normal condition while
#: being raised, and it cannot catch up at kp·0.12 — so the latch is permanent.
#: Symptom (2026-08-07): with the torso slumped on its −60° forward stop, ⌂ Home
#: commanded +15°, the emitted ctrl sat at measured + 0.120 rad (== yield_rest,
#: the fingerprint) indefinitely, and the torso never came back up. The err_max
#: spring still bounds kp·err; only the ratchet is removed.
NO_YIELD_ACTUATORS = ("Torso",)
#: rad — the torso's own error ceiling, replacing the fleet 0.8 now that the
#: ratchet no longer bounds it. Sizing (2026-08-07): the MJCF puts the trunk's
#: peak gravity torque at 8.1 N·m over the whole lean range, but the MEASURED
#: torso torque is 16.7 N·m at +15° — the model under-predicts by ~2.3x — so the
#: real requirement is ~19 N·m. At kp 90 that needs 0.21 rad; 0.45 gives 40 N·m,
#: about 2x headroom, and stays well under the 72 N·m the fleet default would
#: have allowed. This is also exactly why the yield deadlocked: its 0.12 rad
#: gentle hold is only 10.8 N·m — BELOW the torque needed to lift the trunk, so
#: the joint could never "catch back up" and release the latch.
TORSO_ERR_MAX = 0.45
#: rad/s. The neck carries the operator's cameras, so its bound is about what
#: the OPERATOR can stand to watch, not about torque: ~3 rad/s (170 deg/s) is a
#: brisk human head turn and is under the Dynamixels' profile velocity, so the
#: commanded motion is the one delivered. Override with MABEL_NECK_MAX_SPEED.
NECK_MAX_SPEED = float(os.getenv("MABEL_NECK_MAX_SPEED") or "3.0")


def _fw_to_mjcf(fw):
    """'l_j2' -> 'Left Arm 2' (the frame the limiter is keyed in)."""
    try:
        side, j = str(fw).split("_", 1)
        idx = int(j.lstrip("j"))
    except (ValueError, AttributeError):
        return None
    if side not in ("l", "r") or not 1 <= idx <= 7:
        return None
    return f"{'Left' if side == 'l' else 'Right'} Arm {idx}"


def torque_gate_enabled():
    return (os.getenv("MABEL_ARM_TORQUE_GATE", "1").strip().lower()
            not in ("0", "false", "off", "no"))


def err_ceilings_enabled():
    """Per-joint ERROR ceilings — OFF by default (operator direction, 2026-08-01).

    They are the one part of this gate that TIGHTENS an existing bound rather
    than adding a missing one, so they change the feel of every large position
    move. Measured (experiments/retarget_control_smoothness/probe_home_glide.py)
    they do not bind on a Home glide — peak commanded error 0.193 rad against a
    0.321 rad tightest ceiling, and the baseline/fixed traces are identical —
    so they are not implicated in the Home regression; they are off because the
    operator asked for them off, not because they were shown harmful.

    The other two pieces of the gate stay ON, because each closes a hole rather
    than tightening a bound: the velocity feed-forward had NO per-joint limit at
    all (a fleet ±8 rad/s = up to 3.7× a joint's continuous rating), and the
    neck had no rate bound whatsoever.

    MABEL_ARM_ERR_CEILINGS=1 re-enables.
    """
    return (os.getenv("MABEL_ARM_ERR_CEILINGS", "0").strip().lower()
            in ("1", "true", "on", "yes"))


def _registry():
    try:
        from ..control_params import arm_torque_gate
        return arm_torque_gate() or {}
    except Exception:                                           # noqa: BLE001
        return {}


def deployed_glide_limiter(*, max_speed=None, blend_speed=None, err_max=None,
                           torque_budget=None, gains=None, log=None):
    """The command gate the real robot runs.

    Precedence, tightest binding first: explicit argument > environment >
    `control_param.json` `arm.torque_gate` > module default. The registry is
    the editable surface (one file, no env hunt across launch scripts); the env
    stays available for a live A/B without an edit.
    """
    reg = _registry()

    def pick(arg, env, key, default):
        if arg is not None:
            return float(arg)
        v = os.getenv(env)
        if v not in (None, ""):
            return float(v)
        return float(reg.get(key, default))

    max_speed = pick(max_speed, "MABEL_ARM_MAX_SPEED", "max_speed",
                     DEFAULT_MAX_SPEED)
    blend_speed = pick(blend_speed, "MABEL_ARM_BLEND_SPEED", "blend_speed",
                       DEFAULT_BLEND_SPEED)
    err_max = pick(err_max, "MABEL_ARM_ERR_MAX", "err_max_flat", DEFAULT_ERR_MAX)
    budget = pick(torque_budget, "MABEL_ARM_TORQUE_BUDGET", "torque_budget",
                  DEFAULT_TORQUE_BUDGET)
    neck_speed = pick(None, "MABEL_NECK_MAX_SPEED", "neck_max_speed",
                      NECK_MAX_SPEED)
    torso_err = pick(None, "MABEL_TORSO_ERR_MAX", "torso_err_max",
                     TORSO_ERR_MAX)

    per_joint = {n: neck_speed for n in NECK_ACTUATORS}
    per_joint_err = {n: float("inf") for n in NECK_ACTUATORS}
    per_joint_err.update({n: torso_err for n in NO_YIELD_ACTUATORS})

    if torque_gate_enabled() and err_ceilings_enabled():
        if gains is None:
            try:
                from ..control_params import arm_gains
                gains = arm_gains()
            except Exception:                                   # noqa: BLE001
                gains = {}
        ceil = arm_error_ceilings(gains, budget, name_fn=_fw_to_mjcf)
        per_joint_err.update(ceil)
        if log and ceil:
            log("arm wire gate: torque-referenced error ceilings "
                "(budget %.2f x continuous): %s", budget,
                ", ".join(f"{k}={v:.2f}" for k, v in sorted(ceil.items())))
    elif log:
        log("arm wire gate: FLAT error ceiling %.2f rad "
            "(per-joint ceilings off — MABEL_ARM_ERR_CEILINGS=1 enables)",
            err_max)

    return JointGlideLimiter(max_speed=max_speed, blend_speed=blend_speed,
                             err_max=err_max, per_joint=per_joint,
                             per_joint_err=per_joint_err,
                             no_yield=NO_YIELD_ACTUATORS)


#: Fleet fallback for the velocity feed-forward clamp — the value
#: `ros_plant._arm_target_velocity` used before the ceilings existed.
DEFAULT_VEL_FF_MAX = 8.0


def velocity_ff_ceilings(gains=None, torque_budget=None, *, name_fn=None):
    """{firmware joint id: v_ff_max [rad/s]} for `/arms/command.velocity`.

    The MIT law's `kd*vel` term is a torque, and it was clamped to one
    fleet-wide +-8 rad/s — up to 3.7x a joint's continuous rating at the
    deployed kd (see `rate_limit.arm_velocity_ff_ceilings`). This derives it
    per joint from the same torque budget as the error ceiling, so the two
    halves of the commanded torque are bounded consistently.

    Returns {} when the gate is disabled or the registry is unreadable, which
    the caller must read as "use the fleet fallback" — never as "no limit".
    """
    if not torque_gate_enabled():
        return {}
    reg = _registry()
    budget = float(torque_budget if torque_budget is not None
                   else (os.getenv("MABEL_ARM_TORQUE_BUDGET")
                         or reg.get("torque_budget", DEFAULT_TORQUE_BUDGET)))
    if gains is None:
        try:
            from ..control_params import arm_gains
            gains = arm_gains()
        except Exception:                                       # noqa: BLE001
            return {}
    from .rate_limit import arm_velocity_ff_ceilings
    return arm_velocity_ff_ceilings(gains, budget, name_fn=name_fn,
                                    hi=DEFAULT_VEL_FF_MAX)
