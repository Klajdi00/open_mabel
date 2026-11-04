"""
motors.py
=========

Profiles for the Damiao actuators on MABEL.  The CAN limits (``p_max``,
``v_max``, ``t_max``) come straight from the firmware
(``firmware/openarm/include/config.h``) and **must match the values programmed
into each motor with the Damiao desktop tool** -- the MIT-mode protocol packs
floats into fixed-width integers using exactly these ranges, so a mismatch
silently rescales every position / velocity / torque on the wire.

``gear_ratio`` and ``kt_out`` are metadata used only for *reporting* (converting
the identified reflected inertia back to a rotor inertia, sanity-checking torque
magnitudes).  The identification itself works in joint/output units and reads the
torque the motor reports over CAN, so it does not depend on them being exact --
but fill them from the datasheet when you can.

NOTE on torque feedback: Damiao MIT feedback reports the *output-side* (post-
gearbox) torque estimate in N*m, derived from the q-axis current and the motor
torque constant.  All identification here is therefore in output/joint units,
which is also what MuJoCo's ``frictionloss`` / ``damping`` / ``armature`` expect.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MotorProfile:
    """Static description of one Damiao motor model."""

    name: str            # Damiao model, e.g. "DM-J4340"
    p_max: float         # rad   -- MIT position half-range  (+/- p_max)
    v_max: float         # rad/s -- MIT velocity half-range  (+/- v_max)
    t_max: float         # N*m   -- MIT torque half-range    (+/- t_max)
    gear_ratio: float    # output:rotor reduction N  (VERIFY from datasheet)
    kt_out: float        # N*m/A output-side torque constant (VERIFY; 0 = unknown)
    role: str            # where it sits on MABEL

    # MIT-mode gain ranges are fixed by the Damiao protocol, not per-motor.
    kp_max: float = 500.0
    kd_max: float = 5.0

    # Datasheet torque-speed envelope (joint/output side), for Isaac Lab DCMotor.
    # 0 = unknown -> the calibration falls back to t_max / v_max or flags it.
    #   tau_cont : continuous (rated) torque -> effort_limit       (THERMAL: datasheet only)
    #   tau_stall: stall/peak torque         -> saturation_effort  (measured/extrapolated)
    #   w_noload : no-load speed             -> velocity_limit     (measured/extrapolated)
    tau_cont: float = 0.0
    tau_stall: float = 0.0
    w_noload: float = 0.0


# Values mirror firmware/openarm/include/config.h.  gear_ratio / kt_out marked
# (VERIFY) are best-known defaults -- confirm against your datasheet revision.
#
# Two SEPARATE notions, do not confuse them:
#   * (p_max, v_max, t_max) are the MIT-protocol ENCODING ranges (DM_Control
#     Limit_Param) that the motor is flashed with -- they set the wire scaling and
#     MUST match the firmware.  They are NOT the datasheet physical limits.
#   * gear_ratio / tau_cont / tau_stall / w_noload are the DATASHEET physical
#     specs (used for reporting + as Isaac DCMotor fallbacks).
#
# Datasheet values verified 2026-06-17 against the official DaMiao site
# (damiaokeji.com), the OpenArm spec table, and the Seeed/Foxtech datasheets.
# DaMiao naming = DM-J<frame><ratio>: e.g. J43-10 = 43mm/10:1, J43-40 = 43mm/40:1,
# J80-09 = 9:1, J35-07 = 7:1, J104-22 = 22:1.  (This is why the old 10:1 on the
# J4340/J10422P was wrong.)
MOTORS = {
    # DM-J4310-2EC V1.1: ratio 10:1, rated 3 N*m / peak 7 N*m, no-load 200rpm@24V
    # / 400rpm@48V.  MIT ranges [12.5, 30, 10] (DM_CAN Limit_Param) -- note the
    # encoding t_max=10 is wider than the 7 N*m physical peak.
    "DM-J4310": MotorProfile(
        name="DM-J4310", p_max=12.5, v_max=30.0, t_max=10.0,
        gear_ratio=10.0, kt_out=0.0, role="bench/test",
        tau_cont=3.0, tau_stall=7.0),
    # DM-J10422P-2EC: ratio 22:1, rated 100 N*m / peak 400 N*m.  MIT [12.566,12.5,400].
    "DM-J10422P": MotorProfile(
        name="DM-J10422P", p_max=12.566, v_max=12.5, t_max=400.0,
        gear_ratio=22.0, kt_out=0.0, role="torso",
        tau_cont=100.0, tau_stall=400.0),
    # DM-J8009(P)-2EC: ratio 9:1, rated 20 N*m / peak 40 N*m.  MIT [12.566,33.5,40].
    "DM-J8009P": MotorProfile(
        name="DM-J8009P", p_max=12.566, v_max=33.5, t_max=40.0,
        gear_ratio=9.0, kt_out=0.0, role="shoulder",
        tau_cont=20.0, tau_stall=40.0),
    # DM-J4340-2EC: ratio 40:1, rated 9 N*m / peak 27 N*m, no-load 52.5rpm@24V /
    # 100rpm@48V (=10.5 rad/s ~ MIT v_max).  MIT [12.566, 10.4, 27].
    "DM-J4340": MotorProfile(
        name="DM-J4340", p_max=12.566, v_max=10.4, t_max=27.0,
        gear_ratio=40.0, kt_out=0.0, role="elbow",
        tau_cont=9.0, tau_stall=27.0, w_noload=10.4),
    # DM-J4340P-2EC: same specs as DM-J4340 (P = pro variant).  The motor on the
    # bench (ID 1).  no-load 100rpm@48V = 10.5 rad/s ~ MIT v_max=10.4.
    "DM-J4340P": MotorProfile(
        name="DM-J4340P", p_max=12.566, v_max=10.4, t_max=27.0,
        gear_ratio=40.0, kt_out=0.0, role="elbow/bench",
        tau_cont=9.0, tau_stall=27.0, w_noload=10.4),
    # DM-J3507-2EC: ratio 7:1, peak ~3 N*m.  MIT [12.566, 48, 3].  (rated torque
    # not published separately; left 0/unknown.)
    "DM-J3507": MotorProfile(
        name="DM-J3507", p_max=12.566, v_max=48.0, t_max=3.0,
        gear_ratio=7.0, kt_out=0.0, role="wrist/neck",
        tau_stall=3.0),
}


def get(name: str) -> MotorProfile:
    """Look up a profile by model name, with a friendly error."""
    if name not in MOTORS:
        raise KeyError(
            f"unknown motor '{name}'. known: {', '.join(sorted(MOTORS))}")
    return MOTORS[name]
