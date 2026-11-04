"""Gravity compensation must run whether or not an operator pose is streaming.

THE BUG (2026-07-19): "when I do torque on the robot arm still sags down... when
user now does e-stop then torque on the robot arm just keeps dropping down."

`TeleopEngine.update()` recomputed the gravity feed-forward INSIDE
`if new_frame:` — the branch that only runs when a VR/webcam **pose** frame
arrives. Drive/navigation is stick-only: the iPhone and web clients send
`head=notAvailable` and no hands, so `pose is None`, `new_frame` is never true,
`d.qfrc_applied` stayed all-zero, and the arms hung on bare position servos.

Measured at the time:

    joint / impedance      16 gravity-comp DoFs   0.32 deg sag
    navigation / impedance  0 gravity-comp DoFs   3.68 deg sag   <-- the default!

Worse, `resume_hold()` adopts the MEASURED pose as the new command by design (so
re-energising never lurches), so each e-stop -> torque-on cycle baked the sag in
and the arm ratcheted further down every time.

Gravity comp is a property of the ROBOT holding itself up, not of anyone's body
stream, so it now runs on its own >=60 Hz clock.
"""
import numpy as np
import mujoco
import pytest

from conftest import _load, needs_model

ARM_ACTUATORS = [f"{side} Arm {i}" for side in ("Left", "Right") for i in range(1, 8)]

# The failure is ~3.7 deg; healthy is ~0.01-0.32 deg. 1.0 deg cleanly separates
# them and leaves room for solver/timestep noise.
SAG_LIMIT_DEG = 1.0


def _q(model, data, aid):
    return float(data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]])


def _settled_rig():
    model, data = _load()
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    teleop.controller.init_hold()
    teleop.controller.settle_done()
    for _ in range(300):
        mujoco.mj_step(model, data)
    return model, data, teleop


def _hold_and_measure(method, control_type, ticks=1200):
    """Enter a control mode, run the real facade loop, return (gravity DoFs, sag)."""
    model, data, teleop = _settled_rig()
    aids = [(model.actuator(n).id, n) for n in ARM_ACTUATORS]
    teleop.set_control_spec(method=method, control_type=control_type,
                            region="arm", stiffness=27.0)
    teleop.resume_hold()
    target = {n: _q(model, data, a) for a, n in aids}
    for _ in range(ticks):
        try:
            teleop.update(None, None, 1.0 / 240.0)
        except Exception:
            pass
        mujoco.mj_step(model, data)
    sag = max(abs(_q(model, data, a) - target[n]) for a, n in aids)
    return int(np.count_nonzero(data.qfrc_applied)), np.degrees(sag), (model, data, teleop)


@needs_model
@pytest.mark.parametrize("method,control_type", [
    ("navigation", "impedance"),     # Drive — the DEFAULT space, and the regression
    ("navigation", "compliance"),
    ("joint", "impedance"),          # Task / Joint — was already healthy
])
def test_arms_hold_against_gravity_in_every_control_mode(method, control_type):
    dofs, sag_deg, _ = _hold_and_measure(method, control_type)
    assert dofs > 0, (
        f"{method}/{control_type}: NO gravity compensation is being applied "
        "(qfrc_applied is all zero). This is the regression: gravity comp must "
        "not be gated on an operator pose frame — Drive is stick-only."
    )
    assert sag_deg < SAG_LIMIT_DEG, (
        f"{method}/{control_type}: arms sagged {sag_deg:.2f} deg (limit "
        f"{SAG_LIMIT_DEG}). Gravity feed-forward is missing or too weak."
    )


@needs_model
def test_navigation_is_not_the_weak_mode():
    """Pin the ASYMMETRY that made this so confusing: Drive behaved differently
    from Task/Joint even though the arms are doing the same thing (nothing)."""
    nav_dofs, nav_sag, _ = _hold_and_measure("navigation", "impedance")
    jnt_dofs, jnt_sag, _ = _hold_and_measure("joint", "impedance")
    assert nav_dofs > 0 and jnt_dofs > 0
    assert abs(nav_sag - jnt_sag) < SAG_LIMIT_DEG, (
        f"navigation sags {nav_sag:.2f} deg vs joint {jnt_sag:.2f} deg — the "
        "arms are unloaded in both, so the modes must agree"
    )


@needs_model
def test_estop_then_torque_on_does_not_ratchet_the_arm_down():
    """The operator-visible failure: cycle the e-stop and the arm walks downward.

    `resume_hold()` adopts the MEASURED pose on purpose (no lurch on re-energise),
    so ANY steady-state sag compounds once per cycle. With gravity comp live the
    measured pose IS the target, and the cycle is a no-op.
    """
    model, data, teleop = _settled_rig()
    teleop.set_control_spec(method="navigation", control_type="impedance",
                            region="arm", stiffness=27.0)
    teleop.resume_hold()
    aid = model.actuator("Left Arm 2").id

    def run(ticks):
        for _ in range(ticks):
            try:
                teleop.update(None, None, 1.0 / 240.0)
            except Exception:
                pass
            mujoco.mj_step(model, data)

    run(600)
    start = _q(model, data, aid)
    for _ in range(3):
        for _ in range(240):            # LIMP: no facade tick, gravity pulls
            mujoco.mj_step(model, data)
        teleop.resume_hold()            # torque back on: adopt measured
        run(600)
    drift_deg = abs(np.degrees(_q(model, data, aid) - start))
    assert drift_deg < SAG_LIMIT_DEG, (
        f"Left Arm 2 walked {drift_deg:.2f} deg over 3 e-stop -> torque-on "
        "cycles. The arm must return to the same place every time."
    )


@needs_model
def test_gravity_comp_survives_a_pose_free_stream():
    """The precise mechanism, pinned: no pose frames ever, gravity comp anyway."""
    model, data, teleop = _settled_rig()
    teleop.set_control_spec(method="navigation", control_type="impedance",
                            region="arm", stiffness=27.0)
    teleop.resume_hold()
    data.qfrc_applied[:] = 0.0
    for _ in range(120):                 # pose=None on EVERY tick
        try:
            teleop.update(None, None, 1.0 / 240.0)
        except Exception:
            pass
        mujoco.mj_step(model, data)
    assert np.count_nonzero(data.qfrc_applied) > 0, (
        "no pose frame ever arrived and gravity comp never ran — this is the "
        "exact `if new_frame:` gate that caused the bug"
    )
