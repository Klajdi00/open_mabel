"""Compliance control: back-drivability under a push, holding after release,
region selection (arms keeps the lift on its servo; upper body frees it), and
the whole-body base glide through the runtime engine."""
import numpy as np
import mujoco

from conftest import needs_model, settle, arm_qpos

import mabel.config as C
from mabel.plant import SimPlant
from mabel.whole_body_controller.compliance import ComplianceController
from mabel.whole_body_controller import WBCEngine


def _push(model, data, body, F, ctl_update, seconds):
    bid = model.body(body).id
    for _ in range(int(seconds / model.opt.timestep)):
        data.xfrc_applied[bid, :3] = F
        ctl_update()
        mujoco.mj_step(model, data)
    data.xfrc_applied[bid, :3] = 0.0


@needs_model
def test_arms_backdrivable_and_hold_after_release(fresh):
    model, data = fresh()
    settle(model, data)
    ctl = ComplianceController.for_region(model, data, SimPlant(model, data),
                                          region="arms")
    left = [j for j in ctl.compliant if j.startswith("Left")]
    q0 = arm_qpos(model, data, left)

    # push the left palm sideways for 1 s -> the arm must yield
    _push(model, data, C.LEFT_PALM, np.array([0.0, 15.0, 0.0]), ctl.update, 1.0)
    q_pushed = arm_qpos(model, data, left)
    moved = np.abs(q_pushed - q0).max()
    assert moved > 0.1, f"arm did not yield to a 15 N push ({moved:.3f} rad)"

    # release -> it should settle near where it was left (gravity comp holds)
    for _ in range(int(1.0 / model.opt.timestep)):
        ctl.update()
        mujoco.mj_step(model, data)
    settle_drift = np.abs(arm_qpos(model, data, left) - q_pushed).max()
    assert settle_drift < 0.35, f"arm ran away after release ({settle_drift:.3f} rad)"


@needs_model
def test_region_arms_keeps_lift_servo(fresh):
    model, data = fresh()
    settle(model, data)
    lift_aid = model.actuator("Lift").id
    gain_before = model.actuator_gainprm[lift_aid].copy()
    ComplianceController.for_region(model, data, SimPlant(model, data), region="arms")
    assert np.allclose(model.actuator_gainprm[lift_aid], gain_before), \
        "arms region must NOT neutralise the lift servo"


@needs_model
def test_region_upper_body_frees_lift(fresh):
    model, data = fresh()
    settle(model, data)
    lift_aid = model.actuator("Lift").id
    ComplianceController.for_region(model, data, SimPlant(model, data),
                                    region="upper_body")
    assert np.allclose(model.actuator_gainprm[lift_aid], 0.0), \
        "upper_body region must neutralise the lift servo (back-drivable lift)"


@needs_model
def test_wbc_engine_whole_body_base_glides_under_push(fresh):
    model, data = fresh()
    settle(model, data)
    eng = WBCEngine(model, data)
    eng.enter("compliance", "whole_body")
    dt = model.opt.timestep
    bid = model.body(C.LEFT_PALM).id
    # sustained strong pull forward on the palm -> the base must admit + glide
    for _ in range(int(2.0 / dt)):
        data.xfrc_applied[bid, :3] = np.array([-30.0, 0.0, 0.0])
        eng.step(dt)
        mujoco.mj_step(model, data)
    speed = float(np.hypot(eng.vx, eng.vy))
    assert speed > 0.02, f"whole-body compliance base did not glide (|v|={speed:.3f})"
    # and the commanded twist obeys the tip-safe envelope
    from mabel.motion_model import tip_usage
    assert tip_usage(eng.vx, eng.vy, eng.w, 0, 0, 0, 0.0) <= 1.0 + 1e-6

    # release: the base must coast to rest, not roll forever
    data.xfrc_applied[bid, :3] = 0.0
    for _ in range(int(3.0 / dt)):
        eng.step(dt)
        mujoco.mj_step(model, data)
    assert abs(eng.vx) < 0.02 and abs(eng.vy) < 0.02, \
        f"base kept rolling after release ({eng.vx:.3f}, {eng.vy:.3f})"
    eng.exit()


@needs_model
def test_wbc_engine_arm_region_holds_base(fresh):
    model, data = fresh()
    settle(model, data)
    eng = WBCEngine(model, data)
    eng.enter("compliance", "arm")
    dt = model.opt.timestep
    bid = model.body(C.LEFT_PALM).id
    for _ in range(int(1.0 / dt)):
        data.xfrc_applied[bid, :3] = np.array([-25.0, 0.0, 0.0])
        eng.step(dt)
        mujoco.mj_step(model, data)
    assert eng.vx == 0.0 and eng.vy == 0.0 and eng.w == 0.0, \
        "arm-region compliance must hold the base still"
    eng.exit()


@needs_model
def test_engine_exit_restores_servos(fresh):
    model, data = fresh()
    gains0 = model.actuator_gainprm.copy()
    eng = WBCEngine(model, data)
    eng.enter("compliance", "upper_body")
    assert not np.allclose(model.actuator_gainprm, gains0)
    eng.exit()
    assert np.allclose(model.actuator_gainprm, gains0), "exit() must restore servos"
