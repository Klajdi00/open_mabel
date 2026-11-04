"""Region dynamic models: joint sets, gravity feedforward really holds the
robot in simulation, the region mass matrix is sane, and the momentum observer
recovers an applied external push (direction + rough magnitude + side)."""
import numpy as np
import pytest
import mujoco

from conftest import needs_model, settle, arm_qpos

from mabel.dynamic_models import (ArmsDynamicModel, UpperBodyDynamicModel,
                                  WholeBodyDynamicModel, region_joints,
                                  COMPLIANT_ARM_JOINTS)
from mabel.plant import SimPlant
from mabel.whole_body_controller.gravity_compensation import GravityCompController


def test_region_joint_sets():
    arms = region_joints("arms")
    upper = region_joints("upper_body")
    whole = region_joints("whole_body")
    assert len(arms) == 16 and set(arms) == set(COMPLIANT_ARM_JOINTS)
    assert len(upper) == 17 and "Torso" in upper
    assert whole == upper                       # base adds no torque joints
    assert region_joints("arm") == arms         # wire-protocol alias
    with pytest.raises(ValueError):
        region_joints("legs")


@needs_model
def test_region_flags_and_mass_matrix(fresh):
    model, data = fresh()
    arms = ArmsDynamicModel(model, data)
    upper = UpperBodyDynamicModel(model, data)
    whole = WholeBodyDynamicModel(model, data)
    assert not arms.includes_lift and not arms.includes_base
    assert upper.includes_lift and not upper.includes_base
    assert whole.includes_lift and whole.includes_base
    M = arms.mass_matrix()
    assert M.shape == (16, 16)
    assert np.allclose(M, M.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(M) > 0)


@needs_model
def test_gravity_ff_holds_arms_and_no_comp_sags(fresh):
    dt_run = 1.5    # seconds of physics per condition

    # Condition A: servos off + gravity-comp feedforward -> the arm HOLDS.
    model, data = fresh()
    settle(model, data)
    arms = ArmsDynamicModel(model, data)
    ctl = GravityCompController(arms, SimPlant(model, data))
    q0 = arm_qpos(model, data, arms.joints)
    for _ in range(int(dt_run / model.opt.timestep)):
        ctl.update()
        mujoco.mj_step(model, data)
    drift_comp = np.abs(arm_qpos(model, data, arms.joints) - q0).max()

    # Condition B: servos off, NO feedforward -> the arm sags under gravity.
    model, data = fresh()
    settle(model, data)
    arms_b = ArmsDynamicModel(model, data)
    plant = SimPlant(model, data)
    for j in arms_b.joints:
        plant.disable_servo(j)
    q0 = arm_qpos(model, data, arms_b.joints)
    for _ in range(int(dt_run / model.opt.timestep)):
        plant.begin()
        mujoco.mj_step(model, data)
    drift_sag = np.abs(arm_qpos(model, data, arms_b.joints) - q0).max()

    assert drift_comp < 0.15, f"gravity comp failed to hold: {drift_comp:.3f} rad"
    assert drift_sag > 2 * drift_comp and drift_sag > 0.3, \
        f"control condition did not sag ({drift_sag:.3f} rad) — test not discriminating"


@needs_model
def test_upper_body_region_holds_lift(fresh):
    model, data = fresh()
    settle(model, data)
    upper = UpperBodyDynamicModel(model, data)
    ctl = GravityCompController(upper, SimPlant(model, data))
    import mabel.config as C
    lift0 = data.joint(C.LIFT_JOINT).qpos[0]
    for _ in range(int(1.5 / model.opt.timestep)):
        ctl.update()
        mujoco.mj_step(model, data)
    lift_drift = abs(data.joint(C.LIFT_JOINT).qpos[0] - lift0)
    assert lift_drift < 0.05, f"gravity-comp'd lift drifted {lift_drift:.3f} m"


@needs_model
def test_observer_recovers_external_push(fresh):
    model, data = fresh()
    settle(model, data)
    arms = ArmsDynamicModel(model, data)
    ctl = GravityCompController(arms, SimPlant(model, data))

    F = np.array([12.0, 0.0, 0.0])          # world-frame push on the left palm
    import mabel.config as C
    bid = model.body(C.LEFT_PALM).id
    dt = model.opt.timestep
    est = {}
    for i in range(int(1.0 / dt)):
        data.xfrc_applied[bid, :3] = F if i > int(0.3 / dt) else 0.0
        ctl.update()
        tau_cmd = data.qfrc_applied + data.qfrc_actuator
        est = arms.external_torques(tau_cmd, dt)
        mujoco.mj_step(model, data)

    # The push loads the LEFT arm's joints, not the right's.
    l_mag = max(abs(est[j]) for j in arms.joints if j.startswith("Left"))
    r_mag = max(abs(est[j]) for j in arms.joints if j.startswith("Right"))
    assert l_mag > 1.0, f"push not visible in residual (max {l_mag:.2f} Nm)"
    assert l_mag > 3 * r_mag, f"wrong side: L {l_mag:.2f} vs R {r_mag:.2f} Nm"


@needs_model
def test_whole_body_base_wrench_and_contacts(fresh):
    model, data = fresh()
    settle(model, data)
    whole = WholeBodyDynamicModel(model, data)
    ctl = GravityCompController(UpperBodyDynamicModel(model, data),
                                SimPlant(model, data))
    import mabel.config as C
    bid = model.body(C.LEFT_PALM).id
    F = np.array([15.0, 0.0, 0.0])
    dt = model.opt.timestep
    fx = fy = mz = 0.0
    for i in range(int(1.0 / dt)):
        data.xfrc_applied[bid, :3] = F if i > int(0.3 / dt) else 0.0
        ctl.update()
        tau_cmd = data.qfrc_applied + data.qfrc_actuator
        fx, fy, mz = whole.base_wrench(tau_cmd, dt)
        mujoco.mj_step(model, data)

    assert np.isfinite([fx, fy, mz]).all()
    assert len(whole.contacts) >= 1, "contact localization found nothing"
    c = whole.contacts[0]
    assert c["joint"].startswith("Left"), f"localized on wrong arm: {c['joint']}"
    # Direction: estimated force should roughly align with the applied push.
    f_est = np.asarray(c["force"])
    cos = float(f_est @ F / (np.linalg.norm(f_est) * np.linalg.norm(F) + 1e-9))
    assert cos > 0.6, f"force direction off (cos {cos:.2f}); est {f_est}"
    # Magnitude: same order as applied.
    assert 0.25 * np.linalg.norm(F) < np.linalg.norm(f_est) < 4 * np.linalg.norm(F)
