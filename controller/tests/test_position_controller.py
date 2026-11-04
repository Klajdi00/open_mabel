"""Day-1 PD position controller: step tracking with NO model knowledge, the
gravity-feedforward hook (Day-2) shrinking the droop, and the safety clamps."""
import numpy as np
import mujoco

from conftest import needs_model, settle

from mabel.plant import SimPlant
from mabel.dynamic_models import ArmsDynamicModel, region_joints
from mabel.whole_body_controller import JointPositionController

STEP_JOINT = "Left Arm 2"
STEP = 0.3            # rad


def _run(model, data, ctl, seconds):
    taus = []
    for _ in range(int(seconds / model.opt.timestep)):
        taus.append(ctl.update())
        mujoco.mj_step(model, data)
    return taus


@needs_model
def test_day1_pd_tracks_step_without_model(fresh):
    model, data = fresh()
    settle(model, data)
    plant = SimPlant(model, data)
    ctl = JointPositionController(plant, region_joints("arms"))  # servo-inherited gains
    q0 = plant.q(STEP_JOINT)
    ctl.set_targets({STEP_JOINT: q0 + STEP})
    _run(model, data, ctl, 2.0)
    err = abs(plant.q(STEP_JOINT) - (q0 + STEP))
    assert err < 0.15, f"Day-1 PD steady error {err:.3f} rad (gravity droop too large)"
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    # other joints must not have run away
    for j in region_joints("arms"):
        if j != STEP_JOINT:
            assert abs(plant.qd(j)) < 2.0, f"{j} oscillating"


@needs_model
def test_gravity_feedforward_shrinks_droop(fresh):
    # Same PD; hold the CURRENT pose. Without ff the arm droops under gravity,
    # with ff = gravity_ff the error shrinks by a large factor.
    def droop(feedforward):
        model, data = fresh()
        settle(model, data)
        plant = SimPlant(model, data)
        arms = ArmsDynamicModel(model, data)
        ctl = JointPositionController(
            plant, region_joints("arms"), kp=25.0, kd=2.0,
            feedforward=(arms.gravity_ff if feedforward else None))
        ctl.hold_current()
        q_des = dict(ctl.q_des)
        _run(model, data, ctl, 1.5)
        return max(abs(plant.q(j) - q_des[j]) for j in ctl.joints)

    d_plain = droop(False)
    d_ff = droop(True)
    assert d_ff < 0.05, f"PD+gravity_ff droop {d_ff:.3f} rad"
    assert d_ff < 0.5 * d_plain, f"feedforward did not help: {d_ff:.3f} vs {d_plain:.3f}"


@needs_model
def test_torque_limit_is_respected(fresh):
    model, data = fresh()
    settle(model, data)
    plant = SimPlant(model, data)
    lim = 5.0
    ctl = JointPositionController(plant, region_joints("arms"),
                                  kp=200.0, kd=2.0, tau_limit=lim)
    ctl.set_targets({STEP_JOINT: plant.q(STEP_JOINT) + 2.0})   # huge error
    taus = _run(model, data, ctl, 0.2)
    peak = max(abs(t) for tick in taus for t in tick.values())
    assert peak <= lim + 1e-9, f"tau clamp violated: {peak:.2f} > {lim}"


@needs_model
def test_targets_ignore_unmanaged_joints(fresh):
    model, data = fresh()
    plant = SimPlant(model, data)
    ctl = JointPositionController(plant, ["Left Arm 1", "Left Arm 2"])
    ctl.set_targets({"Right Arm 4": 1.0, "Left Arm 1": 0.2})
    assert "Right Arm 4" not in ctl.q_des
    assert ctl.q_des["Left Arm 1"] == 0.2
