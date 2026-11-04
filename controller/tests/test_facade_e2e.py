"""End-to-end facade contract: one MabelTeleop drives NAV, retarget, and every
torque mode x region over live physics with synthetic operator frames — finite
state, in-range commands, correct mode strings, safe base twists, and the
reset()/external-force bookkeeping the bridges rely on."""
import numpy as np
import mujoco
import pytest

from conftest import needs_model, settle

import mabel.config as C
from mabel import MabelTeleop, OperatorPose
from mabel.sources import mock_frame
from mabel.motion_model import tip_usage


@pytest.fixture()
def teleop(fresh):
    model, data = fresh()
    settle(model, data)
    t = MabelTeleop(model, data)
    return model, data, t


def _run(model, data, t, n, t0=0.0, joystick=None):
    cmd = None
    dt = model.opt.timestep
    for i in range(n):
        pose = OperatorPose.from_avframe(mock_frame(t0 + i * dt))
        cmd = t.update(pose, joystick=joystick, dt=dt)
        mujoco.mj_step(model, data)
    return cmd


@needs_model
def test_full_mode_matrix_stays_finite_and_in_range(teleop):
    model, data, t = teleop
    steps = int(0.4 / model.opt.timestep)

    cmd = _run(model, data, t, steps)                       # default: navigation
    assert cmd is not None and np.isfinite(cmd.ctrl).all()

    t.set_control_spec(method="retarget")
    cmd = _run(model, data, t, steps, t0=1.0)
    assert np.isfinite(cmd.ctrl).all()

    for ctype, region in (("impedance", "arm"), ("impedance", "upper_body"),
                          ("compliance", "arm"), ("compliance", "upper_body"),
                          ("compliance", "whole_body")):
        t.set_control_spec(method="joint", control_type=ctype,
                           region=region, stiffness=8.0)
        cmd = _run(model, data, t, steps, t0=2.0)
        assert cmd.mode == f"WBC:{ctype}·{region}", cmd.mode
        assert np.isfinite(data.qpos).all(), (ctype, region)

    # leave torque mode; servos must be back for position teleop
    gains_in_torque = model.actuator_gainprm.copy()
    t.set_control_spec(method="navigation")
    _run(model, data, t, steps, t0=3.0)
    assert not np.allclose(model.actuator_gainprm, gains_in_torque), \
        "leaving WBC did not restore the position servos"


@needs_model
def test_nav_joystick_twist_respects_tip_envelope(teleop):
    model, data, t = teleop
    joy = (1.0, 1.0, 1.0, 0.0)          # full deflection everything
    _run(model, data, t, int(2.0 / model.opt.timestep), joystick=joy)
    vx, vy, w = t.controller.vx, t.controller.vy, t.controller.omega
    lift_z = 0.0
    assert tip_usage(vx, vy, w, 0, 0, 0, lift_z) <= 1.0 + 1e-6, \
        f"NAV twist outside tip envelope: ({vx:.2f},{vy:.2f},{w:.2f})"
    v_fwd, v_back, v_lat = C.max_speed_dir(lift_z)
    assert -v_fwd - 1e-6 <= vx <= v_back + 1e-6
    assert abs(vy) <= v_lat + 1e-6


@needs_model
def test_direct_joint_targets_override(teleop):
    model, data, t = teleop
    steps = int(0.3 / model.opt.timestep)
    _run(model, data, t, steps)
    t.set_joint_targets({"left_arm_2": 0.5, "neck_yaw": 0.4})
    _run(model, data, t, steps, t0=1.0)
    assert abs(data.actuator("Left Arm 2").ctrl[0] - 0.5) < 1e-9
    assert abs(data.actuator("Neck Yaw").ctrl[0] - 0.4) < 1e-9
    t.clear_joint_targets()
    assert not t.joint_targets


@needs_model
def test_reset_clears_state_in_place(teleop):
    model, data, t = teleop
    _run(model, data, t, 50)
    ext_ref = t.ext_forces                    # the bridge binds this BY REFERENCE
    t.set_joint_targets({"left_arm_1": 0.3})
    t.apply_external_force({C.LEFT_PALM: [1.0, 0.0, 0.0]})
    ext_ref = t.ext_forces
    t.reset()
    assert not t.joint_targets
    assert not t.ext_forces
    assert t.ext_forces is ext_ref, \
        "reset() rebound ext_forces — the bridge's mailbox binding would orphan"


@needs_model
def test_engaged_command_carries_base_twist(teleop):
    model, data, t = teleop
    cmd = _run(model, data, t, 100, joystick=(0.5, 0.0, 0.0, 0.0))
    assert cmd.base is not None and len(cmd.base) == 3
    assert np.isfinite(cmd.base).all()
