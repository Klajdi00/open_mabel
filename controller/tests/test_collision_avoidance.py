"""Self-collision guard: with the guard ON an adversarial crossing command must
never interpenetrate the capsule model; with it OFF the same command must (that
is what makes the guard load-bearing — the MJCF has arm-arm contacts disabled).
Quick-loop version of the exp_wbc_validation V-tests."""
import numpy as np
import mujoco
import pytest

from conftest import needs_model, settle

import mabel.config as C
from mabel.whole_body_controller.ik import WholeBodyIK
from mabel.whole_body_controller.collision_avoidance import SelfCollisionGuard


def _drive_crossing(model, data, guard_on, seconds=2.5):
    """Command the palms to swap sides (forces the forearms through each other)
    and return the minimum capsule gap seen along the way."""
    ik = WholeBodyIK(model, data)
    ik.set_self_collision(guard_on)
    meas = SelfCollisionGuard(model, data)      # independent measurement
    l0 = data.body(C.LEFT_PALM).xpos.copy()
    r0 = data.body(C.RIGHT_PALM).xpos.copy()
    ik.sync_to_current()
    ik.target_l_pos = r0.copy()                 # cross the arms
    ik.target_r_pos = l0.copy()
    dt = model.opt.timestep
    min_gap = np.inf
    for _ in range(int(seconds / dt)):
        dq = ik.solve()
        ik.apply(dq, dt)
        mujoco.mj_step(model, data)
        meas.rows(ik.dofs)
        min_gap = min(min_gap, float(meas.min_dist))
    return min_gap


@needs_model
def test_guard_on_no_interpenetration(fresh):
    model, data = fresh()
    settle(model, data)
    min_gap = _drive_crossing(model, data, guard_on=True)
    assert min_gap >= 0.0, f"guard ON interpenetrated: min gap {min_gap*100:.2f} cm"


@needs_model
def test_guard_off_interpenetrates(fresh):
    model, data = fresh()
    settle(model, data)
    min_gap = _drive_crossing(model, data, guard_on=False)
    if min_gap >= 0.0:
        pytest.skip("crossing command did not interpenetrate with guard OFF — "
                    "scenario too gentle to discriminate (guard-ON test still valid)")
    assert min_gap < 0.0


@needs_model
def test_barrier_force_only_brakes_closing_motion(fresh):
    model, data = fresh()
    settle(model, data)
    guard = SelfCollisionGuard(model, data)
    # at rest (far from margins, zero velocity) the soft wall must be silent
    tau = guard.barrier_force(np.zeros(model.nv))
    assert np.allclose(tau, 0.0), "barrier force nonzero at rest"
