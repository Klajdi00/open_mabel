"""The ablation switches (neck / torso / lift frozen) are PARKS on top of ONE
retargeter, not a different one (2026-10-16). Proven structurally, because the
end-to-end arm-range statistic varies ±30° between identical runs of the
wall-clocked harness (exp_ablation_toggles.py) and cannot resolve it:

  1. each park writes ONLY its own actuators — disjoint from every arm, wrist
     and finger actuator;
  2. from an IDENTICAL state, one engine tick gives bit-identical ARM commands
     whether neck/torso/lift are live or frozen.
"""
import copy

import mujoco
import numpy as np
import pytest

from conftest import MODEL_XML, needs_model


def _engine():
    from mabel.facade import MabelTeleop
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, MabelTeleop(model, data)


@needs_model
def test_parks_touch_only_their_own_actuators():
    _m, _d, tele = _engine()
    eng = tele.controller
    arm = {int(a) for s in ("right", "left") for a in eng.arm_act[s]}
    arm |= {int(eng.wrist_act[s]) for s in ("right", "left")}
    fingers = {int(a) for s in ("right", "left") for a in eng.finger_act[s]}
    assert set(eng.park["neck"].aids.tolist()) == {int(a) for a in eng.neck_act}
    assert set(eng.park["torso"].aids.tolist()) == {int(eng.torso_act)}
    assert set(eng.park["lift"].aids.tolist()) == {int(eng.lift_act)}
    for name in ("neck", "torso", "lift"):
        aids = set(eng.park[name].aids.tolist())
        assert not (aids & arm), f"{name} park would write an ARM actuator"
        assert not (aids & fingers), f"{name} park would write a FINGER actuator"


@needs_model
def test_one_tick_from_identical_state_gives_identical_arm_commands():
    model, data, tele = _engine()
    eng = tele.controller
    arm = [int(a) for s in ("right", "left") for a in eng.arm_act[s]] + \
          [int(eng.wrist_act[s]) for s in ("right", "left")]
    # Two engines in the SAME state; apply only the parks (the part the
    # switches add), on top of an identical live command.
    d_live, d_frz = copy.deepcopy(data), copy.deepcopy(data)
    rng = np.random.default_rng(0)
    live_cmd = data.ctrl + rng.normal(0, 0.05, size=data.ctrl.shape)
    d_live.ctrl[:] = live_cmd
    d_frz.ctrl[:] = live_cmd
    eng.track_neck = eng.track_torso = eng.track_lift = True
    for p in ("neck", "torso", "lift"):
        eng.park[p].update(d_live, True, 0.01)
    for p in ("neck", "torso", "lift"):
        eng.park[p].update(d_frz, False, 0.01)
    assert np.array_equal(d_live.ctrl[arm], d_frz.ctrl[arm]), \
        "freezing neck/torso/lift changed an ARM command"
    # …and the frozen parts DID move toward home (the switch is not a no-op)
    moved = [int(eng.torso_act), int(eng.lift_act)] + [int(a) for a in eng.neck_act]
    assert not np.allclose(d_live.ctrl[moved], d_frz.ctrl[moved])
