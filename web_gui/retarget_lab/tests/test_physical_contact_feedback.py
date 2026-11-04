"""Hardware contact uses actual FK and never substitutes the virtual plant."""
import mujoco
import numpy as np
import pytest

from retarget_lab.server.worker import KinematicRetargeter
from mabel.whole_body_controller.collision_avoidance.capsule_guard import _seg_seg

START = np.array([1.1900482940786836, -.3631403540553816, -.669623839473904,
                  -.7212422641439105, -.7524352410318523, -.20003345823249818,
                  -1.2177241499911977, 0., -.7683729355076082, -.1722146879726992,
                  -.2550765689433324, .3017640374197829, -1.570796,
                  -.3586616442129984, -1.2361538247362494, 0.])


@pytest.fixture(scope="module")
def model():
    return KinematicRetargeter("shipped_v5").model


def rig(model, variant="shipped_v5"):
    k = KinematicRetargeter(variant, model=model)
    k.apply_settings({"robot": "mabel", "scope": "local", "ee": "gripper"})
    p = k.eng.projector
    p.enable_gripper_contact = k.eng.enable_gripper_contact
    p.handoff_alpha, p.handoff_separation = 1., np.zeros(3)
    p.active_sides = frozenset(("right", "left"))
    p.q_cmd = START.copy()
    k.data.qpos[p.qadr] = START
    mujoco.mj_forward(model, k.data)
    return k


def position(k, q=START):
    p = k.eng.projector
    return {f"{s}_arm_{i + 1}": float(q[index * p.split + i])
            for index, s in enumerate(("right", "left")) for i in range(7)}


def feedback(k, q=START, valid=True, reason="", torso=0., age_s=0., compute_budget_s=0.):
    k.set_physical_feedback(position(k, q) if valid else {}, valid=valid, reason=reason, age_s=age_s,
                            compute_budget_s=compute_budget_s)
    p, e = k.eng.projector, k.eng
    p.arm_feedback_required = e.arm_feedback_required
    p.arm_feedback_valid = e.arm_feedback_valid
    p.arm_feedback_reason = e.arm_feedback_reason
    p.arm_feedback_age_s = e.arm_feedback_age_s
    p.arm_feedback_compute_s = e.arm_feedback_compute_s
    p.arm_measured = e.arm_measured
    p.torso_measured = torso


def project(k, desired=START):
    p = k.eng.projector
    return p.project(desired[:7], desired[8:15], 1. / 60., wrist_r=0., wrist_l=0.)


def tip_gap(k, data):
    g = k.eng.projector.guard
    a, b = g.caps["r_grip_finger_tip"], g.caps["l_grip_finger_tip"]
    a0, a1 = a.world(data)
    b0, b1 = b.world(data)
    u, v = _seg_seg(a0, a1, b0, b1)
    return float(np.linalg.norm(u - v) - a.radius - b.radius)


def test_missing_physical_feedback_holds_exactly_and_reset_stays_required(model):
    k = rig(model)
    feedback(k, valid=False, reason="joint feedback stale")
    before = k.eng.projector.q_cmd.copy()
    project(k, START + .1)
    np.testing.assert_array_equal(k.eng.projector.q_cmd, before)
    assert k.eng.projector.physical_feedback_hold
    assert "stale" in k.eng.projector.contact_guard_reason
    k.reset()
    assert k.eng.arm_feedback_required
    assert not k.eng.arm_feedback_valid
    assert not k.eng.arm_measured


def test_partial_or_nonfinite_feedback_is_not_accepted(model):
    k = rig(model)
    for bad in ({}, {**position(k), "right_arm_7": float("nan")},
                {name: value for name, value in position(k).items() if name != "left_arm_1"}):
        k.set_physical_feedback(bad, valid=True, age_s=0.)
        assert k.eng.arm_feedback_required
        assert not k.eng.arm_feedback_valid
    k.set_physical_feedback(position(k), valid=True)
    assert not k.eng.arm_feedback_valid, "unknown source age must not claim fresh hardware feedback"


def test_measured_rows_use_actual_arms_and_torso_without_mutating_virtual_pose(model, monkeypatch):
    k = rig(model)
    q = START.copy()
    q[0], q[8] = q[0] - .15, q[8] + .15
    feedback(k, q, torso=.1)
    before = k.data.qpos.copy()
    p, seen = k.eng.projector, []
    original = p.guard.rows
    def record(*args, **kwargs):
        seen.append(p.guard.data.qpos.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(p.guard, "rows", record)
    project(k)
    assert len(seen) == 2
    np.testing.assert_array_equal(k.data.qpos, before)
    np.testing.assert_array_equal(seen[-1][k.eng.arm["right"].qadr], q[:7])
    np.testing.assert_array_equal(seen[-1][k.eng.arm["left"].qadr], q[8:15])
    torso = model.jnt_qposadr[model.joint("Torso").id]
    assert seen[-1][torso] == .1
    expected = mujoco.MjData(model)
    expected.qpos[:] = before
    expected.qpos[k.eng.arm["right"].qadr] = q[:7]
    expected.qpos[k.eng.arm["left"].qadr] = q[8:15]
    expected.qpos[torso] = .1
    mujoco.mj_kinematics(model, expected)
    np.testing.assert_allclose(p.actual.xpos, expected.xpos, atol=1e-13, rtol=0.)


def test_lagging_actual_fingers_prevent_further_command_closure(model):
    k = rig(model)
    p = k.eng.projector
    q = START.copy()
    q[0], q[8] = q[0] - .15, q[8] + .15
    feedback(k, q)
    p._end_effectors()
    before = tip_gap(k, k.data)
    desired = START.copy()
    desired[0], desired[8] = desired[0] + .1, desired[8] - .1
    project(k, desired)
    actual_gap = tip_gap(k, p.actual)
    assert actual_gap - before > .02   # reproduce a physical closing lag over 20 mm
    assert p.closing_lead > .02
    assert "waiting for measured" in p.contact_guard_reason
    assert tip_gap(k, p.scratch) >= before - 1e-9
    assert p.guard._margin_for(p.guard.caps["r_hand"], p.guard.caps["torso"]) == p.guard.d_safe


@pytest.mark.parametrize("age_s", [0., .02, .05, .1])
def test_fixed_measured_pose_cannot_exceed_the_shared_age_and_following_budget(model, age_s):
    k = rig(model)
    feedback(k, age_s=age_s)
    p = k.eng.projector
    desired = START.copy()
    desired[0], desired[8] = desired[0] + .2, desired[8] - .2
    for _ in range(100):
        before = p.q_cmd.copy()
        project(k, desired)
        assert np.max(np.abs(p.q_cmd - before)) <= .5 / 60. + 1e-8
        assert tip_gap(k, p.actual) - tip_gap(k, p.scratch) <= p.closing_lead_limit + 1e-8
        key = tuple(sorted(("r_grip_finger_tip", "l_grip_finger_tip")))
        measured_pad = p._measured_contact_margins[key] - .001 + .004
        assert p.closing_lead_limit + .03 * age_s + .002 <= measured_pad + 1e-12
        k.data.qpos[p.qadr] = p.q_cmd
        mujoco.mj_kinematics(model, k.data)
        mujoco.mj_comPos(model, k.data)


def test_recovered_feedback_keeps_command_step_bounded_and_missing_torso_holds(model):
    k = rig(model)
    feedback(k, torso=None)
    p = k.eng.projector
    before = p.q_cmd.copy()
    project(k, START + .1)
    np.testing.assert_array_equal(p.q_cmd, before)
    assert p.physical_feedback_hold
    feedback(k)
    project(k, START + .1)
    assert not p.physical_feedback_hold
    assert np.max(np.abs(p.q_cmd - before)) <= .5 / 60. + 1e-8


def test_v4_ignores_external_feedback_metadata(model):
    reference, tagged = rig(model, "shipped_v4"), rig(model, "shipped_v4")
    feedback(tagged, valid=False, reason="stale")
    for _ in range(20):
        for k in (reference, tagged):
            project(k, START + .02)
            p = k.eng.projector
            k.data.qpos[p.qadr] = p.q_cmd
            mujoco.mj_kinematics(model, k.data)
            mujoco.mj_comPos(model, k.data)
        np.testing.assert_array_equal(reference.eng.projector.q_cmd, tagged.eng.projector.q_cmd)


def _provider(k, q=START, age_s=.02):
    return lambda: {"position": position(k, q), "valid": True, "reason": "", "age_s": age_s, "torso": 0.}


def _warm(k):
    from retarget_lab.tests.test_gripper_handoff import _pose
    for i in range(30):
        k.step(_pose(i / 60., (0., 1.32, -.42)), i / 60.)
    return _pose


def test_provider_refreshes_after_both_ik_solves_and_budgets_only_the_late_phase(model, monkeypatch):
    k = rig(model)
    pose = _warm(k)
    clock, completed, seen = [10.], [], []
    k._physical_now = lambda: clock[0]
    original = k.eng._solve_side
    actual = START.copy()
    def solve(side, *args):
        result = original(side, *args)
        completed.append(side)
        clock[0] += .04  # deliberately slow IK must precede the late snapshot
        return result
    def provider():
        seen.append(tuple(sorted(completed)))
        return _provider(k, actual)()
    monkeypatch.setattr(k.eng, "_solve_side", solve)
    k.set_physical_feedback_provider(provider, compute_budget_s=.025, publication_reserve_s=.005)
    record = k.step(pose(.5, (0., 1.32, -.42)), .5)
    assert seen == [("left", "right")]
    assert k._physical_phase_start == pytest.approx(10.08)
    assert k.physical_command_deadline == pytest.approx(10.105)
    assert not record["physical_contact"]["hold"]
    assert record["physical_contact"]["effective_age_ms"] == pytest.approx(45.)
    np.testing.assert_array_equal(k.eng.projector.actual.qpos[k.eng.arm["right"].qadr], actual[:7])


def test_publication_recheck_rolls_back_an_earlier_acceptance_exactly(model):
    k = rig(model)
    clock = [10.]
    k._physical_now = lambda: clock[0]
    k.set_physical_feedback_provider(_provider(k))
    before = k._physical_state()
    k._begin_physical_command()
    k._refresh_physical_feedback()
    p = k.eng.projector
    p.q_cmd += .001
    k.data.qpos[p.qadr] += .001
    acts = np.r_[k.eng.arm_act["right"], [k.eng.wrist_act["right"]],
                 k.eng.arm_act["left"], [k.eng.wrist_act["left"]]].astype(int)
    k.data.ctrl[acts] += .001
    clock[0] += .01
    assert k.accept_physical_command()
    clock[0] += .011
    assert not k.accept_physical_command()
    for restored, old in zip(k._physical_state(), before):
        np.testing.assert_array_equal(restored, old)
    assert "deadline exceeded" in p.contact_guard_reason
    assert not k.accept_physical_command(), "a rejected frame cannot become eligible again"


def test_overrun_after_recording_restores_the_previous_published_pose(model, monkeypatch):
    k = rig(model)
    pose = _warm(k)
    clock = [10.]
    k._physical_now = lambda: clock[0]
    k.set_physical_feedback_provider(_provider(k))
    before = k._physical_state()
    original = k._record
    def slow_record(*args):
        record = original(*args)
        clock[0] += .03
        return record
    monkeypatch.setattr(k, "_record", slow_record)
    record = k.step(pose(.5, (0., 1.32, -.42)), .5, defer_physical_accept=True)
    assert record["physical_contact"]["publication_rejected"]
    assert record["physical_contact"]["hold"]
    for restored, old in zip(k._physical_state(), before):
        np.testing.assert_array_equal(restored, old)
    for side in ("right", "left"):
        for i in range(1, 8):
            name = f"{side}_arm_{i}"
            assert record["q"][name] == k.data.qpos[k.qnames[name]]


def test_provider_failure_and_stale_age_plus_compute_budget_hold(model):
    k = rig(model)
    pose = _warm(k)
    for provider in (lambda: None, _provider(k, age_s=.08)):
        k.set_physical_feedback_provider(provider, compute_budget_s=.025)
        before = k._physical_state()
        record = k.step(pose(.5, (0., 1.32, -.42)), .5)
        assert record["physical_contact"]["hold"]
        assert not k.eng.arm_feedback_valid
        for restored, old in zip(k._physical_state(), before):
            np.testing.assert_array_equal(restored, old)
    k.reset()
    assert k._physical_provider is not None
    assert k.eng.arm_feedback_required
    assert not k.eng.arm_feedback_valid


def test_discarded_published_packets_cannot_ratchet_closure_without_actual_motion(model):
    k = rig(model)
    actual = START.copy()
    actual[0], actual[8] = actual[0] - .15, actual[8] + .15
    p = k.eng.projector
    p.q_cmd = actual.copy()
    k.data.qpos[p.qadr] = actual
    mujoco.mj_forward(model, k.data)
    for _ in range(100):
        # The child accepts/publishes each command, but an expired receiver
        # discards it: physical feedback deliberately remains at the same pose.
        feedback(k, actual, age_s=.02, compute_budget_s=.025)
        k._begin_physical_command()
        project(k, START)
        k.data.qpos[p.qadr] = p.q_cmd
        mujoco.mj_kinematics(model, k.data)
        mujoco.mj_comPos(model, k.data)
        assert k.accept_physical_command()
        assert p.closing_lead_limit == pytest.approx(.00065)
        assert tip_gap(k, p.actual) - tip_gap(k, p.scratch) <= .00065 + 1e-8


def test_v4_never_consumes_late_provider_or_deadline(model):
    k = rig(model, "shipped_v4")
    def forbidden():
        raise AssertionError("old variants must not consume physical provider")
    k.set_physical_feedback_provider(forbidden)
    assert not k.eng.arm_feedback_required
    pose = _warm(k)
    k.step(pose(.5, (0., 1.32, -.42)), .5, defer_physical_accept=True)
    assert k.accept_physical_command()
    assert k.physical_command_deadline is None
