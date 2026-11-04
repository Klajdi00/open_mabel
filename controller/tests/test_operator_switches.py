"""The two operator switches: gravity compensation, and which retargeting runs.

Operator, 2026-08-23: a toggle for gravity comp, and a button for whole-body
retargeting — ON = the original retargeting, OFF = the wrist-focused one.

Both are real control-law changes rather than UI state, so they are tested where
the law lives. The properties that matter:

  * they DO something (a disabled feed-forward stops feeding; a switched profile
    actually changes the constants the solver runs);
  * they are REVERSIBLE and round-trip exactly, because an operator will flip
    them back and forth mid-session;
  * flipping them does not JUMP the arm — the whole point of switching in place
    rather than rebuilding the retargeter;
  * the defaults are the safe/better answer.
"""
import numpy as np
import mujoco
import pytest

from mabel.facade import MabelTeleop, _DEFAULT_SPEC
from mabel.teleop_engine.hyperparams import (XML_FILE, ARM_W_ORIENT, ARM_W_ELBOW,
                                             ARM_J7_YAW_LIM, ARM_TRUST_RATE_DISTAL)
from mabel.dynamic_models.gravity import UpperBodyGravityComp

SIDES = ("right", "left")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML_FILE)


@pytest.fixture
def teleop(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.nkey - 1)
    mujoco.mj_forward(model, d)
    return MabelTeleop(model, d)


# ── defaults ────────────────────────────────────────────────────────────────
def test_defaults_are_the_safe_answer(teleop):
    """Compensated, and running the retargeting that measures better on the
    wrist. A toggle should never default to the worse of its two states."""
    assert teleop.spec["gravity_comp"] is True
    assert teleop.spec["whole_body_retarget"] is False
    assert teleop.controller.gravcomp.enabled
    assert teleop.controller.arm["right"].profile == "wrist"


def test_default_spec_declares_both(model):
    for k in ("gravity_comp", "whole_body_retarget"):
        assert k in _DEFAULT_SPEC, f"{k} missing from the facade default spec"


# ── gravity compensation ────────────────────────────────────────────────────
def test_gravity_switch_actually_stops_the_feed_forward(model):
    """OFF must mean no torque, not a smaller one."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.nkey - 1)
    mujoco.mj_forward(model, d)
    g = UpperBodyGravityComp(model)
    g.apply(d, 1.0, clear=True, force=True)
    on = d.qfrc_applied[g.dofs].copy()
    assert np.abs(on).max() > 0.1, "the layer fed nothing with gravity comp ON"

    g.set_enabled(False)
    g.apply(d, 2.0, clear=True, force=True)
    assert np.abs(d.qfrc_applied[g.dofs]).max() == 0.0

    g.set_enabled(True)
    g.apply(d, 3.0, clear=True, force=True)
    assert np.allclose(d.qfrc_applied[g.dofs], on, atol=1e-9), "gain not restored"


def test_gravity_off_still_clears_the_buffer(model):
    """OFF is implemented as gain -> 0 rather than by skipping apply(), because
    apply(clear=True) is the sole writer of qfrc_applied on the kinematic path.
    Skipping it would leave a stale torque from a just-exited torque mode."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.nkey - 1)
    mujoco.mj_forward(model, d)
    g = UpperBodyGravityComp(model)
    d.qfrc_applied[:] = 7.0                       # residue from another law
    g.set_enabled(False)
    g.apply(d, 1.0, clear=True, force=True)
    assert np.abs(d.qfrc_applied).max() == 0.0, "stale torque survived"


def test_gravity_switch_makes_the_arm_droop(teleop, model):
    """The observable consequence: uncompensated, the arm sags under its own
    weight. This is the whole reason the toggle is interesting."""
    d = teleop.data
    j = model.joint("Left Arm 2")
    qadr = model.jnt_qposadr[j.id]

    def settle(seconds=2.0):
        for _ in range(int(seconds / model.opt.timestep)):
            teleop.controller.gravcomp.apply(d, 0.0, clear=True, force=True)
            mujoco.mj_step(model, d)
        return float(d.qpos[qadr])

    d.ctrl[:] = 0.0
    d.ctrl[model.actuator("Lift").id] = 0.3175
    d.ctrl[model.actuator("Left Arm 2").id] = 0.9      # hold the arm out
    teleop.controller.set_gravity_comp(True)
    held = settle()
    teleop.controller.set_gravity_comp(False)
    sagged = settle()
    assert abs(sagged - 0.9) > abs(held - 0.9), (
        f"no droop appeared with gravity comp off "
        f"(held {held:.4f}, sagged {sagged:.4f}, target 0.900)")


# ── the retargeting profile ─────────────────────────────────────────────────
def test_profiles_differ_in_every_constant_that_defines_them(teleop):
    """The switch must move the actual solver constants, not just a label."""
    arm = teleop.controller.arm["right"]
    teleop.controller.set_whole_body_retarget(True)
    orig = dict(table=arm.table is not None, w_orient=arm.w_orient,
                w_fwd=arm.w_orient_fwd, w_elbow=arm.w_elbow,
                distal=arm.trust_rate_distal, j7=float(arm.hi[6]))
    teleop.controller.set_whole_body_retarget(False)
    new = dict(table=arm.table is not None, w_orient=arm.w_orient,
               w_fwd=arm.w_orient_fwd, w_elbow=arm.w_elbow,
               distal=arm.trust_rate_distal, j7=float(arm.hi[6]))
    for k in orig:
        assert orig[k] != new[k], f"{k} is the same in both profiles ({orig[k]})"
    # ON is the ORIGINAL retargeting: the pre-2026-08-23 constants, exactly.
    assert orig["table"] is False
    assert (orig["w_orient"], orig["w_fwd"], orig["w_elbow"]) == (20.0, 20.0, 30.0)
    assert orig["distal"] == 2.0
    assert orig["j7"] == pytest.approx(0.45, abs=1e-9)
    # OFF is the shipped wrist-focused tuning.
    assert new["table"] is True
    assert (new["w_orient"], new["w_fwd"], new["w_elbow"]) == \
        (ARM_W_ORIENT, pytest.approx(12.0), ARM_W_ELBOW)
    assert new["distal"] == ARM_TRUST_RATE_DISTAL
    assert new["j7"] == pytest.approx(ARM_J7_YAW_LIM, abs=1e-9)


def test_profile_round_trips(teleop):
    arm = teleop.controller.arm["left"]
    def snap():
        return (arm.table is not None, arm.w_orient, arm.w_orient_fwd, arm.w_elbow,
                arm.trust_rate_distal, tuple(np.round(arm.lo, 9)),
                tuple(np.round(arm.hi, 9)))
    a = snap()
    teleop.controller.set_whole_body_retarget(True)
    teleop.controller.set_whole_body_retarget(False)
    assert snap() == a, "a there-and-back switch did not restore the profile"


def test_j7_clamp_never_ratchets(teleop):
    """The clamp is recomputed from the MODEL range each time. Narrowing the
    PREVIOUS bound instead would shrink it a little on every switch until the
    joint was pinned — the kind of bug that only shows after a long session."""
    arm = teleop.controller.arm["right"]
    for _ in range(12):
        teleop.controller.set_whole_body_retarget(True)
        teleop.controller.set_whole_body_retarget(False)
    assert float(arm.hi[6]) == pytest.approx(ARM_J7_YAW_LIM, abs=1e-9)
    assert float(arm.lo[6]) == pytest.approx(-ARM_J7_YAW_LIM, abs=1e-9)


def test_both_arms_switch_together(teleop):
    teleop.controller.set_whole_body_retarget(True)
    assert {teleop.controller.arm[s].profile for s in SIDES} == {"whole_body"}
    teleop.controller.set_whole_body_retarget(False)
    assert {teleop.controller.arm[s].profile for s in SIDES} == {"wrist"}


def test_switching_does_not_jump_the_arm(teleop, model):
    """Switching profiles mid-session must not step the command. The trust
    region walks the arm over; the switch itself preserves last_q."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from exp_front_workspace import make_grid, _hk_world, _r_rob, FRAME_HZ
    from mabel.teleop_engine.workspace_table import ArmChain, palm_target, PALM_ROLLS

    d = teleop.data
    ch = ArmChain(model, "right")
    ch.sync(d)
    g = make_grid("right")
    p_s = g.point(2, 2, 3)
    R_s = palm_target(g.canonical_fwd(p_s), PALM_ROLLS["down"], "right")
    hk, _, _ = _hk_world(ch, p_s, R_s)
    rt = teleop.controller.arm["right"]
    rt.sync(d)
    for _ in range(90):                       # settle on the target
        rt.solve(hk, dt=1.0 / FRAME_HZ)
    before = rt.ik.qpos[rt.opt_qadr].copy()
    teleop.controller.set_whole_body_retarget(True)
    after = rt.solve(hk, dt=1.0 / FRAME_HZ)
    step = float(np.max(np.abs(after[:len(before) - 1] - before[:len(before) - 1])))
    # one frame of the trust region, with headroom — NOT a jump to the new optimum
    assert step < 0.30, f"the profile switch stepped the arm {step:.3f} rad"


# ── the facade seam the wire drives ─────────────────────────────────────────
def test_facade_accepts_the_bridge_splat(teleop):
    """The bridge does `set_control_spec(**spec)`, so the spec's KEYS have to be
    this function's parameter names. Mismatched case would be a TypeError at
    runtime on the first toggle."""
    spec = dict(_DEFAULT_SPEC)
    spec.update(gravity_comp=False, whole_body_retarget=True)
    teleop.set_control_spec(**spec)
    assert teleop.controller.gravcomp.enabled is False
    assert teleop.controller.arm["right"].profile == "whole_body"
    spec.update(gravity_comp=True, whole_body_retarget=False)
    teleop.set_control_spec(**spec)
    assert teleop.controller.gravcomp.enabled is True
    assert teleop.controller.arm["right"].profile == "wrist"


def test_switches_survive_an_unrelated_spec_change(teleop):
    """Changing the control space must not silently reset the operator's
    toggles — they are orthogonal axes."""
    teleop.set_control_spec(gravity_comp=False, whole_body_retarget=True)
    teleop.set_control_spec(method="retarget", region="upper_body")
    assert teleop.spec["gravity_comp"] is False
    assert teleop.spec["whole_body_retarget"] is True
    assert teleop.controller.gravcomp.enabled is False
    assert teleop.controller.arm["right"].profile == "whole_body"


@pytest.mark.slow
def test_each_profile_does_what_its_label_claims(teleop, model):
    """The toggle is only honest if each side wins on the axis it advertises:
    whole-body should track the operator's ELBOW better, and the wrist profile
    should place the PALM better. Measured on the same grid with a REALISTIC
    operator elbow (swivelled 35 deg off the relaxed prior — the case the elbow
    hint exists for). If either loses on its own axis, the label is wrong."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    from exp_front_workspace import (make_grid, _hk_world, SIDES, FRAME_HZ,
                                     SETTLE_FRAMES, _pose_error)
    from mabel.teleop_engine.workspace_table import (
        ArmChain, palm_target, natural_elbow, rot_axis, PALM_ORDER, PALM_ROLLS)

    d = teleop.data
    res = {}
    for profile in ("whole_body", "wrist"):
        teleop.controller.set_whole_body_retarget(profile == "whole_body")
        for side in SIDES:
            ch = ArmChain(model, side)
            ch.sync(d)
            grid = make_grid(side)
            rt = teleop.controller.arm[side]
            tilt, elb = [], []
            for i in range(0, grid.shape[0], 2):
                for k in range(0, grid.shape[2], 3):
                    p_s = grid.point(i, grid.shape[1] // 2, k)
                    e_nat = natural_elbow(np.zeros(3), p_s, ch.l_upper, ch.l_fore, side)
                    if e_nat is None:
                        continue
                    u = p_s / (np.linalg.norm(p_s) + 1e-12)
                    e_s = rot_axis(u, np.radians(35.0)) @ e_nat
                    e_w = ch.frame.to_world_point(e_s, ch.mount_pos, ch.mount_R)
                    for name in PALM_ORDER:
                        R_s = palm_target(grid.canonical_fwd(p_s), PALM_ROLLS[name], side)
                        hk, _, _ = _hk_world(ch, p_s, R_s, natural=False)
                        hk["elbow"] = e_w
                        rt.sync(d)
                        for _ in range(SETTLE_FRAMES):
                            rt.solve(hk, dt=1.0 / FRAME_HZ)
                        q = rt.ik.qpos[rt.opt_qadr].copy()
                        _, et, _ = _pose_error(ch, q, p_s, R_s)
                        ch.set_q(q)
                        tilt.append(np.degrees(et))
                        elb.append(float(np.linalg.norm(ch.elbow_s() - e_s)))
            res[(profile, side)] = (float(np.percentile(tilt, 90)), float(np.median(elb)))

    for side in SIDES:
        wb_tilt, wb_elb = res[("whole_body", side)]
        wr_tilt, wr_elb = res[("wrist", side)]
        assert wb_elb < wr_elb, (
            f"{side}: 'whole-body' does not track the elbow better "
            f"({wb_elb*1000:.0f} vs {wr_elb*1000:.0f} mm) — the label is wrong")
        assert wr_tilt < wb_tilt, (
            f"{side}: the wrist profile does not place the palm better "
            f"({wr_tilt:.1f} vs {wb_tilt:.1f} deg) — the label is wrong")


def test_status_reports_both(teleop):
    st = teleop.controller.status()
    assert st["gravity_comp"] is True and st["whole_body_retarget"] is False
    teleop.controller.set_gravity_comp(False)
    teleop.controller.set_whole_body_retarget(True)
    st = teleop.controller.status()
    assert st["gravity_comp"] is False and st["whole_body_retarget"] is True
