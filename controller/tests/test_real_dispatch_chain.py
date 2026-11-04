"""Real-dispatch (real:ros) whole-body teleop chain — the regressions that made
VP whole-body teleop fail (or lurch) on the REAL robot while looking fine in sim.

Root causes pinned here (2026-07):

  * FROZEN ARMS — resume_hold (mandatory on real: the ENABLE-TORQUE edge) seeded
    arms/wrists/torso into joint_targets; _apply_joint_targets re-applied them
    AFTER the retargeter every tick, freezing the whole upper body. Old sim runs
    never noticed because the VP app's torque/e-stop wire never reached the
    server (client-side only until 2026-07-18).
  * HOME SNAP — EngagementSupervisor blended HOME <-> live by a global alpha, so
    at alpha=0 it WROTE ctrl := HOME outright: a full-authority position step
    whenever the robot was not at home (which on real is always). Now every
    transition eases from an ANCHOR (the last commanded pose).

The full chain (mock VP stream -> facade/TeleopEngine -> command_extract -> the
canonical hardware commands) is exercised end-to-end by
experiments/exp_real_retarget_chain.py, run here as one (slow, realtime-paced)
test; the units below cover the supervisor law in virtual time.
"""
import os
import sys

import numpy as np
import pytest

from conftest import needs_model

_EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)


def _sup(model):
    scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "simulation", "mabel_mujoco", "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from teleop.safety import EngagementSupervisor
    return EngagementSupervisor(model, model.actuator("Lift").id)


# ── EngagementSupervisor law (virtual time — fast) ───────────────────────────
@needs_model
def test_supervisor_transparent_until_first_engagement(fresh):
    """Fresh boot with NO operator EVER engaged: the supervisor must write NOTHING
    — ctrl (including the LIFT) holds wherever bring-up / user commands put it.
    Regression: "the lift moves by itself on real_run startup" (2026-07-27) — the
    disengaged-at-boot state eased the whole body to HOME at the retreat rate,
    walking the real lift to SAFE_HOME_LIFT at ~13 mm/s with nobody connected."""
    model, data = fresh()
    sup = _sup(model)
    aid = model.actuator("Left Arm 2").id
    lift = model.actuator("Lift").id
    data.ctrl[aid] = 0.8
    data.ctrl[lift] = 0.264
    now = 100.0
    for k in range(500):                       # 10 s — far past every ease/timeout
        sup.step(now + k * 0.02, 0.02)
        sup.apply(data)
    assert float(data.ctrl[aid]) == 0.8 and float(data.ctrl[lift]) == 0.264, \
        "supervisor moved the robot with no operator ever engaged"
    assert sup.state() == "standby (hold)"


@needs_model
def test_supervisor_never_snaps_to_home(fresh):
    """Tracking LOST with the commanded pose AWAY from home: apply() must ease, not
    write home (the real-robot enable lurch). Engagement is primed first — with no
    operator EVER engaged the supervisor is transparent (test above)."""
    model, data = fresh()
    sup = _sup(model)
    aid = model.actuator("Left Arm 2").id
    data.ctrl[aid] = 0.8                       # commanded pose far from home (0)
    now = 100.0
    sup.t_stream = sup.t_head = sup.t_hand = sup.t_arm = now   # prime: engaged once
    sup.step(now, 0.0)
    now += 1.0                                 # tracking lost (signals stale)
    first = None
    for k in range(50):                        # 1 s of 50 Hz virtual time
        sup.step(now + k * 0.02, 0.02)
        sup.apply(data)
        if first is None:
            first = float(data.ctrl[aid])
        assert abs(float(data.ctrl[aid]) - 0.8) <= 0.8 * (k + 1) * 0.02 / 4.0 + 1e-6, \
            "eases faster than the SAFE_RETREAT rate"
    assert abs(first - 0.8) < 0.01, f"first apply stepped to {first} (home snap)"
    assert float(data.ctrl[aid]) < 0.65, "no progress toward home in 1 s"


@needs_model
def test_supervisor_engage_eases_from_current_not_through_home(fresh):
    """Re-acquire with the body away from BOTH home and the live pose: the ease
    must run current -> live directly (the old alpha blend detoured via home)."""
    model, data = fresh()
    sup = _sup(model)
    aid = model.actuator("Left Arm 2").id
    i = list(sup.aids).index(aid)
    data.ctrl[aid] = 0.6                        # held pose (e.g. eased partway)
    now = 100.0
    sup.step(now, 0.02); sup.apply(data)        # disengaged tick: anchor at 0.6
    sup.live[i] = 0.9                           # operator pose to ease into
    sup.t_stream = sup.t_head = sup.t_hand = sup.t_arm = now   # signals fresh
    lo = hi = 0.6
    for k in range(200):                        # 4 s: full engage
        sup.step(now + k * 0.02, 0.02)
        sup.apply(data)
        v = float(data.ctrl[aid])
        lo, hi = min(lo, v), max(hi, v)
        sup.t_stream = sup.t_head = sup.t_hand = sup.t_arm = now + k * 0.02
    # lo tolerance: the single DISENGAGED setup tick legitimately eases ~1 mm
    # toward home before the signals freshen — the check is against a HOME
    # detour (0.6 -> ~0 -> 0.9), not that millimetre.
    assert hi <= 0.9 + 1e-6 and lo >= 0.6 - 0.01, \
        f"engage detoured outside [held, live] ([{lo:.3f}, {hi:.3f}])"
    assert abs(float(data.ctrl[aid]) - 0.9) < 1e-3, "did not reach the live pose"


@needs_model
def test_supervisor_rebase_adopts_held_pose(fresh):
    """rebase() (hold_current / resume_hold) must restart the ease at the ctrl
    that was just adopted — not keep blending from the stale anchor."""
    model, data = fresh()
    sup = _sup(model)
    aid = model.actuator("Left Arm 2").id
    now = 100.0
    data.ctrl[aid] = 0.8
    sup.t_stream = sup.t_head = sup.t_hand = sup.t_arm = now   # prime: engaged once
    sup.step(now, 0.0)
    now += 1.0                                 # tracking lost
    for k in range(25):                        # ease partway toward home
        sup.step(now + k * 0.02, 0.02)
        sup.apply(data)
    data.ctrl[aid] = 0.5                       # hold_current adopts the measured pose
    sup.rebase()
    sup.step(now + 26 * 0.02, 0.02)
    sup.apply(data)
    assert abs(float(data.ctrl[aid]) - 0.5) < 0.01, \
        "apply after rebase did not ease from the freshly-held ctrl"


# ── facade ownership in retarget (the frozen-arms regression) ────────────────
@needs_model
def test_resume_hold_retarget_keeps_engine_ownership(fresh):
    """In retarget, resume_hold must NOT seed arm/wrist/torso joint targets (the
    override would freeze the retargeter); in navigation it MUST (arms hold)."""
    import mujoco
    from mabel.facade import MabelTeleop
    model, data = fresh()
    t = MabelTeleop(model, data)
    t.controller.init_hold()
    mujoco.mj_forward(model, data)

    t.set_control_spec(method="retarget", control_type="impedance",
                       region="whole_body",
                       # P4 asserts that WALKING drives the base, which is
                       # GLOBAL-scope behaviour. LOCAL (the default since
                       # 2026-07-31) deliberately keeps the base out of the
                       # follow loop and leaves it to the joystick, so this
                       # chain has to ask for the scope it is measuring.
                       scope="global")
    t.update(None, None, 0.02)
    t.joint_targets["left_arm_2"] = 0.3        # stale slider from another mode
    t.resume_hold()
    frozen = [k for k in t.joint_targets
              if k.startswith(("left_arm", "right_arm"))
              or k in ("torso", "left_wrist", "right_wrist",
                       "lift_lower", "lift_upper")]
    assert not frozen, f"retarget resume seeded freezing targets: {frozen}"

    t.set_control_spec(method="navigation")
    t.update(None, None, 0.02)
    t.resume_hold()
    assert "left_arm_2" in t.joint_targets and "torso" in t.joint_targets, \
        "navigation resume must seed the hold targets"


# ── the full chain, end-to-end (realtime-paced, ~25 s) ───────────────────────
@needs_model
def test_real_dispatch_chain_end_to_end():
    """Mock VP stream -> facade (retarget) -> command_extract, through displaced
    torque-enable, engage, whole-body walk (base+lift), stream loss, re-acquire.
    Every published command step-bounded; every subsystem follows. Realtime-paced
    (the engine's watchdogs run on time.time())."""
    import exp_real_retarget_chain as chain
    ok, metrics = chain.run()
    failed = [(n, v) for n, v, o in metrics if not o]
    assert ok, f"real-dispatch chain checks failed: {failed}"
