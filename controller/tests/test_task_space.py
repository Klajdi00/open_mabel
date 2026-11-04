"""Task-space whole-body reach: the Sync-Lift-aware IK, the BaseFollow
follower, the CoM stability guard, and the lift-units hold/resume fixes.

Background (Control Lab bench, 2026-07): in the wrist / task-space method the
lift never moved (its IK target was dropped before the wire) and the base never
moved (nothing mapped palm targets to a twist). Fixing those surfaced three
more library bugs — the tendon-cascaded Lift ("Sync Lift": Upper == Lower,
ctrl = stage SUM) was commanded with single-stage joint units in
TeleopController.hold_current and read through a bogus tendon-id-as-joint-id
lookup in MabelTeleop._measured_ctrl, and the 18-DOF IK both planned and
integrated the lift in inconsistent units. The uncapped torso slam also
face-planted the robot chasing far targets (static margin is only ~0.094 m).
"""
import math

import numpy as np
import pytest

from conftest import needs_model


@pytest.fixture()
def wik(fresh):
    """A WholeBodyIK bound to a private model+data (kinematic planning only)."""
    import mujoco
    from mabel.whole_body_controller import WholeBodyIK
    model, data = fresh()
    ik = WholeBodyIK(model, data)
    return model, data, ik


def _plan(model, data, ik, seconds, dt=0.008, arms_only=False):
    """Kinematic planning loop, the way the Control Lab drives it: solve →
    apply → write the integrated targets back into qpos (both lift stages)."""
    import mujoco
    for _ in range(int(seconds / dt)):
        dq = ik.solve(arms_only=arms_only)
        ik.apply(dq, dt, arms_only=arms_only)
        for name, val in ik.target_q.items():
            joints = (("Lift Lower", "Lift Upper") if name == "Lift" else (name,))
            for j in joints:
                data.qpos[model.joint(j).qposadr[0]] = val
        mujoco.mj_kinematics(model, data)


# ── Sync-Lift units in the 18-DOF IK ─────────────────────────────────────────
@needs_model
def test_ik_recruits_lift_for_high_targets(wik):
    """A target straight above the palms must extend the lift (18-DOF solve),
    with the target held in JOINT units inside the stage joint range and the
    tendon ctrl written as the 2x stage sum."""
    model, data, ik = wik
    lift0 = ik.target_q["Lift"]
    ik.target_l_pos = ik.target_l_pos + np.array([0.0, 0.0, 0.5])
    ik.target_r_pos = ik.target_r_pos + np.array([0.0, 0.0, 0.5])
    _plan(model, data, ik, seconds=6.0)
    lo, hi = model.joint("Lift Lower").range
    tq = ik.target_q["Lift"]
    assert tq - lift0 > 0.05, f"lift did not extend (Δ={tq - lift0:.4f})"
    assert lo - 1e-9 <= tq <= hi + 1e-9, f"lift target left the stage joint range ({tq})"
    assert data.actuator("Lift").ctrl[0] == pytest.approx(2.0 * tq), \
        "tendon ctrl must be the Sync-Lift stage sum (2x the joint-unit target)"


@needs_model
def test_ik_lift_rate_capped_at_hardware_ceiling(wik):
    """Total lift travel rate (2x the joint-unit dq) must respect the 60 mm/s
    hardware ceiling."""
    import mabel.config as C
    model, data, ik = wik
    ik.target_l_pos = ik.target_l_pos + np.array([0.0, 0.0, 0.6])
    ik.target_r_pos = ik.target_r_pos + np.array([0.0, 0.0, 0.6])
    t0 = ik.target_q["Lift"]
    _plan(model, data, ik, seconds=1.0)
    travel_rate = 2.0 * (ik.target_q["Lift"] - t0) / 1.0
    assert travel_rate <= C.MAX_LIFT_IK_SPEED + 1e-6, \
        f"lift travel {travel_rate:.4f} m/s exceeds the hardware cap"


@needs_model
def test_ik_torso_rate_capped(wik):
    """A far-forward target must not slam the torso: the swing carries the whole
    upper-body inertia past the static support edge (the measured face-plant)."""
    import mabel.config as C
    model, data, ik = wik
    ik.target_l_pos = ik.target_l_pos + np.array([-1.2, 0.0, 0.0])
    ik.target_r_pos = ik.target_r_pos + np.array([-1.2, 0.0, 0.0])
    t0 = ik.target_q["Torso"]
    _plan(model, data, ik, seconds=0.5)
    rate = abs(ik.target_q["Torso"] - t0) / 0.5
    assert rate <= C.MAX_TORSO_IK_SPEED + 1e-6, f"torso rate {rate:.2f} rad/s uncapped"


@needs_model
def test_com_guard_projects_margin_diving_dq(wik, monkeypatch):
    """The CBF net: near the stop line, a dq that dives the CoM margin must come
    out of the projection consuming margin no faster than the damper allows —
    and strictly slower than the raw dive. (The line is monkeypatched up to
    0.06 m so a plain torso lean from home reaches the approach band; the
    production line 0.01 m sits below the natural operating band by design.)"""
    import mabel.config as C
    model, data, ik = wik
    acts = list(C.IK_ACTUATORS)
    # Reach a forward posture first (guard + SC off): torso lean alone barely
    # moves the CoM with the arms tucked, so extend toward a far target, then
    # park the stop line just under the reached margin (band membership).
    # The production torso weight is deliberately LAZY (30) — restore an eager
    # torso for the SETUP so the plan actually consumes margin within seconds.
    ik.com_guard = False
    ik.self_collision = False
    ik.W_inv[1, 1] = 1.0 / 2.0
    ik.target_l_pos = ik.target_l_pos + np.array([-1.2, 0.0, 0.0])
    ik.target_r_pos = ik.target_r_pos + np.array([-1.2, 0.0, 0.0])
    _plan(model, data, ik, seconds=4.0)
    m0 = ik._margin_at(acts)
    assert m0 < 0.09, "setup: the forward plan should have consumed some margin"
    monkeypatch.setattr(C, "COM_GUARD_MARGIN", m0 - 0.005)
    ik.com_guard = True
    dive = np.zeros(len(acts))
    dive[1] = -1.0                              # slam the torso further forward
    proj = ik._com_guard_project(dive.copy(), acts)
    dt = 0.05
    m_dive = ik._margin_at(acts, dive, dt)
    m_proj = ik._margin_at(acts, proj, dt)
    assert m_proj > m_dive + 1e-4, "projection must consume less margin than the dive"
    assert m_proj >= C.COM_GUARD_MARGIN - 5e-3, \
        f"projected step still crosses the stop line ({m_proj:.4f})"


# ── BaseFollow (task-space base follower) ────────────────────────────────────
@needs_model
def test_base_follow_far_target_drives_forward(kmodel):
    from mabel.whole_body_controller import BaseFollow
    import mabel.config as C
    model, data = kmodel
    bf = BaseFollow(model)
    ch = data.body(C.CHASSIS_BODY).xpos
    p = ch + np.array([-1.8, 0.0, 0.6])         # far ahead (front = -X)
    v = None
    for _ in range(100):                        # let safe_base_step ramp
        v = bf.step(data, 0.02, 0.0, stick=None, p_l=p, p_r=p)
    assert v[0] < -0.1, f"far forward target must command a forward (-X) twist, got {v}"
    assert abs(v[2]) < 1e-9, "dead-ahead target must not command yaw"


@needs_model
def test_base_follow_stick_overrides_and_rests_at_home_offset(kmodel):
    from mabel.whole_body_controller import BaseFollow
    import mabel.config as C
    model, data = kmodel
    bf = BaseFollow(model)
    ch = data.body(C.CHASSIS_BODY).xpos
    far = ch + np.array([-1.8, 0.0, 0.6])
    # Targets AT the palms' actual pose == the home working offset: the position
    # servo's error is ~0 there, so the base must rest (snap semantics: a target
    # meaningfully CLOSER than the home offset makes it back up instead).
    p_l0 = data.body(C.LEFT_PALM).xpos.copy()
    p_r0 = data.body(C.RIGHT_PALM).xpos.copy()
    for _ in range(100):
        v = bf.step(data, 0.02, 0.0, stick=(0.5, 0.0, 0.0), p_l=far, p_r=far)
    assert v[0] > 0.1, "a deliberate stick must beat the follower"
    bf.reset()
    for _ in range(100):
        v = bf.step(data, 0.02, 0.0, stick=(0.0, 0.0, 0.0), p_l=p_l0, p_r=p_r0)
    assert np.allclose(v, 0.0, atol=1e-6), \
        f"targets at the home working offset must rest the base, got {v}"
    bf.reset()
    for _ in range(50):
        v = bf.step(data, 0.02, 0.0, stick=None, p_l=None, p_r=None)
    assert np.allclose(v, 0.0, atol=1e-6), "no targets + no stick -> no motion"


@needs_model
def test_base_follow_lateral_strafe_and_pair_rotation(kmodel):
    """Pair-alignment heading: equal-lateral targets strafe WITHOUT yaw; a
    rotated pair commands yaw in the rotation's direction."""
    from mabel.whole_body_controller import BaseFollow
    import mabel.config as C
    model, data = kmodel
    p_l0 = data.body(C.LEFT_PALM).xpos.copy()
    p_r0 = data.body(C.RIGHT_PALM).xpos.copy()
    bf = BaseFollow(model)
    lat = np.array([0.0, -0.5, 0.0])            # both left (robot left = -Y)
    v = None
    for _ in range(100):
        v = bf.step(data, 0.02, 0.0, p_l=p_l0 + lat, p_r=p_r0 + lat)
    assert v[1] < -0.05, f"lateral pair must strafe left, got {v}"
    assert abs(v[2]) < 1e-6, f"equal lateral drag must not yaw, got {v}"
    bf.reset()
    cen = 0.5 * (p_l0 + p_r0)
    th = math.radians(40.0)
    R = np.array([[math.cos(th), -math.sin(th), 0],
                  [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    for _ in range(100):
        v = bf.step(data, 0.02, 0.0,
                    p_l=cen + R @ (p_l0 - cen), p_r=cen + R @ (p_r0 - cen))
    assert v[2] > 0.1, f"a CCW pair rotation must command CCW yaw, got {v}"


# ── lift units in hold/resume (facade + kinematic controller) ────────────────
@needs_model
def test_hold_current_preserves_lift_height(fresh):
    """hold_current used to command the Lift tendon servo with the single-stage
    qpos — HALF the real extension — so every Soft→Stiff hand-back dropped a
    raised lift to half height."""
    import mujoco
    from mabel.facade import MabelTeleop
    model, data = fresh()
    t = MabelTeleop(model, data)
    for j in ("Lift Lower", "Lift Upper"):
        data.joint(j).qpos[0] = 0.15
    mujoco.mj_forward(model, data)
    t.controller.hold_current()
    assert data.actuator("Lift").ctrl[0] == pytest.approx(0.30, abs=1e-6), \
        "hold_current must command the tendon with the stage SUM"


@needs_model
def test_resume_hold_reseeds_lift_from_measured_tendon(fresh):
    """_measured_ctrl indexed jnt_qposadr with a TENDON id for the Lift and read
    the free joint's x — torque re-enable then snapped a stored lift target to
    garbage instead of holding the measured extension."""
    import mujoco
    from mabel.facade import MabelTeleop
    model, data = fresh()
    t = MabelTeleop(model, data)
    for j in ("Lift Lower", "Lift Upper"):
        data.joint(j).qpos[0] = 0.12
    data.qpos[0] = 3.0                     # base far from origin: the old bug read this
    mujoco.mj_forward(model, data)
    t.joint_targets["lift_lower"] = 0.05   # stale stored target
    t.joint_targets["lift_upper"] = 0.05
    t.resume_hold()
    assert t.joint_targets["lift_lower"] == pytest.approx(0.12, abs=1e-6)
    assert t.joint_targets["lift_upper"] == pytest.approx(0.12, abs=1e-6)
