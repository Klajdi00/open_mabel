"""The wrist-tracking self-collision path, end to end in sim (no robot).

Layers under test:
  1. SelfCollisionGuard.project — the CBF velocity projection itself: free motion
     bit-identical, tangential component preserved, normal closing excess removed.
  2. WholeBodyIK sc_method="project" — the retarget-path law: an adversarial
     crossing never interpenetrates, and the hand makes tangential progress
     (slides) instead of stopping dead.
  3. JointTargetProjector — the manipulate choke point (web/iOS/VP client-planned
     joint targets): a raw client pose that penetrates is filtered onto a path
     that never crosses any pair's margin; free targets pass through exactly.
  4. MabelTeleop facade — joint_command wrist targets in method "joint"/"wrist"
     ride the projection; an adversarial sweep of colliding targets never drives
     the physical sim inside a margin.
  5. The web preview twin (web_gui/*/web/js/collision.js) mirrors the controller
     constants — a drift guard, since the browser only PREVIEWS what the
     server-side filter will do.
"""
import json
import os
import re

import numpy as np
import mujoco
import pytest

from conftest import needs_model, settle, MODEL_XML

import mabel.config as C
from mabel.whole_body_controller.ik import WholeBodyIK
from mabel.whole_body_controller.collision_avoidance import (
    SelfCollisionGuard, JointTargetProjector)
from mabel.whole_body_controller.collision_avoidance.capsule_guard import _seg_seg

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worst_slack(model, guard, ref_data, qpos, scratch):
    """min over monitored pairs of (gap − pair margin) at a full qpos."""
    scratch.qpos[:] = qpos
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    guard.data = scratch
    worst = np.inf
    for ca, cb in guard.pairs:
        pa0, pa1 = ca.world(scratch)
        pb0, pb1 = cb.world(scratch)
        wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
        d = float(np.linalg.norm(wa - wb)) - ca.radius - cb.radius
        worst = min(worst, d - guard._margin_for(ca, cb))
    guard.data = ref_data
    return worst


# ── 1. the projection law itself ─────────────────────────────────────────────
@needs_model
def test_project_free_motion_untouched(fresh):
    model, data = fresh()
    settle(model, data)
    guard = SelfCollisionGuard(model, data)
    proj = JointTargetProjector(model, data, guard=guard)
    dofs = list(proj._dof.values())
    rng = np.random.default_rng(1)
    for _ in range(20):
        dq = rng.normal(0.0, 0.02, len(dofs))
        G, h = guard.rows(dofs, braking=True)
        if len(G) and np.any(G @ dq > h):
            continue                        # not a free sample at this pose
        assert np.array_equal(guard.project(dq, dofs), dq)


@needs_model
def test_project_keeps_tangential_removes_normal(fresh):
    """Construct a pose with EXACTLY one active pair, split a commanded velocity
    into normal (gap-closing) + tangential parts, and check the projection:
    the tangential part must survive UNCHANGED; the closing rate must be clipped
    to the barrier allowance."""
    model, data = fresh()
    settle(model, data)
    proj = JointTargetProjector(model, data)
    guard = proj.guard
    dofs = list(proj._dof.values())

    # Park the left HAND near the torso so a hand~torso row is active (the old
    # arm-fold scenario engaged only the elbow row, removed 2026-07-30): drive
    # an unguarded IK to a chest point, then let the filter park on the boundary.
    ik = WholeBodyIK(model, data)
    ik.set_self_collision(False)
    chest = data.body("torso").xpos + np.array([0.0, 0.05, 0.15])
    ik.sync_to_current()
    ik.target_l_pos = chest.copy()
    for _ in range(600):
        ik.apply(ik.solve(arms_only=True), 0.005, arms_only=True)
    tgt = {n: float(data.qpos[proj._qadr[n]]) for n in proj.names}
    tgt.update({n: float(ik.target_q[n]) for n in ik.arm_actuators})
    path = proj.plan_path(tgt, dt=0.05, max_steps=300)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    for n, v in path[-1].items():
        scratch.qpos[proj._qadr[n]] = v
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    guard.data = scratch
    try:
        G, h = guard.rows(dofs, braking=True)
        assert len(G) >= 1, "flow never engaged a barrier — scenario too gentle"
        a = G[0] / np.linalg.norm(G[0])

        rng = np.random.default_rng(2)
        v = rng.normal(0.0, 0.5, len(dofs))
        v_t = v - a * (a @ v)                      # tangential to the first row
        # tangential-only input w.r.t. every OTHER active row too: project out
        for g_row in G[1:]:
            gn = g_row / max(np.linalg.norm(g_row), 1e-12)
            v_t = v_t - gn * (gn @ v_t)
        out_t = guard.project(v_t, dofs)
        assert np.allclose(out_t, v_t, atol=1e-6), \
            "a purely tangential velocity was altered"

        # now add a hard normal CLOSING component (G row is −nᵀJ: closing has
        # G·v > 0). The output must satisfy the CBF rows and keep the
        # tangential part intact.
        v_in = v_t + G[0] * 2.0
        out = guard.project(v_in, dofs)
        G2, h2 = guard.rows(dofs, braking=True)
        assert np.all(G2 @ out <= h2 + 1e-6), "projected velocity violates the CBF"
        # tangential content preserved: the correction (out − v_t) must lie in
        # the ROW SPACE of the active constraints.
        #
        # Removed one row at a time this is only correct when the rows are
        # mutually ORTHOGONAL — sequential projection along non-orthogonal
        # directions does not remove their span, it removes a bit and reintroduces
        # a bit. That held while the scenario engaged a single row; it stopped
        # holding on 2026-08-23 when the arm~body pairs were restored and a hand
        # parked on the chest legitimately activates several. Least squares
        # against the row space is the same claim, computed correctly for any
        # number of rows.
        resid = out - v_t
        coef, *_ = np.linalg.lstsq(G.T, resid, rcond=None)
        off_span = float(np.linalg.norm(resid - G.T @ coef))
        assert off_span < 1e-6, \
            f"the correction was not purely along the constraint normals "\
            f"({off_span:.2e} off the row space of {len(G)} active rows)"
    finally:
        guard.data = data


# ── 2. the retarget-path law (WholeBodyIK, method "project") ─────────────────
def _drive_crossing_project(model, data, seconds=2.5):
    ik = WholeBodyIK(model, data)
    ik.set_self_collision(True, method="project")
    meas = SelfCollisionGuard(model, data)
    l0 = data.body(C.LEFT_PALM).xpos.copy()
    r0 = data.body(C.RIGHT_PALM).xpos.copy()
    ik.sync_to_current()
    ik.target_l_pos = r0.copy()
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
def test_project_method_no_interpenetration(fresh):
    model, data = fresh()
    settle(model, data)
    min_gap = _drive_crossing_project(model, data)
    assert min_gap >= 0.0, \
        f"'project' interpenetrated: min gap {min_gap*100:.2f} cm"


@needs_model
def test_project_method_slides_instead_of_stopping(fresh):
    """A wrist target INSIDE the chest: under "stop" the hand halts on the
    approach ray; under "project" it must keep moving TANGENTIALLY and end
    closer to the target than "stop" left it (that is the whole point of the
    law). Both must stay outside the capsule model."""
    model, data = fresh()
    settle(model, data)

    def run(method):
        m2, d2 = mujoco.MjModel.from_xml_path(MODEL_XML), None
        d2 = mujoco.MjData(m2)
        d2.qpos[:] = data.qpos
        mujoco.mj_forward(m2, d2)
        ik = WholeBodyIK(m2, d2)
        ik.set_self_collision(True, method=method)
        chest = d2.body("torso").xpos + np.array([0.0, 0.05, 0.10])
        ik.sync_to_current()
        ik.target_l_pos = chest.copy()
        dt = m2.opt.timestep
        for _ in range(int(2.5 / dt)):
            ik.apply(ik.solve(arms_only=True), dt, arms_only=True)
            mujoco.mj_step(m2, d2)
        meas = SelfCollisionGuard(m2, d2)
        meas.rows(ik.dofs)
        gap = float(meas.min_dist)
        return float(np.linalg.norm(d2.body(C.LEFT_PALM).xpos - chest)), gap

    d_stop, gap_stop = run("stop")
    d_proj, gap_proj = run("project")
    assert gap_proj >= 0.0, f"'project' interpenetrated ({gap_proj*100:.2f} cm)"
    assert d_proj <= d_stop + 1e-6, \
        (f"'project' ({d_proj*100:.1f} cm from target) did not get at least as "
         f"close as 'stop' ({d_stop*100:.1f} cm) — no tangential sliding")


# ── 3. the manipulate choke point (client-planned joint targets) ─────────────
@needs_model
def test_target_filter_blocks_penetrating_client_pose(fresh):
    model, data = fresh()
    settle(model, data)

    # a client-planned (unguarded IK) wrist pose inside the chest
    ik = WholeBodyIK(model, data)
    ik.set_self_collision(False)
    chest = data.body("torso").xpos + np.array([0.0, 0.05, 0.15])
    ik.sync_to_current()
    ik.target_l_pos = chest.copy()
    for _ in range(600):
        ik.apply(ik.solve(arms_only=True), 0.005, arms_only=True)
    raw = {n: float(ik.target_q[n]) for n in ik.arm_actuators}

    proj = JointTargetProjector(model, data)
    guard = proj.guard
    scratch = mujoco.MjData(model)

    s = mujoco.MjData(model)
    s.qpos[:] = data.qpos
    for n, v in raw.items():
        s.qpos[proj._qadr[n]] = v
    assert _worst_slack(model, guard, data, s.qpos, scratch) < 0, \
        "raw client pose does not collide — scenario too gentle to discriminate"

    path = proj.plan_path(raw, dt=0.05, max_steps=400)
    worst = np.inf
    for wp in path:
        s.qpos[:] = data.qpos
        for n, v in wp.items():
            s.qpos[proj._qadr[n]] = v
        worst = min(worst, _worst_slack(model, guard, data, s.qpos, scratch))
    assert worst >= -1e-3, \
        f"filtered path entered a margin by {-worst*1000:.1f} mm"

    # tangential progress: the palm must still end CLOSER to the chest point
    s.qpos[:] = data.qpos
    for n, v in path[-1].items():
        s.qpos[proj._qadr[n]] = v
    mujoco.mj_kinematics(model, s)
    d_end = float(np.linalg.norm(s.body(C.LEFT_PALM).xpos - chest))
    d_start = float(np.linalg.norm(data.body(C.LEFT_PALM).xpos - chest))
    assert d_end < d_start - 0.05, \
        f"no progress toward the target ({d_start*100:.1f} → {d_end*100:.1f} cm)"


@needs_model
def test_target_filter_free_target_reached_exactly(fresh):
    model, data = fresh()
    settle(model, data)
    proj = JointTargetProjector(model, data)
    tgt = {n: float(data.qpos[proj._qadr[n]]) for n in proj.names}
    tgt["Right Arm 2"] = float(tgt["Right Arm 2"]) - 0.4    # lift the arm outward
    path = proj.plan_path(tgt, dt=0.05, max_steps=200)
    end = path[-1]
    for n, v in tgt.items():
        assert abs(end[n] - v) < 1e-3, f"free target not reached on {n}"


@needs_model
def test_target_filter_adversarial_sweep_never_penetrates(fresh):
    """40 random colliding-ish target poses for both arms: the filtered path
    must never cross any pair's margin by more than 1 mm (discrete-step slack)."""
    model, data = fresh()
    settle(model, data)
    proj = JointTargetProjector(model, data)
    guard = proj.guard
    scratch = mujoco.MjData(model)
    s = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    lims = {n: model.joint(n).range for n in proj.names}
    worst_all = np.inf
    for _ in range(40):
        tgt = {}
        for n in proj.names:
            lo, hi = lims[n]
            tgt[n] = float(rng.uniform(lo, hi))
        proj.reset()
        path = proj.plan_path(tgt, dt=0.06, max_steps=150)
        for wp in path[::3] + [path[-1]]:
            s.qpos[:] = data.qpos
            for n, v in wp.items():
                s.qpos[proj._qadr[n]] = v
            worst_all = min(worst_all,
                            _worst_slack(model, guard, data, s.qpos, scratch))
    assert worst_all >= -1e-3, \
        f"adversarial sweep entered a margin by {-worst_all*1000:.2f} mm"


# ── 4. the facade path (what web/iOS/VP joint_command actually rides) ────────
@needs_model
def test_facade_filters_manipulate_targets(fresh):
    from mabel.facade import MabelTeleop
    model, data = fresh()
    settle(model, data)
    teleop = MabelTeleop(model, data)
    teleop.set_control_spec(method="joint", control_type="impedance",
                            region="arm", stiffness=1.0)

    # the same penetrating client pose as above, shipped as joint_command
    ik = WholeBodyIK(model, data)
    ik.set_self_collision(False)
    chest = data.body("torso").xpos + np.array([0.0, 0.05, 0.15])
    ik.sync_to_current()
    ik.target_l_pos = chest.copy()
    for _ in range(600):
        ik.apply(ik.solve(arms_only=True), 0.005, arms_only=True)
    ios = {f"left_arm_{i}": float(ik.target_q[f"Left Arm {i}"]) for i in range(1, 8)}
    teleop.set_joint_targets(ios)

    meas = SelfCollisionGuard(model, data)
    dofs = list(range(model.nv))
    dt = model.opt.timestep
    min_gap = np.inf
    for _ in range(int(4.0 / dt)):
        teleop.update(None, dt=dt)
        mujoco.mj_step(model, data)
        meas.rows(dofs)
        min_gap = min(min_gap, float(meas.min_dist))
    assert min_gap >= -1e-3, \
        f"facade let a manipulate target interpenetrate ({min_gap*100:.2f} cm)"
    assert teleop._target_proj is not None and not teleop._target_proj_dead, \
        "the target projector never engaged"


# ── 5. the web preview twin cannot drift silently ─────────────────────────────
def test_web_twin_constants_match():
    js_path = os.path.join(_REPO, "web_gui", "control_center", "web", "js",
                           "collision.js")
    if not os.path.isfile(js_path):
        pytest.skip("web twin not present in this checkout")
    js = open(js_path).read()

    def js_const(name):
        m = re.search(rf"export const {name} = ([0-9.]+);", js)
        assert m, f"{name} missing from collision.js"
        return float(m.group(1))

    assert js_const("SC_MARGIN") == C.SC_MARGIN
    assert js_const("SC_INFLUENCE") == C.SC_INFLUENCE
    assert js_const("SC_ALPHA") == C.SC_ALPHA
    assert js_const("SC_STOP_DECEL") == C.SC_STOP_DECEL
    m = re.search(r"SC_RADII = Object\.freeze\(\{\s*([^}]+)\}", js)
    assert m, "SC_RADII missing from collision.js"
    js_radii = dict(re.findall(r"(\w+):\s*([0-9.]+)", m.group(1)))
    for k, v in C.SC_RADII.items():
        assert float(js_radii[k]) == float(v), f"SC_RADII[{k}] drifted in the web twin"
    # the sim twin must be byte-identical (one product)
    sim_path = os.path.join(_REPO, "web_gui", "simulation_control_center",
                            "web", "js", "collision.js")
    if os.path.isfile(sim_path):
        assert open(sim_path).read() == js, \
            "collision.js drifted between the two control-center studios"


@needs_model
def test_avoidance_never_walks_j1_to_its_limit(fresh):
    """Operator report 2026-07-30: J1 kept hitting its range limit — the
    elbow~torso rows (gradients ~pure J1/J2) walked it there. With those pairs
    removed, a hand-near-torso approach may use a LITTLE shoulder motion in its
    corrections, but the filtered J1 must stay close to what the operator
    commanded — never pinned at the range clamp."""
    model, data = fresh()
    settle(model, data)
    proj = JointTargetProjector(model, data)
    tgt = {n: float(data.qpos[proj._qadr[n]]) for n in proj.names}
    j1_cmd = tgt["Right Arm 1"]
    # drive the right hand toward the chest via elbow/wrist joints
    tgt["Right Arm 2"] = -1.2
    tgt["Right Arm 4"] = -1.4
    path = proj.plan_path(tgt, dt=0.05, max_steps=300)
    lo, hi = model.joint("Right Arm 1").range
    m = float(proj.p_range_margin) if hasattr(proj, "p_range_margin") else 0.06
    for wp in path:
        j1 = wp["Right Arm 1"]
        assert abs(j1 - j1_cmd) < 0.25, \
            f"avoidance walked r_j1 {abs(j1-j1_cmd):.2f} rad off its command"
        assert lo + m / 2 < j1 < hi - m / 2, "r_j1 pinned at its range limit"
