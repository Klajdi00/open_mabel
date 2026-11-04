"""The arms must not collide with the body.

Operator, 2026-08-23: "the arms should not collide with the body ... right now
the robot arms will collide with the robot body". They did: measured over the
front-workspace posture table, the right forearm was inside the torso capsule on
17 % of commanded postures (up to 122 mm deep) and the upper arm on 9 %, because
those pairs were not monitored at all — only the hands and wrist mounts were.

These tests pin the four things that had to be true for the guard to work, each
of which was independently broken:

  1. the torso capsule sits on the torso (it was 43 mm off-centre and tilted, so
     the two arms were guarded asymmetrically at the same pose);
  2. the arm~body pairs are actually monitored;
  3. the ORCA wrist is a decision variable (the hand capsule turns with it, so
     without that column the barrier could not oppose a wrist-driven approach);
  4. the closed loop keeps the margin — and the check is sensitive enough to
     fail when the guard is off.
"""
import numpy as np
import mujoco
import pytest

from mabel import config as C
from mabel.teleop_engine.hyperparams import XML_FILE
from mabel.teleop_engine.workspace_table import ArmChain, palm_target, PALM_ROLLS
from mabel.teleop_engine.arm import ArmRetargeter
from mabel.teleop_engine.hand import HandOptimizer
from mabel.teleop_engine.collision import ArmCollisionProjector
from mabel.whole_body_controller.collision_avoidance.capsule_guard import (
    SelfCollisionGuard, _seg_seg)

SIDES = ("right", "left")
ARM_BODY = [("l_fore", "torso"), ("r_fore", "torso"),
            ("l_upper", "torso"), ("r_upper", "torso"),
            ("l_hand", "torso"), ("r_hand", "torso"),
            ("l_wristmt", "torso"), ("r_wristmt", "torso"),
            ("l_hand", "head"), ("r_hand", "head")]


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML_FILE)


@pytest.fixture
def rest(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.nkey - 1)
    d.ctrl[:] = 0.0
    d.ctrl[model.actuator("Lift").id] = 0.3175
    mujoco.mj_forward(model, d)
    return d


def _gap(g, d, a, b):
    ca, cb = g.caps[a], g.caps[b]
    pa0, pa1 = ca.world(d)
    pb0, pb1 = cb.world(d)
    wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
    return float(np.linalg.norm(wa - wb)) - ca.radius - cb.radius


# ── 1. the capsule is where the torso is ─────────────────────────────────────
def test_torso_capsule_sits_on_the_torso(model, rest):
    """Built from body ORIGINS it did not: `body_mount` is 43 mm right of the
    chest, so the capsule leaned into the right arm and the same rest pose read
    13 mm of clearance on the right and 100 mm on the left."""
    g = SelfCollisionGuard(model, rest)
    p0, p1 = g.caps["torso"].world(rest)
    axis = p1 - p0
    axis = axis / np.linalg.norm(axis)
    assert abs(float(np.dot(axis, [0, 0, 1]))) > 0.999, "torso capsule is tilted"

    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    gs = list(range(model.body_geomadr[tb], model.body_geomadr[tb] + model.body_geomnum[tb]))
    w = np.array([max(float(np.max(model.geom_size[i])), 1e-6) ** 3 for i in gs])
    cen = (w[:, None] * np.array([rest.geom_xpos[i] for i in gs])).sum(0) / w.sum()
    mid = 0.5 * (p0 + p1)
    lateral = np.linalg.norm((cen - mid)[:2])
    assert lateral < 0.010, f"capsule centre is {lateral*1000:.0f} mm off the mesh"


def test_rest_pose_clears_the_body_symmetrically(model, rest):
    """Both arms must see the same clearance at a mirror-image rest pose, or one
    of them is being guarded against a phantom."""
    g = SelfCollisionGuard(model, rest)
    for a, b in (("l_fore", "r_fore"), ("l_upper", "r_upper"), ("l_hand", "r_hand")):
        ga, gb = _gap(g, rest, a, "torso"), _gap(g, rest, b, "torso")
        assert abs(ga - gb) < 0.010, f"{a} {ga*1000:.0f} mm vs {b} {gb*1000:.0f} mm"
        assert ga > C.SC_MARGIN, f"{a} rests inside the margin ({ga*1000:.0f} mm)"


# ── 2/3. the guard is configured to be able to do its job ────────────────────
def test_arm_body_pairs_are_monitored(model, rest):
    g = SelfCollisionGuard(model, rest)
    have = {(a.name, b.name) for a, b in g.pairs} | {(b.name, a.name) for a, b in g.pairs}
    for pair in [("l_fore", "torso"), ("r_fore", "torso"),
                 ("l_upper", "torso"), ("r_upper", "torso")]:
        assert pair in have, f"{pair} is not guarded — the arm can enter the body"


def test_margin_is_a_thin_layer():
    """Operator asked for ~1 cm. Thin enough to keep the workspace, thick enough
    that the discrete barrier cannot step through it."""
    assert 0.005 <= C.SC_MARGIN <= 0.015


def test_orca_wrist_is_a_decision_variable(model, rest):
    """The hand capsule spans wrist->palm and turns with the ORCA wrist. Without
    that column the QP cannot oppose a wrist-driven approach at all — measured
    27 mm of interpenetration with every other constraint satisfied."""
    arms = {s: ArmRetargeter(model, s, HandOptimizer(model, s).R_rob) for s in SIDES}
    p = ArmCollisionProjector(model, rest, arms)
    assert p.n == 2 * (len(arms["right"].jids) + 1), \
        f"projector controls {p.n} DoFs; expected the arms plus both ORCA wrists"
    for s in SIDES:
        w = model.jnt_qposadr[arms[s].orca_wrist_jid]
        assert w in list(p.qadr), f"{s} ORCA wrist is not in the QP"


# ── 4. the closed loop keeps the margin ─────────────────────────────────────
def _closed_loop(model, d, guard_on, dirs=("down", "up", "inward"), step=3, hold=0.9):
    arms, chains = {}, {}
    for s in SIDES:
        chains[s] = ArmChain(model, s)
        chains[s].sync(d)
        arms[s] = ArmRetargeter(model, s, HandOptimizer(model, s).R_rob)
        arms[s].sync(d)
    proj = ArmCollisionProjector(model, d, arms) if guard_on else None
    g = proj.guard if proj else SelfCollisionGuard(model, d)
    act = {s: [model.actuator(f"{s.capitalize()} Arm {i}").id for i in range(1, 8)]
           for s in SIDES}
    wact = {s: model.actuator(f"{s.capitalize()} Wrist").id for s in SIDES}
    fps = 30.0
    from experiments.exp_front_workspace import make_grid, _hk_world
    worst = np.inf
    gr = make_grid("right")
    for i in range(0, gr.shape[0], step):
        for j in range(0, gr.shape[1], step):
            for k in range(0, gr.shape[2], step):
                for nm in dirs:
                    hk = {}
                    for s in SIDES:
                        chains[s].sync(d)
                        gg = make_grid(s)
                        p_s = gg.point(i, j, k)
                        R_s = palm_target(gg.canonical_fwd(p_s), PALM_ROLLS[nm], s)
                        hk[s], _, _ = _hk_world(chains[s], p_s, R_s)
                    for _ in range(int(hold * fps)):
                        for s in SIDES:
                            arms[s].sync(d)
                        qr = arms["right"].solve(hk["right"], dt=1 / fps)
                        ql = arms["left"].solve(hk["left"], dt=1 / fps)
                        wr = float(arms["right"].ik.qpos[arms["right"].orca_wrist_qadr])
                        wl = float(arms["left"].ik.qpos[arms["left"].orca_wrist_qadr])
                        if proj is not None:
                            qr, ql, wr, wl = proj.project(qr, ql, 1 / fps,
                                                          wrist_r=wr, wrist_l=wl)
                        d.ctrl[act["right"]] = qr[:7]
                        d.ctrl[act["left"]] = ql[:7]
                        d.ctrl[wact["right"]] = wr
                        d.ctrl[wact["left"]] = wl
                        for _ in range(int((1 / fps) / model.opt.timestep)):
                            mujoco.mj_step(model, d)
                        for a, b in ARM_BODY:
                            worst = min(worst, _gap(g, d, a, b))
    return worst


@pytest.mark.slow
def test_closed_loop_never_enters_the_body(model, rest):
    """The real path: retarget -> CBF projection -> servos -> physics."""
    worst = _closed_loop(model, rest, guard_on=True)
    assert worst > 0.0, f"arm entered the body: worst capsule gap {worst*1000:.1f} mm"


@pytest.mark.slow
def test_the_check_is_sensitive(model, rest):
    """A no-penetration result means nothing unless this fails with the guard
    off. Measured: -109 mm, on 6-10 % of frames."""
    worst = _closed_loop(model, rest, guard_on=False)
    assert worst < -0.02, ("guard OFF did not penetrate — the closed-loop check "
                           "is not exercising the geometry it claims to")
