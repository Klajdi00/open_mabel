"""Whole-body drift fixes (Retarget Lab, 2026-10-15): "the body rotation and the whole
re-targeting drift over time", "the lift is moving so fast — max ~3 cm/s".

Pins:
  * LazyHeadHeading: bounded by its slack at every instant, converges to the head
    whenever the head is held (NO drift — the defect), follows a real turn, rejects a
    glance-and-back, holds through a dropout, and is in the engine's body_yaw convention;
  * BASE_TRANSLATION_MODE "body": walking forward along the operator's own facing drives
    the robot forward whatever ARKit direction they faced; walking away and back returns
    the base target to its start (1:1, no gain);
  * the lift follow is slew-limited to LIFT_FOLLOW_SPEED and the torso to
    TORSO_FOLLOW_SPEED per SECOND when dt is given.
"""
import math

import numpy as np
import pytest

from mabel.whole_body_retargeting.heading import LazyHeadHeading, head_heading, SLACK

DT = 1 / 60


def H(yaw, p=(0.0, 1.6, 0.0)):
    """ARKit head pose, yaw about +y (0 = facing -z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    M = np.eye(4)
    M[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    M[:3, 3] = p
    return M


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(model, yaws):
    return np.array([model.update(H(y), DT) for y in yaws])


def test_convention_matches_the_engine():
    # facing ARKit -z: body right = ARKit +x -> robot (0, 1, 0) -> atan2(0, -1) = pi
    assert head_heading(H(0.0)) == pytest.approx(math.pi)
    # a CCW (left) head turn is a positive change of the heading
    assert wrap(head_heading(H(0.3)) - head_heading(H(0.0))) == pytest.approx(0.3)


def test_never_further_than_the_slack_and_converges_when_held():
    m = LazyHeadHeading()
    m.calibrate(H(0.0))
    rng = np.random.default_rng(0)
    yaws = np.cumsum(rng.normal(0, 0.04, 3000))          # a random walk of the head, 50 s
    out = run(m, yaws)
    h = np.array([head_heading(H(y)) for y in yaws])
    assert np.max(np.abs(wrap(out - h))) <= SLACK + 1e-9
    held = run(m, [yaws[-1]] * int(12 / DT))              # head held 12 s
    assert abs(wrap(held[-1] - head_heading(H(yaws[-1])))) < math.radians(0.5)


def test_returns_after_turns_no_latched_offset():
    """The defect: after turning and coming back, the estimate kept 30-75 deg."""
    m = LazyHeadHeading()
    m.calibrate(H(0.0))
    seq = list(np.linspace(0, -1.6, 90)) + [-1.6] * 300 + list(np.linspace(-1.6, 0, 90)) + [0.0] * 720
    out = run(m, seq)
    assert abs(wrap(out[-1] - head_heading(H(0.0)))) < math.radians(1.0)


def test_follows_a_real_turn():
    m = LazyHeadHeading()
    m.calibrate(H(0.0))
    out = run(m, list(np.linspace(0, math.radians(120), 60)) + [math.radians(120)] * 720)
    assert abs(wrap(out[-1] - head_heading(H(math.radians(120))))) < math.radians(1.0)
    assert abs(wrap(out[60] - head_heading(H(math.radians(120))))) <= SLACK + 1e-6   # during: within slack


def test_glance_and_back_is_rejected():
    m = LazyHeadHeading()
    m.calibrate(H(0.0))
    g = math.radians(20)
    seq = list(np.linspace(0, g, 15)) + [g] * 30 + list(np.linspace(g, 0, 15)) + [0.0] * 60   # 0.5 s look
    out = run(m, seq)
    dev = np.max(np.abs(wrap(out - head_heading(H(0.0)))))
    assert dev < math.radians(6), math.degrees(dev)


def test_holds_through_a_dropout():
    m = LazyHeadHeading()
    m.calibrate(H(0.4))
    v = m.update(None, DT)
    assert v == pytest.approx(head_heading(H(0.4)))


# ── base follow + rate limits ────────────────────────────────────────────────
mujoco = pytest.importorskip("mujoco")
from pathlib import Path  # noqa: E402

import mabel.teleop_engine.wholebody as WB  # noqa: E402

SCENE = Path(__file__).resolve().parents[2] / "simulation/mabel_mujoco/scenes/mabel_bare.xml"


def _wbc(monkeypatch, base=True):
    monkeypatch.setattr(WB, "BASE_TRANSLATION_MODE", "body")
    monkeypatch.setattr(WB, "HEADING_MODE", "head")
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    w = WB.WholeBodyController(m, d, torso_act=aid("Torso"), lift_act=aid("Lift"))
    w.set_base_in_loop(base, d)
    return w, d, aid


@pytest.mark.parametrize("facing_deg", [0, 90, -135])
def test_walking_forward_drives_the_robot_forward_whatever_the_arkit_facing(monkeypatch, facing_deg):
    w, d, _ = _wbc(monkeypatch)
    yaw = math.radians(facing_deg)
    H0 = H(yaw)
    fwd = H0[:3, :3] @ np.array([0, 0, -1.0])
    by = head_heading(H0)
    w.update_targets(d, H0[:3, 3], H0[:3, 3], by, 1.6, dt=DT)
    p = H0[:3, 3] + 0.6 * fwd
    w.update_targets(d, p, p, by, 1.6, dt=DT)
    vx, vy, _ = w.base_tgt
    assert vx < -0.1 and abs(vy) < 1e-6                     # chassis front = -x


def test_walk_away_and_back_returns_the_target(monkeypatch):
    w, d, _ = _wbc(monkeypatch)
    H0 = H(0.0)
    by = head_heading(H0)
    w.update_targets(d, H0[:3, 3], H0[:3, 3], by, 1.6, dt=DT)
    away = H0[:3, 3] + np.array([0.0, 0.0, 0.8])            # 0.8 m back
    w.update_targets(d, away, away, by, 1.6, dt=DT)
    assert abs(w.base_tgt[0]) > 0.1
    w.update_targets(d, H0[:3, 3], H0[:3, 3], by, 1.6, dt=DT)
    assert np.allclose(w.base_tgt[:2], 0.0, atol=1e-9)     # back at the start: no residual offset


def test_lift_and_torso_follow_are_slew_limited(monkeypatch):
    # Lift follows only in whole-body scope; upper-body scope holds the column.
    w, d, aid = _wbc(monkeypatch, base=True)
    la, ta = aid("Lift"), aid("Torso")
    H0 = H(0.0)
    by = head_heading(H0)
    w.update_targets(d, H0[:3, 3], H0[:3, 3], by, 1.6, dt=DT)
    up = H0[:3, 3] + np.array([0, 0.25, 0])                 # head 25 cm up: the lift wants +0.2 m
    l0 = float(d.ctrl[la])
    steps = []
    for _ in range(120):
        prev = float(d.ctrl[la])
        w.update_targets(d, up, up, by, 1.85, dt=DT)
        steps.append(abs(float(d.ctrl[la]) - prev))
    assert max(steps) <= WB.LIFT_FOLLOW_SPEED * DT + 1e-12
    assert float(d.ctrl[la]) - l0 > 0.05                    # it does get there, slowly
    bend = H0[:3, 3] + np.array([0, -0.13, -0.40])          # a 35 deg hip bend in one frame
    monkeypatch.setattr(WB, "TORSO_LEAN_MODE", "angle")
    t_prev = float(d.ctrl[ta])
    w.update_targets(d, bend, bend, by, 1.47, dt=DT)
    assert abs(float(d.ctrl[ta]) - t_prev) <= WB.TORSO_FOLLOW_SPEED * DT + 1e-12


def test_upper_body_scope_holds_lift(monkeypatch):
    w,d,aid=_wbc(monkeypatch,base=False)
    initial=float(d.ctrl[aid('Lift')])
    for height in (1.6,1.85,1.4):
        p=np.array([0.,height,0.])
        w.update_targets(d,p,p,head_heading(H(0.)),height,dt=DT)
    assert float(d.ctrl[aid('Lift')])==initial
