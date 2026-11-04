"""The hip-pivot torso lean model (hyperparams TORSO_LEAN_MODE = "angle").

Operator report (Retarget Lab, 2026-10-15): "my torso is not bending forward a
lot when I am bending forward by a lot". The shipped "box" law clips the head's
forward offset to BASE_BOX_FWD before the gain, so the robot torso saturates at
TORSO_GAIN * (0.18 - 0.06) = 0.30 rad whatever the operator does. These tests pin
  * the defect itself (box mode caps at 0.30 rad on a 40 deg bend) — the control;
  * the angle model following a 40 deg bend to ~40 deg minus the deadband;
  * GLOBAL scope telling a bend (head drops) from a walk (head level): a walk
    drives the base and leaves the torso inside the box law; a bend does not
    drive the base;
  * the lift not reading a bend as "the operator got shorter".
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pathlib import Path

import mabel.teleop_engine.wholebody as WB

SCENE = Path(__file__).resolve().parents[2] / "simulation/mabel_mujoco/scenes/mabel_bare.xml"
L = WB.TORSO_LEAN_PIVOT
H0 = np.array([0.0, 1.40, 0.0])          # ARKit: Y up, -Z forward


def _wbc(mode, base_in_loop, monkeypatch):
    monkeypatch.setattr(WB, "TORSO_LEAN_MODE", mode)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    w = WB.WholeBodyController(m, d, torso_act=aid("Torso"), lift_act=aid("Lift"))
    w.set_base_in_loop(base_in_loop, d)
    return w, d, aid("Torso"), aid("Lift")


def _head(fwd, drop):
    return H0 + np.array([0.0, -drop, -fwd])


def _bend(theta):
    return _head(L * np.sin(theta), L * (1 - np.cos(theta)))


def _run(w, d, head, n=60):
    w.update_targets(d, H0, H0, 0.0, float(H0[1]))       # anchor
    info = None
    for _ in range(n):
        info = w.update_targets(d, head, head, 0.0, float(head[1]))
    return info


def test_box_mode_saturates_at_17deg(monkeypatch):
    """The control: the shipped law cannot follow a deep bend."""
    w, d, ta, _ = _wbc("box", False, monkeypatch)
    _run(w, d, _bend(np.radians(40)))
    lean = w.torso.t0 - float(d.ctrl[ta])
    assert lean == pytest.approx(WB.TORSO_GAIN * (WB.BASE_BOX_FWD - WB.TORSO_DEADBAND), abs=1e-6)
    assert np.degrees(lean) < 18


@pytest.mark.parametrize("deg", [20, 40, 55])
def test_angle_mode_follows_the_bend(monkeypatch, deg):
    w, d, ta, _ = _wbc("angle", False, monkeypatch)
    th = np.radians(deg)
    _run(w, d, _bend(th))
    lean = w.torso.t0 - float(d.ctrl[ta])
    want = min(WB.TORSO_LEAN_GAIN * (th - WB.TORSO_LEAN_DEADBAND), w.torso.t0 - w.torso.lo)
    assert lean == pytest.approx(want, abs=0.01)


def test_global_walk_drives_base_not_torso(monkeypatch):
    """Head 0.5 m forward, level: a walk. The base moves; the torso keeps the
    within-box share only (the box law's own quantity)."""
    w, d, ta, _ = _wbc("angle", True, monkeypatch)
    _run(w, d, _head(0.5, 0.0))
    lean = w.torso.t0 - float(d.ctrl[ta])
    assert lean <= np.arctan2(WB.BASE_BOX_FWD, L) + 1e-6
    assert abs(w.base_tgt[0]) + abs(w.base_tgt[1]) > 0.05


def test_global_bend_does_not_walk(monkeypatch):
    w, d, ta, _ = _wbc("angle", True, monkeypatch)
    _run(w, d, _bend(np.radians(40)))
    lean = w.torso.t0 - float(d.ctrl[ta])
    assert np.degrees(lean) > 30
    assert abs(w.base_tgt[0]) + abs(w.base_tgt[1]) < 1e-6


def test_bend_is_not_read_as_shrinking(monkeypatch):
    """A 40 deg bend drops the head 16 cm; the lift should not chase it."""
    w, d, _, la = _wbc("angle", False, monkeypatch)
    lift0 = float(d.ctrl[la])
    _run(w, d, _bend(np.radians(40)), n=200)
    assert abs(float(d.ctrl[la]) - lift0) < 0.01


def test_deep_bend_with_more_drop_than_a_hip_pivot(monkeypatch):
    """The measured deep bend (live take, t = 60 s): head 0.227 m forward and
    0.152 m down. Forward alone reads 22 deg; the drop reads 38 deg and the
    forward travel allows it (cap asin(0.227 / 0.35) = 40 deg)."""
    w, d, ta, _ = _wbc("angle", False, monkeypatch)
    _run(w, d, _head(0.227, 0.152))
    lean = np.degrees(w.torso.t0 - float(d.ctrl[ta]))
    assert 30 < lean < 36


def test_squat_is_not_a_bend(monkeypatch):
    """Head 0.20 m straight down, no forward travel: the lift's job, not the torso's."""
    w, d, ta, _ = _wbc("angle", False, monkeypatch)
    _run(w, d, _head(0.0, 0.20))
    assert abs(w.torso.t0 - float(d.ctrl[ta])) < 1e-6
