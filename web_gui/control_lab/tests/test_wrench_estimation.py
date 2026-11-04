"""External-wrench pipeline: native-viewer force injection (bidirectional
memmap) + estimator accuracy/stability against MuJoCo ground truth.
DETERMINISTIC: step_sync worlds; ground truth = surface-point xfrc pushes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from control_lab.server.simworld import SimWorld                 # noqa: E402


@pytest.fixture(scope="module")
def world():
    # torque ON + a WBC method (joint/impedance) mounts the momentum-observer
    # contact estimator; the arm holds stiff and a push is localised + estimated.
    w = SimWorld(state_file="/tmp/mabel_control_lab_TEST_pert.f64")
    w.settle()
    w.set_torque(True)
    w.set_control_spec(method="joint", control_type="impedance", region="arm",
                       stiffness=45.0)      # max app stiffness → most rigid hold
    w.step_sync(1.0)
    return w


def _est(w):
    ov = w.snapshot()["overlay"].get("wrench") or {}
    return ov.get("est")


def _steady(w, seconds=0.6, dt=0.05):
    Fs, Ps, Bs = [], [], []
    for _ in range(int(seconds / dt)):
        w.step_sync(dt)
        e = _est(w)
        if e:
            Fs.append(e["force"]); Ps.append(e["point"]); Bs.append(e["body"])
    return np.array(Fs), np.array(Ps), Bs


def test_viewer_push_via_memmap(world):
    """The mjpython mirror's Ctrl+drag lands in the studio physics: inject a
    perturbation into the shared state file exactly as viewer_mirror does and
    assert the wrench is applied + surfaced as ground truth."""
    w = world
    assert w._mm is not None
    bid = w.model.body("left_palm").id
    po = w._pert_off
    w._mm[po + 1] = float(bid)
    w._mm[po + 2:po + 8] = [0.0, 15.0, 0.0, 0.0, 0.0, 0.0]
    w._mm[po] = 1.0
    w.step_sync(0.2)
    assert abs(w.data.xfrc_applied[bid, 1] - 15.0) < 1e-9
    tw = (w.snapshot()["overlay"].get("wrench") or {}).get("true")
    assert tw and tw["body"] == "left_palm"
    assert abs(tw["force"][1] - 15.0) < 1e-6
    # truth point must sit on the robot, not at the CoM / in the air
    from mabel.whole_body_controller.admittance.wrench_estimation import (
        _ray_bodies, body_subtree)
    p = np.asarray(tw["point"])
    d = np.array([0.0, 1.0, 0.0])
    hit = _ray_bodies(w.model, w.data, body_subtree(w.model, bid),
                      p - d * 0.2, d)
    assert hit >= 0 and abs(hit - 0.2) < 0.03, "true point not on the surface"
    w._mm[po] = 0.0                       # release
    w.step_sync(0.2)
    assert float(np.abs(w.data.xfrc_applied[bid]).sum()) == 0.0
    w.step_sync(1.0)


def test_estimator_palm_accuracy_and_surface(world):
    """12 N palm push: correct link, honest magnitude (friction-compensated),
    aligned direction, point ON the surface near the true application point."""
    w = world
    w.apply_push("left_palm", [0, 12, 0], seconds=2.5)
    w.step_sync(1.4)
    Fs, Ps, Bs = _steady(w)
    # score the point against the LIVE truth: under a torque-impedance hold the
    # palm DEFLECTS ~10 cm into the push, so the ground-truth application point
    # moves with it (w.push["point"] is recomputed on the deflected surface each
    # tick). Comparing to the pre-deflection point would falsely fail.
    p_true = np.array(w.push["point"])
    assert len(Fs) >= 8, "contact not held during a steady push"
    assert set(Bs) == {"left_palm"}
    Fm, Pm = Fs.mean(0), Ps.mean(0)
    # under a torque-impedance hold the arm deflects (K≤45 is far softer than a
    # position servo), so the observer reads a modestly HIGH magnitude from the
    # deflection dynamics — direction + localization stay sharp.
    assert abs(np.linalg.norm(Fm) - 12.0) < 3.5
    cos = Fm @ [0, 12, 0] / (np.linalg.norm(Fm) * 12.0)
    assert cos > 0.95
    assert np.linalg.norm(Pm - p_true) < 0.08
    # stability: steady-state jitter
    assert np.linalg.norm(Fs.std(0)) < 1.5
    assert np.linalg.norm(Ps.std(0)) < 0.01
    w.step_sync(2.0)


def test_estimator_forearm_not_just_palm(world):
    """Pushes on PROXIMAL links must not vanish (pure-force refit) and must
    localize to the pushed link, on its surface."""
    w = world
    w.apply_push("l_forearm", [0, 10, 0], seconds=2.5)
    w.step_sync(1.4)
    Fs, Ps, Bs = _steady(w)
    p_true = np.array(w.push["point"])         # LIVE point on the deflected link
    assert len(Fs) >= 8
    assert set(Bs) == {"l_forearm"}
    assert abs(np.linalg.norm(Fs.mean(0)) - 10.0) < 3.5
    assert np.linalg.norm(Ps.mean(0) - p_true) < 0.06
    w.step_sync(2.0)


def test_estimator_negative_controls(world):
    """No contact at rest, none from post-push friction hysteresis, none while
    the arm runs a trajectory — the failure modes that made the estimate
    'unstable' before the deadband/hysteresis rework."""
    w = world
    # rest
    Fs, _, _ = _steady(w, seconds=1.0)
    assert len(Fs) == 0, "phantom contact at rest"
    # post-push hysteresis
    w.apply_push("left_palm", [0, 12, 0], seconds=0.8)
    w.step_sync(1.5)
    Fs, _, _ = _steady(w, seconds=1.0)
    assert len(Fs) == 0, "phantom contact from friction hysteresis"
    # moving arm (impedance tracks the injected sine target)
    w.set_trajectory("sine", joint="left_arm_2", amp=0.25, period=2.5)
    w.step_sync(0.8)
    Fs, _, _ = _steady(w, seconds=1.0)
    assert len(Fs) == 0, "phantom contact while tracking a trajectory"
    w.set_trajectory("hold")
    w.step_sync(0.3)
