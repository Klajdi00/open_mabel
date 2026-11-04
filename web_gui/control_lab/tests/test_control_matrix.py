"""Plot contract for the three control-engineering telemetry series the Control
Lab draws ON TOP of main's stock benches:

  * tau / tau_ff  -> total applied joint torque vs the pure-gravity feed-forward
  * tau_sc        -> self-collision "soft wall" barrier torque
  * ee_rot_*_deg  -> end-effector ORIENTATION error (position-only is incomplete)

Stock bench behaviour (the control-mode matrix, wrench estimation, wrist ring-up)
is already covered by test_simworld.py and test_wrench_estimation.py, so this
module guards ONLY the extra plots -- that each is fed finite data by the bench
it belongs to. Written against the current SimWorld API (set_control_spec /
set_torque / set_self_collision); the old bench/layer API these plots first
shipped against was removed when control_lab was rewritten.

DETERMINISTIC: everything is driven by step_sync (sim time), immune to load.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from control_lab.server.simworld import SimWorld           # noqa: E402


@pytest.fixture(scope="module")
def world():
    w = SimWorld()
    w.settle()
    yield w


def _last(w, key):
    """snapshot()['plots'][key] is the LAST (t, value) sample, or None if the
    series was never fed."""
    pt = w.snapshot()["plots"].get(key)
    assert pt is not None, f"plot series {key!r} was never fed"
    return pt[1]


def _finite(w, key):
    v = _last(w, key)
    assert v is not None and np.isfinite(v), f"{key} not finite: {v!r}"
    return v


def test_joint_torque_plots(world):
    """The joint bench feeds tau (total applied) and tau_ff (gravity FF), both
    finite, and the gravity FF carries a real, non-trivial load on the default
    heavy arm test joint (left_arm_2)."""
    world.reset_robot()
    world.set_torque(True)
    world.set_control_spec(method="joint", control_type="impedance",
                           region="upper_body")
    world.step_sync(0.6)
    _finite(world, "tau")
    ff = abs(_finite(world, "tau_ff"))
    assert ff > 0.1, f"gravity FF should carry a real load on left_arm_2 (got {ff})"


def test_self_collision_barrier_plot(world):
    """tau_sc is finite and non-negative: identically 0 with the guard OFF (the
    plot mirrors engine.step's gate, so a persisting-but-disabled guard reads 0)
    and a valid soft-wall magnitude with it ON."""
    world.reset_robot()
    world.set_torque(True)
    world.set_control_spec(method="joint", control_type="compliance",
                           region="upper_body")
    world.set_self_collision(False)
    world.step_sync(0.4)
    assert _finite(world, "tau_sc") == 0.0, "barrier torque must be 0 with guard OFF"
    world.set_self_collision(True)
    world.step_sync(0.4)
    assert _finite(world, "tau_sc") >= 0.0
    world.set_self_collision(False)


def test_wrist_orientation_error_plot(world):
    """The wrist bench feeds ee_rot_{l,r}_deg alongside the position error --
    finite and non-negative."""
    world.reset_robot()
    world.set_torque(True)
    world.set_control_spec(method="wrist", control_type="compliance",
                           region="whole_body")
    world.step_sync(0.6)
    for side in ("l", "r"):
        assert _finite(world, f"ee_rot_{side}_deg") >= 0.0
