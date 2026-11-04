"""Regression guard for the tip-safe motion model.

`mabel_navigation.motion_model` is a thin re-export of the single source of truth,
`mabel.config` (the whole-body controller). These tests assert (1) it really is that
one source -- not a drifting copy -- and (2) the tip-safety invariants the navigation
stack relies on. Skips cleanly if the controller library is not on the path (e.g. a
lint-only CI image without it installed).
"""
import math

import pytest

mm = pytest.importorskip(
    "mabel_navigation.motion_model",
    reason="controller library (mabel) not importable; see motion_model docstring")


def _have_source():
    try:
        import mabel.config  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_source(), reason="mabel.config not importable")
def test_single_source_not_a_copy():
    """The re-exported symbols must BE the controller's, so they cannot drift."""
    import mabel.config as C
    for name in ("safe_base_step", "tip_usage", "com_height", "accel_safe_scale",
                 "TIP_LEVER_FWD", "COM_Z0", "COM_DZDLIFT"):
        assert getattr(mm, name) is getattr(C, name), f"{name} is a copy, not the source"


def test_speed_cap_shrinks_with_lift():
    """A raised lift must lower the tip-safe forward speed cap (CoM gets taller)."""
    v_low = mm.max_speed_dir(0.0)[0]
    v_high = mm.max_speed_dir(0.635)[0]
    assert v_high < v_low, (v_high, v_low)
    assert v_high >= mm.V_SPEED_FLOOR - 1e-9   # never below the always-allowed crawl


def test_step_respects_speed_cap_and_is_finite():
    """A large forward target is clamped to the tip-safe forward ceiling, eventually."""
    v_fwd = mm.max_speed_dir(0.0)[0]
    vx = vy = om = 0.0
    for _ in range(2000):                       # converge from rest at 50 Hz
        vx, vy, om = mm.safe_base_step(vx, vy, om, -10.0, 0.0, 0.0, 0.02, 0.0)
        assert all(math.isfinite(c) for c in (vx, vy, om))
    assert -vx <= v_fwd + 1e-6                  # forward is -X; |vx| within the cap


def test_accel_safe_scale_contract():
    """accel_safe_scale returns a scale in [0,1] that keeps the scaled command's ZMP
    usage within the never-deadlock cap max(1, u0) (u0 = centripetal-only usage)."""
    cases = [(0.6, 0.0, 0.0, +5.0, 0.0, 0.0, 0.635),
             (0.0, 0.3, 0.5, 0.0, -4.0, 2.0, 0.4),
             (-0.5, 0.2, -1.0, -3.0, 1.0, -2.0, 0.2),
             (0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0)]
    for vx, vy, om, ax, ay, aw, z in cases:
        s = mm.accel_safe_scale(vx, vy, om, ax, ay, aw, z)
        assert 0.0 <= s <= 1.0, s
        u0 = mm.tip_usage(vx, vy, om, 0.0, 0.0, 0.0, z)
        cap = max(1.0, u0)
        u = mm.tip_usage(vx, vy, om, s * ax, s * ay, s * aw, z)
        assert u <= cap + 1e-6, (u, cap)
