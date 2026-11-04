"""GravityFFScheduler — the pose-triggered gravity-FF evaluation gate.

Pins the timing semantics that fixed the 2026-07-30 "step-like motion when I
push the arm in stiff" report: a moving pose re-evaluates promptly (bounded by
min_period), a resting pose stays on the cheap drift-safety clock, and the
motion trigger keys off the pose the CACHED evaluation used, not the previous
tick (creep across many sub-eps ticks must still trigger).
"""
from mabel.arm_control.ff_schedule import GravityFFScheduler


def test_max_period_always_fires():
    s = GravityFFScheduler(min_period=0.04, max_period=0.25, pose_eps=0.01)
    s.mark(0.0, {"l_j2": 0.0})
    assert not s.due(0.24, {"l_j2": 0.0})
    assert s.due(0.26, {"l_j2": 0.0}), "drift-safety clock did not fire"


def test_min_period_blocks_even_large_motion():
    s = GravityFFScheduler(min_period=0.04, max_period=0.25, pose_eps=0.01)
    s.mark(0.0, {"l_j2": 0.0})
    assert not s.due(0.02, {"l_j2": 1.0}), "min_period must bound the eval rate"
    assert s.due(0.05, {"l_j2": 1.0})


def test_pose_motion_triggers_between_clocks():
    s = GravityFFScheduler(min_period=0.04, max_period=0.25, pose_eps=0.01)
    s.mark(0.0, {"l_j2": 0.0, "l_j4": 0.5})
    assert not s.due(0.1, {"l_j2": 0.005, "l_j4": 0.5}), "sub-eps motion fired"
    assert s.due(0.1, {"l_j2": 0.02, "l_j4": 0.5}), "eps motion did not fire"
    assert s.due(0.1, {"l_j2": 0.0, "l_j4": 0.48}), "second joint's motion ignored"


def test_trigger_is_against_the_cached_pose_not_the_last_tick():
    """A slow push moves < eps per tick but far over many ticks — the trigger
    compares against the pose of the cached EVALUATION, so creep accumulates."""
    s = GravityFFScheduler(min_period=0.04, max_period=10.0, pose_eps=0.01)
    s.mark(0.0, {"l_j2": 0.0})
    q, t = 0.0, 0.0
    fired_at = None
    for _ in range(50):
        t += 0.02
        q += 0.002                       # 0.1 rad/s — sub-eps per tick
        if s.due(t, {"l_j2": q}):
            fired_at = q
            break
    assert fired_at is not None and fired_at <= 0.015, \
        f"slow creep never (or too late) triggered: {fired_at}"


def test_new_joint_in_pose_triggers():
    s = GravityFFScheduler(min_period=0.0, max_period=10.0, pose_eps=0.01)
    s.mark(0.0, {"l_j2": 0.0})
    assert s.due(0.1, {"l_j2": 0.0, "torso": -0.2}), \
        "a joint absent from the cached pose must force a re-eval"
