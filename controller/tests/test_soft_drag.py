"""DragFollow — SOFT mode as position-based drag-and-hold (operator design).

The law must: hold at the measured pose on engage; follow a real push (fast,
displaced) staying `band` short; FREEZE the instant the push stops; never follow
quasi-static gravity-residual sag (below the velocity gate); and hold through
missing feedback. No torque enters the loop, so there is nothing to self-excite.
"""
from mabel.arm_control.drag_follow import DragFollow, DRAG_DEFAULTS

J = "l_j2"
DT = 0.02


def make(**over):
    return DragFollow((J,), params=over or None)


def run(d, q_fn, qd_fn, seconds, t0=0.0):
    t = t0
    out = {}
    for _ in range(int(seconds / DT)):
        t += DT
        out = d.update({J: q_fn(t)}, {J: qd_fn(t)}, DT, t)
    return out, t


def test_engage_holds_and_noise_never_moves_the_target():
    d = make()
    d.engage({J: 0.5})
    import math
    out, _ = run(d, lambda t: 0.5 + 0.01 * math.sin(20 * t), lambda t: 0.2 * math.cos(20 * t),
                 seconds=10.0)
    # fast-but-tiny wiggle: velocity gate open, displacement gate CLOSED → frozen
    assert abs(out[J] - 0.5) < 1e-9, "sensor wiggle inside the band moved the target"


def test_push_follows_and_release_freezes():
    d = make()
    d.engage({J: 0.0})
    band = DRAG_DEFAULTS["band"]
    # push: the arm moves at 0.5 rad/s for 1 s → the target must chase, band short
    out, t = run(d, lambda t: 0.5 * t, lambda t: 0.5, seconds=1.0)
    assert out[J] > 0.5 - 2.5 * band, f"target did not follow the push: {out[J]}"
    assert out[J] <= 0.5 - band + 1e-6, "target overran the measured position"
    # release: the arm springs back by ~band and stops (qd → 0) → target FROZEN
    frozen = out[J]
    out, t = run(d, lambda tt: frozen, lambda tt: 0.0, seconds=5.0, t0=t)
    assert abs(out[J] - frozen) < 1e-9, "target moved after the push stopped"


def test_quasi_static_sag_never_follows():
    """Gravity-model residual sags the joint SLOWLY (below qd_min): the target
    must hold so the kp spring arrests the sag — the anti-creep guarantee."""
    d = make()
    d.engage({J: 0.0})
    # 0.03 rad/s sag — displaced beyond the band within ~3 s, but always slow
    out, _ = run(d, lambda t: -0.03 * t, lambda t: -0.03, seconds=10.0)
    assert abs(out[J] - 0.0) < 1e-9, \
        f"target followed a quasi-static sag ({out[J]:+.4f}) — creep!"


def test_missing_feedback_holds():
    d = make()
    d.engage({J: 0.3})
    out = d.update({J: None}, {J: None}, DT, 1.0)
    assert out[J] == 0.3, "lost feedback must hold the last target"
    out = d.update({J: 0.4}, {J: None}, DT, 1.1)
    assert out[J] == 0.3, "no velocity feedback → the follow gate must stay shut"


def test_joint_limits_respected():
    d = make()
    d.engage({J: 0.0})
    lim = {J: (-0.5, 0.5)}
    t, q = 0.0, 0.0
    for _ in range(200):
        t += DT
        q += 0.5 * DT                       # pushed steadily toward the stop
        out = d.update({J: q}, {J: 0.5}, DT, t, limits=lim)
    assert out[J] <= 0.5 - DRAG_DEFAULTS["range_margin"] + 1e-9, \
        "target parked past the joint-stop margin"
