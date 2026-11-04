"""Neck-yaw -> base-yaw handoff (operator 2026-10-23: "360 view as I move my head").

Pins the law in `mabel.teleop_engine.neck_handoff`: no base motion inside the neck's
range, a bounded turn that closes on MEASURED chassis yaw outside it, and a latch-off
when the chassis does not respond (stale odometry must never spin the base forever).
"""
import math

from mabel.teleop_engine.neck_handoff import NeckYawHandoff

H = math.radians(80)


def make():
    return NeckYawHandoff(H, gain=2.0, wmax=0.8, stall=math.radians(45))


def test_inside_neck_range_never_moves_the_base():
    h = make()
    for deg in (-80, -45, 0, 30, 79.9):
        h.observe(math.radians(deg))
        assert h.rate(0.0, 0.01) == 0.0


def test_look_past_the_limit_turns_the_base_the_same_way():
    h = make()
    h.observe(math.radians(120))           # looking left, past the limit
    assert h.rate(0.0, 0.01) > 0           # CCW (+) = left, same sense as neck yaw
    h = make()
    h.observe(math.radians(-120))
    assert h.rate(0.0, 0.01) < 0


def test_rate_is_capped():
    h = make()
    h.observe(math.radians(179))
    assert h.rate(0.0, 0.01) == 0.8


def test_closed_loop_settles_with_the_neck_back_in_range():
    """Operator holds the head 150 deg left; an ideal base integrates the rate and the
    neck yaw (= head - chassis, as neck_angles computes it) returns to the handoff."""
    h, chassis, head, dt = make(), 0.0, math.radians(150), 0.01
    for _ in range(2000):
        h.observe(head - chassis)
        chassis += h.rate(chassis, dt) * dt
    assert not h.stalled
    assert abs(math.degrees(head - chassis) - 80) < 0.5


def test_full_circle_the_base_follows_past_180():
    """A slow 360 deg turn of the operator: the unwrapped stall bookkeeping must not
    false-trip when the chassis yaw wraps through +-180."""
    h, chassis, dt = make(), 0.0, 0.01
    for k in range(3000):                  # 30 s, head turns 2*pi + 80 deg
        head = (k / 3000) * (2 * math.pi + math.radians(80))
        h.observe(_wrap(head - chassis))
        chassis = _wrap(chassis + h.rate(chassis, dt) * dt)
    assert not h.stalled


def test_stale_odometry_latches_off_and_reset_rearms():
    h = make()
    h.observe(math.radians(150))
    stopped_at = None
    for k in range(1000):                  # chassis yaw never updates
        if h.rate(0.0, 0.01) == 0.0:
            stopped_at = k
            break
    assert h.stalled and stopped_at is not None
    assert stopped_at * 0.01 < 1.5         # 45 deg at 0.8 rad/s ~ 1 s
    h.observe(math.radians(150))
    assert h.rate(0.0, 0.01) == 0.0        # stays off
    h.reset()
    h.observe(math.radians(150))
    assert h.rate(0.0, 0.01) > 0


def test_not_live_or_untracked_neck_commands_nothing():
    h = make()
    h.observe(math.radians(150))
    assert h.rate(0.0, 0.01, live=False) == 0.0
    h.observe(None)
    assert h.rate(0.0, 0.01) == 0.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
