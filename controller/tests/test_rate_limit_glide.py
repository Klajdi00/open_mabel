"""JointGlideLimiter — the green-ball smoothness fix (2026-07-29).

Pins the four properties that matter:
  * a FAR target (task-space Execute) glides at blend speed and takes 2-3 s;
  * the command stream is SMOOTH (velocity-continuous, no chatter) even with a
    noisy, lagging measured signal — the failure mode of the measured-anchored
    JointRateLimiter this replaces;
  * the measured-error torque bound (kp*err_max) still holds at all times;
  * small nudges (sliders, eased app bursts) keep full tracking speed.
"""
import math

import pytest

from mabel.arm_control import JointGlideLimiter

DT = 0.02          # the ~50 Hz control tick
J = "Left Arm 2"


def run(lim, target, measured_fn, n, dt=DT):
    """Drive the limiter n ticks; measured_fn(cmd_prev, k) simulates the arm."""
    cmds, cmd = [], None
    for k in range(n):
        cur = measured_fn(cmd, k)
        cmd = lim.step({J: target}, {J: cur}, dt=dt)[J]
        cmds.append(cmd)
    return cmds


def lagging_arm(lag=0.9):
    """First-order arm model: measured chases the command with lag."""
    state = {"q": 0.0}
    def fn(cmd_prev, k):
        if cmd_prev is not None:
            state["q"] += (1 - lag) * (cmd_prev - state["q"])
        return state["q"]
    return fn


def test_far_target_takes_2_to_3_seconds():
    # lag=0.7 ≈ a real arm with velocity+gravity FF (tracking error ~0.06 rad at
    # 1 rad/s — inside err_max, so the torque bound never binds mid-glide).
    lim = JointGlideLimiter()
    cmds = run(lim, 2.5, lagging_arm(lag=0.7), n=int(6.0 / DT))
    # arrival = first tick within 1% of target
    arrive = next(i for i, c in enumerate(cmds) if abs(c - 2.5) < 0.025)
    t = arrive * DT
    assert 2.0 <= t <= 3.2, f"2.5 rad task move arrived in {t:.2f}s (want ~2.5s)"


def test_command_stream_is_smooth_under_measurement_noise():
    """THE regression: with the old measured-anchored limiter, encoder noise
    modulated the command every tick (chatter). The glide must stay monotone
    mid-travel regardless of noise on the measured signal."""
    lim = JointGlideLimiter()
    arm = lagging_arm(lag=0.7)
    noisy = lambda cmd_prev, k: arm(cmd_prev, k) + 0.008 * math.sin(k * 2.1)
    cmds = run(lim, 2.0, noisy, n=int(1.5 / DT))
    deltas = [b - a for a, b in zip(cmds[5:-1], cmds[6:])]   # mid-travel
    assert all(d >= -1e-9 for d in deltas), "command reversed mid-glide (chatter)"
    # velocity continuity: per-tick delta never jumps more than max_accel*dt^2
    jerk = [abs(b - a) for a, b in zip(deltas, deltas[1:])]
    assert max(jerk) <= lim.max_accel * DT * DT + 1e-9, \
        "velocity discontinuity mid-glide (the FF differentiates this into a spike)"


def test_jammed_arm_ceiling_then_thermal_yield():
    """A target gliding away from a frozen arm binds INSTANTLY at the hard
    ceiling (err_max), then the THERMAL YIELD walks it down to the gentle hold —
    the two-stage jammed-arm guarantee."""
    lim = JointGlideLimiter()
    n_ceiling = int(1.5 / DT)
    cmds = run(lim, 3.0, lambda cp, k: 0.0, n=n_ceiling)
    assert max(cmds) <= lim.err_max + 1e-9, "ceiling did not bind"
    seconds = lim.yield_after + (lim.err_max - lim.yield_rest) / lim.yield_rate + 1.0
    cmds = run(lim, 3.0, lambda cp, k: 0.0, n=int(seconds / DT))
    assert cmds[-1] <= lim.yield_rest + 1e-6, \
        f"jammed-arm error not yielded: {cmds[-1]:.3f} rad sustained"


def test_optional_hard_ceiling_still_works():
    lim = JointGlideLimiter(err_max=0.45)
    cmds = run(lim, 3.0, lambda cp, k: 0.0, n=50)            # 1 s: yield not yet
    assert max(cmds) <= 0.45 + 1e-9


def test_small_nudges_keep_full_tracking_speed():
    lim = JointGlideLimiter()
    arm = lagging_arm(lag=0.5)
    cmds = run(lim, 0.10, arm, n=40)
    arrive = next(i for i, c in enumerate(cmds) if abs(c - 0.10) < 0.005)
    assert arrive * DT < 0.4, "a 0.1 rad slider nudge must stay snappy"


def test_reset_and_stall_reseed_from_measured():
    lim = JointGlideLimiter()
    run(lim, 2.0, lagging_arm(), n=20)
    lim.reset()
    # after reset the first command holds where the arm IS (no snap to old glide)
    cmd = lim.step({J: 2.0}, {J: 0.3}, dt=DT)[J]
    assert abs(cmd - 0.3) <= lim.max_speed * DT + 1e-9


# ── the STIFF spring + thermal yield (2026-07-30) ────────────────────────────
def test_push_meets_a_spring_not_a_drag():
    """Pushing the arm away from a HELD target must NOT drag the target along:
    resistance keeps growing with displacement (kp·err), and release springs
    back to the original target — the old 0.15 drag-clamp saturated at
    ~3.75 N·m then yielded plastically ('torque drop after a certain region')."""
    lim = JointGlideLimiter()
    for _ in range(10):
        lim.step({J: 0.0}, {J: 0.0}, dt=DT)
    q = 0.0
    for _ in range(35):                      # push the ARM to 0.35 rad in 0.7 s
        q += 0.01
        cmd = lim.step({J: 0.0}, {J: q}, dt=DT)[J]
    assert abs(cmd - 0.0) < 1e-9, f"the held target was dragged to {cmd}"
    cmd = lim.step({J: 0.0}, {J: 0.0}, dt=DT)[J]     # release → full spring-back
    assert abs(cmd) < 1e-9


def test_spring_holds_inside_the_ceiling_and_binds_beyond():
    # inside the 0.8 ceiling a shove meets the full spring — target unmoved
    lim = JointGlideLimiter()
    for _ in range(10):
        lim.step({J: 0.0}, {J: 0.0}, dt=DT)
    cmd = lim.step({J: 0.0}, {J: 0.7}, dt=DT)[J]
    assert abs(cmd) < 1e-9, "the held target moved inside the ceiling"
    # beyond it, the ceiling binds exactly
    cmd = lim.step({J: 0.0}, {J: 1.2}, dt=DT)[J]
    assert abs(cmd - 1.2) <= lim.err_max + 1e-9


def test_sustained_force_yields_then_recovers():
    """Leaning on the arm past yield_err for yield_after seconds must make the
    setpoint give way to the gentle hold (yield_rest) — sustained near-rating
    torque protection — and the spring must RESTORE once the arm catches up."""
    lim = JointGlideLimiter()
    for _ in range(10):
        lim.step({J: 0.0}, {J: 0.0}, dt=DT)
    cmd = 0.0
    for _ in range(int(4.0 / DT)):           # lean on it, displaced 0.4 rad, 4 s
        cmd = lim.step({J: 0.0}, {J: 0.4}, dt=DT)[J]
    err = abs(cmd - 0.4)
    assert err <= lim.yield_rest + 1e-6, \
        f"no thermal yield: still {err:.3f} rad of sustained error"
    # hand-off: the operator releases and the target is re-seated at the arm
    # (resume_hold semantics) → error collapses → the yield latch releases
    hold = cmd
    for _ in range(20):
        cmd = lim.step({J: hold}, {J: hold}, dt=DT)[J]
    assert J not in lim._yielded, "yield latch never released after catching up"
    # a fresh quick push must meet the FULL spring again (no dragging)
    q = hold
    for _ in range(15):
        q += 0.01
        cmd2 = lim.step({J: hold}, {J: q}, dt=DT)[J]
    assert abs(cmd2 - hold) < 1e-6, "spring did not restore after the yield"
