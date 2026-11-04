"""Analytic invariants of the tip-over safety motion model (NumPy-only).

These are the properties the physical V1/V2 validation (exp_wbc_validation)
established; here they run in milliseconds on every test invocation:
  1. a stop command is NEVER throttled,
  2. usage is degree-2 homogeneous in a steady twist (the closed-form projection),
  3. speed/decel caps shrink monotonically as the lift raises the CoM,
  4. safe_base_step keeps the ZMP contract u <= max(1, u0) along random sweeps,
  5. an over-limit state recovers (no dead-lock) and a braking command passes.
"""
import numpy as np

from mabel.motion_model import tip_over as T

RNG = np.random.default_rng(7)
LIFTS = [0.0, 0.15, 0.3, 0.45, 0.635]


def test_stop_never_throttled():
    for z in LIFTS:
        assert T.tip_speed_scale(0.0, 0.0, 0.0, z) == 1.0


def test_usage_homogeneity_in_twist():
    for _ in range(200):
        vx, vy, w = RNG.uniform(-1.5, 1.5, 3)
        z = RNG.uniform(0, 0.635)
        u1 = T.tip_usage(vx, vy, w, 0, 0, 0, z)
        s = 0.5
        u2 = T.tip_usage(s * vx, s * vy, s * w, 0, 0, 0, z)
        assert abs(u2 - s ** 2 * u1) < 1e-9 * max(1.0, u1)


def test_speed_scale_projects_into_envelope():
    for _ in range(200):
        vx, vy, w = RNG.uniform(-2.0, 2.0, 3)
        z = RNG.uniform(0, 0.635)
        s = T.tip_speed_scale(vx, vy, w, z)
        assert 0.0 < s <= 1.0
        assert T.tip_usage(s * vx, s * vy, s * w, 0, 0, 0, z) <= 1.0 + 1e-9


def test_caps_monotone_nonincreasing_with_lift():
    zs = np.linspace(0, 0.635, 12)
    speeds = np.array([T.max_speed_dir(z) for z in zs])
    decels = np.array([T.max_decel_dir(z) for z in zs])
    for k in range(3):
        assert np.all(np.diff(speeds[:, k]) <= 1e-12)
        assert np.all(np.diff(decels[:, k]) <= 1e-12)
    # and the speed floor is honoured even fully raised
    assert min(T.max_speed_dir(0.635)) >= T.V_SPEED_FLOOR - 1e-12


def test_safe_base_step_zmp_contract_random_sweeps():
    dt = 0.01
    for _ in range(30):
        z = RNG.uniform(0, 0.635)
        vx = vy = w = 0.0
        for _ in range(300):
            tvx, tvy, tw = RNG.uniform(-2, 2, 3)
            u0 = T.tip_usage(vx, vy, w, 0, 0, 0, z)
            nvx, nvy, nw = T.safe_base_step(vx, vy, w, tvx, tvy, tw, dt, z)
            ax, ay, aw = (nvx - vx) / dt, (nvy - vy) / dt, (nw - w) / dt
            u = T.tip_usage(vx, vy, w, ax, ay, aw, z)
            assert u <= max(1.0, u0) + 1e-6
            vx, vy, w = nvx, nvy, nw
            v_fwd, v_back, v_lat = T.max_speed_dir(z)
            assert -v_fwd - 1e-9 <= vx <= v_back + 1e-9
            assert abs(vy) <= v_lat + 1e-9


def test_overlimit_state_recovers_no_deadlock():
    # The validated liveness claim (V1 of exp_wbc_validation): from the
    # over-limit twist (0.5, 0.8, 1.8) at FULL lift, a stop command converges.
    z = T.LIFT_RANGE[1]
    vx, vy, w = 0.5, 0.8, 1.8
    dt = 0.01
    t_stop = None
    for i in range(int(6.0 / dt)):
        vx, vy, w = T.safe_base_step(vx, vy, w, 0.0, 0.0, 0.0, dt, z)
        if abs(vx) < 0.01 and abs(vy) < 0.01 and abs(w) < 0.01:
            t_stop = i * dt
            break
    assert t_stop is not None and t_stop < 3.0, "stop from over-limit twist dead-locked"


import pytest


@pytest.mark.xfail(reason="KNOWN LIMITATION: for an INJECTED twist far outside "
                   "the envelope (fast spin + fast translation, u0 >> 1) every "
                   "braking direction transiently raises the worst-edge ZMP "
                   "usage, so accel_safe_scale freezes the state instead of "
                   "decelerating. Unreachable through safe_base_step itself "
                   "(velocities only build inside the envelope), but a raised "
                   "lift shrinking the envelope under a fast twist moves toward "
                   "this region. Candidate fix: allow a small (~2%) transient "
                   "usage excess for strictly-decelerating commands (liveness "
                   "tolerance) — needs re-validation of V1/V2 before shipping.",
                   strict=True)
def test_extreme_overlimit_state_currently_freezes():
    z = T.LIFT_RANGE[1]
    vx, vy, w = 1.2, 1.0, 2.0            # far outside: u0 >> 1
    dt = 0.01
    for i in range(int(6.0 / dt)):
        vx, vy, w = T.safe_base_step(vx, vy, w, 0.0, 0.0, 0.0, dt, z)
        if abs(vx) < 0.01 and abs(vy) < 0.01 and abs(w) < 0.01:
            return                        # recovered -> xfail(strict) will flag
    raise AssertionError("extreme over-limit state froze (documented limitation)")


def test_braking_command_passes_unscaled():
    # Braking lowers usage -> accel_safe_scale must not scale it.
    z = 0.3
    vx, vy, w = 0.5, 0.0, 0.0
    s = T.accel_safe_scale(vx, vy, w, -2.0, 0.0, 0.0, z)
    assert s == 1.0


def test_com_height_clamps_to_lift_range():
    assert T.com_height(-1.0) == T.com_height(0.0)
    assert T.com_height(9.0) == T.com_height(T.LIFT_RANGE[1])
    assert T.com_height(0.3) == T.COM_Z0 + T.COM_DZDLIFT * 0.3
