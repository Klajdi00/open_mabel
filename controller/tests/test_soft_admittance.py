"""SoftArmAdmittance law-level tests (mock gravity model — fast, no MuJoCo).

The floating soft mode must: never drift under model error (entry tare +
deadband + quiet re-tare), follow a push with the right sign, coast to rest and
PARK on release, respect the err_max/vmax/range rails, and hold any joint whose
feedback goes stale. The MuJoCo end-to-end (emulated MIT loop, deliberate
mass/CoM mismatch) lives in experiments/exp_soft_admittance.py.
"""
import math

import pytest

from mabel.arm_control.admittance import SoftArmAdmittance, FLOAT_DEFAULTS


class MockGravity:
    """τ_g(q) = A·sin(q) per joint — a stand-in gravity law the tests control."""

    def __init__(self, amp=8.0):
        self.amp = amp

    def tff(self, pose, frame="model"):
        assert frame == "model"
        return {j: self.amp * math.sin(v) for j, v in pose.items()}


J = ("l_j2",)


def make(**over):
    p = dict(over)
    adm = SoftArmAdmittance(MockGravity(), J, params=p)
    return adm


def run(adm, q, tau_meas_fn, seconds, dt=0.02, t0=0.0, qd=None, track=True):
    """Iterate update() with τ from a callable. ``track=True`` makes the plant a
    perfect servo (q follows the streamed target each tick) — physically
    faithful, and required by the creep-kill discriminator, which classifies a
    target that leads a NON-moving arm as residual creep (that is the point).
    ``track=False`` emulates a physically STUCK joint."""
    t = t0
    out = {}
    n = int(seconds / dt)
    for _ in range(n):
        t += dt
        tau = {j: tau_meas_fn(j, t, q) for j in J}
        out = adm.update(dict(q), tau, dt, t, qd_meas=qd)
        if track:
            for j, v in out.items():
                q[j] = v
    return out, t


def test_entry_tare_absorbs_constant_model_error():
    """Imperfect sysid: the measured hold torque differs from the model by a
    constant 3 N·m at the engage pose. The ENTRY TARE must zero it — no drift,
    ever — even though 3 N·m is far above the deadband."""
    adm = make()
    g = MockGravity()
    q = {"l_j2": 0.5}
    adm.engage(q, now=0.0)                   # motor holds 3 N·m off the model:
    out, _ = run(adm, q, lambda j, t, qq: g.tff(qq)[j] - 3.0, seconds=30.0)
    tgt = adm.state()["targets"]["l_j2"]
    assert abs(tgt - 0.5) < 1e-6, f"drifted {math.degrees(tgt-0.5):.3f}° under model error"


def servo_run(adm, g, st, seconds, resid=0.0, tau_ext=0.0,
              dt=0.02, t0=0.0, kp=25.0, c=2.0):
    """1-DOF micro servo plant — the honest fixture for push-vs-creep dynamics:

        c·q̇ = kp(q* − q) + resid + τ_ext        (massless joint, viscous c)
        τ_meas = kp(q* − q) + g_model(q)          (what the motor reports)

    Under a PUSH the arm settles BEYOND the target in the push direction (the
    arm leads — the creep-kill must stay silent); under a RESIDUAL the target
    walks and the arm chases from behind (the target leads — creep-kill fires).
    """
    t = t0
    for _ in range(int(seconds / dt)):
        t += dt
        tgt = adm.state()["targets"].get("l_j2", st["q"])
        qdot = (kp * (tgt - st["q"]) + resid + tau_ext) / c
        st["q"] += qdot * dt
        tau_m = kp * (tgt - st["q"]) + g.tff({"l_j2": st["q"]})["l_j2"]
        adm.update({"l_j2": st["q"]}, {"l_j2": tau_m}, dt, t,
                   qd_meas={"l_j2": qdot})
    return t


def test_push_moves_with_the_right_sign_and_release_parks():
    adm = make()
    g = MockGravity()
    st = {"q": 0.5}
    adm.engage({"l_j2": 0.5}, now=0.0)
    t = servo_run(adm, g, st, 1.0, resid=1.0)             # settle (entry tare)
    q_settled = st["q"]

    # steady DOWNWARD push: the arm must float down substantially
    t = servo_run(adm, g, st, 2.0, resid=1.0, tau_ext=-4.0, t0=t)
    assert st["q"] < q_settled - 0.15, \
        f"arm did not float under a 4 N·m push (q {st['q']:.3f})"

    # release: short coast, then PARK — no spring-back toward the start
    q_rel = st["q"]
    t = servo_run(adm, g, st, 1.5, resid=1.0, t0=t)
    q_1s = st["q"]
    t = servo_run(adm, g, st, 2.0, resid=1.0, t0=t)
    assert abs(st["q"] - q_1s) < 0.02, "arm did not PARK after release (creep)"
    assert st["q"] < q_settled - 0.10, "arm sprang back toward the pre-drag pose"


def test_below_deadband_never_moves():
    adm = make()
    g = MockGravity()
    q = {"l_j2": 0.3}
    adm.engage(q, now=0.0)
    db = FLOAT_DEFAULTS["deadband_nm"]["j2"]
    _, _ = run(adm, q, lambda j, t, qq: g.tff(qq)[j] + 0.8 * db, seconds=20.0)
    tgt = adm.state()["targets"]["l_j2"]
    assert abs(tgt - 0.3) < 1e-6, "sub-deadband torque moved the target"


def test_pose_residual_creep_is_budget_bounded():
    """A residual above the deadband (a pose far from the last tare, imperfect
    sysid) is INDISTINGUISHABLE from a push at this sensing level — so the law
    does not try: the FLOAT BUDGET bounds the excursion of any uninterrupted
    out-of-band episode, and the arm then freezes until the estimate re-enters
    the band (a residual never eases; a human hand does)."""
    adm = make()
    g = MockGravity()
    st = {"q": 0.5}
    adm.engage({"l_j2": 0.5}, now=0.0)
    t = servo_run(adm, g, st, 1.0, resid=0.0)             # clean settle
    # the residual jumps to 2.2 N·m (>> deadband) with NOBODY pushing
    t = servo_run(adm, g, st, 6.0, resid=2.2, t0=t)
    q_a = st["q"]
    t = servo_run(adm, g, st, 4.0, resid=2.2, t0=t)
    budget = FLOAT_DEFAULTS["episode_budget_rad"]
    assert abs(st["q"] - q_a) < 0.02, \
        f"crept past the budget freeze ({math.degrees(abs(st['q']-q_a)):.1f}° in 4 s)"
    # the budget bounds the TARGET; the plant adds its elastic deflection
    # (resid/kp) plus the LP tail on top — allow that physical margin.
    assert abs(st["q"] - 0.5) <= budget + 0.25, \
        f"excursion {abs(st['q']-0.5):.2f} rad exceeds the episode budget {budget}"


def test_err_max_bounds_target_runaway():
    """A stuck joint (q frozen) under a huge sustained push: the target may lead
    the measured pose by at most err_max — kp·error, hence torque, is bounded."""
    adm = make()
    g = MockGravity()
    q = {"l_j2": 0.0}
    adm.engage(q, now=0.0)
    run(adm, q, lambda j, t, qq: g.tff(qq)[j] + 30.0, seconds=5.0, track=False)
    tgt = adm.state()["targets"]["l_j2"]
    assert abs(tgt - 0.0) <= FLOAT_DEFAULTS["err_max"] + 1e-9, \
        f"target ran {tgt} rad away from a stuck joint"


def test_stale_feedback_holds():
    adm = make()
    q = {"l_j2": 0.5}
    adm.engage(q, now=0.0)
    out = adm.update({"l_j2": 0.5}, {"l_j2": None}, 0.02, 1.0)
    assert out == {}, "a joint with no τ feedback must be omitted (= hold)"
    out = adm.update({"l_j2": None}, {"l_j2": 1.0}, 0.02, 1.1)
    assert out == {}, "a joint with no position feedback must be omitted"


def test_vmax_caps_target_speed():
    adm = make()
    g = MockGravity()
    q = {"l_j2": 0.0}
    adm.engage(q, now=0.0)
    tau_hold = g.tff(q)["l_j2"]
    dt = 0.02
    prev = 0.0
    t = 0.0
    worst = 0.0
    for _ in range(100):
        t += dt
        # keep q tracking the target so err_max never clamps (isolates vmax)
        qq = {"l_j2": adm.state()["targets"]["l_j2"]}
        tau = {"l_j2": g.tff(qq)["l_j2"] + 25.0}
        out = adm.update(qq, tau, dt, t)
        cur = out.get("l_j2", prev)
        worst = max(worst, abs(cur - prev) / dt)
        prev = cur
    assert worst <= FLOAT_DEFAULTS["vmax"] + 1e-6, f"target speed {worst} > vmax"
