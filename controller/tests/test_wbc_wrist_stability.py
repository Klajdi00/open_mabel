"""The ORCA wrists must never be driven by the WBC effort channel.

THE BUG (2026-07-19, operator report): "the wrist position when I switch from
drive to task space is not correct — the wrist rotated 180 in the orientation."

The torque law itself is sound: driving the raw engine in-process with the
wrists parked ON their limit holds at every stiffness 0.5-30
(controller/experiments/wbc_wrist_limit.py). The failure belongs to the SPLIT
deployment. Under `sim:bridge` (server/run.sh, and the mirror of the real
robot's gateway/HAL split) the effort is computed from a MIRRORED qpos/qvel and
shipped at SimBridgePlant.CMD_HZ = 120 Hz to a bridge stepping at 500 Hz. That
delay is harmless on an arm link and fatal on the ORCA wrist, whose effective
inertia (2.5e-3 kg m^2) is ~100x smaller: measured on the live stack the wrist
rings up to -118.6 rad/s, tunnels through its MJCF limit inside one substep, and
parks wrapped at ~+3.85 rad (221 deg) — where it stays.

The fix keeps the wrists on their position servo. These tests pin BOTH halves:
the wrists are unmanaged (so the torque LAW never drives them), and the arm
joints are still managed (so the fix did not quietly disarm the torque law).

UPDATED 2026-07-24: unmanaged no longer means untouched. The wrists carry the
shared upper-body gravity feed-forward, because withholding it was the OTHER
operator-visible bug -- they lost their compensation the moment the operator
left Drive and sagged 1.6 deg (test_mode_switch_no_sag.py). The invariant that
actually prevents the ring-up is that nothing VELOCITY-dependent reaches them,
which is what the effort test below now states.
"""
import os

import pytest

from conftest import _load, needs_model

from mabel.whole_body_controller import engine as E

REGIONS = ("arm", "upper_body", "whole_body")
WRISTS = ("Left Wrist", "Right Wrist")


def _wbc(model, data):
    from mabel import MabelTeleop
    return MabelTeleop(model, data).wbc


@needs_model
@pytest.mark.parametrize("region", REGIONS)
def test_wrists_are_never_torque_managed(region):
    """THE REGRESSION. A managed wrist gets an effort every tick, which is the
    input to the delayed-feedback ring-up."""
    model, data = _load()
    managed = list(_wbc(model, data)._managed_for(region))
    leaked = [w for w in WRISTS if w in managed]
    assert not leaked, (
        f"region {region!r} torque-manages {leaked} — across the gateway->bridge "
        "split this rings the wrist up and wraps it past its MJCF limit")


@needs_model
@pytest.mark.parametrize("region", REGIONS)
def test_the_arms_are_still_managed(region):
    """The other half of the invariant: excluding the wrists must not disarm the
    torque law on the joints it exists to control."""
    model, data = _load()
    managed = list(_wbc(model, data)._managed_for(region))
    for side in ("Left", "Right"):
        for i in range(1, 8):
            assert f"{side} Arm {i}" in managed, f"{side} Arm {i} lost from {region!r}"
    if region in ("upper_body", "whole_body"):
        assert "Torso" in managed, f"Torso lost from {region!r}"


@needs_model
def test_wrist_effort_is_velocity_independent_at_any_velocity():
    """THE STABILITY INVARIANT, stated directly: whatever the effort channel
    emits for a wrist must not depend on how fast the wrist is moving.

    Velocity feedback is the ring-up. Across the gateway->bridge hold, a $-D\\qd$
    term is computed from a state one RPC hop old, so it lands out of phase and
    pumps energy into the lightest joint in the robot until it wraps past its
    MJCF limit. A term that is constant in $\\qd$ cannot do that at any lag --- it
    is a bias, and a bias only shifts the equilibrium.

    This used to assert exactly zero, which was stricter than the property above
    and became wrong on 2026-07-24: the wrists now carry the shared upper-body
    GRAVITY feed-forward (mabel/dynamic_models/gravity.py), without which they
    dropped their compensation the moment the operator left Drive and sagged
    1.6 deg -- the mode-switch sag. That feed-forward is evaluated at qd = 0 on
    private scratch state, so it is pure g(q): present, bounded, and identical at
    0 and 50 rad/s, which is what this test now checks.

    Magnitude is bounded too: a real ring-up needs authority comparable to the
    joint's own servo, and g(q) here is ~0.03 N.m."""
    import mujoco
    model, data = _load()
    wbc = _wbc(model, data)
    wbc.set_stiffness(30.0)
    wbc.enter("impedance", "arm")

    dofs = {w: int(model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, w)]) for w in WRISTS}
    seen = {}
    for qd in (0.0, 50.0, -50.0):                # still, then violent both ways
        data.qvel[:] = 0.0
        for d in dofs.values():
            data.qvel[d] = qd
        wbc.step(model.opt.timestep)
        for w, d in dofs.items():
            seen.setdefault(w, {})[qd] = float(data.qfrc_applied[d])

    for w, by_qd in seen.items():
        tau0 = by_qd[0.0]
        assert abs(tau0) < 1.0, (
            f"{w} received {tau0:+.4f} N.m at rest — far above the ~0.03 N.m "
            "gravity term, so something velocity-free is over-driving it")
        for qd, tau in by_qd.items():
            assert abs(tau - tau0) < 1e-9, (
                f"{w} effort changed with velocity: {tau0:+.6f} N.m at rest vs "
                f"{tau:+.6f} at qd={qd} rad/s. A velocity-dependent term on this "
                "joint is the ring-up — it must stay off the effort channel.")


@needs_model
def test_wrist_servos_are_left_gained():
    """`enter()` neutralises the position servo of every joint it manages. The
    wrists are not managed, so their servo must survive — it is now the ONLY
    thing holding them."""
    model, data = _load()
    wbc = _wbc(model, data)
    wbc.enter("impedance", "arm")
    for w in WRISTS:
        aid = model.actuator(w).id
        assert abs(float(model.actuator_gainprm[aid, 0])) > 1e-9, (
            f"{w} servo was neutralised but nothing drives it — the joint would drift")


@needs_model
def test_wrists_hold_their_ref_through_a_mode_switch():
    """End-to-end: the operator's drive -> task/joint switch must leave the
    wrists at their MJCF ref (+/-0.15), not drifting toward 0 and not wrapped."""
    import mujoco
    model, data = _load()
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    teleop.controller.init_hold()
    teleop.controller.settle_done()
    for _ in range(300):
        mujoco.mj_step(model, data)

    teleop.set_control_spec(method="navigation", control_type="impedance",
                            region="whole_body")
    for _ in range(200):
        try:
            teleop.update(None, None, 1.0 / 240.0)
        except Exception:
            pass
        mujoco.mj_step(model, data)

    teleop.set_control_spec(method="wrist", control_type="impedance",
                            region="arm", stiffness=27.0)
    for _ in range(900):
        try:
            teleop.update(None, None, 1.0 / 240.0)
        except Exception:
            pass
        mujoco.mj_step(model, data)

    for name, sign in (("Left Wrist", +1), ("Right Wrist", -1)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q = float(data.qpos[model.jnt_qposadr[jid]])
        lo, hi = model.jnt_range[jid]
        assert lo - 1e-3 <= q <= hi + 1e-3, (
            f"{name} left its MJCF limit [{lo:+.3f},{hi:+.3f}] at {q:+.4f} — the wrap")
        assert abs(q - sign * 0.15) < 0.05, (
            f"{name} settled at {q:+.4f}, expected ~{sign * 0.15:+.2f} (its MJCF ref)")
