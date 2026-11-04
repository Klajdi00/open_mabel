"""SimWorld control-model tests — the lab as a mini-gateway around MabelTeleop,
mirroring the app's four axes: TORQUE on/off · STIFF/SOFT · REGION · MODE.

Every control tick drives the SAME facade the iOS / Vision Pro apps drive, so
these lock in: boot-limp → torque holds the current pose (no lurch); the
control_mode matrix enters the right facade state and stays STABLE (the wrist
discrete-damping regression); joint/wrist tracking; drive signs + odometry; Soft
wrench estimation; and whole-body retarget from a mock operator."""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from control_lab.server.simworld import (SimWorld, METHODS, CONTROL_TYPES,   # noqa: E402
                                         REGIONS, WBC_METHODS)
from control_lab.server.vpnode import VPNode                     # noqa: E402


@pytest.fixture(scope="module")
def world():
    """Deterministic world: step_sync-driven (sim time, load-immune)."""
    w = SimWorld()
    w.settle()
    return w


@pytest.fixture(scope="module")
def rtworld():
    """Realtime world — only for the tests that need the VPNode threads."""
    w = SimWorld()
    w.start()
    yield w
    w.stop()


def _latest(w, key):
    p = w.snapshot()["plots"].get(key)
    return None if p is None else p[1]


def _healthy(w):
    s = w.snapshot()
    assert not s["status"].startswith(("control error", "plot error")), s["status"]
    assert np.isfinite(list(s["base"].values())).all()
    assert np.isfinite(w.data.qvel).all() and np.abs(w.data.qvel).max() < 50, \
        "physics diverged"
    return s


# ── torque master gate ─────────────────────────────────────────────────────
def test_boot_limp_then_torque_holds(world):
    # (the construction default method=navigation/region=arm is checked with a
    # FRESH instance in test_app::test_ws_contract; reset_robot preserves the last
    # spec by design — re-home keeps the control mode — so don't assert it here.)
    world.reset_robot()
    s = world.snapshot()
    assert s["torque"] is False, "reset returns to LIMP"
    # limp is stable (no explosion), arms float
    world.step_sync(0.4)
    _healthy(world)
    # ENABLE TORQUE holds the CURRENT pose (resume_hold: command := measured)
    q_before = {j: world.plant.q(j) for j in ("Left Arm 2", "Right Arm 2", "Torso")}
    world.set_torque(True)
    world.step_sync(0.6)
    s = world.snapshot()
    assert s["torque"] is True
    drift = max(abs(world.plant.q(j) - q0) for j, q0 in q_before.items())
    assert drift < 0.03, f"torque enable lurched the arm ({drift:.3f} rad)"
    # E-STOP → back to limp
    world.set_torque(False)
    world.step_sync(0.2)
    assert world.snapshot()["torque"] is False
    _healthy(world)


def test_limp_gates_control(world):
    """No control input takes effect while limp (torque off)."""
    world.reset_robot()                        # limp
    world.set_control_spec(method="joint", control_type="impedance", region="arm")
    q0 = world.plant.q("Left Arm 2")
    world.set_joint_targets({"left_arm_2": q0 + 0.5})
    world.step_sync(0.6)
    assert abs(world.plant.q("Left Arm 2") - q0) < 0.05, \
        "a joint target moved the arm while torque was OFF"
    assert not world.teleop.wbc.active, "WBC must not engage while limp"


# ── control_mode matrix + the wrist discrete-damping REGRESSION ─────────────
@pytest.mark.parametrize("method", ["navigation", "joint", "wrist"])
@pytest.mark.parametrize("ctype", CONTROL_TYPES)
@pytest.mark.parametrize("region", REGIONS)
def test_control_mode_matrix_stable(world, method, ctype, region):
    world.reset_robot()
    world.set_torque(True)
    # high stiffness on the whole-body region is the worst case for the wrist
    # discrete-damping instability that the WBC cap fixes — exercise it.
    world.set_control_spec(method=method, control_type=ctype, region=region,
                           stiffness=45.0)
    world.step_sync(1.0)
    _healthy(world)
    # Stiff (impedance) = position servos (no WBC); only Soft (compliance) engages
    # the WBC torque engine for the joint/wrist methods.
    engages_wbc = (method in WBC_METHODS and ctype == "compliance")
    assert world.teleop.wbc.active == engages_wbc
    # the wrist must never ring up (position servos are rigid; WBC has the damping cap)
    assert abs(world.plant.qd("Right Wrist")) < 5.0
    world.set_torque(False)


# ── drive ───────────────────────────────────────────────────────────────────
def test_drive_signs_and_odometry(world):
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="navigation")
    world.nav.update({"lx": 0.0, "ly": 1.0, "rx": 0.0, "ry": 0.0})   # stick UP
    world.step_sync(1.2)
    assert world._last_base[0] < -0.1, \
        f"stick-up must command a forward (−X) twist, got vx={world._last_base[0]}"
    assert _latest(world, "tip_usage") is not None
    err = _latest(world, "odom_err")
    assert err is not None and err < 0.3, f"command odometry drifted {err} m"
    # strafe LEFT (wire lx>0) → vy<0
    world.nav.update({"lx": 0.9, "ly": 0.0, "rx": 0.0, "ry": 0.0})
    world.step_sync(0.5)
    assert world._last_base[1] < -0.05, f"stick-left must strafe left, vy={world._last_base[1]}"
    world.nav.update({"lx": 0, "ly": 0, "rx": 0, "ry": 0})
    # yaw LEFT (wire rx>0) → CCW, base yaw increases (rotation works in sim)
    world.reset_robot(); world.set_torque(True); world.set_control_spec(method="navigation")
    world.nav.update({"rx": 0.9}); world.step_sync(1.2); world.nav.update({"rx": 0.0})
    assert world.snapshot()["base"]["yaw"] > 0.05, "wire rx>0 must yaw LEFT (CCW)"
    world.reset_robot()


def test_drive_ry_jogs_the_lift(world):
    """Drive mode's RIGHT stick Y must jog the lift up AND down (operator ask:
    'the right joystick should toggle up down of the lift'). Rides the engine's
    _apply_stick_drone jog — the same axis the apps' nav stick uses.

    REALTIME-PACED: the TeleopEngine + EngagementSupervisor run on the WALL
    clock (they are realtime controllers — the live server paces wall≈sim and
    the jog runs at the full 0.20 m/s there). Fast-forwarded step_sync starves
    the wall-clock frame_dt and the 2.5 s engage ease, so this test holds
    wall≈sim explicitly; thresholds stay generous for loaded CI machines."""
    def rt_step(secs):
        t0 = time.time()
        stepped = 0.0
        while stepped < secs:
            world.step_sync(0.05)
            stepped += 0.05
            lag = stepped - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)

    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="navigation")
    rt_step(2.5)                                   # let the engage ease reach s=1
    lift0 = world.snapshot()["lift"]
    world.nav.update({"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 1.0})
    rt_step(1.5)
    world.nav.update({"ry": 0.0})
    up = world.snapshot()["lift"]
    assert up - lift0 > 0.12, f"ry up must raise the lift (Δ={up - lift0:.3f} m)"
    world.nav.update({"ry": -1.0})
    rt_step(1.0)
    world.nav.update({"ry": 0.0})
    down = world.snapshot()["lift"]
    assert up - down > 0.08, f"ry down must lower the lift (Δ={up - down:.3f} m)"
    world.reset_robot()


def test_drive_speed_capped_by_lift_height(world):
    """LIFT-HEIGHT AWARENESS: the tip-over motion model must cap the realized
    forward speed harder as the lift rises (max_speed_dir: fwd 0.80 m/s lowered
    → 0.47 m/s fully raised). Full stick forward at a raised lift must therefore
    run measurably slower than at the lowered lift."""
    def top_speed():
        world.nav.update({"lx": 0.0, "ly": 1.0, "rx": 0.0, "ry": 0.0})
        peak = 0.0
        for _ in range(10):
            world.step_sync(0.3)
            peak = max(peak, abs(world._last_base[0]))
        world.nav.update({"ly": 0.0})
        world.step_sync(1.5)                       # ramp back to rest
        return peak

    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="navigation")
    world.step_sync(0.3)
    v_low = top_speed()
    # raise the lift most of the way (joint targets hold it there in navigation)
    world.set_joint_targets({"lift_lower": 0.3, "lift_upper": 0.3})
    world.step_sync(2.0)
    assert world.snapshot()["lift"] > 0.4, "lift did not rise for the raised case"
    v_high = top_speed()
    assert v_low > 0.55, f"lowered-lift top speed should approach the 0.8 cap ({v_low:.2f})"
    assert v_high < v_low - 0.08, \
        f"raised lift must slow the base (low {v_low:.2f} vs high {v_high:.2f} m/s)"
    world.reset_robot()


def test_drive_uses_the_coarse_timestep(world):
    """The bridge's TS_DRIVE trick, mirrored: navigation runs coarse physics
    (drive at real time), manipulation methods restore the validated fine step.
    (dt 4 ms cost ~3.9 ms wall on a laptop → RT 0.4-0.5x = 'drive feels dead'.)"""
    from control_lab.server.simworld import TS_DRIVE
    from control_lab.server import config as CFG
    world.set_control_spec(method="navigation")
    assert abs(world.model.opt.timestep - TS_DRIVE) < 1e-12
    world.set_control_spec(method="joint")
    assert abs(world.model.opt.timestep - float(CFG.TIMESTEP)) < 1e-12
    world.set_control_spec(method="navigation")
    assert abs(world.model.opt.timestep - TS_DRIVE) < 1e-12


# ── joint control ────────────────────────────────────────────────────────────
def test_joint_command_tracks(world):
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="joint", control_type="impedance", region="arm",
                           stiffness=30.0)
    world.step_sync(0.3)
    assert not world.teleop.wbc.active, "Stiff = position servos (no WBC torque engine)"
    q0 = world.plant.q("Left Arm 4")
    world.set_joint_targets({"left_arm_4": q0 + 0.3})
    world.step_sync(2.0)
    err = q0 + 0.3 - world.plant.q("Left Arm 4")
    assert abs(err) < 0.05, f"position servo did not track the joint target ({err:.3f} rad)"
    # Soft (compliance) floats — a push deflects and the target rides the pose
    world.set_control_spec(control_type="compliance")
    world.step_sync(0.4)
    assert world.estimator is not None, "Soft (WBC) must mount the contact estimator"
    world.reset_robot()


@pytest.mark.parametrize("region", ["arm", "upper_body", "whole_body"])
def test_lift_commandable_in_every_region_under_stiff(world, region):
    """The lift must go up/down from a joint command in EVERY region under Stiff
    (the WBC used to neutralise the Lift servo in upper/whole → the slider was
    inert). Under Soft in an upper/whole region it is freed (floats) by design."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="joint", control_type="impedance", region=region,
                           stiffness=30.0)
    world.step_sync(0.3)
    lift0 = world.snapshot()["lift"]
    world.set_joint_targets({"lift_lower": 0.3, "lift_upper": 0.3})   # → Lift 0.6
    world.step_sync(1.5)
    lift1 = world.snapshot()["lift"]
    # Assert the commanded HEIGHT is reached, not a fixed rise: the model now boots
    # half-raised (~0.29 m after servo droop), so a ">0.4 m rise" would only be
    # reachable from a bottomed start. 0.6 commanded, ~0.03 m servo droop under load.
    assert lift1 > 0.55, (
        f"lift did not reach the commanded 0.6 m under a Stiff command "
        f"(started {lift0:.3f}, ended {lift1:.3f})")
    assert lift1 > lift0, "lift did not rise at all"
    world.reset_robot()


# ── wrist task-space ─────────────────────────────────────────────────────────
def _palm(world, side="l"):
    return np.asarray(world.snapshot()["bodies"][f"palm_{side}"])


def test_wrist_ik_tracks_without_oscillation(world):
    """RIGOROUS wrist tracking: for several dragged targets the palm must (a) SETTLE
    — no sustained oscillation — and (b) track the target closely. The IK plans on
    its own scratch target state (decoupled from the impedance-lagging arm); solving
    against the measured arm cascades the two loops and oscillates (the bug)."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="wrist", control_type="impedance", region="arm",
                           stiffness=45.0)
    world.step_sync(0.3)
    base = np.asarray(world.snapshot()["bodies"]["palm_l"])
    for dv in ([0.0, 0.10, 0.05], [0.05, 0.05, -0.05], [-0.06, 0.12, 0.03]):
        goal = (base + np.asarray(dv)).tolist()
        world.set_wrist_target("l", goal)
        world.step_sync(3.0)
        # settling: palm position std over the last 1 s must be tiny (no oscillation)
        tail = []
        for _ in range(20):
            world.step_sync(0.05)
            tail.append(_palm(world))
        settle = float(np.linalg.norm(np.asarray(tail).std(0)))
        assert settle < 0.003, f"palm oscillates (last-1s std {settle*1000:.1f} mm) at dv={dv}"
        tg = world.snapshot()["overlay"]["targets"]["l"]
        ee = np.linalg.norm(np.asarray(tg["target"]) - np.asarray(tg["actual"]))
        assert ee < 0.025, f"palm did not track the target ({ee*1000:.1f} mm) at dv={dv}"
    assert _latest(world, "ee_err_l_mm") is not None
    assert _latest(world, "ik_solve_ms") is not None
    # SOFT floats; STIFF re-engages bumplessly (targets rode the measured pose)
    world.set_control_spec(control_type="compliance")
    world.step_sync(0.8)
    world.set_control_spec(control_type="impedance")
    world.step_sync(0.8)
    _healthy(world)
    with pytest.raises(ValueError):
        world.set_wrist_target("x", goal)
    world.reset_robot()


def test_wrist_human_posture(world):
    """Human-like redundancy (the VP teleop's ARM_REG_W null-space constraints)
    keeps wrist-yaw (Arm 7) closer to rest than the generic uniform posture."""
    def drag(human):
        world.reset_robot(); world.set_torque(True)
        world.set_control_spec(method="wrist", control_type="impedance",
                               region="arm", stiffness=45.0)
        world.set_wrist_human(human)
        world.step_sync(0.3)
        rest = world.plant.q("Left Arm 7")
        t0 = world.snapshot()["overlay"]["targets"]["l"]["target"]
        world.set_wrist_target("l", [t0[0] - 0.06, t0[1] - 0.16, t0[2] + 0.06])
        world.step_sync(2.5)
        return abs(world.plant.q("Left Arm 7") - rest)

    dev_generic = drag(False)
    dev_human = drag(True)
    assert world.snapshot()["wristHuman"] is True
    assert dev_human <= dev_generic + 0.02, (
        f"human posture should restrain wrist-yaw: {dev_human:.3f} vs {dev_generic:.3f} rad")
    assert world.ik.posture_w[2 + 6] > 5.0            # Left Arm 7
    world.reset_robot()


def test_wrist_upper_body_recruits_lift(world):
    """UPPER-BODY task-space: dragging both palm targets straight UP must extend
    the LIFT (the 18-DOF solve recruits it; its joint-unit target rides the wire
    as lift_lower/lift_upper). Regression: _IK_WIRE had no "Lift" entry, so the
    solved lift motion was silently dropped — "the lift is stuck"."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="wrist", control_type="impedance",
                           region="upper_body", stiffness=45.0)
    world.step_sync(0.3)
    lift0 = world.snapshot()["lift"]
    for side in ("l", "r"):
        p = _palm(world, side)
        world.set_wrist_target(side, (p + np.array([0.0, 0.0, 0.5])).tolist())
    world.step_sync(6.0)
    s = _healthy(world)
    dlift = s["lift"] - lift0
    # measured: ~0.36 m of tendon travel by 6 s (rate-capped at the hardware
    # 60 mm/s ceiling); assert half that so load jitter can't flake the test.
    assert dlift > 0.18, f"lift did not extend under a high task-space target (Δ={dlift:.3f} m)"
    assert abs(s["basePose"]["dz"]) < 0.05, "chassis must stay upright"
    world.reset_robot()


def test_wrist_whole_body_lateral_strafe_and_snap(world):
    """Side work: both targets dragged LEFT must make the base STRAFE (the old
    centroid-bearing turn left the far palm 200 mm short — pair-alignment says
    don't turn, side-step), both palms must converge, and the follower must
    STOP DEAD (snap, not float: hysteretic servo, zero residual twist)."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="wrist", control_type="impedance",
                           region="whole_body", stiffness=45.0)
    world.step_sync(0.3)
    for side in ("l", "r"):
        p = _palm(world, side)
        world.set_wrist_target(side, (p + np.array([0.0, -0.5, 0.0])).tolist())
    world.step_sync(8.0)
    s = _healthy(world)
    assert s["base"]["y"] < -0.3, f"base did not strafe left ({s['base']})"
    assert abs(s["base"]["yaw"]) < 0.15, \
        f"equal lateral drag must strafe, not turn (yaw={s['base']['yaw']})"
    for side in ("l", "r"):
        tg = s["overlay"]["targets"][side]
        ee = np.linalg.norm(np.asarray(tg["target"]) - np.asarray(tg["actual"]))
        assert ee < 0.06, f"palm_{side} did not converge ({ee*1000:.0f} mm)"
    assert not world.base_follow.active, "follower must disengage after settling"
    assert max(abs(v) for v in world._base_v) < 1e-3, \
        f"residual base twist (float): {world._base_v}"
    world.reset_robot()


def test_wrist_whole_body_base_follows_targets(world):
    """WHOLE-BODY task-space: dragging both palm targets far ahead must drive the
    chassis after them (BaseFollow comfort box → safe_base_step → swerve), stay
    UPRIGHT (torso rate cap + CoM guard: the uncapped chase used to face-plant),
    and yield to a deliberate stick. Regression: nothing mapped wrist targets to
    the base — "the mobile base is not moving at all"."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="wrist", control_type="impedance",
                           region="whole_body", stiffness=45.0)
    world.step_sync(0.3)
    for side in ("l", "r"):
        p = _palm(world, side)
        world.set_wrist_target(side, (p + np.array([-1.2, 0.0, 0.0])).tolist())
    world.step_sync(1.0)
    assert world._last_base[0] < -0.1, \
        f"base did not follow the far targets (vx={world._last_base[0]})"
    wheels = [abs(float(world.data.actuator(f"{k.upper()} Drive").ctrl[0]))
              for k in ("fl", "fr", "b")]
    assert max(wheels) > 1.0, f"wheels not commanded ({wheels})"
    world.step_sync(5.0)                     # arrive + settle
    s = _healthy(world)
    assert abs(s["basePose"]["dz"]) < 0.08, "chassis must stay upright (no face-plant)"
    assert s["base"]["x"] < -0.4, f"chassis did not translate toward the targets ({s['base']})"
    # a deliberate stick beats the follower (unified-teleop arbitration)
    world.nav.update({"lx": 0.0, "ly": -1.0, "rx": 0.0, "ry": 0.0})
    world.step_sync(1.2)
    assert world._last_base[0] > 0.05, \
        f"stick-down must override the follower (vx={world._last_base[0]})"
    world.nav.update({"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0})
    world.reset_robot()


# ── wrench estimation (re-homed from the old controllers bench) ──────────────
def test_stiff_wrench_estimation(world):
    # Estimation is accurate in STIFF (impedance): the arm resists the push and
    # stays quasi-static, so the momentum-observer residual is a clean read of
    # the external wrench (Soft yields + runs away — see the whole-body glide
    # test). +Y (lateral) is well-conditioned; ±X at home is near-unobservable.
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="joint", control_type="impedance", region="arm",
                           stiffness=30.0)
    world.step_sync(1.5)                          # observer convergence
    world.apply_push("left_palm", [0.0, 12.0, 0.0], seconds=2.0)
    world.step_sync(1.2)                          # mid-push, settled
    s = world.snapshot()
    f_true = _latest(world, "F_true")
    f_est = _latest(world, "F_est")
    cos = _latest(world, "F_cos")
    assert f_true == pytest.approx(12.0)
    assert f_est is not None and 0.5 * 12 < f_est < 1.6 * 12, f"|F̂|={f_est}"
    assert cos is not None and cos > 0.9, f"direction cos {cos}"
    w = s["overlay"]["wrench"]
    assert w["true"]["body"] == "left_palm"
    assert w["est"] is not None and w["est"]["joint"].startswith("Left")
    world.reset_robot()


def test_whole_body_soft_glides(world):
    """Whole-body Soft: the base admits the net push and glides (the only config
    where the chassis moves under an operator wrench)."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="joint", control_type="compliance", region="whole_body")
    world.step_sync(0.5)
    world.apply_push("left_palm", [0.0, 14.0, 0.0], seconds=1.5)
    world.step_sync(1.0)
    assert abs(world._last_base[1]) > 0.05, "whole-body Soft base did not glide under the push"
    _healthy(world)
    world.reset_robot()


def test_whole_body_stiff_base_drives(world):
    """In the whole_body region the base is drivable by the sticks (whole-body =
    arms + base): a manipulation mode still lets you move the chassis with the
    joystick, while the arms hold their position-controlled targets."""
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="joint", control_type="impedance", region="whole_body")
    world.nav.update({"lx": 0, "ly": 1.0, "rx": 0, "ry": 0})     # forward stick
    world.step_sync(1.2)
    assert world._last_base[0] < -0.1, \
        f"whole-body base did not drive under the stick (vx={world._last_base[0]})"
    world.nav.update({"lx": 0, "ly": 0, "rx": 0, "ry": 0})
    _healthy(world)
    world.reset_robot()


# ── whole-body retarget (realtime, mock operator) ───────────────────────────
def test_retarget_with_mock(rtworld):
    world = rtworld
    world.set_torque(True)
    world.set_control_spec(method="retarget", control_type="impedance", region="upper_body")
    vp = VPNode(world)
    vp.set_source("mock")
    try:
        time.sleep(3.5)
        s = world.snapshot()
        assert s.get("calibrated") is True
        assert s["vp"]["connected"] and s["vp"]["rate_hz"] > 5
        for key in ("jerk", "stream_hz", "retarget_ms"):
            assert _latest(world, key) is not None, f"metric {key} missing"
        sk = s["overlay"].get("skeleton") or {}
        assert "head" in sk and "wrist_l" in sk and "wrist_r" in sk
    finally:
        vp.stop()
        world.set_torque(False)


def test_vp_zmq_stream_source_ingests(world, monkeypatch):
    """The 'avp' source must consume the vision_pro_stream.py ZMQ publisher
    (PUB, conflated JSON frames) through the same OperatorPose seam as the
    headset path — the lab then retargets + overlays the operator skeleton."""
    zmq = pytest.importorskip("zmq")
    import json
    import numpy as np

    addr = "tcp://127.0.0.1:19555"
    monkeypatch.setenv("MABEL_CTS_AVP_ADDR", addr)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(addr)
    vp = VPNode(world)
    try:
        world._mailbox = None
        vp.set_source("avp")
        eye = np.eye(4).flatten().tolist()
        frame = {"head_mat": eye,
                 "right_hand_mats": [eye] * 21, "left_hand_mats": [eye] * 21,
                 "timestamp": 123.456}
        deadline = time.time() + 5.0
        while world._mailbox is None and time.time() < deadline:
            pub.send_string(json.dumps(frame))     # PUB/SUB needs a warm-up
            frame["timestamp"] += 0.016
            time.sleep(0.05)
        assert world._mailbox is not None, "ZMQ frame never reached the ingest mailbox"
        pose, device, _ = world._mailbox
        assert device == "Vision Pro (ZMQ stream)"
        assert pose.head is not None and pose.right_hand is not None
    finally:
        vp.stop()
        pub.close(0)


# ── snapshot + reset ─────────────────────────────────────────────────────────
def test_snapshot_streams_body_positions(world):
    s = world.snapshot()
    for k in ("head", "palm_l", "palm_r"):
        assert s["bodies"].get(k) is not None and len(s["bodies"][k]) == 3
        assert all(np.isfinite(s["bodies"][k]))
    assert set(s["spec"]) == {"method", "type", "region", "stiffness"}


def test_reset_restores_home(world):
    world.reset_robot(); world.set_torque(True)
    world.set_control_spec(method="navigation")
    world.nav.update({"ly": 1.0}); world.step_sync(1.0); world.nav.update({"ly": 0.0})
    world.reset_robot()
    world.step_sync(0.3)
    s = world.snapshot()
    assert s["torque"] is False, "reset returns to LIMP"
    assert abs(s["base"]["x"]) < 0.05 and abs(s["base"]["y"]) < 0.05
    assert abs(s["odom"]["x"]) < 0.05
