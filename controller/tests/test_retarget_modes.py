"""The two retargeting modes do what the paper says they do.

The load-bearing tests here are not "does the code run" but three PROPERTIES
that the architecture claims and that are easy to break by accident:

  1. LOCAL MODE IS INVARIANT TO THE WORLD ESTIMATE. Drive an identical operator
     through an identical task twice, once with an arbitrary ARKit-anchor drift
     superposed, and the robot targets must be identical to machine precision —
     not close, identical. This is the whole reason local mode exists, and it
     dies silently the moment anything F_a-referenced leaks into the local path.
  2. THE ADMITTED MEASUREMENT SET IS ENFORCED STRUCTURALLY. `LocalTorsoFit`
     must have no parameter through which the device pose could arrive. A test
     on the signature is worth more than a test on behaviour, because behaviour
     tests pass right up until someone adds the parameter "just for logging".
  3. THE HANDOFF IS EXACT. Eq. (15)'s alignment latch is supposed to make
     R_bo continuous, so a mode switch mid-motion must not move the targets.

The rest pin the shape of the pipeline: mode-invariance of eqs. (16)-(18), the
base channel being zero in local, and the ramps behaving.
"""
import ast
import time
import inspect
import textwrap

import numpy as np
import pytest

from mabel.retargeting import (BodyRetargeter, DeviceMeasurements, LocalTorsoFit,
                               OperatorModel, RetargetMode, RobotFrames,
                               WorldMeasurements, device_measurements,
                               rigid_inverse, wrist_targets, AVP_TO_ROBOT)
from mabel.retargeting.body_dofs import swivel_target
from mabel.retargeting.frames import log_so3
from mabel.retargeting.local_frame import _R_from_ypr, _ypr_from_R, project_cervical
from mabel.retargeting.operator_model import (ANTHRO_ARM_RATIO,
                                              ANTHRO_DEVICE_REACH_RATIO,
                                              ANTHRO_FOREARM_RATIO,
                                              ANTHRO_UPPER_ARM_RATIO)
from mabel.retargeting.synthetic import (SyntheticOperator, VehicleDrift,
                                         mabel_like_robot)

DT = 1.0 / 60.0


def _task(op, t):
    """A repeatable bimanual task: both hands trace a circle in front of the
    torso at a near-full reach (so the fit is well conditioned)."""
    a = 2.0 * np.pi * 0.25 * t
    for side, s in (("right", 1.0), ("left", -1.0)):
        op.wrist[side] = (0.75, np.array([s * 0.35 + 0.15 * np.cos(a),
                                          -0.2 + 0.15 * np.sin(a), 0.9]))


def run(mode, drift=None, seconds=4.0, dt=DT, switch_at=None, op=None):
    """Drive the retargeter through the task; return the per-frame record."""
    op = op or SyntheticOperator()
    rt = BodyRetargeter(mode=mode)
    robot = mabel_like_robot()
    rec = []
    for i in range(int(seconds / dt)):
        t = i * dt
        _task(op, t)
        if switch_at is not None and abs(t - switch_at) < 0.5 * dt:
            rt.set_mode(RetargetMode.GLOBAL if rt.mode == RetargetMode.LOCAL
                        else RetargetMode.LOCAL)
        T_ad, hands, skel = op.frame(drift(t) if drift else None)
        dev = device_measurements(T_ad, hands, skel, timestamp=t)
        out = rt.update(dev, WorldMeasurements(T_ad), robot, dt)
        rec.append(out)
    return rt, rec


def _targets(rec, skip=1.5, dt=DT):
    """Stacked wrist targets after the estimators have settled."""
    n0 = int(skip / dt)
    return np.array([np.concatenate([o.wrist_targets[s] for s in ("left", "right")])
                     for o in rec[n0:] if o.valid and len(o.wrist_targets) == 2])


# ── 1. the property local mode exists for ────────────────────────────────────
def test_local_targets_are_invariant_to_anchor_drift():
    """A vehicle corrupts F_a; local mode must not notice AT ALL.

    20 deg/s of anchor yaw (the gyro integrating a turn), 6 deg of gravity-
    reference tilt (lateral acceleration) and a translational wander. Local
    mode's targets must be bit-identical to the undrifted run.
    """
    veh = VehicleDrift(yaw_rate=np.deg2rad(20), tilt_amp=np.deg2rad(6),
                       drift_vel=(0.02, 0.0, 0.01))
    base = _targets(run("local")[1])
    drifted = _targets(run("local", veh)[1])
    n = min(len(base), len(drifted))
    assert n > 100
    assert np.abs(base[:n] - drifted[:n]).max() < 1e-12


def test_global_targets_are_NOT_invariant_to_anchor_drift():
    """The contrast that makes the previous test meaningful: global mode reads
    T_ad, so the same drift moves its targets by centimetres. If this ever
    passes at 1e-12 the two modes have collapsed into one."""
    veh = VehicleDrift(yaw_rate=np.deg2rad(20), tilt_amp=np.deg2rad(6),
                       drift_vel=(0.02, 0.0, 0.01))
    base = _targets(run("global")[1])
    drifted = _targets(run("global", veh)[1])
    n = min(len(base), len(drifted))
    assert np.abs(base[:n] - drifted[:n]).max() > 0.01


def test_local_base_channel_is_zero():
    """nu_b^cmd = 0 in local mode (eq. 15), even while the operator's body is
    doing whatever the vehicle is doing."""
    veh = VehicleDrift(yaw_rate=np.deg2rad(30), tilt_amp=np.deg2rad(8))
    _, rec = run("local", veh)
    assert max(abs(v) for o in rec for v in o.base_twist) == 0.0


# ── 2. the admitted set, checked on the signature ────────────────────────────
def test_local_fit_cannot_see_the_device_pose():
    """S_L is enforced by the type signature, not by discipline: LocalTorsoFit
    must accept only device-frame measurements.

    Checked on the parsed CODE rather than the raw source, so prose in the
    docstrings explaining what local mode deliberately excludes does not trip
    the test that enforces the exclusion.
    """
    params = list(inspect.signature(LocalTorsoFit.update).parameters)
    assert params == ["self", "dev", "dt"], params
    tree = ast.parse(textwrap.dedent(inspect.getsource(LocalTorsoFit)))
    banned = {"T_ad", "R_ad", "world", "WorldMeasurements"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in banned, f"{node.id} reached the local estimator"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in banned, f".{node.attr} reached the local fit"
        elif isinstance(node, ast.arg):
            assert node.arg not in banned, f"{node.arg} is a local-fit parameter"


def test_device_frame_reduction_cancels_the_anchor():
    """T_dh = T_ad^-1 T_ah is anchor-free: premultiplying every pose by an
    arbitrary D leaves the device-frame measurement unchanged."""
    op = SyntheticOperator()
    rng = np.random.default_rng(0)
    for _ in range(20):
        D = np.eye(4)
        D[:3, :3] = _R_from_ypr(*rng.uniform(-2, 2, 3))
        D[:3, 3] = rng.uniform(-5, 5, 3)
        a = device_measurements(*op.frame(None)[:1], op.frame(None)[1])
        b = device_measurements(*op.frame(D)[:1], op.frame(D)[1])
        for side in ("left", "right"):
            assert np.abs(a.wrists[side] - b.wrists[side]).max() < 1e-12


# ── 3. the handoff ───────────────────────────────────────────────────────────
def test_mode_switch_does_not_move_the_targets():
    """The alignment latch (eq. 15's R_b*) makes R_bo continuous, so the frame
    after a switch must land where the frame before it did. Compared against
    the motion the task itself produces in one frame, which is the only
    meaningful yardstick."""
    for start in ("local", "global"):
        _, rec = run(start, seconds=4.0, switch_at=2.0)
        i = next(i for i, o in enumerate(rec) if o.diag.get("switched"))
        pre, post = rec[i - 1], rec[i]
        jump = max(np.linalg.norm(post.wrist_targets[s] - pre.wrist_targets[s])
                   for s in ("left", "right"))
        # one frame of ordinary task motion, for scale
        step = max(np.linalg.norm(rec[i - 1].wrist_targets[s]
                                  - rec[i - 2].wrist_targets[s])
                   for s in ("left", "right"))
        assert jump < max(2.0 * step, 1e-3), f"{start}: jump {jump:.4f} m"


def test_base_ramps_out_rather_than_cutting():
    """Switching global -> local must RAMP the base to zero, not step it."""
    _, rec = run("global", seconds=4.0, switch_at=2.0)
    i = next(i for i, o in enumerate(rec) if o.diag.get("switched"))
    after = [np.linalg.norm(o.base_twist) for o in rec[i:i + 90]]
    assert after[-1] == 0.0                     # settles to exactly zero
    if after[0] > 1e-6:                         # only meaningful if it was moving
        assert after[0] < np.linalg.norm(rec[i - 1].base_twist) + 1e-9
        assert all(b <= a + 1e-9 for a, b in zip(after, after[1:]))


def test_switch_takes_effect_on_the_next_frame():
    rt, rec = run("local", seconds=3.0, switch_at=1.0)
    assert rt.mode == RetargetMode.GLOBAL
    assert sum(1 for o in rec if o.diag.get("switched")) == 1


# ── the local fit itself ─────────────────────────────────────────────────────
def _fit_yaw(neck, rho=0.7, frames=90):
    op = SyntheticOperator()
    op.set_reach(rho)
    op.neck = tuple(np.deg2rad(neck))
    fit = LocalTorsoFit(OperatorModel())
    for _ in range(frames):
        T_ad, hands, skel = op.frame()
        fit.update(device_measurements(T_ad, hands, skel), DT)
    return np.rad2deg(_ypr_from_R(fit.R_do.T))[0]


@pytest.mark.parametrize("neck", [(0, 0, 0), (30, 0, 0), (-45, 0, 0),
                                  (0, 25, 0), (15, 10, -10), (-25, -15, 5),
                                  (-70, 40, -30)])
def test_local_fit_recovers_torso_yaw(neck):
    """Yaw is the DOF that matters — it sets left/right of the whole body frame.
    Pitch/roll carry a small systematic offset from the robust weighting, which
    the mode-entry latch absorbs as a constant; yaw cannot be absorbed and so is
    what is pinned here. These are ordinary teleop head poses; measured mean
    2.6 deg, worst 8.3 deg over this set."""
    yaw = _fit_yaw(neck)
    assert abs(yaw - neck[0]) < 10.0, f"yaw {yaw:.1f} vs {neck[0]}"


@pytest.mark.parametrize("neck", [(60, -20, 15), (45, 30, -20), (-60, -40, 25),
                                  (70, 50, 30), (-30, 55, -40)])
def test_local_fit_degrades_gracefully_at_rom_corners(neck):
    """At simultaneous large yaw AND pitch AND roll the objective's global
    minimum is no longer the truth (see SEED_YAWS' note), so the estimate loses
    accuracy — measured mean 20 deg, worst 67 deg over this set. What it must
    NOT do is run away: the failure a teleoperator can work around is a rotated
    reach frame, not a body frame that has flipped or diverged. Bounding it at
    90 deg is the property under test, not the accuracy."""
    yaw = _fit_yaw(neck)
    assert abs(yaw - neck[0]) < 90.0, f"yaw {yaw:.1f} vs {neck[0]}"
    assert np.isfinite(yaw)


def _wobble(neck_fn, seconds=8.0):
    """Robot-target motion while the HANDS are fixed relative to the torso and
    only the neck moves. Anything nonzero here is the retargeter moving the
    robot without the operator having commanded it."""
    op = SyntheticOperator()
    op.set_reach(0.7)
    rt = BodyRetargeter(mode="local")
    robot = mabel_like_robot()
    tg = []
    for i in range(int(seconds / DT)):
        t = i * DT
        op.neck = neck_fn(t)
        T_ad, hands, skel = op.frame()
        out = rt.update(device_measurements(T_ad, hands, skel, timestamp=t),
                        WorldMeasurements(T_ad), robot, DT)
        if t > 2.0 and out.valid:
            tg.append(np.concatenate([out.wrist_targets[s]
                                      for s in ("left", "right")]))
    tg = np.array(tg).reshape(-1, 2, 3)
    return np.linalg.norm(tg - tg.mean(axis=0), axis=2).mean() * 1000


def test_a_still_head_moves_nothing():
    assert _wobble(lambda t: (0.0, 0.0, 0.0)) < 1e-6      # nanometres


def test_glancing_around_barely_moves_the_targets():
    """The dominant head motion in teleop. Measured 3.5 mm mean; the bound is
    where "the robot twitches when I look around" would start being visible."""
    assert _wobble(lambda t: (np.deg2rad(60) * np.sin(2 * np.pi * 0.2 * t),
                              0.0, 0.0)) < 15.0


def test_nodding_wobble_is_bounded_and_recorded():
    """The estimator's known weak spot, pinned so it cannot silently get WORSE.
    A sustained +-30 deg nod moves the targets ~157 mm mean because the tilt
    prior pulls the torso's up toward the device's (see TILT_REF). This is a
    real limitation, recorded as a number rather than as a hope."""
    assert _wobble(lambda t: (0.0, np.deg2rad(30) * np.sin(2 * np.pi * 0.15 * t),
                              0.0)) < 250.0


def test_local_fit_holds_when_no_hands_are_tracked():
    fit = LocalTorsoFit(OperatorModel())
    op = SyntheticOperator()
    T_ad, hands, skel = op.frame()
    for _ in range(30):
        fit.update(device_measurements(T_ad, hands, skel), DT)
    held = fit.R_do.copy()
    for _ in range(30):
        fit.update(DeviceMeasurements(), DT)
    assert np.abs(fit.R_do - held).max() == 0.0


def test_cervical_projection_is_idempotent_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(200):
        R = _R_from_ypr(*rng.uniform(-np.pi, np.pi, 3)).T
        P, _ = project_cervical(R)
        P2, clamped = project_cervical(P)
        assert np.abs(P - P2).max() < 1e-9 and not clamped
        y, p, r = np.rad2deg(_ypr_from_R(P.T))
        assert abs(y) <= 80.1 and abs(p) <= 60.1 and abs(r) <= 45.1


def test_elbow_makes_the_anthropometry_bootstrap_exact():
    """The forearm is a rigid segment, so it sizes the operator at any arm
    configuration; the reach-max fallback needs a full extension it may never
    see (measured: 1.51 m for a 1.75 m operator who only ever reaches rho=0.6)."""
    op = SyntheticOperator(stature=1.75)
    op.set_reach(0.6)
    with_elbow, without = OperatorModel(), OperatorModel()
    for _ in range(20):
        T_ad, hands, skel = op.frame()
        with_elbow.observe(device_measurements(T_ad, hands, skel))
        without.observe(device_measurements(T_ad, hands,
                                            op.frame(elbows=False)[2]))
    assert abs(with_elbow.stature - 1.75) < 0.02
    assert abs(without.stature - 1.75) > 0.10


# ── eqs. (16)-(18) are mode-invariant ────────────────────────────────────────
def test_wrist_targets_depend_only_on_R_bo():
    """Same body-relative geometry through the same R_bo must give the same
    target, whichever frame it was expressed in — that IS mode invariance."""
    from mabel.retargeting.types import BodyFrameEstimate
    model = OperatorModel()
    robot = mabel_like_robot()
    est = BodyFrameEstimate(basis=np.eye(3), root=np.zeros(3), valid=True,
                            shoulders={"right": np.array([0.22, 0.0, 0.0]),
                                       "left": np.array([-0.22, 0.0, 0.0])})
    pts = {"right": np.array([0.30, -0.1, 0.45]),
           "left": np.array([-0.30, -0.1, 0.45])}
    a, _ = wrist_targets(pts, est, robot, model, AVP_TO_ROBOT)
    # Re-express the SAME physical configuration in a rotated source frame and
    # compensate R_bo; targets must not move.
    Q = _R_from_ypr(0.7, -0.2, 0.3)
    est_q = BodyFrameEstimate(basis=Q, root=np.zeros(3), valid=True,
                              shoulders={k: Q @ v for k, v in est.shoulders.items()})
    b, _ = wrist_targets({k: Q @ v for k, v in pts.items()}, est_q, robot, model,
                         AVP_TO_ROBOT @ Q.T)
    for side in ("left", "right"):
        assert np.abs(a[side] - b[side]).max() < 1e-12


def test_full_extension_maps_to_full_robot_extension():
    """The guarantee of the extension branch: rho = 1 puts the robot wrist at
    its own arm's length from its own shoulder, in whatever direction."""
    from mabel.retargeting.types import BodyFrameEstimate
    model, robot = OperatorModel(), mabel_like_robot()
    L = model.arm_length("right")
    for d in (np.array([0, 0, 1.0]), np.array([0.6, -0.5, 0.6]),
              np.array([-0.3, 0.8, 0.5])):
        d = d / np.linalg.norm(d)
        s = np.array([0.22, 0.0, 0.0])
        est = BodyFrameEstimate(basis=np.eye(3), root=np.zeros(3), valid=True,
                                shoulders={"right": s})
        t, diag = wrist_targets({"right": s + d * L}, est, robot, model,
                                AVP_TO_ROBOT)
        r = np.linalg.norm(t["right"] - robot.shoulders["right"])
        assert abs(diag["right"]["rho"] - 1.0) < 1e-9
        assert abs(r - robot.reach["right"]) < 1e-9


def test_swivel_deadzone():
    """Eq. (19): nothing inside +-60 deg, then it follows, then it clamps."""
    assert swivel_target(np.deg2rad(30)) == 0.0
    assert swivel_target(np.deg2rad(-45)) == 0.0
    assert abs(swivel_target(np.deg2rad(80)) - np.deg2rad(20)) < 1e-9
    assert abs(swivel_target(np.deg2rad(-80)) + np.deg2rad(20)) < 1e-9
    assert swivel_target(np.deg2rad(200)) == pytest.approx(np.deg2rad(90))


# ── plumbing ─────────────────────────────────────────────────────────────────
def test_mode_normalisation_defaults_to_local():
    """An unknown or malformed wire value must never grant base authority."""
    for bad in (None, "", "wholebody", "GLOBALISH", 7):
        assert RetargetMode.normalize(bad) == RetargetMode.LOCAL
    assert RetargetMode.normalize("GLOBAL") == RetargetMode.GLOBAL
    assert RetargetMode.normalize(" local ") == RetargetMode.LOCAL


def test_avp_to_robot_matches_the_engine():
    """This package deliberately re-declares AVP_TO_ROBOT so it imports nothing
    from the engine; that duplication has to be checked, not trusted."""
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT as ENGINE
    assert np.abs(AVP_TO_ROBOT - ENGINE).max() == 0.0


def test_anthropometric_ratios_are_self_consistent():
    assert abs(ANTHRO_UPPER_ARM_RATIO + ANTHRO_FOREARM_RATIO
               - ANTHRO_ARM_RATIO) < 1e-9
    # the device-frame bootstrap scale is derived from the other ratios
    op = SyntheticOperator()
    op.set_reach(1.0)
    op.wrist = {s: (1.0, np.array([0.0, 0.0, 1.0])) for s in ("right", "left")}
    T_ad, hands, _ = op.frame()
    r = float(np.linalg.norm((rigid_inverse(T_ad) @ hands["right"])[:3, 3]))
    assert abs(r / op.H - ANTHRO_DEVICE_REACH_RATIO) < 0.02


# ── the live stack: does the switch actually hold the base? ──────────────────
def _drive_facade(fresh, seconds=3.0, dt=1.0 / 60.0, scope="global"):
    """Run the REAL facade with a WALKING operator and report the largest base
    velocity it commanded.

    The commanded twist, not the realised pose: the base only actually moves
    once physics steps, and what the mode governs is the COMMAND. Measuring
    xpos in a non-stepping harness reads zero for both modes and proves
    nothing — which is exactly what the first version of this test did.
    """
    from mabel.facade import MabelTeleop
    from mabel.types import OperatorPose

    model, data = fresh()
    teleop = MabelTeleop(model, data)
    teleop.set_control_spec(method="retarget", scope=scope)
    op = SyntheticOperator()
    peak = 0.0
    for i in range(int(seconds / dt)):
        t = i * dt
        # The engine's anchor-stability hold is measured on the WALL clock
        # (HEAD_ANCHOR_HOLD), not on the pose timestamps, so a harness that
        # replays 3 simulated seconds in 30 wall-milliseconds never anchors and
        # the whole retarget path — base included — silently no-ops. Wait it out
        # once, on the first frame, rather than sleeping every iteration.
        if i == 1:
            time.sleep(0.45)
        _task(op, t)
        # The operator WALKS: world translation plus a body turn, which must
        # command the base.
        op.torso_yaw = np.deg2rad(30) * np.sin(2 * np.pi * 0.25 * t)
        T_ad, hands, skel = op.frame()
        T_ad = T_ad.copy()
        T_ad[:3, 3] += np.array([0.0, 0.0, -0.4 * t])
        pose = OperatorPose(head=T_ad, right_joints=skel["right"],
                            left_joints=skel["left"], timestamp=t)
        teleop.update(pose, dt=dt)
        peak = max(peak, max(abs(v) for v in teleop.controller.wbc.base_tgt))
    return peak


def test_walking_drives_the_base_in_GLOBAL_scope(fresh):
    """A walking, turning operator commands base motion — in GLOBAL scope."""
    glob = _drive_facade(fresh, scope="global")
    assert glob > 0.05, f"walking should drive the base, commanded {glob:.4f}"


def test_walking_does_NOT_drive_the_base_in_LOCAL_scope(fresh):
    """…and in LOCAL scope (the DEFAULT, 2026-07-31) the same walk commands
    nothing: the base is out of the follow loop entirely and belongs to the
    operator's joystick. This is the whole point of the mode — a fixed root
    makes the arm IK an easier problem, and the base never moves by surprise."""
    loc = _drive_facade(fresh, scope="local")
    assert loc < 1e-6, f"LOCAL scope must command no base motion, got {loc:.4f}"


def test_local_is_the_default_scope(fresh):
    """Unasked-for base authority is the consequential state, so it is never
    the default — an omitted or unrecognised scope stays LOCAL."""
    model, data = fresh()
    from mabel.facade import MabelTeleop
    teleop = MabelTeleop(model, data)
    assert teleop.controller.retarget_scope == "local"
    assert teleop.controller.wbc.base_in_loop is False
    assert teleop.controller.set_retarget_scope("nonsense") == "local"


def test_robot_frames_snapshot_needs_no_mujoco():
    """The package must stay importable and runnable headless."""
    robot = mabel_like_robot()
    assert isinstance(robot, RobotFrames)
    rt = BodyRetargeter()
    op = SyntheticOperator()
    T_ad, hands, skel = op.frame()
    out = rt.update(device_measurements(T_ad, hands, skel),
                    WorldMeasurements(T_ad), robot, DT)
    assert out.valid and len(out.wrist_targets) == 2
