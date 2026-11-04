"""ArmsSubsystem — the calibration layer, which is the bridge's core promise:
everything ABOVE this line is MODEL-frame (MJCF/URDF) radians, everything BELOW
is raw motor radians, and the map is applied in exactly ONE place.

Uses the deterministic MockJointGroupDriver as the plant, so these pin the math,
not a transport.
"""
import json
import math

import pytest

from mabel_hw.calibration import CalibrationStore
from mabel_hw.drivers.seam import MockJointGroupDriver
from mabel_hw.subsystems.arms import ArmsSubsystem
from mabel_hw.types import JointCommand

JOINTS = ["l_j1", "l_j2", "l_j3"]

# l_j1: dir=+1, window centred on raw 0   -> model == raw
# l_j2: dir=-1, offset window             -> model flips AND shifts
# l_j3: absent from the calib             -> uncalibrated pass-through
TREE = {"arm": {"openarm-dual": {"calib": {
    "l_j1": {"raw_min": -1.0, "raw_max": 1.0, "dir": 1, "model_mid": 0.0},
    "l_j2": {"raw_min": 1.0, "raw_max": 3.0, "dir": -1, "model_mid": 0.0},
}}}}


def _sub(tmp_path, joints=JOINTS, alpha=1.0):
    root = tmp_path / "calibration_data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "mabel.json").write_text(json.dumps(TREE))
    store = CalibrationStore(root=root, mabel_root=tmp_path / "MABEL").load()
    drv = MockJointGroupDriver(joints, alpha=alpha)
    drv.connect()
    sub = ArmsSubsystem("left_arm", {"calib_slice": "arm/openarm-dual"}, {},
                        store=store, driver=drv)
    sub.set_torque(True)
    return sub, drv


# ── read: raw in -> model out ───────────────────────────────────────────────

def test_state_is_model_frame(tmp_path):
    sub, drv = _sub(tmp_path)
    drv._pos["l_j1"] = 0.5          # raw
    drv._pos["l_j2"] = 2.5          # raw; raw_mid=2.0, dir=-1 -> model = -0.5
    st = sub.state()
    by = dict(zip(st.names, st.position))
    assert by["l_j1"] == pytest.approx(0.5)
    assert by["l_j2"] == pytest.approx(-0.5)


def test_uncalibrated_joint_passes_through(tmp_path):
    sub, drv = _sub(tmp_path)
    drv._pos["l_j3"] = 1.23
    assert dict(zip(sub.state().names, sub.state().position))["l_j3"] == \
        pytest.approx(1.23)


def test_velocity_and_effort_follow_dir(tmp_path):
    """Rates map by dir alone. A dir=-1 joint whose torque did NOT flip would
    report effort pointing the opposite way to the motion causing it."""
    sub, drv = _sub(tmp_path)
    drv._vel["l_j2"] = 0.4
    monkey = drv.read()
    # MockJointGroupDriver reports effort 0; assert the sign math via a stub
    st = sub.state()
    by = dict(zip(st.names, st.velocity))
    assert by["l_j2"] == pytest.approx(-0.4)     # dir=-1 flips it
    assert monkey.velocity[1] == pytest.approx(0.4)   # ...raw was untouched


# ── write: model in -> raw out ──────────────────────────────────────────────

def test_set_targets_converts_to_raw(tmp_path):
    sub, drv = _sub(tmp_path)
    sub.set_targets({"l_j2": -0.5})              # model
    assert drv._targets["l_j2"] == pytest.approx(2.5)    # raw


def test_targets_clamp_to_the_calibrated_window(tmp_path):
    """The calibrated raw end-stops are the hard envelope — the one position
    limit the HAL keeps."""
    sub, drv = _sub(tmp_path)
    sub.set_targets({"l_j1": 99.0})
    assert drv._targets["l_j1"] == pytest.approx(1.0)    # raw_max
    sub.set_targets({"l_j1": -99.0})
    assert drv._targets["l_j1"] == pytest.approx(-1.0)   # raw_min


def test_command_maps_effort_and_velocity_into_the_raw_frame(tmp_path):
    """A gravity feedforward computed in MODEL frame by controller/ must reach a
    dir=-1 motor with its sign flipped, or it pushes the joint the wrong way."""
    seen = {}

    sub, drv = _sub(tmp_path)
    drv.command = lambda position, velocity=None, kp=None, kd=None, effort=None: \
        seen.update(position=position, velocity=velocity, effort=effort)
    sub.command(JointCommand(position={"l_j2": -0.5}, velocity={"l_j2": 0.3},
                             effort={"l_j2": 2.0}))
    assert seen["position"]["l_j2"] == pytest.approx(2.5)    # raw
    assert seen["velocity"]["l_j2"] == pytest.approx(-0.3)   # dir flips
    assert seen["effort"]["l_j2"] == pytest.approx(-2.0)     # dir flips


def test_round_trip_model_raw_model(tmp_path):
    sub, _ = _sub(tmp_path)
    for name in ("l_j1", "l_j2"):
        for model in (-0.4, 0.0, 0.37):
            raw = sub._to_raw(name, model)
            assert sub._to_model(name, raw) == pytest.approx(model, abs=1e-9)


# ── the commandable range (what a teleop slider clamps to) ──────────────────

def test_commandable_ranges_are_model_frame(tmp_path):
    sub, _ = _sub(tmp_path)
    r = sub.commandable_ranges()
    assert set(r) == {"l_j1", "l_j2"}            # l_j3 uncalibrated -> absent
    assert r["l_j1"] == [pytest.approx(-1.0), pytest.approx(1.0)]
    lo, hi = r["l_j2"]                           # dir=-1: window flips sign
    assert lo == pytest.approx(-1.0) and hi == pytest.approx(1.0)


def test_commandable_ranges_track_recalibration(tmp_path):
    """The store hot-reloads; the exposed range must follow with no restart."""
    sub, _ = _sub(tmp_path)
    assert sub.commandable_ranges()["l_j1"][1] == pytest.approx(1.0)
    tree = json.loads(json.dumps(TREE))
    tree["arm"]["openarm-dual"]["calib"]["l_j1"]["raw_max"] = 2.0
    root = tmp_path / "calibration_data"
    (root / "mabel.json").write_text(json.dumps(tree))
    sub.recalibrate(CalibrationStore(root=root,
                                     mabel_root=tmp_path / "MABEL").load())
    assert sub.commandable_ranges()["l_j1"][1] == pytest.approx(1.5)


# ── the multi-turn fold (the l_j4 cooking bug) ──────────────────────────────

def test_targets_fold_into_the_motors_revolution(tmp_path):
    """THE 2-pi BUG: a joint whose window sits a full turn from where the motor
    boots must be commanded in the MOTOR's revolution. Emitting the window's
    revolution reads to the motor as 'rotate -356 deg' — a full-torque stall
    against a stop, which is what cooked l_j4."""
    tau = 2.0 * math.pi
    sub, drv = _sub(tmp_path)
    drv._pos["l_j1"] = 0.5 + tau                 # motor booted one turn up
    sub.set_targets({"l_j1": 0.5})               # model: "stay where you are"
    cmd = drv._targets["l_j1"]
    assert cmd == pytest.approx(0.5 + tau, abs=1e-6), \
        "target was emitted in the window's revolution, not the motor's"
    assert abs(cmd - drv._pos["l_j1"]) < 0.05    # i.e. "stay put", not a turn


# ── safety ──────────────────────────────────────────────────────────────────

def test_stop_holds_in_place_and_keeps_torque(tmp_path):
    """stop() is hold-in-place, never torque-off: dropping torque under load
    would DROP whatever the arm is holding."""
    sub, drv = _sub(tmp_path)
    drv._pos["l_j1"] = 0.42
    sub.stop()
    assert drv._targets["l_j1"] == pytest.approx(0.42)   # raw, no clamp path
    assert sub.is_torque_enabled


# ── the end-stop margin is WIRED, not just supported by JointCalib ───────────

def test_subsystem_applies_the_end_stop_margin(tmp_path):
    """REGRESSION (2026-07-16, "J4 always hits the limit and stalls -> overheat +
    over-current"): the SUBSYSTEM must hold end_stop_margin off the calibrated hard
    stop. The calibrated window IS the measured stop, so a target clamped exactly
    onto it parks the motor against the stop at full kp*error — and the HAL has no
    stall/thermal guard to catch that any more.

    This asserts the WIRING (cfg -> _to_raw), not JointCalib's support for a margin:
    passing margin= by hand in the test would still pass if the subsystem stopped
    threading it through."""
    root = tmp_path / "calibration_data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "mabel.json").write_text(json.dumps(TREE))
    store = CalibrationStore(root=root, mabel_root=tmp_path / "MABEL").load()
    drv = MockJointGroupDriver(JOINTS, alpha=1.0)
    drv.connect()
    sub = ArmsSubsystem("left_arm", {"calib_slice": "arm/openarm-dual",
                                     "end_stop_margin": 0.1}, {},
                        store=store, driver=drv)
    sub.set_torque(True)
    # l_j1 window is raw [-1, +1] (dir=+1, model==raw). Command far past the stop.
    sub.set_targets({"l_j1": 99.0})
    assert drv._targets["l_j1"] == pytest.approx(0.9), \
        "target was clamped ONTO the mechanical stop (margin not applied)"
    sub.set_targets({"l_j1": -99.0})
    assert drv._targets["l_j1"] == pytest.approx(-0.9)


def test_commandable_ranges_carry_the_margin(tmp_path):
    """/arms/joint_limits is the slider's authority — it must advertise the range a
    client may COMMAND, not the mechanical stop. If it advertised the stop, a slider
    dragged to its own maximum would ask the motor to drive into it."""
    root = tmp_path / "calibration_data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "mabel.json").write_text(json.dumps(TREE))
    store = CalibrationStore(root=root, mabel_root=tmp_path / "MABEL").load()
    drv = MockJointGroupDriver(JOINTS, alpha=1.0)
    drv.connect()
    sub = ArmsSubsystem("left_arm", {"calib_slice": "arm/openarm-dual",
                                     "end_stop_margin": 0.1}, {},
                        store=store, driver=drv)
    assert sub.commandable_ranges()["l_j1"] == [pytest.approx(-0.9),
                                                pytest.approx(0.9)]


def test_margin_never_inverts_a_narrow_range(tmp_path):
    """A joint whose whole range is narrower than the margin must not report an
    INVERTED [lo>hi] range — a slider would clamp everything to nonsense."""
    tree = json.loads(json.dumps(TREE))
    tree["arm"]["openarm-dual"]["calib"]["l_j1"] = {
        "raw_min": -0.02, "raw_max": 0.02, "dir": 1, "model_mid": 0.0}
    root = tmp_path / "calibration_data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "mabel.json").write_text(json.dumps(tree))
    store = CalibrationStore(root=root, mabel_root=tmp_path / "MABEL").load()
    drv = MockJointGroupDriver(JOINTS, alpha=1.0)
    drv.connect()
    sub = ArmsSubsystem("left_arm", {"calib_slice": "arm/openarm-dual",
                                     "end_stop_margin": 0.1}, {},
                        store=store, driver=drv)
    lo, hi = sub.commandable_ranges()["l_j1"]
    assert lo <= hi, "margin inverted a narrow range"


# ── gravity feedforward arrives with the RIGHT SIGN ─────────────────────────

def test_model_frame_effort_reaches_the_motor_dir_corrected(tmp_path):
    """The gravity FF contract: controller/ computes tau_g in the MODEL frame and puts
    it on /arms/command.effort; the subsystem dir-corrects it into the raw motor frame.

    THE TRAP (2026-07-16): ArmGravityCompensator.tff() defaults to frame="raw" — it has
    ALREADY applied dir. Publishing that would double-apply dir here and drive every
    dir=-1 joint the WRONG WAY at full feed-forward torque, into its stop. The gateway
    must ask for frame="model". This pins the subsystem half of that contract."""
    seen = {}
    sub, drv = _sub(tmp_path)
    drv.command = lambda position, velocity=None, kp=None, kd=None, effort=None: \
        seen.update(effort=effort)
    # l_j1 dir=+1, l_j2 dir=-1 (see TREE)
    sub.command(JointCommand(position={"l_j1": 0.0, "l_j2": 0.0},
                             effort={"l_j1": +2.0, "l_j2": +2.0}))
    assert seen["effort"]["l_j1"] == pytest.approx(+2.0), "dir=+1 must pass through"
    assert seen["effort"]["l_j2"] == pytest.approx(-2.0), \
        "dir=-1 effort was NOT flipped into the raw frame — gravity would push the " \
        "joint the wrong way"
