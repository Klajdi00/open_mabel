"""CalibrationStore + JointCalib: the read-first mirror policy, the fallback
chain, per-family anchor math (center vs neutral — MUST match the upstream
bridges), atomic save_slice, and the watch/hot-reload loop."""
import json
import math
import os
import time

import pytest

from mabel_hw.calibration import CalibrationStore, JointCalib
from mabel_hw.calibration.schema import CENTER, NEUTRAL
from mabel_hw.errors import CalibrationError

from conftest import BRIDGE_DIR


ARM_ENTRY = {"raw_min": -2.0, "raw_max": 4.0, "dir": -1,
             "model_mid": 0.5, "reliability": 1.01}
NECK_ENTRY = {"raw_min": -1.5, "raw_max": 1.0, "dir": 1, "raw_neutral": -0.25}

TREE = {
    "arm": {"openarm-dual": {
        "teensy_ip": "192.168.124.11", "port": 5800, "stream_hz": 1000,
        "calib": {"l_j1": ARM_ENTRY}}},
    "neck": {"rssr": {"calib": {
        "motor_L1": NECK_ENTRY,
        "_task": {"pitch_min": -0.7, "pitch_max": 0.65,
                  "yaw_min": -1.4, "yaw_max": 1.4}}}},
    "base": {"maxswerve": {"teensy_ip": "192.168.124.10", "port": 5800}},
}


# ── anchor math ────────────────────────────────────────────────────────────────

def test_center_anchored_roundtrip():
    jc = JointCalib.from_dict(ARM_ENTRY, "l_j1")
    assert jc.anchor == CENTER and jc.raw_mid == pytest.approx(1.0)
    # upstream arm map: model = model_mid + dir*(raw - raw_mid)
    assert jc.to_model(1.0) == pytest.approx(0.5)
    assert jc.to_model(2.0) == pytest.approx(0.5 - 1.0)
    for m in (-1.0, 0.0, 0.5, 2.3):
        assert jc.to_model(jc.to_raw(m)) == pytest.approx(m)


def test_neutral_anchored_roundtrip():
    jc = JointCalib.from_dict(NECK_ENTRY, "motor_L1")
    assert jc.anchor == NEUTRAL
    # upstream neck map: model = dir*(raw - raw_neutral); model 0 == level head
    assert jc.to_model(-0.25) == pytest.approx(0.0)
    assert jc.to_model(0.75) == pytest.approx(1.0)
    assert jc.to_raw(0.5) == pytest.approx(0.25)


def test_to_raw_clamps_to_endstops():
    jc = JointCalib.from_dict(ARM_ENTRY, "l_j1")
    assert jc.to_raw(1e6) in (jc.raw_min, jc.raw_max)
    assert jc.to_raw(-1e6) in (jc.raw_min, jc.raw_max)
    nk = JointCalib.from_dict(NECK_ENTRY, "m")
    assert nk.to_raw(1e6) == nk.raw_max


def test_out_of_range_target_saturates_at_the_NEAR_endstop():
    """An out-of-range target must clamp to the stop it is heading TOWARD.

    Regression (2026-07-16): to_raw folded the TARGET toward `cur`, so any target
    more than pi outside the window wrapped into the wrong revolution and clamped
    to the OPPOSITE stop — a +4.0 rad target drove the joint to its NEGATIVE stop.
    Since the HAL's only position limit is this clamp, getting the side wrong parks
    the motor against a mechanical stop at full PD torque (how l_j4 cooked). The
    map must stay monotonic in `model` for every target, at any distance."""
    jc = JointCalib.from_dict(
        {"raw_min": -1.0, "raw_max": 1.0, "dir": 1, "model_mid": 0.0}, "j")
    for far in (1.2, 2.0, 4.0, 99.0):
        assert jc.to_raw(far, 0.0) == pytest.approx(jc.raw_max), \
            f"+{far} clamped to the wrong stop"
        assert jc.to_raw(-far, 0.0) == pytest.approx(jc.raw_min), \
            f"-{far} clamped to the wrong stop"


def test_out_of_range_target_saturates_correctly_in_the_motors_revolution():
    """...and the same holds when the encoder booted a whole turn off the window:
    clamp in the window frame, command in the motor's frame."""
    tau = 2.0 * math.pi
    jc = JointCalib.from_dict(
        {"raw_min": -1.0, "raw_max": 1.0, "dir": 1, "model_mid": 0.0}, "j")
    cur = tau                                     # motor one turn up from raw_mid
    assert jc.to_raw(99.0, cur) == pytest.approx(jc.raw_max + tau)
    assert jc.to_raw(-99.0, cur) == pytest.approx(jc.raw_min + tau)
    assert jc.to_raw(0.5, cur) == pytest.approx(0.5 + tau)     # in-range: stay put


def test_bad_entries_raise():
    with pytest.raises(CalibrationError):
        JointCalib.from_dict({"raw_min": 1.0, "raw_max": -1.0, "dir": 1}, "x")
    with pytest.raises(CalibrationError):
        JointCalib.from_dict({"raw_min": 0, "raw_max": 1, "dir": 2}, "x")
    with pytest.raises(CalibrationError):
        JointCalib.from_dict({}, "x")


# ── store: precedence + fallback ───────────────────────────────────────────────

def _store(tmp_path, mirror=None, legacy=None):
    root = tmp_path / "calibration_data"
    mabel = tmp_path / "MABEL"
    upstream = mabel / "upstream" / "MABEL" / "modules" / "calib" / "config"
    if mirror is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "mabel.json").write_text(json.dumps(mirror))
    if legacy is not None:
        upstream.mkdir(parents=True, exist_ok=True)
        (upstream / "mabel.json").write_text(json.dumps(legacy))
    return CalibrationStore(root=root, mabel_root=mabel).load()


def test_mirror_wins_over_legacy(tmp_path):
    legacy = json.loads(json.dumps(TREE))
    legacy["arm"]["openarm-dual"]["port"] = 9999
    store = _store(tmp_path, mirror=TREE, legacy=legacy)
    assert store.slice("arm/openarm-dual/port") == 5800
    assert "calibration_data" in str(store.source["mabel.json"])


def test_legacy_fallback_when_mirror_absent(tmp_path):
    store = _store(tmp_path, mirror=None, legacy=TREE)
    assert store.slice("arm/openarm-dual/port") == 5800
    assert "upstream" in str(store.source["mabel.json"])


def test_no_tree_at_all_is_nonfatal(tmp_path):
    store = _store(tmp_path)
    assert store.slice("arm/openarm-dual") is None
    assert store.joint("arm/openarm-dual", "l_j1") is None
    assert store.group("arm/openarm-dual", ["l_j1"]) == {}
    with pytest.raises(CalibrationError):
        store.group("arm/openarm-dual", ["l_j1"], strict=True)


def test_slice_joint_group_task_connection(tmp_path):
    store = _store(tmp_path, mirror=TREE)
    jc = store.joint("arm/openarm-dual", "l_j1")
    assert jc is not None and jc.anchor == CENTER
    grp = store.group("neck/rssr", ["motor_L1", "motor_head"])
    assert set(grp) == {"motor_L1"}          # motor_head uncalibrated -> absent
    task = store.task_limits("neck/rssr")
    assert task["pitch_max"] == pytest.approx(0.65)
    conn = store.connection("arm/openarm-dual")
    assert conn == {"teensy_ip": "192.168.124.11", "port": 5800, "stream_hz": 1000}
    # '_task'/'_doc' are metadata blocks, never joints
    assert store.joint("neck/rssr", "_task") is None


def test_save_slice_atomic_and_readback(tmp_path):
    store = _store(tmp_path, mirror=TREE)
    store.save_slice("neck/rssr/calib/motor_head",
                     {"raw_min": -1.0, "raw_max": 2.0, "dir": 1,
                      "raw_neutral": 0.4})
    assert store.joint("neck/rssr", "motor_head").raw_neutral == pytest.approx(0.4)
    on_disk = json.loads((tmp_path / "calibration_data" / "mabel.json").read_text())
    assert on_disk["neck"]["rssr"]["calib"]["motor_head"]["raw_neutral"] == 0.4
    assert on_disk["arm"]["openarm-dual"]["calib"]["l_j1"] == ARM_ENTRY  # untouched
    assert not (tmp_path / "calibration_data" / "mabel.json.tmp").exists()


def test_save_slice_creates_mirror_from_fallback(tmp_path):
    """First bridge-local save on a robot still running legacy-only files must
    seed the mirror with the WHOLE fallback tree, not just the saved slice."""
    store = _store(tmp_path, mirror=None, legacy=TREE)
    store.save_slice("base/maxswerve/port", 5801)
    on_disk = json.loads((tmp_path / "calibration_data" / "mabel.json").read_text())
    assert on_disk["base"]["maxswerve"]["port"] == 5801
    assert on_disk["arm"]["openarm-dual"]["calib"]["l_j1"] == ARM_ENTRY


def test_watch_hot_reload(tmp_path):
    store = _store(tmp_path, mirror=TREE)
    changed = []
    stop = store.watch(changed.append, period_s=0.05)
    try:
        p = tmp_path / "calibration_data" / "mabel.json"
        tree = json.loads(p.read_text())
        tree["arm"]["openarm-dual"]["calib"]["l_j1"]["model_mid"] = 0.75
        time.sleep(0.1)                       # ensure a distinct mtime
        p.write_text(json.dumps(tree))
        os.utime(p)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not changed:
            time.sleep(0.02)
        assert changed and changed[0] == "mabel.json"
        assert store.joint("arm/openarm-dual", "l_j1").model_mid == pytest.approx(0.75)
    finally:
        stop()


# ── parity: the seeded repo mirror matches its sources ─────────────────────────

def test_seeded_mirror_matches_upstream_source():
    """The families the upstream calib server OWNS (arm, hand — every studio save routes
    through it and mirrors here) must be byte-identical in both files, so a studio save can
    never silently revert a hand-fixed window (the 2026-07-16 l_j4 re-seed trap). The
    master additionally carries direct-write slices (torso/lift/neck — studio writes them
    straight into this file, they never pass through the upstream server), so whole-file
    equality is no longer the invariant; the server's _deep_merge preserves them on save."""
    mirror = os.path.join(BRIDGE_DIR, "calibration_data", "mabel.json")
    src = os.path.join(os.path.dirname(BRIDGE_DIR), "upstream", "MABEL",
                       "modules", "calib", "config", "mabel.json")
    if not os.path.exists(src):
        pytest.skip("upstream checkout absent")
    mirror_tree, src_tree = json.load(open(mirror)), json.load(open(src))
    for family in ("arm", "hand"):
        assert mirror_tree.get(family) == src_tree.get(family), (
            f"'{family}' slice diverged between the HAL master and the upstream source — "
            "the next studio save will silently revert one of them")


def test_repo_store_loads_and_serves_real_slices():
    store = CalibrationStore().load()         # the real repo calibration_data/
    assert "calibration_data" in str(store.source["mabel.json"])
    grp = store.group("arm/openarm-dual",
                      [f"{s}_j{i}" for s in "lr" for i in range(1, 8)])
    assert len(grp) == 14                     # all 14 arm joints calibrated
    assert all(j.anchor == CENTER for j in grp.values())
    neck = store.group("neck/rssr", ["motor_L1", "motor_L2", "motor_head"])
    assert len(neck) == 3
    assert all(j.anchor == NEUTRAL for j in neck.values())
    assert store.task_limits("neck/rssr")
    hp = store.hand_calibration_path("right")
    assert hp is not None and "calibration_data" in str(hp)
