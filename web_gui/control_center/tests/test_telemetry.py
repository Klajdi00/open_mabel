"""Telemetry hub: state store semantics + the mock plant's full-coverage feed."""
import time

from control_center.server import telemetry
from control_center.server.joints import JOINT_NAMES, JOINT_GROUP


def test_state_update_and_snapshot_order():
    st = telemetry.TelemetryState()
    st.update_joints(["left_arm_1", "nonexistent_joint"], [0.5, 9.9], [0.1, 0], [0.2, 0])
    st.update_targets(["left_arm_1"], [0.6])
    st.update_temps(["left_thumb_pip"], [61.5])
    snap = st.snapshot()
    i = JOINT_NAMES.index("left_arm_1")
    assert snap["p"][i] == 0.5 and snap["v"][i] == 0.1 and snap["e"][i] == 0.2
    assert snap["tgt"][i] == 0.6
    j = JOINT_NAMES.index("left_thumb_pip")
    assert snap["temp"][j] == 61.5
    # unknown joints never leak into the arrays
    assert len(snap["p"]) == 60
    assert all(v is None or isinstance(v, float) for v in snap["p"])


def test_state_ignores_short_arrays():
    st = telemetry.TelemetryState()
    st.update_joints(["left_arm_1", "left_arm_2"], [0.1])     # vel/eff absent
    snap = st.snapshot()
    assert snap["p"][JOINT_NAMES.index("left_arm_1")] == 0.1
    assert snap["p"][JOINT_NAMES.index("left_arm_2")] is None


def _wait_for(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_mock_source_full_coverage():
    st = telemetry.TelemetryState()
    src = telemetry.MockSource(st, hz=100.0)
    src.start()
    try:
        assert _wait_for(lambda: st.snapshot()["p"][0] is not None)
        snap = st.snapshot()
        assert snap["source"] == "mock"
        # every joint gets pos+vel+eff+target
        for key in ("p", "v", "e", "tgt"):
            assert all(x is not None for x in snap[key]), key
        # temps only on hand modules
        for i, n in enumerate(JOINT_NAMES):
            hand = JOINT_GROUP[n] in ("left_hand", "right_hand")
            assert (snap["temp"][i] is not None) == hand, n
        # synthetic scan: 360 bins, plausible room ranges
        assert len(snap["scan"]) == 360
        assert all(0.5 < r < 4.0 for r in snap["scan"])
        assert snap["torque"] == {"arms": True, "handL": True, "handR": True}
    finally:
        src.stop()


def test_mock_lag_joints_exceed_warn_threshold():
    """The mock intentionally lags a few joints so the UI grading paths are
    exercised: their |pos−target| must cross the 0.18 rad red threshold
    sometimes and always cross the 0.05 warn threshold."""
    st = telemetry.TelemetryState()
    src = telemetry.MockSource(st, hz=200.0)
    src.start()
    try:
        assert _wait_for(lambda: st.snapshot()["p"][0] is not None)
        maxerr = 0.0
        for _ in range(30):
            snap = st.snapshot()
            i = JOINT_NAMES.index("left_arm_4")
            maxerr = max(maxerr, abs(snap["p"][i] - snap["tgt"][i]))
            time.sleep(0.02)
        assert maxerr > 0.18
        # and a well-tracking joint stays green
        snap = st.snapshot()
        k = JOINT_NAMES.index("neck_yaw")
        assert abs(snap["p"][k] - snap["tgt"][k]) < 0.05
    finally:
        src.stop()


def test_make_source_mock_kind():
    st = telemetry.TelemetryState()
    src = telemetry.make_source(st, kind="mock")
    try:
        assert isinstance(src, telemetry.MockSource)
        assert st.source == "mock"
    finally:
        src.stop()
