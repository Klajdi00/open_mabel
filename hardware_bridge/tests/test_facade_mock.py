"""MabelHardware(mock=True) end-to-end: the full facade on the mock plants —
snapshot schema, claim/write/read round trips per subsystem, lease gating at
the facade layer, estop bypass, degraded subsystems, calibration attachment,
subscribe(). This is the API contract every consumer programs against."""
import time

import pytest

from mabel_hw import MabelHardware, RobotState
from mabel_hw.errors import GroupBusy, NotLeaseHolder, SubsystemUnavailable
from mabel_hw.types import JointCommand


@pytest.fixture()
def hw():
    with MabelHardware(mock=True) as hw:
        yield hw


def test_open_close_lifecycle():
    hw = MabelHardware(mock=True)
    assert not hw.is_open and not hw.left_arm.available
    hw.open()
    try:
        assert hw.is_open and hw.left_arm.available and hw.base.available
    finally:
        hw.close()
    assert not hw.is_open and not hw.left_arm.available


def test_read_snapshot_schema(hw):
    st = hw.read()
    assert isinstance(st, RobotState)
    assert st.left_arm is not None and st.right_arm is not None
    assert st.neck is not None and st.torso is not None
    assert st.base is not None and st.lift is not None
    assert st.left_hand is not None and st.right_hand is not None
    assert st.sensors is not None and st.sensors.lidar is not None
    assert len(st.sensors.cameras) == 5
    assert st.age() < 1.0
    # snapshots are frozen — a reader can never mutate shared state
    with pytest.raises(Exception):
        st.base = None


def test_arm_claim_write_read(hw):
    assert hw.left_arm.names == ("l_j1", "l_j2", "l_j3", "l_j4",
                                 "l_j5", "l_j6", "l_j7")
    with hw.left_arm.claim("test") as arm:
        arm.set_torque(True)
        arm.set_joint_positions({"l_j1": 0.4})
        for _ in range(20):
            hw.left_arm.tick()
    st = hw.read().left_arm
    assert st.as_dict()["l_j1"] == pytest.approx(0.4, abs=1e-3)
    assert st.torque_enabled


def test_single_writer_per_group(hw):
    with hw.left_arm.claim("one"):
        with pytest.raises(GroupBusy):
            hw.left_arm.claim("two")
        hw.right_arm.claim("two").release()      # other group: fine
    hw.left_arm.claim("three").release()         # released: free again


def test_stale_writer_is_locked_out(hw):
    old = hw.base.claim("teleop")
    new = hw.base.claim("recovery", steal=True)
    with pytest.raises(NotLeaseHolder):
        old.set_twist(0.5, 0.0, 0.0)             # command NOT sent
    new.set_twist(0.2, 0.0, 0.0)
    new.release()


def test_base_twist_and_estop(hw):
    with hw.base.claim("drive") as base:
        base.set_twist(0.3, 0.0, 0.0)
        for _ in range(20):
            hw.base.tick()
        st = hw.read().base
        assert st.connected and not st.estop
        assert max(abs(m.drive_velocity) for m in st.modules) > 0.2
        # estop is lease-free and beats the writer
        hw.base.stop()
        for _ in range(20):
            hw.base.tick()
        st = hw.read().base
        assert st.estop
        assert max(abs(m.drive_velocity) for m in st.modules) < 0.05
    hw.clear_estop()
    assert hw.read().base.estop is False


def test_global_estop_needs_no_lease(hw):
    hw.left_arm.claim("someone-else")            # group held by another writer
    hw.estop()                                   # must not raise, ever


def test_hands_torque_and_write(hw):
    st = hw.read().left_hand
    assert st.connected and not st.torque_enabled
    initial = st.position["index_mcp"]
    with hw.left_hand.claim("grasp") as hand:
        hand.set_joint_positions({"index_mcp": 0.5})   # torque off: dropped
        assert hw.read().left_hand.position["index_mcp"] == pytest.approx(initial, abs=0.02)
        hand.set_torque(True)
        hand.set_joint_positions({"index_mcp": 0.5})
        assert hw.read().left_hand.position["index_mcp"] == pytest.approx(0.5, abs=0.05)


def test_lift_write_and_state(hw):
    with hw.lift.claim("gui") as lift:
        lift.set_height(0.33)
    st = hw.read().lift
    assert st.connected and st.homed
    assert st.height == pytest.approx(0.33, abs=1e-6)


def test_neck_gaze_serial_direct_drive(hw):
    """The as-built serial neck: task space IS motor space, so gaze is ALWAYS
    available (no neck_studio model) and each axis drives its motor 1:1. All three
    motors are wired (yaw since 2026-07-27; pitch/roll on the arms Teensy's serial
    bus since 2026-09-30), so yaw reads a real angle, not None."""
    assert hw.neck.task_space_available          # serial: no linkage model needed
    with hw.neck.claim("gaze") as neck:
        neck.set_torque(True)                    # MIT motors hold, not move, w/o it
        neck.set_gaze(0.1, 0.05, 0.0)
        for _ in range(30):
            hw.neck.tick()
        st = hw.read().neck
        assert st.pitch == pytest.approx(0.1, abs=0.05)
        assert st.roll == pytest.approx(0.05, abs=0.05)
        assert st.yaw == pytest.approx(0.0, abs=0.05)   # commanded 0.0; yaw IS wired


def test_joint_command_full_mit(hw):
    with hw.torso.claim("wbc") as torso:
        torso.set_torque(True)
        torso.command(JointCommand(position={"torso": 0.2},
                                   kp={"torso": 40.0}, kd={"torso": 2.0}))
        for _ in range(20):
            hw.torso.tick()
    assert hw.read().torso.as_dict()["torso"] == pytest.approx(0.2, abs=1e-3)


def test_degraded_subsystem_reads_none():
    hw = MabelHardware(mock=True, subsystems=["base", "lift"])
    with hw:
        st = hw.read()
        assert st.base is not None and st.lift is not None
        assert st.left_arm is None and st.left_hand is None
        assert not hw.left_arm.available
        with pytest.raises(SubsystemUnavailable):
            hw.left_arm.claim("x").set_joint_positions({"l_j1": 0})
        hw.estop()                               # still safe with holes


def test_calibration_store_attached(hw):
    assert hw.calibration is not None
    assert hw.calibration.slice("arm/openarm-dual/port") == 5800


def test_subscribe_stream(hw):
    got = []
    sub = hw.subscribe(got.append, rate_hz=100.0)
    time.sleep(0.15)
    sub.cancel()
    n = len(got)
    assert n >= 5                                # ~15 expected; generous floor
    assert all(isinstance(s, RobotState) for s in got)
    time.sleep(0.1)
    assert len(got) == n                         # cancel really stops it


def test_claims_snapshot(hw):
    lease = hw.base.claim("dds:/mabel/hw/base/cmd_vel")
    snap = hw.claims()
    assert snap["base"].owner == "dds:/mabel/hw/base/cmd_vel"
    assert snap["left_arm"] is None
    lease.release()


def test_no_flock_in_mock_mode(tmp_path, monkeypatch):
    """mock mode must not touch the production lock file."""
    lock = tmp_path / "hb.lock"
    monkeypatch.setenv("MABEL_HW_LOCK", str(lock))
    with MabelHardware(mock=True) as hw:
        hw.read()
    assert not lock.exists()
