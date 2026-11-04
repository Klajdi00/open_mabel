"""test_dynamixel.py — the dynamixel_sdk-backed DynamixelGroup seam.

Hardware-free: the group is driven against a FakeBus so no serial port (and no
dynamixel_sdk) is touched. The live-bus path is exercised by experiments/exp_neck.py
and the neck probe, not here.
"""
import math

import pytest

from mabel_hw.drivers import build_joint_driver
from mabel_hw.drivers import dynamixel as dx
from mabel_hw.drivers.seam import GroupFeedback


# ── helpers ──────────────────────────────────────────────────────────────────────
def test_signed_helpers():
    assert dx._s16(0xFFFF) == -1 and dx._s16(0x0001) == 1
    assert dx._s32(0xFFFFFFFF) == -1 and dx._s32(0x00000002) == 2


# ── the group seam, against a fake bus ───────────────────────────────────────────
class FakeBus:
    """Stands in for DynamixelBus: two servos at fixed state, records goals."""

    def __init__(self, present):
        self._present = dict(present)               # id -> position rad
        self._open = False
        self.goals = None
        self.torque = {}

    def connect(self):
        self._open = True

    def connected(self):
        return self._open

    def close(self):
        self._open = False

    @property
    def port(self):
        return "/dev/fake"

    def ping(self, i):
        return 1090 if i in self._present else None

    def read_block(self, i):
        if i not in self._present:
            return None
        return dict(err=0, current_ma=0.0, velocity=0.0,
                    position=self._present[i], voltage=12.1, temperature=31.0)

    def write_torque(self, i, on):
        self.torque[i] = on
        return True

    def write_profile(self, i, velocity, accel):
        self.profile = {"id": i, "vel": velocity, "accel": accel}

    def write_position_p_gain(self, i, gain):
        # soft-head stiffness write (reg 84) — recorded so tests can assert on it
        self.p_gain = getattr(self, "p_gain", {})
        self.p_gain[i] = int(gain)
        return True

    def hardware_error(self, i):
        return getattr(self, "hw_err", {}).get(i, 0)   # 0 = healthy by default

    def reboot(self, i):
        self.rebooted = getattr(self, "rebooted", []) + [i]

    def read_all(self, ids):
        return {i: self.read_block(i) for i in ids if self.read_block(i) is not None}

    def sync_write_goal(self, goals):
        self.goals = dict(goals)


@pytest.fixture()
def group():
    g = dx.DynamixelGroup({"neck_pitch": 1, "neck_roll": 2}, device="/dev/fake", baud=1)
    g._bus = FakeBus({1: 0.5, 2: -0.25})
    return g


def test_names_and_id_map():
    g = dx.DynamixelGroup({"neck_pitch": 1, "neck_roll": 2}, device="", baud=1)
    assert g.names == ("neck_pitch", "neck_roll")


def test_connect_pings_and_marks_live(group):
    group.connect()
    assert group.connected()
    fb = group.read()
    assert isinstance(fb, GroupFeedback)
    assert all(fb.live)


def test_connect_asserts_limp_so_a_retained_torque_boots_off(group):
    """X-series servos LATCH Torque Enable in the servo — it survives a host serial
    reconnect. connect() must ASSERT the safe limp state (write torque-disable) so a
    neck energised in a PREVIOUS session boots OFF, not stiff-and-holding. It only
    re-energises on an explicit set_torque(True) (the operator's 'Torque ON')."""
    group.connect()
    assert group._bus.torque == {1: False, 2: False}   # both written torque-OFF
    assert group.is_torque_enabled is False


def test_tick_polls_present_into_read(group):
    group.connect()
    group._poll_once()
    fb = group.read()
    by = dict(zip(fb.names, fb.position))
    assert by["neck_pitch"] == pytest.approx(0.5)
    assert by["neck_roll"] == pytest.approx(-0.25)


def test_silent_unplug_marks_down_then_auto_reconnects(group):
    """A USB unplug that surfaces as silent read-timeouts (port stays 'open', every
    servo goes quiet) must read as DOWN and then AUTO-RECOVER on replug — without it
    the driver would ping a dead-open handle forever. But a SINGLE missed read is
    graced (see _LIVENESS_GRACE_S): only SUSTAINED silence declares down, so a transient
    timeout under CPU load no longer close/reopen/limp-flaps the neck."""
    group.connect()
    group._poll_once()
    assert group.connected() and all(group.read().live)

    # SILENT unplug: servos stop answering, but the FakeBus stays open (no wire fault).
    group._bus._present.clear()
    group._poll_once()                                        # ONE miss -> graced, still live
    assert all(group.read().live)                       # transient miss held, not dropped

    # Sustained silence past the grace window -> now DOWN.
    group._last_ok = {n: dx.monotonic() - dx._LIVENESS_GRACE_S - 1.0 for n in group.names}
    group._poll_once()
    assert not any(group.read().live)
    assert not group.connected()                        # open, but nothing alive = down

    # service() sees DOWN, force-recloses + re-resolves; nothing answers yet -> stays down.
    group._service_once()
    assert group._down_logged
    assert not any(group.read().live)

    # REPLUG: servos answer again -> next service() reconnects and clears the DOWN edge.
    # (Reset the reconnect backoff to simulate the elapsed retry interval — a dead bus is
    # retried with exponential backoff so it can't stall the other HAL modules.)
    group._bus._present.update({1: 0.5, 2: -0.25})
    group._reconnect_at = 0.0
    group._service_once()
    assert not group._down_logged
    assert group.connected() and all(group.read().live)


def test_service_never_crashes_on_reconnect_fault(group):
    """A transient open failure on replug must NOT propagate out of service() and
    crash the bridge (the ROS timer runs it). It logs + retries instead."""
    group.connect()
    group._poll_once()
    group._bus._present.clear()                          # go silent
    group._last_ok = {n: dx.monotonic() - dx._LIVENESS_GRACE_S - 1.0 for n in group.names}
    group._poll_once()                                         # sustained silence -> DOWN
    assert not any(group.read().live)

    # Simulate serial.Serial() blowing up mid-replug: connect() raises a RAW error.
    def _boom():
        raise RuntimeError("device or resource busy")   # not SubsystemUnavailable

    group._bus.connect = _boom
    group._reconnect_at = 0.0
    group._service_once()                                      # must NOT raise
    assert not any(group.read().live)                    # stays down, no crash

    # Once the open succeeds again, the next service() recovers (backoff reset = interval elapsed).
    group._bus = FakeBus({1: 0.5, 2: -0.25})
    group._reconnect_at = 0.0
    group._service_once()
    assert group.connected() and all(group.read().live)


def test_reconnect_backoff_does_not_hog_the_loop(group):
    """A down bus is retried with EXPONENTIAL BACKOFF, not on every service() call. Each
    attempt PINGS, and the dynamixel_sdk ping busy-waits its read timeout holding the GIL,
    so at 1 Hz a dead neck starved the OTHER HAL modules (the hands fell to ~4 Hz). Within
    one backoff window, repeated service() calls make NO new reconnect attempt."""
    group.connect(); group._poll_once()
    group._bus._present.clear()
    group._last_ok = {n: dx.monotonic() - dx._LIVENESS_GRACE_S - 1.0 for n in group.names}
    group._poll_once()                                          # sustained silence -> DOWN
    calls = {"n": 0}
    orig = group._bus.connect
    def _counted():
        calls["n"] += 1
        orig()
    group._bus.connect = _counted
    group._service_once(); group._service_once(); group._service_once()     # 3 rapid calls == one window
    assert calls["n"] == 1, f"backoff failed: {calls['n']} reconnect attempts in one window"
    group._reconnect_at = 0.0                              # window elapsed
    group._service_once()
    assert calls["n"] == 2                                 # exactly one more attempt


def test_tick_starts_dedicated_io_thread_and_close_stops_it(group):
    """tick()/service() are NON-BLOCKING — they only keep the dedicated `dxl-neck-io`
    daemon alive, which owns ALL serial I/O so a down neck can never stall the ROS
    executor / the hands (mirrors the arm's openarm-rx thread). The thread polls present
    state into read() on its own; close() stops it."""
    import time
    group.connect()
    assert group._io_thread is None                        # not started until a ROS tick
    group.tick()                                            # non-blocking: just starts the io thread
    assert group._io_thread is not None and group._io_thread.is_alive()
    for _ in range(100):                                   # thread polls independently
        by = dict(zip(group.read().names, group.read().position))
        if by.get("neck_pitch") == pytest.approx(0.5):
            break
        time.sleep(0.01)
    assert dict(zip(group.read().names, group.read().position))["neck_pitch"] == pytest.approx(0.5)
    group.close()
    assert group._io_thread is None                        # daemon joined/cleared on close


def test_torque_off_writes_no_goal(group):
    group.connect()
    group.set_targets({"neck_pitch": 0.3})
    group._poll_once()
    assert group._bus.goals is None                 # limp: no goal stream w/o torque


def test_torque_on_sync_writes_ticks(group):
    group.connect()
    group.set_torque(True)
    assert group._bus.torque == {1: True, 2: True}
    group.set_targets({"neck_pitch": math.pi / 2})  # 90 deg -> 1024 ticks
    group._poll_once()
    assert group._bus.goals[1] == pytest.approx(1024, abs=1)


def test_unknown_joint_rejected(group):
    with pytest.raises(KeyError):
        group.set_targets({"nose": 0.0})


def test_health_reports_temp_and_current(group):
    group.connect()
    group._poll_once()
    h = group.health()
    assert h["neck_pitch"]["tempC"] == 31.0
    assert "currentmA" in h["neck_pitch"]


# ── full telemetry: pos + vel + torque + temp, like every other joint ─────────────
class _FakeBusTelemetry(FakeBus):
    """FakeBus that also reports nonzero current/velocity/temperature so the
    torque estimate (tau = Kt * I) and the rate/temp pass-through are testable."""

    def read_block(self, i):
        if i not in self._present:
            return None
        return dict(err=0, current_ma=500.0, velocity=1.25,
                    position=self._present[i], voltage=12.1, temperature=42.0)


@pytest.fixture()
def telem_group():
    g = dx.DynamixelGroup({"neck_pitch": 1, "neck_roll": 2}, device="/dev/fake", baud=1)
    g._bus = _FakeBusTelemetry({1: 0.5, 2: -0.25})
    g.connect()
    g._poll_once()
    return g


def test_read_reports_velocity_position_and_temperature(telem_group):
    """The neck must report pos + vel + temp per joint (torque below), not zeros."""
    fb = telem_group.read()
    i = fb.names.index("neck_pitch")
    assert fb.position[i] == pytest.approx(0.5)
    assert fb.velocity[i] == pytest.approx(1.25)      # present velocity, not zero-filled
    assert fb.temperature[i] == pytest.approx(42.0)


def test_read_reports_torque_from_present_current(telem_group):
    """effort = Kt * I: 0.51 N·m/A * 0.5 A = 0.255 N·m (was hardcoded 0.0)."""
    fb = telem_group.read()
    i = fb.names.index("neck_pitch")
    expect = dx._TORQUE_CONST_NM_PER_A * 0.5      # 500 mA = 0.5 A
    assert fb.effort[i] == pytest.approx(expect)
    assert fb.effort[i] != 0.0


def test_health_reports_torque_nm(telem_group):
    """/body/motor_health carries torqueNm (Kt*I), not a hardcoded 0.0."""
    h = telem_group.health()
    expect = round(dx._TORQUE_CONST_NM_PER_A * 0.5, 4)
    assert h["neck_pitch"]["torqueNm"] == pytest.approx(expect)
    assert h["neck_pitch"]["torqueNm"] != 0.0


def test_torque_sign_follows_current_sign(telem_group):
    """Present Current is signed → effort sign is the drive direction (the
    calibrated subsystem then dir-corrects it)."""
    telem_group._bus._present  # keep fixture
    # id 2 with a negative current draws negative torque
    class _Neg(_FakeBusTelemetry):
        def read_block(self, i):
            b = super().read_block(i)
            if b is not None:
                b["current_ma"] = -500.0
            return b
    telem_group._bus = _Neg({1: 0.5, 2: -0.25})
    telem_group.connect(); telem_group._poll_once()
    fb = telem_group.read()
    assert fb.effort[fb.names.index("neck_pitch")] < 0.0


# ── factory wiring ───────────────────────────────────────────────────────────────
def test_build_joint_driver_parses_dxl_ids():
    drv = build_joint_driver(
        {"backend": "dynamixel", "joints": ["neck_pitch", "neck_roll"],
         "dxl_ids": "neck_pitch:3, neck_roll:4", "device": "", "baud": 1000000},
        links={})
    assert isinstance(drv, dx.DynamixelGroup)
    assert drv.names == ("neck_pitch", "neck_roll")
    assert drv._id == {"neck_pitch": 3, "neck_roll": 4}


def test_build_joint_driver_defaults_ids_from_joints():
    drv = build_joint_driver(
        {"backend": "dynamixel", "joints": ["neck_pitch", "neck_roll"]}, links={})
    assert drv._id == {"neck_pitch": 1, "neck_roll": 2}
