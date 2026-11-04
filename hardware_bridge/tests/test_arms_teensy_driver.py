"""TeensyLink / TeensyJointGroup against the vendored MockTeensy — full
loopback of the robot_host wire protocol: Hello->RobotInfo self-description,
left/right slice merging on ONE link, damping-hold semantics, watchdog
re-send, health, and connect-failure degradation. No hardware, no ROS."""
import time

import pytest

from mabel_hw.drivers.teensy_link import TeensyLink
from mabel_hw.errors import SubsystemUnavailable

from fakes.mock_arm_teensy import LAYOUTS, MockTeensy

LEFT = [f"l_j{i}" for i in range(1, 8)]
RIGHT = [f"r_j{i}" for i in range(1, 8)]


def _wait(pred, timeout=5.0, dt=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture()
def mock():
    m = MockTeensy(LAYOUTS["arms"]).start()
    yield m
    m.stop()


@pytest.fixture()
def link(mock):
    lk = TeensyLink("127.0.0.1", mock.port, stream_hz=200)
    lk.connect()
    yield lk
    lk.close()


def test_handshake_self_description(link, mock):
    info = link.info
    assert info.name == "mockbot"
    assert info.names == LAYOUTS["arms"]        # layout LEARNED, not hard-coded
    assert mock.hello_count >= 1
    assert _wait(link.connected)


def test_group_resolution_and_unknown_names(link):
    left = link.group(LEFT)
    assert left.names == tuple(LEFT)
    with pytest.raises(SubsystemUnavailable) as e:
        link.group(["l_j1", "no_such_joint"])
    assert "no_such_joint" in str(e.value) and "l_j1" in str(e.value)


def test_two_groups_merge_one_command_vector(link, mock):
    left, right = link.group(LEFT), link.group(RIGHT)
    left.set_torque(True)
    left.set_targets({"l_j1": 0.5})              # latch (seam contract)...
    left.tick()                                  # ...tick ships the merged vector
    assert _wait(lambda: mock.last_cmd is not None)
    cmd = mock.last_cmd
    assert cmd["pos"][0] == pytest.approx(0.5, abs=1e-5)
    assert cmd["kp"][0] > 0                      # left slice: active hold
    assert cmd["kp"][7] == 0.0                   # right slice: damping (untouched)
    assert cmd["kd"][7] == pytest.approx(1.0)    # firmware default_kd
    # now the right arm joins on the SAME link
    right.set_torque(True)
    right.set_targets({"r_j1": -0.25})
    right.tick()
    assert _wait(lambda: mock.last_cmd["pos"][7] == pytest.approx(-0.25, abs=1e-5))
    assert mock.last_cmd["pos"][0] == pytest.approx(0.5, abs=1e-5)   # left kept


def test_encoders_track_and_read_slices(link, mock):
    left = link.group(LEFT)
    left.set_torque(True)
    left.set_targets({"l_j2": 1.0})
    for _ in range(40):                          # keep the stream fed
        left.tick()
        time.sleep(0.01)
    fb = left.read()
    assert fb.names == tuple(LEFT)
    assert fb.position[1] == pytest.approx(1.0, abs=0.05)
    assert all(fb.live)


def test_torque_off_reverts_to_damping(link, mock):
    left = link.group(LEFT)
    left.set_torque(True)
    left.set_targets({"l_j1": 0.8})
    left.tick()
    assert _wait(lambda: mock.last_cmd is not None and mock.last_cmd["kp"][0] > 0)
    left.set_torque(False)
    assert _wait(lambda: mock.last_cmd["kp"][0] == 0.0)
    pos_before = mock.pos[0]
    time.sleep(0.1)                              # damping hold: encoder stays put
    assert mock.pos[0] == pytest.approx(pos_before)
    assert not left.is_torque_enabled


def test_tick_feeds_command_watchdog(link, mock):
    left = link.group(LEFT)
    left.set_torque(True)
    left.set_targets({"l_j1": 0.1})
    left.tick()
    assert _wait(lambda: mock.cmd_count > 0)
    n0 = mock.cmd_count
    for _ in range(5):
        left.tick()
    assert _wait(lambda: mock.cmd_count >= n0 + 5)


def test_idle_link_sends_no_commands(link, mock):
    link.group(LEFT)                             # group made but never commanded
    time.sleep(0.2)
    assert mock.cmd_count == 0                   # firmware damping hold suffices


def test_health_and_dead_actuator():
    mock = MockTeensy(LAYOUTS["arms"], dead=(3,)).start()
    try:
        link = TeensyLink("127.0.0.1", mock.port, stream_hz=200)
        link.connect()
        try:
            left = link.group(LEFT)
            left.service()                       # health refresh
            h = left.health()
            assert h["l_j1"]["tempC"] == 25      # t_mos from the Health reply
            fb = left.read()
            assert fb.live[3] is False           # dead index (l_j4) not live
            assert fb.live[0] is True
        finally:
            link.close()
    finally:
        mock.stop()


def test_unknown_joint_write_rejected(link):
    left = link.group(LEFT)
    with pytest.raises(KeyError):
        left.set_targets({"r_j1": 0.0})          # right joint via the LEFT group


def test_no_teensy_degrades(mock):
    lk = TeensyLink("127.0.0.1", 1)              # nothing listens on port 1
    with pytest.raises(SubsystemUnavailable):
        lk.connect()
    assert not lk.connected()


def test_registry_shares_links(mock):
    a = TeensyLink.get("127.0.0.1", mock.port)
    b = TeensyLink.get("127.0.0.1", mock.port)
    c = TeensyLink.get("127.0.0.1", mock.port + 1)
    assert a is b and a is not c


def test_push_callbacks(link):
    got = []
    left = link.group(LEFT)
    unsub = left.on_state(got.append)
    assert _wait(lambda: len(got) > 3)           # driven by packet arrival
    fb = got[-1]
    assert fb.names == tuple(LEFT) and len(fb.position) == 7
    unsub()
    n = len(got)
    time.sleep(0.1)
    assert len(got) == n
