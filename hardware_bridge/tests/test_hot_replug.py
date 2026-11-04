"""Hot-replug resilience — the bridge must reconnect BY ITSELF when a device
is unplugged and re-plugged mid-run, and announce what left/rejoined:

  * CAN: a dying socket marks the bus DEAD (RX thread exits cleanly, sends fail
    fast) and DamiaoGroup.service() rebuilds it — re-add + re-enable — the
    moment the channel opens again. Per-joint liveness exposes a single silent
    motor.
  * Teensy: a group created with the board ABSENT resolves itself via
    service() when the board appears; a link that dies and returns resets its
    groups to limp/no-targets (never snap to stale commands).
  * Facade: hw.presence() + the 1 Hz watcher logging LEFT/REJOINED.
"""
import time

import pytest

can = pytest.importorskip("can")

from mabel_hw.drivers.damiao_can import DamiaoGroup, parse_joint_motors
from mabel_hw.drivers.teensy_link import TeensyLink

from fakes.mock_arm_teensy import LAYOUTS, MockTeensy


def _wait(pred, timeout=5.0, dt=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


# ── CAN: dead-bus detection + rebuild ─────────────────────────────────────────

class _Boom(Exception):
    pass


@pytest.fixture()
def group(request):
    channel = f"replug_{request.node.name}"
    joints = parse_joint_motors(f"torso:{channel}:1", kp=30.0, kd=1.0,
                                p_max=12.5, v_max=30.0, t_max=10.0)
    g = DamiaoGroup(joints, bustype="virtual")
    g.connect()
    yield g, channel
    g.shutdown()


def _kill_bus(bus):
    """Simulate the adapter being yanked: recv AND send raise from now on."""
    def boom(*a, **k):
        raise _Boom("device unplugged")
    bus.bus.recv = boom
    bus.bus.send = boom


def test_can_bus_death_is_detected_not_silent(group):
    g, channel = group
    bus = g.buses[channel]
    assert not bus.dead
    _kill_bus(bus)
    g.set_targets({"torso": 0.4})
    g.stream()                                    # send fails -> marks dead
    assert _wait(lambda: bus.dead, timeout=2.0), \
        "a dying CAN socket must mark the bus dead, not die silently"


def test_can_bus_rebuilds_on_service(group):
    g, channel = group
    old_bus = g.buses[channel]
    _kill_bus(old_bus)
    g.stream()                                    # trip the death detection
    assert _wait(lambda: old_bus.dead, timeout=2.0)
    g.service()                                   # virtual channel reopens instantly
    new_bus = g.buses[channel]
    assert new_bus is not old_bus and not new_bus.dead
    # the rebuilt bus really carries frames again: a peer sees the MIT re-send
    peer = can.interface.Bus(channel=channel, bustype="virtual")
    try:
        g.set_targets({"torso": 0.7})
        g.stream()
        msg = peer.recv(timeout=1.0)
        assert msg is not None and msg.arbitration_id == 1
    finally:
        peer.shutdown()


def test_can_rebuild_respects_torque_off(group):
    g, channel = group
    g.set_torque(False)
    _kill_bus(g.buses[channel])
    g.stream()
    assert _wait(lambda: g.buses[channel].dead, timeout=2.0)
    peer = can.interface.Bus(channel=channel, bustype="virtual")
    try:
        g.service()
        # rebuild must NOT re-enable (0xFC) a group the user left in free-drive
        msg = peer.recv(timeout=0.3)
        while msg is not None:
            assert bytes(msg.data) != b"\xff" * 7 + b"\xfc"
            msg = peer.recv(timeout=0.1)
    finally:
        peer.shutdown()


def test_can_rebuild_rate_limited(group):
    g, channel = group
    _kill_bus(g.buses[channel])
    g.stream()
    assert _wait(lambda: g.buses[channel].dead, timeout=2.0)
    g.service()
    first = g.buses[channel]
    _kill_bus(first)                              # dies again immediately
    first.dead = True
    g.service()                                   # within 2 s: must NOT rebuild
    assert g.buses[channel] is first


def test_can_per_joint_liveness(group):
    """One silent motor is visible individually via MotorSpec.live()."""
    g, channel = group
    peer = can.interface.Bus(channel=channel, bustype="virtual")
    try:
        spec = g.joints["torso"]
        assert not spec.live()                    # no feedback yet
        peer.send(can.Message(arbitration_id=0x11,
                              data=bytes([0x01, 0, 0, 0, 0, 0, 30, 28]),
                              is_extended_id=False))
        assert _wait(lambda: spec.live(), timeout=2.0)
        assert not spec.live(timeout=0.0)         # and it decays when silent
    finally:
        peer.shutdown()


# ── Teensy: absent at boot -> hot-join; die -> limp-on-rejoin ─────────────────

def test_teensy_hot_join_after_boot():
    """Group exists first, board appears later: service() attaches it. The
    mock's socket is bound from the start but SILENT until .start() — the
    race-free stand-in for a Teensy that boots after the bridge."""
    mock = MockTeensy(LAYOUTS["neck"])            # bound, not answering yet
    port = mock.port

    link = TeensyLink("127.0.0.1", port, stream_hz=200)
    group = link.group(["motor_L1", "motor_L2", "motor_head"], strict=False)
    assert not group.resolved
    group.service()                               # no RobotInfo yet: quiet fail
    assert not group.resolved

    mock.start()                                  # board boots NOW
    try:
        link._last_connect_try = 0.0              # skip the retry rate limit
        assert _wait(lambda: (group.service() or group.resolved), timeout=8.0)
        assert _wait(group.connected)             # first State packet lands
        group.set_torque(True)
        group.set_targets({"motor_head": 0.4})
        group.tick()
        assert _wait(lambda: mock.last_cmd is not None
                     and mock.last_cmd["pos"][2] == pytest.approx(0.4, abs=1e-5))
    finally:
        mock.stop()
        link.close()


def test_teensy_rejoin_resets_to_limp():
    """A Teensy that reboots mid-run must come back LIMP (targets cleared,
    torque off) — never snapping to the pre-reboot command."""
    mock = MockTeensy(LAYOUTS["neck"]).start()
    port = mock.port
    link = TeensyLink("127.0.0.1", port, stream_hz=200)
    try:
        link.connect()
        group = link.group(["motor_L1", "motor_L2", "motor_head"])
        group.set_torque(True)
        group.set_targets({"motor_head": 0.8})
        group.tick()
        assert _wait(lambda: mock.cmd_count > 0)

        mock.stop()                               # board dies
        time.sleep(0.05)
        assert _wait(lambda: not link.connected(), timeout=2.0)

        mock2 = MockTeensy(LAYOUTS["neck"], port=port).start()   # replug
        try:
            link._last_connect_try = 0.0
            assert _wait(lambda: (group.service() or link.connected()),
                         timeout=5.0)
            # safety semantics: rejoined LIMP — no torque, no stale target
            assert not group.is_torque_enabled
            n0 = mock2.cmd_count
            group.tick()
            time.sleep(0.1)
            assert mock2.cmd_count == n0          # nothing streams until re-commanded
            # and re-commanding works normally
            group.set_torque(True)
            group.set_targets({"motor_head": -0.2})
            group.tick()
            assert _wait(lambda: mock2.last_cmd is not None
                         and mock2.last_cmd["pos"][2] == pytest.approx(-0.2, abs=1e-5))
        finally:
            mock2.stop()
    finally:
        link.close()


# ── Facade: presence map + LEFT/REJOINED announcements ───────────────────────

def test_presence_map_and_transitions():
    from mabel_hw import MabelHardware
    logs = []
    with MabelHardware(mock=True, logger=logs.append) as hw:
        pres = hw.presence()
        for name in ("base", "left_arm", "right_hand", "lift", "lidar"):
            assert pres[name] is True, name
        # unplug the lift (mock): presence flips and the watcher announces it
        hw._lift._driver.close()
        assert hw.presence()["lift"] is False
        assert _wait(lambda: any("lift LEFT the system" in m for m in logs),
                     timeout=3.0), logs[-5:]
        hw._lift._driver.connect()                # replug
        assert _wait(lambda: any("lift REJOINED" in m for m in logs),
                     timeout=3.0)
