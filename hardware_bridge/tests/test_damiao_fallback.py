"""DamiaoGroup (the CAN/MIT fallback backend) over python-can's in-process
'virtual' bus: real frames on a real Bus object, no socketcan needed — a peer
Bus on the same channel plays the motor side and checks the MIT framing."""
import time

import pytest

can = pytest.importorskip("can")

from mabel_hw.drivers.damiao_can import (DamiaoGroup, f2u, u2f,  # noqa: E402
                                         parse_joint_motors)

KP, KD, P_MAX, V_MAX, T_MAX = 30.0, 1.0, 12.5, 30.0, 10.0


@pytest.fixture()
def channel(request):
    # unique per test: virtual buses pair by channel name process-wide
    return f"vtest_{request.node.name}"


@pytest.fixture()
def peer(channel):
    bus = can.interface.Bus(channel=channel, bustype="virtual")
    yield bus
    bus.shutdown()


@pytest.fixture()
def group(channel, peer):
    joints = parse_joint_motors(f"torso:{channel}:1, neck_yaw:{channel}:2",
                                kp=KP, kd=KD, p_max=P_MAX, v_max=V_MAX, t_max=T_MAX)
    g = DamiaoGroup(joints, bustype="virtual")
    g.connect()
    yield g
    g.shutdown()


def _drain(bus, timeout=0.5):
    out = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = bus.recv(timeout=0.05)
        if msg is None:
            if out:
                break
            continue
        out.append(msg)
    return out


def test_connect_sends_enable_frames(group, peer):
    msgs = _drain(peer)
    enables = [m for m in msgs if bytes(m.data) == b"\xff" * 7 + b"\xfc"]
    assert {m.arbitration_id for m in enables} == {1, 2}


def test_stream_sends_mit_position(group, peer):
    _drain(peer)                                  # drop the enable frames
    group.set_targets({"torso": 1.0})
    names, pos, vel, eff = group.stream()
    assert names == ["torso", "neck_yaw"]
    msgs = _drain(peer)
    cmd = [m for m in msgs if m.arbitration_id == 1 and len(m.data) == 8]
    assert cmd, "no MIT frame for the targeted joint"
    d = cmd[-1].data
    p_u = (d[0] << 8) | d[1]
    assert u2f(p_u, -P_MAX, P_MAX, 16) == pytest.approx(1.0, abs=1e-3)
    kp_u = ((d[3] & 0xF) << 8) | d[4]
    assert u2f(kp_u, 0.0, 500.0, 12) == pytest.approx(KP, abs=0.2)
    # untargeted joint (target=None) must not be commanded
    assert not [m for m in msgs if m.arbitration_id == 2 and len(m.data) == 8
                and bytes(m.data)[:7] != b"\xff" * 7]


def test_target_clamped_to_pmax(group, peer):
    _drain(peer)
    group.set_targets({"torso": 99.0})
    group.stream()
    msgs = _drain(peer)
    d = [m for m in msgs if m.arbitration_id == 1][-1].data
    p_u = (d[0] << 8) | d[1]
    assert u2f(p_u, -P_MAX, P_MAX, 16) == pytest.approx(P_MAX, abs=1e-3)


def test_torque_off_goes_limp(group, peer):
    group.set_targets({"torso": 0.5})
    group.stream()
    _drain(peer)
    group.set_torque(False)
    msgs = _drain(peer)                           # the disable frames
    disables = [m for m in msgs if bytes(m.data) == b"\xff" * 7 + b"\xfd"]
    assert {m.arbitration_id for m in disables} == {1, 2}
    group.stream()                                # limp: no MIT re-send
    assert not [m for m in _drain(peer, timeout=0.3)
                if len(m.data) == 8 and bytes(m.data)[:7] != b"\xff" * 7]


def test_feedback_parses_into_state(group, peer):
    """Play the motor: emit an MIT feedback frame and check pos/vel/tau/temp."""
    pos_u = f2u(0.7, -P_MAX, P_MAX, 16)
    vel_u = f2u(-2.0, -V_MAX, V_MAX, 12)
    tau_u = f2u(1.5, -T_MAX, T_MAX, 12)
    data = bytes([
        0x01,                                     # mid 1 in the low nibble
        (pos_u >> 8) & 0xFF, pos_u & 0xFF,
        (vel_u >> 4) & 0xFF, ((vel_u & 0xF) << 4) | ((tau_u >> 8) & 0xF),
        tau_u & 0xFF,
        42, 40,                                   # T_MOS, T_rotor degC
    ])
    peer.send(can.Message(arbitration_id=0x11, data=data, is_extended_id=False))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        names, pos, vel, eff = group.stream()
        if abs(pos[0] - 0.7) < 1e-2:
            break
        time.sleep(0.02)
    assert pos[0] == pytest.approx(0.7, abs=1e-2)
    assert vel[0] == pytest.approx(-2.0, abs=2e-2)
    assert eff[0] == pytest.approx(1.5, abs=1e-2)
    assert group.health()["torso"]["tempC"] == 42
