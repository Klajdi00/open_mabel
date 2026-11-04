"""SwerveBase against the wire-exact fake Teensy (tools/fake_teensy.py) —
full loopback of the production UDP/protobuf path: Hello handshake, command
latch + tick re-send, encoder feedback, estop zeroing. No hardware, no ROS.
"""
import importlib.util
import os
import time

import pytest

from conftest import BRIDGE_DIR


def _load_fake_teensy():
    path = os.path.join(BRIDGE_DIR, "tools", "fake_teensy.py")
    spec = importlib.util.spec_from_file_location("fake_teensy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wait(pred, timeout=5.0, dt=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture()
def fake():
    ft = _load_fake_teensy().FakeTeensy(port=0).start()   # port=0 -> ephemeral
    yield ft
    ft.running = False
    try:
        ft.sock.close()
    except Exception:
        pass


@pytest.fixture()
def base(fake):
    from mabel_hw.drivers.swerve import SwerveBase
    port = fake.sock.getsockname()[1]
    sb = SwerveBase(teensy_ip="127.0.0.1", port=port, stream_hz=100,
                    apply_host_calib=False)   # raw IK on the wire: assertable numbers
    yield sb
    sb.close()


def test_connect_and_telemetry(base, fake):
    assert _wait(lambda: fake.hello_count > 0), "no Hello reached the fake Teensy"
    assert _wait(lambda: base.joint_state() is not None), "no State ever streamed back"
    names, pos, vel = base.joint_state()
    assert len(names) == 6 and len(pos) == 6 and len(vel) == 6
    assert base.connected()


def test_command_reaches_teensy_and_encoders_follow(base, fake):
    assert _wait(lambda: base.joint_state() is not None)
    base.set_twist(0.3, 0.0, 0.0)
    base.tick()
    # command latched by the fake: pure +x -> equal drive speeds on all modules
    assert _wait(lambda: fake.cmd_count > 0 and max(abs(d) for d in fake.drive) > 0.01)
    assert max(fake.drive) - min(fake.drive) < 1e-6
    assert abs(fake.drive[0] - 0.3) < 0.05
    # the fake's encoders follow the command -> drive velocity comes back
    def encoders_follow():
        js = base.joint_state()
        return js is not None and max(abs(v) for v in js[2][3:]) > 0.1
    assert _wait(encoders_follow), "encoder feedback never reflected the command"


def test_estop_zeroes_drive(base, fake):
    assert _wait(lambda: base.joint_state() is not None)
    base.set_twist(0.5, 0.0, 0.0)
    base.tick()
    assert _wait(lambda: max(abs(d) for d in fake.drive) > 0.01)
    base.set_estop(True)
    base.tick()
    assert _wait(lambda: max(abs(d) for d in fake.drive) < 1e-9), \
        "estop did not zero the drive command"
    # commands while estopped stay zero
    base.set_twist(1.0, 0.0, 0.0)
    base.tick()
    time.sleep(0.05)
    assert max(abs(d) for d in fake.drive) < 1e-9


def test_stale_command_zeroes_drive(fake):
    from mabel_hw.drivers.swerve import SwerveBase
    port = fake.sock.getsockname()[1]
    sb = SwerveBase(teensy_ip="127.0.0.1", port=port, stream_hz=100,
                    apply_host_calib=False, cmd_timeout=0.1)
    try:
        assert _wait(lambda: sb.joint_state() is not None)
        sb.set_twist(0.4, 0.0, 0.0)
        sb.tick()
        assert _wait(lambda: max(abs(d) for d in fake.drive) > 0.01)
        time.sleep(0.25)                     # command goes stale (> cmd_timeout)
        sb.tick()
        assert _wait(lambda: max(abs(d) for d in fake.drive) < 1e-9), \
            "stale command was not zeroed by the watchdog"
    finally:
        sb.close()


def test_flaky_link_drops_client_instead_of_hammering():
    """A base whose setCommand keeps timing out must NOT be hammered 50x/s: after
    _CMD_FAIL_DROP consecutive failures the client is dropped (base idle) so the
    native keepalive thread stops getting chances to abort the whole process. This
    is the regression that crash-looped the bridge and took the ARMS down with the
    (unreachable) base — see swerve.tick()."""
    from mabel_hw.drivers.swerve import SwerveBase

    class FlakyClient:
        def __init__(self): self.calls = 0; self.closed = False
        def setCommand(self, *_a, **_k):
            self.calls += 1
            raise TimeoutError("timed out")
        def getState(self): return None
        def close(self): self.closed = True

    logs = []
    sb = SwerveBase(teensy_ip="127.0.0.1", port=1, apply_host_calib=False,
                    logger=logs.append)
    sb.sw = FlakyClient()                      # pretend a link that connected then went flaky
    sb._cmd_fail = 0
    sb.set_twist(0.1, 0.0, 0.0)                # a non-zero, non-stale command
    for _ in range(SwerveBase._CMD_FAIL_DROP + 5):
        sb.tick()

    assert sb.sw is None, "flaky client was not dropped"
    # it stopped calling setCommand at the drop threshold, not 15 times
    # (the FlakyClient is gone, so no further calls are possible)
    assert any("dropping the client" in m for m in logs), logs
    # and it did NOT emit a log line per failure (rate-limited)
    assert sum("setCommand failed" in m for m in logs) <= 2, logs
