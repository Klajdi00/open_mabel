"""LiftDriver (the bridge-side wrapper) with the firmware host class stubbed —
unit-tests OUR layer: metres API pass-through, snapshot mapping, graceful
degradation when the Pico is absent, drop-and-retry on telemetry failure.
(The Pico protocol itself is firmware/lift's own test surface.)"""
import pytest

from mabel_hw.drivers import lift as lift_mod
from mabel_hw.errors import SubsystemUnavailable


class FakeLift:
    """Stands in for firmware/lift/host lift.Lift (metres API)."""
    fail_telemetry = False

    def __init__(self, port, baud, max_pos=0.5):
        self.port, self.baud, self.max_pos = port, baud, max_pos
        self.min_pos = 0.0
        self.height = 0.1
        self.vel = 0.0
        self.homed = True
        self.calls = []

    def get_lift_height(self):
        if self.fail_telemetry:
            raise IOError("pico went away")
        return self.height

    def get_velocity(self):
        return self.vel

    def lift_to_height(self, m, v_max_mps=None, wait=True):
        self.calls.append(("to_height", m, wait))
        self.height = m

    def set_velocity_control(self, mps):
        self.calls.append(("vel", mps))

    def lift_home(self, wait=True, timeout=60.0):
        self.calls.append(("home", wait))

    def calibrate(self, wait=True, timeout=60.0):
        self.calls.append(("calibrate", wait))
        self.max_pos = 0.9525          # measured floor+ceiling envelope

    def lift_stop(self):
        self.calls.append(("stop",))

    def set_torque(self, on=True):
        self.calls.append(("torque", on))

    def close(self):
        self.calls.append(("close",))


@pytest.fixture()
def driver(monkeypatch):
    monkeypatch.setattr(lift_mod, "_import_lift_class", lambda: FakeLift)
    d = lift_mod.LiftDriver(port="/dev/fake", baud=115200, max_pos=0.6)
    yield d
    d.close()


def test_connect_read_write(driver):
    driver.connect()
    assert driver.connected()
    st = driver.read()
    assert st.connected and st.homed and st.height == pytest.approx(0.1)
    driver.set_height(0.42)
    st = driver.read()
    assert st.height == pytest.approx(0.42)
    fake = driver._lift
    assert ("to_height", 0.42, False) in fake.calls   # non-blocking by contract
    driver.home()
    assert ("home", False) in fake.calls
    # calibration + the measured min/max envelope surface in the snapshot
    assert driver.read().max_pos == pytest.approx(0.6)   # config default until calibrated
    driver.calibrate()
    assert ("calibrate", False) in fake.calls
    st = driver.read()
    assert st.calibrated and st.min_pos == pytest.approx(0.0)
    assert st.max_pos == pytest.approx(0.9525)
    driver.stop()
    assert ("stop",) in fake.calls


def test_absent_device_degrades(monkeypatch):
    class Unreachable(FakeLift):
        def __init__(self, *a, **k):
            raise IOError("no such port")
    monkeypatch.setattr(lift_mod, "_import_lift_class", lambda: Unreachable)
    d = lift_mod.LiftDriver(port="/dev/nope")
    with pytest.raises(SubsystemUnavailable):
        d.connect()
    st = d.read()                       # reading a dead lift is safe, not fatal
    assert not st.connected and st.height is None
    with pytest.raises(SubsystemUnavailable):
        d.set_height(0.3)               # writing is not
    d.stop()                            # the safety path never raises


def test_telemetry_failure_drops_link_for_retry(driver):
    driver.connect()
    driver._lift.fail_telemetry = True
    st = driver.read()
    assert not st.connected             # snapshot degrades...
    assert not driver.connected()       # ...and the link is dropped
    FakeLift.fail_telemetry = False
    driver.service()                    # slow-path watchdog reconnects
    assert driver.connected()
    assert driver.read().connected


def test_missing_firmware_checkout(monkeypatch):
    def no_checkout():
        raise SubsystemUnavailable("firmware/lift/host not found")
    monkeypatch.setattr(lift_mod, "_import_lift_class", no_checkout)
    d = lift_mod.LiftDriver()
    with pytest.raises(SubsystemUnavailable):
        d.connect()


class _MuteCtl:
    """Stands in for LiftController: telemetry OFF until 'json on' (enable_telemetry)."""
    def __init__(self):
        self.streaming = False
        self.enables = 0
    def enable_telemetry(self, attempts=3, wait=0.2):
        self.enables += 1
        self.streaming = True            # the re-kick turns the stream on
        return True


class MuteLift(FakeLift):
    """A Pico that opens fine but boots with telemetry OFF (json_stream=False):
    get_lift_height() is None until enable_telemetry() is (re)sent."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._c = _MuteCtl()
    def get_lift_height(self):
        return self.height if self._c.streaming else None


def test_mute_telemetry_is_rekicked_then_recovers(monkeypatch):
    """The connect-time 'json on' was lost, so the Pico is open but mute. read()
    yields no height (node stays silent); service() must re-send 'json on' and the
    stream must come back — no full reconnect needed."""
    monkeypatch.setattr(lift_mod, "_import_lift_class", lambda: MuteLift)
    d = lift_mod.LiftDriver(port="/dev/fake")
    d.connect()
    assert d.connected()                 # serial is open
    assert d.read().height is None       # ...but nothing streams yet
    d.service()                          # 1 Hz watchdog re-sends 'json on'
    assert d._lift._c.enables == 1
    st = d.read()
    assert st.height is not None         # telemetry recovered without a reconnect
    d.close()
