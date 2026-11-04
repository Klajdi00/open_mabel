"""drivers/solo.py tests — the single-DaMiao seam adapter + the multi-bus
composite, hardware-free.

The solo adapter is what puts the torso (HDSC serial dongle) and the neck yaw
(DM-USB2FDCAN) behind the JointGroupDriver seam, so these pin the wiring that
makes them appear in /body/joint_states (the real_run dashboard rows):

  • read() is a cache hit that ages to live=False (publish-gate-on-live)
  • tick() is the only wire touch: poll when idle, MIT re-send when driving
  • torque edges CLEAR the latched target (no stale-setpoint replay/snap)
  • a silent motor refuses to energize without crashing the node
  • the composite merges buses, splits writes, degrades per-bus
  • build_joint_driver wires backend 'serial', 'fdcan', and the neck composite
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from mabel_hw.drivers import build_joint_driver, solo
from mabel_hw.drivers.damiao_serial import DamiaoSerialDriver
from mabel_hw.drivers.seam import MockJointGroupDriver
from mabel_hw.drivers.solo import CompositeJointGroup, DamiaoSoloJoint

_spec = importlib.util.spec_from_file_location(
    "test_damiao_serial", Path(__file__).with_name("test_damiao_serial.py"))
_st = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("test_damiao_serial", _st)
_spec.loader.exec_module(_st)
FakeMotor, FakeSerial, CAN_ID = _st.FakeMotor, _st.FakeSerial, _st.CAN_ID


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    """Deterministic ticks: no wall-clock pacing in the adapter."""
    monkeypatch.setattr(solo, "_LIVE_POLL_S", 0.0)
    monkeypatch.setattr(solo, "_DEAD_POLL_S", 0.0)


def make_solo(motor, joint="torso", slew=5.0):
    drv = DamiaoSerialDriver(port="/dev/fake", can_id=CAN_ID)
    drv._ser = FakeSerial(motor)
    drv._resolved = "/dev/fake"
    # NOTE: no connect() → the 50 Hz stream thread is NOT started; tests drive
    # the glide deterministically via _glide_step(dt).
    return DamiaoSoloJoint(drv, joint, slew=slew), drv


# ── solo adapter ──────────────────────────────────────────────────────────────
def test_read_is_cached_and_tick_polls():
    sj, _ = make_solo(FakeMotor(pos=0.7))
    fb = sj.read()
    assert fb.names == ("torso",) and fb.live == (False,)   # nothing polled yet
    sj.tick()
    fb = sj.read()
    assert fb.live == (True,) and fb.position[0] == pytest.approx(0.7, abs=1e-3)


def test_silent_motor_reads_dead_and_refuses_torque():
    sj, drv = make_solo(FakeMotor(silent=True))
    sj.tick()
    assert sj.read().live == (False,)
    sj.set_torque(True)                       # refused inside — logged, not raised
    assert sj.is_torque_enabled is False
    assert drv.is_torque_enabled is not True


def test_drive_path_glides_with_velocity_ff():
    motor = FakeMotor(pos=0.2)
    sj, drv = make_solo(motor, slew=1.0)
    calls = []
    orig = drv.set_target
    drv.set_target = lambda p, v=0.0, t=0.0: (calls.append((p, v, t)), orig(p, v, t))[1]
    sj.set_torque(True)
    assert sj.is_torque_enabled and motor.enabled
    sj.command({"torso": 0.9}, effort={"torso": 5.0})   # target + gravity t_ff
    sj.tick()                                 # tick never drives — the streamer does
    assert not calls
    sj._glide_step(0.02)                      # seeds from measured 0.2, slews 0.02
    p1, v1, t1 = calls[0]
    assert p1 == pytest.approx(0.22, abs=1e-3)          # ≤ slew·dt from live pose
    #                                                     (seed = 16-bit-quantized read)
    assert v1 == pytest.approx(1.0, abs=1e-9)           # velocity FF = slew rate
    assert t1 == pytest.approx(5.0, abs=1e-9)           # commander's torque FF rides along
    for _ in range(100):                      # …and converges on the goal
        sj._glide_step(0.02)
    assert motor.pos == pytest.approx(0.9, abs=1e-3)
    assert calls[-1][1] == 0.0                # arrival frame: v_ff = 0 (settle, no hum)
    assert calls[-1][2] == pytest.approx(5.0, abs=1e-9)  # gravity hold persists at rest


def test_torque_edge_clears_target_no_replay():
    motor = FakeMotor(pos=0.2)
    sj, _ = make_solo(motor)
    sj.set_torque(True)
    sj.set_targets({"torso": 0.9})
    for _ in range(60):
        sj._glide_step(0.02)
    assert motor.pos == pytest.approx(0.9, abs=1e-3)
    # torque OFF→ON must not replay the stale 0.9
    sj.set_torque(False)
    sj.set_torque(True)
    hold = motor.last_mit[0]                  # safe hold at the measured pose
    assert sj._glide_step(0.02) is False      # no target — streamer idles
    sj.tick()                                 # idle path polls only
    assert motor.last_mit[0] == pytest.approx(hold, abs=1e-3)   # no replay


def test_unknown_joint_raises():
    sj, _ = make_solo(FakeMotor())
    with pytest.raises(KeyError):
        sj.set_targets({"nope": 0.0})


# ── composite ─────────────────────────────────────────────────────────────────
def make_composite():
    yaw = MockJointGroupDriver(["neck_yaw"])
    dxl = MockJointGroupDriver(["neck_pitch", "neck_roll"])
    return CompositeJointGroup([yaw, dxl]), yaw, dxl


def test_composite_merges_names_and_reads():
    comp, yaw, dxl = make_composite()
    assert comp.names == ("neck_yaw", "neck_pitch", "neck_roll")
    comp.connect()
    assert comp.connected()
    fb = comp.read()
    assert fb.names == comp.names and fb.live == (True, True, True)


def test_composite_splits_writes_and_rejects_unknown():
    comp, yaw, dxl = make_composite()
    comp.connect()
    comp.set_torque(True)
    comp.set_targets({"neck_yaw": 0.5, "neck_roll": -0.2})
    for _ in range(40):
        comp.tick()
    assert yaw.read().position[0] == pytest.approx(0.5, abs=1e-3)
    assert dxl.read().position[1] == pytest.approx(-0.2, abs=1e-3)
    assert dxl.read().position[0] == pytest.approx(0.0, abs=1e-6)   # untouched
    with pytest.raises(KeyError):
        comp.set_targets({"bogus": 0.0})


def test_composite_degrades_per_bus():
    comp, yaw, dxl = make_composite()
    comp.connect()
    yaw.close()                               # yaw bus dies
    fb = comp.read()
    assert fb.live == (False, True, True)     # yaw dead, pitch/roll alive
    assert comp.connected()                   # group still up on the other bus
    comp.set_torque(True)                     # reaches the live bus
    assert dxl.is_torque_enabled


def test_composite_connect_raises_only_when_all_buses_fail():
    from mabel_hw.errors import SubsystemUnavailable

    class Dead(MockJointGroupDriver):
        def connect(self):
            raise SubsystemUnavailable("no bus")

    comp = CompositeJointGroup([Dead(["a"]), Dead(["b"])])
    with pytest.raises(SubsystemUnavailable):
        comp.connect()
    ok = MockJointGroupDriver(["c"])
    comp2 = CompositeJointGroup([Dead(["a"]), ok])
    comp2.connect()                           # partial is fine — hot-join later
    assert comp2.connected()


# ── factory wiring ────────────────────────────────────────────────────────────
def test_factory_serial_backend_builds_solo_torso():
    drv = build_joint_driver(dict(backend="serial", joints=["torso"],
                                  port="auto", serial_baud=921600, can_id=1,
                                  kp=450.0, kd=2.5), links={})
    assert isinstance(drv, DamiaoSoloJoint) and drv.names == ("torso",)
    assert drv._drv.kp == 450.0 and drv._drv.can_id == 1


def test_factory_fdcan_backend_builds_solo():
    drv = build_joint_driver(dict(backend="fdcan", joints=["neck_yaw"],
                                  can_id=2), links={})
    assert isinstance(drv, DamiaoSoloJoint) and drv.names == ("neck_yaw",)


def test_factory_neck_composes_two_buses():
    """The LEGACY wiring (NECK_BACKEND=dynamixel: pitch/roll on the USB adapter + yaw on
    the arms Teensy) still composes two buses behind one group. Since 2026-09-30 the
    default backend is `openarm` (all three on the Teensy — see test_openarm_neck.py),
    so the legacy path is selected explicitly here."""
    from mabel_hw import config
    cfg = dict(config.NECK, backend="dynamixel")
    assert "neck_yaw" in cfg["joints"], "NECK_YAW should default ON now"
    assert cfg["yaw_backend"] == "openarm", \
        "yaw rides the arms Teensy bus by default since 2026-07-28"
    drv = build_joint_driver(cfg, links=dict(config.LINKS))
    assert isinstance(drv, CompositeJointGroup)
    assert drv.names == ("neck_yaw", "neck_pitch", "neck_roll")


def test_factory_neck_yaw_fdcan_fallback():
    from mabel_hw import config
    cfg = dict(config.NECK, backend="dynamixel")
    cfg["yaw_backend"] = "fdcan"
    drv = build_joint_driver(cfg, links={})
    assert isinstance(drv, CompositeJointGroup)
    assert drv.names == ("neck_yaw", "neck_pitch", "neck_roll")


def test_factory_neck_default_is_one_openarm_group():
    """Default since 2026-09-30: the whole neck is ONE OpenArmGroup on the arms Teensy."""
    from mabel_hw import config
    from mabel_hw.drivers.openarm import OpenArmGroup
    drv = build_joint_driver(dict(config.NECK), links=dict(config.LINKS))
    assert isinstance(drv, OpenArmGroup)
    assert drv.names == ("neck_yaw", "neck_pitch", "neck_roll")
    drv._link.close()


def test_config_torso_defaults_openarm_enabled():
    """Torso rides the arms Teensy's third CAN bus since 2026-07-28: default
    backend `openarm` (shared link, does NOT energize on connect) with
    TORSO_BACKEND=serial as the HDSC-dongle fallback."""
    from mabel_hw import config
    assert config.TORSO["backend"] == "openarm"
    assert config.TORSO["link"] == "arms_openarm"
    assert config.TORSO["enabled"] is True


def test_openarm_wire_tmax_guard_refuses_mismatched_manifest():
    """2026-07-28 safety guard: the firmware ENCODES t_ff with the SD manifest's
    t_max while the motor DECODES with its flash register — a smaller manifest
    limit multiplies every commanded torque (torso: manifest 100 vs register
    500 = 5x). set_torque(True) must REFUSE until the two agree."""
    import threading
    from types import SimpleNamespace
    from mabel_hw.drivers.openarm import OpenArmGroup

    class _StubLink:
        def __init__(self, t_max):
            self._groups = []
            self.teensy_ip = "stub"
            self.logs = []
            self._spec = SimpleNamespace(name="torso", motor="DM10422",
                                         bus=2, mid=1, p_max=12.5,
                                         v_max=30.0, t_max=t_max)
            self._joint = SimpleNamespace(pos=0.1, vel=0.0, tau=0.0)
        def _log(self, msg): self.logs.append(msg)
        def specs(self): return [self._spec]
        def state(self): return SimpleNamespace(joints=[self._joint])
        def connect(self): pass
        def connected(self): return True
        def write_slice(self, *a, **k): pass
        def damp_slice(self, *a, **k): pass

    # mismatched manifest -> refused, stays limp
    bad = _StubLink(t_max=100.0)
    g = OpenArmGroup(bad, ("torso",), kp=150.0, kd=3.0,
                     wire_t_max={"torso": 500.0})
    bad._groups.append(g)
    g._resolve()
    g.set_torque(True)
    assert g.is_torque_enabled is False
    assert any("REFUSED" in m and "t_max" in m for m in bad.logs)

    # matching manifest -> engages
    good = _StubLink(t_max=500.0)
    g2 = OpenArmGroup(good, ("torso",), kp=150.0, kd=3.0,
                      wire_t_max={"torso": 500.0})
    good._groups.append(g2)
    g2._resolve()
    g2.set_torque(True)
    assert g2.is_torque_enabled is True
