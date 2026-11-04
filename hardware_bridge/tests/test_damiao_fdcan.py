"""damiao_fdcan driver tests — the DM-USB2FDCAN dongle backend, hardware-free.

A FakeLib stands in for the vendored ``libdm_device`` at the ``dmcan`` binding
seam, re-using ``test_damiao_serial.FakeMotor`` as the wire-accurate motor model
(the CAN-side DaMiao protocol is identical across dongles). Replies travel
through the REAL recv-callback + ctypes struct path, so these tests pin:

  • the param/MIT frame builders (same packing law as damiao_serial/damiao_can)
  • scale registers (RIDs 21/22/23) read from the MOTOR at connect
  • the safety contract: connect never energizes; enable is a SAFE HOLD at the
    measured pose; commanding with torque off raises; a dead motor reads
    live=False; close() disables the motor
  • register write (0x55) + save (0xAA) — the commissioning path (id changes)
"""
import ctypes as ct
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

from mabel_hw.drivers import damiao_fdcan as fd
from mabel_hw.drivers import dmcan
from mabel_hw.errors import SubsystemUnavailable

_spec = importlib.util.spec_from_file_location(
    "test_damiao_serial", Path(__file__).with_name("test_damiao_serial.py"))
_serial_tests = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("test_damiao_serial", _serial_tests)
_spec.loader.exec_module(_serial_tests)
FakeMotor, CAN_ID = _serial_tests.FakeMotor, _serial_tests.CAN_ID


class FakeSavingMotor(FakeMotor):
    """FakeMotor + the 0x55 register-write / 0xAA save commissioning surface."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.saved = False

    def __call__(self, arb, data):
        if not self.silent and arb == 0x7FF and (data[0] | (data[1] << 8)) == CAN_ID:
            if data[2] == 0x55:
                rid = data[3]
                self.registers[rid] = (
                    int.from_bytes(data[4:8], "little") if rid in self.INT_RIDS
                    else struct.unpack("<f", data[4:8])[0])
                return [(0x7FF, bytes(data[:4]) + bytes(data[4:8]))]
            if data[2] == 0xAA:
                self.saved = True
                return []
        return super().__call__(arb, data)


class FakeLib:
    """The dmcan function surface the driver touches, backed by a FakeMotor.
    send_can feeds replies through the driver's REAL recv callback with real
    ctypes usb_rx_frame structs — the same decode path as the hardware."""

    def __init__(self, motor):
        self.motor, self.cb = motor, None
        self.tx = []                     # decoded (arb, data8) in order
        self.closed = self.channel_enabled = False
        self.fail_sends = False

    def dmcan_context_create(self, pctx):
        pass

    def dmcan_context_destroy(self, ctx):
        pass

    def dmcan_find_devices_with_type(self, ctx, typ):
        return 1

    def dmcan_device_get(self, ctx, pdev, index):
        return True

    def dmcan_device_open(self, dev):
        return True

    def dmcan_device_close(self, dev):
        self.closed = True

    def dmcan_device_enable_channel(self, dev, ch):
        self.channel_enabled = True
        return True

    def dmcan_device_get_channel_baudrate(self, dev, ch, info):
        return False                     # driver must cope without baud info

    def dmcan_device_hook_recv_callback(self, dev, cb):
        self.cb = cb

    def dmcan_device_send_can(self, dev, ch, arb, canfd, ext, rtr, brs, dlen, buf):
        if self.fail_sends:
            return False
        data = bytes(buf[:dlen])
        self.tx.append((arb, data))
        for rarb, rdata in self.motor(arb, data):
            f = dmcan.usb_rx_frame()
            f.head.can_id, f.head.channel, f.head.dlc = rarb, ch, 8
            for i, b in enumerate(rdata):
                f.payload[i] = b
            self.cb(None, ct.pointer(f))
        return True


def make_driver(monkeypatch, motor, **kw):
    lib = FakeLib(motor)
    monkeypatch.setattr(dmcan, "load", lambda: lib)
    drv = fd.DamiaoFdcanDriver(can_id=CAN_ID, **kw)
    return drv, lib


# ── frame builders (one packing law across the DaMiao drivers) ────────────────
def test_param_frame_layout():
    assert fd.param_frame(CAN_ID, 0xCC) == bytes([1, 0, 0xCC, 0, 0, 0, 0, 0])
    assert fd.param_frame(CAN_ID, 0x33, 21) == bytes([1, 0, 0x33, 21, 0, 0, 0, 0])
    w = fd.param_frame(CAN_ID, 0x55, 8, (2).to_bytes(4, "little"))
    assert w == bytes([1, 0, 0x55, 8, 2, 0, 0, 0])


def test_mit_frame_matches_serial_packing():
    from mabel_hw.drivers.damiao_serial import DamiaoSerialDriver
    drv = DamiaoSerialDriver(port="/dev/fake", can_id=CAN_ID,
                             kp=20.0, kd=1.0, p_max=12.5, v_max=30.0, t_max=10.0)
    sent = []
    drv._ser = type("S", (), {"reset_input_buffer": lambda self: None})()
    drv._tx = lambda arb, d: sent.append((arb, bytes(d)))
    drv._rx_until = lambda *a, **k: None
    drv._mit(0.7, 0.0, 20.0, 1.0, 0.0)
    assert sent[0] == (CAN_ID, fd.mit_frame(0.7, 0.0, 20.0, 1.0, 0.0,
                                            12.5, 30.0, 10.0))


# ── connect ──────────────────────────────────────────────────────────────────
def test_connect_reads_scales_and_never_energizes(monkeypatch):
    motor = FakeMotor(pos=0.4, p_max=12.5, v_max=30.0, t_max=10.0)
    drv, lib = make_driver(monkeypatch, motor, p_max=99.0, v_max=99.0, t_max=99.0)
    drv.connect()
    assert drv.connected() and lib.channel_enabled
    assert drv.scales_from_motor and drv.p_max == 12.5 and drv.t_max == 10.0
    assert drv.ctrl_mode == 1
    assert motor.enabled is False and motor.last_mit is None      # nothing energized


# ── telemetry ────────────────────────────────────────────────────────────────
def test_read_live_and_dead(monkeypatch):
    motor = FakeMotor(pos=1.2)
    drv, _lib = make_driver(monkeypatch, motor)
    drv.connect()
    st = drv.read()
    assert st.live and st.position == pytest.approx(1.2, abs=1e-3)
    motor.silent = True
    st = drv.read()
    assert not st.live and st.position == pytest.approx(1.2, abs=1e-3)  # last-known


# ── safety contract ──────────────────────────────────────────────────────────
def test_enable_is_a_safe_hold_at_the_measured_pose(monkeypatch):
    motor = FakeMotor(pos=0.9)
    drv, _lib = make_driver(monkeypatch, motor)
    drv.connect()
    drv.set_torque(True)
    assert motor.enabled is True
    p_des, kp, _kd = motor.last_mit
    assert p_des == pytest.approx(0.9, abs=1e-3)                  # hold HERE, no snap
    assert kp == pytest.approx(drv.kp, rel=0.05)


def test_command_with_torque_off_raises(monkeypatch):
    motor = FakeMotor(pos=0.0)
    drv, _lib = make_driver(monkeypatch, motor)
    drv.connect()
    with pytest.raises(SubsystemUnavailable):
        drv.set_position(0.3)
    assert motor.last_mit is None


def test_enable_on_a_silent_motor_refuses(monkeypatch):
    motor = FakeMotor(silent=True)
    drv, _lib = make_driver(monkeypatch, motor)
    drv.connect()
    with pytest.raises(SubsystemUnavailable):
        drv.set_torque(True)
    assert motor.enabled is False


def test_set_position_drives_and_close_disables(monkeypatch):
    motor = FakeMotor(pos=0.0)
    drv, lib = make_driver(monkeypatch, motor)
    drv.connect()
    drv.set_torque(True)
    st = drv.set_position(0.5)
    assert motor.pos == pytest.approx(0.5, abs=1e-3)
    assert st.live and st.position == pytest.approx(0.5, abs=1e-3)
    drv.close()
    assert motor.enabled is False and lib.closed and not drv.connected()


# ── commissioning: register write + save ─────────────────────────────────────
def test_write_param_and_save(monkeypatch):
    motor = FakeSavingMotor(pos=0.0)
    drv, _lib = make_driver(monkeypatch, motor)
    drv.connect()
    assert drv.write_param(8, 2) is True                          # ESC id → 2
    assert motor.registers[8] == 2
    drv.save_params()
    assert motor.saved is True and motor.enabled is False         # saved de-energized
