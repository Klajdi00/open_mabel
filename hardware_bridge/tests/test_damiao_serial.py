"""damiao_serial driver tests — the DaMiao USB-CAN dongle protocol, hardware-free.

A FakeSerial + FakeMotor pair emulate the dongle wire exactly (30-byte TX wrap,
16-byte RX wrap, 0x7FF param bus, MST-id feedback), so these tests pin:

  • the serial framing (byte-identical to the proven dm_configurator/DM_CAN template)
  • MIT pack→unpack consistency through the motor's fixed-point scales
  • the scale registers (RIDs 21/22/23) being read from the MOTOR at connect
  • the safety contract: connect never energizes; enable is a SAFE HOLD at the
    measured pose; commanding with torque off raises; a dead motor reads live=False
  • hot-replug: a failed write drops the socket instead of wedging
"""
import math
import struct
import sys
import types

import pytest

from mabel_hw.drivers import damiao_serial as ds
from mabel_hw.drivers.damiao_can import f2u, u2f
from mabel_hw.errors import SubsystemUnavailable

CAN_ID, MST_ID = 1, 0x11


def _rx_wrap(arb, data8):
    """dongle→host 16-byte frame: AA 11 08 <arb LE ×4> <8 data> 55."""
    return bytes([0xAA, 0x11, 0x08, arb & 0xFF, (arb >> 8) & 0xFF,
                  (arb >> 16) & 0xFF, (arb >> 24) & 0xFF]) + bytes(data8) + b"\x55"


class FakeMotor:
    """Answers the wire like a real DM motor: 0xCC/0x33 on the 0x7FF param bus,
    enable/disable/MIT on its CAN id, feedback on its MST id."""

    INT_RIDS = frozenset(range(7, 11)) | frozenset(range(13, 17)) | {35, 36}

    def __init__(self, pos=0.0, p_max=12.5, v_max=10.4, t_max=200.0,
                 registers=None, silent=False):
        self.pos, self.enabled, self.silent = pos, False, silent
        self.p_max, self.v_max, self.t_max = p_max, v_max, t_max
        self.err = 0
        self.last_mit = None
        self.registers = {21: p_max, 22: v_max, 23: t_max, 10: 1}
        if registers is not None:
            self.registers = registers

    def feedback(self):
        # byte-0 high nibble is the motor STATE: 0 disabled, 1 enabled, 0x8+ faults —
        # exactly what the real DM-J10422 was measured to report.
        state = self.err if self.err else (1 if self.enabled else 0)
        d = [((state & 0xF) << 4) | (CAN_ID & 0xF)]
        p = f2u(self.pos, -self.p_max, self.p_max, 16)
        v = f2u(0.0, -self.v_max, self.v_max, 12)
        t = f2u(0.0, -self.t_max, self.t_max, 12)
        d += [(p >> 8) & 0xFF, p & 0xFF,
              (v >> 4) & 0xFF, ((v & 0xF) << 4) | ((t >> 8) & 0xF), t & 0xFF,
              27, 22]
        return (MST_ID, bytes(d))

    def __call__(self, arb, data):
        if self.silent:
            return []
        if arb == 0x7FF:
            if (data[0] | (data[1] << 8)) != CAN_ID:
                return []
            if data[2] == 0xCC:
                return [self.feedback()]
            if data[2] == 0x33:
                rid = data[3]
                val = self.registers.get(rid)
                if val is None:
                    return []
                enc = (int(val).to_bytes(4, "little") if rid in self.INT_RIDS
                       else struct.pack("<f", float(val)))
                return [(0x7FF, bytes([data[0], data[1], 0x33, rid]) + enc)]
            return []
        if arb == CAN_ID:
            if bytes(data[:7]) == b"\xff" * 7:
                if data[7] == 0xFC:
                    self.enabled = True
                elif data[7] == 0xFD:
                    self.enabled = False
                return [self.feedback()]
            # MIT frame — unpack with the motor's own scales
            p_des = u2f((data[0] << 8) | data[1], -self.p_max, self.p_max, 16)
            kp = u2f(((data[3] & 0xF) << 8) | data[4], 0.0, 500.0, 12)
            kd = u2f((data[5] << 4) | (data[6] >> 4), 0.0, 5.0, 12)
            self.last_mit = (p_des, kp, kd)
            self.last_tff = u2f(((data[6] & 0xF) << 8) | data[7],
                                -self.t_max, self.t_max, 12)
            if self.enabled:
                self.pos = p_des
            return [self.feedback()]
        return []


class FakeSerial:
    def __init__(self, responder):
        self.responder = responder
        self.rx = bytearray()
        self.tx = []                       # decoded (arb, data8) in order
        self.fail_writes = False
        self.closed = False

    def write(self, buf):
        if self.fail_writes:
            raise OSError(5, "Input/output error")
        assert len(buf) == 30 and buf[0] == 0x55 and buf[1] == 0xAA and buf[2] == 0x1E
        arb = buf[13] | (buf[14] << 8) | (buf[15] << 16) | (buf[16] << 24)
        data = bytes(buf[21:29])
        self.tx.append((arb, data))
        for rarb, rdata in self.responder(arb, data):
            self.rx += _rx_wrap(rarb, rdata)

    def read(self, n):
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def reset_input_buffer(self):
        self.rx.clear()

    def close(self):
        self.closed = True


def make_driver(motor, **kw):
    drv = ds.DamiaoSerialDriver(port="/dev/fake", can_id=CAN_ID, **kw)
    drv._ser = FakeSerial(motor)
    drv._resolved = "/dev/fake"
    return drv


# ── framing ───────────────────────────────────────────────────────────────────
def test_frame_roundtrip():
    data = bytes(range(8))
    frames = list(ds.parse_frames(_rx_wrap(0x11, data) + _rx_wrap(0x7FF, data)))
    assert frames == [(0x11, data), (0x7FF, data)]


def test_parse_skips_garbage_and_partials():
    good = _rx_wrap(MST_ID, bytes(8))
    noisy = b"\x00\xaa\x03" + good + good[:7]        # junk + one frame + a partial
    assert list(ds.parse_frames(noisy)) == [(MST_ID, bytes(8))]


def test_tx_template_matches_dm_configurator():
    # The proven 30-byte template from dm_configurator.py / DM_CAN.py — byte-for-byte.
    expected = bytes([0x55, 0xAA, 0x1E, 0x03, 0x01, 0, 0, 0, 0x0A, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert ds.build_frame(0, bytes(8)) == expected
    f = ds.build_frame(0x7FF, bytes([1, 0, 0xCC, 0, 0, 0, 0, 0]))
    assert (f[13], f[14]) == (0xFF, 0x07) and f[21] == 1 and f[23] == 0xCC


# ── connect-time register reads ──────────────────────────────────────────────
def test_scales_read_from_motor():
    motor = FakeMotor(p_max=12.5, v_max=10.4, t_max=200.0)
    drv = make_driver(motor, p_max=99.0, v_max=99.0, t_max=99.0)   # bogus config seeds
    drv._read_scales()
    assert drv.scales_from_motor
    assert (drv.p_max, drv.v_max, drv.t_max) == pytest.approx((12.5, 10.4, 200.0))
    assert drv.ctrl_mode == 1


def test_scales_keep_config_when_registers_silent():
    motor = FakeMotor(registers={})                                 # answers nothing on 0x33
    drv = make_driver(motor, p_max=12.5, v_max=30.0, t_max=10.0)
    drv._read_scales()
    assert not drv.scales_from_motor
    assert (drv.p_max, drv.v_max, drv.t_max) == (12.5, 30.0, 10.0)


def test_read_param_int_vs_float():
    motor = FakeMotor(registers={8: 1, 7: 0x11, 21: 12.5})
    drv = make_driver(motor)
    assert drv.read_param(8) == 1.0                                 # can_id (uint32)
    assert drv.read_param(7) == float(0x11)                         # mst_id (uint32)
    assert drv.read_param(21) == pytest.approx(12.5)                # pmax (float32)


# ── telemetry ────────────────────────────────────────────────────────────────
def test_read_live_position():
    motor = FakeMotor(pos=1.234)
    drv = make_driver(motor)
    st = drv.read()
    assert st.live and st.position == pytest.approx(1.234, abs=1e-3)
    assert (st.t_mos, st.t_rotor, st.err) == (27, 22, 0)
    assert not st.enabled and st.fault == 0


def test_read_dead_motor_keeps_last_pose():
    motor = FakeMotor(pos=0.8)
    drv = make_driver(motor)
    assert drv.read().live
    motor.silent = True                                             # 24 V dropped
    st = drv.read()
    assert not st.live and st.position == pytest.approx(0.8, abs=1e-3)


def test_state_nibble_split_enabled_vs_fault():
    # nibble 1 = ENABLED, a healthy state — must never read as a fault (the bug that
    # painted "fault 0x1" on an energized torso)
    motor = FakeMotor()
    drv = make_driver(motor)
    drv.set_torque(True)
    st = drv.read()
    assert st.err == 1 and st.enabled and st.fault == 0
    motor.err = 0x8                                                 # a REAL fault (over-voltage)
    st = drv.read()
    assert st.err == 0x8 and st.fault == 0x8 and not st.enabled


# ── the safety contract ──────────────────────────────────────────────────────
def test_enable_is_safe_hold_at_measured_pose():
    motor = FakeMotor(pos=2.5)
    drv = make_driver(motor)
    drv.set_torque(True)
    assert drv.is_torque_enabled is True and motor.enabled
    # the hold target is the measured pose, so the motor did not move
    p_des, kp, kd = motor.last_mit
    assert p_des == pytest.approx(2.5, abs=1e-3)
    assert motor.pos == pytest.approx(2.5, abs=1e-3)
    assert kp == pytest.approx(drv.kp, rel=0.01) and kd == pytest.approx(drv.kd, rel=0.02)
    # wire order: poll (0x7FF/0xCC) BEFORE enable (FF..FC) BEFORE the hold MIT frame
    kinds = [(arb, d[7] if bytes(d[:7]) == b"\xff" * 7 else
              ("cc" if arb == 0x7FF and d[2] == 0xCC else "mit"))
             for arb, d in drv._ser.tx]
    assert kinds == [(0x7FF, "cc"), (CAN_ID, 0xFC), (CAN_ID, "mit")]


def test_enable_refuses_a_dead_motor():
    drv = make_driver(FakeMotor(silent=True))
    with pytest.raises(SubsystemUnavailable):
        drv.set_torque(True)
    assert drv.is_torque_enabled is not True
    # nothing beyond the failed poll went out — no enable, no MIT
    assert all(arb == 0x7FF for arb, _ in drv._ser.tx)


def test_disable_sends_fd():
    motor = FakeMotor()
    drv = make_driver(motor)
    drv.set_torque(True)
    drv.set_torque(False)
    assert drv.is_torque_enabled is False and not motor.enabled
    assert drv._ser.tx[-1] == (CAN_ID, b"\xff" * 7 + b"\xfd")


def test_set_position_requires_torque():
    drv = make_driver(FakeMotor())
    with pytest.raises(SubsystemUnavailable):
        drv.set_position(1.0)


def test_set_position_roundtrip_through_motor_scales():
    motor = FakeMotor(pos=0.0, p_max=12.5)
    drv = make_driver(motor, p_max=12.5)
    drv.set_torque(True)
    st = drv.set_position(2.0)
    assert motor.pos == pytest.approx(2.0, abs=25.0 / 65535 + 1e-9)  # 16-bit quantum
    assert st.live and st.position == pytest.approx(2.0, abs=1e-3)


def test_torque_feedforward_rides_the_mit_frame():
    # the full MIT law τ = kp·e + kd·ė + t_ff — t_ff must survive pack→unpack through
    # the motor's REAL torque scale (t_max 200 here; quantum = 2·t_max/4095)
    motor = FakeMotor(pos=1.0, t_max=200.0)
    drv = make_driver(motor, t_max=200.0)
    drv.set_torque(True, hold_t_ff=17.5)               # enable-hold carries FF (no sag)
    assert motor.last_tff == pytest.approx(17.5, abs=400.0 / 4095 + 1e-9)
    drv.set_position(1.2, t_ff=-19.0)
    assert motor.last_tff == pytest.approx(-19.0, abs=400.0 / 4095 + 1e-9)
    drv.stop(t_ff=12.0)
    assert motor.last_tff == pytest.approx(12.0, abs=400.0 / 4095 + 1e-9)


def test_stop_holds_present_pose():
    motor = FakeMotor(pos=1.0)
    drv = make_driver(motor)
    drv.set_torque(True)
    motor.pos = 1.6                                    # joint moved (or was moving)
    drv.stop()
    p_des, _, _ = motor.last_mit
    assert p_des == pytest.approx(1.6, abs=1e-3)


def test_stop_noop_when_deenergized():
    motor = FakeMotor()
    drv = make_driver(motor)
    drv.stop()
    assert drv._ser.tx == []                           # a free joint is already safe


# ── hot-replug ───────────────────────────────────────────────────────────────
def test_write_failure_drops_socket():
    drv = make_driver(FakeMotor())
    drv._ser.fail_writes = True
    with pytest.raises(SubsystemUnavailable):
        drv.read()
    assert not drv.connected()                         # next connect() re-resolves by VID


def test_close_disables_first():
    motor = FakeMotor()
    drv = make_driver(motor)
    drv.set_torque(True)
    ser = drv._ser
    drv.close()
    assert ser.closed and not motor.enabled
    assert ser.tx[-1] == (CAN_ID, b"\xff" * 7 + b"\xfd")
    assert not drv.connected() and drv.is_torque_enabled is None


# ── port resolution ──────────────────────────────────────────────────────────
def test_resolve_port_explicit_wins():
    assert ds.resolve_port("/dev/ttyACM7") == "/dev/ttyACM7"


def test_resolve_port_auto_by_vid(monkeypatch):
    class P:
        def __init__(self, device, vid):
            self.device, self.vid = device, vid
    fake_lp = types.SimpleNamespace(
        comports=lambda: [P("/dev/ttyACM0", 0x16C0),   # base Teensy — must be skipped
                          P("/dev/ttyACM3", 0x2E88)])  # the DaMiao dongle
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    monkeypatch.setitem(sys.modules, "serial.tools", types.ModuleType("serial.tools"))
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_lp)
    assert ds.resolve_port("auto") == "/dev/ttyACM3"


def test_mit_quantization_error_bounded():
    # command-side pack + feedback-side unpack stay within one quantum across the range
    for x in (-12.5, -3.3, 0.0, 0.7331, 12.5):
        q = u2f(f2u(x, -12.5, 12.5, 16), -12.5, 12.5, 16)
        assert math.isclose(q, x, abs_tol=25.0 / 65535)
