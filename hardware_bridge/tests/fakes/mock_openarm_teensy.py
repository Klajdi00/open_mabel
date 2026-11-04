"""MockOpenArmTeensy — a fake arms Teensy speaking the openarm-dual wire protocol.

The counterpart to mock_arm_teensy.py (which speaks the robot_host protocol for
the not-yet-flashed ArmLink gateway). This one mimics the firmware ACTUALLY
FLASHED on the robot: protobuf-over-UDP, self-describing MANIFEST, STATE +
HEALTH telemetry, and the MIT COMMAND the bridge sends.

Encoders FOLLOW the last commanded position (first-order lag) only while a
position term is present (kp > 0), so a damping hold (kp == 0) holds still —
like the real firmware. Standalone smoke:

    python3 tests/fakes/mock_openarm_teensy.py --port 5810
"""
import os
import socket
import struct
import sys
import threading
import time

if __package__ in (None, ""):        # run as a script: find mabel_hw
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

TAG_COMMAND, TAG_HREQ, TAG_HELLO = 0x01, 0x02, 0x03
TAG_STATE, TAG_HEALTH, TAG_MANIFEST = 0x10, 0x11, 0x12

# (name, motor, bus, id, t_max) — mirrors the real openarm-dual manifest: DM8009
# shoulders, DM4340 elbows, DM4310 wrists; right = bus 1, left = bus 2.
LAYOUT = ([(f"l_j{i}", m, 2, i, t) for i, (m, t) in enumerate(
              [("DM8009", 54.0), ("DM8009", 54.0), ("DM4340", 27.0),
               ("DM4340", 27.0), ("DM4310", 10.0), ("DM4310", 10.0),
               ("DM4310", 10.0)], start=1)]
          + [(f"r_j{i}", m, 1, i, t) for i, (m, t) in enumerate(
              [("DM8009", 54.0), ("DM8009", 54.0), ("DM4340", 27.0),
               ("DM4340", 27.0), ("DM4310", 10.0), ("DM4310", 10.0),
               ("DM4310", 10.0)], start=1)])


def _varint(b, i):
    v = s = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        if not x & 0x80:
            return v, i
        s += 7


def _enc_varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _tag(field, wt):
    return _enc_varint((field << 3) | wt)


def _len_delim(field, body):
    return _tag(field, 2) + _enc_varint(len(body)) + body


def _f32(field, v):
    return _tag(field, 5) + struct.pack("<f", float(v))


def _fields(b):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield f, v
        elif wt == 5:
            (v,) = struct.unpack_from("<f", b, i)
            i += 4
            yield f, v
        elif wt == 2:
            ln, i = _varint(b, i)
            yield f, b[i:i + ln]
            i += ln
        else:
            raise ValueError(f"wire type {wt}")


def _unpack_floats(body):
    return [struct.unpack_from("<f", body, o)[0] for o in range(0, len(body), 4)]


class MockOpenArmTeensy:
    """UDP server: Hello -> Manifest, streams State, answers HealthRequest."""

    def __init__(self, layout=None, host="127.0.0.1", port=0, *, alpha=0.5,
                 temp_mos=35, temp_rotor=40):
        self.layout = list(layout or LAYOUT)
        self.n = len(self.layout)
        self.names = [j[0] for j in self.layout]
        self.alpha = float(alpha)
        self.temp_mos, self.temp_rotor = temp_mos, temp_rotor
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.client = None
        self.stream_hz = 200
        self.running = False
        self.hello_count = 0
        self.cmd_count = 0
        self.last_seq = 0
        self.last_cmd = None            # dict(pos=, vel=, kp=, kd=, t_ff=)
        self.pos = [0.0] * self.n
        self.tau = [0.0] * self.n

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        self.running = True
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._tx, daemon=True).start()
        return self

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.sock.close()

    def reboot(self, silence: float = 1.3):
        """Go dark, then come back — a power-cycle / replug / cable glitch.

        Restarts BOTH loops: stopping only _tx leaves the board deaf, so
        `last_cmd` freezes at the pre-gap frame and a test reads stale bytes as
        if they were live traffic (that artifact hid a real limp-arm-driven bug).
        A real board that returns both streams AND listens again."""
        self.running = False
        time.sleep(silence)
        self.running = True
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._tx, daemon=True).start()
        return self

    def set_pos(self, **by_name):
        """Force encoder readings, e.g. m.set_pos(l_j4=1.23)."""
        for k, v in by_name.items():
            self.pos[self.names.index(k)] = float(v)

    # ── encode ───────────────────────────────────────────────────────────────
    def _manifest(self):
        body = _len_delim(1, b"openarm-dual")
        for name, motor, bus, mid, t_max in self.layout:
            spec = (_len_delim(1, name.encode()) + _len_delim(2, motor.encode())
                    + _tag(3, 0) + _enc_varint(bus)
                    + _tag(4, 0) + _enc_varint(mid)
                    + _f32(6, 12.5) + _f32(7, 30.0) + _f32(8, t_max))
            body += _len_delim(4, spec)
        return bytes([TAG_MANIFEST]) + body

    def _state(self):
        body = _tag(2, 0) + _enc_varint(int(time.monotonic() * 1e6) & 0xFFFFFFFF)
        for i in range(self.n):
            body += _len_delim(4, _f32(1, self.pos[i]) + _f32(2, 0.0)
                               + _f32(3, self.tau[i]))
        return bytes([TAG_STATE]) + body

    def _health(self, seq):
        body = _tag(1, 0) + _enc_varint(seq)
        for _ in range(self.n):
            body += _len_delim(3, _tag(2, 0) + _enc_varint(self.temp_mos)
                               + _tag(3, 0) + _enc_varint(self.temp_rotor))
        return bytes([TAG_HEALTH]) + body

    # ── loops ────────────────────────────────────────────────────────────────
    def _rx(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            if not data:
                continue
            self.client = addr
            tag, body = data[0], data[1:]
            try:
                if tag == TAG_HELLO:
                    self.hello_count += 1
                    self.sock.sendto(self._manifest(), addr)
                elif tag == TAG_HREQ:
                    seq = next((v for f, v in _fields(body) if f == 1), 0)
                    self.sock.sendto(self._health(seq), addr)
                elif tag == TAG_COMMAND:
                    self._on_command(body)
            except Exception:
                pass

    def _on_command(self, body):
        got = {}
        for f, v in _fields(body):
            if f == 1:
                self.last_seq = v
            elif f in (2, 3, 4, 5, 6):
                got[{2: "pos", 3: "vel", 4: "kp", 5: "kd",
                     6: "t_ff"}[f]] = _unpack_floats(v)
        if len(got) != 5:
            return
        self.cmd_count += 1
        self.last_cmd = got
        # Track only where a position term exists; kp==0 is a damping hold.
        for i in range(min(self.n, len(got["pos"]))):
            if got["kp"][i] > 1e-6:
                err = got["pos"][i] - self.pos[i]
                self.pos[i] += self.alpha * err
                self.tau[i] = got["kp"][i] * err + got["t_ff"][i]
            else:
                self.tau[i] = got["t_ff"][i]

    def _tx(self):
        while self.running:
            time.sleep(1.0 / self.stream_hz)
            if self.client is not None:
                try:
                    self.sock.sendto(self._state(), self.client)
                except OSError:
                    pass


if __name__ == "__main__":       # pragma: no cover — manual smoke
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5810)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    m = MockOpenArmTeensy(host=a.host, port=a.port).start()
    print(f"mock openarm-dual on {a.host}:{m.port} ({m.n} actuators)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        m.stop()
