#!/usr/bin/env python3
"""fake_teensy.py — a software mock of the MAXSwerve Teensy (UDP + protobuf).

Speaks the EXACT wire protocol of the real firmware (maxswerve_host.swerve tags +
maxswerve_pb2 messages) so the hardware bridge can be tested end-to-end with NO real
Teensy and NO motion risk. Point the bridge at 127.0.0.1 and this stands in:

    * Hello (0x03)        -> registers the client, (re)sets stream rate
    * Command (0x01)      -> latches drive/steer, LOGS them (so a test can assert the
                             exact IK+calibration the bridge sent), echoes seq
    * HealthRequest(0x02) -> replies Health (0x11)
    * streams State (0x10) at stream_hz, with encoders that FOLLOW the last command
      (drive_velocity = commanded drive m/s, steer_angle = commanded steer rot) so the
      full state round-trip (/joint_states -> /odom) is exercised.

Prints one status line/second: RX command count, how many DISTINCT commands were seen
in the last second (1 == perfectly steady, >1 == the bridge is oscillating), and the
last drive/steer. Run:  python3 tools/fake_teensy.py [--host 127.0.0.1] [--port 5800]
"""

import argparse
import os
import socket
import sys
import threading
import time

# the vendored protobuf lives in the library now
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from mabel_hw.drivers.maxswerve_host import maxswerve_pb2 as pb   # noqa: E402

TAG_COMMAND, TAG_HEALTH_REQUEST, TAG_HELLO, TAG_STATE, TAG_HEALTH = 0x01, 0x02, 0x03, 0x10, 0x11


class FakeTeensy:
    def __init__(self, host="127.0.0.1", port=5800):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.client = None
        self.stream_hz = 200
        self.drive = [0.0, 0.0, 0.0]
        self.steer = [0.0, 0.0, 0.0]
        self.cmd_count = 0
        self.hello_count = 0
        self._recent = []          # commands seen since the last status print
        self._lock = threading.Lock()
        self.running = True

    def start(self):
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._stream, daemon=True).start()
        threading.Thread(target=self._status, daemon=True).start()
        return self

    def _rx(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            self.client = addr
            tag, body = data[0], data[1:]
            if tag == TAG_HELLO:
                h = pb.Hello(); h.ParseFromString(body)
                self.hello_count += 1
                if h.stream_hz:
                    self.stream_hz = h.stream_hz
            elif tag == TAG_COMMAND:
                c = pb.Command(); c.ParseFromString(body)
                drive, steer = list(c.drive), list(c.steer)
                key = (tuple(round(x, 4) for x in drive), tuple(round(x, 4) for x in steer))
                with self._lock:
                    changed = (not self._recent) or key != self._recent[-1]
                    self.drive = drive
                    self.steer = steer
                    self.cmd_count += 1
                    self._recent.append(key)
                # On a DISTINCT command, emit a precise RX timestamp (CLOCK_MONOTONIC,
                # system-wide on Linux) so a latency harness can correlate pub->rx.
                if changed:
                    print(f"RXSTEP {time.monotonic():.6f} drive0={drive[0]:.4f}", flush=True)
                self._reply_state(c.seq)         # immediate echo too
            elif tag == TAG_HEALTH_REQUEST:
                hr = pb.HealthRequest(); hr.ParseFromString(body)
                self._reply_health(hr.seq)

    def _reply_state(self, seq):
        if self.client is None:
            return
        st = pb.State()
        st.seq = seq
        st.t_us = int(time.monotonic() * 1e6) & 0xFFFFFFFF
        st.live_mask = 0b111
        with self._lock:
            drive, steer = list(self.drive), list(self.steer)
        for i in range(3):
            m = st.modules.add()
            m.drive_velocity = float(drive[i])      # encoders follow command (m/s)
            m.drive_position = 0.0
            m.steer_angle = float(steer[i]) % 1.0   # rotations [0,1)
            m.drive_torque_est = 0.0
        try:
            self.sock.sendto(bytes([TAG_STATE]) + st.SerializeToString(), self.client)
        except OSError:
            pass

    def _reply_health(self, seq):
        if self.client is None:
            return
        h = pb.Health(); h.seq = seq; h.t_us = int(time.monotonic() * 1e6) & 0xFFFFFFFF
        for _ in range(3):
            m = h.modules.add()
            m.drive_voltage = 24.0; m.drive_live = True; m.steer_live = True
        try:
            self.sock.sendto(bytes([TAG_HEALTH]) + h.SerializeToString(), self.client)
        except OSError:
            pass

    def _stream(self):
        while self.running:
            self._reply_state(0)
            time.sleep(1.0 / max(self.stream_hz, 1))

    def _status(self):
        while self.running:
            time.sleep(1.0)
            with self._lock:
                recent = self._recent; self._recent = []
                drive, steer = list(self.drive), list(self.steer)
            distinct = len(set(recent))
            flag = "STEADY" if distinct <= 1 else f"OSCILLATING({distinct} distinct)"
            print(f"[fake_teensy] client={self.client} hello={self.hello_count} "
                  f"cmds={self.cmd_count} last_s={len(recent)} {flag} "
                  f"drive={[round(x,3) for x in drive]} steer={[round(x,3) for x in steer]}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5800)
    args = ap.parse_args()
    ft = FakeTeensy(args.host, args.port).start()
    print(f"[fake_teensy] listening on {args.host}:{args.port} (mock MAXSwerve Teensy)",
          flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ft.running = False


if __name__ == "__main__":
    main()
