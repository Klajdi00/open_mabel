#!/usr/bin/env python3
"""handshake_seq_test.py — proves the Teensy command handshake is thread-safe.

Every Command carries a sequence number ("handshake byte/word") that the firmware
echoes and uses to ignore any command not newer than the last one it applied. If two
host threads ever interleave the non-atomic "bump seq -> serialize -> sendto" critical
section, commands reach the Teensy OUT OF ORDER (or with a duplicated seq); the firmware
then DROPS the loser and the affected module holds its last setpoint — an intermittent
single-wheel stall. SwerveClient must serialize that section.

Concurrent senders are real in this stack: ROS cmd_vel, optional PS5 pad teleop, a
health poller, and the keepalive can all hit one SwerveClient. This test hammers
setCommand from several threads (plus getHealthState + the keepalive) against an
in-process recording Teensy and asserts, at the receiver:

    * NO duplicate seqs            (no two commands reused a handshake number)
    * NO out-of-order seqs         (the firmware never sees a stale command after a newer)
    * NO exceptions in the path    (shared RTT bookkeeping is locked vs the RX thread)
    * exact per-module round-trip  (drive/steer serialize module-aligned)

Runs entirely on 127.0.0.1 — NO real Teensy, NO motion risk. Exit 0 = pass.
"""
import os
import socket
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "base"))
from maxswerve_host import maxswerve_pb2 as pb           # noqa: E402
from maxswerve_host.swerve import SwerveClient           # noqa: E402

TAG_COMMAND, TAG_HEALTH_REQUEST, TAG_HELLO, TAG_STATE, TAG_HEALTH = 0x01, 0x02, 0x03, 0x10, 0x11


class RecordingTeensy:
    """Mock Teensy that records every Command's (seq, drive, steer) in arrival order."""

    def __init__(self, host="127.0.0.1", port=5800):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.client = None
        self.cmds = []           # (seq, tuple(drive), tuple(steer)) in receive order
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        threading.Thread(target=self._rx, daemon=True).start()
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
            if tag == TAG_COMMAND:
                c = pb.Command(); c.ParseFromString(body)
                with self.lock:
                    self.cmds.append((c.seq, tuple(c.drive), tuple(c.steer)))
                self._reply_state(c.seq)
            elif tag == TAG_HEALTH_REQUEST:
                hr = pb.HealthRequest(); hr.ParseFromString(body)
                self._reply_health(hr.seq)

    def _reply_state(self, seq):
        if self.client is None:
            return
        st = pb.State(); st.seq = seq; st.live_mask = 0b111
        for _ in range(3):
            st.modules.add()
        try:
            self.sock.sendto(bytes([TAG_STATE]) + st.SerializeToString(), self.client)
        except OSError:
            pass

    def _reply_health(self, seq):
        if self.client is None:
            return
        h = pb.Health(); h.seq = seq
        for _ in range(3):
            m = h.modules.add(); m.drive_live = True
        try:
            self.sock.sendto(bytes([TAG_HEALTH]) + h.SerializeToString(), self.client)
        except OSError:
            pass


def main():
    PORT = 5809
    teensy = RecordingTeensy(port=PORT).start()
    time.sleep(0.2)
    sc = SwerveClient(teensy_ip="127.0.0.1", port=PORT, keepalive_interval=0.05)
    sc.connect()
    time.sleep(0.3)

    fails = []

    # --- exact per-module serialization round-trip --------------------------
    sc.setCommand(np.float32([1.234, 0.0, -2.5]), np.float32([0.137, 0.951, 0.5]))
    time.sleep(0.05)
    with teensy.lock:
        _, d, s = teensy.cmds[-1]
    if not (np.allclose(d, [1.234, 0.0, -2.5], atol=1e-4)
            and np.allclose(s, [0.137, 0.951, 0.5], atol=1e-4)):
        fails.append(f"serialization mismatch: drive={d} steer={s}")

    # --- concurrent senders + health poller ---------------------------------
    errs = []
    NTHREADS, PER = 6, 500

    def hammer(tid):
        try:
            for _ in range(PER):
                sc.setCommand(np.float32([tid, tid, tid]) * 0.01,
                              np.float32([0.1, 0.2, 0.3]))
        except Exception as e:                      # pragma: no cover
            errs.append(f"sender{tid}: {type(e).__name__}: {e}")

    def health():
        try:
            for _ in range(80):
                sc.getHealthState(timeout=0.02)
        except Exception as e:                      # pragma: no cover
            errs.append(f"health: {type(e).__name__}: {e}")

    with teensy.lock:
        base_n = len(teensy.cmds)
    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(NTHREADS)]
    threads.append(threading.Thread(target=health))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(0.2)

    with teensy.lock:
        seqs = [c[0] for c in teensy.cmds[base_n:]]
    dup = len(seqs) - len(set(seqs))
    nonmono = sum(1 for a, b in zip(seqs, seqs[1:]) if b <= a)

    print(f"commands received under concurrency : {len(seqs)}")
    print(f"duplicate handshake seqs           : {dup}")
    print(f"out-of-order handshake seqs        : {nonmono}")
    print(f"exceptions in send/health path     : {len(errs)}")
    for e in errs[:5]:
        print("   !", e)

    if dup:
        fails.append(f"{dup} duplicate handshake seqs (concurrent seq reuse)")
    if nonmono:
        fails.append(f"{nonmono} out-of-order handshake seqs (firmware would drop -> stall)")
    if errs:
        fails.append(f"{len(errs)} exceptions in the send/health path")

    sc.close()
    teensy.running = False
    time.sleep(0.1)

    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("\nPASS — handshake seq is unique, ordered, and exception-free under concurrency.")
    sys.exit(0)


if __name__ == "__main__":
    main()
