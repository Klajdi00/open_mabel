"""MockTeensy — a fake arm/neck Teensy speaking the robot_host wire protocol.

VENDORED from the upstream arm stack and
adapted: parameterizable actuator names (openArm 14-joint and RSSR-neck 3-motor
layouts ship as presets), positions FOLLOW the last command (first-order lag)
so driver tests can assert tracking, and a standalone main() for the thor-side
isolated-domain smoke:

    python3 tests/fakes/mock_arm_teensy.py --port 5810 --layout arms
"""
import os
import socket
import sys
import threading
import time

if __package__ in (None, ""):        # run as a script (thor smoke): find mabel_hw
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

from mabel_hw.drivers.robot_host import robot_pb2 as pb          # noqa: E402

TAG_COMMAND, TAG_HREQ, TAG_HELLO = 0x01, 0x02, 0x03
TAG_STATE, TAG_HEALTH, TAG_ROBOT_INFO = 0x10, 0x11, 0x12

LAYOUTS = {
    "arms": [f"{s}_j{i}" for s in "lr" for i in range(1, 8)],   # openarm-dual
    "neck": ["motor_L1", "motor_L2", "motor_head"],             # RSSR neck
    "tiny": ["a1", "a2", "a3", "gripper"],                      # upstream's 4-DOF
}


class MockTeensy:
    """UDP server: Hello -> RobotInfo, streams State, answers HealthRequest.
    Encoders follow the last commanded position with per-tick alpha (so a
    torqueless damping hold — kp==0 — holds still, like the real firmware)."""

    def __init__(self, names=None, host="127.0.0.1", port=0, *,
                 robot_name="mockbot", alpha=0.5, dead=()):
        self.names = list(names or LAYOUTS["arms"])
        self.n = len(self.names)
        self.robot_name = robot_name
        self.alpha = float(alpha)
        self.dead = set(dead)                     # indices that report not-live
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
        self.last_cmd = None                      # dict(pos=…, vel=…, kp=…, kd=…, t_ff=…)
        self.pos = [0.0] * self.n

    def start(self):
        self.running = True
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._stream, daemon=True).start()
        return self

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass

    # ── wire ──────────────────────────────────────────────────────────────────
    def _rx(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.client = addr
            tag, body = data[0], data[1:]
            try:
                if tag == TAG_HELLO:
                    h = pb.Hello()
                    h.ParseFromString(body)
                    self.stream_hz = h.stream_hz or 200
                    self.hello_count += 1
                    self._send_robot_info()
                elif tag == TAG_COMMAND:
                    c = pb.Command()
                    c.ParseFromString(body)
                    self.last_seq = c.seq
                    self.last_cmd = dict(pos=list(c.pos), vel=list(c.vel),
                                         kp=list(c.kp), kd=list(c.kd),
                                         t_ff=list(c.t_ff))
                    self.cmd_count += 1
                elif tag == TAG_HREQ:
                    r = pb.HealthRequest()
                    r.ParseFromString(body)
                    self._send_health(r.seq)
            except OSError:
                # a reply to a STALE client addr (pre-reboot Hello) can raise
                # ECONNREFUSED via ICMP on loopback — a real Teensy wouldn't
                # care and neither do we; keep serving the live client
                continue

    def _send_robot_info(self):
        info = pb.RobotInfo(name=self.robot_name,
                            stream_default_hz=self.stream_hz, stream_max_hz=1000)
        for i, nm in enumerate(self.names):
            info.actuators.add(name=nm, model="DM4310", bus=1, can_id=i + 1,
                               mst_id=0x11 + i, p_max=12.5, v_max=30.0,
                               t_max=10.0, kp_max=500, kd_max=5,
                               default_kp=0.0, default_kd=1.0)
        self.sock.sendto(bytes([TAG_ROBOT_INFO]) + info.SerializeToString(),
                         self.client)

    def _step(self):
        """Encoders follow the command ONLY where kp > 0 (a pure damping hold
        — kp==0 — keeps position, like the real firmware idle)."""
        cmd = self.last_cmd
        if cmd is None:
            return
        for i in range(self.n):
            if i < len(cmd["kp"]) and cmd["kp"][i] > 1e-6:
                self.pos[i] += self.alpha * (cmd["pos"][i] - self.pos[i])

    def _stream(self):
        while self.running:
            time.sleep(1.0 / max(1, self.stream_hz))
            if self.client is None:
                continue
            self._step()
            live_mask = 0
            for i in range(self.n):
                if i not in self.dead:
                    live_mask |= 1 << i
            st = pb.State(seq=self.last_seq,
                          t_us=int(time.monotonic() * 1e6) & 0xFFFFFFFF,
                          live_mask=live_mask)
            for i in range(self.n):
                st.actuators.add(position=self.pos[i], velocity=0.0, torque=0.0)
            try:
                self.sock.sendto(bytes([TAG_STATE]) + st.SerializeToString(),
                                 self.client)
            except OSError:
                break

    def _send_health(self, seq):
        h = pb.Health(seq=seq, t_us=0)
        for i in range(self.n):
            live = i not in self.dead
            h.actuators.add(err=1 if live else 0, t_mos=25 + i, t_rotor=23,
                            enabled=live, live=live,
                            stale_ms=0 if live else 9999)
        self.sock.sendto(bytes([TAG_HEALTH]) + h.SerializeToString(), self.client)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="fake arm/neck Teensy (robot_host wire)")
    ap.add_argument("--port", type=int, default=5810,
                    help="UDP port to serve on (NOT 5800 — that's the base fake)")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="arms")
    args = ap.parse_args(argv)
    mock = MockTeensy(LAYOUTS[args.layout], port=args.port)
    mock.start()
    print(f"mock {args.layout} teensy on 127.0.0.1:{mock.port} "
          f"({mock.n} actuators); Ctrl-C to stop")
    try:
        while True:
            time.sleep(2.0)
            print(f"  hellos={mock.hello_count} cmds={mock.cmd_count} "
                  f"pos[0]={mock.pos[0]:+.3f}")
    except KeyboardInterrupt:
        mock.stop()


if __name__ == "__main__":
    main()
