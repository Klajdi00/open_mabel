#!/usr/bin/env python3
"""
test_live_sim.py — LIVE integration test of run.sh cockpit mode on the real
sim stack (MuJoCo + bridge + cameras + telemetry). What it proves:

  L1  ./run.sh (pty) boots the whole stack with the cockpit as the foreground
      TUI — ports 8788/9090/8080/9600 all come up. (NOT --mock: mock mode
      deliberately serves no network, so the cockpit has nothing to read.)
  L2  a second read-only cockpit attached to the SAME live :9600 paints real
      joints / robot name / green port dots (multi-subscriber plane)
  L3  the stdin FIFO really reaches the bridge: 'r' → "simulation reset to
      home" lands in the bridge console log
  L4  'q' in the cockpit tears EVERYTHING down: run.sh exits, every listener
      is freed, no sim process survives, the FIFO is removed

⚠ This intentionally REPLACES any sim stack already running on this machine
(exactly what ./run.sh always does). Takes ~1-3 min (MuJoCo cold load).
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
COCKPIT = os.path.join(HERE, "mabel_cockpit.py")
FIFO = "/tmp/mabel_bridge_stdin.fifo"
BRIDGE_LOG = "/tmp/mabel_bridge_console.log"
PORTS = {"physics": 8788, "teleop": 9090, "video": 8080, "telemetry": 9600}

results: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""),
          flush=True)


def port_up(port: int) -> bool:
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=0.4)
        s.close()
        return True
    except OSError:
        return False


def wait_ports(want_up: bool, timeout: float) -> tuple[bool, float]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        states = [port_up(p) for p in PORTS.values()]
        if all(s == want_up for s in states):
            return True, time.time() - t0
        time.sleep(1.0)
    return False, time.time() - t0


def spawn_run_sh() -> tuple[subprocess.Popen, int]:
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", MABEL_NO_BROWSER="1",
               LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8")
    # NOTE: no --mock — mock mode serves NO network at all (that is its point),
    # so the telemetry-driven cockpit only makes sense on a live (client-less
    # is fine) bridge. run.sh itself falls back to plain output on --mock.
    proc = subprocess.Popen(
        ["./run.sh", "--no-studio", "--no-monitor"],
        cwd=SERVER, stdin=slave, stdout=slave, stderr=slave,
        env=env, start_new_session=True)
    os.close(slave)
    # drain the cockpit's screen stream (and keep it for post-mortems)
    cap = open("/tmp/mabel_tui_test_pty.log", "wb")
    def _drain():
        while True:
            try:
                r, _, _ = select.select([master], [], [], 0.5)
                if r:
                    data = os.read(master, 65536)
                    if not data:
                        return
                    cap.write(data)
                    cap.flush()
            except OSError:
                return
    threading.Thread(target=_drain, daemon=True).start()
    return proc, master


def second_cockpit_dump() -> str:
    dump = tempfile.mktemp(prefix="live_dump_")
    m2, s2 = pty.openpty()
    fcntl.ioctl(s2, termios.TIOCSWINSZ, struct.pack("HHHH", 34, 120, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", LANG="en_US.UTF-8",
               LC_ALL="en_US.UTF-8")
    p = subprocess.Popen([sys.executable, COCKPIT, "--mode", "sim",
                          "--frames", "60", "--dump", dump],
                         stdin=s2, stdout=s2, stderr=subprocess.DEVNULL, env=env)
    os.close(s2)
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < 30:
        r, _, _ = select.select([m2], [], [], 0.2)
        if r:
            try:
                os.read(m2, 65536)
            except OSError:
                break
    os.close(m2)
    if p.poll() is None:
        p.kill()
    out = open(dump).read() if os.path.exists(dump) else ""
    if os.path.exists(dump):
        os.unlink(dump)
    return out


def main() -> int:
    t_all = time.time()
    # pre-clear any previous stack so "ports up" below means OUR fresh stack
    print("L0 clearing any previous sim stack")
    subprocess.run(["pkill", "-9", "-f",
                    "sim_teleop_bridge|sim_camera_server|simulation_bridge.simulation_bridge|"
                    "bash .*start_teleop.sh|bash .*server/run.sh"],
                   capture_output=True)
    wait_ports(False, 15.0)
    print("L1 boot: ./run.sh --no-studio --no-monitor (cockpit pty)")
    proc, master = spawn_run_sh()
    up, took = wait_ports(True, 180.0)
    check("all four listeners up (8788/9090/8080/9600)", up, f"{took:.0f}s to boot")
    if not up:
        os.killpg(proc.pid, signal.SIGKILL)
        return finish()
    time.sleep(4.0)          # let telemetry + mock hands warm up

    print("L2 second read-only cockpit on the LIVE :9600")
    scr = second_cockpit_dump()
    check("left the boot screen", "INSERT OPERATOR" not in scr and " drive " in scr)
    check("live robot name in header", "MABEL Sim" in scr)
    check("real joint rows painted", bool(re.search(r"a1\s+[+-]\d+\.\d", scr)))
    check("port dots green", all(f"●{n}" in scr for n in PORTS))
    m = re.search(r"state\s+(\d+) Hz", scr)
    check("state stream alive", bool(m) and int(m.group(1)) >= 25,
          f"{m.group(1) if m else '?'} Hz")

    print("L3 stdin FIFO → bridge ('r' = sim reset)")
    ok_fifo = False
    try:
        fd = os.open(FIFO, os.O_WRONLY | os.O_NONBLOCK)
        os.write(fd, b"r\n")
        os.close(fd)
        for _ in range(20):
            if "simulation reset to home" in open(BRIDGE_LOG).read():
                ok_fifo = True
                break
            time.sleep(0.5)
    except OSError as e:
        check("fifo open/write", False, str(e))
    check("'r' via FIFO reset the sim (bridge log confirms)", ok_fifo)

    print("L4 'q' in the cockpit tears everything down")
    try:
        os.write(master, b"q")
    except OSError as e:
        check("cockpit pty still alive for 'q'", False, str(e))
    t0 = time.time()
    rc = None
    while time.time() - t0 < 45:
        rc = proc.poll()
        if rc is not None:
            break
        time.sleep(0.5)
    check("run.sh exited on q", rc is not None, f"rc={rc} in {time.time() - t0:.0f}s")
    down, took = wait_ports(False, 20.0)
    check("every listener freed", down, f"{took:.0f}s")
    left = subprocess.run(
        ["pgrep", "-lf", "sim_teleop_bridge|sim_camera_server|simulation_bridge.simulation_bridge"],
        capture_output=True, text=True).stdout.strip()
    check("no sim process survives", left == "", left[:120])
    check("stdin FIFO removed", not os.path.exists(FIFO))
    if rc is None:
        os.killpg(proc.pid, signal.SIGKILL)
    print(f"\ntotal wall time {time.time() - t_all:.0f}s")
    return finish()


def finish() -> int:
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"{'PASS' if npass == len(results) else 'FAIL'}: {npass}/{len(results)} checks")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
