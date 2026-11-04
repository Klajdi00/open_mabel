#!/usr/bin/env python3
"""
test_cockpit.py — offline test harness for the MABEL terminal cockpit.

Runs mabel_cockpit.py under a REAL pty (fixed 120×34 window, TERM=xterm-256color)
and asserts on the screen dumps it writes:

  T1  boot screen      (demo, 10 frames)   logo + port checklist render
  T2  idle cockpit     (demo, ~5 s)        all panels + sprite + joints table
  T3  driving          (demo, ~10 s)       asked twist ≠ 0, wheels spin, radar
  T4  e-stop           (demo, ~17.5 s)     E-STOP banner takes over
  T5  key handling     (demo + pty stdin)  'l' hides log, ENTER hits the FIFO,
                                           'q' quits early with rc 0
  T6  real WS server   (websockets lib)    TinyWS vs the SAME library the bridge
                                           serves :9600 with — pings answered,
                                           frames land, joints painted

Run:  python3 tui/test_cockpit.py            (any python3; T6 needs the mabel
                                              python for `websockets` — pass
                                              --py, defaults to $MABEL_PY)
"""
from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COCKPIT = os.path.join(HERE, "mabel_cockpit.py")
COLS, ROWS = 120, 34

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def run_cockpit(extra: list[str], keys: bytes = b"", timeout: float = 40.0,
                keys_after: float = 1.5) -> tuple[int, str, str]:
    """Run the cockpit on a pty; return (rc, dumped screen text, metrics line)."""
    dump = tempfile.mktemp(prefix="cockpit_dump_")
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", LANG="en_US.UTF-8",
               LC_ALL="en_US.UTF-8")
    proc = subprocess.Popen(
        [sys.executable, COCKPIT, "--dump", dump] + extra,
        stdin=slave, stdout=slave, stderr=subprocess.PIPE, env=env)
    os.close(slave)
    t0 = time.time()
    sent = not keys
    try:
        while proc.poll() is None:
            if not sent and time.time() - t0 >= keys_after:
                os.write(master, keys)
                sent = True
            r, _, _ = select.select([master], [], [], 0.2)
            if r:
                try:
                    os.read(master, 65536)          # drain the screen stream
                except OSError:
                    pass
            if time.time() - t0 > timeout:
                proc.send_signal(signal.SIGKILL)
                return (-9, "", "TIMEOUT")
    finally:
        os.close(master)
    err = proc.stderr.read().decode(errors="replace")
    text, metrics = "", ""
    if os.path.exists(dump):
        raw = open(dump).read()
        m = re.search(r"--- cockpit metrics: (.*) ---", raw)
        metrics = m.group(1) if m else ""
        text = raw
        os.unlink(dump)
    if proc.returncode not in (0,) and err.strip():
        print("    stderr:", err.strip().splitlines()[-1][:160])
    return (proc.returncode, text, metrics)


def t1_boot():
    print("T1 boot screen")
    rc, scr, met = run_cockpit(["--demo", "--frames", "10"])
    check("exits 0", rc == 0)
    check("logo pixels drawn", "▀" in scr or "█" in scr)
    check("cockpit title", "T E L E O P   C O C K P I T" in scr)
    check("port checklist", "telemetry" in scr and ":9600" in scr)
    check("insert-operator footer", "INSERT OPERATOR" in scr)
    print(f"    metrics: {met}")


def t2_idle():
    print("T2 idle cockpit (demo t≈5s)")
    rc, scr, met = run_cockpit(["--demo", "--frames", "75"])
    check("exits 0", rc == 0)
    for panel in ("drive", "joints", "status", "robot", "video", "lidar", "link"):
        check(f"panel '{panel}'", f" {panel} " in scr)
    check("sprite half-blocks", scr.count("▀") + scr.count("▄") > 40,
          f"{scr.count('▀') + scr.count('▄')} px chars")
    check("joint rows painted", bool(re.search(r"a1\s+[+-]\d+\.\d", scr)))
    check("lift row", "lift" in scr and "cm" in scr)
    check("video roster rows", "wrist L" in scr and "fps" in scr)
    check("odom line", "odom" in scr and "yaw" in scr)
    check("link chart labelled", "full bar 50" in scr)
    # the hardware panel (NOVA's, ported 2026-10-01): every module with a count, the
    # degraded ones named — the demo roster has a dead right shoulder2 + neck servos
    check("panel 'hardware' (compact: motors only at 34 rows)", " hardware " in scr and "cam head" not in scr)
    check("hardware badge counts motors/modules", bool(re.search(r"● \d+/\d+ motors · \d+/\d+ modules", scr)))
    check("arm L live 7/7", bool(re.search(r"● arm L\s+7/7", scr)))
    check("arm R partial 6/7, shoulder2 named", bool(re.search(r"◐ arm R\s+6/7", scr)) and "off: shoulder2" in scr)
    check("hands live 5/5 (roster count)", bool(re.search(r"● hand L\s+5/5", scr)))
    check("neck servos offline 0/2", bool(re.search(r"○ neck pit\S*\s+0/2", scr)))
    check("lift offline from roster", bool(re.search(r"○ lift\s+0/1", scr)))
    check("host row", "host" in scr and "mabel-demo" in scr and "lan " in scr)
    m = re.search(r"render avg=(\d+\.\d+)ms max=(\d+\.\d+)ms", met or "")
    if m:
        check("render < 25 ms/frame avg", float(m.group(1)) < 25.0,
              f"avg {m.group(1)} ms · max {m.group(2)} ms")
    print(f"    metrics: {met}")


def t3_drive():
    print("T3 driving (demo t≈10s)")
    rc, scr, met = run_cockpit(["--demo", "--frames", "150"])
    check("exits 0", rc == 0)
    check("asked vx nonzero", bool(re.search(r"vx fwd\s+\+0\.5", scr)))
    check("mode badge wholebody", "wholebody" in scr)
    check("ENGAGED badge", "ENGAGED" in scr)
    check("lidar points on radar", "•" in scr)
    check("wheels spinner", "◍" in scr)
    print(f"    metrics: {met}")


def t4_estop():
    print("T4 e-stop (demo t≈17.5s)")
    rc, scr, met = run_cockpit(["--demo", "--frames", "262"])
    check("exits 0", rc == 0)
    check("E-STOP banner", "E-STOP" in scr)
    print(f"    metrics: {met}")


def t5_keys():
    print("T5 keys: l shows log (auto-hidden at 34 rows), ENTER→FIFO, q quits early")
    fifo = tempfile.mktemp(prefix="cockpit_fifo_")
    os.mkfifo(fifo)
    rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)   # play the bridge's role
    rc, scr, _ = run_cockpit(
        ["--demo", "--frames", "90", "--fifo", fifo],
        keys=b"l\n", keys_after=3.0, timeout=30.0)     # keys, then redraws
    got = b""
    try:
        got = os.read(rfd, 64)
    except OSError:
        pass
    os.close(rfd)
    os.unlink(fifo)
    check("exits 0", rc == 0)
    check("log panel shown by 'l'", " log " in scr)
    check("ENTER reached the bridge FIFO", got == b"\n", repr(got))
    t0 = time.time()
    rc, _, _ = run_cockpit(["--demo", "--frames", "3000"],
                           keys=b"q", keys_after=2.0, timeout=30.0)
    check("q quits early (rc 0)", rc == 0 and time.time() - t0 < 25.0,
          f"took {time.time() - t0:.1f}s")


def t6_real_ws(py: str):
    print("T6 TinyWS vs a real `websockets` server (the bridge's library)")
    if not os.path.exists(py):
        check("mabel python present", False, f"{py} missing — skipped")
        return
    server_src = r"""
import asyncio, json, math, time, sys
import websockets

async def handler(ws):
    t0 = time.time()
    await ws.send(json.dumps({"type": "hello", "robot": {"name": "wstest",
        "server": "mabel-sim-test"}, "rates": {}}))
    while True:
        t = time.time() - t0
        joints = {f"left_arm_{i}": {"q": round(0.3*math.sin(t+i), 4),
                  "dq": 0.0, "tau": 0.0, "temp": None} for i in range(1, 8)}
        joints.update({f"right_arm_{i}": {"q": 0.1, "dq": 0.0, "tau": 0.0,
                      "temp": None} for i in range(1, 8)})
        await ws.send(json.dumps({"type": "proprio", "t": time.time(),
            "stamp": t, "joints": joints,
            "base": {"x": 0.0, "y": 0.0, "yaw": 0.0}}))
        await ws.send(json.dumps({"type": "control", "t": time.time(),
            "mode": {"method": "navigation"}, "desired": {"Left Arm 1": 0.5},
            "baseTwist": {"vx": 0, "vy": 0, "wz": 0}, "engaged": False,
            "estop": False}))
        await ws.send(json.dumps({"type": "status", "t": time.time(),
            "estop": False, "torque": {"arms": True}, "latency": {},
            "driver": {"clients": 1}}))
        await asyncio.sleep(0.02)

async def main():
    # ping_interval=1: exercise the cockpit's ping->pong within the test window
    async with websockets.serve(handler, "127.0.0.1", 9639, ping_interval=1,
                                ping_timeout=4):
        print("READY", flush=True)
        await asyncio.Future()

asyncio.run(main())
"""
    srv = subprocess.Popen([py, "-c", server_src], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    try:
        line = srv.stdout.readline()
        if "READY" not in line:
            check("ws test server started", False, line.strip()[:120])
            return
        rc, scr, met = run_cockpit(
            ["--frames", "120", "--telemetry-port", "9639"], timeout=30.0)
        check("exits 0", rc == 0)
        check("left out of boot (frames landed)", "INSERT OPERATOR" not in scr)
        check("robot name from hello", "wstest" in scr)
        check("joint value painted", bool(re.search(r"a1\s+[+-]\d+\.\d", scr)))
        check("survived server pings ≥ 7 s", "LINK" in scr and " 0 Hz" not in scr.split("LINK")[1][:8],
              scr.split("LINK")[1][:10].strip() if "LINK" in scr else "")
        print(f"    metrics: {met}")
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=5)
        except subprocess.TimeoutExpired:
            srv.kill()


def t7_real_mode():
    print("T7 real mode: watchdog [hw] line parsed into the systems row + real port set")
    watch = tempfile.mktemp(prefix="cockpit_watch_")
    with open(watch, "w") as f:
        f.write("  [hw 12:00:01] head+wrists ok | RS-usb:on(bound) ZED-usb:on aud-usb:on"
                " | odom:✓ arms:✓ neck-dxl:✓ neck-damiao:· torso:✓ lift:✓ handL:✓ handR:·"
                " wristL:✓ wristR:✓ scan:✓ audio:✓ | SLAM map:✓ mesh:✓\n")
    env_logs = f"watch={watch}"
    os.environ["MABEL_COCKPIT_LOGS"] = env_logs
    try:
        rc, scr, met = run_cockpit(["--mode", "real", "--demo", "--frames", "75"])
    finally:
        del os.environ["MABEL_COCKPIT_LOGS"]
        os.unlink(watch)
    check("exits 0", rc == 0)
    check("hw systems row painted", "[12:00:01]" in scr and "neck-dxl✓" in scr)
    check("real port set probed (studio :8094)", ":8094" in scr)
    check("map plane probed (:9092)", ":9092" in scr)
    print(f"    metrics: {met}")


def t8_fullscreen():
    print("T8 fullscreen 250×68: HD sprite + module separators + hand rows")
    global COLS, ROWS
    COLS, ROWS = 250, 68
    try:
        rc, scr, met = run_cockpit(["--demo", "--frames", "80"])
    finally:
        COLS, ROWS = 120, 34
    check("exits 0", rc == 0)
    px = scr.count("▀") + scr.count("▄") + scr.count("█")
    check("HD sprite fills the panel", px > 400, f"{px} px chars")
    for sec in ("torso", "lift", "neck", "hands"):
        check(f"'{sec}' separator rule", f" {sec} ─" in scr)
    check("deviation tracks", "·│·" in scr.replace("··", "·"))
    lines = scr.splitlines()
    ll = [i for i, ln in enumerate(lines) if "handL" in ln]
    rr = [i for i, ln in enumerate(lines) if "handR" in ln]
    check("handL and handR on their own lines",
          bool(ll and rr) and ll[0] != rr[0])
    check("neck joint rows", "pitch" in scr and "roll" in scr)
    check("hardware sensor cells (fullscreen)", "cam head" in scr and "lidar" in scr)
    check("hardware fault lines, one per module", "off: shoulder2" in scr)
    print(f"    metrics: {met}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=os.environ.get(
        "MABEL_PY", os.path.expanduser("~/anaconda3/bin/python3")),
        help="python with `websockets` for the T6 server")
    ap.add_argument("--only", default="", help="comma list e.g. 1,5")
    args = ap.parse_args()
    tests = {"1": t1_boot, "2": t2_idle, "3": t3_drive, "4": t4_estop,
             "5": t5_keys, "6": lambda: t6_real_ws(args.py), "7": t7_real_mode,
             "8": t8_fullscreen}
    picks = args.only.split(",") if args.only else sorted(tests)
    t0 = time.time()
    for k in picks:
        tests[k]()
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'PASS' if npass == len(results) else 'FAIL'}: "
          f"{npass}/{len(results)} checks in {time.time() - t0:.1f}s")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
