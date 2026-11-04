#!/usr/bin/env python3
"""diagnose.py — MABEL real-robot connectivity check (run ON thor).

Today only the DRIVE BASE is wired to thor, so this checks exactly that: the
Teensy link, the base controller, and the three swerve modules (per-module
drive / steer / CAN liveness + V/A/°C + live steer angle). It does NOT command
anything. (Upper body + sensors aren't connected yet — noted at the end; extend
this when they are.)

It asks the running bridge (ws://127.0.0.1:9090) for a read-only `diag` snapshot.
The bridge owns the Teensy link and queries getState() + getHealthState(), so we
never open a 2nd Teensy client that would fight the control stream. If the bridge
is down, the base shows UNKNOWN. It also inventories thor's USB/serial directly.

Exit code 0 only if the base link is up and all 3 wheel modules are live.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import subprocess
import sys
import time

BRIDGE_URL = os.environ.get("MABEL_BRIDGE", "ws://127.0.0.1:9090/teleop")
TEENSY_IP = os.environ.get("MABEL_TEENSY_IP", "192.168.124.10")

# ── ANSI (skip when not a TTY / NO_COLOR) ────────────────────────────────────
_C = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def c(s, code): return f"\033[{code}m{s}\033[0m" if _C else s
OK   = lambda s="✓": c(s, "1;32")    # green
BAD  = lambda s="✗": c(s, "1;31")    # red
NA   = lambda s="○": c(s, "2;37")    # dim — not present
WARN = lambda s="!": c(s, "1;33")    # yellow
def head(s): print("\n" + c(s, "1;36"))


# ── host hardware probes ─────────────────────────────────────────────────────
def _lsusb() -> str:
    try:
        return subprocess.run(["lsusb"], capture_output=True, text=True, timeout=4).stdout
    except Exception:
        return ""

def _ping(ip: str) -> bool:
    try:
        return subprocess.run(["ping", "-c1", "-W2", ip], capture_output=True,
                              timeout=4).returncode == 0
    except Exception:
        return False

def probe_host() -> dict:
    usb = _lsusb()
    acm = sorted(glob.glob("/dev/ttyACM*"))
    ttyusb = sorted(glob.glob("/dev/ttyUSB*"))     # RS-485 / CAN actuator buses, lidar
    video = sorted(glob.glob("/dev/video*"))
    return {
        "teensy_usb": ("Teensyduino" in usb) or ("16c0:0483" in usb),
        "ttyACM": acm, "ttyUSB": ttyusb, "video": video,
        "teensy_ping": _ping(TEENSY_IP),
    }


# ── ask the bridge for the base/wheel health ─────────────────────────────────
async def query_bridge() -> dict | None:
    try:
        import websockets
    except Exception:
        return {"error": "websockets not installed in this interpreter"}
    try:
        async with websockets.connect(BRIDGE_URL, open_timeout=5) as ws:
            # drain hello, then ask for diag
            await asyncio.wait_for(ws.recv(), 5)
            await ws.send(json.dumps({"type": "diag", "payload": {}}))
            for _ in range(60):                      # skip robot_state until diag_report
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                if m.get("type") == "diag_report":
                    return m.get("payload", {})
            return {"error": "no diag_report from bridge"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── render ───────────────────────────────────────────────────────────────────
def row(label, mark, detail=""):
    print(f"  {label:<18} {mark}  {detail}")

def main() -> int:
    host = probe_host()
    base = asyncio.run(query_bridge()) or {}
    ok_overall = True

    print(c("━" * 64, "1;36"))
    print(c("  MABEL ROBOT DIAGNOSIS", "1;37"),
          c(f"· thor · {time.strftime('%Y-%m-%d %H:%M:%S')}", "2;37"))
    print(c("━" * 64, "1;36"))

    # ── DRIVE BASE ──
    head("DRIVE BASE  (Teensy → SPARK → MAXSwerve)")
    err = base.get("error")
    link = bool(base.get("link"))
    if err:
        row("Teensy link", WARN("?"), c(f"bridge unreachable ({err})", "2;37"))
        row("Base controller", WARN("?"), "start mabel-bridge to read base health")
        ok_overall = False
    else:
        rtt = base.get("rttMs")
        row("Teensy link", OK() if link else BAD(),
            f"{base.get('teensyIp', TEENSY_IP)}" + (f"  rtt {rtt}ms" if rtt else "")
            + (f"  {base.get('streamPackets')} pkts" if base.get("streamPackets") else ""))
        row("Base controller", OK("online") if link else BAD("OFFLINE"),
            "" if link else "no telemetry — is the robot powered + Teensy cabled?")
        row("Armed", OK("yes — motion enabled") if base.get("armed") else NA("no — observe-only"),
            "" if base.get("armed") else "run mabel-arm to drive")
        ok_overall &= link

    # ── WHEELS (3 swerve modules) ──
    head("WHEELS  (3 swerve modules: drive + steer live)")
    mods = base.get("modules") or []
    if not mods:
        row("modules", WARN("?"), "no module data (bridge down or no telemetry)")
        ok_overall = False
    for m in mods:
        live = m.get("live")                    # passive live_mask: drive & steer both live
        mark = OK() if live else BAD("not live")
        if live is not None:
            ok_overall &= bool(live)
        det = []
        if m.get("steerAngleRot") is not None:
            det.append(c(f"steer={m['steerAngleRot']}rot  drive={m.get('driveVel')}m/s", "2;37"))
        if live is False:
            det.append(BAD("drive/steer not reporting — CAN fault? power-cycle the module"))
        row(m.get("name", f"module {m.get('index')}"), mark, "  ".join(det))

    # ── USB / serial inventory (context) ──
    head("THOR I/O")
    row("Teensy USB", OK() if host["teensy_usb"] else BAD(),
        "16c0:0483 Teensyduino" if host["teensy_usb"] else "Teensy USB not seen")
    row("Serial (ACM)", OK(", ".join(host["ttyACM"])) if host["ttyACM"] else NA("none"))
    row("Serial (USB)", OK(", ".join(host["ttyUSB"])) if host["ttyUSB"] else NA("none"))
    row("Robot net", OK() if host["teensy_ping"] else BAD(),
        f"Teensy {TEENSY_IP} " + ("reachable" if host["teensy_ping"] else "no ping"))

    # Not wired to thor yet — noted briefly so the report is honest, not cluttered.
    print(c("\n  Not on thor yet: arms · hands · torso · lift · neck · cameras · lidar "
            "(base-only robot today)", "2;37"))

    print(c("━" * 64, "1;36"))
    nlive = sum(1 for m in mods if (m.get("driveLive") and m.get("steerLive") and not m.get("busDown"))
                or (m.get("driveLive") is None and m.get("live")))
    summary = f"drive base {'UP' if link else 'DOWN'} · wheels {nlive}/3 modules live"
    print("  " + (c("SUMMARY: ", "1;37")) + (OK(summary) if ok_overall else WARN(summary)))
    print(c("━" * 64, "1;36"))
    return 0 if ok_overall else 1


if __name__ == "__main__":
    sys.exit(main())
