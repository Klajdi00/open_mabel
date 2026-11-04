#!/usr/bin/env python3
"""
exp_neck_stomp.py — does a live head in the teleop stream stomp joint-space
neck commands?

The VP app's joint-space neck pad sends `joint_command` for neck_yaw/pitch/roll
while its 60 Hz `teleop_frame` also carries a head block. If that head reads
trackingState "normal", the engine's relative neck mapping re-asserts the
anchor pose every frame and the joint commands do nothing ("the neck joystick
does not work"). iOS marks the head notAvailable when head-drive is off — the
VP app now does the same in windowed mode.

This probe emulates both variants against a LIVE sim bridge: stream frames with
head state X while ramping neck_yaw joint commands to 0.6 rad, and report how
far the realized neck_yaw actually moved.

    python3 server/tools/exp_neck_stomp.py            # both variants, verdicts
"""
import asyncio
import json
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

URI = "ws://127.0.0.1:9090/teleop"
SEND_HZ = 60.0
RAMP_S = 6.0
TARGET = 0.6


def mat(tx=0.0, ty=0.0, tz=0.0):
    return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1]


async def run(head_state):
    got = {"neck_yaw": 0.0}
    async with websockets.connect(URI, max_size=None) as ws:
        stop = asyncio.Event()

        async def rx():
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if d.get("type") != "robot_state":
                    continue
                j = d.get("payload", {}).get("jointPositions", {}) or {}
                if "neck_yaw" in j:
                    got["neck_yaw"] = j["neck_yaw"]

        rxt = asyncio.create_task(rx())
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "joint", "controlType": "impedance",
            "region": "arm", "stiffness": 300.0}}))
        t0 = time.time()
        seq = 0
        while time.time() - t0 < RAMP_S:
            seq += 1
            frac = min(1.0, (time.time() - t0) / (RAMP_S * 0.6))
            await ws.send(json.dumps({"type": "teleop_frame", "payload": {
                "sequence": seq, "timestamp": time.time(), "mode": "Arms & Hands",
                "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0},
                "head": {"transform": {"matrix": mat(0, 1.5, 0)},
                         "trackingState": head_state},
            }}))
            await ws.send(json.dumps({"type": "joint_command", "payload": {
                "joints": {"neck_yaw": TARGET * frac}}}))
            await asyncio.sleep(1.0 / SEND_HZ)
        stop.set()
        await asyncio.sleep(0.2)
        rxt.cancel()
    return got["neck_yaw"]


async def main():
    for state in ("normal", "notAvailable"):
        yaw = await run(state)
        moved = abs(yaw) > 0.3 * TARGET
        print(f"  head trackingState={state:13s} → realized neck_yaw {yaw:+.3f} rad  "
              f"({'COMMANDS WIN ✓' if moved else 'STOMPED ✗ (head holds the neck)'})")


if __name__ == "__main__":
    asyncio.run(main())
