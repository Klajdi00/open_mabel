#!/usr/bin/env python3
"""swerve_axis_probe.py — drive ONE controlled navJoystick pulse at the MABEL
swerve bridge and print the feedback it streams back.

Speaks the exact MABEL teleop protocol the iOS / Vision Pro / web apps use, so it
exercises the real network path (this Mac -> thor:9090) and the bridge's decode +
IK, while letting us command a precise, bounded micro-move and read the robot's
response. Use it to verify each axis sign before trusting the apps.

    python swerve_axis_probe.py --host 192.168.1.42 --axis fwd --mag 0.4 --dur 0.5

Axes (app stick semantics): fwd|back (left stick Y), left|right (left stick X),
yawL|yawR (right stick X). The pulse ramps the stick to ±mag for --dur seconds,
then holds zero for 0.6 s (watchdog + settle). Ctrl-C sends zeros and exits.
"""

import asyncio
import argparse
import json
import time

import websockets

AXIS = {
    "fwd":  ("ly", -1.0),   # nav.ly < 0 = forward
    "back": ("ly", +1.0),
    "right":("lx", +1.0),   # nav.lx > 0 = strafe right
    "left": ("lx", -1.0),
    "yawR": ("rx", +1.0),
    "yawL": ("rx", -1.0),
}


async def run(args):
    key, sign = AXIS[args.axis]
    url = f"ws://{args.host}:{args.ws_port}/teleop?client=probe"
    print(f"[probe] connecting {url}")
    async with websockets.connect(url, open_timeout=6, max_size=2 ** 20) as ws:
        seq = 0
        t0 = time.time()
        last_print = 0.0

        async def reader():
            nonlocal last_print
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
                env = json.loads(msg)
                if env.get("type") == "hello":
                    print(f"[probe] hello: server={env.get('payload',{}).get('server')}")
                elif env.get("type") == "robot_state":
                    p = env.get("payload", {})
                    now = time.time()
                    if now - last_print > 0.1:   # ~10 Hz print
                        last_print = now
                        steer = p.get("base", {}).get("steer")
                        vel = p.get("driveVelocity")
                        steer_deg = [round(s * 57.2958, 1) for s in steer] if steer else None
                        print(f"   t={now-t0:4.1f}s  steer_deg(app fl,fr,b)={steer_deg}  "
                              f"driveVel={[round(v,3) for v in vel] if vel else None}  "
                              f"lat={p.get('latencyMs',0):.1f}ms")

        async def writer():
            nonlocal seq
            hz, period = 20.0, 1.0 / 20.0
            # Phase 1: the pulse.
            end = time.time() + args.dur
            while time.time() < end:
                seq += 1
                await send_frame(ws, seq, **{key: sign * args.mag})
                await asyncio.sleep(period)
            # Phase 2: zero hold (watchdog + settle).
            end = time.time() + 0.6
            while time.time() < end:
                seq += 1
                await send_frame(ws, seq)
                await asyncio.sleep(period)

        rtask = asyncio.create_task(reader())
        print(f"[probe] PULSE {args.axis}  {key}={sign*args.mag:+.2f}  for {args.dur}s "
              f"-> WATCH THE ROBOT")
        await writer()
        await asyncio.sleep(0.3)
        rtask.cancel()
        print("[probe] done (sent zeros).")


async def send_frame(ws, seq, lx=0.0, ly=0.0, rx=0.0, ry=0.0):
    env = {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]},
                 "trackingState": "normal"},
        "mode": "Base Driving",
        "navJoystick": {"lx": lx, "ly": ly, "rx": rx, "ry": ry}}}
    await ws.send(json.dumps(env))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.42")
    ap.add_argument("--ws-port", type=int, default=9090)
    ap.add_argument("--axis", choices=list(AXIS), default="fwd")
    ap.add_argument("--mag", type=float, default=0.4, help="stick deflection 0..1")
    ap.add_argument("--dur", type=float, default=0.5, help="pulse duration (s)")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[probe] interrupted")


if __name__ == "__main__":
    main()
