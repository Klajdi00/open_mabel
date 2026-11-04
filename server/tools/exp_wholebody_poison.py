"""Whole-body 'does nothing' regression probe.

Reproduces the operator's real session shape: use a joint-mode surface first
(joint_command leaves ARM keys in the server's accumulated joint_targets),
THEN engage whole-body retargeting and stream frames with a sweeping wrist.
Before the fix the stale direct arm targets are re-applied AFTER the engine
every tick and the arms never move.
"""
import asyncio, json, os, sys, time

import numpy as np
import websockets

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "vision_os_apps", "docs", "tools"))
from exp_hand_retarget import synth_hand   # noqa: E402

IDENT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def frame(seq, wrist_xyz):
    joints = []
    for name, T in synth_hand(mcp=0.1, pip=0.1, thumb=0.1).items():
        joints.append({"joint": name,
                       "localTransform": {"matrix": np.asarray(T, float).reshape(16).tolist()},
                       "isTracked": True})
    dx, dy, dz = wrist_xyz
    anchor = [1.0, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz, 0, 0, 0, 1]
    return {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": IDENT}, "trackingState": "normal"},
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
        "rightHand": {"chirality": "right", "anchorTransform": {"matrix": anchor},
                      "joints": joints, "isTracked": True}}}


async def main():
    uri = "ws://127.0.0.1:%s/teleop?client=visionpro" % os.environ.get("MABEL_WS_PORT", "9090")
    async with websockets.connect(uri, max_size=2**20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello

        async def send(t, p):
            await ws.send(json.dumps({"type": t, "payload": p}))

        telemetry = {}
        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "robot_state":
                    telemetry.update(msg["payload"].get("jointPositions", {}))
        rtask = asyncio.create_task(reader())

        # 1) POISON: a joint-mode session leaves arm keys in joint_targets
        #    (the VP link re-sends its whole accumulated dict, so any prior
        #    Joint-space nudge / executed plan does exactly this).
        await send("control_mode", {"method": "joint", "controlType": "impedance",
                                    "region": "arm", "stiffness": 0.5})
        await asyncio.sleep(0.6)
        arm_keys = {f"right_arm_{i}": float(telemetry.get(f"right_arm_{i}", 0.0))
                    for i in range(1, 8)}
        await send("joint_command", {"joints": arm_keys})
        await asyncio.sleep(0.4)

        # 2) WHOLE-BODY: retarget spec (the app's applyTeleopSpec shape) + clutch.
        await send("control_mode", {"method": "retarget", "controlType": "impedance",
                                    "region": "whole_body", "stiffness": 0.0})
        await send("recalibrate", {})

        # 3) Stream: 2 s anchor at rest, then sweep the wrist forward 16 cm.
        seq = 0
        t0 = time.time()
        baseline = None
        while time.time() - t0 < 7.0:
            t = time.time() - t0
            g = 0.0 if t < 2.0 else min(1.0, (t - 2.0) / 1.5)
            seq += 1
            await ws.send(json.dumps(frame(seq, (0.16 * g, 0.0, -0.05 * g))))
            if 1.5 < t < 2.0 and baseline is None:
                baseline = {k: v for k, v in telemetry.items() if k.startswith("right_arm")}
            await asyncio.sleep(0.05)
        rtask.cancel()

        final = {k: v for k, v in telemetry.items() if k.startswith("right_arm")}
        base = baseline or {}
        delta = max((abs(final.get(k, 0) - base.get(k, 0)) for k in base), default=0.0)
        print(f"max right-arm delta after wrist sweep: {np.degrees(delta):.1f} deg")
        ok = delta > np.radians(8)
        print("VERDICT:", "RETARGET DRIVES THE ARMS (poison survived)" if ok
              else "ARMS FROZEN — stale joint targets override the retargeter")
        sys.exit(0 if ok else 1)


asyncio.run(main())
