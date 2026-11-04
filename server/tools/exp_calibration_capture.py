"""Regression probe for the 2026-07-18 calibration stall.

Speaks the Vision Pro's exact wire shapes: sets a WBC method (joint — the
branch that returned before controller.update and STARVED the capture),
starts the 1-pose body calibration, streams steady arms-forward frames, and
asserts robot_state.calibrationStatus reaches done=true with a measured reach.
"""
import asyncio, json, os, sys, time

import numpy as np
import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "vision_os_apps", "docs", "tools"))
from exp_hand_retarget import synth_hand   # noqa: E402

IDENT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def frame(seq, side_x):
    joints = []
    for name, T in synth_hand(mcp=0.05, pip=0.05, thumb=0.1).items():
        joints.append({"joint": name,
                       "localTransform": {"matrix": np.asarray(T, float).reshape(16).tolist()},
                       "isTracked": True})
    def hand(chir, x):
        anchor = [1.0, 0, 0, x, 0, 1, 0, -0.15, 0, 0, 1, -0.68, 0, 0, 0, 1]
        return {"chirality": chir, "anchorTransform": {"matrix": anchor},
                "joints": joints, "isTracked": True}
    return {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": IDENT}, "trackingState": "normal"},
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
        "leftHand": hand("left", -side_x), "rightHand": hand("right", side_x)}}


async def main():
    uri = "ws://127.0.0.1:%s/teleop?client=visionpro" % os.environ.get("MABEL_WS_PORT", "9090")
    async with websockets.connect(uri, max_size=2**20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello

        async def send(t, p):
            await ws.send(json.dumps({"type": t, "payload": p}))

        # The STARVED path: a WBC method active while calibrating.
        await send("control_mode", {"method": "joint", "controlType": "impedance",
                                    "region": "arm", "stiffness": 0.5})
        await send("body_calibration", {"action": "start", "user": "probe"})

        status = {}
        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "robot_state":
                    cs = msg["payload"].get("calibrationStatus")
                    if cs:
                        status.update(cs)
        rtask = asyncio.create_task(reader())

        seq = 0
        t0 = time.time()
        while time.time() - t0 < 10.0 and not status.get("done"):
            seq += 1
            await ws.send(json.dumps(frame(seq, 0.18)))
            await asyncio.sleep(0.05)
        rtask.cancel()
        print("STATUS:", json.dumps(status))
        ok = status.get("done") and status.get("reach")
        print("VERDICT:", "CALIBRATION COMPLETES (done + reach on the wire)"
              if ok else "STILL STUCK")
        sys.exit(0 if ok else 1)


asyncio.run(main())
