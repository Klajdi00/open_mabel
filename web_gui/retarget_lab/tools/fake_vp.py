#!/usr/bin/env python3
"""
A stand-in Vision Pro: streams an EgoDex clip as teleop_frame wire payloads to
a gateway, exactly as the MABEL visionOS app would (60 Hz, ping 1 Hz, row-major
matrices, named ARKit joints, anchorTransform = identity so localTransform is
the world pose). For developing/testing the lab without the headset.

    python3 tools/fake_vp.py [--host 127.0.0.1] [--clip <egodex .json.gz>] [--seconds 20]
    python3 tools/fake_vp.py --record 4    # also press the headset's record for 4 s
"""
import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CLIPS = REPO / "controller/experiments/data/egodex/benchmark/clips"
I16 = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def to_wire(p, seq, ts):
    def hand(side):
        j = p.get(f"{side}_joints")
        if not j:
            return None
        return {"chirality": side, "anchorTransform": {"matrix": I16}, "isTracked": True,
                "joints": [{"joint": k, "localTransform": {"matrix": v}, "isTracked": True}
                           for k, v in j.items()]}
    f = {"sequence": seq, "timestamp": ts,
         "head": {"transform": {"matrix": p["head_mat"]}, "trackingState": "normal"},
         "mode": "Arms & Hands", "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0}}
    for side, key in (("left", "leftHand"), ("right", "rightHand")):
        h = hand(side)
        if h:
            f[key] = h
    return {"type": "teleop_frame", "payload": f}


async def main(a):
    import websockets
    clip = Path(a.clip) if a.clip else sorted(CLIPS.glob("*.json.gz"))[0]
    pay = json.load(gzip.open(clip))["payloads"]
    url = f"ws://{a.host}:{a.port}/teleop?client=visionpro"
    async with websockets.connect(url, user_agent_header="MabelTeleop/fake visionOS") as ws:
        hello = json.loads(await ws.recv())
        print("hello:", hello["payload"])
        got = {"robot_state": 0}

        async def rx():
            async for m in ws:
                t = json.loads(m)["type"]
                got[t] = got.get(t, 0) + 1
        rt = asyncio.create_task(rx())
        t0 = time.time()
        seq = 0
        rec_on = False
        while time.time() - t0 < a.seconds:
            el = time.time() - t0
            i = int(el * 30.0) % len(pay)              # EgoDex is 30 fps; send at 60 Hz
            seq += 1
            await ws.send(json.dumps(to_wire(pay[i], seq, 7.8e8 + el)))
            if seq % 60 == 0:
                await ws.send(json.dumps({"type": "ping", "payload": {"t": time.time()}}))
            if a.record and not rec_on and el > 1.0:
                await ws.send(json.dumps({"type": "record", "payload": {"active": True, "task": "fake_vp"}}))
                rec_on = True
            if a.record and rec_on and el > 1.0 + a.record:
                await ws.send(json.dumps({"type": "record", "payload": {"active": False}}))
                a.record = 0
            await asyncio.sleep(1 / 60)
        rt.cancel()
        print(f"sent {seq} frames in {a.seconds}s; received {got}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--clip")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--record", type=float, default=0)
    asyncio.run(main(ap.parse_args()))
