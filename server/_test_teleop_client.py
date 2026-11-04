"""Headless teleop client: streams synthetic head+hand frames to the bridge over the
real WebSocket and records the robot's arm response — to verify the robot FOLLOWS and
does not keep resetting to HOME. No Vision Pro / device needed."""
import asyncio
import os
import json
import sys
import time

import numpy as np
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "controller"))
from mabel.calibration import synthesize_pose, POSE_HANDS

HAND = {"palm_len": 0.09, "fingers": {"thumb": 0.062, "index": 0.082, "middle": 0.094,
                                      "ring": 0.088, "pinky": 0.071},
        "pinch_open": 0.09, "pinch_close": 0.02}
BASE = synthesize_pose(POSE_HANDS, height=1.75, hand=HAND, n=1)[0]


def _ml(M):
    return np.asarray(M, float).flatten().tolist()


def _hand(joints, chir, offset):
    if not joints:
        return None
    js = []
    for name, M in joints.items():
        Mo = np.asarray(M, float).copy()
        Mo[:3, 3] = Mo[:3, 3] + offset
        js.append({"joint": name, "localTransform": {"matrix": _ml(Mo)}, "isTracked": True})
    return {"chirality": chir, "anchorTransform": {"matrix": _ml(np.eye(4))},
            "isTracked": True, "joints": js}


def frame(seq, t, offset):
    return {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": t, "mode": "Arms & Hands",
        "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0},
        "head": {"transform": {"matrix": _ml(BASE.head)}, "trackingState": "normal"},
        "rightHand": _hand(BASE.right_joints, "right", offset),
        "leftHand": _hand(BASE.left_joints, "left", offset)}}


async def main():
    uri = "ws://127.0.0.1:9090/teleop?client=test"
    arm_keys = [f"right_arm_{i}" for i in range(1, 8)]
    samples = []     # (t, phase, [arm joints])
    phase = {"v": "A"}
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "control_mode",
                                  "payload": {"method": "retarget", "controlType": "impedance",
                                              "region": "whole_body", "stiffness": 0.0}}))

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "robot_state":
                    jp = m["payload"].get("jointPositions", {})
                    if jp:
                        samples.append((time.time(), phase["v"], [jp.get(k, 0.0) for k in arm_keys]))

        rt = asyncio.create_task(reader())
        t0 = time.time()
        seq = 0
        # A (0-3s): head+hands, reach sinusoidally -> arm must FOLLOW.
        # B (3-6s): HEAD ONLY (no hands, e.g. base driving) -> arms must HOLD, not reset.
        # C (6-9s): stop streaming entirely (headset off) -> arms ease to HOME once.
        while time.time() - t0 < 9.0:
            tt = time.time() - t0
            phase["v"] = "A" if tt < 3.0 else ("B" if tt < 6.0 else "C")
            seq += 1
            if phase["v"] == "A":
                reach = 0.0 if tt < 1.0 else 0.18 * np.sin((tt - 1.0) * 1.2)
                offset = np.array([0.0, 0.10 * max(0, reach), -abs(reach)])
                f = frame(seq, time.time(), offset)
            elif phase["v"] == "B":
                f = frame(seq, time.time(), np.zeros(3))
                f["payload"]["rightHand"] = None      # head only (no hands)
                f["payload"]["leftHand"] = None
            else:
                f = None                               # C: send nothing (stream stalls)
            if f is not None:
                await ws.send(json.dumps(f))
            await asyncio.sleep(1.0 / 30.0)
        rt.cancel()

    if not samples:
        print("NO robot_state received"); return
    arr = np.array([s[2] for s in samples])
    phs = np.array([s[1] for s in samples])
    rest = arr[0].copy()
    dev = np.linalg.norm(arr - rest, axis=1)

    def seg(p):
        return dev[phs == p]
    a, b, c = seg("A"), seg("B"), seg("C")
    print(f"samples: {len(samples)}")
    # A: follows
    a_max = a.max() if len(a) else 0
    a_collapse = sum(1 for i in range(1, len(a)) if a[i-1] > 0.1 and a[i] < 0.04)
    print(f"[A hands]   follow max-dev {a_max:.2f} rad | collapses-to-rest {a_collapse}")
    # B: holds (deviation should stay near where A left off, NOT collapse to rest)
    if len(b):
        b_end_ref = a[-1] if len(a) else 0.0
        b_drift = abs(b[-1] - b[0])
        b_to_rest = b.min()
        print(f"[B no-hands] HOLD: dev start {b[0]:.2f} end {b[-1]:.2f} (drift {b_drift:.2f}) min {b_to_rest:.2f}")
    # C: eases to home and settles (low, stable)
    if len(c):
        c_settle = c[-5:].std() if len(c) >= 5 else 0.0
        print(f"[C headset off] HOME ease: dev end {c[-1]:.2f}, settled (std last5 {c_settle:.3f})")
    follows = a_max > 0.1 and a_collapse == 0
    holds = (len(b) == 0) or (b.min() > 0.3 * (a[-1] if len(a) else 1.0))   # didn't drop to rest
    print("VERDICT:", "FOLLOWS + HOLDS (no resetting)" if (follows and holds)
          else f"PROBLEM follows={follows} holds={holds}")


asyncio.run(main())
