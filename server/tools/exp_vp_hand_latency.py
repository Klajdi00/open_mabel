"""VISION PRO HAND LATENCY — over the LIVE wire, with BOTH hands moving.

THE OPERATOR REPORT (2026-07-31)
--------------------------------
  "severe delay ... either the left or the right hand is working, both cannot
   work together"

`controller/experiments/exp_vp_bimanual_latency.py` measured this IN PROCESS
against a synthetic operator and pinned the cause on the body-frame estimator
reading a two-handed lateral sweep as a torso turn (20.2 deg of phantom yaw,
each arm tracking the OTHER hand at r = -0.84). That harness drives the engine
directly. This one drives the REAL SERVER over the REAL WIRE, so it also
includes the websocket, the gateway's driver arbitration and pose parse, the
control loop's tick, and the sim's servo — everything between a headset and the
metal except ARKit itself.

WHY IT HAS TO BE A SEPARATE HARNESS FROM exp_retarget_latency.py: that one's
`frame()` mirrors ONE wrist offset onto both hands, so the two hands always move
together. A symmetric motion cannot detect cross-coupling — it is exactly the
case where a coupled and an uncoupled arm look identical. Here each side gets
its OWN frequency, so an arm being driven by the wrong hand cannot score well.

WHAT IT MEASURES
----------------
  BIMANUAL   both hands sweeping, per side: phase lag (lock-in demodulation at
             that side's own frequency) and tracking correlation, plus the CROSS
             correlation of each arm against the OTHER hand.
  SOLO       the same, one hand at a time — the control. If bimanual matches
             solo, the two hands are independent, which is the whole claim.

Reported lag is END-TO-END over the wire and therefore strictly larger than the
engine-only number; what matters is that it is the SAME on both sides and that
cross-coupling is near zero.

Usage
-----
    server/run.sh --no-studio                     # or any live sim bridge
    python3 server/tools/exp_vp_hand_latency.py
"""
import asyncio
import json
import math
import os
import sys
import time

import numpy as np
import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from exp_arm_fidelity import FK, _mat, _rot, hand_joints, EYE   # noqa: E402

PORT = os.environ.get("MABEL_WS_PORT", "9090")
URI = f"ws://127.0.0.1:{PORT}/teleop?client=visionpro"

RATE_HZ = 30.0                     # the app's send cadence
SECS = 14.0                        # long enough for ~5 cycles of the slower side
SETTLE = 3.0                       # let the retargeter anchor before measuring
AMP = 0.13                         # m, half-amplitude of each hand's sweep
# DIFFERENT per side on purpose: a side driven by the wrong hand cannot fake a
# good score, and the lag can be demodulated at a known frequency instead of
# being scanned for (a scan locks onto the frame's axis flip — see the doc).
HZ = {"left": 0.35, "right": 0.55}
HOME = {"right": np.array([0.26, -0.35, -0.34]),
        "left": np.array([-0.26, -0.35, -0.34])}      # head-relative, ARKit axes


def wrist_of(side, t, moving=True):
    p = HOME[side].copy()
    if moving:
        p[0] += AMP * math.sin(2 * math.pi * HZ[side] * t)
        p[1] += 0.07 * math.sin(2 * math.pi * HZ[side] * t + 1.0)
    return p


def frame(seq, t, moving):
    """teleop_frame with INDEPENDENT per-side wrists (the wire's own schema)."""
    head_t = np.array([0.0, EYE, 0.0])
    payload = {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": _mat(np.eye(3), head_t).reshape(16).tolist()},
                 "trackingState": "normal"},
        "mode": "Whole Body",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
    }
    for side in ("left", "right"):
        W = head_t + wrist_of(side, t, moving=moving.get(side, True))
        payload[f"{side}Hand"] = {
            "chirality": side,
            "anchorTransform": {"matrix": _mat(np.eye(3), W).reshape(16).tolist()},
            "joints": hand_joints(side, 0.0), "isTracked": True,
        }
    return {"type": "teleop_frame", "payload": payload}


def lockin_lag(t, op, rob, f):
    """Phase lag of `rob` behind `op` [s] at the KNOWN frequency f."""
    t = np.asarray(t, float); a = np.asarray(op, float); b = np.asarray(rob, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 32:
        return float("nan")
    t, a, b = t[m], a[m] - a[m].mean(), b[m] - b[m].mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    if float(np.corrcoef(a, b)[0, 1]) < 0:        # the frame inverts one axis
        b = -b
    ref = np.exp(-2j * np.pi * f * t)
    A, B = np.sum(a * ref), np.sum(b * ref)
    if abs(A) < 1e-12 or abs(B) < 1e-12:
        return float("nan")
    ph = np.angle(B / A)
    if ph > np.pi / 2:
        ph -= 2 * np.pi
    return float(-ph / (2 * np.pi * f))


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 32 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def principal(x3):
    """Scalar progress along a trajectory's own dominant axis — the operator and
    robot signals live in different frames, so like-named COMPONENTS are not
    comparable (it made a tracked arm look anti-correlated)."""
    x = np.asarray(x3, float)
    m = np.all(np.isfinite(x), axis=1)
    out = np.full(len(x), np.nan)
    if m.sum() < 8:
        return out
    c = x[m] - x[m].mean(axis=0)
    _, _, V = np.linalg.svd(c, full_matrices=False)
    out[m] = c @ V[0]
    return out


async def reader(ws, state):
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") == "robot_state":
            j = (msg.get("payload") or {}).get("jointPositions")
            if j:
                state["joints"] = j


async def run_case(ws, fk, label, moving):
    rec = {"t": [], "op": {"left": [], "right": []}, "rob": {"left": [], "right": []}}
    state = {"joints": None}
    rtask = asyncio.create_task(reader(ws, state))
    seq = 0
    t0 = time.time()
    try:
        while True:
            el = time.time() - t0
            if el >= SETTLE + SECS:
                break
            seq += 1
            await ws.send(json.dumps(frame(seq, el, moving)))
            if state["joints"] and el >= SETTLE:
                p = fk.pose(dict(state["joints"]))
                rec["t"].append(el - SETTLE)
                for s in ("left", "right"):
                    rec["op"][s].append(wrist_of(s, el, moving.get(s, True)).copy())
                    rec["rob"][s].append(p.wrist(s))
            await asyncio.sleep(1.0 / RATE_HZ)
    finally:
        rtask.cancel()

    t = np.asarray(rec["t"], float)
    out = {}
    if len(t) < 40:
        print(f"  {label:12s} NO DATA (is the bridge streaming robot_state?)")
        return out
    op = {s: principal(np.asarray(rec["op"][s], float)) for s in ("left", "right")}
    rob = {s: principal(np.asarray(rec["rob"][s], float)) for s in ("left", "right")}
    for s in ("right", "left"):
        if not moving.get(s, True):
            continue
        other = "left" if s == "right" else "right"
        out[s] = {
            "lag_ms": lockin_lag(t, op[s], rob[s], HZ[s]) * 1e3,
            "own": abs(corr(op[s], rob[s])),
            "cross": abs(corr(op[other], rob[s])) if moving.get(other, True) else float("nan"),
            "travel_mm": float(np.nanmax(rob[s]) - np.nanmin(rob[s])) * 1e3,
        }
        r = out[s]
        print(f"  {label:12s} {s:5s}  lag {r['lag_ms']:6.1f} ms   own r {r['own']:.3f}   "
              f"cross r {r['cross']:.3f}   travel {r['travel_mm']:5.1f} mm")
    return out


async def main():
    fk = FK()
    print(f"\nVISION PRO HAND LATENCY over the live wire — {URI}")
    print(f"stream {RATE_HZ:.0f} Hz · right {HZ['right']} Hz · left {HZ['left']} Hz · "
          f"{SECS:.0f} s per case\n")
    async with websockets.connect(URI, max_size=2 ** 20) as ws:
        # Energise + put the server in whole-body retargeting, exactly as the app does.
        await ws.send(json.dumps({"type": "estop", "payload": {"active": False}}))
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "retarget", "controlType": "impedance",
            "region": "whole_body", "stiffness": 22.0,
            "tracking": {"neck": True, "arms": True, "hands": True}}}))
        await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))
        await asyncio.sleep(0.5)

        res = {}
        res["both"] = await run_case(ws, fk, "BOTH hands", {"left": True, "right": True})
        res["right"] = await run_case(ws, fk, "RIGHT only", {"left": False, "right": True})
        res["left"] = await run_case(ws, fk, "LEFT only", {"left": False, "right": True} |
                                     {"left": True, "right": False})

        print("\nVERDICT")
        ok = True
        for s in ("right", "left"):
            b = res["both"].get(s)
            solo = res[s].get(s)
            if not b or not solo:
                print(f"  {s}: incomplete"); ok = False; continue
            d = abs(b["lag_ms"] - solo["lag_ms"])
            print(f"  {s:5s}  bimanual {b['lag_ms']:6.1f} ms vs solo {solo['lag_ms']:6.1f} ms "
                  f"(Δ {d:5.1f})   cross r {b['cross']:.3f}")
            if not np.isfinite(b["own"]) or not np.isfinite(b["lag_ms"]):
                # A harness that PASSES on no data is worse than no harness: the
                # first run against a wedged bridge returned all-NaN and still
                # printed PASS, because every comparison against NaN is False.
                print("         FAIL no usable data (robot never moved — is the "
                      "sim healthy and are you the active driver?)")
                ok = False
                continue
            if b["own"] < 0.7:
                print(f"         FAIL own-hand tracking {b['own']:.3f} < 0.70"); ok = False
            if b["cross"] > b["own"]:
                print(f"         FAIL tracks the OTHER hand better than its own"); ok = False
        print("\n  " + ("PASS — both hands track independently over the wire"
                        if ok else "FAIL — see above"))
    print()


if __name__ == "__main__":
    asyncio.run(main())
