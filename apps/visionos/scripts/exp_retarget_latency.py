#!/usr/bin/env python3
"""Retargeting-latency harness for the Vision Pro V2 teleop path.

Emulates the app's wire behaviour exactly (60 Hz `teleop_frame` with a full
right-hand skeleton, "Arms & Hands" mode) against a LOCAL sim bridge
(ws://127.0.0.1:9090/teleop) and measures, from the wire's point of view:

  1. robot_state arrival period + jitter        (state-stream health)
  2. sinusoid phase lag: wrist x(t) = A sin(wt)  ->  best-matching right-arm
     joint response, cross-correlated on a common 100 Hz grid  (end-to-end
     wire -> retarget -> IK -> state latency, EXCLUDING ARKit + rendering)
  3. step response: +12 cm wrist step -> time until the joint moves 10 %
     of its final delta                          (first-motion latency)

Run:  server/start_teleop.sh   (plain, no --mock), then
      python3 VisionOS_V2/scripts/exp_retarget_latency.py
"""
import asyncio, json, math, time, statistics, sys

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

URI = "ws://127.0.0.1:9090/teleop"
SEND_HZ = 60.0
ANCHOR_S, SINE_S, STEP_S = 2.5, 12.0, 3.0
AMP, FREQ = 0.10, 0.4          # m, Hz
STEP = 0.12                    # m

NAMES_21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]

def mat(tx=0.0, ty=0.0, tz=0.0):
    # ROW-major 4x4 (Transform4x4 wire layout: translation at indices 3,7,11)
    return [1,0,0,tx, 0,1,0,ty, 0,0,1,tz, 0,0,0,1]

def hand_payload(x_off, pinched=False):
    joints = []
    # wrist at local origin. The SE(3) clutch engages on PINCH: thumb tip and
    # index tip meet (<1 cm) to clutch in — held closed through the whole sweep.
    if pinched:
        offs = {"wrist": (0,0,0), "thumb_tip": (-0.015,0,-0.085), "index_tip": (-0.02,0,-0.09),
                "middle_tip": (0,0,-0.10), "ring_tip": (0.02,0,-0.09), "little_tip": (0.04,0,-0.08)}
    else:
        offs = {"wrist": (0,0,0), "thumb_tip": (0.05,0,-0.05), "index_tip": (-0.02,0,-0.09),
                "middle_tip": (0,0,-0.10), "ring_tip": (0.02,0,-0.09), "little_tip": (0.04,0,-0.08)}
    for n in NAMES_21:
        o = offs.get(n, (0.01, 0, -0.05))
        joints.append({"joint": n, "isTracked": True,
                       "localTransform": {"matrix": mat(*o)}})
    return {"isTracked": True,
            "anchorTransform": {"matrix": mat(0.35 + x_off, 1.05, -0.45)},
            "joints": joints}

def frame(seq, x_off, pinched):
    return json.dumps({"type": "teleop_frame", "payload": {
        "sequence": seq,
        "timestamp": time.time(),
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0},
        "head": {"transform": {"matrix": mat(0, 1.5, 0)}, "trackingState": "normal"},
        "rightHand": hand_payload(x_off, pinched),
    }})

async def main():
    cmd, states, periods = [], [], []
    async with websockets.connect(URI, max_size=None) as ws:
        stop = asyncio.Event()

        async def rx():
            last = None
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                t = time.time()
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if d.get("type") != "robot_state":
                    continue
                if last is not None:
                    periods.append((t - last) * 1000)
                last = t
                j = d.get("payload", {}).get("jointPositions", {}) or d.get("payload", {}).get("joints", {})
                arm = {k: v for k, v in j.items() if k.startswith("right_arm")}
                if arm:
                    states.append((t, arm))

        rxt = asyncio.create_task(rx())
        # The app's body-tracking spec: ENGINE retargeting = method "retarget"
        # region "wholeBody" ("navigation" drives the base only; "joint"/"wrist"
        # switch the arms to the WBC slider law — no hand tracking there).
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "retarget", "controlType": "impedance",
            "region": "wholeBody", "stiffness": 300.0}}))
        t0 = time.time()
        seq = 0
        phase = "anchor"
        anchored = False
        print("anchoring clutch…")
        while True:
            now = time.time() - t0
            if now < ANCHOR_S:
                x = 0.0
            elif now < ANCHOR_S + SINE_S:
                if phase == "anchor":
                    phase = "sine"; print("sine sweep…")
                x = AMP * math.sin(2 * math.pi * FREQ * (now - ANCHOR_S))
            elif now < ANCHOR_S + SINE_S + 1.0:
                x = 0.0
            elif now < ANCHOR_S + SINE_S + 1.0 + STEP_S:
                if phase == "sine":
                    phase = "step"; step_t = time.time(); print("step…")
                x = STEP
            else:
                break
            if not anchored and now >= ANCHOR_S * 0.6:
                # the bridge anchors the SE(3) clutch on `recalibrate` while a
                # tracked hand is streaming — exactly what the app's Start /
                # re-center does.
                await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))
                anchored = True
            # The engine's clutch is a DOUBLE-PINCH toggle: two quick pinches
            # during the anchor phase engage tracking, hand stays open after.
            if 1.2 <= now < 1.4 or 1.6 <= now < 1.8:
                pinch = True
            else:
                pinch = False
            seq += 1
            await ws.send(frame(seq, x, pinched=pinch))
            cmd.append((time.time(), x))
            await asyncio.sleep(1.0 / SEND_HZ)
        stop.set()
        await asyncio.sleep(0.3)
        rxt.cancel()

    # ── analysis ─────────────────────────────────────────────────────────
    print(f"\nstates: {len(states)}  cmd frames: {len(cmd)}")
    if periods:
        print(f"robot_state period: mean {statistics.mean(periods):.1f} ms  "
              f"p95 {sorted(periods)[int(0.95*len(periods))]:.1f} ms  "
              f"max {max(periods):.1f} ms")
    if not states:
        sys.exit("no right_arm joints in robot_state — is the sim bridge running (not --mock)?")

    keys = sorted(states[-1][1].keys())
    t_s0, t_s1 = ANCHOR_S + 1.0, ANCHOR_S + SINE_S     # analysis window (rel t0)
    tref = cmd[0][0]
    # resample both signals to a 100 Hz grid
    grid = [t_s0 + i / 100.0 for i in range(int((t_s1 - t_s0) * 100))]
    def resample(series):
        out, i = [], 0
        for g in grid:
            tg = tref + g
            while i + 1 < len(series) and series[i + 1][0] <= tg:
                i += 1
            out.append(series[i][1])
        return out
    cx = resample([(t, x) for t, x in cmd])
    best = None
    for k in keys:
        sig = resample([(t, a.get(k, 0.0)) for t, a in states])
        if statistics.pstdev(sig) < 1e-4:
            continue
        mu_c, sd_c = statistics.mean(cx), statistics.pstdev(cx)
        mu_s, sd_s = statistics.mean(sig), statistics.pstdev(sig)
        zc = [(v - mu_c) / sd_c for v in cx]
        zs = [(v - mu_s) / sd_s for v in sig]
        for lag in range(0, 60):                     # 0..600 ms
            n = len(zc) - lag
            r = sum(zc[i] * zs[i + lag] for i in range(n)) / n
            if best is None or abs(r) > abs(best[2]):
                best = (k, lag * 10.0, r)
    if best:
        print(f"phase lag: {best[1]:.0f} ms  (joint {best[0]}, |r|={abs(best[2]):.2f})")

    # step first-motion
    step_cmd_t = next(t for t, x in cmd if x >= STEP * 0.99)
    k = best[0] if best else keys[0]
    pre = [a.get(k, 0.0) for t, a in states if step_cmd_t - 0.5 <= t < step_cmd_t]
    post = [(t, a.get(k, 0.0)) for t, a in states if t >= step_cmd_t]
    if pre and post:
        base = statistics.mean(pre)
        final = statistics.mean([v for _, v in post[-30:]])
        thresh = base + 0.10 * (final - base)
        rising = final > base
        for t, v in post:
            if (v > thresh) == rising and abs(v - base) > 0.1 * abs(final - base):
                print(f"step first-motion: {(t - step_cmd_t) * 1000:.0f} ms  "
                      f"({k}: {base:+.3f} → {final:+.3f} rad)")
                break

asyncio.run(main())
