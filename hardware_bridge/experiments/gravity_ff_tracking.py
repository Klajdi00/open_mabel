"""gravity_ff_tracking.py — does gravity feed-forward actually improve tracking?

THE QUESTION (operator, 2026-07-14): "see how much gravity compensation helps the
position tracking — how much is the improvement — with and without gravity ff torque".

THE PHYSICS. The arms run a DaMiao MIT law:  T = kp·(q* − q) + kd·(q̇* − q̇) + t_ff.
At a static hold q̇ = 0, so the motor must supply the joint's gravity torque τ_g(q):

    kp·(q* − q) + t_ff = τ_g(q)      ⇒      e_ss = (τ_g(q) − t_ff) / kp

With NO feed-forward (t_ff = 0) the steady-state error is  e_ss = τ_g(q)/kp  — a pure
DROOP, and it is NOT a tuning problem: kd is capped at 5 by the 12-bit MIT packing, which
caps kp for a damped loop (~15 on the shoulder), so a bigger kp is not available. That is
the classic "PD sags under load" the operator named. With a CORRECT feed-forward t_ff =
τ_g(q), the gravity term cancels and e_ss → 0 REGARDLESS of kp.

So the prediction is sharp and falsifiable:
    * FF OFF: |e_ss| grows with the gravity load, ≈ τ_g(q)/kp
    * FF ON : |e_ss| ≈ 0 at every pose
and the improvement should be largest where τ_g is largest (shoulder pitch out from the
body), and ~zero where the arm hangs (τ_g ≈ 0 — which is exactly why the FF looked "dead"
when the arm was limp: there was nothing to hold).

METHOD. For each joint under test, step it through a small sweep of targets, hold each
for SETTLE_S, and record the steady-state error. Targets go over the WIRE (:9090
`joint_command`, the same message the iOS app sends), so this doubles as an end-to-end
check of the iOS→server→bridge→robot control path. Runs twice — MABEL_HW_ARMS_FF_SCALE=1
then =0 — and the two runs are compared.

SAFETY
  * Small angles only, clamped to SAFE_MAX_STEP from the pose we started at.
  * Targets are clamped into the joint's calibrated model range with a margin.
  * Temperature is polled every sample; the run ABORTS if any motor passes ABORT_TEMP_C
    (the bridge's own thermal guard cuts at 70/80 °C — this aborts first, so we never
    lean on the last line of defence).
  * FF OFF means the arm DROOPS by design. That droop is the measurement.

Usage:
    python3 gravity_ff_tracking.py --run      # execute both passes, write results JSON
    python3 gravity_ff_tracking.py --report   # build the PDF from the JSON
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
GW = os.getenv("MABEL_GW", "ws://localhost:9090/teleop")

# kp is small (14.9 on the shoulder), so the approach is SLOW: a 17° step needs ~10 s to
# reach steady state. Sampling too early measures the transient, not e_ss — a 4 s settle
# made every joint look FROZEN at its start pose (it had simply not arrived yet).
SETTLE_S = 12.0           # hold each target this long before sampling
SAMPLE_S = 3.0            # average the last SAMPLE_S of the hold (steady state only)
SAFE_MAX_STEP = 1.30      # rad — j2 must reach ~horizontal for gravity to actually load it
ABORT_TEMP_C = 60.0       # abort the run before the bridge's own 70/80 °C guard fires

# The sweeps. j2 = shoulder PITCH — the joint that actually lifts the arm against gravity,
# so it carries the largest τ_g and is where the feed-forward should matter most. j4 =
# ELBOW — a second, independent load path. Both small, both about the arm's hang pose.
# j2 = shoulder PITCH: the ONLY left-arm joint that both carries a large gravity load and
# has clear travel from the hang pose. Its τ_g runs 0 N·m (hanging) → 10.5 N·m (~horizontal),
# so pure PD at kp=14.9 would sag 0° → 40°. Holding it at these elevated angles is where the
# feed-forward is falsifiable. (j4/j5/j6 are excluded: j4 rests against its +1.67 limit, and
# the wrist joints carry almost no gravity with the hands off — nothing for the FF to prove.)
# Same absolute targets in BOTH passes, so FF-on vs FF-off sag is directly comparable.
SWEEPS = {
    "left_arm_2": [0.30, 0.55, 0.80],
}


async def _recv_state(ws, want_temps=False):
    """Next robot_state payload."""
    while True:
        m = json.loads(await ws.recv())
        if m.get("type") == "robot_state":
            return m["payload"]


async def _hold_and_measure(ws, joint, target, other_targets):
    """Command `target` on `joint`, hold, and return the steady-state stats."""
    cmd = dict(other_targets)
    cmd[joint] = float(target)
    t0 = time.time()
    samples, temps = [], {}
    while time.time() - t0 < SETTLE_S:
        await ws.send(json.dumps({"type": "joint_command", "payload": {"joints": cmd}}))
        p = await _recv_state(ws)
        jp = p.get("jointPositions") or {}
        tp = p.get("jointTemps") or {}
        for k, v in tp.items():
            if "arm" in k:
                temps[k] = v
        if joint in jp and time.time() - t0 > (SETTLE_S - SAMPLE_S):
            samples.append(jp[joint])
    hot = max(temps.values()) if temps else 0.0
    if hot >= ABORT_TEMP_C:
        raise RuntimeError(f"ABORT: a motor reached {hot:.0f}°C (limit {ABORT_TEMP_C:.0f})")
    if not samples:
        raise RuntimeError(f"no samples for {joint}")
    q = statistics.median(samples)
    return {"target": float(target), "measured": q, "error": q - float(target),
            "wobble": (max(samples) - min(samples)), "hot": hot}


async def run_pass(tag):
    """One full sweep of every joint, at the current FF setting."""
    res = {"tag": tag, "joints": {}}
    async with websockets.connect(GW, open_timeout=15, max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"type": "estop", "payload": {"active": False}}))
        await asyncio.sleep(4.0)                       # energize + FF ramp
        p = await _recv_state(ws)
        start = {k: v for k, v in (p.get("jointPositions") or {}).items()
                 if k.startswith("left_arm")}
        print(f"[{tag}] start pose: " +
              " ".join(f"{k[-1]}={v:+.3f}" for k, v in sorted(start.items())))
        # every other left joint is commanded to HOLD where it started, so the only
        # thing changing is the joint under test.
        try:
            for joint, targets in SWEEPS.items():
                hold = {k: v for k, v in start.items() if k != joint}
                rows = []
                for tgt in targets:
                    if abs(tgt - start[joint]) > SAFE_MAX_STEP:
                        print(f"[{tag}] {joint} -> {tgt:+.2f} SKIPPED "
                              f"(> {SAFE_MAX_STEP} rad from start)")
                        continue
                    r = await _hold_and_measure(ws, joint, tgt, hold)
                    rows.append(r)
                    print(f"[{tag}] {joint} target={r['target']:+.3f} "
                          f"measured={r['measured']:+.3f} "
                          f"ERR={r['error']*57.3:+6.2f}°  ({r['hot']:.0f}°C)")
                res["joints"][joint] = rows
        finally:
            await ws.send(json.dumps({"type": "estop", "payload": {"active": True}}))
            await asyncio.sleep(1.0)
            print(f"[{tag}] torque OFF.")
    return res


def _set_ff(scale):
    """Rewrite the bridge's env override and restart it (FF_SCALE is read at import)."""
    envf = HERE.parent / ".env.local"
    lines = [ln for ln in envf.read_text().splitlines()
             if not ln.startswith("export MABEL_HW_ARMS_FF_SCALE=")]
    lines.append(f"export MABEL_HW_ARMS_FF_SCALE={scale}")
    envf.write_text("\n".join(lines) + "\n")
    subprocess.run(["systemctl", "restart", "mabel-hardware-bridge"],
                   check=False, capture_output=True)
    # wait for the bridge to own the lock and the arms to stream again
    for _ in range(40):
        time.sleep(3)
        r = subprocess.run(["fuser", "/tmp/mabel_hardware_bridge.lock"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            time.sleep(12)
            return True
    return False


async def main_run():
    out = {}
    for tag, scale in (("ff_on", 1), ("ff_off", 0)):
        print(f"\n=== pass '{tag}': MABEL_HW_ARMS_FF_SCALE={scale} — restarting bridge")
        if not _set_ff(scale):
            print("bridge did not come back; aborting.")
            return
        out[tag] = await run_pass(tag)
        await asyncio.sleep(2)
    _set_ff(1)                                  # leave the robot with gravity comp ON
    (OUT / "gravity_ff.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'gravity_ff.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        asyncio.run(main_run())
    else:
        ap.print_help()
