"""Calibration-capture robustness sweep: frame RATE x wrist TREMOR.

The 1-pose reach capture (teleop/engine.py `_calibrate_step`) requires the
head->wrist reach to sit inside a stability window before it locks REACH_SCALE.
This experiment streams synthetic arms-forward frames over the real wire
(ws://:9090/teleop, Vision Pro dialect) while sweeping

  * frame rate      [Hz]  — the app's effective teleop_frame rate (a loaded
                            main thread / slow uplink degrades the nominal 60)
  * tremor sigma    [m]   — Gaussian wrist-position jitter per axis (real
                            operator tremor + ARKit wrist noise)

and measures the time for the server to report calibrationStatus.done=true
(timeout = FAIL). It regenerates docs/figures/calibration_robustness.pdf.

Run:  python3 tools/exp_calibration_robustness.py            # needs a local bridge
      python3 tools/exp_calibration_robustness.py --quick    # 1 trial/cell
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "vision_os_apps", "docs", "tools"))
from exp_hand_retarget import synth_hand   # noqa: E402

IDENT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
TIMEOUT_S = 12.0          # generous: the capture window itself is <1 s
RATES_HZ = [60.0, 30.0, 15.0, 8.0, 5.0, 3.0]
SIGMAS_M = [0.000, 0.005, 0.010, 0.020]
SEED = 0


def _hand_payload(chir, x, jitter):
    joints = []
    for name, T in synth_hand(mcp=0.05, pip=0.05, thumb=0.1).items():
        joints.append({"joint": name,
                       "localTransform": {"matrix": np.asarray(T, float).reshape(16).tolist()},
                       "isTracked": True})
    anchor = [1.0, 0, 0, x + jitter[0], 0, 1, 0, -0.15 + jitter[1],
              0, 0, 1, -0.68 + jitter[2], 0, 0, 0, 1]
    return {"chirality": chir, "anchorTransform": {"matrix": anchor},
            "joints": joints, "isTracked": True}


def _frame(seq, sigma, rng):
    j = rng.normal(0.0, sigma, 3) if sigma > 0 else np.zeros(3)
    return {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": IDENT}, "trackingState": "normal"},
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
        "leftHand": _hand_payload("left", -0.22, j),
        "rightHand": _hand_payload("right", 0.22, j)}}


WS_PORT = int(os.environ.get("MABEL_WS_PORT", "9090"))


async def _one_trial(rate_hz, sigma, rng):
    uri = f"ws://127.0.0.1:{WS_PORT}/teleop?client=visionpro"
    async with websockets.connect(uri, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello

        async def send(t, p):
            await ws.send(json.dumps({"type": t, "payload": p}))

        status = {}
        seen_active = [False]     # robot_state frames in flight can still carry the
                                  # PREVIOUS trial's done=true — only count a done
                                  # that follows a fresh active=true of THIS capture.

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "robot_state":
                    cs = msg["payload"].get("calibrationStatus")
                    if cs:
                        if cs.get("active"):
                            seen_active[0] = True
                        if seen_active[0]:
                            status.update(cs)

        rtask = asyncio.create_task(reader())
        await send("body_calibration", {"action": "start", "user": "sweep"})
        t0 = time.time()
        seq = 0
        done_at = None
        while time.time() - t0 < TIMEOUT_S:
            seq += 1
            await send("teleop_frame", _frame(seq, sigma, rng)["payload"])
            if status.get("done"):
                done_at = time.time() - t0
                break
            await asyncio.sleep(1.0 / rate_hz)
        await send("body_calibration", {"action": "cancel"})
        rtask.cancel()
        return done_at


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 trial per cell")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    trials = 1 if args.quick else args.trials

    rng = np.random.default_rng(SEED)
    results = np.full((len(SIGMAS_M), len(RATES_HZ)), np.nan)
    for i, sigma in enumerate(SIGMAS_M):
        for k, rate in enumerate(RATES_HZ):
            times = []
            for _ in range(trials):
                t = await _one_trial(rate, sigma, rng)
                times.append(t if t is not None else np.nan)
            ok = [t for t in times if not np.isnan(t)]
            results[i, k] = float(np.mean(ok)) if len(ok) == len(times) else np.nan
            cell = f"{results[i, k]:.2f}s" if not np.isnan(results[i, k]) else "FAIL"
            print(f"  rate {rate:5.1f} Hz  sigma {sigma * 1000:4.1f} mm -> {cell}"
                  f"   ({sum(1 for t in times if not np.isnan(t))}/{len(times)} ok)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, sigma in enumerate(SIGMAS_M):
        row = results[i]
        ax.plot(RATES_HZ, row, "o-", label=f"tremor σ = {sigma * 1000:.0f} mm")
        for k, v in enumerate(row):
            if np.isnan(v):
                ax.plot(RATES_HZ[k], TIMEOUT_S, "x", color=ax.lines[-1].get_color())
    ax.axhline(TIMEOUT_S, ls=":", lw=0.8, color="gray")
    ax.text(RATES_HZ[-1], TIMEOUT_S, " never completes", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("teleop_frame rate [Hz]")
    ax.set_ylabel("time to calibration done [s]")
    ax.set_xscale("log")
    ax.set_xticks(RATES_HZ)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    ax.set_title("1-pose reach capture vs frame rate and tremor")
    fig.tight_layout()
    out = os.path.join(_HERE, "..", "docs", "figures", "calibration_robustness.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"figure -> {os.path.normpath(out)}")

    n_fail = int(np.isnan(results).sum())
    print(f"SUMMARY: {n_fail}/{results.size} cells never complete "
          f"(rates {RATES_HZ} Hz x sigma {SIGMAS_M} m)")


if __name__ == "__main__":
    asyncio.run(main())
