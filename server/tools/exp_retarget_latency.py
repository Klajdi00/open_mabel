"""RETARGET LATENCY — how far behind the operator is the robot's arm?

THE OPERATOR REPORT (2026-07-28)
--------------------------------
  "the teleop whole body re-targeting is way too slow — I need it to be fast
   and more real time for me."

"Slow" is two different defects and they need different fixes, so this harness
separates them:

  A. TRANSPORT / TICK LAG — the robot starts moving late (dead time).
  B. FOLLOWING LAG        — the robot starts on time but crawls, because a
                            per-frame rate limit (ARM_TRUST_STEP) or a temporal
                            smoothness weight (ARM_W_SMOOTH) holds it back.

Both show up as "laggy" to the person wearing the headset; only (B) gets worse
with the SIZE of the motion, which is how they are told apart here.

WHAT IT MEASURES (all over the live wire, FK'd from streamed joint angles)
-------------------------------------------------------------------------
  1. TICK RATE       the effective retarget rate — measured from the arm's own
                     saturated speed. On a step far larger than one trust step,
                     the arm moves at exactly ARM_TRUST_STEP rad per RETARGET
                     TICK, so peak |dq|/dt IS trust_step x tick_hz. This reads
                     the bridge's real retarget cadence without instrumenting it.
  2. STEP RESPONSE   dead time, t63 and t90 of the robot wrist for a 25 cm
                     operator hand step. Dead time is (A); t63-dead is (B).
  3. PHASE LAG       the operator's hand oscillates at TRACK_HZ; cross-correlate
                     operator vs robot wrist to get the steady-state following
                     delay in ms. This is the number the operator actually
                     perceives while teleoperating, because teleop is continuous
                     motion, not steps.

Usage
-----
    server/run.sh --no-studio            # in another terminal
    python3 server/tools/exp_retarget_latency.py [--plot]

Environment knobs under test (set them on the SERVER, then re-run):
    MABEL_ARM_TRUST_STEP   per-tick joint step, rad   (default 0.10)
    MABEL_ARM_W_SMOOTH     temporal smoothness weight (default 2.0)
    MABEL_ARM_GN_ITERS     Gauss-Newton iterations    (default 8)
"""
import argparse
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
from exp_arm_fidelity import FK, Rig, frame, EYE   # noqa: E402

PORT = os.environ.get("MABEL_WS_PORT", "9090")
URI = f"ws://127.0.0.1:{PORT}/teleop?client=visionpro"

RATE_HZ = 30.0             # the app's transmissionRateHz — stream at what it streams
TRACK_HZ = 0.40            # operator oscillation for the phase-lag test
TRACK_AMP = 0.18           # m, half-amplitude of the hand sweep
TRACK_SECS = 12.0

# Budgets. These are what "real time" means for a person: 100 ms of end-to-end
# delay is the classic teleoperation threshold at which an operator stops
# feeling coupled to the remote arm and starts open-loop over-correcting.
DEAD_BUDGET = 0.120        # s
T63_BUDGET = 0.350         # s
PHASE_BUDGET = 0.180       # s

ARM_JOINTS = [f"right_arm_{i}" for i in range(1, 8)]


class Trace:
    """Timestamped log of streamed joint dicts, so every metric below is
    measured against wall-clock arrival rather than against send count."""

    def __init__(self):
        self.t = []
        self.q = []          # right-arm joint vectors
        self.p = []          # robot right-wrist position (FK)

    def add(self, t, joints, fk):
        q = np.array([float(joints.get(j, 0.0)) for j in ARM_JOINTS])
        self.t.append(t)
        self.q.append(q)
        self.p.append(fk.pose(joints).wrist("right"))

    def arr(self):
        return np.asarray(self.t), np.asarray(self.q), np.asarray(self.p)


async def stream(rig, trace, fk, secs, pose_fn):
    """Stream `pose_fn(elapsed) -> wrist_local` and trace the robot's reply.

    The trace samples on the ROBOT_STATE that `rig`'s reader has most recently
    latched, at send cadence — enough for 0.4 Hz phase and for a step whose
    interesting features last hundreds of ms.
    """
    t0 = time.time()
    while True:
        el = time.time() - t0
        if el >= secs:
            return
        rig.seq += 1
        await rig.ws.send(json.dumps(frame(rig.seq, pose_fn(el))))
        if rig.joints:
            trace.add(el, dict(rig.joints), fk)
        await asyncio.sleep(1.0 / RATE_HZ)


def _demean(a):
    """Deviation from the signal's own mean — the shape, without the offset."""
    return np.asarray(a, float) - float(np.mean(a))


def phase_lag(t, op, rob, max_lag=0.6):
    """Delay in seconds that best aligns the robot's motion with the operator's.

    Cross-correlation on a UNIFORM grid: the send loop is not perfectly
    isochronous, so both signals are resampled first. Correlating the raw
    unevenly-sampled arrays would fold the loop's own jitter into the answer.
    """
    grid = np.arange(t[0], t[-1], 1.0 / 200.0)
    a = _demean(np.interp(grid, t, op))
    b = _demean(np.interp(grid, t, rob))
    n = int(max_lag * 200)
    best, best_c = 0, -np.inf
    for k in range(0, n):
        # b delayed by k: compare b[k:] against a[:len-k]
        seg_a, seg_b = a[: len(a) - k], b[k:]
        d = np.linalg.norm(seg_a) * np.linalg.norm(seg_b)
        if d <= 0:
            continue
        c = float(np.dot(seg_a, seg_b) / d)
        if c > best_c:
            best_c, best = c, k
    return best / 200.0, best_c


def step_metrics(t, sig, t_step):
    """Dead time, t63 and t90 measured from the moment the operator stepped.

    `sig` is a scalar that moves monotonically from a settled start to a settled
    end. Start = mean of the 0.3 s before the step (not the single last sample:
    the arm has residual servo ripple); end = mean of the final 0.5 s.
    """
    t = np.asarray(t, float)
    sig = np.asarray(sig, float)
    pre = sig[(t > t_step - 0.3) & (t <= t_step)]
    post = sig[t > t[-1] - 0.5]
    s0, s1 = float(np.mean(pre)), float(np.mean(post))
    span = s1 - s0
    if abs(span) < 1e-4:
        return None
    # Dead time = first sample that has moved 5 % of the way. Below that the
    # signal is indistinguishable from the settled ripple.
    def cross(frac):
        for ti, v in zip(t, sig):
            if ti < t_step:
                continue
            if (v - s0) / span >= frac:
                return ti - t_step
        return float("nan")
    return {"span": span, "dead": cross(0.05),
            "t63": cross(0.63), "t90": cross(0.90)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    fk = FK()
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")

    async with websockets.connect(URI, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        rig = Rig(ws, fk)
        last = {"t": 0.0}

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "robot_state":
                    rig.joints.update(m["payload"].get("jointPositions", {}))
                    last["t"] = time.time()
        rtask = asyncio.create_task(reader())

        async def spec(mode="global"):
            await rig.send("control_mode", {
                "method": "retarget", "controlType": "impedance",
                "region": "whole_body", "stiffness": 0.0, "frame": mode,
                "tracking": {"neck": True, "arms": True, "hands": True}})

        await rig.send("reset", {})
        await asyncio.sleep(2.0)
        await rig.send("estop", {"active": False})
        await asyncio.sleep(0.4)
        await spec("global")
        await rig.send("recalibrate", {})
        await rig.hold(2.5, (0.30, -0.10, -0.60))     # calibration reach

        # ── 1. STEP RESPONSE + effective tick rate ──────────────────────────
        print("\n=== STEP: operator hand jumps 25 cm ===")
        A = (0.22, -0.05, -0.35)
        B = (0.22, -0.05, -0.60)
        await rig.hold(2.0, A)
        tr = Trace()
        T_STEP = 1.0
        await stream(rig, tr, fk, 5.0, lambda el: A if el < T_STEP else B)
        t, q, p = tr.arr()

        # Wrist travel along the step direction (−z of the operator body frame
        # is the robot's +x reach), taken as distance from the pre-step point.
        p0 = np.mean(p[(t > T_STEP - 0.3) & (t <= T_STEP)], axis=0)
        dist = np.linalg.norm(p - p0, axis=1)
        m = step_metrics(t, dist, T_STEP)
        if m is None:
            check("step: the arm moved at all", False, "wrist never left the start pose")
        else:
            print(f"    wrist travelled {m['span'] * 100:.1f} cm")
            check("step: dead time", m["dead"] <= DEAD_BUDGET,
                  f"{m['dead'] * 1000:.0f} ms (budget {DEAD_BUDGET * 1000:.0f})")
            check("step: t63 (63 % of travel)", m["t63"] <= T63_BUDGET,
                  f"{m['t63'] * 1000:.0f} ms (budget {T63_BUDGET * 1000:.0f})")
            print(f"    t90 = {m['t90'] * 1000:.0f} ms")

        # Effective retarget tick rate, read off the saturated joint speed.
        dt = np.diff(t)
        dq = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        good = dt > 1e-3
        speed = dq[good] / dt[good]
        peak = float(np.percentile(speed, 98)) if speed.size else 0.0
        trust = float(os.environ.get("MABEL_ARM_TRUST_STEP", "0.10"))
        tick_hz = peak / trust if trust > 0 else float("nan")
        print(f"    peak joint speed {peak:.2f} rad/s  ->  retarget tick "
              f"~{tick_hz:.0f} Hz at trust step {trust:.3f} rad")
        check("tick: retarget keeps up with the 30 Hz stream", tick_hz >= 20.0,
              f"{tick_hz:.0f} Hz effective (stream is {RATE_HZ:.0f} Hz)")

        # ── 2. PHASE LAG under continuous motion ────────────────────────────
        print(f"\n=== TRACKING: operator hand sweeps at {TRACK_HZ} Hz ===")

        def sweep(el):
            return (0.22, TRACK_AMP * math.sin(2 * math.pi * TRACK_HZ * el), -0.45)

        await rig.hold(1.5, sweep(0.0))
        tr2 = Trace()
        await stream(rig, tr2, fk, TRACK_SECS, sweep)
        t2, _, p2 = tr2.arr()
        # Drop the first cycle: the arm is still converging onto the orbit.
        keep = t2 > 1.0 / TRACK_HZ
        t2, p2 = t2[keep], p2[keep]
        op = np.array([sweep(e)[1] for e in t2])       # operator hand height
        rob = p2[:, 2]                                  # robot wrist height
        lag, corr = phase_lag(t2, op, rob)
        amp_op = float(np.ptp(op))
        amp_rob = float(np.ptp(rob))
        print(f"    correlation {corr:.3f}   operator sweep {amp_op * 100:.1f} cm "
              f"-> robot {amp_rob * 100:.1f} cm ({amp_rob / amp_op:.2f}x)")
        check("tracking: following lag", lag <= PHASE_BUDGET,
              f"{lag * 1000:.0f} ms (budget {PHASE_BUDGET * 1000:.0f})")
        # A rate limiter does not just delay — it ATTENUATES, and attenuation is
        # what makes a big gesture feel dead. Report it as its own line.
        check("tracking: amplitude is not squashed", amp_rob / amp_op >= 0.55,
              f"robot follows {amp_rob / amp_op * 100:.0f} % of the sweep")

        rtask.cancel()

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.4))
        ax[0].plot(t, dist * 100, lw=1.4)
        ax[0].axvline(T_STEP, color="0.6", ls="--", lw=0.9)
        if m:
            for key, c in (("dead", "tab:orange"), ("t63", "tab:green"),
                           ("t90", "tab:blue")):
                ax[0].axvline(T_STEP + m[key], color=c, lw=0.9,
                              label=f"{key} {m[key] * 1000:.0f} ms")
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("wrist travel (cm)")
        ax[0].set_title("Step response to a 25 cm operator hand jump")
        ax[0].legend(fontsize=7)
        ax[1].plot(t2, (op - np.mean(op)) * 100, lw=1.2, label="operator hand")
        ax[1].plot(t2, (rob - np.mean(rob)) * 100, lw=1.2,
                   label=f"robot wrist (lag {lag * 1000:.0f} ms)")
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("height about mean (cm)")
        ax[1].set_title(f"Continuous tracking at {TRACK_HZ} Hz")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(_ROOT, "controller", "docs", "figures",
                           "retarget_latency.pdf")
        fig.savefig(out)
        print(f"\n  figure -> {out}")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
