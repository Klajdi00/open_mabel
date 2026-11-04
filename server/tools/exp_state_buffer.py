#!/usr/bin/env python3
"""
exp_state_buffer.py — prove WHERE the multi-second robot_state delay lives.

The apps (iOS v2 / VisionOS V2) showed the 3D model lagging the sim by 2–5 s
while video stayed live. exp_state_stream.py already showed the SERVER emits a
regular stream; this harness measures the other half: what a client sees as a
function of ITS OWN drain rate.

Mechanism under test: the bridge streams robot_state per client (100 Hz as of
2026-07). The apps drained the socket on the MAIN thread — one URLSession
receive completion per main-queue pass — so under UI load their drain rate sat
below the arrival rate. TCP then queues the overflow CLIENT-SIDE (kernel recv
buffer + URLSession's internal queue): a standing FIFO seconds deep. The
server cannot shed it (write_limit=8 KB only prevents SERVER-side queueing;
once the client's TCP window fills, ws.send() blocks and the send loop
coalesces — but the frames already in the client's buffer replay in order).

Two probes against a RUNNING bridge (same machine ⇒ same clock, so the new
additive `sentAt` field gives TRUE one-way staleness):

  • FAST reader — drains as fast as recv() returns.        Expected: ~0 ms.
  • SLOW reader — drains at --slow-hz (default 40).        Expected: staleness
    climbs and settles seconds deep = the exact app symptom.

Both run CONCURRENTLY, proving per-client isolation (a slow client never
degrades a fast one — same reason video stayed fine).

Passive: never sends teleop frames or joint commands (no driver claim).

Outputs a metrics table + staleness-vs-time figure →
server/docs/figures/state_buffer_<tag>.pdf

Run:  MABEL/server$ python3 tools/exp_state_buffer.py --secs 45
      python3 tools/exp_state_buffer.py --url ws://<host>:9090/teleop --slow-hz 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.normpath(os.path.join(_HERE, "..", "docs", "figures"))


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


async def reader(url: str, secs: float, drain_hz: float | None, label: str):
    """Read robot_state for `secs`; return (t_rel[], staleness_ms[], n_states, n_total).

    drain_hz=None ⇒ drain at wire speed. Otherwise sleep to cap the READ rate —
    the Python analogue of an app main thread that only services the socket at
    drain_hz. max_queue=1 keeps the websockets lib from hiding the backlog in
    its own deque: the standing buffer forms in the kernel, as in the apps.
    """
    t_rel, stale_ms = [], []
    n_states = n_total = 0
    period = (1.0 / drain_hz) if drain_hz else 0.0
    async with websockets.connect(url, ping_interval=None, max_size=2**20,
                                  max_queue=1) as ws:
        t0 = time.time()
        next_read = t0
        while True:
            now = time.time()
            if now - t0 >= secs:
                break
            if period:
                next_read += period
                delay = next_read - now
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_read = time.time()
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            n_total += 1
            trx = time.time()
            try:
                env = json.loads(raw)
            except Exception:
                continue
            if env.get("type") != "robot_state":
                continue
            sent = (env.get("payload") or {}).get("sentAt")
            if not sent:
                continue
            n_states += 1
            t_rel.append(trx - t0)
            stale_ms.append((trx - sent) * 1000.0)
    return np.asarray(t_rel), np.asarray(stale_ms), n_states, n_total


async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--url", default="ws://127.0.0.1:9090/teleop")
    ap.add_argument("--secs", type=float, default=45.0)
    ap.add_argument("--slow-hz", type=float, default=40.0,
                    help="drain rate of the SLOW reader (app-main-thread analogue)")
    ap.add_argument("--tag", default="loopback")
    args = ap.parse_args()

    print(f"probing {args.url} for {args.secs:.0f}s  "
          f"(fast reader + slow reader @ {args.slow_hz:.0f} Hz, concurrent)")
    (fast, slow) = await asyncio.gather(
        reader(args.url, args.secs, None, "fast"),
        reader(args.url, args.secs, args.slow_hz, "slow"))

    rows = []
    for label, (t, s, n_states, n_total), hz in (
            ("fast (wire-speed drain)", fast, None),
            (f"slow ({args.slow_hz:.0f} Hz drain)", slow, args.slow_hz)):
        eff = n_total / args.secs
        rows.append((label, eff, pct(s, 50), pct(s, 95),
                     float(s.max()) if len(s) else float("nan"),
                     float(s[-1]) if len(s) else float("nan")))
        print(f"\n  {label}")
        print(f"    messages read        {n_total}  ({eff:.1f}/s; "
              f"{n_states} robot_state with sentAt)")
        print(f"    staleness ms   p50 {pct(s,50):8.1f}   p95 {pct(s,95):8.1f}   "
              f"max {s.max() if len(s) else float('nan'):8.1f}   "
              f"final {s[-1] if len(s) else float('nan'):8.1f}")

    os.makedirs(FIGS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(fast[0], fast[1] / 1000.0, lw=1.0, label="fast reader (wire-speed drain)")
    ax.plot(slow[0], slow[1] / 1000.0, lw=1.0,
            label=f"slow reader ({args.slow_hz:.0f} Hz drain — busy-main-thread analogue)")
    ax.set_xlabel("time since connect [s]")
    ax.set_ylabel("robot_state staleness [s]")
    ax.set_title("Client-side receive-buffer buildup vs drain rate "
                 f"({args.tag}, stream @ bridge rate)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    out = os.path.join(FIGS, f"state_buffer_{args.tag}.pdf")
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nfigure → {out}")


if __name__ == "__main__":
    asyncio.run(main())
