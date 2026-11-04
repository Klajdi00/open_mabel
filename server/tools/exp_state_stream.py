#!/usr/bin/env python3
"""
exp_state_stream.py — measure the LIVE robot-state stream a client actually sees.

The offline budget (exp_loop_latency.py) times each compute stage in isolation
and composes a MODELLED end-to-end number. It cannot see the thing the operator
complains about: the joint-state stream "freezing then jumping" — an intermittent,
whole-system, scheduling-level phenomenon. This harness measures it empirically.

It connects to a RUNNING bridge as an ordinary teleop client (passive: it never
sends a teleop frame or a joint command, so it does NOT claim the driver token and
does NOT move the robot — it only listens and PINGs), then quantifies:

  • ROBOT_STATE inter-arrival gaps  → effective Hz, jitter, p50/p95/p99/max,
    and the count of "stall" gaps (> 2x the nominal 20 ms period). These stalls
    ARE the visible "freeze then jump".
  • application-level RTT via PING/PONG (one ping outstanding; the bridge's pong
    carries the SERVER clock, so RTT is measured purely client-side).
  • the distribution of the server-reported `latencyMs` field — to show what that
    number actually means (command staleness, NOT stream/transport latency).

On loopback (--url ws://localhost:9090/teleop) the network term is ~0, so the gap
statistics isolate SERVER-SIDE regularity: event-loop occupancy + the GIL tug with
the physics/control thread. Run it against a remote host (LAN / Tailscale) to add the network term.

Outputs a per-metric table and a figure (gap time-series + gap histogram + RTT
histogram) → server/docs/figures/state_stream_<tag>.pdf

Run:  python3 server/tools/exp_state_stream.py --secs 30
      python3 server/tools/exp_state_stream.py --url ws://192.168.1.42:9090/teleop --tag lan
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
FIGS = os.path.normpath(os.path.join(_HERE, "..", "docs", "figures"))

NOMINAL_HZ = 50.0                  # bridge _send_state target (asyncio.sleep 0.02)
NOMINAL_DT = 1000.0 / NOMINAL_HZ   # 20 ms


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


async def run(url: str, secs: float, ping_hz: float):
    gaps_ms: list[float] = []          # ROBOT_STATE inter-arrival, ms
    state_t: list[float] = []          # arrival time (perf_counter), s — for the time-series
    srv_lat: list[float] = []          # server-reported latencyMs field
    rtts_ms: list[float] = []          # PING→PONG round trip, ms
    n_state = 0

    async with websockets.connect(url, ping_interval=None, max_size=2**20) as ws:
        t0 = time.perf_counter()
        last = None
        ping_out = {}                  # token → send perf_counter
        next_ping = 0.0
        ping_dt = 1.0 / ping_hz if ping_hz > 0 else 1e9
        ping_seq = 0

        while time.perf_counter() - t0 < secs:
            # schedule a ping at ping_hz (one outstanding at a time)
            now = time.perf_counter()
            if now >= next_ping and len(ping_out) == 0:
                ping_seq += 1
                ping_out[ping_seq] = now
                # carry our own token in the payload so we can match the pong by order
                await ws.send(json.dumps({"type": "ping", "payload": {"t": time.time(), "c": ping_seq}}))
                next_ping = now + ping_dt

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            now = time.perf_counter()

            try:
                env = json.loads(raw)
            except Exception:
                continue
            mtype = env.get("type")
            payload = env.get("payload", {})

            if mtype == "robot_state":
                n_state += 1
                state_t.append(now - t0)
                if last is not None:
                    gaps_ms.append((now - last) * 1000.0)
                last = now
                lm = payload.get("latencyMs")
                if isinstance(lm, (int, float)):
                    srv_lat.append(float(lm))
            elif mtype == "pong":
                # the bridge echoes only its own clock, so match by recency:
                # pop the oldest outstanding ping (we keep only one).
                if ping_out:
                    k = next(iter(ping_out))
                    rtts_ms.append((now - ping_out.pop(k)) * 1000.0)
            elif mtype == "hello":
                pass

    return dict(gaps=np.asarray(gaps_ms), state_t=np.asarray(state_t),
                srv_lat=np.asarray(srv_lat), rtts=np.asarray(rtts_ms),
                n_state=n_state, secs=secs)


def report(url, tag, R):
    gaps, state_t, srv_lat, rtts = R["gaps"], R["state_t"], R["srv_lat"], R["rtts"]
    secs = R["secs"]
    eff_hz = R["n_state"] / secs if secs else 0.0
    stalls = gaps[gaps > 2 * NOMINAL_DT] if len(gaps) else np.array([])

    print(f"\n=== live robot_state stream  [{tag}]  {url} ===")
    print(f"  duration                {secs:6.1f} s   frames {R['n_state']}")
    print(f"  effective rate          {eff_hz:6.1f} Hz   (nominal {NOMINAL_HZ:.0f} Hz, dt {NOMINAL_DT:.0f} ms)")
    if len(gaps):
        print(f"  inter-arrival gap  mean {gaps.mean():6.1f} ms   p50 {pct(gaps,50):5.1f}  "
              f"p95 {pct(gaps,95):5.1f}  p99 {pct(gaps,99):5.1f}  max {gaps.max():6.1f}")
        print(f"  jitter (std)            {gaps.std():6.1f} ms")
        print(f"  STALLS (> {2*NOMINAL_DT:.0f} ms)        {len(stalls):4d}   "
              f"({100.0*len(stalls)/len(gaps):.1f}% of gaps)   "
              f"worst {gaps.max():.0f} ms" + (f"   total frozen {stalls.sum()/1000:.2f} s" if len(stalls) else ""))
    if len(rtts):
        print(f"  app PING→PONG RTT  mean {rtts.mean():6.1f} ms   p50 {pct(rtts,50):5.1f}  "
              f"p95 {pct(rtts,95):5.1f}  max {rtts.max():6.1f}   (n={len(rtts)})")
    if len(srv_lat):
        print(f"  server latencyMs field  mean {srv_lat.mean():6.1f} ms   "
              f"min {srv_lat.min():.0f}  max {srv_lat.max():.0f}   "
              f"(= time since last INBOUND command, NOT stream latency)")

    os.makedirs(FIGS, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.0))
    if len(gaps):
        ax[0].plot(state_t[1:], gaps, lw=0.6, color="#444")
        ax[0].axhline(NOMINAL_DT, color="tab:green", ls="--", lw=1, label=f"nominal {NOMINAL_DT:.0f} ms")
        ax[0].axhline(2 * NOMINAL_DT, color="tab:red", ls=":", lw=1, label="stall threshold")
        ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("inter-arrival gap [ms]")
        ax[0].set_title(f"state-stream gaps · {eff_hz:.0f} Hz eff"); ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.25)

        ax[1].hist(gaps, bins=60, color="#666")
        ax[1].axvline(NOMINAL_DT, color="tab:green", ls="--", lw=1)
        ax[1].axvline(2 * NOMINAL_DT, color="tab:red", ls=":", lw=1)
        ax[1].set_xlabel("gap [ms]"); ax[1].set_ylabel("count")
        ax[1].set_title(f"gap dist · p99 {pct(gaps,99):.0f} ms · max {gaps.max():.0f} ms")
        ax[1].grid(alpha=0.25)
    if len(rtts):
        ax[2].hist(rtts, bins=40, color="tab:blue")
        ax[2].set_xlabel("RTT [ms]"); ax[2].set_ylabel("count")
        ax[2].set_title(f"app RTT · mean {rtts.mean():.1f} ms · p95 {pct(rtts,95):.0f} ms")
        ax[2].grid(alpha=0.25)
    fig.suptitle(f"Live teleop state stream — {tag} ({url})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGS, f"state_stream_{tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"  figure → {out}")


async def run_many(url, secs, ping_hz, n):
    """Spawn n concurrent passive clients; return their per-client result dicts.
    Reveals per-client scaling: if the bridge re-encodes the (identical) state
    payload once per client on its single event loop, effective Hz should sag and
    p99 gaps should grow as n rises."""
    return await asyncio.gather(*[run(url, secs, ping_hz if i == 0 else 0.0) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:9090/teleop")
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--ping-hz", type=float, default=5.0)
    ap.add_argument("--tag", default="local")
    ap.add_argument("--clients", type=int, default=1,
                    help="number of concurrent passive clients (scaling test)")
    args = ap.parse_args()
    if args.clients <= 1:
        R = asyncio.run(run(args.url, args.secs, args.ping_hz))
        report(args.url, args.tag, R)
        return
    Rs = asyncio.run(run_many(args.url, args.secs, args.ping_hz, args.clients))
    hz = [r["n_state"] / r["secs"] for r in Rs]
    p99 = [pct(r["gaps"], 99) for r in Rs]
    mx = [float(r["gaps"].max()) if len(r["gaps"]) else float("nan") for r in Rs]
    print(f"\n=== {args.clients} concurrent clients  [{args.tag}]  {args.url} ===")
    print(f"  effective Hz   per-client: min {min(hz):.1f}  mean {np.mean(hz):.1f}  max {max(hz):.1f}")
    print(f"  gap p99 [ms]   per-client: min {min(p99):.1f}  mean {np.mean(p99):.1f}  max {max(p99):.1f}")
    print(f"  gap max [ms]   per-client: worst {max(mx):.1f}")
    report(args.url, args.tag, Rs[0])   # figure for client 0 (it also carries RTT)


if __name__ == "__main__":
    main()
