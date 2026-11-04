#!/usr/bin/env python3
"""
exp_connection.py — is the teleop connection FAST and STABLE?

The app's link to the robot is one WebSocket: ws://<host>:9090/teleop. Every
other complaint ("laggy", "won't connect", "model lags the sim") bottoms out in
one of the six numbers below, so measure them directly rather than eyeballing
the app.

  E1  CONNECT SPEED     N cold connects — TCP+WS handshake, `hello`, first
                        `robot_state`. This is what "tap Connect → live" costs.
  E2  STREAM RATE       robot_state Hz + inter-arrival jitter on a held link.
                        The bridge targets ~50 Hz to clients (100 Hz internal).
  E3  STABILITY         a long hold: drops, errors, and STALLS (gaps ≫ period).
                        A gap is what the operator sees as a frozen robot.
  E4  COMMAND RTT       joint_command → the commanded joint actually moves in
                        robot_state. Closed-loop responsiveness, not just
                        downstream rate.
  E5  RECONNECT         abrupt close → reconnect. The app failover path (a
                        dropped Wi-Fi packet must not cost seconds).
  E6  CONCURRENT        N clients at once (app + web + headset): does the rate
                        hold, or does one client starve the others?

Usage:
  python3 tools/exp_connection.py                     # localhost sim
  python3 tools/exp_connection.py --host thor         # the real robot
  python3 tools/exp_connection.py --quick             # short run (CI)

Figures → docs/figures/connection_*.pdf ; every number printed is measured.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "docs", "figures")


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def ms(x):
    return x * 1000.0


class Result:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str):
        self.rows.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<38} {detail}")
        return ok

    def summary(self) -> bool:
        n = len(self.rows)
        p = sum(1 for _, ok, _ in self.rows if ok)
        print("\n" + "=" * 62)
        print(f"  {p}/{n} checks passed")
        for name, ok, detail in self.rows:
            if not ok:
                print(f"    ✗ {name}: {detail}")
        return p == n


def url(host, port):
    tok = os.getenv("MABEL_TOKEN", "").strip()
    u = f"ws://{host}:{port}/teleop?client=exp_connection"
    return u + (f"&key={tok}" if tok else "")


# ── E1 ─────────────────────────────────────────────────────────────────────
async def e1_connect_speed(host, port, n, res):
    print(f"\n── E1  CONNECT SPEED  ({n} cold connects)")
    ws_t, hello_t, state_t = [], [], []
    fails = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            async with websockets.connect(url(host, port), ping_interval=None,
                                          open_timeout=5) as ws:
                ws_t.append(time.perf_counter() - t0)
                got_hello = got_state = None
                while got_state is None:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    if isinstance(raw, bytes):
                        continue
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "hello" and got_hello is None:
                        got_hello = time.perf_counter() - t0
                    elif t == "robot_state":
                        got_state = time.perf_counter() - t0
                if got_hello is not None:
                    hello_t.append(got_hello)
                state_t.append(got_state)
        except Exception as e:                                  # noqa: BLE001
            fails += 1
            if fails == 1:
                print(f"      connect failed: {type(e).__name__}: {e}")
        await asyncio.sleep(0.05)

    res.check("E1 every connect succeeded", fails == 0,
              f"{n - fails}/{n} ok")
    if not state_t:
        return
    res.check("E1 first robot_state < 500 ms (p95)", pct(state_t, 95) < 0.5,
              f"p50 {ms(pct(state_t,50)):.0f} ms · p95 {ms(pct(state_t,95)):.0f} ms · "
              f"max {ms(max(state_t)):.0f} ms")
    print(f"      ws handshake  p50 {ms(pct(ws_t,50)):.1f} ms · max {ms(max(ws_t)):.1f} ms")
    if hello_t:
        print(f"      hello         p50 {ms(pct(hello_t,50)):.1f} ms")
    print(f"      first state   p50 {ms(pct(state_t,50)):.1f} ms")
    return state_t


# ── E2 / E3 ────────────────────────────────────────────────────────────────
async def e2_e3_rate_stability(host, port, secs, res):
    print(f"\n── E2/E3  RATE + STABILITY  (hold {secs}s)")
    gaps, arrive = [], []
    n_state = 0
    drops = 0
    t_start = time.perf_counter()
    last = None
    try:
        async with websockets.connect(url(host, port), ping_interval=20,
                                      open_timeout=5) as ws:
            while time.perf_counter() - t_start < secs:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                except asyncio.TimeoutError:
                    drops += 1
                    break
                if isinstance(raw, bytes):
                    continue
                if json.loads(raw).get("type") != "robot_state":
                    continue
                now = time.perf_counter()
                n_state += 1
                arrive.append(now)
                if last is not None:
                    gaps.append(now - last)
                last = now
    except Exception as e:                                      # noqa: BLE001
        drops += 1
        print(f"      link error: {type(e).__name__}: {e}")

    dur = time.perf_counter() - t_start
    hz = n_state / dur if dur else 0
    res.check("E2 state rate ≥ 40 Hz", hz >= 40,
              f"{hz:.1f} Hz over {dur:.0f}s ({n_state} frames)")
    if gaps:
        res.check("E2 jitter: p99 inter-arrival < 60 ms", pct(gaps, 99) < 0.060,
                  f"p50 {ms(pct(gaps,50)):.1f} · p95 {ms(pct(gaps,95)):.1f} · "
                  f"p99 {ms(pct(gaps,99)):.1f} ms")
        big = [g for g in gaps if g > 0.250]
        res.check("E3 no STALLS > 250 ms", not big,
                  f"{len(big)} stalls · worst gap {ms(max(gaps)):.0f} ms")
    res.check("E3 link never dropped", drops == 0, f"{drops} drops")
    return gaps


# ── E4 ─────────────────────────────────────────────────────────────────────
async def e4_command_rtt(host, port, trials, res):
    """joint_command → the joint MOVES in robot_state. Sim only (it commands
    motion); skipped against a real robot unless --allow-motion."""
    print(f"\n── E4  COMMAND ROUND-TRIP  ({trials} trials)")
    rtts = []

    def joints_of(m):
        """robot_state wraps its body in `payload`, and the joint map is
        `jointPositions` (NOT `joints` — that's the COMMAND field, wire_protocol.py).
        Reading it wrong is why an earlier version of this harness hung forever."""
        p = m.get("payload") if isinstance(m.get("payload"), dict) else m
        return p.get("jointPositions") or {}

    try:
        async with websockets.connect(url(host, port), ping_interval=None,
                                      open_timeout=5) as ws:
            # Learn a joint name + its current value from a live state (bounded:
            # never spin forever on a state that carries no joints).
            joints = {}
            t_learn = time.perf_counter()
            while not joints and time.perf_counter() - t_learn < 5.0:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                if m.get("type") == "robot_state":
                    joints = dict(joints_of(m))
            if not joints:
                res.check("E4 robot_state carries joints", False,
                          "no jointPositions in 5 s — nothing to drive")
                return []
            name = next((j for j in ("neck_yaw", "neck_pitch", "torso_yaw", "lift")
                         if j in joints), next(iter(joints)))
            print(f"      driving joint '{name}' (start {joints[name]:+.4f}, "
                  f"{len(joints)} joints in state)")

            for i in range(trials):
                cur = joints[name]
                target = cur + (0.05 if i % 2 == 0 else -0.05)
                t0 = time.perf_counter()
                await ws.send(json.dumps({"type": "joint_command",
                                          "payload": {"joints": {name: target}}}))
                moved = None
                while moved is None and time.perf_counter() - t0 < 2.0:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2)
                    if isinstance(raw, bytes):
                        continue
                    m = json.loads(raw)
                    if m.get("type") != "robot_state":
                        continue
                    js = joints_of(m)
                    if name in js:
                        joints[name] = js[name]
                        # "responded" = moved clear of sensor/servo noise.
                        if abs(js[name] - cur) > 0.015:
                            moved = time.perf_counter() - t0
                if moved is not None:
                    rtts.append(moved)
                await asyncio.sleep(0.3)
    except Exception as e:                                      # noqa: BLE001
        print(f"      E4 error: {type(e).__name__}: {e}")

    res.check("E4 every command produced motion", len(rtts) == trials,
              f"{len(rtts)}/{trials} responded")
    if rtts:
        res.check("E4 command→motion < 150 ms (p95)", pct(rtts, 95) < 0.150,
                  f"p50 {ms(pct(rtts,50)):.0f} ms · p95 {ms(pct(rtts,95)):.0f} ms · "
                  f"max {ms(max(rtts)):.0f} ms")
    return rtts


# ── E5 ─────────────────────────────────────────────────────────────────────
async def e5_reconnect(host, port, n, res):
    print(f"\n── E5  RECONNECT RECOVERY  ({n} abrupt drops)")
    recov = []
    fails = 0
    for _ in range(n):
        try:
            ws = await websockets.connect(url(host, port), ping_interval=None,
                                          open_timeout=5)
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if not isinstance(raw, bytes) and \
                        json.loads(raw).get("type") == "robot_state":
                    break
            await ws.close()                     # abrupt: like Wi-Fi dropping
            t0 = time.perf_counter()
            ws2 = await websockets.connect(url(host, port), ping_interval=None,
                                           open_timeout=5)
            while True:
                raw = await asyncio.wait_for(ws2.recv(), timeout=5)
                if not isinstance(raw, bytes) and \
                        json.loads(raw).get("type") == "robot_state":
                    break
            recov.append(time.perf_counter() - t0)
            await ws2.close()
        except Exception as e:                                  # noqa: BLE001
            fails += 1
            if fails == 1:
                print(f"      reconnect failed: {type(e).__name__}: {e}")
        await asyncio.sleep(0.1)

    res.check("E5 every reconnect recovered", fails == 0, f"{n - fails}/{n} ok")
    if recov:
        res.check("E5 recovery < 500 ms (p95)", pct(recov, 95) < 0.5,
                  f"p50 {ms(pct(recov,50)):.0f} ms · max {ms(max(recov)):.0f} ms")
    return recov


# ── E6 ─────────────────────────────────────────────────────────────────────
async def e6_concurrent(host, port, n_clients, secs, res):
    print(f"\n── E6  CONCURRENT CLIENTS  ({n_clients} clients, {secs}s)")

    async def one(idx):
        cnt = 0
        t0 = time.perf_counter()
        try:
            async with websockets.connect(url(host, port), ping_interval=None,
                                          open_timeout=5) as ws:
                while time.perf_counter() - t0 < secs:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3)
                    if not isinstance(raw, bytes) and \
                            json.loads(raw).get("type") == "robot_state":
                        cnt += 1
        except Exception:                                       # noqa: BLE001
            return None
        return cnt / (time.perf_counter() - t0)

    rates = await asyncio.gather(*(one(i) for i in range(n_clients)))
    ok = [r for r in rates if r is not None]
    res.check("E6 all clients stayed connected", len(ok) == n_clients,
              f"{len(ok)}/{n_clients} alive")
    if ok:
        res.check("E6 slowest client ≥ 40 Hz", min(ok) >= 40,
                  "rates: " + " · ".join(f"{r:.0f}" for r in ok) + " Hz")
    return ok


def plot(gaps, connect_t, rtts, recov, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                           # noqa: BLE001
        print("\n(matplotlib unavailable — skipping figures)")
        return
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    if gaps:
        ax[0].hist([ms(g) for g in gaps], bins=60, color="#4C78A8")
        ax[0].axvline(20, color="#E45756", ls="--", lw=1, label="20 ms (50 Hz)")
        ax[0].set_xlabel("robot_state inter-arrival (ms)")
        ax[0].set_ylabel("frames")
        ax[0].set_title("E2 stream jitter")
        ax[0].legend(fontsize=7)

    for xs, lab, c in ((connect_t, "connect→first state", "#4C78A8"),
                       (recov, "reconnect recovery", "#F58518")):
        if xs:
            ax[1].plot(sorted([ms(x) for x in xs]),
                       [i / len(xs) for i in range(len(xs))], marker=".",
                       label=lab, color=c)
    ax[1].set_xlabel("ms")
    ax[1].set_ylabel("CDF")
    ax[1].set_title("E1/E5 connect + recovery")
    ax[1].legend(fontsize=7)

    if rtts:
        ax[2].plot([ms(r) for r in rtts], marker="o", color="#54A24B")
        ax[2].set_xlabel("trial")
        ax[2].set_ylabel("command→motion (ms)")
        ax[2].set_title("E4 command RTT")
        ax[2].set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(outdir, "connection_health.pdf")
    fig.savefig(out)
    print(f"\n  figure → {os.path.relpath(out)}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-motion", action="store_true",
                    help="skip E4 (it commands a joint) — use on the real robot")
    a = ap.parse_args()

    n_conn = 8 if a.quick else 30
    hold = 15 if a.quick else 60
    trials = 4 if a.quick else 10
    n_re = 4 if a.quick else 10

    print(f"exp_connection — ws://{a.host}:{a.port}/teleop")
    res = Result()
    connect_t = await e1_connect_speed(a.host, a.port, n_conn, res)
    gaps = await e2_e3_rate_stability(a.host, a.port, hold, res)
    rtts = [] if a.no_motion else await e4_command_rtt(a.host, a.port, trials, res)
    recov = await e5_reconnect(a.host, a.port, n_re, res)
    await e6_concurrent(a.host, a.port, 3, 10 if a.quick else 20, res)
    plot(gaps or [], connect_t or [], rtts or [], recov or [], FIGDIR)
    sys.exit(0 if res.summary() else 1)


if __name__ == "__main__":
    asyncio.run(main())
