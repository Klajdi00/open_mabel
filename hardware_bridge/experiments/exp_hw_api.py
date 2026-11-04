#!/usr/bin/env python3
"""exp_hw_api.py — quantify the mabel_hw API layer (all hardware-free).

Regenerates every figure docs/mabel_hw_api.tex cites, from scratch, and prints
a summary table. No robot, no ROS, no root — mocks and the wire-exact fake
Teensy only, so it runs identically on the Mac and on thor.

  E1  lease claim/release + contended-handover latency      (arbitration)
  E2  read() snapshot throughput vs number of live subsystems
  E3  base command-path overhead: direct SwerveBase vs leased facade writer,
      both against tools/fake_teensy.py (the wire-exact UDP fake)
  E4  cold-import cost: the compat shims vs mabel_hw itself

    python3 experiments/exp_hw_api.py            # all, figures + table
    python3 experiments/exp_hw_api.py --only e1

Figures -> docs/figures/hwapi_*.pdf (vector). Seeded; wall-clock numbers move
a little run-to-run, distributions don't.
"""
import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.dirname(_HERE)
sys.path.insert(0, _BRIDGE)
sys.path.insert(0, os.path.join(_BRIDGE, "tests"))

FIGDIR = os.path.join(_BRIDGE, "docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)

RESULTS = {}


def _pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def _savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _BRIDGE)}")


# ── E1: arbitration latency ────────────────────────────────────────────────────

def e1_arbitration(n=5000):
    print("\n[E1] lease claim/release + contended handover")
    from mabel_hw.arbitration import LeaseBroker
    broker = LeaseBroker()

    cycle_us = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        lease = broker.claim("base", "bench")
        broker.check("base", lease.token)
        lease.release()
        cycle_us.append((time.perf_counter_ns() - t0) / 1e3)

    handover_us = []
    for _ in range(500):
        first = broker.claim("base", "one")
        got = {}

        def waiter():
            got["lease"] = broker.claim("base", "two", wait=5.0)
            got["t"] = time.perf_counter_ns()

        th = threading.Thread(target=waiter)
        th.start()
        time.sleep(0.001)                      # let the waiter block
        t0 = time.perf_counter_ns()
        first.release()
        th.join()
        handover_us.append((got["t"] - t0) / 1e3)
        got["lease"].release()

    RESULTS["E1 claim+check+release p50 (us)"] = _pct(cycle_us, 50)
    RESULTS["E1 claim+check+release p99 (us)"] = _pct(cycle_us, 99)
    RESULTS["E1 contended handover p50 (us)"] = _pct(handover_us, 50)
    RESULTS["E1 contended handover p99 (us)"] = _pct(handover_us, 99)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, data, title in ((axes[0], cycle_us, "claim+check+release"),
                            (axes[1], handover_us, "contended handover")):
        ax.hist(data, bins=np.geomspace(min(data), max(data), 60),
                color="#4878a8", alpha=0.85)
        ax.set_xscale("log")
        ax.axvline(_pct(data, 50), color="k", lw=1, ls="--",
                   label=f"p50 {_pct(data, 50):.1f} us")
        ax.axvline(_pct(data, 99), color="crimson", lw=1, ls=":",
                   label=f"p99 {_pct(data, 99):.1f} us")
        ax.set_title(title)
        ax.set_xlabel("latency [us]")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle(f"E1 — write-lease arbitration latency (n={n} / 500)")
    _savefig(fig, "hwapi_arbitration.pdf")


# ── E2: read() throughput vs subsystem count ───────────────────────────────────

def e2_read_throughput(n_reads=3000):
    print("\n[E2] read() throughput vs live subsystems")
    from mabel_hw import MabelHardware
    ladder = [["base"], ["base", "lift"], ["base", "lift", "torso"],
              ["base", "lift", "torso", "left_arm"],
              ["base", "lift", "torso", "left_arm", "right_arm"],
              ["base", "lift", "torso", "left_arm", "right_arm", "neck"],
              ["base", "lift", "torso", "left_arm", "right_arm", "neck",
               "hands"],
              ["base", "lift", "torso", "left_arm", "right_arm", "neck",
               "hands", "lidar", "cameras"]]
    counts, rates, lat_us = [], [], []
    for subs in ladder:
        with MabelHardware(mock=True, subsystems=subs) as hw:
            hw.read()                          # warm
            t0 = time.perf_counter()
            for _ in range(n_reads):
                hw.read()
            dt = time.perf_counter() - t0
        n_live = len(subs) + (1 if "hands" in subs else 0)  # hands = 2 sides
        counts.append(n_live)
        rates.append(n_reads / dt)
        lat_us.append(dt / n_reads * 1e6)
        print(f"  {n_live} subsystems: {rates[-1]:,.0f} snapshots/s "
              f"({lat_us[-1]:.1f} us/read)")
    RESULTS["E2 read() rate @ full robot (Hz)"] = rates[-1]
    RESULTS["E2 read() latency @ full robot (us)"] = lat_us[-1]

    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    ax1.plot(counts, rates, "o-", color="#4878a8")
    ax1.set_xlabel("live subsystems in the snapshot")
    ax1.set_ylabel("RobotState snapshots / s", color="#4878a8")
    ax2 = ax1.twinx()
    ax2.plot(counts, lat_us, "s--", color="#b05a4a")
    ax2.set_ylabel("latency per read() [us]", color="#b05a4a")
    ax1.set_title(f"E2 — read() scales with subsystem count (n={n_reads}/point)")
    _savefig(fig, "hwapi_read_throughput.pdf")


# ── E3: facade overhead on the base command path ───────────────────────────────

def e3_facade_overhead(n=3000):
    print("\n[E3] base command path: direct SwerveBase vs leased facade writer")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fake_teensy", os.path.join(_BRIDGE, "tools", "fake_teensy.py"))
    ft_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft_mod)

    from mabel_hw.arbitration import LeaseBroker
    from mabel_hw.api import BaseWriter
    from mabel_hw.drivers.swerve import SwerveBase
    from mabel_hw.subsystems.base import BaseSubsystem

    fake = ft_mod.FakeTeensy(port=0).start()
    port = fake.sock.getsockname()[1]
    rng = np.random.default_rng(7)
    try:
        base = SwerveBase(teensy_ip="127.0.0.1", port=port, stream_hz=100,
                          apply_host_calib=False)
        time.sleep(0.3)

        direct_us = []
        for _ in range(n):
            vx = float(rng.uniform(-1, 1))
            t0 = time.perf_counter_ns()
            base.set_twist(vx, 0.0, 0.0)
            base.tick()
            direct_us.append((time.perf_counter_ns() - t0) / 1e3)

        core = BaseSubsystem({"_": 0}, swerve=base)     # same link, typed layer
        broker = LeaseBroker()
        writer = BaseWriter(broker, broker.claim("base", "bench"), core)
        facade_us = []
        for _ in range(n):
            vx = float(rng.uniform(-1, 1))
            t0 = time.perf_counter_ns()
            writer.set_twist(vx, 0.0, 0.0)              # lease check + tick inside
            facade_us.append((time.perf_counter_ns() - t0) / 1e3)
        base.close()
    finally:
        fake.running = False
        fake.sock.close()

    d50, d99 = _pct(direct_us, 50), _pct(direct_us, 99)
    f50, f99 = _pct(facade_us, 50), _pct(facade_us, 99)
    RESULTS["E3 direct p50 (us)"] = d50
    RESULTS["E3 facade p50 (us)"] = f50
    RESULTS["E3 added latency p50 (us)"] = f50 - d50
    print(f"  direct p50/p99 {d50:.1f}/{d99:.1f} us; "
          f"facade p50/p99 {f50:.1f}/{f99:.1f} us "
          f"(adds {f50 - d50:+.1f} us at p50)")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    parts = ax.violinplot([direct_us, facade_us], showmedians=True)
    for pc, color in zip(parts["bodies"], ("#4878a8", "#b05a4a")):
        pc.set_facecolor(color)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"direct SwerveBase\np50 {d50:.1f} us",
                        f"facade writer (lease-gated)\np50 {f50:.1f} us"])
    ax.set_ylabel("set_twist + push latency [us]")
    ax.set_yscale("log")
    ax.set_title(f"E3 — the API layer adds {f50 - d50:.1f} us at p50 "
                 f"(n={n}, fake Teensy loopback)")
    _savefig(fig, "hwapi_facade_overhead.pdf")


# ── E4: cold import cost ───────────────────────────────────────────────────────

def e4_import_cost(trials=7):
    print("\n[E4] cold-import cost (subprocess, median of trials)")
    cases = [
        ("import config (shim)", "import config"),
        ("import mabel_hw (lazy)", "import mabel_hw"),
        ("MabelHardware resolved", "from mabel_hw import MabelHardware"),
        ("drivers.swerve (numpy+pb)", "import mabel_hw.drivers.swerve"),
    ]
    labels, med_ms = [], []
    for label, stmt in cases:
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "-c", stmt], cwd=_BRIDGE,
                           check=True, capture_output=True)
            times.append((time.perf_counter() - t0) * 1e3)
        labels.append(label)
        med_ms.append(statistics.median(times))
        print(f"  {label:<28} {med_ms[-1]:7.1f} ms")
    RESULTS["E4 import mabel_hw (ms)"] = med_ms[1]
    RESULTS["E4 MabelHardware resolved (ms)"] = med_ms[2]

    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.barh(labels[::-1], med_ms[::-1], color="#4878a8")
    for i, v in enumerate(med_ms[::-1]):
        ax.text(v, i, f" {v:.0f} ms", va="center", fontsize=9)
    ax.set_xlabel(f"cold python -c import [ms], median of {trials} "
                  "(includes interpreter start)")
    ax.set_title("E4 — import cost: lazy package vs resolved facade")
    _savefig(fig, "hwapi_import_cost.pdf")


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4"])
    args = ap.parse_args(argv)
    todo = {"e1": e1_arbitration, "e2": e2_read_throughput,
            "e3": e3_facade_overhead, "e4": e4_import_cost}
    for key, fn in todo.items():
        if args.only in (None, key):
            fn()
    print("\n== summary ==")
    for k, v in RESULTS.items():
        print(f"  {k:<42} {v:12,.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
