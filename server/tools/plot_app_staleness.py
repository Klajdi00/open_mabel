#!/usr/bin/env python3
"""
plot_app_staleness.py — figure + stats from the apps' [MABEL-LAG] debug logs.

The iOS v2 / VisionOS V2 links log a 1 Hz line in DEBUG builds when the bridge
stamps `sentAt` (same-host clocks only — the visionOS/iOS SIMULATOR against a
loopback bridge is the intended rig):

    [MABEL-LAG] robot_state staleness avg=87348.0ms max=87644.9ms n=2

Capture with:  xcrun simctl launch --console-pty <UDID> <bundle-id> > run.log
Then:          python3 tools/plot_app_staleness.py fixed=run_fixed.log \
                   broken=run_broken.log --tag v2_loaded

Prints per-run stats and writes a staleness-vs-time figure to
server/docs/figures/app_staleness_<tag>.pdf. Companion to exp_state_buffer.py
(the headless fast/slow-reader probe of the same client-side buffer mechanism).
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.normpath(os.path.join(_HERE, "..", "docs", "figures"))

LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*\[MABEL-LAG\] "
    r"robot_state staleness avg=(?P<avg>[\d.]+)ms max=(?P<max>[\d.]+)ms n=(?P<n>\d+)")


def parse(path):
    t_rel, avg, mx = [], [], []
    t0 = None
    from datetime import datetime
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S.%f")
            if t0 is None:
                t0 = ts
            t_rel.append((ts - t0).total_seconds())
            avg.append(float(m["avg"]))
            mx.append(float(m["max"]))
    return np.asarray(t_rel), np.asarray(avg), np.asarray(mx)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("runs", nargs="+", help="label=path/to/console.log")
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            sys.exit(f"bad run spec (want label=path): {spec}")
        t, avg, mx = parse(path)
        if not len(t):
            print(f"  {label:14s}  NO [MABEL-LAG] lines in {path}")
            continue
        print(f"  {label:14s}  {len(t)} samples over {t[-1]:.0f}s   "
              f"avg-of-avgs {avg.mean():8.1f} ms   p95(avg) {np.percentile(avg, 95):8.1f} ms   "
              f"worst max {mx.max():8.1f} ms   final avg {avg[-1]:8.1f} ms")
        ax.plot(t, avg / 1000.0, lw=1.1, label=f"{label} (1 s avg)")
    ax.set_xlabel("time since launch [s]")
    ax.set_ylabel("robot_state staleness [s]")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"In-app robot_state staleness ({args.tag})")
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, f"app_staleness_{args.tag}.pdf")
    fig.tight_layout()
    fig.savefig(out)
    print(f"figure → {out}")


if __name__ == "__main__":
    main()
