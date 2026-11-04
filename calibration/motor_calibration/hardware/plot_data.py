#!/usr/bin/env python3
"""
plot_data.py
================

Visualise the RAW data logged by ``collect.py`` (or the simulator) --
independent of the identification step, so you can eyeball a run before/without
fitting it.

  * staircase CSV -> a 2-panel figure:
      (top)    velocity readback ``w`` and commanded ``w_cmd`` vs time, with the
               "steady" (settled constant-velocity) samples shaded -- this is
               where you SEE whether the motor actually reached each level and
               held it (stall, overshoot, or clean track).
      (bottom) the friction curve: mean torque vs mean velocity on each steady
               level (the exact points the identifier fits), with +/- std bars.

  * chirp / dynamic CSV -> a 2-panel figure:
      (top)    q, w, a vs time   (bottom) reported torque vs time.

Usage
-----
    # auto-find a motor's logs under results/<motor>/logs/
    python plot_data.py --motor DM-J4310

    # explicit files
    python plot_data.py --staircase ../results/DM-J4310/logs/staircase.csv \
                        --chirp     ../results/DM-J4310/logs/chirp.csv

    python plot_data.py --motor DM-J4310 --show     # also pop up a window

Writes PNGs to results/<motor>/plots/ (or next to explicit CSVs); offline, no hardware.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path, defines EXP_DIR)

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402

from mabel_actid.data_logging import CalibrationLog  # noqa: E402


def _shade_steady(ax, t, steady):
    """Shade the contiguous spans where ``steady`` is True."""
    s = steady.astype(int)
    edges = np.flatnonzero(np.diff(s) != 0) + 1
    spans = np.split(np.arange(t.size), edges)
    first = True
    for span in spans:
        if span.size and steady[span[0]]:
            ax.axvspan(t[span[0]], t[span[-1]], color="tab:green", alpha=0.12,
                       label="steady (fit) window" if first else None)
            first = False


def plot_staircase(path: str, show: bool, out: str = None) -> str:
    log = CalibrationLog.load(path)
    w_lvl, tau_lvl = log.steady_points()

    # per-level std for the error bars (recompute the same segmentation)
    s = log.steady
    edges = np.flatnonzero(np.diff(s.astype(int)) != 0) + 1
    segs = [seg for seg in np.split(np.arange(len(log)), edges)
            if seg.size and s[seg[0]]]
    w_std, tau_std = [], []
    for seg in segs:
        wc = float(np.median(log.w_cmd[seg]))
        keep = seg[np.abs(log.w[seg] - wc) <= 0.25 * abs(wc) + 0.01]
        if keep.size >= max(3, 0.2 * seg.size):
            w_std.append(float(np.std(log.w[keep])))
            tau_std.append(float(np.std(log.tau[keep])))
    w_std = np.asarray(w_std); tau_std = np.asarray(tau_std)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8))

    # --- panel 0: velocity vs time -------------------------------------
    _shade_steady(ax0, log.t, log.steady)
    ax0.plot(log.t, log.w_cmd, color="tab:orange", lw=1.4, label="commanded w_cmd")
    ax0.plot(log.t, log.w, color="tab:blue", lw=0.9, label="readback w")
    ax0.axhline(0, color="0.8", lw=0.8)
    ax0.set_xlabel("time [s]"); ax0.set_ylabel("velocity [rad/s]")
    ax0.set_title(f"Velocity staircase — {log.motor or 'motor'}  "
                  f"({path.split(os.sep)[-1]})")
    ax0.legend(loc="upper right", fontsize=8); ax0.grid(True, alpha=0.3)

    # --- panel 1: friction curve ---------------------------------------
    ax1.axhline(0, color="0.8", lw=0.8); ax1.axvline(0, color="0.8", lw=0.8)
    if w_lvl.size:
        order = np.argsort(w_lvl)
        yerr = tau_std[order] if tau_std.size == w_lvl.size else None
        xerr = w_std[order] if w_std.size == w_lvl.size else None
        ax1.errorbar(w_lvl[order], tau_lvl[order], yerr=yerr, xerr=xerr,
                     fmt="o-", ms=6, color="tab:red", ecolor="0.6", capsize=3,
                     label="steady (constant-v) points")
    else:
        ax1.text(0.5, 0.5, "no steady points recovered\n"
                 "(motor stalled, or v_max scaling wrong?)",
                 ha="center", va="center", transform=ax1.transAxes, color="crimson")
    ax1.set_xlabel("joint velocity  w  [rad/s]")
    ax1.set_ylabel("friction torque  tau_f  [N*m]")
    ax1.set_title("Friction curve (what the identifier fits)")
    ax1.legend(loc="upper left", fontsize=8); ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_data.png"
    fig.savefig(out, dpi=130)
    print(f"  staircase: {len(log)} samples, {w_lvl.size} steady levels "
          f"-> {os.path.relpath(out, _bootstrap.CALIB_ROOT)}")
    if not show:
        plt.close(fig)
    return out


def plot_dynamic(path: str, show: bool, out: str = None) -> str:
    log = CalibrationLog.load(path)
    a = log.a
    if not np.all(np.isfinite(a)):
        a = np.gradient(log.w, log.t)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax0.plot(log.t, log.q, color="tab:blue", lw=1.0, label="q [rad]")
    ax0.plot(log.t, log.w, color="tab:green", lw=0.9, label="w [rad/s]")
    ax0.plot(log.t, a, color="tab:purple", lw=0.7, alpha=0.7, label="a [rad/s^2]")
    ax0.axhline(0, color="0.8", lw=0.8)
    ax0.set_ylabel("kinematics")
    ax0.set_title(f"Dynamic sweep — {log.motor or 'motor'}  "
                  f"({path.split(os.sep)[-1]})")
    ax0.legend(loc="upper right", fontsize=8); ax0.grid(True, alpha=0.3)

    ax1.plot(log.t, log.tau, color="tab:red", lw=0.9, label="reported torque")
    ax1.axhline(0, color="0.8", lw=0.8)
    ax1.set_xlabel("time [s]"); ax1.set_ylabel("torque [N*m]")
    ax1.legend(loc="upper right", fontsize=8); ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_data.png"
    fig.savefig(out, dpi=130)
    print(f"  dynamic:   {len(log)} samples "
          f"-> {os.path.relpath(out, _bootstrap.CALIB_ROOT)}")
    if not show:
        plt.close(fig)
    return out


def _autofind(motor: str):
    logs = os.path.join(_bootstrap.RESULTS_DIR, motor, "logs")
    stair = os.path.join(logs, "staircase.csv")
    chirp = os.path.join(logs, "chirp.csv")
    return (stair if os.path.exists(stair) else None,
            chirp if os.path.exists(chirp) else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", default="DM-J4310",
                    help="motor name; auto-finds results/<motor>/logs/*.csv")
    ap.add_argument("--staircase", default=None, help="explicit staircase CSV")
    ap.add_argument("--chirp", "--dynamic", dest="chirp", default=None,
                    help="explicit chirp/dynamic CSV")
    ap.add_argument("--show", action="store_true", help="also display the figures")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    stair, chirp = args.staircase, args.chirp
    explicit = stair is not None or chirp is not None
    if not explicit:
        stair, chirp = _autofind(args.motor)

    if not stair and not chirp:
        print(f"no data for motor {args.motor} in {_bootstrap.RESULTS_DIR} "
              f"(run collect.py first, or pass --staircase/--chirp)")
        return 1

    # route generated plots into results/<motor>/plots/ unless files were explicit
    plots = None if explicit else _bootstrap.result_dir(args.motor, "plots")
    print("plotting:")
    if stair:
        plot_staircase(stair, args.show,
                       os.path.join(plots, "staircase_data.png") if plots else None)
    if chirp:
        plot_dynamic(chirp, args.show,
                     os.path.join(plots, "chirp_data.png") if plots else None)
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
