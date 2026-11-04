#!/usr/bin/env python3
"""
make_doc_figures.py — regenerate every figure in docs/figures/ for server.tex.

All numbers here are the values *measured on the development Mac* and recorded in
the server runbook (README.md); this harness only
plots them so the document and the figures never drift. Re-run after any change
that affects the render budget, the weighted schedule, or the physics timestep:

    cd server && python scripts/make_doc_figures.py     # or: make -C docs figures

Figures written (vector PDF):
    docs/figures/camera_fps.pdf       per-stream MJPEG frame rate vs. mode
    docs/figures/realtime_factor.pdf  sim real-time factor vs. physics timestep
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: no display, vector output only
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "docs", "figures"))
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.autolayout": True,
})


def fig_camera_fps() -> str:
    """Per-stream MJPEG frame rate, before vs. after the shadow fix.

    The off-screen render is a fixed per-frame stall on the deprecated-OpenGL
    macOS path, INDEPENDENT of camera resolution. Measured on the dev Mac
    (mabel_home scene): with shadows ON it is ~73 ms/frame -> ~13.7 render
    slots/s total; with shadows OFF the shadow pass (~32 ms) is gone, ~41 ms
    -> ~24.2 slots/s. Per-stream fps = budget x the camera's schedule share.

    BEFORE: shadows on + head-weighted schedule (head w=8, wrist w=1, /10)
            -> head ~11 fps, each wrist a laggy ~1.4 fps.
    AFTER:  shadows off (default) + balanced schedule (head w=4, wrist w=2, /8)
            -> head ~12 fps, each wrist ~6 fps. `--shadows` restores BEFORE.
    """
    streams = ["Head\n(main)", "Wrist L", "Wrist R"]
    before = [10.9, 1.4, 1.4]   # shadows on, weights 8/1/1 over 13.7 slots/s
    after = [12.1, 6.0, 6.0]    # shadows off, weights 4/2/2 over 24.2 slots/s

    x = range(len(streams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    b1 = ax.bar([i - w / 2 for i in x], before, w, label="before (shadows on, 8/1/1)",
                color="#b0683b")
    b2 = ax.bar([i + w / 2 for i in x], after, w, label="after (shadows off, 4/2/2)",
                color="#3b6fb0")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(streams)
    ax.set_ylabel("MJPEG frame rate (fps)")
    ax.set_ylim(0, 14)
    ax.set_title("Per-stream camera frame rate: shadow fix")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIGDIR, "camera_fps.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_realtime_factor() -> str:
    """Sim wall-clock real-time factor vs. physics timestep.

    mj_step costs a fixed ~3.9 ms for the 116-DOF model regardless of timestep
    (Newton converges well before the iteration cap), so the only lever for
    real-time tracking is opt.timestep. Values from sim_teleop_bridge.run().
    """
    dt_ms = [2.0, 4.0, 5.0]
    rtf = [0.5, 1.0, 1.3]
    labels = ["native\n(jumpy IK lag)", "default\n(real-time)", "fast\n(jumpier)"]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(dt_ms, rtf, "o-", color="#3b6fb0", lw=2, ms=7)
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(2.05, 1.03, "real-time", color="gray", fontsize=9)
    for x, y, t in zip(dt_ms, rtf, labels):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, -18),
                    fontsize=8, color="#444")
    ax.set_xlabel("physics timestep (ms)")
    ax.set_ylabel("real-time factor (x)")
    ax.set_title("Sim real-time factor vs. timestep (mj\\_step $\\approx$ 3.9 ms fixed)")
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.3)
    out = os.path.join(FIGDIR, "realtime_factor.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_channel_rates() -> str:
    """Configured teleop-channel rates for the WebSocket/MJPEG protocol.

    Uplink (client -> bridge, TeleopFrame command stream): the iPhone app sends
    at 20 Hz (WebSocketLink.swift), the Vision Pro at 60 Hz
    (transmissionRateHz, AppConfiguration.swift); ARKit samples the hands at
    ~90 Hz upstream of the rate limiter. The bridge consumes newest-only.
    Downlink (bridge -> client, MJPEG video): the measured head/wrist rates from
    fig_camera_fps. Plotted on one axis so the asymmetry is visible.
    """
    labels = ["iPhone uplink\n(commands)", "Vision Pro uplink\n(commands)",
              "ARKit hands\n(raw, upstream)", "Head video\ndownlink", "Wrist video\ndownlink"]
    rates = [20.0, 60.0, 90.0, 12.0, 6.0]
    colors = ["#3b6fb0", "#3b6fb0", "#7a93b8", "#c06a3b", "#c06a3b"]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bars = ax.bar(labels, rates, color=colors)
    ax.bar_label(bars, fmt="%.0f Hz", padding=3, fontsize=9)
    ax.set_ylabel("rate (Hz / fps)")
    ax.set_ylim(0, 100)
    ax.set_title("Teleop channel rates (uplink commands vs. downlink video)")
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(FIGDIR, "channel_rates.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (fig_camera_fps, fig_realtime_factor, fig_channel_rates):
        path = fn()
        print(f"wrote {os.path.relpath(path, os.path.join(HERE, '..'))}")


if __name__ == "__main__":
    main()
