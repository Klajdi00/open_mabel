#!/usr/bin/env python3
"""
exp_viewer_contention.py — the real-time-factor + viewer/camera-contention study.

This is the reproducible record behind two figures in docs/server.tex:

  docs/figures/realtime_factor.pdf        sim real-time factor across the configs
                                          that actually move it (LiDAR sensors,
                                          the desktop --viewer, the macOS App Nap
                                          throttle, and the full multi-client stack)
  docs/figures/viewer_camera_tradeoff.pdf head+wrist camera fps + p99 frame gap
                                          versus the desktop viewer's fps cap

WHY a recorded-data harness (not a one-shot orchestrator): reproducing every bar
needs three GPU processes (bridge, sim_camera_server, mjpython view_robot) plus an
iOS-simulator client, several of which need a *foreground* macOS window and cannot
be spawned head-less from one script. So the measured numbers from the 2026-06-28
ground-up audit are recorded below WITH their exact conditions, and the live probes
that produced them are included so any single bar can be re-measured against a
running stack:

    # bring a stack up first (server/run.sh [--viewer]), drive the robot, then:
    python3 tools/exp_viewer_contention.py --live-rtf          # measure RTF now
    python3 tools/exp_viewer_contention.py --live-fps head_left # measure cam fps now
    python3 tools/exp_viewer_contention.py                      # (re)build the PDFs

Real-time factor is read straight from the qpos memmap the bridge publishes:
state[0] is data.time (sim seconds), so RTF = d(sim)/d(wall); 1.0 = real-time.
All RTF rows were taken with the robot MOVING (an oscillating nav stick) so the
viewer renders at its full cap — the worst case for contention.
"""
from __future__ import annotations
import argparse, os, sys, time

SERVER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(SERVER, "docs", "figures")
STATE  = "/tmp/mabel_teleop_state.f64"

# ---------------------------------------------------------------------------
# Measured data — MABEL dev MacBook Pro, mabel_home.xml (116-DOF), 2026-06-28.
# Robot moving throughout. RTF = measured sim-seconds / wall-seconds (operating
# value, i.e. AFTER the catch-up accumulator's 1.0x cap), worst of >=3 windows.
# ---------------------------------------------------------------------------
# (label, RTF, note, ok?) — ordered as the story is told in the doc.
RTF_ROWS = [
    ("LiDAR sensors ON\n(h=4 ms)",          0.31, "mj_step 12.7 ms", False),
    ("sensors OFF\nno viewer",              1.00, "mj_step 2.0 ms",  True),
    ("+ viewer\nNO App-Nap guard",          0.40, "loop -> 5-12 Hz", False),
    ("+ viewer\nApp-Nap guard",             1.00, "loop ~300 Hz",    True),
    ("full stack\niOS+web+cams",            1.00, "RTT p50 0.7 ms",  True),
    ("full stack\n+ viewer",                0.88, "GPU-bound dip",   None),
]

# Camera fps vs the desktop viewer's fps cap (head_left + wrist_left pulled
# together, robot moving). p99 = 99th-pct inter-frame gap (ms). cam caps ~25 fps.
VIEWER_CFG   = ["viewer OFF", "viewer 12 fps\n(new default)", "viewer 20 fps\n(old default)"]
CAM_FPS_HEAD = [24.5, 18.8, 14.7]
CAM_FPS_WRST = [26.5, 20.1, 15.8]
CAM_P99_GAP  = [47.0, 67.0, 185.0]   # ms, worst of the two streams


def _simtime() -> float:
    import numpy as np
    a = np.memmap(STATE, dtype=np.float64, mode="r", shape=(1,))
    v = float(a[0]); del a; return v


def live_rtf(dur: float = 1.5, n: int = 5) -> None:
    """Measure RTF now against a running bridge (drive the robot first)."""
    if not os.path.exists(STATE):
        sys.exit(f"no state file {STATE} — is the bridge running?")
    worst = 9e9
    for i in range(n):
        w0 = time.time(); s0 = _simtime(); time.sleep(dur)
        w1 = time.time(); s1 = _simtime()
        rtf = (s1 - s0) / (w1 - w0); worst = min(worst, rtf)
        print(f"  win {i}: RTF={rtf:5.3f}  (sim {s1-s0:6.3f}s / wall {w1-w0:5.3f}s)")
    print(f"# WORST RTF = {worst:.3f}  {'OK' if worst > 0.9 else 'SLOW'}")


def live_fps(cam: str = "head_left", secs: float = 9.0, port: int = 8080) -> None:
    """Measure delivered MJPEG fps + p99 gap now for one camera stream."""
    import urllib.request
    url = f"http://127.0.0.1:{port}/camera/{cam}/stream.mjpg"
    r = urllib.request.urlopen(url, timeout=5)
    buf = b""; frames = 0; t0 = time.time(); last = t0; gaps = []
    while time.time() - t0 < secs:
        chunk = r.read(8192)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
            if a != -1 and b != -1:
                now = time.time(); gaps.append((now - last) * 1e3); last = now
                frames += 1; buf = buf[b + 2:]
            else:
                break
    dt = time.time() - t0
    gs = sorted(gaps[1:]) or [0.0]
    p99 = gs[min(len(gs) - 1, int(len(gs) * 0.99))]
    print(f"  {cam}: {frames/dt:.1f} fps   p99 gap {p99:.1f} ms")


def build_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)

    OK, BAD, MID = "#2e8b57", "#c0392b", "#e08e3c"

    # ---- Figure 1: real-time factor across configurations -----------------
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels = [r[0] for r in RTF_ROWS]
    vals   = [r[1] for r in RTF_ROWS]
    cols   = [OK if r[3] is True else (BAD if r[3] is False else MID) for r in RTF_ROWS]
    x = range(len(RTF_ROWS))
    bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.66)
    ax.axhline(1.0, color="black", lw=1.0, ls="--", zorder=0)
    ax.text(len(RTF_ROWS) - 0.5, 1.02, "real-time (1.0)", ha="right", va="bottom", fontsize=8)
    ax.axhspan(0.0, 0.9, color="#c0392b", alpha=0.06, zorder=0)
    for b, r in zip(bars, RTF_ROWS):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                f"{r[1]:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # fold the engineering note into a 3rd, muted line of each x-tick label
    ticklabels = [f"{r[0]}\n({r[2]})" for r in RTF_ROWS]
    ax.set_xticks(list(x)); ax.set_xticklabels(ticklabels, fontsize=7.0)
    ax.set_ylabel("sim real-time factor"); ax.set_ylim(0, 2.2)
    ax.set_title("Real-time factor: the two regressions and their fixes (robot moving)",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "realtime_factor.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)

    # ---- Figure 2: viewer fps cap vs camera throughput --------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    import numpy as np
    x = np.arange(len(VIEWER_CFG)); w = 0.34
    ax.bar(x - w / 2, CAM_FPS_HEAD, w, label="head fps", color="#3b6ea5", edgecolor="black", lw=0.5)
    ax.bar(x + w / 2, CAM_FPS_WRST, w, label="wrist fps", color="#7fb0d6", edgecolor="black", lw=0.5)
    for xi, (h, wr) in enumerate(zip(CAM_FPS_HEAD, CAM_FPS_WRST)):
        ax.text(xi - w / 2, h + 0.4, f"{h:.0f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, wr + 0.4, f"{wr:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(VIEWER_CFG, fontsize=8)
    ax.set_ylabel("delivered camera fps"); ax.set_ylim(0, 30)
    ax.set_title("Desktop viewer fps cap vs camera throughput (both streams pulled, robot moving)",
                 fontsize=9.2)
    ax.spines[["top"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, CAM_P99_GAP, "o-", color="#c0392b", lw=1.6, label="p99 frame gap")
    for xi, g in enumerate(CAM_P99_GAP):
        ax2.text(xi, g + 6, f"{g:.0f} ms", ha="center", fontsize=7.6, color="#c0392b")
    ax2.set_ylabel("p99 inter-frame gap (ms)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b"); ax2.set_ylim(0, 220)
    ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.8, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "viewer_camera_tradeoff.pdf")
    fig.savefig(out); plt.close(fig); print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-rtf", action="store_true", help="measure RTF now (bridge must be up)")
    ap.add_argument("--live-fps", metavar="CAM", help="measure one camera's fps now (e.g. head_left)")
    args = ap.parse_args()
    if args.live_rtf:
        live_rtf()
    elif args.live_fps:
        live_fps(args.live_fps)
    else:
        build_figures()


if __name__ == "__main__":
    main()
