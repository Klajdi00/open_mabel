#!/usr/bin/env python3
"""
Whole-body wrist-tracking EDGE-CASE sweep — robustness + snap audit.

Operator directive (2026-07-18): "move the target up down, left right, closer
and more separate — see the robustness of the controller; it should be snapped
and not float all the time." Each case runs a REALTIME world (the float/hunt
pathologies do not show under step_sync), drags BOTH palm targets (or each
separately for squeeze/spread), and measures:

    settle   first time both palm errors stay < 40 mm (n/a = never)
    err L/R  final palm tracking errors
    float    post-settle motion over the last 2.5 s: palm std [mm] and the
             mean |base twist| [m/s] — "snapped" means both ~0
    base     chassis displacement + yaw actually used by the follower
    pitch    max |chassis pitch| (upright = stable)

Geometrically infeasible cases (spread beyond arm span, down at the lift's
bottom stop) are expected to show bounded error WITHOUT oscillation — the
robustness criterion is stability, not magic reach.

Figure: docs/figures/lab_wrist_edgecases.pdf.  Table: printed + returned.
"""
import os
import sys
import time
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("MABEL_CTS_STATE_FILE", "/tmp/mabel_control_lab_EDGE_state.f64")
os.environ.setdefault("MABEL_CTS_VP_PORT", "19431")

from control_lab.server.simworld import SimWorld            # noqa: E402
import mabel.config as C                                    # noqa: E402

FIGDIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")

# (label, d_left, d_right) — WORLD frame: robot front = -X, robot left = -Y.
CASES = [
    ("fwd 0.6",        (-0.6,  0.0,  0.0), (-0.6,  0.0,  0.0)),
    ("back 0.25",      (+0.25, 0.0,  0.0), (+0.25, 0.0,  0.0)),
    ("left 0.5",       ( 0.0, -0.5,  0.0), ( 0.0, -0.5,  0.0)),
    ("right 0.5",      ( 0.0, +0.5,  0.0), ( 0.0, +0.5,  0.0)),
    ("up 0.3",         ( 0.0,  0.0, +0.3), ( 0.0,  0.0, +0.3)),
    ("down 0.15",      ( 0.0,  0.0, -0.15),( 0.0,  0.0, -0.15)),
    ("diag fwd-left",  (-0.5, -0.4,  0.0), (-0.5, -0.4,  0.0)),
    ("far-left turn",  (-0.1, -0.8,  0.0), (-0.1, -0.8,  0.0)),
    ("squeeze 0.15",   ( 0.0, +0.15, 0.0), ( 0.0, -0.15, 0.0)),
    ("spread 0.25",    ( 0.0, -0.25, 0.0), ( 0.0, +0.25, 0.0)),
    ("fwd+up",         (-0.4,  0.0, +0.25),(-0.4,  0.0, +0.25)),
    # Rotate the PAIR 40° CCW about its centroid (an operator turning their
    # body): the pair-alignment heading law must YAW the robot ~+40°.
    ("pair rotate 40", (+0.133, +0.048, 0.0), (-0.133, -0.048, 0.0)),
]

SETTLE_MM = 40.0
RUN_S = 14.0


def run_case(label, dl, dr):
    w = SimWorld()
    w.settle()
    w.start()
    try:
        time.sleep(0.3)
        w.set_torque(True)
        w.set_control_spec(method="wrist", control_type="impedance",
                           region="whole_body", stiffness=45.0)
        time.sleep(0.3)
        with w.lock:
            p0 = {s: np.asarray(w.snapshot()["bodies"][f"palm_{s}"]) for s in ("l", "r")}
            x0, y0, yaw0 = w._true_base()
        w.set_wrist_target("l", (p0["l"] + np.asarray(dl)).tolist())
        w.set_wrist_target("r", (p0["r"] + np.asarray(dr)).tolist())
        wall0 = time.time()
        with w.lock:
            t0 = w.snapshot()["t"]
        T, EL, ER, PLM, TW = [], [], [], [], []
        pitch_max = 0.0
        settle_t = None
        # Pace on SIM time (load-immune): under CPU contention the realtime
        # factor drops and a wall-clock window samples mid-convergence.
        while True:
            time.sleep(0.05)
            with w.lock:
                snap = w.snapshot()
                bf = w.base_follow
                R = w.data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
                f = -R[:, 0]
            t = snap["t"] - t0
            if t >= RUN_S or time.time() - wall0 > 3.5 * RUN_S:
                break
            tg = snap["overlay"].get("targets", {})
            if "l" not in tg or "r" not in tg:
                continue
            e = {k: 1e3 * np.linalg.norm(np.asarray(tg[k]["target"])
                                         - np.asarray(tg[k]["actual"]))
                 for k in ("l", "r")}
            T.append(t); EL.append(e["l"]); ER.append(e["r"])
            PLM.append(np.asarray(snap["bodies"]["palm_l"]))
            TW.append(math.hypot(bf.vx, bf.vy) + abs(bf.omega))
            pitch_max = max(pitch_max, abs(math.degrees(
                math.asin(max(-1.0, min(1.0, -f[2]))))))
            if settle_t is None and e["l"] < SETTLE_MM and e["r"] < SETTLE_MM:
                settle_t = t
            elif settle_t is not None and (e["l"] > SETTLE_MM or e["r"] > SETTLE_MM):
                settle_t = None                      # must STAY settled
        with w.lock:
            x1, y1, yaw1 = w._true_base()
        tail = [i for i, t in enumerate(T) if t > RUN_S - 2.5]
        float_mm = 1e3 * float(np.linalg.norm(np.asarray(
            [PLM[i] for i in tail]).std(0))) if len(tail) > 5 else float("nan")
        float_tw = float(np.mean([TW[i] for i in tail])) if tail else float("nan")
        return dict(label=label, T=T, EL=EL, ER=ER,
                    settle=settle_t, el=EL[-1], er=ER[-1],
                    float_mm=float_mm, float_tw=float_tw,
                    base=math.hypot(x1 - x0, y1 - y0), dyaw=yaw1 - yaw0,
                    pitch=pitch_max)
    finally:
        w.stop()


def main():
    rows = []
    for label, dl, dr in CASES:
        rows.append(run_case(label, dl, dr))
        r = rows[-1]
        print(f"  {label:14s} settle={r['settle'] if r['settle'] is None else round(r['settle'],1)!s:>5} "
              f"err L/R={r['el']:6.1f}/{r['er']:6.1f}mm  float={r['float_mm']:5.2f}mm "
              f"twist={r['float_tw']:5.3f}  base={r['base']:.2f}m yaw={math.degrees(r['dyaw']):+5.1f}° "
              f"pitch={r['pitch']:4.2f}°", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    for r, c in zip(rows, cmap):
        axs[0].plot(r["T"], r["EL"], color=c, label=r["label"], lw=1.2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("left palm err [mm]")
    axs[0].set_title("Edge-case tracking error (realtime)")
    axs[0].axhline(SETTLE_MM, color="0.5", ls="--", lw=0.8)
    axs[0].legend(fontsize=6, ncol=2)
    labels = [r["label"] for r in rows]
    axs[1].barh(labels, [r["float_mm"] for r in rows], color="#1f77b4")
    axs[1].set_xlabel("post-settle palm float, std [mm]")
    axs[1].set_title('"snapped, not floating"')
    axs[1].invert_yaxis()
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "lab_wrist_edgecases.pdf"), bbox_inches="tight")
    print(f"  figure -> docs/figures/lab_wrist_edgecases.pdf")
    return rows


if __name__ == "__main__":
    main()
