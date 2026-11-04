#!/usr/bin/env python3
"""
exp_retarget_lab.py — the Retarget Lab's own experiment: every variant, a fixed
task-stratified slice of the EgoDex benchmark clips, kinematic (ideal-servo)
replay, every frame. Regenerates every figure in docs/retarget_lab.tex.

  E1  comparison    per-variant medians (pooled over clips) of each metric
  E2  metric study  do the metrics agree?  Spearman rank correlation of the
                    per-(clip, variant) medians across metrics, and the
                    anchor-bias pair (cos_reach vs place_pct)
  E3  repeatability the same clip twice through the same variant -> max |Δq|
                    (with the episode clock: 0 by construction; the wall-clock
                    finger pacing it replaced measured up to 0.26 rad)
  E4  cost          per-solve time distribution per variant

Run (from anywhere; ~3-6 min on a 10-core Mac):
    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_retarget_lab.py [--clips 16]
Outputs: docs/figures/rtl_*.pdf + experiments/results/exp_retarget_lab.json, and a
summary table on stdout. Deterministic: fixed clip selection, episode clock.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# E1-E4 compare MAPS, so every variant gets the instant servo (the map alone, no robot
# slew in between; the lab's live default is the physical slew servo — E6/E7 measure that).
os.environ.setdefault("MABEL_RTL_SERVO", "ideal")

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG.parent))                      # web_gui/
sys.path.insert(0, str(PKG.parent.parent / "controller"))

from retarget_lab.server import config as CFG              # noqa: E402
from retarget_lab.server.variants import VARIANTS          # noqa: E402
from retarget_lab.server.worker import METRICS             # noqa: E402

FIG = PKG / "docs" / "figures"
RES = HERE / "results"
SHOW = ["cos_reach", "cos_upper", "cos_fore", "elbow_err", "place_pct", "ext_err",
        "orient_deg", "sat_pct", "qvel", "selfcol", "track_mm"]


def pick_clips(n):
    """One clip per task (the first '-all__' episode, sorted), cycling tasks
    until n — task coverage before depth."""
    by_task = defaultdict(list)
    for f in sorted(CFG.EGODEX_CLIPS.glob("*-all__*.json.gz")):
        by_task[f.name.split("-all__")[0]].append(f)
    out, k = [], 0
    while len(out) < n and any(len(v) > k for v in by_task.values()):
        for t in sorted(by_task):
            if len(by_task[t]) > k and len(out) < n:
                out.append("egodex:" + by_task[t][k].name[:-8])
        k += 1
    return out


def run_variant(vid, eids):
    """All clips through one variant (one process: its module knobs are private)."""
    sys.path.insert(0, str(PKG.parent)); sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from retarget_lab.server.evaluate import _eval_variant, summarize
    out = {}
    for eid in eids:
        _, r = _eval_variant(vid, eid)
        out[eid] = {"summary": summarize(r), "solve_ms": r["m"]["solve_ms"],
                    "frames": len(r["t"])}
    return vid, out


def repeat_q(vid, eid):
    sys.path.insert(0, str(PKG.parent)); sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, frames = episodes.load(eid)
    qs = []
    for _ in range(2):
        kr = KinematicRetargeter(vid)
        Q = []
        for fr in frames:
            rec = kr.step(pose_from_frame(fr), fr["t"])
            Q.append([rec["q"][k] for k in sorted(rec["q"])])
        qs.append(np.asarray(Q))
    return float(np.max(np.abs(qs[0] - qs[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    vids = a.variants.split(",")
    eids = pick_clips(a.clips)
    print(f"{len(eids)} clips × {len(vids)} variants")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(vids)) as ex:
        res = dict(f.result() for f in [ex.submit(run_variant, v, eids) for v in vids])
        rep = ex.submit(repeat_q, "ours", eids[0]).result()
    wall = time.time() - t0
    n_frames = sum(res[vids[0]][e]["frames"] for e in eids)
    print(f"done in {wall:.0f} s · {n_frames} frames per variant · repeat max|Δq| = {rep:.2e} rad")

    # per (variant, clip, metric) median; pooled = median over clips
    med = {v: {k: np.array([(res[v][e]["summary"].get(k) or {}).get("p50", np.nan) for e in eids])
               for k in SHOW} for v in vids}
    grasp = {v: np.array([np.nanmean([(res[v][e]["summary"].get(f"grasp_r_{s}") or {}).get("mean", np.nan)
                                      for s in "lr"]) for e in eids]) for v in vids}
    table = {v: {k: float(np.nanmedian(med[v][k])) for k in SHOW} | {"grasp_r": float(np.nanmedian(grasp[v]))}
             for v in vids}
    hdr = "variant".ljust(26) + "".join(k[:9].rjust(10) for k in SHOW + ["grasp_r"])
    print(hdr)
    for v in vids:
        print(VARIANTS[v]["label"][:25].ljust(26) + "".join(f"{table[v][k]:10.3f}" for k in SHOW + ["grasp_r"]))

    RES.mkdir(exist_ok=True)
    (RES / "exp_retarget_lab.json").write_text(json.dumps({
        "clips": eids, "variants": vids, "wall_s": wall, "frames_per_variant": n_frames,
        "repeat_max_dq_rad": rep, "table": table,
        "per_clip": {v: {k: med[v][k].tolist() for k in SHOW} for v in vids},
        "grasp_r_per_clip": {v: grasp[v].tolist() for v in vids},
        "solve_ms": {v: {"p50": float(np.nanmedian(np.concatenate([res[v][e]["solve_ms"] for e in eids]))),
                         "p95": float(np.nanpercentile(np.concatenate([res[v][e]["solve_ms"] for e in eids]), 95))}
                     for v in vids}}, indent=1))
    figures(vids, eids, med, grasp, res)


def figures(vids, eids, med, grasp, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    col = [VARIANTS[v]["color"] for v in vids]
    lab = [VARIANTS[v]["label"].replace("Ours · ", "Ours·").replace("Ours − ", "Ours−") for v in vids]

    # E1: per-clip medians, one dot per clip, bar = median across clips
    keys = ["cos_reach", "cos_fore", "elbow_err", "place_pct", "orient_deg", "qvel"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, k in zip(axes.ravel(), keys):
        x = np.arange(len(vids))
        ax.bar(x, [np.nanmedian(med[v][k]) for v in vids], color=col, alpha=.55, width=.7)
        for i, v in enumerate(vids):
            y = med[v][k]
            ax.scatter(np.full(len(y), i) + np.linspace(-.22, .22, len(y)), y, s=4, color=col[i], zorder=3)
        lbl, unit = METRICS[k][0], METRICS[k][1]
        ax.set_title(f"{lbl}" + (f" [{unit}]" if unit else ""), fontsize=7.5)
        ax.set_xticks(x, lab, rotation=65, ha="right", fontsize=6)
        if k == "qvel":
            ax.set_yscale("log")
    fig.suptitle(f"E1 — per-clip medians ({len(eids)} EgoDex clips, dots) and their median (bar)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rtl_e1_comparison.pdf")
    plt.close(fig)

    # E2a: Spearman correlation between metrics over (clip, variant) medians
    from scipy.stats import spearmanr
    mk = ["cos_reach", "cos_upper", "cos_fore", "elbow_err", "place_pct", "ext_err", "orient_deg", "qvel", "track_mm"]
    # competent maps only: the direct-copy strawman's errors are an order of
    # magnitude larger and would decide every rank by themselves
    cv = [v for v in vids if v != "direct"]
    M = np.array([[med[v][k][i] for k in mk] for v in cv for i in range(len(eids))])
    ok = np.all(np.isfinite(M), axis=1)
    rho = spearmanr(M[ok]).statistic
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mk)), mk, rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(mk)), mk, fontsize=6.5)
    for i in range(len(mk)):
        for j in range(len(mk)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="w" if abs(rho[i, j]) > .6 else "k")
    fig.colorbar(im, fraction=.046)
    ax.set_title(f"E2 — Spearman ρ between metrics\n(n={ok.sum()} clip×variant, direct copy excluded)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "rtl_e2_metric_corr.pdf")
    plt.close(fig)

    # E2b: the anchor-bias pair
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for v, c, l in zip(vids, col, lab):
        ax.scatter(med[v]["place_pct"], med[v]["cos_reach"], s=10, color=c, label=l)
    ax.set_xlabel("placement error [% reach] (head-anchored)")
    ax.set_ylabel("reach-direction cosine (shoulder-anchored)")
    ax.set_xscale("log")
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("E2 — two anchors, two rankings (one dot per clip)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "rtl_e2_anchor_bias.pdf")
    plt.close(fig)

    # E4: solve time
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    data = [np.concatenate([res[v][e]["solve_ms"] for e in eids]) for v in vids]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6)
    for p, c in zip(bp["boxes"], col):
        p.set_facecolor(c); p.set_alpha(.6)
    ax.set_xticks(range(1, len(vids) + 1), lab, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("solve time [ms]")
    ax.set_title("E4 — per-frame retarget cost (one process per variant, 10-core Mac)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "rtl_e4_solve_ms.pdf")
    plt.close(fig)
    print("figures →", FIG)


if __name__ == "__main__":
    main()
