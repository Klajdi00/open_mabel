"""
Robustness test for the curation detectors: run them over the imperfect_sim bags
and score detections against the injected ground truth (injected_defects_json).

Reports per-defect-type RECALL (fraction of injected defects caught) and PRECISION
(fraction of detections that correspond to a real injected defect), plus a figure.
A detection matches a ground-truth defect when their types are compatible, their
frame spans overlap (±tol), and their scope agrees (same joints / camera / stream).

Run:
    cd learning
    python -m data_curation.server.tools.make_imperfect --n 6     # (once, to create the bags)
    python -m data_curation.server.tools.exp_robustness
"""
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import catalog, quality

_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "robustness.pdf"))
_TOL = 3          # frame tolerance when matching spans
_DATASET = "imperfect_sim"

# a detected type counts for these ground-truth types (and vice versa)
FAMILY = {
    "motor_dropout": {"motor_dropout", "arm_dead"},
    "arm_dead": {"arm_dead", "motor_dropout"},
    "torque_dead": {"torque_dead"},
    "camera_freeze": {"camera_freeze", "camera_blackout"},
    "camera_blackout": {"camera_blackout", "camera_freeze"},
    "low_rate": {"low_rate", "rate_change"},
    "rate_change": {"rate_change", "low_rate"},
    "lag": {"lag"},
    "timing_gap": {"timing_gap"},
    "nan_corrupt": {"nan_corrupt"},
}


def _overlap(a, b):
    """True if defect dicts a,b overlap in time (±tol) and agree in scope."""
    if max(a.get("start", 0), b.get("start", 0)) >= min(a.get("end", 0), b.get("end", 0)) + _TOL:
        return False
    ja, jb = set(a.get("joints", [])), set(b.get("joints", []))
    if ja and jb and not (ja & jb):
        return False
    ca, cb = a.get("cam"), b.get("cam")
    if ca and cb and ca != cb:
        return False
    return True


def _compat(t_gt, t_det):
    return t_det in FAMILY.get(t_gt, {t_gt})


def run(dataset=_DATASET):
    eps = catalog.list_episodes(dataset)
    if not eps:
        raise SystemExit(f"no episodes in {dataset}; run make_imperfect first")
    types = ["motor_dropout", "arm_dead", "torque_dead", "camera_freeze",
             "camera_blackout", "low_rate", "rate_change", "lag", "timing_gap", "nan_corrupt"]
    tp_r = {t: 0 for t in types}; n_gt = {t: 0 for t in types}       # recall
    tp_p = {t: 0 for t in types}; n_det = {t: 0 for t in types}      # precision

    print(f"{'episode':<16}{'quality':>8}{'#GT':>5}{'#det':>5}  matched")
    for ep in eps:
        path = catalog.episode_path(dataset, ep["file"])
        with h5py.File(path, "r") as f:
            gt = json.loads(f.attrs.get("injected_defects_json", "[]"))
        rep = quality.analyze(path)
        det = rep["defects"]
        # recall: each GT defect caught by some compatible overlapping detection?
        caught = 0
        for g in gt:
            n_gt[g["type"]] = n_gt.get(g["type"], 0) + 1
            if any(_compat(g["type"], d["type"]) and _overlap(g, d) for d in det):
                tp_r[g["type"]] += 1; caught += 1
        # precision: each detection backed by some compatible overlapping GT?
        for d in det:
            n_det[d["type"]] = n_det.get(d["type"], 0) + 1
            if any(_compat(g["type"], d["type"]) and _overlap(g, d) for g in gt):
                tp_p[d["type"]] += 1
        print(f"{ep['file']:<16}{rep['quality_score']:>8.3f}{len(gt):>5}{len(det):>5}  {caught}/{len(gt)}")

    print(f"\n{'defect type':<16}{'recall':>10}{'precision':>12}")
    rows = []
    for t in types:
        rec = tp_r[t] / n_gt[t] if n_gt[t] else float("nan")
        prec = tp_p[t] / n_det[t] if n_det[t] else float("nan")
        rows.append((t, rec, prec, n_gt[t], n_det[t]))
        rec_s = f"{rec:.2f}" if n_gt[t] else "  -"
        prec_s = f"{prec:.2f}" if n_det[t] else "  -"
        print(f"{t:<16}{rec_s:>10}{prec_s:>12}   (gt={n_gt[t]}, det={n_det[t]})")
    tot_gt, tot_tpr = sum(n_gt.values()), sum(tp_r.values())
    tot_det, tot_tpp = sum(n_det.values()), sum(tp_p.values())
    print(f"\nOVERALL recall {tot_tpr}/{tot_gt} = {tot_tpr/max(1,tot_gt):.2f}   "
          f"precision {tot_tpp}/{tot_det} = {tot_tpp/max(1,tot_det):.2f}")
    print("note: a dropout injected into a command-idle & torque-nonzero window is "
          "physically indistinguishable from rest and is expected to be missed.")

    # figure: grouped recall/precision bars per type (only types with GT or detections)
    shown = [r for r in rows if r[3] or r[4]]
    labels = [r[0].replace("_", "\n") for r in shown]
    rec = [r[1] if r[3] else 0 for r in shown]
    prec = [r[2] if r[4] else 0 for r in shown]
    x = np.arange(len(shown)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w / 2, rec, w, color="#C2571F", label="recall")
    ax.bar(x + w / 2, prec, w, color="#5B8DB8", label="precision")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(f"Curation detector robustness on {_DATASET} "
                 f"(recall {tot_tpr/max(1,tot_gt):.2f}, precision {tot_tpp/max(1,tot_det):.2f})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); os.makedirs(os.path.dirname(_FIG), exist_ok=True); fig.savefig(_FIG)
    print(f"\nwrote {_FIG}")


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset", default="imperfect_sim"); run(ap.parse_args().dataset)
