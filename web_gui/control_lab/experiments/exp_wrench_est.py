#!/usr/bin/env python3
"""
Rigorous ground-truth validation of the external-wrench estimator
(momentum observer -> friction compensation -> hypothesis-scoring surface
localization) against the MuJoCo simulator.

    python3 web_control_gui/control_lab/experiments/exp_wrench_est.py

Protocol: the SAME SimWorld the studio serves (controllers bench, admittance
layer on), stepped synchronously for determinism. Each case applies a TRUE
point force at a known SURFACE point (entry of the force line through the
body CoM, applied as an equivalent xfrc force+moment), holds it 2.5 s,
and measures over the final 1.0 s steady window:

  |F|_err   force magnitude error [N] (mean over window)
  cos       direction cosine est vs true (mean)
  pt_err    ||p_est - p_true|| [cm] (mean)
  surf      distance from p_est to the arm surface [cm] (ray probe; the
            estimator's contract is that the point lies ON the robot skin)
  jit_F     steady-state force jitter, std ||F_est|| [N]
  jit_p     steady-state point jitter, std ||p_est|| [cm]
  link      estimated contact link (vs the true body)

Plus two negative controls the estimator must NOT fire on:
  rest      2 s at rest, no push            -> no contact
  post-push 2 s after a 12 N push released  -> no contact (friction
            hysteresis leaves a standing sub-fl residual; see
            friction_compensate's docstring)
and a moving-arm phantom check (sine trajectory, no push).

Figures (vector PDF, docs/figures/):
  lab_wrench_sweep.pdf    per-case bars: |F| est vs true, point error, cos
  lab_wrench_traces.pdf   time traces of one palm + one forearm case:
                          |F_est| vs |F_true|, point error, contact on/off
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
os.environ.setdefault("MABEL_CTS_STATE_FILE",
                      "/tmp/mabel_control_lab_EXP_state.f64")

from control_lab.server.simworld import SimWorld              # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(_HERE))), "controller"))
from mabel.whole_body_controller.admittance.wrench_estimation import (  # noqa: E402
    _LEFT_ARM, _RIGHT_ARM, _arm_bodies, _ray_bodies)

FIGDIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"true": "#2ca02c", "est": "#d62728", "pt": "#1f77b4"}

CASES = [  # (body, force [N], label)
    ("left_palm",  [0, 12, 0],   "palm +Y 12N"),
    ("left_palm",  [0, -12, 0],  "palm -Y 12N"),
    ("left_palm",  [0, 0, -12],  "palm -Z 12N"),
    ("left_palm",  [0, -6, 0],   "palm -Y 6N"),
    ("left_palm",  [0, 14, -14], "palm diag 20N"),
    ("right_palm", [0, -12, 0],  "R palm -Y 12N"),
    ("l_forearm",  [0, 10, 0],   "forearm +Y 10N"),
    ("l_forearm",  [0, 7, -7],   "forearm diag 10N"),
    ("l_elbow",    [0, 10, 0],   "elbow +Y 10N"),
    # +X was structurally unobservable under the OLD pipeline (line ∥ distal
    # axes, couple-soaked DLS); friction compensation + the pure-force refit
    # recover it in the servo-stiff regime — kept in the sweep as proof.
    ("left_palm",  [12, 0, 0],   "palm +X 12N"),
]

np.random.seed(0)


def surf_dist(w, pt, arm):
    """Min distance from pt to the arm's geom surfaces (26-direction ray probe)."""
    dirs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x or y or z:
                    v = np.array([x, y, z], float)
                    dirs.append(v / np.linalg.norm(v))
    best = np.inf
    bodies = _arm_bodies(w.model, arm)
    for d in dirs:
        h = _ray_bodies(w.model, w.data, bodies, np.asarray(pt) - d * 0.25, d)
        if h >= 0:
            best = min(best, abs(h - 0.25))
    return best


def read_est(w):
    ov = w.snapshot()["overlay"].get("wrench") or {}
    return ov.get("est")


def run_case(w, body, F, sample_dt=0.05, hold=2.5, window=1.0):
    w.apply_push(body, F, seconds=hold)
    w.step_sync(hold - window)
    Fs, Ps, Bs, PtErr = [], [], [], []
    for i in range(int(window / sample_dt)):
        w.step_sync(sample_dt)
        ov = w.snapshot()["overlay"].get("wrench") or {}
        e, tw = ov.get("est"), ov.get("true")
        if e and tw:
            Fs.append(e["force"]); Ps.append(e["point"]); Bs.append(e["body"])
            # score against the LIVE truth (the arm deflects under the push)
            PtErr.append(np.linalg.norm(np.asarray(e["point"])
                                        - np.asarray(tw["point"])))
    w.step_sync(1.5)                        # release + settle before next case
    return np.array(PtErr), np.array(Fs), np.array(Ps), Bs


def main():
    w = SimWorld()
    w.settle()
    # torque ON + a WBC method (joint/impedance) mounts the momentum-observer
    # contact estimator; the arm holds stiff so a push is localised + estimated.
    w.set_torque(True)
    w.set_control_spec(method="joint", control_type="impedance", region="arm",
                       stiffness=45.0)
    w.step_sync(1.0)

    rows = []
    print(f"\n{'case':>26} | {'|F̂|':>5} {'|F|':>4} {'cos':>5} {'pt_err':>6} "
          f"{'surf':>5} {'jit_F':>5} {'jit_p':>6}  link (true)")
    for body, F, label in CASES:
        PtErr, Fs, Ps, Bs = run_case(w, body, F)
        Ft = np.asarray(F, float)
        if len(Fs) == 0:
            print(f"{label:>26} |   --- no contact detected ---")
            rows.append(dict(label=label, det=False, Ft=np.linalg.norm(Ft)))
            continue
        Fm, Pm = Fs.mean(0), Ps.mean(0)
        arm = _LEFT_ARM if body.startswith(("l", "left")) else _RIGHT_ARM
        row = dict(
            label=label, det=True, body=body, link=max(set(Bs), key=Bs.count),
            Ft=float(np.linalg.norm(Ft)), Fh=float(np.linalg.norm(Fm)),
            cos=float(Fm @ Ft / (np.linalg.norm(Fm) * np.linalg.norm(Ft))),
            pt=float(PtErr.mean()) * 100,
            surf=float(surf_dist(w, Pm, arm)) * 100,
            jf=float(np.linalg.norm(Fs.std(0))),
            jp=float(np.linalg.norm(Ps.std(0))) * 100)
        rows.append(row)
        print(f"{label:>26} | {row['Fh']:5.1f} {row['Ft']:4.0f} {row['cos']:5.2f} "
              f"{row['pt']:5.1f}cm {row['surf']:4.1f}cm {row['jf']:5.2f} "
              f"{row['jp']:5.2f}cm  {row['link']} ({body})")

    # ── negative controls ────────────────────────────────────────────────
    print("\nnegative controls (must be contact-free):")
    controls = {}
    w.step_sync(1.0)
    n = sum(1 for _ in range(40) if (w.step_sync(0.05) or read_est(w)))
    controls["rest"] = n
    print(f"  rest 2s:                {n}/40 samples with contact")
    w.apply_push("left_palm", [0, 12, 0], seconds=1.0)
    w.step_sync(1.6)                       # push over; hysteresis remains
    n = sum(1 for _ in range(40) if (w.step_sync(0.05) or read_est(w)))
    controls["post"] = n
    print(f"  post-push 2s:           {n}/40 samples with contact")
    w.set_trajectory("sine", joint="left_arm_2", amp=0.3, period=2.5)
    w.step_sync(1.0)
    n = sum(1 for _ in range(40) if (w.step_sync(0.05) or read_est(w)))
    controls["moving"] = n
    print(f"  sine motion (no push):  {n}/40 samples with contact")
    w.set_trajectory("hold")

    # ── figures ──────────────────────────────────────────────────────────
    det = [r for r in rows if r.get("det")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.arange(len(det))
    labels = [r["label"] for r in det]
    axes[0].bar(x - 0.2, [r["Ft"] for r in det], 0.4,
                color=COL["true"], label="true")
    axes[0].bar(x + 0.2, [r["Fh"] for r in det], 0.4,
                color=COL["est"], label="estimated")
    axes[0].errorbar(x + 0.2, [r["Fh"] for r in det],
                     yerr=[r["jf"] for r in det], fmt="none", ecolor="k",
                     capsize=2, lw=0.8)
    axes[0].set_ylabel("|F| [N]"); axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.2, [r["pt"] for r in det], 0.4,
                color=COL["pt"], label="point error")
    axes[1].bar(x + 0.2, [r["surf"] for r in det], 0.4,
                color="#9467bd", label="dist. to surface")
    axes[1].set_ylabel("[cm]"); axes[1].legend(fontsize=8)
    axes[2].bar(x, [r["cos"] for r in det], 0.5, color="#e9a679")
    axes[2].set_ylabel("direction cosine"); axes[2].set_ylim(-1.05, 1.05)
    axes[2].axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    fig.suptitle("Wrench estimator vs MuJoCo ground truth "
                 "(steady window, surface-point true forces)", fontsize=10)
    _save(fig, "lab_wrench_sweep.pdf")

    # traces: one palm + one forearm case
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex="col")
    for col, (body, F, label) in enumerate(
            (("left_palm", [0, 12, 0], "palm +Y 12N"),
             ("l_forearm", [0, 10, 0], "forearm +Y 10N"))):
        w.step_sync(1.0)
        t0 = w._t
        w.apply_push(body, F, seconds=2.0)
        ts, fs, ps = [], [], []
        for _ in range(int(3.0 / 0.02)):
            w.step_sync(0.02)
            ov = w.snapshot()["overlay"].get("wrench") or {}
            e, tw = ov.get("est"), ov.get("true")
            ts.append(w._t - t0)
            fs.append(np.linalg.norm(e["force"]) if e else np.nan)
            ps.append(np.linalg.norm(np.asarray(e["point"])
                                     - np.asarray(tw["point"])) * 100
                      if e and tw else np.nan)
        axes[0][col].plot(ts, fs, color=COL["est"], label="|F̂|")
        axes[0][col].axhline(np.linalg.norm(F), color=COL["true"], ls="--",
                             label="|F| true")
        axes[0][col].axvspan(0, 2.0, color=COL["true"], alpha=0.06)
        axes[0][col].set_title(label, fontsize=9)
        axes[0][col].set_ylabel("|F| [N]"); axes[0][col].legend(fontsize=8)
        axes[1][col].plot(ts, ps, color=COL["pt"])
        axes[1][col].set_ylabel("point error [cm]")
        axes[1][col].set_xlabel("t [s]  (shaded = push active)")
    fig.suptitle("Estimator transient + steady behaviour (contact gaps = "
                 "correctly dropped)", fontsize=10)
    _save(fig, "lab_wrench_traces.pdf")

    # summary the paper cites
    palm = [r for r in det if "palm" in r["label"]]
    prox = [r for r in det if "forearm" in r["label"] or "elbow" in r["label"]]
    print("\nsummary (paper numbers):")
    if palm:
        print(f"  palm cases (n={len(palm)}): |F| err "
              f"{np.mean([abs(r['Fh']-r['Ft']) for r in palm]):.1f} N, cos "
              f"{np.mean([r['cos'] for r in palm]):.3f}, point "
              f"{np.mean([r['pt'] for r in palm]):.1f} cm, surface "
              f"{np.mean([r['surf'] for r in palm]):.1f} cm, jitter "
              f"{np.mean([r['jp'] for r in palm]):.2f} cm")
    if prox:
        print(f"  forearm/elbow (n={len(prox)}): |F| err "
              f"{np.mean([abs(r['Fh']-r['Ft']) for r in prox]):.1f} N, cos "
              f"{np.mean([r['cos'] for r in prox]):.3f}, point "
              f"{np.mean([r['pt'] for r in prox]):.1f} cm, surface "
              f"{np.mean([r['surf'] for r in prox]):.1f} cm")
    print(f"  negative controls: rest {controls['rest']}/40, post-push "
          f"{controls['post']}/40, moving {controls['moving']}/40 "
          f"(all should be 0)")
    return rows, controls


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name))
    plt.close(fig)
    print(f"  figure -> docs/figures/{name}")


if __name__ == "__main__":
    main()
