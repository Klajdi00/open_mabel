#!/usr/bin/env python3
"""
deploy_report.py — compare deployment runs (per-step rollout logs) with each other and with the demos.

    python deploy_report.py <deploy_runs>/*.npz
    python deploy_report.py <logs...> --dataset <path/to/lerobot_dataset>       # + the demo reference row
    python deploy_report.py <logs...> --csv report.csv

A log is an .npz with `t` (s), `joints` (names), `q` (measured), `des` (policy target) and optionally `config`
(JSON string) or a sibling .json with {"config": {...}}; write one from whatever loop runs train.Runtime (sim or
robot). Offline analysis only.

Every number is computed from the logged control steps (30 Hz): measured joints q, the policy target `des` (after any
--smooth-hz), and the rate-limited command `sent`.

  stuck %        1-s windows in which no arm/wrist joint moved more than 0.02 rad
  arm path       total arm+wrist joint travel, rad per minute (how much the robot actually does)
  arm v p90      90th percentile measured arm joint speed, rad/s (demos: the same statistic on the actions)
  target v p90   90th percentile speed of the policy's own arm target, rad/s (slow here = the policy is slow,
                 not the robot)
  grasp rev/min  direction reversals (> 0.08 rad, hysteresis) of the right hand's mean finger flexion target —
                 open/close dithering; compare with the demos' rate
  closed %       time the right hand is measured closed (mean flexion > 0.8 rad)
  HF arm/finger  fraction of target motion power above 3 Hz (jitter; demos have ~0.1 %)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

FPS = 30.0
FLEX = ("index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip")


def groups(J):
    return {"arm": [i for i, j in enumerate(J) if "_arm_" in j or j.endswith("_wrist")],
            "finger": [i for i, j in enumerate(J) if any(k in j for k in ("thumb", "index", "middle", "ring", "pinky"))]}


def closure(X, J, side="right"):
    idx = [J.index(f"{side}_{f}") for f in FLEX if f"{side}_{f}" in J]
    return X[:, idx].mean(1) if idx else np.zeros(len(X))


def reversals(c, thr=0.08):
    n, ext, d = 0, c[0], 0
    for x in c[1:]:
        if d >= 0 and x < ext - thr:
            n += d > 0; d, ext = -1, x
        elif d <= 0 and x > ext + thr:
            n += d < 0; d, ext = 1, x
        else:
            ext = max(ext, x) if d >= 0 else min(ext, x)
    return n


def hf(X, fc=3.0):
    if len(X) < 64:
        return np.nan
    X = X - X.mean(0)
    P = np.abs(np.fft.rfft(X * np.hanning(len(X))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(X), 1 / FPS)
    tot = P[f > 0.05].sum(0)
    m = tot > 1e-6
    return float(np.median(P[f > fc][:, m].sum(0) / tot[m])) if m.any() else np.nan


def p90_speed(X, idx, dt):
    return float(np.percentile(np.abs(np.diff(X[:, idx], axis=0)) / dt, 90))


def config_of(path, d):
    for src in (Path(path).with_suffix(".json"),):
        if src.exists():
            c = json.loads(src.read_text()).get("config", {})
            break
    else:
        c = json.loads(str(d["config"])) if "config" in d else {}
    pol = Path(str(c.get("policy", "?"))).parent.name if c.get("policy") else "?"
    bits = [pol, f"rate {c.get('max_rate', '?')}", f"exec {c.get('replan_s', 0) or 'cont'}"]
    if c.get("smooth_hz"):
        bits.append(f"lp {c['smooth_hz']:g}Hz")
    if c.get("noise_seed") is not None:
        bits.append(f"seed {c['noise_seed']}")
    return " · ".join(bits)


def run_row(path):
    d = np.load(path, allow_pickle=False)
    J = [str(j) for j in d["joints"]]
    t, q, des = d["t"], d["q"].astype(float), d["des"].astype(float)
    G = groups(J)
    dt = np.diff(t)[:, None]
    minutes = (t[-1] - t[0]) / 60
    w = int(FPS)
    moves = [np.abs(q[k + w, G["arm"]] - q[k, G["arm"]]).max() for k in range(0, len(t) - w, w)]
    c_t, c_q = closure(des, J), closure(q, J)
    return {"run": Path(path).stem, "config": config_of(path, d), "minutes": round(minutes, 2),
            "stuck_pct": 100 * float(np.mean(np.array(moves) < 0.02)) if moves else np.nan,
            "arm_path_rad_min": float(np.abs(np.diff(q[:, G["arm"]], axis=0)).sum() / minutes),
            "arm_v_p90": p90_speed(q, G["arm"], dt), "target_v_p90": p90_speed(des, G["arm"], dt),
            "grasp_rev_min": reversals(c_t) / minutes, "closed_pct": 100 * float(np.mean(c_q > 0.8)),
            "hf_arm_pct": 100 * hf(des[:, G["arm"]]), "hf_finger_pct": 100 * hf(des[:, G["finger"]])}


def demo_row(dataset, names):
    import pandas as pd
    files = sorted(glob.glob(str(Path(dataset).expanduser() / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    G = groups(names)
    rev, mins, v, hfa, hff, closed = 0, 0.0, [], [], [], []
    for _, g in df.groupby("episode_index"):
        A = np.stack(g.action.values).astype(float)[:, :len(names)]
        mins += len(A) / FPS / 60
        rev += reversals(closure(A, names))
        closed.append(closure(A, names) > 0.8)
        v.append(np.abs(np.diff(A[:, G["arm"]], axis=0)) * FPS)
        if len(A) > 120:
            hfa.append(hf(A[:, G["arm"]])); hff.append(hf(A[:, G["finger"]]))
    v = np.concatenate(v)
    return {"run": "DEMOS", "config": Path(dataset).name, "minutes": round(mins, 1), "stuck_pct": np.nan,
            "arm_path_rad_min": np.nan, "arm_v_p90": float(np.percentile(v, 90)), "target_v_p90": float(np.percentile(v, 90)),
            "grasp_rev_min": rev / mins, "closed_pct": 100 * float(np.mean(np.concatenate(closed))),
            "hf_arm_pct": 100 * float(np.nanmedian(hfa)), "hf_finger_pct": 100 * float(np.nanmedian(hff))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--dataset", help="LeRobot dataset the policy trained on: adds the demo reference row")
    ap.add_argument("--csv")
    a = ap.parse_args()
    paths = [p for p in a.logs if p.endswith(".npz") and not p.endswith(".tmp.npz")]
    rows = []
    for p in sorted(paths):
        try:
            rows.append(run_row(p))
        except Exception as e:
            print(f"skip {p}: {e}")
    if a.dataset and paths:
        rows.insert(0, demo_row(a.dataset, [str(j) for j in np.load(paths[0])["joints"]]))
    cols = [("run", 26, "s"), ("config", 40, "s"), ("minutes", 5, ".1f"), ("stuck_pct", 6, ".0f"),
            ("arm_path_rad_min", 8, ".1f"), ("arm_v_p90", 7, ".3f"), ("target_v_p90", 8, ".3f"),
            ("grasp_rev_min", 8, ".1f"), ("closed_pct", 7, ".0f"), ("hf_arm_pct", 7, ".2f"), ("hf_finger_pct", 7, ".2f")]
    head = ["run", "config", "min", "stuck%", "path/min", "arm v90", "tgt v90", "grasp/min", "closed%", "HFarm%", "HFfng%"]
    print("  ".join(f"{h:>{w}s}" if i > 1 else f"{h:<{w}s}" for i, (h, (_, w, _)) in enumerate(zip(head, cols))))
    for r in rows:
        out = []
        for i, (k, w, f) in enumerate(cols):
            v = r[k]
            s = (f"{v:{f}}" if not (isinstance(v, float) and np.isnan(v)) else "–") if f != "s" else str(v)[:w]
            out.append(f"{s:<{w}s}" if i < 2 else f"{s:>{w}s}")
        print("  ".join(out))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
