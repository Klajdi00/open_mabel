#!/usr/bin/env python3
"""
start_pose.py — the pose a policy should START from, measured from its training data, written into the run.

    python start_pose.py <path/to/lerobot_dataset> runs/<run>

Writes <run>/start_pose.json: every training episode's first-frame state, summarised as
  * medoid — the REAL recorded start (one episode's frame 0) closest to the per-joint mean, distance measured in
             per-joint std units over the upper body. A real pose, so it is reachable and self-consistent; a per-joint
             mean of differently-shaped starts need not be (the arms' joints are coupled).
  * mean / std / min / max — per joint, for reference.
A deployment driver can glide the upper body to the medoid before the policy takes over, so the first chunk is
predicted from an in-distribution state instead of jumping from wherever the arms happened to be (no robot-side
driver ships with this release; this script only reads data and writes a JSON file).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def upper(name):
    return not name.startswith("lift_") and not name.endswith(("_steer", "_drive"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="the LeRobot dataset the run trained on")
    ap.add_argument("run", type=Path, help="the training run folder (runs/<run>, or a folder holding best/ and last/)")
    a = ap.parse_args()
    cands = [a.run / "checkpoints" / t / "pretrained_model" / "nova_policy.json" for t in ("last", "best")]
    cands += [a.run / "last" / "nova_policy.json", a.run / "best" / "nova_policy.json", a.run / "nova_policy.json"]
    policy = next((p for p in cands if p.exists()), None)
    if policy is None:
        raise SystemExit(f"no nova_policy.json under {a.run}")
    meta = json.loads(policy.read_text())
    names = meta.get("robot_joints", meta["state_names"])   # the dataset's columns (v2 policies merge the lift)
    files = sorted(glob.glob(str(a.dataset / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files])
    first = df[df.frame_index == 0].sort_values("episode_index")
    S = np.stack(first["observation.state"].values).astype(np.float64)
    if S.shape[1] != len(names):
        raise SystemExit(f"state dim {S.shape[1]} != {len(names)} policy state names")
    up = [i for i, n in enumerate(names) if upper(n)]
    mu, sd = S.mean(0), S.std(0)
    z = (S[:, up] - mu[up]) / (sd[up] + 1e-3)
    rms = np.sqrt((z ** 2).mean(1))
    k = int(np.argmin(rms))
    out = {
        "dataset": a.dataset.name, "episodes": int(len(S)), "medoid_episode": int(first.episode_index.values[k]),
        "medoid_rms_z": round(float(rms[k]), 3), "median_rms_z": round(float(np.median(rms)), 3),
        "medoid": {n: float(S[k, i]) for i, n in enumerate(names) if upper(n)},
        "mean": {n: float(mu[i]) for i, n in enumerate(names)},
        "std": {n: float(sd[i]) for i, n in enumerate(names)},
        "min": {n: float(S[:, i].min()) for i, n in enumerate(names)},
        "max": {n: float(S[:, i].max()) for i, n in enumerate(names)},
    }
    dst = a.run / "start_pose.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"{dst}: medoid = episode {out['medoid_episode']} of {len(S)} (rms z {rms[k]:.2f}, median {np.median(rms):.2f})")


if __name__ == "__main__":
    main()
