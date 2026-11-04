#!/usr/bin/env python3
"""
Timeline-sync diagnostic. Answers: what is the trajectory's sync timestep, and is
every stream index-aligned to the one timeline clock? For each episode it prints
the clock (fps + dt), whether time/sim is monotonic & uniform, and whether every
stream — trajectory pose/* included — has the same frame count as the timeline
(index-aligned ⇒ automatically synced). Then it folds in the measured skew from
quality.analyze (systematic lag / drift / timing gaps).

Run:  python -m data_curation.server.tools.sync_report [dataset ...]
"""
import glob
import os
import sys

import numpy as np

from .. import config as CFG
from .. import h5pool, quality

STREAMS = [
    ("proprio joint_pos", "obs/joint_pos"),
    ("proprio joint_vel", "obs/joint_vel"),
    ("proprio joint_torque", "obs/joint_torque"),
    ("action ctrl", "action/ctrl"),
    ("base_twist", "obs/base_twist"),
    ("teleop base_cmd", "teleop/base_cmd"),
    ("lidar", "obs/lidar"),
]


def _report_episode(path):
    with h5pool.reading(path) as f:
        fps = quality._fps(f)
        dt = 1.0 / fps if fps else float("nan")
        T = (f["obs/joint_pos"].shape[0] if "obs/joint_pos" in f
             else int(f.attrs.get("num_frames", 0)))
        print(f"\n  ── {os.path.basename(path)} ──")
        print(f"  clock:  fps = {fps:.4f} Hz   dt (sync timestep) = {dt*1000:.2f} ms   T = {T} frames")

        # time/sim monotonic + uniform?
        if "time/sim" in f:
            ts = np.asarray(f["time/sim"][:], dtype=np.float64)
            d = np.diff(ts)
            mono = bool(np.all(d >= -1e-9))
            print(f"  time/sim: {'monotonic ✓' if mono else 'NON-MONOTONIC ✗'}   "
                  f"mean Δ = {d.mean()*1000:.2f} ms   std Δ = {d.std()*1000:.2f} ms "
                  f"(uniform ⇒ single clock)")
        else:
            print("  time/sim: absent (frame index IS the clock)")

        # per-stream index alignment vs the timeline T
        print("  stream alignment (frames vs timeline T):")
        aligned = misaligned = 0
        for label, key in STREAMS:
            if key not in f:
                continue
            n = f[key].shape[0]
            ok = (n == T)
            aligned += ok; misaligned += (not ok)
            print(f"     {'OK ' if ok else 'MIS'}  {label:<22} {n:>5} f  {'= T (synced)' if ok else f'≠ T ({n-T:+d})'}")

        # trajectory pose streams (what the Trajectory button traces)
        print("  trajectory streams (pose/* — traced by the Trajectory button):")
        if "obs/pose" in f:
            for tf in f["obs/pose"]:
                n = f["obs/pose/" + tf].shape[0]
                ok = (n == T)
                aligned += ok; misaligned += (not ok)
                print(f"     {'OK ' if ok else 'MIS'}  pose/{tf:<17} {n:>5} f  "
                      f"{'= T ⇒ trajectory[i] pairs with frame i' if ok else f'≠ T ({n-T:+d})'}")
        else:
            print("     (no obs/pose group)")

        # cameras
        print("  camera streams (index-aligned to the same frame):")
        if "obs/cam" in f:
            for cam in f["obs/cam"]:
                if "rgb" not in f["obs/cam/" + cam]:
                    continue
                n = f["obs/cam/" + cam + "/rgb"].shape[0]
                ok = (n == T)
                aligned += ok; misaligned += (not ok)
                print(f"     {'OK ' if ok else 'MIS'}  cam/{cam:<18} {n:>5} f  {'= T' if ok else f'≠ T ({n-T:+d})'}")

    # measured temporal skew from the sync engine
    q = quality.analyze(path)
    fl = q.get("flags", {})
    h = q.get("headline") or {}
    print("  measured skew (sync engine):")
    print(f"     systematic action↔obs lag : {'YES ' + str(h.get('lag_frames')) + 'f' if fl.get('systematic_lag') else 'none (≤ tol)'}")
    print(f"     clock drift               : {'YES ' + str(h.get('drift_per_kframe')) + ' f/kframe' if fl.get('clock_drift') else 'none'}")
    print(f"     timing gaps               : {q.get('gaps',{}).get('n_gaps',0)}")
    print(f"     quality score             : {q.get('quality_score')}")
    return aligned, misaligned


def main():
    names = sys.argv[1:] or ["my_demo", "multirate_sim", "stress_sim"]
    print("=" * 74)
    print("TIMELINE-SYNC REPORT — trajectory sync timestep + stream alignment")
    print("=" * 74)
    print("Model: ONE global timeline frame → seq.resolve → source frame `sf` is fed")
    print("to EVERY consumer (robot, joints, cameras, lidar, trajectory, plots). The")
    print("Trajectory button has NO separate timestep: it traces pose/<tf>[0..sf] on")
    print("the same clock. dt = 1/fps. modOffset (Sync / Align-to-pixels) is applied")
    print("at Save/Export, not in the live view.")
    tot_ok = tot_bad = 0
    for name in names:
        eps = sorted(glob.glob(os.path.join(CFG.DATASET_ROOT, name, "episode_*.h5")))
        if not eps:
            print(f"\n[{name}] no episodes"); continue
        print(f"\n########## dataset: {name}  ({len(eps)} episodes, showing 1) ##########")
        a, b = _report_episode(eps[0]); tot_ok += a; tot_bad += b
    print("\n" + "=" * 74)
    print(f"VERDICT: {tot_ok} streams index-aligned to the timeline, {tot_bad} misaligned.")
    print("Index-aligned ⇒ automatically timeline-synced by construction.")
    print("=" * 74)


if __name__ == "__main__":
    main()
