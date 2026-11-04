"""
Synthetic-skew recovery experiment for the sync engine (Phase 1 proof).

Take a clean MABEL episode, inject a KNOWN temporal shift m into the command
stream (action/ctrl), and check the estimator recovers m from the command-vs-
state cross-correlation. This is the evidence that the "detect misalignment"
guarantee actually works: recovered offset should track the injected shift 1:1.

Run:
    cd learning
    python -m data_curation.server.tools.exp_sync                # default sample
    python -m data_curation.server.tools.exp_sync --dataset my_demo --episode episode_0000.h5

Writes docs/figures/sync_recovery.pdf (vector) and prints a metrics table.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")                       # headless, per research workflow
import matplotlib.pyplot as plt
import numpy as np

from .. import catalog, quality, h5pool

_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "sync_recovery.pdf"))


def _shift(arr, m):
    """Delay a stream by m frames (edge-held): out[t] = arr[t - m]. Positive m models
    the realistic bug -- a command stream logged m frames LATE (state lags command)."""
    T = arr.shape[0]
    idx = np.clip(np.arange(T) - m, 0, T - 1)
    return arr[idx]


def _measure_lag(cmd, state, K):
    """Estimator lag (frames) + confidence for a command/state pair, reusing the
    exact quality.py motion-energy + cross-correlation."""
    a = quality._motion_energy(cmd, True)       # command velocity (d/dt)
    b = quality._motion_energy(state, False)    # measured velocity
    r = quality._xcorr_lag(a, b, K)
    return (None, 0.0) if r is None else r


def run(dataset, episode, sweep=range(-1, 7), K=8):     # causal regime: state lags command (lag>=0)
    path = catalog.episode_path(dataset, episode)
    with h5pool.reading(path) as f:
        cmd0 = np.asarray(f["action/ctrl"][:], dtype=np.float64)
        state = np.asarray(f["obs/joint_vel"][:], dtype=np.float64)
        fps = quality._fps(f)

    base_lag, base_conf = _measure_lag(cmd0, state, K)
    rows, injected, recovered = [], [], []
    for m in sweep:
        lag, conf = _measure_lag(_shift(cmd0, m), state, K)
        rec = base_lag - lag                    # recovered shift (see quality.py xcorr sign)
        rows.append((m, lag, rec, conf))
        injected.append(m); recovered.append(rec)

    injected = np.array(injected, float); recovered = np.array(recovered, float)
    err = recovered - injected
    bias_f, rms_f = float(err.mean()), float(np.sqrt((err ** 2).mean()))
    ms = 1000.0 / fps

    print(f"episode: {dataset}/{episode}   fps={fps:.2f}   baseline_lag={base_lag} (conf={base_conf:.3f})")
    print(f"{'injected':>8} {'meas_lag':>8} {'recovered':>9} {'err':>5} {'conf':>6}")
    for m, lag, rec, conf in rows:
        print(f"{m:>8d} {lag:>8d} {rec:>9d} {rec - m:>5d} {conf:>6.3f}")
    print(f"\nrecovery bias = {bias_f:+.2f} frames ({bias_f*ms:+.1f} ms)   "
          f"RMS = {rms_f:.2f} frames ({rms_f*ms:.1f} ms)   |err|<=1 in "
          f"{int((np.abs(err)<=1).sum())}/{len(err)} cases")

    # figure: recovered vs injected (ideal = y=x)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo, hi = injected.min() - 0.5, injected.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="#9a938c", lw=1, label="ideal (y=x)")
    ax.plot(injected, recovered, "o-", color="#C2571F", lw=1.4, ms=5, label="estimator")
    ax.set_xlabel("injected shift (frames)")
    ax.set_ylabel("recovered offset (frames)")
    ax.set_title(f"Sync recovery — {dataset}/{episode}\nbias {bias_f:+.2f}f, RMS {rms_f:.2f}f ({rms_f*ms:.0f} ms)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(_FIG), exist_ok=True)
    fig.savefig(_FIG)
    print(f"\nwrote {_FIG}")
    return {"bias_frames": bias_f, "rms_frames": rms_f, "fps": fps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--episode", default=None)
    a = ap.parse_args()
    ds, ep = a.dataset, a.episode
    if ds is None or ep is None:
        cat = catalog.full_catalog()["datasets"][0]
        ds, ep = cat["name"], cat["episodes"][0]["file"]
    run(ds, ep)
