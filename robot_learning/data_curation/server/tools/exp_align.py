"""
Proof for the robust Align-to-pixels estimator: inject a KNOWN camera↔proprio shift
on a clean episode and check the multi-window median recovers it.

Run:
    cd learning
    python -m data_curation.server.tools.exp_align
Writes docs/figures/align_recovery.pdf.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import catalog, quality, h5pool

_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "align_recovery.pdf"))


def _shift(a, m):
    T = a.shape[0]; return a[np.clip(np.arange(T) - m, 0, T - 1)]


def _robust_lag(ce, pe, K, nwin=6):
    W = len(ce); seg = max(W // nwin, K * 3); ests = []
    for s in range(0, W, seg):
        r = quality._xcorr_lag(ce[s:s + seg], pe[s:s + seg], K)
        if r and r[1] >= 0.30:
            ests.append(r[0])
    g = quality._xcorr_lag(ce, pe, K)                    # global (full-signal) fallback
    if len(ests) >= 3:
        return int(round(float(np.median(ests))))
    return g[0] if g else 0


def run(dataset=None, episode=None, sweep=range(-1, 7), K=8):
    """Validate the estimator on realistically-COUPLED data: a real episode's EE-speed
    profile provides the motion; the 'camera' sees the same motion + sensor noise
    (camera_motion ≈ EE-speed + noise, which is exactly the physical coupling on real
    teleop). Inject a known shift and recover it across noise levels."""
    if dataset is None:
        cat = [d for d in catalog.full_catalog()["datasets"] if "sim" not in d["name"]][0]
        dataset, episode = cat["name"], cat["episodes"][0]["file"]
    path = catalog.episode_path(dataset, episode)
    with h5pool.reading(path) as f:
        fps = quality._fps(f); T = int(f["obs/joint_vel"].shape[0])
        pe = quality._ee_speed(f, T)
    pe = (pe - pe.mean()) / (pe.std() + 1e-9)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    lo, hi = min(sweep) - 0.5, max(sweep) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="#9a938c", lw=1, label="ideal")
    colors = {0.2: "#2E7D5B", 0.5: "#C2571F", 1.0: "#5B8DB8"}
    print(f"episode {dataset}/{episode}  fps={fps:.1f}  (coupled camera = EE-speed + noise)")
    for snr in (0.2, 0.5, 1.0):                          # camera noise σ (rel. to signal)
        ce = pe + snr * rng.standard_normal(T)           # camera sees EE motion + noise
        base = _robust_lag(ce, pe, K)
        inj, rec = [], []
        for m in sweep:
            lag = _robust_lag(ce, _shift(pe, m), K); inj.append(m); rec.append(lag - base)
        inj = np.array(inj, float); rec = np.array(rec, float); err = rec - inj
        rms = float(np.sqrt((err ** 2).mean()))
        print(f"  noise σ={snr:>3}: RMS {rms:.2f}f, |err|<=1 in {int((np.abs(err)<=1).sum())}/{len(err)}")
        ax.plot(inj, rec, "o-", color=colors[snr], lw=1.3, ms=4, label=f"noise σ={snr} (RMS {rms:.1f}f)")
    ax.set_xlabel("injected shift (frames)"); ax.set_ylabel("recovered (frames)")
    ax.set_title(f"Align-to-pixels recovery vs camera noise — {dataset}")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.dirname(_FIG), exist_ok=True); fig.savefig(_FIG)
    print(f"wrote {_FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset"); ap.add_argument("--episode")
    a = ap.parse_args(); run(a.dataset, a.episode)
