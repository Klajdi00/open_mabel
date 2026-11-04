"""
Offline evaluation: every frame of one episode through every chosen variant.

Unlike the live view (latest-wins: a busy worker skips frames), this steps
each KinematicRetargeter through EVERY frame on the episode's own clock, so
the result is deterministic and two variants saw exactly the same input.
Each variant runs in its own spawned process (module-level knobs, see
variants.py), all in parallel.

Output: data/evaluations/<episode>__<stamp>.json
    {episode, variants, frames, t, series{vid:{metric:[...]}}, summary{vid:{metric:{...}}}}
and, with figures=True, vector PDFs (+ PNG previews) next to it.

Summary statistics are taken over ANCHORED frames only (before the engine has
latched its world anchor every map is holding the rest pose, which is a
bring-up behaviour, not retargeting). Per-side metrics pool both hands.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config as CFG
from .worker import METRICS


def _eval_variant(vid, eid, settings=None):
    from . import episodes
    from .worker import make_retargeter, pose_from_frame
    _, frames = episodes.load(eid)
    kr = make_retargeter(vid)
    if settings:
        kr.apply_settings(settings)
    t, anch, rows = [], [], []
    for fr in frames:
        pose = pose_from_frame(fr)
        if pose is None:
            continue
        rec = kr.step(pose, fr["t"], frame=fr)
        t.append(rec["t"])
        anch.append(rec["anchored"])
        rows.append(rec["m"])
    keys = list(rows[0]) if rows else []
    return vid, {"t": t, "anchored": anch,
                 "m": {k: [float(r.get(k, np.nan)) for r in rows] for k in keys}}


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "p50": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max()),
            "sem": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0,
            "n": int(x.size)}


def summarize(res):
    out = {}
    anchored = np.asarray(res["anchored"], bool)
    m = {k: np.asarray(v, float) for k, v in res["m"].items()}
    for k, (_, _, _, per_side) in METRICS.items():
        if per_side:
            if k.startswith("aperture"):
                continue
            both = np.concatenate([m[f"{k}_l"][anchored], m[f"{k}_r"][anchored]])
            out[k] = _stats(both)
            out[k + "_l"] = _stats(m[f"{k}_l"][anchored])
            out[k + "_r"] = _stats(m[f"{k}_r"][anchored])
        else:
            out[k] = _stats(m[k][anchored])
    # grasp: aperture is a CORRELATION, not an error (a frozen hand scores r~0
    # and cannot fake it; an absolute error is dominated by the ORCA hand's
    # 54-97 mm span vs the human's 2-147 mm).
    for s in ("l", "r"):
        h, r = m[f"aperture_h_{s}"][anchored], m[f"aperture_r_{s}"][anchored]
        ok = np.isfinite(h) & np.isfinite(r)
        out[f"grasp_r_{s}"] = ({"mean": float(np.corrcoef(h[ok], r[ok])[0, 1])}
                                  if ok.sum() > 5 and h[ok].std() > 1e-6 and r[ok].std() > 1e-6
                                  else None)
    out["anchored_frac"] = {"mean": float(anchored.mean()) if anchored.size else 0.0}
    return out


def run(eid, vids, figures=True, progress=None, settings=None):
    """`settings` = the studio's switches (scope, robot, torso / neck / lift channels), so an
    ablation's evaluation runs exactly what the live view showed; recorded in the output."""
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=len(vids)) as ex:
        futs = [ex.submit(_eval_variant, v, eid, settings) for v in vids]
        for f in futs:
            vid, r = f.result()
            results[vid] = r
            if progress:
                progress(len(results), len(vids))
    summary = {v: summarize(results[v]) for v in vids}
    CFG.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = eid.replace(":", "_").replace("/", "_")
    out = {"episode": eid, "variants": vids, "created": time.time(), "settings": settings,
           "wall_s": round(time.time() - t0, 1),
           "frames": len(next(iter(results.values()))["t"]) if results else 0,
           "series": results, "summary": summary,
           "metrics": {k: {"label": v[0], "unit": v[1], "higher": v[2], "per_side": v[3]}
                       for k, v in METRICS.items()}}
    path = CFG.EVAL_DIR / f"{name}__{stamp}.json"
    path.write_text(json.dumps(out, allow_nan=True))
    out["file"] = path.name
    if figures:
        try:
            out["figures"] = plot(out, CFG.EVAL_DIR / f"{name}__{stamp}")
        except Exception as e:                   # figures are a bonus, not the result
            out["figures_error"] = str(e)
    return out


def plot(ev, stem, fig_dir=None):
    """Vector PDF figures for one evaluation: per-metric time series + a
    summary bar chart. Returns the written file names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .variants import VARIANTS
    vids = ev["variants"]
    written = []
    show = ["cos_reach", "cos_fore", "place_pct", "orient_deg", "elbow_err", "qvel"]
    fig, axes = plt.subplots(len(show), 1, figsize=(7.0, 1.55 * len(show)), sharex=True)
    for ax, k in zip(axes, show):
        lab, unit, _, per_side = METRICS[k]
        for v in vids:
            s = ev["series"][v]
            y = (np.nanmean([s["m"][k + "_l"], s["m"][k + "_r"]], axis=0)
                 if per_side else np.asarray(s["m"][k], float))
            y = np.where(np.asarray(s["anchored"]), y, np.nan)
            ax.plot(s["t"], y, lw=1.0, color=VARIANTS[v]["color"], label=VARIANTS[v]["label"])
        ax.set_ylabel(f"{lab}\n[{unit}]" if unit else lab, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlabel("episode time [s]", fontsize=8)
    axes[0].legend(fontsize=6, ncol=min(4, len(vids)), loc="lower left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        f = f"{stem}_series.{ext}" if fig_dir is None else f"{fig_dir}/{stem}_series.{ext}"
        fig.savefig(f, dpi=130)
        written.append(str(f).split("/")[-1])
    plt.close(fig)

    bars = ["cos_reach", "cos_upper", "cos_fore", "place_pct", "orient_deg", "elbow_err"]
    fig, axes = plt.subplots(1, len(bars), figsize=(7.0, 2.1))
    for ax, k in zip(axes, bars):
        vals = [(ev["summary"][v].get(k) or {}).get("p50", np.nan) for v in vids]
        p90 = [(ev["summary"][v].get(k) or {}).get("p90", np.nan) for v in vids]
        x = np.arange(len(vids))
        ax.bar(x, vals, color=[VARIANTS[v]["color"] for v in vids], width=0.7)
        ax.scatter(x, p90, marker="_", color="k", s=60, lw=1.2, zorder=3)
        ax.set_xticks(x, [VARIANTS[v]["label"].split(" ")[0] for v in vids],
                      rotation=60, fontsize=6)
        ax.set_title(METRICS[k][0].replace(" error", "\nerror").replace(" cosine", "\ncosine"),
                     fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("median (bar) and p90 (tick), anchored frames, both hands", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        f = f"{stem}_summary.{ext}" if fig_dir is None else f"{fig_dir}/{stem}_summary.{ext}"
        fig.savefig(f, dpi=130)
        written.append(str(f).split("/")[-1])
    plt.close(fig)
    return written


def list_evaluations():
    out = []
    if CFG.EVAL_DIR.exists():
        for f in sorted(CFG.EVAL_DIR.glob("*.json"), reverse=True):
            try:
                d = json.loads(f.read_text())
                out.append({"file": f.name, "episode": d["episode"], "variants": d["variants"],
                            "frames": d["frames"], "created": d["created"]})
            except Exception:
                pass
    return out
