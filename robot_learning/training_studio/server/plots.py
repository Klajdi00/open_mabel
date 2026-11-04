"""plots.py — server-side rendering of a run's training curves to PNGs.

The studio dashboard draws charts client-side (live), but a finished run should
carry durable, shareable plots on disk. This reads the run's `train/metrics.jsonl`
(the StudioLogger contract: one JSON record/line, `_type` in config/metric/checkpoint)
and writes PNGs into the run's `plots/` dir. Pure matplotlib (Agg), no display.

Rendered (whichever series exist): loss, sub-losses (l1/kld/…), learning rate,
grad norm, throughput, and GPU/CPU utilisation (from the SQLite `system` rows if a
metrics_store is passed). Missing series are simply skipped — never raises.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(metrics_path):
    """Return {series_name: [(step, value), …]} from metrics.jsonl."""
    series: dict[str, list] = {}
    if not os.path.exists(metrics_path):
        return series
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("_type") not in (None, "metric"):
                continue
            step = rec.get("step") or rec.get("_step")
            if step is None:
                continue
            for k, v in rec.items():
                if k in ("_type", "step", "_step", "mode", "_mode"):
                    continue
                if isinstance(v, (int, float)):
                    series.setdefault(k, []).append((step, v))
    for k in series:
        series[k].sort(key=lambda t: t[0])
    return series


_LOSS_KEYS = ("loss", "train_loss", "l1_loss", "kld_loss", "l2_loss",
              "mse_loss", "diffusion_loss", "val_loss", "eval_loss")
_LR_KEYS = ("lr", "learning_rate")
_GRAD_KEYS = ("grad_norm", "grad")
_TP_KEYS = ("steps_per_sec", "samples_per_sec", "throughput", "dataloading_s")


def _plot(series, keys, path, title, ylabel, logy=False):
    present = [k for k in keys if series.get(k)]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    for k in present:
        xs, ys = zip(*series[k])
        ax.plot(xs, ys, label=k, linewidth=1.6)
    ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if len(present) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def render_run_plots(layout, metrics_store=None, run_id=None):
    """Render all available plots into layout.plots_dir. Returns list of PNG paths."""
    os.makedirs(layout.plots_dir, exist_ok=True)
    series = _read_metrics(layout.metrics_path)
    out = []
    def add(p):
        if p:
            out.append(p)
    add(_plot(series, _LOSS_KEYS, os.path.join(layout.plots_dir, "loss.png"),
              "Training loss", "loss", logy=True))
    add(_plot(series, _LR_KEYS, os.path.join(layout.plots_dir, "lr.png"),
              "Learning rate", "lr"))
    add(_plot(series, _GRAD_KEYS, os.path.join(layout.plots_dir, "grad_norm.png"),
              "Gradient norm", "‖grad‖"))
    add(_plot(series, _TP_KEYS, os.path.join(layout.plots_dir, "throughput.png"),
              "Throughput", "rate"))

    # system utilisation from SQLite (sysmon writes system.* rows directly)
    if metrics_store is not None and run_id is not None:
        try:
            sysrows = metrics_store.get_series(run_id, mode="system")
            sysser: dict[str, list] = {}
            for r in sysrows:
                sysser.setdefault(r["name"], []).append((r["step"], r["value"]))
            for k in sysser:
                sysser[k].sort(key=lambda t: t[0])
            add(_plot(sysser, ("system/gpu_util", "system/cpu_util", "system/gpu_mem_used_gb"),
                      os.path.join(layout.plots_dir, "system.png"),
                      "System utilisation", "%/GB"))
        except Exception:
            pass
    return out
