"""
StudioLogger — a drop-in replacement for LeRobot's ``WandBLogger`` that writes a
``metrics.jsonl`` sidecar into the run's ``output_dir`` instead of talking to
Weights & Biases. This IS the studio's metric contract; everything downstream
(metrics_store → SQLite → WebSocket → plots/logs/tables) only ever tails this
file, so the studio is W&B-free while the training code is unmodified LeRobot.

Same method surface as WandBLogger: __init__(cfg), log_dict(d, step, mode,
custom_step_key), log_policy(checkpoint_dir), log_video(path, step, mode).
"""
import json
import os
import time


class StudioLogger:
    def __init__(self, cfg):
        self.output_dir = str(getattr(cfg, "output_dir", ".") or ".")
        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, "metrics.jsonl")
        self._f = open(self.path, "a", buffering=1)   # line-buffered
        try:
            conf = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
        except Exception:
            conf = {}
        self._write({"_type": "config", "job_name": getattr(cfg, "job_name", None),
                     "config": conf})
        self._write({"_type": "start"})

    # ── WandBLogger-compatible API ───────────────────────────────────────────
    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        rec = {"_type": "metric", "mode": mode, "step": step}
        for k, v in d.items():
            if isinstance(v, bool):
                rec[k] = v
            elif isinstance(v, (int, float, str)):
                rec[k] = v
        self._write(rec)

    def log_policy(self, checkpoint_dir):
        name = os.path.basename(str(checkpoint_dir).rstrip("/"))
        try:
            step = int(name)
        except (TypeError, ValueError):
            step = None
        self._write({"_type": "checkpoint", "step": step, "path": str(checkpoint_dir)})

    def log_video(self, video_path, step, mode="train"):
        self._write({"_type": "video", "step": step, "mode": mode, "path": str(video_path)})

    # ── internal ─────────────────────────────────────────────────────────────
    def _write(self, rec):
        rec.setdefault("t", time.time())
        try:
            self._f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass
