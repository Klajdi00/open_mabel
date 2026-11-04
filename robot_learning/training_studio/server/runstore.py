"""runstore.py — the experiment-management layer for the Training Studio.

Turns the flat `workspace/runs/<run_id>` scratch into a durable, human-browsable,
scalable tree under `robot_learning/training` and `robot_learning/eval`:

    training/<arch>/<dataset>/<run_id>/
        run.json                 # the manifest — SOURCE OF TRUTH for a run
        config/                  # resolved train config + feature selection
            train_config.json  selection.json
        logs/                    # train.log (raw)  train.jsonl (filtered)
        plots/                   # server-rendered PNGs (loss, lr, gpu, …)
        train/                   # == the backend's output_dir (LeRobot owns this)
            metrics.jsonl
            checkpoints/<step>/pretrained_model/…   |   checkpoints/step_XXXX.pt (native)
            checkpoints/last -> …

    eval/<arch>/<dataset>/<run_id>/<eval_id>/
        eval.json  results.json  plots/  videos/  logs/

Design rules that make this robust and scalable:
  * The MANIFEST (run.json) is the source of truth — a run survives a DB wipe or a
    manual `mv`, and the registry can be rebuilt by scanning the tree.
  * `output_dir` is a SUBDIR (`train/`) of the run root, so LeRobot (which refuses a
    pre-existing output_dir) still creates its own dir while we pre-create the
    managed siblings (config/logs/plots) and write run.json.
  * Path layout is `<arch>/<dataset>/<run_id>` so it scales across many
    archs × datasets × tasks × runs and is trivially globbable.
  * Everything is class/manager based (RunLayout, RunManifest, TrainingRun,
    RunRegistry, EvalRun, EvalRegistry) so callers never hand-build paths.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

from . import config as CFG

# ── arch → canonical directory slug ──────────────────────────────────────────
# Keyed on either the UI arch name or the LeRobot policy_type. Keep these stable:
# they are directory names on disk.
_ARCH_SLUGS = {
    "act": "act", "aloha": "act",
    "dp": "dp", "diffusion": "dp",
    "dit": "dit",
    "pi0": "pi0",
    "pi05": "pi05", "pi0.5": "pi05",
    "smolvla": "smolvla",
}


def arch_slug(arch: str) -> str:
    a = re.sub(r"[^a-z0-9]+", "", (arch or "").lower())
    return _ARCH_SLUGS.get((arch or "").lower(), _ARCH_SLUGS.get(a, a or "misc"))


def slug(text: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "").strip()).strip("-._")
    return (s[:maxlen] or "unnamed")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def new_run_id(now: datetime | None = None) -> str:
    """Sortable, human-legible, collision-resistant: <YYYYmmdd_HHMMSS>_<hex4>."""
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{os.urandom(2).hex()}"


# ── path layout ──────────────────────────────────────────────────────────────
class RunLayout:
    """Pure path resolution for one run. Never touches the backend."""

    def __init__(self, arch: str, dataset: str, run_id: str, root: str | None = None):
        self.arch = arch
        self.arch_slug = arch_slug(arch)
        self.dataset = dataset
        self.dataset_slug = slug(dataset)
        self.run_id = run_id
        self.root = os.path.join(root or CFG.TRAINING_ROOT,
                                 self.arch_slug, self.dataset_slug, run_id)

    # managed siblings (we own these)
    @property
    def config_dir(self): return os.path.join(self.root, "config")
    @property
    def logs_dir(self): return os.path.join(self.root, "logs")
    @property
    def plots_dir(self): return os.path.join(self.root, "plots")
    @property
    def metrics_dir(self): return os.path.join(self.root, "metrics")
    @property
    def manifest_path(self): return os.path.join(self.root, "run.json")
    # backend-owned (LeRobot / native trainer creates this; must NOT pre-exist)
    @property
    def output_dir(self): return os.path.join(self.root, "train")
    # metrics live OUTSIDE output_dir (LeRobot refuses a pre-existing output_dir),
    # in a managed sibling the studio Tailer reads.
    @property
    def metrics_path(self): return os.path.join(self.metrics_dir, "metrics.jsonl")
    @property
    def checkpoints_dir(self): return os.path.join(self.output_dir, "checkpoints")
    @property
    def train_log(self): return os.path.join(self.logs_dir, "train.log")
    @property
    def train_jsonl(self): return os.path.join(self.logs_dir, "train.jsonl")
    @property
    def train_config(self): return os.path.join(self.config_dir, "train_config.json")
    @property
    def selection_json(self): return os.path.join(self.config_dir, "selection.json")

    def ensure(self):
        """Create the managed dirs — NOT output_dir (the backend owns/creates it)."""
        for d in (self.root, self.config_dir, self.logs_dir, self.plots_dir, self.metrics_dir):
            os.makedirs(d, exist_ok=True)
        return self


# ── manifest (run.json) ──────────────────────────────────────────────────────
class RunManifest:
    """The durable description of a run. Source of truth; the SQLite row is an index."""

    def __init__(self, data: dict):
        self.data = data

    # convenience accessors
    @property
    def run_id(self): return self.data["run_id"]
    @property
    def arch(self): return self.data.get("arch")
    @property
    def dataset(self): return self.data.get("dataset")
    @property
    def status(self): return self.data.get("status")

    @classmethod
    def create(cls, run_id, arch, dataset, *, task=None, backend="lerobot",
               device=None, hyperparams=None, selection=None, dataset_root=None,
               dataset_hash=None, notes=None, now=None):
        ts = (now or datetime.now()).isoformat(timespec="seconds")
        return cls({
            "run_id": run_id, "arch": arch, "arch_slug": arch_slug(arch),
            "dataset": dataset, "dataset_slug": slug(dataset),
            "task": task or dataset, "backend": backend,
            "status": "created", "created": ts, "updated": ts,
            "device": device, "git_sha": git_sha(),
            "hyperparams": hyperparams or {}, "selection": selection or {},
            "dataset_root": dataset_root, "dataset_hash": dataset_hash,
            "notes": notes, "steps_target": (hyperparams or {}).get("steps"),
            "best_checkpoint": None, "last_checkpoint": None,
            "metrics_summary": {}, "eval_runs": [],
        })

    def update(self, **kw):
        self.data.update(kw)
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        return self

    def save(self, layout: RunLayout):
        layout.ensure()
        tmp = layout.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, layout.manifest_path)     # atomic
        return self

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))


# ── a run = layout + manifest + checkpoint introspection ─────────────────────
class TrainingRun:
    def __init__(self, layout: RunLayout, manifest: RunManifest):
        self.layout = layout
        self.manifest = manifest

    @classmethod
    def create(cls, arch, dataset, **kw):
        run_id = kw.pop("run_id", None) or new_run_id()
        layout = RunLayout(arch, dataset, run_id).ensure()
        man = RunManifest.create(run_id, arch, dataset, **kw).save(layout)
        return cls(layout, man)

    @classmethod
    def load(cls, manifest_path):
        man = RunManifest.load(manifest_path)
        layout = RunLayout(man.arch, man.dataset, man.run_id)
        return cls(layout, man)

    def list_checkpoints(self):
        """Both layouts: LeRobot `checkpoints/<step>/pretrained_model/` dirs and
        native `checkpoints/step_XXXX.pt` files. Sorted by step ascending."""
        d = self.layout.checkpoints_dir
        out = []
        if not os.path.isdir(d):
            return out
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if name in ("last", "best"):
                continue
            if os.path.isdir(p):
                m = re.search(r"(\d+)", name)
                out.append({"step": int(m.group(1)) if m else 0, "path": p, "kind": "lerobot"})
            elif name.endswith(".pt"):
                m = re.search(r"(\d+)", name)
                out.append({"step": int(m.group(1)) if m else 0, "path": p, "kind": "native"})
        return sorted(out, key=lambda c: c["step"])

    def latest_checkpoint(self):
        cks = self.list_checkpoints()
        return cks[-1] if cks else None

    def set_status(self, status, **extra):
        self.manifest.update(status=status, **extra).save(self.layout)
        return self

    def to_summary(self):
        d = dict(self.manifest.data)
        d["path"] = self.layout.root
        d["n_checkpoints"] = len(self.list_checkpoints())
        return d


# ── registry: create / list / find across the whole tree ─────────────────────
class RunRegistry:
    def __init__(self, root: str | None = None):
        self.root = root or CFG.TRAINING_ROOT

    def create(self, arch, dataset, **kw) -> TrainingRun:
        return TrainingRun.create(arch, dataset, **kw)

    def all(self):
        runs = []
        for man in glob.glob(os.path.join(self.root, "*", "*", "*", "run.json")):
            try:
                runs.append(TrainingRun.load(man))
            except Exception:
                continue
        return sorted(runs, key=lambda r: r.manifest.data.get("created", ""), reverse=True)

    def find(self, run_id):
        for man in glob.glob(os.path.join(self.root, "*", "*", run_id, "run.json")):
            return TrainingRun.load(man)
        return None

    def filter(self, arch=None, dataset=None, task=None, status=None):
        def ok(r):
            m = r.manifest.data
            return ((arch is None or arch_slug(arch) == m.get("arch_slug"))
                    and (dataset is None or slug(dataset) == m.get("dataset_slug"))
                    and (task is None or task == m.get("task"))
                    and (status is None or status == m.get("status")))
        return [r for r in self.all() if ok(r)]

    def group_by_arch(self):
        groups: dict[str, list] = {}
        for r in self.all():
            groups.setdefault(r.manifest.data.get("arch_slug", "misc"), []).append(r.to_summary())
        return groups


# ── evaluation artifacts ─────────────────────────────────────────────────────
def new_eval_id(kind="eval", now=None):
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{kind}_{ts}_{os.urandom(2).hex()}"


class EvalLayout:
    def __init__(self, arch, dataset, run_id, eval_id, root=None):
        self.run_id, self.eval_id = run_id, eval_id
        self.root = os.path.join(root or CFG.EVAL_ROOT, arch_slug(arch),
                                 slug(dataset), run_id, eval_id)

    @property
    def config_path(self): return os.path.join(self.root, "eval.json")
    @property
    def results_path(self): return os.path.join(self.root, "results.json")
    @property
    def plots_dir(self): return os.path.join(self.root, "plots")
    @property
    def videos_dir(self): return os.path.join(self.root, "videos")
    @property
    def logs_dir(self): return os.path.join(self.root, "logs")

    def ensure(self):
        for d in (self.root, self.plots_dir, self.videos_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        return self


class EvalRun:
    def __init__(self, layout: EvalLayout):
        self.layout = layout

    @classmethod
    def create(cls, run: TrainingRun, *, kind="open_loop", checkpoint=None, config=None):
        eid = new_eval_id(kind)
        m = run.manifest.data
        layout = EvalLayout(m["arch"], m["dataset"], m["run_id"], eid).ensure()
        cfg = {"eval_id": eid, "run_id": m["run_id"], "arch": m["arch"],
               "dataset": m["dataset"], "kind": kind, "checkpoint": checkpoint,
               "created": datetime.now().isoformat(timespec="seconds"),
               "config": config or {}}
        with open(layout.config_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        # link back into the training run's manifest
        run.manifest.data.setdefault("eval_runs", []).append(
            {"eval_id": eid, "kind": kind, "path": layout.root})
        run.manifest.save(run.layout)
        return cls(layout)

    def write_results(self, results: dict):
        with open(self.layout.results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return self.layout.results_path


class EvalRegistry:
    def __init__(self, root=None):
        self.root = root or CFG.EVAL_ROOT

    def for_run(self, run_id):
        out = []
        for cfg in glob.glob(os.path.join(self.root, "*", "*", run_id, "*", "eval.json")):
            try:
                with open(cfg) as f:
                    c = json.load(f)
                rp = os.path.join(os.path.dirname(cfg), "results.json")
                c["results"] = json.load(open(rp)) if os.path.exists(rp) else None
                c["path"] = os.path.dirname(cfg)
                out.append(c)
            except Exception:
                continue
        return sorted(out, key=lambda c: c.get("created", ""), reverse=True)
