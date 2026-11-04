"""experiment.py — the high-level training/eval manager over the runstore tree.

One class, `ExperimentManager`, is the single entry point the CLI (mabel_train.py)
and the GUI both call. It owns the *workflow* (create run → ensure v3.0 dataset →
launch the LeRobot backend → render plots → update manifest → register evals),
while `runstore` owns *paths/manifests* and `backends/run_lerobot.py` owns the
actual LeRobot subprocess. Backends are spawned with the LeRobot-env python
(CFG.TRAIN_PYTHON) so the studio (mabel env) never imports LeRobot.

Verbs: train, resume, evaluate, finalize, list_runs, get_run — each returns a
plain dict so it is trivially JSON-able for the API.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

from . import config as CFG
from . import runstore as RS
from . import plots

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUN_LEROBOT = os.path.join(_HERE, "backends", "run_lerobot.py")
_RUN_EVAL = os.path.join(_HERE, "backends", "run_eval.py")

# UI arch → LeRobot policy hp defaults that keep tiny/verification runs light. The
# GUI's arch_registry supplies the rich schema; these are safe fallbacks.
_ARCH_DEFAULTS = {
    "act": {}, "dp": {}, "pi0": {}, "pi05": {},
}


class ExperimentManager:
    def __init__(self, train_python=None, v30_builder=None):
        self.registry = RS.RunRegistry()
        self.evals = RS.EvalRegistry()
        # the LeRobot-env python that can import lerobot (the studio runs in mabel env)
        self.train_python = train_python or os.environ.get(
            "MABEL_LEROBOT_PYTHON", os.path.expanduser("~/act_train_env/bin/python"))

    # ── dataset → v3.0 (LeRobot 0.6 trains on v3.0) ──────────────────────────
    def ensure_v30(self, dataset_root, repo_id, dataset_id, fps=10):
        """Return a v3.0 dataset root for `dataset_root`, building it once (cached
        under workspace/datasets/.v30/<dataset_id>) from the h5 episodes if the
        curated export is v2.1. If already v3.0, returns it unchanged."""
        info = os.path.join(dataset_root, "meta", "info.json")
        ver = json.load(open(info)).get("codebase_version", "") if os.path.exists(info) else ""
        if ver >= "v3.0":
            return dataset_root
        v30 = os.path.join(CFG.DATASETS_DIR, ".v30", RS.slug(dataset_id))
        if os.path.exists(os.path.join(v30, "meta", "info.json")):
            return v30
        # build natively from the source h5 episodes if we can locate them; else
        # from the v2.1 export's own frames via the LeRobot dataset writer.
        raise RuntimeError(
            f"dataset {dataset_id} is {ver or 'pre-v3.0'} and no v3.0 build exists at {v30}. "
            f"Build it with:  python3 robot_learning/h5_to_lerobot_v30.py --sessions ... --out {v30}")

    # ── launch a training run ────────────────────────────────────────────────
    def train(self, arch, dataset_id, dataset_root, *, repo_id="mabel/dataset",
              hp=None, device="cuda", task=None, notes=None, extra=None,
              detach=True, on_done=None):
        hp = {**_ARCH_DEFAULTS.get(RS.arch_slug(arch), {}), **(hp or {})}
        v30 = self.ensure_v30(dataset_root, repo_id, dataset_id, fps=hp.get("fps", 10))
        run = self.registry.create(arch, dataset_id, task=task or dataset_id,
                                   backend="lerobot", device=device, hyperparams=hp,
                                   dataset_root=v30, dataset_hash=None, notes=notes)
        # persist the resolved config next to the run
        with open(run.layout.train_config, "w") as f:
            json.dump({"arch": arch, "policy_hp": hp, "device": device,
                       "dataset_id": dataset_id, "dataset_root": v30, "repo_id": repo_id,
                       "extra": extra or []}, f, indent=2)
        run.set_status("running", pid=None)
        cmd = self._train_cmd(run, arch, v30, repo_id, hp, device, resume=False, extra=extra)
        return self._spawn(run, cmd, detach=detach, on_done=on_done)

    def resume(self, run_id, *, steps=None, checkpoint=None, detach=True, on_done=None):
        run = self.registry.find(run_id)
        if run is None:
            raise ValueError(f"no such run {run_id}")
        ck = checkpoint or (run.latest_checkpoint() or {}).get("path")
        if not ck:
            raise ValueError("no checkpoint to resume from")
        cfg = json.load(open(run.layout.train_config))
        hp = dict(cfg.get("policy_hp", {}))
        if steps:
            hp["steps"] = steps
        run.set_status("running")
        cfg_path = os.path.join(ck, "pretrained_model", "train_config.json")
        cmd = self._train_cmd(run, cfg["arch"], cfg["dataset_root"], cfg["repo_id"],
                              hp, cfg.get("device", "cuda"), resume=True,
                              extra=cfg.get("extra"), config_path=cfg_path)
        return self._spawn(run, cmd, detach=detach, on_done=on_done)

    def _train_cmd(self, run, arch, v30, repo_id, hp, device, resume, extra, config_path=None):
        L = run.layout
        cmd = [self.train_python, _RUN_LEROBOT,
               "--policy", RS.arch_slug(arch),
               "--dataset-root", v30, "--repo-id", repo_id,
               "--output-dir", L.output_dir, "--metrics-path", L.metrics_path,
               "--steps", str(int(hp.get("steps", 20000))),
               "--batch", str(int(hp.get("batch", hp.get("batch_size", 8)))),
               "--save-freq", str(int(hp.get("save_freq", max(1, int(hp.get("steps", 20000)) // 4)))),
               "--log-freq", str(int(hp.get("log_freq", 100))),
               "--num-workers", str(int(hp.get("num_workers", 4))),
               "--device", device]
        if resume:
            cmd.append("--resume")
            if config_path:
                cmd += ["--config-path", config_path]
        # forward arch-specific LeRobot fields (chunk_size, n_action_steps, …) as passthrough
        passthrough = []
        for k, v in hp.items():
            if k in ("steps", "batch", "batch_size", "save_freq", "log_freq", "num_workers", "fps"):
                continue
            passthrough.append(f"--policy.{k}={v}")
        passthrough += list(extra or [])
        if passthrough:
            cmd += ["--extra"] + passthrough
        return cmd

    def _spawn(self, run, cmd, *, detach, on_done):
        L = run.layout
        logf = open(L.train_log, "a")
        logf.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        run.set_status("running", pid=proc.pid)

        def _watch():
            rc = proc.wait()
            logf.close()
            self.finalize(run.manifest.run_id, status=("done" if rc == 0 else "failed"),
                          returncode=rc)
            if on_done:
                on_done(run.manifest.run_id, rc)

        if detach:
            threading.Thread(target=_watch, name=f"watch-{run.manifest.run_id}", daemon=True).start()
            return {"run_id": run.manifest.run_id, "pid": proc.pid, "path": L.root,
                    "status": "running", "detached": True}
        rc = proc.wait(); logf.close()
        res = self.finalize(run.manifest.run_id, status=("done" if rc == 0 else "failed"),
                            returncode=rc)
        res["returncode"] = rc
        return res

    # ── finalize: plots + manifest ───────────────────────────────────────────
    def finalize(self, run_id, status=None, returncode=None):
        run = self.registry.find(run_id)
        if run is None:
            return {"error": "run gone"}
        pngs = []
        try:
            pngs = plots.render_run_plots(run.layout)
        except Exception:
            pass
        cks = run.list_checkpoints()
        summary = self._metrics_summary(run.layout.metrics_path)
        run.manifest.update(
            status=status or run.manifest.status,
            returncode=returncode,
            n_checkpoints=len(cks),
            last_checkpoint=(cks[-1]["path"] if cks else None),
            best_checkpoint=(cks[-1]["path"] if cks else None),
            metrics_summary=summary,
            plots=[os.path.basename(p) for p in pngs],
        ).save(run.layout)
        return run.to_summary()

    @staticmethod
    def _metrics_summary(mpath):
        first = last = None
        n = 0
        if os.path.exists(mpath):
            with open(mpath) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("_type") == "metric" and "loss" in r:
                        first = first if first is not None else r["loss"]
                        last = r["loss"]; n += 1
        out = {"metric_points": n, "first_loss": first, "last_loss": last}
        if first and last:
            out["loss_improvement_pct"] = round(100 * (1 - last / first), 1)
        return out

    # ── evaluation from ANY checkpoint → eval tree ───────────────────────────
    def evaluate(self, run_id, *, checkpoint=None, kind="open_loop", episodes=1,
                 detach=False):
        run = self.registry.find(run_id)
        if run is None:
            raise ValueError(f"no such run {run_id}")
        ck = checkpoint or (run.latest_checkpoint() or {}).get("path")
        if not ck:
            raise ValueError("no checkpoint to evaluate")
        cfg = json.load(open(run.layout.train_config))
        ev = RS.EvalRun.create(run, kind=kind, checkpoint=ck,
                               config={"episodes": episodes, "dataset_root": cfg["dataset_root"]})
        cmd = [self.train_python, _RUN_EVAL,
               "--checkpoint", os.path.join(ck, "pretrained_model"),
               "--dataset-root", cfg["dataset_root"], "--repo-id", cfg["repo_id"],
               "--out", ev.layout.results_path, "--episodes", str(episodes),
               "--device", cfg.get("device", "cuda")]
        logp = os.path.join(ev.layout.logs_dir, "eval.log")
        with open(logp, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        results = None
        if os.path.exists(ev.layout.results_path):
            results = json.load(open(ev.layout.results_path))
        return {"eval_id": ev.layout.eval_id, "run_id": run_id, "checkpoint": ck,
                "returncode": rc, "results": results, "path": ev.layout.root}

    # ── read side (for the API/GUI) ──────────────────────────────────────────
    def list_runs(self, arch=None, dataset=None, status=None):
        runs = self.registry.filter(arch=arch, dataset=dataset, status=status)
        return [r.to_summary() for r in runs]

    def group_by_arch(self):
        return self.registry.group_by_arch()

    def get_run(self, run_id):
        run = self.registry.find(run_id)
        if run is None:
            return None
        d = run.to_summary()
        d["checkpoints"] = run.list_checkpoints()
        d["evals"] = self.evals.for_run(run_id)
        return d
