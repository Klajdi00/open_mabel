"""
Trainer — orchestrates a run end to end and manages the training subprocess.

start_training(params) (run in a background thread) does the whole prep:
  combine sources → build scoped view → convert to v3.0 → split → build
  train_config.json → spawn ``run_train`` (StudioLogger patched in) → start the
  metrics tailer + system monitor. pause()/stop() control the process; retrain is
  just a new run with resume/pretrained-checkpoint.
"""
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid

from . import (arch_registry as AR, config as CFG, config_builder,
               dataset_registry as DR, feature_select as FS, metrics_store as MS,
               sysmon, compute)

_runs = {}          # run_id -> {proc, output_dir, sysmon, status}
_lock = threading.Lock()


def new_run_id():
    return "run_" + uuid.uuid4().hex[:10]


_LOGS_DIR = os.path.join(CFG.RUNS_DIR, "_logs")
_CFG_DIR = os.path.join(CFG.RUNS_DIR, "_configs")


_NOISE = ("pynvml package is deprecated", "import pynvml", "FutureWarning",
          "nvidia-ml-py instead", "Training:   0%", "it/s]", "s/step]")


def _log(run_id, line):
    s = line.strip()
    if not s or any(n in s for n in _NOISE):
        return  # drop framework noise + tqdm bars; keep meaningful log lines
    MS.append_log(run_id, line.rstrip("\n"))
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        with open(os.path.join(_LOGS_DIR, f"{run_id}.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "line": line.rstrip("\n")}) + "\n")
    except Exception:
        pass


# ── dataset prep ──────────────────────────────────────────────────────────────
def prepare_dataset(datasets, selection, split_ratios, seed, progress=None):
    """sources → (combined) → (scoped view) → v3.0 → split. Returns a dict with
    the v3.0 training root (for lerobot) and the v2.1 read root (for eval/io)."""
    # 1) combine (always, so single + multi are uniform + renumbered)
    combined_name = "ds_" + "_".join(datasets)[:40] + "_" + str(abs(hash(tuple(datasets))) % 10000)
    if progress:
        progress(1, 5, stage=0)
    combined = DR.combine(datasets, combined_name)
    v21_root = combined["path"]
    n_eps = combined["n_episodes"]

    # 2) scoped view (skip if full selection)
    sel = selection
    if not FS.is_full_selection(v21_root, sel):
        if progress:
            progress(2, 5, stage=0)
        view_name = FS.selection_key(combined_name, sel)
        view = FS.build_view(v21_root, os.path.join(CFG.DATASETS_DIR, view_name), sel)
        v21_root = view["path"]

    # 3) split by episode (an early parameter)
    sp = DR.split(n_eps, split_ratios, seed)

    # 4) materialize a TRAIN-ONLY dataset (episodes renumbered 0..k-1) so training
    #    never selects non-zero episode indices — that trips a LeRobot global/local
    #    index bug with multi-step temporal windows. eval/test read the full v21.
    if progress:
        progress(3, 5, stage=0)
    train_name = os.path.basename(v21_root.rstrip("/")) + "__train"
    train_v21 = DR.subset_dataset(v21_root, sp["train"], train_name)["path"]

    # 5) convert the train dataset to v3.0 for LeRobot
    if progress:
        progress(4, 5, stage=0)
    v30_root = DR.ensure_v30(train_v21)

    # 5b) build an eval-split v3.0 dataset too (for the live validation-loss curve)
    eval_v30 = None
    if sp["eval"]:
        try:
            eval_name = os.path.basename(v21_root.rstrip("/")) + "__eval"
            eval_v21 = DR.subset_dataset(v21_root, sp["eval"], eval_name)["path"]
            eval_v30 = DR.ensure_v30(eval_v21)
        except Exception:
            eval_v30 = None
    if progress:
        progress(5, 5, stage=0)
    return {"v21_root": v21_root, "train_v21": train_v21, "v30_root": v30_root,
            "eval_v30": eval_v30, "n_episodes": n_eps, "split": sp,
            "version_hash": DR.version_hash(v21_root)}


# ── subprocess management ─────────────────────────────────────────────────────
def _spawn(run_id, config_path, output_dir, extra_args=None, tailer_reset=False):
    env = dict(os.environ)
    env.update(CFG.LEROBOT_ENV)
    env["PYTHONPATH"] = CFG.LEARNING_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [CFG.TRAIN_PYTHON, "-m", "training_studio.server.lerobot_shim.run_train",
           f"--config_path={config_path}"] + list(extra_args or [])
    os.makedirs(_LOGS_DIR, exist_ok=True)
    logfile = os.path.join(_LOGS_DIR, f"{run_id}.log")
    # APPEND so a resumed run keeps the full terminal history of earlier attempts
    logf = open(logfile, "a", buffering=1)
    logf.write(f"$ {' '.join(cmd)}\n")
    # start_new_session=True detaches the child into its own session, so the
    # TRAINING KEEPS RUNNING even if the studio server is killed.
    proc = subprocess.Popen(cmd, cwd=CFG.LEARNING_DIR, env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    with _lock:
        _runs[run_id]["proc"] = proc
    MS.update_run(run_id, status="running", pid=proc.pid)
    MS.start_tailer(run_id, output_dir, logfile, reset=tailer_reset)   # tail metrics.jsonl + the log file

    def _watch():
        code = proc.wait()
        try:
            logf.close()
        except Exception:
            pass
        mon = _runs.get(run_id, {}).get("sysmon")
        if mon:
            mon.stop()
        cur = MS.get_run(run_id) or {}
        if cur.get("status") == "paused":
            status = "paused"
        elif code == 0:
            status = "done"
        elif cur.get("status") == "stopped":
            status = "stopped"
        else:
            status = "failed"
        MS.update_run(run_id, status=status, error=None if code == 0 else f"exit code {code}")

    threading.Thread(target=_watch, daemon=True).start()
    return proc


def launch(run_id, arch, hp, sel, v30_root, train_episodes, device="auto",
           resume=False, pretrained_ckpt=None, job_name=None,
           eval_root=None, val_freq=None):
    output_dir = os.path.join(CFG.RUNS_DIR, run_id)   # LeRobot creates this itself
    os.makedirs(_CFG_DIR, exist_ok=True)
    config_path = os.path.join(_CFG_DIR, f"{run_id}.json")
    built = config_builder.build(
        arch, hp, sel, v30_root, output_dir, config_path, train_episodes,
        device=device, resume=resume, pretrained_ckpt=pretrained_ckpt, job_name=job_name)
    # sidecar consumed by run_train to compute a live validation-loss curve
    if eval_root:
        try:
            with open(config_path + ".studio.json", "w") as f:
                json.dump({"eval_root": os.path.abspath(eval_root),
                           "val_freq": int(val_freq or hp.get("log_freq", 10))}, f)
        except Exception:
            pass
    with _lock:
        _runs[run_id] = {"output_dir": output_dir, "proc": None}
    # persist the output_dir + config so the run is reconstructable after a
    # server restart (reconcile reads output_dir to re-attach or mark interrupted).
    MS.update_run(run_id, output_dir=output_dir, config_json=json.dumps(built["config"]))
    ensure_sysmon(run_id)
    _spawn(run_id, built["path"], output_dir)   # spawns detached + starts the tailer
    return built


def ensure_sysmon(run_id):
    """Start (or re-start after a server restart) the system-metrics poller for a
    run, so GPU/CPU keeps streaming — including for reconciled detached jobs."""
    with _lock:
        r = _runs.setdefault(run_id, {})
        mon = r.get("sysmon")
        if mon is not None and mon.is_alive():
            return
        mon = sysmon.SysMonitor(run_id)
        mon.start()
        r["sysmon"] = mon


def resume_run(run_id, steps=None, checkpoint=None):
    """TRUE resume: continue the SAME run from a checkpoint — optimizer state and
    the step counter pick up where training stopped (LeRobot ``resume=true`` with
    the checkpoint's own train_config.json). Only trains the REMAINING steps up to
    the (optionally updated) total. Metrics/logs append to the run's history."""
    run = MS.get_run(run_id)
    if not run:
        raise ValueError(f"no such run: {run_id}")
    if run.get("status") in ("running", "preparing", "queued"):
        raise ValueError("run is already active")
    ckpts = MS.get_checkpoints(run_id)
    if not ckpts:
        raise ValueError("no checkpoint to resume from — use retrain-from-scratch instead")
    ck = checkpoint or ckpts[-1]["path"]
    ck_cfg = os.path.join(ck, "pretrained_model", "train_config.json")
    if not os.path.isfile(ck_cfg):
        raise ValueError(f"checkpoint config missing: {ck_cfg}")
    output_dir = run.get("output_dir") or os.path.join(CFG.RUNS_DIR, run_id)
    # carry the eval-loss sidecar over so the validation curve continues
    side_src = os.path.join(_CFG_DIR, f"{run_id}.json.studio.json")
    if os.path.isfile(side_src):
        try:
            shutil.copyfile(side_src, ck_cfg + ".studio.json")
        except Exception:
            pass
    extra = ["--resume=true"]
    if steps:
        extra.append(f"--steps={int(steps)}")
    # fresh tailer: metrics.jsonl + the run log are append-mode, so a reset
    # re-ingests the FULL history (no duplicates, nothing lost)
    MS.stop_tailer(run_id)
    with _lock:
        _runs.setdefault(run_id, {})["output_dir"] = output_dir
    MS.update_run(run_id, status="queued", error=None)
    ensure_sysmon(run_id)
    _spawn(run_id, ck_cfg, output_dir, extra_args=extra, tailer_reset=True)
    return {"run_id": run_id, "resumed_from": ck,
            "steps": steps, "note": "same run — step counter continues"}


def start_training(params, progress=None):
    """Full orchestration (prep + launch). Returns run metadata. Meant to run in a
    background thread; the run_id is registered up front so the UI can poll."""
    run_id = params["run_id"]
    arch = params["arch"]
    hp = {**AR.defaults(arch), **params.get("hp", {})}
    sel = params["selection"]
    datasets = params["datasets"]
    device = compute.resolve_device(params.get("device", "auto"))
    try:
        MS.update_run(run_id, status="preparing", device=device,
                      dataset_id=",".join(datasets), scope=sel.get("scope"))
        prep = prepare_dataset(datasets, sel, params.get("split", (0.8, 0.1, 0.1)),
                               params.get("seed", hp.get("seed", 1000)), progress)
        MS.update_run(run_id, dataset_hash=prep["version_hash"],
                      meta_json=json.dumps({"selection": sel, "split": prep["split"],
                                            "hp": hp, "prep": {"v21": prep["v21_root"],
                                                              "v30": prep["v30_root"]}}))
        # train on ALL episodes of the train-only v3.0 dataset (episodes=None)
        launch(run_id, arch, hp, sel, prep["v30_root"], None,
               device=device, resume=params.get("resume", False),
               pretrained_ckpt=params.get("pretrained_ckpt"),
               job_name=params.get("name"),
               eval_root=prep.get("eval_v30"), val_freq=hp.get("log_freq", 10))
        return {"run_id": run_id, "prep": prep}
    except Exception as e:
        import traceback
        MS.update_run(run_id, status="failed", error=f"{e}\n{traceback.format_exc()}")
        raise


# ── control (works via the proc handle OR the persisted PID after a restart) ──
def _signal(run_id, sig):
    with _lock:
        r = _runs.get(run_id)
    if r and r.get("proc"):
        try:
            r["proc"].send_signal(sig)
            return True
        except Exception:
            pass
    pid = (MS.get_run(run_id) or {}).get("pid")
    if pid and MS.pid_alive(pid):
        try:
            os.kill(int(pid), sig)
            return True
        except Exception:
            return False
    return False


def pause(run_id):
    MS.update_run(run_id, status="paused")
    return _signal(run_id, signal.SIGTERM)


def stop(run_id):
    MS.update_run(run_id, status="stopped")
    return _signal(run_id, signal.SIGTERM)


def is_active(run_id):
    with _lock:
        r = _runs.get(run_id)
    if r and r.get("proc"):
        return r["proc"].poll() is None
    pid = (MS.get_run(run_id) or {}).get("pid")
    return MS.pid_alive(pid)


def latest_checkpoint(run_id):
    cks = MS.get_checkpoints(run_id)
    return cks[-1]["path"] if cks else None
