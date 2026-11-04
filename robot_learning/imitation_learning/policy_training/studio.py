#!/usr/bin/env python3
"""
studio.py — NOVA Training Studio: pick a curated dataset, ACT / Diffusion Policy / π0 / π0.5, edit the config, train,
watch the curves, evaluate on the test split and compare runs. A single-page web UI served on :8260 (stdlib only).

    python studio.py                          # http://127.0.0.1:8260

Datasets are LeRobot exports under POLICY_DATASET_ROOT (see train.py), re-scanned on every request. ACT, DP and π0.5
train through train.py, each run its own process (quitting the studio never stops one; Stop asks it to save and exit).
π0 goes through LeRobot's own lerobot-train and needs a CUDA GPU: without one the studio saves the command in the run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T  # noqa: E402

PI = {"pi0": "lerobot/pi0_base"}                 # π0.5 trains in train.py (frozen VLM + action expert)
ARCHS = [
    {"id": "act", "name": "ACT", "family": "CVAE transformer", "where": "any GPU (CUDA / MPS)",
     "blurb": "Action Chunking Transformer: ResNet18 per camera, transformer encoder–decoder, CVAE latent. Fast, deterministic chunks."},
    {"id": "dp", "name": "Diffusion Policy", "family": "conditional 1-D U-Net, DDIM", "where": "any GPU (CUDA / MPS)",
     "blurb": "Denoises the 4 s chunk from noise conditioned on image keypoints + state history. Multi-modal, slower inference."},
    {"id": "pi0", "name": "π0", "family": "VLA · flow matching", "where": "CUDA GPU ≥ 24 GB",
     "blurb": "PaliGemma 3B vision-language backbone + flow-matching action expert, fine-tuned from lerobot/pi0_base."},
    {"id": "pi05", "name": "π0.5", "family": "VLA · knowledge insulation", "where": "CUDA GPU ≥ 16 GB",
     "blurb": "π0 successor, fine-tuned from lerobot/pi05_base: VLM frozen in bf16, action expert trained (train.py)."},
]
_REPORTS: dict = {}
_LOCK = threading.Lock()


def disk_free_gb():
    T.RUNS.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(T.RUNS).free / 2**30


def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"device": "cuda", "name": torch.cuda.get_device_name(0)}
        if torch.backends.mps.is_available():
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            return {"device": "mps", "name": name or "Apple silicon"}
    except Exception:                                          # noqa: BLE001
        pass
    return {"device": "cpu", "name": "cpu"}


def datasets():
    out = []
    for info in sorted(T.DATASETS.glob("**/meta/info.json")):
        root = info.parent.parent
        try:
            i = json.loads(info.read_text())
        except ValueError:
            continue
        cur = root / "meta" / "curation.json"
        src = json.loads(cur.read_text()).get("source", {}) if cur.exists() else {}
        cams = [k.split(".")[-1] for k, v in i["features"].items() if v.get("dtype") == "video"]
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        cache = T.cache_dir(root.name, tuple(T.DEFAULTS["image_size"]))
        fresh = (cache / "meta.json").exists() and json.loads((cache / "meta.json").read_text()).get("stamp") == info.stat().st_mtime
        out.append({"name": root.name, "path": str(root), "episodes": i["total_episodes"], "frames": i["total_frames"],
                     "fps": i["fps"], "minutes": i["total_frames"] / i["fps"] / 60, "cameras": cams,
                     "state_dim": i["features"]["observation.state"]["shape"][0], "action_dim": i["features"]["action"]["shape"][0],
                     "size_mb": size / 2**20, "task": src.get("task"), "takes": len(src.get("items", [])),
                     "created": info.stat().st_mtime, "cached": fresh, "preparing": _PREP.get(root.name) == "running"})
    return out


def run_info(run: Path, full=False):
    rd = lambda n: json.loads((run / n).read_text()) if (run / n).exists() else None
    cfg, st = rd("config.json") or rd("pending.json") or {}, rd("status.json") or {}
    pid = st.get("pid")
    alive = bool(pid) and pid_alive(pid)
    state = st.get("state", "queued")
    if state in ("preparing", "training", "evaluating") and not alive:
        state = "crashed" if not (run / "checkpoints" / "last").exists() else "stopped"
    if (run / "lerobot_cmd.txt").exists() and not alive and state == "queued":
        state = "external"
    test = rd("eval/test.json")
    out = {"name": run.name, "policy": cfg.get("policy"), "dataset": cfg.get("dataset"), "state": state, "alive": alive,
           "step": st.get("step", 0), "steps": cfg.get("steps"), "loss": st.get("loss"), "eta_s": st.get("eta_s") if alive else None,
           "steps_per_s": st.get("steps_per_s"), "best_val_l1": st.get("best_val_l1"), "params": st.get("params"),
           "created": run.stat().st_mtime, "updated": st.get("updated"),
           "test": {"l1": test["val_l1"], "bins_deg": test["bins_deg"], "latency_ms": test.get("latency_ms")} if test else None}
    if full:
        out.update(config=cfg, report=rd("data_report.json"), best=rd("best.json"), split=rd("split.json"), test_full=test,
                   plots=sorted(p.name for p in (run / "plots").glob("*.png")) if (run / "plots").exists() else [],
                   checkpoints=sorted(p.name for p in (run / "checkpoints").iterdir()) if (run / "checkpoints").exists() else [],
                   log=tail(run / "train.log", 40), cmd=(run / "lerobot_cmd.txt").read_text() if (run / "lerobot_cmd.txt").exists() else None,
                   reproduce=f"python train.py train --policy {cfg.get('policy')} --name {run.name}_repro "
                             f"--config {run / 'config.json'}")
    return out


def tail(p: Path, n):
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16000))
        return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def runs():
    T.RUNS.mkdir(parents=True, exist_ok=True)
    return sorted((run_info(r) for r in T.RUNS.iterdir() if r.is_dir() and not r.name.startswith("compare_")),
                  key=lambda r: -r["created"])


def spawn(args, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "USE_TF": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "ab") as f:
        return subprocess.Popen([sys.executable, "-u", *args], cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True).pid


def start_train(body):
    kind, name = body["policy"], body["name"].strip().replace(" ", "_")
    if not name or "/" in name:
        raise ValueError("Give the run a name (letters, digits, _ and -)")
    run = T.RUNS / name
    if run.exists():
        raise ValueError(f"A run called {name} exists — pick another name")
    cfg = T.deep_update(copy.deepcopy(T.DEFAULTS), body.get("config") or {})
    cfg["policy"] = kind
    if kind in PI:
        return start_pi(kind, name, cfg)
    if kind == "pi05":                                    # its pretraining chunk, current state only (see train.py)
        T.deep_update(cfg, {k: T.PI05_RUN[k] for k in ("history_s", "chunk_s")})
    if disk_free_gb() < 3:
        raise ValueError(f"Only {disk_free_gb():.1f} GB free on disk — checkpoints need ~2 GB; free some space first")
    run.mkdir(parents=True)
    (run / "pending.json").write_text(json.dumps(cfg, indent=1))
    spawn([str(HERE / "train.py"), "train", "--policy", kind, "--name", name, "--config", str(run / "pending.json")], run / "train.log")
    return {"ok": True, "name": name}


def start_pi(kind, name, cfg):
    """π0 / π0.5 via lerobot-train on the export itself (LeRobot handles its tokenizer and image layout)."""
    ds = T.dataset_root(cfg["dataset"])
    out = T.RUNS / name
    cmd = ["lerobot-train", f"--policy.type={kind}", f"--policy.pretrained_path={PI[kind]}",
           f"--dataset.repo_id=local/{Path(cfg['dataset']).name}", f"--dataset.root={ds}", "--dataset.video_backend=pyav",
           f"--output_dir={out / 'lerobot'}", f"--job_name={name}", f"--steps={cfg['steps']}",
           f"--batch_size={cfg.get('pi_batch_size', 8)}", f"--policy.chunk_size={int(cfg['chunk_s'] * cfg.get('fps', 30))}",
           f"--policy.n_action_steps={cfg['n_action_steps']}", "--policy.push_to_hub=false", "--wandb.enable=false",
           f"--policy.device={cfg['device'] if cfg['device'] != 'mps' else 'cuda'}"]
    out.mkdir(parents=True)
    (out / "pending.json").write_text(json.dumps(cfg, indent=1))
    (out / "lerobot_cmd.txt").write_text("USE_TF=0 " + " ".join(cmd) + "\n")
    dev = device_info()
    if dev["device"] != "cuda":
        return {"ok": False, "name": name, "external": True,
                "message": f"π0 fine-tunes a 3B-parameter VLA: it needs a CUDA GPU with ≥ 24 GB "
                           f"and the gated google/paligemma-3b-pt-224 weights (~7 GB; {disk_free_gb():.0f} GB free here). "
                           f"This host has {dev['name']} ({dev['device']}). The exact command is saved in the run — "
                           "copy it to a CUDA host with this dataset."}
    spawn(["-m", "lerobot.scripts.lerobot_train", *cmd[1:]], out / "train.log")
    return {"ok": True, "name": name}


_PREP: dict = {}


def prepare_async(name):
    if _PREP.get(name) == "running":
        return
    _PREP[name] = "running"
    def go():
        try:
            T.prepare(name, tuple(T.DEFAULTS["image_size"]))
            _PREP[name] = "done"
        except Exception as e:                                 # noqa: BLE001
            _PREP[name] = f"failed: {e}"
    threading.Thread(target=go, daemon=True).start()


def report(cfg):
    """Cleaning + split report for a config (what the run would train on) — memoised on the data-relevant keys."""
    keys = ("dataset", "cameras", "image_size", "chunk_s", "history_s", "history_stride_s", "split", "seed", "state_sigma_s",
            "action_sigma_s", "max_interp_s", "max_cam_age_s", "max_stale_frac", "min_future_s", "std_floor")
    c = T.deep_update(copy.deepcopy(T.DEFAULTS), cfg)
    key = json.dumps({k: c[k] for k in keys}, sort_keys=True)
    with _LOCK:
        if key not in _REPORTS:
            if not (T.cache_dir(c["dataset"], tuple(c["image_size"])) / "meta.json").exists():
                prepare_async(c["dataset"])
                return {"pending": True, "message": "Building the frame cache for this dataset (a few minutes, once)…"}
            B = T.build(c, verbose=False)
            _REPORTS[key] = B["report"]
        return _REPORTS[key]


def make_handler():
    page = HERE / "studio.html"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, obj, code=200, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self.send(page.read_bytes(), ctype="text/html; charset=utf-8")
                if path == "/api/health":
                    return self.send({"service": "nova_training", "ok": True})
                if path == "/api/state":
                    return self.send({"datasets": datasets(), "runs": runs(), "device": device_info(), "disk_free_gb": disk_free_gb(),
                                      "curation": None, "prep": _PREP, "archs": ARCHS, "defaults": T.DEFAULTS,
                                      "groups": list(T.GROUPS), "hbins": [f"{a:g}-{b:g}s" for a, b in T.HBINS]})
                if path.startswith("/api/run/"):
                    parts = path.split("/")
                    run = T.RUNS / parts[3]
                    if not run.is_dir() or ".." in parts[3]:
                        return self.send({"error": "no such run"}, 404)
                    if len(parts) == 4:
                        return self.send(run_info(run, full=True))
                    if parts[4] == "metrics":
                        since = int(q.get("since", 0))
                        lines = (run / "metrics.jsonl").read_text().splitlines() if (run / "metrics.jsonl").exists() else []
                        return self.send({"n": len(lines), "rows": [json.loads(l) for l in lines[since:] if l.strip()]})
                    if parts[4] == "plot":
                        f = run / "plots" / Path(parts[5]).name
                        return self.send(f.read_bytes(), ctype="image/png") if f.exists() else self.send({"error": "no plot"}, 404)
                if path.startswith("/api/compare/"):
                    f = T.RUNS / Path(path.split("/")[3]).name / Path(path.split("/")[4]).name
                    if f.suffix == ".png" and f.exists():
                        return self.send(f.read_bytes(), ctype="image/png")
                    if f.suffix == ".md" and f.exists():
                        return self.send({"md": f.read_text()})
                    return self.send({"error": "not ready"}, 404)
                return self.send({"error": "not found"}, 404)
            except Exception as e:                             # noqa: BLE001
                return self.send({"error": str(e)}, 500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                b = self.body()
                if path == "/api/train":
                    return self.send(start_train(b))
                if path == "/api/report":
                    return self.send(report(b.get("config") or {}))
                if path == "/api/sync":
                    for d in datasets():
                        if not d["cached"]:
                            prepare_async(d["name"])
                    return self.send({"ok": True})
                if path == "/api/prepare":
                    prepare_async(b["dataset"])
                    return self.send({"ok": True})
                if path == "/api/compare":
                    names = [Path(n).name for n in b["runs"]]
                    out = T.RUNS / ("compare_" + "_vs_".join(names))
                    spawn([str(HERE / "train.py"), "compare", *[str(T.RUNS / n) for n in names], "--out", str(out)],
                          out / "compare.log")
                    return self.send({"ok": True, "dir": out.name})
                if path.startswith("/api/run/"):
                    parts = path.split("/")
                    run = T.RUNS / Path(parts[3]).name
                    act = parts[4]
                    info = run_info(run)
                    if act == "stop":
                        (run / "STOP").write_text("stop")
                        return self.send({"ok": True})
                    if info["alive"]:
                        raise ValueError("The run is busy — stop it first")
                    if act == "resume":
                        extra = [f"steps={int(b['steps'])}"] if b.get("steps") else []
                        spawn([str(HERE / "train.py"), "train", "--policy", info["policy"], "--name", run.name, "--resume",
                               "--set", *extra], run / "train.log")
                        return self.send({"ok": True})
                    if act == "eval":
                        spawn([str(HERE / "train.py"), "eval", "--run", str(run)], run / "train.log")
                        return self.send({"ok": True})
                    if act == "delete":
                        shutil.rmtree(run)
                        return self.send({"ok": True})
                return self.send({"error": "not found"}, 404)
            except Exception as e:                             # noqa: BLE001
                return self.send({"error": str(e)}, 400)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8260)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler())
    srv.daemon_threads = True
    print(f"NOVA Training · http://127.0.0.1:{a.port} · datasets {T.DATASETS} · runs {T.RUNS}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
