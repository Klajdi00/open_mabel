"""run_lerobot.py — standalone LeRobot training wrapper (runs in the LeRobot env).

The studio backend (mabel env) spawns THIS with the LeRobot-env python. It drives
`lerobot-train` and translates its stdout into the studio's `metrics.jsonl`
contract (`_type` in {config,start,metric,checkpoint}) so the existing Tailer →
SQLite → dashboard pipeline works unchanged, without depending on LeRobot's
internal WandBLogger. Stdlib only — no studio imports (different env).

Usage (all after `--`, LeRobot passthrough optional via --extra):
    python run_lerobot.py --policy act --dataset-root <v30> --repo-id mabel/x \
        --output-dir <run>/train --steps 20000 --batch 8 --save-freq 5000 \
        --device cuda [--resume] [--extra --policy.chunk_size=50 ...]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time

# UI arch → LeRobot policy.type
ARCH_POLICY = {"act": "act", "dp": "diffusion", "diffusion": "diffusion",
               "pi0": "pi0", "pi05": "pi05", "pi0.5": "pi05", "smolvla": "smolvla"}

_KV = re.compile(r"(\w[\w/]*)\s*:\s*([-+0-9.eE]+)")
_METRIC_KEYS = {"loss": "loss", "grdn": "grad_norm", "lr": "lr",
                "smp/s": "samples_per_sec", "updt_s": "update_s",
                "data_s": "dataloading_s", "mem_gb": "gpu_mem_used_gb", "epch": "epoch"}
_CKPT = re.compile(r"Checkpoint policy after step (\d+)")


def _emit(mpath, rec):
    with open(mpath, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="mabel/dataset")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metrics-path", default=None,
                    help="where to write metrics.jsonl; MUST be outside --output-dir "
                         "(LeRobot refuses a pre-existing output_dir). Defaults to a "
                         "sibling metrics/ dir.")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config-path", default=None,
                    help="on resume: the checkpoint's train_config.json (LeRobot 0.6 "
                         "requires --config_path to resume).")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    policy = ARCH_POLICY.get(args.policy.lower(), args.policy.lower())
    # metrics.jsonl must NOT live inside output_dir (LeRobot refuses a pre-existing
    # output_dir). Default to a sibling metrics/ dir next to the run root.
    mpath = args.metrics_path or os.path.join(
        os.path.dirname(args.output_dir.rstrip("/")), "metrics", "metrics.jsonl")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)

    if args.resume:
        # LeRobot 0.6 resume: point at the checkpoint's train_config.json (it carries
        # output_dir/policy/dataset); only steps + device are overridden on top.
        cp = args.config_path or os.path.join(
            args.output_dir, "checkpoints", "last", "pretrained_model", "train_config.json")
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
               f"--config_path={cp}", "--resume=true", f"--steps={args.steps}",
               f"--policy.device={args.device}", "--wandb.enable=false"]
    else:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train",
               f"--policy.type={policy}", f"--policy.device={args.device}",
               "--policy.push_to_hub=false",
               f"--dataset.repo_id={args.repo_id}", f"--dataset.root={args.dataset_root}",
               f"--output_dir={args.output_dir}", f"--steps={args.steps}",
               f"--batch_size={args.batch}", f"--save_freq={args.save_freq}",
               f"--log_freq={args.log_freq}", f"--num_workers={args.num_workers}",
               "--wandb.enable=false"]
        cmd += [a for a in args.extra if a != "--"]

    _emit(mpath, {"_type": "config", "policy": policy, "steps": args.steps,
                  "batch": args.batch, "device": args.device,
                  "cmd": " ".join(cmd), "t": time.time()})
    _emit(mpath, {"_type": "start", "t": time.time()})
    print("[run_lerobot] " + " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        ck = _CKPT.search(line)
        if ck:
            step = int(ck.group(1))
            _emit(mpath, {"_type": "checkpoint", "step": step,
                          "path": os.path.join(args.output_dir, "checkpoints", f"{step:06d}")})
            continue
        if "step:" not in line:
            continue
        pairs = dict(_KV.findall(line))
        if "step" not in pairs:
            continue
        rec = {"_type": "metric", "step": int(float(pairs["step"]))}
        for k, out in _METRIC_KEYS.items():
            if k in pairs:
                try:
                    rec[out] = float(pairs[k])
                except ValueError:
                    pass
        _emit(mpath, rec)
    rc = proc.wait()
    _emit(mpath, {"_type": "end", "returncode": rc, "t": time.time()})
    sys.exit(rc)


if __name__ == "__main__":
    main()
