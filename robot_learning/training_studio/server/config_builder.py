"""
Config builder — turns (architecture + hyperparameters + dataset view + split +
device) into a LeRobot ``train_config.json`` that ``lerobot-train --config_path``
consumes directly. This is the launch contract; it also pins the run to an exact
dataset version + embodiment view for reproducibility.

Notes on LeRobot 0.4.4 specifics:
  - the split is expressed as ``dataset.episodes`` (the train indices);
  - ``video_backend='pyav'`` avoids the broken torchcodec on this Mac;
  - ``wandb.enable=True`` is set so the (monkeypatched) StudioLogger is created —
    it writes metrics.jsonl and never touches wandb;
  - ``eval_freq=0`` disables mid-training sim eval (we deploy in MuJoCo on demand);
  - ``use_policy_training_preset=True`` builds the optimizer from the policy's
    preset using our ``optimizer_lr``.
"""
import json
import os

from . import arch_registry as AR
from . import compute


def build(arch, hp, sel, dataset_root, output_dir, config_path, train_episodes,
          repo_id=None, device="auto", resume=False, pretrained_ckpt=None,
          job_name=None):
    dev = compute.resolve_device(device)
    repo_id = repo_id or f"mabel/{os.path.basename(dataset_root.rstrip('/'))}"

    policy = AR.policy_block(arch, hp, sel, device=dev)
    policy["push_to_hub"] = False           # local training; no hub repo_id needed
    if pretrained_ckpt:
        policy["pretrained_path"] = pretrained_ckpt

    cfg = {
        "dataset": {
            "repo_id": repo_id,
            "root": os.path.abspath(dataset_root),
            "video_backend": "pyav",
            "use_imagenet_stats": True,
        },
        "policy": policy,
        "output_dir": os.path.abspath(output_dir),
        "job_name": job_name or f"{arch}_{sel.get('scope','whole_body')}",
        "seed": int(hp.get("seed", 1000)),
        "num_workers": int(hp.get("num_workers", 4)),
        "batch_size": int(hp.get("batch_size", 8)),
        "steps": int(hp.get("steps", 5000)),
        "log_freq": int(hp.get("log_freq", 50)),
        "save_freq": int(hp.get("save_freq", 1000)),
        "save_checkpoint": True,
        # curator fps (~15.15) yields float32 timestamps a hair off the encoded
        # video frames; a 20ms tolerance (< 1/3 frame) absorbs it safely.
        "tolerance_s": 0.02,
        "eval_freq": 0,                      # no gym env → deploy in MuJoCo on demand
        "use_policy_training_preset": True,
        "resume": bool(resume),
        "wandb": {"enable": True, "project": "studio"},   # → StudioLogger
    }
    # Train only on the split's train episodes (LeRobot selects by index).
    if train_episodes is not None:
        cfg["dataset"]["episodes"] = list(train_episodes)

    # IMPORTANT: never create output_dir here — LeRobot refuses a pre-existing one
    # (unless resume). The config file lives OUTSIDE output_dir.
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return {"path": config_path, "config": cfg, "device": dev}
