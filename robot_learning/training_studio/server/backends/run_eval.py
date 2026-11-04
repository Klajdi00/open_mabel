"""run_eval.py — evaluate a trained checkpoint (open-loop) on held-out data.

Runs in the LeRobot env (spawned by experiment.ExperimentManager.evaluate). Loads
ANY checkpoint's `pretrained_model/`, replays it over dataset episode(s) with the
policy's own action selection, and reports per-joint + overall action error plus
the held-out forward loss. Writes results.json for the eval tree. Stdlib+torch+lerobot.

    python run_eval.py --checkpoint <ckpt>/pretrained_model --dataset-root <v30> \
        --repo-id mabel/x --out results.json --episodes 1 --device cuda
"""
from __future__ import annotations
import argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="mabel/dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = dev
    policy = make_policy(cfg, ds_meta=ds.meta)
    policy.to(dev).eval()
    pre = post = None   # 0.6 folds normalization into the policy; no external processors

    # obs-only open-loop rollout on logged observations: feed each frame's obs to
    # the policy, compare its selected action to the logged (ground-truth) action.
    obs_keys = [k for k in ds.features if k.startswith("observation.")]
    ep_ids = list(range(min(args.episodes, ds.meta.total_episodes)))
    per_joint_l1, all_err = [], []
    n_frames, first_err = 0, None
    for ep in ep_ids:
        idxs = [i for i in range(len(ds)) if int(ds[i]["episode_index"]) == ep]
        if hasattr(policy, "reset"):
            policy.reset()
        for i in idxs:
            item = ds[i]
            batch = {k: (item[k].unsqueeze(0).to(dev) if torch.is_tensor(item[k]) else item[k])
                     for k in obs_keys if k in item}
            batch["task"] = item.get("task", "")
            gt = item["action"].numpy()
            gt = gt[0] if gt.ndim > 1 else gt
            with torch.no_grad():
                try:
                    act = policy.select_action(batch)
                    pred = act.squeeze(0).float().cpu().numpy()
                    pred = pred[0] if pred.ndim > 1 else pred
                    m = min(len(pred), len(gt))
                    err = np.abs(pred[:m] - gt[:m])
                    per_joint_l1.append(err)
                    all_err.append(float(err.mean()))
                except Exception as e:
                    if first_err is None:
                        first_err = f"{type(e).__name__}: {e}"
            n_frames += 1

    pj = np.stack(per_joint_l1).mean(0).tolist() if per_joint_l1 else None
    results = {
        "checkpoint": args.checkpoint, "episodes": ep_ids, "frames": n_frames,
        "action_l1_mean": (float(np.mean(all_err)) if all_err else None),
        "action_l1_per_joint": pj, "select_action_error": first_err, "device": dev,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("EVAL:", json.dumps({k: results[k] for k in
          ("frames", "action_l1_mean", "select_action_error")}, indent=2))


if __name__ == "__main__":
    main()
