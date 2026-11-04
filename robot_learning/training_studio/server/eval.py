"""
Open-loop evaluation — run the trained network over held-out TEST episodes and
compare predicted actions to ground truth, frame by frame. This is the "action
output line graph LIVE from the trained neural network" over the same
data-curation-style timeline. Uses LeRobot's canonical inference path
(make_policy + make_pre_post_processors + predict_action) on the checkpoint.
"""
import os

import numpy as np

from . import config as CFG
from . import lerobot_dataset as L


def _load_policy(checkpoint_dir, ds_meta_root, device):
    import torch  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    pm = os.path.join(checkpoint_dir, "pretrained_model")
    if not os.path.isdir(pm):
        pm = checkpoint_dir
    cfg = PreTrainedConfig.from_pretrained(pm)
    cfg.pretrained_path = pm
    cfg.device = device
    meta = LeRobotDatasetMetadata(os.path.basename(ds_meta_root.rstrip("/")), root=ds_meta_root)
    policy = make_policy(cfg, ds_meta=meta)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=pm, dataset_stats=meta.stats)
    return policy, pre, post, cfg, meta


def _decode_all_frames(root, cam, ep_idx, T, info):
    """Decode up to T RGB frames of one camera in a single pass (fast)."""
    import av
    path = L.video_path(root, f"observation.images.{cam}", ep_idx, info)
    out = []
    if not os.path.exists(path):
        return out
    container = av.open(path)
    try:
        for i, frame in enumerate(container.decode(container.streams.video[0])):
            if i >= T:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return out


def run_open_loop(checkpoint_dir, view_v21_root, ds_meta_root, episode_idx,
                  device="cpu", max_frames=200, progress=None):
    """Return predicted-vs-GT action arrays + per-joint error for one episode."""
    import torch
    from lerobot.utils.control_utils import predict_action

    policy, pre, post, cfg, meta = _load_policy(checkpoint_dir, ds_meta_root, device)
    dev = torch.device(device)
    cams = [k.split("observation.images.")[-1] for k in getattr(cfg, "input_features", {})
            if k.startswith("observation.images.")]

    info = L.read_info(view_v21_root)
    arrs = L.read_episode_arrays(view_v21_root, episode_idx)
    state, gt = arrs["observation.state"], arrs["action"]
    T = int(min(len(gt), max_frames)) if max_frames else len(gt)
    frames = {c: _decode_all_frames(view_v21_root, c, episode_idx, T, info) for c in cams}
    ep_meta = next((m for m in L.read_jsonl(view_v21_root, "episodes.jsonl")
                    if m.get("episode_index") == episode_idx), {})
    task = (ep_meta.get("tasks") or ["mabel task"])[0]

    policy.reset()
    preds = []
    for f in range(T):
        obs = {"observation.state": state[f].astype(np.float32)}
        for c in cams:
            if f < len(frames[c]):
                obs[f"observation.images.{c}"] = frames[c][f]
        act = predict_action(obs, policy, dev, pre, post, use_amp=False, task=task)
        preds.append(np.asarray(act.detach().to("cpu")).reshape(-1))
        if progress:
            progress(f + 1, T, stage=0)

    pred = np.stack(preds)[:, : gt.shape[1]]
    gt = gt[:T]
    l1 = np.abs(pred - gt).mean(axis=0)
    mse = ((pred - gt) ** 2).mean(axis=0)
    return {
        "episode": episode_idx, "frames": T,
        "action_dim": int(gt.shape[1]),
        "pred": pred.astype(float).tolist(),
        "gt": gt.astype(float).tolist(),
        "l1_per_joint": l1.astype(float).tolist(),
        "mse_per_joint": mse.astype(float).tolist(),
        "l1_mean": float(l1.mean()), "mse_mean": float(mse.mean()),
        "cameras": cams,
    }
