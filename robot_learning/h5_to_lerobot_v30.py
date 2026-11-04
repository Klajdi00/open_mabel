"""h5_to_lerobot_v30.py — build a LeRobot **v3.0** dataset directly from MABEL
curation `episode_XXXX.h5` files, using LeRobot's own native writer.

Why not convert the v2.1 export: LeRobot 0.6.0 trains on v3.0 only, and its
`convert_dataset_v21_to_v30` re-muxes our MP4s through a PyAV concat path that
chokes on our encoder settings. Writing v3.0 natively (LeRobotDataset.create +
add_frame + save_episode) sidesteps that entirely and is the robust train-data
builder. RGB is decoded from the h5 JPEG blobs (lossless re-decode).

    python3 h5_to_lerobot_v30.py --sessions '/…/real/vision_os/pick_place_*' \
        --repo-id mabel/pick_place --out <v30_root> [--limit N] [--cams zed_left,...]
"""
from __future__ import annotations
import argparse, glob, json, os

import cv2
import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _episodes(sessions):
    h5s = []
    for hit in sorted(glob.glob(sessions)):
        if hit.endswith(".h5"):
            h5s.append(hit)
        elif os.path.isdir(hit):
            h5s += sorted(glob.glob(os.path.join(hit, "*.h5")))
    return [h for h in h5s if os.path.getsize(h) > 0]


def _cam_names(f):
    return sorted(k for k in f["obs/cam"].keys()) if "obs/cam" in f else []


def build(sessions, repo_id, out_root, fps=10, limit=None, only_cams=None, task="mabel task"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    h5s = _episodes(sessions)
    if limit:
        h5s = h5s[:limit]
    if not h5s:
        raise SystemExit(f"no episodes matched {sessions!r}")

    # derive features from the first episode
    with h5py.File(h5s[0], "r") as f:
        state_dim = f["obs/joint_pos"].shape[1]
        act_dim = f["action/ctrl"].shape[1]
        cams = _cam_names(f)
        if only_cams:
            cams = [c for c in cams if c in only_cams]
        # decode one rgb to get HxW
        sample = cv2.imdecode(np.frombuffer(f[f"obs/cam/{cams[0]}/rgb"][0].tobytes(), np.uint8),
                              cv2.IMREAD_COLOR)
        h, w = sample.shape[:2]

    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [act_dim], "names": None},
    }
    for c in cams:
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"]}

    if os.path.exists(out_root):
        import shutil; shutil.rmtree(out_root)
    ds = LeRobotDataset.create(repo_id, fps=int(round(fps)), features=features,
                               root=out_root, robot_type="mabel", use_videos=True)
    print(f"creating v3.0 dataset '{repo_id}' @ {out_root}  cams={cams}  {w}x{h}  "
          f"state={state_dim} act={act_dim}")

    for ei, hp in enumerate(h5s):
        with h5py.File(hp, "r") as f:
            state = f["obs/joint_pos"][:].astype(np.float32)
            action = f["action/ctrl"][:].astype(np.float32)
            T = len(state)
            rgb = {c: f[f"obs/cam/{c}/rgb"] for c in cams}
            for t in range(T):
                frame = {"observation.state": state[t], "action": action[t], "task": task}
                for c in cams:
                    img = cv2.imdecode(np.frombuffer(rgb[c][t].tobytes(), np.uint8), cv2.IMREAD_COLOR)
                    frame[f"observation.images.{c}"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ds.add_frame(frame)
            ds.save_episode()
        print(f"  + episode {ei} ({os.path.basename(os.path.dirname(hp))}): {T} frames")
    print(f"DONE: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames -> {out_root}")
    return out_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--repo-id", default="mabel/pick_place")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cams", default=None, help="comma list to keep (default all)")
    ap.add_argument("--task", default="mabel manipulation")
    args = ap.parse_args()
    only = set(args.cams.split(",")) if args.cams else None
    build(args.sessions, args.repo_id, args.out, fps=args.fps, limit=args.limit,
          only_cams=only, task=args.task)


if __name__ == "__main__":
    main()
