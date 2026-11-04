"""
Export a (curated) MABEL episode HDF5 to a LeRobot v2.1 dataset:

    <out>/meta/info.json | tasks.jsonl | episodes.jsonl | episodes_stats.jsonl
    <out>/data/chunk-000/episode_000000.parquet          (observation.state, action, indices)
    <out>/videos/chunk-000/observation.images.<cam>/episode_000000.mp4

observation.state = obs/joint_pos, action = action/ctrl; each RGB camera becomes a
video feature. Needs pyarrow (parquet) + imageio-ffmpeg (mp4) + cv2 (JPEG decode).
"""
import json
import os
import shutil

import numpy as np

from . import h5pool


def _fps(f):
    for k in ("time/sim", "time/wall"):
        if k in f:
            t = np.asarray(f[k][:], float)
            if t.size >= 2 and t[-1] > t[0]:
                return float((t.size - 1) / (t[-1] - t[0]))
    return 15.0


def _stats(a, T):
    a = np.asarray(a, np.float64).reshape(T, -1)
    cl = lambda x: np.nan_to_num(x, nan=0.0).tolist()          # NaN-safe (severe data may have NaN)
    with np.errstate(all="ignore"):
        return {"min": cl(np.nanmin(a, 0)), "max": cl(np.nanmax(a, 0)),
                "mean": cl(np.nanmean(a, 0)), "std": cl(np.nanstd(a, 0)), "count": [T]}


def _read_episode(h5_path):
    """Decode one MABEL episode into (T, fps, state, action, task, {cam:frames})."""
    import cv2
    with h5pool.reading(h5_path) as f:
        T = int(f.attrs.get("num_frames", f["action/ctrl"].shape[0]))
        fps = _fps(f)
        state = np.asarray(f["obs/joint_pos"][:T], np.float32)
        action = np.asarray(f["action/ctrl"][:T], np.float32)
        task = str(f.attrs.get("task", "mabel task"))
        sj = json.loads(f.attrs.get("schema_json", "{}"))
        cams = [c["name"] for c in sj.get("cameras", []) if c.get("rgb")]
        cam_frames = {}
        for cam in cams:
            k = f"obs/cam/{cam}/rgb"
            if k not in f:
                continue
            frames = []
            for i in range(T):
                im = cv2.imdecode(np.frombuffer(np.asarray(f[k][i], np.uint8), np.uint8), cv2.IMREAD_COLOR)
                frames.append((im[:, :, ::-1] if im is not None else np.zeros((270, 480, 3), np.uint8)))
            cam_frames[cam] = frames
    return T, fps, state, action, task, cam_frames


def to_lerobot_dataset(h5_paths, out_dir, repo_id="mabel/curated", progress=None):
    """Aggregate MANY curated episodes into ONE LeRobot v2.1 dataset (the real
    multi-episode layout: episode_{i:06d}.parquet + per-cam mp4, a global `index`
    running across episodes, unique tasks.jsonl, one episodes(_stats).jsonl line
    per episode). This is what a finished, whole-folder export produces."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import imageio

    h5_paths = list(h5_paths)
    if not h5_paths:
        raise ValueError("no episodes to export")
    out = os.path.join(out_dir, repo_id.replace("/", "_"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out, "data", "chunk-000"), exist_ok=True)

    tasks = []                                  # unique task strings; index = position
    def task_idx(t):
        if t not in tasks:
            tasks.append(t)
        return tasks.index(t)

    ep_lines, stat_lines = [], []
    cam_hw, all_cams = {}, []
    total_frames = global_index = 0
    fps = 15.0
    state_dim = action_dim = 0
    for ei, h5 in enumerate(h5_paths):
        if progress:
            progress(ei, len(h5_paths), 0)
        T, fps, state, action, task, cam_frames = _read_episode(h5)
        ti = task_idx(task)
        for cam, frames in cam_frames.items():
            vdir = os.path.join(out, "videos", "chunk-000", f"observation.images.{cam}")
            os.makedirs(vdir, exist_ok=True)
            h, w = frames[0].shape[:2]; cam_hw[cam] = (h, w)
            if cam not in all_cams:
                all_cams.append(cam)
            wtr = imageio.get_writer(os.path.join(vdir, f"episode_{ei:06d}.mp4"), fps=fps,
                                     codec="libx264", macro_block_size=None, quality=8)
            for fr in frames:
                wtr.append_data(fr)
            wtr.close()
        idx = np.arange(T)
        tbl = pa.table({
            "observation.state": [state[i].tolist() for i in range(T)],
            "action": [action[i].tolist() for i in range(T)],
            "timestamp": (idx / fps).astype(np.float32),
            "frame_index": idx.astype(np.int64),
            "episode_index": np.full(T, ei, np.int64),
            "index": (global_index + idx).astype(np.int64),      # GLOBAL across episodes
            "task_index": np.full(T, ti, np.int64),
        })
        pq.write_table(tbl, os.path.join(out, "data", "chunk-000", f"episode_{ei:06d}.parquet"))
        ep_lines.append({"episode_index": ei, "tasks": [task], "length": T})
        stat_lines.append({"episode_index": ei, "stats": {
            "observation.state": _stats(state, T), "action": _stats(action, T)}})
        total_frames += T; global_index += T
        state_dim, action_dim = int(state.shape[1]), int(action.shape[1])

    N = len(h5_paths)
    vid_feats = {f"observation.images.{cam}": {
        "dtype": "video", "shape": [cam_hw[cam][0], cam_hw[cam][1], 3],
        "names": ["height", "width", "channel"],
        "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False}}
        for cam in all_cams}
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        **vid_feats,
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "mabel",
        "total_episodes": N, "total_frames": total_frames, "total_tasks": len(tasks),
        "total_videos": N * len(all_cams), "total_chunks": 1, "chunks_size": 1000,
        "fps": fps, "splits": {"train": f"0:{N}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    json.dump(info, open(os.path.join(out, "meta", "info.json"), "w"), indent=2)
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as fh:
        for i, t in enumerate(tasks):
            fh.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as fh:
        for e in ep_lines:
            fh.write(json.dumps(e) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as fh:
        for s in stat_lines:
            fh.write(json.dumps(s) + "\n")
    return {"path": out, "episodes": N, "frames": total_frames, "fps": round(fps, 2),
            "tasks": len(tasks), "cameras": all_cams}


def to_lerobot(h5_path, out_dir, repo_id="mabel/curated"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import cv2
    import imageio

    with h5pool.reading(h5_path) as f:
        T = int(f.attrs.get("num_frames", f["action/ctrl"].shape[0]))
        fps = _fps(f)
        state = np.asarray(f["obs/joint_pos"][:T], np.float32)
        action = np.asarray(f["action/ctrl"][:T], np.float32)
        task = str(f.attrs.get("task", "mabel task"))
        sj = json.loads(f.attrs.get("schema_json", "{}"))
        cams = [c["name"] for c in sj.get("cameras", []) if c.get("rgb")]
        cam_frames = {}
        for cam in cams:
            k = f"obs/cam/{cam}/rgb"
            if k not in f:
                continue
            frames = []
            for i in range(T):
                im = cv2.imdecode(np.frombuffer(np.asarray(f[k][i], np.uint8), np.uint8), cv2.IMREAD_COLOR)
                frames.append((im[:, :, ::-1] if im is not None else np.zeros((270, 480, 3), np.uint8)))
            cam_frames[cam] = frames

    out = os.path.join(out_dir, repo_id.replace("/", "_"))
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    os.makedirs(os.path.join(out, "data", "chunk-000"), exist_ok=True)

    # ── videos (one mp4 per RGB camera) ───────────────────────────────────────
    vid_feats = {}
    for cam, frames in cam_frames.items():
        vdir = os.path.join(out, "videos", "chunk-000", f"observation.images.{cam}")
        os.makedirs(vdir, exist_ok=True)
        h, w = frames[0].shape[:2]
        wtr = imageio.get_writer(os.path.join(vdir, "episode_000000.mp4"), fps=fps,
                                 codec="libx264", macro_block_size=None, quality=8)
        for fr in frames:
            wtr.append_data(fr)
        wtr.close()
        vid_feats[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
            "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False}}

    # ── parquet (low-rate signals + LeRobot index columns) ────────────────────
    idx = np.arange(T)
    tbl = pa.table({
        "observation.state": [state[i].tolist() for i in range(T)],
        "action": [action[i].tolist() for i in range(T)],
        "timestamp": (idx / fps).astype(np.float32),
        "frame_index": idx.astype(np.int64),
        "episode_index": np.zeros(T, np.int64),
        "index": idx.astype(np.int64),
        "task_index": np.zeros(T, np.int64),
    })
    pq.write_table(tbl, os.path.join(out, "data", "chunk-000", "episode_000000.parquet"))

    # ── meta ──────────────────────────────────────────────────────────────────
    features = {
        "observation.state": {"dtype": "float32", "shape": [int(state.shape[1])], "names": None},
        "action": {"dtype": "float32", "shape": [int(action.shape[1])], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        **vid_feats,
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "mabel",
        "total_episodes": 1, "total_frames": T, "total_tasks": 1,
        "total_videos": len(vid_feats), "total_chunks": 1, "chunks_size": 1000, "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    json.dump(info, open(os.path.join(out, "meta", "info.json"), "w"), indent=2)
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as fh:
        fh.write(json.dumps({"episode_index": 0, "tasks": [task], "length": T}) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as fh:
        fh.write(json.dumps({"episode_index": 0,
                             "stats": {"observation.state": _stats(state, T), "action": _stats(action, T)}}) + "\n")

    return {"path": out, "frames": T, "fps": round(fps, 2), "cameras": list(cam_frames.keys())}
