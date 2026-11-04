"""
Light reader/writer for LeRobot v2.1 datasets — enough to compute stats, slice
features, and pull arrays/frames for eval + I/O viz WITHOUT importing the heavy
``lerobot`` package (that's only needed to actually train). Layout:

    <root>/meta/info.json | episodes.jsonl | episodes_stats.jsonl | tasks.jsonl
    <root>/data/chunk-{c:03d}/episode_{e:06d}.parquet   (observation.state, action, …)
    <root>/videos/chunk-{c:03d}/{video_key}/episode_{e:06d}.mp4
"""
import json
import os
import threading
from collections import OrderedDict

import numpy as np

CHUNK_SIZE = 1000

# (root, cam, ep) → list[jpeg bytes], so the I/O-viz plays ONE episode's frames
# after a single full decode instead of re-decoding from frame 0 every request
# (the old path was O(n²) and made dataset preview crawl). Bounded LRU.
_FRAME_CACHE = OrderedDict()
_FRAME_CACHE_MAX = 16          # cam-episodes kept resident
_FRAME_LOCKS = {}             # per-key lock so concurrent requests decode once
_FRAME_LOCKS_GUARD = threading.Lock()


# ── meta readers ──────────────────────────────────────────────────────────────
def read_info(root):
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


def read_jsonl(root, name):
    p = os.path.join(root, "meta", name)
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def is_lerobot_dataset(root):
    return os.path.isfile(os.path.join(root, "meta", "info.json"))


def chunk_of(ep_idx, info=None):
    size = (info or {}).get("chunks_size", CHUNK_SIZE)
    return ep_idx // size


def parquet_path(root, ep_idx, info=None):
    c = chunk_of(ep_idx, info)
    return os.path.join(root, "data", f"chunk-{c:03d}", f"episode_{ep_idx:06d}.parquet")


def video_path(root, video_key, ep_idx, info=None):
    c = chunk_of(ep_idx, info)
    return os.path.join(root, "videos", f"chunk-{c:03d}", video_key,
                        f"episode_{ep_idx:06d}.mp4")


def camera_keys(info):
    return [k for k, v in info.get("features", {}).items()
            if k.startswith("observation.images.") and v.get("dtype") == "video"]


def camera_names(info):
    return [k.split("observation.images.")[-1] for k in camera_keys(info)]


# ── size / stats ──────────────────────────────────────────────────────────────
def dir_size_bytes(root):
    total = 0
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def stats_record(root):
    """Full dataset-stats dict for the LEFT-rail cards + Info panel."""
    info = read_info(root)
    eps = read_jsonl(root, "episodes.jsonl")
    feats = info.get("features", {})
    fps = float(info.get("fps", 15.0)) or 15.0
    n_frames = int(info.get("total_frames", sum(e.get("length", 0) for e in eps)))
    n_eps = int(info.get("total_episodes", len(eps)))
    state = feats.get("observation.state", {})
    action = feats.get("action", {})
    cams = camera_names(info)
    cam_shapes = {c: feats.get(f"observation.images.{c}", {}).get("shape")
                  for c in cams}
    size_b = dir_size_bytes(root)
    return {
        "path": root,
        "name": os.path.basename(root.rstrip("/")),
        "robot_type": info.get("robot_type", "mabel"),
        "codebase_version": info.get("codebase_version", "v2.1"),
        "n_episodes": n_eps,
        "total_frames": n_frames,
        "total_timesteps": n_frames,
        "hours": round(n_frames / fps / 3600.0, 4),
        "seconds": round(n_frames / fps, 2),
        "fps": round(fps, 3),
        "size_mb": round(size_b / 1e6, 2),
        "n_cameras": len(cams),
        "cameras": cams,
        "camera_shapes": cam_shapes,
        "obs_state_dim": int(state.get("shape", [0])[0]) if state else 0,
        "action_dim": int(action.get("shape", [0])[0]) if action else 0,
        "obs_state_dtype": state.get("dtype"),
        "action_dtype": action.get("dtype"),
        "control_type": "joint position (absolute)",
        "modalities": (["images"] if cams else []) + (["proprio.state"] if state else []),
        "episode_lengths": [e.get("length", 0) for e in eps],
        "tasks": sorted({t for e in eps for t in (e.get("tasks") or [])}),
        "feature_keys": list(feats.keys()),
        # full per-feature schema {name: {shape, dtype}} for the detail view
        "features": {k: {"shape": v.get("shape"), "dtype": v.get("dtype"),
                         "names": v.get("names")} for k, v in feats.items()},
        # "what data does it have?" — a modality-capability breakdown with per-
        # modality rate (all LeRobot features are frame-synced → the dataset fps).
        "modality_report": _modality_report(feats, fps, state, action),
    }


def _modality_report(feats, fps, state, action):
    fk = list(feats.keys())
    def match(*subs):
        return [k for k in fk if any(s in k.lower() for s in subs)]
    rgb = [k for k in fk if k.startswith("observation.images") and "depth" not in k.lower()]
    depth = match("depth")
    pcd = match("point", "pcd", "cloud")
    lidar = match("lidar", "scan", "laser", "range")
    tf = match("tf", "transform", "pose", "odom")
    audio = match("audio", "mic")
    lang = [k for k in fk if "task" in k.lower() or "language" in k.lower() or "instruction" in k.lower()]

    def shp(keys):
        return feats.get(keys[0], {}).get("shape") if keys else None
    rate = round(fps, 3)
    return [
        {"type": "RGB camera", "present": bool(rgb), "count": len(rgb),
         "hz": rate if rgb else None, "shape": shp(rgb), "keys": rgb},
        {"type": "Depth", "present": bool(depth), "count": len(depth),
         "hz": rate if depth else None, "shape": shp(depth), "keys": depth},
        {"type": "Point cloud", "present": bool(pcd), "count": len(pcd),
         "hz": rate if pcd else None, "shape": shp(pcd), "keys": pcd},
        {"type": "LiDAR / laser", "present": bool(lidar), "count": len(lidar),
         "hz": rate if lidar else None, "shape": shp(lidar), "keys": lidar},
        {"type": "TF / poses", "present": bool(tf), "count": len(tf),
         "hz": rate if tf else None, "shape": shp(tf), "keys": tf},
        {"type": "Proprioception", "present": bool(state), "count": 1 if state else 0,
         "hz": rate if state else None, "shape": state.get("shape") if state else None, "keys": ["observation.state"] if state else []},
        {"type": "Action", "present": bool(action), "count": 1 if action else 0,
         "hz": rate if action else None, "shape": action.get("shape") if action else None, "keys": ["action"] if action else []},
        {"type": "Language / task", "present": bool(lang), "count": len(lang),
         "hz": rate if lang else None, "shape": None, "keys": lang},
        {"type": "Audio", "present": bool(audio), "count": len(audio),
         "hz": rate if audio else None, "shape": shp(audio), "keys": audio},
    ]


# ── array / frame access (for eval + io_viz) ──────────────────────────────────
def read_episode_arrays(root, ep_idx, keys=("observation.state", "action", "timestamp")):
    """Return {key: np.ndarray [T, dim]} for one episode via pyarrow."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(parquet_path(root, ep_idx, read_info(root)))
    out = {}
    for k in keys:
        if k not in tbl.column_names:
            continue
        col = tbl.column(k).to_pylist()
        arr = np.asarray(col, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out[k] = arr
    return out


def read_video_frame(root, cam, ep_idx, frame_i):
    """Decode a single RGB frame (H,W,3 uint8) from a camera mp4 via pyav.
    Sequential-scan; kept for eval/deploy. For the fast preview path use
    ``read_video_frame_jpeg`` (whole-episode cache)."""
    import av
    path = video_path(root, f"observation.images.{cam}", ep_idx, read_info(root))
    if not os.path.exists(path):
        return None
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_i:
                return frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    return None


def _decode_episode_jpegs(root, cam, ep_idx, quality=82):
    """Decode a whole camera-episode once → list of JPEG byte strings."""
    import av
    import cv2
    path = video_path(root, f"observation.images.{cam}", ep_idx, read_info(root))
    if not os.path.exists(path):
        return []
    frames = []
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"          # multi-threaded decode
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        for frame in container.decode(stream):
            bgr = frame.to_ndarray(format="bgr24")
            ok, buf = cv2.imencode(".jpg", bgr, enc)
            if ok:
                frames.append(buf.tobytes())
    finally:
        container.close()
    return frames


def read_video_frame_jpeg(root, cam, ep_idx, frame_i):
    """Return ONE frame's JPEG bytes from a cached whole-episode decode. The first
    call for a (root,cam,ep) decodes the entire clip once (fast); every later
    frame is an O(1) list index — so scrubbing/looping a preview is instant."""
    key = (str(root), cam, int(ep_idx))
    cached = _FRAME_CACHE.get(key)
    if cached is None:
        with _FRAME_LOCKS_GUARD:                       # one lock object per key
            lock = _FRAME_LOCKS.setdefault(key, threading.Lock())
        with lock:
            cached = _FRAME_CACHE.get(key)             # re-check inside the lock
            if cached is None:
                cached = _decode_episode_jpegs(root, cam, ep_idx)
                _FRAME_CACHE[key] = cached
                _FRAME_CACHE.move_to_end(key)
                while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
                    _FRAME_CACHE.popitem(last=False)
        with _FRAME_LOCKS_GUARD:
            _FRAME_LOCKS.pop(key, None)
    else:
        _FRAME_CACHE.move_to_end(key)
    if not cached:
        return None
    return cached[frame_i % len(cached)]
