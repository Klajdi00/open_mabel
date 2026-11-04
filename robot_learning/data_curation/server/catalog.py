"""
Discover datasets and episodes under the dataset root and summarize them without
ever reading a full dataset (attrs + two scalar time reads only).

Layout (written by data_collection/recorder.py):
    DATASET_ROOT/<dataset>/episode_XXXX.h5
    DATASET_ROOT/<dataset>/dataset_meta.json   (optional)
"""
import glob
import json
import os

import h5py

from . import config as CFG


def _safe_attr(attrs, key, default=None):
    if key in attrs:
        v = attrs[key]
        return v.item() if hasattr(v, "item") else v
    return default


def _episode_card(path):
    """Cheap per-episode summary: attrs + first/last time sample only."""
    fname = os.path.basename(path)
    try:
        with h5py.File(path, "r") as f:
            a = f.attrs
            num = int(_safe_attr(a, "num_frames", 0) or 0)
            idx = int(_safe_attr(a, "episode_index", -1))
            task = _safe_attr(a, "task", "")
            created = _safe_attr(a, "created", "")
            schema = {}
            if "schema_json" in a:
                try:
                    schema = json.loads(a["schema_json"])
                except Exception:
                    schema = {}
            cams = [c["name"] for c in schema.get("cameras", [])]
            depth_cams = [c["name"] for c in schema.get("cameras", []) if c.get("depth")]
            # duration / fps from two scalar reads
            dur, fps = 0.0, 0.0
            if "time/sim" in f and num > 1:
                ts = f["time/sim"]
                t0, t1 = float(ts[0]), float(ts[-1])
                dur = t1 - t0
                fps = (num - 1) / dur if dur > 0 else 0.0
            modalities = []
            if any(k.startswith("obs/cam/") for k in _toplevel_keys(f)):
                modalities.append("rgb")
            if depth_cams:
                modalities.append("depth")
            if "obs/lidar" in f:
                modalities.append("lidar")
            if "obs/joint_pos" in f:
                modalities.append("joint")
            if any(("obs/pose/" + b) in f for b in ("chassis", "head")):
                modalities.append("pose")
            if any(k.startswith("teleop/") for k in _toplevel_keys(f)):
                modalities.append("teleop")
            return {
                "index": idx,
                "file": fname,
                "num_frames": num,
                "duration_s": round(dur, 3),
                "fps": round(fps, 3),
                "task": task,
                "created": created,
                "cameras": cams,
                "depth_cameras": depth_cams,
                "modalities": modalities,
                "size_bytes": os.path.getsize(path),
            }
    except Exception as e:
        return {"index": -1, "file": fname, "num_frames": 0, "error": str(e),
                "cameras": [], "modalities": [], "size_bytes": 0}


def _toplevel_keys(f):
    """All dataset paths (cheap visit, names only)."""
    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return out


def list_datasets():
    root = CFG.DATASET_ROOT
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        eps = glob.glob(os.path.join(d, "*.h5"))
        if not eps:
            continue
        meta = {}
        mp = os.path.join(d, "dataset_meta.json")
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except Exception:
                meta = {}
        out.append({"name": name, "n_episodes": len(eps),
                    "source": meta.get("source", "unknown")})
    return out


def list_episodes(dataset):
    d = os.path.join(CFG.DATASET_ROOT, dataset)
    eps = sorted(glob.glob(os.path.join(d, "*.h5")))
    return [_episode_card(p) for p in eps]


def episode_path(dataset, file_or_index):
    """Resolve an episode by filename ('episode_0001.h5') or numeric index."""
    d = os.path.join(CFG.DATASET_ROOT, dataset)
    if isinstance(file_or_index, str) and file_or_index.endswith(".h5"):
        p = os.path.join(d, file_or_index)
    else:
        try:
            i = int(file_or_index)
            p = os.path.join(d, f"episode_{i:04d}.h5")
        except (TypeError, ValueError):
            p = os.path.join(d, str(file_or_index))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def full_catalog():
    cat = []
    for ds in list_datasets():
        cat.append({**ds, "episodes": list_episodes(ds["name"])})
    return {"datasets": cat}
