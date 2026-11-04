"""
Dataset registry — the trainer's dataset picker.

Scans the curator's LeRobot exports (and the studio's own derived datasets),
reports full per-dataset stats, and provides:
  - combine():  concatenate several (often single-episode) exports into ONE
                multi-episode LeRobot v2.1 dataset (renumbered episode/index).
  - split():    episode-index lists for train/eval/test (an early UI parameter);
                LeRobot 0.4.4 selects episodes via dataset.episodes, so the split
                is just which indices go to each bucket.
"""
import hashlib
import json
import os
import random
import shutil

import numpy as np

from . import config as CFG
from . import lerobot_dataset as L

# Studio keeps datasets in the readable v2.1 layout; LeRobot 0.4.4 trains on v3.0,
# so we materialize a throwaway v3.0 copy here right before launching a run.
V30_DIR = os.path.join(CFG.DATASETS_DIR, ".v30")
_V30_STAGE = os.path.join(CFG.DATASETS_DIR, ".v30_stage")


def _scan_dir(base, source):
    out = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        root = os.path.join(base, name)
        if L.is_lerobot_dataset(root):
            try:
                rec = L.stats_record(root)
                rec["dataset_id"] = name
                rec["source"] = source
                rec["version_hash"] = version_hash(root)
                out.append(rec)
            except Exception as e:  # pragma: no cover
                out.append({"dataset_id": name, "path": root, "source": source,
                            "error": str(e)})
    return out


import glob as _glob


def _scan_curated(base, source="curated"):
    """Scan the curated_dataset tree: curated_dataset/{real,simulation}/<name_NNN_date>/<repo>/.
    dataset_id = the curation-session folder name (human-legible, unique)."""
    out = []
    if not os.path.isdir(base):
        return out
    for info in sorted(_glob.glob(os.path.join(base, "*", "*", "*", "meta", "info.json"))):
        root = os.path.dirname(os.path.dirname(info))          # .../<repo>
        session = os.path.basename(os.path.dirname(root))       # <name_NNN_date>
        realm = os.path.basename(os.path.dirname(os.path.dirname(root)))  # real|simulation
        if not L.is_lerobot_dataset(root):
            continue
        try:
            rec = L.stats_record(root)
            rec.update({"dataset_id": session, "source": source, "realm": realm,
                        "version_hash": version_hash(root)})
            out.append(rec)
        except Exception as e:  # pragma: no cover
            out.append({"dataset_id": session, "path": root, "source": source, "error": str(e)})
    return out


def _curated_index():
    """dataset_id -> lerobot root, for find()."""
    idx = {}
    base = getattr(CFG, "CURATED_ROOT", None)
    if base and os.path.isdir(base):
        for info in _glob.glob(os.path.join(base, "*", "*", "*", "meta", "info.json")):
            root = os.path.dirname(os.path.dirname(info))
            idx[os.path.basename(os.path.dirname(root))] = root
    return idx


def list_datasets():
    """All available datasets: curated exports (new tree) + legacy curator + studio views."""
    curated = _scan_curated(getattr(CFG, "CURATED_ROOT", ""))
    cur = _scan_dir(CFG.CURATOR_EXPORTS, "curator")
    own = _scan_dir(CFG.DATASETS_DIR, "studio")
    return {"datasets": curated + cur + own}


def version_hash(root):
    """Stable hash of the dataset meta (for pinning a run to an exact version)."""
    try:
        info = L.read_info(root)
        eps = L.read_jsonl(root, "episodes.jsonl")
        payload = json.dumps({"info": info, "n": len(eps),
                              "lens": [e.get("length") for e in eps]}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def import_dataset(path):
    """Register an external LeRobot dataset folder by absolute path (symlinked
    into the studio datasets dir so it appears in the picker). Accepts v2.1 or
    v3.0 layouts."""
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"not a folder: {path}")
    if not L.is_lerobot_dataset(path):
        raise ValueError(f"not a LeRobot dataset (no meta/info.json): {path}")
    name = os.path.basename(path.rstrip("/"))
    dst = os.path.join(CFG.DATASETS_DIR, name)
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(path):
            pass  # already imported
        else:
            name = f"{name}_{version_hash(path)}"
            dst = os.path.join(CFG.DATASETS_DIR, name)
    if not os.path.exists(dst):
        os.symlink(path, dst)
    rec = L.stats_record(dst)
    rec.update({"dataset_id": name, "source": "studio", "version_hash": version_hash(dst),
                "imported_from": path})
    return rec


def find(dataset_id):
    for base, src in ((CFG.CURATOR_EXPORTS, "curator"), (CFG.DATASETS_DIR, "studio")):
        root = os.path.join(base, dataset_id)
        if L.is_lerobot_dataset(root):
            return root
    curated = _curated_index()          # dataset_id -> <repo> root in curated_dataset tree
    if dataset_id in curated:
        return curated[dataset_id]
    raise FileNotFoundError(f"no such dataset: {dataset_id}")


# ── combine several exports into one multi-episode dataset ────────────────────
def combine(dataset_ids, out_name, task_default="mabel task", progress=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    roots = [find(d) for d in dataset_ids]
    if not roots:
        raise ValueError("no datasets to combine")
    base_info = L.read_info(roots[0])
    feats0 = base_info["features"]
    cam_keys0 = set(L.camera_keys(base_info))
    for r in roots[1:]:
        info = L.read_info(r)
        if info["features"].get("action", {}).get("shape") != feats0.get("action", {}).get("shape"):
            raise ValueError(f"action dim mismatch: {r}")
        if set(L.camera_keys(info)) != cam_keys0:
            raise ValueError(f"camera set mismatch: {r}")

    # gather source episodes (each source may itself be multi-episode)
    src_eps = []  # (root, src_ep_idx, info)
    for r in roots:
        info = L.read_info(r)
        for e in range(int(info.get("total_episodes", 1))):
            src_eps.append((r, e, info))
    rec = _assemble(src_eps, out_name, cam_keys0, base_info, progress)
    rec["combined_from"] = list(dataset_ids)
    return rec


def subset_dataset(src_root, indices, out_name, progress=None):
    """Materialize a new dataset from a subset of ``src_root``'s episodes,
    renumbered 0..k-1. Used to make a train-only dataset so training never has to
    select non-zero episode indices (which trips a LeRobot global/local index bug
    with multi-step temporal windows)."""
    info = L.read_info(src_root)
    cam_keys0 = set(L.camera_keys(info))
    src_eps = [(src_root, int(i), info) for i in indices]
    return _assemble(src_eps, out_name, cam_keys0, info, progress)


def _assemble(src_eps, out_name, cam_keys0, base_info, progress=None, task_default="mabel task"):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(CFG.DATASETS_DIR, out_name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)

    tasks_index = {}       # task string -> index
    ep_records, stat_records = [], []
    global_index = 0
    total_frames = 0
    n_out = len(src_eps)

    for out_ep, (root, se, info) in enumerate(src_eps):
        tbl = pq.read_table(L.parquet_path(root, se, info))
        T = tbl.num_rows
        # episode task
        ep_meta = next((m for m in L.read_jsonl(root, "episodes.jsonl")
                        if m.get("episode_index") == se), {})
        task = (ep_meta.get("tasks") or [task_default])[0]
        ti = tasks_index.setdefault(task, len(tasks_index))
        # rewrite index columns
        cols, names = [], []
        for name in tbl.column_names:
            if name == "episode_index":
                cols.append(pa.array(np.full(T, out_ep, np.int64)))
            elif name == "index":
                cols.append(pa.array(np.arange(global_index, global_index + T, dtype=np.int64)))
            elif name == "task_index":
                cols.append(pa.array(np.full(T, ti, np.int64)))
            else:
                cols.append(tbl.column(name))
            names.append(name)
        c = out_ep // base_info.get("chunks_size", 1000)
        ddir = os.path.join(out, "data", f"chunk-{c:03d}")
        os.makedirs(ddir, exist_ok=True)
        pq.write_table(pa.table(cols, names=names),
                       os.path.join(ddir, f"episode_{out_ep:06d}.parquet"))
        # videos
        for key in cam_keys0:
            src_mp4 = L.video_path(root, key, se, info)
            if os.path.exists(src_mp4):
                vdir = os.path.join(out, "videos", f"chunk-{c:03d}", key)
                os.makedirs(vdir, exist_ok=True)
                shutil.copy2(src_mp4, os.path.join(vdir, f"episode_{out_ep:06d}.mp4"))
        # meta records
        ep_records.append({"episode_index": out_ep, "tasks": [task], "length": T})
        srec = next((m for m in L.read_jsonl(root, "episodes_stats.jsonl")
                     if m.get("episode_index") == se), {"stats": {}})
        stat_records.append({"episode_index": out_ep, "stats": srec.get("stats", {})})
        global_index += T
        total_frames += T
        if progress:
            progress(out_ep + 1, n_out, stage=1)

    # meta/info.json
    new_info = json.loads(json.dumps(base_info))
    new_info.update({"total_episodes": n_out, "total_frames": total_frames,
                     "total_tasks": len(tasks_index),
                     "total_videos": len(cam_keys0) * n_out,
                     "total_chunks": (n_out // base_info.get("chunks_size", 1000)) + 1,
                     "splits": {"train": f"0:{n_out}"}})
    json.dump(new_info, open(os.path.join(out, "meta", "info.json"), "w"), indent=2)
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as fh:
        for t, i in sorted(tasks_index.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as fh:
        for r in ep_records:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "meta", "episodes_stats.jsonl"), "w") as fh:
        for r in stat_records:
            fh.write(json.dumps(r) + "\n")

    rec = L.stats_record(out)
    rec.update({"dataset_id": out_name, "source": "studio",
                "version_hash": version_hash(out)})
    return rec


def _inject_image_stats(v30_root):
    """The curator export writes only state/action stats; LeRobot's make_dataset
    needs a stats entry per camera key (imagenet overrides mean/std at load, but
    the key must exist). Image stats are per-channel, shape (3,1,1), range [0,1]."""
    sp = os.path.join(v30_root, "meta", "stats.json")
    if not os.path.isfile(sp):
        return
    info = L.read_info(v30_root)
    stats = json.load(open(sp))
    n = int(info.get("total_frames", 1))
    changed = False
    img = {"mean": [[[0.485]], [[0.456]], [[0.406]]],
           "std": [[[0.229]], [[0.224]], [[0.225]]],
           "min": [[[0.0]], [[0.0]], [[0.0]]],
           "max": [[[1.0]], [[1.0]], [[1.0]]], "count": [n]}
    for key in L.camera_keys(info):
        if key not in stats:
            stats[key] = json.loads(json.dumps(img))
            changed = True
    if changed:
        json.dump(stats, open(sp, "w"))


# ── v2.1 → v3.0 training copy ─────────────────────────────────────────────────
def ensure_v30(src_root, name=None, force=False, progress=None):
    """Return a v3.0 copy of a v2.1 dataset (LeRobot 0.4.4 trains on v3.0 only).
    The v2.1 source is untouched; the v3.0 copy lives under DATASETS_DIR/.v30/."""
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

    name = name or os.path.basename(src_root.rstrip("/"))
    final = os.path.join(V30_DIR, name)
    info = L.read_info(src_root)
    if info.get("codebase_version") == "v3.0":
        return src_root
    # cache: skip if a fresh v3.0 copy already exists
    stamp = os.path.join(final, ".src_hash")
    src_hash = version_hash(src_root)
    if not force and os.path.isfile(stamp) and open(stamp).read().strip() == src_hash:
        return final

    if progress:
        progress(0, 1, stage=0)
    os.makedirs(_V30_STAGE, exist_ok=True)
    stage = os.path.join(_V30_STAGE, name)
    for extra in (stage, f"{stage}_old", f"{stage}_v30"):
        if os.path.exists(extra):
            shutil.rmtree(extra)
    shutil.copytree(src_root, stage)
    convert_dataset(repo_id=name, root=_V30_STAGE, push_to_hub=False, force_conversion=True)
    # stage is now v3.0; stage_old is the v2.1 copy — drop it
    if os.path.exists(f"{stage}_old"):
        shutil.rmtree(f"{stage}_old")
    _inject_image_stats(stage)   # add the camera stats LeRobot needs
    os.makedirs(V30_DIR, exist_ok=True)
    if os.path.exists(final):
        shutil.rmtree(final)
    shutil.move(stage, final)
    with open(os.path.join(final, ".src_hash"), "w") as f:
        f.write(src_hash)
    if progress:
        progress(1, 1, stage=0)
    return final


# ── train / eval / test split (by episode) ────────────────────────────────────
def split(n_episodes, ratios=(0.8, 0.1, 0.1), seed=42):
    """Return {train,eval,test: [episode indices]} for an early split parameter.
    With very few episodes we guarantee at least 1 train episode and place any
    remainder into eval then test only if room allows."""
    idx = list(range(n_episodes))
    rng = random.Random(seed)
    rng.shuffle(idx)
    tr, ev, te = ratios
    s = tr + ev + te
    tr, ev, te = tr / s, ev / s, te / s
    n_tr = max(1, int(round(n_episodes * tr))) if n_episodes else 0
    n_ev = int(round(n_episodes * ev))
    # keep at least train; clamp so buckets fit
    n_tr = min(n_tr, n_episodes)
    n_ev = min(n_ev, max(0, n_episodes - n_tr))
    n_te = max(0, n_episodes - n_tr - n_ev)
    train = sorted(idx[:n_tr])
    ev_ = sorted(idx[n_tr:n_tr + n_ev])
    test = sorted(idx[n_tr + n_ev:n_tr + n_ev + n_te])
    return {"train": train, "eval": ev_, "test": test,
            "counts": {"train": len(train), "eval": len(ev_), "test": len(test)},
            "ratios": {"train": tr, "eval": ev, "test": te}, "seed": seed}
