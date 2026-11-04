"""
episodes.py — the save-confirm flow for freshly recorded sim episodes.

The bridge's EpisodeRecorder writes `<EPISODES_DIR>/<task>/episode_<idx>_<ts>/`
(meta.json + timestamps.npy + qpos/qvel/ctrl/lidar + cam/ + depth/). After the
operator hits ■ Stop the studio pops "save this episode?": this module finds
the newest episode, summarises it (length, modalities, frequencies, size), and
either COMMITS it (rename to the env-stamped `<env>_<YYYYMMDD>_<seq>` default,
optionally into another task folder / custom name) or DISCARDS it.

Everything is confined to EPISODES_DIR — paths that resolve outside it are
rejected.
"""
import json
import os
import re
import shutil
import time
from datetime import datetime

from . import config as CFG

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _root():
    return os.path.realpath(CFG.EPISODES_DIR)


def _inside_root(path):
    rp = os.path.realpath(path)
    root = _root()
    return rp == root or rp.startswith(root + os.sep)


def _is_episode_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json"))


def sanitize(name, fallback="episode"):
    s = _SAFE.sub("_", (name or "").strip()).strip("._")
    return s or fallback


def latest_episode():
    """Newest episode dir (by meta.json mtime) anywhere under EPISODES_DIR."""
    root = _root()
    best, best_t = None, -1.0
    if not os.path.isdir(root):
        return None
    for task in os.listdir(root):
        tdir = os.path.join(root, task)
        if not os.path.isdir(tdir) or task.startswith("."):
            continue
        for ep in os.listdir(tdir):
            d = os.path.join(tdir, ep)
            mj = os.path.join(d, "meta.json")
            if not os.path.isfile(mj):
                continue
            t = os.path.getmtime(mj)
            if t > best_t:
                best, best_t = d, t
    return best


def _dir_size(path, cap=20000):
    total = n = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
            n += 1
            if n > cap:
                return total, n
    return total, n


def info(path):
    """Everything the save dialog shows about one episode."""
    if not (_inside_root(path) and _is_episode_dir(path)):
        return None
    meta = {}
    try:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
    except Exception:
        pass
    frames = int(meta.get("frames") or 0)
    dur = None
    ts_path = os.path.join(path, "timestamps.npy")
    if os.path.isfile(ts_path):
        try:
            import numpy as np
            ts = np.load(ts_path)
            frames = int(len(ts)) or frames
            if len(ts) >= 2:
                dur = float(ts[-1] - ts[0])
        except Exception:
            pass
    if dur is None and frames and meta.get("fps"):
        dur = frames / float(meta["fps"])
    hz = round(frames / dur, 2) if (dur and frames) else meta.get("fps")

    modalities = []
    cams = sorted(os.listdir(os.path.join(path, "cam"))) \
        if os.path.isdir(os.path.join(path, "cam")) else []
    for c in cams:
        modalities.append({"name": f"rgb/{c}", "kind": "rgb", "hz": hz,
                           "frames": frames})
    deps = sorted(os.listdir(os.path.join(path, "depth"))) \
        if os.path.isdir(os.path.join(path, "depth")) else []
    for c in deps:
        modalities.append({"name": f"depth/{c}", "kind": "depth", "hz": hz,
                           "frames": frames})
    for arr, kind in (("qpos", "joint state"), ("qpos_des", "joint targets"),
                      ("qvel", "joint velocity"), ("ctrl", "actuators"),
                      ("lidar", "lidar")):
        if os.path.isfile(os.path.join(path, f"{arr}.npy")):
            modalities.append({"name": arr, "kind": kind, "hz": hz,
                               "frames": frames})
    for csvf, kind in (("base.csv", "base odometry"),
                       ("control.csv", "teleop command")):
        if os.path.isfile(os.path.join(path, csvf)):
            modalities.append({"name": csvf.replace(".csv", ""), "kind": kind,
                               "hz": hz, "frames": frames})
    size, files = _dir_size(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "task": os.path.basename(os.path.dirname(path)),
        "frames": frames,
        "durationS": round(dur, 2) if dur else None,
        "hz": hz,
        "sizeBytes": size,
        "files": files,
        "scene": meta.get("scene"),
        "startedAt": meta.get("start_wall"),
        "jointCount": len(meta.get("joint_names") or []),
        "modalities": modalities,
    }


def suggest_name(env, task=None):
    """Default committed name: <env>_<YYYYMMDD>_<seq> (seq per env+day)."""
    env = sanitize(env, "sim")
    stamp = datetime.now().strftime("%Y%m%d")
    prefix = f"{env}_{stamp}_"
    seq = 0
    root = _root()
    if os.path.isdir(root):
        for tdir in os.listdir(root):
            full = os.path.join(root, tdir)
            if not os.path.isdir(full):
                continue
            for ep in os.listdir(full):
                m = re.match(re.escape(prefix) + r"(\d+)$", ep)
                if m:
                    seq = max(seq, int(m.group(1)))
    return f"{prefix}{seq + 1:03d}"


def commit(path, name=None, task=None, env=None):
    """Keep the episode: rename (default = env-stamped) and/or move task dirs."""
    if not (_inside_root(path) and _is_episode_dir(path)):
        return {"ok": False, "error": "not an episode under the episodes dir"}
    cur_task = os.path.basename(os.path.dirname(path))
    new_task = sanitize(task or cur_task, cur_task)
    new_name = sanitize(name or "", "") or suggest_name(env or new_task)
    dest_dir = os.path.join(_root(), new_task)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, new_name)
    n = 1
    while os.path.exists(dest) and os.path.realpath(dest) != os.path.realpath(path):
        dest = os.path.join(dest_dir, f"{new_name}_{n}")
        n += 1
    if not _inside_root(dest):
        return {"ok": False, "error": "destination escapes the episodes dir"}
    if os.path.realpath(dest) != os.path.realpath(path):
        shutil.move(path, dest)
    # stamp the final identity into meta.json
    try:
        mj = os.path.join(dest, "meta.json")
        with open(mj) as f:
            meta = json.load(f)
        meta.update(episode=os.path.basename(dest), task=new_task,
                    committed_wall=time.time())
        with open(mj, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return {"ok": True, "path": dest, "name": os.path.basename(dest),
            "task": new_task}


def discard(path):
    """Delete a just-recorded episode (must be a real episode dir)."""
    if not (_inside_root(path) and _is_episode_dir(path)):
        return {"ok": False, "error": "not an episode under the episodes dir"}
    shutil.rmtree(path, ignore_errors=True)
    # drop the task folder too if it just became empty (keeps the library tidy)
    parent = os.path.dirname(path)
    try:
        if _inside_root(parent) and parent != _root() and not os.listdir(parent):
            os.rmdir(parent)
    except OSError:
        pass
    return {"ok": True}
