"""
Collected-data summary for the Data pane: episode demos grouped by task
(EpisodeRecorder layout: <root>/<task>/<episode_dir>/meta.json) plus rosbag
MCAP sessions (<recordings>/<stamp>_<label>/*.mcap). Read-only disk scan.
"""
import json
import os

from . import config as CFG


def _dir_size(path, cap_files=4000):
    total, n = 0, 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            n += 1
            if n > cap_files:                    # keep scans bounded
                return total, True
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total, False


def episodes(root=None):
    """{tasks: [{task, episodes, latest, bytes}], total}"""
    root = root or CFG.EPISODES_DIR
    tasks = []
    total = 0
    if os.path.isdir(root):
        for task in sorted(os.listdir(root)):
            tdir = os.path.join(root, task)
            if not os.path.isdir(tdir):
                continue
            eps = [e for e in os.listdir(tdir)
                   if os.path.isdir(os.path.join(tdir, e))
                   or e.endswith((".h5", ".hdf5"))]
            if not eps:
                continue
            latest = max((os.path.getmtime(os.path.join(tdir, e)) for e in eps),
                         default=0)
            size, _ = _dir_size(tdir)
            tasks.append({"task": task, "episodes": len(eps),
                          "latest": latest, "bytes": size})
            total += len(eps)
    return {"root": root, "tasks": tasks, "total": total}


def bags(root=None):
    """{sessions: [{name, label, mcaps, bytes, meta}], total}"""
    root = root or CFG.RECORDINGS_DIR
    sessions = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root), reverse=True):
            sdir = os.path.join(root, name)
            if not os.path.isdir(sdir):
                continue
            mcaps = [f for f in os.listdir(sdir) if f.endswith(".mcap")]
            meta = {}
            sj = os.path.join(sdir, "session.json")
            if os.path.isfile(sj):
                try:
                    with open(sj) as f:
                        meta = json.load(f)
                except Exception:
                    meta = {}
            size, _ = _dir_size(sdir)
            label = name.split("_", 2)[2] if name.count("_") >= 2 else name
            sessions.append({"name": name, "label": label,
                             "mcaps": len(mcaps), "bytes": size, "meta": meta})
    return {"root": root, "sessions": sessions, "total": len(sessions)}


def summary():
    return {"episodes": episodes(), "bags": bags()}
