"""
EDL project persistence — JSON sidecars under WORKSPACE/projects/<name>.edl.json.
Lives OUTSIDE the dataset root so source episodes stay pristine. This is just
the autosave store for in-progress timeline edits.
"""
import json
import os
import re
import time

from . import config as CFG

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(name):
    name = _SAFE.sub("_", str(name)).strip("._") or "untitled"
    return name


def _path(name):
    return os.path.join(CFG.PROJECTS_DIR, f"{_safe_name(name)}.edl.json")


def list_projects():
    out = []
    for fn in sorted(os.listdir(CFG.PROJECTS_DIR)) if os.path.isdir(CFG.PROJECTS_DIR) else []:
        if not fn.endswith(".edl.json"):
            continue
        p = os.path.join(CFG.PROJECTS_DIR, fn)
        try:
            edl = json.load(open(p))
        except Exception:
            continue
        out.append({"name": edl.get("name", fn[:-9]),
                    "n_segments": len(edl.get("segments", [])),
                    "fps": edl.get("fps", 15.0),
                    "modified": os.path.getmtime(p)})
    return out


def get(name):
    p = _path(name)
    if not os.path.exists(p):
        raise FileNotFoundError(name)
    return json.load(open(p))


def save(name, edl):
    edl = dict(edl)
    edl.setdefault("edl_version", 1)
    edl["name"] = name
    edl.setdefault("fps", 15.0)
    edl["modified"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_path(name), "w") as fp:
        json.dump(edl, fp, indent=2)
    return edl


def delete(name):
    p = _path(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False
