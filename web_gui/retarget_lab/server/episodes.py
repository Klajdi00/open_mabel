"""
Human episodes: record the Vision Pro stream, list and load episodes.

On-disk format (one folder per episode, lossless — the RAW wire payloads, so
any future retargeter or metric can be replayed against exactly what the
headset sent):

    data/episodes/<id>/
        frames.jsonl.gz   one line per teleop_frame: {"rt": recv unix s, "p": payload}
        meta.json         {id, name, created, frames, duration_s, device, notes, ...}

The clock a replay drives the engine with is the payload's own `timestamp`
(Vision Pro: seconds since 2001-01-01, monotonic within a session), re-zeroed
at the first frame; the receive time is the fallback.

Also loadable, read-only: the EgoDex benchmark clips
(controller/experiments/data/egodex/benchmark/clips/*.json.gz — the "MABEL
Vision Pro wire payload" dialect of mabel.sources.avp). CC-by-NC-ND, local only.
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
from pathlib import Path

from . import config as CFG


def _slug(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip()).strip("_")
    return s[:48] or "episode"


class Recorder:
    """Appends raw frames to an open episode. Thread-safe; cheap per frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._fh = None
        self.meta = None
        self.dir = None

    @property
    def active(self):
        return self._fh is not None

    def start(self, name="", device="", notes=""):
        with self._lock:
            if self._fh is not None:
                return self.meta
            CFG.EPISODES_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            eid = f"{stamp}_{_slug(name)}" if name else stamp
            self.dir = CFG.EPISODES_DIR / eid
            self.dir.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(self.dir / "frames.jsonl.gz", "wt", compresslevel=3)
            self.meta = {"id": eid, "name": name or eid, "created": time.time(),
                         "device": device, "notes": notes, "frames": 0,
                         "duration_s": 0.0, "hands_tracked": {"left": 0, "right": 0},
                         "source": "vision_pro", "format": "teleop_frame wire payload "
                         "(server/wire_protocol.py), raw"}
            self._t0 = None
            return self.meta

    def add(self, payload, rt):
        with self._lock:
            if self._fh is None:
                return
            self._fh.write(json.dumps({"rt": rt, "p": payload}, separators=(",", ":")))
            self._fh.write("\n")
            m = self.meta
            m["frames"] += 1
            if self._t0 is None:
                self._t0 = rt
            m["duration_s"] = rt - self._t0
            for s, k in (("left", "leftHand"), ("right", "rightHand")):
                h = payload.get(k)
                if h and h.get("isTracked"):
                    m["hands_tracked"][s] += 1

    def stop(self):
        with self._lock:
            if self._fh is None:
                return None
            self._fh.close()
            self._fh = None
            (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
            meta, self.meta = self.meta, None
            return meta

    def status(self):
        with self._lock:
            if self.meta is None:
                return {"active": False}
            return {"active": True, "id": self.meta["id"], "frames": self.meta["frames"],
                    "duration_s": round(self.meta["duration_s"], 2)}


def list_episodes():
    out = []
    if CFG.EPISODES_DIR.exists():
        for d in sorted(CFG.EPISODES_DIR.iterdir(), reverse=True):
            mf = d / "meta.json"
            if mf.exists():
                try:
                    m = json.loads(mf.read_text())
                    m["kind"] = "recorded"
                    out.append(m)
                except Exception:
                    pass
    if CFG.EGODEX_CLIPS.exists():
        for f in sorted(CFG.EGODEX_CLIPS.glob("*.json.gz")):
            out.append({"id": "egodex:" + f.name[:-8], "name": f.name[:-8],
                        "kind": "egodex", "source": "EgoDex (CC-by-NC-ND, local only)"})
    return out


def load(eid):
    """-> (meta, frames) with frames = [{kind, p, t}] on a zero-based clock."""
    if eid.startswith("egodex:"):
        f = CFG.EGODEX_CLIPS / (eid[len("egodex:"):] + ".json.gz")
        d = json.load(gzip.open(f))
        frames = [{"kind": "avp", "p": p, "t": float(p.get("timestamp", i / d["fps"]))}
                  for i, p in enumerate(d["payloads"])]
        meta = {"id": eid, "name": d.get("task", eid), "kind": "egodex",
                "instruction": d.get("instruction"), "frames": len(frames),
                "duration_s": d.get("duration_s"), "fps": d.get("fps")}
    else:
        root = (CFG.EPISODES_DIR / eid).resolve()
        if CFG.EPISODES_DIR.resolve() not in root.parents:
            raise FileNotFoundError(eid)
        meta = json.loads((root / "meta.json").read_text())
        meta["kind"] = "recorded"
        frames = []
        with gzip.open(root / "frames.jsonl.gz", "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    break                       # a crash-truncated tail
                ts = r["p"].get("timestamp")
                frames.append({"kind": "wire", "p": r["p"],
                               "t": float(ts) if ts is not None else float(r["rt"])})
    if frames:
        t0 = frames[0]["t"]
        for fr in frames:
            fr["t"] -= t0
        # enforce a strictly increasing clock (duplicate stamps would read as dt=0)
        for i in range(1, len(frames)):
            if frames[i]["t"] <= frames[i - 1]["t"]:
                frames[i]["t"] = frames[i - 1]["t"] + 1e-3
    return meta, frames


def delete(eid):
    import shutil
    root = (CFG.EPISODES_DIR / eid).resolve()
    if CFG.EPISODES_DIR.resolve() not in root.parents or not root.exists():
        raise FileNotFoundError(eid)
    shutil.rmtree(root)


def rename(eid, name):
    mf = CFG.EPISODES_DIR / eid / "meta.json"
    m = json.loads(mf.read_text())
    m["name"] = name
    mf.write_text(json.dumps(m, indent=2))
    return m
