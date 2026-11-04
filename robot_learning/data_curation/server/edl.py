"""
Edit Decision List (EDL) — the non-destructive representation of a timeline edit.

An EDL is just an ordered list of source segments [start, end) (half-open, in
source-frame indices). Every Final-Cut operation in the UI (trim, split, delete,
merge, append, reorder) is a pure transform on this list; nothing touches the
source .h5 until the user materializes (see materialize.py).

    {
      "edl_version": 1, "name": "demo_clean", "fps": 15.0,
      "segments": [
        {"dataset": "studio_sample", "episode": "episode_0000.h5", "start": 30, "end": 210},
        {"dataset": "studio_sample", "episode": "episode_0003.h5", "start": 0,  "end": 95}
      ]
    }
"""
import os

import h5py

from . import catalog


def resolve_source(seg):
    """seg -> absolute .h5 path."""
    return catalog.episode_path(seg["dataset"], seg.get("episode", seg.get("episode_index")))


def normalize(edl):
    """Coerce + clamp segments to valid half-open ranges against each source's
    num_frames. Drops empty/invalid segments. Returns (clean_edl, warnings)."""
    warnings = []
    out = []
    for k, seg in enumerate(edl.get("segments", [])):
        try:
            p = resolve_source(seg)
        except Exception as e:
            warnings.append(f"segment {k}: source not found ({e})")
            continue
        with h5py.File(p, "r") as f:
            n = int(f.attrs["num_frames"]) if "num_frames" in f.attrs else 0
        a = max(0, int(seg.get("start", 0)))
        b = min(n, int(seg.get("end", n)))
        if b <= a:
            warnings.append(f"segment {k}: empty range [{a},{b}) skipped")
            continue
        out.append({"dataset": seg["dataset"],
                    "episode": os.path.basename(p),
                    "start": a, "end": b,
                    "modkeep": dict(seg.get("modkeep", {})),     # per-modality clip-local keep ranges
                    "modoffset": dict(seg.get("modoffset", {})), # per-modality time offset (frames)
                    "rate": float(seg.get("rate", 1) or 1)})     # retime / resample factor
    clean = dict(edl)
    clean["segments"] = out
    return clean, warnings


def total_frames(edl):
    return sum(s["end"] - s["start"] for s in edl.get("segments", []))


def preview_map(edl):
    """Output-frame -> source-frame map for scrubbing an UNSAVED edit without
    materializing: [{out, dataset, episode, src}] (one row per timeline frame).
    Collapsed to per-segment spans for compactness."""
    spans = []
    out = 0
    for seg in edl.get("segments", []):
        n = seg["end"] - seg["start"]
        spans.append({"out_start": out, "out_end": out + n,
                      "dataset": seg["dataset"], "episode": seg["episode"],
                      "src_start": seg["start"]})
        out += n
    return {"total": out, "fps": edl.get("fps", 15.0), "spans": spans}


def validate(edl):
    """Dry-run compatibility report (no write): total frames, per-key presence
    and trailing-shape/dtype agreement across all distinct sources, warnings."""
    clean, warnings = normalize(edl)
    segs = clean["segments"]
    src_paths = {}
    for s in segs:
        src_paths.setdefault((s["dataset"], s["episode"]), resolve_source(s))

    # union of keys + per-key (dtype, trailing shape) per source
    keyinfo = {}     # key -> {"present_in": set, "dtype": str, "trail": tuple, "conflict": bool}
    for (ds, ep), p in src_paths.items():
        with h5py.File(p, "r") as f:
            def _v(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                trail = tuple(obj.shape[1:])
                dt = str(obj.dtype)
                info = keyinfo.setdefault(name, {"present_in": set(), "dtype": dt,
                                                 "trail": trail, "conflict": False})
                info["present_in"].add(f"{ds}/{ep}")
                if info["dtype"] != dt or info["trail"] != trail:
                    info["conflict"] = True
            f.visititems(_v)

    nsrc = len(src_paths)
    keys = []
    for name, info in sorted(keyinfo.items()):
        present_all = len(info["present_in"]) == nsrc
        if info["conflict"]:
            warnings.append(f"key '{name}': dtype/shape differs across sources — will be dropped")
        elif not present_all:
            warnings.append(f"key '{name}': missing from some sources — filled where absent")
        keys.append({"name": name, "present_in_all": present_all,
                     "dtype": info["dtype"], "trail": list(info["trail"]),
                     "conflict": info["conflict"]})

    T = total_frames(clean)
    fps = clean.get("fps", 15.0)
    return {
        "ok": T > 0,
        "total_frames": T,
        "duration_s": round(T / fps, 3) if fps else 0.0,
        "fps": fps,
        "n_segments": len(segs),
        "n_sources": nsrc,
        "keys": keys,
        "warnings": warnings,
    }
