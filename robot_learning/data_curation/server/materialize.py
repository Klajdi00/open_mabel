"""
Materialize an EDL into a NEW episode_XXXX.h5 by slicing + concatenating the
source HDF5 datasets frame-by-frame. Source files are never mutated.

Reuses recorder.py's write conventions exactly: vlen-uint8 for ``*/rgb`` JPEG
blobs, gzip for numeric arrays > 4 KB, the same attrs (schema_json, num_frames,
episode_index, created, task). time/* is rebased to be strictly monotonic across
segment joins. Numbering uses max(existing_index)+1 so deletions can't clobber.
"""
import glob
import json
import os
import re
import shutil
import time

import h5py
import numpy as np

from . import config as CFG, edl as EDL


def _next_index(out_dir):
    mx = -1
    for p in glob.glob(os.path.join(out_dir, "episode_*.h5")):
        m = re.search(r"episode_(\d+)\.h5$", os.path.basename(p))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def _union_keys(srcs):
    keys = set()
    for f in srcs.values():
        f.visititems(lambda n, o: keys.add(n) if isinstance(o, h5py.Dataset) else None)
    keys.discard("time/sim")
    keys.discard("time/wall")
    return keys


# modality id -> HDF5 keys (trailing '/' = prefix). Mirrors web/js/names.js.
def _modality_keys(mod_id, all_keys):
    if mod_id == "proprio": pats = ["obs/joint_pos", "obs/joint_vel", "obs/joint_torque"]
    elif mod_id == "pos": pats = ["obs/joint_pos"]
    elif mod_id == "vel": pats = ["obs/joint_vel"]
    elif mod_id == "torque": pats = ["obs/joint_torque"]
    elif mod_id == "action": pats = ["action/ctrl"]
    elif mod_id == "lidar": pats = ["obs/lidar"]
    elif mod_id == "pose": pats = ["obs/pose/"]
    elif mod_id == "teleop": pats = ["teleop/"]
    elif mod_id == "state": pats = ["obs/qpos", "obs/qvel"]
    elif mod_id.startswith("cam:"):
        n = mod_id[4:]; pats = [f"obs/cam/{n}/rgb", f"obs/cam/{n}/depth"]
    else: pats = []
    out = set()
    for p in pats:
        if p.endswith("/"):
            out.update(k for k in all_keys if k.startswith(p))
        elif p in all_keys:
            out.add(p)
    return out


def _fill(out_ds, off, count, ref):
    """Null-fill `count` frames of out_ds at `off`: empty JPEG blob for vlen rgb,
    NaN (float) / 0 (int) for numeric arrays."""
    if out_ds.dtype.kind == "O":                      # vlen rgb
        for j in range(count):
            out_ds[off + j] = np.zeros(0, dtype=np.uint8)
    else:
        fill = np.nan if np.issubdtype(ref.dtype, np.floating) else 0
        out_ds[off:off + count] = np.full((count,) + tuple(ref.shape[1:]), fill, dtype=ref.dtype)


def _slerp(q0, q1, t):
    """SLERP between wxyz quaternions at t∈[0,1]; robust to sign + near-parallel."""
    q0 = np.asarray(q0, dtype=np.float64); q1 = np.asarray(q1, dtype=np.float64)
    n0 = np.linalg.norm(q0); n1 = np.linalg.norm(q1)
    if n0 < 1e-9 or n1 < 1e-9:
        return q1 if t >= 0.5 else q0
    q0 = q0 / n0; q1 = q1 / n1
    d = float(np.dot(q0, q1))
    if d < 0.0:                       # take the shortest arc
        q1 = -q1; d = -d
    if d > 0.9995:                    # near-parallel → nlerp (SLERP is ill-conditioned)
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) or 1.0)
    th0 = np.arccos(np.clip(d, -1.0, 1.0)); s = np.sin(th0)
    return (np.sin((1.0 - t) * th0) / s) * q0 + (np.sin(t * th0) / s) * q1


def _resample(src_ds, key, i0, i1, frac, near, ns):
    """§2.4 modality-aware resample of ONE output frame from source rows i0..i1:
    SLERP for pose quaternions, linear for other float scalars, nearest for
    JPEG/int/mode/grasp (interpolating a category or a compressed blob is wrong)."""
    if src_ds.dtype.kind == "O":                         # vlen JPEG → nearest
        return src_ds[min(max(near, 0), ns - 1)]
    v0 = src_ds[i0]
    if i0 == i1 or frac <= 0.0:
        return v0
    if not np.issubdtype(src_ds.dtype, np.floating):     # int / mode / grasp → nearest
        return src_ds[min(max(near, 0), ns - 1)]
    v1 = src_ds[i1]
    if key.startswith("obs/pose/") and v0.shape[-1] >= 7:  # [x,y,z, qw,qx,qy,qz]
        out = (1.0 - frac) * v0 + frac * v1               # linear translation
        out[3:7] = _slerp(v0[3:7], v1[3:7], frac)         # SLERP the wxyz quaternion
        return out.astype(src_ds.dtype, copy=False)
    return ((1.0 - frac) * v0 + frac * v1).astype(src_ds.dtype, copy=False)


def materialize(edl, out_dataset, task=None, progress=None, out_file=None):
    """Write the EDL to DATASET_ROOT/<out_dataset>/episode_XXXX.h5. Returns
    {file, num_frames, report}. If `out_file` is given, write to exactly that
    filename (OVERWRITING) instead of auto-numbering — this makes per-episode
    batch export idempotent (a source episode always maps to the same output, so
    re-exporting after more edits updates in place instead of duplicating)."""
    clean, warnings = EDL.normalize(edl)
    segs = clean["segments"]
    if not segs:
        raise ValueError("EDL has no valid segments")
    fps = float(clean.get("fps", 15.0)) or 15.0
    T = EDL.total_frames(clean)

    out_dir = os.path.join(CFG.DATASET_ROOT, out_dataset)
    os.makedirs(out_dir, exist_ok=True)

    # open every distinct source read-only
    src_paths = {}
    for s in segs:
        src_paths.setdefault(s["episode"] and (s["dataset"], s["episode"]),
                             EDL.resolve_source(s))
    srcs = {k: h5py.File(p, "r") for k, p in src_paths.items()}

    try:
        keys = _union_keys(srcs)
        # reference dataset (dtype + trailing shape) per key; skip conflicting keys
        ref = {}
        dropped = []
        for k in sorted(keys):
            cand = [f[k] for f in srcs.values() if k in f]
            dt = str(cand[0].dtype)
            trail = tuple(cand[0].shape[1:])
            if any(str(c.dtype) != dt or tuple(c.shape[1:]) != trail for c in cand):
                dropped.append(k)
                continue
            ref[k] = cand[0]
        for k in dropped:
            keys.discard(k)

        # per-segment, per-key DROPPED clip-local frame ranges (from modkeep).
        # modkeep[mod] = kept spans (clip-local); dropped = complement within [0,n).
        def _complement(keep, n):
            ks = sorted([[max(0, min(a, n)), max(0, min(b, n))] for a, b in keep if b > a])
            res, cur = [], 0
            for a, b in ks:
                if a > cur:
                    res.append([cur, a])
                cur = max(cur, b)
            if cur < n:
                res.append([cur, n])
            return res

        seg_drop = []                          # list of {key: [[lo,hi],…]}
        seg_off = []                           # list of {key: int frame offset}
        for s in segs:
            src = srcs[(s["dataset"], s["episode"])]
            src_keys = set()
            src.visititems(lambda n, o: src_keys.add(n) if isinstance(o, h5py.Dataset) else None)
            n = s["end"] - s["start"]
            dk = {}
            for mid, keep in s.get("modkeep", {}).items():
                drops = _complement(keep, n)
                if not drops:
                    continue
                for k in _modality_keys(mid, src_keys):
                    dk.setdefault(k, []).extend(drops)
            seg_drop.append(dk)
            ko = {}
            for mid, off in s.get("modoffset", {}).items():
                if not off:
                    continue
                for k in _modality_keys(mid, src_keys):
                    ko[k] = int(off)
            seg_off.append(ko)

        def _dropped_len(si, k):
            n = segs[si]["end"] - segs[si]["start"]
            rs = sorted(seg_drop[si].get(k, []))
            tot, cur = 0, -1
            for a, b in rs:                    # merged length
                a = max(a, cur); tot += max(0, b - a); cur = max(cur, b)
            return min(tot, n)

        def _has_data(si, k):
            src = srcs[(segs[si]["dataset"], segs[si]["episode"])]
            n = segs[si]["end"] - segs[si]["start"]
            return (k in src) and (_dropped_len(si, k) < n)

        # "drop key if fully unused": omit a key only if NO segment contributes it
        omitted = {k for k in keys if not any(_has_data(si, k) for si in range(len(segs)))}
        keys -= omitted

        # ── reject ranges: ripple-REMOVE timeline frames (global, post-concat) ──
        keep = np.ones(T, dtype=bool)
        for r in clean.get("ranges", []):
            if r.get("kind") == "reject":
                a = max(0, int(r.get("start", 0))); b = min(T, int(r.get("end", 0)))
                if b > a:
                    keep[a:b] = False
        T2 = int(keep.sum())
        if T2 == 0:
            raise ValueError("every frame is inside a reject range")
        cum = np.concatenate([[0], np.cumsum(keep)])         # cum[g] = #kept in [0,g)
        g0s, _acc = [], 0                                    # per-segment global start (pre-reject)
        for s in segs:
            g0s.append(_acc); _acc += s["end"] - s["start"]

        def _runs(mask):                                     # contiguous True runs [lo,hi)
            runs, i, m = [], 0, len(mask)
            while i < m:
                if mask[i]:
                    j = i
                    while j < m and mask[j]:
                        j += 1
                    runs.append((i, j)); i = j
                else:
                    i += 1
            return runs

        # per-segment kept runs (after reject) + retimed output length per run
        def _ol(rl, rate):
            return rl if rate == 1.0 else max(1, int(round(rl / rate)))
        seg_runs, T_out = [], 0
        for si, s in enumerate(segs):
            n = s["end"] - s["start"]; rate = float(s.get("rate", 1) or 1)
            runs = []
            for (j0, j1) in _runs(keep[g0s[si]:g0s[si] + n]):
                ol = _ol(j1 - j0, rate); runs.append((j0, j1, ol)); T_out += ol
            seg_runs.append(runs)

        if out_file:
            fn = out_file if out_file.endswith(".h5") else out_file + ".h5"
            final_path = os.path.join(out_dir, os.path.basename(fn))
            m = re.search(r"(\d+)", os.path.basename(fn)); number = int(m.group(1)) if m else 0
        else:
            number = _next_index(out_dir)
            final_path = os.path.join(out_dir, f"episode_{number:04d}.h5")
        tmp = final_path + ".tmp"

        with h5py.File(tmp, "w") as out:
            # pre-create datasets at final (post-reject, post-retime) length
            for k in sorted(keys):
                if k.endswith("/rgb"):
                    out.create_dataset(k, shape=(T_out,), dtype=h5py.vlen_dtype(np.uint8))
                else:
                    trail = tuple(ref[k].shape[1:])
                    sample = np.zeros((1,) + trail, dtype=ref[k].dtype)
                    comp = "gzip" if sample.nbytes > 4096 else None
                    out.create_dataset(k, shape=(T_out,) + trail, dtype=ref[k].dtype,
                                       compression=comp)

            t_sim = np.empty(T_out, dtype=np.float64)
            interp_mask = np.zeros(T_out, dtype=bool)      # §2.4: True where a frame was resampled
            oi = 0
            t_acc = 0.0
            dt_default = 1.0 / fps
            for si, s in enumerate(segs):
                src = srcs[(s["dataset"], s["episode"])]
                a = s["start"]; rate = float(s.get("rate", 1) or 1)
                for (j0, j1, ol) in seg_runs[si]:
                    rl = j1 - j0
                    if rate == 1.0:
                        # ── source-rate path: reject + per-modality offset + modkeep ──
                        for k in sorted(keys):
                            if k not in src:               # whole segment missing -> fill
                                _fill(out[k], oi, rl, ref[k]); continue
                            off = seg_off[si].get(k, 0)    # per-modality time offset (edge-held)
                            if k.endswith("/rgb"):         # vlen JPEG: element-wise (robust)
                                ns = src[k].shape[0]
                                for t in range(rl):
                                    out[k][oi + t] = src[k][min(max(a + j0 + t - off, 0), ns - 1)]
                            elif off == 0:
                                out[k][oi:oi + rl] = src[k][a + j0:a + j1]   # copy the kept run...
                            else:
                                ns = src[k].shape[0]
                                for t in range(rl):
                                    out[k][oi + t] = src[k][min(max(a + j0 + t - off, 0), ns - 1)]
                            for lo, hi in seg_drop[si].get(k, []):           # ...then modkeep null-fills
                                clo = max(lo, j0); chi = min(hi, j1)
                                if chi > clo:
                                    _fill(out[k], oi + (clo - j0), chi - clo, ref[k])
                        if "time/sim" in src:
                            seg_t = np.asarray(src["time/sim"][a + j0:a + j1], dtype=np.float64)
                            d = np.diff(seg_t, prepend=seg_t[0]); d[d < 0] = dt_default
                        else:
                            d = np.full(rl, dt_default); d[0] = 0.0
                        for t in range(rl):
                            t_acc += d[t]; t_sim[oi + t] = t_acc
                    else:
                        # ── retimed path: modality-aware resample to `ol` frames (§2.4) ──
                        # linear for float scalars, SLERP for pose quaternions, nearest
                        # for JPEG/int/mode/grasp. Every retimed frame is marked in
                        # interpolated_mask (surfaced, not hidden).
                        drops_for = seg_drop[si]
                        for u in range(ol):
                            ff = j0 + (u * (rl - 1) / (ol - 1) if ol > 1 else 0.0)
                            f0 = int(np.floor(ff)); frac = ff - f0
                            fnear = int(round(ff))
                            for k in sorted(keys):
                                if k not in src or any(lo <= fnear < hi for lo, hi in drops_for.get(k, [])):
                                    _fill(out[k], oi + u, 1, ref[k]); continue
                                off = seg_off[si].get(k, 0); ns = src[k].shape[0]
                                i0 = min(max(a + f0 - off, 0), ns - 1)
                                i1 = min(max(a + f0 + 1 - off, 0), ns - 1)
                                out[k][oi + u] = _resample(src[k], k, i0, i1, frac,
                                                           a + fnear - off, ns)
                            interp_mask[oi + u] = True
                            t_acc += dt_default; t_sim[oi + u] = t_acc
                    oi += ol
                    if progress:
                        progress(oi, T_out, si)

            out.create_dataset("time/sim", data=t_sim)
            out.create_dataset("time/wall", data=t_sim + time.time())
            if interp_mask.any():                                  # §2.4 provenance
                out.create_dataset("edit/interpolated_mask", data=interp_mask)

            # schema/attrs from the first source (warn if sources differ)
            first = srcs[(segs[0]["dataset"], segs[0]["episode"])]
            schema_json = first.attrs.get("schema_json", "{}")
            out.attrs["schema_json"] = schema_json
            out.attrs["num_frames"] = T_out
            out.attrs["episode_index"] = number
            out.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            out.attrs["task"] = task or clean.get("name", "edited")
            out.attrs["edl_json"] = json.dumps(clean)        # provenance
            # language annotations (frame-ranged) — remapped through the reject
            # compaction into OUTPUT frame indices; the VLA / skill-seg payload.
            anns = []
            for an in clean.get("annotations", []):
                s0 = max(0, min(int(an.get("start", 0)), T)); e0 = max(0, min(int(an.get("end", 0)), T))
                ns, ne = int(cum[s0]), int(cum[e0])
                if ne > ns:
                    anns.append({"text": an.get("text", ""), "start": ns, "end": ne, "kind": an.get("kind", "language")})
            out.attrs["annotations_json"] = json.dumps(anns)
            # per-frame language instruction (VLA): the language annotation(s)
            # covering each output frame, as a vlen-string dataset.
            lang = [""] * T_out
            for an in anns:
                if an["kind"] != "language":
                    continue
                for f in range(max(0, an["start"]), min(T_out, an["end"])):
                    lang[f] = (lang[f] + " · " + an["text"]) if lang[f] else an["text"]
            out.create_dataset("lang/instruction", data=np.array(lang, dtype=h5py.string_dtype()))

        os.replace(tmp, final_path)
        # carry a dataset_meta.json so the new folder is self-describing
        meta_dst = os.path.join(out_dir, "dataset_meta.json")
        if not os.path.exists(meta_dst):
            src_meta = os.path.join(os.path.dirname(EDL.resolve_source(segs[0])),
                                    "dataset_meta.json")
            if os.path.exists(src_meta):
                shutil.copy2(src_meta, meta_dst)

        return {"file": os.path.basename(final_path), "dataset": out_dataset,
                "num_frames": T_out, "rejected_frames": T - T2, "dropped_keys": dropped,
                "omitted_modalities": sorted(omitted), "warnings": warnings}
    finally:
        for f in srcs.values():
            try:
                f.close()
            except Exception:
                pass
