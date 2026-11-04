"""
Generate MULTI-RATE MABEL episodes to test frequency visualization.

MABEL's payload is one 15 Hz index-aligned clock, so to represent genuinely
different per-stream rates we attach a per-channel effective-Hz profile + a
nominal rate (this is the sensor-reported cadence a real rig would log):

    freq/<channel>/hz   float32[T]   effective Hz at each frame
    freq  (group).attrs["meta_json"] = {channel: {nominal_fps, kind}}
    <file>.attrs["freq_profile_json"] = ground-truth drop spans

Channels = each camera + each actuator group (body/base/L-arm/R-arm/L-hand/
R-hand/neck) + action. Cameras run 20-40 Hz and randomly drop to ~5 Hz; proprio
runs 50-60 Hz and one or two MODULES (an arm / a hand) drop to 10-20 Hz over
spans (an unstable connection). `frequency_tracks` reads these back.

Run:
    cd learning
    python -m data_curation.server.tools.make_multirate --n 4     # -> dataset 'multirate_sim'
"""
import argparse
import json
import os
import shutil

import numpy as np

from .. import config as CFG, catalog

OUT = "multirate_sim"
_GENERATED = ("multirate_sim", "imperfect_sim", "stress_sim", "curated", "curated_test")


def _channels(f):
    sj = json.loads(f.attrs["schema_json"])
    cams = [c["name"] for c in sj.get("cameras", [])]
    groups = list(sj.get("actuator_groups", {}).keys())
    return cams, groups


def _profile(T, nominal, n_drops, rng, floor):
    """hz[T] = nominal (small jitter), dropping into `floor` Hz over n_drops spans."""
    hz = nominal * (1 + 0.04 * rng.standard_normal(T))
    spans = []
    for _ in range(n_drops):
        s = int(rng.integers(5, max(6, T - 25))); e = min(T, s + int(rng.integers(15, 45)))
        low = float(rng.uniform(*floor))
        hz[s:e] = low * (1 + 0.06 * rng.standard_normal(e - s))
        spans.append({"start": s, "end": e, "hz": round(low, 1)})
    return np.clip(hz, 1.0, None).astype(np.float32), spans


def make_one(base, out_path, seed):
    import h5py
    shutil.copy(base, out_path)
    rng = np.random.default_rng(seed)
    with h5py.File(out_path, "r+") as f:
        cams, groups = _channels(f)
        T = int(f.attrs.get("num_frames", f["obs/joint_pos"].shape[0]))
        meta, gt = {}, []
        grp = f.require_group("freq")
        # cameras: 20-40 Hz nominal, drop to ~5 Hz
        for cam in cams:
            nom = float(rng.uniform(20, 40))
            hz, spans = _profile(T, nom, int(rng.integers(1, 3)), rng, (4, 7))
            grp.create_dataset(f"{cam}/hz", data=hz)
            meta[cam] = {"nominal_fps": round(nom, 1), "kind": "camera"}
            gt += [{"channel": cam, "nominal": round(nom, 1), **sp} for sp in spans]
        # proprio modules: 50-60 Hz nominal; 1-2 arm/hand modules drop to 10-20 Hz
        armhand = [g for g in groups if "arm" in g or "hand" in g] or groups
        drop_groups = set(rng.choice(armhand, size=min(2, len(armhand)), replace=False).tolist())
        for g in groups:
            nom = float(rng.uniform(50, 60))
            hz, spans = _profile(T, nom, int(rng.integers(1, 3)) if g in drop_groups else 0, rng, (10, 20))
            grp.create_dataset(f"{g}/hz", data=hz)
            meta[g] = {"nominal_fps": round(nom, 1), "kind": "proprio"}
            gt += [{"channel": g, "nominal": round(nom, 1), **sp} for sp in spans]
        # action/control channel tracks proprio rate
        nom = float(rng.uniform(50, 60))
        hz, _ = _profile(T, nom, 0, rng, (10, 20))
        grp.create_dataset("action/hz", data=hz)
        meta["action"] = {"nominal_fps": round(nom, 1), "kind": "action"}

        f["freq"].attrs["meta_json"] = json.dumps(meta)
        f.attrs["freq_profile_json"] = json.dumps(gt)
        f.attrs["task"] = "multi-rate stress"
    return meta, gt


def run(n, base_dataset=None):
    bases = []
    for ds in catalog.full_catalog()["datasets"]:
        if ds["name"] in _GENERATED or (base_dataset and ds["name"] != base_dataset):
            continue
        for ep in ds["episodes"]:
            bases.append(catalog.episode_path(ds["name"], ep["file"]))
    if not bases:
        raise SystemExit("no clean base episodes found")
    out_dir = os.path.join(CFG.DATASET_ROOT, OUT)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        base = bases[i % len(bases)]
        _, gt = make_one(base, os.path.join(out_dir, f"episode_{i:04d}.h5"), seed=2000 + i)
        drops = "; ".join(f"{d['channel']}→{d['hz']}Hz@{d['start']}-{d['end']}" for d in gt)
        print(f"  episode_{i:04d}  {len(gt)} drop(s): {drops}")
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as fh:
        json.dump({"source": "sim-multirate", "note": "per-stream multi-rate frequency stress test"}, fh)
    print(f"wrote {n} episodes to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    run(a.n, a.base)
