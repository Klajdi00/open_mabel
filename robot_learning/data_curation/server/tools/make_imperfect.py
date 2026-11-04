"""
Generate imperfect MABEL episodes that mirror real-world failures, to stress-test
the curation tool. Each output episode is a COPY of a clean base episode with a
random mix of injected defects; the ground-truth spans are written to the
`injected_defects_json` attr so detection can be scored (see exp_robustness.py).

Injected failure modes (all realistic on a real robot):
  - motor_dropout   : one/several motors stop sending data -> value frozen, vel 0
  - arm_dead        : a power wire drops -> a whole arm frozen + torque 0, then recovers
  - torque_dead     : a motor stops reporting torque (torque~0) while it still moves
  - camera_freeze   : a camera hangs -> the same frame repeats (rgb + depth)
  - camera_blackout : a camera drops -> black/empty frames
  - low_rate        : a stream runs slower -> each sample held k frames (lower Hz)
  - rate_change     : the hold cadence changes partway through the episode
  - lag             : a stream is delayed k frames
  - timing_gap      : a capture stall -> a big jump in time/wall

Run:
    cd learning
    python -m data_curation.server.tools.make_imperfect --n 6        # -> dataset 'imperfect_sim'
"""
import argparse
import json
import os
import shutil

import numpy as np

from .. import config as CFG, catalog

try:
    import cv2
except Exception:
    cv2 = None

OUT_DATASET = "imperfect_sim"


def _groups(f):
    sj = json.loads(f.attrs["schema_json"])
    names = sj["actuator_names"]
    idx = {g: [names.index(n) for n in members if n in names]
           for g, members in sj.get("actuator_groups", {}).items()}
    cams = [c["name"] for c in sj.get("cameras", [])]
    depth_cams = [c["name"] for c in sj.get("cameras", []) if c.get("depth")]
    return sj, names, idx, cams, depth_cams


def _black_jpeg(shape):
    if cv2 is None:
        return None
    h, w = shape[:2]
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    return np.frombuffer(buf.tobytes(), np.uint8) if ok else None


# ── defect injectors (operate in-place on an open r+ HDF5; return a GT record) ──
def d_motor_dropout(f, rng, gidx, T):
    joints = sorted(rng.choice(range(59), size=int(rng.integers(1, 4)), replace=False).tolist())
    s = int(rng.integers(5, T - 20)); e = min(T, s + int(rng.integers(10, 30)))
    pos, vel = f["obs/joint_pos"], f["obs/joint_vel"]
    for j in joints:
        pos[s:e, j] = pos[s - 1, j]            # frozen at last received
        vel[s:e, j] = 0.0
    return {"type": "motor_dropout", "start": s, "end": e, "joints": joints,
            "label": f"motors {joints} dropout"}


def d_arm_dead(f, rng, gidx, T):
    arm = "left_arm" if rng.random() < 0.5 else "right_arm"
    js = gidx[arm]
    s = int(rng.integers(5, T - 25)); e = min(T, s + int(rng.integers(15, 40)))
    pos, vel, tq = f["obs/joint_pos"], f["obs/joint_vel"], f["obs/joint_torque"]
    for j in js:
        pos[s:e, j] = pos[s - 1, j]; vel[s:e, j] = 0.0; tq[s:e, j] = 0.0
    return {"type": "arm_dead", "start": s, "end": e, "joints": js, "group": arm,
            "label": f"{arm} power loss (dead {s}-{e})"}


def d_torque_dead(f, rng, gidx, T):
    g = ["left_arm", "right_arm", "left_hand", "right_hand"][int(rng.integers(0, 4))]
    js = gidx[g]
    s = int(rng.integers(5, T - 20)); e = min(T, s + int(rng.integers(12, 35)))
    tq = f["obs/joint_torque"]
    for j in js:
        tq[s:e, j] = 0.0                        # torque telemetry dead; pos/vel still move
    return {"type": "torque_dead", "start": s, "end": e, "joints": js, "group": g,
            "label": f"{g} torque dead {s}-{e}"}


def d_camera_freeze(f, rng, cams, depth_cams, T):
    cam = cams[int(rng.integers(0, len(cams)))]
    s = int(rng.integers(5, T - 20)); e = min(T, s + int(rng.integers(10, 30)))
    rgb = f.get(f"obs/cam/{cam}/rgb")
    if rgb is not None:
        held = rgb[s - 1]
        for i in range(s, e):
            rgb[i] = held
    dk = f"obs/cam/{cam}/depth"
    if dk in f:
        f[dk][s:e] = f[dk][s - 1]
    return {"type": "camera_freeze", "start": s, "end": e, "cam": cam,
            "label": f"{cam} frozen {s}-{e}"}


def d_camera_blackout(f, rng, cams, depth_cams, T):
    cam = cams[int(rng.integers(0, len(cams)))]
    s = int(rng.integers(5, T - 20)); e = min(T, s + int(rng.integers(8, 25)))
    rgb = f.get(f"obs/cam/{cam}/rgb")
    if rgb is not None:
        # infer size from a decoded frame
        shp = (270, 480)
        if cv2 is not None:
            im = cv2.imdecode(np.frombuffer(np.asarray(rgb[0], np.uint8), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                shp = im.shape
        blk = _black_jpeg(shp)
        if blk is not None:
            for i in range(s, e):
                rgb[i] = blk
    dk = f"obs/cam/{cam}/depth"
    if dk in f:
        f[dk][s:e] = 0
    return {"type": "camera_blackout", "start": s, "end": e, "cam": cam,
            "label": f"{cam} blackout {s}-{e}"}


def d_low_rate(f, rng, T):
    key = "obs/joint_vel" if rng.random() < 0.5 else "obs/base_twist"
    k = int(rng.integers(2, 5))                 # effective rate = fps / k
    a = np.asarray(f[key][:])
    for i in range(T):
        a[i] = a[i - (i % k)]                    # hold every k-th sample
    f[key][:] = a
    return {"type": "low_rate", "start": 0, "end": T, "key": key, "k": k,
            "label": f"{key} low-rate x1/{k}"}


def d_rate_change(f, rng, T):
    key = "obs/joint_pos"
    mid = T // 2
    k1, k2 = 1, int(rng.integers(3, 6))         # normal, then degraded cadence
    a = np.asarray(f[key][:])
    for i in range(mid, T):
        base = mid + ((i - mid) // k2) * k2
        a[i] = a[min(base, T - 1)]
    f[key][:] = a
    return {"type": "rate_change", "start": mid, "end": T, "key": key, "k": k2,
            "label": f"{key} rate drop x1/{k2} @ {mid}"}


def d_lag(f, rng, T):
    key = ["obs/joint_vel", "action/ctrl", "obs/base_twist"][int(rng.integers(0, 3))]
    k = int(rng.integers(3, 7))
    a = np.asarray(f[key][:])
    idx = np.clip(np.arange(T) - k, 0, T - 1)
    f[key][:] = a[idx]
    return {"type": "lag", "start": 0, "end": T, "key": key, "k": k,
            "label": f"{key} lag +{k}f"}


def d_nan(f, rng, T):
    """NaN corruption in a joint stream (bad sensor read / serialization glitch)."""
    key = "obs/joint_pos" if rng.random() < 0.5 else "obs/joint_vel"
    joints = sorted(rng.choice(range(59), size=int(rng.integers(1, 4)), replace=False).tolist())
    s = int(rng.integers(5, T - 15)); e = min(T, s + int(rng.integers(6, 20)))
    a = np.asarray(f[key][:])
    a[s:e][:, joints] = np.nan
    f[key][:] = a
    return {"type": "nan_corrupt", "start": s, "end": e, "joints": joints, "key": key,
            "label": f"{key} NaN {joints} {s}-{e}"}


def d_timing_gap(f, rng, T):
    if "time/wall" not in f:
        return None
    tw = np.asarray(f["time/wall"][:])
    n = int(rng.integers(1, 3)); gts = []
    for _ in range(n):
        fr = int(rng.integers(10, T - 5))
        extra = float(rng.uniform(0.5, 2.0))    # a stall of 0.5-2 s
        tw[fr:] += extra
        gts.append(fr)
    f["time/wall"][:] = tw
    return {"type": "timing_gap", "start": min(gts), "end": T, "frames": gts,
            "label": f"timing stall @ {gts}"}


_MOTOR = [d_motor_dropout, d_arm_dead, d_torque_dead]
_RATE = [d_low_rate, d_rate_change, d_lag, d_timing_gap]


def make_one(base_path, out_path, seed, severity):
    shutil.copy(base_path, out_path)
    rng = np.random.default_rng(seed)
    import h5py
    with h5py.File(out_path, "r+") as f:
        sj, names, gidx, cams, depth_cams = _groups(f)
        T = int(f.attrs.get("num_frames", f["obs/joint_pos"].shape[0]))
        defects = []
        if severity == "clean":
            pass
        elif severity == "severe":
            # stack many diverse failures: both arms, several motors, multiple
            # cameras out, torque loss, NaNs, big lag, rate change, timing stalls
            defects.append(d_arm_dead(f, rng, gidx, T))
            defects.append(d_arm_dead(f, rng, gidx, T))
            for _ in range(4):
                defects.append(d_motor_dropout(f, rng, gidx, T))
            defects.append(d_torque_dead(f, rng, gidx, T))
            defects.append(d_camera_blackout(f, rng, cams, depth_cams, T))
            defects.append(d_camera_freeze(f, rng, cams, depth_cams, T))
            defects.append(d_camera_blackout(f, rng, cams, depth_cams, T))
            defects.append(d_nan(f, rng, T))
            defects.append(d_lag(f, rng, T))
            defects.append(d_rate_change(f, rng, T))
            defects.append(d_timing_gap(f, rng, T))
            defects.append(d_timing_gap(f, rng, T))
        elif severity == "catastrophic":
            defects.append(d_arm_dead(f, rng, gidx, T))
            defects.append(d_camera_blackout(f, rng, cams, depth_cams, T))
            defects.append(d_motor_dropout(f, rng, gidx, T))
            defects.append(d_nan(f, rng, T))
            defects.append(d_timing_gap(f, rng, T))
        else:
            n_motor = int(rng.integers(1, 3)); n_cam = int(rng.integers(0, 2)); n_rate = int(rng.integers(1, 3))
            for fn in rng.choice(_MOTOR, size=n_motor, replace=False):
                defects.append(fn(f, rng, gidx, T))
            for _ in range(n_cam):
                fn = d_camera_freeze if rng.random() < 0.6 else d_camera_blackout
                defects.append(fn(f, rng, cams, depth_cams, T))
            for fn in rng.choice(_RATE, size=n_rate, replace=False):
                defects.append(fn(f, rng, T))
        defects = [d for d in defects if d]
        f.attrs["injected_defects_json"] = json.dumps(defects)
        f.attrs["task"] = f"imperfect ({severity})"
    return defects


def run(n, base_dataset=None, out=OUT_DATASET, severe=False):
    cat = catalog.full_catalog()["datasets"]
    # pick clean base episodes (prefer a given dataset, else the first non-generated one)
    bases = []
    for ds in cat:
        if ds["name"] in (OUT_DATASET, out) or ds["name"].startswith("stress"):
            continue
        if base_dataset and ds["name"] != base_dataset:
            continue
        for ep in ds["episodes"]:
            bases.append(catalog.episode_path(ds["name"], ep["file"]))
    if not bases:
        raise SystemExit("no clean base episodes found")

    out_dir = os.path.join(CFG.DATASET_ROOT, out)
    os.makedirs(out_dir, exist_ok=True)
    if severe:
        plan = ["severe"] * n                                  # worst-case stress set
    else:
        plan = (["clean", "catastrophic"] + ["mixed"] * max(0, n - 2))[:n]
    for i, sev in enumerate(plan):
        base = bases[i % len(bases)]
        out = os.path.join(out_dir, f"episode_{i:04d}.h5")
        defs = make_one(base, out, seed=1000 + i, severity=sev)
        print(f"  episode_{i:04d}  base={os.path.basename(os.path.dirname(base))}/{os.path.basename(base)}"
              f"  severity={sev:12s}  {len(defs)} defect(s): "
              + "; ".join(d["label"] for d in defs))
    with open(os.path.join(out_dir, "dataset_meta.json"), "w") as fh:
        json.dump({"source": "sim-imperfect",
                   "note": "synthetic real-world failures for curation stress-testing"}, fh)
    print(f"wrote {len(plan)} episodes to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--base", default=None, help="only use this dataset as the clean base")
    ap.add_argument("--out", default=OUT_DATASET, help="output dataset name")
    ap.add_argument("--severe", action="store_true", help="all episodes = worst-case stacked defects")
    a = ap.parse_args()
    run(a.n, a.base, a.out, a.severe)
