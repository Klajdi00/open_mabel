#!/usr/bin/env python3
"""
train.py — train ACT, Diffusion Policy and π0.5 (LeRobot's own models) on a curated LeRobot export, with the data
cleaning, clip sampling, train/val/test split, evaluation and plots in one place: frozen config + stats beside every
checkpoint, seeded split, curves.

    python train.py prepare --dataset <name or path>                     # frame cache + clean table (once)
    python train.py train --policy act --name act_v1 --set dataset=<name> # or --policy dp; --set key=value
    python train.py train --policy pi05 --name pi05_v1 --set dataset=<name>   # CUDA
    python train.py eval  --run runs/act_v1                              # test split, last checkpoint (--ckpt best)
    python train.py compare runs/act_v1 runs/dp_v1                       # side-by-side figures + table

Robot profiles (`--robot nova|mabel`, see ROBOTS): `nova` is the bimanual NOVA configuration — inputs head_left +
wrist_left + wrist_right (240×320) and 1 s of measured joint positions (6 samples, 0.2 s apart); output 4 s (120
steps at 30 Hz) of the desired 64-vector (joint targets, base twist, lift rate). `mabel` reads any export from
../../curate_export.py / h5_to_lerobot_v30.py (unnamed state/action columns, every camera in the export).

Cleaning, all measured from the data: a joint without fresh feedback (`observation.joint_live` = 0, when the export
has that column) is interpolated and a clip that needs more than `max_interp_s` of it is dropped; a robot-log gap
(held command) is interpolated the same way; a camera frame older than `max_cam_age_s` is stale — a clip whose current
frame is stale, or whose window is more than `max_stale_frac` stale, is dropped; state and action are smoothed with a
zero-phase Gaussian. Every valid frame of an episode anchors one clip. Split is by take (an episode's spans never
straddle splits). Camera / log ages need the raw recordings (meta/nova_episodes.jsonl + a raw-take reader, see
RAW_READER); without them every frame is treated as fresh.

π0.5 (`lerobot/pi05_base`, PaliGemma 3B + 300M action expert, flow matching): the VLM stays frozen in bf16 and the
action expert + action/time projections train on fp32 master weights (fits a 16 GB GPU). Its action layers were
pretrained for 32 dims: they are widened to ours, the pretrained 32 copied in. It reads the CURRENT state only, binned
into its prompt, so state and action are quantile-normalized to [-1, 1]; images go in at [0, 1] (no ImageNet norm).
Horizon 50 steps (1.67 s at 30 Hz), its pretraining chunk. PI05_RUN holds these run-level overrides.

Runs: runs/<name>/{config.json, split.json, stats.json, metrics.jsonl, status.json, plots/, checkpoints/
{best,last}/pretrained_model, eval/}. pretrained_model = LeRobot save_pretrained + nova_policy.json (the I/O contract).
Locations: POLICY_DATASET_ROOT (datasets, default <repo>/curated_dataset), POLICY_RUNS_ROOT (default ./runs),
POLICY_CACHE (frame cache, default ~/.cache/policy_training).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # repository root (robot_learning/imitation_learning/policy_training)
DATASETS = Path(os.environ.get("POLICY_DATASET_ROOT", REPO / "curated_dataset"))
RUNS = Path(os.environ.get("POLICY_RUNS_ROOT", HERE / "runs"))
CACHE = Path(os.environ.get("POLICY_CACHE", Path.home() / ".cache" / "policy_training"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Optional raw-take reader (module name) providing camera_index(take, cam)["t"] and take_arrays(take)["t"]: the
# recorder-side timestamps that let prepare() measure camera-frame and robot-log ages. Not part of this release;
# without it every frame is treated as fresh (the stale-frame / held-log checks then pass everything).
RAW_READER = os.environ.get("POLICY_RAW_READER", "")

GROUPS = {"torso+neck": ("torso", "neck_"), "left arm": ("left_arm_",), "right arm": ("right_arm_",),
          "wrists": ("left_wrist", "right_wrist"), "left hand": ("left_thumb", "left_index", "left_middle", "left_ring", "left_pinky"),
          "right hand": ("right_thumb", "right_index", "right_middle", "right_ring", "right_pinky")}
HBINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]          # s — error by how far ahead the chunk looks

# Robot profiles: run-level overrides applied with `--robot` (after DEFAULTS, before --set). `cameras: "all"` = every
# video stream in the export. Joint groups (GROUPS) that match no column are skipped; with none matching, the
# per-group reports collapse to one "all joints" group.
ROBOTS = {
    "nova": {"robot": "nova"},
    "mabel": {"robot": "mabel", "cameras": "all", "lift_joints": 2},
}
ROLLOUT_JOINTS = ("right_arm_1", "right_arm_3", "right_arm_5", "right_wrist", "right_index_mcp", "right_thumb_mcp")

DEFAULTS = {
    "robot": "nova",
    "dataset": "pick_and_place",     # a name under POLICY_DATASET_ROOT, or a path to a LeRobot dataset
    "cameras": ["head_left", "wrist_left", "wrist_right"],
    "image_size": [240, 320],
    "chunk_s": 4.0,                  # predicted horizon (output)
    "history_s": 1.0,                # proprio history (input)
    "history_stride_s": 0.2,         # → 6 samples: t-1.0 … t
    "lift_joints": 1,                # the lift is one motor: its two model stages merge into `lift` (m); runs without the key = 2
    "n_action_steps": 30,            # LeRobot select_action only; deploy Runtime predicts continuously
    "split": [0.8, 0.1, 0.1],        # train / val / test, by take
    "seed": 42,
    "state_sigma_s": 0.033,          # zero-phase Gaussian smoothing (σ, s); -3 dB at 0.1325/σ ≈ 4 Hz
    "action_sigma_s": 0.033,
    "max_interp_s": 0.5,             # longest interpolated joint / log gap a clip may contain
    "max_cam_age_s": 0.1,            # a frame older than this (3 frames at 30 Hz) is stale
    "max_stale_frac": 0.2,           # clip dropped if more of its window is stale
    "min_future_s": 0.5,             # real (unpadded) future a clip needs
    "std_floor": 0.02,               # rad — no joint is scaled up beyond this
    "state_noise": 0.05,             # σ in normalized units, train only
    "aug_shift_px": 12, "aug_brightness": 0.2, "aug_contrast": 0.2,
    "batch_size": 24, "num_workers": 6, "steps": 8000,
    "eval_every": 500, "val_clips": 384, "log_every": 25,
    "device": "cuda",                # falls back to mps, then cpu, when unavailable
    "act": {"dim_model": 512, "n_heads": 8, "dim_feedforward": 3200, "n_encoder_layers": 4, "n_decoder_layers": 1,
            "latent_dim": 32, "n_vae_encoder_layers": 4, "dropout": 0.1, "kl_weight": 10.0,
            "lr": 4e-5, "lr_backbone": 1e-5, "weight_decay": 1e-4, "warmup": 500, "grad_clip": 10.0,
            "temporal_ensemble_coeff": None},
    "dp": {"down_dims": [256, 512, 1024], "kernel_size": 5, "n_groups": 8, "diffusion_step_embed_dim": 128,
           "spatial_softmax_num_keypoints": 32, "num_train_timesteps": 100, "num_inference_steps": 10,
           "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon", "clip_sample_range": 5.0,
           "lr": 1e-4, "weight_decay": 1e-6, "warmup": 500, "grad_clip": 10.0},
    "pi05": {"base": "lerobot/pi05_base",                                          # null = random init (smoke tests)
             "task": "pick up the object and place onto the plate",
             "train_expert_only": True, "gradient_checkpointing": True, "num_inference_steps": 10,
             "max_state_dim": 64, "max_action_dim": 64,
             "lr": 2.5e-5, "weight_decay": 0.01, "warmup": 1000, "grad_clip": 1.0,       # openpi fine-tune optimizer
             "adam_8bit": True},
}
PI05_RUN = {"history_s": 0.0, "chunk_s": 50 / 30, "n_action_steps": 25, "batch_size": 8, "num_workers": 8,
            "steps": 30000, "device": "cuda"}


def deep_update(d, u):
    for k, v in u.items():
        d[k] = deep_update(dict(d.get(k) or {}), v) if isinstance(v, dict) else v
    return d


def parse_set(pairs):
    """--set dp.lr=3e-4 steps=5000 cameras=head_left,wrist_left → nested dict (JSON values where they parse)."""
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            v = v.split(",") if "," in v else v
        node = out
        *path, last = k.split(".")
        for x in path:
            node = node.setdefault(x, {})
        node[last] = v
    return out


# ───────────────────────────── prepare: frame cache + per-row table (once per dataset × size) ─────────────────────

def dataset_root(dataset):
    """A dataset path, a name under DATASETS, or a name anywhere below it (curate_export nests <realm>/<run>/<name>)."""
    for cand in (Path(dataset).expanduser(), DATASETS / dataset):
        if (cand / "meta" / "info.json").exists():
            return cand.resolve()
    hits = sorted(DATASETS.glob(f"**/{dataset}/meta/info.json"))
    if hits:
        return hits[-1].parent.parent
    raise SystemExit(f"dataset {dataset!r} not found (a path, or a name under {DATASETS}; set POLICY_DATASET_ROOT)")


def cache_dir(dataset, size):
    return CACHE / Path(dataset).name / f"{size[0]}x{size[1]}"


def feature_names(info, key):
    """A feature's column names; LeRobot allows a list, {"motors": [...]} or null (→ key_0, key_1, …)."""
    f = info["features"][key]
    n = f.get("names")
    if isinstance(n, dict):
        n = next(iter(n.values()), None)
    if not isinstance(n, list) or len(n) != f["shape"][0]:
        n = [f"{key.split('.')[-1]}_{i}" for i in range(f["shape"][0])]
    return [str(x) for x in n]


def video_files(root, key):
    """Episode-ordered mp4s of one camera: v3.0 videos/<key>/chunk-*/file-*.mp4 or v2.1 videos/chunk-*/<key>/*.mp4."""
    return sorted((root / "videos" / key).glob("*/*.mp4")) or sorted(root.glob(f"videos/*/{key}/*.mp4"))


def raw_reader():
    if not RAW_READER:
        return None
    import importlib
    return importlib.import_module(RAW_READER)


def prepare(dataset, size=(240, 320), force=False):
    """Decode every export video once into JPEG blobs at `size`, and build table.npz: per row the exported state,
    action and joint_live, plus what the export cannot say — each camera's frame age and the robot-log age, measured
    from the raw take when a raw-take reader is configured (the export holds the last frame / tick through a drop)."""
    import av
    import cv2
    import pyarrow.parquet as pq
    root = dataset_root(dataset)
    out = cache_dir(dataset, size)
    stamp = (root / "meta" / "info.json").stat().st_mtime
    if (out / "meta.json").exists() and not force and json.loads((out / "meta.json").read_text()).get("stamp") == stamp:
        return out
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    cams = [k.split(".")[-1] for k, v in info["features"].items() if v.get("dtype") == "video"]
    has_live = "observation.joint_live" in info["features"]
    cols = ["episode_index", "frame_index", "observation.state", "action"] + (["observation.joint_live"] if has_live else [])
    tabs = [pq.read_table(f, columns=cols) for f in sorted((root / "data").glob("*/*.parquet"))]
    def get(k):
        cs = [t.column(k) for t in tabs]
        return np.concatenate([np.stack(c.to_numpy(zero_copy_only=False)) if "list" in str(c.type) else c.to_numpy() for c in cs])
    ep, fi = get("episode_index").astype(np.int32), get("frame_index").astype(np.int32)
    order = np.lexsort((fi, ep))
    ep, fi = ep[order], fi[order]
    state, action = (get(k)[order].astype(np.float32) for k in ("observation.state", "action"))
    live = get("observation.joint_live")[order].astype(np.float32) if has_live else np.ones_like(state)
    n = len(ep)
    fps = info["fps"]
    ep_file = root / "meta" / "nova_episodes.jsonl"          # episode → raw take (optional; recorder-specific)
    if ep_file.exists():
        eps = [json.loads(l) for l in ep_file.read_text().splitlines() if l.strip()]
    else:
        eps = [{"episode_index": int(k), "t0": 0.0, "raw": None} for k in np.unique(ep)]
    reader = raw_reader()
    cam_age = np.zeros((n, len(cams)), np.float32)
    log_age = np.zeros(n, np.float32)
    meta_eps = []
    for e in eps:
        k = e["episode_index"]
        rows = np.where(ep == k)[0]
        if not len(rows):
            continue
        t = e.get("t0", 0.0) + np.arange(len(rows)) / fps
        raw = Path(e["raw"]) if e.get("raw") else None
        found = bool(reader and raw and raw.exists())
        if found:
            for c, name in enumerate(cams):
                ts = reader.camera_index(raw, name)["t"]
                j = np.searchsorted(ts, t, side="right") - 1
                cam_age[rows, c] = np.where(j >= 0, t - ts[np.clip(j, 0, None)], 1e3)
            tt = reader.take_arrays(raw)["t"]
            j = np.searchsorted(tt, t, side="right") - 1
            log_age[rows] = np.where(j >= 0, t - tt[np.clip(j, 0, None)], 1e3)
        meta_eps.append({"episode": k, "take": "/".join(raw.parts[-3:]) if raw else f"episode_{k}",
                         "frames": int(len(rows)), "row0": int(rows[0]), "raw_found": found})
    if not any(m["raw_found"] for m in meta_eps):
        print("no raw takes read — camera and log ages unknown (every frame treated as fresh)")
    # frames: decode each episode's mp4 sequentially, resize, JPEG
    h, w = size
    for name in cams:
        key = f"observation.images.{name}"
        blob, offs = bytearray(), np.zeros(n + 1, np.int64)
        row = 0
        for v in video_files(root, key):
            c = av.open(str(v))
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for f in c.decode(s):
                img = cv2.resize(f.to_ndarray(format="rgb24"), (w, h), interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                blob += jpg.tobytes()
                row += 1
                offs[row] = len(blob)
            c.close()
        if row != n:
            raise ValueError(f"{name}: decoded {row} frames for {n} rows — check the export's videos against its data")
        (out / f"{name}.jpg.bin").write_bytes(bytes(blob))
        np.save(out / f"{name}.offs.npy", offs)
        print(f"{name}: {n} frames → {len(blob) / 2**20:.0f} MB", flush=True)
    np.savez(out / "table.npz", episode=ep, frame=fi, state=state, action=action, live=live, cam_age=cam_age, log_age=log_age)
    (out / "meta.json").write_text(json.dumps({"dataset": str(dataset), "root": str(root), "stamp": stamp, "fps": fps,
                                               "size": list(size), "cams": cams,
                                               "state_names": feature_names(info, "observation.state"),
                                               "action_names": feature_names(info, "action"), "episodes": meta_eps,
                                               "rows": int(n)}, indent=1))
    return out


# ───────────────────────────── clean + clips + split + stats ─────────────────────────────

def gaussian_smooth(x, sigma):
    """Zero-phase Gaussian along axis 0, edges held (no phase lag; a causal filter would delay the target)."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1).astype(np.float32)


def runs_of(mask):
    """Lengths of True runs, as a per-element array (0 where False)."""
    out = np.zeros(len(mask), np.int32)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def build(cfg, verbose=True):
    """The cleaned arrays, the valid clip anchors per split, and normalization stats — all from the config."""
    size = tuple(cfg["image_size"])
    cdir = prepare(cfg["dataset"], size)
    meta = json.loads((cdir / "meta.json").read_text())
    z = np.load(cdir / "table.npz")
    fps = meta["fps"]
    ep, state, action, live = z["episode"], z["state"].copy(), z["action"].copy(), z["live"] > 0
    if cfg["cameras"] == "all":
        cfg["cameras"] = list(meta["cams"])
    missing = [c for c in cfg["cameras"] if c not in meta["cams"]]
    if missing:
        raise SystemExit(f"cameras {missing} not in the export (has {meta['cams']}); --set cameras=a,b or --robot mabel")
    cam_cols = [meta["cams"].index(c) for c in cfg["cameras"]]
    cam_age, log_age = z["cam_age"][:, cam_cols], z["log_age"]
    snames, anames = meta["state_names"], meta["action_names"]
    hist = [-int(round(k * cfg["history_stride_s"] * fps)) for k in range(int(round(cfg["history_s"] / cfg["history_stride_s"])), -1, -1)]
    chunk = int(round(cfg["chunk_s"] * fps))
    max_gap = int(round(cfg["max_interp_s"] * fps))
    report = {"rows": int(len(ep)), "fps": fps, "history_offsets": hist, "chunk": chunk}
    bad_prop = np.zeros(len(ep), bool)
    stale = (cam_age > cfg["max_cam_age_s"]).any(1)
    raw_state, raw_action = state.copy(), action.copy()
    held_rows = interp_rows = 0
    for e in np.unique(ep):
        r = np.where(ep == e)[0]
        # robot-log gaps: the export held the last command — interpolate it between fresh ticks
        held = log_age[r] > 1.5 / fps
        if held.any() and not held.all():
            fresh = ~held
            for j in range(action.shape[1]):
                action[r[held], j] = np.interp(r[held], r[fresh], action[r[fresh], j])
            held_rows += int(held.sum())
        long_held = runs_of(held) > max_gap
        # joint feedback: the export interpolated it already; a long dead run is fabrication → unusable rows
        dead = ~live[r]
        interp_rows += int(dead.any(1).sum())
        long_dead = np.zeros(len(r), bool)
        for j in range(dead.shape[1]):
            if dead[:, j].any():
                long_dead |= runs_of(dead[:, j]) > max_gap
        bad_prop[r] = long_held | long_dead
        state[r] = gaussian_smooth(state[r], cfg["state_sigma_s"] * fps)
        action[r] = gaussian_smooth(action[r], cfg["action_sigma_s"] * fps)
    report.update(held_log_rows=held_rows, interpolated_joint_rows=interp_rows, bad_prop_rows=int(bad_prop.sum()),
                  stale_video_rows=int(stale.sum()))
    # smoothing removed (rms, deg) per group — the noise floor we take out
    gi = group_index(snames)
    report["smoothing_removed_deg"] = {g: [float(np.degrees(np.sqrt(np.mean((state[:, ix] - raw_state[:, ix]) ** 2)))),
                                           float(np.degrees(np.sqrt(np.mean((action[:, ix] - raw_action[:, ix]) ** 2))))]
                                       for g, ix in gi.items()}
    if cfg.get("lift_joints", 2) == 1 and {"lift_lower", "lift_upper"} <= set(snames) & set(anames):
        state, snames = merge_lift(state, snames)
        action, anames = merge_lift(action, anames)
    # split by take
    takes = [m["take"] for m in meta["episodes"]]
    uniq = sorted(set(takes))
    rng = np.random.default_rng(cfg["seed"])
    perm = [uniq[i] for i in rng.permutation(len(uniq))]
    ntr = int(round(cfg["split"][0] * len(uniq)))
    nva = int(round(cfg["split"][1] * len(uniq)))
    split_of_take = {t: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, t in enumerate(perm)}
    ep_split = {m["episode"]: split_of_take[m["take"]] for m in meta["episodes"]}
    # clip anchors: every frame with a fresh current image, ≤ max_stale_frac stale window, no long interpolation,
    # and min_future_s of real future
    clips = {"train": [], "val": [], "test": []}
    drop = {"stale_now": 0, "stale_window": 0, "prop": 0, "short_future": 0}
    min_future = int(round(cfg["min_future_s"] * fps))
    for m in meta["episodes"]:
        r0, nf = m["row0"], m["frames"]
        for t in range(nf):
            g = r0 + t
            lo, hi = max(0, t + hist[0]), min(nf, t + chunk)
            if nf - t < min_future:
                drop["short_future"] += 1; continue
            if stale[g]:
                drop["stale_now"] += 1; continue
            if stale[r0 + lo:r0 + hi].mean() > cfg["max_stale_frac"]:
                drop["stale_window"] += 1; continue
            if bad_prop[r0 + lo:r0 + hi].any():
                drop["prop"] += 1; continue
            clips[ep_split[m["episode"]]].append((m["episode"], r0, nf, t))
    report["dropped_clips"] = drop
    report["split"] = {s: {"takes": sum(1 for t in uniq if split_of_take[t] == s),
                           "episodes": sum(1 for v in ep_split.values() if v == s),
                           "frames": int(sum(m["frames"] for m in meta["episodes"] if ep_split[m["episode"]] == s)),
                           "clips": len(clips[s])} for s in clips}
    # stats on train rows; masked dims (wheels, lift, base, anything static) are held at their mean
    tr_rows = np.concatenate([np.arange(m["row0"], m["row0"] + m["frames"]) for m in meta["episodes"] if ep_split[m["episode"]] == "train"])
    def stats(x, names):
        mu, sd = x[tr_rows].mean(0), x[tr_rows].std(0)
        mask = np.array([n.endswith(("_steer", "_drive")) or n.startswith(("lift", "base_")) for n in names]) | (sd < 1e-4)
        if cfg.get("policy") == "pi05":               # q01 → -1, q99 → +1 (openpi's quantile norm), as centre / half-range
            lo, hi = np.quantile(x[tr_rows], [0.01, 0.99], axis=0)
            mu, sd = (lo + hi) / 2, (hi - lo) / 2
        sd = np.maximum(sd, cfg["std_floor"])
        return {"mean": mu.astype(np.float32), "std": sd.astype(np.float32), "mask": mask}
    st = {"state": stats(state, snames), "action": stats(action, anames)}
    report["masked"] = {"state": [n for n, k in zip(snames, st["state"]["mask"]) if k],
                        "action": [n for n, k in zip(anames, st["action"]["mask"]) if k]}
    if verbose:
        print(json.dumps({k: v for k, v in report.items() if k != "smoothing_removed_deg"}, indent=1))
    return {"cdir": cdir, "meta": meta, "state": state, "action": action, "clips": clips, "stats": st, "hist": hist,
            "chunk": chunk, "report": report, "ep_split": ep_split, "split_of_take": split_of_take,
            "state_names": snames, "action_names": anames}


def merge_lift(x, names):
    """lift_lower + lift_upper (the model's two telescoping stages, always equal: one motor) → one `lift` column, the
    column's extension in m, at lift_lower's place."""
    i, j = names.index("lift_lower"), names.index("lift_upper")
    y = np.delete(x, j, axis=-1)
    y[..., i if i < j else i - 1] = x[..., i] + x[..., j]
    return y, [("lift" if k == i else n) for k, n in enumerate(names) if k != j]


def group_index(names):
    out = {g: [i for i, n in enumerate(names) if n.startswith(p)] for g, p in GROUPS.items()}
    out = {g: ix for g, ix in out.items() if ix}
    return out or {"all joints": list(range(len(names)))}


def norm(x, s):
    y = (x - s["mean"]) / s["std"]
    y[..., s["mask"]] = 0.0
    return y


def denorm(y, s):
    x = y * s["std"] + s["mean"]
    x[..., s["mask"]] = s["mean"][s["mask"]]
    return x


# ───────────────────────────── torch dataset ─────────────────────────────

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Clips:
    """One sample = one anchor frame: current images, the stacked (normalized) state history, the 4 s action chunk
    with a pad mask past the episode end. Images come from the JPEG cache (memmapped, decoded in the worker)."""

    def __init__(self, B, split, cfg, train):
        self.c, self.cfg, self.train = B["clips"][split], cfg, train
        self.cdir, self.cams = str(B["cdir"]), cfg["cameras"]
        self.s = norm(B["state"], B["stats"]["state"]).astype(np.float32)
        self.a = norm(B["action"], B["stats"]["action"]).astype(np.float32)
        self.hist, self.chunk = B["hist"], B["chunk"]
        self._blobs = None

    def __len__(self):
        return len(self.c)

    def _img(self, cam, g):
        import cv2
        if self._blobs is None:
            self._blobs = {c: (np.memmap(f"{self.cdir}/{c}.jpg.bin", np.uint8, "r"), np.load(f"{self.cdir}/{c}.offs.npy"))
                           for c in self.cams}
        blob, offs = self._blobs[cam]
        img = cv2.imdecode(np.asarray(blob[offs[g]:offs[g + 1]]), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        if self.train:
            rng = np.random
            k = self.cfg["aug_shift_px"]
            if k:
                dy, dx = rng.randint(-k, k + 1, 2)
                img = np.roll(img, (dy, dx), (0, 1))
            img = (img - img.mean()) * (1 + rng.uniform(-1, 1) * self.cfg["aug_contrast"]) + img.mean() \
                + rng.uniform(-1, 1) * self.cfg["aug_brightness"] * 0.5
            img = np.clip(img, 0, 1)
        if self.cfg.get("policy") == "pi05":         # SigLIP takes [0, 1] (π0.5 maps it to [-1, 1] itself)
            return img.transpose(2, 0, 1)
        return ((img - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)

    def __getitem__(self, i):
        import torch
        e, r0, nf, t = self.c[i]
        hi = [r0 + max(0, t + o) for o in self.hist]
        st = self.s[hi].copy()
        if self.train and self.cfg["state_noise"]:
            st += np.random.randn(*st.shape).astype(np.float32) * self.cfg["state_noise"]
        idx = np.arange(t, t + self.chunk)
        pad = idx >= nf
        act = self.a[r0 + np.minimum(idx, nf - 1)]
        out = {"observation.state": torch.from_numpy(st.reshape(-1)), "action": torch.from_numpy(act),
               "action_is_pad": torch.from_numpy(pad)}
        for c in self.cams:
            out[f"observation.images.{c}"] = torch.from_numpy(np.ascontiguousarray(self._img(c, r0 + t)))
        return out


# ───────────────────────────── policies ─────────────────────────────

def make_policy(kind, cfg, state_dim, action_dim):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    h, w = cfg["image_size"]
    feats = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    for c in cfg["cameras"]:
        feats[f"observation.images.{c}"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
    outf = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    ident = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY}
    chunk = int(round(cfg["chunk_s"] * cfg.get("fps", 30)))
    if kind == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        p = cfg["act"]
        conf = ACTConfig(input_features=feats, output_features=outf, normalization_mapping=ident, device=cfg["device"],
                         chunk_size=chunk, n_action_steps=1 if p["temporal_ensemble_coeff"] else cfg["n_action_steps"],
                         temporal_ensemble_coeff=p["temporal_ensemble_coeff"],
                         dim_model=p["dim_model"], n_heads=p["n_heads"], dim_feedforward=p["dim_feedforward"],
                         n_encoder_layers=p["n_encoder_layers"], n_decoder_layers=p["n_decoder_layers"],
                         latent_dim=p["latent_dim"], n_vae_encoder_layers=p["n_vae_encoder_layers"], dropout=p["dropout"],
                         kl_weight=p["kl_weight"], optimizer_lr=p["lr"], optimizer_lr_backbone=p["lr_backbone"],
                         optimizer_weight_decay=p["weight_decay"])
        return ACTPolicy(conf)
    if kind == "pi05":
        return make_pi05(cfg, feats, outf, ident, chunk)
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    p = cfg["dp"]
    conf = DiffusionConfig(input_features=feats, output_features=outf, normalization_mapping=ident, device=cfg["device"],
                           n_obs_steps=1, horizon=chunk, n_action_steps=cfg["n_action_steps"], drop_n_last_frames=0,
                           crop_shape=None, crop_is_random=False, use_separate_rgb_encoder_per_camera=False,
                           spatial_softmax_num_keypoints=p["spatial_softmax_num_keypoints"], down_dims=tuple(p["down_dims"]),
                           kernel_size=p["kernel_size"], n_groups=p["n_groups"],
                           diffusion_step_embed_dim=p["diffusion_step_embed_dim"], noise_scheduler_type="DDIM",
                           num_train_timesteps=p["num_train_timesteps"], num_inference_steps=p["num_inference_steps"],
                           beta_schedule=p["beta_schedule"], prediction_type=p["prediction_type"], clip_sample=True,
                           clip_sample_range=p["clip_sample_range"], do_mask_loss_for_padding=True,
                           optimizer_lr=p["lr"], optimizer_weight_decay=p["weight_decay"])
    return DiffusionPolicy(conf)


def make_pi05(cfg, feats, outf, ident, chunk):
    """π0.5 built straight on the GPU in bf16 (no 14 GB fp32 copy in RAM), then the base weights streamed in tensor by
    tensor. Projections pretrained for 32 action dims keep their 32 in the first slots; the new ones start random."""
    import torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    p = cfg["pi05"]
    conf = PI05Config(input_features=feats, output_features=outf, normalization_mapping=ident, device=cfg["device"],
                      dtype="bfloat16", chunk_size=chunk, n_action_steps=cfg["n_action_steps"],
                      max_state_dim=p["max_state_dim"], max_action_dim=p["max_action_dim"],
                      num_inference_steps=p["num_inference_steps"], gradient_checkpointing=p["gradient_checkpointing"],
                      train_expert_only=p["train_expert_only"], freeze_vision_encoder=p["train_expert_only"],
                      optimizer_lr=p["lr"], optimizer_weight_decay=p["weight_decay"])
    policy = build_pi05(conf)
    policy.nova_task = p["task"]
    if p["base"]:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        own, used, grown = policy.state_dict(), 0, []
        with torch.no_grad(), safe_open(hf_hub_download(p["base"], "model.safetensors"), "pt", device="cpu") as f:
            for k in f.keys():
                nk = k.replace("action_time_mlp_", "time_mlp_")
                nk = nk if nk.startswith("model.") else "model." + nk
                if nk not in own:
                    continue
                v, dst = f.get_tensor(k), own[nk]
                if v.shape != dst.shape:
                    grown.append(f"{nk} {tuple(v.shape)}→{tuple(dst.shape)}")
                    dst[tuple(slice(0, s) for s in v.shape)].copy_(v)
                else:
                    dst.copy_(v)
                used += 1
        print(f"pi05 base {p['base']}: {used}/{len(own)} tensors loaded; widened {grown}", flush=True)
    return policy


def build_pi05(conf):
    import torch
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(conf.device):
            policy = PI05Policy(conf)
    finally:
        torch.set_default_dtype(torch.float32)
    m = policy.model
    # the expert's 257k-vocab text head (263 M params) is never called — its hidden states go to action_out_proj —
    # but train_expert_only leaves it trainable: 4 GB of master weights + Adam state for nothing
    m.paligemma_with_expert.gemma_expert.lm_head = None
    m.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")      # its fp32 norms / embeddings back
    for lin in (m.action_in_proj, m.action_out_proj, m.time_mlp_in, m.time_mlp_out):
        lin.float()                                                          # as PI05Pytorch builds them
    return policy


def load_policy_dir(d, device):
    """A saved pretrained_model on `device`, eval mode, with its nova_policy.json."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    mp = json.loads((d / "nova_policy.json").read_text())
    conf = PreTrainedConfig.from_pretrained(d)
    conf.device = str(device)
    if conf.type == "pi05":
        from safetensors.torch import load_file
        pol = build_pi05(conf)
        sd = load_file(str(d / "model.safetensors"), device=str(device))
        pg = "model.paligemma_with_expert.paligemma."      # save_pretrained keeps one of a tied pair: embed_tokens ≡ lm_head
        sd.setdefault(pg + "model.language_model.embed_tokens.weight", sd.get(pg + "lm_head.weight"))
        pol.load_state_dict(sd)
        pol.nova_task = mp.get("task")
    else:
        pol = get_policy_class(conf.type).from_pretrained(d, config=conf)
    return pol.to(device).eval(), mp


_TOK = {}


def pi05_tokens(policy, batch):
    """π0.5's prompt: 'Task: …, State: <each state dim in 256 bins>;\\nAction: ' (openpi PaligemmaTokenizer), padded."""
    import torch
    if "sp" not in _TOK:       # openpi's own copy (public big_vision bucket); the HF google/paligemma one is gated
        import sentencepiece
        import urllib.request
        f = CACHE / "paligemma_tokenizer.model"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve("https://storage.googleapis.com/big_vision/paligemma_tokenizer.model", f)
        _TOK["sp"] = sentencepiece.SentencePieceProcessor(model_file=str(f))
    sp, c, s = _TOK["sp"], policy.config, batch["observation.state"]
    q = torch.nn.functional.pad(s, (0, c.max_state_dim - s.shape[-1])).clamp(-1, 1).float().cpu().numpy()
    bins = np.digitize(q, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    n, L = len(bins), c.tokenizer_max_length
    ids, mask = np.zeros((n, L), np.int64), np.zeros((n, L), bool)               # Gemma pad id = 0, right-padded
    for i, b in enumerate(bins):
        t = ([sp.bos_id()] + sp.encode(f"Task: {policy.nova_task}, State: {' '.join(map(str, b))};\nAction: "))[:L]
        ids[i, :len(t)], mask[i, :len(t)] = t, True
    out = dict(batch)
    out["observation.language.tokens"] = torch.from_numpy(ids).to(s.device)
    out["observation.language.attention_mask"] = torch.from_numpy(mask).to(s.device)
    return out


def to_dp(batch):
    """DP wants a time axis on observations (n_obs_steps = 1: the history lives inside the state vector)."""
    b = dict(batch)
    for k in list(b):
        if k.startswith("observation."):
            b[k] = b[k].unsqueeze(1)
    return b


def predict_chunk(policy, kind, batch, generator=None):
    """Full-horizon prediction in normalized units, (B, chunk, A): ACT with the prior latent (z = 0), DP sampled
    (`generator`: a seeded torch.Generator makes the DP sample a deterministic function of the observation)."""
    import torch
    with torch.no_grad():
        if kind == "act":
            b = dict(batch)
            b["observation.images"] = [b[k] for k in policy.config.image_features]
            return policy.model(b)[0]
        if kind == "pi05":
            return policy.predict_action_chunk(pi05_tokens(policy, batch)).float()
        d = policy.diffusion
        b = to_dp(batch)
        b["observation.images"] = torch.stack([b[k] for k in policy.config.image_features], dim=-4)
        cond = d._prepare_global_conditioning(b)
        return d.conditional_sample(cond.shape[0], global_cond=cond, generator=generator)


def train_loss(policy, kind, batch):
    if kind == "pi05":
        return policy.forward(pi05_tokens(policy, batch))[0], {}
    loss, info = policy.forward(batch if kind == "act" else to_dp(batch))
    return loss, info or {}


# ───────────────────────────── train ─────────────────────────────

def save_ckpt(policy, run, tag, B, cfg, kind, step, extra=None):
    d = run / "checkpoints" / tag / "pretrained_model"
    tmp = d.with_name("pretrained_model.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save_pretrained(tmp)
    if kind != "pi05":                  # π0.5's processor fetches the gated HF tokenizer; Runtime builds its own prompt
        from lerobot.policies.factory import make_pre_post_processors
        pre, post = make_pre_post_processors(policy.config, dataset_stats=None)
        pre.save_pretrained(tmp); post.save_pretrained(tmp)
    st = B["stats"]
    (tmp / "nova_policy.json").write_text(json.dumps({
        "kind": kind, "step": step, "run": run.name, "fps": B["meta"]["fps"], "cameras": cfg["cameras"],
        "image_size": cfg["image_size"], "history_offsets": B["hist"], "chunk": B["chunk"],
        "n_action_steps": cfg["n_action_steps"], "state_names": B["state_names"],
        "action_names": B["action_names"], "robot_joints": B["meta"]["state_names"],
        "stats": {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist(), "mask": v["mask"].tolist()} for k, v in st.items()},
        "imagenet_norm": kind != "pi05", "task": cfg.get("pi05", {}).get("task") if kind == "pi05" else None,
        **(extra or {})}, indent=1))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    return d


class Logger:
    """metrics.jsonl + status.json; run None = a silent logger (torchrun ranks > 0)."""
    def __init__(self, run):
        self.f = open(run / "metrics.jsonl", "a") if run else None
        self.run = run

    def __call__(self, **kv):
        if not self.f:
            return
        kv["time"] = time.time()
        self.f.write(json.dumps(kv) + "\n")
        self.f.flush()

    def status(self, **kv):
        if not self.run:
            return
        p = self.run / "status.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(kv, updated=time.time())
        p.with_suffix(".tmp").write_text(json.dumps(old, indent=1))
        p.with_suffix(".tmp").replace(p)


def evaluate(policy, kind, loader, B, device, max_batches=None):
    """Open-loop error of the predicted chunk vs the recorded desired command: normalized L1 (non-pad, unmasked),
    and degrees per joint group × horizon bin; plus the policy's own objective (val loss)."""
    import torch
    am = ~B["stats"]["action"]["mask"]
    names = B["action_names"]
    gi = group_index(names)
    fps, chunk = B["meta"]["fps"], B["chunk"]
    sd = torch.tensor(B["stats"]["action"]["std"], device=device)
    tot_abs = torch.zeros(chunk, len(names), device=device)
    tot_n = torch.zeros(chunk, 1, device=device)
    l1n, nn_, vloss = 0.0, 0, []
    policy.eval()
    for k, batch in enumerate(loader):
        if max_batches and k >= max_batches:
            break
        batch = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}
        pred = predict_chunk(policy, kind, batch)
        valid = (~batch["action_is_pad"]).float().unsqueeze(-1)
        err = (pred - batch["action"]).abs() * valid
        tot_abs += (err * sd).sum(0)
        tot_n += valid.sum(0)
        l1n += float(err[..., torch.tensor(am, device=device)].sum()) / max(1, int(am.sum()))
        nn_ += float(valid.sum())
        policy.train()
        with torch.no_grad():
            vloss.append(float(train_loss(policy, kind, batch)[0]))
        policy.eval()
    mae = (tot_abs / tot_n.clamp(min=1)).cpu().numpy()              # (chunk, A) in rad
    out = {"val_l1": l1n / max(1, nn_), "val_loss": float(np.mean(vloss)) if vloss else None,
           "horizon_deg": {g: [float(np.degrees(mae[:, ix].mean(1))[i]) for i in range(chunk)] for g, ix in gi.items()},
           "bins_deg": {g: {f"{a:g}-{b:g}s": float(np.degrees(mae[int(a * fps):int(b * fps), ix].mean())) for a, b in HBINS
                            if int(a * fps) < chunk} for g, ix in gi.items()},
           "joint_deg": {n: float(np.degrees(mae[:, i].mean())) for i, n in enumerate(names) if am[i]}}
    policy.train()
    return out


def train(cfg, name, resume=False):
    """One process, or one per GPU under torchrun (data parallel: each rank its own slice of the clips, gradients
    averaged every step, rank 0 logs / validates / checkpoints while the others wait)."""
    import torch
    from torch.utils.data import DataLoader
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    main0 = rank == 0
    if world > 1:
        import datetime
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    kind = cfg["policy"]
    run = RUNS / name
    if (run / "config.json").exists() and not resume and main0:
        raise SystemExit(f"{run} exists — pick another --name or pass --resume")
    if main0:
        run.mkdir(parents=True, exist_ok=True)
        (run / "plots").mkdir(exist_ok=True)
    log = Logger(run) if main0 else Logger(None)
    log.status(state="preparing", policy=kind, name=name, pid=os.getpid(), steps=cfg["steps"], step=0, gpus=world)
    B = build(cfg, verbose=main0)
    if main0:
        (run / "config.json").write_text(json.dumps(cfg, indent=1))  # on resume too: records an extended `steps`
    if not resume and main0:
        (run / "split.json").write_text(json.dumps({"by_take": B["split_of_take"], "by_episode": {str(k): v for k, v in B["ep_split"].items()}}, indent=1))
        (run / "data_report.json").write_text(json.dumps(B["report"], indent=1))
        (run / "stats.json").write_text(json.dumps({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in B["stats"].items()}))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device(pick_device(cfg["device"]))
    if world > 1:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    state_dim = len(B["hist"]) * B["state"].shape[1]
    policy = make_policy(kind, {**cfg, "device": str(device), "fps": B["meta"]["fps"]}, state_dim, B["action"].shape[1]).to(device)
    p = cfg[kind]
    start = 0
    if resume and (run / "checkpoints" / "last" / "pretrained_model").exists():
        from safetensors.torch import load_file
        last = run / "checkpoints" / "last" / "pretrained_model"
        policy.load_state_dict(load_file(str(last / "model.safetensors")), strict=False)
        start = json.loads((last / "nova_policy.json").read_text())["step"]
    if world > 1:            # identical start on every rank (the widened action columns are random)
        with torch.no_grad():
            for t in policy.state_dict().values():
                dist.broadcast(t, 0)
        torch.manual_seed(cfg["seed"] + rank); np.random.seed(cfg["seed"] + rank)   # own noise / augmentation
    live, master = None, None
    if kind == "pi05":       # bf16 weights can't take 2.5e-5-sized steps: AdamW runs on fp32 copies, cast back each step
        live = [q for q in policy.parameters() if q.requires_grad]
        master = [q.detach().float().clone().requires_grad_(True) for q in live]
        if p["adam_8bit"]:   # 430 M trainable: fp32 Adam state is 3.4 GB, 8-bit 0.9 GB — the difference fits 16 GB
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(master, lr=p["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=p["weight_decay"])
        else:
            opt = torch.optim.AdamW(master, lr=p["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=p["weight_decay"])
        print(f"pi05: {sum(q.numel() for q in live) / 1e6:.1f} M trainable", flush=True)
    elif kind == "act":
        opt = torch.optim.AdamW(policy.get_optim_params(), lr=p["lr"], weight_decay=p["weight_decay"])
    else:
        opt = torch.optim.AdamW(policy.get_optim_params(), lr=p["lr"], betas=(0.95, 0.999), eps=1e-8,
                                weight_decay=p["weight_decay"])
    clip_params = master if master else list(policy.parameters())
    base_lr = [g["lr"] for g in opt.param_groups]      # before loading: the saved state holds the decayed lr
    if resume and (run / "checkpoints" / "last" / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(run / "checkpoints" / "last" / "optimizer.pt", map_location=device))
    def lr_at(s):      # linear warmup, cosine to 10 %
        if s < p["warmup"]:
            return (s + 1) / p["warmup"]
        q = (s - p["warmup"]) / max(1, cfg["steps"] - p["warmup"])
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, q)))
    tr = Clips(B, "train", cfg, True)
    va = Clips(B, "val", cfg, False)
    rng = np.random.default_rng(0)
    vsub = sorted(rng.choice(len(va), min(cfg["val_clips"], len(va)), replace=False).tolist()) if len(va) else []
    va.c = [va.c[i] for i in vsub]
    nw = cfg["num_workers"]
    sampler = None
    if world > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True, seed=cfg["seed"], drop_last=True)
    dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=sampler is None, sampler=sampler, num_workers=nw,
                    drop_last=True, persistent_workers=nw > 0, prefetch_factor=4 if nw else None)
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=min(nw, 4))
    nparam = sum(x.numel() for x in policy.parameters())
    log(event="start", policy=kind, params=nparam, train_clips=len(tr), val_clips=len(va), step=start, device=str(device))
    log.status(state="training", params=nparam, train_clips=len(tr), started=time.time(), report=B["report"])
    print(f"{kind}: {nparam / 1e6:.1f} M params, {len(tr)} train clips, {len(va)} val clips, device {device}, "
          f"{world} GPU × batch {cfg['batch_size']}", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    best = json.loads((run / "best.json").read_text())["val_l1"] if (run / "best.json").exists() else float("inf")
    step, t_last, n_last, acc, epoch = start, time.time(), start, [], 0
    grad_params = master if master else [q for q in policy.parameters() if q.requires_grad]
    policy.train()
    while step < cfg["steps"] and not stop["flag"]:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in dl:
            want = step >= cfg["steps"] or stop["flag"] or (run / "STOP").exists()
            if world > 1:            # every rank stops at the same step, or the others hang in all_reduce
                f = torch.tensor([float(want)], device=device)
                dist.all_reduce(f, op=dist.ReduceOp.MAX)
                want = bool(f.item())
            if want:
                stop["flag"] = True
                break
            for g, b0 in zip(opt.param_groups, base_lr):
                g["lr"] = b0 * lr_at(step)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, info = train_loss(policy, kind, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if master:
                for m_, q in zip(master, live):
                    m_.grad, q.grad = (None if q.grad is None else q.grad.float()), None
            if world > 1:
                allreduce_grads(grad_params, world)
            gn = float(torch.nn.utils.clip_grad_norm_(clip_params, p["grad_clip"]))
            opt.step()
            if master:
                with torch.no_grad():
                    for m_, q in zip(master, live):
                        q.copy_(m_)
            step += 1
            acc.append((float(loss.detach()), gn, info))
            if step % cfg["log_every"] == 0:
                dt = time.time() - t_last
                sps = (step - n_last) / max(dt, 1e-6)
                rec = {"event": "train", "step": step, "loss": float(np.mean([a[0] for a in acc])),
                       "grad_norm": float(np.mean([a[1] for a in acc])), "lr": opt.param_groups[0]["lr"], "steps_per_s": sps,
                       "samples_per_s": sps * cfg["batch_size"] * world}
                for k in acc[0][2]:
                    rec[k] = float(np.mean([a[2][k] for a in acc]))
                log(**rec)
                log.status(step=step, loss=rec["loss"], steps_per_s=sps, eta_s=(cfg["steps"] - step) / max(sps, 1e-6))
                acc, t_last, n_last = [], time.time(), step
            if (step % cfg["eval_every"] == 0 or step == cfg["steps"]) and not main0:
                dist.barrier()                                        # rank 0 validates + saves meanwhile
                t_last, n_last = time.time(), step
            elif step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                ev = evaluate(policy, kind, vl, B, device)
                log(event="val", step=step, val_l1=ev["val_l1"], val_loss=ev["val_loss"], bins_deg=ev["bins_deg"])
                save_ckpt(policy, run, "last", B, cfg, kind, step)
                torch.save(opt.state_dict(), run / "checkpoints" / "last" / "optimizer.pt")
                if ev["val_l1"] < best:
                    best = ev["val_l1"]
                    save_ckpt(policy, run, "best", B, cfg, kind, step, {"val_l1": best})
                    (run / "best.json").write_text(json.dumps({"step": step, **ev}, indent=1))
                log.status(val_l1=ev["val_l1"], best_val_l1=best)
                plots(run)
                if world > 1:
                    dist.barrier()
                t_last, n_last = time.time(), step
    if world > 1:
        dist.barrier()
        if not main0:
            dist.destroy_process_group()
            return run
    save_ckpt(policy, run, "last", B, cfg, kind, step)
    torch.save(opt.state_dict(), run / "checkpoints" / "last" / "optimizer.pt")
    (run / "STOP").unlink(missing_ok=True)
    done = step >= cfg["steps"]
    log(event="end", step=step, completed=done)
    log.status(state="evaluating" if done else "stopped", step=step)
    plots(run)
    if done:
        # own process: this one still holds the training model + optimizer (π0.5 OOMs on 16 GB reloading the checkpoint here)
        import subprocess
        subprocess.run([sys.executable, __file__, "eval", "--run", str(run)], check=False)
        log.status(state="done")
    if world > 1:
        dist.destroy_process_group()
    return run


def pick_device(want):
    """The requested device if present, else cuda → mps → cpu."""
    import torch
    have = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if have.get(str(want).split(":")[0]):
        return want
    return next(d for d in ("cuda", "mps", "cpu") if have[d])


def allreduce_grads(params, world):
    """Average gradients over the ranks: big tensors in place, small ones packed ≤ 64 M elements per call."""
    import torch
    import torch.distributed as dist
    small = []
    def flush():
        if small:
            flat = torch.cat([g.reshape(-1) for g in small])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for g in small:
                g.copy_(flat[o:o + g.numel()].view_as(g)); o += g.numel()
            small.clear()
    for q in params:
        g = q.grad
        if g is None:
            continue
        if g.numel() >= 16_000_000:
            dist.all_reduce(g); g /= world
        else:
            small.append(g)
            if sum(x.numel() for x in small) >= 64_000_000:
                flush()
    flush()


# ───────────────────────────── test-split evaluation ─────────────────────────────

def load_run_policy(run, tag="last"):
    import torch
    d = run / "checkpoints" / tag / "pretrained_model"
    if not d.exists():
        d = run / "checkpoints" / "last" / "pretrained_model"
    dev = pick_device("cuda")
    pol, mp = load_policy_dir(d, dev)
    return pol, mp, torch.device(dev), d


def evaluate_run(run, split="test", stride_s=0.5, tag="last"):
    """The last (or best-val) checkpoint on every clip of the split, anchors every stride_s: errors per group × horizon, per joint,
    plus rollout figures (predicted 4 s chunks every 1 s over the recorded command) for three test episodes."""
    import torch
    from torch.utils.data import DataLoader
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    kind = cfg["policy"]
    B = build(cfg, verbose=False)
    pol, mp, dev, d = load_run_policy(run, tag)
    ds = Clips(B, split, cfg, False)
    stride = max(1, int(round(stride_s * B["meta"]["fps"])))
    ds.c = [c for c in ds.c if c[3] % stride == 0]
    t0 = time.time()
    ev = evaluate(pol, kind, DataLoader(ds, batch_size=cfg["batch_size"], num_workers=4), B, dev)
    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    res = {"split": split, "checkpoint": str(d), "step": mp["step"], "clips": len(ds), "stride_s": stride_s, "fps": B["meta"]["fps"],
           "seconds": time.time() - t0, **ev}
    # inference latency (one chunk, batch 1)
    one = {k: v[:1].to(dev) for k, v in next(iter(DataLoader(ds, batch_size=1))).items()}
    for _ in range(2):
        predict_chunk(pol, kind, one)
    ts = []
    for _ in range(10):
        a = time.time(); predict_chunk(pol, kind, one)
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - a)
    res["latency_ms"] = float(np.median(ts) * 1e3)
    (out_dir / f"{split}{'' if tag == 'last' else '_' + tag}.json").write_text(json.dumps(res, indent=1))
    rollout_plot(run, pol, kind, B, cfg, split, dev, tag=tag)
    plots(run)
    print(json.dumps({"split": split, "val_l1": res["val_l1"], "bins_deg": res["bins_deg"], "latency_ms": res["latency_ms"]}, indent=1))
    return res


def rollout_plot(run, pol, kind, B, cfg, split, dev, joints=ROLLOUT_JOINTS, tag="last"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    names = B["action_names"]
    joints = [j for j in joints if j in names]
    if not joints:                                  # other robots: the first six normalized (unmasked) action columns
        joints = [n for n, m in zip(names, B["stats"]["action"]["mask"]) if not m][:6]
    if not joints:
        return
    eps = sorted({c[0] for c in B["clips"][split]}, key=lambda e: -next(m["frames"] for m in B["meta"]["episodes"] if m["episode"] == e))[:3]
    fps = B["meta"]["fps"]
    for e in eps:
        ds = Clips(B, split, cfg, False)
        ds.c = [c for c in ds.c if c[0] == e and c[3] % fps == 0]
        if not ds.c:
            continue
        r0, nf = ds.c[0][1], ds.c[0][2]
        gt = B["action"][r0:r0 + nf]
        fig, axs = plt.subplots(len(joints), 1, figsize=(9, 1.6 * len(joints)), sharex=True)
        for k in range(len(ds)):
            b = {kk: v.unsqueeze(0).to(dev) for kk, v in ds[k].items()}
            pr = denorm(predict_chunk(pol, kind, b)[0].cpu().numpy(), B["stats"]["action"])
            t = ds.c[k][3]
            n = min(B["chunk"], nf - t)
            for ax, j in zip(axs, joints):
                ax.plot((t + np.arange(n)) / fps, np.degrees(pr[:n, names.index(j)]), color="C1", lw=1, alpha=0.8)
        for ax, j in zip(axs, joints):
            ax.plot(np.arange(nf) / fps, np.degrees(gt[:, names.index(j)]), color="k", lw=1.2)
            ax.set_ylabel(f"{j}\n(deg)", fontsize=7)
            ax.tick_params(labelsize=7)
        axs[0].set_title(f"{run.name} · {split} episode {e}: recorded command (black) vs predicted 4 s chunks every 1 s (orange)", fontsize=8)
        axs[-1].set_xlabel("time (s)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(run / "plots" / f"rollout_ep{e}{'' if tag == 'last' else '_' + tag}.{ext}", dpi=110)
        plt.close(fig)


# ───────────────────────────── plots ─────────────────────────────

def read_metrics(run):
    p = Path(run) / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def plots(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    m = read_metrics(run)
    tr = [r for r in m if r.get("event") == "train"]
    va = [r for r in m if r.get("event") == "val"]
    if not tr:
        return
    s = np.array([r["step"] for r in tr])
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    ax = axs[0, 0]
    loss = np.array([r["loss"] for r in tr])
    ax.plot(s, loss, color="C0", alpha=0.35, lw=0.8, label="train (25-step mean)")
    if len(loss) > 10:
        k = max(1, len(loss) // 40)
        ax.plot(s[k - 1:], np.convolve(loss, np.ones(k) / k, "valid"), color="C0", lw=1.6, label="train (smoothed)")
    if va:
        ax.plot([r["step"] for r in va], [r["val_loss"] for r in va], "o-", color="C3", ms=3, label="val (same objective)")
    ax.set_yscale("log"); ax.set_title("training objective"); ax.set_xlabel("step"); ax.legend(fontsize=7)
    ax = axs[0, 1]
    for key, c in (("l1_loss", "C0"), ("kld_loss", "C2")):
        if key in tr[0]:
            ax.plot(s, [r[key] for r in tr], color=c, lw=0.8, label=key)
    if va:
        ax.plot([r["step"] for r in va], [r["val_l1"] for r in va], "o-", color="C3", ms=3, label="val action L1 (inference, norm.)")
    ax.set_yscale("log"); ax.set_title("components / val action error"); ax.set_xlabel("step"); ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(s, [r["lr"] for r in tr], color="C4"); ax.set_title("learning rate"); ax.set_xlabel("step")
    ax2 = ax.twinx(); ax2.plot(s, [r["grad_norm"] for r in tr], color="C5", lw=0.6, alpha=0.6); ax2.set_ylabel("grad norm", color="C5")
    ax = axs[1, 1]
    if va:
        for g in va[-1]["bins_deg"]:
            ax.plot([r["step"] for r in va], [np.mean(list(r["bins_deg"][g].values())) for r in va], "o-", ms=3, label=g)
        ax.set_title("val error by group (deg, mean over 4 s)"); ax.set_xlabel("step"); ax.legend(fontsize=7)
    fig.suptitle(run.name, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run / "plots" / f"training.{ext}", dpi=110)
    plt.close(fig)
    te = run / "eval" / "test.json"
    if te.exists():
        horizon_plot([run], run / "plots" / "test_horizon")


def horizon_plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tes = [json.loads((Path(run) / "eval" / "test.json").read_text()) for run in runs]
    groups = list(tes[0]["horizon_deg"])
    ncol = min(3, len(groups))
    nrow = int(math.ceil(len(groups) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.8 * nrow), sharex=True, squeeze=False)
    for i, (run, te) in enumerate(zip(runs, tes)):
        for ax, g in zip(axs.flat, groups):
            y = te["horizon_deg"].get(g)
            if y is None:
                continue
            ax.plot(np.arange(len(y)) / te.get("fps", 30), y, color=f"C{i}", label=Path(run).name)
            ax.set_title(g, fontsize=9)
    for ax in axs.flat[len(groups):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("prediction time ahead (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("test MAE (deg)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("open-loop test error vs horizon", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=110)
    plt.close(fig)


def compare(runs, out=None):
    """Both policies on one figure + a markdown table (test split)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [Path(r) for r in runs]
    out = Path(out or RUNS / ("compare_" + "_vs_".join(r.name for r in runs)))
    out.mkdir(parents=True, exist_ok=True)
    horizon_plot(runs, out / "test_horizon")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, r in enumerate(runs):
        va = [x for x in read_metrics(r) if x.get("event") == "val"]
        axs[0].plot([x["step"] for x in va], [x["val_l1"] for x in va], "o-", ms=3, color=f"C{i}", label=r.name)
        te = json.loads((r / "eval" / "test.json").read_text())
        g = list(te["bins_deg"])
        axs[1].bar(np.arange(len(g)) + i * 0.4, [np.mean(list(te["bins_deg"][k].values())) for k in g], 0.4, color=f"C{i}", label=r.name)
        axs[1].set_xticks(np.arange(len(g)) + 0.2, g, rotation=20, fontsize=8)
    axs[0].set_title("val action error (normalized L1) vs step"); axs[0].set_xlabel("step"); axs[0].legend(fontsize=8)
    axs[1].set_title("test MAE per group (deg, mean over 4 s)"); axs[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"summary.{ext}", dpi=110)
    plt.close(fig)
    lines = ["| run | params | steps | wall | test L1 (norm.) | " + " | ".join(f"{a:g}-{b:g} s (deg)" for a, b in HBINS) + " | latency |",
             "|" + "---|" * (6 + len(HBINS))]
    for r in runs:
        te = json.loads((r / "eval" / "test.json").read_text())
        m = read_metrics(r)
        st = next(x for x in m if x.get("event") == "start")
        wall = (m[-1]["time"] - st["time"]) / 3600
        allg = [np.mean([te["bins_deg"][g].get(f"{a:g}-{b:g}s", np.nan) for g in te["bins_deg"]]) for a, b in HBINS]
        lines.append(f"| {r.name} | {st['params'] / 1e6:.1f} M | {te['step']} | {wall:.1f} h | {te['val_l1']:.3f} | "
                     + " | ".join("—" if np.isnan(x) else f"{x:.2f}" for x in allg) + f" | {te['latency_ms']:.0f} ms |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


# ───────────────────────────── deployment runtime ─────────────────────────────

class Runtime:
    """The deployment actor for a run checkpoint — call it once per control step, obs → action. Keeps 1 s of measured joints at the control rate, resizes and
    normalizes the images as in training. Inference runs continuously on a background thread: each chunk is predicted
    from the newest observation, aligned to the control step that observation was taken at (so the steps spent
    inferring are skipped, not replayed), and crossfaded into the plan being executed over `blend` steps. The 30 Hz
    loop never waits on the model except for the very first chunk; it holds the last action if the plan runs out.
    The robot-side driver that feeds it observations and forwards its actions to the gateway is not part of this
    release: wire it into your own controller loop (or a simulator) — every command must still pass the controller's
    safety gate."""

    def __init__(self, path, device="mps", replan_s=0.0, blend=10, noise_seed=None):
        import threading
        import torch
        self.torch = torch
        path = Path(path)
        if (path / "checkpoints").exists():
            path = path / "checkpoints" / "last" / "pretrained_model"   # ACT/DP keep improving in closed loop after val plateaus
        self.policy, self.m = load_policy_dir(path, device)
        self.dev = torch.device(device)
        self.kind = self.m["kind"]
        self.st = {k: {kk: np.array(vv, bool if kk == "mask" else np.float32) for kk, vv in v.items()} for k, v in self.m["stats"].items()}
        self.replan_s, self.blend = replan_s, max(1, blend)
        # DP only: the SAME initial + per-step noise for every chunk, so consecutive chunks pick the same mode instead
        # of re-rolling it (re-rolling at a grasp = open/close dithering). None = fresh noise each chunk (as trained).
        self.noise_seed = noise_seed
        self.joints = self.m.get("robot_joints", self.m["state_names"])   # what the robot reports and takes
        self.merged = len(self.joints) != len(self.m["state_names"])          # one `lift` column in the policy
        self.lock, self.wake = threading.Lock(), threading.Event()
        self.epoch, self.latest, self.ready = 0, None, None
        self.reset()
        threading.Thread(target=self._worker, daemon=True).start()
        print(f"NOVA {self.kind} policy from {path} (step {self.m['step']}) on {device}; continuous {self.m['chunk']}-step "
              f"chunks, crossfade {self.blend} steps" + (f", fixed noise seed {noise_seed}" if noise_seed is not None
                                                         and self.kind != "act" else ""), flush=True)

    def reset(self):
        from collections import deque
        with self.lock:
            self.epoch += 1                                    # chunks still in flight from the last rollout are dropped
            self.latest, self.ready = None, None
        self.buf = deque(maxlen=-self.m["history_offsets"][0] + 1)
        self.plan, self.plan0, self.k = None, 0, 0
        self.n_chunks, self.lag, self.starved = 0, [], 0

    def _batch(self, hist, images):
        import cv2
        torch = self.torch
        b = {"observation.state": torch.from_numpy(hist)[None].to(self.dev)}
        h, w = self.m["image_size"]
        for c in self.m["cameras"]:
            img = cv2.resize(np.asarray(images[c]), (w, h), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            if self.m.get("imagenet_norm", True):
                img = (img - IMNET_MEAN) / IMNET_STD
            img = img.transpose(2, 0, 1)
            b[f"observation.images.{c}"] = torch.from_numpy(np.ascontiguousarray(img))[None].to(self.dev)
        return b

    def _predict(self, hist, images):
        gen = None
        if self.noise_seed is not None and self.kind != "act":
            gen = self.torch.Generator(device=self.dev).manual_seed(int(self.noise_seed))
        return denorm(predict_chunk(self.policy, self.kind, self._batch(hist, images), gen)[0].cpu().numpy(), self.st["action"])

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            with self.lock:
                job, epoch = self.latest, self.epoch
            if job is None:
                continue
            t = time.time()
            chunk = self._predict(job[1], job[2])
            with self.lock:
                if epoch == self.epoch:
                    self.ready = (job[0], chunk)
            if self.replan_s:
                time.sleep(max(0.0, self.replan_s - (time.time() - t)))
            self.wake.set()                                    # straight on to the next observation

    def _merge(self, k0, chunk):
        """Plan from step k on = crossfade(current plan, new chunk) over `blend` steps; the chunk's first k - k0 steps
        belong to observations already past."""
        new = chunk[self.k - k0:]
        if self.plan is None:
            self.plan, self.plan0 = new, self.k
            return
        old = self.plan[min(self.k - self.plan0, len(self.plan) - 1):]
        if len(old) < len(new):
            old = np.concatenate([old, np.repeat(old[-1:], len(new) - len(old), 0)])
        w = np.minimum(1.0, np.arange(1, len(new) + 1) / self.blend)[:, None]
        self.plan, self.plan0 = (1 - w) * old[:len(new)] + w * new, self.k

    def __call__(self, obs, task=None):
        q = np.asarray(obs["observation.state"], np.float32)
        if self.merged:
            q = merge_lift(q, self.joints)[0]
        s = norm(q, self.st["state"])
        self.buf.append(s)
        while len(self.buf) < self.buf.maxlen:
            self.buf.appendleft(self.buf[0])
        n = len(self.buf)
        hist = np.concatenate([self.buf[n - 1 + o] for o in self.m["history_offsets"]])
        images = {c: obs[f"observation.images.{c}"] for c in self.m["cameras"]}
        if self.plan is None and self.ready is None:            # first step of a rollout: wait for one chunk
            self._merge(self.k, self._predict(hist, images))
            self.n_chunks += 1
        with self.lock:
            self.latest = (self.k, hist, images)
            ready, self.ready = self.ready, None
        self.wake.set()
        if ready is not None and self.k - ready[0] < len(ready[1]):
            self.lag.append(self.k - ready[0])
            self._merge(*ready)
            self.n_chunks += 1
        i = self.k - self.plan0
        if i >= len(self.plan):
            self.starved += 1
        a = self.plan[min(i, len(self.plan) - 1)]
        self.k += 1
        if self.merged:                                        # back to the robot's two equal lift stages
            li = self.m["action_names"].index("lift")
            a = np.insert(a, li + 1, a[li] / 2)
            a[li] /= 2
        return a

    def summary(self):
        lag = np.array(self.lag or [0])
        return (f"{self.n_chunks} chunks in {self.k} steps (one per {self.k / max(self.n_chunks, 1):.1f} steps), "
                f"inference lag {lag.mean():.1f} steps mean / {lag.max()} max, {self.starved} steps past the plan")


# ───────────────────────────── CLI ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="decode the export once into the frame cache + per-row table")
    p.add_argument("--dataset", default=DEFAULTS["dataset"], help="name under POLICY_DATASET_ROOT, or a path")
    p.add_argument("--force", action="store_true")
    p.add_argument("--robot", choices=sorted(ROBOTS), default="nova", help="profile for --report")
    p.add_argument("--report", action="store_true", help="also print the cleaning / split report for the defaults")
    p = sub.add_parser("train", help="train one policy (torchrun-compatible)")
    p.add_argument("--policy", choices=["act", "dp", "pi05"], required=True); p.add_argument("--name", required=True)
    p.add_argument("--robot", choices=sorted(ROBOTS), default="nova", help="robot profile (ROBOTS) applied before --set")
    p.add_argument("--config", help="a run's config.json to start from"); p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("eval", help="test-split evaluation of a run"); p.add_argument("--run", required=True); p.add_argument("--split", default="test")
    p.add_argument("--ckpt", choices=["last", "best"], default="last")
    p = sub.add_parser("plots", help="redraw a run's figures"); p.add_argument("--run", required=True)
    p = sub.add_parser("compare", help="figures + markdown table over evaluated runs")
    p.add_argument("runs", nargs="+"); p.add_argument("--out")
    p = sub.add_parser("defaults", help="print the default config")
    a = ap.parse_args()
    if a.cmd == "prepare":
        out = prepare(a.dataset, tuple(DEFAULTS["image_size"]), force=a.force)
        print("cache", out)
        if a.report:
            cfg = deep_update(copy.deepcopy(DEFAULTS), copy.deepcopy(ROBOTS[a.robot]))
            print(json.dumps(build({**cfg, "dataset": a.dataset})["report"], indent=1))
    elif a.cmd == "train":
        if a.resume:
            cfg = json.loads((RUNS / a.name / "config.json").read_text())
            deep_update(cfg, parse_set(a.set))
        else:
            cfg = copy.deepcopy(DEFAULTS)
            deep_update(cfg, copy.deepcopy(ROBOTS[a.robot]))
            if a.config:
                deep_update(cfg, json.loads(Path(a.config).read_text()))
            if a.policy == "pi05":
                deep_update(cfg, copy.deepcopy(PI05_RUN))
            deep_update(cfg, parse_set(a.set))
            cfg["policy"] = a.policy
        train(cfg, a.name, resume=a.resume)
    elif a.cmd == "eval":
        evaluate_run(Path(a.run), a.split, tag=a.ckpt)
    elif a.cmd == "plots":
        plots(Path(a.run))
    elif a.cmd == "compare":
        compare(a.runs, a.out)
    elif a.cmd == "defaults":
        print(json.dumps(DEFAULTS, indent=1))


if __name__ == "__main__":
    main()
