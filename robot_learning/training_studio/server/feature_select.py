"""
Modular input/output selection (the user's "pick 2-4 cams + a body module" ask).

A *selection* = a camera subset + a robot-module scope (whole/upper/arms/hands/
custom groups). ``resolve`` turns it into index maps + a scatter map; ``build_view``
materializes a derived LeRobot v2.1 dataset whose observation.state/action columns
are sliced to the selected joints and whose videos are only the chosen cameras —
so LeRobot trains natively on real, correctly-shaped features (no in-loop hacks).

Generalizes to any MABEL embodiment because everything composes the embodiment
group table (the single source of truth).
"""
import hashlib
import json
import os
import shutil

import numpy as np

from . import embodiment as EMB
from . import lerobot_dataset as L


def resolve(scope="whole_body", groups=None, cameras=None,
            action_scope=None, action_groups=None):
    """Return the selection descriptor for a camera subset + an OBSERVATION scope
    and an (independent) ACTION/output scope. The action scope defaults to the obs
    scope, so by default the policy predicts what it observes — but you can e.g.
    feed whole-body proprio yet output only the arm controls."""
    gi = EMB.group_indices()
    names = EMB.actuator_names()
    obs_groups = EMB.resolve_scope(scope, groups)
    state_indices = sorted(set(i for g in obs_groups for i in gi[g]))
    # action scope: explicit if given, else mirror the observation scope
    if action_scope or action_groups:
        act_groups = EMB.resolve_scope(action_scope or scope, action_groups)
    else:
        act_groups, action_scope = obs_groups, scope
    action_indices = sorted(set(i for g in act_groups for i in gi[g]))
    all_cams = [c["name"] for c in EMB.cameras()]
    cams = [c for c in (cameras or all_cams) if c in all_cams]
    if not cams:
        cams = all_cams
    return {
        "scope": scope,
        "groups": obs_groups,
        "action_scope": action_scope or scope,
        "action_groups": act_groups,
        "cameras": cams,
        "camera_keys": [f"observation.images.{c}" for c in cams],
        "state_indices": state_indices,
        "action_indices": action_indices,
        "obs_state_dim": len(state_indices),
        "action_dim": len(action_indices),
        "joint_names": [names[i] for i in state_indices],
        "action_names": [names[i] for i in action_indices],
        # scatter_map[j] = full-DoF ctrl index for the policy's j-th OUTPUT
        "scatter_map": list(action_indices),
        "full_dim": EMB.dof(),
    }


def selection_key(src_name, sel):
    """Stable short hash of (source, selection) for caching the view dir."""
    payload = json.dumps({"src": src_name, "scope": sel["scope"],
                          "cams": sel["cameras"], "idx": sel["state_indices"]},
                         sort_keys=True)
    h = hashlib.sha1(payload.encode()).hexdigest()[:8]
    cam_tag = "all" if len(sel["cameras"]) == EMB.cameras().__len__() else f"{len(sel['cameras'])}cam"
    return f"{src_name}__{sel['scope']}_{cam_tag}_{h}"


def is_full_selection(src_root, sel):
    """True when the selection keeps every feature — no view needed, train the source."""
    info = L.read_info(src_root)
    src_cams = set(L.camera_names(info))
    full = info.get("features", {}).get("action", {}).get("shape", [0])[0]
    return set(sel["cameras"]) == src_cams and sel["action_dim"] == full


def _slice_list_column(tbl, name, idx):
    import pyarrow as pa
    arr = np.asarray(tbl.column(name).to_pylist(), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(len(arr), -1)
    sliced = arr[:, idx]
    return pa.array([row.tolist() for row in sliced])


def build_view(src_root, out_root, sel, progress=None):
    """Materialize a sliced/subset LeRobot dataset at ``out_root``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = L.read_info(src_root)
    n_eps = int(info.get("total_episodes", 1))
    sidx, aidx = sel["state_indices"], sel["action_indices"]
    keep_cam_keys = set(sel["camera_keys"])

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)

    # ── data: slice observation.state / action per episode ────────────────────
    for e in range(n_eps):
        src_pq = L.parquet_path(src_root, e, info)
        if not os.path.exists(src_pq):
            continue
        tbl = pq.read_table(src_pq)
        cols, colnames = [], []
        for name in tbl.column_names:
            if name == "observation.state":
                cols.append(_slice_list_column(tbl, name, sidx))
            elif name == "action":
                cols.append(_slice_list_column(tbl, name, aidx))
            else:
                cols.append(tbl.column(name))
            colnames.append(name)
        c = L.chunk_of(e, info)
        dst_dir = os.path.join(out_root, "data", f"chunk-{c:03d}")
        os.makedirs(dst_dir, exist_ok=True)
        pq.write_table(pa.table(cols, names=colnames),
                       os.path.join(dst_dir, f"episode_{e:06d}.parquet"))
        if progress:
            progress(e + 1, n_eps, stage=1)

    # ── videos: symlink only the selected cameras ─────────────────────────────
    for e in range(n_eps):
        c = L.chunk_of(e, info)
        for key in keep_cam_keys:
            src_mp4 = L.video_path(src_root, key, e, info)
            if not os.path.exists(src_mp4):
                continue
            dst_dir = os.path.join(out_root, "videos", f"chunk-{c:03d}", key)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"episode_{e:06d}.mp4")
            try:
                os.symlink(os.path.abspath(src_mp4), dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src_mp4, dst)

    # ── meta/info.json: rewrite shapes + drop unselected cameras ──────────────
    feats = {}
    for k, v in info["features"].items():
        if k.startswith("observation.images.") and k not in keep_cam_keys:
            continue
        v = json.loads(json.dumps(v))
        if k == "observation.state":
            v["shape"] = [len(sidx)]
        elif k == "action":
            v["shape"] = [len(aidx)]
        feats[k] = v
    new_info = json.loads(json.dumps(info))
    new_info["features"] = feats
    new_info["total_videos"] = len(keep_cam_keys) * n_eps
    json.dump(new_info, open(os.path.join(out_root, "meta", "info.json"), "w"), indent=2)

    # ── meta: copy tasks/episodes; slice stats ────────────────────────────────
    for fn in ("tasks.jsonl", "episodes.jsonl"):
        src = os.path.join(src_root, "meta", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_root, "meta", fn))

    stats = L.read_jsonl(src_root, "episodes_stats.jsonl")
    with open(os.path.join(out_root, "meta", "episodes_stats.jsonl"), "w") as fh:
        for rec in stats:
            st = rec.get("stats", {})
            for key, jidx in (("observation.state", sidx), ("action", aidx)):
                if key in st:
                    for m in ("min", "max", "mean", "std"):
                        if m in st[key] and isinstance(st[key][m], list) \
                                and len(st[key][m]) == sel["full_dim"]:
                            st[key][m] = [st[key][m][i] for i in jidx]
            # drop unselected camera stats if any
            rec["stats"] = {k: v for k, v in st.items()
                            if not k.startswith("observation.images.") or k in keep_cam_keys}
            fh.write(json.dumps(rec) + "\n")

    return {"path": out_root, "n_episodes": n_eps,
            "obs_state_dim": len(sidx), "action_dim": len(aidx),
            "cameras": sel["cameras"]}
