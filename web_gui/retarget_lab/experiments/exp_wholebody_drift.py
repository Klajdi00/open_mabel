#!/usr/bin/env python3
"""
exp_wholebody_drift.py — E6: "the body rotation and the whole re-targeting drift over
time", "the lift is moving so fast (max ~3 cm/s)", "all joint velocities within limits".

Two whole-body takes recorded 2026-10-15 in the lab (Vision Pro, standing operator):
  test2-wholebodywalking  99 s   walking back/forth and sideways, turns to -90 deg
  body_rotations         155 s   repeated in-place body turns up to +-130 deg, walks
replayed frame by frame through KinematicRetargeter("ours") under four conditions
(subprocesses, env-configured):
  before     head+wrist heading, box-gain base law, ideal servo, unbounded lift/torso
             follow (MABEL_LIFT/TORSO_FOLLOW_SPEED=1e3) — the lab before this change
  heading    only the lazy-neck head heading (base law + servo as before)
  after      lazy-neck heading + 1:1 body-frame base + slew servo (MABEL slews) +
             3 cm/s lift + 0.5 rad/s torso — the lab now
  after_nova as `after` with NOVA's slews and base profile (= NOVA's gateway)

Metrics (truth caveats in the doc §E6):
  yaw      |robot base yaw - operator head yaw| (both from their start), over frames
           where the head has been still >= 1 s (a glance is not a body turn)
  return   the same error wherever the operator faces their start direction again
           (head within 10 deg, held 3 s) — the drift the operator saw
  pos      |robot base displacement - operator head displacement| (robot frame, the
           operator's displacement turned into the robot's start frame), p50/p90
  speeds   peak |q_dot| per joint group vs its slew; lift peak speed; share of frames
           a group sits at its slew (demand above what the robot can do)
Outputs results/exp_wholebody_drift.json, docs/figures/rtl_e6_*.pdf. ~6 min.

    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_wholebody_drift.py
"""
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FIG = PKG / "docs" / "figures"
RES = HERE / "results"
TAKES = {"walk": "20261015_170936_test2-wholebodywalking", "rot": "20261015_171134_body_rotations"}
CONDS = {
    "before": {"MABEL_RTL_HEADING": "bodyframe", "MABEL_RTL_BASE_TRANSLATION": "shipped", "MABEL_RTL_SERVO": "ideal",
               "MABEL_LIFT_FOLLOW_SPEED": "1000", "MABEL_TORSO_FOLLOW_SPEED": "1000"},
    "heading": {"MABEL_RTL_HEADING": "head", "MABEL_RTL_BASE_TRANSLATION": "shipped", "MABEL_RTL_SERVO": "ideal",
                "MABEL_LIFT_FOLLOW_SPEED": "1000", "MABEL_TORSO_FOLLOW_SPEED": "1000"},
    "after": {},
    "after_nova": {"MABEL_RTL_ROBOT": "nova"},
}
GROUPS = {"arm_j1-2": ("arm", (1, 2)), "arm_j3-4": ("arm", (3, 4)), "arm_j5-7": ("arm", (5, 6, 7)),
          "torso": ("torso", None), "neck": ("neck", None), "lift": ("lift", None)}


def child(take, out):
    sys.path.insert(0, str(PKG.parent))
    sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from retarget_lab.server import episodes
    from retarget_lab.server import config as CFG
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    from mabel.whole_body_retargeting.heading import head_heading
    _, frames = episodes.load(take)
    kr = KinematicRetargeter("ours")
    names = None
    cols = {k: [] for k in ("t", "hyaw", "hx", "hz", "byaw", "bx", "by", "q", "sat")}
    for f in frames:
        pose = pose_from_frame(f)
        rec = kr.step(pose, f["t"], collide=False)
        if names is None:
            names = sorted(rec["q"])
        H = np.asarray(pose.head, float) if pose is not None and pose.head is not None else None
        hy = head_heading(H) if H is not None else None
        b = rec["base"]
        for k, v in (("t", f["t"]), ("hyaw", np.nan if hy is None else hy), ("hx", H[0, 3] if H is not None else np.nan),
                     ("hz", H[2, 3] if H is not None else np.nan), ("byaw", b["yaw"]), ("bx", b["x"]), ("by", b["y"]),
                     ("sat", rec.get("slew_sat", 0.0))):
            cols[k].append(float(v))
        cols["q"].append([rec["q"][n] for n in names])
    np.savez(out, names=np.array(names), robot=CFG.BASE_PROFILE, **{k: np.asarray(v, float) for k, v in cols.items()})


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def held(mask, t, secs):
    out, start = np.zeros(len(mask), bool), None
    for i, m in enumerate(mask):
        if m:
            start = t[i] if start is None else start
            out[i] = t[i] - start >= secs
        else:
            start = None
    return out


def analyse(d, robot):
    sys.path.insert(0, str(PKG.parent))
    from retarget_lab.server import config as CFG
    t = d["t"]
    hy = d["hyaw"].copy()
    ok = np.isfinite(hy)
    i0 = int(np.argmax(ok))
    # head heading convention -> chassis yaw delta (BASE_YAW_SIGN = +1)
    dh = _wrap(hy - hy[i0]); db = _wrap(d["byaw"] - d["byaw"][i0])
    err = np.degrees(np.abs(_wrap(db - dh)))
    rate = np.zeros_like(hy); rate[1:] = np.abs(_wrap(np.diff(hy))) / np.maximum(np.diff(t), 1e-3)
    still = held(ok & (np.degrees(rate) < 20), t, 1.0)
    back = held(ok & (np.degrees(np.abs(dh)) < 10), t, 3.0)
    # operator displacement in the robot's start frame: ARKit -> robot (x = z, y = x), turned by the
    # operator's start facing (a = heading - pi, heading.py convention)
    a0 = _wrap(hy[i0] - np.pi)
    dx, dy = d["hz"] - d["hz"][i0], d["hx"] - d["hx"][i0]
    ox = math.cos(a0) * dx + math.sin(a0) * dy
    oy = -math.sin(a0) * dx + math.cos(a0) * dy
    perr = np.hypot(d["bx"] - ox, d["by"] - oy)
    names = [str(n) for n in d["names"]]
    q = d["q"]
    dt = np.maximum(np.diff(t), 1e-3)[:, None]
    qd = np.abs(np.diff(q, axis=0)) / dt
    sl = CFG.SLEWS[robot]
    grp = {}
    for g, (key, idx) in GROUPS.items():
        if key == "arm":
            cols = [names.index(f"{s}_arm_{i}") for s in ("left", "right") for i in idx]
            lim = max(sl["arm"][i - 1] for i in idx)
        elif key == "lift":
            cols = [i for i, n in enumerate(names) if n.startswith("lift")]
            lim = sl["lift"]
        elif key == "torso":
            cols, lim = [names.index("torso")], sl["torso"]
        else:
            cols, lim = [i for i, n in enumerate(names) if n.startswith("neck")], sl["neck"]
        v = qd[:, cols].sum(axis=1) if key == "lift" else qd[:, cols].max(axis=1)
        grp[g] = {"peak": float(np.percentile(v, 100)), "p99": float(np.percentile(v, 99)), "limit": float(lim),
                  "at_limit_pct": float(np.mean(v >= 0.98 * lim) * 100)}
    q_ = lambda v, p: float(np.nanpercentile(v, p)) if np.isfinite(v).any() else float("nan")
    fin = np.isfinite(perr)
    return {"yaw_p50": q_(err[still], 50), "yaw_p90": q_(err[still], 90),
            "return_p90": q_(err[back], 90), "return_max": float(np.nanmax(err[back])) if back.any() else float("nan"),
            "pos_p50": q_(perr, 50), "pos_p90": q_(perr, 90), "pos_end": float(perr[fin][-1]) if fin.any() else float("nan"),
            "speeds": grp,
            "_series": {"t": t, "dh": np.degrees(dh), "db": np.degrees(db), "perr": perr, "ox": ox, "bx": d["bx"]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3])
    sys.path.insert(0, str(PKG.parent))
    from retarget_lab.server import config as CFG
    if not all((CFG.EPISODES_DIR / v / "frames.jsonl.gz").exists() for v in TAKES.values()):
        print("takes not present (local data) — nothing to do"); return
    tmp = RES / "wholebody_npz"; tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(c, k) for c in CONDS for k in TAKES]

    def run(job):
        c, k = job
        env = dict(os.environ, MABEL_LOG_LEVEL="WARNING", **CONDS[c])
        subprocess.run([sys.executable, __file__, "--child", TAKES[k], str(tmp / f"{c}_{k}.npz")], env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if "--reuse" not in sys.argv:
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(run, jobs))
    res, series = {}, {}
    for c, k in jobs:
        d = dict(np.load(tmp / f"{c}_{k}.npz"))
        m = analyse(d, str(d["robot"]))
        series[(c, k)] = m.pop("_series")
        res.setdefault(c, {})[k] = m
    RES.mkdir(exist_ok=True)
    (RES / "exp_wholebody_drift.json").write_text(json.dumps(res, indent=1))
    print(f"{'cond':11s} {'take':5s} {'yaw p50/p90 deg':>16s} {'return p90/max':>15s} {'pos p50/p90 m':>14s} {'pos end':>7s}")
    for c in CONDS:
        for k in TAKES:
            m = res[c][k]
            print(f"{c:11s} {k:5s} {m['yaw_p50']:7.1f} {m['yaw_p90']:7.1f}  {m['return_p90']:7.1f} {m['return_max']:6.1f}  "
                  f"{m['pos_p50']:6.2f} {m['pos_p90']:6.2f}  {m['pos_end']:6.2f}")
    print("speeds (peak / limit, % frames at limit):")
    for c in CONDS:
        for k in TAKES:
            sp = res[c][k]["speeds"]
            print(f"  {c:11s} {k:5s} " + "  ".join(f"{g} {v['peak']:.2f}/{v['limit']:.2f} ({v['at_limit_pct']:.0f}%)" for g, v in sp.items()))
    figures(series, res)


def figures(series, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    C = {"before": "#e06c75", "heading": "#e8a33d", "after": "#4c72b0", "after_nova": "#2a9d8f"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for col, k in enumerate(TAKES):
        ax = axes[0, col]
        s = series[("after", k)]
        ax.plot(s["t"], s["dh"], lw=.6, color="#999", label="operator head")
        for c in ("before", "after"):
            s = series[(c, k)]
            ax.plot(s["t"], s["db"], lw=.9, color=C[c], label=f"robot, {c}")
        ax.set_ylabel("yaw from start [deg]"); ax.set_title(k, fontsize=7.5)
        ax.legend(frameon=False, fontsize=6)
        ax = axes[1, col]
        for c in CONDS:
            s = series[(c, k)]
            ax.plot(s["t"], s["perr"], lw=.7, color=C[c], label=c)
        ax.set_ylabel("|robot - operator| position [m]"); ax.set_xlabel("time [s]")
        ax.legend(frameon=False, fontsize=6)
    fig.suptitle("E6 — whole-body drift: heading and position, before vs after", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e6_drift.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    gs = list(GROUPS)
    w = 0.2
    for j, c in enumerate(CONDS):
        r = [res[c]["walk"]["speeds"][g]["peak"] / res[c]["walk"]["speeds"][g]["limit"] for g in gs]
        ax.bar(np.arange(len(gs)) + (j - 1.5) * w, r, w, color=C[c], label=c)
    ax.axhline(1.0, ls="--", lw=.7, color="#333")
    ax.set_yscale("log"); ax.set_xticks(range(len(gs))); ax.set_xticklabels(gs)
    ax.set_ylabel("peak speed / robot slew"); ax.legend(frameon=False, fontsize=6, ncol=4)
    ax.set_title("E6 — joint speeds against the robot's slews (walking take)", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e6_speeds.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
