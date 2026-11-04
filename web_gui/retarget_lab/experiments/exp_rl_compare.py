#!/usr/bin/env python3
"""
exp_rl_compare.py — E8: NOVA's learned whole-body policy (controller/wbc_rl, v25) vs the lab's
engine ("ours") on the operator's own Vision Pro takes.

Every (variant, take, scope) cell replays the recorded wire payloads frame by frame through a
FRESH retargeter (make_retargeter: the TeleopEngine, or rl_variant.RLRetargeter = NOVA's deploy
adapter + policy.npz), on the lab's plant (physical slew servo, NOVA's slews, swerve-module base),
in a subprocess. Measured per cell:
  shape/placement  the lab's metrics (cos_reach, place_pct, orient_deg, cos_upper/fore), medians
                   over settled frames (the engine's pre-anchor frames and the policy's 1 s
                   engage glide excluded alike)
  own tracking     each map against its OWN target: own-target error (mm) and, for the policy,
                   its own palm-frame error (its convention), to separate "tracks badly" from
                   "the lab's palm metric defines the palm differently"
  whole body       heading and position against the operator (E6's definitions)
  feasibility      peak arm joint speed (the plant's), base path
  cost             solve ms per frame
Plus a control: the policy on NOVA's own sim2sim clips through the SAME lab plant vs NOVA's
sim2sim numbers (does the integration reproduce NOVA's own result?).
Outputs results/exp_rl_compare.json, docs/figures/rtl_e8_*.pdf. ~8 min.

    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_rl_compare.py
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
TAKES = {"test1": "20261015_103700_Oct_15th_test1", "walk": "20261015_170936_test2-wholebodywalking",
         "rot": "20261015_171134_body_rotations"}
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from retarget_lab.server.variants import RL_VARIANTS, VARIANTS as _V  # noqa: E402
VARS = ["ours"] + RL_VARIANTS                       # every exported NOVA policy
SCOPES = ["global", "local"]
MKEYS = ["cos_reach", "cos_upper", "cos_fore", "place_pct", "orient_deg", "track_mm"]


def child(vid, take, scope, out):
    sys.path.insert(0, str(PKG.parent)); sys.path.insert(0, str(PKG.parent.parent / "controller"))
    import time
    from retarget_lab.server import episodes
    from retarget_lab.server.worker import make_retargeter, pose_from_frame
    from mabel.whole_body_retargeting.heading import head_heading
    _, frames = episodes.load(take)
    kr = make_retargeter(vid)
    kr.apply_settings({"scope": scope, "robot": "nova"})
    names = None
    C = {k: [] for k in ("t", "anch", "hyaw", "hx", "hz", "byaw", "bx", "by", "q", "solve", "own_or")}
    C.update({f"{k}_{s}": [] for k in MKEYS for s in "lr"})
    for f in frames:
        pose = pose_from_frame(f)
        t0 = time.perf_counter()
        rec = kr.step(pose, f["t"], collide=False, frame=f)
        C["solve"].append((time.perf_counter() - t0) * 1e3)
        names = names or sorted(rec["q"])
        H = np.asarray(pose.head, float) if pose is not None and pose.head is not None else None
        hy = head_heading(H) if H is not None else None
        b = rec["base"]
        for k, v in (("t", f["t"]), ("anch", rec["anchored"]), ("hyaw", np.nan if hy is None else hy),
                     ("hx", H[0, 3] if H is not None else np.nan), ("hz", H[2, 3] if H is not None else np.nan),
                     ("byaw", b["yaw"]), ("bx", b["x"]), ("by", b["y"])):
            C[k].append(float(v))
        for k in MKEYS:
            for s in "lr":
                v = rec["m"].get(f"{k}_{s}")
                C[f"{k}_{s}"].append(np.nan if v is None else float(v))
        oo = (rec.get("rl") or {}).get("own_orient_deg") or {}
        C["own_or"].append(float(np.nanmean([oo.get("left", np.nan), oo.get("right", np.nan)])) if oo else np.nan)
        C["q"].append([rec["q"][n] for n in names])
    np.savez(out, names=np.array(names), **{k: np.asarray(v, float) for k, v in C.items()})


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


def analyse(d):
    t = d["t"]; a = d["anch"] > 0.5
    med = lambda k: float(np.nanmedian(np.r_[d[k + "_l"][a], d[k + "_r"][a]])) if a.any() else float("nan")
    out = {k: med(k) for k in MKEYS}
    out["own_orient_deg"] = float(np.nanmedian(d["own_or"][a])) if np.isfinite(d["own_or"][a]).any() else float("nan")
    hy = d["hyaw"]; ok = np.isfinite(hy); i0 = int(np.argmax(ok))
    dh = _wrap(hy - hy[i0]); db = _wrap(d["byaw"] - d["byaw"][i0])
    err = np.degrees(np.abs(_wrap(db - dh)))
    rate = np.zeros_like(hy); rate[1:] = np.abs(_wrap(np.diff(hy))) / np.maximum(np.diff(t), 1e-3)
    still = held(ok & (np.degrees(rate) < 20), t, 1.0)
    back = held(ok & (np.degrees(np.abs(dh)) < 10), t, 3.0)
    a0 = _wrap(hy[i0] - np.pi)
    dx, dy = d["hz"] - d["hz"][i0], d["hx"] - d["hx"][i0]
    ox = math.cos(a0) * dx + math.sin(a0) * dy; oy = -math.sin(a0) * dx + math.cos(a0) * dy
    perr = np.hypot(d["bx"] - ox, d["by"] - oy)
    q_ = lambda v, p: float(np.nanpercentile(v, p)) if np.isfinite(v).any() else float("nan")
    out.update(yaw_p90=q_(err[still], 90), return_p90=q_(err[back], 90), pos_p50=q_(perr, 50), pos_p90=q_(perr, 90),
               base_path=float(np.nansum(np.hypot(np.diff(d["bx"]), np.diff(d["by"])))))
    names = [str(n) for n in d["names"]]
    arm = [i for i, n in enumerate(names) if "_arm_" in n]
    qd = np.abs(np.diff(d["q"][:, arm], axis=0)) / np.maximum(np.diff(t), 1e-3)[:, None]
    out["arm_speed_p99"] = q_(qd.max(1), 99)
    out["solve_ms_p50"] = float(np.median(d["solve"]))
    out["_series"] = {"t": t, "dh": np.degrees(dh), "db": np.degrees(db), "perr": perr,
                      "place": np.where(a, np.nanmean(np.c_[d["place_pct_l"], d["place_pct_r"]], 1), np.nan)}
    return out


def control_sim2sim(vid):
    """Policy `vid` on NOVA's own clips through the lab plant (the variant's own tick, so its wrist
    solver rule applies) AND through NOVA's sim2sim (MuJoCo, NOVA's chain), same clips, no added latency."""
    sys.path.insert(0, str(PKG.parent)); sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from retarget_lab.server.worker import make_retargeter
    from retarget_lab.server import rl_variant as RV
    kr = make_retargeter(vid)
    D, R = kr.D, kr.R
    sys.path.insert(0, str(RV.WBC_RL))
    import sim2sim as S2
    z = np.load(RV.WBC_RL / "checkpoints" / "clips.npz")
    rows = []
    for clip in range(len(z["names"])):
        ref = S2.run(str(kr.policy_path), clip, "whole", 0, False, kr.wrist_ik)
        kr.reset(); kr.apply_settings({"robot": "nova", "scope": "global"})
        op_all, arm_k = S2.load_clip(z, clip)
        kr.ad = D.Adapter(kr._rl_policy, upper=False, arm_k=arm_k)
        fr = lambda t: {k: v[t] for k, v in op_all.items()}
        kr._op = fr(0); kr.ad.engage(kr._op, kr._q20(), kr._odom())
        qprev = kr._q20(); eh, ew = [], []
        for t in range(len(op_all["head_p"])):
            q = kr._q20(); qd = (q - qprev) / D.DT; qprev = q
            kr._op = fr(t); tg = kr.ad.targets(kr._op, step=False)
            cmd = kr._tick(q, qd, np.zeros(3))
            for n, v in cmd["joints"].items():
                if n in kr._act:
                    kr.data.ctrl[kr._act[n]] = v
            kr.data.ctrl[kr._lift_act] = 2 * (cmd["lift"] + RV.LIFT_OFF)
            vx, vy, w = cmd["twist"]; kr._apply_base((-vx, -vy, w), D.DT)
            kr._servo_ideal(D.DT); kr.mj.mj_kinematics(kr.model, kr.data)
            if t > 1.5 / D.DT:
                cp, _ = kr.ad.tracked(kr._q20()); x, y, yw = kr._odom(); Rb = D.rot_z(yw); o = np.array([x, y, R.CHASSIS_Z])
                cur = [Rb @ c + o for c in cp]
                eh.append(np.linalg.norm(cur[0] - tg["head_p"]))
                ew += [np.linalg.norm(cur[1] - tg["lw_p"]), np.linalg.norm(cur[2] - tg["rw_p"])]
        rows.append({"policy": vid, "clip": str(z["names"][clip]).split(",")[0],
                     "head_cm": 100 * float(np.mean(eh)), "wrist_cm": 100 * float(np.mean(ew)),
                     "s2s_head_cm": ref["head_cm"], "s2s_wrist_cm": ref["wrist_cm"],
                     "s2s_palm_deg": float(np.mean(ref["wrist_deg_l_r"])), "s2s_fell": ref["fell"]})
        print(rows[-1], flush=True)
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(*sys.argv[2:6])
    sys.path.insert(0, str(PKG.parent))
    from retarget_lab.server import config as CFG
    tmp = RES / "rl_npz"; tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(v, k, sc) for v in VARS for k in TAKES for sc in SCOPES
            if (CFG.EPISODES_DIR / TAKES[k] / "frames.jsonl.gz").exists()]

    def run(job):
        v, k, sc = job
        env = dict(os.environ, MABEL_LOG_LEVEL="WARNING", OPENBLAS_NUM_THREADS="1")
        subprocess.run([sys.executable, __file__, "--child", v, TAKES[k], sc, str(tmp / f"{v}_{k}_{sc}.npz")],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, jobs))
    res, series = {"cells": {}}, {}
    for v, k, sc in jobs:
        a = analyse(dict(np.load(tmp / f"{v}_{k}_{sc}.npz")))
        series[(v, k, sc)] = a.pop("_series")
        res["cells"][f"{v}|{k}|{sc}"] = a
    res["control"] = [r for v in VARS if v != "ours" for r in control_sim2sim(v)]
    RES.mkdir(exist_ok=True)
    (RES / "exp_rl_compare.json").write_text(json.dumps(res, indent=1))
    hdr = f"{'variant':10s} {'take':5s} {'scope':6s} {'cos_reach':>9s} {'cos_up':>6s} {'place%':>6s} {'orient':>6s} {'own mm':>6s} {'own or':>6s} {'yaw p90':>7s} {'ret p90':>7s} {'pos p90':>7s} {'arm v99':>7s} {'ms':>5s}"
    print(hdr)
    for key, c in res["cells"].items():
        v, k, sc = key.split("|")
        print(f"{v:10s} {k:5s} {sc:6s} {c['cos_reach']:9.3f} {c['cos_upper']:6.3f} {c['place_pct']:6.1f} {c['orient_deg']:6.1f} "
              f"{c['track_mm']:6.0f} {c['own_orient_deg']:6.1f} {c['yaw_p90']:7.1f} {c['return_p90']:7.1f} {c['pos_p90']:7.2f} "
              f"{c['arm_speed_p99']:7.2f} {c['solve_ms_p50']:5.1f}")
    print("control — the policy on NOVA's sim2sim clips through the lab plant vs NOVA's sim2sim (head / wrist cm):")
    for r in res["control"]:
        print(f"  {r['policy']:7s} {r['clip'][:40]:40s} lab {r['head_cm']:5.1f} / {r['wrist_cm']:5.1f}   "
              f"sim2sim {r['s2s_head_cm']:5.1f} / {r['s2s_wrist_cm']:5.1f}  palm {r['s2s_palm_deg']:4.1f} deg  fell {r['s2s_fell']}")
    figures(series, res, jobs)


def figures(series, res, jobs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    C = {v: _V[v]["color"] for v in VARS}
    takes = [k for k in TAKES if any(j[1] == k for j in jobs)]
    fig, axes = plt.subplots(2, len(takes), figsize=(7.4, 4.2), squeeze=False)
    for j, k in enumerate(takes):
        ax = axes[0, j]
        s = series[("ours", k, "global")]
        ax.plot(s["t"], s["dh"], lw=.5, color="#999", label="operator head")
        for v in VARS:
            s = series[(v, k, "global")]
            ax.plot(s["t"], s["db"], lw=.8, color=C[v], label=v)
        ax.set_title(k, fontsize=7.5); ax.set_ylabel("yaw [deg]") if j == 0 else None
        if j == 0: ax.legend(frameon=False, fontsize=5.5)
        ax = axes[1, j]
        for v in VARS:
            s = series[(v, k, "global")]
            ax.plot(s["t"], s["place"], lw=.6, color=C[v])
        ax.set_xlabel("time [s]"); ax.set_ylabel("placement [% reach]") if j == 0 else None
    fig.suptitle("E8 — NOVA's RL policies vs the lab engine on the operator's takes (whole body)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e8_rl_takes.pdf"); plt.close(fig)
    rows = res["control"]; pols = [v for v in VARS if v != "ours"]
    fig, axs = plt.subplots(1, len(pols), figsize=(7.4, 2.6), sharey=True, squeeze=False)
    for ax, v in zip(axs[0], pols):
        rr = [r for r in rows if r["policy"] == v]; x = np.arange(len(rr))
        ax.bar(x - 0.2, [r["wrist_cm"] for r in rr], 0.4, color=C[v], label="lab plant (kinematic)")
        ax.bar(x + 0.2, [r["s2s_wrist_cm"] for r in rr], 0.4, color="#666", label="NOVA sim2sim (MuJoCo)")
        ax.set_xticks(x); ax.set_xticklabels([r["clip"].split(":")[1][:10] for r in rr], rotation=35, ha="right", fontsize=6)
        ax.set_title(_V[v]["short"], fontsize=7.5)
    axs[0, 0].set_ylabel("wrist error [cm]"); axs[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle("E8 control: same policy, same clips, two plants", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e8_control.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
