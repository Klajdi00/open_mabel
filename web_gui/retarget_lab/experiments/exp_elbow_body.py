#!/usr/bin/env python3
"""
exp_elbow_body.py — E9: straight-elbow lock and upper-arm-into-body on the operator's takes.

Operator report (2026-10-16, NOVA): "many times the elbow got straightened and the robot got
stuck in the singularity, and the upper arm was trying to hit the robot body many times".

Replays each recorded Vision Pro take through
  deployed  NOVA's arm retargeter EXACTLY as the gateway runs it (NOVA/controller/arm_retarget.py
            ArmRetargeter: MabelTeleop -> TeleopEngine, fail-safe + clutch on, 4 rad/s glide,
            self-collision on, NOVA's env knobs), on an ideal plant (measured := last command),
            on a simulated clock at the frames' capture stamps;
  deployed_prof  the same with NOVA's saved operator reach profile applied (profiles/operator.json);
  ours      the lab's "Ours" (KinematicRetargeter, upper-body scope, ideal servo);
and measures per arm, per frame, on the robot's FK:
  flex      elbow flexion = angle between shoulder->elbow and elbow->wrist (0 deg = straight arm;
            the elbow joint's stop is AT straight, so flex ~ 0 is a joint limit AND the
            singularity where wrist-radial motion has no joint to come from);
  upper     clearance of the upper-arm SEGMENT (its minimum is usually the fixed shoulder end, so read
            its tail — % of frames under 20 mm — not its median) (shoulder->elbow capsule, the guard's elbow radius)
            to the guard's own fitted torso capsule — the engine guards only an elbow SPHERE;
  elbow     clearance of that elbow sphere (what the guard sees);
  fore      forearm capsule to torso.
Negative clearance = the capsules overlap.

    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_elbow_body.py
Outputs results/exp_elbow_body.json, docs/figures/rtl_e9_*.pdf.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[1]
NOVA = Path(os.environ.get("MABEL_RTL_NOVA", str(REPO.parent / "NOVA")))
FIG = PKG / "docs" / "figures"
RES = HERE / "results"
TAKES = {"test1": "20261015_103700_Oct_15th_test1", "walk": "20261015_170936_test2-wholebodywalking",
         "rot": "20261015_171134_body_rotations"}
# (retargeter, MABEL_ARM_ELBOW_ABDUCT [m]) — the standoff sweep; 0.0 = before the fix
CONFIGS = [("deployed", 0.0), ("deployed", 0.05), ("deployed", 0.10), ("deployed", 0.15), ("deployed", 0.25),
           ("ours", 0.0), ("ours", 0.15)]
STRAIGHT_DEG = 10.0


def _paths():
    for p in (str(REPO / "web_gui"), str(REPO / "controller"), str(NOVA / "controller"), str(NOVA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("NOVA_MABEL_ROOT", str(REPO))


class Meter:
    """FK metrics on a (model, data) of MABEL's MJCF, with the engine's own capsule guard."""
    def __init__(self, model, data):
        import mujoco
        from mabel import config as C
        from mabel.whole_body_controller.collision_avoidance.capsule_guard import SelfCollisionGuard, _seg_seg
        self.mj, self.m, self.d = mujoco, model, data
        self.g = SelfCollisionGuard(model, mujoco.MjData(model))
        self.g.data = data
        self.seg = _seg_seg
        self.r = C.SC_RADII
        self.b = {s: {k: model.body(f"{s[0]}_{k}").id for k in ("shoulder_2", "elbow", "wrist")} for s in ("left", "right")}

    def measure(self):
        d, out = self.d, {}
        tp = self.g.caps["torso"].world(d)
        rt = self.r["torso"]
        for s in ("left", "right"):
            S, E, W = (d.xpos[self.b[s][k]].copy() for k in ("shoulder_2", "elbow", "wrist"))
            u, v = E - S, W - E
            c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            flex = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            def dist(a0, a1):
                pa, pb = self.seg(a0, a1, tp[0], tp[1])
                return float(np.linalg.norm(pa - pb))
            out[s] = dict(flex=flex,
                          upper=float(dist(S, E) - self.r["upper"] - rt),
                          elbow=float(dist(E, E) - self.r["upper"] - rt),
                          fore=float(dist(E, W) - self.r["forearm"] - rt))
        return out


def child(cfg, take, out, abduct):
    os.environ["MABEL_ARM_ELBOW_ABDUCT"] = str(abduct)   # read at engine import
    _paths()
    import time as _time
    from retarget_lab.server import episodes
    _, frames = episodes.load(take)
    frames = [f for f in frames if f.get("kind") == "wire"]
    log = {f"{s}_{k}": [] for s in ("left", "right") for k in ("flex", "upper", "elbow", "fore")}
    log["t"] = []
    log["track_mm"] = []
    if cfg.startswith("deployed"):
        clock = [frames[0]["t"]]
        _time.time = lambda: clock[0]                      # the engine + glide run on the take's clock
        _time.monotonic = lambda: clock[0]
        import arm_retarget as AR
        ar = AR.ArmRetargeter(log=lambda s: None)
        meter = Meter(ar.model, ar.data)
        ar.set_mode({"method": "retarget", "scope": "local"})
        if cfg == "deployed_prof":                          # NOVA's saved reach profile (profiles/operator.json)
            ar.body_calibration("apply", "operator")
        meas = {u: float(ar.data.qpos[adr]) for u, (_, adr) in ar._act.items()}
        tgt = {}
        _solve = ar.engine._solve_side
        def _spy(side, payload, contact):                    # the wrist target this tick (world)
            if contact is not None:
                tgt[side] = np.asarray(contact, float).copy()
            return _solve(side, payload, contact)
        ar.engine._solve_side = _spy
        wj = {s: ar.engine.arm[s].orca_wrist_jid for s in ("left", "right")}
        engaged_t = None
        for f in frames:
            clock[0] = f["t"]
            ar.feed(f["p"])
            out_q = ar.tick(meas)
            if out_q:
                meas.update(out_q)
                engaged_t = engaged_t or f["t"]
            for u, (_, adr) in ar._act.items():               # FK at the commanded (= ideal-plant) pose
                ar.data.qpos[adr] = meas[u]
            ar.mujoco.mj_kinematics(ar.model, ar.data)
            if engaged_t is not None and f["t"] - engaged_t > 1.5:
                _push(log, f["t"], meter.measure())
                e = [np.linalg.norm(ar.data.xanchor[wj[s]] - tgt[s]) for s in tgt]
                log["track_mm"].append(1e3 * float(np.mean(e)) if e else np.nan)
    else:
        from retarget_lab.server.worker import make_retargeter, pose_from_frame
        kr = make_retargeter(cfg)
        kr.apply_settings({"scope": "local", "robot": "nova", "servo": "ideal"})
        meter = Meter(kr.model, kr.data)
        for f in frames:
            rec = kr.step(pose_from_frame(f), f["t"], collide=False, frame=f)
            if rec.get("anchored"):
                kr.mj.mj_kinematics(kr.model, kr.data)
                _push(log, f["t"], meter.measure())
                m_ = rec.get("m", {})
                log["track_mm"].append(float(np.nanmean([m_.get("track_mm_l", np.nan), m_.get("track_mm_r", np.nan)])))
    np.savez(out, **{k: np.asarray(v, float) for k, v in log.items()})


def _push(log, t, m):
    log["t"].append(t)
    for s, d in m.items():
        for k, v in d.items():
            log[f"{s}_{k}"].append(v)


def runs(L):
    """Longest contiguous stretch (s) with flex < STRAIGHT_DEG."""
    t = L["t"]; best = 0.0
    for s in ("left", "right"):
        on = L[f"{s}_flex"] < STRAIGHT_DEG; t0 = None
        for i, x in enumerate(on):
            if x and t0 is None:
                t0 = t[i]
            if (not x or i == len(on) - 1) and t0 is not None:
                best = max(best, t[i] - t0); t0 = None
    return best


def main():
    tmp = Path(tempfile.mkdtemp())
    res, data = {}, {}
    jobs = [((c, a), k) for c, a in CONFIGS for k in TAKES]
    procs = []
    for (c, a), k in jobs:                                    # 6 at a time
        o = tmp / f"{c}_{a}_{k}.npz"
        procs.append(((c, a), k, o, subprocess.Popen([sys.executable, __file__, "--child", c, TAKES[k], str(o), str(a)],
                                                      stdout=subprocess.DEVNULL)))
        while sum(p.poll() is None for *_, p in procs) >= 6:
            procs[[p.poll() is None for *_, p in procs].index(True)][3].wait()
    for *_, p in procs:
        assert p.wait() == 0
    for (c, a), k, o, _ in procs:
            cfg = f"{c}@{a:g}"
            L = dict(np.load(o)); data[(cfg, k)] = L
            fl = np.r_[L["left_flex"], L["right_flex"]]
            up = np.r_[L["left_upper"], L["right_upper"]]
            el = np.r_[L["left_elbow"], L["right_elbow"]]
            fo = np.r_[L["left_fore"], L["right_fore"]]
            res[f"{cfg}/{k}"] = dict(
                frames=int(len(L["t"])),
                straight_pct=round(100 * float(np.mean(fl < STRAIGHT_DEG)), 1),
                longest_straight_s=round(runs(L), 2),
                flex_p10_deg=round(float(np.percentile(fl, 10)), 1),
                upper_overlap_pct=round(100 * float(np.mean(up < 0)), 1),
                upper_min_mm=round(1e3 * float(up.min()), 1),
                elbow_min_mm=round(1e3 * float(el.min()), 1),
                fore_overlap_pct=round(100 * float(np.mean(fo < 0)), 1),
                elbow_p10_mm=round(1e3 * float(np.percentile(el, 10)), 1),
                elbow_p50_mm=round(1e3 * float(np.percentile(el, 50)), 1),
                upper_lt20_pct=round(100 * float(np.mean(up < 0.02)), 1),
                upper_p10_mm=round(1e3 * float(np.percentile(up, 10)), 1),
                upper_p50_mm=round(1e3 * float(np.percentile(up, 50)), 1),
                track_p50_mm=round(float(np.nanpercentile(L["track_mm"], 50)), 1),
                track_p90_mm=round(float(np.nanpercentile(L["track_mm"], 90)), 1))
            print(cfg, k, res[f"{cfg}/{k}"], flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "exp_elbow_body.json").write_text(json.dumps(res, indent=1))
    figures(data, res)


def figures(data, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    sweep = sorted(a for c, a in CONFIGS if c == "deployed")
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(TAKES), figsize=(7.0, 2.4), sharey=True)
    for j, k in enumerate(TAKES):
        for i, a in enumerate(sweep):
            L = data.get((f"deployed@{a:g}", k))
            up = np.sort(1e3 * np.r_[L["left_elbow"], L["right_elbow"]])
            axes[j].plot(up, np.linspace(0, 1, len(up)), color=cmap(i / max(1, len(sweep) - 1)), lw=1,
                         label=f"standoff {100 * a:g} cm")
        axes[j].axvline(10, color="k", lw=.5, ls=":")
        axes[j].set_title(k); axes[j].set_xlabel("elbow to torso clearance [mm]")
    axes[0].set_ylabel("CDF"); axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("E9 — elbow clearance from the torso, NOVA's deployed retargeter, elbow-hint standoff sweep (dotted: 10 mm guard margin)", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e9_clearance.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for k, mk in zip(TAKES, "osd"):
        ax[0].plot([100 * a for a in sweep], [res[f"deployed@{a:g}/{k}"]["elbow_p50_mm"] for a in sweep], marker=mk, label=k)
        ax[1].plot([100 * a for a in sweep], [res[f"deployed@{a:g}/{k}"]["track_p50_mm"] for a in sweep], marker=mk, label=k)
    ax[0].set_xlabel("elbow-hint standoff [cm]"); ax[0].set_ylabel("elbow to torso p50 [mm]")
    ax[1].set_xlabel("elbow-hint standoff [cm]"); ax[1].set_ylabel("wrist vs own target p50 [mm]")
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e9_tradeoff.pdf"); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:5], float(sys.argv[5]) if len(sys.argv) > 5 else 0.06)
    else:
        main()
