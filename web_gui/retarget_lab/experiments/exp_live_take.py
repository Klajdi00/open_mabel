#!/usr/bin/env python3
"""
exp_live_take.py — E5: what went wrong on the first live Vision Pro session
(2026-10-15, "Oct.15th test1", 69.2 s, 4160 frames) and what each fix buys.

The operator reported three things from inside the headset:
  (a) "the frames are not moving relative to the world" — whole-body motion
      (walking, turning) never showed;
  (b) "the re-targeting will lose track and sometimes freeze";
  (c) "my torso is not bending forward a lot when I am bending forward by a lot".

Conditions (every one replays the SAME recorded wire payloads through
KinematicRetargeter("ours" unless noted), every frame, ideal servo):
  fixed        the lab as shipped now: GLOBAL base (integrated), angle lean,
               clutch off, headset capture clock
  arrival      as `fixed`, but frames clocked by ARRIVAL time (the old live path)
  clutch       as `fixed`, but the engine's double-pinch clutch on (the old lab)
  box          as `fixed`, but the shipped box-clipped torso law
  local        as `fixed`, but LOCAL scope (base held — the old lab)
  fast         variant ours_fast (proximal trust region 6 rad/s, not 2)
  holonomic    as `fixed`, but the base twist realised EXACTLY (no swerve-module model)
  nova         as `fixed`, but NOVA's base profile (its triangle, 0.83 m/s safe wheel,
               0.75 m/s^2 slew, cos choke, twist low-pass) — what NOVA's gateway runs
Plus E5d (swerve): every condition's per-wheel speed / acceleration / torque and steer
rate against the motor limits, the realised-vs-commanded base twist, and a synthetic
forward->sideways step through each profile (the non-holonomic transient).

Outputs docs/figures/rtl_e5_*.pdf + experiments/results/exp_live_take.json and a
summary table. Deterministic (episode clock, fixed conditions). ~4 min.
The take itself is local data (data/episodes/, gitignored): the script exits
cleanly if it is absent.

    ~/anaconda3/bin/python3 web_gui/retarget_lab/experiments/exp_live_take.py
"""
import gzip
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# E5 diagnoses the MAP and the CLOCK on the first take, so the instant servo isolates them
# (with MABEL's 0.8 rad/s HAL slew every arm number is the slew's — that is E6/E7's subject).
os.environ.setdefault("MABEL_RTL_SERVO", "ideal")
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TAKE = "20261015_103700_Oct_15th_test1"
FIG = PKG / "docs" / "figures"
RES = HERE / "results"
CONDS = {
    "fixed":   ({}, "ours", "headset"),
    "arrival": ({}, "ours", "arrival"),
    "clutch":  ({"MABEL_RTL_CLUTCH": "1"}, "ours", "headset"),
    "box":     ({"MABEL_RTL_TORSO_LEAN": "box"}, "ours", "headset"),
    "local":   ({"MABEL_RTL_BASE": "0"}, "ours", "headset"),
    "fast":    ({}, "ours_fast", "headset"),
    "holonomic": ({"MABEL_RTL_SWERVE": "0"}, "ours", "headset"),
    "nova":    ({"MABEL_RTL_ROBOT": "nova"}, "ours", "headset"),
}


def run_one(vid, clock, out):
    """Child process: one condition (env already set) -> npz."""
    sys.path.insert(0, str(PKG.parent))
    sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from retarget_lab.server import config as CFG, episodes
    from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame
    _, frames = episodes.load(TAKE)
    if clock == "arrival":
        raw = [json.loads(line) for line in gzip.open(CFG.EPISODES_DIR / TAKE / "frames.jsonl.gz", "rt")]
        t0 = raw[0]["rt"]
        for fr, r in zip(frames, raw):
            fr["t"] = r["rt"] - t0
        for i in range(1, len(frames)):
            if frames[i]["t"] <= frames[i - 1]["t"]:
                frames[i]["t"] = frames[i - 1]["t"] + 1e-4
    kr = KinematicRetargeter(vid)
    cols = {k: [] for k in ("t", "bx", "by", "byaw", "torso", "lift", "clutched",
                            "trk_l", "trk_r", "cos_l", "cos_r", "armmove", "hx", "hy", "hz",
                            "cvx", "cvy", "cw", "rvx", "rvy", "rw", "wv", "wvreq", "wa", "sr", "se", "scrub", "desat")}
    prev = None
    for fr in frames:
        pose = pose_from_frame(fr)
        rec = kr.step(pose, fr["t"], collide=False)
        b = rec["base"]
        arm = np.array([v for k, v in sorted(rec["q"].items()) if "arm" in k])
        h = np.asarray(pose.head, float)[:3, 3] if pose is not None and pose.head is not None else [np.nan] * 3
        for k, v in (("t", fr["t"]), ("bx", b["x"]), ("by", b["y"]), ("byaw", b["yaw"]),
                     ("torso", rec["torso"]), ("lift", rec["q"].get("lift_upper", np.nan)),
                     ("clutched", rec["clutched"]), ("trk_l", rec["m"]["track_mm_l"]),
                     ("trk_r", rec["m"]["track_mm_r"]), ("cos_l", rec["m"]["cos_reach_l"]),
                     ("cos_r", rec["m"]["cos_reach_r"]),
                     ("armmove", np.abs(arm - prev).max() if prev is not None else np.nan),
                     ("hx", h[0]), ("hy", h[1]), ("hz", h[2])):
            cols[k].append(float(v))
        nan3 = [np.nan] * 3
        c3, r3 = b.get("cmd", nan3), b.get("real", nan3)
        for k, v in (("cvx", c3[0]), ("cvy", c3[1]), ("cw", c3[2]), ("rvx", r3[0]), ("rvy", r3[1]), ("rw", r3[2]),
                     ("wv", np.max(np.abs(b.get("wheel_v", [np.nan])))),
                     ("wvreq", np.max(np.abs(b.get("wheel_v_req", [np.nan])))),
                     ("wa", np.max(np.abs(b.get("wheel_a", [np.nan])))),
                     ("sr", np.max(np.abs(b.get("steer_rate", [np.nan])))),
                     ("se", np.max(np.abs(b.get("steer_err", [np.nan])))),
                     ("scrub", b.get("scrub", np.nan)), ("desat", b.get("desat", np.nan))):
            cols[k].append(float(v))
        prev = arm
    np.savez(out, **{k: np.asarray(v) for k, v in cols.items()})


def longest_still(t, move, eps=1e-6):
    best = cur = 0.0
    for i in range(1, len(t)):
        if move[i] < eps:
            cur += t[i] - t[i - 1]
            best = max(best, cur)
        else:
            cur = 0.0
    return best


def stats(v):
    v = v[np.isfinite(v)]
    return {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "gt150_pct": float((v > 150).mean() * 100)}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    if "--figures" in sys.argv:                       # re-plot from the last run's npz + json
        sys.path.insert(0, str(PKG.parent))
        from retarget_lab.server import config as CFG
        raw = [json.loads(line) for line in gzip.open(CFG.EPISODES_DIR / TAKE / "frames.jsonl.gz", "rt")]
        D = {n: dict(np.load(RES / "live_take_npz" / f"{n}.npz")) for n in CONDS}
        res = json.loads((RES / "exp_live_take.json").read_text())
        rt, ts = np.array([r["rt"] for r in raw]), np.array([r["p"]["timestamp"] for r in raw])
        return figures(D, np.diff(rt), np.diff(ts), res)
    sys.path.insert(0, str(PKG.parent))
    from retarget_lab.server import config as CFG
    src = CFG.EPISODES_DIR / TAKE / "frames.jsonl.gz"
    if not src.exists():
        print(f"take {TAKE} not present (local data) — nothing to do")
        return
    tmp = RES / "live_take_npz"
    tmp.mkdir(parents=True, exist_ok=True)

    def child(name):
        env_extra, vid, clock = CONDS[name]
        env = dict(os.environ, MABEL_LOG_LEVEL="WARNING", **env_extra)
        subprocess.run([sys.executable, __file__, "--child", vid, clock, str(tmp / f"{name}.npz")],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return name

    with ThreadPoolExecutor(len(CONDS)) as ex:
        list(ex.map(child, CONDS))
    D = {n: dict(np.load(tmp / f"{n}.npz")) for n in CONDS}

    # stream timing (the recording itself)
    raw = [json.loads(line) for line in gzip.open(src, "rt")]
    rt = np.array([r["rt"] for r in raw])
    ts = np.array([r["p"]["timestamp"] for r in raw])
    d_arr, d_cap = np.diff(rt), np.diff(ts)

    res = {"take": TAKE, "frames": len(raw), "duration_s": float(rt[-1] - rt[0]),
           "timing": {"arrival_dt_p50_ms": float(np.median(d_arr) * 1e3),
                      "arrival_gaps_gt_90ms": int((d_arr > 0.09).sum()),
                      "arrival_dt_lt_2ms_pct": float((d_arr < 0.002).mean() * 100),
                      "capture_dt_p50_ms": float(np.median(d_cap) * 1e3),
                      "capture_dt_max_ms": float(d_cap.max() * 1e3)},
           "conds": {}}
    for n, d in D.items():
        t = d["t"]
        res["conds"][n] = {
            "track_l": stats(d["trk_l"]), "track_r": stats(d["trk_r"]),
            "longest_frozen_s": longest_still(t, d["armmove"]),
            "clutched_frames": int(d["clutched"].sum()),
            "torso_min_rad": float(np.nanmin(d["torso"])),
            "base_x_range_m": [float(d["bx"].min()), float(d["bx"].max())],
            "base_yaw_range_deg": [float(np.degrees(d["byaw"].min())), float(np.degrees(d["byaw"].max()))],
            "base_path_m": float(np.nansum(np.hypot(np.diff(d["bx"]), np.diff(d["by"])))),
        }
    f = D["fixed"]
    i0 = 30
    res["operator"] = {"head_fwd_max_m": float(np.nanmax(-(f["hz"] - f["hz"][i0]))),
                       "head_back_max_m": float(np.nanmax(f["hz"] - f["hz"][i0])),
                       "head_drop_max_m": float(np.nanmax(np.where(np.abs(f["hy"] - f["hy"][i0]) < 0.5,
                                                                   f["hy"][i0] - f["hy"], np.nan)))}
    res["swerve"] = swerve_summary(D)
    (RES / "exp_live_take.json").write_text(json.dumps(res, indent=1))

    print(f"take {TAKE}: {res['frames']} frames, {res['duration_s']:.1f} s")
    tm = res["timing"]
    print(f"timing: capture dt p50 {tm['capture_dt_p50_ms']:.1f} ms (max {tm['capture_dt_max_ms']:.1f}); "
          f"arrival p50 {tm['arrival_dt_p50_ms']:.1f} ms, {tm['arrival_gaps_gt_90ms']} gaps >90 ms, "
          f"{tm['arrival_dt_lt_2ms_pct']:.1f}% of arrivals <2 ms apart")
    hdr = f"{'cond':9s} {'L p50':>6s} {'L p90':>6s} {'L>150%':>7s} {'R p50':>6s} {'R p90':>6s} {'frozen s':>8s} {'torso':>6s} {'base x':>13s} {'yaw deg':>13s} {'path m':>6s}"
    print(hdr)
    for n, c in res["conds"].items():
        print(f"{n:9s} {c['track_l']['p50']:6.1f} {c['track_l']['p90']:6.1f} {c['track_l']['gt150_pct']:7.2f} "
              f"{c['track_r']['p50']:6.1f} {c['track_r']['p90']:6.1f} {c['longest_frozen_s']:8.2f} "
              f"{c['torso_min_rad']:6.3f} {c['base_x_range_m'][0]:6.2f}..{c['base_x_range_m'][1]:5.2f} "
              f"{c['base_yaw_range_deg'][0]:6.1f}..{c['base_yaw_range_deg'][1]:5.1f} {c['base_path_m']:6.2f}")
    print("swerve (E5d): peak usage vs limits on the take")
    for n, v in res["swerve"].items():
        if n == "step":
            continue
        print(f"  {n:9s} wheel {v['wheel_v_peak']:.3f}/{v['wheel_v_cap']:.2f} m/s (req {v['wheel_v_req_peak']:.3f}) · "
              f"accel {v['wheel_a_peak']:.2f}/{v['wheel_a_cap']:.2f} m/s^2 (torque {v['wheel_tau_peak']:.2f}/{v['wheel_tau_max']:.0f} N m) · "
              f"steer {v['steer_rate_peak']:.1f}/{v['steer_rate_cap']:.1f} rad/s · steer err > deadband {v['steer_choked_pct']:.1f}% · "
              f"|cmd-real| p90 {v['twist_gap_p90'] * 1e3:.0f} mm/s · desat {v['desat_pct']:.1f}% · path {v['path_m']:.2f} m")
    for n, v in res["swerve"]["step"].items():
        print(f"  step fwd->side {n:10s}: 63% {v['t63_ms']:.0f} ms, 95% {v['t95_ms']:.0f} ms, wrong-way peak {v['vx_leak']:.3f} m/s")
    figures(D, d_arr, d_cap, res)


def swerve_summary(D):
    """Peak usage of every base actuator limit on the take, per condition, and the
    synthetic forward->sideways step through each profile."""
    sys.path.insert(0, str(PKG.parent.parent / "controller"))
    from mabel.motion_model import swerve_modules as S
    out = {}
    for n in ("fixed", "nova", "holonomic"):
        d = D[n]
        prof = S.PROFILES["nova" if n == "nova" else "mabel"]
        gap = np.hypot(d["cvx"] - d["rvx"], d["cvy"] - d["rvy"])
        wa = np.nanmax(d["wa"]) if n != "holonomic" else np.nan
        out[n] = {"wheel_v_peak": float(np.nanmax(d["wv"])) if n != "holonomic" else float(np.nanmax(d["wvreq"]) if np.isfinite(d["wvreq"]).any() else np.nan),
                  "wheel_v_req_peak": float(np.nanmax(d["wvreq"])) if np.isfinite(d["wvreq"]).any() else float("nan"),
                  "wheel_v_cap": float(prof.get("wheel_v", S.WHEEL_V_SAFE)),
                  "wheel_a_peak": float(wa), "wheel_a_cap": float(prof.get("wheel_a", S.WHEEL_A_SAFE)),
                  "wheel_tau_peak": float(wa * S.WHEEL_RADIUS * S.ROBOT_MASS / 3.0),
                  "wheel_tau_max": S.WHEEL_TAU_MAX,
                  "steer_rate_peak": float(np.nanmax(d["sr"])) if n != "holonomic" else float("nan"),
                  "steer_rate_cap": float(S.STEER_RATE_MAX),
                  "steer_choked_pct": float(np.nanmean(d["se"] > S.STEER_DEADBAND) * 100) if n != "holonomic" else float("nan"),
                  "twist_gap_p90": float(np.nanpercentile(gap, 90)),
                  "desat_pct": float(np.nanmean(d["desat"] < 1.0) * 100) if n != "holonomic" else float("nan"),
                  "path_m": float(np.nansum(np.hypot(np.diff(d["bx"]), np.diff(d["by"])))),
                  "base_cmd_speed_peak": float(np.nanmax(np.hypot(d["cvx"], d["cvy"]))),
                  "base_cmd_yaw_peak": float(np.nanmax(np.abs(d["cw"])))}
    if "holonomic" in out:
        out["holonomic"]["wheel_v_peak"] = out["fixed"]["wheel_v_req_peak"]   # what an exact twist would ask
    step = {}
    for prof in ("mabel_mjcf", "mabel", "nova"):
        sm = S.SwerveModules(profile=prof)
        for _ in range(100):
            sm.step((0.3, 0.0, 0.0), 0.01)
        tw = np.array([sm.step((0.0, 0.3, 0.0), 0.01) for _ in range(60)])
        step[prof] = {"t63_ms": float(np.argmax(tw[:, 1] > 0.189) * 10), "t95_ms": float(np.argmax(tw[:, 1] > 0.285) * 10),
                      "vx_leak": float(np.abs(tw[:, 0]).max()), "vy": tw[:, 1].tolist(), "vx": tw[:, 0].tolist()}
    out["step"] = step
    return out


def figures(D, d_arr, d_cap, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    C = {"fixed": "#e8a33d", "arrival": "#5aa9e6", "clutch": "#e06c75", "box": "#c678dd",
         "local": "#abb2bf", "fast": "#7bc47f", "holonomic": "#56c2c6", "nova": "#4c72b0"}

    # E5a: stream timing
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    bins = np.logspace(-4, -0.8, 60)
    ax.hist(np.clip(d_cap, 1e-4, None), bins=bins, alpha=.7, label="headset capture stamps", color="#7bc47f")
    ax.hist(np.clip(d_arr, 1e-4, None), bins=bins, alpha=.6, label="arrival at the Mac", color="#5aa9e6")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("inter-frame interval [s]"); ax.set_ylabel("frames")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("E5a — the same 60 Hz stream, captured vs delivered over Wi-Fi", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e5_timing.pdf"); plt.close(fig)

    # E5b: torso lean + base
    f, b, l = D["fixed"], D["box"], D["local"]
    t = f["t"]
    i0 = 30
    fwd = -(f["hz"] - f["hz"][i0]); drop = f["hy"][i0] - f["hy"]
    ok = np.abs(drop) < 0.5
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(t[ok], fwd[ok], lw=.8, color="#333", label="head forward [m]")
    ax.plot(t[ok], drop[ok], lw=.8, color="#999", label="head drop [m]")
    ax.set_ylabel("operator [m]"); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax = axes[1]
    ax.plot(t, np.degrees(-b["torso"]), lw=.9, color=C["box"], label="box law (shipped)")
    ax.plot(t, np.degrees(-f["torso"]), lw=.9, color=C["fixed"], label="angle law (lab)")
    ax.axhline(np.degrees(0.30), ls=":", lw=.7, color=C["box"])
    ax.set_ylabel("robot torso lean [deg]"); ax.legend(frameon=False, fontsize=6.5)
    ax = axes[2]
    ax.plot(t, f["bx"], lw=.9, color=C["fixed"], label="base x, GLOBAL (lab)")
    ax.plot(t, l["bx"], lw=.9, color=C["local"], label="base x, LOCAL (old lab)")
    ax2 = ax.twinx(); ax2.plot(t, np.degrees(f["byaw"]), lw=.7, ls="--", color="#56c2c6")
    ax2.set_ylabel("base yaw [deg]", color="#56c2c6"); ax2.spines["top"].set_visible(False)
    ax.set_ylabel("base x [m] (+ = back)"); ax.set_xlabel("take time [s]")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.suptitle("E5b — whole body on the live take: torso lean and base travel", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e5_wholebody.pdf"); plt.close(fig)

    # E5c: tracking error CDFs + the freeze
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    for n in ("fixed", "arrival", "fast"):
        v = np.sort(np.r_[D[n]["trk_l"], D[n]["trk_r"]]); v = v[np.isfinite(v)]
        ax.plot(v, np.linspace(0, 1, len(v)), color=C[n], lw=1,
                label={"fixed": "Ours, capture clock", "arrival": "Ours, arrival clock",
                       "fast": "Ours, trust 6 rad/s"}[n])
    ax.set_xscale("log"); ax.set_xlabel("own-target error [mm] (both hands)"); ax.set_ylabel("CDF")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("lag on fast motion", fontsize=7.5)
    ax = axes[1]
    for n in ("clutch", "fixed"):
        d = D[n]
        ax.plot(d["t"], np.log10(np.maximum(d["armmove"], 1e-7)), lw=.5, color=C[n],
                label={"clutch": "clutch on (old lab)", "fixed": "clutch off (lab)"}[n])
    ax.set_xlabel("take time [s]"); ax.set_ylabel("log10 max |Δq| per frame [rad]")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(f"the freeze: {res['conds']['clutch']['longest_frozen_s']:.1f} s still after a grasp", fontsize=7.5)
    fig.suptitle("E5c — losing track and freezing", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e5_tracking.pdf"); plt.close(fig)
    # E5d: the base against its motors
    sw = res["swerve"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    for n, lab in (("fixed", "MABEL base"), ("nova", "NOVA base")):
        d = D[n]
        ax.plot(d["t"], d["wv"], lw=.6, color=C[n], label=lab)
    ax.axhline(sw["fixed"]["wheel_v_cap"], ls="--", lw=.7, color=C["fixed"])
    ax.axhline(sw["nova"]["wheel_v_cap"], ls="--", lw=.7, color=C["nova"])
    ax.axhline(1.5, ls=":", lw=.7, color="#e06c75")
    ax.text(1, 1.53, "SPARK collapse 1.5", fontsize=5.5, color="#e06c75")
    ax.set_ylim(0, 1.7)
    ax.set_xlabel("take time [s]"); ax.set_ylabel("max |wheel speed| [m/s]")
    ax.legend(frameon=False, fontsize=6, loc="center right"); ax.set_title("wheel speed vs safe caps (dashed)", fontsize=7)
    ax = axes[1]
    d = D["fixed"]
    ax.plot(d["t"], np.hypot(d["rvx"], d["rvy"]), lw=1.2, color=C["fixed"], label="delivered (3 modules)")
    ax.plot(d["t"], np.hypot(d["cvx"], d["cvy"]), lw=.5, ls="--", color="#222", label="commanded")
    ax.set_xlabel("take time [s]"); ax.set_ylabel("base speed [m/s]")
    ax.legend(frameon=False, fontsize=6); ax.set_title("command vs what the modules deliver", fontsize=7)
    ax = axes[2]
    tt = np.arange(60) * 10
    for prof, col, lab in (("mabel_mjcf", "#abb2bf", "MJCF steer servo"), ("mabel", C["fixed"], "MAXSwerve measured"),
                           ("nova", C["nova"], "NOVA driver")):
        ax.plot(tt, sw["step"][prof]["vy"], lw=1, color=col, label=lab)
    ax.axhline(0.3, ls=":", lw=.6, color="#333")
    ax.set_xlabel("time after the command [ms]"); ax.set_ylabel("sideways speed [m/s]")
    ax.legend(frameon=False, fontsize=6); ax.set_title("forward 0.3 -> sideways 0.3 m/s", fontsize=7)
    fig.suptitle("E5d — the base against its drive and steer motors (live take)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rtl_e5_swerve.pdf"); plt.close(fig)
    print("figures →", FIG)


if __name__ == "__main__":
    main()
