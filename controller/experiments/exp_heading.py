#!/usr/bin/env python3
"""
exp_heading.py — H1-H3: the operator BODY-heading estimate the base yaw and the arm
map's body frame both follow. Head+wrist BodyFrameEstimator (shipped) vs the lazy-neck
head model (whole_body_retargeting/heading.py), on REAL Vision Pro data.

  H1 return  — Retarget Lab live takes (2026-10-15): whenever the operator faces
               their calibration direction again (head yaw within 10 deg of it, held
               3 s) the body estimate must be back too. |psi_b - psi_b0| there.
  H2 follow  — same takes: during held turns (head > 60 deg off calibration for 1 s)
               |psi_b - psi_h| — a real turn must be followed.
  H3 seated  — EgoDex benchmark clips (seated, trunk ~fixed, the operator looks
               around): spread of psi_b from its start, p95/max per clip. Head yaw
               spread printed beside it as the look-around the model must reject.
  Sweep      — SLACK x TAU grid on H1/H2/H3 for the lazy model.

Truth caveat: ARKit gives no trunk; H1/H2 use the operator's own head as the
reference only where it is unambiguous (held facing), H3 uses "seated" as the prior.
Outputs results/exp_heading.json + docs/figures/heading_*.pdf. Deterministic. ~1 min.

    ~/anaconda3/bin/python3 controller/experiments/exp_heading.py
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "controller"))
sys.path.insert(0, str(REPO / "web_gui"))
FIG = REPO / "controller" / "docs" / "figures"
RES = HERE / "results"
TAKES = ["20261015_103700_Oct_15th_test1", "20261015_170936_test2-wholebodywalking",
         "20261015_171134_body_rotations"]
CLIPS = sorted((HERE / "data/egodex/benchmark/clips").glob("*.json.gz"))


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sequences():
    """{name: (t, head_mats, wrist_l, wrist_r)} — VP takes then EgoDex clips."""
    from mabel.types import OperatorPose
    from retarget_lab.server import episodes
    out = {}

    def add(name, poses_t):
        t, H, WL, WR = [], [], [], []
        for pose, tt in poses_t:
            if pose is None:
                continue
            t.append(tt)
            H.append(None if pose.head is None else np.asarray(pose.head, float))
            lh, rh = pose.left_hand, pose.right_hand
            WL.append(np.asarray(lh[0], float).reshape(4, 4) if lh else None)
            WR.append(np.asarray(rh[0], float).reshape(4, 4) if rh else None)
        out[name] = (np.array(t), H, WL, WR)

    for take in TAKES:
        try:
            _, frames = episodes.load(take)
        except Exception:                                     # noqa: BLE001
            continue
        from retarget_lab.server.worker import pose_from_frame
        add("vp:" + take.split("_", 2)[2], [(pose_from_frame(f), f["t"]) for f in frames])
    for c in CLIPS[::8]:                                      # every 8th of 774 clips: 97, all tasks
        d = json.load(gzip.open(c))
        pl = d["payloads"]
        fps = float(d.get("fps", 30.0))
        add("egodex:" + c.name.split("__")[1] + c.name.split("__")[2][:5],
            [(OperatorPose.from_payload(p), i / fps) for i, p in enumerate(pl)])
    return out


def run_bodyframe(seq):
    """The shipped estimator exactly as engine._headset_body_yaw runs it (incl. the
    elbow fallback's hold through dropouts, which is what re-seeds the offset)."""
    from mabel.whole_body_retargeting.body_frame_estimator import BodyFrameEstimator
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT
    t, H, WL, WR = seq
    est, fresh, out, last = BodyFrameEstimator(), False, [], None
    first = True
    for i in range(len(t)):
        dt = (t[i] - t[i - 1]) if i else 1 / 60
        if H[i] is not None and WL[i] is not None and WR[i] is not None:
            if first or not est.ready:
                est.calibrate_facing(H[i], WL[i], WR[i]); first = False
            elif not fresh:
                est.seed(H[i], WL[i], WR[i])
            _, B = est.update(H[i], WL[i], WR[i], dt)
            fresh = True
            r = AVP_TO_ROBOT @ B[:, 0]
            last = math.atan2(r[0], -r[1])
        else:
            fresh = False
        out.append(np.nan if last is None else last)
    return np.array(out)


def run_lazy(seq, slack, tau, still):
    from mabel.whole_body_retargeting.heading import LazyHeadHeading
    t, H = seq[0], seq[1]
    m, out = LazyHeadHeading(slack, tau, still), []
    for i in range(len(t)):
        dt = (t[i] - t[i - 1]) if i else 1 / 60
        if m.psi_b is None and H[i] is not None:
            m.calibrate(H[i])
        v = m.update(H[i], dt)
        out.append(np.nan if v is None else v)
    return np.array(out)


def head_yaw(seq):
    from mabel.whole_body_retargeting.heading import head_heading
    return np.array([np.nan if h is None else (head_heading(h) or np.nan) for h in seq[1]])


def held(mask, t, secs):
    """True where `mask` has been continuously true for >= secs."""
    out = np.zeros_like(mask)
    start = None
    for i, m in enumerate(mask):
        if m:
            start = t[i] if start is None else start
            out[i] = t[i] - start >= secs
        else:
            start = None
    return out


def metrics(t, hy, by):
    ok = np.isfinite(hy) & np.isfinite(by)
    i0 = np.argmax(ok)
    dh = np.degrees(np.abs(_wrap(hy - hy[i0])))
    db = np.degrees(np.abs(_wrap(by - by[i0])))
    gap = np.degrees(np.abs(_wrap(by - hy)))
    ret = held(ok & (dh < 10), t, 3.0)
    turn = held(ok & (dh > 60), t, 1.0)
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
    return {"return_p50": q(db[ret], 50), "return_p90": q(db[ret], 90), "return_max": float(db[ret].max()) if ret.any() else float("nan"),
            "return_s": float(ret.sum() / 60.0),
            "follow_p90": q(gap[turn], 90), "follow_n": int(turn.sum()),
            "spread_p95": q(db[ok], 95), "spread_max": float(db[ok].max()) if ok.any() else float("nan"),
            "head_spread_p95": q(dh[ok], 95)}


def main():
    S = sequences()
    from mabel.whole_body_retargeting import heading as HD
    res = {"slack_deg": math.degrees(HD.SLACK), "tau_s": HD.TAU, "per": {}, "sweep": []}
    traces = {}
    for name, seq in S.items():
        t = seq[0]
        hy = head_yaw(seq)
        bf = run_bodyframe(seq)
        lz = run_lazy(seq, HD.SLACK, HD.TAU, HD.STILL_RATE)
        res["per"][name] = {"bodyframe": metrics(t, hy, bf), "lazy": metrics(t, hy, lz)}
        if name.startswith("vp:"):
            traces[name] = (t, hy, bf, lz)
    vp = [n for n in res["per"] if n.startswith("vp:")]
    ego = [n for n in res["per"] if n.startswith("egodex:")]

    def agg(names, est, key, f=np.nanmax):
        v = np.array([res["per"][n][est][key] for n in names], float)
        return float(f(v)) if np.isfinite(v).any() else float("nan")

    print(f"{'take':34s} {'est':9s} {'return p50/p90/max deg':>24s} {'(s held)':>8s} {'follow p90':>10s}")
    for n in vp:
        for est in ("bodyframe", "lazy"):
            m = res["per"][n][est]
            print(f"{n:34s} {est:9s} {m['return_p50']:7.1f} {m['return_p90']:7.1f} {m['return_max']:7.1f} {m['return_s']:8.1f} {m['follow_p90']:10.1f}")
    for est in ("bodyframe", "lazy"):
        sp = np.array([res["per"][n][est]["spread_p95"] for n in ego], float)
        res[f"egodex_{est}"] = {"spread_p95_median": float(np.nanmedian(sp)), "spread_p95_p90": float(np.nanpercentile(sp, 90)),
                                "spread_max_max": agg(ego, est, "spread_max")}
    hs = np.array([res["per"][n]["lazy"]["head_spread_p95"] for n in ego], float)
    res["egodex_head"] = {"spread_p95_median": float(np.nanmedian(hs)), "spread_p95_p90": float(np.nanpercentile(hs, 90))}
    print(f"EgoDex seated, {len(ego)} clips — body-yaw spread p95 per clip (median / p90 over clips), deg:")
    for est in ("bodyframe", "lazy"):
        e = res[f"egodex_{est}"]
        print(f"  {est:9s} {e['spread_p95_median']:6.1f} / {e['spread_p95_p90']:6.1f}   worst max {e['spread_max_max']:6.1f}")
    print(f"  head yaw  {res['egodex_head']['spread_p95_median']:6.1f} / {res['egodex_head']['spread_p95_p90']:6.1f}   (the look-around)")

    # sweep
    for slack in (25, 35, 45):
        for tau in (1.0, 2.0, 4.0):
            r1, r2, r3 = [], [], []
            for name, seq in S.items():
                t = seq[0]; hy = head_yaw(seq)
                m = metrics(t, hy, run_lazy(seq, math.radians(slack), tau, HD.STILL_RATE))
                if name.startswith("vp:"):
                    r1.append(m["return_p90"]); r2.append(m["follow_p90"])
                else:
                    r3.append(m["spread_p95"])
            row = {"slack": slack, "tau": tau, "return_p90_max": float(np.nanmax(r1)), "follow_p90_max": float(np.nanmax(r2)),
                   "seated_spread_p95_median": float(np.nanmedian(r3)), "seated_spread_p95_p90": float(np.nanpercentile(r3, 90))}
            res["sweep"].append(row)
            print(f"  sweep slack {slack:2d} deg tau {tau:.0f} s: return p90 {row['return_p90_max']:5.1f}  follow p90 {row['follow_p90_max']:5.1f}  "
                  f"seated spread p95 {row['seated_spread_p95_median']:5.1f}/{row['seated_spread_p95_p90']:5.1f}")
    RES.mkdir(exist_ok=True)
    (RES / "exp_heading.json").write_text(json.dumps(res, indent=1))
    figures(traces, res)


def figures(traces, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(traces), 1, figsize=(7.0, 2.0 * len(traces)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (name, (t, hy, bf, lz)) in zip(axes, traces.items()):
        i0 = np.argmax(np.isfinite(hy))
        rel = lambda a: np.degrees(_wrap(a - a[np.argmax(np.isfinite(a))]))
        ax.plot(t, rel(hy), lw=.6, color="#999", label="head yaw")
        ax.plot(t, rel(bf), lw=.8, color="#e06c75", label="head+wrist estimator (shipped)")
        ax.plot(t, rel(lz), lw=1.0, color="#4c72b0", label="lazy-neck head model")
        ax.set_ylabel("yaw [deg]"); ax.set_title(name.replace("vp:", "Vision Pro take: "), fontsize=7.5)
        ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper left")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(FIG / "heading_takes.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for row in res["sweep"]:
        ax.scatter(row["seated_spread_p95_median"], row["return_p90_max"], s=14, color="#4c72b0")
        ax.annotate(f"{row['slack']}°/{row['tau']:.0f}s", (row["seated_spread_p95_median"], row["return_p90_max"]), fontsize=5.5)
    e = res["egodex_bodyframe"]
    bf_ret = max(res["per"][n]["bodyframe"]["return_p90"] for n in res["per"] if n.startswith("vp:"))
    ax.scatter(e["spread_p95_median"], bf_ret, s=24, color="#e06c75", marker="x", label="shipped estimator")
    ax.set_xlabel("seated body-yaw spread p95, median clip [deg]")
    ax.set_ylabel("return error p90, worst take [deg]")
    ax.legend(frameon=False, fontsize=6); ax.set_title("lazy-neck sweep: slack / tau", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "heading_sweep.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
