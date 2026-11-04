#!/usr/bin/env python3
"""
exp_hand_retarget.py — rigorous, headless benchmark of the ORCA finger
retargeting (controller/mabel/hand_ik_retarget.py).

Motivated by a live regression report: "fingertips clearly touch on the webcam
but the robot fingers do not pitch". The experiment measures, for the GN
solver (deployed default), the SLSQP reference-cost port, and hyperparameter
variants of both:

  1. CURL TRANSFER  — a parametric synthetic human hand flexes one finger (or
     all) from 0 → 100 %; we record the robot finger's achieved flexion
     (mean MCP+PIP angle as a fraction of range) and the normalized fingertip
     tracking error. A healthy retargeter gives a monotonic, near-identity
     transfer curve.
  2. PINCH CLOSURE  — thumb–index fingertip touch (the webcam case): the
     metric is the ROBOT's physical thumb–index tip gap (mm, palm-frame FK)
     when the human gap is zero.
  3. WEBCAM DEPTH LOSS — the same battery with the human z (out-of-plane)
     compressed to 40 % and with 2 mm landmark noise: MediaPipe-style input.
  4. SPEED — per-solve wall time.

Nothing touches the live stack: the scene is loaded in-process and only the
retargeter's mini-FK is exercised (kinematics, no mj_step, no /tmp files).

Run:  ~/anaconda3/bin/python3 controller/experiments/exp_hand_retarget.py
      [--quick] [--figures]
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import mujoco  # noqa: E402
from mabel import hand_ik_retarget as hik  # noqa: E402
from mabel import hand_vector_retarget as hvr  # noqa: E402
from mabel.hand_ik_retarget import HandIKRetargeter, human_keypoints  # noqa: E402
from mabel.hand_vector_retarget import HandVectorRetargeter  # noqa: E402

SCENE = os.path.join(_HERE, "..", "..", "simulation", "mabel_mujoco",
                     "scenes", "mabel_home.xml")

# ── synthetic parametric human hand (ARKit wire format) ─────────────────────
# Anthropometry (m): a typical adult RIGHT hand, palm-down, wrist at the
# origin, fingers along +y, flexion toward −z. Palm-down right hand ⇒ the
# INDEX is on the −x side and the thumb further −x (get it wrong and the
# canonical palm frame flips chirality — the first version of this harness
# did exactly that and produced impossible flexion targets).
# The retargeter consumes POSITIONS of: wrist, {index,middle,ring,little}
# {knuckle, intermediate_base, tip}, thumb {intermediate_base,
# intermediate_tip, tip} — so the model only needs those points.
_KN_X = {"index": -0.032, "middle": -0.011, "ring": 0.011, "little": 0.032}
_KN_Y = {"index": 0.090, "middle": 0.095, "ring": 0.090, "little": 0.080}
_SEG1 = {"index": 0.047, "middle": 0.052, "ring": 0.049, "little": 0.038}  # kn→pip
_SEG2 = {"index": 0.045, "middle": 0.049, "ring": 0.046, "little": 0.038}  # pip→tip
_TH_KN = np.array([-0.045, 0.035, 0.0])      # thumb "knuckle" (intermediate_base)
_TH_SEG1, _TH_SEG2 = 0.045, 0.040


def _T(p):
    m = np.eye(4)
    m[:3, 3] = p
    return m


def synth_hand(curl=None, thumb_curl=0.0, pinch=None, z_scale=1.0,
               noise=0.0, rng=None):
    """Build {wire_name: 4x4} for a right hand.

    curl:       {finger: c∈[0,1]} flexion (0 = straight, 1 = full fist-curl);
    thumb_curl: thumb flexion toward the palm;
    pinch:      finger name — OVERRIDES geometry: places that fingertip AND
                the thumb tip at their midpoint (a true touch, gap = 0);
    z_scale:    out-of-plane compression (webcam depth loss model);
    noise:      landmark sigma (m).
    """
    curl = curl or {}
    pts = {"wrist": np.zeros(3)}
    for f in ("index", "middle", "ring", "little"):
        kn = np.array([_KN_X[f], _KN_Y[f], 0.0])
        c = float(curl.get(f, 0.0))
        # flexion folds the two bones toward the palm (−z), progressive:
        # pip rotates by θ, tip by ~2θ, θ_max = 95°
        th = np.deg2rad(95.0) * c
        d1 = np.array([0.0, np.cos(th), -np.sin(th)])
        th2 = min(np.deg2rad(185.0) * c, np.deg2rad(170.0))
        d2 = np.array([0.0, np.cos(th2), -np.sin(th2)])
        pip = kn + _SEG1[f] * d1
        tip = pip + _SEG2[f] * d2
        pts[f"{f}_knuckle"] = kn
        pts[f"{f}_intermediate_base"] = pip
        pts[f"{f}_tip"] = tip
    tc = float(thumb_curl)
    ath = np.deg2rad(80.0) * tc
    # thumb: from its knuckle toward +y, curling toward the palm centre (+x,−z)
    d1 = np.array([np.sin(ath) * 0.6, np.cos(ath), -np.sin(ath) * 0.8])
    d1 /= np.linalg.norm(d1)
    pip = _TH_KN + _TH_SEG1 * d1
    tip = pip + _TH_SEG2 * d1
    pts["thumb_intermediate_base"] = _TH_KN
    pts["thumb_intermediate_tip"] = pip
    pts["thumb_tip"] = tip

    if pinch is not None:
        mid = (pts[f"{pinch}_tip"] + pts["thumb_tip"]) / 2.0
        pts[f"{pinch}_tip"] = mid.copy()
        pts["thumb_tip"] = mid.copy()

    for k in pts:
        p = pts[k].copy()
        p[2] *= z_scale
        if noise > 0 and rng is not None:
            p = p + rng.normal(0.0, noise, 3)
        pts[k] = p
    return {k: _T(p) for k, p in pts.items()}


# ── robot-side measurements ──────────────────────────────────────────────────
def robot_flexion(ret, side, finger, q):
    """Mean (MCP, PIP) angle as a fraction of each joint's range ∈ [0, 1]."""
    hand = ret.hands[side]
    fr = []
    for i, name in enumerate(hand["act_names"]):
        if finger.capitalize() in name and ("MCP" in name or "PIP" in name):
            lo, hi = hand["bounds"][i]
            fr.append((q[i] - lo) / (hi - lo))
    return float(np.mean(fr)) if fr else np.nan


def robot_tip_gap_mm(ret, side, q, finger="index"):
    """Physical thumb↔fingertip gap (mm) from the palm-frame mini-FK."""
    hand = ret.hands[side]
    frames = hand["fk"].frames(q)
    pre = hand["pre"]
    tiplen = hand["tiplen"]

    def tip(body, seg):
        R, p = frames[body]
        return p + R @ np.array([0, 0, tiplen[seg] * hik.TIP_EXTRAPOLATE])

    t = tip(f"{pre}_thumb_dp", "thumb")
    f = tip(f"{pre}_{finger}_ip", finger)
    return float(np.linalg.norm(t - f) * 1000.0)


def solve(ret, side, joints, iters=1):
    """Feed a skeleton `iters` times synchronously (warm start), return q."""
    hand = ret.hands[side]
    for _ in range(iters):
        hand["last_solve"] = 0.0
        ret.apply(side, joints, synchronous=True)
    return hand["last_q"].copy()


def shape_err(ret, side, joints):
    """Mean normalized keypoint error (canonical units) of the solution."""
    hand = ret.hands[side]
    h_raw = human_keypoints(joints, side.lower())
    h = ret._calibrate(h_raw, hand["lengths"], hand["kn0"])
    r = ret._robot_keypoints(hand, hand["last_q"])
    common = [k for k in r if k in h]
    return float(np.mean([np.linalg.norm(r[k] - h[k]) for k in common]))


# ── experiment battery ───────────────────────────────────────────────────────
def run_battery(model, data, solver, label, cfg=None, z_scale=1.0, noise=0.0,
                seed=0, curl_steps=6, settle_iters=8):
    """Returns a metrics dict for one solver configuration and input regime."""
    saved = {}
    for k, v in (cfg or {}).items():
        mod = hvr if k.startswith("WV_") else hik
        saved[(mod.__name__, k)] = getattr(mod, k)
        setattr(mod, k, v)
    rng = np.random.default_rng(seed)
    ret = (HandVectorRetargeter(model, data) if solver == "vector"
           else HandIKRetargeter(model, data, solver=solver))
    side = "Right"
    out = {"label": label, "solver": solver, "cfg": dict(cfg or {}),
           "z": z_scale, "noise": noise}

    # 1. curl transfer, index finger (the "pitch" the operator sees)
    cs = np.linspace(0, 1, curl_steps)
    transfer, errs = [], []
    ret.reset()
    for c in cs:
        sk = synth_hand(curl={"index": c}, z_scale=z_scale, noise=noise, rng=rng)
        q = solve(ret, side, sk, iters=settle_iters)
        transfer.append(robot_flexion(ret, side, "index", q))
        errs.append(shape_err(ret, side, sk))
    out["curl_cmd"] = cs.tolist()
    out["curl_ach"] = transfer
    out["curl_final"] = transfer[-1]
    out["curl_auc"] = float(np.trapezoid(transfer, cs)
                            / max(np.trapezoid(cs, cs), 1e-9))
    out["shape_err"] = float(np.mean(errs))
    # cold-start full curl (no warm-start path dependence)
    ret.reset()
    sk = synth_hand(curl={"index": 1.0}, z_scale=z_scale, noise=noise, rng=rng)
    q = solve(ret, side, sk, iters=settle_iters)
    out["curl_cold"] = robot_flexion(ret, side, "index", q)

    # 2. all-finger fist
    ret.reset()
    sk = synth_hand(curl={f: 1.0 for f in ("index", "middle", "ring", "little")},
                    thumb_curl=0.8, z_scale=z_scale, noise=noise, rng=rng)
    q = solve(ret, side, sk, iters=settle_iters)
    out["fist_flexion"] = float(np.mean(
        [robot_flexion(ret, side, f, q) for f in ("index", "middle", "ring", "pinky")]))

    # 3. thumb-index pinch: human gap = 0 → robot gap (mm)
    ret.reset()
    sk = synth_hand(curl={"index": 0.55}, thumb_curl=0.55, pinch="index",
                    z_scale=z_scale, noise=noise, rng=rng)
    q = solve(ret, side, sk, iters=settle_iters)
    out["pinch_gap_mm"] = robot_tip_gap_mm(ret, side, q, "index")

    # 4. open-hand fidelity (no spurious curl)
    ret.reset()
    sk = synth_hand(z_scale=z_scale, noise=noise, rng=rng)
    q = solve(ret, side, sk, iters=settle_iters)
    out["open_flexion"] = float(np.mean(
        [robot_flexion(ret, side, f, q) for f in ("index", "middle", "ring", "pinky")]))

    # 5. speed
    sk = synth_hand(curl={"index": 0.5})
    t0 = time.perf_counter()
    n = 30
    for _ in range(n):
        solve(ret, side, sk, iters=1)
    out["ms_per_solve"] = (time.perf_counter() - t0) / n * 1000.0

    for (modname, k), v in saved.items():
        setattr(hvr if modname.endswith("hand_vector_retarget") else hik, k, v)
    return out


def fmt(r):
    return (f"{r['label']:34s} curl100={r['curl_final']:.2f} "
            f"auc={r['curl_auc']:.2f} cold={r['curl_cold']:.2f} fist={r['fist_flexion']:.2f} "
            f"open={r['open_flexion']:.2f} pinch={r['pinch_gap_mm']:5.1f}mm "
            f"err={r['shape_err']:.3f} {r['ms_per_solve']:5.2f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="write PDFs into controller/docs/figures/")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    data = mujoco.MjData(model)

    regimes = [("clean 3D", 1.0, 0.0), ("webcam z×0.4 + 2mm", 0.4, 0.002)]
    results = {}

    print("=== BASELINES (deployed defaults) ===")
    for rl, z, nz in regimes:
        for solver in ("gn", "slsqp", "vector"):
            r = run_battery(model, data, solver, f"{solver} [{rl}]",
                            z_scale=z, noise=nz)
            results.setdefault(rl, []).append(r)
            print(fmt(r))

    print("\n=== GN hyperparameter sweep (webcam regime) ===")
    sweeps = [
        ("gn iters=12",            {"GN_ITERS": 12}),
        ("gn reg=0.005",           {"W_REG": 0.005}),
        ("gn smooth=0.2",          {"W_SMOOTH": 0.2}),
        ("gn shape=60",            {"W_SHAPE": 60.0}),
        ("gn iters=12 reg=.005 sm=.2", {"GN_ITERS": 12, "W_REG": 0.005, "W_SMOOTH": 0.2}),
        ("gn TUNED (i12 reg.005 sm.2 sh60)",
         {"GN_ITERS": 12, "W_REG": 0.005, "W_SMOOTH": 0.2, "W_SHAPE": 60.0}),
    ]
    if not args.quick:
        sweeps += [
            ("slsqp iters=20",     {"SLSQP_ITERS": 20}),
            ("vector iters=12",    {"WV_ITERS": 12}),
            ("vector flex=60",     {"WV_FLEX": 60.0}),
            ("vector reg=.005 sm=.3", {"WV_REG": 0.005, "WV_SMOOTH": 0.3}),
            ("vector TUNED (i8 flex60 reg.005)",
             {"WV_ITERS": 8, "WV_FLEX": 60.0, "WV_REG": 0.005}),
        ]
    for label, cfg in sweeps:
        solver = ("slsqp" if label.startswith("slsqp")
                  else "vector" if label.startswith("vector") else "gn")
        r = run_battery(model, data, solver, label, cfg=cfg,
                        z_scale=0.4, noise=0.002)
        results.setdefault("sweep", []).append(r)
        print(fmt(r))

    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figdir = os.path.join(_HERE, "..", "docs", "figures")
        os.makedirs(figdir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
        for ax, rl in zip(axes, [r[0] for r in regimes]):
            for r in results[rl]:
                ax.plot(r["curl_cmd"], r["curl_ach"], marker="o", ms=3,
                        label=r["label"].split(" [")[0])
            best = [r for r in results.get("sweep", []) if "TUNED" in r["label"]]
            if best and "webcam" in rl:
                ax.plot(best[0]["curl_cmd"], best[0]["curl_ach"], marker="s",
                        ms=3, lw=2, label="gn tuned")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="identity")
            ax.set_title(rl, fontsize=9)
            ax.set_xlabel("human index flexion (fraction)")
            ax.set_ylabel("robot flexion (fraction of range)")
            ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "hand_curl_transfer.pdf"))
        print("figure -> docs/figures/hand_curl_transfer.pdf")


if __name__ == "__main__":
    main()
