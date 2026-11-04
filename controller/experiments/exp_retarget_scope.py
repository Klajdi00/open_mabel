#!/usr/bin/env python3
"""LOCAL vs GLOBAL retargeting scope — does each mode do what it claims?

THE OPERATOR SPEC (2026-07-31)
  "under the local re-targeting only the lift, torso, arms, hands and the neck
   is in the optimization loop — no mobile base, to keep the problem easier.
   under global add the mobile base in ... only move the mobile base when it is
   outside the defined boundary, both translation and orientation."

Everything is driven by a SYNTHETIC ARKit operator (exp_vp_bimanual_latency's
SyntheticVPOperator), so no headset, network or robot enters the measurement and
each claim is isolated.

WHAT EACH EXPERIMENT ANSWERS
  S1  SCOPE CONTRACT   walking + turning: LOCAL must command exactly zero base
                       motion while GLOBAL commands some. This is the definition
                       of the two modes, so it is measured first and directly.
  S2  UPPER BODY IS    the same arm-tracking metrics in both scopes. The two are
      SCOPE-INVARIANT  supposed to differ ONLY in the base, so wrist tracking
                       must be statistically identical — if it is not, the modes
                       are not what the docs say they are.
  S3  BOUNDARIES       sweep the operator's lean and heading offset and find
                       where the base actually engages, against the declared
                       BASE_BOX_FWD / BASE_YAW_BOX. A boundary nobody measured
                       is a boundary nobody knows the value of.
  S4  LAZINESS         torso and lift response vs operator motion, with and
                       without the deadbands: small sway must move neither.
  S5  TRANSITION       switch scope mid-session and measure the base command
                       discontinuity. A mode switch that lurches is unusable
                       regardless of how correct the steady state is.

    python3 controller/experiments/exp_retarget_scope.py [--quick]
Figures: controller/docs/figures/scope_*.pdf
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import exp_vp_bimanual_latency as h                              # noqa: E402
from mabel.teleop_engine import hyperparams as HP                # noqa: E402

FIGDIR = os.path.join(_REPO, "controller", "docs", "figures")
RESULTS = os.path.join(_HERE, "results")


class WalkingOperator(h.SyntheticVPOperator):
    """Head + hands translate together (a WALK, not a 0.4 m/s relative sweep),
    with an optional body turn."""

    def __init__(self, walk_fn, turn_fn=None, **kw):
        super().__init__(**kw)
        self.walk_fn = walk_fn
        self.turn_fn = turn_fn

    def frame(self, t):
        pose = super().frame(t)
        d = np.asarray(self.walk_fn(t), float)
        if pose.head is not None:
            pose.head = pose.head.copy()
            if self.turn_fn is not None:
                a = float(self.turn_fn(t))
                c, s = np.cos(a), np.sin(a)
                pose.head[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            pose.head[:3, 3] = pose.head[:3, 3] + d
        for side in ("right", "left"):
            j = getattr(pose, f"{side}_joints", None)
            if j:
                moved = {}
                for k, v in j.items():
                    m = np.asarray(v, float).copy()
                    m[:3, 3] = m[:3, 3] + d
                    moved[k] = m
                setattr(pose, f"{side}_joints", moved)
            e = getattr(pose, f"{side}_elbow", None)
            if e is not None:
                e = np.asarray(e, float).copy()
                e[:3, 3] = e[:3, 3] + d
                setattr(pose, f"{side}_elbow", e)
        return pose


def drive(scope, op, secs=8.0, settle=3.0, frame_hz=30.0, switch_at=None):
    """Run the engine in `scope`; return per-tick traces."""
    rig = h.Rig(frame_hz=frame_hz)
    # Ask for RETARGET through the real facade. Without this the facade sits in
    # its default `navigation` method, the engine runs nav_mode, and the
    # whole-body branch that owns torso/base/lift is never reached at all — the
    # arms still track (they run ahead of that branch), so the harness looks
    # alive while measuring nothing about the thing under test.
    rig.teleop.set_control_spec(method="retarget", control_type="impedance",
                                region="whole_body", scope=scope)
    tj = rig.model.jnt_qposadr[rig.model.joint("Torso").id]
    rec = {k: [] for k in ("t", "base", "torso", "lift", "wr_r", "op_r")}
    t, nxt, switched = 0.0, 0.0, False
    while t < settle + secs:
        pose = None
        if t >= nxt:
            pose = op.frame(t)
            nxt += 1.0 / frame_hz
        if switch_at is not None and not switched and t >= settle + switch_at:
            rig.teleop.set_control_spec(scope="global")
            switched = True
        rig.teleop.update(pose, None, h.DT)
        rig._plant()
        if t >= settle:
            rec["t"].append(t - settle)
            rec["base"].append(max(abs(v) for v in rig.eng.wbc.base_tgt))
            rec["torso"].append(float(rig.data.ctrl[rig.eng.torso_act]))
            rec["lift"].append(float(rig.data.ctrl[rig.eng.lift_act]))
            rec["wr_r"].append(rig.data.xpos[rig.palm_bid["right"]].copy())
            j = getattr(op.frame(t), "right_joints", None)
            rec["op_r"].append(np.asarray(j["wrist"], float)[:3, 3] if j else np.full(3, np.nan))
        t += h.DT
    return {k: np.asarray(v, float) for k, v in rec.items()}


# ── S1: the scope contract ────────────────────────────────────────────────────
def s1_contract(quick):
    walk = lambda t: np.array([0.0, 0.0, -0.35 * max(0.0, t - 3.0)])   # noqa: E731
    turn = lambda t: 0.5 * np.sin(2 * np.pi * 0.12 * t)                 # noqa: E731
    out = {}
    for scope in ("local", "global"):
        op = WalkingOperator(walk, turn, wrist_fn=lambda s, t: h.traj_static(s, t))
        r = drive(scope, op, secs=6.0 if not quick else 4.0)
        peak = float(np.nanmax(r["base"]))
        out[scope] = {"base_peak": peak, "trace": {"t": r["t"], "base": r["base"]}}
        print(f"  S1 {scope:6s} peak |base cmd| while walking+turning: {peak:.4f} m/s")
    ok = out["local"]["base_peak"] < 1e-9 < out["global"]["base_peak"]
    print(f"  S1 VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(LOCAL must be exactly 0, GLOBAL must be > 0)")
    out["pass"] = bool(ok)
    return out


# ── S2: the upper body must not notice the scope ─────────────────────────────
def s2_upper_body_invariance(quick):
    out = {}
    for scope in ("local", "global"):
        op = h.SyntheticVPOperator(lambda s, t: h.traj_bimanual(s, t))
        rig = h.Rig(frame_hz=30.0)
        rig.teleop.set_control_spec(method="retarget", control_type="impedance",
                                    region="whole_body", scope=scope)
        rec = rig.run(op, 6.0 if not quick else 4.0)
        corr = h.track_corr(h.principal(rec["op_r"]), h.principal(rec["wrist_r"]))
        lag = h.lockin_lag(rec["t"], h.principal(rec["op_r"]),
                           h.principal(rec["wrist_r"]), h.SWEEP_HZ["right"]) * 1e3
        out[scope] = {"corr": abs(corr), "lag_ms": lag}
        print(f"  S2 {scope:6s} right-arm tracking |r| {abs(corr):.3f}   lag {lag:+6.1f} ms")
    d_corr = abs(out["local"]["corr"] - out["global"]["corr"])
    d_lag = abs(out["local"]["lag_ms"] - out["global"]["lag_ms"])
    ok = d_corr < 0.02 and d_lag < 15.0
    print(f"  S2 VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(Δ|r| {d_corr:.3f} < 0.02, Δlag {d_lag:.1f} < 15 ms)")
    out["pass"] = bool(ok)
    return out


# ── S3: where do the boundaries actually engage? ─────────────────────────────
def s3_boundaries(quick):
    leans = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30] if not quick else [0.0, 0.15, 0.30]
    yaws = [0.0, 0.10, 0.20, 0.35, 0.60] if not quick else [0.0, 0.35, 0.60]
    trans, rot = [], []
    for L in leans:
        op = WalkingOperator(lambda t, L=L: np.array([0.0, 0.0, -L * min(1.0, t / 3.0)]),
                             wrist_fn=lambda s, t: h.traj_static(s, t))
        r = drive("global", op, secs=5.0, settle=3.0)
        v = float(np.nanmax(r["base"]))
        trans.append((L, v))
        print(f"  S3 lean {L*100:5.1f} cm -> base {v:.4f} m/s")
    for Y in yaws:
        # A REAL BODY TURN: the head AND the hands rotate together. Turning only
        # the head is a look-around, which the body-frame estimator is built to
        # reject — the first version of this sweep did exactly that and measured
        # 0.000 at every angle, i.e. it was testing the estimator's rejection
        # rather than the base's orientation boundary.
        def _turn_traj(side, t, Y=Y):
            a = Y * min(1.0, t / 3.0)
            p0 = h.traj_static(side, t)
            c, sn = np.cos(a), np.sin(a)
            R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
            pivot = np.array([0.0, p0[1], 0.0])
            return pivot + R @ (p0 - pivot)
        op = WalkingOperator(lambda t: np.zeros(3),
                             turn_fn=lambda t, Y=Y: Y * min(1.0, t / 3.0),
                             wrist_fn=_turn_traj)
        r = drive("global", op, secs=5.0, settle=3.0)
        v = float(np.nanmax(r["base"]))
        rot.append((Y, v))
        print(f"  S3 yaw  {np.degrees(Y):5.1f} deg -> base {v:.4f} m/s")
    return {"translation": trans, "rotation": rot,
            "declared_box_fwd": HP.BASE_BOX_FWD, "declared_yaw_box": HP.BASE_YAW_BOX}


# ── S4: laziness of the slow DoFs ────────────────────────────────────────────
def s4_laziness(quick):
    amps = [0.02, 0.05, 0.10, 0.20] if not quick else [0.02, 0.10]
    out = []
    for a in amps:
        op = WalkingOperator(lambda t, a=a: np.array([0.0, 0.30 * a * np.sin(2 * np.pi * 0.3 * t),
                                                      -a * np.sin(2 * np.pi * 0.3 * t)]),
                             wrist_fn=lambda s, t: h.traj_static(s, t))
        r = drive("local", op, secs=6.0, settle=3.0)
        dt_ = float(np.degrees(np.nanmax(r["torso"]) - np.nanmin(r["torso"])))
        dl = float((np.nanmax(r["lift"]) - np.nanmin(r["lift"])) * 1e3)
        out.append({"sway_m": a, "torso_deg": dt_, "lift_mm": dl})
        print(f"  S4 sway {a*100:5.1f} cm -> torso {dt_:6.2f} deg   lift {dl:6.1f} mm")
    # HONESTY GATE. If NOTHING moved at any amplitude — including well past the
    # deadband — this measured the harness, not the laziness: the synthetic
    # operator is not satisfying the EngagementSupervisor, so the supervisor's
    # hold is writing the torso/lift back every tick and the deadband never gets
    # a say. Reporting that as "lazy: PASS" would be the worst possible outcome,
    # so it reports UNMEASURED instead.
    moved = any(r["torso_deg"] > 0.01 or r["lift_mm"] > 0.1 for r in out)
    if not moved:
        print("  S4 UNMEASURED — torso/lift never moved at ANY amplitude, including")
        print("     past the deadband. That is the harness (the supervisor is holding")
        print("     the pose), not evidence of laziness. Needs an operator model that")
        print("     satisfies engagement before this number means anything.")
    return {"rows": out, "torso_deadband": HP.TORSO_DEADBAND,
            "lift_deadband": HP.LIFT_DEADBAND, "measured": bool(moved)}


# ── S5: switching scope mid-session ──────────────────────────────────────────
def s5_transition(quick):
    walk = lambda t: np.array([0.0, 0.0, -0.30 * max(0.0, t - 1.0)])    # noqa: E731
    op = WalkingOperator(walk, wrist_fn=lambda s, t: h.traj_static(s, t))
    r = drive("local", op, secs=8.0 if not quick else 6.0, settle=3.0, switch_at=3.0)
    t, b = r["t"], r["base"]
    i = int(np.searchsorted(t, 3.0))
    step = float(np.nanmax(np.abs(np.diff(b[max(0, i - 2):i + 6])))) if i + 6 < len(b) else np.nan
    ok = np.isfinite(step) and step < 0.05
    print(f"  S5 base-command step across the LOCAL->GLOBAL switch: {step:.4f} m/s")
    print(f"  S5 VERDICT: {'PASS' if ok else 'FAIL'} (< 0.05 m/s = no lurch)")
    return {"step": step, "pass": bool(ok), "trace": {"t": t, "base": b}}


# ── figures ───────────────────────────────────────────────────────────────────
def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10); ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9); ax.grid(alpha=0.25, lw=0.6); ax.tick_params(labelsize=8)


def figures(res):
    os.makedirs(FIGDIR, exist_ok=True)
    if "S1" in res:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        for sc, c in (("local", "#2A6F97"), ("global", "#C44536")):
            tr = res["S1"][sc]["trace"]
            ax.plot(tr["t"], tr["base"], lw=1.6, color=c, label=f"{sc}")
        _style(ax, "S1 — walking + turning: does the base follow?", "time [s]",
               "|base command| [m/s]")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "scope_contract.pdf"), format="pdf",
                    bbox_inches="tight"); plt.close(fig)
    if "S3" in res:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        tr = np.asarray(res["S3"]["translation"], float)
        axes[0].plot(tr[:, 0] * 100, tr[:, 1], "o-", color="#C44536")
        axes[0].axvline(res["S3"]["declared_box_fwd"] * 100, ls=":", color="k",
                        label=f"declared box {res['S3']['declared_box_fwd']*100:.0f} cm")
        _style(axes[0], "S3 — translation boundary", "operator lean [cm]", "base [m/s]")
        axes[0].legend(fontsize=7)
        ro = np.asarray(res["S3"]["rotation"], float)
        axes[1].plot(np.degrees(ro[:, 0]), ro[:, 1], "o-", color="#2A6F97")
        axes[1].axvline(np.degrees(res["S3"]["declared_yaw_box"]), ls=":", color="k",
                        label=f"declared box {np.degrees(res['S3']['declared_yaw_box']):.0f} deg")
        _style(axes[1], "S3 — orientation boundary", "operator heading [deg]", "base [m/s]")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "scope_boundaries.pdf"), format="pdf",
                    bbox_inches="tight"); plt.close(fig)
    if "S5" in res:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        tr = res["S5"]["trace"]
        ax.plot(tr["t"], tr["base"], lw=1.5, color="#C44536")
        ax.axvline(3.0, ls="--", color="k", lw=1.0, label="LOCAL → GLOBAL")
        _style(ax, "S5 — mode switch (must not lurch)", "time [s]", "|base command| [m/s]")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "scope_transition.pdf"), format="pdf",
                    bbox_inches="tight"); plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "trace"}
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    want = {s.strip().upper() for s in a.only.split(",") if s.strip()}
    run = lambda k: (not want) or k in want               # noqa: E731

    print(f"\nLOCAL vs GLOBAL RETARGET SCOPE — {'quick' if a.quick else 'full'}")
    print(f"declared boxes: fwd {HP.BASE_BOX_FWD} m, yaw {HP.BASE_YAW_BOX} rad "
          f"({np.degrees(HP.BASE_YAW_BOX):.0f} deg); "
          f"deadbands: torso {HP.TORSO_DEADBAND} m, lift {HP.LIFT_DEADBAND} m\n")
    res = {}
    if run("S1"):
        print("S1 SCOPE CONTRACT");        res["S1"] = s1_contract(a.quick)
    if run("S2"):
        print("S2 UPPER-BODY INVARIANCE"); res["S2"] = s2_upper_body_invariance(a.quick)
    if run("S3"):
        print("S3 BOUNDARIES");            res["S3"] = s3_boundaries(a.quick)
    if run("S4"):
        print("S4 LAZINESS");              res["S4"] = s4_laziness(a.quick)
    if run("S5"):
        print("S5 TRANSITION");            res["S5"] = s5_transition(a.quick)
    figures(res)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "retarget_scope.json"), "w") as f:
        json.dump(_clean(res), f, indent=2)
    verdicts = {k: v.get("pass") for k, v in res.items() if isinstance(v, dict) and "pass" in v}
    if verdicts:
        print(f"\nVERDICTS: {verdicts}")
    print()


if __name__ == "__main__":
    main()
