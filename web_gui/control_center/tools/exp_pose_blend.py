#!/usr/bin/env python3
"""
Experiment: pose-jump safety in the Control Center choreography timeline.

Regenerates every figure and number in docs/timeline_safety.tex from scratch.

The blend law under test lives in the FRONTEND (web/js/motions.js), which is the
single source of truth for what the executor actually commands. This harness
therefore does NOT reimplement it in Python -- it drives the real ES modules
through node and only does the statistics and plotting here. If the JS changes,
these numbers change with it; there is nothing to keep in sync by hand.

Usage:
    python3 control_center/tools/exp_pose_blend.py            # all figures
    python3 control_center/tools/exp_pose_blend.py --quick    # coarse sweep

Requires: node (>=18) and matplotlib. Writes vector PDFs into docs/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                     # control_center/
WEB_JS = os.path.join(PKG, "web", "js")
FIGDIR = os.path.join(PKG, "docs", "figures")

# Executor sampling period: timeline.js runs its choreography loop at 20 Hz.
DT = 0.05
# Documented budgets (motions.js): authored clips ride MAX_JOINT_VEL, blends are
# held to the stricter SAFE_BLEND_VEL.
CLIP_BUDGET = 2.5
SAFE_BLEND_VEL = 1.0

# Palette: keep the studio's accents so figures read as part of the product.
C_RAW = "#D9463A"      # unbridged (danger)
C_BLEND = "#3ECF8E"    # blended (live/ok)
C_LIMIT = "#6b655f"
C_ACC = "#C25B2A"


def node(script: str) -> dict:
    """Run an ES-module snippet against the real frontend sources."""
    path = os.path.join(WEB_JS, "_exp_pose_blend.mjs")
    with open(path, "w") as fh:
        fh.write(script)
    try:
        out = subprocess.run(
            ["node", path], capture_output=True, text=True, check=True,
            cwd=WEB_JS).stdout
    except FileNotFoundError:
        sys.exit("node not found -- install Node >= 18 to run this experiment")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"node failed:\n{exc.stderr}")
    finally:
        os.remove(path)
    return json.loads(out)


# ── the shared JS preamble: the executor's own sampling semantics ────────────
# The executor HOLDS the last commanded pose across a gap (timeline.js only
# re-sends when the pose changes), so the discontinuity lives at the gap exit.
# Sampling the track alone would hide it; we hold-forward exactly as the robot
# experiences it.
PREAMBLE = """
import { ChoreoTrack, MOTION_CLIPS, POSE_TARGETS, autoFixPoseJumps,
         detectPoseJumps, blendMinDur, clipJointsAt, REST,
         EASE_PEAK, ANCHOR_VEL } from './motions.js';
const DT = %f;

function trajectory(track, { skipXfade = false } = {}) {
  const T = [];
  let prev = null;
  const end = track.duration() + 3;
  for (let t = 0; t <= end + 1e-9; t += DT) {
    const p = track.jointsAt(t, { skipXfade });
    if (p) prev = p;                       // hold-forward: the executor's rule
    T.push({ t, pose: prev ? { ...prev } : null });
  }
  return T;
}
function peakVel(T) {
  let worst = 0, at = null, joint = null;
  for (let i = 1; i < T.length; i++) {
    const a = T[i - 1].pose, b = T[i].pose;
    if (!a || !b) continue;
    for (const k in b) {
      if (a[k] == null) continue;
      const v = Math.abs(b[k] - a[k]) / DT;
      if (v > worst) { worst = v; at = T[i].t; joint = k; }
    }
  }
  return { peak: worst, at, joint };
}
// Two HELD synthetic poses separated by a 4 s dead gap, on one joint, so the
// jump magnitude is exactly `delta`.
//
// WHICH SIDE CARRIES THE JUMP CHANGED. Every clip now plays inside a REST
// anchor, so a block's FIRST tick is REST no matter what it is going to do --
// the leading edge of a clip can no longer jump. What can is the TRAILING edge
// of a `hold` block: it strands the robot at its target (that is the whole
// point of `hold`), and whatever comes next starts at REST. So pose A holds
// REST + delta and pose B holds REST, and the step is A's held pose falling
// back to rest -- which is exactly the junction `pickup` -> anything makes.
//
// The synthetic clips size their own duration the way the registry's `poses()`
// does (the anchor ramp plus a tail), because that is the shipped behaviour: a
// pose block long enough to contain its own transition. Pinning them at a fixed
// 2 s instead measures a clip that does not exist, and makes the anchor look
// like it overruns its budget when it is only being squeezed by the harness.
const REF = REST.left_arm_2;
function heldPose(name, value) {
  const gap = Math.abs(value - REF);
  const dur = Math.max(1.2, Math.ceil(10 * (gap * EASE_PEAK / ANCHOR_VEL + 1.2)) / 10);
  return { title: name, emoji: 'P', dur, lane: 'manip',
    category: 'pose', status: 'available', hold: true,
    joints() { return { left_arm_2: value }; },
    poseTarget: () => ({ left_arm_2: value }) };
}
function jumpTrack(delta) {
  const tr = new ChoreoTrack({ ...MOTION_CLIPS,
    poseA: heldPose('A', REF + delta), poseB: heldPose('B', REF) });
  const [a] = tr.add('poseA', 0); a.out = 4;
  tr.add('poseB', 8);
  return tr;
}
""" % DT


def sweep_jump_sizes(n: int) -> dict:
    """Peak commanded velocity vs pose-jump magnitude, with and without blends."""
    script = PREAMBLE + """
const out = [];
const N = %d;
for (let i = 0; i < N; i++) {
  const delta = 0.02 + (3.0 - 0.02) * i / (N - 1);
  const rawPeak = peakVel(trajectory(jumpTrack(delta), { skipXfade: true }));
  const fixed = jumpTrack(delta);
  const res = autoFixPoseJumps(fixed);
  const jumps = detectPoseJumps(jumpTrack(delta));
  const blendPeak = peakVel(trajectory(fixed));
  const b = fixed.blocks.find(x => x.key === 'blend');
  out.push({ delta,
    detected: jumps.length ? jumps[0].delta : 0,
    inserted: res.inserted.length,
    dur: b ? +(b.out - b.in).toFixed(4) : 0,
    minDur: jumps.length ? jumps[0].minDur : 0,
    rawPeak: rawPeak.peak, blendPeak: blendPeak.peak });
}
console.log(JSON.stringify(out));
""" % n
    return node(script)


def representative_traces() -> dict:
    """Position + velocity traces for the canonical T-pose -> arms-up jump."""
    script = PREAMBLE + """
function realTrack() {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [a] = tr.add('pose_tpose', 0); a.out = 4;   // held T-pose
  tr.add('pose_arms_up', 8);                        // …then arms up, after a gap
  return tr;
}
function trace(skip, j) {
  const tr = realTrack();
  if (!skip) autoFixPoseJumps(tr);
  const T = trajectory(tr, { skipXfade: skip });
  const q = [], v = [], ts = [];
  for (let i = 0; i < T.length; i++) {
    if (!T[i].pose) continue;
    ts.push(T[i].t); q.push(T[i].pose[j] ?? 0);
    const a = i ? T[i-1].pose : null;
    v.push(a && a[j] != null ? Math.abs((T[i].pose[j] - a[j]) / DT) : 0);
  }
  return { t: ts, q, v };
}
const tr = realTrack();
const jumps = detectPoseJumps(tr);
const J = jumps[0] ? jumps[0].joint : 'left_arm_2';
autoFixPoseJumps(tr);
const b = tr.blocks.find(x => x.key === 'blend');
// Peak over the BLEND interval (must respect the 1.0 rad/s cap) vs peak over
// the WHOLE trajectory (which legitimately includes the next clip's own
// authored ramp, bounded by the 2.5 rad/s clip budget). Reporting one number
// for both would blame the blend for motion the animator wrote.
const win = (lo, hi) => {
  let prev = null, worst = 0, at = null, j = null;
  for (let t = 0; t <= 12; t += DT) {
    const p = tr.jointsAt(t);
    if (p) { if (prev && t >= lo && t <= hi)
        for (const k in p) if (prev[k] != null) {
          const v = Math.abs(p[k] - prev[k]) / DT;
          if (v > worst) { worst = v; at = t; j = k; } }
      prev = p; } }
  return { peak: worst, at, joint: j };
};
console.log(JSON.stringify({
  joint: J, raw: trace(true, J), blended: trace(false, J),
  peakInBlend: win(b.t0, b.t0 + (b.out - b.in)),
  peakOverall: win(0, 12),
  rawPeak: peakVel(trajectory(realTrack(), { skipXfade: true })),
  jump: jumps[0] ? { delta: jumps[0].delta, minDur: jumps[0].minDur,
                     joint: jumps[0].joint } : null,
  blend: b ? { t0: b.t0, end: b.t0 + (b.out - b.in), dur: b.out - b.in } : null }));
"""
    return node(script)


def duration_law(n: int) -> dict:
    """The min-duration law itself, straight from blendMinDur()."""
    script = PREAMBLE + """
const out = [];
const N = %d;
for (let i = 0; i < N; i++) {
  const d = 0.01 + (6.0 - 0.01) * i / (N - 1);
  out.push({ delta: d, minDur: blendMinDur(d) });
}
console.log(JSON.stringify(out));
""" % n
    return node(script)


# ── figures ─────────────────────────────────────────────────────────────────
def fig_peak_velocity(sw, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d = [r["delta"] for r in sw]
    ax.semilogy(d, [r["rawPeak"] for r in sw], "o-", ms=3, color=C_RAW,
                label="unbridged (whole jump in one 20 Hz tick)")
    ax.semilogy(d, [max(r["blendPeak"], 1e-3) for r in sw], "o-", ms=3,
                color=C_BLEND, label="auto-blended")
    ax.axhline(CLIP_BUDGET, ls="--", lw=1, color=C_LIMIT)
    ax.axhline(SAFE_BLEND_VEL, ls=":", lw=1, color=C_ACC)
    thr = min((r["delta"] for r in sw if r["inserted"]), default=None)
    if thr is not None:
        ax.axvline(thr, ls="-", lw=.8, color=C_LIMIT, alpha=.5)
        ax.text(thr + .04, 3e-1, "blend threshold\n(below: left alone)",
                fontsize=6.5, color=C_LIMIT, va="bottom")
    ax.text(d[-1], CLIP_BUDGET * 1.12, "clip budget 2.5 rad/s", ha="right",
            fontsize=7, color=C_LIMIT)
    ax.text(d[-1], SAFE_BLEND_VEL * 0.72, "blend cap 1.0 rad/s", ha="right",
            fontsize=7, color=C_ACC)
    ax.set_xlabel("pose-jump magnitude $\\Delta$ [rad, worst joint]")
    ax.set_ylabel("peak commanded joint velocity [rad/s]")
    ax.set_title("Pose jumps are the only source of unbounded commanded velocity",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.grid(alpha=.25, which="both", lw=.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_duration_law(law, sw, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot([r["delta"] for r in law], [r["minDur"] for r in law], "-",
            lw=1.6, color=C_ACC, label="$T_{\\min}(\\Delta)$ (enforced floor)")
    ax.plot([r["delta"] for r in sw], [r["dur"] for r in sw], "o", ms=3,
            color=C_BLEND, label="duration actually inserted")
    ax.axhline(0.4, ls=":", lw=1, color=C_LIMIT)
    ax.text(5.9, 0.5, "floor 0.4 s", ha="right", fontsize=7, color=C_LIMIT)
    ax.set_xlabel("pose-jump magnitude $\\Delta$ [rad]")
    ax.set_ylabel("blend duration [s]")
    ax.set_title("Blend duration is derived from pose distance, never guessed",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=.25, lw=.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_traces(tr, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    b = tr.get("blend")
    for ax in (a1, a2):
        if b:
            ax.axvspan(b["t0"], b["end"], color=C_BLEND, alpha=.10, lw=0)
    a1.plot(tr["raw"]["t"], tr["raw"]["q"], "-", lw=1.4, color=C_RAW,
            label="unbridged")
    a1.plot(tr["blended"]["t"], tr["blended"]["q"], "-", lw=1.4, color=C_BLEND,
            label="auto-blended")
    a1.set_ylabel(tr["joint"].replace("_", "\\_") + " [rad]")
    a1.legend(fontsize=7, frameon=False)
    a1.grid(alpha=.25, lw=.4)
    if b:
        a1.text((b["t0"] + b["end"]) / 2, a1.get_ylim()[1] * .93,
                f"blend {b['dur']:.2f} s", ha="center", fontsize=7,
                color=C_BLEND)
    a2.semilogy(tr["raw"]["t"], [max(v, 1e-3) for v in tr["raw"]["v"]], "-",
                lw=1.4, color=C_RAW)
    a2.semilogy(tr["blended"]["t"], [max(v, 1e-3) for v in tr["blended"]["v"]],
                "-", lw=1.4, color=C_BLEND)
    a2.axhline(CLIP_BUDGET, ls="--", lw=1, color=C_LIMIT)
    a2.axhline(SAFE_BLEND_VEL, ls=":", lw=1, color=C_ACC)
    a2.set_ylabel("|commanded vel| [rad/s]")
    a2.set_xlabel("timeline time [s]")
    a2.grid(alpha=.25, which="both", lw=.4)
    a1.set_title("T-pose (held) $\\rightarrow$ arms-up across a 4 s gap",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cut_sweep() -> dict:
    """THE RULE OF THUMB, measured over the whole library.

    Cutting is what breaks the guard's easy assumptions: a clip cut in half ends
    wherever it happened to be mid-motion, so neither the outgoing pose nor the
    gap is anything the library controls. Every clip, cut in half, followed by
    every clip, at four gaps."""
    script = PREAMBLE + """
const keys = Object.entries(MOTION_CLIPS)
  .filter(([, c]) => c.status === 'available' && c.joints && c.kind !== 'xfade'
                  && !Object.keys(c.params || {}).length)
  .map(([k]) => k);
const gaps = [0, 0.3, 1, 3];
const out = { gaps, n: 0, worst: [], over: [], bridges: 0 };
for (const gap of gaps) {
  let worst = 0, over = 0;
  for (const k of keys) for (const j of keys) {
    const tr = new ChoreoTrack(MOTION_CLIPS);
    const [a] = tr.add(k, 0); if (!a) continue;
    const half = tr.end(a) / 2;
    const right = tr.cut(a.id, half); if (right) tr.remove(right.id);
    tr.add(j, half + gap);
    autoFixPoseJumps(tr);
    out.bridges += tr.blocks.filter((b) => b.key === 'blend').length;
    const p = peakVel(trajectory(tr));
    if (p.peak > worst) worst = p.peak;
    if (p.peak > 2.5 + 1e-9) over++;
    out.n++;
  }
  out.worst.push(+worst.toFixed(3));
  out.over.push(over);
}
console.log(JSON.stringify(out));
"""
    return node(script)


def fig_cut_sweep(cs, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = range(len(cs["gaps"]))
    ax.bar(x, cs["worst"], color=C_BLEND, width=0.55)
    ax.axhline(CLIP_BUDGET, color=C_RAW, ls="--", lw=1.2)
    ax.text(len(cs["gaps"]) - 0.5, CLIP_BUDGET, " clip budget", va="bottom",
            ha="right", fontsize=8, color=C_RAW)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g:g} s gap" if g else "butted" for g in cs["gaps"]])
    ax.set_ylabel("worst commanded speed (rad/s)")
    ax.set_ylim(0, CLIP_BUDGET * 1.25)
    for i, w in enumerate(cs["worst"]):
        ax.text(i, w + 0.05, f"{w:.2f}", ha="center", fontsize=8)
    ax.set_title("Every clip cut in half, followed by every clip "
                 f"({cs['n'] // len(cs['gaps'])} pairs each)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse sweep")
    args = ap.parse_args()
    n = 12 if args.quick else 40
    os.makedirs(FIGDIR, exist_ok=True)

    print("sampling the real frontend blend law through node ...")
    sw = sweep_jump_sizes(n)
    law = duration_law(12 if args.quick else 80)
    tr = representative_traces()

    fig_peak_velocity(sw, os.path.join(FIGDIR, "pose_blend_peak_velocity.pdf"))
    fig_duration_law(law, sw, os.path.join(FIGDIR, "pose_blend_duration.pdf"))
    fig_traces(tr, os.path.join(FIGDIR, "pose_blend_trace.pdf"))

    bridged = [r for r in sw if r["inserted"]]
    unb = [r for r in sw if not r["inserted"]]
    print("\n== pose-blend safety: measured ==")
    print(f"  sweep points                 {len(sw)} "
          f"(Delta {sw[0]['delta']:.3f} .. {sw[-1]['delta']:.3f} rad)")
    print(f"  executor tick                {DT*1000:.0f} ms ({1/DT:.0f} Hz)")
    print(f"  jumps bridged                {len(bridged)}/{len(sw)}"
          f"   (below threshold, left alone: {len(unb)})")
    if bridged:
        print(f"  peak vel unbridged           "
              f"{min(r['rawPeak'] for r in bridged):7.2f} .. "
              f"{max(r['rawPeak'] for r in bridged):7.2f} rad/s")
        print(f"  peak vel blended             "
              f"{min(r['blendPeak'] for r in bridged):7.2f} .. "
              f"{max(r['blendPeak'] for r in bridged):7.2f} rad/s")
        worst = max(r["blendPeak"] for r in bridged)
        print(f"  worst blended peak           {worst:.2f} rad/s "
              f"(cap {SAFE_BLEND_VEL}, clip budget {CLIP_BUDGET})")
        ratios = [r["rawPeak"] / max(r["blendPeak"], 1e-9) for r in bridged]
        print(f"  reduction factor             "
              f"{min(ratios):.0f}x .. {max(ratios):.0f}x")
        print(f"  every blend >= its minimum   "
              f"{all(r['dur'] >= r['minDur'] - 1e-9 for r in bridged)}")
        print(f"  blend duration range         "
              f"{min(r['dur'] for r in bridged):.2f} .. "
              f"{max(r['dur'] for r in bridged):.2f} s")
    if tr.get("jump"):
        j, b = tr["jump"], tr["blend"]
        rp, pb, po = tr["rawPeak"], tr["peakInBlend"], tr["peakOverall"]
        print(f"\n  representative case (T-pose held -> arms-up, 4 s gap):")
        print(f"    jump                       {j['delta']:.2f} rad on "
              f"{j['joint']}  ->  blend {b['dur']:.2f} s "
              f"[{b['t0']:.2f}, {b['end']:.2f}] s")
        print(f"    peak unbridged             {rp['peak']:6.2f} rad/s "
              f"at t={rp['at']:.2f}s ({rp['joint']})")
        print(f"    peak INSIDE the blend      {pb['peak']:6.2f} rad/s "
              f"at t={pb['at']:.2f}s ({pb['joint']})   [cap "
              f"{SAFE_BLEND_VEL}]")
        print(f"    peak over the whole run    {po['peak']:6.2f} rad/s "
              f"at t={po['at']:.2f}s ({po['joint']})   [clip budget "
              f"{CLIP_BUDGET}]")
        print("      ^ this is pose_arms_up's OWN authored ramp, not the blend")
        print(f"    reduction at the jump      "
              f"{rp['peak'] / max(pb['peak'], 1e-9):.0f}x")
    cs = cut_sweep()
    fig_cut_sweep(cs, os.path.join(FIGDIR, "pose_blend_cut_sweep.pdf"))
    print("\n  the rule of thumb, over the whole library:")
    print(f"    compositions               {cs['n']} "
          f"(every clip cut in half, then every clip, at "
          f"{', '.join(f'{g:g}s' for g in cs['gaps'])})")
    print(f"    bridges inserted           {cs['bridges']}")
    for g, w, o in zip(cs["gaps"], cs["worst"], cs["over"]):
        print(f"    worst at {g:g} s gap{'':<10} {w:6.2f} rad/s   "
              f"over the clip budget: {o or 'none'}")
    print(f"\nfigures -> {FIGDIR}")


if __name__ == "__main__":
    main()
