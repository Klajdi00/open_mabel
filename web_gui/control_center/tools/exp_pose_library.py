#!/usr/bin/env python3
"""
Experiment: the choreography MOTION LIBRARY — every clip and every static pose,
measured against the robot model rather than looked at.

The library lives in the frontend (web/js/motions.js), which is the single
source of truth for what the executor commands and what the two apps bake. This
harness does NOT reimplement any of it: it drives the real ES module through
node and does the kinematics (MuJoCo), the statistics and the plotting here.

  E1  REST ANCHOR. Every clip is authored in absolute joint values whose
      envelopes vanish at the clip edges, and the frame they vanish into is qpos
      zero — which is NOT the robot at rest (see
      controller/experiments/exp_rest_pose.py: qpos zero is a 90 deg bent
      elbow). The registry therefore plays each clip inside an anchor that
      interpolates out of REST and back into it. E1 measures the residual: how
      far each clip's first and last sample actually is from REST.

  E2  VELOCITY BUDGET. Every clip sampled at the executor's own 20 Hz, worst
      joint, against MAX_JOINT_VEL — including the anchor's own travel, which is
      the part that could break the budget and so is the part that is sized from
      it (anchorRamp).

  E3  POSE SILHOUETTES. What each static pose target actually does to the arm,
      by forward kinematics: the hand's position relative to its shoulder
      (forward / outboard / up), the elbow bend, the gravity holding torque, any
      joint outside the MJCF's limits, and any self-collision. These are the
      numbers the pose targets were authored against.

  E4  LIMITS AND CONTACTS over whole clips, not just their targets.

    python3 web_gui/control_center/tools/exp_pose_library.py [--quick]

Requires node (>=18), mujoco and matplotlib. Writes vector PDFs into
web_gui/control_center/docs/figures/ and the summary table this doc quotes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import mujoco
except ImportError:                                     # pragma: no cover
    raise SystemExit("needs mujoco:  pip install mujoco")

HERE = os.path.dirname(os.path.abspath(__file__))
STUDIO = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(STUDIO, "..", ".."))
JS_DIR = os.path.join(STUDIO, "web", "js")
FIGS = os.path.join(STUDIO, "docs", "figures")
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")

SAMPLER = r"""
import { MOTION_CLIPS, POSE_TARGETS, REST, RANGES, MAX_JOINT_VEL, GRIP_SCALE,
         BLEND_SCALE, clipDur, defaults, clipJointsAt, anchorRamp } from './motions.js';
const RATE = %(rate)d;
const clips = {};
for (const [key, c] of Object.entries(MOTION_CLIPS)) {
  if (!c.joints || c.kind === 'xfade' || Object.keys(c.params || {}).length) continue;
  const p = defaults(c), d = clipDur(c, p);
  const n = Math.round(d * RATE);
  const frames = [];
  for (let i = 0; i <= n; i++) frames.push(clipJointsAt(c, i / RATE, p));
  clips[key] = { dur: d, category: c.category, hold: !!c.hold,
                 ramp: anchorRamp(c, p), frames };
}
process.stdout.write(JSON.stringify({
  clips, poses: POSE_TARGETS, rest: REST, ranges: RANGES,
  consts: { maxJointVel: MAX_JOINT_VEL, gripScale: GRIP_SCALE, blendScale: BLEND_SCALE,
            rate: RATE },
}));
"""


def registry(rate=20):
    fd, path = tempfile.mkstemp(suffix=".mjs", dir=JS_DIR)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(SAMPLER % {"rate": rate})
        raw = subprocess.run(["node", os.path.basename(path)], cwd=JS_DIR,
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return json.loads(raw)


class FK:
    """Forward kinematics + statics for a wire-named joint dictionary."""

    WIRE = {}
    for _S, _s in (("Left", "left"), ("Right", "right")):
        for _k in range(1, 8):
            WIRE[f"{_s}_arm_{_k}"] = f"{_S} Arm {_k}"
        WIRE[f"{_s}_wrist"] = f"{_S} Wrist"
    WIRE.update({"torso": "Torso", "neck_yaw": "Neck Yaw", "neck_pitch": "Neck Pitch",
                 "neck_roll": "Neck Roll", "lift_lower": "Lift Lower",
                 "lift_upper": "Lift Upper"})
    ARMJ = [f"{s} Arm {k}" for s in ("Left", "Right") for k in range(1, 8)]

    def __init__(self, path=MJCF):
        self.m = mujoco.MjModel.from_xml_path(path)
        self.d = mujoco.MjData(self.m)
        self.jid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i): i
                    for i in range(self.m.njnt)}
        self.key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")

    def place(self, pose):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.key)
        for k, v in pose.items():
            n = self.WIRE.get(k)
            if n is None:
                continue
            lo, hi = self.m.jnt_range[self.jid[n]]
            self.d.qpos[self.m.jnt_qposadr[self.jid[n]]] = float(np.clip(v, lo, hi))
        mujoco.mj_forward(self.m, self.d)
        return self

    def body(self, n):
        return self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()

    def silhouette(self, side="l"):
        sh, el, hd = self.body(f"{side}_shoulder_3"), self.body(f"{side}_elbow"), \
            self.body(f"{side}_hand_mount")
        u, v = el - sh, hd - el
        bend = np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        r = hd - sh
        return {"fwd": float(-r[0]), "out": float(abs(r[1])), "up": float(r[2]),
                "bend": float(bend), "hand_z": float(hd[2])}

    def torque(self):
        self.d.qvel[:] = 0
        self.d.qacc[:] = 0
        mujoco.mj_inverse(self.m, self.d)
        return float(sum(abs(self.d.qfrc_inverse[self.m.jnt_dofadr[self.jid[n]]])
                         for n in self.ARMJ))

    def contacts(self):
        return int(self.d.ncon)

    def contact_pairs(self):
        """Which BODIES are touching. The MJCF's geoms are unnamed, so a raw
        contact list says nothing; the body pair is the diagnosis."""
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                  self.m.geom_bodyid[c.geom1])
            b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                  self.m.geom_bodyid[c.geom2])
            out.add(" ~ ".join(sorted((a or "?", b or "?"))))
        return sorted(out)


def gap(pose, rest, scale):
    return max((abs(v - rest.get(k, 0.0)) * scale.get(k, 1.0) for k, v in pose.items()),
               default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    reg = registry()
    rest, scale = reg["rest"], reg["consts"]["blendScale"]
    vmax, rate = reg["consts"]["maxJointVel"], reg["consts"]["rate"]
    fk = FK()

    # ── E1 / E2 / E4 · per clip ────────────────────────────────────────────
    rows = []
    for key, c in sorted(reg["clips"].items()):
        fr = c["frames"]
        start = gap(fr[0], rest, scale)
        end = gap(fr[-1], rest, scale)
        peak, pj = 0.0, None
        for i in range(1, len(fr)):
            for j, v in fr[i].items():
                dv = abs(v - fr[i - 1].get(j, 0.0)) * scale.get(j, 1.0) * rate
                if dv > peak:
                    peak, pj = dv, j
        step = 1 if args.quick else max(1, len(fr) // 40)
        bad, con, pairs = 0, 0, set()
        for i in range(0, len(fr), step):
            fk.place(fr[i])
            if fk.contacts():
                con = max(con, fk.contacts())
                pairs.update(fk.contact_pairs())
            for j, v in fr[i].items():
                r = reg["ranges"].get(j)
                if r and not (r[0] - 1e-6 <= v <= r[1] + 1e-6):
                    bad += 1
        rows.append({"key": key, "category": c["category"], "dur": c["dur"],
                     "hold": c["hold"], "ramp": c["ramp"], "start": start, "end": end,
                     "peak": peak, "peakJoint": pj, "outOfRange": bad, "contacts": con,
                     "contactPairs": sorted(pairs)})

    # ── E3 · pose silhouettes ─────────────────────────────────────────────
    poses = []
    for name, target in reg["poses"].items():
        fk.place(target)
        s = fk.silhouette("l")
        poses.append({"name": name, **s, "tau": fk.torque(), "contacts": fk.contacts(),
                      "contactPairs": fk.contact_pairs(),
                      "gap": gap(target, rest, scale)})

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    cats = sorted({r["category"] for r in rows})
    colour = dict(zip(cats, plt.cm.tab10.colors))
    for cat in cats:
        sel = [r for r in rows if r["category"] == cat]
        a.scatter([r["dur"] for r in sel], [r["peak"] for r in sel],
                  s=26, color=colour[cat], label=cat)
    a.axhline(vmax, color="crimson", ls="--", lw=1.2)
    a.text(a.get_xlim()[1], vmax, f" MAX_JOINT_VEL {vmax}", va="bottom", ha="right",
           fontsize=8, color="crimson")
    a.set_xlabel("clip duration [s]")
    a.set_ylabel(f"peak commanded joint speed at {rate:g} Hz [rad/s]")
    a.set_title(f"E2 · velocity budget, all {len(rows)} sampled clips")
    a.legend(fontsize=8, frameon=False)

    a = ax[0, 1]
    idx = np.arange(len(rows))
    a.bar(idx - 0.2, [r["start"] for r in rows], width=0.4, label="first sample")
    a.bar(idx + 0.2, [max(r["end"], 1e-9) for r in rows], width=0.4,
          label="last sample", color="darkorange")
    a.set_yscale("symlog", linthresh=1e-9)
    a.set_ylim(0, 4)
    a.set_xlabel("clip")
    a.set_ylabel("distance from REST [rad-equivalent]")
    a.set_title("E1 · the rest anchor: every clip starts (and non-hold clips end) at REST")
    a.legend(fontsize=8, frameon=False)
    a.set_xticks([])

    a = ax[1, 0]
    order = sorted(poses, key=lambda p: p["hand_z"])
    y = np.arange(len(order))
    a.barh(y, [p["hand_z"] for p in order], color="steelblue")
    a.set_yticks(y)
    a.set_yticklabels([p["name"] for p in order], fontsize=7)
    a.axvline(next(p["hand_z"] for p in poses if p["name"] == "rest"),
              color="seagreen", ls="--", lw=1.2)
    a.set_xlabel("left hand height [m]")
    a.set_title("E3 · pose silhouettes: where each target puts the hand")

    a = ax[1, 1]
    a.scatter([p["gap"] for p in poses], [p["bend"] for p in poses], s=30, color="darkorange")
    for p in poses:
        a.annotate(p["name"], (p["gap"], p["bend"]), fontsize=6,
                   xytext=(3, 3), textcoords="offset points")
    a.axhline(next(p["bend"] for p in poses if p["name"] == "rest"),
              color="seagreen", ls="--", lw=1.2)
    a.text(a.get_xlim()[1], next(p["bend"] for p in poses if p["name"] == "rest"),
           " REST bend (the linkage floor)", ha="right", va="bottom",
           fontsize=8, color="seagreen")
    a.set_xlabel("distance from REST [rad-equivalent]  (this sets the pose's duration)")
    a.set_ylabel("elbow bend [deg]")
    a.set_title("E3 · how far each pose travels, and how bent it leaves the elbow")

    fig.tight_layout()
    out = os.path.join(FIGS, "pose_library.pdf")
    fig.savefig(out)
    plt.close(fig)

    # ── report ────────────────────────────────────────────────────────────
    worst = max(rows, key=lambda r: r["peak"])
    print(f"\nE1  {len(rows)} sampled clips · worst start offset "
          f"{max(r['start'] for r in rows):.2e} rad, worst end offset (non-hold) "
          f"{max((r['end'] for r in rows if not r['hold']), default=0):.2e} rad")
    strand = sorted((r["key"], r["end"]) for r in rows if r["end"] >= 0.15)
    print(f"    clips that deliberately end away from REST: "
          f"{', '.join(f'{k} ({v:.2f})' for k, v in strand) or 'none'}")
    print(f"E2  peak {worst['peak']:.2f} rad/s ({worst['key']}, {worst['peakJoint']}) "
          f"vs budget {vmax}")
    over = [r for r in rows if r["peak"] > vmax + 1e-9]
    print(f"    over budget: {[r['key'] for r in over] or 'none'}")
    print(f"E3  {len(poses)} pose targets")
    print(f"    {'pose':<11}{'fwd':>7}{'out':>7}{'up':>7}{'bend':>7}{'handZ':>7}"
          f"{'tau':>8}{'contacts':>9}")
    for p in sorted(poses, key=lambda x: x["name"]):
        print(f"    {p['name']:<11}{p['fwd']:>7.3f}{p['out']:>7.3f}{p['up']:>7.3f}"
              f"{p['bend']:>6.1f}d{p['hand_z']:>7.3f}{p['tau']:>8.2f}{p['contacts']:>9d}")
    bad = [r["key"] for r in rows if r["outOfRange"]]
    con = [(r["key"], r["contacts"]) for r in rows if r["contacts"]]
    print(f"E4  clips leaving a joint range: {bad or 'none'}")
    print("    clips reaching a self-contact:")
    for r in rows:
        if r["contacts"]:
            print(f"      {r['key']:<14} {r['contacts']:>3}  {'; '.join(r['contactPairs'][:4])}")
    print(f"\nfigure: {os.path.relpath(out, REPO)}")

    js = os.path.join(STUDIO, "docs", "figures", "pose_library.json")
    with open(js, "w") as f:
        json.dump({"clips": rows, "poses": poses,
                   "consts": {"maxJointVel": vmax, "rate": rate}}, f, indent=2)
    print(f"results: {os.path.relpath(js, REPO)}")


if __name__ == "__main__":
    main()
