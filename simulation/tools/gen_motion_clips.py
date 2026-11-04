#!/usr/bin/env python3
"""
gen_motion_clips.py — the studio's motion MATH, baked into a table the apps play.

`gen_action_catalog.py` ports the timeline registry's METADATA (title, emoji,
duration, lane, params). It deliberately does not port the `joints(t)` closures,
on the grounds that a second implementation of the motion math in Swift would be
a second source of truth for how MABEL moves. That reasoning stands — so this
script does not re-implement anything either.

It RUNS the real registry (node imports `motions.js` itself) and samples every
clip at the executor's own rate, then writes those samples out as a table. The
app interpolates the table linearly, which is exactly what the robot does with
the 20 Hz command stream the studio sends it: between two commands the servo
holds and ramps. So the baked table is not an approximation of the studio's
motion — at the rate that reaches hardware it IS the studio's motion, and it
cannot drift from the JS because it is produced from the JS.

What stays analytic in Swift (and is NOT baked here), because it is parametric
and short enough to state exactly once:
  · the nav primitives — `navProfile`/`navDur` over a distance/angle parameter,
  · `blend` — the eased from→to interpolation the pose-jump guard inserts.
`MotionMath.swift` carries those, and `MotionClipTests` checks both against the
numbers this script records in the manifest.

    python3 simulation/tools/gen_motion_clips.py           # regenerate
    python3 simulation/tools/gen_motion_clips.py --check   # CI: fail on drift

Re-run it whenever a clip is added, removed or re-authored in motions.js. It
writes the SAME table into both v2 apps (iOS and visionOS), which play the same
choreography through the same Swift engine.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
JS_DIR = os.path.join(REPO, "web_gui", "control_center", "web", "js")
SRC = os.path.join(JS_DIR, "motions.js")
# Both v2 apps play the same choreography, so both get the same table.
OUTS = [
    os.path.join(REPO, "apps", "ios", "MABEL Mobile App",
                 "Resources", "motion_clips.json"),
    os.path.join(REPO, "apps", "visionos", "MabelTeleop",
                 "Resources", "motion_clips.json"),
]

# The executor's command rate. The studio's timeline runs its 20 Hz loop
# (setInterval 50 ms) and the bridge coalesces latest-wins, so this is the
# cadence the robot actually receives — sampling finer would bake detail that
# never reaches a motor.
RATE = 20
# Joint values are radians (or metres for the lift). 1e-4 is ~0.006°, three
# orders of magnitude below anything a servo resolves; it keeps the table small.
DECIMALS = 4

SAMPLER = r"""
// The hand mapping is chosen BEFORE the registry evaluates: motions.js computes its rest poses at
// module load through handmodel.js, so the robot must be set first (dynamic import keeps the order).
const hm = await import('./handmodel.js');
if (hm.setRobot) hm.setRobot('%(robot)s');      // NOVA's handmodel carries the per-robot mapping; MABEL's is single-robot
const { MOTION_CLIPS, clipDur, defaults, clipJointsAt, clipNavAt,
        NAV_SPEED, NAV_SPEED_STRAFE, NAV_YAW_RATE, navProfile,
        SAFE_BLEND_VEL, EASE_PEAK, BLEND_THRESH, BLEND_MIN_DUR,
        MAX_JOINT_VEL, GRIP_SCALE, RANGES, REST,
        LIFT_MAX_VEL, liftRateLimit, EXEC_RATE } = await import('./motions.js');

const RATE = %(rate)d, DEC = %(dec)d;
const round = (v) => Number(v.toFixed(DEC));
// Quantise a step TOWARD the previous sample, never away from it. The rate limit below
// allows 0.00175 m per stage (35 mm/s at 20 Hz) and plain rounding put that on the 1e-4
// grid as 0.0018 — which the two stages SUM to 36.0 mm/s, 3%% OVER the cap the limiter had
// just enforced. Every baked lift clip left the generator over the limit for that reason
// alone. Flooring onto the grid can only ever come out slower.
const GRID = Math.pow(10, -DEC);
const stepTo = (from, to) => {
  const d = to - from;
  return round(from + Math.sign(d) * Math.floor(Math.abs(d) / GRID + 1e-9) * GRID);
};

const clips = {};
for (const [key, clip] of Object.entries(MOTION_CLIPS)) {
  const p = defaults(clip);
  const entry = {
    dur: clip.durFor ? clip.durFor(p) : clip.dur,
    lane: clip.lane, category: clip.category, kind: clip.kind || null,
    hold: !!clip.hold, status: clip.status,
    parametric: Object.keys(clip.params || {}).length > 0,
    hasJoints: !!clip.joints, hasNav: !!clip.nav,
  };
  // `blend` is the one joint clip that is NOT baked: its motion is a function
  // of the from/to poses the pose-jump guard hands it at edit time, so there is
  // no fixed table to sample. MotionMath.swift carries the same eased lerp.
  entry.analytic = clip.kind === 'xfade';
  // Only non-parametric, non-analytic joint clips are baked.
  if (clip.joints && !entry.parametric && !entry.analytic) {
    const n = Math.round(entry.dur * RATE);
    const tracks = {};
    for (let i = 0; i <= n; i++) {
      const j = clipJointsAt(clip, (i / RATE), p) || {};
      for (const [id, v] of Object.entries(j)) {
        (tracks[id] ||= new Array(n + 1).fill(0))[i] = round(v);
      }
    }
    // A clip built on P({...}) returns EVERY joint at EVERY frame, and most of
    // them never move: they sit at their REST value for the whole clip. That
    // is still a command, not an omission — holding the lift half-extended and
    // the wrists level is exactly as load-bearing as the joint that waves — so
    // a flat row is COLLAPSED to its one value rather than dropped. (It used to
    // be a `zeros` list, which was only ever right while REST was qpos zero;
    // with a real rest pose the untouched rows are 0.15875 and ±1.52, and a
    // list of names could not carry them.) Saves ~100 numbers per still joint:
    // the table is 641 KB expanded, 454 KB collapsed.
    // LIFT RATE LIMIT. Bake what the column can actually travel, not what the
    // choreography asks for. The audit found clips demanding up to 488 mm/s from a
    // 35 mm/s column (13.9x), and a table the robot cannot play is a table that
    // desynchronises the whole performance while the lift catches up.
    //
    // The limit is on the SUM of the two prismatic stages — they are commanded
    // together, so capping each at LIFT_MAX_VEL would let the column travel at twice
    // it — and the allowed step is distributed between them in proportion to what
    // each was asked for, which preserves any intended split between the stages.
    {
      const ll = tracks['lift_lower'], lu = tracks['lift_upper'];
      if (ll || lu) {
        const len = (ll || lu).length;
        const g = (t, i) => (t ? t[i] : 0);
        for (let i = 1; i < len; i++) {
          const prev = g(ll, i - 1) + g(lu, i - 1);      // already limited
          const want = g(ll, i) + g(lu, i);              // raw sample
          const lim = liftRateLimit(prev, want, 1 / RATE);
          if (Math.abs(want - lim) > 1e-9) {
            const req = want - prev;
            const scale = Math.abs(req) < 1e-12 ? 0 : (lim - prev) / req;
            if (ll) ll[i] = stepTo(g(ll, i - 1), g(ll, i - 1) + (g(ll, i) - g(ll, i - 1)) * scale);
            if (lu) lu[i] = stepTo(g(lu, i - 1), g(lu, i - 1) + (g(lu, i) - g(lu, i - 1)) * scale);
          }
        }
      }
    }

    const holds = {};
    for (const [id, row] of Object.entries(tracks)) {
      const v0 = row[0];
      if (row.every((v) => Math.abs(v - v0) < 1e-6)) { holds[id] = v0; delete tracks[id]; }
    }
    entry.frames = n + 1;
    entry.tracks = tracks;
    entry.holds = holds;
    // Peak per-sample step, for the velocity-budget assertion in the app's tests.
    let peak = 0, peakJoint = null;
    for (const [id, row] of Object.entries(tracks))
      for (let i = 1; i < row.length; i++) {
        const d = Math.abs(row[i] - row[i - 1]) * RATE;
        if (d > peak) { peak = d; peakJoint = id; }
      }
    entry.peakVel = round(peak);
    entry.peakJoint = peakJoint;
  }
  // A clip can carry BOTH joints and nav (`groove` sways the base while it
  // dances). Non-parametric nav is baked exactly like joints; the parametric
  // nav PRIMITIVES stay analytic in Swift because their duration and amplitude
  // are functions of a distance/angle the operator sets per block.
  if (clip.nav && !entry.parametric) {
    const n = Math.round(entry.dur * RATE);
    const rows = { lx: [], ly: [], rx: [], ry: [] };
    for (let i = 0; i <= n; i++) {
      const nv = clipNavAt(clip, i / RATE, p) || {};
      for (const a of ['lx', 'ly', 'rx', 'ry']) rows[a].push(round(nv[a] || 0));
    }
    for (const a of ['lx', 'ly', 'rx', 'ry'])
      if (rows[a].every((v) => Math.abs(v) < 1e-6)) delete rows[a];
    entry.navTracks = rows;
    entry.navFrames = n + 1;
  }
  if (clip.nav) {
    // Golden sample of the nav stream, so the Swift port (analytic) or the
    // baked table is checked against the registry, not against a comment.
    const d = entry.dur, g = [];
    for (let i = 0; i <= 8; i++) {
      const nv = clipNavAt(clip, (d * i) / 8, p) || {};
      g.push([round(nv.lx || 0), round(nv.ly || 0), round(nv.rx || 0), round(nv.ry || 0)]);
    }
    entry.navGolden = g;
    entry.navParams = p;
  }
  clips[key] = entry;
}

// Constants the Swift side must not restate from memory.
const consts = {
  rate: RATE, navSpeed: NAV_SPEED, navSpeedStrafe: NAV_SPEED_STRAFE,
  navYawRate: NAV_YAW_RATE, safeBlendVel: SAFE_BLEND_VEL, easePeak: EASE_PEAK,
  blendThresh: BLEND_THRESH, blendMinDur: BLEND_MIN_DUR,
  maxJointVel: MAX_JOINT_VEL, gripScale: GRIP_SCALE,
  // Measured on the real column; the apps must not restate it from memory.
  liftMaxVel: LIFT_MAX_VEL,
  // The studio executor's own command rate, so the baker can refuse a table sampled
  // at a rate that robot's executor does not run at (see ROBOTS below).
  execRate: EXEC_RATE,
};
// navProfile golden (a unit 4 s block), so the Swift port of the trapezoid is
// checked numerically, not by eye.
consts.navProfileGolden = Array.from({ length: 9 },
  (_, i) => round(navProfile((4 * i) / 8, 4)));
consts.ranges = Object.fromEntries(
  Object.entries(RANGES).map(([k, v]) => [k, [round(v[0]), round(v[1])]]));
// The rest pose itself. Every clip starts (and every non-hold clip ends) here,
// so the app must not carry a second copy of it: `RobotModel.homePose` is
// checked against this in MotionClipTests.
consts.rest = Object.fromEntries(
  Object.entries(REST).map(([k, v]) => [k, round(v)]));

process.stdout.write(JSON.stringify({ clips, consts }));
"""


# NOVA: ORCA hand v2 (different joint limits and ranges) — its table is baked from NOVA's own copy of the
# registry (the same choreography; its handmodel.js carries the v2 mapping) into nova_motion_clips.json.
#
# RATE PER ROBOT (2026-10-25). NOVA's registry runs its executor at EXEC_RATE = 50 (its gateway, CAN
# links and base all tick at 50 Hz), so its table is sampled at 50: the app's executor also commits at
# 50 Hz (TimelinePlayer.executeHz), and a 20 Hz table interpolated at 50 Hz is a chord of the studio's
# curve, not the curve — e.g. `angry`'s 3 Hz tremor loses ~11 % of its amplitude between 20 Hz samples.
# The lift rate limit and the peak-velocity audit below are computed per sample at the SAME rate, so
# they stay exact at either rate. MABEL's table keeps 20 Hz (its registry still says EXEC_RATE = 20 and
# its committed table must stay byte-identical to what `--check` expects). `bake()` refuses a robot
# whose table rate disagrees with that registry's EXEC_RATE, so this cannot drift silently.
ROBOTS = {
    "mabel": dict(js=JS_DIR, out="motion_clips.json", source="web_gui/control_center/web/js/motions.js",
                  rate=20),
    "nova": dict(js=os.path.expanduser("~/Desktop/NOVA/control_center/web/js"), out="nova_motion_clips.json",
                 source="NOVA/control_center/web/js/motions.js", rate=50),
}


def bake(robot="mabel"):
    global JS_DIR
    JS_DIR = ROBOTS[robot]["js"]
    rate = ROBOTS[robot].get("rate", RATE)
    script = SAMPLER % {"rate": rate, "dec": DECIMALS, "robot": robot}
    fd, path = tempfile.mkstemp(suffix=".mjs", dir=JS_DIR)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        res = subprocess.run(["node", os.path.basename(path)], cwd=JS_DIR, capture_output=True, text=True)
        if res.returncode:
            raise SystemExit(f"node failed sampling {JS_DIR}/motions.js:\n{res.stderr[-1500:]}")
        raw = res.stdout
    finally:
        os.unlink(path)
    data = json.loads(raw)
    exec_rate = data["consts"].pop("execRate", None)
    if exec_rate is not None and exec_rate != rate:
        raise SystemExit(f"{robot}: table rate {rate} Hz != its registry's EXEC_RATE {exec_rate} Hz — "
                         "update ROBOTS[...]['rate'] (and re-check the app's TimelinePlayer.executeHz)")
    data["_source"] = ROBOTS[robot]["source"]
    if robot != "mabel":
        data["_robot"] = robot                  # MABEL's table stays byte-identical to what --check expects
    data["_generator"] = "simulation/tools/gen_motion_clips.py"
    data["_note"] = ("GENERATED — do not edit. Samples of the studio's own "
                     f"clip closures at the executor's {rate} Hz command rate.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed table is stale")
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="mabel",
                    help="whose hand mapping and registry copy to bake (nova -> nova_motion_clips.json)")
    args = ap.parse_args()

    global OUTS
    OUTS = [os.path.join(os.path.dirname(o), ROBOTS[args.robot]["out"]) for o in OUTS]
    data = bake(args.robot)
    text = json.dumps(data, separators=(",", ":"), sort_keys=True)

    if args.check:
        stale = [o for o in OUTS
                 if not os.path.exists(o) or open(o).read() != text]
        if stale:
            for o in stale:
                print(f"STALE: {o} differs from motions.js — re-run without --check")
            return 1
        print(f"ok: {len(OUTS)} tables match {SRC}")
        return 0

    for out in OUTS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(text)

    baked = [k for k, c in data["clips"].items()
             if c.get("tracks") or c.get("holds")]
    navs = [k for k, c in data["clips"].items() if c.get("navGolden")]
    worst = max((c.get("peakVel", 0), k) for k, c in data["clips"].items())
    for out in OUTS:
        print(f"wrote {out}  ({len(text) / 1024:.0f} KB)")
    print(f"  {len(data['clips'])} clips · {len(baked)} baked · {len(navs)} analytic nav")
    print(f"  peak sampled joint velocity: {worst[0]:.2f} rad/s ({worst[1]}) "
          f"vs budget {data['consts']['maxJointVel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
