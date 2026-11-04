// Pose-jump safety, measured the way the ROBOT experiences it.
//
// posejump.test.mjs checks the blend's own trajectory. This file checks the
// thing that actually matters: the stream of poses the 20 Hz executor commands
// while playing a timeline, with the blend included and the last pose HELD
// across gaps. That distinction is the whole trap in this subsystem —
//
//   · `track.jointsAt(t)` returns null in a gap, which HIDES the discontinuity;
//     a harness built on it measures nothing and reports success.
//   · `samplePoseBefore` models the hold but deliberately looks THROUGH blends
//     (skipXfade) because that is the DETECTOR's question. Measuring the
//     executor with it reports the unguarded velocity even when the guard is
//     working perfectly — which is exactly the false negative that hid this
//     bug: the whole feature threw ReferenceError on every call and the
//     timeline shipped commanding 29 rad/s.
//
// So: sample `jointsAt` INCLUDING blends, and carry the hold yourself.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MOTION_CLIPS, ChoreoTrack, autoFixPoseJumps, detectPoseJumps,
         SAFE_BLEND_VEL, MAX_JOINT_VEL, BLEND_MIN_DUR, BLEND_THRESH,
         EXEC_RATE } from '../../web/js/motions.js';

const DT = 1 / 20;                     // the executor's tick

/** The pose stream the executor commands: blends included, gaps held. */
function commanded(track, until = 16){
  const rows = [];
  let held = null;
  for (let t = 0; t <= until + 1e-9; t += DT){
    const j = track.jointsAt(t);
    if (j) held = { ...(held || {}), ...j };
    rows.push({ t, p: held ? { ...held } : null });
  }
  return rows;
}

/** Worst per-joint speed over [lo,hi] of a commanded stream. */
function peak(rows, lo = 0, hi = Infinity){
  let best = 0, at = 0, joint = null;
  for (let i = 1; i < rows.length; i++){
    const a = rows[i - 1].p, b = rows[i].p;
    if (!a || !b || rows[i].t < lo || rows[i].t > hi) continue;
    for (const k in b){
      if (a[k] == null) continue;
      const v = Math.abs(b[k] - a[k]) / DT;
      if (v > best){ best = v; at = rows[i].t; joint = k; }
    }
  }
  return { peak: best, at, joint };
}

// T-pose held to 4 s, then arms-up at 8 s: a 4 s dead gap the executor holds
// through, then a large pose delta commanded in a single tick.
function jumpyTrack(){
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [a] = tr.add('pose_tpose', 0);
  a.out = 4;
  tr.add('pose_arms_up', 8);
  return tr;
}

test('an unguarded pose jump really is the hazard (the thing being fixed)', () => {
  const p = peak(commanded(jumpyTrack()));
  assert.ok(p.peak > 10,
    `expected a violent jump to measure against, got ${p.peak.toFixed(2)} rad/s`);
  assert.ok(Math.abs(p.at - 8) < 0.2, `jump should land at the block start, got ${p.at}`);
});

test('the blend holds the commanded stream inside the blend budget', () => {
  const tr = jumpyTrack();
  const r = autoFixPoseJumps(tr);
  assert.equal(r.inserted.length, 1, 'a blend should have been inserted');
  const b = tr.blocks.find((x) => x.key === 'blend');
  const rows = commanded(tr);
  const inBlend = peak(rows, b.t0, tr.end(b));
  assert.ok(inBlend.peak <= SAFE_BLEND_VEL + 1e-2,
    `blend commands ${inBlend.peak.toFixed(3)} rad/s, over the ${SAFE_BLEND_VEL} budget`);
});

test('no commanded velocity anywhere on a guarded timeline exceeds the clip budget', () => {
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  const p = peak(commanded(tr));
  assert.ok(p.peak <= MAX_JOINT_VEL,
    `${p.joint} hits ${p.peak.toFixed(2)} rad/s at ${p.at.toFixed(2)}s`);
});

test('the guard cuts the peak by more than an order of magnitude', () => {
  const before = peak(commanded(jumpyTrack())).peak;
  const tr = jumpyTrack(); autoFixPoseJumps(tr);
  const after = peak(commanded(tr)).peak;
  assert.ok(before / after > 10,
    `only ${(before / after).toFixed(1)}× reduction (${before.toFixed(1)} → ${after.toFixed(1)})`);
});

test('reconciliation converges — repeated passes are a no-op', () => {
  // Non-convergence was the live bug: a blend erased the very jump that
  // justified it (jointsAt ignored skipXfade), so the next pass deleted it and
  // the pass after that re-inserted it. A timeline would flip between guarded
  // and unguarded on alternate edits.
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  for (let i = 0; i < 5; i++){
    const r = autoFixPoseJumps(tr);
    assert.equal(r.inserted.length + r.updated.length + r.removed.length, 0,
      `pass ${i + 2} still churned blocks`);
  }
  assert.equal(tr.blocks.filter((b) => b.key === 'blend').length, 1,
    'exactly one blend should survive');
});

test('a blend stays inside budget across a sweep of jump sizes', () => {
  for (const key of ['pose_arms_up', 'pose_tpose', 'pose_bow', 'pose_tall', 'pose_reach_fwd']){
    for (const first of ['pose_tpose', 'pose_arms_up', 'pose_bow']){
      if (first === key) continue;
      const tr = new ChoreoTrack(MOTION_CLIPS);
      const [a] = tr.add(first, 0);
      a.out = 4;
      tr.add(key, 12);                       // room for the longest safe blend
      autoFixPoseJumps(tr);
      const b = tr.blocks.find((x) => x.key === 'blend');
      if (!b) continue;                      // sub-threshold pairs need no blend
      const inBlend = peak(commanded(tr), b.t0, tr.end(b));
      assert.ok(inBlend.peak <= SAFE_BLEND_VEL + 1e-2,
        `${first}→${key}: ${inBlend.peak.toFixed(3)} rad/s over budget`);
      assert.ok(b.out - b.in >= BLEND_MIN_DUR - 1e-9, `${first}→${key}: under the floor`);
      assert.ok(b.t0 >= tr.end(a) - 1e-6,
        `${first}→${key}: blend starts inside the block it blends away from`);
    }
  }
});

// THE REGRESSION THAT MATTERS: a block ENDING is a boundary too.
//
// The studio's own demo timeline layers `pose_tall` [2.5, 6.2] over a running
// `pickup` [0, 8]. At 6.2 the pose stops and `pickup` takes the joints back
// mid-carry. The guard used to look only at block STARTS, so it reported no
// jump here at all -- and the executor commanded 31.75 rad/s, 13x the clip
// budget, on the app's default document.
test('a block ENDING mid-motion is a jump, and it is bridged', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('pickup', 0);                       // runs underneath, 8 s
  tr.add('pose_tall', 2.5);                  // layered on top, ends at 6.2
  const before = peak(commanded(tr), 0, 12);
  assert.ok(before.peak > 10,
    `the unguarded handover should be violent, measured ${before.peak.toFixed(2)}`);

  autoFixPoseJumps(tr);
  const tail = tr.blocks.find((b) => b.key === 'blend' && b.params.edge === 'tail');
  assert.ok(tail, 'a trailing blend should be inserted where the pose ends');
  assert.ok(Math.abs(tail.t0 - 6.2) < 1e-6, 'it starts exactly at the handover');
  const after = peak(commanded(tr), 0, 12);
  assert.ok(after.peak <= MAX_JOINT_VEL + 1e-2,
    `guarded run still commands ${after.peak.toFixed(2)} rad/s at t=${after.at}`);
});

// A blend is continuous at BOTH of its own edges because each endpoint is
// sampled from the signal at that edge -- which is what lets it overlap the
// clip it is leaving instead of needing a gap to live in.
test('a blend never steps at its own edges, even with no gap to sit in', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [a] = tr.add('pickup', 0);
  tr.add('wave', tr.end(a));                 // back to back: zero room
  autoFixPoseJumps(tr);
  const b = tr.blocks.find((x) => x.key === 'blend');
  assert.ok(b, 'a blend should still be inserted');
  assert.ok(b.t0 < tr.end(a) - 1e-6, 'with no gap it must overlap the outgoing clip');
  const p = peak(commanded(tr), 0, tr.duration() + 1);
  assert.ok(p.peak <= MAX_JOINT_VEL + 1e-2,
    `back-to-back run commands ${p.peak.toFixed(2)} rad/s at t=${p.at} (${p.joint})`);
});

// The one placement rule left: a blend may not span another block's START,
// because that block begins later and out-ranks it halfway through.
test('a blend never swallows another block\'s start', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('pose_arms_up', 0);
  tr.add('pose_bow', 8);
  tr.add('pose_tpose', 9);                   // starts inside pose_bow's lead blend
  autoFixPoseJumps(tr);
  const starts = tr.blocks
    .filter((b) => MOTION_CLIPS[b.key].kind !== 'xfade')
    .map((b) => b.t0);
  for (const b of tr.blocks.filter((x) => x.key === 'blend'))
    for (const s of starts)
      assert.ok(!(s > b.t0 + 1e-6 && s < tr.end(b) - 1e-6),
        `blend [${b.t0.toFixed(2)}, ${tr.end(b).toFixed(2)}] spans a start at ${s}`);
});

// The whole library, both edges, against the budget: compose every pair of a
// hold clip and a follower and assert the guarded run never leaves the budget.
test('every hold clip can be followed by anything without leaving the budget', () => {
  const holds = ['pickup', 'pose_arms_up', 'pose_tpose', 'pose_carry', 'sleep', 'lift_up'];
  const next = ['wave', 'pose_home', 'celebrate', 'pose_reach_fwd'];
  for (const h of holds) {
    for (const n of next) {
      const tr = new ChoreoTrack(MOTION_CLIPS);
      const [a] = tr.add(h, 0);
      if (!a) continue;
      tr.add(n, tr.end(a) + 1);
      autoFixPoseJumps(tr);
      const p = peak(commanded(tr), 0, tr.duration() + 2);
      assert.ok(p.peak <= MAX_JOINT_VEL + 1e-2,
        `${h} -> ${n}: ${p.peak.toFixed(2)} rad/s at t=${p.at} (${p.joint})`);
    }
  }
});


// THE RULE OF THUMB, stated as a test: whatever you compose, the pose the
// executor commands is continuous. Not "the library is safe" (that is per clip)
// and not "this junction is safe" (that is one case) — EVERY junction, for every
// pair, at the rate the robot receives.
//
// Cutting is what exposes it. A clip cut in half ends wherever it happened to be
// mid-motion, so the guard cannot lean on "clips start and end at rest": both
// the outgoing pose and the gap are arbitrary.
test('any clip cut in half, followed by any other, stays inside the budget', () => {
  const keys = Object.entries(MOTION_CLIPS)
    .filter(([, c]) => c.status === 'available' && c.joints && c.kind !== 'xfade'
                    && !Object.keys(c.params || {}).length)
    .map(([k]) => k);
  // A representative spread rather than all 3600 pairs: the full sweep lives in
  // web_gui/control_center/tools/exp_pose_library.py, this is the gate.
  const from = ['wave', 'pickup', 'happy', 'pose_tuck', 'lift_up', 'tai_chi'];
  const onto = ['wave', 'pose_home', 'celebrate', 'angry', 'pose_tuck', 'lift_down'];
  for (const k of from.filter((k) => keys.includes(k))) {
    for (const j of onto.filter((j) => keys.includes(j))) {
      for (const gap of [0, 0.3, 1, 3]) {
        const tr = new ChoreoTrack(MOTION_CLIPS);
        const [a] = tr.add(k, 0);
        const half = tr.end(a) / 2;
        const right = tr.cut(a.id, half);
        if (right) tr.remove(right.id);        // ends mid-motion, like a real cut
        tr.add(j, half + gap);
        autoFixPoseJumps(tr);
        const p = peak(commanded(tr), 0, tr.duration() + 2);
        assert.ok(p.peak <= MAX_JOINT_VEL + 1e-9,
          `${k} cut, then ${j} after ${gap}s: ${p.peak.toFixed(2)} rad/s ` +
          `at t=${p.at} on ${p.joint}`);
      }
    }
  }
});

// The bridge must OWN the pose while it runs. Another lane keeping part of it
// and then stopping is the same reveal as a jump: `lift_up` is lane 'nav' but
// commands joints, and while it overrode the lift underneath a bridge the
// bridge never actually moved the lift — 19.9 rad/s the tick it ended.
test('a bridge is not overridden by another lane while it runs', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [lift] = tr.add('lift_up', 0);           // lane 'nav', commands the lift
  tr.add('wave', tr.end(lift) + 1);
  autoFixPoseJumps(tr);
  const p = peak(commanded(tr), 0, tr.duration() + 2);
  assert.ok(p.peak <= MAX_JOINT_VEL + 1e-9,
    `lift_up -> wave: ${p.peak.toFixed(2)} rad/s at t=${p.at} on ${p.joint}`);
});

// The threshold below which a jump is left alone is DERIVED from the budget, not
// chosen: one tick of it must still be inside the budget.
test('the benign-jump threshold is one tick of the velocity budget', () => {
  assert.equal(BLEND_THRESH, MAX_JOINT_VEL / EXEC_RATE);
  assert.ok(BLEND_THRESH * EXEC_RATE <= MAX_JOINT_VEL + 1e-9,
    'a sub-threshold jump must be absorbable in one command tick');
});

test('a benign jump is left alone (no blend spam)', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('wave', 0);                          // starts and ends at neutral
  tr.add('happy', 6);
  const r = autoFixPoseJumps(tr);
  assert.equal(r.inserted.length, 0);
  assert.equal(detectPoseJumps(tr).length, 0);
});

test('the blend is not addable from the clip palette', () => {
  // It is only meaningful anchored to a jump, with from/to baked in by the
  // detector. It stays out of the menu by CATEGORY (no section matches
  // 'transition'), so guard that rather than its status.
  assert.equal(MOTION_CLIPS.blend.category, 'transition');
  assert.ok(!['animation', 'policy', 'nav', 'pose'].includes(MOTION_CLIPS.blend.category),
    'a menu section would pick the blend up');
});
