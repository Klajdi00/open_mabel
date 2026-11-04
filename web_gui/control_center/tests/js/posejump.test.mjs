// Pose-jump safety: detection of pose discontinuities between manipulation
// blocks, auto-inserted blend transitions (duration floored by pose distance),
// blend velocity cap, min-duration trim enforcement, and reconciliation
// (re-anchor on move, removal when the jump disappears).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MOTION_CLIPS, ChoreoTrack, POSE_TARGETS, clipJointsAt,
         detectPoseJumps, autoFixPoseJumps, poseDelta, blendMinDur,
         samplePoseBefore, makeBlendBlock, ease,
         SAFE_BLEND_VEL, EASE_PEAK, BLEND_THRESH, BLEND_MIN_DUR } from '../../web/js/motions.js';

const FS = 20, DT = 1 / FS;

// two held poses with a large jump between them: T-pose … arms-up
function jumpyTrack(){
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [a] = tr.add('pose_tpose', 0);
  a.out = 4;                                   // hold clip stretched: pose holds
  tr.add('pose_arms_up', 8);
  return tr;
}

test('detectPoseJumps finds the T-pose → arms-up discontinuity', () => {
  const jumps = detectPoseJumps(jumpyTrack());
  assert.equal(jumps.length, 1);
  const j = jumps[0];
  assert.equal(j.t, 8);
  assert.ok(j.delta > BLEND_THRESH, 'delta above threshold');
  assert.ok(j.minDur >= EASE_PEAK * j.delta / SAFE_BLEND_VEL - 1e-6, 'min from distance');
  assert.ok(j.minDur >= BLEND_MIN_DUR);
});

// The duration law is only correct if EASE_PEAK really is the peak/average
// velocity ratio of `ease`. Measure it rather than trusting the constant —
// swapping the easing without updating it would silently break the cap.
test('EASE_PEAK matches the easing actually used', () => {
  let peak = 0;
  const N = 20000;
  for (let i = 1; i <= N; i++)
    peak = Math.max(peak, (ease(i / N) - ease((i - 1) / N)) * N);
  assert.ok(Math.abs(peak - EASE_PEAK) < 1e-3,
    `measured ${peak.toFixed(5)} vs EASE_PEAK ${EASE_PEAK.toFixed(5)}`);
});

test('a continuous sequence (animation from/to neutral) needs no blend', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('wave', 0);                            // starts and ends at neutral
  tr.add('happy', 6);                           // also starts at neutral
  assert.equal(detectPoseJumps(tr).length, 0);
});

test('autoFixPoseJumps inserts a blend that ends exactly at the next block', () => {
  const tr = jumpyTrack();
  const r = autoFixPoseJumps(tr);
  assert.equal(r.inserted.length, 1);
  const blend = tr.blocks.find((b) => b.key === 'blend');
  assert.ok(blend, 'blend inserted');
  assert.ok(Math.abs(tr.end(blend) - 8) < 1e-6, 'anchored to the jump');
  assert.ok(blend.out - blend.in >= blend.params.min - 1e-9);
  // idempotent: a second pass changes nothing
  const r2 = autoFixPoseJumps(tr);
  assert.equal(r2.inserted.length + r2.updated.length + r2.removed.length, 0);
});

test('the blend keeps every joint under the safe velocity at 20 Hz', () => {
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  const blend = tr.blocks.find((b) => b.key === 'blend');
  const clip = MOTION_CLIPS.blend;
  let prev = null;
  for (let t = 0; t <= blend.out + 1e-9; t += DT){
    const j = clipJointsAt(clip, t, blend.params);
    if (prev){
      for (const k in j){
        const v = Math.abs(j[k] - prev[k]) / DT;
        assert.ok(v <= SAFE_BLEND_VEL + 0.05, `${k} vel ${v.toFixed(2)} rad/s`);
      }
    }
    prev = j;
  }
  // the blend lands ON the next block's start pose (no residual jump)
  const end = clipJointsAt(clip, blend.out, blend.params);
  const start = clipJointsAt(MOTION_CLIPS.pose_arms_up, 1e-3);
  assert.ok(poseDelta(end, start).delta < 1e-6, 'blend end == next block start');
});

test('trim can stretch a blend but never below the safety minimum', () => {
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  const blend = tr.blocks.find((b) => b.key === 'blend');
  const min = blend.params.min;
  tr.trim(blend.id, 'l', 8 - min / 2);          // try to make it half the minimum
  assert.ok(blend.out - blend.in >= min - 1e-9, 'floored at min');
  tr.trim(blend.id, 'l', 0);                    // stretch it way out
  assert.ok(blend.out - blend.in > min, 'stretch allowed');
  assert.equal(blend.params.userDur, blend.out - blend.in, 'user duration kept');
  // reconcile keeps the user's longer duration
  autoFixPoseJumps(tr);
  assert.ok(Math.abs((blend.out - blend.in) - blend.params.userDur) < 1e-6);
});

test('moving the protected block re-anchors the blend; removing it removes it', () => {
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  const target = tr.blocks.find((b) => b.key === 'pose_arms_up');
  tr.move(target.id, 12);
  autoFixPoseJumps(tr);
  const blend = tr.blocks.find((b) => b.key === 'blend');
  assert.ok(Math.abs(tr.end(blend) - 12) < 1e-6, 're-anchored to 12s');
  tr.remove(target.id);
  autoFixPoseJumps(tr);
  assert.ok(!tr.blocks.some((b) => b.key === 'blend'), 'orphan blend removed');
});

test('blends cannot be bladed (a half-blend is a new jump)', () => {
  const tr = jumpyTrack();
  autoFixPoseJumps(tr);
  const blend = tr.blocks.find((b) => b.key === 'blend');
  assert.equal(tr.cut(blend.id, blend.t0 + (blend.out - blend.in) / 2), null);
});

test('samplePoseBefore holds the last activity pose across a dead gap', () => {
  const tr = jumpyTrack();                      // T-pose ends (held) at 4 s
  const p = samplePoseBefore(tr, null, 8);      // sampled in the 4–8 s gap
  assert.ok(p, 'pose found');
  assert.ok(Math.abs(p.left_arm_2 - POSE_TARGETS.tpose.left_arm_2) < 1e-6,
    'held T-pose arm');
});

test('makeBlendBlock floors the duration at the jump minimum', () => {
  const [j] = detectPoseJumps(jumpyTrack());
  const b = makeBlendBlock(j, 0.01);
  assert.ok(b.out - b.in >= j.minDur - 1e-9);
});
