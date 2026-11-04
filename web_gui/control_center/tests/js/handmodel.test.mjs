// ORCA hand model: ranges, percent⇄rad, firmware poses, grasp blend, wire cmd.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { HAND_JOINTS, RANGE, REF, FINGERS, POSES, pctToRad, radToPct,
         graspPose, poseToCommand, rangeFor, refFor } from '../../web/js/handmodel.js';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

function findRepoRoot(from){
  let d = from;
  for (let i = 0; i < 10; i++){
    if (existsSync(join(d, 'web_gui', 'shared_assets', 'mabel_joints.json'))) return d;
    const up = dirname(d);
    if (up === d) break;
    d = up;
  }
  throw new Error('mabel_joints.json not found above ' + from);
}
const REPO = findRepoRoot(dirname(fileURLToPath(import.meta.url)));
const MANIFEST = JSON.parse(
  readFileSync(join(REPO, 'web_gui', 'shared_assets', 'mabel_joints.json'), 'utf8'));
const MJ = Object.fromEntries(MANIFEST.joints.map(j => [j.name, j]));

test('17 joints, all with sane ranges', () => {
  assert.equal(HAND_JOINTS.length, 17);
  for (const [n, lo, hi] of HAND_JOINTS) assert.ok(hi > lo, n);
});

test('pct↔rad roundtrip and endpoints, per hand', () => {
  for (const side of ['left', 'right'])
    for (const [n] of HAND_JOINTS){
      const [lo, hi] = rangeFor(n, side);
      assert.ok(Math.abs(pctToRad(n, 0, side) - lo) < 1e-9, `${side} ${n}`);
      assert.ok(Math.abs(pctToRad(n, 100, side) - hi) < 1e-9, `${side} ${n}`);
      for (const pct of [0, 25, 50, 99])
        assert.ok(Math.abs(radToPct(n, pctToRad(n, pct, side), side) - pct) < 1e-6,
          `${side} ${n}`);
    }
});

test('pct clamps outside 0..100', () => {
  const [lo, hi] = rangeFor('wrist', 'left');
  assert.ok(Math.abs(pctToRad('wrist', -20) - lo) < 1e-9);
  assert.ok(Math.abs(pctToRad('wrist', 200) - hi) < 1e-9);
});

// ── the drift guard ────────────────────────────────────────────────────────
// `mabel_joints.json` is GENERATED from the MJCF (urdf_to_glb.py), so it is the
// source of truth for both the travel range and `ref`. This table is hand-kept,
// and it HAD drifted: thumb_pip/thumb_dip ranges were transposed, and `ref` was
// never applied — so every ref != 0 joint was commanded `ref` off intent (wrist
// 8.6°, index_abd 22.9°, pinky_abd 30.0°), thumb_abd landed outside its
// ctrlrange and clamped to its hard stop, and the mirrored wrists were sent the
// same absolute angle. That is the "hands are messed up after a reset" bug.
// Lock the table to the manifest so it cannot recur.
test('ranges + refs match the generated manifest, both hands', () => {
  for (const side of ['left', 'right'])
    for (const [n] of HAND_JOINTS){
      const j = MJ[`${side}_${n}`];
      assert.ok(j, `${side}_${n} missing from mabel_joints.json`);
      assert.ok(Math.abs(refFor(n, side) - j.ref) < 1e-5,
        `${side}_${n} ref ${refFor(n, side)} != manifest ${j.ref}`);
      const [lo, hi] = rangeFor(n, side);
      assert.ok(Math.abs(lo - (j.lower + j.ref)) < 1e-4,
        `${side}_${n} lower ${lo} != manifest ${j.lower + j.ref}`);
      assert.ok(Math.abs(hi - (j.upper + j.ref)) < 1e-4,
        `${side}_${n} upper ${hi} != manifest ${j.upper + j.ref}`);
    }
});

// The neutral/assembled pose must be commandable and must sit INSIDE the range.
test('ref (the level, assembled pose) is reachable on both hands', () => {
  for (const side of ['left', 'right'])
    for (const [n] of HAND_JOINTS){
      const [lo, hi] = rangeFor(n, side);
      const ref = refFor(n, side);
      assert.ok(ref >= lo - 1e-9 && ref <= hi + 1e-9,
        `${side}_${n}: ref ${ref} outside [${lo}, ${hi}]`);
    }
});

// The regression itself, stated as a number: a "neutral" wrist command must put
// the wrist at its LEVEL pose, not 24 degrees off it.
test('a neutral wrist command lands near level on both hands', () => {
  for (const side of ['left', 'right']){
    const ref = refFor('wrist', side);
    const pctOfLevel = radToPct('wrist', ref, side);
    const err = Math.abs(pctToRad('wrist', pctOfLevel, side) - ref);
    assert.ok(err < 1e-6, `${side} wrist neutral round-trip off by ${err}`);
    // and the two hands must be mirror images
    if (side === 'right')
      assert.ok(Math.abs(ref + refFor('wrist', 'left')) < 1e-9, 'wrists mirror');
  }
});

test('all 9 firmware poses cover all 17 joints', () => {
  const names = HAND_JOINTS.map(([n]) => n);
  assert.equal(Object.keys(POSES).length, 9);
  for (const [pname, pose] of Object.entries(POSES)){
    assert.deepEqual(Object.keys(pose).sort(), [...names].sort(), pname);
    for (const v of Object.values(pose)) assert.ok(v >= 0 && v <= 100, pname);
  }
});

test('firmware pose spot checks (poses.py values)', () => {
  assert.equal(POSES.fist.index_pip, 85);
  assert.equal(POSES.fist.thumb_mcp, 70);      // ORCA thumb_cmc
  assert.equal(POSES.thumbs_up.thumb_mcp, 5);
  assert.equal(POSES.peace.ring_pip, 85);
  assert.equal(POSES.spider.middle_abd, 50);
  assert.equal(POSES.ok.thumb_abd, 70);
});

test('grasp blend: 0 = flat, 1 = fist, monotone in between', () => {
  assert.deepEqual(graspPose(0), POSES.flat);
  assert.deepEqual(graspPose(1), POSES.fist);
  const mid = graspPose(0.5);
  assert.equal(mid.index_pip, (10 + 85) / 2);
  assert.equal(graspPose(-1).index_pip, POSES.flat.index_pip);   // clamped
});

test('fingers partition the 17 joints', () => {
  const all = Object.values(FINGERS).flat().sort();
  assert.deepEqual(all, HAND_JOINTS.map(([n]) => n).sort());
});

test('poseToCommand emits side-prefixed ABSOLUTE radian targets in range', () => {
  for (const side of ['left', 'right']){
    const { name, position } = poseToCommand(side, POSES.fist);
    assert.equal(name.length, 17);
    for (let i = 0; i < name.length; i++){
      assert.ok(name[i].startsWith(`${side}_`), name[i]);
      const suffix = name[i].slice(side.length + 1);
      // the wire carries ABSOLUTE qpos, so the bound is the manifest's
      // ref-shifted range — not the relative travel in RANGE
      const [lo, hi] = rangeFor(suffix, side);
      assert.ok(position[i] >= lo - 1e-9 && position[i] <= hi + 1e-9,
        `${name[i]} = ${position[i]} outside [${lo}, ${hi}]`);
      const j = MJ[name[i]];
      assert.ok(position[i] >= j.lower + j.ref - 1e-4
             && position[i] <= j.upper + j.ref + 1e-4,
        `${name[i]} outside the MJCF ctrlrange`);
    }
  }
  // unknown joints dropped, not crashed
  const out = poseToCommand('right', { bogus: 50, wrist: 50 });
  assert.deepEqual(out.name, ['right_wrist']);
});
