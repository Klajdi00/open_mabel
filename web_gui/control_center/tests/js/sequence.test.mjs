// Setpoint sequence model: ordering, interpolation, persistence.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SetpointSeq, lerpPose } from '../../web/js/sequence.js';

test('add keeps keys sorted; duration = last key', () => {
  const s = new SetpointSeq();
  s.add(3, { a: 1 });
  s.add(1, { a: 0 });
  s.add(2, { a: 0.5 });
  assert.deepEqual(s.keys.map(k => k.t), [1, 2, 3]);
  assert.equal(s.duration(), 3);
});

test('poseAt holds before first and after last, lerps between', () => {
  const s = new SetpointSeq();
  s.add(1, { j: 0 });
  s.add(3, { j: 1 });
  assert.equal(s.poseAt(0).j, 0);
  assert.equal(s.poseAt(5).j, 1);
  assert.ok(Math.abs(s.poseAt(2).j - 0.5) < 1e-9);
  assert.ok(Math.abs(s.poseAt(1.5).j - 0.25) < 1e-9);
});

test('poseAt on empty sequence is null', () => {
  assert.equal(new SetpointSeq().poseAt(1), null);
});

test('move re-sorts and clamps to t>=0', () => {
  const s = new SetpointSeq();
  const a = s.add(1, { j: 0 });
  s.add(2, { j: 1 });
  s.move(a, 5);
  assert.deepEqual(s.keys.map(k => k.t), [2, 5]);
  s.move(a, -3);
  assert.equal(s.keys[0].t, 0);
});

test('remove deletes exactly the key', () => {
  const s = new SetpointSeq();
  const a = s.add(1, { j: 0 });
  s.add(2, { j: 1 });
  s.remove(a);
  assert.equal(s.keys.length, 1);
  s.remove(a);                                   // double-remove is a no-op
  assert.equal(s.keys.length, 1);
});

test('JSON roundtrip', () => {
  const s = new SetpointSeq();
  s.add(0, { left_arm_1: 0.3, torso: -0.1 }, 'reach');
  s.add(2.5, { left_arm_1: 0.9 });
  const back = SetpointSeq.fromJSON(JSON.parse(JSON.stringify(s.toJSON())));
  assert.equal(back.keys.length, 2);
  assert.equal(back.keys[0].label, 'reach');
  assert.equal(back.poseAt(2.5).left_arm_1, 0.9);
  assert.equal(SetpointSeq.fromJSON(null).keys.length, 0);
});

test('lerpPose handles disjoint joint sets', () => {
  const out = lerpPose({ a: 0, b: 2 }, { a: 1 }, 0.5);
  assert.equal(out.a, 0.5);
  assert.equal(out.b, 2);                        // held from a
  const out2 = lerpPose({ }, { c: 4 }, 0.25);
  assert.equal(out2.c, 4);                       // held from b
});

test('onChange fires for add/move/remove', () => {
  const s = new SetpointSeq();
  let n = 0;
  s.onChange = () => n++;
  const k = s.add(1, {});
  s.move(k, 2);
  s.remove(k);
  assert.equal(n, 3);
});
