// POV transition choreography: schedule correctness + path smoothness.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { DURATION_MS, easeInOut, povSchedule, povPose, maxStep }
  from '../../web/js/transition.js';

const A = { pos: new THREE.Vector3(0.2, 1.1, 0), fov: 83.3,
            quat: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.4) };
const B = { pos: new THREE.Vector3(2.0, 1.5, 2.4), fov: 42,
            quat: new THREE.Quaternion() };

test('duration is the requested 2 seconds', () => {
  assert.equal(DURATION_MS, 2000);
});

test('easeInOut is clamped, monotone, and endpoint-exact', () => {
  assert.equal(easeInOut(-1), 0);
  assert.equal(easeInOut(0), 0);
  assert.equal(easeInOut(1), 1);
  assert.equal(easeInOut(2), 1);
  let prev = 0;
  for (let i = 1; i <= 100; i++){
    const v = easeInOut(i / 100);
    assert.ok(v >= prev); prev = v;
  }
});

test('to3rd: footage fades out fully BEFORE the flight finishes', () => {
  assert.equal(povSchedule('to3rd', 0).a, 1);         // starts opaque
  assert.equal(povSchedule('to3rd', 0).f, 0);         // camera holds
  assert.equal(povSchedule('to3rd', 0.2).f, 0);       // still holding during fade
  assert.equal(povSchedule('to3rd', 0.5).a, 0);       // fully faded
  assert.ok(povSchedule('to3rd', 0.6).f > 0.2);       // now flying
  assert.equal(povSchedule('to3rd', 1).f, 1);
});

test('to1st: flight lands BEFORE the footage fades in', () => {
  assert.equal(povSchedule('to1st', 0).a, 0);
  assert.ok(povSchedule('to1st', 0.5).f > 0.5);
  assert.equal(povSchedule('to1st', 0.75).f, 1);      // arrived at head pose
  assert.ok(povSchedule('to1st', 0.8).a > 0);         // fading in
  assert.equal(povSchedule('to1st', 1).a, 1);
});

test('povPose hits both endpoints exactly and arcs upward between', () => {
  const p0 = povPose(A, B, 0), p1 = povPose(A, B, 1);
  assert.ok(p0.pos.distanceTo(A.pos) < 1e-9);
  assert.ok(p1.pos.distanceTo(B.pos) < 1e-9);
  assert.equal(p0.fov, A.fov);
  assert.equal(p1.fov, B.fov);
  const mid = povPose(A, B, 0.5);
  assert.ok(mid.pos.y > (A.pos.y + B.pos.y) / 2 + 0.2, 'arc lifts the midpoint');
});

test('60 fps path has no jumps (max per-frame step bounded)', () => {
  const dist = A.pos.distanceTo(B.pos);
  const worst = maxStep(A, B, 120);
  // smoothstep peak velocity = 1.5×average; arc adds a little more
  assert.ok(worst < (dist / 120) * 2.5, `worst step ${worst}`);
  assert.ok(worst > 0);
});

test('quaternion slerp stays normalized along the path', () => {
  for (let i = 0; i <= 10; i++){
    const q = povPose(A, B, i / 10).quat;
    assert.ok(Math.abs(q.length() - 1) < 1e-6);
  }
});
