// IK: 3×3 solver + damped-least-squares convergence on a synthetic 3-joint arm
// built from real THREE Object3Ds mimicking the Rig contract.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { solveIK, solve3 } from '../../web/js/kinematics.js';

test('solve3 solves a known system', () => {
  const A = [[2, 0, 0], [0, 3, 0], [0, 0, 4]];
  const y = solve3(A, [2, 6, 8]);
  assert.deepEqual(y.map(v => Math.round(v * 1e9) / 1e9), [1, 2, 2]);
  assert.equal(solve3([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [1, 1, 1]), null);
});

// minimal Rig-compatible fixture: base → j1(revolute z) → link(1m x) →
// j2(revolute z) → link(1m x) → j3(revolute y) → ee(0.5m x)
function makeArm(){
  const root = new THREE.Group(); root.name = 'root';
  const world = new THREE.Group(); world.add(root);
  const mk = (name, x) => {
    const o = new THREE.Object3D(); o.name = name; o.position.set(x, 0, 0);
    return o;
  };
  const j1 = mk('j1', 0), j2 = mk('j2', 1), j3 = mk('j3', 1), ee = mk('ee', 0.5);
  root.add(j1); j1.add(j2); j2.add(j3); j3.add(ee);
  const joints = {};
  for (const [name, node, axis, lo, hi] of [
    ['j1', j1, [0, 0, 1], -Math.PI, Math.PI],
    ['j2', j2, [0, 0, 1], -Math.PI, Math.PI],
    ['j3', j3, [0, 1, 0], -Math.PI, Math.PI],
  ]){
    joints[name] = { type: 'revolute', lower: lo, upper: hi, _n: node,
      _p0: node.position.clone(), _q0: node.quaternion.clone(),
      _ax: new THREE.Vector3(...axis) };
  }
  const rig = {
    root, world, joints,
    setJoint(name, val){
      const j = this.joints[name];
      j._n.quaternion.copy(j._q0)
        .multiply(new THREE.Quaternion().setFromAxisAngle(j._ax, val));
      j._n.position.copy(j._p0);
    },
  };
  world.updateMatrixWorld(true);
  return rig;
}

test('solveIK converges to a reachable target', () => {
  const rig = makeArm();
  const target = new THREE.Vector3(1.2, 1.2, 0.4);      // inside 2.5m reach
  let sol = null;
  for (let i = 0; i < 40; i++)                          // iterative drag ticks
    sol = solveIK(rig, ['j1', 'j2', 'j3'], 'ee', target,
      { iters: 8, maxErr: 10 }) || sol;                 // cold solve: no drag guard
  assert.ok(sol, 'solution returned');
  const p = new THREE.Vector3();
  rig.root.getObjectByName('ee').getWorldPosition(p);
  assert.ok(p.distanceTo(target) < 0.03, `residual ${p.distanceTo(target)}`);
});

test('solveIK respects joint limits', () => {
  const rig = makeArm();
  rig.joints.j1.lower = -0.2; rig.joints.j1.upper = 0.2;
  const target = new THREE.Vector3(0, 2.0, 0);          // wants j1 ≈ +90°
  let sol = null;
  for (let i = 0; i < 30; i++)
    sol = solveIK(rig, ['j1', 'j2', 'j3'], 'ee', target,
      { iters: 8, maxErr: 10 }) || sol;
  assert.ok(sol);
  assert.ok(sol.j1 <= 0.2 + 1e-9 && sol.j1 >= -0.2 - 1e-9);
});

test('solveIK bails on far-out-of-reach targets', () => {
  const rig = makeArm();
  const sol = solveIK(rig, ['j1', 'j2', 'j3'], 'ee',
    new THREE.Vector3(50, 0, 0));
  assert.equal(sol, null);
});

test('solveIK returns null for unknown joints or ee', () => {
  const rig = makeArm();
  assert.equal(solveIK(rig, ['nope'], 'ee', new THREE.Vector3()), null);
  assert.equal(solveIK(rig, ['j1'], 'missing_ee', new THREE.Vector3()), null);
});

// ── the REAL 7-DOF MABEL arm ────────────────────────────────────────────────
// The fixtures above are a 3-joint toy arm, which has no null space and so could
// never have caught the bug this section guards: on the real 7-DOF arm a
// position-only solve leaves 4 unconstrained dimensions, and what filled them was
// the operator's mouse path. Measured before the fix (experiments/exp_browser_ik.mjs):
// the same target reached by four different drag paths gave arm solutions up to
// 168° apart, and the palm tumbled through 22° on a drag that was pure translation.
import { buildRig, restPose, ARM, nodePose } from './fixtures/rig.mjs';
import { REST_POSE } from '../../web/js/homepose.js';

const EE = 'right_palm__anchor';
const CHAIN = ARM('right');
const DEG = 180 / Math.PI;

function posedRig(){
  const rig = buildRig();
  const q = { ...restPose(), ...REST_POSE };
  rig.pose(q);
  rig._q = {};
  for (const n in q) rig._q[n] = q[n];
  return rig;
}

// Drag the wrist ball from where it is to `goal`, holding the palm's orientation,
// exactly as viewport3d._solveArm does.
function dragTo(rig, goal, ticks = 60){
  const start = nodePose(rig, EE);
  const opts = { rest: REST_POSE, targetQuat: start.quat.clone() };
  const t = new THREE.Vector3();
  let sol = null;
  for (let k = 1; k <= ticks; k++){
    t.lerpVectors(start.pos, goal, k / ticks);
    sol = solveIK(rig, CHAIN, EE, t, opts) || sol;
  }
  return { sol, start };
}

test('real arm: a pure-translation drag does not tumble the palm', () => {
  const rig = posedRig();
  const start = nodePose(rig, EE);
  const goal = start.pos.clone().add(new THREE.Vector3(0.22, 0, -0.10));
  dragTo(rig, goal);
  const end = nodePose(rig, EE);
  const rot = 2 * Math.acos(Math.min(1, Math.abs(end.quat.dot(start.quat)))) * DEG;
  assert.ok(end.pos.distanceTo(goal) < 0.08, `residual ${end.pos.distanceTo(goal)}`);
  assert.ok(rot < 15, `palm rotated ${rot.toFixed(1)}° on a translation drag`);
});

test('real arm: the solution does not depend on the drag PATH', () => {
  const goalOf = (r) => nodePose(r, EE).pos.clone()
    .add(new THREE.Vector3(0.30, 0.08, -0.12));
  const direct = posedRig();
  const goal = goalOf(direct);
  dragTo(direct, goal);

  const viaHigh = posedRig();
  dragTo(viaHigh, nodePose(viaHigh, EE).pos.clone().add(new THREE.Vector3(0, 0.30, 0)), 40);
  dragTo(viaHigh, goal);

  for (const n of CHAIN){
    const d = Math.abs(direct._q[n] - viaHigh._q[n]) * DEG;
    assert.ok(d < 60, `${n} differs by ${d.toFixed(0)}° between drag paths`);
  }
});

test('real arm: one tick can never teleport a joint', () => {
  const rig = posedRig();
  const start = nodePose(rig, EE);
  // a target grabbed far from the hand, in ONE tick
  const far = start.pos.clone().add(new THREE.Vector3(0.6, 0.3, 0));
  const before = CHAIN.map(n => rig._q[n]);
  solveIK(rig, CHAIN, EE, far, { rest: REST_POSE, targetQuat: start.quat.clone() });
  CHAIN.forEach((n, i) => {
    const step = Math.abs(rig._q[n] - before[i]);
    assert.ok(step <= 0.12 + 1e-9, `${n} moved ${(step * DEG).toFixed(1)}° in one tick`);
  });
});

test('real arm: holding an unreachable target does not wander', () => {
  const rig = posedRig();
  const start = nodePose(rig, EE);
  const goal = start.pos.clone().add(new THREE.Vector3(0.9, 0.2, 0));
  const opts = { rest: REST_POSE, targetQuat: start.quat.clone() };
  for (let k = 0; k < 60; k++) solveIK(rig, CHAIN, EE, goal, opts);
  const settled = CHAIN.map(n => rig._q[n]);
  for (let k = 0; k < 100; k++) solveIK(rig, CHAIN, EE, goal, opts);
  CHAIN.forEach((n, i) => {
    const drift = Math.abs(rig._q[n] - settled[i]) * DEG;
    assert.ok(drift < 2, `${n} drifted ${drift.toFixed(2)}° while merely holding`);
  });
});

test('a cold IK cache seeds from the live pose, never from zero', () => {
  const rig = posedRig();
  rig._q = {};                                   // cache cold (fresh page / reconnect)
  const here = nodePose(rig, EE).pos.clone();
  const sol = solveIK(rig, CHAIN, EE, here,
    { seed: REST_POSE, rest: REST_POSE, targetQuat: nodePose(rig, EE).quat.clone() });
  assert.ok(sol);
  // asked to stay put from the rest pose → must not swing toward an all-zero pose
  for (const n of CHAIN){
    const moved = Math.abs(sol[n] - REST_POSE[n]) * DEG;
    assert.ok(moved < 10, `${n} jumped ${moved.toFixed(1)}° on a cold-cache hold`);
  }
});
