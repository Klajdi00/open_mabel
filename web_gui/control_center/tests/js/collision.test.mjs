// The self-collision CBF twin (collision.js): segment distance, the velocity
// projection (keep tangential, remove the normal closing excess), and the
// collision-avoided path planner — on a synthetic 2-DOF cartesian rig with one
// hand sphere vs a static torso capsule, where every quantity is analytic.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { makeGuard, segseg, SC_MARGIN, SC_INFLUENCE } from '../../web/js/collision.js';

test('segseg: closest points of skew segments', () => {
  const a = segseg(new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 0, 0),
                   new THREE.Vector3(0, 1, -1), new THREE.Vector3(0, 1, 1),
                   new THREE.Vector3(), new THREE.Vector3());
  assert.ok(Math.abs(a - 1) < 1e-12);
});

// 2-DOF cartesian rig: jx/jy translate a carriage carrying the "hand" node.
// A static vertical "torso" capsule (body_mount → head_mount) stands at x=0.
// makeGuard sees exactly one pair: l_hand (sphere r=0.055) vs torso (r=0.09).
function makeRig(){
  const root = new THREE.Group(); root.name = 'root';
  const world = new THREE.Group(); world.add(root);
  const mk = (name, x, y, z, parent) => {
    const o = new THREE.Object3D(); o.name = name; o.position.set(x, y, z);
    parent.add(o); return o;
  };
  mk('body_mount', 0, 0, 0, root);
  mk('head_mount', 0, 1, 0, root);
  const carriage = mk('carriage', 0.6, 0.5, 0, root);
  mk('l_wrist', 0, 0, 0, carriage);
  mk('left_palm', 0, 0, 0, carriage);
  const joints = { jx: { node: carriage, axis: 'x', base: 0.6 },
                   jy: { node: carriage, axis: 'y', base: 0.5 } };
  const rig = {
    root, world, joints,
    setJoint(name, v){
      const j = this.joints[name];
      j.node.position[j.axis] = j.base + v;
    },
  };
  world.updateMatrixWorld(true);
  return rig;
}
const RSUM = 0.055 + 0.09;          // hand + torso radii

test('free motion far from the torso is untouched', () => {
  const rig = makeRig();
  const g = makeGuard(rig);
  assert.equal(g.pairs.length, 1);
  const step = [-0.01, 0.02];       // toward the torso but far outside influence
  const out = g.projectStep(['jx', 'jy'], [0, 0], step.slice(), 0.05);
  assert.deepEqual(out, step);
});

test('head-on approach is clipped, tangential slide is preserved', () => {
  const rig = makeRig();
  const g = makeGuard(rig);
  // park the hand close to the capsule surface: gap = 0.6 − rsum + jx
  const q = [-(0.6 - RSUM - 0.04), 0];       // gap = 4 cm < influence 6 cm
  const dt = 0.05;
  // pure tangential motion (along the capsule axis, +y): must pass unchanged
  let out = g.projectStep(['jx', 'jy'], q.slice(), [0, 0.03], dt);
  assert.ok(Math.abs(out[0] - 0) < 1e-9 && Math.abs(out[1] - 0.03) < 1e-9,
    `tangential altered: ${out}`);
  // head-on closing at 0.6 m/s (0.03/0.05): allowed is min(α·slack, √(2a·slack))
  out = g.projectStep(['jx', 'jy'], q.slice(), [-0.03, 0.01], dt);
  const slack = 0.04 - SC_MARGIN;
  const allow = Math.min(6.0 * slack, Math.sqrt(2 * 2.0 * slack));
  assert.ok(-out[0] / dt <= allow + 1e-6,
    `closing ${-out[0] / dt} m/s exceeds the CBF allowance ${allow}`);
  assert.ok(Math.abs(out[1] - 0.01) < 1e-9, 'tangential component was altered');
});

test('planPath slides AROUND a finite capsule without ever entering the margin', () => {
  const rig = makeRig();
  const g = makeGuard(rig);
  // command the hand THROUGH the capsule to the far side, offset in y: the
  // legal route is around the capsule's top, and the flow must find it while
  // keeping every waypoint outside the margin
  const res = g.planPath(['jx', 'jy'], { jx: 0, jy: 0 }, { jx: -1.2, jy: 0.9 },
                         { vmax: 1.0, dt: 0.05, maxSteps: 200 });
  assert.ok(res.path.length > 2, 'no path produced');
  assert.ok(res.minSlack >= -1e-3,
    `path entered the margin by ${(-res.minSlack * 1000).toFixed(2)} mm`);
  const end = res.path[res.path.length - 1];
  assert.ok(end.jy > 0.5, `no tangential slide: jy=${end.jy}`);
});

test('planPath into an effectively infinite wall PINS on the boundary', () => {
  const rig = makeRig();
  // stretch the torso capsule into a wall no 2-DOF slide can circumvent
  rig.root.getObjectByName('body_mount').position.y = -100;
  rig.root.getObjectByName('head_mount').position.y = 100;
  rig.world.updateMatrixWorld(true);
  const g = makeGuard(rig);
  const res = g.planPath(['jx', 'jy'], { jx: 0, jy: 0 }, { jx: -1.2, jy: 0.4 },
                         { vmax: 1.0, dt: 0.05, maxSteps: 200 });
  assert.ok(res.minSlack >= -1e-3,
    `path entered the margin by ${(-res.minSlack * 1000).toFixed(2)} mm`);
  const end = res.path[res.path.length - 1];
  // pinned outside the wall: never through, boundary-riding in x…
  assert.ok(0.6 + end.jx > 0.09 + 0.055, `passed through the wall: jx=${end.jx}`);
  // …while the tangential target component was still honoured
  assert.ok(Math.abs(end.jy - 0.4) < 1e-3, `tangential goal missed: jy=${end.jy}`);
});

test('planPath to a reachable target arrives exactly', () => {
  const rig = makeRig();
  const g = makeGuard(rig);
  const res = g.planPath(['jx', 'jy'], { jx: 0, jy: 0 }, { jx: 0.2, jy: -0.3 },
                         { vmax: 1.0, dt: 0.05, maxSteps: 100 });
  const end = res.path[res.path.length - 1];
  assert.ok(Math.abs(end.jx - 0.2) < 1e-3 && Math.abs(end.jy + 0.3) < 1e-3,
    `free target missed: ${JSON.stringify(end)}`);
  assert.equal(res.blocked, false);
});

test('guard degrades to nothing on a rig without the capsule nodes', () => {
  const root = new THREE.Group();
  const world = new THREE.Group(); world.add(root);
  const rig = { root, world, joints: {}, setJoint(){} };
  const g = makeGuard(rig);
  assert.equal(g.pairs.length, 0);
  const out = g.projectStep(['j'], [0], [0.5], 0.05);
  assert.deepEqual(out, [0.5]);
});
