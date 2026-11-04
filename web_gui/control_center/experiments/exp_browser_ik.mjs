// exp_browser_ik.mjs — does the Simulation Control Center's TASK-SPACE arm
// control produce sane arm/hand configurations?
//
// THE REPORT: "when I do task or joint space control the arm and hand run into
// optimization exploration and the hand twists into weird configurations."
//
// THE MECHANISM UNDER TEST: task space in this studio is solved IN THE BROWSER
// (web/js/kinematics.js `solveIK`, driven by viewport3d.js `_solveArm`) — the
// server never sees a Cartesian target, only the resulting `joint_command`. The
// solve is POSITION-ONLY over a 7-DOF arm, so four dimensions of the answer are
// unconstrained. This harness quantifies what fills that null space.
//
// It runs the REAL solver against the REAL robot geometry (the mabel_rig.glb
// node graph + mabel_joints.json manifest, rebuilt headlessly by
// tests/js/fixtures/rig.mjs), with the posture task OFF (the historical bare
// DLS) and ON (the shipped fix), and reports:
//
//   E1  DRAG SWEEP     — 8 directions × 3 distances from rest. Metrics: palm
//                        ORIENTATION drift during a pure TRANSLATION drag (the
//                        visible "hand twist"), the hand-mount twist joint
//                        (arm_7) and forearm roll (arm_5) excursions, worst
//                        per-tick joint jump, and joints parked ON a limit.
//   E2  PATH SPREAD    — one goal reached by four different drag paths. Metric:
//                        max−min per joint across the solutions. A solver that
//                        resolves redundancy answers the same way regardless of
//                        how the operator got there; spread IS the "exploration".
//   E3  HOLD STABILITY — an unreachable target held for 200 ticks. Metric: drift
//                        while merely holding (guards against a solver that
//                        wanders at a singularity).
//
// Run:  node experiments/exp_browser_ik.mjs        (from the studio root)
// Numbers are printed as a table and the process exits non-zero on a FAIL.
import * as THREE from 'three';
import { solveIK } from '../web/js/kinematics.js';
import { buildRig, restPose, ARM, nodePose } from '../tests/js/fixtures/rig.mjs';
import { REST_POSE } from '../web/js/homepose.js';

const D = 180 / Math.PI;
const EE = 'right_palm__anchor';
const names = ARM('right');
const FAILURES = [];
const check = (ok, label, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  — ' + detail : ''}`);
  if (!ok) FAILURES.push(label);
};

// A fresh rig at the pose the robot is ACTUALLY in when an operator starts
// dragging: the studio's Home/REST_POSE (arms hanging at the sides) laid over
// the manifest ref pose, which supplies the 14 ORCA joints REST_POSE does not
// name. The IK cache is seeded from it exactly as viewport3d.updateRobot() does
// from `robot_state`. Starting anywhere else makes the posture task look like a
// disturbance rather than a regulariser.
function fresh(){
  const rig = buildRig();
  const q0 = { ...restPose(), ...REST_POSE };
  rig.pose(q0);
  rig._q = {};
  for (const n in q0) rig._q[n] = q0[n];
  return { rig, q0 };
}

// `holdOrientation` reproduces what the studio does: latch the palm's current
// orientation when the grab starts and track it for the whole drag.
function drag(rig, from, to, opts, ticks = 60){
  const t = new THREE.Vector3();
  let prev = null, last = null, maxJump = 0, nulls = 0;
  const o = { ...opts };
  if (o.holdOrientation){
    delete o.holdOrientation;
    o.targetQuat = nodePose(rig, EE).quat.clone();
  }
  for (let k = 1; k <= ticks; k++){
    t.lerpVectors(from, to, k / ticks);
    const sol = solveIK(rig, names, EE, t, o);
    if (!sol){ nulls++; continue; }
    if (prev) maxJump = Math.max(maxJump, ...names.map(n => Math.abs(sol[n] - prev[n])));
    prev = { ...sol };
    last = sol;
  }
  return { last, maxJump, nulls };
}

// The BASELINE must be the historical solver, not "the new one minus posture" —
// so it explicitly disables orientation tracking, the step cap and the soft
// limits. `ON` is what viewport3d._solveArm now ships: hold the palm orientation
// the drag started from, plus the null-space posture task.
const OFF = { rest: null, maxStep: Infinity, limitMargin: 0, posDeadband: 0, rotDeadband: 0 };
const ON  = { rest: REST_POSE, holdOrientation: true };

const DIRS = {
  'forward +x': [1, 0, 0], 'back -x': [-1, 0, 0], 'out +z': [0, 0, 1],
  'across -z': [0, 0, -1], 'up +y': [0, 1, 0], 'down -y': [0, -1, 0],
  'fwd+up': [0.7, 0.7, 0], 'fwd+across': [0.7, 0, -0.7],
};
const DISTS = [0.25, 0.55, 0.9];

// ── E1 ──────────────────────────────────────────────────────────────────────
function e1(opts){
  const rows = [];
  for (const [label, v] of Object.entries(DIRS))
    for (const dist of DISTS){
      const { rig, q0 } = fresh();
      const s = nodePose(rig, EE);
      const goal = s.pos.clone()
        .add(new THREE.Vector3(...v).normalize().multiplyScalar(dist));
      const r = drag(rig, s.pos.clone(), goal, opts);
      if (!r.last) continue;
      const ee = nodePose(rig, EE);
      // joints resting ON a hard limit (within 0.5°)
      let pinned = 0;
      for (const n of names){
        const j = rig.joints[n];
        if (Math.min(Math.abs(r.last[n] - j.lower), Math.abs(r.last[n] - j.upper)) < 0.009) pinned++;
      }
      rows.push({
        label, dist,
        residual: ee.pos.distanceTo(goal),
        palmRot: 2 * Math.acos(Math.min(1, Math.abs(ee.quat.dot(s.quat)))) * D,
        arm5: Math.abs(r.last.right_arm_5 - (q0.right_arm_5 || 0)) * D,
        arm7: Math.abs(r.last.right_arm_7 - (q0.right_arm_7 || 0)) * D,
        jump: r.maxJump * D,
        pinned,
      });
    }
  rows.forEach(r => { r.key = `${r.label}@${r.dist}`; });
  return { rows, byKey: Object.fromEntries(rows.map(r => [r.key, r])) };
}

// Most of the 24 probe targets are deliberately OUTSIDE the arm's workspace —
// 0.9 m in any direction, or straight DOWN from an already-hanging arm. On those
// the arm is stretched flat against its limits by construction, so their posture
// measures the workspace boundary, not the solver. Scoring them would just be
// measuring the robot's arm length.
//
// REACHABLE is therefore defined by the OLD solver: a target it drove the palm
// to within 6 cm of is a target this arm can reach, so the new solver has no
// excuse. That keeps the comparison honest in the strict direction — the
// baseline picks the test set, and it picks the set it was good at.
// A baseline row only counts if the OLD solver reached it AND did so with a pose
// an operator would have accepted. Rows it "reached" by tumbling the palm through
// >45° are the contortions this change exists to stop; scoring the new solver for
// declining to reproduce them would be scoring it for working.
const REACH_TOL = 0.06;
const SANE_ROT = 45;
function score(base, cand){
  const keys = Object.keys(base.byKey).filter(k =>
    base.byKey[k].residual < REACH_TOL && base.byKey[k].palmRot < SANE_ROT);
  const pick = (src, f) => Math.max(...keys.map(k => f(src.byKey[k])));
  return {
    keys,
    palmRot: pick(cand, r => r.palmRot), arm7: pick(cand, r => r.arm7),
    arm5: pick(cand, r => r.arm5), jump: pick(cand, r => r.jump),
    pinned: keys.reduce((a, k) => a + cand.byKey[k].pinned, 0),
    worstResidual: pick(cand, r => r.residual),
    worstRegression: Math.max(...keys.map(k =>
      cand.byKey[k].residual - base.byKey[k].residual)),
  };
}

// ── E2 ──────────────────────────────────────────────────────────────────────
function e2(opts){
  const s0 = (() => { const { rig } = fresh(); return nodePose(rig, EE).pos.clone(); })();
  const goal = s0.clone().add(new THREE.Vector3(0.35, 0.10, -0.15));
  const vias = [null,
    s0.clone().add(new THREE.Vector3(0, 0.35, 0)),
    s0.clone().add(new THREE.Vector3(0, -0.30, 0)),
    s0.clone().add(new THREE.Vector3(0.35, 0, 0.30))];
  const sols = [];
  for (const via of vias){
    const { rig } = fresh();
    if (via) drag(rig, nodePose(rig, EE).pos.clone(), via, opts, 40);
    const r = drag(rig, nodePose(rig, EE).pos.clone(), goal, opts, 60);
    if (r.last) sols.push(names.map(n => r.last[n]));
  }
  const spread = names.map((_, j) =>
    (Math.max(...sols.map(q => q[j])) - Math.min(...sols.map(q => q[j]))) * D);
  return { sols, spread, worst: Math.max(...spread) };
}

// ── E3 ──────────────────────────────────────────────────────────────────────
function e3(opts){
  const { rig } = fresh();
  const s = nodePose(rig, EE);
  const goal = s.pos.clone().add(new THREE.Vector3(1.0, 0.2, 0));
  const e3opts = { ...opts };
  if (e3opts.holdOrientation){
    delete e3opts.holdOrientation;
    e3opts.targetQuat = nodePose(rig, EE).quat.clone();
  }
  drag(rig, s.pos.clone(), goal, opts, 60);
  const before = names.map(n => rig._q[n]);
  let prev = before.slice(), wander = 0;
  for (let k = 0; k < 200; k++){
    const sol = solveIK(rig, names, EE, goal, e3opts);
    if (!sol) continue;
    const q = names.map(n => sol[n]);
    wander = Math.max(wander, ...q.map((v, i) => Math.abs(v - prev[i])));
    prev = q;
  }
  return { drift: Math.max(...names.map((n, i) => Math.abs(rig._q[n] - before[i]))) * D,
           wander: wander * D };
}

// ── run ─────────────────────────────────────────────────────────────────────
const a1 = e1(OFF), b1 = e1(ON);
const a2 = e2(OFF), b2 = e2(ON);
const a3 = e3(OFF), b3 = e3(ON);

const f = (v, u = '') => (v.toFixed(u === 'm' ? 4 : 1) + u).padStart(10);
const A = score(a1, a1), B = score(a1, b1);   // both scored on the OLD solver's reachable set

console.log('\n=== E1  drag sweep: 8 directions x 3 distances (right arm, from rest) ===');
console.log(`scored on the ${A.keys.length} of ${a1.rows.length} probe targets the OLD solver reached (<${REACH_TOL} m)`);
console.log('metric                                    OLD          NEW');
console.log(`  palm rotation on a TRANSLATION drag ${f(A.palmRot)}deg${f(B.palmRot)}deg`);
console.log(`  hand-mount twist  arm_7 excursion   ${f(A.arm7)}deg${f(B.arm7)}deg`);
console.log(`  forearm roll      arm_5 excursion   ${f(A.arm5)}deg${f(B.arm5)}deg`);
console.log(`  worst per-tick joint jump (@30 Hz)  ${f(A.jump)}deg${f(B.jump)}deg`);
console.log(`  joints parked ON a hard limit       ${f(A.pinned)}   ${f(B.pinned)}`);
console.log(`  worst residual                      ${f(A.worstResidual, 'm')}${f(B.worstResidual, 'm')}`);
console.log(`  worst reach regression vs OLD                  ${f(B.worstRegression, 'm')}`);

console.log('\n=== E2  one goal, four drag paths: null-space spread (deg) ===');
console.log('  joint          q1    q2    q3    q4    q5    q6    q7');
console.log('  OLD   ', a2.spread.map(v => v.toFixed(0).padStart(5)).join(' '));
console.log('  NEW   ', b2.spread.map(v => v.toFixed(0).padStart(5)).join(' '));

console.log('\n=== E3  holding an UNREACHABLE target, 200 ticks ===');
console.log(`  drift while holding                 ${f(a3.drift)}deg${f(b3.drift)}deg`);
console.log(`  worst per-tick wander               ${f(a3.wander)}deg${f(b3.wander)}deg`);

// per-row dump (diagnostic)
if (process.env.DUMP){
  console.log('\ndir           dist  residual  palmRot  arm5  arm7  jump  pinned  reachable');
  for (const r of b1.rows)
    console.log(`${r.label.padEnd(12)} ${r.dist.toFixed(2)} ${r.residual.toFixed(3)}m ` +
      `${r.palmRot.toFixed(0).padStart(7)} ${r.arm5.toFixed(0).padStart(5)} ` +
      `${r.arm7.toFixed(0).padStart(5)} ${r.jump.toFixed(1).padStart(5)} ${String(r.pinned).padStart(5)}` +
      `   ${A.keys.includes(r.key) ? 'yes' : '-'}`);
}

console.log('\n=== VERDICT ===');
check(B.palmRot < 15, 'palm holds its orientation through a translation drag',
  `${A.palmRot.toFixed(0)} -> ${B.palmRot.toFixed(0)} deg`);
check(B.arm7 < 35, 'hand-mount twist (arm_7) stays modest',
  `${A.arm7.toFixed(0)} -> ${B.arm7.toFixed(0)} deg`);
check(B.jump <= 0.12 * D + 0.6, 'per-tick joint motion is bounded by the step cap',
  `${A.jump.toFixed(0)} -> ${B.jump.toFixed(0)} deg/tick`);
check(B.pinned <= A.pinned, 'no more joints park on a hard limit than before',
  `${A.pinned} -> ${B.pinned}`);
check(b2.worst < a2.worst * 0.5, 'path dependence (null-space spread) is cut',
  `${a2.worst.toFixed(0)} -> ${b2.worst.toFixed(0)} deg`);
check(b3.wander < 1.0, 'no wander while holding a singular target',
  `${b3.wander.toFixed(2)} deg/tick`);
check(B.worstRegression < 0.05, 'reach does not regress on targets the old solver reached',
  `worst ${B.worstRegression >= 0 ? '+' : ''}${B.worstRegression.toFixed(4)} m`);

console.log(FAILURES.length ? `\n${FAILURES.length} FAILURE(S)` : '\nALL CHECKS PASS');
process.exit(FAILURES.length ? 1 : 0);
