// Damped-least-squares IK over a rig joint chain (numeric Jacobian): drag the
// wrist ball → arm joint targets. Pure three.js math on the live Rig object —
// node-testable (three.module.js is plain ESM).
//
// WHY THIS IS A 6-DOF SOLVE.
// It used to track POSITION ONLY. On a 7-DOF arm that leaves a 4-dimensional
// null space with nothing to fill it, so the realised posture became a path
// integral of however the operator wiggled the mouse. Measured on the real rig
// (experiments/exp_browser_ik.mjs): the SAME target reached by four different
// drag paths gave shoulder solutions 182° apart, and the palm rotated through up
// to 176° during a drag that was pure TRANSLATION. Nothing was wrong with the
// linear algebra — the solver simply had no opinion about which of the infinitely
// many arm configurations reaching a point is the one a human would pick, and no
// opinion whatsoever about which way the hand should face. That is the operator's
// "optimization exploration / the hand twists into weird configurations".
//
// Tracking ORIENTATION as well as position makes the task 6-DOF, leaving a 1-DOF
// null space (the elbow swivel) — which is the redundancy a human arm actually
// has. The default orientation target is "whatever the palm was facing when the
// drag started", so dragging the ball TRANSLATES the hand instead of tumbling it.
// This is the browser twin of the server's 18-DOF solver
// (controller/mabel/whole_body_controller/ik.py), which has always tracked full
// SE(3) palm targets with `_so3_error` and a null-space posture task.
//
// The remaining pieces mirror that solver too:
//   · a NULL-SPACE POSTURE pull toward a rest configuration, weighted per joint
//     (POSTURE_W) the way config.ARM_REG_W is — roll resolved at the forearm
//     rather than by spinning the shoulder or the hand mount;
//   · SOFT joint limits — a velocity damper that decelerates into a stop instead
//     of the hard clamp that used to pin a joint AT its limit for a whole drag;
//   · a PER-CALL step cap, so one 33 ms tick cannot teleport a joint (the old
//     solver produced 69°/tick ≈ 36 rad/s swings on far targets).
import * as THREE from 'three';

const _p = new THREE.Vector3();

// ── constants mirrored from controller/mabel/config.py ───────────────────────
// The server's 18-DOF solver is the one that drives the REAL robot, so this
// browser solver must agree with it or the studio's preview lies about what the
// robot will do. Keep these in sync with config.py if they are retuned there.
export const IK_KP_POS = 6.0;          // config.IK_KP_POS — EE position gain
export const IK_KP_ROT = 2.0;          // config.IK_KP_ROT — EE orientation gain
export const IK_DAMPING = 0.05;        // config.IK_DAMPING — DLS lambda
export const IK_POSTURE_GAIN = 0.5;    // config.IK_POSTURE_GAIN — null-space pull
export const IK_POS_DEADBAND = 0.002;  // config.IK_POS_DEADBAND — m
export const IK_ROT_DEADBAND = 0.0087; // config.IK_ROT_DEADBAND — rad (~0.5 deg)

// Shrink a vector's magnitude by `db`, clamped at zero (soft, continuous) —
// the server's `_deadband` in whole_body_controller/ik.py. Sub-threshold error
// maps to no command, so the arm holds still instead of creeping.
function deadband(v, db){
  const nrm = Math.hypot(...v);
  if (!(db > 0) || nrm <= 1e-12) return v;
  const s = Math.max(0, nrm - db) / nrm;
  return v.map(x => x * s);
}

// Per-joint null-space posture weights. Mirrors config.ARM_REG_W in
// controller/mabel/config.py: firm return on shoulder-roll (arm_3) and hand-mount
// yaw (arm_7), light on the forearm (arm_5), so orientation is resolved the way a
// human's is. Torso/lift are pulled lightly so a widened control region does not
// drift the whole body to serve a hand target.
export const POSTURE_W = Object.freeze({
  arm_1: 0.08, arm_2: 0.08, arm_3: 0.40, arm_4: 0.08,
  arm_5: 0.02, arm_6: 0.08, arm_7: 0.60,
  torso: 0.25, lift_lower: 0.25, lift_upper: 0.25,
});

// weight for a rig joint name ("right_arm_7" → arm_7 → 0.60)
export function postureWeight(name){
  if (name in POSTURE_W) return POSTURE_W[name];
  const m = /(arm_[1-7])$/.exec(name);
  return m ? POSTURE_W[m[1]] : 0.1;
}

function eePos(rig, eeName, out){
  rig.root.getObjectByName(eeName).getWorldPosition(out);
  return out;
}

const _m3 = new THREE.Matrix4();
function eeBasis(rig, eeName, out){
  const o = rig.root.getObjectByName(eeName);
  o.updateWorldMatrix(true, false);
  out.setFromMatrix4(_m3.extractRotation(o.matrixWorld));
  return out;
}

// Orientation error as a body-frame rotation vector — the sum of column crosses.
// Verbatim the server's `_so3_error` (whole_body_controller/ik.py), so the two
// solvers agree on what "the hand is turned wrong" means.
export function so3Error(Rcur, Rtrg){
  const c = Rcur.elements, t = Rtrg.elements;   // three.js Matrix3 is column-major
  const col = (e, i) => [e[i * 3], e[i * 3 + 1], e[i * 3 + 2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                           a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0]];
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++){
    const x = cross(col(c, i), col(t, i));
    out[0] += x[0]; out[1] += x[1]; out[2] += x[2];
  }
  return out.map(v => 0.5 * v);
}

// Dense linear solve (Gaussian elimination with partial pivoting). Returns null
// if singular — the damping term normally keeps A well conditioned.
export function solveN(A, b){
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++){
    let piv = c;
    for (let r = c + 1; r < n; r++)
      if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < n; r++){
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}

// One damped-least-squares step over a 6-DOF task:
//   dq = Jᵀ(J Jᵀ + λ²I)⁻¹ e  +  (I − J⁺J)·dq_posture
//
// opts.targetQuat  THREE.Quaternion — palm orientation to hold/track. Omit to
//                  fall back to POSITION-ONLY (the historical behaviour).
// opts.seed        {name: rad} — COLD-START pose, used only where `rig._q` has no
//                  entry. `rig._q` still wins: it carries the accumulated solution
//                  across the ticks of one drag, and re-seeding from the (lagging)
//                  measured pose every tick would make the arm crawl behind the
//                  ball. The point of `seed` is to replace the old `?? 0`
//                  fallback, which started a cold solve at all-zeros — a pose the
//                  robot is never in, so the first drag after a reconnect snapped.
// opts.rest        {name: rad} — posture the null space returns to (null = off).
export function solveIK(rig, jointNames, eeName, targetWorld,
                        { iters = 6, lambda = IK_DAMPING, step = 1e-3, maxErr = 1.5,
                          targetQuat = null, wRot = IK_KP_ROT / IK_KP_POS,
                          seed = null, rest = null, postureGain = IK_POSTURE_GAIN,
                          posDeadband = IK_POS_DEADBAND, rotDeadband = IK_ROT_DEADBAND,
                          limitMargin = 0.15, maxStep = 0.12 } = {}){
  const joints = jointNames.map(n => rig.joints[n]).filter(Boolean);
  if (joints.length !== jointNames.length) return null;
  const ee = rig.root.getObjectByName(eeName);
  if (!ee) return null;

  rig._q = rig._q || {};
  const q = jointNames.map(n => rig._q[n] ?? seed?.[n] ?? 0);
  const q0 = q.slice();
  const n = q.length;
  const rows = targetQuat ? 6 : 3;

  const setAll = (vals) => {
    for (let i = 0; i < n; i++) rig.setJoint(jointNames[i], vals[i]);
    rig.world.updateMatrixWorld(true);
  };

  const wPost = jointNames.map(nm => postureWeight(nm));
  const qRest = rest ? jointNames.map((nm, i) => rest[nm] ?? q0[i]) : null;
  const Rtrg = targetQuat
    ? new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(targetQuat))
    : null;

  const err = new THREE.Vector3();
  const Rcur = new THREE.Matrix3(), Rp = new THREE.Matrix3(), Rm = new THREE.Matrix3();

  for (let it = 0; it < iters; it++){
    setAll(q);
    eePos(rig, eeName, _p);
    err.copy(targetWorld).sub(_p);
    if (err.length() > maxErr) return null;          // target out of reach: bail

    const e = deadband([err.x, err.y, err.z], posDeadband);
    if (Rtrg){
      eeBasis(rig, eeName, Rcur);
      const er = deadband(so3Error(Rcur, Rtrg), rotDeadband);
      e.push(wRot * er[0], wRot * er[1], wRot * er[2]);
    }
    const eNorm = Math.hypot(...e);
    if (eNorm < 1e-4 && !qRest) break;

    // Numeric Jacobian by central differences. J[i] is the COLUMN for joint i:
    // linear velocity rows, then (if tracking orientation) angular rows expressed
    // the same way the error is — as a rotation vector.
    const J = [];
    for (let i = 0; i < n; i++){
      const qs = q.slice(); qs[i] += step;
      setAll(qs);
      const p1 = eePos(rig, eeName, new THREE.Vector3());
      if (Rtrg) eeBasis(rig, eeName, Rp);
      qs[i] -= 2 * step;
      setAll(qs);
      const p0 = eePos(rig, eeName, new THREE.Vector3());
      if (Rtrg) eeBasis(rig, eeName, Rm);
      const col = [(p1.x - p0.x) / (2 * step), (p1.y - p0.y) / (2 * step),
                   (p1.z - p0.z) / (2 * step)];
      if (Rtrg){
        // d(rotvec)/dq: so3Error(R−, R+) measures the rotation taking R− to R+.
        const dr = so3Error(Rm, Rp);
        col.push(wRot * dr[0] / (2 * step), wRot * dr[1] / (2 * step),
                 wRot * dr[2] / (2 * step));
      }
      J.push(col);
    }

    // A = J Jᵀ + λ²I  (rows×rows), solve A y = e, then dq_task = Jᵀ y
    const A = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: rows }, (_, c) => (r === c ? lambda * lambda : 0)));
    for (let r = 0; r < rows; r++) for (let c = 0; c < rows; c++)
      for (let i = 0; i < n; i++) A[r][c] += J[i][r] * J[i][c];
    const y = solveN(A, e);
    if (!y) break;

    const dq = new Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let r = 0; r < rows; r++) dq[i] += J[i][r] * y[r];

    // NULL-SPACE POSTURE: dq += (I − J⁺J)·dq_post, computed without ever forming
    // the n×n projector — subtract dq_post's task-space image:
    //   dq_ns = dq_post − Jᵀ(J Jᵀ + λ²I)⁻¹ J dq_post
    if (qRest){
      const dqp = new Array(n);
      for (let i = 0; i < n; i++) dqp[i] = postureGain * wPost[i] * (qRest[i] - q[i]);
      const Jv = new Array(rows).fill(0);
      for (let r = 0; r < rows; r++) for (let i = 0; i < n; i++) Jv[r] += J[i][r] * dqp[i];
      const z = solveN(A, Jv);
      if (z) for (let i = 0; i < n; i++){
        let img = 0;
        for (let r = 0; r < rows; r++) img += J[i][r] * z[r];
        dq[i] += dqp[i] - img;
      }
    }

    for (let i = 0; i < n; i++){
      let v = Math.max(-0.2, Math.min(0.2, dq[i]));   // per-iter step clamp
      const j = rig.joints[jointNames[i]];
      // SOFT LIMITS: fade the step to zero across `limitMargin` of the stop being
      // approached. A hard clamp (the old behaviour) let the solver drive a joint
      // flat onto its limit and park there — measured as arm_5 pinned at exactly
      // 90° for the rest of a drag, which reads as a locked, contorted forearm.
      if (j && j.lower != null && j.upper != null){
        const room = v > 0 ? j.upper - q[i] : q[i] - j.lower;
        if (limitMargin > 0 && room < limitMargin) v *= Math.max(0, room / limitMargin);
        q[i] = Math.max(j.lower, Math.min(j.upper, q[i] + v));
      } else {
        q[i] += v;
      }
    }
  }

  // PER-CALL STEP CAP: bound how far any joint may move in one solve (one 30 Hz
  // drag tick), so a far or newly-grabbed target ramps instead of snapping.
  if (Number.isFinite(maxStep))
    for (let i = 0; i < n; i++)
      q[i] = Math.max(q0[i] - maxStep, Math.min(q0[i] + maxStep, q[i]));

  setAll(q);
  const out = {};
  for (let i = 0; i < n; i++){ out[jointNames[i]] = q[i]; rig._q[jointNames[i]] = q[i]; }
  return out;
}

// 3×3 linear solve (Cramer). Returns null if singular. Kept as the documented
// small-system helper (and covered by its own test).
export function solve3(A, b){
  const det = (m) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(A);
  if (Math.abs(D) < 1e-12) return null;
  const col = (m, i, v) => m.map((row, r) => row.map((x, c) => (c === i ? v[r] : x)));
  return [det(col(A, 0, b)) / D, det(col(A, 1, b)) / D, det(col(A, 2, b)) / D];
}
