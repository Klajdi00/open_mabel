// Self-collision guard — the browser twin of the controller's capsule CBF
// (controller/mabel/whole_body_controller/collision_avoidance/). The server is
// the AUTHORITY: every manipulate joint target is re-filtered facade-side by
// the same law (JointTargetProjector), so this twin exists so the studio's
// task-space PREVIEW shows the collision-avoided path the robot will actually
// take — a wrist ball dragged into the torso previews a hand that SLIDES along
// the body and settles on the boundary, not one that passes through it.
//
// THE LAW (control barrier function, velocity-projection form). Each protected
// link is a capsule (segment + radius); for a pair the surface gap is
// d = ‖c_a − c_b‖ − r_a − r_b. Per plan step the joint velocity v = Δq/dt is
// projected onto the half-spaces
//
//     ḋ = (∂d/∂q)·v ≥ −α (d − d_s)         for every pair with d < d_influence
//
// — the minimal correction removes ONLY each barrier's normal (gap-closing)
// excess and keeps the TANGENTIAL component, so motion along the body is never
// impeded and free motion is untouched. ∂d/∂q is numeric (forward differences
// on the rig), matching kinematics.js's numeric-Jacobian style.
//
// Constants and the capsule/pair tables are MIRRORED from
// controller/mabel/config.py (SC_*) and capsule_guard._build_capsules/_pairs —
// keep them in sync if retuned there (drift-guarded by
// controller/tests/test_collision_avoidance.py::test_web_twin_constants_match).
import * as THREE from 'three';

// ── constants mirrored from controller/mabel/config.py ───────────────────────
export const SC_MARGIN = 0.010;      // m  safety distance d_s (thin layer, 2026-08-23)
export const SC_INFLUENCE = 0.06;    // m  influence distance d_i
export const SC_ALPHA = 6.0;         // 1/s barrier gain
export const SC_STOP_DECEL = 2.0;    // m/s² braking-distance cap
export const SC_UPPER_MARGIN = 0.003; // m  per-pair margin for the elbow spheres
export const SC_RADII = Object.freeze({
  forearm: 0.035, hand: 0.055, torso: 0.09, head: 0.12, upper: 0.035,
});

// Capsules as PAIRS OF RIG NODES (a === b ⇒ sphere) + radius — the same spans
// capsule_guard builds from the MJCF body origins; the GLB rig carries the same
// link frames (URDF-derived), so node world positions reproduce the segments.
export const CAPSULES = Object.freeze({
  l_fore:  ['l_elbow', 'l_wrist', SC_RADII.forearm],
  r_fore:  ['r_elbow', 'r_wrist', SC_RADII.forearm],
  l_hand:  ['l_wrist', 'left_palm', SC_RADII.hand],
  r_hand:  ['r_wrist', 'right_palm', SC_RADII.hand],
  torso:   ['body_mount', 'head_mount', SC_RADII.torso],
  head:    ['head', 'head', SC_RADII.head],
  l_upper: ['l_elbow', 'l_elbow', SC_RADII.upper],
  r_upper: ['r_elbow', 'r_elbow', SC_RADII.upper],
  // the ORCA wrist assembly bends off the straight wrist→palm axis — this
  // covers the J7→wrist-tower segment (2026-07-30)
  l_wristmt: ['l_wrist', 'left_tower', SC_RADII.hand],
  r_wristmt: ['r_wrist', 'right_tower', SC_RADII.hand],
});

// Monitored pairs (the third slot is a per-pair margin override; null = the
// global SC_MARGIN). Per-pair margins are no longer used: the elbow~torso pairs
// once ran at 3 mm because the settled elbow rested at +6 mm of the combined
// radii, and that was an artefact of a mis-placed torso capsule rather than of
// the robot — see capsule_guard._fit_to_geometry.
export const PAIRS = Object.freeze([
  ['l_hand', 'torso', null], ['l_hand', 'head', null],
  ['l_hand', 'r_hand', null], ['l_hand', 'r_fore', null],
  ['r_hand', 'torso', null], ['r_hand', 'head', null], ['r_hand', 'l_fore', null],
  ['l_fore', 'r_fore', null],
  ['l_wristmt', 'torso', null], ['l_wristmt', 'head', null],
  ['r_wristmt', 'torso', null], ['r_wristmt', 'head', null],
  // ARM ~ BODY, restored 2026-08-23: the forearm and upper arm were unguarded
  // against the torso entirely, and the right forearm was measured inside it on
  // 17 % of commanded postures.
  ['l_fore', 'torso', null], ['r_fore', 'torso', null],
  ['l_upper', 'torso', null], ['r_upper', 'torso', null],
]);

// ── closest points between two segments (Ericson, RTCD §5.1.9) ───────────────
const _d1 = new THREE.Vector3(), _d2 = new THREE.Vector3(), _r = new THREE.Vector3();
export function segseg(p1, q1, p2, q2, outA, outB){
  _d1.subVectors(q1, p1); _d2.subVectors(q2, p2); _r.subVectors(p1, p2);
  const a = _d1.dot(_d1), e = _d2.dot(_d2), f = _d2.dot(_r);
  const eps = 1e-12;
  let s = 0, t = 0;
  if (a <= eps && e <= eps){ s = 0; t = 0; }
  else if (a <= eps){ s = 0; t = Math.min(1, Math.max(0, f / e)); }
  else {
    const c = _d1.dot(_r);
    if (e <= eps){ t = 0; s = Math.min(1, Math.max(0, -c / a)); }
    else {
      const b = _d1.dot(_d2), denom = a * e - b * b;
      s = denom > eps ? Math.min(1, Math.max(0, (b * f - c * e) / denom)) : 0;
      t = (b * s + f) / e;
      if (t < 0){ t = 0; s = Math.min(1, Math.max(0, -c / a)); }
      else if (t > 1){ t = 1; s = Math.min(1, Math.max(0, (b - c) / a)); }
    }
  }
  outA.copy(_d1).multiplyScalar(s).add(p1);
  outB.copy(_d2).multiplyScalar(t).add(p2);
  return outA.distanceTo(outB);
}

// ── the guard over a live rig ─────────────────────────────────────────────────
const _pa = new THREE.Vector3(), _pb = new THREE.Vector3();
const _qa = new THREE.Vector3(), _qb = new THREE.Vector3();
const _wa = new THREE.Vector3(), _wb = new THREE.Vector3();

export function makeGuard(rig, { capsules = CAPSULES, pairs = PAIRS } = {}){
  // resolve nodes once; pairs whose nodes the rig lacks are simply not guarded
  const caps = {};
  for (const [name, [na, nb, r]] of Object.entries(capsules)){
    const a = rig.root.getObjectByName(na);
    const b = na === nb ? a : rig.root.getObjectByName(nb);
    if (a && b) caps[name] = { a, b, r };
  }
  const live = [];
  for (const [an, bn, m] of pairs){
    if (caps[an] && caps[bn])
      live.push({ a: caps[an], b: caps[bn], margin: m ?? SC_MARGIN });
  }

  const gap = (p) => {
    p.a.a.getWorldPosition(_pa); p.a.b.getWorldPosition(_qa);
    p.b.a.getWorldPosition(_pb); p.b.b.getWorldPosition(_qb);
    return segseg(_pa, _qa, _pb, _qb, _wa, _wb) - p.a.r - p.b.r;
  };

  const setAll = (names, q) => {
    for (let i = 0; i < names.length; i++) rig.setJoint(names[i], q[i]);
    rig.world.updateMatrixWorld(true);
  };

  return {
    pairs: live,
    // {pair gaps} at the rig's CURRENT pose — telemetry/tests
    gaps(){
      return live.map((p) => ({ margin: p.margin, gap: gap(p) }));
    },
    minSlack(){
      let m = Infinity;
      for (const p of live) m = Math.min(m, gap(p) - p.margin);
      return m;
    },
    // Project the joint STEP (rad, over dt seconds) taken from pose q so no
    // active pair closes faster than the barrier allows. Leaves the rig posed
    // at q. Returns the projected step (same length as names).
    projectStep(names, q, stepArr, dt = 0.1){
      setAll(names, q);
      const rows = [];
      for (const p of live){
        const d = gap(p);
        if (d >= SC_INFLUENCE) continue;
        const slack = d - p.margin;
        let allow = SC_ALPHA * slack;                       // ḋ ≥ −allow
        if (slack > 0) allow = Math.min(allow, Math.sqrt(2 * SC_STOP_DECEL * slack));
        // numeric ∂d/∂q by forward differences on the rig. J1/J2 are
        // correction-EXEMPT (mirrors config.SC_EXEMPT_JOINTS): their entries
        // are zeroed so corrections redistribute to the distal joints.
        const eps = 1e-3, grad = new Array(names.length);
        for (let i = 0; i < names.length; i++){
          if (/_arm_[12]$/.test(names[i])){ grad[i] = 0; continue; }
          q[i] += eps; setAll(names, q);
          grad[i] = (gap(p) - d) / eps;
          q[i] -= eps;
        }
        setAll(names, q);
        rows.push({ grad, allow });
      }
      const v = stepArr.map((s) => s / dt);
      // cyclic projection onto the half-spaces grad·v ≥ −allow (few rows; the
      // single-row case — by far the common one — is exact in one pass)
      for (let pass = 0; pass < 8; pass++){
        let worst = 0;
        for (const { grad, allow } of rows){
          let gv = 0, gg = 0;
          for (let i = 0; i < grad.length; i++){ gv += grad[i] * v[i]; gg += grad[i] * grad[i]; }
          const viol = -allow - gv;                          // >0 ⇒ closing too fast
          if (viol > 1e-12 && gg > 1e-12){
            for (let i = 0; i < grad.length; i++) v[i] += grad[i] * (viol / gg);
            worst = Math.max(worst, viol);
          }
        }
        if (worst <= 1e-9) break;
      }
      return v.map((x) => x * dt);
    },
    // The collision-avoided PATH from qFrom toward qTo: glide at vmax, project
    // every step. Returns { path: [{name: rad}, ...], blocked, minSlack }.
    // blocked = the flow settled (slid to the boundary) short of the raw target.
    // The rig is left posed at the LAST waypoint — callers restore (viewport
    // re-poses to live after every solve).
    planPath(names, qFrom, qTo, { vmax = 1.0, dt = 0.1, maxSteps = 120 } = {}){
      const q = names.map((n) => qFrom[n] ?? 0);
      const tgt = names.map((n) => qTo[n] ?? 0);
      const snap = (arr) => Object.fromEntries(names.map((n, i) => [n, arr[i]]));
      const path = [snap(q)];
      let minSlack = Infinity, blocked = false;
      for (let k = 0; k < maxSteps; k++){
        const raw = q.map((v, i) =>
          Math.max(-vmax * dt, Math.min(vmax * dt, tgt[i] - v)));
        if (raw.every((s) => Math.abs(s) < 1e-4)) break;    // arrived
        const step = this.projectStep(names, q, raw, dt);
        minSlack = Math.min(minSlack, this.minSlack());
        if (step.every((s) => Math.abs(s) < 1e-5)){          // pinned on the boundary
          blocked = true;
          break;
        }
        for (let i = 0; i < q.length; i++) q[i] += step[i];
        path.push(snap(q));
      }
      if (!blocked && path.length > maxSteps - 1) blocked = true;
      return { path, blocked, minSlack };
    },
  };
}
