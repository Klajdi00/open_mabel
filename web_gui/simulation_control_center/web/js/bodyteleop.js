// BodyTeleop — desktop WEBCAM full-body teleop: the Vision Pro path, mimicked
// with MediaPipe tasks-vision. PoseLandmarker (33 world landmarks, meters) gives
// body/arms/torso/legs + head orientation; HandLandmarker (2×21) gives fingers.
// The tracked human is synthesized into the SAME `teleop_frame` envelopes the
// headset streams (server/wire_protocol.py: head 4x4 + per-hand anchorTransform
// + named ARKit joint localTransforms), so the server-side retarget engine
// (controller/mabel + scripts/teleop TeleopEngine) drives head tracking, arm
// retargeting, per-finger hands, torso and base UNCHANGED.
//
// ── Frame convention (derived from controller/mabel/config.py AVP_TO_ROBOT) ──
// MediaPipe world coords (operator FACING the camera): x = image right,
// y = image down, z = depth away from camera, meters.
// Virtual ARKit operator world (what TeleopFrame carries): X = operator's
// right, Y = up, −Z = operator's forward (toward the camera). Therefore
//     X_arkit = −x_mp    (the selfie mirror: your right hand moving to YOUR
//                         right lands on the robot's right — AVP_TO_ROBOT then
//                         maps +X_a → +Y_robot = robot right)
//     Y_arkit = −y_mp    (up)
//     Z_arkit = +z_mp    (toward camera = −Z = forward → −X_robot = robot front)
// det(diag(−1,−1,1)) = +1: a proper rotation, so hand chirality is preserved.
// Positions are OPERATOR-RELATIVE and clutched like the headset: on start /
// recalibrate the shoulder-center world position is captured as the anchor;
// everything is sent relative to it (scaled by 0.38 m / measured shoulder
// width) and lifted to a nominal shoulder height so head-height (lift-follow)
// and head→hand offsets (contact retarget) look like a real standing operator.
//
// Everything math is exported as pure node-safe functions (unit-tested in
// tests/js/bodyteleop.test.mjs and replayed live by the validation harness).

// ── constants ────────────────────────────────────────────────────────────────
export const NOMINAL_SHOULDER_M = 0.38;  // nominal biacromial width → scale ref
export const ANCHOR_HEIGHT_M = 1.35;     // virtual shoulder-center height (ARKit Y)
export const VIS_T = 0.5;                // landmark visibility threshold
export const TRACK_LOSS_MS = 500;        // >0.5 s without tracking → idle frames
export const REGION_HOLD_S = 1.0;        // region hysteresis
export const SEND_HZ = 30;

// MediaPipe pose landmark indices (33-point BlazePose topology).
export const PL = {
  nose: 0, leftEye: 2, rightEye: 5, leftEar: 7, rightEar: 8,
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16, leftHip: 23, rightHip: 24,
  leftKnee: 25, rightKnee: 26, leftAnkle: 27, rightAnkle: 28,
};

// ARKit HandJoint rawValue names for the 21 MediaPipe hand landmarks, in
// MediaPipe order (exactly the engine's _NAMES_21; wire.py needs wrist +
// thumb_tip/index_tip, the per-finger retargeter reads the rest).
export const HAND_JOINT_NAMES = [
  'wrist',
  'thumb_knuckle', 'thumb_intermediate_base', 'thumb_intermediate_tip', 'thumb_tip',
  'index_knuckle', 'index_intermediate_base', 'index_intermediate_tip', 'index_tip',
  'middle_knuckle', 'middle_intermediate_base', 'middle_intermediate_tip', 'middle_tip',
  'ring_knuckle', 'ring_intermediate_base', 'ring_intermediate_tip', 'ring_tip',
  'little_knuckle', 'little_intermediate_base', 'little_intermediate_tip', 'little_tip',
];
// ARKit metacarpals MediaPipe lacks — synthesized on the wrist→knuckle segment
// (geometry.operator_frame prefers them for the natural palm frame).
export const METACARPALS = {
  index_metacarpal: [0, 5, 0.35], middle_metacarpal: [0, 9, 0.35],
  ring_metacarpal: [0, 13, 0.35], little_metacarpal: [0, 17, 0.35],
};

export const IDENT4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

// ── small vector helpers (arrays [x,y,z]) ────────────────────────────────────
export const v3 = (p) => (Array.isArray(p) ? [p[0], p[1], p[2]] : [p.x, p.y, p.z]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a); return n < 1e-9 ? null : scl(a, 1 / n); };
const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                            a[2] + (b[2] - a[2]) * t];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// MediaPipe camera frame → virtual ARKit operator frame (see header).
export const mpToArkit = (p) => [-p[0], -p[1], p[2]];

// Row-major homogeneous 4x4 from rotation COLUMNS [x̂, ŷ, ẑ] and translation.
export function mat4RT(R, t){
  const c = R || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return [c[0][0], c[1][0], c[2][0], t[0],
          c[0][1], c[1][1], c[2][1], t[1],
          c[0][2], c[1][2], c[2][2], t[2],
          0, 0, 0, 1];
}

// Anchor-relative, anthropometry-scaled ARKit world point (adds the nominal
// standing height so lift-follow / head→hand offsets are body-plausible).
export function worldPoint(p, anchor, scaleF){
  const a = mpToArkit(sub(v3(p), anchor));
  return [a[0] * scaleF, a[1] * scaleF + ANCHOR_HEIGHT_M, a[2] * scaleF];
}

// Deep NaN/Inf guard — never put a non-finite number on the wire.
export function finiteDeep(o){
  if (typeof o === 'number') return Number.isFinite(o);
  if (o == null || typeof o === 'boolean' || typeof o === 'string') return true;
  if (Array.isArray(o)) return o.every(finiteDeep);
  if (typeof o === 'object') return Object.values(o).every(finiteDeep);
  return true;
}

// ── clutch anchor: shoulder-center + anthropometric scale ────────────────────
export function computeAnchor(world, vis){
  const v = (i) => (vis && vis[i] != null ? vis[i] : 1);
  if (!world || v(PL.leftShoulder) < VIS_T || v(PL.rightShoulder) < VIS_T) return null;
  const L = v3(world[PL.leftShoulder]), R = v3(world[PL.rightShoulder]);
  const w = norm(sub(L, R));
  if (!(w > 0.05) || !Number.isFinite(w)) return null;
  return { anchor: scl(add(L, R), 0.5),
           scale: clamp(NOMINAL_SHOULDER_M / w, 0.5, 3.0),
           shoulderM: w };
}

// ── head pose: ARKit-style camera transform from nose/ears ──────────────────
// Head local axes: X̂ = operator right, Ŷ = up, ẑ = BACK (gaze = −Z, matching
// AVP_GAZE_LOCAL). Identity rotation ⇔ operator squarely facing the camera.
export function headTransform(world, vis, anchor, scaleF){
  const v = (i) => (vis && vis[i] != null ? vis[i] : 1);
  if (!world || v(PL.nose) < 0.3 || v(PL.leftEar) < 0.15 || v(PL.rightEar) < 0.15)
    return null;
  const nose = mpToArkit(v3(world[PL.nose]));
  const earL = mpToArkit(v3(world[PL.leftEar]));
  const earR = mpToArkit(v3(world[PL.rightEar]));
  const mid = scl(add(earL, earR), 0.5);
  const f = unit(sub(nose, mid));                 // gaze direction
  if (!f) return null;
  const z = scl(f, -1);
  let x = sub(earR, earL);                        // toward the operator's right
  x = unit(sub(x, scl(z, dot(x, z))));            // orthogonalize against ẑ
  if (!x) return null;
  const y = cross(z, x);
  // position: ear midpoint, back in MediaPipe space for the anchor mapping
  const midMp = scl(add(v3(world[PL.leftEar]), v3(world[PL.rightEar])), 0.5);
  return mat4RT([x, y, z], worldPoint(midMp, anchor, scaleF));
}

// ── anatomical hand decomposition / re-synthesis ─────────────────────────────
// The server's finger retargeter (scripts/teleop/hand.py HandOptimizer) consumes
// the METRIC positions of landmarks {1..4 thumb, kn/pip/tip per finger} relative
// to the wrist, rotated into the palm frame. Raw MediaPipe worldLandmarks carry
// per-frame bone-length noise and depth-compressed curls, so instead of passing
// them through we DECOMPOSE the 21 points into anatomical joint angles (abduction
// in the palm plane, MCP/PIP/DIP flexion about the finger's flexion axis, thumb
// CMC direction + MP/IP flexion) and RE-SYNTHESIZE an anatomically consistent
// skeleton with calibrated (temporally stable) bone lengths. Angles — not
// positions — are filtered, so the filter can never skew the hand's shape.
// Full derivation: simulation_control_center/docs/retargeting.tex §3.

// MediaPipe finger base indices (knuckle); chain is base, +1 PIP, +2 DIP, +3 tip.
export const FINGER_BASE = { index: 5, middle: 9, ring: 13, little: 17 };
// Anatomical clamp ranges [rad] for the decomposed angles.
export const FLEX_LIMITS = {
  mcp: [-0.35, 1.90], pip: [-0.30, 2.05], dip: [-0.30, 1.65], abd: [-0.50, 0.50],
  tmp: [-0.45, 1.30], tip: [-0.45, 1.70],
};

const clampRange = (v, [lo, hi]) => clamp(v, lo, hi);
// Signed angle from u to v about unit axis n̂ (u, v need not be unit).
export const signedAngle = (u, v, n) =>
  Math.atan2(dot(cross(u, v), n), dot(u, v));
// Rodrigues rotation of v about unit axis â by θ.
export function rotAxis(v, a, th){
  const c = Math.cos(th), s = Math.sin(th);
  return add(add(scl(v, c), scl(cross(a, v), s)), scl(a, dot(a, v) * (1 - c)));
}

// Palm frame [x̂ ŷ ẑ] (columns) from a hand-centered ARKit shape: ŷ = wrist→
// middle knuckle, ẑ = DORSAL palm normal (χ = +1 right / −1 left makes the same
// formula chirality-correct: fingers always flex toward −ẑ), x̂ = ŷ×ẑ.
export function palmFrame(shape, chirality){
  const w = shape[0];
  const chi = chirality === 'left' ? -1 : 1;
  const yr = sub(shape[9], w);
  const zr = scl(cross(sub(shape[5], w), sub(shape[17], w)), chi);
  const y = unit(yr);
  if (!y || norm(zr) < 1e-12) return null;
  const z = unit(sub(zr, scl(y, dot(zr, y))));
  if (!z) return null;
  return [cross(y, z), y, z];                     // columns [x̂, ŷ, ẑ]
}
const toPalm = (B, p) => [dot(p, B[0]), dot(p, B[1]), dot(p, B[2])];
const fromPalm = (B, c) => add(add(scl(B[0], c[0]), scl(B[1], c[1])), scl(B[2], c[2]));

// Decompose a hand-centered ARKit 21-point shape into anatomical angles +
// measured geometry. Returns null when the palm is degenerate.
//   fingers[f]  : { abd, mcp, pip, dip }  [rad]
//   thumb       : { dir, axis (palm coords, unit), mp, ip }
//   geom        : calibratable geometry — per finger { kn (palm coords), L[3] },
//                 thumb { cmc (palm coords), L[3] }
export function decomposeHand(shape, chirality){
  if (!shape || shape.length < 21) return null;
  const B = palmFrame(shape, chirality);
  if (!B) return null;
  const w = shape[0], z = B[2];
  const fingers = {}, geom = { fingers: {}, thumb: null };

  for (const [f, base] of Object.entries(FINGER_BASE)){
    const kn = sub(shape[base], w);
    const d1 = sub(shape[base + 1], shape[base]);
    const d2 = sub(shape[base + 2], shape[base + 1]);
    const d3 = sub(shape[base + 3], shape[base + 2]);
    const mPl = unit(sub(kn, scl(z, dot(kn, z))));
    if (!mPl) return null;
    // Abduction from the palm-plane projection of the proximal phalanx; the
    // projection degenerates at MCP ≈ 90° where anatomical splay locks to 0,
    // so it is faded out with the projection magnitude (smooth, singular-free).
    const d1pl = sub(d1, scl(z, dot(d1, z)));
    const wAbd = clamp(norm(d1pl) / (0.35 * (norm(d1) + 1e-9)), 0, 1);
    const d1plU = unit(d1pl);
    const abd = d1plU ? wAbd * signedAngle(mPl, d1plU, z) : 0;
    // Flexion axis â ⊥ the finger's sagittal plane; flexion > 0 curls to −ẑ.
    const r = rotAxis(mPl, z, abd);
    const a = unit(cross(z, r)) || cross(z, mPl);
    const u1 = unit(d1), u2 = unit(d2), u3 = unit(d3);
    if (!u1 || !u2 || !u3) return null;
    fingers[f] = {
      abd: clampRange(abd, FLEX_LIMITS.abd),
      mcp: clampRange(signedAngle(r, u1, a), FLEX_LIMITS.mcp),
      pip: clampRange(signedAngle(u1, u2, a), FLEX_LIMITS.pip),
      dip: clampRange(signedAngle(u2, u3, a), FLEX_LIMITS.dip),
    };
    geom.fingers[f] = { kn: toPalm(B, kn), L: [norm(d1), norm(d2), norm(d3)] };
  }

  // Thumb: CMC base + metacarpal DIRECTION are kept as measured (2-DOF
  // opposition — the palm frame cannot parameterize it with one angle), then
  // MP/IP flexion about the thumb-plane normal.
  const c = sub(shape[1], w);
  const t = sub(shape[2], shape[1]);
  const pMp = sub(shape[3], shape[2]);
  const pIp = sub(shape[4], shape[3]);
  const tU = unit(t), mpU = unit(pMp), ipU = unit(pIp);
  if (!tU || !mpU || !ipU) return null;
  let axis = cross(tU, mpU);
  if (norm(axis) < 0.15) axis = cross(tU, ipU);
  if (norm(axis) < 0.15) axis = cross(z, tU);       // straight thumb: angles ≈ 0
  const aT = unit(axis);
  if (!aT) return null;
  const thumb = {
    dir: toPalm(B, tU), axis: toPalm(B, aT),
    mp: clampRange(signedAngle(tU, mpU, aT), FLEX_LIMITS.tmp),
    ip: clampRange(signedAngle(mpU, ipU, aT), FLEX_LIMITS.tip),
  };
  geom.thumb = { cmc: toPalm(B, c), L: [norm(t), norm(pMp), norm(pIp)] };
  return { fingers, thumb, geom, B };
}

// Exact FK of the anatomical hand model: angles + geometry (+ current palm
// orientation B) → hand-centered ARKit 21-point shape. Inverse of
// decomposeHand for in-range angles (round-trip unit-tested).
export function synthesizeHand(dec, geomCal){
  const g = geomCal || dec.geom;
  const B = dec.B, z = B[2];
  const out = new Array(21);
  out[0] = [0, 0, 0];
  for (const [f, base] of Object.entries(FINGER_BASE)){
    const A = dec.fingers[f], G = g.fingers[f];
    const kn = fromPalm(B, G.kn);
    const mPl = unit(sub(kn, scl(z, dot(kn, z)))) || B[1];
    const r = rotAxis(mPl, z, A.abd);
    const a = unit(cross(z, r)) || cross(z, mPl);
    const u1 = rotAxis(r, a, A.mcp);
    const u2 = rotAxis(u1, a, A.pip);
    const u3 = rotAxis(u2, a, A.dip);
    out[base] = kn;
    out[base + 1] = add(kn, scl(u1, G.L[0]));
    out[base + 2] = add(out[base + 1], scl(u2, G.L[1]));
    out[base + 3] = add(out[base + 2], scl(u3, G.L[2]));
  }
  const T = dec.thumb, GT = g.thumb;
  const c = fromPalm(B, GT.cmc);
  const tU = fromPalm(B, T.dir);
  const aT = fromPalm(B, T.axis);
  const uMp = rotAxis(tU, aT, T.mp);
  const uIp = rotAxis(uMp, aT, T.ip);
  out[1] = c;
  out[2] = add(c, scl(tU, GT.L[0]));
  out[3] = add(out[2], scl(uMp, GT.L[1]));
  out[4] = add(out[3], scl(uIp, GT.L[2]));
  return out;
}

// A TRUE pinch must survive re-synthesis exactly (the server's grasp curve and
// double-pinch clutch read the thumb↔index tip gap in meters): inside snapDist
// the thumb tip (and half the IP joint) is slid along the gap direction so the
// re-synthesized gap approaches the MEASURED gap as the pinch closes.
export function preservePinch(refined, raw, snapDist = 0.05){
  const gapRaw = norm(sub(raw[4], raw[8]));
  if (!(gapRaw < snapDist)) return refined;
  const gapNew = norm(sub(refined[4], refined[8]));
  const wgt = 1 - gapRaw / snapDist;                // 0 at snapDist → 1 at contact
  const gStar = wgt * gapRaw + (1 - wgt) * gapNew;
  const dir = unit(sub(refined[8], refined[4]));
  if (!dir) return refined;
  const d = gapNew - gStar;
  const out = refined.slice();
  out[4] = add(refined[4], scl(dir, d));
  out[3] = add(refined[3], scl(dir, 0.5 * d));
  return out;
}

// One-shot refine (stateless): decompose → re-synthesize with the frame's own
// geometry (normalizes nothing across time but enforces anatomical consistency).
export function refineHandShape(shape, chirality){
  const dec = decomposeHand(shape, chirality);
  if (!dec) return shape;
  return preservePinch(synthesizeHand(dec), shape);
}

// ── bone-length depth restoration ────────────────────────────────────────────
// MediaPipe world landmarks systematically ATTENUATE depth (z): a finger curled
// toward the camera reads much flatter than it is, which is the #1 cause of
// "the fingers don't curl". Bone lengths are constant, so given CALIBRATED
// lengths L (max-tracked while the bones sweep through the image plane, where
// they measure true), the lost |z| of every bone is recoverable from the rigid
// constraint  b_z'^2 = L² − b_x² − b_y²  keeping the measured z SIGN. Applied
// down the kinematic tree so children inherit the corrected parents.
export const HAND_BONES = [
  [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
  [0, 1], [1, 2], [2, 3], [3, 4],
];
// Anthropometric priors: each phalanx length as a fraction of the wrist→middle-
// knuckle (palm) length, per HAND_BONES index (null = palm bone, measured
// directly — palm bones lie near the image plane when the operator faces the
// camera, so THEY calibrate true and anchor the metric for the rest). Ratios
// follow standard hand-proportion data (≈ Buryanov & Kotiuk 2010) and this
// repo's canonical hand geometry.
export const BONE_PRIOR = [
  null, 0.423, 0.254, 0.212,      // index: meta measured; prox/mid/dist
  null, 0.466, 0.296, 0.222,      // middle
  null, 0.434, 0.275, 0.212,      // ring
  null, 0.339, 0.212, 0.180,      // little
  0.399, 0.487, 0.339, 0.296,     // thumb: cmc offset, meta, prox, dist
];
const PALM_BONE_MID = 4;          // HAND_BONES[4] = [0, 9] = wrist→middle knuckle
// The constraint's z-estimate has unbounded noise gain as a bone approaches
// the image plane (d|z|/dL → ∞ at ρ → 0), so the correction is weighted by the
// SIGNIFICANCE of the length deficit relative to the landmark noise floor:
// only deficits ≥ lo(L) = max(REST_LO, REST_KSIG·σ/L) restore, fully from
// lo(L)+REST_RAMP. Short (distal) bones therefore restore only under extreme,
// unambiguous attenuation — their deficit noise (σ/L) is too large otherwise.
export const REST_LO = 0.12, REST_RAMP = 0.13;
export const REST_SIGMA_M = 0.004;      // assumed bone-length noise floor [m]
export const REST_KSIG = 2.5;
export function restoreDepth(shape, lengths){
  const out = shape.slice();
  for (let i = 0; i < HAND_BONES.length; i++){
    const [pa, ch] = HAND_BONES[i];
    const b = sub(shape[ch], shape[pa]);            // measured bone (raw frame)
    const L = lengths[i];
    let bz = b[2];
    if (L){
      const meas = norm(b);
      const deficit = (L - meas) / L;
      const lo = Math.max(REST_LO, REST_KSIG * REST_SIGMA_M / L);
      const w = clamp((deficit - lo) / REST_RAMP, 0, 1);
      const rho2 = L * L - b[0] * b[0] - b[1] * b[1];
      if (w > 0 && rho2 > 0){
        const zr = Math.max(Math.sqrt(rho2), Math.abs(b[2]));   // only GROW |z|
        bz = b[2] + w * ((b[2] < 0 ? -zr : zr) - b[2]);
      }
    }
    out[ch] = add(out[pa], [b[0], b[1], bz]);
  }
  return out;
}

// GLOBAL per-hand depth-attenuation estimate. Monocular hand landmarkers get
// depth from a weakly-supervised head, and its error is (to first order) one
// multiplicative factor κ per hand:  ẑ = κ·z. Each Z-DOMINANT bone i measures
// it:  κ_i = |b̂_z,i| / sqrt(L_i² − |b̂_xy,i|²)  (rigid-length constraint) —
// well-conditioned exactly where the per-bone estimate is identifiable. The
// weighted mean (weights ∝ L·|b̂_z|, the sensitivity) gives κ̂; z/κ̂ then
// corrects ALL landmarks, including the mid-chain bones whose own deficit is
// below the noise floor. Returns {kappa, weight} or null (no z-dominant bone).
export function estimateZScale(shape, lengths){
  let num = 0, den = 0;
  for (let i = 0; i < HAND_BONES.length; i++){
    const [pa, ch] = HAND_BONES[i];
    const b = sub(shape[ch], shape[pa]);
    const L = lengths[i];
    if (!L || L < 0.015) continue;
    const bz = Math.abs(b[2]);
    const rho2 = L * L - b[0] * b[0] - b[1] * b[1];
    // identifiability: the bone must be clearly out of the image plane both as
    // measured AND as implied by the length constraint.
    if (bz < Math.max(0.35 * L, 0.008) || rho2 < (0.3 * L) * (0.3 * L)) continue;
    const k = bz / Math.sqrt(rho2);
    const w = L * bz;
    num += w * clamp(k, 0.3, 2.0);
    den += w;
  }
  if (den < 1e-5) return null;
  return { kappa: num / den, weight: den };
}

// Stateful per-hand tracker: calibrates bone lengths / knuckle geometry with a
// robust EMA and one-euro-filters the ANGLES (not the positions), so filtering
// can never lag one axis of a curl behind the others (shape-skew-free).
export class HandShapeTracker {
  // Angle-domain one-euro: LOW static cutoff (0.8 Hz — beats the legacy
  // position smoothing on standstill jitter) with a HIGH velocity gain (a fast
  // curl at ~8 rad/s opens the cutoff to ~10 Hz — near-zero lag). Filtering
  // angles instead of positions means the filter can never skew the SHAPE.
  constructor(chirality, { minCutoff = 0.8, beta = 1.2 } = {}){
    this.chirality = chirality;
    this.geo = null;                    // calibrated geometry (EMA)
    this.n = 0;
    this.len = null;                    // calibrated HAND_BONES lengths
    this.kappa = null;                  // per-hand global z-attenuation EMA
    this._f = new Map();                // angle name → OneEuro
    this._opts = { minCutoff, beta };
  }
  _flt(key, v, t){
    let f = this._f.get(key);
    if (!f){ f = new OneEuro(this._opts); this._f.set(key, f); }
    return f.filter(v, t);
  }
  // Bone-length calibration for the depth restoration: apparent length only
  // ever SHRINKS under depth attenuation / foreshortening, so track toward an
  // upper quantile (asymmetric EMA: rises faster than it decays) — the true
  // length is learned whenever the bone sweeps near the image plane. The
  // asymmetry is kept mild so landmark noise cannot ratchet the estimate up.
  _calLen(shape){
    const meas = HAND_BONES.map(([a, b]) => norm(sub(shape[b], shape[a])));
    if (!this.len) this.len = meas.slice();
    else {
      const aUp = Math.max(1 / (this.n + 1), 0.06), aDn = 0.02;
      this.len = this.len.map((L, i) => {
        const m = clamp(meas[i], 0, L * 1.25);
        return L + (m > L ? aUp : aDn) * (m - L);
      });
    }
    // Anthropometric floor: a bone that has never swept near the image plane
    // (e.g. a fist held from the first frame) still gets a usable length from
    // the palm-anchored prior — without it the depth correction is CIRCULAR
    // (lengths calibrated from attenuated z hide the attenuation).
    const Lp = this.len[PALM_BONE_MID];
    return this.len.map((L, i) =>
      BONE_PRIOR[i] ? Math.max(L, 0.85 * BONE_PRIOR[i] * Lp) : L);
  }
  _calGeom(geom){
    // EMA rate 1/(n+1) for fast initial convergence, floored at 0.02 so the
    // estimate stays adaptive; per-update bone-length change clamped to ±20 %.
    if (!this.geo){
      this.geo = JSON.parse(JSON.stringify(geom));
      this.n = 1;
      return this.geo;
    }
    const a = Math.max(1 / (this.n + 1), 0.02);
    const mix = (o, m) => o + a * (clamp(m, o * 0.8, o * 1.25) - o);
    for (const f of Object.keys(FINGER_BASE)){
      const G = this.geo.fingers[f], M = geom.fingers[f];
      G.kn = lerp3(G.kn, M.kn, a);
      G.L = G.L.map((L, i) => mix(L, M.L[i]));
    }
    this.geo.thumb.cmc = lerp3(this.geo.thumb.cmc, geom.thumb.cmc, a);
    this.geo.thumb.L = this.geo.thumb.L.map((L, i) => mix(L, geom.thumb.L[i]));
    this.n++;
    return this.geo;
  }
  update(shapeRaw, t){
    // 1) depth correction on the RAW landmarks: (a) global per-hand z
    //    attenuation κ̂ (EMA'd; identified from z-dominant bones), then
    //    (b) per-bone length restoration for the residuals. THEN the
    //    anatomical decomposition of the corrected shape.
    const lens = this._calLen(shapeRaw);
    const est = estimateZScale(shapeRaw, lens);
    if (est && est.weight > 2e-4){
      const a = clamp(est.weight / 2e-3, 0.05, 0.25);       // confidence-scaled EMA
      this.kappa = this.kappa == null ? est.kappa
        : this.kappa + a * (est.kappa - this.kappa);
    }
    const k = clamp(this.kappa == null ? 1 : this.kappa, 0.4, 1.6);
    const zFixed = Math.abs(k - 1) < 0.03 ? shapeRaw
      : shapeRaw.map((p) => [p[0], p[1], p[2] / k]);        // wrist z = 0: safe
    const shape = restoreDepth(zFixed, lens);
    const dec = decomposeHand(shape, this.chirality);
    if (!dec) return shapeRaw;                       // degenerate: raw passthrough
    const geo = this._calGeom(dec.geom);
    for (const f of Object.keys(dec.fingers))
      for (const k of ['abd', 'mcp', 'pip', 'dip'])
        dec.fingers[f][k] = this._flt(f + k, dec.fingers[f][k], t);
    dec.thumb.mp = this._flt('tmp', dec.thumb.mp, t);
    dec.thumb.ip = this._flt('tip', dec.thumb.ip, t);
    // thumb direction/axis: filter palm-frame components, renormalize, and keep
    // the axis ⊥ the direction (the pair parameterizes the thumb plane).
    const dir = unit([0, 1, 2].map((i) => this._flt('tdir' + i, dec.thumb.dir[i], t)));
    let axis = [0, 1, 2].map((i) => this._flt('taxis' + i, dec.thumb.axis[i], t));
    if (dir){
      dec.thumb.dir = dir;
      axis = unit(sub(axis, scl(dir, dot(axis, dir))));
      if (axis) dec.thumb.axis = axis;
    }
    return preservePinch(synthesizeHand(dec, geo), shape);
  }
}

// ── palm-plane reflection (server hand-convention compensation) ──────────────
// MEASURED FACT (validation harness, 2026-07): the sim ORCA "right" hand curls
// its fingers toward +f̂×(pinky−index) of its own palm frame, while every TRUE
// right-hand skeleton (MediaPipe, anatomical invariant) curls toward the
// OPPOSITE side — i.e. the model+solver convention matches a MIRRORED skeleton
// (the reference mock/synthetic hands were built mirrored the same way, which
// is why the shipped stack is self-consistent). Feeding a true skeleton makes
// the keypoint IK EXTEND a curling finger. Until the model convention is
// resolved at the source, the client compensates: reflect the hand across its
// own palm plane (a chirality flip), TRANSLATED so the MIDDLE FINGERTIP is
// unchanged — the arm's contact target, the pinch distances, and the palm
// orientation R_op are all preserved exactly; only the finger-shape reading
// flips to the server's convention. See docs/retargeting.tex §5.4.
export function reflectPalm(shape, chirality){
  const B = palmFrame(shape, chirality);
  if (!B) return shape;
  const n = B[2];
  const w = shape[0];
  const mid0 = shape[12];
  const refl = shape.map((p) => {
    const d = sub(p, w);
    return add(w, sub(d, scl(n, 2 * dot(d, n))));
  });
  const pin = sub(mid0, refl[12]);       // pin the middle fingertip
  return refl.map((p) => add(p, pin));
}

// ── two-hand wrist-pair fusion ───────────────────────────────────────────────
// The two wrists' RELATIVE placement decides whether fingertips can meet across
// hands. PoseLandmarker world wrists carry cm-level relative error, while both
// hands' NORMALIZED image landmarks live in ONE image frame with px accuracy.
// Fuse: keep the pair midpoint + depth split from the pose (image can't see
// depth), replace the image-plane (x, y) inter-wrist vector with the image
// measurement converted to meters via the per-hand knuckle-span scale.
//   pwL/pwR : MediaPipe world pose wrists [x,y,z]
//   img     : { left/right: {wrist:{x,y}, idx:{x,y}, pky:{x,y}}, aspect }
//   spanL/R : metric index→little knuckle span per hand (world landmarks)
// Returns [pwL', pwR'] or null when the image data is missing/inconsistent.
export function fuseWristPair(pwL, pwR, img, spanL, spanR){
  if (!pwL || !pwR || !img || !img.left || !img.right) return null;
  const ar = img.aspect || (4 / 3);
  const ac = (p) => [p.x * ar, p.y];
  const span2 = (h) => Math.hypot(ac(h.idx)[0] - ac(h.pky)[0],
                                  ac(h.idx)[1] - ac(h.pky)[1]);
  const siL = span2(img.left), siR = span2(img.right);
  if (!(siL > 1e-4) || !(siR > 1e-4) || !(spanL > 0.01) || !(spanR > 0.01)) return null;
  const sL = spanL / siL, sR = spanR / siR;
  if (Math.max(sL, sR) / Math.min(sL, sR) > 1.8) return null;   // depth mismatch
  const s = 0.5 * (sL + sR);
  const dImg = [(ac(img.right.wrist)[0] - ac(img.left.wrist)[0]) * s,
                (ac(img.right.wrist)[1] - ac(img.left.wrist)[1]) * s];
  // PROXIMITY gate: the fusion matters (and its perspective error vanishes)
  // when the hands are NEAR each other — the fingertip-contact regime. When
  // far apart, the span scale (estimated at knuckle depth) vs wrist depth
  // mismatch is ~4 % of the separation, so fade the correction out.
  const sep = Math.hypot(dImg[0], dImg[1]);
  const w = clamp((0.50 - sep) / (0.50 - 0.30), 0, 1);
  if (w <= 0) return null;
  const dPose = sub(pwR, pwL);
  let delta = scl([dImg[0] - dPose[0], dImg[1] - dPose[1], 0], w);  // x,y only
  const dn = norm(delta);
  if (dn > 0.30) delta = scl(delta, 0.30 / dn);                 // sanity clamp
  const half = scl(delta, 0.5);
  return [sub(pwL, half), add(pwR, half)];
}

// Natural palm frame from the hand-local ARKit shape (index 0 = wrist):
// ŷ = wrist→middle-knuckle, ẑ = palm normal, x̂ completes (retarget._natural_frame).
export function wristRotation(shape){
  const w = shape[0];
  const fwd = unit(sub(shape[9], w));
  const nrm0 = cross(sub(shape[5], w), sub(shape[17], w));
  if (!fwd || norm(nrm0) < 1e-9) return null;
  const n0 = unit(nrm0);
  const side = unit(cross(fwd, n0));
  if (!side) return null;
  return [side, fwd, cross(side, fwd)];           // columns [x̂, ŷ, ẑ]
}

// ── one hand → wire HandPose ─────────────────────────────────────────────────
// hand21: MediaPipe hand worldLandmarks (hand-centered, metric); its PLACEMENT
// comes from the pose wrist so the arm carries it through the operator's
// workspace. opts (all optional — omitting them reproduces the legacy behavior):
//   tracker    : HandShapeTracker — anatomical refine + angle-domain filtering
//   refine     : true — stateless refineHandShape (when no tracker given)
//   scaleShape : true — scale the hand SHAPE by scaleF too, so the WHOLE
//                operator (wrist spacing AND finger geometry) goes through ONE
//                similarity transform — required for cross-hand fingertip
//                contact to be preserved exactly (docs/retargeting.tex §5).
//   t          : sample time [s] for the tracker's filters
export function handPose(hand21, chirality, poseWrist, anchor, scaleF, opts = {}){
  if (!hand21 || hand21.length < 21 || !poseWrist) return null;
  const w0 = v3(hand21[0]);
  let shape = hand21.map((p) => mpToArkit(sub(v3(p), w0)));
  if (opts.tracker) shape = opts.tracker.update(shape, opts.t || 0);
  else if (opts.refine) shape = refineHandShape(shape, chirality);
  if (opts.mirrorPalm) shape = reflectPalm(shape, chirality);
  if (opts.scaleShape) shape = shape.map((p) => scl(p, scaleF));
  const wristWorld = worldPoint(poseWrist, anchor, scaleF);
  const R = wristRotation(shape);
  const joints = [];
  const push = (name, local, Rj) => joints.push({
    joint: name, localTransform: { matrix: mat4RT(Rj || null, local) },
    isTracked: true });
  HAND_JOINT_NAMES.forEach((name, i) => push(name, shape[i], i === 0 ? R : null));
  for (const [name, [a, b, t]] of Object.entries(METACARPALS))
    push(name, lerp3(shape[a], shape[b], t));
  return { chirality, anchorTransform: { matrix: mat4RT(null, wristWorld) },
           joints, isTracked: true };
}

// ── adaptive control region ──────────────────────────────────────────────────
export function trackingParts(vis, hands){
  const v = (i) => (vis && vis[i] != null ? vis[i] : 0);
  return {
    head: v(PL.nose) >= VIS_T && Math.max(v(PL.leftEar), v(PL.rightEar)) >= VIS_T,
    hands: !!(hands && (hands.left || hands.right)),
    torso: Math.min(v(PL.leftShoulder), v(PL.rightShoulder),
                    v(PL.leftHip), v(PL.rightHip)) >= VIS_T,
    legs: Math.min(v(PL.leftKnee), v(PL.rightKnee),
                   v(PL.leftAnkle), v(PL.rightAnkle)) >= VIS_T,
  };
}

export function regionForParts(p){
  if (p.torso && p.legs) return 'whole_body';
  if (p.torso) return 'upper_body';
  return 'arm';
}

// Region switch only after the candidate persists REGION_HOLD_S (~1 s).
export class RegionHysteresis {
  constructor(holdS = REGION_HOLD_S){
    this.holdS = holdS; this.stable = null; this._cand = null; this._candT = 0;
  }
  update(region, now){
    if (this.stable === null){ this.stable = region; return this.stable; }
    if (region === this.stable){ this._cand = null; return this.stable; }
    if (region !== this._cand){ this._cand = region; this._candT = now; return this.stable; }
    if (now - this._candT >= this.holdS){ this.stable = region; this._cand = null; }
    return this.stable;
  }
}

// ── one-euro filter (per scalar / per landmark) ──────────────────────────────
export class OneEuro {
  constructor({ minCutoff = 1.2, beta = 0.35, dCutoff = 1.0 } = {}){
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.t = null; this.x = 0; this.dx = 0;
  }
  static alpha(cutoff, dt){ const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(x, t){
    if (this.t == null || t - this.t > 0.5 || t <= this.t){
      this.t = t; this.x = x; this.dx = 0; return x;
    }
    const dt = t - this.t; this.t = t;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    this.dx += ad * ((x - this.x) / dt - this.dx);
    const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt);
    this.x += a * (x - this.x);
    return this.x;
  }
}
export class OneEuroPoint {
  constructor(opts){ this.f = [new OneEuro(opts), new OneEuro(opts), new OneEuro(opts)]; }
  filter(p, t){
    return { x: this.f[0].filter(p.x, t), y: this.f[1].filter(p.y, t),
             z: this.f[2].filter(p.z, t), visibility: p.visibility };
  }
}

// ── handedness routing: nearest pose wrist beats the (mirror-prone) label ────
// dets: [{idx, wrist:{x,y}, label}] in normalized image coords; poseNorm: 33 pts.
export function routeHands(dets, poseNorm){
  const out = { left: null, right: null };
  if (!dets || !dets.length) return out;
  const used = new Set();
  if (poseNorm){
    const pw = { left: poseNorm[PL.leftWrist], right: poseNorm[PL.rightWrist] };
    const cands = [];
    for (const d of dets)
      for (const side of ['left', 'right']){
        if (!pw[side]) continue;
        const dx = d.wrist.x - pw[side].x, dy = d.wrist.y - pw[side].y;
        cands.push({ side, idx: d.idx, d2: dx * dx + dy * dy });
      }
    cands.sort((a, b) => a.d2 - b.d2);
    for (const c of cands){
      if (out[c.side] !== null || used.has(c.idx) || c.d2 > 0.05) continue;
      out[c.side] = c.idx; used.add(c.idx);
    }
  }
  for (const d of dets){                          // label fallback
    if (used.has(d.idx)) continue;
    const side = d.label === 'Left' ? 'left' : 'right';
    const other = side === 'left' ? 'right' : 'left';
    if (out[side] === null){ out[side] = d.idx; used.add(d.idx); }
    else if (out[other] === null){ out[other] = d.idx; used.add(d.idx); }
  }
  return out;
}

// ── frame synthesis ──────────────────────────────────────────────────────────
// sample: { t (s), pose: {world:[33], vis:[33]},
//           hands: {left:[21]|null, right:[21]|null},
//           img?:  {left/right: {wrist,idx,pky (normalized {x,y})}, aspect} }
// calib:  { anchor:[3] (mp), scale }        → wire payload dict, or null (unusable).
// opts (all optional; omitted = legacy):
//   trackers  : {left, right: HandShapeTracker} anatomical refine + filtering
//   refine    : stateless refine when no tracker
//   mirrorPalm: reflect the hand across its palm plane, middle-tip pinned —
//               compensates the server/model finger-curl convention (reflectPalm)
//   scaleShape: hand shape follows the operator similarity transform (see handPose)
//   fuse      : image-frame wrist-pair fusion when both hands are tracked
//   elbows    : attach leftElbow/rightElbow transforms (pose landmarks) so the
//               server's arm IK can match the operator's elbow direction
export function synthesizeFrame(sample, calib, seq, opts = {}){
  if (!sample || !sample.pose || !sample.pose.world || !calib) return null;
  const { world, vis } = sample.pose;
  const headM = headTransform(world, vis, calib.anchor, calib.scale);
  if (!headM) return null;
  const hands = sample.hands || {};
  // two-hand wrist-pair fusion (image-frame relative placement)
  let pw = { left: world[PL.leftWrist], right: world[PL.rightWrist] };
  if (opts.fuse && hands.left && hands.right && sample.img && pw.left && pw.right){
    const spanOf = (h21) => norm(sub(v3(h21[5]), v3(h21[17])));
    const fused = fuseWristPair(v3(pw.left), v3(pw.right), sample.img,
                                spanOf(hands.left), spanOf(hands.right));
    if (fused) pw = { left: fused[0], right: fused[1] };
  }
  const mk = (h21, side) => (h21 && pw[side]
    ? handPose(h21, side, pw[side], calib.anchor, calib.scale, {
        tracker: opts.trackers && opts.trackers[side],
        refine: opts.refine, mirrorPalm: opts.mirrorPalm,
        scaleShape: opts.scaleShape, t: sample.t })
    : null);
  const payload = {
    sequence: seq, timestamp: sample.t,
    head: { transform: { matrix: headM }, trackingState: 'normal' },
    leftHand: mk(hands.left, 'left'),
    rightHand: mk(hands.right, 'right'),
    mode: 'Arms & Hands',
    navJoystick: { lx: 0, ly: 0, rx: 0, ry: 0 },
  };
  if (opts.elbows){
    const v = (i) => (vis && vis[i] != null ? vis[i] : 1);
    for (const [key, wi, hand] of [['leftElbow', PL.leftElbow, payload.leftHand],
                                   ['rightElbow', PL.rightElbow, payload.rightHand]]){
      if (hand && world[wi] && v(wi) >= VIS_T)
        payload[key] = { matrix: mat4RT(null,
          worldPoint(world[wi], calib.anchor, calib.scale)) };
    }
  }
  return finiteDeep(payload) ? payload : null;
}

// SAFETY idle frame: identity head marked notAvailable (server treats it as
// non-claiming / untracked), untracked hands, zeroed navJoystick.
export function idleFrame(seq, t){
  const hand = (side) => ({ chirality: side,
    anchorTransform: { matrix: [...IDENT4] }, joints: [], isTracked: false });
  return { sequence: seq, timestamp: t,
    head: { transform: { matrix: [...IDENT4] }, trackingState: 'notAvailable' },
    leftHand: hand('left'), rightHand: hand('right'),
    mode: 'Arms & Hands', navJoystick: { lx: 0, ly: 0, rx: 0, ry: 0 } };
}

// ── browser runtime ──────────────────────────────────────────────────────────
// simulation_control_center/web/vendor/fetch_mediapipe.sh.
const CDN = '/vendor/mediapipe';
const POSE_MODEL = '/vendor/mediapipe/pose_landmarker_lite.task';
const HAND_MODEL = '/vendor/mediapipe/hand_landmarker.task';
// Reject a hung promise so the webcam button never sticks on "loading".
function withTimeout(promise, ms, msg){
  return Promise.race([promise, new Promise((_, rej) =>
    setTimeout(() => rej(new Error(msg || `timed out after ${ms}ms`)), ms))]);
}


const POSE_LINKS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],       // shoulders + arms
  [11, 23], [12, 24], [23, 24],                           // torso box
  [23, 25], [25, 27], [24, 26], [26, 28],                 // legs
  [7, 0], [8, 0],                                         // ears → nose
];
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

export class BodyTeleop {
  constructor(link, ctl, viewport, els = {}){
    this.link = link;
    this.ctl = ctl;
    this.viewport = viewport;
    this.els = els;                 // {button,status,regionChip,partsChip,preview,
    this._on = false;               //  video,canvas,regionSeg,onEngage,onStop,toast}
    this.seq = 0;
    this._filters = new Map();      // landmark one-euro filters
  }

  toggle(){ return this._on ? this.stop() : this.start(); }
  get active(){ return this._on; }
  _toast(msg, err){ this.els.toast && this.els.toast(msg, err); }

  // ── loading overlay (shown while the tracker + models load) ────────────────
  _setLoad(pct, step){
    this._loadPct = Math.max(0, Math.min(100, pct));
    if (this.els.loadFill) this.els.loadFill.style.width = this._loadPct + '%';
    if (step && this.els.loadStep) this.els.loadStep.textContent = step;
  }
  // indeterminate creep toward a cap while an un-measurable await runs, so the
  // bar keeps easing forward instead of freezing at the last milestone
  _creepTo(cap){
    this._stopCreep();
    this._creep = setInterval(() => {
      const cur = this._loadPct || 0;
      if (cur >= cap){ this._stopCreep(); return; }
      this._setLoad(cur + Math.max(0.4, (cap - cur) * 0.06));
    }, 350);
  }
  _stopCreep(){ if (this._creep){ clearInterval(this._creep); this._creep = null; } }
  _showLoading(on){
    const s = this.els.status;
    if (!s) return;
    if (on){ s.classList.remove('hidden'); s.classList.add('webcam-active', 'loading'); }
    else s.classList.remove('loading');
    this.els.preview && this.els.preview.classList.remove('hidden');
  }

  async start(){
    if (this._on) return;
    // Refuse while the robot is limp. Whole-body retarget on a de-energised
    // robot streams frames nothing acts on, and the operator watches a loading
    // spinner conclude in silence. Same rule as the Home glide.
    if (this.ctl && (this.ctl.estop || !this.ctl.torque)){
      this._toast('energise the robot first — webcam teleop needs torque on', true);
      return;
    }
    const b = this.els.button;
    if (b){ b.disabled = true; b.textContent = '… loading tracker'; }
    // The MediaPipe bundles are large; without visible progress a slow load
    // reads as a crash. Stages are real milestones, and `_creepTo` only ever
    // creeps WITHIN the current stage so the bar never over-claims.
    this._showLoading(true);
    this._setLoad(5, 'loading body tracker…');   // _setLoad takes a PERCENT
    this._creepTo(45);
    try {
      // Both awaits are bounded: a blocked model fetch and an un-answered
      // camera-permission prompt are the two ways this hangs forever, and a
      // spinner that never resolves is worse than an error you can act on.
      if (!this._lm)
        await withTimeout(this._loadModels(), 60000,
                          'tracker load timed out — the MediaPipe bundles are ~13 MB; '
                          + 'check /vendor/mediapipe is being served');
      this._setLoad(55, 'tracker ready — opening the camera…');
      this._creepTo(85);
      this.els.video.srcObject = await withTimeout(
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }),
        20000, 'camera permission not granted');
      await this.els.video.play();
      this._setLoad(100, 'camera live');
    } catch (err){
      this._stopCreep();
      this._showLoading(false);
      console.warn('body teleop:', err);
      this._toast('webcam teleop unavailable: ' + (err && err.message || err), true);
      if (b){ b.disabled = false; b.textContent = '▶ Webcam Teleop'; }
      return;
    }
    // engage
    this._on = true;
    this.seq = 0;
    this._calib = null;
    this._hyst = new RegionHysteresis();
    this._sentRegion = null;
    this._lastGood = 0;
    this._lastIdle = 0;
    this._lastTick = 0;
    this._filters.clear();
    // Per-hand anatomical shape trackers: angle-domain filtering + bone-length
    // calibration (see HandShapeTracker) — reset per session like the filters.
    this._handTrack = { left: new HandShapeTracker('left'),
                        right: new HandShapeTracker('right') };
    this.ctl.webcam = true;
    this.ctl.teleop = true;
    // Declare RETARGET immediately, before a single frame goes out. The region
    // hysteresis below only re-declares on a CHANGE and is gated on the head
    // being tracked, so an operator whose head never resolves (or whose first
    // stable region happens to equal the default) never entered retarget at all
    // and the robot ignored every frame.
    this._sentRegion = null;
    this._sendRegion(this.ctl.region || 'arm');
    this.els.status && this.els.status.classList.remove('hidden');
    this.els.preview && this.els.preview.classList.remove('hidden');
    if (b){ b.disabled = false; b.textContent = '■ Stop Webcam Teleop'; b.classList.add('on'); }
    this._stopCreep();
    this._showLoading(false);
    this._setChips(null, null, 'stand in view — capturing anchor…');
    this.els.onEngage && this.els.onEngage();
    this._toast('webcam teleop: tracking… face the camera');
    const loop = () => { if (this._on){ this._tick(); this._raf = requestAnimationFrame(loop); } };
    this._raf = requestAnimationFrame(loop);
  }

  stop(){
    this._stopCreep();
    this._showLoading(false);
    if (!this._on) return;
    this._on = false;
    cancelAnimationFrame(this._raf);
    // SAFETY: idle head (non-claiming) + untracked hands + zero navJoystick,
    // then the page's control spec is re-sent by onStop (applyControlSpec).
    try { this.link.send('teleop_frame', idleFrame(this.seq++, Date.now() / 1000)); }
    catch { /* link down */ }
    this.ctl.webcam = false;
    this.ctl.teleop = false;
    this._sentRegion = null;
    const v = this.els.video;
    if (v && v.srcObject){ v.srcObject.getTracks().forEach((t) => t.stop()); v.srcObject = null; }
    this.els.status && this.els.status.classList.add('hidden');
    this.els.preview && this.els.preview.classList.add('hidden');
    const b = this.els.button;
    if (b){ b.textContent = '▶ Webcam Teleop'; b.classList.remove('on'); }
    this.els.onStop && this.els.onStop();
    this._toast('webcam teleop stopped');
  }
  destroy(){ this.stop(); }

  async _loadModels(){
    const vision = await import(`${CDN}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    this._lm = {
      pose: await vision.PoseLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: POSE_MODEL }, runningMode: 'VIDEO', numPoses: 1 }),
      hand: await vision.HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: HAND_MODEL }, runningMode: 'VIDEO', numHands: 2 }),
    };
  }

  _smooth(key, p, t){
    let f = this._filters.get(key);
    if (!f){ f = new OneEuroPoint(); this._filters.set(key, f); }
    return f.filter(p, t);
  }

  _tick(){
    const now = performance.now();
    if (now - this._lastTick < 1000 / SEND_HZ) return;      // ~30 Hz
    this._lastTick = now;
    const video = this.els.video;
    if (!video || !video.videoWidth || !this._lm) return;

    let poseRes = null, handRes = null;
    try {
      poseRes = this._lm.pose.detectForVideo(video, now);
      handRes = this._lm.hand.detectForVideo(video, now);
    } catch { return; }
    const t = now / 1000;

    // pose sample (smoothed world landmarks + visibility)
    let pose = null;
    const pw = poseRes && poseRes.worldLandmarks && poseRes.worldLandmarks[0];
    const pn = poseRes && poseRes.landmarks && poseRes.landmarks[0];
    if (pw && pn){
      const world = pw.map((p, i) => this._smooth('p' + i, p, t));
      const vis = pn.map((p, i) => (p.visibility != null ? p.visibility
        : (pw[i].visibility != null ? pw[i].visibility : 1)));
      pose = { world, vis };
    }

    // hands: route detections to left/right by nearest pose wrist
    const dets = (handRes && handRes.landmarks || []).map((lms, i) => ({
      idx: i, wrist: lms[0],
      label: (handRes.handednesses && handRes.handednesses[i]
              && handRes.handednesses[i][0] || {}).categoryName || 'Right',
    }));
    const route = routeHands(dets, pn || null);
    // Hand landmarks are passed RAW: the HandShapeTracker filters the decomposed
    // JOINT ANGLES instead (position-domain smoothing lagged the curl axes
    // unevenly and skewed the hand shape — see docs/retargeting.tex §3.5).
    const handWorld = (idx) => (idx == null ? null : handRes.worldLandmarks[idx]);
    const hands = { left: handWorld(route.left), right: handWorld(route.right) };
    // Shared-image-frame wrist data for the two-hand fusion (aspect-corrected
    // in fuseWristPair); only used when BOTH hands are tracked.
    let img = null;
    if (route.left != null && route.right != null){
      const im = (idx) => ({ wrist: handRes.landmarks[idx][0],
                             idx: handRes.landmarks[idx][5],
                             pky: handRes.landmarks[idx][17] });
      img = { left: im(route.left), right: im(route.right),
              aspect: video.videoWidth / Math.max(1, video.videoHeight) };
    }

    // clutch anchor: first usable shoulders (re-captured via recalibrate())
    if (pose && !this._calib){
      this._calib = computeAnchor(pose.world, pose.vis);
      if (this._calib){
        this.link.send('recalibrate', {});
        this._toast(`anchored — shoulder width ${this._calib.shoulderM.toFixed(2)} m `
          + `(scale ×${this._calib.scale.toFixed(2)})`);
      }
    }

    // adaptive region (with ~1 s hysteresis)
    const parts = trackingParts(pose && pose.vis, hands);
    const stable = this._hyst.update(regionForParts(parts), t);
    if (stable !== this._sentRegion && pose && this._calib
        && !this.ctl.estop && parts.head){
      this._sendRegion(stable);
    }

    // synthesize + send (idle fallback after >0.5 s of loss / during e-stop)
    let sent = false;
    if (!this.ctl.estop && this.ctl.teleop && pose && this._calib){
      const payload = synthesizeFrame({ t: Date.now() / 1000, pose, hands, img },
                                      this._calib, this.seq,
                                      { trackers: this._handTrack, scaleShape: true,
                                        fuse: true, elbows: true, mirrorPalm: true });
      if (payload){
        this.link.send('teleop_frame', payload);
        this.seq++;
        this._lastGood = now;
        sent = true;
      }
    }
    if (!sent && now - this._lastGood > TRACK_LOSS_MS
        && now - this._lastIdle > 100){
      this.link.send('teleop_frame', idleFrame(this.seq++, Date.now() / 1000));
      this._lastIdle = now;
    }

    this._setChips(stable, parts, this._calib ? null
      : 'step back — shoulders not visible');
    this._draw(pn, handRes, route);
  }

  recalibrate(){
    this._calib = null;                    // recaptured on the next good frame
    this._toast('re-anchoring…');
  }

  _sendRegion(region){
    const type = this.ctl.feel === 'stiff' ? 'impedance' : 'compliance';
    this.link.controlMode('retarget', type, region, this.ctl.stiffness * 45.0);
    this._sentRegion = region;
    // keep the page's control state coherent (region segment + client-side IK)
    this.ctl.region = region;
    if (this.viewport) this.viewport.controlRegion = region;
    const seg = this.els.regionSeg;
    if (seg) seg.querySelectorAll('button').forEach((x) =>
      x.classList.toggle('on', x.dataset.v === region));
    this._toast('control region → ' + region.replace('_', ' ').toUpperCase());
  }

  _setChips(region, parts, note){
    const rc = this.els.regionChip, pc = this.els.partsChip;
    if (rc) rc.innerHTML = 'REGION: <b>'
      + (region ? region.replace('_', ' ').toUpperCase() : '—') + '</b>';
    if (!pc) return;
    if (note){ pc.textContent = note; return; }
    const mk = (on) => `<span class="${on ? 'ok' : 'no'}">${on ? '✓' : '✗'}</span>`;
    pc.innerHTML = `tracking: head ${mk(parts.head)} hands ${mk(parts.hands)} `
      + `torso ${mk(parts.torso)} legs ${mk(parts.legs)}`;
  }

  _draw(poseNorm, handRes, route){
    const video = this.els.video, canvas = this.els.canvas;
    if (!canvas || !video.videoWidth) return;
    const w = video.videoWidth, h = video.videoHeight;
    if (canvas.width !== w){ canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    // the operator must SEE the pose estimate working: bold glowing skeleton
    if (poseNorm){
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(120,255,170,.95)';
      ctx.fillStyle = 'rgba(120,255,170,1)';
      ctx.shadowColor = 'rgba(30,120,60,.9)';
      ctx.shadowBlur = 4;
      for (const [a, b] of POSE_LINKS){
        const pa = poseNorm[a], pb = poseNorm[b];
        if (!pa || !pb) continue;
        ctx.beginPath(); ctx.moveTo(pa.x * w, pa.y * h); ctx.lineTo(pb.x * w, pb.y * h); ctx.stroke();
      }
      for (const p of poseNorm){
        ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 3.2, 0, Math.PI * 2); ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
    const lms = handRes && handRes.landmarks || [];
    for (let i = 0; i < lms.length; i++){
      const side = route.left === i ? 'left' : route.right === i ? 'right' : null;
      const col = side === 'left' ? 'rgba(122,183,255,.95)'
        : side === 'right' ? 'rgba(255,166,87,.95)' : 'rgba(255,255,255,.6)';
      ctx.strokeStyle = col;
      ctx.lineWidth = 2.2;
      for (const [a, b] of HAND_LINKS){
        ctx.beginPath();
        ctx.moveTo(lms[i][a].x * w, lms[i][a].y * h);
        ctx.lineTo(lms[i][b].x * w, lms[i][b].y * h);
        ctx.stroke();
      }
      // fingertip dots + a wrist label so L/R routing is obvious at a glance
      ctx.fillStyle = col;
      for (const tip of [4, 8, 12, 16, 20]){
        ctx.beginPath();
        ctx.arc(lms[i][tip].x * w, lms[i][tip].y * h, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (side){
        ctx.font = 'bold 13px Geist Mono, monospace';
        ctx.fillText(side === 'left' ? 'L' : 'R',
          lms[i][0].x * w + 8, lms[i][0].y * h + 4);
      }
    }
  }
}
