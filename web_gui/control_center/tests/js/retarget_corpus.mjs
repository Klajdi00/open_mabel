// retarget_corpus.mjs — scripted landmark corpus for the webcam-retargeting
// validation harness (docs/retargeting.tex §6).
//
// Generates synthetic MediaPipe outputs (PoseLandmarker 33 world landmarks +
// HandLandmarker 21 worldLandmarks per hand + normalized image landmarks) for
// canonical operator shapes, WITH ground truth: the exact per-joint angles each
// hand was posed with and the exact fingertip positions in the operator frame.
//
// The hand forward kinematics here is an INDEPENDENT implementation from
// web/js/bodyteleop.js (different parameterization: thumb via spherical
// azimuth/elevation, fingers via explicit axis-angle chains over the geometry
// constants below), so round-tripping through decomposeHand is a real test.
//
// Conventions:
//   * "op space"  = the virtual ARKit operator frame (X = operator's RIGHT,
//     Y = up, Z = back; −Z faces the camera). Anchor = shoulder-center.
//   * MediaPipe camera coords = (x,y negated) op space:  p_mp = (−X, −Y, Z).
//   * Hands are generated in a canonical RIGHT-hand palm frame (x̂ = thumb
//     side, ŷ = fingers, ẑ = dorsal) and mirrored across x̂ for the left hand.
//     A neutral (R = I) hand shows its palm to the camera, fingers up.

// ── tiny vec helpers (self-contained) ────────────────────────────────────────
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => scl(a, 1 / (norm(a) || 1));
const rot = (v, a, th) => {                       // Rodrigues about unit axis
  const c = Math.cos(th), s = Math.sin(th);
  return add(add(scl(v, c), scl(cross(a, v), s)), scl(a, dot(a, v) * (1 - c)));
};
const matv = (M, v) => [dot(M[0], v), dot(M[1], v), dot(M[2], v)];
const matmul = (A, B) => A.map((_, i) => [0, 1, 2].map((j) =>
  A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));
export const Rx = (t) => [[1, 0, 0], [0, Math.cos(t), -Math.sin(t)], [0, Math.sin(t), Math.cos(t)]];
export const Ry = (t) => [[Math.cos(t), 0, Math.sin(t)], [0, 1, 0], [-Math.sin(t), 0, Math.cos(t)]];
export const Rz = (t) => [[Math.cos(t), -Math.sin(t), 0], [Math.sin(t), Math.cos(t), 0], [0, 0, 1]];
const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// op space (ARKit) ↔ MediaPipe camera coords (involution)
export const opToMp = (p) => [-p[0], -p[1], p[2]];

// ── canonical adult hand geometry (right hand, palm frame, meters) ──────────
export const HAND_GEOM = {
  knuckles: { index: [0.030, 0.088, 0.000], middle: [0.010, 0.094, 0.000],
              ring: [-0.012, 0.089, 0.000], little: [-0.034, 0.081, 0.000] },
  fingerL: { index: [0.040, 0.024, 0.020], middle: [0.044, 0.028, 0.021],
             ring: [0.041, 0.026, 0.020], little: [0.032, 0.020, 0.017] },
  thumbCmc: [0.030, 0.022, -0.006],
  thumbL: [0.046, 0.032, 0.028],          // metacarpal, proximal, distal
};
const FINGER_BASE = { index: 5, middle: 9, ring: 13, little: 17 };

// Neutral (slightly relaxed) pose and a full fist. Thumb parameterization:
// az rotates the metacarpal direction in the palm plane from +x̂ toward +ŷ;
// el dives it below the palm plane toward −ẑ (opposition); mp/ip flex.
export const OPEN_ANGLES = () => ({
  fingers: Object.fromEntries(['index', 'middle', 'ring', 'little'].map((f) =>
    [f, { abd: 0.00, mcp: 0.06, pip: 0.05, dip: 0.03 }])),
  thumb: { az: 0.55, el: 0.15, mp: 0.10, ip: 0.10 },
});
export const FIST_ANGLES = () => ({
  fingers: Object.fromEntries(['index', 'middle', 'ring', 'little'].map((f) =>
    [f, { abd: 0.0, mcp: 1.15, pip: 1.55, dip: 0.9 }])),
  thumb: { az: 0.9, el: 0.55, mp: 0.85, ip: 0.85 },
});
export const HALF = { abd: 0.0, mcp: 0.45, pip: 0.60, dip: 0.30 };

// ── independent hand FK: angles → 21 palm-frame points (right hand) ─────────
export function handFK(angles){
  const G = HAND_GEOM;
  const zHat = [0, 0, 1];
  const p = new Array(21);
  p[0] = [0, 0, 0];
  for (const [f, base] of Object.entries(FINGER_BASE)){
    const A = angles.fingers[f];
    const kn = G.knuckles[f];
    const L = G.fingerL[f];
    const mRef = unit([kn[0], kn[1], 0]);          // palm-plane knuckle ray
    const r = rot(mRef, zHat, A.abd);              // abducted ray (palm plane)
    const ax = unit(cross(zHat, r));               // flexion axis
    const u1 = rot(r, ax, A.mcp);                  // flexion curls toward −ẑ
    const u2 = rot(u1, ax, A.pip);
    const u3 = rot(u2, ax, A.dip);
    p[base] = kn;
    p[base + 1] = add(kn, scl(u1, L[0]));
    p[base + 2] = add(p[base + 1], scl(u2, L[1]));
    p[base + 3] = add(p[base + 2], scl(u3, L[2]));
  }
  const T = angles.thumb;
  const dPl = rot([1, 0, 0], zHat, T.az);          // palm-plane component
  const dir = rot(dPl, unit(cross(zHat, dPl)), T.el);   // dive toward −ẑ
  let axT = cross(dir, [0, 0, -1]);                // flexion axis ⊥ dir
  if (norm(axT) < 1e-6) axT = [0, 1, 0];
  axT = unit(axT);
  const c = HAND_GEOM.thumbCmc, L = HAND_GEOM.thumbL;
  const uMp = rot(dir, axT, T.mp);
  const uIp = rot(uMp, axT, T.ip);
  p[1] = c;
  p[2] = add(c, scl(dir, L[0]));
  p[3] = add(p[2], scl(uMp, L[1]));
  p[4] = add(p[3], scl(uIp, L[2]));
  return p;
}

// Palm-frame → op space for one hand: mirror across x̂ for the left, orient by
// R (op-space rotation), hand-centered. Returns the 21 op-space offsets.
export function handInOp(angles, chirality, R = I3){
  const pts = handFK(angles);
  const mir = chirality === 'left' ? pts.map((q) => [-q[0], q[1], q[2]]) : pts;
  return mir.map((q) => matv(R, q));
}

// ── operator body model (mp coords; x = image right, y = down, z = away) ────
export const PL = {
  nose: 0, leftEye: 2, rightEye: 5, leftEar: 7, rightEar: 8,
  leftShoulder: 11, rightShoulder: 12, leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16, leftHip: 23, rightHip: 24,
  leftKnee: 25, rightKnee: 26, leftAnkle: 27, rightAnkle: 28,
};
export const SHOULDER_HALF = 0.21;      // half biacromial width → scaleF ≠ 1
const ANCHOR_MP = [0, -0.40, 0];        // shoulder-center in mp coords

// Facing the camera. The operator's LEFT side sits at image +x (mp +x).
export function baseBody(){
  const world = Array.from({ length: 33 }, () => [0, 0, 0]);
  const vis = new Array(33).fill(0);
  const set = (i, p, v = 1) => { world[i] = p; vis[i] = v; };
  const hc = [0, -0.62, -0.02];
  set(PL.nose, [hc[0], hc[1], hc[2] - 0.10]);
  set(PL.leftEar, [hc[0] + 0.08, hc[1], hc[2]]);
  set(PL.rightEar, [hc[0] - 0.08, hc[1], hc[2]]);
  set(PL.leftEye, [hc[0] + 0.03, hc[1] - 0.03, hc[2] - 0.08]);
  set(PL.rightEye, [hc[0] - 0.03, hc[1] - 0.03, hc[2] - 0.08]);
  set(PL.leftShoulder, [SHOULDER_HALF, -0.40, 0]);
  set(PL.rightShoulder, [-SHOULDER_HALF, -0.40, 0]);
  set(PL.leftHip, [0.13, 0.02, 0]);
  set(PL.rightHip, [-0.13, 0.02, 0]);
  return { world, vis };
}

// Place one arm: wrist + elbow given in OP space (anchor-relative); stored in
// mp coords. NOTE op X = operator's right, so the LEFT hand has X < 0.
function placeArm(body, side, wristOp, elbowOp){
  const wi = side === 'left' ? PL.leftWrist : PL.rightWrist;
  const ei = side === 'left' ? PL.leftElbow : PL.rightElbow;
  body.world[wi] = add(ANCHOR_MP, opToMp(wristOp));
  body.vis[wi] = 1;
  body.world[ei] = add(ANCHOR_MP, opToMp(elbowOp));
  body.vis[ei] = 1;
}

// ── pinhole projection for the normalized image landmarks ────────────────────
export const CAM = { Z0: 1.5, wf: 2 * Math.tan((60 / 2) * Math.PI / 180), aspect: 4 / 3 };
export function project(mpPoint){
  const Z = CAM.Z0 + mpPoint[2];
  return { x: 0.5 + mpPoint[0] / (Z * CAM.wf),
           y: 0.5 + mpPoint[1] / (Z * CAM.wf / CAM.aspect) };
}

// ── seeded RNG (LCG + Box–Muller) for reproducible degradation ───────────────
export function makeRng(seed = 12345){
  let s = seed >>> 0;
  const u = () => ((s = (1664525 * s + 1013904223) >>> 0) / 4294967296);
  const g = () => {
    const a = Math.max(u(), 1e-12), b = u();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  };
  return { u, g };
}

// MediaPipe-like degradation constants (documented in retargeting.tex §6.2):
// hand worldLandmarks: depth compressed ×0.65 + σ=2.5 mm noise;
// pose wrists/elbows: fixed per-side bias (~2–3 cm, models the inter-hand
// inconsistency of BlazePose world coords) + σ=6/10 mm noise;
// image landmarks: σ=0.002 normalized noise.
export const DEGRADE = { zAtten: 0.65, handNoise: 0.0025, imgNoise: 0.002,
  poseBias: { left: [0.022, -0.012, 0.018], right: [-0.018, 0.010, -0.015] },
  poseNoiseXY: 0.006, poseNoiseZ: 0.010 };

// ── shape catalog ────────────────────────────────────────────────────────────
function shapes(){
  const open = OPEN_ANGLES;
  const S = {};
  const both = (angles, R, wl, wr, el, er) => ({
    left: { angles: angles(), R, wristOp: wl, elbowOp: el },
    right: { angles: angles(), R, wristOp: wr, elbowOp: er },
  });
  // wrists in OP space: left hand X<0, right hand X>0
  const CHEST_L = [-0.22, -0.10, -0.32], CHEST_R = [0.22, -0.10, -0.32];
  const ELB_L = [-0.26, -0.30, -0.10], ELB_R = [0.26, -0.30, -0.10];

  S.open_palm = {
    hands: both(open, I3, CHEST_L, CHEST_R, ELB_L, ELB_R),
    touch: [], desc: 'both hands open, palms to camera, chest height',
  };
  S.fist = {
    hands: both(FIST_ANGLES, I3, CHEST_L, CHEST_R, ELB_L, ELB_R),
    touch: [], desc: 'both hands full fist',
  };
  for (const f of ['index', 'middle', 'ring', 'little']){
    const a = open();
    a.fingers[f] = { ...HALF };
    S[`half_${f}`] = {
      hands: {
        left: { angles: open(), R: I3, wristOp: CHEST_L, elbowOp: ELB_L },
        right: { angles: a, R: I3, wristOp: CHEST_R, elbowOp: ELB_R },
      },
      touch: [], desc: `right ${f} half-curled, rest open`,
    };
  }
  // thumb–index pinch (right hand). Angles grid-searched over the generator FK
  // so the generated tips TRULY touch: gap = 4.9 mm.
  const pinch = open();
  pinch.fingers.index = { abd: 0.20, mcp: 1.00, pip: 1.15, dip: 0.10 };
  pinch.thumb = { az: 1.50, el: 0.20, mp: 0.75, ip: 0.70 };
  S.pinch = {
    hands: {
      left: { angles: open(), R: I3, wristOp: CHEST_L, elbowOp: ELB_L },
      right: { angles: pinch, R: I3, wristOp: CHEST_R, elbowOp: ELB_R },
    },
    touch: [['right', 4, 'right', 8]],
    desc: 'right thumb–index pinch (intra-hand touch), left open',
  };
  // two-hand fingertip HEART: hands lean toward the midline; the index tips
  // MEET exactly (wrists solved from the constraint); thumb-tip pair reported.
  // Lean (±1.4 rad) + thumb angles grid-searched so, with the index tips
  // meeting exactly, the thumb-tip pair gap is only 20 mm (a closed heart).
  const heartA = () => {
    const a = open();
    a.fingers.index = { abd: 0.10, mcp: 0.55, pip: 0.55, dip: 0.35 };
    a.fingers.middle = { abd: 0.0, mcp: 0.75, pip: 0.75, dip: 0.45 };
    a.fingers.ring = { abd: 0.0, mcp: 0.80, pip: 0.80, dip: 0.45 };
    a.fingers.little = { abd: 0.0, mcp: 0.85, pip: 0.85, dip: 0.45 };
    a.thumb = { az: 1.40, el: 0.10, mp: 0.10, ip: 0.00 };
    return a;
  };
  S.heart = {
    hands: {
      left: { angles: heartA(), R: Rz(-1.4), wristOp: null,
              elbowOp: [-0.30, -0.25, -0.18] },
      right: { angles: heartA(), R: Rz(+1.4), wristOp: null,
               elbowOp: [0.30, -0.25, -0.18] },
    },
    meet: { pair: [8, 8], at: [0.0, 0.12, -0.34] },
    touch: [['left', 8, 'right', 8], ['left', 4, 'right', 4]],
    desc: 'two-hand fingertip heart: index tips touch, thumb tips near',
  };
  // hands forward, index fingertips touching at the midline
  const point = () => {
    const a = open();
    for (const f of ['middle', 'ring', 'little'])
      a.fingers[f] = { abd: 0, mcp: 1.0, pip: 1.3, dip: 0.7 };
    a.thumb = { az: 0.8, el: 0.45, mp: 0.55, ip: 0.45 };
    return a;
  };
  S.touch_forward = {
    hands: {
      left: { angles: point(), R: Rz(-1.35), wristOp: null,
              elbowOp: [-0.28, -0.28, -0.16] },
      right: { angles: point(), R: Rz(+1.35), wristOp: null,
               elbowOp: [0.28, -0.28, -0.16] },
    },
    meet: { pair: [8, 8], at: [0.0, -0.05, -0.38] },
    touch: [['left', 8, 'right', 8]],
    desc: 'hands forward, index fingertips touching at the midline',
  };
  // salute: right hand flat at the forehead (palm down-forward); left arm down
  S.salute = {
    hands: {
      left: { angles: open(), R: I3, wristOp: [-0.24, -0.35, -0.10],
              elbowOp: [-0.27, -0.55, -0.05] },
      right: { angles: open(), R: matmul(Rz(0.5), Rx(-1.2)),
               wristOp: [0.10, 0.22, -0.12], elbowOp: [0.32, 0.02, -0.16] },
    },
    touch: [], desc: 'right-hand salute at the forehead, left arm down',
  };
  S.arms_up = {
    hands: both(open, I3, [-0.18, 0.45, -0.06], [0.18, 0.45, -0.06],
                [-0.24, 0.15, -0.04], [0.24, 0.15, -0.04]),
    touch: [], desc: 'both arms raised overhead, palms to camera',
  };
  S.hands_on_hips = {
    hands: both(open, Rx(-0.9), [-0.15, -0.45, -0.10], [0.15, -0.45, -0.10],
                [-0.34, -0.30, -0.08], [0.34, -0.30, -0.08]),
    touch: [], desc: 'hands on hips, elbows out',
  };
  return S;
}

// ── build one corpus sample (ideal or degraded) ──────────────────────────────
export function buildSample(name, spec, { degrade = false, rng = null, t = 0 } = {}){
  const body = baseBody();
  const hands = {}, img = { aspect: CAM.aspect };
  const truth = { angles: {}, tips: {}, wristOp: {}, touch: [] };
  const opPts = {};

  const built = {};
  for (const side of ['left', 'right'])
    built[side] = handInOp(spec.hands[side].angles, side, spec.hands[side].R);
  let wrists = { left: spec.hands.left.wristOp, right: spec.hands.right.wristOp };
  if (spec.meet){
    const [iL, iR] = spec.meet.pair;
    wrists = { left: sub(spec.meet.at, built.left[iL]),
               right: sub(spec.meet.at, built.right[iR]) };
  }

  for (const side of ['left', 'right']){
    const sh = built[side];
    const w = wrists[side];
    truth.angles[side] = spec.hands[side].angles;
    truth.wristOp[side] = w;
    opPts[side] = sh.map((q) => add(w, q));
    truth.tips[side] = { 4: opPts[side][4], 8: opPts[side][8], 12: opPts[side][12] };
    placeArm(body, side, w, spec.hands[side].elbowOp);
    let world21 = sh.map((q) => opToMp(q));         // hand-centered, mp coords
    if (degrade && rng)
      world21 = world21.map((q) => [q[0] + DEGRADE.handNoise * rng.g(),
                                    q[1] + DEGRADE.handNoise * rng.g(),
                                    q[2] * DEGRADE.zAtten + DEGRADE.handNoise * rng.g()]);
    hands[side] = world21.map((q) => ({ x: q[0], y: q[1], z: q[2] }));
    const mpAbs = opPts[side].map((q) => add(ANCHOR_MP, opToMp(q)));
    const im = (i) => {
      const p = project(mpAbs[i]);
      return (degrade && rng)
        ? { x: p.x + DEGRADE.imgNoise * rng.g(), y: p.y + DEGRADE.imgNoise * rng.g() }
        : p;
    };
    img[side] = { wrist: im(0), idx: im(5), pky: im(17) };
  }

  for (const [sa, ia, sb, ib] of spec.touch || [])
    truth.touch.push({ pair: [sa, ia, sb, ib],
                       gap: norm(sub(opPts[sa][ia], opPts[sb][ib])) });

  if (degrade && rng){
    for (const side of ['left', 'right']){
      const wi = side === 'left' ? PL.leftWrist : PL.rightWrist;
      const ei = side === 'left' ? PL.leftElbow : PL.rightElbow;
      for (const i of [wi, ei])
        body.world[i] = add(add(body.world[i], DEGRADE.poseBias[side]),
                            [DEGRADE.poseNoiseXY * rng.g(),
                             DEGRADE.poseNoiseXY * rng.g(),
                             DEGRADE.poseNoiseZ * rng.g()]);
    }
  }

  return {
    name, desc: spec.desc, t,
    pose: { world: body.world, vis: body.vis },
    hands, img,
    truth: { ...truth, anchorMp: ANCHOR_MP, shoulderM: 2 * SHOULDER_HALF },
  };
}

export function corpus({ degrade = false, seed = 20260702 } = {}){
  const rng = degrade ? makeRng(seed) : null;
  const out = {};
  for (const [name, spec] of Object.entries(shapes()))
    out[name] = { spec, sample: (t = 0) => buildSample(name, spec, { degrade, rng, t }) };
  return out;
}

export const SHAPE_NAMES = Object.keys(shapes());
