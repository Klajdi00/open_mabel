// bodyteleop.js — webcam → TeleopFrame synthesis math (pure, node-safe).
// Verifies the MediaPipe→virtual-ARKit frame convention against the server's
// AVP_TO_ROBOT mapping (controller/mabel/config.py) and neck_angles
// (scripts/teleop/geometry.py), the clutch anchor/scale, hand skeleton wire
// format, adaptive-region hysteresis, and the NaN/idle safety guards.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  NOMINAL_SHOULDER_M, ANCHOR_HEIGHT_M, PL, HAND_JOINT_NAMES, METACARPALS,
  IDENT4, v3, mpToArkit, mat4RT, worldPoint, finiteDeep, computeAnchor,
  headTransform, wristRotation, handPose, trackingParts, regionForParts,
  RegionHysteresis, OneEuro, routeHands, synthesizeFrame, idleFrame,
  palmFrame, decomposeHand, synthesizeHand, preservePinch, refineHandShape,
  restoreDepth, estimateZScale, HandShapeTracker, fuseWristPair, reflectPalm,
  HAND_BONES, FINGER_BASE,
} from '../../web/js/bodyteleop.js';
import {
  handInOp, buildSample, corpus, OPEN_ANGLES, FIST_ANGLES, HALF, Rz,
  makeRng, DEGRADE,
// (fixture copied from simulation_control_center/experiments/ — repo rule:
// the control centers copy, they don't cross-import)
} from './retarget_corpus.mjs';

// ── server-side references, replicated for the chain checks ─────────────────
// AVP_TO_ROBOT (config.py): +X_avp→+Y_rob, +Y_avp→+Z_rob, +Z_avp→+X_rob
const toRobot = (v) => [v[2], v[0], v[1]];
// 3x3 from a row-major 4x4 wire matrix
const rotOf = (m) => [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
const posOf = (m) => [m[3], m[7], m[11]];
const matMul = (A, B) => A.map((r, i) => r.map((_, j) =>
  r.reduce((s, _v, k) => s + A[i][k] * B[k][j], 0)));
const transp = (A) => A.map((r, i) => r.map((_, j) => A[j][i]));
// neck_angles (geometry.py): dR = M (Rh Rh0ᵀ) Mᵀ, yaw = atan2(dR[1][0], dR[0][0])
const M_A2R = [[0, 0, 1], [1, 0, 0], [0, 1, 0]];
function robotYaw(Rnow, R0){
  const dR = matMul(matMul(M_A2R, matMul(Rnow, transp(R0))), transp(M_A2R));
  return Math.atan2(dR[1][0], dR[0][0]);
}

const near = (a, b, tol = 1e-6, msg) => assert.ok(Math.abs(a - b) < tol,
  msg || `${a} !~ ${b}`);
const nearV = (a, b, tol = 1e-6) => a.forEach((x, i) => near(x, b[i], tol));

// ── synthetic operator (MediaPipe world coords: x right, y down, z away) ─────
// Facing the camera. Their RIGHT side appears at image-left → mp −x.
function neutralPose({ hipVis = 0, legVis = 0 } = {}){
  const world = Array.from({ length: 33 }, () => [0, 0, 0]);
  const vis = new Array(33).fill(0);
  const set = (i, p, v = 1) => { world[i] = p; vis[i] = v; };
  const hc = [0, -0.62, -0.02];                       // head center (ear mid)
  set(PL.nose, [hc[0], hc[1], hc[2] - 0.10]);         // nose toward camera (−z)
  set(PL.leftEar, [hc[0] + 0.08, hc[1], hc[2]]);      // their left = image +x
  set(PL.rightEar, [hc[0] - 0.08, hc[1], hc[2]]);
  set(PL.leftEye, [hc[0] + 0.03, hc[1] - 0.03, hc[2] - 0.08]);
  set(PL.rightEye, [hc[0] - 0.03, hc[1] - 0.03, hc[2] - 0.08]);
  set(PL.leftShoulder, [0.20, -0.40, 0]);
  set(PL.rightShoulder, [-0.20, -0.40, 0]);
  set(PL.leftElbow, [0.26, -0.15, -0.05]);
  set(PL.rightElbow, [-0.26, -0.15, -0.05]);
  set(PL.leftWrist, [0.22, -0.05, -0.25]);            // hands in front of chest
  set(PL.rightWrist, [-0.22, -0.05, -0.25]);
  set(PL.leftHip, [0.13, 0.02, 0], hipVis);
  set(PL.rightHip, [-0.13, 0.02, 0], hipVis);
  set(PL.leftKnee, [0.13, 0.45, 0], legVis);
  set(PL.rightKnee, [-0.13, 0.45, 0], legVis);
  set(PL.leftAnkle, [0.13, 0.85, 0], legVis);
  set(PL.rightAnkle, [-0.13, 0.85, 0], legVis);
  return { world, vis };
}

// A rigid 21-point hand template (hand-centered metric, like HandLandmarker
// worldLandmarks): fingers along −y (up in image = toward the head), palm
// facing the camera. Thumb-index tip gap ≈ 9 cm (well clear of the pinch clutch).
function handTemplate(){
  const pts = [];
  pts[0] = [0, 0, 0];                                              // wrist
  pts[1] = [-0.02, -0.02, -0.01]; pts[2] = [-0.04, -0.04, -0.015];
  pts[3] = [-0.055, -0.055, -0.02]; pts[4] = [-0.07, -0.065, -0.02]; // thumb
  const fingers = [[5, -0.025], [9, 0.0], [13, 0.02], [17, 0.04]];
  for (const [base, x] of fingers){
    pts[base] = [x, -0.09, 0];
    pts[base + 1] = [x, -0.12, -0.005];
    pts[base + 2] = [x, -0.145, -0.01];
    pts[base + 3] = [x, -0.165, -0.012];
  }
  return pts.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
}

const CAL = { anchor: [0, -0.40, 0], scale: 1.0 };    // shoulder-center anchor

// ── mpToArkit: axes + chirality ──────────────────────────────────────────────
test('mpToArkit maps camera axes to the ARKit operator frame', () => {
  nearV(mpToArkit([1, 0, 0]), [-1, 0, 0]);   // image right = operator LEFT → −X
  nearV(mpToArkit([0, 1, 0]), [0, -1, 0]);   // image down → −Y (Y is up)
  nearV(mpToArkit([0, 0, 1]), [0, 0, 1]);    // away from camera = back = +Z
  // det = +1 (proper rotation → hand chirality preserved)
  const [x, y, z] = [mpToArkit([1, 0, 0]), mpToArkit([0, 1, 0]), mpToArkit([0, 0, 1])];
  const c = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  nearV(c, z);
});

test('operator-right hand motion lands on the ROBOT right (+Y_rob)', () => {
  // operator moves a hand to THEIR right = image −x → +X arkit → +Y robot
  const d = mpToArkit([-0.25, 0, 0]);
  nearV(toRobot(d), [0, 0.25, 0]);
  // toward the camera (forward, −z_mp) → −Z arkit → −X robot = robot FRONT
  nearV(toRobot(mpToArkit([0, 0, -0.3])), [-0.3, 0, 0]);
  // up (−y_mp) → +Y arkit → +Z robot
  nearV(toRobot(mpToArkit([0, -0.1, 0])), [0, 0, 0.1]);
});

// ── anchor + scale ───────────────────────────────────────────────────────────
test('computeAnchor: shoulder-center anchor + 0.38 m nominal scale', () => {
  const { world, vis } = neutralPose();
  const cal = computeAnchor(world, vis);
  assert.ok(cal);
  nearV(cal.anchor, [0, -0.40, 0]);
  near(cal.shoulderM, 0.40, 1e-9);
  near(cal.scale, NOMINAL_SHOULDER_M / 0.40, 1e-9);   // 0.95
  // shoulders not visible → no anchor
  const v2 = [...vis]; v2[PL.leftShoulder] = 0.2;
  assert.equal(computeAnchor(world, v2), null);
});

test('worldPoint: anchor-relative, scaled, lifted to standing height', () => {
  const p = worldPoint([0, -0.40, 0], [0, -0.40, 0], 2.0);
  nearV(p, [0, ANCHOR_HEIGHT_M, 0]);                  // the anchor itself
  const q = worldPoint([-0.10, -0.40, 0], [0, -0.40, 0], 2.0);
  nearV(q, [0.20, ANCHOR_HEIGHT_M, 0]);               // scaled + mirrored
});

// ── head transform ───────────────────────────────────────────────────────────
test('head facing the camera → identity rotation, plausible height', () => {
  const { world, vis } = neutralPose();
  const m = headTransform(world, vis, CAL.anchor, CAL.scale);
  assert.ok(m);
  const R = rotOf(m);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    near(R[i][j], i === j ? 1 : 0, 1e-6, `R[${i}][${j}]`);
  const p = posOf(m);
  near(p[0], 0, 1e-6);
  near(p[1], ANCHOR_HEIGHT_M + 0.22, 1e-6);           // ears 0.22 above shoulders
});

test('head turned LEFT by θ → robot yaw +θ through the server neck math', () => {
  const th = 0.35;
  const { world, vis } = neutralPose();
  const R0 = rotOf(headTransform(world, vis, CAL.anchor, CAL.scale));
  // turn left: nose → image +x (their left); ear axis rotates with it
  const hc = [0, -0.62, -0.02];
  const f = [Math.sin(th), 0, -Math.cos(th)];               // mp gaze dir
  const r = [-Math.cos(th), 0, -Math.sin(th)];              // mp operator-right dir
  world[PL.nose] = [hc[0] + 0.10 * f[0], hc[1], hc[2] + 0.10 * f[2]];
  world[PL.rightEar] = [hc[0] + 0.08 * r[0], hc[1], hc[2] + 0.08 * r[2]];
  world[PL.leftEar] = [hc[0] - 0.08 * r[0], hc[1], hc[2] - 0.08 * r[2]];
  const R1 = rotOf(headTransform(world, vis, CAL.anchor, CAL.scale));
  const yaw = robotYaw(R1, R0);
  near(yaw, th, 1e-6, 'robot yaw must equal the head turn, same sign');
});

test('head looking UP pitches the robot gaze up (gaze −Z gains +Y)', () => {
  const th = 0.30;
  const { world, vis } = neutralPose();
  const hc = [0, -0.62, -0.02];
  // look up: nose rises (−y_mp) while staying toward the camera
  world[PL.nose] = [hc[0], hc[1] - 0.10 * Math.sin(th), hc[2] - 0.10 * Math.cos(th)];
  const m = headTransform(world, vis, CAL.anchor, CAL.scale);
  const R = rotOf(m);
  // gaze = R · (0,0,−1): must gain +Y (up) by sin θ
  near(-R[1][2], Math.sin(th), 1e-6);
});

// ── hand pose wire format ────────────────────────────────────────────────────
test('handPose: anchor at the pose wrist, ARKit names, metric finger shape', () => {
  const { world } = neutralPose();
  const h = handPose(handTemplate(), 'right', world[PL.rightWrist],
                     CAL.anchor, CAL.scale);
  assert.ok(h && h.isTracked);
  assert.equal(h.chirality, 'right');
  // anchor translation = scaled anchor-relative wrist, mirrored: the mp wrist
  // (−0.22,−0.05,−0.25) rel anchor (0,−0.40,0) = (−0.22, +0.35, −0.25), i.e.
  // 0.35 m BELOW the shoulders (mp y is down) → arkit (0.22, −0.35 + H, −0.25)
  const t = posOf(h.anchorTransform.matrix);
  nearV(t, [0.22, ANCHOR_HEIGHT_M - 0.35, -0.25], 1e-9);
  // all 21 landmark names + the 4 synthesized metacarpals, all tracked
  const names = h.joints.map((j) => j.joint);
  for (const n of HAND_JOINT_NAMES) assert.ok(names.includes(n), n);
  for (const n of Object.keys(METACARPALS)) assert.ok(names.includes(n), n);
  assert.ok(h.joints.every((j) => j.isTracked));
  // world = anchor @ local: wrist world == anchor position (zero local offset)
  const wristLocal = h.joints.find((j) => j.joint === 'wrist').localTransform.matrix;
  nearV(posOf(wristLocal), [0, 0, 0]);
  // pinch gap (thumb_tip↔index_tip) survives in METERS — drives the server
  // grasp + double-pinch clutch thresholds (0.030/0.055 m)
  const J = (n) => posOf(h.joints.find((j) => j.joint === n).localTransform.matrix);
  const tt = J('thumb_tip'), it = J('index_tip');
  const tpl = handTemplate();
  const gapIn = Math.hypot(tpl[4].x - tpl[8].x, tpl[4].y - tpl[8].y, tpl[4].z - tpl[8].z);
  near(Math.hypot(tt[0] - it[0], tt[1] - it[1], tt[2] - it[2]), gapIn, 1e-9);
  assert.ok(gapIn > 0.055, 'template must not accidentally pinch-clutch');
});

test('wristRotation: orthonormal, palm frame follows the fingers', () => {
  const tpl = handTemplate();
  const w0 = v3(tpl[0]);
  const shape = tpl.map((p) => mpToArkit([p.x - w0[0], p.y - w0[1], p.z - w0[2]]));
  const R = wristRotation(shape);
  assert.ok(R);
  const [x, y, z] = R;
  near(Math.hypot(...x), 1, 1e-9); near(Math.hypot(...y), 1, 1e-9);
  near(x[0] * y[0] + x[1] * y[1] + x[2] * y[2], 0, 1e-9);
  // ŷ = fingers-forward: template fingers point up in the image → +Y arkit
  assert.ok(y[1] > 0.9);
  // right-handed: x̂ × ŷ = ẑ
  nearV([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2],
         x[0] * y[1] - x[1] * y[0]], z, 1e-9);
});

// ── adaptive region + hysteresis ─────────────────────────────────────────────
test('region from visible parts: arm → upper_body → whole_body', () => {
  const hands = { left: [], right: handTemplate() };
  let p = trackingParts(neutralPose().vis, hands);
  assert.deepEqual([p.head, p.hands, p.torso, p.legs], [true, true, false, false]);
  assert.equal(regionForParts(p), 'arm');
  p = trackingParts(neutralPose({ hipVis: 1 }).vis, hands);
  assert.equal(regionForParts(p), 'upper_body');
  p = trackingParts(neutralPose({ hipVis: 1, legVis: 1 }).vis, hands);
  assert.equal(regionForParts(p), 'whole_body');
  // no pose at all → arm (hands+head fallback)
  assert.equal(regionForParts(trackingParts(null, hands)), 'arm');
});

test('region hysteresis: ~1 s hold, flicker rejected', () => {
  const h = new RegionHysteresis(1.0);
  assert.equal(h.update('arm', 0.0), 'arm');            // first commit immediate
  assert.equal(h.update('upper_body', 0.2), 'arm');     // candidate starts
  assert.equal(h.update('upper_body', 0.9), 'arm');     // still holding
  assert.equal(h.update('arm', 1.0), 'arm');            // flicker back → reset
  assert.equal(h.update('upper_body', 1.1), 'arm');     // new candidate
  assert.equal(h.update('upper_body', 2.0), 'arm');
  assert.equal(h.update('upper_body', 2.15), 'upper_body');  // held ≥1 s → switch
  assert.equal(h.update('whole_body', 2.2), 'upper_body');
  assert.equal(h.update('whole_body', 3.3), 'whole_body');
});

// ── handedness routing ───────────────────────────────────────────────────────
test('routeHands: nearest pose wrist wins over the label', () => {
  // pose wrists in normalized image coords: person's right at image LEFT
  const poseNorm = [];
  poseNorm[PL.leftWrist] = { x: 0.7, y: 0.6 };
  poseNorm[PL.rightWrist] = { x: 0.3, y: 0.6 };
  const dets = [
    { idx: 0, wrist: { x: 0.31, y: 0.61 }, label: 'Left' },   // mislabeled
    { idx: 1, wrist: { x: 0.69, y: 0.59 }, label: 'Right' },
  ];
  assert.deepEqual(routeHands(dets, poseNorm), { left: 1, right: 0 });
  // no pose → label fallback
  assert.deepEqual(routeHands(dets, null), { left: 0, right: 1 });
  assert.deepEqual(routeHands([], poseNorm), { left: null, right: null });
});

// ── full frame synthesis + safety ────────────────────────────────────────────
test('synthesizeFrame: wire-complete payload the adapter can consume', () => {
  const pose = neutralPose({ hipVis: 1 });
  const f = synthesizeFrame({ t: 123.456, pose,
    hands: { left: null, right: handTemplate() } }, CAL, 7);
  assert.ok(f);
  assert.equal(f.sequence, 7);
  assert.equal(f.timestamp, 123.456);
  assert.equal(f.head.trackingState, 'normal');
  assert.equal(f.head.transform.matrix.length, 16);
  assert.equal(f.leftHand, null);
  assert.ok(f.rightHand.isTracked);
  assert.equal(f.mode, 'Arms & Hands');
  assert.deepEqual(f.navJoystick, { lx: 0, ly: 0, rx: 0, ry: 0 });
  assert.ok(finiteDeep(f));
  // head non-identity → claims the driver token (_head_active on the bridge)
  const m = f.head.transform.matrix;
  assert.ok(Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]) > 0.02);
});

test('NaN anywhere → frame refused (never on the wire)', () => {
  const pose = neutralPose();
  pose.world[PL.nose][2] = NaN;
  assert.equal(synthesizeFrame({ t: 1, pose, hands: {} }, CAL, 0), null);
  const ok = neutralPose();
  const badHand = handTemplate(); badHand[8].y = Infinity;
  assert.equal(synthesizeFrame({ t: 1, pose: ok,
    hands: { left: null, right: badHand } }, CAL, 0), null);
  // missing calib / pose → null, not garbage
  assert.equal(synthesizeFrame({ t: 1, pose: ok, hands: {} }, null, 0), null);
  assert.equal(synthesizeFrame(null, CAL, 0), null);
});

test('idleFrame: identity notAvailable head, untracked hands, zero nav', () => {
  const f = idleFrame(3, 9.0);
  assert.deepEqual(f.head.transform.matrix, IDENT4);
  assert.equal(f.head.trackingState, 'notAvailable');   // server: non-claiming
  assert.equal(f.leftHand.isTracked, false);
  assert.equal(f.rightHand.isTracked, false);
  assert.deepEqual(f.navJoystick, { lx: 0, ly: 0, rx: 0, ry: 0 });
  assert.ok(finiteDeep(f));
});

test('one-euro: converges, resets after a gap, kills jitter', () => {
  const f = new OneEuro();
  let y = 0;
  for (let i = 0; i < 200; i++) y = f.filter(1.0, i * 0.033);
  near(y, 1.0, 1e-3);
  // 0.5 s gap → hard reset to the new value (no cross-loss smoothing)
  near(f.filter(5.0, 200 * 0.033 + 1.0), 5.0, 1e-12);
  // high-frequency ±1 mm jitter is attenuated
  let lo = Infinity, hi = -Infinity, t0 = 300;
  for (let i = 0; i < 100; i++){
    const v = f.filter(5.0 + (i % 2 ? 1e-3 : -1e-3), t0 + i * 0.033);
    if (i > 20){ lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  assert.ok(hi - lo < 1.2e-3, `jitter band ${hi - lo}`);
});

test('mat4RT: row-major, translation at 3/7/11 (Transform4x4.position_xyz)', () => {
  const m = mat4RT(null, [1, 2, 3]);
  assert.deepEqual([m[3], m[7], m[11]], [1, 2, 3]);
  assert.deepEqual([m[12], m[13], m[14], m[15]], [0, 0, 0, 1]);
  // columns land in the right cells: X̂ down column 0
  const R = [[0, 1, 0], [0, 0, 1], [1, 0, 0]];   // x̂=(0,1,0)ᵀ etc. (columns)
  const mm = mat4RT(R, [0, 0, 0]);
  assert.deepEqual([mm[0], mm[4], mm[8]], [0, 1, 0]);   // first column = x̂
});

// ═══ anatomical hand decomposition / re-synthesis (docs/retargeting.tex §3) ══
// Ground truth comes from the INDEPENDENT corpus FK (experiments/
// retarget_corpus.mjs) — a different parameterization, so the round trips
// below are genuine inversions, not identities.
const dsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dnorm = (a) => Math.hypot(a[0], a[1], a[2]);

test('palmFrame: orthonormal, ẑ = dorsal for BOTH chiralities', () => {
  for (const side of ['right', 'left']){
    const sh = handInOp(OPEN_ANGLES(), side);      // palm to camera, fingers up
    const B = palmFrame(sh, side);
    assert.ok(B);
    const [x, y, z] = B;
    near(dnorm(x), 1, 1e-9); near(dnorm(y), 1, 1e-9); near(dnorm(z), 1, 1e-9);
    near(x[0] * y[0] + x[1] * y[1] + x[2] * y[2], 0, 1e-9);
    // palm faces the camera (−Z) → dorsal normal ẑ points BACK (+Z), same
    // formula for both hands (χ handles the mirrored knuckle order).
    assert.ok(z[2] > 0.95, `${side} dorsal ẑ ${z}`);
    assert.ok(y[1] > 0.95, `${side} fingers ŷ ${y}`);
  }
});

test('decomposeHand recovers the corpus FK angles exactly (open/fist/half)', () => {
  const half = () => { const a = OPEN_ANGLES(); a.fingers.middle = { ...HALF }; return a; };
  for (const mk of [OPEN_ANGLES, FIST_ANGLES, half])
    for (const side of ['right', 'left']){
      const truth = mk();
      const dec = decomposeHand(handInOp(truth, side), side);
      assert.ok(dec, `${side} decompose`);
      for (const f of ['index', 'middle', 'ring', 'little'])
        for (const k of ['abd', 'mcp', 'pip', 'dip'])
          near(dec.fingers[f][k], truth.fingers[f][k], 1e-6, `${side} ${f} ${k}`);
      near(dec.thumb.mp, truth.thumb.mp, 1e-6, `${side} thumb mp`);
      near(dec.thumb.ip, truth.thumb.ip, 1e-6, `${side} thumb ip`);
    }
});

test('decompose is invariant to a rigid hand rotation (angles, not poses)', () => {
  const truth = FIST_ANGLES();
  const dec = decomposeHand(handInOp(truth, 'right', Rz(0.7)), 'right');
  near(dec.fingers.index.pip, truth.fingers.index.pip, 1e-6);
  near(dec.fingers.index.mcp, truth.fingers.index.mcp, 1e-6);
});

test('synthesizeHand is the exact inverse of decomposeHand (round trip)', () => {
  const sh = handInOp(FIST_ANGLES(), 'right');
  const dec = decomposeHand(sh, 'right');
  const rebuilt = synthesizeHand(dec);
  // same 21 points (up to the wrist-centering the synthesis guarantees)
  for (let i = 0; i < 21; i++)
    near(dnorm(dsub(rebuilt[i], dsub(sh[i], sh[0]))), 0, 1e-9, `landmark ${i}`);
  // and decomposing the rebuild returns the same angles (idempotent)
  const dec2 = decomposeHand(rebuilt, 'right');
  for (const f of ['index', 'middle', 'ring', 'little'])
    for (const k of ['abd', 'mcp', 'pip', 'dip'])
      near(dec2.fingers[f][k], dec.fingers[f][k], 1e-9);
});

test('preservePinch: a true pinch survives re-synthesis in meters', () => {
  // raw pinch with tips 6 mm apart; a re-synthesis that opened it to 25 mm
  // must be pulled back near the measured gap (grasp curve + clutch integrity)
  const raw = handInOp(OPEN_ANGLES(), 'right');
  raw[4] = [raw[8][0] + 0.006, raw[8][1], raw[8][2]];
  const refined = raw.slice();
  refined[4] = [raw[8][0] + 0.025, raw[8][1], raw[8][2]];
  const out = preservePinch(refined, raw);
  const gap = dnorm(dsub(out[4], out[8]));
  assert.ok(gap < 0.010, `gap ${gap}`);
  // far apart → untouched
  const open = handInOp(OPEN_ANGLES(), 'right');
  assert.deepEqual(preservePinch(open.slice(), open), open);
});

test('restoreDepth + estimateZScale recover a depth-attenuated fist', () => {
  const sh = handInOp(FIST_ANGLES(), 'right');
  const lens = HAND_BONES.map(([a, b]) => dnorm(dsub(sh[b], sh[a])));
  const att = sh.map((p) => [p[0], p[1], p[2] * 0.65]);      // MediaPipe-style
  // the global attenuation coefficient is identified from z-dominant bones
  const est = estimateZScale(att, lens);
  assert.ok(est && Math.abs(est.kappa - 0.65) < 0.05, `kappa ${est && est.kappa}`);
  // full pipeline: κ-descale + per-bone restoration → angles close to truth
  const fixed = restoreDepth(att.map((p) => [p[0], p[1], p[2] / est.kappa]), lens);
  const dec = decomposeHand(fixed, 'right');
  const truth = FIST_ANGLES();
  for (const f of ['index', 'middle', 'ring', 'little']){
    assert.ok(Math.abs(dec.fingers[f].mcp - truth.fingers[f].mcp) < 0.06, `${f} mcp`);
    assert.ok(Math.abs(dec.fingers[f].pip - truth.fingers[f].pip) < 0.10, `${f} pip`);
  }
  // versus the UNCORRECTED attenuated shape, which under-curls badly
  const decRaw = decomposeHand(att, 'right');
  assert.ok(truth.fingers.index.mcp - decRaw.fingers.index.mcp > 0.15,
    'attenuation must visibly under-curl without the correction');
});

test('HandShapeTracker: filters angles, calibrates lengths, converges on truth', () => {
  const tr = new HandShapeTracker('right');
  const rng = makeRng(7);
  const truth = FIST_ANGLES();
  const clean = handInOp(truth, 'right');
  let out = null;
  for (let k = 0; k < 60; k++){
    const noisy = clean.map((p) => [p[0] + 0.0025 * rng.g(),
                                    p[1] + 0.0025 * rng.g(),
                                    p[2] * DEGRADE.zAtten + 0.0025 * rng.g()]);
    out = tr.update(noisy, k / 30);
  }
  const dec = decomposeHand(out, 'right');
  // pitch recovered well inside the raw attenuation error (~0.3 rad)
  assert.ok(Math.abs(dec.fingers.index.mcp - truth.fingers.index.mcp) < 0.17,
    `tracked mcp ${dec.fingers.index.mcp} vs ${truth.fingers.index.mcp}`);
  // κ is identified conservatively (the anthropometric length floor biases it
  // toward 1, the per-bone restoration picks up the residual — see §4.3): it
  // must clearly DETECT the attenuation, the angle bound above is the measure.
  assert.ok(tr.kappa !== null && tr.kappa < 0.92
            && Math.abs(tr.kappa - DEGRADE.zAtten) < 0.25, `kappa ${tr.kappa}`);
  // degenerate input passes through unrefined (no NaNs, no throw)
  const flat = Array.from({ length: 21 }, () => [0, 0, 0]);
  assert.deepEqual(tr.update(flat, 3.0), flat);
});

test('fuseWristPair: image frame fixes the inter-wrist vector, midpoint kept', () => {
  const s = buildSample('touch', {
    hands: {
      left: { angles: OPEN_ANGLES(), R: Rz(-1.35), wristOp: [-0.14, -0.05, -0.38],
              elbowOp: [-0.28, -0.28, -0.16] },
      right: { angles: OPEN_ANGLES(), R: Rz(+1.35), wristOp: [0.14, -0.05, -0.38],
               elbowOp: [0.28, -0.28, -0.16] },
    }, touch: [], desc: 't',
  });
  const spanOf = (h) => dnorm(dsub(v3(h[5]), v3(h[17])));
  const pwL = v3(s.pose.world[PL.leftWrist]);
  // inject a 5 cm pose error on the LEFT wrist only (image stays truthful)
  const pwLbad = [pwL[0] + 0.05, pwL[1], pwL[2]];
  const pwR = v3(s.pose.world[PL.rightWrist]);
  const fused = fuseWristPair(pwLbad, pwR, s.img,
                              spanOf(s.hands.left), spanOf(s.hands.right));
  assert.ok(fused);
  const [fL, fR] = fused;
  // midpoint preserved…
  nearV([(fL[0] + fR[0]) / 2, (fL[1] + fR[1]) / 2],
        [(pwLbad[0] + pwR[0]) / 2, (pwLbad[1] + pwR[1]) / 2], 1e-9);
  // …and the x,y inter-wrist vector snapped back to the true geometry (the
  // residual is the perspective error of the span-scale, ≪ the injected 5 cm)
  const dTrue = dsub(pwR, pwL), dFused = dsub(fR, fL);
  assert.ok(Math.abs(dFused[0] - dTrue[0]) < 0.012, `dx ${dFused[0]} vs ${dTrue[0]}`);
  // depth (z) untouched: image cannot see it
  near(dFused[2], pwR[2] - pwLbad[2], 1e-9);
  // far-apart hands: fusion declines (proximity gate)
  const far = { ...s.img };
  far.left = { ...far.left, wrist: { x: 0.05, y: 0.5 } };
  far.right = { ...far.right, wrist: { x: 0.95, y: 0.5 } };
  assert.equal(fuseWristPair(pwLbad, pwR, far,
                             spanOf(s.hands.left), spanOf(s.hands.right)), null);
});

test('reflectPalm: chirality flip, middle tip PINNED, distances preserved', () => {
  const sh = handInOp(FIST_ANGLES(), 'right');
  const r = reflectPalm(sh, 'right');
  // middle fingertip (the server arm's contact point) must not move
  nearV(r[12], sh[12], 1e-12);
  // reflection preserves ALL pairwise distances (isometry): pinch integrity
  for (const [a, b] of [[4, 8], [0, 8], [5, 17]])
    near(dnorm(dsub(r[a], r[b])), dnorm(dsub(sh[a], sh[b])), 1e-9, `${a}-${b}`);
  // it IS a chirality flip: the reflected right hand decomposes as a LEFT
  // hand with the ORIGINAL flexion angles (this is what the sim's finger
  // solver convention expects — see reflectPalm docstring)
  const dec = decomposeHand(r, 'left');
  const truth = FIST_ANGLES();
  for (const f of ['index', 'middle', 'ring', 'little'])
    for (const k of ['mcp', 'pip', 'dip'])
      near(dec.fingers[f][k], truth.fingers[f][k], 1e-6, `${f} ${k}`);
  // …whereas read as a RIGHT hand it curls the WRONG way (the pre-fix bug)
  const wrong = decomposeHand(r, 'right');
  assert.ok(wrong.fingers.index.mcp < 0.0, `mirror flips flexion ${wrong.fingers.index.mcp}`);
  // involution up to the pin translation: reflect twice ≈ original
  const rr = reflectPalm(r, 'right');
  for (let i = 0; i < 21; i++)
    near(dnorm(dsub(dsub(rr[i], rr[0]), dsub(sh[i], sh[0]))), 0, 1e-9);
});

test('two-hand consistency: touching fingertips STAY touching on the wire', () => {
  // The heart shape: index tips coincide in the operator's real space. With
  // the refined options (ONE similarity transform: scaled shape + fused
  // wrists) the WIRE-frame tips must coincide too — this is what makes the
  // robot's fingertips meet (server maps both through one world transform).
  const C = corpus();
  const s = C.heart.sample(0);
  const calib = computeAnchor(s.pose.world, s.pose.vis);
  near(calib.scale, NOMINAL_SHOULDER_M / 0.42, 1e-9);   // deliberately ≠ 1
  const trackers = { left: new HandShapeTracker('left'),
                     right: new HandShapeTracker('right') };
  const f = synthesizeFrame({ t: 0, pose: s.pose, hands: s.hands, img: s.img },
                            calib, 0, { trackers, scaleShape: true, fuse: true,
                                        elbows: true, mirrorPalm: true });
  assert.ok(f);
  const world = (hand, name) => {
    const j = hand.joints.find((x) => x.joint === name);
    const a = hand.anchorTransform.matrix, l = j.localTransform.matrix;
    return [a[3] + l[3], a[7] + l[7], a[11] + l[11]];   // identity anchor rot
  };
  const gap = dnorm(dsub(world(f.leftHand, 'index_tip'),
                         world(f.rightHand, 'index_tip')));
  assert.ok(gap < 0.004, `refined wire gap ${gap}`);
  // legacy pipeline on the SAME sample: the unscaled shape vs scaled wrist
  // mixing leaves a cm-scale gap — the bug this upgrade removes.
  const g = synthesizeFrame({ t: 0, pose: s.pose, hands: s.hands }, calib, 0);
  const gapLegacy = dnorm(dsub(world(g.leftHand, 'index_tip'),
                               world(g.rightHand, 'index_tip')));
  assert.ok(gapLegacy > 0.015, `legacy wire gap ${gapLegacy}`);
  // elbows attached for the server's arm-direction objective
  assert.ok(f.leftElbow && f.rightElbow);
  assert.equal(f.leftElbow.matrix.length, 16);
  assert.ok(finiteDeep(f));
});
