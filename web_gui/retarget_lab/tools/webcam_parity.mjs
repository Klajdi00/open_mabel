/* Does the website's webcam demo run the retargeter NOVA runs?
 *
 * A synthetic operator — MediaPipe-shaped world landmarks, the same shapes
 * camtest.py feeds the page — is pushed through the demo's own front end
 * (website/assets/bodyteleop-core.js: anchor, hand pose, elbows) into `teleop_frame`
 * payloads, exactly what the page builds from a camera. The SAME frames then go
 * to two retargeters:
 *
 *   Python   ../NOVA/controller/lab_retarget.py LabRetargeter, offline
 *            (tools/webcam_parity_ref.py) — what NOVA steps on the robot
 *   JS       website/assets/deployed-retarget.js — what the page runs
 *
 * and every commanded joint is compared frame by frame. Two runs:
 *
 *   port     self-collision OFF on both sides: the map + arm IK alone
 *   deploy   as NOVA deploys it: the self-collision projector ON (its capsule
 *            barrier QP is active on about half the frames — a hanging upper
 *            arm sits inside the 10 cm influence distance of the torso)
 *
 * Both must agree to round-off: it is the same arithmetic. The operator also
 * drops the left hand for 0.5 s (the arm must hold) and loses the right elbow
 * landmark for 1 s (the engine then synthesizes an elbow).
 *
 *     node web_gui/retarget_lab/tools/webcam_parity.mjs
 *         (needs python3 + MuJoCo + scipy, and the NOVA checkout beside this repo)
 */
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as BT from '../../../website/assets/bodyteleop-core.js';
import { DeployedRetargeter, Kin } from '../../../website/assets/deployed-retarget.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = join(HERE, '..', '..', '..', 'website');
const D = JSON.parse(readFileSync(join(SITE, 'assets/data/deployed-retarget.json'), 'utf8'));

let bad = 0;
const fail = (m) => { console.log('   *** ' + m); bad++; };

/* ── 0. forward kinematics against MuJoCo's own fixtures ───────────────── */
{
  const kin = new Kin(D);
  let worst = 0;
  for (const fx of D.fk_fixtures) {
    const F = kin.fk(fx.q);
    for (const [b, p] of Object.entries(fx.xpos)) {
      const q = F.body[kin.bix[b]].p;
      worst = Math.max(worst, ...q.map((v, i) => Math.abs(v - p[i])));
    }
    for (const [b, R] of Object.entries(fx.palmR)) {
      const Q = F.body[kin.bix[b]].R;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) worst = Math.max(worst, Math.abs(Q[i][j] - R[i][j]));
    }
    for (const [S, p] of Object.entries(fx.wrist_anchor)) {
      const q = F.anchor[`${S} Wrist`];
      worst = Math.max(worst, ...q.map((v, i) => Math.abs(v - p[i])));
    }
  }
  console.log(`forward kinematics vs MuJoCo: max error ${worst.toExponential(2)} (m / unit), ${D.fk_fixtures.length} random poses`);
  if (worst > 1e-9) fail('the kinematic tree does not reproduce MuJoCo');
}

/* ── 1. a synthetic operator (MediaPipe world: m, hip origin, +y DOWN,
         +z away from the camera; landmark 11 = the operator's LEFT shoulder) ─ */
function mkPose(o) {
  const lean = o.lean || 0, hd = o.head || [0, 0, 0];
  const P = new Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 1 }));
  const put = (i, v) => { P[i] = { x: v[0], y: v[1], z: v[2], visibility: 1 }; };
  const hc = [0.0, -0.655, 0.02];
  const yaw = hd[0] * 4, pit = hd[1] * 4, rol = hd[2] * 4;
  const rot = ([x, y, z]) => {
    let c = Math.cos(yaw), s = Math.sin(yaw);
    const x1 = x * c - z * s, z1 = x * s + z * c;
    c = Math.cos(pit); s = Math.sin(pit);
    const y1 = y * c - z1 * s, z2 = y * s + z1 * c;
    c = Math.cos(rol); s = Math.sin(rol);
    return [hc[0] + x1 * c - y1 * s, hc[1] + x1 * s + y1 * c, hc[2] + z2];
  };
  put(0, rot([0, -0.005, -0.075])); put(2, rot([0.03, -0.028, -0.055]));
  put(5, rot([-0.03, -0.028, -0.055])); put(7, rot([0.07, -0.005, 0]));
  put(8, rot([-0.07, -0.005, 0]));
  put(11, [0.19, -0.50, 0]); put(12, [-0.19, -0.50, 0]);
  put(23, [0.11, 0, 0]); put(24, [-0.11, 0, 0]);
  put(25, [0.11, 0.42, 0.02]); put(26, [-0.11, 0.42, 0.02]);
  put(27, [0.11, 0.84, 0.04]); put(28, [-0.11, 0.84, 0.04]);
  put(13, o.le); put(14, o.re); put(15, o.lw); put(16, o.rw);
  /* LEAN = a hip bend of `lean` rad toward the camera: everything above the hips
     pivots about them, so the head DROPS by h(1 − cos) as a real bend does (the
     engine reads an upper-body lean from that drop; a forward slide is a walk) */
  if (lean) {
    const c = Math.cos(lean), s = Math.sin(lean);
    for (const i of [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]) {
      const { y, z } = P[i];
      P[i] = { ...P[i], y: c * y - s * z, z: s * y + c * z };
    }
  }
  return P;
}
function mkHand(side, fingers, faces, at) {
  const nrm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const f = nrm(fingers);
  let n = faces.map((v, i) => v - f[i] * (faces[0] * f[0] + faces[1] * f[1] + faces[2] * f[2]));
  n = nrm(n);
  const a = [f[1] * n[2] - f[2] * n[1], f[2] * n[0] - f[0] * n[2], f[0] * n[1] - f[1] * n[0]];
  const s = side === 'left' ? -1 : 1;
  const P = new Array(21).fill(null).map(() => ({ x: at[0], y: at[1], z: at[2] }));
  const put = (i, al, ac, no) => {
    P[i] = { x: at[0] + f[0] * al + a[0] * ac * s + n[0] * no,
             y: at[1] + f[1] * al + a[1] * ac * s + n[1] * no,
             z: at[2] + f[2] * al + a[2] * ac * s + n[2] * no };
  };
  put(0, 0, 0, 0); put(5, 0.075, 0.039, 0); put(9, 0.082, 0, 0);
  put(13, 0.079, -0.013, 0); put(17, 0.072, -0.039, 0);
  for (const [k, b] of [[8, 5], [12, 9], [16, 13], [20, 17]])
    P[k] = { x: P[b].x + f[0] * 0.07, y: P[b].y + f[1] * 0.07, z: P[b].z + f[2] * 0.07 };
  for (const [k, b] of [[6, 5], [7, 5], [10, 9], [11, 9], [14, 13], [15, 13], [18, 17], [19, 17]])
    P[k] = { x: P[b].x + f[0] * 0.03, y: P[b].y + f[1] * 0.03, z: P[b].z + f[2] * 0.03 };
  [2, 3, 4].forEach((i, k) => {
    P[i] = { x: P[5].x + a[0] * 0.02 * s * (k + 1), y: P[5].y + a[1] * 0.02 * s * (k + 1),
             z: P[5].z + a[2] * 0.02 * s * (k + 1) };
  });
  return P;
}

/* keyposes: wrists / elbows (mp), hand fingers + palm-facing directions (mp),
   torso lean, head [turn, down, roll]. +x is the operator's LEFT. */
const DOWN = [0, 1, 0], UP = [0, -1, 0], TOWARD = [0, 0, -1], LEFTX = [1, 0, 0], RIGHTX = [-1, 0, 0];
const K = [
  { name: 'rest',          lw: [0.26, -0.16, -0.02], rw: [-0.26, -0.16, -0.02], le: [0.25, -0.31, 0.01], re: [-0.25, -0.31, 0.01],
    lh: [DOWN, RIGHTX], rh: [DOWN, LEFTX] },
  { name: 'right hand up', lw: [0.26, -0.16, -0.02], rw: [-0.24, -0.78, -0.22], le: [0.25, -0.31, 0.01], re: [-0.27, -0.52, -0.12],
    lh: [DOWN, RIGHTX], rh: [UP, TOWARD] },
  { name: 'reach forward', lw: [0.14, -0.48, -0.46], rw: [-0.14, -0.48, -0.46], le: [0.21, -0.43, -0.20], re: [-0.21, -0.43, -0.20],
    lh: [TOWARD, DOWN], rh: [TOWARD, DOWN] },
  { name: 'reach right',   lw: [0.18, -0.30, -0.25], rw: [-0.70, -0.46, -0.06], le: [0.24, -0.33, -0.08], re: [-0.45, -0.49, 0.0],
    lh: [TOWARD, DOWN], rh: [RIGHTX, DOWN] },
  { name: 'cross body',    lw: [-0.12, -0.40, -0.30], rw: [-0.26, -0.18, -0.02], le: [0.17, -0.36, -0.12], re: [-0.25, -0.31, 0.01],
    lh: [RIGHTX, TOWARD], rh: [DOWN, LEFTX] },
  { name: 'hands meet',    lw: [0.05, -0.42, -0.36], rw: [-0.05, -0.42, -0.36], le: [0.21, -0.38, -0.16], re: [-0.21, -0.38, -0.16],
    lh: [TOWARD, RIGHTX], rh: [TOWARD, LEFTX] },
  { name: 'lean in',       lw: [0.15, -0.40, -0.55], rw: [-0.15, -0.40, -0.55], le: [0.21, -0.42, -0.30], re: [-0.21, -0.42, -0.30],
    lh: [TOWARD, DOWN], rh: [TOWARD, DOWN], lean: 0.6 },
  { name: 'palm up, look', lw: [0.26, -0.16, -0.02], rw: [-0.20, -0.40, -0.40], le: [0.25, -0.31, 0.01], re: [-0.24, -0.33, -0.12],
    lh: [DOWN, RIGHTX], rh: [TOWARD, UP], head: [0.08, 0.06, 0.0] },
  { name: 'near chest',    lw: [0.09, -0.44, -0.13], rw: [-0.09, -0.44, -0.13], le: [0.24, -0.28, -0.02], re: [-0.24, -0.28, -0.02],
    lh: [RIGHTX, TOWARD], rh: [LEFTX, TOWARD] },
  { name: 'rest',          lw: [0.26, -0.16, -0.02], rw: [-0.26, -0.16, -0.02], le: [0.25, -0.31, 0.01], re: [-0.25, -0.31, 0.01],
    lh: [DOWN, RIGHTX], rh: [DOWN, LEFTX] },
];
const HZ = 30, MOVE = 1.0, HOLD = 0.8;
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 - 0.5; };
const lerp = (a, b, k) => a.map((v, i) => v + (b[i] - v) * k);
function at(t) {
  const seg = MOVE + HOLD;
  const i = Math.min(K.length - 2, Math.floor(t / seg));
  const u = Math.min(1, Math.max(0, (t - i * seg - HOLD) / MOVE));
  const k = 0.5 - 0.5 * Math.cos(Math.PI * u);
  const A = K[i], B = K[i + 1];
  const g = (key, dflt) => lerp(A[key] || dflt, B[key] || dflt, k);
  return { lw: g('lw'), rw: g('rw'), le: g('le'), re: g('re'),
           lh: [lerp(A.lh[0], B.lh[0], k), lerp(A.lh[1], B.lh[1], k)],
           rh: [lerp(A.rh[0], B.rh[0], k), lerp(A.rh[1], B.rh[1], k)],
           lean: (A.lean || 0) + ((B.lean || 0) - (A.lean || 0)) * k,
           head: lerp(A.head || [0, 0, 0], B.head || [0, 0, 0], k),
           name: u < 0.5 ? A.name : B.name };
}
const T_END = (K.length - 1) * (MOVE + HOLD) + HOLD;
const frames = [];
const trackers = { left: new BT.HandShapeTracker('left'), right: new BT.HandShapeTracker('right') };
let calib = null, seq = 0;
for (let n = 0; n * (1 / HZ) <= T_END; n++) {
  const t = n / HZ, o = at(t);
  const jit = (v) => v.map((x) => x + 0.003 * rnd());          // landmark noise, ±1.5 mm
  const world = mkPose({ ...o, lw: jit(o.lw), rw: jit(o.rw), le: jit(o.le), re: jit(o.re) });
  const vis = world.map(() => 1);
  if (t >= 6.2 && t < 7.2) vis[14] = 0;                        // right elbow not seen
  if (!calib) calib = BT.computeAnchor(world, vis);
  const hands = {
    left: t >= 4.4 && t < 4.9 ? null                            // left hand drops out
      : mkHand('left', o.lh[0], o.lh[1], [world[15].x, world[15].y, world[15].z]),
    right: mkHand('right', o.rh[0], o.rh[1], [world[16].x, world[16].y, world[16].z]),
  };
  const f = BT.synthesizeFrame({ t, pose: { world, vis }, hands }, calib, seq++,
    { trackers, refine: true, scaleShape: true, fuse: false, elbows: true });
  if (f) frames.push({ t, name: o.name, p: f });
}
console.log(`operator: ${frames.length} webcam frames at ${HZ} Hz, ${T_END.toFixed(1)} s, ` +
            `${K.length} keyposes (${[...new Set(K.map((k) => k.name))].join(', ')})`);

/* ── 2. both retargeters on the same frames ─────────────────────────────── */
const WIRE = (n) => n.toLowerCase().replace(/ /g, '_');
const tmp = mkdtempSync(join(tmpdir(), 'rtparity-'));
const fin = join(tmp, 'frames.json');
writeFileSync(fin, JSON.stringify(frames));

function python(extra) {
  const fout = join(tmp, 'out.json');
  try {
    execFileSync(process.env.PYTHON || 'python3',
      [join(HERE, 'webcam_parity_ref.py'), fin, fout, ...extra],
      { stdio: ['ignore', 'ignore', 'pipe'], timeout: 600000 });
  } catch (e) {
    console.log(String(e.stderr || e).split('\n').slice(-6).join('\n'));
    return null;
  }
  return JSON.parse(readFileSync(fout, 'utf8'));
}
function js(selfCollision) {
  const Dx = { ...D, hp: { ...D.hp, SELF_COLLISION: selfCollision } };
  const eng = new DeployedRetargeter(Dx);
  const out = [];
  const t0 = process.hrtime.bigint();
  for (const f of frames) {
    const r = eng.step({ ...f.p, timestamp: f.t }, f.t);
    const q = {};
    for (const [k, v] of Object.entries(r.q)) q[WIRE(k)] = v;
    Object.assign(q, r.neck);
    out.push({ q, targets: r.targets, engaged: r.engaged, qp_rows: eng.proj ? eng.proj.active : 0 });
  }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / frames.length;
  return { out, ms };
}

const GROUPS = {
  'arms (14)': (k) => /_arm_\d$/.test(k),
  'wrists (2)': (k) => /_wrist$/.test(k),
  'torso': (k) => k === 'torso',
  'neck (3)': (k) => k.startsWith('neck_'),
};
function compare(label, py, J, skip) {
  console.log(`\n${label}`);
  const keys = Object.keys(py.frames[0].q);
  const stats = {};
  let tgtWorst = 0, engagedMismatch = 0, n = 0;
  for (let i = 0; i < frames.length; i++) {
    if (skip && skip(i)) continue;
    const a = py.frames[i], b = J.out[i];
    if (a.engaged !== b.engaged) engagedMismatch++;
    n++;
    for (const k of keys) {
      const g = Object.keys(GROUPS).find((gk) => GROUPS[gk](k));
      if (!g) continue;
      const e = Math.abs(a.q[k] - b.q[k]);
      const s = stats[g] || (stats[g] = { max: 0, sum: 0, n: 0, at: '', lo: Infinity, hi: -Infinity });
      s.lo = Math.min(s.lo, a.q[k]); s.hi = Math.max(s.hi, a.q[k]);
      if (e > s.max) { s.max = e; s.at = `${k} @ ${frames[i].t.toFixed(2)} s (${frames[i].name})`; }
      s.sum += e; s.n++;
    }
    for (const s of ['right', 'left']) {
      const ta = a.targets[s], tb = b.targets[s];
      if (ta && tb) tgtWorst = Math.max(tgtWorst, ...ta.map((v, j) => Math.abs(v - tb[j])));
      else if (!!ta !== !!tb) engagedMismatch++;
    }
  }
  for (const [g, s] of Object.entries(stats))
    console.log(`  ${g.padEnd(11)} max |dq| ${s.max.toExponential(2)} rad   mean ${(s.sum / s.n).toExponential(2)} rad` +
                `   (NOVA swept ${s.lo.toFixed(2)}..${s.hi.toFixed(2)} rad)` +
                (s.max > 1e-6 ? `   worst ${s.at}` : ''));
  console.log(`  wrist targets max |dp| ${tgtWorst.toExponential(2)} m over ${n} frames; ` +
              `engagement/target-presence mismatches ${engagedMismatch}`);
  return { stats, tgtWorst, engagedMismatch };
}

const pyPort = python(['--no-self-collision']);
if (!pyPort) {
  console.log('\nthe Python reference could not run (needs python3 with MuJoCo + scipy, and the NOVA checkout)');
  console.log('RESULT: FAIL');
  rmSync(tmp, { recursive: true, force: true });
  process.exit(1);
}
const jsPort = js(false);
const P = compare('port — self-collision off on both sides (the arithmetic itself)', pyPort, jsPort);
for (const s of Object.values(P.stats)) if (s.max > 1e-6) { fail('the port disagrees with the Python engine'); break; }
if (P.engagedMismatch) fail('the two engage on different frames');

const pyDep = python([]);
const jsDep = js(true);
const active = pyDep.frames.map((f) => f.qp_rows > 0);
const nAct = active.filter(Boolean).length;
const firstAct = active.indexOf(true);
console.log(`\ndeployed (NOVA: self-collision QP on) — the QP had active barrier rows on ` +
            `${nAct}/${frames.length} frames (${(100 * nAct / frames.length).toFixed(1)} %)` +
            (firstAct >= 0 ? `, first at ${frames[firstAct].t.toFixed(2)} s (${frames[firstAct].name})` : ''));
const rowDiff = pyDep.frames.filter((f, i) => f.qp_rows !== jsDep.out[i].qp_rows).length;
console.log(`  barrier rows per frame: Python and JS disagree on ${rowDiff} frames`);
if (rowDiff) fail('the two projectors see different active capsule pairs');
const Dall = compare('deploy — as NOVA runs it (projector on)', pyDep, jsDep);
for (const s of Object.values(Dall.stats)) if (s.max > 1e-6) { fail('the deployed port disagrees with NOVA'); break; }
for (const [g, s] of Object.entries(Dall.stats))       // a parity over a robot that never moved proves nothing
  if (s.hi - s.lo < 0.1) fail(`${g}: the operator never moved it (span ${(s.hi - s.lo).toFixed(3)} rad)`);
if (Dall.engagedMismatch) fail('the two engage on different frames');
console.log(`\nJS step cost: ${jsDep.ms.toFixed(2)} ms per frame (node, this machine)`);
if (jsDep.ms > 8) fail('the port is too slow for a 30 Hz camera');

rmSync(tmp, { recursive: true, force: true });
console.log('\nRESULT:', bad ? 'FAIL' : 'PASS');
process.exit(bad ? 1 : 0);
