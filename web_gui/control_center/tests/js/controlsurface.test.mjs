// The control-surface pieces that are pure logic: the one drive-sign conversion,
// robot selection from the gateway hello, gravity comp on control_mode, the
// timeline's capability-gated nav primitives, the IK posture cap, and the hand
// library's wrist rule. No DOM, no network (stub socket).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { navAxes } from '../../web/js/teleopctl.js';
import { RobotLink } from '../../web/js/link.js';
import { robotKind, setRobot, refFor } from '../../web/js/handmodel.js';
import { REST_POSE } from '../../web/js/homepose.js';
import { navMoveFor, navMoveSeconds, NAV_MOVE_SPEED } from '../../web/js/timeline.js';
import { POSTURE_MAX_STEP } from '../../web/js/kinematics.js';
import { ANIMATIONS, SHAPES } from '../../web/js/handanims.js';
import { EXEC_RATE, BLEND_THRESH, MAX_JOINT_VEL } from '../../web/js/motions.js';

function stubbed(){
  const link = new RobotLink();
  const sent = [];
  link.ws = { readyState: 1, send: (s) => sent.push(JSON.parse(s)), close(){} };
  return { link, sent };
}

test('navAxes keeps this studio\'s MABEL wire signs (the one sign conversion)', () => {
  // screen: up = forward, right = strafe right, clockwise = yaw right
  assert.deepEqual(navAxes({ lx: 0, ly: 1, rx: 0, ry: 0 }), { lx: -0, ly: 1, rx: -0, ry: 0 });
  assert.deepEqual(navAxes({ lx: 1, ly: 0, rx: 0, ry: 0 }), { lx: -1, ly: 0, rx: -0, ry: 0 });
  assert.deepEqual(navAxes({ lx: 0, ly: 0, rx: 1, ry: 0 }), { lx: -0, ly: 0, rx: -1, ry: 0 });
  assert.deepEqual(navAxes({ lx: 0, ly: 0, rx: 0, ry: 0.5 }), { lx: -0, ly: 0, rx: -0, ry: 0.5 });
});

test('hello.robot selects the hand mapping; MABEL is the default', () => {
  // a fake WebSocket so connect() wires its real onmessage handler
  const prev = globalThis.WebSocket;
  let sock = null;
  globalThis.WebSocket = class { constructor(){ sock = this; this.readyState = 1; }
                                 send(){} close(){} };
  try {
    const link = new RobotLink();
    link.connect();
    const hello = (p) => sock.onmessage({ data: JSON.stringify({ type: 'hello', payload: p }) });
    hello({ robot: 'NOVA', name: 'nova' });
    assert.equal(robotKind(), 'nova');
    assert.equal(REST_POSE.left_wrist, 0, 'NOVA: URDF-frame wire, a level wrist is 0');
    assert.equal(refFor('wrist', 'left'), 0);
    hello({ robot: 'MABEL' });
    assert.equal(robotKind(), 'mabel');
    assert.equal(REST_POSE.left_wrist, 0.15, 'MABEL: the v1 MJCF ref');
    assert.equal(REST_POSE.right_wrist, -0.15);
    hello({});                                   // an older gateway with no `robot`
    assert.equal(robotKind(), 'mabel');
    link.close();
  } finally {
    setRobot('mabel');
    globalThis.WebSocket = prev;
  }
});

test('gravity comp rides control_mode as its own key, alone or with the spec', () => {
  const { link, sent } = stubbed();
  link.controlMode('joint', 'impedance', 'arm', 27, false);
  link.gravityComp(true);
  link.controlMode('joint', 'impedance', 'arm', 27);           // undefined = leave it alone
  assert.deepEqual(sent[0].payload, { method: 'joint', controlType: 'impedance',
                                      region: 'arm', stiffness: 27, gravityComp: false });
  assert.deepEqual(sent[1], { type: 'control_mode', payload: { gravityComp: true } });
  assert.equal('gravityComp' in sent[2].payload, false);
  link.close();
});

test('nav primitives: one nav_move per block, only where the gateway advertises it', () => {
  assert.equal(navMoveFor('wave', {}), null, 'not a nav primitive');
  const fwd = navMoveFor('nav_forward', { dist: 1.2 });
  assert.deepEqual(fwd, { dx: 1.2, dy: 0, dyaw: 0, speed: NAV_MOVE_SPEED });
  const tr = navMoveFor('nav_turn_right', { angle: 90 });
  assert.ok(Math.abs(tr.dyaw + Math.PI / 2) < 1e-9);
  assert.equal(tr.dx, 0); assert.equal(tr.dy, 0);
  // no calibration table -> no estimate (the gateway plans it)
  assert.equal(navMoveSeconds(fwd, null, { vmax: 0.6, wmax: 1.2 }), null);
  const cal = { kinds: { forward: { entries: [{ speed: 0.2, rate: 0.2, t0: 0.3 },
                                               { speed: 0.4, rate: 0.38, t0: 0.4 }] } } };
  // speed 0.5 x vmax 0.6 = 0.3 m/s -> the fastest entry not above it (0.2)
  assert.ok(Math.abs(navMoveSeconds(fwd, cal, { vmax: 0.6, wmax: 1.2 }) - (1.2 / 0.2 + 0.3)) < 1e-9);
});

test('the executor runs at the chain\'s 50 Hz and the blend threshold follows it', () => {
  assert.equal(EXEC_RATE, 50);
  assert.equal(BLEND_THRESH, MAX_JOINT_VEL / EXEC_RATE);
});

test('IK posture pull is capped per iteration', () => {
  assert.ok(POSTURE_MAX_STEP > 0 && POSTURE_MAX_STEP <= 0.05);
});

test('hand library: every clip is percent-space and stays inside 0..100', () => {
  assert.ok(Object.keys(ANIMATIONS).length >= 5);
  assert.ok(Object.keys(SHAPES).length >= 5);
  for (const [name, a] of Object.entries(ANIMATIONS)){
    assert.ok(a.dur > 0, `${name} dur`);
    for (let t = 0; t <= a.dur; t += 0.05){
      const p = a.at(t); if (!p) continue;
      for (const [k, v] of Object.entries(p))
        assert.ok(v >= -1e-6 && v <= 100 + 1e-6, `${name} ${k}=${v} at t=${t.toFixed(2)}`);
    }
  }
});
