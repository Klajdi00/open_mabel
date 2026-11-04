// bus pub/sub, actuator-name mapping, turbo colormap, hand-landmark retarget.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as bus from '../../web/js/bus.js';
import { buildActMap, actuatorsToRig, GROUP_ORDER } from '../../web/js/names.js';
import { retargetLandmarks } from '../../web/js/dexterity.js';

test('bus on/emit/off + unsubscribe return', () => {
  let got = null;
  const off = bus.on('x', (v) => { got = v; });
  bus.emit('x', 42);
  assert.equal(got, 42);
  off();
  bus.emit('x', 43);
  assert.equal(got, 42);
});

test('bus survives a throwing subscriber', () => {
  const orig = console.error; console.error = () => {};
  try {
    let ok = false;
    bus.on('boom', () => { throw new Error('sub blew up'); });
    bus.on('boom', () => { ok = true; });
    bus.emit('boom');
    assert.ok(ok);
  } finally { console.error = orig; }
});

test('actuator map splits Lift into the two prismatic joints', () => {
  const map = buildActMap(['Lift', 'Torso', 'Left Arm 1']);
  const q = actuatorsToRig(new Float32Array([0.4, 0.1, -0.2]), map);
  const close = (a, b) => Math.abs(a - b) < 1e-6;      // Float32Array precision
  assert.ok(close(q.lift_lower, 0.2) && close(q.lift_upper, 0.2));
  assert.ok(close(q.torso, 0.1));
  assert.ok(close(q.left_arm_1, -0.2));
  assert.equal(GROUP_ORDER.length, 7);
});

test('retargetLandmarks: open hand → low curl, closed fist → high curl', () => {
  // A plausible RIGHT hand in MediaPipe camera coords (x right, y down, z away):
  // fingers up (−y), palm toward the camera, knuckles spread across x.
  // (The old all-collinear synthetic hand had NO palm plane — the anatomical
  // decomposition rightly refuses it, so use realistic geometry.)
  const mk = () => {
    const p = [];
    p[0] = { x: 0, y: 0, z: 0 };                                     // wrist
    p[1] = { x: -0.025, y: -0.02, z: -0.005 };                       // thumb CMC
    p[2] = { x: -0.05, y: -0.045, z: -0.01 };
    p[3] = { x: -0.065, y: -0.07, z: -0.012 };
    p[4] = { x: -0.075, y: -0.09, z: -0.014 };
    const bases = { 5: -0.03, 9: -0.01, 13: 0.012, 17: 0.034 };
    for (const [b, x] of Object.entries(bases)){
      const i = +b;
      p[i] = { x, y: -0.09, z: 0 };                                  // knuckle
      p[i + 1] = { x, y: -0.125, z: 0 };                             // PIP
      p[i + 2] = { x, y: -0.148, z: 0 };                             // DIP
      p[i + 3] = { x, y: -0.168, z: 0 };                             // tip
    }
    return p;
  };
  const open = mk();
  const po = retargetLandmarks(open, 'right');
  assert.ok(po.index_pip < 25, `open index ${po.index_pip}`);
  assert.equal(po.index_abd, 50);

  // curled fist: PIP folds ~100° toward the palm (toward the camera, −z)
  const curled = mk();
  for (const b of [5, 9, 13, 17]){
    const kn = curled[b];
    curled[b + 1] = { x: kn.x, y: kn.y - 0.030, z: -0.018 };         // MCP ~45°
    curled[b + 2] = { x: kn.x, y: kn.y - 0.028, z: -0.042 };         // PIP fold
    curled[b + 3] = { x: kn.x, y: kn.y - 0.010, z: -0.050 };         // tip in palm
  }
  const pc = retargetLandmarks(curled, 'right');
  assert.ok(pc.index_pip > po.index_pip + 20,
    `curl must rise: ${po.index_pip} → ${pc.index_pip}`);
  assert.ok(pc.index_mcp > po.index_mcp + 15,
    `MCP must rise: ${po.index_mcp} → ${pc.index_mcp}`);
  // outputs stay in 0..100 for every joint
  for (const v of Object.values(pc)) assert.ok(v >= 0 && v <= 100);
  // degenerate (collinear) input → safe neutral, not NaN
  const flatline = Array.from({ length: 21 }, (_, i) => ({ x: 0, y: i * 0.05, z: 0 }));
  const pf = retargetLandmarks(flatline, 'right');
  for (const v of Object.values(pf)) assert.ok(Number.isFinite(v));
});
