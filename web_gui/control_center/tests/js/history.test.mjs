// TelemetryHistory — the ring buffer behind the Trends band and the floating
// Joint-History window.
//
// The bug these guard against rendered a COMPLETELY BLANK graph while every
// "is the data arriving?" check passed: the buffer filled, the autoscale found
// finite min/max, and the lines were drawn ~1e7 screen-widths off-canvas
// because the server stamps `t` with time.time() (wall-clock epoch, ~1.78e9)
// and every plot measures "now" with performance.now()/1000 (~1e1). Asserting
// on sample COUNT would have stayed green throughout — so assert on the CLOCK
// DOMAIN instead.
import test from 'node:test';
import assert from 'node:assert/strict';
import { TelemetryHistory, DEFAULT_WINDOW_S } from '../../web/js/history.js';

const SCHEMA = { names: ['a', 'b'], modules: [{ key: 'm', label: 'M', joints: ['a', 'b'] }] };
const mk = () => { const h = new TelemetryHistory(15); h.setSchema(SCHEMA); return h; };

test('samples are stamped in the plots\' clock domain, not the server\'s', () => {
  const h = mk();
  const EPOCH = 1784482135.87;              // a real time.time() from the backend
  h.push({ t: EPOCH, p: [0.1, 0.2], tgt: [0.1, 0.2], v: [0, 0], e: [0, 0] });
  const { t } = h.series(0, 'p');
  const now = performance.now() / 1000;
  // the stored stamp must be comparable to the plots' "now", i.e. seconds of
  // page uptime — NOT the epoch the wire carried
  assert.ok(Math.abs(now - t[0]) < 5,
    `stamp ${t[0]} is not in the performance.now() domain (now=${now})`);
  assert.ok(t[0] < 1e6, 'a wall-clock epoch leaked into the buffer');
});

test('a full window of samples lands inside the plotted window', () => {
  const h = mk();
  for (let i = 0; i < 60; i++)
    h.push({ t: 1784482135 + i / 15, p: [i * 0.01, 0], tgt: [0, 0], v: [0, 0], e: [0, 0] });
  const { t } = h.series(0, 'p', DEFAULT_WINDOW_S);
  const now = performance.now() / 1000;
  // this is exactly what histband.js's px() computes; off-window => off-canvas
  for (const ti of t){
    const u = 1 - (now - ti) / DEFAULT_WINDOW_S;
    assert.ok(u >= 0 && u <= 1.001, `sample maps to x-fraction ${u} — off canvas`);
  }
});

test('velocity falls back to d(pos)/dt when the robot publishes no velocity', async () => {
  const h = mk();
  // /joint_states assembled by joint_state_publisher ZERO-FILLS velocity, so
  // the real robot often streams nulls here while position is perfectly good.
  // Push with REAL gaps: push() stamps from performance.now(), and a
  // differentiator that cannot see a dt correctly refuses to invent one.
  const RATE = 1.0;                              // rad/s the joint is moving
  const t0 = performance.now() / 1000;
  for (let i = 0; i < 8; i++){
    const pos = (performance.now() / 1000 - t0) * RATE;
    h.push({ t: 0, p: [pos, 0], tgt: [0, 0], v: [null, null], e: [null, null] });
    await new Promise((r) => setTimeout(r, 12));
  }
  const s = h.series(0, 'v');
  assert.equal(s.derived, true, 'should report the series as derived');
  const finite = [...s.v].filter(Number.isFinite);
  assert.ok(finite.length > 0, 'derived velocity produced no samples');
  // timer jitter makes the exact value machine-dependent; the point is that it
  // recovers the right magnitude rather than zero or garbage
  const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
  assert.ok(Math.abs(mean - RATE) < 0.5 * RATE,
    `derived velocity ${mean.toFixed(2)} is not ~${RATE} rad/s`);
});

test('the derivative refuses to differentiate over a zero time gap', () => {
  const h = mk();
  // several samples inside one millisecond: dt ~= 0 would divide into a
  // meaningless spike, so the fallback must decline rather than fabricate
  for (let i = 0; i < 6; i++)
    h.push({ t: 0, p: [i * 0.5, 0], tgt: [0, 0], v: [null, null], e: [null, null] });
  const s = h.series(0, 'v');
  const finite = [...s.v].filter(Number.isFinite);
  assert.ok(finite.every((v) => Math.abs(v) < 1e4),
    'a near-zero dt produced a divide-by-zero spike');
});

test('a real velocity stream is NOT replaced by the derivative', () => {
  const h = mk();
  for (let i = 0; i < 5; i++)
    h.push({ t: 0, p: [i * 0.1, 0], tgt: [0, 0], v: [7.0, 7.0], e: [0, 0] });
  const s = h.series(0, 'v');
  assert.equal(s.derived, false);
  assert.equal(s.v[s.v.length - 1], 7.0);
});

test('hasChannel reports what the robot is actually publishing', () => {
  const h = mk();
  h.push({ t: 0, p: [0.5, 0.5], tgt: [0.5, 0.5], v: [null, null], e: [null, null] });
  assert.equal(h.hasChannel('p'), true);
  assert.equal(h.hasChannel('v'), false, 'null velocity must not read as present');
  assert.equal(h.hasChannel('e'), false);
});

test('the ring buffer wraps without corrupting the time axis', () => {
  const h = new TelemetryHistory(1);          // 1 Hz => cap = 300
  h.setSchema(SCHEMA);
  for (let i = 0; i < 350; i++)
    h.push({ t: 0, p: [i, 0], tgt: [0, 0], v: [0, 0], e: [0, 0] });
  const s = h.series(0, 'p', 300);
  assert.equal(s.v.length, 300);
  assert.equal(s.v[s.v.length - 1], 349, 'newest sample must be last');
  assert.equal(s.v[0], 50, 'oldest retained sample must be first');
  for (let i = 1; i < s.t.length; i++)
    assert.ok(s.t[i] >= s.t[i - 1], 'time axis must be monotonic across the wrap');
});
