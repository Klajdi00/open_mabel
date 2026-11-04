// Joint-panel grading: tracking-error and temperature thresholds.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { trackStatus, tempStatus, normVal, STATUS_COL, ERR_THRESH }
  from '../../web/js/grading.js';

test('trackStatus pos thresholds green/yellow/red', () => {
  assert.equal(trackStatus(0.0, 0.02, 'pos'), 0);          // 0.02 < 0.05
  assert.equal(trackStatus(0.0, 0.05, 'pos'), 1);          // at warn
  assert.equal(trackStatus(0.0, 0.10, 'pos'), 1);
  assert.equal(trackStatus(0.0, 0.18, 'pos'), 2);          // at bad
  assert.equal(trackStatus(1.0, 0.5, 'pos'), 2);
});

test('trackStatus is symmetric and null-safe', () => {
  assert.equal(trackStatus(0.3, 0.0, 'pos'), trackStatus(0.0, 0.3, 'pos'));
  assert.equal(trackStatus(null, 0.1, 'pos'), null);
  assert.equal(trackStatus(0.1, null, 'pos'), null);
});

test('trackStatus per-mode thresholds', () => {
  assert.equal(trackStatus(0, 0.4, 'vel'), 1);
  assert.equal(trackStatus(0, 1.2, 'vel'), 2);
  assert.equal(trackStatus(0, 0.4, 'torque'), 0);
  assert.equal(trackStatus(0, 0.8, 'torque'), 1);
  assert.equal(trackStatus(0, 2.0, 'torque'), 2);
});

test('tempStatus thresholds', () => {
  assert.equal(tempStatus(40), 0);
  assert.equal(tempStatus(55), 1);
  assert.equal(tempStatus(69.9), 1);
  assert.equal(tempStatus(70), 2);
  assert.equal(tempStatus(null), null);
  assert.equal(tempStatus(60, 65, 80), 0);                 // custom thresholds
});

test('normVal clamps to ±1 and null → 0', () => {
  assert.equal(normVal(null, 'pos'), 0);
  assert.equal(normVal(99, 'pos'), 1);
  assert.equal(normVal(-99, 'vel'), -1);
  assert.ok(Math.abs(normVal(Math.PI / 2, 'pos') - 0.5) < 1e-9);
});

test('status palette has 3 entries', () => {
  assert.equal(STATUS_COL.length, 3);
  assert.ok(ERR_THRESH.pos[0] < ERR_THRESH.pos[1]);
});
