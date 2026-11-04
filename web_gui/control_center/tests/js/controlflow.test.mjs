// Guards for the 2026-07-17 control-flow revision. These read the app source
// (like viewrail.test.mjs) so a future edit that quietly reverts one of the
// safety-relevant behaviours fails loudly instead of silently.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const rd = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const APP = rd('../../web/js/app.js');
const HTML = rd('../../web/index.html');
const VP = rd('../../web/js/viewport3d.js');
const SP = rd('../../web/js/statuspanel.js');

// ── item 5: ONE power button, gating on energised ──────────────────────────
test('the top bar has ONE power button, not separate torque + e-stop', () => {
  assert.ok(HTML.includes('id="powerBtn"'), 'powerBtn must exist');
  assert.ok(!HTML.includes('id="estopBtn"'), 'the separate E-STOP button must be gone');
  assert.ok(!HTML.includes('id="torqueBtn"'), 'the separate Torque button must be gone');
});

test('energising is the gate — the space picker hides until energised', () => {
  // refreshOverlay must derive `energized` from torque && !estop and hide the
  // space picker / feel toggle when not energised.
  assert.match(APP, /const energized = ctl\.torque && !ctl\.estop/);
  assert.match(APP, /\$\('spaceSeg'\)\.classList\.toggle\('hidden', !kb\)/);
  assert.match(APP, /\$\('feelSeg'\)\.classList\.toggle\('hidden', !energized\)/);
});

test('the power button toggles the soft estop path (instant, reversible)', () => {
  // energised → estop active:true (stop); limp → estop active:false (energise).
  assert.match(APP, /link\.send\('estop', \{ active: true \}\)/);
  assert.match(APP, /link\.send\('estop', \{ active: false \}\)/);
});

// ── item 2: task space is plan → preview → execute, NOT live streaming ──────
test('task-space IK routes to a plan, never straight to the wire', () => {
  // onJointTargets: task space stages a plan; other spaces command live.
  assert.match(APP, /if \(ctl\.space === 'task'\)\{ wristPlan\?\.stage\(m\); return; \}/);
  // and a limp / soft / e-stopped robot takes nothing at all
  assert.match(APP, /if \(ctl\.estop \|\| !ctl\.torque \|\| ctl\.feel === 'soft'\) return;/);
});

test('the plan previews on the MoveIt ghost, not by moving the solid robot', () => {
  assert.match(VP, /setPlanHold\(joints\)/);
  assert.match(VP, /clearPlanHold\(\)/);
  // the plan renders on the translucent ghost rig; the solid robot stays live
  assert.match(VP, /_poseGhostPlan\(\)/);
  assert.match(VP, /clearPlanHold\(\)\{[\s\S]*?poseGhost\(null\)/);
  // solveIK restores the solid rig to the live pose after solving (the ghost
  // shows the plan) — the old code left the solid rig posed at the solution
  assert.match(VP, /this\.rig\.pose\(this\._liveQ/);
  assert.ok(HTML.includes('id="planBar"') && HTML.includes('id="planExec"'),
    'the Execute bar must exist');
});

// ── item 1: the wrist ball belongs to Task space only ──────────────────────
test('the wrist layer follows the space — on in Task, off elsewhere', () => {
  assert.match(APP, /viewport\.setLayer\('wrist', v === 'task'\)/);
  // the old bug set it on for ANY non-drive space (task OR joint)
  assert.ok(!APP.includes("if (v !== 'drive'){\n      viewport.setLayer('wrist', true);"),
    'must NOT enable the wrist ball for every non-drive space');
});

// ── item 4: SLAM sits in the left cluster ──────────────────────────────────
test('SLAM/maps live in the left cluster, not topright', () => {
  const left = HTML.indexOf('class="topleft"');
  const slam = HTML.indexOf('class="slam-wrap"');
  const right = HTML.indexOf('class="topright"');
  assert.ok(left >= 0 && slam > left && slam < right,
    'the SLAM menu must sit inside .topleft, before .topright');
});

// ── item 6: the status panel renders the hardware roster ───────────────────
test('the status panel builds a hardware roster from modules + telemetry', () => {
  assert.match(SP, /_hardwareRows\(\)/);
  assert.match(SP, /_rowsFromModules/);
  assert.match(SP, /_rowsFromTelemetry/);
  // grouped Body / Sensors / Audio
  assert.ok(SP.includes("'speaker'") && SP.includes("'mic'"), 'audio rows');
  assert.match(SP, /np-sec">Hardware/);
});

// ── joint state + trends live in the RIGHT dock now (collapsible) ──────────
test('joint state + trends are collapsible sections of the Status dock', () => {
  // moved off the bottom band into #dockTelemetry (a second Status-tab pane)
  assert.ok(HTML.includes('id="dockTelemetry"'), 'the dock telemetry pane must exist');
  assert.ok(HTML.includes('id="jointSec"') && HTML.includes('id="trendsSec"'),
    'the joint + trends sections must exist in the dock');
  // both carry their canvas/graph and a collapsible header
  assert.ok(HTML.includes('id="jointCanvas"') && HTML.includes('id="histBand"'));
  assert.match(HTML, /class="dt-head" data-sec="joints"/);
  const CSS = rd('../../web/css/control.css');
  assert.match(CSS, /\.dt-sec\.collapsed \.dt-body\{ display:none/);
});

test('the bottom band is CAMERAS ONLY — no joint/trends there', () => {
  // the plots section must hold the camera strip and nothing else
  assert.match(HTML, /<section class="plots plots-cams"[\s\S]*?id="camStrip"[\s\S]*?<\/section>/);
  // the old bottom-band scaffolding is gone
  assert.ok(!HTML.includes('id="bandsRow"') && !HTML.includes('id="jointBand"'),
    'the old side-by-side bands must be removed from the bottom');
  assert.ok(!HTML.includes('id="camsToggle"'), 'the old band-toggle chips are gone');
});
