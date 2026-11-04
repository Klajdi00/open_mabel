// FRAME CONTRACT REGRESSION GUARDS — read web-rig-frame-conventions before editing.
//
// Two frame bugs in the 3-D viewer have now recurred four times between them,
// and both look like "the visualization is slightly wrong" rather than like a
// crash, so nothing caught them but a human squinting at a screenshot. These
// are the cheap, deterministic checks that make them impossible to reintroduce.
//
//   A. WORLD-OVERLAY ALIGNMENT — the pointclouds / LiDAR / SLAM mesh must sit in
//      the odom frame the robot is posed in. The GLB bakes urdf_to_glb's
//      `T_center` (bbox-centred X/Z, feet at y=0) into base_link; the fix is to
//      strip that LATERALLY off the RIG, never to shove the WORLD by it.
//      Measured when it was wrong: floor cloud +7.8 cm above the wheels,
//      everything +23.3 cm sideways.
//
//   B. HAND JOINT REF — the :9090 wire streams ABSOLUTE MuJoCo qpos while the
//      GLB is baked at the assembled pose (qpos == ref), so every joint must be
//      driven by `qpos − ref`. The 14 ORCA finger/wrist joints have ref != 0, as
//      do the 2 lift stages (assembled at mid-travel so the default pose is
//      half-raised), so ignoring it leaves the arms perfect and the HANDS contorted
//      — thumbs hyperextended (ref −0.733 rad ≈ 42°) and fingers out of register
//      with both the MuJoCo view and their own pointcloud.
//
// These are source-level assertions on purpose: viewport3d.js imports `three`,
// which is not resolvable under node, so it cannot be imported here. The
// numeric, in-browser counterpart lives in
// simulation_control_center/tests/ui_layout_playwright.py.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, '..', '..', 'web');
const repo = join(here, '..', '..', '..', '..');
const vp3d = readFileSync(join(web, 'js', 'viewport3d.js'), 'utf8');

// ── A. world-overlay alignment ─────────────────────────────────────────────

test('navWorld (the odom group) sits at the ORIGIN', () => {
  assert.match(vp3d, /this\.navWorld\.position\.set\(0,\s*0,\s*0\)/,
    'navWorld must be pinned to the odom origin');
  assert.ok(!/navWorld\.position\.copy\(/.test(vp3d),
    'navWorld.position.copy(rig.baseP0) is THE bug: it shoves every world ' +
    'overlay by the robot\'s GLB bake, and copies a Z-up vector into a Y-up ' +
    'group so the lateral term becomes a height. Strip the bake off the RIG.');
});

test('the rig strips the GLB lateral bake but KEEPS its height', () => {
  const m = vp3d.match(/this\.world\.position\.set\(([^)]*)\)/);
  assert.ok(m, 'the rig must be shifted so base_link lands on the odom origin');
  const args = m[1].replace(/\s+/g, '');
  assert.equal(args, '-this.baseP0.x,0,-this.baseP0.z',
    'strip X and Z only — baseP0.y is base_link\'s real height above the ' +
    'floor, and zeroing it sinks the robot through the ground plane');
  // ...and it must live INSIDE Rig.load(), so EVERY instance gets it. The
  // MoveIt ghost is a second Rig: when the strip sat at the call site the
  // ghost kept the bake and rendered ~23 cm beside the live robot.
  const load = vp3d.match(/async load\(stage, manifest\)\{[\s\S]*?\n  \}/);
  assert.ok(load, 'Rig.load present');
  assert.match(load[0], /this\.world\.position\.set\(-this\.baseP0\.x, 0, -this\.baseP0\.z\)/,
    'the bake strip must be inside Rig.load(), not at the call site');
});

// ── B. hand joint ref ──────────────────────────────────────────────────────

test('setJoint drives joints by qpos MINUS ref, not raw qpos', () => {
  const m = vp3d.match(/setJoint\(name, val\)\{[\s\S]*?\n  \}/);
  assert.ok(m, 'setJoint present');
  const body = m[0];
  assert.match(body, /val\s*-\s*\(j\.ref\s*\|\|\s*0\)/,
    'the displacement must be qpos − ref');
  // and the raw value must not survive anywhere in the transform
  assert.ok(!/multiplyScalar\(val\)/.test(body) &&
            !/setFromAxisAngle\(j\._ax, val\)/.test(body),
    'raw `val` still reaches the transform — the hands will render contorted');
});

test('the web joint manifest carries ref for the ORCA hands and the lift', () => {
  const man = JSON.parse(
    readFileSync(join(repo, 'web_gui', 'shared_assets', 'mabel_joints.json'), 'utf8'));
  const withRef = man.joints.filter((j) => j.ref !== undefined);
  assert.equal(withRef.length, man.joints.length,
    'every joint needs an explicit ref (0 for the ones assembled at zero) — a ' +
    'missing field is indistinguishable from "not regenerated yet"');
  const nz = man.joints.filter((j) => Math.abs(j.ref) > 1e-9);
  // Assert by IDENTITY, not by a bare count: a count alone stays green when one
  // group's refs vanish and another's appear. Two groups are assembled off zero:
  //   - the 14 ORCA finger/wrist joints (thumb chain to -0.733 rad), and
  //   - the 2 lift stages, assembled at mid-travel so the DEFAULT pose is
  //     half-raised (ref 0.15875 m each = 0.3175 m of the 0.635 m stroke).
  const hands = nz.filter((j) => !/^lift_/.test(j.name));
  const lift = nz.filter((j) => /^lift_/.test(j.name));
  assert.equal(hands.length, 14,
    `expected exactly 14 non-zero hand refs (the ORCA finger/wrist joints), got ` +
    `${hands.length}. If the MJCF genuinely changed, re-derive with ` +
    `ros/src/mabel_description/urdf_to_glb.py and update this number.`);
  assert.deepEqual(lift.map((j) => j.name).sort(), ['lift_lower', 'lift_upper'],
    'both lift stages must carry a ref — the GLB is baked half-raised, so a ' +
    'client that ignores it renders the lift 0.3175 m off');
  for (const j of lift) {
    assert.ok(Math.abs(j.ref - 0.15875) < 1e-6,
      `lift ref must be half the 0.3175 m per-stage stroke, got ${j.ref}`);
  }
  // the thumb chain is the one that shows: guard its magnitude
  const thumb = man.joints.filter((j) => /thumb/.test(j.node) && Math.abs(j.ref) > 0.4);
  assert.ok(thumb.length >= 4,
    'the thumb chain carries the largest refs (to −0.733 rad); losing them is ' +
    'what hyperextends the last thumb joint');
});

test('the generator emits ref, so a regen cannot silently drop it', () => {
  const gen = readFileSync(
    join(repo, 'ros', 'src', 'mabel_description', 'urdf_to_glb.py'), 'utf8');
  assert.match(gen, /'ref':\s*JOINT_REF\.get\(name, 0\.0\)/,
    'urdf_to_glb.py must stamp ref into mabel_joints.json');
  assert.match(gen, /qpos0\[m\.jnt_qposadr\[j\]\]/,
    'refs come from the MJCF qpos0 — the same source glb_to_rig.py uses');
});
