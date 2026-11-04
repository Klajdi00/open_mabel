// The module vocabulary (web/js/modules.js) — the rail, the panel, the camera
// and the status dock all partition the robot through THIS list, so a joint
// that belongs to no module (or to two) is a silent hole in the control surface.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MODULES, MODULE_IDS, moduleById, moduleJoints, moduleOf, isHand,
         iconSvg } from '../../web/js/modules.js';

// the rig's real joint names (web_gui/shared_assets/mabel_joints.json order)
const RIG = [
  'fl_steer', 'fl_drive', 'fr_steer', 'fr_drive', 'b_steer', 'b_drive',
  'lift_lower', 'lift_upper', 'torso', 'neck_yaw', 'neck_pitch', 'neck_roll',
  ...Array.from({ length: 7 }, (_, i) => `left_arm_${i + 1}`),
  'left_wrist',
  'left_thumb_mcp', 'left_thumb_abd', 'left_thumb_pip', 'left_thumb_dip',
  'left_index_abd', 'left_index_mcp', 'left_index_pip',
  'left_middle_abd', 'left_middle_mcp', 'left_middle_pip',
  'left_ring_abd', 'left_ring_mcp', 'left_ring_pip',
  'left_pinky_abd', 'left_pinky_mcp', 'left_pinky_pip',
  ...Array.from({ length: 7 }, (_, i) => `right_arm_${i + 1}`),
  'right_wrist',
  'right_thumb_mcp', 'right_thumb_abd', 'right_thumb_pip', 'right_thumb_dip',
  'right_index_abd', 'right_index_mcp', 'right_index_pip',
  'right_middle_abd', 'right_middle_mcp', 'right_middle_pip',
  'right_ring_abd', 'right_ring_mcp', 'right_ring_pip',
  'right_pinky_abd', 'right_pinky_mcp', 'right_pinky_pip',
];

test('the rail is the iOS ModuleFocus list, in order', () => {
  assert.deepEqual(MODULE_IDS,
    ['arm_left', 'arm_right', 'neck', 'hand_left', 'hand_right', 'lift', 'torso']);
});

test('every module claims the joints it should — and only those', () => {
  assert.deepEqual(moduleJoints('arm_left', RIG),
    Array.from({ length: 7 }, (_, i) => `left_arm_${i + 1}`));
  assert.deepEqual(moduleJoints('arm_right', RIG),
    Array.from({ length: 7 }, (_, i) => `right_arm_${i + 1}`));
  assert.deepEqual(moduleJoints('neck', RIG), ['neck_yaw', 'neck_pitch', 'neck_roll']);
  assert.deepEqual(moduleJoints('lift', RIG), ['lift_lower', 'lift_upper']);
  assert.deepEqual(moduleJoints('torso', RIG), ['torso']);
  // a hand is wrist + the 16 finger joints = the full ORCA 17
  assert.equal(moduleJoints('hand_left', RIG).length, 17);
  assert.equal(moduleJoints('hand_right', RIG).length, 17);
  assert.ok(moduleJoints('hand_left', RIG).includes('left_wrist'));
});

test('the hand modules never poach each other (side prefixes hold)', () => {
  for (const n of moduleJoints('hand_left', RIG)) assert.ok(n.startsWith('left_'), n);
  for (const n of moduleJoints('hand_right', RIG)) assert.ok(n.startsWith('right_'), n);
});

test('the modules PARTITION the robot: every joint lands in exactly one', () => {
  const claimed = new Map();
  for (const id of MODULE_IDS)
    for (const n of moduleJoints(id, RIG)){
      assert.ok(!claimed.has(n), `${n} claimed by ${claimed.get(n)} and ${id}`);
      claimed.set(n, id);
    }
  // everything but the swerve base is reachable from the rail — a joint with no
  // module would be a DOF the operator simply cannot select
  const orphans = RIG.filter(n => !claimed.has(n));
  assert.deepEqual(orphans,
    ['fl_steer', 'fl_drive', 'fr_steer', 'fr_drive', 'b_steer', 'b_drive'],
    'only the swerve base (which Drive space owns) may sit outside the rail');
  // 14 arm + 34 hand + 3 neck + 2 lift + 1 torso
  assert.equal(claimed.size, 54);
});

test('moduleOf inverts moduleJoints', () => {
  assert.equal(moduleOf('right_arm_4'), 'arm_right');
  assert.equal(moduleOf('left_index_pip'), 'hand_left');
  assert.equal(moduleOf('left_wrist'), 'hand_left');
  assert.equal(moduleOf('neck_roll'), 'neck');
  assert.equal(moduleOf('lift_upper'), 'lift');
  assert.equal(moduleOf('torso'), 'torso');
  assert.equal(moduleOf('fl_drive'), null);      // the base is Drive space's
});

test('hands are the two modules that render dexterity controls', () => {
  assert.deepEqual(MODULE_IDS.filter(isHand), ['hand_left', 'hand_right']);
});

test('each module frames a camera target that EXISTS on the rig', () => {
  for (const m of MODULES)
    assert.ok(RIG.includes(m.focus), `${m.id} focuses missing node ${m.focus}`);
});

test('sided modules carry a side; the midline ones do not', () => {
  assert.equal(moduleById('arm_left').side, 'left');
  assert.equal(moduleById('hand_right').side, 'right');
  for (const id of ['neck', 'lift', 'torso'])
    assert.equal(moduleById(id).side, null);
});

test('the id → twin swap the rail uses for "[ ]" stays inside the vocabulary', () => {
  // app.js flips the active side with id.replace(side, other) — it must land
  // on a real module, or the keyboard and the rail would silently diverge
  for (const m of MODULES.filter(x => x.side)){
    const other = m.side === 'left' ? 'right' : 'left';
    assert.ok(moduleById(m.id.replace(m.side, other)), `${m.id} has no ${other} twin`);
  }
});

test('every module renders one icon, and only the left twins mirror', () => {
  for (const m of MODULES){
    const svg = iconSvg(m);
    assert.match(svg, /^<svg /);
    assert.ok(svg.includes('stroke="currentColor"'));
    assert.equal(svg.includes('scale(-1,1)'), m.mirror);
  }
  assert.deepEqual(MODULES.filter(m => m.mirror).map(m => m.id),
    ['arm_left', 'hand_left']);
});

test('an unknown module is null, not a crash', () => {
  assert.equal(moduleById('nope'), null);
  assert.deepEqual(moduleJoints('nope', RIG), []);
});
