// Home / rest pose: iOS-parity values, joint-range legality, and a glide that
// is monotone, speed-capped, and lands exactly on the pose.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { REST_POSE, HOME_VMAX, HOME_MIN_S, HOME_EPS, LIFT_JOINTS, homeDuration,
         homePoseAt, homeError, isAtHome, liftHome, smoothstep } from '../../web/js/homepose.js';
import { RANGES } from '../../web/js/motions.js';

// joints outside RANGES that the rest pose legitimately commands (server-side
// registry: control_center/server/joints.py MODULES)
const EXTRA_JOINTS = ['neck_yaw', 'neck_pitch', 'neck_roll',
                      'fl_steer', 'fr_steer', 'b_steer'];

test('every rest-pose joint is a real, in-range robot joint', () => {
  for (const [j, v] of Object.entries(REST_POSE)){
    assert.ok(RANGES[j] || EXTRA_JOINTS.includes(j), `unknown joint ${j}`);
    const r = RANGES[j];
    if (r) assert.ok(v >= r[0] - 1e-9 && v <= r[1] + 1e-9,
      `${j}=${v} outside [${r[0]}, ${r[1]}]`);
  }
});

test('the arms hang DOWN and the wrists sit at their MJCF ref', () => {
  // elbow (arm_4) near fully folded, mirrored L/R — the hanging-arm signature
  assert.ok(REST_POSE.left_arm_4 > 1.5 && REST_POSE.right_arm_4 < -1.5);
  assert.ok(Math.abs(REST_POSE.left_arm_4 + REST_POSE.right_arm_4) < 0.01, 'mirrored');
  // shoulders essentially neutral (arms at the sides, not raised)
  assert.ok(Math.abs(REST_POSE.left_arm_1) < 0.05);
  assert.ok(Math.abs(REST_POSE.right_arm_1) < 0.05);
  // wrists at ref, NOT zero (0 would twist the hand by −ref)
  assert.equal(REST_POSE.left_wrist, 0.15);
  assert.equal(REST_POSE.right_wrist, -0.15);
  // torso / neck neutral, wheels forward
  for (const j of ['torso', 'neck_yaw', 'neck_pitch', 'neck_roll',
                   'fl_steer', 'fr_steer', 'b_steer'])
    assert.equal(REST_POSE[j], 0, `${j} must be neutral`);
});

test('duration is floored, and stretches so no joint outruns the cap', () => {
  assert.equal(homeDuration(REST_POSE), HOME_MIN_S, 'already home → the floor');
  // a big swing: 3 rad away on one joint ⇒ 1.5·3/1.0 = 4.5 s
  const far = { ...REST_POSE, left_arm_2: REST_POSE.left_arm_2 + 3.0 };
  assert.ok(Math.abs(homeDuration(far) - 4.5) < 1e-9);
});

test('the glide is monotone, speed-capped, and lands ON the pose', () => {
  const start = { left_arm_4: -0.8, right_arm_4: 0.8, left_arm_2: 2.9,
                  torso: -0.9, left_wrist: -1.0, left_grip: 1 };
  const dur = homeDuration(start);
  const DT = 1 / 30;
  let prev = homePoseAt(start, 0);
  // an unreported joint starts AT its target (holds still), never at a fabricated 0
  assert.ok(Object.entries(prev).every(([k, v]) =>
    Math.abs(v - (start[k] ?? REST_POSE[k])) < 1e-9), 'u=0 == the live pose (no snap)');
  for (let t = DT; t <= dur + 1e-9; t += DT){
    const p = homePoseAt(start, t / dur);
    for (const k in p){
      const vel = Math.abs(p[k] - prev[k]) / DT;
      assert.ok(vel <= HOME_VMAX + 1e-6, `${k} ${vel.toFixed(3)} rad/s over cap`);
      // monotone: never overshoots past the target and comes back
      const s = start[k] ?? REST_POSE[k], tgt = REST_POSE[k];
      const lo = Math.min(s, tgt) - 1e-9, hi = Math.max(s, tgt) + 1e-9;
      assert.ok(p[k] >= lo && p[k] <= hi, `${k} overshot`);
    }
    prev = p;
  }
  const end = homePoseAt(start, 1);
  for (const k in REST_POSE){
    if (LIFT_JOINTS.includes(k)){ assert.equal(end[k], undefined, `${k} must not be streamed`); continue; }
    assert.ok(Math.abs(end[k] - REST_POSE[k]) < 1e-9, `${k} did not land`);
  }
});

test('the lift column is part of Home but goes ONCE, as one absolute target', () => {
  const lh = liftHome();
  assert.deepEqual(Object.keys(lh).sort(), [...LIFT_JOINTS].sort());
  for (const k of LIFT_JOINTS) assert.equal(lh[k], REST_POSE[k]);
  // half travel per stage (the two wire keys sum into one actuator)
  assert.ok(Math.abs(REST_POSE.lift_lower - 0.15875) < 1e-9);
  assert.ok(Math.abs(REST_POSE.lift_upper - 0.15875) < 1e-9);
  // a far-off column does not stretch the ARM glide: it is not streamed
  assert.equal(homeDuration({ ...REST_POSE, lift_lower: 0, lift_upper: 0 }), HOME_MIN_S);
});

test('already home is a no-op; NO measurement is never "at home"', () => {
  assert.ok(isAtHome(REST_POSE));
  assert.ok(isAtHome({ ...REST_POSE, left_arm_2: REST_POSE.left_arm_2 + 0.5 * HOME_EPS }));
  assert.ok(!isAtHome({ ...REST_POSE, left_arm_2: REST_POSE.left_arm_2 + 2 * HOME_EPS }));
  // an empty / missing state must make Home RUN, not toast "already home"
  assert.ok(Number.isNaN(homeError({})));
  assert.ok(!isAtHome({}));
  assert.ok(!isAtHome(undefined));
  // the column is weighted: 2 mm on one stage is already outside the band
  assert.ok(!isAtHome({ ...REST_POSE, lift_lower: REST_POSE.lift_lower + 0.002 }));
});

test('smoothstep starts and ends at zero velocity', () => {
  const d = 1e-4;
  assert.ok(smoothstep(d) / d < 0.01, 'zero velocity in');
  assert.ok((1 - smoothstep(1 - d)) / d < 0.01, 'zero velocity out');
  assert.equal(smoothstep(-5), 0);
  assert.equal(smoothstep(5), 1);
});
