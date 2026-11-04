// The Home / rest pose and its glide math — pure, node-testable.
import { refFor, onRobotChange } from './handmodel.js';
//
// REST_POSE is the minimal-holding-torque posture derived from the MJCF as the
// arms' gravity equilibrium: both arms hang at the sides with the elbows
// EXTENDED, torso upright and neck level, swerve wheels facing forward, hands in
// a gentle relaxed curl, and the lift column at half travel. Re-derived 2026-09
// by controller/experiments/exp_rest_pose.py, which minimises Στ² under
// mj_inverse over the symmetric (shoulder-roll, elbow) pair:
//
//   qpos zero        Σ|τ| = 38.587 Nm    elbow bend 91.3°  (a 90° bent elbow!)
//   REST_POSE        Σ|τ| =  1.638 Nm    elbow bend 15.1°  (the linkage's floor)
//
// It is SYMMETRIC, and arm_4 is ±1.52, not ±1.542: the elbows are not mirrored
// in the model, so the mirror of the right arm's optimum is not a legal left-arm
// value, and 1.542 IS the left limit. ±1.52 keeps 0.022 rad of margin on both
// sides. arm_3/5/6/7 are gravity-FLAT once the arm hangs (they rotate about the
// hanging axis), so they are pinned NEUTRAL rather than optimised.
//
// Values are ABSOLUTE joint units (what the wire commands use) and match `REST`
// in motions.js (minus the steer joints) and `RobotController.restBodyPose` +
// `homeGrip` in the iOS and Vision Pro apps — ONE home pose across every
// client. The arms/torso/neck/steer joints have ref 0, so those numbers are also
// the neutral displacement; the two WRISTS carry ref ±0.15 on MABEL (their
// neutral qpos), so a LEVEL hand is commanded by the ref value, not 0 — sending
// 0 would twist the wrist by −ref.
// NOT frozen: the two wrist values are rebuilt in place when the robot changes (below).
export const REST_POSE = {
  left_arm_1: 0, left_arm_2: -0.1711, left_arm_3: 0, left_arm_4: 1.52,
  left_arm_5: 0, left_arm_6: 0, left_arm_7: 0,
  right_arm_1: 0, right_arm_2: 0.1711, right_arm_3: 0, right_arm_4: -1.52,
  right_arm_5: 0, right_arm_6: 0, right_arm_7: 0,
  left_wrist: 0.15, right_wrist: -0.15,          // a LEVEL hand — per robot, see refreshWrists
  torso: 0,                                      // UPRIGHT: chest vertical, head on the horizon
  // NECK LEVEL — horizon. All three neck joints have symmetric MJCF ranges about
  // 0 and ref 0, so 0 IS the horizon: eyes level, no yaw, no head tilt. (MJCF
  // sign convention, for anyone tempted to "correct" these: +pitch = look DOWN,
  // +yaw = look LEFT, +roll = tilt toward the robot's LEFT ear.) Home must
  // level the head — a parked robot staring at the floor or off to one side
  // reads as still-being-driven.
  neck_yaw: 0, neck_pitch: 0, neck_roll: 0,
  fl_steer: 0, fr_steer: 0, b_steer: 0,
  // HANDS: the same gentle curl every clip rests at (motions.js REST_GRIP) and the apps'
  // `homeGrip` — ONE rest hand across every client. Driven through the grip scalar, never
  // per-finger: raw finger targets fight the gateway's grasp expansion.
  left_grip: 0.2, right_grip: 0.2,
  // THE COLUMN IS PART OF HOME. Half travel — the same SAFE_HOME_LIFT the animation
  // library parks at (motions.js HOME_JOINTS) and the qpos the rig is baked at (ref
  // 0.15875, mid-stroke), so Home leaves the column centred and off both hard stops.
  // Omitting it did not merely leave the lift where it was: motions.js measured the
  // consequence — arms hanging from a torso 31.75 cm lower put the hands 12.8 cm inside
  // the chassis mesh. The two wire keys SUM into the one actuator, so each carries half.
  lift_lower: 0.15875, lift_upper: 0.15875,
};
// A level hand is the v1 MJCF ref (±0.15, mirrored) on MABEL and 0 on NOVA's URDF-frame
// wire; handmodel.js knows which robot the gateway said hello as.
function refreshWrists(){
  REST_POSE.left_wrist = refFor('wrist', 'left');
  REST_POSE.right_wrist = refFor('wrist', 'right');
}
refreshWrists();
onRobotChange(refreshWrists);

// The lift is the one joint here that is not an angle, and every test below is in
// radians. Weight its metres so a metre of column costs the same velocity budget as
// 28.57 rad: 1 / LIFT_MAX_VEL (0.035 m/s, measured on this column) x HOME_VMAX (1 rad/s).
// That makes homeDuration() stretch a long column move instead of commanding a speed the
// lift cannot reach, and makes HOME_EPS mean 0.7 mm per stage — 1.4 mm of column — rather than a
// meaningless 50 mm. (2.5x tighter than the 1.75 mm this note used to claim: LIFT_WEIGHT
// carries the factor 2 because the two wire keys SUM into one actuator, so a per-stage
// error is twice as much column as it looks. Measured against the code 2026-09-24.) Deliberately NOT imported from motions.js, which imports THIS file — keep the two
// in step by hand (motions.js LIFT_RAD_PER_M).
// 0.028 m/s, NOT the measured 0.035 ceiling — the glide needs headroom for two reasons.
// smoothstep's PEAK velocity is 1.5x its mean, and homeDuration's 1.5x factor cancels
// exactly that, so budgeting at 0.035 commands a peak of exactly 0.035 with nothing in
// hand. On top of that the firmware TAPERS effort over the last 25 mm to stop the column
// slamming into its stop band, so the final approach is deliberately slower than the
// ceiling. Budgeting at 80% of the ceiling lets the column actually follow the glide
// instead of falling behind it and then catching up in one lurch after the stream ends
// (measured: 20.4 mm of lag with a 0.035 budget).
export const LIFT_RAD_PER_M = 1 / 0.028;                       // 35.7 rad-equivalents per metre
// ...times TWO, because these tests see one STAGE and the column moves the SUM of both:
// a stage metre is two column metres, so weighting by LIFT_RAD_PER_M alone sized the glide
// for half the real travel and commanded 70 mm/s — exactly double the limit.
const LIFT_WEIGHT = 2 * LIFT_RAD_PER_M;                        // 71.43 per stage-metre

// THE COLUMN IS NOT STREAMED. It is a BANG-BANG servo: its only speeds are ~37 mm/s and
// zero, so a trajectory asking for anything slower is tracked by stop-starting — 54 stops
// on a full-stroke Home, each one a fresh full-duty pulse and a current spike, which is
// what "the lift moves in jittery motion" is. Home therefore sends the column ONE absolute
// target and lets the firmware run its own profile and taper (4 stops instead of 124 across
// four presses, in a lift simulation). The arms still glide — they are PD-tracked
// and a streamed trajectory is exactly right for them.
export const LIFT_JOINTS = ['lift_lower', 'lift_upper'];
const isLift = (k) => LIFT_JOINTS.includes(k);
const WEIGHT = (k) => (k === 'lift_lower' || k === 'lift_upper' ? LIFT_WEIGHT : 1);

// Pressing Home while already home must be a NO-OP, not a re-glide. The old
// button switched control mode and re-streamed the pose unconditionally, and
// the mode switch reseeds every joint target to the MEASURED pose — so each
// press nudged the arms by whatever they had sagged and then let them sag
// again, which is the oscillation the operator saw. Anything inside this band
// counts as "already there"; it is well under the arms' own stiction floor.
export const HOME_EPS = 0.05;                    // rad (~2.9°)

// Largest per-joint distance from `target`, over the joints the robot actually
// reports. Joints absent from `live` are skipped rather than treated as 0 — a
// missing reading is "unknown", not "at zero", and assuming zero manufactures
// motion on joints (steer, neck, grips) that no one is streaming.
// Returns NaN when NOTHING in `target` could be compared. The per-joint rule below — a
// missing reading is "unknown", not "at zero" — has to hold for the whole pose too, and it
// did not: with no state at all the loop ran zero times and returned 0, which reads as
// "perfectly at home". So Home silently did nothing whenever the state feed had not arrived
// yet or the page had just reconnected, toasting "already home (within 0.0°)" (2026-09-24).
// An empty measurement is the one case where the answer must not be a number.
export function homeError(live, target = REST_POSE) {
  let err = 0, seen = 0;
  for (const k in target)
    if (live?.[k] != null) { err = Math.max(err, Math.abs(target[k] - live[k]) * WEIGHT(k)); seen++; }
  return seen ? err : NaN;
}
// Unknown is NOT at home: without a reading Home must run, not bail.
export const isAtHome = (live, target = REST_POSE) => {
  const e = homeError(live, target);
  return Number.isFinite(e) && e <= HOME_EPS;
};

// Peak per-joint speed during the glide, and the floor duration. The glide runs
// HOME_MIN_S unless a joint would exceed the cap, in which case it stretches — a
// big swing simply takes longer, it never moves faster.
//
// Home is a POSE INTERPOLATION and nothing more: the interpolated targets go out
// on the ordinary joint wire, so the deployed PD + gravity feed-forward and the
// operator's own gains do the tracking, exactly as they do for a slider. There
// is no Home-specific control mode, stiffness or arrival logic.
export const HOME_VMAX = 1.0;                    // rad/s
export const HOME_MIN_S = 2.0;                   // s — operator floor: "at least 2 seconds"

// smoothstep: zero velocity at both ends (no jerk in, no overshoot out)
export const smoothstep = (u) => {
  const t = u < 0 ? 0 : u > 1 ? 1 : u;
  return t * t * (3 - 2 * t);
};

// Glide duration for a given start pose: floored at HOME_MIN_S and stretched so
// the PEAK velocity (smoothstep peaks at 1.5× the average) stays under the cap.
// A joint with no start value defaults to the TARGET (= "already there", so it
// contributes no motion), never to 0 — starting an untracked joint from a
// fabricated 0 is a commanded sweep the operator never asked for.
export function homeDuration(start, target = REST_POSE) {
  let maxD = 0;
  for (const k in target) {
    if (isLift(k)) continue;              // not streamed — see LIFT_JOINTS
    maxD = Math.max(maxD, Math.abs(target[k] - (start?.[k] ?? target[k])) * WEIGHT(k));
  }
  return Math.max(HOME_MIN_S, 1.5 * maxD / HOME_VMAX);
}

/// The column's one-shot Home target — sent ONCE, at the start of the glide.
export const liftHome = (target = REST_POSE) =>
  Object.fromEntries(LIFT_JOINTS.filter((k) => target[k] != null).map((k) => [k, target[k]]));

// The commanded pose at normalized progress u ∈ [0,1] of the glide.
export function homePoseAt(start, u, target = REST_POSE) {
  const e = smoothstep(u);
  const out = {};
  for (const k in target) {
    if (isLift(k)) continue;               // the column is commanded once, not streamed
    const s = start?.[k] ?? target[k];      // unknown start ⇒ hold, don't sweep from 0
    out[k] = s + (target[k] - s) * e;
  }
  return out;
}
