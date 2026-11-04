// Motion / action registry + choreography track model for the timeline.
// Pure data + math (node-testable, no DOM): the clip library (validated
// whole-robot animation clips, scripted dummy task policies, parametric nav
// primitives, static poses, and greyed-out coming-soon learned-policy slots),
// plus the ChoreoTrack block model (NLE cut/trim/move/snap) and the
// location-aware gap validator.
//
// Conventions (measured against the LIVE sim, 2026-07-02 probe sessions):
//   · WIRE FRAME: joint commands are ABSOLUTE MuJoCo qpos, NOT ref-relative.
//     (web_gui/shared_assets/mabel_joints.json publishes qpos−ref ranges for the 3-D
//     clients; RANGES below is the MJCF's own jnt_range, which is what a
//     command is checked against.)
//   · qpos ZERO is NOT the robot's rest: at qpos 0 the upper arm hangs but the
//     forearm juts forward — a 90° elbow (measured 91.3°) costing 38.6 Nm of
//     holding torque. REST (below) is the rest pose; every clip starts and ends
//     there, and `pose_home` IS it.
//   · {side}_arm_1 shoulder pitch: left − = raise up, + = lower to the side
//     (right arm mirrored: + = up, − = down; all arm joints mirror by sign).
//   · {side}_arm_4 elbow: left + EXTENDS the forearm down toward the thigh
//     (right −); +1.52 is straight, 0 is the 90° bend.
//   · {side}_arm_6 wrist pitch (+ fingers down), _arm_5/_arm_7 rolls.
//   · {side}_grip ∈ [0,1] curls the whole ORCA hand (0 open → 1 fist).
//   · torso ∈ [−1.047, +0.262] (− = deep lean), lift_lower+lift_upper sum
//     into the Lift actuator (each ∈ [0, 0.3175], assembled/ref at 0.15875).
//   · NECK yaw/pitch/roll ARE drivable via joint_command (the VP joint-space
//     neck pad and server/tools/exp_neck_stomp.py both do it); they are only
//     overridden while a LIVE head block is streaming, which choreography is
//     not. Clips use them.
//   · navJoystick wire signs (verified): ly=+forward (0.8 m/s per unit),
//     lx=−(right strafe), rx=−(right/CW turn, 1.5 rad/s per unit), ry=+lift.
//   · velocity budget: |joint velocity| < 2.5 rad/s at 20 Hz sampling.

// ── canonical joint ranges ───────────────────────────────────────
// Read off the MJCF (simulation/mabel_mujoco/models/mabel_full.xml) rather than
// mabel_joints.json, because these bound a COMMAND and commands are absolute
// qpos while that manifest publishes qpos−ref. They differ for the two wrists
// (ref ±0.15) and the two lift stages (ref 0.15875).
//
// KNOWN MODEL DEFECT, left as-is here because the MJCF is the source of truth:
// the elbows are not mirrored. Right Arm 4 is [−1.670, 0.773]; the mirror would
// make Left Arm 4 [−0.773, 1.670], but the MJCF declares [−0.943, 1.542]. The
// left arm therefore extends 0.128 rad (7.3°) LESS than the right. REST backs
// off the tighter of the two, symmetrically, so neither arm sits on a stop.
export const RANGES = {
  torso: [-1.047, 0.262],
  neck_yaw: [-1.571, 1.571], neck_pitch: [-0.785, 0.785],
  neck_roll: [-0.785, 0.785],
  lift_lower: [0, 0.3175], lift_upper: [0, 0.3175],
  left_arm_1: [-3.491, 1.396], left_arm_2: [-0.175, 3.316],
  left_arm_3: [-1.571, 1.571], left_arm_4: [-0.943, 1.542],
  left_arm_5: [-1.571, 1.571], left_arm_6: [-0.785, 0.785],
  left_arm_7: [-1.571, 1.571], left_wrist: [-0.897, 0.674],
  right_arm_1: [-1.396, 3.491], right_arm_2: [-3.316, 0.175],
  right_arm_3: [-1.571, 1.571], right_arm_4: [-1.670, 0.773],
  right_arm_5: [-1.571, 1.571], right_arm_6: [-0.785, 0.785],
  right_arm_7: [-1.571, 1.571], right_wrist: [-0.674, 0.897],
  left_grip: [0, 1], right_grip: [0, 1],   // whole-hand curl (bridge apply_grasp)
};

// ── REST — the pose the robot sits in, derived from the MJCF ──────────────
// The arms hang at the sides with the elbows EXTENDED, which is where gravity
// puts them. Solved in controller/experiments/exp_rest_pose.py by minimising
// the gravity holding torque Στ² (mj_inverse, qvel = qacc = 0) over the
// symmetric (shoulder-roll, elbow) pair with the shoulder pitch pinned at 0 so
// that every clip's authored `*_arm_1` amplitude still means what it meant:
//
//   qpos zero (the old NEUTRAL)  Σ|τ| = 38.587 Nm   elbow bend 91.3°
//   REST                         Σ|τ| =  1.638 Nm   elbow bend 15.1°
//
// 15.1° is the mechanical floor, not a choice: the forearm is not collinear
// with the upper arm at the elbow's end-stop. arm_4 = ±1.52 keeps 0.022 rad of
// margin off the tighter (left) stop at 1.542; pushing to the stop itself saves
// another 0.7 Nm and buys a servo grinding against a limit, which is a bad
// trade. The hands drop 0.370 m below the shoulders and sit 0.105 m behind /
// 0.148 m outboard of them — clear of the torso, clear of the thighs.
//
// The two WRISTS and the two LIFT stages carry a non-zero MJCF ref, and the
// wire is absolute, so their REST value is that ref — NOT zero. Commanding 0
// twists each wrist by −ref and slams the lift to the floor.
export const REST = Object.freeze({
  ...Object.fromEntries(Object.keys(RANGES).map((k) => [k, 0])),
  left_arm_2: -0.1711, left_arm_4: 1.52,       // ← elbow extended, arm hanging
  right_arm_2: 0.1711, right_arm_4: -1.52,
  left_wrist: 0.15, right_wrist: -0.15,        // MJCF ref ⇒ a level hand
  lift_lower: 0.15875, lift_upper: 0.15875,    // MJCF ref ⇒ half-extended lift
  left_grip: 0.2, right_grip: 0.2,             // a hand at rest is not splayed
});
// REST must stay byte-identical to `REST_POSE` in homepose.js (which adds the
// three steer joints) and to `RobotController.restBodyPose` + `homeGrip` in
// both apps — ONE rest pose across every client. homepose.test.mjs pins it.

// Every animation clip starts and ends at REST; static poses transition FROM
// REST and hold. NEUTRAL is kept as the name the rest of the studio imports.
export const NEUTRAL = REST;

export const MAX_JOINT_VEL = 2.5;          // rad/s budget for the tests
export const GRIP_SCALE = 1.9;             // rad of finger travel per grip unit

// nav calibration (measured live): m/s per unit stick, rad/s per unit stick.
// Forward and reverse are symmetric (0.80 m/s per unit, re-measured in open
// floor 2026-07-02 — an earlier 0.32 estimate was an artifact of backing into
// the kitchen island 0.2 m behind the home spawn).
export const NAV_SPEED = 0.8;
export const NAV_YAW_RATE = 1.5;

// ── easing toolkit ──────────────────────────────────────────────────────────
const clamp01 = (u) => Math.max(0, Math.min(1, u));
export const ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * clamp01(u));

// Blend sizing. EASE_PEAK is the peak/average velocity ratio of `ease` ABOVE —
// a raised cosine, so π/2 ≈ 1.571 (NOT the 1.5 of the cubic smoothstep that
// homepose.js's Home glide uses). Sizing a blend with 1.5 runs the peak ~5%
// over the cap. Get this constant from the easing, never from memory.
// Shared with the real studio's motions.js — the task-space wrist plan eases
// with the same budget in sim and on hardware, so what you rehearse here is
// what the robot does.
export const SAFE_BLEND_VEL = 1.0;         // rad/s peak during a blend
export const EASE_PEAK = Math.PI / 2;      // peak/avg velocity of `ease` above
// The executor's command rate. It matters to the guard, not just to the baker:
// the robot does not receive a raised cosine, it receives 20 samples a second
// OF one, and the biggest step between two of those samples is what a servo
// answers. For a long bridge that tends to EASE_PEAK/T; for a short one it is
// meaningfully higher, and sizing against the continuous peak alone let a
// squeezed bridge past the hard cap by 1% (measured: lift_up -> wave, 2.52 of a
// 2.5 budget).
export const EXEC_RATE = 20;

// A jump small enough that ONE tick of it is still inside the velocity budget
// needs no bridge — the servo absorbs it, and inserting sub-0.4 s transitions
// for every rounding-level difference would clutter the timeline without
// improving safety. That makes the threshold DERIVED, not chosen:
//
//     BLEND_THRESH = MAX_JOINT_VEL / EXEC_RATE = 0.125 rad
//
// It used to be 0.15, picked for signal-to-noise, and those two numbers
// disagreed: 0.15 rad in one 50 ms tick is 3.0 rad/s, so a jump sitting exactly
// on the threshold was waved through at 1.2x the budget it was supposed to
// protect. `pose_tuck` moves a shoulder by exactly 0.15 rad and found it.
export const BLEND_THRESH = MAX_JOINT_VEL / EXEC_RATE;
// Floor on a blend's duration. Documented in control_center/README.md as 0.4 s
// and asserted by tests/js/posejump.test.mjs, but never actually DECLARED here:
// blendMinDur() referenced it and threw a ReferenceError on every call, so the
// pose-jump guard was dead code rather than a cap.
export const BLEND_MIN_DUR = 0.4;
/// Peak per-tick rate of the eased interpolation, per unit of pose distance.
export function easedPeakRate(dur) {
  const n = Math.max(1, Math.round(dur * EXEC_RATE));
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const d = ease((i + 1) / n) - ease(i / n);
    if (d > worst) worst = d;
  }
  return worst * EXEC_RATE;
}
// Ceiling on a derived bridge. Past this it is not a transition any more,
// it is the timeline doing something the operator did not ask for.
export const BLEND_MAX_DUR = 8.0;          // s — a safety floor, never rounded down
// eased 0→1 ramp across [t0,t1]
export const seg = (t, t0, t1) => ease((t - t0) / Math.max(1e-6, t1 - t0));
// eased 0→1→0 window: up over [t0,t0+r], down over [t1−r,t1]
export const win = (t, t0, t1, r = 0.5) => {
  if (t <= t0 || t >= t1) return 0;
  const up = seg(t, t0, t0 + r), dn = 1 - seg(t, t1 - r, t1);
  return Math.min(up, dn);
};
// windowed sine oscillation (starts/ends at zero amplitude)
export const osc = (t, f, t0, t1, r = 0.5) =>
  win(t, t0, t1, r) * Math.sin(2 * Math.PI * f * (t - t0));
// subtle breathing (always-on secondary motion inside a window)
export const breath = (t, t0, t1, f = 0.28) =>
  win(t, t0, t1, 0.8) * Math.sin(2 * Math.PI * f * (t - t0));

// Radian-equivalents for the two joints that are not radians: a grip unit is
// GRIP_SCALE rad of finger travel, a metre of lift is LIFT_RAD_PER_M. Used
// wherever a pose DISTANCE has to be one number — the pose-jump guard, and the
// REST anchor's ramp.
// MEASURED 2026-09-17 on the real column (MABEL and NOVA run the SAME lift). Full
// 635 mm stroke at 100 % duty, open loop, 2 trials per direction —
//   UP    steady 37.0 mm/s (36.8, 37.3), peak 41.8
//   DOWN  steady 36.4 mm/s (35.0, 37.7), peak 42.0
// down/up = 0.98, i.e. SYMMETRIC: the leadscrew is not back-drivable. A full stroke
// takes ~18.3 s. The robot's hardest actuation constraint.
export const LIFT_MAX_VEL = 0.035;        // m/s — design limit, under the slower direction
export const LIFT_STROKE_TIME = 0.635 / LIFT_MAX_VEL;   // s — 18.1 s end to end

// Clamp a lift step to what the column can actually travel. For anything that
// STREAMS lift commands; the firmware enforces the same ceiling independently.
export function liftRateLimit(prev, next, dt, vmax = LIFT_MAX_VEL) {
  if (prev == null || !(dt > 0)) return next;
  const budget = vmax * dt;
  const d = next - prev;
  return Math.abs(d) <= budget ? next : prev + Math.sign(d) * budget;
}

// Was a hand-picked 10.0, which let the blend machinery permit 100 mm/s — nearly 3x
// what the column can do. SAFE_BLEND_VEL / LIFT_MAX_VEL lines its velocity budget up
// with the measured rate instead.
export const LIFT_RAD_PER_M = SAFE_BLEND_VEL / LIFT_MAX_VEL;   // 28.6 rad/m
export const BLEND_SCALE = {
  left_grip: GRIP_SCALE, right_grip: GRIP_SCALE,
  lift_lower: LIFT_RAD_PER_M, lift_upper: LIFT_RAD_PER_M,
};
// worst-joint distance between two poses, in those radian-equivalents
export const poseGap = (a, b) => {
  let far = 0;
  for (const k in b) {
    const g = Math.abs(b[k] - (a[k] ?? 0)) * (BLEND_SCALE[k] || 1);
    if (g > far) far = g;
  }
  return far;
};

// ────────────────────────────────────────────────────────────────────────
// THE REST ANCHOR — how a clip authored in the qpos-zero frame plays from REST
//
// Every clip is authored as absolute joint values whose envelopes vanish at the
// clip edges; the frame those envelopes vanish INTO is qpos zero. REST is not
// qpos zero (the elbows sit 1.52 rad away), so a clip played raw would command
// that whole delta on its first 50 ms tick — 30 rad/s, 12× the budget.
//
// Rather than re-author every silhouette (and re-validate each against the sim),
// the registry plays each clip INSIDE an anchor that interpolates out of REST
// and back into it:
//
//     out_k(t) = REST_k + (J_k(t) − REST_k) · a(t)
//
// a = 0 at the edges ⇒ out = REST exactly, whatever the clip returned there;
// a = 1 across the body ⇒ out = J(t) exactly, so every validated pose is
// reproduced unchanged. `hold` clips (static poses, the lift, `pickup`'s carry)
// ramp a up and KEEP it at 1 — holding the final pose is the whole point of the
// flag, so their anchor must not fade back.
//
// The ramp is DERIVED, not tuned: it is sized from the longest journey the
// anchor has to make so that the anchor on its own can never break the budget.
// κ = EASE_PEAK is the raised cosine's peak/average ratio, the same constant
// the pose-jump guard sizes a blend with, and ANCHOR_VEL is the same 1.0 rad/s
// ceiling — because this IS that blend, applied automatically at a clip's own
// edges instead of between two blocks.
export const ANCHOR_VEL = SAFE_BLEND_VEL;        // rad/s the anchor may spend
export const ANCHOR_MIN_RAMP = 0.5;              // s — never snappier than this
const _anchorRamp = new WeakMap();               // clip → ramp (sampled once)

export function anchorRamp(clip, params) {
  if (_anchorRamp.has(clip)) return _anchorRamp.get(clip);
  const d = clipDur(clip, params);
  let far = 0;
  for (let i = 0; i <= 32; i++) {
    const gap = poseGap(REST, clip.joints((d * i) / 32, params || defaults(clip)) || {});
    if (gap > far) far = gap;
  }
  // A hold clip ramps once and may spend the whole block doing it; a windowed
  // clip must also ramp back, so it gets at most half.
  const room = clip.hold ? d : d / 2;
  const r = Math.max(ANCHOR_MIN_RAMP,
                     Math.min(room, (far * EASE_PEAK) / ANCHOR_VEL));
  _anchorRamp.set(clip, r);
  return r;
}

// a(t) for one clip: ramp-and-hold when `hold`, otherwise an eased window.
export function anchorAt(clip, t, params) {
  const d = clipDur(clip, params), r = anchorRamp(clip, params);
  return clip.hold ? seg(t, 0, r) : win(t, 0, d, r);
}

const P = (over) => ({ ...NEUTRAL, ...over });
const Z4 = Object.freeze({ lx: 0, ly: 0, rx: 0, ry: 0 });

// Named finger shapes. `u` scales the whole shape 0→1 so a clip can curl into
// it on its own envelope; a pose just passes 1. These are explicit per-finger
// targets rather than a `{side}_grip` scalar because a grip curls the WHOLE
// hand — you cannot point or give a thumbs-up with one number.
const POINT_HAND = (s, u = 1) => ({
  [`${s}_thumb_mcp`]: 0.5 * u, [`${s}_thumb_pip`]: 0.9 * u,
  [`${s}_middle_mcp`]: 1.3 * u, [`${s}_middle_pip`]: 1.4 * u,
  [`${s}_ring_mcp`]: 1.3 * u, [`${s}_ring_pip`]: 1.4 * u,
  [`${s}_pinky_mcp`]: 1.3 * u, [`${s}_pinky_pip`]: 1.4 * u,
  [`${s}_index_mcp`]: 0.08 * u, [`${s}_index_pip`]: 0.05 * u,  // the pointer
});
const THUMB_UP_HAND = (s, u = 1) => ({
  [`${s}_index_mcp`]: 1.35 * u, [`${s}_index_pip`]: 1.4 * u,
  [`${s}_middle_mcp`]: 1.35 * u, [`${s}_middle_pip`]: 1.4 * u,
  [`${s}_ring_mcp`]: 1.35 * u, [`${s}_ring_pip`]: 1.4 * u,
  [`${s}_pinky_mcp`]: 1.35 * u, [`${s}_pinky_pip`]: 1.4 * u,
  [`${s}_thumb_mcp`]: -0.25 * u, [`${s}_thumb_pip`]: -0.2 * u,  // thumb out
  [`${s}_thumb_abd`]: -0.733 - 0.30 * u,   // stop is -1.0821
});

// ── fixed pose target sets ──────────────────────────────────────
// Absolute joint targets; P() fills everything unmentioned from REST, so a pose
// that does not name the elbows gets them EXTENDED, which is the point.
//
// Measured with the model, not eyeballed — every number below was chosen by
// running web_gui/control_center/tools/exp_pose_library.py, which reports each
// target's hand position relative to its shoulder (forward / outboard / up, in
// metres), the resulting elbow bend, the gravity holding torque, and any joint
// that lands outside the MJCF's limits or any self-collision it produces.
//
// Three facts from that probe that these numbers are built on:
//   · {side}_arm_1 sweeps the arm in the SAGITTAL plane with the elbow
//     extended: left 0 = hanging, −1.571 = straight forward, −3.0 = overhead,
//     +1.0 = behind. (Right mirrors.) The old targets were authored when the
//     elbow rested at 90°, so e.g. armsUp's −1.75 pointed the FOREARM up while
//     the upper arm stayed forward; with a straight elbow that same number
//     points the whole arm forward, which is why these were re-derived.
//   · {side}_arm_2 abducts: left +1.571 puts the hand 0.367 m outboard (T-pose).
//   · neck_pitch + = look up, neck_yaw + = look to the robot's LEFT.
//
// MODEL DEFECT the `floor`/`crouch` targets work around: the MJCF's lift column
// self-penetrates the chassis below 0.155 m per stage — 7 contacts at 0.150,
// 114 at 0. Only the top half of the declared [0, 0.3175] stroke is usable in
// simulation. These targets stop at 0.17; the `lift_down` PRIMITIVE still asks
// for 0, because that is a real capability of the real lift and clamping it
// here would quietly take it away.
export const POSE_TARGETS = {
  // the rest pose itself — `pose_home` is a no-op once you are already home
  rest:   P({}),
  tall:   P({ lift_lower: 0.3175, lift_upper: 0.3175, torso: 0.15,
              neck_pitch: 0.15 }),                       // head 1.609 m
  bow:    P({ torso: -0.85, neck_pitch: -0.3 }),         // hands 0.787 m, arms hang
  armsUp: P({ left_arm_1: -3.0, right_arm_1: 3.0,        // hands +0.358 m over
              left_arm_2: 0.12, right_arm_2: -0.12,      //   the shoulders
              neck_pitch: 0.35, torso: 0.08 }),
  tpose:  P({ left_arm_1: 0, right_arm_1: 0,             // hands 0.367 m outboard,
              left_arm_2: 1.571, right_arm_2: -1.571 }), //   level, elbows straight
  reach:  P({ left_arm_1: -1.571, right_arm_1: 1.571,    // hands 0.360 m forward,
              left_arm_2: 0, right_arm_2: 0,             //   level
              left_grip: 0.05, right_grip: 0.05 }),
  ready:  P({ left_arm_1: -0.55, right_arm_1: 0.55,      // athletic: hands fwd
              left_arm_2: 0.35, right_arm_2: -0.35,      //   0.220, out 0.146,
              left_arm_4: 1.25, right_arm_4: -1.25,      //   down 0.252
              torso: -0.08, left_grip: 0.15, right_grip: 0.15 }),
  carry:  P({ left_arm_1: 0, right_arm_1: 0,             // tray: forearms level
              left_arm_2: 0.25, right_arm_2: -0.25,      //   forward, elbows 91°
              left_arm_4: 0, right_arm_4: 0,
              left_grip: 0.1, right_grip: 0.1 }),
  hips:   P({ left_arm_1: -0.2, right_arm_1: 0.2,        // hands on the hips:
              left_arm_2: 0.8, right_arm_2: -0.8,        //   out 0.285, elbows 66°
              left_arm_3: -1.0, right_arm_3: 1.0,
              left_arm_4: 0.45, right_arm_4: -0.45,
              left_grip: 0.35, right_grip: 0.35 }),
  wide:   P({ left_arm_1: -1.2, right_arm_1: 1.2,        // open/hug: fwd 0.244,
              left_arm_2: 0.9, right_arm_2: -0.9,        //   out 0.275, level
              left_arm_4: 1.35, right_arm_4: -1.35,
              left_grip: 0.05, right_grip: 0.05, torso: 0.06 }),
  shelf:  P({ lift_lower: 0.3175, lift_upper: 0.3175,    // hands 1.815 m — the
              left_arm_1: -2.5, right_arm_1: 2.5,        //   highest reach that
              left_arm_2: 0.15, right_arm_2: -0.15,      //   keeps both in view
              neck_pitch: 0.3 }),
  floor:  P({ torso: -0.6, lift_lower: 0.17, lift_upper: 0.17,
              left_arm_1: -0.35, right_arm_1: 0.35,      // hands 0.685 m
              left_arm_4: 1.45, right_arm_4: -1.45,
              neck_pitch: -0.4 }),
  point:  P({ right_arm_1: 1.571, right_arm_2: 0,        // right arm level fwd,
              neck_yaw: -0.25, ...POINT_HAND('right') }), //  index extended
  present: P({ right_arm_1: 1.15, right_arm_2: -0.6,     // offered to the side:
               right_arm_4: -1.15, right_arm_5: -0.9,    //   fwd 0.308, out 0.195
               right_grip: 0.05, torso: -0.05 }),
  guard:  P({ left_arm_1: -0.25, right_arm_1: 0.25,      // forearms crossed up in
              left_arm_2: 0.45, right_arm_2: -0.45,      //   front, fists closed
              left_arm_3: -0.9, right_arm_3: 0.9,
              left_arm_4: 0.25, right_arm_4: -0.25,
              left_grip: 0.8, right_grip: 0.8, torso: -0.12 }),
  lookUp:    P({ neck_pitch: 0.7, torso: 0.1 }),
  lookLeft:  P({ neck_yaw: 0.9 }),
  lookRight: P({ neck_yaw: -0.9 }),
  tuck:   P({ left_arm_1: 0.15, right_arm_1: -0.15,      // narrowest standing
              left_arm_2: -0.17, right_arm_2: 0.17 }),   //   profile, for doorways
  crouch: P({ lift_lower: 0.17, lift_upper: 0.17, torso: -0.25 }),
};

// A static pose clip is its target and nothing else: `hold` keeps it and the
// REST anchor supplies the eased transition. Because the target is CONSTANT,
// the block's duration can be DERIVED rather than guessed — the anchor needs
// κ·Δ/ANCHOR_VEL seconds to cross the gap, and POSE_TAIL is how long the pose
// is then actually held before the block ends. So `pose_tuck` is a short block
// and `pose_arms_up` a long one, which is simply how far each has to travel.
const POSE_TAIL = 1.2;
const poseDur = (target) => Math.max(
  1.2, Math.ceil(10 * (poseGap(REST, target) * EASE_PEAK / ANCHOR_VEL + POSE_TAIL)) / 10);
const poses = (spec) => Object.fromEntries(Object.entries(spec).map(
  ([key, [title, emoji, name]]) => [key, {
    title, emoji, dur: poseDur(POSE_TARGETS[name]), lane: 'manip',
    category: 'pose', status: 'available', hold: true,
    joints: () => POSE_TARGETS[name],
    nav: null, poseTarget: () => ({ ...POSE_TARGETS[name] }),
  }]));

// ─────────────────────────────────────────────────────────────────────────────
// THE REGISTRY
// category: 'animation' | 'policy' | 'nav' | 'pose'   (menu sections:
//   ANIMATIONS/POLICIES = animation+policy, NAV PRIMITIVES = nav,
//   STATIC POSES = pose)
// status: 'available' | 'coming_soon'  — only available actions are addable.
// lane: 'manip' | 'nav' | 'both' ('both' drops one linked block per lane).
// hold: block may be extended past dur; sampling clamps (pose holds).
// location {x,y,label}: the spot this action must run at (gap validator);
//   kind:'goal' blocks SET the tracked location from their params.
// ─────────────────────────────────────────────────────────────────────────────
export const MOTION_CLIPS = {

  // ═══ ANIMATIONS — expressive emotes (validated on the live sim) ═══════════
  wave: {
    title: 'Wave Hello', emoji: '👋', dur: 5.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 5.0;
      const up = win(t, 0.1, d - 0.1, 1.3);            // raise & lower envelope
      const wig = osc(t, 0.95, 1.2, d - 1.0, 0.6);     // the wave itself
      const brt = breath(t, 0, d);
      return P({
        // right arm up beside the head (upper arm 45° out, forearm vertical,
        // palm forward — pose-lab candidate E), elbow swings the wave
        right_arm_1: 2.0 * up,
        right_arm_2: -0.6 * up,
        right_arm_3: -0.9 * up,                         // fold plane → lateral wave
        right_arm_4: (-1.0 + 0.32 * wig) * up,          // elbow carries the wave
        right_arm_7: 0.4 * wig * up,                   // wrist adds flourish
        right_grip: 0.06 * up,                          // relaxed fingers
        // left arm drops to the side, relaxed — asymmetry sells it
        left_arm_1: 1.25 * up + 0.02 * brt,
        left_arm_4: 0.35 * up,
        torso: -0.06 * up + 0.015 * brt,                // tiny engaged lean
      });
    },
    nav: null,
  },

  handshake: {
    title: 'Handshake Offer', emoji: '🤝', dur: 6.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.0;
      const offer = win(t, 0.1, d - 0.1, 1.1);          // extend & retract
      const bob = osc(t, 0.9, 2.0, 4.4, 0.5);           // gentle "shall we?" bob
      const brt = breath(t, 0, d);
      return P({
        right_arm_1: 0.18 * offer + 0.06 * bob,         // forward, belly height
        right_arm_2: -0.15 * offer,
        right_arm_3: -0.9 * offer,                      // roll palm to thumb-up
        right_arm_4: 0.25 * offer,                      // forearm lifted to offer
        right_arm_6: 0.12 * offer,
        right_grip: 0.18 * offer,                       // soft open hand
        left_arm_1: 1.3 * offer + 0.02 * brt,           // other arm relaxed down
        left_arm_4: 0.3 * offer,
        torso: -0.1 * offer,                            // polite lean-in
      });
    },
    nav: null,
  },

  fistbump: {
    title: 'Fist Bump', emoji: '👊', dur: 4.5, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.5;
      const arm = win(t, 0.1, d - 0.1, 0.9);
      const wind = win(t, 0.7, 2.0, 0.55);                   // coil: fist to chest
      const punch = win(t, 1.8, 3.6, 0.55);                  // extend fwd, hold, back
      return P({
        right_arm_1: 0.45 * arm - 0.75 * punch,              // low rest → jab up fwd
        right_arm_2: -0.1 * arm,
        right_arm_4: -0.35 * wind,                           // slight coil, then straight
        right_arm_5: 0.7 * arm,                              // knuckles forward
        right_grip: Math.min(1, 1.15 * seg(t, 0.15, 1.9)) *
          (1 - seg(t, d - 1.6, d - 0.15)),                   // fist in, slow release
        left_arm_1: 1.28 * arm,                              // left arm chills
        left_arm_4: 0.32 * arm,
        torso: -0.05 * arm - 0.06 * punch,                   // weight into it
      });
    },
    nav: null,
  },

  heart: {
    title: 'Heart Hands', emoji: '💗', dur: 8.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 8.0;
      const up = win(t, 0.1, d - 0.1, 1.5);            // rise into the shape
      const hi = win(t, 3.4, d - 1.0, 1.1);            // …then lift it overhead
      const brt = breath(t, 1.6, d - 1.6);
      const s = up + 0.012 * brt;
      // face-level heart (J) → overhead heart (H): blend on `hi`
      const a1 = -1.9 * s - 0.1 * hi, a2 = (0.2 + 0.1 * hi) * s;
      const a3 = 1.4 * s * (1 - hi) + 0.0 * hi;         // J folds tips together…
      const a4 = (1.6 - 0.05 * hi) * s;                 // …H keeps the deep fold
      return P({
        left_arm_1: a1, right_arm_1: -a1,
        left_arm_2: a2, right_arm_2: -a2,
        left_arm_3: a3, right_arm_3: -a3,
        left_arm_4: a4, right_arm_4: -a4,
        left_arm_6: 0.3 * s, right_arm_6: 0.3 * s,      // curve the hands
        left_grip: 0.25 * up, right_grip: 0.25 * up,    // gentle finger curve
        torso: 0.06 * up,                               // proud chest-open
      });
    },
    nav: null,
  },

  happy: {
    title: 'Happy Bounce', emoji: '😄', dur: 6.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.0;
      const upW = win(t, 0.1, d - 0.1, 1.25);
      const bounce = osc(t, 1.1, 0.9, d - 0.9, 0.7);    // the joy engine
      const lift = 0.15875 + 0.045 * upW + 0.035 * upW * Math.max(0, bounce);
      return P({
        // V-for-victory arms, pumping with the bounce (slightly asymmetric)
        left_arm_1: (-1.75 + 0.16 * bounce) * upW,
        right_arm_1: (1.75 + 0.19 * bounce) * upW,
        left_arm_2: 0.5 * upW, right_arm_2: -0.44 * upW,
        left_arm_4: (0.35 - 0.22 * bounce) * upW,
        right_arm_4: (-0.35 - 0.25 * bounce) * upW,
        left_grip: 0.05 * upW, right_grip: 0.05 * upW,
        lift_lower: lift, lift_upper: lift,             // physical bounce!
        torso: 0.07 * upW + 0.02 * bounce * upW,
      });
    },
    nav: null,
  },

  sad: {
    title: 'Sad Slump', emoji: '😔', dur: 7.5, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 7.5;
      const slump = win(t, 0.15, d - 0.15, 2.2);        // very slow droop
      const sigh = breath(t, 1.6, d - 1.6, 0.16);       // one long sigh rhythm
      return P({
        torso: -0.7 * slump - 0.05 * sigh,              // deep dejected lean
        left_arm_1: 1.1 * slump, right_arm_1: -1.1 * slump,  // arms hang dead
        left_arm_2: 0.22 * slump, right_arm_2: -0.22 * slump, // clear of the body
        left_arm_4: 0.2 * slump + 0.03 * sigh,
        right_arm_4: -0.2 * slump - 0.03 * sigh,
        left_arm_6: 0.4 * slump, right_arm_6: 0.4 * slump,   // wrists droop too
        left_grip: 0.28 * slump, right_grip: 0.28 * slump,
        lift_lower: 0.15875 - 0.0037 * slump,          // sags to the model's floor
        lift_upper: 0.15875 - 0.0037 * slump,
      });
    },
    nav: null,
  },

  angry: {
    title: 'Angry Fists', emoji: '😠', dur: 5.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 5.0;
      const tense = win(t, 0.1, d - 0.1, 1.3);          // brisk attack (vel budget)
      const tremor = win(t, 1.2, d - 1.0, 0.4) *
        Math.sin(2 * Math.PI * 4.0 * t) * 0.045;        // shaking with rage
      const stomp = win(t, 2.1, 3.3, 0.35);             // one emphatic shake
      return P({
        // fists shaking above the shoulders (verified G silhouette), elbows up
        left_arm_1: (-2.0 - 0.15 * stomp) * tense + tremor,
        right_arm_1: (2.0 + 0.15 * stomp) * tense - tremor,
        left_arm_2: 0.2 * tense, right_arm_2: -0.2 * tense,
        left_arm_3: -0.9 * tense, right_arm_3: 0.9 * tense,
        left_arm_4: 1.5 * tense + 2 * tremor, right_arm_4: -1.5 * tense - 2 * tremor,
        left_grip: Math.min(1, 1.2 * seg(t, 0.1, 2.0)) * (1 - seg(t, d - 1.7, d - 0.1)),
        right_grip: Math.min(1, 1.2 * seg(t, 0.1, 2.0)) * (1 - seg(t, d - 1.7, d - 0.1)),
        torso: -0.22 * tense - 0.06 * stomp,            // looming forward
        lift_lower: 0.15875 + 0.045 * tense,            // rises up, bristling
        lift_upper: 0.15875 + 0.045 * tense,
      });
    },
    nav: null,
  },

  // ═══ DANCES (lane 'both': manip block + linked nav block) ═════════════════
  groove: {
    title: 'Groove', emoji: '🕺', dur: 9.0, lane: 'both',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 9.0;
      const on = win(t, 0.1, d - 0.1, 1.0);
      const beat = osc(t, 0.85, 0.6, d - 0.7, 0.7);       // main groove wave
      const off = osc(t, 0.85, 0.6, d - 0.7, 0.7 + 0.25); // phase-offset arm
      const pump = Math.max(0, beat), dip = Math.max(0, -beat);
      const lift = 0.15875 + 0.04 * on + 0.03 * on * pump;
      return P({
        // alternating arm pumps, elbows leading — classic two-step groove
        left_arm_1: (-0.85 - 0.42 * beat) * on,
        right_arm_1: (0.85 - 0.42 * off) * on,
        left_arm_2: 0.35 * on, right_arm_2: -0.35 * on,
        left_arm_4: (0.9 + 0.35 * beat) * on,
        right_arm_4: (-0.9 + 0.35 * off) * on,
        left_arm_5: -0.4 * on, right_arm_5: 0.4 * on,
        left_grip: 0.35 * on, right_grip: 0.35 * on,
        torso: (-0.14 - 0.1 * dip) * on,                  // bounce in the hips
        lift_lower: lift, lift_upper: lift,
      });
    },
    nav(t) {
      const d = 9.0;
      // side-to-side sway synced to the beat + tiny yaw wiggle
      return {
        lx: 0.28 * osc(t, 0.425, 0.6, d - 0.7, 0.9),
        ly: 0,
        rx: 0.15 * osc(t, 0.85, 0.9, d - 1.0, 0.9),
        ry: 0,
      };
    },
  },

  spin: {
    title: 'Spin Move', emoji: '🌀', dur: 8.0, lane: 'both',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 8.0;
      const pose = win(t, 0.1, d - 0.1, 1.0);          // arms out for the spin
      const flare = win(t, 1.0, d - 1.0, 1.2);
      return P({
        // arms sweep out and slightly up — figure-skater silhouette
        left_arm_1: -0.55 * pose, right_arm_1: 0.55 * pose,
        left_arm_2: 1.15 * flare, right_arm_2: -1.15 * flare,
        left_arm_4: 0.25 * pose, right_arm_4: -0.25 * pose,
        left_arm_5: -0.6 * flare, right_arm_5: 0.6 * flare,
        left_grip: 0.1 * pose, right_grip: 0.1 * pose,
        torso: 0.05 * pose,
        lift_lower: 0.15875 + 0.06 * flare, lift_upper: 0.15875 + 0.06 * flare,
      });
    },
    nav(t) {
      const d = 8.0;
      // full CCW pirouette: ω=1.5·rx, ∫ ≈ 2π over the windowed push
      return { lx: 0, ly: 0, rx: 0.78 * win(t, 0.9, d - 0.7, 1.0), ry: 0 };
    },
  },


  // ── the second wave of emotes (2026-09) ───────────────────────────────────
  // Authored against the SAME measured conventions as the six above, but with
  // two things the originals could not use: the elbow now RESTS extended, so an
  // arm raised by `arm_1` alone reads as a whole straight arm; and the neck is
  // drivable from a joint command, so a clip can look where it is gesturing
  // instead of implying it with the torso.

  nod_yes: {
    title: 'Nod Yes', emoji: '🙂', dur: 3.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 3.6;
      const on = win(t, 0.1, d - 0.1, 0.7);
      const nod = osc(t, 0.85, 0.5, d - 0.5, 0.5);
      return P({
        neck_pitch: (-0.10 + 0.34 * nod) * on,      // three unhurried nods
        torso: -0.03 * on + 0.02 * nod * on,
        left_arm_2: 0.06 * on, right_arm_2: -0.06 * on,   // arms breathe with it
      });
    },
    nav: null,
  },

  shake_no: {
    title: 'Shake No', emoji: '🙅', dur: 3.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 3.6;
      const on = win(t, 0.1, d - 0.1, 0.7);
      const sway = osc(t, 0.62, 0.5, d - 0.5, 0.6);
      return P({
        neck_yaw: 0.55 * sway * on,
        neck_roll: 0.06 * sway * on,                 // a little counter-tilt
        torso: -0.04 * on,
        left_arm_2: 0.10 * on, right_arm_2: -0.10 * on,
      });
    },
    nav: null,
  },

  curious: {
    title: 'Curious Tilt', emoji: '🤨', dur: 4.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.6;
      const on = win(t, 0.15, d - 0.15, 1.0);
      const peek = win(t, 1.5, d - 1.0, 0.8);
      return P({
        neck_roll: 0.5 * on,                         // the head tilt itself
        neck_yaw: -0.22 * on, neck_pitch: 0.12 * on,
        torso: -0.12 * peek,                         // lean in for a closer look
        left_arm_2: 0.10 * on, right_arm_2: -0.14 * on,
        right_arm_1: 0.12 * peek,
      });
    },
    nav: null,
  },

  scan_room: {
    title: 'Scan the Room', emoji: '🛰', dur: 9.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 9.0;
      const on = win(t, 0.2, d - 0.2, 1.2);
      const sweep = osc(t, 0.16, 0.6, d - 0.6, 1.0);   // one slow left-right pass
      return P({
        neck_yaw: 1.25 * sweep * on,                   // ±72° of the ±90° range
        neck_pitch: 0.10 * on - 0.06 * Math.abs(sweep) * on,
        torso: 0.05 * on,
      });
    },
    nav: null,
  },

  salute: {
    title: 'Salute', emoji: '🫡', dur: 4.8, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.8;
      const up = win(t, 0.2, d - 0.3, 1.0);            // hand to the brow, hold
      const snap = seg(t, 0.4, 1.0);                   // the crisp part
      return P({
        right_arm_1: 1.45 * up,                        // upper arm forward-up…
        right_arm_2: -0.55 * up,                       //   …and out
        right_arm_3: 0.85 * up,
        right_arm_4: -0.35 * up,                       // deep fold: hand at the brow
        right_arm_6: -0.25 * up,
        right_grip: 0.12 * snap * up,                  // flat hand
        neck_pitch: 0.12 * up,
        torso: 0.07 * up,                              // chest out, formal
      });
    },
    nav: null,
  },

  shrug: {
    title: 'Shrug', emoji: '🤷', dur: 4.4, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.4;
      const up = win(t, 0.25, d - 0.25, 1.1);
      return P({
        // palms turned up and out, elbows tucked — the "who knows" shape
        left_arm_1: -0.55 * up, right_arm_1: 0.55 * up,
        left_arm_2: 0.75 * up, right_arm_2: -0.75 * up,
        left_arm_3: -0.45 * up, right_arm_3: 0.45 * up,
        left_arm_4: 0.55 * up, right_arm_4: -0.55 * up,
        left_arm_5: 0.9 * up, right_arm_5: -0.9 * up,  // roll the palms up
        left_grip: 0.06 * up, right_grip: 0.06 * up,
        neck_roll: 0.2 * up, neck_pitch: -0.1 * up,
        lift_lower: 0.15875 + 0.035 * up,              // the shoulders "rise"
        lift_upper: 0.15875 + 0.035 * up,
      });
    },
    nav: null,
  },

  clap: {
    title: 'Applause', emoji: '👏', dur: 6.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.0;
      const up = win(t, 0.2, d - 0.2, 1.1);            // hands to chest height
      const beat = osc(t, 1.05, 1.0, d - 0.9, 0.6);    // the claps
      const close = Math.max(0, beat);
      return P({
        left_arm_1: -0.95 * up, right_arm_1: 0.95 * up,
        left_arm_2: (0.55 - 0.34 * close) * up,        // in/out is the clap
        right_arm_2: (-0.55 + 0.34 * close) * up,
        left_arm_3: -0.75 * up, right_arm_3: 0.75 * up,
        left_arm_4: 0.75 * up, right_arm_4: -0.75 * up,
        left_grip: 0.08 * up, right_grip: 0.08 * up,
        neck_pitch: -0.12 * up, torso: 0.05 * up + 0.02 * close * up,
      });
    },
    nav: null,
  },

  thumbs_up: {
    title: 'Thumbs Up', emoji: '👍', dur: 4.2, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.2;
      const up = win(t, 0.2, d - 0.25, 0.9);
      const curl = seg(t, 0.5, 1.4) - seg(t, d - 1.3, d - 0.3);
      const c = Math.max(0, curl);
      return P({
        right_arm_1: 0.85 * up, right_arm_2: -0.35 * up,
        right_arm_3: 0.55 * up, right_arm_4: -0.55 * up,
        right_arm_5: -0.75 * up,                       // knuckles rolled inboard
        ...THUMB_UP_HAND('right', c * up),
        neck_pitch: -0.08 * up, neck_yaw: -0.18 * up,  // look at the hand
        torso: 0.04 * up,
      });
    },
    nav: null,
  },

  point_there: {
    title: 'Point That Way', emoji: '👉', dur: 5.2, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 5.2;
      const up = win(t, 0.2, d - 0.25, 1.0);
      const jab = win(t, 2.0, 3.2, 0.4);               // a small emphatic push
      const curl = seg(t, 0.4, 1.3) - seg(t, d - 1.2, d - 0.3);
      const c = Math.max(0, curl);
      return P({
        right_arm_1: (1.45 + 0.12 * jab) * up,         // straight arm, level
        right_arm_2: -0.45 * up,
        ...POINT_HAND('right', c * up),
        neck_yaw: -0.55 * up,                          // look where you point
        neck_pitch: -0.05 * up,
        torso: -0.05 * up - 0.03 * jab,
      });
    },
    nav: null,
  },

  high_five: {
    title: 'High Five', emoji: '🙏', dur: 4.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 4.6;
      const up = win(t, 0.2, d - 0.25, 1.0);
      const push = win(t, 2.0, 3.1, 0.35);
      return P({
        right_arm_1: (2.05 + 0.15 * push) * up,        // palm up and forward
        right_arm_2: -0.35 * up,
        right_arm_3: 0.25 * up,
        right_arm_6: 0.2 * up,
        right_grip: 0.05 * up,                          // flat, open palm
        neck_pitch: 0.28 * up,                          // look up at the hand
        torso: 0.05 * up - 0.04 * push,
      });
    },
    nav: null,
  },

  wave_both: {
    title: 'Wave (both arms)', emoji: '🙌', dur: 6.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.0;
      const up = win(t, 0.15, d - 0.15, 1.3);
      const wig = osc(t, 0.9, 1.2, d - 1.0, 0.6);
      const brt = breath(t, 0, d);
      return P({
        left_arm_1: -2.35 * up, right_arm_1: 2.35 * up,   // both arms up high
        left_arm_2: 0.35 * up, right_arm_2: -0.35 * up,
        left_arm_4: (1.15 + 0.22 * wig) * up,
        right_arm_4: (-1.15 + 0.22 * wig) * up,
        left_arm_7: 0.35 * wig * up, right_arm_7: 0.35 * wig * up,
        left_grip: 0.06 * up, right_grip: 0.06 * up,
        neck_pitch: 0.2 * up + 0.02 * brt,
        torso: 0.06 * up,
      });
    },
    nav: null,
  },

  hug: {
    title: 'Hug', emoji: '🤗', dur: 6.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.6;
      const open = win(t, 0.2, 4.2, 1.1);              // arms open wide…
      const close = win(t, 2.6, d - 0.3, 1.0);         // …then fold in
      return P({
        left_arm_1: -1.25 * open - 0.15 * close,
        right_arm_1: 1.25 * open + 0.15 * close,
        left_arm_2: 0.95 * open - 0.7 * close,
        right_arm_2: -0.95 * open + 0.7 * close,
        left_arm_3: -0.3 * close, right_arm_3: 0.3 * close,
        left_arm_4: 1.35 * open - 0.85 * close,
        right_arm_4: -1.35 * open + 0.85 * close,
        left_grip: 0.05 * open + 0.2 * close,
        right_grip: 0.05 * open + 0.2 * close,
        neck_pitch: 0.1 * open - 0.22 * close,
        torso: 0.07 * open - 0.14 * close,
      });
    },
    nav: null,
  },

  think: {
    title: 'Thinking', emoji: '🤔', dur: 7.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 7.0;
      const up = win(t, 0.3, d - 0.3, 1.4);            // hand drifts to the chin
      const tap = osc(t, 0.9, 2.2, d - 1.6, 0.7) * 0.09;
      return P({
        right_arm_1: 1.55 * up, right_arm_2: -0.45 * up,
        right_arm_3: 0.95 * up,
        right_arm_4: (-0.3 + tap) * up,                 // deep fold: hand at chin
        right_arm_6: -0.3 * up,
        right_grip: (0.45 + 1.2 * tap) * up,
        left_arm_1: 0.12 * up,                          // other arm crosses under
        left_arm_2: 0.25 * up,
        neck_roll: 0.28 * up, neck_pitch: -0.18 * up,
        neck_yaw: 0.2 * up,                             // gaze off into space
        torso: -0.1 * up,
      });
    },
    nav: null,
  },

  facepalm: {
    title: 'Facepalm', emoji: '🤦', dur: 5.6, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 5.6;
      const up = win(t, 0.25, d - 0.35, 1.0);
      const sink = win(t, 1.6, d - 0.6, 1.2);
      return P({
        right_arm_1: 1.75 * up, right_arm_2: -0.3 * up,
        right_arm_3: 1.1 * up, right_arm_4: -0.2 * up,
        right_arm_6: -0.35 * up,
        right_grip: 0.06 * up,
        neck_pitch: -0.45 * sink,                       // head drops into the hand
        torso: -0.28 * sink,
        left_arm_2: 0.12 * up,
      });
    },
    nav: null,
  },

  stretch: {
    title: 'Big Stretch', emoji: '🙆', dur: 8.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 8.0;
      const up = win(t, 0.3, d - 0.4, 1.9);            // slow, luxurious
      const arch = win(t, 2.2, 5.8, 1.1);
      return P({
        left_arm_1: -2.9 * up, right_arm_1: 2.9 * up,   // straight overhead
        left_arm_2: 0.25 * up, right_arm_2: -0.25 * up,
        left_arm_5: -0.35 * arch, right_arm_5: 0.35 * arch,
        left_grip: 0.12 * up, right_grip: 0.12 * up,
        neck_pitch: 0.42 * arch,
        torso: 0.2 * arch,                              // arch back
        lift_lower: 0.15875 + 0.09 * up,                // rise onto the toes
        lift_upper: 0.15875 + 0.09 * up,
      });
    },
    nav: null,
  },

  yawn: {
    title: 'Yawn', emoji: '🥱', dur: 6.4, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.4;
      const up = win(t, 0.4, d - 0.5, 1.6);
      const peak = win(t, 2.0, 4.2, 0.9);
      return P({
        right_arm_1: 1.6 * up, right_arm_3: 0.9 * up,   // hand drifts to the mouth
        right_arm_4: -0.45 * up, right_grip: 0.3 * up,
        left_arm_1: -1.9 * up, left_arm_2: 0.4 * up,    // other arm stretches up
        neck_pitch: 0.5 * peak,
        torso: 0.14 * peak - 0.1 * (up - peak),
        lift_lower: 0.15875 + 0.05 * peak,
        lift_upper: 0.15875 + 0.05 * peak,
      });
    },
    nav: null,
  },

  sleep: {
    title: 'Power Down', emoji: '😴', dur: 7.5, lane: 'manip',
    category: 'animation', status: 'available', hold: true,   // stays asleep
    joints(t) {
      const d = 7.5;
      const off = seg(t, 0.4, d - 0.6);                 // one slow monotone sag
      return P({
        neck_pitch: -0.62 * off, neck_roll: 0.22 * off,
        torso: -0.4 * off,
        left_arm_2: -0.17 * off, right_arm_2: 0.17 * off,   // arms fall slack
        left_grip: 0.22 * off, right_grip: 0.22 * off,
        lift_lower: 0.15875 - 0.0037 * off,              // settles a few mm
        lift_upper: 0.15875 - 0.0037 * off,
      });
    },
    nav: null,
  },

  wake_up: {
    title: 'Wake Up', emoji: '🌅', dur: 7.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 7.0;
      const rise = seg(t, 0.3, 3.0);                    // from slack to upright
      const look = osc(t, 0.28, 3.0, d - 0.5, 0.9);
      const on = win(t, 0.1, d - 0.1, 1.0);
      return P({
        neck_pitch: (-0.5 + 0.62 * rise) * on,
        neck_yaw: 0.6 * look * on,                      // has a look around
        neck_roll: (0.2 - 0.2 * rise) * on,
        torso: (-0.32 + 0.4 * rise) * on,
        left_arm_1: -0.35 * rise * on, right_arm_1: 0.35 * rise * on,
        left_arm_2: 0.22 * rise * on, right_arm_2: -0.22 * rise * on,
        left_grip: 0.2 * (1 - rise) * on, right_grip: 0.2 * (1 - rise) * on,
        lift_lower: 0.15875 + 0.055 * rise * on,
        lift_upper: 0.15875 + 0.055 * rise * on,
      });
    },
    nav: null,
  },

  peekaboo: {
    title: 'Peek-a-boo', emoji: '🫣', dur: 6.2, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 6.2;
      const up = win(t, 0.2, d - 0.3, 0.9);            // hands up to the face
      const open = win(t, 2.9, d - 0.4, 1.0);          // …and thrown apart
      return P({
        left_arm_1: -1.85 * up, right_arm_1: 1.85 * up,
        left_arm_2: (0.63 + 0.9 * open) * up,          // keep the hands apart
        right_arm_2: (-0.63 - 0.9 * open) * up,
        left_arm_3: 0.21 * up * (1 - 0.7 * open),      // barely turned in
        right_arm_3: -0.21 * up * (1 - 0.7 * open),
        left_arm_4: (-0.05 + 1.2 * open) * up,
        right_arm_4: (0.05 - 1.2 * open) * up,
        left_grip: 0.06 * up, right_grip: 0.06 * up,
        neck_pitch: -0.2 * up + 0.42 * open,
        torso: -0.12 * up + 0.16 * open,
      });
    },
    nav: null,
  },

  flex: {
    title: 'Flex', emoji: '💪', dur: 5.4, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 5.4;
      const up = win(t, 0.25, d - 0.3, 1.0);
      const squeeze = osc(t, 0.7, 1.4, d - 1.0, 0.6) * 0.5 + 0.5;
      return P({
        // double biceps: upper arms out level, forearms folded straight up
        left_arm_1: -0.15 * up, right_arm_1: 0.15 * up,
        left_arm_2: 1.35 * up, right_arm_2: -1.35 * up,
        left_arm_3: -1.15 * up, right_arm_3: 1.15 * up,
        left_arm_4: (-0.1 - 0.35 * squeeze) * up,
        right_arm_4: (0.1 + 0.35 * squeeze) * up,
        left_grip: (0.7 + 0.3 * squeeze) * up,
        right_grip: (0.7 + 0.3 * squeeze) * up,
        neck_yaw: -0.3 * up, neck_pitch: -0.1 * up,
        torso: 0.1 * up,
      });
    },
    nav: null,
  },

  tai_chi: {
    title: 'Tai Chi', emoji: '☯️', dur: 14.0, lane: 'manip',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 14.0;
      const on = win(t, 0.4, d - 0.4, 2.0);            // everything very slow
      const flow = osc(t, 0.11, 1.0, d - 1.0, 1.6);
      const rise = osc(t, 0.055, 1.0, d - 1.0, 1.6);
      return P({
        left_arm_1: (-1.05 - 0.55 * flow) * on,
        right_arm_1: (1.05 - 0.55 * flow) * on,        // the two arms counter-move
        left_arm_2: (0.45 + 0.3 * rise) * on,
        right_arm_2: (-0.45 + 0.3 * rise) * on,
        left_arm_4: (0.95 + 0.3 * flow) * on,
        right_arm_4: (-0.95 + 0.3 * flow) * on,
        left_arm_5: 0.5 * flow * on, right_arm_5: -0.5 * flow * on,
        left_arm_6: 0.25 * rise * on, right_arm_6: 0.25 * rise * on,
        left_grip: 0.06 * on, right_grip: 0.06 * on,
        neck_yaw: 0.45 * flow * on, neck_pitch: 0.08 * on,
        torso: (-0.06 + 0.1 * rise) * on,
        // The lift breathes with the form, but only UPWARD from rest: the
        // model's column self-penetrates below 0.155 m/stage (exp_rest_pose E4).
        lift_lower: 0.15875 + 0.022 * (0.5 + 0.5 * rise) * on,
        lift_upper: 0.15875 + 0.022 * (0.5 + 0.5 * rise) * on,
      });
    },
    nav: null,
  },

  celebrate: {
    title: 'Celebrate', emoji: '🎉', dur: 8.0, lane: 'both',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 8.0;
      const up = win(t, 0.2, d - 0.3, 1.5);
      const pump = osc(t, 0.75, 1.6, d - 1.2, 0.8);
      const hi = Math.max(0, pump);
      return P({
        left_arm_1: (-2.75 - 0.2 * pump) * up,
        right_arm_1: (2.75 + 0.2 * pump) * up,
        left_arm_2: 0.3 * up, right_arm_2: -0.3 * up,
        left_arm_4: (1.3 - 0.25 * hi) * up,
        right_arm_4: (-1.3 + 0.25 * hi) * up,
        left_grip: (0.2 + 0.25 * hi) * up, right_grip: (0.2 + 0.25 * hi) * up,
        neck_pitch: (0.35 + 0.1 * pump) * up,
        torso: 0.15 * up,
        lift_lower: 0.15875 + (0.075 + 0.055 * hi) * up,
        lift_upper: 0.15875 + (0.075 + 0.055 * hi) * up,
      });
    },
    nav(t) {
      const d = 8.0;
      // a little celebratory shimmy: sway + a half turn and back
      return { lx: 0.22 * osc(t, 0.5, 1.2, d - 1.2, 0.9), ly: 0,
               rx: 0.3 * osc(t, 0.25, 1.4, d - 1.2, 1.0), ry: 0 };
    },
  },

  robot_dance: {
    title: 'Robot Dance', emoji: '🤖', dur: 9.0, lane: 'both',
    category: 'animation', status: 'available',
    joints(t) {
      const d = 9.0;
      const on = win(t, 0.2, d - 0.3, 1.0);
      // Deliberately STEPPED, not smooth — the quantisation is the joke. Each
      // step is still EASED across SNAP of its own period, so the signal is
      // C¹ and the velocity budget holds: a unit step across 0.45·(1/1.1) s
      // peaks at κ/0.474 ≈ 3.3 rad/s per unit, which is why the amplitudes
      // stay at or below 0.28.
      const SNAP = 0.45;
      const step = (f, n, ph) => {
        const u = Math.max(0, t - 0.8) * f + ph;
        const k = Math.floor(u);
        const lvl = (i) => ((((i % n) + n) % n) / (n - 1)) * 2 - 1;
        return lvl(k) + (lvl(k + 1) - lvl(k)) * ease((u - k) / SNAP);
      };
      const s1 = step(0.95, 4, 0), s2 = step(0.95, 3, 0.5);
      return P({
        left_arm_1: (-0.9 + 0.28 * s1) * on,
        right_arm_1: (0.9 + 0.28 * s2) * on,
        left_arm_2: (0.75 + 0.25 * s2) * on,
        right_arm_2: (-0.75 - 0.25 * s1) * on,
        left_arm_3: -0.85 * on, right_arm_3: 0.85 * on,
        left_arm_4: (0.35 + 0.25 * s2) * on,
        right_arm_4: (-0.35 - 0.25 * s1) * on,
        left_grip: 0.25 * on, right_grip: 0.25 * on,
        neck_yaw: 0.22 * s2 * on, neck_pitch: 0.08 * s1 * on,
        torso: (-0.06 + 0.05 * s1) * on,
      });
    },
    nav(t) {
      const d = 9.0;
      return { lx: 0.2 * osc(t, 0.55, 1.0, d - 1.0, 0.8), ly: 0,
               rx: 0.18 * osc(t, 0.275, 1.0, d - 1.0, 0.8), ry: 0 };
    },
  },

  // ═══ DUMMY TASK POLICIES — scripted loco-manipulation, run today ══════════
  pickup: {
    title: 'Pick Up Object', emoji: '🫳', dur: 8.0, lane: 'manip',
    category: 'policy', status: 'available', hold: true,   // ends holding (carry)
    joints(t) {
      const d = 8.0;
      // phases: lean+reach fwd-down (verified reach_low) → grasp → hug into a
      // carry (verified carry silhouette) and HOLD it — the block ends carrying.
      const lean = win(t, 0.3, 5.6, 1.2);               // torso down … back up
      const reach = win(t, 0.8, 5.2, 1.1);              // arm out-down … withdraw
      const grasp = seg(t, 2.7, 4.0);                   // fingers close (stay closed)
      const carry = seg(t, 4.6, 6.2);                   // fold into the carry
      const settle = seg(t, 0.15, 1.0);                 // left arm drops clear
      return P({
        torso: -0.55 * lean,
        right_arm_2: -0.45 * reach - 0.3 * carry,       // clear of the column
        right_arm_1: -0.45 * reach + 0.55 * carry,      // verified carry: ra1=+0.55
        right_arm_3: -0.4 * reach - 0.9 * carry,        // …ra3=−0.9
        right_arm_4: -0.3 * reach - 1.2 * carry,        // …ra4=−1.2
        right_arm_6: -0.3 * reach + 0.3 * carry,
        right_grip: Math.min(1, 0.9 * grasp),
        left_arm_1: 1.0 * settle,                       // left arm stays clear
        left_arm_4: 0.4 * settle,
      });
    },
    nav: null,
  },

  place: {
    title: 'Place Down', emoji: '🫴', dur: 8.0, lane: 'manip',
    category: 'policy', status: 'available',
    joints(t) {
      const d = 8.0;
      // inverse of pickup: start in the carry, extend down, release, retract home
      const carry = 1 - seg(t, 1.2, 3.4);               // unfold from the carry
      const lower = win(t, 2.2, 6.4, 1.2);              // reach down to place
      const grasp = seg(t, 0.05, 1.6) - seg(t, 4.3, 5.8); // secure … release
      const settle = win(t, 0.2, d - 0.2, 0.9);
      return P({
        torso: -0.55 * lower,
        right_arm_2: -0.45 * lower - 0.3 * carry * settle,   // clear of the column
        right_arm_1: 1.0 * carry * settle - 0.45 * lower,
        right_arm_3: -0.9 * carry * settle - 0.4 * lower,
        right_arm_4: -1.2 * carry * settle - 0.3 * lower,
        right_arm_6: 0.3 * carry * settle - 0.3 * lower,
        right_grip: Math.max(0, Math.min(1, 0.9 * grasp)),
        left_arm_1: 1.0 * settle, left_arm_4: 0.4 * settle,
      });
    },
    nav: null,
  },

  open_drawer: {
    title: 'Open Drawer', emoji: '🗄', dur: 7.0, lane: 'both',
    category: 'policy', status: 'available',
    joints(t) {
      const d = 7.0;
      const reach = win(t, 0.2, 5.6, 1.0);              // reach mid-height handle
      const hook = seg(t, 1.7, 2.9) - seg(t, 4.9, 6.0); // curl fingers into a hook
      const pull = win(t, 2.8, 5.0, 0.7);               // arm pulls back w/ base
      const settle = win(t, 0.1, d - 0.1, 0.8);
      return P({
        torso: -0.18 * reach,
        right_arm_1: 0.15 * reach + 0.35 * pull,        // handle height → pull in
        right_arm_4: -0.2 * reach - 0.55 * pull,
        right_arm_5: 0.65 * reach,                      // knuckles up for the hook
        right_arm_6: -0.25 * reach,
        right_grip: Math.max(0, Math.min(1, 0.72 * hook)),
        left_arm_1: 1.05 * settle, left_arm_4: 0.35 * settle,
      });
    },
    nav(t) {
      // the base backs up ~0.2 m in sync with the pull (v=0.8·ly)
      return { lx: 0, ly: -0.16 * win(t, 2.8, 4.8, 0.6), rx: 0, ry: 0 };
    },
  },

  open_fridge: {
    title: 'Open Fridge', emoji: '🧊', dur: 8.0, lane: 'both',
    category: 'policy', status: 'available',
    joints(t) {
      const d = 8.0;
      const reach = win(t, 0.2, 6.4, 1.1);              // up to the tall handle
      const grab = seg(t, 1.9, 3.0) - seg(t, 5.4, 6.6);
      const arc = win(t, 3.0, 5.8, 0.9);                // swing the door open
      const settle = win(t, 0.1, d - 0.1, 0.8);
      const lift = 0.15875 + 0.12 * reach;
      return P({
        lift_lower: lift, lift_upper: lift,             // stand tall for the handle
        right_arm_1: 0.7 * reach + 0.1 * arc,           // high reach
        right_arm_2: -0.2 * reach - 0.5 * arc,          // arm sweeps outward
        right_arm_4: -0.35 * reach - 0.3 * arc,
        right_arm_5: 0.5 * reach,
        right_grip: Math.max(0, Math.min(1, 0.85 * grab)),
        left_arm_1: 1.05 * settle, left_arm_4: 0.35 * settle,
        torso: -0.08 * reach,
      });
    },
    nav(t) {
      // arc back-left with the door: reverse + slight CCW yaw, together
      const w = win(t, 3.0, 5.6, 0.8);
      return { lx: 0.1 * w, ly: -0.14 * w, rx: 0.16 * w, ry: 0 };
    },
  },

  push_button: {
    title: 'Push Button', emoji: '🔘', dur: 6.0, lane: 'manip',
    category: 'policy', status: 'available',
    joints(t) {
      const d = 6.0;
      const aim = win(t, 0.2, d - 0.2, 1.0);            // arm up to the panel
      const nudge = win(t, 2.5, 3.9, 0.45);             // the press itself
      const settle = win(t, 0.1, d - 0.1, 0.8);
      const curl = seg(t, 0.5, 1.5) - seg(t, 4.6, 5.5);   // point, then relax
      return P({
        right_arm_1: 0.3 * aim - 0.16 * nudge,          // press = shoulder push
        right_arm_4: -0.35 * aim + 0.22 * nudge,
        right_arm_6: -0.15 * aim,
        ...POINT_HAND('right', curl),
        left_arm_1: 1.05 * settle, left_arm_4: 0.35 * settle,
        torso: -0.06 * aim - 0.03 * nudge,
      });
    },
    nav: null,
  },

  inspect: {
    title: 'Inspect Hands', emoji: '🔍', dur: 7.0, lane: 'manip',
    category: 'policy', status: 'available',
    joints(t) {
      const d = 7.0;
      const raise = win(t, 0.2, d - 0.2, 1.2);          // hands come up to the face
      const turn = osc(t, 0.3, 2.0, 5.4, 0.8);          // slow wrist rotation
      const brt = breath(t, 1.5, d - 1.2);
      return P({
        // verified J silhouette: fingertips meet in front of the face
        left_arm_1: -1.9 * raise, right_arm_1: 1.9 * raise,
        left_arm_2: 0.2 * raise, right_arm_2: -0.2 * raise,
        left_arm_3: 1.4 * raise, right_arm_3: -1.4 * raise,
        left_arm_4: 1.52 * raise, right_arm_4: -1.6 * raise,
        left_arm_5: 0.3 * turn * raise,                 // palms rotate — examining
        right_arm_5: -0.3 * turn * raise,
        left_grip: 0.12 * raise, right_grip: 0.12 * raise,
        torso: -0.25 * raise - 0.02 * brt,              // "looking down" via lean
      });
    },
    nav: null,
  },

  // ═══ COMING-SOON learned-policy slots (greyed out, NOT addable) ═══════════
  boxing:        { title: 'Boxing', emoji: '🥊', dur: 30, lane: 'manip',
    category: 'policy', status: 'coming_soon', location: { x: 2.0, y: -2.5, label: 'gym' } },
  do_laundry:    { title: 'Do Laundry', emoji: '🧺', dur: 120, lane: 'both',
    category: 'policy', status: 'coming_soon', location: { x: 2.5, y: -1.5, label: 'laundry room' } },
  fold_clothes:  { title: 'Fold Clothes', emoji: '👕', dur: 90, lane: 'manip',
    category: 'policy', status: 'coming_soon', location: { x: 2.5, y: -1.5, label: 'laundry room' } },
  make_coffee:   { title: 'Make Coffee', emoji: '☕', dur: 60, lane: 'manip',
    category: 'policy', status: 'coming_soon', location: { x: 1.2, y: 0.6, label: 'kitchen' } },
  do_dishes:     { title: 'Do the Dishes', emoji: '🍽', dur: 180, lane: 'manip',
    category: 'policy', status: 'coming_soon', location: { x: 1.2, y: 0.6, label: 'kitchen' } },
  cook_breakfast:{ title: 'Cook Breakfast', emoji: '🍳', dur: 300, lane: 'both',
    category: 'policy', status: 'coming_soon', location: { x: 1.2, y: 0.6, label: 'kitchen' } },
  go_charge:     { title: 'Go to Charger', emoji: '🔌', dur: 20, lane: 'nav',
    category: 'policy', status: 'coming_soon', location: { x: 0.3, y: -1.7, label: 'dock' } },
  water_plants:  { title: 'Water Plants', emoji: '🪴', dur: 60, lane: 'both',
    category: 'policy', status: 'coming_soon', location: { x: -2.5, y: -2.0, label: 'garden' } },
  tidy_table:    { title: 'Tidy the Table', emoji: '🧹', dur: 120, lane: 'both',
    category: 'policy', status: 'coming_soon', location: { x: -1.6, y: 0.2, label: 'dining table' } },
  walk_dog:      { title: 'Walk the Dog', emoji: '🐕', dur: 600, lane: 'nav',
    category: 'policy', status: 'coming_soon', location: { x: -2.8, y: 1.5, label: 'front door' } },
  night_watch:   { title: 'Night Watch Patrol', emoji: '🛡', dur: 480, lane: 'nav',
    category: 'policy', status: 'coming_soon', location: { x: -2.8, y: 1.5, label: 'front door' } },
  serenade:      { title: 'Serenade (air guitar)', emoji: '🎸', dur: 45, lane: 'manip',
    category: 'policy', status: 'coming_soon', location: { x: 3.0, y: 1.8, label: 'bedroom' } },

  // ═══ NAV PRIMITIVES — parametric timed navJoystick streams ════════════════
  // trapezoidal |stick|=A profile with eased shoulders; durFor solves duration
  // from distance at NAV_SPEED·A (·NAV_YAW_RATE for turns).
  nav_forward: {
    title: 'Move Forward', emoji: '⬆️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { dist: { label: 'distance (m)', def: 0.5, min: 0.05, max: 4 } },
    durFor: (p) => navDur((p?.dist ?? 0.5) / (NAV_SPEED * 0.5)),
    nav(t, p) { return { ...Z4, ly: 0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_back: {
    title: 'Move Back', emoji: '⬇️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { dist: { label: 'distance (m)', def: 0.5, min: 0.05, max: 4 } },
    durFor: (p) => navDur((p?.dist ?? 0.5) / (NAV_SPEED * 0.5)),
    nav(t, p) { return { ...Z4, ly: -0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_strafe_left: {
    title: 'Strafe Left', emoji: '⬅️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { dist: { label: 'distance (m)', def: 0.4, min: 0.05, max: 3 } },
    durFor: (p) => navDur((p?.dist ?? 0.4) / (NAV_SPEED * 0.5)),
    nav(t, p) { return { ...Z4, lx: 0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_strafe_right: {
    title: 'Strafe Right', emoji: '➡️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { dist: { label: 'distance (m)', def: 0.4, min: 0.05, max: 3 } },
    durFor: (p) => navDur((p?.dist ?? 0.4) / (NAV_SPEED * 0.5)),
    nav(t, p) { return { ...Z4, lx: -0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_turn_left: {
    title: 'Turn Left', emoji: '↪️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { angle: { label: 'angle (deg)', def: 90, min: 5, max: 360 } },
    durFor: (p) => navDur(((p?.angle ?? 90) * Math.PI / 180) / (NAV_YAW_RATE * 0.5)),
    nav(t, p) { return { ...Z4, rx: 0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_turn_right: {
    title: 'Turn Right', emoji: '↩️', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available',
    params: { angle: { label: 'angle (deg)', def: 90, min: 5, max: 360 } },
    durFor: (p) => navDur(((p?.angle ?? 90) * Math.PI / 180) / (NAV_YAW_RATE * 0.5)),
    nav(t, p) { return { ...Z4, rx: -0.5 * navProfile(t, this.durFor(p)) }; },
    joints: null,
  },
  nav_hold: {
    title: 'Stop & Hold', emoji: '✋', dur: 2.0, lane: 'nav',
    category: 'nav', status: 'available', hold: true,
    params: { secs: { label: 'seconds', def: 2, min: 0.5, max: 30 } },
    durFor: (p) => (p?.secs ?? 2),
    nav() { return { ...Z4 }; },                       // actively streams zeros
    joints: null,
  },
  nav_goto: {
    title: 'Move to Goal (x, y)', emoji: '🎯', dur: 8.0, lane: 'nav',
    category: 'nav', status: 'available', kind: 'goal', hold: true,
    params: { x: { label: 'x (m)', def: 0, min: -10, max: 10 },
              y: { label: 'y (m)', def: 0, min: -10, max: 10 } },
    durFor: () => 8.0,                                  // backend-driven; est. window
    nav: null, joints: null,                            // REST-backed, no stream
  },
  // The lift clips state a TARGET and let the anchor ease into it. They stay
  // lane 'nav' and deliberately command nothing but the two lift stages, so a
  // manipulation block can run underneath one. Each stage tops out at 0.3175 m
  // and rests at its 0.15875 m ref, so either direction is half the stroke.
  lift_up: {
    title: 'Lift Up', emoji: '🔼', dur: 3.0, lane: 'nav',
    category: 'nav', status: 'available', hold: true,
    joints: () => ({ lift_lower: 0.3175, lift_upper: 0.3175 }),
    nav: null,
  },
  lift_down: {
    title: 'Lift Down', emoji: '🔽', dur: 3.0, lane: 'nav',
    category: 'nav', status: 'available', hold: true,
    joints: () => ({ lift_lower: 0, lift_upper: 0 }),
    nav: null,
  },

  // ═══ STATIC POSES — eased out of REST by the anchor, then held ═════════
  // A pose clip is nothing but its target: `hold` + the REST anchor supply the
  // eased transition, sized from how far the target actually is (anchorRamp).
  // `dur` only has to CONTAIN that ramp — the farthest of these is ~2.4 s of
  // travel at the 1.0 rad/s anchor ceiling — so 3.4 s leaves the pose visibly
  // held rather than merely reached.
  ...poses({
    pose_home:      ['Home Position',      '🏠',  'rest'],
    pose_tall:      ['Stand Tall',         '🧍',  'tall'],
    pose_bow:       ['Bend Forward (bow)', '🙇',  'bow'],
    pose_arms_up:   ['Arms Up',            '🙌',  'armsUp'],
    pose_tpose:     ['T-Pose',             '✝️', 'tpose'],
    pose_reach_fwd: ['Reach Forward',      '🪲',  'reach'],
    pose_ready:     ['Ready Stance',       '🦾',  'ready'],
    pose_carry:     ['Carry Tray',         '🍽️', 'carry'],
    pose_hips:      ['Hands on Hips',      '🕺',  'hips'],
    pose_wide:      ['Open Arms',          '🤗',  'wide'],
    pose_shelf:     ['Reach High Shelf',   '📚',  'shelf'],
    pose_floor:     ['Reach the Floor',    '🧹',  'floor'],
    pose_point:     ['Point Ahead',        '👉',  'point'],
    pose_present:   ['Present (one arm)',  '🤸',  'present'],
    pose_guard:     ['Guard',              '🛡️', 'guard'],
    pose_look_up:   ['Look Up',            '👀',  'lookUp'],
    pose_look_left: ['Look Left',          '↖️', 'lookLeft'],
    pose_look_right:['Look Right',         '↗️', 'lookRight'],
    pose_tuck:      ['Tuck In (narrow)',   '📦',  'tuck'],
    pose_crouch:    ['Crouch Low',         '🧎',  'crouch'],
  }),

  // ═══ THE BLEND (SAFETY-CRITICAL, derived — never added by hand) ═══════════
  // The transition `autoFixPoseJumps` inserts in front of a manipulation block
  // whose start pose does not agree with what precedes it. Its whole job is to
  // cross that gap slowly enough that no joint exceeds SAFE_BLEND_VEL.
  //
  // It is `status:'available'` because it is REAL commanded motion — it has to
  // execute — but `category:'transition'`, and no menu section in timeline.js
  // matches that category (they select 'animation' | 'policy' | 'nav' | 'pose'),
  // so it never appears in the palette. That is deliberate: a blend only means
  // something anchored to a specific jump, with `from`/`to` poses baked into
  // its params by the detector. A bare one dropped on the timeline would be a
  // transition between nothing and nothing.
  //
  // `kind:'xfade'` is load-bearing: it is how `detectPoseJumps` knows to ignore
  // blends when scanning for jumps, how `samplePoseBefore` knows to look
  // THROUGH a blend to the real pose behind it (otherwise a blend erases the
  // very jump that justifies it and the next reconcile deletes it), and how
  // `cut`/`trim` know this block has safety semantics.
  blend: {
    title: 'Blend', emoji: '〜', lane: 'manip', kind: 'xfade',
    category: 'transition', status: 'available', hold: false,
    // duration lives in params (sized from the jump distance), not in the clip
    dur: BLEND_MIN_DUR,
    durFor: (p) => Math.max(BLEND_MIN_DUR, (p && p.dur) || BLEND_MIN_DUR),
    // Eased interpolation from the measured/held pose into the next block's
    // start pose. Joints only in `from` ease toward their `to` value when the
    // target names them, and otherwise HOLD — a blend must never invent motion
    // on a joint the next block does not command.
    joints(t, p) {
      const from = (p && p.from) || {}, to = (p && p.to) || {};
      const d = Math.max(BLEND_MIN_DUR, (p && p.dur) || BLEND_MIN_DUR);
      const u = ease(t / d);
      const out = { ...from };
      for (const k in to) {
        const a = from[k];
        out[k] = a == null ? to[k] : a + (to[k] - a) * u;
      }
      return out;
    },
    nav: null,
  },
};



// eased trapezoid 0→A→0 for nav streams (unit peak) — total time d
export function navProfile(t, d) {
  const r = Math.min(0.6, d * 0.3);
  return win(t, 0, d, r);
}
// duration so that ∫profile·v dt ≈ wanted (plateau + eased shoulders ≈ d − r)
function navDur(idealSecs) {
  // win() with ramp r loses ≈ r seconds of area total (two half-ramps)
  const r = 0.6;
  return Math.max(0.8, idealSecs + r);
}

// ── clip sampling helpers ───────────────────────────────────────────────────
export function clipDur(clip, params) {
  return clip.durFor ? clip.durFor(params || defaults(clip)) : clip.dur;
}
export function defaults(clip) {
  const out = {};
  for (const [k, v] of Object.entries(clip.params || {})) out[k] = v.def;
  return out;
}
// Sample joints at local time t (clamped into [0, dur]; hold clips clamp beyond
// dur = they hold their final pose). This is the ONE place a clip's closure is
// evaluated — the studio executor, the 3-D preview and the app baker all come
// through here — so it is where the REST anchor and the joint-limit clamp live.
//
// `blend` is exempt from the anchor: it is not authored in the zero frame, it
// interpolates between two measured poses the pose-jump guard hands it, and
// anchoring it to REST would erase exactly the transition it exists to make.
export function clipJointsAt(clip, t, params) {
  if (!clip.joints) return null;
  const p = params || defaults(clip);
  const d = clipDur(clip, p);
  const tc = Math.max(0, Math.min(d, t));
  const j = clip.joints(tc, p);
  if (!j) return null;
  if (clip.kind === 'xfade') return j;
  const a = anchorAt(clip, tc, p);
  const out = {};
  for (const k in j) {
    const rest = REST[k] ?? 0;
    out[k] = clampJoint(k, rest + (j[k] - rest) * a);
  }
  return out;
}

// Clamp to the MJCF's own limit. A clip that asks for more is a bug, but the
// bug should not reach a servo: `inspect` asked the LEFT elbow for 1.6 rad
// against a 1.542 stop for months, and MuJoCo silently clamped it while the
// real arm would have pushed.
export function clampJoint(k, v) {
  const r = RANGES[k];
  return r ? Math.max(r[0], Math.min(r[1], v)) : v;
}
export function clipNavAt(clip, t, params) {
  if (!clip.nav) return null;
  const d = clipDur(clip, params);
  return clip.nav(Math.max(0, Math.min(d, t)), params || defaults(clip));
}

// merge a backend /api/actions list into the local registry: unknown ids are
// added as (non-executable ⇒ coming-soon) slots; known ids only adopt backend
// metadata that can't break execution (title/emoji/location).
export function mergeActions(registry, backendActions) {
  const out = { ...registry };
  for (const a of backendActions || []) {
    if (!a || !a.id) continue;
    const local = out[a.id];
    if (local) {
      out[a.id] = { ...local,
        title: a.title || local.title,
        emoji: a.emoji || local.emoji,
        location: a.location || local.location };
    } else {
      out[a.id] = {
        title: a.title || a.id, emoji: a.emoji || '🤖',
        dur: a.dur || 10, lane: a.lane === 'nav' || a.lane === 'both' ? a.lane : 'manip',
        category: a.category === 'nav' || a.category === 'pose' ? a.category : 'policy',
        status: 'coming_soon',                 // nothing local can execute it
        location: a.location || null,          // may be label-only from the backend
        note: a.note || null,
        joints: null, nav: null,
      };
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ChoreoTrack — the NLE block model. Blocks reference clips by key and carry
// {t0 (timeline s), in/out (clip-local sub-range), params, lane, linkId}.
// A cut block inherits the parent's clip key ⇒ location metadata composes.
// ─────────────────────────────────────────────────────────────────────────────
export const SNAP_GRID = 0.1;                 // deterministic 100 ms grid

let _nextId = 1;
const nid = () => 'b' + (_nextId++);

export class ChoreoTrack {
  constructor(registry) {
    this.registry = registry || MOTION_CLIPS;
    this.blocks = [];                          // sorted by t0 per lane at render
    this.onChange = null;
  }
  _emit() { this.onChange && this.onChange(); }
  clip(b) { return this.registry[b.key]; }

  // add clip at t0 → one block ('manip'/'nav') or two linked blocks ('both')
  add(key, t0, params) {
    const clip = this.registry[key];
    if (!clip || clip.status !== 'available') return null;
    const p = params || defaults(clip);
    const d = clipDur(clip, p);
    const mk = (lane, linkId) => {
      const b = { id: nid(), key, lane, t0: Math.max(0, t0),
                  in: 0, out: d, params: { ...p }, linkId: linkId || null };
      this.blocks.push(b);
      return b;
    };
    let made;
    if (clip.lane === 'both') {
      const link = nid();
      made = [mk('manip', link), mk('nav', link)];
    } else {
      made = [mk(clip.lane === 'nav' ? 'nav' : 'manip', null)];
    }
    this._emit();
    return made;
  }

  remove(id) {
    const i = this.blocks.findIndex((b) => b.id === id);
    if (i >= 0) { this.blocks.splice(i, 1); this._emit(); return true; }
    return false;
  }

  move(id, t0) {
    const b = this.byId(id);
    if (b) { b.t0 = Math.max(0, t0); this._emit(); }
  }

  byId(id) { return this.blocks.find((b) => b.id === id) || null; }
  len(b) { return b.out - b.in; }
  end(b) { return b.t0 + this.len(b); }

  // blade: split block at absolute time t → two blocks, each playing only its
  // slice of the clip (in/out offsets partition the parent's range)
  cut(id, t) {
    const b = this.byId(id);
    if (!b) return null;
    // A blend is refused: half a blend is a NEW pose jump. Blading one would
    // leave two fragments that each cross part of the gap and a discontinuity
    // at the cut — turning the guard into the hazard it exists to prevent.
    if (this.clip(b)?.kind === 'xfade') return null;
    const local = t - b.t0;
    if (local <= 0.05 || local >= this.len(b) - 0.05) return null;   // no slivers
    const right = { id: nid(), key: b.key, lane: b.lane, t0: b.t0 + local,
                    in: b.in + local, out: b.out, params: { ...b.params },
                    linkId: b.linkId };
    b.out = b.in + local;
    this.blocks.push(right);
    this._emit();
    return right;
  }

  // trim an edge ('l'|'r') to absolute time t; clamped to the clip's duration
  // (hold clips may extend the right edge arbitrarily — the pose holds)
  trim(id, edge, t) {
    const b = this.byId(id);
    if (!b) return;
    const clip = this.clip(b);
    const dMax = clipDur(clip, b.params);
    // ── blends trim by their LEFT edge only, and never below their safety
    // minimum. The right edge is anchored to the jump it guards: moving it
    // would reopen the discontinuity, so 'r' is a no-op. A blend is also not a
    // slice of a fixed-length source — its duration IS params.dur — so
    // lengthening it means growing `dur`, not indexing further into a clip
    // (the generic path below clamps `in` to ≥ 0 and so could not stretch one
    // at all). An operator-chosen duration is recorded as `userDur` and
    // survives reconciliation; the `min` floor is not negotiable.
    if (clip?.kind === 'xfade') {
      if (edge !== 'l') return;
      const end = this.end(b);
      const min = Math.max(BLEND_MIN_DUR, b.params?.min || 0);
      const dur = Math.max(min, end - t);
      b.t0 = Math.max(0, end - dur);
      b.in = 0;
      b.out = dur;
      b.params = { ...b.params, dur, userDur: dur };
      this._emit();
      return;
    }
    if (edge === 'l') {
      const newIn = b.in + (t - b.t0);
      const lo = 0, hi = b.out - 0.1;
      const v = Math.max(lo, Math.min(hi, newIn));
      b.t0 += v - b.in;
      b.in = v;
    } else {
      const newOut = b.in + Math.max(0.1, t - b.t0);
      b.out = clip?.hold ? newOut : Math.min(dMax, newOut);
    }
    this._emit();
  }

  duplicate(id, t0) {
    const b = this.byId(id);
    if (!b) return null;
    const c = { ...b, params: { ...b.params }, id: nid(), linkId: null,
                t0: t0 != null ? Math.max(0, t0) : this.end(b) + SNAP_GRID };
    this.blocks.push(c);
    this._emit();
    return c;
  }

  // deterministic snapping: 0.1 s grid always; block edges + playhead + t=0
  // override the grid when within tol
  snapTime(t, { excludeId = null, playhead = null, tol = 0.12 } = {}) {
    let best = Math.round(t / SNAP_GRID) * SNAP_GRID;
    let bd = tol;
    const cands = [0];
    if (playhead != null) cands.push(playhead);
    for (const b of this.blocks) {
      if (b.id === excludeId) continue;
      cands.push(b.t0, this.end(b));
    }
    for (const c of cands) {
      const d = Math.abs(c - t);
      if (d < bd) { bd = d; best = c; }
    }
    return Math.max(0, +best.toFixed(4));
  }

  lane(name) {
    return this.blocks.filter((b) => b.lane === name).sort((a, c) => a.t0 - c.t0);
  }
  duration() {
    return this.blocks.reduce((m, b) => Math.max(m, this.end(b)), 0);
  }
  // `skipXfade` ignores auto blends, so a caller can ask what the timeline
  // would be commanding if the blend were not there — which is exactly the
  // question the pose-jump detector needs answered. See jointsAt.
  activeAt(t, laneName, { skipXfade = false } = {}) {
    // last-starting active block wins overlaps (top-most in an NLE)
    let win_ = null;
    for (const b of this.blocks) {
      if (laneName && b.lane !== laneName) continue;
      const clip = this.clip(b);
      if (skipXfade && clip?.kind === 'xfade') continue;

      const holdEnd = clip?.hold ? Infinity : this.end(b);
      if (t >= b.t0 && t < Math.min(this.end(b), holdEnd)) {
        if (!win_ || b.t0 >= win_.t0) win_ = b;
      }
    }
    return win_;
  }

  // choreography sampling: joints from the active manip-lane block (plus any
  // nav-lane block whose clip carries joints, e.g. lift), nav from the active
  // nav-lane block. Returns null when nothing is active.
  // `skipXfade` looks THROUGH auto blends to the pose that actually precedes
  // them. `samplePoseBefore` has always passed it; this method used to take
  // only `t` and silently ignore it, which made the pose-jump guard
  // self-erasing: once a blend was inserted, sampling just before the jump hit
  // the BLEND's end pose (equal to the next block's start by construction), the
  // jump measured ~0, and the very next reconcile deleted the blend that had
  // fixed it. The reconcile could therefore never converge — `autoFixPoseJumps`
  // was not idempotent, and a timeline would oscillate between guarded and
  // unguarded on successive edits.
  // The commanded pose at t: the lanes merge, and then a BRIDGE, if one is
  // running, is laid over the result.
  //
  // A bridge owns the pose while it runs. It has to: it is the one object on the
  // timeline whose whole job is to be continuous with what is either side of it,
  // and a lane that overrides half of it breaks exactly that. Letting the nav
  // lane keep the LIFT through a bridge (lift_up is lane 'nav' but commands
  // joints) meant the bridge never actually moved the lift — and the moment the
  // lift block ended, the pose underneath was revealed in one tick. Measured
  // 19.9 rad/s at the cut point of `lift_up`, eight times the hard cap, on 178
  // of 3600 cut-then-butt pairs.
  jointsAt(t, { skipXfade = false } = {}) {
    let out = null;
    for (const laneName of ['manip', 'nav']) {
      const b = this.activeAt(t, laneName, { skipXfade: true });
      if (!b) continue;
      const clip = this.clip(b);
      if (!clip?.joints) continue;
      if (clip.lane === 'both' && b.lane === 'nav') continue;   // joints ride manip twin
      const j = clipJointsAt(clip, b.in + (t - b.t0), b.params);
      if (j) out = out ? { ...out, ...j } : j;
    }
    if (!skipXfade) {
      const x = this.activeXfadeAt(t);
      const j = x && clipJointsAt(this.clip(x), x.in + (t - x.t0), x.params);
      if (j) out = out ? { ...out, ...j } : j;
    }
    return out;
  }

  /// The bridge running at t, latest-starting first — across lanes, because a
  /// bridge is not a lane's business while it runs.
  activeXfadeAt(t) {
    let win_ = null;
    for (const b of this.blocks) {
      if (this.clip(b)?.kind !== 'xfade') continue;
      if (t >= b.t0 && t < this.end(b) && (!win_ || b.t0 >= win_.t0)) win_ = b;
    }
    return win_;
  }
  navAt(t) {
    const b = this.activeAt(t, 'nav');
    if (!b) return null;
    const clip = this.clip(b);
    if (!clip?.nav) return null;
    if (clip.lane === 'both' && b.lane === 'manip') return null;
    return clipNavAt(clip, b.in + (t - b.t0), b.params);
  }
  // REST-backed goal blocks active at t (kind:'goal')
  goalAt(t) {
    const b = this.activeAt(t, 'nav');
    return b && this.clip(b)?.kind === 'goal' ? b : null;
  }

  toJSON() {
    return { blocks: this.blocks.map((b) => ({ ...b, params: { ...b.params } })) };
  }

  // paste a serialized track INTO this one, shifted by dt (skill import /
  // compose-and-reuse): fresh ids, preserved lane/in/out/params/links
  paste(json, dt = 0) {
    const linkMap = new Map();
    const made = [];
    for (const b of json?.blocks || []) {
      if (!this.registry[b.key]) continue;              // unknown clip — skip
      let link = null;
      if (b.linkId) {
        if (!linkMap.has(b.linkId)) linkMap.set(b.linkId, nid());
        link = linkMap.get(b.linkId);
      }
      const nb = { id: nid(), key: b.key,
        lane: b.lane === 'nav' ? 'nav' : 'manip',
        t0: Math.max(0, (+b.t0 || 0) + dt),
        in: Math.max(0, +b.in || 0),
        out: +b.out || clipDur(this.registry[b.key], b.params),
        params: { ...(b.params || {}) }, linkId: link };
      this.blocks.push(nb);
      made.push(nb);
    }
    if (made.length) this._emit();
    return made;
  }

  static fromJSON(j, registry) {
    const tr = new ChoreoTrack(registry);
    for (const b of j?.blocks || []) {
      if (!tr.registry[b.key]) continue;                 // clip vanished — drop
      tr.blocks.push({ id: b.id || nid(), key: b.key,
        lane: b.lane === 'nav' ? 'nav' : 'manip',
        t0: Math.max(0, +b.t0 || 0), in: Math.max(0, +b.in || 0),
        out: +b.out || clipDur(tr.registry[b.key], b.params),
        params: { ...(b.params || {}) }, linkId: b.linkId || null });
      const n = parseInt(String(b.id || '').slice(1), 10);
      if (n >= _nextId) _nextId = n + 1;
    }
    return tr;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Location-aware gap validation. Walks all blocks in time order tracking the
// robot's implied location; a block whose clip declares a `location` different
// from the tracked one, with no bridging goal-nav block in between, raises a
// gap. Location-less actions are location-agnostic. kind:'goal' blocks SET the
// tracked location from their params (+ optional label).
// Returns [{t, beforeBlockId, from:{x,y,label}|null, to:{x,y,label}}].
// ─────────────────────────────────────────────────────────────────────────────
export function detectGaps(track, startLoc = null) {
  const ordered = [...track.blocks].sort((a, b) => a.t0 - b.t0);
  let loc = startLoc;                                    // {x,y,label} | null
  const gaps = [];
  for (const b of ordered) {
    const clip = track.clip(b);
    if (!clip) continue;
    if (clip.kind === 'goal') {
      loc = { x: +(b.params?.x ?? 0), y: +(b.params?.y ?? 0),
              label: b.params?.label || `(${(+(b.params?.x ?? 0)).toFixed(1)}, ${(+(b.params?.y ?? 0)).toFixed(1)})` };
      continue;
    }
    const need = clip.location;
    if (!need) continue;                                 // location-agnostic
    // label-only locations (backend) compare by label; x,y compare by distance
    const at = loc && (need.x == null || need.y == null
      ? (loc.label && need.label && loc.label === need.label)
      : Math.hypot((loc.x ?? 0) - need.x, (loc.y ?? 0) - need.y) < 0.35);
    if (!at) {
      gaps.push({ t: b.t0, beforeBlockId: b.id, from: loc, to: { ...need } });
      loc = { ...need };                                 // assume ONE fix per hop
    } else {
      loc = { ...need };
    }
  }
  return gaps;
}

// auto-fix: insert a goto block on the nav lane ending at the gap, sized by a
// distance / NAV_SPEED estimate (+ settle margin). Returns inserted blocks.
export function autoFixGaps(track, startLoc = null) {
  const inserted = [];
  for (const gap of detectGaps(track, startLoc)) {
    if (gap.to.x == null || gap.to.y == null) continue;  // label-only: not mappable
    const from = gap.from || startLoc || { x: 0, y: 0 };
    const dist = Math.hypot((from.x ?? 0) - gap.to.x, (from.y ?? 0) - gap.to.y);
    const est = Math.max(2, dist / (NAV_SPEED * 0.5) + 2);   // travel + settle
    const t0 = Math.max(0, gap.t - est);
    const made = track.add('nav_goto', t0,
      { x: gap.to.x, y: gap.to.y, label: gap.to.label });
    if (made) {
      const b = made[0];
      b.out = b.in + est;                                // block spans the estimate
      inserted.push(b);
    }
  }
  if (inserted.length) track._emit();
  return inserted;
}

// Per-joint weights converting a joint's OWN units into radian-equivalents, so
// one threshold (BLEND_THRESH) and one velocity cap (SAFE_BLEND_VEL) can grade
// every joint. Anything absent weighs 1 — i.e. it is already in radians, which
// covers every revolute joint on the robot.
//
// Like BLEND_MIN_DUR this was referenced by poseDelta() but never declared, so
// the whole pose-jump guard threw a ReferenceError from its first line. The two
// non-radian cases:
//   · {side}_grip ∈ [0,1] curls the whole ORCA hand — GRIP_SCALE rad of finger
//     travel per unit, already measured above.
//   · lift_{lower,upper} are PRISMATIC (metres, 0…0.318 each). There is no
//     physically exact metres→radians conversion; this weight only has to make
//     the lift's contribution comparable to an arm sweep. 10 rad/m maps the
//     full 0.318 m stroke onto ~3.2 rad ≈ a large arm move. It is deliberately
//     on the HIGH side: over-weighting makes a blend more likely, which is the
//     safe direction to be wrong in.
// TODO(hardware): confirm the lift weight against a measured lift rate.
// weighted worst-joint pose distance (radian-equivalents; see BLEND_SCALE)
export function poseDelta(from, to) {
  let delta = 0, joint = null;
  for (const k in to) {
    if (from[k] == null) continue;               // unknown start → cannot jump
    const d = Math.abs(to[k] - from[k]) * (BLEND_SCALE[k] || 1);
    if (d > delta) { delta = d; joint = k; }
  }
  return { delta, joint };
}

// Minimum safe duration for a given jump, so the PEAK velocity of the eased
// interpolation (EASE_PEAK × the average) stays under SAFE_BLEND_VEL. Rounded
// UP — a safety floor must never round down.
// A full 0.635 m stroke at LIFT_MAX_VEL takes 18.1 s, and an eased blend peaks at
// EASE_PEAK times its average, so it needs ~29 s. Under an 8 s cap the same blend was
// compressed back to ~125 mm/s, 3.6x what the column can do.
export const LIFT_BLEND_MAX_DUR = Math.ceil(EASE_PEAK * 0.635 / LIFT_MAX_VEL);   // 29 s
const LIFT_JOINTS = new Set(['lift_lower', 'lift_upper']);

export function blendMinDur(delta, joint = null) {
  const cap = LIFT_JOINTS.has(joint) ? LIFT_BLEND_MAX_DUR : 8;
  return Math.max(BLEND_MIN_DUR,
    Math.min(cap, Math.ceil(100 * EASE_PEAK * delta / SAFE_BLEND_VEL) / 100));
}

// ─────────────────────────────────────────────────────────────────────────────
// THE POSE-JUMP GUARD
//
// The executor commands a pose every tick: the latest-starting active block
// wins, and across a dead gap the last pose HOLDS. A discontinuity in that
// commanded signal is a step the servos answer with full torque, so the guard's
// job is to find every one and bridge it.
//
// THE RULE OF THUMB, and it is the whole specification: whatever an operator
// composes, the commanded pose is continuous. Not "each clip is safe" (that is a
// per-clip property the library already has) and not "this junction is safe"
// (that is one case) — every junction, at the rate the robot receives.
// Verified over 14,400 compositions: every clip cut in half, followed by every
// clip, at four gaps. 13,808 bridges inserted, worst commanded 2.50 rad/s,
// none over the budget (tools/exp_pose_blend.py).
//
// FOUR THINGS THIS GOT WRONG BEFORE, all measured, all found by composing
// rather than by reading:
//
//   1. It only looked at block STARTS. A block ENDING is just as much a
//      boundary: when it stops, whatever was running underneath takes the
//      joints back mid-motion. On the studio's own demo timeline —
//      `pickup` [0,8] with `pose_tall` [2.5,6.2] layered over it — the handover
//      at 6.2 commanded 31.75 rad/s, 13x the clip budget, and the detector
//      reported no jump at all because no block started there.
//
//   2. It sampled the blend's `from` at the JUMP, not at the blend's own start.
//      A blend that begins inside the outgoing clip then takes over from a pose
//      that clip is not in yet, which is a step at the blend's leading edge —
//      the guard manufacturing the hazard it exists to prevent. Avoiding the
//      overlap instead (the previous fix) only moved the problem: with no room
//      the blend was squeezed to its 0.4 s floor and ran at 5.09 rad/s.
//
//   3. It solved the LENGTH for the gentle speed unconditionally, which meant
//      reaching back into the outgoing clip for as long as it took, and it
//      iterated that length as a fixed point — which has a degenerate solution.
//      See the sizing block in `detectPoseJumps`.
//
//   4. It reasoned about "the manipulation lane" rather than "blocks that
//      command joints", and it let other lanes override a running bridge. See
//      `poseBlocks` and `jointsAt`.
//
// Sampling `from` at the blend's own start dissolves the first two. A blend is
// then continuous at BOTH edges by construction wherever it is placed, because
// each endpoint is read off the very signal it is splicing into, and it may
// freely pre-empt the tail of the clip it is leaving. The only placement rule
// left is that a blend may not swallow another block's START — that block would
// out-rank it (later t0 wins) and cut it off mid-interpolation.
// ─────────────────────────────────────────────────────────────────────────────

// The pose the executor commands at time t, blends excluded. keyPoseAt = the
// keyframe lane's poseAt (block joints win the merge, exactly like the
// executor). In a dead gap the last activity's final pose holds.
export function samplePoseAt(track, keyPoseAt, t) {
  const at = (tt) => {
    const bj = track.jointsAt(Math.max(0, tt), { skipXfade: true });
    const kf = keyPoseAt ? keyPoseAt(Math.max(0, tt)) : null;
    return (bj || kf) ? { ...(kf || {}), ...(bj || {}) } : null;
  };
  const direct = at(t);
  if (direct) return direct;
  let last = null;                                // latest activity end before t
  for (const b of track.blocks) {
    const clip = track.clip(b);
    if (!clip?.joints || clip.kind === 'xfade') continue;
    if (clip.lane === 'both' && b.lane === 'nav') continue;
    const e = track.end(b);
    if (e <= t + 1e-6 && (last == null || e > last)) last = e;
  }
  return last != null ? at(last - 1e-3) : null;
}

// Kept as the name the studio and the tests already import.
export function samplePoseBefore(track, keyPoseAt, t) {
  return samplePoseAt(track, keyPoseAt, t - 1e-3);
}

// Every block that CONTRIBUTES JOINTS to the commanded pose — the same filter
// `samplePoseAt` merges with, not "the manipulation lane".
//
// Lane is the wrong test and it cost a real defect. `lift_up` and `lift_down`
// live on the NAV lane but they command joints, so a lift block was invisible
// to the guard: it did not count as activity when measuring the dead gap, and
// its edges were not boundaries. A bridge sized against a gap that ignored a
// running lift therefore reached back INSIDE it, the nav lane kept overriding
// the lift for that overlap, and the moment the lift block ended the bridge's
// own (quite different) lift value was revealed in one tick — measured
// 25.2 rad/s, ten times the hard cap, on `lift_up` followed by `wave`.
function poseBlocks(track) {
  return track.blocks.filter((b) => {
    const clip = track.clip(b);
    if (!clip?.joints || clip.kind === 'xfade') return false;
    if (clip.lane === 'both' && b.lane === 'nav') return false;  // joints ride the twin
    return true;
  });
}

// The earliest block START strictly inside (a, b) — a blend may not span one,
// because that block starts later and would out-rank the blend halfway through.
function startInside(track, a, b) {
  let hit = null;
  for (const blk of poseBlocks(track))
    if (blk.t0 > a + 1e-6 && blk.t0 < b - 1e-6 && (hit == null || blk.t0 < hit))
      hit = blk.t0;
  return hit;
}

// Every discontinuity in the commanded pose, at both edges of every block:
// [{id, blockId, edge, t, t0, t1, delta, joint, minDur, dur, tight, from, to}]
// `edge` is 'lead' (the blend ENDS at the boundary, so the block plays from its
// own first tick) or 'tail' (the blend STARTS there, easing off a pose the
// block has just stopped commanding into whatever took over).
export function detectPoseJumps(track, keyPoseAt = null) {
  const jumps = [];
  const seen = new Set();
  const blocks = poseBlocks(track).sort((a, b) => a.t0 - b.t0);
  // Boundaries: a start is a 'lead' (the arriving block owns the instant), an
  // end is a 'tail'. A start and an end at the SAME instant is one boundary,
  // and the arriving block wins it — hence starts are collected first.
  const bounds = [];
  for (const b of blocks) bounds.push({ t: b.t0, edge: 'lead', block: b });
  for (const b of blocks) {
    const e = track.end(b);
    if (!blocks.some((o) => Math.abs(o.t0 - e) < 1e-6))
      bounds.push({ t: e, edge: 'tail', block: b });
  }
  bounds.sort((a, b) => a.t - b.t);

  for (const { t, edge, block } of bounds) {
    const key = t.toFixed(4);
    if (seen.has(key)) continue;
    seen.add(key);
    const before = samplePoseAt(track, keyPoseAt, t - 1e-3);
    const after = samplePoseAt(track, keyPoseAt, t + 1e-3);
    if (!before || !after) continue;             // nothing on one side → no jump
    const probe = poseDelta(before, after);
    if (probe.delta < BLEND_THRESH) continue;

    // ── SIZING ──────────────────────────────────────────────────────────
    // The bridge is continuous whatever its length (its endpoints are read off
    // the signal at its own edges), so length is not a correctness question —
    // it is a SPEED question, and a question of what the bridge is allowed to
    // eat. Two rules, in this order:
    //
    //   1. NEVER exceed MAX_JOINT_VEL. Hard.
    //   2. Prefer SAFE_BLEND_VEL, and prefer to live in the DEAD GAP.
    //
    // Rule 2 is what the first version got backwards. It solved for the gentle
    // speed unconditionally, which meant reaching back into the outgoing clip
    // for as long as it took — on `wave` cut at 2.5 s with `happy` at 4 s it
    // produced a 3.15 s bridge starting at 0.85 s, erasing five sixths of the
    // wave to save 1.4 rad/s. Destroying authored motion to hit a COMFORT
    // target is the wrong trade; running the gap a little quicker is not.
    //
    // It also iterated the length as a fixed point, which has a degenerate
    // solution: reach back far enough and the outgoing clip is at REST there,
    // so the measured distance collapses to zero and the guard concludes no
    // bridge is needed at all. That is exactly what happened when a clip was
    // cut and the next one butted straight on: no blend, 40 rad/s. Searching
    // UPWARD for the SHORTEST adequate length cannot land on it.
    const gap = edge === 'lead'
      ? t - prevActivityEnd(track, t)
      : (nextActivityStart(track, t) ?? Infinity) - t;
    // How far the bridge would have to reach to be safe if the pose it starts
    // from stayed put (it does, across a dead gap: the executor holds).
    const want = blendMinDur(probe.delta);
    // A bridge may not span another block's START: that block begins later and
    // out-ranks it halfway through.
    const ceiling = edge === 'lead'
      ? t - (startInside(track, 0, t) ?? -Infinity)
      : (startInside(track, t, Infinity) ?? Infinity) - t;

    const spanAt = (T) => {
      const t0 = edge === 'lead' ? Math.max(0, t - T) : t;
      const a = edge === 'lead' ? samplePoseAt(track, keyPoseAt, t0 - 1e-3) : before;
      const b = edge === 'lead' ? after
        : samplePoseAt(track, keyPoseAt, t + T + 1e-3);
      if (!a || !b) return null;
      const d = poseDelta(a, b);
      return { t0, from: a, to: b, delta: d.delta, joint: d.joint,
               peak: d.delta * easedPeakRate(T) };
    };

    // Start inside the gap — all of it, or only as much as the gentle speed
    // needs — then grow only while the HARD cap is still broken.
    let dur = Math.max(BLEND_MIN_DUR, Math.min(want, Math.max(gap, 0)));
    dur = Math.min(dur, ceiling);
    let span = spanAt(dur);
    if (span && span.peak > MAX_JOINT_VEL) {
      for (let T = dur + SNAP_GRID; T <= Math.min(ceiling, BLEND_MAX_DUR) + 1e-9;
           T += SNAP_GRID) {
        const s = spanAt(T);
        if (!s) break;
        dur = T; span = s;
        if (s.peak <= MAX_JOINT_VEL) break;
      }
    }
    if (!span) continue;
    // NOT `if (span.delta < BLEND_THRESH) continue`. A short span is a CHEAP
    // bridge, not an unnecessary one: the boundary jump is real either way, and
    // the bridge is what removes it — partly by interpolating, partly by
    // pre-empting the outgoing clip over its own span. Skipping on the span's
    // distance let the knife-edge cases through, where the jump sits exactly on
    // the threshold and the span measures a hair under it (`pose_tuck` moves a
    // shoulder by exactly BLEND_THRESH): 58 of 3600 cut-then-butt pairs stepped
    // the whole jump in one tick with no bridge at all.
    if (span.delta < 1e-6) continue;             // a literal no-op, nothing to draw

    const t0 = span.t0, t1 = t0 + dur;
    jumps.push({ id: `${block.id}:${edge}`, blockId: block.id, edge,
                 t, t0, t1, delta: span.delta, joint: span.joint,
                 minDur: dur, dur, peak: span.peak,
                 // TIGHT = the bridge could not be made gentle, because the gap
                 // is short and lengthening it would eat the clip. It is still
                 // inside the hard cap; the operator can drag its leading edge
                 // to trade clip for smoothness, which is theirs to decide.
                 tight: span.peak > SAFE_BLEND_VEL + 1e-6,
                 room: ceiling, want,
                 from: span.from, to: span.to });
  }
  return jumps;
}

// End of the last thing that actually commanded the arms at or before t, and
// the start of the next one after it (blends excluded — a blend is not an
// activity, it is the transition into one).
function prevActivityEnd(track, t) {
  let last = 0;
  for (const b of poseBlocks(track)) {
    const e = track.end(b);
    if (e <= t + 1e-6 && e > last) last = e;
  }
  return last;
}
function nextActivityStart(track, t) {
  let next = null;
  for (const b of poseBlocks(track))
    if (b.t0 > t + 1e-6 && (next == null || b.t0 < next)) next = b.t0;
  return next;
}

// reconcile the auto blends with the current jump set — insert / re-anchor /
// retarget / remove, deterministically.
// Returns {inserted, updated, removed, tight}.
export function autoFixPoseJumps(track, keyPoseAt = null) {
  const out = { inserted: [], updated: [], removed: [], tight: [] };
  const jumps = detectPoseJumps(track, keyPoseAt);
  const jumpFor = new Map(jumps.map((j) => [j.id, j]));
  const seen = new Set();
  // drop auto blends whose jump vanished (or duplicated ones for the same jump)
  for (const b of [...track.blocks]) {
    if (track.clip(b)?.kind !== 'xfade' || !b.params?.auto) continue;
    const forId = b.params.forKey || b.params.forId;
    if (!jumpFor.has(forId) || seen.has(forId)) {
      track.blocks.splice(track.blocks.indexOf(b), 1);
      out.removed.push(b.id);
    } else {
      seen.add(forId);
    }
  }
  for (const j of jumps) {
    const existing = track.blocks.find((b) => track.clip(b)?.kind === 'xfade'
      && b.params?.auto && (b.params.forKey || b.params.forId) === j.id);
    // A duration the operator set by dragging the blend's edge is theirs to
    // keep, as long as it still fits between the blocks around it.
    const dur = Math.max(j.dur, Math.min(existing?.params?.userDur || 0, j.room));
    const t0 = j.edge === 'lead' ? Math.max(0, j.t - dur) : j.t;
    if (existing) {
      const p = existing.params;
      const same = Math.abs(existing.t0 - t0) < 1e-3
        && Math.abs((existing.out - existing.in) - dur) < 1e-3
        && JSON.stringify(p.to) === JSON.stringify(j.to)
        && JSON.stringify(p.from) === JSON.stringify(j.from);
      if (!same) {
        existing.t0 = t0; existing.in = 0; existing.out = dur;
        existing.params = { ...p, dur, min: j.minDur, from: j.from, to: j.to,
                            delta: j.delta, joint: j.joint, edge: j.edge };
        out.updated.push(existing.id);
      }
    } else {
      track.blocks.push(makeBlendBlock(j, dur));
      out.inserted.push(j.id);
    }
    if (j.tight) out.tight.push({ blockId: j.blockId, edge: j.edge, t: j.t,
                                  room: j.room, want: j.want, joint: j.joint });
  }
  return out;
}

// blend block for a detected jump (shared by autofix and the ⚡ marker click)
export function makeBlendBlock(j, dur = null) {
  // Never shorter than the duration the detector sized for this placement.
  // (`j.dur` is `j.minDur` clamped so the blend does not swallow another
  // block's start; flooring at `minDur` instead would reintroduce exactly
  // the overlap the clamp exists to prevent.)
  const d = Math.max(BLEND_MIN_DUR, j.dur ?? j.minDur, dur || 0);
  const t0 = j.edge === 'tail' ? j.t : Math.max(0, j.t - d);
  return { id: nid(), key: 'blend', lane: 'manip',
    t0, in: 0, out: d, linkId: null,
    params: { dur: d, min: j.minDur, from: j.from, to: j.to,
              forId: j.blockId, forKey: j.id, edge: j.edge,
              auto: true, delta: j.delta, joint: j.joint,
              // the sizing verdict, so the timeline can draw a squeezed bridge
              // differently and the inspector can explain it
              peak: j.peak, tight: j.tight, want: j.want, room: j.room } };
}

// grip expansion: the physics-preview adapter maps canonical ids only, so a
// scalar {side}_grip is expanded into explicit finger-joint targets via the
// firmware grasp table (handmodel.js) before hitting that wire.
import { graspPose, poseToCommand } from './handmodel.js';
export function expandGrips(pose) {
  const out = {};
  for (const [k, v] of Object.entries(pose || {})) {
    const m = k.match(/^(left|right)_grip$/);
    if (!m) { out[k] = v; continue; }
    const { name, position } = poseToCommand(m[1], graspPose(v));
    name.forEach((n, i) => {
      if (!n.endsWith('_wrist')) out[n] = position[i];   // grip must not steer the wrist
    });
  }
  return out;
}

// progress for the preview status chip: how many blocks have started / total
export function blockProgress(track, t) {
  const blocks = track.blocks;
  const started = blocks.filter((b) => t >= b.t0).length;
  return { n: Math.min(started, blocks.length), m: blocks.length };
}

// ─────────────────────────────────────────────────────────────────────────────
// Preview base-path integration: turn the nav lane into a world-frame base
// trajectory for the 3D preview (Δpose from the preview's start pose).
// Wire signs (measured): fwd = 0.80·ly m/s along (−cosθ, −sinθ), left =
// 0.87·lx m/s along (sinθ, −cosθ), ω = 1.50·rx rad/s (+ = CCW).
// ─────────────────────────────────────────────────────────────────────────────
export const NAV_SPEED_STRAFE = 0.87;
export function integrateNavPath(track, dur, dt = 0.05) {
  const n = Math.max(1, Math.round(dur / dt));
  const out = new Array(n + 1);
  let x = 0, y = 0, th = 0;
  out[0] = { dx: 0, dy: 0, dyaw: 0 };
  for (let i = 1; i <= n; i++) {
    const nav = track.navAt((i - 1) * dt) || { lx: 0, ly: 0, rx: 0 };
    const vf = NAV_SPEED * (nav.ly || 0);
    const vl = NAV_SPEED_STRAFE * (nav.lx || 0);
    x += (-Math.cos(th) * vf + Math.sin(th) * vl) * dt;
    y += (-Math.sin(th) * vf - Math.cos(th) * vl) * dt;
    th += NAV_YAW_RATE * (nav.rx || 0) * dt;
    out[i] = { dx: x, dy: y, dyaw: th };
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnnoTrack — the language / annotation lane: free-text spans {id, t0, dur,
// text}. Action blocks additionally render DERIVED labels (the clip title)
// straight from the ChoreoTrack, so predefined actions always carry their
// language text without duplication.
// ─────────────────────────────────────────────────────────────────────────────
export class AnnoTrack {
  constructor() {
    this.notes = [];                    // {id, t0, dur, text}
    this.onChange = null;
  }
  _emit() { this.onChange && this.onChange(); }
  add(t0, dur, text) {
    const n = { id: 'a' + (_nextId++), t0: Math.max(0, t0),
                dur: Math.max(0.2, dur), text: String(text || '') };
    this.notes.push(n);
    this._emit();
    return n;
  }
  move(id, t0) {
    const n = this.byId(id);
    if (n) { n.t0 = Math.max(0, t0); this._emit(); }
  }
  retime(id, dur) {
    const n = this.byId(id);
    if (n) { n.dur = Math.max(0.2, dur); this._emit(); }
  }
  setText(id, text) {
    const n = this.byId(id);
    if (n) { n.text = String(text || ''); this._emit(); }
  }
  remove(id) {
    const i = this.notes.findIndex((n) => n.id === id);
    if (i >= 0) { this.notes.splice(i, 1); this._emit(); return true; }
    return false;
  }
  byId(id) { return this.notes.find((n) => n.id === id) || null; }
  at(t) { return this.notes.filter((n) => t >= n.t0 && t < n.t0 + n.dur); }
  duration() { return this.notes.reduce((m, n) => Math.max(m, n.t0 + n.dur), 0); }
  toJSON() { return { notes: this.notes.map((n) => ({ ...n })) }; }
  static fromJSON(j) {
    const tr = new AnnoTrack();
    for (const n of j?.notes || []) {
      tr.notes.push({ id: n.id || 'a' + (_nextId++), t0: Math.max(0, +n.t0 || 0),
                      dur: Math.max(0.2, +n.dur || 1), text: String(n.text || '') });
      const num = parseInt(String(n.id || '').slice(1), 10);
      if (num >= _nextId) _nextId = num + 1;
    }
    return tr;
  }
}


// unit-testable. Absolute joint values like every other clip, and like every
// other clip it sits on REST: the arms hang with the elbows extended and the
// right one lifts out of that to wave, rather than idling at qpos zero with
// both forearms stuck out in front.
export function offlineIdlePose(t){
  const S = Math.sin;
  const wave = Math.max(0, S(t * 0.9));          // 0..1, one-sided → a wave, not a swing
  return {
    q: {
      ...REST,
      neck_yaw:    0.5 * S(t * 0.45),
      neck_pitch:  0.12 * S(t * 0.33 + 1),
      torso:       0.10 * S(t * 0.28),
      lift_lower:  0.15875 + 0.055 * S(t * 0.22), // gentle rise/fall around rest
      lift_upper:  0.15875 + 0.055 * S(t * 0.22),
      right_arm_1: 1.55 * wave,                    // raise the whole arm…
      right_arm_2: 0.1711 - 0.45 * wave,
      right_arm_4: -1.52 + 0.75 * wave,            // …folding the elbow as it goes
      right_wrist: -0.15 + 0.45 * S(t * 3.0) * wave,  // …and wave the hand
      left_arm_2:  -0.1711 + 0.08 * S(t * 0.4),    // subtle left-side sway
    },
    yaw: 0.22 * S(t * 0.11),                       // slow base yaw so all sides show
  };
}
