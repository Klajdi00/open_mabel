// The Home / rest pose and its glide math — pure, node-testable.
//
// REST_POSE is the minimal-holding-torque posture derived from the MJCF as the
// arms' gravity equilibrium: both arms hang at the sides with the elbows
// EXTENDED, torso and neck neutral, swerve wheels facing forward, hands in a
// gentle relaxed curl. Re-derived 2026-09 by
// controller/experiments/exp_rest_pose.py, which minimises Στ² under
// mj_inverse over the symmetric (shoulder-roll, elbow) pair:
//
//   qpos zero        Σ|τ| = 38.587 Nm    elbow bend 91.3°  (a 90° bent elbow!)
//   REST_POSE        Σ|τ| =  1.638 Nm    elbow bend 15.1°  (the linkage's floor)
//
// Two things changed in that re-derivation and both matter:
//   · it is now SYMMETRIC. The old numbers came from a per-arm solve and were
//     0.002-0.008 rad apart per joint, which is invisible but means the two
//     arms were never quite commanded the same pose.
//   · the old left elbow, 1.5795, is OUTSIDE the MJCF's own limit for that
//     joint ([-0.943, 1.542] — the elbows are not mirrored in the model, see
//     motions.js RANGES). MuJoCo clamped it silently; the real arm would have
//     pushed into its stop. ±1.52 keeps 0.022 rad of margin on both sides.
//
// Values are ABSOLUTE joint units (what the wire commands use) and are
// byte-identical to `REST` in motions.js (minus the steer joints) and to
// `RobotController.restBodyPose` + `homeGrip` in both apps
// (MABEL Mobile App/Services/RobotController.swift) — ONE home pose across
// every client. The arms/torso/neck/steer joints have ref 0, so those numbers
// are also the neutral displacement; the two WRISTS carry ref ±0.15 (their
// neutral qpos), so a LEVEL hand is commanded by the ref value, not 0 —
// sending 0 would twist the wrist by −ref.
export const REST_POSE = Object.freeze({
  left_arm_1: 0, left_arm_2: -0.1711, left_arm_3: 0, left_arm_4: 1.52,
  left_arm_5: 0, left_arm_6: 0, left_arm_7: 0,
  right_arm_1: 0, right_arm_2: 0.1711, right_arm_3: 0, right_arm_4: -1.52,
  right_arm_5: 0, right_arm_6: 0, right_arm_7: 0,
  left_wrist: 0.15, right_wrist: -0.15,          // MJCF ref → level (neutral) hand
  torso: 0, neck_yaw: 0, neck_pitch: 0, neck_roll: 0,
  fl_steer: 0, fr_steer: 0, b_steer: 0,
  left_grip: 0.2, right_grip: 0.2,               // gentle relaxed curl
});

// Peak per-joint speed during the glide, and the default floor duration. The
// glide runs HOME_MIN_S unless a joint would exceed the cap, in which case it
// stretches — a big swing simply takes longer, it never moves faster.
export const HOME_VMAX = 1.0;                    // rad/s
export const HOME_MIN_S = 3.0;                   // s

// smoothstep: zero velocity at both ends (no jerk in, no overshoot out)
export const smoothstep = (u) => {
  const t = u < 0 ? 0 : u > 1 ? 1 : u;
  return t * t * (3 - 2 * t);
};

// Glide duration for a given start pose: floored at HOME_MIN_S and stretched so
// the PEAK velocity (smoothstep peaks at 1.5× the average) stays under the cap.
export function homeDuration(start, target = REST_POSE) {
  let maxD = 0;
  for (const k in target)
    maxD = Math.max(maxD, Math.abs(target[k] - (start?.[k] ?? 0)));
  return Math.max(HOME_MIN_S, 1.5 * maxD / HOME_VMAX);
}

// The commanded pose at normalized progress u ∈ [0,1] of the glide.
export function homePoseAt(start, u, target = REST_POSE) {
  const e = smoothstep(u);
  const out = {};
  for (const k in target) {
    const s = start?.[k] ?? 0;
    out[k] = s + (target[k] - s) * e;
  }
  return out;
}
