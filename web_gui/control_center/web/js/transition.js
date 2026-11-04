// POV transition math — pure and node-testable. The 1st↔3rd person move is a
// single 2 s choreography driven by wall-clock progress u ∈ [0,1]:
//   to3rd : hold at the head-camera pose while the head-cam footage fades into
//           the 3D scene it is looking at (u 0→0.35), then pull UP and BACK
//           along an arced path to the orbit pose (u 0.3→1, eased).
//   to1st : fly from orbit down into the head-camera pose (u 0→0.7, eased),
//           then fade the live footage in over the matching 3D view (0.65→1).
// The flight and the fade overlap slightly so nothing ever looks static.
import * as THREE from 'three';

export const DURATION_MS = 2000;

export const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
// C1-continuous smoothstep: zero velocity at both ends → no visible jerk
export const easeInOut = (t) => { t = clamp01(t); return t * t * (3 - 2 * t); };

// per-direction schedule: f = flight progress (0 = start pose), a = fpv alpha
export function povSchedule(dir, u){
  u = clamp01(u);
  if (dir === 'to3rd'){
    return { f: easeInOut((u - 0.3) / 0.7), a: u < 0.35 ? 1 - u / 0.35 : 0 };
  }
  return { f: easeInOut(u / 0.7), a: u > 0.65 ? (u - 0.65) / 0.35 : 0 };
}

// Interpolated camera pose between A (flight start) and B (flight end), with a
// vertical arc so the pull-back reads as "up and over" rather than a straight
// slide. f is the EASED flight progress from povSchedule.
export function povPose(A, B, f, lift = 0.45){
  const pos = A.pos.clone().lerp(B.pos, f);
  pos.y += Math.sin(Math.PI * f) * lift;
  const quat = A.quat.clone().slerp(B.quat, f);
  const fov = A.fov + (B.fov - A.fov) * f;
  return { pos, quat, fov };
}

// Largest per-frame camera step for a given frame budget — used by the tests
// to prove the path has no jumps at 60 fps.
export function maxStep(A, B, frames = 120, lift = 0.45){
  let prev = povPose(A, B, 0, lift).pos, worst = 0;
  for (let i = 1; i <= frames; i++){
    const p = povPose(A, B, easeInOut(i / frames), lift).pos;
    worst = Math.max(worst, p.distanceTo(prev));
    prev = p;
  }
  return worst;
}
