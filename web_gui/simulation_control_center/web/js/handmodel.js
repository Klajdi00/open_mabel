// ORCA hand model: the 17 joints, percent-of-range ⇄ radians conversion, the
// firmware's predefined poses (percent units, hardware_bridge/hands/orcahand/
// poses.py), finger groupings, and the grasp-slider blend. Pure data + math —
// node-testable; the wire format is sensor_msgs-style {name[], position[]}
// via the bridge's /<side>_hand/command relay (radians, URDF names).

// URDF joint suffixes with calibrated rad ranges (left_hand.yaml; thumb wire
// names are URDF-native: cmc→mcp, mp→dip, ip→pip per hands_subsystem.py).
//
// THESE RANGES ARE **RELATIVE TO `ref`** — they are joint TRAVEL about the
// assembled pose, exactly as `mabel_joints.json` reports lower/upper. The wire,
// however, carries ABSOLUTE MuJoCo qpos, so a command is `ref + travel`.
//
// Ignoring `ref` is the "the ORCA wrist/hand is messed up" bug, and it had three
// separate heads:
//   1. NO REF. Every ref != 0 joint was commanded `ref` radians off intent —
//      wrist 0.15 rad (8.6°), index_abd 0.40 (22.9°), pinky_abd 0.52 (30.0°),
//      ring_abd 0.17 (9.7°), and the whole thumb chain.
//   2. THUMB_ABD OUT OF RANGE. Its absolute ctrlrange is [−1.08211, 0.0]; the
//      ref-less conversion emitted +0.192 for a flat hand, which the actuator
//      clamped to 0.0 — the thumb slammed to its hard stop on every hand command.
//   3. SIDE-BLIND WRIST. The two wrists MIRROR (left ref +0.15 / right −0.15,
//      ranges swapped), but one shared table sent both the same absolute angle,
//      so the hands were ~0.22 rad (12.8°) from being mirror images.
//   4. thumb_pip / thumb_dip travel ranges were transposed.
// Ranges and refs are now checked against the generated manifest by
// tests/js/handmodel.test.mjs, so they cannot silently drift again.
export const HAND_JOINTS = [
  ['wrist',      -1.04720, 0.52360],
  ['thumb_mcp',  -0.87266, 0.87266],  // ORCA thumb_cmc
  ['thumb_abd',  -0.34907, 0.73304],
  ['thumb_pip',  -0.20944, 1.81496],  // ORCA thumb_ip
  ['thumb_dip',  -0.34907, 1.95477],  // ORCA thumb_mp
  ['index_abd',  -0.64577, 0.64577],
  ['index_mcp',  -0.34907, 1.65806],
  ['index_pip',  -0.34907, 1.88496],
  ['middle_abd', -0.64577, 0.64577],
  ['middle_mcp', -0.34907, 1.58825],
  ['middle_pip', -0.34907, 1.86750],
  ['ring_abd',   -0.64577, 0.63577],
  ['ring_mcp',   -0.34907, 1.58825],
  ['ring_pip',   -0.34907, 1.86750],
  ['pinky_abd',  -0.64577, 0.64577],
  ['pinky_mcp',  -0.34907, 1.71042],
  ['pinky_pip',  -0.34907, 1.88496],
];
export const RANGE = Object.fromEntries(HAND_JOINTS.map(([n, lo, hi]) => [n, [lo, hi]]));

// MJCF `ref` per joint suffix — the absolute qpos of the ASSEMBLED (neutral)
// pose. Identical on both hands for every finger joint; only the WRIST differs,
// because the two wrists mirror (left +0.15 / right −0.15). Anything not listed
// has ref 0, so absolute == relative and the old maths was accidentally right —
// which is precisely why this bug only ever showed up in the hands.
export const REF = Object.freeze({
  wrist: { left: 0.15, right: -0.15 },
  thumb_abd: -0.73304, thumb_pip: -0.58496, thumb_dip: -0.50477,
  index_abd: -0.40000, ring_abd: 0.17000, pinky_abd: 0.52333,
});

// The wrist is the ONE mirrored joint: its travel is negated and swapped on the
// right hand ([−1.0472, 0.5236] → [−0.5236, 1.0472]). Every finger joint instead
// mirrors via a negated MJCF axis, so its numeric range is side-independent.
const MIRRORED = new Set(['wrist']);

export function refFor(name, side = 'left'){
  const r = REF[name];
  if (r == null) return 0;
  return typeof r === 'number' ? r : (r[side] ?? 0);
}

// ABSOLUTE qpos range [lo, hi] for one joint on one hand — what the wire wants.
export function rangeFor(name, side = 'left'){
  const r = RANGE[name];
  if (!r) return null;
  const ref = refFor(name, side);
  const [lo, hi] = (side === 'right' && MIRRORED.has(name)) ? [-r[1], -r[0]] : r;
  return [lo + ref, hi + ref];
}

export const FINGERS = {
  thumb: ['thumb_mcp', 'thumb_abd', 'thumb_pip', 'thumb_dip'],
  index: ['index_abd', 'index_mcp', 'index_pip'],
  middle: ['middle_abd', 'middle_mcp', 'middle_pip'],
  ring: ['ring_abd', 'ring_mcp', 'ring_pip'],
  pinky: ['pinky_abd', 'pinky_mcp', 'pinky_pip'],
  wrist: ['wrist'],
};

// 0 % = open (range min), 100 % = closed (range max) — orcahand/joints.py.
// Both directions work in ABSOLUTE qpos (the wire's units), so they need to know
// WHICH HAND: the wrist's range and ref are mirrored between the two.
export const pctToRad = (name, pct, side = 'left') => {
  const [lo, hi] = rangeFor(name, side);
  return lo + (Math.max(0, Math.min(100, pct)) / 100) * (hi - lo);
};
export const radToPct = (name, rad, side = 'left') => {
  const [lo, hi] = rangeFor(name, side);
  return Math.max(0, Math.min(100, ((rad - lo) / (hi - lo)) * 100));
};

// firmware presets (percent, ORCA joint keys mapped to URDF suffixes)
const P = (idx, mid, ring, pinky, thumb, wrist = 50) => ({
  index_pip: idx[0], index_mcp: idx[1], index_abd: idx[2],
  middle_pip: mid[0], middle_mcp: mid[1], middle_abd: mid[2],
  ring_pip: ring[0], ring_mcp: ring[1], ring_abd: ring[2],
  pinky_pip: pinky[0], pinky_mcp: pinky[1], pinky_abd: pinky[2],
  thumb_pip: thumb[0], thumb_dip: thumb[1], thumb_abd: thumb[2], thumb_mcp: thumb[3],
  wrist,
});
export const POSES = {
  flat:      P([10, 10, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50, 10]),
  fist:      P([85, 80, 50], [85, 80, 50], [85, 80, 50], [85, 80, 50], [70, 70, 20, 70]),
  pinch:     P([60, 60, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50], [50, 55, 65, 50]),
  point:     P([10, 10, 50], [85, 80, 50], [85, 80, 50], [85, 80, 50], [60, 60, 30, 60]),
  peace:     P([10, 10, 50], [10, 10, 50], [85, 80, 50], [85, 80, 50], [60, 60, 30, 60]),
  ok:        P([65, 65, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50], [55, 60, 70, 45]),
  thumbs_up: P([80, 80, 50], [80, 80, 50], [80, 80, 50], [80, 80, 50], [10, 10, 30, 5]),
  rock:      P([10, 10, 50], [80, 80, 50], [80, 80, 50], [10, 10, 50], [50, 50, 40, 50]),
  spider:    P([50, 30, 60], [50, 30, 50], [50, 30, 40], [50, 30, 60], [30, 30, 60, 20]),
};

// grasp slider: 0 → flat, 1 → fist, linear per-joint blend (percent space)
export function graspPose(t){
  t = Math.max(0, Math.min(1, t));
  const a = POSES.flat, b = POSES.fist, out = {};
  for (const k in a) out[k] = a[k] + (b[k] - a[k]) * t;
  return out;
}

// percent pose {suffix: pct} → wire {name[], position[]} in radians for a side
export function poseToCommand(side, pctPose){
  const name = [], position = [];
  for (const [suffix, pct] of Object.entries(pctPose)){
    if (!RANGE[suffix]) continue;
    name.push(`${side}_${suffix}`);
    position.push(pctToRad(suffix, pct, side));
  }
  return { name, position };
}
