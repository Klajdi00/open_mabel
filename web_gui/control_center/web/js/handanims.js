// Hand animation library — percent-space clips for ONE ORCA hand, streamed by dexterity.js at 20 Hz
// to whichever hand module is selected, so every clip plays on the left and the right alike.
// Percent per joint: 0 = open / extended, 100 = closed / flexed; abd 50 = neutral (lower = toward
// the thumb side); wrist 50 = neutral. H(index, middle, ring, pinky, thumb, wrist): fingers
// [pip, mcp, abd], thumb [pip, dip, abd, mcp]. Every clip is wrapped: eased in from the open hand
// and out again, a little organic drift on every joint, clamped 0–100. Pure data + math, no DOM.
import { graspPose } from './handmodel.js';

const H = (i, m, r, p, t, wrist = 50) => ({
  index_pip: i[0], index_mcp: i[1], index_abd: i[2], middle_pip: m[0], middle_mcp: m[1], middle_abd: m[2],
  ring_pip: r[0], ring_mcp: r[1], ring_abd: r[2], pinky_pip: p[0], pinky_mcp: p[1], pinky_abd: p[2],
  thumb_pip: t[0], thumb_dip: t[1], thumb_abd: t[2], thumb_mcp: t[3], wrist,
});
const c01 = (u) => Math.max(0, Math.min(1, u));
const E = (u) => 0.5 - 0.5 * Math.cos(Math.PI * c01(u));                       // ease in-out
const OV = (u) => { u = c01(u); const s = 1.6; return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2; };  // overshoot + settle
const osc = (t, hz, lo = 0, hi = 1, ph = 0) => lo + (hi - lo) * (0.5 - 0.5 * Math.cos(2 * Math.PI * hz * t + ph));
const pulse = (t, hz, sharp = 4) => Math.max(0, Math.sin(2 * Math.PI * hz * t)) ** sharp;   // short beats
export const mix = (a, b, u) => { const w = c01(u), o = {}; for (const k in a) o[k] = a[k] + ((b[k] ?? a[k]) - a[k]) * w; return o; };
const W = (p, wrist) => ({ ...p, wrist });
const per = (base, fn) => { const p = { ...base }; ['index', 'middle', 'ring', 'pinky'].forEach((f, i) => fn(p, f, i)); return p; };
const curl = (p, f, c) => { p[`${f}_mcp`] = c; p[`${f}_pip`] = c; };
/** keyframes [[t, pose, ease?], …] → eased blend; the third item picks the ease into that frame. */
const kf = (frames) => (t) => {
  if (t <= frames[0][0]) return frames[0][1];
  for (let i = 0; i < frames.length - 1; i++){
    const [t0, a] = frames[i], [t1, b, ez = E] = frames[i + 1];
    if (t < t1) return mix(a, b, ez((t - t0) / (t1 - t0)));
  }
  return frames[frames.length - 1][1];
};

// ── shapes (all carry wrist 50) — audited on the ORCA v2 MuJoCo model: no mesh penetration over
// 0.5 mm in this hand's calibrated windows nor in the CAD frame (11 shapes were nudged ≤ 65 % total,
// mostly the thumb's middle joint; pose audit 2026-09-21). On MABEL's ORCA v1 hands the same
// percentages land on the v1 ranges via handmodel.js; they were not mesh-audited on v1. ─────────
const FLAT  = () => ({ ...graspPose(0), wrist: 50 });
const FIST  = () => ({ ...graspPose(1), wrist: 50 });
const OPEN  = H([2, 2, 50], [2, 2, 50], [2, 2, 50], [2, 2, 50], [5, 5, 50, 5]);
const SPLAY = H([4, 4, 20], [4, 4, 50], [4, 4, 70], [4, 4, 90], [5, 5, 80, 5]);
const TIGHT = H([98, 90, 50], [98, 90, 50], [98, 90, 50], [98, 90, 50], [45, 85, 0, 75]);
const CLAPH = H([18, 15, 80], [18, 15, 50], [18, 15, 20], [18, 15, 10], [25, 25, 50, 35]);    // fingers together
export const SHAPES = {
  middle_finger: H([85, 80, 50], [5, 5, 50], [85, 80, 50], [85, 80, 50], [55, 70, 10, 60]),
  point:         H([5, 5, 50], [85, 80, 50], [85, 80, 50], [85, 80, 50], [50, 60, 30, 60]),
  peace:         H([5, 5, 35], [5, 5, 65], [85, 80, 50], [85, 80, 50], [60, 60, 30, 60]),
  gun:           H([5, 5, 45], [5, 5, 55], [85, 80, 50], [85, 80, 50], [5, 5, 60, 5]),
  loser:         H([5, 5, 50], [85, 80, 50], [85, 80, 50], [85, 80, 50], [5, 5, 70, 5]),
  horns:         H([5, 5, 40], [85, 80, 50], [85, 80, 50], [5, 5, 60], [50, 60, 30, 60]),
  shaka:         H([85, 80, 50], [85, 80, 50], [85, 80, 50], [5, 5, 60], [5, 5, 60, 5]),
  thumbs_up:     H([80, 80, 50], [80, 80, 50], [80, 80, 50], [80, 80, 50], [10, 10, 30, 5]),
  ok:            H([65, 65, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50], [60, 60, 70, 45]),
  pinch:         H([60, 60, 50], [10, 10, 50], [10, 10, 50], [10, 10, 50], [55, 55, 65, 50]),
  claw:          H([55, 40, 25], [55, 40, 45], [55, 40, 60], [55, 40, 80], [45, 45, 60, 40]),
  vulcan:        H([4, 4, 70], [4, 4, 40], [4, 4, 60], [4, 4, 25], [5, 5, 70, 5]),      // 🖖 two pairs, V in the middle
  cup:           H([45, 45, 50], [45, 45, 50], [45, 45, 50], [45, 45, 50], [35, 40, 55, 30]),
  shush:         H([5, 5, 50], [85, 80, 50], [85, 80, 50], [85, 80, 50], [70, 75, 55, 60]),
  three:         H([5, 5, 40], [5, 5, 50], [5, 5, 60], [85, 80, 50], [70, 70, 25, 70]),
  four:          H([5, 5, 35], [5, 5, 45], [5, 5, 60], [5, 5, 70], [70, 70, 25, 70]),
  splay: SPLAY, open: OPEN, tight: TIGHT,
  // GRASP-taxonomy shapes (Feix et al. 2016): power (cylindrical), tripod, lateral / key pinch, hook
  power:     H([72, 78, 50], [72, 78, 50], [72, 78, 50], [72, 78, 50], [90, 55, 35, 65]),
  tripod:    H([58, 55, 45], [58, 55, 55], [80, 78, 50], [80, 78, 50], [50, 50, 70, 55]),
  key_pinch: H([70, 60, 50], [72, 65, 50], [74, 70, 50], [76, 72, 50], [30, 25, 10, 60]),
  hook:      H([88, 30, 50], [88, 30, 50], [88, 30, 50], [88, 30, 50], [8, 8, 50, 8]),
};
const S = SHAPES;
const REST = FLAT();

// ── the clips: name → { title, dur, at(t), snap? } ────────────────────────────
// `snap` marks a clip with one deliberate fast beat (a flick, a punch); everything else is eased.
const A = {
  // hello
  wave:      { title: 'wave', dur: 3.6, at: (t) => W(per(FLAT(), (p, f, i) => curl(p, f, 8 + 70 * Math.max(0, Math.sin(t * 4.2 - i * 0.9)))), osc(t, 2.1, 22, 78)) },
  hi_wiggle: { title: 'hi (wiggle)', dur: 3.0, at: (t) => W(per(SPLAY, (p, f, i) => { p[`${f}_abd`] += (i % 2 ? 1 : -1) * 26 * Math.sin(t * 9); }), osc(t, 1.5, 38, 62)) },
  salute:    { title: 'salute', dur: 3.6, snap: true, at: kf([[0, REST], [0.55, W(CLAPH, 28), OV], [2.5, W(CLAPH, 30)], [2.75, W(CLAPH, 66), OV], [3.6, REST]]) },
  high_five: { title: 'high five', dur: 2.8, snap: true, at: kf([[0, REST], [0.5, W(H([14, 12, 40], [14, 12, 50], [14, 12, 60], [14, 12, 70], [20, 20, 55, 20]), 30)], [0.95, W(SPLAY, 74), OV], [1.5, W(H([8, 6, 20], [8, 6, 50], [8, 6, 80], [8, 6, 88], [6, 6, 78, 6]), 66)], [2.1, W(S.cup, 56)], [2.8, REST]]) },
  fist_bump: { title: 'fist bump', dur: 2.8, snap: true, at: kf([[0, REST], [0.6, W(FIST(), 44)], [1.05, W(TIGHT, 68), OV], [1.6, W(FIST(), 60)], [2.2, W(SPLAY, 48)], [2.8, REST]]) },
  peace_out: { title: 'peace out', dur: 3.2, at: (t) => W(mix(S.peace, H([5, 5, 25], [5, 5, 75], [85, 80, 50], [85, 80, 50], [60, 60, 30, 60]), osc(t, 1.6)), osc(t, 1.6, 34, 66)) },
  // happy
  jazz_hands: { title: 'jazz hands', dur: 3.6, at: (t) => W(per(SPLAY, (p, f, i) => { p[`${f}_abd`] += 18 * Math.sin(t * 12 + i * 1.3); p[`${f}_mcp`] = 6 + 9 * Math.max(0, Math.sin(t * 12 + i)); }), 50 + 16 * Math.sin(t * 6)) },
  thumbs_up_bounce: { title: 'thumbs up!', dur: 3.2, at: (t) => { const p = { ...S.thumbs_up }; const b = osc(t, 2.2); p.thumb_mcp = 5 + 22 * b; p.thumb_abd = 30 + 14 * b; return W(p, 42 + 18 * b); } },
  clap_solo:  { title: 'clap (one hand)', dur: 3.2, at: (t) => W(mix(CLAPH, H([40, 55, 80], [40, 55, 50], [40, 55, 20], [40, 55, 10], [35, 35, 50, 45]), osc(t, 3.0)), osc(t, 3.0, 36, 64)) },
  ok_ok:      { title: 'ok ok', dur: 3.0, at: (t) => W(mix(S.ok, H([75, 70, 50], [5, 5, 40], [5, 5, 60], [5, 5, 70], [60, 65, 72, 50]), osc(t, 1.8)), osc(t, 1.8, 40, 62)) },
  heart_half: { title: 'half heart', dur: 3.2, at: kf([[0, REST], [0.9, W(H([55, 60, 45], [55, 60, 50], [60, 65, 55], [70, 70, 65], [45, 50, 60, 55]), 34)], [2.4, W(H([55, 60, 45], [55, 60, 50], [60, 65, 55], [70, 70, 65], [45, 50, 60, 55]), 30)], [3.2, REST]]) },
  chefs_kiss: { title: "chef's kiss", dur: 3.0, snap: true, at: kf([[0, REST], [0.6, W(H([62, 60, 45], [62, 60, 50], [62, 60, 55], [62, 60, 60], [58, 60, 65, 58]), 30)], [1.5, W(H([62, 60, 45], [62, 60, 50], [62, 60, 55], [62, 60, 60], [58, 60, 65, 58]), 26)], [1.85, W(SPLAY, 72), OV], [3.0, REST]]) },
  rock_on:    { title: 'rock on 🤘', dur: 3.4, at: (t) => { const p = { ...S.horns }; p.index_abd = 40 - 12 * Math.sin(t * 8); p.pinky_abd = 60 + 12 * Math.sin(t * 8); return W(p, osc(t, 2.0, 28, 72)); } },
  // sassy
  talk_to_the_hand: { title: 'talk to the hand', dur: 3.2, snap: true, at: kf([[0, REST], [0.4, W(OPEN, 40)], [0.7, W(H([0, 0, 20], [0, 0, 50], [0, 0, 80], [0, 0, 88], [0, 0, 80, 0]), 18), OV], [2.6, W(H([0, 0, 20], [0, 0, 50], [0, 0, 80], [0, 0, 88], [0, 0, 80, 0]), 22)], [3.2, REST]]) },
  finger_wag: { title: 'no no no', dur: 3.2, at: (t) => { const p = { ...S.point }; p.index_abd = 50 + 30 * Math.sin(t * 7.5); return W(p, 50 + 18 * Math.sin(t * 7.5 + 0.6)); } },
  whatever:   { title: 'whatever (L)', dur: 2.8, at: kf([[0, REST], [0.6, W(S.loser, 40)], [1.4, W(S.loser, 36)], [2.0, W(S.loser, 44)], [2.8, REST]]) },
  hair_flip:  { title: 'hair flip', dur: 2.6, at: (t) => W(per(FLAT(), (p, f, i) => { const u = c01((t - 0.4 - i * 0.15) / 0.6); curl(p, f, 10 + 40 * Math.sin(Math.PI * u)); p[`${f}_abd`] = 50 + 20 * Math.sin(Math.PI * u) * (i - 1.5); }), 50 + 26 * Math.sin(Math.PI * c01((t - 0.3) / 1.5))) },
  snap:       { title: 'snap', dur: 2.0, snap: true, at: kf([[0, REST], [0.5, W(H([25, 20, 50], [72, 68, 50], [40, 35, 50], [40, 35, 50], [55, 58, 70, 55]), 58)], [0.95, W(H([25, 20, 50], [72, 68, 50], [40, 35, 50], [40, 35, 50], [55, 58, 70, 55]), 62)], [1.07, W(H([25, 20, 50], [88, 85, 50], [40, 35, 50], [40, 35, 50], [30, 30, 55, 25]), 34), OV], [2.0, REST]]) },
  mic_drop:   { title: 'mic drop', dur: 2.8, snap: true, at: kf([[0, REST], [0.5, W(S.pinch, 40)], [1.6, W(S.pinch, 38)], [1.78, W(SPLAY, 82), OV], [2.8, REST]]) },
  slow_clap:  { title: 'slow clap', dur: 4.2, at: (t) => W(mix(CLAPH, H([35, 50, 80], [35, 50, 50], [35, 50, 20], [35, 50, 10], [35, 35, 50, 45]), osc(t, 1.0)), osc(t, 1.0, 30, 70)) },
  // mean
  middle_finger: { title: 'middle finger', dur: 3.4, at: kf([[0, REST], [0.7, W(S.middle_finger, 30), OV], [1.6, W(S.middle_finger, 34)], [2.0, W(S.middle_finger, 28)], [2.7, W(S.middle_finger, 32)], [3.4, REST]]) },
  slow_bird:  { title: 'the slow bird', dur: 4.8, at: (t) => { const p = FLAT(); ['pinky', 'ring', 'index'].forEach((f, i) => curl(p, f, 10 + 75 * E((t - 0.4 - i * 0.7) / 0.6))); const th = E((t - 2.6) / 0.6); p.thumb_pip = 10 + 60 * th; p.thumb_dip = 10 + 60 * th; p.thumb_mcp = 10 + 60 * th; p.thumb_abd = 50 - 30 * th; return W(p, 50 - 22 * E((t - 0.3) / 3.2)); } },
  double_bird_tap: { title: 'bird tap-tap', dur: 3.4, at: (t) => { const p = { ...S.middle_finger }; const tap = Math.max(0, Math.sin(t * 10)); p.middle_mcp = 5 + 28 * tap; p.middle_pip = 5 + 14 * tap; return W(p, 32 + 10 * tap); } },
  flick_off:  { title: 'flick off', dur: 2.4, snap: true, at: kf([[0, REST], [0.5, W(H([85, 80, 50], [70, 65, 50], [85, 80, 50], [85, 80, 50], [55, 60, 45, 60]), 58)], [1.0, W(H([85, 80, 50], [70, 65, 50], [85, 80, 50], [85, 80, 50], [55, 60, 45, 60]), 62)], [1.14, W(S.middle_finger, 26), OV], [1.9, W(S.middle_finger, 30)], [2.4, REST]]) },
  shush:      { title: 'shush', dur: 3.0, at: kf([[0, REST], [0.6, W(S.shush, 38)], [1.5, W(S.shush, 42)], [2.3, W(S.shush, 36)], [3.0, REST]]) },
  dismiss:    { title: 'shoo', dur: 3.0, at: (t) => W(per(FLAT(), (p, f, i) => { curl(p, f, 30 + 22 * Math.sin(t * 6 - i * 0.5)); p[`${f}_abd`] = 50 + (i - 1.5) * 12 * Math.sin(t * 6); }), 50 + 24 * Math.sin(t * 6 - 0.8)) },
  // aggressive
  fist_pump:  { title: 'fist pump', dur: 3.2, at: (t) => W(mix(FIST(), TIGHT, osc(t, 2.5)), osc(t, 2.5, 30, 72)) },
  knuckle_crack: { title: 'knuckle crack', dur: 3.8, snap: true, at: kf([[0, REST], [0.5, W(H([70, 40, 50], [70, 40, 50], [70, 40, 50], [70, 40, 50], [50, 50, 40, 50]), 60)], [1.05, W(TIGHT, 26), OV], [1.4, W(H([70, 40, 50], [70, 40, 50], [70, 40, 50], [70, 40, 50], [50, 50, 40, 50]), 58)], [1.95, W(TIGHT, 76), OV], [2.3, W(H([70, 40, 50], [70, 40, 50], [70, 40, 50], [70, 40, 50], [50, 50, 40, 50]), 50)], [2.85, W(TIGHT, 24), OV], [3.8, REST]]) },
  claw_scratch: { title: 'claw', dur: 3.2, at: (t) => W(mix(S.claw, H([80, 55, 20], [80, 55, 45], [80, 55, 62], [80, 55, 85], [60, 60, 55, 55]), osc(t, 3.5)), 50 + 16 * Math.sin(t * 7 + 0.5)) },
  come_at_me: { title: 'come at me', dur: 3.4, at: (t) => W(per(H([20, 20, 15], [20, 20, 50], [20, 20, 85], [20, 20, 90], [20, 20, 70, 20]), (p, f, i) => curl(p, f, 20 + 55 * Math.max(0, Math.sin(t * 5.5 - i * 0.35)))), osc(t, 0.875, 62, 34)) },
  punch:      { title: 'punch', dur: 2.0, snap: true, at: kf([[0, REST], [0.35, W(FIST(), 42)], [0.55, W(TIGHT, 70), OV], [1.4, W(TIGHT, 66)], [2.0, REST]]) },
  crush:      { title: 'crush', dur: 3.2, at: kf([[0, REST], [0.8, W(FIST(), 46)], [1.7, W(TIGHT, 36)], [2.3, W(TIGHT, 34)], [3.2, REST]]) },
  finger_jab: { title: 'finger jab', dur: 2.6, at: (t) => { const p = { ...S.point }; const jab = pulse(t, 1.25, 2); p.index_mcp = 5 + 34 * jab; p.index_pip = 5 + 10 * jab; return W(p, 44 + 28 * jab); } },
  // fun
  count:      { title: 'count to 5', dur: 5.2, at: kf([[0, REST], [0.5, W(FIST(), 44)], [0.8, W(S.point, 48)], [1.6, W(S.peace, 50)], [2.4, W(S.three, 46)], [3.2, W(S.four, 50)], [4.0, W(SPLAY, 56)], [4.6, W(SPLAY, 52)], [5.2, REST]]) },
  piano:      { title: 'piano', dur: 4.2, at: (t) => W(per(FLAT(), (p, f, i) => { const tap = Math.max(0, Math.sin(t * 9 - i * 1.6)); p[`${f}_mcp`] = (12 + tap * 62) * 0.55; p[`${f}_pip`] = 12 + tap * 62; }), 50 + 8 * Math.sin(t * 2.2)) },
  breathe:    { title: 'breathe', dur: 6.0, at: (t) => W(graspPose(0.5 - 0.5 * Math.cos(t * (2 * Math.PI / 3))), 50 + 10 * Math.sin(t * (2 * Math.PI / 3) - 0.7)) },
  rps:        { title: 'rock·paper·scissors', dur: 4.6, snap: true, at: (t) => {   // rock, rock, rock … SHOOT (paper), then scissors
    const scissors = H([5, 5, 35], [5, 5, 65], [85, 80, 50], [85, 80, 50], [70, 70, 25, 70]);
    if (t < 2.4) return W(FIST(), osc(t, 1.25, 30, 66));                          // three pumps on the wrist
    if (t < 3.3) return W(mix(FIST(), OPEN, OV((t - 2.4) / 0.25)), 58);          // shoot!
    return W(mix(OPEN, scissors, E((t - 3.3) / 0.35)), 52); } },
  spider_walk: { title: 'spider walk', dur: 4.2, at: (t) => W(per(S.claw, (p, f, i) => { const s = Math.max(0, Math.sin(t * 5 - i * 1.4)); curl(p, f, 25 + 45 * s); p[`${f}_abd`] += 10 * Math.sin(t * 5 - i * 1.4); }), 50 + 14 * Math.sin(t * 2.5)) },
  typing:     { title: 'typing', dur: 4.2, at: (t) => W(per(H([20, 25, 30], [20, 25, 45], [20, 25, 60], [20, 25, 75], [30, 30, 50, 30]), (p, f, i) => { const k = Math.max(0, Math.sin(t * 13 + i * 2.1 + Math.sin(t * 0.7 + i))); p[`${f}_mcp`] = 25 + 30 * k; p[`${f}_pip`] = 20 + 20 * k; }), 44 + 6 * Math.sin(t * 3.1)) },
  rain:       { title: 'rain', dur: 4.2, at: (t) => W(per(FLAT(), (p, f, i) => curl(p, f, 10 + 40 * Math.max(0, Math.sin(t * 6.5 + i * 2.4)))), 50 + 12 * Math.sin(t * 1.3)) },
  beckon:     { title: 'come here', dur: 3.6, at: (t) => { const p = { ...S.point }; const u = osc(t, 2.4); p.index_mcp = 10 + 60 * u; p.index_pip = 15 + 70 * u; return W(p, 42 + 26 * u); } },
  gun_shot:   { title: 'pew pew', dur: 3.0, snap: true, at: (t) => { const p = { ...S.gun }; const k = pulse(t, 1.0, 6); p.thumb_mcp = 5 + 55 * k; p.thumb_abd = 60 - 20 * k; p.index_mcp = 5 + 10 * k; return W(p, 50 - 30 * k); } },
  money:      { title: 'money money', dur: 3.2, at: (t) => { const p = { ...S.pinch }; const u = osc(t, 4); p.index_pip = 55 + 20 * u; p.middle_pip = 45 + 20 * u; p.middle_mcp = 35 + 15 * u; p.thumb_pip = 45 + 15 * (1 - u); return W(p, 50 + 6 * Math.sin(t * 8)); } },
  live_long:  { title: 'live long 🖖', dur: 3.6, at: kf([[0, REST], [0.5, W(OPEN, 46)], [1.1, W(S.vulcan, 30)], [2.9, W(S.vulcan, 34)], [3.6, REST]]) },
  bunny:      { title: 'bunny ears', dur: 3.2, at: (t) => { const p = { ...S.peace }; const u = Math.max(0, Math.sin(t * 7)); curl(p, 'index', 5 + 45 * u); curl(p, 'middle', 5 + 45 * (1 - u)); return W(p, 44 + 10 * Math.sin(t * 3.5)); } },
  phone:      { title: 'call me', dur: 3.2, at: (t) => { const p = { ...S.shaka }; p.pinky_abd = 60 + 10 * Math.sin(t * 5); p.thumb_abd = 60 + 10 * Math.sin(t * 5 + 1); return W(p, 36 + 10 * Math.sin(t * 5)); } },
  countdown:  { title: 'countdown 5→0', dur: 5.4, at: kf([[0, REST], [0.5, W(SPLAY, 52)], [1.3, W(S.four, 48)], [2.1, W(S.three, 52)], [2.9, W(S.peace, 46)], [3.7, W(S.point, 52)], [4.5, W(FIST(), 40)], [5.4, REST]]) },
  drum_roll:  { title: 'drum roll', dur: 3.2, snap: true, at: (t) => W(per(H([15, 25, 40], [15, 25, 50], [15, 25, 60], [15, 25, 70], [30, 30, 50, 30]), (p, f, i) => { p[`${f}_pip`] = 15 + 45 * Math.max(0, Math.sin(t * 22 + i * Math.PI)); }), 46 + 5 * Math.sin(t * 22)) },
};

// ── the wrapper: ease in from / out to the open hand, organic drift, clamp ────
const IN = 0.35, OUT = 0.45;
const hash = (k) => [...k].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) % 997, 7);
const life = (t, k) => (k === 'wrist' ? 1.6 : 2.4) * (Math.sin(t * 1.7 + hash(k)) + 0.5 * Math.sin(t * 2.9 + 2 * hash(k)));
export const ANIMATIONS = Object.fromEntries(Object.entries(A).map(([name, a]) => [name, { ...a, at: (t) => {
  let p = a.at(t);
  if (t < IN) p = mix(REST, p, E(t / IN));
  if (t > a.dur - OUT) p = mix(p, REST, E((t - (a.dur - OUT)) / OUT));
  const o = {};
  for (const k in p) o[k] = Math.max(0, Math.min(100, p[k] + (a.still ? 0 : life(t, k))));
  return o;
} }]));
