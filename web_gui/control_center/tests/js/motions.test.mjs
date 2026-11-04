// Motion library + choreography track model: schema completeness, joint-range
// and velocity budgets at 20 Hz, neutral start/end, nav-ends-at-zero, NLE
// block ops (cut/trim/snap/duplicate), location gap detection + auto-fix,
// backend action merging.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MOTION_CLIPS, RANGES, NEUTRAL, restOf, MAX_JOINT_VEL, GRIP_SCALE, BLEND_SCALE,
         ChoreoTrack, SNAP_GRID, clipDur, clipJointsAt, clipNavAt, defaults,
         mergeActions, detectGaps, autoFixGaps, liftRateLimit, EXEC_RATE } from '../../web/js/motions.js';
// Hand values on the wire are ABSOLUTE qpos (ref + travel), so bounds come from
// rangeFor(suffix, side) — the relative RANGE table is not a valid bound for the
// six ref != 0 joints (wrist, thumb_abd/pip/dip, index/ring/pinky abd).
import { RANGE as HAND_RANGE, rangeFor } from '../../web/js/handmodel.js';

const FS = 20, DT = 1 / FS;
const clips = Object.entries(MOTION_CLIPS);
const avail = clips.filter(([, c]) => c.status === 'available');
const executable = avail.filter(([, c]) => c.joints || c.nav);

function rangeOf(id){
  if (RANGES[id]) return RANGES[id];
  const m = id.match(/^(left|right)_(.+)$/);
  if (m && HAND_RANGE[m[2]]) return rangeFor(m[2], m[1]);
  return null;
}
function sample(clip){
  const p = defaults(clip);
  const d = clipDur(clip, p);
  const out = [];
  for (let i = 0; i <= Math.round(d * FS); i++)
    out.push({ t: i * DT,
      joints: clip.joints ? clipJointsAt(clip, i * DT, p) : null,
      nav: clip.nav ? clipNavAt(clip, i * DT, p) : null });
  return { d, frames: out };
}

test('clip schema completeness', () => {
  for (const [id, c] of clips){
    assert.ok(c.title, `${id} title`);
    assert.ok(c.emoji, `${id} emoji`);
    assert.ok(clipDur(c, defaults(c)) > 0, `${id} dur`);
    assert.ok(['manip', 'nav', 'both'].includes(c.lane), `${id} lane`);
    assert.ok(['animation', 'policy', 'nav', 'pose', 'transition'].includes(c.category), `${id} category`);
    assert.ok(['available', 'coming_soon'].includes(c.status), `${id} status`);
    if (c.status === 'available' && c.kind !== 'goal')
      assert.ok(c.joints || c.nav, `${id} must be executable today`);
    if (c.status === 'coming_soon')
      assert.ok(!c.joints && !c.nav, `${id} coming-soon must not execute`);
  }
});

test('joints stay within ranges over the whole clip', () => {
  for (const [id, c] of executable){
    if (!c.joints) continue;
    for (const { t, joints } of sample(c).frames){
      for (const [j, v] of Object.entries(joints)){
        const r = rangeOf(j);
        assert.ok(r, `${id}: unknown joint ${j}`);
        assert.ok(v >= r[0] - 1e-6 && v <= r[1] + 1e-6,
          `${id} t=${t.toFixed(2)} ${j}=${v.toFixed(3)} outside [${r[0]}, ${r[1]}]`);
      }
    }
  }
});

// BLEND_SCALE, not just GRIP_SCALE: the LIFT is a linear axis and its metres
// have to be converted too, or the budget does not apply to it at all. It did
// not, and `happy` was bouncing the column at 0.76 m/s (7.1 rad-equivalents/s,
// 2.9x the budget) in a test that passed.
//
// The LIFT is checked on what the executor actually STREAMS: timeline.js clamps the
// column (the sum of the two stage keys) with liftRateLimit() on every tick, because
// several clips (`happy`, …) were authored against a column far faster than the real
// one. Sampled at EXEC_RATE, the rate the executor runs at, through the same clamp.
function executorStream(clip){
  const p = defaults(clip), d = clipDur(clip, p), dt = 1 / EXEC_RATE, out = [];
  let prev = null;
  for (let i = 0; i <= Math.round(d * EXEC_RATE); i++){
    const t = i * dt, j = { ...clipJointsAt(clip, t, p) };
    if (j.lift_lower != null || j.lift_upper != null){
      const col = (j.lift_lower || 0) + (j.lift_upper || 0);
      const lim = prev == null ? col : liftRateLimit(prev, col, dt);
      if (j.lift_lower != null) j.lift_lower = lim / 2;
      if (j.lift_upper != null) j.lift_upper = lim / 2;
      prev = lim;
    }
    out.push({ t, joints: j });
  }
  return out;
}
test(`per-tick joint velocity < ${MAX_JOINT_VEL} rad/s at 20 Hz (lift: as streamed at EXEC_RATE)`, () => {
  for (const [id, c] of executable){
    if (!c.joints) continue;
    const { frames } = sample(c);
    for (let i = 1; i < frames.length; i++){
      for (const j of Object.keys(frames[i].joints)){
        if (j === 'lift_lower' || j === 'lift_upper') continue;     // checked below, as streamed
        const dv = Math.abs(frames[i].joints[j] - (frames[i - 1].joints[j] ?? 0));
        const scale = BLEND_SCALE[j] || 1;
        assert.ok(dv * FS * scale <= MAX_JOINT_VEL + 1e-9,
          `${id} ${j} @t=${frames[i].t.toFixed(2)}: ${(dv * FS * scale).toFixed(2)} rad/s`);
      }
    }
    const st = executorStream(c);
    for (let i = 1; i < st.length; i++)
      for (const j of ['lift_lower', 'lift_upper']){
        if (st[i].joints[j] == null || st[i - 1].joints[j] == null) continue;
        const v = Math.abs(st[i].joints[j] - st[i - 1].joints[j]) * EXEC_RATE * BLEND_SCALE[j];
        assert.ok(v <= MAX_JOINT_VEL + 1e-9, `${id} ${j} @t=${st[i].t.toFixed(2)}: ${v.toFixed(2)} rad/s streamed`);
      }
  }
});

// Nothing in the library may drive the robot into itself. The one exception is
// `lift_down`, which asks the column for 0 -- a real capability of the real
// lift that the MJCF cannot represent, because its column self-penetrates the
// chassis below 0.155 m per stage (exp_rest_pose.py E4). Clamping the primitive
// here would quietly take half the stroke away from the hardware, so the defect
// is recorded instead. Measured by web_gui/control_center/tools/exp_pose_library.py.
test('no clip commands the lift below the model\'s usable floor (except lift_down)', () => {
  const FLOOR = 0.155;
  for (const [id, c] of executable){
    if (!c.joints || id === 'lift_down') continue;
    for (const { t, joints } of sample(c).frames)
      for (const j of ['lift_lower', 'lift_upper'])
        if (joints[j] !== undefined)
          assert.ok(joints[j] >= FLOOR - 1e-9,
            `${id} ${j}=${joints[j].toFixed(4)} at t=${t.toFixed(2)} is below ${FLOOR}`);
  }
});

// Every clip starts from REST, whatever it is; only a clip that does NOT hold
// has to come back to it. `hold` means the block keeps its final pose (that is
// what the flag is for, and what `pose/hold clips settle to a steady end pose`
// below checks), so `sleep` is allowed to stay asleep.
test('every clip starts at REST, and every non-hold clip ends there too', () => {
  for (const [id, c] of executable){
    if (!c.joints || c.kind === 'xfade') continue;
    const { d } = sample(c);
    for (const t of c.hold ? [0] : [0, d]){
      const j = clipJointsAt(c, t, defaults(c));
      for (const [k, v] of Object.entries(j))
        assert.ok(Math.abs(v - restOf(k)) < 1e-6,          // fingers rest at the REST grip (restOf), not 0 rad
          `${id} ${k}=${v} not REST at t=${t}`);
    }
  }
});

test('pose/hold clips settle to a steady end pose (held)', () => {
  for (const [id, c] of executable){
    if (!c.hold || !c.joints) continue;
    const p = defaults(c), d = clipDur(c, p);
    const a = clipJointsAt(c, d - 0.05, p), b = clipJointsAt(c, d, p);
    const beyond = clipJointsAt(c, d + 5, p);           // hold clamps past dur
    for (const k of Object.keys(b)){
      assert.ok(Math.abs(b[k] - a[k]) < 0.02, `${id} ${k} not steady at end`);
      assert.equal(b[k], beyond[k], `${id} ${k} must hold beyond dur`);
    }
  }
});

test('nav clips: |stick| ≤ 1 always, zeros at t=0 and t=dur', () => {
  for (const [id, c] of executable){
    if (!c.nav) continue;
    const { d, frames } = sample(c);
    for (const { t, nav } of frames)
      for (const [k, v] of Object.entries(nav))
        assert.ok(Math.abs(v) <= 1 + 1e-9, `${id} ${k}=${v} at t=${t}`);
    for (const t of [0, d]){
      const nav = clipNavAt(c, t, defaults(c));
      for (const [k, v] of Object.entries(nav))
        assert.ok(Math.abs(v) < 1e-6, `${id} nav.${k}=${v} not zero at t=${t}`);
    }
  }
});

// ── ChoreoTrack: NLE ops ─────────────────────────────────────────────────────
test("adding a 'both' clip creates two linked blocks, one per lane", () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const made = tr.add('groove', 2.0);
  assert.equal(made.length, 2);
  assert.deepEqual(made.map((b) => b.lane).sort(), ['manip', 'nav']);
  assert.ok(made[0].linkId && made[0].linkId === made[1].linkId);
  const one = tr.add('wave', 0);
  assert.equal(one.length, 1);
  assert.equal(one[0].lane, 'manip');
});

test('coming-soon actions are not addable', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  assert.equal(tr.add('boxing', 0), null);
  assert.equal(tr.blocks.length, 0);
});

test('cut splits a block into two slices that partition the clip range', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [b] = tr.add('wave', 1.0);                     // dur 5 → [1, 6]
  const right = tr.cut(b.id, 3.0);
  assert.ok(right);
  assert.equal(tr.blocks.length, 2);
  assert.ok(Math.abs(b.out - 2.0) < 1e-9);             // left slice: clip [0, 2]
  assert.ok(Math.abs(right.in - 2.0) < 1e-9);          // right slice: clip [2, 5]
  assert.ok(Math.abs(right.t0 - 3.0) < 1e-9);
  // each half plays only its slice: sampling at timeline t hits clip-local t
  const jLeft = tr.jointsAt(2.0);                      // inside left → clip t=1.0
  const jClip = clipJointsAt(MOTION_CLIPS.wave, 1.0);
  assert.ok(Math.abs(jLeft.right_arm_1 - jClip.right_arm_1) < 1e-9);
  const jRight = tr.jointsAt(4.0);                     // inside right → clip t=3.0
  const jClip3 = clipJointsAt(MOTION_CLIPS.wave, 3.0);
  assert.ok(Math.abs(jRight.right_arm_1 - jClip3.right_arm_1) < 1e-9);
  // no slivers
  assert.equal(tr.cut(b.id, 1.01), null);
});

test('trim clamps to clip duration; hold clips may extend', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [b] = tr.add('wave', 0);                       // dur 5, not hold
  tr.trim(b.id, 'r', 9.0);
  assert.ok(Math.abs(b.out - 5.0) < 1e-9, 'clamped to clip dur');
  tr.trim(b.id, 'l', 1.0);
  assert.ok(Math.abs(b.in - 1.0) < 1e-9 && Math.abs(b.t0 - 1.0) < 1e-9);
  const [h] = tr.add('pose_tall', 10);                 // hold: extendable
  tr.trim(h.id, 'r', 30);
  assert.ok(h.out > 10, 'hold clip extended');
});

test('snapTime: deterministic 0.1 s grid + block edges + playhead override', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('wave', 2.0);                                 // edges at 2.0 and 7.0
  assert.equal(tr.snapTime(3.333), 3.3);               // grid
  assert.equal(tr.snapTime(3.333), 3.3);               // deterministic
  assert.equal(tr.snapTime(7.06), 7.0);                // block edge wins
  assert.equal(tr.snapTime(4.46, { playhead: 4.5 }), 4.5); // playhead wins
  assert.equal(tr.snapTime(0.04), 0);                  // t=0 anchor
  assert.equal(SNAP_GRID, 0.1);
});

test('duplicate copies params, JSON roundtrip preserves blocks', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const [b] = tr.add('nav_forward', 1.0, { dist: 1.5 });
  const c = tr.duplicate(b.id, 8.0);
  assert.equal(c.params.dist, 1.5);
  assert.notEqual(c.id, b.id);
  const back = ChoreoTrack.fromJSON(JSON.parse(JSON.stringify(tr.toJSON())), MOTION_CLIPS);
  assert.equal(back.blocks.length, 2);
  assert.equal(back.byId(c.id).params.dist, 1.5);
});

test('navAt streams the active nav block, jointsAt merges manip + lift', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('nav_forward', 0, { dist: 0.5 });
  const mid = clipDur(MOTION_CLIPS.nav_forward, { dist: 0.5 }) / 2;
  assert.ok(tr.navAt(mid).ly > 0.2, 'forward stream mid-block');
  assert.equal(tr.navAt(100), null, 'nothing active later');
  tr.add('lift_up', 0);                                // nav-lane block with joints
  const j = tr.jointsAt(2.4);
  assert.ok(j.lift_lower > 0.25, 'lift joints ride the nav lane');
});

test('location gaps: detected, auto-fixed with a sized move-to-goal', () => {
  const reg = { ...MOTION_CLIPS,
    fake_kitchen_task: { title: 'Fake', emoji: '🍳', dur: 4, lane: 'manip',
      category: 'policy', status: 'available',
      location: { x: 1.2, y: 0.6, label: 'kitchen' },
      joints: () => ({ ...NEUTRAL }), nav: null } };
  const tr = new ChoreoTrack(reg);
  tr.add('fake_kitchen_task', 20);
  const gaps = detectGaps(tr, { x: 0, y: 0, label: 'home' });
  assert.equal(gaps.length, 1);
  assert.equal(gaps[0].to.label, 'kitchen');
  const ins = autoFixGaps(tr, { x: 0, y: 0, label: 'home' });
  assert.equal(ins.length, 1);
  assert.equal(ins[0].key, 'nav_goto');
  assert.ok(tr.end(ins[0]) <= 20 + 1e-9, 'goal block ends before the task');
  assert.equal(detectGaps(tr, { x: 0, y: 0, label: 'home' }).length, 0, 're-validate clean');
});

test('label-only locations gap but are not auto-fixable', () => {
  const reg = { ...MOTION_CLIPS,
    label_task: { title: 'L', emoji: 'x', dur: 4, lane: 'manip',
      category: 'policy', status: 'available',
      location: { label: 'laundry room' },
      joints: () => ({ ...NEUTRAL }), nav: null } };
  const tr = new ChoreoTrack(reg);
  tr.add('label_task', 10);
  assert.equal(detectGaps(tr, null).length, 1);
  assert.equal(autoFixGaps(tr, null).length, 0);
});

test('cut blocks inherit the parent clip (location metadata composes)', () => {
  const reg = { ...MOTION_CLIPS,
    loc_task: { title: 'T', emoji: 'x', dur: 6, lane: 'manip',
      category: 'policy', status: 'available',
      location: { x: 2, y: 2, label: 'spot' },
      joints: () => ({ ...NEUTRAL }), nav: null } };
  const tr = new ChoreoTrack(reg);
  const [b] = tr.add('loc_task', 0);
  const right = tr.cut(b.id, 3);
  assert.equal(right.key, 'loc_task');
  assert.equal(tr.clip(right).location.label, 'spot');
});

test('mergeActions: backend slots appear as coming-soon, locals stay executable', () => {
  const merged = mergeActions(MOTION_CLIPS, [
    { id: 'policy_dishes2', title: 'do dishes v2', category: 'policy',
      lane: 'manip', status: 'available',                // backend says available…
      location: { label: 'kitchen' }, note: 'no checkpoint yet' },
    { id: 'wave', title: 'Wave Hello (srv)' },
  ]);
  assert.equal(merged.policy_dishes2.status, 'coming_soon'); // …but nothing local executes it
  assert.equal(merged.policy_dishes2.note, 'no checkpoint yet');
  assert.equal(merged.wave.status, 'available');
  assert.equal(merged.wave.title, 'Wave Hello (srv)');
  assert.ok(merged.wave.joints, 'local exec preserved');
  const tr = new ChoreoTrack(merged);
  assert.equal(tr.add('policy_dishes2', 0), null, 'coming-soon not addable');
});

// ── AnnoTrack (language lane) + preview base-path integration ────────────────
import { AnnoTrack, integrateNavPath, NAV_SPEED, NAV_YAW_RATE }
  from '../../web/js/motions.js';

test('AnnoTrack: add/move/retime/edit/remove + JSON roundtrip', () => {
  const a = new AnnoTrack();
  let fired = 0; a.onChange = () => fired++;
  const n = a.add(2.0, 3.0, 'grab the red cup');
  a.move(n.id, -1);                                  // clamps to 0
  assert.equal(a.byId(n.id).t0, 0);
  a.retime(n.id, 0.05);                              // clamps to 0.2 min
  assert.equal(a.byId(n.id).dur, 0.2);
  a.setText(n.id, 'place it down');
  assert.equal(a.at(0.1)[0].text, 'place it down');
  assert.equal(a.at(5).length, 0);
  const back = AnnoTrack.fromJSON(JSON.parse(JSON.stringify(a.toJSON())));
  assert.equal(back.notes.length, 1);
  assert.equal(back.notes[0].text, 'place it down');
  a.remove(n.id);
  assert.equal(a.notes.length, 0);
  assert.ok(fired >= 5);
});

test('integrateNavPath: forward block moves −x, turn integrates yaw', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('nav_forward', 0, { dist: 0.5 });
  const path = integrateNavPath(tr, tr.duration() + 0.5);
  const end = path[path.length - 1];
  assert.ok(end.dx < -0.4 && end.dx > -0.6, `forward ≈0.5 m along −x, got ${end.dx}`);
  assert.ok(Math.abs(end.dy) < 0.02 && Math.abs(end.dyaw) < 0.02);

  const tr2 = new ChoreoTrack(MOTION_CLIPS);
  tr2.add('nav_turn_left', 0, { angle: 90 });
  const p2 = integrateNavPath(tr2, tr2.duration() + 0.5);
  const yaw = p2[p2.length - 1].dyaw * 180 / Math.PI;
  assert.ok(yaw > 80 && yaw < 100, `90° CCW, got ${yaw.toFixed(1)}°`);
});

// ── SkillStore: compose / import / insert (paste) roundtrip ──────────────────
import { SkillStore } from '../../web/js/skills.js';
import { SetpointSeq } from '../../web/js/sequence.js';

function fakeStorage(){
  const m = new Map();
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v) };
}
function fakeTimeline(){
  const tl = {
    track: new ChoreoTrack(MOTION_CLIPS),
    seq: new SetpointSeq(),
    annos: new AnnoTrack(),
    t: 0,
    duration(){ return Math.max(this.track.duration(), this.seq.duration(), this.annos.duration()); },
  };
  return tl;
}

test('compose a timeline into a skill and re-insert it shifted', () => {
  const store = new SkillStore(fakeStorage());
  const tl = fakeTimeline();
  tl.track.add('wave', 1.0);
  tl.track.add('groove', 6.0);                        // linked pair
  tl.seq.add(0.5, { torso: -0.2 }, 'lean');
  tl.annos.add(1.0, 2.0, 'say hi');
  const s = store.compose('greet & groove', tl);
  assert.equal(s.blocks, 3);
  assert.equal(s.keyframes, 1);
  assert.ok(Math.abs(s.dur - 15.0) < 1e-6);           // groove@6 + 9s

  const tl2 = fakeTimeline();
  const n = store.insertInto(s.id, tl2, 10.0);
  assert.equal(n, 3 + 1 + 1, 'blocks + keyframe + annotation');
  const waves = tl2.track.blocks.filter((b) => b.key === 'wave');
  assert.ok(Math.abs(waves[0].t0 - 11.0) < 1e-9, 'shifted by insertion time');
  const grooves = tl2.track.blocks.filter((b) => b.key === 'groove');
  assert.ok(grooves[0].linkId && grooves[0].linkId === grooves[1].linkId,
    'link preserved with fresh id');
  assert.equal(tl2.seq.keys[0].t, 10.5);
  assert.equal(tl2.annos.notes[0].t0, 11.0);
});

test('skill JSON export/import roundtrip; garbage rejected', () => {
  const store = new SkillStore(fakeStorage());
  const tl = fakeTimeline();
  tl.track.add('nav_forward', 0, { dist: 1.0 });
  const s = store.compose('go', tl);
  const store2 = new SkillStore(fakeStorage());
  const added = store2.importJSON(store.exportJSON(s.id));
  assert.equal(added.length, 1);
  assert.notEqual(added[0].id, s.id, 'fresh id on import');
  assert.equal(added[0].choreo.blocks[0].params.dist, 1.0);
  assert.equal(store2.importJSON('{"nope": true}').length, 0);
});

test('ChoreoTrack.paste skips unknown clips, offsets t0', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  const made = tr.paste({ blocks: [
    { key: 'wave', lane: 'manip', t0: 0, in: 0, out: 5 },
    { key: 'does_not_exist', lane: 'manip', t0: 1, in: 0, out: 2 },
  ] }, 3.0);
  assert.equal(made.length, 1);
  assert.equal(made[0].t0, 3.0);
});

// ── physics-preview sink helpers ─────────────────────────────────────────────
import { expandGrips, blockProgress } from '../../web/js/motions.js';
import { RANGE as HR } from '../../web/js/handmodel.js';

test('expandGrips: scalar grips become finger joints; wrist untouched', () => {
  const out = expandGrips({ left_arm_1: -1.0, right_grip: 1.0, left_grip: 0.0 });
  assert.equal(out.left_arm_1, -1.0, 'non-grip keys pass through');
  assert.ok(!('right_grip' in out) && !('left_grip' in out), 'grips consumed');
  assert.ok(!('right_wrist' in out) && !('left_wrist' in out),
    'grip must not steer the wrist');
  assert.ok(out.right_index_mcp > 1.0, 'full grip curls the index');
  assert.ok(Math.abs(out.left_index_mcp - (HR.index_mcp[0] +
    0.10 * (HR.index_mcp[1] - HR.index_mcp[0]))) < 0.15,
    'open grip ≈ flat pose');
  for (const [k, v] of Object.entries(out)){
    const m = k.match(/^(left|right)_(.+)$/);
    if (m && HR[m[2]]){
      const [lo, hi] = rangeFor(m[2], m[1]);
      assert.ok(v >= lo - 1e-9 && v <= hi + 1e-9, `${k} within calibrated range`);
    }
  }
});

test('blockProgress counts started blocks for the status chip', () => {
  const tr = new ChoreoTrack(MOTION_CLIPS);
  tr.add('wave', 0);
  tr.add('nav_forward', 6);
  tr.add('happy', 10);
  assert.deepEqual(blockProgress(tr, -1), { n: 0, m: 3 });
  assert.deepEqual(blockProgress(tr, 0), { n: 1, m: 3 });
  assert.deepEqual(blockProgress(tr, 7), { n: 2, m: 3 });
  assert.deepEqual(blockProgress(tr, 99), { n: 3, m: 3 });
});
