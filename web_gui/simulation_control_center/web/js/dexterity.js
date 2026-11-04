// Dexterity — the controls for ONE ORCA hand:
//   · grasp slider (flat→fist blend)
//   · POSES (the 9 firmware presets)
//   · ANIMATIONS (natural keyframed motions: wave, count, piano, breathe,
//     rock-paper-scissors) streamed client-side at 20 Hz
//   · all 17 joint sliders, always visible, TARGET (slider) vs CURRENT
//     (live tick + readout) — no joint/finger/hand mode switching
//
// A hand is a MODULE, not a dock: the module rail selects L/R Hand and these
// controls render inside the module panel (jointspace.js), beside the arm /
// neck / lift / torso sliders. Webcam hand control lives on the full-body
// webcam-teleop path (bodyteleop.js), which retargets fingers with the same
// anatomical decomposition — see retargetLandmarks below.
// Commands go to the bridge as side-prefixed joint targets (joint_command);
// torque follows e-stop server-side.
import * as bus from './bus.js';
import { HAND_JOINTS, FINGERS, POSES, RANGE, graspPose, poseToCommand,
         pctToRad, radToPct } from './handmodel.js';
import { decomposeHand } from './bodyteleop.js';

// ── natural hand animations: t (s) → pct pose (null = finished) ─────────────
const flat = () => graspPose(0);
const withFingers = (base, fn) => {
  const p = { ...base };
  for (const [f, i] of [['index', 0], ['middle', 1], ['ring', 2], ['pinky', 3]])
    fn(p, f, i);
  return p;
};
export const ANIMATIONS = {
  // fingers ripple down and back up, like waving hello
  wave: { dur: 3.2, at: (t) => withFingers(flat(), (p, f, i) => {
    const ph = Math.sin(t * 4.2 - i * 0.9);
    const c = Math.max(0, ph) * 82;
    p[`${f}_mcp`] = c; p[`${f}_pip`] = c;
  }) },
  // count 1-2-3-4-5 on the fingers, then relax
  count: { dur: 5.0, at: (t) => {
    const step = Math.min(5, Math.floor(t / 0.8) + 1);
    const p = graspPose(1);                 // start from a fist
    const order = ['index', 'middle', 'ring', 'pinky'];
    order.forEach((f, i) => {
      const open = step >= i + 1;
      p[`${f}_mcp`] = open ? 4 : 88; p[`${f}_pip`] = open ? 4 : 88;
    });
    if (step >= 5){ p.thumb_mcp = 8; p.thumb_pip = 8; p.thumb_dip = 8; p.thumb_abd = 20; }
    return t < 4.6 ? p : flat();
  } },
  // quick alternating fingertip taps, like playing piano
  piano: { dur: 4.0, at: (t) => withFingers(flat(), (p, f, i) => {
    const tap = Math.max(0, Math.sin(t * 9 - i * 1.6));
    const c = 12 + tap * 62;
    p[`${f}_mcp`] = c * 0.55; p[`${f}_pip`] = c;
  }) },
  // slow, organic open↔close
  breathe: { dur: 6.0, at: (t) =>
    graspPose(0.5 - 0.5 * Math.cos(t * (2 * Math.PI / 3))) },
  // rock → paper → scissors, one second each, twice
  rps: { dur: 6.2, at: (t) => {
    const phase = Math.floor(t) % 3;
    if (phase === 0) return graspPose(1);                       // rock
    if (phase === 1) return flat();                             // paper
    const p = graspPose(1);                                     // scissors
    for (const f of ['index', 'middle']){ p[`${f}_mcp`] = 5; p[`${f}_pip`] = 5; }
    return p;
  } },
};

export class Dexterity {
  constructor(panel, link, getJointPos, opts = {}){
    this.body = panel.body;
    this.link = link;
    this.getJointPos = getJointPos;      // () => {jointName: rad}
    this.side = opts.side || 'right';    // one column per hand in the dock
    this._build();
    this._offTelemetry = bus.on('robotState', () => this._reflect());
  }

  _send(pctPose){
    const { name, position } = poseToCommand(this.side, pctPose);
    const joints = {};
    name.forEach((n, i) => { joints[n] = position[i]; });
    this.link.joints(joints);
  }

  _slider(label, min, max, val, step = 1){
    const row = document.createElement('div');
    row.className = 'dex-row';
    row.innerHTML = `<span class="dex-lab">${label}</span>
      <input type="range" min="${min}" max="${max}" step="${step}" value="${val}">
      <b class="dex-val mono">–</b><i class="dex-cur" title="current"></i>`;
    return row;
  }

  _build(){
    this.body.classList.add('dex');
    this.body.innerHTML = '';
    const host = this.body;

    // grasp
    const grasp = this._slider('grasp', 0, 100, 0);
    {
      const inp = grasp.querySelector('input'), val = grasp.querySelector('.dex-val');
      inp.oninput = () => {
        this._stopAnim();
        val.textContent = (+inp.value).toFixed(0) + '%';
        this._send(graspPose(+inp.value / 100));
      };
    }
    host.appendChild(grasp);

    // per-hand poses
    const ph = document.createElement('div');
    ph.className = 'dex-sec mono';
    ph.textContent = 'POSES';
    host.appendChild(ph);
    const poses = document.createElement('div');
    poses.className = 'dex-poses';
    for (const name of Object.keys(POSES)){
      const b = document.createElement('button');
      b.className = 'gbtn';
      b.textContent = name.replace('_', ' ');
      b.onclick = () => { this._stopAnim(); this._send(POSES[name]); };
      poses.appendChild(b);
    }
    host.appendChild(poses);

    // per-hand animations
    const ah = document.createElement('div');
    ah.className = 'dex-sec mono';
    ah.textContent = 'ANIMATIONS';
    host.appendChild(ah);
    const anims = document.createElement('div');
    anims.className = 'dex-poses';
    for (const name of Object.keys(ANIMATIONS)){
      const b = document.createElement('button');
      b.className = 'gbtn';
      b.textContent = { rps: 'rock·paper·scissors' }[name] || name;
      b.onclick = () => this._toggleAnim(name, b);
      anims.appendChild(b);
    }
    host.appendChild(anims);

    // all 17 joints, target vs current, always visible
    const jh = document.createElement('div');
    jh.className = 'dex-sec mono';
    jh.textContent = 'JOINTS · target ▸ current';
    host.appendChild(jh);
    this._jrows = {};
    for (const [suffix] of HAND_JOINTS){
      // start at the hand's CURRENT position (not mid-range); the thumb keeps
      // following the live joint until the operator grabs it
      const jp = this.getJointPos?.() || {};
      const cur = jp[`${this.side}_${suffix}`];
      const init = cur != null ? radToPct(suffix, cur, this.side) : 50;
      const row = this._slider(suffix, 0, 100, Math.round(init));
      const inp = row.querySelector('input'), val = row.querySelector('.dex-val');
      inp.oninput = () => {
        this._stopAnim();
        row.dataset.touched = '1';
        val.textContent = (+inp.value).toFixed(0) + '%';
        this._send({ [suffix]: +inp.value });
      };
      host.appendChild(row);
      this._jrows[suffix] = row;
    }
    this._reflect();
  }

  // Re-seat every finger slider on the measured pose and DROP "touched", so the
  // panel starts where the hand IS. Called when torque (re)engages: the robot
  // holds its measured pose, so the sliders must show that — not a grip the
  // operator dialled in while the hand was limp. (ModulePanel.resync cascades
  // here for the hand modules; the arm/neck sliders have their own copy.)
  resync(){
    this._stopAnim();
    if (this._jrows)
      for (const s in this._jrows) delete this._jrows[s].row.dataset.touched;
    this._reflect();
  }

  _toggleAnim(name, btn){
    if (this._anim === name){ this._stopAnim(); return; }
    this._stopAnim();
    this._anim = name;
    this._animBtn = btn;
    btn.classList.add('on');
    const a = ANIMATIONS[name];
    const t0 = performance.now();
    this._animTimer = setInterval(() => {
      const t = (performance.now() - t0) / 1000;
      if (t >= a.dur){ this._stopAnim(); this._send(flat()); return; }
      const pose = a.at(t);
      if (pose) this._send(pose);
    }, 50);
  }
  _stopAnim(){
    if (this._animTimer) clearInterval(this._animTimer);
    this._animBtn?.classList.remove('on');
    this._anim = this._animTimer = this._animBtn = null;
  }

  // target (slider) vs CURRENT (tick + numeric) from live joint telemetry
  _reflect(){
    if (!this._jrows || !this.getJointPos) return;
    const jp = this.getJointPos() || {};
    for (const suffix in this._jrows){
      const cur = jp[`${this.side}_${suffix}`];
      const row = this._jrows[suffix];
      const tick = row.querySelector('.dex-cur');
      const val = row.querySelector('.dex-val');
      if (cur == null){ tick.style.opacity = 0; continue; }
      const pct = radToPct(suffix, cur, this.side);
      const inp = row.querySelector('input');
      // Park the live tick on the SLIDER's real geometry (offsetLeft/Width in
      // the row, its offset parent). The old `calc(pct% * .6 + 28%)` baked in
      // the wide dock's label/slider proportions and drifts in any other layout
      // — and the hand now lives in the narrow module panel.
      if (inp.offsetWidth){
        tick.style.opacity = 1;
        tick.style.left = `${inp.offsetLeft + (pct / 100) * inp.offsetWidth}px`;
      }
      // untouched sliders track the live hand — the thumb IS the current pose
      // until the operator takes over
      if (!row.dataset.touched && !this._anim) inp.value = Math.round(pct);
      const tgt = inp.value;
      val.textContent = `${(+tgt).toFixed(0)}▸${pct.toFixed(0)}%`;
    }
  }

  destroy(){ this._stopAnim(); this._offTelemetry?.(); }
}

// landmark → per-joint retarget: the SAME anatomical decomposition the
// full-body webcam path uses (bodyteleop.decomposeHand: MCP/PIP flexion in the
// finger's sagittal plane, thumb MP/IP down the thumb chain, thumb opposition
// from the metacarpal's dive below the palm plane), mapped rad → percent
// through the calibrated ORCA joint ranges (handmodel.radToPct) instead of a
// heuristic mean-curl. Prefer metric worldLandmarks; normalized also works
// (angles are scale-invariant; only aspect distortion is lost).
export function retargetLandmarks(lms, chirality = 'right'){
  // chirality IS the hand: rad<->pct is side-dependent (mirrored wrist).
  const side = chirality === 'left' ? 'left' : 'right';
  const w0 = [lms[0].x, lms[0].y, lms[0].z];
  // MediaPipe camera coords → ARKit-style (x,y negated) hand-centered shape.
  const shape = lms.map((p) => [-(p.x - w0[0]), -(p.y - w0[1]), p.z - w0[2]]);
  const dec = decomposeHand(shape, chirality);
  const pose = {};
  if (!dec){                       // degenerate palm: safe neutral
    for (const f of ['index', 'middle', 'ring', 'pinky', 'thumb'])
      for (const s of f === 'thumb' ? ['mcp', 'pip', 'dip', 'abd'] : ['mcp', 'pip', 'abd'])
        pose[`${f}_${s}`] = s === 'abd' ? 50 : 0;
    return pose;
  }
  const fingerKey = { index: 'index', middle: 'middle', ring: 'ring', little: 'pinky' };
  for (const [f, k] of Object.entries(fingerKey)){
    const A = dec.fingers[f];
    pose[`${k}_mcp`] = radToPct(`${k}_mcp`, A.mcp, side);
    // ORCA PIP tendon-couples the anatomical PIP+DIP pair: take the dominant
    // PIP and credit half the DIP so a distal-only curl still registers.
    pose[`${k}_pip`] = radToPct(`${k}_pip`, A.pip + 0.5 * Math.max(0, A.dip), side);
    pose[`${k}_abd`] = 50;         // splay left neutral (sign is per-mount)
  }
  // Thumb URDF naming remap (handmodel.js): thumb_mcp=ORCA CMC (opposition),
  // thumb_dip=ORCA MP, thumb_pip=ORCA IP.
  const T = dec.thumb;
  // Opposition = the metacarpal's dive below the palm plane (−ẑ dorsal): the
  // palm-frame z component of the stored unit direction (dir is palm coords).
  const opp = Math.asin(Math.max(-1, Math.min(1, -T.dir[2])));
  pose.thumb_mcp = radToPct('thumb_mcp', opp, side);
  pose.thumb_dip = radToPct('thumb_dip', T.mp, side);
  pose.thumb_pip = radToPct('thumb_pip', T.ip, side);
  pose.thumb_abd = 50;
  return pose;
}
