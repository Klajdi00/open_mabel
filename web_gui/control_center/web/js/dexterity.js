// Dexterity — the controls for ONE ORCA hand:
//   · grasp slider (flat→fist blend)
//   · POSES (the 9 firmware presets + the animation library's named shapes)
//   · ANIMATIONS (handanims.js, the same library for either hand) streamed
//     client-side at 20 Hz; both sections fold (click the heading)
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

// ── hand animations + named shapes: handanims.js (one library, both hands) ──
import { ANIMATIONS, SHAPES } from './handanims.js';
// the open hand, WITHOUT a wrist value: a static pose never steers the wrist
const flat = () => { const p = graspPose(0); delete p.wrist; return p; };

export class Dexterity {
  constructor(panel, link, getJointPos, opts = {}){
    this.body = panel.body;
    this.link = link;
    this.getJointPos = getJointPos;      // () => {jointName: rad}
    this.side = opts.side || 'right';    // one column per hand in the dock
    this._build();
    this._offTelemetry = bus.on('robotState', () => this._reflect());
  }

  _send(pctPose, keepWrist = false){
    // A static pose or the grasp slider never steers the wrist (every firmware preset carries
    // `wrist: 50`, which would snap a deliberately turned wrist back to neutral). Animations
    // author the wrist on purpose and pass keepWrist; the wrist's own slider sends {wrist} alone.
    if (!keepWrist && Object.keys(pctPose).length > 1 && 'wrist' in pctPose){
      pctPose = { ...pctPose }; delete pctPose.wrist;
    }
    const { name, position } = poseToCommand(this.side, pctPose);
    const joints = {};
    name.forEach((n, i) => { joints[n] = position[i]; });
    this.link.joints(joints);
  }

  // a folding section: the heading toggles its body; folded unless the operator opened it before
  _fold(title, body){
    const h = document.createElement('div');
    h.className = 'dex-sec mono dex-fold';
    const k = `dex.fold.${title}`; let open = false;
    try { open = localStorage.getItem(k) === '1'; } catch {}
    const paint = () => { h.textContent = `${open ? '▾' : '▸'} ${title}`; body.hidden = !open; };
    h.onclick = () => { open = !open; try { localStorage.setItem(k, open ? '1' : '0'); } catch {} paint(); };
    paint();
    return h;
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

    // per-hand poses: the firmware presets, then the library's shapes they do not already name
    const poses = document.createElement('div');
    poses.className = 'dex-poses';
    host.appendChild(this._fold('POSES', poses));
    for (const [name, pose] of [...Object.entries(POSES), ...Object.entries(SHAPES).filter(([n]) => !POSES[n])]){
      const b = document.createElement('button');
      b.className = 'gbtn';
      b.textContent = name.replace(/_/g, ' ');
      b.onclick = () => { this._stopAnim(); this._send(pose); };
      poses.appendChild(b);
    }
    host.appendChild(poses);

    // per-hand animations — one list, the same library on the left and the right hand
    const anims = document.createElement('div');
    anims.className = 'dex-poses';
    host.appendChild(this._fold('ANIMATIONS', anims));
    for (const [name, a] of Object.entries(ANIMATIONS)){
      const b = document.createElement('button');
      b.className = 'gbtn';
      b.textContent = a.title || name;
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
      for (const s in this._jrows) delete this._jrows[s].dataset.touched;   // _jrows[s] IS the row
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
      if (t >= a.dur){ this._stopAnim(); this._send({ ...flat(), wrist: 50 }, true); return; }
      const pose = a.at(t);
      if (pose) this._send(pose, true);
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
