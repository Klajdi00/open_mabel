// The MODULE control panel (manipulate → Joint space) — the web twin of the
// iOS app's `moduleControls`. The rail picks ONE module; this panel shows only
// that module's controls, so the operator sees seven joints, not fifty-nine:
//
//   L/R Arm   · one slider per joint (7-DOF chain, limits from the rig manifest)
//   Neck      · yaw / pitch / roll
//   Lift      · ONE height setpoint driving both stages (they are mechanically
//               synced on the robot — two sliders would be a lie you can fight)
//   Torso     · pitch
//   L/R Hand  · the ORCA dexterity controls (grasp · poses · animations · the
//               17 joints) — a hand is a MODULE, not a separate panel. This is
//               what replaced the old both-hands dock.
//
// Sliders are TARGETS. Any slider the operator is not touching re-seats itself
// to the measured pose every telemetry tick, so the panel is always a truthful
// picture of the robot and grabbing a slider never snaps the joint.
import * as bus from './bus.js';
import { MODULES, moduleById, moduleJoints, isHand, iconSvg } from './modules.js';
import { Dexterity } from './dexterity.js';

// Human names for the rig's joint ids (same vocabulary as iOS `JointSpec`).
const LABEL = {
  arm: ['Shoulder Pitch', 'Shoulder Roll', 'Upper Yaw', 'Elbow',
        'Forearm Yaw', 'Wrist Pitch', 'Wrist Roll'],
  neck_yaw: 'Yaw', neck_pitch: 'Pitch', neck_roll: 'Roll',
  torso: 'Pitch',
};
const labelFor = (name) => {
  const arm = /^(left|right)_arm_(\d)$/.exec(name);
  if (arm) return LABEL.arm[+arm[2] - 1] || name;
  return LABEL[name] || name;
};

export class ModulePanel {
  constructor(el, link, getJointPos){
    this.el = el;
    this.link = link;
    this.getJointPos = getJointPos;       // () => {jointName: rad}
    this.module = null;
    this.rows = {};                       // jointName → {inp, val}
    this._holding = null;                 // the slider under the operator's finger
    this._hands = {};                     // side → Dexterity (built on first use)
    this._liftJog = null;                 // active hold-to-jog ramp (lift), or null
    bus.on('robotState', () => this._reflect());
  }

  build(manifest){
    if (this._built || !manifest) return;
    this._built = true;
    this.limits = {};
    for (const j of manifest.joints)
      this.limits[j.name] = [j.lower ?? -3.14, j.upper ?? 3.14];
    this.names = manifest.joints.map(j => j.name);
    if (this.module) this.setModule(this.module, true);
  }

  /** Show one module's controls. Rebuilds only when the module changes. */
  setModule(id, force = false){
    if (id === this.module && !force) return;
    this.module = id;
    if (!this._built) return;             // manifest not in yet — build() replays
    const m = moduleById(id);
    if (!m) return;

    this._stopLiftJog();                  // leaving the panel ends any live jog ramp
    this.rows = {};
    this._holding = null;
    this.el.innerHTML = '';
    this.el.appendChild(this._head(m));

    const body = document.createElement('div');
    body.className = 'mp-body';
    this.el.appendChild(body);

    if (isHand(id)) this._buildHand(body, m.side);
    else if (id === 'lift') this._buildLift(body);
    else for (const name of moduleJoints(id, this.names))
      body.appendChild(this._slider(name, labelFor(name)));

    this._reflect();
  }

  _head(m){
    const h = document.createElement('div');
    h.className = 'mp-head';
    h.innerHTML = `<span class="mp-title mono">${m.label.toUpperCase()}</span>
      <span class="mp-hint mono">target ▸ live</span>`;
    return h;
  }

  // ── one joint, one slider ────────────────────────────────────────────────
  // A slider is a TARGET, not a mirror. It used to be re-seated to the measured
  // angle on every telemetry tick (50 Hz), so the instant you let go of the knob it
  // sprang back to wherever the joint actually was — and if the joint was limp, or
  // merely lagging, that was "the slider does nothing / won't stay put". Now the
  // slider OWNS its target once you touch it (dexterity.js's target ▸ current model,
  // and iOS's syncTargetsToState); the measured angle rides alongside as the live
  // read-out + tick, so tracking error is visible instead of being papered over.
  _slider(name, label, opt = {}){
    const [lo, hi] = this.limits[name] || [-3.14, 3.14];
    const dec = opt.dec ?? 2;
    const row = document.createElement('div');
    row.className = 'mp-row';
    row.innerHTML = `<span class="mp-lab">${label}</span>
      <input type="range" min="${lo}" max="${hi}" step="${opt.step ?? 0.005}" value="0">
      <b class="mp-val mono">–</b><i class="mp-cur" title="measured"></i>`;
    const inp = row.querySelector('input');
    const val = row.querySelector('.mp-val');
    const send = opt.send || ((v) => this.link.joints({ [name]: v }));
    this.rows[name] = { inp, val, row, dec, fmt: opt.fmt, cur: row.querySelector('.mp-cur') };
    inp.addEventListener('pointerdown', () => { this._holding = name; });
    const release = () => { if (this._holding === name) this._holding = null; };
    inp.addEventListener('pointerup', release);
    inp.addEventListener('pointercancel', release);
    inp.addEventListener('blur', release);
    inp.addEventListener('input', () => {
      row.dataset.touched = '1';          // the operator owns this target now
      send(+inp.value);
    });
    return row;
  }

  /** Re-seat every slider on the measured pose and hand the joints back to the
   *  robot (drop "touched"). The controls must start where the robot IS — on a
   *  module switch and whenever teleop (re)engages — or the first touch would
   *  fling the joint from a stale target. iOS calls this syncTargetsToState(). */
  resync(){
    for (const name in this.rows) delete this.rows[name].row.dataset.touched;
    // a hand module's controls ARE a Dexterity panel, not slider rows — cascade
    // so the finger sliders re-seat to measured on energise too
    for (const side in this._hands) this._hands[side].resync();
    this._reflect();
  }

  // ── lift: both stages ride ONE height setpoint ───────────────────────────
  // Two ways to drive it, both down the SAME absolute-position path (joint_command
  // → /lift/command → Pico closed loop):
  //   · hold-to-jog (▲/▼) — DIRECTION control: press ramps a height target at a
  //     fixed rate, release snaps the target to the measured height so it stops
  //     where it is. No velocity topic — the Pico just follows a moving setpoint.
  //   · slider — go straight to an absolute height.
  _buildLift(body){
    // hold-to-jog, modelled on the manipulate sticks' up/down hold-buttons.
    const jog = document.createElement('div');
    jog.className = 'mp-jog';
    jog.innerHTML =
      `<button data-dir="1" title="hold to raise the lift">▲&nbsp;Up</button>` +
      `<span class="mp-jog-lab mono">hold&nbsp;·&nbsp;release&nbsp;stops</span>` +
      `<button data-dir="-1" title="hold to lower the lift">▼&nbsp;Down</button>`;
    jog.querySelectorAll('button').forEach((b) => {
      const dir = +b.dataset.dir;
      b.addEventListener('pointerdown', (e) => {
        try { b.setPointerCapture(e.pointerId); } catch { /* not capturable */ }
        this._startLiftJog(dir);
      });
      const stop = () => this._stopLiftJog();
      b.addEventListener('pointerup', stop);
      b.addEventListener('pointercancel', stop);
      b.addEventListener('lostpointercapture', stop);
    });
    body.appendChild(jog);

    // reflect off the LOWER stage; command BOTH (the robot's sync-lift equality)
    body.appendChild(this._slider('lift_lower', 'Height', {
      step: 0.001, dec: 3,
      send: (v) => this.link.joints({ lift_lower: v, lift_upper: v }),
    }));
    const note = document.createElement('div');
    note.className = 'mp-note';
    note.textContent = '▲/▼ hold = jog (release stops) · slider = go to height. '
      + 'Both stages are mechanically synced — one height, one command.';
    body.appendChild(note);
  }

  // Hold-to-jog: ramp the height TARGET at a fixed rate and stream it as an
  // absolute joint_command — the SAME path the slider uses, so nothing new is
  // needed downstream. 0.04 m/s stays under the Pico's 60 mm/s profile ceiling,
  // so the closed loop tracks the moving setpoint instead of chasing a step.
  _startLiftJog(dir){
    this._stopLiftJog();                       // never stack two ramps
    const [lo, hi] = this.limits['lift_lower'] || [0, 1];
    // Rate is a fraction of the slider's OWN range (full travel in ~16 s) rather
    // than a fixed m/s: lift_lower may be half-stroke units (the URDF splits the
    // column into two prismatic joints), so a fixed m/s could translate to ~2×
    // real speed and outrun the Pico's 60 mm/s profile. Range-relative keeps the
    // real motion ~40 mm/s regardless of the unit scaling.
    const HZ = 30, dt = 1 / HZ;
    const RATE = Math.max(1e-4, (hi - lo) / 16);   // slider-units per second
    const row = this.rows['lift_lower'];
    const measured = () => this.getJointPos?.()?.lift_lower;
    // seed from where the lift ACTUALLY is (fall back to the slider) so a jog
    // never begins with a jump
    let seed = measured();
    if (seed == null) seed = row ? +row.inp.value : lo;
    const job = { dir, target: seed };
    this._liftJog = job;
    const tick = () => {
      if (this._liftJog !== job) return;
      job.target = Math.min(hi, Math.max(lo, job.target + dir * RATE * dt));
      this.link.joints({ lift_lower: job.target, lift_upper: job.target });
      if (row){ row.row.dataset.touched = '1'; row.inp.value = job.target; }
      this._reflect();
    };
    // window-level safety: a pointer released outside the button, or the tab
    // losing focus mid-hold, must still stop the ramp (belt-and-suspenders on
    // top of the button's own pointerup).
    job.safety = () => this._stopLiftJog();
    window.addEventListener('pointerup', job.safety);
    window.addEventListener('blur', job.safety);
    tick();                                    // move on the first frame
    job.timer = setInterval(tick, 1000 / HZ);
  }

  // Release / module-switch / focus-loss: stop ramping and hold where the lift IS
  // now (snap the target to the measured height → the loop settles, no coast past).
  _stopLiftJog(){
    const job = this._liftJog; this._liftJog = null;
    if (!job) return;
    clearInterval(job.timer);
    if (job.safety){
      window.removeEventListener('pointerup', job.safety);
      window.removeEventListener('blur', job.safety);
    }
    const cur = this.getJointPos?.()?.lift_lower;
    const hold = (cur == null) ? job.target : cur;
    this.link.joints({ lift_lower: hold, lift_upper: hold });
    const row = this.rows['lift_lower'];
    if (row) row.inp.value = hold;
    this._reflect();
  }

  // ── hand: the ORCA dexterity controls, in-panel ──────────────────────────
  _buildHand(body, side){
    body.classList.add('mp-hand');
    const host = document.createElement('div');
    body.appendChild(host);
    // one Dexterity per side, kept alive across module switches (it owns its
    // own telemetry subscription + animation timers)
    if (!this._hands[side])
      this._hands[side] = new Dexterity({ body: host }, this.link,
        () => this.getJointPos?.(), { side });
    else host.appendChild(this._hands[side].body);
  }

  // ── TARGET (slider) vs CURRENT (tick + read-out) ─────────────────────────
  // An UNTOUCHED slider tracks the live joint, so the panel always opens on the
  // robot's real pose. Once the operator moves one, it holds their target and the
  // measured angle shows beside it — the joint is then free to lag, fight, or fail
  // to reach it, and you can SEE that instead of the slider quietly lying.
  _reflect(){
    if (this.el.classList.contains('hidden')) return;
    const jp = this.getJointPos?.() || {};
    for (const name in this.rows){
      const r = this.rows[name];
      const v = jp[name];
      if (v == null){ r.cur.style.opacity = 0; continue; }
      const held = name === this._holding;
      const owned = r.row.dataset.touched === '1';
      if (!held && !owned) r.inp.value = v;          // idle: follow the robot
      const lo = +r.inp.min, hi = +r.inp.max;
      const pct = hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0;
      r.cur.style.opacity = 1;
      r.cur.style.left = `${r.inp.offsetLeft + pct * r.inp.offsetWidth}px`;
      const f = r.fmt || ((x) => x.toFixed(r.dec));     // a slider may carry its own unit format
      r.val.textContent = owned ? `${f(+r.inp.value)}▸${f(v)}` : f(v);
    }
  }

  show(on){ this.el.classList.toggle('hidden', !on); }
}

/**
 * The module rail — the vertical nav bar down the left edge of the viewer,
 * the web twin of the iOS `moduleRail`. It is the ONE place the operator says
 * "this is the part of the robot I am working on", and it means the same thing
 * in both manipulate spaces:
 *   Task  → which wrist the sticks and arrow keys drive
 *   Joint → which module's controls the panel shows
 * Either way, picking a module flies the 3-D camera to it.
 */
export class ModuleRail {
  constructor(el, onPick){
    this.el = el;
    this.onPick = onPick;
    this.module = null;
    this.btns = {};
    el.classList.add('mod-rail');     // ADD — assigning className would drop `hidden`
    for (const m of MODULES){
      const b = document.createElement('button');
      b.className = 'mod-chip';
      b.dataset.m = m.id;
      b.title = m.label;
      b.innerHTML = iconSvg(m) +
        (m.side ? `<i class="mod-side">${m.side === 'left' ? 'L' : 'R'}</i>` : '') +
        `<span class="mod-lab">${m.label}</span>`;
      b.onclick = () => this.select(m.id);
      el.appendChild(b);
      this.btns[m.id] = b;
    }
  }

  select(id, silent = false){
    if (id === this.module) return;
    this.module = id;
    for (const k in this.btns) this.btns[k].classList.toggle('on', k === id);
    if (!silent) this.onPick?.(id);
  }

  show(on){ this.el.classList.toggle('hidden', !on); }
}

// re-exported so the rail and the panel never disagree about the vocabulary
export { MODULES };
