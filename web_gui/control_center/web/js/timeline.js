// Timeline v2 — long-horizon policy composer. Two labeled lanes:
//   NAVIGATION   — base motion: nav primitives, move-to-goal, dances' sway
//   MANIPULATION — arms/hands/torso/lift: setpoint keyframes (◆, the v1
//                  feature, unchanged) + action blocks from the library
// NLE editing on blocks: drag w/ deterministic snapping (0.1 s grid + block
// edges + playhead), trim edges (clamped to clip duration; hold-clips extend),
// blade/cut at the playhead (B), alt-drag duplicate, delete. 'both'-lane clips
// drop one linked block per lane. Location-aware gap detection renders ⚠
// markers on the nav lane; Auto-fix (toolbar toggle, default ON) inserts
// move-to-goal blocks sized by distance/speed. Run = EXEC_RATE choreography:
// manipulation → joint_command (keyframes merged under blocks), navigation →
// teleop_frame stream (zeros between blocks, zeros streamed at stop, never a
// non-zero last frame); goal blocks ride the bridge wire (nav_goal -> ROS
// Nav2) when cfg.ros, else POST /api/nav/goal + poll status + cancel at
// block end (graceful toast when the backend lacks the endpoint).
import * as bus from './bus.js';
import { conf } from './config.js';
import { SetpointSeq } from './sequence.js';
import { MOTION_CLIPS, ChoreoTrack, AnnoTrack, mergeActions, detectGaps,
         autoFixGaps, clipDur, defaults, clipJointsAt, integrateNavPath,
         detectPoseJumps, autoFixPoseJumps, makeBlendBlock,
         expandGrips, blockProgress, liftRateLimit, EXEC_RATE } from './motions.js';

const LS_KEY = 'mabel.cc.setpoints.v1';
const LS_CHOREO = 'mabel.cc.choreo.v1';
const LS_ANNOS = 'mabel.cc.annos.v1';

const LANES = [
  { id: 'nav', label: 'NAVIGATION' },
  { id: 'manip', label: 'MANIPULATION' },
];
// module sub-tracks for the expandable manipulation lane (read-only viz)
const MODULES = [
  ['left arm', (j) => j.startsWith('left_arm_')],
  ['right arm', (j) => j.startsWith('right_arm_')],
  ['left hand', (j) => /^left_(wrist|grip|thumb|index|middle|ring|pinky)/.test(j)],
  ['right hand', (j) => /^right_(wrist|grip|thumb|index|middle|ring|pinky)/.test(j)],
  ['torso · neck', (j) => j.startsWith('torso') || j.startsWith('neck')],
  ['lift', (j) => j.startsWith('lift')],
];
const ZNAV = { lx: 0, ly: 0, rx: 0, ry: 0 };

// NAV PRIMITIVES ON THE ROBOT. A gateway that advertises `nav_move` (hello.caps) plays the move itself, open loop
// from its own calibration table — the stick profile below assumes 0.8 m/s / 1.5 rad/s per unit, which not every
// base drives at. So the block sends ONE nav_move at its start and streams zeros (which never preempt); without
// the cap (MABEL's gateway today) the old stick stream runs unchanged. Clip ids and params are unchanged.
const NAV_MOVES = { nav_forward: ['dx', 1], nav_back: ['dx', -1], nav_strafe_left: ['dy', 1],
                    nav_strafe_right: ['dy', -1], nav_turn_left: ['dyaw', 1], nav_turn_right: ['dyaw', -1] };
export const NAV_MOVE_SPEED = 0.5;              // fraction of the operator's speed caps (the gateway's default)
export function navMoveFor(key, params, registry = MOTION_CLIPS){
  const m = NAV_MOVES[key];
  if (!m) return null;
  const p = { ...defaults(registry[key]), ...(params || {}) };
  const amount = m[0] === 'dyaw' ? (+p.angle) * Math.PI / 180 : +p.dist;
  return { dx: 0, dy: 0, dyaw: 0, [m[0]]: m[1] * amount, speed: NAV_MOVE_SPEED };
}
// The time the robot will take (s), from hello.navCalibration + the speed caps, as the gateway plans it.
export function navMoveSeconds(move, cal, caps){
  if (!move || !cal?.kinds) return null;
  const axis = move.dyaw ? 'dyaw' : move.dx ? 'dx' : 'dy';
  const v = move[axis];
  const kind = { dx: v > 0 ? 'forward' : 'back', dy: v > 0 ? 'strafe_left' : 'strafe_right',
                 dyaw: v > 0 ? 'turn_left' : 'turn_right' }[axis];
  const ents = [...(cal.kinds[kind]?.entries || [])].sort((a, b) => a.speed - b.speed);
  if (!ents.length) return null;
  const want = move.speed * (axis === 'dyaw' ? caps.wmax : caps.vmax);
  const e = ents.filter((x) => x.speed <= want + 1e-9).pop() || ents[0];
  return Math.max(0, Math.abs(v) / e.rate + e.t0);
}

export class Timeline {
  constructor(els, link, ctl, opts = {}){
    this.scroll = els.scroll;
    this.ruler = els.ruler;
    this.tracks = els.tracks;
    this.playhead = els.playhead;
    this.toolbar = els.toolbar;
    this.link = link;
    this.ctl = ctl;
    this.getPose = opts.getPose || (() => null);        // current pose
    this.getTarget = opts.getTarget || (() => null);    // commanded pose
    this.onPreview = opts.onPreview || (() => {});      // preview pose | null
    this.onExecuteView = opts.onExecuteView || null;    // viewer: preview -> live
    this.pxPerSec = 60;
    this.t = 0;
    this.sel = null;                 // selected keyframe (v1 semantics)
    this.selBlock = null;            // selected block id
    this.preview = true;
    this.running = false;
    this.autofix = true;
    this.manipOpen = false;          // expandable manipulation lane
    this.registry = { ...MOTION_CLIPS };
    this._modCache = new Map();      // clip key -> Set(module label)
    this.seq = this._load();
    this.seq.onChange = () => { this._save(); this._validate(); this.render(); };
    this.track = this._loadChoreo();
    this.track.onChange = () => { this._saveChoreo(); this._validate();
      this._navPath = null; this.render(); };
    this.annos = this._loadAnnos();
    this.annos.onChange = () => { this._saveAnnos(); this.render(); };
    this._navPath = null;               // integrated preview base path cache
    this.previewingRun = false;         // ▶ Play = 3D preview (no wire)
    this.gaps = [];
    this.poseJumps = [];
    this._goal = null;               // active goal block {id, timer}
    this._wire();
    this._fetchActions();
    this._validate();
    this.render();
  }

  // ── persistence ────────────────────────────────────────────────────────────
  _load(){
    try { return SetpointSeq.fromJSON(JSON.parse(localStorage.getItem(LS_KEY) || 'null')); }
    catch { return new SetpointSeq(); }
  }
  _save(){ localStorage.setItem(LS_KEY, JSON.stringify(this.seq.toJSON())); }
  _loadChoreo(){
    try {
      return ChoreoTrack.fromJSON(
        JSON.parse(localStorage.getItem(LS_CHOREO) || 'null'), this.registry);
    } catch { return new ChoreoTrack(this.registry); }
  }
  _saveChoreo(){ localStorage.setItem(LS_CHOREO, JSON.stringify(this.track.toJSON())); }
  _loadAnnos(){
    try { return AnnoTrack.fromJSON(JSON.parse(localStorage.getItem(LS_ANNOS) || 'null')); }
    catch { return new AnnoTrack(); }
  }
  _saveAnnos(){ localStorage.setItem(LS_ANNOS, JSON.stringify(this.annos.toJSON())); }

  // backend-declared actions (coming-soon policy slots etc.)
  async _fetchActions(){
    try {
      const r = await fetch('/api/actions');
      if (!r.ok) return;
      const j = await r.json();
      this.registry = mergeActions(this.registry, j.actions || []);
      this.track.registry = this.registry;
      this._buildMenu();
      this.render();
    } catch { /* offline backend: local registry only */ }
  }

  _x = (t) => t * this.pxPerSec + 10;
  _t = (x) => Math.max(0, (x - 10) / this.pxPerSec);

  // ── wiring ─────────────────────────────────────────────────────────────────
  _wire(){
    this.toolbar.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      const tool = b.dataset.tool;
      if (tool && tool.startsWith('add-')) this._toggleMenu(tool.slice(4), b);
      if (tool === 'del') this._deleteSelected();
      if (tool === 'blade') this._blade();
      if (tool === 'dup' && this.selBlock) this.track.duplicate(this.selBlock);
      if (tool === 'note') this._addNote();
      if (tool === 'skill') bus.emit('composeSkill');
      if (tool === 'play') this.physicsPreview();
      if (tool === 'execute') this.run();
      if (tool === 'stop') this.stop();
      if (tool === 'preview'){ this.preview = !this.preview; b.classList.toggle('on', this.preview);
        if (!this.preview) this.onPreview(null); }
      if (tool === 'autofix'){ this.autofix = !this.autofix; b.classList.toggle('on', this.autofix);
        this._validate(); this.render(); }
      if (tool === 'zoomin') this.zoom(1.4);
      if (tool === 'zoomout') this.zoom(1 / 1.4);
      if (tool === 'zoomfit') this.zoomFit();
    });
    // scrub anywhere on the ruler
    const scrub = (e) => {
      const r = this.scroll.getBoundingClientRect();
      this.setTime(this._t(e.clientX - r.left + this.scroll.scrollLeft));
    };
    const dragScrub = (e) => {
      scrub(e);
      const move = (ev) => scrub(ev);
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    };
    this.ruler.addEventListener('pointerdown', dragScrub);
    this.playhead.addEventListener('pointerdown', (e) => { e.stopPropagation(); dragScrub(e); });
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'k' || e.key === 'K'){ const p = this.getPose(); p && this.seq.add(this.t, p); }
      if (e.key === 'b' || e.key === 'B') this._blade();
      if (e.key === 'Backspace' || e.key === 'Delete') this._deleteSelected();
      if (e.key === ' ' && !this.ctl.teleop){   // Space = preview play / stop
        e.preventDefault();
        (this.running || this.previewingRun) ? this.stop() : this.previewRun();
      }
    });
    window.addEventListener('studio-resize', () => this.render());
    document.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('.action-menu') && !e.target.closest('.tl-add'))
        this._menuEl()?.classList.add('hidden');
    });
  }

  _menuEl(){ return this.toolbar.querySelector('.action-menu'); }

  _deleteSelected(){
    if (this.selNote){ this.annos.remove(this.selNote); this.selNote = null; }
    else if (this.selBlock){ this.track.remove(this.selBlock); this.selBlock = null; }
    else if (this.sel){ this.seq.remove(this.sel); this.sel = null; }
  }
  _addNote(){
    const text = prompt('annotation text (language instruction)', '');
    if (!text) return;
    const n = this.annos.add(this.t, 2.0, text);
    this.selNote = n.id;
  }
  _blade(){
    if (!this.selBlock) { bus.emit('toast', { msg: 'select a block to blade (B)' }); return; }
    const cut = this.track.cut(this.selBlock, this.t);
    if (cut) bus.emit('toast', { msg: 'block cut at playhead' });
  }

  // ── "+ Nav / + Policy / + Animation / + Pose" family menus ────────────────
  _toggleMenu(category, anchor){
    const m = this._menuEl();
    if (m && this._menuCat === category && !m.classList.contains('hidden')){
      m.classList.add('hidden');
      return;
    }
    this._buildMenu(category);
    const menu = this._menuEl();
    menu.classList.remove('hidden');
    const r = (anchor || this.toolbar.querySelector('.tl-add')).getBoundingClientRect();
    menu.style.left = Math.max(8, Math.min(r.left, innerWidth - 296)) + 'px';
    menu.style.bottom = (innerHeight - r.top + 8) + 'px';
  }
  _buildMenu(category = 'anim'){
    let m = this._menuEl();
    if (m) m.remove();
    this._menuCat = category;
    m = document.createElement('div');
    m.className = 'action-menu hidden';
    const SECTIONS = {
      nav: [['NAVIGATION — base motion', (c) => c.category === 'nav']],
      policy: [['TASK POLICIES', (c) => c.category === 'policy']],
      anim: [['ANIMATIONS · DANCES', (c) => c.category === 'animation']],
      pose: [['STATIC POSES · KEYFRAMES', (c) => c.category === 'pose']],
    };
    // the +Pose menu leads with the two live-capture tools (formerly toolbar)
    if (category === 'pose'){
      const cap = (label, title, fn) => {
        const row = document.createElement('div');
        row.className = 'am-item';
        row.innerHTML = `<span class="am-emoji">◆</span><span class="am-title">${label}</span>`;
        row.title = title;
        row.onclick = () => { fn(); this._menuEl()?.classList.add('hidden'); };
        m.appendChild(row);
      };
      const h = document.createElement('div');
      h.className = 'am-head';
      h.textContent = 'CAPTURE FROM THE ROBOT';
      m.appendChild(h);
      cap('Keyframe current pose (K)', 'insert the robot\'s live pose as a keyframe at the playhead',
        () => { const p = this.getPose(); p && this.seq.add(this.t, p); });
      cap('Keyframe target pose', 'insert the commanded TARGET pose as a keyframe',
        () => { const p = this.getTarget() || this.getPose(); p && this.seq.add(this.t, p); });
    }
    for (const [label, match] of SECTIONS[category] || SECTIONS.anim){
      const h = document.createElement('div');
      h.className = 'am-head';
      h.textContent = label;
      m.appendChild(h);
      for (const [key, clip] of Object.entries(this.registry)){
        if (!match(clip)) continue;
        const row = document.createElement('div');
        const soon = clip.status !== 'available';
        row.className = 'am-item' + (soon ? ' soon' : '');
        const dur = clipDur(clip, defaults(clip));
        row.innerHTML = `<span class="am-emoji">${clip.emoji || '🤖'}</span>` +
          `<span class="am-title">${clip.title}</span>` +
          (soon ? `<span class="am-soon">coming soon</span>`
                : `<span class="am-dur mono">${dur.toFixed(1)}s</span>`);
        if (clip.note || clip.location?.label)
          row.title = [clip.note, clip.location?.label && `@ ${clip.location.label}`]
            .filter(Boolean).join(' · ');
        if (!soon){
          if (clip.category === 'pose'){
            // poses ARE keyframes too: offer both insertion modes
            const kf = document.createElement('button');
            kf.className = 'am-mini'; kf.textContent = '◆ key';
            kf.title = 'insert the pose as an interpolated keyframe at the playhead';
            kf.onclick = (e) => { e.stopPropagation(); this._addPoseKeyframe(key); };
            const bl = document.createElement('button');
            bl.className = 'am-mini'; bl.textContent = '▭ block';
            bl.title = 'insert as a transition-and-hold block';
            bl.onclick = (e) => { e.stopPropagation(); this._addAction(key); };
            row.append(kf, bl);
          } else {
            row.onclick = () => this._addAction(key);
          }
        }
        m.appendChild(row);
      }
    }
    this.toolbar.querySelector('.action-wrap')?.appendChild(m);
  }
  _addPoseKeyframe(key){
    const clip = this.registry[key];
    const target = clip.poseTarget ? clip.poseTarget()
      : clipJointsAt(clip, clipDur(clip, defaults(clip)));
    if (target) this.seq.add(this.t, target, clip.title);
    this._menuEl()?.classList.add('hidden');
    bus.emit('toast', { msg: `keyframe: ${clip.title}` });
  }
  _addAction(key){
    const clip = this.registry[key];
    if (!clip || clip.status !== 'available') return;
    const params = defaults(clip);
    for (const [p, spec] of Object.entries(clip.params || {})){
      const v = prompt(`${clip.title} — ${spec.label}`, String(spec.def));
      if (v == null) return;
      params[p] = Math.max(spec.min ?? -1e9, Math.min(spec.max ?? 1e9, parseFloat(v) || spec.def));
    }
    this.track.add(key, this.t, params);
    this._menuEl()?.classList.add('hidden');
    bus.emit('toast', { msg: `added: ${clip.title}` });
  }

  // ── location gap validation ────────────────────────────────────────────────
  _startLoc(){
    const b = this.link.state?.base;
    return b ? { x: b.x, y: b.y, label: 'here' } : { x: 0, y: 0, label: 'home' };
  }
  _validate(){
    if (this.autofix){
      const fixed = autoFixGaps(this.track, this._startLoc());
      if (fixed.length){
        bus.emit('toast', { msg: `auto-fix: inserted ${fixed.length} move-to-goal` });
        this._saveChoreo();
      }
      // SAFETY: bridge every pose discontinuity between manipulation blocks
      // with an eased blend sized by the pose distance (min duration enforced).
      // The 20 Hz executor HOLDS the last commanded pose across a gap, so at the
      // gap exit it would command the whole delta in one 50 ms tick — measured
      // 29 rad/s against a 2.5 rad/s budget. See control_center/README.md.
      const pj = autoFixPoseJumps(this.track, (t) => this.seq.poseAt(t));
      if (pj.inserted.length)
        bus.emit('toast', { msg: `safety: ${pj.inserted.length} pose-blend transition${pj.inserted.length > 1 ? 's' : ''} inserted` });
      // A bridge the gap could not make gentle is still inside the hard cap, but
      // the operator is the one who can give it more room — say so once.
      if (pj.tight?.length)
        bus.emit('toast', { msg: `safety: ${pj.tight.length} transition${pj.tight.length > 1 ? 's are' : ' is'} brisk — widen the gap or lengthen the blend to soften` });
      if (pj.inserted.length || pj.updated.length || pj.removed.length)
        this._saveChoreo();
      this.poseJumps = [];
    } else {
      // autofix off → surface unbridged jumps as ⚠ markers instead.
      // Keyed by blockId:edge, not blockId: a block has TWO edges and both can
      // jump, so a hand-placed bridge on one of them must not hide the other.
      this.poseJumps = detectPoseJumps(this.track, (t) => this.seq.poseAt(t))
        .filter((j) => !this.track.blocks.some((b) =>
          this.track.clip(b)?.kind === 'xfade'
          && (b.params?.forKey || b.params?.forId) === j.id));
    }
    this.gaps = detectGaps(this.track, this._startLoc());
  }

  setTime(t){
    this.t = t;
    this.playhead.style.left = this._x(t) - this.scroll.scrollLeft + 'px';
    if (this.preview && !this.running)
      this.onPreview(this._poseAt(t), this._baseAt(t));
    this._drawRuler();
  }

  // integrated nav-lane base path for the 3D preview (cached per edit)
  _baseAt(t){
    if (!this.track.blocks.some((b) => this.track.clip(b)?.nav)) return null;
    if (!this._navPath)
      this._navPath = integrateNavPath(this.track, Math.max(this.duration(), t) + 1);
    const i = Math.max(0, Math.min(this._navPath.length - 1, Math.round(t / 0.05)));
    return this._navPath[i];
  }

  // ▶ Play: run the choreography in the 3D viewer only — nothing on the wire.
  previewRun(){
    if (this.running || this.previewingRun) return;
    this.previewingRun = true;
    const t0 = performance.now() / 1000 - this.t;
    this._previewTimer = setInterval(() => {
      const t = performance.now() / 1000 - t0;
      this.setTime(t);
      if (t >= this.duration()){
        clearInterval(this._previewTimer);
        this.previewingRun = false;
      }
    }, 50);
  }
  // choreography pose (keyframes under blocks; block joints win per-joint)
  _poseAt(t){
    const kf = this.seq.poseAt(t);
    const bj = this.track.jointsAt(t);
    if (!kf && !bj) return null;
    return { ...(kf || {}), ...(bj || {}) };
  }

  zoom(mul){ this.pxPerSec = Math.max(8, Math.min(400, this.pxPerSec * mul)); this.render(); }
  zoomFit(){
    const dur = Math.max(2, this.duration());
    this.pxPerSec = Math.max(8, (this.scroll.clientWidth - 40) / dur);
    this.render();
  }
  duration(){ return Math.max(this.seq.duration(), this.track.duration(), this.annos.duration()); }

  // ── run: EXEC_RATE choreography executor (one loop, two sinks) ───────────
  // live sink = the robot wire; preview sink = the isolated physics world.
  run(){
    if (this.ctl.estop || this.ctl.feel === 'soft'){
      bus.emit('toast', { msg: this.ctl.estop ? 'e-stop engaged' : 'soft mode: robot not commandable', err: true });
      return;
    }
    this.onExecuteView?.();             // 3D viewer: fade preview -> live robot
    bus.emit('toast', { msg: 'executing choreography on the robot' });
    this._startRun(this.link, { live: true });
  }
  _startRun(sink, opts = {}){
    if (this.running || (!this.seq.keys.length && !this.track.blocks.length)) return;
    this.running = true;
    this._runLive = !!opts.live;
    // Tell the stick pump that a choreography owns the base. Both live on the same socket, so the
    // gateway cannot tell their frames apart: a centred stick streaming nav = 0 at 30 Hz
    // interleaved with this loop's nav made the base alternate between moving and stopping many
    // times a second, which reads as the base simply not moving. teleopctl still sends a
    // DEFLECTED stick, so the operator can always take over.
    if (this._runLive) this.ctl.choreo = true;
    this._sink = sink;
    if (this._previewTimer){ clearInterval(this._previewTimer); this.previewingRun = false; }
    this.onPreview(null);
    const t0 = performance.now() / 1000 - this.t;
    const hasNav = this.track.blocks.some((b) => b.lane === 'nav') ||
                   this.track.blocks.some((b) => this.track.clip(b)?.nav);
    let lastPose = '';
    // LIFT RATE LIMIT. The clips were authored against a column far faster than the real
    // one: measured peaks run to 1520 mm/s (`happy`) against a column that does 36.4-37.0
    // mm/s steady (motions.js LIFT_MAX_VEL), so 17 of them simply command a speed the hardware cannot reach. motions.js
    // has carried liftRateLimit() and the measured LIFT_MAX_VEL for exactly this and
    // nothing ever called it. Clamping HERE — the one place every clip frame, blends
    // included, is streamed — keeps each animation's shape as close to its authored form
    // as physics allows, and covers every clip written from now on for free.
    let liftPrev = null, liftT = null;
    const navMoveCap = this._runLive && (this.link.hello?.caps || []).includes('nav_move');
    const runId = Date.now().toString(36);
    this._navMove = null;
    this._timer = setInterval(() => {
      const t = performance.now() / 1000 - t0;
      if (this._runLive && (this.ctl.estop || this.ctl.feel === 'soft')){ this.stop(); return; }
      this.setTime(t);
      const pose = this._poseAt(t);
      if (pose && (pose.lift_lower != null || pose.lift_upper != null)){
        // both stages ride one height: the column's extension is their sum
        const col = (pose.lift_lower || 0) + (pose.lift_upper || 0);
        const dt = liftT == null ? 0 : t - liftT;
        const lim = liftPrev == null ? col : liftRateLimit(liftPrev, col, dt);
        if (pose.lift_lower != null) pose.lift_lower = lim / 2;
        if (pose.lift_upper != null) pose.lift_upper = lim / 2;
        liftPrev = lim; liftT = t;
      }
      // skip identical re-sends (held poses): a constant joint_command stream
      // steals base-drive authority from concurrent nav blocks (measured)
      if (pose){
        const enc = JSON.stringify(pose);
        if (enc !== lastPose){ sink.joints(pose); lastPose = enc; }
      }
      let navBlock = null;
      if (navMoveCap){
        const b = this.track.activeAt(t, 'nav');
        const move = b && navMoveFor(b.key, b.params, this.track.registry);
        if (move){
          navBlock = b;
          if (this._navMove?.block !== b.id){
            const id = `${b.id}-${runId}`;
            this._navMove = { block: b.id, id };
            this.link.send('nav_move', { id, ...move });
            const s = this.link.state || {}, caps = s.speedCaps || this.link.hello || {};
            const need = navMoveSeconds(move, this.link.hello?.navCalibration, { vmax: caps.vmax, wmax: caps.wmax });
            const have = this.track.end(b) - t;
            if (need != null && need > have + 0.2)
              bus.emit('toast', { msg: `nav: this move needs ~${need.toFixed(1)} s, the block is ${have.toFixed(1)} s — the next nav block will cut it short`, err: true });
          }
        }
      }
      if (hasNav) sink.frame(navBlock ? { ...ZNAV } : (this.track.navAt(t) || { ...ZNAV }));
      if (this._runLive) this._serviceGoal(t);   // REST goals drive the LIVE
                                                 // navigator — never in preview
      opts.onTick?.(t);
      if (t >= this.duration()){
        this.stop({ keepNav: true });          // a nav_move still finishing completes on the robot
        opts.onDone?.();
        if (this._runLive) bus.emit('toast', { msg: 'choreography complete' });
      }
      // EXEC_RATE, not a literal: this is the rate the robot is actually driven at, and every
      // safety number in motions.js derives from the same constant. It used to be 50 ms (20 Hz)
      // feeding a chain that runs at 50 Hz end to end, so each pose was held for two or three
      // ticks while the joint slewed toward it and stopped — the step-like jitter in playback.
    }, 1000 / EXEC_RATE);
  }
  stop({ keepNav = false } = {}){
    if (this._previewTimer){ clearInterval(this._previewTimer); this.previewingRun = false; }
    if (this._navMove && !keepNav) this.link.send('nav_move', { id: this._navMove.id, cancel: true });
    this._navMove = null;
    if (!this.running && !this._timer){ this._pvIdle(); return; }
    this.running = false;
    this.ctl.choreo = false;                 // the base is the operator's again
    clearInterval(this._timer); this._timer = null;
    const sink = this._sink || this.link;
    if (this._runLive) this._endGoal();
    // STOP = stream zeros briefly; never leave a non-zero last frame
    let n = 0;
    const z = setInterval(() => {
      sink.frame({ ...ZNAV });
      if (++n >= 6){
        clearInterval(z);
        this._pvIdle();
      }
    }, 40);
  }

  // ── TRUE physics preview: run the plan in an ISOLATED MuJoCo world ────────
  // POST /api/preview/start spawns (or reuses) an ephemeral simulation_bridge
  // instance seeded from the live robot; /ws/preview accepts the same wire
  // envelopes; the PHYSICS PREVIEW window streams its render. The live world
  // (sim or real) is untouched by design.
  async physicsPreview(){
    if (this.running && !this._runLive){ this.stop(); return; }   // toggle off
    if (this.running){ bus.emit('toast', { msg: 'already executing on the robot', err: true }); return; }
    if (!this.track.blocks.length && !this.seq.keys.length){
      bus.emit('toast', { msg: 'timeline is empty — nothing to preview', err: true });
      return;
    }
    this._pvChip('spinning up…');
    bus.emit('toast', { msg: 'preview world spinning up — seeded from the current pose' });
    bus.emit('physicsPreviewWindow');            // app: open/reuse the viewer window
    let ok = false;
    try {
      const r = await fetch('/api/preview/start', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: true }) }).then((x) => x.json());
      ok = !!(r.ok || r.running);
    } catch { /* fall through */ }
    if (!ok){
      this._pvChip('preview backend unavailable');
      bus.emit('toast', { msg: 'physics preview unavailable', err: true });
      return;
    }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/preview`);
    const send = (type, payload = {}) => {
      if (ws.readyState === 1) ws.send(JSON.stringify({ type, payload }));
    };
    let seq = 0;
    const sink = {
      joints: (map) => send('joint_command', { joints: expandGrips(map) }),
      frame: (nav) => send('teleop_frame', { sequence: seq++,
        timestamp: Date.now() / 1000, mode: 'Base Driving', navJoystick: nav }),
      send,
    };
    this._pvWs = ws;
    ws.onerror = () => { this._pvChip('preview link failed'); };
    ws.onopen = () => {
      // start the preview run from the LIVE robot's pose (re-seed on reuse)
      const b = this.link.state?.base;
      if (b) send('reset', { pose: { x: b.x, y: b.y, yaw: b.yaw } });
      setTimeout(() => {
        this.setTime(0);
        this._startRun(sink, {
          live: false,
          onTick: (t) => {
            const { n, m } = blockProgress(this.track, t);
            this._pvChip(`running block ${n}/${m} · ${t.toFixed(1)}s`);
          },
          onDone: () => {
            this._pvChip('done — settled result in the PHYSICS PREVIEW window');
            // per contract: zeros (stop()) then a clean reset for the next run,
            // after a beat so the settled outcome stays visible
            setTimeout(() => send('reset', {}), 4000);
          },
        });
      }, 600);
    };
  }
  stopPhysicsPreview(){
    if (this.running && !this._runLive) this.stop();
    this._pvChip(null);
  }
  _pvIdle(){
    if (this._pvWs && this._pvWs.readyState === 1 && !this.running){
      /* leave the socket open; the preview world idles out server-side */
    }
  }
  _pvChip(text){
    const el = document.getElementById('tlPvChip');
    if (!el) return;
    el.classList.toggle('hidden', !text);
    if (text) el.textContent = `physics preview: ${text}`;
  }

  // move-to-goal blocks: on the real robot (cfg.ros) they ride the bridge
  // wire (nav_goal → ROS Nav2); otherwise the backend's /api/nav REST plane
  // (graceful toast when neither is available)
  _serviceGoal(t){
    const g = this.track.goalAt(t);
    if (g && this._goal?.id !== g.id){
      this._endGoal();
      this._goal = { id: g.id, poll: 0 };
      const x = +(g.params?.x ?? 0), y = +(g.params?.y ?? 0);
      if (conf().ros){
        const ok = this.link.navGoal ? (this.link.navGoal(x, y, 0), true)
          : this.link.send('nav_goal', { x, y, yaw: 0 });
        bus.emit('toast', ok === false
          ? { msg: 'nav goal: bridge offline', err: true }
          : { msg: `nav goal → (${x}, ${y}) · Nav2` });
        return;
      }
      fetch('/api/nav/goal', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y }) })
        .then((r) => { if (!r.ok) throw 0;
          bus.emit('toast', { msg: `nav goal → (${x}, ${y})` }); })
        .catch(() => bus.emit('toast', { msg: 'nav goal API not available', err: true }));
    } else if (!g && this._goal){
      this._endGoal();
    } else if (g && this._goal && !conf().ros
               && (this._goal.poll = (this._goal.poll + 1) % 10) === 0){
      fetch('/api/nav/status').then((r) => r.ok ? r.json() : null).then((s) => {
        if (s && (s.state === 'reached' || s.state === 'failed'))
          bus.emit('toast', { msg: `nav: ${s.state}`, err: s.state === 'failed' });
      }).catch(() => {});
    }
  }
  _endGoal(){
    if (!this._goal) return;
    this._goal = null;
    if (conf().ros){ this.link.send('nav_cancel', {}); return; }
    fetch('/api/nav/cancel', { method: 'POST' }).catch(() => {});
  }

  // ── module involvement (expandable manipulation lane) ─────────────────────
  _modulesFor(key, params){
    if (this._modCache.has(key)) return this._modCache.get(key);
    const clip = this.registry[key];
    const out = new Set();
    if (clip?.joints){
      const d = clipDur(clip, params || defaults(clip));
      for (let i = 0; i <= 8; i++){
        const j = clipJointsAt(clip, (d * i) / 8, params);
        for (const [id, v] of Object.entries(j || {})){
          if (Math.abs(v) < 0.02) continue;
          for (const [label, match] of MODULES) if (match(id)) out.add(label);
        }
      }
    }
    this._modCache.set(key, out);
    return out;
  }

  // ── rendering ──────────────────────────────────────────────────────────────
  render(){
    this.tracks.innerHTML = '';
    const width = this._x(Math.max(this.duration() + 4, 20)) + 'px';
    this.tracks.appendChild(this._langLane(width));
    for (const lane of LANES){
      const el = document.createElement('div');
      el.className = `tl-lane lane-${lane.id}`;
      el.style.width = width;
      const tag = document.createElement('span');
      tag.className = 'lane-tag mono';
      if (lane.id === 'manip'){
        const tog = document.createElement('button');
        tog.className = 'lane-toggle';
        tog.textContent = this.manipOpen ? '▾' : '▸';
        tog.title = 'expand per-module sub-tracks';
        tog.onclick = (e) => { e.stopPropagation(); this.manipOpen = !this.manipOpen; this.render(); };
        tag.appendChild(tog);
      }
      tag.appendChild(document.createTextNode(lane.label));
      el.appendChild(tag);
      // blocks
      for (const b of this.track.lane(lane.id)) el.appendChild(this._blockEl(b));
      // keyframes live on the manipulation lane (v1 behaviour preserved)
      if (lane.id === 'manip'){ this._renderKeys(el); this._renderPoseJumps(el); }
      // gap markers live on the navigation lane
      if (lane.id === 'nav') this._renderGaps(el);
      el.addEventListener('pointerdown', (e) => {
        if (e.target === el){
          const r = this.scroll.getBoundingClientRect();
          this.setTime(this._t(e.clientX - r.left + this.scroll.scrollLeft));
          this.sel = null; this.selBlock = null; this.render();
        }
      });
      this.tracks.appendChild(el);
      if (lane.id === 'manip' && this.manipOpen) this._renderModuleTracks(width);
    }
    this._drawRuler();
    this.setTime(this.t);
  }

  // ── LANGUAGE lane: free-text annotations + auto labels from action blocks ──
  _langLane(width){
    const el = document.createElement('div');
    el.className = 'tl-lane lane-lang';
    el.style.width = width;
    const tag = document.createElement('span');
    tag.className = 'lane-tag mono';
    tag.textContent = 'LANGUAGE';
    el.appendChild(tag);
    // derived: every action block carries its language text automatically
    for (const b of this.track.blocks){
      if (b.lane === 'nav' && b.linkId) continue;        // one label per linked pair
      const clip = this.track.clip(b);
      if (!clip) continue;
      const d = document.createElement('div');
      d.className = 'lang-derived mono';
      d.style.left = this._x(b.t0) + 'px';
      d.style.width = Math.max(24, (b.out - b.in) * this.pxPerSec) + 'px';
      d.textContent = this._langText(b, clip);
      d.title = 'auto label (moves with the block)';
      el.appendChild(d);
    }
    // user annotations: draggable, dbl-click to edit, Delete removes
    for (const n of this.annos.notes){
      const d = document.createElement('div');
      d.className = 'lang-note' + (n.id === this.selNote ? ' sel' : '');
      d.style.left = this._x(n.t0) + 'px';
      d.style.width = Math.max(24, n.dur * this.pxPerSec) + 'px';
      d.textContent = n.text;
      d.title = `${n.text} · ${n.t0.toFixed(1)}s (drag to move, double-click to edit)`;
      d.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        this.selNote = n.id; this.selBlock = null; this.sel = null;
        const r = this.scroll.getBoundingClientRect();
        const grab0 = this._t(e.clientX - r.left + this.scroll.scrollLeft) - n.t0;
        const move = (ev) => this.annos.move(n.id,
          this.track.snapTime(this._t(ev.clientX - r.left + this.scroll.scrollLeft) - grab0,
            { playhead: this.t }));
        const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
        window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
        this.render();
      });
      d.addEventListener('dblclick', (e) => {
        e.stopPropagation();
        const t = prompt('annotation text', n.text);
        if (t != null) this.annos.setText(n.id, t);
      });
      el.appendChild(d);
    }
    return el;
  }
  _langText(b, clip){
    const p = b.params || {};
    if (p.angle != null) return `${clip.title.toLowerCase()} ${p.angle}\u00b0`;
    if (p.dist != null) return `${clip.title.toLowerCase()} ${p.dist} m`;
    if (clip.kind === 'goal')
      return `move to ${p.label || `(${p.x}, ${p.y})`}`;
    return clip.title.toLowerCase();
  }

  _renderKeys(laneEl){
    for (const k of this.seq.keys){
      const el = document.createElement('div');
      el.className = 'sp-key' + (k === this.sel ? ' sel' : '');
      el.style.left = this._x(k.t) - 7 + 'px';
      el.title = `${k.t.toFixed(2)}s · ${Object.keys(k.pose).length} joints` +
        (k.label ? ` · ${k.label}` : '');
      el.textContent = '◆';
      el.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        this.sel = k; this.selBlock = null; this.render();
        const r = this.scroll.getBoundingClientRect();
        const move = (ev) => this.seq.move(k,
          this.track.snapTime(this._t(ev.clientX - r.left + this.scroll.scrollLeft),
            { playhead: this.t }));
        const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
        window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
      });
      laneEl.appendChild(el);
    }
  }

  // unbridged pose-jump ⚠ markers (autofix off) — click inserts the blend
  _renderPoseJumps(laneEl){
    for (const j of this.poseJumps || []){
      const el = document.createElement('button');
      el.className = 'gap-marker pose';
      el.style.left = this._x(j.t) - 9 + 'px';
      el.textContent = '⚡';
      el.title = `pose jump: ${j.delta.toFixed(2)} rad on ${j.joint} — the robot ` +
        `would snap here. Click to insert a ${j.minDur.toFixed(1)}s safety blend.`;
      el.onclick = (e) => {
        e.stopPropagation();
        this.track.blocks.push(makeBlendBlock(j));
        this.track._emit();
        bus.emit('toast', { msg: `safety blend inserted (${j.minDur.toFixed(1)}s)` });
      };
      laneEl.appendChild(el);
    }
  }

  _blockEl(b){
    const clip = this.track.clip(b);
    const el = document.createElement('div');
    // `xfade` is what makes a safety blend LOOK like one (control.css styles
    // .clip-block.xfade purple). Without it a blend rendered as an ordinary
    // clip — the operator could not tell a guard the studio inserted from a
    // clip they authored, on a block they are not allowed to blade.
    el.className = 'clip-block' + (b.id === this.selBlock ? ' sel' : '') +
      (clip?.kind === 'goal' ? ' goal' : '') +
      (clip?.kind === 'xfade' ? ' xfade' : '') + (b.lane === 'nav' ? ' nav' : '');
    el.style.left = this._x(b.t0) + 'px';
    el.style.width = Math.max(18, (b.out - b.in) * this.pxPerSec) + 'px';
    const sub = (b.in > 0.001 || (clip && !clip.hold && b.out < clipDur(clip, b.params) - 0.001))
      ? ` [${b.in.toFixed(1)}–${b.out.toFixed(1)}]` : '';
    el.innerHTML =
      `<span class="cb-emoji">${clip?.emoji || '🤖'}</span>` +
      `<span class="cb-title">${clip?.title || b.key}</span>` +
      `<span class="cb-dur mono">${(b.out - b.in).toFixed(1)}s${sub}</span>` +
      `<i class="cb-trim l"></i><i class="cb-trim r"></i>`;
    el.title = `${clip?.title || b.key} · ${b.t0.toFixed(2)}s → ${this.track.end(b).toFixed(2)}s` +
      (b.linkId ? ' · linked' : '') +
      (clip?.location?.label ? ` · @${clip.location.label}` : '');

    el.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      this.selBlock = b.id; this.sel = null;
      const r = this.scroll.getBoundingClientRect();
      const px2t = (ev) => this._t(ev.clientX - r.left + this.scroll.scrollLeft);
      const edge = e.target.classList.contains('cb-trim')
        ? (e.target.classList.contains('l') ? 'l' : 'r') : null;
      const grab0 = px2t(e) - b.t0;
      let dragged = false;
      let target = b;
      if (!edge && e.altKey){                       // alt-drag duplicates
        target = this.track.duplicate(b.id, b.t0);
        this.selBlock = target.id;
      }
      const move = (ev) => {
        dragged = true;
        const t = px2t(ev);
        if (edge){
          this.track.trim(b.id, edge,
            this.track.snapTime(t, { excludeId: b.id, playhead: this.t }));
        } else {
          this.track.move(target.id,
            this.track.snapTime(t - grab0, { excludeId: target.id, playhead: this.t }));
        }
      };
      const up = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        if (!dragged) this.render();               // simple select
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
      this.render();
    });
    return el;
  }

  _renderGaps(laneEl){
    for (const g of this.gaps){
      const el = document.createElement('button');
      el.className = 'gap-marker';
      el.style.left = this._x(g.t) - 9 + 'px';
      el.textContent = '⚠';
      const from = g.from?.label || 'start';
      el.title = `location gap: ${from} → ${g.to.label}. Click to insert the missing move.`;
      el.onclick = (e) => {
        e.stopPropagation();
        const label = g.to.label || `${g.to.x}, ${g.to.y}`;
        if (g.to.x == null){ bus.emit('toast', { msg: `"${label}" has no map coordinates yet`, err: true }); return; }
        const ok = confirm(`insert: move to ${label} (${g.to.x}, ${g.to.y})?`);
        if (!ok) return;
        const made = this.track.add('nav_goto', Math.max(0, g.t - 6),
          { x: g.to.x, y: g.to.y, label: g.to.label });
        if (made) made[0].out = made[0].in + 6;
      };
      laneEl.appendChild(el);
    }
  }

  _renderModuleTracks(width){
    for (const [label] of MODULES){
      const row = document.createElement('div');
      row.className = 'tl-lane tl-module';
      row.style.width = width;
      const tag = document.createElement('span');
      tag.className = 'lane-tag mono sub';
      tag.textContent = label;
      row.appendChild(tag);
      for (const b of this.track.blocks){
        const clip = this.track.clip(b);
        if (!clip?.joints) continue;
        if (clip.lane === 'both' && b.lane === 'nav') continue;
        if (!this._modulesFor(b.key, b.params).has(label)) continue;
        const seg = document.createElement('div');
        seg.className = 'mod-seg';
        seg.style.left = this._x(b.t0) + 'px';
        seg.style.width = Math.max(8, (b.out - b.in) * this.pxPerSec) + 'px';
        seg.title = `${clip.title} drives ${label}`;
        row.appendChild(seg);
      }
      // keyframes drive every module they carry joints for
      for (const k of this.seq.keys){
        const touches = MODULES.find(([l, m]) => l === label &&
          Object.keys(k.pose).some((j) => Math.abs(k.pose[j]) > 0.02 && m(j)));
        if (!touches) continue;
        const dot = document.createElement('div');
        dot.className = 'mod-key';
        dot.style.left = this._x(k.t) - 2 + 'px';
        row.appendChild(dot);
      }
      this.tracks.appendChild(row);
    }
  }

  _drawRuler(){
    const cv = this.ruler, w = this.scroll.clientWidth, h = 22;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, w, h);
    x.font = '9px Geist Mono, monospace';
    x.fillStyle = '#6b655f'; x.strokeStyle = 'rgba(242,238,230,.14)';
    const step = this.pxPerSec >= 50 ? 1 : this.pxPerSec >= 20 ? 2 : 5;
    const t0 = Math.floor(this._t(this.scroll.scrollLeft) / step) * step;
    for (let t = t0; this._x(t) - this.scroll.scrollLeft < w; t += step){
      const px = this._x(t) - this.scroll.scrollLeft;
      x.beginPath(); x.moveTo(px, h - 7); x.lineTo(px, h); x.stroke();
      x.fillText(t.toFixed(0) + 's', px + 3, h - 9);
    }
  }
}
