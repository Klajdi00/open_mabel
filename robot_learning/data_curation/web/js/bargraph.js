// Dense 59-joint "equalizer" (one bar per actuator, grouped). Click a bar to
// drill into a value-over-time STRIP that compares multiple joints: add more
// joints, quick-add whole groups (L arm, L hand, base, …), and toggle whether
// measured / target / both are drawn. Synced to the playhead.
import { GROUP_ORDER, GROUP_LABEL } from './names.js';
import { jointStatusAt } from './defects.js';

const C = { accent: '#C25B2A', soft: '#f0b68b', ash: '#6b655f', mid: 'rgba(242,238,230,.15)' };
// per-joint health: 0 ok (green), 1 warn/low-rate (yellow), 2 missing (red)
const STATUS_COL = ['#3ecf8e', '#E7C24B', '#D9463A'];
const QUICK = [['L arm', 'left_arm'], ['R arm', 'right_arm'], ['L hand', 'left_hand'], ['R hand', 'right_hand'],
               ['Base', 'base'], ['Neck', 'neck'], ['Lift/Torso', 'lift_torso']];
const color = (j) => `hsl(${(j * 47) % 360},72%,63%)`;

export class BarGraph {
  constructor(barCanvas, stripCanvas, onScrub){
    this.bc = barCanvas; this.sc = stripCanvas; this.onScrub = onScrub;
    this.backBtn = document.getElementById('barBack');
    this.stripCtl = document.getElementById('stripCtl');
    this.legendEl = document.querySelector('.plots-legend');
    this.hintEl = document.getElementById('plotsHint');
    this.mode = 'pos'; this.entry = null; this.frame = 0;
    this.strip = false; this.stripJoints = new Set(); this.show = 'both';
    this._dpr();
    window.addEventListener('studio-resize', () => { this._dpr(); this.draw(); });
    new ResizeObserver(() => { this._dpr(); this.draw(); }).observe(barCanvas.parentElement);
    barCanvas.addEventListener('click', (e) => this._pickBar(e));
    stripCanvas.addEventListener('pointerdown', (e) => this._stripScrub(e));
  }

  _dpr(){
    for (const cv of [this.bc, this.sc]){
      const r = cv.getBoundingClientRect();
      cv.width = Math.max(1, r.width * devicePixelRatio); cv.height = Math.max(1, r.height * devicePixelRatio);
    }
  }
  // Re-sync a canvas's backing size to its live CSS box every draw. Returns false
  // if the canvas has no layout yet (0px) — a draw then would be invisible, so we
  // bail and let the next update()/ResizeObserver tick repaint. Self-heals the
  // "joint-state window sometimes blank" layout race.
  _fit(cv){
    const r = cv.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const w = Math.round(r.width * devicePixelRatio), h = Math.round(r.height * devicePixelRatio);
    if (cv.width !== w) cv.width = w;
    if (cv.height !== h) cv.height = h;
    return true;
  }

  setEpisode(entry){
    this.entry = entry; this.strip = false; this.stripJoints.clear();
    this.sc.classList.add('hidden'); this.bc.classList.remove('hidden');
    this.backBtn?.classList.add('hidden'); this.stripCtl?.classList.add('hidden');
    this.legendEl?.classList.remove('hidden'); if (this.hintEl) this.hintEl.textContent = 'click a bar → time strip';
    const names = entry.meta.actuator_names || [], groups = entry.meta.actuator_groups || {};
    this.blocks = [];
    for (const g of GROUP_ORDER){ const arr = groups[g]; if (!arr) continue;
      const idx = arr.map(n => names.indexOf(n)).filter(i => i >= 0);
      if (idx.length) this.blocks.push({ g, label: GROUP_LABEL[g] || g, idx }); }
    this.norm = {};
    for (const [topic, key] of [['joint_pos', 'pos'], ['joint_vel', 'vel'], ['joint_torque', 'torque'], ['action_ctrl', 'ctrl']]){
      const t = entry.bundle.topics[topic]; if (!t) continue;
      const dim = t.dim, T = entry.T, mx = new Float32Array(dim);
      for (let f = 0; f < T; f++) for (let j = 0; j < dim; j++){ const v = Math.abs(t.data[f * dim + j]); if (v > mx[j]) mx[j] = v; }
      for (let j = 0; j < dim; j++) if (mx[j] < 1e-6) mx[j] = 1;
      this.norm[key] = mx;
    }
    this.draw();
  }

  setMode(m){ this.mode = m; this._stripDirty = true; this.draw(); }
  update(entry, f){ this.frame = f; this.draw(); }
  _measTopic(){ return this.mode === 'pos' ? 'joint_pos' : this.mode === 'vel' ? 'joint_vel' : 'joint_torque'; }
  _measNorm(){ return this.mode === 'pos' ? this.norm.pos : this.mode === 'vel' ? this.norm.vel : this.norm.torque; }

  draw(){
    if (this.strip){ this._drawStrip(); return; }
    const cv = this.bc; if (!this._fit(cv)) return;        // no layout yet → repaint next tick
    const x = cv.getContext('2d'); const W = cv.width, H = cv.height;
    x.clearRect(0, 0, W, H);
    if (!this.entry || !this.blocks) return;
    const pad = 8 * devicePixelRatio, top = pad, bot = H - 16 * devicePixelRatio;
    const half = (bot - top) / 2, cy = top + half;
    x.strokeStyle = C.mid; x.lineWidth = 1; x.beginPath(); x.moveTo(0, cy); x.lineTo(W, cy); x.stroke();
    const meas = this.entry.bundle.topics[this._measTopic()], tgt = this.entry.bundle.topics['action_ctrl'];
    const mNorm = this._measNorm(), tNorm = this.norm.ctrl;
    const f = Math.max(0, Math.min(this.frame, this.entry.T - 1));
    const jstat = jointStatusAt(this.entry.quality, f, meas ? meas.dim : 59);   // per-joint health @ frame
    const totalBars = this.blocks.reduce((a, b) => a + b.idx.length, 0);
    const gaps = (this.blocks.length - 1) * 10 * devicePixelRatio + 2 * pad;
    const bw = (W - gaps) / totalBars;
    let bx = pad; this._barHit = [];
    for (let bi = 0; bi < this.blocks.length; bi++){
      const blk = this.blocks[bi];
      const blockX0 = bx;
      for (const j of blk.idx){
        const v = meas ? meas.data[f * meas.dim + j] / mNorm[j] : 0;
        const h = Math.max(1, Math.abs(v) * half * 0.94), yTop = v >= 0 ? cy - h : cy;
        const st = jstat[j] || 0;                                   // 0 ok / 1 warn / 2 missing
        const tw = Math.max(1.2 * devicePixelRatio, bw * 0.5);
        if (st === 0){                                             // good = green
          const grad = x.createLinearGradient(0, cy - half, 0, cy + half);
          grad.addColorStop(0, '#7ee6b0'); grad.addColorStop(1, STATUS_COL[0]); x.fillStyle = grad;
        } else {                                                    // yellow = low-freq/warn · red = missing
          x.fillStyle = STATUS_COL[st];
        }
        x.fillRect(bx + (bw - tw) / 2, yTop, tw, h);
        if (st === 2){                                              // missing joint: red floor tick
          x.fillStyle = STATUS_COL[2]; x.fillRect(bx + (bw - tw) / 2, cy - devicePixelRatio, tw, 2 * devicePixelRatio);
        }
        if (tgt){ const tv = tgt.data[f * tgt.dim + j] / tNorm[j], ty = cy - tv * half * 0.94;
          x.fillStyle = C.soft; x.fillRect(bx + (bw - tw) / 2 - devicePixelRatio, ty - devicePixelRatio, tw + 2 * devicePixelRatio, 1.6 * devicePixelRatio); }
        this._barHit.push({ j, x0: bx, x1: bx + bw }); bx += bw;
      }
      x.fillStyle = C.ash; x.font = `${9 * devicePixelRatio}px Geist Mono, monospace`; x.textAlign = 'center';
      x.fillText(blk.label, (blockX0 + bx) / 2, H - 4 * devicePixelRatio);
      // faint divider between groups (not after the last)
      if (bi < this.blocks.length - 1){
        const dx = bx + 5 * devicePixelRatio;
        x.strokeStyle = 'rgba(242,238,230,.10)'; x.lineWidth = devicePixelRatio;
        x.beginPath(); x.moveTo(dx, top); x.lineTo(dx, bot); x.stroke();
      }
      bx += 10 * devicePixelRatio;
    }
  }

  _pickBar(e){
    if (!this._barHit) return;
    const r = this.bc.getBoundingClientRect(), px = (e.clientX - r.left) * devicePixelRatio;
    const hit = this._barHit.find(b => px >= b.x0 && px < b.x1); if (!hit) return;
    this.strip = true; this.stripJoints = new Set([hit.j]); this._stripDirty = true;
    this.bc.classList.add('hidden'); this.sc.classList.remove('hidden'); this.backBtn?.classList.remove('hidden');
    this.legendEl?.classList.add('hidden'); if (this.hintEl) this.hintEl.textContent = 'drag to scrub · esc for all joints';
    this._buildStripCtl(); this._dpr(); this._drawStrip();
  }
  clearSelection(){ this.strip = false; this.stripJoints.clear(); this.sc.classList.add('hidden'); this.bc.classList.remove('hidden');
    this.backBtn?.classList.add('hidden'); this.stripCtl?.classList.add('hidden');
    this.legendEl?.classList.remove('hidden'); if (this.hintEl) this.hintEl.textContent = 'click a bar → time strip';
    this._dpr(); this.draw(); }

  addGroup(gkey){
    const groups = this.entry?.meta.actuator_groups || {}, names = this.entry?.meta.actuator_names || [];
    (groups[gkey] || []).forEach(n => { const i = names.indexOf(n); if (i >= 0) this.stripJoints.add(i); });
    this._stripDirty = true; this._drawStrip();
  }
  setShow(s){ this.show = s; this.stripCtl?.querySelectorAll('[data-show]').forEach(b => b.classList.toggle('on', b.dataset.show === s)); this._stripDirty = true; this._drawStrip(); }

  _buildStripCtl(){
    if (!this.stripCtl) return;
    this.stripCtl.classList.remove('hidden');
    this.stripCtl.innerHTML =
      `<div class="seg strip-show">
         <button data-show="measured" class="${this.show === 'measured' ? 'on' : ''}">meas</button>
         <button data-show="target" class="${this.show === 'target' ? 'on' : ''}">target</button>
         <button data-show="both" class="${this.show === 'both' ? 'on' : ''}">both</button></div>
       <span class="strip-add">add:</span>` +
      QUICK.filter(([, g]) => (this.entry?.meta.actuator_groups || {})[g])
        .map(([lbl, g]) => `<button class="strip-grp" data-g="${g}">+ ${lbl}</button>`).join('') +
      `<button class="strip-grp clr" data-clr="1">clear</button>`;
    this.stripCtl.querySelectorAll('[data-show]').forEach(b => b.onclick = () => this.setShow(b.dataset.show));
    this.stripCtl.querySelectorAll('[data-g]').forEach(b => b.onclick = () => this.addGroup(b.dataset.g));
    this.stripCtl.querySelector('[data-clr]').onclick = () => { const f = [...this.stripJoints][0]; this.stripJoints = new Set(f != null ? [f] : []); this._stripDirty = true; this._drawStrip(); };
  }

  // static curves + center line + legend → offscreen; rebuilt only on change
  _drawStripStatic(W, H){
    const dpr = devicePixelRatio;
    if (!this._stripOC) this._stripOC = document.createElement('canvas');
    this._stripOC.width = W; this._stripOC.height = H;
    const x = this._stripOC.getContext('2d'); x.clearRect(0, 0, W, H);
    const meas = this.entry.bundle.topics[this._measTopic()], tgt = this.entry.bundle.topics['action_ctrl'];
    const names = this.entry.meta.actuator_names || [], T = this.entry.T;
    const joints = [...this.stripJoints];
    const showM = this.show !== 'target', showT = this.show !== 'measured' && tgt;
    const top = 18 * dpr, bot = H - 6 * dpr, cy = (top + bot) / 2, half = (bot - top) / 2;
    let mx = 1e-6;
    for (const j of joints) for (let f = 0; f < T; f++){
      if (showM) mx = Math.max(mx, Math.abs(meas.data[f * meas.dim + j]));
      if (showT) mx = Math.max(mx, Math.abs(tgt.data[f * tgt.dim + j])); }
    const X = (f) => (f / Math.max(1, T - 1)) * W, Y = (v) => cy - (v / mx) * half * 0.92;
    x.strokeStyle = C.mid; x.beginPath(); x.moveTo(0, cy); x.lineTo(W, cy); x.stroke();
    for (const j of joints){
      const col = color(j);
      if (showM){ x.beginPath(); for (let f = 0; f < T; f++){ const xx = X(f), yy = Y(meas.data[f * meas.dim + j]); f ? x.lineTo(xx, yy) : x.moveTo(xx, yy); }
        x.strokeStyle = col; x.lineWidth = 1.6 * dpr; x.stroke(); }
      if (showT){ x.beginPath(); for (let f = 0; f < T; f++){ const xx = X(f), yy = Y(tgt.data[f * tgt.dim + j]); f ? x.lineTo(xx, yy) : x.moveTo(xx, yy); }
        x.strokeStyle = col; x.globalAlpha = 0.7; x.lineWidth = dpr; x.setLineDash([4 * dpr, 3 * dpr]); x.stroke(); x.setLineDash([]); x.globalAlpha = 1; }
    }
    x.font = `${9.5 * dpr}px Geist Mono, monospace`; x.textAlign = 'left';
    let lx = 8 * dpr;
    for (const j of joints.slice(0, 10)){
      x.fillStyle = color(j); x.fillRect(lx, 6 * dpr, 8 * dpr, 8 * dpr);
      x.fillStyle = '#cfc8be'; const nm = (names[j] || j) + ''; x.fillText(nm, lx + 11 * dpr, 13 * dpr);
      lx += (12 + nm.length * 5.6) * dpr;
    }
    if (joints.length > 10){ x.fillStyle = C.ash; x.fillText(`+${joints.length - 10} more`, lx, 13 * dpr); }
    this._stripT = T; this._stripCy = cy; this._stripHalf = half; this._stripMx = mx;
    this._stripMeas = showM ? meas : null; this._stripDirty = false;
  }
  _drawStrip(){
    const cv = this.sc; if (!this._fit(cv)) return;       // no layout yet → repaint next tick
    const x = cv.getContext('2d'); const W = cv.width, H = cv.height;
    x.clearRect(0, 0, W, H);
    if (!this.entry || !this.stripJoints.size) return;
    if (this._stripDirty || !this._stripOC || this._stripOC.width !== W || this._stripOC.height !== H) this._drawStripStatic(W, H);
    if (this._stripOC) x.drawImage(this._stripOC, 0, 0);
    const dpr = devicePixelRatio, T = this._stripT, half = this._stripHalf, cy = this._stripCy, mx = this._stripMx;
    const top = 18 * dpr, bot = H - 6 * dpr, ph = (Math.min(this.frame, T - 1) / Math.max(1, T - 1)) * W;
    x.strokeStyle = '#fff'; x.lineWidth = dpr; x.beginPath(); x.moveTo(ph, top); x.lineTo(ph, bot); x.stroke();
    const m = this._stripMeas, f = Math.min(this.frame, T - 1);
    if (m) for (const j of this.stripJoints){ x.fillStyle = color(j);
      x.beginPath(); x.arc(ph, cy - (m.data[f * m.dim + j] / mx) * half * 0.92, 2.4 * dpr, 0, 7); x.fill(); }
  }

  _stripScrub(e){
    if (!this.strip || !this.entry) return;
    const r = this.sc.getBoundingClientRect();
    const move = (ev) => { const f = Math.round(((ev.clientX - r.left) / r.width) * (this.entry.T - 1));
      this.onScrub?.(Math.max(0, Math.min(f, this.entry.T - 1))); };
    move(e);
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
}
