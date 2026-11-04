// Final-Cut-style timeline: ruler + a magnetic clip track + playhead, with
// trim (drag edges), blade (split), ripple-delete, merge, drag-reorder, append
// (drop episode), snapping and zoom. Drives + is driven by the global playhead.
import { api } from './api.js';
import { laneDefects, modalityDefects, categoryWord } from './defects.js';
import { get as statusGet } from './status.js';

// robot modules (label → actuator group) for the proprio/action drill-down.
// body = lift + torso. Each expands to its DOFs.
const MODULES = [['body', 'lift_torso'], ['base', 'base'], ['left arm', 'left_arm'],
  ['right arm', 'right_arm'], ['left hand', 'left_hand'], ['right hand', 'right_hand'], ['neck', 'neck']];

export class Timeline {
  constructor({ scrollEl, tracksEl, rulerEl, playheadEl, nameEl, seq, getFrame, setFrame, onToast, modalitiesFor, entryFor }){
    this.scroll = scrollEl; this.tracks = tracksEl; this.ruler = rulerEl;
    this.playhead = playheadEl; this.nameEl = nameEl;
    this.seq = seq; this.getFrame = getFrame; this.setFrame = setFrame; this.toast = onToast;
    this.modalitiesFor = modalitiesFor || (() => []);
    this.entryFor = entryFor || (() => null);   // clip → loaded episode entry (for clip waveforms)
    this.waveCh = 'action';                      // FCP-style "audio waveform" channel under clips
    this.expanded = new Set();          // clip ids whose modality lanes are open
    this.modExpanded = new Set();       // 'proprio'|'action' expanded into 7 modules
    this.dofExpanded = new Set();       // 'proprio:left_arm' expanded into per-DOF lanes
    this.markers = []; this.qc = [];    // [{frame,kind,label,color}] event markers / QC issues
    this.pxPerFrame = 6; this.snap = true; this.rctx = rulerEl.getContext('2d');
    this._wireScrubbing();
    new ResizeObserver(() => this.render()).observe(scrollEl);
    window.addEventListener('studio-resize', () => this.render());
    this._wireDrop();
  }

  setSnap(on){ this.snap = on; }
  setWave(ch){ this.waveCh = ch; this.render(); }
  zoom(mul){ this.pxPerFrame = Math.max(1, Math.min(40, this.pxPerFrame * mul)); this.render(); }
  zoomFit(){
    const w = this.scroll.clientWidth - 20;
    if (this.seq.totalFrames > 0) this.pxPerFrame = Math.max(1, w / this.seq.totalFrames);
    this.render();
  }

  _x(frame){ return frame * this.pxPerFrame; }
  _frame(px){ return Math.round(px / this.pxPerFrame); }

  _snapFrame(frame){
    if (!this.snap) return frame;
    const edges = [0, this.seq.totalFrames, this.getFrame()];
    let acc = 0; for (const c of this.seq.clips){ edges.push(acc); acc += c.len; } edges.push(acc);
    const tol = 6 / this.pxPerFrame;
    let best = frame, bd = tol;
    for (const e of edges){ const d = Math.abs(e - frame); if (d < bd){ bd = d; best = e; } }
    return best;
  }

  render(){
    const total = Math.max(this.seq.totalFrames, 60);
    const W = total * this.pxPerFrame + 20;
    this.tracks.style.width = W + 'px';
    this.ruler.style.width = W + 'px';
    this._drawRuler(W);
    this._drawClips();
    this.movePlayhead(this.getFrame());
    this.nameEl.textContent = this.seq.name + `  ·  ${this.seq.totalFrames}f / ${(this.seq.totalFrames / this.seq.fps).toFixed(1)}s`;
  }

  _drawRuler(W){
    const cv = this.ruler, x = this.rctx;
    cv.width = W * devicePixelRatio; cv.height = 26 * devicePixelRatio;
    cv.style.height = '26px';
    x.clearRect(0, 0, cv.width, cv.height); x.scale(devicePixelRatio, devicePixelRatio);
    x.fillStyle = '#6b655f'; x.font = '9px Geist Mono, monospace'; x.textBaseline = 'middle';
    const fps = this.seq.fps;
    // tick every ~80px, snapped to whole seconds when possible
    let stepF = Math.max(1, Math.round(80 / this.pxPerFrame));
    stepF = Math.max(stepF, Math.round(fps));         // at least 1s if zoomed out
    for (let f = 0; f <= this.seq.totalFrames; f += stepF){
      const px = this._x(f);
      x.strokeStyle = 'rgba(242,238,230,.18)'; x.beginPath(); x.moveTo(px, 14); x.lineTo(px, 26); x.stroke();
      const s = f / fps; x.fillText(`${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`, px + 3, 8);
    }
    // event markers (top ticks + dots) and QC issues (triangles)
    for (const m of this.markers){ const px = this._x(m.frame); const c = m.color || '#5eb0ff';
      x.strokeStyle = c; x.lineWidth = 1.5; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, 10); x.stroke();
      x.fillStyle = c; x.beginPath(); x.arc(px, 3, 2.2, 0, 7); x.fill(); }
    x.lineWidth = 1;
    for (const q of this.qc){ const px = this._x(q.frame);
      x.fillStyle = q.kind === 'error' ? '#D23B2A' : '#E9A679';
      x.beginPath(); x.moveTo(px - 3.5, 0); x.lineTo(px + 3.5, 0); x.lineTo(px, 7); x.closePath(); x.fill(); }
    x.setTransform(1, 0, 0, 1, 0, 0);
  }
  setEvents(markers, qc){ this.markers = markers || []; this.qc = qc || []; this.render(); }

  _drawClips(){
    this.tracks.innerHTML = '';
    // annotation lane (language labels) on top, FCP "titles" style
    const anno = document.createElement('div'); anno.className = 'tl-anno-lane';
    const atag = document.createElement('span'); atag.className = 'lane-tag'; atag.textContent = 'T · annotations';
    anno.appendChild(atag);
    for (const a of this.seq.annotations) anno.appendChild(this._annoEl(a));
    this.tracks.appendChild(anno);
    // ratings lane (reject = excised on save, favorite = keep)
    const rate = document.createElement('div'); rate.className = 'tl-rate-lane';
    const rtag = document.createElement('span'); rtag.className = 'lane-tag'; rtag.textContent = '⚑ ratings';
    rate.appendChild(rtag);
    for (const r of this.seq.ranges) rate.appendChild(this._rangeEl(r));
    this.tracks.appendChild(rate);
    // mode/command state lane (teleop/mode → MANIP / NAV segments, derived)
    const modeLane = this._modeLane(); if (modeLane) this.tracks.appendChild(modeLane);
    // clip lane
    const lane = document.createElement('div'); lane.className = 'tl-track';
    const tag = document.createElement('span'); tag.className = 'lane-tag'; tag.textContent = 'V1 · episodes';
    lane.appendChild(tag);
    for (const c of this.seq.clips) lane.appendChild(this._clipEl(c));
    this.tracks.appendChild(lane);
    // modality accordion: sub-lanes for expanded clips (keep/drop per modality)
    this._drawModalityLanes();
  }

  _groupIdx(entry){
    const names = entry?.meta?.actuator_names || [], groups = entry?.meta?.actuator_groups || {};
    const out = {};
    for (const [, g] of MODULES) out[g] = (groups[g] || []).map(n => names.indexOf(n)).filter(i => i >= 0);
    return out;
  }
  _jointLabel(entry, j){ return (entry?.meta?.actuator_names || [])[j] || ('DOF ' + j); }

  // ratio array (0..1 effective/nominal Hz) for a lane's channel, or null
  _laneRatio(clip, modId, group){
    const fr = this.entryFor(clip)?.freq; if (!fr || !fr.channels) return null;
    const ch = fr.channels;
    if (group) return ch[group]?.ratio || null;
    if (modId.startsWith('cam:')) return ch[modId.slice(4)]?.ratio || null;
    if (modId === 'action') return ch['action']?.ratio || null;
    if (modId === 'proprio'){                              // worst (min) across proprio modules
      const prop = Object.values(ch).filter(t => t.kind === 'proprio' && t.ratio);
      if (!prop.length) return null;
      const T = prop[0].ratio.length, out = new Array(T).fill(1);
      for (const t of prop) for (let i = 0; i < T; i++) out[i] = Math.min(out[i], t.ratio[i] ?? 1);
      return out;
    }
    return null;
  }
  // continuous green→amber→red frequency gradient under one clip on a lane
  _freqStrip(row, clip, ratio, label){
    if (!ratio || !ratio.length) return;
    const N = Math.min(64, Math.max(2, clip.len)), stops = [];
    let worst = 1;
    for (let i = 0; i < N; i++){
      const f = clip.inFrame + Math.round(i / (N - 1) * (clip.len - 1));
      const r = ratio[Math.min(f, ratio.length - 1)] ?? 1; worst = Math.min(worst, r);
      const h = Math.max(0, Math.min(1, (r - 0.2) / 0.8)) * 130;         // 0=red → 130=green
      stops.push(`hsl(${h.toFixed(0)},68%,47%) ${(i / (N - 1) * 100).toFixed(1)}%`);
    }
    const s = document.createElement('div'); s.className = 'mod-freq';
    s.style.left = this._x(clip.start) + 'px'; s.style.width = Math.max(4, this._x(clip.len)) + 'px';
    s.style.background = `linear-gradient(90deg,${stops.join(',')})`;
    s.title = `${label} frequency — worst ${(worst * 100).toFixed(0)}% of nominal`;
    row.appendChild(s);
  }
  // append coloured defect bars (yellow=warning, red=serious) for one clip's spans
  _appendDefects(row, clip, spans){
    for (const s of spans){
      const w = Math.max(3, this._x(s.hi - s.lo));
      const d = document.createElement('div'); d.className = 'mod-defect ' + s.cat;
      d.style.left = this._x(clip.start + s.lo) + 'px'; d.style.width = w + 'px';
      d.title = `${categoryWord(s.cat)} — ${s.label}`;
      if (w > 44){ const t = document.createElement('span'); t.className = 'md-lbl'; t.textContent = categoryWord(s.cat); d.appendChild(t); }
      row.appendChild(d);
    }
  }
  // a read-only drill-down lane (module or DOF) showing only defect highlighting
  _subLane(kind, key, label, jointSet, open, depth, expandable){
    const row = document.createElement('div'); row.className = 'tl-mod-lane sub depth' + depth;
    const tag = document.createElement('span'); tag.className = 'mod-tag';
    if (expandable){
      tag.classList.add('expandable');
      const cv = document.createElement('span'); cv.className = 'mod-caret';
      cv.textContent = this.dofExpanded.has(key) ? '▾' : '▸';
      tag.appendChild(cv);
      tag.onclick = (e) => { e.stopPropagation(); this.dofExpanded.has(key) ? this.dofExpanded.delete(key) : this.dofExpanded.add(key); this.render(); };
    }
    tag.appendChild(document.createTextNode(label));
    row.appendChild(tag);
    const group = depth === 1 ? key.split(':')[1] : null;   // module lane → per-module frequency
    for (const c of open){
      const entry = this.entryFor(c);
      if (group) this._freqStrip(row, c, this._laneRatio(c, kind, group), label);
      this._appendDefects(row, c, laneDefects(entry?.quality, kind, jointSet, c.inFrame, c.len));
    }
    this.tracks.appendChild(row);
  }

  _drawModalityLanes(){
    const open = this.seq.clips.filter(c => this.expanded.has(c.id));
    if (!open.length) return;
    const mods = this.modalitiesFor(open[0]);
    const gidx = this._groupIdx(this.entryFor(open[0]));
    for (const m of mods){
      const isVid = m.id.startsWith('cam:');
      const modExpandable = m.id === 'proprio' || m.id === 'action';
      const row = document.createElement('div'); row.className = 'tl-mod-lane' + (isVid ? ' vid' : '');
      const tag = document.createElement('span'); tag.className = 'mod-tag';
      if (modExpandable){
        tag.classList.add('expandable');                 // whole label is the toggle (clickable)
        const cv = document.createElement('span'); cv.className = 'mod-caret';
        cv.textContent = this.modExpanded.has(m.id) ? '▾' : '▸';
        tag.appendChild(cv);
        tag.onclick = (e) => { e.stopPropagation(); this.modExpanded.has(m.id) ? this.modExpanded.delete(m.id) : this.modExpanded.add(m.id); this.render(); };
      }
      tag.appendChild(document.createTextNode(m.label));
      row.appendChild(tag);
      // merge-back affordance if any open clip has this modality edited
      if (open.some(c => this.seq.isEdited(c, m.id))){
        const mb = document.createElement('button'); mb.className = 'mod-mergeback'; mb.textContent = '⤺'; mb.title = 'merge back (restore full)';
        mb.onclick = (e) => { e.stopPropagation(); open.forEach(c => this.seq.restoreModality(c, m.id)); };
        row.appendChild(mb);
      }
      for (const c of open){
        const ranges = this.seq.modRanges(c, m.id);
        if (!ranges.length){                          // fully dropped → ghost block + undelete
          const g = document.createElement('div'); g.className = 'mod-blk dropped ghost';
          g.style.left = this._x(c.start) + 'px'; g.style.width = Math.max(6, this._x(c.len) - 2) + 'px';
          g.title = `${m.label} — DROPPED`;
          if (this._isSelMod(c, m.id, null)) g.classList.add('sel');
          g.innerHTML = `<button class="mod-undelete" title="undelete — bring this modality back">↺ undelete</button>`;
          g.querySelector('.mod-undelete').onclick = (e) => { e.stopPropagation(); this.seq.restoreModality(c, m.id); };
          g.onpointerdown = (e) => { if (e.target.closest('.mod-undelete')) return; e.stopPropagation(); this.selectMod(c, m.id, null); };
          row.appendChild(g); continue;
        }
        const off = this.seq.getModOffset(c, m.id);
        ranges.forEach(([lo, hi], idx) => {
          const blk = document.createElement('div');
          blk.className = 'mod-blk' + (isVid ? ' vid' : '') + (off ? ' shifted' : '') + (this._isSelMod(c, m.id, idx) ? ' sel' : '');
          blk.style.left = this._x(c.start + lo + off) + 'px';      // visually shifted by the time offset
          blk.style.width = Math.max(6, this._x(hi - lo) - 1) + 'px';
          if (isVid){
            const cam = m.id.slice(4), w = this._x(hi - lo), n = Math.max(1, Math.min(10, Math.round(w / 40)));
            let strip = '';
            for (let k = 0; k < n; k++){ const sf = Math.round(c.inFrame + lo + (k + 0.5) / n * (hi - lo));
              strip += `<img src="${api.camFrameUrl(c.ds, c.ep, cam, sf)}" loading="lazy">`; }
            blk.innerHTML = `<div class="mod-strip">${strip}</div>`;
          }
          blk.innerHTML += `<div class="edge l"></div><div class="edge r"></div>` + (off ? `<span class="mod-off">${off > 0 ? '+' : ''}${off}f</span>` : '');
          blk.title = `${m.label} [${lo}–${hi}]${off ? ` · offset ${off}f` : ''} — click select · drag edges trim · ⌥drag to time-offset · B blade · ⌫ delete · ⌘J restore`;
          blk.onpointerdown = (e) => {
            if (e.target.classList.contains('edge')) return this._beginModTrim(e, c, m.id, idx, e.target.classList.contains('l') ? 'l' : 'r');
            if (e.altKey) return this._beginModOffset(e, c, m.id);
            e.stopPropagation(); this.selectMod(c, m.id, idx);
          };
          row.appendChild(blk);
        });
      }
      // frequency heat (under) + defect overlay (over) for this modality
      for (const c of open){
        const entry = this.entryFor(c);
        this._freqStrip(row, c, this._laneRatio(c, m.id), m.label);
        const spans = modExpandable
          ? laneDefects(entry?.quality, m.id, null, c.inFrame, c.len)
          : modalityDefects(entry?.quality, m.id, isVid ? m.id.slice(4) : null, c.inFrame, c.len);
        this._appendDefects(row, c, spans);
      }
      this.tracks.appendChild(row);
      // module drill-down (7 modules → DOFs) for proprio / action
      if (modExpandable && this.modExpanded.has(m.id)){
        for (const [lbl, g] of MODULES){
          const js = gidx[g] || []; if (!js.length) continue;
          const key = m.id + ':' + g;
          this._subLane(m.id, key, lbl, new Set(js), open, 1, true);
          if (this.dofExpanded.has(key))
            for (const j of js)
              this._subLane(m.id, key + ':' + j, this._jointLabel(this.entryFor(open[0]), j), new Set([j]), open, 2, false);
        }
      }
    }
  }

  _isSelMod(c, modId, idx){ const s = this.selMod; return s && s.clip === c && s.modId === modId && s.idx === idx; }
  selectMod(clip, modId, idx){ this.selMod = { clip, modId, idx }; this.seq.clearSelection(); this.render(); }
  clearSelMod(){ this.selMod = null; }
  _beginModTrim(e, clip, modId, idx, side){
    e.stopPropagation(); e.preventDefault(); this.selectMod(clip, modId, idx);
    const startX = e.clientX; const r0 = this.seq.modRanges(clip, modId)[idx].slice(); let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      const val = side === 'l' ? r0[0] + df : r0[1] + df;
      this.seq.trimModality(clip, modId, idx, side, val);
      this._previewAtGlobal(clip.start + (side === 'l' ? this.seq.modRanges(clip, modId)[idx][0] : this.seq.modRanges(clip, modId)[idx][1] - 1), ev.clientX, ev.clientY); };
    const up = () => { this._hidePreview(); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  // ⌥-drag a modality lane horizontally → set its per-clip time offset (J/L-cut /
  // sensor-sync). Editor shows it shifted; materialize applies it on Save.
  _beginModOffset(e, clip, modId){
    e.stopPropagation(); e.preventDefault();
    const startX = e.clientX, o0 = this.seq.getModOffset(clip, modId); let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      clip.modOffset = clip.modOffset || {};
      const n = o0 + df; if (n) clip.modOffset[modId] = n; else delete clip.modOffset[modId];
      this.seq._changed(); };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }

  toggleExpand(clip){
    if (this.expanded.has(clip.id)) this.expanded.delete(clip.id); else this.expanded.add(clip.id);
    this.render();
  }

  _annoEl(a){
    const kind = a.kind || 'language';
    const KINDS = ['language', 'skill', 'phase', 'success', 'failure'];
    const TAG = { language: '💬', skill: '◆', phase: '▮', success: '✓', failure: '✕' };
    const el = document.createElement('div'); el.className = 'anno kind-' + kind;
    el.style.left = this._x(a.start) + 'px';
    el.style.width = Math.max(28, this._x(a.end - a.start) - 2) + 'px';
    el.innerHTML = `<button class="anno-kind" title="type: ${kind} — click to change">${TAG[kind] || '•'}</button>` +
      `<span class="anno-text">${a.text}</span><div class="edge l"></div><div class="edge r"></div>`;
    el.title = `${kind} — drag move · edges resize · double-click edit · click ${TAG[kind] || '•'} to change type`;
    const kb = el.querySelector('.anno-kind');
    kb.onpointerdown = (e) => e.stopPropagation();
    kb.onclick = (e) => { e.stopPropagation(); this.seq.setAnnotationKind(a.id, KINDS[(KINDS.indexOf(kind) + 1) % KINDS.length]); };
    el.onpointerdown = (e) => {
      if (e.target.classList.contains('edge')) return this._annoResize(e, a, e.target.classList.contains('l') ? 'start' : 'end');
      this._annoDrag(e, a, el);
    };
    el.ondblclick = (e) => { e.stopPropagation(); this._editAnno(a, el); };
    return el;
  }

  // derived state lane: teleop/mode collapsed into contiguous MANIP/NAV segments
  _modeLane(){
    const segs = [];
    for (const c of this.seq.clips){
      const e = this.entryFor(c); if (!e) continue;
      const t = e.bundle.topics['teleop/mode']; if (!t) continue;
      let s0 = null, val = null;
      for (let f = c.inFrame; f < c.outFrame; f++){
        const v = Math.round(t.data[f * t.dim]);
        if (val === null){ val = v; s0 = f; }
        else if (v !== val){ segs.push({ g0: c.start + (s0 - c.inFrame), g1: c.start + (f - c.inFrame), val }); val = v; s0 = f; }
      }
      if (val !== null) segs.push({ g0: c.start + (s0 - c.inFrame), g1: c.start + (c.outFrame - c.inFrame), val });
    }
    if (!segs.length) return null;
    const lane = document.createElement('div'); lane.className = 'tl-mode-lane';
    const tag = document.createElement('span'); tag.className = 'lane-tag'; tag.textContent = '◧ mode'; lane.appendChild(tag);
    for (const s of segs){
      const el = document.createElement('div'); el.className = 'mode-seg ' + (s.val > 0.5 ? 'nav' : 'manip');
      el.style.left = this._x(s.g0) + 'px'; el.style.width = Math.max(2, this._x(s.g1 - s.g0) - 1) + 'px';
      el.textContent = s.val > 0.5 ? 'NAV' : 'MANIP'; el.title = `${s.val > 0.5 ? 'NAV' : 'MANIP'} [${s.g0}–${s.g1}]`;
      lane.appendChild(el);
    }
    return lane;
  }
  _rangeEl(r){
    const el = document.createElement('div'); el.className = 'rate ' + r.kind;
    el.style.left = this._x(r.start) + 'px';
    el.style.width = Math.max(16, this._x(r.end - r.start) - 2) + 'px';
    el.innerHTML = `<span class="rate-lbl">${r.kind === 'reject' ? '✕ reject' : '★ keep'}</span>
      <button class="rate-x" title="remove">×</button><div class="edge l"></div><div class="edge r"></div>`;
    el.title = 'drag move · edges resize · double-click toggles reject/keep';
    el.onpointerdown = (e) => {
      if (e.target.classList.contains('rate-x')) return;
      if (e.target.classList.contains('edge')) return this._rangeResize(e, r, e.target.classList.contains('l') ? 'start' : 'end');
      this._rangeDrag(e, r, el);
    };
    el.querySelector('.rate-x').onclick = (e) => { e.stopPropagation(); this.seq.removeRange(r.id); };
    el.ondblclick = (e) => { e.stopPropagation(); this.seq.toggleRangeKind(r.id); };
    return el;
  }
  _rangeDrag(e, r, el){
    e.preventDefault(); e.stopPropagation();
    const startX = e.clientX, s0 = r.start, len = r.end - r.start; let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      const ns = Math.max(0, s0 + df); this.seq.updateRange(r.id, { start: ns, end: ns + len }); };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  _rangeResize(e, r, side){
    e.preventDefault(); e.stopPropagation();
    const startX = e.clientX, s0 = r.start, e0 = r.end; let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      if (side === 'start') this.seq.updateRange(r.id, { start: Math.max(0, Math.min(s0 + df, e0 - 1)) });
      else this.seq.updateRange(r.id, { end: Math.max(s0 + 1, e0 + df) }); };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }

  _editAnno(a, el){
    const span = el.querySelector('.anno-text');
    const inp = document.createElement('input'); inp.className = 'anno-input'; inp.value = a.text;
    span.replaceWith(inp); inp.focus(); inp.select();
    const done = (save) => { if (save && inp.value.trim()) this.seq.setAnnotationText(a.id, inp.value.trim());
      else this.render(); };
    inp.onkeydown = (e) => { if (e.key === 'Enter'){ e.preventDefault(); done(true); } else if (e.key === 'Escape') done(false); e.stopPropagation(); };
    inp.onblur = () => done(true);
    inp.onpointerdown = (e) => e.stopPropagation();
  }
  _annoDrag(e, a, el){
    e.preventDefault(); e.stopPropagation();
    const startX = e.clientX, s0 = a.start, len = a.end - a.start; let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      const ns = Math.max(0, s0 + df); this.seq.updateAnnotation(a.id, { start: ns, end: ns + len });
      this._previewAtGlobal(ns, ev.clientX, ev.clientY); };
    const up = () => { this._hidePreview(); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  _annoResize(e, a, side){
    e.preventDefault(); e.stopPropagation();
    const startX = e.clientX, s0 = a.start, e0 = a.end; let committed = false;
    const move = (ev) => { const df = Math.round((ev.clientX - startX) / this.pxPerFrame); if (!df && !committed) return;
      if (!committed){ this.seq.commit(); committed = true; }
      let edge;
      if (side === 'start'){ edge = Math.max(0, Math.min(s0 + df, e0 - 1)); this.seq.updateAnnotation(a.id, { start: edge }); }
      else { edge = Math.max(s0 + 1, e0 + df); this.seq.updateAnnotation(a.id, { end: edge }); }
      this._previewAtGlobal(side === 'start' ? edge : edge - 1, ev.clientX, ev.clientY); };
    const up = () => { this._hidePreview(); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }

  _clipEl(c){
    const el = document.createElement('div');
    el.className = 'clip' + (c.selected ? ' sel' : '');
    el.dataset.clipId = c.id;
    el.style.left = this._x(c.start) + 'px';
    el.style.width = Math.max(8, this._x(c.len) - 2) + 'px';
    el.style.background = `linear-gradient(180deg, ${c.color}55, ${c.color}33)`;
    const tStrip = `<div class="thumb-strip">${this._thumbs(c).map(u => `<img src="${u}" loading="lazy">`).join('')}</div>`;
    const dropN = (c.drop || []).length;
    // per-clip curation state: green ✓ = exported/saved, amber ◐ = opened not saved
    const st = statusGet(c.ds, c.ep);
    const stBadge = st ? `<div class="clip-stat ${st}" title="${st === 'saved' ? 'exported (done)' : 'edited — not exported yet'}">${st === 'saved' ? '✓' : '◐'}</div>` : '';
    if (st) el.classList.add('is-' + st);
    el.innerHTML = `${tStrip}${stBadge}
      <canvas class="clip-wave"></canvas>
      <button class="clip-exp" title="expand modalities (E)">${this.expanded.has(c.id) ? '▾' : '▸'}</button>
      <button class="clip-rate${c.rate && c.rate !== 1 ? ' on' : ''}" title="retime / resample on Save — click to cycle ×0.5 / 1 / 2 / 4">${c.rate && c.rate !== 1 ? '×' + c.rate : '⏱'}</button>
      <div class="clip-label">${c.epLabel}</div>
      <div class="clip-sub">${c.inFrame}–${c.outFrame} · ${c.len}f${dropN ? ` · −${dropN} mod` : ''}${c.rate && c.rate !== 1 ? ` · ×${c.rate}` : ''}</div>
      <div class="edge l"></div><div class="edge r"></div>`;
    const rb = el.querySelector('.clip-rate');
    rb.onpointerdown = (e) => e.stopPropagation();
    rb.onclick = (e) => { e.stopPropagation(); const order = [1, 2, 4, 0.5]; const cur = c.rate || 1;
      this.seq.setRate(c, order[(order.indexOf(cur) + 1) % order.length]); };
    if (this.waveCh && this.waveCh !== 'off') this._drawWave(el.querySelector('.clip-wave'), c);
    el.querySelector('.clip-exp').onpointerdown = (e) => { e.stopPropagation(); };
    el.querySelector('.clip-exp').onclick = (e) => { e.stopPropagation(); this.toggleExpand(c); };
    el.onpointerdown = (e) => {
      if (e.target.classList.contains('edge')) return this._beginTrim(e, c, e.target.classList.contains('l') ? 'in' : 'out');
      this._beginDrag(e, c, el);
    };
    return el;
  }

  // FCP-style waveform under a clip: a chosen scalar across the clip, so events
  // (grasp, contact, motion bursts) are scannable without expanding modalities.
  _drawWave(cv, c){
    const e = this.entryFor(c); if (!e || !cv) return;
    const B = e.bundle.topics; let getter = null;
    const norm = (t) => (f) => { let s = 0; for (let k = 0; k < t.dim; k++){ const v = t.data[f * t.dim + k]; s += v * v; } return Math.sqrt(s); };
    if (this.waveCh === 'action' && B.action_ctrl) getter = norm(B.action_ctrl);
    else if (this.waveCh === 'jvel' && B.joint_vel) getter = norm(B.joint_vel);
    else if (this.waveCh === 'speed' && B.base_twist){ const t = B.base_twist; getter = (f) => Math.hypot(t.data[f * t.dim], t.data[f * t.dim + 1]); }
    else if (this.waveCh === 'graspL' && B['teleop/grasp']){ const t = B['teleop/grasp']; getter = (f) => t.data[f * t.dim]; }
    else if (this.waveCh === 'graspR' && B['teleop/grasp']){ const t = B['teleop/grasp']; getter = (f) => t.data[f * t.dim + 1]; }
    if (!getter) return;
    const dpr = devicePixelRatio, cw = Math.max(8, this._x(c.len) - 2), H = 16 * dpr;
    cv.width = Math.round(cw * dpr); cv.height = H; cv.style.width = cw + 'px';
    const x = cv.getContext('2d'), W = cv.width;
    const N = Math.max(2, Math.min(c.len, Math.round(cw)));
    const vals = new Float32Array(N); let mx = 1e-6;
    for (let i = 0; i < N; i++){ const f = c.inFrame + Math.round(i / (N - 1) * (c.len - 1)); const v = Math.abs(getter(f) || 0); vals[i] = v; if (v > mx) mx = v; }
    x.fillStyle = 'rgba(247,213,184,.28)'; x.beginPath(); x.moveTo(0, H);
    for (let i = 0; i < N; i++){ x.lineTo(i / (N - 1) * W, H - (vals[i] / mx) * H * 0.92); }
    x.lineTo(W, H); x.closePath(); x.fill();
  }

  _thumbs(c){
    const n = Math.max(1, Math.min(10, Math.round(this._x(c.len) / 56)));
    const cam = c.cam || 'zed_left';            // the head cam filmstrip, FCP-style
    const out = [];
    for (let k = 0; k < n; k++){
      const sf = Math.round(c.inFrame + (k + 0.5) / n * (c.outFrame - c.inFrame));
      out.push(api.camFrameUrl(c.ds, c.ep, cam, sf));
    }
    return out;
  }

  // Floating trim-preview: the head-cam frame at the edge being dragged, so you
  // can see exactly where you're cutting (like FCP's skimming preview).
  _trimPreview(clip, sourceFrame, clientX, clientY){
    let el = this._tp;
    if (!el){
      el = this._tp = document.createElement('div'); el.className = 'trim-preview';
      el.innerHTML = '<img><div class="tp-cap"></div>'; document.body.appendChild(el);
    }
    el.style.display = 'block';
    el.style.left = Math.min(window.innerWidth - 230, clientX + 14) + 'px';
    el.style.top = Math.max(8, clientY - 150) + 'px';
    el.querySelector('img').src = api.camFrameUrl(clip.ds, clip.ep, clip.cam || 'zed_left', sourceFrame);
    el.querySelector('.tp-cap').textContent = `frame ${sourceFrame}`;
  }
  _hidePreview(){ if (this._tp) this._tp.style.display = 'none'; }

  // scrub the whole studio to a global timeline frame + show the head-cam thumb
  // (used by annotation move/resize so you can place a label precisely)
  _previewAtGlobal(globalFrame, clientX, clientY){
    const g = Math.max(0, Math.min(globalFrame, Math.max(0, this.seq.totalFrames - 1)));
    this.setFrame(g);
    const r = this.seq.resolve(g);
    if (r) this._trimPreview(r.clip, r.sourceFrame, clientX, clientY);
  }

  movePlayhead(frame){
    this.playhead.style.left = this._x(frame) + 'px';
    // span the full content height so it reaches the last (expanded) modality lane
    const h = (this.ruler.offsetHeight || 26) + this.tracks.offsetHeight;
    this.playhead.style.height = h + 'px';
  }

  _wireScrubbing(){
    const scrubAt = (clientX) => {
      const r = this.scroll.getBoundingClientRect();
      const px = clientX - r.left + this.scroll.scrollLeft;
      let f = this._snapFrame(this._frame(px));
      f = Math.max(0, Math.min(f, Math.max(0, this.seq.totalFrames - 1)));
      this.movePlayhead(f);         // move the line immediately, before the heavier panel sync
      this.setFrame(f);
    };
    const down = (e) => {
      if (e.target.closest('.clip')) return;     // clip handles its own pointer
      scrubAt(e.clientX);
      const move = (ev) => scrubAt(ev.clientX);
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    };
    this.scroll.addEventListener('pointerdown', down);
    this.scroll.addEventListener('wheel', (e) => {
      if (e.ctrlKey || e.metaKey){ e.preventDefault(); this.zoom(e.deltaY < 0 ? 1.15 : 0.87); }
    }, { passive: false });
  }

  _beginTrim(e, clip, side){
    e.stopPropagation(); e.preventDefault();
    this.seq.select(clip);
    const startX = e.clientX;
    const in0 = clip.inFrame, out0 = clip.outFrame;
    let committed = false;
    const move = (ev) => {
      const df = Math.round((ev.clientX - startX) / this.pxPerFrame);
      if (df === 0) return;
      if (!committed){ this.seq.commit(); committed = true; }   // one undo step per trim
      let srcFrame;
      if (side === 'in'){ this.seq.trim(clip, 'in', in0 + df, clip.srcMax); srcFrame = clip.inFrame; }
      else { this.seq.trim(clip, 'out', out0 + df, clip.srcMax); srcFrame = clip.outFrame - 1; }
      // live preview: scrub the whole studio to the edge frame + floating thumb
      const globalFrame = side === 'in' ? clip.start : clip.start + clip.len - 1;
      this.setFrame(globalFrame);
      this._trimPreview(clip, srcFrame, ev.clientX, ev.clientY);
    };
    const up = () => { this._hidePreview(); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }

  _beginDrag(e, clip, el){
    e.preventDefault();
    this.selMod = null;                              // clip selection clears modality selection
    this.seq.select(clip, e.shiftKey);               // NOTE: re-renders, detaching `el`
    el = this.tracks.querySelector(`.clip[data-clip-id="${clip.id}"]`) || el;   // re-acquire live element
    const startX = e.clientX; let moved = false;
    const move = (ev) => {
      const dx = ev.clientX - startX;
      if (!moved && Math.abs(dx) < 4) return; moved = true;
      // original clip lifts in place; a small clip "ghost" rides the cursor
      el.classList.add('dragging'); document.body.classList.add('tl-dragging');
      this._showDragGhost(clip, ev);
      // insertion index among the OTHER clips, by midpoint
      const others = this.seq.clips.filter(c => c !== clip);
      const r = this.scroll.getBoundingClientRect();
      const f = this._frame(ev.clientX - r.left + this.scroll.scrollLeft);
      let oidx = 0, acc = 0;
      for (const cc of others){ if (f > acc + cc.len / 2) oidx++; acc += cc.len; }
      this._dropOIdx = oidx;
      let bf = 0; for (let k = 0; k < oidx; k++) bf += others[k].len;
      this._showDropMark(bf);
    };
    const up = () => {
      el.classList.remove('dragging'); document.body.classList.remove('tl-dragging');
      this._hideDropMark(); this._hideDragGhost();
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      if (moved && this._dropOIdx != null){ this.seq.commit(); this.seq.reorder(clip, this._dropOIdx); this._dropOIdx = null; }
    };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  // a small floating clip (thumbnail + label) pinned to the cursor while dragging
  _showDragGhost(clip, ev){
    let g = this._dragGhost;
    if (!g || !g.isConnected){
      g = document.createElement('div'); g.className = 'tl-drag-ghost';
      g.innerHTML = '<img class="tl-dg-thumb" alt=""><span class="tl-dg-label"></span>';
      document.body.appendChild(g); this._dragGhost = g;
    }
    if (g._clipId !== clip.id){                          // set thumb + label once per drag
      g._clipId = clip.id;
      g.querySelector('.tl-dg-thumb').src = api.camFrameUrl(clip.ds, clip.ep, clip.cam || 'zed_left', clip.inFrame);
      g.querySelector('.tl-dg-label').textContent = clip.epLabel || clip.ep || 'clip';
    }
    g.style.display = 'flex';
    g.style.left = Math.min(window.innerWidth - 150, ev.clientX + 14) + 'px';
    g.style.top = Math.max(6, ev.clientY + 12) + 'px';
  }
  _hideDragGhost(){ if (this._dragGhost){ this._dragGhost.style.display = 'none'; this._dragGhost._clipId = null; } }
  // vertical insertion line showing where the dragged clip will drop
  _showDropMark(frame){
    if (!this._dropMark || !this._dropMark.isConnected){
      this._dropMark = document.createElement('div'); this._dropMark.className = 'tl-dropmark';
      this.tracks.appendChild(this._dropMark);
    }
    this._dropMark.style.left = this._x(frame) + 'px'; this._dropMark.style.display = 'block';
  }
  _hideDropMark(){ if (this._dropMark) this._dropMark.style.display = 'none'; }

  _wireDrop(){
    this.scroll.addEventListener('dragover', (e) => e.preventDefault());
    this.scroll.addEventListener('drop', (e) => {
      e.preventDefault();
      try { const d = JSON.parse(e.dataTransfer.getData('text/plain')); document.dispatchEvent(new CustomEvent('drop-episode', { detail: d })); } catch {}
    });
  }
}
