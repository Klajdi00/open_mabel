// Magnetic single-track sequence of clips. Every Final-Cut op is a pure
// transform on the clip list; resolve() maps a global timeline frame to a
// (clip, source frame) so scrubbing works across multi-source cuts. toEDL()
// produces the backend save payload.
import * as bus from './bus.js';

let _id = 1;
const PALETTE = ['#C25B2A', '#A8642F', '#8B5A3C', '#9E5530', '#B5683A'];

export class Sequence {
  constructor(fps = 15){
    this.fps = fps; this.clips = []; this.name = 'untitled sequence';
    this.annotations = [];                 // [{id, text, start, end}] language labels (timeline frames)
    this.ranges = [];                      // [{id, start, end, kind:'reject'|'favorite'}] take-rating spans
    this.undoStack = []; this.redoStack = [];
    this._relayout();
  }

  // ── annotations (typed labels on the timeline) ─────────────────────────
  // kind ∈ language | skill | phase | success | failure
  addAnnotation(text, start, end, kind = 'language'){
    this.commit();
    const a = { id: _id++, text: text || 'label', start: Math.max(0, start | 0), end: Math.max(start + 1, end | 0), kind };
    this.annotations.push(a); this._changed(); return a;
  }
  setAnnotationKind(id, kind){ this.commit(); this.updateAnnotation(id, { kind }); }
  // Replace all 'phase' annotations with segments between consecutive boundaries
  // (auto-segmentation from event markers). bounds = sorted [{frame,label}].
  autoPhases(bounds){
    this.commit();
    this.annotations = this.annotations.filter(a => a.kind !== 'phase');
    for (let i = 0; i < bounds.length - 1; i++){
      const a = bounds[i].frame | 0, b = bounds[i + 1].frame | 0; if (b <= a) continue;
      this.annotations.push({ id: _id++, text: bounds[i].label || ('phase ' + (i + 1)), start: a, end: b, kind: 'phase' });
    }
    this._changed();
  }
  updateAnnotation(id, fields){
    const a = this.annotations.find(x => x.id === id); if (!a) return;
    Object.assign(a, fields);
    if (a.end <= a.start) a.end = a.start + 1;
    this._changed();
  }
  setAnnotationText(id, text){ this.commit(); this.updateAnnotation(id, { text }); }
  removeAnnotation(id){ this.commit(); this.annotations = this.annotations.filter(x => x.id !== id); this._changed(); }

  // ── take-rating ranges (reject = excise on save; favorite = keep/mark) ──────
  addRange(start, end, kind = 'reject'){
    this.commit();
    const r = { id: _id++, start: Math.max(0, start | 0), end: Math.max((start | 0) + 1, end | 0), kind };
    this.ranges.push(r); this._changed(); return r;
  }
  updateRange(id, fields){
    const r = this.ranges.find(x => x.id === id); if (!r) return;
    Object.assign(r, fields); if (r.end <= r.start) r.end = r.start + 1; this._changed();
  }
  toggleRangeKind(id){ this.commit(); const r = this.ranges.find(x => x.id === id);
    if (r){ r.kind = r.kind === 'reject' ? 'favorite' : 'reject'; this._changed(); } }
  removeRange(id){ this.commit(); this.ranges = this.ranges.filter(x => x.id !== id); this._changed(); }
  rejectedFrames(){ return this.ranges.filter(r => r.kind === 'reject').reduce((a, r) => a + (r.end - r.start), 0); }

  // ── undo / redo ────────────────────────────────────────────────────────
  // Clips hold only primitives, so a shallow per-clip copy is a full snapshot.
  _cloneClip(c){ return { ...c, modKeep: JSON.parse(JSON.stringify(c.modKeep || {})), modOffset: { ...(c.modOffset || {}) } }; }
  // per-modality time offset (frames; +n delays this modality, edge-held at clip bounds)
  getModOffset(clip, modId){ return (clip.modOffset && clip.modOffset[modId]) || 0; }
  setModOffset(clip, modId, n){ this.commit(); clip.modOffset = clip.modOffset || {};
    if (n) clip.modOffset[modId] = n | 0; else delete clip.modOffset[modId]; this._changed(); }
  // retime (speed factor): resamples the clip on Save only — editor stays source-rate
  setRate(clip, r){ this.commit(); if (r && r !== 1) clip.rate = r; else delete clip.rate; this._changed(); }
  _snap(){ return { clips: this.clips.map(c => this._cloneClip(c)), annotations: this.annotations.map(a => ({ ...a })),
    ranges: this.ranges.map(r => ({ ...r })), name: this.name, fps: this.fps }; }
  _restore(s){ this.clips = s.clips.map(c => this._cloneClip(c)); this.annotations = (s.annotations || []).map(a => ({ ...a }));
    this.ranges = (s.ranges || []).map(r => ({ ...r }));
    this.name = s.name; this.fps = s.fps; this._relayout(); this._changed(); }

  // ── per-modality keep-ranges (clip-local frames) ───────────────────────────
  // modKeep[modId] = sorted [[lo,hi),…] kept spans. Absent id = full keep; [] = fully dropped.
  modLen(clip){ return clip.outFrame - clip.inFrame; }
  modRanges(clip, modId){
    const n = this.modLen(clip);
    return (clip.modKeep && clip.modKeep[modId]) ? clip.modKeep[modId] : [[0, n]];
  }
  // normalize: clamp, drop empties, sort, merge OVERLAPping (touching stays split so blades survive)
  _setMod(clip, modId, ranges){
    const n = this.modLen(clip);
    let r = ranges.map(([a, b]) => [Math.max(0, Math.min(a, n)), Math.max(0, Math.min(b, n))])
      .filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0]);
    const out = [];
    for (const seg of r){ const last = out[out.length - 1];
      if (last && seg[0] < last[1]) last[1] = Math.max(last[1], seg[1]); else out.push(seg.slice()); }
    clip.modKeep = clip.modKeep || {};
    if (out.length === 1 && out[0][0] === 0 && out[0][1] === n) delete clip.modKeep[modId];  // back to default
    else clip.modKeep[modId] = out;
  }
  splitModalityAt(clip, modId, f){            // f = clip-local frame (blade)
    this.commit();
    const out = [];
    for (const [a, b] of this.modRanges(clip, modId)) (f > a && f < b) ? out.push([a, f], [f, b]) : out.push([a, b]);
    this._setMod(clip, modId, out); this._changed();
  }
  deleteModalityRange(clip, modId, idx){      // drop one kept span
    this.commit();
    const r = this.modRanges(clip, modId).map(s => s.slice()); r.splice(idx, 1);
    this._setMod(clip, modId, r); this._changed();
  }
  dropModality(clip, modId){ this.commit(); clip.modKeep = clip.modKeep || {}; clip.modKeep[modId] = []; this._changed(); }
  trimModality(clip, modId, idx, side, val){  // drag (caller commits once at gesture start)
    const r = this.modRanges(clip, modId).map(s => s.slice()); if (!r[idx]) return;
    if (side === 'l') r[idx][0] = Math.max(0, Math.min(val, r[idx][1] - 1));
    else r[idx][1] = Math.max(r[idx][0] + 1, val);
    this._setMod(clip, modId, r); this._changed();
  }
  restoreModality(clip, modId){               // MERGE BACK → full keep
    this.commit(); if (clip.modKeep) delete clip.modKeep[modId]; this._changed();
  }
  isDropped(clip, modId){ return this.modRanges(clip, modId).length === 0; }
  isEdited(clip, modId){ return !!(clip.modKeep && clip.modKeep[modId]); }
  modEditCount(clip){ return clip.modKeep ? Object.keys(clip.modKeep).length : 0; }
  // Call BEFORE a logical edit (atomic ops do this themselves; drags commit once
  // at gesture start). Coalesces a whole drag into a single undo step.
  commit(){ this.undoStack.push(this._snap()); if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack.length = 0; }
  undo(){ if (!this.undoStack.length) return false; this.redoStack.push(this._snap());
    this._restore(this.undoStack.pop()); return true; }
  redo(){ if (!this.redoStack.length) return false; this.undoStack.push(this._snap());
    this._restore(this.redoStack.pop()); return true; }
  canUndo(){ return this.undoStack.length > 0; }
  canRedo(){ return this.redoStack.length > 0; }

  _relayout(){
    let acc = 0;
    for (const c of this.clips){ c.start = acc; c.len = c.outFrame - c.inFrame; acc += c.len; }
    this.totalFrames = acc;
  }

  addClip({ ds, ep, epLabel, inFrame, outFrame, fps, srcMax }, index = null){
    this.commit();
    if (this.clips.length === 0 && fps) this.fps = fps;
    const clip = { id: _id++, ds, ep, epLabel: epLabel || ep,
      inFrame, outFrame, srcMax: srcMax ?? outFrame, modKeep: {}, modOffset: {},
      color: PALETTE[(_id) % PALETTE.length], selected: false };
    if (index == null) this.clips.push(clip); else this.clips.splice(index, 0, clip);
    this._relayout(); this._changed(); return clip;
  }

  resolve(globalFrame){
    if (!this.clips.length) return null;
    let acc = 0;
    for (let k = 0; k < this.clips.length; k++){
      const c = this.clips[k], len = c.outFrame - c.inFrame;
      if (globalFrame < acc + len)
        return { clip: c, index: k, sourceFrame: c.inFrame + (globalFrame - acc), localFrame: globalFrame - acc };
      acc += len;
    }
    const c = this.clips[this.clips.length - 1];
    return { clip: c, index: this.clips.length - 1, sourceFrame: c.outFrame - 1, localFrame: c.len - 1 };
  }

  clipAt(globalFrame){ return this.resolve(globalFrame)?.clip || null; }

  splitAt(globalFrame){
    const r = this.resolve(globalFrame);
    if (!r) return;
    const c = r.clip;
    const cut = r.sourceFrame;
    if (cut <= c.inFrame || cut >= c.outFrame) return;          // nothing to split
    this.commit();
    const right = { id: _id++, ds: c.ds, ep: c.ep, epLabel: c.epLabel,
      inFrame: cut, outFrame: c.outFrame, srcMax: c.srcMax,
      modKeep: JSON.parse(JSON.stringify(c.modKeep || {})), modOffset: { ...(c.modOffset || {}) },
      rate: c.rate, color: c.color, selected: false };
    const origMK = c.modKeep || {};
    const L = cut - c.inFrame;                          // new left length (clip-local cut)
    c.outFrame = cut;
    // remap per-modality keep-ranges to the two new clip boundaries
    c.modKeep = {}; right.modKeep = {};
    for (const mid in origMK){
      const lr = [], rr = [];
      for (const [a, b] of origMK[mid]){
        const la = Math.max(0, Math.min(a, L)), lb = Math.max(0, Math.min(b, L)); if (lb > la) lr.push([la, lb]);
        const ra = Math.max(0, a - L), rb = Math.max(0, b - L); if (rb > ra) rr.push([ra, rb]);
      }
      this._setMod(c, mid, lr); this._setMod(right, mid, rr);
    }
    this.clips.splice(r.index + 1, 0, right);
    this._relayout(); this._changed();
    return right;
  }

  trim(clip, side, newSourceFrame, srcMax){
    if (side === 'in') clip.inFrame = Math.max(0, Math.min(newSourceFrame, clip.outFrame - 1));
    else clip.outFrame = Math.min(srcMax, Math.max(newSourceFrame, clip.inFrame + 1));
    this._relayout(); this._changed();
  }

  remove(clip){
    const i = this.clips.indexOf(clip);
    if (i < 0) return;
    this.commit();
    this.clips.splice(i, 1);                                    // ripple (magnetic closes gap)
    this._relayout(); this._changed();
  }

  // Merge two adjacent clips that are contiguous in the SAME source.
  mergeAdjacent(a, b){
    const ia = this.clips.indexOf(a), ib = this.clips.indexOf(b);
    if (Math.abs(ia - ib) !== 1) return { ok: false, why: 'not adjacent' };
    const [first, second] = ia < ib ? [a, b] : [b, a];
    if (first.ds !== second.ds || first.ep !== second.ep || first.outFrame !== second.inFrame)
      return { ok: false, why: 'not contiguous in source' };
    this.commit();
    first.outFrame = second.outFrame;
    this.clips.splice(this.clips.indexOf(second), 1);
    this._relayout(); this._changed(); return { ok: true };
  }

  reorder(clip, toIndex){
    const i = this.clips.indexOf(clip); if (i < 0) return;
    this.clips.splice(i, 1);
    this.clips.splice(Math.max(0, Math.min(toIndex, this.clips.length)), 0, clip);
    this._relayout(); this._changed();
  }

  select(clip, additive = false){
    if (!additive) this.clips.forEach(c => c.selected = false);
    if (clip) clip.selected = !additive ? true : !clip.selected;
    bus.emit('selection', this.selected());
  }
  selected(){ return this.clips.filter(c => c.selected); }
  clearSelection(){ this.clips.forEach(c => c.selected = false); bus.emit('selection', []); }

  toEDL(name){
    return {
      edl_version: 1, name: name || this.name, fps: this.fps,
      segments: this.clips.map(c => ({ dataset: c.ds, episode: c.ep, start: c.inFrame, end: c.outFrame,
        modkeep: c.modKeep || {}, modoffset: c.modOffset || {}, rate: c.rate || 1 })),
      annotations: this.annotations.map(a => ({ text: a.text, start: a.start, end: a.end, kind: a.kind || 'language' })),
      ranges: this.ranges.map(r => ({ start: r.start, end: r.end, kind: r.kind })),
    };
  }

  // ONE single-segment EDL per clip, for PER-EPISODE export — robot demos are
  // atomic (no cross-episode merge), so each clip becomes its own output episode
  // carrying only its own edits (trim/modkeep/modoffset/rate) plus the ranges &
  // annotations that fall inside it, translated to that clip's local frames.
  perEpisodeEDLs(name){
    return this.clips.map((c) => {
      const lo = c.start, hi = c.start + c.len;
      const clamp = (v) => Math.max(0, Math.min(v - lo, c.len));
      const ranges = this.ranges.filter(r => r.end > lo && r.start < hi)
        .map(r => ({ start: clamp(r.start), end: clamp(r.end), kind: r.kind }))
        .filter(r => r.end > r.start);
      const annotations = this.annotations.filter(a => a.end > lo && a.start < hi)
        .map(a => ({ text: a.text, start: clamp(a.start), end: clamp(a.end), kind: a.kind || 'language' }))
        .filter(a => a.end > a.start);
      return {
        sourceDs: c.ds, source: c.ep, epLabel: c.epLabel,
        edited: !!(c.modEdits || this.modEditCount(c) || c.inFrame > 0 || c.outFrame < c.srcMax || c.rate || ranges.length),
        edl: {
          edl_version: 1, name: name || c.epLabel || this.name, fps: this.fps,
          segments: [{ dataset: c.ds, episode: c.ep, start: c.inFrame, end: c.outFrame,
            modkeep: c.modKeep || {}, modoffset: c.modOffset || {}, rate: c.rate || 1 }],
          annotations, ranges,
        },
      };
    });
  }

  _changed(){ bus.emit('sequenceChanged', this); }
}
