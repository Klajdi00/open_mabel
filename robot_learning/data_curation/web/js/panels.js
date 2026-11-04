// macOS-Sequoia–style window tiling for ADD-able windows. Drag a window's title
// bar and it suggests snap zones over the center workspace — halves, quarters,
// or full (maximize) — highlighting the target; release to snap. Every window is
// freely resizable (corner handle) and re-snappable. Dropping in the middle
// leaves it free-floating. The permanent UI (3D viewer, joint-state, timeline,
// library, right dock) is the backdrop these windows overlay.
let _pid = 1;

// ── snap geometry (pure, unit-testable — see js/__test_snap.mjs) ─────────────
// The whole snapping RULE lives here. Windows snap to a fixed grid of slots —
// two full-height side columns (left/right), a bottom band, four corner quarters,
// and full/maximize — and the slots NEGOTIATE the shared corners so they never
// overlap: a side column shrinks UP to sit above an occupied bottom band, and a
// bottom band shrinks IN to avoid an occupied side column. Deterministic: each
// slot has a fixed position (clipped by occupancy), so dropping near a border
// always lands the same predictable place.
export const SIDE_FRAC = 0.22, BOTTOM_FRAC = 0.30, SIDE_MAX = 380, BOTTOM_MAX = 320;

// Maximal sub-range of [lo,hi] that CONTAINS `ref` and avoids every [s,e] blocker
// interval. Returns null if `ref` sits inside a blocker (slot unavailable) or the
// surviving range is too thin. This is the shrink primitive for both axes.
function freeSpan(lo, hi, ref, blockers){
  let a = lo, b = hi;
  for (const [s, e] of blockers){
    if (e <= ref + 0.5) a = Math.max(a, e);          // blocker fully below ref → floor rises
    else if (s >= ref - 0.5) b = Math.min(b, s);     // blocker fully above ref → ceiling drops
    else return null;                                // ref inside blocker → slot taken
  }
  return (b - a > 44) ? [a, b] : null;
}

// Compute every candidate slot rect for a workspace `ws` given occupancy `occ`
// (rects of already-snapped windows + reserved docks). Each slot is clipped so it
// never overlaps occupancy; unavailable slots come back null.
export function computeZones(ws, occ){
  const SW = Math.min(ws.w * SIDE_FRAC, SIDE_MAX);
  const SH = Math.min(ws.h * BOTTOM_FRAC, BOTTOM_MAX);
  const midY = ws.y + ws.h / 2, midX = ws.x + ws.w / 2;
  const half = ws.y + ws.h / 2;
  // vertical blockers within a side column x-range → the side shrinks in Y
  const colV = (x0) => occ.filter(o => o.right > x0 + 2 && o.left < x0 + SW - 2)
                          .map(o => [o.top, o.bottom]);
  // horizontal blockers within the bottom band y-range → the band shrinks in X
  const bandY = ws.y + ws.h - SH;
  const bandH = occ.filter(o => o.bottom > bandY + 2 && o.top < ws.y + ws.h - 2)
                   .map(o => [o.left, o.right]);
  const side = (x0, ref) => { const s = freeSpan(ws.y, ws.y + ws.h, ref, colV(x0));
    return s ? { x: x0, y: s[0], w: SW, h: s[1] - s[0] } : null; };
  const corner = (x0, top) => { const lo = top ? ws.y : half, hi = top ? half : ws.y + ws.h;
    const ref = top ? ws.y + ws.h * 0.25 : ws.y + ws.h * 0.75;
    const s = freeSpan(lo, hi, ref, colV(x0));
    return s ? { x: x0, y: s[0], w: SW, h: s[1] - s[0] } : null; };
  const bottom = () => { const s = freeSpan(ws.x, ws.x + ws.w, midX, bandH);
    return s ? { x: s[0], y: bandY, w: s[1] - s[0], h: SH } : null; };
  const rx = ws.x + ws.w - SW;
  return {
    full: { x: ws.x, y: ws.y, w: ws.w, h: ws.h },
    left: side(ws.x, midY), right: side(rx, midY),
    tl: corner(ws.x, true), bl: corner(ws.x, false),
    tr: corner(rx, true), br: corner(rx, false),
    bottom: bottom(),
  };
}

// Pick which slot the cursor targets: must be near a border (macOS-style); corner
// bands select quarters, otherwise the side/bottom/full slot. Falls back to the
// full-height side (or full) when a quarter is unavailable.
export function pickZone(Z, ws, cx, cy, edge = 64){
  const nearL = cx <= ws.x + edge, nearR = cx >= ws.x + ws.w - edge;
  const nearT = cy <= ws.y + edge, nearB = cy >= ws.y + ws.h - edge;
  if (!(nearL || nearR || nearT || nearB)) return null;
  const cb = 0.28;
  const top = (cy - ws.y) / ws.h < cb, bot = (ws.y + ws.h - cy) / ws.h < cb;
  const left = (cx - ws.x) / ws.w < cb, right = (ws.x + ws.w - cx) / ws.w < cb;
  if (nearL) return top ? (Z.tl || Z.left) : bot ? (Z.bl || Z.left) : Z.left;
  if (nearR) return top ? (Z.tr || Z.right) : bot ? (Z.br || Z.right) : Z.right;
  if (nearB) return left ? (Z.bl || Z.bottom) : right ? (Z.br || Z.bottom) : Z.bottom;
  if (nearT) return left ? (Z.tl || Z.full) : right ? (Z.tr || Z.full) : Z.full;
  return null;
}

export class PanelManager {
  constructor(host){
    this.host = host; this.panels = []; this.factories = {}; this._last = null;
    this.reserved = [];                       // permanent docked views to never overlap (e.g. the camera dock)
    this.ws = document.createElement('div'); this.ws.className = 'fp-workspace'; host.appendChild(this.ws);
    this.hintEl = document.createElement('div'); this.hintEl.className = 'fp-snaphint'; host.appendChild(this.hintEl);
    window.addEventListener('resize', () => this._reflow());
    window.addEventListener('studio-resize', () => this._reflow());
  }
  // register a permanent on-screen element that floating windows must tile AROUND
  reserve(el){ if (el) this.reserved.push(el); }

  register(type, label, make){ this.factories[type] = { label, make }; }
  list(){ return Object.entries(this.factories).map(([type, f]) => ({ type, label: f.label })); }

  add(type, preset){
    const f = this.factories[type]; if (!f) return null;
    const el = document.createElement('div'); el.className = 'fpanel';
    el.innerHTML = `<div class="fp-bar"><span class="fp-title">${f.label}</span>
      <button class="fp-close" title="close">✕</button></div>
      <div class="fp-body"></div>
      <div class="fp-rz n"></div><div class="fp-rz s"></div><div class="fp-rz e"></div><div class="fp-rz w"></div>
      <div class="fp-rz se"></div><div class="fp-rz sw"></div><div class="fp-rz ne"></div><div class="fp-rz nw"></div>`;
    this.host.appendChild(el);
    const panel = { id: _pid++, type, el, body: el.querySelector('.fp-body'), inst: null, zone: null, preset };
    panel.inst = f.make(panel);
    this._wire(panel); this.panels.push(panel);
    // open as a SMALL FREE-FLOATING window near the top-left of the free area
    // (zone stays null so it isn't tiled); cascade repeated adds. Drag it to a
    // border to snap. Launchers/cameras override this with their own snap.
    const fr = this._freeRect(panel);
    const w = Math.min(440, fr.w * 0.34), h = Math.min(340, fr.h * 0.46);
    const k = (this.panels.length % 6) * 26;
    panel.el.style.left = (fr.x + 24 + k) + 'px'; panel.el.style.top = (fr.y + 18 + k) + 'px';
    panel.el.style.width = w + 'px'; panel.el.style.height = h + 'px';
    panel.zone = null; panel.inst?.resize?.();
    this._raise(panel);
    if (this._last) panel.inst?.update?.(this._last.entry, this._last.frame);
    return panel;
  }
  // spawn a panel at an explicit rect {x,y,w,h} (used by layout presets)
  addAt(type, rect, preset){ const p = this.add(type, preset); if (p && rect) this._snap(p, rect); return p; }
  // Dock a panel to a side (left|right) and re-tile EVERY panel on that side so
  // they share the column without overlapping (IMU above 2D-map, etc.). The
  // column auto-shrinks in height to whatever free space exists (e.g. above the
  // cameras dock), and each panel gets an equal slice.
  dockToSide(panel, side){ panel.dockSide = side; this._retileSide(side); this._raise(panel); }
  _retileSide(side){
    const items = this.panels.filter(p => p.dockSide === side);
    if (!items.length) return;
    const col = this._zones(this._freeRect(new Set(items)))[side];   // the 22% side strip, full free height
    const n = items.length;
    items.forEach((p, i) => this._snap(p, { x: col.x, y: col.y + (col.h * i) / n, w: col.w, h: col.h / n }));
  }
  clearAll(){ [...this.panels].forEach(p => this.remove(p)); }
  wsRect(){ return this._wsRect(); }

  _wire(panel){
    const el = panel.el;
    el.querySelector('.fp-bar').addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; this._drag(e, panel); });
    el.querySelector('.fp-close').onclick = (e) => { e.stopPropagation(); this.remove(panel); };
    el.querySelectorAll('.fp-rz').forEach(h => h.addEventListener('pointerdown', (e) => {
      const dir = [...h.classList].find(c => c !== 'fp-rz'); this._resize(e, panel, dir); }));
    el.addEventListener('pointerdown', () => this._raise(panel));
  }
  _raise(panel){ let z = 42; this.panels.forEach(p => p.el.style.zIndex = p === panel ? 47 : z++); }

  // Workspace rect, with the top pushed below the 3D-view toggle chips and the
  // cam-projection controls (they wrap to multiple rows), so snapped windows
  // never cover those buttons.
  _wsRect(){
    const r = this.ws.getBoundingClientRect();
    let top = r.top;
    for (const sel of ['#vpToggles', '#vpCamCtl', '#vpSubtitle']){
      const e = document.querySelector(sel);
      if (e && e.offsetParent !== null){ const b = e.getBoundingClientRect().bottom + 8; if (b > top) top = b; }
    }
    return { x: r.left, y: top, w: r.width, h: Math.max(120, r.bottom - top) };
  }
  // The largest free rectangle of the workspace — the workspace minus any
  // already-snapped windows (full-width / full-height bands are carved out
  // iteratively). So with the cameras docked along the bottom, the free rect is
  // the area ABOVE them, and every snap zone is computed inside it: a window
  // never overlaps an existing one and "right" fills the full free height.
  _freeRect(exclude){
    const ex = exclude instanceof Set ? exclude : (exclude ? new Set([exclude]) : new Set());
    let f = this._wsRect();
    // 1) carve the left/right dock COLUMNS by the union of their members, so a
    //    stacked dock (IMU above 2D-map) is treated as one full-height band and
    //    a bottom panel never overlaps it.
    const lefts = this.panels.filter(p => !ex.has(p) && p.zone && p.dockSide === 'left');
    const rights = this.panels.filter(p => !ex.has(p) && p.zone && p.dockSide === 'right');
    if (lefts.length){ const r = Math.max(...lefts.map(p => p.el.getBoundingClientRect().right));
      if (r > f.x && r < f.x + f.w) f = { x: r, y: f.y, w: f.x + f.w - r, h: f.h }; }
    if (rights.length){ const l = Math.min(...rights.map(p => p.el.getBoundingClientRect().left));
      if (l > f.x && l < f.x + f.w) f = { x: f.x, y: f.y, w: l - f.x, h: f.h }; }
    // 2) carve full-span bands from the remaining (non-dock) windows + reserved
    const occ = this.panels.filter(p => !ex.has(p) && p.zone && !p.dockSide).map(p => p.el.getBoundingClientRect());
    for (const el of this.reserved){
      if (el && el.offsetParent !== null){ const r = el.getBoundingClientRect(); if (r.width > 4 && r.height > 4) occ.push(r); }
    }
    for (let guard = 0; guard < 12; guard++){
      let changed = false;
      for (const o of occ){
        const ix = Math.max(f.x, o.left), iy = Math.max(f.y, o.top);
        const ax = Math.min(f.x + f.w, o.right), ay = Math.min(f.y + f.h, o.bottom);
        if (ax - ix <= 1 || ay - iy <= 1) continue;             // no real overlap
        const ex = Math.max(8, f.w * 0.05), ey = Math.max(8, f.h * 0.05);
        const spansW = ix <= f.x + ex && ax >= f.x + f.w - ex;
        const spansH = iy <= f.y + ey && ay >= f.y + f.h - ey;
        if (spansW && iy <= f.y + ey){ f = { x: f.x, y: ay, w: f.w, h: f.y + f.h - ay }; changed = true; break; }      // top band
        if (spansW && ay >= f.y + f.h - ey){ f = { x: f.x, y: f.y, w: f.w, h: iy - f.y }; changed = true; break; }     // bottom band
        if (spansH && ix <= f.x + ex){ f = { x: ax, y: f.y, w: f.x + f.w - ax, h: f.h }; changed = true; break; }      // left band
        if (spansH && ax >= f.x + f.w - ex){ f = { x: f.x, y: f.y, w: ix - f.x, h: f.h }; changed = true; break; }     // right band
      }
      if (!changed) break;
    }
    return f;
  }
  _zones(w){
    w = w || this._freeRect();
    const ws = this._wsRect(), hy = w.h / 2;
    const sw = Math.min(w.w * 0.5, ws.w * 0.20);      // side dock ≤ 20% of the viewer width
    const sh = Math.min(w.h * 0.5, ws.h * 0.25), by = w.y + w.h - sh;   // bottom dock ≤ 25% of height
    return {
      full: { x: w.x, y: w.y, w: w.w, h: w.h },
      left: { x: w.x, y: w.y, w: sw, h: w.h }, right: { x: w.x + w.w - sw, y: w.y, w: sw, h: w.h },
      top: { x: w.x, y: w.y, w: w.w, h: hy }, bottom: { x: w.x, y: by, w: w.w, h: sh },
      tl: { x: w.x, y: w.y, w: sw, h: hy }, tr: { x: w.x + w.w - sw, y: w.y, w: sw, h: hy },
      bl: { x: w.x, y: by, w: sw, h: sh }, br: { x: w.x + w.w - sw, y: by, w: sw, h: sh },
    };
  }
  // Occupancy = every already-SNAPPED/docked window (free-floaters don't tile) +
  // reserved permanent docks (the camera band). These are the obstacles the snap
  // grid negotiates around. NOTE: .fpanel is position:fixed so offsetParent is
  // always null — use the bounding rect for visibility.
  _occRects(exclude){
    const ex = exclude instanceof Set ? exclude : (exclude ? new Set([exclude]) : new Set());
    const occ = [];
    for (const p of this.panels){
      if (ex.has(p) || !p.el || (!p.zone && !p.dockSide)) continue;
      const r = p.el.getBoundingClientRect(); if (r.width > 2 && r.height > 2) occ.push(r);
    }
    for (const el of this.reserved){
      if (!el) continue; const r = el.getBoundingClientRect(); if (r.width > 2 && r.height > 2) occ.push(r);
    }
    return occ;
  }
  // Snap target under the cursor: the predefined grid slot (side/bottom/corner/
  // full), clipped so it never overlaps an existing tile — all in computeZones/
  // pickZone (pure + unit-tested). Returns null when the cursor is over a tile or
  // not near a border.
  _zoneAt(cx, cy, exclude){
    const ws = this._wsRect();
    if (cx < ws.x || cx > ws.x + ws.w || cy < ws.y || cy > ws.y + ws.h) return null;
    const occ = this._occRects(exclude);
    for (const o of occ) if (cx > o.left && cx < o.right && cy > o.top && cy < o.bottom) return null;
    return pickZone(computeZones(ws, occ), ws, cx, cy);
  }
  _showHint(rect){ const h = this.hintEl; if (!rect){ h.style.display = 'none'; return; }
    h.style.display = 'block'; h.style.left = rect.x + 'px'; h.style.top = rect.y + 'px'; h.style.width = rect.w + 'px'; h.style.height = rect.h + 'px'; }

  _snap(panel, rect){
    panel.zone = rect;
    panel.el.style.left = rect.x + 'px'; panel.el.style.top = rect.y + 'px';
    panel.el.style.width = rect.w + 'px'; panel.el.style.height = rect.h + 'px';
    panel.inst?.resize?.();
  }
  // re-apply zones as fractions on window resize (snapped panels track the layout)
  _reflow(){ /* zones recompute live; snapped rects are absolute, so re-snap to same fractional zone */ }

  _drag(e, panel){
    e.preventDefault(); this._raise(panel); panel.zone = null; panel.dockSide = null;
    const sx = e.clientX, sy = e.clientY, r0 = panel.el.getBoundingClientRect();
    const move = (ev) => {
      panel.el.style.left = (r0.left + ev.clientX - sx) + 'px';
      panel.el.style.top = Math.max(46, r0.top + ev.clientY - sy) + 'px';
      this._showHint(this._zoneAt(ev.clientX, ev.clientY, panel));
    };
    const up = (ev) => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      const z = this._zoneAt(ev.clientX, ev.clientY, panel); this._showHint(null);
      if (z) this._snap(panel, z); };               // else: stays where dropped (free float)
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  // resize from any edge/corner (like a real window), dir ∈ n,s,e,w,ne,nw,se,sw
  _resize(e, panel, dir){
    e.preventDefault(); e.stopPropagation(); panel.zone = null; panel.dockSide = null; this._raise(panel);
    const sx = e.clientX, sy = e.clientY, r = panel.el.getBoundingClientRect();
    const E = dir.includes('e'), Wd = dir.includes('w'), S = dir.includes('s'), N = dir.includes('n');
    const move = (ev) => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      let left = r.left, top = r.top, w = r.width, h = r.height;
      if (E) w = r.width + dx;
      if (Wd){ w = r.width - dx; left = r.left + dx; }
      if (S) h = r.height + dy;
      if (N){ h = r.height - dy; top = r.top + dy; }
      if (w < 220){ if (Wd) left = r.left + (r.width - 220); w = 220; }
      if (h < 130){ if (N) top = r.top + (r.height - 130); h = 130; }
      panel.el.style.left = left + 'px'; panel.el.style.top = Math.max(46, top) + 'px';
      panel.el.style.width = w + 'px'; panel.el.style.height = h + 'px';
      panel.inst?.resize?.();
    };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }

  remove(panel){ const side = panel.dockSide; panel.inst?.destroy?.(); panel.el.remove();
    this.panels = this.panels.filter(p => p !== panel); if (side) this._retileSide(side); }
  update(entry, frame){ this._last = { entry, frame }; for (const p of this.panels) p.inst?.update?.(entry, frame); }
}
