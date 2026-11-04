// The VIEWER RAIL — every "what do I see / where do I look" control, in one
// compact vertical column down the LEFT edge of the 3-D view.
//
// This is the web twin of the visionOS `leftRail`
// (apps/visionos/MabelTeleop/Features/Robot/RobotSpaceView.swift):
// discrete glass buttons stacked vertically, and anything with sub-options
// EXPANDS RIGHTWARD into a flyout instead of stealing a row from the top bar.
//
// Why a rail and not the old chip strip: the strip lived across the top of the
// viewer and grew with every layer we added, until it wrapped and collided with
// the teleop cluster. The top of the view is reserved for the ONE control
// decision (Drive · Task · Joint) — everything a layer does now hangs here.
//
// Grouping, top → bottom:
//   VIEW    Center · Fit · Follow      (camera verbs — where you look)
//   SCENE   Robot · Grid · TF▸ · …     (what is drawn)
// A separator, never a header: seven words down the left edge would shout over
// the robot. Names arrive on hover (.rail-lab), exactly like the module rail.
//
// Pure DOM + callbacks — no THREE, no links. The layer semantics stay in
// app.js (which owns `viewport`), so this file is node-testable: see
// tests/js/viewrail.test.js. `ICONS` and `RAIL_GROUPS` are data.

// Inline SVG (currentColor) — same 19px box as the module rail's glyphs, so the
// two rails read as one family even though they say different things.
const svg = (d, extra = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">${d}${extra}</svg>`;

export const ICONS = {
  center: svg('<circle cx="12" cy="12" r="3.2"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>'),
  fit: svg('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'),
  follow: svg('<path d="M15 8.5V6.2a1.2 1.2 0 0 0-1.2-1.2H4.2A1.2 1.2 0 0 0 3 6.2v11.6A1.2 1.2 0 0 0 4.2 19h9.6a1.2 1.2 0 0 0 1.2-1.2v-2.3z"/><path d="m15 12 6-3.6v11.2L15 16z"/>'),
  robot: svg('<rect x="5" y="8" width="14" height="11" rx="2.4"/><path d="M12 8V4.5M9.6 13h.01M14.4 13h.01"/><circle cx="12" cy="3.4" r="1.3"/>'),
  grid: svg('<path d="M3 3h18v18H3zM9 3v18M15 3v18M3 9h18M3 15h18"/>'),
  tf: svg('<path d="M5 19V5h14"/><path d="m5 19 6-6M5 19 9 8l6 5z" opacity=".55"/>'),
  lidar: svg('<circle cx="12" cy="12" r="2"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>'),
  cloud: svg('<path d="M6 8h.01M10 5h.01M15 7h.01M19 10h.01M4 13h.01M8 12h.01M13 11h.01M17 15h.01M6 18h.01M11 16h.01M16 19h.01M20 18h.01"/>'),
  mesh: svg('<path d="M12 3 3 8l9 5 9-5z"/><path d="m3 13 9 5 9-5" opacity=".55"/>'),
  map2d: svg('<path d="m9 4-6 2.5v13L9 17l6 2.5 6-2.5v-13L15 7z"/><path d="M9 4v13M15 7v12.5"/>'),
  path: svg('<path d="M4 20c6 0 4-8 8-8s2-8 8-8"/><circle cx="4" cy="20" r="1.6"/><circle cx="20" cy="4" r="1.6"/>'),
  navgoal: svg('<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.2"/><path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/>'),
  wrist: svg('<circle cx="12" cy="12" r="3"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5" opacity=".6"/>'),
  cam3d: svg('<path d="M4 7.5 12 4l8 3.5v9L12 20l-8-3.5z"/><circle cx="12" cy="12" r="2.3"/>'),
  traj: svg('<path d="M3 17c4.5 0 5-10 9.5-10S17 14 21 14"/><circle cx="3" cy="17" r="1.4"/><circle cx="21" cy="14" r="1.4"/>'),
  // AUDIO — voice to/from the robot (the web twin of the iOS speak/mute pair)
  mic: svg('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>'),
  speaker: svg('<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16.5 8.5a5 5 0 0 1 0 7"/>'),
  speakermute: svg('<path d="M4 9v6h4l5 4V5L8 9z"/><path d="m16 9 5 6M21 9l-5 6"/>'),
  // the collapsed rail's one door to every layer (short viewports)
  layers: svg('<path d="M12 3 3 7.5l9 4.5 9-4.5z"/><path d="m3 12 9 4.5L21 12M3 16.5 12 21l9-4.5"/>'),
};

// `menu` names a MENU_ITEMS key in app.js — the entries that EXPAND rightward.
// `layer` is the viewport layer key; `action` entries are camera verbs instead.
export const RAIL_GROUPS = [
  [
    { id: 'recenter', label: 'Center', glyph: 'center', action: 'recenter' },
    { id: 'fit', label: 'Fit', glyph: 'fit', action: 'fit' },
    { id: 'follow', label: 'Follow', glyph: 'follow', action: 'follow', toggle: true },
  ],
  [
    { id: 'robot', label: 'Robot', glyph: 'robot', layer: 'robot' },
    { id: 'grid', label: 'Grid', glyph: 'grid', layer: 'grid' },
    { id: 'tf', label: 'TF', glyph: 'tf', layer: 'tf', menu: 'tf' },
    { id: 'lidar', label: 'LiDAR', glyph: 'lidar', layer: 'lidar' },
    { id: 'cloud', label: 'Point Cloud', glyph: 'cloud', layer: 'cloud', menu: 'cloud' },
    { id: 'mesh', label: 'Mesh Map', glyph: 'mesh', layer: 'mesh' },
    { id: 'occupancy', label: '2D Map', glyph: 'map2d', layer: 'occupancy', menu: 'grid' },
    { id: 'path', label: 'Path', glyph: 'path', layer: 'path' },
    { id: 'navgoal', label: 'Nav Goal', glyph: 'navgoal', layer: 'navgoal' },
    { id: 'wrist', label: 'Wrist Targets', glyph: 'wrist', layer: 'wrist' },
    { id: 'frustum', label: 'Cam 3D', glyph: 'cam3d', layer: 'frustum', menu: 'cams' },
    { id: 'traj', label: 'Trajectory', glyph: 'traj', layer: 'traj', menu: 'traj' },
  ],
  // AUDIO — the real studio's speak + listen pair rides here, over the robot's
  // :9610 AudioHub. SIM-ONLY DIFFERENCE: the simulation has no microphone and
  // no speaker, so the group is empty rather than showing two buttons that can
  // only ever apologise. This is the ONE rail entry that differs from the real
  // Control Center; everything above it is identical by construction.
  [],
];

/** Every rail entry, flat — the rail must never silently drop one. */
export const RAIL_ITEMS = RAIL_GROUPS.flat();
export const railItemById = (id) => RAIL_ITEMS.find((i) => i.id === id) || null;

/** Layer keys the rail exposes, in rail order (node-tested against LAYERS). */
export const railLayerKeys = () => RAIL_ITEMS.filter((i) => i.layer).map((i) => i.layer);

/** Rail entries that expand rightward, by MENU_ITEMS key. */
export const railMenuKeys = () => RAIL_ITEMS.filter((i) => i.menu).map((i) => i.menu);

// The rail is twelve layers tall. That is fine on a desk and impossible on a
// landscape phone, where the 3-D view is ~180px high and the joysticks own the
// bottom corners — the full column simply lands on top of the left stick.
// Below the height it needs, the layers collapse into ONE "Overlays" button
// whose flyout carries the same list. That is not a fallback invented here: it
// is exactly what visionOS's rail does (Center · Fit · Follow · Overlays▸).
//
// The trigger is the 3-D VIEW's measured height, not the window's. Those are
// very different numbers: the telemetry bands and the timeline take ~500px, so
// a perfectly ordinary 1400×700 laptop leaves the view ~140px tall — shorter
// than a landscape phone's. A window-height media query would call that "desk".
//
// Between "fits in one column" and "hopeless" the rail WRAPS into extra
// columns rather than collapsing, so an ordinary 1080p desk still gets the full
// list the operator asked for. It only collapses when even a wrapped rail would
// be wider than the view can spare.
export const RAIL_BTN_H = 32;         // .rail-btn 30px + 2px gap
export const RAIL_PAD = 15;           // 2×4px padding + the 7px separator
export const RAIL_COL_W = 46;         // 38px button + 2×4px padding
export const RAIL_MAX_COLS = 2;       // beyond this the rail is a wall, not a rail

/** Columns the full rail needs at `h` px of view height (≥1). */
export const railCols = (h) => {
  const usable = Math.max(0, h - 20 /* top+bottom margin */) - RAIL_PAD;
  const perCol = Math.max(1, Math.floor(usable / RAIL_BTN_H));
  return Math.ceil(RAIL_ITEMS.length / perCol);
};

/** Does the FULL rail fit `h` px of view height, within the column budget? */
export const railFits = (h) => railCols(h) <= RAIL_MAX_COLS;

/** Rendered width of the rail, so the joysticks can be inset clear of it. */
export const railWidth = (cols) => cols * RAIL_COL_W;

/** The rail's entries for a given height mode — collapsed keeps the verbs. */
export const railFor = (compact) => compact
  ? [[...RAIL_GROUPS[0], { id: 'overlays', label: 'Overlays', glyph: 'layers',
    menu: 'overlays', action: 'overlays' }], RAIL_GROUPS[RAIL_GROUPS.length - 1]]
  : RAIL_GROUPS;

export class ViewRail {
  /**
   * @param {HTMLElement} el      the rail container (#vpRail)
   * @param {object} hooks
   *   isOn(item)      -> bool    current state of a layer / toggle action
   *   setLayer(k, on)            flip a layer
   *   action(id, on)             run a camera verb ('recenter'|'fit'|'follow')
   *   buildMenu(key, host)       render the flyout body for an expandable entry
   */
  constructor(el, hooks){
    this.el = el;
    this.hooks = hooks;
    this.btns = {};
    this.openId = null;
    this._flyout = null;
    el.classList.add('vp-rail');
    // Collapse on the VIEW's height (see RAIL_FULL_MIN_H) — it changes without
    // the window changing at all: opening the Trends band or dragging the
    // timeline splitter shortens the view under a static window.
    this.compact = false;
    this._ro = new ResizeObserver(() => this._fit());
    if (el.parentElement) this._ro.observe(el.parentElement);
    this.render();
    this._fit();

    // one outside-click closer for the whole rail (each flyout does not need
    // its own listener — that leaked one per open in the old chip menus)
    this._onDoc = (e) => {
      if (!this._flyout) return;
      if (this._flyout.contains(e.target) || this.el.contains(e.target)) return;
      this.closeFlyout();
    };
    document.addEventListener('pointerdown', this._onDoc);
    this._onKey = (e) => { if (e.key === 'Escape') this.closeFlyout(); };
    document.addEventListener('keydown', this._onKey);
  }

  /** Re-render only when the height mode actually flips (resize fires a lot). */
  _fit(){
    const host = this.el.parentElement;
    const h = host?.clientHeight || 0;
    if (!h) return;
    const compact = !railFits(h);
    if (compact !== this.compact) this.render(compact);
    // Publish how much of the left edge the rail is using, so the joysticks can
    // sit INBOARD of it instead of under it. The rail's width is not a constant
    // — it grows a column when the view gets short — so a hard-coded stick
    // offset would be right at one window size and wrong at the next.
    host.style.setProperty('--rail-w',
      railWidth(compact ? 1 : railCols(h)) + 'px');
  }

  /** (Re)build the rail for the given height mode. */
  render(compact = this.compact){
    this.closeFlyout();
    this.el.innerHTML = '';
    this.btns = {};
    this.compact = compact;
    this.el.classList.toggle('compact', this.compact);
    railFor(this.compact).forEach((group, gi) => {
      if (gi) this.el.appendChild(Object.assign(document.createElement('div'),
        { className: 'rail-sep' }));
      for (const it of group) this.el.appendChild(this._button(it));
    });
    this.sync();
  }

  _button(it){
    const b = document.createElement('button');
    b.className = 'rail-btn';
    b.dataset.id = it.id;
    // The label is a sibling, not a title=: a native tooltip on a 40px button
    // covers the next button down and fights the flyout.
    b.innerHTML = ICONS[it.glyph] +
      `<span class="rail-lab">${it.label}</span>` +
      (it.menu ? '<span class="rail-caret" aria-hidden="true"></span>' : '');
    b.setAttribute('aria-label', it.label);
    if (it.hold){
      const down = (e) => { e.preventDefault(); if (b.disabled) return;
        b.classList.add('on'); this.hooks.hold?.(it.id, true); };
      const up = () => { if (!b.classList.contains('on')) return;
        b.classList.remove('on'); this.hooks.hold?.(it.id, false); };
      b.onpointerdown = down;
      // release on EVERY way the press can end — a stuck-open mic is the failure
      // mode to design against, so err toward stopping.
      b.onpointerup = up; b.onpointerleave = up; b.onpointercancel = up;
      b.onblur = up;
    } else {
      b.onclick = (e) => this._click(it, b, e);
    }
    this.btns[it.id] = b;
    return b;
  }

  _click(it, b, e){
    // An expandable entry has TWO hit zones: the caret opens the sub-options,
    // the rest of the button toggles the layer itself. Clicking the caret must
    // never also flip the layer. "Overlays" is the exception — it owns no layer
    // of its own, so the whole button opens it.
    if (it.menu && (it.action === 'overlays' || e.target.closest('.rail-caret'))){
      this.toggleFlyout(it, b);
      return;
    }
    if (it.action){
      const on = it.toggle ? !this.hooks.isOn(it) : true;
      this.hooks.action(it.action, on);
      if (it.toggle) b.classList.toggle('on', on);
      return;
    }
    const now = !this.hooks.isOn(it);
    this.hooks.setLayer(it.layer, now);
    this.sync();
  }

  toggleFlyout(it, b){
    if (this.openId === it.id){ this.closeFlyout(); return; }
    this.closeFlyout();
    const f = document.createElement('div');
    f.className = 'rail-flyout';
    f.innerHTML = `<div class="rf-head">${it.label}</div><div class="rf-body"></div>`;
    document.body.appendChild(f);
    this.hooks.buildMenu(it.menu, f.querySelector('.rf-body'), () => this.sync());
    this._flyout = f;
    this.openId = it.id;
    b.classList.add('open');
    this.placeFlyout(b, f);
  }

  /** Anchor the flyout to the RIGHT of its button, clamped into the viewport. */
  placeFlyout(b, f){
    const r = b.getBoundingClientRect();
    f.style.left = Math.round(r.right + 8) + 'px';
    // clamp so a flyout opened near the bottom never runs off-screen
    const h = f.offsetHeight;
    const top = Math.min(Math.max(8, r.top - 6), Math.max(8, innerHeight - h - 8));
    f.style.top = Math.round(top) + 'px';
  }

  closeFlyout(){
    this._flyout?.remove();
    this._flyout = null;
    if (this.openId) this.btns[this.openId]?.classList.remove('open');
    this.openId = null;
  }

  /** Re-read every layer's live state (the viewport is the source of truth). */
  sync(){
    for (const it of RAIL_ITEMS){
      if (it.hold || (it.action && !it.toggle)) continue;
      this.btns[it.id]?.classList.toggle('on', !!this.hooks.isOn(it));
    }
  }

  destroy(){
    this.closeFlyout();
    this._ro?.disconnect();
    document.removeEventListener('pointerdown', this._onDoc);
    document.removeEventListener('keydown', this._onKey);
  }
}
