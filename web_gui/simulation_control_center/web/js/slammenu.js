// The SLAM menu — mapping lives in ONE top-right dropdown, not as loose chips
// over the robot: Live SLAM · the saved-room library · Reset.
//
// SAVED MAPS COME FROM THE GATEWAY, NOT FROM THIS STUDIO'S BACKEND.
// The map store is `server/map_library.py` behind the teleop wire
// (`list_maps` / `load_map` / `save_map` / `delete_map` — server/wire_protocol.py),
// which is the SAME plane the iOS and Vision Pro apps already speak
// (apps/ios/.../Services/WebSocketLink.swift sends list_maps/load_map).
// That is deliberate and matches the repo's map invariant —
//   robot → hardware_bridge → ROS → SLAM → server → devices
// — so a room saved from the phone shows up here, and one saved here shows up
// on the phone. Do NOT reintroduce a control-center-local map store: this
// studio's :8094 backend is only for planes the bridge does not expose, and a
// store here would be invisible to every other client.
//
// Live-SLAM pause/reset ARE this backend's job (/api/slam/pause | /api/slam/reset
// drive the local rclpy service calls in server/streams.py) — they act on the
// running SLAM node, not on the stored rooms.

export class SlamMenu {
  /**
   * @param {HTMLElement} btn   the top-bar button
   * @param {HTMLElement} menu  its dropdown container
   * @param {object} link       RobotLink (the teleop wire)
   * @param {function} toast
   * @param {object} hooks      onFreeze(frozen) · onClear() — the 3-D map view
   */
  constructor(btn, menu, link, toast, hooks = {}){
    this.btn = btn;
    this.menu = menu;
    this.link = link;
    this.toast = toast || (() => {});
    this.hooks = hooks;
    this.live = true;             // SLAM runs by default (the bridge starts it)
    this.maps = [];
    this.loaded = null;           // name of the saved room being shown, if any
    this.busy = false;

    btn.onclick = (e) => { e.stopPropagation(); this.toggle(); };
    document.addEventListener('pointerdown', (e) => {
      if (!menu.classList.contains('hidden') &&
          !menu.contains(e.target) && !btn.contains(e.target)) this.close();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.close();
    });
    this.syncBtn();
  }

  /** The gateway pushes `map_list` on every mutation — one refresh path. */
  onMapList(msg){
    this.maps = Array.isArray(msg?.maps) ? msg.maps : [];
    this.loaded = msg?.loaded || null;
    if (!this.menu.classList.contains('hidden')) this.render();
    this.syncBtn();
  }

  toggle(){
    if (this.menu.classList.contains('hidden')){
      this.link.send('list_maps', {});     // always open on fresh truth
      this.render();
      this.menu.classList.remove('hidden');
      this.btn.classList.add('open');
    } else this.close();
  }

  close(){
    this.menu.classList.add('hidden');
    this.btn.classList.remove('open');
  }

  syncBtn(){
    // The button says what the robot's mapping is DOING — live, paused, or
    // showing a saved room — because that is the thing an operator misreads.
    const state = this.loaded ? this.loaded : this.live ? 'live' : 'paused';
    this.btn.classList.toggle('on', this.live && !this.loaded);
    this.btn.classList.toggle('warn', !this.live || !!this.loaded);
    this.btn.innerHTML =
      `<span class="slam-dot"></span>SLAM<span class="slam-state mono">${state}</span>`;
    this.btn.title = this.loaded
      ? `Showing the saved room "${this.loaded}" — live mapping is paused`
      : this.live ? 'SLAM is mapping live' : 'SLAM is paused — the map is frozen';
  }

  render(){
    const m = this.menu;
    m.innerHTML = '';
    // ── live mapping ──
    m.appendChild(this._head('Mapping'));
    const liveRow = this._row(this.live ? 'Live SLAM · on' : 'Live SLAM · paused',
      this.live, () => this.setLive(!this.live));
    liveRow.classList.add('rf-live');
    m.appendChild(liveRow);

    // ── the saved-room library (shared with iOS / Vision Pro) ──
    m.appendChild(this._head('Saved maps'));
    const liveItem = this._row('Live map', !this.loaded, () => this.loadMap(''));
    m.appendChild(liveItem);
    if (!this.maps.length){
      m.appendChild(Object.assign(document.createElement('div'),
        { className: 'rf-empty', textContent: 'no saved maps' }));
    }
    for (const mp of this.maps){
      const r = this._row(mp.name, this.loaded === mp.name, () => this.loadMap(mp.name));
      const meta = [];
      if (mp.has_mesh) meta.push('3D');
      if (mp.has_reloc) meta.push('reloc');
      if (meta.length){
        r.appendChild(Object.assign(document.createElement('span'),
          { className: 'rf-meta mono', textContent: meta.join(' · ') }));
      }
      const del = document.createElement('button');
      del.className = 'rf-del';
      del.textContent = '×';
      del.title = `Delete "${mp.name}"`;
      del.onclick = (e) => { e.stopPropagation(); this.deleteMap(mp.name); };
      r.appendChild(del);
      m.appendChild(r);
    }

    // ── actions ──
    m.appendChild(this._head('Actions'));
    m.appendChild(this._action('⤓ Save current map', () => this.saveMap()));
    const reset = this._action('↻ Reset SLAM', () => this.resetMap());
    reset.classList.add('rf-danger');
    m.appendChild(reset);
  }

  _head(t){
    return Object.assign(document.createElement('div'),
      { className: 'rf-sec', textContent: t });
  }

  _row(label, on, cb){
    const b = document.createElement('button');
    b.className = 'vp-mi' + (on ? ' on' : '');
    b.innerHTML = `<span class="vmi-dot"></span><span class="rf-name">${label}</span>`;
    b.onclick = cb;
    return b;
  }

  _action(label, cb){
    const b = document.createElement('button');
    b.className = 'vp-mi rf-act';
    b.textContent = label;
    b.onclick = cb;
    return b;
  }

  setLive(on){
    this.live = on;
    // Pausing freezes the map; re-enabling restarts mapping FRESH (clearing +
    // relaunching the stacks takes seconds — hence the spinner) — the same
    // contract the old SLAM chip carried.
    this.busy = true;
    this.btn.classList.add('loading');
    this.syncBtn();
    if (!on){
      this.hooks.onFreeze?.(true);
      fetch('/api/slam/pause', { method: 'POST' })
        .then(() => this.toast('SLAM paused — map frozen'))
        .catch(() => this.toast('SLAM pause failed', true))
        .finally(() => this._done());
      return;
    }
    fetch('/api/slam/reset', { method: 'POST' })
      .then((r) => r.json())
      .then((j) => {
        this.hooks.onClear?.();
        this.hooks.onFreeze?.(false);
        this.toast(j.ok ? 'SLAM restarted — mapping fresh'
          : 'SLAM resumed (no reset service reachable)', !j.ok);
      })
      .catch(() => {
        this.hooks.onFreeze?.(false);
        this.toast('SLAM resume failed', true);
      })
      .finally(() => this._done());
  }

  _done(){
    this.busy = false;
    this.btn.classList.remove('loading');
    this.syncBtn();
    if (!this.menu.classList.contains('hidden')) this.render();
  }

  loadMap(name){
    // "" / "live" → back to the live SLAM map (the gateway's contract).
    this.link.send('load_map', { name });
    this.loaded = name || null;
    this.toast(name ? `loading saved room "${name}"` : 'back to the live map');
    this.syncBtn();
    this.close();
  }

  saveMap(){
    const name = prompt('Save the current map as:', '');
    if (name == null) return;
    const clean = name.trim();
    // Mirror the gateway's validation so a bad name fails HERE with a readable
    // message instead of silently no-op'ing on the robot.
    if (!/^[A-Za-z0-9_-]+$/.test(clean)){
      this.toast('map name: letters, numbers, - and _ only', true);
      return;
    }
    this.link.send('save_map', { name: clean });
    this.toast(`saving map "${clean}" …`);
    this.close();
  }

  deleteMap(name){
    if (!confirm(`Delete the saved map "${name}"? This cannot be undone.`)) return;
    this.link.send('delete_map', { name });
    this.toast(`deleted "${name}"`);
  }

  resetMap(){
    if (!confirm('Reset SLAM? The current map is cleared and mapping restarts from scratch.')) return;
    this.live = true;
    this.loaded = null;
    this.setLive(true);
    this.close();
  }
}
