// Right dock: Status pane (iOS-settings-style rows fed by robot_state + hello
// + the studio's own control state) and Data pane (episodes-by-task + rosbag
// sessions collected on the robot host).
import * as bus from './bus.js';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtBytes = (b) => !b ? '—' : b > 1e9 ? (b / 1e9).toFixed(1) + ' GB'
  : b > 1e6 ? (b / 1e6).toFixed(0) + ' MB' : (b / 1e3).toFixed(0) + ' kB';

export class StatusPanel {
  constructor(statusEl, dataEl, ctl){
    this.el = statusEl;
    this.dataEl = dataEl;
    this.ctl = ctl;                 // the app's control-state object
    this.state = null; this.hello = null; this.latency = null;
    this.recordings = null;
    this.schema = null; this.tele = null;   // for the mock/no-gateway fallback
    bus.on('robotState', (s) => { this.state = s; this.render(); });
    bus.on('hello', (h) => { this.hello = h; this.render(); });
    bus.on('latency', (l) => { this.latency = l; });
    bus.on('telemetrySchema', (s) => { this.schema = s; });
    // telemetry is 15 Hz — keep the latest but only repaint the panel ~1 Hz
    bus.on('telemetry', (t) => {
      this.tele = t;
      const now = Date.now();
      if (now - (this._teleAt || 0) > 1000){ this._teleAt = now; this.render(); }
    });
    bus.on('ctl', () => this.render());
    bus.on('link', () => this.render());
    this.render();
    this.refreshData();
    setInterval(() => this.refreshData(), 15000);
  }

  // ── hardware connection roster ────────────────────────────────────────────
  // Preferred source is robot_state.modules (the gateway's module_status.py:
  // base/torso/lift/neck/arms/hands/cameras/lidar/speaker/mic, each with ok +
  // detail). When that is absent (mock / no gateway) fall back to the studio's
  // own /ws/telemetry: a subsystem is live if any of its joints reported, and
  // the LiDAR is live if a scan arrived.
  _hardwareRows(){
    const mods = this.state?.modules;
    if (Array.isArray(mods) && mods.length) return this._rowsFromModules(mods);
    return this._rowsFromTelemetry();
  }

  // A hardware chip is TRI-state (iOS StatusDetailPanel parity): ok / down /
  // wait. "We have not heard from it yet" is NOT the same fault as "it is
  // broken", and rendering both as a red dot sends operators to go power-cycle
  // a subsystem that was merely still booting. `st` is 'ok' | 'bad' | 'wait'.
  _chip(label, st, detail){
    const title = detail ? ` title="${esc(detail)}"` : '';
    return `<span class="st-chip ${st}"${title}><i></i>${esc(label)}</span>`;
  }

  _chipsBlock(title, chips){
    if (!chips.length) return '';
    return `<div class="np-sec">${esc(title)}</div>` +
      `<div class="st-chips">${chips.join('')}</div>`;
  }

  _rowsFromModules(mods){
    const AUDIO = new Set(['speaker', 'mic']);
    const by = { body: [], sensors: [], audio: [] };
    for (const m of mods){
      const g = AUDIO.has(m.id) ? 'audio' : (m.group === 'sensors' ? 'sensors' : 'body');
      // the gateway reports ok as a tri-state too: null/undefined = not yet known
      const st = m.ok == null ? 'wait' : (m.ok ? 'ok' : 'bad');
      by[g].push(this._chip(m.label, st, m.detail));
    }
    return this._chipsBlock('Hardware', by.body)
      + this._chipsBlock('Sensors', by.sensors)
      + this._chipsBlock('Audio', by.audio);
  }

  _rowsFromTelemetry(){
    const t = this.tele, sch = this.schema;
    if (!t || !sch?.modules?.length)
      return `<div class="np-sec">Hardware</div>` +
        `<div class="np-empty">waiting for telemetry…</div>`;
    const idx = {};
    sch.names.forEach((n, i) => { idx[n] = i; });
    // any joint reporting a finite position = the subsystem is talking
    const live = (joints) => joints.some((n) => Number.isFinite(t.p?.[idx[n]]));
    const label = { left_arm: 'L Arm', right_arm: 'R Arm', left_hand: 'L Hand',
      right_hand: 'R Hand', neck: 'Neck', torso: 'Torso', lift: 'Lift',
      lift_torso: 'Lift/Torso', base: 'Base' };
    const body = (sch.modules || []).map((m) =>
      this._chip(label[m.key] || m.label || m.key,
        live(m.joints || []) ? 'ok' : 'bad'));
    // the studio taps /scan into telemetry, so the LiDAR is honestly probeable
    const sensors = [this._chip('LiDAR',
      Array.isArray(t.scan) && t.scan.length > 0 ? 'ok' : 'wait')];
    return this._chipsBlock('Hardware', body) + this._chipsBlock('Sensors', sensors);
  }

  async refreshData(){
    try { this.recordings = await fetch('/api/recordings').then(r => r.json()); }
    catch { this.recordings = null; }
    this.renderData();
  }

  _row(label, value, tint = ''){
    return `<div class="st-row"><span>${esc(label)}</span><b class="${tint}">${value}</b></div>`;
  }

  // A KPI tile: big value over a small caption. Two per row, so the four
  // numbers an operator actually scans (link, latency, battery, torque) cost
  // two rows instead of eight.
  _kpi(caption, value, tint = ''){
    return `<div class="st-kpi ${tint}"><b>${value}</b><span>${esc(caption)}</span></div>`;
  }

  // A dense label→value pair for the 2-column Control grid.
  _pair(label, value, tint = ''){
    return `<div class="st-pair"><span>${esc(label)}</span><b class="${tint}">${value}</b></div>`;
  }

  render(){
    const s = this.state, h = this.hello, c = this.ctl;
    const connected = !!s;

    // ── the vitals strip: the four numbers worth a glance ────────────────────
    // (iOS GlobalStatusBar parity — link · latency · battery · torque.) Missing
    // channels print "—" rather than a plausible zero: on the sim there IS no
    // battery, and a fake 0% reads as a flat pack.
    const lat = this.latency;
    const batt = s?.battery?.percentage;
    let out = `<div class="st-kpis">`;
    out += this._kpi('link', connected ? 'LIVE' : 'DOWN', connected ? 'ok' : 'bad');
    out += this._kpi('latency', lat != null ? `${lat.toFixed(0)}<u>ms</u>` : '—',
      lat == null ? '' : lat < 60 ? 'ok' : lat < 150 ? 'warn' : 'bad');
    out += this._kpi('battery', batt != null ? `${(batt * 100).toFixed(0)}<u>%</u>` : '—',
      batt == null ? '' : batt > 0.3 ? 'ok' : 'warn');
    out += this._kpi('torque', c.estop ? 'STOP' : c.torque ? 'ON' : 'LIMP',
      c.estop ? 'bad' : c.torque ? 'ok' : 'warn');
    out += `</div>`;

    // ── link detail, only what varies ────────────────────────────────────────
    const linkPairs = [];
    if (h){
      linkPairs.push(this._pair('Server', esc(h.name || h.server || '?')));
      linkPairs.push(this._pair('Driving', h.kind === 'sim' ? 'Sim' : 'Real'));
      linkPairs.push(this._pair('Transport', esc(s?.server?.transport || h.transport || 'LAN')));
    }
    if (s?.driver)
      linkPairs.push(this._pair('Driver', s.driver.you ? 'You' : `other · ${s.driver.clients ?? '?'}`,
        s.driver.you ? 'ok' : 'warn'));
    // The Connection grid is COLLAPSED BY DEFAULT. Server name / transport /
    // driver are identity, not state: you read them once when something is
    // confusing, never at a glance. Leaving them expanded cost ~110px of a
    // ~700px dock, which pushed the Trends graph below the fold — and the joint
    // trace and the live pose are the two things that must ALWAYS be visible.
    if (linkPairs.length)
      out += `<details class="st-conn"><summary class="np-sec">Connection</summary>` +
             `<div class="st-grid">${linkPairs.join('')}</div></details>`;

    // Every subsystem's connection status — arms, body, base, hands, sensors,
    // audio (the operator's first question when something "isn't moving").
    out += this._hardwareRows();

    // ── control state, two per row ───────────────────────────────────────────
    // CONTROL / POWER / RECORDING deliberately REMOVED.
    //
    // Every one of them duplicated something already on screen and always in
    // view: Space and Feel are the segmented controls in the top bar and over
    // the viewer, Module is whichever chip is lit in the module rail, Torque is
    // the power button itself, and Recording is the ● Record button plus its
    // task chip. They cost ~200 px of dock, which is exactly what pushed the
    // JOINT STATE bars and the TRENDS graph below the fold — the operator
    // reported the trends as "not working" when in fact they were rendering
    // perfectly, just off-screen. The dock's job is the things with NO other
    // home: link health, hardware roster, sensors. Keep it that way.
    this.el.innerHTML = out;
  }

  renderData(){
    const r = this.recordings;
    let out = `<div class="np-sec">Episodes by Task</div>`;
    if (!r){ this.dataEl.innerHTML = out + `<div class="np-empty">backend offline</div>`; return; }
    const eps = r.episodes || {};
    out += this._row('Total episodes', eps.total ?? 0);
    for (const t of eps.tasks || []){
      out += `<div class="st-task"><div class="st-row"><span>▸ ${esc(t.task)}</span>
        <b>${t.episodes} ep · ${fmtBytes(t.bytes)}</b></div></div>`;
    }
    if (!(eps.tasks || []).length) out += `<div class="np-empty">no episodes in ${esc(eps.root || '?')}</div>`;
    const bags = r.bags || {};
    out += `<div class="np-sec">Rosbag Sessions · ${bags.total ?? 0}</div>`;
    for (const b of (bags.sessions || []).slice(0, 12)){
      out += `<div class="st-row"><span>${esc(b.label)}</span><b>${b.mcaps} mcap · ${fmtBytes(b.bytes)}</b></div>`;
    }
    this.dataEl.innerHTML = out;
  }
}
