// Left rail: the robot server as a network citizen — host + interface IPs,
// service-plane liveness (teleop/map/video/studio), CPU/mem/thermals, and the
// live client roster from the bridge /monitor plane (device type, transport,
// geo, latency/rate, driver) + this studio's own viewers. This page's OWN
// teleop connection is recognized by its per-tab ?sid= tag and rendered as
// "Current window" instead of a generic Website client.
import * as bus from './bus.js';
import { sessionId } from './config.js';

const DEVICE_ICON = { 'Vision Pro': '🥽', iPhone: '📱', iPad: '📱',
                      Website: '🌐', Keyboard: '⌨', Gamepad: '🎮', Linux: '🐧' };
const TRANSPORT_CLASS = { LAN: 't-lan', Tailscale: 't-ts',
                          Localhost: 't-local', Internet: 't-inet' };

const esc = (s) => String(s ?? '').replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtAge = (s) => s == null ? '?'
  : s < 90 ? `${Math.round(s)}s`
  : s < 5400 ? `${Math.round(s / 60)}m` : `${(s / 3600).toFixed(1)}h`;

export class NetPanel {
  constructor(el){
    this.el = el;
    this.system = null;
    this.monitor = { clients: [], global: null };
    this.viewers = [];
    this.links = { robot: false, monitor: false, map: false, telemetry: false };
    bus.on('system', (s) => { this.system = s; this.render(); });
    bus.on('monitor', (m) => { this.monitor = m; this.render(); });
    bus.on('telemetry', (t) => { if (t.viewers) this.viewers = t.viewers; });
    bus.on('link', (l) => { this.links.robot = l.connected; this.render(); });
    bus.on('monitorLink', (on) => { this.links.monitor = on; this.render(); });
    bus.on('mapLink', (on) => { this.links.map = on; this.render(); });
    bus.on('telemetryLink', (on) => { this.links.telemetry = on; this.render(); });
    this.render();
  }

  render(){
    const s = this.system, m = this.monitor;
    const dot = (up) => `<i class="np-dot ${up ? 'up' : 'down'}"></i>`;
    let h = '';

    // ── host-not-activated banner ──
    // The control link is the robot bridge (thor). When it's down the studio can
    // render but CANNOT drive anything — say so unmistakably at the top instead of
    // leaving the operator to infer it from a tiny "down" row.
    if (!this.links.robot){
      h += `<div class="np-offline">⚠ Robot host not activated<span>the control bridge on thor is offline — start the robot stack to connect. Teleop is disabled.</span></div>`;
    }

    // ── server host ──
    h += `<div class="np-sec">Robot Server</div>`;
    if (s){
      h += `<div class="np-row"><span>host</span><b>${esc(s.hostname)}</b></div>`;
      for (const i of s.interfaces || [])
        h += `<div class="np-row"><span>${esc(i.name)}</span><b class="mono">${esc(i.ip)}</b></div>`;
      const pc = s.cpu_pct != null ? s.cpu_pct.toFixed(0) + '%' : `load ${s.load?.[0]?.toFixed(2)}`;
      h += `<div class="np-row"><span>cpu</span><b>${pc}</b></div>`;
      h += `<div class="np-row"><span>mem</span><b>${(s.mem?.pct ?? 0).toFixed(0)}%</b></div>`;
      const hot = Object.entries(s.thermal || {}).sort((a, b) => b[1] - a[1])[0];
      if (hot) h += `<div class="np-row"><span>therm · ${esc(hot[0])}</span><b>${hot[1].toFixed(0)}°C</b></div>`;
    } else {
      h += `<div class="np-empty">no backend telemetry</div>`;
    }

    // ── service planes ──
    h += `<div class="np-sec">Services</div>`;
    if (s?.services){
      for (const [k, v] of Object.entries(s.services))
        h += `<div class="np-row" title="${esc(v.what)}">${dot(v.up)}<span>${esc(k)}</span><b class="mono">:${v.port}</b></div>`;
    }
    h += `<div class="np-row">${dot(this.links.robot)}<span>control link</span><b>${this.links.robot ? 'live' : 'down'}</b></div>`;
    h += `<div class="np-row">${dot(this.links.map)}<span>map link</span><b>${this.links.map ? 'live' : 'down'}</b></div>`;

    // ── connected clients (bridge /monitor) ──
    const cl = m.clients || [];
    h += `<div class="np-sec">Clients · ${cl.length}</div>`;
    if (!cl.length) h += `<div class="np-empty">${this.links.monitor ? 'no clients connected' : 'monitor plane offline'}</div>`;
    // this page's own connection first, then drivers, then the rest
    const isSelf = (c) => c.session && c.session === sessionId;
    cl.sort((a, b) => (isSelf(b) - isSelf(a)) || (b.driver - a.driver));
    for (const c of cl){
      const self = isSelf(c);
      const icon = self ? '🪟' : (DEVICE_ICON[c.device] || '·');
      const name = self ? 'Current window' : (c.device || c.platform || 'client');
      const tcls = TRANSPORT_CLASS[c.net?.transport] || '';
      const geo = c.net?.geo ? ` · ${esc(c.net.geo.city || '')} ${esc(c.net.geo.country || '')}` : '';
      const sys = [c.browser, c.os].filter(Boolean).join(' · ');
      h += `<div class="np-client ${c.driver ? 'driver' : ''} ${self ? 'self' : ''}">
        <div class="np-cl-head"><span class="np-ic">${icon}</span>
          <b>${esc(name)}</b>
          ${c.driver ? '<span class="np-badge">DRIVER</span>' : ''}
          <span class="np-chip ${tcls}">${esc(c.net?.transport || '?')}</span></div>
        <div class="np-cl-sub mono">${esc(c.net?.ip || c.peer || '')}${geo}</div>
        ${sys && !self ? `<div class="np-cl-sub">${esc(sys)}</div>` : ''}
        <div class="np-cl-sub">${(c.rateHz ?? 0).toFixed?.(0) ?? c.rateHz} Hz · ${esc(c.mode || '')} ${esc(c.region || '')} · idle ${(c.idleS ?? 0).toFixed?.(0) ?? '?'}s · up ${fmtAge(c.ageS)}</div>
      </div>`;
    }

    // ── studio viewers (our own backend registry) ──
    h += `<div class="np-sec">Studio Viewers · ${this.viewers.length}</div>`;
    for (const v of this.viewers){
      const ua = /Chrome/.test(v.ua) ? 'Chrome' : /Safari/.test(v.ua) ? 'Safari'
        : /Firefox/.test(v.ua) ? 'Firefox' : 'browser';
      h += `<div class="np-row"><span>🌐 ${ua}</span><b class="mono">${esc(v.ip)}</b></div>`;
    }
    this.el.innerHTML = h;
  }
}
