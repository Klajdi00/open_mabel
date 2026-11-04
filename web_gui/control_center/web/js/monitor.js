// MonitorLink — read-only subscriber of the bridge's /monitor fan-out
// (ws :9090/monitor): per-client stats {id, peer, device, platform, net{ip,
// transport, geo}, driver, rateHz, mode, ...} + a global snapshot {online,
// driverId, recording, status}. Feeds the networking rail.
import * as bus from './bus.js';
import { monitorUrl } from './config.js';

export class MonitorLink {
  constructor(){
    this.clients = [];
    this.global = null;
    this.connected = false;
    this._closed = false;
    this._retry = 1500;
  }

  connect(){
    this._closed = false;
    let ws;
    try { ws = new WebSocket(monitorUrl()); }
    catch { this._scheduleRetry(); return; }
    this.ws = ws;
    ws.onopen = () => { this.connected = true; this._retry = 1500; bus.emit('monitorLink', true); };
    ws.onclose = () => {
      this.connected = false; this.ws = null;
      bus.emit('monitorLink', false);
      if (!this._closed) this._scheduleRetry();
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      // snapshots arrive either flat {clients, ...global} or typed envelopes
      const p = msg.payload || msg;
      if (Array.isArray(p.clients)) this.clients = p.clients;
      const { clients, ...rest } = p;
      if (Object.keys(rest).length) this.global = { ...this.global, ...rest };
      bus.emit('monitor', { clients: this.clients, global: this.global });
    };
  }

  _scheduleRetry(){
    setTimeout(() => { if (!this._closed && !this.ws) this.connect(); }, this._retry);
    this._retry = Math.min(this._retry * 1.6, 10000);
  }
  close(){ this._closed = true; this.ws?.close(); }
}
