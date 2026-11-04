// TelemetryLink — client of OUR backend's /ws/telemetry: the 59-DOF schema
// (once) then parallel arrays of pos/vel/effort/targets/temps + estop/torque
// at ~15 Hz, with a ~1 Hz host/system + viewers sidecar. Auto-reconnects.
import * as bus from './bus.js';
import { telemetryUrl } from './config.js';

// Same silent-death guard as RobotLink: /ws/telemetry pushes at ~15 Hz (+1 Hz
// sidecar), so more than STALE_MS of inbound silence means the socket died
// without firing onclose — force it closed so the reconnect fires automatically
// instead of the joint bars freezing until a page refresh.
const STALE_MS = 5000;
const CONNECT_TIMEOUT_MS = 8000;

export class TelemetryLink {
  constructor(){
    this.schema = null;             // {names, modules, dof}
    this.last = null;               // latest telemetry payload
    this.system = null;             // latest system sidecar
    this.viewers = [];
    this.connected = false;
    this._closed = false;
    this._retry = 1000;
    this._lastRx = 0;
    this._connectAt = 0;
    this._watchTimer = setInterval(() => this._watch(), 1500);
  }

  connect(){
    this._closed = false;
    this._connectAt = performance.now();
    let ws;
    try { ws = new WebSocket(telemetryUrl()); }
    catch { this._scheduleRetry(); return; }
    this.ws = ws;
    ws.onopen = () => { this.connected = true; this._retry = 1000; this._lastRx = performance.now(); bus.emit('telemetryLink', true); };
    ws.onclose = () => {
      if (this.ws && this.ws !== ws) return;   // a stale socket's late close — ignore
      this.connected = false; this.ws = null;
      bus.emit('telemetryLink', false);
      if (!this._closed) this._scheduleRetry();
    };
    ws.onmessage = (ev) => {
      this._lastRx = performance.now();
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'schema'){
        this.schema = msg.payload; bus.emit('telemetrySchema', this.schema);
      } else if (msg.type === 'telemetry'){
        this.last = msg.payload;
        if (msg.payload.system){ this.system = msg.payload.system; bus.emit('system', this.system); }
        if (msg.payload.viewers){ this.viewers = msg.payload.viewers; }
        bus.emit('telemetry', msg.payload);
      }
    };
  }

  _scheduleRetry(){
    setTimeout(() => { if (!this._closed && !this.ws) this.connect(); }, this._retry);
    this._retry = Math.min(this._retry * 1.6, 8000);
  }
  // heartbeat watchdog: force-close a dead-but-OPEN socket (or a stuck handshake)
  // so onclose → _scheduleRetry reconnects without a manual refresh
  _watch(){
    if (!this.ws) return;
    const idle = performance.now() - (this.connected ? this._lastRx : this._connectAt);
    if (idle > (this.connected ? STALE_MS : CONNECT_TIMEOUT_MS)){
      try { this.ws.close(); } catch { /* onclose still schedules the retry */ }
    }
  }
  close(){ this._closed = true; clearInterval(this._watchTimer); this.ws?.close(); }

  // {jointName: value} view of one channel of the latest sample
  channel(key){
    if (!this.schema || !this.last) return {};
    const out = {};
    const arr = this.last[key] || [];
    this.schema.names.forEach((n, i) => { if (arr[i] != null) out[n] = arr[i]; });
    return out;
  }
}
