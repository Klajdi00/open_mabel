// RobotLink — the control-plane client of the bridge (ws :9090/teleop), the
// same JSON-envelope protocol the iOS app speaks: send teleop_frame /
// joint_command / control_mode / estop / nav_goal / record; receive hello,
// robot_state @50 Hz, pong, map_list, nav_result. Auto-reconnects.
import * as bus from './bus.js';
import { teleopUrl } from './config.js';

// A WebSocket can die SILENTLY — a NAT / VPN / Wi-Fi roam drops the TCP link but
// the browser keeps the socket in OPEN state and never fires `onclose`. The old
// code only reconnected from onclose, so a silent death left the studio frozen
// with a stale robot until a manual page refresh (the exact "state drops off,
// had to refresh" report). The watchdog below catches it: robot_state streams
// at 50 Hz and pong replies at 1 Hz, so more than STALE_MS of TOTAL inbound
// silence means the link is dead → force it closed to trigger the reconnect.
const STALE_MS = 4000;
// A connect that never completes its handshake (proxy accepts the TCP but never
// upgrades) also hangs forever with no onclose — abort it after this long.
const CONNECT_TIMEOUT_MS = 8000;

export class RobotLink {
  constructor(){
    this.ws = null;
    this.connected = false;
    this.hello = null;
    this.state = null;              // latest robot_state payload
    this.latencyMs = null;
    this.seq = 0;
    this._retry = 1000;
    this._closed = false;
    this._lastRx = 0;               // performance.now() of the last inbound frame
    this._connectAt = 0;            // performance.now() when the current connect began
    // one 1 Hz timer sends the latency ping AND runs the heartbeat watchdog
    this._pingTimer = setInterval(() => this._tick(), 1000);
  }

  connect(){
    this._closed = false;
    this._connectAt = performance.now();
    let ws;
    try { ws = new WebSocket(teleopUrl()); }
    catch { this._scheduleRetry(); return; }
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true; this._retry = 1000;
      this._lastRx = performance.now();
      bus.emit('link', { connected: true });
    };
    ws.onclose = () => {
      if (this.ws && this.ws !== ws) return;   // a stale socket's late close — ignore
      const was = this.connected;
      this.connected = false; this.ws = null;
      if (was) bus.emit('link', { connected: false });
      if (!this._closed) this._scheduleRetry();
    };
    ws.onerror = () => {};
    ws.onmessage = (ev) => {
      this._lastRx = performance.now();        // proof of life for the watchdog
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      const p = msg.payload || {};
      if (msg.type === 'robot_state'){
        this.state = p;
        if (typeof p.latencyMs === 'number') this.latencyMs = p.latencyMs;
        bus.emit('robotState', p);
      } else if (msg.type === 'hello'){
        this.hello = p; bus.emit('hello', p);
        this.send('list_maps', {});
      } else if (msg.type === 'pong'){
        if (p.t) this.latencyMs = Math.max(0, performance.now() - p.t);
        bus.emit('latency', this.latencyMs);
      } else if (msg.type === 'map_list'){
        bus.emit('mapList', p);
      } else if (msg.type === 'nav_result'){
        bus.emit('navResult', p);
      }
    };
  }

  _scheduleRetry(){
    setTimeout(() => { if (!this._closed && !this.ws) this.connect(); },
      this._retry);
    this._retry = Math.min(this._retry * 1.6, 8000);
  }

  close(){ this._closed = true; clearInterval(this._pingTimer); this.ws?.close(); }

  send(type, payload = {}){
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify({ type, payload }));
    return true;
  }

  // 1 Hz: latency ping + heartbeat watchdog (see STALE_MS above). A dead-but-OPEN
  // socket or a stuck handshake is force-closed so onclose → _scheduleRetry fires
  // — the same recovery a page refresh does, but automatic.
  _tick(){
    this._ping();
    if (!this.ws) return;
    const idle = performance.now() - (this.connected ? this._lastRx : this._connectAt);
    if (idle > (this.connected ? STALE_MS : CONNECT_TIMEOUT_MS)){
      try { this.ws.close(); } catch { /* onclose still schedules the retry */ }
    }
  }

  _ping(){ this.send('ping', { t: performance.now() }); }

  // ── command sugar (wire shapes per server/wire_protocol.py) ────────────────
  // navJoystick: ly=forward, lx=strafe(right = −), rx=yaw, ry=lift rate
  frame(nav, head){
    this.send('teleop_frame', {
      sequence: this.seq++, timestamp: Date.now() / 1000,
      mode: 'Base Driving', navJoystick: nav,
      ...(head ? { head } : {}),
    });
  }
  joints(map){ this.send('joint_command', { joints: map }); }
  controlMode(method, controlType, region, stiffness){
    this.send('control_mode', { method, controlType, region, stiffness });
  }
  estop(active){ this.send('estop', { active }); }
  navGoal(x, y, yaw){ this.send('nav_goal', { x, y, yaw }); }
  navPreview(x, y, yaw){ this.send('nav_goal_preview', { x, y, yaw }); }
  record(active, task){ this.send('record', active ? { active, task } : { active: false }); }
  selectTask(task){ this.send('record', { select: task }); }
  dataCollect(active, label){ this.send('data_collect', { active, label }); }
  externalForce(forces){ this.send('external_force', { forces }); }
}
