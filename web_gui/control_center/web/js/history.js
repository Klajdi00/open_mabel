// Rolling telemetry history since page load: ring buffer of the 15 Hz samples
// (pos / target / vel / effort per joint) so windows can plot "what happened",
// curation-studio style. Capacity = 5 min (the max view window); the graphs
// choose how much of it to show (default 2 min).
export const MAX_WINDOW_S = 300;
export const DEFAULT_WINDOW_S = 120;

export class TelemetryHistory {
  constructor(hz = 15){
    this.hz = hz;
    this.cap = MAX_WINDOW_S * hz;
    this.t = new Float64Array(this.cap);
    this.buf = { p: [], tgt: [], v: [], e: [] };
    this.n = 0;                       // total samples ever pushed
    this.names = null;
  }

  setSchema(schema){
    this.names = schema.names;
    for (const k in this.buf)
      this.buf[k] = Array.from({ length: schema.names.length },
        () => new Float32Array(this.cap).fill(NaN));
  }

  push(payload){
    if (!this.names) return;
    const i = this.n % this.cap;
    // CLIENT clock, deliberately — NOT payload.t.
    //
    // The server stamps `t` with time.time(), i.e. WALL-CLOCK UNIX EPOCH
    // (~1.78e9), while every plot draws its "now" from performance.now()/1000
    // (page uptime, ~1e1). Storing the server's stamp mixed the two clock
    // domains, so `now - t` came out at ~-1.8e9 seconds and every line was
    // laid out roughly 1e7 screen-widths off the right edge of the canvas:
    // the buffer filled correctly, the autoscale found finite min/max, and the
    // graph still rendered COMPLETELY BLANK. That is the "trends show no data"
    // bug. Stamping on arrival keeps the buffer in the same monotonic domain
    // the plots measure against, and is immune to the server's clock, to NTP
    // steps, and to Tailscale-vs-LAN skew.
    this.t[i] = performance.now() / 1000;
    for (const k of ['p', 'tgt', 'v', 'e']){
      const arr = payload[k] || [];
      const store = this.buf[k];
      for (let j = 0; j < this.names.length; j++)
        store[j][i] = arr[j] == null ? NaN : arr[j];
    }
    this.n++;
  }

  // last `windowS` seconds of one joint+channel → {t:[], v:[], derived} oldest-first.
  //
  // `derived` flags a velocity series reconstructed from position: the real
  // robot's /joint_states is assembled by joint_state_publisher, which
  // ZERO-FILLS velocity and effort, so v/e arrive null unless the
  // per-subsystem topics are up. Differentiating position is what the iOS app
  // does in the same situation — a real (if noisier) trace beats a blank pane.
  series(jointIdx, key, windowS = DEFAULT_WINDOW_S){
    const count = Math.min(this.n, Math.min(windowS * this.hz, this.cap));
    const store = this.buf[key]?.[jointIdx];
    if (!store) return { t: [], v: [], derived: false };
    const t = new Float64Array(count), v = new Float32Array(count);
    let finite = 0;
    for (let k = 0; k < count; k++){
      const i = (this.n - count + k) % this.cap;
      t[k] = this.t[i];
      v[k] = store[i];
      if (Number.isFinite(v[k])) finite++;
    }
    if (finite === 0 && key === 'v'){
      const d = this._derivative(jointIdx, count);
      if (d) return { ...d, derived: true };
    }
    return { t, v, derived: false };
  }

  // d(pos)/dt over the same window — the velocity fallback above.
  _derivative(jointIdx, count){
    const pos = this.buf.p?.[jointIdx];
    if (!pos || count < 2) return null;
    const t = new Float64Array(count), v = new Float32Array(count).fill(NaN);
    let any = false;
    for (let k = 0; k < count; k++){
      const i = (this.n - count + k) % this.cap;
      t[k] = this.t[i];
      if (k === 0) continue;
      const j = (this.n - count + k - 1) % this.cap;
      const dt = this.t[i] - this.t[j];
      if (!(dt > 1e-4) || !Number.isFinite(pos[i]) || !Number.isFinite(pos[j])) continue;
      v[k] = (pos[i] - pos[j]) / dt;
      any = true;
    }
    return any ? { t, v } : null;
  }

  // does any joint carry real samples on this channel? (drives the honest
  // "the robot is not publishing this" note instead of an empty canvas)
  hasChannel(key){
    const store = this.buf[key];
    if (!store || !this.n) return false;
    const count = Math.min(this.n, this.cap);
    for (let j = 0; j < store.length; j++)
      for (let k = 0; k < count; k++)
        if (Number.isFinite(store[j][(this.n - count + k) % this.cap])) return true;
    return false;
  }

  spanS(){ return this.n ? Math.min(this.n, this.cap) / this.hz : 0; }
}
