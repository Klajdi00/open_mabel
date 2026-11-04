// MapLink — binary client of the SLAM map plane (ws :9092/map, send-only from
// the server). One WS frame = one little-endian record, tagged by first byte:
//   0x01 MESH  u8 flags(b0 final, b1 indexed) u16 pad u32 nv, nv×{f32 xyz, u8 rgba}
//              [+ u32 ni, ni×u32 indices]           (nvblox surface, chunked)
//   0x02 ODOM  pad3, f32 x y z, f32 qx qy qz qw     (robot pose, odom frame)
//   0x03 PATH  pad3, u32 n, n×{f32 xyz}             (Nav2 /plan; n=0 clears)
//   0x04 GRID  u8 flags u16 pad u32 w h, f32 res ox oy oz yaw, w*h×u8 cells
//   0x05 CLOUD same vertex layout as MESH           (live sensor cloud)
// Decoded records are emitted on the bus; three.js consumption lives in
// viewport3d. Protocol: server/docs/data_streams.md.
import * as bus from './bus.js';
import { mapUrl } from './config.js';

export function decodeRecord(buf){
  const dv = new DataView(buf);
  const type = dv.getUint8(0);
  if (type === 0x01 || type === 0x05){
    const flags = dv.getUint8(1);
    const nv = dv.getUint32(4, true);
    const stride = 16, base = 8;
    const pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++){
      const o = base + i * stride;
      pos[i * 3] = dv.getFloat32(o, true);
      pos[i * 3 + 1] = dv.getFloat32(o + 4, true);
      pos[i * 3 + 2] = dv.getFloat32(o + 8, true);
      col[i * 3] = dv.getUint8(o + 12) / 255;
      col[i * 3 + 1] = dv.getUint8(o + 13) / 255;
      col[i * 3 + 2] = dv.getUint8(o + 14) / 255;
    }
    let idx = null;
    if (flags & 2){
      const io = base + nv * stride;
      const ni = dv.getUint32(io, true);
      idx = new Uint32Array(ni);
      for (let i = 0; i < ni; i++) idx[i] = dv.getUint32(io + 4 + i * 4, true);
    }
    return { kind: type === 0x01 ? 'mesh' : 'cloud',
             final: !!(flags & 1), pos, col, idx, count: nv };
  }
  if (type === 0x02){
    return { kind: 'odom',
             x: dv.getFloat32(4, true), y: dv.getFloat32(8, true), z: dv.getFloat32(12, true),
             qx: dv.getFloat32(16, true), qy: dv.getFloat32(20, true),
             qz: dv.getFloat32(24, true), qw: dv.getFloat32(28, true) };
  }
  if (type === 0x03){
    const n = dv.getUint32(4, true);
    const pts = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) pts[i] = dv.getFloat32(8 + i * 4, true);
    return { kind: 'path', pts, count: n };
  }
  if (type === 0x04){
    const w = dv.getUint32(4, true), h = dv.getUint32(8, true);
    const res = dv.getFloat32(12, true);
    const ox = dv.getFloat32(16, true), oy = dv.getFloat32(20, true),
          oz = dv.getFloat32(24, true), yaw = dv.getFloat32(28, true);
    const cells = new Uint8Array(buf, 32, w * h);
    return { kind: 'grid', w, h, res, ox, oy, oz, yaw, cells };
  }
  return null;
}

// The map plane is send-only and can be legitimately QUIET (no new mesh while
// the robot sits in a mapped room), so its silent-death threshold is generous —
// odom/mesh return well inside this whenever anything is actually happening, and
// a truly dead-but-OPEN socket still recovers instead of freezing the map.
const STALE_MS = 15000;
const CONNECT_TIMEOUT_MS = 8000;

export class MapLink {
  constructor(){
    this.connected = false;
    this._closed = false;
    this._retry = 1500;
    this._lastRx = 0;
    this._connectAt = 0;
    this.stats = { mesh: 0, cloud: 0, grid: 0, path: 0, odom: 0 };
    this._watchTimer = setInterval(() => this._watch(), 3000);
  }

  connect(){
    this._closed = false;
    this._connectAt = performance.now();
    let ws;
    try { ws = new WebSocket(mapUrl()); }
    catch { this._scheduleRetry(); return; }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => { this.connected = true; this._retry = 1500; this._lastRx = performance.now(); bus.emit('mapLink', true); };
    ws.onclose = () => {
      if (this.ws && this.ws !== ws) return;   // a stale socket's late close — ignore
      this.connected = false; this.ws = null;
      bus.emit('mapLink', false);
      if (!this._closed) this._scheduleRetry();
    };
    ws.onmessage = (ev) => {
      this._lastRx = performance.now();
      if (!(ev.data instanceof ArrayBuffer)) return;
      let rec;
      try { rec = decodeRecord(ev.data); } catch { return; }
      if (!rec) return;
      this.stats[rec.kind] = (this.stats[rec.kind] || 0) + 1;
      bus.emit('map:' + rec.kind, rec);
    };
  }

  _scheduleRetry(){
    setTimeout(() => { if (!this._closed && !this.ws) this.connect(); }, this._retry);
    this._retry = Math.min(this._retry * 1.6, 10000);
  }
  _watch(){
    if (!this.ws) return;
    const idle = performance.now() - (this.connected ? this._lastRx : this._connectAt);
    if (idle > (this.connected ? STALE_MS : CONNECT_TIMEOUT_MS)){
      try { this.ws.close(); } catch { /* onclose still schedules the retry */ }
    }
  }
  close(){ this._closed = true; clearInterval(this._watchTimer); this.ws?.close(); }
}
