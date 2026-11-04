// StreamLink — client of OUR backend's /ws/stream: selectable depth clouds
// (head / front base, odom-framed) and the 2D grid from either SLAM stack.
// Same binary records as the map plane, so decodeRecord is shared. The last
// record's source is inferred server-side ordering-free by tagging: we send
// selections and route grids to 'stream:grid', clouds round-robin by the
// selection order embedded in each request cycle — to stay unambiguous we
// simply re-decode and dispatch by record kind + our single-grid / per-cloud
// bookkeeping (cloud sources are distinguished by dedicated sockets).
import * as bus from './bus.js';
import { decodeRecord } from './maplink.js';

const wsProto = () => (location.protocol === 'https:' ? 'wss' : 'ws');

// one socket per cloud source keeps attribution trivial and lets the browser
// drop a heavy source without disturbing the others
export class StreamLink {
  constructor(){
    this.sockets = {};            // key -> WebSocket
    this.grid = null;             // 'lidar' | 'visual' | null
  }

  _open(key, select){
    if (this.sockets[key]) return;
    let ws;
    try { ws = new WebSocket(`${wsProto()}://${location.host}/ws/stream`); }
    catch { return; }
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => ws.send(JSON.stringify(select));
    ws.onmessage = (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      let rec;
      try { rec = decodeRecord(ev.data); } catch { return; }
      if (!rec) return;
      if (rec.kind === 'grid') bus.emit('stream:grid', rec);
      else if (rec.kind === 'cloud') bus.emit('stream:cloud', { ...rec, source: key.slice(6) });
    };
    ws.onclose = () => { delete this.sockets[key]; };
    this.sockets[key] = ws;
  }
  _close(key){ this.sockets[key]?.close(); delete this.sockets[key]; }

  setCloud(source, on){
    const key = 'cloud:' + source;
    if (on) this._open(key, { cloud: [source] });
    else this._close(key);
  }
  setGrid(source){
    this.grid = source;
    this._close('grid');
    if (source) this._open('grid', { grid: source });
  }
}
