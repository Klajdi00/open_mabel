// Add-Window panel types (curation-studio style, tiled by PanelManager):
//   cam        one live camera, selectable stream
//   tftree     the rig's transform tree (indented, joints highlighted)
//   grid2d     top-down 2D occupancy map + robot pose
//   jointhist  line graphs of joint history since boot (rolling window,
//              default 2 min, adjustable up to 5 min), per robot module
import * as bus from './bus.js';
import { CamView } from './camview.js';
import { GROUP_LABEL } from './names.js';
import { DEFAULT_WINDOW_S, MAX_WINDOW_S } from './history.js';

const CAMS = ['main', 'front', 'wrist_left', 'wrist_right'];

export class CamWindow {
  constructor(panel){
    this.body = panel.body;
    this.body.innerHTML = `
      <div class="win-bar"><div class="seg" id="s"></div></div>
      <div class="win-cam"><img alt="camera"></div>`;
    const seg = this.body.querySelector('#s');
    const img = this.body.querySelector('img');
    // H.264/WebCodecs with an MJPEG fallback — see camview.js
    this._view = new CamView(img, CAMS[0]);
    CAMS.forEach((n, i) => {
      const b = document.createElement('button');
      b.textContent = n.replace('_', ' ');
      if (!i) b.classList.add('on');
      b.onclick = () => {
        seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
        this._view.setName(n);
      };
      seg.appendChild(b);
    });
    // re-attach after a scene switch / camera restart
    this._offRefresh = bus.on?.('streamsRefresh', () => this._view.connect());
  }
  destroy(){ this._view.stop(); this._offRefresh?.(); }
}

export class TfTreeWindow {
  constructor(panel, viewport){
    this.body = panel.body;
    this.viewport = viewport;
    this.body.classList.add('win-tftree');
    this._timer = setInterval(() => this.render(), 1500);
    this.render();
  }
  render(){
    const tree = this.viewport.getTfTree();
    if (!tree){ this.body.innerHTML = '<div class="np-empty">rig loading…</div>'; return; }
    const rows = [];
    const walk = (n, depth) => {
      rows.push(`<div class="tft-row ${n.joint ? 'joint' : ''}" style="padding-left:${8 + depth * 14}px">
        <span class="tft-name">${n.name}</span>
        <span class="tft-pos mono">${n.pos.map(v => v.toFixed(2)).join(', ')}</span></div>`);
      n.children.forEach(c => walk(c, depth + 1));
    };
    walk(tree, 0);
    this.body.innerHTML = rows.join('');
  }
  destroy(){ clearInterval(this._timer); }
}

export class Grid2DWindow {
  constructor(panel){
    this.body = panel.body;
    this.body.classList.add('win-grid2d');
    this.canvas = document.createElement('canvas');
    this.body.appendChild(this.canvas);
    this.grid = null; this.pose = null;
    this._off = [
      bus.on('map:grid', (g) => { this.grid = g; this.draw(); }),
      bus.on('stream:grid', (g) => { this.grid = g; this.draw(); }),
      bus.on('robotState', (s) => { this.pose = s.base; }),
    ];
    this._timer = setInterval(() => this.draw(), 800);
  }
  draw(){
    const g = this.grid, cv = this.canvas, host = this.body;
    const W = host.clientWidth, H = host.clientHeight;
    if (!W || !H) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.fillStyle = '#0d0c0b'; x.fillRect(0, 0, W, H);
    if (!g){ x.fillStyle = '#6b655f'; x.font = '11px Geist Mono'; x.fillText('no grid yet', 12, 20); return; }
    const s = Math.min(W / g.w, H / g.h);
    const ox = (W - g.w * s) / 2, oy = (H + g.h * s) / 2;   // y flipped (map y-up)
    const im = x.createImageData(g.w, g.h);
    for (let i = 0; i < g.w * g.h; i++){
      const v = g.cells[i];
      const k = v === 255 ? 30 : 220 - v * 1.9;
      im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = k;
      im.data[i * 4 + 3] = v === 255 ? 90 : 255;
    }
    const tmp = document.createElement('canvas');
    tmp.width = g.w; tmp.height = g.h;
    tmp.getContext('2d').putImageData(im, 0, 0);
    x.save();
    x.translate(ox, oy); x.scale(s, -s);
    x.imageSmoothingEnabled = false;
    x.drawImage(tmp, 0, 0);
    // robot pose (odom → grid cells)
    if (this.pose){
      const gx = (this.pose.x - g.ox) / g.res, gy = (this.pose.y - g.oy) / g.res;
      x.fillStyle = '#C25B2A';
      x.beginPath(); x.arc(gx, gy, 4 / s, 0, 7); x.fill();
      x.strokeStyle = '#C25B2A'; x.lineWidth = 2 / s;
      x.beginPath(); x.moveTo(gx, gy);
      x.lineTo(gx + Math.cos(this.pose.yaw || 0) * 10 / s,
               gy + Math.sin(this.pose.yaw || 0) * 10 / s);
      x.stroke();
    }
    x.restore();
  }
  resize(){ this.draw(); }
  destroy(){ this._off.forEach(f => f()); clearInterval(this._timer); }
}

export class JointHistoryWindow {
  constructor(panel, history, schema){
    this.body = panel.body;
    this.history = history;
    this.schema = schema;
    this.moduleKey = 'left_arm';
    this.chan = 'p';
    this.windowS = DEFAULT_WINDOW_S;
    this.body.classList.add('win-jhist');
    this.body.innerHTML = `
      <div class="win-bar">
        <select id="mod" class="tl-wave"></select>
        <div class="seg" id="ch">
          <button class="on" data-c="p">pos</button>
          <button data-c="v">vel</button>
          <button data-c="e">torque</button>
        </div>
        <label class="cc-lbl">window <input type="range" id="win" min="30"
          max="${MAX_WINDOW_S}" step="10" value="${DEFAULT_WINDOW_S}">
          <b id="winV" class="mono">${DEFAULT_WINDOW_S}s</b></label>
      </div>
      <canvas></canvas>`;
    const mod = this.body.querySelector('#mod');
    for (const m of schema.modules){
      const o = document.createElement('option');
      o.value = m.key; o.textContent = GROUP_LABEL[m.key] || m.label;
      mod.appendChild(o);
    }
    mod.value = this.moduleKey;
    mod.onchange = () => { this.moduleKey = mod.value; };
    this.body.querySelector('#ch').onclick = (e) => {
      const b = e.target.closest('button'); if (!b) return;
      this.body.querySelectorAll('#ch button').forEach(x => x.classList.toggle('on', x === b));
      this.chan = b.dataset.c;
    };
    const win = this.body.querySelector('#win');
    win.oninput = () => {
      this.windowS = +win.value;
      this.body.querySelector('#winV').textContent = win.value + 's';
    };
    this.canvas = this.body.querySelector('canvas');
    this._timer = setInterval(() => this.draw(), 500);
  }

  draw(){
    const cv = this.canvas, host = this.body;
    const W = host.clientWidth, H = host.clientHeight - 34;
    if (W < 20 || H < 20) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, W, H);
    const mod = this.schema.modules.find(m => m.key === this.moduleKey);
    if (!mod) return;
    // gather series + global scale
    const series = [];
    let lo = Infinity, hi = -Infinity;
    for (const jn of mod.joints){
      const ji = this.schema.names.indexOf(jn);
      const s = this.history.series(ji, this.chan, this.windowS);
      const tg = this.chan === 'p' ? this.history.series(ji, 'tgt', this.windowS) : null;
      for (const v of s.v) if (Number.isFinite(v)){ lo = Math.min(lo, v); hi = Math.max(hi, v); }
      series.push({ name: jn, s, tg });
    }
    if (!Number.isFinite(lo)){ x.fillStyle = '#6b655f'; x.font = '11px Geist Mono'; x.fillText('no history yet', 12, 20); return; }
    if (hi - lo < 1e-6){ hi += 0.5; lo -= 0.5; }
    const t1 = performance.now() / 1000;
    const px = (t) => W - (t1 - t) / this.windowS * W;
    const py = (v) => H - 6 - (v - lo) / (hi - lo) * (H - 12);
    series.forEach((sr, i) => {
      const hue = (i * 47) % 360;
      const line = (data, dash) => {
        if (!data || !data.t.length) return;
        x.strokeStyle = `hsl(${hue},70%,${dash ? 75 : 60}%)`;
        x.setLineDash(dash ? [4, 4] : []);
        x.lineWidth = 1.2;
        x.beginPath();
        let started = false;
        for (let k = 0; k < data.t.length; k++){
          const v = data.v[k];
          if (!Number.isFinite(v)){ started = false; continue; }
          const X = px(data.t[k]), Y = py(v);
          if (!started){ x.moveTo(X, Y); started = true; } else x.lineTo(X, Y);
        }
        x.stroke();
        x.setLineDash([]);
      };
      line(sr.s, false);
      line(sr.tg, true);
    });
    // scale labels
    x.fillStyle = '#6b655f'; x.font = '9px Geist Mono';
    x.fillText(hi.toFixed(2), 4, 10);
    x.fillText(lo.toFixed(2), 4, H - 2);
    x.fillText(`-${this.windowS}s`, 4, H - 14);
  }
  resize(){ this.draw(); }
  destroy(){ clearInterval(this._timer); }
}

// ── the native MuJoCo viewer, in the browser ────────────────────────────────
// Full-scene render streamed from /ws/simview: drag = orbit, wheel = zoom,
// shift-drag = pan (drops follow), double-click = re-follow the robot.
// On the real robot the gateway mirrors the encoders into the same qpos
// memmap, so this window shows the MIRRORED REAL ROBOT in the MJCF scene.
export class SimViewWindow {
  constructor(panel, opts = {}){
    this.body = panel.body;
    this.instance = opts.instance || null;    // e.g. 'preview' — the isolated
                                              // physics-preview world's viewer
    this.body.innerHTML = `
      <div class="win-bar mono" style="font-size:9px;color:var(--ink-soft);gap:10px">
        <span>${opts.title || 'MUJOCO VIEWER'}</span>
        <span style="opacity:.65">drag orbit · wheel zoom · ⇧drag pan · dbl-click follow</span>
        <span id="svStat" style="margin-left:auto"></span>
      </div>
      <div class="win-cam win-simview"><img alt="mujoco viewer" draggable="false"></div>`;
    this.img = this.body.querySelector('img');
    this.stat = this.body.querySelector('#svStat');
    this.cam = { az: 150, el: -20, dist: 3.2, lookat: [0, 0, 0.9], follow: true };
    this._url = null;
    this._open();
    this._wireInput();
    this._ro = new ResizeObserver(() => this._sendSize());
    this._ro.observe(this.body);
  }

  _open(){
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/simview`
      + (this.instance ? `?instance=${this.instance}` : ''));
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => { this.stat.textContent = '● live'; this._sendSize(); this._send(); };
    ws.onclose = () => {
      this.stat.textContent = '○ reconnecting…';
      if (!this._closed) setTimeout(() => this._open(), 1500);
    };
    ws.onmessage = (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      const url = URL.createObjectURL(new Blob([ev.data], { type: 'image/jpeg' }));
      if (this._url) URL.revokeObjectURL(this._url);
      this._url = url;
      this.img.src = url;
    };
  }

  _send(){
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(this.cam));
  }
  _sendSize(){
    const r = this.img.parentElement.getBoundingClientRect();
    if (this.ws?.readyState === 1 && r.width > 40){
      this.ws.send(JSON.stringify({ w: Math.round(r.width),
        h: Math.round(r.height || r.width * 0.62) }));
    }
  }

  _wireInput(){
    const el = this.img.parentElement;
    let drag = null;
    el.style.cursor = 'grab';
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 };
      el.style.cursor = 'grabbing';
    });
    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan){
        // pan in the camera plane (x-y world) — leaves follow mode
        const s = this.cam.dist * 0.0022;
        const az = this.cam.az * Math.PI / 180;
        this.cam.follow = false;
        this.cam.lookat[0] += (-dx * Math.sin(az) + dy * Math.cos(az)) * s;
        this.cam.lookat[1] += (dx * Math.cos(az) + dy * Math.sin(az)) * s;
      } else {
        this.cam.az = (this.cam.az - dx * 0.4) % 360;
        this.cam.el = Math.max(-89, Math.min(-2, this.cam.el - dy * 0.35));
      }
      this._send();
    });
    const up = (e) => { drag = null; el.style.cursor = 'grab'; };
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.cam.dist = Math.max(0.6, Math.min(18,
        this.cam.dist * Math.exp(e.deltaY * 0.0012)));
      this._send();
    }, { passive: false });
    el.addEventListener('dblclick', () => { this.cam.follow = true; this._send(); });
    el.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  destroy(){
    this._closed = true;
    this._ro.disconnect();
    try { this.ws?.close(); } catch { /* gone */ }
    if (this._url) URL.revokeObjectURL(this._url);
  }
}
