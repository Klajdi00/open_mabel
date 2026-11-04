// IMU viewer (Rerun-style) — a derived IMU for the head (ZED) or front-base
// (RealSense D435i) mount. Episodes don't log a raw IMU, so we reconstruct it
// from the body-pose ground truth: orientation (quaternion/RPY), body-frame
// angular velocity (quaternion derivative) and specific-force acceleration
// (2nd-diff of position rotated into the body frame, + gravity) — what an IMU
// measures.
//
// The orientation is drawn as a real 3D board (isometric canvas-2D projection of
// the rotated sensor plane + body axes), with two synced line graphs: linear
// acceleration (x,y,z) and angular velocity (roll,pitch,yaw rates). Pure 2D
// canvas — no extra WebGL context, so it's light and renders everywhere.
const MOUNTS = [
  { id: 'head', label: 'head (ZED)', key: 'pose/head' },
  { id: 'front_base', label: 'front base (RealSense D435i)', key: 'pose/chassis' },
];
const G = 9.81;
const COLS = ['#ff6b5e', '#5eff9b', '#5eb0ff'];   // x / y / z  (and roll / pitch / yaw)
const ISO = Math.PI / 6;                            // isometric view angle

function quatEuler(w, x, y, z){
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const sp = 2 * (w * y - z * x);
  const pitch = Math.abs(sp) >= 1 ? Math.sign(sp) * Math.PI / 2 : Math.asin(sp);
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [roll, pitch, yaw];
}
function toBody(v, w, x, y, z){
  const cx = -x, cy = -y, cz = -z;
  const ix = w * v[0] + cy * v[2] - cz * v[1];
  const iy = w * v[1] + cz * v[0] - cx * v[2];
  const iz = w * v[2] + cx * v[1] - cy * v[0];
  const iw = -cx * v[0] - cy * v[1] - cz * v[2];
  return [ix * w + iw * -cx + iy * -cz - iz * -cy,
          iy * w + iw * -cy + iz * -cx - ix * -cz,
          iz * w + iw * -cz + ix * -cy - iy * -cx];
}
// rotate vector v by quaternion q=[w,x,y,z]  (world frame)
function qrot(q, v){
  const [w, ux, uy, uz] = q;
  const tx = 2 * (uy * v[2] - uz * v[1]), ty = 2 * (uz * v[0] - ux * v[2]), tz = 2 * (ux * v[1] - uy * v[0]);
  return [v[0] + w * tx + (uy * tz - uz * ty),
          v[1] + w * ty + (uz * tx - ux * tz),
          v[2] + w * tz + (ux * ty - uy * tx)];
}

function computeIMU(data, dim, T, fps){
  const dt = 1 / (fps || 15);
  const rpy = [], omega = [], accel = [], quat = [];
  const Q = (i) => [data[i * dim + 3], data[i * dim + 4], data[i * dim + 5], data[i * dim + 6]];
  const P = (i) => [data[i * dim], data[i * dim + 1], data[i * dim + 2]];
  for (let i = 0; i < T; i++){
    const [w, x, y, z] = Q(i);
    quat.push([w, x, y, z]); rpy.push(quatEuler(w, x, y, z));
    if (i > 0){
      const [w0, x0, y0, z0] = Q(i - 1);
      const dw = w * w0 + x * x0 + y * y0 + z * z0;
      const rx = x * w0 - w * x0 + z * y0 - y * z0;
      const ry = y * w0 - w * y0 + x * z0 - z * x0;
      const rz = z * w0 - w * z0 + y * x0 - x * y0;
      const vlen = Math.hypot(rx, ry, rz);
      const ang = 2 * Math.atan2(vlen, Math.abs(dw));
      const s = vlen > 1e-9 ? (ang / vlen) / dt * Math.sign(dw || 1) : 0;
      omega.push(toBody([rx * s, ry * s, rz * s], w, x, y, z));
    } else omega.push([0, 0, 0]);
    if (i > 0 && i < T - 1){
      const pm = P(i - 1), p0 = P(i), pp = P(i + 1);
      const aw = [(pp[0] - 2 * p0[0] + pm[0]) / (dt * dt),
                  (pp[1] - 2 * p0[1] + pm[1]) / (dt * dt),
                  (pp[2] - 2 * p0[2] + pm[2]) / (dt * dt) + G];
      accel.push(toBody(aw, w, x, y, z));
    } else accel.push([0, 0, G]);
  }
  return { rpy, omega, accel, quat, T };
}

export class IMU {
  constructor(pane){
    this.pane = pane; this.mount = 'head'; this.entry = null; this.imu = null; this.frame = 0;
    pane.innerHTML = `<div class="imu-root">
      <div class="imu-head"><span>Mount</span>
        <select class="imu-mount">${MOUNTS.map(m => `<option value="${m.id}">${m.label}</option>`).join('')}</select></div>
      <div class="imu-3d"><canvas class="imu-canvas"></canvas><div class="imu-rpy"></div></div>
      <div class="imu-sec">
        <div class="imu-plt-lbl">linear acceleration<i>m/s²</i>
          <span class="imu-leg"><i style="background:${COLS[0]}"></i>x<i style="background:${COLS[1]}"></i>y<i style="background:${COLS[2]}"></i>z</span></div>
        <canvas class="imu-accel"></canvas></div>
      <div class="imu-sec">
        <div class="imu-plt-lbl">angular velocity<i>rad/s</i>
          <span class="imu-leg"><i style="background:${COLS[0]}"></i>roll<i style="background:${COLS[1]}"></i>pitch<i style="background:${COLS[2]}"></i>yaw</span></div>
        <canvas class="imu-omega"></canvas></div>
    </div>`;
    this.c3d = pane.querySelector('.imu-canvas'); this.rpyEl = pane.querySelector('.imu-rpy');
    this.cAccel = pane.querySelector('.imu-accel'); this.cOmega = pane.querySelector('.imu-omega');
    pane.querySelector('.imu-mount').onchange = (e) => { this.mount = e.target.value; this._recompute(); this.draw(); };
    new ResizeObserver(() => this.draw()).observe(pane);
  }

  setEpisode(entry){ this.entry = entry; this._recompute(); this.draw(); }
  _recompute(){
    if (!this.entry){ this.imu = null; return; }
    const key = MOUNTS.find(m => m.id === this.mount).key;
    const t = this.entry.bundle.topics[key];
    this.imu = t ? computeIMU(t.data, t.dim, this.entry.T, this.entry.fps) : null;
  }
  update(entry, f){ this.frame = f; if (this.pane.offsetParent) this.draw(); }

  draw(){
    if (!this.pane.offsetParent) return;
    if (!this.imu){ this.rpyEl.textContent = ''; return; }
    const f = Math.max(0, Math.min(this.frame, this.imu.T - 1));
    const [r, p, yw] = this.imu.rpy[f];
    const dg = (v) => (v * 180 / Math.PI).toFixed(0) + '°';
    this.rpyEl.innerHTML =
      `<span style="color:${COLS[0]}">R ${dg(r)}</span><span style="color:${COLS[1]}">P ${dg(p)}</span><span style="color:${COLS[2]}">Y ${dg(yw)}</span>`;
    this._board(this.imu.quat[f]);
    this._plot(this.cAccel, this.imu.accel, f);
    this._plot(this.cOmega, this.imu.omega, f);
  }

  // isometric 3D board: a flat sensor plane + body axes, rotated by the IMU quat
  _board(q){
    const c = this.c3d, x = c.getContext('2d');
    const W = c.width = Math.max(2, c.clientWidth * devicePixelRatio);
    const H = c.height = Math.max(2, c.clientHeight * devicePixelRatio);
    x.clearRect(0, 0, W, H);
    const s = Math.min(W, H) * 0.30, cx = W / 2, cy = H / 2 + s * 0.35;
    const proj = (p) => [cx + (p[0] - p[1]) * Math.cos(ISO) * s, cy + (p[0] + p[1]) * Math.sin(ISO) * s - p[2] * s];
    // floor grid (z=0) for spatial reference
    x.strokeStyle = 'rgba(242,238,230,.07)'; x.lineWidth = devicePixelRatio;
    for (let i = -2; i <= 2; i++){
      let a = proj([i * 0.6, -1.2, 0]), b = proj([i * 0.6, 1.2, 0]);
      x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke();
      a = proj([-1.2, i * 0.6, 0]); b = proj([1.2, i * 0.6, 0]);
      x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke();
    }
    if (!q) return;
    // board shadow on the floor (corners flattened to z=0)
    const corner = (sx, sy) => qrot(q, [sx * 0.8, sy * 0.55, 0]);
    const cs = [corner(1, 1), corner(-1, 1), corner(-1, -1), corner(1, -1)];
    x.fillStyle = 'rgba(0,0,0,.28)'; x.beginPath();
    cs.forEach((p, i) => { const q2 = proj([p[0], p[1], 0]); i ? x.lineTo(q2[0], q2[1]) : x.moveTo(q2[0], q2[1]); });
    x.closePath(); x.fill();
    // the board (shaded by how much its normal faces up)
    const n = qrot(q, [0, 0, 1]); const lit = 0.5 + 0.5 * Math.max(0, n[2]);
    const pr = cs.map(proj);
    const col = (a) => `rgba(${Math.round(194 * lit + 30)},${Math.round(91 * lit + 20)},${Math.round(42 * lit + 14)},${a})`;
    x.beginPath(); pr.forEach((q2, i) => i ? x.lineTo(q2[0], q2[1]) : x.moveTo(q2[0], q2[1])); x.closePath();
    x.fillStyle = col(0.95); x.fill();
    x.strokeStyle = 'rgba(247,213,184,.9)'; x.lineWidth = 1.4 * devicePixelRatio; x.stroke();
    // body axes arrows from the board centre
    const o = proj([0, 0, 0]);
    const axis = (v, c2, len) => { const e = qrot(q, v).map(k => k * len); const pe = proj(e);
      x.strokeStyle = c2; x.lineWidth = 2.4 * devicePixelRatio; x.beginPath(); x.moveTo(o[0], o[1]); x.lineTo(pe[0], pe[1]); x.stroke();
      x.fillStyle = c2; x.beginPath(); x.arc(pe[0], pe[1], 3 * devicePixelRatio, 0, 7); x.fill(); };
    axis([1, 0, 0], COLS[0], 1.05); axis([0, 1, 0], COLS[1], 0.9); axis([0, 0, 1], COLS[2], 0.9);
  }

  // Static curves are cached to an offscreen canvas (c._oc); each frame we only
  // blit + draw the moving playhead + dots — so playback stays cheap.
  _plotStatic(c, series){
    const dpr = devicePixelRatio, W = c.width, H = c.height;
    if (!c._oc) c._oc = document.createElement('canvas');
    c._oc.width = W; c._oc.height = H;
    const x = c._oc.getContext('2d'); x.clearRect(0, 0, W, H);
    const T = series.length; let mx = 1e-6;
    for (const v of series) mx = Math.max(mx, Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2]));
    const pad = 4 * dpr, cy = H / 2, sc = (H / 2 - pad) / mx;
    x.strokeStyle = 'rgba(242,238,230,.16)'; x.lineWidth = dpr;
    x.beginPath(); x.moveTo(0, cy); x.lineTo(W, cy); x.stroke();
    x.strokeStyle = 'rgba(242,238,230,.06)';
    x.beginPath(); x.moveTo(0, pad); x.lineTo(W, pad); x.moveTo(0, H - pad); x.lineTo(W, H - pad); x.stroke();
    for (let k = 0; k < 3; k++){
      x.strokeStyle = COLS[k]; x.lineWidth = 1.4 * dpr; x.beginPath();
      for (let i = 0; i < T; i++){ const px = (i / (T - 1)) * W, py = cy - series[i][k] * sc; i ? x.lineTo(px, py) : x.moveTo(px, py); }
      x.stroke();
    }
    x.fillStyle = 'rgba(151,144,138,.9)'; x.font = `${8.5 * dpr}px Geist Mono, monospace`;
    x.textAlign = 'right'; x.fillText('±' + mx.toFixed(1), W - 3 * dpr, 10 * dpr);
    c._mx = mx; c._sc = sc; c._cy = cy; c._series = series;
  }
  _plot(c, series, f){
    const dpr = devicePixelRatio;
    const W = Math.max(2, c.clientWidth * dpr), H = Math.max(2, c.clientHeight * dpr);
    if (c.width !== W || c.height !== H || c._series !== series || c._dirty){ c.width = W; c.height = H; c._dirty = false; this._plotStatic(c, series); }
    const x = c.getContext('2d'); x.clearRect(0, 0, W, H);
    if (c._oc) x.drawImage(c._oc, 0, 0);
    const T = series.length; if (!T) return;
    const ph = (f / Math.max(1, T - 1)) * W;
    x.strokeStyle = 'rgba(255,255,255,.85)'; x.lineWidth = dpr;
    x.beginPath(); x.moveTo(ph, 0); x.lineTo(ph, H); x.stroke();
    for (let k = 0; k < 3; k++){ x.fillStyle = COLS[k];
      x.beginPath(); x.arc(ph, c._cy - series[f][k] * c._sc, 2.4 * dpr, 0, 7); x.fill(); }
  }

  destroy(){}
}
