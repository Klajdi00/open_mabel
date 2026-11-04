// Live metric plots (loss / sub-losses / lr / grad-norm / throughput / system).
// Canvas multi-series line chart fed incrementally by the training WebSocket.

const MODES = {
  loss:       { title: 'Loss', keys: ['loss', 'eval_loss'], mode: 'train' },   // train + eval on one plot
  sublosses:  { title: 'Sub-losses', keys: ['l1_loss', 'kld_loss', 'kld', 'reconstruction_loss'], mode: 'train' },
  lr:         { title: 'Learning rate', keys: ['lr'], mode: 'train' },
  grad:       { title: 'Grad norm', keys: ['grad_norm'], mode: 'train' },
  throughput: { title: 'Throughput', keys: ['update_s', 'dataloading_s'], mode: 'train', byTime: true },
  system:     { title: 'GPU / CPU', keys: ['system.gpu_util', 'system.cpu_percent', 'system.gpu_mem_used_gb', 'system.ram_percent'], mode: 'system', byTime: true },
};
// human-readable legend labels (so it's obvious which line is which)
const KEY_LABEL = {
  loss: 'training loss', eval_loss: 'validation loss',
  l1_loss: 'L1 (recon)', kld_loss: 'KL divergence', kld: 'KL divergence', reconstruction_loss: 'reconstruction',
  lr: 'learning rate', grad_norm: 'grad norm',
  update_s: 'update (s/it)', dataloading_s: 'dataloading (s)',
  'system.gpu_util': 'GPU util %', 'system.cpu_percent': 'CPU %',
  'system.gpu_mem_used_gb': 'GPU mem (GB)', 'system.ram_percent': 'RAM %',
};
// fixed, distinct colours per metric so every line is clearly colour-graded
const KEY_COL = {
  loss: '#E8973C', eval_loss: '#4FB39C',                 // train=rust, val=teal
  l1_loss: '#E8973C', kld_loss: '#A97FCB', kld: '#A97FCB', reconstruction_loss: '#5FB85A',
  lr: '#6FA8DC', grad_norm: '#E070A8',
  update_s: '#E8973C', dataloading_s: '#6FA8DC',
  'system.gpu_util': '#5FB85A', 'system.cpu_percent': '#E8973C',
  'system.gpu_mem_used_gb': '#B084E0', 'system.ram_percent': '#6FA8DC',
};
const COL = (key, i) => KEY_COL[key] || `hsl(${(i * 67 + 20) % 360} 68% 62%)`;
const LABEL = (key) => KEY_LABEL[key] || key.replace('system.', '');

export class MetricsPlot {
  constructor(canvas, headSeg, titleEl) {
    this.cv = canvas; this.titleEl = titleEl; this.seg = headSeg;
    this.series = new Map();     // "mode:key" -> [{x, value}]
    this.mode = 'loss';
    headSeg.innerHTML = Object.entries(MODES)
      .map(([k, v], i) => `<button data-m="${k}" class="${i === 0 ? 'on' : ''}">${v.title}</button>`).join('');
    headSeg.onclick = (e) => {
      const b = e.target.closest('button'); if (!b) return;
      this.setMode(b.dataset.m);
    };
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas.parentElement);
  }

  setMode(mode) {
    if (!MODES[mode]) return;
    this.mode = mode;
    if (this.seg) this.seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.m === mode));
    this.draw();
  }
  destroy() { try { this._ro.disconnect(); } catch {} }

  reset() { this.series.clear(); this.draw(); }

  addRows(rows) {
    for (const r of rows) {
      const k = `${r.mode}:${r.key}`;
      const arr = this.series.get(k) || this.series.set(k, []).get(k);
      arr.push({ x: r.mode === 'system' ? r.t : r.step, value: r.value });
    }
    this.draw();
  }

  draw() {
    const cfg = MODES[this.mode];
    if (this.titleEl) this.titleEl.textContent = 'Training Metrics · ' + cfg.title;
    const cv = this.cv, r = Math.min(window.devicePixelRatio || 1, 2);
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * r; cv.height = H * r;
    const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = { l: 44, r: 10, t: 10, b: 18 };

    // sub-losses are ARCH-dependent (ACT: L1+KL; DiT: single denoising MSE, none)
    // → discover whatever sub-loss series actually exist rather than hard-coding.
    let keys = cfg.keys;
    if (this.mode === 'sublosses') {
      keys = [...this.series.keys()].filter(k => k.startsWith('train:')).map(k => k.slice(6))
        .filter(k => k !== 'loss' && k !== 'eval_loss' && /(loss|kld|kl|recon|entropy|bc|mse|vlb)/i.test(k));
    }
    const active = [];
    keys.forEach((key, i) => {
      const arr = this.series.get(`${cfg.mode}:${key}`);
      if (arr && arr.length) active.push({ key, arr, color: COL(key, i), label: LABEL(key) });
    });
    // top legend area reserved so curves never overlap the labels
    pad.t = 22;
    if (!active.length) {
      ctx.fillStyle = 'rgba(151,144,138,.55)'; ctx.font = '11px Geist'; ctx.textAlign = 'center';
      const ended = window.__studioState && window.__studioState.runEnded;
      const msg = this.mode === 'sublosses'
        ? 'no sub-losses — this policy trains on a single loss'
        : ended
          ? `no ${cfg.title.toLowerCase()} was recorded before this job ended — check its Logs`
          : 'waiting for ' + cfg.title.toLowerCase() + ' data…';
      ctx.fillText(msg, W / 2, H / 2); return;
    }
    let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
    for (const s of active) for (const p of s.arr) {
      mnX = Math.min(mnX, p.x); mxX = Math.max(mxX, p.x);
      mnY = Math.min(mnY, p.value); mxY = Math.max(mxY, p.value);
    }
    if (mxX === mnX) mxX += 1;
    if (mxY === mnY) { mxY += 1; mnY -= 1; }
    const px = (x) => pad.l + (x - mnX) / (mxX - mnX) * (W - pad.l - pad.r);
    const py = (v) => H - pad.b - (v - mnY) / (mxY - mnY) * (H - pad.t - pad.b);

    // gridlines + y labels
    ctx.strokeStyle = 'rgba(242,238,230,.07)'; ctx.fillStyle = 'rgba(151,144,138,.7)';
    ctx.font = '9px "Geist Mono", monospace'; ctx.textAlign = 'right';
    for (let g = 0; g <= 3; g++) {
      const v = mnY + (mxY - mnY) * g / 3, y = py(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillText(this._fmt(v), pad.l - 4, y + 3);
    }
    // x-axis (the horizon): baseline + step/time ticks along the bottom
    const byTime = !!cfg.byTime;
    const xlab = byTime ? 's' : 'step';
    ctx.strokeStyle = 'rgba(242,238,230,.16)'; ctx.beginPath();
    ctx.moveTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b); ctx.stroke();
    ctx.strokeStyle = 'rgba(242,238,230,.05)';
    ctx.fillStyle = 'rgba(151,144,138,.7)'; ctx.textAlign = 'center';
    for (let g = 0; g <= 4; g++) {
      const xv = mnX + (mxX - mnX) * g / 4, X = px(xv);
      ctx.beginPath(); ctx.moveTo(X, pad.t); ctx.lineTo(X, H - pad.b); ctx.stroke();
      const lbl = byTime ? (+xv.toFixed(0)) : this._fmtX(xv);
      ctx.fillStyle = 'rgba(151,144,138,.7)'; ctx.fillText(String(lbl), X, H - pad.b + 11);
    }
    ctx.fillStyle = 'rgba(151,144,138,.5)'; ctx.textAlign = 'right';
    ctx.fillText(xlab, W - pad.r, H - 2);
    // curves
    for (const s of active) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
      s.arr.forEach((p, i) => { const X = px(p.x), Y = py(p.value); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
      const last = s.arr[s.arr.length - 1];
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(px(last.x), py(last.value), 2.4, 0, 7); ctx.fill();
    }
    // legend — line swatch + friendly name + current value, always visible
    ctx.textAlign = 'left'; ctx.font = '9.5px "Geist Mono", monospace';
    let lx = pad.l + 2;
    for (const s of active) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(lx, 9); ctx.lineTo(lx + 12, 9); ctx.stroke();
      ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(lx + 6, 9, 2, 0, 7); ctx.fill();
      const last = s.arr[s.arr.length - 1];
      const txt = `${s.label}  ${this._fmt(last.value)}`;
      ctx.fillStyle = 'rgba(214,208,198,.95)'; ctx.fillText(txt, lx + 17, 12);
      lx += 26 + ctx.measureText(txt).width;
    }
  }
  _fmt(v) {
    if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
    return (+v.toFixed(3)).toString();
  }
  _fmtX(v) {   // compact step labels: 1.2k, 34k, …
    if (Math.abs(v) >= 1000) return (+(v / 1000).toFixed(1)) + 'k';
    return String(Math.round(v));
  }
}
