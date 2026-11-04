/* Rolling strip-chart on canvas 2D — the studio family has no chart lib
   (plots are hand-rolled canvas, cf. jointpanel.js); this is the time-series
   analogue: multi-series, windowed on t, autoscaled, DPR-aware, legend. */

const COLORS = ['#f0b68b', '#3fb56b', '#7fb2ff', '#e9a679', '#d23b2a', '#b48ef0'];

export class StripChart {
  constructor(canvas, opts = {}) {
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.series = {};                    // key -> {label,color,data:[(t,v)]}
    this.order = [];
    this.windowS = opts.windowS ?? 12;
    this.yMin = opts.yMin; this.yMax = opts.yMax;   // fixed range if given
    this.refLines = opts.refLines || [];            // [{y,label,color}]
    this.title = opts.title || '';
    this.unit = opts.unit || '';
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
  }
  def(key, label, color) {
    if (!this.series[key]) {
      this.series[key] = { label, color: color ||
        COLORS[this.order.length % COLORS.length], data: [] };
      this.order.push(key);
    }
    return this;
  }
  push(key, t, v) {
    const s = this.series[key]; if (!s) return;
    if (v == null || Number.isNaN(v)) return;
    const d = s.data;
    if (d.length && t < d[d.length - 1][0] - 1) d.length = 0;  // bench restarted
    if (!d.length || t > d[d.length - 1][0]) d.push([t, v]);
    const cut = t - this.windowS;
    while (d.length && d[0][0] < cut) d.shift();
  }
  seed(key, pts) {
    const s = this.series[key]; if (!s) return;
    s.data = pts.filter(p => p[1] != null);
  }
  clear() { for (const k of this.order) this.series[k].data = []; this.draw(); }

  draw() {
    const el = this.c.parentElement; if (!el) return;
    const W = el.clientWidth, H = el.clientHeight; if (!W || !H) return;
    const dpr = Math.min(devicePixelRatio, 2);
    if (this.c.width !== W * dpr) { this.c.width = W * dpr; this.c.height = H * dpr; }
    const g = this.ctx; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const padL = 34, padR = 6, padT = 16, padB = 14;
    const iw = W - padL - padR, ih = H - padT - padB;

    // data extent
    let tMax = -Infinity, lo = Infinity, hi = -Infinity, any = false;
    for (const k of this.order) for (const [t, v] of this.series[k].data) {
      any = true; if (t > tMax) tMax = t;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    for (const r of this.refLines) { if (r.y < lo) lo = r.y; if (r.y > hi) hi = r.y; }
    if (!any) { lo = 0; hi = 1; tMax = this.windowS; }
    if (this.yMin != null) lo = this.yMin;
    if (this.yMax != null) hi = this.yMax;
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const pad = (hi - lo) * 0.08; if (this.yMin == null) lo -= pad;
    if (this.yMax == null) hi += pad;
    const t0 = tMax - this.windowS;
    const X = t => padL + ((t - t0) / this.windowS) * iw;
    const Y = v => padT + (1 - (v - lo) / (hi - lo)) * ih;

    // frame + gridlines
    g.strokeStyle = 'rgba(242,238,230,.09)'; g.lineWidth = 1;
    g.strokeRect(padL, padT, iw, ih);
    g.font = '9px "Geist Mono",monospace'; g.fillStyle = '#6b655f';
    for (let i = 0; i <= 2; i++) {
      const v = lo + (hi - lo) * i / 2, y = Y(v);
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + iw, y);
      g.strokeStyle = 'rgba(242,238,230,.05)'; g.stroke();
      g.fillText(this._fmt(v), 2, y + 3);
    }
    // reference lines (e.g. tip usage u=1)
    for (const r of this.refLines) {
      const y = Y(r.y);
      g.strokeStyle = r.color || '#d23b2a'; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + iw, y); g.stroke();
      g.setLineDash([]);
    }
    // series
    for (const k of this.order) {
      const s = this.series[k]; if (!s.data.length) continue;
      g.strokeStyle = s.color; g.lineWidth = 1.4; g.beginPath();
      let started = false;
      for (const [t, v] of s.data) {
        if (t < t0) continue;
        const x = X(t), y = Y(v);
        if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      }
      g.stroke();
    }
    // legend first (right-aligned), then the title clipped to the room left
    g.font = '9px "Geist Mono",monospace';
    let lx = W - padR;
    for (let i = this.order.length - 1; i >= 0; i--) {
      const s = this.series[this.order[i]];
      const last = s.data.length ? s.data[s.data.length - 1][1] : null;
      const txt = `${s.label} ${last == null ? '—' : this._fmt(last)}`;
      const tw = g.measureText(txt).width;
      if (lx - (tw + 14) < padL + 40) break;         // never collide with title
      lx -= tw + 14;
      g.fillStyle = s.color; g.fillRect(lx - 8, 4, 6, 6);
      g.fillStyle = '#cfc8be'; g.fillText(txt, lx, 10);
    }
    g.fillStyle = '#97908a';
    let title = this.title.toUpperCase();
    while (title.length > 3 &&
           g.measureText(title).width > lx - padL - 16) {
      title = title.slice(0, -2);
    }
    if (title !== this.title.toUpperCase()) title += '…';
    g.fillText(title, padL, 10);
  }
  _fmt(v) {
    const a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  }
}
