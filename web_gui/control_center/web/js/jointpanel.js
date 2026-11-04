// The 59-DOF telemetry band: one vertical bar per joint, grouped into the 7
// robot modules, drawn on canvas like the curation studio's bar-graph.
// Solid bar = CURRENT, peach tick = TARGET; bar color grades the tracking
// error (green / yellow / red); the strip under the hands grades actuator
// temperature. pos/vel/torque via the header seg. Hover → detail readout.
import { conf } from './config.js';
import { GROUP_LABEL } from './names.js';
import { normVal, tempStatus, trackStatus, STATUS_COL } from './grading.js';

const C = { soft: '#f0b68b', ash: '#6b655f', mid: 'rgba(242,238,230,.15)',
            ink: '#cfc8be' };

export class JointPanel {
  constructor(canvas, hintEl){
    this.canvas = canvas;
    this.hint = hintEl;
    this.mode = 'pos';
    this.schema = null;          // {names, modules}
    this.sample = null;          // latest telemetry payload
    this._hover = -1;
    this._bars = [];
    const resize = () => this.draw();
    new ResizeObserver(resize).observe(canvas.parentElement);
    window.addEventListener('studio-resize', resize);
    canvas.addEventListener('pointermove', (e) => this._onMove(e));
    canvas.addEventListener('pointerleave', () => { this._hover = -1; this._updateHint(); this.draw(); });
    // click a bar → pin its full readout (angle°, vel, torque, temp)
    canvas.addEventListener('click', () => {
      this._pinName = (this._hover >= 0 && this._bars[this._hover] &&
        this._pinName !== this._bars[this._hover].name)
        ? this._bars[this._hover].name : null;
      this._updateHint(); this.draw();
    });
  }

  setSchema(s){
    this.schema = s;
    // flat draw list: [{name, module, label, span}]
    this._flat = [];
    for (const m of s.modules){
      for (const n of m.joints)
        this._flat.push({ name: n, module: m.key, mLabel: m.label });
    }
    this.draw();
  }
  setMode(m){ this.mode = m; this.draw(); }
  update(sample){ this.sample = sample; this.draw(); }

  _chan(key){ return this.sample?.[key] || []; }
  _idx(name){ return this.schema ? this.schema.names.indexOf(name) : -1; }

  draw(){
    const cv = this.canvas, host = cv.parentElement;
    if (!this.schema || !host) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const W = host.clientWidth, H = cv.clientHeight || (host.clientHeight - 34);
    if (W < 10 || H < 10) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, W, H);

    const flat = this._flat;
    const n = flat.length;
    const padL = 6, padR = 6, gGap = 14;
    const groups = this.schema.modules;
    const innerW = W - padL - padR - gGap * (groups.length - 1);
    const bw = innerW / n;
    const tempH = 5;                   // temperature strip height
    const labH = 13;
    const midY = (H - labH - tempH) * 0.52 + 2;
    const amp = midY - 8;

    const p = this._chan('p'), v = this._chan('v'), e = this._chan('e'),
          tgt = this._chan('tgt'), temp = this._chan('temp');
    const cur = this.mode === 'pos' ? p : this.mode === 'vel' ? v : e;
    const cfgT = conf();

    // center line
    x.strokeStyle = C.mid; x.lineWidth = 1;
    x.beginPath(); x.moveTo(padL, midY); x.lineTo(W - padR, midY); x.stroke();

    this._bars = [];
    let cx = padL, gi = 0;
    for (const m of groups){
      const gx0 = cx;
      for (const jn of m.joints){
        const i = this.schema.names.indexOf(jn);
        const cv_ = cur[i], tv = this.mode === 'pos' ? tgt[i] : null;
        const st = trackStatus(cv_, tv, this.mode);
        const hn = normVal(cv_, this.mode);
        const bh = hn * amp;
        const bx = cx + bw * 0.15, bws = Math.max(1.5, bw * 0.7);
        // measured bar
        let fill;
        if (st == null) fill = cv_ == null ? 'rgba(242,238,230,.10)' : C.ink;
        else fill = STATUS_COL[st];
        if (st === 0){
          const grad = x.createLinearGradient(0, midY - Math.abs(bh), 0, midY);
          grad.addColorStop(0, '#7ee6b0'); grad.addColorStop(1, '#3ecf8e');
          fill = grad;
        }
        x.fillStyle = fill;
        if (bh >= 0) x.fillRect(bx, midY - bh, bws, Math.max(1, bh));
        else x.fillRect(bx, midY, bws, Math.max(1, -bh));
        // target tick
        if (tv != null){
          const th = normVal(tv, this.mode) * amp;
          x.fillStyle = C.soft;
          x.fillRect(bx - 1, midY - th - 1, bws + 2, 2);
        }
        // temperature cell
        const ts = tempStatus(temp[i], cfgT.tempWarn, cfgT.tempHot);
        x.fillStyle = ts == null ? 'rgba(242,238,230,.06)' : STATUS_COL[ts];
        x.fillRect(bx, H - labH - tempH, bws, tempH - 1);
        // hover / pinned highlight
        if (this._bars.length === this._hover || jn === this._pinName){
          x.strokeStyle = jn === this._pinName ? 'rgba(240,182,139,.9)'
            : 'rgba(242,238,230,.7)';
          x.strokeRect(cx + 0.5, 4, bw - 1, H - labH - 6);
        }
        this._bars.push({ x0: cx, x1: cx + bw, name: jn, i });
        cx += bw;
      }
      // group divider + label. In the narrow right dock the modules are only a
      // few bars wide, so full labels ran into each other and rendered as one
      // unreadable word ("BASELIFT/TORSONECK"). Draw the label only if it fits
      // its own group's width; otherwise drop it — the dividers still delimit
      // the groups, and no label beats a smeared one.
      x.fillStyle = C.ash;
      x.font = '9px Geist Mono, monospace';
      x.textAlign = 'center';
      const lab = (GROUP_LABEL[m.key] || m.label).toUpperCase();
      if (x.measureText(lab).width <= (cx - gx0) - 2)
        x.fillText(lab, (gx0 + cx) / 2, H - 3);
      if (gi < groups.length - 1){
        x.strokeStyle = 'rgba(242,238,230,.08)';
        x.beginPath(); x.moveTo(cx + gGap / 2, 2); x.lineTo(cx + gGap / 2, H - labH); x.stroke();
      }
      cx += gGap; gi++;
    }
  }

  _onMove(e){
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const hit = this._bars.findIndex(b => px >= b.x0 && px < b.x1);
    if (hit !== this._hover){ this._hover = hit; this._updateHint(); this.draw(); }
  }

  _updateHint(){
    if (!this.hint) return;
    const b = (this._hover >= 0 && this._bars[this._hover])
      || this._bars.find?.((x) => x.name === this._pinName)
      || null;
    if (!b || !this.sample){
      this.hint.textContent = 'hover a bar → detail · click to pin';
      return;
    }
    const s = this.sample;
    const deg = (v) => (v == null ? '—' : (v * 180 / Math.PI).toFixed(1) + '°');
    const num = (a) => (a?.[b.i] == null ? '—' : a[b.i].toFixed(3));
    const t = s.temp?.[b.i];
    this.hint.textContent =
      `${b.name}${b.name === this._pinName ? ' 📌' : ''} · ` +
      `angle ${deg(s.p?.[b.i])} → ${deg(s.tgt?.[b.i])} (${num(s.p)} rad) · ` +
      `vel ${num(s.v)} rad/s · torque ${num(s.e)} N·m · ` +
      `temp ${t != null ? t.toFixed(0) + '°C' : '—'}`;
  }
}
