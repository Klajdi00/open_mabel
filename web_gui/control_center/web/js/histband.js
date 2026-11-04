// TRENDS — the rolling joint-history line graph, as a permanent band beside the
// camera strip and the joint bars.
//
// The web twin of the apps' "Trends" panel (the third of the visionOS
// `Cams · Joints · Trends` quick toggles — VisionOS_V2 RobotSpaceView.swift).
// The bars answer "where is every joint NOW"; this answers "what did it just
// DO" — the question you actually ask when a joint sags, ratchets, or a target
// and its measurement diverge.
//
// It reads the SAME TelemetryHistory ring buffer the floating Joint-History
// window uses (history.js, 15 Hz since page load, 5 min capacity). Default view
// = the last 2 minutes, the window the apps default to.
//
// Module selection is MULTI-SELECT CHIPS (iOS `JointGraphs` parity), not the
// old single `<select>`: comparing an arm against its own hand is the whole
// point of a trend view. At least one chip always stays on — an empty chart
// reads as broken, not as "everything hidden".
//
// Colour is the legend: one hue per module (shared with the chips), and within
// a module each joint fades along a brightness ramp so its curves stay
// separable without a 20-row key. Solid line = measured, dashed = target
// (pos only — vel/torque have no target stream).

import { DEFAULT_WINDOW_S, MAX_WINDOW_S } from './history.js';
import { MODULE_HUE, GROUP_LABEL } from './names.js';

export class HistBand {
  /**
   * @param {HTMLElement} el      the band container (#histBand)
   * @param {TelemetryHistory} history
   */
  constructor(el, history){
    this.el = el;
    this.history = history;
    this.schema = { names: [], modules: [] };
    this.picked = new Set();
    this.chan = 'p';
    this.windowS = DEFAULT_WINDOW_S;
    this.visible = false;

    el.classList.add('hist-band');
    el.innerHTML = `
      <div class="hb-bar">
        <span class="hb-title mono">Trends</span>
        <div class="seg hb-ch">
          <button class="on" data-c="p">pos</button>
          <button data-c="v">vel</button>
          <button data-c="e">torque</button>
        </div>
        <label class="hb-win mono">window
          <input type="range" min="30" max="${MAX_WINDOW_S}" step="10"
                 value="${DEFAULT_WINDOW_S}" aria-label="history window">
          <b class="mono">${DEFAULT_WINDOW_S}s</b></label>
        <span class="hb-legend mono"><i class="meas"></i>measured&nbsp;<i class="tgt"></i>target</span>
      </div>
      <div class="hb-chips" role="group" aria-label="modules"></div>
      <canvas></canvas>`;

    this.chips = el.querySelector('.hb-chips');
    this.chips.onclick = (e) => {
      const b = e.target.closest('button'); if (!b) return;
      const k = b.dataset.k;
      // multi-select, but never let the last one off
      if (this.picked.has(k)){
        if (this.picked.size === 1) return;
        this.picked.delete(k);
      } else this.picked.add(k);
      this._syncChips();
      this.draw();
    };
    el.querySelector('.hb-ch').onclick = (e) => {
      const b = e.target.closest('button'); if (!b) return;
      el.querySelectorAll('.hb-ch button').forEach((x) => x.classList.toggle('on', x === b));
      this.chan = b.dataset.c;
      this.draw();
    };
    const win = el.querySelector('.hb-win input'), wv = el.querySelector('.hb-win b');
    win.oninput = () => {
      this.windowS = +win.value;
      wv.textContent = win.value + 's';
      this.draw();
    };
    this.canvas = el.querySelector('canvas');
    // 4 Hz: the buffer fills at 15 Hz, but a 2-minute line at 4 Hz is already
    // smoother than the pixels can show — and this band shares the frame with
    // a 3-D view and four camera decoders.
    this._timer = setInterval(() => { if (this.visible) this.draw(); }, 250);
    window.addEventListener('studio-resize', () => this.draw());
  }

  setSchema(schema){
    this.schema = schema || { names: [], modules: [] };
    const mods = this.schema.modules || [];
    this.chips.innerHTML = '';
    for (const m of mods){
      const b = document.createElement('button');
      b.dataset.k = m.key;
      b.innerHTML = `<i></i>${GROUP_LABEL[m.key] || m.label || m.key}`;
      b.style.setProperty('--hue', MODULE_HUE[m.key] || '#8a8178');
      this.chips.appendChild(b);
    }
    // keep the operator's picks across a schema refresh; otherwise open on the
    // first module the robot actually reports
    for (const k of [...this.picked]) if (!mods.some((m) => m.key === k)) this.picked.delete(k);
    if (!this.picked.size && mods[0]) this.picked.add(mods[0].key);
    this._syncChips();
    this.draw();
  }

  _syncChips(){
    this.chips.querySelectorAll('button').forEach((b) => {
      const on = this.picked.has(b.dataset.k);
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  show(on){
    this.visible = !!on;
    this.el.style.display = on ? '' : 'none';
    if (on) this.draw();
  }

  draw(){
    if (!this.visible) return;
    const cv = this.canvas;
    // measure the CANVAS, not the band: the chip row is a sibling, so the
    // band's height is not the plot's height
    const W = cv.clientWidth || this.el.clientWidth, H = cv.clientHeight || 0;
    if (W < 20 || H < 16) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, W, H);

    const note = (t) => {
      x.fillStyle = '#6b655f';
      x.font = '11px Geist Mono, monospace';
      x.textAlign = 'left';
      x.fillText(t, 12, H / 2);
    };
    const mods = (this.schema.modules || []).filter((m) => this.picked.has(m.key));
    if (!mods.length){ note('waiting for telemetry'); return; }

    // gather every picked module's joints, coloured by module hue + a
    // brightness ramp across that module's own joints
    const series = [];
    let lo = Infinity, hi = -Infinity, derived = false;
    for (const m of mods){
      const joints = m.joints || [];
      joints.forEach((jn, k) => {
        const ji = this.schema.names.indexOf(jn);
        if (ji < 0) return;
        const s = this.history.series(ji, this.chan, this.windowS);
        const tg = this.chan === 'p' ? this.history.series(ji, 'tgt', this.windowS) : null;
        if (s.derived) derived = true;
        for (const v of s.v) if (Number.isFinite(v)){ lo = Math.min(lo, v); hi = Math.max(hi, v); }
        if (tg) for (const v of tg.v) if (Number.isFinite(v)){ lo = Math.min(lo, v); hi = Math.max(hi, v); }
        const fade = joints.length > 1 ? 0.55 * k / (joints.length - 1) : 0;
        series.push({ name: jn, s, tg, hue: MODULE_HUE[m.key] || '#8a8178', alpha: 1 - fade });
      });
    }
    // Honest empty states — say WHICH thing is missing rather than draw a blank
    // pane. "no history yet" and "the robot never publishes this channel" are
    // different faults with different fixes.
    if (!Number.isFinite(lo)){
      note(this.history.n === 0
        ? 'no telemetry yet — waiting for the robot'
        : this.chan === 'p'
          ? 'no position history for these modules yet'
          : `the robot is not publishing ${this.chan === 'v' ? 'velocity' : 'torque'} for these joints`);
      return;
    }
    if (hi - lo < 1e-6){ hi += 0.5; lo -= 0.5; }

    const PADL = 34, PADR = 8;
    const t1 = performance.now() / 1000;
    const px = (t) => PADL + (1 - (t1 - t) / this.windowS) * (W - PADL - PADR);
    const py = (v) => H - 8 - (v - lo) / (hi - lo) * (H - 18);

    // baseline + scale rules (a torque plot is unreadable without a zero line)
    x.strokeStyle = 'rgba(242,238,230,.10)';
    x.lineWidth = 1;
    for (const gv of [lo, (lo + hi) / 2, hi]){
      const Y = Math.round(py(gv)) + 0.5;
      x.beginPath(); x.moveTo(PADL, Y); x.lineTo(W - PADR, Y); x.stroke();
    }
    if (lo < 0 && hi > 0){
      const Y = Math.round(py(0)) + 0.5;
      x.strokeStyle = 'rgba(242,238,230,.24)';
      x.beginPath(); x.moveTo(PADL, Y); x.lineTo(W - PADR, Y); x.stroke();
    }

    x.save();
    x.beginPath(); x.rect(PADL, 0, W - PADL - PADR, H); x.clip();
    for (const sr of series){
      const line = (data, dash) => {
        if (!data || !data.t.length) return;
        x.strokeStyle = sr.hue;
        x.globalAlpha = dash ? sr.alpha * 0.55 : sr.alpha;
        x.setLineDash(dash ? [4, 4] : []);
        x.lineWidth = dash ? 1 : 1.3;
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
        x.globalAlpha = 1;
      };
      line(sr.s, false);
      line(sr.tg, true);
    }
    x.restore();

    x.fillStyle = '#6b655f';
    x.font = '9px Geist Mono, monospace';
    x.textAlign = 'right';
    x.fillText(hi.toFixed(2), PADL - 5, 10);
    x.fillText(lo.toFixed(2), PADL - 5, H - 4);
    x.textAlign = 'left';
    x.fillText(`−${this.windowS}s`, PADL + 2, H - 4);
    x.textAlign = 'right';
    x.fillText('now', W - PADR, H - 4);
    // a differentiated velocity trace must SAY so — it is noisier than a
    // streamed one, and reading encoder noise as real motion wastes an hour
    if (derived && this.chan === 'v'){
      x.fillStyle = '#8a8178';
      x.fillText('d(pos)/dt', W - PADR, 10);
    }
    x.textAlign = 'left';
  }

  destroy(){ clearInterval(this._timer); }
}
