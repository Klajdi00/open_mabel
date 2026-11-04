// Add-able windows. The +Window menu offers TRAINING PLOTS (metrics, GPU/CPU,
// eval, I/O graphs, log); the robot/deploy 3D view is auto-docked on the LEFT by
// app.js and excluded from the menu. All plot windows are fed by the shared
// metric bus so they update live during training.
import { MetricsPlot } from './metrics.js';
import * as bus from './bus.js';

export function registerPanels(panels, ctx) {
  const { api, state, toast } = ctx;

  // ── live metrics (loss / sub-losses / lr / grad / throughput / gpu-cpu) ────
  panels.register('metrics', 'Loss & metrics', (panel) => {
    panel.body.innerHTML = '<div class="seg pw-seg"></div><canvas class="pw-cv"></canvas>';
    const plot = new MetricsPlot(panel.body.querySelector('.pw-cv'), panel.body.querySelector('.pw-seg'), null);
    const off1 = bus.on('metricRows', (r) => plot.addRows(r));
    const off2 = bus.on('metricReset', () => plot.reset());
    return { resize: () => plot.draw(), destroy: () => { off1(); off2(); } };
  });

  // ── GPU / CPU gauges ───────────────────────────────────────────────────────
  panels.register('sysmon', 'GPU / CPU', (panel) => {
    panel.body.innerHTML = '<div class="mon-wrap"></div>';
    const wrap = panel.body.querySelector('.mon-wrap');
    const keys = [['system.gpu_util', 'GPU util', '%'], ['system.gpu_mem_used_gb', 'GPU mem', 'GB'],
      ['system.cpu_percent', 'CPU', '%'], ['system.ram_percent', 'RAM', '%'],
      ['system.gpu_temp', 'GPU °C', '°'], ['system.gpu_power_w', 'GPU W', 'W']];
    async function tick() {
      if (!state.run) { wrap.innerHTML = '<div class="mon-empty">no active run</div>'; return; }
      const v = {};
      for (const [k] of keys) { try { const { series } = await api.series(state.run, k, 'system'); if (series.length) v[k] = series[series.length - 1].value; } catch {} }
      wrap.innerHTML = keys.filter(([k]) => v[k] != null).map(([k, label, u]) => {
        const pct = u === '%' ? Math.min(100, v[k]) : null;
        return `<div class="gauge"><div class="g-label">${label}</div>
          <div class="g-val">${(+v[k]).toFixed(u === '%' ? 0 : 2)}<i>${u}</i></div>
          ${pct != null ? `<div class="g-bar"><div style="width:${pct}%"></div></div>` : ''}</div>`;
      }).join('') || '<div class="mon-empty">waiting for system metrics…</div>';
    }
    const iv = setInterval(tick, 2000); tick();
    return { resize() {}, destroy() { clearInterval(iv); } };
  });

  // ── training + deployment logs (raw terminal, for debugging) ───────────────
  const logWindow = (label, backlogKey, rowsEv, resetEv) => (panel) => {
    panel.body.innerHTML = '<div class="log-scroll pw-log"></div>';
    const el = panel.body.querySelector('.pw-log');
    const render = (lines) => {
      const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
      el.insertAdjacentHTML('beforeend', lines.map(l => {
        const s = l.line != null ? l.line : l;
        const cls = l.cls || (/error|fail|exception|traceback/i.test(s) ? 'err' : /warn|nan/i.test(s) ? 'warn'
          : /loss|step|checkpoint|run |done|complete/i.test(s) ? 'ok' : '');
        return `<div class="log-line ${cls}">${String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))}</div>`;
      }).join(''));
      if (atBottom) el.scrollTop = el.scrollHeight;
    };
    render(window[backlogKey] || []);
    const off1 = bus.on(rowsEv, render);
    const off2 = resetEv ? bus.on(resetEv, () => { el.innerHTML = ''; }) : () => {};
    return { resize() {}, destroy() { off1(); off2(); } };
  };
  panels.register('log', 'Training log', logWindow('Training log', '__logBuffer', 'logRows', 'logReset'));
  panels.register('deploylog', 'Deployment log', logWindow('Deployment log', '__depLog', 'depLogRows', null));

  // ── eval: predicted vs ground truth ────────────────────────────────────────
  panels.register('eval', 'Eval · predicted vs GT', (panel) => {
    panel.body.innerHTML = `<div class="ev-head"><button class="mini prim" id="ev-run">Run eval</button>
      <select id="ev-joint"></select><span class="ev-stat" id="ev-stat"></span></div><canvas class="ev-cv"></canvas>`;
    const cv = panel.body.querySelector('.ev-cv'); let data = null;
    const draw = () => {
      if (!data) return; const ji = +panel.body.querySelector('#ev-joint').value || 0;
      const r = Math.min(devicePixelRatio, 2), W = cv.clientWidth, H = cv.clientHeight;
      cv.width = W * r; cv.height = H * r; const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
      const gt = data.gt.map(x => x[ji]), pd = data.pred.map(x => x[ji]); const all = gt.concat(pd);
      let mn = Math.min(...all), mx = Math.max(...all); if (mx === mn) { mx += 1; mn -= 1; }
      const px = i => 6 + i / (gt.length - 1) * (W - 12), py = v => H - 12 - (v - mn) / (mx - mn) * (H - 20);
      const line = (a, c) => { ctx.strokeStyle = c; ctx.lineWidth = 1.6; ctx.beginPath(); a.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v))); ctx.stroke(); };
      line(gt, 'rgba(151,144,138,.9)'); line(pd, '#C25B2A');
      ctx.font = '10px "Geist Mono"'; ctx.fillStyle = 'rgba(151,144,138,.9)'; ctx.textAlign = 'left'; ctx.fillText('GT', 8, 12); ctx.fillStyle = '#C25B2A'; ctx.fillText('pred (NN)', 30, 12);
    };
    panel.body.querySelector('#ev-run').onclick = async () => {
      if (!state.run) return toast('select/launch a run first', 'err');
      panel.body.querySelector('#ev-stat').textContent = 'running…';
      try {
        const { job } = await api.evalRun(state.run, { max_frames: 200 }); data = await api.pollJob(job);
        const sel = panel.body.querySelector('#ev-joint');
        sel.innerHTML = Array.from({ length: data.action_dim }, (_, i) => `<option value="${i}">joint ${i}</option>`).join(''); sel.onchange = draw;
        panel.body.querySelector('#ev-stat').textContent = `L1 ${data.l1_mean.toFixed(3)} · MSE ${data.mse_mean.toFixed(3)} · ${data.frames}f`; draw();
      } catch (e) { panel.body.querySelector('#ev-stat').textContent = 'error: ' + e.message; }
    };
    return { resize: draw, destroy() {} };
  });

  // ── I/O graphs (input camera + proprio + action) ───────────────────────────
  panels.register('ioviz', 'I/O · input & output', (panel) => {
    panel.body.innerHTML = `<div class="io-head"><select id="io-ds"></select><span class="ev-stat" id="io-stat">pick a dataset</span></div>
      <div class="io-body"><img class="io-cam" id="io-cam"><canvas class="io-cv" id="io-cv"></canvas></div>`;
    const dsSel = panel.body.querySelector('#io-ds'), img = panel.body.querySelector('#io-cam'), cv = panel.body.querySelector('#io-cv');
    let arrays = null, ds = null, cam = null, frame = 0, iv = null;
    api.datasets().then(({ datasets }) => { const o = datasets.filter(d => !d.error); dsSel.innerHTML = o.map(d => `<option value="${d.dataset_id}" data-cam="${d.cameras[0]}">${d.dataset_id}</option>`).join(''); if (o.length) load(); });
    dsSel.onchange = load;
    async function load() { ds = dsSel.value; cam = dsSel.selectedOptions[0].dataset.cam; arrays = await api.epArrays(ds, 0); frame = 0; panel.body.querySelector('#io-stat').textContent = `${cam} · ${arrays.action.length}f`; clearInterval(iv); iv = setInterval(tick, 100); }
    function tick() {
      if (!arrays) return; const T = arrays.action.length; frame = (frame + 1) % T; img.src = api.frameUrl(ds, 0, cam, frame);
      const r = Math.min(devicePixelRatio, 2), W = cv.clientWidth, H = cv.clientHeight; cv.width = W * r; cv.height = H * r; const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
      const draw = (series, col, y0, y1, label) => {
        const dim = series[0].length; let mn = 1e9, mx = -1e9; for (const row of series) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); } if (mx === mn) { mx += 1; mn -= 1; }
        for (let dd = 0; dd < Math.min(dim, 12); dd++) { ctx.strokeStyle = col.replace('A', (0.4 + 0.5 * dd / dim).toFixed(2)); ctx.lineWidth = 1; ctx.beginPath(); series.forEach((row, i) => { const x = 4 + i / (series.length - 1) * (W - 8), y = y1 - (row[dd] - mn) / (mx - mn) * (y1 - y0); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); }
        const fx = 4 + frame / (series.length - 1) * (W - 8); ctx.strokeStyle = '#fff'; ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(fx, y0); ctx.lineTo(fx, y1); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.fillStyle = 'rgba(151,144,138,.9)'; ctx.font = '9px "Geist Mono"'; ctx.textAlign = 'left'; ctx.fillText(label, 6, y0 + 10);
      };
      const mid = H / 2; draw(arrays['observation.state'], 'hsla(200,60%,60%,A)', 6, mid - 4, 'proprio in'); draw(arrays.action, 'hsla(20,70%,60%,A)', mid + 4, H - 4, 'action out');
    }
    return { resize() {}, destroy() { clearInterval(iv); } };
  });
}
