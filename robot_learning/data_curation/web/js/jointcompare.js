// Multi-joint comparison panel (a Visualize panel). Holds one or more INDEPENDENT
// sub-plots — "+ split view" adds another — so you can compare e.g. left-hand
// joints beside right-hand joints without overlaying them in one plot. Each
// sub-plot has its own pos/vel/torque mode, group-add shortcuts, and joint set,
// all synced to the playhead.
import { GROUP_ORDER, GROUP_LABEL } from './names.js';

const TOPIC = { pos: 'joint_pos', vel: 'joint_vel', torque: 'joint_torque' };
const color = (i) => `hsl(${(i * 47) % 360},72%,62%)`;

export function jointCompareFactory(panel){
  panel.body.innerHTML = `<div class="jc-root"><div class="jc-bar"><button class="jc-split">+ split view</button>
    <span class="jc-hint">add joints by group · drag panel edge to resize</span></div>
    <div class="jc-grid"></div></div>`;
  const grid = panel.body.querySelector('.jc-grid');
  panel.body.querySelector('.jc-split').onclick = () => { addSub(); };
  let entry = null, frame = 0;
  const subs = [];

  function addSub(preset){
    const el = document.createElement('div'); el.className = 'jc-sub';
    el.innerHTML = `<div class="jc-ctl">
      <div class="seg jc-mode"><button class="on" data-m="pos">pos</button><button data-m="vel">vel</button><button data-m="torque">torque</button></div>
      <div class="jc-groups"></div>
      <button class="jc-clear">clear</button>${subs.length ? '<button class="jc-remove" title="remove sub-plot">✕</button>' : ''}</div>
      <canvas class="jc-canvas"></canvas><div class="jc-legend"></div>`;
    grid.appendChild(el);
    const sub = { el, sel: new Set(), mode: 'pos', cv: el.querySelector('.jc-canvas'),
                  legend: el.querySelector('.jc-legend'), groupsEl: el.querySelector('.jc-groups') };
    el.querySelector('.jc-mode').onclick = (e) => { const b = e.target.closest('button'); if (!b) return;
      el.querySelectorAll('.jc-mode button').forEach(x => x.classList.toggle('on', x === b)); sub.mode = b.dataset.m; sub.dirty = true; draw(sub); };
    el.querySelector('.jc-clear').onclick = () => { sub.sel.clear(); sub.dirty = true; draw(sub); };
    el.querySelector('.jc-remove')?.addEventListener('click', () => { grid.removeChild(el); subs.splice(subs.indexOf(sub), 1); });
    sub.cv.addEventListener('pointerdown', (e) => scrub(e, sub));
    new ResizeObserver(() => draw(sub)).observe(el);   // redraw when the user drag-resizes the sub-plot
    subs.push(sub); buildGroups(sub); if (preset) preset.forEach(i => sub.sel.add(i)); draw(sub);
    return sub;
  }

  function buildGroups(sub){
    if (!entry) return;
    const groups = entry.meta.actuator_groups || {}, names = entry.meta.actuator_names || [];
    sub.groupsEl.innerHTML = '';
    for (const g of GROUP_ORDER){
      if (!groups[g]) continue;
      const idx = groups[g].map(n => names.indexOf(n)).filter(i => i >= 0);
      const b = document.createElement('button'); b.className = 'jc-grp'; b.textContent = '+ ' + (GROUP_LABEL[g] || g);
      b.onclick = () => { idx.forEach(i => sub.sel.add(i)); sub.dirty = true; draw(sub); };
      sub.groupsEl.appendChild(b);
    }
  }

  // Render the (static) curves + axes once into an offscreen canvas. Re-run only
  // when the data / selection / mode / size changes — NOT every frame.
  function renderStatic(sub){
    const dpr = devicePixelRatio, W = sub.cv.width, H = sub.cv.height;
    if (!sub.oc) sub.oc = document.createElement('canvas');
    sub.oc.width = W; sub.oc.height = H;
    const ctx = sub.oc.getContext('2d'); ctx.clearRect(0, 0, W, H);
    sub.idxs = []; sub.t = null; sub.dirty = false;
    if (!entry) return;
    const t = entry.bundle.topics[TOPIC[sub.mode]]; if (!t) return;
    const T = entry.T, dim = t.dim, idxs = [...sub.sel].filter(i => i < dim);
    let mn = Infinity, mx = -Infinity;
    for (const j of idxs) for (let f = 0; f < T; f++){ const v = t.data[f * dim + j]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (!isFinite(mn)){ mn = -1; mx = 1; } if (mx - mn < 1e-6){ mx += 1; mn -= 1; }
    const pad = 6 * dpr, Y = (v) => H - pad - ((v - mn) / (mx - mn)) * (H - 2 * pad);
    if (mn < 0 && mx > 0){ ctx.strokeStyle = 'rgba(242,238,230,.12)'; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke(); }
    for (const j of idxs){ ctx.strokeStyle = color(j); ctx.lineWidth = 1.3 * dpr; ctx.beginPath();
      for (let f = 0; f < T; f++){ const px = (f / Math.max(1, T - 1)) * W, py = Y(t.data[f * dim + j]); f ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); }
    Object.assign(sub, { t, dim, T, mn, mx, pad, idxs });
    const names = entry.meta.actuator_names || [];
    sub.legend.innerHTML = idxs.length
      ? idxs.map(j => `<span class="jc-leg" data-j="${j}"><i style="background:${color(j)}"></i>${names[j] || j}</span>`).join('')
      : '<span class="jc-empty">pick a group above to compare joints</span>';
    sub.legend.querySelectorAll('.jc-leg').forEach(s => s.onclick = () => { sub.sel.delete(+s.dataset.j); sub.dirty = true; draw(sub); });
  }
  // Per-frame: blit the cached curves + draw only the moving playhead + dots (cheap).
  function draw(sub){
    const cv = sub.cv; if (!panel.el.isConnected || cv.offsetWidth === 0) return;
    const dpr = devicePixelRatio;
    const W = Math.max(2, cv.clientWidth * dpr), H = Math.max(2, cv.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H || sub.dirty){ cv.width = W; cv.height = H; renderStatic(sub); }
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, W, H);
    if (sub.oc) ctx.drawImage(sub.oc, 0, 0);
    if (!sub.t) return;
    const T = sub.T, ph = (Math.min(frame, T - 1) / Math.max(1, T - 1)) * W;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, H); ctx.stroke();
    const f = Math.min(frame, T - 1), Y = (v) => H - sub.pad - ((v - sub.mn) / (sub.mx - sub.mn)) * (H - 2 * sub.pad);
    for (const j of sub.idxs){ ctx.fillStyle = color(j); ctx.beginPath(); ctx.arc(ph, Y(sub.t.data[f * sub.dim + j]), 2.2 * dpr, 0, 7); ctx.fill(); }
  }

  function scrub(e, sub){
    if (!entry) return; const r = sub.cv.getBoundingClientRect();
    const f = Math.round(((e.clientX - r.left) / r.width) * (entry.T - 1));
    document.dispatchEvent(new CustomEvent('panel-scrub', { detail: Math.max(0, Math.min(f, entry.T - 1)) }));
  }

  function selectGroup(sub, gkey){
    const groups = entry?.meta.actuator_groups || {}, names = entry?.meta.actuator_names || [];
    (groups[gkey] || []).forEach(n => { const i = names.indexOf(n); if (i >= 0) sub.sel.add(i); });
    sub.dirty = true; draw(sub);
  }
  let presetDone = false;
  function applyPreset(){
    if (presetDone || !entry || !panel.preset || !panel.preset.length) return;
    presetDone = true;
    while (subs.length){ grid.removeChild(subs[0].el); subs.shift(); }
    for (const g of panel.preset){ const sub = addSub(); selectGroup(sub, g); }
  }

  addSub();    // first sub-plot (replaced if a preset is applied)
  return {
    update(e, f){ const ch = e !== entry; entry = e; frame = f; if (ch){ subs.forEach(s => s.dirty = true); subs.forEach(buildGroups); } applyPreset(); subs.forEach(draw); },
    resize(){ subs.forEach(draw); },
  };
}
