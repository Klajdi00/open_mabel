// General "Signals" panel (Rerun Scalars / SeriesLine). Plots ANY scalar channel
// from the dense bundle — base twist, grasp, base command, mode, end-effector
// pose components, look-at — that the joint plots don't cover. Multiple
// independent sub-plots ("+ split view"), group quick-adds, offscreen-cached
// curves (only the playhead moves per frame), playhead scrub.
const color = (i) => `hsl(${(i * 53 + 200) % 360},70%,62%)`;

// channel catalog: [groupLabel, [[bundleTopic, component, label], …]]
const CH_GROUPS = [
  ['Base', [['base_twist', 0, 'base vx'], ['base_twist', 1, 'base vy'], ['base_twist', 2, 'base ω'],
            ['pose/chassis', 0, 'base x'], ['pose/chassis', 1, 'base y']]],
  ['Grasp', [['teleop/grasp', 0, 'grasp L'], ['teleop/grasp', 1, 'grasp R']]],
  ['Base cmd', [['teleop/base_cmd', 0, 'cmd vx'], ['teleop/base_cmd', 1, 'cmd ω']]],
  ['Mode', [['teleop/mode', 0, 'mode']]],
  ['L palm', [['pose/left_palm', 0, 'L palm x'], ['pose/left_palm', 1, 'L palm y'], ['pose/left_palm', 2, 'L palm z']]],
  ['R palm', [['pose/right_palm', 0, 'R palm x'], ['pose/right_palm', 1, 'R palm y'], ['pose/right_palm', 2, 'R palm z']]],
  ['Look-at', [['teleop/look_at', 0, 'look x'], ['teleop/look_at', 1, 'look y'], ['teleop/look_at', 2, 'look z']]],
];
const CH = [];                                          // flat list; id = index
const CH_BY_ID = {};
for (const [grp, items] of CH_GROUPS) for (const [topic, comp, label] of items){
  const id = CH.length; CH.push({ id, grp, topic, comp, label }); CH_BY_ID[id] = CH[id];
}
const colorOf = (id) => color(id);

export function signalsFactory(panel){
  panel.body.innerHTML = `<div class="jc-root"><div class="jc-bar"><button class="jc-split">+ split view</button>
    <span class="jc-hint">add channels by group · drag panel edge to resize</span></div>
    <div class="jc-grid"></div></div>`;
  const grid = panel.body.querySelector('.jc-grid');
  panel.body.querySelector('.jc-split').onclick = () => addSub();
  let entry = null, frame = 0;
  const subs = [];

  function avail(){                                     // channel ids present in this episode
    if (!entry) return new Set();
    const ok = new Set();
    for (const c of CH){ const t = entry.bundle.topics[c.topic]; if (t && c.comp < t.dim) ok.add(c.id); }
    return ok;
  }

  function addSub(preset){
    const el = document.createElement('div'); el.className = 'jc-sub';
    el.innerHTML = `<div class="jc-ctl"><div class="jc-groups"></div>
      <button class="jc-clear">clear</button>${subs.length ? '<button class="jc-remove" title="remove sub-plot">✕</button>' : ''}</div>
      <canvas class="jc-canvas"></canvas><div class="jc-legend"></div>`;
    grid.appendChild(el);
    const sub = { el, sel: new Set(), cv: el.querySelector('.jc-canvas'),
                  legend: el.querySelector('.jc-legend'), groupsEl: el.querySelector('.jc-groups'), dirty: true };
    el.querySelector('.jc-clear').onclick = () => { sub.sel.clear(); sub.dirty = true; draw(sub); };
    el.querySelector('.jc-remove')?.addEventListener('click', () => { grid.removeChild(el); subs.splice(subs.indexOf(sub), 1); });
    sub.cv.addEventListener('pointerdown', (e) => scrub(e, sub));
    new ResizeObserver(() => draw(sub)).observe(el);
    subs.push(sub); buildGroups(sub); if (preset) preset.forEach(id => sub.sel.add(id)); draw(sub);
    return sub;
  }

  function buildGroups(sub){
    if (!entry) return;
    const ok = avail(); sub.groupsEl.innerHTML = '';
    for (const [grp, items] of CH_GROUPS){
      const ids = items.map(it => CH.find(c => c.topic === it[0] && c.comp === it[1])?.id).filter(id => ok.has(id));
      if (!ids.length) continue;
      const b = document.createElement('button'); b.className = 'jc-grp'; b.textContent = '+ ' + grp;
      b.onclick = () => { ids.forEach(id => sub.sel.add(id)); sub.dirty = true; draw(sub); };
      sub.groupsEl.appendChild(b);
    }
  }

  function renderStatic(sub){
    const dpr = devicePixelRatio, W = sub.cv.width, H = sub.cv.height;
    if (!sub.oc) sub.oc = document.createElement('canvas');
    sub.oc.width = W; sub.oc.height = H;
    const ctx = sub.oc.getContext('2d'); ctx.clearRect(0, 0, W, H);
    sub.ids = []; sub.dirty = false;
    if (!entry) return;
    const T = entry.T, ids = [...sub.sel].filter(id => { const c = CH_BY_ID[id], t = entry.bundle.topics[c.topic]; return t && c.comp < t.dim; });
    let mn = Infinity, mx = -Infinity;
    for (const id of ids){ const c = CH_BY_ID[id], t = entry.bundle.topics[c.topic];
      for (let f = 0; f < T; f++){ const v = t.data[f * t.dim + c.comp]; if (v < mn) mn = v; if (v > mx) mx = v; } }
    if (!isFinite(mn)){ mn = -1; mx = 1; } if (mx - mn < 1e-6){ mx += 1; mn -= 1; }
    const pad = 6 * dpr, Y = (v) => H - pad - ((v - mn) / (mx - mn)) * (H - 2 * pad);
    if (mn < 0 && mx > 0){ ctx.strokeStyle = 'rgba(242,238,230,.12)'; ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke(); }
    for (const id of ids){ const c = CH_BY_ID[id], t = entry.bundle.topics[c.topic];
      ctx.strokeStyle = colorOf(id); ctx.lineWidth = 1.3 * dpr; ctx.beginPath();
      for (let f = 0; f < T; f++){ const px = (f / Math.max(1, T - 1)) * W, py = Y(t.data[f * t.dim + c.comp]); f ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); }
    Object.assign(sub, { ids, mn, mx, pad, T });
    sub.legend.innerHTML = ids.length
      ? ids.map(id => `<span class="jc-leg" data-id="${id}"><i style="background:${colorOf(id)}"></i>${CH_BY_ID[id].label}</span>`).join('')
      : '<span class="jc-empty">pick a channel group above</span>';
    sub.legend.querySelectorAll('.jc-leg').forEach(s => s.onclick = () => { sub.sel.delete(+s.dataset.id); sub.dirty = true; draw(sub); });
  }

  function draw(sub){
    const cv = sub.cv; if (!panel.el.isConnected || cv.offsetWidth === 0) return;
    const dpr = devicePixelRatio;
    const W = Math.max(2, cv.clientWidth * dpr), H = Math.max(2, cv.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H || sub.dirty){ cv.width = W; cv.height = H; renderStatic(sub); }
    const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, W, H);
    if (sub.oc) ctx.drawImage(sub.oc, 0, 0);
    if (!entry || !sub.ids || !sub.ids.length) return;
    const T = sub.T, ph = (Math.min(frame, T - 1) / Math.max(1, T - 1)) * W;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = dpr; ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, H); ctx.stroke();
    const f = Math.min(frame, T - 1), Y = (v) => H - sub.pad - ((v - sub.mn) / (sub.mx - sub.mn)) * (H - 2 * sub.pad);
    for (const id of sub.ids){ const c = CH_BY_ID[id], t = entry.bundle.topics[c.topic];
      ctx.fillStyle = colorOf(id); ctx.beginPath(); ctx.arc(ph, Y(t.data[f * t.dim + c.comp]), 2.2 * dpr, 0, 7); ctx.fill(); }
  }

  function scrub(e, sub){
    if (!entry) return; const r = sub.cv.getBoundingClientRect();
    const f = Math.round(((e.clientX - r.left) / r.width) * (entry.T - 1));
    document.dispatchEvent(new CustomEvent('panel-scrub', { detail: Math.max(0, Math.min(f, entry.T - 1)) }));
  }

  addSub();
  return {
    update(e, f){ const ch = e !== entry; entry = e; frame = f; if (ch){ subs.forEach(s => s.dirty = true); subs.forEach(buildGroups); } subs.forEach(draw); },
    resize(){ subs.forEach(draw); },
  };
}
