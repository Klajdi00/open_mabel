/* Retarget Lab — UI controller. One /ws/state stream (30 Hz) carries the
   gateway, recorder, player and every variant's latest record; plots and the
   rolling table are accumulated here from those records (keyed on the
   engine clock `rec.t`, deduped on `rec.seq`). */
import { LabViewer } from './viewer.js';
import { StripChart } from './charts.js';

const $ = (s) => document.querySelector(s);
const api = (action, body = {}) => fetch('/api/command', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ action, ...body }) }).then(r => r.json());

let CFG = null, STATE = null, CAT = {};
let side = 'both', winS = 15;
const hist = {};             // vid -> {seq, rows:[[t, m]]}
const viewer = new LabViewer($('#view'), $('#labels'));
window.__viewer = viewer;          // debugging handle (tests read it)

// metrics shown as plots (user-selectable) and as table rows
const PLOT_DEFAULT = ['cos_reach', 'cos_fore', 'place_pct', 'orient_deg', 'elbow_err', 'aperture_r', 'qvel', 'solve_ms'];
let plotOn = JSON.parse(localStorageGet('rtl.plots') || 'null') || PLOT_DEFAULT.slice(0, 4);
const TABLE = [
  ['Shape — scale-free', null],
  ['cos_reach'], ['cos_upper'], ['cos_fore'], ['elbow_err'],
  ['Placement & orientation', null],
  ['place_pct'], ['ext_err'], ['orient_deg'],
  ['Feasibility & smoothness', null],
  ['sat_pct'], ['qvel'], ['selfcol'],
  ['Controls (never rank on these)', null],
  ['track_mm', 'ctrl'], ['solve_ms', 'ctrl'],
];

function localStorageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

/* ── bootstrap ─────────────────────────────────────────────────────────── */
async function boot() {
  CFG = await fetch('/api/config').then(r => r.json());
  for (const v of CFG.variants) CAT[v.id] = v;
  renderGuide();
  renderChips();
  buildPlots();
  await refreshEpisodes();
  connect();
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onopen = () => { $('#conn').className = 'conn ok'; };
  ws.onclose = () => { $('#conn').className = 'conn bad'; setTimeout(connect, 1000); };
  ws.onmessage = (e) => onState(JSON.parse(e.data));
}

/* ── state ─────────────────────────────────────────────────────────────── */
let lastActiveKey = '', lastEvalLast = null, frameN = 0;
function onState(s) {
  STATE = s;
  const active = Object.keys(s.variants);
  const key = active.join(',');
  if (key !== lastActiveKey) {
    lastActiveKey = key;
    viewer.setVariants(active.map(id => CAT[id]));
    buildPlots();
  }
  for (const [id, v] of Object.entries(s.variants)) {
    const rec = v.rec; if (!rec) continue;
    const h = hist[id] || (hist[id] = { seq: null, rows: [] });
    if (v.seq === h.seq) continue;
    h.seq = v.seq;
    if (h.rows.length && rec.t < h.rows[h.rows.length - 1][0] - 0.5) h.rows.length = 0;
    if (rec.anchored) h.rows.push([rec.t, rec.m]);
    const cut = rec.t - 60;
    while (h.rows.length && h.rows[0][0] < cut) h.rows.shift();
    for (const k of plotOn) charts[k]?.push(id, rec.t, sideVal(rec.m, k));
  }
  renderTop(s);
  renderVariants(s);
  viewer.update(s, liveNums());
  if (++frameN % 6 === 0) { renderLiveTable(); for (const c of Object.values(charts)) c.draw(); }
  else for (const c of Object.values(charts)) c.draw();
  renderTransport(s.player, s.source);
  renderEvalStatus(s.eval);
  if (s.eval && s.eval.last && s.eval.last !== lastEvalLast) {
    lastEvalLast = s.eval.last; loadEvalList(s.eval.last); refreshEpisodes();
  }
}

function sideVal(m, k) {
  const meta = CFG.metrics[k];
  if (!meta.per_side) return m[k];
  if (side === 'l') return m[`${k}_l`];
  if (side === 'r') return m[`${k}_r`];
  const a = m[`${k}_l`], b = m[`${k}_r`];
  if (a == null) return b; if (b == null) return a;
  return (a + b) / 2;
}

function renderTop(s) {
  for (const b of document.querySelectorAll('#srcSeg button')) b.classList.toggle('on', b.dataset.src === s.source);
  // keep the KINEMATIC tag and the banner BELOW the stage chips, however many rows they wrap to
  { const st = $('.stage').getBoundingClientRect(), tb = $('.stage-tools').getBoundingClientRect().bottom - st.top;
    $('#tagKin').style.top = `${Math.round(tb + 6)}px`;
    $('#banner').style.top = `${Math.round(tb + ((s.physics || {}).on ? 36 : 8))}px`; }
  const ph = s.physics || {};
  const pb = $('#physBtn'); pb.classList.toggle('on', !!ph.on);
  const waiting = !!ph.on && !ph.q;                       // on, but no physics frame yet
  const PH = { starting: "Starting NOVA's simulator (MuJoCo + gateway)", connecting: "Connecting to NOVA's gateway",
               seeding: 'Placing the sim robot at the kinematic pose', live: 'Waiting for the first physics frame',
               error: ph.status || 'Physics failed' };
  if (performance.now() > (window._physClickT || 0) + 1500 || ph.on) {
    pb.classList.toggle('loading', waiting);
    pb.innerHTML = !ph.on ? (ph.phase === 'error' ? 'Physics · failed' : 'Physics')
      : waiting ? `<span class="spin"></span>Physics · ${ph.phase || '…'} ${ph.phase_s != null ? ph.phase_s.toFixed(0) + ' s' : ''}` : 'Physics · on';
  }
  $('#physLoading').classList.toggle('hidden', !waiting);
  if (waiting) $('#physLoadingTxt').textContent = `${PH[ph.phase] || 'Starting'}… ${ph.phase_s != null ? ph.phase_s.toFixed(1) + ' s' : ''}`;
  if (!ph.on && ph.phase === 'error' && ph.status && window._physErr !== ph.status) { window._physErr = ph.status; alertMsg(ph.status); }
  if (ph.phase !== 'error') window._physErr = null;
  for (const id of ['tagKin', 'tagPhys', 'splitLine']) $('#' + id).classList.toggle('hidden', !ph.on);
  const lg = ph.lag;
  $('#physLag').textContent = !ph.on ? '' : !ph.q ? `· ${ph.status || ''}` :
    lg ? `· lag ${lg.lag_at_cap ? '≥ ' + (lg.time_lag_ms / 1000).toFixed(1) + ' s' : lg.time_lag_ms.toFixed(0) + ' ms'} · now ${lg.err_now_max.toFixed(2)} rad (${lg.err_now_joint}) · rms ${lg.err_rms.toFixed(3)} rad` : '· measuring…';
  const st = s.settings || {};
  for (const b of document.querySelectorAll('#scopeSeg button')) b.classList.toggle('on', b.dataset.scope === st.scope);
  for (const b of document.querySelectorAll('#robotSeg button')) b.classList.toggle('on', b.dataset.robot === st.robot);
  for (const b of document.querySelectorAll('#eeSeg button')) b.classList.toggle('on', b.dataset.ee === (st.ee || 'orca'));
  for (const g of document.querySelectorAll('.seg.chan')) for (const b of g.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === (st[g.dataset.ch] || 'free'));
  const g = s.gateway, card = $('#vpCard');
  const act = g.clients.find(c => c.active) || g.clients.find(c => c.hands.length || c.head);
  card.className = 'vp-card ' + (g.error ? 'err' : act && act.ageS < 1 ? 'live' : g.clients.length ? 'idle' : '');
  if (g.error) {
    $('#vpL1').textContent = 'Gateway down'; $('#vpL2').textContent = g.error;
  } else if (act) {
    $('#vpL1').textContent = `${act.device} · ${act.addr}`;
    $('#vpL2').textContent = `${act.hz.toFixed(0)} Hz · ${act.frames} frames${act.ageS > 1 ? ` · silent ${act.ageS.toFixed(0)} s` : ''}`;
  } else {
    $('#vpL1').textContent = g.clients.length ? `${g.clients.length} device(s) — no tracking yet` : 'Waiting for Vision Pro…';
    $('#vpL2').textContent = `${g.bonjour || 'ws'} · ws://${g.lan}:${g.port}/teleop`;
  }
  const on = { head: act?.head, left: act?.hands.includes('left'), right: act?.hands.includes('right') };
  for (const sp of document.querySelectorAll('#trk span')) sp.classList.toggle('on', !!on[sp.dataset.k] && act.ageS < 1);
  const r = s.recording, btn = $('#recBtn');
  // a click in flight owns the button until the server state agrees (or 3 s pass): no flicker back
  const pend = window._recPending;
  if (pend && (r.active === pend.on || performance.now() - pend.t > 3000)) window._recPending = null;
  if (!window._recPending) {
    btn.classList.toggle('on', r.active);
    $('#recLbl').textContent = r.active ? `Stop · ${r.duration_s.toFixed(1)} s · ${r.frames}` : 'Record';
  }
  const ban = $('#banner');
  let msg = '';
  if (g.error) msg = `⚠ ${g.error}`;
  else if (s.source === 'live' && !act) msg = `Open <b>MABEL</b> on the Vision Pro — it finds this Mac on the Wi-Fi as <b>${(g.bonjour || 'Retarget Lab').replace(/ \(.*/, '')}</b> (or enter <b>${g.lan}</b>). Hands stream once the tracking space is open. No headset? <b>Replay</b> an EgoDex clip.`;
  ban.innerHTML = msg; ban.classList.toggle('hidden', !msg); ban.classList.toggle('err', !!g.error);
}

function liveNums() {
  const out = {};
  for (const id of Object.keys(hist)) {
    const st = stats(id, 'cos_reach'), pl = stats(id, 'place_pct');
    out[id] = st ? `${st.mean.toFixed(3)} · ${pl ? pl.mean.toFixed(0) : '—'}%` : '';
  }
  return out;
}

function renderVariants(s) {
  const el = $('#vlist');
  if (!el.children.length) {
    for (const v of CFG.variants) {
      const row = document.createElement('div');
      row.className = 'vrow'; row.dataset.id = v.id; row.title = v.blurb;
      row.innerHTML = `<span class="sw" style="background:${v.color}"></span><span class="nm">${v.label}</span>
        <span class="stat"></span><span class="mir" title="mirror this retargeter on the headset"></span>`;
      row.onclick = (e) => {
        if (e.target.classList.contains('mir')) { api('mirror', { id: v.id }); return; }
        const act = Object.keys(STATE.variants);
        const next = act.includes(v.id) ? act.filter(x => x !== v.id) : [...act, v.id];
        api('variants', { active: CFG.variants.map(x => x.id).filter(x => next.includes(x)) });
      };
      el.appendChild(row);
    }
  }
  for (const row of el.children) {
    const v = s.variants[row.dataset.id], on = !!v;
    row.classList.toggle('on', on);
    row.querySelector('.mir').classList.toggle('on', s.mirror === row.dataset.id);
    const st = row.querySelector('.stat');
    st.classList.toggle('err', !!(v && v.error));
    st.textContent = !v ? '' : v.error ? 'error' : !v.ready ? 'starting…' : v.rec ? `${v.rec.m.solve_ms?.toFixed(1)} ms · ${v.hz.toFixed(0)} Hz` : 'ready';
    if (v && v.error) row.title = v.error;
  }
}

/* ── rolling statistics ────────────────────────────────────────────────── */
function stats(id, k) {
  const h = hist[id]; if (!h || !h.rows.length) return null;
  const tEnd = h.rows[h.rows.length - 1][0], xs = [];
  for (let i = h.rows.length - 1; i >= 0 && h.rows[i][0] >= tEnd - winS; i--) {
    const v = sideVal(h.rows[i][1], k); if (v != null && Number.isFinite(v)) xs.push(v);
  }
  if (!xs.length) return null;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, xs.length - 1));
  // frames are strongly autocorrelated at 60 Hz: count ~1 independent sample
  // per 0.25 s rather than per frame, or the "noise" band would be fiction
  const nEff = Math.max(1, Math.min(xs.length, winS / 0.25));
  return { mean, sem: sd / Math.sqrt(nEff), n: xs.length };
}
const fmt = (v, k) => v == null ? '—' : k.startsWith('cos') ? v.toFixed(3) : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);

function bestOf(ids, k, getter) {
  const hib = CFG.metrics[k]?.higher; if (hib == null) return null;
  const vals = ids.map(id => [id, getter(id)]).filter(([, s]) => s && s.mean != null);
  if (vals.length < 2) return null;
  vals.sort((a, b) => hib ? b[1].mean - a[1].mean : a[1].mean - b[1].mean);
  const [w, r] = vals;
  return Math.abs(w[1].mean - r[1].mean) > 2 * Math.hypot(w[1].sem || 0, r[1].sem || 0) ? w[0] : null;
}

function renderLiveTable() {
  if (!STATE) return;
  const ids = Object.keys(STATE.variants);
  let h = '<tr><th>metric</th>' + ids.map(id => `<th><span class="sw" style="background:${CAT[id].color}"></span>${CAT[id].short || CAT[id].label.split(' ')[0]}</th>`).join('') + '</tr>';
  for (const [k, kind] of TABLE) {
    if (kind === null) { h += `<tr class="sec"><td colspan="${ids.length + 1}">${k}</td></tr>`; continue; }
    const m = CFG.metrics[k], best = kind === 'ctrl' ? null : bestOf(ids, k, id => stats(id, k));
    h += `<tr class="${kind || ''}"><td title="${m.label}">${m.label} <span class="u">${m.unit}</span></td>` +
      ids.map(id => { const s = stats(id, k); return `<td class="${best === id ? 'best' : ''}">${s ? fmt(s.mean, k) : '—'}</td>`; }).join('') + '</tr>';
  }
  // grasp: rolling correlation of human vs robot pinch aperture
  h += `<tr class="sec"><td colspan="${ids.length + 1}">Grasp</td></tr><tr><td>Aperture corr. r <span class="u">h vs r</span></td>` +
    ids.map(id => `<td>${fmt(apertureR(id), 'cos')}</td>`).join('') + '</tr>';
  $('#liveTable').innerHTML = h;
}

function apertureR(id) {
  const h = hist[id]; if (!h || h.rows.length < 10) return null;
  const tEnd = h.rows[h.rows.length - 1][0], a = [], b = [];
  for (const [t, m] of h.rows) {
    if (t < tEnd - winS) continue;
    for (const s of side === 'both' ? ['l', 'r'] : [side]) {
      const x = m[`aperture_h_${s}`], y = m[`aperture_r_${s}`];
      if (x != null && y != null) { a.push(x); b.push(y); }
    }
  }
  if (a.length < 10) return null;
  const ma = a.reduce((p, c) => p + c) / a.length, mb = b.reduce((p, c) => p + c) / b.length;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa > 1e-9 && sbb > 1e-9 ? sab / Math.sqrt(saa * sbb) : null;
}

/* ── plots ─────────────────────────────────────────────────────────────── */
const charts = {};
function renderChips() {
  const el = $('#mchips'); el.innerHTML = '';
  const order = ['cos_reach', 'cos_upper', 'cos_fore', 'elbow_err', 'place_pct', 'ext_err', 'orient_deg',
    'aperture_h', 'aperture_r', 'track_mm', 'sat_pct', 'qvel', 'selfcol', 'solve_ms'];
  for (const k of order) {
    const b = document.createElement('button');
    b.className = 'chip' + (plotOn.includes(k) ? ' on' : '');
    b.textContent = CFG.metrics[k].label.replace('Pinch aperture · ', 'aperture ').replace(' (control)', '');
    b.onclick = () => {
      plotOn = plotOn.includes(k) ? plotOn.filter(x => x !== k) : [...plotOn, k].slice(-8);
      localStorageSet('rtl.plots', JSON.stringify(plotOn));
      renderChips(); buildPlots();
    };
    el.appendChild(b);
  }
}
function buildPlots() {
  const grid = $('#pgrid'); grid.innerHTML = '';
  for (const k of Object.keys(charts)) delete charts[k];
  const ids = STATE ? Object.keys(STATE.variants) : [];
  grid.style.gridTemplateColumns = `repeat(${Math.min(4, Math.max(1, plotOn.length))},1fr)`;
  for (const k of plotOn) {
    const m = CFG.metrics[k];
    const cell = document.createElement('div'); cell.className = 'pcell';
    const cv = document.createElement('canvas'); cell.appendChild(cv); grid.appendChild(cell);
    const ch = new StripChart(cv, { windowS: winS, legend: false, title: `${m.label}${m.unit ? ' [' + m.unit + ']' : ''}`,
      yMin: k.startsWith('cos') ? undefined : 0, yMax: k.startsWith('cos') ? 1 : undefined });
    for (const id of ids) ch.def(id, CAT[id].short || CAT[id].label.split(' ')[0], CAT[id].color);
    for (const id of ids) {                         // seed from history
      const h = hist[id]; if (!h) continue;
      ch.seed(id, h.rows.map(([t, mm]) => [t, sideVal(mm, k)]));
    }
    charts[k] = ch;
  }
}

/* ── replay transport ──────────────────────────────────────────────────── */
let scrubbing = false;
function renderTransport(p, src) {
  $('#transport').classList.toggle('hidden', src !== 'replay' || !p.loaded);
  if (!p.loaded) return;
  $('#playBtn').textContent = p.playing ? '❚❚' : '▶';
  $('#playBtn').classList.toggle('on', p.playing);
  $('#tc').textContent = `${p.t.toFixed(1)} / ${p.duration.toFixed(1)} s`;
  if (!scrubbing) $('#scrub').value = p.n > 1 ? Math.round(1000 * p.i / (p.n - 1)) : 0;
  $('#speedBtn').textContent = `${p.speed}×`;
  $('#loopBtn').classList.toggle('on', p.loop);
  $('#tname').textContent = p.name || '';
  for (const e of document.querySelectorAll('.ep')) e.classList.toggle('cur', e.dataset.id === p.id);
}
$('#playBtn').onclick = () => api('play', { on: !STATE.player.playing });
$('#scrub').oninput = () => { scrubbing = true; };
$('#scrub').onchange = (e) => { scrubbing = false; resetHist(); api('seek', { frac: e.target.value / 1000 }); };
$('#speedBtn').onclick = () => {
  const s = [0.25, 0.5, 1, 2], i = s.indexOf(STATE.player.speed);
  api('speed', { value: s[(i + 1) % s.length] });
};
$('#loopBtn').onclick = () => api('loop', { on: !STATE.player.loop });

function resetHist() { for (const k of Object.keys(hist)) delete hist[k]; for (const c of Object.values(charts)) c.clear(); }

/* ── top-bar controls ──────────────────────────────────────────────────── */
for (const b of document.querySelectorAll('#srcSeg button')) b.onclick = () => { resetHist(); api('source', { source: b.dataset.src }); };
for (const b of document.querySelectorAll('#scopeSeg button')) b.onclick = () => api('settings', { scope: b.dataset.scope });
for (const b of document.querySelectorAll('#robotSeg button')) b.onclick = () => api('settings', { robot: b.dataset.robot });
for (const b of document.querySelectorAll('#eeSeg button')) b.onclick = () => {
  viewer.eeOverride = b.dataset.ee;                          // draw it now, before the server's echo
  for (const o of document.querySelectorAll('#eeSeg button')) o.classList.toggle('on', o === b);
  api('settings', { ee: b.dataset.ee });
};
for (const g of document.querySelectorAll('.seg.chan')) for (const b of g.querySelectorAll('button')) b.onclick = () => api('settings', { [g.dataset.ch]: b.dataset.v });
$('#physBtn').onclick = () => {
  const b = $('#physBtn'), on = !b.classList.contains('on');
  window._physClickT = performance.now();                  // instant feedback before the server answers
  if (on) { b.classList.add('loading'); b.innerHTML = '<span class="spin"></span>Physics · starting…'; }
  api('physics', { on });
};
$('#recBtn').onclick = async () => {
  // One click = one toggle, shown at once. The decision is the button's own state (what the operator sees),
  // not the last state frame; repeat clicks while the request is in flight are ignored.
  if (window._recPending) return;
  const btn = $('#recBtn'), on = !btn.classList.contains('on');
  window._recPending = { on, t: performance.now() };
  btn.classList.toggle('on', on);
  $('#recLbl').textContent = on ? 'Starting…' : 'Saving…';
  let r;
  try { r = await api('record', { on, name: $('#recName').value.trim() }); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || r.ok === false) {                           // failed: show it, give the button back
    window._recPending = null;
    btn.classList.toggle('on', !on);
    $('#recLbl').textContent = `Record failed: ${(r && r.error) || 'no answer'}`;
    return r;
  }
  if (!on) { $('#recName').value = ''; refreshEpisodes(); }
  return r;
};
$('#anchorBtn').onclick = () => { resetHist(); api('reanchor'); };
$('#homeBtn').onclick = () => { resetHist(); api('home'); };
for (const b of document.querySelectorAll('#layoutSeg button')) b.onclick = () => {
  for (const x of document.querySelectorAll('#layoutSeg button')) x.classList.toggle('on', x === b);
  viewer.setLayout(b.dataset.layout); viewer.view('front');
};
$('#tgtChip').onclick = (e) => { viewer.showTargets = !viewer.showTargets; e.target.classList.toggle('on', viewer.showTargets); };
$('#humChip').onclick = (e) => { viewer.op.on = viewer.op.on === false; e.target.classList.toggle('on', viewer.op.on !== false); };
$('#followChip').onclick = (e) => { viewer.follow = !viewer.follow; e.currentTarget.classList.toggle('on', viewer.follow); };
$('#camFront').onclick = () => viewer.view('front');
$('#camSide').onclick = () => viewer.view('side');
$('#camTop').onclick = () => viewer.view('top');
for (const b of document.querySelectorAll('#sideSeg button')) b.onclick = () => {
  side = b.dataset.side; for (const x of document.querySelectorAll('#sideSeg button')) x.classList.toggle('on', x === b);
  buildPlots(); renderLiveTable();
};
for (const b of document.querySelectorAll('#winSeg button')) b.onclick = () => {
  winS = +b.dataset.win; for (const x of document.querySelectorAll('#winSeg button')) x.classList.toggle('on', x === b);
  buildPlots(); renderLiveTable();
};
for (const t of document.querySelectorAll('.dock-tab')) t.onclick = () => {
  for (const x of document.querySelectorAll('.dock-tab')) x.classList.toggle('on', x === t);
  for (const p of document.querySelectorAll('.dock-pane')) p.classList.toggle('on', p.dataset.pane === t.dataset.tab);
  if (t.dataset.tab === 'eval') loadEvalList();
};

/* ── episodes ──────────────────────────────────────────────────────────── */
async function refreshEpisodes() {
  const { episodes } = await fetch('/api/episodes').then(r => r.json());
  const el = $('#eplist'); el.innerHTML = '';
  const rec = episodes.filter(e => e.kind === 'recorded'), ego = episodes.filter(e => e.kind === 'egodex');
  const card = (e) => {
    const d = document.createElement('div'); d.className = 'ep'; d.dataset.id = e.id;
    const when = e.created ? new Date(e.created * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
    const meta = e.kind === 'recorded'
      ? `${(e.duration_s || 0).toFixed(1)} s · ${e.frames} fr · ${when}` : 'EgoDex · local only';
    const nm = e.kind === 'egodex' ? egoName(e.name) : e.name;
    d.innerHTML = `<div class="t" title="${e.name}">${nm}</div><div class="m">${meta}</div>
      <div class="b"><button data-a="play">Replay</button><button data-a="eval">Evaluate</button>
      ${e.kind === 'recorded' ? '<button data-a="del" class="del">Delete</button>' : ''}</div>`;
    d.querySelector('[data-a=play]').onclick = () => { resetHist(); api('load', { id: e.id }); };
    d.querySelector('[data-a=eval]').onclick = () => {
      api('evaluate', { id: e.id }).then(r => { if (r.ok === false) alertMsg(r.error); });
      document.querySelector('.dock-tab[data-tab=eval]').click();
    };
    const del = d.querySelector('[data-a=del]');
    if (del) del.onclick = async () => {
      if (del.dataset.armed) { await api('delete', { id: e.id }); refreshEpisodes(); }
      else { del.dataset.armed = 1; del.textContent = 'Sure?'; setTimeout(() => { delete del.dataset.armed; del.textContent = 'Delete'; }, 2500); }
    };
    return d;
  };
  const head = (t) => { const h = document.createElement('div'); h.className = 'ep-h'; h.textContent = t; el.appendChild(h); };
  head('Recorded · Vision Pro');
  if (!rec.length) {
    const e = document.createElement('div'); e.className = 'empty';
    e.innerHTML = 'No episodes yet. Press <b>Record</b> (here or on the headset) while the Vision Pro streams.';
    el.appendChild(e);
  }
  for (const e of rec) el.appendChild(card(e));
  head(`EgoDex benchmark clips · ${ego.length}`);
  if (!ego.length) { const e = document.createElement('div'); e.className = 'empty'; e.textContent = 'None found (local-only corpus).'; el.appendChild(e); return; }
  const filt = document.createElement('input');
  filt.className = 'ep-filter'; filt.placeholder = 'filter clips (task, ep 12, hi/lo)…'; filt.value = egoFilter;
  el.appendChild(filt);
  const groups = {};
  for (const e of ego) (groups[e.name.split('-')[0]] ||= []).push(e);
  const box = document.createElement('div'); el.appendChild(box);
  const draw = () => {
    box.innerHTML = '';
    const q = filt.value.trim().toLowerCase();
    for (const [task, list] of Object.entries(groups)) {
      const hit = q ? list.filter(e => (e.name + ' ' + egoName(e.name)).toLowerCase().includes(q)) : list;
      if (!hit.length) continue;
      const det = document.createElement('details'); det.className = 'ep-grp';
      det.open = !!q || openGroups.has(task);
      det.ontoggle = () => det.open ? openGroups.add(task) : openGroups.delete(task);
      det.innerHTML = `<summary>${task.replace(/_/g, ' ')} <span>${hit.length}</span></summary>`;
      // cards are built only when opened: 774 of them up front is a slow sidebar
      const fill = () => { if (det.dataset.filled) return; det.dataset.filled = 1; for (const e of hit.slice(0, 60)) det.appendChild(card(e)); };
      if (det.open) fill(); else det.addEventListener('toggle', fill, { once: true });
      box.appendChild(det);
    }
  };
  filt.oninput = () => { egoFilter = filt.value; draw(); };
  draw();
}
let egoFilter = '';
const openGroups = new Set();
const egoName = (n) => n.replace(/^[^-]*-/, '').replace(/__.*__ep0*(\d+)$/, (m, k) => ` · ep ${k}`).replace(/_/g, ' ');
$('#epRefresh').onclick = refreshEpisodes;
$('#vRefresh').onclick = async () => {
  const r = await api('catalog');
  const hint = $('#vRefresh').previousElementSibling, say = (m) => {
    hint.textContent = m; clearTimeout(hint._t);
    hint._t = setTimeout(() => { hint.textContent = 'toggle · ◉ = headset mirror'; }, 6000);
  };
  if (!r.ok) { say(`⚠ variants.py: ${r.error}`); return; }
  CFG.variants = r.variants; for (const v of r.variants) CAT[v.id] = v;
  $('#vlist').replaceChildren();
  if (STATE) renderVariants(STATE);
  const part = (k, w) => r[k].length ? `${w} ${r[k].join(', ')}` : '';
  say([part('added', 'new:'), part('changed', 'restarted:'), part('removed', 'removed:')].filter(Boolean).join(' · ')
      || `no change · ${r.variants.length} retargeters`);
};
function alertMsg(m) { const b = $('#banner'); b.innerHTML = m; b.classList.remove('hidden'); setTimeout(() => b.classList.add('hidden'), 4000); }

/* ── evaluation tab ────────────────────────────────────────────────────── */
function renderEvalStatus(ev) {
  const el = $('#evStatus');
  if (!ev) return;
  el.textContent = ev.running ? `evaluating ${ev.running} … ${ev.progress ? ev.progress.join('/') : ''} variants`
    : ev.error ? `error: ${ev.error}` : '';
}
async function loadEvalList(select) {
  const { evaluations } = await fetch('/api/evaluations').then(r => r.json());
  const sel = $('#evSel');
  const cur = select || sel.value;
  sel.innerHTML = evaluations.map(e => `<option value="${e.file}">${e.episode.replace('egodex:', 'EgoDex · ')} — ${e.variants.length} variants · ${new Date(e.created * 1000).toLocaleTimeString()}</option>`).join('');
  if (!evaluations.length) { $('#evBody').innerHTML = '<div class="empty">No evaluations yet.</div>'; return; }
  sel.value = evaluations.some(e => e.file === cur) ? cur : evaluations[0].file;
  showEval(sel.value);
}
$('#evSel').onchange = (e) => showEval(e.target.value);
async function showEval(file) {
  const ev = await fetch(`/api/evaluations/${encodeURIComponent(file)}`).then(r => r.json());
  const ids = ev.variants;
  const row = (k, stat = 'p50') => {
    const m = ev.metrics[k];
    const vals = ids.map(id => ev.summary[id][k]);
    const get = (id) => { const s = ev.summary[id][k]; return s ? { mean: s[stat], sem: s.sem } : null; };
    const best = bestOf(ids, k, get);
    return `<tr><td title="${m.label}">${m.label} <span class="u">${m.unit}</span></td>` +
      vals.map((s, i) => `<td class="${best === ids[i] ? 'best' : ''}">${s ? fmt(s[stat], k) : '—'}<span class="u"> ${s ? fmt(s.p90, k) : ''}</span></td>`).join('') + '</tr>';
  };
  let h = `<div class="note">${ev.frames} frames · ${ev.wall_s} s wall · median <span class="u">p90</span>, anchored frames, both hands pooled.</div>`;
  h += '<table class="mtable"><tr><th>metric</th>' + ids.map(id => `<th><span class="sw" style="background:${CAT[id]?.color}"></span>${CAT[id]?.short || (CAT[id]?.label || id).split(' ')[0]}</th>`).join('') + '</tr>';
  for (const [k, kind] of TABLE) {
    if (kind === null) { h += `<tr class="sec"><td colspan="${ids.length + 1}">${k}</td></tr>`; continue; }
    h += row(k);
  }
  h += `<tr class="sec"><td colspan="${ids.length + 1}">Grasp</td></tr><tr><td>Aperture corr. r <span class="u">L / R</span></td>` +
    ids.map(id => { const s = ev.summary[id]; const f = (x) => x ? x.mean.toFixed(2) : '—'; return `<td>${f(s.grasp_r_l)} / ${f(s.grasp_r_r)}</td>`; }).join('') + '</tr></table>';
  if (ev.figures) {
    const png = ev.figures.filter(f => f.endsWith('.png')), pdf = ev.figures.filter(f => f.endsWith('.pdf'));
    h += png.map(f => `<img class="evfig" src="/api/evaluations/${encodeURIComponent(f)}">`).join('');
    h += `<div class="evlinks">${pdf.map(f => `<a href="/api/evaluations/${encodeURIComponent(f)}" target="_blank">${f.split('__').pop()}</a>`).join('')}
      <a href="/api/evaluations/${encodeURIComponent(file)}" target="_blank">raw JSON</a></div>`;
  }
  $('#evBody').innerHTML = h;
}

/* ── the metric guide ──────────────────────────────────────────────────── */
function renderGuide() {
  const G = [
    ['Read this first', null, `<p class="lead">A retargeter is a map from a human body to a robot body with different proportions
      (MABEL's shoulders are ~3× narrower, its arms ~1.7× longer than an operator's). So "error" needs a reference, and every
      reference flatters some map. The lab therefore reports several <b>complementary</b> axes; no single number ranks maps.</p>
      <p>All human vectors are mapped through the engine's own body-frame rotation <span class="eq">M</span> (operator right/up/fwd → robot right/up/fwd)
      before comparison. Statistics use <b>anchored</b> frames only. Live numbers drop frames when a worker is busy — use
      <b>Evaluate</b> for numbers you quote.</p>`],
    ['cos_reach', 'shape', `<div class="eq">cos∠( M(w<sub>h</sub>−s<sub>h</sub>), w<sub>r</sub>−s<sub>r</sub> )</div>
      <p>Direction of shoulder→wrist, human vs robot. Scale-free, so a 0.6 m human arm and a 0.67 m robot arm compare directly.
      <b>Caveat:</b> it is anchored at the shoulder, so a per-shoulder map (Extension) scores ≈1 <i>by construction</i> while
      pulling both hands inboard — always read it next to Placement. 0.99 ≈ 8°, 0.95 ≈ 18°, 0.90 ≈ 26°.</p>`],
    ['cos_upper', 'shape', `<div class="eq">cos∠( M(e<sub>h</sub>−s<sub>h</sub>), e<sub>r</sub>−s<sub>r</sub> ) &nbsp;·&nbsp; cos_fore: elbow→wrist</div>
      <p>Per-segment cosines: does the arm <i>bend like the human's</i>? The hand position fixes the shoulder→wrist chord; the elbow's
      swivel about it is what makes an arm look human. Human elbow = ARKit <code>forearm_arm</code>; human shoulder = the engine's
      anthropometric estimate (ARKit has no scapula), so cos_upper carries that estimate's error.</p>`],
    ['elbow_err', 'shape', `<p>|interior elbow angle, human − robot| in degrees. Catches "elbow stays bent when I fully extend" — MABEL's elbow
      hard-stops at 161° interior, so ~19° is the floor at full extension, not a retargeting defect.</p>`],
    ['place_pct', 'placement', `<div class="eq">‖ (w<sub>r</sub>−h<sub>r</sub>)/L<sub>r</sub> − M(w<sub>h</sub>−h<sub>h</sub>)/L<sub>h</sub> ‖ × 100</div>
      <p>Head→wrist vector, each normalised by its <b>own</b> arm reach (the N-MPJPE idea: remove body size with one scale per body).
      In % of reach. Head-anchored, so it flatters head-anchored scaling (Abs.) the way cos_reach flatters Extension — read the pair.</p>`],
    ['ext_err', 'placement', `<p>|ρ<sub>h</sub> − ρ<sub>r</sub>| with ρ = |wrist−shoulder| / arm length. "Full human extension should be full robot extension."</p>`],
    ['orient_deg', 'placement', `<p>Palm orientation: the worse of the palm-normal and finger-axis angles, against the axes the solver was actually given
      (operator palm frame through M, left normal flipped to palmar). Scoring only the normal hides the hand's twist about it.</p>`],
    ['aperture', 'grasp', `<p>Thumb-tip↔index-tip distance, human and robot. Compare them by <b>correlation r</b>, not by error: the ORCA hand spans
      54–97 mm against a human's 2–147 mm, so an absolute error rewards a frozen hand. A frozen hand gives r≈0 and cannot fake it.</p>`],
    ['sat_pct', 'feasibility', `<p>Share of arm joints within 1% of travel from a stop. Saturation is where a map silently stops tracking.</p>`],
    ['qvel', 'smoothness', `<p>Peak arm joint speed between consecutive solves [rad/s] — basin hops and flips show up as spikes (the trust region caps
      a solve at 0.10 rad).</p>`],
    ['selfcol', 'feasibility', `<p>Contacts between one arm and another segment (other arm, trunk), from MuJoCo's collision pass on the commanded pose.
      Contacts inside one arm (a closing hand) don't count.</p>`],
    ['track_mm', 'control', `<p>Distance from the map's <b>own</b> wrist target to the achieved wrist. It scores the IK, not the map: a map that commands
      the current pose scores 0 forever. Never rank maps on it — it is here to show the IK keeps up (Direct copy has no target, so it reads huge).</p>`],
    ['Kinematic, by design', null, `<p>There is no plant: each solve's targets are written straight into qpos (ideal servo). Differences are purely the map —
      no lag, no gravity sag, no engagement ease. The base never moves.</p>`],
  ];
  $('#guide').innerHTML = '<div class="guide">' + G.map(([k, tag, body]) => {
    const title = CFG.metrics[k]?.label || (k === 'aperture' ? 'Pinch aperture' : k);
    const cls = tag === 'control' ? 'warn' : tag ? 'good' : '';
    return `<h4>${title}${tag ? ` <span class="tag ${cls}">${tag}</span>` : ''}</h4>${body}`;
  }).join('') + '</div>';
}

boot();
