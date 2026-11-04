// MABEL Trainer Studio — boot + orchestration.
import { api } from './api.js';
import { initSplitters } from './splitters.js';
import { PanelManager } from './panels.js';
import { ArchDiagram } from './archdiagram.js';
import { ChartGrid } from './chartgrid.js';
import { DatasetsPanel } from './datasets.js';
import { DockPanes } from './archpicker.js';
import { DeployPanel } from './deploypanel.js';
import { registerPanels } from './panelfactories.js';
import * as bus from './bus.js';

const $ = (s) => document.querySelector(s);
const state = {
  arch: 'ACT', hpByArch: {}, selection: null, resolved: null, issues: [],
  selected: new Set(), selectedDatasets: [], split: { train: 0.8, eval: 0.1, test: 0.1, seed: 42 },
  run: null, checkpoints: [], device: 'auto', mode: 'design',
  totalSteps: 0, curStep: 0, lastUpdateS: 0, dsCompat: { ok: true },
};

let diagram, chartGrid, datasets, panes, panels, deployPanel, ws;
const logBuffer = [];

function toast(msg, kind = '') {
  const t = $('#toast'); t.textContent = msg; t.className = 'toast show ' + kind;
  setTimeout(() => t.classList.remove('show'), 3200);
}
window.__toast = toast;
window.__studioState = state;
window.__logBuffer = logBuffer;

async function boot() {
  initSplitters();
  try { await api.health(); $('#conn').classList.add('ok'); } catch { $('#conn').classList.add('bad'); }
  const [compute, emb, schema] = await Promise.all([api.compute(), api.embodiment(), api.archSchema()]);
  $('#deviceBadge').textContent = `${compute.device.toUpperCase()} · ${(compute.gpu_names || [''])[0]}`;
  state.emb = emb; state.compute = compute; state.schema = schema;

  diagram = new ArchDiagram($('#archCanvas'), $('#archLegend'), $('#archTip'), {
    onEdit: onDiagramEdit, data: api,
  });
  window.__diagram = diagram;   // debug/verification handle
  chartGrid = new ChartGrid($('#tbCharts'));   // bottom plot grid (bus-fed, splittable)

  datasets = new DatasetsPanel($('#dataPane'), state, () => { syncDatasets(); refresh(); });
  await datasets.load(api);
  window.__datasets = datasets.datasets;   // for the diagram's data-inspector window
  if (datasets.datasets[0]) diagram.setEpisode(datasets.datasets[0].dataset_id);  // show live data by default

  panes = new DockPanes({
    configEl: $('#configPane'), inputsEl: $('#inputsPane'), outputEl: $('#outputPane'), infoEl: $('#infoPane'),
    schema, embodiment: emb, state, onChange: onConfigEdit,
  });
  state.selection = { scope: 'whole_body', groups: [], cameras: emb.cameras.map(c => c.name) };
  applyDeepLink(schema, emb);
  panes.renderConfig(); panes.renderInputs(); panes.renderOutput();

  // 3D robot model in the CENTER (beside the architecture); shared with deploy
  let rig = null;
  try { const { Rig3D } = await import('./rig3d.js'); rig = new Rig3D($('#robotStage')); }
  catch (e) { $('#robotStage').innerHTML = '<div class="mon-empty">3D viewer unavailable (no WebGL)</div>'; }
  // permanent DEPLOYMENT config panel on the right
  try { deployPanel = new DeployPanel($('#deployPane'), { api, state, toast, rig, robotSub: $('#robotSub'), confirm: showConfirm }); }
  catch (e) { console.error('deploy panel', e); }

  // floating windows host (training plots added via + Window)
  const host = document.createElement('div'); host.className = 'fpanel-host'; document.body.appendChild(host);
  panels = new PanelManager(host);
  registerPanels(panels, { api, state, toast });
  wireAddWindow();

  wireTabs(); wireButtons(); wireDiagramControls(); wireRobotResize();
  refresh();
  // browser refresh / studio restart: reconnect to any live job so the training
  // visualization + progress persist (the detached job keeps running regardless).
  (async () => {
    try {
      const { runs } = await api.runs();
      const live = runs.find(r => ['running', 'preparing', 'queued'].includes(r.status));
      if (live) { toast('reconnected to ' + live.run_id, 'ok'); selectRun(live.run_id); }
      loadJobs();
    } catch {}
  })();
  setInterval(() => { api.health().then(() => $('#conn').classList.replace('bad', 'ok'))
    .catch(() => $('#conn').classList.replace('ok', 'bad')); }, 6000);
}

function applyDeepLink(schema, emb) {
  const q = new URLSearchParams(location.search);
  const arch = q.get('arch'); if (arch && schema.order.includes(arch)) state.arch = arch;
  const scope = q.get('scope'); if (scope && (emb.scopes[scope] || scope === 'custom')) state.selection.scope = scope;
  const cams = q.get('cams'); if (cams) state.selection.cameras = cams.split(',').filter(c => emb.cameras.some(x => x.name === c));
}

// ── dataset compatibility (dims + cameras must match to merge) ──────────────
function syncDatasets() {
  state.selectedDatasets = datasets.selectedRecords();
  const recs = state.selectedDatasets;
  if (recs[0] && diagram) diagram.setEpisode(recs[0].dataset_id);   // play the selected dataset in the diagram
  if (recs.length <= 1) { state.dsCompat = { ok: true }; return; }
  const a0 = recs[0].action_dim, cams0 = JSON.stringify([...recs[0].cameras].sort());
  const bad = recs.find(d => d.action_dim !== a0 || JSON.stringify([...d.cameras].sort()) !== cams0);
  if (bad) {
    state.dsCompat = { ok: false,
      msg: `"${bad.dataset_id}" has ${bad.action_dim}-D action / ${bad.cameras.length} cams — cannot merge with "${recs[0].dataset_id}" (${a0}-D / ${recs[0].cameras.length} cams).` };
    toast('datasets not mergeable: ' + state.dsCompat.msg, 'err');
  } else state.dsCompat = { ok: true };
}

// ── reactive: selection + diagram + validation ──────────────────────────────
let _rt;
function refresh() {
  clearTimeout(_rt);
  _rt = setTimeout(async () => {
    try {
      const { spec, issues, selection } = await api.archSpec({
        arch: state.arch, hp: state.hpByArch[state.arch] || {}, selection: state.selection });
      state.resolved = selection; state.issues = issues;
      diagram.setSpec(spec);
      $('#archTitle').textContent = spec.label;
      const sb2 = $('#srcBtn'); if (sb2) { sb2.classList.remove('on'); sb2.textContent = '⌁ code'; }
      panes.setResolved(selection); panes.renderInfo();
    } catch (e) { console.error(e); }
  }, 120);
}

// editing the config/inputs while VIEWING a run forks it into a fresh, editable
// design — the run itself (and its locked architecture) stays untouched.
function onConfigEdit() {
  if (state.mode === 'view') { state.run = null; setMode('design'); toast('editing forked this into a new job design', 'ok'); }
  refresh();
}

// diagram editor changed a layer count → write back into hyperparameters
function onDiagramEdit(patch) {
  const hp = state.hpByArch[state.arch] || (state.hpByArch[state.arch] = {});
  Object.assign(hp, patch.hp || {});
  if (patch.camera && state.selection) {   // delete a camera input
    state.selection.cameras = state.selection.cameras.filter(c => c !== patch.camera);
    panes.renderInputs();
  }
  panes.renderConfig();
  refresh();
}

function wireDiagramControls() {
  $('#zoomIn').onclick = () => diagram.zoomBy(1.2);
  $('#zoomOut').onclick = () => diagram.zoomBy(1 / 1.2);
  $('#zoomFit').onclick = () => diagram.fit();
  $('#tidyBtn').onclick = () => diagram.autoOrganize();
  $('#saveBlockBtn').onclick = () => {
    const name = prompt('Save this view as a reusable block — name:');
    if (name) diagram.saveBlock(name.trim());
  };
  // one unified "＋ Add block" menu (elementary layers + composite blocks + saved)
  $('#addBlockBtn').onclick = (e) => { e.stopPropagation(); diagram.openAddBlockMenu(e.currentTarget); };
  // ⚙ compile — COMPILE THE GRAPH: type-check every connection's dimensions,
  // flag unwired blocks, and offer/apply auto-fixes (adapters, auto-complete).
  $('#compileBtn').onclick = () => diagram.compileGraph();
  // ⌁ code — ground-truth view built from the real nn.Module (read-only reference)
  const sb2 = $('#srcBtn');
  sb2.onclick = async () => {
    if (diagram.isCompiled()) { sb2.classList.remove('on'); sb2.textContent = '⌁ code'; refresh(); return; }
    sb2.textContent = '⌁ building…';
    try {
      const g = await api.archCompile({ arch: state.arch, hp: state.hpByArch[state.arch] || {}, selection: state.selection });
      if (g.ok) { diagram.showCompiled(g); sb2.classList.add('on'); sb2.textContent = '⌁ code ✓'; }
      else { toast('code view: ' + (g.reason || 'needs pretrained weights'), 'warn'); sb2.textContent = '⌁ code'; }
    } catch (e) { toast('code view failed: ' + e.message, 'err'); sb2.textContent = '⌁ code'; }
  };
  $('#animToggle').onclick = (e) => {
    const on = e.target.classList.toggle('on'); diagram.setAnimate(on);
    e.target.textContent = on ? '▶' : '⏸';
  };
}

function wireRobotResize() {
  const h = $('#robotResize'); if (!h) return;
  h.addEventListener('pointerdown', (e) => {
    e.preventDefault(); h.setPointerCapture(e.pointerId);
    const startX = e.clientX;
    const startW = parseInt(getComputedStyle(document.documentElement).getPropertyValue('--robot-w')) || 322;
    const move = (ev) => {
      const w = Math.max(200, Math.min(680, startW - (ev.clientX - startX)));
      document.documentElement.style.setProperty('--robot-w', w + 'px');
      window.dispatchEvent(new Event('studio-resize'));
    };
    const up = (ev) => { h.releasePointerCapture(ev.pointerId); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  });
}

function wireTabs() {
  document.querySelectorAll('.rail-tab').forEach(b => b.onclick = () => {
    document.querySelectorAll('.rail-tab').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('.rail-pane').forEach(p => p.classList.toggle('on', p.dataset.pane === b.dataset.tab));
    if (b.dataset.tab === 'jobs') loadJobs();
  });
}

// ── train / pause / retrain ─────────────────────────────────────────────────
function wireButtons() {
  $('#trainBtn').onclick = startTraining;
  $('#newJobBtn').onclick = newJob;
  $('#pauseBtn').onclick = async () => { if (state.run) { await api.pause(state.run); toast('pause requested — saving checkpoint', 'ok'); } };
  $('#retrainBtn').onclick = retrain;
  // Deploy group (mirrors the right-rail Deployment panel)
  $('#deployTopBtn').onclick = () => { if (deployPanel) deployPanel.deploy(); };
  $('#redeployBtn').onclick = () => { if (deployPanel) deployPanel.deploy(); };
  $('#depPauseBtn').onclick = () => { if (deployPanel) deployPanel.stopDeploy(); };
  wireSplitMenu();
  document.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') startTraining(); });
}

// Split the bottom visualizer into 2/3/4 sub-windows, or drop in a preset.
function wireSplitMenu() {
  const sb = $('#splitBtn'), menu = $('#splitMenu');
  if (!sb || !menu) return;
  menu.innerHTML = `
    <div class="split-sec">Layout</div>
    <button data-n="1">◻ single</button>
    <button data-n="2">◧ 2 plots</button>
    <button data-n="3">▥ 3 plots</button>
    <button data-n="4">▦ 4 plots</button>
    <div class="split-sec">Preset</div>
    <button data-preset="standard">★ Loss · Grad · GPU/CPU</button>`;
  sb.onclick = (e) => {
    e.stopPropagation();
    const hidden = menu.classList.contains('hidden');
    if (hidden) {   // position above the button (escapes the trainbar's overflow:hidden)
      const rc = sb.getBoundingClientRect();
      menu.style.left = rc.left + 'px';
      menu.style.bottom = (window.innerHeight - rc.top + 6) + 'px';
    }
    menu.classList.toggle('hidden');
  };
  document.addEventListener('click', () => menu.classList.add('hidden'));
  menu.onclick = (e) => {
    const b = e.target.closest('button'); if (!b) return;
    e.stopPropagation(); menu.classList.add('hidden');
    if (b.dataset.n) { chartGrid.setLayout(+b.dataset.n); toast(`${b.dataset.n}-up plots — pick a metric per pane`, 'ok'); }
    else if (b.dataset.preset === 'standard') { chartGrid.setPreset(['loss', 'grad', 'system']); toast('preset: loss · grad · GPU/CPU', 'ok'); }
  };
}

function collectParams() {
  return {
    arch: state.arch, hp: state.hpByArch[state.arch] || {}, selection: state.selection,
    datasets: state.selectedDatasets.map(d => d.dataset_id),
    split: [state.split.train, state.split.eval, state.split.test], seed: state.split.seed,
    device: state.device, name: `${state.arch}·${state.selection.scope}`,
  };
}

// Centered confirmation modal — a key/value summary plus optional EDITABLE fields:
//   inputs:  [{key,label,value,type:'number'|'text'}]                (text boxes)
//            [{key,label,type:'select',options:[{v,l}],value}]       (dropdown)
//            [{key,label,type:'checks',options:[{v,l,checked}]}]     (multi-check)
//   details: [{label, html}]  — collapsible sections (e.g. full hyperparameters)
// Resolves false on cancel, or {values:{key:val|array}} on confirm (true if no inputs).
function showConfirm({ title, sub, rows = [], inputs = [], details = [], confirmLabel = 'Confirm', kind = '' }) {
  return new Promise((resolve) => {
    const back = document.createElement('div'); back.className = 'confirm-back';
    const field = (f) => {
      if (f.type === 'select') {
        const opts = f.options.map(o => `<option value="${o.v}" ${String(o.v) === String(f.value) ? 'selected' : ''}>${o.l}</option>`).join('');
        return `<div class="cf-row"><span class="cf-k">${f.label}</span><select class="cf-input cf-select" id="cfi_${f.key}">${opts}</select></div>`;
      }
      if (f.type === 'checks') {
        const boxes = f.options.map(o => `<label class="cf-check"><input type="checkbox" data-ck="${f.key}" value="${o.v}" ${o.checked ? 'checked' : ''}><span>${o.l}</span></label>`).join('');
        return `<div class="cf-row cf-row-col"><span class="cf-k">${f.label}</span><div class="cf-checks">${boxes}</div></div>`;
      }
      return `<div class="cf-row"><span class="cf-k">${f.label}</span>
        <input class="cf-input" id="cfi_${f.key}" type="${f.type || 'number'}" value="${f.value}"></div>`;
    };
    const detailsHtml = details.filter(Boolean).map((d, i) =>
      `<div class="cf-det"><button class="cf-det-btn" data-det="${i}">▸ ${d.label}</button>
        <div class="cf-det-body hidden" id="cfd_${i}">${d.html}</div></div>`).join('');
    back.innerHTML = `<div class="confirm">
      <div class="confirm-head"><span class="confirm-title">${title}</span>${sub ? `<span class="confirm-sub">${sub}</span>` : ''}</div>
      <div class="confirm-body">${rows.filter(Boolean).map(([k, v]) => `<div class="cf-row"><span class="cf-k">${k}</span><span class="cf-v">${v}</span></div>`).join('')}${inputs.map(field).join('')}${detailsHtml}</div>
      <div class="confirm-acts"><button class="gbtn" id="cfCancel">Cancel</button><button class="gbtn primary ${kind}" id="cfOk">${confirmLabel}</button></div>
    </div>`;
    document.body.appendChild(back);
    back.querySelectorAll('.cf-det-btn').forEach(b => b.onclick = () => {
      const el = back.querySelector('#cfd_' + b.dataset.det);
      const open = el.classList.toggle('hidden');
      b.textContent = (open ? '▸ ' : '▾ ') + b.textContent.slice(2);
    });
    const collect = () => {
      const v = {};
      inputs.forEach(f => {
        if (f.type === 'checks') v[f.key] = [...back.querySelectorAll(`[data-ck="${f.key}"]:checked`)].map(x => x.value);
        else { const el = back.querySelector('#cfi_' + f.key); v[f.key] = el ? el.value : f.value; }
      });
      return v;
    };
    const done = (ok) => { const vals = ok ? collect() : null; back.remove(); document.removeEventListener('keydown', esc); resolve(ok ? (inputs.length ? { values: vals } : true) : false); };
    const esc = (e) => { if (e.key === 'Escape') done(false); if (e.key === 'Enter' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') done(true); };
    document.addEventListener('keydown', esc);
    back.onclick = (e) => { if (e.target === back) done(false); };
    back.querySelector('#cfCancel').onclick = () => done(false);
    back.querySelector('#cfOk').onclick = () => done(true);
    back.querySelector('#cfOk').focus();
  });
}
window.__confirm = showConfirm;

// full hyperparameter table as collapsible-details HTML
function hpDetailsHtml() {
  const hp = state.hpByArch[state.arch] || {};
  const A = (state.schema || {}).arches ? state.schema.arches[state.arch] : null;
  const fields = A ? A.fields : [];
  const rows = fields.map(f => {
    const v = hp[f.name] !== undefined ? hp[f.name] : f.default;
    return `<div class="cf-row"><span class="cf-k">${f.label}</span><span class="cf-v">${v}</span></div>`;
  }).join('');
  return rows || '<div class="cf-row"><span class="cf-v">defaults</span></div>';
}
// dataset checkboxes (curator exports only — internal studio artifacts hidden),
// restricted to FORMAT-COMPATIBLE datasets (same action dim + camera set)
function datasetChecks(selectedIds) {
  const all = ((datasets && datasets.datasets) || []).filter(d => d.source === 'curator');
  const sel = all.filter(d => selectedIds.includes(d.dataset_id));
  const ref = sel[0] || all[0];
  return all.filter(d => !ref || (d.action_dim === ref.action_dim &&
      JSON.stringify([...d.cameras].sort()) === JSON.stringify([...ref.cameras].sort())))
    .map(d => ({ v: d.dataset_id, l: `${d.dataset_id} · ${d.n_episodes} ep`, checked: selectedIds.includes(d.dataset_id) }));
}

// Operator auth is handled by the styled sign-in popup (auth.js) that api.js
// triggers on any mutating action, so the confirm modals no longer carry a
// passkey field — the popup asks for username + passkey once per session.

async function startTraining() {
  const errs = (state.issues || []).filter(i => i.level === 'error');
  if (errs.length) return toast('fix config: ' + errs[0].msg, 'err');
  if (!state.dsCompat.ok) return toast('datasets not mergeable — ' + state.dsCompat.msg, 'err');
  const params = collectParams();
  if (!params.datasets.length) return toast('select at least one dataset', 'err');
  // confirmation summary before committing — with LAST-MINUTE edits: steps, batch,
  // lr, and extra (format-compatible) datasets, plus the full hp table on demand.
  const sel = state.resolved || {};
  const hp = state.hpByArch[state.arch] || {};
  const sp = state.split;
  const res = await showConfirm({
    title: 'Start training?', sub: state.arch,
    confirmLabel: '▶ Start training', kind: 'go',
    rows: [
      ['Architecture', state.arch],
      ['Split', `train ${Math.round(sp.train * 100)}% · eval ${Math.round(sp.eval * 100)}% · test ${Math.round(sp.test * 100)}%`],
      ['Input', `${(sel.cameras || []).length} cams · proprio ${sel.obs_state_dim ?? '?'}-D (${sel.scope || ''})`],
      ['Output', `action ${sel.action_dim ?? '?'}-D (${sel.action_scope || sel.scope || ''})`],
      ['Device', (state.compute && state.compute.device || 'auto').toUpperCase()],
    ],
    inputs: [
      { key: 'datasets', label: 'Datasets', type: 'checks', options: datasetChecks(params.datasets) },
      { key: 'steps', label: 'Training steps', value: hp.steps || 5000 },
      { key: 'batch_size', label: 'Batch size', value: hp.batch_size || 8 },
      { key: 'lr', label: 'Learning rate', type: 'text', value: hp.optimizer_lr ?? 1e-5 },
    ],
    details: [{ label: 'All hyperparameters', html: hpDetailsHtml() }],
  });
  if (!res) return;
  const v = res.values || {};
  if (v.datasets && v.datasets.length) params.datasets = v.datasets;
  if (!params.datasets.length) return toast('select at least one dataset', 'err');
  const hpNew = { ...hp, steps: Math.max(1, parseInt(v.steps) || hp.steps || 5000),
    batch_size: Math.max(1, parseInt(v.batch_size) || hp.batch_size || 8),
    optimizer_lr: parseFloat(v.lr) || hp.optimizer_lr };
  state.hpByArch[state.arch] = hpNew; params.hp = hpNew;
  $('#trainBtn').disabled = true;
  try {
    state.totalSteps = (state.hpByArch[state.arch] || {}).steps || 5000;
    const { run_id } = await api.trainStart(params);
    attachRun(run_id); toast('training launched · ' + run_id, 'ok');
  } catch (e) { toast('launch failed: ' + e.message, 'err'); }
  finally { $('#trainBtn').disabled = false; }
}

async function retrain() {
  if (!state.run && !state.checkpoints.length) return toast('no run/checkpoint to retrain from', 'err');
  const hp = state.hpByArch[state.arch] || {};
  const doneSteps = state.curStep || 0;
  const total = hp.steps || state.totalSteps || 5000;
  const run = state.run ? await api.run(state.run).catch(() => null) : null;
  const ckpts = (run && run.checkpoints) || state.checkpoints || [];
  if (!ckpts.length) return toast('no checkpoints saved for this run', 'err');
  const curDatasets = run ? (run.dataset_id || '').split(',').filter(Boolean)
                          : state.selectedDatasets.map(d => d.dataset_id);
  // RESUME continues the SAME run (optimizer + step counter pick up — only the
  // remaining steps are trained). Adding datasets switches to FINE-TUNE: a new
  // run initialised from the checkpoint's weights (post-training on more data).
  const res = await showConfirm({
    title: 'Resume / fine-tune?', sub: state.arch, confirmLabel: '↻ Continue training', kind: 'go',
    rows: [
      ['Run', state.run || '—'],
      ['Progress', `${doneSteps} / ${total} steps done`],
      ['Mode', 'resume = same run, trains only the remaining steps · adding datasets = fine-tune as a new run'],
    ],
    inputs: [
      { key: 'checkpoint', label: 'From checkpoint', type: 'select',
        value: ckpts[ckpts.length - 1].path,
        options: [...ckpts].reverse().map(c => ({ v: c.path, l: `step ${c.step}` })) },
      { key: 'steps', label: 'Total steps (target)', value: total },
      { key: 'datasets', label: 'Datasets (add = fine-tune)', type: 'checks',
        options: datasetChecks(curDatasets) },
    ],
    details: [{ label: 'All hyperparameters', html: hpDetailsHtml() }],
  });
  if (!res) return;
  const v = res.values || {};
  const newSteps = Math.max(1, parseInt(v.steps) || total);
  const pickedDs = v.datasets || curDatasets;
  const sameData = JSON.stringify([...pickedDs].sort()) === JSON.stringify([...curDatasets].sort());
  state.hpByArch[state.arch] = { ...hp, steps: newSteps };
  state.totalSteps = newSteps;
  try {
    if (sameData && state.run) {
      // TRUE RESUME — same run_id, step counter continues from the checkpoint
      const r = await api.resume(state.run, { steps: newSteps, checkpoint: v.checkpoint });
      toast(`resuming ${r.run_id} from ${(r.resumed_from || '').split('/').pop()} → ${newSteps} steps total`, 'ok');
      setTimeout(() => selectRun(state.run), 800);
    } else {
      // FINE-TUNE — new run, weights initialised from the chosen checkpoint
      const { run_id, from_checkpoint } = await api.trainRetrain({
        ...collectParams(), hp: state.hpByArch[state.arch], datasets: pickedDs,
        run_id: state.run, pretrained_ckpt: v.checkpoint });
      attachRun(run_id);
      toast(`fine-tuning as ${run_id} from ${from_checkpoint ? from_checkpoint.split('/').pop() : 'scratch'} on ${pickedDs.length} dataset(s)`, 'ok');
    }
  } catch (e) { toast('retrain failed: ' + e.message, 'err'); }
}

function connectWS(runId) {
  if (ws) try { ws.close(); } catch {}
  $('#pauseBtn').disabled = false;
  let lastStatus = '';
  ws = api.wsTrain(runId, (msg) => {
    if (state.run !== runId) return;   // ignore stale socket after switching runs
    if (msg.metrics && msg.metrics.length) { bus.emit('metricRows', msg.metrics); ingestProgress(msg.metrics); }
    if (msg.logs && msg.logs.length) { for (const l of msg.logs) logBuffer.push(l); bus.emit('logRows', msg.logs); }
    if (msg.checkpoints) state.checkpoints = msg.checkpoints;
    if (msg.status && msg.status !== lastStatus) {
      lastStatus = msg.status; $('#runChip').textContent = `${runId} · ${msg.status}`;
      $('#runChip').className = 'run-chip ' + msg.status; setTrainbar(msg.status);
      if (state.mode === 'view' && state.run === runId) setMode('view', runId, msg.status);
      if (['done', 'failed', 'stopped', 'paused', 'interrupted'].includes(msg.status)) {
        $('#pauseBtn').disabled = true;
        toast(`run ${msg.status}`, msg.status === 'done' ? 'ok' : msg.status === 'failed' || msg.status === 'interrupted' ? 'err' : '');
        loadJobs(); if (deployPanel) deployPanel.refreshRuns();
      }
    }
  });
}

function attachRun(runId) {   // fresh run — reset everything, then stream
  state.run = runId; state.checkpoints = []; state.curStep = 0; state.lastUpdateS = 0; state.curEpoch = 0; state._lastEpoch = null;
  state.runEnded = false;
  bus.emit('metricReset'); logBuffer.length = 0; bus.emit('logReset');
  setTrainbar('preparing', 0);
  $('#runChip').textContent = runId;
  setMode('view', runId, 'preparing');   // you're now watching the run you launched
  connectWS(runId);
}

// Is the centre canvas a NEW-job design surface, or a view of an existing run?
function setMode(mode, runId, status) {
  state.mode = mode;
  if (diagram) diagram.setReadOnly(mode === 'view');   // lock the arch while viewing a run
  const el = $('#archMode'); if (!el) return;
  if (mode === 'view') {
    el.className = 'arch-mode view';
    el.textContent = `viewing ${runId}${status ? ' · ' + status : ''}`;
  } else {
    el.className = 'arch-mode design';
    el.textContent = 'designing new job';
  }
}

// Fresh design surface for the next job. The currently-running job keeps training
// (it's detached) and stays in the Jobs tab — click it there to watch it again.
function newJob() {
  if (ws) { try { ws.close(); } catch {} ws = null; }
  state.run = null; state.checkpoints = []; state.curStep = 0; state.totalSteps = 0; state._lastEpoch = null;
  bus.emit('metricReset'); logBuffer.length = 0; bus.emit('logReset');
  setTrainbar('idle', 0);
  $('#pauseBtn').disabled = true;
  $('#runChip').textContent = 'no run'; $('#runChip').className = 'run-chip';
  setMode('design');
  loadJobs();
  toast('new job — design your architecture, then ▶ Train', 'ok');
}

// ── clean training bar (progress + ETA + stats) ─────────────────────────────
function ingestProgress(rows) {
  let loss, gpu, cpu, epoch, dl, upd, grad, lr;
  for (const r of rows) {
    if (r.key === 'steps') state.curStep = Math.max(state.curStep, r.value);
    else if (r.key === 'update_s') { state.lastUpdateS = r.value; upd = r.value; }
    else if (r.key === 'dataloading_s') dl = r.value;
    else if (r.key === 'epochs') { epoch = r.value; state.curEpoch = r.value; }
    else if (r.key === 'loss' && r.mode === 'train') loss = r.value;
    else if (r.key === 'grad_norm') grad = r.value;
    else if (r.key === 'lr') lr = r.value;
    else if (r.key === 'system.gpu_util') gpu = r.value;
    else if (r.key === 'system.cpu_percent') cpu = r.value;
  }
  const run = state.run; if (!run) return;
  // epoch-completion feedback in the log
  if (epoch != null) {
    const e = Math.floor(epoch);
    if (state._lastEpoch == null) state._lastEpoch = e;
    else if (e > state._lastEpoch) {
      state._lastEpoch = e;
      const line = `✓ epoch ${e} complete · step ${state.curStep}${loss != null ? ` · loss ${loss.toFixed(3)}` : ''}`;
      logBuffer.push({ line }); bus.emit('logRows', [{ line }]);
    }
  }
  const total = state.totalSteps || 1, step = state.curStep;
  setTrainbar(null, step / total);
  // per-epoch bar (fractional part of the epochs metric) — moves fast so it
  // never looks frozen even when the overall run bar barely moves.
  const ep = state.curEpoch != null ? state.curEpoch : (epoch || 0);
  const frac = ep - Math.floor(ep);
  $('#tbEpFill').style.width = (frac * 100).toFixed(0) + '%';
  $('#tbEpPct').textContent = ep > 0 ? `#${Math.floor(ep)}` : '';
  const perStep = (upd || state.lastUpdateS || 0) + (dl || 0);
  const remain = Math.max(0, total - step) * (perStep || state.lastUpdateS || 0.001);
  $('#tbEta').textContent = step > 0 ? `~${fmtTime(remain)} left` : '';
  const ips = perStep > 0 ? (1 / perStep).toFixed(2) : '—';
  const chips = [
    ['iter', `${step} / ${total}`],
    epoch != null ? ['epoch', ep.toFixed(1)] : null,
    ['iter/s', ips],
    loss != null ? ['loss', loss.toFixed(3)] : null,
    grad != null ? ['grad', grad.toFixed(2)] : null,
    lr != null ? ['lr', lr.toExponential(1)] : null,
    gpu != null ? ['gpu', gpu.toFixed(0) + '%'] : null,
    cpu != null ? ['cpu', cpu.toFixed(0) + '%'] : null,
  ].filter(Boolean);
  $('#tbStats').innerHTML = chips.map(([k, v]) => `<span class="tb-chip"><i>${k}</i>${v}</span>`).join('');
}

function setTrainbar(status, pct) {
  if (status != null) {
    const el = $('#tbStatus'); el.textContent = status; el.className = 'tb-status ' + status;
    $('#tbTitle').textContent = status === 'idle' ? 'Training' : `Training · ${state.arch}`;
  }
  if (pct != null) {
    const p = Math.max(0, Math.min(1, pct));
    $('#tbFill').style.width = (p * 100).toFixed(1) + '%';
    $('#tbPct').textContent = (p * 100).toFixed(0) + '%';
  }
}
function fmtTime(s) {
  s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${ss}s` : `${ss}s`;
}

function loadRuns() { if (deployPanel) deployPanel.refreshRuns(); loadJobs(); }

const JOB_CLR = { done: 'ok', running: 'run', preparing: 'warn', queued: 'warn', paused: 'warn', failed: 'err', stopped: 'err', interrupted: 'int' };
function fmtHrs(h) { return h == null ? '' : (h < 0.05 ? Math.round(h * 3600) + 's' : h < 1 ? Math.round(h * 60) + 'm' : h.toFixed(2) + 'h'); }
function fmtDateTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
         d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function jobCard(r) {
  const clr = JOB_CLR[r.status] || '';
  const canRe = ['done', 'paused', 'stopped', 'interrupted', 'failed'].includes(r.status);
  const isActive = ['running', 'preparing', 'queued'].includes(r.status);
  const ep = r.epoch != null ? `ep ${(+r.epoch).toFixed(1)}` : '';
  const it = r.step != null ? `iter ${r.step}` : '';
  return `<div class="job-card ${r.status === 'running' ? 'running' : ''} ${r.run_id === state.run ? 'on' : ''}" data-run="${r.run_id}">
    <div class="job-top"><span class="job-dot ${clr}"></span><b class="job-name">${r.name || r.arch}</b>
      <span class="job-status ${clr}">${r.status}</span></div>
    <div class="job-meta">${r.arch} · ${r.scope || ''} · ${(r.dataset_id || '').split(',')[0]}</div>
    <div class="job-meta2">${[it, ep, fmtHrs(r.trained_hours), r.device].filter(Boolean).join(' · ')}</div>
    <div class="job-times">
      <span title="created">◷ created ${fmtDateTime(r.created_at)}</span>
      <span title="last active">↻ active ${fmtDateTime(r.updated_at)}</span>
    </div>
    ${r.output_dir ? `<div class="path-row" title="model stored at ${r.output_dir}"><span class="path-ico">💾</span><span class="path-txt">${r.output_dir}</span></div>` : ''}
    <div class="job-acts"><button class="mini job-open">view</button>${isActive ? '<button class="mini job-kill">✕ kill</button>' : ''}${canRe ? '<button class="mini job-re">retrain</button>' : ''}</div>
  </div>`;
}
async function loadJobs() {
  const el = $('#jobsPane'); if (!el) return;
  try {
    const { runs } = await api.runs();
    const active = runs.filter(r => ['running', 'preparing', 'queued'].includes(r.status));
    const past = runs.filter(r => !['running', 'preparing', 'queued'].includes(r.status));
    el.innerHTML = `<div class="job-refresh"><button class="mini" id="jobsRefresh">↻ refresh</button>
        <span class="job-count">${runs.length} jobs</span></div>` +
      (active.length ? `<div class="ds-group-h">Active</div>${active.map(jobCard).join('')}` : '') +
      (past.length ? `<div class="ds-group-h">History</div>${past.map(jobCard).join('')}` : '') +
      (!runs.length ? '<div class="ds-empty">no training jobs yet</div>' : '');
    el.querySelector('#jobsRefresh').onclick = () => loadJobs();
    el.querySelectorAll('.job-card').forEach(c => {
      c.querySelector('.job-open').onclick = () => selectRun(c.dataset.run);
      const rt = c.querySelector('.job-re'); if (rt) rt.onclick = (e) => { e.stopPropagation(); state.run = c.dataset.run; retrain(); };
      const kl = c.querySelector('.job-kill'); if (kl) kl.onclick = async (e) => {
        e.stopPropagation();
        const rid = c.dataset.run;
        const ok = await showConfirm({
          title: 'Kill this job?', sub: rid, confirmLabel: '✕ Kill job', kind: 'danger',
          rows: [
            ['Job', c.querySelector('.job-name').textContent],
            ['Effect', 'training stops after the current step'],
            ['Kept', 'all checkpoints, metrics and logs stay — resumable from the last checkpoint'],
          ],
        });
        if (!ok) return;
        kl.disabled = true; kl.textContent = 'killing…';
        try { await api.stop(rid); toast('kill requested — stopping', 'ok'); }
        catch (err) { toast('kill failed: ' + err.message, 'err'); kl.disabled = false; kl.textContent = '✕ kill'; return; }
        setTimeout(loadJobs, 600);
      };
    });
  } catch {}
}

async function selectRun(runId) {
  const run = await api.run(runId);
  state.run = runId; state.checkpoints = run.checkpoints || []; state.curStep = 0; state._lastEpoch = null;
  const meta = JSON.parse(run.meta_json || '{}');
  state.totalSteps = (meta.hp || {}).steps || state.totalSteps || 1;
  // pull the EXACT architecture this run was trained with into the canvas, so the
  // centre mirrors what the job used (it becomes a read-only record via setMode).
  if (run.arch) state.arch = run.arch;
  if (meta.hp) state.hpByArch[state.arch] = { ...meta.hp };
  if (meta.selection) state.selection = meta.selection;
  bus.emit('metricReset'); logBuffer.length = 0; bus.emit('logReset');
  // re-load the persisted terminal + training logs so a stopped/failed job stays
  // fully debuggable (view → scroll its whole log even after it died)
  try {
    const { logs } = await api.logs(runId);
    if (logs && logs.length) { for (const l of logs) logBuffer.push(l); bus.emit('logRows', logs); }
  } catch {}
  for (const key of run.metric_keys || []) {
    for (const mode of ['train', 'eval', 'system']) {
      try {
        const { series } = await api.series(runId, key, mode);
        if (series.length) { const rows = series.map(s => ({ key, mode, step: s.step, value: s.value, t: s.t })); bus.emit('metricRows', rows); ingestProgress(rows); }
      } catch {}
    }
  }
  $('#runChip').textContent = `${runId} · ${run.status}`; $('#runChip').className = 'run-chip ' + run.status;
  // viewing a job → default the bottom to the 3-up plot view (loss+eval · grad · GPU/CPU),
  // unless the user already arranged a custom split
  if (chartGrid && chartGrid.count() <= 1) chartGrid.setPreset(['loss', 'grad', 'system']);
  setMode('view', runId, run.status);   // the centre now mirrors this run's config (read-only)
  // rebuild the diagram + config/inputs panes from the run's stored arch/hp/selection
  if (panes) { try { panes.renderConfig(); panes.renderInputs(); } catch {} }
  refresh();
  state.runEnded = !['running', 'preparing', 'queued'].includes(run.status);
  if (!state.runEnded) { setTrainbar(run.status); connectWS(runId); }
  else {
    // stopped/failed/interrupted → show the ACTUAL progress reached (not 100%)
    const frac = run.status === 'done' ? 1 : (state.totalSteps ? Math.min(1, state.curStep / state.totalSteps) : 0);
    setTrainbar(run.status, frac);
  }
  loadJobs();
}

function wireAddWindow() {
  const btn = $('#addWinBtn'), menu = $('#addWinMenu');
  btn.onclick = (e) => {
    e.stopPropagation();
    // training plots only — the robot/deploy view is the permanent right rail
    menu.innerHTML = panels.list().filter(p => p.type !== 'deploy')
      .map(p => `<button data-t="${p.type}">${p.label}</button>`).join('');
    menu.classList.toggle('hidden');
    menu.querySelectorAll('button').forEach(b => b.onclick = () => { panels.add(b.dataset.t); menu.classList.add('hidden'); });
  };
  document.addEventListener('click', () => menu.classList.add('hidden'));
}

boot();
