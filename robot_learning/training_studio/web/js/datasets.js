// LEFT rail — dataset library with full stats, multi-select, and the train/
// eval/test split control (an early parameter). Selected datasets feed the run.

export class DatasetsPanel {
  constructor(el, state, onChange) {
    this.el = el; this.state = state; this.onChange = onChange;
    this.datasets = [];
  }

  async load(api) {
    this.api = api || this.api;
    const { datasets } = await this.api.datasets();
    this.datasets = datasets.filter(d => !d.error);
    if (!this.state.selected) this.state.selected = new Set();
    this.render();
    this._wireDrop();
  }

  async importPath(path) {
    if (!path) return;
    try {
      const rec = await this.api.importDataset(path.trim());
      await this.load();
      this.state.selected.add(rec.dataset_id);
      this.render();
      this.onChange();
      window.__toast && window.__toast('imported ' + rec.dataset_id, 'ok');
    } catch (e) {
      window.__toast && window.__toast('import failed: ' + e.message, 'err');
    }
  }

  _wireDrop() {
    const body = this.el;
    body.ondragover = (e) => { e.preventDefault(); body.classList.add('drop-on'); };
    body.ondragleave = () => body.classList.remove('drop-on');
    body.ondrop = (e) => {
      e.preventDefault(); body.classList.remove('drop-on');
      const it = e.dataTransfer.items && e.dataTransfer.items[0];
      const entry = it && it.webkitGetAsEntry && it.webkitGetAsEntry();
      const name = entry ? entry.name : (e.dataTransfer.files[0] && e.dataTransfer.files[0].name) || '';
      // browsers can't hand us the absolute path — prefill the import box with the
      // dropped folder name and let the user complete/confirm the path.
      this._openImport(name);
    };
  }

  _openImport(prefill = '') {
    const box = this.el.querySelector('#importBox');
    if (!box) return;
    box.classList.remove('hidden');
    const inp = box.querySelector('#importPath');
    inp.value = prefill && !prefill.startsWith('/') ? prefill : prefill;
    inp.focus();
  }

  _fmtStats(d) {
    return `${d.n_episodes} ep · ${d.total_frames} frames · ${d.hours}h · ${d.size_mb}MB`;
  }

  // information-rich detail panel — format, size, path, per-modality data + rates,
  // camera specs, and the full feature schema.
  _detail(d) {
    const kv = (k, v) => `<div class="dd-row"><span class="dd-k">${k}</span><span class="dd-v">${v}</span></div>`;
    const shp = (a) => (a && a.length) ? a.join('×') : '';
    const mods = (d.modality_report || []).map(m => `
      <tr class="${m.present ? 'on' : 'off'}">
        <td class="dd-yn">${m.present ? '✓' : '·'}</td><td>${m.type}</td>
        <td>${m.present ? (m.count || 1) : ''}</td>
        <td>${m.hz != null ? m.hz + ' Hz' : '—'}</td>
        <td class="dd-shp">${shp(m.shape)}</td></tr>`).join('');
    const cams = (d.cameras || []).map(c => {
      const sh = (d.camera_shapes || {})[c] || [];
      const res = sh.length >= 2 ? `${sh[1]}×${sh[0]}${sh[2] ? '×' + sh[2] : ''}` : '';
      return `<div class="dd-cam"><b>${c}</b><span>${res} · ${d.fps} Hz</span></div>`;
    }).join('');
    const feats = Object.entries(d.features || {}).map(([k, v]) =>
      `<div class="dd-feat"><span class="dd-fk">${k}</span><span class="dd-fv">${shp(v.shape) || '—'}${v.dtype ? ' · ' + v.dtype : ''}</span></div>`).join('');
    return `
      <div class="dd-grid">
        ${kv('Format', 'LeRobot ' + (d.codebase_version || ''))}
        ${kv('Robot', d.robot_type || '—')}
        ${kv('Episodes', d.n_episodes)}
        ${kv('Frames', d.total_frames + ' (' + d.total_timesteps + ' steps)')}
        ${kv('Duration', d.hours + ' h · ' + d.seconds + ' s')}
        ${kv('Size on disk', d.size_mb + ' MB')}
        ${kv('Sample rate', d.fps + ' Hz')}
        ${kv('Control', d.control_type || '—')}
        ${kv('Obs state', d.obs_state_dim + '-D · ' + (d.obs_state_dtype || ''))}
        ${kv('Action', d.action_dim + '-D · ' + (d.action_dtype || ''))}
        ${kv('Tasks', (d.tasks || []).join(', ') || '—')}
        ${kv('Version', d.version_hash || '—')}
      </div>
      <div class="dd-sec">What data it has · per-modality rate</div>
      <table class="dd-mods"><thead><tr><th></th><th>modality</th><th>#</th><th>rate</th><th>shape</th></tr></thead><tbody>${mods}</tbody></table>
      <div class="dd-sec">Cameras · ${d.n_cameras}</div>
      <div class="dd-cams">${cams || '<span class="dd-none">none</span>'}</div>
      <div class="dd-sec">Feature schema · ${(d.feature_keys || []).length} keys</div>
      <div class="dd-feats">${feats}</div>
      <div class="dd-sec">Storage path</div>
      <div class="dd-path" title="click to select">${d.path}</div>`;
  }

  render() {
    const s = this.state;
    const split = s.split || (s.split = { train: 0.8, eval: 0.1, test: 0.1, seed: 42 });
    const chips = (d) => `
      <div class="ds-tags">
        <span>${d.n_cameras} cam</span><span>${d.action_dim}-D act</span>
        <span>${d.fps}fps</span><span title="${d.control_type}">joint pos</span>
      </div>`;
    this._expanded = this._expanded || new Set();
    const card = (d) => {
      const on = s.selected.has(d.dataset_id);
      const open = this._expanded.has(d.dataset_id);
      return `<div class="ds-card ${on ? 'on' : ''}" data-id="${d.dataset_id}">
        <div class="ds-top"><span class="ds-check">${on ? '✓' : '+'}</span>
          <span class="ds-name" title="${d.path}">${d.dataset_id}</span>
          <button class="ds-view" data-vid="${d.dataset_id}" title="Show full dataset details">${open ? '⌃ details' : '⌄ details'}</button>
          <span class="ds-src ${d.source}">${d.source}</span></div>
        <div class="ds-nums">${this._fmtStats(d)}</div>
        ${chips(d)}
        <div class="path-row" title="${d.path}"><span class="path-ico">📁</span><span class="path-txt">${d.path}</span></div>
        <div class="ds-detail ${open ? 'open' : ''}">${open ? this._detail(d) : ''}</div>
      </div>`;
    };
    const nSel = s.selected.size;
    // only the CURATOR exports are user-pickable; studio-derived datasets (combined
    // views, scoped slices, v3.0 copies) are internal build artifacts — hidden.
    const curated = this.datasets.filter(d => d.source === 'curator');
    this.el.innerHTML = `
      <div class="ds-import-row">
        <button class="mini prim" id="importBtn">＋ Import dataset</button>
        <span class="ds-drop-hint">or drop a folder</span>
      </div>
      <div class="import-box hidden" id="importBox">
        <input type="text" id="importPath" placeholder="/absolute/path/to/lerobot_dataset">
        <div class="import-btns"><button class="mini prim" id="importGo">Import</button>
          <button class="mini" id="importCancel">Cancel</button></div>
      </div>
      <div class="split-box">
        <div class="split-h">Train / Eval / Test split</div>
        <div class="split-bars" id="splitBars">
          ${['train', 'eval', 'test'].map(k => `
            <label class="split-row"><span>${k}</span>
              <input type="range" min="0" max="100" value="${Math.round(split[k] * 100)}" data-k="${k}">
              <b id="sv-${k}">${Math.round(split[k] * 100)}%</b></label>`).join('')}
        </div>
        <label class="split-seed">seed <input type="number" id="splitSeed" value="${split.seed}"></label>
      </div>
      <div class="ds-group-h">Curator exports</div>
      ${curated.map(card).join('') || '<div class="ds-empty">no exports yet</div>'}
      <div class="ds-selcount">${nSel} selected for training</div>`;

    this.el.querySelector('#importBtn').onclick = () => this._openImport();
    const box = this.el.querySelector('#importBox');
    this.el.querySelector('#importGo').onclick = () =>
      this.importPath(this.el.querySelector('#importPath').value);
    this.el.querySelector('#importCancel').onclick = () => box.classList.add('hidden');
    this.el.querySelector('#importPath').onkeydown = (e) => {
      if (e.key === 'Enter') this.importPath(e.target.value);
    };
    this.el.querySelectorAll('.ds-card').forEach(c => c.onclick = () => {
      const id = c.dataset.id;
      s.selected.has(id) ? s.selected.delete(id) : s.selected.add(id);
      this.state.selectedDatasets = this.selectedRecords();
      this.render(); this.onChange();
    });
    // "⌄ details" toggles the rich detail panel WITHOUT selecting the dataset
    this.el.querySelectorAll('.ds-view').forEach(b => b.onclick = (e) => {
      e.stopPropagation();
      const id = b.dataset.vid;
      this._expanded.has(id) ? this._expanded.delete(id) : this._expanded.add(id);
      this.render();
    });
    this.el.querySelectorAll('input[type=range]').forEach(r => r.oninput = () => {
      // set this bucket, then distribute the remainder across the OTHER two
      // (proportional to their current sizes) so the three always sum to 100%.
      const k = r.dataset.k, v = Math.max(0, Math.min(1, +r.value / 100));
      const others = ['train', 'eval', 'test'].filter(x => x !== k);
      const rem = 1 - v;
      const cur = others.reduce((s, x) => s + split[x], 0);
      split[k] = v;
      if (cur > 1e-6) others.forEach(x => split[x] = rem * split[x] / cur);
      else others.forEach(x => split[x] = rem / 2);
      // guarantee at least a sliver of train
      if (split.train < 0.01 && k !== 'train') { split.train = 0.01; }
      ['train', 'eval', 'test'].forEach(x => {
        const el = this.el.querySelector(`input[data-k="${x}"]`);
        const lab = this.el.querySelector(`#sv-${x}`);
        const pct = Math.round(split[x] * 100);
        if (el && el !== document.activeElement) el.value = pct;
        if (lab) lab.textContent = pct + '%';
      });
      this.onChange();
    });
    const seed = this.el.querySelector('#splitSeed');
    if (seed) seed.onchange = () => { split.seed = +seed.value; this.onChange(); };
  }

  selectedRecords() {
    return this.datasets.filter(d => this.state.selected.has(d.dataset_id));
  }
}
