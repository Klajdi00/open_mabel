// RIGHT dock panes: Config (arch + hyperparameters), Inputs (modular camera +
// module-scope selection), Info (network I/O contract + validation). All feed the
// same state; any change re-runs the reactive diagram + validators.

export class DockPanes {
  constructor({ configEl, inputsEl, outputEl, infoEl, schema, embodiment, state, onChange }) {
    this.configEl = configEl; this.inputsEl = inputsEl; this.outputEl = outputEl; this.infoEl = infoEl;
    this.schema = schema; this.emb = embodiment; this.state = state;
    this.onChange = onChange;
    if (!state.hpByArch) state.hpByArch = {};
  }

  // OUTPUT tab — what the policy PREDICTS: an action scope (independent from the
  // input scope) + the resulting output tensor, so you can e.g. observe the whole
  // body but only output the arm controls.
  // effective hyperparameter value: user override, else the arch's schema default
  _hp(key) {
    const hp = this.state.hpByArch[this.state.arch] || {};
    if (hp[key] !== undefined) return hp[key];
    const A = this.schema && this.schema.arches[this.state.arch];
    const f = A && A.fields.find(x => x.name === key);
    return f ? f.default : undefined;
  }

  renderOutput() {
    if (!this.outputEl) return;
    const s = this.state, e = this.emb, r = s.resolved || {};
    const sel = s.selection || (s.selection = { scope: 'whole_body', groups: [], cameras: [] });
    const opts = `<option value="__same" ${!sel.action_scope ? 'selected' : ''}>Same as input scope</option>`
      + Object.entries(e.scopes).map(([k, v]) =>
        `<option value="${k}" ${k === sel.action_scope ? 'selected' : ''}>${v.label}</option>`).join('');
    const outNames = (r.action_names || r.joint_names || []);
    const chips = outNames.slice(0, 24).map(nm => `<span class="out-jt">${nm}</span>`).join('')
      + (outNames.length > 24 ? `<span class="out-jt more">+${outNames.length - 24}</span>` : '');
    this.outputEl.innerHTML = `
      <div class="pane-sec"><div class="pane-h">Output modality</div>
        <div class="out-mod"><span class="out-badge on">● Action chunk</span>
          <div class="out-note">absolute joint-position targets (${this.state.arch})</div></div></div>
      <div class="pane-sec"><div class="pane-h">Output scope <i class="pane-hint">controls the policy predicts</i></div>
        <select id="actScopeSel" class="scope-sel">${opts}</select>
        <div class="out-io">
          <div class="out-row"><span>action dim</span><b>${r.action_dim ?? '?'}-D</b></div>
          <div class="out-row"><span>predicts</span><b>${r.action_scope || sel.scope || '—'}</b></div>
          <div class="out-row"><span>chunk × dim</span><b>${(this._hp('chunk_size') ?? this._hp('horizon') ?? '—')} × ${r.action_dim ?? '?'}</b></div>
        </div></div>
      <div class="pane-sec"><div class="pane-h">Predicted joints <b>${outNames.length}</b></div>
        <div class="out-jts">${chips || '<span class="pane-hint">—</span>'}</div></div>`;
    this.outputEl.querySelector('#actScopeSel').onchange = (ev) => {
      sel.action_scope = ev.target.value === '__same' ? null : ev.target.value;
      this.onChange();
    };
  }

  _field(f, val, onInput) {
    const id = 'f_' + f.name;
    let ctl;
    if (f.type === 'bool') {
      ctl = `<input type="checkbox" id="${id}" ${val ? 'checked' : ''}>`;
    } else if (f.type === 'select') {
      ctl = `<select id="${id}">${f.options.map(o =>
        `<option ${o == val ? 'selected' : ''}>${o}</option>`).join('')}</select>`;
    } else if (f.type === 'text') {
      ctl = `<input type="text" id="${id}" value="${val ?? ''}" placeholder="optional">`;
    } else {
      const step = f.type === 'float' ? (f.step || 'any') : (f.step || 1);
      ctl = `<input type="number" id="${id}" value="${val}" step="${step}"
        ${f.min != null ? `min="${f.min}"` : ''} ${f.max != null ? `max="${f.max}"` : ''}>`;
    }
    return `<label class="fld" title="${f.help || ''}"><span>${f.label}</span>${ctl}</label>`;
  }

  renderConfig() {
    const s = this.state, sc = this.schema;
    const hp = s.hpByArch[s.arch] || (s.hpByArch[s.arch] = {});
    const A = sc.arches[s.arch];
    const val = (f) => (hp[f.name] !== undefined ? hp[f.name] : f.default);
    this.configEl.innerHTML = `
      <div class="pane-sec">
        <div class="pane-h">Architecture</div>
        <div class="arch-tabs">${sc.order.map(a =>
          `<button data-arch="${a}" class="arch-tab ${a === s.arch ? 'on' : ''}">${a}</button>`).join('')}</div>
        <div class="arch-desc">${A.label}<br><i>${A.paper || ''}</i>${A.native ? '' :
          ' · <span class="warn-tag">custom plugin</span>'}</div>
      </div>
      <div class="pane-sec"><div class="pane-h">Hyperparameters</div>
        ${A.fields.map(f => this._field(f, val(f))).join('')}</div>
      <div class="pane-sec"><div class="pane-h">Training</div>
        ${sc.train_fields.map(f => this._field(f, val(f))).join('')}</div>`;

    this.configEl.querySelectorAll('.arch-tab').forEach(b => b.onclick = () => {
      s.arch = b.dataset.arch; this.renderConfig(); this.onChange();
    });
    const bind = (f) => {
      const e = this.configEl.querySelector('#f_' + f.name); if (!e) return;
      e.oninput = e.onchange = () => {
        hp[f.name] = f.type === 'bool' ? e.checked
          : (f.type === 'select' || f.type === 'text') ? e.value : +e.value;
        this.onChange();
      };
    };
    A.fields.forEach(bind); sc.train_fields.forEach(bind);
  }

  renderInputs() {
    const s = this.state, e = this.emb;
    const sel = s.selection || (s.selection = { scope: 'whole_body', groups: [], cameras: e.cameras.map(c => c.name) });
    const camList = e.cameras.map(c => {
      const on = sel.cameras.includes(c.name);
      return `<label class="cam-pick ${on ? 'on' : ''}"><input type="checkbox" data-cam="${c.name}" ${on ? 'checked' : ''}>
        <span>${c.name}</span><i>${c.w}×${c.h}</i></label>`;
    }).join('');
    const scopes = Object.entries(e.scopes).map(([k, v]) =>
      `<option value="${k}" ${k === sel.scope ? 'selected' : ''}>${v.label}</option>`).join('')
      + `<option value="custom" ${sel.scope === 'custom' ? 'selected' : ''}>Custom groups…</option>`;
    const groupChecks = e.group_order.map(g => {
      const inScope = (e.scopes[sel.scope]?.groups || sel.groups || []).includes(g);
      return `<label class="grp-pick ${inScope ? 'on' : ''}"><input type="checkbox" data-grp="${g}"
        ${inScope ? 'checked' : ''} ${sel.scope !== 'custom' ? 'disabled' : ''}>
        <span>${g}</span><i>${e.groups[g].n}</i></label>`;
    }).join('');
    this.inputsEl.innerHTML = `
      <div class="pane-sec"><div class="pane-h">Cameras <b id="camCount">${sel.cameras.length}/${e.cameras.length}</b></div>
        <div class="cam-grid">${camList}</div></div>
      <div class="pane-sec"><div class="pane-h">Input scope <i class="pane-hint">obs proprio</i></div>
        <select id="scopeSel" class="scope-sel">${scopes}</select>
        <div class="grp-grid">${groupChecks}</div></div>
      <div class="pane-sec dims-box" id="dimsBox"></div>`;
    this._renderDims();

    this.inputsEl.querySelectorAll('input[data-cam]').forEach(c => c.onchange = () => {
      const name = c.dataset.cam;
      if (c.checked) { if (!sel.cameras.includes(name)) sel.cameras.push(name); }
      else sel.cameras = sel.cameras.filter(x => x !== name);
      this.renderInputs(); this.onChange();
    });
    this.inputsEl.querySelector('#scopeSel').onchange = (ev) => {
      sel.scope = ev.target.value;
      if (sel.scope === 'custom' && (!sel.groups || !sel.groups.length))
        sel.groups = [...(e.scopes.whole_body.groups)];
      this.renderInputs(); this.onChange();
    };
    this.inputsEl.querySelectorAll('input[data-grp]').forEach(c => c.onchange = () => {
      sel.groups = sel.groups || [];
      const g = c.dataset.grp;
      if (c.checked) { if (!sel.groups.includes(g)) sel.groups.push(g); }
      else sel.groups = sel.groups.filter(x => x !== g);
      this.renderInputs(); this.onChange();
    });
  }

  // called by app after /api/selection resolves, to show live dims
  setResolved(resolved) { this.state.resolved = resolved; this._renderDims(); this.renderInfo(); this.renderOutput(); }

  _renderDims() {
    const box = this.inputsEl.querySelector('#dimsBox'); if (!box) return;
    const r = this.state.resolved;
    if (!r) { box.innerHTML = '<div class="pane-h">Resulting dims</div><i class="dim-muted">…</i>'; return; }
    box.innerHTML = `<div class="pane-h">Resulting network I/O</div>
      <div class="dim-row"><span>input cameras</span><b>${r.cameras.length}</b></div>
      <div class="dim-row"><span>proprio (obs) dim</span><b>${r.obs_state_dim}</b></div>
      <div class="dim-row"><span>action (output) dim</span><b>${r.action_dim}</b></div>
      <div class="dim-row"><span>joints</span><b class="jnames">${r.joint_names.slice(0, 6).join(', ')}${r.joint_names.length > 6 ? '…' : ''}</b></div>`;
  }

  _freqSection() {
    const s = this.state, e = this.emb;
    const dsets = s.selectedDatasets || [];
    if (!dsets.length) return '<div class="pane-sec"><div class="pane-h">Frequency</div><i class="dim-muted">select a dataset</i></div>';
    // average rate across selected datasets (curation resamples every modality to one rate)
    const avg = dsets.reduce((a, d) => a + (d.fps || 0), 0) / dsets.length;
    const hz = (avg || 0).toFixed(2);
    const camsAvail = new Set(dsets.flatMap(d => d.cameras || []));
    const selCams = new Set((s.resolved && s.resolved.cameras) || []);
    const camRow = (label, members) => {
      const present = members.filter(m => camsAvail.has(m));
      if (!present.length) return '';
      const used = present.some(m => selCams.has(m));
      return `<div class="freq-row ${used ? 'used' : ''}"><span>${label}</span>
        <i>${present.join(', ')}</i><b>${hz} Hz</b></div>`;
    };
    const selGroups = new Set((s.resolved && s.resolved.groups) || []);
    const propRow = (label, groups, dof) => {
      const used = groups.some(g => selGroups.has(g));
      return `<div class="freq-row ${used ? 'used' : ''}"><span>${label}</span>
        <i>${dof} DOF</i><b>${hz} Hz</b></div>`;
    };
    const g = e.groups;
    return `<div class="pane-sec"><div class="pane-h">Frequency <b>avg ${hz} Hz</b></div>
      <div class="freq-cat">Video</div>
      ${camRow('head cam', e.camera_groups.head_cam)}
      ${camRow('wrist cam', e.camera_groups.wrist_cam)}
      ${camRow('base cam', e.camera_groups.base_cam)}
      <div class="freq-cat">Proprioception</div>
      ${propRow('arms', ['left_arm', 'right_arm'], g.left_arm.n + g.right_arm.n)}
      ${propRow('left hand', ['left_hand'], g.left_hand.n)}
      ${propRow('right hand', ['right_hand'], g.right_hand.n)}
      ${propRow('neck', ['neck'], g.neck.n)}
      ${propRow('base', ['base'], g.base.n)}
      ${propRow('lift + torso', ['lift_torso'], g.lift_torso.n)}
      <div class="freq-cat">Lidar</div>
      <div class="freq-row"><span>lidar</span><i>not in curated export</i><b>—</b></div>
      <div class="freq-note">Curation resamples every sensor/module to the unified dataset rate; used-for-this-run rows are highlighted.</div>
    </div>`;
  }

  renderInfo() {
    if (!this.infoEl) return;   // Info tab removed — dataset "⌄ details" covers this now
    const s = this.state, r = s.resolved, issues = s.issues || [];
    const io = r ? `
      <div class="pane-sec"><div class="pane-h">Network I/O contract</div>
        <div class="io-block"><b>Inputs</b>
          ${r.cameras.map(c => `<div class="io-row">observation.images.${c}<i>3×270×480</i></div>`).join('')}
          <div class="io-row">observation.state<i>${r.obs_state_dim}</i></div></div>
        <div class="io-block"><b>Output</b>
          <div class="io-row">action<i>${r.action_dim} · joint pos (abs)</i></div></div>
      </div>` : '';
    const issuesHtml = issues.length ? `<div class="pane-sec"><div class="pane-h">Validation</div>
      ${issues.map(i => `<div class="issue ${i.level}">${i.level === 'error' ? '✕' : '⚠'} ${i.msg}</div>`).join('')}</div>`
      : `<div class="pane-sec"><div class="issue ok">✓ config valid</div></div>`;
    this.infoEl.innerHTML = io + this._freqSection() + issuesHtml;
  }
}
