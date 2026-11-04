// Deployment CONFIG panel — permanent on the right. Deploy target (Sim / Real),
// a checkpoint picker that pulls back a logged run's architecture + hyperparameters,
// editable deployment hyperparameters, and the deployment log. The 3D robot model
// lives in the center (beside the architecture) and is shared via ctx.rig.
import { actuatorsToJoints } from './rig3d.js';
import * as bus from './bus.js';

export class DeployPanel {
  constructor(el, ctx) {
    this.el = el; this.api = ctx.api; this.state = ctx.state; this.toast = ctx.toast;
    this.rig = ctx.rig || { pose() {}, resize() {}, destroy() {} };
    this.robotSub = ctx.robotSub || null;
    this.confirm = ctx.confirm || (async () => true);
    this.target = 'sim'; this.run = null; this.runMeta = null; this.checkpoint = null;
    this.dep = { control_steps: 200, num_inference_steps: '', device: 'auto' };
    this._render();
    this._loadRuns();
  }

  _render() {
    this.el.innerHTML = `
      <div class="dep-sec">
        <div class="dep-h">Deploy target</div>
        <div class="dep-target" id="depTarget">
          <button data-t="sim" class="on">Simulation · MuJoCo</button>
          <button data-t="real">Real robot</button>
        </div>
        <div class="dep-h">Checkpoint</div>
        <select class="dep-run-sel" id="depRun"><option value="">— select a run —</option></select>
        <select class="dep-ckpt-sel" id="depCkpt"></select>
        <div class="dep-summary" id="depSummary">no run selected</div>
      </div>
      <div class="dep-sec" id="depHyper"></div>
      <div class="dep-sec">
        <button class="dep-btn" id="depGo" disabled>Deploy policy</button>
      </div>
      <div class="dep-sec">
        <div class="dep-h">Eval · test set <span id="evalStat" style="color:var(--accent-soft)"></span></div>
        <button class="mini prim" id="evalGo" style="width:100%;margin-bottom:8px">Run eval on test episodes</button>
        <select class="dep-ckpt-sel" id="evalJoint" style="display:none;margin-bottom:6px"></select>
        <canvas id="evalCv" style="width:100%;height:110px;display:block;background:var(--bg-2);border-radius:8px"></canvas>
      </div>
      <div class="dep-sec" style="flex:1;min-height:0;display:flex;flex-direction:column">
        <div class="dep-h">Deployment log</div>
        <div class="dep-log" id="depLog"><span style="color:var(--ash-dim)">idle</span></div>
      </div>`;

    this._renderHyper();

    this.el.querySelector('#depTarget').onclick = (e) => {
      const b = e.target.closest('button'); if (!b) return;
      this.target = b.dataset.t;
      this.el.querySelectorAll('#depTarget button').forEach(x => x.classList.toggle('on', x === b));
    };
    this.el.querySelector('#depRun').onchange = (e) => this._selectRun(e.target.value);
    this.el.querySelector('#depCkpt').onchange = (e) => { this.checkpoint = e.target.value; };
    this.el.querySelector('#depGo').onclick = () => this._deploy();
    this.el.querySelector('#evalGo').onclick = () => this._eval();
    this.el.querySelector('#evalJoint').onchange = () => this._drawEval();
  }

  _setBar(status, pct, stats) {
    const s = document.getElementById('depBarStatus'); if (s) { s.textContent = status; s.className = 'tb-status ' + status.replace(/\s.*/, ''); }
    const pb = document.getElementById('depPauseBtn'); if (pb) pb.disabled = status !== 'running';
    if (pct != null) { const f = document.getElementById('depBarFill'); const p = document.getElementById('depBarPct'); if (f) f.style.width = (pct * 100).toFixed(0) + '%'; if (p) p.textContent = (pct * 100).toFixed(0) + '%'; }
    const st = document.getElementById('depBarStats'); if (st && stats) st.innerHTML = stats.map(([k, v]) => `<span class="tb-chip"><i>${k}</i>${v}</span>`).join('');
  }

  async _eval() {
    if (!this.run) return this.toast('select a run/checkpoint', 'err');
    this.el.querySelector('#evalStat').textContent = 'running…';
    try {
      const { job } = await this.api.evalRun(this.run, { checkpoint: this.checkpoint, max_frames: 200 });
      this.evalData = await this.api.pollJob(job);
      const sel = this.el.querySelector('#evalJoint');
      sel.style.display = 'block';
      sel.innerHTML = Array.from({ length: this.evalData.action_dim }, (_, i) => `<option value="${i}">joint ${i}</option>`).join('');
      this.el.querySelector('#evalStat').textContent = `L1 ${this.evalData.l1_mean.toFixed(3)} · MSE ${this.evalData.mse_mean.toFixed(3)}`;
      this._drawEval();
    } catch (e) { this.el.querySelector('#evalStat').textContent = 'error'; this._log('eval error: ' + e.message, 'err'); }
  }
  _drawEval() {
    const d = this.evalData; if (!d) return;
    const cv = this.el.querySelector('#evalCv'), ji = +this.el.querySelector('#evalJoint').value || 0;
    const r = Math.min(devicePixelRatio, 2), W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * r; cv.height = H * r; const ctx = cv.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0); ctx.clearRect(0, 0, W, H);
    const gt = d.gt.map(x => x[ji]), pd = d.pred.map(x => x[ji]), all = gt.concat(pd);
    let mn = Math.min(...all), mx = Math.max(...all); if (mx === mn) { mx += 1; mn -= 1; }
    const px = i => 4 + i / (gt.length - 1) * (W - 8), py = v => H - 6 - (v - mn) / (mx - mn) * (H - 12);
    const line = (a, c) => { ctx.strokeStyle = c; ctx.lineWidth = 1.4; ctx.beginPath(); a.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v))); ctx.stroke(); };
    line(gt, 'rgba(151,144,138,.9)'); line(pd, '#C25B2A');
    ctx.font = '8px "Geist Mono"'; ctx.fillStyle = 'rgba(151,144,138,.9)'; ctx.textAlign = 'left'; ctx.fillText('GT', 6, 10); ctx.fillStyle = '#C25B2A'; ctx.fillText('pred', 26, 10);
  }

  _renderHyper() {
    const h = this.el.querySelector('#depHyper');
    const f = (name, label, val, hint = '') => `
      <label class="fld" title="${hint}"><span>${label}</span>
        <input type="text" data-dep="${name}" value="${val}"></label>`;
    h.innerHTML = `<div class="dep-h">Deployment hyperparameters</div>
      ${f('control_steps', 'Control steps', this.dep.control_steps, 'closed-loop rollout length')}
      ${f('num_inference_steps', 'Inference steps', this.dep.num_inference_steps, 'override denoise/flow steps (blank = as trained)')}
      ${f('device', 'Device', this.dep.device, 'auto / cuda / mps / cpu')}`;
    h.querySelectorAll('[data-dep]').forEach(i => i.onchange = () => { this.dep[i.dataset.dep] = i.value; });
  }

  async _loadRuns() {
    try {
      const { runs } = await this.api.runs();
      const done = runs.filter(r => ['done', 'paused', 'stopped', 'running'].includes(r.status));
      const sel = this.el.querySelector('#depRun');
      sel.innerHTML = '<option value="">— select a run —</option>' +
        done.map(r => `<option value="${r.run_id}">${r.name || r.arch} · ${r.status}</option>`).join('');
      // auto-pick the active/most-recent run
      if (this.state.run && done.find(r => r.run_id === this.state.run)) this._selectRun(this.state.run);
    } catch {}
  }
  refreshRuns() { this._loadRuns(); }

  async _selectRun(runId) {
    if (!runId) { this.run = null; this.el.querySelector('#depGo').disabled = true; return; }
    this.run = runId;
    const run = await this.api.run(runId);
    this.runMeta = run;
    const meta = JSON.parse(run.meta_json || '{}');
    const cks = run.checkpoints || [];
    const cs = this.el.querySelector('#depCkpt');
    cs.innerHTML = cks.length ? cks.map(c => `<option value="${c.path}">step ${c.step}</option>`).join('')
      : '<option value="">no checkpoint yet</option>';
    this.checkpoint = cks.length ? cks[cks.length - 1].path : null;
    if (cks.length) cs.value = this.checkpoint;
    const hp = meta.hp || {};
    this.el.querySelector('#depSummary').innerHTML =
      `<b>${run.arch}</b> · ${run.scope || ''}<br>dataset ${run.dataset_id || '—'}<br>` +
      `chunk ${hp.chunk_size || hp.horizon || '—'} · lr ${hp.optimizer_lr || '—'}`;
    this.el.querySelector('#depRun').value = runId;
    this.el.querySelector('#depGo').disabled = !cks.length;
  }

  _log(msg, cls = '') {
    const el = this.el.querySelector('#depLog');
    if (el.firstChild && el.firstChild.style) el.innerHTML = '';
    el.insertAdjacentHTML('beforeend', `<div class="${cls}">${msg}</div>`);
    el.scrollTop = el.scrollHeight;
    (window.__depLog || (window.__depLog = [])).push({ line: msg, cls });
    bus.emit('depLogRows', [{ line: msg, cls }]);
  }

  // public: called from the top-bar Deploy / Redeploy / Pause group
  deploy() { this._deploy(); }
  stopDeploy() {
    if (!this._deploying) { this.toast('no active deployment', 'warn'); return; }
    this._depCancel = true; this._setBar('stopped', null, [['target', 'MuJoCo']]);
    this._log('deployment stop requested', 'err');
  }

  async _deploy() {
    if (!this.run) return this.toast('select a run/checkpoint (right panel)', 'err');
    if (this.target === 'real') {
      this._log('Real-robot deploy targets the robot host (thor). Requires the on-robot bridge; not run from this Mac.', 'err');
      this.toast('real-robot deploy: connect the robot host', 'err');
      return;
    }
    const body = { checkpoint: this.checkpoint, control_steps: +this.dep.control_steps || 120,
      device: this.dep.device };
    if (this.dep.num_inference_steps) body.num_inference_steps = +this.dep.num_inference_steps;
    // confirmation summary before rolling out — control/inference steps editable
    const res = await this.confirm({
      title: 'Deploy policy?', sub: (this.runMeta && this.runMeta.arch) || '', confirmLabel: '◆ Deploy in sim', kind: 'go',
      rows: [
        ['Run', this.run],
        ['Checkpoint', this.checkpoint ? this.checkpoint.split('/').pop() : 'latest'],
        ['Target', 'MuJoCo (sim)'],
        ['Device', String(this.dep.device || 'auto').toUpperCase()],
      ],
      inputs: [
        { key: 'control_steps', label: 'Control steps', value: body.control_steps },
        { key: 'num_inference_steps', label: 'Inference steps (diffusion/flow)', type: 'text', value: this.dep.num_inference_steps || '' },
      ],
    });
    if (!res) return;
    const v = res.values || {};
    body.control_steps = Math.max(1, parseInt(v.control_steps) || body.control_steps);
    if (v.num_inference_steps) body.num_inference_steps = +v.num_inference_steps;
    else delete body.num_inference_steps;
    this.el.querySelector('#depGo').disabled = true;
    this._deploying = true; this._depCancel = false;
    if (this.robotSub) this.robotSub.textContent = 'deploying…';
    this._setBar('running', 0, [['target', 'MuJoCo'], ['steps', body.control_steps]]);
    this._log(`Deploying ${this.runMeta.arch} in MuJoCo (${body.control_steps} steps)…`, 'ok');
    try {
      const { job } = await this.api.deployRun(this.run, body);
      const res = await this.api.pollJob(job, (j) => {
        if (this._depCancel) return;
        if (j.total) { this._log(`rollout ${j.done}/${j.total}`); this._setBar('running', j.done / j.total, [['target', 'MuJoCo'], ['step', `${j.done}/${j.total}`]]); }
      });
      if (this._depCancel) { this._log('deployment stopped', 'err'); }
      else {
        this._log(`done · ${res.frames} control steps · ${res.action_dim}-D action`, 'ok');
        this._setBar('done', 1, [['target', 'MuJoCo'], ['steps', res.frames]]);
        if (this.robotSub) this.robotSub.textContent = 'closed-loop rollout';
        this._play(res.trajectory);
      }
    } catch (e) { this._log('error: ' + e.message, 'err'); this._setBar('failed', null, [['error', '1']]); }
    finally { this.el.querySelector('#depGo').disabled = false; this._deploying = false; }
  }

  _play(traj) {
    if (!traj || !traj.length) return;
    cancelAnimationFrame(this._anim); let i = 0;
    const step = () => {
      const f = traj[i % traj.length];
      this.rig.pose(actuatorsToJoints(f.actuator_names, f.actuator_positions));
      i++; this._anim = requestAnimationFrame(() => setTimeout(step, 40));
    };
    step();
  }
}
