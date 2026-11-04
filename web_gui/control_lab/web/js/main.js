/* Control Lab frontend — the app-mirror control console (Torque · Stiff/Soft ·
   Region · Mode), live charts, 3D overlays, VP cluster. The command surface
   mirrors the iOS / Vision Pro wire protocol (control_mode / joint_command /
   external_force / estop), so driving here is identical to driving there. */
import { LabViewer } from './viewer.js';
import { StripChart } from './charts.js';
import { Sticks } from './joysticks.js';

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];

const viewer = new LabViewer($('#gl'));
window._lab = { viewer };            // test/debug handle (playwright probes)
let ws = null, state = null, charts = {}, modeBuilt = null;
let lanIp = 'this-mac', jointSchema = null;
let lastState = { t: 0, wall: 0 }, rtFactor = null;
fetch('/api/config').then(r => r.json()).then(c => { lanIp = c.lanIp || lanIp; });

/* the four control axes we send (== the app's control_mode message) */
const spec = { method: 'navigation', controlType: 'impedance',
               region: 'arm', stiffness: 20 };

/* ── commands ─────────────────────────────────────────────────────────── */
async function cmd(body) {
  const r = await fetch('/api/command', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return r.json();
}
const sendSpec = () => cmd({ action: 'control_mode', ...spec });
const wsSend = m => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); };

/* ── websocket ────────────────────────────────────────────────────────── */
function connect() {
  ws = new WebSocket(`ws${location.protocol === 'https:' ? 's' : ''}://${location.host}/ws/state`);
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'state') onState(msg.payload);
    else if (msg.type === 'plots_history') seedHistory(msg.payload);
    else if (msg.type === 'schema') jointSchema = msg.payload;
  };
  ws.onopen = () => $('#connDot').className = 'dot ok';
  ws.onclose = () => { $('#connDot').className = 'dot'; setTimeout(connect, 1200); };
}
connect();

const sticks = new Sticks($('#stickL'), $('#stickR'), v => {
  // EXACT cockpit wire convention: lx/rx pre-negated on the wire; app_joystick()
  // on the server negates lx/ly again into the controller frame.
  wsSend({ type: 'input', nav: { lx: -v.lx, ly: v.ly, rx: -v.rx, ry: v.ry } });
});

let _lastWristSend = 0;
viewer.onWristDrag = (side, pos) => {
  const now = performance.now();
  if (now - _lastWristSend < 33) return;
  _lastWristSend = now;
  wsSend({ type: 'wrist', side, pos });
};

/* ── per-mode chart definitions (re-homed research plots) ─────────────── */
const MODE_CHARTS = {
  navigation: [
    { key: 'vx', title: 'forward vx [m/s]', series: [['cmd_vx', 'cmd'], ['safe_vx', 'safe']] },
    { key: 'vy', title: 'strafe vy [m/s]', series: [['cmd_vy', 'cmd'], ['safe_vy', 'safe']] },
    { key: 'w', title: 'yaw ω [rad/s]', series: [['cmd_w', 'cmd'], ['safe_w', 'safe']] },
    { key: 'usage', title: 'ZMP tip usage (red = limit)', yMin: 0,
      refLines: [{ y: 1, color: '#d23b2a' }], series: [['tip_usage', 'u']] },
    { key: 'caps', title: 'lift-aware caps [m/s] + lift [m]',
      series: [['cap_fwd', 'cap fwd'], ['cap_lat', 'cap lat'], ['lift_z', 'lift z']] },
    { key: 'odom', title: 'command-odometry error [m]', yMin: 0, series: [['odom_err', 'err']] },
  ],
  joint: [
    { key: 'q', title: 'test joint: q vs q_des [rad]',
      series: [['q_des', 'q_des', '#3fb56b'], ['q', 'q', '#f0b68b']] },
    { key: 'err', title: 'region tracking error [rad]',
      series: [['q_err', 'joint'], ['err_max', 'max'], ['err_rms', 'rms']] },
    { key: 'tau', title: 'torque τ [Nm] — total vs gravity FF',
      series: [['tau', 'τ total', '#f0b68b'], ['tau_ff', 'gravity ff', '#3fb56b']] },
    { key: 'tausc', title: 'self-collision barrier torque [Nm] (soft-wall response)',
      yMin: 0, series: [['tau_sc', 'τ_sc']] },
    { key: 'F', title: 'wrench |F| [N] — true vs estimated (push)',
      series: [['F_true', 'true', '#3fb56b'], ['F_est', 'est', '#d23b2a']] },
    { key: 'cos', title: 'force direction cosine', yMin: -1, yMax: 1,
      refLines: [{ y: 0.8, color: '#3fb56b' }], series: [['F_cos', 'cos']] },
  ],
  wrist: [
    { key: 'ee', title: 'EE tracking error [mm]',
      series: [['ee_err_l_mm', 'left'], ['ee_err_r_mm', 'right']] },
    { key: 'rot', title: 'EE orientation error [deg]',
      series: [['ee_rot_l_deg', 'left'], ['ee_rot_r_deg', 'right']] },
    { key: 'solve', title: 'differential-IK solve [ms]', yMin: 0, series: [['ik_solve_ms', 'solve']] },
    { key: 'F', title: 'wrench |F| [N] — true vs estimated (push)',
      series: [['F_true', 'true', '#3fb56b'], ['F_est', 'est', '#d23b2a']] },
  ],
  retarget: [
    { key: 'ee', title: 'EE tracking error [mm]',
      series: [['ee_err_l_mm', 'left'], ['ee_err_r_mm', 'right']] },
    { key: 'mpjpe', title: 'MPJPE — mean keypoint error [mm]', yMin: 0, series: [['mpjpe_mm', 'mpjpe']] },
    { key: 'base', title: 'base recruitment + tip usage',
      refLines: [{ y: 1, color: '#d23b2a' }],
      series: [['base_v', '|v| m/s'], ['base_w', 'ω rad/s'], ['tip_usage', 'u']] },
    { key: 'jerk', title: 'EE smoothness — RMS jerk [m/s³]', yMin: 0, series: [['jerk', 'jerk']] },
    { key: 'lat', title: 'latency [ms]',
      series: [['retarget_ms', 'control'], ['age_ms', 'frame age']] },
  ],
};

/* joint-slider groups (Joint mode ctl panel). Wrists are WBC-managed but
   untracked (gravity-comp float), so they're not exposed as targets. */
const JOINT_GROUPS = [
  { h: 'Left Arm', ids: ['left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6', 'left_arm_7'] },
  { h: 'Right Arm', ids: ['right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6', 'right_arm_7'] },
  { h: 'Torso · Lift', ids: ['torso', 'lift'] },
  { h: 'Neck', ids: ['neck_yaw', 'neck_pitch', 'neck_roll'] },
  { h: 'Grip', ids: ['left_grip', 'right_grip'] },
];

/* ── dock builders (per mode) ─────────────────────────────────────────── */
function buildDock(method) {
  const plots = $('#plotsDock'), dock = $('#ctlDock');
  plots.innerHTML = ''; dock.innerHTML = ''; charts = {}; modeBuilt = method;

  if (method === 'joint') { dock.appendChild(jointCard()); dock.appendChild(diagnosticsCard()); }
  else if (method === 'wrist') { dock.appendChild(wristCard()); dock.appendChild(diagnosticsCard()); }
  else if (method === 'navigation') dock.appendChild(driveCard());
  else if (method === 'retarget') { dock.appendChild(teleopCard()); plots.appendChild(metricsCard()); }

  for (const def of (MODE_CHARTS[method] || [])) {
    const c = card('');
    const div = document.createElement('div'); div.className = 'plot';
    const cv = document.createElement('canvas');
    div.appendChild(cv); c.appendChild(div); plots.appendChild(c);
    const ch = new StripChart(cv, { title: def.title, yMin: def.yMin, yMax: def.yMax,
      refLines: def.refLines, windowS: 12 });
    for (const [k, label, color] of def.series) ch.def(k, label, color);
    charts[def.key] = ch;
  }
  if (method === 'navigation') viewer.clearPaths();
  fetch('/api/plots').then(r => r.json()).then(seedHistory);
}

function card(title) {
  const el = document.createElement('div'); el.className = 'card';
  if (title) el.innerHTML = `<h3>${title}</h3>`;
  return el;
}

function jointCard() {
  const c = card('Joint control — sliders → joint_command');
  c.insertAdjacentHTML('beforeend',
    `<div class="row" style="color:var(--ash);font-size:10.5px;line-height:1.5">
       Stiff springs each arm/torso joint to the slider (impedance K); Soft floats
       (the sliders idle). Lift/neck/grips drive directly.</div>
     <div class="jsliders" id="jsliders"></div>`);
  const wrap = c.querySelector('#jsliders');
  const R = (jointSchema && jointSchema.ranges) || {};
  const cur = (state && state.jointPositions) || {};
  for (const g of JOINT_GROUPS) {
    const h = document.createElement('div'); h.className = 'jgroup-h';
    h.textContent = g.h; wrap.appendChild(h);
    for (const id of g.ids) {
      const rng = R[id] || (id.endsWith('_grip') ? [0, 1] : [-3.14, 3.14]);
      const v0 = id === 'lift' ? liftFromState() : (id.endsWith('_grip') ? 0 : (cur[id] ?? 0));
      const row = document.createElement('div'); row.className = 'jslider';
      row.innerHTML =
        `<label title="${id}">${id.replace(/_/g, ' ')}</label>
         <input type="range" min="${rng[0]}" max="${rng[1]}" step="0.01" value="${v0}">
         <span class="jv">${(+v0).toFixed(2)}</span>`;
      const inp = row.querySelector('input'), out = row.querySelector('.jv');
      inp.oninput = () => {
        out.textContent = (+inp.value).toFixed(2);
        sendJoint(id, +inp.value);
      };
      wrap.appendChild(row);
    }
  }
  return c;
}

function sendJoint(id, v) {
  if (id === 'lift') wsSend({ type: 'joint', id: 'lift_lower', value: v / 2 }),
    wsSend({ type: 'joint', id: 'lift_upper', value: v / 2 });
  else wsSend({ type: 'joint', id, value: v });
}
function liftFromState() { return state ? (state.lift || 0) : 0; }

function wristCard() {
  const c = card('Wrist tracking — task-space IK');
  c.insertAdjacentHTML('beforeend',
    `<div class="row" style="color:var(--ash);font-size:10.5px;line-height:1.5">
       Drag the <b style="color:var(--accent)">orange wrist targets</b> in the viewer —
       differential IK tracks them into joint targets the impedance follows. Soft floats.</div>
     <div class="row"><label>posture</label>
       <div class="seg mini" id="postureSeg" title="Human: the VP teleop's anatomical null-space constraints; Generic: uniform posture">
         <button data-h="1" class="on">Human</button><button data-h="0">Generic</button></div></div>
     <div class="row"><label>col law</label>
       <div class="seg mini" id="scSeg" title="CBF slides around the obstacle; Stop brakes at it">
         <button data-m="cbf" class="on">CBF · slide</button><button data-m="stop">Stop · wall</button></div></div>
     <div class="row"><button class="gbtn" id="wristSync">Sync to current pose</button></div>`);
  c.querySelector('#wristSync').onclick = () => cmd({ action: 'wrist_sync' });
  c.querySelectorAll('#scSeg button').forEach(b =>
    b.onclick = () => { cmd({ action: 'sc_method', method: b.dataset.m }); segSet('#scSeg', b); });
  c.querySelectorAll('#postureSeg button').forEach(b =>
    b.onclick = () => { cmd({ action: 'wrist_human', on: b.dataset.h === '1' }); segSet('#postureSeg', b); });
  return c;
}

function driveCard() {
  const c = card('Drive — base joystick (navigation)');
  c.insertAdjacentHTML('beforeend',
    `<div class="row" style="color:var(--ash);font-size:10.5px;line-height:1.5">
       Left stick translates, right stick yaws + lifts. Twists pass through the
       tip-over motion model (safe = post-limit). Stick-up drives −X (forward).</div>
     <div class="row"><label>self-col</label>
       <div class="layer compact" data-layer="self_collision" id="driveSC"><span class="sw"></span>Collision guard</div></div>`);
  const sc = c.querySelector('#driveSC');
  sc.onclick = () => cmd({ action: 'self_collision', on: !sc.classList.contains('on') });
  return c;
}

function teleopCard() {
  const c = card('Whole-body teleop — operator retarget');
  c.insertAdjacentHTML('beforeend',
    `<div class="row" style="color:var(--ash);font-size:10.5px;line-height:1.5">
       Pick an input device (Device = the Vision Pro / iPhone app on :9090, or Mock).
       The robot mirrors the operator's head + arms + fingers; region scopes it.</div>
     <div class="row"><label>hands</label>
       <div class="seg mini" id="sideSegDock">
         <button data-side="left">Left</button><button data-side="both" class="on">Both</button>
         <button data-side="right">Right</button></div></div>`);
  c.querySelectorAll('#sideSegDock button').forEach(b =>
    b.onclick = () => { cmd({ action: 'hand_side', side: b.dataset.side }); segSet('#sideSegDock', b);
      segSet('#sideSeg', $(`#sideSeg [data-side="${b.dataset.side}"]`));
      viewer.flyTo(b.dataset.side === 'left' ? 'left_arm' : b.dataset.side === 'right' ? 'right_arm' : 'hands'); });
  return c;
}

function diagnosticsCard() {
  const c = card('Diagnostics — inject a push / trajectory');
  c.insertAdjacentHTML('beforeend', `
    <div class="jgroup-h">external wrench (ground truth for F̂ in Soft)</div>
    <div class="row"><label>body</label><select id="pushBody"></select></div>
    <div class="row"><label>dir</label>
      <div class="seg mini" id="pushDir">
        <button data-d="0,1,0" class="on">+Y</button><button data-d="0,-1,0">−Y</button>
        <button data-d="0,0,-1">−Z</button><button data-d="1,0,0">+X</button><button data-d="-1,0,0">−X</button></div></div>
    <div class="row"><label>|F|</label>
      <input type="range" id="pushMag" min="2" max="40" step="1" value="15">
      <span id="pushMagV" style="font-family:var(--mono);font-size:10px">15 N</span></div>
    <div class="row"><button class="gbtn primary" id="pushGo">Push 1.5 s</button>
      <span style="font-family:var(--mono);font-size:9px;color:var(--ash)">green=applied · red=est</span></div>
    <div class="jgroup-h">joint trajectory (Joint mode)</div>
    <div class="row"><label>traj</label>
      <div class="seg mini" id="trajSeg">
        <button data-kind="hold" class="on">Hold</button><button data-kind="step">Step</button>
        <button data-kind="sine">Sine</button></div></div>
    <div class="row"><label>joint</label>
      <select id="trajJoint">${['left_arm_2', 'left_arm_4', 'right_arm_2', 'right_arm_4', 'torso']
        .map(j => `<option>${j}</option>`).join('')}</select></div>
    <div class="row"><label>amp</label>
      <input type="range" id="trajAmp" min="0.05" max="0.8" step="0.05" value="0.3">
      <span id="trajAmpV" style="font-family:var(--mono);font-size:10px">0.30 rad</span></div>
    <div class="row"><label>period</label>
      <input type="range" id="trajPer" min="1" max="10" step="0.5" value="4">
      <span id="trajPerV" style="font-family:var(--mono);font-size:10px">4.0 s</span></div>`);
  fetch('/api/bodies').then(r => r.json()).then(d => {
    c.querySelector('#pushBody').innerHTML = d.bodies.map(b =>
      `<option ${b === 'left_palm' ? 'selected' : ''}>${b}</option>`).join('');
  });
  c.querySelectorAll('#pushDir button').forEach(b => b.onclick = () => segSet('#pushDir', b));
  c.querySelector('#pushMag').oninput = e => c.querySelector('#pushMagV').textContent = e.target.value + ' N';
  c.querySelector('#pushGo').onclick = () => {
    const d = c.querySelector('#pushDir .on').dataset.d.split(',').map(Number);
    const m = +c.querySelector('#pushMag').value;
    cmd({ action: 'push', body: c.querySelector('#pushBody').value, force: d.map(v => v * m), seconds: 1.5 });
  };
  const sendTraj = () => cmd({ action: 'traj',
    kind: c.querySelector('#trajSeg .on').dataset.kind, joint: c.querySelector('#trajJoint').value,
    amp: +c.querySelector('#trajAmp').value, period: +c.querySelector('#trajPer').value });
  c.querySelectorAll('#trajSeg button').forEach(b => b.onclick = () => { segSet('#trajSeg', b); sendTraj(); });
  c.querySelector('#trajJoint').onchange = sendTraj;
  c.querySelector('#trajAmp').oninput = e => c.querySelector('#trajAmpV').textContent = (+e.target.value).toFixed(2) + ' rad';
  c.querySelector('#trajAmp').onchange = sendTraj;
  c.querySelector('#trajPer').oninput = e => c.querySelector('#trajPerV').textContent = (+e.target.value).toFixed(1) + ' s';
  c.querySelector('#trajPer').onchange = sendTraj;
  return c;
}

function metricsCard() {
  const c = card('Teleop pipeline latency (measured)');
  c.insertAdjacentHTML('beforeend', `<div class="metrics" id="latGrid"></div>`);
  const grid = c.querySelector('#latGrid');
  for (const [k, label] of [['rate', 'stream Hz'], ['jitter', 'jitter ms'], ['age', 'frame age ms'],
      ['retarget', 'control ms'], ['physics', 'physics ms'], ['rt', 'RT factor']])
    grid.insertAdjacentHTML('beforeend',
      `<div class="metric"><div class="k">${label}</div><div class="v" id="m_${k}">—</div></div>`);
  return c;
}

/* ── per-state UI update ──────────────────────────────────────────────── */
function seedHistory(hist) {
  for (const ch of Object.values(charts))
    for (const key in ch.series) if (hist[key]) ch.seed(key, hist[key]);
}

function onState(s) {
  state = s;
  viewer.update(s);
  const wall = performance.now() / 1000, dw = wall - lastState.wall;
  if (lastState.wall && s.t > lastState.t && dw > 0.02) {
    const f = Math.min(3, (s.t - lastState.t) / dw);
    rtFactor = rtFactor == null ? f : rtFactor * 0.9 + f * 0.1;
  }
  if (s.t < lastState.t) rtFactor = null;
  lastState = { t: s.t, wall };

  const method = s.method || 'navigation';
  if (method !== modeBuilt) buildDock(method);

  // torque master gate
  const on = !!s.torque;
  const tb = $('#torqueBtn');
  tb.className = 'torque-btn ' + (on ? 'on' : 'off');
  $('#torqueLbl').textContent = on ? 'E-STOP' : 'Enable Torque';
  $('#ctlGroup').classList.toggle('disabled', !on);
  $('#limpCta').classList.toggle('hidden', on);

  // reflect the four axes from server truth
  segSet('#feelSeg', $(`#feelSeg [data-type="${(s.spec || {}).type}"]`));
  segSet('#regionSeg', $(`#regionSeg [data-region="${s.region}"]`));
  segSet('#modeSeg', $(`#modeSeg [data-method="${method}"]`));
  const feelActive = (method === 'joint' || method === 'wrist');
  const isSoft = (s.spec || {}).type === 'compliance';
  $('#feelSeg').style.opacity = 1;                       // always usable (Soft hops into Joint)
  $('#stiffWrap').classList.toggle('dim', isSoft || !feelActive);
  // SOFT hint: the robot is a passive float — manipulate it in the MuJoCo viewer
  $('#softHint').classList.toggle('hidden', !(on && isSoft && feelActive));
  if (document.activeElement !== $('#stiffSlider')) {
    $('#stiffSlider').value = Math.round((s.spec || {}).stiffness ?? spec.stiffness);
    $('#stiffVal').textContent = 'K ' + Math.round((s.spec || {}).stiffness ?? spec.stiffness);
  }

  // input device (whole-body teleop only)
  const isRt = method === 'retarget';
  const vpMode = (s.vpNode || {}).mode || 'off';
  $('#vpSeg').classList.toggle('hidden', !isRt);
  $('#sideSeg').classList.toggle('hidden', !isRt);
  if (isRt) segSet('#vpSeg', $(`#vpSeg [data-vp="${vpMode}"]`));
  renderDeviceChip(s, vpMode);

  // chips
  $('#statusChip').textContent = s.status;
  $('#rtChip').textContent = rtFactor == null ? 'RT —' : `RT ${rtFactor.toFixed(2)}×`;
  // sticks drive the base in Drive / Teleop, AND in the whole_body region of any
  // mode (whole-body = arms + base: drive the base while manipulating).
  // VISIBLE whenever the CONTEXT can drive, even limp — hiding them until
  // torque-on made a fresh Drive tab read as "drive mode does not work"
  // (nothing appears, no affordance). Limp just dims them; the limp CTA
  // already says what to press, and the server limp-holds any input anyway.
  const stickCtx = (method === 'navigation' || method === 'retarget'
    || s.region === 'whole_body');
  $('#sticks').classList.toggle('hidden', !stickCtx);
  $('#sticks').classList.toggle('limp', !on);
  viewer.setDragEnabled(on && method === 'wrist');
  if (method === 'wrist' && $('#scSeg')) segSet('#scSeg', $(`#scSeg [data-m="${s.scMethod}"]`));
  if (method === 'wrist' && $('#postureSeg')) segSet('#postureSeg', $(`#postureSeg [data-h="${s.wristHuman ? 1 : 0}"]`));
  if ($('#driveSC')) $('#driveSC').classList.toggle('on', !!s.selfCollision);

  $('#streamChip').classList.toggle('hidden', !isRt);
  if (isRt) {
    const v = s.vp || {};
    $('#streamChip').innerHTML = `<span class="dot ${v.connected ? 'ok' : ''}"></span>` +
      `${v.device || v.source} · ${v.rate_hz ?? 0} Hz` + (v.age_ms != null ? ` · age ${v.age_ms} ms` : '');
  }
  $('#calChip').classList.toggle('hidden', !isRt);
  if (isRt) {
    $('#calChip').textContent = s.calibrated ? 'TRACKING' : 'NOT CALIBRATED';
    $('#calChip').style.color = s.calibrated ? 'var(--live)' : 'var(--warn)';
  }

  // charts
  for (const ch of Object.values(charts)) {
    for (const key in ch.series) { const p = (s.plots || {})[key]; if (p) ch.push(key, p[0], p[1]); }
    ch.draw();
  }
  if (isRt && $('#m_rate')) {
    const t = s.timing.avg, v = s.vp;
    setMetric('m_rate', v.rate_hz, x => x >= 25, x => x >= 10);
    setMetric('m_jitter', v.jitter_ms, x => x <= 8, x => x <= 20);
    setMetric('m_age', v.age_ms, x => x <= 50, x => x <= 150);
    setMetric('m_retarget', t.update_ms, x => x <= 5, x => x <= 12);
    setMetric('m_physics', t.physics_ms, x => x <= 4, x => x <= 6);
    setMetric('m_rt', rtFactor, x => x >= 0.9, x => x >= 0.6);
  }
  $('#footStatus').textContent =
    `${on ? 'torque on' : 'LIMP'} · ${method} · ${((s.spec || {}).type) || ''} · region ${s.region} · sim t ${s.t.toFixed(1)} s`;
  const tm = s.timing.avg;
  $('#footTiming').textContent = Object.entries(tm)
    .map(([k, v]) => `${k.replace('_ms', '')} ${v.toFixed(2)}ms`).join(' · ');
}

function renderDeviceChip(s, mode) {
  const chip = $('#deviceChip'), vn = s.vpNode || {}, vp = s.vp || {};
  if ((s.method !== 'retarget') || (mode === 'off' && !vn.error)) { chip.classList.add('hidden'); return; }
  chip.classList.remove('hidden');
  let html;
  if (vn.error) html = `<span class="dot"></span>${vn.error}`;
  else if (mode === 'mock') html = `<span class="dot ok"></span>Mock operator · ${vp.rate_hz ?? 0} Hz`;
  else if (mode === 'avp') html = vp.connected
    ? `<span class="dot ok"></span>VP stream (ZMQ) · ${vp.rate_hz ?? 0} Hz`
    : `<span class="dot search"></span>waiting for vision_pro_stream.py (tcp://127.0.0.1:5556)…`;
  else if (!vn.clients) html = `<span class="dot search"></span>connect the app to ws://${lanIp}:${vn.port || 9090}`;
  else if (!vp.connected) {
    const d = (vn.roster && vn.roster[0] && vn.roster[0].device) || 'device';
    html = `<span class="dot warn"></span>${d} found — waiting for tracking…`;
  } else html = `<span class="dot ok"></span>${vn.activeDevice || vp.device || 'device'} · ${vp.rate_hz ?? 0} Hz`;
  chip.innerHTML = html;
}

function setMetric(id, v, okFn, warnFn) {
  const el = $('#' + id); if (!el) return;
  if (v == null) { el.textContent = '—'; el.className = 'v'; return; }
  el.textContent = typeof v === 'number' ? v.toFixed(1) : v;
  el.className = 'v ' + (okFn(v) ? 'ok' : warnFn(v) ? 'warn' : 'bad');
}

function segSet(segSel, btn) {
  if (!btn) return;
  $$(`${segSel} button, ${segSel} .vp-chip`).forEach(b => b.classList.toggle('on', b === btn));
}

/* ── static wiring ────────────────────────────────────────────────────── */
$('#torqueBtn').onclick = () => cmd({ action: 'torque', on: !(state && state.torque) });

$$('#feelSeg button').forEach(b => b.onclick = () => {
  spec.controlType = b.dataset.type; segSet('#feelSeg', b);
  // Stiff/Soft (position-control ↔ passive float) engages the WBC torque engine,
  // which only runs in the Joint / Wrist methods. If the user flips the feel from
  // Drive/Teleop, hop into Joint so the whole robot actually goes stiff / soft.
  if (spec.method !== 'joint' && spec.method !== 'wrist') {
    spec.method = 'joint'; segSet('#modeSeg', $('#modeSeg [data-method="joint"]'));
  }
  sendSpec();
});
$('#stiffSlider').oninput = e => {
  spec.stiffness = +e.target.value; $('#stiffVal').textContent = 'K ' + e.target.value;
};
$('#stiffSlider').onchange = sendSpec;
$$('#regionSeg button').forEach(b => b.onclick = () => {
  spec.region = b.dataset.region; segSet('#regionSeg', b); sendSpec();
  viewer.flyTo({ arm: 'arms', upper_body: 'upper', whole_body: 'whole' }[b.dataset.region]);
});
$$('#modeSeg button').forEach(b => b.onclick = () => {
  spec.method = b.dataset.method; segSet('#modeSeg', b); sendSpec();
});
$$('#vpSeg button').forEach(b => b.onclick = async () => {
  const r = await cmd({ action: 'vp', source: b.dataset.vp }); segSet('#vpSeg', b);
  if (r.vp && r.vp.error) $('#statusChip').textContent = r.vp.error;
});
$$('#sideSeg button').forEach(b => b.onclick = () => {
  cmd({ action: 'hand_side', side: b.dataset.side }); segSet('#sideSeg', b);
});
$('#calibBtn').onclick = async () => {
  const r = await cmd({ action: 'calibration' });
  $('#calibBtn').textContent = r.ok
    ? `Calibration: ${r.user} (${(r.arm_length_m ?? 0).toFixed(2)} m arm)` : `Calibration: ${r.kind}`;
};
$('#resetBtn').onclick = () => { cmd({ action: 'reset' }); viewer.clearPaths(); viewer.clearTrails(); };
$$('#camChips .vp-chip[data-cam]').forEach(ch => ch.onclick = () => { segSet('#camChips', ch); viewer.flyTo(ch.dataset.cam); });
$('#trailsChip').onclick = () => {
  const on = !$('#trailsChip').classList.contains('on');
  $('#trailsChip').classList.toggle('on', on); viewer.setTrails(on);
};
