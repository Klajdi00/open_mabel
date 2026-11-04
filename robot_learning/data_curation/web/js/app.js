// MABEL Data Studio — boot + orchestration. One global playhead drives the 3D
// viewport, the video wall, the dense bar-graph and the timeline; everything
// repaints only when the frame changes (the 3D stage renders continuously on
// its own loop for orbit damping).
import * as bus from './bus.js';
import { api, waitJob } from './api.js';
import * as store from './store.js';
import { Sequence } from './sequence.js';
import { initSplitters } from './splitters.js';
import { Viewport3D } from './viewport3d.js';
import { VideoWall } from './videowall.js';
import { BarGraph } from './bargraph.js';
import { Browser } from './browser.js';
import { Topics } from './topics.js';
import { defectCategory, defectMessage, categoryWord } from './defects.js';
import { Timeline } from './timeline.js';
import { IMU } from './imu.js';
import { PanelManager } from './panels.js';
import { jointCompareFactory } from './jointcompare.js';
import { signalsFactory } from './signals.js';
import { modalitiesFor } from './names.js';
import * as status from './status.js';

const $ = (s) => document.querySelector(s);
const SPEEDS = [0.25, 0.5, 1, 2, 4];

const state = { frame: 0, playing: false, dir: 1, speed: 1, loop: false };
const seq = new Sequence(15);
let active = null;            // {ds, ep, entry}

// ── panels ────────────────────────────────────────────────────────────────
initSplitters();
const viewport = new Viewport3D($('#viewport3d'));
const videowall = new VideoWall();   // detached grid; hosted by the floating "Cameras" panel
const bargraph = new BarGraph($('#barCanvas'), $('#stripCanvas'), (srcFrame) => {
  // strip scrub gives a SOURCE frame in the active episode -> map back to global
  if (!active) return; setGlobal(globalFromSource(active, srcFrame)); pause();
});
const topics = new Topics($('#topicsPane'), $('#inspectorPane'), {
  // toggling a camera shows/hides BOTH its video tile and its 3D projection,
  // so the projections in the 3D view always match the live video streams shown
  onCamera: (name, on) => {
    videowall.toggle(name, on);
    viewport.setActiveCams(videowall.active);
    if (on){ launchCameras(); setLayer('frustum', true); }
    if (active) videowall.update(active.ds, active.ep, curSource());
  },
  onLayer: (key, on) => setLayer(key, on),
  onCloudCam: (name, on) => { viewport.setCloudCam(name, on); if (active && !state.playing) applyFrame(); },  // per-camera 3D point cloud
  onTraceTf: (tf, on) => { viewport.setTraceTf(tf, on); if (active && !state.playing) applyFrame(); },        // multi-select TF trajectory
  onSlamCam: (name, on) => viewport.setReconMeshCam(name, on),  // cuSLAM mesh per camera (which SLAM cam)
});
viewport.onDistribution = (legend) => topics.setTrajLegend(legend);

// One place to flip a 3D-scene layer; keeps the viewport, the toggle chips and
// the Topics checkboxes in sync.
function setLayer(key, on){
  viewport.setLayer(key, on);
  const chip = document.querySelector(`.vp-chip[data-layer="${key}"]`);
  if (chip) chip.classList.toggle('on', on);
  topics.reflectLayer(key, on);
  if (active && !state.playing) applyFrame();          // reflect the toggle immediately
}
// ── dockable floating-panel system (rviz/rerun-style add-able windows) ──────
const panelHost = document.createElement('div'); panelHost.className = 'fpanel-host';
document.body.appendChild(panelHost);
const panels = new PanelManager(panelHost);
panels.register('jointCompare', 'Joint Compare', jointCompareFactory);
panels.register('signals', 'Signals', signalsFactory);
panels.register('cameras', 'Cameras', camerasPanelFactory);
panels.register('map2d', '2D Occupancy Map', map2dPanelFactory);
panels.register('depth', '2D Depth', depthPanelFactory);
panels.register('tftree', 'TF Tree', (panel) => {
  panel.body.classList.add('tf-host');
  (viewport._ready || Promise.resolve()).then(() => renderTfTree(panel.body));
  return { update(){}, };               // the transform tree is static per rig
});
panels.register('markers', 'Markers', eventsPanelFactory('markers'));
panels.register('qc', 'QC · Validation', eventsPanelFactory('qc'));
panels.register('sync', 'Sync · Temporal', syncPanelFactory);
panels.register('corpus', 'Dataset Quality', corpusPanelFactory);
panels.register('overview', 'Dataset Overview', overviewPanelFactory);
panels.register('issues', 'Issues', issuesPanelFactory);
panels.register('imu', 'IMU', (panel) => {
  const inst = new IMU(panel.body); let cur = null;
  return { update(e, f){ if (e !== cur){ cur = e; inst.setEpisode(e); } inst.update(e, f); },
    resize(){ inst.draw(); }, destroy(){ inst.destroy?.(); } };
});
// all active cameras as a strip (one tile per camera, synced to the playhead)
// THE camera view — a floating panel that hosts the shared VideoWall grid (the
// active cameras toggled in Topics). Closing the panel detaches the grid; the
// selection persists so re-opening restores it.
function camerasPanelFactory(panel){
  panel.body.appendChild(videowall.el); _camerasOpen = true;
  if (active) videowall.update(active.ds, active.ep, curSource());
  return {
    update(e, f){ if (active) videowall.update(active.ds, active.ep, f); },
    destroy(){ if (videowall.el.parentNode === panel.body) videowall.el.remove(); _camerasOpen = false; },
  };
}
// Open (or focus) the Cameras panel as a FULL-WIDTH bottom band (20% tall), then
// re-tile any left/right docks so they sit ABOVE it. This makes the dashboard
// layout order-independent: cameras always own the bottom, side docks never
// overlap it no matter what was placed first.
function launchCameras(){
  let p = panels.panels.find(x => x.type === 'cameras');
  if (p){ panels._raise(p); return p; }
  p = panels.add('cameras');
  if (p){
    const ws = panels.wsRect(), h = ws.h * 0.20;
    panels._snap(p, { x: ws.x, y: ws.y + ws.h - h, w: ws.w, h });   // full-width bottom band
    panels._retileSide('left'); panels._retileSide('right');         // push side docks above it
  }
  return p;
}
// 2D occupancy map — gmapping grid recoloured into the warm UI palette (bright
// walls on a dark unknown field, faint "explored" wash, subtle grid) with the
// live robot position + heading. Recolour happens ONCE on load (small grid).
function map2dPanelFactory(panel){
  panel.body.innerHTML = `<canvas class="m2d-cv"></canvas><div class="m2d-empty">no occupancy (needs lidar + chassis pose)</div>`;
  const cv = panel.body.querySelector('.m2d-cv'), empty = panel.body.querySelector('.m2d-empty');
  let entry = null, meta = null, styled = null, frame = 0;

  function recolor(im){                                  // grayscale gmapping → UI palette (once)
    const oc = document.createElement('canvas'); oc.width = im.width; oc.height = im.height;
    const g = oc.getContext('2d'); g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, im.width, im.height), a = d.data;
    for (let i = 0; i < a.length; i += 4){ const v = a[i];
      if (v < 60){ a[i] = 242; a[i + 1] = 238; a[i + 2] = 230; a[i + 3] = 236; }       // occupied → bright wall
      else if (v > 200){ a[i] = 242; a[i + 1] = 238; a[i + 2] = 230; a[i + 3] = 18; }  // free → faint wash
      else { a[i + 3] = 0; }                                                            // unknown → transparent
    }
    g.putImageData(d, 0, 0); return oc;
  }
  function load(e){
    entry = e; meta = null; styled = null; empty.textContent = 'building occupancy…'; empty.style.display = 'block';
    api.occupancy(e.ds, e.ep).then(o => {
      if (e !== entry) return;
      if (o && o.png_b64){ meta = o; const im = new Image(); im.onload = () => { styled = recolor(im); empty.style.display = 'none'; draw(); }; im.src = 'data:image/png;base64,' + o.png_b64; }
      else { empty.textContent = 'no occupancy (needs lidar + chassis pose)'; empty.style.display = 'block'; }
    }).catch(() => {});
  }
  function draw(){
    const dpr = devicePixelRatio;
    const W = cv.width = Math.max(2, cv.clientWidth * dpr), H = cv.height = Math.max(2, cv.clientHeight * dpr);
    const x = cv.getContext('2d'); x.clearRect(0, 0, W, H);
    if (!styled || !meta || !entry) return;
    const s = Math.min(W / styled.width, H / styled.height);
    const iw = styled.width * s, ih = styled.height * s, ox = (W - iw) / 2, oy = (H - ih) / 2;
    x.imageSmoothingEnabled = false; x.drawImage(styled, ox, oy, iw, ih);
    // subtle grid every ~0.5 m
    const step = (0.5 / meta.res) * s;
    if (step > 6){ x.strokeStyle = 'rgba(242,238,230,.05)'; x.lineWidth = dpr;
      for (let gx = ox; gx <= ox + iw + 0.5; gx += step){ x.beginPath(); x.moveTo(gx, oy); x.lineTo(gx, oy + ih); x.stroke(); }
      for (let gy = oy; gy <= oy + ih + 0.5; gy += step){ x.beginPath(); x.moveTo(ox, gy); x.lineTo(ox + iw, gy); x.stroke(); } }
    // robot from chassis pose at the current frame
    const t = entry.bundle.topics['pose/chassis']; if (!t) return;
    const f = Math.max(0, Math.min(frame, entry.T - 1)), d = t.data, dim = t.dim;
    const W2P = (X, Y) => [ox + (X - meta.minx) / meta.res * s, oy + (meta.h - (Y - meta.miny) / meta.res) * s];  // world→pixel (image flipped: +Y up)
    // fading past trail
    x.strokeStyle = 'rgba(194,91,42,.55)'; x.lineWidth = 2 * dpr; x.beginPath();
    for (let i = 0; i <= f; i += 2){ const [tx, ty] = W2P(d[i * dim], d[i * dim + 1]); i ? x.lineTo(tx, ty) : x.moveTo(tx, ty); }
    x.stroke();
    const [sx, sy] = W2P(d[f * dim], d[f * dim + 1]);
    const qw = d[f * dim + 3], qx = d[f * dim + 4], qy = d[f * dim + 5], qz = d[f * dim + 6];
    const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    const hx = -Math.cos(yaw), hy = Math.sin(yaw), rr = 8 * dpr;   // robot front; +Y up on screen, chassis yaw is 180° from facing
    // glowing heading wedge
    x.save(); x.shadowColor = 'rgba(194,91,42,.85)'; x.shadowBlur = 9 * dpr;
    x.fillStyle = '#C25B2A'; x.beginPath();
    x.moveTo(sx + hx * rr * 2.4, sy + hy * rr * 2.4);
    x.lineTo(sx - hy * rr - hx * rr * 0.4, sy + hx * rr - hy * rr * 0.4);
    x.lineTo(sx + hy * rr - hx * rr * 0.4, sy - hx * rr - hy * rr * 0.4);
    x.closePath(); x.fill(); x.restore();
    // body dot
    x.fillStyle = '#C25B2A'; x.strokeStyle = '#fff'; x.lineWidth = 2 * dpr;
    x.beginPath(); x.arc(sx, sy, rr, 0, 7); x.fill(); x.stroke();
  }
  return {
    update(e, f){ frame = f; if (e !== entry) load(e); else draw(); },
    resize(){ draw(); },
  };
}

// ── event detection + QC (Markers / QC panels) ──────────────────────────────
function tc(f){ const s = f / seq.fps; return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}:${String(Math.floor((s % 1) * seq.fps)).padStart(2, '0')}`; }
// Scan the timeline (each clip's dense bundle, mapped to global frames) for
// salient events → markers, and data problems → QC issues. Edge-triggered so
// the lists stay short. Drives the ruler ticks + the Markers/QC panels.
function scanTimeline(){
  const markers = [], qc = []; const TG = 0.5, TS = 0.08, TJ = 0.5;   // grasp / speed / join thresholds
  let prevLast = null;
  for (const clip of seq.clips){
    const e = store.peek(clip.ds, clip.ep); if (!e) continue;
    const B = e.bundle.topics, grasp = B['teleop/grasp'], mode = B['teleop/mode'], bt = B['base_twist'], jp = B['joint_pos'];
    const G = (f) => clip.start + (f - clip.inFrame);
    let moving = null, nanRun = false;
    for (let f = clip.inFrame; f < clip.outFrame; f++){
      if (grasp && f > clip.inFrame) for (let h = 0; h < 2; h++){
        const a = grasp.data[(f - 1) * grasp.dim + h], b = grasp.data[f * grasp.dim + h];
        if (a < TG && b >= TG) markers.push({ frame: G(f), kind: 'grasp', label: (h ? 'R' : 'L') + ' grasp close', color: '#5eff9b' });
        else if (a >= TG && b < TG) markers.push({ frame: G(f), kind: 'grasp', label: (h ? 'R' : 'L') + ' release', color: '#ff6b5e' }); }
      if (mode && f > clip.inFrame && mode.data[(f - 1) * mode.dim] !== mode.data[f * mode.dim])
        markers.push({ frame: G(f), kind: 'mode', label: 'mode → ' + (mode.data[f * mode.dim] > 0.5 ? 'NAV' : 'MANIP'), color: '#f0b68b' });
      if (bt){ const sp = Math.hypot(bt.data[f * bt.dim], bt.data[f * bt.dim + 1]); const mv = sp > TS;
        if (moving === null) moving = mv; else if (mv !== moving){ markers.push({ frame: G(f), kind: 'motion', label: mv ? 'move start' : 'move stop', color: '#5eb0ff' }); moving = mv; } }
      if (jp){ let nan = false; for (let k = 0; k < jp.dim; k++) if (Number.isNaN(jp.data[f * jp.dim + k])){ nan = true; break; }
        if (nan && !nanRun) qc.push({ frame: G(f), kind: 'error', label: 'NaN in joint_pos' }); nanRun = nan; }
    }
    if (prevLast && jp){ let mx = 0; for (let k = 0; k < jp.dim; k++){ const dd = Math.abs(jp.data[clip.inFrame * jp.dim + k] - prevLast[k]); if (dd > mx) mx = dd; }
      if (mx > TJ) qc.push({ frame: clip.start, kind: 'warn', label: `join jump ${mx.toFixed(2)} rad` }); }
    if (jp){ const last = clip.outFrame - 1, arr = []; for (let k = 0; k < jp.dim; k++) arr.push(jp.data[last * jp.dim + k]); prevLast = arr; }
  }
  timeline.setEvents(markers.slice(0, 500), qc.slice(0, 200));
  document.dispatchEvent(new CustomEvent('studio-events'));
}
let _scanT; function scanSoon(){ clearTimeout(_scanT); _scanT = setTimeout(scanTimeline, 250); }
function jumpMarker(dir){
  const ms = timeline.markers.map(m => m.frame).sort((a, b) => a - b); if (!ms.length) return;
  const f = state.frame; let target = dir > 0 ? ms.find(x => x > f) : [...ms].reverse().find(x => x < f);
  if (target != null){ setGlobal(target); pause(); }
}
// Markers / QC list panels (kind = 'markers' | 'qc'); re-render on 'studio-events'.
// Sync · Temporal-integrity panel: fetches quality.json for the active episode and
// shows the action->obs control lag, per-probe confidence, timing gaps, and a
// one-gesture "Apply lag correction" that writes clip.modOffset. Also mirrors the
// flags into the QC lane so they show on the timeline.
function syncPanelFactory(panel){
  let curKey = null, lastRep = null;
  const badge = (bad, txt) => `<span class="sync-badge ${bad ? 'bad' : 'ok'}">${txt}</span>`;
  const render = (rep) => {
    lastRep = rep;
    if (!rep){ panel.body.innerHTML = `<div class="evt-empty">load an episode</div>`; return; }
    const h = rep.headline;
    panel.body.innerHTML = `<div class="sync-body">
      <div class="sync-row">${badge(rep.flags.systematic_lag, 'lag')} ${badge(rep.flags.timing_gaps, 'gaps')}
        <span class="sync-fps">${rep.fps} Hz · ${rep.T}f</span></div>
      ${h ? `<div class="sync-lag">action→obs lag <b>${h.lag_frames}f</b> · ${h.lag_ms} ms · conf ${h.confidence}</div>`
          : `<div class="sync-lag dim">no usable command↔state probe</div>`}
      <div class="sync-probes">${rep.probes.map(p => `<div class="sync-probe ${p.usable ? '' : 'dim'}">${p.probe} — ${p.usable ? `${p.lag_frames}f · c${p.confidence}${p.flag ? ' ⚠' : ''}` : 'flat'}</div>`).join('')}</div>
      <div class="sync-gaps">${rep.gaps.n_gaps} timing gap(s) · wall ${rep.gaps.fps_wall ?? '?'} Hz (med Δt ${rep.gaps.median_dt ?? '?'}s)</div>
      ${rep.recommended_offset
        ? `<button class="sync-fix">Apply lag correction · ${rep.recommended_offset.frames}f on ${rep.recommended_offset.modality}</button>`
        : `<div class="sync-ok">✓ temporally aligned — safe to export</div>`}
      ${rep.defects && rep.defects.some(d => defectCategory(d.type) === 'dropped')
        ? `<button class="sync-cull">✂ Auto-cull dropped frames</button>` : ''}
      <button class="sync-align">⇉ Align proprio + control to pixels</button></div>`;
    const fix = panel.body.querySelector('.sync-fix');
    if (fix) fix.onclick = () => {
      const clip = seq.selected()[0] || seq.clips[0];
      if (!clip) return toast('no clip on the timeline');
      seq.setModOffset(clip, rep.recommended_offset.modality, rep.recommended_offset.frames);
      toast(`applied ${rep.recommended_offset.frames}f offset on ${rep.recommended_offset.modality}`, 'ok');
    };
    const cull = panel.body.querySelector('.sync-cull');
    if (cull) cull.onclick = () => autoCull(rep);
    const align = panel.body.querySelector('.sync-align');
    if (align) align.onclick = () => alignToPixels();
  };
  const refresh = () => {
    if (!active){ curKey = null; render(null); return; }
    const key = active.ds + '|' + active.ep;
    if (key === curKey) return;                       // fetch once per episode, not per frame
    curKey = key;
    api.quality(active.ds, active.ep).then(render).catch(() => render(null));
  };
  refresh();
  return { update(){ refresh(); }, destroy(){} };
}

// robustly align proprioception + control to the camera pixels (multi-window median
// camera↔EE cross-correlation). Fail-safe: only applies when windows agree.
async function alignToPixels(){
  if (!active) return toast('no active episode');
  toast('aligning proprio+control to pixels…');
  try {
    const a = await api.align(active.ds, active.ep);
    if (!a.usable || !a.recommended_offset)
      return toast(`can't align: ${a.reason || 'low confidence'} (conf ${a.confidence ?? '?'})`, 'err');
    const clip = seq.selected()[0] || seq.clips.find(c => c.ds === active.ds && c.ep === active.ep) || seq.clips[0];
    if (!clip) return toast('no clip on the timeline');
    seq.commit();
    for (const mod of a.recommended_offset.modalities) seq.setModOffset(clip, mod, a.recommended_offset.frames);
    timeline.render();
    toast(`aligned proprio+control ${a.recommended_offset.frames}f → ${a.cam} pixels (conf ${a.confidence}, ${a.n_windows} windows)`, 'ok');
  } catch (e){ toast('align failed: ' + e.message, 'err'); }
}

// one-gesture removal of obviously-unusable spans (data missing / disconnection):
// writes reject ranges over every "dropped"-category defect on the active clip.
function autoCull(rep){
  if (!active) return toast('no active episode');
  const clip = seq.clips.find(c => c.ds === active.ds && c.ep === active.ep);
  if (!clip) return toast('active episode not on the timeline');
  const dropped = (rep.defects || []).filter(d => defectCategory(d.type) === 'dropped');
  if (!dropped.length) return toast('nothing to cull', 'ok');
  seq.commit();
  let spans = 0, frames = 0;
  for (const d of dropped){
    const lo = Math.max(clip.inFrame, d.start), hi = Math.min(clip.outFrame, d.end);
    if (hi <= lo) continue;
    seq.addRange(clip.start + (lo - clip.inFrame), clip.start + (hi - clip.inFrame), 'reject');
    spans++; frames += hi - lo;
  }
  timeline.render(); autosave();
  toast(`auto-culled ${frames}f across ${spans} dropped span(s) — excised on Save`, 'ok');
}

function cqcol(q){ return q >= 0.9 ? '#3ecf8e' : q >= 0.7 ? '#E7C24B' : '#D9463A'; }

// live before→current curation delta for the ACTIVE episode: how many frames the
// current reject ranges remove and how much that lifts the quality score.
function curationDelta(){
  if (!active || !active.entry || !active.entry.quality) return null;
  const q = active.entry.quality, sev = q.severity || [], T = q.T || sev.length;
  const clip = seq.clips.find(c => c.ds === active.ds && c.ep === active.ep);
  const rejected = new Set();
  if (clip) for (const r of seq.ranges){
    if (r.kind !== 'reject') continue;
    const gs = Math.max(clip.start, r.start), ge = Math.min(clip.start + clip.len, r.end);
    for (let g = gs; g < ge; g++) rejected.add(clip.inFrame + (g - clip.start));
  }
  let kept = 0, keptSev = 0, allSev = 0;
  for (let f = 0; f < T; f++){ allSev += sev[f] || 0; if (!rejected.has(f)){ kept++; keptSev += sev[f] || 0; } }
  const beforeQ = 1 - allSev / Math.max(1, T), currentQ = kept ? 1 - keptSev / kept : 1;
  const imp = beforeQ > 0 ? (currentQ - beforeQ) / beforeQ * 100 : 0;
  return { beforeFrames: T, currentFrames: kept, rejected: T - kept, beforeQ, currentQ, improvement: imp };
}

// always-on Dataset Overview: dataset size/hours/modalities/task/quality + a live
// before→current curation delta with the quality-improvement %.
function overviewPanelFactory(panel){
  let curDs = null, ov = null;
  const fmt = (n) => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : '' + n;
  const render = () => {
    const d = curationDelta(), q = ov;
    panel.body.innerHTML = `<div class="ov-body">
      <div class="ov-sec">Dataset</div>
      ${q ? `<div class="ov-task">${q.overview.task}</div>
        <dl class="ov-dl">
          <dt>episodes</dt><dd>${q.n_episodes}</dd><dt>frames</dt><dd>${fmt(q.overview.total_frames)}</dd>
          <dt>duration</dt><dd>${q.overview.hours} h</dd><dt>size</dt><dd>${q.overview.size_mb} MB</dd>
          <dt>quality</dt><dd style="color:${cqcol(q.dataset_score)}">${q.dataset_score}</dd>
          <dt>clean</dt><dd>${q.n_clean}/${q.n_episodes}</dd></dl>
        <div class="ov-mods">${q.overview.modalities.map(m => `<span class="ov-chip">${m}</span>`).join('')}</div>`
        : `<div class="evt-empty">load a dataset</div>`}
      <div class="ov-sec">Curation · before → current</div>
      ${d ? `<div class="ov-delta">
        <div class="ov-row"><span class="ov-k">frames</span><b>${d.beforeFrames}</b><span class="ov-arw">→</span><b class="cur">${d.currentFrames}</b></div>
        <div class="ov-row"><span class="ov-k">quality</span><b>${d.beforeQ.toFixed(2)}</b><span class="ov-arw">→</span><b class="cur" style="color:${cqcol(d.currentQ)}">${d.currentQ.toFixed(2)}</b></div>
        <div class="ov-imp ${d.improvement >= 0 ? 'up' : 'down'}">${d.improvement >= 0 ? '▲' : '▼'} ${Math.abs(d.improvement).toFixed(0)}% quality${d.rejected ? ` · −${d.rejected}f culled` : ''}</div>
        <div class="ov-note">policy-eval Δ — run training (Phase 4)</div></div>`
        : `<div class="evt-empty">no active episode</div>`}</div>`;
  };
  const refresh = () => {
    if (active && active.ds !== curDs){ curDs = active.ds;
      api.datasetQuality(active.ds).then(x => { ov = x; render(); }).catch(() => { ov = null; render(); }); }
    render();
  };
  refresh();
  const unsub = bus.on('sequenceChanged', render);      // live update as the user culls/edits
  return { update(){ refresh(); }, destroy(){ unsub(); } };
}

// Always-on Issues log (right side): a concise, colour-coded stream of the active
// episode's problems — dropped data, out-of-sync, low frequency — click to jump.
function issuesPanelFactory(panel){
  const rejectedSet = () => {
    const set = new Set();
    const clip = active && seq.clips.find(c => c.ds === active.ds && c.ep === active.ep);
    if (!clip) return set;
    for (const r of seq.ranges){
      if (r.kind !== 'reject') continue;
      const gs = Math.max(clip.start, r.start), ge = Math.min(clip.start + clip.len, r.end);
      for (let g = gs; g < ge; g++) set.add(clip.inFrame + (g - clip.start));
    }
    return set;
  };
  const ORDER = { dropped: 0, unsynced: 1, degraded: 2 };
  const render = () => {
    const q = active && active.entry && active.entry.quality;
    if (!q){ panel.body.innerHTML = `<div class="evt-empty">no episode loaded</div>`; return; }
    const defs = q.defects || [], cat = { dropped: 0, unsynced: 0, degraded: 0 };
    for (const d of defs) cat[defectCategory(d.type)]++;
    const rej = rejectedSet();
    const culled = (d) => { let n = 0, c = 0; for (let f = d.start; f < d.end; f++){ n++; if (rej.has(f)) c++; } return n && c / n > 0.6; };
    const sorted = [...defs].sort((a, b) => (ORDER[defectCategory(a.type)] - ORDER[defectCategory(b.type)]) || (a.start - b.start));
    panel.body.innerHTML = `<div class="lg-head">
        <div class="lg-count" style="color:${defs.length ? '' : '#3ecf8e'}">${defs.length || '✓'}</div>
        <div class="lg-sub">${defs.length ? `issue${defs.length !== 1 ? 's' : ''}` : 'clean'}
          <span class="lg-legend"><i class="dropped"></i>${cat.dropped} <i class="unsynced"></i>${cat.unsynced} <i class="degraded"></i>${cat.degraded}</span></div></div>
      <div class="lg-list">${defs.length
        ? sorted.map(d => { const single = (d.end - d.start) <= 1; return `<button class="lg-row ${defectCategory(d.type)}${culled(d) ? ' culled' : ''}" data-f="${d.start}">
            <i class="lg-dot"></i><span class="lg-msg">${defectMessage(d)}</span>
            <span class="lg-frames">f${d.start}${single ? '' : '–' + d.end}</span></button>`; }).join('')
        : `<div class="lg-clean">✓ no issues detected — dataset is clean</div>`}</div>`;
    panel.body.querySelectorAll('.lg-row').forEach(b => b.onclick = () => { setGlobal(+b.dataset.f); pause(); });
  };
  render();
  const unsub = bus.on('sequenceChanged', render);      // reflect culls/edits live
  return { update(){ render(); }, destroy(){ unsub(); } };
}

// whole-dataset quality dashboard: per-episode scores worst-first + a joint score.
function corpusPanelFactory(panel){
  let curDs = null;
  const cq = (q) => q >= 0.9 ? '#3ecf8e' : q >= 0.7 ? '#E7C24B' : '#D9463A';
  const render = (c) => {
    if (!c){ panel.body.innerHTML = `<div class="evt-empty">load a dataset</div>`; return; }
    panel.body.innerHTML = `<div class="cq-head"><div class="cq-score" style="color:${cq(c.dataset_score)}">${c.dataset_score}</div>
        <div class="cq-sub">${c.dataset}<br>${c.n_episodes} eps · ${c.n_clean} clean · ${c.n_flagged} flagged · min ${c.min_quality}</div></div>
      <div class="cq-list">${c.episodes.map(e => `<button class="cq-row" data-ep="${e.file}">
        <span class="cq-bar"><i style="width:${Math.round(e.quality * 100)}%;background:${cq(e.quality)}"></i></span>
        <span class="cq-q">${e.quality.toFixed(2)}</span>
        <span class="cq-name">${e.file.replace('.h5', '')}</span>
        <span class="cq-types">${Object.entries(e.types).map(([t, n]) => `${t}×${n}`).join(' · ') || 'clean'}</span></button>`).join('')}</div>`;
    panel.body.querySelectorAll('.cq-row').forEach(b => b.onclick = () => addEpisode({ ds: c.dataset, ep: b.dataset.ep }));
  };
  const refresh = () => {
    if (!active){ curDs = null; render(null); return; }
    if (active.ds === curDs) return; curDs = active.ds;
    panel.body.innerHTML = `<div class="evt-empty">scoring ${active.ds}…</div>`;
    api.datasetQuality(active.ds).then(render).catch(() => render(null));
  };
  refresh();
  return { update(){ refresh(); }, destroy(){} };
}

function eventsPanelFactory(kind){
  return (panel) => {
    const render = () => {
      const items = kind === 'markers' ? timeline.markers : timeline.qc;
      panel.body.innerHTML = `<div class="evt-list">` + (items.length
        ? items.map(m => `<button class="evt-row" data-f="${m.frame}"><i class="evt-dot" style="background:${m.color || (m.kind === 'error' ? '#D23B2A' : '#E9A679')}"></i><span class="evt-lbl">${m.label}</span><b class="evt-tc">${tc(m.frame)}</b></button>`).join('')
        : `<div class="evt-empty">${kind === 'markers' ? 'no events detected' : 'no issues found'}</div>`) + `</div>`;
      panel.body.querySelectorAll('.evt-row').forEach(b => b.onclick = () => { setGlobal(+b.dataset.f); pause(); });
    };
    render(); const h = () => render(); document.addEventListener('studio-events', h);
    return { update(){}, destroy(){ document.removeEventListener('studio-events', h); } };
  };
}

// 2D depth view (Rerun DepthImage) — colorized depth that auto-follows the ACTIVE
// cameras (the ones toggled on in Topics / shown in the camera + 3D views); no
// manual selection. One tile per active depth-capable camera, playhead-synced.
function depthPanelFactory(panel){
  panel.body.innerHTML = `<div class="vw-grid depth-grid"></div><div class="depth-empty">toggle a depth camera in Topics</div>`;
  const grid = panel.body.querySelector('.depth-grid'), empty = panel.body.querySelector('.depth-empty');
  let entry = null, sig = '', imgs = {};
  return { update(e, f){ entry = e; if (!entry) return;
      const depthCams = entry.cams.filter(c => c.depth).map(c => c.name);
      const act = [...videowall.active].filter(n => depthCams.includes(n));
      const s = act.join(',');
      if (s !== sig){ sig = s; grid.innerHTML = ''; imgs = {};
        empty.style.display = act.length ? 'none' : 'flex';
        for (const n of act){
          const t = document.createElement('div'); t.className = 'cam-tile';
          const im = document.createElement('img'); im.decoding = 'async';
          const lb = document.createElement('span'); lb.className = 'cam-tag'; lb.textContent = n + ' · depth';
          t.append(im, lb); grid.appendChild(t); imgs[n] = im;
        } }
      for (const n in imgs) imgs[n].src = api.depthUrl(entry.ds, entry.ep, n, f);
    } };
}

// "+ Window" dropdown in the top-right (next to Generate Data / Save Episode)
function buildAddWindowMenu(){
  const btn = $('#addWinBtn'), menu = $('#addWinMenu');
  menu.innerHTML = `<div class="awm-sec">Add window</div>` +
    panels.list().map(p => `<button class="awm-item" data-t="${p.type}">+ ${p.label}</button>`).join('') +
    `<div class="awm-hint">drag a window to a border to tile (½ / ¼ / full)</div>` +
    `<div class="awm-sec">Layouts</div>
     <button class="awm-item awm-preset" data-p="analysis">▦ Analysis layout</button>
     <button class="awm-item awm-preset" data-p="clear">▢ Clear windows</button>`;
  menu.querySelectorAll('.awm-item[data-t]').forEach(b => b.onclick = () => { panels.add(b.dataset.t); menu.classList.add('hidden'); });
  menu.querySelectorAll('.awm-preset').forEach(b => b.onclick = () => { applyLayout(b.dataset.p); menu.classList.add('hidden'); });
  btn.onclick = (e) => { e.stopPropagation(); menu.classList.toggle('hidden'); };
  document.addEventListener('click', () => menu.classList.add('hidden'));
}

// Quick-tile presets. "analysis": cameras along the bottom, joint-state of the
// four limbs stacked on the right, IMU + 2D map on the left — the 3D viewer
// stays visible in the middle.
function applyLayout(name){
  panels.clearAll();
  if (name === 'clear') return;
  const W = panels.wsRect();
  if (name === 'analysis'){
    const camH = Math.round(W.h * 0.30), upper = W.h - camH;
    panels.addAt('cameras', { x: W.x, y: W.y + upper, w: W.w, h: camH });
    panels.addAt('jointCompare', { x: W.x + W.w * 0.66, y: W.y, w: W.w * 0.34, h: upper },
      ['left_arm', 'left_hand', 'right_arm', 'right_hand']);
    panels.addAt('imu', { x: W.x, y: W.y, w: W.w * 0.24, h: upper / 2 });
    panels.addAt('map2d', { x: W.x, y: W.y + upper / 2, w: W.w * 0.24, h: upper / 2 });
  }
}
document.addEventListener('panel-scrub', (e) => { if (active) setGlobal(globalFromSource(active, e.detail)); pause(); });

const browser = new Browser($('#browserBody'), addEpisode, loadBatch);
const timeline = new Timeline({
  scrollEl: $('#tlScroll'), tracksEl: $('#tlTracks'), rulerEl: $('#rulerCanvas'),
  playheadEl: $('#tlPlayhead'), nameEl: $('#tlName'),
  seq, getFrame: () => state.frame, setFrame: (f) => { setGlobal(f); pause(); }, onToast: toast,
  modalitiesFor: (clip) => { const e = store.peek(clip.ds, clip.ep); return e ? modalitiesFor(e.meta) : []; },
  entryFor: (clip) => store.peek(clip.ds, clip.ep),
});

const _vpMenus = [];                                 // open viewer dropdown menus (declared before buildToggles runs)
function _closeVpMenus(){ for (const m of _vpMenus) m.classList.add('hidden'); }
// position a menu under its button in viewport coords (menus live on <body> at a
// high z-index so floating panels never cover them)
function _openVpMenu(btn, menu){
  const r = btn.getBoundingClientRect();
  menu.style.left = Math.min(r.left, window.innerWidth - 210) + 'px';
  menu.style.top = (r.bottom + 6) + 'px';
  menu.classList.remove('hidden');
}
buildToggles();
buildAddWindowMenu();
browser.load();
checkHealth();
maybeDemo();
// always-on dataset overview (beside the library) + issues log (right side)
try { const _ov = panels.add('overview'); if (_ov) panels.dockToSide(_ov, 'left'); } catch (e){ console.error('overview panel', e); }
try { const _iss = panels.add('issues'); if (_iss) panels.dockToSide(_iss, 'right'); } catch (e){ console.error('issues panel', e); }

// Library self-heals: manual ↻, auto-refresh when the tab regains focus, and a
// retry if the catalog is momentarily empty (e.g. a tab opened before data
// existed) — so "no data" never gets stuck on screen.
$('#libRefresh').onclick = () => browser.load();
window.addEventListener('focus', () => browser.load());

// ?demo=1 → auto-load the first episode of the first dataset (handy first run + tests)
async function maybeDemo(){
  if (seq.clips.length) return;                        // already have a sequence — don't clobber it
  try {
    let cat = await api.catalog();
    for (let tries = 0; !cat.datasets.length && tries < 5; tries++){   // wait out an empty catalog
      await new Promise(r => setTimeout(r, 800)); cat = await api.catalog();
    }
    const ds = cat.datasets[0]; if (!ds) return;
    // two example clips, ready to scrub
    for (const ep of ds.episodes.slice(0, 2))
      await addEpisode({ ds: ds.name, ep: ep.file, epLabel: ep.task, num_frames: ep.num_frames, fps: ep.fps, cameras: ep.cameras });
    if (active){
      // head cam + base front cam + both wrist cams (whichever this episode has)
      const want = ['zed_left', 'body_front_cam', 'l_wrist_cam', 'r_wrist_cam'];
      for (const c of want) if (active.entry.cams.some(x => x.name === c)){ videowall.toggle(c, true); topics.reflectCamera(c, true); }
      viewport.setActiveCams(videowall.active);
      setLayer('frustum', true);                       // 3D camera projection on
      launchCameras();                                 // camera view (bottom)
      launchPanel('map2d', 'left');                    // 2D occupancy map (left)
      videowall.update(active.ds, active.ep, 0);
    }
  } catch (e){ console.error('default-view load failed', e); }
}

// ── episode add / active-episode switching ─────────────────────────────────
// fetch the temporal-integrity + defect report for an episode once and attach it
// to the loaded entry, so the timeline can highlight problems per modality/DOF.
function attachQuality(entry, ds, ep){
  if (!entry || entry.quality || entry._qLoading) return;
  entry._qLoading = true;
  api.quality(ds, ep).then(q => {
    entry.quality = q;
    if (active && active.entry === entry) videowall.quality = q;   // colour the live camera tiles
    timeline.render();
  }).catch(() => {}).finally(() => { entry._qLoading = false; });
  if (!entry.freq && !entry._fLoading){                            // per-channel frequency tracks (heat)
    entry._fLoading = true;
    api.frequency(ds, ep).then(fr => { entry.freq = fr; timeline.render(); })
      .catch(() => {}).finally(() => { entry._fLoading = false; });
  }
}

async function addEpisode({ ds, ep, epLabel, num_frames, fps, cameras }){
  try {
    const entry = await store.ensure(ds, ep);
    status.markOpened(ds, ep);                       // inspected → "where you left off"
    attachQuality(entry, ds, ep);
    // head cam for the timeline filmstrip (FCP-style), falling back gracefully
    const cam = (entry.cams.find(c => c.name === 'zed_left')
              || entry.cams.find(c => c.name === 'body_cam')
              || entry.cams.find(c => c.name === 'body_front_cam') || entry.cams[0])?.name;
    const clip = seq.addClip({ ds, ep, epLabel: epLabel || entry.meta.task || ep,
      inFrame: 0, outFrame: entry.T, srcMax: entry.T, fps: entry.fps });
    clip.cam = cam;
    if (seq.clips.length === 1){ await setActive(ds, ep); setGlobal(0); }
    timeline.render(); autosave();
    toast(`added ${epLabel || ep} (${entry.T}f)`, 'ok');
  } catch (e){ console.error('addEpisode failed', e); toast('failed to add: ' + e.message, 'err'); }
}

// Batch-load MANY episodes of a dataset as INDEPENDENT clips (robot demos, no
// cross-episode merge) — the whole reel in one click, no importing over and over.
// Uses light meta+cameras (not the full array bundle) so 20 loads stay snappy; the
// active episode fetches its full bundle via setActive.
async function loadBatch(dsName, eps){
  if (!eps || !eps.length) return toast('nothing to load');
  toast(`loading ${eps.length} episode(s)…`);
  let added = 0, firstEp = null;
  for (const ep of eps){
    try {
      const [meta, cams] = await Promise.all([api.meta(dsName, ep.file), api.cameras(dsName, ep.file)]);
      status.markOpened(dsName, ep.file);
      const T = meta.num_frames || ep.num_frames || 1, fps = meta.fps || ep.fps || 15;
      const cam = (cams.find(c => c.name === 'zed_left') || cams.find(c => c.name === 'body_cam')
                || cams.find(c => c.name === 'body_front_cam') || cams[0])?.name;
      const clip = seq.addClip({ ds: dsName, ep: ep.file, epLabel: ep.task || meta.task || ep.file,
        inFrame: 0, outFrame: T, srcMax: T, fps });
      clip.cam = cam;
      if (!firstEp) firstEp = ep.file;
      added++;
    } catch (e){ console.error('loadBatch skip', ep.file, e); }
  }
  if (firstEp && !active){ await setActive(dsName, firstEp); setGlobal(0); }
  timeline.render(); autosave();
  toast(`loaded ${added} episode(s) — edit each, then Export Episodes`, 'ok');
}

async function setActive(ds, ep){
  if (active && active.ds === ds && active.ep === ep) return active;
  const entry = await store.ensure(ds, ep);
  attachQuality(entry, ds, ep);
  active = { ds, ep, entry };
  videowall.quality = entry.quality;                  // camera-tile health (filled when quality loads)
  bargraph.setEpisode(entry);
  videowall.setAvailable(entry.cams);
  topics.cloudCams = viewport.cloudCams;
  topics.traceTfs = viewport.traceTfs;
  topics.slamCams = viewport.reconMeshCams;
  topics.setEpisode(entry, viewport.layers, videowall.active);
  viewport.setEpisode(entry);
  $('#plotsTitle').textContent = `Joint State · ${entry.meta.task || ep}`;
  return active;
}

// global frame -> (clip, source frame)
function curResolve(){ return seq.resolve(state.frame); }
function curSource(){ const r = curResolve(); return r ? r.sourceFrame : 0; }
function globalFromSource(act, srcFrame){
  // find first clip on the active episode containing srcFrame, else nearest
  let acc = 0;
  for (const c of seq.clips){
    if (c.ds === act.ds && c.ep === act.ep && srcFrame >= c.inFrame && srcFrame < c.outFrame)
      return acc + (srcFrame - c.inFrame);
    acc += c.len;
  }
  return state.frame;
}

// ── the playhead ────────────────────────────────────────────────────────────
function setGlobal(f){
  state.frame = Math.max(0, Math.min(f, Math.max(0, seq.totalFrames - 1)));
  applyFrame();
}
function pause(){ state.playing = false; $('#transport .play').classList.remove('on'); $('#transport .play').textContent = '▶'; }
function play(dir = 1){ if (!seq.totalFrames) return; state.playing = true; state.dir = dir;
  $('#transport .play').classList.add('on'); $('#transport .play').textContent = '⏸'; }
// play/pause toggle: if paused at the end of the timeline, restart from frame 0
function playPause(){
  if (state.playing){ pause(); return; }
  if (seq.totalFrames && state.frame >= seq.totalFrames - 1) setGlobal(0);
  play(1);
}

let _lastApplied = -1, _lastEp = '', _camerasOpen = false;
async function applyFrame(){
  const r = curResolve(); if (!r){ return; }
  const key = r.clip.ds + '/' + r.clip.ep;
  if (key !== _lastEp){ _lastEp = key; await setActive(r.clip.ds, r.clip.ep); }
  if (!active) return;
  const sf = r.sourceFrame;
  // Update camera <img> src FIRST so the browser can start decoding the next frame
  // while we do the (now lightweight) canvas work below — keeps video smooth even
  // with every window open.
  if (_camerasOpen) videowall.update(active.ds, active.ep, sf);
  viewport.update(active.entry, sf);
  bargraph.update(active.entry, sf);
  topics.updateInspector(active.entry, sf);
  panels.update(active.entry, sf);
  updateSubtitle();
  timeline.movePlayhead(state.frame);
  // timecode
  const tot = seq.totalFrames, s = state.frame / seq.fps;
  $('#timecode').textContent =
    `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}:${String(Math.floor((s % 1) * seq.fps)).padStart(2, '0')} · ${state.frame}/${tot}`;
  _lastApplied = state.frame;
}

let _last = performance.now(), _acc = 0;
function loop(t){
  const dt = Math.min(0.1, (t - _last) / 1000); _last = t;
  if (state.playing && seq.totalFrames > 0){
    _acc += dt * seq.fps * state.speed * state.dir;
    const adv = Math.trunc(_acc); _acc -= adv;
    if (adv){
      let nf = state.frame + adv;
      if (nf >= seq.totalFrames){ if (state.loop) nf = 0; else { nf = seq.totalFrames - 1; pause(); } }
      if (nf < 0){ if (state.loop) nf = seq.totalFrames - 1; else { nf = 0; pause(); } }
      state.frame = nf; applyFrame();
    }
  }
  requestAnimationFrame(loop);
}
const STATIC = new URLSearchParams(location.search).has('static');
if (!STATIC) requestAnimationFrame(loop);
else { let n = 0; const t = setInterval(() => { applyFrame(); viewport.renderOnce(); if (++n > 100) clearInterval(t); }, 50); }

// keep the timeline in sync when the sequence changes
bus.on('sequenceChanged', () => { timeline.render(); updateSubtitle(); if (state.frame >= seq.totalFrames) setGlobal(seq.totalFrames - 1); autosave(); scanSoon(); });
bus.on('selection', () => { timeline.render(); const eb = $('#expandBtn'); if (eb){ const on = seq.selected().length > 0; eb.disabled = !on; eb.classList.toggle('lit', on); } });

// ── transport + tools ───────────────────────────────────────────────────────
$('#transport').addEventListener('click', (e) => {
  const b = e.target.closest('.tbtn'); if (!b) return;
  const a = b.dataset.act;
  if (a === 'play') playPause();
  else if (a === 'fwd'){ state.speed = nextSpeed(1); play(1); }
  else if (a === 'back'){ play(-1); }
  else if (a === 'start') setGlobal(0);
  else if (a === 'end') setGlobal(seq.totalFrames - 1);
  else if (a === 'loop'){ state.loop = !state.loop; $('#loopBtn').classList.toggle('on', state.loop); }
});
$('#speedSel').onclick = () => { state.speed = nextSpeed(1); $('#speedSel').textContent = state.speed + '×'; };
function nextSpeed(){ const i = SPEEDS.indexOf(state.speed); const ns = SPEEDS[(i + 1) % SPEEDS.length]; $('#speedSel').textContent = ns + '×'; return ns; }

$('#tlToolbar').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  const tool = b.dataset.tool;
  if (tool === 'annotate') addAnnotation();
  else if (tool === 'reject') addRejectRange();
  else if (tool === 'autoseg') autoSegment();
  else if (tool === 'expand'){ const s = seq.selected(); if (s.length) s.forEach(c => timeline.toggleExpand(c)); }
  else if (tool === 'blade') bladeAction();
  else if (tool === 'del') deleteAction();
  else if (tool === 'merge') mergeAction();
  else if (tool === 'snap'){ const on = b.classList.toggle('on'); timeline.setSnap(on); }
  else if (tool === 'zoomin') timeline.zoom(1.25);
  else if (tool === 'zoomout') timeline.zoom(0.8);
  else if (tool === 'zoomfit') timeline.zoomFit();
});
document.querySelector('.tl-wave').onchange = (e) => timeline.setWave(e.target.value);
$('#barBack').onclick = () => bargraph.clearSelection();
$('#plotMode').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $('#plotMode').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  bargraph.setMode(b.dataset.m);
});

function doUndo(){ if (seq.undo()){ if (state.frame >= seq.totalFrames) setGlobal(Math.max(0, seq.totalFrames - 1)); else applyFrame(); toast('undo'); } else toast('nothing to undo'); }
function doRedo(){ if (seq.redo()){ if (state.frame >= seq.totalFrames) setGlobal(Math.max(0, seq.totalFrames - 1)); else applyFrame(); toast('redo'); } else toast('nothing to redo'); }

// ── TF tree panel (Rerun-style transform hierarchy) ─────────────────────────
function renderTfTree(pane){
  const root = viewport.getTfTree();
  if (!pane) return;
  if (!root){ pane.innerHTML = '<div class="tf-tree"><div class="empty">rig not loaded</div></div>'; return; }
  const row = (n, depth) => {
    const has = n.children.length > 0;
    let html = `<div class="tf-row" data-name="${n.name}" style="padding-left:${depth * 14 + 8}px">
      <span class="tf-twist">${has ? '▾' : '·'}</span>
      <span class="tf-name ${n.joint ? 'jt' : ''}">${n.name}</span></div>`;
    if (has) html += `<div class="tf-children">${n.children.map(c => row(c, depth + 1)).join('')}</div>`;
    return html;
  };
  pane.innerHTML = `<div class="tf-tree">${root.children.map(c => row(c, 0)).join('') || row(root, 0)}</div>`;
  pane.querySelectorAll('.tf-row').forEach(r => {
    r.onclick = (e) => {
      if (e.target.classList.contains('tf-twist')){ const kids = r.nextElementSibling;
        if (kids && kids.classList.contains('tf-children')){ kids.classList.toggle('collapsed'); e.target.textContent = kids.classList.contains('collapsed') ? '▸' : '▾'; return; } }
      pane.querySelectorAll('.tf-row').forEach(x => x.classList.remove('sel'));
      r.classList.add('sel'); viewport.highlightFrame(r.dataset.name);
    };
  });
}

// Show the language label(s) covering the playhead as a subtitle on the 3D view.
function updateSubtitle(){
  const el = $('#vpSubtitle');
  const hits = seq.annotations.filter(a => state.frame >= a.start && state.frame < a.end).map(a => a.text);
  if (hits.length){
    el.textContent = hits.join('  ·  '); el.classList.add('show');
    // sit just below the toggle / cam-control row, however the chips wrapped
    const vp = el.parentElement.getBoundingClientRect();
    const tg = $('#vpToggles').getBoundingClientRect(), cc = $('#vpCamCtl').getBoundingClientRect();
    el.style.top = Math.round(Math.max(tg.bottom, cc.bottom) - vp.top + 8) + 'px';
  } else el.classList.remove('show');
}

function addAnnotation(){
  if (!seq.totalFrames) return toast('add a clip first');
  const dur = Math.round(seq.fps * 2);                 // ~2 s default
  const a = seq.addAnnotation('label', state.frame, Math.min(seq.totalFrames, state.frame + dur));
  // open it for editing on the next render tick
  setTimeout(() => { const el = [...document.querySelectorAll('.anno')].find(e => e.querySelector('.anno-text')?.textContent === 'label');
    if (el) el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })); }, 30);
  toast('annotation added — type the label');
}

function addRejectRange(){
  if (!seq.totalFrames) return toast('add a clip first');
  const end = Math.min(seq.totalFrames, state.frame + Math.round(seq.fps));   // ~1 s default
  seq.addRange(state.frame, end, 'reject');
  toast('reject range — drag edges to size · double-click toggles keep · excised on Save', 'ok');
}

// Turn detected event markers into phase annotations (one per inter-event span).
function autoSegment(){
  if (!seq.totalFrames) return toast('add a clip first');
  const ms = [...timeline.markers].sort((a, b) => a.frame - b.frame);
  if (!ms.length) return toast('no events detected to segment');
  const bounds = [{ frame: 0, label: 'start' }, ...ms.map(m => ({ frame: m.frame, label: m.label })), { frame: seq.totalFrames, label: '' }];
  seq.autoPhases(bounds);
  toast(`auto-segmented into ${bounds.length - 1} phases — rename / retype as needed`, 'ok');
}

// Blade / Delete / Merge route to a selected MODALITY segment if one is selected
// (FCP-style per-modality editing), else to the selected clip.
function bladeAction(){
  const m = timeline.selMod;
  if (m && m.idx != null){ seq.splitModalityAt(m.clip, m.modId, state.frame - m.clip.start); toast('blade ' + m.modId); }
  else {
    const r = seq.resolve(state.frame); const wasExp = r && timeline.expanded.has(r.clip.id);
    const right = seq.splitAt(state.frame);
    if (right && wasExp){ timeline.expanded.add(right.id); timeline.render(); }   // keep both halves expanded
    toast('split');
  }
}
function deleteAction(){
  const m = timeline.selMod;
  if (m){ if (m.idx != null) seq.deleteModalityRange(m.clip, m.modId, m.idx); else toast('already dropped'); timeline.clearSelMod(); toast('dropped ' + (m.modId || '')); }
  else deleteSel();
}
function mergeAction(){
  const m = timeline.selMod;
  if (m){ seq.restoreModality(m.clip, m.modId); timeline.clearSelMod(); toast('merged back ' + m.modId); }
  else mergeSel();
}

function deleteSel(){ const s = seq.selected(); if (!s.length) return toast('select a clip first'); s.forEach(c => seq.remove(c)); toast('deleted ' + s.length + ' clip(s)', 'ok'); }
function mergeSel(){ const s = seq.selected(); if (s.length !== 2) return toast('select exactly 2 adjacent clips');
  // Robot demos are atomic — merging ACROSS episodes is not allowed (only rejoining
  // two halves of the SAME source episode, e.g. after a blade).
  if (s[0].ds !== s[1].ds || s[0].ep !== s[1].ep)
    return toast('merging across demos isn\'t allowed — each episode exports on its own', 'err');
  const r = seq.mergeAdjacent(s[0], s[1]); toast(r.ok ? 'merged' : 'can\'t merge: ' + r.why, r.ok ? 'ok' : 'err'); }

// ── export episodes / lerobot / generate ─────────────────────────────────────
// PER-EPISODE export: every clip is an independent demo (no cross-episode merge)
// → each is materialized on its OWN into <sourceDataset>_processed/<sourceEp>.h5
// carrying only its edits. Idempotent (overwrites), so re-exporting after more
// edits updates in place. Marks each source episode ✓ saved.
$('#saveBtn').onclick = exportEpisodes;
async function exportEpisodes(){
  if (!seq.clips.length) return toast('timeline is empty');
  const task = seq.clips[0]?.epLabel || 'curated';
  const groups = {};
  for (const it of seq.perEpisodeEDLs(task)) (groups[it.sourceDs] ||= []).push(it);
  let totalOut = 0; const folders = [], failed = [];
  // Isolate each source dataset — one bad demo (missing source, etc.) must not
  // abort the whole batch, which matters at 200–500 episodes.
  for (const [ds, list] of Object.entries(groups)){
    const seen = {};
    const payload = list.map(it => {
      const base = String(it.source).replace(/\.h5$/, '');
      const n = (seen[base] = (seen[base] || 0) + 1);            // dedupe if a source was bladed into >1 clip
      return { edl: it.edl, source: it.source, out_file: n === 1 ? base + '.h5' : `${base}_${n}.h5` };
    });
    const outDs = ds + '_processed';
    try {
      toast(`exporting ${payload.length} episode(s) → ${outDs}…`);
      const { job } = await api.exportEpisodes(payload, outDs, task);
      const res = await waitJob(job, (j) => { if (j.total) toast(`exported ${j.done}/${j.total}…`); });
      totalOut += res.result.count; folders.push(outDs);
      list.forEach(it => status.markSaved(ds, it.source));       // each source demo → ✓ done
    } catch (e){ console.error('export group', ds, e); failed.push(ds); }
  }
  timeline.render(); browser.load();
  if (failed.length) toast(`exported ${totalOut} · FAILED: ${failed.join(', ')}`, 'err');
  else toast(`exported ${totalOut} episode(s) → ${folders.join(', ')}`, 'ok');
}

// LeRobot is the FINAL step — aggregate the whole PROCESSED folder into one
// LeRobot v2.1 dataset, only once every episode in the source folder is done.
$('#lerobotBtn').onclick = exportLeRobot;
async function exportLeRobot(){
  if (!seq.clips.length) return toast('timeline is empty');
  const srcDs = [...new Set(seq.clips.map(c => c.ds))];
  if (!(await inspectionGate())) return;
  try {
    for (const ds of srcDs){
      const processed = ds + '_processed';
      toast(`aggregating ${processed} → LeRobot v2.1…`);
      const { job } = await api.exportLeRobotDataset(processed, `mabel/${processed}`);
      const res = await waitJob(job, (j) => { if (j.total) toast(`LeRobot ${j.done}/${j.total} episodes…`); });
      const lr = res.result;
      toast(`LeRobot ✓ ${lr.episodes} eps · ${lr.frames}f · ${lr.cameras.length} cams → ${lr.path.split('/').slice(-1)[0]}`, 'ok');
    }
  } catch (e){
    const msg = /no episodes/.test(e.message || '') ? 'run "Export Episodes" first — the processed folder is empty' : e.message;
    toast('LeRobot export failed: ' + msg, 'err');
  }
}

// Warn if the source dataset(s) behind this export aren't fully inspected (every
// episode saved/curated). Returns true to proceed, false to go back and process.
async function inspectionGate(){
  let cat; try { cat = await api.catalog(); } catch { return true; }   // offline → don't block
  const byName = Object.fromEntries((cat.datasets || []).map(d => [d.name, d]));
  const srcDs = [...new Set(seq.clips.map(c => c.ds))];
  const unfinished = [];
  for (const name of srcDs){
    const d = byName[name]; if (!d) continue;
    const p = status.datasetProgress(name, d.episodes);
    if (p.saved < p.total) unfinished.push({ name, ...p });
  }
  if (!unfinished.length) return true;
  const rows = unfinished.map(u =>
    `<li><b>${u.name}</b> — only <b>${u.saved}/${u.total}</b> episodes saved` +
    (u.opened ? ` (${u.opened} opened, not saved)` : ` (${u.total - u.saved} never inspected)`) + `</li>`).join('');
  return confirmModal({
    title: '⚠ Dataset not fully inspected',
    body: `Some source data hasn't been reviewed & saved before export:<ul class="modal-list">${rows}</ul>
           Exporting now bakes un-inspected episodes into the LeRobot dataset.`,
    okLabel: 'Export anyway', cancelLabel: 'Go back to process', danger: true,
  });
}

$('#genBtn').onclick = async () => {
  const n = parseInt(prompt('How many sample episodes to generate?', '2') || '0', 10);
  if (!n) return;
  const frames = parseInt(prompt('Frames per episode (7 cams render ~2 Hz):', '120') || '120', 10);
  try {
    toast('generating ' + n + ' episodes… (this takes a while)');
    const { job } = await api.generate({ name: 'studio_sample', episodes: n, frames });
    await waitJob(job, (j) => { if (j.total) toast(`rendering ${j.done}/${j.total} frames…`); });
    toast('generated ' + n + ' episodes', 'ok'); browser.load();
  } catch (e){ toast('generate failed: ' + e.message, 'err'); }
};

document.addEventListener('drop-episode', (e) => addEpisode({ ds: e.detail.ds, ep: e.detail.ep }));
document.addEventListener('cam-untoggle', (e) => { topics.reflectCamera(e.detail, false); viewport.setActiveCams(videowall.active); });

// the "Cameras" button (bottom-center of the 3D view) opens the camera panel
$('#camRestore').onclick = () => launchCameras();

// ── 3D viewer controls: what to visualize on THIS view, in logical groups ────
// (_vpMenus / _closeVpMenus are declared up by the buildToggles() call site so
// they're initialized before it runs.)
function buildToggles(){
  const host = $('#vpToggles');
  // grouped layer chips (plain on/off). Names chosen to be unambiguous.
  const GROUPS = [
    ['Scene',  [['robot', 'Robot'], ['tf', 'TF Frames'], ['grid', 'Grid']]],
    ['__cloud__'],                                   // Point Cloud dropdown
    ['__mesh__'],                                    // cuSLAM Mesh dropdown
    ['Maps',   [['map', '3D Map'], ['occupancy', 'Occupancy'], ['lidar', 'LiDAR']]],
    ['Camera', [['frustum', 'Camera Frustum']]],
    ['Motion', [['target', 'Nav Target'], ['ghost', 'EE Targets'], ['distribution', 'Dataset Overlay']]],
    ['__traj__'],                                    // Trajectory dropdown (pick TF + style)
  ];
  let first = true;
  for (const g of GROUPS){
    if (!first){ const s = document.createElement('span'); s.className = 'vp-sep'; host.appendChild(s); }
    first = false;
    if (g[0] === '__cloud__'){ buildSourceDropdown(host, { label: 'Point Cloud', key: 'cloud' }); continue; }
    if (g[0] === '__mesh__'){ buildSourceDropdown(host, { label: 'cuSLAM Mesh', key: 'mesh' }); continue; }
    if (g[0] === '__traj__'){ buildTrajectoryDropdown(host); continue; }
    for (const [key, label] of g[1]){
      const b = document.createElement('button'); b.className = 'vp-chip' + (viewport.layers[key] ? ' on' : '');
      b.dataset.layer = key; b.textContent = label;
      b.onclick = () => setLayer(key, !viewport.layers[key]);
      host.appendChild(b);
    }
  }
  buildCamCtl();
  buildEdgeLaunchers();
  document.addEventListener('click', _closeVpMenus);
}

// A viewer button that opens a dropdown (like "+ Window") to pick which cameras
// publish this layer — point clouds, or cuSLAM meshes ("which SLAM cam"). The
// button reads ON when any source is active. Cloud menu also carries density.
function buildSourceDropdown(host, { label, key }){
  const btn = document.createElement('button'); btn.className = 'vp-chip vp-dd';
  const menu = document.createElement('div'); menu.className = 'vp-menu hidden';
  host.appendChild(btn); document.body.appendChild(menu); _vpMenus.push(menu);   // menu on <body> so panels can't cover it
  const set = key === 'cloud' ? viewport.cloudCams : viewport.reconMeshCams;
  const on = (name) => set.has(name);
  const toggle = (name, v) => {
    if (key === 'cloud'){ viewport.setCloudCam(name, v); if (active && !state.playing) applyFrame(); }
    else viewport.setReconMeshCam(name, v);
    const tcb = document.querySelector(`input[data-${key === 'cloud' ? 'cloud' : 'slam'}="${CSS.escape(name)}"]`);
    if (tcb) tcb.checked = v;                          // keep the Topics checkbox in sync
  };
  const refresh = () => { btn.classList.toggle('on', set.size > 0); btn.textContent = ''; btn.append(label + ' ', chevron()); };
  const build = () => {
    const cams = (active?.entry?.cams || []).filter(c => c.depth);
    let html = `<div class="awm-sec">${label} · cameras</div>`;
    html += cams.length
      ? cams.map(c => `<button class="vp-mi ${on(c.name) ? 'on' : ''}" data-src="${c.name}"><span class="vmi-dot"></span>${c.name}</button>`).join('')
      : `<div class="awm-hint">no depth cameras in this episode</div>`;
    menu.innerHTML = html;
    menu.querySelectorAll('.vp-mi[data-src]').forEach(b => b.onclick = (e) => {
      e.stopPropagation(); const name = b.dataset.src, v = !on(name);
      toggle(name, v); b.classList.toggle('on', v); refresh();
    });
  };
  btn.onclick = (e) => {
    e.stopPropagation(); const show = menu.classList.contains('hidden'); _closeVpMenus();
    if (show){ build(); _openVpMenu(btn, menu); }
  };
  refresh();
  return btn;
}
function chevron(){ const s = document.createElement('span'); s.className = 'vp-cv'; s.textContent = '▾'; return s; }

// Trajectory dropdown: pick which frame(s) to trace + the drawing style. Reads ON
// when a trace is active with a visible style.
function buildTrajectoryDropdown(host){
  const btn = document.createElement('button'); btn.className = 'vp-chip vp-dd';
  const menu = document.createElement('div'); menu.className = 'vp-menu hidden';
  host.appendChild(btn); document.body.appendChild(menu); _vpMenus.push(menu);   // menu on <body> so panels can't cover it
  const TFS = [['left_palm', 'L palm (EE)'], ['right_palm', 'R palm (EE)'], ['chassis', 'chassis'], ['head', 'head']];
  const active_ = () => viewport.traceTfs.size > 0 && (viewport.layers.trail || viewport.layers.clippath) || viewport.layers.path;
  const refresh = () => { btn.classList.toggle('on', active_()); btn.textContent = ''; btn.append('Trajectory ', chevron()); };
  const build = () => {
    const has = (tf) => active?.entry?.bundle?.topics?.['pose/' + tf];
    const avail = TFS.filter(([tf]) => has(tf));
    let html = `<div class="awm-sec">Trajectory · frames</div>`;
    html += avail.length
      ? avail.map(([tf, lbl]) => `<button class="vp-mi ${viewport.traceTfs.has(tf) ? 'on' : ''}" data-tf="${tf}"><span class="vmi-dot"></span>${lbl}</button>`).join('')
      : `<div class="awm-hint">no pose data in this episode</div>`;
    html += `<div class="awm-sec">style</div>`
      + `<button class="vp-mi ${viewport.layers.clippath ? 'on' : ''}" data-style="clippath"><span class="vmi-dot"></span>Whole-clip path</button>`
      + `<button class="vp-mi ${viewport.layers.trail ? 'on' : ''}" data-style="trail"><span class="vmi-dot"></span>Fading trail</button>`
      + `<button class="vp-mi ${viewport.layers.path ? 'on' : ''}" data-style="path"><span class="vmi-dot"></span>Base path (odom)</button>`;
    menu.innerHTML = html;
    menu.querySelectorAll('[data-tf]').forEach(b => b.onclick = (e) => {
      e.stopPropagation(); const tf = b.dataset.tf, v = !viewport.traceTfs.has(tf);
      viewport.setTraceTf(tf, v); b.classList.toggle('on', v);
      const tcb = document.querySelector(`input[data-trace="${tf}"]`); if (tcb) tcb.checked = v;
      if (active && !state.playing) applyFrame(); refresh();
    });
    menu.querySelectorAll('[data-style]').forEach(b => b.onclick = (e) => {
      e.stopPropagation(); const k = b.dataset.style, v = !viewport.layers[k];
      setLayer(k, v); b.classList.toggle('on', v);
      if (active && !state.playing) applyFrame(); refresh();
    });
  };
  btn.onclick = (e) => {
    e.stopPropagation(); const show = menu.classList.contains('hidden'); _closeVpMenus();
    if (show){ build(); _openVpMenu(btn, menu); }
  };
  refresh();
}

// Open (or focus) a docked window of `type`; left-side launchers snap to the
// narrow left dock, right-side to the right — the 3D viewer stays visible.
function launchPanel(type, side){
  const ex = panels.panels.find(p => p.type === type);
  if (ex){ panels._raise(ex); return ex; }
  const p = panels.add(type);
  if (p) panels.dockToSide(p, side === 'left' ? 'left' : 'right');
  return p;
}
// Always-on edge tabs: Joints on the right; 2D Map + IMU on the left — mirrors
// the "Cameras" restore chip so every common view is one click away.
function buildEdgeLaunchers(){
  const mk = (host, label, type, side) => {
    const b = document.createElement('button'); b.textContent = label;
    b.title = `Open ${label} window`; b.onclick = () => launchPanel(type, side);
    $(host).appendChild(b);
  };
  mk('#vpEdgeR', 'Joints', 'jointCompare', 'right');
  mk('#vpEdgeL', '2D Map', 'map2d', 'left');
  mk('#vpEdgeL', 'IMU', 'imu', 'left');
}

// Camera-frustum controls at the TOP-RIGHT of the 3D viewer, with the point-cloud
// density (intensity) toggle unified right below them.
function buildCamCtl(){
  const el = $('#vpCamCtl');
  el.innerHTML = `
    <div class="cc-row">
      <span class="cc-lbl">camera frustum</span>
      <input type="range" class="cc-size" min="0.05" max="1.5" step="0.05" value="${viewport.camProjD}" title="projection size">
      <span class="cc-val">${viewport.camProjD.toFixed(2)}m</span>
      <button class="cc-dist ${viewport.camFlat ? '' : 'on'}" title="lens distortion on/off">distortion</button>
    </div>
    <div class="cc-row">
      <span class="cc-lbl">cloud density</span>
      <div class="vp-mdens" id="ccDensity">`
        + ['ultra', 'high', 'med', 'low'].map(l => `<button data-dens="${l}" class="${viewport.cloudDensity === l ? 'on' : ''}">${l}</button>`).join('')
      + `</div>
    </div>`;
  el.querySelector('.cc-size').oninput = (e) => { const d = parseFloat(e.target.value);
    el.querySelector('.cc-val').textContent = d.toFixed(2) + 'm'; viewport.setCamProjDistance(d); setLayer('frustum', true); };
  el.querySelector('.cc-dist').onclick = (e) => { const on = e.target.classList.toggle('on');
    viewport.setCamFlat(!on); setLayer('frustum', true); };
  el.querySelectorAll('#ccDensity button').forEach(b => b.onclick = () => {
    viewport.setCloudDensity(b.dataset.dens);
    el.querySelectorAll('#ccDensity button').forEach(x => x.classList.toggle('on', x.dataset.dens === b.dataset.dens));
  });
}

// ── keyboard (NLE conventions) ──────────────────────────────────────────────
window.addEventListener('keydown', (e) => {
  if (/input|textarea/i.test(e.target.tagName)) return;
  const k = e.key;
  if (k === ' '){ e.preventDefault(); playPause(); }
  else if (k === 'l' || k === 'L'){ state.speed = state.playing && state.dir > 0 ? nextSpeed() : state.speed; play(1); }
  else if (k === 'j' || k === 'J'){ play(-1); }
  else if (k === 'k' || k === 'K'){ pause(); }
  else if (k === 'ArrowRight'){ e.preventDefault(); pause(); setGlobal(state.frame + (e.shiftKey ? edgeStep(1) : 1)); }
  else if (k === 'ArrowLeft'){ e.preventDefault(); pause(); setGlobal(state.frame + (e.shiftKey ? edgeStep(-1) : -1)); }
  else if (k === 'Home'){ setGlobal(0); }
  else if (k === 'End'){ setGlobal(seq.totalFrames - 1); }
  else if (k === 'b' || k === 'B'){ bladeAction(); }
  else if (k === 't' || k === 'T'){ addAnnotation(); }
  else if (k === 'x' || k === 'X'){ addRejectRange(); }
  else if (k === ','){ jumpMarker(-1); }
  else if (k === '.'){ jumpMarker(1); }
  else if (k === 'e' || k === 'E'){ const s = seq.selected(); if (s.length) s.forEach(c => timeline.toggleExpand(c)); else { const c = seq.clipAt(state.frame); if (c) timeline.toggleExpand(c); } }
  else if (k === 'Delete' || k === 'Backspace'){ deleteAction(); }
  else if ((e.metaKey || e.ctrlKey) && (k === 'j' || k === 'J')){ e.preventDefault(); mergeAction(); }
  else if ((e.metaKey || e.ctrlKey) && (k === 's' || k === 'S')){ e.preventDefault(); exportEpisodes(); }
  else if ((e.metaKey || e.ctrlKey) && (k === 'z' || k === 'Z') && e.shiftKey){ e.preventDefault(); doRedo(); }
  else if ((e.metaKey || e.ctrlKey) && (k === 'y' || k === 'Y')){ e.preventDefault(); doRedo(); }
  else if ((e.metaKey || e.ctrlKey) && (k === 'z' || k === 'Z')){ e.preventDefault(); doUndo(); }
  else if (k === 's' || k === 'S'){ const btn = document.querySelector('[data-tool="snap"]'); btn.click(); }
  else if (k === '='||k === '+'){ timeline.zoom(1.25); }
  else if (k === '-'){ timeline.zoom(0.8); }
  else if ((k === 'z' || k === 'Z') && e.shiftKey){ timeline.zoomFit(); }
  else if (k === 'r' || k === 'R'){ browser.load(); toast('library reloaded'); }
  else if (k === '?' || (k === '/' && e.shiftKey)){ toggleHelp(); }
  else if (k === 'f' || k === 'F'){ viewport.frameRobot(); }
  else if (k === 'Escape'){ if (helpOpen) toggleHelp(); else { seq.clearSelection(); bargraph.clearSelection(); } }
  else if (k === 'a' || k === 'A'){ if (!(e.metaKey || e.ctrlKey)) return; e.preventDefault(); seq.clips.forEach(c => c.selected = true); bus.emit('selection', seq.selected()); }
});
function edgeStep(dir){
  let acc = 0; const edges = [0];
  for (const c of seq.clips){ acc += c.len; edges.push(acc); }
  if (dir > 0){ for (const e of edges) if (e > state.frame) return e - state.frame; }
  else { for (let i = edges.length - 1; i >= 0; i--) if (edges[i] < state.frame) return edges[i] - state.frame; }
  return dir;
}

// ── misc ────────────────────────────────────────────────────────────────────
$('.dock-tabs').addEventListener('click', (e) => {
  const b = e.target.closest('.dock-tab'); if (!b) return;
  document.querySelectorAll('.dock-tab').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.dock-pane').forEach(p => p.classList.toggle('on', p.dataset.pane === b.dataset.tab));
});

// ── keyboard help overlay ───────────────────────────────────────────────────
let helpOpen = false, _helpEl = null;
const SHORTCUTS = [
  ['Space', 'play / pause'], ['J / K / L', 'reverse / stop / forward (tap L to ramp)'],
  ['← / →', 'step 1 frame'], ['⇧← / ⇧→', 'jump to clip edge'], ['Home / End', 'start / end'],
  ['B', 'blade — split at playhead'], ['T', 'add annotation'], ['E', 'expand clip modalities'],
  ['⌫ / Delete', 'ripple-delete clip'], ['⌘J', 'merge adjacent clips'],
  ['⌘Z / ⌘⇧Z', 'undo / redo'],
  ['S', 'toggle snapping'], ['+ / −', 'zoom timeline'], ['⇧Z', 'zoom to fit'], ['⌘A', 'select all'],
  ['F', 'frame the robot'], ['R', 'reload library'], ['⌘S', 'save episode'], ['Esc', 'deselect / close'],
  ['?', 'this help'],
];
function toggleHelp(){
  helpOpen = !helpOpen;
  if (!_helpEl){
    _helpEl = document.createElement('div'); _helpEl.className = 'help-overlay';
    _helpEl.innerHTML = `<div class="help-card"><h3>Keyboard</h3><div class="help-grid">${
      SHORTCUTS.map(([k, d]) => `<kbd>${k}</kbd><span>${d}</span>`).join('')}</div>
      <div class="help-foot">click outside to close</div></div>`;
    _helpEl.onclick = (e) => { if (e.target === _helpEl) toggleHelp(); };
    document.body.appendChild(_helpEl);
  }
  _helpEl.classList.toggle('show', helpOpen);
}

let _saveT;
function autosave(){ clearTimeout(_saveT); _saveT = setTimeout(() => {
  if (seq.clips.length) api.putProject('autosave', seq.toEDL('autosave')).catch(() => {}); }, 1200); }

async function checkHealth(){
  try { await api.health(); $('#conn').classList.add('ok'); $('#conn').title = 'server connected'; }
  catch { $('#conn').classList.add('bad'); }
}

function toast(msg, kind = ''){
  const t = $('#toast'); t.textContent = msg; t.className = 'toast show ' + kind;
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('show'), 2600);
}

// Apple-style two-choice modal → resolves true (primary) / false (secondary).
// Used for the "dataset not fully inspected" export gate.
function confirmModal({ title, body, okLabel = 'Proceed', cancelLabel = 'Go back', danger = false }){
  return new Promise((resolve) => {
    const back = document.createElement('div'); back.className = 'modal-back';
    back.innerHTML = `<div class="modal">
      <div class="modal-title">${title}</div>
      <div class="modal-body">${body}</div>
      <div class="modal-btns">
        <button class="mbtn cancel">${cancelLabel}</button>
        <button class="mbtn ok${danger ? ' danger' : ''}">${okLabel}</button>
      </div></div>`;
    document.body.appendChild(back);
    const done = (v) => { back.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
    back.querySelector('.cancel').onclick = () => done(false);
    back.querySelector('.ok').onclick = () => done(true);
    back.onclick = (e) => { if (e.target === back) done(false); };
    const onKey = (e) => { if (e.key === 'Escape') done(false); else if (e.key === 'Enter') done(true); };
    document.addEventListener('keydown', onKey);
    back.querySelector('.ok').focus();
  });
}

// Debug/test hook (used by the CDP smoke test; harmless in normal use).
window.__studio = {
  seq, state, setGlobal, addEpisode, loadBatch, exportEpisodes, exportLeRobot,
  timeline, panels, viewport, status, confirmModal,
  blade: () => bladeAction(), del: () => deleteAction(), merge: () => mergeAction(),
  undo: () => doUndo(), redo: () => doRedo(),
  saveAs: async (ds, task) => { const { job } = await api.save({ edl: seq.toEDL(task), out_dataset: ds, task });
    return (await waitJob(job)).result; },
};

window.addEventListener('error', (e) => console.error('WINDOW_ERROR', e.message, e.filename + ':' + e.lineno));
window.addEventListener('unhandledrejection', (e) => console.error('UNHANDLED_REJECTION', e.reason?.message || e.reason));
window.__studioReady = () => 'STUDIO_READY clips=' + seq.clips.length + ' frames=' + seq.totalFrames +
  ' rigLoaded=' + !!viewport.rig + ' active=' + (active ? active.ep : 'none') +
  ' topicRows=' + document.querySelectorAll('.topic-row').length +
  ' clipEls=' + document.querySelectorAll('.clip').length;

timeline.render();
