// rviz/rerun-style topic tree built from the active episode's schema. Joint
// groups, cameras (toggle video tiles), lidar/map/TF/ghost layers (toggle 3D
// layers). Also an Inspector pane with the live value at the playhead.
import { GROUP_ORDER, GROUP_LABEL } from './names.js';
import { api } from './api.js';

export class Topics {
  constructor(paneEl, inspectorEl, { onCamera, onLayer, onCloudCam, onTraceTf, onSlamCam }){
    this.pane = paneEl; this.inspector = inspectorEl;
    this.onCamera = onCamera; this.onLayer = onLayer; this.onCloudCam = onCloudCam;
    this.onTraceTf = onTraceTf; this.onSlamCam = onSlamCam;
    this.traceTf = 'left_palm'; this.entry = null;
  }

  setEpisode(entry, layerState, activeCams){
    this.entry = entry;
    const groups = entry.meta.actuator_groups || {};
    const cams = entry.cams || [];
    const depthCams = cams.filter(c => c.depth);
    const has = (t) => !!entry.bundle.topics[t];
    const cloudOn = this.cloudCams || new Set();
    const traceOn = this.traceTfs || new Set();
    const mountTag = (n) => n.includes('zed') ? 'head' : n.includes('wrist') ? 'wrist'
      : (n.includes('body_front') || n.includes('body_back')) ? 'base' : n.includes('body_cam') ? 'torso' : 'cam';

    const grp = (title, rowsHtml, open = false, tag = '') =>
      `<details class="topic-grp" ${open ? 'open' : ''}><summary>▸ ${title}${tag ? `<span class="grp-tag">${tag}</span>` : ''}</summary>${rowsHtml}</details>`;
    const layerRow = (key, label, on, tag = '') =>
      `<label class="topic-row"><input type="checkbox" data-layer="${key}" ${on ? 'checked' : ''}><span>${label}</span>${tag ? `<span class="tv">${tag}</span>` : ''}</label>`;
    const multiRow = (attr, val, label, on, tag = '') =>
      `<label class="topic-row"><input type="checkbox" data-${attr}="${val}" ${on ? 'checked' : ''}><span>${label}</span>${tag ? `<span class="tv">${tag}</span>` : ''}</label>`;

    // ── Robots (robot_description / URDF) — mabel; multi-robot ready ────────
    const robots = layerRow('robot', 'mabel', layerState.robot, 'urdf');

    // ── Transforms (tf2) ──────────────────────────────────────────────────
    const transforms = layerRow('tf', 'TF frames + labels', layerState.tf);

    // ── Image · camera streams (sensor_msgs/Image) ────────────────────────
    const camRows = cams.map(c =>
      multiRow('cam', c.name, c.name, activeCams.has(c.name), mountTag(c.name))).join('');

    // ── PointCloud · cameras (sensor_msgs/PointCloud2) ────────────────────
    // Each depth camera can publish a point cloud — toggle which ones render.
    // (Density lives on the 3D viewer, not here.)
    const cloud3d = depthCams.length
      ? depthCams.map(c => multiRow('cloud', c.name, c.name, cloudOn.has(c.name), mountTag(c.name))).join('')
      : `<div class="topic-note">no depth cameras in this episode</div>`;

    // ── cuSLAM Mesh · per camera (fused surface) ──────────────────────────
    // Pick the SLAM cam(s): each reconstructs its own world-frame mesh.
    const slamOn = this.slamCams || new Set();
    const slam = depthCams.length
      ? depthCams.map(c => multiRow('slam', c.name, c.name, slamOn.has(c.name), mountTag(c.name))).join('')
      : `<div class="topic-note">no depth cameras (SLAM needs depth)</div>`;

    // ── 2D LiDAR (sensor_msgs/LaserScan) ──────────────────────────────────
    const lidar = layerRow('lidar', '2D LiDAR scan', layerState.lidar, has('lidar') ? `${entry.bundle.topics.lidar.dim} pts` : '')
                + layerRow('accumulate', 'Accumulate → cloud', layerState.accumulate);

    // ── Maps ──────────────────────────────────────────────────────────────
    const maps = layerRow('map', '3D map (lidar mesh)', layerState.map)
               + layerRow('occupancy', '2D occupancy map', layerState.occupancy);

    // ── Trajectories · TF (multi-select frames + representation) ───────────
    const TF_LBL = { left_palm: 'L palm (EE)', right_palm: 'R palm (EE)', chassis: 'chassis (odom)', head: 'head' };
    let traj = '';
    for (const tf of ['left_palm', 'right_palm', 'chassis', 'head'])
      if (has('pose/' + tf)) traj += multiRow('trace', tf, TF_LBL[tf] || tf, traceOn.has(tf));
    traj += `<div class="topic-sub">representation</div>`
          + layerRow('trail', 'Trail (fading)', layerState.trail)
          + layerRow('clippath', 'Whole-clip path', layerState.clippath)
          + layerRow('distribution', 'Dataset overlay', layerState.distribution, 'distrib')
          + `<div id="trajLegend" class="traj-legend"></div>`;

    // ── Navigation ────────────────────────────────────────────────────────
    const nav = layerRow('path', 'Base path (odom)', layerState.path)
              + layerRow('target', 'Look-at / nav target', layerState.target, has('teleop/look_at') ? '' : 'n/a')
              + layerRow('ghost', 'EE target ghosts', layerState.ghost);

    // ── Joints (info) ─────────────────────────────────────────────────────
    let joints = '';
    for (const g of GROUP_ORDER) if (groups[g]) joints += `<label class="topic-row"><input type="checkbox" checked disabled><span>${GROUP_LABEL[g]} <span style="color:var(--ash-dim)">(${groups[g].length})</span></span></label>`;

    this.pane.innerHTML =
      grp('Robots', robots, true, 'urdf') +
      grp('Image', camRows, true, `Image · ${cams.length}`) +
      grp('PointCloud', cloud3d, false, `PointCloud2 · ${depthCams.length}`) +
      grp('cuSLAM Mesh', slam, false, `Mesh · ${depthCams.length}`) +
      grp('LaserScan', lidar, true, 'LaserScan') +
      grp('Maps', maps, false, 'OccupancyGrid') +
      grp('Transforms', transforms, false, 'tf2') +
      grp('Trajectories · TF', traj, true) +
      grp('Navigation', nav, false) +
      grp('JointState', joints, false, `${entry.meta.actuator_names?.length || 0}`) +
      grp('Camera Calibration', `<div id="calibBody"><div class="topic-row" style="color:var(--ash-dim)">loading…</div></div>`, false);

    this.pane.querySelectorAll('input[data-layer]').forEach(cb =>
      cb.onchange = () => this.onLayer(cb.dataset.layer, cb.checked));
    this.pane.querySelectorAll('input[data-cam]').forEach(cb =>
      cb.onchange = () => this.onCamera(cb.dataset.cam, cb.checked));
    this.pane.querySelectorAll('input[data-cloud]').forEach(cb =>
      cb.onchange = () => this.onCloudCam?.(cb.dataset.cloud, cb.checked));
    this.pane.querySelectorAll('input[data-trace]').forEach(cb =>
      cb.onchange = () => this.onTraceTf?.(cb.dataset.trace, cb.checked));
    this.pane.querySelectorAll('input[data-slam]').forEach(cb =>
      cb.onchange = () => this.onSlamCam?.(cb.dataset.slam, cb.checked));

    // lazy-load intrinsics/extrinsics summary (fov, focal length, mount, distortion)
    api.calibration(entry.ds, entry.ep).then(cal => {
      const body = this.pane.querySelector('#calibBody'); if (!body) return;
      body.innerHTML = cal.map(c => `<div class="calib-row">
        <div class="calib-name">${c.name}</div>
        <div class="calib-kv">FOV ${c.fovx_deg}×${c.fovy_deg}° · f ${c.fx}px · ${c.w}×${c.h}<br>
        mount: ${c.mount_body} · dist: ${c.distortion}</div></div>`).join('');
    }).catch(() => {});
  }

  // distribution legend (called by viewport with the color key)
  setTrajLegend(legend){
    const el = this.pane.querySelector('#trajLegend'); if (!el) return;
    el.innerHTML = legend.map(l => `<div class="leg-row"><span class="sw" style="background:${l.color}"></span>
      <span class="leg-name">${l.label}</span></div>`).join('');
  }

  // sync a 3D-scene layer checkbox toggled from the viewport chips
  reflectLayer(key, on){
    const cb = this.pane.querySelector(`input[data-layer="${key}"]`);
    if (cb) cb.checked = on;
  }

  // reflect a camera toggle that happened elsewhere (tile close button)
  reflectCamera(name, on){
    const cb = this.pane.querySelector(`input[data-cam="${CSS.escape(name)}"]`);
    if (cb) cb.checked = on;
  }

  updateInspector(entry, f){
    if (!entry){ this.inspector.innerHTML = ''; return; }
    const m = entry.meta || {}, sc = m.schema || {};
    const src = sc.source || 'unknown';
    const device = src === 'sim' ? 'MuJoCo simulation' : src === 'real' ? 'MABEL hardware' : src;
    const collection = src === 'sim' ? 'Vision Pro teleop → mock AVP' : 'Vision Pro teleop';
    const [date = '—', time = ''] = String(m.created || '').split(' ');
    const cams = m.cameras || [], depthN = cams.filter(c => c.depth).length;
    const lidar = (m.lidar && m.lidar.n_rays) || entry.bundle.topics.lidar?.dim || 0;
    const dur = (m.t1 != null && m.t0 != null) ? (m.t1 - m.t0).toFixed(2) : (entry.T / entry.fps).toFixed(2);
    const dof = sc.nq != null ? `${sc.nq} / ${sc.nv} / ${sc.nu}` : '—';

    const fv = (t) => entry.bundle.topics[t]?.data.subarray(f * entry.bundle.topics[t].dim, f * entry.bundle.topics[t].dim + entry.bundle.topics[t].dim);
    const bt = fv('base_twist'), gr = fv('teleop/grasp'), md = fv('teleop/mode');

    const clip = [
      ['dataset', entry.ds], ['episode', entry.ep], ['task', m.task || '—'],
      ['date', date], ['time', time],
      ['device', device], ['collection', collection], ['source', src],
      ['fps', (entry.fps || 0).toFixed(2)], ['frames', entry.T],
      ['duration', dur + ' s'], ['DOF q/v/u', dof],
      ['cameras', `${cams.length} (${depthN} +depth)`], ['lidar', lidar + ' rays'],
    ];
    const live = [['frame', f], ['t (s)', (fv('time_sim')?.[0] ?? f / entry.fps).toFixed(2)]];
    if (bt) live.push(['base v', `${bt[0].toFixed(2)},${bt[1].toFixed(2)} ω${bt[2].toFixed(2)}`]);
    if (gr) live.push(['grasp L/R', `${gr[0].toFixed(2)} / ${gr[1].toFixed(2)}`]);
    if (md) live.push(['mode', md[0] < 0.5 ? 'MANIP' : 'NAV']);

    const dl = (rows) => `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`;
    this.inspector.innerHTML = `<div class="inspector">
      <div class="info-h">Clip</div>${dl(clip)}
      <div class="info-h">Playhead</div>${dl(live)}</div>`;
  }
}
