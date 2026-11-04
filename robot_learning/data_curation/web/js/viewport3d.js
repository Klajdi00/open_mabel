// The rviz/Rerun-style 3D scene: the MABEL rig (posed from joint angles) inside
// the map it reconstructs from its own sensors, plus standard robot/ROS layers:
// TF frames, 2D LiDAR scan, 3D camera point cloud, 3D lidar map, 2D occupancy
// map, the base trajectory, the look-at/nav target, and EE ghosts.
//
// Stage / Rig / Q_ZUP / setJoint are vendored from website/assets/teleop-app.js
// (not exported there, so per the repo rule we copy, not import).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { frameVec } from './store.js';
import { api } from './api.js';

const RIG_GLB = '/website-assets/mabel_rig.glb';
const JOINTS_JSON = '/website-assets/mabel_joints.json';
const Q_ZUP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const LIDAR_Z = 0.25;

// Point-cloud density presets (user-selectable sparsity). `stride`/`maxPoints`
// control the per-camera depth clouds; `reconPs`/`reconVoxel` the fused SLAM
// cloud; `pointSize` the splat size (denser → smaller). Local mode defaults to
// ULTRA (full-res, prefetched); the deployed site defaults to MED (bandwidth).
const CLOUD_DENSITY = {
  ultra: { stride: 1, maxPoints: 400000, pointSize: 0.006, reconPs: 1, reconVoxel: 0.006 },
  high:  { stride: 2, maxPoints: 200000, pointSize: 0.010, reconPs: 2, reconVoxel: 0.012 },
  med:   { stride: 4, maxPoints: 80000,  pointSize: 0.016, reconPs: 3, reconVoxel: 0.020 },
  low:   { stride: 8, maxPoints: 30000,  pointSize: 0.024, reconPs: 5, reconVoxel: 0.040 },
};

// TF frames to show — attached to the ACTUAL rig nodes so they match the robot
// parts exactly (label, GLB node name, triad size).
const TF_FRAMES = [
  ['base_link', 'base_link', 0.16], ['lift_top', 'lift_top', 0.12], ['torso', 'torso', 0.13],
  ['head', 'head', 0.12],
  ['l_shoulder', 'l_shoulder_1', 0.1], ['l_elbow', 'l_elbow', 0.09], ['l_wrist', 'l_wrist', 0.08], ['l_palm', 'left_palm', 0.07],
  ['r_shoulder', 'r_shoulder_1', 0.1], ['r_elbow', 'r_elbow', 0.09], ['r_wrist', 'r_wrist', 0.08], ['r_palm', 'right_palm', 0.07],
];

function turbo(t){
  t = Math.max(0, Math.min(1, t));
  const r = 34.6 + t * (1172 - t * (10794 - t * (33300 - t * (38394 - t * 14825))));
  const g = 23.3 + t * (557 + t * (1225 - t * (3575 - t * (1074 + t * 708))));
  const b = 27.2 + t * (3211 - t * (15328 - t * (27814 - t * (22569 - t * 6839))));
  return [Math.max(0, Math.min(255, r)) / 255, Math.max(0, Math.min(255, g)) / 255, Math.max(0, Math.min(255, b)) / 255];
}

// Invert Brown-Conrady distortion: image (distorted normalized) -> true ray
// normalized. Zero coeffs (ideal pinhole) -> identity. Iterative.
function undistort(xd, yd, k){
  if (!k || (k[0] === 0 && k[1] === 0 && k[2] === 0 && k[3] === 0 && (k[4] || 0) === 0)) return [xd, yd];
  let x = xd, y = yd;
  for (let it = 0; it < 8; it++){
    const r2 = x * x + y * y;
    const rad = 1 + k[0] * r2 + k[1] * r2 * r2 + (k[4] || 0) * r2 * r2 * r2;
    const dx = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
    const dy = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
    x = (xd - dx) / rad; y = (yd - dy) / rad;
  }
  return [x, y];
}
// Undistorted normalized image coords for pixel (u,v).
function pixelUndist(c, u, v){
  const xd = (u - c.cx) / c.fx, yd = (v - c.cy) / c.fy;
  return undistort(xd, yd, c.dist);
}
// Back-project an image pixel (u,v) to a point at distance D in the camera frame.
// flat=false → on the true ray at range D (curves for wide-FOV/distorted lenses).
// flat=true  → on the rectilinear plane z=−D at the undistorted position (the
//              rectified, "undistorted & flat" image, like a pinhole sensor).
function pixelPoint(c, u, v, D, flat){
  const [xn, yn] = pixelUndist(c, u, v);
  if (flat){
    // rectified plane at the focal distance; scale by cos(½fov) so the flat image
    // is the SAME size as the curved ray-cap (otherwise wide FOV blows it up ~2×)
    const s = Math.cos((c.fovy_deg || 90) * Math.PI / 360);
    return new THREE.Vector3(xn * D * s, -yn * D * s, -D);
  }
  return new THREE.Vector3(xn, -yn, -1).normalize().multiplyScalar(D);
}
const pixelRay = (c, u, v) => pixelPoint(c, u, v, 1, false);
// A subdivided mesh of the image; curved ray-mesh by default, flat rectified plane
// when flat=true.
function imageMesh(c, D, flat){
  const NX = 32, NY = 20, row = NX + 1, pos = [], uv = [], idx = [];
  for (let j = 0; j <= NY; j++) for (let i = 0; i <= NX; i++){
    const r = pixelPoint(c, (i / NX) * c.w, (j / NY) * c.h, D, flat);
    pos.push(r.x, r.y, r.z); uv.push(i / NX, 1 - j / NY);
  }
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++){
    const a = j * row + i, b = a + 1, cc = a + row, d = cc + 1;
    idx.push(a, cc, b, b, cc, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx); return g;
}

let _gltf = null;
const loadGltf = () => (_gltf = _gltf || new Promise((res, rej) => new GLTFLoader().load(RIG_GLB, res, undefined, rej)));

class Stage {
  constructor(el){
    this.el = el;
    this.canvas = el.querySelector('canvas') || el.appendChild(document.createElement('canvas'));
    const r = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace; r.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer = r;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 200);
    this.camera.position.set(2.0, 1.5, 2.4);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(2.5, 3.5, 2); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-2.5, 1.4, -1.6); this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffe6cf, 0.55); rim.position.set(0, 1.9, -3.2); this.scene.add(rim);
    const g = new THREE.Mesh(new THREE.CircleGeometry(6, 64).rotateX(-Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: 0x14110f, roughness: .95 }));
    g.position.y = -0.004; this.scene.add(g);
    this.grid = new THREE.GridHelper(12, 48, 0x2a2622, 0x1d1a17);
    this.grid.material.transparent = true; this.grid.material.opacity = .5; this.scene.add(this.grid);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = .08;
    this.controls.maxPolarAngle = Math.PI * .52; this.controls.target.set(0, .6, 0);
    const resize = () => { const w = el.clientWidth || 2, h = el.clientHeight || 2;
      r.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); window.addEventListener('studio-resize', resize); resize();
  }
  render(){ this.controls.update(); this.renderer.render(this.scene, this.camera); }
}

class Rig {
  async load(stage, manifest){
    const gltf = await loadGltf();
    this.root = gltf.scene.clone(true);
    this.world = new THREE.Group(); this.world.add(this.root); stage.scene.add(this.world);
    this.base = this.root.getObjectByName('base_link') || this.root;
    this.baseP0 = this.base.position.clone(); this.baseQ0 = this.base.quaternion.clone();
    this.joints = {};
    for (const j of manifest.joints){
      const n = this.root.getObjectByName(j.node); if (!n) continue;
      this.joints[j.name] = { ...j, _n: n, _p0: n.position.clone(), _q0: n.quaternion.clone(),
        _ax: new THREE.Vector3(...j.axis).normalize() };
    }
    return this;
  }
  setJoint(name, val){
    const j = this.joints[name]; if (!j) return;
    if (j.type === 'prismatic'){
      j._n.position.copy(j._p0).add(j._ax.clone().applyQuaternion(j._q0).multiplyScalar(val));
      j._n.quaternion.copy(j._q0);
    } else {
      j._n.quaternion.copy(j._q0).multiply(new THREE.Quaternion().setFromAxisAngle(j._ax, val));
      j._n.position.copy(j._p0);
    }
  }
  pose(q, bx, by, yaw){
    for (const n in q) this.setJoint(n, q[n]);
    const off = new THREE.Vector3(bx, by, 0).applyQuaternion(this.baseQ0);
    this.base.position.copy(this.baseP0).add(off);
    this.base.quaternion.copy(this.baseQ0)
      .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw));
    this.world.updateMatrixWorld(true);
  }
}

function labelSprite(text, color = '#f0b68b'){
  const c = document.createElement('canvas'); c.width = 256; c.height = 64;
  const x = c.getContext('2d');
  x.font = '600 30px Geist Mono, monospace'; x.fillStyle = color; x.textBaseline = 'middle';
  x.fillText(text, 6, 34);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(0.3, 0.075, 1); sp.center.set(0, 0.5); sp.renderOrder = 1000;
  return sp;
}

// rviz-style triad: X=red, Y=green, Z=blue, in the node's LOCAL frame.
function triad(size, withLabel){
  const g = new THREE.Group();
  const dirs = [[0xff4d3d, new THREE.Vector3(1, 0, 0)], [0x4dff8a, new THREE.Vector3(0, 1, 0)], [0x4d9dff, new THREE.Vector3(0, 0, 1)]];
  for (const [col, d] of dirs){
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, size, 6),
      new THREE.MeshBasicMaterial({ color: col, depthTest: false }));
    m.position.copy(d.clone().multiplyScalar(size / 2));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
    m.renderOrder = 999; g.add(m);
  }
  if (withLabel){ const lab = labelSprite(withLabel); lab.position.set(size * 0.4, size * 0.4, size * 0.4); g.add(lab); }
  return g;
}

export class Viewport3D {
  constructor(el){
    this.el = el;
    this.stage = new Stage(el);
    this.navWorld = new THREE.Group(); this.navWorld.quaternion.copy(Q_ZUP);
    this.stage.scene.add(this.navWorld);
    this.layers = { robot: true, map: true, lidar: true, accumulate: false,
                    occupancy: false, tf: false, ghost: false, path: false, target: false,
                    frustum: false, grid: true, trail: false, clippath: false, distribution: false };
    this.traceTfs = new Set();                  // TFs to trace (multi-select, like camera streams)
    this.trajGroup = new THREE.Group(); this.navWorld.add(this.trajGroup);
    this.trails = {}; this.clipPaths = {};      // per-TF line objects
    this.distLines = []; this.onDistribution = null;
    this.cloudCams = new Set();                 // per-camera 3D point clouds (topic-based on/off)
    this.camClouds = {}; this._cloudBusy = {}; this._cloudAt = {};
    // ── two point-cloud pipelines: LOCAL (localhost / ?hires) = full-res, every
    // frame prefetched so scrubbing never blinks; REMOTE (deployed) = downsampled
    // + lazy so bandwidth stays sane. Force with ?hires / ?local=1 or ?remote=1.
    const _p = new URLSearchParams(location.search);
    const _localHost = /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])$/i.test(location.hostname);
    this.localHiRes = _p.get('remote') === '1' ? false : (_localHost || _p.has('hires') || _p.get('local') === '1');
    this.cloudDensity = this.localHiRes ? 'ultra' : 'med';
    this.cloudCfg = { prefetchAll: this.localHiRes, cacheCap: this.localHiRes ? 4000 : 240,
                      byteBudget: this.localHiRes ? 1.6e9 : 1.2e8, ...CLOUD_DENSITY[this.cloudDensity] };
    this._cloudBytes = 0; this._prefetchTok = {};
    this.reconMeshCams = new Set();              // cameras whose cuSLAM mesh is shown (multi-source)
    this.reconMesh = {}; this._reconBusy = {};   // name -> THREE.Mesh
    this.rig = null; this.entry = null;
    this.lidarCloud = null; this.accumCloud = null; this.mapCloud = null;
    this.occPlane = null; this.pathLine = null; this.targetMarker = null;
    this.camViz = {}; this.camVizGroup = new THREE.Group(); this.navWorld.add(this.camVizGroup);
    this._extrAt = 0; this._extrBusy = false; this.camProjD = 0.2;    // image distance from lens (user-adjustable)
    this.camFlat = true;                                              // default: flat/undistorted (no lens distortion)
    this.activeCams = new Set();                                      // 3D projections shown = active video cams
    this.ghostGroup = new THREE.Group(); this.navWorld.add(this.ghostGroup);
    this._accumPts = []; this._camFetchAt = 0; this._camBusy = false;
    this._ready = this._init();
    if (!new URLSearchParams(location.search).has('static'))
      this.stage.renderer.setAnimationLoop(() => this.stage.render());
  }
  renderOnce(){ this.stage.render(); }

  async _init(){
    const manifest = await fetch(JOINTS_JSON).then(r => r.json());
    this.rig = await new Rig().load(this.stage, manifest);
    this.navWorld.position.copy(this.rig.baseP0);
    this._buildTF();
    this._buildGhosts();
    this._buildTarget();
  }

  // TF triads attached to the actual rig nodes -> always aligned with the parts.
  _buildTF(){
    this.tf = [];
    for (const [label, node, size] of TF_FRAMES){
      const n = this.rig.root.getObjectByName(node);
      if (!n) continue;
      const g = triad(size, label); g.visible = false; n.add(g); this.tf.push(g);
    }
  }
  _buildGhosts(){
    this.ghosts = {};
    for (const [name, col] of [['left_palm_target', 0xf0b68b], ['right_palm_target', 0xC25B2A]]){
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.045),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.45 }));
      m.visible = false; this.ghostGroup.add(m); this.ghosts[name] = m;
    }
  }
  _buildTarget(){
    const g = new THREE.Group();
    const C = 0x4dff8a;
    // flat ring on the floor (TorusGeometry lies in the XY plane = flat in z-up)
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.014, 10, 40),
      new THREE.MeshBasicMaterial({ color: C }));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), new THREE.MeshBasicMaterial({ color: C }));
    // heading arrow lying flat on the floor, pointing +X (group is yawed to aim it)
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0.015), 0.34, C, 0.12, 0.09);
    g.add(ring, dot, arrow); g.visible = false; this.navWorld.add(g); this.targetMarker = g;
  }

  setLayer(name, on){
    this.layers[name] = on;
    if (name === 'grid') this.stage.grid.visible = on;
    if (name === 'map' && this.mapCloud) this.mapCloud.visible = on;
    if (name === 'occupancy' && this.occPlane) this.occPlane.visible = on;
    if (name === 'path' && this.pathLine) this.pathLine.visible = on;
    if (name === 'accumulate' && !on) this._clearAccum();
    if (name === 'lidar' && this.lidarCloud) this.lidarCloud.visible = on;
    if (name === 'robot' && this.rig) this.rig.world.visible = on;
    if (name === 'tf') this.tf?.forEach(g => g.visible = on);
    if (name === 'target' && this.targetMarker) this.targetMarker.visible = on;
    if (name === 'frustum'){ this._applyCamVis(); this._extrAt = 0; }
    if ((name === 'clippath' || name === 'trail') && on && this.traceTfs.size === 0)
      for (const tf of ['left_palm', 'right_palm']) this.setTraceTf(tf, true);   // a trajectory needs a traced TF
    if (name === 'clippath') for (const k in this.clipPaths) this.clipPaths[k].visible = on;
    if (name === 'distribution'){ if (on) this._loadDistribution(); else this._clearDist(); }
    if (name === 'trail' && !on) for (const k in this.trails){ this.trajGroup.remove(this.trails[k]); this.trails[k].geometry.dispose(); delete this.trails[k]; }
    if (name === 'ghost') Object.values(this.ghosts || {}).forEach(m => { if (!on) m.visible = false; });
  }
  // toggle a TF in/out of the traced set (multi-select, like the camera streams)
  setTraceTf(tf, on){
    if (on) this.traceTfs.add(tf);
    else { this.traceTfs.delete(tf);
      for (const map of [this.trails, this.clipPaths]){ if (map[tf]){ this.trajGroup.remove(map[tf]); map[tf].geometry.dispose(); delete map[tf]; } } }
    if (this.entry) this._buildClipPath(this.entry);
    if (this.layers.distribution) this._loadDistribution();
  }
  _tfColor(tf){ const m = { left_palm: [0.76, 0.36, 0.16], right_palm: [0.94, 0.71, 0.55], chassis: [0.30, 0.62, 1.0], head: [0.30, 1.0, 0.54] }; return m[tf] || [0.8, 0.8, 0.85]; }
  // per-camera 3D point cloud on/off (topic-based)
  setCloudCam(name, on){
    if (on){ this.cloudCams.add(name); if (this.cloudCfg.prefetchAll) this._prefetchClouds(name); }
    else {
      this.cloudCams.delete(name);
      this._prefetchTok[name] = (this._prefetchTok[name] || 0) + 1;   // cancel any in-flight prefetch
      if (this._cloudShown) delete this._cloudShown[name];
      if (this.camClouds[name]){ this.navWorld.remove(this.camClouds[name]); this.camClouds[name].geometry.dispose(); delete this.camClouds[name]; }
    }
  }
  // change point-cloud sparsity (ultra/high/med/low). Clears the (stride-specific)
  // cache and rebuilds the active clouds + fused reconstruction at the new density.
  setCloudDensity(level){
    if (!CLOUD_DENSITY[level] || level === this.cloudDensity) return;
    this.cloudDensity = level;
    Object.assign(this.cloudCfg, CLOUD_DENSITY[level]);
    this._cloudCache?.clear(); this._cloudBytes = 0; this._cloudShown = {};
    for (const name of this.cloudCams){
      this._prefetchTok[name] = (this._prefetchTok[name] || 0) + 1;
      if (this.camClouds[name]){ this.navWorld.remove(this.camClouds[name]); this.camClouds[name].geometry.dispose(); delete this.camClouds[name]; }
      if (this.cloudCfg.prefetchAll) this._prefetchClouds(name);
    }
  }
  // ── cuSLAM surface mesh, PER CAMERA (multi-source) ────────────────────────
  // Each depth camera can publish its own fused world-frame reconstruction mesh
  // (Rerun Mesh3D). Static over the episode, so fetched once per toggle.
  setReconMeshCam(name, on){
    if (on){ this.reconMeshCams.add(name); this._fetchReconMesh(name); }
    else { this.reconMeshCams.delete(name); this._clearReconMesh(name); }
  }
  _clearReconMesh(name){
    const o = this.reconMesh[name];
    if (o){ this.navWorld.remove(o); o.geometry.dispose(); o.material?.dispose?.(); delete this.reconMesh[name]; }
  }
  async _fetchReconMesh(name){
    if (!this.entry) return;
    const tag = name + ':' + this.entry.ds + '/' + this.entry.ep;
    this._reconBusy[name] = tag;
    try {
      const frames = this.localHiRes ? 10 : 6;
      const buf = await api.recon(this.entry.ds, this.entry.ep, name, 'mesh', frames);
      if (this._reconBusy[name] !== tag || !this.reconMeshCams.has(name)) return;   // superseded / off
      this._clearReconMesh(name);
      const m = this._parseMesh(buf); if (!m) return;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(m.pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(m.col, 3));
      geo.setIndex(new THREE.BufferAttribute(m.idx, 1));
      geo.computeVertexNormals();                                 // shaded surface (Rerun Mesh3D)
      this.reconMesh[name] = new THREE.Mesh(geo,
        new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide,
                                         roughness: 0.92, metalness: 0.0, flatShading: false }));
      this.navWorld.add(this.reconMesh[name]);
    } catch { /* leave prior mesh in place */ }
  }
  _parseMesh(buf){
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (dv.getUint8(0) !== 0x06) return null;
    const nv = dv.getUint32(4, true), nf = dv.getUint32(8, true);
    if (!nv || !nf) return null;
    const voff = 12, coff = voff + nv * 12, foff = coff + nv * 4;
    const pos = new Float32Array(buf.buffer, buf.byteOffset + voff, nv * 3).slice();
    const col = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++){
      col[i * 3] = dv.getUint8(coff + i * 4) / 255;
      col[i * 3 + 1] = dv.getUint8(coff + i * 4 + 1) / 255;
      col[i * 3 + 2] = dv.getUint8(coff + i * 4 + 2) / 255;
    }
    const idx = new Uint32Array(buf.buffer, buf.byteOffset + foff, nf * 3).slice();
    return { pos, col, idx };
  }

  // 3D camera projections shown = the cameras currently shown in the video wall
  setActiveCams(set){ this.activeCams = new Set(set); this._applyCamVis(); this._extrAt = 0; }
  _applyCamVis(){ const on = this.layers.frustum;
    for (const name in this.camViz) this.camViz[name].group.visible = on && this.activeCams.has(name); }
  // adjust how far the projected image sits from the lens, then rebuild + re-pose
  setCamProjDistance(d){
    this.camProjD = Math.max(0.03, d);
    if (this.calib && this.entry){ this._buildCamViz(this.calib, this.entry.ds, this.entry.ep); this._extrAt = 0; }
  }
  // toggle flat/rectified ("undistort") vs curved ray-mesh projection
  setCamFlat(on){
    this.camFlat = on;
    if (this.calib && this.entry){ this._buildCamViz(this.calib, this.entry.ds, this.entry.ep); this._extrAt = 0; }
  }

  async setEpisode(entry){
    this.entry = entry;
    await this._ready;
    // navWorld holds world-frame content (lidar / camera clouds / camera frusta /
    // map) in MJCF ABSOLUTE coords. The rig's base sits at baseP0.y in three.js
    // but at z=chassisZ in MJCF, so absolute MJCF z must be shifted by
    // baseP0.y − chassisZ — otherwise the base height is double-counted and the
    // clouds float ~chassisZ too high. (chassisZ ≈ constant base ride height.)
    const cp = entry.bundle.topics['pose/chassis'];
    const chassisZ = cp ? cp.data[2] : 0;
    this.navWorld.position.set(this.rig.baseP0.x, this.rig.baseP0.y - chassisZ, this.rig.baseP0.z);
    this._clearAccum();
    this._buildPath(entry);
    const { ds, ep } = entry;
    // 3D lidar map
    try { this._setMap(this._parseCloud(await api.scene(ds, ep, 0.05, 2))); } catch {}
    // re-fuse any active cuSLAM meshes for the new episode
    for (const name of Object.keys(this.reconMesh)) this._clearReconMesh(name);
    for (const name of this.reconMeshCams) this._fetchReconMesh(name);
    // per-camera clouds: drop old-episode geometry; (local) prefetch the new one so
    // scrubbing is blink-free from the first frame
    this._cloudShown = {};
    for (const name of this.cloudCams){
      if (this.camClouds[name]){ this.navWorld.remove(this.camClouds[name]); this.camClouds[name].geometry.dispose(); delete this.camClouds[name]; }
      if (this.cloudCfg.prefetchAll) this._prefetchClouds(name);
    }
    // 2D occupancy map
    try { this._setOccupancy(await fetch(`/api/ds/${encodeURIComponent(ds)}/ep/${encodeURIComponent(ep)}/occupancy.json`).then(r => r.ok ? r.json() : null)); } catch {}
    // Foxglove-style camera frustums + projected images
    try { this._buildCamViz(await api.calibration(ds, ep), ds, ep); } catch {}
    // prefetch ALL camera world-poses once → per-frame projection needs no network.
    // Fire-and-forget (NEVER await) so a cold model-compile can't freeze the load;
    // projection starts tracking the moment it resolves.
    this.extrAll = null;
    api.extrinsicsAll(ds, ep).then(e => { if (this.entry && this.entry.ds === ds && this.entry.ep === ep) this.extrAll = e; }).catch(() => {});
    // trajectory clip-path for the traced TF
    this._buildClipPath(entry);
    if (this.layers.distribution) this._loadDistribution();
  }

  // Build one frustum + image plane per camera (Foxglove camera-in-3D).
  _buildCamViz(calib, ds, ep){
    for (const v of Object.values(this.camViz)){ this.camVizGroup.remove(v.group); v.group.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); }); }
    this.camViz = {}; this.calib = calib;
    for (const c of calib){
      const D = this.camProjD;                          // range from the lens (user-adjustable)
      const grp = new THREE.Group();
      // frustum edges along the TRUE corner rays (so they bound the curved image)
      const flat = this.camFlat;
      const A = new THREE.Vector3(0, 0, 0);
      const TL = pixelPoint(c, 0, 0, D, flat), TR = pixelPoint(c, c.w, 0, D, flat);
      const BR = pixelPoint(c, c.w, c.h, D, flat), BL = pixelPoint(c, 0, c.h, D, flat);
      const segs = [A, TL, A, TR, A, BR, A, BL, TL, TR, TR, BR, BR, BL, BL, TL];
      grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs),
        new THREE.LineBasicMaterial({ color: 0x4d9dff })));
      // up-tick (green) marking image top
      const topMid = pixelPoint(c, c.w / 2, 0, D, flat);
      grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([topMid, topMid.clone().multiplyScalar(1.25)]),
        new THREE.LineBasicMaterial({ color: 0x4dff8a })));
      // projected image — curved ray-mesh through the lens model (distortion-aware)
      const img = new Image(); img.crossOrigin = 'anonymous';
      const tex = new THREE.Texture(img); tex.colorSpace = THREE.SRGBColorSpace;
      img.onload = () => { tex.needsUpdate = true; };
      const mesh = new THREE.Mesh(imageMesh(c, D, flat),
        new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.97 }));
      grp.add(mesh);
      const lab = labelSprite(`${c.name}`, '#4d9dff'); lab.scale.multiplyScalar(0.55); lab.position.copy(TL); grp.add(lab);
      this.camVizGroup.add(grp);
      this.camViz[c.name] = { group: grp, img, tex, calib: c, ds, ep };
    }
    this._applyCamVis();
  }

  // ── trajectories ────────────────────────────────────────────────────────
  _poseSeries(entry, tf){
    const t = entry.bundle.topics['pose/' + tf];
    if (!t) return null;
    return { data: t.data, dim: t.dim, T: entry.T };
  }
  // fading past-trail per traced TF up to the current frame (recent = bright)
  _updateTrail(entry, f){
    for (const tf of this.traceTfs){
      const s = this._poseSeries(entry, tf); if (!s) continue;
      const n = Math.max(2, f + 1), pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
      const base = this._tfColor(tf);
      for (let i = 0; i < n; i++){
        pos[i * 3] = s.data[i * s.dim]; pos[i * 3 + 1] = s.data[i * s.dim + 1]; pos[i * 3 + 2] = s.data[i * s.dim + 2];
        const a = 0.12 + 0.88 * (i / (n - 1));
        col[i * 3] = base[0] * a + 0.04; col[i * 3 + 1] = base[1] * a + 0.04; col[i * 3 + 2] = base[2] * a + 0.04;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      if (this.trails[tf]){ this.trajGroup.remove(this.trails[tf]); this.trails[tf].geometry.dispose(); }
      this.trails[tf] = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true }));
      this.trajGroup.add(this.trails[tf]);
    }
  }
  _buildClipPath(entry){
    for (const tf of this.traceTfs){
      const s = this._poseSeries(entry, tf); if (!s) continue;
      const pts = [];
      for (let i = 0; i < s.T; i++) pts.push(new THREE.Vector3(s.data[i * s.dim], s.data[i * s.dim + 1], s.data[i * s.dim + 2]));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const c = this._tfColor(tf);
      if (this.clipPaths[tf]){ this.trajGroup.remove(this.clipPaths[tf]); this.clipPaths[tf].geometry.dispose(); }
      this.clipPaths[tf] = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: new THREE.Color(c[0], c[1], c[2]) }));
      this.clipPaths[tf].visible = this.layers.clippath; this.trajGroup.add(this.clipPaths[tf]);
    }
    for (const k in this.clipPaths) if (!this.traceTfs.has(k)){ this.trajGroup.remove(this.clipPaths[k]); this.clipPaths[k].geometry.dispose(); delete this.clipPaths[k]; }
  }
  _clearDist(){
    for (const l of this.distLines){ this.trajGroup.remove(l); l.geometry.dispose(); l.material.dispose(); }
    this.distLines = []; this.onDistribution && this.onDistribution([]);
  }
  async _loadDistribution(){
    this._clearDist();
    const legend = [];
    for (const tf of this.traceTfs){
      let traj;
      try { traj = await fetch(`/api/trajectories?tf=${encodeURIComponent(tf)}&stride=3`).then(r => r.json()); }
      catch { continue; }
      traj.forEach((t, idx) => {
        const hue = (idx * 47) % 360, c = new THREE.Color(`hsl(${hue},70%,60%)`);
        const pts = t.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.6 }));
        this.trajGroup.add(line); this.distLines.push(line);
        legend.push({ label: `${tf} · ${t.episode}`, color: `hsl(${hue},70%,60%)`, n: t.n });
      });
    }
    this.onDistribution && this.onDistribution(legend);
  }

  // Per-frame camera projection — pure local indexing into the prefetched
  // extrinsics (no network, no MuJoCo, no throttle) so it tracks playback 1:1.
  _maybeExtrinsics(entry, f){
    if (!this.activeCams.size) return;
    const E = this.extrAll; if (!E || !E.cams) return;
    const T = E.T || 1, i = Math.max(0, Math.min(f, T - 1));
    for (const name in this.camViz){
      if (!this.activeCams.has(name)) continue;
      const v = this.camViz[name], ex = E.cams[name]; if (!ex) continue;
      const p = ex.pos, m = ex.mat, o3 = i * 3, o9 = i * 9;
      v.group.position.set(p[o3], p[o3 + 1], p[o3 + 2]);
      const xa = new THREE.Vector3(m[o9], m[o9 + 3], m[o9 + 6]);
      const ya = new THREE.Vector3(m[o9 + 1], m[o9 + 4], m[o9 + 7]);
      const za = new THREE.Vector3(m[o9 + 2], m[o9 + 5], m[o9 + 8]);
      v.group.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(xa, ya, za));
      const url = api.camFrameUrl(v.ds, v.ep, name, i);
      if (v.img.getAttribute('src') !== url) v.img.src = url;
    }
  }

  _buildPath(entry){
    if (this.pathLine){ this.navWorld.remove(this.pathLine); this.pathLine.geometry.dispose(); this.pathLine = null; }
    const t = entry.bundle.topics['pose/chassis']; if (!t) return;
    const pts = [];
    for (let f = 0; f < entry.T; f++) pts.push(new THREE.Vector3(t.data[f * t.dim], t.data[f * t.dim + 1], 0.03));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    this.pathLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xC25B2A }));
    this.pathLine.visible = this.layers.path; this.navWorld.add(this.pathLine);
  }

  _parseCloud(buf){
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const count = dv.getUint32(4, true);
    const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
    let o = 8;
    for (let i = 0; i < count; i++){
      pos[i * 3] = dv.getFloat32(o, true); pos[i * 3 + 1] = dv.getFloat32(o + 4, true); pos[i * 3 + 2] = dv.getFloat32(o + 8, true);
      col[i * 3] = dv.getUint8(o + 12) / 255; col[i * 3 + 1] = dv.getUint8(o + 13) / 255; col[i * 3 + 2] = dv.getUint8(o + 14) / 255;
      o += 16;
    }
    return { pos, col, count };
  }
  _points(pos, col, size){
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true }));
  }
  // Zero-copy geometry straight off the CLOUD (0x05) binary: bind the interleaved
  // 16-byte records (f32 xyz + u8 rgba) as GPU attributes — no per-point JS loop,
  // so even a full-res (100k+ pt) cloud builds instantly. Returns null if empty.
  _geoFromBin(buf){
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const count = dv.getUint32(4, true);
    if (!count) return null;
    const off = buf.byteOffset + 8;                       // records start after the 8-byte header
    const geo = new THREE.BufferGeometry();
    const posIB = new THREE.InterleavedBuffer(new Float32Array(buf.buffer, off, count * 4), 4);       // 16B stride = 4 floats
    geo.setAttribute('position', new THREE.InterleavedBufferAttribute(posIB, 3, 0));
    const colIB = new THREE.InterleavedBuffer(new Uint8Array(buf.buffer, off, count * 16), 16);       // rgba at byte 12
    geo.setAttribute('color', new THREE.InterleavedBufferAttribute(colIB, 3, 12, true));              // u8 → normalized
    geo.userData.count = count;
    return geo;
  }
  _pointsFromBin(buf, size){
    const g = this._geoFromBin(buf);
    return g ? new THREE.Points(g, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true })) : null;
  }
  _setMap({ pos, col }){
    if (this.mapCloud){ this.navWorld.remove(this.mapCloud); this.mapCloud.geometry.dispose(); }
    this.mapCloud = this._points(pos, col, 0.03); this.mapCloud.visible = this.layers.map;
    this.navWorld.add(this.mapCloud);
  }
  _setOccupancy(occ){
    if (this.occPlane){ this.navWorld.remove(this.occPlane); this.occPlane.geometry.dispose(); this.occPlane = null; }
    if (!occ) return;
    const tex = new THREE.TextureLoader().load('data:image/png;base64,' + occ.png_b64);
    tex.colorSpace = THREE.SRGBColorSpace;
    const w = occ.maxx - occ.minx, h = occ.maxy - occ.miny;
    const geo = new THREE.PlaneGeometry(w, h);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    m.position.set((occ.minx + occ.maxx) / 2, (occ.miny + occ.maxy) / 2, 0.012);
    m.visible = this.layers.occupancy; this.occPlane = m; this.navWorld.add(m);
  }
  _clearAccum(){
    this._accumPts = [];
    if (this.accumCloud){ this.navWorld.remove(this.accumCloud); this.accumCloud.geometry.dispose(); this.accumCloud = null; }
  }

  update(entry, f){
    if (!this.rig) return;
    const jp = frameVec(entry, 'joint_pos', f);
    if (jp){
      const q = {}; const am = entry.actMap;
      for (let i = 0; i < am.length; i++){ const m = am[i];
        if (m.kind === 'split') for (const t of m.targets) q[t] = jp[i] * m.factor; else q[m.target] = jp[i]; }
      let bx = 0, by = 0, yaw = 0;
      const cp = frameVec(entry, 'pose/chassis', f);
      if (cp){ bx = cp[0]; by = cp[1];
        const qw = cp[3], qx = cp[4], qy = cp[5], qz = cp[6];
        yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)); }
      this.rig.pose(q, bx, by, yaw); this._lastBase = { bx, by, yaw };
    }
    if (this.layers.lidar || this.layers.accumulate) this._updateLidar(entry, f);
    if (this.layers.ghost) this._updateGhosts(entry, f);
    if (this.layers.target) this._updateTarget(entry, f);
    if (this.cloudCams.size) this._updateCamClouds(entry, f);
    if (this.layers.frustum) this._maybeExtrinsics(entry, f);
    if (this.layers.trail) this._updateTrail(entry, f);
  }

  _updateLidar(entry, f){
    const lid = frameVec(entry, 'lidar', f);
    if (!lid || !this._lastBase) return;
    const { bx, by, yaw } = this._lastBase, n = lid.length, c = Math.cos(yaw), s = Math.sin(yaw);
    const pts = [], cols = [];
    for (let i = 0; i < n; i++){ const r = lid[i]; if (r <= 0.05) continue;
      const th = i * Math.PI / 180, lx = r * Math.cos(th), ly = r * Math.sin(th);
      pts.push(bx + lx * c - ly * s, by + lx * s + ly * c, LIDAR_Z);
      const [cr, cg, cb] = turbo(r / 8); cols.push(cr, cg, cb); }
    if (this.lidarCloud){ this.navWorld.remove(this.lidarCloud); this.lidarCloud.geometry.dispose(); }
    this.lidarCloud = this._points(new Float32Array(pts), new Float32Array(cols), 0.045);
    this.lidarCloud.visible = this.layers.lidar; this.navWorld.add(this.lidarCloud);
    if (this.layers.accumulate){
      for (const v of pts) this._accumPts.push(v);
      if (this.accumCloud){ this.navWorld.remove(this.accumCloud); this.accumCloud.geometry.dispose(); }
      const ap = new Float32Array(this._accumPts), ac = new Float32Array(ap.length).fill(0.55);
      this.accumCloud = this._points(ap, ac, 0.025); this.navWorld.add(this.accumCloud);
    }
  }
  _updateGhosts(entry, f){
    for (const name in this.ghosts){ const p = frameVec(entry, 'teleop/' + name, f); const m = this.ghosts[name];
      if (!p){ m.visible = false; continue; } m.visible = true;
      m.position.set(p[0], p[1], p[2]); m.quaternion.set(p[4], p[5], p[6], p[3]); }
  }
  _updateTarget(entry, f){
    const p = frameVec(entry, 'teleop/look_at', f);
    if (!p){ this.targetMarker.visible = false; return; }
    this.targetMarker.visible = true;
    this.targetMarker.position.set(p[0], p[1], 0.02);          // ON THE FLOOR (z≈0)
    // heading: aim from the robot toward the target (the nav direction)
    let yaw = this._lastBase ? this._lastBase.yaw : 0;
    if (this._lastBase){ const dx = p[0] - this._lastBase.bx, dy = p[1] - this._lastBase.by;
      if (Math.hypot(dx, dy) > 1e-3) yaw = Math.atan2(dy, dx); }
    this.targetMarker.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);
  }
  _cloudKey(entry, name, f){ return entry.ds + '|' + entry.ep + '|' + name + '|' + f; }
  _cacheCloud(key, buf){
    if (!this._cloudCache) this._cloudCache = new Map();
    if (this._cloudCache.has(key)) return;
    this._cloudCache.set(key, buf); this._cloudBytes += buf.byteLength;
    const cfg = this.cloudCfg;                             // LRU-evict by count AND byte budget
    while (this._cloudCache.size > cfg.cacheCap || this._cloudBytes > cfg.byteBudget){
      const k0 = this._cloudCache.keys().next().value; if (k0 === undefined) break;
      const b0 = this._cloudCache.get(k0); this._cloudCache.delete(k0); this._cloudBytes -= (b0?.byteLength || 0);
    }
  }
  _updateCamClouds(entry, f){
    if (!this._cloudCache) this._cloudCache = new Map();
    const now = performance.now(), cfg = this.cloudCfg;
    for (const name of this.cloudCams){
      const key = this._cloudKey(entry, name, f);
      const hit = this._cloudCache.get(key);
      if (hit){                                            // cached → swap geometry instantly (no parse, no blink)
        if (this._cloudShown?.[name] !== key){ this._showCamCloud(name, hit); (this._cloudShown ||= {})[name] = key; }
        continue;
      }
      // LOCAL: prefetch fills the cache; still allow a lazy fetch as a fallback for
      // frames not yet prefetched (throttled). REMOTE: lazy per-frame only.
      if (this._cloudBusy[name] || now - (this._cloudAt[name] || 0) < (cfg.prefetchAll ? 60 : 250)) continue;
      this._cloudBusy[name] = true; this._cloudAt[name] = now;
      api.camCloud(entry.ds, entry.ep, name, f, cfg.stride, cfg.maxPoints).then(buf => {
        if (!this.cloudCams.has(name)) return;             // toggled off mid-fetch
        this._cacheCloud(key, buf);
        this._showCamCloud(name, buf); (this._cloudShown ||= {})[name] = key;
      }).catch(() => {}).finally(() => { this._cloudBusy[name] = false; });
    }
  }
  // Prefetch EVERY frame's cloud for a camera (local mode) so playback/scrub is a
  // pure cache lookup — the fix for the "point cloud blinks" stutter. Concurrency-
  // limited, cancellable (token), and stops at the byte budget.
  async _prefetchClouds(name){
    if (!this.entry) return;
    const cfg = this.cloudCfg, entry = this.entry, T = entry.T;
    const mine = (this._prefetchTok[name] = (this._prefetchTok[name] || 0) + 1);
    let next = 0;
    const worker = async () => {
      while (next < T){
        const f = next++;
        if (this._prefetchTok[name] !== mine || !this.cloudCams.has(name)) return;
        if (this._cloudBytes > cfg.byteBudget) return;     // memory guard — rest stays lazy
        const key = this._cloudKey(entry, name, f);
        if (this._cloudCache?.has(key)) continue;
        try {
          const buf = await api.camCloud(entry.ds, entry.ep, name, f, cfg.stride, cfg.maxPoints);
          if (this._prefetchTok[name] !== mine) return;
          this._cacheCloud(key, buf);
        } catch { /* skip this frame */ }
      }
    };
    await Promise.all(Array.from({ length: 8 }, worker));   // 8-way parallel (localhost = free)
  }
  _showCamCloud(name, buf){
    const geo = this._geoFromBin(buf);
    const obj = this.camClouds[name];
    if (obj){                                              // reuse object+material, swap geometry (no GC churn)
      obj.geometry.dispose();
      if (geo){ obj.geometry = geo; obj.visible = true; } else obj.visible = false;
    } else if (geo){
      const m = new THREE.PointsMaterial({ size: this.cloudCfg.pointSize, vertexColors: true, sizeAttenuation: true });
      this.camClouds[name] = new THREE.Points(geo, m); this.navWorld.add(this.camClouds[name]);
    }
  }

  // The transform tree, straight from the rig's scene graph (= the TF tree).
  getTfTree(){
    if (!this.rig) return null;
    const round = (v) => Math.round(v * 1000) / 1000;
    const isJoint = new Set(Object.values(this.rig.joints).map(j => j._n.name));
    const build = (o) => {
      const children = [];
      for (const c of o.children){
        if (c.isMesh) continue;                    // skip visual geometry leaves
        if (c.name) children.push(build(c));
        else children.push(...build(c).children);   // flatten unnamed wrappers
      }
      return { name: o.name || '(root)', pos: [round(o.position.x), round(o.position.y), round(o.position.z)],
               joint: isJoint.has(o.name), children };
    };
    return build(this.rig.root);
  }
  // pulse a bright triad on a clicked frame so you can locate it in 3D
  highlightFrame(name){
    if (!this.rig) return;
    const n = this.rig.root.getObjectByName(name); if (!n) return;
    if (!this._hl){
      this._hl = triad(0.22, null);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.1, 0.009, 8, 28),
        new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
      ring.renderOrder = 1002; this._hl.add(ring);
    }
    n.add(this._hl); this._hlName = name;
  }
  clearHighlight(){ if (this._hl && this._hl.parent) this._hl.parent.remove(this._hl); this._hlName = null; }

  frameRobot(){
    if (!this.rig) return;
    const box = new THREE.Box3().setFromObject(this.rig.world);
    const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    this.stage.controls.target.copy(c);
    const d = Math.max(sz.x, sz.y, sz.z) || 2;
    this.stage.camera.position.copy(c).add(new THREE.Vector3(d * .9, d * .6, d * 1.2));
  }
}
