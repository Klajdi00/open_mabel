// The live rviz/Foxglove-style 3D scene: the MABEL rig posed from the 50 Hz
// robot_state mirror, inside the map it is building right now — nvblox mesh,
// 2D occupancy grid, Nav2 path, live sensor cloud, LiDAR, TF triads, camera
// frusta with the LIVE MJPEG streams projected through the lens model, the
// nav-goal marker (click-drag to command) and draggable wrist-target balls
// (task-space IK → joint_command).
//
// Stage / Rig / setJoint / triad / turbo / undistort are vendored from
// robot_learning/data_curation/web/js/viewport3d.js (per repo rule: copy, not import).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as bus from './bus.js';
import { CamView } from './camview.js';
import { solveIK } from './kinematics.js';
import { makeGuard } from './collision.js';
import { REST_POSE } from './homepose.js';
import { moduleById } from './modules.js';

// armed nav-goal cursor: a small green goal reticle (ring + heading tick)
const GOAL_CURSOR = `url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 28 28'><circle cx='14' cy='14' r='8' fill='none' stroke='%234dff8a' stroke-width='2.4'/><circle cx='14' cy='14' r='2.2' fill='%234dff8a'/><path d='M22 14h5M24.6 11.4 27 14l-2.4 2.6' stroke='%234dff8a' stroke-width='2' fill='none'/></svg>") 14 14, crosshair`;

// task-space plan preview: the ghost sweeps live → planned, pauses on the
// target so the endpoint is readable, then loops. Same raised cosine the
// timeline blends use (motions.js `ease`), so every eased motion in the studio
// has one shape.
const PLAN_EASE = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));
const PLAN_LOOP_HOLD_S = 0.55;

const RIG_GLB = '/website-assets/mabel_rig.glb';
const JOINTS_JSON = '/website-assets/mabel_joints.json';
const Q_ZUP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const LIDAR_Z = 0.25;

const TF_FRAMES = [
  ['base_link', 'base_link', 0.16], ['lift_top', 'lift_top', 0.12], ['torso', 'torso', 0.13],
  ['head', 'head', 0.12],
  ['l_shoulder', 'l_shoulder_1', 0.1], ['l_elbow', 'l_elbow', 0.09], ['l_wrist', 'l_wrist', 0.08], ['l_palm', 'left_palm', 0.07],
  ['r_shoulder', 'r_shoulder_1', 0.1], ['r_elbow', 'r_elbow', 0.09], ['r_wrist', 'r_wrist', 0.08], ['r_palm', 'right_palm', 0.07],
];

// stream name → rig-manifest camera name (mabel_camera_rig.json)
const CAM_TO_RIG = { main: 'zed_left', front: 'body_front_cam',
                     wrist_left: 'l_wrist_cam', wrist_right: 'r_wrist_cam' };
// MJCF body names → GLB/URDF node names (the rig json speaks MJCF)
const MOUNT_ALIAS = { mabel_chassis: 'base_link' };

export function turbo(t){
  t = Math.max(0, Math.min(1, t));
  const r = 34.6 + t * (1172 - t * (10794 - t * (33300 - t * (38394 - t * 14825))));
  const g = 23.3 + t * (557 + t * (1225 - t * (3575 - t * (1074 + t * 708))));
  const b = 27.2 + t * (3211 - t * (15328 - t * (27814 - t * (22569 - t * 6839))));
  return [Math.max(0, Math.min(255, r)) / 255, Math.max(0, Math.min(255, g)) / 255, Math.max(0, Math.min(255, b)) / 255];
}

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
function pixelPoint(c, u, v, D, flat){
  const [xn, yn] = undistort((u - c.cx) / c.fx, (v - c.cy) / c.fy, c.dist);
  if (flat){
    // rectified pinhole plane AT depth D: matches the curved ray-cap exactly on
    // the optical axis and the frustum edges at the same plane, so toggling
    // distort/undistort keeps the focus distance (no size/position jump)
    return new THREE.Vector3(xn * D, -yn * D, -D);
  }
  return new THREE.Vector3(xn, -yn, -1).normalize().multiplyScalar(D);
}
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
// MESHOPT: the current rig ships EXT_meshopt_compression + KHR_mesh_quantization (1.9 MB
// instead of 27 MB). A bare GLTFLoader cannot read it and fails the whole scene, which is
// why this page had to be fed an uncompressed rig and therefore kept showing the OLD head
// and body.
// CONTENT-KEYED URL. Cache-Control alone cannot rescue a browser that already stored the
// rig BEFORE the headers were fixed — and the rig keeps one URL across geometry changes, so
// a stale entry renders the old head and body forever. HEAD it first (no-store, ~0 bytes),
// then load with the ETag in the query: the URL changes only when the file does, so a real
// change always refetches and an unchanged rig still hits cache.
const loadGltf = () => (_gltf = _gltf || (async () => {
  let v = '';
  try {
    const h = await fetch(RIG_GLB, { method: 'HEAD', cache: 'no-store' });
    v = (h.headers.get('etag') || h.headers.get('last-modified') || '').replace(/\W/g, '');
  } catch { /* offline or HEAD refused — fall back to the bare URL */ }
  const url = v ? `${RIG_GLB}?v=${v}` : RIG_GLB;
  return new Promise((res, rej) =>
    new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).load(url, res, undefined, rej));
})());

class Stage {
  constructor(el){
    this.el = el;
    // DIRECT child only — the host section also contains nested panel canvases
    // (e.g. the webcam-teleop preview #btCanvas); a bare querySelector('canvas')
    // grabbed the first one and the 3D scene rendered INTO the webcam window
    // while the main viewport stayed black.
    this.canvas = el.querySelector(':scope > canvas') || el.appendChild(document.createElement('canvas'));
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
    const resize = () => {
      // in PiP mode the canvas has its own CSS box (240×170) — size the buffer
      // + camera aspect from the CANVAS, never the host, or the view stretches
      const w = this.canvas.clientWidth || el.clientWidth || 2;
      const h = this.canvas.clientHeight || el.clientHeight || 2;
      r.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(el); window.addEventListener('studio-resize', resize); resize();
  }
  render(){
    // a disabled OrbitControls must not steer the camera — the POV transition
    // owns position + quaternion while it runs
    if (this.controls.enabled) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

class Rig {
  async load(stage, manifest){
    const gltf = await loadGltf();
    this.root = gltf.scene.clone(true);
    this.world = new THREE.Group(); this.world.add(this.root); stage.scene.add(this.world);
    this.base = this.root.getObjectByName('base_link') || this.root;
    this.baseP0 = this.base.position.clone(); this.baseQ0 = this.base.quaternion.clone();
    // Strip the GLB's LATERAL centring so base_link lands on the odom origin —
    // done HERE, per Rig, so every instance is correct by construction. Doing it
    // at the call site meant the MoveIt ghost (a second Rig) kept the bake and
    // rendered ~23 cm beside the live robot. X/Z only: baseP0.y is base_link's
    // real height, and zeroing it sinks the rig through the floor.
    this.world.position.set(-this.baseP0.x, 0, -this.baseP0.z);
    this.joints = {};
    this.mimics = {};             // leader → [follower]: a joint with `mimic` has no wire value of its own
    for (const j of manifest.joints){
      const n = this.root.getObjectByName(j.node); if (!n) continue;
      this.joints[j.name] = { ...j, _n: n, _p0: n.position.clone(), _q0: n.quaternion.clone(),
        _ax: new THREE.Vector3(...j.axis).normalize() };
      if (j.mimic) (this.mimics[j.mimic] ||= []).push(j.name);
    }
    return this;
  }
  // The :9090 wire streams ABSOLUTE MuJoCo qpos, but the GLB geometry is baked at
  // the ASSEMBLED pose (qpos == ref) — so a joint must be driven by the
  // DISPLACEMENT `qpos − ref`, never the raw value. Only the 14 ORCA finger/
  // wrist joints have ref != 0 (the thumb chain reaches −0.733 rad ≈ 42°), which
  // is exactly why ignoring it renders the HANDS contorted — thumbs hyper-
  // extended at the last joint, fingers out of register with the MuJoCo view and
  // with their own pointcloud — while every arm/torso/neck/lift joint (ref = 0)
  // looks perfectly fine. iOS and Vision Pro already subtract it (rig.json);
  // this is the web viewers doing the same from mabel_joints.json.
  // NEVER "fix" this by streaming qpos−ref from the bridge: that double-
  // subtracts against every already-correct client.
  // A `mimic` joint (manifest `mimic: <leader>`, e.g. a parallel gripper's second finger) is drawn at
  // its leader's value — its own axis already points the opposite way, so the pair moves together.
  setJoint(name, val){
    const j = this.joints[name]; if (!j) return;
    for (const f of this.mimics[name] || []) this.setJoint(f, val);
    const d = val - (j.ref || 0);
    if (j.type === 'prismatic'){
      j._n.position.copy(j._p0).add(j._ax.clone().applyQuaternion(j._q0).multiplyScalar(d));
      j._n.quaternion.copy(j._q0);
    } else {
      j._n.quaternion.copy(j._q0).multiply(new THREE.Quaternion().setFromAxisAngle(j._ax, d));
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
  constructor(el, opts = {}){
    this.el = el;
    this.stage = new Stage(el);
    this.onNavGoal = opts.onNavGoal || null;        // (x, y, yaw) in odom frame
    this.onNavPreview = opts.onNavPreview || null;  // live preview while dragging
    this.onGoalPlaced = opts.onGoalPlaced || null;  // marker dropped/adjusted
    this.onJointTargets = opts.onJointTargets || null;  // {joint: rad} from IK drag
    this.navWorld = new THREE.Group(); this.navWorld.quaternion.copy(Q_ZUP);
    this.stage.scene.add(this.navWorld);
    this.layers = { robot: true, grid: true, tf: false, lidar: false,
                    vptrack: false, cloud: false,
                    mesh: true, occupancy: false, path: true, navgoal: true,
                    wrist: false, frustum: false, traj: false };
    this.pointSize = 0.02;
    this.camProjSize = 0.3;       // projected image HEIGHT in meters (uniform
                                  // across cameras; per-camera distance derives
                                  // from each lens's true FOV)
    this.camFlat = true;
    this.activeCams = new Set();
    this.rig = null;
    this.lidarCloud = null; this.liveCloud = null; this.occPlane = null;
    this.pathLine = null;
    this._meshLive = new THREE.Group(); this._meshPending = new THREE.Group();
    this.navWorld.add(this._meshLive);
    this.camViz = {};
    this._navArm = false;                            // "set nav goal" mode armed
    // Tap-the-floor-to-place. App-driven (Drive space + 3rd person only) —
    // app.js flips it; the old arm-a-cursor path stays for `armNavGoal()`.
    this.navTapEnabled = false;
    this._tap = null;                                // pointerdown origin
    this._lastBase = { bx: 0, by: 0, yaw: 0 };
    this._drag = null;
    this._ready = this._init();
    this._wirePointer();
    this._trajInit();
    this.stage.renderer.setAnimationLoop(() => this._tick());
    // rAF is throttled/paused in occluded or headless windows; a low-rate
    // interval floor keeps telemetry visibly moving there (real tabs get rAF)
    this._fallback = setInterval(() => this._tick(), 250);
  }

  async _init(){
    const manifest = await fetch(JOINTS_JSON).then(r => r.json());
    this.manifest = manifest;
    this.rig = await new Rig().load(this.stage, manifest);
    // Seat it at the REST pose, not the GLB's baked qpos-0 pose. Zero folds the arms UP
    // and out with the elbows square (homepose.js measured it: palms 7.4 cm ABOVE the
    // shoulder, 9.38 N.m of holding torque), so a robot that is merely not reporting yet
    // was drawn in a pose it would never rest in. Joints the robot DOES report overwrite
    // this on the next frame; joints it never reports simply stay somewhere plausible.
    this.rig.pose(REST_POSE, 0, 0, 0);
    // navWorld is the ODOM frame: every world-referenced overlay lives in it
    // (pointclouds, LiDAR, the SLAM mesh, the path, the occupancy plane), and
    // their coordinates arrive already in odom — MEASURED: the live cloud's
    // dominant plane (37% of its points) sits at odom z = 0, and the rig's feet
    // sit at world y = 0. They coincide, so this group belongs at the ORIGIN.
    //
    // It used to be translated by `rig.baseP0` — the GLB's bbox-centring bake
    // (urdf_to_glb's T_center, measured here as x +23.2 cm / +7.8 cm). That is
    // wrong twice over: the world data never had the bake applied, and baseP0
    // is a Z-UP vector being copied into a Y-UP group, so its lateral component
    // became a HEIGHT. The visible result was the floor cloud floating ~8 cm
    // above the robot's wheels with everything shifted sideways — the
    // frame-contract bug the root README warns about.
    this.navWorld.position.set(0, 0, 0);
    this._buildTF();
    this._buildNavTarget();
    this._buildWristTargets();
    try {
      const rig = await fetch('/api/camera-rig').then(r => r.json());
      let live = {};
      try { live = await fetch('/api/camera-info').then(r => r.json()); } catch {}
      // overlay LIVE intrinsics + distortion (real lenses) onto the baked rig
      const RIG_TO_INFO = { body_front_cam: 'front', zed_left: 'zed',
                            l_wrist_cam: 'left_wrist', r_wrist_cam: 'right_wrist' };
      const cams = (rig.cameras || rig).map((c) => {
        const l = live[RIG_TO_INFO[c.name] || c.name];
        return l ? { ...c, w: l.w, h: l.h, fx: l.fx, fy: l.fy,
                     cx: l.cx, cy: l.cy, dist: l.d } : c;
      });
      this._buildCamViz(cams);
    } catch { /* no camera rig manifest — frusta unavailable */ }
    bus.emit('viewportReady', this);
  }

  _tick(){
    // VP tracking overlay: hide when the operator_points stream goes stale
    // (>1 s — VP disconnected / retarget stopped) instead of freezing a ghost.
    if (this.opOverlay && this.opOverlay.g.visible && this._opStamp
        && performance.now() - this._opStamp > 1000)
      this.opOverlay.g.visible = false;
    if (this.opOverlay && this.layers.vptrack && this._opStamp
        && performance.now() - this._opStamp <= 1000)
      this.opOverlay.g.visible = true;
    // Camera textures update continuously; retex the visible projections from
    // whichever surface CamView currently has live (H.264 canvas or MJPEG <img>).
    for (const name in this.camViz){
      const v = this.camViz[name];
      if (!v.group.visible || !v.view || !v.view.ready()) continue;
      const src = v.view.surface();
      if (v.tex.image !== src) v.tex.image = src;
      v.tex.needsUpdate = true;
    }
    this.stage.render();
  }

  _buildTF(){
    this.tf = {};                              // label -> triad (per-frame toggles)
    for (const [label, node, size] of TF_FRAMES){
      const n = this.rig.root.getObjectByName(node);
      if (!n) continue;
      const g = triad(size, label); g.visible = false; n.add(g); this.tf[label] = g;
    }
  }
  tfFrames(){ return Object.keys(this.tf || {}); }
  tfOn(label){ return !!this.tf?.[label]?.visible; }
  setTfFrame(label, on){
    const g = this.tf?.[label]; if (!g) return;
    g.visible = on;
    this.layers.tf = Object.values(this.tf).some(t => t.visible);
  }
  camOn(name){ return this.activeCams.has(name); }
  toggleCam(name, on){
    if (on) this.activeCams.add(name); else this.activeCams.delete(name);
    this.layers.frustum = this.layers.frustum || on;
    this._applyCamVis();
  }

  // ── nav goal: green floor ring + heading arrow. Click-drag places it, and
  // the placed marker stays INTERACTIVE: drag the ring/dot to move it, drag
  // the arrow to re-aim it. Active handles glow + a yaw dial appears while
  // rotating, so it always reads what the drag is doing.
  _buildNavTarget(){
    const g = new THREE.Group();
    const C = 0x4dff8a;
    const mat = () => new THREE.MeshBasicMaterial({ color: C });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.014, 10, 40), mat());
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), mat());
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0.015), 0.34, C, 0.12, 0.09);
    // fat invisible grab pads so the thin visuals are easy to hit
    const movePad = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.06, 16).rotateX(Math.PI / 2),
      new THREE.MeshBasicMaterial({ visible: false }));
    const rotPad = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.16, 0.1),
      new THREE.MeshBasicMaterial({ visible: false }));
    rotPad.position.set(0.28, 0, 0.02);
    // yaw dial: faint full circle shown only while rotating
    const dial = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.006, 8, 48),
      new THREE.MeshBasicMaterial({ color: C, transparent: true, opacity: 0.35 }));
    dial.visible = false;
    g.add(ring, dot, arrow, movePad, rotPad, dial);
    g.visible = false;
    this.navWorld.add(g);
    this.navMarker = g;
    this._nm = { ring, dot, arrow, movePad, rotPad, dial, yaw: 0 };
  }
  // drag-state feedback: scale + brighten what the pointer is manipulating
  _goalFeedback(mode){
    const m = this._nm; if (!m) return;
    const hot = 0x9dffc0, base = 0x4dff8a;
    m.ring.material.color.set(mode === 'move' ? hot : base);
    m.dot.material.color.set(mode === 'move' ? hot : base);
    m.ring.scale.setScalar(mode === 'move' ? 1.25 : 1);
    m.arrow.setColor(mode === 'rotate' ? hot : base);
    m.dial.visible = mode === 'rotate';
    this.el.style.cursor = mode === 'move' ? 'grabbing'
      : mode === 'rotate' ? 'ew-resize'
      : this._navArm ? GOAL_CURSOR : '';
  }
  _setGoalYaw(yaw){
    this._nm.yaw = yaw;
    this.navMarker.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);
  }
  // dropping / adjusting a goal only PREVIEWS (Nav2 plans the path); driving
  // starts when the operator presses GO (app-side, reads getGoal())
  _emitGoal(final){
    const p = this.navMarker.position;
    this.onNavPreview && this.onNavPreview(p.x, p.y, this._nm.yaw);
    if (final && this.onGoalPlaced) this.onGoalPlaced(this.getGoal());
  }
  getGoal(){
    if (!this.navMarker?.visible) return null;
    const p = this.navMarker.position;
    return { x: p.x, y: p.y, yaw: this._nm.yaw };
  }
  // freeze/unfreeze live map ingestion (SLAM disabled ⇒ the view holds still)
  setMapFrozen(on){ this.mapFrozen = on; }
  clearMap(){
    this.navWorld.remove(this._meshLive);
    this._meshLive.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    this._meshLive = new THREE.Group();
    this._meshLive.visible = this.layers.mesh;
    this.navWorld.add(this._meshLive);
    this._meshPending = new THREE.Group();
    if (this.occPlane){ this.navWorld.remove(this.occPlane); this.occPlane.geometry.dispose(); this.occPlane = null; }
  }

  // ── wrist targets: a draggable green ball at the ORCA WRIST JOINT
  // (`<side>_palm__anchor`), plus roll/pitch/yaw ORIENTATION RINGS around it.
  // Drag the ball to move the wrist (position IK); drag a ring to spin the palm
  // about that axis (the orientation half of the same 6-DOF task).
  //
  // The rings were removed 2026-07-19 to keep task space to one handle. They are
  // back because the iOS and Vision Pro apps kept theirs, and an operator moving
  // between the three surfaces found the same green ball did LESS here with no
  // way to tell. Colours and planes match the apps exactly
  // (RealRobotScene.addOrientationRings): roll blue XZ, pitch orange XY, yaw red YZ.
  _buildWristTargets(){
    this.wrist = {};
    // THE EULERS ARE NOT THE APP'S, AND CANNOT BE. The planes and colours match
    // RealRobotScene.addOrientationRings exactly, but the numbers that produce them do not,
    // because the two toolkits start from different toruses: SceneKit's SCNTorus lies in the
    // XZ plane with its axis along +Y, three.js's TorusGeometry lies in the XY plane with its
    // axis along +Z. Copying the Swift triples across gave `yaw` a rotation ABOUT ITS OWN AXIS,
    // which does nothing — so yaw landed exactly on top of roll and the operator saw two rings
    // where there should be three, with no way to turn the wrist about the third direction. It
    // also swapped roll and pitch against ringAxis() below, so two of the three dragged about
    // the wrong axis. Each triple here is the rotation that takes three's +Z axis to the axis
    // named in the comment; ringAxis() is the same axis again and the two must agree —
    // in SIGN as well as direction, or the ring turns the opposite way to the drag.
    const RINGS = [
      ['roll',  0x4f9dff, [-Math.PI / 2, 0, 0]],      // XZ plane, axis +Y
      ['pitch', 0xe7913a, [0, 0, 0]],                 // XY plane, axis +Z
      ['yaw',   0xe0503d, [0, Math.PI / 2, 0]],       // YZ plane, axis +X
    ];
    for (const side of ['left', 'right']){
      const grp = new THREE.Group();
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.032, 18, 18),
        new THREE.MeshBasicMaterial({ color: 0x4dff8a, transparent: true, opacity: 0.8, depthTest: false }));
      ball.renderOrder = 998;
      grp.add(ball);
      const rings = {};
      for (const [axis, color, euler] of RINGS){
        const r = new THREE.Mesh(
          new THREE.TorusGeometry(0.075, 0.0055, 8, 64),
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85, depthTest: false }));
        r.rotation.set(euler[0], euler[1], euler[2]);
        r.renderOrder = 997;
        r.userData.axis = axis;
        // A fatter invisible torus for picking: a 5.5 mm tube is honest to look at
        // and miserable to hit with a finger or a trackpad.
        const pick = new THREE.Mesh(
          new THREE.TorusGeometry(0.075, 0.018, 6, 32),
          new THREE.MeshBasicMaterial({ visible: false }));
        pick.rotation.copy(r.rotation);
        pick.userData.axis = axis;
        grp.add(r, pick);
        rings[axis] = { mesh: r, pick };
      }
      grp.visible = false;
      this.stage.scene.add(grp);
      this.wrist[side] = { grp, ball, rings, side };
    }
  }

  /// Local axis each ring spins about, matching the euler set above.
  static ringAxis(axis){
    return axis === 'roll'  ? new THREE.Vector3(0, 1, 0)
         : axis === 'pitch' ? new THREE.Vector3(0, 0, 1)
                            : new THREE.Vector3(1, 0, 0);
  }

  /// Pointer angle around `axisWorld` in the plane through `centre`. Returns null
  /// when the ray is parallel to the plane (edge-on ring), which is exactly when a
  /// rotation drag would be ill-conditioned anyway.
  _ringAngle(ray, centre, axisWorld){
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(axisWorld, centre);
    const hit = new THREE.Vector3();
    if (!ray.ray.intersectPlane(plane, hit)) return null;
    const v = hit.sub(centre);
    if (v.lengthSq() < 1e-8) return null;
    // Any stable basis in the plane; only the DELTA between two angles is used.
    const u = new THREE.Vector3(1, 0, 0);
    if (Math.abs(u.dot(axisWorld)) > 0.9) u.set(0, 1, 0);
    const e1 = u.clone().sub(axisWorld.clone().multiplyScalar(u.dot(axisWorld))).normalize();
    const e2 = new THREE.Vector3().crossVectors(axisWorld, e1);
    return Math.atan2(v.dot(e2), v.dot(e1));
  }
  // snap targets onto the current WRIST JOINT poses (world / three.js frame)
  syncWristTargets(){
    for (const side of ['left', 'right']){
      const anchor = this.rig?.root.getObjectByName(`${side}_palm__anchor`)
        || this.rig?.root.getObjectByName(side === 'left' ? 'left_palm' : 'right_palm');
      if (!anchor) continue;
      anchor.getWorldPosition(this.wrist[side].grp.position);
      anchor.getWorldQuaternion(this.wrist[side].grp.quaternion);
    }
  }
  // drag feedback: the grabbed ring thickens and goes opaque, the others dim, so
  // it is obvious WHICH axis is being turned mid-drag.
  _ringFeedback(side, axis){
    const w = this.wrist[side]; if (!w || !w.rings) return;
    for (const a in w.rings){
      const m = w.rings[a].mesh;
      const on = axis === a;
      m.material.opacity = axis ? (on ? 1.0 : 0.25) : 0.85;
      m.scale.setScalar(on ? 1.08 : 1);
    }
  }

  // drag feedback: the grabbed ball glows
  _wristFeedback(side, active){
    const w = this.wrist[side]; if (!w) return;
    w.ball.material.color.set(active === 'ball' ? 0xd2ffe4 : 0x4dff8a);
    w.ball.material.opacity = 0.8;
    w.ball.scale.setScalar(active === 'ball' ? 1.15 : 1);
  }
  setWristVisible(on){
    this.layers.wrist = on;
    if (on) this.syncWristTargets();
    for (const side in this.wrist) this.wrist[side].grp.visible = on;
  }

  // ── layer switch ────────────────────────────────────────────────────────────
  setLayer(name, on){
    this.layers[name] = on;
    if (name === 'grid') this.stage.grid.visible = on;
    if (name === 'robot' && this.rig) this.rig.world.visible = on;
    if (name === 'tf') Object.values(this.tf || {}).forEach(g => g.visible = on);
    if (name === 'mesh') this._meshLive.visible = on;
    if (name === 'occupancy' && this.occPlane) this.occPlane.visible = on;
    if (name === 'path' && this.pathLine) this.pathLine.visible = on;
    if (name === 'lidar' && this.lidarCloud) this.lidarCloud.visible = on;
    if (name === 'cloud')
      for (const s in this.clouds || {})
        this.clouds[s].visible = on && this.cloudSourceOn(s);
    if (name === 'navgoal' && this.navMarker && !on) this.navMarker.visible = false;
    if (name === 'wrist') this.setWristVisible(on);
    if (name === 'vptrack' && this.opOverlay) this.opOverlay.g.visible = on;
    if (name === 'frustum') this._applyCamVis();
    if (name === 'traj') this._drawTraj();
  }
  setPointSize(s){
    this.pointSize = s;
    if (this.lidarCloud) this.lidarCloud.material.size = s;
    for (const k in this.clouds || {}) this.clouds[k].material.size = s;
  }
  setCamProjDistance(d){ this.camProjSize = Math.max(0.05, d); this._rebuildCamViz(); }
  setCamFlat(on){ this.camFlat = on; this._rebuildCamViz(); }
  setActiveCams(set){ this.activeCams = new Set(set); this._applyCamVis(); }
  armNavGoal(){ this._navArm = true; this.el.style.cursor = GOAL_CURSOR; }

  // ── live robot state (50 Hz) ────────────────────────────────────────────────
  updateRobot(state){
    if (!this.rig || !state) return;
    const q = state.jointPositions || {};
    const b = state.base || {};
    const yaw = b.yaw ?? 0;
    this._haveRobotState = true;
    this._lastBase = { bx: b.x || 0, by: b.y || 0, yaw };
    this._liveQ = q;                 // cached for the ghost preview + IK restore
    // timeline preview owns the rig: keep tracking live state, don't pose it
    if (this.previewing) return;
    // The SOLID robot ALWAYS tracks the live/measured pose.
    this.rig.pose(q, b.x || 0, b.y || 0, yaw);
    // seed the IK's angle cache from the mirror unless a drag is solving now
    if (!this._drag){
      this.rig._q = this.rig._q || {};
      for (const n in q) this.rig._q[n] = q[n];
    }
    // TASK-SPACE PLAN (MoveIt-style): the plan is shown as a translucent GHOST,
    // NOT by moving the solid robot — so you see current (solid) vs planned
    // (ghost) and press Execute to commit. Re-pose the ghost each tick so its
    // base and un-planned joints track live while the planned arm joints stay.
    if (this._planHold) this._poseGhostPlan();
    // Keep the draggable wrist ball on the LIVE wrist when idle. During a drag
    // it follows the pointer; while a plan is pending it stays parked on the
    // planned target (so the marker shows what you are about to execute).
    if (this.layers.wrist && !this._drag && !this._planHold) this.syncWristTargets();
    this._applyFollow();
  }

  // ── task-space plan preview: the MoveIt orange ghost ─────────────────────
  // The plan is the accumulated IK solution from dragging the wrist ball. It
  // lives ONLY in the viewer (on the ghost rig) until Execute.
  // The ghost ANIMATES the plan rather than standing at its endpoint. A static
  // ghost at the final pose answers "where will it end up" but not "how does it
  // get there" — and on a 7-DOF arm those are different questions: an IK
  // solution can reach the same wrist pose by swinging the elbow through the
  // torso. The ghost sweeps live → planned on a raised cosine, holds briefly,
  // and loops, so the path is visible before anything is committed.
  //
  // While a drag is in progress the ghost stays pinned at the target (u = 1):
  // the operator is *authoring* the pose then, and a ghost animating out from
  // under the ball they are holding reads as lag, not as a preview.
  _poseGhostPlan(){
    if (!this._planHold) return;
    const live = this._liveQ || {};
    let u = 1;
    const A = this._planAnim;
    if (A && !this._drag){
      const ph = ((performance.now() - A.t0) / 1000) % (A.dur + PLAN_LOOP_HOLD_S);
      u = ph >= A.dur ? 1 : PLAN_EASE(ph / A.dur);
    }
    const q = { ...live };
    // Joints covered by the COLLISION-AVOIDED path animate ALONG it (the CBF
    // flow slides around the torso/head/other arm — a straight joint-space
    // lerp would sweep through the very collision the plan avoids); everything
    // else keeps the straight eased lerp.
    const onPath = this._samplePlanPath(u);
    for (const j in this._planHold){
      if (onPath && j in onPath){ q[j] = onPath[j]; continue; }
      const a = live[j], b = this._planHold[j];
      q[j] = (a == null || u >= 1) ? b : a + (b - a) * u;
    }
    this.poseGhost(q);
  }
  // Interpolate the stored collision-avoided path at parameter u ∈ [0,1]
  // (piecewise-linear between waypoints). Null when no path is staged.
  _samplePlanPath(u){
    const P = this._planPath;
    if (!P || P.length < 2) return null;
    const s = Math.min(1, Math.max(0, u)) * (P.length - 1);
    const i = Math.min(P.length - 2, Math.floor(s));
    const f = s - i;
    const out = {};
    for (const j in P[i]) out[j] = P[i][j] + (P[i + 1][j] - P[i][j]) * f;
    return out;
  }
  // The staged collision-avoided waypoint path ({name: rad} list), or null.
  // app.js's eased Execute streams ALONG this instead of a straight lerp, so
  // the robot performs the same avoidance the ghost previewed.
  planPathJoints(){
    return this._planPath ? this._planPath.map((wp) => ({ ...wp })) : null;
  }
  setPlanHold(joints){
    this._planHold = { ...(this._planHold || {}), ...joints };
    // Do NOT restart the loop on every IK solve — stage() fires continuously
    // while dragging, so re-seeding t0 here would keep u pinned near 0 and the
    // ghost would never leave the live pose. The loop starts on release.
    if (!this._drag && !this._planAnim) this.startPlanAnim();
    this._poseGhostPlan();
  }
  // The last MEASURED joint pose, as {name: rad}. The eased wrist execute
  // needs a start pose to interpolate from, and it must be the measured one:
  // easing from a commanded/target pose would begin the ramp wherever the
  // robot was *told* to be, which is exactly the discrepancy the ease exists
  // to cross safely.
  liveJoints(){ return { ...(this._liveQ || {}) }; }

  // (re)start the looping preview — called when a wrist drag releases
  startPlanAnim(){
    if (!this._planHold) return;
    const live = this._liveQ || {};
    let d = 0;
    for (const j in this._planHold){
      const a = live[j];
      if (a != null) d = Math.max(d, Math.abs(this._planHold[j] - a));
    }
    // long enough to read, short enough not to feel like a wait
    const dur = Math.min(3.0, Math.max(0.9, d / 0.9));
    this._planAnim = { t0: performance.now(), dur };
    // The plan preview must keep moving even when robot_state stalls (the tick
    // that normally re-poses the ghost) — otherwise the animation freezes
    // exactly when the operator is staring at it.
    clearInterval(this._planTimer);
    this._planTimer = setInterval(() => this._poseGhostPlan(), 33);
  }
  clearPlanHold(){
    this._planHold = null;
    this._planAnim = null;
    this._planPath = null;           // drop the collision-avoided path with it
    this._planBlocked = false;
    clearInterval(this._planTimer);
    this._planTimer = null;
    this.poseGhost(null);            // hide the ghost
  }

  // ── choreography preview: the MAIN rig plays the timeline (no wire) ───────
  // beginPreview() freezes live-state posing; previewPose() drives the rig
  // with timeline joints + the integrated nav base path; endPreview(fade)
  // hands the rig back to robot_state — with a short veil fade so the snap
  // from preview pose to the real robot reads as an intentional transition.
  beginPreview(){
    if (this.previewing) return;
    this.previewing = true;
    this._previewBase0 = { ...this._lastBase };
    if (this.ghost) this.ghost.world.visible = false;
    this._previewBadge(true);
  }
  previewPose(pose, base){
    if (!this.rig) return;
    if (!this.previewing) this.beginPreview();
    const b0 = this._previewBase0 || this._lastBase;
    const b = base
      ? { bx: b0.bx + base.dx, by: b0.by + base.dy, yaw: b0.yaw + base.dyaw }
      : b0;
    this.rig.pose(pose || {}, b.bx, b.by, b.yaw);
  }
  endPreview(fade = false){
    if (!this.previewing) return;
    this.previewing = false;
    this._previewBadge(false);
    if (fade) this._fadeFlash();
    // next robot_state re-poses the rig; force it now if we have one
    if (this._lastBase && this.rig){
      const { bx, by, yaw } = this._lastBase;
      this.rig.pose({}, bx, by, yaw);
    }
  }
  _previewBadge(on){
    let v = this.el.querySelector('.vp-preview-veil');
    if (!v){
      v = document.createElement('div');
      v.className = 'vp-preview-veil mono';
      v.textContent = 'TIMELINE PREVIEW — not commanding the robot';
      this.el.appendChild(v);
    }
    v.classList.toggle('on', on);
  }
  _fadeFlash(){
    let f = this.el.querySelector('.vp-fadeflash');
    if (!f){
      f = document.createElement('div');
      f.className = 'vp-fadeflash';
      this.el.appendChild(f);
    }
    f.classList.remove('run');
    void f.offsetWidth;                       // restart the animation
    f.classList.add('run');
  }

  // ── live LiDAR scan (from /ws/telemetry sidecar: ranges[], angle step 1°) ──
  updateLidar(ranges){
    if (!this.layers.lidar || !ranges) return;
    const { bx, by, yaw } = this._lastBase, c = Math.cos(yaw), s = Math.sin(yaw);
    const pts = [], cols = [];
    for (let i = 0; i < ranges.length; i++){
      const r = ranges[i]; if (!r || r <= 0.05) continue;
      const th = (i / ranges.length) * 2 * Math.PI;
      const lx = r * Math.cos(th), ly = r * Math.sin(th);
      pts.push(bx + lx * c - ly * s, by + lx * s + ly * c, LIDAR_Z);
      const [cr, cg, cb] = turbo(r / 8); cols.push(cr, cg, cb);
    }
    if (this.lidarCloud){ this.navWorld.remove(this.lidarCloud); this.lidarCloud.geometry.dispose(); }
    this.lidarCloud = this._points(new Float32Array(pts), new Float32Array(cols), this.pointSize * 2);
    this.lidarCloud.visible = this.layers.lidar; this.navWorld.add(this.lidarCloud);
  }

  // ── map plane records (ws :9092) ───────────────────────────────────────────
  onMesh(rec){
    if (this.mapFrozen) return;
    // chunked: accumulate into pending, swap on final
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(rec.pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(rec.col, 3));
    let obj;
    if (rec.idx && rec.idx.length){
      geo.setIndex(new THREE.BufferAttribute(rec.idx, 1));
      geo.computeVertexNormals();
      // FrontSide on purpose: nvblox normals face INTO the room, so the shell
      // is transparent from outside and shaded from within (rviz behavior) —
      // DoubleSide turns the room into a black box that occludes the robot.
      obj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial(
        { vertexColors: true, side: THREE.FrontSide, roughness: 0.92 }));
    } else {
      obj = new THREE.Points(geo, new THREE.PointsMaterial(
        { size: this.pointSize, vertexColors: true, sizeAttenuation: true }));
    }
    this._meshPending.add(obj);
    if (rec.final){
      this.navWorld.remove(this._meshLive);
      this._meshLive.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      this._meshLive = this._meshPending;
      this._meshLive.visible = this.layers.mesh;
      this.navWorld.add(this._meshLive);
      this._meshPending = new THREE.Group();
    }
  }
  // multi-source clouds: 'live' = the bridge's map-plane cloud; 'head'/'front'
  // = per-camera depth clouds from the backend stream hub
  onCloud(rec, source = 'live'){
    this.clouds = this.clouds || {};
    const old = this.clouds[source];
    if (old){ this.navWorld.remove(old); old.geometry.dispose(); }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(rec.pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(rec.col, 3));
    const obj = new THREE.Points(geo, new THREE.PointsMaterial(
      { size: this.pointSize, vertexColors: true, sizeAttenuation: true }));
    obj.visible = this.layers.cloud && this.cloudSourceOn(source);
    this.clouds[source] = obj;
    this.navWorld.add(obj);
    this.liveCloud = this.clouds.live || null;    // legacy alias
  }
  cloudSourceOn(source){
    this._cloudSources = this._cloudSources || new Set(['live']);
    return this._cloudSources.has(source);
  }
  setCloudSource(source, on){
    this._cloudSources = this._cloudSources || new Set(['live']);
    if (on) this._cloudSources.add(source); else this._cloudSources.delete(source);
    if (on) this.layers.cloud = true;
    for (const s in this.clouds || {})
      this.clouds[s].visible = this.layers.cloud && this._cloudSources.has(s);
  }
  clearNavGoal(){
    if (this.navMarker) this.navMarker.visible = false;
  }
  // the transform tree, straight from the rig's scene graph (= the TF tree)
  getTfTree(){
    if (!this.rig) return null;
    const round = (v) => Math.round(v * 1000) / 1000;
    const isJoint = new Set(Object.values(this.rig.joints).map(j => j._n.name));
    const build = (o) => {
      const children = [];
      for (const c of o.children){
        if (c.isMesh || c.isLight || c.isSprite) continue;
        if (c.name) children.push(build(c));
        else children.push(...build(c).children);
      }
      return { name: o.name || '(root)',
               pos: [round(o.position.x), round(o.position.y), round(o.position.z)],
               joint: isJoint.has(o.name), children };
    };
    return build(this.rig.root);
  }
  onPath(rec){
    if (this.pathLine){ this.navWorld.remove(this.pathLine); this.pathLine.geometry.dispose(); this.pathLine = null; }
    if (!rec.count) return;
    const pts = [];
    for (let i = 0; i < rec.count; i++)
      pts.push(new THREE.Vector3(rec.pts[i * 3], rec.pts[i * 3 + 1], 0.03));
    this.pathLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xC25B2A }));
    this.pathLine.visible = this.layers.path;
    this.navWorld.add(this.pathLine);
  }
  onGrid(rec){
    if (this.mapFrozen) return;
    if (this.occPlane){ this.navWorld.remove(this.occPlane); this.occPlane.geometry.dispose(); this.occPlane.material.map?.dispose(); this.occPlane = null; }
    const { w, h, res, ox, oy, yaw, cells } = rec;
    const data = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++){
      const v = cells[i];
      let r, g, b, a;
      if (v === 255){ r = g = b = 60; a = 40; }              // unknown: faint
      else { const k = 1 - v / 100; r = g = b = 40 + 180 * k; a = 200; }
      data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = a;
    }
    const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
    tex.needsUpdate = true;
    const geo = new THREE.PlaneGeometry(w * res, h * res);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial(
      { map: tex, transparent: true, depthWrite: false }));
    // grid origin = corner cell (0,0); plane geometry is centered
    m.position.set(ox + (w * res) / 2, oy + (h * res) / 2, 0.012);
    m.rotation.z = yaw;
    m.visible = this.layers.occupancy;
    this.occPlane = m; this.navWorld.add(m);
  }
  // ── VP tracking overlay (retarget debug) ─────────────────────────────────
  // Two families, one 'vptrack' layer, both in the SAME frame as the robot by
  // construction — the group carries rig.baseQ0, the GLB base_link's one
  // Z-up→Y-up rotation, i.e. EXACTLY the transform that places the robot from
  // odom, so no new frame math exists to get wrong:
  //  · MAPPED (magenta R / cyan L): the engine's own wrist target, elbow hint,
  //    palm axes and hand skeleton — the precise quantities the arm IK
  //    consumed. When tracking is perfect the target ball sits ON the robot's
  //    wrist; any gap on screen IS the tracking/retargeting error.
  //  · RAW (amber): the unprocessed VP points — head + both full hand
  //    skeletons — through nothing but the fixed AVP_TO_ROBOT axis relabel.
  //    True metric scale, floating at the ARKit session origin: what the
  //    Vision Pro app itself displays.
  onOperatorPoints(rec){
    if (!this.rig) return;
    this._opStamp = performance.now();     // staleness fade (see _tick)
    if (!this.opOverlay){
      const g = new THREE.Group();
      g.quaternion.copy(this.rig.baseQ0);        // odom Z-up → scene Y-up
      g.visible = !!this.layers.vptrack;
      this.navWorld.add(g);
      const ball = (color, r) => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(r, 14, 14),
          new THREE.MeshBasicMaterial({ color, transparent: true,
                                        opacity: 0.85, depthTest: false }));
        m.renderOrder = 997; m.visible = false; g.add(m); return m;
      };
      const pts = (color, n, size) => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position',
          new THREE.BufferAttribute(new Float32Array(n * 3), 3));
        geo.setDrawRange(0, 0);
        const p = new THREE.Points(geo, new THREE.PointsMaterial(
          { color, size, depthTest: false, sizeAttenuation: true }));
        p.renderOrder = 996; p.visible = false; g.add(p); return p;
      };
      const line = (color) => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position',
          new THREE.BufferAttribute(new Float32Array(2 * 3), 3));
        const l = new THREE.Line(geo,
          new THREE.LineBasicMaterial({ color, depthTest: false }));
        l.renderOrder = 996; l.visible = false; g.add(l); return l;
      };
      const side = (c, cp) => ({ ball: ball(c, 0.024), elbow: ball(c, 0.014),
        pts: pts(cp, 32, 0.014), fwd: line(0x4dff8a), up: line(0x4d9dff) });
      this.opOverlay = { g,
        sides: { right: side(0xff4dd2, 0xffb3ea), left: side(0x4dd2ff, 0xb3ecff) },
        raw: { head: ball(0xf0b68b, 0.05),
               right: pts(0xf0b68b, 32, 0.012), left: pts(0xf0b68b, 32, 0.012) } };
    }
    const setPts = (p, list) => {
      const arr = p.geometry.attributes.position;
      const n = Math.min(list?.length || 0, 32);
      for (let i = 0; i < n; i++)
        arr.setXYZ(i, list[i][0], list[i][1], list[i][2]);
      p.geometry.setDrawRange(0, n);
      arr.needsUpdate = true;
      return n > 0;
    };
    for (const s of ['right', 'left']){
      const o = this.opOverlay.sides[s];
      const d = rec?.sides?.[s];
      o.ball.visible = o.pts.visible = o.fwd.visible = o.up.visible = !!d;
      o.elbow.visible = !!(d && d.elbow);
      if (!d) continue;
      o.ball.position.set(...d.target);
      if (d.elbow) o.elbow.position.set(...d.elbow);
      setPts(o.pts, d.hand);
      const ax = (l, dir) => {
        const a = l.geometry.attributes.position;
        a.setXYZ(0, d.target[0], d.target[1], d.target[2]);
        a.setXYZ(1, d.target[0] + dir[0] * 0.08, d.target[1] + dir[1] * 0.08,
                 d.target[2] + dir[2] * 0.08);
        a.needsUpdate = true;
      };
      if (d.fwd) ax(o.fwd, d.fwd);
      if (d.up) ax(o.up, d.up);
    }
    const raw = rec?.raw || {};
    this.opOverlay.raw.head.visible = !!raw.head;
    if (raw.head) this.opOverlay.raw.head.position.set(...raw.head);
    for (const s of ['right', 'left'])
      this.opOverlay.raw[s].visible = setPts(this.opOverlay.raw[s], raw[s]);
  }

  onOdom(rec){ /* base pose already comes from robot_state; kept for map-only viewing */
    if (!this.rig || this._haveRobotState) return;
    const yaw = Math.atan2(2 * (rec.qw * rec.qz + rec.qx * rec.qy),
                           1 - 2 * (rec.qy * rec.qy + rec.qz * rec.qz));
    this.rig.pose({}, rec.x, rec.y, yaw);
    this._lastBase = { bx: rec.x, by: rec.y, yaw };
  }

  // ── camera frusta + live projections, attached to their mount links ───────
  _buildCamViz(cameras){
    this._camRig = cameras;
    for (const v of Object.values(this.camViz)){
      v.view?.stop();        // drop the stream too, or a rebuild leaks it
      v.group.parent?.remove(v.group);
      v.group.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    }
    this.camViz = {};
    for (const [stream, rigName] of Object.entries(CAM_TO_RIG)){
      const c = cameras.find?.(x => x.name === rigName) || cameras[rigName];
      if (!c) continue;
      const mount = this.rig?.root.getObjectByName(c.mount_body)
        || this.rig?.root.getObjectByName(MOUNT_ALIAS[c.mount_body] || '');
      if (!mount) continue;
      // uniform IMAGE SIZE across cameras: every projected image is
      // camProjSize meters tall, so the per-camera distance follows the TRUE
      // vertical FOV (fy) — wide lenses sit close, narrow lenses sit farther,
      // and aspect = w/h exactly (true to resolution + intrinsics)
      const halfFovY = Math.atan((c.h / 2) / c.fy);
      const D = (this.camProjSize / 2) / Math.tan(halfFovY);
      const flat = this.camFlat;
      const grp = new THREE.Group();
      const A = new THREE.Vector3(0, 0, 0);
      const TL = pixelPoint(c, 0, 0, D, flat), TR = pixelPoint(c, c.w, 0, D, flat);
      const BR = pixelPoint(c, c.w, c.h, D, flat), BL = pixelPoint(c, 0, c.h, D, flat);
      const segs = [A, TL, A, TR, A, BR, A, BL, TL, TR, TR, BR, BR, BL, BL, TL];
      grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs),
        new THREE.LineBasicMaterial({ color: 0x4d9dff })));
      const topMid = pixelPoint(c, c.w / 2, 0, D, flat);
      grp.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(
        [topMid, topMid.clone().multiplyScalar(1.25)]),
        new THREE.LineBasicMaterial({ color: 0x4dff8a })));
      // Two possible pixel sources for the projection: the H.264 canvas (fast
      // path) or the MJPEG <img> (fallback). CamView picks; _tick re-points the
      // texture at whichever is live. Both are same-origin, so neither taints.
      const img = new Image(); img.crossOrigin = 'anonymous';
      const canvas = document.createElement('canvas');
      const tex = new THREE.Texture(img); tex.colorSpace = THREE.SRGBColorSpace;
      img.onload = () => { tex.needsUpdate = true; };
      const mesh = new THREE.Mesh(imageMesh(c, D, flat),
        new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.97 }));
      grp.add(mesh);
      const lab = labelSprite(stream, '#4d9dff'); lab.scale.multiplyScalar(0.55); lab.position.copy(TL); grp.add(lab);
      // mount: pos + quat relative to the parent link (MJCF/URDF frame, z-up →
      // camera looks along −Z in camera frame; the rig json quats already encode it)
      grp.position.set(...(c.mount_pos || [0, 0, 0]));
      if (c.mount_quat){
        // MuJoCo quats are (w,x,y,z), body<-cam (camcloud.py:514); the camera
        // frame itself matches three.js: look along −Z, +Y up
        const [qw, qx, qy, qz] = c.mount_quat;
        grp.quaternion.set(qx, qy, qz, qw);
      }
      grp.visible = false;
      mount.add(grp);
      this.camViz[stream] = { group: grp, img, canvas, tex, calib: c, view: null };
    }
    this._applyCamVis();
  }
  _rebuildCamViz(){ if (this._camRig) this._buildCamViz(this._camRig); }
  _applyCamVis(){
    const on = this.layers.frustum;
    for (const name in this.camViz){
      const v = this.camViz[name];
      const vis = on && this.activeCams.has(name);
      // LIVE stream into the projection texture — H.264 when available, MJPEG
      // otherwise; `proxied` keeps the fallback <img> same-origin so WebGL can
      // texture it. Streams are torn down when the layer is hidden.
      if (vis && !v.view){
        v.view = new CamView(v.img, name, { canvas: v.canvas, proxied: true });
      } else if (!vis && v.view){
        v.view.stop(); v.view = null;
      }
      v.group.visible = vis;
    }
  }

  // ── pointer: nav-goal click-drag + wrist ball dragging ─────────────────────
  _wirePointer(){
    const ray = new THREE.Raycaster();
    const ndc = (e) => {
      const r = this.stage.canvas.getBoundingClientRect();
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                               -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    const floorHit = (e) => {
      ray.setFromCamera(ndc(e), this.stage.camera);
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const p = new THREE.Vector3();
      if (!ray.ray.intersectPlane(plane, p)) return null;
      // three.js world → odom/navWorld frame
      return this.navWorld.worldToLocal(p.clone());
    };
    this.stage.canvas.addEventListener('pointerdown', (e) => {
      // wrist target: the ball is the only handle — position drag
      if (this.layers.wrist){
        ray.setFromCamera(ndc(e), this.stage.camera);
        for (const side in this.wrist){
          const w = this.wrist[side];
          if (!w.grp.visible) continue;
          // Rings first: they surround the ball, so testing the ball first would
          // swallow every grab aimed at a ring edge near the centre.
          let ringHit = null;
          for (const axis in w.rings){
            const h = ray.intersectObject(w.rings[axis].pick, false);
            if (h.length && (!ringHit || h[0].distance < ringHit.d)) ringHit = { axis, d: h[0].distance };
          }
          if (ringHit){
            const centre = w.grp.getWorldPosition(new THREE.Vector3());
            const axisWorld = Viewport3D.ringAxis(ringHit.axis)
              .applyQuaternion(w.grp.getWorldQuaternion(new THREE.Quaternion())).normalize();
            const a0 = this._ringAngle(ray, centre, axisWorld);
            if (a0 !== null){
              this._ringDrag = { side, axis: ringHit.axis, axisWorld, centre,
                                 a0, q0: w.grp.quaternion.clone() };
              this._dragQuat = this._palmQuat(side);
              bus.emit('wristGrab', side);
              this._ringFeedback(side, ringHit.axis);
              this.stage.controls.enabled = false;
              this.stage.canvas.setPointerCapture(e.pointerId);
              return;
            }
          }
          const hit = ray.intersectObject(w.ball, false);
          if (hit.length){
            this._drag = { side, dist: hit[0].distance };
            // LATCH the palm's current orientation: dragging the ball is a
            // TRANSLATION, so the hand must keep facing the way it faces. The
            // solver tracks this as the orientation half of a 6-DOF task, which
            // is what stops the 7-DOF null space from tumbling the wrist.
            this._dragQuat = this._palmQuat(side);
            bus.emit('wristGrab', side);
            this._wristFeedback(side, 'ball');
            this.stage.controls.enabled = false;
            this.stage.canvas.setPointerCapture(e.pointerId);
            return;
          }
        }
      }
      // placed marker: grab its handles (ring/dot pad = move, arrow pad = rotate)
      if (this.layers.navgoal && this.navMarker.visible && !this._navArm){
        ray.setFromCamera(ndc(e), this.stage.camera);
        const hitRot = ray.intersectObject(this._nm.rotPad, false).length;
        const hitMove = !hitRot && ray.intersectObject(this._nm.movePad, false).length;
        if (hitRot || hitMove){
          this._navDrag = { mode: hitRot ? 'rotate' : 'move' };
          this._goalFeedback(this._navDrag.mode);
          this.stage.controls.enabled = false;
          this.stage.canvas.setPointerCapture(e.pointerId);
          return;
        }
      }
      // ARMED (rviz-style): click places the goal and dragging aims it in one
      // gesture. Kept for `armNavGoal()` callers; the UI no longer arms.
      if (this._navArm && this.layers.navgoal){
        const p = floorHit(e);
        if (!p) return;
        this._navDrag = { mode: 'place', x: p.x, y: p.y };
        this.navMarker.position.set(p.x, p.y, 0.02);
        this._setGoalYaw(this._nm.yaw || 0);
        this.navMarker.visible = true;
        this._goalFeedback('rotate');
        this.stage.controls.enabled = false;
        this.stage.canvas.setPointerCapture(e.pointerId);
        return;
      }
      // UNARMED: the floor is always live — remember where this gesture began
      // so pointerup can tell a TAP (place a goal) from a DRAG (orbit the
      // camera). Without that split, arming was the only way to keep "click the
      // floor" from stealing every orbit; iOS gets it for free because a tap
      // and a pan are different gestures there.
      this._tap = { x: e.clientX, y: e.clientY, t: performance.now(),
                    id: e.pointerId };
    });
    this.stage.canvas.addEventListener('pointermove', (e) => {
      if (this._ringDrag){
        const r = this._ringDrag, w = this.wrist[r.side];
        ray.setFromCamera(ndc(e), this.stage.camera);
        const a = this._ringAngle(ray, r.centre, r.axisWorld);
        if (a !== null){
          // Spin about the world axis the ring lies in. Pre-multiplying applies the
          // delta in WORLD space, which is what the ring visually promises — a
          // post-multiply would spin about the palm's own drifting frame and the
          // handle would run away from the pointer.
          const dq = new THREE.Quaternion().setFromAxisAngle(r.axisWorld, a - r.a0);
          w.grp.quaternion.copy(r.q0).premultiply(dq);
          this._dragQuat = w.grp.quaternion.clone();
          this._solveArm(r.side);
        }
        return;
      }
      if (this._drag){
        const d = this._drag, w = this.wrist[d.side];
        ray.setFromCamera(ndc(e), this.stage.camera);
        const p = ray.ray.at(d.dist, new THREE.Vector3());
        w.grp.position.copy(p);
        this._solveArm(d.side);
        return;
      }
      if (this._navDrag){
        const p = floorHit(e);
        if (!p) return;
        const nd = this._navDrag, mp = this.navMarker.position;
        if (nd.mode === 'move'){
          this.navMarker.position.set(p.x, p.y, 0.02);
        } else {                       // 'rotate' | 'place': aim toward pointer
          const dx = p.x - mp.x, dy = p.y - mp.y;
          if (Math.hypot(dx, dy) > 1e-3) this._setGoalYaw(Math.atan2(dy, dx));
        }
        this._emitGoal(false);
        return;
      }
      // hover affordance over the placed marker
      if (this.layers.navgoal && this.navMarker.visible && !this._navArm){
        ray.setFromCamera(ndc(e), this.stage.camera);
        const overRot = ray.intersectObject(this._nm.rotPad, false).length;
        const overMove = !overRot && ray.intersectObject(this._nm.movePad, false).length;
        this.el.style.cursor = overRot ? 'ew-resize' : overMove ? 'grab' : '';
      }
    });
    this.stage.canvas.addEventListener('pointerup', (e) => {
      const tap = this._tap;
      this._tap = null;
      if (this._ringDrag){
        this._ringFeedback(this._ringDrag.side, null);
        this._ringDrag = null;
        this._dragQuat = null;
        this.stage.controls.enabled = true;
        this.startPlanAnim();
        return;
      }
      if (this._drag){
        this._wristFeedback(this._drag.side, null);
        this._drag = null;
        this._dragQuat = null;      // next grab re-latches from the live palm
        this.stage.controls.enabled = true;
        // the pose is authored — now play it back as a looping preview
        this.startPlanAnim();
        return;
      }
      if (this._navDrag){
        this._navDrag = null;
        this._navArm = false;
        this._goalFeedback(null);
        this.el.style.cursor = '';
        this.stage.controls.enabled = true;
        this._emitGoal(true);
        return;
      }
      // TAP ON THE FLOOR → drop a nav goal (iOS: "tap the floor to send a Nav2
      // goal"). Deliberately narrow, so it can never hijack a camera move or
      // fire while the operator is working on the robot itself:
      //   · it must be a tap — under 6 px of travel and under 400 ms
      //   · only in Drive space: a stray floor tap while you are posing a wrist
      //     in Task/Joint must not send the base across the room
      //   · only in 3rd person, and only with the Nav Goal layer on
      // Aiming afterwards is unchanged: drag the placed marker's ring.
      if (!tap || tap.id !== e.pointerId) return;
      if (!this.navTapEnabled || !this.layers.navgoal) return;
      const moved = Math.hypot(e.clientX - tap.x, e.clientY - tap.y);
      if (moved > 6 || performance.now() - tap.t > 400) return;
      const p = floorHit(e);
      if (!p) return;
      this.navMarker.position.set(p.x, p.y, 0.02);
      this._setGoalYaw(this._nm.yaw || 0);
      this.navMarker.visible = true;
      this._emitGoal(true);
    });
  }

  // damped-least-squares IK: drag ball → 7-DOF arm joint targets (throttled)
  // World-frame orientation of a palm anchor on the live rig.
  _palmQuat(side){
    const n = this.rig?.root?.getObjectByName(`${side}_palm__anchor`)
           || this.rig?.root?.getObjectByName(side === 'left' ? 'left_palm' : 'right_palm');
    if (!n) return undefined;
    n.updateWorldMatrix(true, false);
    return new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().extractRotation(n.matrixWorld));
  }

  _solveArm(side){
    if (!this.rig || !this.onJointTargets) return;
    const now = performance.now();
    if (now - (this._ikAt || 0) < 33) return;        // ≤30 Hz
    this._ikAt = now;
    // The CONTROL REGION widens the optimization chain (mirrors the server's
    // reach-assist scope): arm = 7 arm joints; upper_body = + torso + lift;
    // whole_body = + a base-follow assist toward out-of-reach targets.
    const region = this.controlRegion || 'arm';
    const arm = Array.from({ length: 7 }, (_, i) => `${side}_arm_${i + 1}`);
    let names = arm;
    if (region === 'upper_body' || region === 'whole_body'){
      const extra = ['torso', 'lift_lower', 'lift_upper']
        .filter((n) => this.rig.joints[n]);
      names = [...extra, ...arm];
    }
    const target = this.wrist[side].grp.position;
    const ee = this.rig.root.getObjectByName(`${side}_palm__anchor`)
      ? `${side}_palm__anchor`
      : (side === 'left' ? 'left_palm' : 'right_palm');
    // Seed from the LIVE measured pose, and resolve the 4-DOF redundancy with a
    // null-space pull toward REST_POSE plus a held palm orientation. Without
    // these the solution depended on the drag PATH, not the target, and the palm
    // tumbled through ~180 deg on a pure translation (the "hand twists into weird
    // configurations" report). See kinematics.js.
    const sol = solveIK(this.rig, names, ee, target,
      { seed: this._liveQ || undefined, rest: REST_POSE,
        targetQuat: this._dragQuat || undefined });
    if (sol){
      // the two lift stages are ONE actuator on the robot — command them equal
      if (sol.lift_lower !== undefined || sol.lift_upper !== undefined){
        const l = ((sol.lift_lower ?? sol.lift_upper) +
                   (sol.lift_upper ?? sol.lift_lower)) / 2;
        if (sol.lift_lower !== undefined) sol.lift_lower = l;
        if (sol.lift_upper !== undefined) sol.lift_upper = l;
      }
      // SELF-COLLISION AVOIDANCE (CBF, twin of the server's target filter): the
      // raw IK solution can put the hand inside the torso/head/other arm — the
      // server would refuse to execute that, so previewing it would lie. Plan
      // the collision-avoided path live → solution (keep tangential velocity,
      // remove the normal closing excess — the hand SLIDES along the body) and
      // stage the path's END pose; the ghost then animates ALONG the path, so
      // the preview shows exactly the avoidance the robot will perform.
      this._scGuard = this._scGuard || makeGuard(this.rig);
      const from = {};
      for (const n of names)
        from[n] = this._liveQ?.[n] ?? this.rig._q?.[n] ?? sol[n];
      const plan = this._scGuard.planPath(names, from, sol);
      this._planPath = plan.path.length > 1 ? plan.path : null;
      this._planBlocked = !!plan.blocked;
      const end = plan.path[plan.path.length - 1];
      for (const n of names) if (n in end) sol[n] = end[n];
      if (sol.lift_lower !== undefined && sol.lift_upper !== undefined){
        const l = (sol.lift_lower + sol.lift_upper) / 2;   // re-sync after clip
        sol.lift_lower = l; sol.lift_upper = l;
      }
      // solveIK left the SOLID rig posed at the solution — but the plan is
      // previewed on the GHOST, so restore the solid rig to the live pose
      // (the IK used it only as its kinematic engine). onJointTargets stages
      // the plan, which poses the ghost via setPlanHold. Without a live pose
      // (demo, no robot_state) we leave the solid rig at the solution and the
      // ghost overlaps it — still correct, just nothing to compare against.
      if (this._liveQ){
        const { bx, by, yaw } = this._lastBase;
        this.rig.pose(this._liveQ, bx, by, yaw);
      }
      this.onJointTargets(sol);
    }
    // whole body: creep the base toward targets beyond comfortable reach
    // (the client-side analog of the server's lazy base follower). The vector
    // is RE-EVALUATED every pump tick against the LIVE base pose
    // (refreshBaseAssist) so the creep stops in the deadband instead of
    // streaming the stale release-time vector forever.
    if (region === 'whole_body'){
      this._assistSide = side;
      this._computeBaseAssist(side);
    } else { this._assistSide = null; this.baseAssist = null; }
  }

  // recompute the whole-body base-follow vector from the CURRENT base pose;
  // called per input-pump tick (teleopctl) — the motion model matches the
  // server's lazy follower: deadband 0.15 m over a 0.65 m comfortable reach,
  // gain 0.8, speed cap 0.5 (units of joystick deflection)
  refreshBaseAssist(){
    if (!this._assistSide){ this.baseAssist = null; return; }
    this._computeBaseAssist(this._assistSide);
  }
  _computeBaseAssist(side){
    const t = this.wrist?.[side]?.grp;
    if (!t || !t.visible){ this.baseAssist = null; return; }
    const { bx, by, yaw } = this._lastBase || { bx: 0, by: 0, yaw: 0 };
    // target in three.js world (y-up): x, z ground plane; convert to robot
    // world x-y via the same mapping used everywhere (world x = tx, y = -tz)
    const tx = t.position.x, ty = -t.position.z;
    const dx = tx - bx, dy = ty - by;
    const exb = Math.cos(yaw) * dx + Math.sin(yaw) * dy;    // body x
    const eyb = -Math.sin(yaw) * dx + Math.cos(yaw) * dy;   // body y
    // chassis front is −x body; comfortable reach ≈ 0.65 m, deadband 0.15
    const fwd = -exb, left = -eyb;
    const dist = Math.hypot(fwd, left);
    const over = dist - 0.65;
    if (over > 0.15 && dist > 1e-6){
      const g = Math.min(0.5, over * 0.8);
      this.baseAssist = { ly: (fwd / dist) * g, lx: (-left / dist) * g };
    } else this.baseAssist = null;
  }

  // ── POV transition + viewer navigation ────────────────────────────────────
  // world pose of the head camera (the 'main' stream's frustum group); falls
  // back to the head link when the camera rig manifest is unavailable
  headCamPose(){
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
    let fov = 83.3;
    // ALWAYS the robot's head camera: the transition is the operator entering
    // the robot's eyes, regardless of which stream backs the fpv footage
    const v = this.camViz.main;
    if (v){
      v.group.getWorldPosition(pos);
      v.group.getWorldQuaternion(quat);
      // the fpv footage is object-fit:COVER — match the 3D camera to the SAME
      // center crop so the crossfade is pixel-consistent (no stretch): if the
      // canvas is wider than the image, the width fills → derive the visible
      // vertical FOV from the camera's TRUE horizontal FOV / canvas aspect
      const c = v.calib;
      const A = (this.stage.canvas.clientWidth || 16) /
                (this.stage.canvas.clientHeight || 9);
      const vfov = 2 * Math.atan((c.h / 2) / c.fy);
      const hfov = 2 * Math.atan((c.w / 2) / c.fx);
      fov = (A >= c.w / c.h)
        ? 2 * Math.atan(Math.tan(hfov / 2) / A) * 180 / Math.PI
        : vfov * 180 / Math.PI;
    } else if (this.rig){
      const h = this.rig.root.getObjectByName('head') || this.rig.base;
      h.getWorldPosition(pos);
      h.getWorldQuaternion(quat);
      pos.y += 0.05;
    }
    return { pos, quat, fov };
  }
  orbitPose(){
    const c = this.stage.camera;
    return { pos: c.position.clone(), quat: c.quaternion.clone(), fov: c.fov,
             target: this.stage.controls.target.clone() };
  }
  applyPov(p){
    const c = this.stage.camera;
    c.position.copy(p.pos);
    c.quaternion.copy(p.quat);
    if (p.fov && p.fov !== c.fov){ c.fov = p.fov; c.updateProjectionMatrix(); }
  }
  setControlsEnabled(on){ this.stage.controls.enabled = on; }

  // recenter: aim the orbit target at the robot, keep the camera where it is
  recenter(){
    if (!this.rig) return;
    const c = new THREE.Vector3();
    this.rig.base.getWorldPosition(c);
    c.y += 0.6;
    this.stage.controls.target.copy(c);
  }

  // ── module focus: the rail picks a module, the camera flies to frame it ──
  // (the iOS `focusToken` behaviour). We keep the operator's current viewing
  // DIRECTION and only dolly in on the new center — a camera that teleports
  // around the robot is disorienting, and re-orienting after every rail tap is
  // exactly the friction the rail exists to remove.
  focusModule(id){
    const m = moduleById(id);
    if (!m || !this.rig || this.follow) return;
    const j = this.rig.joints[m.focus];
    const node = j?._n || this.rig.root.getObjectByName(m.focus);
    if (!node) return;
    const target = new THREE.Vector3();
    node.getWorldPosition(target);
    const cam = this.stage.camera;
    const dir = cam.position.clone().sub(this.stage.controls.target);
    if (dir.lengthSq() < 1e-6) dir.set(0, 0.4, 1);
    dir.normalize();
    this._flyTo(target.clone().addScaledVector(dir, m.dist), target);
  }

  _flyTo(camTo, targetTo, ms = 420){
    cancelAnimationFrame(this._flyRaf);
    const cam = this.stage.camera, ctr = this.stage.controls;
    const cam0 = cam.position.clone(), tgt0 = ctr.target.clone();
    const t0 = performance.now();
    const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
    const step = () => {
      const k = Math.min(1, (performance.now() - t0) / ms);
      const e = ease(k);
      cam.position.lerpVectors(cam0, camTo, e);
      ctr.target.lerpVectors(tgt0, targetTo, e);
      ctr.update();
      if (k < 1) this._flyRaf = requestAnimationFrame(step);
    };
    step();
  }
  // follow: the head-cam TF IS the orbit center — rotate/zoom stay free,
  // panning is disabled, and the camera rides along with the robot
  setFollow(on){
    this.follow = on;
    this.stage.controls.enablePan = !on;
    if (on) this._applyFollow();
    else this._followPrev = null;
  }
  _followAnchor(out){
    const v = this.camViz['main'];
    if (v) v.group.getWorldPosition(out);
    else if (this.rig){
      const h = this.rig.root.getObjectByName('head') || this.rig.base;
      h.getWorldPosition(out);
    }
    return out;
  }
  _applyFollow(){
    if (!this.follow || !this.rig || !this.stage.controls.enabled) return;
    const now = this._followAnchor(new THREE.Vector3());
    const d = now.clone().sub(this.stage.controls.target);
    if (d.lengthSq() > 1e-10){
      this.stage.camera.position.add(d);       // keep the orbit offset...
      this.stage.controls.target.copy(now);    // ...while the head stays centered
    }
    this._followPrev = now;
  }

  // ── trajectories: rolling world-space trails of key frames ────────────────
  // sampled at 5 Hz into ring buffers (max 5 min); drawn for enabled traces
  // over the user-set window (default 2 min)
  _trajInit(){
    this.traj = { windowS: 120, on: new Set(), bufs: {}, lines: {}, i: 0, n: 0 };
    this._trajNodes = {
      'left wrist': () => this.rig?.root.getObjectByName('left_palm__anchor'),
      'right wrist': () => this.rig?.root.getObjectByName('right_palm__anchor'),
      head: () => this.rig?.root.getObjectByName('head'),
      odom: () => this.rig?.base,
    };
    this._trajColor = { 'left wrist': 0xf0b68b, 'right wrist': 0xC25B2A,
                        head: 0x4dff8a, odom: 0x4d9dff };
    const CAP = 5 * 60 * 5;                      // 5 min @ 5 Hz
    for (const k in this._trajNodes)
      this.traj.bufs[k] = { t: new Float64Array(CAP), p: new Float32Array(CAP * 3) };
    this._trajCap = CAP;
    this._trajTimer = setInterval(() => this._trajSample(), 200);
    this._trajDraw = setInterval(() => this._drawTraj(), 500);
  }
  _trajSample(){
    if (!this.rig) return;
    const i = this.traj.i;
    const v = new THREE.Vector3();
    for (const k in this._trajNodes){
      const n = this._trajNodes[k]();
      if (!n) continue;
      n.getWorldPosition(v);
      const b = this.traj.bufs[k];
      b.t[i] = performance.now() / 1000;
      b.p[i * 3] = v.x; b.p[i * 3 + 1] = v.y; b.p[i * 3 + 2] = v.z;
    }
    this.traj.i = (i + 1) % this._trajCap;
    this.traj.n++;
  }
  _drawTraj(){
    if (!this.traj) return;
    const now = performance.now() / 1000;
    for (const k in this._trajNodes){
      const line = this.traj.lines[k];
      if (!this.layers.traj || !this.traj.on.has(k)){
        if (line) line.visible = false;
        continue;
      }
      const b = this.traj.bufs[k];
      const count = Math.min(this.traj.n, this._trajCap);
      const pts = [];
      for (let s = 0; s < count; s++){
        const idx = (this.traj.i - count + s + this._trajCap) % this._trajCap;
        if (now - b.t[idx] > this.traj.windowS) continue;
        pts.push(b.p[idx * 3], b.p[idx * 3 + 1], b.p[idx * 3 + 2]);
      }
      if (pts.length < 6){ if (line) line.visible = false; continue; }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      if (line){ line.geometry.dispose(); line.geometry = geo; line.visible = true; }
      else {
        this.traj.lines[k] = new THREE.Line(geo, new THREE.LineBasicMaterial(
          { color: this._trajColor[k], transparent: true, opacity: 0.85 }));
        this.stage.scene.add(this.traj.lines[k]);
      }
    }
  }
  trajOn(k){ return this.traj?.on.has(k); }
  setTrajectory(k, on){
    if (on){ this.traj.on.add(k); this.layers.traj = true; }
    else this.traj.on.delete(k);
    this._drawTraj();
  }
  setTrajWindow(s){ this.traj.windowS = s; this._drawTraj(); }

  // ── ghost rig: translucent copy for setpoint-timeline preview ─────────────
  async ensureGhost(){
    if (this.ghost || this._ghostLoading) return;
    this._ghostLoading = true;
    const g = await new Rig().load(this.stage, this.manifest);
    g.root.traverse(o => {
      if (o.isMesh){
        o.material = new THREE.MeshBasicMaterial(
          { color: 0xf0b68b, transparent: true, opacity: 0.22, depthWrite: false });
      }
    });
    g.world.visible = false;
    this.ghost = g;
    this._ghostLoading = false;
  }
  poseGhost(pose){
    if (!pose){ if (this.ghost) this.ghost.world.visible = false; return; }
    if (!this.ghost){ this.ensureGhost(); return; }
    this.ghost.world.visible = true;
    const { bx, by, yaw } = this._lastBase;
    this.ghost.pose(pose, bx, by, yaw);
  }

  frameRobot(){
    if (!this.rig) return;
    const box = new THREE.Box3().setFromObject(this.rig.world);
    const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    this.stage.controls.target.copy(c);
    const d = Math.max(sz.x, sz.y, sz.z) || 2;
    this.stage.camera.position.copy(c).add(new THREE.Vector3(d * .9, d * .6, d * 1.2));
  }

  _points(pos, col, size){
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true }));
  }
}
