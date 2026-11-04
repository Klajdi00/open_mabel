/* Control Lab 3D viewer — Stage + Rig vendored from the studio family
   (canonical origin: robot_learning/data_curation/web/js/viewport3d.js; repo rule:
   copy, not import), plus the lab overlays: force arrows (true vs estimated),
   the operator skeleton, palm target/actual whiskers, and the odometry trace.

   Coordinate note: MuJoCo world is Z-up; the GLB rig bakes Z-up→Y-up into
   base_link. Free-space overlay points therefore map (x,y,z) → (x, z, -y).   */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const RIG_GLB = '/website-assets/mabel_rig.glb';
const JOINTS_JSON = '/website-assets/mabel_joints.json';

export const mjToThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);

let _gltf = null;
const loadGltf = () => (_gltf = _gltf || new Promise((res, rej) =>
  ((l) => (l.setMeshoptDecoder(MeshoptDecoder), l))(new GLTFLoader()).load(RIG_GLB, res, undefined, rej)));

class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 200);
    this.camera.position.set(2.0, 1.5, 2.4);
    const key = new THREE.DirectionalLight(0xfff3e6, 2.4); key.position.set(3, 5, 2);
    const fill = new THREE.DirectionalLight(0xbfd4ff, .7); fill.position.set(-4, 3, -2);
    this.scene.add(key, fill, new THREE.AmbientLight(0xffffff, .5));
    const disc = new THREE.Mesh(new THREE.CircleGeometry(7, 64),
      new THREE.MeshStandardMaterial({ color: 0x141210, roughness: .95 }));
    disc.rotation.x = -Math.PI / 2; disc.position.y = -0.002;
    this.scene.add(disc);
    this.scene.add(new THREE.GridHelper(12, 48, 0x3a342e, 0x232019));
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = .08;
    this.controls.maxPolarAngle = Math.PI * .52; this.controls.target.set(0, .6, 0);
    new ResizeObserver(() => this._size()).observe(canvas.parentElement);
    this._size();
  }
  _size() {
    const el = this.canvas.parentElement, w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  render() {
    if (this.controls.enabled) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

class Rig {
  async load(stage, manifest) {
    const gltf = await loadGltf();
    this.root = gltf.scene.clone(true);
    this.world = new THREE.Group(); this.world.add(this.root);
    stage.scene.add(this.world);
    this.base = this.root.getObjectByName('base_link') || this.root;
    this.baseP0 = this.base.position.clone(); this.baseQ0 = this.base.quaternion.clone();
    // Constant frame offset between the GLB base_link and the MJCF chassis
    // (in the MJ frame), auto-measured against the streamed ground truth —
    // the exporter bakes a ~0.23 m x-shift that would otherwise misalign the
    // rendered robot with every truth-anchored overlay.
    this.corrMj = new THREE.Vector3();
    this.joints = {};
    for (const j of manifest.joints) {
      const n = this.root.getObjectByName(j.node); if (!n) continue;
      this.joints[j.name] = { ...j, _n: n, _p0: n.position.clone(),
        _q0: n.quaternion.clone(), _ax: new THREE.Vector3(...j.axis).normalize() };
    }
    return this;
  }
  setJoint(name, val) {
    const j = this.joints[name]; if (!j || val == null) return;
    if (j.type === 'prismatic') {
      j._n.position.copy(j._p0).add(
        j._ax.clone().applyQuaternion(j._q0).multiplyScalar(val));
      j._n.quaternion.copy(j._q0);
    } else {
      j._n.quaternion.copy(j._q0).multiply(
        new THREE.Quaternion().setFromAxisAngle(j._ax, val));
      j._n.position.copy(j._p0);
    }
  }
  pose(q, bx, by, yaw, dz, qd) {
    for (const n in q) this.setJoint(n, q[n]);
    const corr = this.corrMj.clone()
      .applyAxisAngle(new THREE.Vector3(0, 0, 1), yaw);   // rotate in MJ frame
    // base_link's LOCAL frame is MJ-frame (Z-up); baseQ0 maps it to three's world.
    // Height drop (dz) rides the local Z; the body-frame rotation delta (qd, MJ
    // [w,x,y,z]) is applied in the local frame so roll/pitch/yaw — i.e. tip-over
    // and falls — all render, generalising the old yaw-only path (qd = Rz(yaw)).
    const off = new THREE.Vector3(bx + corr.x, by + corr.y, corr.z + (dz || 0))
      .applyQuaternion(this.baseQ0);
    this.base.position.copy(this.baseP0).add(off);
    const rot = (qd && qd.length === 4)
      ? new THREE.Quaternion(qd[1], qd[2], qd[3], qd[0])   // [w,x,y,z] → (x,y,z,w)
      : new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw);
    this.base.quaternion.copy(this.baseQ0).multiply(rot);
    this.world.updateMatrixWorld(true);
  }
}

/* Time-graded trajectory trail: a polyline whose vertex colors fade from the
   background (oldest) to the series color (newest) — the fade IS the time
   axis — with a bright dot at "now" and a dim dot at the start. */
const BG = new THREE.Color(0x0d0c0b);
class Trail {
  constructor(scene, colorHex, { floorY = null, max = 700, minStep = 0.006 } = {}) {
    this.color = new THREE.Color(colorHex);
    this.floorY = floorY; this.max = max; this.minStep = minStep;
    this.pts = [];
    this.line = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true }));
    this.line.frustumCulled = false;
    this.now = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12),
      new THREE.MeshBasicMaterial({ color: colorHex }));
    this.start = new THREE.Mesh(new THREE.SphereGeometry(0.008, 10, 10),
      new THREE.MeshBasicMaterial({ color: this.color.clone().lerp(BG, 0.55) }));
    this.now.visible = this.start.visible = false;
    scene.add(this.line, this.now, this.start);
  }
  push(v) {
    if (this.floorY != null) v = new THREE.Vector3(v.x, this.floorY, v.z);
    const last = this.pts[this.pts.length - 1];
    if (last && last.distanceTo(v) < this.minStep) {
      this.now.position.copy(v);                       // keep "now" live
      return;
    }
    this.pts.push(v.clone());
    if (this.pts.length > this.max) this.pts.shift();
    const n = this.pts.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const p = this.pts[i];
      pos[3 * i] = p.x; pos[3 * i + 1] = p.y; pos[3 * i + 2] = p.z;
      c.copy(BG).lerp(this.color, Math.pow((i + 1) / n, 1.6));  // ease the fade
      col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b;
    }
    const g = this.line.geometry;
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.computeBoundingSphere?.();
    this.now.position.copy(v);
    this.start.position.copy(this.pts[0]);
    this.now.visible = this.start.visible = this.line.visible = this._on !== false;
  }
  setOn(on) {
    this._on = on;
    this.line.visible = this.now.visible = this.start.visible =
      on && this.pts.length > 0;
  }
  clear() {
    this.pts = [];
    this.line.geometry.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(0), 3));
    this.now.visible = this.start.visible = false;
  }
}

/* Camera presets — computed AT FLIGHT TIME from the robot's live transforms:
   the target is the actual body part's position and the camera offset is
   expressed in the robot's yaw frame, so every preset frames the part
   wherever the robot has driven. (MuJoCo yaw ψ about Z maps to three.js
   R_y(+ψ) under the (x,y,z)→(x,z,−y) world map.) */
const PRESET_DEFS = {
  //         camera offset (robot frame)   target
  orbit:     { off: [2.0, 1.5, 2.4],  tgt: st => st.basePt(0.6) },
  front:     { off: [-2.4, 0.6, 0.0], tgt: st => st.basePt(0.8) },
  arms:      { off: [-1.35, 0.35, 0.0], tgt: st => st.palmsMid() },
  left_arm:  { off: [-1.0, 0.35, -0.85], tgt: st => st.body('palm_l') },
  right_arm: { off: [-1.0, 0.35, 0.85],  tgt: st => st.body('palm_r') },
  hands:     { off: [-0.95, 0.25, 0.0],  tgt: st => st.palmsMid() },
  upper:     { off: [-1.9, 0.7, 1.3],  tgt: st => st.body('head', -0.25) },
  whole:     { off: [2.6, 1.7, 2.6],   tgt: st => st.basePt(0.5) },
  base:      { off: [1.6, 0.55, 1.6],  tgt: st => st.basePt(0.15) },
  top:       { off: [0.01, 5.2, 0.01], tgt: st => st.basePt(0.0) },
};

function presetPose(name, state) {
  const def = PRESET_DEFS[name];
  if (!def || !state) return null;
  const b = state.base || {};
  const yaw = b.yaw ?? 0;
  const st = {
    basePt: h => new THREE.Vector3(b.x || 0, h, -(b.y || 0)),
    body: (k, dy = 0) => {
      const p = (state.bodies || {})[k];
      return p ? mjToThree(p).add(new THREE.Vector3(0, dy, 0))
               : new THREE.Vector3(b.x || 0, 0.9, -(b.y || 0));
    },
    palmsMid: () => {
      const l = (state.bodies || {}).palm_l, r = (state.bodies || {}).palm_r;
      if (!l || !r) return new THREE.Vector3(b.x || 0, 0.95, -(b.y || 0));
      return mjToThree(l).add(mjToThree(r)).multiplyScalar(0.5);
    },
  };
  const tgt = def.tgt(st);
  const off = new THREE.Vector3(...def.off)
    .applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  return { pos: tgt.clone().add(off), tgt };
}

export class LabViewer {
  constructor(canvas) {
    this.stage = new Stage(canvas);
    this.ready = this._init();
    this._fly = null;
    this._followBase = true;
    this._odomPts = [];
    this._lastBench = null;
    this.onWristDrag = null;               // set by main.js (side, mjPos) =>
    this._drag = null;
    this._mkOverlays();
    this._mkTrails(canvas);
    const tick = () => { this._animate(); this.stage.render(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  async _init() {
    const manifest = await (await fetch(JOINTS_JSON)).json();
    this.rig = await new Rig().load(this.stage, manifest);
  }

  /* ── overlays ─────────────────────────────────────────────────────────── */
  _mkOverlays() {
    const S = this.stage.scene;
    this.grp = new THREE.Group(); S.add(this.grp);
    // force arrows: true (green) / estimated (red)
    this.arrTrue = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(), 0.4, 0x3fb56b, 0.09, 0.05);
    this.arrEst = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(), 0.4, 0xd23b2a, 0.09, 0.05);
    this.ptTrue = this._sphere(0.02, 0x3fb56b);
    this.ptEst = this._sphere(0.02, 0xd23b2a);
    for (const o of [this.arrTrue, this.arrEst, this.ptTrue, this.ptEst])
      { o.visible = false; this.grp.add(o); }
    // operator skeleton (peach) + bone lines
    this.skelPts = {}; this.skelMat = new THREE.MeshBasicMaterial({ color: 0xf0b68b });
    this.skelLines = new THREE.LineSegments(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xf0b68b, transparent: true, opacity: .55 }));
    this.grp.add(this.skelLines);
    // palm target/actual whiskers (+ soft halos for hover/drag feedback)
    this.tgtMarks = { l: this._sphere(0.018, 0xc25b2a), r: this._sphere(0.018, 0xc25b2a) };
    this.tgtHalos = {};
    for (const side of ['l', 'r']) {
      const h = new THREE.Mesh(new THREE.SphereGeometry(0.030, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xf0b68b, transparent: true,
          opacity: 0.0, depthWrite: false }));
      h.raycast = () => {};            // halo must never swallow the drag ray
      this.tgtHalos[side] = h;
      this.tgtMarks[side].add(h);              // halo rides the mark
    }
    this.errLines = new THREE.LineSegments(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe9a679 }));
    this.grp.add(this.tgtMarks.l, this.tgtMarks.r, this.errLines);
    this.tgtMarks.l.visible = this.tgtMarks.r.visible = false;
    this._hover = null;                        // 'l' | 'r' | null
    // odometry: true path (green line) + odom estimate (dashed-ish amber)
    this.pathTrue = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x3fb56b }));
    this.pathOdom = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xe9a679, transparent: true, opacity: .8 }));
    this.grp.add(this.pathTrue, this.pathOdom);
  }
  _sphere(r, color) {
    return new THREE.Mesh(new THREE.SphereGeometry(r, 12, 12),
      new THREE.MeshBasicMaterial({ color }));
  }
  _mkTrails(canvas) {
    const S = this.stage.scene;
    this.trails = {
      head:   new Trail(S, 0xf0b68b),                      // peach
      palm_l: new Trail(S, 0x7fb2ff),                      // port = blue
      palm_r: new Trail(S, 0xe7c24b),                      // starboard = gold
      base:   new Trail(S, 0x3fb56b, { floorY: 0.012, minStep: 0.01 }), // floor
    };
    this._trailsOn = true;
    // wrist-target dragging (wrist_ik bench): grab an orange target sphere,
    // drag on a camera-facing plane through the grab point
    this._ray = new THREE.Raycaster();
    const ndc = e => {
      const r = canvas.getBoundingClientRect();
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                               -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    canvas.addEventListener('pointerdown', e => {
      if (!this._dragEnabled) return;
      this._ray.setFromCamera(ndc(e), this.stage.camera);
      const hits = this._ray.intersectObjects(
        [this.tgtMarks.l, this.tgtMarks.r].filter(m => m.visible), false);
      if (!hits.length) return;
      const side = hits[0].object === this.tgtMarks.l ? 'l' : 'r';
      const n = new THREE.Vector3();
      this.stage.camera.getWorldDirection(n);
      this._drag = { side,
        plane: new THREE.Plane().setFromNormalAndCoplanarPoint(n, hits[0].object.position) };
      this.stage.controls.enabled = false;
      canvas.style.cursor = 'grabbing';
      canvas.setPointerCapture(e.pointerId);
      e.stopPropagation();
    });
    canvas.addEventListener('pointermove', e => {
      if (this._drag) {
        this._ray.setFromCamera(ndc(e), this.stage.camera);
        const p = new THREE.Vector3();
        if (!this._ray.ray.intersectPlane(this._drag.plane, p)) return;
        p.y = Math.max(0.05, p.y);                     // keep above the floor
        this.tgtMarks[this._drag.side].position.copy(p);
        // three (x,y,z) -> MuJoCo (x, -z, y)
        this.onWristDrag && this.onWristDrag(this._drag.side, [p.x, -p.z, p.y]);
        return;
      }
      if (!this._dragEnabled) return;
      this._ray.setFromCamera(ndc(e), this.stage.camera);
      const hits = this._ray.intersectObjects(
        [this.tgtMarks.l, this.tgtMarks.r].filter(m => m.visible), false);
      this._hover = hits.length
        ? (hits[0].object === this.tgtMarks.l ? 'l' : 'r') : null;
      canvas.style.cursor = this._hover ? 'grab' : '';
    });
    const endDrag = () => {
      if (!this._drag) return;
      this._drag = null;
      canvas.style.cursor = this._hover ? 'grab' : '';
      this._syncControls();
      this.stage.controls.enabled = true;
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
  }
  setTrails(on) {
    this._trailsOn = on;
    for (const t of Object.values(this.trails)) t.setOn(on);
  }
  clearTrails() { for (const t of Object.values(this.trails)) t.clear(); }
  setDragEnabled(on) {
    this._dragEnabled = on;
    if (!on) { this._hover = null; this.stage.canvas.style.cursor = ''; }
  }
  _arrow(arrow, pt, w) {
    if (!w) { arrow.visible = pt.visible = false; return; }
    const p = mjToThree(w.point);
    const f = new THREE.Vector3(w.force[0], w.force[2], -w.force[1]);
    const mag = f.length();
    if (mag < 1e-3) { arrow.visible = pt.visible = false; return; }
    arrow.position.copy(p); arrow.setDirection(f.normalize());
    arrow.setLength(Math.min(0.15 + mag * 0.02, 0.9), 0.08, 0.045);
    pt.position.copy(p);
    arrow.visible = pt.visible = true;
  }

  /* ── per-state update ─────────────────────────────────────────────────── */
  update(s) {
    if (!this.rig) return;
    const b = s.base || {};
    const bp = s.basePose || {};
    this.rig.pose(s.jointPositions || {}, b.x || 0, b.y || 0, b.yaw ?? 0, bp.dz || 0, bp.qd);
    this._state = s;
    // one-time rig↔truth calibration (needs a near-zero yaw sample)
    if (!this._corrDone && s.bodies && s.bodies.palm_l
        && Math.abs(b.yaw ?? 0) < 0.05) {
      const node = this.rig.root.getObjectByName('left_palm');
      if (node) {
        const wp = new THREE.Vector3(); node.getWorldPosition(wp);
        const d = wp.sub(mjToThree(s.bodies.palm_l));    // world delta
        if (d.length() > 0.005 && d.length() < 1.0) {
          // world → MJ components: (x, y, z)mj = (wx, −wz, wy); subtract
          this.rig.corrMj.set(-d.x, d.z, -d.y);
          this._corrDone = true;
        } else if (d.length() <= 0.005) { this._corrDone = true; }
      }
    }
    if (s.method !== this._lastBench) { this._lastBench = s.method; this.clearTrails(); }
    if (this._trailsOn && s.bodies) {
      if (s.bodies.head) this.trails.head.push(mjToThree(s.bodies.head));
      if (s.bodies.palm_l) this.trails.palm_l.push(mjToThree(s.bodies.palm_l));
      if (s.bodies.palm_r) this.trails.palm_r.push(mjToThree(s.bodies.palm_r));
      this.trails.base.push(new THREE.Vector3(b.x || 0, 0, -(b.y || 0)));
    }
    // odometry traces (Drive mode)
    if (s.method === 'navigation') {
      const last = this._odomPts[this._odomPts.length - 1];
      if (!last || Math.hypot(last.tx - b.x, last.ty - b.y) > 0.01 ||
          Math.hypot(last.ox - s.odom.x, last.oy - s.odom.y) > 0.01) {
        this._odomPts.push({ tx: b.x, ty: b.y, ox: s.odom.x, oy: s.odom.y });
        if (this._odomPts.length > 600) this._odomPts.shift();
        const T = [], O = [];
        for (const p of this._odomPts) {
          T.push(new THREE.Vector3(p.tx, 0.01, -p.ty));
          O.push(new THREE.Vector3(p.ox, 0.012, -p.oy));
        }
        this.pathTrue.geometry.setFromPoints(T);
        this.pathOdom.geometry.setFromPoints(O);
      }
    }
    // wrench overlays
    const w = (s.overlay || {}).wrench || {};
    this._arrow(this.arrTrue, this.ptTrue, w.true);
    this._arrow(this.arrEst, this.ptEst, w.est);
    // operator skeleton
    this._skeleton((s.overlay || {}).skeleton);
    // palm targets
    this._targets((s.overlay || {}).targets);
    // camera base-follow while driving — NEVER during a preset flight (it
    // dragged the orbit target mid-flight and the landing snapped).
    if (this._followBase && s.method === 'navigation' && !this._fly) {
      const t = new THREE.Vector3(b.x || 0, 0.5, -(b.y || 0));
      this.stage.controls.target.lerp(t.add(new THREE.Vector3(0, 0.1, 0)), 0.06);
    }
  }
  _skeleton(sk) {
    for (const k in this.skelPts) this.skelPts[k].visible = false;
    if (!sk) { this.skelLines.visible = false; return; }
    const segs = [];
    let head = null;
    for (const [k, p] of Object.entries(sk)) {
      let m = this.skelPts[k];
      if (!m) { m = this.skelPts[k] = this._sphere(k === 'head' ? 0.035 : 0.014, 0xf0b68b); this.grp.add(m); }
      m.position.copy(mjToThree(p)); m.visible = true;
      if (k === 'head') head = m.position;
    }
    for (const side of ['l', 'r']) {
      const w = this.skelPts[`wrist_${side}`], e = this.skelPts[`elbow_${side}`];
      if (head && e && e.visible) segs.push(head, e.position);
      if (e && e.visible && w && w.visible) segs.push(e.position, w.position);
      else if (head && w && w.visible) segs.push(head, w.position);
      for (const f of ['thumb', 'index', 'middle', 'ring', 'little']) {
        const k = this.skelPts[`${side}_${f}_knuckle`], t = this.skelPts[`${side}_${f}_tip`];
        if (w && w.visible && k && k.visible) segs.push(w.position, k.position);
        if (k && k.visible && t && t.visible) segs.push(k.position, t.position);
        else if (w && w.visible && t && t.visible) segs.push(w.position, t.position);
      }
    }
    this.skelLines.geometry.setFromPoints(segs.map(v => v.clone()));
    this.skelLines.visible = segs.length > 0;
  }
  _targets(tg) {
    this.tgtMarks.l.visible = this.tgtMarks.r.visible = false;
    if (!tg) { this.errLines.visible = false; return; }
    const segs = [];
    for (const side of ['l', 'r']) {
      const v = tg[side]; if (!v) continue;
      const t = mjToThree(v.target), a = mjToThree(v.actual);
      if (!(this._drag && this._drag.side === side))   // don't fight the drag
        this.tgtMarks[side].position.copy(t);
      this.tgtMarks[side].visible = true;
      segs.push(this.tgtMarks[side].position.clone(), a);
    }
    this.errLines.geometry.setFromPoints(segs);
    this.errLines.visible = segs.length > 0;
  }
  clearPaths() { this._odomPts = []; this.pathTrue.geometry.setFromPoints([]);
    this.pathOdom.geometry.setFromPoints([]); }

  /* ── camera fly-to presets ────────────────────────────────────────────── */
  _syncControls() {
    // Flush + zero OrbitControls' internal spherical/pan deltas: one update()
    // with damping off applies any residual momentum NOW and clears it, so
    // re-enabling after a flight can't snap the camera.
    const c = this.stage.controls;
    const d = c.enableDamping;
    c.enableDamping = false; c.update(); c.enableDamping = d;
  }
  flyTo(name, ms = 850) {
    const p = presetPose(name, this._state); if (!p) return;
    if (!this._fly) this._syncControls();       // drop in-progress momentum
    const cam = this.stage.camera, ctl = this.stage.controls;
    ctl.enabled = false;
    // The ORIENTATION must fly too (SCC viewport3d.flyToView parity): with
    // OrbitControls disabled nothing rotates the camera during the flight,
    // so a position-only lerp kept the OLD look direction — the view drifted
    // AWAY from the robot, then SNAPPED to the target at landing.
    const q1 = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(p.pos, p.tgt, cam.up));
    this._fly = { t0: performance.now(), ms,
      p0: cam.position.clone(), p1: p.pos,
      q0: cam.quaternion.clone(), q1,
      t0v: ctl.target.clone(), t1v: p.tgt };
  }
  _animate() {
    // hover / drag highlight: ease scale + halo opacity (Apple-soft)
    for (const side of ['l', 'r']) {
      const m = this.tgtMarks[side], h = this.tgtHalos[side];
      if (!m) continue;
      const state = (this._drag && this._drag.side === side) ? 2
                  : (this._hover === side ? 1 : 0);
      const base = this._dragEnabled ? 1.35 : 1.0;
      const sTgt = base * (state === 2 ? 1.28 : state === 1 ? 1.14 : 1.0);
      const oTgt = state === 2 ? 0.45 : state === 1 ? 0.28 : 0.0;
      m.scale.setScalar(m.scale.x + (sTgt - m.scale.x) * 0.25);
      h.material.opacity += (oTgt - h.material.opacity) * 0.25;
    }
    const f = this._fly; if (!f) return;
    const u = Math.min(1, (performance.now() - f.t0) / f.ms);
    const e = u * u * (3 - 2 * u);
    this.stage.camera.position.lerpVectors(f.p0, f.p1, e);
    this.stage.camera.quaternion.copy(f.q0).slerp(f.q1, e);
    this.stage.controls.target.lerpVectors(f.t0v, f.t1v, e);
    if (u >= 1) {
      this.stage.camera.position.copy(f.p1);    // land EXACTLY on the preset
      this.stage.camera.quaternion.copy(f.q1);
      this.stage.controls.target.copy(f.t1v);
      this._fly = null;
      this._syncControls();                     // rebuild spherical state clean
      this.stage.controls.enabled = true;
    }
  }
}
