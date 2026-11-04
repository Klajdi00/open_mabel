/* Retarget Lab 3D stage — one MABEL rig per active retargeter, side by side
   (or superimposed as ghosts), plus the operator's tracked skeleton.

   Frame rules (the root README, "The two frame rules every 3-D client must obey"):
   1. the GLB's base_link carries the website centering bake (T_center) — strip
      it off the ROBOT (rig.world.position = (-baseP0.x, 0, -baseP0.z)), never
      move the overlays by it.
   2. render q - ref, never raw qpos (the 14 ORCA finger/wrist joints have ref≠0).
   World overlays (wrist targets, achieved wrists) arrive in the MuJoCo world
   (Z up) and map (x,y,z) -> (x, z, -y). Any residual GLB↔MJ offset is MEASURED
   once against the streamed wrist anchor, never eyeballed.                    */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const RIG_GLB = '/website-assets/mabel_rig.glb';
const JOINTS_JSON = '/website-assets/mabel_joints.json';
export const mjToThree = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
// The robot faces MJ -X (three -X), so its RIGHT is MJ +Y → three -Z. The
// lineup runs along three +Z: seen from the front camera that is screen-right,
// so the operator (slot 0) stands leftmost and the robots follow.
const SPACING = 1.15;
const RIGHT = new THREE.Vector3(0, 0, -1);
const LINEUP = new THREE.Vector3(0, 0, 1);

let _gltf = null;
const loadGltf = () => (_gltf = _gltf || new Promise((res, rej) =>
  ((l) => (l.setMeshoptDecoder(MeshoptDecoder), l))(new GLTFLoader()).load(RIG_GLB, res, undefined, rej)));
// END EFFECTOR: the OpenArm v1 gripper (teleop_engine/end_effector.py), served once by /api/ee_meshes as raw
// arrays (arm-global mm, Enactic's export). Built here, cloned per rig, mounted on the J7 node at the pose the
// worker streams (geo[side].mount = OpenArm link7 frame in the J7 body frame).
let _ee = null;
const _b64 = (b, T) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
const loadEE = () => (_ee = _ee || fetch('/api/ee_meshes').then(r => r.json()).then(j => {
  const geos = {};
  for (const [name, m] of Object.entries(j.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(_b64(m.pos, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(_b64(m.idx, Uint32Array), 1));
    g.computeVertexNormals();
    geos[name] = g;
  }
  return { geos, spec: j.spec };
}));
function buildGripper(ee) {
  const { geos, spec } = ee, root = new THREE.Group(), hand = new THREE.Group(), fingers = [];
  root.name = 'openarm_gripper'; root.matrixAutoUpdate = false;
  const mesh = (name, flip) => {
    const m = new THREE.Mesh(geos[name], new THREE.MeshStandardMaterial({
      color: spec.colors[name] || '#cccccc', roughness: .45, metalness: .3, side: THREE.DoubleSide }));
    m.scale.set(0.001, flip ? -0.001 : 0.001, 0.001);
    return m;
  };
  for (const n of ['link7_0', 'link7_1']) {               // OpenArm link7: wrist housing + the gripper's motor
    if (!geos[n]) continue;
    const m = mesh(n, false); m.position.set(...spec.housing_geom_pos); root.add(m);
  }
  hand.position.set(...spec.hand_pos); root.add(hand);
  for (const n of ['hand_0', 'hand_1']) { const m = mesh(n, false); m.position.set(...spec.hand_geom_pos); hand.add(m); }
  for (const f of spec.fingers) {
    const g = new THREE.Group(); g.position.set(...f.pos); hand.add(g);
    for (const n of ['finger_0', 'finger_1']) { const m = mesh(n, f.flip); m.position.set(...f.geom_pos); g.add(m); }
    fingers.push({ g, p0: new THREE.Vector3(...f.pos), ax: new THREE.Vector3(...f.axis) });
  }
  return { root, fingers, range: spec.finger_range };
}
let _manifest = null;
const loadManifest = () => (_manifest = _manifest || fetch(JOINTS_JSON).then(r => r.json()));

class Rig {
  async load(scene, manifest, color) {
    const gltf = await loadGltf();
    this.root = gltf.scene.clone(true);
    this.root.traverse(o => { if (o.isMesh) o.material = o.material.clone(); });
    this.world = new THREE.Group(); this.world.add(this.root);
    scene.add(this.world);
    this.base = this.root.getObjectByName('base_link') || this.root;
    this.baseP0 = this.base.position.clone();
    this.joints = {};
    for (const j of manifest.joints) {
      const n = this.root.getObjectByName(j.node); if (!n) continue;
      this.joints[j.name] = { ...j, _n: n, _p0: n.position.clone(), _q0: n.quaternion.clone(),
        _ax: new THREE.Vector3(...j.axis).normalize(), ref: j.ref || 0 };
    }
    this.color = new THREE.Color(color);
    // frame rule 1: strip T_center's X/Z off the robot (keep base_link's height).
    // Stripped on the ROOT, so `world` is a pure chassis pose: base_link sits on
    // its origin and the streamed base yaw turns the robot about its own centre.
    this.root.position.set(-this.baseP0.x, 0, -this.baseP0.z);
    this.slot = new THREE.Vector3();
    this.baseOff = new THREE.Vector3();
    this.place(this.slot);
    return this;
  }
  place(slot) {
    this.slot.copy(slot);
    this.world.position.copy(slot).add(this.baseOff);
    this.world.updateMatrixWorld(true);
  }
  // the chassis world pose the engine integrated (whole-body: the base walks
  // and turns). MJ (x,y) -> three (x,-y); MJ yaw about +Z == three yaw about +Y.
  setBase(b) {
    if (!b) return;
    this.baseOff.set(b.x || 0, 0, -(b.y || 0));
    this.world.rotation.y = b.yaw || 0;
    this.place(this.slot);
  }
  setJoint(name, val) {
    const j = this.joints[name]; if (!j || val == null) return;
    const v = val - j.ref;                         // frame rule 2: q - ref
    if (j.type === 'prismatic') {
      j._n.position.copy(j._p0).add(j._ax.clone().applyQuaternion(j._q0).multiplyScalar(v));
    } else {
      j._n.quaternion.copy(j._q0).multiply(new THREE.Quaternion().setFromAxisAngle(j._ax, v));
    }
  }
  pose(q) { for (const n in q) this.setJoint(n, q[n]); this.world.updateMatrixWorld(true); }
  wristWorld(side) {
    const j = this.joints[`${side}_wrist`]; if (!j) return null;
    return j._n.getWorldPosition(new THREE.Vector3());
  }
  // ORCA hand v1 <-> OpenArm gripper on J7. `kind` is the lab's setting (so a toggle shows at once, with or
  // without frames streaming), `mounts` the worker's gripper pose on J7 per side (sent with "ready"), `open`
  // the record's per-side opening [m]. The gripper is built as soon as meshes + mounts exist, then only shown.
  setEndEffector(kind, mounts, open) {
    const grip = kind === 'gripper';
    if (!this._eeSpec && !this._eeLoading) {
      this._eeLoading = true;
      loadEE().then(ee => { this._eeSpec = ee; this._eeLoading = false; }).catch(() => { this._eeLoading = false; });
    }
    if (this._eeSpec && !this._ee && mounts && mounts.left && mounts.right) {
      this._ee = {};
      for (const side of ['left', 'right']) {
        const j7 = this.joints[`${side}_arm_7`], mount = mounts[side];
        if (!j7 || !mount) continue;
        const g = buildGripper(this._eeSpec);
        g.root.matrix.set(...mount); g.root.updateMatrixWorld(true);
        j7._n.add(g.root);
        this._ee[side] = g;
      }
      if (this._ghost) this.setGhost(true);
      this._eeShown = null;                                   // re-apply visibility now that it exists
    }
    // what is drawn: the gripper once its meshes are built, else the ORCA hand (also while they load)
    const show = grip && this._ee ? 'gripper' : 'orca';
    if (show !== this._eeShown) {
      this._eeShown = show;
      for (const side of ['left', 'right']) {               // ORCA = every mesh under J7 that is not the gripper
        const j7 = this.joints[`${side}_arm_7`]; if (!j7) continue;
        j7._n.traverse(o => {
          if (!o.isMesh) return;
          let a = o, ours = false;
          while (a && a !== j7._n) { if (a.name === 'openarm_gripper') { ours = true; break; } a = a.parent; }
          o.visible = ours ? show === 'gripper' : show === 'orca';
        });
      }
    }
    if (show === 'gripper') this._setOpen(open);
  }
  _setOpen(open) {
    if (!this._ee || !open) return;
    for (const [side, g] of Object.entries(this._ee)) {
      const q = Math.min(g.range[1], Math.max(g.range[0], (open[side] || 0) / 2));   // per finger
      for (const f of g.fingers) f.g.position.copy(f.p0).addScaledVector(f.ax, q);
    }
  }
  setGhost(on) {
    this._ghost = on;
    this.root.traverse(o => {
      if (!o.isMesh) return;
      const m = o.material;
      m.transparent = on; m.opacity = on ? 0.22 : 1; m.depthWrite = !on;
      // coplanar with the solid mirror robot wherever the maps agree: push the
      // ghost back so the mirror always wins the depth test
      m.polygonOffset = on; m.polygonOffsetFactor = on ? 2 : 0; m.polygonOffsetUnits = on ? 8 : 0;
      m.needsUpdate = true;
      if (m.emissive) { m.emissive.copy(on ? this.color : new THREE.Color(0)); m.emissiveIntensity = on ? 0.45 : 0; }
    });
  }
  dispose(scene) { scene.remove(this.world); }
}

/* the operator: finger chains + forearm + head, metric, unscaled */
class Operator {
  constructor(scene) {
    this.g = new THREE.Group(); scene.add(this.g);
    const mat = new THREE.LineBasicMaterial({ color: 0xefeae3, transparent: true, opacity: .9 });
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), mat);
    this.lines.frustumCulled = false;
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 20, 16),
      new THREE.MeshStandardMaterial({ color: 0xcfc8be, roughness: .6, transparent: true, opacity: .55 }));
    this.visor = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.07, 0.16),
      new THREE.MeshStandardMaterial({ color: 0x222222, roughness: .3 }));
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(0.007, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xf0b68b }), 64);
    this.joints.frustumCulled = false;
    this.g.add(this.lines, this.head, this.visor, this.joints);
    this.slot = new THREE.Vector3();
  }
  update(sk) {
    this.g.visible = !!sk && this.on !== false;
    if (!sk) return;
    const P = (p) => mjToThree(p).add(this.slot);
    const seg = [], pts = [];
    for (const side of ['left', 'right']) {
      const h = sk.hands[side]; if (!h) continue;
      for (const ch of h.fingers) for (let i = 0; i + 1 < ch.length; i++) seg.push(P(ch[i]), P(ch[i + 1]));
      if (h.elbow) seg.push(P(h.wrist), P(h.elbow));
      for (const ch of h.fingers) pts.push(P(ch[ch.length - 1]));
      pts.push(P(h.wrist)); if (h.elbow) pts.push(P(h.elbow));
    }
    const hp = P(sk.head);
    // neck + shoulders sketch so the figure reads as a person (not tracked:
    // ARKit has no torso; drawn from the head only)
    const down = new THREE.Vector3(0, -0.24, 0);
    const neck = hp.clone().add(down);
    seg.push(hp.clone().add(new THREE.Vector3(0, -0.09, 0)), neck);
    seg.push(neck.clone().add(RIGHT.clone().multiplyScalar(0.19)), neck.clone().add(RIGHT.clone().multiplyScalar(-0.19)));
    this.lines.geometry.setFromPoints(seg);
    this.head.position.copy(hp);
    const gz = mjToThree(sk.gaze).normalize();
    this.visor.position.copy(hp).add(gz.clone().multiplyScalar(0.075));
    this.visor.lookAt(hp.clone().add(gz.clone().multiplyScalar(2)));
    this.visor.rotateY(Math.PI / 2);
    const m = new THREE.Matrix4();
    this.joints.count = Math.min(pts.length, 64);
    for (let i = 0; i < this.joints.count; i++) this.joints.setMatrixAt(i, m.makeTranslation(pts[i].x, pts[i].y, pts[i].z));
    this.joints.instanceMatrix.needsUpdate = true;
  }
}

/* per-variant wrist markers: ring = the map's target, dot = achieved */
class Markers {
  constructor(scene, color) {
    this.g = new THREE.Group(); scene.add(this.g);
    const c = new THREE.Color(color);
    this.items = {};
    for (const s of ['left', 'right']) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.03, 0.004, 8, 28),
        new THREE.MeshBasicMaterial({ color: c, depthTest: false, transparent: true, opacity: .9 }));
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 10),
        new THREE.MeshBasicMaterial({ color: c, depthTest: false }));
      const line = new THREE.Line(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: c, depthTest: false, transparent: true, opacity: .8 }));
      ring.renderOrder = dot.renderOrder = line.renderOrder = 10;
      this.g.add(ring, dot, line);
      this.items[s] = { ring, dot, line };
    }
  }
  update(geo, off, cam, on) {
    this.g.visible = on;
    if (!on || !geo) return;
    for (const s of ['left', 'right']) {
      const it = this.items[s], g = geo[s];
      const ok = g && g.tgt;
      it.ring.visible = it.line.visible = !!ok; it.dot.visible = !!g;
      if (!g) continue;
      const w = mjToThree(g.wr).add(off);
      it.dot.position.copy(w);
      if (ok) {
        const t = mjToThree(g.tgt).add(off);
        it.ring.position.copy(t); it.ring.lookAt(cam.position);
        it.line.geometry.setFromPoints([t, w]);
      }
    }
  }
  dispose(scene) { scene.remove(this.g); }
}

export class LabViewer {
  constructor(canvas, labelsEl) {
    this.canvas = canvas; this.labelsEl = labelsEl;
    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    r.outputColorSpace = THREE.SRGBColorSpace; r.toneMapping = THREE.ACESFilmicToneMapping;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.02, 200);
    const key = new THREE.DirectionalLight(0xfff3e6, 2.4); key.position.set(-3, 5, 2);
    const fill = new THREE.DirectionalLight(0xbfd4ff, .7); fill.position.set(4, 3, -2);
    this.scene.add(key, fill, new THREE.AmbientLight(0xffffff, .55));
    const disc = new THREE.Mesh(new THREE.CircleGeometry(9, 64),
      new THREE.MeshStandardMaterial({ color: 0x141210, roughness: .95 }));
    disc.rotation.x = -Math.PI / 2; disc.position.y = -0.002;
    this.scene.add(disc, new THREE.GridHelper(16, 64, 0x3a342e, 0x232019));
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true; this.controls.dampingFactor = .08;
    this.controls.maxPolarAngle = Math.PI * .52;
    this.rigs = {};          // vid -> {rig, markers, label, color, name}
    this.follow = true;      // the camera rides with the walking robots / operator
    this.split = false;      // physics on: top kinematic, bottom NOVA controller
    this.phys = null;
    this.layout = 'lineup';
    this.showTargets = true;
    this.op = new Operator(this.scene);
    this.off = null;         // measured MJ->GLB overlay correction (three units)
    this._order = [];
    new ResizeObserver(() => this._size()).observe(canvas.parentElement);
    this._size();
    this.view('front', 0);
    const loop = () => { requestAnimationFrame(loop); this._frame(); };
    loop();
  }
  _size() {
    const el = this.canvas.parentElement, w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.camera.aspect = w / (this.split ? h / 2 : h); this.camera.updateProjectionMatrix();
  }
  /* PHYSICS split: top = kinematic (the retargeter's command), bottom = NOVA's
     whole-body controller in MuJoCo (solid) over a ghost of the kinematic pose,
     same camera — the gap between solid and ghost IS the controller lag. */
  async _ensurePhys() {
    if (this.phys) return this.phys;
    const sc = new THREE.Scene();
    const key = new THREE.DirectionalLight(0xfff3e6, 2.4); key.position.set(-3, 5, 2);
    const fill = new THREE.DirectionalLight(0xbfd4ff, .7); fill.position.set(4, 3, -2);
    sc.add(key, fill, new THREE.AmbientLight(0xffffff, .55));
    const disc = new THREE.Mesh(new THREE.CircleGeometry(9, 64),
      new THREE.MeshStandardMaterial({ color: 0x10161a, roughness: .95 }));
    disc.rotation.x = -Math.PI / 2; disc.position.y = -0.002;
    sc.add(disc, new THREE.GridHelper(16, 64, 0x2c3a40, 0x1b2428));
    const manifest = await loadManifest();
    this.phys = { scene: sc, rig: await new Rig().load(sc, manifest, '#2a9d8f'),
                  ghost: await new Rig().load(sc, manifest, '#e8a33d') };
    this.phys.ghost.setGhost(true);
    return this.phys;
  }
  setSplit(on) {
    on = !!on;
    if (on === this.split) return;
    this.split = on;
    if (on) this._ensurePhys();
    this._size();
    this.view('front', 400);            // re-frame: each half is half as tall
  }
  async setVariants(list) {            // [{id,label,color}] in display order
    const manifest = await loadManifest();
    const want = new Set(list.map(v => v.id));
    for (const id of Object.keys(this.rigs)) if (!want.has(id)) {
      const r = this.rigs[id]; r.rig.dispose(this.scene); r.markers.dispose(this.scene);
      r.label.remove(); delete this.rigs[id];
    }
    for (const v of list) if (!this.rigs[v.id]) {
      const entry = { color: v.color, name: v.label, rig: null, markers: new Markers(this.scene, v.color) };
      entry.label = document.createElement('div');
      entry.label.className = 'vlabel'; entry.label.style.borderColor = v.color; entry.label.style.color = v.color;
      entry.label.innerHTML = `${v.label}<span class="v"></span>`;
      this.labelsEl.appendChild(entry.label);
      this.rigs[v.id] = entry;
      entry.rig = await new Rig().load(this.scene, manifest, v.color);
    }
    const first = !this._order.length;
    this._order = list.map(v => v.id);
    this._layout();
    if (first) this.view('front', 0);
  }
  setLayout(l) { this.layout = l; this._layout(); }
  _slot(i) { return LINEUP.clone().multiplyScalar(i * SPACING); }
  _layout() {
    const ids = this._order.filter(id => this.rigs[id]);
    const lineup = this.layout === 'lineup';
    // operator first (leftmost from the camera), then the robots
    this.op.slot.copy(lineup ? this._slot(0) : this._slot(-1));
    ids.forEach((id, i) => {
      const e = this.rigs[id]; if (!e.rig) return;
      e.slot = lineup ? this._slot(i + 1) : this._slot(0);
      e.rig.place(e.slot);
      e.rig.setGhost(!lineup && id !== this.mirror);
    });
    this.center = (lineup ? this._slot((ids.length) / 2) : this._slot(-0.5)).setY(0.9);
  }
  view(name, ms = 700) {
    const c = (this.center || new THREE.Vector3(0, .9, 0)).clone().add(this._walk || new THREE.Vector3());
    const n = Math.max(1, this._order.length + 1), span = this.layout === 'lineup' ? n * SPACING : 2.2;
    const d = Math.max(2.6, span * 1.25) * (this.split ? 1.35 : 1);
    const to = name === 'side' ? c.clone().add(new THREE.Vector3(-d * .45, .6, d * .9))
      : name === 'top' ? c.clone().add(new THREE.Vector3(-0.01, d * 1.2, 0))
      : c.clone().add(new THREE.Vector3(-d, .45, 0));
    const from = this.camera.position.clone(), tf = this.controls.target.clone(), t0 = performance.now();
    const step = () => {
      const a = ms ? Math.min(1, (performance.now() - t0) / ms) : 1, e = 1 - Math.pow(1 - a, 3);
      this.camera.position.lerpVectors(from, to, e); this.controls.target.lerpVectors(tf, c, e);
      if (a < 1) requestAnimationFrame(step);
    };
    step();
  }
  update(state, liveNums) {
    if (state.mirror !== this.mirror) { this.mirror = state.mirror; this._layout(); }
    // a click shows its end effector at once; the server's setting takes over once it agrees
    if (this.eeOverride && state.settings && state.settings.ee === this.eeOverride) this.eeOverride = null;
    this.op.update(state.human);
    // FOLLOW: shift camera + orbit target by how far the group walked, so the
    // user's own orbit is kept while the scene stays in view (whole body walks)
    const m = this.rigs[this.mirror] || Object.values(this.rigs)[0];
    const b = m && state.variants[this.mirror] && state.variants[this.mirror].rec && state.variants[this.mirror].rec.base;
    if (b) {
      const now = new THREE.Vector3(b.x || 0, 0, -(b.y || 0));
      if (this.follow && this._walk) {
        const dv = now.clone().sub(this._walk);
        this.camera.position.add(dv); this.controls.target.add(dv);
      }
      this._walk = now;
    }
    const ph = state.physics;
    this.setSplit(!!(ph && ph.on));
    if (this.split && this.phys) {
      const me = this.rigs[this.mirror], mv = state.variants[this.mirror], mrec = mv && mv.rec;
      const slot = (me && me.slot) || new THREE.Vector3();
      this.phys.rig.place(slot); this.phys.ghost.place(slot);
      if (mrec) { this.phys.ghost.setBase(mrec.base); this.phys.ghost.pose(mrec.q); }
      this.phys.ghost.setEndEffector(this.eeOverride || (state.settings && state.settings.ee) || 'orca',
                                     mv && mv.ee_mount, mrec && mrec.gripper);
      this.phys.rig.world.visible = !!(ph.q);
      if (ph.q) { this.phys.rig.setBase(ph.base); this.phys.rig.pose(ph.q); }
    }
    for (const [id, e] of Object.entries(this.rigs)) {
      const v = state.variants[id]; const rec = v && v.rec;
      if (e.rig) e.rig.setEndEffector(this.eeOverride || (state.settings && state.settings.ee) || 'orca',
                                      v && v.ee_mount, rec && rec.gripper);
      if (!e.rig || !rec) { e.markers.update(null); continue; }
      e.rig.setBase(rec.base);
      e.rig.pose(rec.q);
      // measure the GLB<->MJ residual once, on the wrist anchor (both sides)
      if (!this.off && rec.geo.left && rec.geo.right) {
        const acc = new THREE.Vector3(); let n = 0;
        for (const s of ['left', 'right']) {
          const g = e.rig.wristWorld(s); if (!g) continue;
          acc.add(g.sub(mjToThree(rec.geo[s].wa || rec.geo[s].wr)).sub(e.slot)); n++;
        }
        if (n) this.off = acc.multiplyScalar(1 / n);
      }
      const off = (this.off || new THREE.Vector3()).clone().add(e.slot);
      e.markers.update(rec.geo, off, this.camera, this.showTargets);
      e._labelAt = mjToThree([rec.head[0], rec.head[1], rec.head[2] + 0.30]).add(off);
      e.label.querySelector('.v').textContent = liveNums[id] || '';
    }
  }
  _frame() {
    this.controls.update();
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const r = this.renderer;
    if (this.split && this.phys) {
      r.setScissorTest(true);
      r.setViewport(0, h / 2, w, h / 2); r.setScissor(0, h / 2, w, h / 2);
      r.render(this.scene, this.camera);
      r.setViewport(0, 0, w, h / 2); r.setScissor(0, 0, w, h / 2);
      r.render(this.phys.scene, this.camera);
      r.setScissorTest(false);
    } else {
      r.setViewport(0, 0, w, h);
      r.render(this.scene, this.camera);
    }
    const hv = this.split ? h / 2 : h;
    const lineup = this.layout === 'lineup';
    for (const id of this._order) {
      const e = this.rigs[id]; if (!e || !e._labelAt) continue;
      const p = e._labelAt.clone().project(this.camera);
      // overlay: one robot place, so only the solid (mirror) robot is labelled —
      // the ghosts are identified by colour (sidebar swatches)
      const yTop = (-p.y * .5 + .5) * hv;
      const vis = p.z < 1 && (lineup || id === this.mirror) && yTop > 100;   // never over the stage chips
      e.label.style.display = vis ? '' : 'none';
      e.label.style.left = `${(p.x * .5 + .5) * w}px`;
      e.label.style.top = `${(-p.y * .5 + .5) * hv}px`;
    }
  }
}
