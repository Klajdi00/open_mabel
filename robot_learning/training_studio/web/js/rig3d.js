// Compact three.js viewer for the MABEL GLB rig — poses the robot from a joint
// dict streamed by the closed-loop MuJoCo deploy. (Derived from the curator's
// viewport3d Rig posing; trimmed to just what the deploy viewer needs.)
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const RIG_GLB = '/website-assets/mabel_rig.glb';
const JOINTS = '/website-assets/mabel_joints.json';

export class Rig3D {
  constructor(el) {
    this.el = el;
    this.canvas = document.createElement('canvas');
    el.appendChild(this.canvas);
    const r = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = r;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 200);
    this.camera.position.set(1.8, 1.4, 2.2);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(2.5, 3.5, 2);
    this.scene.add(key);
    this.grid = new THREE.GridHelper(8, 32, 0x2a2622, 0x1b1815); this.scene.add(this.grid);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true; this.controls.target.set(0, 0.7, 0);
    this.joints = {}; this.ready = false;
    new ResizeObserver(() => this.resize()).observe(el);
    this._loop = this._loop.bind(this); requestAnimationFrame(this._loop);
    this._load();
  }

  async _load() {
    const manifest = await fetch(JOINTS).then(r => r.json());
    const gltf = await new GLTFLoader().loadAsync(RIG_GLB);
    this.world = gltf.scene;
    this.scene.add(this.world);
    // recenter / scale-fit
    const box = new THREE.Box3().setFromObject(this.world);
    const c = box.getCenter(new THREE.Vector3());
    this.world.position.sub(new THREE.Vector3(c.x, box.min.y, c.z));
    this.controls.target.set(0, (box.max.y - box.min.y) / 2, 0);
    for (const j of manifest.joints) {
      const n = this.world.getObjectByName(j.node);
      if (!n) continue;
      this.joints[j.name] = { j, n, p0: n.position.clone(), q0: n.quaternion.clone(),
        ax: new THREE.Vector3(...j.axis).normalize() };
    }
    this.ready = true;
  }

  // jointDict: normalized joint name -> value (rad or m)
  pose(jointDict) {
    if (!this.ready) return;
    for (const name in this.joints) {
      const jj = this.joints[name];
      const v = jointDict[name];
      if (v == null) continue;
      if (jj.j.type === 'prismatic') {
        jj.n.position.copy(jj.p0).add(jj.ax.clone().applyQuaternion(jj.q0).multiplyScalar(v));
        jj.n.quaternion.copy(jj.q0);
      } else {
        jj.n.quaternion.copy(jj.q0).multiply(new THREE.Quaternion().setFromAxisAngle(jj.ax, v));
      }
    }
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (w < 2 || h < 2) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  _loop() { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(this._loop); }
  destroy() { try { this.renderer.dispose(); } catch {} this.el.innerHTML = ''; }
}

// map MABEL actuator names ("Left Arm 1", "Lift") -> rig joint names
export function actuatorsToJoints(names, values) {
  const out = {};
  names.forEach((nm, i) => {
    const v = values[i];
    if (nm === 'Lift') { out['lift_lower'] = v / 2; out['lift_upper'] = v / 2; return; }
    out[nm.toLowerCase().replace(/\s+/g, '_')] = v;
  });
  return out;
}
