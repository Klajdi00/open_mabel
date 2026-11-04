// A headless, node-testable clone of viewport3d.js's `Rig` built from the REAL
// robot geometry (web_gui/shared_assets/mabel_rig.glb node graph, dumped to
// rig_nodes.json) + the REAL joint manifest (web_gui/shared_assets/mabel_joints.json).
//
// This exists so the browser IK (web/js/kinematics.js) can be exercised against
// the actual 7-DOF MABEL arm in `node --test`, not just the 3-joint toy arm the
// original kinematics.test.mjs used. setJoint/pose are copied verbatim from
// viewport3d.js's Rig (including the `qpos − ref` displacement rule) so a
// behaviour measured here is the behaviour the operator sees.
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';

const HERE = dirname(fileURLToPath(import.meta.url));
// Walk up until we find the generated joint manifest, so this fixture works from
// any studio's tests/js/ regardless of how deep it sits in the repo.
function findRepoRoot(from){
  let d = from;
  for (let i = 0; i < 10; i++){
    if (existsSync(join(d, 'web_gui', 'shared_assets', 'mabel_joints.json'))) return d;
    const up = dirname(d);
    if (up === d) break;
    d = up;
  }
  throw new Error('mabel_joints.json not found above ' + from);
}
const REPO = findRepoRoot(HERE);

export function loadManifest(){
  return JSON.parse(
    readFileSync(join(REPO, 'web_gui', 'shared_assets', 'mabel_joints.json'), 'utf8'));
}

// Rebuild the GLB scene graph as bare Object3Ds (no meshes — IK only ever reads
// world transforms of named nodes).
function buildGraph(){
  const spec = JSON.parse(readFileSync(join(HERE, 'rig_nodes.json'), 'utf8'));
  // GLTF nodes here carry a column-major `matrix`; GLTFLoader decomposes it into
  // position/quaternion/scale, which is exactly what Rig._p0/_q0 capture.
  const objs = spec.nodes.map((n) => {
    const o = new THREE.Object3D();
    o.name = n.name || '';
    if (n.m) new THREE.Matrix4().fromArray(n.m)
      .decompose(o.position, o.quaternion, o.scale);
    return o;
  });
  spec.nodes.forEach((n, i) => n.children.forEach((c) => objs[i].add(objs[c])));
  const root = new THREE.Group();
  spec.roots.forEach((i) => root.add(objs[i]));
  return root;
}

// Mirrors Rig.load / Rig.setJoint / Rig.pose from web/js/viewport3d.js.
export function buildRig(){
  const manifest = loadManifest();
  const root = buildGraph();
  const world = new THREE.Group();
  world.add(root);

  const base = root.getObjectByName('base_link') || root;
  const joints = {};
  for (const j of manifest.joints){
    const n = root.getObjectByName(j.node);
    if (!n) continue;
    joints[j.name] = { ...j, _n: n, _p0: n.position.clone(), _q0: n.quaternion.clone(),
      _ax: new THREE.Vector3(...j.axis).normalize() };
  }

  const rig = {
    root, world, joints, base,
    baseP0: base.position.clone(), baseQ0: base.quaternion.clone(),
    setJoint(name, val){
      const j = this.joints[name]; if (!j) return;
      const d = val - (j.ref || 0);
      if (j.type === 'prismatic'){
        j._n.position.copy(j._p0).add(j._ax.clone().applyQuaternion(j._q0).multiplyScalar(d));
        j._n.quaternion.copy(j._q0);
      } else {
        j._n.quaternion.copy(j._q0).multiply(new THREE.Quaternion().setFromAxisAngle(j._ax, d));
        j._n.position.copy(j._p0);
      }
    },
    pose(q, bx = 0, by = 0, yaw = 0){
      for (const n in q) this.setJoint(n, q[n]);
      const off = new THREE.Vector3(bx, by, 0).applyQuaternion(this.baseQ0);
      this.base.position.copy(this.baseP0).add(off);
      this.base.quaternion.copy(this.baseQ0)
        .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw));
      this.world.updateMatrixWorld(true);
    },
  };
  world.updateMatrixWorld(true);
  return rig;
}

// The robot's assembled/rest configuration on the wire: every joint at its `ref`
// (qpos == ref is the pose the GLB geometry is baked at). This is what a fresh
// `robot_state` looks like for an idle robot, so it is the honest IK seed.
export function restPose(){
  const q = {};
  for (const j of loadManifest().joints) q[j.name] = j.ref || 0;
  return q;
}

export const ARM = (side) => Array.from({ length: 7 }, (_, i) => `${side}_arm_${i + 1}`);

// World-frame pose of a named rig node.
export function nodePose(rig, name){
  const o = rig.root.getObjectByName(name);
  const p = new THREE.Vector3(), qt = new THREE.Quaternion(), s = new THREE.Vector3();
  o.updateWorldMatrix(true, false);
  o.matrixWorld.decompose(p, qt, s);
  return { pos: p, quat: qt };
}
