#!/usr/bin/env python3
"""Bake mabel_full.urdf into a HIERARCHICAL GLB (one node per link, parented by
the joint tree) plus a joints manifest, for the website's articulated THREE.js
viewer. Visual meshes only; zero joint config; Z-up (URDF) -> Y-up (glTF)."""
import os, json
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
from collections import defaultdict
from trimesh.transformations import euler_matrix, translation_matrix, concatenate_matrices

PKG  = os.path.dirname(os.path.abspath(__file__))
WEB  = os.path.abspath(os.path.join(PKG, '..', '..', '..', 'web_gui', 'shared_assets'))
# optional paths: build_hand_variants.py bakes the same robot with NOVA's ORCA v2 hands swapped in
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument('--urdf', default=os.path.join(PKG, 'urdf', 'mabel_full.urdf'))
_ap.add_argument('--out', default=os.path.join(WEB, 'mabel_rig.glb'))
_ap.add_argument('--joints', default=os.path.join(WEB, 'mabel_joints.json'))
_A, _ = _ap.parse_known_args()
URDF, OUT, JOUT = _A.urdf, _A.out, _A.joints
# the canonical robot model — the ONE source of the per-joint qpos0 refs
MJCF = os.path.abspath(os.path.join(PKG, '..', '..', '..',
                       'simulation', 'mabel_mujoco', 'models', 'mabel_full.xml'))

def origin_T(el):
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in (el.get('xyz') or '0 0 0').split()]
    rpy = [float(v) for v in (el.get('rpy') or '0 0 0').split()]
    return concatenate_matrices(translation_matrix(xyz),
                                euler_matrix(rpy[0], rpy[1], rpy[2], 'sxyz'))


# ── Per-joint MuJoCo reference values (qpos0) ────────────────────────────────
# The URDF/GLB geometry is baked at the ASSEMBLED pose (MuJoCo qpos == ref), and
# the :9090 wire streams ABSOLUTE qpos — so every 3-D client must rotate each
# joint by (qpos − ref), never raw qpos. Only the 14 ORCA finger/wrist joints
# have ref != 0 (the thumb chain reaches −0.73 rad), which is why ignoring it
# renders the HANDS contorted — thumbs hyperextended at the last joint — while
# the arms look perfectly fine. `glb_to_rig.py` stamps the same values into the
# apps' rig.json; this puts them in the WEB manifest so the browser viewers can
# do the same subtraction. Shipping the manifest WITHOUT ref is the bug.
def _joint_refs():
    refs = {}
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(MJCF)
        for j in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n:
                # The manifest keys joints by their URDF names, which
                # mjcf_to_urdf.sanitize() derives from the MJCF display names
                # ('Left Thumb Abd' -> 'left_thumb_abd'). Key both so the lookup
                # cannot silently miss and zero every ref.
                v = float(m.qpos0[m.jnt_qposadr[j]])
                refs[n] = v
                refs["_".join(n.split()).lower()] = v
    except Exception as e:                       # mujoco absent / MJCF moved
        # Carry forward whatever the existing manifest already has rather than
        # silently regressing every hand to raw-qpos rendering.
        try:
            old = json.load(open(JOUT))
            refs = {j['name']: j['ref'] for j in old.get('joints', [])
                    if j.get('ref')}
            print(f"WARNING: could not read joint refs from the MJCF ({e}); "
                  f"carried {len(refs)} refs over from the existing manifest")
        except Exception:
            print(f"WARNING: no joint refs available ({e}) — the web hands "
                  f"WILL render contorted until this is regenerated with mujoco")
    return refs


JOINT_REF = _joint_refs()

def resolve(fn):
    return os.path.join(PKG, fn.replace('package://mabel_description/', ''))

root = ET.parse(URDF).getroot()
links = {l.get('name'): l for l in root.findall('link')}
materials = {}
for mat in root.findall('material'):
    c = mat.find('color')
    if c is not None and mat.get('name') and c.get('rgba'):
        materials[mat.get('name')] = [float(v) for v in c.get('rgba').split()]
DEFAULT_RGBA = [0.78, 0.78, 0.78, 1.0]

# joint tree
parent_of = {}        # child_link -> (parent_link, jointT)
joints_meta = []
for j in root.findall('joint'):
    child = j.find('child').get('link')
    parent = j.find('parent').get('link')
    T = origin_T(j.find('origin'))
    parent_of[child] = (parent, T)
    jtype = j.get('type')
    if jtype == 'fixed':
        continue
    ax = j.find('axis')
    axis = [float(v) for v in (ax.get('xyz') if ax is not None else '0 0 1').split()]
    lim = j.find('limit')
    lo = float(lim.get('lower')) if (lim is not None and lim.get('lower')) else None
    hi = float(lim.get('upper')) if (lim is not None and lim.get('upper')) else None
    if jtype == 'continuous':
        lo, hi = -np.pi, np.pi
    name = j.get('name')
    n = name.lower()
    if any(k in n for k in ('thumb', 'index', 'middle', 'ring', 'pinky', 'wrist')):
        grp = 'hands'
    elif 'arm' in n:
        grp = 'arms'
    elif any(k in n for k in ('lift', 'torso', 'neck')):
        grp = 'body'
    elif any(k in n for k in ('steer', 'drive')):
        grp = 'base'
    else:
        grp = 'body'
    joints_meta.append({'name': name, 'type': jtype, 'axis': axis,
                        'lower': lo, 'upper': hi, 'node': child, 'group': grp,
                        'ref': JOINT_REF.get(name, 0.0)})

def world_T(link):
    T = np.eye(4); chain = []; cur = link
    while cur in parent_of:
        p, Tj = parent_of[cur]; chain.append(Tj); cur = p
    for Tj in reversed(chain):
        T = T @ Tj
    return T

ROOT_FIX = euler_matrix(-np.pi / 2, 0, 0, 'sxyz')   # Z-up -> Y-up

# ---- load meshes once; collect for centering and for hierarchy ----
loaded = []   # (link, mesh_local, Tvis, color_rgba)
missing = []
allmin = np.array([np.inf]*3); allmax = np.array([-np.inf]*3)
for name, link in links.items():
    for visual in link.findall('visual'):
        me = visual.find('.//mesh')
        if me is None:
            continue
        path = resolve(me.get('filename'))
        if not os.path.exists(path):
            missing.append(path); continue
        m = trimesh.load(path, force='mesh')
        if m is None or len(m.vertices) == 0:
            continue
        sc = me.get('scale')
        if sc:
            s = [float(v) for v in sc.split()]
            m.apply_scale(s if len(s) == 3 else s[0])
        Tvis = origin_T(visual.find('origin'))
        mat_el = visual.find('material')
        rgba = materials.get(mat_el.get('name')) if mat_el is not None else None
        rgba = (rgba or DEFAULT_RGBA)
        if len(rgba) < 4:
            rgba = list(rgba) + [1.0]
        loaded.append((name, m, Tvis, rgba))
        # world bounds for centering (with root fix applied)
        Tw = ROOT_FIX @ world_T(name) @ Tvis
        v = trimesh.transform_points(m.bounds, Tw)  # 2 corners is approximate; use full verts for accuracy
        vv = trimesh.transform_points(m.vertices, Tw)
        allmin = np.minimum(allmin, vv.min(axis=0))
        allmax = np.maximum(allmax, vv.max(axis=0))

center = (allmin + allmax) / 2.0
# center X/Z on origin, drop so feet sit at y=0
T_center = translation_matrix([-center[0], -allmin[1], -center[2]])
BASE_FRAME = T_center @ ROOT_FIX

# ---- build hierarchical scene ----
scene = trimesh.Scene()
# define every link frame relative to its parent (base_link gets BASE_FRAME under world)
def ensure_frame(link):
    if link in parent_of:
        p, Tj = parent_of[link]
        ensure_frame(p)
        scene.graph.update(frame_from=p, frame_to=link, matrix=Tj)
    else:
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=link, matrix=BASE_FRAME)

for name in links:
    ensure_frame(name)

n = 0
for (link, m, Tvis, rgba) in loaded:
    col = [int(round(max(0.0, min(1.0, c)) * 255)) for c in rgba]
    m.visual = trimesh.visual.ColorVisuals(mesh=m, face_colors=np.tile(col, (len(m.faces), 1)))
    scene.add_geometry(m, node_name=f'{link}__v{n}', parent_node_name=link, transform=Tvis)
    n += 1

os.makedirs(WEB, exist_ok=True)
scene.export(OUT)
groups = defaultdict(int)
for jm in joints_meta:
    groups[jm['group']] += 1
json.dump({'joints': joints_meta}, open(JOUT, 'w'), indent=1)

print(f"meshes: {n}  missing: {len(missing)}  movable joints: {len(joints_meta)}")
print("group counts:", dict(groups))
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
print(f"wrote {JOUT}")
