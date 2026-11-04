#!/usr/bin/env python3
"""Validate the generated URDF against the MuJoCo model (numeric FK check).

Compares, at the assembled / zero pose:
  A) every URDF link's world transform (chained from the parsed URDF joint
     origins) vs MuJoCo's body world transform (data.xpos/xmat at qpos0)
  B) every URDF mesh <visual> placement: the ORIGINAL STL vertices mapped by
     link_world o visual_origin vs MuJoCo's rendered (recentered) vertices in
     world. This catches the mesh-recenter bug.

Run with the conda env that has mujoco + trimesh.
"""
import os
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

# Repo root, derived from this script's location so it is portable across
# machines/users (e.g. /home/mabel on Ubuntu 22 vs /home/mabel on the
# Jetson): scripts/ -> mabel_description/ -> src/ -> ros/ -> <repo root>.
_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MJCF = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else \
    os.path.join(_REPO, "simulation/mabel_mujoco/models/mabel_full.xml")
URDF = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else \
    os.path.join(os.path.dirname(__file__), "..", "urdf", "mabel_full.urdf")
MESHDIR = os.path.join(_REPO, "simulation/mabel_mujoco/assets/meshes")


def quat2mat(q):
    w, x, y, z = np.asarray(q) / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1 - 2*(x*x+y*y)]])


def rpy2mat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def T(p, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


# ---- MuJoCo ground truth ---------------------------------------------------
m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)


def bname(i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)


mj_body_world = {bname(i): T(np.array(d.xpos[i]), np.array(d.xmat[i]).reshape(3, 3))
                 for i in range(m.nbody)}

# ---- parse URDF ------------------------------------------------------------
root = ET.parse(URDF).getroot()
joints = {}  # child -> (parent, origin_T)
for j in root.findall("joint"):
    parent = j.find("parent").get("link")
    child = j.find("child").get("link")
    o = j.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0, 0, 0]
    joints[child] = (parent, T(xyz, rpy2mat(rpy)))

link_names = [l.get("name") for l in root.findall("link")]
ROOT = "mabel_chassis"


def link_world(name, cache={}):
    if name in cache:
        return cache[name]
    if name == ROOT:
        cache[name] = mj_body_world[ROOT]   # anchor URDF root to MuJoCo root
        return cache[name]
    parent, origin = joints[name]
    W = link_world(parent) @ origin
    cache[name] = W
    return W


# ---- A) link world pose check ---------------------------------------------
print("=== A) link world poses (URDF FK vs MuJoCo) ===")
max_perr = max_rerr = 0.0
worst = ""
for name in link_names:
    if name.endswith("__anchor"):
        continue
    if name not in mj_body_world:
        continue
    Wu = link_world(name)
    Wm = mj_body_world[name]
    perr = np.linalg.norm(Wu[:3, 3] - Wm[:3, 3])
    rerr = np.linalg.norm(Wu[:3, :3] - Wm[:3, :3])
    if perr > max_perr:
        max_perr, worst = perr, name
    max_rerr = max(max_rerr, rerr)
print(f"  max position error : {max_perr:.3e} m   (worst link: {worst})")
print(f"  max rotation error : {max_rerr:.3e}")

# ---- B) mesh visual placement check ---------------------------------------
# map mesh rel-path -> original verts (scaled)
print("=== B) mesh visual vertex placement (URDF vs MuJoCo) ===")
# build geom world recentered verts per mesh-name+body
mesh_files = {}   # rel -> verts
def load_verts(rel):
    if rel not in mesh_files:
        tm = trimesh.load(os.path.join(MESHDIR, rel))   # default processing == MuJoCo
        mesh_files[rel] = np.array(tm.vertices) * 0.001
    return mesh_files[rel]

# MuJoCo: for each mesh geom, world recentered verts
def mesh_name(i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i)

# parse <visual> per link with mesh filename + origin
prefix = "package://mabel_description/meshes/"
max_verr = 0.0
worstv = ""
checked = 0
for link in root.findall("link"):
    lname = link.get("name")
    if lname not in mj_body_world and lname.endswith("__anchor"):
        continue
    for vis in link.findall("visual"):
        meshel = vis.find("geometry/mesh")
        if meshel is None:
            continue
        fn = meshel.get("filename").replace(prefix, "")
        o = vis.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        Tvis = T(xyz, rpy2mat(rpy))
        Vo = load_verts(fn)
        # URDF world points
        Wl = link_world(lname)
        Pu = (Wl @ Tvis @ np.c_[Vo, np.ones(len(Vo))].T).T[:, :3]
        # MuJoCo: find the mesh geom on this body matching by vertex count
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lname)
        gsel = None
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                gmid = m.geom_dataid[g]
                if m.mesh_vertnum[gmid] == len(Vo):
                    gsel = (g, gmid)
                    break
        if gsel is None:
            continue
        g, gmid = gsel
        adr = m.mesh_vertadr[gmid]
        Vmj = np.array(m.mesh_vert[adr:adr+m.mesh_vertnum[gmid]])
        gx = np.array(d.geom_xpos[g])
        gR = np.array(d.geom_xmat[g]).reshape(3, 3)
        Pm = (gx + (gR @ Vmj.T).T)
        err = np.abs(Pu - Pm).max()
        checked += 1
        if err > max_verr:
            max_verr, worstv = err, f"{lname}:{fn.split('/')[-1]}"
print(f"  meshes checked     : {checked}")
print(f"  max vertex error   : {max_verr:.3e} m   (worst: {worstv})")
print()
# rotation tolerance is loose: rpy is singular near pitch=±pi/2 (head chain),
# a representational round-off only -- position & vertex errors are what matter.
ok = max_perr < 1e-3 and max_rerr < 5e-3 and max_verr < 1e-3
print("RESULT:", "PASS (sub-mm) ✓" if ok else "FAIL ✗")
