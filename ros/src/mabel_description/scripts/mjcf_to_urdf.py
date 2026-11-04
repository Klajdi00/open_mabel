#!/usr/bin/env python3
"""Convert the MABEL MuJoCo model (mabel_full.xml) into a ROS URDF.

This is an OFFLINE generation tool, run once whenever the MuJoCo model
changes. It loads the MJCF with MuJoCo itself, so every `class` / `childclass`
default, `eulerseq`, joint anchor and resolved inertia is already baked in --
we just walk the compiled kinematic tree and emit URDF.

    python3 mjcf_to_urdf.py \
        --mjcf ~/open_mabel/simulation/mabel_mujoco/models/mabel_full.xml \
        --out  ../urdf/mabel_full.urdf \
        --mesh-prefix package://mabel_description/meshes

Requires the `mujoco` python package (use the conda env that already has it).
The generated URDF is what ROS uses at runtime -- runtime needs no MuJoCo.

Conventions handled:
  * body pos/quat  -> joint <origin> (parent -> child)
  * hinge/slide    -> revolute|continuous|prismatic, axis = jnt_axis
  * non-zero joint anchor (jnt_pos) -> exact 2-link split so the axis passes
    through the real anchor, not the body origin
  * free joint     -> the body becomes the URDF root link (mobile base)
  * resolved principal inertia -> <inertial> (diagonal in its own frame)
  * group==2 geoms -> <visual>, group==1 geoms -> <collision>
  * mesh scale (0.001 mm->m) referenced on the original STL
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

FREE, BALL, SLIDE, HINGE = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def quat_to_mat(q):
    """MuJoCo quat (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def mat_to_rpy(R):
    """Rotation matrix -> URDF rpy (extrinsic XYZ == R = Rz(yaw)Ry(pitch)Rx(roll))."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def quat_to_rpy(q):
    return mat_to_rpy(quat_to_mat(q))


def rt_inv(p, R):
    """Inverse of a rigid transform (p, R)."""
    return (-R.T @ np.asarray(p), R.T)


def rt_mul(pA, RA, pB, RB):
    """Compose rigid transforms: (pA,RA) then (pB,RB)."""
    return (np.asarray(pA) + RA @ np.asarray(pB), RA @ RB)


def fmt(vals):
    return " ".join(f"{v:.8g}" for v in vals)


# --------------------------------------------------------------------------- #
# mesh-file map (MjModel doesn't store file paths -> read them from the XML)
# --------------------------------------------------------------------------- #
def build_mesh_file_map(mjcf_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    default_scale = "0.001 0.001 0.001"
    # global default <mesh scale=.../>
    for d in root.iter("default"):
        for m in d.findall("mesh"):
            if m.get("scale"):
                default_scale = m.get("scale")
    out = {}
    asset = root.find("asset")
    if asset is not None:
        for m in asset.findall("mesh"):
            name = m.get("name")
            f = m.get("file")
            if not name or not f:
                continue
            rel = f.split("assets/meshes/", 1)[-1] if "assets/meshes/" in f else os.path.basename(f)
            out[name] = (rel, m.get("scale", default_scale))
    return out


def sanitize(name):
    """'Left Thumb MCP' -> 'left_thumb_mcp', 'FL Steer' -> 'fl_steer'."""
    return "_".join(name.split()).lower()


# --------------------------------------------------------------------------- #
# main conversion
# --------------------------------------------------------------------------- #
def convert(mjcf_path, mesh_prefix):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    mesh_files = build_mesh_file_map(mjcf_path)

    def bname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)

    def jname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)

    def mesh_name(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)

    materials = {}   # rgba tuple -> material name

    def mat_for(rgba):
        key = tuple(round(float(c), 4) for c in rgba)
        if key not in materials:
            materials[key] = f"mat_{len(materials)}"
        return materials[key]

    links = []   # list of xml strings
    joints = []  # list of xml strings

    # ---- per-body geom collection -----------------------------------------
    def geoms_of(body_id):
        adr = model.body_geomadr[body_id]
        num = model.body_geomnum[body_id]
        return range(adr, adr + num)

    def geom_xml(g, tag):
        """tag = 'visual' or 'collision'. Returns xml string or None."""
        gtype = model.geom_type[g]
        pos = np.array(model.geom_pos[g])
        Rg = quat_to_mat(model.geom_quat[g])
        rpy = mat_to_rpy(Rg)
        # color
        matid = model.geom_matid[g]
        rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[g]
        if tag == "visual" and float(rgba[3]) == 0.0:
            return None  # skip invisible visuals
        # geometry
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            nm = mesh_name(mid)
            if nm not in mesh_files:
                return None
            rel, scale = mesh_files[nm]
            # MuJoCo recenters mesh verts on load (mesh_pos/mesh_quat map the
            # recentered frame -> original STL frame). We reference the ORIGINAL
            # STL, so the visual origin must be T_geom o T_mesh^-1.
            Rm = quat_to_mat(model.mesh_quat[mid])
            pm = np.array(model.mesh_pos[mid])
            pmi, Rmi = rt_inv(pm, Rm)
            pos, Rv = rt_mul(model.geom_pos[g], Rg, pmi, Rmi)
            rpy = mat_to_rpy(Rv)
            geo = (f'<geometry><mesh filename="{mesh_prefix}/{rel}" '
                   f'scale="{scale}"/></geometry>')
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]
            geo = f'<geometry><box size="{fmt([2*s[0], 2*s[1], 2*s[2]])}"/></geometry>'
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            s = model.geom_size[g]
            geo = f'<geometry><cylinder radius="{s[0]:.6g}" length="{2*s[1]:.6g}"/></geometry>'
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            s = model.geom_size[g]
            geo = f'<geometry><sphere radius="{s[0]:.6g}"/></geometry>'
        else:
            return None  # planes etc.

        origin = f'<origin xyz="{fmt(pos)}" rpy="{fmt(rpy)}"/>'
        if tag == "visual":
            return (f'    <visual>{origin}{geo}'
                    f'<material name="{mat_for(rgba)}"/></visual>')
        return f'    <collision>{origin}{geo}</collision>'

    def inertial_xml(body_id, override_mass=None):
        m = float(model.body_mass[body_id]) if override_mass is None else override_mass
        if m <= 0:
            m = 1e-4
            ipos = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            I = (1e-6, 1e-6, 1e-6)
        else:
            ipos = model.body_ipos[body_id]
            rpy = quat_to_rpy(model.body_iquat[body_id])
            I = model.body_inertia[body_id]
            I = (max(I[0], 1e-9), max(I[1], 1e-9), max(I[2], 1e-9))
        return (f'    <inertial><origin xyz="{fmt(ipos)}" rpy="{fmt(rpy)}"/>'
                f'<mass value="{m:.8g}"/>'
                f'<inertia ixx="{I[0]:.8g}" ixy="0" ixz="0" '
                f'iyy="{I[1]:.8g}" iyz="0" izz="{I[2]:.8g}"/></inertial>')

    def link_xml(name, body_id, override_mass=None, geoms=True):
        parts = [f'  <link name="{name}">', inertial_xml(body_id, override_mass)]
        if geoms:
            for g in geoms_of(body_id):
                grp = model.geom_group[g]
                if grp == 2:
                    v = geom_xml(g, "visual")
                    if v:
                        parts.append(v)
                elif grp == 1:
                    c = geom_xml(g, "collision")
                    if c:
                        parts.append(c)
        parts.append("  </link>")
        return "\n".join(parts)

    # ---- walk every body ---------------------------------------------------
    root_link = None
    for b in range(1, model.nbody):          # skip world (0)
        name = bname(b)
        parent = model.body_parentid[b]
        njnt = int(model.body_jntnum[b])
        jadr = int(model.body_jntadr[b])

        # free-joint body == robot root
        if njnt == 1 and model.jnt_type[jadr] == FREE:
            root_link = name
            links.append(link_xml(name, b))
            continue

        links.append(link_xml(name, b))

        if parent == 0:
            # a fixed body hanging straight off world (shouldn't happen here)
            continue

        p_name = bname(parent)
        origin_xyz = model.body_pos[b]
        origin_rpy = quat_to_rpy(model.body_quat[b])

        if njnt == 0:
            # fixed joint
            joints.append(
                f'  <joint name="{name}_fixed" type="fixed">\n'
                f'    <parent link="{p_name}"/>\n    <child link="{name}"/>\n'
                f'    <origin xyz="{fmt(origin_xyz)}" rpy="{fmt(origin_rpy)}"/>\n'
                f'  </joint>')
            continue

        # exactly one articulated joint
        j = jadr
        jt = model.jnt_type[j]
        axis = model.jnt_axis[j]
        anchor = model.jnt_pos[j]
        rng = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        jn = sanitize(jname(j))

        if jt == SLIDE:
            jtype = "prismatic"
        elif jt == HINGE:
            jtype = "revolute" if limited else "continuous"
        else:
            raise SystemExit(f"unsupported joint type {jt} on body {name}")

        # URDF q=0 is the assembled pose == MuJoCo qpos=ref, so URDF
        # displacement q = qpos - ref. Shift the limits by ref accordingly.
        ref = float(model.qpos0[model.jnt_qposadr[j]])
        if limited:
            limit = (f'<limit lower="{rng[0] - ref:.6g}" upper="{rng[1] - ref:.6g}" '
                     f'effort="50" velocity="5"/>')
        else:
            limit = '<limit effort="50" velocity="20"/>'

        anchor_off = float(np.linalg.norm(anchor))
        if anchor_off < 1e-9:
            # axis through body origin -> single joint, child frame = body origin
            joints.append(
                f'  <joint name="{jn}" type="{jtype}">\n'
                f'    <parent link="{p_name}"/>\n    <child link="{name}"/>\n'
                f'    <origin xyz="{fmt(origin_xyz)}" rpy="{fmt(origin_rpy)}"/>\n'
                f'    <axis xyz="{fmt(axis)}"/>\n    {limit}\n  </joint>')
        else:
            # exact split: parent --(joint @ anchor)--> <name>__anchor --(fixed -anchor)--> name
            anchor_link = f"{name}__anchor"
            tiny_inertial = ('    <inertial><origin xyz="0 0 0" rpy="0 0 0"/>'
                             '<mass value="1e-4"/><inertia ixx="1e-6" ixy="0" ixz="0" '
                             'iyy="1e-6" iyz="0" izz="1e-6"/></inertial>')
            links.append(f'  <link name="{anchor_link}">\n{tiny_inertial}\n  </link>')
            joints.append(
                f'  <joint name="{jn}" type="{jtype}">\n'
                f'    <parent link="{p_name}"/>\n    <child link="{anchor_link}"/>\n'
                f'    <origin xyz="{fmt(origin_xyz + model_rot(origin_rpy, anchor))}" '
                f'rpy="{fmt(origin_rpy)}"/>\n'
                f'    <axis xyz="{fmt(axis)}"/>\n    {limit}\n  </joint>')
            joints.append(
                f'  <joint name="{name}_anchor_fixed" type="fixed">\n'
                f'    <parent link="{anchor_link}"/>\n    <child link="{name}"/>\n'
                f'    <origin xyz="{fmt(-anchor)}" rpy="0 0 0"/>\n  </joint>')

    # ---- assemble ----------------------------------------------------------
    mat_blocks = []
    for rgba, mname in materials.items():
        mat_blocks.append(
            f'  <material name="{mname}"><color rgba="{fmt(rgba)}"/></material>')

    # Dummy massless root (KDL/robot_state_publisher dislike inertia on the root
    # link). base_link == mabel_chassis frame via a fixed joint.
    base = [
        '  <link name="base_link"/>',
        '  <joint name="base_joint" type="fixed">\n'
        f'    <parent link="base_link"/>\n    <child link="{root_link}"/>\n'
        '    <origin xyz="0 0 0" rpy="0 0 0"/>\n  </joint>',
    ]

    body = ['<?xml version="1.0"?>',
            '<robot name="mabel">',
            "  <!-- AUTO-GENERATED from mabel_full.xml by scripts/mjcf_to_urdf.py."
            " Do not hand-edit. -->",
            *mat_blocks,
            *base,
            *links,
            *joints,
            "</robot>"]
    return "\n".join(body), root_link, len(links), len(joints)


def model_rot(rpy, vec):
    """Rotate vec by the rpy (so the anchor offset lands in the parent frame)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return (Rz @ Ry @ Rx) @ np.asarray(vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mesh-prefix", default="package://mabel_description/meshes")
    args = ap.parse_args()

    urdf, root, nlinks, njoints = convert(os.path.expanduser(args.mjcf),
                                          args.mesh_prefix)
    out = os.path.expanduser(args.out)
    with open(out, "w") as f:
        f.write(urdf)
    print(f"wrote {out}")
    print(f"  root link : {root}")
    print(f"  links     : {nlinks}")
    print(f"  joints    : {njoints}")


if __name__ == "__main__":
    main()
