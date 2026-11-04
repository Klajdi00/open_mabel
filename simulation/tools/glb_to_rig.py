#!/usr/bin/env python3
"""
glb_to_rig.py -- regenerate the COMPLETE on-device robot rig (iOS + Vision Pro) from
the website GLB, so the apps' 3D model always reflects the MuJoCo MJCF.

Pipeline (run in order whenever models/mabel_full.xml changes):
    1. ros/src/mabel_description/scripts/mjcf_to_urdf.py   MJCF  -> URDF
    2. ros/src/mabel_description/urdf_to_glb.py            URDF  -> website GLB + joints.json
    3. simulation/tools/glb_to_rig.py  (THIS)                  GLB   -> iOS/VP rig.json + PLY

This SUPERSEDES export_hand_rig.py (which only re-emitted the hand subtrees onto a
hand-maintained body rig, so a body change -- e.g. the lift_base drop -- never reached
the apps). Here we walk the WHOLE GLB scene graph from `base_link` and emit one rig
group PER NODE (the same per-node convention export_hand_rig used for the fingers, which
the on-device loader already renders): local transform relative to the parent, the joint
id + axis if the node articulates (from mabel_joints.json), and its mesh as a binary PLY.

    python3 simulation/tools/glb_to_rig.py
"""
import os
import json
import glob
import shutil
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
GLB = os.path.join(HERE, "..", "..", "web_gui", "shared_assets", "mabel_rig.glb")
JOINTS = os.path.join(HERE, "..", "..", "web_gui", "shared_assets", "mabel_joints.json")
MJCF = os.path.join(HERE, "..", "mabel_mujoco", "models", "mabel_full.xml")

# Both apps consume an identical Resources/MabelRig folder:
#   apps/visionos/MabelTeleop/Resources/MabelRig       (primary output)
#   apps/ios/MABEL Mobile App/Resources/MabelRig       (mirrored copy)
def _rigs(app_root, *want):
    hits = sorted(glob.glob(os.path.join(HERE, "..", "..", app_root, "**", "MabelRig"),
                            recursive=True))
    return [h for h in hits if all(w.lower() in h.lower() for w in want)]

IOS_RIG = _rigs("apps", "visionos")   # primary output (name kept for the code below)
VP_RIG = []
V2_RIG = []
IOS2_RIG = _rigs("apps", "ios")    # the iPhone app
ROOT_LINK = "base_link"          # the rig root (parent=None); carries the Z-up->Y-up bake


def main():
    if not IOS_RIG:
        raise SystemExit("apps/visionos/.../MabelRig folder not found")
    rig_dir = IOS_RIG[0]

    scene = trimesh.load(GLB)
    graph = scene.graph
    edges = graph.to_edgelist()
    parent_of = {c: p for p, c, _ in edges}
    children_of = {}
    for p, c, _ in edges:
        children_of.setdefault(p, []).append(c)

    # node -> (ios joint id, axis, type) for EVERY articulated joint (base, lift,
    # arms, hands). The manifest carries the joint kind (revolute / continuous /
    # prismatic); the app slides prismatic joints and rotates the rest, so the
    # lift MUST be emitted as prismatic or it gets rotated about Z (invisible).
    jd = json.load(open(JOINTS))
    node_to_joint = {j["node"]: (j["name"], j["axis"], j.get("type", "revolute"))
                     for j in jd["joints"] if j.get("node")}

    # Per-joint MuJoCo reference values (qpos0). The URDF/GLB geometry is baked
    # at the ASSEMBLED pose == MuJoCo qpos = ref, so a client must rotate each
    # joint by (streamed qpos − ref), NOT raw qpos — the thumb (Abd/PIP/DIP carry
    # ref up to −0.73 rad) rendered visibly wrong wherever ref was ignored.
    # Stamped into rig.json as an optional "ref" per articulated group.
    joint_ref = {}
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(MJCF)
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            ios_id = name.lower().replace(" ", "_")
            joint_ref[ios_id] = float(m.qpos0[m.jnt_qposadr[j]])
    except Exception as e:  # pragma: no cover — rig still regenerates without refs
        print(f"WARNING: could not read joint refs from the MJCF ({e}); "
              "rig.json will carry no 'ref' fields")

    def world(n):
        return np.array(graph.get(n)[0], dtype=float)

    # Clear any previous rig PLYs (body rig_*, legacy hand_*, and our node PLYs).
    for fn in os.listdir(rig_dir):
        if fn.endswith(".ply"):
            os.remove(os.path.join(rig_dir, fn))

    groups = []
    n_parts = [0]

    def emit(n, rig_parent, parent_world):
        local = np.linalg.inv(parent_world) @ world(n)
        if n == ROOT_LINK:
            # The GLB recenters the model for the WEBSITE (urdf_to_glb.py
            # T_center: bbox-centred X/Z, feet dropped to y=0) and bakes that
            # translation into base_link. The apps park the rig root at the
            # live odom pose, which is the FREE-JOINT/base_link frame — so the
            # bake double-offsets the whole robot (~23 cm back, 8 cm up) and
            # every world-frame overlay (camera pointclouds, SLAM mesh)
            # misregisters against the body. Keep only the Z-up->Y-up
            # rotation; base_link's local translation must be zero.
            local = local.copy()
            local[:3, 3] = 0.0
        g = {"name": n, "parent": rig_parent, "m": local.T.flatten().tolist()}
        if n in node_to_joint:
            jid, axis, jtype = node_to_joint[n]
            g["joint"] = jid
            # App distinguishes "prismatic" (slide) from everything else (rotate);
            # map continuous->revolute so wheels still spin, keep prismatic for the lift.
            g["type"] = "prismatic" if jtype == "prismatic" else "revolute"
            g["axis"] = [float(a) for a in axis]
            ref = joint_ref.get(jid, 0.0)
            if abs(ref) > 1e-9:
                g["ref"] = ref
        else:
            g["type"] = "fixed"
        parts = []
        geom_name = graph.get(n)[1]
        if geom_name is not None and geom_name in scene.geometry:
            mesh = scene.geometry[geom_name]
            try:
                col = mesh.visual.main_color[:3]
            except Exception:
                col = (180, 180, 180)
            fname = f"rig_{n}.ply"
            mesh.export(os.path.join(rig_dir, fname), file_type="ply", encoding="binary")
            parts.append({"file": fname, "rgb": [int(c) for c in col]})
            n_parts[0] += 1
        g["parts"] = parts
        groups.append(g)

    def walk(n, rig_parent, parent_world):
        emit(n, rig_parent, parent_world)
        for c in children_of.get(n, []):
            walk(c, n, world(n))

    # Root: base_link under the GLB world frame (its m carries the Z-up->Y-up bake).
    walk(ROOT_LINK, None, np.eye(4))

    rig = {"groups": groups}
    out_json = os.path.join(rig_dir, "rig.json")
    json.dump(rig, open(out_json, "w"))

    # Mirror the whole folder to every app that bundles the rig (identical asset).
    for targets in (VP_RIG, V2_RIG, IOS2_RIG):
        if not targets:
            continue
        dst = targets[0]
        for fn in os.listdir(dst):
            if fn.endswith(".ply") or fn == "rig.json":
                os.remove(os.path.join(dst, fn))
        for fn in os.listdir(rig_dir):
            if fn.endswith(".ply") or fn == "rig.json":
                shutil.copy2(os.path.join(rig_dir, fn), os.path.join(dst, fn))

    jcount = sum(1 for g in groups if g.get("joint"))
    mb = sum(os.path.getsize(os.path.join(rig_dir, fn)) for fn in os.listdir(rig_dir)
             if fn.endswith(".ply")) / 1e6
    print(f"rig.json: {len(groups)} groups ({jcount} articulated joints, {n_parts[0]} meshes, "
          f"{mb:.1f} MB)")
    print(f"  Vision Pro -> {out_json}")
    if VP_RIG:
        print(f"  Vision Pro -> {os.path.join(VP_RIG[0], 'rig.json')}")
    if V2_RIG:
        print(f"  VisionOS_V2 -> {os.path.join(V2_RIG[0], 'rig.json')}")
    if IOS2_RIG:
        print(f"  iPhone -> {os.path.join(IOS2_RIG[0], 'rig.json')}")
    nref = sum(1 for g in groups if "ref" in g)
    print(f"  joints with nonzero MuJoCo ref: {nref}")


if __name__ == "__main__":
    main()
