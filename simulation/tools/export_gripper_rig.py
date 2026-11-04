#!/usr/bin/env python3
"""export_gripper_rig.py — MABEL wearing the ORIGINAL OpenArm v1 gripper, for the iOS + Vision Pro rigs.

The Retarget Lab's End effector toggle (setting "ee" = "gripper") draws OpenArm's link7 (wrist housing + the
gripper's drive motor), the gripper hand and its two fingers on J7, and hides everything else under J7 (the ORCA
hand-mount adapter and the ORCA hand). This bakes exactly that as an app overlay, the same shape as NOVA's
nova_gripper_v1_<side>.json: the apps splice it under `<l|r>_hand_mount` (the J7 node) when the gateway's
hello.hands reports that side as "gripper", dropping the J7 node's own subtree.

Same sources as the lab, so the two cannot disagree:
  * placement   mabel.teleop_engine.end_effector.gripper_frame (the lab worker's `ee_mount`) + GRIPPER spec
  * meshes      simulation/mabel_mujoco/models/meshes/visual/{arm/link7_*,gripper/*}.obj (Enactic, Apache-2.0)
  * joint       `<side>_gripper` = ONE finger's opening, 0 closed .. 0.044 m open (the lab's opening / 2);
                the second finger is `<side>_gripper_mirror`, "mimic": "<side>_gripper"

Writes Resources/MabelRig/mabel_gripper_v1_{left,right}.json + mabel_gripper_v1_*.ply into BOTH apps.
PLYs are positions only (colour goes in the part's rgb — a face-colour PLY crashes the loader).

    ~/anaconda3/bin/python3 simulation/tools/export_gripper_rig.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "controller"))
MESHES = REPO / "simulation" / "mabel_mujoco" / "models" / "meshes"
SCENE = REPO / "simulation" / "mabel_mujoco" / "scenes" / "mabel_bare.xml"   # = the lab's CFG.SCENE
APPS = [REPO / "apps" / "visionos" / "MabelTeleop" / "Resources" / "MabelRig"]
MOUNT = {"left": "l_hand_mount", "right": "r_hand_mount"}
FACES = {"link7_0": 4000, "link7_1": 1500, "hand_0": 3000, "hand_1": 3000, "finger_0": 2500, "finger_1": 2500}
KEEP_WHOLE = 400                                         # a component this small is never decimated
BOUNDS_TOL_MM = 1.0                                      # decimation may not move a part's bbox further
COVER_TOL_MM = 0.5                                       # p99.9 distance, source surface -> decimated mesh


def coverage_gap(src, dec, n=3000):
    """p99.9 distance (mesh units) from points sampled on `src`'s surface to `dec` — what decimation lost."""
    pts, _ = trimesh.sample.sample_surface(src, n, seed=0)
    return float(np.percentile(trimesh.proximity.closest_point(dec, pts)[1], 99.9))


def mounts():
    """{side: 4x4} the gripper (OpenArm link7 frame) in the J7 body frame — worker._grip_mount, verbatim."""
    import mujoco
    from mabel.teleop_engine.end_effector import gripper_frame
    from mabel.teleop_engine.hand import HandOptimizer
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    out = {}
    for s in ("left", "right"):
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_palm")
        out[s] = gripper_frame(m, s, palm, HandOptimizer(m, s).R_rob)
    return out


def obj(rel):
    v, f = [], []
    for line in open(MESHES / rel):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            f += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
    return np.asarray(v), np.asarray(f, dtype=np.int64)


def part(rel, offset, flip=False):
    """One OBJ as the lab's viewer places it in its group: scale 0.001 (y negated when flipped), then + offset."""
    import fast_simplification as fs
    v, f = obj(rel)
    name = rel.rsplit("/", 1)[-1][:-4]
    t = trimesh.Trimesh(v, f, process=False)
    t.merge_vertices()                                   # Enactic's OBJs split every vertex per face
    if len(t.faces) > FACES[name]:
        # Decimate PER CONNECTED COMPONENT and leave small ones whole: quadric decimation of the whole OBJ
        # silently DELETED link7_1's 252-face open cylinder wall (the gripper's drive motor, z 614.5-647 mm)
        # and kept only its 6.5 mm end cap — the apps drew a thin disc where the motor is.
        comps = t.split(only_watertight=False)
        budget = FACES[name] / len(t.faces)
        out = []
        for c in comps:
            # Halve the reduction until the result still COVERS the component's surface (bounds alone missed
            # 6 mm of eroded chamfers/bosses); at zero it is kept as authored.
            r = 1.0 - max(budget, KEEP_WHOLE / len(c.faces)) if len(c.faces) > KEEP_WHOLE else 0.0
            while r > 0.05:
                p, q = fs.simplify(c.vertices.astype(np.float32), c.faces.astype(np.int32), target_reduction=r)
                s = trimesh.Trimesh(np.asarray(p, float), np.asarray(q), process=False)
                if len(s.faces) and coverage_gap(c, s) <= COVER_TOL_MM:
                    c = s
                    break
                r /= 2
            out.append(c)
        d = trimesh.util.concatenate(out)
        err = np.abs(d.bounds - t.bounds).max()
        if err > BOUNDS_TOL_MM:                          # a decimated part must still fill its source's box
            raise SystemExit(f"{rel}: decimation moved the bounds by {err:.2f} mm (> {BOUNDS_TOL_MM} mm)")
        t = d
    v, f = t.vertices, t.faces
    v = np.asarray(v, float) * 0.001
    if flip:
        v[:, 1] *= -1.0
        f = np.asarray(f)[:, ::-1]                       # keep the winding outward after the mirror
    return name, trimesh.Trimesh(v + np.asarray(offset), f, process=False)


def col(T):
    return [float(x) for x in np.asarray(T, float).T.ravel()]     # rig.json "m" is column-major


def tr(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def build():
    from mabel.teleop_engine.end_effector import GRIPPER
    G = GRIPPER
    rgb = lambda n: [int(G["colors"][n][i:i + 2], 16) for i in (1, 3, 5)]
    files, overlays = {}, {}
    M = mounts()
    for side in ("left", "right"):
        tag = f"mabel_gripper_v1_{side}"
        groups = []

        def grp(name, parent, T, items, **kw):
            parts = []
            for k, (n, mesh) in enumerate(items):
                fn = f"{tag}_{name.split(side + '_', 1)[-1]}_{n}.ply"
                files[fn] = mesh
                parts.append({"file": fn, "rgb": rgb(n)})
            groups.append({"name": name, "parent": parent, "m": col(T), **kw, "parts": parts})

        housing = [part(r, G["housing_geom_pos"]) for r in G["meshes"]["housing"]]
        grp(f"{side}_openarm_link7", MOUNT[side], M[side], housing, type="fixed")
        hand = [part(r, G["hand_geom_pos"]) for r in G["meshes"]["hand"]]
        grp(f"{side}_gripper_base", f"{side}_openarm_link7", tr(G["hand_pos"]), hand, type="fixed")
        for i, f in enumerate(G["fingers"]):
            fing = [part(r, f["geom_pos"], f["flip"]) for r in G["meshes"]["finger"]]
            kw = {"joint": f"{side}_gripper", "type": "prismatic", "axis": list(f["axis"])} if i == 0 else \
                 {"joint": f"{side}_gripper_mirror", "type": "prismatic", "axis": list(f["axis"]),
                  "mimic": f"{side}_gripper"}
            grp(f"{side}_gripper_finger{i + 1}", f"{side}_gripper_base", tr(f["pos"]), fing, **kw)
        overlays[tag] = {"groups": groups}
    return overlays, files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="build and report, write nothing")
    a = ap.parse_args()
    overlays, files = build()
    for tag, o in overlays.items():
        g0 = o["groups"][0]
        print(f"{tag}: {len(o['groups'])} groups, mount on {g0['parent']} at "
              f"{np.round(np.asarray(g0['m'][12:15]) * 1e3, 1).tolist()} mm")
    print(f"{len(files)} PLYs, {sum(len(m.faces) for m in files.values())} faces")
    if a.check:
        return
    for d in APPS:
        for p in d.glob("mabel_gripper_v1_*"):
            p.unlink()
        for tag, o in overlays.items():
            (d / f"{tag}.json").write_text(json.dumps(o, indent=1) + "\n")
        for fn, mesh in files.items():
            mesh.export(d / fn, file_type="ply", encoding="binary")
        print("wrote", d.relative_to(REPO))


if __name__ == "__main__":
    main()
