#!/usr/bin/env python3
"""Bake MABEL's web rig with NOVA's ORCA v2 hands swapped in, so the 3D viewers draw the hands that are
actually mounted (NOVA's v2 hands mount on MABEL — NOVA servos/orca_hands.py; the HAL picks per side by
adapter, hardware_bridge/mabel_hw/drivers/hands.py). The v2 subtrees come from NOVA's nova.urdf (seated on
the same l/r_hand_mount centre line as MABEL's v1 — verified 2026-10-05), meshes read from the NOVA
checkout ($NOVA_REPO, default ~/Desktop/NOVA) at bake time only.

    web_gui/control_center/rig_variants/mabel_rig-hands-<L>-<R>.glb + mabel_joints-hands-<L>-<R>.json
    (v1-v1 = web_gui/shared_assets/mabel_rig.glb; the Control Center serves the matching one)

A v2 hand streams its calibration frame (NOVA's v2 URDF frame, no assembled-pose offset), so its joints'
`ref` is 0 in the manifest; MABEL's v1 joints keep their MJCF qpos0 refs. meshopt-compressed with gltfpack
(npx) like NOVA's rig. Re-run after mabel_full.urdf or nova.urdf changes:

    python3 ros/src/mabel_description/build_hand_variants.py
"""
import copy, json, os, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET

PKG = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(PKG, "urdf", "mabel_full.urdf")
NOVA = os.path.expanduser(os.environ.get("NOVA_REPO", "~/Desktop/NOVA"))
NOVA_DESC = os.path.join(NOVA, "ros_ws", "src", "nova_description")
# NOT website/assets (the public site): only the Control Center serves these
WEB = os.path.abspath(os.path.join(PKG, "..", "..", "..", "web_gui", "control_center", "rig_variants"))
MOUNT = {"left": "l_hand_mount", "right": "r_hand_mount"}
HAND = ("thumb", "index", "middle", "ring", "pinky", "wrist")


def subtree(root, mount):
    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    links = {l.get("name"): l for l in root.findall("link")}
    js, ls, st = [], [], [mount]
    while st:
        for j in kids.get(st.pop(), []):
            ch = j.find("child").get("link"); js.append(j); ls.append(links[ch]); st.append(ch)
    return js, ls


def compose(gens):
    root = ET.parse(URDF).getroot()
    nova = ET.parse(os.path.join(NOVA_DESC, "urdf", "nova.urdf")).getroot()
    for side, gen in gens.items():
        if gen != "v2":
            continue
        js, ls = subtree(root, MOUNT[side])
        for e in js + ls:
            root.remove(e)
        vj, vl = subtree(nova, MOUNT[side])
        for e in vl + vj:
            e = copy.deepcopy(e)
            for m in e.iter("mesh"):                       # NOVA's meshes, by absolute path (bake time only)
                m.set("filename", m.get("filename").replace("package://nova_description/", NOVA_DESC + "/"))
            root.append(e)
    return root


def main():
    if not os.path.isdir(NOVA_DESC):
        sys.exit(f"NOVA checkout not found at {NOVA} (set NOVA_REPO)")
    os.makedirs(WEB, exist_ok=True)
    for gens in ({"left": "v2", "right": "v2"}, {"left": "v2", "right": "v1"}, {"left": "v1", "right": "v2"}):
        tag = f"hands-{gens['left']}-{gens['right']}"
        tmp = os.path.join(tempfile.gettempdir(), f"mabel_{tag}.urdf")
        ET.ElementTree(compose(gens)).write(tmp, encoding="unicode")
        full = os.path.join(tempfile.gettempdir(), f"mabel_{tag}_full.glb")
        out, jout = os.path.join(WEB, f"mabel_rig-{tag}.glb"), os.path.join(WEB, f"mabel_joints-{tag}.json")
        print(f"== {tag}")
        subprocess.run([sys.executable, os.path.join(PKG, "urdf_to_glb.py"), "--urdf", tmp, "--out", full, "--joints", jout],
                       check=True, cwd=PKG)
        man = json.load(open(jout))
        n = 0
        for j in man["joints"]:
            side = "left" if j["name"].startswith("left_") else "right" if j["name"].startswith("right_") else None
            if side and gens[side] == "v2" and any(k in j["name"] for k in HAND):
                j["ref"] = 0.0; n += 1
        assert n == 17 * sum(g == "v2" for g in gens.values()), (tag, n)
        json.dump(man, open(jout, "w"), indent=1)
        try:
            subprocess.run(["npx", "-y", "gltfpack", "-i", full, "-o", out, "-cc", "-kn", "-ke", "-si", "0.35"],
                           check=True, capture_output=True, timeout=600)
        except Exception as e:                             # noqa: BLE001
            print(f"WARNING: gltfpack failed ({e}) — shipping the uncompressed GLB"); shutil.copy(full, out)
        print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB) + {os.path.basename(jout)} ({n} v2 hand joints, ref 0)")


if __name__ == "__main__":
    main()
