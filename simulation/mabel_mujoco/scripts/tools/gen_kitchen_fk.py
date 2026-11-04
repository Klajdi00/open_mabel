#!/usr/bin/env python3
"""
gen_kitchen_fk.py -- convert the classic Franka Kitchen environment (D4RL /
relay-policy-learning "Adept envs", Apache-2.0, vendored under
third_party/franka_kitchen/) into a single MABEL-compatible include:

    scenes/kitchen_fk_env.xml

What it does (the third_party files are NOT modified):
  1. merges the seven item_assets (oven, counters, backwall, slide cabinet,
     hinge cabinet, microwave, kettle): <asset> + <default> + <equality> blocks
     and the articulated body chains, dropping the Franka robot entirely;
  2. rewrites mesh/texture paths to be relative to scenes/
     (../third_party/franka_kitchen/{meshes,textures}/...);
  3. applies a uniform geometric scale S to every pos/size/slide-range so the
     original ~1.6 m counter height (built for a pedestal-mounted Franka) comes
     down to ~1.6*S m, reachable by the MABEL mobile manipulator;
  4. wraps everything in a static body "kitchen_root" placed PLACE_POS from the
     scene origin, keeping the mandatory 1.2 m clear spawn circle.

The include is consumed by scenes/mabel_kitchen_fk.xml. Re-run after changing
SCALE / PLACE_POS:
    python3 scripts/tools/gen_kitchen_fk.py
"""
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FK = os.path.join(ROOT, "third_party", "franka_kitchen")
OUT = os.path.join(ROOT, "scenes", "kitchen_fk_env.xml")

SCALE = 0.65                 # 1.6 m counters -> ~1.04 m (MABEL lift range 0-0.635 m)
PLACE_POS = (0.0, 1.85, 0.0)  # kitchen_root offset: nearest counter ~1.33 m from origin

ITEMS = ["oven", "counters", "backwall", "slidecabinet", "hingecabinet",
         "microwave", "kettle"]
# wrapper placements from the original kitchen_env_model.xml (pre-scale, desk frame)
DESK_POS = (-0.1, 0.75, 0.0)
SUBBODY = {
    "counters":     ("counters1",    (0, 0, 0),            None),
    "oven":         ("oven",         (0, 0, 0),            None),
    "backwall":     ("backwall",     (0, 0, 0),            None),
    "slidecabinet": ("slidecabinet", (0.4, 0.3, 2.6),      None),
    "hingecabinet": ("hingecabinet", (-0.504, 0.28, 2.6),  None),
    "microwave":    ("microwave",    (-0.750, -0.025, 1.6), "0 0 0.3"),
}
KETTLE_POS = (-0.269, 0.35, 1.626)   # world frame (outside the desk body)


def sv(vals):
    return " ".join(f"{v:.6g}" for v in vals)


def scale_attr(el, name, s=SCALE):
    if name in el.attrib:
        el.set(name, sv([float(v) * s for v in el.get(name).split()]))


def fix_asset_paths(el):
    f = el.get("file")
    if f is None:
        return
    if el.tag == "mesh":
        el.set("file", "../third_party/franka_kitchen/meshes/" + os.path.basename(f))
        el.set("scale", sv([SCALE] * 3))
    elif el.tag == "texture":
        el.set("file", "../third_party/franka_kitchen/textures/" + os.path.basename(f))


def scale_tree(el):
    """Recursively scale all length-like attributes in a body chain."""
    if el.tag in ("body", "geom", "site", "joint", "light", "inertial", "camera"):
        scale_attr(el, "pos")
    if el.tag in ("geom", "site"):
        scale_attr(el, "size")
        scale_attr(el, "fromto")
    if el.tag == "joint" and el.get("type") == "slide":
        scale_attr(el, "range")
    for child in el:
        scale_tree(child)


def parse(path):
    return ET.parse(path).getroot()


assets, defaults, equalities, bodies = [], [], [], []

for item in ITEMS:
    aroot = parse(os.path.join(FK, "item_assets", f"{item}_asset.xml"))
    for blk in aroot:
        if blk.tag == "asset":
            for el in blk:
                fix_asset_paths(el)
                assets.append(el)
        elif blk.tag == "default":
            defaults.extend(list(blk))
        elif blk.tag == "equality":
            equalities.extend(list(blk))     # skips empty <equality/> blocks

    croot = parse(os.path.join(FK, "item_assets", f"{item}_chain.xml"))
    chain = list(croot)                       # top-level chain elements
    for el in chain:
        scale_tree(el)

    if item == "kettle":
        # freejoint bodies must live at the top level of worldbody -> compose
        # the kitchen_root offset into the kettle's world pos directly
        wrap = ET.Element("body", {"name": "kettle_root",
                                   "pos": sv([c * SCALE + o for c, o
                                              in zip(KETTLE_POS, PLACE_POS)])})
        fj = ET.SubElement(wrap, "freejoint", {"name": "kettle"})
        for el in chain:
            if el.tag == "body":              # kettle chain root body: splice contents
                for a in ("childclass",):
                    if el.get(a):
                        wrap.set(a, el.get(a))
                for sub in el:
                    wrap.append(sub)
            else:
                wrap.append(el)
        bodies.append(("world", wrap))
    else:
        name, pos, euler = SUBBODY[item]
        wrap = ET.Element("body", {"name": name, "pos": sv([c * SCALE for c in pos])})
        if euler:
            wrap.set("euler", euler)
        for el in chain:
            wrap.append(el)
        bodies.append(("desk", wrap))

# ---- assemble the include ----
root = ET.Element("mujoco", {"model": "kitchen_fk_env"})
root.append(ET.Comment(
    " AUTO-GENERATED by scripts/tools/gen_kitchen_fk.py -- do not edit by hand.\n"
    "     Franka Kitchen environment (Apache-2.0, third_party/franka_kitchen),\n"
    f"     robot removed, uniformly scaled x{SCALE}, placed {PLACE_POS} from origin. "))

asset_el = ET.SubElement(root, "asset")
asset_el.extend(assets)
default_el = ET.SubElement(root, "default")
default_el.extend(defaults)
if equalities:
    eq_el = ET.SubElement(root, "equality")
    eq_el.extend(equalities)

wb = ET.SubElement(root, "worldbody")
kroot = ET.SubElement(wb, "body", {"name": "kitchen_root", "pos": sv(PLACE_POS)})
desk = ET.SubElement(kroot, "body", {"name": "desk",
                                     "pos": sv([c * SCALE for c in DESK_POS])})
for dest, b in bodies:
    (desk if dest == "desk" else wb).append(b)   # "world" = kettle (freejoint)

ET.indent(root, space="  ")
ET.ElementTree(root).write(OUT, encoding="unicode", xml_declaration=False)
with open(OUT, "a") as f:
    f.write("\n")
print("wrote", OUT)
