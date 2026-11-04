#!/usr/bin/env python3
"""
gen_scene_thumbs.py -- render a GUI thumbnail for every scene in scenes/scenes.json.

For each registry entry the scene XML is loaded WITH THE ROBOT REMOVED (the
`<include file="../models/mabel_full.xml"/>` line and the teleop mocap-target
bodies are stripped from a temp copy written into scenes/, so all relative
include/asset paths stay valid), rendered offscreen with mujoco.Renderer at
640x400 from an elevated 3/4 view, and saved to scenes/thumbs/<slug>.png.

Run from the repo root (headless-safe, no GUI needed):
    python3 scripts/tools/gen_scene_thumbs.py
"""
import json
import os
import re

import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SCENES_DIR = os.path.join(ROOT, "scenes")
THUMBS_DIR = os.path.join(SCENES_DIR, "thumbs")
REGISTRY = os.path.join(SCENES_DIR, "scenes.json")

WIDTH, HEIGHT = 640, 400
SETTLE_STEPS = 250          # let clutter rest on its surface before the shot

# per-scene camera overrides: (azimuth deg, elevation deg, distance factor,
# lookat offset xyz). Distance factor multiplies the auto model extent.
CAM = {
    "default":       (135, -32, 0.85, (0.0, 0.0, 0.0)),
    "home":          (-35, -33, 0.72, (0.5, 0.0, -0.3)),
    "kitchen_fk":    (65, -22, 0.34, (0.0, 1.5, 0.5)),
    "living_room":   (140, -30, 0.75, (0.0, 0.0, -0.3)),
    "bedroom":       (125, -32, 0.75, (0.0, 0.0, -0.3)),
    "office":        (145, -32, 0.75, (0.0, 0.0, -0.3)),
    "factory":       (135, -33, 0.80, (0.0, 0.0, -0.2)),
    "assembly_line": (115, -30, 0.75, (0.0, 0.5, -0.2)),
    "warehouse":     (110, -28, 0.80, (0.0, 0.0, 0.0)),
    "outdoor":       (140, -22, 0.28, (0.0, 0.0, 0.6)),
    # generated set (scripts/tools/scenegen)
    "washroom":           (135, -35, 0.80, (0.0, 0.0, -0.3)),
    "living_room_loft":   (140, -32, 0.72, (0.0, 0.0, -0.3)),
    "living_room_family": (140, -32, 0.75, (0.0, 0.0, -0.3)),
    "full_house":         (105, -48, 0.66, (0.0, 0.0, -0.9)),
    "gym":                (135, -32, 0.75, (0.0, 0.0, -0.3)),
    "supermarket":        (120, -35, 0.70, (0.0, 0.0, -0.4)),
    "cafe":               (140, -32, 0.75, (0.0, 0.0, -0.3)),
    "cafe_industrial":    (140, -32, 0.75, (0.0, 0.0, -0.3)),
    "restaurant":         (135, -32, 0.72, (0.0, 0.0, -0.3)),
    "fastfood":           (55, -32, 0.72, (0.0, 0.0, -0.3)),
    "fine_dining":        (140, -30, 0.72, (0.0, 0.0, -0.3)),
    "golf":               (15, -25, 0.42, (3.5, 0.8, 0.3)),
    "front_yard":         (90, -30, 0.52, (0.0, 0.5, -0.2)),
    "rooftop_lounge":     (55, -25, 0.16, (-1.4, -2.9, -9.9)),
    "airport":            (35, -24, 0.30, (0.0, 2.0, 0.0)),
    "intersection":       (225, -32, 0.14, (6.5, 6.5, 0.5)),
    "boulevard":          (12, -16, 0.32, (0.0, 5.0, 1.0)),
    "plaza":              (50, -36, 0.50, (1.0, 1.0, -0.3)),
    "la_suburb":          (56, -26, 0.36, (0.0, 4.0, 2.0)),
    "venice_beach":       (82, -22, 0.36, (0.0, 0.0, 2.0)),
}

STRIP_PATTERNS = [
    # the robot itself
    re.compile(r'^\s*<include\s+file="\.\./models/mabel_full\.xml"\s*/>\s*$',
               re.MULTILINE),
    # teleop mocap targets (robot-dependent, just floating dots in a thumb)
    re.compile(r'^\s*<body\s+name="(?:target_l|target_r|gaze_point)"[^>]*>.*?</body>\s*$',
               re.MULTILINE | re.DOTALL),
]


def strip_robot(xml_text):
    for pat in STRIP_PATTERNS:
        xml_text = pat.sub("", xml_text)
    return xml_text


def render_scene(entry):
    slug, xml_name = entry["name"], entry["xml"]
    with open(os.path.join(SCENES_DIR, xml_name)) as f:
        text = strip_robot(f.read())

    tmp = os.path.join(SCENES_DIR, f"_thumb_tmp_{slug}.xml")
    with open(tmp, "w") as f:
        f.write(text)
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)

    # make sure the offscreen buffer fits our render size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)

    # thumbnails read best a touch brighter than the in-sim defaults
    model.vis.headlight.ambient = np.maximum(model.vis.headlight.ambient, 0.52)
    model.vis.headlight.diffuse = np.maximum(model.vis.headlight.diffuse, 0.32)

    data = mujoco.MjData(model)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    az, el, dist_f, off = CAM.get(slug, CAM["default"])

    # dollhouse cutaway: hide the room walls on the camera side of the lookat
    # point, so the elevated 3/4 view actually shows the interior.
    fwd = np.array([np.cos(np.radians(az)), np.sin(np.radians(az))])
    lookat_xy = model.stat.center[:2] + np.asarray(off)[:2]
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if not name.startswith("wall_"):
            continue
        if np.dot(model.geom_pos[g][:2] - lookat_xy, fwd) < -0.1:
            model.geom_matid[g] = -1
            model.geom_rgba[g] = (0, 0, 0, 0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = az, el
    cam.distance = dist_f * model.stat.extent
    cam.lookat[:] = model.stat.center + np.asarray(off)

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=cam)
        px = renderer.render()

    out = os.path.join(THUMBS_DIR, f"{slug}.png")
    Image.fromarray(px).save(out)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    return out


def main():
    os.makedirs(THUMBS_DIR, exist_ok=True)
    with open(REGISTRY) as f:
        registry = json.load(f)
    for entry in registry:
        render_scene(entry)
    print("done.")


if __name__ == "__main__":
    main()
