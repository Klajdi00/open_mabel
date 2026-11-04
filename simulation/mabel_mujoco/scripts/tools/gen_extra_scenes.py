#!/usr/bin/env python3
"""
gen_extra_scenes.py -- regenerate the extended MABEL scene library.

Builds every scenegen-owned scene (supermarket, restaurants, cafes, gym,
washroom, extra living rooms, full house, golf, front yard, rooftop lounge,
airport, and the dynamic city streets), their *.dynamics.json actor sidecars,
the assembly-line worker include, and rewrites scenes/scenes.json with the
categorized registry. Textures must exist first (gen_home_textures.py,
gen_env_textures.py, gen_env_textures2.py).

Run from the repo root:
    python3 scripts/tools/gen_extra_scenes.py

Then validate + refresh thumbnails:
    python3 scripts/tools/verify_scenes.py
    python3 scripts/tools/gen_scene_thumbs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenegen import generate_all   # noqa: E402

if __name__ == "__main__":
    scenes = generate_all()
    dyn = sum(1 for s in scenes if s.dynamic)
    print(f"generated {len(scenes)} scenes ({dyn} dynamic) + assembly-line retrofit")
