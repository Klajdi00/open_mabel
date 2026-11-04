#!/usr/bin/env python3
"""
gen_home_textures.py -- generate tileable PBR-ish textures for the home scene.

Writes PNGs into assets/textures/ used by scenes/mabel_home.xml:
    wood_floor.png      oak-plank floor
    wood_furniture.png  walnut wood for table / chairs / cabinets
    marble_counter.png  light marble/granite worktop
    wall.png            warm off-white plaster wall
    tile.png            kitchen backsplash tiles

No network or heavy deps -- pure numpy + PIL. Run once from the repo root:
    python scripts/tools/gen_home_textures.py
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "textures"))
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)          # fixed seed -> reproducible textures


def value_noise(h, w, octaves=5, persistence=0.5):
    """Tileable multi-octave value noise in [0,1] via upsampled random grids."""
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 1)
        g = RNG.random((n, n)).astype(np.float32)
        g = np.vstack([g, g[:1]]); g = np.hstack([g, g[:, :1]])   # wrap for tiling
        up = np.asarray(Image.fromarray((g * 255).astype(np.uint8))
                        .resize((w, h), Image.BICUBIC), np.float32) / 255.0
        acc += amp * up
        tot += amp; amp *= persistence
    acc /= tot
    return (acc - acc.min()) / (np.ptp(acc) + 1e-9)


def save(name, arr):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(OUT, name))
    print("wrote", os.path.join(OUT, name))


def wood(name, base, dark, planks=6, vertical=False, grain_freq=26.0):
    """Plank wood: along-grain rings + plank seams + colour variation."""
    H = W = 512
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (x / W, y / H) if not vertical else (y / H, x / W)
    n = value_noise(H, W, octaves=5)
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * grain_freq * u + 6.0 * n + 1.5 * np.sin(8 * v))
    grain = grain ** 2
    plank = (v * planks) % 1.0
    seam = np.clip(1.0 - np.minimum(plank, 1.0 - plank) / 0.03, 0, 1)   # dark seam lines
    shade = (0.20 + 0.80 * grain)[..., None]
    col = base[None, None, :] * shade + dark[None, None, :] * (1 - shade)
    # per-plank tint so planks differ
    pid = np.floor(v * planks).astype(int)
    tint = (0.85 + 0.3 * RNG.random(planks + 1))[pid][..., None]
    col = col * tint
    col = col * (1 - 0.7 * seam[..., None])
    col += (n[..., None] - 0.5) * 14
    save(name, col)


def marble(name):
    H = W = 512
    base = np.array([232, 230, 224], np.float32)
    vein = np.array([120, 118, 112], np.float32)
    t = value_noise(H, W, octaves=6)
    x = np.linspace(0, 1, W, dtype=np.float32)[None, :].repeat(H, 0)
    flow = np.sin(2 * np.pi * (3 * x + 1.2 * t) + 8 * t)
    v = np.clip(1.0 - np.abs(flow) / 0.25, 0, 1) ** 2      # thin dark veins
    speck = (value_noise(H, W, octaves=7) > 0.7) * 10.0
    col = base[None, None, :] * (1 - v[..., None]) + vein[None, None, :] * v[..., None]
    col += (t[..., None] - 0.5) * 10 - speck[..., None]
    save(name, col)


def wall(name):
    H = W = 512
    base = np.array([224, 216, 203], np.float32)            # warm off-white
    n = value_noise(H, W, octaves=6)
    col = base[None, None, :] + (n[..., None] - 0.5) * 18
    save(name, col)


def tile(name, cols=6):
    H = W = 512
    grout = np.array([205, 200, 192], np.float32)
    base = np.array([238, 244, 246], np.float32)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = (x / W * cols) % 1.0
    gy = (y / H * cols) % 1.0
    g = np.clip(np.minimum(gx, 1 - gx) / 0.06, 0, 1) * np.clip(np.minimum(gy, 1 - gy) / 0.06, 0, 1)
    n = value_noise(H, W, octaves=5)
    tid = (np.floor(x / W * cols) * 31 + np.floor(y / H * cols)).astype(int)
    tint = (0.92 + 0.12 * RNG.random(int(tid.max()) + 1))[tid][..., None]
    col = base[None, None, :] * tint * g[..., None] + grout[None, None, :] * (1 - g[..., None])
    col += (n[..., None] - 0.5) * 6
    save(name, col)


if __name__ == "__main__":
    wood("wood_floor.png",     np.array([178, 132, 82], np.float32),  np.array([120, 82, 45], np.float32), planks=7)
    wood("wood_furniture.png", np.array([110, 71, 45], np.float32),   np.array([66, 40, 24], np.float32),  planks=4, grain_freq=20)
    marble("marble_counter.png")
    wall("wall.png")
    tile("tile.png")
    print("done.")
