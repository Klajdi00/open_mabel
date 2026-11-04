#!/usr/bin/env python3
"""
gen_env_textures.py -- generate tileable textures for the extended scene library
(living room, bedroom, office, factory, assembly line, warehouse, outdoor).

Companion to gen_home_textures.py (which makes the kitchen/home set). Writes PNGs
into assets/textures/ referenced by scenes/mabel_<env>.xml:

    concrete.png       sealed concrete slab (factory / warehouse floor)
    metal_brushed.png  brushed sheet metal (machines, conveyor frames)
    conveyor.png       dark rubber belt with transverse cleats
    hazard.png         yellow/black diagonal safety stripes
    cardboard.png      kraft cardboard with tape band (warehouse boxes)
    grass.png          lawn grass (outdoor ground)
    foliage.png        dense leaf canopy (tree crowns)
    bark.png           tree bark, vertical ridges
    gravel.png         gravel/stone path
    carpet_gray.png    low-pile office carpet tiles
    carpet_beige.png   warm bedroom carpet
    fabric_blue.png    woven upholstery (sofa)
    bedding.png        quilted duvet with stitch lines
    rug.png            bordered living-room rug
    brick.png          red running-bond brick wall
    book_spines.png    colored book spines (bookshelf filler)
    whiteboard.png     whiteboard with faint marker scribbles
    monitor.png        desktop screen glow (office monitors / TV)

No network or heavy deps -- pure numpy + PIL, fixed seed. Run from repo root:
    python3 scripts/tools/gen_env_textures.py
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "textures"))
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(11)          # fixed seed -> reproducible textures


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


def concrete():
    H = W = 512
    base = np.array([146, 144, 140], np.float32)
    n = value_noise(H, W, octaves=6)
    stains = value_noise(H, W, octaves=3)
    col = base[None, None, :] + (n[..., None] - 0.5) * 26 - (stains[..., None] ** 2) * 34
    # fine speckle (aggregate)
    speck = (RNG.random((H, W)) > 0.985)[..., None] * 40.0
    col -= speck
    # saw-cut expansion joints every half tile
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    for v in (x / W, y / H):
        d = np.minimum((v * 2) % 1.0, 1 - (v * 2) % 1.0)
        col -= np.clip(1 - d / 0.006, 0, 1)[..., None] * 55
    save("concrete.png", col)


def metal_brushed():
    H = W = 512
    base = np.array([176, 180, 186], np.float32)
    streak = value_noise(H, 8, octaves=4)
    streak = np.asarray(Image.fromarray((streak * 255).astype(np.uint8))
                        .resize((W, H), Image.BILINEAR), np.float32) / 255.0
    n = value_noise(H, W, octaves=6)
    col = base[None, None, :] + (streak[..., None] - 0.5) * 30 + (n[..., None] - 0.5) * 8
    save("metal_brushed.png", col)


def conveyor():
    H = W = 512
    base = np.array([52, 54, 58], np.float32)
    y = np.mgrid[0:H, 0:W][0].astype(np.float32) / H
    cleat = (y * 12) % 1.0
    ridge = np.clip(1 - np.abs(cleat - 0.5) / 0.08, 0, 1)
    n = value_noise(H, W, octaves=5)
    col = base[None, None, :] + ridge[..., None] * 26 + (n[..., None] - 0.5) * 10
    save("conveyor.png", col)


def hazard():
    H = W = 512
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    s = ((x + y) / W * 4) % 1.0
    yellow = np.array([235, 195, 30], np.float32)
    black = np.array([28, 28, 30], np.float32)
    m = (s < 0.5)[..., None].astype(np.float32)
    n = value_noise(H, W, octaves=5)
    col = yellow[None, None, :] * m + black[None, None, :] * (1 - m)
    col += (n[..., None] - 0.5) * 14        # grime
    save("hazard.png", col)


def cardboard():
    H = W = 512
    base = np.array([187, 152, 108], np.float32)
    n = value_noise(H, W, octaves=6)
    fiber = value_noise(4, W, octaves=3)
    fiber = np.asarray(Image.fromarray((fiber * 255).astype(np.uint8))
                       .resize((W, H), Image.BILINEAR), np.float32) / 255.0
    col = base[None, None, :] + (n[..., None] - 0.5) * 18 + (fiber[..., None] - 0.5) * 10
    # packing-tape band across the middle
    y = np.mgrid[0:H, 0:W][0].astype(np.float32) / H
    tape = np.clip(1 - np.abs(y - 0.5) / 0.06, 0, 1)[..., None]
    col = col * (1 - 0.35 * tape) + np.array([205, 199, 185], np.float32)[None, None, :] * 0.35 * tape
    save("cardboard.png", col)


def grass():
    H = W = 512
    base = np.array([92, 138, 62], np.float32)
    dark = np.array([56, 96, 40], np.float32)
    n = value_noise(H, W, octaves=7)
    blades = RNG.random((H, W)).astype(np.float32)
    m = (0.35 + 0.65 * n)[..., None]
    col = base[None, None, :] * m + dark[None, None, :] * (1 - m)
    col += (blades[..., None] - 0.5) * 26
    save("grass.png", col)


def foliage():
    H = W = 512
    base = np.array([64, 110, 48], np.float32)
    dark = np.array([30, 62, 26], np.float32)
    n = value_noise(H, W, octaves=6)
    clump = value_noise(H, W, octaves=3)
    m = (0.3 + 0.7 * (0.6 * n + 0.4 * clump))[..., None]
    col = base[None, None, :] * m + dark[None, None, :] * (1 - m)
    col += (RNG.random((H, W, 1)) - 0.5) * 20
    save("foliage.png", col)


def bark():
    H = W = 512
    base = np.array([110, 86, 62], np.float32)
    dark = np.array([62, 46, 34], np.float32)
    x = np.mgrid[0:H, 0:W][1].astype(np.float32) / W
    n = value_noise(H, W, octaves=6)
    ridges = 0.5 + 0.5 * np.sin(2 * np.pi * (18 * x + 2.5 * n))
    m = (0.25 + 0.75 * ridges)[..., None]
    col = base[None, None, :] * m + dark[None, None, :] * (1 - m)
    col += (n[..., None] - 0.5) * 16
    save("bark.png", col)


def gravel():
    H = W = 512
    base = np.array([150, 143, 132], np.float32)
    n = value_noise(H, W, octaves=7)
    stones = value_noise(H, W, octaves=5)
    edges = np.abs(np.gradient(stones)[0]) + np.abs(np.gradient(stones)[1])
    col = base[None, None, :] + (n[..., None] - 0.5) * 34 - np.clip(edges[..., None] * 900, 0, 60)
    save("gravel.png", col)


def carpet(name, base):
    H = W = 512
    n = value_noise(H, W, octaves=7)
    pile = RNG.random((H, W)).astype(np.float32)
    col = base[None, None, :] + (n[..., None] - 0.5) * 12 + (pile[..., None] - 0.5) * 10
    # carpet-tile seams (2x2 per tile)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    for v in (x / W, y / H):
        d = np.minimum((v * 2) % 1.0, 1 - (v * 2) % 1.0)
        col -= np.clip(1 - d / 0.004, 0, 1)[..., None] * 10
    save(name, col)


def fabric_blue():
    H = W = 512
    base = np.array([70, 92, 122], np.float32)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    weave = (0.5 + 0.5 * np.sin(2 * np.pi * x / 4)) * (0.5 + 0.5 * np.sin(2 * np.pi * y / 4))
    n = value_noise(H, W, octaves=5)
    col = base[None, None, :] + (weave[..., None] - 0.5) * 16 + (n[..., None] - 0.5) * 14
    save("fabric_blue.png", col)


def bedding():
    H = W = 512
    base = np.array([226, 224, 218], np.float32)
    n = value_noise(H, W, octaves=5)
    col = base[None, None, :] + (n[..., None] - 0.5) * 10
    # quilt stitch grid (4x4) with soft puff shading
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    gx, gy = (x / W * 4) % 1.0, (y / H * 4) % 1.0
    puff = np.sin(np.pi * gx) * np.sin(np.pi * gy)
    col += (puff[..., None] - 0.5) * 14
    for v in (gx, gy):
        d = np.minimum(v, 1 - v)
        col -= np.clip(1 - d / 0.01, 0, 1)[..., None] * 18
    save("bedding.png", col)


def rug():
    H = W = 512
    base = np.array([128, 62, 58], np.float32)     # deep red field
    border = np.array([196, 168, 120], np.float32)  # cream border
    n = value_noise(H, W, octaves=6)
    col = base[None, None, :] + (n[..., None] - 0.5) * 16
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    bx = np.minimum(x / W, 1 - x / W)
    by = np.minimum(y / H, 1 - y / H)
    b = np.minimum(bx, by)
    band = ((b > 0.04) & (b < 0.10)).astype(np.float32)[..., None]
    col = col * (1 - band) + border[None, None, :] * band
    # simple diamond medallion
    cx, cy = np.abs(x / W - 0.5), np.abs(y / H - 0.5)
    diamond = ((cx + cy) < 0.18).astype(np.float32)[..., None]
    col = col * (1 - 0.5 * diamond) + border[None, None, :] * 0.5 * diamond
    save("rug.png", col)


def brick():
    H = W = 512
    mortar = np.array([190, 184, 174], np.float32)
    base = np.array([158, 82, 62], np.float32)
    rows = 8
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (y / H * rows)
    row = np.floor(v).astype(int)
    u = (x / W * rows / 2 + (row % 2) * 0.5) % 1.0
    fy = v % 1.0
    g = (np.clip(np.minimum(u, 1 - u) / 0.04, 0, 1) *
         np.clip(np.minimum(fy, 1 - fy) / 0.08, 0, 1))
    n = value_noise(H, W, octaves=6)
    tid = (row * 31 + np.floor(x / W * rows / 2)).astype(int)
    tint = (0.82 + 0.36 * RNG.random(int(tid.max()) + 1))[tid][..., None]
    col = base[None, None, :] * tint * g[..., None] + mortar[None, None, :] * (1 - g[..., None])
    col += (n[..., None] - 0.5) * 16
    save("brick.png", col)


def book_spines():
    H = W = 512
    col = np.zeros((H, W, 3), np.float32)
    palette = np.array([[152, 60, 54], [58, 94, 130], [92, 118, 74], [180, 150, 92],
                        [88, 70, 110], [160, 110, 60], [70, 70, 74], [130, 82, 96]], np.float32)
    x = 0
    while x < W:
        w = int(RNG.integers(14, 34))
        c = palette[int(RNG.integers(0, len(palette)))] * (0.85 + 0.3 * RNG.random())
        h_top = int(H * (0.04 + 0.05 * RNG.random()))          # varying book heights
        col[h_top:, x:x + w] = c
        col[h_top:, x:x + 2] *= 0.55                            # gap shadow between books
        # spine label band
        yb = int(H * (0.25 + 0.4 * RNG.random()))
        col[yb:yb + 14, x + 3:x + w - 3] = np.array([214, 205, 188], np.float32)
        x += w
    n = value_noise(H, W, octaves=5)
    col += (n[..., None] - 0.5) * 10
    col[:int(H * 0.04)] = np.array([44, 34, 26], np.float32)    # shelf shadow at top
    save("book_spines.png", col)


def whiteboard():
    H = W = 512
    base = np.array([243, 245, 246], np.float32)
    n = value_noise(H, W, octaves=4)
    col = base[None, None, :] + (n[..., None] - 0.5) * 5
    # faint marker scribbles: a few random polylines
    for c, ink in [((190, 60, 50), 3), ((50, 80, 160), 4), ((40, 40, 40), 4)]:
        px, py = RNG.integers(40, W - 40), RNG.integers(60, H - 60)
        for _ in range(60):
            dx, dy = RNG.integers(-14, 15), RNG.integers(-8, 9)
            nx, ny = int(np.clip(px + dx, 4, W - 5)), int(np.clip(py + dy, 4, H - 5))
            steps = max(abs(nx - px), abs(ny - py), 1)
            for t in np.linspace(0, 1, steps * 2):
                xi, yi = int(px + (nx - px) * t), int(py + (ny - py) * t)
                col[yi - 1:yi + 1, xi - 1:xi + 1] = np.array(c, np.float32)
            px, py = nx, ny
    save("whiteboard.png", col)


def monitor():
    H, W = 320, 512
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    # dark blue desktop gradient
    col = np.zeros((H, W, 3), np.float32)
    col[..., 0] = 18 + 30 * (y / H)
    col[..., 1] = 32 + 52 * (y / H)
    col[..., 2] = 70 + 90 * (y / H)
    # fake windows
    for (x0, y0, w, h) in [(40, 40, 210, 170), (280, 90, 190, 150)]:
        col[y0:y0 + 18, x0:x0 + w] = np.array([228, 230, 234], np.float32)     # title bar
        col[y0 + 18:y0 + h, x0:x0 + w] = np.array([248, 249, 250], np.float32)  # body
        for li in range(4):                                                      # text lines
            ly = y0 + 34 + li * 22
            col[ly:ly + 6, x0 + 12:x0 + w - 30] = np.array([150, 154, 160], np.float32)
    # taskbar
    col[H - 22:, :] = np.array([24, 26, 32], np.float32)
    save("monitor.png", col)


if __name__ == "__main__":
    concrete()
    metal_brushed()
    conveyor()
    hazard()
    cardboard()
    grass()
    foliage()
    bark()
    gravel()
    carpet("carpet_gray.png", np.array([124, 130, 138], np.float32))
    carpet("carpet_beige.png", np.array([196, 178, 152], np.float32))
    fabric_blue()
    bedding()
    rug()
    brick()
    book_spines()
    whiteboard()
    monitor()
    print("done.")
