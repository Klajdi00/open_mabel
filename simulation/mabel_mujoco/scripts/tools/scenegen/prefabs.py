"""prefabs.py -- reusable multi-geom pieces (architecture, furniture, props).

Every prefab appends named geoms to a Scene at (x, y) with an optional yaw.
Positions are prefab-local offsets rotated into the world, so any piece can be
placed at any angle. Collidable unless stated; purely decorative bits are
contype/conaffinity 0 to keep the contact broadphase lean in the big scenes.
"""
import math

from .core import rot


def _p(x, y, dx, dy, yaw):
    ox, oy = rot(dx, dy, yaw)
    return x + ox, y + oy


# ── architecture ──────────────────────────────────────────────────────────────
def floor(s, w, d, mat, friction="1.5 0.05 0.001"):
    s.geom("floor", "plane", (0, 0, 0), (w / 2, d / 2, 0.1), mat=mat,
           extra=f'contype="1" conaffinity="1" friction="{friction}"')


def room_shell(s, w, d, h, floor_mat, wall_mat, wainscot=None):
    """Rectangular room: floor plane + 4 perimeter walls (wall_* for cutaway)."""
    floor(s, w, d, floor_mat)
    hw, hd, hh = w / 2, d / 2, h / 2
    t = 0.05
    s.geom("wall_n", "box", (0, hd + t, hh), (hw + 2 * t, t, hh), mat=wall_mat)
    s.geom("wall_s", "box", (0, -hd - t, hh), (hw + 2 * t, t, hh), mat=wall_mat)
    s.geom("wall_e", "box", (hw + t, 0, hh), (t, hd, hh), mat=wall_mat)
    s.geom("wall_w", "box", (-hw - t, 0, hh), (t, hd, hh), mat=wall_mat)
    if wainscot:
        for nm, p, sz in (("wall_wain_n", (0, hd - 0.001, 0.55), (hw, 0.012, 0.55)),
                          ("wall_wain_s", (0, -hd + 0.001, 0.55), (hw, 0.012, 0.55))):
            s.geom(nm, "box", p, sz, mat=wainscot, collide=False)


def partition(s, name, x0, y0, x1, y1, h=2.3, mat="m_wall", t=0.05):
    """Interior wall segment between two floor points."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = math.hypot(x1 - x0, y1 - y0)
    yaw = math.atan2(y1 - y0, x1 - x0)
    s.geom(name, "box", (cx, cy, h / 2), (L / 2, t, h / 2), euler=(0, 0, yaw),
           mat=mat)


def window_band(s, name, x, y, yaw, L, z0=0.9, z1=2.2, mat="m_storefront"):
    """Glass band set into a wall face (visual, sits proud of the wall)."""
    s.geom(name, "box", (x, y, (z0 + z1) / 2), (L / 2, 0.02, (z1 - z0) / 2),
           euler=(0, 0, yaw), mat=mat, collide=False)


def ceiling_lamp(s, tag, x, y, z=2.5, r=0.16):
    s.geom(f"{tag}_stem", "cylinder", (x, y, z + 0.14), (0.012, 0.14), mat="m_dark", collide=False)
    s.geom(f"{tag}_shade", "cylinder", (x, y, z), (r, 0.05), mat="m_dark", collide=False)
    s.geom(f"{tag}_bulb", "cylinder", (x, y, z - 0.052), (r * 0.8, 0.008), mat="m_lamp", collide=False)


# ── tables & seating ─────────────────────────────────────────────────────────
def rect_table(s, tag, x, y, yaw=0.0, w=1.2, d=0.7, h=0.74, top="m_wood",
               legs="m_dark", cloth=False):
    if cloth:
        s.geom(f"{tag}_cloth", "box", (x, y, h - 0.10), (w / 2 + 0.03, d / 2 + 0.03, 0.10),
               euler=(0, 0, yaw), mat=top)
        s.geom(f"{tag}_top", "box", (x, y, h + 0.012), (w / 2 + 0.01, d / 2 + 0.01, 0.012),
               euler=(0, 0, yaw), mat=top)
        return
    s.geom(f"{tag}_top", "box", (x, y, h), (w / 2, d / 2, 0.025), euler=(0, 0, yaw), mat=top)
    for i, (dx, dy) in enumerate(((w / 2 - 0.06, d / 2 - 0.06), (-w / 2 + 0.06, d / 2 - 0.06),
                                  (w / 2 - 0.06, -d / 2 + 0.06), (-w / 2 + 0.06, -d / 2 + 0.06))):
        px, py = _p(x, y, dx, dy, yaw)
        s.geom(f"{tag}_l{i + 1}", "box", (px, py, (h - 0.025) / 2),
               (0.035, 0.035, (h - 0.025) / 2), mat=legs)


def round_table(s, tag, x, y, r=0.40, h=0.74, top="m_wood", base="m_dark",
                cloth=None):
    if cloth:
        s.geom(f"{tag}_cloth", "cylinder", (x, y, h - 0.18), (r + 0.02, 0.19), mat=cloth)
    s.geom(f"{tag}_top", "cylinder", (x, y, h), (r, 0.022), mat=top)
    s.geom(f"{tag}_post", "cylinder", (x, y, h / 2), (0.04, h / 2 - 0.03), mat=base)
    s.geom(f"{tag}_foot", "cylinder", (x, y, 0.02), (r * 0.6, 0.02), mat=base)


def chair(s, tag, x, y, yaw=0.0, seat="m_wood", h=0.46, back=True):
    """Chair FACING +local-x (rotate with yaw to face the table)."""
    s.geom(f"{tag}_seat", "box", (x, y, h), (0.21, 0.21, 0.025), euler=(0, 0, yaw), mat=seat)
    for i, (dx, dy) in enumerate(((0.17, 0.17), (-0.17, 0.17), (0.17, -0.17), (-0.17, -0.17))):
        px, py = _p(x, y, dx, dy, yaw)
        s.geom(f"{tag}_l{i + 1}", "box", (px, py, h / 2 - 0.0125), (0.02, 0.02, h / 2 - 0.0125),
               mat="m_dark")
    if back:
        bx, by = _p(x, y, -0.19, 0, yaw)
        s.geom(f"{tag}_back", "box", (bx, by, h + 0.30), (0.022, 0.20, 0.28),
               euler=(0, 0, yaw), mat=seat)


def stool(s, tag, x, y, h=0.63, seat="m_frame"):
    s.geom(f"{tag}_post", "cylinder", (x, y, h / 2 - 0.015), (0.03, h / 2 - 0.015), mat="m_dark")
    s.geom(f"{tag}_seat", "cylinder", (x, y, h), (0.17, 0.03), mat=seat)


def bench(s, tag, x, y, yaw=0.0, L=1.6, seat="m_wood", back=True, h=0.45):
    s.geom(f"{tag}_seat", "box", (x, y, h), (L / 2, 0.24, 0.03), euler=(0, 0, yaw), mat=seat)
    for i, dx in enumerate((-L / 2 + 0.12, L / 2 - 0.12)):
        px, py = _p(x, y, dx, 0, yaw)
        s.geom(f"{tag}_leg{i + 1}", "box", (px, py, h / 2), (0.03, 0.20, h / 2), mat="m_dark")
    if back:
        bx, by = _p(x, y, 0, -0.22, yaw)
        s.geom(f"{tag}_back", "box", (bx, by, h + 0.28), (L / 2, 0.025, 0.26),
               euler=(0, 0, yaw), mat=seat)


def booth(s, tag, x, y, yaw=0.0, color="0.72 0.18 0.16 1"):
    """Fast-food booth: two facing benches + table between."""
    for i, off in enumerate((-0.85, 0.85)):
        bx, by = _p(x, y, 0, off, yaw)
        s.geom(f"{tag}_seat{i + 1}", "box", (bx, by, 0.44), (0.62, 0.26, 0.06),
               euler=(0, 0, yaw), rgba=color)
        s.geom(f"{tag}_base{i + 1}", "box", (bx, by, 0.19), (0.62, 0.24, 0.19),
               euler=(0, 0, yaw), mat="m_dark")
        kx, ky = _p(x, y, 0, off + (0.24 if off > 0 else -0.24), yaw)
        s.geom(f"{tag}_back{i + 1}", "box", (kx, ky, 0.75), (0.62, 0.05, 0.38),
               euler=(0, 0, yaw), rgba=color)
    s.geom(f"{tag}_top", "box", (x, y, 0.74), (0.55, 0.38, 0.025), euler=(0, 0, yaw), mat="m_white")
    s.geom(f"{tag}_post", "box", (x, y, 0.36), (0.05, 0.05, 0.36), euler=(0, 0, yaw), mat="m_chrome")


def sofa(s, tag, x, y, yaw=0.0, L=2.0, mat="m_fabric"):
    """Sofa FACING +local-x."""
    s.geom(f"{tag}_base", "box", (x, y, 0.24), (0.42, L / 2, 0.24), euler=(0, 0, yaw), mat=mat)
    bx, by = _p(x, y, -0.34, 0, yaw)
    s.geom(f"{tag}_back", "box", (bx, by, 0.62), (0.10, L / 2, 0.20), euler=(0, 0, yaw), mat=mat)
    for i, off in enumerate((-L / 2 + 0.09, L / 2 - 0.09)):
        ax, ay = _p(x, y, 0, off, yaw)
        s.geom(f"{tag}_arm{i + 1}", "box", (ax, ay, 0.52), (0.42, 0.09, 0.10),
               euler=(0, 0, yaw), mat=mat)
    cx, cy = _p(x, y, 0.05, 0, yaw)
    s.geom(f"{tag}_cush", "box", (cx, cy, 0.50), (0.36, L / 2 - 0.2, 0.045),
           euler=(0, 0, yaw), mat=mat, collide=False)


def tv_wall(s, tag, x, y, yaw=0.0, w=1.5):
    """Media console + wall TV, screen FACING +local-x."""
    s.geom(f"{tag}_console", "box", (x, y, 0.25), (0.22, w / 2 + 0.15, 0.25),
           euler=(0, 0, yaw), mat="m_wood")
    fx, fy = _p(x, y, 0.02, 0, yaw)
    s.geom(f"{tag}_frame", "box", (fx, fy, 1.35), (0.03, w / 2, 0.44), euler=(0, 0, yaw), mat="m_dark")
    sx, sy = _p(x, y, 0.055, 0, yaw)
    s.geom(f"{tag}_screen", "box", (sx, sy, 1.35), (0.004, w / 2 - 0.04, 0.40),
           euler=(0, 0, yaw), mat="m_screen", collide=False)


def bookshelf(s, tag, x, y, yaw=0.0, w=1.0, h=1.9):
    s.geom(f"{tag}_case", "box", (x, y, h / 2), (0.16, w / 2, h / 2), euler=(0, 0, yaw), mat="m_wood")
    for i in range(4):
        z = 0.35 + i * 0.42
        bx, by = _p(x, y, 0.13, 0, yaw)
        s.geom(f"{tag}_books{i + 1}", "box", (bx, by, z), (0.035, w / 2 - 0.06, 0.15),
               euler=(0, 0, yaw), mat="m_book", collide=False)


def rug(s, tag, x, y, w, d, mat="m_rug", yaw=0.0):
    s.geom(f"{tag}", "box", (x, y, 0.006), (w / 2, d / 2, 0.006), euler=(0, 0, yaw),
           mat=mat, collide=False)


def plant(s, tag, x, y, h=1.1):
    s.geom(f"{tag}_pot", "cylinder", (x, y, 0.16), (0.14, 0.16), mat="m_dark")
    s.geom(f"{tag}_bush", "sphere", (x, y, h), (0.30,), mat="m_foliage", collide=False)
    s.geom(f"{tag}_stem", "cylinder", (x, y, (h + 0.32) / 2 - 0.05), (0.02, (h - 0.32) / 2), mat="m_bark", collide=False)


def counter(s, tag, x, y, yaw=0.0, L=2.4, top="m_marble", body="m_wood", h=0.92):
    s.geom(f"{tag}_body", "box", (x, y, (h - 0.04) / 2), (L / 2, 0.32, (h - 0.04) / 2),
           euler=(0, 0, yaw), mat=body)
    s.geom(f"{tag}_top", "box", (x, y, h), (L / 2 + 0.03, 0.36, 0.02), euler=(0, 0, yaw), mat=top)


def shelf_unit(s, tag, x, y, yaw=0.0, L=1.8, h=1.8, mat="m_metal"):
    """Open shelving rack (warehouse / gym / back-of-house)."""
    for i, dx in enumerate((-L / 2, L / 2)):
        for j, dy in enumerate((-0.30, 0.30)):
            px, py = _p(x, y, dx, dy, yaw)
            s.geom(f"{tag}_p{i}{j}", "box", (px, py, h / 2), (0.025, 0.025, h / 2), mat=mat)
    for k in range(4):
        z = 0.15 + k * (h - 0.3) / 3
        s.geom(f"{tag}_s{k + 1}", "box", (x, y, z), (L / 2, 0.32, 0.02), euler=(0, 0, yaw), mat=mat)


def gondola(s, tag, x, y, yaw=0.0, L=4.0, h=1.7):
    """Supermarket aisle shelf, product faces on both sides."""
    s.geom(f"{tag}_core", "box", (x, y, h / 2), (L / 2, 0.28, h / 2), euler=(0, 0, yaw), mat="m_steel")
    s.geom(f"{tag}_base", "box", (x, y, 0.10), (L / 2 + 0.02, 0.36, 0.10), euler=(0, 0, yaw), mat="m_dark")
    for i, off in enumerate((-0.295, 0.295)):
        px, py = _p(x, y, 0, off, yaw)
        s.geom(f"{tag}_face{i + 1}", "box", (px, py, 0.95), (L / 2 - 0.02, 0.012, 0.72),
               euler=(0, 0, yaw), mat="m_products", collide=False)
    s.geom(f"{tag}_topper", "box", (x, y, h + 0.08), (L / 2, 0.20, 0.08), euler=(0, 0, yaw), mat="m_steel")


def fridge_wall(s, tag, x, y, yaw=0.0, L=4.0):
    """Glass-door cooler bank against a wall, doors FACING +local-y."""
    s.geom(f"{tag}_body", "box", (x, y, 1.05), (L / 2, 0.38, 1.05), euler=(0, 0, yaw), mat="m_steel")
    px, py = _p(x, y, 0, 0.395, yaw)
    s.geom(f"{tag}_glass", "box", (px, py, 1.10), (L / 2 - 0.05, 0.012, 0.85),
           euler=(0, 0, yaw), mat="m_storefront", collide=False)


def checkout(s, tag, x, y, yaw=0.0):
    s.geom(f"{tag}_desk", "box", (x, y, 0.46), (1.1, 0.42, 0.46), euler=(0, 0, yaw), mat="m_white")
    bx, by = _p(x, y, 0.3, 0, yaw)
    s.geom(f"{tag}_belt", "box", (bx, by, 0.93), (0.7, 0.30, 0.015), euler=(0, 0, yaw), mat="m_belt")
    px, py = _p(x, y, -0.8, 0.1, yaw)
    s.geom(f"{tag}_pole", "cylinder", (px, py, 1.25), (0.03, 0.35), mat="m_steel")
    s.geom(f"{tag}_sign", "box", (px, py, 1.72), (0.12, 0.02, 0.12), euler=(0, 0, yaw), mat="m_sig_amber", collide=False)


def bed(s, tag, x, y, yaw=0.0, w=1.6, L=2.0):
    s.geom(f"{tag}_frame", "box", (x, y, 0.22), (L / 2, w / 2, 0.10), euler=(0, 0, yaw), mat="m_wood")
    s.geom(f"{tag}_mattress", "box", (x, y, 0.42), (L / 2 - 0.03, w / 2 - 0.03, 0.11),
           euler=(0, 0, yaw), mat="m_bedding")
    hx, hy = _p(x, y, -L / 2 + 0.03, 0, yaw)
    s.geom(f"{tag}_head", "box", (hx, hy, 0.62), (0.04, w / 2, 0.34), euler=(0, 0, yaw), mat="m_wood")
    for i, off in enumerate((-w / 4, w / 4)):
        px, py = _p(x, y, -L / 2 + 0.35, off, yaw)
        s.geom(f"{tag}_pillow{i + 1}", "box", (px, py, 0.57), (0.20, 0.28, 0.06),
               euler=(0, 0, yaw), mat="m_white", collide=False)


def wardrobe(s, tag, x, y, yaw=0.0, w=1.2, h=2.0):
    s.geom(f"{tag}_body", "box", (x, y, h / 2), (0.30, w / 2, h / 2), euler=(0, 0, yaw), mat="m_wood")
    for i, off in enumerate((-w / 4, w / 4)):
        px, py = _p(x, y, 0.305, off, yaw)
        s.geom(f"{tag}_knob{i + 1}", "sphere", (px, py, h / 2), (0.02,), mat="m_chrome", collide=False)


def toilet(s, tag, x, y, yaw=0.0):
    s.geom(f"{tag}_base", "box", (x, y, 0.20), (0.19, 0.15, 0.20), euler=(0, 0, yaw), mat="m_white")
    bx, by = _p(x, y, 0.10, 0, yaw)
    s.geom(f"{tag}_bowl", "cylinder", (bx, by, 0.40), (0.19, 0.035), mat="m_white")
    tx, ty = _p(x, y, -0.22, 0, yaw)
    s.geom(f"{tag}_tank", "box", (tx, ty, 0.55), (0.09, 0.21, 0.16), euler=(0, 0, yaw), mat="m_white")


def vanity(s, tag, x, y, yaw=0.0, L=1.4, mirror=True):
    """Bathroom vanity, basin up, mirror on the -local-x wall side."""
    s.geom(f"{tag}_body", "box", (x, y, 0.42), (0.28, L / 2, 0.42), euler=(0, 0, yaw), mat="m_wood")
    s.geom(f"{tag}_top", "box", (x, y, 0.86), (0.30, L / 2 + 0.02, 0.02), euler=(0, 0, yaw), mat="m_marble")
    for i, off in enumerate([-L / 4, L / 4] if L > 1.0 else [0]):
        px, py = _p(x, y, 0.02, off, yaw)
        s.geom(f"{tag}_basin{i + 1}", "cylinder", (px, py, 0.895), (0.16, 0.018), mat="m_white")
        fx, fy = _p(x, y, -0.18, off, yaw)
        s.geom(f"{tag}_tap{i + 1}", "box", (fx, fy, 0.95), (0.02, 0.02, 0.07), euler=(0, 0, yaw), mat="m_chrome", collide=False)
    if mirror:
        mx, my = _p(x, y, -0.27, 0, yaw)
        s.geom(f"{tag}_mirror", "box", (mx, my, 1.55), (0.012, L / 2 - 0.08, 0.45),
               euler=(0, 0, yaw), mat="m_mirror", collide=False)


def bathtub(s, tag, x, y, yaw=0.0):
    s.geom(f"{tag}_shell", "box", (x, y, 0.28), (0.85, 0.40, 0.28), euler=(0, 0, yaw), mat="m_white")
    s.geom(f"{tag}_water", "box", (x, y, 0.472), (0.72, 0.28, 0.02), euler=(0, 0, yaw), mat="m_water", collide=False)
    fx, fy = _p(x, y, 0.70, 0, yaw)
    s.geom(f"{tag}_tap", "box", (fx, fy, 0.62), (0.03, 0.03, 0.06), euler=(0, 0, yaw), mat="m_chrome", collide=False)


def shower(s, tag, x, y, yaw=0.0):
    s.geom(f"{tag}_tray", "box", (x, y, 0.03), (0.50, 0.50, 0.03), euler=(0, 0, yaw), mat="m_white")
    for i, (dx, dy) in enumerate(((0.50, 0), (0, 0.50))):
        px, py = _p(x, y, dx, dy, yaw)
        s.geom(f"{tag}_glass{i + 1}", "box", (px, py, 1.05),
               (0.012, 0.50, 1.0) if dx else (0.50, 0.012, 1.0),
               euler=(0, 0, yaw), mat="m_glass")
    hx, hy = _p(x, y, -0.42, -0.42, yaw)
    s.geom(f"{tag}_head", "sphere", (hx, hy, 2.0), (0.05,), mat="m_chrome", collide=False)


# ── outdoor ──────────────────────────────────────────────────────────────────
def tree(s, tag, x, y, h=1.5, r=0.9):
    s.geom(f"{tag}_trunk", "cylinder", (x, y, h / 2), (0.09 + 0.03 * r, h / 2), mat="m_bark")
    s.geom(f"{tag}_can1", "sphere", (x, y, h + r * 0.8), (r,), mat="m_foliage", collide=False)
    s.geom(f"{tag}_can2", "sphere", (x + r * 0.45, y + r * 0.35, h + r * 0.45), (r * 0.6,), mat="m_foliage", collide=False)
    s.geom(f"{tag}_can3", "sphere", (x - r * 0.4, y - r * 0.3, h + r * 0.5), (r * 0.55,), mat="m_foliage", collide=False)


def hedge(s, tag, x, y, yaw=0.0, L=2.0, h=0.5):
    s.geom(f"{tag}", "box", (x, y, h / 2 + 0.02), (L / 2, 0.25, h / 2), euler=(0, 0, yaw), mat="m_foliage")


def palm_tree(s, tag, x, y, h=4.5, lean=0.0):
    """A palm: tall slightly-leaning trunk + a crown of drooping fronds."""
    tx = x + math.sin(lean) * h * 0.5
    s.geom(f"{tag}_trunk", "capsule", (x, y, 0.1), (0.13,),
           fromto=(x, y, 0.0, tx, y, h), mat="m_bark")
    for i in range(7):
        a = i * 2 * math.pi / 7
        dx, dy = math.cos(a), math.sin(a)
        s.geom(f"{tag}_frond{i}", "capsule", (tx, y, h), (0.05,),
               fromto=(tx, y, h, tx + dx * 1.5, y + dy * 1.5, h - 0.55),
               mat="m_foliage", collide=False)
    s.geom(f"{tag}_top", "sphere", (tx, y, h + 0.05), (0.22,), mat="m_foliage", collide=False)


def plum_tree(s, tag, x, y, h=2.2):
    """Ornamental plum: dark trunk + a rounded pink-purple canopy."""
    s.geom(f"{tag}_trunk", "cylinder", (x, y, h / 2), (0.09, h / 2), mat="m_bark")
    for i, (dx, dy, dz, r) in enumerate(((0, 0, h + 0.5, 1.0), (0.5, 0.3, h + 0.2, 0.7),
                                         (-0.4, 0.4, h + 0.25, 0.6), (0.1, -0.5, h + 0.3, 0.65))):
        s.geom(f"{tag}_can{i}", "sphere", (x + dx, y + dy, dz), (r,),
               rgba="0.72 0.42 0.62 1", collide=False)


def house(s, tag, x, y, yaw=0.0, w=8.0, d=7.0, wall="m_cream", roof="m_dark",
          garage=True):
    """Single-family suburban house: walls, pitched roof, door, windows, garage."""
    hw, hd, h = w / 2, d / 2, 2.6
    s.geom(f"{tag}_walls", "box", (x, y, h / 2), (hw, hd, h / 2), euler=(0, 0, yaw), mat=wall)
    # pitched roof (two slabs)
    for sgn in (-1, 1):
        rx, ry = rot(0, sgn * hd * 0.5, yaw)
        s.geom(f"{tag}_roof{sgn}", "box", (x + rx, y + ry, h + 0.55),
               (hw + 0.3, hd * 0.62, 0.08), euler=(sgn * 0.5, 0, yaw), mat=roof, collide=False)
    s.geom(f"{tag}_ridge", "box", (x, y, h + 1.0), (hw + 0.3, 0.08, 0.08), euler=(0, 0, yaw), mat=roof, collide=False)
    # door + windows on the +local-x face (front)
    fx, fy = rot(hw + 0.02, 0, yaw)
    s.geom(f"{tag}_door", "box", (x + fx, y + fy, 1.0), (0.04, 0.45, 1.0), euler=(0, 0, yaw), mat="m_wood", collide=False)
    for off in (-hd * 0.5, hd * 0.5):
        wx, wy = rot(hw + 0.02, off, yaw)
        s.geom(f"{tag}_win{int(off*10)}", "box", (x + wx, y + wy, 1.5), (0.03, 0.45, 0.45),
               euler=(0, 0, yaw), mat="m_storefront", collide=False)
    if garage:
        gx, gy = rot(hw + 0.02, -hd + 1.0, yaw)
        s.geom(f"{tag}_garage", "box", (x + gx, y + gy, 0.9), (0.05, 1.0, 0.9), euler=(0, 0, yaw), mat="m_dark", collide=False)


def streetlight(s, tag, x, y, yaw=0.0, h=4.6):
    s.geom(f"{tag}_pole", "cylinder", (x, y, h / 2), (0.06, h / 2), mat="m_dark")
    ax, ay = _p(x, y, 0.6, 0, yaw)
    s.geom(f"{tag}_arm", "box", ((x + ax) / 2, (y + ay) / 2, h - 0.05), (0.6, 0.04, 0.04),
           euler=(0, 0, yaw), mat="m_dark", collide=False)
    s.geom(f"{tag}_lum", "box", (ax, ay, h - 0.12), (0.22, 0.09, 0.05), euler=(0, 0, yaw),
           mat="m_lamp", collide=False)


def traffic_light(s, tag, x, y, yaw=0.0):
    s.geom(f"{tag}_pole", "cylinder", (x, y, 1.9), (0.05, 1.9), mat="m_dark")
    hx, hy = _p(x, y, 0.12, 0, yaw)
    s.geom(f"{tag}_head", "box", (hx, hy, 3.45), (0.10, 0.14, 0.42), euler=(0, 0, yaw), mat="m_dark")
    for i, (mat, dz) in enumerate((("m_sig_red", 0.26), ("m_sig_amber", 0.0), ("m_sig_green", -0.26))):
        lx, ly = _p(x, y, 0.23, 0, yaw)
        s.geom(f"{tag}_lamp{i + 1}", "cylinder", (lx, ly, 3.45 + dz), (0.07, 0.012),
               euler=(0, 1.570795, yaw), mat=mat, collide=False)


def hydrant(s, tag, x, y):
    s.geom(f"{tag}_body", "cylinder", (x, y, 0.32), (0.10, 0.32), rgba="0.78 0.16 0.12 1")
    s.geom(f"{tag}_cap", "sphere", (x, y, 0.66), (0.09,), rgba="0.78 0.16 0.12 1", collide=False)
    s.geom(f"{tag}_arm", "box", (x, y, 0.48), (0.16, 0.05, 0.045), rgba="0.78 0.16 0.12 1", collide=False)


def trash_can(s, tag, x, y, mat="m_dark"):
    s.geom(f"{tag}", "cylinder", (x, y, 0.42), (0.26, 0.42), mat=mat)
    s.geom(f"{tag}_rim", "cylinder", (x, y, 0.86), (0.27, 0.02), mat="m_steel", collide=False)


def parasol(s, tag, x, y, mat="m_awning_red"):
    s.geom(f"{tag}_pole", "cylinder", (x, y, 1.15), (0.03, 1.15), mat="m_steel")
    s.geom(f"{tag}_top", "cylinder", (x, y, 2.32), (1.15, 0.035), mat=mat, collide=False)
    s.geom(f"{tag}_tip", "cylinder", (x, y, 2.42), (0.02, 0.07), mat="m_steel", collide=False)


def awning(s, tag, x, y, yaw=0.0, L=3.0, mat="m_awning_red", z=2.6):
    """Sloped canopy off a facade, hanging over +local-y."""
    s.geom(f"{tag}", "box", (x, y, z), (L / 2, 0.65, 0.02), euler=(0.28, 0, yaw), mat=mat, collide=False)
    vx, vy = _p(x, y, 0, 0.62, yaw)
    s.geom(f"{tag}_valance", "box", (vx, vy, z - 0.26),
           (L / 2, 0.015, 0.09), euler=(0, 0, yaw), mat=mat, collide=False)


def building(s, tag, x, y, yaw=0.0, w=10.0, d=8.0, h=16.0, shopfront=True,
             mat="m_facade", collide=True):
    """City block building: facade-textured box + ground-floor storefront band."""
    s.geom(f"{tag}_mass", "box", (x, y, h / 2 + 3.2), (w / 2, d / 2, h / 2 - 1.6),
           euler=(0, 0, yaw), mat=mat, collide=collide)
    s.geom(f"{tag}_ground", "box", (x, y, 1.8), (w / 2 + 0.06, d / 2 + 0.06, 1.8),
           euler=(0, 0, yaw), mat="m_storefront" if shopfront else "m_concrete",
           collide=collide)
    s.geom(f"{tag}_cornice", "box", (x, y, h + 1.72), (w / 2 + 0.12, d / 2 + 0.12, 0.12),
           euler=(0, 0, yaw), mat="m_dark", collide=False)


# ── realistic buildings + entrances ───────────────────────────────────────────
def _face(front):
    """(outward normal, along-face tangent, yaw) for a front direction (fx,fy)."""
    fx, fy = front
    n = math.hypot(fx, fy) or 1.0
    fx, fy = fx / n, fy / n
    return (fx, fy), (-fy, fx), math.atan2(fy, fx)


def door(s, tag, x, y, front=(0, -1), w=2.2, h=2.6, canopy=True, glass="m_glass"):
    """A recessed lobby entrance on a building's front face, centred at ground
    point (x, y). The recess is a walk-in dark alcove so a person can 'enter'."""
    (nx, ny), (tx, ty), yaw = _face(front)
    rx, ry = x - nx * 0.28, y - ny * 0.28
    s.geom(f"{tag}_recess", "box", (rx, ry, h / 2), (0.28, w / 2, h / 2),
           euler=(0, 0, yaw), mat="m_dark", collide=False)
    gx, gy = x + nx * 0.02, y + ny * 0.02
    s.geom(f"{tag}_glass", "box", (gx, gy, h * 0.43), (0.02, w / 2 - 0.1, h * 0.40),
           euler=(0, 0, yaw), mat=glass, collide=False)
    for i, sgn in enumerate((-1, 1)):
        jx, jy = x + tx * (w / 2) * sgn, y + ty * (w / 2) * sgn
        s.geom(f"{tag}_jamb{i}", "box", (jx, jy, h / 2), (0.09, 0.09, h / 2),
               euler=(0, 0, yaw), mat="m_frame", collide=False)
    s.geom(f"{tag}_lintel", "box", (x, y, h + 0.06), (0.12, w / 2 + 0.12, 0.1),
           euler=(0, 0, yaw), mat="m_frame", collide=False)
    if canopy:
        cx, cy = x + nx * 0.75, y + ny * 0.75
        s.geom(f"{tag}_canopy", "box", (cx, cy, h + 0.22), (0.75, w / 2 + 0.2, 0.04),
               euler=(0, 0, yaw), mat="m_dark", collide=False)


def fire_escape(s, tag, x, y, front=(0, -1), w=3.0, floors=5, base_z=3.4, fh=3.2):
    """Zig-zag steel fire escape on a building's front face."""
    (nx, ny), (tx, ty), yaw = _face(front)
    ox, oy = x + nx * 0.55, y + ny * 0.55          # stands proud of the facade
    for f in range(floors):
        z = base_z + f * fh
        s.geom(f"{tag}_land{f}", "box", (ox, oy, z), (0.55, w / 2, 0.03),
               euler=(0, 0, yaw), mat="m_dark", collide=False)
        # railing
        rx, ry = ox + nx * 0.5, oy + ny * 0.5
        s.geom(f"{tag}_rail{f}", "box", (rx, ry, z + 0.4), (0.03, w / 2, 0.4),
               euler=(0, 0, yaw), mat="m_dark", collide=False)
        # diagonal ladder to the next landing, alternating sides
        side = 1 if f % 2 == 0 else -1
        lx = ox + tx * (w / 2 - 0.3) * side
        ly = oy + ty * (w / 2 - 0.3) * side
        s.geom(f"{tag}_ladder{f}", "box", (lx, ly, z + fh / 2),
               (0.4, 0.05, fh / 2), euler=(0.5 * side, 0, yaw), mat="m_dark", collide=False)


def glass_tower(s, tag, x, y, front=(0, -1), w=9.0, d=9.0, h=40.0,
                setbacks=1, collide=True):
    """Modern curtain-wall office/condo tower, optional upper setback + lobby."""
    (nx, ny), _, yaw = _face(front)
    tiers = max(1, setbacks + 1)
    cw, cd, z0 = w, d, 0.0
    for t in range(tiers):
        th = h / tiers
        s.geom(f"{tag}_t{t}", "box", (x, y, z0 + th / 2), (cw / 2, cd / 2, th / 2),
               euler=(0, 0, yaw), mat="m_curtain", collide=collide and t == 0)
        z0 += th; cw *= 0.82; cd *= 0.82
    # glass lobby at the base + entrance
    s.geom(f"{tag}_lobby", "box", (x, y, 2.1), (w / 2 + 0.05, d / 2 + 0.05, 2.1),
           euler=(0, 0, yaw), mat="m_storefront", collide=collide)
    door(s, f"{tag}_door", x + nx * (d / 2 + 0.06), y + ny * (d / 2 + 0.06),
         front=front, w=2.6, h=3.2)
    # rooftop mechanical + parapet
    s.geom(f"{tag}_mech", "box", (x, y, h + 0.6), (cw / 1.2, cd / 1.2, 0.6),
           euler=(0, 0, yaw), mat="m_dark", collide=False)


def brick_walkup(s, tag, x, y, front=(0, -1), w=8.0, d=9.0, h=18.0,
                 mat="m_brownstone", collide=True):
    """Brownstone / brick row building: stoop + door + zig-zag fire escape."""
    (nx, ny), (tx, ty), yaw = _face(front)
    s.geom(f"{tag}_mass", "box", (x, y, h / 2 + 0.9), (w / 2, d / 2, h / 2 - 0.45),
           euler=(0, 0, yaw), mat=mat, collide=collide)
    s.geom(f"{tag}_cornice", "box", (x, y, h + 0.85), (w / 2 + 0.15, d / 2 + 0.15, 0.18),
           euler=(0, 0, yaw), mat="m_dark", collide=False)
    # stoop + raised door
    sx, sy = x + nx * (d / 2 + 0.5), y + ny * (d / 2 + 0.5)
    for i in range(4):
        s.geom(f"{tag}_step{i}", "box", (sx - nx * i * 0.16, sy - ny * i * 0.16, 0.1 + i * 0.16),
               (0.5 - i * 0.14, 0.9, 0.08), euler=(0, 0, yaw), mat="m_concrete")
    door(s, f"{tag}_door", x + nx * (d / 2 + 0.04), y + ny * (d / 2 + 0.04),
         front=front, w=1.5, h=2.4, canopy=False)
    fire_escape(s, f"{tag}_fe", x - tx * w * 0.22, y - ty * w * 0.22, front=front,
                w=w * 0.5, floors=max(2, int(h / 3.4)))


def art_deco(s, tag, x, y, front=(0, -1), w=14.0, d=14.0, h=70.0, collide=True):
    """Art-deco setback skyscraper (Empire State style): stepped tiers, stone
    piers, a crown and a spire."""
    (nx, ny), _, yaw = _face(front)
    tiers = 4
    cw, cd, z0 = w, d, 0.0
    for t in range(tiers):
        th = h * (0.5 if t == 0 else 0.5 / (tiers - 1))
        s.geom(f"{tag}_t{t}", "box", (x, y, z0 + th / 2), (cw / 2, cd / 2, th / 2),
               euler=(0, 0, yaw), mat="m_artdeco", collide=collide and t == 0)
        s.geom(f"{tag}_set{t}", "box", (x, y, z0 + th - 0.05), (cw / 2 + 0.2, cd / 2 + 0.2, 0.2),
               euler=(0, 0, yaw), mat="m_dark", collide=False)
        z0 += th; cw *= 0.72; cd *= 0.72
    # crown + spire
    s.geom(f"{tag}_crown", "box", (x, y, z0 + 1.0), (cw / 2, cd / 2, 1.0),
           euler=(0, 0, yaw), mat="m_steel", collide=False)
    s.geom(f"{tag}_spire", "cylinder", (x, y, z0 + 4.0), (0.4, 3.0), mat="m_steel", collide=False)
    s.geom(f"{tag}_mast", "cylinder", (x, y, z0 + 8.0), (0.08, 1.5), mat="m_steel", collide=False)
    # grand entrance
    door(s, f"{tag}_door", x + nx * (d / 2 + 0.06), y + ny * (d / 2 + 0.06),
         front=front, w=3.2, h=4.2)


def condo_tower(s, tag, x, y, front=(0, -1), w=10.0, d=10.0, h=44.0, collide=True):
    """Residential tower with protruding balcony slabs on the front face."""
    (nx, ny), (tx, ty), yaw = _face(front)
    s.geom(f"{tag}_mass", "box", (x, y, h / 2 + 2.0), (w / 2, d / 2, h / 2 - 1.0),
           euler=(0, 0, yaw), mat="m_facade", collide=collide)
    s.geom(f"{tag}_base", "box", (x, y, 2.0), (w / 2 + 0.05, d / 2 + 0.05, 2.0),
           euler=(0, 0, yaw), mat="m_storefront", collide=collide)
    for f in range(3, int(h / 3.2)):
        z = f * 3.2
        for sgn in (-1, 1):
            bx = x + nx * (d / 2 + 0.35) + tx * (w / 4) * sgn
            by = y + ny * (d / 2 + 0.35) + ty * (w / 4) * sgn
            s.geom(f"{tag}_bal{f}{sgn}", "box", (bx, by, z), (0.35, w / 5, 0.05),
                   euler=(0, 0, yaw), mat="m_concrete", collide=False)
            s.geom(f"{tag}_balr{f}{sgn}", "box", (bx + nx * 0.33, by + ny * 0.33, z + 0.4),
                   (0.02, w / 5, 0.4), euler=(0, 0, yaw), mat="m_glass", collide=False)
    door(s, f"{tag}_door", x + nx * (d / 2 + 0.06), y + ny * (d / 2 + 0.06),
         front=front, w=2.6, h=3.0)


def fence_picket(s, tag, x, y, yaw=0.0, L=3.0):
    s.geom(f"{tag}_rail1", "box", (x, y, 0.35), (L / 2, 0.02, 0.03), euler=(0, 0, yaw), mat="m_white")
    s.geom(f"{tag}_rail2", "box", (x, y, 0.80), (L / 2, 0.02, 0.03), euler=(0, 0, yaw), mat="m_white")
    n = max(2, int(L / 0.3))
    for i in range(n + 1):
        dx = -L / 2 + i * L / n
        px, py = _p(x, y, dx, 0, yaw)
        s.geom(f"{tag}_pk{i}", "box", (px, py, 0.55), (0.035, 0.012, 0.45), euler=(0, 0, yaw),
               mat="m_white", collide=False)


# ── free (graspable) props ───────────────────────────────────────────────────
def free_box(s, tag, x, y, z, size, rgba, mass=0.3, yaw=0.0):
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}">
      <freejoint name="{tag}_joint"/>
      <geom name="{tag}_g" type="box" size="{' '.join(f'{v:.3f}' for v in size)}" rgba="{rgba}"
            condim="4" friction="1 0.05 0.001" mass="{mass}" priority="1"
            solref="0.004 1" solimp="0.99 0.999 0.001"/>
    </body>""")


def free_sphere(s, tag, x, y, z, r, rgba, mass=0.15):
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint name="{tag}_joint"/>
      <geom name="{tag}_g" type="sphere" size="{r:.3f}" rgba="{rgba}"
            condim="4" friction="0.9 0.02 0.001" mass="{mass}" priority="1"
            solref="0.004 1" solimp="0.99 0.999 0.001"/>
    </body>""")


def free_cylinder(s, tag, x, y, z, r, h, rgba, mass=0.25, yaw=0.0):
    s.raw(f"""    <body name="{tag}" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 {yaw:.4f}">
      <freejoint name="{tag}_joint"/>
      <geom name="{tag}_g" type="cylinder" size="{r:.3f} {h:.3f}" rgba="{rgba}"
            condim="4" friction="1 0.05 0.001" mass="{mass}" priority="1"
            solref="0.004 1" solimp="0.99 0.999 0.001"/>
    </body>""")
