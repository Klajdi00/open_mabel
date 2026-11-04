#!/usr/bin/env python3
"""
make_app_icon.py — render the MABEL app icon.

A bright, friendly "home" mark built from simple geometry in a layered
liquid-glass style: a vivid colour core, a frosted translucent house panel on
top (so the core glows through as a rim), a glass door, and specular highlights
— no face, no eyes. Rendered at 3x supersampling, Lanczos-downsized for crisp,
Apple-grade edges. Emits the three iOS appearances + a transparent mark.

    python3 make_app_icon.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "MABEL Mobile App", "MABEL Mobile App", "Assets.xcassets", "AppIcon.appiconset"))
SS = 3
N = 1024 * SS


def C(h, a=255):
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255, a)


def s(x):
    return int(round(x * SS))


def vgrad(top, bot):
    t = np.linspace(0, 1, N)[:, None]
    arr = np.array(top[:3]) * (1 - t) + np.array(bot[:3]) * t
    arr = np.repeat(arr[:, None, :], N, axis=1)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB").convert("RGBA")


def radial_glow(center, radius, color, strength=0.5):
    """A soft radial light blob (RGBA) for glassy depth."""
    yy, xx = np.mgrid[0:N, 0:N]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    k = np.clip(1 - d, 0, 1) ** 2 * strength
    img = np.zeros((N, N, 4), dtype=np.float32)
    img[..., 0] = color[0]; img[..., 1] = color[1]; img[..., 2] = color[2]
    img[..., 3] = k * 255
    return Image.fromarray(img.astype("uint8"), "RGBA")


def _round_pts(pts, r, steps=24):
    out = []
    n = len(pts)
    P = [np.array(p, float) for p in pts]
    for i in range(n):
        v, p, nx = P[i], P[(i - 1) % n], P[(i + 1) % n]
        d1, d2 = p - v, nx - v
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        rr = min(r, l1 / 2, l2 / 2)
        t1, t2 = v + d1 / l1 * rr, v + d2 / l2 * rr
        for j in range(steps + 1):
            u = j / steps
            b = (1 - u) ** 2 * t1 + 2 * (1 - u) * u * v + u ** 2 * t2
            out.append((float(b[0]), float(b[1])))
    return out


def house_pts(cx, cy, w, h, roof):
    """Rounded-gable house centred at (cx,cy): width w, body height h, roof rise."""
    hw = w / 2
    return [(cx, cy - h / 2 - roof),            # apex
            (cx + hw, cy - h / 2 + roof * 0.30),  # right shoulder
            (cx + hw, cy + h / 2),                # right bottom
            (cx - hw, cy + h / 2),                # left bottom
            (cx - hw, cy - h / 2 + roof * 0.30)]  # left shoulder


def layer(fill_fn):
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    fill_fn(ImageDraw.Draw(img), img)
    return img


def draw_icon(bg, *, core_top, core_bot, glass, door, rim, glow=None, shadow=True):
    cx, cy = s(512), s(528)
    w, h, roof = s(470), s(300), s(176)
    rcorner = s(46)
    pts = house_pts(cx, cy, w, h, roof)

    # Soft ambient glow for depth.
    if glow is not None:
        bg.alpha_composite(glow)

    # Drop shadow.
    if shadow:
        sh = layer(lambda d, im: d.polygon(_round_pts([(x, y + s(20)) for x, y in pts], rcorner),
                                           fill=(40, 18, 8, 150)))
        bg.alpha_composite(sh.filter(ImageFilter.GaussianBlur(s(30))))

    # 1) Vivid colour core (the glass beneath), full house.
    core = vgrad(core_top, core_bot)
    cmask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(cmask).polygon(_round_pts(pts, rcorner), fill=255)
    core.putalpha(cmask)
    bg.alpha_composite(core)

    # 2) Frosted glass panel on top — slightly inset so the core shows as a rim.
    ip = [( (x - cx) * 0.86 + cx, (y - cy) * 0.86 + cy ) for x, y in pts]
    glass_layer = layer(lambda d, im: d.polygon(_round_pts(ip, s(38)), fill=glass))
    # Vertical gloss: brighter at the top, fading down.
    grad = np.linspace(70, 0, N)[:, None]
    g = np.zeros((N, N, 4), np.float32); g[..., :3] = 255
    g[..., 3] = np.repeat(grad, N, axis=1)
    gmask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(gmask).polygon(_round_pts(ip, s(38)), fill=255)
    gloss = Image.fromarray(g.astype("uint8"), "RGBA"); gloss.putalpha(
        Image.composite(gloss.getchannel("A"), Image.new("L", (N, N), 0), gmask))
    bg.alpha_composite(glass_layer)
    bg.alpha_composite(gloss)

    d = ImageDraw.Draw(bg)
    # Bright rim light along the upper roof edges (the lit glass edge).
    apex = ip[0]
    d.line([(ip[4][0], ip[4][1]), apex, (ip[1][0], ip[1][1])],
           fill=rim, width=s(8), joint="curve")

    # 4) Glass door — a simple rounded slot, unmistakably "home", no face.
    dw, dh = s(116), s(152)
    dx0, dy0 = cx - dw / 2, cy + h / 2 - dh - s(4)
    d.rounded_rectangle([dx0, dy0, dx0 + dw, dy0 + dh], radius=s(40), fill=door)
    # a crisp top-edge highlight on the door (lit glass)
    d.line([(dx0 + s(40), dy0 + s(3)), (dx0 + dw - s(40), dy0 + s(3))],
           fill=rim, width=s(5))

    # 5) One clean diagonal gloss streak across the upper panel (not a blob).
    streak = layer(lambda dr, im: dr.line(
        [(cx - s(150), cy - s(120)), (cx + s(60), cy - s(180))],
        fill=(255, 255, 255, 110), width=s(26)))
    streak = streak.filter(ImageFilter.GaussianBlur(s(14)))
    streak.putalpha(Image.composite(streak.getchannel("A"), Image.new("L", (N, N), 0), gmask))
    bg.alpha_composite(streak)


def finish(img):
    return img.convert("RGB").resize((1024, 1024), Image.LANCZOS)


def universal():
    bg = vgrad(C(0xFFA632), C(0xF55B3C))      # vivid amber → warm coral
    draw_icon(bg, core_top=C(0xFFFBF2), core_bot=C(0xFFE3B4),
              glass=C(0xFFFFFF, 150), door=C(0xF26B3A, 235), rim=C(0xFFFFFF, 235),
              glow=radial_glow((s(360), s(300)), s(540), C(0xFFEFC4), 0.40))
    return finish(bg)


def dark():
    bg = vgrad(C(0x3A2A1E), C(0x140D08))
    draw_icon(bg, core_top=C(0xFFC074), core_bot=C(0xE87A3C),
              glass=C(0xF4E9DC, 70), door=C(0x1C120B, 200), rim=C(0xFFE3BC, 200),
              glow=radial_glow((s(512), s(380)), s(520), C(0xC9763C), 0.4))
    return finish(bg)


def tinted():
    bg = vgrad(C(0x141414), C(0x000000))
    draw_icon(bg, core_top=C(0xCFCFCF), core_bot=C(0x8C8C8C),
              glass=C(0xFFFFFF, 80), door=C(0x0A0A0A, 210), rim=C(0xFFFFFF, 200))
    return finish(bg)


def mark():
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    draw_icon(img, core_top=C(0xFFF0D2), core_bot=C(0xFFC074),
              glass=C(0xFFFFFF, 120), door=C(0xF07A45, 200), rim=C(0xFFFFFF, 220),
              shadow=False)
    return img.resize((1024, 1024), Image.LANCZOS)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    universal().save(os.path.join(OUT, "AppIcon-1024.png"))
    dark().save(os.path.join(OUT, "AppIcon-1024-dark.png"))
    tinted().save(os.path.join(OUT, "AppIcon-1024-tinted.png"))
    mark().save(os.path.join(OUT, "..", "mabel-logo.png"))
    print("wrote AppIcon-1024[/-dark/-tinted].png and mabel-logo.png to", OUT)
