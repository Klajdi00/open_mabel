#!/usr/bin/env python3
"""
make_visionos_icon.py — render the MABEL Teleop visionOS layered app icon.

The SAME mark as the iOS app (server/tools/make_app_icon.py, "universal"
variant): a layered liquid-glass house — vivid amber→coral canvas, a warm
colour-core house, a frosted translucent panel with a glass door, rim light
and gloss streak. visionOS icons are a 3-layer "solid image stack" that the
system parallaxes inside a glass orb, so the iOS composite is split by depth:

  Back   — amber→coral gradient + ambient glow (opaque)
  Middle — drop shadow + the vivid colour-core house (transparent)
  Front  — frosted glass panel, gloss, rim light, door, streak (transparent)

Each layer renders 1024×1024 @2x at 3× supersampling, Lanczos-downsized.

    python3 tools/make_visionos_icon.py
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
STACK = os.path.normpath(os.path.join(
    HERE, "..", "MabelTeleop", "Assets.xcassets", "AppIcon.solidimagestack"))
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
    return [(cx, cy - h / 2 - roof),
            (cx + hw, cy - h / 2 + roof * 0.30),
            (cx + hw, cy + h / 2),
            (cx - hw, cy + h / 2),
            (cx - hw, cy - h / 2 + roof * 0.30)]


def layer(fill_fn):
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    fill_fn(ImageDraw.Draw(img), img)
    return img


# Shared geometry (matches the iOS universal icon).
CX, CY = s(512), s(528)
W, H, ROOF = s(470), s(300), s(176)
RCORNER = s(46)
PTS = house_pts(CX, CY, W, H, ROOF)
IP = [((x - CX) * 0.86 + CX, (y - CY) * 0.86 + CY) for x, y in PTS]   # inset panel


def back_layer():
    """Amber→coral canvas + ambient glow (opaque)."""
    bg = vgrad(C(0xFFA632), C(0xF55B3C))
    bg.alpha_composite(radial_glow((s(360), s(300)), s(540), C(0xFFEFC4), 0.40))
    return bg.convert("RGB")


def middle_layer():
    """Drop shadow + the vivid colour core of the house (transparent)."""
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))

    sh = layer(lambda d, im: d.polygon(
        _round_pts([(x, y + s(20)) for x, y in PTS], RCORNER), fill=(40, 18, 8, 150)))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(s(30))))

    core = vgrad(C(0xFFFBF2), C(0xFFE3B4))
    cmask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(cmask).polygon(_round_pts(PTS, RCORNER), fill=255)
    core.putalpha(cmask)
    img.alpha_composite(core)
    return img


def front_layer():
    """Frosted glass panel + gloss + rim light + glass door + streak."""
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))

    # Frosted glass panel, inset so the core shows as a rim.
    glass = layer(lambda d, im: d.polygon(_round_pts(IP, s(38)), fill=C(0xFFFFFF, 150)))
    img.alpha_composite(glass)

    # Vertical gloss: brighter at the top, fading down (masked to the panel).
    grad = np.linspace(70, 0, N)[:, None]
    g = np.zeros((N, N, 4), np.float32); g[..., :3] = 255
    g[..., 3] = np.repeat(grad, N, axis=1)
    gmask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(gmask).polygon(_round_pts(IP, s(38)), fill=255)
    gloss = Image.fromarray(g.astype("uint8"), "RGBA")
    gloss.putalpha(Image.composite(gloss.getchannel("A"), Image.new("L", (N, N), 0), gmask))
    img.alpha_composite(gloss)

    d = ImageDraw.Draw(img)
    # Bright rim light along the upper roof edges (the lit glass edge).
    apex = IP[0]
    d.line([(IP[4][0], IP[4][1]), apex, (IP[1][0], IP[1][1])],
           fill=C(0xFFFFFF, 235), width=s(8), joint="curve")

    # Glass door — a simple rounded slot, unmistakably "home", no face.
    dw, dh = s(116), s(152)
    dx0, dy0 = CX - dw / 2, CY + H / 2 - dh - s(4)
    d.rounded_rectangle([dx0, dy0, dx0 + dw, dy0 + dh], radius=s(40), fill=C(0xF26B3A, 235))
    d.line([(dx0 + s(40), dy0 + s(3)), (dx0 + dw - s(40), dy0 + s(3))],
           fill=C(0xFFFFFF, 235), width=s(5))

    # One clean diagonal gloss streak across the upper panel.
    streak = layer(lambda dr, im: dr.line(
        [(CX - s(150), CY - s(120)), (CX + s(60), CY - s(180))],
        fill=(255, 255, 255, 110), width=s(26)))
    streak = streak.filter(ImageFilter.GaussianBlur(s(14)))
    streak.putalpha(Image.composite(streak.getchannel("A"), Image.new("L", (N, N), 0), gmask))
    img.alpha_composite(streak)
    return img


def write_layer(name, img):
    folder = os.path.join(STACK, f"{name}.solidimagestacklayer", "Content.imageset")
    fname = f"{name.lower()}.png"
    img.resize((1024, 1024), Image.LANCZOS).save(os.path.join(folder, fname))
    with open(os.path.join(folder, "Contents.json"), "w") as f:
        json.dump({
            "images": [{"idiom": "vision", "scale": "2x", "filename": fname}],
            "info": {"author": "xcode", "version": 1},
        }, f, indent=2)
    print(f"wrote {folder}/{fname}")


if __name__ == "__main__":
    write_layer("Back", back_layer())
    write_layer("Middle", middle_layer())
    write_layer("Front", front_layer())
