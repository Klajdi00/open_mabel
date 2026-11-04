"""Render sprite.py frames (baked from the MJCF) to a PNG for review."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import sprite

X256 = {255: (240, 240, 240), 251: (200, 200, 200), 245: (150, 150, 150),
        240: (90, 90, 90), 237: (50, 50, 50), 234: (30, 30, 32),
        208: (255, 135, 0), 130: (175, 95, 0), 220: (255, 215, 0),
        178: (178, 148, 0), 196: (255, 0, 0)}

variants = [
    ("idle", dict()),
    ("teleop hi", dict(arm_l=(0.0, 0.0), arm_r=(0.0, -0.9), lift_m=0.55)),
    ("yaw 90° up", dict(yaw=1.57, arm_l=(-1.0, 0), arm_r=(1.0, 0), lift_m=0.55)),
    ("yaw 180°", dict(yaw=3.14)),
    ("drive", dict(speed=0.5, t=0.35)),
    ("estop flash", dict(estop=True, t=0.17)),
]
fig, axes = plt.subplots(1, len(variants), figsize=(2.6 * len(variants), 4.6))
for ax, (name, kw) in zip(axes, variants):
    rows = sprite.build_frame(**kw)
    img = np.zeros((len(rows), sprite.SPRITE_W, 3), np.uint8)
    img[:] = (13, 13, 18)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in sprite.PALETTE_256:
                img[y, x] = X256[sprite.PALETTE_256[ch]]
    ax.imshow(img, interpolation="nearest")
    ax.set_title(name, fontsize=8)
    ax.axis("off")
out = os.path.join(os.path.dirname(__file__), "sprite_preview.png")
plt.tight_layout()
plt.savefig(out, dpi=110)
print("wrote", out)
