"""sprite.py — the cockpit's robot sprite, BAKED FROM THE CANONICAL MJCF.

The frames come from tui/bake_sprites.py, which renders
simulation/mabel_mujoco/models/mabel_full.xml offscreen from an isometric
camera (8 azimuths × 4 arm stances/side × 2 lift levels) and quantizes to
the retro palette — so the terminal robot IS MABEL: binocular sensor head,
rounded torso, thin lift column, rounded base, orange wrists. Re-run the
baker after visual MJCF changes (same sync rule as URDF/GLB/rig).

This module keeps the sprite LIVE:
  * odom yaw spins the view through the 8 baked azimuths;
  * each arm shows the baked stance NEAREST to its real (a1, a2) joints —
    the classifier uses the very table the baker posed the model with;
  * lift travel picks the baked lift level;
  * a dithered circular floor disc sits under the robot (microduck-style);
  * driving shimmers the base/floor, e-stop flashes the shell red,
    torque-off hangs both arms, an idle bob breathes.

Stdlib-only (sprite_frames.py is zlib+base64 JSON).
"""
from __future__ import annotations

import math

# ── palette keys → 256-color xterm indices (8-color fallback in cockpit) ─────
PALETTE_256 = {
    "W": 255, "w": 251, "G": 245, "g": 240, "K": 237, "k": 234,
    "O": 208, "o": 130, "Y": 220, "y": 178, "R": 196,
}
PALETTE_8 = {
    "W": 7, "w": 7, "G": 7, "g": 0, "K": 0, "k": 0,
    "O": 3, "o": 3, "Y": 3, "y": 3, "R": 1,
}

try:
    from sprite_frames import DATA as _DATA
except Exception:                                    # bake missing: tiny stub
    _DATA = None

# two baked resolutions: sets[0] = small (normal terminals), sets[1] = HD
# (fullscreen). Legacy single-set bakes still load.
_SETS = (_DATA.get("sets") if _DATA else None) or (
    [{"w": _DATA["w"], "h": _DATA["h"], "frames": _DATA["frames"]}] if _DATA else [])
SPRITE_W = _SETS[0]["w"] if _SETS else 38
SPRITE_H = (_SETS[0]["h"] if _SETS else 54) + 3      # + floor rows
HD_W = _SETS[-1]["w"] if _SETS else 54
HD_H = (_SETS[-1]["h"] if _SETS else 76) + 3


def classify_arm(side: str, a1: float | None, a2: float | None) -> str:
    """Live (a1, a2) → the nearest baked stance (same table the baker used)."""
    if _DATA is None or a1 is None:
        return "down" if side == "right" else "ready"
    best, bd = "ready", 1e9
    for name, (s1, s2) in _DATA["stances"][side].items():
        dd = (a1 - s1) ** 2 + ((a2 or 0.0) - s2) ** 2
        if dd < bd:
            best, bd = name, dd
    return best


def _az_index(yaw: float) -> int:
    """World yaw → baked azimuth: a fixed virtual camera at 45° (the 3/4
    front view) watching a robot that really rotates."""
    if _DATA is None:
        return 0
    rel = (45.0 - math.degrees(yaw)) % 360.0
    return int(round(rel / 45.0)) % len(_DATA["azimuths"])


def build_frame(t: float = 0.0, *, yaw: float = 0.0,
                arm_l: tuple = (None, None), arm_r: tuple = (None, None),
                lift_m: float = 0.3, speed: float = 0.0, turning: float = 0.0,
                estop: bool = False, limp: bool = False,
                engaged: bool = False, hd: bool = False) -> list[list[str]]:
    """One sprite frame as a grid of palette keys ('.' = transparent)."""
    if _DATA is None:
        msg = "run tui/bake_sprites.py"
        grid = [["."] * SPRITE_W for _ in range(SPRITE_H)]
        for i, ch in enumerate(msg[: SPRITE_W]):
            grid[SPRITE_H // 2][i] = "W" if ch != " " else "."
        return grid

    moving = abs(speed) > 0.03 or abs(turning) > 0.05
    beat = int(t * 6) % 2
    pose_l = "down" if limp else classify_arm("left", *arm_l)
    pose_r = "down" if limp else classify_arm("right", *arm_r)
    lifts = _DATA["lifts"]
    li = min(range(len(lifts)), key=lambda i: abs(lifts[i] - lift_m))
    fset = _SETS[-1] if (hd and len(_SETS) > 1) else _SETS[0]
    key = f"{_az_index(yaw)}|{pose_l}|{pose_r}|{li}"
    src = fset["frames"].get(key) or next(iter(fset["frames"].values()))
    body = [list(r) for r in src.split("\n")]
    w, h = fset["w"], fset["h"]

    bob = 1 if (not moving and int(t * 0.8) % 2 == 1) else 0
    grid = [["."] * w for _ in range(h + 3)]

    # floor disc: a dithered dark ellipse under the base (microduck-style)
    base_bottom = max((y for y, row in enumerate(body)
                       if any(c != "." for c in row)), default=h - 1)
    cy = min(h + 1, base_bottom + bob + 2)
    cx, rx, ry = w / 2 - 0.5, w / 2 - 2.0, 3.4
    for y in range(len(grid)):
        for x in range(w):
            dx, dy = (x - cx) / rx, (y - cy) / ry
            d2 = dx * dx + dy * dy
            if d2 <= 1.0:
                grid[y][x] = "g" if (x * 7 + y * 13) % 11 == 0 else "k"
            elif d2 <= 1.35 and (x * 5 + y * 9) % 7 == 0:
                grid[y][x] = "k"                     # crumbly rim

    # blit the robot over the floor (with the idle bob)
    for y, row in enumerate(body):
        gy = y + bob
        if gy >= len(grid):
            break
        for x, chv in enumerate(row):
            if chv != ".":
                grid[gy][x] = chv

    # e-stop: flash the shell red on alternating beats
    if estop and beat:
        for row in grid:
            for x, chv in enumerate(row):
                if chv in ("W", "w", "G"):
                    row[x] = "R"

    # driving: shimmer the wheel band + kick up floor dust
    if moving:
        tick = int(t * 12)
        for y in range(max(0, base_bottom - 2 + bob), min(len(grid), base_bottom + 1 + bob)):
            for x, chv in enumerate(grid[y]):
                if chv in ("g", "G") and (x + tick) % 3 == 0:
                    grid[y][x] = "K" if chv == "g" else "w"
        for k in range(4):                            # dust puffs on the disc
            dx = int((tick * 2 + k * 9) % (rx * 2 - 2))
            gy = int(cy + (k % 2))
            if 0 <= gy < len(grid) and grid[gy][2 + dx] in ("k", "g"):
                grid[gy][2 + dx] = "G"
    return grid


def frame_size(rows: list[list[str]]) -> tuple[int, int]:
    return (len(rows), max(len(r) for r in rows))
