"""
gridmap.py — log-odds occupancy grid + inflation, the slam_env deploy
conventions (fast_nav.py): p→1 = occupied, p≈0.5 = unknown, p→0 = free.

Hyperparameters ported from the validated course deploy:
  resolution 0.05 m · L_OCC +0.85 · L_FREE −0.40 (guarded: free update only
  while log-odds < 1.5, so confirmed walls are never erased) · clamp ±5.
"""
from __future__ import annotations

import numpy as np

L_OCC = 0.85
L_FREE = -0.40
FREE_GUARD = 1.5          # don't erase confirmed walls
CLAMP = 5.0


class GridMap:
    def __init__(self, size_m=24.0, resolution=0.05, origin=(0.0, 0.0)):
        self.res = float(resolution)
        n = int(round(size_m / resolution))
        self.n = n
        self.log = np.zeros((n, n), dtype=np.float32)
        # world coords of cell (0,0) corner: centered on `origin`
        self.ox = origin[0] - size_m / 2.0
        self.oy = origin[1] - size_m / 2.0

    # ── indexing ─────────────────────────────────────────────────────────────
    def to_cell(self, x, y):
        return (int((x - self.ox) / self.res), int((y - self.oy) / self.res))

    def to_cells(self, xy):
        c = ((xy - (self.ox, self.oy)) / self.res).astype(np.int32)
        return c

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.n and 0 <= cy < self.n

    def to_world(self, cx, cy):
        return (self.ox + (cx + 0.5) * self.res, self.oy + (cy + 0.5) * self.res)

    # ── mapping update (inverse sensor model, Bresenham rays) ────────────────
    def integrate_scan(self, px, py, angles, ranges, max_range):
        """Robot at (px,py); `angles` are WORLD-frame beam angles."""
        c0 = self.to_cell(px, py)
        hits = ranges < (max_range - 0.1)
        ex = px + np.cos(angles) * np.minimum(ranges, max_range - 0.05)
        ey = py + np.sin(angles) * np.minimum(ranges, max_range - 0.05)
        log = self.log
        for i in range(len(ranges)):
            c1 = self.to_cell(ex[i], ey[i])
            for (cx, cy) in _bresenham(c0[0], c0[1], c1[0], c1[1])[:-1]:
                if self.in_bounds(cx, cy) and log[cy, cx] < FREE_GUARD:
                    log[cy, cx] += L_FREE
            if hits[i] and self.in_bounds(*c1):
                log[c1[1], c1[0]] += L_OCC
        np.clip(log, -CLAMP, CLAMP, out=log)

    # ── readouts ─────────────────────────────────────────────────────────────
    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.log))

    def score_endpoints(self, xs, ys):
        """Correlative likelihood: Σ (p−0.5) at scan endpoints (fast_nav)."""
        cx = ((xs - self.ox) / self.res).astype(np.int32)
        cy = ((ys - self.oy) / self.res).astype(np.int32)
        ok = (cx >= 0) & (cx < self.n) & (cy >= 0) & (cy < self.n)
        if not ok.any():
            return 0.0
        lo = self.log[cy[ok], cx[ok]]
        p = 1.0 / (1.0 + np.exp(-lo))
        return float(np.sum(p - 0.5))

    def inflated(self, robot_radius=0.36, clearance=0.15):
        """Two-tier binary obstacle mask (confirmed p>0.8 inflated by
        radius+clearance; probable p>0.65 by radius) — fast_nav's costmap.
        Radii sized to MABEL's true footprint (module half-diagonal 0.24 m +
        bumper): a fatter margin seals furnished rooms once SLAM drift
        double-prints the walls."""
        from scipy.ndimage import binary_dilation
        p = self.prob()
        conf = p > 0.80
        prob = p > 0.65
        r1 = max(1, int((robot_radius + clearance) / self.res))
        r2 = max(1, int(robot_radius / self.res))
        out = binary_dilation(conf, _disk(r1)) | binary_dilation(prob, _disk(r2))
        return out

    def occupancy_wire(self):
        """(w, h, res, ox, oy) + row-major u8 cells 0..100, 255=unknown —
        the nav_msgs/OccupancyGrid convention the map plane's 0x04 uses."""
        p = self.prob()
        cells = np.full(p.shape, 255, dtype=np.uint8)
        known = np.abs(p - 0.5) > 0.1
        cells[known] = np.clip(p[known] * 100.0, 0, 100).astype(np.uint8)
        return self.n, self.n, self.res, self.ox, self.oy, cells


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts
