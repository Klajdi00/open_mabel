"""
planner.py — A* global planner on the inflated occupancy grid (the Nav2
NavFn analog; structure ported from slam_env fast_nav.py::a_star_plan).

8-connected A*, Euclidean heuristic, start/goal clearance bubbles so plans
can always enter/exit near inflated walls, path subsampled to 0.2 m spacing.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

GOAL_RADIUS_CELLS = 4
BUBBLE = 6                  # cleared cells around start/goal
SUBSAMPLE_M = 0.20


def astar(grid, occ, start_xy, goal_xy):
    """grid: GridMap · occ: bool inflated mask · returns [(x,y), ...] or None."""
    n = grid.n
    s = grid.to_cell(*start_xy)
    g = grid.to_cell(*goal_xy)
    if not (grid.in_bounds(*s) and grid.in_bounds(*g)):
        return None
    occ = occ.copy()
    for (cx, cy) in (s, g):
        x0, x1 = max(0, cx - BUBBLE), min(n, cx + BUBBLE + 1)
        y0, y1 = max(0, cy - BUBBLE), min(n, cy + BUBBLE + 1)
        occ[y0:y1, x0:x1] = False

    h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
    openq = [(h(s), 0.0, s, None)]
    came = {}
    gscore = {s: 0.0}
    goal_cell = None
    NB = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
          (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]
    while openq:
        _f, gc, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if h(cur) <= GOAL_RADIUS_CELLS:
            goal_cell = cur
            break
        for dx, dy, w in NB:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or occ[nxt[1], nxt[0]]:
                continue
            ng = gc + w
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                heapq.heappush(openq, (ng + h(nxt), ng, nxt, cur))
    if goal_cell is None:
        return None
    cells = []
    c = goal_cell
    while c is not None:
        cells.append(c)
        c = came[c]
    cells.reverse()
    pts = [grid.to_world(cx, cy) for cx, cy in cells]
    pts.append(tuple(goal_xy))
    return _subsample(pts, SUBSAMPLE_M)


def _subsample(pts, spacing):
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= spacing:
            out.append(p)
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def path_blocked(grid, occ, path, ahead=8):
    """True when a newly inflated cell sits on the next `ahead` waypoints."""
    for (x, y) in path[:ahead]:
        c = grid.to_cell(x, y)
        if grid.in_bounds(*c) and occ[c[1], c[0]]:
            return True
    return False
