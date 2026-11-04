"""
navigator.py — goal lifecycle for the Mac nav fallback (Nav2's behavior-tree
role, minimally): A* plan → MPPI follow → reached / failed, with replanning
on blocked paths and stuck recovery (slam_env-style motion watchdog).
"""
from __future__ import annotations

import math
import time

from . import planner
from .mppi import MPPI

GOAL_TOL = 0.25          # m
REPLAN_PERIOD = 2.0      # s — also replan when the path gets blocked
STUCK_WINDOW = 4.0       # s with < STUCK_DIST progress → replan; twice → fail
STUCK_DIST = 0.08
GOAL_TIMEOUT = 120.0


class Navigator:
    def __init__(self, seed=0):
        self.mppi = MPPI(seed=seed)
        self.state = "idle"          # idle|following|reached|failed
        self.goal = None
        self.path = []
        self.status_msg = ""
        self._t_goal = 0.0
        self._t_plan = 0.0
        self._stuck_ref = None
        self._stuck_t = 0.0
        self._stuck_strikes = 0

    def set_goal(self, x, y, yaw=None):
        self.goal = (float(x), float(y))
        self.state = "following"
        self.status_msg = "planning"
        self.path = []
        self._t_goal = time.time()
        self._t_plan = 0.0
        self._stuck_ref = None
        self._stuck_strikes = 0
        self.mppi.reset()

    def cancel(self):
        self.state = "idle"
        self.goal = None
        self.path = []
        self.status_msg = "cancelled"
        self.mppi.reset()

    def active(self):
        return self.state == "following"

    # ── one control tick (~10-15 Hz) ─────────────────────────────────────────
    def tick(self, pose, twist, grid, occ):
        """Returns (vx, vy, w) body twist to command (zeros unless following)."""
        if self.state != "following" or self.goal is None:
            return (0.0, 0.0, 0.0)
        now = time.time()
        px, py, _ = pose
        gd = math.hypot(self.goal[0] - px, self.goal[1] - py)
        if gd < GOAL_TOL:
            self.state = "reached"
            self.status_msg = f"reached ({gd:.2f} m)"
            self.path = []
            return (0.0, 0.0, 0.0)
        if now - self._t_goal > GOAL_TIMEOUT:
            self.state = "failed"
            self.status_msg = "timeout"
            return (0.0, 0.0, 0.0)

        # (re)plan: periodic, on empty path, or when newly blocked
        if (not self.path or now - self._t_plan > REPLAN_PERIOD
                or planner.path_blocked(grid, occ, self.path)):
            self._t_plan = now
            path = planner.astar(grid, occ, (px, py), self.goal)
            if path is None:
                self.state = "failed"
                self.status_msg = "no path (goal unreachable on current map)"
                return (0.0, 0.0, 0.0)
            self.path = path
        # prune passed waypoints
        while (len(self.path) > 1 and
               math.hypot(self.path[0][0] - px, self.path[0][1] - py) < 0.25):
            self.path.pop(0)

        # stuck watchdog
        if self._stuck_ref is None or now - self._stuck_t > STUCK_WINDOW:
            if (self._stuck_ref is not None and
                    math.hypot(px - self._stuck_ref[0],
                               py - self._stuck_ref[1]) < STUCK_DIST):
                self._stuck_strikes += 1
                self.path = []                     # force replan
                if self._stuck_strikes >= 3:
                    self.state = "failed"
                    self.status_msg = "stuck"
                    return (0.0, 0.0, 0.0)
            self._stuck_ref = (px, py)
            self._stuck_t = now

        self.status_msg = f"{gd:.2f} m to goal"
        return self.mppi.solve(pose, twist, self.path, grid, occ, self.goal)

    def snapshot(self):
        return {"state": self.state, "goal": self.goal, "msg": self.status_msg,
                "pathLen": len(self.path)}
