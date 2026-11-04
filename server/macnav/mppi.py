"""
mppi.py — Model-Predictive Path-Integral local controller (the Nav2 MPPI
analog; the upgrade over the iOS app's pure pursuit).

Samples K noisy twist sequences around the previous solution, rolls each out
through the HOLONOMIC base model with the SAME motion limits the teleop stack
uses (controller/mabel/config.py MAX_LIN_SPEED / MAX_ANG_SPEED and per-step
accel clamps mirroring safe_base_step's smoothing role), scores them, and
returns the exponentially-weighted average's first command:

  cost = path-tracking distance + goal progress + obstacle mask (inflated
         grid; hard penalty) + heading-vs-velocity alignment (the robot FACES
         its velocity direction) + control smoothness
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO, "controller") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "controller"))
from mabel import config as C          # noqa: E402

# the chassis' forward axis is −x (front camera side) — heading alignment
# must face the robot's FRONT along the velocity direction
FRONT_OFFSET = math.pi


class MPPI:
    def __init__(self, horizon=20, dt=0.1, samples=256, lam=0.35, seed=0,
                 sigma=(0.18, 0.18, 0.5)):
        self.T = horizon
        self.dt = dt
        self.K = samples
        self.lam = lam
        self.sigma = np.array(sigma)
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((self.T, 3))               # warm-started plan
        self.vmax = C.MAX_LIN_SPEED
        self.wmax = C.MAX_ANG_SPEED
        self.acc = 1.6                               # m/s² per-step clamp
        self.wacc = 4.0

    def reset(self):
        self.U[:] = 0.0

    def solve(self, pose, twist, path, grid, occ, goal):
        """pose (x,y,yaw) · twist (vx,vy,w) body · path [(x,y)..] ahead ·
        grid GridMap · occ inflated bool mask · goal (x,y).
        Returns (vx, vy, w) in the BODY frame."""
        K, T, dt = self.K, self.T, self.dt
        # sampled control sequences around the shifted previous solution
        U0 = np.roll(self.U, -1, axis=0)
        U0[-1] = U0[-2]
        eps = self.rng.normal(0, 1, (K, T, 3)) * self.sigma
        U = U0[None] + eps
        U[..., 0] = np.clip(U[..., 0], -self.vmax, self.vmax)
        U[..., 1] = np.clip(U[..., 1], -self.vmax, self.vmax)
        U[..., 2] = np.clip(U[..., 2], -self.wmax, self.wmax)
        # per-step accel clamp (the smoothing role safe_base_step plays)
        prev = np.tile(np.asarray(twist, dtype=float), (K, 1))
        for t in range(T):
            dv = np.clip(U[:, t, :2] - prev[:, :2],
                         -self.acc * dt, self.acc * dt)
            dw = np.clip(U[:, t, 2] - prev[:, 2],
                         -self.wacc * dt, self.wacc * dt)
            U[:, t, :2] = prev[:, :2] + dv
            U[:, t, 2] = prev[:, 2] + dw
            prev = U[:, t]

        # rollout (holonomic; heading integrates w)
        x = np.full(K, pose[0])
        y = np.full(K, pose[1])
        th = np.full(K, pose[2])
        cost = np.zeros(K)
        pa = np.asarray(path[:24]) if len(path) else np.asarray([goal])
        gx, gy = goal
        res, ox, oy, n = grid.res, grid.ox, grid.oy, grid.n
        for t in range(T):
            vx, vy, w = U[:, t, 0], U[:, t, 1], U[:, t, 2]
            cth, sth = np.cos(th), np.sin(th)
            x = x + (vx * cth - vy * sth) * dt
            y = y + (vx * sth + vy * cth) * dt
            th = th + w * dt
            # obstacle cost: inflated mask = near-collision
            cx = np.clip(((x - ox) / res).astype(np.int32), 0, n - 1)
            cy = np.clip(((y - oy) / res).astype(np.int32), 0, n - 1)
            cost += occ[cy, cx] * 400.0
            # path tracking: distance to the nearest of the next waypoints
            d2 = (x[:, None] - pa[:, 0]) ** 2 + (y[:, None] - pa[:, 1]) ** 2
            cost += np.sqrt(d2.min(axis=1)) * 6.0
            # goal progress
            cost += np.hypot(gx - x, gy - y) * 1.5
            # face the velocity direction: the chassis front is the −x body
            # axis, so the body-frame travel direction atan2(vy,vx) should sit
            # at ±π — any deviation means the robot isn't facing its motion
            sp = np.hypot(vx, vy)
            moving = sp > 0.08
            herr = _wrapv(np.arctan2(vy, vx) - FRONT_OFFSET)
            cost += np.where(moving, np.abs(herr) * 1.2 * sp, 0.0)
        # smoothness
        cost += np.sum(np.abs(np.diff(U, axis=1)), axis=(1, 2)) * 0.5

        beta = cost.min()
        wgt = np.exp(-(cost - beta) / self.lam)
        wgt /= wgt.sum()
        self.U = np.einsum("k,ktc->tc", wgt, U)
        u0 = self.U[0]
        return float(u0[0]), float(u0[1]), float(u0[2])


def _wrapv(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
