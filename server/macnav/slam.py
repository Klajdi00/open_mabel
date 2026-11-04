"""
slam.py — gmapping-style Rao-Blackwellized particle-filter SLAM (FastSLAM
with per-particle occupancy grids), ported from the validated slam_env deploy
(fast_nav.py) with its hyperparameters, adapted to a HOLONOMIC base:

  proposal    each particle propagates through the odometry increment with
              Gaussian noise on (dx, dy, dθ)  (the swerve twist replaces the
              Ackermann (v, δ) sampling of the reference)
  weighting   correlative map likelihood: Σ(p_endpoint − 0.5) · GAIN on the
              particle's own map (subsampled beams)
  resampling  low-variance, gated on N_eff < N/2 AND actual motion,
              with roughening jitter against impoverishment
  mapping     log-odds Bresenham update (gridmap.py), motion-gated

No loop closure (as in the reference); the map plane shows the best particle.
"""
from __future__ import annotations

import math

import numpy as np

from .gridmap import GridMap

# ── ported hyperparameters (slam_env deploy) ─────────────────────────────────
NUM_PARTICLES = 30
GAIN = 10.0                       # SCAN_LIKELIHOOD_GAIN
RAY_STEP = 4                      # subsample every 4th beam
MOTION_SKIP_DIST = 0.03           # m — skip update below this motion…
MOTION_SKIP_ROT = math.radians(2)  # …and this rotation
RESAMPLE_DIST = 0.2               # resample only after real motion
RESAMPLE_ROT = 0.2
ROUGH_XY = 0.005
ROUGH_TH = math.radians(0.5)
BOOTSTRAP_UPDATES = 10
# holonomic proposal noise (per meter / per rad of odometry increment)
NOISE_XY_PER_M = 0.15
NOISE_TH_PER_RAD = 0.15
NOISE_XY_FLOOR = 0.004
NOISE_TH_FLOOR = 0.002
# hill-climb proposal offsets (module constant so experiments can ablate the
# scan-match refinement to the identity proposal ((0,0,0),) )
HILL_OFFS = ((0, 0, 0), (0.05, 0, 0), (-0.05, 0, 0), (0, 0.05, 0),
             (0, -0.05, 0), (0, 0, 0.05), (0, 0, -0.05))


class SlamEngine:
    def __init__(self, n_particles=NUM_PARTICLES, size_m=24.0, resolution=0.05,
                 max_range=8.0, seed=0, lidar_offset=(0.122, 0.0)):
        """lidar_offset: the scanner's mount position in the BODY frame —
        MABEL's ring sits 0.122 m ahead of the chassis origin (MJCF `lidar`
        body). Ignoring it puts a hard ~0.12-0.17 m floor under the pose RMSE
        (measured in exp_macnav E2), so every projection below emanates from
        the OFFSET origin, not the base."""
        self.rng = np.random.default_rng(seed)
        self.n = n_particles
        self.max_range = float(max_range)
        self.lidar_offset = (float(lidar_offset[0]), float(lidar_offset[1]))
        self.poses = np.zeros((n_particles, 3))
        self.weights = np.full(n_particles, 1.0 / n_particles)
        self.maps = [GridMap(size_m, resolution) for _ in range(n_particles)]
        self._updates = 0
        self._since_update = [0.0, 0.0]      # dist, rot
        self._since_resample = [0.0, 0.0]
        self._last_odom = None

    # ── the step: odometry increment + scan ──────────────────────────────────
    def step(self, odom_pose, scan_angles, scan_ranges):
        """odom_pose (x,y,yaw) from SwerveOdometry; scan in the BODY frame
        (angle i relative to base +x). Returns the best pose estimate."""
        if self._last_odom is None:
            self._last_odom = tuple(odom_pose)
            self.poses[:] = odom_pose
        # odometry increment in the PREVIOUS body frame
        ox, oy, oth = self._last_odom
        nx, ny, nth = odom_pose
        c, s = math.cos(-oth), math.sin(-oth)
        dx_w, dy_w = nx - ox, ny - oy
        dx = dx_w * c - dy_w * s
        dy = dx_w * s + dy_w * c
        dth = _wrap(nth - oth)
        self._last_odom = tuple(odom_pose)

        # propagate every particle by the increment + proposal noise
        d = math.hypot(dx, dy)
        sxy = NOISE_XY_FLOOR + NOISE_XY_PER_M * d
        sth = NOISE_TH_FLOOR + NOISE_TH_PER_RAD * abs(dth) + 0.05 * d
        th = self.poses[:, 2]
        cx, sx = np.cos(th), np.sin(th)
        self.poses[:, 0] += dx * cx - dy * sx + self.rng.normal(0, sxy, self.n)
        self.poses[:, 1] += dx * sx + dy * cx + self.rng.normal(0, sxy, self.n)
        self.poses[:, 2] = _wrapv(self.poses[:, 2] + dth +
                                  self.rng.normal(0, sth, self.n))

        for k, v in ((0, d), (1, abs(dth))):
            self._since_update[k] += v
            self._since_resample[k] += v

        moved = (self._since_update[0] >= MOTION_SKIP_DIST
                 or self._since_update[1] >= MOTION_SKIP_ROT)
        if not moved and self._updates >= BOOTSTRAP_UPDATES:
            return self.best_pose()
        self._since_update = [0.0, 0.0]

        # subsampled beams with valid returns
        a = np.asarray(scan_angles)[::RAY_STEP]
        r = np.asarray(scan_ranges, dtype=float)[::RAY_STEP]
        ok = np.isfinite(r) & (r > 0.1)
        a, r = a[ok], np.minimum(r[ok], self.max_range)
        hit = r < (self.max_range - 0.1)

        # hill-climb scan-match refinement (gmapping's improved proposal, the
        # reference's 7-pose greedy search: ±5 cm x/y, ±0.05 rad θ) — actively
        # pulls a drifted particle back onto its own map before weighting, so
        # odometry scrub error is CORRECTED, not just diluted; then the
        # correlative likelihood weights the refined particles.
        if self._updates >= 1 and hit.any():
            ah, rh = a[hit], r[hit]
            n_h = max(1, int(hit.sum()))
            offs = HILL_OFFS
            ol, ot = self.lidar_offset
            logw = np.empty(self.n)
            for i in range(self.n):
                gm = self.maps[i]
                best = None
                for (dx2, dy2, dt2) in offs:
                    px = self.poses[i, 0] + dx2
                    py = self.poses[i, 1] + dy2
                    pth = self.poses[i, 2] + dt2
                    cth, sth = math.cos(pth), math.sin(pth)
                    lx = px + ol * cth - ot * sth      # scanner origin
                    ly = py + ol * sth + ot * cth
                    xs = lx + np.cos(ah + pth) * rh
                    ys = ly + np.sin(ah + pth) * rh
                    sc = gm.score_endpoints(xs, ys)
                    # motion penalty (fastGmapping): don't wander for free
                    sc -= (math.hypot(dx2, dy2) * 40 + abs(dt2) * 20) * 0.02
                    if best is None or sc > best[0]:
                        best = (sc, px, py, pth)
                logw[i] = best[0] * GAIN / n_h
                self.poses[i] = (best[1], best[2], _wrap(best[3]))
            logw -= logw.max()
            w = np.exp(logw)
            self.weights = w / w.sum()

        # resample (gated on N_eff and real motion)
        neff = 1.0 / float(np.sum(self.weights ** 2))
        if (neff < self.n / 2 and
                (self._since_resample[0] > RESAMPLE_DIST
                 or self._since_resample[1] > RESAMPLE_ROT)):
            self._resample()
            self._since_resample = [0.0, 0.0]

        # map update on every particle (its own pose, its own map) — rays
        # emanate from the OFFSET scanner origin, not the base
        ol, ot = self.lidar_offset
        for i in range(self.n):
            px, py, pth = self.poses[i]
            cth, sth = math.cos(pth), math.sin(pth)
            self.maps[i].integrate_scan(px + ol * cth - ot * sth,
                                        py + ol * sth + ot * cth,
                                        a + pth, r, self.max_range)
        self._updates += 1
        return self.best_pose()

    def _resample(self):
        # low-variance resampling + map handoff + roughening
        idx = np.zeros(self.n, dtype=int)
        r0 = self.rng.uniform(0, 1.0 / self.n)
        cum = np.cumsum(self.weights)
        j = 0
        for i in range(self.n):
            u = r0 + i / self.n
            while u > cum[j]:
                j += 1
            idx[i] = j
        self.poses = self.poses[idx].copy()
        # copy-on-select maps (duplicate ancestors share until next write —
        # cheap deep copy of the float grid keeps semantics simple)
        used = {}
        new_maps = []
        for j in idx:
            if j in used:
                m = GridMap.__new__(GridMap)
                src = self.maps[j]
                m.res, m.n, m.ox, m.oy = src.res, src.n, src.ox, src.oy
                m.log = src.log.copy()
                new_maps.append(m)
            else:
                used[j] = True
                new_maps.append(self.maps[j])
        self.maps = new_maps
        self.weights[:] = 1.0 / self.n
        self.poses[:, 0] += self.rng.normal(0, ROUGH_XY, self.n)
        self.poses[:, 1] += self.rng.normal(0, ROUGH_XY, self.n)
        self.poses[:, 2] = _wrapv(self.poses[:, 2] +
                                  self.rng.normal(0, ROUGH_TH, self.n))

    # ── readouts ─────────────────────────────────────────────────────────────
    def best_index(self):
        return int(np.argmax(self.weights))

    def best_pose(self):
        return tuple(self.poses[self.best_index()])

    def best_map(self) -> GridMap:
        return self.maps[self.best_index()]

    def spread(self):
        sx = float(np.std(self.poses[:, 0]))
        sy = float(np.std(self.poses[:, 1]))
        R = float(np.hypot(np.mean(np.cos(self.poses[:, 2])),
                           np.mean(np.sin(self.poses[:, 2]))))
        sth = math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-9))))
        return sx, sy, sth

    def neff(self):
        return 1.0 / float(np.sum(self.weights ** 2))


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _wrapv(x):
    return (x + np.pi) % (2 * np.pi) - np.pi
