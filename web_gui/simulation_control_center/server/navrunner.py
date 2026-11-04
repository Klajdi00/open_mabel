"""
navrunner.py — the native Mac navigation + SLAM service (the ROS Nav2 /
slam_toolbox fallback), wired into the sim control center backend:

  inputs   mirror.base_state  (wheel steer/drive + gyro from the bridge's
                               state memmap — the same signals the real robot
                               would give)
           state.scan         (the 360-ray lidar the mirror computes)
  engine   server/macnav      (SwerveOdometry → FastSLAM → A* → MPPI)
  outputs  map plane          grid 0x04 + path 0x03 + SLAM-pose diagnostics
           driving            teleop_frame navJoystick streams over the
                              backend's own bridge socket (SimSource) — the
                              navigator is just another wire client, so the
                              driver arbitration, e-stop, deadman, and motion
                              model all apply exactly as for a human operator

Wire-sign mapping (verified empirically against the live bridge):
  wire ly+ = robot FORWARD (body −x) · wire lx+ = strafe LEFT ·
  wire rx+ = yaw CCW.  MPPI twists are body-frame (vx, vy, w) with the
  chassis front on −x.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time

import numpy as np

from . import config as CFG

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO, "server") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "server"))
if os.path.join(_REPO, "controller") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "controller"))

from macnav.odometry import NoiseModel, SwerveOdometry     # noqa: E402
from macnav.slam import SlamEngine                         # noqa: E402
from macnav.navigator import Navigator                     # noqa: E402
from mabel import config as C                              # noqa: E402

log = logging.getLogger("scc.nav")

MAX_RANGE = 10.0
# Rays that return closer than this hit the ROBOT ITSELF (the lift column and
# body sit within ~0.35 m of the lidar ring; real lidars mask this too).
# Such beams carry no world information — they are DROPPED, not treated as
# free or occupied, or the map walls the robot in with phantom self-hits.
MIN_RANGE = 0.40
GRID_PUBLISH_S = 1.0
INFLATE_CACHE_S = 1.0
TICK_HZ = 12.0
# body twist → navJoystick wire (front = −x body):
FWD_SIGN = -1.0        # ly =  FWD_SIGN * vx / MAX_LIN
LX_SIGN = -1.0         # lx =  LX_SIGN  * vy / MAX_LIN
RX_SIGN = 1.0          # rx =  RX_SIGN  * w  / MAX_ANG


class NavRunner:
    def __init__(self, state, mirror, hub, source, ms):
        """state: TelemetryState · mirror: SimMirror · hub: MirrorMapHub ·
        source: SimSource (the bridge socket) · ms: map_stream module."""
        self.state = state
        self.mirror = mirror
        self.hub = hub
        self.source = source
        self.ms = ms
        noise = NoiseModel(
            wheel_speed_std=float(os.environ.get("MABEL_NAV_WHEEL_STD", "0.01")),
            steer_std=float(os.environ.get("MABEL_NAV_STEER_STD", "0.005")),
            gyro_std=float(os.environ.get("MABEL_NAV_GYRO_STD", "0.002")),
            gyro_bias=float(os.environ.get("MABEL_NAV_GYRO_BIAS", "0.0")))
        self.odom = SwerveOdometry(noise=noise)
        self.slam = SlamEngine(max_range=MAX_RANGE)
        self.nav = Navigator()
        self._angles = np.radians(np.arange(360, dtype=float))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._occ = None
        self._occ_t = 0.0
        self._grid_t = 0.0
        self._path_sent = None
        self._last_t = None
        self._seq = 0
        self.pose = (0.0, 0.0, 0.0)
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="scc-macnav")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    # ── public API (endpoints) ───────────────────────────────────────────────
    def set_goal(self, x, y):
        with self._lock:
            self.nav.set_goal(x, y)
        log.info("nav goal → (%.2f, %.2f)", x, y)
        return self.status()

    def cancel(self):
        with self._lock:
            self.nav.cancel()
        self._drive((0.0, 0.0, 0.0))
        return self.status()

    def reinit(self):
        """Re-anchor after a sim reset / relocate teleport: odometry can't
        integrate through a teleport, so re-seed from the known placed pose
        (ground truth at placement — legitimate: we placed it) + fresh map."""
        bs = getattr(self.mirror, "base_state", None)
        gt = bs["gt"] if bs else (0.0, 0.0, 0.0)
        with self._lock:
            self.nav.cancel()
            self.odom.reset(*gt)
            self.slam = SlamEngine(max_range=MAX_RANGE)
            self._last_t = None
            self._occ = None
            self._path_sent = None
        log.info("macnav re-anchored at (%.2f, %.2f, %.2f)", *gt)

    def status(self):
        sx, sy, sth = self.slam.spread()
        return {
            "backend": "macnav (native — this Mac cannot run ROS Nav2/SLAM)",
            **self.nav.snapshot(),
            "pose": [round(v, 3) for v in self.pose],
            "odom": [round(v, 3) for v in self.odom.pose()],
            "neff": round(self.slam.neff(), 1),
            "particles": self.slam.n,
            "spread": [round(sx, 3), round(sy, 3), round(sth, 3)],
            "updates": self.slam._updates,
        }

    # ── driving (one more wire client of the bridge) ─────────────────────────
    def _drive(self, twist):
        vx, vy, w = twist
        self._seq += 1
        self.source.send("teleop_frame", {
            "sequence": self._seq, "timestamp": time.time(),
            "mode": "Base Driving",
            "navJoystick": {
                "lx": max(-1, min(1, LX_SIGN * vy / C.MAX_LIN_SPEED)),
                "ly": max(-1, min(1, FWD_SIGN * vx / C.MAX_LIN_SPEED)),
                "rx": max(-1, min(1, RX_SIGN * w / C.MAX_ANG_SPEED)),
                "ry": 0.0}})

    # ── the loop ─────────────────────────────────────────────────────────────
    def _run(self):
        period = 1.0 / TICK_HZ
        was_active = False
        while not self._stop.is_set():
            t0 = time.time()
            try:
                bs = getattr(self.mirror, "base_state", None)
                scan = self.state.scan
                if bs is None or scan is None:
                    self._stop.wait(0.5)
                    continue
                # sim-time dt
                t = bs["t"]
                dt = 0.0 if self._last_t is None else t - self._last_t
                self._last_t = t
                if dt <= 0 or dt > 0.5:
                    self._stop.wait(period)
                    continue
                self.odom.update(bs["steer"], bs["drive"], bs["gyro_wz"], dt)
                ranges = np.array([MAX_RANGE if r is None else min(r, MAX_RANGE)
                                   for r in scan], dtype=float)
                ranges[ranges < MIN_RANGE] = np.nan     # self-hits → dropped
                self.pose = self.slam.step(self.odom.pose(), self._angles, ranges)

                grid = self.slam.best_map()
                now = time.time()
                if now - self._grid_t > GRID_PUBLISH_S:
                    self._grid_t = now
                    w_, h_, res, ox, oy, cells = grid.occupancy_wire()
                    self.hub.publish_grid(self.ms.encode_occupancy(
                        w_, h_, res, ox, oy, 0.0, 0.0, cells))

                estop = bool(self.state.estop)
                with self._lock:
                    active = self.nav.active()
                    if active and estop:
                        self.nav.cancel()
                        active = False
                    if active:
                        if self._occ is None or now - self._occ_t > INFLATE_CACHE_S:
                            self._occ = grid.inflated()
                            self._occ_t = now
                        twist = self.nav.tick(self.pose, self.odom.twist(),
                                              grid, self._occ)
                    else:
                        twist = (0.0, 0.0, 0.0)
                if active:
                    self._drive(twist)
                    was_active = True
                    path = self.nav.path
                    key = (len(path), tuple(path[0]) if path else None)
                    if key != self._path_sent:
                        self._path_sent = key
                        pts = np.array([(x, y, 0.05) for x, y in path],
                                       dtype=np.float32)
                        self.hub.publish_path(self.ms.encode_path(pts))
                elif was_active:
                    was_active = False
                    self._drive((0.0, 0.0, 0.0))       # release the base
                    self.hub.publish_path(self.ms.encode_path(
                        np.zeros((0, 3), np.float32)))
            except Exception as e:
                self.error = str(e)
                log.warning("nav tick failed: %s", e)
                self._stop.wait(0.5)
            dtw = time.time() - t0
            self._stop.wait(max(0.0, period - dtw))
