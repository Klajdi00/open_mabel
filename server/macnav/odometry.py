"""
odometry.py — swerve wheel odometry + gyro fusion for the Mac nav fallback.

Forward kinematics of the 3-module MAXSwerve base, using the SAME geometry
constants as the teleop controller and the ROS swerve_odometry node
(controller/mabel/config.py: MODULE_POSITIONS, MODULE_YAW_OFFSETS,
WHEEL_RADIUS — one motion model everywhere, per the repo's dedup rule).

Body twist from wheel states by least squares over the rigid-body constraint
    v_i = [vx − ω·r_iy,  vy + ω·r_ix]      (wheel ground velocity, body frame)
with v_i measured as  drive_speed·WHEEL_RADIUS  along  steer + yaw_offset.

Yaw is fused with the gyro (complementary filter, gyro-dominant): wheel yaw
rate drifts with slip; the IMU's wz is the robust heading source — exactly the
role the IMU plays on the ROS side. Configurable noise injection reproduces a
realistic sensor stack for the robustness experiments.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(_REPO, "controller"),):
    if p not in sys.path:
        sys.path.insert(0, p)
from mabel import config as C          # noqa: E402  (MuJoCo-free import)

MODULES = list(C.MODULE_POSITIONS.keys())          # ['fl', 'fr', 'b'] order

# SCRUB GATE threshold: translation is dropped when the wheel LS residual
# exceeds  BASE + SLOPE·|v|  (module constants so experiments can ablate it).
SCRUB_RESID_BASE = 0.06
SCRUB_RESID_SLOPE = 0.25


class NoiseModel:
    """Injected sensor noise for robustness experiments (0 = clean sim)."""

    def __init__(self, wheel_speed_std=0.0, steer_std=0.0,
                 gyro_std=0.0, gyro_bias=0.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.wheel_speed_std = wheel_speed_std
        self.steer_std = steer_std
        self.gyro_std = gyro_std
        self.gyro_bias = gyro_bias


class SwerveOdometry:
    def __init__(self, noise: NoiseModel | None = None, gyro_alpha=0.98):
        self.noise = noise or NoiseModel()
        self.gyro_alpha = float(gyro_alpha)   # 1.0 = gyro-only yaw rate
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = self.w = 0.0      # body twist estimate
        # module lever arms r_i (body frame) — the shared geometry
        self._r = np.array([C.MODULE_POSITIONS[k] for k in MODULES], dtype=float)
        self._yaw_off = np.array([C.MODULE_YAW_OFFSETS[k] for k in MODULES])

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.vx = self.vy = self.w = 0.0

    # ── one step ─────────────────────────────────────────────────────────────
    def update(self, steer, drive_rad_s, gyro_wz, dt):
        """steer[3] module steer angles (rad, module frame), drive_rad_s[3]
        wheel speeds, gyro_wz body yaw rate (rad/s), dt seconds."""
        n = self.noise
        steer = np.asarray(steer, dtype=float) + \
            (n.rng.normal(0, n.steer_std, 3) if n.steer_std else 0.0)
        speed = np.asarray(drive_rad_s, dtype=float) * C.WHEEL_RADIUS
        if n.wheel_speed_std:
            speed = speed + n.rng.normal(0, n.wheel_speed_std, 3)
        gyro = float(gyro_wz) + n.gyro_bias + \
            (n.rng.normal(0, n.gyro_std) if n.gyro_std else 0.0)

        # wheel ground-velocity vectors in the BODY frame
        ang = steer + self._yaw_off
        vwx = speed * np.cos(ang)
        vwy = speed * np.sin(ang)
        # least squares [vx, vy, w]: v_i = (vx − w·ry, vy + w·rx)
        A = np.zeros((6, 3))
        b = np.zeros(6)
        for i in range(3):
            rx, ry = self._r[i]
            A[2 * i] = (1.0, 0.0, -ry)
            A[2 * i + 1] = (0.0, 1.0, rx)
            b[2 * i] = vwx[i]
            b[2 * i + 1] = vwy[i]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        vx, vy, w_wheel = sol
        # SCRUB GATE: during steering transitions the three wheels disagree
        # (the rigid-body model can't fit them) — that translation is slip,
        # not motion. A high LS residual means "don't integrate this".
        resid = float(np.linalg.norm(A @ sol - b))
        if resid > SCRUB_RESID_BASE + SCRUB_RESID_SLOPE * float(np.hypot(vx, vy)):
            vx = vy = 0.0

        # gyro-dominant yaw-rate fusion
        w = self.gyro_alpha * gyro + (1.0 - self.gyro_alpha) * w_wheel
        self.vx, self.vy, self.w = float(vx), float(vy), float(w)

        # integrate in the world frame (midpoint heading)
        yaw_mid = self.yaw + 0.5 * w * dt
        cy, sy = math.cos(yaw_mid), math.sin(yaw_mid)
        self.x += (vx * cy - vy * sy) * dt
        self.y += (vx * sy + vy * cy) * dt
        self.yaw = _wrap(self.yaw + w * dt)
        return self.x, self.y, self.yaw

    def pose(self):
        return (self.x, self.y, self.yaw)

    def twist(self):
        return (self.vx, self.vy, self.w)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
