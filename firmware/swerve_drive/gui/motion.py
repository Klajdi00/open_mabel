"""Swerve-drive kinematics — the motion model the telemetry feeds into.

Mirrors the YOR robot's approach (robot/base_motor.py): a linear map between
chassis velocity (vx, vy, omega) and per-module (wheel speed, steer angle),
using each module's (x, y) position from robot center.

    * inverse(vx, vy, omega)        → [(speed_mps, angle_rad), ...]   (command)
    * forward(speeds, angles)       → (vx, vy, omega)                 (odometry)

``forward`` is the one that consumes live **drive-velocity feedback** + steer
angles to estimate how the chassis is actually moving (least-squares over all
modules, like YOR's matrix C). Pure Python so the GUI keeps a zero-dependency
footprint (tkinter only).
"""

import math

from . import config as C


class SwerveKinematics:
    def __init__(self, modules=None):
        # Module positions (x forward, y left), in MODULES order.
        self.pos = [(m["x"], m["y"]) for m in (modules or C.MODULES)]

    # ── Inverse: chassis velocity → per-wheel (speed, angle) ───────────────────
    def inverse(self, vx, vy, omega):
        """Desired chassis velocity (m/s, m/s, rad/s) → list of (speed_mps,
        angle_rad) per module. v_wheel = v_chassis + omega × r."""
        out = []
        for (x, y) in self.pos:
            wx = vx - omega * y
            wy = vy + omega * x
            out.append((math.hypot(wx, wy), math.atan2(wy, wx)))
        return out

    # ── Forward: per-wheel (speed, angle) → chassis velocity (odometry) ────────
    def forward(self, speeds, angles):
        """Measured wheel speeds (m/s) + steer angles (rad) → estimated chassis
        velocity (vx, vy, omega). Least-squares over every module:

            wx_i = vx - omega*y_i = speed_i*cos(angle_i)
            wy_i = vy + omega*x_i = speed_i*sin(angle_i)

        Solved via the 3x3 normal equations (AᵀA)·s = Aᵀb — no numpy needed."""
        # Build AᵀA (3x3, symmetric) and Aᵀb (3) incrementally.
        ata = [[0.0] * 3 for _ in range(3)]
        atb = [0.0, 0.0, 0.0]
        for (x, y), sp, ang in zip(self.pos, speeds, angles):
            # two rows per module: [1,0,-y]·s = sp·cos ;  [0,1,x]·s = sp·sin
            rows = ((1.0, 0.0, -y, sp * math.cos(ang)),
                    (0.0, 1.0,  x, sp * math.sin(ang)))
            for a0, a1, a2, b in rows:
                r = (a0, a1, a2)
                for i in range(3):
                    atb[i] += r[i] * b
                    for j in range(3):
                        ata[i][j] += r[i] * r[j]
        vx, vy, omega = _solve3(ata, atb)
        return vx, vy, omega


def _solve3(m, b):
    """Solve a 3x3 linear system m·x = b via Gaussian elimination with partial
    pivoting. Returns (0,0,0) if singular."""
    a = [row[:] + [b[i]] for i, row in enumerate(m)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return 0.0, 0.0, 0.0
        a[col], a[piv] = a[piv], a[col]
        f = a[col][col]
        a[col] = [v / f for v in a[col]]
        for r in range(3):
            if r != col:
                g = a[r][col]
                a[r] = [v - g * cv for v, cv in zip(a[r], a[col])]
    return a[0][3], a[1][3], a[2][3]
