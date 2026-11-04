"""Per-module actuator model of the 3-module MAXSwerve base (NumPy-only).

`base_response` treats the base as a body-twist filter; it never looks at a wheel.
This module does: it turns the body twist the controller SENDS into what three
real modules can deliver, one module at a time, and returns the body twist they
actually produce. It answers two questions the body-level model cannot:

  * are the drive and steer motors inside their speed / torque limits, with margin?
  * how non-holonomic is the base?  A swerve module must first ROTATE its wheel to
    the new heading; until it has, the drive is choked (the controller's own law,
    `swerve_ik`: zero drive past STEER_DEADBAND, cos(err) inside it), so a
    sideways command does not move the robot sideways at once.

Per module i at chassis position r_i (config.MODULE_POSITIONS, chassis frame):

    command      u_i = (vx - w r_iy,  vy + w r_ix)                (rigid body)
    desaturate   if max_i |u_i| > WHEEL_V_SAFE, scale the WHOLE twist down
                 (keeps the direction; what every swerve stack does)
    steer        theta_i <- theta_i + clip(err / STEER_TAU, +-STEER_RATE_MAX) dt,
                 err from the shortest-path target (optimize_steering)
    drive        s_i* = |u_i| * dir * choke(err),   choke = 0 if err > deadband else cos(err)
                 s_i  <- s_i + clip(s_i* - s_i, +-WHEEL_A_SAFE dt)       (torque limit)
    realised     wheel ground velocity g_i = s_i (cos phi_i, sin phi_i),  phi_i = theta_i + yaw_i
                 body twist = least-squares fit of the rigid map to the three g_i;
                 the residual is wheel SCRUB (the wheels disagree about the motion).

Limits (all at the module output; sources in the comments):
  drive  REV MAXSwerve, NEO, 14T pinion (4.71:1), 3 in wheel. Free speed 5676 rpm ->
         126 rad/s -> 4.80 m/s (= the MJCF drive ctrlrange +-126 rad/s). BUT the real
         SPARK velocity loop collapses above ~1.5 m/s commanded (hardware_bridge doc,
         drive_diag.py: 2 of 3 wheels drop to ~0 at 17-21 A), so the SAFE wheel speed
         is 1.2 m/s = 80 % of the collapse, 25 % of free speed.
         Stall torque at the wheel 2.6 N m * 4.71 = 12.2 N m (MJCF forcerange 11);
         WHEEL_TAU_SAFE = 50 % of the MJCF 11 N m -> a per-wheel acceleration cap of
         tau / (R * M/3) with the MJCF's 61.7 kg.
  steer  NEO 550 through the MAXSwerve azimuth 9424:203 (46.42:1): free 11000 rpm ->
         24.8 rad/s at the module (STEER_RATE_FREE). The steer runs at FULL speed, no
         software derate. What full speed IS on the hardware was MEASURED on NOVA's
         identical modules (NOVA/hardware_bridge/nova_hw/config.py, 2026-09-24, kP swept
         7..30): a 90 deg step takes ~170 ms on every module, and it is CURRENT-limited
         (kP 30 drew 34 A and went slower). STEER_RATE_HW = 10 rad/s with a STEER_TAU_HW
         = 20 ms position loop reproduces that (172 ms to 2 deg). Both profiles use it:
         the modules are the same part. The MJCF steer servo (kp 350, kv 12) peaks at
         43 rad/s on a 180 deg step and settles 90 deg in 112 ms -- the sim base is MORE
         holonomic than the hardware ("mabel_mjcf" profile: 24.8 rad/s, 35 ms).

PROFILES: "mabel" is the above. "nova" is NOVA's base as its own driver runs it
(NOVA/hardware_bridge/nova_hw: config.BASE + subsystems/base.py): the same MAXSwerve
modules on a different triangle (front track 0.34 m, wheelbase 0.36 m, incenter
origin), OPEN-LOOP duty drive whose measured top speed is 1.04 m/s forward (safe =
80 % of it), a 0.75 m/s^2 wheel slew, the anti-slip choke cos(err) with NO deadband,
and a 0.08 s low-pass on the body twist. Geometry is given in the frame of the
model being integrated (MABEL's chassis: front = -x, left = -y), so a profile is
only a set of numbers.

`check_limits` reports peak usage of each limit for a logged command sequence.
"""
import math

import numpy as np

from .tip_over import MAX_LIN_ACCEL

# geometry: the controller's (mabel/config.py) -- read lazily to stay import-light
WHEEL_RADIUS = 0.0381

# ── drive (NEO + MAXSwerve 14T) ─────────────────────────────────────────────
DRIVE_FREE_RAD_S = 5676.0 / 4.71 * 2 * math.pi / 60.0   # 126.2 rad/s at the wheel
WHEEL_V_FREE = DRIVE_FREE_RAD_S * WHEEL_RADIUS           # 4.81 m/s
WHEEL_V_COLLAPSE = 1.5                                   # m/s, measured SPARK loop collapse
WHEEL_V_SAFE = 1.2                                       # m/s, 80 % of the collapse
WHEEL_TAU_MAX = 11.0                                     # N m, MJCF drive forcerange
WHEEL_TAU_SAFE = 0.5 * WHEEL_TAU_MAX                     # N m, 50 % margin
ROBOT_MASS = 61.7                                        # kg, MJCF subtree mass
WHEEL_A_SAFE = WHEEL_TAU_SAFE / (WHEEL_RADIUS * ROBOT_MASS / 3.0)   # 7.0 m/s^2

# ── steer (NEO 550 + MAXSwerve azimuth 46.42:1) ─────────────────────────────
STEER_RATE_FREE = 11000.0 / (9424.0 / 203.0) * 2 * math.pi / 60.0  # 24.8 rad/s, motor free speed
STEER_RATE_HW = 10.0                                     # rad/s, measured: current-limited (NOVA)
STEER_TAU_HW = 0.020                                     # s, measured: 90 deg in ~170 ms
STEER_RATE_MAX, STEER_TAU = STEER_RATE_HW, STEER_TAU_HW  # the defaults = the hardware
STEER_DEADBAND = 0.52                                    # rad, = config.STEER_DEADBAND


def _nova_positions(front_track=0.34, wheelbase=0.36):
    """NOVA base.module_positions (body frame +x fwd, +y left, incenter origin),
    returned in MABEL's chassis frame (front -x, left -y) in fl, fr, b order."""
    fl, fr, bk = (0.0, front_track / 2), (0.0, -front_track / 2), (-wheelbase, 0.0)
    d = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])
    a, b, c = d(fr, bk), d(fl, bk), d(fl, fr)
    ix = (a * fl[0] + b * fr[0] + c * bk[0]) / (a + b + c)
    iy = (a * fl[1] + b * fr[1] + c * bk[1]) / (a + b + c)
    return np.array([[-(x - ix), -(y - iy)] for x, y in (fl, fr, bk)])


PROFILES = {
    "mabel": dict(),
    "mabel_mjcf": dict(steer_rate=STEER_RATE_FREE, steer_tau=0.035),
    "nova": dict(positions=_nova_positions(), wheel_v=0.8 * 1.04, wheel_a=0.75,
                 choke="cos", twist_tau=0.08),
}


def optimize_steering(target, current):
    """Shortest-path heading and drive sign; mirrors
    whole_body_controller.base_controller.swerve.optimize_steering (that module
    pulls MuJoCo in; this one must not). A test pins the two equal."""
    target = current + ((target - current + math.pi) % (2 * math.pi) - math.pi)
    if target - current > math.pi / 2:
        return target - math.pi, -1.0
    if target - current < -math.pi / 2:
        return target + math.pi, -1.0
    return target, 1.0


def _geometry():
    from .. import config as C
    keys = list(C.MODULE_POSITIONS)
    r = np.array([C.MODULE_POSITIONS[k] for k in keys], float)
    off = np.array([C.MODULE_YAW_OFFSETS[k] for k in keys], float)
    return keys, r, off


def module_velocities(vx, vy, w, r):
    """(3,2) ground-velocity vectors the rigid twist asks of each module."""
    return np.stack([vx - w * r[:, 1], vy + w * r[:, 0]], axis=1)


class SwerveModules:
    """Stateful three-module model. `step(cmd, dt)` -> realised (vx, vy, w)."""

    def __init__(self, steer0=None, choke="deadband", steer_rate=STEER_RATE_MAX,
                 steer_tau=STEER_TAU, wheel_v=WHEEL_V_SAFE, wheel_a=WHEEL_A_SAFE,
                 positions=None, twist_tau=0.0, profile=None):
        if profile is not None:
            return self.__init__(steer0=steer0, **PROFILES[profile])
        self.keys, self.r, self.off = _geometry()
        if positions is not None:
            self.r = np.asarray(positions, float).reshape(len(self.keys), 2)
        self.twist_tau = float(twist_tau)
        self._tw_f = np.zeros(3)
        n = len(self.keys)
        self.theta = np.zeros(n) if steer0 is None else np.array(steer0, float)
        self.s = np.zeros(n)                   # signed wheel ground speed, m/s
        self.choke = choke
        self.steer_rate, self.steer_tau = float(steer_rate), float(steer_tau)
        self.wheel_v, self.wheel_a = float(wheel_v), float(wheel_a)
        # least-squares forward kinematics: g = A [vx vy w]
        A = np.zeros((2 * n, 3))
        A[0::2, 0] = 1.0
        A[0::2, 2] = -self.r[:, 1]
        A[1::2, 1] = 1.0
        A[1::2, 2] = self.r[:, 0]
        self._A, self._Ap = A, np.linalg.pinv(A)
        self.last = {}

    SUBSTEP = 0.005          # s; the explicit steer/drive update is exact well below STEER_TAU

    def step(self, cmd, dt):
        """Advance by dt (sub-stepped at <= SUBSTEP); returns the MEAN realised twist
        over dt (what integrates the pose exactly), `last` holds the worst per-module
        usage seen inside the interval."""
        n = max(1, int(math.ceil(dt / self.SUBSTEP)))
        h = dt / n
        acc = np.zeros(3)
        worst = None
        for _ in range(n):
            acc += self._step(cmd, h)
            if worst is None:
                worst = {k: (np.abs(v).copy() if isinstance(v, np.ndarray) else v)
                         for k, v in self.last.items()}
            else:
                for k, v in self.last.items():
                    if isinstance(v, np.ndarray) and k != "wheel_v":
                        worst[k] = np.maximum(worst[k], np.abs(v))
                    elif k == "scrub":
                        worst[k] = max(worst[k], v)
                    elif k == "desat":
                        worst[k] = min(worst[k], v)
        worst["wheel_v"] = self.s.copy()          # signed, end of interval
        self.last = worst
        return tuple(float(x) for x in acc / n)

    def _step(self, cmd, dt):
        if self.twist_tau > 0:                 # the driver's body-twist low-pass (NOVA)
            self._tw_f += (1.0 - math.exp(-dt / self.twist_tau)) * (np.asarray(cmd, float) - self._tw_f)
            cmd = self._tw_f
        vx, vy, w = (float(c) for c in cmd)
        u = module_velocities(vx, vy, w, self.r)
        need = np.linalg.norm(u, axis=1)
        k = 1.0
        if need.max() > self.wheel_v:          # desaturate the whole twist
            k = self.wheel_v / need.max()
            u, need = u * k, need * k
        err = np.zeros(len(self.keys))
        s_tgt = np.zeros(len(self.keys))
        steer_w = np.zeros(len(self.keys))
        for i in range(len(self.keys)):
            if need[i] > 1e-3:
                tgt, sign = optimize_steering(math.atan2(u[i, 1], u[i, 0]) - self.off[i],
                                              self.theta[i])
                e = tgt - self.theta[i]
                steer_w[i] = float(np.clip(e / self.steer_tau, -self.steer_rate, self.steer_rate))
                self.theta[i] += steer_w[i] * dt
                if abs(steer_w[i] * dt) >= abs(e):
                    self.theta[i] = tgt
                err[i] = abs(tgt - self.theta[i])
                if self.choke == "deadband":     # the controller's swerve_ik
                    choke = 0.0 if err[i] > STEER_DEADBAND else math.cos(err[i])
                elif self.choke == "cos":        # NOVA's driver: max(0, cos err)
                    choke = max(0.0, math.cos(err[i]))
                else:                            # MABEL's HAL base_ik: none
                    choke = 1.0
                s_tgt[i] = need[i] * sign * choke
        ds = np.clip(s_tgt - self.s, -self.wheel_a * dt, self.wheel_a * dt)
        acc = ds / dt if dt > 0 else np.zeros_like(ds)
        self.s += ds
        phi = self.theta + self.off
        g = np.stack([self.s * np.cos(phi), self.s * np.sin(phi)], axis=1).reshape(-1)
        tw = self._Ap @ g
        scrub = np.linalg.norm(g - self._A @ tw) / math.sqrt(len(self.keys))
        self.last = {"desat": k, "steer_err": err, "steer_rate": steer_w,
                     "wheel_v": self.s.copy(), "wheel_a": acc, "wheel_v_req": need / k,
                     "scrub": float(scrub)}
        return tw


def check_limits(cmds, dt, **kw):
    """Run a (N,3) command sequence through SwerveModules and report peak usage.

    Returns a dict of arrays (per step) plus `peak` = the worst value of each limit
    and `margin` = 1 - peak/limit (negative = violated)."""
    sm = SwerveModules(**kw)
    out = {k: [] for k in ("tw", "wheel_v", "wheel_v_req", "wheel_a", "steer_rate",
                           "steer_err", "scrub", "desat")}
    for c in np.asarray(cmds, float):
        out["tw"].append(sm.step(c, dt))
        for k in out:
            if k != "tw":
                out[k].append(sm.last[k])
    out = {k: np.asarray(v) for k, v in out.items()}
    wv_req = float(np.abs(out["wheel_v_req"]).max())
    wv = float(np.abs(out["wheel_v"]).max())
    wa = float(np.abs(out["wheel_a"]).max())
    sr = float(np.abs(out["steer_rate"]).max())
    out["peak"] = {"wheel_v_req": wv_req, "wheel_v": wv, "wheel_a": wa, "steer_rate": sr,
                   "wheel_tau": wa * WHEEL_RADIUS * ROBOT_MASS / 3.0}
    out["margin"] = {"wheel_v_safe": 1 - wv / sm.wheel_v,
                     "wheel_v_collapse": 1 - wv / WHEEL_V_COLLAPSE,
                     "wheel_v_free": 1 - wv / WHEEL_V_FREE,
                     "wheel_tau_max": 1 - out["peak"]["wheel_tau"] / WHEEL_TAU_MAX,
                     "steer_rate": 1 - sr / sm.steer_rate,
                     "steer_rate_free": 1 - sr / STEER_RATE_FREE}
    return out


__all__ = ["SwerveModules", "PROFILES", "check_limits", "optimize_steering", "module_velocities",
           "WHEEL_V_SAFE", "WHEEL_V_COLLAPSE", "WHEEL_V_FREE", "WHEEL_A_SAFE",
           "WHEEL_TAU_MAX", "WHEEL_TAU_SAFE", "STEER_RATE_MAX", "STEER_TAU",
           "STEER_RATE_FREE", "STEER_RATE_HW", "STEER_TAU_HW",
           "MAX_LIN_ACCEL"]
