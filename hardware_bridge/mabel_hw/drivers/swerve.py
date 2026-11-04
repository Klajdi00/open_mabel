"""swerve_base.py — high-level owner of the MAXSwerve mobile base.

This is "ps5_teleop minus the joystick": the SAME validated pipeline
(``twist -> inverse_kinematics -> apply_calibration -> SwerveClient``) the proven
``ps5_teleop.py`` uses, with the joystick replaced by ``set_twist()``. So the wire
path that reaches the motors is byte-identical whether the pad, ROS, or a CLI drives
it — that code-path identity is the safety guarantee.

The hardware bridge owns this object; nothing else opens the Teensy link. It is
deliberately ROS-free so it can be driven standalone (pad teleop, a CLI test) before
the ROS stack is up.

Safety / liveness (see maxswerve_host.swerve docstring):
  * ``connect`` enables the controllers (output stage live, drive forced to 0).
  * ``tick`` is called at a fixed rate (50 Hz) to RE-SEND the latched setpoint so the
    Teensy command-watchdog (~150 ms) stays fed. If the last ``set_twist`` is older
    than ``cmd_timeout`` (or e-stop is latched) ``tick`` sends ZERO drive (steer held)
    instead of latching the last speed — a dead commander => no motion.
  * ``close`` stops the link; the Teensy disables within ~150 ms (firmware fail-safe).
"""

import time

import numpy as np

from . import base_ik
from .maxswerve_host import SwerveClient

# Joint names the bridge publishes (and swerve_odometry consumes). Hardware module
# order is [back, front-right, front-left] — the SAME order as base_ik.MODULE_POS
# and SwerveClient.getState() arrays, so feedback maps straight through.
STEER_JOINTS = ["b_steer", "fr_steer", "fl_steer"]
DRIVE_JOINTS = ["b_drive", "fr_drive", "fl_drive"]


class SwerveBase:
    """Owns the Teensy swerve link + the validated ps5 IK/calibration."""

    # Drop the (native) client after this many consecutive setCommand timeouts so a
    # flaky base link cannot hammer the abort-prone keepalive thread — service()
    # reconnects when telemetry is healthy again.
    _CMD_FAIL_DROP = 10

    def __init__(self, *, teensy_ip="192.168.124.10", port=5800, stream_hz=200,
                 cmd_timeout=0.5, wheel_radius=0.0381, apply_host_calib=True,
                 drive_gain=1.0, reconnect_interval=2.0, logger=None):
        self.cmd_timeout = float(cmd_timeout)
        self.R = float(wheel_radius)
        self.apply_host_calib = bool(apply_host_calib)
        self.drive_gain = float(drive_gain)   # scales the velocity setpoint (SPARK kF=0 fudge)
        self.reconnect_interval = float(reconnect_interval)
        self._conn = dict(teensy_ip=teensy_ip, port=port, stream_hz=stream_hz)
        self._log = logger or (lambda *a, **k: None)

        # latched setpoint (tick re-sends it to feed the Teensy watchdog)
        self._drive = np.zeros(3)
        self._steer = np.zeros(3)       # prev_steer: held when a module speed ~0
        self._estop = False
        self._last_cmd_t = None
        self.sw = None
        self._last_reconnect = -1e9
        self._link_up = False           # last known telemetry-healthy state
        self._cmd_fail = 0              # consecutive setCommand failures (flaky link)
        self._connect()

    def _connect(self):
        """(Re)open the Teensy link. Non-fatal: a missing/down base leaves the bridge
        alive and idle (it auto-reconnects via service() when the link/Teensy returns)."""
        try:
            if self.sw is not None:
                self.sw.close()
        except Exception:
            pass
        self.sw = None
        self._last_reconnect = time.monotonic()
        sw = None
        try:
            sw = SwerveClient(**self._conn)
            sw.connect()                # may raise ENETUNREACH while the link is down
            self.sw = sw
            self._log(f"SwerveClient connecting -> {self._conn['teensy_ip']}:"
                      f"{self._conn['port']} @ {self._conn['stream_hz']} Hz")
        except Exception as e:                          # pragma: no cover
            try:                        # close the partial client so its threads don't leak
                if sw is not None:
                    sw.close()
            except Exception:
                pass
            self.sw = None
            self._log(f"SwerveClient connect failed ({e}); base IDLE, will retry.")

    def service(self):
        """Auto-recover the link: if telemetry has been stale (Teensy rebooted, cable
        re-plugged, or the link was down at startup so the keepalive died), tear the
        SwerveClient down and reopen it — at most once per reconnect_interval. Call from
        a LOW-rate timer (not the fast command path, since reconnect can block ~1 s)."""
        st = self.sw.getState() if self.sw is not None else None
        healthy = st is not None and st.age() < 1.0
        if healthy:
            if not self._link_up:
                self._log("Teensy link UP (telemetry streaming).")
            self._link_up = True
            return
        if self._link_up:
            self._log("Teensy link LOST (telemetry stale); retrying...")
        self._link_up = False
        if time.monotonic() - self._last_reconnect >= self.reconnect_interval:
            self._connect()

    # ---- command intake -----------------------------------------------------
    def set_twist(self, vx, vy, wz):
        """Latch a body twist (m/s, m/s, rad/s) in the REAL robot frame.

        Frame is ps5_teleop's: +x forward, +y left, wz CCW-positive. Any sim-frame
        reconciliation (the 180-deg yaw flip) is the caller's job, so the base stays
        in the one canonical frame the validated path uses.
        """
        self._last_cmd_t = time.monotonic()
        if self._estop:
            self._drive = np.zeros(3)
            return
        drive, steer = base_ik.inverse_kinematics(float(vx), float(vy), float(wz),
                                                   self._steer)
        self._drive = drive
        self._steer = steer

    def set_estop(self, on: bool):
        self._estop = bool(on)
        if self._estop:
            self._drive = np.zeros(3)

    # ---- fixed-rate streamer (call at ~50 Hz) -------------------------------
    def tick(self):
        """Re-send the latched setpoint, zeroing drive on e-stop / stale command."""
        if self.sw is None:
            return
        stale = (self._last_cmd_t is None or
                 time.monotonic() - self._last_cmd_t > self.cmd_timeout)
        drive = np.zeros(3) if (self._estop or stale) else self._drive * self.drive_gain
        steer = self._steer
        if self.apply_host_calib:
            # Host-side per-module steer-zero + drive-polarity, EXACTLY as
            # ps5_teleop.apply_calibrated. The bridge owns calibration.
            drive, steer = base_ik.apply_calibration(drive, steer)
        try:
            self.sw.setCommand(np.asarray(drive, dtype=float),
                               np.asarray(steer, dtype=float))
            self._cmd_fail = 0
        except Exception as e:                          # pragma: no cover
            # A FLAKY base link (Teensy reachable at connect, then setCommand times
            # out) used to be hammered 50x/s here — the log filled with hundreds of
            # "setCommand failed: timed out", and every one of those calls is another
            # chance for the NATIVE SwerveClient keepalive thread to throw and abort
            # the WHOLE process (`terminate called without an active exception`),
            # which takes the arms/hands/cameras DOWN with the base. A dead link
            # must not keep being commanded: after a few consecutive failures, drop
            # the client (base goes idle, motion already zeroed on estop/stale) and
            # let service() reopen it ONLY once telemetry is genuinely healthy again.
            self._cmd_fail += 1
            if self._cmd_fail <= 1 or self._cmd_fail == self._CMD_FAIL_DROP:
                self._log(f"setCommand failed: {e}"
                          + ("; base link unhealthy — dropping the client until it"
                             " recovers (service will reconnect)."
                             if self._cmd_fail == self._CMD_FAIL_DROP else ""))
            if self._cmd_fail >= self._CMD_FAIL_DROP:
                try:
                    self.sw.close()                     # stop the native keepalive thread
                except Exception:
                    pass
                self.sw = None
                self._link_up = False
                self._cmd_fail = 0

    # ---- feedback -----------------------------------------------------------
    def joint_state(self):
        """Teensy encoder readback -> (names, positions[rad], velocities[rad/s]).

        getState() is in hardware order [back, fr, fl]:
          * steer position (rad): st.steer_angle_rad[i]      -> <m>_steer
          * drive velocity (rad/s): st.drive_velocity[i] / R -> <m>_drive
        drive_velocity is the wheel GROUND speed (m/s); swerve_odometry recovers ground
        speed as velocity*R, so dividing by the SAME R here is lossless (R must match).
        Returns None when no telemetry has arrived yet.
        """
        if self.sw is None:
            return None
        st = self.sw.getState()
        if st is None:
            return None
        try:
            steer_rad = [float(v) for v in st.steer_angle_rad]   # [back, fr, fl]
            drive_mps = [float(v) for v in st.drive_velocity]    # [back, fr, fl]
        except Exception:                                        # pragma: no cover
            return None
        names = STEER_JOINTS + DRIVE_JOINTS
        position = steer_rad + [0.0, 0.0, 0.0]
        velocity = [0.0, 0.0, 0.0] + [v / self.R for v in drive_mps]
        return names, position, velocity

    def connected(self) -> bool:
        return self.sw is not None and self.sw.connected()

    # ---- lifecycle ----------------------------------------------------------
    def close(self):
        if self.sw is None:
            return
        try:
            self.sw.setCommand(np.zeros(3), self._steer)   # zero drive, hold steer
            self.sw.close()
        except Exception:                                  # pragma: no cover
            pass
        self.sw = None
