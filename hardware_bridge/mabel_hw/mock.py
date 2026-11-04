"""mock.py — hardware-free plants for every subsystem.

``MabelHardware(mock=True)`` wires these in place of the device drivers, so the
WHOLE facade — handles, leases, snapshots, estop — runs on any machine with
zero hardware and zero root. They are deliberately simple first-order plants
(deterministic under paced tick()s), not simulators; the MuJoCo/Genesis twins
own physics.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np

from .types import monotonic

__all__ = ["MockSwerve", "MockSwerveState", "MockHand", "MockLiftDriver",
           "MockLidar", "MockAudio"]


class MockSwerveState:
    """Duck-type of maxswerve_host.SwerveState (the fields BaseSubsystem reads)."""

    def __init__(self, steer, drive_v, drive_p):
        self.steer_angle_rad = list(steer)
        self.drive_velocity = list(drive_v)      # m/s
        self.drive_position = list(drive_p)      # rotations
        self.drive_torque_est = [0.0, 0.0, 0.0]
        self.recv_time = monotonic()

    def age(self):
        return monotonic() - self.recv_time


class MockSwerve:
    """SwerveBase-surface plant: latched twist -> per-module IK -> first-order
    wheel response on tick(). Also serves as its own `sw` (getState duck)."""

    def __init__(self, *, alpha: float = 0.5, wheel_radius: float = 0.0381):
        self.R = wheel_radius
        self._alpha = alpha
        self._lock = threading.Lock()
        self._twist = (0.0, 0.0, 0.0)
        self._estop = False
        self._steer = np.zeros(3)
        self._drive_cmd = np.zeros(3)
        self._drive_v = np.zeros(3)
        self._drive_p = np.zeros(3)
        self.sw = self                            # BaseSubsystem reads base.sw

    # SwerveBase surface -------------------------------------------------------
    def set_twist(self, vx, vy, wz):
        from .drivers import base_ik
        with self._lock:
            self._twist = (vx, vy, wz)
            if self._estop:
                self._drive_cmd = np.zeros(3)
                return
            drive, steer = base_ik.inverse_kinematics(vx, vy, wz, self._steer)
            self._drive_cmd = np.asarray(drive, dtype=float)
            self._steer = np.asarray(steer, dtype=float)

    def set_estop(self, on: bool):
        self._estop = bool(on)
        if self._estop:
            with self._lock:
                self._drive_cmd = np.zeros(3)

    def tick(self):
        with self._lock:
            self._drive_v += self._alpha * (self._drive_cmd - self._drive_v)
            self._drive_p += self._drive_v / (2 * math.pi * self.R) * 0.02

    def service(self):
        pass

    def joint_state(self):
        from .drivers.swerve import STEER_JOINTS, DRIVE_JOINTS
        names = STEER_JOINTS + DRIVE_JOINTS
        with self._lock:
            pos = [float(s) * 2 * math.pi for s in self._steer] + [0.0] * 3
            vel = [0.0] * 3 + [float(v) / self.R for v in self._drive_v]
        return names, pos, vel

    def connected(self):
        return True

    def close(self):
        pass

    # sw (SwerveClient) duck ----------------------------------------------------
    def getState(self):
        with self._lock:
            return MockSwerveState([s * 2 * math.pi for s in self._steer],
                                   self._drive_v, self._drive_p)

    def getHealthState(self, timeout=None):
        return None


class MockHand:
    """Controller-Hand duck over OrcaHand.mock(side): free-drive-safe (targets
    are dropped while torque is off, like the real controller)."""

    def __init__(self, side: str):
        from .drivers.orcahand import OrcaHand
        self.side = side
        self.hand = OrcaHand.mock(side)           # .hand != None => "connected"
        self.hand.connect(enable_torque=False, scan=True)

    def set_positions(self, targets):
        if not self.hand.is_torque_enabled:
            return
        self.hand.set_joint_positions(dict(targets))

    def set_torque(self, on: bool):
        if bool(on):
            self.hand.enable_torque()
        else:
            self.hand.disable_torque()

    def is_torque_enabled(self) -> bool:
        return bool(self.hand.is_torque_enabled)

    def get_state(self):
        return self.hand.get_state()

    @property
    def joint_names(self):
        return list(self.hand.joint_names)

    def service(self):
        pass

    def close(self):
        try:
            self.hand.disconnect()
        except Exception:
            pass


class MockLiftDriver:
    """LiftDriver-surface plant (instant-ish height tracking, homed).

    Heights are ABSOLUTE, like the real firmware: the column lives in
    ``[min_pos, max_pos]`` and set_height clamps to it. ``min_pos`` defaults to 0
    (a flat, datum-less plant) so existing tests see pass-through; a test can set
    ``.min_pos``/``.max_pos`` to model a real calibrated datum (floor 0.3175 m) and
    exercise the lift_node stroke<->absolute conversion."""

    def __init__(self, *, max_pos: float = 0.5, min_pos: float = 0.0):
        self.max_pos = max_pos
        self.min_pos = min_pos
        self._h = min_pos
        self._connected = False
        self._calibrated = False
        self._torque = True         # firmware boots energized; a released motor won't drive
        self._vel = 0.0             # last commanded jog velocity (m/s)

    def connect(self):
        self._connected = True

    def close(self):
        self._connected = False

    def connected(self):
        return self._connected

    def service(self):
        pass

    def read(self):
        from .types import LiftState
        return LiftState(height=self._h if self._connected else None,
                         velocity=0.0, homed=self._connected,
                         connected=self._connected, stamp=monotonic(),
                         calibrated=self._calibrated, min_pos=self.min_pos,
                         max_pos=self.max_pos)

    def set_height(self, meters: float):
        if not self._torque:        # a released (torque-off) column ignores drive commands
            return
        self._h = max(self.min_pos, min(self.max_pos, float(meters)))

    def set_velocity(self, mps: float):
        self._vel = float(mps)      # velocity-mode jog hold (torque must be on to matter)

    def home(self, wait: bool = False):
        self._h = self.min_pos

    def calibrate(self, wait: bool = False):
        self._h = self.min_pos
        self._calibrated = True

    def stop(self):
        self._vel = 0.0             # halt jog motion

    def set_torque(self, on: bool):
        self._torque = bool(on)     # a released (torque-off) column ignores set_height


class MockLidar:
    """LidarSubsystem-surface plant: a synthetic 360° revolution (4 m walls) on
    every latest() call; on_scan callbacks are fired by an explicit spin()."""

    def __init__(self, bins: int = 360):
        self._bins = bins
        self._cbs: list = []

    def connect(self):
        pass

    def close(self):
        pass

    def connected(self):
        return True

    def _make(self):
        from .types import LidarScan
        angles = np.linspace(0.0, 2 * math.pi, self._bins, endpoint=False)
        ranges = np.full(self._bins, 4.0) + 0.5 * np.sin(3 * angles)
        scan = LidarScan(angles=angles, ranges=ranges,
                         quality=np.full(self._bins, 47.0), stamp=monotonic())
        object.__setattr__(scan, "_scan_time", 0.2)
        return scan

    def latest(self) -> Optional[object]:
        return self._make()

    def on_scan(self, callback):
        self._cbs.append(callback)
        return lambda: self._cbs.remove(callback)

    def spin(self):
        scan = self._make()
        for cb in list(self._cbs):
            cb(scan)


class MockAudio:
    """AudioSubsystem-surface plant: a 440 Hz tone as the "mic", a sink as the
    "speaker". play() records into `played` so a test can assert what was sent
    without a sound card; on_chunk callbacks fire on an explicit spin()."""

    RATE = 16000
    CHANNELS = 1
    FRAME_MS = 20

    def __init__(self):
        self._cbs: list = []
        self._seq = 0
        self._muted = False
        self.played: list = []          # every frame handed to play(), in order
        self.volume = 70

    def connect(self):
        pass

    def close(self):
        pass

    def connected(self):
        return True

    def _make(self):
        from .types import AudioChunk
        n = int(self.RATE * self.FRAME_MS / 1000)
        t = (np.arange(n) + self._seq * n) / self.RATE
        pcm = (0.2 * np.sin(2 * math.pi * 440.0 * t) * 32767).astype("<i2")
        if self._muted:
            pcm = np.zeros_like(pcm)
        chunk = AudioChunk(pcm=pcm.tobytes(), seq=self._seq, rate=self.RATE,
                           channels=self.CHANNELS, stamp=monotonic())
        self._seq += 1
        return chunk

    def latest(self) -> Optional[object]:
        return self._make()

    def on_chunk(self, callback):
        self._cbs.append(callback)

        def _unsub():
            try:
                self._cbs.remove(callback)
            except ValueError:
                pass
        return _unsub

    def state(self):
        from .types import AudioState
        return AudioState(device="mock", card="mock", connected=True,
                          capturing=bool(self._cbs), playing=bool(self.played),
                          muted=self._muted, rate=self.RATE,
                          channels=self.CHANNELS, playback_volume=self.volume,
                          capture_volume=100, mic_rms=0.2, mic_silent=False,
                          stamp=monotonic())

    def health(self):
        return {"speaker": {"open": bool(self.played), "volumePct": self.volume,
                            "queuedFrames": 0},
                "mic": {"open": bool(self._cbs), "volumePct": 100,
                        "muted": self._muted, "rms": 0.2, "frames": self._seq}}

    def play(self, pcm: bytes):
        self.played.append(bytes(pcm))

    def set_volume(self, percent: int, capture: bool = False):
        if not capture:
            self.volume = int(percent)

    def set_muted(self, muted: bool):
        self._muted = bool(muted)

    def stop(self):
        self.played.clear()

    def spin(self):
        chunk = self._make()
        for cb in list(self._cbs):
            cb(chunk)
