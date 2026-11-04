"""types.py — frozen contracts between drivers, subsystems, the facade, and clients.

Units everywhere: rad, rad/s, N·m, m, m/s, degC, V, A; time in seconds from
``time.monotonic()``. ``stamp`` is the host-monotonic receive time of the newest
datum in the snapshot; ``age()`` is its staleness now. Every state type is an
immutable snapshot — reading never blocks writing, and a reference handed to a
client can never mutate under it.

Frame conventions: base twists are the REAL robot frame (+x forward, +y left,
+z up, yaw CCW — REP-103). Joint positions are MODEL-frame radians (calibration
is applied above the driver layer, so consumers never see raw motor units).

This module is deliberately stdlib-only (dataclasses + typing + time): it must be
importable anywhere — the Mac GUI, a ROS node on thor, a unit test — without
pulling numpy, rclpy, or any device SDK. (LidarScan *carries* numpy arrays at
runtime, but this module never imports numpy.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple


def monotonic() -> float:
    """The one clock every ``stamp`` in this package uses."""
    return time.monotonic()


class _Stamped:
    """Mixin: ``age()`` = seconds since this snapshot's newest datum."""
    stamp: float

    def age(self) -> float:
        return monotonic() - self.stamp


# ── joint groups (arms / neck motors / torso) ──────────────────────────────────

@dataclass(frozen=True)
class JointGroupState(_Stamped):
    """One MIT joint group snapshot, parallel tuples in canonical joint order."""
    names: Tuple[str, ...]
    position: Tuple[float, ...]        # rad, model frame
    velocity: Tuple[float, ...]        # rad/s
    effort: Tuple[float, ...]          # N·m
    temperature: Tuple[float, ...]     # degC (MOSFET for DaMiao; 0.0 if unreported)
    live: Tuple[bool, ...]             # per-joint feedback freshness
    torque_enabled: bool
    stamp: float

    def as_dict(self) -> Mapping[str, float]:
        """{joint: position} — the shape most consumers actually want."""
        return dict(zip(self.names, self.position))


@dataclass(frozen=True)
class NeckState(_Stamped):
    """The serial neck: motor space plus its 1:1 task-space alias (direct drive,
    so pitch/roll/yaw ARE the calibrated motor angles)."""
    motors: JointGroupState            # neck_pitch, neck_roll (+ neck_yaw when wired)
    pitch: Optional[float]             # rad; None when that motor is not wired
    roll: Optional[float]
    yaw: Optional[float]
    stamp: float


# ── mobile base (MAXSwerve) ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SwerveModuleState:
    name: str                          # "back" | "front_right" | "front_left"
    steer_angle: float                 # rad
    drive_velocity: float              # m/s ground speed
    drive_position: float              # rad, accumulated wheel angle
    drive_torque_est: float            # N·m (SPARK current-based estimate)


@dataclass(frozen=True)
class BaseHealth(_Stamped):
    """Per-module electrical health, [drive..., steer...] hardware order."""
    voltage: Tuple[float, ...]         # V
    current: Tuple[float, ...]         # A
    temperature: Tuple[float, ...]     # degC
    duty: Tuple[float, ...]            # [-1, 1]
    stamp: float


@dataclass(frozen=True)
class BaseState(_Stamped):
    modules: Tuple[SwerveModuleState, ...]   # [back, front_right, front_left]
    estop: bool
    connected: bool
    health: Optional[BaseHealth]       # refreshed at low rate; None until first reply
    stamp: float


# ── hands (ORCA) ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HandState(_Stamped):
    """Per-side ORCA snapshot, keyed by the driver's native ORCA joint names.
    (The URDF thumb rename is a ROS-contract concern and stays in the ROS adapter.)"""
    side: str                          # "left" | "right"
    position: Mapping[str, float]      # rad
    velocity: Mapping[str, float]      # rad/s
    effort: Mapping[str, float]        # normalized load [-1, 1]
    temperature: Mapping[str, float]   # degC
    voltage: Mapping[str, float]       # V
    torque_enabled: bool
    connected: bool
    stamp: float


# ── lift ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LiftState(_Stamped):
    height: Optional[float]            # m; None until the Pico reports
    velocity: float                    # m/s
    homed: bool                        # on-Pico calibration valid
    connected: bool
    stamp: float
    calibrated: bool = False           # floor+ceiling stall-seek done → min/max are measured
    min_pos: float = 0.0               # m; calibrated bottom-of-travel datum
    max_pos: float = 0.0               # m; calibrated top-of-travel (default until calibrated)


# ── sensors ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LidarScan(_Stamped):
    """One full revolution. Arrays are numpy at runtime (typed loosely so this
    module stays numpy-free): angles rad REP-103 CCW, ranges m, quality raw."""
    angles: Any
    ranges: Any
    quality: Any
    stamp: float


@dataclass(frozen=True)
class AudioChunk(_Stamped):
    """One capture frame from the mic, in the wire format (bytes, never numpy).

    ``pcm`` is raw little-endian S16 interleaved — the SAME bytes the :9610 plane
    and the ROS /audio/mic topic carry, so a frame is never re-encoded between the
    sound card and the operator's ear. ``seq`` counts frames since capture started
    and is the ONLY way a consumer detects a gap: audio is a SEQUENCE, not a
    snapshot, so a dropped frame is an audible click, not a stale render.
    """
    pcm: bytes                         # S16_LE interleaved, frame_ms of audio
    seq: int                           # frame counter since capture start
    rate: int                          # Hz
    channels: int
    stamp: float

    def samples(self) -> int:
        """Sample frames in this chunk (per channel)."""
        return len(self.pcm) // (2 * max(self.channels, 1))


@dataclass(frozen=True)
class AudioState(_Stamped):
    """Sound-card snapshot: presence, per-direction liveness, and mixer truth.

    ``playback_volume``/``capture_volume`` are percent as the CARD reports them —
    not what we asked for. A bar that boots its PCM control at 0% plays silence
    while every write "succeeds", so the realized value is the only honest one.
    """
    device: str                        # resolved ALSA device, "" when absent
    card: str                          # ALSA card id ("Bar")
    connected: bool                    # the card enumerated
    capturing: bool                    # capture stream open and delivering frames
    playing: bool                      # playback stream open
    muted: bool                        # capture muted BY US (not by the device)
    rate: int
    channels: int
    playback_volume: Optional[int]     # %, None when the card has no such control
    capture_volume: Optional[int]      # %
    mic_rms: float                     # 0..1 of full scale, newest capture frame
    mic_silent: bool                   # capturing, but delivering EXACT digital
                                       # silence for a sustained window — a live
                                       # mic always has a noise floor > 0, so this
                                       # means muted-at-the-device or dead hardware
                                       # (NOT muted-by-us: see `muted`). False while
                                       # not capturing (unknown, not "ok").
    stamp: float


@dataclass(frozen=True)
class CameraStreamInfo:
    """A camera as clients consume it: liveness + where its streams are served.
    Frames intentionally do NOT flow through this API — the per-process video
    plane (:8080 MJPEG, :9091/:9092 UDP, :962x dedicated) is the frame contract."""
    name: str                          # "front" | "main" | "head_left" | "wrist_left" | ...
    alive: bool                        # worker process running / stream serving
    mjpeg_url: str                     # http://<host>:8080/camera/<name>/stream.mjpg
    udp_jpeg: Optional[Tuple[str, int]]
    udp_h264: Optional[Tuple[str, int]]
    dedicated_port: Optional[int]      # 9620-9623 plane, when this camera has one


@dataclass(frozen=True)
class SensorsState(_Stamped):
    lidar: Optional[LidarScan]
    cameras: Tuple[CameraStreamInfo, ...]
    audio: Optional[AudioState]
    stamp: float


# ── the composite read() snapshot ──────────────────────────────────────────────

@dataclass(frozen=True)
class RobotState(_Stamped):
    """The one ``MabelHardware.read()`` return. A subsystem that is disabled,
    never connected, or currently degraded is simply None — the snapshot itself
    always succeeds (readers must not fail because a limb is unplugged)."""
    left_arm: Optional[JointGroupState]
    right_arm: Optional[JointGroupState]
    neck: Optional[NeckState]
    torso: Optional[JointGroupState]
    base: Optional[BaseState]
    lift: Optional[LiftState]
    left_hand: Optional[HandState]
    right_hand: Optional[HandState]
    sensors: Optional[SensorsState]
    stamp: float


# ── commands (write side) ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class JointCommand:
    """Full MIT command; position-only is the common case. Dicts keyed by joint
    name; a joint absent from ``position`` keeps its latched target."""
    position: Mapping[str, float]                                  # rad, model frame
    velocity: Mapping[str, float] = field(default_factory=dict)    # rad/s feedforward
    kp: Mapping[str, float] = field(default_factory=dict)          # per-joint gain override
    kd: Mapping[str, float] = field(default_factory=dict)
    effort: Mapping[str, float] = field(default_factory=dict)      # N·m feedforward


@dataclass(frozen=True)
class TwistCommand:
    """Base twist, REAL frame (+x fwd, +y left, yaw CCW)."""
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class GazeCommand:
    """Neck task-space target; direct-drive, so each axis maps 1:1 to its motor."""
    pitch: float                       # rad
    roll: float
    yaw: float


# A push-stream subscription's cancel function.
Unsubscribe = Callable[[], None]
