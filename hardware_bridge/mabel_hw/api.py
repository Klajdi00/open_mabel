"""api.py — MabelHardware: the ONE in-process entry to MABEL's hardware.

    from mabel_hw import MabelHardware

    with MabelHardware() as hw:                    # flock -> calibration -> connect
        state = hw.read()                          # RobotState frozen snapshot
        with hw.left_arm.claim("teleop") as arm:   # exclusive writer for the group
            arm.set_joint_positions({"l_j1": 0.3})
        hw.estop()                                 # lease-free, always allowed

READERS SCALE: read()/subscribe() only touch driver-cached snapshots — any
number of concurrent readers, never blocked by writers. WRITERS ARE ONE PER
GROUP: every mutating method hangs off a Writer you can only get from
``handle.claim()`` (mabel_hw.arbitration leases: wait/steal/ttl), so two
components interleaving commands on one actuator group is unrepresentable.
estop()/stop() bypass leases by design.

Out-of-process clients (server, ROS nodes, GUIs, apps) do NOT import this —
they keep the DDS topics / video ports / :8779 hand RPC, which the adapters in
mabel_hw.ros / mabel_hw.rpc serve on top of this same object. The bridge passes
its MabelHardware into each ROS node (build(cfg, hw=...)); the flock singleton
guarantees one device-owning process per host either way.

Pacing: inside the bridge, the ROS nodes' timers call handle.tick()/service().
A standalone (ROS-less) consumer calls ``hw.pace()`` once to start the built-in
pacing thread — without pacing, watchdog-fed drivers (Teensy/CAN) go limp.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Mapping, Optional, Sequence

from . import config as default_config
from .arbitration import GROUPS, Lease, LeaseBroker, LeaseInfo
from .calibration import CalibrationStore
from .errors import HardwareError, SubsystemUnavailable
from .types import (GazeCommand, JointCommand, RobotState, SensorsState,
                    TwistCommand, Unsubscribe, monotonic)

__all__ = ["MabelHardware", "Subscription"]

_SUBSYSTEMS = ("base", "left_arm", "right_arm", "neck", "torso",
               "hands", "lift", "lidar", "audio", "cameras")


def _cfg_get(config, name: str) -> dict:
    if isinstance(config, Mapping):
        return dict(config.get(name, {}))
    return dict(getattr(config, name, {}) or {})


# ── writers (lease-bound command surfaces) ─────────────────────────────────────

class _Writer(Lease):
    """A Lease whose methods actually command a subsystem. Every write is
    gated by broker.check() — a stolen/expired lease raises NotLeaseHolder and
    the command never reaches hardware."""

    def __init__(self, broker: LeaseBroker, lease: Lease, subsystem):
        super().__init__(broker, lease.group, lease.owner, lease.token)
        self._broker = broker
        self._sub = subsystem

    def _gate(self):
        self._broker.check(self.group, self.token)
        if self._sub is None:
            raise SubsystemUnavailable(f"{self.group} is not up")
        return self._sub


class JointGroupWriter(_Writer):
    def set_joint_positions(self, targets: Mapping[str, float]) -> None:
        """MODEL-frame radians, e.g. {'l_j1': 0.3} (partial dicts fine)."""
        self._gate().set_targets(targets)

    def command(self, cmd: JointCommand) -> None:
        self._gate().command(cmd)

    def set_torque(self, on: bool) -> None:
        self._gate().set_torque(on)


class NeckWriter(JointGroupWriter):
    def set_gaze(self, pitch: float, roll: float, yaw: float) -> None:
        """Task-space (rad) for the serial neck; each axis drives its motor 1:1,
        clamped to the calibrated _task limits (level head == 0)."""
        self._gate().set_gaze(pitch, roll, yaw)

    def command_gaze(self, cmd: GazeCommand) -> None:
        self.set_gaze(cmd.pitch, cmd.roll, cmd.yaw)


class BaseWriter(_Writer):
    def set_twist(self, vx: float, vy: float, wz: float) -> None:
        """REAL frame: +x forward, +y left, wz CCW (m/s, m/s, rad/s)."""
        sub = self._gate()
        sub.set_twist(vx, vy, wz)
        sub.tick()                                # push now — low latency

    def command(self, cmd: TwistCommand) -> None:
        self.set_twist(cmd.vx, cmd.vy, cmd.wz)


class HandWriter(_Writer):
    def set_joint_positions(self, targets: Mapping[str, float]) -> None:
        """ORCA joint names -> radians (no-op while torque is off)."""
        self._gate().set_joint_positions(targets)

    def set_torque(self, on: bool) -> None:
        self._gate().set_torque(on)


class LiftWriter(_Writer):
    def set_height(self, meters: float) -> None:
        self._gate().set_height(meters)

    def set_velocity(self, mps: float) -> None:
        self._gate().set_velocity(mps)

    def home(self, wait: bool = False) -> None:
        self._gate().home(wait=wait)

    def calibrate(self, wait: bool = False) -> None:
        self._gate().calibrate(wait=wait)

    def set_torque(self, on: bool) -> None:
        self._gate().set_torque(on)


class SpeakerWriter(_Writer):
    """The speaker is leased like an actuator even though it moves nothing: two
    components talking over each other is exactly as broken as two commanding an
    arm, and the lease is what makes 'who is speaking' answerable."""

    def play(self, pcm: bytes) -> None:
        self._gate().play(pcm)

    def set_volume(self, percent: int) -> None:
        self._gate().set_volume(percent)


# ── handles (per-subsystem views on the facade) ────────────────────────────────

class _Handle:
    _writer_cls = _Writer

    def __init__(self, hw: "MabelHardware", group: str, attr: str):
        self._hw = hw
        self._group = group
        self._attr = attr

    @property
    def _sub(self):
        return getattr(self._hw, "_" + self._attr)

    @property
    def available(self) -> bool:
        return self._sub is not None

    def connected(self) -> bool:
        sub = self._sub
        return bool(sub is not None and sub.connected())

    def claim(self, owner: str, *, wait: float = 0.0, steal: bool = False,
              ttl: Optional[float] = None):
        """The EXCLUSIVE writer for this group (context manager). One live
        writer per group; see mabel_hw.arbitration for wait/steal/ttl."""
        lease = self._hw._broker.claim(self._group, owner, wait=wait,
                                       steal=steal, ttl=ttl)
        return self._writer_cls(self._hw._broker, lease, self._sub)

    def holder(self) -> Optional[LeaseInfo]:
        return self._hw._broker.holder(self._group)

    # pacing passthroughs (bridge node timers / hw.pace())
    def tick(self) -> None:
        sub = self._sub
        if sub is not None and hasattr(sub, "tick"):
            sub.tick()

    def service(self) -> None:
        sub = self._sub
        if sub is not None and hasattr(sub, "service"):
            sub.service()


class JointGroupHandle(_Handle):
    _writer_cls = JointGroupWriter

    @property
    def names(self):
        sub = self._sub
        return tuple(sub.names) if sub is not None else ()

    def state(self):
        sub = self._sub
        return sub.state() if sub is not None else None

    def health(self) -> Dict[str, Dict[str, float]]:
        sub = self._sub
        return sub.health() if sub is not None else {}

    def on_state(self, callback) -> Unsubscribe:
        sub = self._sub
        return sub.on_state(callback) if sub is not None else (lambda: None)

    def stop(self) -> None:
        """Lease-FREE hold-in-place (never drops torque)."""
        sub = self._sub
        if sub is not None:
            sub.stop()


class ArmHandle(JointGroupHandle):
    def commandable_ranges(self) -> Dict[str, list]:
        """{joint: [lo, hi]} — the calibrated MODEL-frame range each joint may be
        commanded to (rad). Read-side, so lease-free: a teleop client clamps its
        sliders against this without claiming the arm."""
        sub = self._sub
        return sub.commandable_ranges() if sub is not None else {}


class NeckHandle(JointGroupHandle):
    _writer_cls = NeckWriter

    @property
    def task_space_available(self) -> bool:
        sub = self._sub
        return bool(sub is not None and sub.task_space_available)


class BaseHandle(_Handle):
    _writer_cls = BaseWriter

    def state(self):
        sub = self._sub
        return sub.state() if sub is not None else None

    def stop(self) -> None:
        """Lease-FREE: latch estop + zero drive NOW."""
        sub = self._sub
        if sub is not None:
            sub.stop()

    def set_estop(self, on: bool) -> None:
        """Lease-FREE estop latch (the DDS estop topic is the authority; anyone
        may set OR clear, exactly like today's contract)."""
        sub = self._sub
        if sub is not None:
            sub.set_estop(on)
            sub.tick()


class HandHandle(_Handle):
    _writer_cls = HandWriter

    def state(self):
        sub = self._sub
        return sub.state() if sub is not None else None

    def health(self) -> Dict[str, Dict[str, float]]:
        sub = self._sub
        return sub.health() if sub is not None else {}

    def stop(self) -> None:
        sub = self._sub
        if sub is not None:
            sub.stop()


class LiftHandle(_Handle):
    _writer_cls = LiftWriter

    def state(self):
        sub = self._sub
        return sub.state() if sub is not None else None

    def stop(self) -> None:
        sub = self._sub
        if sub is not None:
            sub.stop()


class SpeakerHandle(_Handle):
    """The speaker half of the sound card (the mic half is on SensorsHandle —
    one device, but capture is a read and playback is a write)."""
    _writer_cls = SpeakerWriter

    def state(self):
        sub = self._sub
        return sub.state() if sub is not None else None

    def health(self) -> Dict[str, Dict[str, float]]:
        sub = self._sub
        return sub.health() if sub is not None else {}

    def stop(self) -> None:
        sub = self._sub
        if sub is not None:
            sub.stop()


class SensorsHandle:
    """Read-only — no leases anywhere near sensors."""

    def __init__(self, hw: "MabelHardware"):
        self._hw = hw

    def lidar(self):
        sub = self._hw._lidar
        return sub.latest() if sub is not None else None

    def on_scan(self, callback) -> Unsubscribe:
        sub = self._hw._lidar
        return sub.on_scan(callback) if sub is not None else (lambda: None)

    def mic(self):
        sub = self._hw._audio
        return sub.latest() if sub is not None else None

    def on_audio(self, callback) -> Unsubscribe:
        """Subscribe to mic frames. Capture is DEMAND-DRIVEN: the sound card is
        not opened until the first subscriber and is released after the last."""
        sub = self._hw._audio
        return sub.on_chunk(callback) if sub is not None else (lambda: None)

    def audio(self):
        sub = self._hw._audio
        return sub.state() if sub is not None else None

    def set_mic_muted(self, muted: bool) -> None:
        """Mute is a property of the MIC, not a command to the robot — it stays
        on the read-side handle so a monitor can mute without taking a lease."""
        sub = self._hw._audio
        if sub is not None:
            sub.set_muted(muted)

    def cameras(self):
        suite = self._hw._cameras
        return suite.streams() if suite is not None else ()

    def camera_url(self, name: str, transport: str = "mjpeg") -> str:
        suite = self._hw._cameras
        if suite is None:
            raise SubsystemUnavailable("camera suite is not up")
        return suite.url(name, transport)

    def state(self) -> SensorsState:
        return SensorsState(lidar=self.lidar(), cameras=self.cameras(),
                            audio=self.audio(), stamp=monotonic())


class Subscription:
    def __init__(self, cancel: Callable[[], None]):
        self._cancel = cancel

    def cancel(self) -> None:
        self._cancel()


# ── the facade ─────────────────────────────────────────────────────────────────

class MabelHardware:
    """One instance per device-owning process (flock-enforced); every in-process
    consumer shares it. See the module docstring for the contract."""

    def __init__(self, config=None, *, mock: bool = False,
                 subsystems: Optional[Sequence[str]] = None,
                 calibration_root: Optional[str] = None,
                 acquire_singleton: bool = True, logger=None):
        self._config = config if config is not None else default_config
        self._mock = bool(mock)
        self._only = tuple(subsystems) if subsystems is not None else None
        self._calibration_root = calibration_root
        self._acquire = bool(acquire_singleton) and not self._mock
        self._log = logger or (lambda *a, **k: None)
        self._broker = LeaseBroker(GROUPS, logger=None)
        self._lock_fd = None
        self._open = False
        self._store: Optional[CalibrationStore] = None
        self._store_unwatch = None
        self._pacer: Optional[threading.Thread] = None
        self._pacer_stop = threading.Event()
        self._watch_stop = threading.Event()
        self._hands_controller = None
        for attr in _SUBSYSTEMS:
            setattr(self, "_" + attr, None)
        self._left_hand = self._right_hand = None

        # public handles (valid before open(); available=False until then)
        self.base = BaseHandle(self, "base", "base")
        self.left_arm = ArmHandle(self, "left_arm", "left_arm")
        self.right_arm = ArmHandle(self, "right_arm", "right_arm")
        self.neck = NeckHandle(self, "neck", "neck")
        self.torso = JointGroupHandle(self, "torso", "torso")
        self.left_hand = HandHandle(self, "left_hand", "left_hand")
        self.right_hand = HandHandle(self, "right_hand", "right_hand")
        self.lift = LiftHandle(self, "lift", "lift")
        self.speaker = SpeakerHandle(self, "speaker", "audio")
        self.sensors = SensorsHandle(self)

    # ── config / gating ───────────────────────────────────────────────────────
    def _cfg(self, name: str) -> dict:
        return _cfg_get(self._config, name)

    def _wanted(self, name: str, cfg: dict) -> bool:
        if self._only is not None:
            return name in self._only
        if self._mock:
            return True
        return bool(cfg.get("enabled", False))

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def open(self) -> "MabelHardware":
        if self._open:
            return self
        if self._acquire:
            self._lock_fd = self._acquire_flock()
        self._store = CalibrationStore(
            root=self._calibration_root or
            (self._cfg("CALIBRATION").get("root") or None)).load()
        cal_cfg = self._cfg("CALIBRATION")
        if cal_cfg.get("watch", True):
            self._store_unwatch = self._store.watch(
                self._on_calibration_change,
                period_s=float(cal_cfg.get("watch_period", 2.0)))
        for name, builder in (("base", self._build_base),
                              ("left_arm", self._build_joint_group),
                              ("right_arm", self._build_joint_group),
                              ("neck", self._build_joint_group),
                              ("torso", self._build_joint_group),
                              ("hands", self._build_hands),
                              ("lift", self._build_lift),
                              ("lidar", self._build_lidar),
                              ("audio", self._build_audio),
                              ("cameras", self._build_cameras)):
            cfg = self._cfg(name.upper())
            if not self._wanted(name, cfg):
                continue
            try:                                  # graceful degradation, always
                builder(name, cfg)
                self._log(f"[hw] {name} up")
            except Exception as e:                # noqa: BLE001
                self._log(f"[hw] {name} FAILED to come up ({e}); "
                          "staying alive without it")
        self._open = True
        self._watch_stop.clear()
        threading.Thread(target=self._presence_loop, daemon=True,
                         name="mabel-hw-presence").start()
        return self

    def close(self) -> None:
        if not self._open:
            return
        self._pacer_stop.set()
        self._watch_stop.set()
        if self._store_unwatch:
            self._store_unwatch()
        for attr in ("_base", "_left_arm", "_right_arm", "_neck", "_torso",
                     "_lift", "_lidar", "_audio"):
            sub = getattr(self, attr)
            if sub is not None:
                try:
                    sub.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._hands_controller is not None:
            try:
                self._hands_controller.close()
            except Exception:
                pass
            self._hands_controller = None
        self._left_hand = self._right_hand = None
        self._cameras = None
        if self._lock_fd is not None:
            try:
                import fcntl
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception:
                pass
            self._lock_fd = None
        self._open = False

    def __enter__(self) -> "MabelHardware":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._open

    def _acquire_flock(self):
        """Library-friendly flock: raise HardwareError on contention instead of
        the bridge entrypoint's exit-3/standby behavior."""
        import os
        from . import singleton
        fd, other = singleton._try_lock()
        if fd is None and (not other or not singleton._owner_alive(other)):
            try:
                os.unlink(singleton._LOCK_PATH)
            except OSError:
                pass
            fd, other = singleton._try_lock()
        if fd is None:
            raise HardwareError(
                f"another process owns MABEL's hardware (lock "
                f"{singleton._LOCK_PATH}, pid {other or '?'}); a second owner "
                "would double-command the devices. Pass "
                "acquire_singleton=False ONLY if that owner handed you this "
                "object.")
        return fd

    # ── subsystem builders ────────────────────────────────────────────────────
    def _build_base(self, name, cfg):
        from .subsystems.base import BaseSubsystem
        if self._mock:
            from .mock import MockSwerve
            self._base = BaseSubsystem(cfg or self._cfg("BASE"),
                                       swerve=MockSwerve(), logger=self._log)
        else:
            self._base = BaseSubsystem(cfg, logger=self._log)

    def _build_joint_group(self, name, cfg):
        from .errors import SubsystemUnavailable
        from .subsystems.arms import ArmsSubsystem
        from .subsystems.joint_group import JointGroupSubsystem
        from .subsystems.neck import NeckSubsystem
        cls = {"neck": NeckSubsystem, "left_arm": ArmsSubsystem,
               "right_arm": ArmsSubsystem}.get(name, JointGroupSubsystem)
        driver = None
        if self._mock:
            from .drivers.seam import MockJointGroupDriver
            driver = MockJointGroupDriver(cfg.get("joints") or ["j1"])
            driver.connect()
        sub = cls(name, cfg, self._cfg("LINKS"), store=self._store,
                  driver=driver, logger=self._log)
        setattr(self, "_" + name, sub)           # attach BEFORE connect: a
        if not self._mock:                       # device absent at boot still
            try:                                 # hot-joins via service()
                sub.connect()
            except SubsystemUnavailable as e:
                self._log(f"[hw] {name} device not up yet ({e}); "
                          "will hot-join when it appears")

    def _build_hands(self, name, cfg):
        from .subsystems.hands import HandSubsystem
        if self._mock:
            from .mock import MockHand
            self._left_hand = HandSubsystem("left", MockHand("left"),
                                            logger=self._log)
            self._right_hand = HandSubsystem("right", MockHand("right"),
                                             logger=self._log)
            return
        from . import paths
        from .drivers.hands import HandsController

        def _hand_cfg(side: str) -> str:
            override = cfg.get("config_dir") or ""
            if not override:
                return ""
            import os
            return os.path.join(paths.hand_config_dir(override),
                                f"{side}_hand.yaml")

        self._hands_controller = HandsController(
            left_port=cfg["left_port"], right_port=cfg["right_port"],
            baudrate=int(cfg["baudrate"]),
            auto_enable_torque=bool(cfg["auto_enable_torque"]),
            left_config=_hand_cfg("left"), right_config=_hand_cfg("right"),
            reconnect_interval=float(cfg.get("reconnect_interval", 2.0)),
            adapter_serials=cfg.get("adapter_serials") or {},
            logger=self._log)
        self._left_hand = HandSubsystem(
            "left", self._hands_controller.hand("left"), logger=self._log)
        self._right_hand = HandSubsystem(
            "right", self._hands_controller.hand("right"), logger=self._log)

    def _build_lift(self, name, cfg):
        from .subsystems.lift import LiftSubsystem
        driver = None
        if self._mock:
            from .mock import MockLiftDriver
            driver = MockLiftDriver(max_pos=float(cfg.get("max_pos", 0.5)))
        self._lift = LiftSubsystem(cfg, driver=driver, logger=self._log)
        self._lift.connect()

    def _build_lidar(self, name, cfg):
        if self._mock:
            from .mock import MockLidar
            self._lidar = MockLidar()
            return
        from .subsystems.lidar import LidarSubsystem
        self._lidar = LidarSubsystem(cfg, logger=self._log)

    def _build_audio(self, name, cfg):
        if self._mock:
            from .mock import MockAudio
            self._audio = MockAudio()
            return
        from .subsystems.audio import AudioSubsystem
        self._audio = AudioSubsystem(cfg, logger=self._log)

    def _build_cameras(self, name, cfg):
        from .subsystems.cameras import CameraSuite
        self._cameras = CameraSuite(cfg, logger=self._log)

    # ── calibration hot reload ────────────────────────────────────────────────
    def _on_calibration_change(self, logical: str) -> None:
        self._log(f"[hw] calibration changed ({logical}); re-deriving maps")
        for attr in ("_left_arm", "_right_arm", "_neck", "_torso"):
            sub = getattr(self, attr)
            if sub is not None:
                try:
                    sub.recalibrate(self._store)
                except Exception as e:            # noqa: BLE001
                    self._log(f"[hw] recalibrate {attr[1:]} failed: {e}")

    @property
    def calibration(self) -> Optional[CalibrationStore]:
        return self._store

    # ── presence: who is on the bus right now ─────────────────────────────────
    def presence(self) -> Dict[str, bool]:
        """{subsystem: connected} for every subsystem that was brought up —
        the live who's-here map (the watcher below logs the transitions)."""
        out: Dict[str, bool] = {}
        for name in ("base", "left_arm", "right_arm", "neck", "torso", "lift"):
            handle = getattr(self, name)
            if handle.available:
                out[name] = handle.connected()
        for name in ("left_hand", "right_hand"):
            sub = getattr(self, "_" + name)
            if sub is not None:
                out[name] = bool(sub.connected())
        if self._lidar is not None:
            out["lidar"] = bool(self._lidar.connected())
        if self._audio is not None:
            out["audio"] = bool(self._audio.connected())
        return out

    def _presence_loop(self):
        """1 Hz connection watcher: announces, per subsystem, the moment a
        device LEAVES the system (unplug, power loss, link death) and the
        moment it REJOINS (every driver auto-reconnects on its own; this is the
        unified detection/announcement layer on top)."""
        seen = self.presence()                   # seed silently (open() logged)
        while not self._watch_stop.wait(1.0):
            now = self.presence()
            for name, up in now.items():
                was = seen.get(name)
                if was is None or up == was:
                    continue
                if up:
                    self._log(f"[hw] ++ {name} REJOINED the system")
                else:
                    self._log(f"[hw] -- {name} LEFT the system (device "
                              "unplugged/silent) — auto-reconnect armed")
            seen = now

    # ── read side (many simultaneous readers) ─────────────────────────────────
    def read(self) -> RobotState:
        """Composite non-blocking snapshot; a degraded subsystem is None, the
        snapshot itself ALWAYS succeeds."""
        def grab(sub, *, method="state"):
            if sub is None:
                return None
            try:
                return getattr(sub, method)()
            except Exception:                     # noqa: BLE001 — never fail read
                return None

        sensors = None
        if (self._lidar is not None or self._cameras is not None
                or self._audio is not None):
            try:
                sensors = self.sensors.state()
            except Exception:                     # noqa: BLE001
                sensors = None
        return RobotState(
            left_arm=grab(self._left_arm), right_arm=grab(self._right_arm),
            neck=grab(self._neck), torso=grab(self._torso),
            base=grab(self._base), lift=grab(self._lift),
            left_hand=grab(self._left_hand), right_hand=grab(self._right_hand),
            sensors=sensors, stamp=monotonic())

    def subscribe(self, callback: Callable[[RobotState], None], *,
                  rate_hz: float = 50.0) -> Subscription:
        """Push snapshots at rate_hz on a dedicated thread. Any number of
        independent subscriptions; cancel() stops one."""
        stop = threading.Event()

        def _loop():
            period = 1.0 / max(rate_hz, 0.1)
            while not stop.wait(period):
                try:
                    callback(self.read())
                except Exception as e:            # noqa: BLE001
                    self._log(f"[hw] subscriber failed: {e}")

        threading.Thread(target=_loop, daemon=True,
                         name="mabel-hw-subscribe").start()
        return Subscription(stop.set)

    # ── arbitration / safety ──────────────────────────────────────────────────
    def claims(self) -> Dict[str, Optional[LeaseInfo]]:
        return self._broker.snapshot()

    def estop(self) -> None:
        """GLOBAL stop, NEVER lease-gated: base zero-drive latch, joint groups
        + hands hold-in-place, lift halt."""
        for handle in (self.base, self.left_arm, self.right_arm, self.neck,
                       self.torso, self.left_hand, self.right_hand, self.lift):
            try:
                handle.stop()
            except Exception:                     # noqa: BLE001 — stop everything
                pass

    def clear_estop(self) -> None:
        self.base.set_estop(False)

    # ── standalone pacing (ROS-less consumers) ────────────────────────────────
    def pace(self, stream_hz: float = 50.0, service_hz: float = 1.0) -> None:
        """Start the built-in pacing thread: tick() every actuator subsystem at
        stream_hz (feeds firmware watchdogs) and service() at service_hz.
        The ROS bridge does NOT call this — its node timers pace instead."""
        if self._pacer is not None and self._pacer.is_alive():
            return
        self._pacer_stop.clear()

        def _loop():
            period = 1.0 / max(stream_hz, 1.0)
            every = max(1, int(stream_hz / max(service_hz, 0.1)))
            n = 0
            handles = (self.base, self.left_arm, self.right_arm, self.neck,
                       self.torso, self.lift)
            while not self._pacer_stop.wait(period):
                n += 1
                for h in handles:
                    try:
                        h.tick()
                        if n % every == 0:
                            h.service()
                    except Exception:             # noqa: BLE001
                        pass
                if n % every == 0 and self._hands_controller is not None:
                    try:
                        self._hands_controller.service()
                    except Exception:             # noqa: BLE001
                        pass

        self._pacer = threading.Thread(target=_loop, daemon=True,
                                       name="mabel-hw-pace")
        self._pacer.start()
