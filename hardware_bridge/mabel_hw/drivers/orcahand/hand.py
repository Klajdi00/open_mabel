"""OrcaHand -- the joint-space hardware interface for the MABEL ORCA hand.

This is the one class controllers, learned policies, and the ROS node all use.
It speaks SI units (radians, rad/s) and joint names; everything below it (tick
conversion, calibration clamping, serial framing) is hidden.

Quick start
-----------
::

    from orcahand import OrcaHand

    with OrcaHand("right") as hand:          # auto-detects the USB port
        hand.set_joint_positions({"index_mcp": 0.5, "thumb_cmc": 0.1})  # radians
        state = hand.get_state()
        print(state.position["index_mcp"], state.temperature.hottest())

Design contract
---------------
* All angles are **radians**; all velocities **rad/s**.
* Inputs are dicts ``{joint_name: angle}`` (partial dicts are fine -- only the
  named joints move) or arrays ordered by :attr:`joint_names`.
* Every command is clamped to the calibrated tick range -- a bad target can
  never drive a tendon past its hardware end-stop.
* ``connect`` / ``disconnect`` are idempotent; ``disconnect`` always disables
  torque first. The class is a context manager.
* Thread-safe: the underlying :class:`~orcahand.bus.FeetechBus` serialises every
  transaction, so a feedback thread and a command thread can share one hand.

For development without hardware use :meth:`OrcaHand.mock`, which swaps in an
in-memory bus exposing the identical API.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Union

from . import constants as C
from .bus import FeetechBus
from .config import HandConfig
from .poses import POSES
from .state import HandState

JointTargets = Union[Dict[str, float], Sequence[float]]


class OrcaHand:
    """High-level joint-space interface to one physical ORCA hand."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        config_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        port: Optional[str] = None,
        baudrate: int = C.DEFAULT_BAUDRATE,
        torque_limit: int = C.DEFAULT_TORQUE_LIMIT,
        goal_speed: int = C.DEFAULT_GOAL_SPEED,
        goal_acc: int = C.DEFAULT_GOAL_ACC,
        bus: Optional[FeetechBus] = None,
        use_model_calib: bool = True,
    ):
        """Create a hand interface (does not open the port until ``connect``).

        Parameters
        ----------
        model
            ``"left"`` or ``"right"`` -- loads the packaged canonical config.
            Ignore and pass ``config_path`` to use a project-specific config.
        config_path, calibration_path
            explicit config / calibration files (e.g. a ROS share dir).
        port
            serial device; ``None`` auto-detects the Feetech USB adapter.
        torque_limit, goal_speed, goal_acc
            motion defaults written to every motor on connect.
        bus
            inject a custom transport (used by :meth:`mock`); normally ``None``.
        """
        self.config = HandConfig.load(
            model=model, config_path=config_path, calibration_path=calibration_path,
            use_model_calib=use_model_calib)
        self._by_name = self.config.by_name()
        self._by_id = self.config.by_id()
        self.torque_limit = torque_limit
        self.goal_speed = goal_speed
        self.goal_acc = goal_acc

        self._bus = bus if bus is not None else FeetechBus(port=port, baudrate=baudrate)
        self._connected = False
        self._torque_on = False
        self._active_ids: set = set(self.config.motor_ids)

    # ----------------------------------------------------------- introspection
    @property
    def side(self) -> str:
        return self.config.side

    @property
    def joint_names(self) -> List[str]:
        """Canonical joint order for this hand."""
        return self.config.joint_names

    @property
    def motor_ids(self) -> List[int]:
        return self.config.motor_ids

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque_on

    @property
    def bus(self) -> FeetechBus:
        return self._bus

    # ----------------------------------------------------------------- mock
    @classmethod
    def mock(cls, model: str = "right", **kwargs) -> "OrcaHand":
        """An OrcaHand backed by an in-memory bus -- no hardware required.

        The mock uses the PACKAGED tick config (``use_model_calib=False``): there
        is no live studio in a mock context, and this keeps hardware-free tests
        deterministic regardless of what ``mabel.json`` currently holds.
        """
        from .mock import MockFeetechBus
        kwargs.setdefault("use_model_calib", False)
        cfg = HandConfig.load(model=model,
                              config_path=kwargs.get("config_path"),
                              calibration_path=kwargs.get("calibration_path"),
                              use_model_calib=kwargs["use_model_calib"])
        kwargs.pop("config_path", None)
        kwargs.pop("calibration_path", None)
        kwargs.pop("port", None)
        kwargs.pop("bus", None)
        return cls(model=model, bus=MockFeetechBus(cfg.motor_ids), **kwargs)

    # --------------------------------------------------------------- lifecycle
    def connect(self, enable_torque: bool = True, scan: bool = True) -> "OrcaHand":
        """Open the bus, push motion defaults, optionally enable torque.

        If ``scan`` is True, motors that do not answer a ping are dropped from
        the active set (commands/reads skip them) and their names are returned
        via :attr:`missing_joints` -- the hand still runs on whatever responds.
        """
        self._bus.open()

        if scan:
            found = set(self._bus.scan(range(1, 21)))
            self._active_ids = {mid for mid in self.config.motor_ids if mid in found}
        else:
            self._active_ids = set(self.config.motor_ids)

        self._bus.set_torque_limit(self.torque_limit)
        self._bus.set_goal_acc(self.goal_acc)
        self._bus.set_goal_speed(self.goal_speed)

        self._connected = True
        if enable_torque:
            self.enable_torque()
        return self

    def disconnect(self) -> None:
        """Disable torque and close the bus. Idempotent and exception-safe."""
        try:
            if self._bus.is_open:
                self.disable_torque()
                time.sleep(0.05)
        finally:
            self._bus.close()
            self._connected = False

    def __enter__(self) -> "OrcaHand":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.disconnect()

    @property
    def missing_joints(self) -> List[str]:
        """Configured joints whose motor did not answer the connect-time scan."""
        return [j.name for j in self.config.joints if j.motor_id not in self._active_ids]

    # ------------------------------------------------------------------ torque
    def enable_torque(self, joints: Optional[Sequence[str]] = None) -> None:
        self._set_torque(True, joints)

    def disable_torque(self, joints: Optional[Sequence[str]] = None) -> None:
        self._set_torque(False, joints)

    def _set_torque(self, on: bool, joints: Optional[Sequence[str]]) -> None:
        if on:
            # SAFETY — hold-in-place: the servos drive to their last GOAL
            # register the instant torque comes on, which snaps the hand to a
            # stale target. Latch goal := PRESENT position for every joint
            # being energized first, so enabling torque simply freezes the
            # hand where it physically is.
            try:
                ids = ([self._by_name[n].motor_id for n in joints] if joints
                       else list(self._active_ids))
                present = self._bus.read_positions(ids)
                if present:
                    self._bus.sync_write_positions(present)
            except Exception:
                pass                    # never block energizing on a read hiccup
        if joints is None:
            self._bus.set_torque_enable(on)            # broadcast
            self._torque_on = on
        else:
            for name in joints:
                self._bus.set_torque_enable(on, self._by_name[name].motor_id)

    def verify_torque(self, expected: bool) -> List[str]:
        """Read TORQUE_ENABLE back from every active motor; return the joint
        names whose register DISAGREES with ``expected`` (or that no-answered).

        Broadcast writes carry no status packet, so ``enable/disable_torque``
        alone cannot know a packet was dropped — this is the ground truth."""
        want = 1 if expected else 0
        bad: List[str] = []
        for name, j in self._by_name.items():
            if j.motor_id not in self._active_ids:
                continue
            try:
                v = self._bus.read_register(j.motor_id, C.ADDR_TORQUE_ENABLE, 1)
            except Exception:
                v = None
            if v is None or int(v) != want:
                bad.append(name)
        return bad

    def disable_torque_verified(self, attempts: int = 3) -> List[str]:
        """Disable torque and CONFIRM it landed, motor by motor.

        The plain broadcast disable is fire-and-forget: one dropped TTL packet
        left the servos energised while the cached flag — and therefore
        /torque_state, the app's indicator, and the gateway's torque reconcile —
        all said "off" (the 2026-07-18 'E-STOP but the hand is still stiff'
        bug). Retries per-motor for anything still hot; returns the joint names
        that STILL refuse ([] = confirmed limp). The cached flag afterwards
        reflects the BUS truth, so a failed disable keeps reporting torque ON
        and upstream watchdogs keep re-sending instead of trusting a lie."""
        bad: List[str] = []
        for _ in range(max(1, attempts)):
            self._bus.set_torque_enable(False)         # broadcast first (fast path)
            bad = self.verify_torque(False)
            if not bad:
                self._torque_on = False
                return []
            for name in bad:                           # targeted per-motor retry
                try:
                    self._bus.set_torque_enable(False, self._by_name[name].motor_id)
                except Exception:
                    pass
        bad = self.verify_torque(False)
        self._torque_on = bool(bad)                    # truth, not intent
        return bad

    # --------------------------------------------------------------- commanding
    def set_joint_positions(self, targets: JointTargets) -> None:
        """Command joint angles in **radians**.

        ``targets`` is either a ``{joint_name: angle}`` dict (partial is fine)
        or a full sequence ordered by :attr:`joint_names`. Each angle is clamped
        to its joint range and mapped to a calibrated tick before a single
        SYNC_WRITE drives the whole hand.
        """
        angle_by_name = self._coerce(targets)
        id_to_raw: Dict[int, int] = {}
        for name, angle in angle_by_name.items():
            j = self._by_name.get(name)
            if j is None or j.motor_id not in self._active_ids:
                continue
            id_to_raw[j.motor_id] = j.angle_to_raw(float(angle))
        if id_to_raw:
            self._bus.sync_write_positions(id_to_raw)

    def set_joint_percent(self, targets: Dict[str, float]) -> None:
        """Command joints as 0-100 % of their calibrated range (GUI/demo helper)."""
        id_to_raw = {}
        for name, pct in targets.items():
            j = self._by_name.get(name)
            if j is None or j.motor_id not in self._active_ids:
                continue
            id_to_raw[j.motor_id] = j.percent_to_raw(float(pct))
        if id_to_raw:
            self._bus.sync_write_positions(id_to_raw)

    def set_named_position(self, name: str) -> None:
        """Drive to a named pose from :data:`orcahand.poses.POSES`."""
        if name not in POSES:
            raise KeyError(f"unknown pose {name!r}; options: {sorted(POSES)}")
        pose = {jn: pct for jn, pct in POSES[name].items() if jn in self._by_name}
        self.set_joint_percent(pose)

    @property
    def pose_names(self) -> List[str]:
        return sorted(POSES)

    # ----------------------------------------------------------------- feedback
    def get_joint_positions(self, as_array: bool = False) -> Union[Dict[str, float], list]:
        """Present joint angles in **radians** (dict by name, or ordered array)."""
        raw = self._bus.read_positions(sorted(self._active_ids))
        out: Dict[str, float] = {}
        for j in self.config.joints:
            if j.motor_id in raw:
                out[j.name] = j.raw_to_angle(raw[j.motor_id])
        if as_array:
            return [out.get(n, float("nan")) for n in self.joint_names]
        return out

    def get_state(self) -> HandState:
        """A full synchronized snapshot: pos/vel/effort/torque/temp/voltage + ticks.

        `effort` is the servo's normalised PWM duty (legacy); `current` (A) and
        `torque` (N·m) are the MEASURED load, present only when the servo serves
        its Present_Current register (see bus.read_feedback)."""
        pos, vel, eff, temp, volt, rawp, cur, tau = {}, {}, {}, {}, {}, {}, {}, {}
        nm_per_a = getattr(self.config, "torque_constant_nm_per_a",
                           None) or C.TORQUE_NM_PER_AMP
        for j in self.config.joints:
            if j.motor_id not in self._active_ids:
                continue
            fb = self._bus.read_feedback(j.motor_id)
            if fb is None:
                continue
            rawp[j.name] = fb["position"]
            pos[j.name] = j.raw_to_angle(fb["position"])
            vel[j.name] = j.speed_to_rad_s(fb["speed"])
            eff[j.name] = j.load_to_effort(fb["load"])
            temp[j.name] = fb["temperature"]
            volt[j.name] = fb["voltage"]
            amps = fb.get("current")
            if amps is not None:                    # absent ⇒ leave BOTH empty
                cur[j.name] = amps
                tau[j.name] = j.current_to_torque(amps, nm_per_a)
        return HandState(position=pos, velocity=vel, effort=eff,
                         temperature=temp, voltage=volt, raw_position=rawp,
                         current=cur, torque=tau, stamp=time.time())

    # ----------------------------------------------------------------- helpers
    def _coerce(self, targets: JointTargets) -> Dict[str, float]:
        if isinstance(targets, dict):
            return targets
        names = self.joint_names
        seq = list(targets)
        if len(seq) != len(names):
            raise ValueError(
                f"array of {len(seq)} positions but hand has {len(names)} joints; "
                f"pass a dict for partial commands")
        return dict(zip(names, seq))
