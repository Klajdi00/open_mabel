"""hands_controller.py — high-level owner of the MABEL ORCA hands.

This is the hands analogue of ``base/swerve_base.py``: the SAME validated stack the
firmware (``firmware/orca_hand/feetech``) uses — joint-name + radian API on top of the
``orcahand.OrcaHand`` HAL — with the CLI/GUI front-end replaced by ``set_positions()``.
So the wire path that reaches the Feetech motors is identical whether a firmware script,
ROS, or the server drives it; that code-path identity is the safety guarantee.

The hardware bridge owns this object; nothing else opens the TTL serial bus. It is
deliberately ROS-free so it can be driven standalone (a CLI test, calibration) before
the ROS stack is up. ONE controller owns BOTH hands — one ``OrcaHand`` per side, each on
its own USB serial adapter. A side whose port is "none" (or whose hardware is absent)
stays idle; the bridge never crashes on a missing hand.

Safety / liveness (mirrors SwerveBase):
  * ``connect`` opens each bus, scans for motors, pushes motion defaults, optionally
    enables torque. Motors that do not answer are dropped (the hand runs on the rest).
  * Every command is clamped to the calibrated tick range inside ``OrcaHand`` — a bad
    target can never drive a tendon past its hardware end-stop.
  * ``service`` is a LOW-rate watchdog that auto-reconnects a side whose adapter was
    absent at startup or dropped mid-run (cable re-plug). Kept off the fast command path.
  * ``close`` disables torque and releases the bus (free-drive, motors de-energised).
"""

import threading
import time
from typing import Dict, List, Optional

from .orcahand import OrcaHand, detect_side_port


class Hand:
    """One physical ORCA hand: the OrcaHand HAL + (re)connect policy.

    ROS-free. Holds its construction parameters so it can rebuild a fresh OrcaHand on
    reconnect, which reloads the calibration from the SETUP STUDIO master
    ``mabel.json`` (``hand/orca-<side>/calib``) — the single source of truth shared
    with every teleop device and the studio's 3D viewport (see ``orcahand/config.py``).
    The legacy ``<side>_calibration.json`` tick files are a per-joint fallback only.

    Calibration is NOT done here: it is a separate, ROS-free bench step run with the
    bridge stopped (the setup studio, ``web_control_gui/setup_studio``), which owns
    the bus exclusively and writes ``mabel.json``; this object reloads it on the next
    (re)connect. The running bridge only ever commands already-calibrated angles.
    """

    def __init__(self, side: str, *, port: Optional[str], baudrate: int,
                 config_path: Optional[str] = None, auto_enable_torque: bool = True,
                 reconnect_interval: float = 2.0, stale_reads: int = 25,
                 adapter_serials: Optional[Dict[str, str]] = None, logger=None):
        self.side = side
        self.requested_port = port
        self.adapter_serials = dict(adapter_serials or {})
        self.baudrate = int(baudrate)
        self.config_path = config_path
        self.auto_enable_torque = bool(auto_enable_torque)
        self.reconnect_interval = float(reconnect_interval)
        # Retry period when the ADAPTER answers but ZERO motors do (hand unpowered):
        # each probe costs ~0.5 s of serial timeouts, so retry slowly (see _connect).
        self._dead_retry = max(10.0, self.reconnect_interval)
        self.stale_reads = int(stale_reads)
        self._log = logger or (lambda *a, **k: None)

        self.hand: Optional[OrcaHand] = None
        # SWAPPABLE HANDS: "v1" = MABEL's own ORCA hand, "v2" = NOVA's, decided per (re)connect by which
        # registered adapter is plugged in (adapter_serials "<side>" vs "<side>_v2"); each brings its own
        # wiring + calibration (<side>_hand.yaml vs <side>_v2_hand.yaml, hand/orca-<side> vs hand/orca-v2-<side>).
        self.generation = "v1"
        self._want_torque = bool(auto_enable_torque)
        self._busy = threading.Lock()      # serializes command vs feedback bus access
        self._miss = 0                     # consecutive empty feedback reads
        self._last_reconnect = -1e9
        self._link_up = False
        self._idle_reason = None           # dedupe: log an idle cause once, not per retry
        self._connect()

    # ---- lifecycle ----------------------------------------------------------
    def _new_hand(self, port: Optional[str]) -> OrcaHand:
        """Build a fresh OrcaHand from the stored config (reloads calibration)."""
        if self.config_path:
            return OrcaHand(config_path=self.config_path,
                            port=port, baudrate=self.baudrate)
        model = f"{self.side}_v2" if self.generation == "v2" else self.side
        return OrcaHand(model=model, port=port, baudrate=self.baudrate)

    def _resolve_port(self) -> Optional[str]:
        """Requested port -> concrete device. "auto"/"" resolves SIDE-AWARELY through
        the adapter-serial registry (never first-come-first-served across sides);
        None means no adapter for this side is present right now (idle + retry)."""
        req = (self.requested_port or "").strip()
        if req.lower() not in ("", "auto"):
            self.generation = "v1"
            return req                      # explicit device path: use as-is
        port = detect_side_port(self.side, self.adapter_serials, log=self._say_idle)
        if port is not None:
            self.generation = "v1"
            return port
        v2 = (self.adapter_serials.get(f"{self.side}_v2") or "").strip()
        if v2:                              # NOVA's v2 hand on this side?
            import serial.tools.list_ports as _lp
            hit = next((p for p in _lp.comports() if (p.serial_number or "") == v2), None)
            if hit is not None:
                self.generation = "v2"
                return hit.device
        return None

    def _say_idle(self, msg: str) -> None:
        """Log an idle/detect diagnostic ONCE until it changes (the 2 s reconnect
        watchdog re-runs detection forever; per-retry logging would flood the journal)."""
        if msg != self._idle_reason:
            self._idle_reason = msg
            self._log(msg)

    def _connect(self) -> None:
        """(Re)open the bus. Non-fatal: a missing/unpowered hand leaves the side idle
        and is retried by ``service`` when the adapter returns."""
        self._last_reconnect = time.monotonic()
        try:
            if self.hand is not None:
                self.hand.disconnect()
        except Exception:
            pass
        self.hand = None
        if (self.requested_port or "").lower() in ("none", "off", "disabled"):
            self._say_idle(f"[{self.side}] disabled (port='{self.requested_port}'); idle.")
            return
        port = self._resolve_port()
        if port is None:
            self._say_idle(f"[{self.side}] no adapter detected for this side; "
                           "idle, hot-plug will be picked up.")
            return
        try:
            hand = self._new_hand(port)
            # Always connect LIMP (enable_torque=False) so opening the bus never snaps the
            # hand to a stale goal register. Torque, if wanted, is engaged below via the
            # hold-in-place path. A bridge bring-up thus commands/energises nothing unless
            # auto_enable_torque was explicitly set.
            hand.connect(enable_torque=False, scan=True)
        except Exception as e:
            self.hand = None
            self._say_idle(f"[{self.side}] hardware not available ({e}); idle, will retry.")
            return
        if hand.missing_joints and len(hand.missing_joints) >= len(hand.joint_names):
            # The ADAPTER answered (bus opened) but ZERO motors respond — the hand
            # itself is unpowered/disconnected. Holding this dead bus open is a
            # TARPIT: the 50 Hz feedback streamer and the 1 Hz watchdog then spend
            # essentially all of the hands node inside per-joint serial timeouts,
            # starving the OTHER hand's feedback/commands to zero (hit live
            # 2026-07-11: the mute LEFT hand froze the working RIGHT hand). Treat
            # the hand as absent: release the port, retry slowly; a powered-on
            # hand is picked up on the next probe.
            dead_port = getattr(hand.bus, "port", "?")
            try:
                hand.disconnect()
            except Exception:
                pass
            self.hand = None
            self._say_idle(f"[{self.side}] adapter {dead_port} present but NO motors "
                           f"respond (hand unpowered?); idle, probing every "
                           f"{self._dead_retry:.0f} s.")
            # Push the next service() retry out to the slow dead-hand cadence.
            self._last_reconnect = (time.monotonic()
                                    + self._dead_retry - self.reconnect_interval)
            return
        self.hand = hand
        self._miss = 0
        self._link_up = True
        self._idle_reason = None           # connected: a future idle logs its cause again
        self.side = hand.side          # config may rename (e.g. model autodetect)
        calib = hand.config.calibration_path
        self._log(f"[{self.side}] ORCA {self.generation} hand · bus open on {hand.bus.port} "
                  f"({len(hand.joint_names)} joints; "
                  f"calib: {calib if calib else 'NONE — full-range fallback'})")
        if hand.missing_joints:
            self._log(f"[{self.side}] NOT responding: {hand.missing_joints}")
        if self._want_torque:          # only if explicitly requested; holds in place
            with self._busy:
                try:
                    self._hold_then_enable()
                except Exception as e:
                    self._log(f"[{self.side}] torque enable failed: {e}")

    def service(self) -> None:
        """Low-rate link watchdog: retry a side that is idle (absent at startup or
        dropped mid-run). Call from a slow timer, never the fast command path."""
        if self.hand is None:
            if (self.requested_port or "").lower() not in ("none", "off", "disabled") \
                    and time.monotonic() - self._last_reconnect >= self.reconnect_interval:
                self._connect()
            return
        # Connected but the bus has gone quiet (unplugged): drop and let it reopen.
        if self._miss >= self.stale_reads:
            if self._link_up:
                self._log(f"[{self.side}] link LOST (no feedback); retrying...")
            self._link_up = False
            self._connect()

    def close(self) -> None:
        if self.hand is not None:
            try:
                self.hand.disconnect()         # disables torque, then closes the bus
            except Exception:
                pass
            self.hand = None

    # ---- command intake -----------------------------------------------------
    def set_positions(self, targets: Dict[str, float]) -> None:
        """Command joint angles [rad] by (bare or side-prefixed) name. No-op while
        free-drive (torque off) — position targets are ignored then."""
        if self.hand is None or not targets:
            return
        if not self.hand.is_torque_enabled:
            return
        prefix = f"{self.side}_"
        bare = {(n[len(prefix):] if n.startswith(prefix) else n): a
                for n, a in targets.items()}
        with self._busy:
            try:
                self.hand.set_joint_positions(bare)
            except Exception as e:
                self._log(f"[{self.side}] set_positions failed: {e}")

    def set_torque(self, on: bool) -> None:
        self._want_torque = bool(on)
        if self.hand is None:
            return
        # ENABLE may trust the cached flag (re-enabling is never urgent).
        # DISABLE must NOT: the flag flips optimistically on an unACKed
        # broadcast, so after one dropped packet the guard returned without
        # touching the bus while the servos stayed energised — and every retry
        # (the gateway's 2 s torque reconcile included) hit the same guard.
        # "E-STOP but the hand is still stiff", 2026-07-18. A disable is a
        # stop-class command: always send it, verified when the driver can.
        if on and bool(self.hand.is_torque_enabled):
            return
        with self._busy:
            try:
                if on:
                    self._hold_then_enable()
                else:
                    verified = getattr(self.hand, "disable_torque_verified", None)
                    if verified is None:
                        self.hand.disable_torque()
                    else:
                        still_hot = verified()
                        if still_hot:
                            # Loud + truthful: torque_state stays ON for these,
                            # so upstream keeps re-sending the disable.
                            self._log(f"[{self.side}] TORQUE DISABLE UNCONFIRMED "
                                      f"on {still_hot} — servos may still be "
                                      f"energised; will keep retrying")
            except Exception as e:
                self._log(f"[{self.side}] torque toggle failed: {e}")

    def _hold_then_enable(self) -> None:
        """Engage torque WITHOUT a snap. A Feetech servo, once energised, drives to its
        Goal-Position register — which on bring-up holds a STALE value, so a naive
        enable_torque makes the hand lurch (looks like an unasked-for command). We first
        latch each responding motor's goal to its PRESENT position, so energising holds
        in place. Caller holds ``self._busy``; the bus has its own lock."""
        try:
            present = self.hand.get_joint_positions()      # {name: rad}, responders only
            if present:
                self.hand.set_joint_positions(present)     # goal := present (no motion)
        except Exception as e:
            self._log(f"[{self.side}] pre-enable hold failed: {e}")
        self.hand.enable_torque()

    # ---- feedback -----------------------------------------------------------
    def is_torque_enabled(self) -> bool:
        return bool(self.hand is not None and self.hand.is_torque_enabled)

    def get_state(self):
        """A full HandState snapshot, or None if idle / no data this cycle.

        Tracks consecutive empty reads so ``service`` can detect an unplugged hand.
        """
        if self.hand is None:
            return None
        with self._busy:
            try:
                state = self.hand.get_state()
            except Exception as e:
                self._log(f"[{self.side}] read failed: {e}")
                self._miss += 1
                return None
        if not state.position:
            self._miss += 1
            return None
        self._miss = 0
        return state

    @property
    def joint_names(self) -> List[str]:
        return self.hand.joint_names if self.hand is not None else []


class HandsController:
    """Owns BOTH ORCA hands (left + right). The single owner of the TTL serial buses."""

    def __init__(self, *, left_port: str, right_port: str, baudrate: int = 1_000_000,
                 auto_enable_torque: bool = True, left_config: Optional[str] = None,
                 right_config: Optional[str] = None, reconnect_interval: float = 2.0,
                 adapter_serials: Optional[Dict[str, str]] = None, logger=None):
        self._log = logger or (lambda *a, **k: None)
        self.hands: Dict[str, Hand] = {
            "left": Hand("left", port=left_port, baudrate=baudrate, config_path=left_config,
                         auto_enable_torque=auto_enable_torque,
                         reconnect_interval=reconnect_interval,
                         adapter_serials=adapter_serials, logger=logger),
            "right": Hand("right", port=right_port, baudrate=baudrate, config_path=right_config,
                          auto_enable_torque=auto_enable_torque,
                          reconnect_interval=reconnect_interval,
                          adapter_serials=adapter_serials, logger=logger),
        }

    def hand(self, side: str) -> Optional[Hand]:
        return self.hands.get(side)

    def service(self) -> None:
        for h in self.hands.values():
            h.service()

    def close(self) -> None:
        for h in self.hands.values():
            h.close()
