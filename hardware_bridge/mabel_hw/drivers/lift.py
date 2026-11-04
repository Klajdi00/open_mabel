"""lift.py — the bridge's lift driver: a typed wrapper over the proven
``firmware/lift/host`` library (Pico + BTS7960 standing-desk column).

The Pico runs the real control (trapezoidal position + PI velocity, on-device
calibration persisted to its flash) — this driver only speaks the host API and
owns the sys.path bootstrap that finds it, so no other bridge module needs to
know where the firmware checkout lives. Units are METRES / m·s⁻¹, up-positive.

Graceful degradation: a missing checkout or absent Pico leaves the driver
alive-but-disconnected (``connect()`` raises SubsystemUnavailable for the
caller to log; ``read()`` then reports connected=False forever until a
``service()`` retry succeeds).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from .. import paths
from ..errors import SubsystemUnavailable
from ..types import LiftState, monotonic

__all__ = ["LiftDriver"]

# Hard cap on how long the firmware Lift constructor may take. It opens the Pico
# serial AND soft-reboots the board (Ctrl-C / Ctrl-D + ~2 s of sleeps + a telemetry
# handshake). A WEDGED USB-CDC device — one that opens but never drains/answers —
# makes that handshake (or serial.flush()/tcdrain) block INDEFINITELY. connect()
# runs inside the HAL's SYNCHRONOUS hw.open(), so an unbounded block hangs the ENTIRE
# bridge bring-up: no ROS adapter nodes are ever built, so /arms/joint_states, the
# cameras, and every other subsystem never come up. Bounding it keeps the documented
# "idle-safe" contract even when a Pico is PRESENT but hung. Override via env.
_CONNECT_TIMEOUT_S = float(os.getenv("MABEL_HW_LIFT_CONNECT_TIMEOUT", "8.0"))

# The lift firmware runs on a Raspberry Pi Pico (RP2040), USB vendor id 0x2E8A.
# We locate it by vendor id rather than a fixed /dev/ttyACMx because ttyACM
# numbering is arrival-order — the Pico is NOT reliably ACM0 (on the current hub
# ACM0 is the base Teensy, vendor 0x16C0), so a hard-coded default silently
# pointed the lift at the wrong board. The Teensy (0x16C0) and hand adapters
# (0x1A86) have different vendor ids, so this filter can never seize their buses.
_PICO_VID = 0x2E8A
_PICO_CDC_PID = 0x0005              # MicroPython/Pico USB-CDC product id (preferred)
_AUTO = "auto"

# The Pico only streams telemetry after it receives "json on" — which the firmware
# host sends in a short window right after the soft-reboot at connect. If those
# enables are lost (main.py not ready yet, or the reader thread starved during the
# bridge's heavy multi-subsystem startup) the stream never turns on, get_lift_height()
# stays None forever, and the DDS node publishes nothing while still looking
# "connected". service() detects that mute-but-open state and re-kicks "json on"
# for this many 1 Hz cycles before escalating to a full reconnect (fresh soft-reboot).
_MUTE_REKICK_CYCLES = 3


def _import_lift_class():
    """Resolve firmware/lift/host's ``Lift`` (metres API). Import is deferred so
    a checkout without firmware/ can still import mabel_hw."""
    if not paths.add_lift_host():
        raise SubsystemUnavailable(
            "firmware/lift/host not found under MABEL root "
            f"({paths.mabel_root()}) — set MABEL_ROOT or fetch the firmware tree")
    from lift import Lift                                       # noqa: PLC0415
    return Lift


class LiftDriver:
    """Owns the Pico serial link. Thread-safe; one instance per bridge."""

    def __init__(self, port: str = _AUTO, baud: int = 115200,
                 max_pos: float = 0.5, logger=None):
        self._port = port
        self._baud = baud
        self._max_pos = max_pos
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.Lock()
        self._lift = None                                        # firmware Lift
        self._resolved_port: Optional[str] = None                # last port we opened
        self._mute_cycles = 0                                    # consecutive no-telemetry service() ticks

    # ── device resolution (by USB vendor id, so replug/renumber is transparent) ─
    def _resolve_port(self) -> Optional[str]:
        """Locate the Pico. An explicit, existing device path (a
        ``/dev/serial/by-id`` symlink or a real ttyACMx) always wins; otherwise
        the board is found by USB vendor id, so re-enumeration to a different
        ttyACM number after a replug is transparent. Returns None when no Pico is
        present (the caller then leaves the lift idle and retries via service())."""
        p = self._port
        if p and p != _AUTO and os.path.exists(p):
            return p                                             # explicit pin wins
        try:
            import serial.tools.list_ports as lp                # noqa: PLC0415
            cands = [c for c in lp.comports() if c.vid == _PICO_VID]
            # Prefer the MicroPython CDC PID if several RP boards are attached.
            cands.sort(key=lambda c: 0 if c.pid == _PICO_CDC_PID else 1)
            if cands:
                return cands[0].device
        except Exception:                                        # noqa: BLE001
            pass
        # Nothing matched by vendor id; keep an explicit-but-absent path for a
        # later hot-plug at that exact node, else report absent.
        return None if (not p or p == _AUTO) else p

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        lift_cls = _import_lift_class()           # raises SubsystemUnavailable if absent
        with self._lock:
            if self._lift is not None:
                return
            # Resolve the Pico afresh on every attempt so a replug that lands on a
            # different ttyACM number is picked up by the next service() retry.
            port = self._resolve_port()
            if port is None:
                raise SubsystemUnavailable(
                    f"no lift Pico found (no USB vendor 0x{_PICO_VID:04x} device "
                    f"present; configured port {self._port!r}) — leaving lift idle; "
                    "service() will retry")
            # Build the firmware Lift in a WATCHED daemon thread with a hard timeout so a
            # hung Pico handshake can never hang hw.open() (see _CONNECT_TIMEOUT_S). On
            # timeout the lift stays idle and a later service() retry reconnects it.
            box: dict = {}

            def _build():
                try:
                    box["lift"] = lift_cls(port, self._baud,
                                           max_pos=self._max_pos)
                except Exception as e:            # noqa: BLE001
                    box["err"] = e

            t = threading.Thread(target=_build, name="lift-connect", daemon=True)
            t.start()
            t.join(_CONNECT_TIMEOUT_S)
            if t.is_alive():
                raise SubsystemUnavailable(
                    f"lift Pico on {port} did not initialise within "
                    f"{_CONNECT_TIMEOUT_S:.0f}s (present but not responding) — leaving "
                    "lift idle; service() will retry")
            if "err" in box:
                raise SubsystemUnavailable(
                    f"lift Pico not reachable on {port}: {box['err']}"
                ) from box["err"]
            self._lift = box["lift"]
            self._resolved_port = port
            self._mute_cycles = 0
        # Serial opened. Report telemetry health HONESTLY — a Pico whose stream
        # never started is "connected" but publishes nothing, so don't claim a
        # clean link when it is mute (service() will re-kick it).
        if self._telemetry_live(box["lift"]):
            self._log(f"[lift] connected on {port} (telemetry live)")
        else:
            self._log(f"[lift] connected on {port} but telemetry NOT streaming yet "
                      "— service() will re-send 'json on'")

    def close(self) -> None:
        with self._lock:
            lift, self._lift = self._lift, None
        if lift is not None:
            try:
                lift.set_neutral()               # free-wheel, matching the
            except Exception:                    # pre-refactor shutdown path
                pass
            try:
                lift.close()
            except Exception:
                pass

    def connected(self) -> bool:
        return self._lift is not None

    # ── telemetry health ──────────────────────────────────────────────────────
    @staticmethod
    def _telemetry_live(lift) -> bool:
        """True iff the Pico is actually STREAMING (a telemetry frame is readable).
        A freshly-opened serial port with no JSON stream reads height=None."""
        try:
            return lift.get_lift_height() is not None
        except Exception:                                        # noqa: BLE001
            return False

    @staticmethod
    def _kick_telemetry(lift) -> None:
        """Re-send 'json on' to (re)start the Pico's telemetry stream. Reaches the
        firmware host's LiftController.enable_telemetry via getattr so it no-ops on
        test fakes / wrappers that don't expose it."""
        ctl = getattr(lift, "_c", None)
        fn = getattr(ctl, "enable_telemetry", None)
        if callable(fn):
            try:
                fn(attempts=3, wait=0.2)
            except Exception:                                    # noqa: BLE001
                pass

    def service(self) -> None:
        """Slow-path watchdog (call ~1 Hz): reconnect an absent Pico, and revive a
        connected-but-mute one whose telemetry stream never started."""
        if self._lift is None:
            self._mute_cycles = 0
            try:
                self.connect()
            except SubsystemUnavailable:
                pass
            return

        # Serial is open — is the Pico actually streaming? If not, re-kick 'json on';
        # escalate to a full reconnect (fresh soft-reboot) if it stays mute. Without
        # this the DDS node would publish nothing forever while reporting connected.
        lift = self._lift
        if self._telemetry_live(lift):
            self._mute_cycles = 0
            return
        self._mute_cycles += 1
        if self._mute_cycles == 1:
            self._log("[lift] connected but no telemetry — re-sending 'json on'")
        if self._mute_cycles <= _MUTE_REKICK_CYCLES:
            self._kick_telemetry(lift)
        else:
            self._log("[lift] telemetry still mute — reconnecting (soft-reboot)")
            self._mute_cycles = 0
            self.close()
            try:
                self.connect()
            except SubsystemUnavailable:
                pass

    # ── read ──────────────────────────────────────────────────────────────────
    def read(self) -> LiftState:
        lift = self._lift
        if lift is None:
            return LiftState(height=None, velocity=0.0, homed=False,
                             connected=False, stamp=monotonic())
        try:
            return LiftState(height=lift.get_lift_height(),
                             velocity=lift.get_velocity(),
                             homed=bool(lift.homed),
                             connected=True, stamp=monotonic(),
                             calibrated=bool(lift.homed),
                             min_pos=float(lift.min_pos),
                             max_pos=float(lift.max_pos))
        except Exception as e:
            self._log(f"[lift] telemetry failed ({e}); dropping link for retry")
            self.close()
            return LiftState(height=None, velocity=0.0, homed=False,
                             connected=False, stamp=monotonic())

    # ── write (lease-gated at the subsystem layer) ────────────────────────────
    def _require(self):
        lift = self._lift
        if lift is None:
            raise SubsystemUnavailable("lift is not connected")
        return lift

    def _write(self, fn):
        """Run a Pico WRITE, dropping the link on a serial I/O error so service()
        reconnects. The Pico's USB-CDC can re-enumerate (a soft-reset renumbers
        /dev/ttyACMx) or wedge mid-write → OSError [Errno 5]; because the telemetry
        health-check only READS, a write-only failure would otherwise loop forever
        writing to a dead handle ("lift move failed: [Errno 5]" every command). Closing
        here makes the next service() re-resolve the port BY VID (finding the new ACM
        number) and soft-reboot into a fresh link — self-healing across a re-enumeration."""
        try:
            fn(self._require())
        except SubsystemUnavailable:
            raise
        except OSError as e:                         # serial I/O (Errno 5), device gone, ...
            self._log(f"[lift] write failed ({e}); dropping link — service() will reconnect")
            self.close()
            raise

    def set_height(self, meters: float) -> None:
        """Absolute target, non-blocking (the Pico runs the profile)."""
        self._write(lambda lift: lift.lift_to_height(float(meters), wait=False))

    def set_velocity(self, mps: float) -> None:
        self._write(lambda lift: lift.set_velocity_control(float(mps)))

    def home(self, wait: bool = False) -> None:
        """Stall-seek the floor (on-Pico). wait=True blocks until homed."""
        self._write(lambda lift: lift.lift_home(wait=wait))

    def calibrate(self, wait: bool = False) -> None:
        """Stall-seek floor THEN ceiling (on-Pico); measures + persists the true
        min/max travel envelope. wait=True blocks until it settles."""
        self._write(lambda lift: lift.calibrate(wait=wait))

    def stop(self) -> None:
        """Halt motion NOW. Lease-free at the facade (safety path)."""
        lift = self._lift
        if lift is not None:
            try:
                lift.lift_stop()
            except Exception:
                pass

    def set_torque(self, on: bool) -> None:
        self._write(lambda lift: lift.set_torque(bool(on)))
