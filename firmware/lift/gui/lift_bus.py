"""LiftBus — GUI-facing backend around the host LiftController.

Holds a plain-data LiftState that a background thread keeps fresh from the
Pico's telemetry stream. The GUI reads bus.state on its own Tk `after` loop
(so no Tk widget is ever touched from a worker thread) and calls the command
methods here, which forward to the serial LiftController.
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Make the host library importable (../host/lift_controller.py).
_HERE = Path(__file__).resolve().parent
_HOST = _HERE.parent / "host"
if str(_HOST) not in sys.path:
    sys.path.insert(0, str(_HOST))

from lift_controller import LiftController  # noqa: E402


@dataclass
class LiftState:
    connected: bool = False
    has_telemetry: bool = False
    height: float = 0.0
    vel: float = 0.0
    vset: float = 0.0
    effort: float = 0.0
    count: int = 0
    mode: int = 0
    status: int = 0
    mode_name: str = "—"
    status_name: str = "—"
    calibrated: bool = False
    min_height: float = 0.0
    max_height: float = 0.0
    torque: bool = True


class LiftBus:
    def __init__(self):
        self.state = LiftState()
        self.port = None
        self._ctrl = None
        self._log = []
        self._running = True
        self._t = threading.Thread(target=self._poll_loop, daemon=True)
        self._t.start()

    # ----------------------------------------------------------- logging
    def log(self, msg):
        stamp = time.strftime("%H:%M:%S")
        self._log.append(f"{stamp}  {msg}")
        if len(self._log) > 200:
            self._log = self._log[-200:]

    def log_tail(self, n=25):
        return self._log[-n:]

    # -------------------------------------------------------- connection
    def connect(self, port):
        """Blocking; call from a worker thread. Returns (ok, message)."""
        try:
            ctrl = LiftController(port, on_log=self.log)   # firmware -> event log
        except Exception as exc:  # noqa: BLE001
            self.log(f"Connect failed: {exc}")
            return False, str(exc)
        self._ctrl = ctrl
        self.port = port
        self.state.connected = True
        if getattr(ctrl, "telemetry_ok", False):
            self.log(f"Connected to {port}")
            return True, "ok"
        self.log(f"Connected to {port} (no telemetry — is main.py running?)")
        return True, "connected, but no telemetry"

    def disconnect(self):
        if self._ctrl is not None:
            try:
                self._ctrl.close()
            except Exception:  # noqa: BLE001
                pass
            self._ctrl = None
        self.state.connected = False
        self.state.has_telemetry = False
        self.log("Disconnected")

    # ------------------------------------------------------------ poll
    def _poll_loop(self):
        prev_mode = prev_status = prev_cal = None
        tick = 0
        while self._running:
            ctrl = self._ctrl
            if ctrl is not None:
                t = ctrl.telemetry()
                if t is not None:
                    s = self.state
                    s.has_telemetry = True
                    s.height = t.height
                    s.vel = t.vel
                    s.vset = t.vset
                    s.effort = t.effort
                    s.count = t.count
                    s.mode = t.mode
                    s.status = t.status
                    s.mode_name = t.mode_name
                    s.status_name = t.status_name
                    s.calibrated = t.calibrated
                    s.min_height = t.min_height
                    s.max_height = t.max_height
                    s.torque = t.torque

                    # Log notable state transitions.
                    if t.mode_name != prev_mode:
                        self.log(f"Mode → {t.mode_name}")
                        prev_mode = t.mode_name
                    if t.status_name != prev_status:
                        self.log(f"Status → {t.status_name}")
                        prev_status = t.status_name
                    if t.calibrated != prev_cal:
                        if prev_cal is not None:
                            self.log("Calibrated" if t.calibrated
                                     else "Calibration cleared")
                        prev_cal = t.calibrated
                    # Heartbeat of live status every ~2 s.
                    tick += 1
                    if tick % 40 == 0:
                        self.log(f"h={t.height:.1f}mm v={t.vel:+.1f} "
                                 f"cnt={t.count} {t.status_name}")
            time.sleep(0.05)

    # --------------------------------------------------------- commands
    def _send(self, fn, label):
        if self._ctrl is None:
            self.log(f"Not connected ({label})")
            return
        try:
            fn(self._ctrl)
        except Exception as exc:  # noqa: BLE001
            self.log(f"{label} failed: {exc}")

    def stop(self):
        self._send(lambda c: c.stop(), "stop")
        self.log("STOP")

    def set_torque(self, on):
        self._send(lambda c: c.set_torque(on), "torque")
        self.log("Torque " + ("ON" if on else "OFF"))

    def jog(self, effort):
        self._send(lambda c: c.drive(effort), "jog")

    def set_velocity(self, mm_s):
        self._send(lambda c: c.set_velocity(mm_s), "velocity")
        self.log(f"Velocity hold {mm_s:+.0f} mm/s")

    def move_to(self, height_mm, v_max=None):
        self._send(lambda c: c.move_to(height_mm, v_max), "move")
        self.log(f"Move to {height_mm:.0f} mm" +
                 (f" @ {v_max:.0f} mm/s" if v_max else ""))

    def calibrate(self):
        self._send(lambda c: c.calibrate(), "calibrate")
        self.log("Calibrating…")

    def home(self):
        self._send(lambda c: c.home(), "home")
        self.log("Homing…")

    def set_home(self):
        self._send(lambda c: c.set_home(), "set home")
        self.log("Home set at current position")

    def park(self):
        self._send(lambda c: c.park(), "park")
        self.log("Park")

    def set_park(self, mm):
        self._send(lambda c: c.set_park(mm), "setpark")
        self.log(f"Park height = {mm:.0f} mm")

    def zero(self):
        self._send(lambda c: c.zero(), "zero")
        self.log("Zeroed here")

    def set_height(self, mm):
        self._send(lambda c: c.set_height(mm), "set height")
        self.log(f"Height set to {mm:.0f} mm")

    def set_home_height(self, mm):
        self._send(lambda c: c.send(f"homeheight {mm:.3f}"), "homeheight")
        self.log(f"Home height = {mm:.0f} mm")

    def enable_limits(self, enabled):
        self._send(lambda c: c.enable_limits(enabled), "limits")
        self.log("Soft limits " + ("ON" if enabled else "OFF"))

    def shutdown(self):
        self._running = False
        self.disconnect()
