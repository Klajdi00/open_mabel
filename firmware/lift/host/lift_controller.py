"""Host-side driver for the Pico lift firmware.

Wraps the serial command/telemetry protocol in a small, importable API so the
rest of MABEL (or a quick script) can command the lift without speaking raw
serial. Mirrors the firmware's protocol.py command set.

Example
-------
    from lift_controller import LiftController

    with LiftController("/dev/ttyACM0") as lift:
        lift.calibrate(wait=True)
        lift.move_to(150.0, v_max=40.0, wait=True)
        print(lift.telemetry())
        lift.set_velocity(20.0)   # creep up at 20 mm/s
        lift.stop()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

import serial  # pyserial


@dataclass
class Telemetry:
    status: int
    mode: int
    height: float
    vel: float
    vset: float
    effort: float
    count: int
    calibrated: bool
    max_height: float
    min_height: float = 0.0
    mm_per_count: float = 0.0
    torque: bool = True

    STATUS_NAMES = {0: "READY", 1: "MAX", 2: "MIN", 3: "MOVING"}
    MODE_NAMES = {
        0: "IDLE", 1: "MANUAL", 2: "VELOCITY", 3: "POSITION",
        4: "CALIB_DOWN", 5: "CALIB_UP", 6: "HOME",
    }

    @classmethod
    def from_dict(cls, d: dict) -> "Telemetry":
        return cls(
            status=int(d["status"]), mode=int(d["mode"]),
            height=float(d["height"]), vel=float(d["vel"]),
            vset=float(d["vset"]), effort=float(d["effort"]),
            count=int(d["count"]), calibrated=bool(d["calibrated"]),
            max_height=float(d["max_height"]),
            min_height=float(d.get("min_height", 0.0)),
            mm_per_count=float(d.get("mm_per_count", 0.0)),
            torque=bool(d.get("torque", True)),
        )

    @property
    def status_name(self) -> str:
        return self.STATUS_NAMES.get(self.status, "?")

    @property
    def mode_name(self) -> str:
        return self.MODE_NAMES.get(self.mode, "?")


class LiftController:
    """Serial wrapper around the Pico lift firmware."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0,
                 reset_on_connect: bool = True, on_log=None):
        self._ser = serial.Serial(port, baudrate, timeout=timeout)
        self._lock = threading.Lock()
        self._latest: Optional[Telemetry] = None
        self._reader: Optional[threading.Thread] = None
        self._running = False
        self.on_log = on_log          # called with firmware log messages
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        # Make sure main.py is actually running (the board may be sitting at
        # the bare REPL). A soft reboot starts it cleanly, like mpremote does.
        if reset_on_connect:
            self._restart_firmware()
        self._start_reader()
        # Returns False if no telemetry comes back (firmware not running).
        self.telemetry_ok = self.enable_telemetry(attempts=10, wait=0.4)

    def _restart_firmware(self) -> None:
        """Interrupt anything running and soft-reboot so main.py runs.

        Ctrl-C drops a running script to the REPL (our main.py disables the
        motor in its finally block first); Ctrl-D then soft-reboots, which
        re-runs main.py from a clean state.
        """
        try:
            self._ser.write(b"\r\x03\x03")   # Ctrl-C twice: interrupt
            self._ser.flush()
            time.sleep(0.2)
            self._ser.write(b"\x04")          # Ctrl-D: soft reboot
            self._ser.flush()
            time.sleep(1.3)                    # let it reboot + import modules
            self._ser.reset_input_buffer()
        except Exception:
            pass

    # ------------------------------------------------------------- plumbing
    def _start_reader(self) -> None:
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._ser.readline().decode("utf-8", "ignore").strip()
            except (serial.SerialException, OSError):
                break
            if not raw or not raw.startswith("{"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # A firmware log event, not a telemetry frame.
            if "log" in data:
                if self.on_log:
                    try:
                        self.on_log(str(data["log"]))
                    except Exception:
                        pass
                continue
            try:
                self._latest = Telemetry.from_dict(data)
            except (KeyError, ValueError):
                continue

    def send(self, command: str) -> None:
        """Send a raw protocol command line."""
        with self._lock:
            self._ser.write((command.strip() + "\n").encode("utf-8"))
            self._ser.flush()

    def enable_telemetry(self, attempts: int = 5, wait: float = 0.4) -> bool:
        """Turn on the JSON stream, retrying until frames actually arrive.

        Returns True once telemetry is flowing, False if the firmware never
        answers (e.g. main.py isn't running on the Pico).
        """
        for _ in range(attempts):
            self.send("json on")
            deadline = time.time() + wait
            while time.time() < deadline:
                if self._latest is not None:
                    return True
                time.sleep(0.05)
        return False

    # ------------------------------------------------------------- commands
    def stop(self) -> None:
        self.send("stop")

    def set_torque(self, on: bool) -> None:
        """Enable/disable the motor driver (free-wheels when off)."""
        self.send("torque on" if on else "torque off")

    def set_velocity(self, mm_s: float) -> None:
        """Closed-loop velocity hold (signed: + up / - down)."""
        self.send(f"vel {mm_s:.3f}")

    def move_to(self, height_mm: float, v_max: Optional[float] = None,
                wait: bool = False, timeout: float = 30.0) -> None:
        """Trapezoidal move to an absolute height in mm."""
        cmd = f"pos {height_mm:.3f}"
        if v_max is not None:
            cmd += f" {v_max:.3f}"
        self.send(cmd)
        if wait:
            self.wait_until_idle(timeout)

    def drive(self, effort: float) -> None:
        """Open-loop manual drive, effort in [-1, 1] (+ up / - down)."""
        if effort >= 0:
            self.send(f"up {effort:.3f}")
        else:
            self.send(f"down {abs(effort):.3f}")

    def calibrate(self, wait: bool = False, timeout: float = 60.0) -> None:
        self.send("calibrate")
        if wait:
            self.wait_until_idle(timeout)

    def home(self, wait: bool = False, timeout: float = 60.0) -> None:
        self.send("home")
        if wait:
            self.wait_until_idle(timeout)

    def set_home(self) -> None:
        """Treat the current position as the home/bottom datum (no motion)."""
        self.send("sethome")

    def park(self) -> None:
        self.send("park")

    def set_park(self, height_mm: float) -> None:
        self.send(f"setpark {height_mm:.3f}")

    def zero(self) -> None:
        self.send("zero")

    def set_height(self, height_mm: float) -> None:
        self.send(f"set {height_mm:.3f}")

    def enable_limits(self, enabled: bool = True) -> None:
        self.send("limit on" if enabled else "limit off")

    # ------------------------------------------------------------- feedback
    def telemetry(self) -> Optional[Telemetry]:
        """Most recent telemetry frame, or None if nothing parsed yet."""
        return self._latest

    def wait_until_idle(self, timeout: float = 30.0,
                        poll: float = 0.05) -> bool:
        """Block until the lift reports it is no longer MOVING.

        Returns True if it settled, False on timeout.
        """
        deadline = time.time() + timeout
        # Let the new command register first so we don't read a stale READY.
        time.sleep(0.15)
        while time.time() < deadline:
            t = self._latest
            if t is not None and t.status != 3:  # 3 == MOVING
                return True
            time.sleep(poll)
        return False

    # -------------------------------------------------------------- context
    def close(self) -> None:
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        try:
            self.send("stop")
            self.send("json off")
        except Exception:
            pass
        self._ser.close()

    def __enter__(self) -> "LiftController":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
