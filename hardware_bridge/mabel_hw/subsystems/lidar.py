"""lidar.py — RPLIDAR capture as a typed, ROS-free subsystem.

EXTRACTED from the proven lidar node (the capture session, the motor-stall
escalation ladder, and the USB-brownout detector move here verbatim; the ROS
node keeps only LaserScan binning/publishing on top of ``on_scan``). Single
owner of the lidar serial port; consumers get REP-103 revolutions:

    scan = lidar.latest()          # types.LidarScan | None
    unsub = lidar.on_scan(cb)      # push per revolution

Angles are already converted here (RPLIDAR raw angles grow CLOCKWISE from the
top; REP-103 wants CCW about +z: theta = -raw + yaw_offset, wrapped to [0,2pi))
so every consumer — ROS, the facade, the data plane — sees one frame convention.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from ..types import LidarScan, monotonic

__all__ = ["LidarSubsystem"]


class LidarSubsystem:
    def __init__(self, cfg: dict, *, logger=None, autostart: bool = True):
        self._cfg = cfg
        self._log = logger or (lambda *a, **k: None)
        self._stop = threading.Event()
        self._latest: Optional[LidarScan] = None
        self._cbs: list = []
        self._power_suspects = 0
        self._stalls = 0                 # consecutive motor-stall sessions
        self._thread = None
        if autostart:
            self.connect()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mabel-hw-lidar")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def connected(self) -> bool:
        s = self._latest
        return s is not None and s.age() < 2.0

    # ── read ──────────────────────────────────────────────────────────────────
    def latest(self) -> Optional[LidarScan]:
        return self._latest

    def on_scan(self, callback):
        self._cbs.append(callback)
        return lambda: self._cbs.remove(callback)

    # ── device resolution (verbatim policy from the node) ─────────────────────
    def _resolve(self):
        import os
        p = self._cfg["port"]
        if p and os.path.exists(p):
            return p
        # Fallback: any CP210x (VID 0x10C4) — the lidar kit's adapter chip. The
        # hands' CH340s are VID 0x1A86, so this can never open a hand bus, unlike
        # a blind /dev/ttyUSB0 (ttyUSB numbering is arrival-order and /dev/ttyUSB0
        # IS a hand adapter whenever the lidar enumerates second).
        try:
            import serial.tools.list_ports as lp
            for c in lp.comports():
                if c.vid == 0x10C4:
                    return c.device
        except Exception:
            pass
        return None

    # ── capture thread ────────────────────────────────────────────────────────
    def _run(self):
        from ..drivers.rplidar import RPLidarError
        while not self._stop.is_set():
            try:
                self._session()
            except RPLidarError as e:
                self._log(f"lidar error ({e}); retrying in 5 s.")
                self._stop.wait(5.0)
            except Exception as e:
                self._log(f"lidar error ({e}); retrying in 5 s.")
                self._stop.wait(5.0)

    def _session(self):
        from ..drivers.rplidar import RPLidar, RPLidarError
        path = self._resolve()
        if path is None:
            self._stop.wait(2.0)                 # absent; poll for hot-plug
            return
        lid = RPLidar(path, baud=int(self._cfg["baud"]))
        started = time.monotonic()
        try:
            lid.stop_scan()
            info = lid.get_info()
            status, err = lid.get_health()
            if status == 2:                      # protective stop
                self._log(f"lidar health ERROR ({err:#06x}); sending reset.")
                lid.reset()
                lid.stop_scan()
            # Motor-stall escalation LADDER: comms fine but zero samples means the
            # ROTOR never turned. Cycle three strategies until one spins it:
            #   1  protocol RESET + 100%-duty breakaway kick (PWM path)
            #   2  + DTR DEASSERTED: classic Slamtec serial adapters drive MOTOCTL
            #      from DTR (deassert = motor ON) and IGNORE the A5 F0 PWM command;
            #      pyserial's open leaves DTR asserted = motor held OFF on those.
            #   3  + RTS deasserted too (some adapter revisions use RTS).
            # A line flip that kills serial makes the comms probe below raise —
            # the session retries and the fresh port open restores the lines.
            level = 0 if self._stalls == 0 else ((self._stalls - 1) % 3) + 1
            if level >= 2:
                self._log(f"lidar motor stalled {self._stalls}x — deasserting DTR"
                          f"{' + RTS' if level >= 3 else ''} (adapter MOTOCTL line).")
                lid.ser.dtr = False
                if level >= 3:
                    lid.ser.rts = False
                time.sleep(2.0)
                lid.get_health()                 # comms probe (raises if dead)
            if level >= 1:
                self._log(f"lidar motor stalled {self._stalls}x — RESET + "
                          "full-duty breakaway kick before this attempt.")
                lid.reset()
                lid.stop_scan()
            lid.start_motor(int(self._cfg["motor_pwm"]), kick=level >= 1)
            lid.start_scan()
            self._log(f"RPLIDAR model {info['model']:#04x} fw {info['fw']} "
                      f"streaming on {path} @ {self._cfg['baud']}.")
            self._power_suspects = 0
            last = time.monotonic()
            for angles, dists, qual in lid.iter_scans():
                if self._stop.is_set():
                    return
                self._stalls = 0                 # samples flow: the rotor turns
                now = time.monotonic()
                self._emit(angles, dists, qual, scan_time=now - last)
                last = now
        except RPLidarError as e:
            if "stalled" in str(e):
                self._stalls += 1
            # Serial went dark within seconds of engaging the motor: the classic
            # USB-power brownout of this kit (the motor surge collapses VBUS and
            # the lidar MCU wedges until a physical replug). Say so — retrying
            # cannot fix a power problem.
            if time.monotonic() - started < 15.0:
                self._power_suspects += 1
                if self._power_suspects >= 2:
                    self._log(
                        "lidar died right after motor start (twice) — likely USB "
                        "POWER brownout. Move the adapter to a powered hub / root "
                        "port and replug it; retrying is futile until then.")
            raise
        finally:
            try:
                lid.close()
            except Exception:
                pass

    # ── revolution -> LidarScan ───────────────────────────────────────────────
    def _emit(self, angles_deg, dists_m, qual, scan_time):
        # CW sensor angles -> CCW REP-103 about +z, plus the mounting yaw offset.
        theta = (-np.radians(angles_deg)
                 + float(self._cfg["yaw_offset"])) % (2.0 * math.pi)
        scan = LidarScan(angles=theta, ranges=np.asarray(dists_m),
                         quality=np.asarray(qual), stamp=monotonic())
        # stash the scan_time for LaserScan assembly without widening the type
        object.__setattr__(scan, "_scan_time", float(scan_time))
        self._latest = scan
        for cb in list(self._cbs):
            try:
                cb(scan)
            except Exception as e:               # a consumer never kills capture
                self._log(f"lidar on_scan callback failed: {e}")
