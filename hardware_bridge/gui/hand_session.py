"""hand_session.py - a ROS-free, GUI-owned direct connection to ONE ORCA hand.

The Control Center is normally a pure DDS client of the hardware bridge, but end-stop
calibration is NOT a ROS action: it opens the Feetech serial bus *directly*, exactly
like the standalone firmware GUI. This session is that direct path. It wraps one
``OrcaHand`` (the SAME vendored HAL the bridge uses), so the wire path to the motors is
identical - only the front-end differs.

Because the bus is opened exclusively, the bridge's hand subsystem for this side MUST be
stopped first (otherwise ``connect`` fails with the port busy). The session:

  * opens/closes the bus for a chosen side + port (``"" `` = auto-detect, ``mock`` = no HW),
  * polls raw tick positions on a background thread for the live per-joint view,
  * runs Tension / Auto / Select-finger / Manual routines on a worker thread, pushing
    fine-grained progress to a callback (the panel marshals it onto the Tk thread),
  * writes the measured limits to the side's ``<side>_calibration.yaml`` (the very file
    the bridge HAL loads as its clamp + tick<->radian scale), then reloads so the live
    clamps update immediately.

Nothing here imports rclpy.
"""

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # hardware_bridge/

from mabel_hw.drivers.orcahand import OrcaHand
from mabel_hw.drivers.orcahand.calibrate import (CalibrationParams, Calibrator, Term,
                                      _ordered_joints, load_prev, write_calibration)
from mabel_hw.drivers.orcahand.calibration import dump_mapping, read_mapping

# Low-level sweep tunables (shared by Tension + Manual; Auto uses CalibrationParams).
STEP_RAW = 5             # raw ticks nudged per step
STEP_DELAY = 0.02        # s between steps (~50 Hz)
NUM_STABLE = 10          # rolling buffer length for stall detection
STALL_ATOL = 7           # raw window that counts as "stalled"
TENSION_TORQUE = 400     # finger torque ceiling during tension
TENSION_SPEED = 150      # goal speed during tension
POLL_HZ = 8              # background live-position poll rate


class _UiTerm(Term):
    """A :class:`Term` whose progress lines are mirrored to a GUI callback.

    Subclasses the real Term (so every formatting/colour method the Calibrator may call
    still exists) with colour OFF, and forwards ``step``/``info`` to ``emit`` so the
    panel can show "Step k/n ..." live without us re-implementing the sweep.
    """

    def __init__(self, emit: Callable[[dict], None]):
        super().__init__(color=False)
        self._emit = emit

    def step(self, idx, total, label):
        self._emit({"type": "step", "idx": idx, "total": total, "label": label})
        return super().step(idx, total, label)

    def info(self, s):
        self._emit({"type": "log", "msg": str(s)})
        return super().info(s)


class HandSession:
    """Direct, single-hand HAL connection for the calibration tab."""

    def __init__(self, log: Optional[Callable[[str], None]] = None):
        self._log = log or (lambda *_: None)
        self.hand: Optional[OrcaHand] = None
        self.side = "left"
        self.port = ""
        self.joint_id: Dict[str, int] = {}
        self.id_joint: Dict[int, str] = {}
        self.wrist_id: Optional[int] = None
        self.missing: List[str] = []

        self._lock = threading.RLock()       # serialises every bus transaction
        self._worker_active = False          # pause the poll while a routine drives the bus
        self._positions: Dict[int, int] = {}  # latest raw ticks per motor id (live view)
        self._poll_stop: Optional[threading.Event] = None
        self._poll_thread: Optional[threading.Thread] = None

    # ── connection ───────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self.hand is not None

    @property
    def motor_ids(self) -> List[int]:
        return sorted(self.id_joint)

    @property
    def calib_path(self) -> Optional[Path]:
        if self.hand is None:
            return None
        cfg = self.hand.config
        if cfg.calibration_path is not None:
            return Path(cfg.calibration_path)
        if cfg.config_path is not None:                 # default next to the yaml
            return Path(cfg.config_path).parent / f"{self.side}_calibration.yaml"
        return None

    def connect(self, side: str, port: str) -> None:
        """Open the bus for ``side``. Raises on failure (port busy / no hardware)."""
        self.disconnect()
        side = "right" if str(side).lower().startswith("r") else "left"
        port = (port or "").strip()
        if port.lower() in ("mock", "sim"):
            hand = OrcaHand.mock(side)
        else:
            hand = OrcaHand(model=side, port=port or None)
        hand.connect(enable_torque=False, scan=True)
        self.hand = hand
        self.side = hand.side
        self.port = port
        self.joint_id = {j.name: j.motor_id for j in hand.config.joints}
        self.id_joint = {mid: name for name, mid in self.joint_id.items()}
        self.wrist_id = self.joint_id.get("wrist")
        self.missing = list(hand.missing_joints)
        self._positions = {}
        self._start_poll()
        self._log(f"[{self.side}] connected on {hand.bus.port} "
                  f"({len(self.joint_id) - len(self.missing)}/{len(self.joint_id)} motors)")

    def disconnect(self) -> None:
        self._stop_poll()
        if self.hand is not None:
            try:
                self.hand.disconnect()         # disables torque, closes bus
            except Exception:
                pass
        self.hand = None
        self._positions = {}

    # ── live position poll ─────────────────────────────────────────────────--
    def _start_poll(self):
        self._poll_stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_poll(self):
        if self._poll_stop is not None:
            self._poll_stop.set()
        self._poll_thread = None

    def _poll_loop(self):
        ids = self.motor_ids
        period = 1.0 / POLL_HZ
        while self._poll_stop is not None and not self._poll_stop.is_set():
            if not self._worker_active and self.hand is not None and ids:
                try:
                    with self._lock:
                        pos = self.hand.bus.read_positions(ids)
                    if pos:
                        self._positions.update(pos)
                except Exception:
                    pass
            time.sleep(period)

    def positions(self) -> Dict[int, int]:
        """Latest raw ticks per motor id (thread-safe snapshot)."""
        return dict(self._positions)

    # ── helpers ────────────────────────────────────────────────────────────--
    def _read_positions(self, ids: List[int]) -> Dict[int, int]:
        with self._lock:
            pos = self.hand.bus.read_positions(ids)
        self._positions.update(pos)
        return pos

    def set_all_torque(self, on: bool) -> None:
        with self._lock:
            if on:
                self.hand.enable_torque()
            else:
                self.hand.disable_torque()

    def calibration_view(self) -> Dict[int, dict]:
        """Current saved {motor_id: {min,max,center,range}} (for the manual coverage view)."""
        path = self.calib_path
        if path and path.exists():
            try:
                return {int(k): v for k, v in read_mapping(path).items()}
            except Exception:
                pass
        return {}

    def _reload(self) -> None:
        """Rebuild the OrcaHand so freshly-saved limits become the live clamp."""
        side, port = self.side, self.port
        self.disconnect()
        self.connect(side, port)

    # ── routines (run on a worker thread; ``emit`` is called from that thread) ──
    def run_tension(self, stop: threading.Event, emit: Callable[[dict], None]) -> None:
        """Drive every finger motor into its hard-stop and HOLD, so the operator can
        take up tendon slack on the slack side. Wrist is skipped. Releases on stop."""
        ids = [m for m in self.motor_ids if m != self.wrist_id]
        self._worker_active = True
        holding: Dict[int, int] = {}
        try:
            with self._lock:
                for m in ids:
                    self.hand.bus.set_torque_limit(TENSION_TORQUE, m)
                    self.hand.bus.set_goal_speed(TENSION_SPEED, m)
                    self.hand.bus.set_torque_enable(True, m)
            time.sleep(0.1)
            targets = self._read_positions(ids)
            for m in ids:
                targets.setdefault(m, 2047)
            buffers = {m: deque(maxlen=NUM_STABLE) for m in ids}
            stalled: set = set()
            emit({"type": "tension", "total": len(ids), "stalled": 0})
            while not stop.is_set() and len(stalled) < len(ids):
                write = dict(holding)
                for m in ids:
                    if m in stalled:
                        continue
                    targets[m] = max(0, min(4095, targets[m] + STEP_RAW))
                    write[m] = targets[m]
                with self._lock:
                    self.hand.bus.sync_write_positions(write)
                time.sleep(STEP_DELAY)
                pos = self._read_positions([m for m in ids if m not in stalled])
                for m, p in pos.items():
                    buffers[m].append(p)
                    if (len(buffers[m]) == NUM_STABLE
                            and max(buffers[m]) - min(buffers[m]) <= STALL_ATOL):
                        stalled.add(m)
                        holding[m] = p
                        emit({"type": "tension", "total": len(ids), "stalled": len(stalled),
                              "joint": self.id_joint.get(m, m)})
                if holding:
                    with self._lock:
                        self.hand.bus.sync_write_positions(holding)
            emit({"type": "log",
                  "msg": "tendons held at the hard-stop - secure the slack side, then Stop"
                         if not stop.is_set() else "tension stopped"})
            # Hold under torque until the user stops (so they can secure the tendons).
            while not stop.is_set():
                with self._lock:
                    if holding:
                        self.hand.bus.sync_write_positions(holding)
                time.sleep(0.1)
        finally:
            try:
                self.set_all_torque(False)
            except Exception:
                pass
            self._worker_active = False
            emit({"type": "done", "mode": "tension"})

    def run_auto(self, joints: Optional[List[str]], stop: threading.Event,
                 emit: Callable[[dict], None]) -> None:
        """Auto stall-sweep via the validated Calibrator. ``joints=None`` = whole hand."""
        self._worker_active = True
        try:
            order = _ordered_joints(self.hand)
            sel = [j for j in order if (joints is None or j in set(joints))]
            if not sel:
                emit({"type": "log", "msg": "no joints selected"})
                return
            for j in sel:
                emit({"type": "joint", "joint": j, "status": "pending"})
            params = CalibrationParams()
            term = _UiTerm(emit)
            calib = Calibrator(self.hand, params, term)
            # Calibrator.run blocks; honour Stop by polling between joints via term.step.
            results = calib.run(sel, load_prev(self.calib_path) if self.calib_path else {})
            for r in results:
                emit({"type": "joint", "joint": r.name, "status": r.status,
                      "min": r.raw_min, "max": r.raw_max, "center": r.center})
            if results:
                merge = joints is not None      # subset runs are non-destructive
                write_calibration(self.calib_path, results, self.hand, merge=merge, term=term)
                self._reload()
            emit({"type": "done", "mode": "auto",
                  "count": len(results), "path": str(self.calib_path)})
        except Exception as e:                 # surface, don't crash the worker silently
            emit({"type": "log", "msg": f"calibration error: {e}"})
            emit({"type": "done", "mode": "auto", "count": 0})
        finally:
            try:
                self.set_all_torque(False)
            except Exception:
                pass
            self._worker_active = False

    def save_manual(self, observed: Dict[int, tuple]) -> int:
        """Persist manually-swept {mid: (lo, hi)} limits (merged into the existing file)."""
        path = self.calib_path
        data = {}
        if path and path.exists():
            try:
                data = read_mapping(path)
            except Exception:
                data = {}
        saved = 0
        for mid, (lo, hi) in observed.items():
            lo, hi = sorted((int(lo), int(hi)))
            if hi - lo < 30:                    # ignore joints that never really moved
                continue
            data[str(mid)] = {"joint": self.id_joint.get(mid, f"id_{mid}"),
                              "min": lo, "max": hi, "center": (lo + hi) // 2, "range": hi - lo}
            saved += 1
        if saved:
            order = {j.motor_id: i for i, j in enumerate(self.hand.config.joints)}
            data = dict(sorted(data.items(), key=lambda kv: order.get(int(kv[0]), 999)))
            dump_mapping(path, data)
            self._reload()
        return saved
