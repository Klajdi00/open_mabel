"""Shared hardware connection + live state for the GUI -- backed by ``orcahand``.

``HandBus`` is the single object every panel shares. It used to drive its own
copy of the Feetech protocol (via ``orca_core.hardware.feetech``); it is now a
thin adapter over the lean :class:`orcahand.OrcaHand` API -- the same driver the
ROS node and the controllers use -- so the GUI can never drift from them. The
public surface (``connect``/``sync_write``/``set_torque``/``pct_to_raw``/
``state.motors``/...) is unchanged, so the panels did not need to change.

Pass ``port="mock"`` to ``connect`` to run the whole GUI with no hardware.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

# Make the package importable from a bare checkout (feetech/ is the repo root).
_FEETECH = pathlib.Path(__file__).resolve().parent.parent
if str(_FEETECH) not in sys.path:
    sys.path.insert(0, str(_FEETECH))

from orcahand import OrcaHand  # noqa: E402
from orcahand import constants as _C  # noqa: E402

BAUDRATE = 1_000_000
TOTAL = 17

# Valid single-turn encoder window. Every calibrated limit MUST live here; a
# value outside it is a garbled serial read (see orcahand.bus._decode_feedback)
# that slipped into the file before this guard existed.
RAW_LO, RAW_HI, RAW_MID = 0, 4095, 2047


def sanitize_limits(lo, hi, center=None) -> tuple[int, int, int, bool]:
    """Force a (min, max, center) triple into the valid encoder window [0, 4095].

    Returns ``(min, max, center, ok)``. ``ok`` is False when the stored values
    were out of range / unusable (so the caller can flag the joint as needing a
    fresh calibration); in that case the joint falls back to the full encoder
    span so the hand still behaves sanely instead of commanding wild positions.
    """
    try:
        lo, hi = int(lo), int(hi)
    except (TypeError, ValueError):
        return RAW_LO, RAW_HI, RAW_MID, False
    ok = (RAW_LO <= lo <= RAW_HI) and (RAW_LO <= hi <= RAW_HI)
    lo = max(RAW_LO, min(RAW_HI, lo))
    hi = max(RAW_LO, min(RAW_HI, hi))
    if lo > hi:
        lo, hi = hi, lo
    if hi - lo < 1:                       # collapsed/garbled span -> unusable
        lo, hi, ok = RAW_LO, RAW_HI, False
    if center is None:
        center = (lo + hi) // 2
    else:
        try:
            center = int(center)
        except (TypeError, ValueError):
            center = (lo + hi) // 2
    if not (lo <= center <= hi):          # center must sit inside the span
        center = (lo + hi) // 2
    return lo, hi, center, ok

# Legacy data files (poses / animations / a fallback calib path) live under data/.
CALIB_PATH = _FEETECH / "data" / "calibration" / "joint_limits.json"
POSES_PATH = _FEETECH / "data" / "gestures" / "saved_normalized_poses.json"
ANIM_PATH = _FEETECH / "data" / "animations" / "saved_animations.json"

# Kept for panels that still read it directly; the live map comes from the config.
JOINT_ID = {
    "left": {
        "middle_pip": 1, "pinky_mcp": 2, "pinky_pip": 3, "pinky_abd": 4,
        "middle_abd": 5, "index_abd": 6, "index_mcp": 7, "index_pip": 8,
        "thumb_mp": 9, "thumb_ip": 10, "thumb_abd": 11, "thumb_cmc": 12,
        "ring_abd": 13, "ring_pip": 14, "ring_mcp": 15, "middle_mcp": 16,
        "wrist": 17,
    },
    "right": {
        "index_pip": 1, "index_mcp": 2, "index_abd": 3, "middle_abd": 4,
        "pinky_abd": 5, "pinky_pip": 6, "pinky_mcp": 7, "middle_pip": 8,
        "middle_mcp": 9, "ring_mcp": 10, "ring_pip": 11, "ring_abd": 12,
        "thumb_cmc": 13, "thumb_abd": 14, "thumb_mp": 15, "thumb_ip": 16,
        "wrist": 17,
    },
}


@dataclass
class MotorState:
    mid: int
    joint: str = "?"
    raw: Optional[int] = None
    pct: float = 0.0
    vel: float = 0.0
    load: float = 0.0
    temp: float = 0.0
    responding: bool = False


@dataclass
class HandState:
    motors: dict = field(default_factory=dict)  # {mid: MotorState}
    torque_on: bool = False
    connected: bool = False
    hand: str = "left"


class HandBus:
    """Thread-safe hand connection shared across all GUI panels (orcahand-backed)."""

    POLL_HZ = 10

    def __init__(self):
        self.port: Optional[str] = None
        self.hand: str = "left"
        self.calib: dict = {}                       # {mid: {min, max, center}}
        self.state = HandState()
        self.joint_id: dict = {}
        self.id_joint: dict = {}

        self._oh: Optional[OrcaHand] = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable] = []
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._log: list[str] = []

    # ── Connection ──────────────────────────────────────────────────────────
    def connect(self, port: str, hand: str = "left") -> tuple[bool, str]:
        self.port = port
        self.hand = hand
        try:
            if port and port.lower() in ("mock", "sim"):
                self._oh = OrcaHand.mock(hand)
            else:
                self._oh = OrcaHand(hand, port=port or None)
            self._oh.connect(enable_torque=False, scan=True)
        except Exception as e:
            self._oh = None
            msg = str(e)
            # The bus opens the port with an exclusive lock so two processes
            # can't fight over one adapter. The usual cause of a failure here is
            # a previous GUI/script still holding it -- make that actionable
            # instead of surfacing a raw errno.
            if "exclusively lock" in msg or "Resource temporarily unavailable" in msg \
                    or "Errno 35" in msg or "Errno 16" in msg:
                dev = port or "the adapter"
                msg = (f"Port {dev} is busy -- another program still holds it "
                       f"(an old GUI, a script, or the ROS node). Close it "
                       f"(e.g. `lsof {dev}` then kill the PID) and reconnect.")
            return False, msg

        cfg = self._oh.config
        self.joint_id = {j.name: j.motor_id for j in cfg.joints}
        self.id_joint = {j.motor_id: j.name for j in cfg.joints}
        self.calib = {}
        for j in cfg.joints:
            lo, hi, ctr, ok = sanitize_limits(j.raw_min, j.raw_max, j.center)
            self.calib[j.motor_id] = {"min": lo, "max": hi, "center": ctr,
                                      "offset": int(getattr(j, "offset", 0)) % 4096}
            if not ok:
                # Repair the live driver so a poisoned file can't drive the hand.
                j.raw_min, j.raw_max, j.center = lo, hi, ctr
                self.log(f"⚠ {j.name} (id {j.motor_id}) had an out-of-range "
                         f"calibration — reset to full span; please recalibrate it.")

        self.state = HandState(hand=hand, connected=True)
        for j in cfg.joints:
            self.state.motors[j.motor_id] = MotorState(mid=j.motor_id, joint=j.name)

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.log(f"Connected to {self._oh.bus.port} ({hand} hand) via orcahand")
        return True, "OK"

    def disconnect(self):
        self._running = False
        try:
            if self._oh is not None:
                self._oh.disconnect()
        except Exception:
            pass
        self.state.connected = False
        self.state.torque_on = False
        self.log("Disconnected")

    # ── Write ───────────────────────────────────────────────────────────────
    def sync_write(self, targets: dict, speed: int = 2000, torque: int = 500) -> bool:
        """Send {motor_id: raw_pos} in one SyncWrite. ``torque`` kept for API parity."""
        if not self.state.connected or self._oh is None:
            return False
        try:
            if speed:
                self._oh.bus.set_goal_speed(int(speed))
            self._oh.bus.sync_write_positions(
                {int(m): max(0, min(4095, int(r))) for m, r in targets.items()})
            return True
        except Exception as e:
            self.log(f"Write error: {e}")
            return False

    def write_register(self, mid: int, reg: int, val: int, nbytes: int = 1) -> bool:
        if not self.state.connected or self._oh is None:
            return False
        try:
            data = [val & 0xFF] if nbytes == 1 else [val & 0xFF, (val >> 8) & 0xFF]
            self._oh.bus.write_register(mid, reg, data)
            return True
        except Exception:
            return False

    def read_register(self, mid: int, reg: int, nbytes: int = 1) -> Optional[int]:
        if not self.state.connected or self._oh is None:
            return None
        try:
            return self._oh.bus.read_register(mid, reg, nbytes)
        except Exception:
            return None

    def set_torque(self, on: bool):
        if not self.state.connected or self._oh is None:
            return
        (self._oh.enable_torque if on else self._oh.disable_torque)()
        self.state.torque_on = on
        self.log(f"Torque {'ON' if on else 'OFF'}")
        self._notify()

    def center_offset(self, mid: int, target: int = 2048) -> bool:
        """Re-zero: burn the motor's CURRENT physical pose as ``target`` (2048 =
        centre) into EEPROM, so it immediately reads ~2048.

        Sends the Feetech OFSCAL instruction (0x0B) — the servo recomputes and
        stores its own encoder offset — after disabling torque (required to burn
        the offset safely). This is the dependency-free port of the old
        ``reOfsCal`` sequence; the previous "write 128 to register 40" was wrong
        and never actually moved the zero. Verifies by reading the pose back, so
        a return of ``True`` means the motor really reads ~``target`` now.
        Backs the Setup ↺/Center and the Manual tab's Center / per-joint ↺.
        """
        if not self.state.connected or self._oh is None:
            return False
        bus = getattr(self._oh, "bus", None)
        if bus is None or not hasattr(bus, "ofs_cal"):
            self.log(f"center_offset: bus cannot re-zero (id {mid})")
            return False
        try:
            bus.set_torque_enable(False, mid)     # torque OFF before EEPROM burn
            time.sleep(0.05)
            bus.ofs_cal(mid, target)
            time.sleep(0.12)                      # let the offset settle
            fb = bus.read_feedback(mid)           # confirm it actually re-zeroed
            if fb is None:
                self.log(f"center_offset id {mid}: no read-back after OFSCAL")
                return False
            err = abs(int(fb["position"]) - target)
            if err > 120:
                self.log(f"center_offset id {mid}: pose still {fb['position']} "
                         f"(Δ{err}) after OFSCAL — re-zero did not take")
                return False
            # Re-zero shifts the encoder, but the servo's GOAL register still holds
            # the OLD goal (now in a different frame). Align it with the new pose so
            # enabling torque holds steady instead of snapping to a stale goal --
            # the "acts weird after recenter" symptom. Torque is off here, so this
            # just stores the goal; no motion.
            try:
                bus.sync_write_positions({mid: target})
            except Exception:
                pass
            return True
        except Exception as e:
            self.log(f"center_offset failed for id {mid}: {e}")
            return False

    def recenter(self, mid: int, span: Optional[int] = None) -> tuple[bool, str]:
        """Hardware re-center: latch the joint's *present* pose as the servo
        midpoint (2048) and rebuild its calibrated range as ``2048 ± span/2``.

        This physically moves the encoder zero (burns the STS offset), so the
        software ``offset`` is cleared to 0 -- the seam is now a half-turn from
        the travel in hardware. Use it as the fallback when a joint can't be made
        contiguous in software (``at_encoder_boundary`` stays true) or when you
        explicitly want a clean hardware home. Shared by the Setup panel's ↺ and
        the Calibration panel's Center / Re-center actions. Returns ``(ok, msg)``.
        """
        jname = self.id_joint.get(mid, f"ID {mid}")
        if not self.ping(mid):
            return False, f"ID {mid} ({jname}) not responding — connect the hand first."
        if span is None:
            c = self.calib.get(mid, {"min": 0, "max": 4095})
            span = abs(int(c.get("max", 4095)) - int(c.get("min", 0)))
        # Clamp so a full-span (0–4095) fallback can't rebuild a range that still
        # touches the boundary; 2048 is dead-centre so any span ≤ 4095 fits.
        span = max(100, min(3800, int(span)))
        if not self.center_offset(mid):
            return False, f"ID {mid} ({jname}): failed to set midpoint."
        lo, hi = 2048 - span // 2, 2048 + span // 2
        self.calib[mid] = {"min": lo, "max": hi, "center": 2048, "offset": 0}
        self.save_calib()
        return True, f"↺ {jname}: centred at 2048 → range {lo}–{hi} (span {span}, saved)."

    def commit_swept_range(self, mid: int, cont_lo: float, cont_hi: float) -> dict:
        """Store a swept *continuous* (unwrapped) tick range as this joint's
        calibration, picking a software ``offset`` only if the range straddles
        the 0/4095 seam.

        ``cont_lo``/``cont_hi`` come from integrating ``wrapped_delta`` over the
        sweep (see :func:`orcahand.constants.wrapped_delta`), so they describe the
        true contiguous arc even when it crosses the wrap. If that arc fits inside
        0–4095 without crossing the seam, ``offset`` stays 0 and min/max are the
        plain raw ticks (identical to the legacy behaviour). Otherwise we relocate
        the seam in software: the arc's midpoint is mapped to corrected 2048, so
        min/max land near 2048 and never touch 0/4095 -- no hardware "set home".

        Returns the stored ``{min, max, center, offset}`` dict (not yet saved;
        the caller decides when to ``save_calib``).
        """
        lo_c, hi_c = sorted((int(round(cont_lo)), int(round(cont_hi))))
        span = max(1, min(4095, hi_c - lo_c))
        lo_raw = lo_c % _C.WRAP
        if lo_raw + span <= _C.RAW_MAX:
            # Contiguous arc already fits in-window: no offset, store raw ticks.
            off, lo, hi = 0, lo_raw, lo_raw + span
        else:
            # Arc straddles the seam: re-center in software around 2048.
            r_c = int(round((lo_c + hi_c) / 2.0)) % _C.WRAP    # raw tick at arc centre
            off = (r_c - 2048) % _C.WRAP
            lo, hi = 2048 - span // 2, 2048 + span // 2
        c = {"min": int(lo), "max": int(hi), "center": int((lo + hi) // 2),
             "offset": int(off)}
        self.calib[mid] = c
        return c

    def read_positions_direct(self) -> dict[int, int]:
        """Fresh one-shot read of every motor position (bypasses the poll)."""
        if not self.state.connected or self._oh is None:
            return {}
        try:
            return self._oh.bus.read_positions(sorted(self._oh._active_ids))
        except Exception:
            return {}

    def ping(self, mid: int) -> bool:
        if not self.state.connected or self._oh is None:
            return False
        try:
            return self._oh.bus.ping(mid)
        except Exception:
            return False

    # ── Position helpers ──────────────────────────────────────────────────────
    # ONE percent convention shared by the telemetry bar, the control sliders,
    # poses and macros: 0 % = fully open, 100 % = fully closed. For an
    # `invert: true` joint (wound opposite to rising ticks) that means 0 % = raw_max
    # and 100 % = raw_min. Because every GUI surface uses these same two helpers,
    # the telemetry and the control toggle always agree with each other and with
    # the saved min/max calibration.
    def _inverted(self, mid: int) -> bool:
        if self._oh is None:
            return False
        j = self._oh.config.by_id().get(mid)
        return bool(j and j.invert)

    def pct_to_raw(self, mid: int, pct: float) -> int:
        c = self.calib.get(mid, {"min": 0, "max": 4095})
        lo, hi, off = int(c["min"]), int(c["max"]), int(c.get("offset", 0))
        p = max(0.0, min(100.0, pct)) / 100.0
        if self._inverted(mid):
            p = 1.0 - p
        corrected = int(lo + p * (hi - lo))
        return _C.remove_offset(corrected, off)   # corrected -> real encoder tick

    def out_of_range(self, mid: int, raw: Optional[int],
                     margin_frac: float = 0.12, margin_min: int = 50) -> bool:
        """True when a live reading sits meaningfully outside the joint's
        calibrated [min, max] window — i.e. the calibration no longer matches
        where the joint actually is and needs re-centering.

        Compared in the joint's *corrected* (offset) frame so a seam-straddling
        joint is judged against its true contiguous range, not the raw ticks.
        A small margin (the larger of ``margin_frac`` of the span or
        ``margin_min`` ticks) absorbs the normal slop just past an end-stop, so
        only a genuinely shifted joint trips the warning. A joint left at the
        full-span fallback (0–4095) can never trip it — that is the separate
        "uncalibrated" state, not "drifted out of range".
        """
        c = self.calib.get(mid)
        if c is None or raw is None:
            return False
        lo, hi, off = int(c["min"]), int(c["max"]), int(c.get("offset", 0))
        if hi <= lo:
            return False
        corrected = _C.apply_offset(raw, off)
        margin = max(margin_min, int((hi - lo) * margin_frac))
        return corrected < lo - margin or corrected > hi + margin

    def at_encoder_boundary(self, mid: int, margin: int = 20) -> bool:
        """True when a joint's calibrated range still touches the 0/4095 wrap
        *even after* its software offset — i.e. the travel is so close to a full
        turn that no offset can fully clear the seam (rare for a finger tendon).

        The corrected range is what the encoder actually reads through; a seam
        crossing there is the single-turn discontinuity (dropped/wrapped reads
        get discarded, so the joint appears to stall or mis-read). Normal joints
        get an auto-offset that moves the seam a half-turn away, so this returns
        False for them; when it does trip, a hardware re-center (Set Home) is the
        only remaining fix.
        """
        c = self.calib.get(mid)
        if c is None:
            return False
        lo, hi = int(c["min"]), int(c["max"])
        return lo <= margin or hi >= RAW_HI - margin

    def raw_to_pct(self, mid: int, raw: int) -> float:
        c = self.calib.get(mid, {"min": 0, "max": 4095})
        lo, hi, off = int(c["min"]), int(c["max"]), int(c.get("offset", 0))
        if hi == lo:
            return 0.0
        corrected = _C.apply_offset(raw, off)
        p = max(0.0, min(1.0, (corrected - lo) / (hi - lo)))
        if self._inverted(mid):
            p = 1.0 - p
        return p * 100.0

    # ── Calibration data ──────────────────────────────────────────────────────
    def _calib_path(self) -> pathlib.Path:
        if self._oh is not None and self._oh.config.calibration_path is not None:
            return pathlib.Path(self._oh.config.calibration_path)
        return CALIB_PATH

    @staticmethod
    def _repo_root() -> Optional[pathlib.Path]:
        """Walk up to the MABEL repo root (the dir holding both ros/ and
        firmware/), so calibration can be synced to every copy."""
        for parent in pathlib.Path(__file__).resolve().parents:
            if (parent / "ros").is_dir() and (parent / "firmware").is_dir():
                return parent
        return None

    def _calib_sync_paths(self) -> list[pathlib.Path]:
        """Every ``<side>_calibration.json`` that drives this hand, so a save
        keeps the GUI/driver, the packaged config, and the ROS package in sync.
        Only paths whose directory exists are returned."""
        fname = f"{self.hand}_calibration.json"
        cands = [self._calib_path()]
        root = self._repo_root()
        if root is not None:
            # the hardware bridge's read-first mirror (see
            # hardware_bridge/calibration_data/README.md for the topology)
            cands.append(root / "hardware_bridge" / "calibration_data" /
                         "hands" / fname)
            cands.append(root / "firmware" / "orca_hand" / "feetech" /
                         "orcahand" / "configs" / fname)
            cands.append(root / "ros" / "src" / "mabel_hand_hardware" /
                         "config" / fname)
        out, seen = [], set()
        for p in cands:
            p = pathlib.Path(p).resolve()
            if p not in seen and p.parent.exists():
                seen.add(p)
                out.append(p)
        return out

    def save_calib(self):
        """Persist edited limits and keep the whole stack in sync.

        Writes the same ``<side>_calibration.json`` to every known location (the
        packaged config the GUI/driver reads *and* the ROS package's copy) and
        pushes the limits into the live :class:`~orcahand.joints.Joint` objects so
        the running hand uses them immediately — no reconnect needed.
        """
        by_id = self._oh.config.by_id() if self._oh else {}
        out = {}
        for mid, c in self.calib.items():
            # Never persist a value outside the encoder window: it would re-poison
            # the file and every synced copy (the bug this guards against). min/max
            # are in the *corrected* frame, so they stay in 0-4095 even for a
            # seam-straddling joint -- the wrap is absorbed by ``offset``.
            lo, hi, ctr, ok = sanitize_limits(c.get("min"), c.get("max"),
                                              c.get("center"))
            off = int(c.get("offset", 0)) % 4096
            if not ok:
                off = 0   # a joint reset to full span has no meaningful offset
                self.log(f"⚠ refusing to save out-of-range limits for id {mid} "
                         f"({self.id_joint.get(mid, mid)}); wrote full span instead.")
            self.calib[mid] = {"min": lo, "max": hi, "center": ctr, "offset": off}
            name = self.id_joint.get(mid, f"id_{mid}")
            out[str(mid)] = {"joint": name, "min": lo, "max": hi,
                             "center": ctr, "offset": off, "range": hi - lo}
            j = by_id.get(mid)          # update the LIVE driver
            if j is not None:
                j.raw_min, j.raw_max, j.center = lo, hi, ctr
                j.offset = off
        blob = json.dumps(out, indent=4)
        written = []
        for path in self._calib_sync_paths():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(blob)
                written.append(path)
            except Exception as e:
                self.log(f"Calib write failed for {path}: {e}")
        names = ", ".join(str(p.parent.name) + "/" + p.name for p in written)
        self.log(f"Calibration saved + synced → {names or 'nowhere'}")

    def reload_calib(self):
        path = self._calib_path()
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            by_id = self._oh.config.by_id() if self._oh else {}
            for k, v in raw.items():
                mid = int(k)
                lo, hi, ctr, ok = sanitize_limits(v.get("min"), v.get("max"),
                                                  v.get("center"))
                off = int(v.get("offset", 0)) % 4096
                if not ok:
                    off = 0
                    self.log(f"⚠ {self.id_joint.get(mid, mid)} (id {mid}) on disk "
                             f"is out of range — using full span; recalibrate it.")
                self.calib[mid] = {"min": lo, "max": hi, "center": ctr, "offset": off}
                j = by_id.get(mid)
                if j is not None:
                    j.raw_min, j.raw_max, j.center = lo, hi, ctr
                    j.offset = off
        self.log("Calibration reloaded")

    # ── Pose / animation data (GUI user data; stays file-based under data/) ────
    def load_poses(self) -> dict:
        return json.load(open(POSES_PATH)) if POSES_PATH.exists() else {}

    def save_poses(self, data: dict):
        POSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(POSES_PATH, "w"), indent=2)

    def load_animations(self) -> dict:
        return json.load(open(ANIM_PATH)) if ANIM_PATH.exists() else {}

    def save_animations(self, data: dict):
        ANIM_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(ANIM_PATH, "w"), indent=2)

    # ── Log ───────────────────────────────────────────────────────────────────
    def log(self, msg: str):
        entry = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        self._log.append(entry)
        if len(self._log) > 200:
            self._log = self._log[-200:]
        self._notify()

    # ── Subscription ───────────────────────────────────────────────────────────
    def subscribe(self, cb: Callable):
        self._callbacks.append(cb)

    def unsubscribe(self, cb: Callable):
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

    def _notify(self):
        for cb in list(self._callbacks):
            try:
                cb(self.state, self._log)
            except Exception:
                pass

    # ── Background poll ─────────────────────────────────────────────────────────
    def _poll_loop(self):
        dt = 1.0 / self.POLL_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                self._poll_once()
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _poll_once(self):
        if self._oh is None:
            return
        st = self._oh.get_state()
        for mid, ms in self.state.motors.items():
            name = self.id_joint.get(mid)
            raw = st.raw_position.get(name)
            if raw is not None:
                ms.raw = raw
                ms.pct = self.raw_to_pct(mid, raw)
                ms.vel = float(st.velocity.get(name, 0.0))
                ms.load = abs(float(st.effort.get(name, 0.0))) * 1000.0
                ms.temp = float(st.temperature.get(name, 0.0))
                ms.responding = True
            else:
                ms.responding = False
        self._notify()
