"""hand_bus.py — the GUI's hand connection, but routed through the HARDWARE BRIDGE.

Drop-in for the firmware GUI's ``HandBus``: the public surface (``connect`` /
``sync_write`` / ``set_torque`` / ``pct_to_raw`` / ``state.motors`` / ``calib`` / …) is
byte-for-byte the same, so the ported Control / Setup / Calibration / Poses / Teleop
panels run against it unchanged. The ONE difference from the firmware bus is internal:
it never opens the Feetech serial port itself. Every hardware action is an RPC to the
ROS-free hand bridge (``mabel_hw.rpc.hand_server``), which is the sole owner of the bus — so the
GUI is a pure bridge client, and it works on macOS where rclpy is unavailable.

The per-joint calibration MATH (pct<->raw, offset/wrap, out-of-range, swept-range commit)
stays here, identical to the firmware bus; only the transactions that touch a motor —
open/close, state poll, sync-write, register r/w, ping, OFSCAL re-zero, calibration save —
are delegated to the bridge over the socket.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from orcahand import constants as _C          # top-level orcahand (see package __init__)
from orcahand.joints import Joint
from mabel_hw.rpc.hand_server import HandBridgeClient, HandBridgeError, ensure_daemon

BAUDRATE = 1_000_000
TOTAL = 17

# Valid single-turn encoder window (any stored limit MUST live here).
RAW_LO, RAW_HI, RAW_MID = 0, 4095, 2047

# GUI user data (recorded poses / animations) — kept file-based, alongside the bridge.
_DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
POSES_PATH = _DATA / "gestures" / "saved_normalized_poses.json"
ANIM_PATH = _DATA / "animations" / "saved_animations.json"


def sanitize_limits(lo, hi, center=None) -> tuple[int, int, int, bool]:
    """Force a (min, max, center) triple into the valid encoder window [0, 4095].

    Returns ``(min, max, center, ok)``; ``ok`` is False when the stored values were
    unusable (so the joint falls back to the full span and is flagged for recalibration).
    Identical to the firmware bus so a poisoned file behaves the same here.
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
    if hi - lo < 1:
        lo, hi, ok = RAW_LO, RAW_HI, False
    if center is None:
        center = (lo + hi) // 2
    else:
        try:
            center = int(center)
        except (TypeError, ValueError):
            center = (lo + hi) // 2
    if not (lo <= center <= hi):
        center = (lo + hi) // 2
    return lo, hi, center, ok


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


# ── tiny client-side shims so panels that reach `bus._oh.config` keep working ──
class _RemoteConfig:
    def __init__(self, joints, calibration_path, config_path):
        self.joints = joints
        self.calibration_path = calibration_path
        self.config_path = config_path

    def by_name(self):
        return {j.name: j for j in self.joints}

    def by_id(self):
        return {j.motor_id: j for j in self.joints}


class _RemoteHand:
    """Stands in for ``OrcaHand`` for the read-only `.config` / `._active_ids` access the
    Teleop and Calibration panels do — every real bus op still goes over the RPC."""

    def __init__(self, config, active_ids):
        self.config = config
        self._active_ids = set(active_ids)


class HandBus:
    """Thread-safe hand connection shared across all studio panels (bridge-backed)."""

    POLL_HZ = 10

    def __init__(self):
        self.port: Optional[str] = None
        self.hand: str = "left"
        self.calib: dict = {}                       # {mid: {min, max, center, offset}}
        self.state = HandState()
        self.joint_id: dict = {}
        self.id_joint: dict = {}

        self._oh: Optional[_RemoteHand] = None
        self._joints: list = []
        self._client: Optional[HandBridgeClient] = None
        self._lock = threading.RLock()
        self._callbacks: list[Callable] = []
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._log: list[str] = []

    # ── connection ────────────────────────────────────────────────────────────
    def connect(self, port: str, hand: str = "left") -> tuple[bool, str]:
        self.port = port
        self.hand = "right" if str(hand).lower().startswith("r") else "left"
        try:
            if self._client is None or not self._client.connected:
                self._client, how = ensure_daemon(logger=self.log)
                self.log(f"hand bridge {how}")
            payload = self._client.call("open", side=self.hand, port=port or "")
        except Exception as e:
            return False, str(e)

        self._apply_payload(payload)
        self.state = HandState(hand=self.hand, connected=True)
        for j in self._joints:
            self.state.motors[j.motor_id] = MotorState(mid=j.motor_id, joint=j.name)

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.log(f"Connected to {payload.get('port')} ({self.hand} hand) via hand bridge")
        return True, "OK"

    def _apply_payload(self, payload: dict):
        """Rebuild joint map + calibration + the `_oh` shim from a bridge config payload
        (the reply to open / reload_config / save_calib)."""
        joints = []
        self.calib = {}
        for jd in payload.get("joints", []):
            mid = int(jd["id"])
            lo, hi, ctr, ok = sanitize_limits(jd["raw_min"], jd["raw_max"], jd["center"])
            off = int(jd.get("offset", 0)) % 4096
            self.calib[mid] = {"min": lo, "max": hi, "center": ctr, "offset": off}
            if not ok:
                self.log(f"⚠ {jd['name']} (id {mid}) had an out-of-range calibration — "
                         f"using full span; please recalibrate it.")
            joints.append(Joint(
                name=jd["name"], motor_id=mid,
                angle_min=float(jd["angle_min"]), angle_max=float(jd["angle_max"]),
                invert=bool(jd["invert"]),
                raw_min=int(jd["raw_min"]), raw_max=int(jd["raw_max"]),
                center=int(jd["center"]), offset=off))
        self._joints = joints
        self.joint_id = {j.name: j.motor_id for j in joints}
        self.id_joint = {j.motor_id: j.name for j in joints}
        cfg = _RemoteConfig(joints, payload.get("calibration_path"),
                            payload.get("config_path"))
        self._oh = _RemoteHand(cfg, payload.get("active_ids", []))

    def disconnect(self):
        self._running = False
        try:
            if self._client is not None and self._client.connected:
                self._client.call("close", side=self.hand)
        except Exception:
            pass
        self.state.connected = False
        self.state.torque_on = False
        self.log("Disconnected")

    # ── RPC helper ────────────────────────────────────────────────────────────
    def _call(self, op, **a):
        if self._client is None or not self._client.connected:
            raise HandBridgeError("hand bridge not connected")
        return self._client.call(op, side=self.hand, **a)

    # ── write ─────────────────────────────────────────────────────────────────
    def sync_write(self, targets: dict, speed: int = 2000, torque: int = 500) -> bool:
        """Send {motor_id: raw_pos} via the bridge. ``torque`` kept for API parity."""
        if not self.state.connected:
            return False
        try:
            self._call("sync_write",
                       targets={str(int(m)): int(r) for m, r in targets.items()},
                       speed=int(speed))
            return True
        except Exception as e:
            self.log(f"Write error: {e}")
            return False

    def write_register(self, mid: int, reg: int, val: int, nbytes: int = 1) -> bool:
        if not self.state.connected:
            return False
        try:
            data = [val & 0xFF] if nbytes == 1 else [val & 0xFF, (val >> 8) & 0xFF]
            return bool(self._call("write_register", mid=int(mid), reg=int(reg), data=data))
        except Exception:
            return False

    def read_register(self, mid: int, reg: int, nbytes: int = 1) -> Optional[int]:
        if not self.state.connected:
            return None
        try:
            return self._call("read_register", mid=int(mid), reg=int(reg), n=int(nbytes))
        except Exception:
            return None

    def set_torque(self, on: bool):
        if not self.state.connected:
            return
        try:
            self._call("set_torque", on=bool(on))
        except Exception as e:
            self.log(f"Torque toggle failed: {e}")
            return
        self.state.torque_on = on
        self.log(f"Torque {'ON' if on else 'OFF'}")
        self._notify()

    def center_offset(self, mid: int, target: int = 2048) -> bool:
        """Re-zero one motor: burn its CURRENT pose as ``target`` (OFSCAL) on the bridge."""
        if not self.state.connected:
            return False
        try:
            return bool(self._call("center_offset", mid=int(mid), target=int(target)))
        except Exception as e:
            self.log(f"center_offset failed for id {mid}: {e}")
            return False

    def recenter(self, mid: int, span: Optional[int] = None) -> tuple[bool, str]:
        """Hardware re-center: latch the joint's present pose as 2048 and rebuild its
        range as 2048 ± span/2 (offset cleared). Same as the firmware bus."""
        jname = self.id_joint.get(mid, f"ID {mid}")
        if not self.ping(mid):
            return False, f"ID {mid} ({jname}) not responding — connect the hand first."
        if span is None:
            c = self.calib.get(mid, {"min": 0, "max": 4095})
            span = abs(int(c.get("max", 4095)) - int(c.get("min", 0)))
        span = max(100, min(3800, int(span)))
        if not self.center_offset(mid):
            return False, f"ID {mid} ({jname}): failed to set midpoint."
        lo, hi = 2048 - span // 2, 2048 + span // 2
        self.calib[mid] = {"min": lo, "max": hi, "center": 2048, "offset": 0}
        self.save_calib()
        return True, f"↺ {jname}: centred at 2048 → range {lo}–{hi} (span {span}, saved)."

    def commit_swept_range(self, mid: int, cont_lo: float, cont_hi: float) -> dict:
        """Store a swept continuous (unwrapped) tick range as this joint's calibration,
        picking a software ``offset`` only if it straddles the 0/4095 seam. Pure math —
        identical to the firmware bus (see orcahand.constants.wrapped_delta)."""
        lo_c, hi_c = sorted((int(round(cont_lo)), int(round(cont_hi))))
        span = max(1, min(4095, hi_c - lo_c))
        lo_raw = lo_c % _C.WRAP
        if lo_raw + span <= _C.RAW_MAX:
            off, lo, hi = 0, lo_raw, lo_raw + span
        else:
            r_c = int(round((lo_c + hi_c) / 2.0)) % _C.WRAP
            off = (r_c - 2048) % _C.WRAP
            lo, hi = 2048 - span // 2, 2048 + span // 2
        c = {"min": int(lo), "max": int(hi), "center": int((lo + hi) // 2),
             "offset": int(off)}
        self.calib[mid] = c
        return c

    def read_positions_direct(self) -> dict[int, int]:
        """Fresh one-shot read of every motor position (bypasses the poll)."""
        if not self.state.connected:
            return {}
        try:
            pos = self._call("read_positions", ids=[])
            return {int(m): int(r) for m, r in (pos or {}).items()}
        except Exception:
            return {}

    def ping(self, mid: int) -> bool:
        if not self.state.connected:
            return False
        try:
            return bool(self._call("ping", mid=int(mid)))
        except Exception:
            return False

    # ── position helpers (ONE convention; identical to the firmware bus) ───────
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
        return _C.remove_offset(corrected, off)

    def out_of_range(self, mid: int, raw: Optional[int],
                     margin_frac: float = 0.12, margin_min: int = 50) -> bool:
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

    # ── direction / flip (per-joint invert) ───────────────────────────────────
    def is_inverted(self, mid: int) -> bool:
        return self._inverted(mid)

    def set_invert(self, mid: int, invert: bool) -> bool:
        """Flip a joint's direction mapping.  Applies LIVE immediately (pct↔raw use
        it), and tries to persist through the bridge's config."""
        if self._oh is not None:
            j = self._oh.config.by_id().get(mid)
            if j is not None:
                try:
                    j.invert = bool(invert)
                except Exception:
                    pass
        try:
            return bool(self._call("set_invert", mid=int(mid), invert=bool(invert)))
        except Exception:
            self.log(f"↔ invert applied live for id {mid} (bridge persist unavailable)")
            return False

    def nudge(self, mid: int, pct: float, torque: bool = True) -> bool:
        """Enable torque on this motor and drive it toward `pct` — used to VISUALISE a
        joint's positive direction while checking / flipping calibration."""
        try:
            if torque:
                self.write_register(mid, 40, 1, 1)
            return self.sync_write({int(mid): self.pct_to_raw(mid, pct)}, speed=800, torque=350)
        except Exception:
            return False

    # ── calibration data (bridge owns the file write + live reload) ────────────
    def save_calib(self):
        """Persist edited limits through the bridge, which writes every synced copy and
        reloads the live driver so the new clamp/scale takes effect immediately."""
        out = {}
        for mid, c in self.calib.items():
            lo, hi, ctr, ok = sanitize_limits(c.get("min"), c.get("max"), c.get("center"))
            off = int(c.get("offset", 0)) % 4096
            if not ok:
                off = 0
                self.log(f"⚠ refusing to save out-of-range limits for id {mid} "
                         f"({self.id_joint.get(mid, mid)}); wrote full span instead.")
            self.calib[mid] = {"min": lo, "max": hi, "center": ctr, "offset": off}
            name = self.id_joint.get(mid, f"id_{mid}")
            out[str(mid)] = {"joint": name, "min": lo, "max": hi,
                             "center": ctr, "offset": off, "range": hi - lo}
        try:
            res = self._call("save_calib", mapping=out)
        except Exception as e:
            self.log(f"Calibration save failed: {e}")
            return
        if isinstance(res, dict) and res.get("config"):
            self._apply_payload(res["config"])
        written = (res or {}).get("written", []) if isinstance(res, dict) else []
        names = ", ".join(pathlib.Path(p).parent.name + "/" + pathlib.Path(p).name
                          for p in written)
        self.log(f"Calibration saved + synced → {names or 'nowhere'}")

    def reload_calib(self):
        try:
            payload = self._call("reload_calib")
            self._apply_payload(payload)
            self.log("Calibration reloaded")
        except Exception as e:
            self.log(f"Calibration reload failed: {e}")

    # ── pose / animation data (GUI user data; file-based) ──────────────────────
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

    # ── log / subscription ─────────────────────────────────────────────────────
    def log(self, msg: str):
        entry = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        self._log.append(entry)
        if len(self._log) > 200:
            self._log = self._log[-200:]
        self._notify()

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

    # ── background poll (state over RPC) ───────────────────────────────────────
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
        if not self.state.connected:
            return
        try:
            st = self._call("state")
        except Exception:
            return
        raw_position = st.get("raw_position", {})
        velocity = st.get("velocity", {})
        effort = st.get("effort", {})
        temperature = st.get("temperature", {})
        for mid, ms in self.state.motors.items():
            name = self.id_joint.get(mid)
            raw = raw_position.get(name)
            if raw is not None:
                ms.raw = int(raw)
                ms.pct = self.raw_to_pct(mid, ms.raw)
                ms.vel = float(velocity.get(name, 0.0))
                ms.load = abs(float(effort.get(name, 0.0))) * 1000.0
                ms.temp = float(temperature.get(name, 0.0))
                ms.responding = True
            else:
                ms.responding = False
        self._notify()
