"""hand_rpc.py — the ORCA-hand bridge SERVICE: the one owner of the Feetech bus.

This is the transport-agnostic core of the hand bridge. It wraps one
:class:`orcahand.OrcaHand` per side (the SAME vendored HAL the ROS bridge uses) and
exposes the *full* low-level surface the firmware GUI needs — not just calibrated joint
angles, but raw tick reads/writes, register access, ping, and the OFSCAL re-zero — so a
GUI client can do Control / Setup / Calibration / Poses / Teleop entirely through the
bridge, never opening the serial port itself.

Why this exists: the rest of the hardware bridge speaks ROS DDS, which is unavailable on
macOS (no rclpy). This service is pure pyserial + plain Python, so the SAME "GUI is a
client, the bridge owns the hardware" architecture works on a Mac hand-bench. On the
robot it can be hosted next to the ROS subsystem; here it runs standalone (see
``hand_bridge.py``).

Every method is JSON-able in and out and is serialised by one re-entrant lock, so it is
safe to drive from several client connections at once. ``op(name, **args)`` dispatches by
name (the wire protocol in ``hand_bridge.py`` is a thin shell over this).
"""

from __future__ import annotations

import pathlib
import threading
import time
from typing import Dict, List, Optional

from ..drivers.orcahand import OrcaHand
from ..drivers.orcahand.calibration import dump_mapping


# Ops that take a ``side`` and need that side already open. Everything routes through
# ``HandBridgeService.op`` so the socket server stays a dumb pipe.
class HandBridgeError(Exception):
    """Raised for a client-visible failure (returned as {"ok": false, "e": msg})."""


class HandBridgeService:
    """Owns the ORCA hands; the single owner of the TTL serial bus on this host."""

    def __init__(self, logger=None):
        self._log = logger or (lambda *a, **k: None)
        self._lock = threading.RLock()
        self._hands: Dict[str, OrcaHand] = {}   # side -> connected OrcaHand
        self._ports: Dict[str, str] = {}        # side -> the port string used

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def shutdown(self):
        with self._lock:
            for side, oh in list(self._hands.items()):
                try:
                    oh.disconnect()
                except Exception:
                    pass
            self._hands.clear()

    def _hand(self, side: str) -> OrcaHand:
        oh = self._hands.get(side)
        if oh is None:
            raise HandBridgeError(f"{side} hand not open")
        return oh

    @staticmethod
    def _norm_side(side: str) -> str:
        return "right" if str(side or "").lower().startswith("r") else "left"

    # ── dispatch ──────────────────────────────────────────────────────────────
    def op(self, name: str, args: dict):
        fn = getattr(self, f"_op_{name}", None)
        if fn is None:
            raise HandBridgeError(f"unknown op: {name}")
        with self._lock:
            return fn(args or {})

    # ── ops ───────────────────────────────────────────────────────────────────
    def _op_hello(self, a):
        return {"server": "mabel-hand-bridge", "version": 1,
                "sides_open": sorted(self._hands.keys())}

    def _config_payload(self, oh: OrcaHand, side: str) -> dict:
        cfg = oh.config
        joints = []
        for j in cfg.joints:
            joints.append({
                "name": j.name, "id": int(j.motor_id),
                "invert": bool(j.invert),
                "raw_min": int(j.raw_min), "raw_max": int(j.raw_max),
                "center": int(j.center), "offset": int(getattr(j, "offset", 0)),
                "angle_min": float(j.angle_min), "angle_max": float(j.angle_max),
            })
        return {
            "side": side,
            "port": getattr(oh.bus, "port", None),
            "calibration_path": (str(cfg.calibration_path)
                                 if cfg.calibration_path is not None else None),
            "config_path": (str(cfg.config_path)
                            if getattr(cfg, "config_path", None) is not None else None),
            "missing": list(oh.missing_joints),
            "active_ids": sorted(int(i) for i in oh._active_ids),
            "joints": joints,
        }

    def _op_open(self, a):
        side = self._norm_side(a.get("side", "left"))
        port = (a.get("port") or "").strip()
        # Reopen cleanly if already open (e.g. switching ports / sides).
        old = self._hands.pop(side, None)
        if old is not None:
            try:
                old.disconnect()
            except Exception:
                pass
            time.sleep(0.2)
        try:
            if port.lower() in ("mock", "sim"):
                oh = OrcaHand.mock(side)
            else:
                oh = OrcaHand(model=side, port=port or None)
            oh.connect(enable_torque=False, scan=True)
        except Exception as e:
            msg = str(e)
            if ("exclusively lock" in msg or "Resource temporarily unavailable" in msg
                    or "Errno 35" in msg or "Errno 16" in msg):
                dev = port or "the adapter"
                msg = (f"Port {dev} is busy — another program still holds it (an old GUI, "
                       f"a script, or the ROS bridge). Close it and reconnect.")
            raise HandBridgeError(msg)
        self._hands[side] = oh
        self._ports[side] = port
        self._log(f"[{side}] open on {getattr(oh.bus, 'port', '?')} "
                  f"({len(oh._active_ids)} motors, missing={oh.missing_joints})")
        return self._config_payload(oh, side)

    def _op_close(self, a):
        side = self._norm_side(a.get("side", "left"))
        oh = self._hands.pop(side, None)
        if oh is not None:
            try:
                oh.disconnect()
            except Exception:
                pass
            self._log(f"[{side}] closed")
        return True

    def _op_reload_config(self, a):
        """Rebuild the OrcaHand from its config file so a freshly-saved calibration
        becomes the live clamp/scale (no reconnect dance for the client)."""
        side = self._norm_side(a.get("side", "left"))
        port = self._ports.get(side, "")
        # Re-open the same side/port; reuses _op_open's reconnect-clean path.
        return self._op_open({"side": side, "port": port})

    def _op_state(self, a):
        side = self._norm_side(a.get("side", "left"))
        oh = self._hands.get(side)
        if oh is None:
            return {}
        st = oh.get_state()
        return {
            "raw_position": dict(st.raw_position),
            "velocity": dict(st.velocity),
            "effort": dict(st.effort),          # PWM duty [-1,1] (drive effort)
            "current": dict(st.current),        # A   (measured)
            "torque": dict(st.torque),          # N·m (measured, τ = Kt·I)
            "temperature": dict(st.temperature),
            "voltage": dict(st.voltage),
        }

    # ── writes ────────────────────────────────────────────────────────────────
    def _op_sync_write(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        targets = a.get("targets") or {}
        speed = a.get("speed")
        if speed:
            oh.bus.set_goal_speed(int(speed))
        clamped = {int(m): max(0, min(4095, int(r))) for m, r in targets.items()}
        oh.bus.sync_write_positions(clamped)
        return True

    def _op_set_torque(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        if bool(a.get("on")):
            oh.enable_torque()
        else:
            oh.disable_torque()
        return True

    def _op_set_torque_enable(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        oh.bus.set_torque_enable(bool(a.get("on")), int(a["mid"]))
        return True

    def _op_set_torque_limit(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        oh.bus.set_torque_limit(int(a["val"]), int(a["mid"]))
        return True

    def _op_set_goal_speed(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        oh.bus.set_goal_speed(int(a["val"]), int(a["mid"]))
        return True

    def _op_write_register(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        try:
            oh.bus.write_register(int(a["mid"]), int(a["reg"]), list(a["data"]))
            return True
        except Exception:
            return False

    def _op_read_register(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        try:
            return oh.bus.read_register(int(a["mid"]), int(a["reg"]), int(a.get("n", 1)))
        except Exception:
            return None

    def _op_ping(self, a):
        oh = self._hands.get(self._norm_side(a.get("side", "left")))
        if oh is None:
            return False
        try:
            return bool(oh.bus.ping(int(a["mid"])))
        except Exception:
            return False

    def _op_read_positions(self, a):
        oh = self._hands.get(self._norm_side(a.get("side", "left")))
        if oh is None:
            return {}
        ids = a.get("ids")
        if not ids:
            ids = sorted(oh._active_ids)
        try:
            pos = oh.bus.read_positions([int(i) for i in ids])
            return {str(int(m)): int(r) for m, r in pos.items()}
        except Exception:
            return {}

    def _op_read_feedback(self, a):
        oh = self._hand(self._norm_side(a.get("side", "left")))
        try:
            return oh.bus.read_feedback(int(a["mid"]))
        except Exception:
            return None

    def _op_center_offset(self, a):
        """Burn the motor's CURRENT pose as ``target`` (2048 = centre) via OFSCAL.

        Server-side so the disable-torque → ofs_cal → settle → verify sequence keeps
        tight timing without round-trips. Returns True only if the motor really reads
        ~target afterwards (the same guard the firmware GUI used)."""
        oh = self._hand(self._norm_side(a.get("side", "left")))
        mid = int(a["mid"])
        target = int(a.get("target", 2048))
        bus = oh.bus
        if not hasattr(bus, "ofs_cal"):
            return False
        try:
            bus.set_torque_enable(False, mid)
            time.sleep(0.05)
            bus.ofs_cal(mid, target)
            time.sleep(0.12)
            fb = bus.read_feedback(mid)
            if fb is None:
                return False
            return abs(int(fb["position"]) - target) <= 120
        except Exception as e:
            self._log(f"center_offset id {mid} failed: {e}")
            return False

    # ── calibration persistence (the bridge owns the file + live reload) ───────
    @staticmethod
    def _repo_root() -> Optional[pathlib.Path]:
        for parent in pathlib.Path(__file__).resolve().parents:
            if (parent / "ros").is_dir() and (parent / "firmware").is_dir():
                return parent
        return None

    def _calib_sync_paths(self, oh: OrcaHand, side: str) -> List[pathlib.Path]:
        """Every calibration file that drives this hand, so a save keeps the bridge's
        own config, the firmware package, and the ROS package all in sync. The bridge's
        OWN ``calibration_path`` (what it reloads from) is always first."""
        fname_json = f"{side}_calibration.json"
        fname_yaml = f"{side}_calibration.yaml"
        cands: List[pathlib.Path] = []
        if oh.config.calibration_path is not None:
            cands.append(pathlib.Path(oh.config.calibration_path))   # authoritative
        root = self._repo_root()
        if root is not None:
            # the bridge's own read-first mirror (calibration_data/README.md)
            cd = root / "hardware_bridge" / "calibration_data" / "hands"
            cands.append(cd / fname_json)
            fw = root / "firmware" / "orca_hand" / "feetech" / "orcahand" / "configs"
            cands += [fw / fname_yaml, fw / fname_json]
            cands.append(root / "ros" / "src" / "mabel_hand_hardware" /
                         "config" / fname_json)
            hb = (root / "hardware_bridge" / "mabel_hw" / "drivers"
                  / "orcahand" / "configs")
            cands += [hb / fname_yaml, hb / fname_json]
        out, seen = [], set()
        for p in cands:
            p = pathlib.Path(p).resolve()
            if p not in seen and p.parent.exists():
                seen.add(p)
                out.append(p)
        return out

    def _op_save_calib(self, a):
        """Write a {motor-id: {joint,min,max,center,offset,range}} mapping to every
        synced location (format chosen by extension) and reload the live driver so the
        new clamp/scale takes effect immediately. Returns the written paths + config."""
        side = self._norm_side(a.get("side", "left"))
        mapping = a.get("mapping") or {}
        oh = self._hand(side)
        mapping = {str(k): v for k, v in mapping.items()}
        written = []
        for p in self._calib_sync_paths(oh, side):
            try:
                dump_mapping(p, mapping)
                written.append(str(p))
            except Exception as e:
                self._log(f"calib write failed for {p}: {e}")
        # Reload from the freshly-written file (rebuild OrcaHand on the same port).
        self._op_open({"side": side, "port": self._ports.get(side, "")})
        return {"written": written, "config": self._config_payload(self._hand(side), side)}

    def _op_reload_calib(self, a):
        side = self._norm_side(a.get("side", "left"))
        return self._op_open({"side": side, "port": self._ports.get(side, "")})
