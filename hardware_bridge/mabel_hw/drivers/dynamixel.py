"""dynamixel.py — ROBOTIS Dynamixel Protocol 2.0 via the official ``dynamixel_sdk``.

The `JointGroupDriver` backend for MABEL's neck: two **XC330-T181** servos (model
1090) on a half-duplex TTL bus behind a FT232H adapter (`/dev/ttyUSB0`, serial
``FTB8HR1Y``, 1 Mbaud). One `DynamixelBus` owns the port (SDK PortHandler +
PacketHandler); the group multiplexes the joints on it by servo ID.

Why the SDK (was: an in-house Protocol-2.0 codec): the SDK's `GroupSyncRead` reads
BOTH servos' present state in ONE bus transaction, which lifts telemetry from ~20 Hz
(two per-servo reads at the FT232H 16 ms latency timer) to ~60 Hz — past the 50 Hz the
other actuators run at — and `write1ByteTxRx` ACKs the torque write, so a failed
enable is visible instead of silent. Framing/CRC/half-duplex timing are the SDK's job.

RAW units at this layer are **motor radians** (0..2*pi over 4096 ticks); the calibrated
model-frame view (neutral-anchored: level == 0) lives one layer up in
`subsystems.joint_group`. Connect ASSERTS the safe limp state (torque disabled) — an
X-series servo LATCHES Torque Enable across a host reconnect, so a prior session that
energised the neck must not leave it stiff at bring-up.

Control table (X-series, shared by XC330/XL330/XL430/XM430/XM540):
    64  Torque Enable      (1)     116  Goal Position       (4)
    70  Hardware Error     (1)     126  Present Current     (2)
    84  Position P Gain    (2)     128  Present Velocity    (4)
    82  Position I Gain    (2)     132  Present Position    (4)
    80  Position D Gain    (2)     144  Present Voltage     (2)
                                   146  Present Temperature (1)
"""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import Dict, Mapping, Optional, Tuple

from ..errors import SubsystemUnavailable
from ..types import monotonic
from .seam import GroupFeedback, JointGroupDriver

__all__ = ["DynamixelBus", "DynamixelGroup", "scan", "resolve_device"]

PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_POSITION_P_GAIN = 84         # (2) XC330 factory default ≈900 — stiff for a 0.5 kg head
ADDR_PROFILE_ACCELERATION = 108   # (4) 0 = unlimited
ADDR_PROFILE_VELOCITY = 112       # (4) 0 = unlimited
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_BLOCK = 126          # Present Current(126)..Present Temperature(146)
LEN_PRESENT_BLOCK = 21

# Position-mode Profile Velocity/Acceleration for the neck. 0 = UNLIMITED: the servo
# P-tracks each streamed goal directly instead of re-planning a trapezoidal move per
# goal — the latter makes 50 Hz position streaming JERKY and adds lag. Streamed targets
# are small steps (smooth operator input), so unlimited tracks them smoothly. Override
# with MABEL_HW_NECK_PROFILE_VEL / _ACC (raw register units) to cap neck speed.
_PROFILE_VELOCITY = int(os.getenv("MABEL_HW_NECK_PROFILE_VEL", "0"))
_PROFILE_ACCEL = int(os.getenv("MABEL_HW_NECK_PROFILE_ACC", "0"))

TICKS_PER_REV = 4096
_RAD_PER_TICK = 2.0 * 3.141592653589793 / TICKS_PER_REV
_VEL_LSB_RAD_S = 0.229 * 2.0 * 3.141592653589793 / 60.0   # 0.229 rev/min per LSB
_CUR_LSB_MA = 1.0                                          # XC330: 1.0 mA per LSB

# OUTPUT torque estimate from present current: tau = Kt * I. Kt is SPEC-DERIVED, NOT
# measured — XC330-T181 stall torque 0.92 N·m @ 5.0V / 1.8 A stall current ⇒ effective
# geared-output Kt ≈ 0.51 N·m/A. Signed (drive direction); the calibrated subsystem
# dir-corrects it. ⚠ UNVALIDATED — override per-bus via MABEL_HW_NECK_KT.
_TORQUE_CONST_NM_PER_A = float(os.getenv("MABEL_HW_NECK_KT", "0.51"))

# Model-number -> label (MABEL's neck reports 1090 = XC330-T181). Labels only.
MODEL_NAMES = {
    1090: "XC330-T181", 1080: "XC330-T288", 1230: "XC330-M181", 1240: "XC330-M288",
    1190: "XL330-M288", 1200: "XL330-M077",
    1020: "XM430-W350", 1030: "XM430-W210", 1000: "XM540-W270", 1010: "XM540-W150",
    1070: "XL430-W250", 1160: "2XL430-W250", 1170: "2XC430-W250", 350: "XL320",
}


def _import_sdk():
    """Deferred import so a checkout without dynamixel_sdk can still import mabel_hw
    (the neck simply stays idle, like a missing adapter)."""
    try:
        import dynamixel_sdk as sdk                       # noqa: PLC0415
    except Exception as e:                                # not installed
        raise SubsystemUnavailable(
            f"dynamixel_sdk not installed ({e}) — pip install dynamixel_sdk") from e
    return sdk


def _s16(v: int) -> int:
    return v - 0x10000 if v & 0x8000 else v


def _s32(v: int) -> int:
    return v - 0x100000000 if v & 0x80000000 else v


# ── device resolution ──────────────────────────────────────────────────────────
def resolve_device(device: str = "", serial_id: str = "") -> Optional[str]:
    """Pin the adapter by its FTDI serial (stable across replug), falling back to an
    explicit ``device`` path, then the first ``/dev/ttyUSB*``. None when absent."""
    if serial_id:
        for link in glob.glob("/dev/serial/by-id/*"):
            if serial_id in os.path.basename(link):
                return os.path.realpath(link)
    if device and os.path.exists(device):
        return device
    usb = sorted(glob.glob("/dev/ttyUSB*"))
    return usb[0] if usb else (device or None)


# ── the shared bus (SDK PortHandler + PacketHandler) ────────────────────────────
class DynamixelBus:
    """One serial port, half-duplex, shared by every servo on it. Thread-safe: all
    wire access is under a single lock so tick()/service() (bridge timer) and
    set_torque() (DDS callback) never interleave a packet."""

    def __init__(self, device: str, baud: int, *, serial_id: str = "",
                 timeout: float = 0.02, logger=None):
        self._device = device
        self._serial_id = serial_id
        self._baud = int(baud)
        self._timeout = float(timeout)
        self._log = logger or (lambda *a, **k: None)
        self._sdk = None
        self._port = None                    # SDK PortHandler
        self._pkt = None                     # SDK PacketHandler
        self._path: Optional[str] = None
        self._lock = threading.RLock()

    @property
    def port(self) -> Optional[str]:
        return self._path

    def connect(self) -> None:
        sdk = _import_sdk()
        path = resolve_device(self._device, self._serial_id)
        if not path:
            raise SubsystemUnavailable(
                f"no Dynamixel adapter found (serial={self._serial_id!r}, "
                f"device={self._device!r})")
        # Opening can fail TRANSIENTLY on a hot-replug (udev re-creates the by-id
        # symlink a beat before the tty is openable). Map EVERY open failure to the
        # graceful SubsystemUnavailable the callers retry on — an uncaught error here
        # would propagate through the body_node ROS timer and crash the whole bridge.
        try:
            with self._lock:
                port = sdk.PortHandler(path)
                if not port.openPort():
                    raise OSError(f"openPort failed on {path}")
                if not port.setBaudRate(self._baud):
                    port.closePort()
                    raise OSError(f"setBaudRate({self._baud}) failed on {path}")
                self._sdk, self._port, self._pkt, self._path = (
                    sdk, port, sdk.PacketHandler(PROTOCOL_VERSION), path)
        except SubsystemUnavailable:
            raise
        except Exception as e:                            # SerialException / OSError / …
            self._port = self._pkt = None
            raise SubsystemUnavailable(
                f"Dynamixel adapter {path} present but not openable yet "
                f"(replug settling? busy? permissions?): {e}") from e
        self._set_low_latency(port)
        self._log(f"[dxl] bus open {path} @ {self._baud} (dynamixel_sdk)")

    @staticmethod
    def _set_low_latency(port) -> None:
        """Drop the FTDI 16 ms latency timer to 1 ms per-fd (ASYNC_LOW_LATENCY) — no
        root needed, unlike the sysfs latency_timer node. Best-effort."""
        ser = getattr(port, "ser", None)
        try:
            if ser is not None and hasattr(ser, "set_low_latency_mode"):
                ser.set_low_latency_mode(True)
        except Exception:
            pass

    def connected(self) -> bool:
        return self._port is not None

    def close(self) -> None:
        with self._lock:
            if self._port is not None:
                try:
                    self._port.closePort()
                finally:
                    self._port = self._pkt = None

    # ── operations ──────────────────────────────────────────────────────────────
    def ping(self, dxl_id: int) -> Optional[int]:
        """Model number if the servo answers, else None. Read-only (no motion)."""
        if self._port is None:
            return None
        with self._lock:
            try:
                model, comm, _err = self._pkt.ping(self._port, dxl_id)
            except Exception as e:
                self._log(f"[dxl] ping error id={dxl_id}: {e}")
                self.close()
                return None
        return model if comm == self._sdk.COMM_SUCCESS else None

    def read_all(self, ids) -> Dict[int, dict]:
        """Present current/velocity/position/voltage/temperature for EVERY id in ONE
        GroupSyncRead transaction (this is what makes the neck run at ~60 Hz). Missing
        responders are simply absent from the returned dict."""
        out: Dict[int, dict] = {}
        if self._port is None:
            return out
        with self._lock:
            try:
                gsr = self._sdk.GroupSyncRead(self._port, self._pkt,
                                              ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK)
                for i in ids:
                    gsr.addParam(i)
                comm = gsr.txRxPacket()
                if comm != self._sdk.COMM_SUCCESS:
                    return out
                for i in ids:
                    if not gsr.isAvailable(i, ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK):
                        continue
                    cur = gsr.getData(i, 126, 2)
                    vel = gsr.getData(i, 128, 4)
                    pos = gsr.getData(i, 132, 4)
                    volt = gsr.getData(i, 144, 2)
                    temp = gsr.getData(i, 146, 1)
                    out[i] = dict(
                        current_ma=_s16(cur) * _CUR_LSB_MA,
                        velocity=_s32(vel) * _VEL_LSB_RAD_S,
                        position=_s32(pos) * _RAD_PER_TICK,
                        voltage=volt / 10.0,
                        temperature=float(temp),
                    )
            except Exception as e:                        # wire fault -> drop the bus
                self._log(f"[dxl] read_all error: {e}")
                self.close()
        return out

    def read_block(self, dxl_id: int) -> Optional[dict]:
        """Single-servo present block (compat + scan())."""
        return self.read_all([dxl_id]).get(dxl_id)

    def write_position_p_gain(self, dxl_id: int, gain: int) -> bool:
        """Position P Gain (reg 84). The XC330 ships ≈900, which is far stiffer than a
        ~0.5 kg camera head needs (operator: 'neck too stiff', 2026-07-29) — the neck
        group softens it on every (re)connect. Fire-and-forget semantics like the other
        setup writes."""
        if self._port is None:
            return False
        with self._lock:
            try:
                comm, err = self._pkt.write2ByteTxRx(
                    self._port, dxl_id, ADDR_POSITION_P_GAIN, int(gain))
            except Exception as e:
                self._log(f"[dxl] p-gain write error id={dxl_id}: {e}")
                return False
        return comm == self._sdk.COMM_SUCCESS and err == 0

    def write_torque(self, dxl_id: int, on: bool) -> bool:
        """Torque Enable, ACK'd. Returns True on a clean write."""
        if self._port is None:
            return False
        with self._lock:
            try:
                comm, err = self._pkt.write1ByteTxRx(
                    self._port, dxl_id, ADDR_TORQUE_ENABLE, 1 if on else 0)
            except Exception as e:
                self._log(f"[dxl] write_torque error id={dxl_id}: {e}")
                self.close()
                return False
        if comm != self._sdk.COMM_SUCCESS:
            self._log(f"[dxl] torque write id={dxl_id} comm={comm}")
            return False
        if err != 0:
            self._log(f"[dxl] torque write id={dxl_id} hw-error={err}")
        return True

    def hardware_error(self, dxl_id: int) -> Optional[int]:
        """Hardware Error Status (reg 70): 0 = healthy, else a fault bitmask
        (overload/over-temp/voltage). None if the read fails."""
        if self._port is None:
            return None
        with self._lock:
            try:
                val, comm, _err = self._pkt.read1ByteTxRx(
                    self._port, dxl_id, ADDR_HARDWARE_ERROR)
            except Exception as e:
                self._log(f"[dxl] hw-error read id={dxl_id}: {e}")
                self.close()
                return None
        return val if comm == self._sdk.COMM_SUCCESS else None

    def reboot(self, dxl_id: int) -> None:
        """REBOOT instruction — clears a latched Hardware Error and boots the servo
        torque-OFF. The only way (short of a power cycle) to recover a faulted servo."""
        if self._port is None:
            return
        with self._lock:
            try:
                self._pkt.reboot(self._port, dxl_id)
            except Exception as e:
                self._log(f"[dxl] reboot id={dxl_id}: {e}")
                self.close()

    def write_profile(self, dxl_id: int, velocity: int, accel: int) -> None:
        """Set Position-mode Profile Velocity/Acceleration (RAM, writable live). 0 =
        unlimited → smooth direct tracking of a streamed goal (see _PROFILE_VELOCITY)."""
        if self._port is None:
            return
        with self._lock:
            try:
                self._pkt.write4ByteTxRx(self._port, dxl_id,
                                         ADDR_PROFILE_ACCELERATION, int(accel) & 0xFFFFFFFF)
                self._pkt.write4ByteTxRx(self._port, dxl_id,
                                         ADDR_PROFILE_VELOCITY, int(velocity) & 0xFFFFFFFF)
            except Exception as e:
                self._log(f"[dxl] write_profile error id={dxl_id}: {e}")
                self.close()

    def sync_write_goal(self, goals: Mapping[int, int]) -> None:
        """One GroupSyncWrite of Goal Position (ticks) to every id at once."""
        if not goals or self._port is None:
            return
        with self._lock:
            try:
                gsw = self._sdk.GroupSyncWrite(self._port, self._pkt,
                                               ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
                for dxl_id, ticks in goals.items():
                    t = max(0, min(TICKS_PER_REV - 1, int(ticks)))
                    gsw.addParam(dxl_id, bytes([t & 0xFF, (t >> 8) & 0xFF,
                                                (t >> 16) & 0xFF, (t >> 24) & 0xFF]))
                gsw.txPacket()
                gsw.clearParam()
            except Exception as e:
                self._log(f"[dxl] sync_write_goal error: {e}")
                self.close()


# ── the joint-group driver ──────────────────────────────────────────────────────
# A servo is declared dead only after this much CONTINUOUS silence, NOT on a single
# missed read. One GroupSyncRead timeout (comm != SUCCESS -> read_all returns {}) is
# routine when the Jetson preempts the HAL mid-transaction; dropping liveness on ONE
# miss made service() close+reopen+limp the neck every few seconds — the measured
# 1.5 s /body/joint_states gaps and the neck torque-flap during whole-body teleop. The
# grace rides out transient starvation holding the last pose; a real unplug still
# surfaces within grace + the ~1 s service interval. Tunable for a genuinely flaky bus.
_LIVENESS_GRACE_S = float(os.getenv("MABEL_HW_NECK_GRACE_S", "0.5"))
# Reconnect BACKOFF: a genuinely-down bus (unplugged / UNPOWERED servos) is retried with
# an EXPONENTIAL delay (1→2→4→…→cap s), not every ~1 s. Each attempt closes+re-opens the
# port and PINGS the servos, and the dynamixel_sdk ping BUSY-WAITS its read timeout holding
# the GIL — at 1 Hz that starved the OTHER HAL modules (the hands dropped to ~4 Hz when the
# neck lost power). Backing off keeps a DEAD module from stalling the LIVE ones; it still
# recovers within the cap once power/USB returns.
_RECONNECT_MIN_S = float(os.getenv("MABEL_HW_NECK_RECONNECT_MIN_S", "1.0"))
_RECONNECT_MAX_S = float(os.getenv("MABEL_HW_NECK_RECONNECT_MAX_S", "8.0"))


class DynamixelGroup(JointGroupDriver):
    """`JointGroupDriver` over a `DynamixelBus`: canonical joint names mapped to servo
    IDs, present state polled (bulk) in tick(), goals sync-written when torque is on.
    Degrades gracefully — a missing adapter or an unpowered servo leaves the group
    alive and idle (``live`` goes False), and it hot-joins via service()."""

    def __init__(self, id_map: Mapping[str, int], *, device: str, baud: int,
                 serial_id: str = "", logger=None):
        self.names: Tuple[str, ...] = tuple(id_map.keys())
        self._id = {n: int(i) for n, i in id_map.items()}
        self._log = logger or (lambda *a, **k: None)
        self._bus = DynamixelBus(device, baud, serial_id=serial_id, logger=logger)
        self._torque = False
        self._targets: Dict[str, float] = {}          # raw rad, latched
        self._fb: Dict[str, dict] = {}                # last good per-joint block
        self._live: Dict[str, bool] = {n: False for n in self.names}
        self._last_ok: Dict[str, float] = {}          # monotonic t of last good read (grace)
        self._down_logged = False                     # edge-log DOWN/RECONNECTED once
        self._reconnect_at = 0.0                       # monotonic t of next reconnect attempt
        self._reconnect_backoff = _RECONNECT_MIN_S     # grows on repeated failure (backoff)
        # DEDICATED I/O THREAD (matches the arm's openarm-rx): ALL neck serial I/O — the
        # bulk read AND the blocking/backed-off reconnect — runs here, OFF the ROS executor,
        # so a DOWN neck never stalls the other HAL modules (the hands). Daemon: a stuck
        # serial op can't block bridge shutdown either. tick()/service() just keep it alive.
        self._io_thread: Optional[threading.Thread] = None
        self._io_stop = threading.Event()
        self._poll_dt = 1.0 / max(float(os.getenv("MABEL_HW_NECK_POLL_HZ", "60")), 1.0)
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def connect(self) -> None:
        self._bus.connect()                            # raises SubsystemUnavailable
        found = []
        for n, i in self._id.items():
            model = self._bus.ping(i)
            self._live[n] = model is not None
            if model is not None:
                found.append(f"{n}=id{i}:{MODEL_NAMES.get(model, model)}")
        if not any(self._live.values()):
            raise SubsystemUnavailable(
                f"Dynamixel bus open on {self._bus.port} but no servos answered "
                f"(ids {sorted(self._id.values())})")
        # Clear any LATCHED hardware error (0x80 Alert). A servo that faulted — overload
        # from being driven into a stop (e.g. commanded UNCALIBRATED), over-temp, or a
        # voltage dip — REFUSES Torque Enable until rebooted, and reg 70 survives a host
        # reconnect. Reboot clears it and the servo boots torque-OFF (safe). Only a
        # genuinely faulted servo is rebooted.
        for n, i in list(self._id.items()):
            if not self._live[n]:
                continue
            he = self._bus.hardware_error(i)
            if he:
                self._log(f"[dxl] {n} (id{i}) HARDWARE ERROR 0x{he:02x} — rebooting to clear")
                self._bus.reboot(i)
                time.sleep(1.2)                        # let it reboot + re-enumerate
                self._live[n] = self._bus.ping(i) is not None
        # ASSERT the physical Torque-Enable to match our intent on every (re)connect.
        # X-series servos LATCH Torque Enable (it survives a host reconnect while
        # powered), so a previous session that energised the neck would boot it STIFF.
        # `self._torque` is False at bring-up ⇒ guaranteed limp boot; only an explicit
        # /body/torque_enable=true holds it, and a mid-session reconnect restores intent.
        for n, i in self._id.items():
            if self._live[n]:
                # Profile 0 = smooth direct tracking of the 50 Hz goal stream (no
                # per-goal trapezoidal re-plan → the neck jitter/lag fix).
                self._bus.write_profile(i, _PROFILE_VELOCITY, _PROFILE_ACCEL)
                # Soften the servo's internal position loop for the ~0.5 kg head —
                # factory ≈900 is rigid; ~1/3 keeps tracking but yields to a hand.
                self._bus.write_position_p_gain(
                    i, int(os.getenv("MABEL_HW_NECK_DXL_P_GAIN", "300")))
                self._bus.write_torque(i, self._torque)
                self._last_ok[n] = monotonic()         # seed the liveness grace window
        self._log(f"[dxl] neck up ({'torque ON' if self._torque else 'limp'}): "
                  f"{', '.join(found)}")

    def close(self) -> None:
        self._io_stop.set()                            # stop the dedicated I/O thread
        t = self._io_thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._io_thread = None
        self._bus.close()

    def connected(self) -> bool:
        return self._bus.connected() and any(self._live.values())

    # ── write side (raw rad) ─────────────────────────────────────────────────────
    def set_targets(self, targets: Mapping[str, float]) -> None:
        with self._lock:
            for name, value in targets.items():
                if name not in self._id:
                    raise KeyError(f"unknown joint {name!r}; "
                                   f"neck has {', '.join(self.names)}")
                self._targets[name] = float(value)
        # LOW-LATENCY push: write the goal to the servos the MOMENT a command arrives
        # (when energised) instead of waiting up to one tick (~20 ms at 50 Hz, coupled
        # to the loaded ROS executor). /body/command streams at ~50 Hz, so pushing on
        # arrival gives smooth, low-latency neck tracking; tick() still writes as a
        # keepalive + reads. Skipped when limp (no goal stream while free-driving).
        if self._torque:
            self._push_goals()

    def _push_goals(self) -> None:
        """Write the latched raw-rad targets to the live servos as Goal Position."""
        with self._lock:
            goals = {self._id[n]: round(self._targets[n] / _RAD_PER_TICK)
                     for n in self._targets if self._live.get(n)}
        self._bus.sync_write_goal(goals)

    def set_torque(self, on: bool) -> None:
        on = bool(on)
        for name, i in self._id.items():
            if self._live[name]:
                self._bus.write_torque(i, on)
        self._torque = on

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque

    # ── read side (cache; never touches the wire) ────────────────────────────────
    def read(self) -> GroupFeedback:
        with self._lock:
            pos, vel, eff, temp, live = [], [], [], [], []
            for n in self.names:
                b = self._fb.get(n)
                pos.append(b["position"] if b else 0.0)
                vel.append(b["velocity"] if b else 0.0)
                # tau = Kt * I (present current); dir-corrected one layer up.
                eff.append(_TORQUE_CONST_NM_PER_A * b["current_ma"] / 1000.0
                           if b else 0.0)
                temp.append(b["temperature"] if b else 0.0)
                live.append(bool(self._live.get(n)))
            return GroupFeedback(
                names=self.names, position=tuple(pos), velocity=tuple(vel),
                effort=tuple(eff), temperature=tuple(temp), live=tuple(live),
                stamp=monotonic())

    def health(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            out: Dict[str, Dict[str, float]] = {}
            for n in self.names:
                b = self._fb.get(n)
                if b:
                    out[n] = {"torqueNm": round(_TORQUE_CONST_NM_PER_A
                                                 * b["current_ma"] / 1000.0, 4),
                              "tempC": b["temperature"],
                              "currentmA": b["current_ma"], "voltV": b["voltage"]}
            return out

    # ── pacing (ROS timers just keep the I/O thread alive; all serial I/O is on it) ──
    def tick(self) -> None:
        """ROS-timer entry — a NO-OP beyond keeping the dedicated I/O thread alive (mirrors
        openarm.tick()). ALL neck serial I/O runs on `dxl-neck-io`, so a down neck's blocking
        reconnect never stalls the ROS executor / the hands."""
        self._ensure_io()

    def service(self) -> None:
        """ROS-timer entry — keep the I/O thread alive; the reconnect itself runs there."""
        self._ensure_io()

    def _ensure_io(self) -> None:
        if self._io_thread is None or not self._io_thread.is_alive():
            self._io_stop.clear()
            self._io_thread = threading.Thread(target=self._io_loop, name="dxl-neck-io",
                                               daemon=True)
            self._io_thread.start()

    def _io_loop(self) -> None:
        """The neck's own I/O loop (off the ROS executor). Reconnect (backed off) when down,
        else bulk-poll at the poll rate. The serial reads release the GIL, so a live neck
        streams at ~poll_hz and a DEAD one only busy-pings on THIS thread, every backoff
        interval — never on the shared executor that also serves the hands."""
        while not self._io_stop.is_set():
            try:
                self._service_once()                     # reconnect (backoff) / hot-join
                if self._bus.connected() and any(self._live.values()):
                    self._poll_once()                    # bulk GroupSyncRead -> _fb
                    self._io_stop.wait(self._poll_dt)
                else:
                    self._io_stop.wait(0.1)              # down: backoff gates real attempts
            except Exception as e:                       # noqa: BLE001 — never let it die
                self._log(f"[dxl] neck io-loop error (continuing): {e}")
                self._io_stop.wait(0.2)

    def _poll_once(self) -> None:
        """Push goals if energized, then bulk-poll present state (ONE GroupSyncRead)."""
        if not self._bus.connected():
            return
        if self._torque:                               # keepalive goal write (commands
            self._push_goals()                         # also push immediately on arrival)
        live_ids = [i for n, i in self._id.items() if self._live.get(n)]
        blocks = self._bus.read_all(live_ids) if live_ids else {}
        now = monotonic()
        with self._lock:
            for n, i in self._id.items():
                if not self._live.get(n):
                    continue
                b = blocks.get(i)
                if b is not None:
                    self._fb[n] = b
                    self._last_ok[n] = now
                elif now - self._last_ok.get(n, now) > _LIVENESS_GRACE_S:
                    # SUSTAINED silence only — a single missed GroupSyncRead (transient
                    # timeout under CPU-starvation) holds the last pose and stays live, so
                    # we don't close/reopen/limp-flap the neck. A real unplug crosses the
                    # grace and drops here → the loop's _service_once reconnects.
                    self._live[n] = False

    def _service_once(self) -> None:
        """DETECT a down bus and auto-RECONNECT it (exponential backoff), else re-ping the
        servos that dropped. Runs on the dedicated io thread, so the blocking serial open +
        busy pings never touch the ROS executor. Handles a wire fault (port closed) and a
        SILENT unplug (port open but every servo quiet): either counts as DOWN → close +
        re-resolve (pinned by FTDI serial, so a replug onto a new /dev path recovers)."""
        down = (not self._bus.connected()) or (not any(self._live.values()))
        if down:
            now = monotonic()
            if now < self._reconnect_at:
                return                                 # backing off — don't hog the GIL/bus
            self._reconnect_at = now + self._reconnect_backoff
            self._reconnect_backoff = min(self._reconnect_backoff * 2.0, _RECONNECT_MAX_S)
            if not self._down_logged:
                self._log("[dxl] neck DOWN — no servo answering (unplugged / "
                          "unpowered?); reconnecting with backoff (1→8 s) so it can't "
                          "stall the other modules.")
                self._down_logged = True
            self._bus.close()
            try:
                self.connect()                         # re-resolve + re-ping + assert intent
            except SubsystemUnavailable:
                return                                 # still gone; retry after the backoff
            except Exception as e:                     # noqa: BLE001 — never crash the timer
                self._log(f"[dxl] neck reconnect error (backing off): {e}")
                return
            self._reconnect_backoff = _RECONNECT_MIN_S  # SUCCESS — reset the backoff
            self._reconnect_at = 0.0
            if self._torque:                           # re-assert intent on rejoin
                self.set_torque(True)
            self._down_logged = False
            self._log("[dxl] neck RECONNECTED — servos answering again.")
            return
        for n, i in self._id.items():                  # hot-join a single dropped servo
            if not self._live.get(n):
                model = self._bus.ping(i)
                if model is not None:
                    self._live[n] = True
                    self._last_ok[n] = monotonic()
                    self._log(f"[dxl] {n} (id{i}) REJOINED")


def scan(device: str = "", *, serial_id: str = "",
         bauds=(1000000, 57600, 2000000, 3000000, 115200),
         id_range=range(0, 21), logger=None) -> dict:
    """Read-only bus probe for `experiments/exp_neck.py` and bring-up: for each baud,
    ping and read state for every responder. Moves NOTHING."""
    log = logger or (lambda *a, **k: None)
    path = resolve_device(device, serial_id)
    if not path:
        return {"device": None, "error": "no adapter found"}
    result = {"device": path, "bauds": {}}
    for baud in bauds:
        bus = DynamixelBus(path, baud, logger=log)
        try:
            bus.connect()
        except SubsystemUnavailable as e:
            result["bauds"][baud] = {"error": str(e)}
            continue
        motors = {}
        for i in id_range:
            model = bus.ping(i)
            if model is not None:
                st = bus.read_block(i) or {}
                motors[i] = {"model": MODEL_NAMES.get(model, model), "model_no": model,
                             **{k: round(v, 3) if isinstance(v, float) else v
                                for k, v in st.items()}}
        bus.close()
        if motors:
            result["bauds"][baud] = motors
    return result
