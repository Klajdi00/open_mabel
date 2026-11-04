#!/usr/bin/env python3
"""
MABEL OpenArm — DaMiao 7-Joint Teleop & Calibration GUI

Matches the exact architecture and design tokens of the Lift and ORCA Hand GUIs.

Backends
--------
  serial   : DaMiao USB-to-CAN adapter — the official DM_Control_Python serial
             protocol (921600 baud, 0x55/0xAA framing). Most reliable on macOS.
  gs_usb   : CANable / candleLight over USB via libusb/pyusb (no SocketCAN).
  socketcan: Linux kernel CAN (``--channel can0``).

Default backend: on macOS, ``serial`` if a DaMiao serial adapter is detected,
else ``gs_usb``; on Linux, ``socketcan``. All three share the same DaMiao CAN
payload + feedback parsing — only the physical transport differs.

Usage
-----
  python scripts/teleop_gui.py                          # auto backend
  python scripts/teleop_gui.py --interface serial       # DaMiao serial adapter
  python scripts/teleop_gui.py --interface serial --port /dev/cu.usbmodem1101
  python scripts/teleop_gui.py --interface gs_usb       # force CANable/gs_usb
  python scripts/teleop_gui.py --channel can1           # Linux SocketCAN

Keyboard shortcuts (no Entry/Text focused)
------------------------------------------
  ← →     Nudge selected joint by step
  ↑ ↓     Double / halve step size
  S        Scan bus      Q  Quit
"""

import argparse, can, signal, struct, subprocess, sys, threading, time, tkinter as tk
from pathlib import Path

_DARWIN = sys.platform == "darwin"


def _ensure_libusb_backend():
    """Return a pyusb libusb1 backend, hunting Homebrew paths that
    ``ctypes.util.find_library`` misses on Apple Silicon. Returns None on Linux
    or if pyusb isn't needed — gs_usb falls back to its own discovery then."""
    try:
        import usb.backend.libusb1
    except Exception:
        return None
    be = usb.backend.libusb1.get_backend()
    if be is not None:
        return be
    import os
    for cand in ("/opt/homebrew/lib/libusb-1.0.dylib",   # Apple Silicon brew
                 "/usr/local/lib/libusb-1.0.dylib",       # Intel brew
                 "/opt/local/lib/libusb-1.0.dylib"):      # MacPorts
        if os.path.exists(cand):
            be = usb.backend.libusb1.get_backend(find_library=lambda _: cand)
            if be is not None:
                return be
    return None

_STYLE = Path(__file__).resolve().parents[2] / "lift" / "gui"
sys.path.insert(0, str(_STYLE))
from style import *
from style import _FONT_BASE

WIN_W     = 1280
WIN_H     = 820
SIDEBAR_W = 280
CTRL_HZ   = 50
GLOBAL_ID = 0x7FF

# ── Joint definitions (motor ID 1→7, top→bottom of arm) ──────────────────────
#   Theoretical limits from simulation/mabel_mujoco/models/right_arm.xml
# OpenArm = 7 arm joints (CAN id 1→7) + Torso (id 8). Edit a Torso CAN id here if
# it differs on your bus; you can also pick its motor type live in the GUI.
JOINT_INFO = [
    dict(mid=1, name="Shoulder Rot",   profile="DM-J8009P", lo=-3.4907, hi= 1.3963),
    dict(mid=2, name="Shoulder Pitch", profile="DM-J8009P", lo=-0.1745, hi= 3.3161),
    dict(mid=3, name="Shoulder Roll",  profile="DM-J4340",  lo=-1.5708, hi= 1.5708),
    dict(mid=4, name="Elbow Flex",     profile="DM-J4340",  lo=-0.7735, hi= 1.6700),
    dict(mid=5, name="Wrist Pitch",    profile="DM-J4310",  lo=-1.5708, hi= 1.5708),
    dict(mid=6, name="Wrist Roll",     profile="DM-J4310",  lo=-0.7854, hi= 0.7854),
    dict(mid=7, name="Wrist Rot",      profile="DM-J4310",  lo=-1.5708, hi= 1.5708),
    dict(mid=8, name="Torso",          profile="DM-J10010L", lo=-3.1416, hi= 3.1416),
]
JOINT_BY_ID = {j["mid"]: j for j in JOINT_INFO}

# Per-joint position-hold PD defaults {mid: (kp, kd)}. The DM-J4310 wrist joints
# J5/J6/J7 are low-inertia and buzz/oscillate at the heavier joints' stiffness, so
# they get a much softer Kp and a touch less Kd (high Kd on their noisy velocity
# only adds to the shake). Everything else keeps the original 30 / 1.0.
DEFAULT_PD = (30.0, 1.0)
PD_DEFAULTS = {
    5: (14.0, 0.8),   # Wrist Pitch (DM-J4310)
    6: (10.0, 0.8),   # Wrist Roll  (DM-J4310)
    7: (10.0, 0.8),   # Wrist Rot   (DM-J4310)
}

# Full DaMiao motor catalog (p_max/v_max/t_max/kp_max/kd_max). v_max/t_max from the
# official DM_Control_Python Limit_Param; p_max=12.566 (4π) and kp/kd caps are the
# DaMiao MIT defaults. Adding a type here makes it selectable in the GUI.
PROFILES = {
    "DM-J4310":   dict(p_max=12.5,   v_max=30.0,  t_max=10.0,  kp_max=500.0, kd_max=5.0),
    "DM-J4340":   dict(p_max=12.566, v_max=10.4,  t_max=27.0,  kp_max=500.0, kd_max=5.0),
    "DM-J6006":   dict(p_max=12.566, v_max=45.0,  t_max=20.0,  kp_max=500.0, kd_max=5.0),
    "DM-J8006":   dict(p_max=12.566, v_max=45.0,  t_max=40.0,  kp_max=500.0, kd_max=5.0),
    "DM-J8009P":  dict(p_max=12.566, v_max=33.5,  t_max=40.0,  kp_max=500.0, kd_max=5.0),
    "DM-J10010L": dict(p_max=12.566, v_max=25.0,  t_max=200.0, kp_max=500.0, kd_max=5.0),
    "DM-J10010":  dict(p_max=12.566, v_max=20.0,  t_max=200.0, kp_max=500.0, kd_max=5.0),
    "DM-J10422P": dict(p_max=12.566, v_max=12.5,  t_max=400.0, kp_max=500.0, kd_max=5.0),
    "DM-H3510":   dict(p_max=12.566, v_max=280.0, t_max=1.0,   kp_max=500.0, kd_max=5.0),
    "DM-J3507":   dict(p_max=12.566, v_max=48.0,  t_max=3.0,   kp_max=500.0, kd_max=5.0),
    "DM-G6220":   dict(p_max=12.566, v_max=45.0,  t_max=10.0,  kp_max=500.0, kd_max=5.0),
}
DEFAULT_PROFILE = "DM-J4310"
DEFAULT_LIM     = PROFILES[DEFAULT_PROFILE]

ERR_NAMES = {
    0x0: "OK",        0x2: "ROTOR ENC",  0x3: "OUTPUT ENC",
    0x8: "OVER VOLT", 0x9: "UNDER VOLT", 0xA: "OVER CURR",
    0xB: "MOS TEMP",  0xC: "COIL TEMP",  0xD: "COMM LOSS",
    0xE: "OVERLOAD",
}

MODE_DESC = {
    "Position":  "τ = Kp·(p_des − p)  +  Kd·(0 − v)         stiff position hold",
    "Velocity":  "τ = Kd·(v_des − v)                          smooth velocity tracking",
    "Torque":    "τ = t_ff                                     direct force control",
    "Impedance": "τ = Kp·(p_eq − p)  +  Kd·(0 − v)  +  t_ff   compliant + gravity comp",
}

# The four official DaMiao control modes (DM_Control_Python Control_Type), with the
# CTRL_MODE register value (RID 10) each maps to. `ctrl_mode` selects the CAN frame
# format + arbitration-id offset; an internal MIT sub-mode (below) keeps send_mit
# and the calibration sweep working unchanged.
CTRL_MODES = {"MIT": 1, "Pos-Vel": 2, "Vel": 3, "Force": 4}
CTRL_MODE_DESC = {
    "MIT":     "Full impedance:  τ = Kp·(q_des−q) + Kd·(dq_des−v) + τ_ff",
    "Pos-Vel": "Position + velocity setpoint  (motor's internal trajectory loop)",
    "Vel":     "Velocity setpoint  (motor's internal velocity loop)",
    "Force":   "Position + velocity + current-limit  (force/position hybrid)",
}
# ctrl_mode → send_mit's internal sub-mode (so calibration's velocity sweep and the
# stall-stop safety keep working; MIT = full impedance via the position branch).
_CTRL_TO_SUBMODE = {"MIT": "impedance", "Pos-Vel": "position",
                    "Vel": "velocity", "Force": "position"}

# Global calibrated joint limits {mid: (lo, hi)}
_joint_cal: dict[int, tuple[float, float]] = {}


# ── Motor state ───────────────────────────────────────────────────────────────
class Motor:
    def __init__(self, mid: int):
        self.mid   = mid
        ji         = JOINT_BY_ID.get(mid, {})
        prof_name  = ji.get("profile", DEFAULT_PROFILE)
        self.profile = prof_name
        self.lim   = dict(PROFILES.get(prof_name, DEFAULT_LIM))

        self.pos = self.vel = self.tau = 0.0
        self.t_mos   = 0.0
        self.err     = 0
        self.t_fb    = 0.0
        # MST_ID (master/feedback CAN id) — learned live from the arbitration id of
        # the motor's reply frames (RID 7). None until the motor has answered once.
        self.mst_id  = None
        # enabled == user has requested torque control (0xFC sent, MIT frames
        # streaming). Telemetry does NOT depend on this — it comes from continuous
        # 0xCC refresh polling, so a found motor streams whether enabled or not.
        self.enabled    = False

        self.ctrl_mode  = "MIT"      # official DaMiao mode (CTRL_MODES)
        self.mode       = "position"  # MIT sub-behaviour used by send_mit / calibration
        self.pos_target = None
        self.vel_target = 0.0
        self.cur_target = 0.0        # current setpoint (0–10000) for Force mode
        self.kp, self.kd = PD_DEFAULTS.get(mid, DEFAULT_PD)
        self.vel_ff     = 0.0
        self.t_ff       = 0.0
        self.step         = 0.02
        self.vel_step     = 0.5
        self._just_enabled = False  # force pos_target=m.pos on the very first MIT packet
        self._stall_since  = None   # perf_counter timestamp when stall started
        self._enabled_at   = None   # perf_counter when enabled — grace before auto-disable

    def parse(self, d: bytes):
        self.err   = (d[0] >> 4) & 0xF
        p_int = (d[1] << 8) | d[2]
        v_int = (d[3] << 4) | (d[4] >> 4)
        t_int = ((d[4] & 0xF) << 8) | d[5]
        lim   = self.lim
        self.pos   = _u2f(p_int, -lim["p_max"], lim["p_max"], 16)
        self.vel   = _u2f(v_int, -lim["v_max"], lim["v_max"], 12)
        self.tau   = _u2f(t_int, -lim["t_max"], lim["t_max"], 12)
        self.t_mos = float(d[6]) if len(d) > 6 else 0.0
        self.t_fb  = time.perf_counter()
        if self.pos_target is None:
            self.pos_target = self.pos

    @property
    def stale(self):
        # Feedback silent > 0.5 s — motor likely disconnected
        return self.t_fb > 0 and time.perf_counter() - self.t_fb > 0.5

    @property
    def found(self):
        # Motor has ever responded; resets after 5 s silence (hot-swap detection)
        return self.t_fb > 0 and time.perf_counter() - self.t_fb < 5.0


def _f32le(x):
    """4-byte little-endian IEEE-754 float — the DaMiao serial/CAN setpoint format."""
    return struct.pack("<f", float(x))

def _u2f(x, lo, hi, bits):
    return x * (hi - lo) / ((1 << bits) - 1) + lo

def _f2u(x, lo, hi, bits):
    return int(max(0, min((1 << bits) - 1, (x - lo) * ((1 << bits) - 1) / (hi - lo))))


# ── CAN transports ────────────────────────────────────────────────────────────
# The DaMiao CAN payload + feedback format is identical across adapters; only the
# physical transport differs. Each transport exposes the same tiny interface:
#   send(arb_id:int, data:bytes)        — transmit one standard (11-bit) CAN frame
#   poll(timeout)->(arb_id, data)|None  — receive one frame (returns the 8 payload
#                                         bytes, so the protocol layer is shared)
#   close()                             — release the device

def _find_serial_port():
    """First likely DaMiao USB-to-CAN serial adapter on macOS/Linux, else None."""
    import glob
    pats = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
            "/dev/cu.SLAB_USBtoUART*", "/dev/ttyACM*", "/dev/ttyUSB*")
    for pat in pats:
        hits = sorted(g for g in glob.glob(pat) if "debug-console" not in g)
        if hits:
            return hits[0]
    return None


class _PyCanTransport:
    """SocketCAN (Linux) or gs_usb (CANable/candleLight over USB) via python-can."""
    def __init__(self, interface, channel, bitrate):
        self.interface = interface
        if interface == "gs_usb":
            _ensure_libusb_backend()   # nudge pyusb toward Homebrew libusb on macOS
            # python-can's gs_usb open issues libusb_reset_device(), which this CAN
            # engine needs to bring its channel up — but the reset briefly
            # re-enumerates the adapter, so the first open can race and fail with
            # "No such device"/"Access denied". Retry a few times to ride it out.
            last = None
            for _ in range(5):
                try:
                    self._bus = can.interface.Bus(
                        interface="gs_usb", channel=channel, index=0, bitrate=bitrate)
                    break
                except Exception as e:
                    last = e
                    time.sleep(0.8)
            else:
                raise last
        else:
            self._bus = can.interface.Bus(channel=channel, interface="socketcan")

    def send(self, arb_id, data):
        self._bus.send(can.Message(arbitration_id=arb_id, data=bytes(data),
                                   is_extended_id=False))

    def poll(self, timeout=0.5):
        m = self._bus.recv(timeout=timeout)
        if m is None or not m.is_rx:
            return None
        return (m.arbitration_id, bytes(m.data))

    def close(self):
        self._bus.shutdown()


class _SerialDamiaoTransport:
    """DaMiao USB-to-CAN serial adapter — the official DM_Control_Python protocol.

    Each CAN frame is wrapped in a 30-byte serial packet (0x55 0xAA header, CAN id
    at [13:15], 8 data bytes at [21:29]); replies arrive as 16-byte packets framed
    by 0xAA … 0x55 with the CAN id little-endian at [3:7] and payload at [7:15].
    This serial path is what the official library uses to run cleanly on macOS.
    """
    interface = "serial"
    # Byte template lifted verbatim from DM_CAN.MotorControl.send_data_frame.
    _TX = bytes([0x55, 0xAA, 0x1e, 0x03, 0x01, 0, 0, 0, 0x0a, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def __init__(self, port, baud=921600):
        import serial   # pyserial (lazy import — only needed for this backend)
        self._ser = serial.Serial(port, baud, timeout=0)
        self._buf = bytearray()
        self.port = port

    def send(self, arb_id, data):
        f = bytearray(self._TX)
        f[13] = arb_id & 0xFF
        f[14] = (arb_id >> 8) & 0xFF
        f[21:29] = bytes(data)[:8].ljust(8, b"\x00")
        self._ser.write(bytes(f))

    def poll(self, timeout=0.5):
        frame = self._extract()
        if frame is not None:
            return frame
        self._ser.timeout = timeout
        chunk = self._ser.read(64)            # blocks up to `timeout` for the 1st byte
        if chunk:
            self._buf.extend(chunk)
        n = self._ser.in_waiting               # then drain whatever else arrived
        if n:
            self._buf.extend(self._ser.read(n))
        return self._extract()

    def _extract(self):
        b = self._buf
        while len(b) >= 16:
            if b[0] == 0xAA and b[15] == 0x55:
                fr = bytes(b[:16]); del b[:16]
                arb = fr[3] | (fr[4] << 8) | (fr[5] << 16) | (fr[6] << 24)
                return (arb, fr[7:15])
            del b[:1]                          # resync: drop one byte, try again
        return None

    def close(self):
        self._ser.close()


# ── CAN bus layer ─────────────────────────────────────────────────────────────
class DamiaoBus:
    def __init__(self, channel: str = "can0", interface: str | None = None,
                 bitrate: int = 1_000_000, port: str | None = None):
        # Auto-select backend by platform unless explicitly forced.
        self._interface  = interface or ("gs_usb" if _DARWIN else "socketcan")
        self._gs_usb     = self._interface == "gs_usb"
        self._is_serial  = self._interface == "serial"
        self._channel    = channel
        self._bitrate    = bitrate
        self._port       = port
        self._recovering = False
        self._connected  = True   # toggled by the toolbar Connect/Disconnect button
        self._tx_err     = 0
        try:
            self._transport = self._make_transport()
        except Exception as e:
            if self._is_serial:
                sys.exit(f"Cannot open DaMiao serial adapter ({self._port}): {e}\n"
                         "  • Plug in the DaMiao USB-to-CAN adapter (shows up as a\n"
                         "    /dev/cu.usb* serial port), or pass --port <device>.\n"
                         "  • Install deps:  pip install pyserial")
            if self._gs_usb:
                sys.exit(f"Cannot open gs_usb CAN device: {e}\n"
                         "  • Plug in the CANable/candleLight adapter (USB).\n"
                         "  • Install deps:  brew install libusb && "
                         "pip install python-can gs_usb pyusb")
            sys.exit(f"Cannot open {channel}: {e}\n"
                     "Run:  sudo ip link set can0 up type can bitrate 1000000")
        # Pre-populate all 7 arm joints so the sidebar is always complete
        self.motors: dict[int, Motor] = {j["mid"]: Motor(j["mid"]) for j in JOINT_INFO}
        self._lock  = threading.Lock()
        self._log   = []
        self._stop  = threading.Event()
        threading.Thread(target=self._rx_loop, daemon=True).start()

    def _make_transport(self):
        """Open (or reopen) the transport for the active backend."""
        if self._is_serial:
            port = self._port or _find_serial_port()
            if not port:
                raise RuntimeError("no serial adapter found (pass --port <device>)")
            self._port = port
            return _SerialDamiaoTransport(port)
        return _PyCanTransport(self._interface, self._channel, self._bitrate)

    def _recover_bus(self):
        """Reopen the transport after repeated TX/RX errors (bus-off state, which
        commonly happens when new motors are connected while powered).

        On SocketCAN this bounces the interface down/up first; on gs_usb / serial
        the USB device is simply closed and reopened (no `ip link` on macOS).
        """
        if self._recovering:
            return
        self._recovering = True
        self.log(f"CAN adapter not responding — resetting {self._channel}…")
        try:
            try:
                self._transport.close()
            except Exception:
                pass

            if not self._gs_usb and not self._is_serial:
                for cmd in (
                    ["sudo", "ip", "link", "set", self._channel, "down"],
                    ["sudo", "ip", "link", "set", self._channel, "up",
                     "type", "can", "bitrate", str(self._bitrate)],
                ):
                    r = subprocess.run(cmd, timeout=5, capture_output=True)
                    if r.returncode != 0:
                        self.log(f"ip link cmd failed: {r.stderr.decode().strip()}")

            time.sleep(0.6)
            self._transport = self._make_transport()
            self._tx_err  = 0
            self.log(f"{self._channel} reset OK — scanning for motors…")
            threading.Thread(target=self.scan, daemon=True).start()
        except Exception as e:
            self.log(f"Recovery failed: {e}  (power-cycle the adapter if this persists)")
        finally:
            self._recovering = False

    def reset_bus(self):
        """Public — called by the Setup panel's Reset Interface button."""
        if not self._recovering:
            threading.Thread(target=self._recover_bus, daemon=True).start()

    # ── Connect / Disconnect (toolbar button) ──────────────────────────────────
    def connect(self):
        """Reopen the transport and resume polling. No-op if already connected."""
        if self._connected:
            return
        try:
            self._transport = self._make_transport()
            self._connected = True
            self._tx_err    = 0
            self.log(f"Connected to {self._channel} — scanning for motors…")
            threading.Thread(target=self.scan, daemon=True).start()
        except Exception as e:
            self.log(f"Connect failed: {e}")

    def disconnect(self):
        """Stop all bus traffic and release the adapter, but keep the GUI alive."""
        if not self._connected:
            return
        self._connected = False
        with self._lock:
            for m in self.motors.values():
                m.enabled = False
        try:
            self._transport.close()
        except Exception:
            pass
        self.log("Disconnected — adapter released")

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"{ts}  {msg}")
        self._log = self._log[-100:]

    def log_tail(self, n=25):
        return self._log[-n:]

    def _rx_loop(self):
        rx_err = 0
        while not self._stop.is_set():
            if self._recovering or not self._connected:
                time.sleep(0.1)
                continue
            try:
                frame = self._transport.poll(timeout=0.5)
                rx_err = 0
            except Exception as e:
                rx_err += 1
                self.log(f"CAN RX error ({rx_err}): {e}")
                if rx_err >= 3:
                    rx_err = 0
                    threading.Thread(target=self._recover_bus, daemon=True).start()
                else:
                    time.sleep(0.3)
                continue
            if frame is None:
                continue
            _arb, d = frame
            if len(d) < 6:
                continue
            if d[2] == 0x33:
                # Param-read response (0x33) — just register the motor as present.
                mid = d[0] | (d[1] << 8)
                with self._lock:
                    if mid not in self.motors:
                        self.motors[mid] = Motor(mid)
                    prev_found = self.motors[mid].found
                    self.motors[mid].t_fb = time.perf_counter()
                    self.motors[mid].mst_id = _arb   # reply CAN id = MST_ID
                if not prev_found:
                    self.log(f"Motor {mid} found — streaming telemetry")
            else:
                # Standard feedback frame (from 0xCC refresh OR MIT control) —
                # parse pos/vel/tau and keep the motor live. No enable needed.
                mid = d[0] & 0x0F
                with self._lock:
                    m = self.motors.get(mid)
                    prev_found = m.found if m else False
                    if m:
                        m.parse(d)
                        m.mst_id = _arb          # feedback CAN id = MST_ID
                if m and not prev_found:
                    self.log(f"Motor {mid} found — streaming telemetry")

    def scan(self):
        if self._recovering or not self._connected:
            return
        self.log("Scanning bus…")
        try:
            for mid in range(1, 16):
                lo, hi = mid & 0xFF, (mid >> 8) & 0xFF
                # 0xCC = refresh_motor_status: the DaMiao status query that
                # reliably makes a motor emit a feedback frame. Some firmware
                # (e.g. this DM-J4310) ignores the 0x33 register-read query, so
                # send both — 0xCC for detection, 0x33 for back-compat.
                self._transport.send(GLOBAL_ID, [lo, hi, 0xCC, 0, 0, 0, 0, 0])
                self._transport.send(GLOBAL_ID, [lo, hi, 0x33, 0x00])
                time.sleep(0.015)
            self.log("Scan complete")
        except Exception as e:
            self.log(f"Scan error: {e}  —  check CAN interface")

    def refresh(self, mid: int):
        """Telemetry poll — send 0xCC (refresh_motor_status) so the motor emits a
        feedback frame. No torque, no MIT mode, no enable: this is exactly the
        official DM_Control refresh_motor_status, and it is what keeps a motor's
        pos/vel/tau live on the bus whether or not it is enabled for torque."""
        if self._recovering or not self._connected:
            return
        try:
            self._transport.send(
                GLOBAL_ID, [mid & 0xFF, (mid >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])
        except Exception as e:
            self._tx_err += 1
            if self._tx_err >= 5:
                self._tx_err = 0
                threading.Thread(target=self._recover_bus, daemon=True).start()

    def _cmd(self, mid: int, byte: int):
        if self._recovering or not self._connected:
            self.log(f"CAN busy / disconnected — retry in a moment")
            return
        try:
            self._transport.send(mid, [0xFF]*7 + [byte])
            self._tx_err = 0
        except Exception as e:
            self.log(f"CAN TX error (cmd 0x{byte:02X} → {mid}): {e}")
            self._tx_err += 1
            if self._tx_err >= 3:
                self._tx_err = 0
                threading.Thread(target=self._recover_bus, daemon=True).start()

    def enable(self, mid: int):
        """Enter MIT mode (0xFC) so the motor accepts torque commands. This is the
        ONLY thing enable does — telemetry already streams via refresh() poll."""
        self._cmd(mid, 0xFC)
        with self._lock:
            if mid in self.motors:
                m = self.motors[mid]
                m.enabled      = True
                m.pos_target   = m.pos  # hold current position — never lunge
                m.vel_target   = 0.0
                m._just_enabled = True  # send_mit will re-snap on the first packet
                m._enabled_at   = time.perf_counter()  # start auto-disable grace
        self.log(f"Motor {mid} ({JOINT_BY_ID.get(mid, {}).get('name', '')}) enabled — torque ON")

    def disable(self, mid: int):
        """Exit MIT mode (0xFD): torque OFF, motor free/backdrivable. Telemetry
        keeps streaming via the refresh() poll — the motor stays on the bus."""
        self._cmd(mid, 0xFD)
        with self._lock:
            if mid in self.motors:
                m = self.motors[mid]
                m.enabled    = False
                m.vel_target = 0.0
                m._stall_since = None
        self.log(f"Motor {mid} disabled — torque OFF, still streaming")

    # E-STOP path: identical to disable (0xFD frees the motor); kept as a distinct
    # name so the UI intent ("kill torque now") stays explicit.
    hard_disable = disable

    def clear_faults(self, mid: int):
        self._cmd(mid, 0xFB)
        self.log(f"Motor {mid} faults cleared")

    def set_zero(self, mid: int):
        self._cmd(mid, 0xFE)
        with self._lock:
            if mid in self.motors:
                self.motors[mid].pos_target = 0.0
        self.log(f"Motor {mid} zeroed")

    def send_mit(self, m: Motor):
        # On the first packet after enable, always target the actual physical position
        # regardless of what the slider or any callback may have written to pos_target.
        if m._just_enabled:
            m._just_enabled = False
            m.pos_target    = m.pos
            m.vel_target    = 0.0
        lim = m.lim
        if not m.enabled:
            # Zero-torque MIT: motor stays free/backdrivable but replies with
            # pos/vel/tau feedback every cycle so telemetry stays live.
            p_des = v_des = kp = kd = t_ff = 0.0
        elif m.mode == "velocity":
            p_des, v_des, kp, kd, t_ff = 0.0, m.vel_target, 0.0, m.kd, m.t_ff
        elif m.mode == "torque":
            p_des, v_des, kp, kd, t_ff = 0.0, 0.0, 0.0, 0.0, m.t_ff
        else:  # position or impedance
            if m.pos_target is None:
                return
            p_des, v_des, kp, kd, t_ff = m.pos_target, m.vel_ff, m.kp, m.kd, m.t_ff

        pi = _f2u(p_des, -lim["p_max"],  lim["p_max"],  16)
        vi = _f2u(v_des, -lim["v_max"],  lim["v_max"],  12)
        ki = _f2u(kp,    0.0,            lim["kp_max"], 12)
        di = _f2u(kd,    0.0,            lim["kd_max"], 12)
        ti = _f2u(t_ff,  -lim["t_max"],  lim["t_max"],  12)
        if self._recovering or not self._connected:
            return
        try:
            self._transport.send(m.mid,
                [(pi>>8)&0xFF, pi&0xFF,
                 (vi>>4)&0xFF, ((vi&0xF)<<4)|((ki>>8)&0xF), ki&0xFF,
                 (di>>4)&0xFF, ((di&0xF)<<4)|((ti>>8)&0xF), ti&0xFF])
            self._tx_err = 0
        except Exception as e:
            self._tx_err += 1
            self.log(f"MIT TX error J{m.mid}: {e}")
            if self._tx_err >= 3:
                self._tx_err = 0
                threading.Thread(target=self._recover_bus, daemon=True).start()

    def _snap_on_enable(self, m: Motor):
        """First packet after enable: target the actual physical position so the
        motor never lunges — shared by every control mode."""
        if m._just_enabled:
            m._just_enabled = False
            m.pos_target    = m.pos
            m.vel_target    = 0.0

    def _tx_command(self, arb_id, data, mid):
        if self._recovering or not self._connected:
            return
        try:
            self._transport.send(arb_id, data)
            self._tx_err = 0
        except Exception as e:
            self._tx_err += 1
            self.log(f"TX error J{mid}: {e}")
            if self._tx_err >= 3:
                self._tx_err = 0
                threading.Thread(target=self._recover_bus, daemon=True).start()

    def send_pos_vel(self, m: Motor):
        """POS_VEL mode — arbitration id 0x100+id, [pos(f32le), vel(f32le)]."""
        self._snap_on_enable(m)
        self._tx_command(0x100 + m.mid,
                         _f32le(m.pos_target or 0.0) + _f32le(m.vel_target), m.mid)

    def send_vel(self, m: Motor):
        """VEL mode — arbitration id 0x200+id, [vel(f32le)]."""
        self._snap_on_enable(m)
        self._tx_command(0x200 + m.mid, _f32le(m.vel_target), m.mid)

    def send_force(self, m: Motor):
        """Torque_Pos (force/position) — id 0x300+id, [pos(f32le), vel(u16le), cur(u16le)]."""
        self._snap_on_enable(m)
        v = int(max(0, min(65535, abs(int(m.vel_target)))))
        c = int(max(0, min(10000, int(m.cur_target))))
        data = _f32le(m.pos_target or 0.0) + bytes([v & 0xFF, (v >> 8) & 0xFF,
                                                    c & 0xFF, (c >> 8) & 0xFF])
        self._tx_command(0x300 + m.mid, data, m.mid)

    def send_command(self, m: Motor):
        """Dispatch to the correct DaMiao control frame for the motor's ctrl_mode."""
        cm = m.ctrl_mode
        if cm == "Pos-Vel":
            self.send_pos_vel(m)
        elif cm == "Vel":
            self.send_vel(m)
        elif cm == "Force":
            self.send_force(m)
        else:                       # MIT (default)
            self.send_mit(m)

    def switch_mode(self, mid: int, mode_name: str):
        """Write the motor's CTRL_MODE register (RID 10) to the Control_Type int.
        Best-effort — switch while disabled, then re-enable."""
        ctrl_int = CTRL_MODES.get(mode_name, 1)
        self._tx_command(GLOBAL_ID,
                         bytes([mid & 0xFF, (mid >> 8) & 0xFF, 0x55, 10])
                         + struct.pack("<I", ctrl_int), mid)
        self.log(f"Motor {mid} control mode → {mode_name} (CTRL_MODE={ctrl_int})")

    def change_id(self, old_mid: int, new_mid: int):
        self._transport.send(GLOBAL_ID,
            [old_mid & 0xFF, (old_mid >> 8) & 0xFF, 0x55, 0x08,
             new_mid & 0xFF, (new_mid >> 8) & 0xFF, 0x00, 0x00])
        time.sleep(0.05)
        self._transport.send(GLOBAL_ID,
            [old_mid & 0xFF, (old_mid >> 8) & 0xFF, 0xAA, 0x01])
        time.sleep(0.1)
        with self._lock:
            if old_mid in self.motors:
                m = self.motors.pop(old_mid)
                m.mid = new_mid
                self.motors[new_mid] = m
        self.log(f"Motor ID {old_mid} → {new_mid}. Power-cycle to confirm.")

    def set_motor_ids(self, cur_mid: int, new_esc=None, new_mst=None):
        """Write a DaMiao motor's CAN/ESC id (RID 8) and/or master id (RID 7),
        then save to flash. Both writes are addressed by the motor's *current*
        ESC id (`cur_mid`), which stays valid until the next power-cycle. Pass
        None for either field to leave it unchanged."""
        if new_mst is None and (new_esc is None or new_esc == cur_mid):
            self.log(f"Motor {cur_mid}: IDs unchanged — nothing to write.")
            return
        lo, hi = cur_mid & 0xFF, (cur_mid >> 8) & 0xFF
        # Register write: 0x7FF, [idL, idH, 0x55, RID, <uint32 little-endian>].
        # Order: MST first, then ESC, then one save — every frame still addressed
        # by cur_mid because the ESC change only takes effect after power-cycle.
        if new_mst is not None:
            self._transport.send(GLOBAL_ID,
                [lo, hi, 0x55, 0x07] + list(struct.pack("<I", new_mst & 0xFFFFFFFF)))
            time.sleep(0.05)
        if new_esc is not None and new_esc != cur_mid:
            self._transport.send(GLOBAL_ID,
                [lo, hi, 0x55, 0x08] + list(struct.pack("<I", new_esc & 0xFFFFFFFF)))
            time.sleep(0.05)
        self._transport.send(GLOBAL_ID, [lo, hi, 0xAA, 0x01])   # save to flash
        time.sleep(0.1)
        with self._lock:
            m = self.motors.get(cur_mid)
            if m:
                if new_mst is not None:
                    m.mst_id = new_mst
                if new_esc is not None and new_esc != cur_mid:
                    self.motors.pop(cur_mid, None)
                    m.mid = new_esc
                    self.motors[new_esc] = m
        parts = []
        if new_esc is not None and new_esc != cur_mid:
            parts.append(f"CAN {cur_mid}→{new_esc}")
        if new_mst is not None:
            parts.append(f"MST→0x{new_mst:02X}")
        self.log(f"Motor {cur_mid}: {', '.join(parts)} saved. Power-cycle to confirm.")

    def calibrate_joint(self, mid: int, vel: float, probe_frac: float,
                        on_status, on_done, stop_event: threading.Event):
        """Home a joint to BOTH hard stops with a bounded drive torque.

        A position target is ramped toward the limit at the sweep velocity, but
        the impedance lead (and therefore the motor torque) is CLAMPED so the
        push never exceeds `probe_frac` of rated torque — enough to move the
        joint against its own friction and weight, never a slam. While free the
        joint follows the ramp at the sweep speed; at a hard stop it can no
        longer keep up, the lead saturates (torque hits the drive bound) and the
        velocity collapses — that point is recorded as the limit. Negative end
        first, then positive."""
        with self._lock:
            m = self.motors.get(mid)
        if m is None:
            on_status(f"J{mid}: not found on bus")
            on_done(mid, None, None)
            return

        name     = JOINT_BY_ID.get(mid, {}).get("name", f"J{mid}")
        t_max    = m.lim["t_max"]
        vmag     = abs(vel) or 0.2
        drive    = max(0.02, probe_frac) * t_max            # bounded push torque (Nm)
        # Tracking stiffness scales with the joint's hold gains so the soft wrist
        # joints (J6/J7) don't buzz during the probe/return like the heavy joints.
        hold_kp, hold_kd = PD_DEFAULTS.get(mid, DEFAULT_PD)
        cal_kp   = min(m.lim.get("kp_max", 500.0), max(15.0, 80.0 * hold_kp / 30.0))
        cal_kd   = min(m.lim.get("kd_max", 5.0),   max(0.8, 2.0 * hold_kd / 1.0))
        lead_max = drive / cal_kp                            # lead → torque = drive
        m.ctrl_mode = "MIT"
        DT = 0.02

        def _move(direction: float, label: str):
            """Ramp toward one limit; return the position the instant the joint
            stalls against the hard stop (or None on cancel/timeout)."""
            with self._lock:
                ptgt = m.pos
            m.mode       = "position"
            m.kp         = cal_kp
            m.kd         = cal_kd
            m.vel_ff     = direction * vmag
            m.t_ff       = 0.0
            m.pos_target = ptgt
            moved        = False
            block_since  = None
            t_start      = time.perf_counter()
            timeout      = t_start + 30.0

            while time.perf_counter() < timeout:
                if stop_event.is_set():
                    return None
                with self._lock:
                    pos = m.pos
                    v   = m.vel
                    tau = abs(m.tau)
                now = time.perf_counter()

                # Advance the target at the sweep velocity, but never let it lead
                # the actual position by more than lead_max → torque ≤ drive.
                ptgt += direction * vmag * DT
                if direction > 0:
                    ptgt = min(ptgt, pos + lead_max)
                else:
                    ptgt = max(ptgt, pos - lead_max)
                m.pos_target    = ptgt
                m._just_enabled = False

                if abs(v) > 0.4 * vmag:
                    moved = True

                lead = abs(ptgt - pos)
                on_status(f"{name}: probing {label}  pos={pos:+.3f} rad  "
                          f"τ={tau:.2f}/{drive:.2f} Nm  v={v:+.2f}")

                # Hard stop = pushing at the full drive (lead saturated) while the
                # joint is no longer moving — only after it has genuinely moved.
                saturated = lead >= 0.9 * lead_max
                blocked   = abs(v) < max(0.04, 0.15 * vmag)
                if moved and now - t_start > 0.3 and saturated and blocked:
                    block_since = block_since or now
                    if now - block_since > 0.2:        # sustained → real stop
                        m.pos_target = pos             # release the push
                        return pos
                else:
                    block_since = None
                time.sleep(DT)
            return None

        on_status(f"{name}: enabling… (probe torque ≤ {drive:.2f} Nm)")
        self.enable(mid)
        time.sleep(0.3)

        # Probe negative end
        lo = _move(-1.0, "→ negative limit")
        if lo is None:
            cancelled = stop_event.is_set()
            on_status(f"{name}: " + ("cancelled" if cancelled
                      else "no negative stop — raise probe torque or velocity"))
            m.t_ff = 0.0; m.vel_ff = 0.0
            self.disable(mid)
            on_done(mid, None, None)
            return
        on_status(f"{name}: negative limit = {lo:+.4f} rad")
        time.sleep(0.4)

        # Probe positive end
        hi = _move(+1.0, "→ positive limit")
        if hi is None:
            cancelled = stop_event.is_set()
            on_status(f"{name}: " + ("cancelled" if cancelled
                      else "no positive stop — raise probe torque or velocity"))
            m.t_ff = 0.0; m.vel_ff = 0.0
            self.disable(mid)
            on_done(mid, lo, None)
            return
        on_status(f"{name}: positive limit = {hi:+.4f} rad")

        # Glide back to centre at a brisk-but-smooth speed — ramp the target with
        # the same lead clamp so the joint moves quickly yet never lunges or snaps.
        centre    = (lo + hi) / 2.0
        ret_speed = min(2.5 * vmag, 1.0)           # faster than the sweep, capped
        m.t_ff    = 0.0
        m.mode    = "position"
        with self._lock:
            ptgt = m.pos
        t_end = time.perf_counter() + 25.0
        while time.perf_counter() < t_end:
            if stop_event.is_set():
                break
            with self._lock:
                pos = m.pos
            step = ret_speed * DT
            if abs(centre - ptgt) <= step:
                ptgt = centre
            else:
                ptgt += step if centre > ptgt else -step
            # Keep the commanded lead bounded so torque stays gentle if it lags.
            ptgt = min(ptgt, pos + lead_max) if centre > pos else max(ptgt, pos - lead_max)
            m.vel_ff = (ret_speed if centre > pos else -ret_speed) \
                       if abs(centre - pos) > 0.02 else 0.0
            m.pos_target    = ptgt
            m._just_enabled = False
            if abs(pos - centre) < 0.02 and abs(ptgt - centre) < 1e-3:
                break
            time.sleep(DT)
        m.vel_ff     = 0.0
        m.pos_target = centre
        time.sleep(0.3)
        self.disable(mid)
        on_done(mid, lo, hi)

    def shutdown(self):
        self._stop.set()
        # Snapshot under the lock, then disable WITHOUT holding it — disable()
        # re-acquires self._lock, and threading.Lock is non-reentrant, so holding
        # it here would deadlock (hanging the close handler and, on gs_usb, never
        # releasing the USB device → next launch fails with "Access denied").
        with self._lock:
            mids = list(self.motors)
        for mid in mids:
            try: self.disable(mid)
            except Exception: pass
        time.sleep(0.15)
        try: self._transport.close()
        except Exception: pass


# ── FloatSlider — AppleSlider but with float output ──────────────────────────
class FloatSlider(AppleSlider):
    """Identical to AppleSlider but reports precise floats, not ints."""
    def _on_drag(self, event):
        w = self.winfo_width()
        if w <= 1: w = int(self.cget("width"))
        x   = max(self.padding, min(w - self.padding, event.x))
        pct = (x - self.padding) / max(1, w - 2 * self.padding)
        self.val = self.min_val + pct * (self.max_val - self.min_val)
        self._redraw()
        if self.command: self.command(self.val)


# ── JointSlider — float-precision slider with dual position indicators ────────
class JointSlider(tk.Canvas):
    """
    iOS-style slider for joint control.
    • Orange fill  → target position (draggable thumb)
    • Green tick   → actual motor position (updated from feedback)
    • Float output; range [lo, hi] rad.
    """
    _R    = 13   # thumb radius
    _TW   = 6    # track line width
    _PAD  = 18   # horizontal inset

    def __init__(self, parent, lo: float, hi: float,
                 command=None, parent_bg=SURFACE, **kwargs):
        super().__init__(parent, height=36, bg=parent_bg,
                         highlightthickness=0, **kwargs)
        self.lo      = lo
        self.hi      = hi
        self._target = (lo + hi) / 2.0
        self._actual = (lo + hi) / 2.0
        self.command = command

        self.bind("<Configure>",    lambda e: self._draw())
        self.bind("<Button-1>",     self._on_press)
        self.bind("<B1-Motion>",    self._on_drag)

    # Public API ──────────────────────────────────────────────────────────────
    def set_target(self, val: float, fire: bool = False):
        """Move the thumb (target) without triggering the command unless fire=True."""
        self._target = max(self.lo, min(self.hi, float(val)))
        self._draw()
        if fire and self.command:
            self.command(self._target)

    def set_actual(self, val: float):
        """Update the green actual-position tick. Called from the refresh loop."""
        v = max(self.lo, min(self.hi, float(val)))
        if abs(v - self._actual) > 5e-4:
            self._actual = v
            self._draw()

    def get(self) -> float:
        return self._target

    # Internal ────────────────────────────────────────────────────────────────
    def _to_x(self, val: float) -> float:
        w   = self.winfo_width() or int(self.cget("width")) or 200
        pct = (val - self.lo) / (self.hi - self.lo) if self.hi != self.lo else 0.0
        return self._PAD + pct * (w - 2 * self._PAD)

    def _to_val(self, x: float) -> float:
        w   = self.winfo_width() or int(self.cget("width")) or 200
        pct = max(0.0, min(1.0, (x - self._PAD) / max(1, w - 2 * self._PAD)))
        return self.lo + pct * (self.hi - self.lo)

    def _draw(self):
        self.delete("all")
        w  = self.winfo_width() or int(self.cget("width")) or 200
        h  = self.winfo_height() or 36
        ty = h // 2
        R  = self._R
        tx = self._to_x(self._target)
        ax = self._to_x(self._actual)

        # Trough
        self.create_line(self._PAD, ty, w - self._PAD, ty,
                         fill=ELEVATED, width=self._TW, capstyle=tk.ROUND)
        # Orange fill — left edge to thumb
        if tx > self._PAD:
            self.create_line(self._PAD, ty, tx, ty,
                             fill=ORANGE, width=self._TW, capstyle=tk.ROUND)
        # Green actual-position tick (behind thumb)
        self.create_line(ax, ty - 8, ax, ty + 8, fill=GREEN, width=2)
        # Thumb shadow
        self.create_oval(tx - R, ty - R + 2, tx + R, ty + R + 2,
                         fill="#111111", outline="")
        # Thumb
        self.create_oval(tx - R, ty - R, tx + R, ty + R,
                         fill=LABEL, outline="")

    def _on_press(self, e): self._update(e.x)
    def _on_drag(self,  e): self._update(e.x)

    def _update(self, x: float):
        self._target = self._to_val(x)
        self._draw()
        if self.command: self.command(self._target)


# ── Card Helper (identical to lift / ORCA GUIs) ───────────────────────────────
def _build_card(parent, title):
    card  = AppleCard(parent, radius=16, bg_color=SURFACE, parent_bg=BG)
    inner = tk.Frame(card, bg=SURFACE)
    inner.pack(fill="both", expand=True, padx=16, pady=16)
    tk.Label(inner, text=title.upper(), bg=SURFACE, fg=ORANGE,
             font=F_CAPTION, anchor="w").pack(fill="x", pady=(0, 4))
    tk.Frame(inner, bg=SEP, height=1).pack(fill="x", pady=(0, 12))
    return card, inner


# ── Telemetry Sidebar ─────────────────────────────────────────────────────────
class TelemetrySidebar(tk.Frame):
    def __init__(self, parent, bus: DamiaoBus, on_select):
        super().__init__(parent, bg=PANEL_BG, width=SIDEBAR_W)
        self.pack_propagate(False)
        self._bus       = bus
        self._on_select = on_select
        self._rows: dict[int, dict] = {}
        self._build()

    def _build(self):
        tk.Label(self, text="MOTOR STATUS", bg=PANEL_BG,
                 fg=LABEL2, font=FONT_SMALL).pack(pady=(22, 10))
        # (Scan Motors now lives in the top toolbar.)

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

        # Column header
        hdr = tk.Frame(self, bg=PANEL_BG)
        hdr.pack(fill="x", padx=14, pady=(5, 3))
        for text, w in [("  ", 2), ("POS", 7), ("VEL", 5), ("TAU", 5), ("°C", 5)]:
            tk.Label(hdr, text=text, width=w, bg=PANEL_BG,
                     fg=LABEL2, font=FONT_MONO_S, anchor="e").pack(side="left")

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

        # Fixed 7-joint rows
        for ji in JOINT_INFO:
            mid  = ji["mid"]
            row  = tk.Frame(self, bg=PANEL_BG, cursor="hand2")
            row.pack(fill="x")

            id_lbl = tk.Label(row, text=f"J{mid}", width=3,
                              bg=PANEL_BG, fg=LABEL3, font=FONT_MONO, anchor="w")
            id_lbl.pack(side="left", padx=(10, 0), pady=5)

            name_lbl = tk.Label(row, text=ji["name"], width=9,
                                bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S, anchor="w")
            name_lbl.pack(side="left")

            v = {k: tk.StringVar(value="──") for k in ("pos", "vel", "tau", "tmp")}
            lbls = {}
            for key, w in [("pos", 7), ("vel", 5), ("tau", 5), ("tmp", 5)]:
                l = tk.Label(row, textvariable=v[key], width=w,
                             bg=PANEL_BG, fg=LABEL3, font=FONT_MONO_S, anchor="e")
                l.pack(side="left")
                lbls[key] = l

            for widget in [row, id_lbl, name_lbl] + list(lbls.values()):
                widget.bind("<Button-1>", lambda e, m=mid: self._on_select(m))

            self._rows[mid] = {"row": row, "id_lbl": id_lbl,
                               "name_lbl": name_lbl, "vars": v, "labels": lbls}

        tk.Frame(self, bg=SEP, height=1).pack(fill="x", pady=(PAD_SM, 0))
        tk.Label(self, text="EVENT LOG", bg=PANEL_BG,
                 fg=LABEL2, font=FONT_SMALL).pack(pady=(8, 4))
        self._log = tk.Text(self, bg=PANEL_BG, fg=LABEL2, font=FONT_MONO_S,
                            relief="flat", bd=0, height=7,
                            insertbackground=LABEL, state="disabled")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def refresh(self, active_mid: int | None):
        with self._bus._lock:
            snap = {mid: (m.pos, m.vel, m.tau, m.enabled, m.err, m.found,
                          m.stale, m.lim["t_max"], m.t_mos)
                    for mid, m in self._bus.motors.items()}

        for mid, data in self._rows.items():
            if mid not in snap:
                continue
            pos, vel, tau, enabled, err, found, stale, t_max, t_mos = snap[mid]
            v, lb = data["vars"], data["labels"]
            is_active = (mid == active_mid)
            row_bg    = ELEVATED if is_active else PANEL_BG

            data["row"].config(bg=row_bg)
            data["id_lbl"].config(bg=row_bg)
            data["name_lbl"].config(bg=row_bg)

            if not found:
                for k in ("pos", "vel", "tau", "tmp"):
                    v[k].set("──")
                    lb[k].config(bg=row_bg, fg=LABEL3)
                data["id_lbl"].config(fg=LABEL3)
                data["name_lbl"].config(fg=LABEL3)
                continue

            # Dim everything when feedback has gone stale (motor disconnected mid-session)
            dim = stale and not enabled
            id_color = (RED   if err
                        else LABEL3 if dim
                        else GREEN  if enabled
                        else ORANGE if is_active
                        else LABEL2)
            data["id_lbl"].config(fg=id_color)
            data["name_lbl"].config(fg=LABEL3 if dim else (LABEL if is_active else LABEL2))

            # Always show live pos/vel/tau — enabled state only controls torque output
            sfx = " ~" if stale else ""   # trailing ~ = stale
            v["pos"].set(f"{pos:+.3f}{sfx}")
            v["vel"].set(f"{vel:+.2f}{sfx}")
            v["tau"].set(f"{tau:+.1f}")
            base = LABEL3 if dim else (LABEL if is_active else LABEL2)
            lb["pos"].config(bg=row_bg, fg=base)
            lb["vel"].config(bg=row_bg, fg=GREEN if abs(vel) > 0.05 and not dim else base)
            tau_ratio = abs(tau) / max(t_max, 0.1)
            tau_fg = RED if tau_ratio > 0.7 else ORANGE if tau_ratio > 0.3 else base
            lb["tau"].config(bg=row_bg, fg=LABEL3 if dim else tau_fg)
            # MOS temperature (°C) — warm >55, hot >75
            v["tmp"].set("──" if t_mos <= 0 else f"{t_mos:.0f}°")
            tmp_fg = (LABEL3 if dim or t_mos <= 0
                      else RED if t_mos > 75 else ORANGE if t_mos > 55 else base)
            lb["tmp"].config(bg=row_bg, fg=tmp_fg)

        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        for e in self._bus.log_tail(25):
            self._log.insert("end", e + "\n")
        self._log.see("end")
        self._log.config(state="disabled")


# ── Live strip-chart ──────────────────────────────────────────────────────────
class LivePlot(tk.Canvas):
    """Rolling strip chart for one telemetry channel, auto-scaling on y.
    Title + current value are drawn along the top; a faint zero baseline is
    shown when zero is in range."""
    _MAXN = 240   # rolling window length (≈ 5 s at 50 Hz with 2× redraw skip)

    def __init__(self, parent, title, unit, color=ORANGE,
                 parent_bg=SURFACE, height=118, width=440, **kw):
        super().__init__(parent, height=height, width=width, bg=SURFACE,
                         highlightthickness=0, **kw)
        self.title  = title
        self.unit   = unit
        self.color  = color
        self._buf:  list[float] = []
        self._skip  = 0
        self.bind("<Configure>", lambda e: self._draw())

    def clear(self):
        self._buf = []
        self._draw()

    def push(self, value: float):
        self._buf.append(float(value))
        if len(self._buf) > self._MAXN:
            del self._buf[0]
        # Redraw at ~half the control rate to keep Tk light on macOS.
        self._skip = (self._skip + 1) % 2
        if self._skip == 0:
            self._draw()

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()  or int(self.cget("width"))
        h = self.winfo_height() or int(self.cget("height"))
        pad_l, pad_r, pad_t, pad_b = 10, 10, 20, 8
        x0, x1 = pad_l, w - pad_r
        y0, y1 = pad_t, h - pad_b

        cur = self._buf[-1] if self._buf else 0.0
        self.create_text(pad_l, 10, text=self.title.upper(), anchor="w",
                         fill=LABEL2, font=F_CAPTION)
        self.create_text(w - pad_r, 10, text=f"{cur:+.3f} {self.unit}",
                         anchor="e", fill=self.color, font=FONT_MONO_S)

        if len(self._buf) < 2:
            return
        lo, hi = min(self._buf), max(self._buf)
        if hi - lo < 1e-6:
            lo -= 1.0; hi += 1.0
        span = hi - lo
        if lo <= 0.0 <= hi:                      # zero baseline
            zy = y1 - (0.0 - lo) / span * (y1 - y0)
            self.create_line(x0, zy, x1, zy, fill=SEP)
        n = len(self._buf)
        coords = []
        for i, v in enumerate(self._buf):
            x = x0 + (i / (n - 1)) * (x1 - x0)
            y = y1 - (v - lo) / span * (y1 - y0)
            coords += [x, y]
        self.create_line(*coords, fill=self.color, width=2)


# ── Control Panel ─────────────────────────────────────────────────────────────
class ControlPanel(tk.Frame):
    """All-joints control + per-motor live feedback. Every joint gets its own
    position target slider and torque toggle in one compact table; impedance
    gains (Kp / Kd / τ_ff) are typed once and applied to all joints. Click a
    joint to stream its pos/vel/τ in the live feedback column on the right.
    Telemetry streams continuously — Enable is solely for torque."""

    def __init__(self, parent, bus: DamiaoBus, get_active, set_active):
        super().__init__(parent, bg=BG)
        self._bus         = bus
        self._get_active  = get_active
        self._set_active  = set_active
        self._last_active = None
        self._build()

    # ── Layout ──────────────────────────────────────────────────────────────
    def _build(self):
        cols  = tk.Frame(self, bg=BG)
        cols.pack(fill="both", expand=True)
        left  = tk.Frame(cols, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))
        right = tk.Frame(cols, bg=BG)
        right.pack(side="left", fill="y")

        # ── All-joints position control ───────────────────────────────────────
        c_j, i_j, _ = self._titled_card(left, "Joints — Position Control")
        c_j.pack(fill="x", pady=(0, 7))

        jhdr = tk.Frame(i_j, bg=SURFACE)
        jhdr.pack(fill="x", pady=(0, 2))
        tk.Label(jhdr, text="Joint", bg=SURFACE, fg=LABEL3, font=F_CAPTION,
                 width=15, anchor="w").pack(side="left")
        tk.Label(jhdr, text="Torque", bg=SURFACE, fg=LABEL3, font=F_CAPTION,
                 width=8, anchor="w").pack(side="left")
        tk.Label(jhdr, text="Target position  (rad)", bg=SURFACE, fg=LABEL3,
                 font=F_CAPTION, anchor="w").pack(side="left")
        tk.Frame(i_j, bg=SEP, height=1).pack(fill="x", pady=(0, 4))

        self._jrows: dict[int, dict] = {}
        for ji in JOINT_INFO:
            mid = ji["mid"]
            row = tk.Frame(i_j, bg=SURFACE)
            row.pack(fill="x", pady=1)
            name = tk.Label(row, text=f"J{mid} {ji['name']}", width=15,
                            bg=SURFACE, fg=LABEL2, font=FONT_MONO_S, anchor="w",
                            cursor="hand2")
            name.pack(side="left")
            name.bind("<Button-1>", lambda e, m=mid: self._set_active(m))
            en = AppleButton(row, "OFF", lambda m=mid: self._toggle_enable(m),
                             bg_color=ELEVATED, fg_color=LABEL3, font=F_CAPTION,
                             parent_bg=SURFACE, width=58, height=26, radius=8)
            en.pack(side="left", padx=(0, PAD_SM))
            val = tk.StringVar(value="0.000")
            tk.Label(row, textvariable=val, bg=SURFACE, fg=LABEL,
                     font=FONT_MONO_S, width=8, anchor="e").pack(side="right")
            lo, hi = ji["lo"], ji["hi"]
            sl = JointSlider(row, lo, hi,
                             command=lambda v, m=mid: self._on_pos(m, v),
                             parent_bg=SURFACE)
            sl.pack(side="left", fill="x", expand=True, padx=PAD_SM)
            self._jrows[mid] = {"row": row, "name": name, "en": en,
                                "sl": sl, "val": val}

        foot = tk.Frame(i_j, bg=SURFACE)
        foot.pack(fill="x", pady=(8, 0))
        AppleButton(foot, "Enable All", lambda: self._all_enable(True),
                    bg_color=GREEN, fg_color=BG, font=F_CAPTION,
                    parent_bg=SURFACE, width=104, height=30, radius=10).pack(side="left", padx=(0, PAD_SM))
        AppleButton(foot, "Disable All", lambda: self._all_enable(False),
                    bg_color=RED, fg_color=LABEL, font=F_CAPTION,
                    parent_bg=SURFACE, width=104, height=30, radius=10).pack(side="left", padx=(0, PAD_SM))
        AppleButton(foot, "Targets → actual", self._sync_targets,
                    bg_color=ELEVATED, fg_color=LABEL, font=F_CAPTION,
                    parent_bg=SURFACE, width=140, height=30, radius=10).pack(side="left")

        # ── Mode + per-joint impedance ────────────────────────────────────────
        c_g, i_g, self._gains_title = self._titled_card(left, "Mode  &  Impedance")
        c_g.pack(fill="x")
        self._mode_seg = AppleSegmentedControl(
            i_g, list(CTRL_MODES.keys()), "MIT",
            self._on_mode_change, width=400, height=32, parent_bg=SURFACE)
        self._mode_seg.pack(fill="x", pady=(0, 5))
        self._v_mode_desc = tk.StringVar(value=CTRL_MODE_DESC["MIT"])
        tk.Label(i_g, textvariable=self._v_mode_desc, bg=SURFACE,
                 fg=LABEL3, font=FONT_MONO_S, anchor="w").pack(fill="x", pady=(0, 6))

        grow = tk.Frame(i_g, bg=SURFACE)
        grow.pack(fill="x")
        # Gains target the SELECTED joint (label shows which) so the soft wrist
        # joints can be tuned without disturbing the heavy ones.
        self._v_gain_joint = tk.StringVar(value="J1")
        tk.Label(grow, textvariable=self._v_gain_joint, bg=SURFACE, fg=ORANGE,
                 font=F_CAPTION, width=3, anchor="w").pack(side="left", padx=(0, 6))
        self._gain_vars: dict[str, tk.StringVar] = {}
        for key, lab, default in [("kp", "Kp", "30"), ("kd", "Kd", "1.0"),
                                  ("tff", "τ_ff", "0.0")]:
            tk.Label(grow, text=lab, bg=SURFACE, fg=LABEL2,
                     font=F_CAPTION).pack(side="left", padx=(0, 4))
            var = tk.StringVar(value=default)
            e = tk.Entry(grow, textvariable=var, width=6, bg=ELEVATED, fg=LABEL,
                         insertbackground=LABEL, relief="flat", bd=6, font=FONT_MONO)
            e.pack(side="left", padx=(0, PAD))
            e.bind("<Return>",   lambda ev: self._apply_gains_active())
            e.bind("<FocusOut>", lambda ev: self._apply_gains_active())
            self._gain_vars[key] = var
        AppleButton(grow, "Apply to all", self._apply_gains_all,
                    bg_color=BLUE, fg_color=BG, font=F_CAPTION,
                    parent_bg=SURFACE, width=110, height=28, radius=9).pack(side="left")
        tk.Label(i_g, text="Kp / Kd / τ_ff for the SELECTED joint — edit and press Enter. "
                 "\"Apply to all\" copies them to every joint. J5–J7 default soft to "
                 "stop the wrists shaking.",
                 bg=SURFACE, fg=LABEL3, font=F_CAPTION, anchor="w",
                 wraplength=400, justify="left").pack(fill="x", pady=(6, 0))

        # ── Right column: live plots + dense telemetry for the SELECTED joint ──
        c_live, i_live, _ = self._titled_card(right, "Live Feedback")
        c_live.pack(fill="x", pady=(0, 7))
        self._plot_pos = LivePlot(i_live, "Position", "rad",   color=ORANGE)
        self._plot_pos.pack(fill="x", pady=(0, 7))
        self._plot_vel = LivePlot(i_live, "Velocity", "rad/s", color=GREEN)
        self._plot_vel.pack(fill="x", pady=(0, 7))
        self._plot_tau = LivePlot(i_live, "Torque",   "Nm",    color=BLUE)
        self._plot_tau.pack(fill="x")

        c_tel, i_tel, _ = self._titled_card(right, "Telemetry")
        c_tel.pack(fill="both", expand=True)
        self._build_tele(i_tel)

        # No global stomp at startup — each motor keeps its per-joint PD default
        # (PD_DEFAULTS); the entries seed from the selected joint in refresh().

    # ── Builders ────────────────────────────────────────────────────────────
    def _titled_card(self, parent, title):
        """Compact card; returns (card, inner, title_label) so titles can gray out."""
        card  = AppleCard(parent, radius=16, bg_color=SURFACE, parent_bg=BG)
        inner = tk.Frame(card, bg=SURFACE)
        inner.pack(fill="both", expand=True, padx=14, pady=11)
        tl = tk.Label(inner, text=title.upper(), bg=SURFACE, fg=ORANGE,
                      font=F_CAPTION, anchor="w")
        tl.pack(fill="x", pady=(0, 4))
        tk.Frame(inner, bg=SEP, height=1).pack(fill="x", pady=(0, 8))
        return card, inner, tl

    def _slider_row(self, parent, label, lo, hi, init, cmd, fmt="{:.2f}"):
        """Single-row FloatSlider: [label | slider | value]. Returns (sl, var, lbl)."""
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", pady=2)
        lb = tk.Label(row, text=label, bg=SURFACE, fg=LABEL2,
                      font=F_CAPTION, width=5, anchor="w")
        lb.pack(side="left")
        v  = tk.StringVar(value=fmt.format(init))
        tk.Label(row, textvariable=v, bg=SURFACE, fg=LABEL,
                 font=FONT_MONO_S, width=9, anchor="e").pack(side="right")
        sl = FloatSlider(row, lo, hi, command=cmd, height=26)
        sl.set(init)
        sl.pack(side="left", fill="x", expand=True, padx=PAD_SM)
        sl._fmt = fmt
        return sl, v, lb

    _TELE_FIELDS = [("pos", "Position"), ("vel", "Velocity"), ("tau", "Torque"),
                    ("tmos", "MOS °C"), ("ptgt", "Target q"), ("vtgt", "Target dq"),
                    ("status", "Status"), ("cid", "CAN id"), ("prof", "Profile"),
                    ("vmax", "v_max"), ("tmax", "τ_max"), ("mode", "Mode")]

    def _build_tele(self, parent):
        grid = tk.Frame(parent, bg=SURFACE)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        self._tele = {}
        for i, (k, lab) in enumerate(self._TELE_FIELDS):
            r, c = divmod(i, 2)
            cell = tk.Frame(grid, bg=SURFACE)
            cell.grid(row=r, column=c, sticky="w", padx=(0, 18), pady=2)
            tk.Label(cell, text=lab, bg=SURFACE, fg=LABEL3,
                     font=F_CAPTION, width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="—")
            tk.Label(cell, textvariable=v, bg=SURFACE, fg=LABEL,
                     font=FONT_MONO_S, width=10, anchor="w").pack(side="left")
            self._tele[k] = v

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_pos(self, mid: int, val: float):
        """One joint's target slider moved. Only commands while torque is on;
        the value label always tracks the slider."""
        with self._bus._lock:
            m = self._bus.motors.get(mid)
        if m and m.enabled:
            m._just_enabled = False
            m.pos_target = val
        r = self._jrows.get(mid)
        if r:
            r["val"].set(f"{val:+.3f}")

    def _toggle_enable(self, mid: int):
        with self._bus._lock:
            m = self._bus.motors.get(mid)
        if not m:
            return
        self._set_active(mid)
        self._bus.disable(mid) if m.enabled else self._bus.enable(mid)

    def _all_enable(self, on: bool):
        with self._bus._lock:
            mids = [mid for mid, m in self._bus.motors.items() if m.found]
        for mid in mids:
            self._bus.enable(mid) if on else self._bus.disable(mid)

    def _sync_targets(self):
        """Snap every joint's target (and its slider) to the actual position —
        handy before enabling so nothing lunges."""
        with self._bus._lock:
            snap = {mid: (m.pos, m.enabled) for mid, m in self._bus.motors.items()}
        for mid, (pos, enabled) in snap.items():
            with self._bus._lock:
                m = self._bus.motors.get(mid)
                if m and enabled:
                    m._just_enabled = False
                    m.pos_target = pos
            r = self._jrows.get(mid)
            if r:
                r["sl"].set_target(pos)
                r["val"].set(f"{pos:+.3f}")

    def _on_mode_change(self, mode: str):
        """Switch the control mode on every joint (disabling torque first)."""
        self._v_mode_desc.set(CTRL_MODE_DESC.get(mode, ""))
        self._gains_title.config(fg=ORANGE if mode == "MIT" else LABEL3)
        with self._bus._lock:
            mids = [mid for mid, m in self._bus.motors.items() if m.found]
        for mid in mids:
            with self._bus._lock:
                m = self._bus.motors.get(mid)
            if not m:
                continue
            if m.enabled:
                self._bus.disable(mid)
            m.ctrl_mode = mode
            m.mode      = _CTRL_TO_SUBMODE.get(mode, "position")
            if mode == "Vel":
                m.vel_target = 0.0
            self._bus.switch_mode(mid, mode)

    def _read_gains(self):
        def _num(key):
            try:
                return float(self._gain_vars[key].get())
            except (ValueError, KeyError):
                return None
        return _num("kp"), _num("kd"), _num("tff")

    @staticmethod
    def _apply_to(m, kp, kd, tff):
        if kp  is not None: m.kp   = kp
        if kd  is not None: m.kd   = kd
        if tff is not None: m.t_ff = tff

    def _apply_gains_active(self):
        """Apply the Kp / Kd / τ_ff entries to the SELECTED joint only."""
        kp, kd, tff = self._read_gains()
        m = self._get_active()
        if m:
            with self._bus._lock:
                self._apply_to(m, kp, kd, tff)

    def _apply_gains_all(self):
        """Copy the entries to EVERY joint (the old global behaviour)."""
        kp, kd, tff = self._read_gains()
        with self._bus._lock:
            for m in self._bus.motors.values():
                self._apply_to(m, kp, kd, tff)

    def _seed_gains(self, m):
        """Load the entries from a motor's live gains (on joint selection)."""
        self._gain_vars["kp"].set(f"{m.kp:g}")
        self._gain_vars["kd"].set(f"{m.kd:g}")
        self._gain_vars["tff"].set(f"{m.t_ff:g}")

    # ── Keyboard (arrows nudge the ACTIVE joint's target / scale step) ────────
    def _nudge_active(self, sign: int):
        m = self._get_active()
        if m is None or not m.enabled:
            return
        if m.ctrl_mode == "Vel":
            vmax = m.lim["v_max"]
            m.vel_target = max(-vmax, min(vmax, m.vel_target + sign * m.vel_step))
        elif m.pos_target is not None:
            m._just_enabled = False
            ji = JOINT_BY_ID.get(m.mid, {})
            # Prefer the calibrated physical range; fall back to the theoretical one.
            lo, hi = _joint_cal.get(m.mid,
                                    (ji.get("lo", -m.lim["p_max"]),
                                     ji.get("hi",  m.lim["p_max"])))
            m.pos_target = max(lo, min(hi, m.pos_target + sign * m.step))
            r = self._jrows.get(m.mid)
            if r:
                r["sl"].set_target(m.pos_target)
                r["val"].set(f"{m.pos_target:+.3f}")

    def key_left(self):  self._nudge_active(-1)
    def key_right(self): self._nudge_active(+1)

    def key_up(self):
        m = self._get_active()
        if m: m.step = max(0.001, min(2.0, round(m.step * 2.0, 4)))

    def key_down(self):
        m = self._get_active()
        if m: m.step = max(0.001, min(2.0, round(m.step * 0.5, 4)))

    # ── Refresh ────────────────────────────────────────────────────────────────
    def refresh(self, active_mid: int | None):
        m = self._get_active()

        # Active joint changed → reset plots, resync the mode segment to it.
        if active_mid != self._last_active:
            self._last_active = active_mid
            for p in (self._plot_pos, self._plot_vel, self._plot_tau):
                p.clear()
            if m:
                self._mode_seg.set(m.ctrl_mode)
                self._v_mode_desc.set(CTRL_MODE_DESC.get(m.ctrl_mode, ""))
                self._gains_title.config(fg=ORANGE if m.ctrl_mode == "MIT" else LABEL3)
                # Load the selected joint's PD into the entries so they can be tuned.
                self._v_gain_joint.set(f"J{active_mid}")
                self._seed_gains(m)

        # Update every joint row (target slider actual tick, torque toggle, name).
        with self._bus._lock:
            snap = {mid: (mo.pos, mo.enabled, mo.found, mo.stale, mo.err)
                    for mid, mo in self._bus.motors.items()}
        for mid, r in self._jrows.items():
            if mid not in snap:
                continue
            pos, enabled, found, stale, err = snap[mid]
            is_active = (mid == active_mid)
            if enabled:
                r["en"].update_colors(GREEN, BG); r["en"].update_text("ON")
            else:
                r["en"].update_colors(ELEVATED, LABEL3); r["en"].update_text("OFF")
            name_fg = (RED   if err
                       else LABEL3 if (not found or (stale and not enabled))
                       else GREEN  if enabled
                       else ORANGE if is_active
                       else LABEL2)
            r["name"].config(fg=name_fg)
            # Adopt the calibrated physical range once calibration has measured it,
            # so the target slider spans the joint's true travel, not the theory.
            cal = _joint_cal.get(mid)
            if cal and r.get("range") != cal:
                r["range"] = cal
                sl = r["sl"]
                sl.lo, sl.hi = cal
                sl.set_target(min(max(sl.get(), cal[0]), cal[1]))   # re-clamp + redraw
            # Green tick = actual; while disabled the thumb follows actual so it
            # never jumps on enable.
            r["sl"].set_actual(pos)
            if not enabled:
                r["sl"].set_target(pos)
                r["val"].set(f"{pos:+.3f}")

        if not m:
            return

        # Right column: live plots + dense telemetry for the selected joint.
        with self._bus._lock:
            pos, vel, tau, t_mos, err = m.pos, m.vel, m.tau, m.t_mos, m.err
            found = m.found
            ptgt = m.pos_target if m.pos_target is not None else 0.0
            vtgt, cid, prof = m.vel_target, m.mid, m.profile
            vmax, tmax, cmode = m.lim["v_max"], m.lim["t_max"], m.ctrl_mode

        if found:
            self._plot_pos.push(pos)
            self._plot_vel.push(vel)
            self._plot_tau.push(tau)

        t = self._tele
        t["pos"].set(f"{pos:+.4f}");   t["vel"].set(f"{vel:+.3f}")
        t["tau"].set(f"{tau:+.2f}");   t["tmos"].set(f"{t_mos:.0f}")
        t["ptgt"].set(f"{ptgt:+.3f}"); t["vtgt"].set(f"{vtgt:+.2f}")
        t["status"].set(ERR_NAMES.get(err, f"0x{err:X}"))
        t["cid"].set(str(cid));        t["prof"].set(prof)
        t["vmax"].set(f"{vmax:.0f}");  t["tmax"].set(f"{tmax:.0f}")
        t["mode"].set(cmode)


# ── Setup Panel ───────────────────────────────────────────────────────────────
class SetupPanel(tk.Frame):
    def __init__(self, parent, bus: DamiaoBus, get_active):
        super().__init__(parent, bg=BG)
        self._bus        = bus
        self._get_active = get_active
        self._build()

    def _build(self):
        # ── Homing ────────────────────────────────────────────────────────────
        c_home, i_home = _build_card(self, "Homing")
        c_home.pack(fill="x", pady=(0, 14))

        tk.Label(i_home, text="Zeros the encoder at the current position. Disable motor first.",
                 bg=SURFACE, fg=LABEL2, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 10))

        joint_names = [f"J{j['mid']}  {j['name']}" for j in JOINT_INFO]
        sel_row = tk.Frame(i_home, bg=SURFACE)
        sel_row.pack(fill="x", pady=(0, 8))
        tk.Label(sel_row, text="Joint", bg=SURFACE, fg=LABEL2,
                 font=FONT_BODY).pack(side="left", padx=(0, PAD))
        self._home_dd = AppleDropDown(
            sel_row, joint_names, default=joint_names[0],
            width=240, height=40, parent_bg=SURFACE)
        self._home_dd.pack(side="left")

        action_row = tk.Frame(i_home, bg=SURFACE)
        action_row.pack(fill="x")
        btn(action_row, "Zero Selected", self._zero_selected,
            parent_bg=SURFACE).pack(side="left")
        btn(action_row, "Zero All (disabled only)", self._zero_all,
            parent_bg=SURFACE).pack(side="left", padx=PAD)
        self._v_home_status = tk.StringVar(value="")
        tk.Label(i_home, textvariable=self._v_home_status, bg=SURFACE,
                 fg=GREEN, font=F_CAPTION).pack(anchor="w", pady=(6, 0))

        # ── Joint limits display ───────────────────────────────────────────────
        c_lim, i_lim = _build_card(self, "Calibrated Joint Limits")
        c_lim.pack(fill="x", pady=(0, 14))

        tk.Label(i_lim, text="Run Joint Limit Calibration to populate these values.",
                 bg=SURFACE, fg=LABEL3, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(0, 8))

        hdr = tk.Frame(i_lim, bg=SURFACE)
        hdr.pack(fill="x")
        for text, w in [("Joint", 18), ("Calibrated Range", 24), ("Theoretical", 22), ("Match", 8)]:
            tk.Label(hdr, text=text, width=w, bg=SURFACE,
                     fg=LABEL2, font=F_CAPTION, anchor="w").pack(side="left")
        tk.Frame(i_lim, bg=SEP, height=1).pack(fill="x", pady=(4, 6))

        self._limit_rows: dict[int, dict] = {}
        for ji in JOINT_INFO:
            mid  = ji["mid"]
            row  = tk.Frame(i_lim, bg=SURFACE)
            row.pack(fill="x", pady=2)
            v_cal   = tk.StringVar(value="not calibrated")
            v_theory= tk.StringVar(value=f"{ji['lo']:+.3f} → {ji['hi']:+.3f} rad")
            v_match = tk.StringVar(value="")
            tk.Label(row, text=f"J{mid}  {ji['name']}", width=18,
                     bg=SURFACE, fg=LABEL2, font=FONT_BODY, anchor="w").pack(side="left")
            cal_lbl = tk.Label(row, textvariable=v_cal, width=24,
                     bg=SURFACE, fg=LABEL3, font=FONT_MONO_S, anchor="w")
            cal_lbl.pack(side="left")
            tk.Label(row, textvariable=v_theory, width=22,
                     bg=SURFACE, fg=LABEL3, font=FONT_MONO_S, anchor="w").pack(side="left")
            match_lbl = tk.Label(row, textvariable=v_match, width=8,
                     bg=SURFACE, fg=GREEN, font=FONT_MONO_S, anchor="w")
            match_lbl.pack(side="left")
            self._limit_rows[mid] = {"v_cal": v_cal, "cal_lbl": cal_lbl,
                                     "v_match": v_match, "match_lbl": match_lbl}

        # ── Motor IDs (CAN / Master) ──────────────────────────────────────────
        c_id, i_id = _build_card(self, "Motor IDs")
        c_id.pack(fill="x")

        tk.Label(i_id, text="Motors detected on the CAN bus — CAN/ESC id and "
                 "master (feedback) id:", bg=SURFACE, fg=LABEL3, font=FONT_SMALL,
                 anchor="w").pack(fill="x", pady=(0, 8))

        # Live table header
        id_hdr = tk.Frame(i_id, bg=SURFACE)
        id_hdr.pack(fill="x")
        for text, w in [("CAN / ESC ID", 16), ("Master ID", 14),
                        ("Joint", 22), ("Live", 8)]:
            tk.Label(id_hdr, text=text, width=w, bg=SURFACE, fg=LABEL2,
                     font=F_CAPTION, anchor="w").pack(side="left")
        tk.Frame(i_id, bg=SEP, height=1).pack(fill="x", pady=(4, 4))

        # Rows are rebuilt by _rebuild_id_table() whenever the bus set changes.
        self._id_table = tk.Frame(i_id, bg=SURFACE)
        self._id_table.pack(fill="x")
        self._id_table_sig = None

        tk.Frame(i_id, bg=SEP, height=1).pack(fill="x", pady=(8, 8))

        # ── Change controls (both IDs) ────────────────────────────────────────
        sel_row2 = tk.Frame(i_id, bg=SURFACE)
        sel_row2.pack(fill="x", pady=(0, 8))
        tk.Label(sel_row2, text="Motor", bg=SURFACE, fg=LABEL2,
                 font=FONT_BODY).pack(side="left", padx=(0, PAD))
        # Dynamic menu — populated by refresh() with currently-found motors only
        self._id_var = tk.StringVar(value="scan to populate")
        self._id_menu = tk.OptionMenu(sel_row2, self._id_var, "scan to populate")
        self._id_menu.config(bg=ELEVATED, fg=LABEL, activebackground=ORANGE,
                             activeforeground=BG, font=FONT_BODY,
                             relief="flat", bd=0, highlightthickness=0,
                             indicatoron=True, width=20)
        self._id_menu["menu"].config(bg=ELEVATED, fg=LABEL,
                                     activebackground=ORANGE, activeforeground=BG,
                                     font=FONT_BODY)
        self._id_menu.pack(side="left", padx=(0, PAD))
        tk.Label(sel_row2, text="New CAN (1–15)", bg=SURFACE,
                 fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(0, PAD_SM))
        self._new_id_var = tk.StringVar()
        tk.Entry(sel_row2, textvariable=self._new_id_var, width=4,
                 bg=ELEVATED, fg=LABEL, insertbackground=LABEL,
                 relief="flat", bd=6, font=FONT_MONO).pack(side="left", padx=(0, PAD))
        tk.Label(sel_row2, text="New Master (hex/dec)", bg=SURFACE,
                 fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(0, PAD_SM))
        self._new_mst_var = tk.StringVar()
        tk.Entry(sel_row2, textvariable=self._new_mst_var, width=6,
                 bg=ELEVATED, fg=LABEL, insertbackground=LABEL,
                 relief="flat", bd=6, font=FONT_MONO).pack(side="left")

        tk.Label(i_id, text="⚠  Leave a field blank to keep it. Motor is disabled "
                 "first; both IDs are saved to flash. Power-cycle after to confirm.",
                 bg=SURFACE, fg=YELLOW, font=F_CAPTION, anchor="w",
                 justify="left", wraplength=560).pack(fill="x")
        id_btn_row = tk.Frame(i_id, bg=SURFACE)
        id_btn_row.pack(fill="x", pady=(PAD_SM, 0))
        accent_btn(id_btn_row, "Apply IDs", self._change_id,
                   color=RED, parent_bg=SURFACE).pack(side="left")
        self._v_id_status = tk.StringVar(value="")
        self._id_status_lbl = tk.Label(id_btn_row, textvariable=self._v_id_status,
                                        bg=SURFACE, fg=LABEL2, font=F_CAPTION)
        self._id_status_lbl.pack(side="left", padx=PAD)

        # ── Interface Recovery ─────────────────────────────────────────────────
        c_rst, i_rst = _build_card(self, "CAN Interface Recovery")
        c_rst.pack(fill="x", pady=(14, 0))

        tk.Label(i_rst,
                 text="If motors stop responding after connecting new hardware, the adapter\n"
                      "may have gone bus-off. This brings the interface down/up in software\n"
                      "and re-scans — no power-cycle of the adapter needed.",
                 bg=SURFACE, fg=LABEL2, font=FONT_SMALL, justify="left",
                 anchor="w").pack(fill="x", pady=(0, 10))
        rst_row = tk.Frame(i_rst, bg=SURFACE)
        rst_row.pack(fill="x")
        accent_btn(rst_row, "Reset Interface  &  Scan",
                   lambda: threading.Thread(target=self._bus.reset_bus, daemon=True).start(),
                   color=ORANGE, parent_bg=SURFACE).pack(side="left")
        self._v_rst_status = tk.StringVar(value="")
        tk.Label(rst_row, textvariable=self._v_rst_status,
                 bg=SURFACE, fg=LABEL2, font=F_CAPTION).pack(side="left", padx=PAD)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _selected_mid(self, dd: AppleDropDown) -> int:
        txt = dd.get()
        return int(txt.split()[0][1:])   # "J3  Shoulder Roll" → 3

    def _zero_selected(self):
        mid = self._selected_mid(self._home_dd)
        with self._bus._lock:
            m = self._bus.motors.get(mid)
            if m and m.enabled:
                self._v_home_status.set("⚠ Disable motor before zeroing")
                return
        self._bus.set_zero(mid)
        self._v_home_status.set(f"J{mid} zeroed ✓")

    def _zero_all(self):
        with self._bus._lock:
            ids = [mid for mid, m in self._bus.motors.items() if m.found and not m.enabled]
        for mid in ids:
            self._bus.set_zero(mid)
        self._v_home_status.set(f"Zeroed {len(ids)} joint(s) ✓")

    def _update_id_menu(self):
        with self._bus._lock:
            found_ids = sorted(mid for mid, m in self._bus.motors.items() if m.found)
        menu = self._id_menu["menu"]
        menu.delete(0, "end")
        if not found_ids:
            self._id_var.set("scan to populate")
            menu.add_command(label="scan to populate",
                             command=lambda: self._id_var.set("scan to populate"))
            return
        choices = [f"J{mid}  {JOINT_BY_ID.get(mid, {}).get('name', f'Motor {mid}')}"
                   for mid in found_ids]
        for ch in choices:
            menu.add_command(label=ch, command=lambda c=ch: self._id_var.set(c))
        if self._id_var.get() not in choices:
            self._id_var.set(choices[0])

    def _rebuild_id_table(self):
        """Repopulate the live motor-ID table — one row per motor on the bus,
        showing its CAN/ESC id and the master id learned from its reply frames.
        Rebuilt only when the set of (esc, mst, stale) actually changes."""
        with self._bus._lock:
            rows = sorted(
                (mid, m.mst_id, m.stale)
                for mid, m in self._bus.motors.items() if m.found)
        sig = tuple(rows)
        if sig == self._id_table_sig:
            return
        self._id_table_sig = sig
        for w in self._id_table.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self._id_table, text="— no motors detected (scan the bus) —",
                     bg=SURFACE, fg=LABEL3, font=FONT_MONO_S, anchor="w").pack(fill="x")
            return
        for mid, mst, stale in rows:
            name    = JOINT_BY_ID.get(mid, {}).get("name", f"Motor {mid}")
            mst_txt = "—" if mst is None else f"0x{mst:02X} ({mst})"
            r = tk.Frame(self._id_table, bg=SURFACE)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=f"0x{mid:02X} ({mid})", width=16, bg=SURFACE,
                     fg=LABEL, font=FONT_MONO_S, anchor="w").pack(side="left")
            tk.Label(r, text=mst_txt, width=14, bg=SURFACE,
                     fg=LABEL, font=FONT_MONO_S, anchor="w").pack(side="left")
            tk.Label(r, text=name, width=22, bg=SURFACE,
                     fg=LABEL2, font=FONT_MONO_S, anchor="w").pack(side="left")
            tk.Label(r, text=("● live" if not stale else "○ stale"), width=8,
                     bg=SURFACE, fg=(GREEN if not stale else YELLOW),
                     font=FONT_MONO_S, anchor="w").pack(side="left")

    def _change_id(self):
        val = self._id_var.get()
        try:
            old_mid = int(val.split()[0][1:])  # "J3  Name" → 3
        except (ValueError, IndexError):
            self._v_id_status.set("⚠ Select a motor"); self._id_status_lbl.config(fg=RED); return
        # New CAN/ESC id — blank means leave unchanged.
        esc_txt = self._new_id_var.get().strip()
        new_esc = None
        if esc_txt:
            try:
                new_esc = int(esc_txt, 0)
            except ValueError:
                self._v_id_status.set("⚠ CAN id must be a number"); self._id_status_lbl.config(fg=RED); return
            if not (1 <= new_esc <= 15):
                self._v_id_status.set("⚠ CAN id must be 1–15"); self._id_status_lbl.config(fg=RED); return
        # New master id — blank means leave unchanged. Accepts hex (0x12) or decimal.
        mst_txt = self._new_mst_var.get().strip()
        new_mst = None
        if mst_txt:
            try:
                new_mst = int(mst_txt, 0)
            except ValueError:
                self._v_id_status.set("⚠ Master id must be a number"); self._id_status_lbl.config(fg=RED); return
            if not (0 <= new_mst <= 0x7FF):
                self._v_id_status.set("⚠ Master id must be 0–0x7FF"); self._id_status_lbl.config(fg=RED); return
        if new_esc is None and new_mst is None:
            self._v_id_status.set("⚠ Enter a new CAN id and/or master id"); self._id_status_lbl.config(fg=RED); return
        self._v_id_status.set("Working…"); self._id_status_lbl.config(fg=ORANGE)
        def _do():
            try:
                self._bus.disable(old_mid); time.sleep(0.1)
                self._bus.set_motor_ids(old_mid, new_esc=new_esc, new_mst=new_mst)
                parts = []
                if new_esc is not None and new_esc != old_mid: parts.append(f"CAN→{new_esc}")
                if new_mst is not None: parts.append(f"MST→0x{new_mst:02X}")
                self._id_table_sig = None   # force the table to redraw
                self._v_id_status.set(f"✓ {', '.join(parts) or 'no change'}  (power-cycle)")
                self._id_status_lbl.config(fg=GREEN)
            except Exception as e:
                self._v_id_status.set(f"✗ {e}"); self._id_status_lbl.config(fg=RED)
        threading.Thread(target=_do, daemon=True).start()

    def refresh(self):
        self._update_id_menu()
        self._rebuild_id_table()
        for mid, row in self._limit_rows.items():
            if mid not in _joint_cal:
                row["v_cal"].set("not calibrated")
                row["cal_lbl"].config(fg=LABEL3)
                row["v_match"].set("")
                continue
            lo_c, hi_c = _joint_cal[mid]
            ji = JOINT_BY_ID[mid]
            lo_t, hi_t = ji["lo"], ji["hi"]
            row["v_cal"].set(f"{lo_c:+.3f} → {hi_c:+.3f} rad")
            row["cal_lbl"].config(fg=LABEL)
            # Check ±0.15 rad tolerance on each end
            ok = abs(lo_c - lo_t) < 0.15 and abs(hi_c - hi_t) < 0.15
            row["v_match"].set("✓ OK" if ok else "⚠ Check")
            row["match_lbl"].config(fg=GREEN if ok else YELLOW)


# ── Calibration Panel ─────────────────────────────────────────────────────────
class CalibrationPanel(tk.Frame):
    def __init__(self, parent, bus: DamiaoBus):
        super().__init__(parent, bg=BG)
        self._bus        = bus
        self._calib_stop = threading.Event()
        self._calib_thr: threading.Thread | None = None
        # Manual (hand-guided) calibration state.
        self._manual_on  = False
        self._man_prev   = {}    # mid -> last position (motion tracking)
        self._man_motion = {}    # mid -> decaying motion accumulator
        self._man_last   = None  # last joint seen moving significantly
        self._man_caps   = {}    # mid -> captured endpoint positions
        self._build()

    def _build(self):
        # ── Motor Internal Calibration ────────────────────────────────────────
        c_int, i_int = _build_card(self, "Motor Internal Calibration")
        c_int.pack(fill="x", pady=(0, 14))

        tk.Label(i_int,
                 text=("Calibrates motor winding resistance and inductance for FOC.\n"
                       "Motor output shaft must be free. Performed via DaMiao Studio "
                       "or by sending the manufacturer calibration frame.\n"
                       "Select a joint and press Run — motor will make noise for ~5 s."),
                 bg=SURFACE, fg=LABEL2, font=FONT_SMALL, wraplength=720,
                 justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        mi_row = tk.Frame(i_int, bg=SURFACE)
        mi_row.pack(fill="x")
        joint_names = [f"J{j['mid']}  {j['name']}" for j in JOINT_INFO]
        self._int_dd = AppleDropDown(
            mi_row, joint_names, default=joint_names[0],
            width=220, height=40, parent_bg=SURFACE)
        self._int_dd.pack(side="left", padx=(0, PAD))
        accent_btn(mi_row, "Run Internal Cal", self._run_internal,
                   color=BLUE, parent_bg=SURFACE).pack(side="left")
        self._v_int_status = tk.StringVar(value="Idle")
        tk.Label(i_int, textvariable=self._v_int_status,
                 bg=SURFACE, fg=LABEL2, font=FONT_MONO_S).pack(anchor="w", pady=(6, 0))

        # ── Joint Limit Calibration ───────────────────────────────────────────
        c_lim, i_lim = _build_card(self, "Joint Limit Calibration")
        c_lim.pack(fill="x")

        tk.Label(i_lim,
                 text=("Homes each joint to BOTH mechanical endpoints. \"All joints\" "
                       "runs J7 → J1, wrist up to the top of the arm.\n"
                       "The joint is driven toward each stop at the sweep velocity "
                       "with the motor torque capped at the probe torque below — "
                       "enough to overcome its own friction and weight, never a slam. "
                       "When it can no longer move (torque saturated, velocity zero) "
                       "that endpoint is recorded. Raise the probe torque if a heavy "
                       "joint won't move; lower it for a gentler touch."),
                 bg=SURFACE, fg=LABEL2, font=FONT_SMALL, wraplength=720,
                 justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        param_row = tk.Frame(i_lim, bg=SURFACE)
        param_row.pack(fill="x", pady=(0, 8))
        tk.Label(param_row, text="Sweep velocity", bg=SURFACE,
                 fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(0, PAD_SM))
        self._v_cal_vel = tk.StringVar(value="0.20")
        tk.Entry(param_row, textvariable=self._v_cal_vel, width=6,
                 bg=ELEVATED, fg=LABEL, insertbackground=LABEL,
                 relief="flat", bd=6, font=FONT_MONO).pack(side="left")
        tk.Label(param_row, text="rad/s", bg=SURFACE,
                 fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(4, PAD_LG))
        tk.Label(param_row, text="Probe torque",
                 bg=SURFACE, fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(0, PAD_SM))
        self._v_cal_probe = tk.StringVar(value="0.30")
        tk.Entry(param_row, textvariable=self._v_cal_probe, width=6,
                 bg=ELEVATED, fg=LABEL, insertbackground=LABEL,
                 relief="flat", bd=6, font=FONT_MONO).pack(side="left")
        tk.Label(param_row, text="× rated  (raise if a joint won't move)", bg=SURFACE,
                 fg=LABEL2, font=FONT_BODY).pack(side="left", padx=(4, 0))

        sel_row = tk.Frame(i_lim, bg=SURFACE)
        sel_row.pack(fill="x", pady=(0, 8))
        tk.Label(sel_row, text="Joint", bg=SURFACE, fg=LABEL2,
                 font=FONT_BODY).pack(side="left", padx=(0, PAD))
        # Joints listed J7 → J1 (wrist up to the top), torso last — matching the
        # calibration sweep order.
        _cal_order = sorted((j for j in JOINT_INFO if j["mid"] <= 7),
                            key=lambda j: -j["mid"]) + \
                     [j for j in JOINT_INFO if j["mid"] > 7]
        self._cal_dd = AppleDropDown(
            sel_row, ["All joints"] + [f"J{j['mid']}  {j['name']}" for j in _cal_order],
            default="All joints", width=220, height=40, parent_bg=SURFACE)
        self._cal_dd.pack(side="left", padx=(0, PAD))
        accent_btn(sel_row, "▶  Start", self._start_cal,
                   parent_bg=SURFACE).pack(side="left", padx=(0, PAD_SM))
        btn(sel_row, "■  Stop", self._stop_cal, parent_bg=SURFACE).pack(side="left")

        tk.Frame(i_lim, bg=SEP, height=1).pack(fill="x", pady=(PAD_SM, PAD_SM))

        self._v_cal_status = tk.StringVar(value="Idle")
        tk.Label(i_lim, textvariable=self._v_cal_status, bg=SURFACE,
                 fg=ORANGE, font=FONT_MONO_S, anchor="w").pack(fill="x")

        # ── Manual (hand-guided) calibration ──────────────────────────────────
        tk.Frame(i_lim, bg=SEP, height=1).pack(fill="x", pady=(PAD_SM, PAD_SM))
        tk.Label(i_lim, text="MANUAL  (HAND-GUIDED)", bg=SURFACE,
                 fg=LABEL2, font=F_CAPTION, anchor="w").pack(fill="x")
        tk.Label(i_lim, text="Releases every motor to zero torque so you can move a "
                 "joint by hand. Move it to one end and press Confirm, then to the "
                 "other end and press Confirm — the joint is detected automatically.",
                 bg=SURFACE, fg=LABEL3, font=FONT_SMALL, wraplength=720,
                 justify="left", anchor="w").pack(fill="x", pady=(2, 6))
        man_row = tk.Frame(i_lim, bg=SURFACE)
        man_row.pack(fill="x", pady=(0, 4))
        self._man_start_btn = AppleButton(
            man_row, "Start Manual  (release torque)", self._toggle_manual,
            bg_color=BLUE, fg_color=BG, font=F_CAPTION,
            parent_bg=SURFACE, width=230, height=32, radius=10)
        self._man_start_btn.pack(side="left", padx=(0, PAD_SM))
        self._man_confirm_btn = AppleButton(
            man_row, "Confirm limit", self._manual_confirm,
            bg_color=GREEN, fg_color=BG, font=F_CAPTION,
            parent_bg=SURFACE, width=130, height=32, radius=10)
        self._man_confirm_btn.pack(side="left")
        self._v_man_detect = tk.StringVar(value="")
        tk.Label(i_lim, textvariable=self._v_man_detect, bg=SURFACE, fg=LABEL2,
                 font=FONT_MONO_S, anchor="w").pack(fill="x", pady=(4, 0))
        self._v_man_status = tk.StringVar(value="Manual mode off")
        tk.Label(i_lim, textvariable=self._v_man_status, bg=SURFACE, fg=ORANGE,
                 font=FONT_MONO_S, anchor="w").pack(fill="x")

        tk.Frame(i_lim, bg=SEP, height=1).pack(fill="x", pady=(PAD_SM, PAD_SM))
        tk.Label(i_lim, text="RESULTS vs THEORETICAL LIMITS",
                 bg=SURFACE, fg=LABEL2, font=F_CAPTION, anchor="w").pack(fill="x")

        hdr2 = tk.Frame(i_lim, bg=SURFACE)
        hdr2.pack(fill="x", pady=(4, 4))
        for text, w in [("Joint", 18), ("Calibrated", 22), ("Theoretical", 22),
                        ("Δ lo", 8), ("Δ hi", 8), ("Status", 10)]:
            tk.Label(hdr2, text=text, width=w, bg=SURFACE,
                     fg=LABEL2, font=F_CAPTION, anchor="w").pack(side="left")
        tk.Frame(i_lim, bg=SEP, height=1).pack(fill="x", pady=(0, 4))

        self._result_rows: dict[int, dict] = {}
        for ji in JOINT_INFO:
            mid  = ji["mid"]
            row  = tk.Frame(i_lim, bg=SURFACE)
            row.pack(fill="x", pady=2)
            vars_ = {k: tk.StringVar(value="──") for k in ("cal", "theory", "dlo", "dhi", "st")}
            vars_["theory"].set(f"{ji['lo']:+.3f} → {ji['hi']:+.3f}")
            lbls_ = {}
            for key, w in [("cal", 22), ("theory", 22), ("dlo", 8), ("dhi", 8), ("st", 10)]:
                l = tk.Label(row, textvariable=vars_[key], width=w,
                             bg=SURFACE, fg=LABEL2, font=FONT_MONO_S, anchor="w")
                l.pack(side="left")
                lbls_[key] = l
            tk.Label(row, text=f"J{mid}  {ji['name']}", width=18,
                     bg=SURFACE, fg=LABEL2, font=FONT_BODY, anchor="w").pack(side="left",
                     before=list(row.winfo_children())[0])
            self._result_rows[mid] = {"vars": vars_, "labels": lbls_}

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _run_internal(self):
        txt = self._int_dd.get()
        mid = int(txt.split()[0][1:])
        self._v_int_status.set(f"J{mid}: sending internal calibration frame…")
        # DaMiao internal calibration: 0x7FF with [mid_L, mid_H, 0x55, 0x06, ...]
        # Frame based on DaMiao SDK documentation
        def _do():
            try:
                self._bus._transport.send(
                    GLOBAL_ID, [mid & 0xFF, (mid >> 8) & 0xFF, 0x55, 0x06,
                                0x00, 0x00, 0x00, 0x00])
                self._bus.log(f"Motor {mid}: internal calibration frame sent")
                self._v_int_status.set(f"J{mid}: calibration frame sent — listen for ~5 s of motor noise")
            except Exception as e:
                self._v_int_status.set(f"J{mid}: error — {e}")
        threading.Thread(target=_do, daemon=True).start()

    def _start_cal(self):
        if self._calib_thr and self._calib_thr.is_alive():
            self._bus.log("Calibration already running")
            return
        sel = self._cal_dd.get()
        # "All joints" → J7 down to J1 (wrist up to the top of the arm); torso (J8)
        # is calibrated only when explicitly selected.
        mids = sorted((j["mid"] for j in JOINT_INFO if j["mid"] <= 7), reverse=True) \
               if sel == "All joints" else [int(sel.split()[0][1:])]
        try:
            vel   = float(self._v_cal_vel.get())
            probe = float(self._v_cal_probe.get())
        except ValueError:
            self._v_cal_status.set("⚠ Invalid velocity or probe-torque value"); return

        self._calib_stop.clear()
        self._v_cal_status.set("Starting…")

        def _run():
            for mid in mids:
                if self._calib_stop.is_set():
                    self._v_cal_status.set("Cancelled")
                    return
                self._bus.calibrate_joint(
                    mid, vel, probe,
                    on_status=lambda msg: self._v_cal_status.set(msg),
                    on_done=self._on_cal_done,
                    stop_event=self._calib_stop)
            if not self._calib_stop.is_set():
                self._v_cal_status.set("Calibration complete")

        self._calib_thr = threading.Thread(target=_run, daemon=True)
        self._calib_thr.start()

    def _stop_cal(self):
        self._calib_stop.set()
        self._v_cal_status.set("Stopping…")
        with self._bus._lock:
            for m in self._bus.motors.values():
                if m.enabled:
                    m.vel_target = 0.0
                    m.vel_ff     = 0.0
                    m.t_ff       = 0.0
                    m.pos_target = m.pos   # drop any probe lead so it stops pushing

    def _on_cal_done(self, mid: int, lo, hi):
        if lo is None or hi is None:
            return
        # Normalise ordering so lo < hi regardless of the motor's rotation sign —
        # this is what fixes J1's reversed limits (its encoder runs the opposite
        # way, so the "negative" probe lands at the physically higher position).
        lo, hi = min(lo, hi), max(lo, hi)
        _joint_cal[mid] = (lo, hi)
        ji  = JOINT_BY_ID[mid]
        row = self._result_rows.get(mid)
        if row is None:
            return
        v, lb = row["vars"], row["labels"]
        v["cal"].set(f"{lo:+.3f} → {hi:+.3f}")
        lb["cal"].config(fg=LABEL)
        dlo = lo - ji["lo"]
        dhi = hi - ji["hi"]
        v["dlo"].set(f"{dlo:+.3f}")
        v["dhi"].set(f"{dhi:+.3f}")
        ok = abs(dlo) < 0.15 and abs(dhi) < 0.15
        v["st"].set("✓ OK" if ok else "⚠ Check")
        lb["st"].config(fg=GREEN if ok else YELLOW)
        lb["dlo"].config(fg=GREEN if abs(dlo) < 0.15 else YELLOW)
        lb["dhi"].config(fg=GREEN if abs(dhi) < 0.15 else YELLOW)

    # ── Manual (hand-guided) calibration ──────────────────────────────────────
    def _toggle_manual(self):
        if self._manual_on:
            self._manual_on = False
            self._man_start_btn.update_text("Start Manual  (release torque)")
            self._man_start_btn.update_colors(BLUE, BG)
            self._v_man_status.set("Manual mode off")
            self._v_man_detect.set("")
            return
        if self._calib_thr and self._calib_thr.is_alive():
            self._v_man_status.set("⚠ Stop the auto sweep first")
            return
        # Release every found motor to zero torque so the joints are backdrivable.
        with self._bus._lock:
            mids = [mid for mid, mo in self._bus.motors.items() if mo.found]
        for mid in mids:
            self._bus.disable(mid)
        self._man_prev = {}; self._man_motion = {}
        self._man_last = None; self._man_caps = {}
        self._manual_on = True
        self._man_start_btn.update_text("Stop Manual")
        self._man_start_btn.update_colors(RED, LABEL)
        self._v_man_status.set("Move a joint by hand to one limit, then press Confirm.")

    def _manual_active_mid(self):
        """The joint the user is currently / most recently moving."""
        if self._man_motion:
            mid, mv = max(self._man_motion.items(), key=lambda kv: kv[1])
            if mv > 0.02:
                return mid
        return self._man_last

    def _manual_confirm(self):
        if not self._manual_on:
            self._v_man_status.set("⚠ Press Start Manual first")
            return
        mid = self._manual_active_mid()
        if mid is None:
            self._v_man_status.set("⚠ Move a joint first so it can be detected")
            return
        with self._bus._lock:
            m = self._bus.motors.get(mid)
            pos = m.pos if m else None
        if pos is None:
            self._v_man_status.set("⚠ Joint not found on bus")
            return
        name = JOINT_BY_ID.get(mid, {}).get("name", f"J{mid}")
        caps = self._man_caps.setdefault(mid, [])
        caps.append(pos)
        if len(caps) == 1:
            self._v_man_status.set(
                f"J{mid} {name}: end 1 = {pos:+.3f} rad — move to the OTHER end and Confirm.")
        else:
            lo, hi = min(caps[-2], caps[-1]), max(caps[-2], caps[-1])
            self._on_cal_done(mid, lo, hi)     # stores normalised range + updates toggles
            self._v_man_status.set(
                f"J{mid} {name} calibrated: {lo:+.3f} → {hi:+.3f} rad. Move the next joint.")
            self._man_caps[mid] = []
            self._man_motion = {}; self._man_last = None   # re-arm detection

    def refresh(self):
        if not self._manual_on:
            return
        with self._bus._lock:
            snap = {mid: mo.pos for mid, mo in self._bus.motors.items() if mo.found}
        for mid, pos in snap.items():
            d = abs(pos - self._man_prev.get(mid, pos))
            self._man_prev[mid] = pos
            mv = self._man_motion.get(mid, 0.0) * 0.85 + d
            self._man_motion[mid] = mv
            if mv > 0.02:
                self._man_last = mid
        a = self._manual_active_mid()
        if a is not None and a in snap:
            nm  = JOINT_BY_ID.get(a, {}).get("name", f"J{a}")
            n   = len(self._man_caps.get(a, []))
            self._v_man_detect.set(
                f"detected: J{a} {nm}  pos={snap[a]:+.3f} rad  (captured {n}/2)")
        else:
            self._v_man_detect.set("detected: —  (move a joint by hand)")


# ── Main App ──────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self, channel: str = "can0", interface: str | None = None,
                 port: str | None = None):
        super().__init__()
        self._bus        = DamiaoBus(channel, interface=interface, port=port)
        # Header label: show the live backend + endpoint (e.g. "serial:/dev/cu.usb…",
        # "gs_usb:canable2" on macOS, or just "can0" on Linux SocketCAN).
        if self._bus._is_serial:
            self._channel = f"serial:{self._bus._port}"
        elif self._bus._interface == "socketcan":
            self._channel = channel
        else:
            self._channel = f"{self._bus._interface}:{channel}"
        self._active_mid: int | None = 1   # default to J1
        self._active_nav = "Control"
        self._nav_btns: dict[str, AppleButton] = {}
        self._panels:   dict = {}

        self.title("MABEL OpenArm Teleop")
        self.configure(bg=BG)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1100, 700)

        self._build_toolbar()
        self._build_body()
        self._bind_keys()

        # Graceful shutdown on Ctrl-C / kill: a signal handler can't safely touch
        # Tk, so it just flips a flag that an after() poll notices and closes on
        # the main thread — this releases the gs_usb USB device cleanly.
        self._sig_close = False
        for _sig in (signal.SIGINT, signal.SIGTERM):
            try: signal.signal(_sig, lambda *_: setattr(self, "_sig_close", True))
            except (ValueError, OSError): pass  # not main thread / unsupported
        self.after(120, self._poll_signal)

        self.after(80,  self._refresh_loop)
        self.after(300, lambda: threading.Thread(
            target=self._bus.scan, daemon=True).start())

    def _poll_signal(self):
        if self._sig_close:
            self._on_close()
            return
        self.after(120, self._poll_signal)

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=85)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        title_f = tk.Frame(bar, bg=SURFACE)
        title_f.pack(side="left", padx=25, pady=14)
        tk.Label(title_f, text="MABEL OpenArm", bg=SURFACE,
                 fg=LABEL,  font=FONT_TITLE).pack(anchor="w")
        tk.Label(title_f, text="Joint Teleop Studio", bg=SURFACE,
                 fg=ORANGE, font=FONT_SMALL).pack(anchor="w")

        nav_f = tk.Frame(bar, bg=SURFACE)
        nav_f.pack(side="left", padx=20, pady=22)
        for name in ["Control", "Setup", "Calibration"]:
            b = AppleButton(nav_f, name, lambda n=name: self._switch_panel(n),
                            bg_color=ELEVATED, fg_color=LABEL2,
                            font=FONT_BODY, parent_bg=SURFACE,
                            width=120, height=40)
            b.pack(side="left", padx=5)
            self._nav_btns[name] = b
        self._mark_nav("Control")

        right = tk.Frame(bar, bg=SURFACE)
        right.pack(side="right", padx=25, pady=22)

        self._status_dot = tk.Label(right, text="●", fg=RED,
                                     bg=SURFACE, font=("Helvetica", 16))
        self._status_dot.pack(side="left", padx=(0, 6))
        # CAN channel chip
        tk.Label(right, text=self._channel, bg=ELEVATED, fg=LABEL2,
                 font=FONT_MONO_S, padx=10, pady=4).pack(side="left", padx=(0, PAD))
        # Scan Motors
        AppleButton(right, "↺  Scan Motors",
                    lambda: threading.Thread(target=self._bus.scan, daemon=True).start(),
                    bg_color=ELEVATED, fg_color=LABEL, font=FONT_BODY,
                    parent_bg=SURFACE, width=130, height=40).pack(side="left", padx=6)
        # Connect / Disconnect (label tracks live state in _refresh_loop)
        self._conn_btn = AppleButton(
            right, "Disconnect", self._toggle_connect,
            bg_color=ELEVATED, fg_color=LABEL, font=FONT_BODY,
            parent_bg=SURFACE, width=120, height=40)
        self._conn_btn.pack(side="left", padx=6)
        # E-STOP
        AppleButton(right, "■  E-STOP", self._estop,
                    bg_color=RED, fg_color=LABEL, font=FONT_BODY,
                    parent_bg=SURFACE, width=120, height=40).pack(side="left", padx=6)

        tk.Frame(self, bg=SEP, height=1).pack(fill="x")

    def _toggle_connect(self):
        if self._bus._connected:
            self._bus.disconnect()
        else:
            threading.Thread(target=self._bus.connect, daemon=True).start()

    def _mark_nav(self, name: str):
        for n, b in self._nav_btns.items():
            b.update_colors(ORANGE if n == name else ELEVATED,
                            LABEL  if n == name else LABEL2)

    # ── Body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._sidebar = TelemetrySidebar(body, self._bus, self._set_active)
        self._sidebar.pack(side="left", fill="y")
        tk.Frame(body, bg=SEP, width=1).pack(side="left", fill="y")

        # ── Scrollable content area ────────────────────────────────────────
        scroll_outer = tk.Frame(body, bg=BG)
        scroll_outer.pack(side="left", fill="both", expand=True)

        self._scroll_canvas = tk.Canvas(scroll_outer, bg=BG, highlightthickness=0)
        vbar = tk.Scrollbar(scroll_outer, orient="vertical",
                            command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)

        # Inner frame — panels live here
        self._content = tk.Frame(self._scroll_canvas, bg=BG)
        self._cwin = self._scroll_canvas.create_window(
            (0, 0), window=self._content, anchor="nw")

        # Keep scroll region and frame width in sync
        self._content.bind("<Configure>", self._on_inner_configure)
        self._scroll_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse-wheel scrolling (Linux uses Button-4/5)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(seq, self._on_mousewheel)

        self._panels["Control"]     = ControlPanel(
            self._content, self._bus, self._get_active, self._set_active)
        self._panels["Setup"]       = SetupPanel(
            self._content, self._bus, self._get_active)
        self._panels["Calibration"] = CalibrationPanel(
            self._content, self._bus)
        self._switch_panel("Control")

    def _on_inner_configure(self, _event):
        self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Stretch the inner frame to fill the canvas width
        self._scroll_canvas.itemconfig(self._cwin, width=event.width)

    def _on_mousewheel(self, event):
        # Don't steal scroll from Text widgets (event log)
        if str(event.widget.winfo_class()) in ("Text", "Listbox"):
            return
        if event.num == 4:       # Linux scroll-up
            self._scroll_canvas.yview_scroll(-2, "units")
        elif event.num == 5:     # Linux scroll-down
            self._scroll_canvas.yview_scroll(2, "units")
        elif event.delta:
            self._scroll_canvas.yview_scroll(int(-event.delta / 60), "units")

    def _switch_panel(self, name: str):
        self._active_nav = name
        self._mark_nav(name)
        for p in self._panels.values():
            p.pack_forget()
        self._panels[name].pack(fill="x", padx=24, pady=24)
        self._scroll_canvas.yview_moveto(0)   # scroll back to top on tab switch

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _bind_keys(self):
        # bind_all fires regardless of which widget has focus (button, slider, etc.)
        # Guard only excludes Entry/Text where the user types text intentionally.
        def _guard(fn):
            def _inner(e):
                if str(e.widget.winfo_class()) in ("Entry", "Text"):
                    return
                fn()
            return _inner
        ctrl = self._panels["Control"]
        self.bind_all("<Left>",  _guard(ctrl.key_left))
        self.bind_all("<Right>", _guard(ctrl.key_right))
        self.bind_all("<Up>",    _guard(ctrl.key_up))
        self.bind_all("<Down>",  _guard(ctrl.key_down))
        self.bind_all("<s>", _guard(lambda: threading.Thread(
                                 target=self._bus.scan, daemon=True).start()))
        self.bind_all("<S>", _guard(lambda: threading.Thread(
                                 target=self._bus.scan, daemon=True).start()))
        self.bind_all("<q>", _guard(self._on_close))
        self.bind_all("<Q>", _guard(self._on_close))

    # ── Actions ───────────────────────────────────────────────────────────────
    def _set_active(self, mid: int):
        self._active_mid = mid
        self.focus_set()  # pull focus back to root so bind_all keys work immediately

    def _get_active(self) -> Motor | None:
        if self._active_mid is None:
            return None
        with self._bus._lock:
            return self._bus.motors.get(self._active_mid)

    def _estop(self):
        self._bus.log("E-STOP — disabling all motors")
        with self._bus._lock:
            ids = list(self._bus.motors.keys())
        for mid in ids:
            self._bus.hard_disable(mid)

    def _on_close(self):
        self._bus.log("Shutting down…")
        self._bus.shutdown()
        self.destroy()

    # ── Refresh loop ──────────────────────────────────────────────────────────
    def _refresh_loop(self):
        try:
            now = time.perf_counter()

            with self._bus._lock:
                motors = list(self._bus.motors.items())

            # Slow discovery sweep (~2 Hz) so newly connected / reconnected motors
            # are picked up without flooding the bus with polls for absent IDs.
            self._disc_tick = getattr(self, "_disc_tick", 0) + 1
            do_discovery = self._disc_tick >= max(1, CTRL_HZ // 2)
            if do_discovery:
                self._disc_tick = 0

            for mid, m in motors:
                if m.enabled:
                    # Torque control: MIT frames carry the command AND elicit
                    # feedback. Only auto-disable on a sustained silence (>1.0 s),
                    # and never within 1.5 s of enabling — otherwise a brief USB
                    # hiccup on a flaky adapter would kill torque right after Enable.
                    silent = m.t_fb > 0 and (now - m.t_fb) > 1.0
                    in_grace = (m._enabled_at is not None
                                and now - m._enabled_at < 1.5)
                    if silent and not in_grace:
                        self._bus.log(f"J{m.mid} feedback lost >1s — torque auto-disabled")
                        self._bus.disable(m.mid)
                        continue
                    # Stall auto-stop: velocity mode only, stall > 0.4 s
                    if (m.mode == "velocity" and abs(m.vel) < 0.05
                            and abs(m.tau) > 0.3 * m.lim["t_max"]):
                        if m._stall_since is None:
                            m._stall_since = now
                        elif now - m._stall_since > 0.4:
                            self._bus.log(f"J{m.mid} stall — auto-stop")
                            m.vel_target   = 0.0
                            m.mode         = "position"
                            m.pos_target   = m.pos
                            m._stall_since = None
                    else:
                        m._stall_since = None
                    try: self._bus.send_command(m)
                    except Exception: pass
                elif m.found:
                    # Found but not enabled → stream telemetry at full loop rate
                    # via 0xCC refresh. No torque, motor free/backdrivable.
                    self._bus.refresh(mid)
                elif do_discovery:
                    # Not yet seen → gentle discovery poll.
                    self._bus.refresh(mid)

            connected = self._bus._connected
            with self._bus._lock:
                any_live = any(m.found and not m.stale
                               for m in self._bus.motors.values())
            self._status_dot.config(fg=GREEN if (connected and any_live)
                                    else (ORANGE if connected else RED))
            # Keep the Connect/Disconnect button label in sync with bus state
            self._conn_btn.update_text("Disconnect" if connected else "Connect")
            self._conn_btn.update_colors(ELEVATED if connected else GREEN,
                                         LABEL if connected else BG)

            self._sidebar.refresh(self._active_mid)
            panel = self._panels.get(self._active_nav)
            if panel:
                if self._active_nav == "Control":
                    panel.refresh(self._active_mid)
                else:
                    panel.refresh()

        except Exception as exc:
            try: self._bus.log(f"UI error: {exc}")
            except Exception: pass
        finally:
            self.after(1000 // CTRL_HZ, self._refresh_loop)


def main():
    ap = argparse.ArgumentParser(description="MABEL OpenArm Teleop GUI")
    ap.add_argument("--channel", default=None,
                    help="CAN channel. SocketCAN iface name on Linux (default "
                         "can0); USB device label on macOS/gs_usb (default canable2)")
    ap.add_argument("--interface", default=None,
                    choices=["socketcan", "gs_usb", "serial"],
                    help="CAN backend. 'serial' = DaMiao USB-to-CAN adapter "
                         "(official protocol). Default: serial if a DaMiao serial "
                         "adapter is present on macOS, else gs_usb; socketcan on Linux")
    ap.add_argument("--port", default=None,
                    help="serial device for --interface serial "
                         "(default: auto-detect, e.g. /dev/cu.usbmodem*)")
    args = ap.parse_args()

    interface = args.interface
    if interface is None:
        if _DARWIN:
            # Prefer the DaMiao serial adapter when one is plugged in — it's the
            # most reliable path on macOS — otherwise fall back to gs_usb.
            interface = "serial" if (args.port or _find_serial_port()) else "gs_usb"
        else:
            interface = "socketcan"

    channel = args.channel or ("canable2" if interface == "gs_usb" else "can0")
    app = App(channel=channel, interface=interface, port=args.port)
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
