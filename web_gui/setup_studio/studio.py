#!/usr/bin/env python3
"""
MABEL Hardware Setup Studio — per-module status · control · calibration with the 3-D rig.

Built on the hardware layer the HAL itself uses (``hardware_bridge/mabel_hw``): the studio
opens ONE ``MabelHardware`` (the same flock singleton as ``hardware_bridge.py``, so it
refuses to start while the HAL owns the devices), claims an exclusive writer lease per
actuator group, and drives everything through the ``mabel_hw`` subsystems — arms, torso and
neck over the arms Teensy (openarm-dual firmware), the ORCA hands over their Feetech
adapters, the Pico lift over USB serial. Every calibration result is written into the
existing ``hardware_bridge/calibration_data/mabel.json`` slices through
``mabel_hw.calibration.CalibrationStore`` (arm/openarm-dual, torso/damiao, neck/teensy,
hand/orca-<side>, lift/pico) in the exact entry format the HAL reads
(``raw_min``/``raw_max``/``dir``/``model_mid``/``reliability``, centre-anchored).

    python web_gui/setup_studio/studio.py          # real hardware  ->  http://127.0.0.1:8300
    python web_gui/setup_studio/studio.py --sim    # mabel_hw mock plants, no hardware

Torque policy: nothing is energised by this page except from the Control tab. A joint group
at rest is torque OFF — on the openarm backend that is the zero-torque damping hold
(kp = t_ff = 0), on the Feetech hands free-drive. E-STOP, a lost page (no poll for
HB_MAX s) and quitting the studio all drop every group back to that state; the lift halts,
the base stays latched in its e-stop (this studio never drives the base).
"""
import argparse, json, math, os, shutil, sys, tempfile, threading, time, webbrowser
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BRIDGE = os.path.join(REPO, "hardware_bridge")
SHARED = os.path.join(REPO, "web_gui", "shared_assets")
sys.path.insert(0, BRIDGE)

from mabel_hw import MabelHardware                       # noqa: E402
from mabel_hw import config as C                         # noqa: E402
from mabel_hw.errors import HardwareError                # noqa: E402
from mabel_hw.subsystems.joint_group import JointGroupSubsystem   # noqa: E402

OWNER = "setup-studio"
HB_MAX, MIN_SPAN = 1.5, 0.14              # s browser deadman; rad a capture must sweep (~8 deg)
TAU = 2.0 * math.pi
TICK = TAU / 4096                         # Feetech single-turn tick, rad
LIFT_JOG = 0.03                           # m/s open-loop jog speed at full deflection
MIME = {"js": "text/javascript", "glb": "model/gltf-binary", "json": "application/json",
        "html": "text/html; charset=utf-8"}
THUMB = {"thumb_cmc": "thumb_mcp", "thumb_mp": "thumb_dip", "thumb_ip": "thumb_pip"}   # ORCA -> URDF


def _wrap_pi(x):
    return (x + math.pi) % TAU - math.pi


def urdf_name(n, side=None):
    """Studio joint name -> the rig / URDF joint name in shared_assets/mabel_joints.json."""
    if len(n) == 4 and n[1:3] == "_j" and n[0] in "lr":
        return ("left" if n[0] == "l" else "right") + "_arm_" + n[3]
    if side and n.startswith(side + "_"):
        bare = n[len(side) + 1:]
        return f"{side}_{THUMB.get(bare, bare)}"
    return n


def load_ranges():
    """{urdf joint: [lower, upper]} from the MJCF-derived joint manifest the 3-D rig uses."""
    try:
        with open(os.path.join(SHARED, "mabel_joints.json")) as f:
            return {j["name"]: [float(j["lower"]), float(j["upper"])]
                    for j in json.load(f)["joints"] if j.get("type") != "continuous"}
    except (OSError, ValueError, KeyError):
        return {}


# --------------------------------------------------------------------------- joint groups
class JointModule:
    """A calibrated MIT joint group (left/right arm, torso, neck) through its mabel_hw subsystem.
    Rest state = torque OFF (the openarm damping hold, kp = 0). Torque is group-wide."""
    kind = "can"

    def __init__(self, S, key, label, hw_name, cfg_name, anchor="center", note=""):
        self.S, self.key, self.label, self.anchor, self.note = S, key, label, anchor, note
        self.side = None
        self.cfg = getattr(C, cfg_name)
        self.slice = self.cfg.get("calib_slice")
        self.sub = getattr(S.hw, "_" + hw_name, None)
        self.handle = getattr(S.hw, hw_name)
        self.names = list(self.sub.names) if self.sub else list(self.cfg.get("joints") or [])
        self.writer = None
        if self.sub is not None:
            try:
                self.writer = self.handle.claim(OWNER)
            except Exception as e:                       # noqa: BLE001
                S.ev("hw", f"{label}: cannot claim the writer lease ({e})", False)
        self.fb, self.model, self.goal, self.health = {}, {}, {}, {}
        self.cap, self.lock, self._t_health = None, threading.Lock(), 0.0
        self.rate, self._n, self._t0 = 0.0, 0, time.monotonic()

    def bare(self, n):
        return n

    # ---- reads (studio poll thread) ----
    def poll(self):
        if self.sub is None:
            return
        try:
            fb = self.sub._driver.read()                         # RAW motor radians (cached, never blocks)
            ms = JointGroupSubsystem.state(self.sub)             # MODEL frame through the live calibration
        except Exception:                                        # noqa: BLE001
            return
        now = time.time()
        with self.lock:
            for i, n in enumerate(fb.names):
                self.fb[n] = dict(raw=fb.position[i], vel=fb.velocity[i], tau=fb.effort[i],
                                  temp=fb.temperature[i], live=bool(fb.live[i]), t=now)
            self.model = dict(zip(ms.names, ms.position))
            if self.cap:
                for n, x, live in zip(fb.names, fb.position, fb.live):
                    c = self.cap.get(n)
                    if c is None or not live:
                        continue
                    if c["last"] is None:
                        c.update(acc=x, min=x, max=x)
                    elif abs(x - c["last"]) < 1.0:               # seam-safe: skip encoder wrap jumps >= 1 rad
                        c["acc"] += x - c["last"]
                        c["min"], c["max"] = min(c["min"], c["acc"]), max(c["max"], c["acc"])
                    c["last"] = x
        self._n += 1
        if time.monotonic() - self._t_health > 1.0:
            self._t_health = time.monotonic()
            try:
                self.health = self.sub.health() or {}
            except Exception:                                    # noqa: BLE001
                self.health = {}
            dt = time.monotonic() - self._t0
            self.rate, self._n, self._t0 = self._n / dt if dt > 0 else 0.0, 0, time.monotonic()

    def live(self, n):
        return bool((self.fb.get(n) or {}).get("live"))

    def raw(self, n):
        return (self.fb.get(n) or {}).get("raw")

    def connected(self):
        return bool(self.sub and self.sub.connected())

    def on(self):
        return bool(self.sub and self.sub.is_torque_enabled)

    def calibrated(self, n):
        try:
            return self.S.store.joint(self.slice, self.bare(n)) is not None
        except Exception:                                        # noqa: BLE001 (a malformed entry must not wedge the studio)
            return False

    def drivable(self):
        return [n for n in self.names if self.live(n) and self.calibrated(n)]

    # ---- writes (lease-gated writer) ----
    def torque(self, on, names=None):
        if self.writer is None:
            raise RuntimeError(f"{self.label}: no driver")
        if on:
            if self.S.estop:
                raise RuntimeError("E-STOP engaged — RE-ENGAGE first")
            if not self.drivable():
                raise RuntimeError("no drivable joint (needs: answering + calibrated)")
            self.cap = None
            self.goal = {}
            self.writer.set_torque(True)                         # the driver latches a hold at the MEASURED pose
            if not self.sub.is_torque_enabled:
                raise RuntimeError(f"{self.label}: torque enable refused by the driver — see the HAL log line in Events")
            self.S.ev("torque", f"{self.label}: torque ON — holding at the measured pose")
        else:
            self.writer.set_torque(False)
            self.goal = {}
            self.S.ev("torque", f"{self.label}: torque off — zero-torque damping hold")
        return self.on()

    def release(self, why):
        """Back to the rest state, never raising (E-STOP / deadman / shutdown)."""
        if self.writer is not None and self.on():
            try:
                self.writer.set_torque(False)
                self.S.ev("torque", f"{self.label}: torque off ({why})", False)
            except Exception as e:                               # noqa: BLE001
                self.S.ev("torque", f"{self.label}: torque-off failed: {e}", False)
        self.goal = {}

    def goals(self, goals):
        if self.S.estop or not self.on():
            return
        ok = {n: float(v) for n, v in goals.items() if n in self.names and self.calibrated(n)}
        if ok:
            self.writer.set_joint_positions(ok)                  # MODEL frame; window clamp + end-stop margin inside
            self.goal.update(ok)

    def capture(self, names):
        names = [n for n in names if n in self.names and self.live(n)]
        if not names:
            raise RuntimeError("no answering joint to capture")
        if self.on():
            self.torque(False)                                   # capture is a hand-move: the group goes limp
        with self.lock:
            self.cap = {n: dict(last=None, acc=None, min=None, max=None) for n in names}
        self.S.ev("calib", f"{self.label}: recording {', '.join(names)} — move to both stops by hand")
        return names

    def capture_end(self, save):
        with self.lock:
            cap, self.cap = self.cap, None
        if not save or not cap:
            return []
        saved = []
        for n, c in cap.items():
            if c["min"] is not None and c["max"] - c["min"] >= MIN_SPAN:
                self.S.save_window(self, n, c["min"], c["max"]); saved.append(n)
        self.S.ev("calib", f"saved {len(saved)} joint range(s): {', '.join(saved)}" if saved else
                  "nothing saved — no joint swept >= 8°", bool(saved))
        return saved

    def sim_move(self, n, raw):
        drv = self.sub._driver
        with drv._lock:
            drv._pos[n] = float(raw)
            drv._targets[n] = float(raw)

    # ---- snapshot ----
    def cfg_row(self, n):
        kp, kd = self.cfg.get("kp"), self.cfg.get("kd")
        pick = lambda g: (g.get(n, g.get("default")) if isinstance(g, dict) else g)   # noqa: E731
        return dict(esc=None, model=None, kp=pick(kp) if kp is not None else None,
                    kd=pick(kd) if kd is not None else None, vmax=self.cfg.get("slew"))

    def view(self):
        out = dict(label=self.label, kind=self.kind, side=self.side, note=self.note, slice=self.slice,
                   anchor=self.anchor, bus=None, port=None, connected=False, detected=False, esc=None,
                   on=False, rate_hz=0, found=[], joints={}, cfg={}, auto=None, cap=None,
                   backend=self.cfg.get("backend"), margin=self.cfg.get("end_stop_margin"))
        for n in self.names:
            out["cfg"][n] = self.joint_cfg(n)
        if self.sub is None:
            return out
        on = self.on()
        with self.lock:
            for n in self.names:
                f = self.fb.get(n) or {}
                h = self.health.get(n) or {}
                live, cal = bool(f.get("live")), self.calibrated(n)
                jc = self.S.store.joint(self.slice, self.bare(n)) if cal else None
                g = self.goal.get(n)
                out["joints"][n] = dict(
                    live=live, pos=f.get("raw") if live else None, vel=f.get("vel"), tau=f.get("tau"),
                    temp=round(f["temp"]) if live and f.get("temp") else None, t=f.get("t", 0),
                    on=on, relax=not on, limits=live, err=h.get("err"), state=h.get("fault") or ("ok" if live else "silent"),
                    goal=jc.raw_of(g) if (jc is not None and g is not None) else None,
                    sp=jc.raw_of(g) if (jc is not None and g is not None) else None,
                    model_pos=self.model.get(n) if (cal and live) else None, esc=None, kind=self.kind, link=None)
            cap = self.cap and {n: {k: v for k, v in c.items()} for n, c in self.cap.items()}
        link = self.cfg.get("link")
        lk = (C.LINKS.get(link) or {}) if link else {}
        out.update(connected=self.connected(), detected=any(j["live"] for j in out["joints"].values()),
                   on=on, rate_hz=round(self.rate, 1), cap=cap,
                   bus=f"{self.cfg.get('backend')}" + (f" · {link}" if link else ""),
                   port=("SIM" if self.S.sim else (f"{lk.get('ip')}:{lk.get('port')}" if lk else "")))
        return out

    def joint_cfg(self, n):
        cal = self.S.store.slice(f"{self.slice}/calib/{self.bare(n)}") if self.slice else None
        return dict(self.cfg_row(n), range=self.S.model_range(self, n),
                    calib=cal if isinstance(cal, dict) else None, kind=self.kind)


# --------------------------------------------------------------------------- hands
class HandModule(JointModule):
    """One ORCA hand (Feetech servos on a USB-TTL adapter) through mabel_hw's HandSubsystem.
    Rest = torque OFF (free-drive). Calibration is the centre-anchored studio map the ORCA driver
    reads from hand/orca-<side>/calib: angle = model_mid + dir * wrap_pi(raw_rad - raw_mid)."""
    kind = "orca"

    def __init__(self, S, side):
        self.S, self.side, self.anchor = S, side, "center"
        self.key, self.label = f"hand-{side}", f"Hand · {side[0].upper()}"
        self.note = "ORCA hand — Feetech servos on a USB-TTL adapter (MABEL_HW_HANDS_<SIDE>_SERIAL pins the adapter)"
        self.cfg = C.HANDS
        self.hsub = getattr(S.hw, f"_{side}_hand", None)
        self.handle = getattr(S.hw, f"{side}_hand")
        self.sub = self.hsub
        self.writer = None
        if self.hsub is not None:
            try:
                self.writer = self.handle.claim(OWNER)
            except Exception as e:                       # noqa: BLE001
                S.ev("hw", f"{self.label}: cannot claim the writer lease ({e})", False)
        self.fb, self.model, self.goal, self.health = {}, {}, {}, {}
        self.cap, self.lock = None, threading.Lock()
        self.rate, self._n, self._t0, self._t_health = 0.0, 0, time.monotonic(), 0.0
        self.generation, self.motor_ids = "v1", {}
        self.names = []
        self._refresh_names()

    @property
    def slice(self):
        return f"hand/orca-v2-{self.side}" if self.generation == "v2" else f"hand/orca-{self.side}"

    def _orca(self):
        return getattr(self.hsub._hand, "hand", None) if self.hsub is not None else None

    def _refresh_names(self):
        orca = self._orca()
        if orca is not None:
            joints = orca.config.joints
            self.generation = getattr(orca.config, "generation", "v1")
        else:
            try:
                from mabel_hw.drivers.orcahand.config import HandConfig
                joints = HandConfig.load(model=self.side, use_model_calib=False).joints
            except Exception:                                    # noqa: BLE001
                joints = []
        self.motor_ids = {f"{self.side}_{j.name}": j.motor_id for j in joints}
        self.names = list(self.motor_ids)

    def bare(self, n):
        return n[len(self.side) + 1:] if n.startswith(self.side + "_") else n

    def entry(self, n):
        e = self.S.store.slice(f"{self.slice}/calib/{self.bare(n)}")
        if not (isinstance(e, dict) and all(k in e for k in ("raw_min", "raw_max", "model_mid"))):
            return None
        try:
            if float(e["raw_max"]) <= float(e["raw_min"]):
                return None
        except (TypeError, ValueError):
            return None
        return e

    def calibrated(self, n):
        return self.entry(n) is not None

    def to_model(self, e, raw):
        rmid = 0.5 * (float(e["raw_min"]) + float(e["raw_max"]))
        return float(e["model_mid"]) + int(e.get("dir", 1)) * _wrap_pi(raw - rmid)

    def drivable(self):
        return [n for n in self.names if self.live(n)]          # raw ticks drive before a window exists

    def connected(self):
        return bool(self.hsub and self.hsub.connected())

    def on(self):
        return bool(self.hsub and self.hsub.is_torque_enabled)

    def poll(self):
        if self.hsub is None:
            return
        if self._orca() is not None and (not self.names or self.generation != getattr(self._orca().config, "generation", "v1")):
            self._refresh_names()
        try:
            st = self.hsub._hand.get_state()                     # orcahand HandState: + raw_position ticks
        except Exception:                                        # noqa: BLE001
            st = None
        now = time.time()
        with self.lock:
            seen = set()
            if st is not None:
                for bare, tick in (st.raw_position or {}).items():
                    n = f"{self.side}_{bare}"
                    x = int(tick) * TICK
                    seen.add(n)
                    self.fb[n] = dict(raw=x, tick=int(tick), vel=(st.velocity or {}).get(bare),
                                      tau=(st.torque or {}).get(bare), temp=(st.temperature or {}).get(bare),
                                      volt=(st.voltage or {}).get(bare), live=True, t=now)
                    if self.cap and n in self.cap:
                        c = self.cap[n]
                        if c["last"] is None:
                            c.update(acc=x, min=x, max=x)
                        else:                                    # shortest arc: continuous across the 0/4095 seam
                            c["acc"] += _wrap_pi(x - c["last"])
                            c["min"], c["max"] = min(c["min"], c["acc"]), max(c["max"], c["acc"])
                        c["last"] = x
            for n in self.names:
                if n not in seen and n in self.fb:
                    self.fb[n]["live"] = False
        self._n += 1
        if time.monotonic() - self._t0 > 1.0:
            dt = time.monotonic() - self._t0
            self.rate, self._n, self._t0 = self._n / dt, 0, time.monotonic()

    def _driver_angle(self, n, target):
        """Studio model angle (or a raw rad for an uncalibrated servo) -> the angle the ORCA
        driver expects, by way of the encoder tick — so the studio's window decides the pose
        even before the driver has reloaded the new calibration."""
        e = self.entry(n)
        if e is not None:
            rmin, rmax = float(e["raw_min"]), float(e["raw_max"])
            rmid, half = 0.5 * (rmin + rmax), 0.5 * (rmax - rmin)
            delta = max(-half, min(half, int(e.get("dir", 1)) * (target - float(e["model_mid"]))))
            raw = (rmid + delta) % TAU
        else:
            raw = target % TAU
        tick = int(round(raw / TICK)) % 4096
        j = self._orca().config.by_name().get(self.bare(n)) if self._orca() is not None else None
        return j.raw_to_angle(tick) if j is not None else None

    def goals(self, goals):
        if self.S.estop or not self.on():
            return
        out = {}
        for n, v in goals.items():
            if n in self.names and self.live(n):
                a = self._driver_angle(n, float(v))
                if a is not None:
                    out[self.bare(n)] = a
                    self.goal[n] = float(v)
        if out:
            self.writer.set_joint_positions(out)

    def torque(self, on, names=None):
        if self.writer is None:
            raise RuntimeError(f"{self.label}: no driver")
        if on:
            if self.S.estop:
                raise RuntimeError("E-STOP engaged — RE-ENGAGE first")
            if not self.drivable():
                raise RuntimeError("no servo answering")
            self.cap = None
            self.writer.set_torque(True)                         # hold-then-enable: no snap to a stale goal
            self.S.ev("torque", f"{self.label}: torque ON — holding")
        else:
            self.writer.set_torque(False)
            self.S.ev("torque", f"{self.label}: torque off — free-drive")
        self.goal = {}
        return self.on()

    def capture(self, names):
        names = [n for n in names if not n.endswith("_wrist")] if any(n.endswith("_wrist") for n in names) else names
        return super().capture(names)

    def capture_end(self, save):
        with self.lock:
            cap, self.cap = self.cap, None
        if not save or not cap:
            return []
        saved, seam, short = [], [], []
        for n, c in cap.items():
            if c["min"] is None or c["max"] - c["min"] < MIN_SPAN:
                short.append(n); continue
            if c["min"] < 0.0 or c["max"] >= TAU:
                seam.append(n); continue
            self.S.save_window(self, n, c["min"], c["max"]); saved.append(n)
        msg = f"saved {len(saved)} joint range(s): {', '.join(saved)}" if saved else "nothing saved"
        if seam:
            msg += f" · REFUSED (window crosses the 0/4095 seam — Set mid at mid-travel first, then recapture): {', '.join(seam)}"
        if short and not saved and not seam:
            msg += " — no joint swept >= 8°"
        self.S.ev("calib", msg, bool(saved) and not seam)
        return saved

    def set_mid(self, n):
        """Servo offset calibration (EEPROM): the current pose becomes tick 2048. Torque must be off."""
        if self.on():
            raise RuntimeError("torque is on — release the hand first (a servo refuses the offset write while holding)")
        orca = self._orca()
        if orca is None or not self.live(n):
            raise RuntimeError(f"{n}: not answering")
        mid = self.motor_ids[n]
        with getattr(self.hsub._hand, "_busy", None) or nullcontext():
            before = orca.bus.read_positions([mid]).get(mid)    # fresh, not the poll cache
            if before is None:
                raise RuntimeError(f"{n}: no position reply")
            orca.bus.ofs_cal(mid, 2048)
        return (before - 2048) * TICK

    def sim_move(self, n, raw):
        bus = self._orca().bus
        bus._pos[self.motor_ids[n]] = int(round((float(raw) % TAU) / TICK)) % 4096

    def view(self):
        out = dict(label=self.label, kind=self.kind, side=self.side, note=self.note, slice=self.slice,
                   anchor=self.anchor, bus=f"ttl-{self.side}", port=None, connected=False, detected=False,
                   esc=None, on=False, rate_hz=0, found=[], joints={}, cfg={}, auto=None, cap=None,
                   generation=self.generation, backend="feetech")
        for n in self.names:
            out["cfg"][n] = self.joint_cfg(n)
        if self.hsub is None:
            return out
        on = self.on()
        orca = self._orca()
        with self.lock:
            for n in self.names:
                f = self.fb.get(n) or {}
                live = bool(f.get("live"))
                e = self.entry(n)
                g = self.goal.get(n)
                out["joints"][n] = dict(
                    live=live, pos=f.get("raw") if live else None, tick=f.get("tick") if live else None,
                    vel=f.get("vel"), tau=f.get("tau"), volt=f.get("volt"),
                    temp=round(f["temp"]) if live and f.get("temp") else None, t=f.get("t", 0),
                    on=on, relax=False, limits=live, err=None, state="ok" if live else "silent",
                    goal=None, sp=None, goal_model=g, model_pos=self.to_model(e, f["raw"]) if (e and live) else None,
                    esc=self.motor_ids.get(n), kind="feetech", link=None)
            cap = self.cap and {n: dict(c) for n, c in self.cap.items()}
        ids = sorted(self.motor_ids.values())
        out.update(connected=self.connected(), detected=any(j["live"] for j in out["joints"].values()),
                   on=on, rate_hz=round(self.rate, 1), cap=cap, esc=[ids[0], ids[-1]] if ids else None,
                   port="SIM" if self.S.sim else (getattr(getattr(orca, "bus", None), "port", "") or ""))
        return out

    def joint_cfg(self, n):
        cal = self.S.store.slice(f"{self.slice}/calib/{self.bare(n)}")
        return dict(esc=self.motor_ids.get(n), model=None, kp=None, kd=None, vmax=None,
                    range=self.S.model_range(self, n), calib=cal if isinstance(cal, dict) else None, kind=self.kind)


# --------------------------------------------------------------------------- lift
class LiftModule:
    """The Pico column lift — one axis in metres; the firmware owns the loop, the stall-seek
    and the counts->mm scale (Pico flash). The studio mirrors the measured travel into
    lift/pico/calib/_travel, the block the HAL reads back."""
    kind = "pico"

    def __init__(self, S):
        self.S, self.key, self.label, self.side, self.anchor = S, "lift", "Lift", None, "center"
        self.cfg, self.slice = C.LIFT, "lift/pico"
        self.note = "Pico lift column (USB serial, autodetected by USB VID 2e8a)" + \
            ("" if (S.sim or C.LIFT.get("enabled")) else " — DISABLED in config: set MABEL_HW_LIFT=true to bring it up")
        self.sub = getattr(S.hw, "_lift", None)
        self.writer = None
        if self.sub is not None:
            try:
                self.writer = S.hw.lift.claim(OWNER)
            except Exception as e:                       # noqa: BLE001
                S.ev("hw", f"{self.label}: cannot claim the writer lease ({e})", False)
        self.names, self.cap, self.auto = [], None, None
        self.torque_on = None                            # the lift reports no torque state: last commanded
        self._was_cal = None

    def connected(self):
        return bool(self.sub and self.sub.connected())

    def poll(self):
        if self.sub is None:
            return
        try:
            s = self.sub.state()
        except Exception:                                # noqa: BLE001
            return
        if self._was_cal is False and s.calibrated:      # a calibration just finished: mirror the travel
            self.record(s)
        self._was_cal = bool(s.calibrated)

    def record(self, s=None):
        s = s or self.sub.state()
        if not (s.connected and s.calibrated and s.max_pos > s.min_pos):
            raise RuntimeError(f"{self.label}: no measured travel to record (calibrate first)")
        travel = {"min_pos": round(s.min_pos, 5), "max_pos": round(s.max_pos, 5),
                  "span": round(s.max_pos - s.min_pos, 5), "homed": bool(s.homed)}
        self.S.store.save_slice(f"{self.slice}/calib/_travel", travel)
        self.S.ev("calib", f"{self.label}: travel {travel['min_pos']:.4f} … {travel['max_pos']:.4f} m -> {self.slice}/calib/_travel")
        return travel

    def view(self):
        out = dict(label=self.label, kind=self.kind, side=None, note=self.note, slice=self.slice, anchor="center",
                   bus=None, port=None, connected=False, detected=False, esc=None, on=False, rate_hz=0,
                   found=[], joints={}, cfg={}, auto=None, cap=None, lift=None)
        if self.sub is None:
            return out
        s = self.sub.state()
        live = bool(s.connected and s.height is not None)
        out.update(bus="usb-pico", port="SIM" if self.S.sim else str(self.cfg.get("port", "auto")),
                   connected=bool(s.connected), detected=live, on=bool(self.torque_on),
                   rate_hz=float(self.cfg.get("publish_rate", 50)) if live else 0,
                   lift=dict(height=s.height, target=s.height, velocity=s.velocity, effort=0.0,
                             mode="calibrated" if s.calibrated else ("homed" if s.homed else "raw"),
                             calibrated=bool(s.calibrated), torque=self.torque_on is not False,
                             torque_known=self.torque_on is not None,
                             min=s.min_pos, max=s.max_pos, live=live,
                             travel=self.S.store.slice(f"{self.slice}/calib/_travel")))
        return out

    def command(self, action, value=None):
        if self.writer is None:
            raise RuntimeError(f"{self.label}: no lift link (is the Pico attached, and MABEL_HW_LIFT enabled?)")
        if action == "stop":
            self.S.hw.lift.stop(); self.S.ev("lift", "STOP — halted"); return True
        if self.S.estop:
            raise RuntimeError("E-STOP engaged — RE-ENGAGE first")
        s = self.sub.state()
        if action == "torque":
            self.writer.set_torque(bool(value)); self.torque_on = bool(value)
            self.S.ev("lift", "torque ON — the column holds station" if value else "torque off — the column free-wheels")
        elif action == "height":
            if not s.calibrated:
                raise RuntimeError("uncalibrated column — calibrate first (jog still works)")
            self.writer.set_height(float(value)); self.S.ev("lift", f"height -> {float(value):.3f} m")
        elif action == "jog":
            v = max(-1.0, min(1.0, float(value)))
            self.writer.set_velocity(v * LIFT_JOG)
            self.S.ev("lift", f"jog {'up' if v > 0 else 'down'} at {abs(v) * LIFT_JOG * 1000:.0f} mm/s")
        elif action == "calibrate":
            self._was_cal = False
            self.writer.calibrate(); self.S.ev("lift", "calibrating — floor then ceiling stall-seek (the Pico owns the motor)")
        elif action == "home":
            self.writer.home(); self.S.ev("lift", "find floor — stall-seek the bottom datum")
        elif action == "park":
            if not (s.calibrated and s.max_pos > s.min_pos):
                raise RuntimeError("uncalibrated column — no mid-stroke to park at")
            mid = 0.5 * (s.min_pos + s.max_pos)
            self.writer.set_height(mid); self.S.ev("lift", f"HOME — mid-stroke {mid:.3f} m, clear of both hard stops")
        elif action == "record":
            return self.record()
        else:
            raise RuntimeError(f"unknown lift action {action!r}")
        return True

    def release(self, why):
        if self.sub is None:
            return
        try:
            self.S.hw.lift.stop()
        except Exception:                                # noqa: BLE001
            pass

    def estop(self):
        self.release("E-STOP")
        if self.writer is not None:
            try:
                self.writer.set_torque(False); self.torque_on = False
            except Exception as e:                       # noqa: BLE001
                self.S.ev("estop", f"{self.label}: {e}", False)


# --------------------------------------------------------------------------- base
class BaseModule:
    """The swerve base, READ-ONLY here: the studio latches its e-stop at start and never drives it.
    Steer offsets and drive senses are host calibration in the base driver, not a mabel.json slice."""
    kind = "swerve"

    def __init__(self, S):
        self.S, self.key, self.label, self.side, self.anchor = S, "base", "Base", None, "center"
        self.cfg, self.slice, self.names, self.cap, self.auto = C.BASE, None, [], None, None
        self.note = "MAXSwerve base on its own Teensy (UDP); status only — this studio keeps it e-stopped"
        self.sub = getattr(S.hw, "_base", None)
        if self.sub is not None:
            try:
                S.hw.base.set_estop(True)
            except Exception:                            # noqa: BLE001
                pass

    def poll(self):
        pass

    def view(self):
        out = dict(label=self.label, kind=self.kind, side=None, note=self.note, slice=None, anchor="center",
                   bus=None, port=None, connected=False, detected=False, esc=None, on=False, rate_hz=0,
                   found=[], joints={}, cfg={}, auto=None, cap=None, base=None)
        if self.sub is None:
            return out
        try:
            s = self.sub.state()
        except Exception:                                # noqa: BLE001
            return out
        mods = []
        for i, m in enumerate(s.modules or ()):
            d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in getattr(m, "__dict__", {}).items()}
            mods.append(dict(d, idx=i))
        out.update(bus="udp", port="SIM" if self.S.sim else f"{self.cfg.get('teensy_ip')}:{self.cfg.get('port')}",
                   connected=bool(s.connected), detected=bool(s.connected),
                   base=dict(connected=bool(s.connected), estop=bool(s.estop), modules=mods))
        return out


# --------------------------------------------------------------------------- studio
class Studio:
    def __init__(self, sim, calib_root):
        self.sim, self.estop, self.log, self.lk = sim, False, [], threading.Lock()
        self.hb = 0.0
        self.ranges = load_ranges()
        want = ["base", "left_arm", "right_arm", "neck", "torso", "hands", "lift"]
        if not sim:
            want = [n for n in want if (getattr(C, n.upper(), None) or {}).get("enabled", False)]
        self.hw = MabelHardware(mock=sim, subsystems=want, calibration_root=calib_root,
                                logger=lambda m: self.ev("hw", str(m)))
        self.hw.open()                                   # raises HardwareError while the HAL holds the lock
        self.store = self.hw.calibration
        self.hw.pace(stream_hz=50.0, service_hz=1.0)     # feeds the drivers' watchdogs / hot-join
        self.mods = {}
        for m in (JointModule(self, "arm-left", "Arm · L", "left_arm", "LEFT_ARM", note="7 DaMiao joints on the arms Teensy (openarm-dual firmware, CAN 1)"),
                  JointModule(self, "arm-right", "Arm · R", "right_arm", "RIGHT_ARM", note="7 DaMiao joints on the arms Teensy (openarm-dual firmware, CAN 2)"),
                  JointModule(self, "torso", "Torso", "torso", "TORSO", note="DaMiao torso joint on the arms Teensy's third CAN bus"),
                  JointModule(self, "neck", "Neck", "neck", "NECK", anchor="level", note="yaw = DaMiao on the arms Teensy; pitch/roll = Dynamixel servos on the Teensy's serial port"),
                  LiftModule(self), HandModule(self, "left"), HandModule(self, "right"), BaseModule(self)):
            self.mods[m.key] = m
        if sim:
            self._seed_sim()
        self.stop = threading.Event()
        for m in self.mods.values():
            if m.kind in ("can", "orca", "pico"):
                threading.Thread(target=self._poll_loop, args=(m,), daemon=True, name=f"poll-{m.key}").start()
        threading.Thread(target=self._deadman, daemon=True, name="deadman").start()

    REST = {"left_arm_2": -0.1711, "left_arm_4": 1.52, "right_arm_2": 0.1711, "right_arm_4": -1.52, "torso": 0.0}

    def _seed_sim(self):
        """--sim: start every mock joint at the REST pose of its saved window (instead of raw 0), and
        the mock column on the recorded travel (lift/pico/calib/_travel) at mid-stroke."""
        lift = getattr(self.hw, "_lift", None)
        drv = getattr(lift, "_driver", None)
        tr = self.store.slice("lift/pico/calib/_travel") or {}
        if drv is not None and hasattr(drv, "_h"):
            try:
                drv.min_pos = float(tr.get("min_pos", drv.min_pos))
                drv.max_pos = float(tr.get("max_pos", drv.max_pos))
                drv._h = 0.5 * (drv.min_pos + drv.max_pos)
            except (TypeError, ValueError):
                pass
        for m in self.mods.values():
            if m.kind not in ("can", "orca") or m.sub is None:
                continue
            for n in m.names:
                try:
                    want = self.REST.get(urdf_name(n, m.side), 0.0)
                    if m.kind == "can":
                        jc = self.store.joint(m.slice, n)
                        if jc is not None:
                            m.sim_move(n, jc.raw_of(max(jc.model_min, min(jc.model_max, want))))
                    else:
                        e = m.entry(n)
                        if e is not None:
                            rmin, rmax = float(e["raw_min"]), float(e["raw_max"])
                            d = int(e.get("dir", 1))
                            half = 0.5 * (rmax - rmin)
                            m.sim_move(n, 0.5 * (rmin + rmax) + max(-half, min(half, d * (want - float(e["model_mid"])))))
                except Exception:                        # noqa: BLE001  (a seed is cosmetic)
                    pass

    def ev(self, op, msg, ok=True):
        print(time.strftime("%H:%M:%S"), op, msg, flush=True)
        with self.lk:
            self.log.append((time.time(), op, bool(ok), msg)); del self.log[:-200]

    def _poll_loop(self, m):
        period = 0.05 if m.kind == "orca" else 0.02
        while not self.stop.wait(period):
            try:
                m.poll()
            except Exception as e:                       # noqa: BLE001
                self.ev("hw", f"{m.label}: poll failed: {e}", False)
                self.stop.wait(1.0)

    def _deadman(self):
        """No page poll for HB_MAX -> every group back to its rest state, the lift halted."""
        while not self.stop.wait(0.2):
            if self.hb and time.monotonic() - self.hb > HB_MAX:
                if any(getattr(m, "on", lambda: False)() for m in self.mods.values() if m.kind in ("can", "orca")):
                    self.release_all(f"page gone {HB_MAX:.1f} s")
                    self.mods["lift"].release("page gone")
                for m in self.mods.values():
                    if getattr(m, "cap", None):
                        m.cap = None
                        self.ev("calib", f"{m.label}: capture cancelled — page gone", False)

    def heartbeat(self):
        self.hb = time.monotonic()

    def release_all(self, why):
        for m in self.mods.values():
            if m.kind in ("can", "orca"):
                m.release(why)

    def mod_of(self, joint):
        return next((m for m in self.mods.values() if joint in getattr(m, "names", ())), None)

    def model_range(self, mod, n):
        return self.ranges.get(urdf_name(n, getattr(mod, "side", None)))

    # ---- calibration writes (the ONE store, the existing mabel.json slices) ----
    def _recal(self):
        try:
            self.hw._on_calibration_change("mabel.json")   # arms / neck / torso re-derive their maps now
        except Exception as e:                           # noqa: BLE001
            self.ev("calib", f"recalibrate failed: {e}", False)

    def _anchor(self, mod, n, e, new_window):
        """Centre anchor (the format every mabel.json slice uses): model = model_mid + dir*(raw - raw_mid).
        On a NEW window model_mid = the centre of the joint's URDF range and reliability = measured
        span / URDF span (a quality number, not applied). A joint without a URDF range keeps model 0
        at the window centre. A flip or an explicit zero leaves the anchor alone."""
        e.setdefault("dir", 1)
        if not new_window:
            return e
        rng = self.model_range(mod, n)
        if rng and rng[1] != rng[0]:
            e["model_mid"] = round((rng[0] + rng[1]) / 2, 6)
            e["reliability"] = round((e["raw_max"] - e["raw_min"]) / abs(rng[1] - rng[0]), 3)
        else:
            e.setdefault("model_mid", 0.0)
            e.pop("reliability", None)
        return e

    def patch_calib(self, mod, n, patch, drop=()):
        path = f"{mod.slice}/calib/{mod.bare(n)}"
        e = dict(self.store.slice(path) or {})
        for k in drop:                                   # e.g. a DERIVED window has no measured reliability
            e.pop(k, None)
        if e.get("raw_min") is None and not ("raw_min" in patch and "raw_max" in patch):
            raise RuntimeError(f"{n}: no saved window yet — calibrate it first")
        new_window = "raw_min" in patch or "raw_max" in patch
        e.update(patch)
        if float(e["raw_max"]) <= float(e["raw_min"]):
            raise RuntimeError(f"{n}: raw_max must exceed raw_min")
        if int(e.get("dir", 1)) not in (-1, 1):
            raise RuntimeError(f"{n}: dir must be ±1")
        e = self._anchor(mod, n, e, new_window and "model_mid" not in patch)
        self.store.save_slice(path, e)
        self._recal()
        return e

    def save_window(self, mod, n, lo, hi):
        return self.patch_calib(mod, n, {"raw_min": round(lo, 5), "raw_max": round(hi, 5)})

    def _level_patch(self, mod, n, frac=1.0):
        """The joint's CURRENT raw angle becomes model 0 and its window is DERIVED from the URDF range
        (`frac` of it, following dir), written centre-anchored. Reads the pose only — moves nothing."""
        if not mod.live(n):
            raise RuntimeError("not answering")
        rng = self.model_range(mod, n)
        if not rng or rng[0] == rng[1]:
            raise RuntimeError("no URDF range")
        cur = self.store.slice(f"{mod.slice}/calib/{mod.bare(n)}") or {}
        d = int(cur.get("dir") or 1)
        pose = float(mod.raw(n))
        lo, hi = sorted((d * frac * rng[0], d * frac * rng[1]))
        patch = {"raw_min": round(pose + lo, 5), "raw_max": round(pose + hi, 5), "dir": d,
                 "model_mid": round(frac * (rng[0] + rng[1]) / 2, 6)}
        return patch, (frac * rng[0], frac * rng[1]), pose

    def level(self, mod):
        done, skipped = [], []
        for n in mod.names:
            try:
                patch, win, pose = self._level_patch(mod, n)
            except RuntimeError as e:
                skipped.append(f"{n} ({e})"); continue
            self.patch_calib(mod, n, patch, drop=("reliability",))
            done.append(f"{n} @ {pose:+.3f} rad -> 0, window {win[0]:+.4f}..{win[1]:+.4f}")
        self.ev("calib", f"{mod.label}: LEVEL — " + (", ".join(done) or "nothing captured")
                + (" · skipped: " + ", ".join(skipped) if skipped else ""), bool(done))
        return done

    def level_joint(self, mod, n):
        patch, win, pose = self._level_patch(mod, n)
        self.patch_calib(mod, n, patch, drop=("reliability",))
        self.ev("calib", f"{n}: LEVEL — raw {pose:+.3f} rad -> 0, window {win[0]:+.4f}..{win[1]:+.4f} (dir {patch['dir']:+d})")
        return patch

    def upright(self, mod, n, frac=0.9):
        if not n.endswith("_wrist"):
            raise RuntimeError(f"{n}: Upright is for the wrist — Manual calibrate a finger")
        patch, win, pose = self._level_patch(mod, n, frac)
        self.patch_calib(mod, n, patch, drop=("reliability",))
        self.ev("calib", f"{n}: UPRIGHT — raw {pose:+.3f} rad -> 0°, window {math.degrees(win[0]):+.1f}° … "
                         f"{math.degrees(win[1]):+.1f}° = {frac:.0%} of the URDF range (dir {patch['dir']:+d})")
        return patch

    def set_mid(self, mod, n):
        x = mod.set_mid(n)
        c = self.store.slice(f"{mod.slice}/calib/{mod.bare(n)}")
        if isinstance(c, dict) and c.get("raw_min") is not None:
            self.patch_calib(mod, n, {"raw_min": round(c["raw_min"] - x, 5), "raw_max": round(c["raw_max"] - x, 5),
                                      "model_mid": c.get("model_mid", 0.0)})
        self.ev("zero", f"{n}: set mid — this pose is now tick 2048 (shift {x:+.3f} rad); saved window shifted to match")

    WRITES = {"calib", "upright", "level", "level_joint", "zero"}

    def refuse_offline_write(self, m, cmd, d):
        if m is None or not (cmd in self.WRITES or (cmd == "capture_end" and d.get("save"))):
            return
        joints = list(d.get("patches") or ([d["joint"]] if d.get("joint") else []))
        for n in joints:
            jm = self.mod_of(n)
            if jm is None or not jm.live(n):
                raise RuntimeError(f"{n}: offline — nothing saved")
        if not joints and not any(m.live(n) for n in m.names):
            raise RuntimeError(f"{m.label}: offline (no joint answering) — nothing saved")

    def do_estop(self):
        self.estop = True
        for m in self.mods.values():
            if getattr(m, "cap", None):
                m.cap = None
        self.release_all("E-STOP")
        self.mods["lift"].estop()
        try:
            self.hw.base.set_estop(True)
        except Exception:                                # noqa: BLE001
            pass
        self.ev("estop", "E-STOP — every joint group torque off (zero-torque damping / free-drive) · lift halted + torque off · base latched · captures cancelled · controls locked", False)

    def snapshot(self):
        mods = {k: m.view() for k, m in self.mods.items()}
        cfg = {"modules": {k: {"joints": list(v["joints"])} for k, v in mods.items()},
               "joints": {n: c for v in mods.values() for n, c in v["cfg"].items()}}
        return {"t": time.time(), "sim": self.sim, "estop": self.estop, "hb_max": HB_MAX,
                "calib_file": os.path.relpath(str(self.store._root / "mabel.json"), REPO) if not self.sim else "(sim copy) mabel.json",
                "cfg": cfg, "modules": mods, "log": self.log[-40:]}

    def close(self):
        self.stop.set()
        self.release_all("studio closed")
        self.mods["lift"].release("studio closed")
        self.hw.close()


class Handler(BaseHTTPRequestHandler):
    S: Studio = None
    protocol_version = "HTTP/1.1"                        # keep-alive: one connection for all polls

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _file(self, p, root):
        p = os.path.normpath(p)
        if not (p.startswith(root + os.sep) and os.path.isfile(p)):
            return self._send(404, b'{"error":"not found"}')
        with open(p, "rb") as f:
            self._send(200, f.read(), MIME.get(p.rsplit(".", 1)[-1], "application/octet-stream"))

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            if "hb=1" in self.path:
                self.S.heartbeat()                       # the page's poll is its heartbeat
            self._send(200, json.dumps(self.S.snapshot()).encode())
        elif path in ("/", "/index.html"):
            self._file(os.path.join(HERE, "studio.html"), HERE)
        elif path == "/assets/rig.glb":
            self._file(os.path.join(SHARED, "mabel_rig.glb"), SHARED)
        elif path == "/assets/joints.json":
            self._file(os.path.join(SHARED, "mabel_joints.json"), SHARED)
        elif path.startswith("/assets/"):                # three.module.js + jsm/ from the shared copy
            self._file(os.path.join(SHARED, "three", path[len("/assets/"):]), os.path.join(SHARED, "three"))
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        S, cmd = self.S, self.path.rsplit("/", 1)[-1]
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}") or b"{}")
            m = (S.mod_of(d["joint"]) if d.get("joint") else None) or S.mods.get(d.get("module"))
            if m is not None and getattr(m, "sub", None) is None and cmd not in ("estop", "reengage"):
                raise RuntimeError(f"{m.label}: not brought up ({'disabled in config' if not S.sim else 'no driver'})")
            if S.estop and cmd not in ("estop", "reengage", "capture_end") and not (cmd == "lift" and d.get("action") == "stop"):
                if cmd in ("torque", "goal", "capture", "zero", "lift", "sim_move"):
                    raise RuntimeError("E-STOP engaged — RE-ENGAGE first")
            S.refuse_offline_write(m, cmd, d)
            if cmd == "estop":
                S.do_estop(); out = True
            elif cmd == "reengage":
                S.estop = False
                S.ev("estop", "controls re-engaged — every group stays torque OFF; Enable torque in Control to hold"); out = True
            elif cmd == "goal":
                m.goals(d.get("goals") or {}); out = True
            elif cmd == "torque":
                out = m.torque(bool(d["on"]))
            elif cmd == "capture":
                out = m.capture(d["joints"])
            elif cmd == "capture_end":
                out = m.capture_end(bool(d.get("save"))) if m is not None else []
            elif cmd == "level":
                out = S.level(m)
            elif cmd == "level_joint":
                out = S.level_joint(m, d["joint"])
            elif cmd == "upright":
                out = S.upright(m, d["joint"])
            elif cmd == "zero":
                if m.kind != "orca":
                    raise RuntimeError("motor zero is not exposed by the openarm firmware — re-anchor with Set zero / Level instead")
                S.set_mid(m, d["joint"]); out = True
            elif cmd == "calib":
                patches = d.get("patches") or {d["joint"]: d["patch"]}
                for n, p in patches.items():
                    S.patch_calib(S.mod_of(n), n, p)
                S.ev("calib", "saved " + ", ".join(f"{n} {p}" for n, p in patches.items())); out = True
            elif cmd == "lift":
                out = m.command(d["action"], d.get("value"))
            elif cmd == "sim_move":                      # --sim only: stands in for moving a joint by hand
                if not S.sim:
                    raise RuntimeError("sim_move exists only with --sim")
                m.sim_move(d["joint"], float(d["raw"])); out = True
            else:
                return self._send(404, b'{"error":"unknown command"}')
            self._send(200, json.dumps({"result": out}).encode())
        except Exception as e:                           # noqa: BLE001
            self._send(400, json.dumps({"error": str(e)}).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="mabel_hw mock plants — no hardware is opened")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http", type=int, default=int(os.environ.get("MABEL_SETUP_PORT", 8300)))
    ap.add_argument("--calib-root", default=None,
                    help="directory holding mabel.json (default: hardware_bridge/calibration_data; "
                         "with --sim a fresh scratch copy of it, so the robot's file is never touched)")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    root = a.calib_root
    if a.sim and not root:
        root = os.path.join(tempfile.gettempdir(), "mabel_setup_studio_sim_calib")
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(os.path.join(BRIDGE, "calibration_data"), root)
    root = root or (C.CALIBRATION.get("root") or None)
    try:
        srv = ThreadingHTTPServer((a.host, a.http), Handler)
    except OSError as e:
        print(f"cannot listen on {a.http}: {e} — another studio running? use --http"); return 2
    srv.daemon_threads = True
    try:
        Handler.S = Studio(a.sim, root)
    except HardwareError as e:
        srv.server_close()
        print(f"REFUSED: {e}\nThe hardware bridge (HAL) owns the devices — stop it first "
              "(stop real_run.sh / the hardware-bridge service), then start the setup studio.", flush=True)
        return 3
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.http}"
    print(f"MABEL Setup Studio on {url}  ({'SIM — mock plants' if a.sim else 'real hardware'}) · "
          f"calibration -> {Handler.S.store._root / 'mabel.json'}", flush=True)
    if not a.no_open:
        threading.Timer(0.3, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.S.close()                                # torque off, leases released, devices closed
    return 0


if __name__ == "__main__":
    sys.exit(main())
