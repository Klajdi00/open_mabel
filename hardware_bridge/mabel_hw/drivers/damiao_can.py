"""damiao.py — DaMiao (OpenArm) CAN/MIT-mode motor driver, shared by arms + body.

Extracted from the proven ``mabel_arms_hardware/arms_node.py`` (single source of the
MIT framing). The MABEL arms AND the body (DaMiao torso + 3-DOF DaMiao neck, e.g.
DMJ4310) are the SAME protocol, so both subsystems share this module. The bridge owns
it; nothing else opens the CAN bus.

Per-joint wiring (which CAN bus + motor id, limits, MIT kp/kd) is robot-specific, so it
is passed as a ``joint_motors`` map string rather than hard-coded:

    "left_arm_1:can0:1, ... right_arm_7:can1:7"     (arms)
    "torso:can0:1, neck_yaw:can0:2, neck_pitch:can0:3, neck_roll:can0:4"   (body)
"""

import threading

from ..types import monotonic
from .seam import GroupFeedback, JointGroupDriver

try:
    import can  # python-can
    CAN_OK, CAN_ERR = True, None
except Exception as _e:                              # pragma: no cover
    can = None
    CAN_OK, CAN_ERR = False, _e


def f2u(x, lo, hi, bits):
    """Float -> unsigned int over [lo, hi] (DaMiao MIT packing)."""
    x = max(lo, min(hi, x))
    return int((x - lo) * ((1 << bits) - 1) / (hi - lo))


def u2f(v, lo, hi, bits):
    return v * (hi - lo) / ((1 << bits) - 1) + lo


def urdfize(name):
    return name.lower().replace(" ", "_")


class MotorSpec:
    __slots__ = ("channel", "mid", "kp", "kd", "p_max", "v_max", "t_max",
                 "pos", "vel", "tau", "temp", "target", "last_rx")

    def __init__(self, channel, mid, kp, kd, p_max, v_max, t_max):
        self.channel, self.mid = channel, mid
        self.kp, self.kd = kp, kd
        self.p_max, self.v_max, self.t_max = p_max, v_max, t_max
        self.pos = self.vel = self.tau = 0.0
        self.temp = 0                                # MOSFET temp, deg C (feedback d[6])
        self.target = None                           # rad; None until first command
        self.last_rx = 0.0                           # monotonic time of last feedback
                                                     # frame (0.0 = never) — per-JOINT
                                                     # liveness, so ONE unplugged motor
                                                     # is visible, not just a dead bus

    def live(self, timeout=1.0) -> bool:
        return self.last_rx > 0.0 and (monotonic() - self.last_rx) < timeout


class DamiaoBus:
    """One socketcan channel: enable/disable + MIT position command + feedback parse.

    HOT-REPLUG: a bus whose socket dies (USB-CAN adapter yanked, interface down,
    bus-off) marks itself ``dead`` — the RX thread exits cleanly instead of
    dying silently, sends start failing fast, and ``DamiaoGroup.service()``
    rebuilds the bus (fresh socket + re-add + re-enable) once it opens again.
    """

    def __init__(self, channel, logger=None, bustype="socketcan"):
        self.channel = channel
        self.log = logger or (lambda *a, **k: None)
        self.motors = {}                             # mid -> MotorSpec
        self.lock = threading.Lock()
        self.dead = False                            # set on any socket-level failure
        # bustype="virtual" lets tests exchange real frames with no hardware
        self.bus = can.interface.Bus(channel=channel, bustype=bustype)
        self._stop = threading.Event()
        threading.Thread(target=self._rx, daemon=True).start()

    def add(self, spec: MotorSpec):
        self.motors[spec.mid] = spec

    def _mark_dead(self, why):
        if not self.dead:
            self.dead = True
            self.log(f"CAN {self.channel} DEAD ({why}) — will rebuild when it returns")

    def _rx(self):
        while not self._stop.is_set():
            try:
                msg = self.bus.recv(timeout=0.5)
            except Exception as e:                   # unplugged adapter / iface down
                self._mark_dead(f"recv: {e}")
                return                               # exit; the rebuilt bus gets a new thread
            if msg is None or len(msg.data) < 8:
                continue
            mid = msg.data[0] & 0x0F
            with self.lock:
                m = self.motors.get(mid)
                if m is None:
                    continue
                d = msg.data
                p = (d[1] << 8) | d[2]
                v = (d[3] << 4) | (d[4] >> 4)
                t = ((d[4] & 0xF) << 8) | d[5]
                m.pos = u2f(p, -m.p_max, m.p_max, 16)
                m.vel = u2f(v, -m.v_max, m.v_max, 12)
                m.tau = u2f(t, -m.t_max, m.t_max, 12)
                m.temp = int(d[6])                   # T_MOS deg C (d[7] = T_rotor)
                m.last_rx = monotonic()

    def _cmd(self, mid, byte):
        try:
            self.bus.send(can.Message(arbitration_id=mid,
                                      data=[0xFF] * 7 + [byte],
                                      is_extended_id=False))
        except Exception as e:
            self._mark_dead(f"send: {e}")
            raise

    def enable(self, mid):       self._cmd(mid, 0xFC)
    def disable(self, mid):      self._cmd(mid, 0xFD)
    def clear_faults(self, mid): self._cmd(mid, 0xFB)

    def send_mit(self, m: MotorSpec):
        if m.target is None:
            return
        pi = f2u(m.target, -m.p_max, m.p_max, 16)
        vi = f2u(0.0, -m.v_max, m.v_max, 12)
        ki = f2u(m.kp, 0.0, 500.0, 12)
        di = f2u(m.kd, 0.0, 5.0, 12)
        ti = f2u(0.0, -m.t_max, m.t_max, 12)
        data = [(pi >> 8) & 0xFF, pi & 0xFF,
                (vi >> 4) & 0xFF, ((vi & 0xF) << 4) | ((ki >> 8) & 0xF), ki & 0xFF,
                (di >> 4) & 0xFF, ((di & 0xF) << 4) | ((ti >> 8) & 0xF), ti & 0xFF]
        try:
            self.bus.send(can.Message(arbitration_id=m.mid, data=data,
                                      is_extended_id=False))
        except Exception as e:
            self._mark_dead(f"send: {e}")
            raise

    def shutdown(self):
        self._stop.set()
        for mid in list(self.motors):
            try:
                self.disable(mid)
            except Exception:
                pass
        try:
            self.bus.shutdown()
        except Exception:
            pass


def parse_joint_motors(spec_str, kp, kd, p_max, v_max, t_max):
    """'left_arm_1:can0:1, torso:can0:1' -> {joint_name: MotorSpec}."""
    out = {}
    for item in filter(None, (s.strip() for s in spec_str.split(","))):
        parts = item.split(":")
        if len(parts) < 3:
            continue
        joint, channel, mid = parts[0], parts[1], int(parts[2])
        out[urdfize(joint)] = MotorSpec(channel, mid, kp, kd, p_max, v_max, t_max)
    return out


class CanLink(JointGroupDriver):
    """JointGroupDriver adapter over :class:`DamiaoGroup` — the socketcan MIT
    fallback/bench backend (today's proven arms/body path). Same seam as
    ``teensy_link.TeensyJointGroup``; select per-subsystem with
    ``MABEL_HW_<NAME>_BACKEND=can``.

    Wire behavior is IDENTICAL to the pre-refactor subsystems: ``tick()`` calls
    ``DamiaoGroup.stream()`` (re-send MIT + refresh the feedback stash) from the
    owner's timer; ``read()`` returns the stash without touching the bus.
    """

    def __init__(self, joint_motors: str, *, kp=30.0, kd=1.0, p_max=12.5,
                 v_max=30.0, t_max=10.0, bustype="socketcan", logger=None):
        specs = parse_joint_motors(joint_motors, kp=kp, kd=kd, p_max=p_max,
                                   v_max=v_max, t_max=t_max)
        if not specs:
            raise ValueError(f"empty/unparsable joint_motors map: {joint_motors!r}")
        self.names = tuple(specs)
        self._group = DamiaoGroup(specs, logger, bustype=bustype)
        self._fb = None                              # (names,pos,vel,eff) stash
        self._stamp = 0.0

    # lifecycle
    def connect(self) -> None:
        self._group.connect()

    def close(self) -> None:
        self._group.shutdown()

    def connected(self) -> bool:
        buses = self._group.buses
        return bool(buses) and any(not b.dead for b in buses.values())

    # write
    def set_targets(self, targets) -> None:
        for name in targets:
            if urdfize(name) not in self._group.joints:
                raise KeyError(f"unknown joint {name!r}; "
                               f"this group has {', '.join(self.names)}")
        self._group.set_targets(targets)

    def set_torque(self, on: bool) -> None:
        self._group.set_torque(on)

    @property
    def is_torque_enabled(self) -> bool:
        return self._group.is_torque_enabled

    # read
    def read(self):
        if self._fb is None:                         # never ticked: peek, no send
            with self._group._lock:
                specs = self._group.joints
                fb = (list(specs), [s.pos for s in specs.values()],
                      [s.vel for s in specs.values()],
                      [s.tau for s in specs.values()])
        else:
            fb = self._fb
        names, pos, vel, eff = fb
        health = self._group.health()
        specs = self._group.joints
        return GroupFeedback(
            names=tuple(names), position=tuple(pos), velocity=tuple(vel),
            effort=tuple(eff),
            temperature=tuple(float(health[n]["tempC"]) for n in names),
            # per-JOINT liveness (feedback frame within 1 s): one unplugged
            # motor shows up individually, not just a whole dead bus
            live=tuple(specs[n].live() if n in specs else False for n in names),
            stamp=self._stamp or monotonic())

    def health(self):
        return self._group.health()

    # pacing
    def tick(self) -> None:
        self._fb = self._group.stream()
        self._stamp = monotonic()

    def service(self) -> None:
        self._group.service()                         # rebuild any dead CAN bus

    def on_state(self, callback):
        return lambda: None                           # poll-only transport


class DamiaoGroup:
    """A set of DaMiao joints across one or more CAN buses, with MIT position control.

    Shared driver behind the arms and body subsystems: build from a joint_motors map,
    ``connect()`` (enable motors), ``set_targets({joint: rad})``, ``stream()`` (re-send
    MIT + return feedback as parallel name/pos/vel/eff lists), ``shutdown()``.
    """

    def __init__(self, joints: dict, logger=None, bustype="socketcan"):
        self.joints = joints                         # {name: MotorSpec}
        self.log = logger or (lambda *a, **k: None)
        self.bustype = bustype
        self.buses = {}                              # channel -> DamiaoBus
        self._lock = threading.Lock()
        self._torque = True                          # connect() enables motors
        self._rebuild_last = {}                      # channel -> monotonic of last try

    def connect(self):
        for joint, spec in self.joints.items():
            try:
                bus = self.buses.get(spec.channel)
                if bus is None:
                    bus = self.buses[spec.channel] = DamiaoBus(
                        spec.channel, self.log, bustype=self.bustype)
                bus.add(spec)
                bus.enable(spec.mid)
            except Exception as e:
                self.log(f"{joint} ({spec.channel} id {spec.mid}) init failed: {e}")
        self.log(f"CAN buses up: {sorted(self.buses)}")

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque

    def set_torque(self, on: bool):
        """Energise (MIT hold) or free-drive every motor. While off, ``stream`` stops
        re-sending MIT targets so the joints stay limp. Enabling HOLDS-IN-PLACE: each
        joint's target is re-latched to its measured position first, so energising
        never snaps to a stale target (the hands' no-lurch rule)."""
        on = bool(on)
        with self._lock:
            if on and not self._torque:
                for spec in self.joints.values():
                    if spec.target is not None:
                        spec.target = spec.pos
            self._torque = on
            for joint, spec in self.joints.items():
                bus = self.buses.get(spec.channel)
                if bus is None:
                    continue
                try:
                    bus.enable(spec.mid) if on else bus.disable(spec.mid)
                except Exception as e:
                    self.log(f"{joint} torque {'on' if on else 'off'} failed: {e}")

    def set_targets(self, targets: dict):
        with self._lock:
            for name, pos in targets.items():
                spec = self.joints.get(urdfize(name))
                if spec is not None:
                    spec.target = max(-spec.p_max, min(spec.p_max, float(pos)))

    def health(self):
        """{joint: {"torqueNm", "tempC"}} from the latest MIT feedback — the app
        telemetry source (robot_state.motorHealth); pos/vel/effort ride joint_states."""
        with self._lock:
            return {joint: {"torqueNm": round(spec.tau, 3), "tempC": int(spec.temp)}
                    for joint, spec in self.joints.items()}

    def service(self):
        """Hot-replug watchdog (call ~1 Hz, OFF the fast path): rebuild any DEAD
        bus — fresh socket, re-add the specs, re-enable if torque was on — so a
        yanked USB-CAN adapter (or a downed/bus-off interface) rejoins the
        moment it comes back, with no bridge restart. Rate-limited to one
        attempt per channel per 2 s; failures stay quiet until success."""
        for channel in list(self.buses):
            bus = self.buses.get(channel)
            if bus is None or not bus.dead:
                continue
            now = monotonic()
            if now - self._rebuild_last.get(channel, 0.0) < 2.0:
                continue
            self._rebuild_last[channel] = now
            try:
                new_bus = DamiaoBus(channel, self.log, bustype=self.bustype)
            except Exception:
                continue                             # still gone; retry next service
            try:
                bus.shutdown()                       # reap the dead one's thread/socket
            except Exception:
                pass
            with self._lock:
                self.buses[channel] = new_bus
                specs = [s for s in self.joints.values() if s.channel == channel]
            ok = 0
            for spec in specs:
                new_bus.add(spec)
                if self._torque:
                    try:
                        new_bus.enable(spec.mid)
                        ok += 1
                    except Exception as e:
                        self.log(f"re-enable id {spec.mid} on {channel}: {e}")
            self.log(f"CAN {channel} REBUILT ({len(specs)} joints re-added"
                     + (f", {ok} re-enabled" if self._torque else ", torque off")
                     + ") — bus rejoined the system")

    def stream(self):
        """Re-send MIT targets (keeps motors live); return (names, pos, vel, eff)."""
        names, pos, vel, eff = [], [], [], []
        with self._lock:
            for joint, spec in self.joints.items():
                bus = self.buses.get(spec.channel)
                if bus is not None and self._torque:      # limp while torque is off
                    try:
                        bus.send_mit(spec)
                    except Exception as e:
                        self.log(f"send_mit {joint}: {e}")
                names.append(joint)
                pos.append(spec.pos)
                vel.append(spec.vel)
                eff.append(spec.tau)
        return names, pos, vel, eff

    def shutdown(self):
        for bus in self.buses.values():
            bus.shutdown()
