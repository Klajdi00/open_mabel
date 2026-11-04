"""openarm.py — the arms as a JointGroupDriver over the openarm-dual firmware.

The :class:`~mabel_hw.drivers.seam.JointGroupDriver` view on
:class:`~mabel_hw.drivers.openarm_link.OpenArmLink`. One group per arm; both
groups share ONE link (one socket, one manifest, one 100 Hz stream), each writing
its own slice of the merged command vector.

RAW MOTOR RADIANS ONLY. This layer does no calibration, no gravity compensation,
no impedance policy, no name translation:

  * calibration (raw<->model, the end-stop window, the multi-turn fold) is applied
    exactly once, one layer up, in ``mabel_hw.subsystems.joint_group``;
  * the control law (gravity feed-forward, impedance/compliance modes) is the
    COMMANDER's job — ``controller/mabel/arm_control`` — and reaches us as plain
    ``kp``/``kd``/``effort`` on :meth:`OpenArmGroup.command`;
  * the firmware names (``l_j1``..``r_j7``) are this driver's canonical names; the
    URDF rename (``left_arm_1``) happens at the DDS contract boundary, in
    ``mabel_hw.ros.arms_node``.

Torque: ``set_torque(True)`` latches a hold at the MEASURED raw angle and starts
streaming; ``set_torque(False)`` reverts the slice to a damping hold (kp=0). The
hold is latched from the measurement with NO clamp in the path — clamping "hold
where you are" is what turns an engage into a snap.

HOT-JOIN: a group may exist before the Teensy answers (unresolved: reads report
dead, writes latch locally). It binds the moment the manifest arrives — at bridge
start or after a mid-run replug — and every (re)bind resets the group to limp, so
a rejoin never snaps to a stale target.
"""
from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional, Tuple

from ..types import monotonic
from .openarm_link import OpenArmLink
from .seam import GroupFeedback, JointGroupDriver

__all__ = ["OpenArmGroup"]

# DaMiao MIT status nibble (DM manual / firmware the root README): 1 = enabled OK,
# >=8 = fault. Named so a red joint in the apps states its cause.
_DM_FAULTS = {8: "overpressure", 9: "undervoltage", 10: "overcurrent",
              11: "mos_overtemp", 12: "coil_overtemp", 13: "comm_loss",
              14: "overload"}

# First-principles per-motor liveness floor (see OpenArmGroup.read()): a powered
# DaMiao on the CAN bus always reports a real MOSFET temperature; at/below this it is
# treated as absent (no health telemetry for that motor).
_MOTOR_TEMP_MIN_C = 1.0


class OpenArmGroup(JointGroupDriver):
    """JointGroupDriver view over a subset of one OpenArmLink's actuators."""

    def __init__(self, link: OpenArmLink, names: Tuple[str, ...], *,
                 kp=30.0, kd=1.0, wire_t_max=None):
        """kp/kd are a scalar OR a {joint_name: value} map.

        PER-JOINT MATTERS: one gain for all 7 does not exist physically. J1/J2
        swing the whole arm; J5-J7 are the wrist, and with no end effector their
        inertia is orders of magnitude smaller. Since w ~ sqrt(kp/J) and the
        damping ratio ~ kd/(2*sqrt(kp*J)), the SAME kp makes the wrist both much
        faster AND much less damped — it rings against the 50 Hz command rate and
        the link latency. That is the "J5/J6/J7 jitter with no hand fitted"
        report. A map lets the wrist be soft while the shoulder stays stiff.
        """
        self.names = tuple(names)
        self._link = link
        self._cfg_kp = self._per_joint(kp, 30.0)
        self._cfg_kd = self._per_joint(kd, 1.0)
        # {joint: expected register T_MAX}. The firmware ENCODES t_ff with the
        # SD-card manifest's limits while the motor DECODES with its own flash
        # registers — a manifest limit below the register silently MULTIPLIES
        # every commanded torque (seen 2026-07-28: torso manifest 100 vs register
        # 500 = 5x). When set, torque enable is REFUSED until the advertised
        # limit matches (same guard as the studio's TeensyJoint).
        self._wire_t_max = dict(wire_t_max or {})
        self._idx: Tuple[int, ...] = ()             # manifest index per name
        self._resolved = False
        self._warned_missing = False
        self._torque = False
        self._lock = threading.Lock()
        # latched MIT frame, raw motor units, parallel to self.names
        n = len(self.names)
        self._pos = [0.0] * n
        self._vel = [0.0] * n
        self._kp = [0.0] * n
        self._kd = [self._cfg_kd[nm] for nm in self.names]
        self._tff = [0.0] * n

    # ── resolution (manifest -> indices) ─────────────────────────────────────
    @property
    def resolved(self) -> bool:
        return self._resolved

    def _resolve(self) -> bool:
        """Bind names to the firmware's manifest layout. Safe to call any time;
        returns True when bound. A mismatch logs ONCE and keeps retrying."""
        if self._resolved:
            return True
        specs = self._link.specs()
        if not specs:
            return False
        order = {s.name: i for i, s in enumerate(specs)}
        missing = [n for n in self.names if n not in order]
        if missing:
            if not self._warned_missing:
                self._warned_missing = True
                self._link._log(
                    f"[openarm {self._link.teensy_ip}] group {self.names} "
                    f"UNRESOLVED: firmware advertises "
                    f"{[s.name for s in specs]}; missing {missing}")
            return False
        with self._lock:
            self._idx = tuple(order[n] for n in self.names)
            self._resolved = True
        return True

    def _on_link_reset(self) -> None:
        """The link (re)connected or the manifest changed: rebind and go LIMP
        until re-commanded, so a rebooted board never adopts a stale target."""
        with self._lock:
            self._resolved = False
            self._torque = False
            n = len(self.names)
            self._kp = [0.0] * n
            self._tff = [0.0] * n
            self._vel = [0.0] * n
        self._resolve()

    def _specs(self):
        specs = self._link.specs()
        return [specs[i] for i in self._idx] if (self._resolved and specs) else []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        """Open the shared link. Does NOT energize the motors — nothing is
        streamed until set_torque(True)."""
        self._link.connect()
        self._resolve()

    def close(self) -> None:
        """Release this group: its slice reverts to the damping hold. The LINK
        stays up for the sibling arm; closing the link itself is the owner's
        (MabelHardware/bridge) shutdown concern."""
        self.set_torque(False)

    def connected(self) -> bool:
        return self._resolved and self._link.connected()

    # ── write side (raw motor rad) ───────────────────────────────────────────
    def set_targets(self, targets: Mapping[str, float]) -> None:
        """Latch position targets (raw rad). Unknown names raise KeyError;
        joints absent from the mapping keep their previous target."""
        with self._lock:
            for name, value in targets.items():
                try:
                    j = self.names.index(name)
                except ValueError:
                    raise KeyError(f"unknown joint {name!r}; this group has "
                                   f"{', '.join(self.names)}") from None
                self._pos[j] = float(value)
            self._push_locked()

    def command(self, position: Mapping[str, float],
                velocity: Optional[Mapping[str, float]] = None,
                kp: Optional[Mapping[str, float]] = None,
                kd: Optional[Mapping[str, float]] = None,
                effort: Optional[Mapping[str, float]] = None) -> None:
        """Full MIT command — the firmware applies
        ``T = kp*(pos - q) + kd*(vel - dq) + t_ff``. Every field is raw motor
        frame; omitted fields keep their latched value (gains fall back to the
        configured defaults). ``effort`` is clamped to the motor's rated t_max
        at the link's send()."""
        velocity, kp, kd, effort = (velocity or {}), (kp or {}), (kd or {}), (effort or {})
        with self._lock:
            for name, value in position.items():
                try:
                    j = self.names.index(name)
                except ValueError:
                    raise KeyError(name) from None
                self._pos[j] = float(value)
                self._vel[j] = float(velocity.get(name, 0.0))
                self._kp[j] = float(kp.get(name, self._cfg_kp[name]))
                self._kd[j] = float(kd.get(name, self._cfg_kd[name]))
                self._tff[j] = float(effort.get(name, 0.0))
            self._push_locked()
        # PUSH NOW (2026-07-29): a latched command used to wait for the next
        # 100 Hz openarm-tx tick (+0-10 ms, ~5 ms mean, on EVERY teleop frame).
        # Ship the merged vector immediately — send() is fully lock-guarded and
        # already the one t_max choke point; the tx thread keeps re-sending for
        # the firmware watchdog exactly as before. (Same "push now — low
        # latency" contract the base has had all along.)
        if self._torque:
            self._link.send()

    def _push_locked(self) -> None:
        """Write our slice into the link's merged vector (caller holds _lock).
        A torque-off group never pushes — its slice stays a damping hold."""
        if not (self._resolved and self._torque):
            return
        self._link.write_slice(self._idx, self._pos, self._vel, self._kp,
                               self._kd, self._tff)

    def _per_joint(self, value, default: float) -> dict:
        """A scalar or a {name: value} map -> a full per-joint dict.

        A map need only name the joints that differ; the rest fall back to the
        map's "default" key if present, else `default`.
        """
        if isinstance(value, Mapping):
            base = float(value.get("default", default))
            return {n: float(value.get(n, base)) for n in self.names}
        return {n: float(value) for n in self.names}

    def set_torque(self, on: bool) -> None:
        """Engage (hold at the MEASURED pose, no lurch) or damping-hold."""
        on = bool(on)
        if on and not self._resolved:
            self._resolve()
        if on == self._torque:
            return
        if on:
            st = self._link.state()
            if not self._resolved or st is None:
                # REFUSE, don't pretend. Engaging means "hold where the joint IS",
                # which needs a measured pose and a bound index map. Without them
                # we would report torque-on while streaming nothing (or, worse,
                # whatever `_pos` happens to hold). Stay limp and stay honest —
                # the adapter republishes torque_state, so the operator sees it.
                self._link._log(
                    f"[openarm] {self.names[0][:1]}-arm torque enable REFUSED: "
                    + ("no telemetry yet" if st is None else "group unresolved")
                    + " — cannot latch a hold at the measured pose. Retry once the"
                    " board is streaming.")
                return
            if self._wire_t_max:
                specs = self._link.specs()
                for j, i in enumerate(self._idx):
                    want = self._wire_t_max.get(self.names[j])
                    if want is None or i >= len(specs):
                        continue
                    have = float(specs[i].t_max or 0.0)
                    if abs(have - float(want)) > 1e-6:
                        ratio = (float(want) / have) if have > 0 else float("inf")
                        self._link._log(
                            f"[openarm] torque enable REFUSED for {self.names[j]}: "
                            f"manifest t_max {have:g} != motor register {want:g} — "
                            f"a mismatched limit RESCALES every commanded torque "
                            f"({ratio:.1f}x here). Fix the SD-card /robot.json "
                            f"and power-cycle the Teensy first.")
                        return
            with self._lock:
                # Latch the hold at the measured RAW angle, straight from the
                # wire. No calibration, no clamp: "hold" must mean exactly where
                # the joint IS, or the engage itself throws the arm.
                for j, i in enumerate(self._idx):
                    if i < len(st.joints):
                        self._pos[j] = float(st.joints[i].pos)
                        self._vel[j] = 0.0
                        self._kp[j] = self._cfg_kp[self.names[j]]
                        self._kd[j] = self._cfg_kd[self.names[j]]
                        self._tff[j] = 0.0
                self._torque = True
                self._push_locked()
        else:
            with self._lock:
                self._torque = False
            # damp_slice takes ONE kd for the slice (the hold is kp=0). Use the
            # SMALLEST configured kd: damping is a brake, and the gentlest joint
            # sets the bound — a wrist damped at the shoulder's kd would snatch.
            self._link.damp_slice(self._idx, kd=min(self._cfg_kd.values()))

    @property
    def is_torque_enabled(self) -> bool:
        return self._torque

    # ── read side (raw motor rad) ────────────────────────────────────────────
    def read(self) -> GroupFeedback:
        st = self._link.state()
        temps = self._link.temps()
        hfull = self._link.health_full()
        fresh = st is not None and st.age() < 1.0 and self._resolved
        pos, vel, eff, temp, live = [], [], [], [], []
        for j, name in enumerate(self.names):
            i = self._idx[j] if j < len(self._idx) else -1
            in_range = fresh and 0 <= i < len(st.joints)
            f = st.joints[i] if in_range else None
            v = float(f.vel) if f else 0.0
            e = float(f.tau) if f else 0.0
            pos.append(float(f.pos) if f else 0.0)
            vel.append(v)
            eff.append(e)
            tt = temps[i] if (in_range and i < len(temps) and temps[i]) else None
            t_mos = tt[0] if tt else None
            # A DYNAMIXEL has ONE temperature sensor; the firmware publishes it as
            # t_rotor and leaves t_mos = 0 ("this number does not exist"). Report
            # whichever the actuator actually measures.
            t_rotor = tt[1] if (tt and len(tt) > 1) else None
            t = float(t_mos) if t_mos else (float(t_rotor) if t_rotor else 0.0)
            temp.append(t)
            # The firmware's own per-actuator verdict (ActuatorHealth.live: feedback
            # fresh within its live window), when a Health reply has arrived.
            h_live = (hfull[i].get("live") if (in_range and i < len(hfull)) else None)
            # PER-MOTOR liveness — NOT link freshness. Both arms ride ONE openarm-dual
            # UDP frame, so `fresh` is True for BOTH whenever EITHER streams and cannot
            # tell a powered motor from an unplugged one (the reason a dead right arm
            # read as "connected"). First principles: a DaMiao that is powered and on the
            # CAN bus always reports LIVE measurement — a real MOSFET temperature and
            # encoder/torque that is never exact-zero — while a disconnected/unpowered
            # motor reports a FROZEN position with velocity==0 AND torque==0 and no
            # temperature (verified on the real robot: dead right arm streamed pos!=0 but
            # vel==eff==0.0). So a motor is alive iff it shows ANY live signal. Position
            # is EXCLUDED on purpose — it stays frozen-nonzero when the motor is dead.
            # This also gives free RECONNECTION: the moment the motor is powered again,
            # its vel/torque/temp come back nonzero and it re-appears (read() is
            # re-evaluated every tick).
            #
            # DYNAMIXEL joints (neck pitch/roll on the Teensy's serial bus, 2026-09-30)
            # break the "any live signal" rule at rest: the transport publishes vel = 0,
            # tau = 0 and t_mos = 0 by design, so a perfectly healthy, motionless
            # servo showed NO signal and read dead. Its temperature arrives as
            # t_rotor (folded into `t` above), and the firmware's ActuatorHealth.live
            # flag is the transport-agnostic answer — honour it when it is present.
            live.append(bool(in_range and (t > _MOTOR_TEMP_MIN_C
                                           or v != 0.0 or e != 0.0
                                           or h_live is True)))
        return GroupFeedback(
            names=self.names, position=tuple(pos), velocity=tuple(vel),
            effort=tuple(eff), temperature=tuple(temp), live=tuple(live),
            stamp=st.recv_time if st is not None else monotonic())

    def health(self) -> Dict[str, Dict[str, float]]:
        """Per-joint health for /arms/motor_health. Since 2026-07-29 this also
        carries the firmware's full ActuatorHealth — the DM fault nibble (with
        a human-readable name), enabled/live flags and feedback staleness — so
        a red joint in the apps finally states WHY (undervoltage vs overload vs
        comm-loss was previously indistinguishable)."""
        fb = self.read()
        full = self._link.health_full()
        out: Dict[str, Dict[str, float]] = {}
        for j, (n, e, t, ok) in enumerate(zip(fb.names, fb.effort,
                                              fb.temperature, fb.live)):
            d: Dict[str, object] = {"torqueNm": round(float(e), 3),
                                    "tempC": round(float(t), 1)}
            i = self._idx[j] if (self._resolved and j < len(self._idx)) else None
            h = full[i] if (i is not None and i < len(full)) else None
            if h is not None:
                if h.get("err") is not None:
                    err = int(h["err"])
                    d["err"] = err
                    if err >= 8:
                        d["fault"] = _DM_FAULTS.get(err, f"fault_0x{err:X}")
                if h.get("t_rotor") is not None:
                    d["tempRotorC"] = h["t_rotor"]
                for k in ("enabled", "live", "stale_ms"):
                    if h.get(k) is not None:
                        d[k] = h[k]
            # ALWAYS say whether the motor is alive. The firmware omits `live` (and
            # `enabled`) for an actuator it has never heard from — only `stale_ms` =
            # its uptime — and a consumer that keys on the flag's presence then falls
            # back to "the joint is in the stream", which a dead motor's frozen
            # position satisfies (the Mac cockpit read a dead shoulder as 7/7,
            # 2026-10-01). read()'s per-motor verdict is the one this driver stands by.
            d.setdefault("live", bool(ok))
            if ok or h is not None:
                out[n] = d
        return out

    # ── pacing ───────────────────────────────────────────────────────────────
    def tick(self) -> None:
        """No-op: OpenArmLink streams the merged vector on its own thread at
        stream_hz (the firmware watchdog needs a faster, steadier cadence than
        a ROS timer gives)."""

    def service(self) -> None:
        """1 Hz: keep the socket up and (re)bind after a hot-replug."""
        if not self._link.connected():
            self._link.connect()
        self._resolve()
