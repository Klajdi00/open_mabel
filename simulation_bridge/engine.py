"""
engine.py — SimWorld: the ONE owner of MuJoCo physics for a simulated MABEL.

This is the simulation twin of hardware_bridge's device layer: where the real
HAL's subsystems write CAN/serial/UDP to motors, SimWorld writes actuator
targets into MuJoCo and steps the world in real time. Nothing else in the
stack may call mj_step for this robot — the teleop server, ROS, cameras, and
viewers are all CLIENTS of the state this engine publishes.

State OUT (per physics substep) — the extended qpos memmap, layout
    [ time, qpos(nq), qvel(nv), ctrl(nu) ]
The legacy readers (sim_camera_server, view_robot, the control-center mirror)
map only the first 1+nq floats, so they work untouched; new consumers (the
gateway's SimBridgePlant) get true velocities + the executed actuator targets.
A `.scene` sidecar records the scene path; the `.pert` memmap (nbody×6) stays
the desktop viewer's Ctrl+drag INBOUND wrench channel, now applied here.

Commands IN (thread-safe, from the RPC server / DDS layer):
  set_twist(vx, vy, wz)      body-frame base twist → swerve IK (the SAME
                             controller/mabel/swerve.py the in-process sim
                             used) — stale twists zero the base (deadman,
                             mirroring hardware_bridge's SwerveBase)
  set_targets({name: rad})   position-servo targets for any non-base actuator
                             (arms / body / hands / lift) — latched, like a
                             real motor holding its last setpoint
  set_effort({joint: τ})     joint-space torques (the effort channel: WBC
                             compliance/impedance) → qfrc_applied; stale → 0
  set_wrench({body: [6]})    external Cartesian pushes → xfrc_applied;
                             stale → 0
  set_estop(bool)            zero base drive instantly (targets hold)
  set_group_torque(group, on) torque_enable analog: while OFF the group's
                             servo targets track the live joint positions, so
                             the limb yields instead of holding
  reset(pose=None)           re-home the world; optional {x,y,yaw} teleport
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time

import numpy as np

from . import config as CFG

log = logging.getLogger("simbridge.engine")

# the controller package (swerve IK + joint grouping) — same path juggling the
# teleop server does (sim_teleop_bridge.py:77)
_CONTROLLER = os.path.join(CFG.REPO, "controller")
if _CONTROLLER not in sys.path:
    sys.path.insert(0, _CONTROLLER)
_SERVER = os.path.join(CFG.REPO, "server")
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)

from mabel import swerve as _swerve                     # noqa: E402
from robot_command import joint_group                   # noqa: E402

from .dynamics import SceneDynamics                     # noqa: E402


class _Cmd:
    """Latest-value command store; every setter is called from RPC/DDS
    threads, every getter from the physics thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.twist = (0.0, 0.0, 0.0)
        self.twist_t = 0.0
        self.estop = False
        self.targets = {}                 # actuator name -> target (latched)
        self.targets_dirty = set()
        self.effort = {}                  # joint name -> tau
        self.effort_t = 0.0
        self.wrench = {}                  # body name -> [fx fy fz tx ty tz]
        self.wrench_t = 0.0
        self.torque_off = set()           # groups with torque disabled
        self.reset_req = None             # None | {} | {"pose": {...}}
        self.caps = (CFG.VMAX, CFG.WMAX)


class SimWorld:
    def __init__(self, scene=None, instance=None):
        import mujoco
        self.mujoco = mujoco
        self.scene = os.path.abspath(scene or CFG.DEFAULT_SCENE)
        self.instance = instance or CFG.INSTANCE
        # HARD single-owner guard at the ENGINE level: whoever constructs a
        # SimWorld claims this instance's flock, or is refused — a stray
        # script instantiating SimWorld must never clobber the live robot's
        # state memmap (that exact accident motivated this).
        self._lock_fd = os.open(CFG.lock_file(self.instance),
                                os.O_RDWR | os.O_CREAT, 0o644)
        try:
            import fcntl
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._lock_fd, 0)
            os.write(self._lock_fd, str(os.getpid()).encode())
        except BlockingIOError:
            os.close(self._lock_fd)
            raise RuntimeError(
                f"simulation instance '{self.instance}' already has a live "
                f"owner — refusing to clobber its state. Use a different "
                f"--instance (and MABEL_STATE_FILE) for side experiments.")
        self.model = mujoco.MjModel.from_xml_path(self.scene)
        # The 360-ray rangefinder ring costs ~12.7 ms/step (vs ~2 ms without)
        # and nothing consumes it from THIS process (cameras render from the
        # memmap; the ROS sensor node re-renders its own) — same measured
        # rationale as the teleop server's Mac path. MABEL_SIM_SENSORS=1 keeps
        # them on for consumers that read sensordata out of the bridge world.
        if not CFG._envb("MABEL_SIM_SENSORS", False):
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
        self.data = mujoco.MjData(self.model)
        # Boot from the model's "home" keyframe, exactly like the reset path in
        # _intake(). A bare MjData leaves ctrl at ZERO while qpos starts at
        # qpos0 — and since the lift stages assemble at mid-travel (ref
        # 0.15875 per stage), the Lift position servo (kp 80000, forcerange
        # ±5000 N) then slams both stages to the bottom on the first steps.
        # That impulse is violent enough to pitch the whole robot over its
        # front wheels at launch (measured: 47.6 deg resting tilt). The home
        # key seeds the lift ctrl to match the assembled pose, so boot ==
        # reset == stable.
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        # scripted world actors (walking people, driving cars, moving belts)
        # from the scene's optional .dynamics.json sidecar
        self.dynamics = SceneDynamics.load(self.scene, self.model)
        self.cmd = _Cmd()
        self._stop = threading.Event()
        self.rate_hz = 0.0                # measured physics rate
        self.started = time.time()

        # ── index caches ────────────────────────────────────────────────────
        m = self.model
        self._act_by_name = {m.actuator(i).name: i for i in range(m.nu)}
        # non-base actuators are command-able as position targets; the base is
        # twist-only (exactly the hardware contract: no one commands wheels)
        self._servo_acts = {n: i for n, i in self._act_by_name.items()
                            if joint_group(n) != "base"}
        self._act_group = {n: joint_group(n) for n in self._servo_acts}
        # actuator -> its joint qpos address (torque-off target tracking)
        self._act_qadr = {}
        for n, i in self._servo_acts.items():
            j = m.actuator_trnid[i][0]
            if 0 <= j < m.njnt:
                self._act_qadr[n] = int(m.jnt_qposadr[j])
        # joint name -> dof address (effort channel)
        self._dof_by_joint = {}
        for j in range(m.njnt):
            jn = m.joint(j).name
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                 mujoco.mjtJoint.mjJNT_SLIDE):
                self._dof_by_joint[jn] = int(m.jnt_dofadr[j])
        self._body_by_name = {m.body(i).name: i for i in range(m.nbody)}
        try:
            self._chassis = m.body("mabel_chassis").id
        except Exception:
            self._chassis = 0

        # ── published state (memmap v2) + viewer pert channel ───────────────
        self.state_path = CFG.state_file(self.instance)
        n = 1 + m.nq + m.nv + m.nu
        self.state = np.memmap(self.state_path, dtype=np.float64, mode="w+",
                               shape=(n,))
        with open(self.state_path + ".scene", "w") as f:
            f.write(self.scene + "\n")
        try:
            self.pert = np.memmap(self.state_path + ".pert", dtype=np.float64,
                                  mode="w+", shape=(m.nbody, 6))
            self.pert[:] = 0.0
        except Exception as e:
            log.warning("pert channel unavailable: %s", e)
            self.pert = None
        self._publish()

    # ── command intake (thread-safe) ─────────────────────────────────────────
    def set_twist(self, vx, vy, wz):
        vmax, wmax = self.cmd.caps
        vx = max(-vmax, min(vmax, float(vx)))
        vy = max(-vmax, min(vmax, float(vy)))
        wz = max(-wmax, min(wmax, float(wz)))
        with self.cmd.lock:
            self.cmd.twist = (vx, vy, wz)
            self.cmd.twist_t = time.time()

    def set_estop(self, active):
        with self.cmd.lock:
            self.cmd.estop = bool(active)

    def set_targets(self, targets):
        with self.cmd.lock:
            for n, v in (targets or {}).items():
                if n in self._servo_acts:
                    self.cmd.targets[n] = float(v)
                    self.cmd.targets_dirty.add(n)

    def set_effort(self, efforts):
        with self.cmd.lock:
            self.cmd.effort = {n: float(v) for n, v in (efforts or {}).items()
                               if n in self._dof_by_joint}
            self.cmd.effort_t = time.time()

    def set_wrench(self, forces):
        with self.cmd.lock:
            self.cmd.wrench = {n: [float(x) for x in (v + [0] * 6)[:6]]
                               for n, v in (forces or {}).items()
                               if n in self._body_by_name}
            self.cmd.wrench_t = time.time()

    def set_group_torque(self, group, enabled):
        with self.cmd.lock:
            if enabled:
                self.cmd.torque_off.discard(group)
            else:
                self.cmd.torque_off.add(group)

    def set_speed_caps(self, vmax=None, wmax=None):
        v, w = self.cmd.caps
        self.cmd.caps = (float(vmax) if vmax else v, float(wmax) if wmax else w)

    def reset(self, pose=None):
        with self.cmd.lock:
            self.cmd.reset_req = {"pose": pose} if pose else {}

    def set_state(self, qpos=None):
        """Seed the world from a full qpos vector (the PREVIEW instance uses
        this to start from the live robot's exact pose). Queued like reset —
        applied on the physics thread before the next substep batch."""
        if qpos is None:
            return False
        q = [float(v) for v in qpos]
        if len(q) != self.model.nq:
            raise ValueError(f"qpos length {len(q)} != nq {self.model.nq}")
        with self.cmd.lock:
            self.cmd.reset_req = {"qpos": q}
        return True

    # ── state readout (RPC `state` op / DDS publishers) ──────────────────────
    def snapshot(self):
        """Small, JSON-able state summary (NOT the bulk channel — that's the
        memmap). Joints as {name: [pos, vel]}, base pose + measured twist."""
        m, d = self.model, self.data
        joints = {}
        for jn, dof in self._dof_by_joint.items():
            j = m.joint(jn)
            joints[jn] = [round(float(d.qpos[int(j.qposadr[0])]), 5),
                          round(float(d.qvel[dof]), 5)]
        b = self._chassis
        xq = d.xquat[b]
        yaw = math.atan2(2 * (xq[0] * xq[3] + xq[1] * xq[2]),
                         1 - 2 * (xq[2] ** 2 + xq[3] ** 2))
        with self.cmd.lock:
            estop = self.cmd.estop
            toff = sorted(self.cmd.torque_off)
            twist = list(self.cmd.twist)
            twist_age = round(time.time() - self.cmd.twist_t, 3)
        return {
            "t": round(float(d.time), 4),
            "cmd": {"twist": [round(v, 3) for v in twist],
                    "twistAge": twist_age},
            "rateHz": round(self.rate_hz, 1),
            "scene": self.scene,
            "instance": self.instance,
            "estop": estop,
            "torqueOff": toff,
            "base": {"x": round(float(d.xpos[b][0]), 4),
                     "y": round(float(d.xpos[b][1]), 4),
                     "z": round(float(d.xpos[b][2]), 4),
                     "yaw": round(yaw, 4)},
            "joints": joints,
        }

    def info(self):
        m = self.model
        return {"scene": self.scene, "instance": self.instance,
                "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody,
                "timestep": m.opt.timestep, "statePath": self.state_path,
                "actuators": sorted(self._servo_acts),
                "dynamics": self.dynamics.describe() if self.dynamics else None,
                "upSince": self.started}

    # ── the physics loop (main thread of the bridge) ─────────────────────────
    def stop(self):
        self._stop.set()

    def run(self):
        mujoco = self.mujoco
        dt = float(self.model.opt.timestep)
        acc = 0.0
        last = time.perf_counter()
        steps = 0
        rate_t0 = last
        log.info("physics up — %s (nq=%d nu=%d dt=%.3fms, instance %s)",
                 os.path.basename(self.scene), self.model.nq, self.model.nu,
                 dt * 1e3, self.instance)
        while not self._stop.is_set():
            now = time.perf_counter()
            acc = min(acc + (now - last), CFG.MAX_CATCHUP)
            last = now
            if acc < dt:
                time.sleep(0.001)
                continue
            self._intake()
            while acc >= dt and not self._stop.is_set():
                self._apply_commands()
                if self.dynamics is not None:
                    self.dynamics.step(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
                self._publish()
                acc -= dt
                steps += 1
            if now - rate_t0 >= 1.0:
                self.rate_hz = steps / (now - rate_t0)
                steps = 0
                rate_t0 = now

    # ── internals (physics thread only) ─────────────────────────────────────
    def _intake(self):
        """Apply queued reset before the substep batch."""
        with self.cmd.lock:
            req, self.cmd.reset_req = self.cmd.reset_req, None
        if req is None:
            return
        mujoco, m, d = self.mujoco, self.model, self.data
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        else:
            mujoco.mj_resetData(m, d)
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        qpos = (req or {}).get("qpos")
        if qpos is not None:                      # full-state seed (preview)
            d.qpos[:] = qpos
            d.qvel[:] = 0.0
            # CLAMP every limited joint back inside its MJCF range. The keyframe
            # path above cannot be out of range by construction, but a SEEDED
            # qpos can: it is whatever the caller captured, and a pose captured
            # from a world where a joint had been driven past its limit would be
            # faithfully restored — reset is then no longer a way out. (The ORCA
            # wrists used to reach ~+3.85 rad against a +0.674 limit before the
            # WBC stopped torque-managing them; see
            # controller/mabel/whole_body_controller/engine.py _managed_for.)
            # Reset must always land on a VALID pose, so make that structural
            # rather than relying on no joint ever escaping again.
            for _j in range(m.njnt):
                if not m.jnt_limited[_j]:
                    continue
                if m.jnt_type[_j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                          mujoco.mjtJoint.mjJNT_SLIDE):
                    continue                      # ball/free: range is not a scalar
                _a = int(m.jnt_qposadr[_j])
                _lo, _hi = m.jnt_range[_j]
                if not (_lo <= d.qpos[_a] <= _hi):
                    log.warning("reset: %s seeded at %+.4f, outside [%+.4f,%+.4f] — clamped",
                                m.joint(_j).name, float(d.qpos[_a]), _lo, _hi)
                    d.qpos[_a] = min(max(float(d.qpos[_a]), _lo), _hi)
            # servo targets track the seeded pose so nothing snaps home
            for n in self._act_qadr:
                d.actuator(n).ctrl[0] = d.qpos[self._act_qadr[n]]
        pose = (req or {}).get("pose")
        if pose:
            try:
                j = m.joint("base_root")
                adr, vadr = int(j.qposadr[0]), int(j.dofadr[0])
                yaw = float(pose.get("yaw", 0.0))
                d.qpos[adr] = float(pose.get("x", 0.0))
                d.qpos[adr + 1] = float(pose.get("y", 0.0))
                d.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0, 0,
                                           math.sin(yaw / 2))
                d.qvel[vadr:vadr + 6] = 0.0
            except Exception as e:
                log.warning("relocate ignored: %s", e)
        with self.cmd.lock:
            self.cmd.targets.clear()
            self.cmd.targets_dirty.clear()
            self.cmd.effort = {}
            self.cmd.wrench = {}
            self.cmd.twist = (0.0, 0.0, 0.0)
        if self.dynamics is not None:
            self.dynamics.reset()
        mujoco.mj_forward(m, d)
        self._publish()
        log.info("world reset%s", f" → {pose}" if pose else "")

    def _apply_commands(self):
        d = self.data
        now = time.time()
        with self.cmd.lock:
            vx, vy, wz = self.cmd.twist
            fresh_twist = (now - self.cmd.twist_t) < CFG.CMD_TIMEOUT
            estop = self.cmd.estop
            targets = {n: self.cmd.targets[n] for n in self.cmd.targets_dirty}
            self.cmd.targets_dirty.clear()
            effort = dict(self.cmd.effort) \
                if (now - self.cmd.effort_t) < CFG.EFFORT_TIMEOUT else {}
            wrench = dict(self.cmd.wrench) \
                if (now - self.cmd.wrench_t) < CFG.WRENCH_TIMEOUT else {}
            torque_off = set(self.cmd.torque_off)

        # base: deadman + estop → zero (steer holds, drive stops)
        if estop or not fresh_twist:
            vx = vy = wz = 0.0
        _swerve.swerve_ik(vx, vy, wz, d)

        # servo targets (latched writes only when they change)
        for n, v in targets.items():
            d.actuator(n).ctrl[0] = v
        # torque-off groups: target tracks the live joint → the limb yields
        if torque_off:
            for n, i in self._servo_acts.items():
                if self._act_group[n] in torque_off and n in self._act_qadr:
                    d.actuator(n).ctrl[0] = d.qpos[self._act_qadr[n]]

        # effort channel (WBC compliance/impedance torques)
        d.qfrc_applied[:] = 0.0
        for jn, tau in effort.items():
            d.qfrc_applied[self._dof_by_joint[jn]] = tau

        # external pushes: desktop-viewer drag (pert memmap) + RPC wrenches
        d.xfrc_applied[:] = 0.0
        if self.pert is not None:
            d.xfrc_applied[:] = self.pert
        for bn, f in wrench.items():
            d.xfrc_applied[self._body_by_name[bn]] += f

    def _publish(self):
        d = self.data
        m = self.model
        s = self.state
        s[0] = d.time
        s[1:1 + m.nq] = d.qpos
        s[1 + m.nq:1 + m.nq + m.nv] = d.qvel
        s[1 + m.nq + m.nv:] = d.ctrl
