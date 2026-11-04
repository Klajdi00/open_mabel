"""Pluggable "advance the world" plant for the teleop server.

The gateway hosts the controller in-process in BOTH the sim and on the real robot; only
the PLANT differs -- the thing that, each control substep, turns the controller's decision
into robot motion and feeds back state:

  * MuJoCoPlant (MABEL_DISPATCH_TARGET=sim:mujoco)  -- the default, the Mac/server-direct
    path. `advance()` applies the desktop perturbation + `mj_step` + the qpos memmap the
    camera server reads. `read_state()` is a no-op (physics owns the state). This is the
    EXACT behaviour the bridge had inline; extracted verbatim so the sim path is unchanged.

  * ROSPlant (MABEL_DISPATCH_TARGET=real:ros)        -- the real robot. `read_state()` mirrors
    the measured encoders/odom into the MuJoCo model (kinematics only, no physics) and
    `advance()` publishes the controller's RobotCommand on the canonical ROS contract that
    the hardware drivers consume. The heavy rclpy + mabel_control imports are LAZY, so the
    Mac sim path never imports ROS.

The bridge's substep loop is plant-agnostic:

    plant.read_state()                       # ros: encoders -> data + mj_forward; sim: no-op
    robot_cmd = teleop.update(pose, joy, dt) # the one control law (unchanged)
    plant.advance(robot_cmd, pert, teleop.wbc.active)   # sim: mj_step; ros: publish contract

Both modes self-pace on the bridge's real-time accumulator: sim `advance()` costs ~8 ms of
physics, ros `advance()` is cheap, and in both cases the loop consumes real elapsed time in
fixed `dt` chunks -> real-time control either way.
"""
import logging
import os
import time

import mujoco

# Module logger — several RealPlant methods log via `_log` (gains publish, the
# FF quarantine warn, …). It used to exist only as a LOCAL inside _arm_gravity,
# so every other reference raised NameError — and the one inside an `except`
# handler REPLACED the caught error and propagated, killing the gateway's
# command/state path each tick ("hardware bridge offline" in the CC while RViz,
# fed directly by the HAL, stayed live; 2026-07-30).
_log = logging.getLogger("mabel.plant")

#: Transmit the effort channel in EVERY control space, not only the torque ones.
#: The controller computes an upper-body gravity feed-forward in all of them
#: (mabel.dynamic_models.gravity); gating its transmission on `wbc_active` meant
#: Drive shipped none of it and the sim robot's arms hung on bare position servos.
#: Set MABEL_UNIFIED_GRAVCOMP=0 to restore the old gating — the before/after arm of
#: controller/experiments/exp_mode_switch_sag.py. See SimBridgePlant.advance.
UNIFIED_GRAVCOMP = os.environ.get("MABEL_UNIFIED_GRAVCOMP", "1").strip().lower() \
    not in ("0", "false", "off", "no")


class MuJoCoPlant:
    """Sim plant: step MuJoCo physics and publish qpos for the camera server.

    `advance()` is the bridge's original inline substep body, moved here unchanged."""

    def __init__(self, model, data, state):
        self.model = model
        self.data = data
        self.state = state          # memmap (nq+1 floats) or None
        self._audio_sink = None

    def read_state(self):
        # Physics owns the state in sim; nothing to mirror in.
        pass

    # ── audio (symmetric with ROSPlant so the :9610 plane is world-agnostic) ──
    def send_speaker_audio(self, pcm: bytes) -> None:
        """No sound card behind the sim plant — drop the frame.

        A no-op and not a loopback ON PURPOSE: echoing the operator's own voice
        back up the mic stream would sound like the robot is hearing them, and
        the Maps-tab lesson (a dummy source nobody could tell from the real one)
        is that a convincing fake is worse than an honest silence. The :9610 plane
        reports `audio_available() == False`, so the apps say WHY.
        """

    def set_mic_sink(self, sink) -> None:
        self._audio_sink = sink     # never called: sim has no mic

    def set_mic_enabled(self, enabled: bool) -> None:
        pass                        # no mic to mute

    def audio_available(self) -> bool:
        return False

    def reset(self):
        # In-process physics: home the model right here (the bridge owns self.data).
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.state is not None:
            self.state[0] = self.data.time
            self.state[1:] = self.data.qpos

    def publish_ros_state(self):
        # No ROS on the Mac sim path; SimRosPlant overrides this.
        pass

    def place_base(self, x, y, yaw=0.0):
        """Teleport the chassis free joint to (x, y, yaw) — sim-only, used by
        the studio's relocate-reset right after reset(). Keeps the default
        spawn height and zeroes the root velocity so nothing launches."""
        import math
        m, d = self.model, self.data
        try:
            j = m.joint("base_root")
        except Exception:
            return False
        adr, vadr = int(j.qposadr[0]), int(j.dofadr[0])
        d.qpos[adr + 0] = float(x)
        d.qpos[adr + 1] = float(y)
        d.qpos[adr + 3:adr + 7] = (math.cos(yaw / 2), 0.0, 0.0,
                                   math.sin(yaw / 2))
        d.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(m, d)
        if self.state is not None:
            self.state[0] = d.time
            self.state[1:] = d.qpos
        return True

    def set_hand_torque(self, enabled, side=None):
        # Sim hands are always actuated, so hand torque-enable is a real-robot concept
        # only. No-op here; the real ROSPlant delegates to its impl which publishes
        # /<side>_hand/torque_enable. Keeps the e-stop->torque call site mode-agnostic.
        pass

    def advance(self, robot_cmd, pert, wbc_active):
        data = self.data
        # xfrc_applied (desktop-viewer Ctrl+drag perturbation + any app external_force
        # push) is now composed ONCE per substep by the bridge BEFORE the control law
        # runs (so the whole-body base-wrench estimate sees the same forces physics
        # will). Nothing to write here — just step the world. `pert`/`wbc_active` are
        # kept in the signature for the ROS/real plants and call-site symmetry.
        mujoco.mj_step(self.model, data)
        # Publish qpos every substep so the camera server renders one frame per new pose.
        if self.state is not None:
            self.state[0] = data.time
            self.state[1:] = data.qpos


class SimBridgePlant:
    """sim:bridge — the gateway twin of ROSPlant with the SIMULATION BRIDGE as
    the robot. The bridge (simulation_bridge/) is the sole owner of mj_step,
    exactly as hardware_bridge owns the physical devices; this plant:

      read_state()  mirrors the bridge's published state memmap
                    ([t, qpos, qvel, ctrl]) into the local MjData + mj_forward
                    — the same encoders→mirror pattern ROSPlant uses
      advance()     projects the controller's decision onto the bridge over
                    its ROS-free RPC: base twist (RobotCommand.base), servo
                    targets (the non-base slice of robot_cmd.ctrl), the WBC
                    effort channel (qfrc_applied), and app external wrenches
                    (xfrc_applied). Stale commands are zeroed BRIDGE-side
                    (deadman), so a dropped link is safe.

    Same building blocks, different source: swap real:ros ↔ sim:bridge and
    the server, ROS side, and every client behave identically — they are just
    told kind = "sim"."""

    CMD_HZ = 120.0                   # RPC command rate cap (loop runs faster)

    def __init__(self, model, data, state=None, host=None, port=None):
        import numpy as _np
        import sys as _sys
        _sb = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _sb not in _sys.path:
            _sys.path.insert(0, _sb)
        from simulation_bridge import config as _scfg
        from simulation_bridge.rpc import SimBridgeClient
        from robot_command import joint_group

        self.model = model
        self.data = data
        self.state = None            # the BRIDGE publishes the memmap, not us
        self._np = _np
        self._state_path = _scfg.state_file()
        self._mm = None              # bridge state memmap (lazy: bridge may
        self._mm_legacy = False      # still be starting)
        self.rpc = SimBridgeClient(host or _scfg.RPC_HOST,
                                   port or _scfg.rpc_port())
        # non-base actuators = the servo-target contract (base is twist-only)
        self._servo_idx = {}         # actuator index -> name
        for i in range(model.nu):
            n = model.actuator(i).name
            if joint_group(n) != "base":
                self._servo_idx[i] = n
        self._dof_by_joint = {}
        for j in range(model.njnt):
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                     mujoco.mjtJoint.mjJNT_SLIDE):
                self._dof_by_joint[model.joint(j).name] = int(model.jnt_dofadr[j])
        self._body_names = {i: model.body(i).name for i in range(model.nbody)}
        self._last_sent = {}         # actuator name -> last target sent
        self._last_cmd_t = 0.0
        self._last_estop = None
        self._sent_effort = False    # was the last effort frame non-empty?

    # ── state in ─────────────────────────────────────────────────────────────
    def _attach(self):
        m = self.model
        full = 1 + m.nq + m.nv + m.nu
        try:
            size = os.path.getsize(self._state_path) // 8
        except OSError:
            return False
        if size >= full:
            self._mm = self._np.memmap(self._state_path, dtype=self._np.float64,
                                       mode="r", shape=(full,))
            self._mm_legacy = False
        elif size >= 1 + m.nq:       # legacy [t, qpos] writer
            self._mm = self._np.memmap(self._state_path, dtype=self._np.float64,
                                       mode="r", shape=(1 + m.nq,))
            self._mm_legacy = True
        else:
            return False
        return True

    def read_state(self):
        if self._mm is None and not self._attach():
            return
        m, d = self.model, self.data
        try:
            d.qpos[:] = self._mm[1:1 + m.nq]
            if not self._mm_legacy:
                d.qvel[:] = self._mm[1 + m.nq:1 + m.nq + m.nv]
            d.time = float(self._mm[0])
        except Exception:
            self._mm = None
            return
        mujoco.mj_forward(m, d)

    # ── command out ──────────────────────────────────────────────────────────
    def advance(self, robot_cmd, pert, wbc_active):
        now = time.time()
        if now - self._last_cmd_t < 1.0 / self.CMD_HZ:
            return
        self._last_cmd_t = now
        d = self.data
        # base twist — the smoothed, tip-safe command (motion model ran once
        # in the controller); e-stop rides the engaged flag like ROSPlant
        vx, vy, wz = (robot_cmd.base if getattr(robot_cmd, "base", None)
                      else (0.0, 0.0, 0.0))
        self.rpc.try_call("set_twist", vx=float(vx), vy=float(vy), wz=float(wz))
        estop = (getattr(robot_cmd, "engaged", 1.0) == 0.0)
        if estop != self._last_estop:
            self._last_estop = estop
            self.rpc.try_call("set_estop", active=bool(estop))
        # servo targets: the non-base slice of the actuator vector, sent only
        # on change (latched bridge-side, like a real motor setpoint)
        ctrl = getattr(robot_cmd, "ctrl", None)
        if ctrl is None:
            ctrl = d.ctrl
        targets = {}
        for i, n in self._servo_idx.items():
            v = float(ctrl[i])
            if abs(v - self._last_sent.get(n, 1e9)) > 1e-5:
                targets[n] = v
                self._last_sent[n] = v
        if targets:
            self.rpc.try_call("set_targets", targets=targets)
        # Effort channel: the WBC's compliance/impedance torques AND the upper-body
        # GRAVITY feed-forward — which the controller now computes in every control
        # space, not only the torque ones (mabel.dynamic_models.gravity).
        #
        # This used to be gated `if wbc_active`, which made the gravity comp a
        # fiction on this deployment: in Drive the gateway dutifully wrote
        # qfrc_applied into its own kinematic MIRROR — a model it never steps — and
        # then threw it away, so the physics owner saw ZERO compensated DoFs and the
        # arms hung on bare position servos (measured 0/22 in Drive vs 14/22 in
        # Joint/Task). The operator felt that as the upper body settling whenever
        # they changed space. `wbc_active` is kept in the signature for the ROS and
        # in-process plants; it no longer decides whether gravity reaches the robot.
        #
        # Cost is one dict of nonzero DoFs per RPC tick (120 Hz), which is exactly
        # what the torque modes already paid. Stale efforts still zero themselves
        # bridge-side after CFG.EFFORT_TIMEOUT, so a dropped link remains safe.
        if wbc_active or UNIFIED_GRAVCOMP:
            eff = {}
            for jn, dof in self._dof_by_joint.items():
                tau = float(d.qfrc_applied[dof])
                if abs(tau) > 1e-6:
                    eff[jn] = tau
            if eff or self._sent_effort:
                self._sent_effort = bool(eff)
                self.rpc.try_call("set_effort", efforts=eff)
        # app external pushes (composed onto the mirror's xfrc by the loop)
        forces = {}
        xf = d.xfrc_applied
        for b in range(1, xf.shape[0]):
            if abs(xf[b]).max() > 1e-9:
                forces[self._body_names[b]] = [float(x) for x in xf[b]]
        if forces:
            self.rpc.try_call("set_wrench", forces=forces)

    # ── the rest of the Plant surface ────────────────────────────────────────
    def reset(self):
        self._last_sent.clear()
        self.rpc.try_call("reset")

    def place_base(self, x, y, yaw=0.0):
        self._last_sent.clear()
        return bool(self.rpc.try_call(
            "reset", pose={"x": float(x), "y": float(y), "yaw": float(yaw)}))

    def publish_ros_state(self):
        pass

    def set_hand_torque(self, enabled, side=None):
        for s in ([side] if side else ["left", "right"]):
            self.rpc.try_call("set_group_torque", group=f"{s}_hand",
                              enabled=bool(enabled))

    def shutdown(self):
        try:
            self.rpc.close()
        except Exception:
            pass


class ROSPlant:
    """Real-robot plant: mirror measured state in, publish the controller command out.

    Wraps `mabel_control.ros_plant.ROSPlant` (the rclpy pub/sub + the offline-tested
    joint_mirror + command_extract) and owns the rclpy node + a background spin thread so
    subscription callbacks (joint_states / odom) keep the mirror fresh. Constructed only
    when MABEL_DISPATCH_TARGET starts with 'real', so the Mac never imports rclpy."""

    def __init__(self, model, data, state=None, node_name="mabel_gateway"):
        import threading
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Empty
        from mabel_control.ros_plant import ROSPlant as _RosPlantImpl

        self.model = model
        self.data = data
        self.state = state          # qpos memmap the camera server reads (nq+1 floats)
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(node_name)
        self._impl = _RosPlantImpl(self._node, model, data)
        self._arm_gravity = self._build_arm_gravity()
        # Feedforward quarantine (firmware joint names) — see _arm_effort for why.
        self._arm_ff_skip = {t.strip() for t in
                             os.environ.get("MABEL_ARM_FF_SKIP", "").split(",") if t.strip()}
        if self._arm_ff_skip:
            _log.warn(f"arm gravity FF QUARANTINED for {sorted(self._arm_ff_skip)} "
                      "(MABEL_ARM_FF_SKIP) — those joints get no feedforward")
        # Reset relay: the sim node (mabel_sim) OWNS the physics, so a reset must go
        # over ROS — resetting our local mirror here would be silently overwritten by
        # the next read_state(). The sim node subscribes to /mabel/reset_sim and homes
        # itself; on the real robot nothing subscribes, so this is a harmless no-op.
        self._reset_pub = self._node.create_publisher(Empty, "/mabel/reset_sim", 10)
        # Planar LiDAR passthrough: latch the newest /scan (RPLIDAR via the ROS
        # graph) as the wire-ready dict the apps' top-down LiDAR tile renders.
        # Decimated to ≤360 wire points and rounded so a sweep stays a few KB; the
        # bridge attaches it to robot_state only when the stamp changes (~scan rate).
        self._lidar = None
        try:
            import math
            from sensor_msgs.msg import LaserScan
            from rclpy.qos import qos_profile_sensor_data

            def _on_scan(msg):
                step = max(1, len(msg.ranges) // 360)
                rmax = float(msg.range_max)
                ranges = [round(float(r), 3) if math.isfinite(r) and r > 0 else rmax
                          for r in msg.ranges[::step]]
                self._lidar = {"angleMin": float(msg.angle_min),
                               "angleInc": float(msg.angle_increment) * step,
                               "rangeMax": rmax, "stamp": time.time(),
                               "ranges": ranges}

            self._node.create_subscription(LaserScan, "/scan", _on_scan,
                                           qos_profile_sensor_data)
        except Exception:
            pass                # no LaserScan on this graph → tile shows NO DATA

        # Audio passthrough (:9610 <-> the HAL's sound card, over DDS). Voice is
        # NOT part of the control contract — it moves nothing — so it rides its
        # own topics and never touches RobotCommand or the safety gate.
        self._audio_sink = None                 # set by the gateway (AudioHub.publish_mic)
        self._pub_audio = None
        self._pub_mic_enable = None
        try:
            from std_msgs.msg import UInt8MultiArray
            from rclpy.qos import qos_profile_sensor_data

            def _on_mic(msg):
                sink = self._audio_sink
                if sink is not None:
                    # Straight to the plane: no queue here. The hub owns the FIFO
                    # (audio is a sequence — see audio_streams.py); a second buffer
                    # in this callback would only add latency nobody can see.
                    sink(bytes(msg.data))

            from std_msgs.msg import Bool
            self._pub_audio = self._node.create_publisher(
                UInt8MultiArray, "/audio/speaker", 10)
            self._pub_mic_enable = self._node.create_publisher(
                Bool, "/audio/enable", 10)
            self._node.create_subscription(UInt8MultiArray, "/audio/mic", _on_mic,
                                           qos_profile_sensor_data)
        except Exception:
            pass                # no audio on this graph → the :9610 plane says so

        # Spin in the background so /joint_states + /odom callbacks update the mirror while
        # the control loop runs on the main thread.
        self._executor_thread = threading.Thread(
            target=self._spin, name="ros-plant-spin", daemon=True)
        self._stop = False
        self._executor_thread.start()

    def lidar_scan(self):
        """Newest planar sweep as a wire-ready dict (None until /scan arrives)."""
        return self._lidar

    def motor_health(self):
        """{"t": wall, "joints": {name: {torqueNm, tempC, err, fault, enabled, live,
        stale_ms, ...}}} merged from the HAL's four */motor_health topics, or None
        until the first one arrives. Same shape as map_stream.latest_motor_health(),
        which the gateway prefers when the map plane runs in-process; with
        perception EXTERNAL (thor since 2026-07-13, the Mac always) that plane is not
        here, and this is what keeps robot_state.motorHealth populated."""
        impl = self.__dict__.get("_impl")
        joints = getattr(impl, "_motor_health", None) if impl is not None else None
        if not joints:
            return None
        return {"t": getattr(impl, "_motor_health_t", 0.0), "joints": joints}

    # ── audio (the :9610 plane's seam onto the real sound card) ───────────────
    def send_speaker_audio(self, pcm: bytes) -> None:
        """Publish one operator voice frame for the robot's speaker."""
        pub = self._pub_audio
        if pub is None or not pcm:
            return                      # no audio on this graph: drop, never queue
        from std_msgs.msg import UInt8MultiArray
        msg = UInt8MultiArray()
        msg.data = list(pcm)
        pub.publish(msg)

    def set_mic_sink(self, sink) -> None:
        """Where mic frames go (AudioHub.publish_mic). None to detach."""
        self._audio_sink = sink

    def set_mic_enabled(self, enabled: bool) -> None:
        """Mute/unmute the ROBOT's mic at the HAL (not just on this socket), so
        every client — and the bar's own status row — agrees on who is hot."""
        pub = self._pub_mic_enable
        if pub is not None:
            from std_msgs.msg import Bool
            pub.publish(Bool(data=bool(enabled)))

    def audio_available(self) -> bool:
        return self._pub_audio is not None

    def __getattr__(self, name):
        # Delegate nav forwarding (send_nav_goal / send_nav_goal_preview /
        # clear_goal_marker, etc.) to the wrapped impl so the bridge's
        # `plant.send_nav_goal(...)` works on the REAL robot exactly as it does in
        # sim. NOTE: __getattr__ runs only for attributes NOT found normally, and
        # reads _impl from __dict__ to avoid recursing during __init__.
        impl = self.__dict__.get("_impl")
        if impl is not None and hasattr(impl, name):
            return getattr(impl, name)
        raise AttributeError(name)

    def _spin(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        # ONE persistent executor. `rclpy.spin_once(node)` constructs a fresh
        # SingleThreadedExecutor, adds the node, and tears it all down EVERY
        # call — measured at ~26% of the control-loop core on thor (py-spy
        # 2026-07-21, wait_for_ready_callbacks rebuilding the waitset per call).
        # A persistent executor keeps the waitset and gives that GIL time back
        # to the 50 Hz control loop.
        ex = SingleThreadedExecutor()
        ex.add_node(self._node)
        try:
            while not self._stop and rclpy.ok():
                ex.spin_once(timeout_sec=0.05)
        finally:
            ex.remove_node(self._node)

    _ARM_FW = {f"{side}_arm_{i}": f"{side[0]}_j{i}"
               for side in ("left", "right") for i in range(1, 8)}
    # Joints the SHARED gravity evaluation covers: the 14 arm joints + the torso.
    # One mj_forward serves all 15 (ArmGravityCompensator optional-joint support,
    # 2026-07-29): the torso's tau becomes ARM-POSE-AWARE (its subtree carries both
    # arms — the single-sinusoid law was fitted at ONE arm pose and went stiff-wrong
    # the moment an arm extended) and the arms' tau becomes TORSO-LEAN-AWARE.
    _FF_FW = {**_ARM_FW, "torso": "torso"}

    def _build_arm_gravity(self):
        """The arm gravity feed-forward, or None.

        Lives HERE (real plant only) because the HAL no longer carries a control law:
        the bridge streams `T = kp*(pos-q) + kd*(vel-dq) + t_ff` and t_ff comes from
        /arms/command.effort. With no feed-forward the position term must fight gravity
        alone and the arm sags by tau_g/kp (measured on j2: 17deg -> -7.2deg; with FF
        the same target holds to -0.3deg). Best-effort: no MuJoCo / no model => None =>
        t_ff 0, i.e. exactly today's behaviour, never a crash.

        MODEL frame (`frame="model"`), so no calibration `dir` map is needed and this
        stays free of any mabel_hw import — the bridge's subsystem dir-corrects effort
        into the raw motor frame itself. Passing the RAW frame here would DOUBLE-apply
        dir and drive every dir=-1 joint the wrong way into its stop."""
        import os
        _log = self._node.get_logger()
        if os.getenv("MABEL_ARM_GRAVITY_FF", "1").strip().lower() in ("0", "false", "off"):
            _log.warn("arm gravity feed-forward DISABLED (MABEL_ARM_GRAVITY_FF=0) — "
                      "the arms will sag by tau_g/kp")
            return None
        try:
            from mabel.arm_control import ArmGravityCompensator
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mjcf = os.path.join(here, "simulation", "mabel_mujoco", "models",
                                "mabel_full.xml")
            if not os.path.exists(mjcf):
                _log.warn(f"arm gravity feed-forward: {mjcf} missing — t_ff = 0")
                return None
            # dir_by_joint = {} is CORRECT for frame="model" (dir is only the raw map).
            # tmax_by_joint = {} disables the local clamp: the HAL already clamps t_ff
            # to each motor's rated t_max from the firmware manifest, which is the
            # authority (OpenArmLink.send).
            g = ArmGravityCompensator(mjcf, {}, {}, log=lambda m, *a: _log.info(str(m)))
            _log.info("arm gravity feed-forward ON -> /arms/command.effort (model frame)")
            return g
        except Exception as e:                                # noqa: BLE001
            _log.warn(f"arm gravity feed-forward unavailable ({e}) — t_ff = 0")
            return None

    # Gravity varies with POSE, not with the control rate — so the re-evaluation
    # is scheduled on POSE MOTION, not on a clock (GravityFFScheduler,
    # controller/mabel/arm_control/ff_schedule.py). History: tff() is a full
    # mj_forward (measured 3.1 ms/call on thor 2026-07-30), so 2026-07-19..21 it
    # was clock-cached at 4 Hz — but a HAND PUSH moves the pose continuously and
    # a fixed 0.25 s cache made the FF up to 250 ms stale: each refresh landed
    # as a 0.6–1.3 N·m torque step every quarter second (the slew stretched each
    # into a ramp — a STAIRCASE), felt as "step-like motion when I push the arm
    # in stiff" (operator report 2026-07-30). Pose-triggered evals track a push
    # at up to 25 Hz (~8% of one core, only WHILE moving) with ≤0.1 N·m steps,
    # and idle cost stays at the old 4 Hz.
    _ARM_EFF_PERIOD = 0.25            # scheduler max_period (drift safety)

    def _arm_effort(self):
        """{urdf joint: tau_g [N.m, model frame]} at the MEASURED pose — the 14 arm
        joints + 'torso' (split by advance() onto their command topics).
        Re-evaluated when the pose MOVES (see the scheduler note above)."""
        g = self._arm_gravity
        if g is None:
            return {}
        now = time.monotonic()
        sched = getattr(self, "_ff_sched", None)
        if sched is None:
            from mabel.arm_control.ff_schedule import GravityFFScheduler
            sched = self._ff_sched = GravityFFScheduler(
                min_period=0.04, max_period=self._ARM_EFF_PERIOD, pose_eps=0.01)
        try:
            jn, jp = self._impl._joints          # measured, URDF names, MODEL frame
            if not jn:
                return {}
            # Evaluate tau_g at the COMMANDED pose, bounded to ±0.2 rad of the
            # measured one (2026-07-29, "J1/J2 too weak to reach the target"):
            # at the compliant gain set the PD had to bridge the gravity
            # DIFFERENCE between measured and target — several N·m on the
            # shoulders, i.e. a large standing error AND sustained motor
            # current. FF-at-(bounded)-command lets the feedforward do that
            # lifting. The bound keeps a runaway/stale command from commanding
            # runaway feedforward (the glide limiter already caps the real
            # command-vs-measured gap at 0.15 rad). MABEL_ARM_FF_AT_CMD=0
            # restores measured-pose evaluation.
            cmd = getattr(self._impl, "_last_arm_cmd", None) \
                if os.getenv("MABEL_ARM_FF_AT_CMD", "1") not in ("0", "false") \
                else None
            def _q_ff(n, q):
                if cmd is None:
                    return float(q)
                qc = cmd.get(n)
                if qc is None:
                    return float(q)
                return float(q) + max(-0.2, min(0.2, float(qc) - float(q)))
            pose = {self._FF_FW[n]: _q_ff(n, q) for n, q in zip(jn, jp)
                    if n in self._FF_FW}
            if not pose:
                return {}
            # Pose-triggered gate: keep streaming the smoothed cache until the
            # EVALUATION pose (measured ± the command bound) actually moved.
            if (getattr(self, "_arm_eff_cache", None) is not None
                    and not sched.due(now, pose)):
                return self._smoothed_arm_eff(now)
            tau = g.tff(pose, frame="model")
            sched.mark(now, pose)
            # Calibrate the torso's MJCF prediction onto the real joint (LIVE trim
            # from control_param.json torso.gravity_feedforward, fitted by
            # exp_torso_gravity_sysid.py). Applied BEFORE the slew smoothing so the
            # stream ramps toward the final deployed value.
            if "torso" in tau:
                try:
                    from mabel.torso_control import posture_tau
                    tau["torso"] = posture_tau(tau["torso"])
                except Exception:                             # noqa: BLE001
                    pass                                      # identity trim
            # QUARANTINE list (MABEL_ARM_FF_SKIP="l_j4,..."): joints whose motor must
            # not receive feedforward — a generic escape hatch for a faulty motor
            # (torque into a miscommutating motor is pure heat). EMPTY by default:
            # every motor is treated identically. History: l_j4's original motor was
            # quarantined here 2026-07-28 (miscommutation), replaced + recalibrated
            # the same day, quarantine lifted 2026-07-29 — pair with
            # MABEL_HW_<SIDE>_ARM_KP_J<n>=0 if ever needed again.
            for fw in self._arm_ff_skip:
                tau.pop(fw, None)
            inv = {v: k for k, v in self._FF_FW.items()}
            out = {inv[fw]: t for fw, t in tau.items() if fw in inv}
        except Exception:                                     # noqa: BLE001
            out = {}                                          # never block the command
        self._arm_eff_cache = out
        self._arm_eff_t = now
        return self._smoothed_arm_eff(now)

    # t_ff slew rate (N·m/s) — a SPIKE guard, no longer a staircase-stretcher.
    # With clock-cached 4 Hz evals each refresh was a multi-N·m step and the slew
    # had to be slow (10) to hide it — which ADDED up to ~130 ms of extra FF lag
    # per step (half of the 2026-07-30 "step-like when pushed" artifact; the
    # other half was the stale pose itself). Pose-triggered evals make refresh
    # steps ≤~0.1 N·m, so the slew only needs to guard eval glitches/noise: 40
    # crosses a full realistic gravity swing in <0.5 s and a normal refresh step
    # in ~3 ms — the FF is effectively real-time against the measured pose.
    _ARM_EFF_SLEW = 40.0

    def _smoothed_arm_eff(self, now: float) -> dict:
        tgt = self._arm_eff_cache or {}
        sm = getattr(self, "_arm_eff_smooth", None)
        if sm is None:
            sm = self._arm_eff_smooth = dict(tgt)
            self._arm_eff_smooth_t = now
            return dict(sm)
        step = self._ARM_EFF_SLEW * max(0.0, min(0.1, now - self._arm_eff_smooth_t))
        self._arm_eff_smooth_t = now
        for j, t in tgt.items():
            cur = sm.get(j, 0.0)
            d = t - cur
            sm[j] = t if abs(d) <= step else cur + (step if d > 0 else -step)
        for j in list(sm):
            if j not in tgt:                 # joint left the FF set → ramp it out
                cur = sm[j]
                sm[j] = 0.0 if abs(cur) <= step else cur - (step if cur > 0 else -step)
                if sm[j] == 0.0:
                    del sm[j]
        return dict(sm)

    def read_state(self):
        # Encoders + odom -> data.qpos; mj_forward (kinematics only, no physics step).
        self._impl.read_state()
        # Publish the mirrored pose to the qpos memmap so the camera server (and the
        # desktop view_robot.py mirror) render the LIVE robot — without this the
        # memmap is written once at init and frozen, so on-body camera views never
        # update (the sim path writes it in MuJoCoPlant.advance(); the ROS path has
        # no physics step, so we write it here right after mirroring the encoders).
        # state[0] is the camera server's "pose changed → re-render" token. The ROS
        # path never calls mj_step, so data.time stays 0; use a real, ever-advancing
        # clock so the token changes every cycle and the cameras stream live (not the
        # frozen single frame you get when the token never moves).
        if self.state is not None:
            self.state[0] = time.monotonic()
            self.state[1:] = self.data.qpos

    def reset(self):
        # Ask the physics owner (the sim node) to home; mirror our local kinematic
        # copy too so the re-anchor / camera view is consistent until the next
        # read_state() mirrors the (now homing) robot back in.
        from std_msgs.msg import Empty
        self._reset_pub.publish(Empty())
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def publish_ros_state(self):
        # The real robot's drivers publish state; nothing to do here.
        pass

    def advance(self, robot_cmd, pert, wbc_active):
        # No physics and no desktop perturbation on the real robot: publish the controller's
        # decision on the canonical contract; the hardware drivers realise it.
        # ONE gravity evaluation covers arms + torso; split by target topic here.
        eff = self._arm_effort()                     # {urdf joint: tau_g, model frame}
        torso_tau = eff.pop("torso", None)
        self._impl.arm_effort = eff                  # tau_g -> /arms/command.effort
        # tau_g -> /body/command.effort. Empty dict = "no posture-aware value this
        # tick" (arms not streaming / MuJoCo absent) — ros_plant._body_effort then
        # falls back to the arm-pose-blind sinusoid hold law.
        self._impl.body_effort = ({"torso": float(torso_tau)}
                                  if torso_tau is not None else {})
        # SOFT mode: drag-and-hold the arms — updates data.ctrl in place so
        # write() extracts the moving targets like any other command.
        self._soft_float(robot_cmd)
        # Mode-appropriate PD gains → /arms/gains (STIFF = arm.gains; SOFT = the
        # slider-scaled soft set). Published on CHANGE only (latched topic).
        self._maybe_publish_gains(robot_cmd)
        self._impl.write(robot_cmd)

    def _maybe_publish_gains(self, robot_cmd):
        """Ship {kp, kd} per URDF arm joint to the HAL when the mode or the
        app's stiffness slider changed. STIFF runs the arm.gains set unscaled;
        SOFT scales kp by the slider (arm.soft_gains: kp_scale_min..max) so the
        operator can dial how firm the drag-and-hold feels — 'via the iOS app
        the user can adjust the stiffness gain' (operator directive)."""
        soft = str(getattr(robot_cmd, "mode", "")).startswith("WBC:compliance")
        stiffness = float(getattr(self, "spec_stiffness", 0.5) or 0.0)
        key = (bool(soft), round(stiffness, 2))
        if key == getattr(self, "_gains_key", None):
            return
        try:
            from mabel.control_params import arm_gains, arm_soft_gains
            base = arm_gains()
            if not base:
                return
            if soft:
                sg = arm_soft_gains()
                # Floor 0.5: real soft must HOLD with PD against the imperfect
                # gravity FF (sag = residual/kp) — softness comes from the
                # drag-follow behavior, not weak gains (sim-soft only feels
                # perfect because it is a torque law with a perfect model).
                lo = float(sg.get("kp_scale_min", 0.5))
                hi = float(sg.get("kp_scale_max", 1.0))
                kp_s = lo + (hi - lo) * max(0.0, min(1.0, stiffness))
                kd_s = float(sg.get("kd_scale", 1.0))
            else:
                kp_s, kd_s = 1.0, 1.0
            inv = {v: k for k, v in self._ARM_FW.items()}   # fw -> urdf
            kp, kd = {}, {}
            for fw, g in base.items():
                if fw not in inv or not isinstance(g, dict):
                    continue
                if "kp" in g:
                    kp[inv[fw]] = float(g["kp"]) * kp_s
                if "kd" in g:
                    kd[inv[fw]] = float(g["kd"]) * kd_s
            if kp or kd:
                self._impl.publish_arm_gains({"kp": kp, "kd": kd})
                self._gains_key = key
                _log.info(f"arm gains → HAL: {'SOFT' if soft else 'STIFF'} "
                          f"(kp×{kp_s:.2f}, kd×{kd_s:.2f}, slider {stiffness:.2f})")
        except Exception as e:                                # noqa: BLE001
            _log.warn(f"arm gains publish failed ({e}) — HAL keeps its defaults")

    # ── FLOATING soft mode (admittance) ───────────────────────────────────────
    # In SOFT (compliance) the facade freezes the arm position targets — the
    # operator could drag the arm only by winding up kp·error, and it rubber-
    # banded back on release. SoftArmAdmittance (controller/mabel/arm_control/
    # admittance.py — the MATH lives there; this is only the host, same layering
    # as the gravity FF) makes the TARGET follow the drag from the motors' own
    # torque feedback, so the arm FLOATS and parks where released. Joints on the
    # FF quarantine list are excluded (their τ feedback is the broken signal the
    # quarantine exists for).
    def _soft_float(self, robot_cmd):
        # SOFT mode default = POSITION-based drag-and-hold (_soft_drag below,
        # operator design 2026-07-30): nonzero P, target follows the pushed arm,
        # freezes on release. The τ-OBSERVER admittance below is QUARANTINED
        # opt-in (MABEL_SOFT_FLOAT=1, bench only): live test found it ERRATIC on
        # the real arm — no torque feedback in the drag-follow loop, so it
        # cannot self-excite.
        if os.getenv("MABEL_SOFT_FLOAT", "0").strip().lower() not in ("1", "true", "on"):
            self._soft_drag(robot_cmd)
            return
        soft = str(getattr(robot_cmd, "mode", "")).startswith("WBC:compliance")
        adm = getattr(self, "_admit", None)
        if not soft:
            if adm is not None and adm.engaged:
                adm.disengage()
            return
        if getattr(self, "_soft_float_dead", False):
            return
        if self._arm_gravity is None or getattr(self._arm_gravity, "dyn", None) is None:
            return                                  # no gravity model → no observer
        now = time.monotonic()
        # measured state, fw-keyed ('l_j1'..'r_j7'), MODEL frame
        jn, jp = self._impl._joints
        q = {self._ARM_FW[n]: float(v) for n, v in zip(jn, jp) if n in self._ARM_FW}
        dyn = self._impl._dyn
        tau, qd = {}, {}
        for n, fw in self._ARM_FW.items():
            d = dyn.get(n)
            if d is not None:
                qd[fw], tau[fw] = d[0], d[1]
        if adm is None:
            try:
                from mabel.arm_control import SoftArmAdmittance
                from mabel.control_params import arm_soft_float
                joints = tuple(fw for fw in self._ARM_FW.values()
                               if fw not in self._arm_ff_skip)
                adm = self._admit = SoftArmAdmittance(
                    self._arm_gravity, joints, params=arm_soft_float(),
                    log=lambda m: _log.info(str(m)))
                # joint ranges (rad) from the mirror model, for the stop margin
                self._admit_limits = {}
                for fw in joints:
                    mj = f"{'Left' if fw[0] == 'l' else 'Right'} Arm {fw[-1]}"
                    try:
                        self._admit_limits[fw] = tuple(self.model.joint(mj).range)
                    except Exception:
                        pass
                _log.info(f"soft-float admittance ready — {len(joints)} joints "
                          f"(quarantined: {sorted(self._arm_ff_skip) or 'none'})")
            except Exception as e:                   # noqa: BLE001
                self._admit = None
                self._soft_float_dead = True
                _log.warn(f"soft-float admittance unavailable ({e}) — "
                          "SOFT stays hold-with-compliant-gains")
                return
        if not adm.engaged:
            adm.engage(q, now)
            self._admit_t = now
            return                                   # first tick: seed only, no motion
        dt = min(max(now - getattr(self, "_admit_t", now), 1e-3), 0.1)
        self._admit_t = now
        targets = adm.update(q, tau, dt, now, limits=self._admit_limits, qd_meas=qd)
        for fw, t in targets.items():
            mj = f"{'Left' if fw[0] == 'l' else 'Right'} Arm {fw[-1]}"
            try:
                self.data.actuator(mj).ctrl[0] = float(t)
            except Exception:
                pass

    # ── SOFT default: position-based drag-and-hold (no torque in the loop) ────
    def _soft_drag(self, robot_cmd):
        soft = str(getattr(robot_cmd, "mode", "")).startswith("WBC:compliance")
        drag = getattr(self, "_drag", None)
        if not soft:
            if drag is not None and drag.engaged:
                drag.disengage()
            return
        if getattr(self, "_soft_drag_dead", False):
            return
        now = time.monotonic()
        jn, jp = self._impl._joints
        q = {self._ARM_FW[n]: float(v) for n, v in zip(jn, jp) if n in self._ARM_FW}
        dyn = self._impl._dyn
        qd = {}
        for n, fw in self._ARM_FW.items():
            d = dyn.get(n)
            if d is not None:
                qd[fw] = d[0]
        if drag is None:
            try:
                from mabel.arm_control import DragFollow
                from mabel.control_params import arm_soft_drag
                joints = tuple(self._ARM_FW.values())
                drag = self._drag = DragFollow(joints, params=arm_soft_drag(),
                                               log=lambda m: _log.info(str(m)))
                self._drag_limits = {}
                for fw in joints:
                    mj = f"{'Left' if fw[0] == 'l' else 'Right'} Arm {fw[-1]}"
                    try:
                        self._drag_limits[fw] = tuple(self.model.joint(mj).range)
                    except Exception:
                        pass
                _log.info(f"soft drag-and-hold ready — {len(joints)} joints")
            except Exception as e:                    # noqa: BLE001
                self._soft_drag_dead = True
                _log.warn(f"soft drag-and-hold unavailable ({e}) — "
                          "SOFT stays hold-with-current-gains")
                return
        if not drag.engaged:
            drag.engage(q)
            self._drag_t = now
            return
        dt = min(max(now - getattr(self, "_drag_t", now), 1e-3), 0.1)
        self._drag_t = now
        targets = drag.update(q, qd, dt, now, limits=self._drag_limits)
        for fw, t in targets.items():
            mj = f"{'Left' if fw[0] == 'l' else 'Right'} Arm {fw[-1]}"
            try:
                self.data.actuator(mj).ctrl[0] = float(t)
            except Exception:
                pass

    def shutdown(self):
        self._stop = True
        try:
            self._node.destroy_node()
        except Exception:
            pass


class SimRosPlant(MuJoCoPlant):
    """Linux sim plant: own MuJoCo physics EXACTLY like MuJoCoPlant (same controller,
    motion model, WBC torque engine, mj_step pacing as the Mac) AND publish the sim
    state on the canonical ROS topics so the sensor renderer / SLAM / Nav2 / RViz
    compose on top unchanged.

    This is the 'bridge owns physics on Linux' path (MABEL_DISPATCH_TARGET=sim:mujoco+ros):
    the bridge becomes the physics + /clock authority, so mabel_sim/mujoco_node's PHYSICS
    node is NOT launched (sim.launch.py physics:=false) -- only its camera/LiDAR renderer,
    swerve_odometry, robot_state_publisher and RViz run, all fed by what we publish here.
    Because physics + control are inherited verbatim from MuJoCoPlant, the ROS sim is
    byte-for-byte the Mac sim; this class only ADDS state publishing.

    Published (mirroring mabel_sim/mujoco_node so consumers can't tell the difference):
      /clock                 sim-time authority (single publisher)
      /joint_states          every hinge/slide joint, URDF-named, qpos-qpos0 + qvel
      /odom_ground_truth     TRUE base pose (the sim renderer places cameras with it)
      /odom (+ odom->base_link TF)   ONLY if MABEL_SIM_PUBLISH_ODOM=1 (else swerve_odometry
                             owns /odom from /joint_states, the default -- Nav unchanged)
      /camera/front/imu (+ static TF)   front D435i IMU for cuVSLAM, if the scene has it

    The heavy rclpy + mabel_sim imports are LAZY (only this target imports them), so the
    Mac sim:mujoco path never touches ROS."""

    def __init__(self, model, data, state=None, node_name="mabel_sim_gateway"):
        super().__init__(model, data, state)
        import os
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState, Imu
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped
        from rosgraph_msgs.msg import Clock
        from builtin_interfaces.msg import Time as TimeMsg
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
        from mabel_sim import interface_config as cfg

        self._np = np
        self._cfg = cfg
        self._JointState, self._Imu, self._Odometry = JointState, Imu, Odometry
        self._TransformStamped, self._Clock, self._TimeMsg = TransformStamped, Clock, TimeMsg
        self._own_odom = os.getenv("MABEL_SIM_PUBLISH_ODOM", "0") == "1"

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node = Node(node_name)

        # --- index caches (same convention as mujoco_node._build_indices) ---
        # /joint_states: every hinge/slide joint, URDF-named -> (name, qadr, vadr).
        self._global_joints = []
        for j in range(model.njnt):
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                         mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            self._global_joints.append(
                (cfg.urdfize(name), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])))

        # base free joint (odom ground truth)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg.BASE_FREE_JOINT)
        if bid < 0:
            self._node.get_logger().warn(
                f"no free joint '{cfg.BASE_FREE_JOINT}'; odom disabled")
            self._base_qadr = self._base_vadr = None
        else:
            self._base_qadr = int(model.jnt_qposadr[bid])
            self._base_vadr = int(model.jnt_dofadr[bid])

        # --- publishers ---
        self.pub_clock = self._node.create_publisher(Clock, "/clock", 10)
        self.pub_js = self._node.create_publisher(JointState, "/joint_states", 10)
        self.pub_odom_gt = self._node.create_publisher(Odometry, "/odom_ground_truth", 10)
        self.pub_odom = (self._node.create_publisher(Odometry, cfg.ODOM_TOPIC, 10)
                         if self._own_odom else None)
        self._tf = TransformBroadcaster(self._node)

        # front IMU (best-effort; degrades to no IMU on older scenes)
        self._build_imu(qos_profile_sensor_data, StaticTransformBroadcaster)

        # per-topic sim-time rate gates
        self._t_clock = self._t_js = self._t_odom = self._t_imu = -1e9
        self._node.get_logger().info(
            "SimRosPlant: bridge owns MuJoCo physics; publishing /clock + /joint_states "
            "+ /odom_ground_truth"
            + ("" if self.pub_odom is None else " + /odom (+TF)")
            + ("" if self.pub_imu is None else f" + {cfg.IMU_TOPIC}"))

        self._setup_nav()

    # ── Nav2 bridge: forward web goals to Nav2 + consume Nav2's base command ────
    # The bridge OWNS the base in sim:mujoco+ros, so it does NOT publish /mabel_cmd
    # — Nav2 (cmd_vel_bridge) is the ONLY publisher, so we can safely subscribe to
    # it and feed it into the controller when a goal is active. Goals go OUT on
    # /goal_pose (NavFn A* + MPPI plan/control). A dedicated node + spin thread so
    # the subscription is fresh independent of the (slow) physics loop.
    def _setup_nav(self):
        import threading
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from mabel_interfaces.msg import MabelSwerveCmd
        self._PoseStamped = PoseStamped
        self._nav_node = Node("mabel_nav_bridge")
        self._pub_goal = self._nav_node.create_publisher(PoseStamped, "/goal_pose", 10)
        # STAGED goal preview — published on tap so the operator's RViz shows where
        # the web/app user aimed BEFORE they commit to drive. A separate topic from
        # /goal_pose so it does NOT trigger Nav2; hardware.rviz shows it (orange).
        self._pub_goal_preview = self._nav_node.create_publisher(
            PoseStamped, "/goal_pose_preview", 10)
        # Green ball at the goal the browser/app placed → RViz Marker, so the
        # operator's RViz stays in sync with the web teleop goal. Transient-local
        # so a late-joining RViz still shows the current goal.
        from visualization_msgs.msg import Marker
        from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                               QoSDurabilityPolicy)
        self._Marker = Marker
        gm_qos = QoSProfile(depth=1)
        gm_qos.reliability = QoSReliabilityPolicy.RELIABLE
        gm_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub_goal_marker = self._nav_node.create_publisher(
            Marker, "/mabel/goal_marker", gm_qos)
        self._nav_twist = (0.0, 0.0, 0.0)
        self._nav_cmd_t = -1e9
        # Nav2's twist arrives on the NON-canonical /nav/mabel_cmd (cmd_vel_bridge's
        # `cmd_out_topic`), never /mabel_cmd — the plant is /mabel_cmd's only
        # publisher. Identical to ROSPlant on the real robot, so nav behaves the
        # same in sim and real; keep the two topics in lockstep.
        import os as _os
        _nav_topic = _os.getenv("MABEL_NAV_CMD_TOPIC", "/nav/mabel_cmd")
        self._nav_node.create_subscription(MabelSwerveCmd, _nav_topic, self._on_nav_cmd, 10)
        threading.Thread(target=self._nav_spin, name="nav-bridge-spin", daemon=True).start()
        self._node.get_logger().info(
            f"Nav2 bridge: goals → /goal_pose ; consuming Nav2 {_nav_topic} for the base")

    def _nav_spin(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        # Persistent executor for the same reason as _spin (waitset reuse).
        ex = SingleThreadedExecutor()
        ex.add_node(self._nav_node)
        while rclpy.ok():
            try:
                ex.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _on_nav_cmd(self, msg):
        import time as _t
        self._nav_twist = (float(msg.vx), float(msg.vy), float(msg.omega))
        self._nav_cmd_t = _t.monotonic()

    def send_nav_goal(self, x, y, yaw=0.0):
        import math, os
        g = self._PoseStamped()
        g.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "odom")
        g.header.stamp = self._nav_node.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.z = math.sin(float(yaw) / 2.0)
        g.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self._pub_goal.publish(g)
        self._publish_goal_marker(x, y)        # green ball in RViz (synced w/ web)

    def send_nav_goal_preview(self, x, y, yaw=0.0):
        """Publish a STAGED goal to /goal_pose_preview (RViz only; does NOT drive)."""
        import math, os
        g = self._PoseStamped()
        g.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "odom")
        g.header.stamp = self._nav_node.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.z = math.sin(float(yaw) / 2.0)
        g.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self._pub_goal_preview.publish(g)
        self._publish_goal_marker(x, y)        # green ball follows the staged goal

    def _publish_goal_marker(self, x, y, show=True):
        """Green SPHERE Marker at the goal (/mabel/goal_marker) so the operator's
        RViz mirrors the goal the browser/app placed. show=False removes it."""
        import os
        m = self._Marker()
        m.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "odom")
        m.header.stamp = self._nav_node.get_clock().now().to_msg()
        m.ns = "mabel_goal"
        m.id = 0
        m.type = self._Marker.SPHERE
        m.action = self._Marker.ADD if show else self._Marker.DELETE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.30
        m.color.r, m.color.g, m.color.b, m.color.a = 0.10, 0.90, 0.25, 0.95
        self._pub_goal_marker.publish(m)

    def clear_goal_marker(self):
        """Remove the green goal ball (goal cancelled / cleared)."""
        self._publish_goal_marker(0.0, 0.0, show=False)

    def nav_twist(self):
        """(vx, vy, omega, age_s) — Nav2's latest base command + its staleness."""
        import time as _t
        vx, vy, w = self._nav_twist
        return vx, vy, w, (_t.monotonic() - self._nav_cmd_t)

    # ── helpers (ported verbatim from mujoco_node so the output is identical) ──
    @staticmethod
    def _split_time(t):
        sec = int(t)
        nsec = int(round((t - sec) * 1e9))
        if nsec >= 1_000_000_000:
            sec += 1
            nsec -= 1_000_000_000
        return sec, nsec

    def _sim_stamp(self):
        sec, nsec = self._split_time(self.data.time)
        return self._TimeMsg(sec=sec, nanosec=nsec)

    def _build_imu(self, sensor_qos, StaticTransformBroadcaster):
        cfg, m, np = self._cfg, self.model, self._np
        self.pub_imu = None
        self._imu_ok = False

        def sensor(name):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return (int(m.sensor_adr[sid]), int(m.sensor_dim[sid])) if sid >= 0 else None

        acc = sensor(cfg.IMU_ACC_SENSOR)
        gyro = sensor(cfg.IMU_GYRO_SENSOR)
        quat = sensor(cfg.IMU_QUAT_SENSOR)
        if acc is None or gyro is None:
            self._node.get_logger().info(
                "front IMU sensors absent from scene -> IMU not published (visual SLAM "
                "still runs without it).")
            return
        self._imu_acc_adr, self._imu_gyro_adr = acc[0], gyro[0]
        self._imu_quat_adr = quat[0] if quat else None

        # rigid base_link(=mabel_chassis) -> front_imu transform, latched static.
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg.IMU_SITE)
        cbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
        R_wb = self.data.xmat[cbid].reshape(3, 3)
        R_ws = self.data.site_xmat[sid].reshape(3, 3)
        R_bs = R_wb.T @ R_ws
        t_bs = R_wb.T @ (self.data.site_xpos[sid] - self.data.xpos[cbid])
        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, R_bs.flatten())
        self._imu_tf_t = t_bs
        self._imu_tf_q = (float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]),
                          float(q_wxyz[0]))      # xyzw
        # BMI055 noise model (white noise + fixed bias + slow random walk).
        self._imu_rng = np.random.default_rng()
        self._imu_acc_noise, self._imu_gyro_noise = cfg.IMU_ACC_NOISE, cfg.IMU_GYRO_NOISE
        self._imu_acc_walk, self._imu_gyro_walk = cfg.IMU_ACC_WALK, cfg.IMU_GYRO_WALK
        self._imu_acc_bias = self._imu_rng.normal(0.0, cfg.IMU_ACC_BIAS, 3)
        self._imu_gyro_bias = self._imu_rng.normal(0.0, cfg.IMU_GYRO_BIAS, 3)
        self._imu_rate = float(cfg.RATE_IMU)

        self.pub_imu = self._node.create_publisher(self._Imu, cfg.IMU_TOPIC, sensor_qos)
        self._static_tf = StaticTransformBroadcaster(self._node)
        tf = self._TransformStamped()
        tf.header.stamp = self._sim_stamp()
        tf.header.frame_id = cfg.BASE_FRAME
        tf.child_frame_id = cfg.IMU_FRAME
        tf.transform.translation.x = float(t_bs[0])
        tf.transform.translation.y = float(t_bs[1])
        tf.transform.translation.z = float(t_bs[2])
        qx, qy, qz, qw = self._imu_tf_q
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self._static_tf.sendTransform(tf)
        self._imu_ok = True

    # ── per-iteration publish (called from the bridge's outer control loop) ───
    def publish_ros_state(self):
        """Publish /clock + the sim state on the canonical topics. Throttled per topic
        on SIM time (data.time) so consumers running use_sim_time integrate correctly
        even at RTF<1. data.time only advances when physics steps, so a stalled sim
        simply stops time -- exactly what every consumer expects."""
        if not self._rclpy.ok():
            return
        t = float(self.data.time)
        cfg = self._cfg
        try:
            if t - self._t_clock >= 0.01 or t < self._t_clock:
                self._t_clock = t
                ck = self._Clock()
                ck.clock.sec, ck.clock.nanosec = self._split_time(t)
                self.pub_clock.publish(ck)
            if t - self._t_js >= 1.0 / cfg.RATE_JOINT_STATES or t < self._t_js:
                self._t_js = t
                self._publish_joint_states()
            if self._base_qadr is not None and (t - self._t_odom >= 1.0 / cfg.RATE_ODOM
                                                 or t < self._t_odom):
                self._t_odom = t
                self._publish_odom()
            if self.pub_imu is not None and (t - self._t_imu >= 1.0 / self._imu_rate
                                             or t < self._t_imu):
                self._t_imu = t
                self._publish_imu()
        except Exception:
            # A publish during interpreter/rclpy teardown must never crash the loop.
            pass

    def _publish_joint_states(self):
        q, q0, qd = self.data.qpos, self.model.qpos0, self.data.qvel
        js = self._JointState()
        js.header.stamp = self._sim_stamp()
        for name, qa, va in self._global_joints:
            js.name.append(name)
            # URDF q=0 is the assembled (ref) pose: publish (qpos - qpos0), else joints
            # whose ref != 0 (wrists/thumb/abductions) slam to their limits in RViz.
            js.position.append(float(q[qa] - q0[qa]))
            js.velocity.append(float(qd[va]))
        self.pub_js.publish(js)

    def _publish_odom(self):
        np, cfg = self._np, self._cfg
        q, qd = self.data.qpos, self.data.qvel
        qa, va = self._base_qadr, self._base_vadr
        px, py, pz = q[qa], q[qa + 1], q[qa + 2]
        qw, qx, qy, qz = q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]
        now = self._sim_stamp()

        odom = self._Odometry()
        odom.header.stamp = now
        odom.header.frame_id = cfg.ODOM_FRAME
        odom.child_frame_id = cfg.BASE_FRAME
        odom.pose.pose.position.x = float(px)
        odom.pose.pose.position.y = float(py)
        odom.pose.pose.position.z = float(pz)
        odom.pose.pose.orientation.w = float(qw)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        # MuJoCo free-joint qvel: linear part is WORLD frame, angular is BODY frame.
        # REP-105 + every consumer (Nav cmd_vel_bridge, EKF, swerve_odometry) want the
        # twist in child_frame_id (base_link); rotate the linear part world->body.
        v_world = np.array([qd[va], qd[va + 1], qd[va + 2]], dtype=np.float64)
        quat = np.array([qw, qx, qy, qz], dtype=np.float64)
        quat_inv = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(quat_inv, quat)
        v_body = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(v_body, v_world, quat_inv)
        odom.twist.twist.linear.x = float(v_body[0])
        odom.twist.twist.linear.y = float(v_body[1])
        odom.twist.twist.linear.z = float(v_body[2])
        odom.twist.twist.angular.x = float(qd[va + 3])
        odom.twist.twist.angular.y = float(qd[va + 4])
        odom.twist.twist.angular.z = float(qd[va + 5])
        self.pub_odom_gt.publish(odom)          # always: TRUE pose for the sim renderer

        if self.pub_odom is None:
            return                              # swerve_odometry owns /odom + the TF
        self.pub_odom.publish(odom)
        tf = self._TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = cfg.ODOM_FRAME
        tf.child_frame_id = cfg.BASE_FRAME
        tf.transform.translation.x = float(px)
        tf.transform.translation.y = float(py)
        tf.transform.translation.z = float(pz)
        tf.transform.rotation.w = float(qw)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        self._tf.sendTransform(tf)

    def _publish_imu(self):
        np, cfg = self._np, self._cfg
        sd = self.data.sensordata
        acc = np.array(sd[self._imu_acc_adr:self._imu_acc_adr + 3], dtype=np.float64)
        gyro = np.array(sd[self._imu_gyro_adr:self._imu_gyro_adr + 3], dtype=np.float64)
        quat_wxyz = (np.array(sd[self._imu_quat_adr:self._imu_quat_adr + 4], dtype=np.float64)
                     if self._imu_quat_adr is not None else None)
        self._imu_acc_bias += self._imu_rng.normal(0.0, self._imu_acc_walk, 3)
        self._imu_gyro_bias += self._imu_rng.normal(0.0, self._imu_gyro_walk, 3)
        acc += self._imu_acc_bias + self._imu_rng.normal(0.0, self._imu_acc_noise, 3)
        gyro += self._imu_gyro_bias + self._imu_rng.normal(0.0, self._imu_gyro_noise, 3)

        msg = self._Imu()
        msg.header.stamp = self._sim_stamp()
        msg.header.frame_id = cfg.IMU_FRAME
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = \
            float(acc[0]), float(acc[1]), float(acc[2])
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = \
            float(gyro[0]), float(gyro[1]), float(gyro[2])
        if quat_wxyz is not None:
            msg.orientation.w = float(quat_wxyz[0])
            msg.orientation.x = float(quat_wxyz[1])
            msg.orientation.y = float(quat_wxyz[2])
            msg.orientation.z = float(quat_wxyz[3])
        else:
            msg.orientation.w = 1.0
        av, lv = self._imu_gyro_noise ** 2, self._imu_acc_noise ** 2
        msg.angular_velocity_covariance = [av, 0.0, 0.0, 0.0, av, 0.0, 0.0, 0.0, av]
        msg.linear_acceleration_covariance = [lv, 0.0, 0.0, 0.0, lv, 0.0, 0.0, 0.0, lv]
        msg.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_imu.publish(msg)

    def shutdown(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


def make_plant(dispatch_target, model, data, state):
    """Build the plant for MABEL_DISPATCH_TARGET:
      'real:*'        -> ROSPlant       (real robot; mirror encoders, publish the contract)
      'sim:bridge'    -> SimBridgePlant (the SIMULATION BRIDGE owns physics — the sim twin
                                         of the real path: mirror its state, command it
                                         over RPC; simulation_bridge/ is the HAL)
      'sim:*+ros'     -> SimRosPlant    (Linux sim: own physics like the Mac AND publish
                                         the canonical ROS state for SLAM/Nav/RViz/sensors)
      anything else   -> MuJoCoPlant    (legacy 'sim:mujoco': in-process physics, no bridge)."""
    t = str(dispatch_target)
    if t.startswith("real"):
        return ROSPlant(model, data, state)
    if "bridge" in t:
        return SimBridgePlant(model, data, state)
    if "ros" in t:
        return SimRosPlant(model, data, state)
    return MuJoCoPlant(model, data, state)
