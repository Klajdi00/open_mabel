#!/usr/bin/env python3
"""MuJoCo <-> ROS 2 bridge for MABEL  (drop-in replacement for the hardware).

Threading model (this is the important part)
--------------------------------------------
Three groups of threads run concurrently and never block each other:

  1. PHYSICS THREAD  (`_physics_loop`)
        The ONLY thread that touches `self.data`.  It runs the passive MuJoCo
        viewer and steps the model in real time.  Each step it (a) copies the
        latest commands out of `_cmd` under `_cmd_lock`, (b) applies controls
        and calls `mj_step`, then (c) copies the resulting state into snapshot
        buffers under `_state_lock`.  It never waits on ROS.

  2. ROS SUBSCRIBER CALLBACKS  (executor threads)
        Write targets into `_cmd` under `_cmd_lock` and return immediately.

  3. ROS PUBLISHER TIMERS  (executor threads)
        Read the snapshot buffers under `_state_lock` and publish.  They never
        reach into the live `self.data`, so a slow publish can never stall the
        physics step (and vice-versa).

Locks are held only for tiny array copies, so there is no MuJoCo->ROS latency.
A MultiThreadedExecutor lets subscribers and publishers run in parallel.

The published / subscribed topics are defined in `interface_config.py` and are
identical to what the real hardware nodes use, so switching between this sim and
`mabel_bringup hardware.launch.py` is transparent to every downstream node.
"""

import os
import math
import threading
import time

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Float64MultiArray, MultiArrayDimension, MultiArrayLayout, String
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as TimeMsg
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from mabel_interfaces.msg import MabelSwerveCmd

from mabel_sim import interface_config as cfg

# ── Optional whole-body COMPLIANCE engine (controller/mabel) ──────────────────
# The "compliance" control mode runs the SAME whole-body law the iOS / Vision Pro
# apps and the standalone scripts/compliance.py drive: arms + torso are gravity-
# compensated and back-drivable, and the mobile base admits the operator's drag
# wrench and glides. It is loaded here lazily via a synthetic `mabel` package (the
# same trick keyboard_teleop uses) so we pull only the light controller modules,
# bypassing the package's heavy AVP/zmq __init__. MuJoCo's data.xfrc_applied -- set
# by the viewer's Ctrl+drag -- IS the operator push the law reads.
try:
    import sys as _sys
    import types as _types
    _CTRL = os.path.expanduser(os.environ.get("MABEL_CONTROLLER", "~/open_mabel/controller"))
    if "mabel" not in _sys.modules:
        _pkg = _types.ModuleType("mabel")
        _pkg.__path__ = [os.path.join(_CTRL, "mabel")]
        _sys.modules["mabel"] = _pkg
    from mabel.wbc_runtime import WBCEngine as _WBCEngine
    _COMPLIANCE_OK = True
except Exception as _e:                       # pragma: no cover
    _COMPLIANCE_OK = False
    _COMPLIANCE_ERR = _e

# Vertical-reach admittance for the lift while in compliance mode: a vertical drag
# on the arms beyond a deadband glides the lift (so pulling a hand up/down past the
# arms' comfortable reach raises/lowers the whole torso), mirroring how the base
# admits the planar drag. Capped at the lift's hardware cruise speed.
LIFT_ADMIT_DEADBAND = 6.0     # N;  vertical drag below this leaves the lift parked
LIFT_ADMIT_DAMP     = 180.0   # N/(m/s);  virtual viscous friction (v_ss = Fz/this)
LIFT_ADMIT_VMAX     = 0.06    # m/s; hardware lift cruise cap (firmware MAX_VELOCITY)


class MujocoBridge(Node):

    def __init__(self):
        super().__init__("mabel_mujoco_sim")

        # ---- parameters ----------------------------------------------------
        self.declare_parameter("scene", "")          # "" => package default
        self.declare_parameter("wheel_radius", cfg.WHEEL_RADIUS)
        self.declare_parameter("realtime", True)
        # When false, this node yields the /odom topic + odom->base_link TF to the
        # swerve_odometry (wheel-odom) node and instead publishes ground truth on
        # /odom_ground_truth for evaluation. Used when running visual SLAM.
        self.declare_parameter("publish_odom", True)
        # Run the physics loop WITHOUT the on-screen GLFW viewer (for headless
        # boxes / automated testing / running on a robot with no display).
        self.declare_parameter("headless", False)
        # Firmly hold the hands/wrists at their commanded pose. The canonical MJCF
        # tunes the finger actuators very soft (kp~1, ~0 force limit) for compliant
        # GRASPING -- great for the Vision Pro teleop, but it means the hands sag /
        # ragdoll under gravity whenever nothing commands them (e.g. keyboard nav).
        # This raises only the SIM's hold gains so the upper body stays put while you
        # drive; the canonical model (and the server's grasp compliance) is untouched.
        self.declare_parameter("hold_hands", True)
        self.declare_parameter("hand_hold_kp", 40.0)
        self.declare_parameter("hand_hold_kv", 4.0)
        self.declare_parameter("hand_hold_force", 12.0)
        # Gravity compensation: feed-forward the upper-body gravity (joint-space bias)
        # onto the arm/torso/lift/hand DOFs every step so the position servos hold the
        # COMMANDED pose with no steady-state droop. Without this, a finite-gain PD
        # servo sags under gravity, so the arms visibly DROP when nothing is driving
        # them (e.g. switching from nav Mode 1 into wrist-tracking Mode 2 captures the
        # sagged pose). With it the held pose stays put. The base free joint + wheels
        # are deliberately excluded (they must keep real gravity for ground contact).
        self.declare_parameter("gravity_comp", True)
        # ---- front D435i IMU (sensor_msgs/Imu on /camera/front/imu) ----------
        # Publishes the MuJoCo accelerometer/gyro on the `front_imu` site with a
        # simulated BMI055 noise+bias model. Enables cuVSLAM IMU fusion; harmless
        # if unused. Disable with imu:=false. Noise params default to config.py.
        self.declare_parameter("imu", True)
        self.declare_parameter("imu_rate", cfg.RATE_IMU)
        self.declare_parameter("imu_accel_noise", cfg.IMU_ACC_NOISE)
        self.declare_parameter("imu_gyro_noise", cfg.IMU_GYRO_NOISE)
        self.declare_parameter("imu_accel_bias", cfg.IMU_ACC_BIAS)
        self.declare_parameter("imu_gyro_bias", cfg.IMU_GYRO_BIAS)
        self.declare_parameter("imu_accel_walk", cfg.IMU_ACC_WALK)
        self.declare_parameter("imu_gyro_walk", cfg.IMU_GYRO_WALK)
        scene_path = self.get_parameter("scene").value or cfg.DEFAULT_SCENE
        scene_path = os.path.expanduser(scene_path)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.realtime = bool(self.get_parameter("realtime").value)
        self.publish_odom = bool(self.get_parameter("publish_odom").value)
        self.headless = bool(self.get_parameter("headless").value)

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")
        self.get_logger().info(f"Loading MuJoCo scene: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # ---- shared state (guarded) ---------------------------------------
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # base command (body-frame twist + lift target + e-stop)
        self._cmd = {"vx": 0.0, "vy": 0.0, "omega": 0.0,
                     "lift": 0.0, "estop": False}
        # position-actuator targets {actuator_id: ctrl_value} for arms/body/hands
        self._pos_targets = {}

        # whole-body compliance ("drag the robot") -- engaged from /mabel/control_mode.
        # The engine + its stepping live ENTIRELY on the physics thread (the only one
        # allowed to touch self.data); callbacks just request a mode under _cmd_lock.
        self._req_ctrl_mode = "position"      # "position" | "compliance"
        self._req_reset = False               # set by /mabel/reset_sim, served on the physics thread
        self._wbc = None                      # WBCEngine while engaged, else None
        self._compliance_failed = False       # stop retrying if the engine won't load
        self._arm_bids = []                   # arm body ids for the vertical-lift wrench
        self._lift_target = 0.0               # lift ctrl target driven by the admittance

        # snapshot buffers written by physics thread, read by publishers
        self._snap_qpos = self.data.qpos.copy()
        self._snap_qvel = self.data.qvel.copy()
        self._snap_sensordata = self.data.sensordata.copy()
        self._snap_time = 0.0

        # ---- build index caches from the model ----------------------------
        self._build_indices()
        self._build_imu()
        self._stiffen_hand_hold()
        self._build_gravcomp()

        # seed lift + all position targets to the spawn pose (no snap on boot)
        with self._cmd_lock:
            self._cmd["lift"] = self._cascaded_lift()
            for aid, (jqadr, _, lo, hi) in self._pos_actuators.items():
                self._pos_targets[aid] = float(np.clip(self.data.qpos[jqadr], lo, hi))

        # ---- ROS interfaces -----------------------------------------------
        grp = ReentrantCallbackGroup()
        self._setup_ros(grp)

        # ---- launch physics thread ----------------------------------------
        self.running = True
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()
        self.get_logger().info(
            "MuJoCo bridge online: base, lift, arms, body, hands -> ROS")

    def _build_imu(self):
        """Resolve the front-IMU MuJoCo sensors + the rigid base_link->IMU transform,
        and seed the noise model. Degrades gracefully (imu disabled) if the scene
        predates the front_imu sensors, so old scenes still load."""
        self._imu_ok = False
        if not bool(self.get_parameter("imu").value):
            self.get_logger().info("front IMU disabled (imu:=false)")
            return
        m = self.model

        def sensor(name):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                return None
            return int(m.sensor_adr[sid]), int(m.sensor_dim[sid])

        acc = sensor(cfg.IMU_ACC_SENSOR)
        gyro = sensor(cfg.IMU_GYRO_SENSOR)
        quat = sensor(cfg.IMU_QUAT_SENSOR)
        if acc is None or gyro is None:
            self.get_logger().warn(
                f"front IMU sensors ('{cfg.IMU_ACC_SENSOR}'/'{cfg.IMU_GYRO_SENSOR}') "
                "absent from the scene -> IMU not published. Re-export the MJCF "
                "(it now defines the front_imu site + sensors).")
            return
        self._imu_acc_adr, _ = acc
        self._imu_gyro_adr, _ = gyro
        self._imu_quat_adr = quat[0] if quat else None

        # Rigid base_link(=mabel_chassis) -> front_imu_optical_frame transform.
        # Constant (IMU is bolted to the chassis), so publish it once as a STATIC
        # TF -- this is the IMU->base extrinsic cuVSLAM needs for fusion.
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, cfg.IMU_SITE)
        cbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
        R_wb = self.data.xmat[cbid].reshape(3, 3)
        R_ws = self.data.site_xmat[sid].reshape(3, 3)
        R_bs = R_wb.T @ R_ws
        t_bs = R_wb.T @ (self.data.site_xpos[sid] - self.data.xpos[cbid])
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, R_bs.flatten())
        self._imu_tf_t = t_bs
        self._imu_tf_q = (float(quat_wxyz[1]), float(quat_wxyz[2]),
                          float(quat_wxyz[3]), float(quat_wxyz[0]))  # xyzw

        # Noise model: white noise (per-sample std) + a fixed per-axis bias drawn
        # once + a slow random walk. Mirrors the BMI055 the D435i carries.
        self._imu_rng = np.random.default_rng()
        self._imu_acc_noise = float(self.get_parameter("imu_accel_noise").value)
        self._imu_gyro_noise = float(self.get_parameter("imu_gyro_noise").value)
        self._imu_acc_walk = float(self.get_parameter("imu_accel_walk").value)
        self._imu_gyro_walk = float(self.get_parameter("imu_gyro_walk").value)
        acc_b = float(self.get_parameter("imu_accel_bias").value)
        gyro_b = float(self.get_parameter("imu_gyro_bias").value)
        self._imu_acc_bias = self._imu_rng.normal(0.0, acc_b, 3)
        self._imu_gyro_bias = self._imu_rng.normal(0.0, gyro_b, 3)
        self._imu_ok = True

    def _stiffen_hand_hold(self):
        """Raise the SIM hold gains on the hand+wrist position actuators so the upper
        body stays put when nothing is driving it (the canonical MJCF tunes fingers very
        soft for grasp compliance, which makes them ragdoll under gravity in nav mode).
        Gains-only, applied to the loaded MjModel -- the canonical file is untouched."""
        if not bool(self.get_parameter("hold_hands").value):
            return
        kp = float(self.get_parameter("hand_hold_kp").value)
        kv = float(self.get_parameter("hand_hold_kv").value)
        fr = float(self.get_parameter("hand_hold_force").value)
        aids = set()
        for side in cfg.HAND_SIDES:
            aids.update(self._hand_cmd_map[side].values())
        for aid in aids:
            self.model.actuator_gainprm[aid][0] = kp          # position gain
            self.model.actuator_biasprm[aid][1] = -kp         # -kp (position error)
            self.model.actuator_biasprm[aid][2] = -kv         # -kv (damping)
            self.model.actuator_forcelimited[aid] = 1
            self.model.actuator_forcerange[aid] = (-fr, fr)
        self.get_logger().info(
            f"hand/wrist hold stiffened: kp={kp} kv={kv} force=+-{fr} N*m "
            f"({len(aids)} actuators) -> hands hold firm in nav (hold_hands:=false to disable)")

    def _build_gravcomp(self):
        """Collect the upper-body joint DOFs to gravity-compensate: every position-
        actuated arm/torso/neck/hand/wrist joint plus the lift. The base free joint
        and the swerve/drive joints are excluded so the chassis keeps real gravity."""
        self._gravcomp = bool(self.get_parameter("gravity_comp").value)
        dofs = set()
        if self._gravcomp:
            for _aid, (_qadr, vadr, _lo, _hi) in self._pos_actuators.items():
                dofs.add(int(vadr))
            lj = self._adr(cfg.LIFT_JOINT) if hasattr(cfg, "LIFT_JOINT") else self._adr("Lift Lower")
            if lj is not None:
                dofs.add(int(lj[1]))
        self._gravcomp_dofs = np.array(sorted(dofs), dtype=int) if dofs else None
        if self._gravcomp_dofs is not None:
            self.get_logger().info(
                f"gravity compensation ON for {len(self._gravcomp_dofs)} upper-body DOFs "
                "-> servos hold the commanded pose with no droop (gravity_comp:=false to disable)")
        else:
            self.get_logger().info("gravity compensation OFF (gravity_comp:=false)")

    # ------------------------------------------------------------------ setup
    def _adr(self, joint_name):
        """(qposadr, qveladr, ctrl_lo, ctrl_hi-or-None) for a named joint's actuator,
        or None if the joint is absent from the model."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        return self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]

    def _act(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _ctrl_range(self, aid):
        if self.model.actuator_ctrllimited[aid]:
            lo, hi = self.model.actuator_ctrlrange[aid]
            return float(lo), float(hi)
        return -math.inf, math.inf

    def _build_indices(self):
        m = self.model

        # --- base free joint (odom) ---
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cfg.BASE_FREE_JOINT)
        if bid < 0:
            self.get_logger().warn(f"no free joint '{cfg.BASE_FREE_JOINT}'; odom disabled")
            self._base_qadr = self._base_vadr = None
        else:
            self._base_qadr = m.jnt_qposadr[bid]
            self._base_vadr = m.jnt_dofadr[bid]

        # --- swerve modules ---
        self._swerve = {}
        for key, mod in cfg.SWERVE_MODULES.items():
            steer_a = self._act(mod["steer"])
            drive_a = self._act(mod["drive"])
            steer_q = self._adr(mod["steer"])     # steer joint shares actuator name
            if steer_a < 0 or drive_a < 0 or steer_q is None:
                self.get_logger().warn(f"swerve module '{key}' missing in model")
                continue
            self._swerve[key] = {"steer_a": steer_a, "drive_a": drive_a,
                                 "steer_q": steer_q[0], "pos": np.array(mod["pos"]),
                                 "yaw": mod["yaw"]}

        # --- lift ---
        lift_a = self._act(cfg.LIFT_ACTUATOR)
        self._lift_a = lift_a if lift_a >= 0 else None
        # report lift position from the lower slide joint if present
        lj = self._adr("Lift Lower")
        self._lift_qadr = lj[0] if lj else None
        # The "Lift" actuator servos the CASCADED tendon length (= Lift Lower +
        # Lift Upper, ctrlrange [0, 0.635]); a single slide joint is only HALF of
        # that. Seeding/holding the lift from a single joint's qpos therefore
        # under-commands the actuator ~2x and DROPS the lift to half height (most
        # visible on the position<->compliance handover). Cache the upper slide too
        # so _cascaded_lift() can read the true height the actuator expects.
        uj = self._adr("Lift Upper")
        self._lift_upper_qadr = uj[0] if uj else None

        # --- position actuators for arms / body / hands ---
        # {actuator_id: (joint_qposadr, joint_qveladr, ctrl_lo, ctrl_hi)}
        self._pos_actuators = {}
        # {ros_command_name: actuator_id}  (accepts URDF + MuJoCo names)
        self._arms_cmd_map = {}
        self._body_cmd_map = {}
        # per side: {orca_name: actuator_id}, and publish list [(full_name, qadr, vadr)]
        self._hand_cmd_map = {s: {} for s in cfg.HAND_SIDES}
        self._hand_pub = {s: [] for s in cfg.HAND_SIDES}
        # per subsystem publish list [(ros_name, qadr, vadr)]
        self._arms_pub = []
        self._body_pub = []

        def register(mj_name, cmd_names, cmd_map, pub_list, ros_name):
            aid = self._act(mj_name)
            adr = self._adr(mj_name)
            if aid < 0 or adr is None:
                self.get_logger().warn(f"actuator/joint '{mj_name}' missing in model")
                return
            qadr, vadr = adr
            lo, hi = self._ctrl_range(aid)
            self._pos_actuators[aid] = (qadr, vadr, lo, hi)
            for cn in cmd_names:
                cmd_map[cn] = aid
            pub_list.append((ros_name, qadr, vadr))

        for mj_name in cfg.ARM_ACTUATORS:
            register(mj_name, {cfg.urdfize(mj_name), mj_name},
                     self._arms_cmd_map, self._arms_pub, cfg.urdfize(mj_name))

        for mj_name in cfg.BODY_ACTUATORS:
            register(mj_name, {cfg.urdfize(mj_name), mj_name},
                     self._body_cmd_map, self._body_pub, cfg.urdfize(mj_name))

        for side in cfg.HAND_SIDES:
            for bare, mj_name in cfg.hand_joint_map(side).items():
                full = f"{side}_{bare}"            # orca-style published name
                register(mj_name, {bare, full, cfg.urdfize(mj_name)},
                         self._hand_cmd_map[side], self._hand_pub[side], full)

        # --- global /joint_states: every hinge/slide joint, URDF-named ---
        self._global_joints = []   # [(urdf_name, qadr, vadr)]
        for j in range(m.njnt):
            jtype = m.jnt_type[j]
            if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            self._global_joints.append(
                (cfg.urdfize(name), m.jnt_qposadr[j], m.jnt_dofadr[j]))

        self.get_logger().info(
            f"indexed: {len(self._swerve)} swerve, "
            f"{len(self._arms_pub)} arm, {len(self._body_pub)} body, "
            f"{sum(len(v) for v in self._hand_pub.values())} hand joints, "
            f"{len(self._global_joints)} total joints")

    def _setup_ros(self, grp):
        # subscribers
        self.create_subscription(MabelSwerveCmd, cfg.CMD_BASE_TOPIC,
                                 self._on_base_cmd, 10, callback_group=grp)
        self.create_subscription(JointState, cfg.CMD_ARMS_TOPIC,
                                 lambda m: self._on_joint_cmd(m, self._arms_cmd_map),
                                 10, callback_group=grp)
        self.create_subscription(JointState, cfg.CMD_BODY_TOPIC,
                                 lambda m: self._on_joint_cmd(m, self._body_cmd_map),
                                 10, callback_group=grp)
        for side in cfg.HAND_SIDES:
            self.create_subscription(
                JointState, f"/{side}_hand/command",
                lambda m, s=side: self._on_joint_cmd(m, self._hand_cmd_map[s]),
                10, callback_group=grp)
        # Whole-body control mode (keyboard_teleop Mode 3 / the apps): "position"
        # (default, normal servo teleop) or "compliance" (drag-the-robot).
        self.create_subscription(String, "/mabel/control_mode",
                                 self._on_control_mode, 10, callback_group=grp)
        # Reset the simulation to its home keyframe (iOS / Vision Pro Reset button,
        # relayed by the teleop bridge over ROS). The physics thread owns self.data,
        # so the callback only flags the request.
        self.create_subscription(Empty, "/mabel/reset_sim",
                                 self._on_reset, 10, callback_group=grp)

        # publishers
        # /clock: this node is the simulation time authority. Published from the
        # physics thread at the snapshot rate; every consumer runs use_sim_time:=true.
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_arms = self.create_publisher(JointState, cfg.ARMS_STATE_TOPIC, 10)
        self.pub_body = self.create_publisher(JointState, cfg.BODY_STATE_TOPIC, 10)
        # Ground-truth odom: own /odom (+TF) normally, or yield it to the
        # GROUND-TRUTH base pose -- ALWAYS published on /odom_ground_truth. The
        # separate sensor-render process places the cameras at the robot's TRUE pose
        # using this, never the odometry estimate: rendering from the estimate would
        # make the camera view agree with wheel-odom drift, leaving cuVSLAM nothing to
        # correct (the robot then "walks out of the room"). When publish_odom=true the
        # sim ALSO owns /odom + the odom->base_link TF (standalone, no wheel-odom /
        # cuVSLAM); when false those are owned by swerve_odometry or cuVSLAM and the
        # sim emits ground truth only.
        self.pub_odom_gt = self.create_publisher(Odometry, "/odom_ground_truth", 10)
        self.pub_odom = (self.create_publisher(Odometry, cfg.ODOM_TOPIC, 10)
                         if self.publish_odom else None)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Front D435i IMU: best-effort sensor QoS (matches realsense2_camera and
        # what Isaac ROS cuVSLAM subscribes with). The rigid base_link->IMU
        # extrinsic is latched once as a STATIC TF for the fusion to chain through.
        self.pub_imu = None
        if getattr(self, "_imu_ok", False):
            self.pub_imu = self.create_publisher(
                Imu, cfg.IMU_TOPIC, qos_profile_sensor_data)
            self._static_tf = StaticTransformBroadcaster(self)
            self._publish_imu_static_tf()
        self.pub_hand_js = {}
        self.pub_hand_temp = {}
        self.pub_hand_volt = {}
        for side in cfg.HAND_SIDES:
            self.pub_hand_js[side] = self.create_publisher(
                JointState, f"/{side}_hand/joint_states", 10)
            self.pub_hand_temp[side] = self.create_publisher(
                Float64MultiArray, f"/{side}_hand/temperatures", 10)
            self.pub_hand_volt[side] = self.create_publisher(
                Float64MultiArray, f"/{side}_hand/voltages", 10)

        # timers
        self.create_timer(1.0 / cfg.RATE_JOINT_STATES, self._publish_joint_states,
                          callback_group=grp)
        self.create_timer(1.0 / cfg.RATE_HANDS, self._publish_hands, callback_group=grp)
        self.create_timer(1.0 / cfg.RATE_ODOM, self._publish_odom, callback_group=grp)
        self.create_timer(1.0 / cfg.RATE_TELEMETRY, self._telemetry, callback_group=grp)
        if self.pub_imu is not None:
            imu_rate = max(float(self.get_parameter("imu_rate").value), 1.0)
            self.create_timer(1.0 / imu_rate, self._publish_imu, callback_group=grp)
            self.get_logger().info(
                f"front IMU online: {cfg.IMU_TOPIC} @ {imu_rate:.0f} Hz "
                f"(frame {cfg.IMU_FRAME}, simulated BMI055 noise)")

    # -------------------------------------------------------------- callbacks
    def _on_base_cmd(self, msg: MabelSwerveCmd):
        with self._cmd_lock:
            if msg.hardware_state == 1:               # e-stop
                self._cmd.update(vx=0.0, vy=0.0, omega=0.0, estop=True)
            else:
                self._cmd.update(vx=float(msg.vx), vy=float(msg.vy),
                                 omega=float(msg.omega), estop=False)
                # NaN = "no lift command" (the real gateway's unseeded gate);
                # keep the previous target rather than poisoning the actuator.
                if math.isfinite(msg.lift_pos):
                    self._cmd["lift"] = float(np.clip(msg.lift_pos,
                                                      cfg.LIFT_MIN, cfg.LIFT_MAX))

    def _on_joint_cmd(self, msg: JointState, name_map):
        if not msg.name or not msg.position:
            return
        with self._cmd_lock:
            for name, angle in zip(msg.name, msg.position):
                aid = name_map.get(name)
                if aid is None:
                    continue
                _, _, lo, hi = self._pos_actuators[aid]
                self._pos_targets[aid] = float(np.clip(angle, lo, hi))

    def _on_control_mode(self, msg: String):
        """Request a whole-body control mode. The physics thread does the actual
        engage/disengage (it owns self.data); here we only record the request."""
        m = (msg.data or "").strip().lower()
        if m not in ("position", "compliance"):
            return
        with self._cmd_lock:
            self._req_ctrl_mode = m

    def _on_reset(self, msg: Empty):
        """Request a sim reset to home. The physics thread (which owns self.data)
        performs the actual mj_reset; here we only record the request."""
        with self._cmd_lock:
            self._req_reset = True

    def _reset_world(self):
        """Reset physics to the home keyframe (or model defaults) and re-seed every
        actuator target to the home pose so the position servos hold home instead of
        snapping back to the pre-reset targets. PHYSICS THREAD ONLY (owns self.data)."""
        if self._wbc is not None:          # drop out of compliance before homing
            self._exit_compliance()
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0    # clear any held gravcomp / external wrench
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Re-seed the base twist + lift + position targets to the (now home) pose.
        with self._cmd_lock:
            self._cmd.update(vx=0.0, vy=0.0, omega=0.0, estop=False)
            self._cmd["lift"] = self._cascaded_lift()
            for aid, (jqadr, _, lo, hi) in self._pos_actuators.items():
                self._pos_targets[aid] = float(np.clip(self.data.qpos[jqadr], lo, hi))
        self._lift_target = self._cascaded_lift()
        self.get_logger().info("simulation reset to home keyframe")

    # ---------------------------------------------------- compliance engine
    def _enter_compliance(self):
        """Engage whole-body compliance: instantiate the controller's WBCEngine and
        switch it to (compliance, whole_body). It neutralises the arm/torso position
        servos and gravity-compensates them (back-drivable), and admits the viewer's
        drag wrench into base motion. Runs ONLY on the physics thread."""
        try:
            eng = _WBCEngine(self.model, self.data)
            eng.enter("compliance", "whole_body")
            # body ids that can receive an operator push -> the vertical-lift wrench
            self._arm_bids = [self.model.body(n).id for n in eng.arm_bodies]
            # Hold the lift exactly where it is on engage. Seed from the last
            # COMMANDED cascaded target (same units the 'Lift' tendon actuator and
            # the position-teleop lift_pos use) so the handover from nav / SE(3) IK
            # into compliance does not drop the lift to half height.
            self._lift_target = float(self._cmd["lift"])
            self._wbc = eng
            self.get_logger().info(
                "COMPLIANCE engaged: arms + torso gravity-compensated & back-drivable; "
                "the base admits the drag wrench and the lift follows a vertical pull. "
                "Ctrl+drag any arm/body in the MuJoCo viewer.")
        except Exception as e:
            self._compliance_failed = True
            self.get_logger().error(f"compliance engage failed ({e}); staying in position mode")

    def _exit_compliance(self):
        """Release compliance: restore the position servos and reseed every position
        target (+ the lift) to the CURRENT, possibly-dragged pose so the stiff servos
        hold the robot exactly where it was left instead of snapping to stale targets."""
        try:
            self._wbc.exit()          # restores actuator gains, clears qfrc/xfrc_applied
        except Exception as e:
            self.get_logger().warn(f"compliance release: {e}")
        self._wbc = None
        self._arm_bids = []
        with self._cmd_lock:
            for aid, (qadr, _, lo, hi) in self._pos_actuators.items():
                self._pos_targets[aid] = float(np.clip(self.data.qpos[qadr], lo, hi))
            # Re-seed the lift command to the (possibly dragged) cascaded height so
            # the restored position servo holds it there instead of snapping to half.
            self._cmd["lift"] = float(self._lift_target)
            self._cmd.update(vx=0.0, vy=0.0, omega=0.0)
        self.get_logger().info(
            "COMPLIANCE released: position servos restored; holding the dragged pose.")

    def _cascaded_lift(self):
        """Current lift height in the CASCADED units the 'Lift' tendon actuator is
        commanded in (Lift Lower + Lift Upper). Use this — never a single slide
        joint — whenever seeding/holding the lift command, so the hold matches where
        the lift actually is instead of snapping to half height."""
        if self._lift_qadr is None:
            return 0.0
        lo = float(self.data.qpos[self._lift_qadr])
        up = (float(self.data.qpos[self._lift_upper_qadr])
              if self._lift_upper_qadr is not None else lo)
        return float(np.clip(lo + up, cfg.LIFT_MIN, cfg.LIFT_MAX))

    def _lift_admit_step(self, lift_target, dt):
        """Glide the lift under a vertical drag on the arms: integrate the net
        vertical external force (the viewer drag, data.xfrc_applied) through a
        deadband + viscous virtual friction, capped at the lift's cruise speed."""
        if not self._arm_bids or self._lift_qadr is None:
            return lift_target
        fz = float(sum(self.data.xfrc_applied[b, 2] for b in self._arm_bids))
        if abs(fz) < LIFT_ADMIT_DEADBAND:
            return lift_target
        v = float(np.clip(fz / LIFT_ADMIT_DAMP, -LIFT_ADMIT_VMAX, LIFT_ADMIT_VMAX))
        return float(np.clip(lift_target + v * dt, cfg.LIFT_MIN, cfg.LIFT_MAX))

    # ------------------------------------------------------------ physics loop
    def _physics_loop(self):
        """Robust real-time physics bridge. The ONLY thread that touches self.data.

        Design (after a long debugging session -- the old "catch-up burst + 5 ms
        sleep" loop was broken two ways):
          1. It stepped a VARIABLE burst then slept 5 ms; that structure roughly
             HALVED the achieved base speed even with full sim time (measured: the
             robot moved at ~48% of commanded vs ~98% for a continuous loop).
          2. When it fell behind (target - sim_time > 0.05 s -- routine under ROS/GIL
             load) it RESET the clock and DISCARDED the missed sim time, so the robot
             got ~1 step per frame and crawled at a few % of commanded.
        Both are gone. This loop steps EXACTLY ONE physics step per iteration
        (continuous -- never bursts, never discards a step), then sleeps just enough
        to pace toward real time. The sleep ALWAYS happens (>=0), which releases the
        GIL so the ROS executor reliably processes /mabel_cmd and runs the publishers.
        If the machine can't sustain real time the loop simply runs at RTF<1 -- every
        step is still simulated, so the motion is always FAITHFUL (never skipped)."""
        dt = self.model.opt.timestep
        cvx = cvy = comega = 0.0          # ramped base twist (this thread only)
        # Step a small batch of physics steps per loop iteration, then snapshot +
        # pace ONCE. The batch is still continuous stepping (every step is taken --
        # motion stays faithful, ~97% of commanded), but it cuts the per-step Python
        # + sleep overhead ~Nx, which RAISES the real-time factor so the robot moves
        # at a sensible wall-clock speed instead of crawling. ~8 ms of sim per batch
        # keeps snapshots fresh (~125 Hz) for the publishers.
        substeps = max(1, int(round(0.008 / dt)))
        # viewer.sync() is expensive; only ~60 Hz of it is needed.
        viewer_decim = max(1, int(round((1.0 / 60.0) / (substeps * dt))))

        import contextlib
        if self.headless:
            self.get_logger().info("headless: physics running without the viewer")
            viewer_cm = contextlib.nullcontext(None)
        else:
            viewer_cm = mujoco.viewer.launch_passive(self.model, self.data)

        with viewer_cm as viewer:
            if viewer is not None:
                # Don't draw the LiDAR as physical lines: mjVIS_RANGEFINDER defaults
                # ON and renders the 360 beams as solid rays in the viewer. The LiDAR
                # should be distance readings only. Also hide the debug-only group-3
                # geoms/sites (camera markers + LiDAR ray sites).
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
                viewer.opt.geomgroup[3] = 0
                viewer.opt.sitegroup[3] = 0
                # Default camera: a framed isometric look at the robot's FRONT. The
                # robot stands at the origin facing -X (head/arms look -X; the lift is
                # mounted with a 180 deg yaw), so we orbit to the -X/-Y front-quarter
                # and pull back enough to see it whole. azimuth=45 + distance 3.2 keep
                # the eye well inside the 6x5 m room (x in [-3,3], y in [-2.5,2.5]) so
                # no wall ends up between the camera and the robot.
                viewer.cam.lookat[:] = [0.0, 0.0, 0.8]   # robot mid-height
                viewer.cam.distance = 3.2
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
            next_wall = time.monotonic()    # wall deadline of the next step
            i = 0
            while ((viewer is None or viewer.is_running())
                   and self.running and rclpy.ok()):
                # read the latest command once (tiny critical section)
                with self._cmd_lock:
                    tvx, tvy, tomega = self._cmd["vx"], self._cmd["vy"], self._cmd["omega"]
                    lift = self._cmd["lift"]
                    pos_targets = list(self._pos_targets.items())
                    req_mode = self._req_ctrl_mode
                    req_reset = self._req_reset
                    self._req_reset = False

                # ---- sim reset (iOS/VP Reset button) -- home the world, drop the
                #      velocity ramp, and restart the loop so the freshly re-seeded
                #      targets are picked up on the next pass.
                if req_reset:
                    self._reset_world()
                    cvx = cvy = comega = 0.0
                    continue

                # ---- engage / disengage whole-body compliance (this thread owns
                #      self.data + the engine) ------------------------------------
                want_compliance = (req_mode == "compliance" and _COMPLIANCE_OK
                                   and not self._compliance_failed)
                if want_compliance and self._wbc is None:
                    self._enter_compliance()
                elif not want_compliance and self._wbc is not None:
                    self._exit_compliance()
                if req_mode == "compliance" and not _COMPLIANCE_OK and not self._compliance_failed:
                    self._compliance_failed = True
                    self.get_logger().warn(
                        f"compliance requested but unavailable ({_COMPLIANCE_ERR}); "
                        "holding position. Check the controller path (MABEL_CONTROLLER).")

                # ---- a batch of continuous physics steps (never burst/discard) ----
                for _ in range(substeps):
                    if self._wbc is not None:
                        # Compliance: the engine gravity-compensates + back-drives the
                        # arms/torso and admits the planar drag into the base (it sets
                        # the swerve ctrl itself); we add the vertical-lift admittance.
                        self._lift_target = self._lift_admit_step(self._lift_target, dt)
                        if self._lift_a is not None:
                            self.data.ctrl[self._lift_a] = self._lift_target
                        self._wbc.step(dt)
                    else:
                        cvx = _ramp(cvx, tvx, cfg.MAX_LIN_ACCEL * dt)
                        cvy = _ramp(cvy, tvy, cfg.MAX_LIN_ACCEL * dt)
                        comega = _ramp(comega, tomega, cfg.MAX_ANG_ACCEL * dt)
                        self._apply_swerve(cvx, cvy, comega)
                        if self._lift_a is not None:
                            self.data.ctrl[self._lift_a] = lift
                        for aid, val in pos_targets:
                            self.data.ctrl[aid] = val
                        # Gravity-comp feed-forward: cancel the upper-body joint-space
                        # bias so the position servos hold the commanded pose with no
                        # gravity droop (the arms don't sag / drop between modes).
                        if self._gravcomp_dofs is not None:
                            self.data.qfrc_applied[self._gravcomp_dofs] = \
                                self.data.qfrc_bias[self._gravcomp_dofs]
                    mujoco.mj_step(self.model, self.data)
                i += 1

                # snapshot for the (snapshot-only) ROS publishers, under the lock
                with self._state_lock:
                    self._snap_qpos[:] = self.data.qpos
                    self._snap_qvel[:] = self.data.qvel
                    self._snap_sensordata[:] = self.data.sensordata
                    self._snap_time = self.data.time

                # /clock authority: publish SIM time so all consumers integrate in sim
                # time (correct even at RTF<1). data.time is owned by this thread.
                # Split via _split_time so the nanosecond round-up CARRIES into sec:
                # the old `round(frac*1e9) % 1e9` zeroed nanosec without bumping sec, so
                # at every integer-second boundary (frac~=0.9999999 rounds to 1e9) it
                # emitted e.g. 125.0 instead of 126.0 -- a ~1 s BACKWARD /clock jump that
                # cleared every downstream TF buffer ("Detected jump back in time") and
                # teleported the robot in RViz once per second.
                sec, nsec = self._split_time(self.data.time)
                ck = Clock()
                ck.clock.sec = sec
                ck.clock.nanosec = nsec
                # On Ctrl-C the rclpy context is torn down while this thread is mid-
                # iteration; a publish then raises "publisher's context is invalid".
                # Bail out of the loop cleanly instead of dumping a thread traceback.
                if not rclpy.ok():
                    break
                try:
                    self.pub_clock.publish(ck)
                except Exception:
                    break

                if viewer is not None and i % viewer_decim == 0:
                    viewer.sync()

                # ---- pace + ALWAYS yield the GIL so ROS comms stay live ----
                if self.realtime:
                    next_wall += substeps * dt
                    slack = next_wall - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)            # paces to real time + yields
                    else:
                        time.sleep(0)                # behind: yield but DON'T skip a step
                        if slack < -0.25:            # cap debt so we don't sprint forever
                            next_wall = time.monotonic()
                else:
                    time.sleep(0)                    # flat-out: yield each batch for ROS

    def _apply_swerve(self, vx, vy, omega):
        # Coordinated swerve, matching the canonical controller IK
        # (controller/mabel/swerve.py) EXACTLY: each module drives as soon as IT is
        # within the steering deadband, with a per-module cosine anti-scrub taper.
        #
        # A previous version gated drive on the WORST-aligned module (no drive until
        # ALL three were within the deadband). That all-or-nothing gate created a
        # stop-start LIMIT CYCLE -- the moment the base moved, tiny chassis yaw nudged
        # one module's steering error back over the deadband, cut ALL drive, the base
        # stalled, re-aligned, lurched, repeat -- so it crawled at ~2% of commanded
        # speed (measured) regardless of scene or friction. Per-module drive (below)
        # is verified at ~99% of commanded in the bare scene, so the SIM now reflects
        # the calibrated base_motion model (controller/mabel/base_motion.py).
        for key, s in self._swerve.items():
            rx, ry = s["pos"]
            v_ix = vx - omega * ry
            v_iy = vy + omega * rx
            v_mag = math.hypot(v_ix, v_iy)
            cur = self.data.qpos[s["steer_q"]]
            if v_mag > 1e-3:
                # global heading -> module-local steer (subtract mount yaw)
                local_target = math.atan2(v_iy, v_ix) - s["yaw"]
                theta, dir_mult = _optimize_steering(local_target, cur)
                err = abs(theta - cur)
                drive_scale = 0.0 if err > cfg.STEER_DEADBAND else math.cos(err)
                self.data.ctrl[s["steer_a"]] = theta
                self.data.ctrl[s["drive_a"]] = (
                    (v_mag / self.wheel_radius) * dir_mult * drive_scale)
            else:
                self.data.ctrl[s["steer_a"]] = cur
                self.data.ctrl[s["drive_a"]] = 0.0

    # --------------------------------------------------------------- publishers
    def _read_snapshot(self):
        with self._state_lock:
            return self._snap_qpos.copy(), self._snap_qvel.copy()

    def _sim_stamp(self):
        """builtin_interfaces/Time at the latest SNAPSHOT *sim* time. mujoco_node is
        the time authority: it stamps every message with sim time (data.time) and
        publishes /clock, so downstream nodes (use_sim_time:=true) integrate odometry
        and SLAM correctly even when the sim runs below real time (RTF<1). With
        wall-clock stamps a 0.5x RTF made odometry over-read ~2x; sim-time stamps make
        it exact regardless of RTF."""
        sec, nsec = self._split_time(self._snap_time)
        return TimeMsg(sec=sec, nanosec=nsec)

    @staticmethod
    def _split_time(t):
        """Split a float sim time (s) into integer (sec, nanosec), CARRYING a
        nanosecond round-up of 1e9 into sec. Without the carry, a time just below
        an integer second rounds nanosec to a full second and -- if masked with
        `% 1e9` -- drops ~1 s, making the stamp jump backward. Used by both the
        /clock publisher and every message stamp so all sim time is monotonic."""
        sec = int(t)
        nsec = int(round((t - sec) * 1e9))
        if nsec >= 1_000_000_000:
            sec += 1
            nsec -= 1_000_000_000
        return sec, nsec

    def _publish_imu_static_tf(self):
        """Latch the constant base_link -> front IMU optical transform (once)."""
        tf = TransformStamped()
        tf.header.stamp = self._sim_stamp()
        tf.header.frame_id = cfg.BASE_FRAME
        tf.child_frame_id = cfg.IMU_FRAME
        tf.transform.translation.x = float(self._imu_tf_t[0])
        tf.transform.translation.y = float(self._imu_tf_t[1])
        tf.transform.translation.z = float(self._imu_tf_t[2])
        qx, qy, qz, qw = self._imu_tf_q
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self._static_tf.sendTransform(tf)

    def _publish_imu(self):
        """Front D435i IMU: MuJoCo accelerometer/gyro (specific force incl. gravity,
        body angular rate) + a simulated BMI055 noise model (white noise + fixed
        bias + slow random walk), as sensor_msgs/Imu in the front_imu frame."""
        with self._state_lock:
            sd = self._snap_sensordata
            acc = sd[self._imu_acc_adr:self._imu_acc_adr + 3].copy()
            gyro = sd[self._imu_gyro_adr:self._imu_gyro_adr + 3].copy()
            quat_wxyz = (sd[self._imu_quat_adr:self._imu_quat_adr + 4].copy()
                         if self._imu_quat_adr is not None else None)

        # bias random walk (per published sample) then white noise on top
        self._imu_acc_bias += self._imu_rng.normal(0.0, self._imu_acc_walk, 3)
        self._imu_gyro_bias += self._imu_rng.normal(0.0, self._imu_gyro_walk, 3)
        acc += self._imu_acc_bias + self._imu_rng.normal(0.0, self._imu_acc_noise, 3)
        gyro += self._imu_gyro_bias + self._imu_rng.normal(0.0, self._imu_gyro_noise, 3)

        msg = Imu()
        msg.header.stamp = self._sim_stamp()
        msg.header.frame_id = cfg.IMU_FRAME
        msg.linear_acceleration.x = float(acc[0])
        msg.linear_acceleration.y = float(acc[1])
        msg.linear_acceleration.z = float(acc[2])
        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        if quat_wxyz is not None:
            msg.orientation.w = float(quat_wxyz[0])
            msg.orientation.x = float(quat_wxyz[1])
            msg.orientation.y = float(quat_wxyz[2])
            msg.orientation.z = float(quat_wxyz[3])
        else:
            msg.orientation.w = 1.0
        # Diagonal covariances from the white-noise variances (REP-145: a leading
        # -1 in orientation_covariance flags "orientation not estimated" for VIO).
        av = self._imu_gyro_noise ** 2
        lv = self._imu_acc_noise ** 2
        msg.angular_velocity_covariance = [av, 0.0, 0.0, 0.0, av, 0.0, 0.0, 0.0, av]
        msg.linear_acceleration_covariance = [lv, 0.0, 0.0, 0.0, lv, 0.0, 0.0, 0.0, lv]
        msg.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_imu.publish(msg)

    def _publish_joint_states(self):
        qpos, qvel = self._read_snapshot()
        now = self._sim_stamp()

        q0 = self.model.qpos0
        js = JointState()
        js.header.stamp = now
        for name, qa, va in self._global_joints:
            js.name.append(name)
            # URDF q=0 is the assembled (ref) pose: mjcf_to_urdf shifts every joint
            # limit by qpos0, so the URDF displacement is (qpos - qpos0). Publishing
            # raw qpos slams the joints whose ref != 0 (wrists, thumb, abductions) to
            # their limits in RViz even though MuJoCo is at neutral. Subtract qpos0.
            js.position.append(float(qpos[qa] - q0[qa]))
            js.velocity.append(float(qvel[va]))
        self.pub_js.publish(js)

        self.pub_arms.publish(self._subset_js(now, self._arms_pub, qpos, qvel))
        self.pub_body.publish(self._subset_js(now, self._body_pub, qpos, qvel))

    def _publish_hands(self):
        qpos, qvel = self._read_snapshot()
        now = self._sim_stamp()
        for side in cfg.HAND_SIDES:
            pub_list = self._hand_pub[side]
            if not pub_list:
                continue
            names = [n for n, _, _ in pub_list]
            js = self._subset_js(now, pub_list, qpos, qvel)
            js.header.frame_id = f"{side}_hand"
            js.effort = [0.0] * len(names)
            self.pub_hand_js[side].publish(js)
            self.pub_hand_temp[side].publish(
                _labelled_array(names, [cfg.HAND_FAKE_TEMP_C] * len(names)))
            self.pub_hand_volt[side].publish(
                _labelled_array(names, [cfg.HAND_FAKE_VOLT_V] * len(names)))

    def _subset_js(self, stamp, pub_list, qpos, qvel):
        q0 = self.model.qpos0
        js = JointState()
        js.header.stamp = stamp
        for name, qa, va in pub_list:
            js.name.append(name)
            # (qpos - qpos0): match the URDF's ref-shifted joint zero (see
            # _publish_joint_states) so hands/arms track MuJoCo neutral in RViz.
            js.position.append(float(qpos[qa] - q0[qa]))
            js.velocity.append(float(qvel[va]))
        return js

    def _publish_odom(self):
        if self._base_qadr is None:
            return
        qpos, qvel = self._read_snapshot()
        now = self._sim_stamp()
        q = self._base_qadr
        v = self._base_vadr
        px, py, pz = qpos[q], qpos[q + 1], qpos[q + 2]
        qw, qx, qy, qz = qpos[q + 3], qpos[q + 4], qpos[q + 5], qpos[q + 6]

        odom = Odometry()
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
        # MuJoCo free-joint qvel: the LINEAR part (v..v+2) is in the WORLD frame, while
        # the ANGULAR part (v+3..v+5) is already in the local BODY frame. ROS (REP-105)
        # and every consumer -- the nav cmd_vel_bridge's velocity-feedback flip, the EKF,
        # and the real robot's swerve_odometry (which emits a body-frame twist) -- require
        # the twist expressed in child_frame_id (base_link = body). Publishing the raw
        # world-frame linear velocity here made MPPI's velocity feedback sign-flip whenever
        # the base was rotated (worst at yaw ~= pi, which is exactly the front-facing nav
        # attitude), so the controller drove toward the goal and then peeled away / reversed
        # on replan. Rotate the linear velocity world->body by the base orientation's
        # conjugate; the angular part is left untouched.
        v_world = np.array([qvel[v], qvel[v + 1], qvel[v + 2]], dtype=np.float64)
        quat = np.array([qw, qx, qy, qz], dtype=np.float64)        # mujoco order (w,x,y,z)
        quat_inv = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(quat_inv, quat)                         # unit-quat conjugate = inverse
        v_body = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(v_body, v_world, quat_inv)           # world -> body
        odom.twist.twist.linear.x = float(v_body[0])
        odom.twist.twist.linear.y = float(v_body[1])
        odom.twist.twist.linear.z = float(v_body[2])
        odom.twist.twist.angular.x = float(qvel[v + 3])            # already body frame
        odom.twist.twist.angular.y = float(qvel[v + 4])
        odom.twist.twist.angular.z = float(qvel[v + 5])
        self.pub_odom_gt.publish(odom)      # always: TRUE pose for sim sensors + eval

        if not self.publish_odom:
            return                      # swerve_odometry / cuVSLAM own /odom + the TF
        self.pub_odom.publish(odom)         # standalone: the sim also owns /odom

        tf = TransformStamped()
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
        self.tf_broadcaster.sendTransform(tf)

    def _telemetry(self):
        if self._lift_qadr is None:
            return
        qpos, _ = self._read_snapshot()
        with self._cmd_lock:
            vx, estop = self._cmd["vx"], self._cmd["estop"]
        lift = float(qpos[self._lift_qadr])             # cascaded = lower + upper
        if self._lift_upper_qadr is not None:
            lift += float(qpos[self._lift_upper_qadr])
        flag = " [E-STOP]" if estop else ""
        print(f"\r[SIM] cmd vx={vx:+.2f} m/s | lift={lift:.3f} m | "
              f"t={self._snap_time:7.1f}s{flag}      ", end="")

    # ----------------------------------------------------------------- shutdown
    def destroy_node(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        super().destroy_node()


# --------------------------------------------------------------------------- #
#  module-level helpers
# --------------------------------------------------------------------------- #

def _ramp(current, target, max_step):
    if current < target:
        return min(target, current + max_step)
    if current > target:
        return max(target, current - max_step)
    return current


def _optimize_steering(target_angle, current_angle):
    """Shortest-path steering with drive-direction flip (avoid >90 deg turns).
    Verbatim from simulation/mabel_mujoco/scripts/mabel/swerve.py."""
    target_angle = current_angle + (
        (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi)
    if target_angle - current_angle > math.pi / 2:
        return target_angle - math.pi, -1.0
    if target_angle - current_angle < -math.pi / 2:
        return target_angle + math.pi, -1.0
    return target_angle, 1.0


def _labelled_array(names, values):
    m = Float64MultiArray()
    m.layout = MultiArrayLayout()
    m.layout.dim = [MultiArrayDimension(label=n, size=1, stride=1) for n in names]
    m.data = [float(v) for v in values]
    return m


def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
