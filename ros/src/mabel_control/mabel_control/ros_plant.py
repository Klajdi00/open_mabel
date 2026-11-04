"""ROS plant: the real-robot replacement for stepping MuJoCo physics.

The gateway (mabel_gateway / the server) hosts the controller in-process in BOTH the sim
and on the real robot; only the PLANT differs. On the Mac the plant is MuJoCoPlant
(`mj_step` advances physics). On the robot this ROSPlant takes its place: each control tick
it mirrors the measured robot state into the MuJoCo model (kinematics only) and publishes
the controller's decision on the canonical command contract that the hardware drivers
consume -- so ROS is the hardware-comms bus and the controller code is unchanged.

Duck-typed plant interface the gateway loop expects (same shape as MuJoCoPlant):

    plant.read_state()        # refresh data.qpos from sensors; mj_forward (no physics)
    plant.write(robot_cmd)    # publish the controller's RobotCommand on the contract

This module imports rclpy and ROS messages, so it is only imported in the ROS workspace
(never on the Mac sim path, which uses MuJoCoPlant in server/). It is the rclpy wrapper
around the pure, offline-tested `joint_mirror` + `command_extract`.
"""
import math
import os
import time

import mujoco
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy)

# Latched QoS to MATCH the bridge's torque_state publishers (RELIABLE +
# TRANSIENT_LOCAL). A VOLATILE subscriber never receives the last-published
# (latched) value from a TRANSIENT_LOCAL publisher, so a freshly-restarted gateway
# saw torque_state = None forever and the torque reconcile could never act — the
# arms/left hand stayed limp with no way to enable them. This makes the subscriber
# get the current state immediately on connect.
_LATCHED_QOS = QoSProfile(depth=1)
_LATCHED_QOS.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
_LATCHED_QOS.reliability = QoSReliabilityPolicy.RELIABLE
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry

from mabel_interfaces.msg import MabelSwerveCmd
from mabel_sim import interface_config as cfg

from .joint_mirror import JointMirror, mirror_base
from .command_extract import extract


class ROSPlant:
    """Mirror robot state into `model`/`data`; publish controller commands on the contract.

    Parameters
    ----------
    node  : rclpy.node.Node     the host node (owns subscriptions/publishers/clock)
    model, data : MuJoCo handles used by the controller as its kinematic/dynamic engine
    """

    def __init__(self, node, model, data):
        self.node = node
        self.model = model
        self.data = data
        self._mirror = JointMirror(model)

        # Latest measured state, refreshed by subscriptions, applied in read_state().
        # Stored as ONE tuple so a reader (the decoupled feedback path) never sees a
        # torn names/positions pair while a callback is mid-update.
        self._joints = ([], [])     # (names, positions)
        self._odom = None           # (x, y, z, (w, x, y, z))
        # Full per-joint dynamics for the telemetry plane: name → (vel, effort).
        # Kept SEPARATE from self._joints (position-only, byte-identical to the
        # original) so every existing consumer of the mirror is untouched.
        # Atomic dict rebinds, same discipline as self._joints.
        self._dyn: dict = {}
        # Hand extras: per-side temperatures/voltages (Float64MultiArray, in the
        # hand's joint_states name order) and the latched torque state.
        self._hand_names: dict = {s: [] for s in cfg.HAND_SIDES}
        self._hand_temps: dict = {s: None for s in cfg.HAND_SIDES}
        self._hand_volts: dict = {s: None for s in cfg.HAND_SIDES}
        self._hand_torque: dict = {s: None for s in cfg.HAND_SIDES}
        # Arm/body motor health ({joint: {torqueNm, tempC}} JSON @2 Hz from the
        # hardware bridge's DaMiao feedback) — the TEMPERATURE source for those
        # joints (their JointState has no temp field) and the τ fallback.
        self._motor_health: dict = {}
        self._motor_health_t = 0.0           # wall time of the newest */motor_health

        # --- state IN: the same topics the sim/hardware publish ---
        node.create_subscription(JointState, "/joint_states",
                                 self._on_joints, qos_profile_sensor_data)
        node.create_subscription(JointState, cfg.ARMS_STATE_TOPIC,
                                 self._on_joints, qos_profile_sensor_data)
        node.create_subscription(JointState, cfg.BODY_STATE_TOPIC,
                                 self._on_joints, qos_profile_sensor_data)
        from std_msgs.msg import Float64MultiArray
        for side in cfg.HAND_SIDES:
            node.create_subscription(
                JointState, f"/{side}_hand/joint_states",
                (lambda m, s=side: self._on_hand_joints(s, m)),
                qos_profile_sensor_data)
            node.create_subscription(
                Float64MultiArray, f"/{side}_hand/temperatures",
                (lambda m, s=side: self._hand_temps.__setitem__(s, list(m.data))),
                qos_profile_sensor_data)
            node.create_subscription(
                Float64MultiArray, f"/{side}_hand/voltages",
                (lambda m, s=side: self._hand_volts.__setitem__(s, list(m.data))),
                qos_profile_sensor_data)
            node.create_subscription(
                Bool, f"/{side}_hand/torque_state",
                (lambda m, s=side: self._hand_torque.__setitem__(s, bool(m.data))),
                _LATCHED_QOS)      # match the bridge's TRANSIENT_LOCAL publisher
        from std_msgs.msg import String
        # All four HAL health topics (hands included): since the perception split
        # (2026-07-13) the map plane's subscriber that used to feed
        # robot_state.motorHealth no longer runs inside the gateway, so this cache is
        # what the gateway falls back to (server/plant.py::ROSPlant.motor_health).
        for _mh_topic in ("/arms/motor_health", "/body/motor_health",
                          "/left_hand/motor_health", "/right_hand/motor_health"):
            node.create_subscription(String, _mh_topic, self._on_motor_health, 5)
        node.create_subscription(Odometry, cfg.ODOM_TOPIC,
                                 self._on_odom, qos_profile_sensor_data)
        # Camera-only bench fallback: when the mobile base is unplugged the
        # wheel ODOM_TOPIC goes silent — cuVSLAM's camera odometry then IS the
        # robot odom (no motion model). Wheel odom always wins while fresh.
        self._wheel_odom_t = 0.0
        node.create_subscription(
            Odometry,
            os.getenv("MABEL_VISUAL_ODOM_TOPIC", "/visual_slam/tracking/odometry"),
            self._on_visual_odom, qos_profile_sensor_data)

        # --- command OUT: the canonical contract the drivers consume ---
        self._pub_base = node.create_publisher(MabelSwerveCmd, cfg.CMD_BASE_TOPIC, 10)
        # Lift JOG velocity (m/s, +up) -> the bridge's /lift/velocity. The drive stick's
        # up/down rides the firmware's velocity mode (move-while-held, stop-on-release),
        # kept OFF the position /lift/command so a held stick does not ramp a setpoint the
        # Pico's position loop would flood on. Published every tick from send_lift_velocity.
        from std_msgs.msg import Float64 as _Float64
        self._pub_lift_vel = node.create_publisher(_Float64, "/lift/velocity", 10)
        # Position /lift command gate. While False, write() publishes lift_pos=NaN —
        # the contract's "no lift command" sentinel (swerve_node drops it, the HAL
        # ignores it) — so an UNSEEDED data.ctrl can never drive the column (the
        # "lift dives to the floor / a stale target on bring-up" bug). The gateway
        # opens the gate only after seeding ctrl from measured /lift feedback.
        self._lift_cmd_enabled = True
        self._lift_fb_t = -1e9       # last time a lift joint arrived (liveness stamp)
        self._mirror_kinematics_only = False   # gateway flips per control mode
        self._pub_arms = node.create_publisher(JointState, cfg.CMD_ARMS_TOPIC, 10)
        self._pub_body = node.create_publisher(JointState, cfg.CMD_BODY_TOPIC, 10)
        self._pub_hands = {s: node.create_publisher(JointState, f"/{s}_hand/command", 10)
                           for s in cfg.HAND_SIDES}
        # Per-hand torque enable (-> the bridge's /<side>_hand/torque_enable). The app
        # ties this to the e-stop: disengaged -> energise so the operator can control
        # the hands; engaged -> release to free-drive. The bridge holds-in-place before
        # energising (hands_controller._hold_then_enable), so there is no lurch.
        self._pub_hand_torque = {
            s: node.create_publisher(Bool, f"/{s}_hand/torque_enable", 10)
            for s in cfg.HAND_SIDES}
        # ARM torque enable (-> the bridge's /arms/torque_enable). Same e-stop tie as
        # the hands: disengaged/teleop-start -> energise so the operator can drive the
        # arms; engaged -> release to free-drive. The bridge hold-then-enables (holds
        # the measured pose before energising), so there is no lurch. Latched cache of
        # /arms/torque_state so the gateway can reconcile intent across bridge restarts.
        self._pub_arms_torque = node.create_publisher(Bool, "/arms/torque_enable", 10)
        # ARM PD GAINS (additive plane, 2026-07-30): String JSON {"kp": {urdf: v},
        # "kd": {...}} -> the HAL applies them as per-command MIT overrides. This
        # is what makes the app's stiffness slider REAL on hardware and lets SOFT
        # run a softer gain set than STIFF. LATCHED so a restarted bridge adopts
        # the current gains. Values are computed controller-side (arm.gains +
        # arm.soft_gains); this class is plumbing.
        from rclpy.qos import QoSProfile as _QoS
        from rclpy.qos import DurabilityPolicy as _Dur
        _gq = _QoS(depth=1)
        _gq.durability = _Dur.TRANSIENT_LOCAL
        self._pub_arm_gains = node.create_publisher(String, "/arms/gains", _gq)
        self._arms_torque = None
        # NECK/BODY torque enable (-> the bridge's /body/torque_enable, consumed by
        # hardware_bridge/mabel_hw/ros/body_node.py::_on_torque). Same e-stop tie as
        # the arms + hands: disengaged/teleop-start -> energise so the operator can
        # drive the neck (nothing energised it before — the neck stayed LIMP and
        # ignored /body/command, the "neck commands do nothing" bug); engaged ->
        # release to limp. The bridge holds-in-place (goal:=present) before the
        # gateway's resume_hold reseeds measured, so re-energising never lurches.
        # Latched cache of /body/torque_state so the gateway can reconcile intent
        # across a bridge restart (the neck boots torque-OFF).
        self._pub_body_torque = node.create_publisher(Bool, "/body/torque_enable", 10)
        self._body_torque = None
        # GRAVITY FEEDFORWARD, {urdf arm joint: tau_g [N.m, MODEL frame]}, published on
        # /arms/command.effort. Set by the owner each tick (server/plant.py::ROSPlant);
        # empty = no feedforward, which makes the position term fight gravity alone and
        # the arm sag by tau_g/kp. NOT computed here: this class is ROS plumbing, and
        # the control law lives in controller/mabel/arm_control.
        self.arm_effort: dict = {}
        # Torso GRAVITY FEEDFORWARD for /body/command.effort, {"torso": tau [N.m,
        # MODEL frame]}. Set by the owner each tick (server/plant.py::ROSPlant) from
        # the SHARED arm+torso gravity evaluation — ARM-POSE-AWARE, unlike the
        # sinusoid fallback in _body_effort() (which covers body-only runs where the
        # arms don't stream). Empty = "no posture-aware value this tick".
        self.body_effort: dict = {}
        node.create_subscription(
            Bool, "/arms/torque_state",
            lambda m: setattr(self, "_arms_torque", bool(m.data)),
            _LATCHED_QOS)          # match the bridge's TRANSIENT_LOCAL publisher
        node.create_subscription(
            Bool, "/body/torque_state",
            lambda m: setattr(self, "_body_torque", bool(m.data)),
            _LATCHED_QOS)          # match body_node's TRANSIENT_LOCAL publisher

        # Nav2 goal forwarding (web/app tap → Nav2 NavFn A* + MPPI), and Nav2's
        # resulting twist back IN. THIS plant is the only publisher of /mabel_cmd,
        # so cmd_vel_bridge publishes its twist on the NON-canonical
        # /nav/mabel_cmd (its `cmd_out_topic` param) and we consume it here —
        # never two publishers on /mabel_cmd fighting for the base.
        #
        # The twist is not forwarded blind: the gateway reads nav_twist() and
        # re-injects it through the controller facade as a synthetic joystick, so
        # Nav2 is gated by the same control law + tip model as any teleop device,
        # and a human stick deflection preempts it (estop > teleop > nav). This is
        # the same wiring SimRosPlant uses — sim and real now behave identically.
        from geometry_msgs.msg import PoseStamped
        from visualization_msgs.msg import Marker
        from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                               QoSDurabilityPolicy)
        self._PoseStamped = PoseStamped
        self._Marker = Marker
        self._pub_goal = node.create_publisher(PoseStamped, "/goal_pose", 10)
        # STAGED goal preview (RViz only, does NOT drive) + a green ball at the goal
        # so the operator's RViz mirrors the web/app goal. Transient-local marker so
        # a late-joining RViz still shows the current goal.
        self._pub_goal_preview = node.create_publisher(
            PoseStamped, "/goal_pose_preview", 10)
        gm_qos = QoSProfile(depth=1)
        gm_qos.reliability = QoSReliabilityPolicy.RELIABLE
        gm_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub_goal_marker = node.create_publisher(
            Marker, "/mabel/goal_marker", gm_qos)

        # Nav2's twist IN (see the note above). Topic must match cmd_vel_bridge's
        # `cmd_out_topic`; MABEL_NAV_CMD_TOPIC overrides both in lockstep.
        from mabel_interfaces.msg import MabelSwerveCmd as _NavCmd
        self._nav_twist_v = (0.0, 0.0, 0.0)
        self._nav_cmd_t = -1e9
        node.create_subscription(
            _NavCmd, os.getenv("MABEL_NAV_CMD_TOPIC", "/nav/mabel_cmd"),
            self._on_nav_cmd, 10)

    def _on_nav_cmd(self, msg):
        import time as _t
        self._nav_twist_v = (float(msg.vx), float(msg.vy), float(msg.omega))
        self._nav_cmd_t = _t.monotonic()

    def nav_twist(self):
        """(vx, vy, omega, age_s) — Nav2's latest base command + its staleness.

        The gateway gates on `age < 0.5` before using it, so a dead/stopped Nav2
        simply ages out rather than latching the last twist onto the base."""
        import time as _t
        vx, vy, w = self._nav_twist_v
        return vx, vy, w, (_t.monotonic() - self._nav_cmd_t)

    def send_nav_goal(self, x, y, yaw=0.0):
        import math, os
        g = self._PoseStamped()
        g.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "map")
        g.header.stamp = self.node.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.z = math.sin(float(yaw) / 2.0)
        g.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self._pub_goal.publish(g)
        self._publish_goal_marker(x, y)        # green ball in RViz (synced w/ web)

    def send_nav_goal_preview(self, x, y, yaw=0.0):
        """Publish a STAGED goal to /goal_pose_preview (RViz only; does NOT drive)
        and drop the green goal ball, so the operator can confirm the goal arrived."""
        import math, os
        g = self._PoseStamped()
        g.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "map")
        g.header.stamp = self.node.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        g.pose.orientation.z = math.sin(float(yaw) / 2.0)
        g.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self._pub_goal_preview.publish(g)
        self._publish_goal_marker(x, y)

    def _publish_goal_marker(self, x, y, show=True):
        """Green SPHERE Marker at the goal (/mabel/goal_marker) — same frame as the
        goal so RViz draws it exactly where the web/app aimed. show=False removes it."""
        import os
        m = self._Marker()
        m.header.frame_id = os.getenv("MABEL_NAV_GOAL_FRAME", "map")
        m.header.stamp = self.node.get_clock().now().to_msg()
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

    # ── state callbacks (cache only; applied synchronously in read_state) ──────
    def _on_joints(self, msg: JointState):
        # Accumulate the most recent angle per joint name across all state topics.
        names, pos = self._joints
        latest = dict(zip(names, pos))
        latest.update(zip(msg.name, msg.position))
        # Single atomic rebind so concurrent readers get a consistent pair.
        self._joints = (list(latest.keys()), list(latest.values()))
        # LIFT LIVENESS: the joint cache above never forgets, so a dead lift keeps
        # its last value forever. Stamp the arrival of any lift joint so the gateway
        # can distinguish "lift streaming" from "stale memory" (lift_feedback_age) —
        # its command gate closes on a disconnect and re-seeds on return.
        if any(n == "lift" or n.startswith("lift_") for n in msg.name):
            self._lift_fb_t = time.time()
        # Dynamics sidecar (velocity/effort) for the telemetry plane. Hardware
        # nodes may omit either array (pad with None) — honest "not measured".
        n = len(msg.name)
        vel = list(msg.velocity) if len(msg.velocity) == n else [None] * n
        eff = list(msg.effort) if len(msg.effort) == n else [None] * n
        dyn = dict(self._dyn)
        for i, name in enumerate(msg.name):
            dyn[name] = (vel[i], eff[i])
        self._dyn = dyn                    # atomic rebind

    def _on_hand_joints(self, side: str, msg: JointState):
        # Remember this hand's joint-name ORDER so the sidecar Float64MultiArray
        # topics (temperatures/voltages, same order) can be mapped to names.
        self._hand_names[side] = list(msg.name)
        self._on_joints(msg)

    def _on_motor_health(self, msg):
        import json
        try:
            h = json.loads(msg.data)
        except Exception:
            return
        if isinstance(h, dict):
            mh = dict(self._motor_health)
            mh.update(h)
            self._motor_health = mh            # atomic rebind
            self._motor_health_t = time.time()

    def joint_dynamics(self) -> dict:
        """Everything the hardware measures beyond position, for the :9600
        telemetry plane. Keys are the ROS/MuJoCo joint names (the gateway maps
        them to its wire ids); absent values are None, never a guess.

            {"joints": {name: {"dq": v|None, "tau": e|None, "temp": C|None}},
             "torque": {"<side>_hand": bool|None},
             "voltages": {"<side>_hand": [V,...]|None}}
        """
        temps: dict = {}
        for side in cfg.HAND_SIDES:
            arr = self._hand_temps.get(side)
            names = self._hand_names.get(side) or []
            if arr and names and len(arr) == len(names):
                temps.update(zip(names, arr))
        dyn = self._dyn                     # one atomic read
        joints = {name: {"dq": v, "tau": e, "temp": temps.get(name)}
                  for name, (v, e) in dyn.items()}
        for name, t in temps.items():       # temp-only joints (no dyn yet)
            if name not in joints:
                joints[name] = {"dq": None, "tau": None, "temp": t}
        # Arm/body DaMiao health: temperature (JointState has no temp field for
        # these joints) + τ fallback if the driver ever omits effort.
        for name, h in self._motor_health.items():
            e = joints.setdefault(name, {"dq": None, "tau": None, "temp": None})
            if e.get("temp") is None and h.get("tempC") is not None:
                e["temp"] = h["tempC"]
            if e.get("tau") is None and h.get("torqueNm") is not None:
                e["tau"] = h["torqueNm"]
        return {"joints": joints,
                "torque": {f"{s}_hand": self._hand_torque.get(s)
                           for s in cfg.HAND_SIDES},
                "voltages": {f"{s}_hand": self._hand_volts.get(s)
                             for s in cfg.HAND_SIDES}}

    def _on_odom(self, msg: Odometry):
        self._wheel_odom_t = time.monotonic()
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._odom = (p.x, p.y, p.z, (q.w, q.x, q.y, q.z))

    def _on_visual_odom(self, msg: Odometry):
        # adopt camera odometry only while the wheel base is silent (>1 s)
        if time.monotonic() - self._wheel_odom_t < 1.0:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._odom = (p.x, p.y, p.z, (q.w, q.x, q.y, q.z))

    # ── plant interface ───────────────────────────────────────────────────────
    def read_state(self):
        """Write the latest measured state into the model and refresh kinematics.
        NO physics step -- the real robot is the integrator; the model is a mirror.

        The gateway flips set_mirror_kinematics_only(True) while the KINEMATIC
        stack (retarget / navigation) is active: those consumers only ever read
        geometry (mj_kinematics + mj_comPos), and the full mj_forward here was
        16% of ALL gateway samples on the control loop alone (py-spy 2026-07-13;
        the 50 Hz control-loop budget). The WBC torque modes need real dynamics, so
        the flag drops back to False whenever they engage."""
        self._mirror_into(self.data,
                          kinematics_only=self._mirror_kinematics_only)

    def set_mirror_kinematics_only(self, flag: bool) -> None:
        """Control-path mirror fidelity: True = geometry only (retarget/nav),
        False = full mj_forward (WBC torque modes). Cheap; called per tick."""
        self._mirror_kinematics_only = bool(flag)

    def mirror_into(self, data_fb):
        """Mirror the latest sensor cache into a SEPARATE data handle + refresh
        kinematics. Lets a decoupled, high-rate feedback thread read fresh joints/
        pose at the stream rate WITHOUT touching (or being gated by) the control
        loop's `data` — the fix for laggy joint/pose feedback in real:ros, where the
        control loop only samples state at its (slow) iteration rate.

        KINEMATICS ONLY: this handle exists purely to be READ (joint angles + body
        poses for the apps' 3D model and the telemetry plane) — nothing solves
        dynamics off it. See _mirror_into for why that distinction is worth 10% of
        the whole process."""
        self._mirror_into(data_fb, kinematics_only=True)

    def _mirror_into(self, data, kinematics_only=False):
        names, pos = self._joints                 # one atomic read of the cache
        self._mirror.apply(data, names, pos)
        if self._odom is not None:
            x, y, z, quat = self._odom
            mirror_base(self.model, data, x, y, z, quat, cfg.BASE_FREE_JOINT)
        if kinematics_only:
            # mj_forward runs the FULL forward-dynamics pipeline — collision
            # detection, constraint assembly, actuator/passive forces — none of
            # which a read-only mirror uses. Profiled 2026-07-13 (py-spy): it was
            # 27% of ALL samples in the gateway (16% on the control loop + 11% on
            # the 60 Hz feedback thread) and, being GIL-holding, it starved the
            # asyncio thread that ships robot_state/telemetry: the apps' joints
            # crawled at ~0.5 Hz. mj_kinematics (+ mj_comPos, which body/site FK
            # consumers expect) is all a mirror needs and is a fraction of the cost.
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
        else:
            mujoco.mj_forward(self.model, data)   # control path: real dynamics

    def send_lift_velocity(self, mps: float) -> None:
        """Publish the lift JOG velocity (m/s, +up) for the bridge's velocity mode. 0.0 =
        released (the bridge stops + drops torque). Called every control tick by the
        gateway from the drive stick; the bridge dedups so a steady speed is not re-planned."""
        from std_msgs.msg import Float64 as _Float64
        self._pub_lift_vel.publish(_Float64(data=float(mps)))

    def lift_feedback_age(self) -> float:
        """Seconds since a lift joint last arrived on any state topic (inf-ish at
        boot). The joint CACHE never forgets, so this stamp is the only honest
        "is the lift actually streaming" signal — the gateway seeds/opens its lift
        command gate only while this is fresh, and closes it when the lift leaves."""
        return time.time() - self._lift_fb_t

    def set_lift_command_enabled(self, enabled: bool) -> None:
        """Open/close the position /lift command gate (see __init__). Closed ->
        write() streams lift_pos=NaN ("no lift command"); the jog /lift/velocity
        path is unaffected. The gateway closes it at bring-up and opens it only
        once data.ctrl["Lift"] is seeded from measured feedback."""
        self._lift_cmd_enabled = bool(enabled)

    def write(self, robot_cmd):
        """Extract the controller's per-frame decision and publish it on the contract."""
        c = extract(self.model, self.data, robot_cmd, cfg)
        now = self.node.get_clock().now().to_msg()

        base = MabelSwerveCmd()
        base.vx, base.vy, base.omega = float(c.vx), float(c.vy), float(c.omega)
        base.lift_pos = float(c.lift) if self._lift_cmd_enabled else math.nan
        base.hardware_state = 1 if getattr(robot_cmd, "engaged", 1.0) == 0.0 else 0
        self._pub_base.publish(base)

        now_s = self.node.get_clock().now().nanoseconds * 1e-9
        arm_vel = self._arm_target_velocity(c.arms, now_s)
        # Remember the commanded pose for the gravity-FF evaluator (plant.py):
        # evaluating tau_g at the COMMANDED pose (bounded near measured) frees
        # the compliant PD from bridging the gravity DIFFERENCE to the target.
        self._last_arm_cmd = dict(c.arms)
        self._pub_arms.publish(self._js(now, c.arms, self.arm_effort, arm_vel))
        self._pub_body.publish(self._js(now, c.body, self._body_effort(c.body),
                                        self._body_target_velocity(c.body, now_s)))
        for side, joints in c.hands.items():
            self._pub_hands[side].publish(self._js(now, joints))

    def _body_effort(self, body: dict):
        """Torque feedforward for /body/command — the TORSO's gravity hold.

        PRIMARY (2026-07-29): the ARM-POSE-AWARE value the owner streams into
        ``self.body_effort`` (server/plant.py::ROSPlant — the shared arm+torso
        MJCF evaluation at the measured pose, live-trimmed via control_param.json
        torso.gravity_feedforward). The torso subtree carries both arms, so this
        is what lets kp drop to the compliant tune without the torso sagging the
        moment an arm extends.

        FALLBACK: the arm-pose-blind sinusoid τ_hold(θ)=A·sin(PHI−θ) from
        controller/mabel/torso_control (hardware-validated 2026-07-26 at the
        parked arm pose), evaluated at the MEASURED torso angle — covers
        body-only runs where the arms don't stream and MuJoCo-less hosts. The
        HAL only relays the number to the MIT wire (its layering rule:
        calibration + motor I/O, no control law). Without any feedforward the
        torso needs kp≈450 just to fight gravity — the "rigid and jittery"
        report."""
        if "torso" not in body:
            return None
        be = self.body_effort
        if be and "torso" in be and math.isfinite(be["torso"]):
            return {"torso": float(be["torso"])}
        if not hasattr(self, "_torso_hold_tau"):
            try:
                from mabel.torso_control import hold_tau, model_to_urdf
                self._torso_hold_tau = lambda m: hold_tau(model_to_urdf(m))
            except Exception:                     # controller pkg absent: no FF
                self._torso_hold_tau = None
        if self._torso_hold_tau is None:
            return None
        theta = float(body["torso"])
        try:
            theta = float(self.data.joint("Torso").qpos[0])   # measured (mirrored)
        except Exception:
            pass
        return {"torso": float(self._torso_hold_tau(theta))}

    def hand_torque_state(self):
        """Latched /<side>_hand/torque_state per side (None until the bridge has
        reported). Lets the gateway reconcile the operator's torque intent across
        bridge restarts — a restarted bridge boots the hands LIMP, and the
        edge-only e-stop handler alone would leave them limp forever."""
        return dict(self._hand_torque)

    def set_hand_torque(self, enabled, side=None):
        """Enable/disable torque on one hand (``side``) or ALL hands (side=None) by
        publishing Bool on /<side>_hand/torque_enable. Used to tie hand torque to the
        e-stop (disengaged -> True, engaged -> False). Idempotent; safe to call on
        every e-stop edge."""
        sides = [side] if side else list(self._pub_hand_torque.keys())
        for s in sides:
            pub = self._pub_hand_torque.get(s)
            if pub is not None:
                pub.publish(Bool(data=bool(enabled)))

    def publish_arm_gains(self, gains: dict) -> None:
        """Publish {"kp": {urdf joint: v}, "kd": {...}} for the HAL's per-command
        MIT gain overrides (arms_node._on_gains). Latched; call on change only."""
        import json as _json
        from std_msgs.msg import String as _String
        self._pub_arm_gains.publish(_String(data=_json.dumps(gains)))

    def set_arms_torque(self, enabled):
        """Enable/disable ARM torque by publishing Bool on /arms/torque_enable.
        Tied to the e-stop / teleop-start exactly like the hands (disengaged ->
        energise; engaged -> free-drive). Idempotent; the bridge hold-then-enables
        so there is no lurch."""
        self._pub_arms_torque.publish(Bool(data=bool(enabled)))

    def arms_torque_state(self):
        """Latched /arms/torque_state (None until the bridge reports) — lets the
        gateway reconcile the operator's torque intent across a bridge restart."""
        return self._arms_torque

    def set_neck_torque(self, enabled):
        """Enable/disable NECK (body) torque by publishing Bool on /body/torque_enable.
        Tied to the e-stop / teleop-start exactly like the arms + hands (disengaged ->
        energise; engaged -> limp). Idempotent; the bridge holds-in-place and the
        gateway resume_hold reseeds the measured pose, so there is no lurch. This is
        the ONLY thing that energises the neck — without it /body/command is inert."""
        self._pub_body_torque.publish(Bool(data=bool(enabled)))

    def neck_torque_state(self):
        """Latched /body/torque_state (None until body_node reports) — lets the
        gateway reconcile the operator's torque intent across a bridge restart (the
        neck boots torque-OFF, so the edge alone never re-fires after a restart)."""
        return self._body_torque

    @staticmethod
    def _js(stamp, name_to_pos, name_to_eff=None, name_to_vel=None):
        """JointState with position, an optional VELOCITY feedforward, and an optional
        effort FEEDFORWARD — the full MIT reference the firmware streams:
        `T = kp*(pos - q) + kd*(vel - dq) + t_ff`.

        `vel` is the target's JOINT VELOCITY. Without it the kd term is pure damping
        (`kd*(0 - dq)`) so a MOVING target (whole-body retargeting tracking the operator)
        is chased by kp alone and the arm TRAILS by the following error — which is why
        teleop feels sluggish while a joint slider (a far, static target) snaps fast. The
        sim's WBC carries vel/accel feedforward; sending it here makes the real arm track
        a moving VP pose like the sim. `effort` carries tau_g so kp*(pos-q) stays small
        (else the arm sags by tau_g/kp). MODEL frame — joint_group dir-corrects both the
        velocity and effort into raw motor frame. Names absent from a map get 0.0."""
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(name_to_pos.keys())
        msg.position = [float(v) for v in name_to_pos.values()]
        if name_to_vel:
            msg.velocity = [float(name_to_vel.get(n, 0.0)) for n in msg.name]
        if name_to_eff:
            msg.effort = [float(name_to_eff.get(n, 0.0)) for n in msg.name]
        return msg

    def _arm_target_velocity(self, arms, now_s):
        """Velocity FEEDFORWARD for the arm MIT: numerical d(target)/dt, lightly
        EMA-smoothed (the retarget IK output has per-frame jitter). Guards against a
        loop stall/gap cashing in as one huge FF spike."""
        prev = getattr(self, "_prev_arms", None)
        t0 = getattr(self, "_prev_arms_t", None)
        vel = getattr(self, "_arm_vel", {})
        self._prev_arms = dict(arms)
        self._prev_arms_t = now_s
        if prev is None or t0 is None:
            self._arm_vel = {n: 0.0 for n in arms}
            return self._arm_vel
        dt = now_s - t0
        if dt <= 1e-4 or dt > 0.2:                 # too small, or a gap → no FF this tick
            self._arm_vel = {n: 0.0 for n in arms}
            return self._arm_vel
        a = 0.4                                    # EMA on the raw diff
        vmax = 8.0                                  # rad/s FF clamp — a bad IK frame
        nv = {}                                     # must not spike kd*vel into the stop
        for n, p in arms.items():
            dq = p - prev.get(n, p)
            # STEP GUARD (2026-07-29, "robot shakes on the FIRST torque-on"):
            # a target DISCONTINUITY — resume_hold adopting the measured pose
            # over the boot ctrl, a mode-switch reseed — is a REBASE, not a
            # velocity. Differentiating it turned into a clamped ±8 rad/s FF
            # spike on every joint at once (kd·8 ≈ 20-30 N·m on the shoulders
            # for a few EMA frames = the whole-robot shake, first enable only
            # because later enables find ctrl already at measured). No
            # continuous command moves 0.15 rad in one ~22 ms tick (the glide
            # limiter caps increments well below that), so treat it as a seed.
            if abs(dq) > 0.15:
                nv[n] = 0.0
                continue
            raw = dq / dt
            ema = a * raw + (1.0 - a) * vel.get(n, 0.0)
            nv[n] = max(-vmax, min(vmax, ema))
        self._arm_vel = nv
        return nv

    def _body_target_velocity(self, body, now_s):
        """Velocity FEEDFORWARD for the TORSO's MIT loop, same construction as the
        arms' (numerical d(target)/dt, EMA, gap guard). At the compliant torso tune
        the kd term is what tracks a MOVING target — without target velocity it is
        pure damping and the torso trails by the following error (the studio's
        hold_pj learned this same lesson, 2026-07-29). Torso only: the neck's
        command path (set_gaze) has no velocity input."""
        if "torso" not in body:
            return None
        prev = getattr(self, "_prev_body", None)
        t0 = getattr(self, "_prev_body_t", None)
        vel = getattr(self, "_body_vel", 0.0)
        p = float(body["torso"])
        self._prev_body = p
        self._prev_body_t = now_s
        if prev is None or t0 is None:
            self._body_vel = 0.0
            return {"torso": 0.0}
        dt = now_s - t0
        if dt <= 1e-4 or dt > 0.2:                 # too small, or a gap → no FF this tick
            self._body_vel = 0.0
            return {"torso": 0.0}
        if abs(p - prev) > 0.15:                    # step guard: a rebase (resume_hold
            self._body_vel = 0.0                    # adoption) is not a velocity —
            return {"torso": 0.0}                   # same first-enable-shake fix as arms
        vmax = 2.0                                  # rad/s — the torso never moves faster
        ema = 0.4 * ((p - prev) / dt) + 0.6 * vel
        self._body_vel = max(-vmax, min(vmax, ema))
        return {"torso": self._body_vel}
