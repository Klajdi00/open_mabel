"""body_node — torso + neck DDS adapter.

Contract (byte-identical to the pre-mabel_hw bridge; same topics the sim uses):

  Subscribed:  <command>       sensor_msgs/JointState   {torso, neck_yaw, neck_pitch,
                                                         neck_roll} targets [rad]
  Published:   <joint_states>  sensor_msgs/JointState   pos/vel/effort feedback
               <torque_state>  std_msgs/Bool (latched); .../motor_health @2Hz

Two modes:
  * hw= (bridge production): torso and neck route through their facade handles.
    The DDS neck names are TASK-space (yaw/pitch/roll); the serial neck maps them
    onto NeckWriter.set_gaze() 1:1 (direct drive, no linkage IK), and feedback
    publishes the same task names, so the contract is unchanged. A joint not yet
    wired (e.g. neck_yaw) simply reports None.
  * hw=None or neither handle up (today's default config): the LEGACY path —
    one DamiaoGroup over the BODY joint_motors map (torso + a direct-drive
    yaw/pitch/roll neck), wire-identical to the pre-refactor node.
"""

import json

from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from ..drivers import damiao_can as damiao
from ..errors import HardwareError, InfeasiblePose
from ._util import add_publisher_watchdog

_NECK_KEYS = ("neck_pitch", "neck_roll", "neck_yaw")


def build(cfg, hw=None, **_):
    return BodyNode(cfg, hw=hw)


class BodyNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_body")
        self.cfg = cfg
        self.group = None
        self._torso = self._neck = None
        self._torso_writer = self._neck_writer = None
        self._gaze = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        self._warned_no_task = False

        if hw is not None and (hw.torso.available or hw.neck.available):
            owner = f"dds:{cfg['topic_command']}"
            if hw.torso.available:
                self._torso = hw.torso
                self._torso_writer = hw.torso.claim(owner, steal=False)
            if hw.neck.available:
                self._neck = hw.neck
                self._neck_writer = hw.neck.claim(owner, steal=False)
                st = hw.neck.state()             # seed gaze from the real pose so
                if st is not None:               # the first command can't jump
                    self._gaze = {"pitch": st.pitch or 0.0,
                                  "roll": st.roll or 0.0,
                                  "yaw": st.yaw or 0.0}
            self.get_logger().info(
                f"body in mabel_hw mode (torso={self._torso is not None}, "
                f"neck={self._neck is not None}, "
                f"neck task-space={bool(self._neck and self._neck.task_space_available)}).")
        else:
            # LEGACY path — wire-identical to the pre-refactor node.
            joints = damiao.parse_joint_motors(
                cfg["joint_motors"], cfg["kp_default"], cfg["kd_default"],
                cfg["p_max"], cfg["v_max"], cfg["t_max"])
            self.group = damiao.DamiaoGroup(
                joints, logger=lambda m: self.get_logger().warn(m))
            if not damiao.CAN_OK:
                self.get_logger().warn(
                    f"python-can unavailable ({damiao.CAN_ERR}); body IDLE.")
            elif not joints:
                self.get_logger().warn(
                    "no joint_motors map (BODY_JOINT_MOTORS); body IDLE. e.g. "
                    "'torso:can0:1, neck_yaw:can0:2, neck_pitch:can0:3, neck_roll:can0:4'.")
            else:
                self.group.connect()

        # NEWEST-ONLY command intake (2026-07-29, same fix as arms_node/hands_node):
        # a depth-10 RELIABLE queue buffers stale teleop targets; the body must
        # always chase the newest one.
        cmd_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, cfg["topic_command"], self._on_cmd,
                                 cmd_qos)
        self._pub = self.create_publisher(JointState, cfg["topic_joint_states"], 10)
        add_publisher_watchdog(self, cfg["topic_command"])
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, cfg["topic_torque"], self._on_torque, 10)
        self._pub_torque = self.create_publisher(Bool, cfg["topic_torque_state"], latched)
        self._publish_torque()
        self.create_timer(1.0 / max(cfg["publish_rate"], 1.0), self._stream)
        # 1 Hz watchdog in BOTH modes: teensy health/reconnect, or dead-CAN-bus
        # rebuild after a hot-replug (DamiaoGroup.service).
        self.create_timer(1.0, self._service)
        # Motor HEALTH (torque Nm + MOSFET temp degC) → robot_state.motorHealth.
        health_topic = cfg["topic_joint_states"].rsplit("/", 1)[0] + "/motor_health"
        self._pub_health = self.create_publisher(String, health_topic, 5)
        self.create_timer(0.5, self._publish_health)
        self.get_logger().info("body online.")

    # ---- commands ----------------------------------------------------------------
    def _on_cmd(self, msg: JointState):
        targets = dict(zip(msg.name, msg.position))
        if self.group is not None:
            self.group.set_targets(targets)
            return
        if self._torso_writer is not None and "torso" in targets:
            try:
                # Pass the commander's torque + velocity feedforward through
                # untouched — the gravity/tracking MATH lives in controller/mabel
                # (evaluated gateway-side); this node is transport only. Velocity
                # rides the MIT kd term (kd*(vel−dq)) so a MOVING target tracks at
                # the compliant gains instead of trailing by the following error.
                eff = (dict(zip(msg.name, msg.effort))
                       if len(msg.effort) == len(msg.name) else {})
                vel = (dict(zip(msg.name, msg.velocity))
                       if len(msg.velocity) == len(msg.name) else {})
                if "torso" in eff:
                    from ..types import JointCommand          # noqa: PLC0415
                    self._torso_writer.command(JointCommand(
                        position={"torso": targets["torso"]},
                        velocity=({"torso": float(vel["torso"])}
                                  if "torso" in vel else {}),
                        effort={"torso": float(eff["torso"])}))
                else:
                    self._torso_writer.set_joint_positions(
                        {"torso": targets["torso"]})
            except HardwareError as e:
                self.get_logger().warn(f"torso write refused: {e}",
                                       throttle_duration_sec=2.0)
        neck_in = {k: targets[k] for k in _NECK_KEYS if k in targets}
        if neck_in and self._neck_writer is not None:
            if not self._neck.task_space_available:
                if not self._warned_no_task:
                    self._warned_no_task = True
                    self.get_logger().error(
                        "neck commands arrived but the RSSR kinematic model is "
                        "unavailable (hardware/neck_studio) — dropping "
                        "task-space neck targets.")
                return
            self._gaze["pitch"] = neck_in.get("neck_pitch", self._gaze["pitch"])
            self._gaze["roll"] = neck_in.get("neck_roll", self._gaze["roll"])
            self._gaze["yaw"] = neck_in.get("neck_yaw", self._gaze["yaw"])
            try:
                self._neck_writer.set_gaze(self._gaze["pitch"],
                                           self._gaze["roll"],
                                           self._gaze["yaw"])
            except InfeasiblePose as e:
                self.get_logger().warn(str(e), throttle_duration_sec=2.0)
            except HardwareError as e:
                self.get_logger().warn(f"neck write refused: {e}",
                                       throttle_duration_sec=2.0)

    def _on_torque(self, msg: Bool):
        on = bool(msg.data)
        # Operational breadcrumb (throttled): 2026-07-27 the live bridge showed
        # enables on the wire with torque_state never flipping — this line is the
        # ground truth for "did the callback actually run".
        self.get_logger().info(f"torque_enable received: {'ON' if on else 'OFF'}",
                               throttle_duration_sec=2.0)
        if self.group is not None:
            self.group.set_torque(on)
        else:
            for writer, what in ((self._torso_writer, "torso"),
                                 (self._neck_writer, "neck")):
                if writer is not None:
                    try:
                        writer.set_torque(on)
                    except HardwareError as e:
                        self.get_logger().warn(f"{what} torque refused: {e}",
                                               throttle_duration_sec=2.0)
        self._publish_torque()

    def _publish_torque(self):
        if self.group is not None:
            on = bool(self.group.is_torque_enabled)
        else:
            states = []
            for handle in (self._torso, self._neck):
                if handle is not None:
                    st = handle.state()
                    motors = getattr(st, "motors", st)   # NeckState nests motors
                    states.append(bool(motors and motors.torque_enabled))
            on = bool(states) and all(states)
        self._pub_torque.publish(Bool(data=on))

    # ---- feedback ------------------------------------------------------------------
    def _stream(self):
        if self.group is not None:
            names, pos, vel, eff = self.group.stream()
        else:
            names, pos, vel, eff = [], [], [], []
            # PUBLISH ONLY LIVE JOINTS on BOTH buses. The torso (DaMiao CAN) and neck
            # (Dynamixel serial pitch/roll + DaMiao yaw) drivers each return the FULL
            # joint vector with zeros + live=False when a motor/link is silent — they
            # never go empty. Unfiltered, a torso whose CAN is down or a neck whose
            # Dynamixel is unplugged would keep streaming phantom zeros to
            # /body/joint_states and read as "connected". Gate every appended joint on
            # its `live` flag so a dead bus goes silent (matches positions()/health()).
            if self._torso is not None:
                self._torso.tick()
                st = self._torso.state()
                if st is not None:
                    tlive = getattr(st, "live", None)
                    for i, n in enumerate(st.names):
                        if tlive is not None and i < len(tlive) and not tlive[i]:
                            continue
                        names.append(n); pos.append(st.position[i])
                        vel.append(st.velocity[i]); eff.append(st.effort[i])
            if self._neck is not None:
                self._neck.tick()
                st = self._neck.state()
                if st is not None:
                    m = st.motors
                    mlive = getattr(m, "live", None)
                    # motor name -> fresh telemetry? (default live if driver omits it)
                    _is_live = {n: (mlive is None or i >= len(mlive) or bool(mlive[i]))
                                for i, n in enumerate(m.names)}
                    # Direct-drive neck: task axis == its motor 1:1, so the task
                    # velocity/effort ARE the motor's dir-corrected velocity/effort
                    # (subsystems/joint_group.state()). Map by motor name so the neck
                    # reports pos+vel+torque like every other joint (temperature rides
                    # /body/motor_health, same as the arms — JointState has no temp).
                    _vel = dict(zip(m.names, m.velocity))
                    _eff = dict(zip(m.names, m.effort))
                    if st.pitch is not None:      # FK up: task-space names (the
                        # contract) — but publish an axis only when its DRIVING motor
                        # is live, so an unplugged neck Dynamixel (pitch/roll) or DaMiao
                        # (yaw) drops out honestly instead of streaming a stale 0.
                        for mn, val in (("neck_yaw", st.yaw or 0.0),
                                        ("neck_pitch", st.pitch),
                                        ("neck_roll", st.roll or 0.0)):
                            if _is_live.get(mn):
                                names.append(mn); pos.append(val)
                                vel.append(_vel.get(mn, 0.0))
                                eff.append(_eff.get(mn, 0.0))
                    else:                         # model absent: motor-space names
                        if not self._warned_no_task:
                            self._warned_no_task = True
                            self.get_logger().warn(
                                "neck FK unavailable — publishing motor-space "
                                "names (motor_L1/L2/head) instead of task names.")
                        for i, mn in enumerate(m.names):
                            if not _is_live.get(mn):
                                continue
                            names.append(mn); pos.append(m.position[i])
                            vel.append(m.velocity[i]); eff.append(m.effort[i])
        if not names:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position, js.velocity, js.effort = names, pos, vel, eff
        self._pub.publish(js)

    def _publish_health(self):
        if self.group is not None:
            h = self.group.health()
        else:
            h = {}
            for handle in (self._torso, self._neck):
                if handle is not None:
                    h.update(handle.health())
        if h:
            self._pub_health.publish(String(data=json.dumps(h)))

    def _service(self):
        # A driver fault here (e.g. a serial open failing mid hot-replug) must NEVER
        # propagate through this ROS timer callback — that took down the ENTIRE bridge
        # on a neck reconnect. Isolate each bus so one subsystem's transient error is
        # logged and retried next tick, never fatal.
        if self.group is not None:
            try:
                self.group.service()
            except Exception as e:                       # noqa: BLE001
                self.get_logger().warn(f"body service error (will retry): {e}",
                                       throttle_duration_sec=5.0)
            return
        for handle, what in ((self._torso, "torso"), (self._neck, "neck")):
            if handle is not None:
                try:
                    handle.service()
                except Exception as e:                   # noqa: BLE001
                    self.get_logger().warn(
                        f"{what} service error (will retry): {e}",
                        throttle_duration_sec=5.0)

    def destroy_node(self):
        try:
            if self.group is not None:
                self.group.shutdown()
            for writer in (self._torso_writer, self._neck_writer):
                if writer is not None:
                    writer.release()
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
BodySubsystem = BodyNode
