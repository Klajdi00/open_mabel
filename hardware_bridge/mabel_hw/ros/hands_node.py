"""hands_node — ORCA hands DDS adapter (Feetech STS/SCS over TTL serial).

Thin plumbing over the controller layer (mabel_hw.drivers.hands) / the facade's
hand handles. ONE node owns BOTH hands. A side whose port is "none" / unpowered
advertises its topics but stays idle (never fatal), and is auto-reconnected
when the adapter returns.

Per-side namespaced contract (unchanged, so teleop / sim / the server are unaffected):

  Subscribed:  /<side>_hand/command        sensor_msgs/JointState  targets [rad]
               /<side>_hand/torque_enable   std_msgs/Bool          engage / free-drive
  Published:   /<side>_hand/joint_states    sensor_msgs/JointState  pos/vel/effort
               /<side>_hand/torque_state    std_msgs/Bool   (latched) engaged?
               /<side>_hand/temperatures    std_msgs/Float64MultiArray  per-joint degC
               /<side>_hand/voltages        std_msgs/Float64MultiArray  per-joint V
               /<side>_hand/load            std_msgs/Float64MultiArray  per-joint PWM duty [-1,1]

JointState.effort is REAL TORQUE in N·m (ROS's contract for that field), measured
from the servo's Present_Current register (τ = Kt·I). It used to carry the servo's
normalised PWM *duty* instead — a [-1,1] number in a newton-metre field, which put
the hands on the same axis as the arms' true N·m in every plot and made hand
"torque" meaningless. That duty is still useful (it saturates at ±1 when a motor is
giving everything it has), so it now rides its own `/load` topic. A servo with no
current register publishes NO effort at all rather than a fake zero.

ROS only commands ALREADY-calibrated joint angles [rad]; end-stop calibration is
the ROS-free bench step (hand studio /:8779 RPC, or
``python3 -m mabel_hw.drivers.orcahand.calibrate --model <side>``) whose saves land in
calibration_data/hands/ + the legacy sync targets.

Names may be bare ("index_mcp") or side-prefixed ("left_index_mcp"); published
names are always side-prefixed AND URDF-native: the ORCA<->URDF thumb rename
below is the ROS contract boundary (the mabel_hw API itself speaks ORCA names).
"""

import json

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import (Bool, Float64MultiArray, MultiArrayDimension,
                          MultiArrayLayout, String)

from .. import paths
from ..errors import HardwareError

# ORCA firmware thumb joint names -> URDF/mabel_description names — the CANONICAL ROS
# contract. Only the thumb differs (fingers abd/mcp/pip + the wrist already agree). The
# mp<->ip swap is hardware-validated (otherwise the URDF thumb showed proximal/tip
# swapped). Emitting URDF names HERE, at the source, makes /<side>_hand/joint_states the
# SINGLE canonical hand state — robot_state_publisher, the controller mirror, and the
# app all consume it directly.
_THUMB_TO_URDF = {
    "thumb_cmc": "thumb_mcp",   # carpometacarpal -> URDF base
    "thumb_mp":  "thumb_dip",   # ORCA metacarpophalangeal -> URDF distal (validated swap)
    "thumb_ip":  "thumb_pip",   # ORCA interphalangeal     -> URDF middle (validated swap)
    # thumb_abd is identical on both sides.
}
# Inverse, for command intake (URDF name in -> ORCA name the bus expects).
_URDF_TO_ORCA = {v: k for k, v in _THUMB_TO_URDF.items()}


def build(cfg, hw=None, **_):
    return HandsNode(cfg, hw=hw)


def _resolve_config(side: str, cfg) -> str:
    """Path to ``<side>_hand.yaml`` if a config_dir override is set, else "" so the
    controller falls back to the vendored packaged config."""
    override = cfg.get("config_dir") or ""
    if not override:
        return ""
    import os
    return os.path.join(paths.hand_config_dir(override), f"{side}_hand.yaml")


class _HandIO:
    """Per-side ROS plumbing around one controller ``Hand`` (+ optional facade
    writer — commands go through the write lease when the bridge shares hw)."""

    def __init__(self, node: Node, side: str, hand, writer=None):
        self.node = node
        self.side = side
        self.hand = hand
        self.writer = writer
        self.frame_id = f"{side}_hand"
        self._prefix = f"{side}_"

        ns = f"/{side}_hand"
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub_js = node.create_publisher(JointState, f"{ns}/joint_states", 10)
        self.pub_temp = node.create_publisher(Float64MultiArray, f"{ns}/temperatures", 10)
        self.pub_volt = node.create_publisher(Float64MultiArray, f"{ns}/voltages", 10)
        # PWM duty [-1,1] — the servo's DRIVE effort, evicted from JointState.effort
        # (which is now real N·m). Saturation (|load| → 1) = the motor is maxed out.
        self.pub_load = node.create_publisher(Float64MultiArray, f"{ns}/load", 10)
        # Uniform motor-health JSON (load % + temp + volts per joint) — same shape
        # as /arms/motor_health + /body/motor_health; ~2 Hz (throttled).
        self.pub_health = node.create_publisher(String, f"{ns}/motor_health", 5)
        self._health_tick = 0
        self.pub_torque = node.create_publisher(Bool, f"{ns}/torque_state", latched)

        # NEWEST-ONLY command intake. The Feetech goal write shares the bus lock with
        # the feedback read, so a slow/timing-out read stalls the write; with the old
        # depth-10 queue that backed up ~10 stale commands = the measured 1-2 s of hand
        # latency. Keep only the LATEST target and drop the backlog — the hand always
        # chases the newest operator pose, so latency is one write-wait, never a queue.
        cmd_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(JointState, f"{ns}/command", self._on_command, cmd_qos)
        node.create_subscription(Bool, f"{ns}/torque_enable", self._on_torque, 10)

        self._publish_torque_state()

    # ---- command intake -----------------------------------------------------
    def _on_command(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        # Accept URDF *or* ORCA names: map the URDF thumb names back to the ORCA
        # firmware names the bus expects. Low latency: straight to bus.
        targets = {_URDF_TO_ORCA.get(n, n): p for n, p in zip(msg.name, msg.position)}
        try:
            if self.writer is not None:
                self.writer.set_joint_positions(targets)
            else:
                self.hand.set_positions(targets)
        except HardwareError as e:
            self.node.get_logger().warn(f"{self.side} hand write refused: {e}",
                                        throttle_duration_sec=2.0)

    def _on_torque(self, msg: Bool):
        # Loud like body_node's "torque_enable received" — the one line that proves a
        # studio/app torque edge reached THIS hand (and what the servos answered).
        want = bool(msg.data)
        log = self.node.get_logger()
        try:
            if self.writer is not None:
                self.writer.set_torque(want)
            else:
                self.hand.set_torque(want)
        except HardwareError as e:
            log.warn(f"{self.side} hand torque refused: {e}", throttle_duration_sec=2.0)
        got = self.hand.is_torque_enabled()
        log.info(f"[{self.side}] torque_enable received: {'ON' if want else 'OFF'} -> "
                 f"torque_state {'ON' if got else 'OFF'}"
                 + ("" if got == want else
                    "  (hand idle / not connected)" if self.hand.hand is None else
                    "  (MISMATCH — see the driver log above)"))
        self._publish_torque_state()

    # ---- feedback -----------------------------------------------------------
    def _publish_torque_state(self):
        self.pub_torque.publish(Bool(data=self.hand.is_torque_enabled()))

    def publish_feedback(self):
        state = self.hand.get_state()
        if state is None:
            return
        names, pos, vel, tau, load, temp, volt = [], [], [], [], [], [], []
        for jn in self.hand.joint_names:
            if jn not in state.position:        # dropped this cycle: omit, don't zero
                continue
            # Publish the URDF/canonical name (thumb remapped); internal lookups
            # stay on the ORCA name `jn`.
            names.append(self._prefix + _THUMB_TO_URDF.get(jn, jn))
            pos.append(state.position[jn])
            vel.append(state.velocity.get(jn, 0.0))
            tau.append(state.torque.get(jn, 0.0))       # N·m (measured, τ = Kt·I)
            load.append(state.effort.get(jn, 0.0))      # PWM duty [-1, 1]
            temp.append(state.temperature.get(jn, 0.0))
            volt.append(state.voltage.get(jn, 0.0))
        if not names:
            return
        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.header.frame_id = self.frame_id
        js.name, js.position, js.velocity = names, pos, vel
        # effort = N·m, per the JointState contract. If this servo has no current
        # register, `state.torque` is EMPTY — publish an empty effort array rather
        # than a column of zeros that reads like "measured: no load".
        js.effort = tau if state.torque else []
        self.pub_js.publish(js)
        self.pub_temp.publish(self._farray(names, temp))
        self.pub_volt.publish(self._farray(names, volt))
        self.pub_load.publish(self._farray(names, load))
        self._health_tick += 1
        if self._health_tick >= 25:                  # 50 Hz stream -> ~2 Hz health
            self._health_tick = 0
            # torqueNm matches the arms' motor_health schema (drivers/openarm.health),
            # so one health view speaks the same units across the whole robot.
            self.pub_health.publish(String(data=json.dumps({
                n: {"torqueNm": round(q, 3), "loadPct": round(e * 100.0, 1),
                    "tempC": round(t, 1), "voltage": round(v, 1)}
                for n, q, e, t, v in zip(names, tau, load, temp, volt)})))

    @staticmethod
    def _farray(names, values) -> Float64MultiArray:
        m = Float64MultiArray()
        m.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label=",".join(names), size=len(values),
                                     stride=len(values))],
            data_offset=0)
        m.data = [float(v) for v in values]
        return m


class HandsNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_hands")
        self.cfg = cfg
        log = self.get_logger()
        self._owns_controller = False

        if hw is not None and hw._hands_controller is not None:
            self.controller = hw._hands_controller     # facade-owned
            writers = {side: getattr(hw, f"{side}_hand").claim(
                f"dds:/{side}_hand/command", steal=False)
                for side in ("left", "right")}
            log.info("hands in mabel_hw mode (write-leased).")
        else:
            from ..drivers.hands import HandsController
            self.controller = HandsController(
                left_port=cfg["left_port"], right_port=cfg["right_port"],
                baudrate=int(cfg["baudrate"]),
                auto_enable_torque=bool(cfg["auto_enable_torque"]),
                left_config=_resolve_config("left", cfg),
                right_config=_resolve_config("right", cfg),
                reconnect_interval=float(cfg.get("reconnect_interval", 2.0)),
                adapter_serials=cfg.get("adapter_serials") or {},
                logger=lambda m: log.info(m))
            self._owns_controller = True
            writers = {"left": None, "right": None}

        self.io = [_HandIO(self, side, self.controller.hand(side),
                           writer=writers[side])
                   for side in ("left", "right")]

        self.create_timer(1.0 / max(cfg["publish_rate"], 1.0), self._stream)
        # Low-rate watchdog: reconnect a side that was absent at startup or got
        # unplugged (off the fast command path).
        self.create_timer(1.0, self.controller.service)
        log.info("hands online (left + right).")

    def _stream(self):
        for io in self.io:
            io.publish_feedback()

    def destroy_node(self):
        try:
            for io in self.io:
                if io.writer is not None:
                    io.writer.release()
            if self._owns_controller:
                self.controller.close()
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
HandsSubsystem = HandsNode
