"""arms_node — the bimanual arms DDS adapter.

Thin plumbing over the facade's ``left_arm``/``right_arm`` write-leased handles.
It owns exactly two things — the DDS contract and the URDF name rename — and no
hardware logic whatsoever: the transport is ``mabel_hw.drivers.openarm``, the
calibration is ``mabel_hw.subsystems.joint_group``, and the control law is the
commander's (``controller/mabel/arm_control``).

Contract (topics FROZEN — tests/contract/contract.json):

  Subscribed:  <command>       sensor_msgs/JointState   MIT setpoint, MODEL frame
                                 position  rad          (required)
                                 velocity  rad/s        (optional feedforward)
                                 effort    N·m          (optional feedforward —
                                   e.g. the controller's gravity compensation)
               <torque_enable> std_msgs/Bool            engage / free-drive
  Published:   <joint_states>  sensor_msgs/JointState   pos/vel/effort, MODEL frame
               <torque_state>  std_msgs/Bool            (latched)
               .../motor_health  std_msgs/String        {joint:{torqueNm,tempC}} @2Hz
               .../joint_limits  std_msgs/String        {joint:[lo,hi]} (latched)

Joint names on the wire are URDF (``left_arm_1..7``); the firmware names
(``l_j1..l_j7``) never leave this module. Everything published is calibrated
model-frame radians, ready to use as-is.

Torque defaults OFF (limp); energising holds-in-place first (no lurch) — the
hands' rule.
"""
from __future__ import annotations

import json
import re

from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from ..errors import HardwareError
from ..types import JointCommand
from ._util import add_publisher_watchdog

_FW_NAME = re.compile(r"^([lr])_j(\d+)$")


def _name_maps(side: str, driver_names):
    """(dds->driver, driver->dds) for one arm. Firmware ``l_jN`` <-> URDF
    ``left_arm_N``; anything else (e.g. a gripper) passes through unchanged."""
    dds_to_drv, drv_to_dds = {}, {}
    for name in driver_names:
        m = _FW_NAME.match(name)
        dds = f"{side}_arm_{m.group(2)}" if m else name
        dds_to_drv[dds] = name
        drv_to_dds[name] = dds
    return dds_to_drv, drv_to_dds


class _SideIO:
    """One facade-backed arm: lease-held writer + name translation."""

    def __init__(self, node, side, handle, owner):
        self.node = node
        self.side = side
        self.handle = handle
        self.writer = handle.claim(owner, steal=False)
        self.dds_to_drv, self.drv_to_dds = _name_maps(side, handle.names)
        # Gain OVERRIDES from /arms/gains (driver-name keyed) — the app's
        # stiffness slider / soft-vs-stiff gain sets, applied per command via
        # JointCommand.kp/kd (the driver falls back to config defaults for any
        # joint absent from the map). This node stays transport-only: the gain
        # VALUES are computed controller-side (control_param.json arm.gains +
        # arm.soft_gains via the gateway).
        self.kp_over: dict = {}
        self.kd_over: dict = {}

    def _mine(self, msg_names, values):
        """The subset of a JointState field addressed to THIS arm, keyed by
        driver name. Empty when the field is absent (JointState allows that)."""
        if not values:
            return {}
        return {self.dds_to_drv[n]: float(v)
                for n, v in zip(msg_names, values) if n in self.dds_to_drv}

    def command(self, msg: JointState):
        position = self._mine(msg.name, msg.position)
        if not position:
            return
        cmd = JointCommand(position=position,
                           velocity=self._mine(msg.name, msg.velocity),
                           effort=self._mine(msg.name, msg.effort),
                           kp=dict(self.kp_over), kd=dict(self.kd_over))
        try:
            self.writer.command(cmd)
        except HardwareError as e:
            self.node.get_logger().warn(f"{self.side} arm write refused: {e}",
                                        throttle_duration_sec=2.0)

    def set_torque(self, on: bool):
        try:
            self.writer.set_torque(on)
        except HardwareError as e:
            self.node.get_logger().warn(f"{self.side} arm torque refused: {e}",
                                        throttle_duration_sec=2.0)

    def feedback(self):
        st = self.handle.state()
        if st is None:
            return [], [], [], []
        # PUBLISH ONLY LIVE JOINTS. The openarm driver returns the FULL joint vector
        # with zeros + live=False when the UDP/Ethernet link is stale or down
        # (OpenArmGroup.read(): fresh = st.age() < 1 s), it never goes empty. An
        # unfiltered feedback therefore streams phantom zeros to /arms/joint_states at
        # 50 Hz and a DISCONNECTED arm reads as "connected" — on the bench, in the apps,
        # and in real_run's [hw] liveness line. Drop the dead joints so the node's
        # `if not names: return` goes silent when the arms Ethernet is off. Mirrors
        # JointGroupSubsystem.positions()/health(), which already gate on `live`.
        live = getattr(st, "live", None)
        names, pos, vel, eff = [], [], [], []
        for i, n in enumerate(st.names):
            if live is not None and i < len(live) and not live[i]:
                continue
            names.append(self.drv_to_dds[n])
            pos.append(float(st.position[i]))
            vel.append(float(st.velocity[i]))
            eff.append(float(st.effort[i]))
        return names, pos, vel, eff

    def health(self):
        return {self.drv_to_dds.get(n, n): v
                for n, v in self.handle.health().items()}

    def limits(self):
        fn = getattr(self.handle, "commandable_ranges", None)
        if not callable(fn):
            return {}
        return {self.drv_to_dds.get(n, n): v for n, v in (fn() or {}).items()}

    @property
    def torque_enabled(self):
        st = self.handle.state()
        return bool(st is not None and st.torque_enabled)


def build(cfg, hw=None, **_):
    return ArmsNode(cfg, hw=hw)


class ArmsNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_arms")
        self.cfg = cfg
        self.sides = []

        if hw is not None:
            owner = f"dds:{cfg['topic_command']}"
            for side, handle in (("left", hw.left_arm), ("right", hw.right_arm)):
                if handle.available:
                    self.sides.append(_SideIO(self, side, handle, owner))
        if not self.sides:
            self.get_logger().warn(
                "arms: neither arm is up (MABEL_HW_LEFT_ARM / _RIGHT_ARM "
                "disabled, or the Teensy is silent) — node idle, will hot-join.")

        # NEWEST-ONLY command intake (2026-07-29, the hands' 1-2 s-backlog lesson —
        # hands_node.py): the old depth-10 RELIABLE queue buffered stale teleop
        # commands behind any executor stall, so the arms replayed history instead
        # of chasing the newest operator pose. Keep only the LATEST command.
        cmd_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, cfg["topic_command"], self._on_cmd,
                                 cmd_qos)
        self._pub = self.create_publisher(JointState, cfg["topic_joint_states"], 10)
        add_publisher_watchdog(self, cfg["topic_command"])

        # ADDITIVE gains plane (2026-07-30): the app's stiffness slider + the
        # soft/stiff gain sets, as String JSON {"kp": {dds_name: v}, "kd": {...}}.
        # Latched so a late-joining/restarted bridge adopts the current gains.
        gains_qos = QoSProfile(depth=1)
        gains_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, cfg.get("topic_gains", "/arms/gains"),
                                 self._on_gains, gains_qos)

        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, cfg["topic_torque"], self._on_torque, 10)
        self._pub_torque = self.create_publisher(Bool, cfg["topic_torque_state"],
                                                 latched)
        self._torque_state = None            # force the first publish
        self._publish_torque()

        base = cfg["topic_joint_states"].rsplit("/", 1)[0]
        # Motor health (torque N·m + MOSFET degC per joint) -> app telemetry: the
        # teleop bridge merges every subsystem's motor_health into
        # robot_state.motorHealth. Slow-changing → 2 Hz JSON is plenty.
        self._pub_health = self.create_publisher(String, base + "/motor_health", 5)
        self.create_timer(0.5, self._publish_health)
        # The calibrated commandable range per joint — the authority a teleop
        # slider clamps against (robot_state.jointLimits). Latched: it changes
        # only on recalibration, and a late client must still see it.
        self._pub_limits = self.create_publisher(String, base + "/joint_limits",
                                                 latched)
        self._limits = None
        self._publish_limits()

        self.create_timer(1.0 / max(cfg["publish_rate"], 1.0), self._stream)
        self.create_timer(1.0, self._service)      # reconnect / hot-join watchdog

        if self.sides:
            if cfg.get("auto_enable_torque", False):
                for s in self.sides:
                    s.set_torque(True)
            self.get_logger().info(
                "arms online: "
                + ", ".join(f"{s.side}({len(s.dds_to_drv)}j)" for s in self.sides))

    # ── commands ─────────────────────────────────────────────────────────────
    def _on_cmd(self, msg: JointState):
        for side in self.sides:
            side.command(msg)

    def _on_gains(self, msg: String):
        """Adopt gain overrides for BOTH sides. Values are validated + clamped
        to sane MIT ranges (kp [0, 500], kd [0, 5] — the DaMiao wire packing);
        malformed payloads are dropped whole (never half-apply a gain set)."""
        try:
            g = json.loads(msg.data)
            kp_in = {str(k): float(v) for k, v in (g.get("kp") or {}).items()}
            kd_in = {str(k): float(v) for k, v in (g.get("kd") or {}).items()}
        except Exception as e:                   # noqa: BLE001
            self.get_logger().warn(f"gains payload rejected: {e}",
                                   throttle_duration_sec=5.0)
            return
        for side in self.sides:
            side.kp_over = {side.dds_to_drv[n]: max(0.0, min(500.0, v))
                            for n, v in kp_in.items() if n in side.dds_to_drv}
            side.kd_over = {side.dds_to_drv[n]: max(0.0, min(5.0, v))
                            for n, v in kd_in.items() if n in side.dds_to_drv}
        n_kp = sum(len(s.kp_over) for s in self.sides)
        self.get_logger().info(f"arm gains adopted: {n_kp} kp overrides "
                               f"(kp span {min(kp_in.values(), default=0):.1f}"
                               f"–{max(kp_in.values(), default=0):.1f})",
                               throttle_duration_sec=1.0)

    def _on_torque(self, msg: Bool):
        for side in self.sides:
            side.set_torque(bool(msg.data))
        self._publish_torque()

    def _publish_torque(self):
        """Publish the ACTUAL torque state (latched). Republished from the 1 Hz
        service whenever it changes underneath us: a board that reboots resets its
        groups to limp by design, and a torque_enable issued before the board was
        streaming is refused — in both cases the operator must SEE that the arm
        went limp rather than trust a stale `True`."""
        on = bool(self.sides) and all(s.torque_enabled for s in self.sides)
        if on != self._torque_state:
            self._torque_state = on
            self._pub_torque.publish(Bool(data=on))

    # ── feedback ─────────────────────────────────────────────────────────────
    def _stream(self):
        names, pos, vel, eff = [], [], [], []
        for side in self.sides:
            n, p, v, e = side.feedback()
            names += n
            pos += p
            vel += v
            eff += e
        if not names:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position, js.velocity, js.effort = names, pos, vel, eff
        self._pub.publish(js)

    def _publish_health(self):
        h = {}
        for side in self.sides:
            h.update(side.health())
        if h:
            self._pub_health.publish(String(data=json.dumps(h)))

    def _publish_limits(self):
        """(Re)publish the calibrated ranges when they appear or change — the
        calibration store hot-reloads, and the arms resolve after a hot-join."""
        lim = {}
        for side in self.sides:
            lim.update(side.limits())
        if lim and lim != self._limits:
            self._limits = lim
            self._pub_limits.publish(String(data=json.dumps(lim)))

    def _service(self):
        for side in self.sides:
            side.handle.service()
        self._publish_limits()
        self._publish_torque()               # reality wins over a stale latch

    def destroy_node(self):
        try:
            for side in self.sides:
                side.writer.release()
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
ArmsSubsystem = ArmsNode
