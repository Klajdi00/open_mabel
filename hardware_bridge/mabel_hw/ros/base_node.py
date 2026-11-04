"""base_node — mobile-base DDS adapter (validated ps5_teleop swerve path).

Thin plumbing over :class:`mabel_hw.subsystems.base.BaseSubsystem`. Contract
(byte-identical to the pre-mabel_hw bridge):

  Subscribed:
    <cmd_vel>  geometry_msgs/Twist  body twist in the REAL frame (+x fwd, +y left)
    <estop>    std_msgs/Bool        latched e-stop (True = zero drive)
  Published:
    <joint_states>  sensor_msgs/JointState  Teensy encoder readback (6 swerve joints)

Optionally also owns DIRECT PS5 pad teleop (no ROS in the loop), arbitrated
estop > pad (deadman held) > ROS cmd_vel — the pad and the DDS intake share the
node's ONE write lease, so the arbitration is node policy, never two writers.

DECLARED EXCEPTION — the pad path is NOT tip-gated
--------------------------------------------------
Every other base twist passes `mabel.motion_model.safe_base_step` before the swerve
IK (see controller/tests/test_base_twist_gate.py). `_pad_loop` does not: it is a raw
`stick x vmax`, with no lift-aware speed/accel caps and no ZMP clamp. Measured
against the model, full stick at the BASE_VMAX=1.0 default commands

    1.25x the tip-safe forward cap with the lift LOWERED   (0.800 m/s)
    2.14x the tip-safe forward cap with the lift RAISED    (0.467 m/s)

This is deliberate and bounded, not an oversight:
  * the pad exists precisely to survive ROS/network/gateway being down, so routing
    it through the gate (which lives in controller/, above this layer) would defeat
    its purpose — and importing the control law here would break the rule that the
    HAL does calibration + read + write ONLY;
  * it is OFF by default (config.py: joystick disabled unless configured);
  * it requires a physically-held deadman, and releasing it zeroes drive.

If you enable it, drive with the lift DOWN and treat BASE_VMAX as the real limit.
The principled fix (a pad teleop client that publishes /cmd_vel and is gated by
cmd_vel_bridge, losing the no-ROS property) is an open architectural choice, not a
bug to be quietly patched here.
"""

import threading
import time

from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState

from ..drivers import joystick as js
from ..subsystems.base import BaseSubsystem as CoreBase
from ._util import add_publisher_watchdog


def build(cfg, hw=None, **_):
    return BaseNode(cfg, hw=hw)


class BaseNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_base")
        self.cfg = cfg
        log = lambda m: self.get_logger().info(m)      # noqa: E731

        if hw is not None and hw.base.available:
            self.core = hw.base._sub                   # facade-owned subsystem
            self._writer = hw.base.claim(
                f"dds:{cfg['topic_cmd_vel']}", steal=False)
        else:
            self.core = CoreBase(cfg, logger=log)      # standalone: own the base
            self._writer = None

        self._pad_until = 0.0     # monotonic deadline while the pad holds priority

        self.create_subscription(Twist, cfg["topic_cmd_vel"], self._on_cmd_vel, 10)
        self.create_subscription(Bool, cfg["topic_estop"], self._on_estop, 10)
        self._js_pub = self.create_publisher(JointState, cfg["topic_joint_states"], 10)
        add_publisher_watchdog(self, cfg["topic_cmd_vel"])

        # Fixed-rate streamer feeds the Teensy watchdog; a dropped cmd_vel zeros drive.
        self.create_timer(1.0 / max(cfg["stream_rate"], 1.0), self.core.tick)
        self.create_timer(1.0 / max(cfg["joint_state_rate"], 1.0), self._publish_js)
        # Low-rate link watchdog: auto-reconnect when the Teensy returns (cable
        # re-plug, Teensy reboot, or link down at startup). Off the fast path.
        self.create_timer(0.5, self.core.service)

        if cfg["joystick"]:
            threading.Thread(target=self._pad_loop, daemon=True).start()
            self.get_logger().info(f"direct PS5 teleop on {cfg['joystick_dev']} "
                                   f"(hold deadman idx {js.BTN_DEADMAN} to move).")

        self.get_logger().info(
            f"base online: cmd_vel={cfg['topic_cmd_vel']} -> Teensy, "
            f"joint_states={cfg['topic_joint_states']} "
            f"(host_calib={cfg['apply_host_calib']}).")

    # ---- the ONE write path (leased in bridge mode) ---------------------------
    def _twist(self, vx, vy, wz):
        if self._writer is not None:
            self._writer.set_twist(vx, vy, wz)         # writer also ticks (push NOW)
        else:
            self.core.set_twist(vx, vy, wz)
            self.core.tick()

    # ---- ROS command intake ---------------------------------------------------
    def _on_cmd_vel(self, msg: Twist):
        # Pad has priority while its deadman is held (recent pad twist within 0.2 s).
        if time.monotonic() < self._pad_until:
            return
        self._twist(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_estop(self, msg: Bool):
        # estop is lease-FREE: it must never wait on (or be blocked by) a writer.
        self.core.set_estop(bool(msg.data))
        self.core.tick()   # estop takes effect immediately, not on the next tick

    def _publish_js(self):
        out = self.core.swerve.joint_state()
        if out is None:
            return
        names, position, velocity = out
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = names
        m.position = position
        m.velocity = velocity
        self._js_pub.publish(m)

    # ---- direct PS5 pad teleop (optional) --------------------------------------
    def _pad_loop(self):
        try:
            pad = js.DualSense(self.cfg["joystick_dev"]).start()
        except Exception as e:
            self.get_logger().warn(f"pad open failed ({e}); ROS-only base control.")
            return
        vmax, wmax = self.cfg["vmax"], self.cfg["wmax"]
        dt = 1.0 / 50.0
        while True:
            tic = time.time()
            if pad.button(js.BTN_DEADMAN):
                # Stick -> body twist (ps5_teleop signs): up=+x fwd, left=+y, CCW yaw.
                vx = -pad.axis(js.AX_LY) * vmax
                vy = -pad.axis(js.AX_LX) * vmax
                wz = -pad.axis(js.AX_RX) * wmax
                self._twist(vx, vy, wz)
                self._pad_until = time.monotonic() + 0.2   # hold priority over ROS
            while time.time() - tic < dt:
                time.sleep(0.001)

    def destroy_node(self):
        try:
            if self._writer is not None:
                self._writer.release()
            else:
                self.core.close()          # standalone: we own the base
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
BaseSubsystem = BaseNode
