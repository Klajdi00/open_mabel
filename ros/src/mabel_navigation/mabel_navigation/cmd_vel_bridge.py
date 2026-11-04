"""Bridge Nav2's (and teleop's) velocity command toward the MABEL swerve base.

Nav2 / keyboard teleop publish a body-frame velocity on ``/cmd_vel``. The MABEL
base accepts ``mabel_interfaces/MabelSwerveCmd`` on ``/mabel_cmd``. Because the
base is holonomic (swerve), the mapping is direct:

    cmd_vel.linear.x  -> vx      (forward)
    cmd_vel.linear.y  -> vy      (strafe; non-zero only because we're holonomic)
    cmd_vel.angular.z -> omega   (yaw)

The commanded twist is passed through the robot's tip-safe MOTION MODEL
(``motion_model.safe_base_step``): per-axis speed/accel caps that shrink with
lift height + a ZMP support-polygon clamp, so neither Nav2 nor a human can
command a motion that tips the robot. This is the same model the whole-body
controller uses, and it runs identically on the real robot -- its only extra
input is the lift height, read from ``/joint_states`` as CASCADE-TOTAL metres
(``lift_lower`` + ``lift_upper`` in sim/URDF, or the bridge's single ``lift``
joint on the real robot -- see ``_on_js``), which is a real sensor, NOT sim
ground truth.

A watchdog drives the target to zero if no ``/cmd_vel`` arrives within
``cmd_timeout`` seconds (the motion model then ramps the base safely to a stop).

Nav2 (Jazzy) can publish either ``geometry_msgs/Twist`` or
``geometry_msgs/TwistStamped`` depending on ``enable_stamped_cmd_vel``; set the
``use_stamped`` parameter to match.

WHERE THE OUTPUT GOES (``cmd_out_topic``, default ``/nav/mabel_cmd``)
--------------------------------------------------------------------
This node does **not** publish the canonical ``/mabel_cmd``. The gateway's plant
(``ROSPlant`` on real, ``SimRosPlant`` in sim) is the single publisher of
``/mabel_cmd``; it subscribes ``/nav/mabel_cmd`` and re-injects this twist through
the controller facade as a synthetic joystick. Consequences, all of them the point:

  * ONE publisher on ``/mabel_cmd`` -- two publishers fight and the base stalls.
  * Nav2's motion is gated by the same control law + tip model as every teleop
    device, instead of being a second, parallel path to the hardware.
  * The operator's stick takes over from Nav2 for free (the gateway marks nav
    inactive on human deflection) -- that IS the estop > teleop > nav arbitration.

The motion-model pass below therefore runs *in addition to* the facade's, not
instead of it: it keeps this node's own output tip-safe even if consumed directly.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry

from mabel_interfaces.msg import MabelSwerveCmd

from mabel_navigation import motion_model as mm


class CmdVelBridge(Node):
    def __init__(self, **kwargs):
        # **kwargs forwards rclpy Node options (notably `parameter_overrides`) so the
        # node can be constructed with a given config in-process — tests must be able
        # to assert the SAFETY parameters behave, without launching a whole stack.
        super().__init__('cmd_vel_bridge', **kwargs)

        self.declare_parameter('lift_pos', 0.0)
        self.declare_parameter('cmd_timeout', 0.5)      # seconds
        self.declare_parameter('use_stamped', False)    # match Nav2's cmd_vel type
        # Safety clamps. These bound the Nav2 command; the tip-safe motion model
        # (safe_base_step) is the REAL, lift-dependent speed/accel cap below.
        self.declare_parameter('max_vx', 0.8)           # clamp for safety (m/s)
        self.declare_parameter('max_vy', 0.7)           # m/s
        self.declare_parameter('max_omega', 2.0)        # rad/s
        self.declare_parameter('use_motion_model', True)
        self.declare_parameter('rate', 50.0)            # Hz; motion-model step rate
        # --- NAV-FRAME FLIP (front = -X) -------------------------------------
        # Nav2 plans/controls in `base_link_nav`, a frame rotated 180 deg about z from
        # base_link, so that its +x axis points along the robot's TRUE FRONT (base_link
        # -X; see mabel_full.xml / config.py). That lets the STOCK MPPI (with its normal
        # forward-preference + coordinated rotation) make the FRONT lead -- no external
        # yaw override, so MPPI's footprint collision check stays valid. This node is the
        # only adapter back to the physical base_link:
        #   * /cmd_vel (base_link_nav twist) -> negate vx,vy -> base_link -> swerve.
        #   * /odom (base_link twist) -> negate vx,vy -> /odom_nav (base_link_nav) so
        #     MPPI's velocity feedback matches the frame it controls in (else it reads
        #     its own motion as reversed and oscillates).
        # The 180 deg rotation about z is exactly (vx,vy)->(-vx,-vy); wz is unchanged.
        self.declare_parameter('nav_frame_flip', True)
        # --- WHERE NAV'S TWIST GOES (the single-publisher rule) ---------------
        # NOT the canonical /mabel_cmd. The gateway's plant (ROSPlant / SimRosPlant)
        # is the ONLY publisher of /mabel_cmd; it consumes this topic and re-injects
        # the twist through the controller facade, so Nav2's motion is gated by the
        # SAME control law + tip model as every teleop device, and the operator's
        # stick can take over (sim_teleop_bridge: human deflection -> nav inactive).
        # Publishing straight to /mabel_cmd would make two publishers fight for the
        # base -- the bug this parameter exists to prevent.
        self.declare_parameter('cmd_out_topic', '/nav/mabel_cmd')
        # Name of the real bridge's single total-height lift joint (mabel_hw
        # LIFT_JOINT). Sim/URDF instead expose the two cascade stages
        # (lift_lower + lift_upper), which _on_js sums. Keep in sync with
        # hardware_bridge/mabel_hw/config.py LIFT_JOINT.
        self.declare_parameter('lift_joint', 'lift')

        self.lift_pos = float(self.get_parameter('lift_pos').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        use_stamped = bool(self.get_parameter('use_stamped').value)
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_omega = float(self.get_parameter('max_omega').value)
        self.use_motion_model = bool(self.get_parameter('use_motion_model').value)
        # `use_motion_model:=false` turns the ONLY lift-aware tip gate on the Nav2
        # path into a runtime no-op — a launch-file typo away from an ungated base.
        # Keep it as a knob (the stability harness needs an un-gated baseline to
        # measure against) but make it deliberate and impossible to do by accident:
        # it now requires an explicit env opt-in, and it is never quiet.
        if not self.use_motion_model:
            import os
            if os.getenv('MABEL_ALLOW_UNGATED_BASE') != '1':
                self.use_motion_model = True
                self.get_logger().error(
                    'use_motion_model:=false IGNORED — the tip-safe motion model is '
                    'the gate that stops Nav2 commanding a motion that tips the '
                    'robot. To really disable it (benchmarks only, never on the '
                    'robot) set MABEL_ALLOW_UNGATED_BASE=1.')
            else:
                self.get_logger().error(
                    'TIP-SAFE MOTION MODEL DISABLED (MABEL_ALLOW_UNGATED_BASE=1). '
                    'The base is UNGATED: no lift-aware speed/accel caps, no ZMP '
                    'clamp. Benchmarks only — never drive the real robot like this.')
        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / max(self.rate, 1.0)
        self.nav_frame_flip = bool(self.get_parameter('nav_frame_flip').value)
        self.cmd_out_topic = str(self.get_parameter('cmd_out_topic').value)
        self.lift_joint = str(self.get_parameter('lift_joint').value)

        self.pub = self.create_publisher(MabelSwerveCmd, self.cmd_out_topic, 10)
        # Velocity feedback for MPPI, expressed in the nav frame it controls in.
        self._odom_pub = self.create_publisher(Odometry, '/odom_nav', 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        if use_stamped:
            self.create_subscription(TwistStamped, '/cmd_vel', self._on_stamped, 10)
        else:
            self.create_subscription(Twist, '/cmd_vel', self._on_twist, 10)
        # Lift height for the tip-safe model (real sensor; defaults to 0 if absent).
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        # latest target + ramped output state
        self._tgt = (0.0, 0.0, 0.0)
        self._vel = (0.0, 0.0, 0.0)
        self._lift = 0.0
        # Whether a lift reading was ever observed. 0.0 means "fully lowered" =
        # the MOST generous tip caps, so silently defaulting to it when the joint
        # is simply absent from /joint_states is a fail-OPEN safety hole. On the
        # real robot /lift/joint_states is not in joint_state_publisher's
        # source_list today, so this stays False and _step() warns.
        self._lift_seen = False
        self._last_cmd_time = self.get_clock().now()
        self._got_cmd = False

        self.create_timer(self.dt, self._step)

        self.get_logger().info(
            f'cmd_vel bridge up: /cmd_vel ({"TwistStamped" if use_stamped else "Twist"}) '
            f'-> {self.cmd_out_topic} | motion_model={"on" if self.use_motion_model else "off"} '
            f'@ {self.rate:.0f} Hz, lift_pos={self.lift_pos}, timeout={self.cmd_timeout}s')

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _on_stamped(self, msg: TwistStamped):
        self._set_target(msg.twist)

    def _on_twist(self, msg: Twist):
        self._set_target(msg)

    def _set_target(self, twist: Twist):
        self._tgt = (
            self._clamp(float(twist.linear.x), -self.max_vx, self.max_vx),
            self._clamp(float(twist.linear.y), -self.max_vy, self.max_vy),
            self._clamp(float(twist.angular.z), -self.max_omega, self.max_omega),
        )
        self._last_cmd_time = self.get_clock().now()
        self._got_cmd = True

    def _on_js(self, msg: JointState):
        """Track the lift height the tip model needs — in CASCADE-TOTAL metres.

        `safe_base_step`'s z_lift is "metres of cascaded lift travel", LIFT_RANGE
        (0, 0.635) -- the WHOLE column, not one stage. Two namings reach us:

          * sim / URDF : `lift_lower` + `lift_upper`, each 0..0.3175 -> SUM them.
            Reading only `lift_lower` (as this did) halves z_lift, so at full lift
            the model computed CoM h=0.512 m instead of 0.655 m and handed out ~28%
            more decel/speed than is tip-safe -- generous exactly when the robot is
            least stable.
          * real / bridge : a single joint (LIFT_JOINT, default `lift`) already in
            total metres -> take it as-is.

        Unknown/absent -> keep the last value; the lift is a real sensor and a
        missing one must not silently read 0.0 (= "fully lowered" = most generous
        caps). See `lift_seen` for whether it was ever observed.
        """
        lower = upper = total = None
        for name, pos in zip(msg.name, msg.position):
            if name == 'lift_lower':
                lower = float(pos)
            elif name == 'lift_upper':
                upper = float(pos)
            elif name == self.lift_joint:
                total = float(pos)

        if total is not None:
            self._lift = total
        elif lower is not None or upper is not None:
            self._lift = (lower or 0.0) + (upper or 0.0)
        else:
            return                       # no lift in this message; keep last known
        self._lift_seen = True

    def _on_odom(self, msg: Odometry):
        # Re-publish wheel odometry's body twist in the nav frame (front = +x) so the
        # controller's velocity feedback matches the frame MPPI commands in. Pose is
        # untouched (the controller takes pose from TF, only the twist from here).
        if not self.nav_frame_flip:
            self._odom_pub.publish(msg)
            return
        msg.child_frame_id = 'base_link_nav'
        msg.twist.twist.linear.x = -msg.twist.twist.linear.x
        msg.twist.twist.linear.y = -msg.twist.twist.linear.y
        self._odom_pub.publish(msg)

    def _step(self):
        # Watchdog: stale -> target zero (motion model ramps safely to a stop).
        if self._got_cmd:
            dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
            tgt = (0.0, 0.0, 0.0) if dt > self.cmd_timeout else self._tgt
        else:
            tgt = (0.0, 0.0, 0.0)

        # NAV-FRAME FLIP: /cmd_vel is in base_link_nav (front = +x). Rotate the twist
        # 180 deg about z into the physical base_link (front = -x) for the swerve IK:
        # (vx,vy) -> (-vx,-vy), wz unchanged. safe_base_step then caps front motion
        # (now -vx in base_link) with the conservative front tip-ceiling.
        if self.nav_frame_flip:
            tgt = (-tgt[0], -tgt[1], tgt[2])

        if self.use_motion_model:
            # A lift we have never seen reads 0.0 = "fully lowered" = the MOST
            # generous tip caps. That is fail-OPEN, so say so (throttled) rather
            # than let Nav2 drive on an assumption. Non-fatal: the per-axis clamps
            # and the ZMP envelope still apply, they are just computed for a
            # lowered robot.
            if not self._lift_seen and any(abs(t) > 1e-6 for t in tgt):
                self.get_logger().warn(
                    'no lift joint seen on /joint_states — tip caps are being '
                    'computed for a FULLY LOWERED robot (z_lift=0). If the lift '
                    'is raised, Nav2 is over-permitted. Expected "lift_lower"+'
                    f'"lift_upper" (sim/URDF) or "{self.lift_joint}" (bridge).',
                    throttle_duration_sec=10.0)
            self._vel = mm.safe_base_step(*self._vel, *tgt, self.dt, self._lift)
        else:
            self._vel = tgt

        # Publish /mabel_cmd ONLY while Nav2 is actively commanding OR still ramping down
        # to a stop. When idle, STAY SILENT: /mabel_cmd has other publishers (keyboard
        # teleop), and two publishers fighting -- 50 Hz idle zeros here vs the teleop's
        # commands -- makes the base STALL. Once idle AND stopped, the base_bridge_client
        # / hardware-bridge watchdog already holds the base at zero, so silence is safe.
        active = bool(self._got_cmd) and \
            (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9 <= self.cmd_timeout
        moving = any(abs(v) > 1e-3 for v in self._vel)
        if not (active or moving):
            return

        out = MabelSwerveCmd()
        out.vx, out.vy, out.omega = (float(self._vel[0]), float(self._vel[1]),
                                     float(self._vel[2]))
        out.lift_pos = float(self.lift_pos)
        out.hardware_state = 0
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
