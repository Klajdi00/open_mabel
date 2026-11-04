"""Pre-navigation whole-body posture: tuck the arms DOWN before the base moves.

WHY.  At the model rest pose (qpos0) the arms hang from the shoulders but the
ELBOW sits at ~90 deg, so each forearm sticks straight OUT in front of the
shoulder (the hand ends up ~0.31 m forward of the body, at desk height). That is
fine for manipulation but is a collision hazard while DRIVING -- the out-flung
forearms clip doorframes, table edges and people. This node folds the elbows so
the forearms hang straight DOWN (hands tucked beside the torso, ~0.48 m off the
arm mount) -- a narrow, snag-free "navigation pose" -- and only THEN lets the
navigation stack start sending the base anywhere.

CONTRACT.  It speaks the SAME canonical arm contract as teleop / the WBC / the
sim and the real arms node:

    publishes  ``/arms/command``       sensor_msgs/JointState   target [rad]
    subscribes ``/arms/joint_states``  sensor_msgs/JointState   measured [rad]

and gates the rest of the stack with ONE latched flag:

    publishes  ``/mabel/body_ready``   std_msgs/Bool (TRANSIENT_LOCAL latched)

False the instant the node starts; True once every commanded joint has settled
within ``settle_tolerance`` of its target for ``settle_time`` seconds -- OR once
``ready_timeout`` elapses (a fail-safe so a missing/idle arms backend can never
DEADLOCK navigation: it proceeds with a warning instead). ``goal_relay`` and
``frontier_explorer`` both wait on this flag, so neither a manual RViz goal nor
an autonomous frontier goal can drive the base until the arms are down.

The node keeps publishing the hold pose (at ``publish_rate``) and keeps the flag
latched for the whole session, so the arms stay tucked throughout navigation and
a late-joining subscriber still learns the body is ready.

THE POSE.  Only the elbow (``*_arm_4``) differs from rest; everything else stays
at 0. The two elbow targets were measured from ``mabel_full.xml`` (sweeping the
elbow for the lowest hand): left_arm_4 = +1.589, right_arm_4 = -1.589 rad drop
the hands from z=0.70 m (flung 0.31 m forward) to z=0.48 m (hanging at the side).
All targets are overridable via the ``arm_targets`` parameter ("joint:rad, ...").
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# Navigation pose, URDF joint name -> target [rad]. All arm joints at rest (0)
# EXCEPT the elbows, folded so the forearms hang straight down (measured from the
# MJCF; see module docstring). Symmetric: the left/right elbow ranges are mirrored
# (left_arm_4 in [-0.77, 1.67], right_arm_4 in [-1.67, 0.77]).
DEFAULT_NAV_POSE = {
    "left_arm_1": 0.0, "left_arm_2": 0.0, "left_arm_3": 0.0,
    "left_arm_4": 1.589,                                       # elbow -> forearm down
    "left_arm_5": 0.0, "left_arm_6": 0.0, "left_arm_7": 0.0,
    "right_arm_1": 0.0, "right_arm_2": 0.0, "right_arm_3": 0.0,
    "right_arm_4": -1.589,                                     # elbow -> forearm down
    "right_arm_5": 0.0, "right_arm_6": 0.0, "right_arm_7": 0.0,
}


def _parse_targets(spec, default):
    """'left_arm_4:1.589, right_arm_4:-1.589' -> {joint: rad}; '' -> default copy."""
    spec = (spec or "").strip()
    if not spec:
        return dict(default)
    out = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, val = item.partition(":")
        name = name.strip()
        try:
            out[name] = float(val)
        except ValueError:
            continue
    return out or dict(default)


class NavPoseSetter(Node):
    def __init__(self):
        super().__init__("nav_pose_setter")

        self.declare_parameter("arm_targets", "")       # "joint:rad, ..."; empty -> DEFAULT_NAV_POSE
        self.declare_parameter("arms_command_topic", "/arms/command")
        self.declare_parameter("arms_state_topic", "/arms/joint_states")
        self.declare_parameter("body_ready_topic", "/mabel/body_ready")
        self.declare_parameter("publish_rate", 10.0)    # Hz; holds the pose for the session
        self.declare_parameter("settle_tolerance", 0.12)  # rad; per-joint "arrived" band
        self.declare_parameter("settle_time", 0.5)      # s held within tolerance -> settled
        self.declare_parameter("ready_timeout", 10.0)   # s fail-safe: proceed even if not settled

        self._targets = _parse_targets(
            self.get_parameter("arm_targets").value, DEFAULT_NAV_POSE)
        self._names = list(self._targets.keys())
        self._tol = float(self.get_parameter("settle_tolerance").value)
        self._settle_time = float(self.get_parameter("settle_time").value)
        self._timeout = float(self.get_parameter("ready_timeout").value)

        self._measured = {}            # joint -> latest measured rad
        self._ready = False
        self._in_band_since = None     # wall time the pose first fell fully in-band
        self._start_t = self._now()

        cmd_topic = self.get_parameter("arms_command_topic").value
        state_topic = self.get_parameter("arms_state_topic").value
        ready_topic = self.get_parameter("body_ready_topic").value

        self._cmd_pub = self.create_publisher(JointState, cmd_topic, 10)
        # Latch the ready flag so explorer/goal_relay learn it whenever they start.
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST)
        self._ready_pub = self.create_publisher(Bool, ready_topic, latched)
        self.create_subscription(JointState, state_topic, self._on_state, 10)

        self._publish_ready(False)     # body is NOT ready until the arms are tucked
        rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"nav_pose_setter up: tucking arms to nav pose ({len(self._names)} joints) "
            f"on {cmd_topic}; holding navigation until '{ready_topic}'. Elbows -> "
            f"forearms down (left_arm_4={self._targets.get('left_arm_4')}, "
            f"right_arm_4={self._targets.get('right_arm_4')}).")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            key = name.lower().replace(" ", "_")     # accept MuJoCo Title-Case too
            if key in self._targets:
                self._measured[key] = float(pos)

    def _publish_ready(self, value):
        m = Bool()
        m.data = bool(value)
        self._ready_pub.publish(m)

    def _all_in_band(self):
        """True only once EVERY commanded joint has a fresh measurement within tol."""
        if len(self._measured) < len(self._names):
            return False
        return all(abs(self._measured[n] - self._targets[n]) <= self._tol
                   for n in self._names)

    def _tick(self):
        # Always (re)publish the hold pose so the arms stay tucked the whole session.
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self._names)
        js.position = [self._targets[n] for n in self._names]
        self._cmd_pub.publish(js)

        if self._ready:
            self._publish_ready(True)        # keep the latch warm for late subscribers
            return

        now = self._now()
        if self._all_in_band():
            if self._in_band_since is None:
                self._in_band_since = now
            elif now - self._in_band_since >= self._settle_time:
                self._mark_ready("arms settled in navigation pose")
                return
        else:
            self._in_band_since = None

        # Fail-safe: never deadlock navigation if the arms backend is absent/idle.
        if now - self._start_t >= self._timeout:
            seen = len(self._measured)
            self._mark_ready(
                f"ready_timeout ({self._timeout:.0f}s) reached with arms NOT confirmed "
                f"down ({seen}/{len(self._names)} joints reporting) -- proceeding anyway. "
                f"Is an arms backend (sim/HAL) publishing {self.get_parameter('arms_state_topic').value}?",
                warn=True)

    def _mark_ready(self, why, warn=False):
        self._ready = True
        self._publish_ready(True)
        (self.get_logger().warn if warn else self.get_logger().info)(
            f"BODY READY -- navigation released: {why}.")


def main(args=None):
    rclpy.init(args=args)
    node = NavPoseSetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
