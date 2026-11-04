"""Relay RViz's "2D Goal Pose" clicks to the Nav2 ``NavigateToPose`` action.

RViz's stock **2D Goal Pose** tool (``rviz_default_plugins/SetGoal``) publishes a
``geometry_msgs/PoseStamped`` on ``/goal_pose`` -- but Nav2's ``bt_navigator``
only offers an ACTION (``navigate_to_pose``), not a topic. This tiny node bridges
the two so the operator can just click a goal in RViz and the robot drives there,
without depending on the ``nav2_rviz_plugins`` panel being present/configured.

A new click PREEMPTS the previous goal (cancel + resend), so you can redirect the
robot mid-drive. It logs the goal, acceptance, and the final result so the
terminal tells the whole story. This is the manual-driving twin of
``frontier_explorer`` -- both feed the SAME Nav2 action, the single command path.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  -- registers PoseStamped transform support
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from nav2_msgs.action import NavigateToPose


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GoalRelay(Node):
    def __init__(self):
        super().__init__('goal_relay')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('nav_action', 'navigate_to_pose')
        # The global frame Nav2 plans in. RViz's "2D Goal Pose" stamps the goal with
        # whatever its *fixed frame* is -- which on the shared sim RViz is 'odom', NOT
        # 'map'. A goal left in 'odom' silently drifts with wheel odometry (Nav2
        # re-transforms it into the global frame every planning cycle), so we lock it
        # to 'map' ONCE here via TF before sending. Same-frame goals pass through.
        self.declare_parameter('global_frame', 'map')
        # Hold a clicked goal until the body is in its navigation pose (arms tucked
        # down) -- nav_pose_setter latches /mabel/body_ready True when the arms are
        # down. A goal clicked before that is BUFFERED and fired the moment the body
        # is ready, so an early click is never lost. Set wait_for_body_ready False to
        # forward goals immediately.
        self.declare_parameter('wait_for_body_ready', True)
        self.declare_parameter('body_ready_topic', '/mabel/body_ready')
        goal_topic = self.get_parameter('goal_topic').value
        nav_action = self.get_parameter('nav_action').value
        self._global_frame = self.get_parameter('global_frame').value
        self._wait_for_body_ready = self.get_parameter('wait_for_body_ready').value
        body_ready_topic = self.get_parameter('body_ready_topic').value

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._nav = ActionClient(self, NavigateToPose, nav_action)
        self._active = None        # current goal handle (for preemption)
        self._body_ready = not self._wait_for_body_ready
        self._pending = None       # latest goal clicked while the body wasn't ready
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 1)
        if self._wait_for_body_ready:
            # Match nav_pose_setter's latched (TRANSIENT_LOCAL) publisher so we get
            # the retained flag even though it was set before we subscribed.
            ready_qos = QoSProfile(
                depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST)
            self.create_subscription(Bool, body_ready_topic, self._on_body_ready, ready_qos)
        self.get_logger().info(
            f"goal_relay up: RViz '2D Goal Pose' ({goal_topic}) -> Nav2 "
            f"'{nav_action}' (goals locked to '{self._global_frame}'). "
            f"Click a goal in RViz to drive there."
            + ("" if self._body_ready else
               " Goals are HELD until the body reaches its navigation pose."))

    def _to_global(self, pose: PoseStamped) -> PoseStamped:
        """Transform an incoming goal into the global frame so it does not drift.

        Returns the pose unchanged if it is already in the global frame, or if the
        transform is unavailable (better to send the raw click than drop it)."""
        if not pose.header.frame_id or pose.header.frame_id == self._global_frame:
            return pose
        try:
            return self._tf_buffer.transform(
                pose, self._global_frame, timeout=Duration(seconds=0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException) as exc:
            self.get_logger().warn(
                f"Could not transform goal {pose.header.frame_id} -> "
                f"{self._global_frame} ({exc}); sending as-is.")
            return pose

    def _on_body_ready(self, msg: Bool):
        was_ready = self._body_ready
        self._body_ready = bool(msg.data)
        if self._body_ready and not was_ready:
            self.get_logger().info("body ready (arms in navigation pose).")
            if self._pending is not None:        # fire the goal that was held back
                pose, self._pending = self._pending, None
                self.get_logger().info("sending the goal held until the body was ready.")
                self._dispatch(pose)

    def _on_goal(self, pose: PoseStamped):
        if not self._body_ready:
            self._pending = pose                 # keep only the latest; fire it on ready
            self.get_logger().warn(
                "body not in its navigation pose yet -- holding this goal until "
                "/mabel/body_ready, then driving.")
            return
        self._dispatch(pose)

    def _dispatch(self, pose: PoseStamped):
        if not self._nav.server_is_ready():
            self.get_logger().warn(
                "Nav2 navigate_to_pose action not ready yet -- is the stack active? "
                "Ignoring this goal.")
            return
        # Preempt any in-flight goal so the latest click wins.
        if self._active is not None:
            try:
                self._active.cancel_goal_async()
            except Exception:        # noqa: BLE001
                pass
            self._active = None

        pose = self._to_global(pose)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(
            f"-> goal ({pose.pose.position.x:+.2f}, {pose.pose.position.y:+.2f}) m, "
            f"heading {math.degrees(_yaw(pose.pose.orientation)):+.0f} deg "
            f"[{pose.header.frame_id}]")
        fut = self._nav.send_goal_async(goal)
        fut.add_done_callback(self._on_response)

    def _on_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal.")
            return
        self._active = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        # action_msgs/GoalStatus: 4=SUCCEEDED 5=CANCELED 6=ABORTED
        status = future.result().status
        msg = {4: "reached goal.", 5: "goal canceled.", 6: "goal aborted (no valid path?)."}
        self.get_logger().info(msg.get(status, f"goal ended status={status}."))
        self._active = None


def main(args=None):
    rclpy.init(args=args)
    node = GoalRelay()
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
