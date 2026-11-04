"""
ros_io.py — the OPTIONAL DDS layer of the simulation bridge (Ubuntu):
the exact hardware_bridge topic contract, namespaced per instance (default
/sim) so real and simulated robots coexist on one graph:

  /sim/mabel/hw/base/cmd_vel    geometry_msgs/Twist          Sub
  /sim/mabel/hw/base/estop      std_msgs/Bool                Sub
  /sim/mabel/hw/base/joint_states  sensor_msgs/JointState    Pub 50 Hz
  /sim/arms/command             sensor_msgs/JointState       Sub
  /sim/arms/joint_states        sensor_msgs/JointState       Pub 50 Hz
  /sim/arms/torque_enable       std_msgs/Bool                Sub
  /sim/arms/torque_state        std_msgs/Bool                Pub (latched)
  /sim/body/…                   (same shape as arms)
  /sim/{left,right}_hand/{command,torque_enable,joint_states,
                          torque_state,temperatures,voltages}
  /sim/lift/command             std_msgs/Float64 (metres)    Sub
  /sim/lift/joint_states        sensor_msgs/JointState       Pub 50 Hz
  /sim/joint_states             merged, URDF names           Pub 50 Hz
  /sim/odom_ground_truth        nav_msgs/Odometry            Pub 50 Hz

Same message types, same rates, same latched-QoS torque_state as the real
HAL — a consumer moves between robots with a namespace remap and is told
kind = "sim" by everything upstream.

rclpy is imported lazily; on a Mac this module is simply never started.
"""
from __future__ import annotations

import logging
import math
import threading

from . import config as CFG

log = logging.getLogger("simbridge.ros")

_STEER = ["b_steer", "fr_steer", "fl_steer"]     # hardware order (SwerveBase)
_DRIVE = ["b_drive", "fr_drive", "fl_drive"]


def _group_of(name):
    from robot_command import joint_group        # server/ (path set by engine)
    return joint_group(name)


class _SimBridgeRos:
    def __init__(self, world, ns):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64, Float64MultiArray

        self.world = world
        self.ns = ns.rstrip("/")
        rclpy.init(args=None)
        self.rclpy = rclpy
        node = Node("mabel_simulation_bridge",
                    namespace=self.ns or "/")
        self.node = node
        latched = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        t = lambda p: f"{self.ns}{p}"            # absolute, namespaced

        # ── joint bucketing from the model (once) ────────────────────────────
        self._buckets = {"arms": [], "body": [], "left_hand": [],
                         "right_hand": [], "lift": []}
        for jn in world._dof_by_joint:
            g = _group_of(jn)
            if g in ("left_arm", "right_arm"):
                self._buckets["arms"].append(jn)
            elif g in ("torso", "neck", "body"):
                self._buckets["body"].append(jn)
            elif g == "lift":
                self._buckets["lift"].append(jn)
            elif g == "gripper":
                side = "left_hand" if jn.lower().startswith(("left", "l ")) \
                    else "right_hand"
                self._buckets[side].append(jn)

        # ── command intake (thin: push straight into the engine) ────────────
        node.create_subscription(Twist, t("/mabel/hw/base/cmd_vel"),
                                 self._on_cmd_vel, 10)
        node.create_subscription(Bool, t("/mabel/hw/base/estop"),
                                 lambda m: world.set_estop(bool(m.data)), 10)
        for grp in ("arms", "body"):
            node.create_subscription(
                JointState, t(f"/{grp}/command"),
                lambda m: world.set_targets(dict(zip(m.name, m.position))), 10)
            node.create_subscription(
                Bool, t(f"/{grp}/torque_enable"),
                self._torque_cb(grp), 10)
        for side in ("left", "right"):
            node.create_subscription(
                JointState, t(f"/{side}_hand/command"),
                lambda m: world.set_targets(dict(zip(m.name, m.position))), 10)
            node.create_subscription(
                Bool, t(f"/{side}_hand/torque_enable"),
                self._torque_cb(f"{side}_hand"), 10)
        node.create_subscription(
            Float64, t("/lift/command"),
            lambda m: world.set_targets({"lift": float(m.data)}), 10)

        # ── state out ────────────────────────────────────────────────────────
        self._JointState = JointState
        self._Odometry = Odometry
        self._Bool = Bool
        self._F64MA = Float64MultiArray
        self.pub = {
            "base_js": node.create_publisher(
                JointState, t("/mabel/hw/base/joint_states"), 10),
            "arms_js": node.create_publisher(JointState, t("/arms/joint_states"), 10),
            "body_js": node.create_publisher(JointState, t("/body/joint_states"), 10),
            "lift_js": node.create_publisher(JointState, t("/lift/joint_states"), 10),
            "all_js": node.create_publisher(JointState, t("/joint_states"), 10),
            "odom": node.create_publisher(Odometry, t("/odom_ground_truth"), 10),
            "arms_tq": node.create_publisher(Bool, t("/arms/torque_state"), latched),
            "body_tq": node.create_publisher(Bool, t("/body/torque_state"), latched),
        }
        for side in ("left", "right"):
            self.pub[f"{side}_js"] = node.create_publisher(
                JointState, t(f"/{side}_hand/joint_states"), 10)
            self.pub[f"{side}_tq"] = node.create_publisher(
                Bool, t(f"/{side}_hand/torque_state"), latched)
            self.pub[f"{side}_temp"] = node.create_publisher(
                Float64MultiArray, t(f"/{side}_hand/temperatures"), 10)
            self.pub[f"{side}_volt"] = node.create_publisher(
                Float64MultiArray, t(f"/{side}_hand/voltages"), 10)
        node.create_timer(1.0 / max(1.0, CFG.ROS_RATE), self._publish)

        self._exec = MultiThreadedExecutor(num_threads=2)
        self._exec.add_node(node)
        self._thread = threading.Thread(target=self._spin, daemon=True,
                                        name="simbridge-ros")
        self._thread.start()
        log.info("DDS mirror up under %s (hardware contract, kind=sim)",
                 self.ns or "/")

    def _on_cmd_vel(self, m):
        self.world.set_twist(m.linear.x, m.linear.y, m.angular.z)

    def _torque_cb(self, group):
        def cb(m):
            self.world.set_group_torque(group, bool(m.data))
        return cb

    def _spin(self):
        try:
            self._exec.spin()
        except Exception:
            pass

    # ── publishing ───────────────────────────────────────────────────────────
    def _js(self, names, snap):
        msg = self._JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [snap["joints"][n][0] for n in names]
        msg.velocity = [snap["joints"][n][1] for n in names]
        msg.effort = [0.0] * len(names)
        return msg

    def _publish(self):
        try:
            snap = self.world.snapshot()
            js = snap["joints"]
            base_names = [n for n in _STEER + _DRIVE if n in js]
            if base_names:
                self.pub["base_js"].publish(self._js(base_names, snap))
            for key, pub in (("arms", "arms_js"), ("body", "body_js"),
                             ("lift", "lift_js"),
                             ("left_hand", "left_js"),
                             ("right_hand", "right_js")):
                names = [n for n in self._buckets[key] if n in js]
                if names:
                    self.pub[pub].publish(self._js(names, snap))
            self.pub["all_js"].publish(self._js(sorted(js), snap))
            # torque states (latched)
            toff = set(snap.get("torqueOff") or [])
            self.pub["arms_tq"].publish(self._Bool(data="arms" not in toff))
            self.pub["body_tq"].publish(self._Bool(data="body" not in toff))
            for side in ("left", "right"):
                self.pub[f"{side}_tq"].publish(
                    self._Bool(data=f"{side}_hand" not in toff))
                names = self._buckets[f"{side}_hand"]
                for pub, val in ((f"{side}_temp", CFG.HAND_TEMP_C),
                                 (f"{side}_volt", CFG.HAND_VOLT_V)):
                    m = self._F64MA()
                    m.data = [float(val)] * len(names)
                    self.pub[pub].publish(m)
            # ground-truth odometry
            b = snap["base"]
            od = self._Odometry()
            od.header.stamp = self.node.get_clock().now().to_msg()
            od.header.frame_id = "odom"
            od.child_frame_id = "base_link"
            od.pose.pose.position.x = b["x"]
            od.pose.pose.position.y = b["y"]
            od.pose.pose.position.z = b["z"]
            od.pose.pose.orientation.w = math.cos(b["yaw"] / 2)
            od.pose.pose.orientation.z = math.sin(b["yaw"] / 2)
            self.pub["odom"].publish(od)
        except Exception as e:                     # never kill the timer
            log.debug("publish skipped: %s", e)

    def stop(self):
        try:
            self._exec.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
            self.rclpy.try_shutdown()
        except Exception:
            pass


def start_ros(world, ns):
    return _SimBridgeRos(world, ns)
