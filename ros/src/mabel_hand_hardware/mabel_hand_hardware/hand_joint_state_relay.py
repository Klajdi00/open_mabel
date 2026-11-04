#!/usr/bin/env python3
"""Merge both ORCA hands' feedback into one URDF-compatible ``/joint_states``.

Each hand node publishes its own ``<side>_hand/joint_states`` using the ORCA
firmware joint names (``thumb_cmc`` / ``thumb_mp`` / ``thumb_ip`` for the thumb).
The ``mabel_description`` URDF instead names the thumb chain ``thumb_mcp`` /
``thumb_pip`` / ``thumb_dip``. The finger and wrist names already agree.

This node:

  * subscribes to every input topic in ``input_topics`` (default: the two hand
    feedback topics),
  * renames the thumb joints to match the URDF,
  * accumulates the latest position for every joint it has ever seen, and
  * republishes the union on a single ``/joint_states`` topic at ``publish_rate``
    Hz so ``robot_state_publisher`` can animate the model live.

Run ``visualize.launch.py`` to get this plus robot_state_publisher + RViz.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# ORCA firmware joint name -> mabel_description URDF joint name.
# Only the thumb differs; fingers (abd/mcp/pip) and the wrist already match.
#
# NOTE on the mp/ip <-> pip/dip routing: the real hand drives the two distal
# thumb joints correctly, but in RViz they appeared swapped (the proximal joint's
# motion showing up on the tip and vice-versa). The motor-ID->joint map is
# correct (feetech config), so the swap lived purely in this name routing -- the
# ORCA "mp"/"ip" feedback was wired to the wrong URDF link. They are crossed here
# so the model tracks the real thumb.
THUMB_REMAP = {
    "thumb_cmc": "thumb_mcp",   # carpometacarpal -> URDF "mcp" (base)
    "thumb_mp": "thumb_dip",    # ORCA metacarpophalangeal -> URDF distal (was pip; swapped)
    "thumb_ip": "thumb_pip",    # ORCA interphalangeal     -> URDF middle (was dip; swapped)
    # thumb_abd is identical on both sides.
}


def to_urdf_name(name):
    """Map a side-prefixed ORCA joint name to its URDF equivalent.

    ``left_thumb_cmc`` -> ``left_thumb_mcp``; everything else is unchanged.
    """
    for side in ("left_", "right_"):
        if name.startswith(side):
            base = name[len(side):]
            return side + THUMB_REMAP.get(base, base)
    return THUMB_REMAP.get(name, name)


class JointStateRelay(Node):

    def __init__(self):
        super().__init__("hand_joint_state_relay")

        self.declare_parameter(
            "input_topics",
            ["/left_hand/joint_states", "/right_hand/joint_states"])
        self.declare_parameter("output_topic", "/joint_states")
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("remap_thumb", True)

        inputs = list(self.get_parameter("input_topics").value)
        out = self.get_parameter("output_topic").value
        rate = float(self.get_parameter("publish_rate").value)
        self.remap = bool(self.get_parameter("remap_thumb").value)

        # latest position seen for every joint, keyed by URDF name
        self._positions = {}

        self.pub = self.create_publisher(JointState, out, 10)
        self._subs = [
            self.create_subscription(
                JointState, topic, self._on_js, 10)
            for topic in inputs
        ]
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._publish)
        self.get_logger().info(
            f"relaying {inputs} -> {out} @ {rate} Hz "
            f"(thumb remap {'on' if self.remap else 'off'})")

    def _on_js(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        for name, pos in zip(msg.name, msg.position):
            key = to_urdf_name(name) if self.remap else name
            self._positions[key] = pos

    def _publish(self):
        if not self._positions:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self._positions.keys())
        js.position = list(self._positions.values())
        self.pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateRelay()
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
