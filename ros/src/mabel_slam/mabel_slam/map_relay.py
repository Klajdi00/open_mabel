"""map_relay.py — relay a provenance-named occupancy grid onto the canonical /map.

The two SLAM stacks publish their grids on DISTINCT topics so `ros2 topic list`
always shows which SLAM produced the map:

    /map_lidar    nav_msgs/OccupancyGrid   slam_toolbox      (2D LiDAR SLAM)
    /map_visual   nav_msgs/OccupancyGrid   slice_to_occupancy (cuVSLAM/nvblox ESDF)

Every consumer (Nav2 static layer, RViz, the app map stream on :9090) keeps reading
the single canonical `/map`; this relay republishes the ACTIVE source onto it. Only
one relay runs at a time (each SLAM launch spawns its own for its topic), so /map
still has exactly one publisher.

QoS: transient_local + reliable on BOTH sides — maps are latched; a late-joining
subscriber (RViz opened later, the app reconnecting) must still receive the last
grid.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy)
from nav_msgs.msg import OccupancyGrid


class MapRelay(Node):
    def __init__(self):
        super().__init__("map_relay")
        self.declare_parameter("source_topic", "/map_lidar")
        src = self.get_parameter("source_topic").value
        qos = QoSProfile(depth=1,
                         reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(OccupancyGrid, "/map", qos)
        self.create_subscription(OccupancyGrid, src, self._pub.publish, qos)
        self.get_logger().info(f"map relay: {src} -> /map (transient_local)")


def main():
    rclpy.init()
    node = MapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
