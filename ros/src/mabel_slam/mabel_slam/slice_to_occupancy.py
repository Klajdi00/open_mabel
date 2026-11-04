#!/usr/bin/env python3
"""Turn the nvblox 2D ESDF slice into a Gmapping-style map + an inflated costmap.

nvblox publishes /nvblox_node/static_map_slice as a *distance field*
(nvblox_msgs/DistanceMapSlice) -- each cell is the metric distance to the
nearest obstacle, with a sentinel for unobserved cells. That isn't the familiar
occupancy grid, so this node converts it into two nav_msgs/OccupancyGrid topics:

  /map          -- Gmapping style:  0 = free (white), 100 = occupied (black),
                   -1 = unknown (grey).  occupied = within `occupied_distance`
                   of a surface; everything else observed is free.
  /map_costmap  -- the SAME slice turned into an inflation costmap (cost grows
                   toward 100 as you approach an obstacle), so RViz shows a cost
                   halo around obstacles. Since the slice is already an ESDF,
                   this is just the standard nav2 inflation curve of the distance.

nvblox's DistanceMapSlice.origin is the LOWER-LEFT (AABB min) corner in the global
frame and its data is row-major y-up (row 0 = min y) -- the SAME convention as
nav_msgs/OccupancyGrid. So the grid origin is s.origin VERBATIM (do NOT shift it by
the map height) and rows are NOT flipped. This keeps the 2D /map perfectly aligned
with the 3D nvblox mesh (both in `odom`), so the robot sits at the same place on
both -- required for navigation. `flip_y` stays as an escape hatch for a build whose
slice rows run y-down, but it ONLY mirrors the data; it never moves the origin.
"""
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import OccupancyGrid

# nvblox_msgs ships with the Isaac ROS overlay (nvblox) -- present on the Jetson
# and in the x86 Isaac ROS container, absent on a plain ROS box running only
# LiDAR SLAM. This node is a pure nvblox consumer; guard the import and exit
# cleanly in main() when nvblox isn't installed so the launch doesn't crash.
try:
    from nvblox_msgs.msg import DistanceMapSlice
    _NVBLOX_IMPORT_ERROR = None
except ImportError as _e:                                  # noqa: BLE001
    DistanceMapSlice = None
    _NVBLOX_IMPORT_ERROR = _e


class SliceToOccupancy(Node):
    def __init__(self):
        super().__init__("slice_to_occupancy")
        self.declare_parameter("occupied_distance", 0.10)  # m: <= => occupied(black)
        self.declare_parameter("inscribed_radius", 0.30)   # m: <= => lethal cost 100
        self.declare_parameter("inflation_radius", 0.60)   # m: cost halo extent
        self.declare_parameter("cost_scaling_factor", 5.0)
        self.declare_parameter("flip_y", False)
        self.occ = float(self.get_parameter("occupied_distance").value)
        self.r_in = float(self.get_parameter("inscribed_radius").value)
        self.r_inf = float(self.get_parameter("inflation_radius").value)
        self.kcost = float(self.get_parameter("cost_scaling_factor").value)
        self.flip_y = bool(self.get_parameter("flip_y").value)

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL   # latched, like a map
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", qos)
        self.pub_cost = self.create_publisher(OccupancyGrid, "/map_costmap", qos)
        self.create_subscription(
            DistanceMapSlice, "/nvblox_node/static_map_slice", self._on_slice, 5)
        self.get_logger().info(
            "slice -> /map (occupancy) + /map_costmap (inflated)")

    def _on_slice(self, s: DistanceMapSlice):
        w, h, res = s.width, s.height, s.resolution
        if w == 0 or h == 0:
            return
        d = np.asarray(s.data, dtype=np.float32).reshape(h, w)
        unknown = np.isclose(d, s.unknown_value)

        # --- occupancy grid (Gmapping semantics) ---
        occ = np.full((h, w), -1, np.int8)              # unknown / grey
        observed = ~unknown
        occ[observed & (d <= self.occ)] = 100           # obstacle / black
        occ[observed & (d > self.occ)] = 0              # free / white

        # --- inflation costmap from the distance field ---
        cost = np.zeros((h, w), np.int8)
        far = d - self.r_in
        infl = observed & (d > self.r_in) & (d < self.r_inf)
        cost[observed & (d <= self.r_in)] = 100         # inscribed -> lethal
        cost[infl] = (99.0 * np.exp(-self.kcost * far[infl])).astype(np.int8)
        cost[unknown] = -1

        # nvblox origin is already the lower-left corner (y-up), exactly what
        # OccupancyGrid wants -> use s.origin.y VERBATIM. flip_y only mirrors the
        # data rows (escape hatch for a y-down build); it must NOT shift the origin,
        # or the 2D map ends up offset from the 3D mesh by the map height.
        oy = s.origin.y
        if self.flip_y:
            occ = occ[::-1, :]
            cost = cost[::-1, :]

        self.pub_map.publish(self._grid(s, res, w, h, oy, occ))
        self.pub_cost.publish(self._grid(s, res, w, h, oy, cost))

    def _grid(self, s, res, w, h, oy, data2d):
        g = OccupancyGrid()
        g.header.stamp = self.get_clock().now().to_msg()
        g.header.frame_id = s.header.frame_id or "odom"
        g.info.resolution = res
        g.info.width = w
        g.info.height = h
        g.info.origin.position.x = float(s.origin.x)
        g.info.origin.position.y = float(oy)
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0
        g.data = data2d.astype(np.int8).flatten().tolist()
        return g


def main(args=None):
    if DistanceMapSlice is None:
        print("[slice_to_occupancy] nvblox_msgs not found "
              f"({_NVBLOX_IMPORT_ERROR}); skipping (Isaac ROS / nvblox not "
              "installed on this machine). This node is only needed for the "
              "visual-SLAM (cuVSLAM + nvblox) stack.", flush=True)
        return
    rclpy.init(args=args)
    node = SliceToOccupancy()
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
