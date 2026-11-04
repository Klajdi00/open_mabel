#!/usr/bin/env python3
"""Re-publish the nvblox 3D map in two plugin-free forms for the SHARED RViz.

nvblox's 3D map is a *triangle mesh* (nvblox_msgs/Mesh on /nvblox_node/mesh).
RViz can only render that natively via nvblox_rviz_plugin/NvbloxMesh, which is an
apt-only /opt/ros plugin -- the sim/hardware RViz runs in the conda RoboStack env
(ABI-incompatible) and CANNOT load it. So this node converts the mesh into two
forms the *stock* rviz_default_plugins can show in that same window:

  /nvblox_node/mesh_marker      visualization_msgs/Marker (TRIANGLE_LIST)
                                -> real shaded mesh SURFACES (rviz Marker display)
  /nvblox_node/mesh_pointcloud  sensor_msgs/PointCloud2
                                -> the mesh vertices as a coloured point cloud
                                   (fallback / lighter-weight view)

It accumulates the INCREMENTAL mesh blocks (each message carries only the blocks
that changed, plus a `clear` flag that wipes the whole map) and republishes the
accumulated result. nvblox mesh vertices are ABSOLUTE world coordinates; each
block also carries int32[] `triangles` (block-local vertex indices, 3 per face)
and a per-vertex ColorRGBA from the camera-integrated colour.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

# nvblox_msgs ships with the NVIDIA Isaac ROS overlay (nvblox). It is present on
# the Jetson and inside the x86 Isaac ROS container, but NOT on a plain ROS box
# (e.g. the workstation running only LiDAR SLAM via slam_toolbox). This node is a
# pure nvblox consumer, so when nvblox isn't installed it has nothing to do --
# guard the import and exit cleanly in main() instead of crashing the launch.
try:
    from nvblox_msgs.msg import Mesh
    _NVBLOX_IMPORT_ERROR = None
except ImportError as _e:                                  # noqa: BLE001
    Mesh = None
    _NVBLOX_IMPORT_ERROR = _e


class MeshToPointCloud(Node):
    def __init__(self):
        super().__init__("mesh_to_pointcloud")
        self.declare_parameter("publish_rate", 2.0)      # Hz; the full mesh is large
        self.declare_parameter("in_topic", "/nvblox_node/mesh")
        self.declare_parameter("cloud_topic", "/nvblox_node/mesh_pointcloud")
        self.declare_parameter("marker_topic", "/nvblox_node/mesh_marker")
        rate = float(self.get_parameter("publish_rate").value)

        # block (x,y,z) -> (xyz Nx3 f32, rgba Nx4 f32 [0..1], tris (M*3,) i32 or
        # None, rgb_f32 N [packed for PointCloud2]). Keeping both colour forms +
        # the triangle indices lets us emit the cloud AND the surface mesh.
        self._blocks = {}
        self._frame = "odom"
        self._dirty = False

        # INPUT: match nvblox's /nvblox_node/mesh stream (reliable, volatile).
        sub_qos = QoSProfile(depth=5)
        sub_qos.reliability = QoSReliabilityPolicy.RELIABLE
        sub_qos.durability = QoSDurabilityPolicy.VOLATILE
        # OUTPUT: LATCHED (transient local), like a map. A late-joining RViz/Foxglove
        # then gets the latest mesh immediately, AND -- critically -- it is QoS
        # compatible with a Transient Local subscriber. A VOLATILE publisher paired
        # with RViz's Transient Local mesh display silently never connects, so the
        # mesh renders blank. (That was the "mesh not visualized in RViz" bug.)
        pub_qos = QoSProfile(depth=1)
        pub_qos.reliability = QoSReliabilityPolicy.RELIABLE
        pub_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        cloud_topic = self.get_parameter("cloud_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        self.cloud_pub = self.create_publisher(PointCloud2, cloud_topic, pub_qos)
        self.marker_pub = self.create_publisher(Marker, marker_topic, pub_qos)
        self.create_subscription(
            Mesh, self.get_parameter("in_topic").value, self._on_mesh, sub_qos)
        self.create_timer(1.0 / max(rate, 0.5), self._publish)
        self.get_logger().info(
            f"mesh -> TRIANGLE_LIST Marker on {marker_topic} (surfaces) + "
            f"PointCloud2 on {cloud_topic} (vertices) @ {rate:.1f} Hz")

    def _on_mesh(self, msg: Mesh):
        self._frame = msg.header.frame_id or self._frame
        # `clear` => nvblox is re-sending the ENTIRE map (e.g. to a reconnecting
        # subscriber, or after a reset); drop the accumulated blocks first so stale
        # geometry can't survive under the freshly-streamed one.
        if msg.clear:
            self._blocks.clear()
            self._dirty = True
        for idx, blk in zip(msg.block_indices, msg.blocks):
            key = (idx.x, idx.y, idx.z)
            if not blk.vertices:                  # empty block -> deallocated
                self._blocks.pop(key, None)
                self._dirty = True
                continue
            # nvblox mesh vertices are ABSOLUTE world coordinates (verified against
            # the block origin) -- do NOT add block_index*block_size, that doubles
            # the offset and throws the mesh far from the real scene.
            xyz = np.array([(p.x, p.y, p.z) for p in blk.vertices], np.float32)
            if blk.colors:
                rgba = np.array([(c.r, c.g, c.b, c.a) for c in blk.colors],
                                np.float32)
            else:
                rgba = np.tile(np.float32([0.53, 0.53, 0.53, 1.0]),
                               (xyz.shape[0], 1))
            tris = (np.asarray(blk.triangles, np.int32)
                    if blk.triangles else None)
            c = (rgba[:, :3] * 255.0).astype(np.uint32)
            rgb = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
            self._blocks[key] = (xyz, rgba, tris, rgb.view(np.float32))
            self._dirty = True

    def _publish(self):
        if not self._dirty or not self._blocks:
            return
        want_cloud = self.cloud_pub.get_subscription_count() > 0
        want_marker = self.marker_pub.get_subscription_count() > 0
        if not (want_cloud or want_marker):
            return                       # nobody listening -> don't burn CPU
        self._dirty = False
        if want_cloud:
            self._publish_cloud()
        if want_marker:
            self._publish_marker()

    def _publish_cloud(self):
        xyz = np.concatenate([b[0] for b in self._blocks.values()])
        rgb = np.concatenate([b[3] for b in self._blocks.values()])
        n = xyz.shape[0]
        pts = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"),
                                 ("z", "<f4"), ("rgb", "<f4")])
        pts["x"], pts["y"], pts["z"], pts["rgb"] = \
            xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.data = pts.tobytes()
        msg.is_dense = True
        self.cloud_pub.publish(msg)

    def _publish_marker(self):
        # A TRIANGLE_LIST lists each face's 3 corner vertices explicitly, so expand
        # the block-local `triangles` index buffer into flat per-corner arrays. The
        # stock rviz Marker display renders these as shaded surfaces (no plugin).
        verts, cols = [], []
        for xyz, rgba, tris, _ in self._blocks.values():
            if tris is not None and tris.size >= 3:
                verts.append(xyz[tris])
                cols.append(rgba[tris])
            elif tris is None and xyz.shape[0] >= 3 and xyz.shape[0] % 3 == 0:
                # No index buffer: nvblox already laid vertices out 3-per-face.
                verts.append(xyz)
                cols.append(rgba)
        if not verts:
            return
        v = np.concatenate(verts)
        c = np.concatenate(cols)

        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self._frame
        m.ns = "nvblox_mesh"
        m.id = 0
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0            # identity: vertices are already world
        m.scale.x = m.scale.y = m.scale.z = 1.0   # required non-zero for TRIANGLE_LIST
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in v]
        m.colors = [ColorRGBA(r=float(q[0]), g=float(q[1]),
                              b=float(q[2]), a=float(q[3])) for q in c]
        self.marker_pub.publish(m)


def main(args=None):
    if Mesh is None:
        # No nvblox on this machine -> nothing to convert. Exit 0 so the launch
        # treats this as "cleanly finished", not a crashed process.
        print("[mesh_to_pointcloud] nvblox_msgs not found "
              f"({_NVBLOX_IMPORT_ERROR}); skipping (Isaac ROS / nvblox not "
              "installed on this machine). This node is only needed for the "
              "visual-SLAM (cuVSLAM + nvblox) stack.", flush=True)
        return
    rclpy.init(args=args)
    node = MeshToPointCloud()
    # Robust spin: a very large incremental Mesh (nvblox sends the ENTIRE map to a
    # new/reconnecting subscriber) can fail to deserialize in rclpy and raise a
    # RuntimeError out of the executor. Skip that one message and keep running
    # instead of crashing the node (and dragging the launch down with it).
    try:
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.2)
            except KeyboardInterrupt:
                break
            except Exception as e:                       # noqa: BLE001
                node.get_logger().warn(f"skipped a mesh message: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
