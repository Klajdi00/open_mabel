"""ros_msgs.py — build sensor_msgs without cv_bridge / pcl (numpy -> message).

Small, dependency-light helpers so the camera drivers can publish the standard
contract messages from raw SDK frames.
"""

import struct

import numpy as np

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField


def image_msg(arr, encoding, frame_id, stamp):
    """numpy array -> sensor_msgs/Image (zero-copy-ish; data is the raw buffer)."""
    arr = np.ascontiguousarray(arr)
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height = int(arr.shape[0])
    m.width = int(arr.shape[1])
    m.encoding = encoding
    m.is_bigendian = 0
    m.step = int(arr.strides[0])
    m.data = arr.tobytes()
    return m


def camera_info_msg(width, height, K, frame_id, stamp, D=None, tx=0.0):
    """Pinhole intrinsics -> sensor_msgs/CameraInfo (plumb_bob, K row-major 3x3).

    `tx` sets P[0,3] = -fx*baseline for the RIGHT camera of a rectified stereo
    pair (the convention cuVSLAM triangulates with); 0 for mono / the left camera.
    """
    m = CameraInfo()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.width = int(width)
    m.height = int(height)
    m.distortion_model = "plumb_bob"
    m.d = [float(x) for x in (D if D is not None else [0.0] * 5)]
    fx, fy, cx, cy = K
    m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [fx, 0.0, cx, float(tx), 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


def _fields(rgb):
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if rgb:
        fields.append(
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
    return fields


def pointcloud2_msg(xyz, frame_id, stamp, rgb=None):
    """(N,3) xyz [m] (+ optional (N,3) uint8 rgb) -> organized-as-list PointCloud2.

    NaN/inf points are kept (clients filter); pass already-masked arrays to drop them.
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    has_rgb = rgb is not None
    point_step = 16 if has_rgb else 12

    if has_rgb:
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        buf = np.zeros((n, point_step), dtype=np.uint8)
        buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
        # pack r,g,b into the float32 rgb slot (b,g,r,0 little-endian = 0x00RRGGBB)
        buf[:, 12] = rgb[:, 2]
        buf[:, 13] = rgb[:, 1]
        buf[:, 14] = rgb[:, 0]
        data = buf.tobytes()
    else:
        data = xyz.tobytes()

    m = PointCloud2()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height = 1
    m.width = n
    m.fields = _fields(has_rgb)
    m.is_bigendian = False
    m.point_step = point_step
    m.row_step = point_step * n
    m.is_dense = False
    m.data = data
    return m
