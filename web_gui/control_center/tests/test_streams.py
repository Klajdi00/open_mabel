"""Stream hub: record encoding + store semantics (no ROS needed)."""
import struct

from control_center.server import streams


def test_encode_grid_layout():
    cells = bytes([0, 50, 100, 255, 0, 100])
    blob = streams.encode_grid(3, 2, 0.05, -1.0, -2.0, 0.0, 0.5, cells)
    assert blob[0] == 0x04
    w, h = struct.unpack_from("<II", blob, 4)
    res, ox, oy, oz, yaw = struct.unpack_from("<fffff", blob, 12)
    assert (w, h) == (3, 2)
    assert abs(res - 0.05) < 1e-6
    assert abs(ox + 1.0) < 1e-6 and abs(oy + 2.0) < 1e-6
    assert abs(yaw - 0.5) < 1e-6
    assert blob[32:] == cells
    assert len(blob) == 32 + 6


def test_hub_latest_revisions():
    hub = streams.StreamHub()
    hub._store("grid:lidar", b"a")
    hub._store("cloud:head", b"b")
    rev, fresh = hub.latest(["grid:lidar", "cloud:head"], 0)
    assert {k for k, _ in fresh} == {"grid:lidar", "cloud:head"}
    # nothing new after consuming
    rev2, fresh2 = hub.latest(["grid:lidar", "cloud:head"], rev)
    assert fresh2 == [] and rev2 == rev
    # only the updated key comes back
    hub._store("grid:lidar", b"a2")
    _, fresh3 = hub.latest(["grid:lidar", "cloud:head"], rev)
    assert fresh3 == [("grid:lidar", b"a2")]


def test_quat_helpers():
    class Q:
        w, x, y, z = 0.7071068, 0.0, 0.0, 0.7071068
    assert abs(streams._quat_yaw(Q) - 1.5707963) < 1e-4
    m = streams._quat_mat(1, 0, 0, 0)
    assert m[0][0] == 1 and m[1][1] == 1 and m[2][2] == 1
