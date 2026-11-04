"""
LiDAR -> 3D geometry for the viewport, in the SAME binary layouts the web map
viewer (``teleop-app.js`` MapStream) already decodes, so the frontend's existing
point-cloud path renders them unchanged:

  CLOUD (0x05): live per-frame scan, replaced each frame.
  MESH  (0x01): the accumulated map — every frame's scan transformed into the
                world frame via the chassis pose and voxel-downsampled. This is
                "the mesh the robot is mapping," reconstructed from the episode's
                own sensor data (works for sim AND real, no extra geometry).

Vertex layout (both): f32 x,y,z (MuJoCo/ROS z-up) + u8 r,g,b,a = 16 bytes.
Header: u8 type, u8 flags(bit0=final), u16 pad, u32 count.
"""
import struct

import numpy as np

from . import h5pool

LIDAR_Z = 0.25            # planar-scan mount height for display (m)


def _turbo(t):
    """Cheap turbo-ish colormap; t in [0,1] -> (r,g,b) uint8 arrays."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05)))), 0, 255)
    g = np.clip(23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56)))), 0, 255)
    b = np.clip(27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66)))), 0, 255)
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)


def _pack(points_xyz, colors_rgb, type_byte, final=True):
    n = points_xyz.shape[0]
    head = struct.pack("<BBHI", type_byte, 1 if final else 0, 0, n)
    if n == 0:
        return head
    buf = np.empty((n, 16), dtype=np.uint8)
    buf[:, 0:12] = points_xyz.astype("<f4").view(np.uint8).reshape(n, 12)
    buf[:, 12:15] = colors_rgb
    buf[:, 15] = 255
    return head + buf.tobytes()


def _scan_to_local(ranges, res_deg):
    """Planar ranges [N] -> local-frame points [(valid),3], dropping no-hit."""
    n = ranges.shape[0]
    theta = np.deg2rad(np.arange(n) * res_deg)
    valid = ranges > 0.05
    r = ranges[valid]
    th = theta[valid]
    x = r * np.cos(th)
    y = r * np.sin(th)
    z = np.full_like(x, LIDAR_Z)
    return np.stack([x, y, z], axis=1), r


def lidar_cloud_frame(path, i, res_deg=1.0, max_m=8.0):
    """Per-frame lidar scan as a CLOUD (0x05) in the CHASSIS frame (the frontend
    parents it under navWorld and re-poses by the base, like the live cloud)."""
    with h5pool.reading(path) as f:
        if "obs/lidar" not in f:
            return _pack(np.empty((0, 3)), None, 0x05)
        num = f["obs/lidar"].shape[0]
        i = max(0, min(int(i), num - 1))
        ranges = np.asarray(f["obs/lidar"][i], dtype=np.float32)
    pts, r = _scan_to_local(ranges, res_deg)
    cr, cg, cb = _turbo(r / max_m)
    return _pack(pts, np.stack([cr, cg, cb], axis=1), 0x05)


def accumulated_scene(path, start=None, end=None, stride=2, voxel=0.05, res_deg=1.0):
    """Accumulate every (strided) frame's scan into the WORLD frame using the
    chassis pose, voxel-downsample, return as MESH (0x01)."""
    with h5pool.reading(path) as f:
        if "obs/lidar" not in f or "obs/pose/chassis" not in f:
            return _pack(np.empty((0, 3)), None, 0x01)
        lid = f["obs/lidar"]
        T = lid.shape[0]
        s = 0 if start is None else max(0, int(start))
        e = T if end is None else min(T, int(end))
        lidar = np.asarray(lid[s:e], dtype=np.float32)
        pose = np.asarray(f["obs/pose/chassis"][s:e], dtype=np.float32)

    res = np.deg2rad(np.arange(lidar.shape[1]) * res_deg)
    cos_t, sin_t = np.cos(res), np.sin(res)
    chunks = []
    for k in range(0, lidar.shape[0], max(1, int(stride))):
        ranges = lidar[k]
        valid = ranges > 0.05
        if not np.any(valid):
            continue
        r = ranges[valid]
        lx = r * cos_t[valid]
        ly = r * sin_t[valid]
        # chassis world pose: x,y + yaw from quat (w,x,y,z)
        x, y = float(pose[k, 0]), float(pose[k, 1])
        qw, qx, qy, qz = (float(c) for c in pose[k, 3:7])
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, sn = np.cos(yaw), np.sin(yaw)
        wx = x + lx * c - ly * sn
        wy = y + lx * sn + ly * c
        wz = np.full_like(wx, LIDAR_Z)
        chunks.append(np.stack([wx, wy, wz], axis=1))
    if not chunks:
        return _pack(np.empty((0, 3)), None, 0x01)
    pts = np.concatenate(chunks, axis=0)
    # voxel downsample
    if voxel and voxel > 0:
        keys = np.floor(pts / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        pts = pts[np.sort(idx)]
    # colour the map by height-independent radial extent for a subtle gradient
    rad = np.linalg.norm(pts[:, :2], axis=1)
    rng = (rad - rad.min()) / (np.ptp(rad) + 1e-6)
    cr, cg, cb = _turbo(0.15 + 0.5 * rng)
    return _pack(pts, np.stack([cr, cg, cb], axis=1), 0x01)


_OCC_CACHE = {}   # path -> (mtime, result)  — occupancy is static per episode


def occupancy_map(path, start=None, end=None, stride=2, res=0.05, res_deg=1.0):
    """Rasterize the accumulated planar lidar into a top-down 2D occupancy grid
    (gmapping: occupied/free/unknown), returned as base64 PNG + world bounds.
    Memoized per episode so re-opening the map panel is instant."""
    import base64, os
    if start is None and end is None:                      # only the full-episode map is cached
        try:
            mt = os.path.getmtime(path)
            c = _OCC_CACHE.get(path)
            if c and c[0] == mt:
                return c[1]
        except OSError:
            mt = None
    else:
        mt = None
    with h5pool.reading(path) as f:
        if "obs/lidar" not in f or "obs/pose/chassis" not in f:
            return None
        lid = f["obs/lidar"]
        T = lid.shape[0]
        s = 0 if start is None else max(0, int(start))
        e = T if end is None else min(T, int(end))
        lidar = np.asarray(lid[s:e], dtype=np.float32)
        pose = np.asarray(f["obs/pose/chassis"][s:e], dtype=np.float32)

    ang = np.deg2rad(np.arange(lidar.shape[1]) * res_deg)
    cos_t, sin_t = np.cos(ang), np.sin(ang)
    xs, ys = [], []
    for k in range(0, lidar.shape[0], max(1, int(stride))):
        r = lidar[k]; v = r > 0.05
        if not np.any(v):
            continue
        lx = r[v] * cos_t[v]; ly = r[v] * sin_t[v]
        x, y = float(pose[k, 0]), float(pose[k, 1])
        qw, qx, qy, qz = (float(c) for c in pose[k, 3:7])
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, sn = np.cos(yaw), np.sin(yaw)
        xs.append(x + lx * c - ly * sn); ys.append(y + lx * sn + ly * c)
    if not xs:
        return None
    X = np.concatenate(xs); Y = np.concatenate(ys)
    pad = 0.5
    # bounds include the robot path so the live robot marker is always on-map
    minx = float(min(X.min(), pose[:, 0].min())) - pad
    maxx = float(max(X.max(), pose[:, 0].max())) + pad
    miny = float(min(Y.min(), pose[:, 1].min())) - pad
    maxy = float(max(Y.max(), pose[:, 1].max())) + pad
    w = max(1, int(np.ceil((maxx - minx) / res)))
    h = max(1, int(np.ceil((maxy - miny) / res)))
    w, h = min(w, 2000), min(h, 2000)
    inv = 1.0 / res

    # gmapping-style occupancy: ray-carve FREE space from the robot to each hit,
    # mark the endpoint OCCUPIED, leave never-observed cells UNKNOWN. We only need
    # booleans (free/occupied), so use flat boolean index-assignment instead of
    # np.add.at counting — an order of magnitude faster to build.
    free = np.zeros(h * w, bool)
    occ = np.zeros(h * w, bool)
    S = 220                                                # samples per ray for free-space carving
    tline = np.linspace(0.0, 1.0, S, dtype=np.float32)
    for k in range(0, lidar.shape[0], max(1, int(stride))):
        r = lidar[k]; v = r > 0.05
        if not np.any(v):
            continue
        lx = r[v] * cos_t[v]; ly = r[v] * sin_t[v]
        x, y = float(pose[k, 0]), float(pose[k, 1])
        qw, qx, qy, qz = (float(c) for c in pose[k, 3:7])
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        c, sn = np.cos(yaw), np.sin(yaw)
        ex = x + lx * c - ly * sn; ey = y + lx * sn + ly * c          # hit endpoints (world)
        sx = x + (ex - x)[:, None] * tline[None, :]                   # samples along each ray
        sy = y + (ey - y)[:, None] * tline[None, :]
        ci = np.clip(((sx - minx) * inv).astype(np.int32), 0, w - 1)
        ri = np.clip(((sy - miny) * inv).astype(np.int32), 0, h - 1)
        free[(ri.ravel() * w + ci.ravel())] = True
        eci = np.clip(((ex - minx) * inv).astype(np.int32), 0, w - 1)
        eri = np.clip(((ey - miny) * inv).astype(np.int32), 0, h - 1)
        occ[eri * w + eci] = True
    val = np.full(h * w, 150, np.uint8)                   # unknown → gray
    val[free] = 244                                       # free    → white
    val[occ] = 24                                         # occupied→ black (wins over free)
    val = val.reshape(h, w)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = val; img[..., 1] = val; img[..., 2] = val; img[..., 3] = 255
    img = np.flipud(img)                                   # PNG row 0 = top (max Y)
    try:
        import cv2
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        png = buf.tobytes() if ok else b""
    except Exception:
        return None
    out = {"w": w, "h": h, "res": res,
           "minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy),
           "png_b64": base64.b64encode(png).decode("ascii")}
    if mt is not None:
        _OCC_CACHE[path] = (mt, out)
    return out
