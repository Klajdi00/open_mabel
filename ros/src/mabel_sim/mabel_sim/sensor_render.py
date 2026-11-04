"""Pure MuJoCo + NumPy helpers for the MABEL sensor bridge.

Camera intrinsics / RGB+depth capture / LiDAR scan extraction are adapted from
``simulation/mabel_mujoco/scripts/visualize_sensors.py`` (whose docstring invited
exactly this reuse: "import these from a ROS bridge node later").  Added here:
depth -> XYZRGB point-cloud packing, MuJoCo->ROS optical-frame conversion, and a
rotation-matrix -> quaternion helper, all kept ROS-message-free so the node owns
the message assembly.
"""

import numpy as np
import mujoco


# --------------------------------------------------------------------------- #
#  Camera rendering (renderer must be created on the SAME thread it renders on)
# --------------------------------------------------------------------------- #
def clean_render_option(*groups):
    """An MjvOption for *physically realistic* camera rendering.

    Disables MuJoCo's debug overlays that otherwise bleed into the images:
      * the LiDAR rangefinder ray lines (``mjVIS_RANGEFINDER`` defaults to ON and
        draws the 360 beams as solid lines -- not something a real camera sees;
        the LiDAR should be distance readings only), and
      * the given geom/site visualization groups (group 3 = camera-marker boxes,
        lens snouts, and the LiDAR ray sites).
    """
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    for g in groups:
        opt.geomgroup[g] = 0
        opt.sitegroup[g] = 0
    return opt


# --------------------------------------------------------------------------- #
#  Sensor noise -- modelled on the REAL hardware (sim is otherwise noise-free).
#
#  RPLIDAR A3 (Slamtec datasheet rev 1.9): distance accuracy is given as a
#  percent of range -- 1% for <=3 m, 2% for 3-5 m, 2.5% for >5 m -- treated as
#  ~1-sigma (standard sim practice; no separate repeatability figure is published).
#
#  Intel RealSense D435: stereo depth error grows QUADRATICALLY with range,
#  sigma_z = z^2 * subpixel / (f_px * baseline).  Empirically (Ahn et al. 2019,
#  "Analysis and Noise Modeling of the Intel RealSense D435 for Mobile Robots")
#  sigma_z ~= base + k*z^2 with ~4 cm RMS at 2 m -> base~0.001 m, k~0.005 /m.
#  Lateral (x,y) error is smaller and grows ~linearly: sigma_xy ~= lat * z.
# --------------------------------------------------------------------------- #
def rplidar_a3_sigma(z):
    """Per-return RPLIDAR A3 range std-dev (m) for an array of ranges (m)."""
    alpha = np.where(z <= 3.0, 0.01, np.where(z <= 5.0, 0.02, 0.025))
    return alpha * z


def add_range_noise(ranges, rng, scale=1.0):
    """Apply the RPLIDAR A3 range-noise model to finite returns (scale=1.0 ==
    datasheet accuracy); +inf (no-return) beams are left untouched."""
    if scale <= 0.0:
        return ranges
    out = np.array(ranges, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    sig = rplidar_a3_sigma(out[finite]) * scale
    out[finite] += rng.standard_normal(int(finite.sum())).astype(np.float32) * sig
    np.clip(out, 0.0, None, out=out)
    return out


def add_depth_noise(depth, base, k, rng):
    """Quadratic D435 axial depth noise (m): std = base + k * z^2, valid pixels
    only.  Returns a noisy copy (used for BOTH the depth image and the cloud)."""
    if base <= 0.0 and k <= 0.0:
        return depth
    valid = (depth > 1e-3) & np.isfinite(depth)
    std = base + k * depth * depth
    noise = rng.standard_normal(depth.shape).astype(np.float32) * std
    out = depth.copy()
    out[valid] = depth[valid] + noise[valid]
    np.clip(out, 0.0, None, out=out)
    return out


# --------------------------------------------------------------------------- #
#  Gross errors / OUTLIERS (on top of Gaussian noise) -- so SLAM/Nav have to be
#  robust to real failure modes, not just clean white noise.
# --------------------------------------------------------------------------- #
def add_lidar_outliers(ranges, dropout_p, outlier_p, range_min, range_max, rng):
    """RPLIDAR A3 gross failures:
      * dropout_p  -- beam returns NO hit (+inf): dark / specular / absorbing /
        out-of-range surfaces (very common indoors on windows, black objects).
      * outlier_p  -- beam returns a SPURIOUS uniform-random range: mixed pixels
        at edges, secondary reflections, dust, inter-sensor cross-talk.
    Applied per beam; the two are mutually exclusive."""
    if dropout_p <= 0.0 and outlier_p <= 0.0:
        return ranges
    out = np.array(ranges, dtype=np.float32, copy=True)
    u = rng.random(out.shape[0])
    omask = u < outlier_p
    out[omask] = rng.uniform(range_min, range_max,
                             int(omask.sum())).astype(np.float32)
    dmask = (u >= outlier_p) & (u < outlier_p + dropout_p)
    out[dmask] = np.inf
    return out


def add_depth_outliers(depth, hole_p, outlier_p, min_m, max_m, rng):
    """D435 gross failures (valid pixels only):
      * hole_p     -- pixel INVALIDATED (set 0 -> dropped from the cloud): stereo
        holes on textureless / dark / reflective / occluded regions.
      * outlier_p  -- 'flying pixel': depth replaced by a random value: edge mixed
        pixels and IR multipath, which produce points floating off surfaces.
    Returns a copy; feeds both the depth image and the point cloud."""
    if hole_p <= 0.0 and outlier_p <= 0.0:
        return depth
    out = depth.copy()
    valid = (out > 1e-3) & np.isfinite(out)
    u = rng.random(out.shape)
    out[valid & (u < hole_p)] = 0.0
    omask = valid & (u >= hole_p) & (u < hole_p + outlier_p)
    out[omask] = rng.uniform(min_m, max_m, int(omask.sum())).astype(np.float32)
    return out


def capture_rgb(renderer, data, cam, scene_option=None):
    """RGB uint8 (H, W, 3) for ``cam``.  ``renderer`` must be in colour mode."""
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0      # off => ~2x faster
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    return renderer.render()


def capture_depth(renderer, data, cam, scene_option=None):
    """Depth float32 (H, W) in metres.  ``renderer`` must have
    ``enable_depth_rendering()`` set (linear depth along the optical +z axis)."""
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    return renderer.render().astype(np.float32)


def rgb_to_mono(rgb):
    """RGB uint8 (H, W, 3) -> mono8 (H, W) using BT.601 luma."""
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587
            + rgb[..., 2] * 0.114).astype(np.uint8)


def camera_intrinsics(model, cam, width, height):
    """Pinhole (fx, fy, cx, cy) from the camera's fovy at the given image size."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fy = (height / 2.0) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2.0)
    return float(fy), float(fy), width / 2.0, height / 2.0


# --------------------------------------------------------------------------- #
#  Depth image -> XYZRGB point cloud (optical frame: +z fwd, +x right, +y down)
# --------------------------------------------------------------------------- #
def depth_to_xyzrgb_bytes(depth, rgb, fx, fy, cx, cy, max_m, stride=1,
                          lat=0.0, rng=None):
    """Return (data_bytes, n_points) for a PointCloud2 with x,y,z,rgb float32
    fields (point_step 16).  Invalid / clipped depth is dropped.  ``stride``
    decimates pixels (2 -> quarter the points) to cap bandwidth.  ``lat`` adds
    D435-like lateral (x,y) noise with std = lat * z (z-noise should already be
    baked into ``depth`` via add_depth_noise)."""
    d = depth[::stride, ::stride]
    c = rgb[::stride, ::stride]
    h, w = d.shape
    us = (np.arange(w) * stride)
    vs = (np.arange(h) * stride)
    uu, vv = np.meshgrid(us, vs)

    valid = (d > 1e-3) & (d < max_m) & np.isfinite(d)
    z = d[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    if lat > 0.0 and rng is not None and z.size:
        x = x + rng.standard_normal(z.shape).astype(np.float32) * (lat * z)
        y = y + rng.standard_normal(z.shape).astype(np.float32) * (lat * z)

    cc = c[valid].astype(np.uint32)
    rgb_packed = (cc[:, 0] << 16) | (cc[:, 1] << 8) | cc[:, 2]

    pts = np.empty(z.shape[0], dtype=[("x", "<f4"), ("y", "<f4"),
                                      ("z", "<f4"), ("rgb", "<f4")])
    pts["x"] = x.astype(np.float32)
    pts["y"] = y.astype(np.float32)
    pts["z"] = z.astype(np.float32)
    pts["rgb"] = rgb_packed.view(np.float32)
    return pts.tobytes(), int(z.shape[0])


# --------------------------------------------------------------------------- #
#  LiDAR
# --------------------------------------------------------------------------- #
def lidar_beams(model, chassis_name="mabel_chassis"):
    """(beams, chassis_body_id) where beams = [(sensor_adr, site_id), ...] for
    every rangefinder sensor (one per replicated ray)."""
    beams = [(model.sensor_adr[i], model.sensor_objid[i])
             for i in range(model.nsensor)
             if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER]
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chassis_name)
    return beams, chassis


def lidar_cutoff(model, default=12.0):
    vals = [model.sensor_cutoff[i] for i in range(model.nsensor)
            if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER]
    return max(vals) if vals else default


def lidar_beam_table(model, data, beams, chassis):
    """Precompute the STATIC LiDAR scan geometry, once, at node start.

    The LiDAR body is rigidly mounted on the chassis (no joint between them), so
    each beam's direction *in the chassis frame* never changes -- only the
    measured range does.  So the per-beam angles, their ascending sort order, and
    the LaserScan angle_min/angle_increment are all constant and can be computed
    a single time here instead of every tick (the old per-tick Python loop over
    360 beams with an arctan2 + a sort cost ~0.9 ms; this drops the live scan to
    a single vectorised ``sensordata`` gather at ~0.01 ms -- see
    ``controller``/benchmark notes).

    Returns ``(sorted_adr, angle_min, angle_increment, cutoff)`` where
    ``sorted_adr`` is an int array of sensordata addresses in ascending-angle
    order; feed it to :func:`lidar_scan_fast`.
    """
    Rc = data.xmat[chassis].reshape(3, 3)
    cutoff = lidar_cutoff(model)
    adrs = np.empty(len(beams), dtype=np.intp)
    angles = np.empty(len(beams), dtype=np.float64)
    for i, (adr, sid) in enumerate(beams):
        bdir = Rc.T @ data.site_xmat[sid].reshape(3, 3)[:, 2]   # ray dir in chassis
        adrs[i] = adr
        angles[i] = np.arctan2(bdir[1], bdir[0])
    order = np.argsort(angles)
    adrs = adrs[order]
    angles = angles[order]
    inc = float(np.mean(np.diff(angles))) if angles.size > 1 else 0.0
    amin = float(angles[0]) if angles.size else 0.0
    return adrs, amin, inc, float(cutoff)


def lidar_scan_fast(data, sorted_adr, cutoff):
    """Live planar scan as a float32 range array (ascending-angle order, matching
    :func:`lidar_beam_table`).  Out-of-range / no-return beams are +inf.  Pure
    vectorised gather -- no Python per-beam loop."""
    dist = data.sensordata[sorted_adr]
    return np.where((dist >= 0.0) & (dist <= cutoff),
                    dist, np.inf).astype(np.float32)


def lidar_scan(model, data, beams, chassis):
    """Planar scan as sensor_msgs/LaserScan fields, expressed in the chassis
    frame (0 rad = chassis +X).  Returns (angle_min, angle_increment, range_max,
    ranges) with no-return beams set to +inf.  Angles are sorted ascending.

    Kept for callers that don't precompute the beam table; the live node uses the
    faster :func:`lidar_beam_table` + :func:`lidar_scan_fast` pair instead."""
    Rc = data.xmat[chassis].reshape(3, 3)
    cutoff = lidar_cutoff(model)
    pairs = []
    for adr, sid in beams:
        dist = data.sensordata[adr]
        bdir = Rc.T @ data.site_xmat[sid].reshape(3, 3)[:, 2]   # ray dir in chassis
        rng = dist if (0.0 <= dist <= cutoff) else np.inf
        pairs.append((float(np.arctan2(bdir[1], bdir[0])), float(rng)))
    pairs.sort()
    angles = np.array([a for a, _ in pairs])
    ranges = np.array([r for _, r in pairs], dtype=np.float32)
    inc = float(np.mean(np.diff(angles))) if angles.size > 1 else 0.0
    amin = float(angles[0]) if angles.size else 0.0
    return amin, inc, float(cutoff), ranges


# --------------------------------------------------------------------------- #
#  Frame math:  MuJoCo camera/body poses -> ROS optical-frame TF in base_link
# --------------------------------------------------------------------------- #
# MuJoCo camera frame is +x right / +y up / -z forward (OpenGL).  ROS optical is
# +x right / +y down / +z forward.  optical = mujoco @ diag(1, -1, -1).
_MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def mat_to_quat(R):
    """3x3 rotation matrix -> (x, y, z, w) quaternion."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def rel_optical_tf(cam_xpos, cam_xmat, base_xpos, base_xmat):
    """base_link -> camera optical frame as (translation xyz, quat xyzw)."""
    Rwc = cam_xmat.reshape(3, 3) @ _MJ_TO_OPTICAL      # world->optical rotation
    Rwb = base_xmat.reshape(3, 3)                      # world->base rotation
    R_bo = Rwb.T @ Rwc
    t_bo = Rwb.T @ (np.asarray(cam_xpos) - np.asarray(base_xpos))
    return t_bo, mat_to_quat(R_bo)


def rel_body_tf(body_xpos, body_xmat, base_xpos, base_xmat):
    """base_link -> a (non-optical) body/site frame as (xyz, quat xyzw).
    Used for the LiDAR link, which keeps the MuJoCo axis convention."""
    Rwb = base_xmat.reshape(3, 3)
    R_bl = Rwb.T @ body_xmat.reshape(3, 3)
    t_bl = Rwb.T @ (np.asarray(body_xpos) - np.asarray(base_xpos))
    return t_bl, mat_to_quat(R_bl)
