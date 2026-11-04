"""
3D point clouds + camera extrinsics for the recorded depth cameras, world frame.

Episodes store depth (uint16 mm) + the full generalized state ``obs/qpos``. The
camera world poses come from forward kinematics on that qpos, and the depth is
unprojected with the camera intrinsics (``cam_fovy``) and coloured by the RGB
frame. Output is the MapStream CLOUD (0x05) binary the viewer already decodes —
the Rerun-style coloured depth cloud, registered to the rig.

Two execution modes, *identical output*:

* **With MuJoCo + the scene model** (dev box / collection host): load the model,
  run FK per frame, read ``cam_xpos`` / ``cam_xmat``. The intrinsics + per-frame
  poses are ALSO written to a tiny ``<episode>.camcache.npz`` sidecar.
* **Cache only** (e.g. a web-only host, which has neither MuJoCo nor the ~300 MB
  compiled scene): read that sidecar and skip MuJoCo entirely. The depth→cloud
  unprojection is pure NumPy, so every camera feature works with no model.

Build the caches ahead of time on a MuJoCo host with::

    python -m data_studio.server.camcloud --dataset studio_sample

then ship the ``*.camcache.npz`` files alongside the episodes.
"""
import json
import os
import struct

import numpy as np

from . import config as _studio_cfg, h5pool   # noqa: F401  (bootstraps sys.path)
from data_collection import config as CFG       # SCENE_XML, DEPTH_SCALE, DEPTH_MAX_M

try:
    import cv2
except Exception:
    cv2 = None

_MODEL = None
_CACHE = {}          # episode path -> in-process cache dict


# ── MuJoCo model (only where the scene + mujoco are available) ────────────────
def _has_model():
    """True iff we can build extrinsics live: mujoco importable AND the scene XML
    on disk. False on a cache-only deployment, which then serves the sidecars."""
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return os.path.exists(CFG.SCENE_XML)


def _model():
    """Compiled MuJoCo model, cached in-process AND on disk. Compiling the full
    scene XML (mesh convexification) costs ~30s; we save the compiled model to a
    ``.mjb`` binary so every later load (this process or a fresh server) is
    near-instant. Re-compiles only if the XML is newer than the cache."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import os, tempfile, hashlib, mujoco
    xml = CFG.SCENE_XML
    tag = hashlib.md5(os.path.abspath(xml).encode()).hexdigest()[:10]
    mjb = os.path.join(tempfile.gettempdir(), f"mabel_scene_{tag}.mjb")
    try:
        if os.path.exists(mjb) and os.path.getmtime(mjb) >= os.path.getmtime(xml):
            _MODEL = mujoco.MjModel.from_binary_path(mjb)
            return _MODEL
    except Exception:
        pass
    _MODEL = mujoco.MjModel.from_xml_path(xml)
    try:
        mujoco.mj_saveModel(_MODEL, mjb)
    except Exception:
        pass
    return _MODEL


# ── per-episode camera cache (intrinsics + per-frame world poses) ─────────────
def _cache_path(path):
    return path + ".camcache.npz"


def _load_cache(path):
    """Cache dict for an episode, or None. In-process memo → ``.camcache.npz``
    sidecar → (only on a model host) build it. The dict is::

        {"calib": [..], "cams": [name..], "T": int,
         "pos": {name: (T,3) f32}, "mat": {name: (T,9) f32},
         "fovy": {name: deg}}
    """
    d = _CACHE.get(path)
    if d is not None:
        return d
    cp = _cache_path(path)
    if os.path.exists(cp):
        try:
            z = np.load(cp, allow_pickle=False)
            calib = json.loads(z["calib_json"].item())
            cams = [str(c) for c in z["cams"].tolist()]
            T = int(z["T"].item())
            pos = {nm: z[f"pos__{nm}"] for nm in cams}
            mat = {nm: z[f"mat__{nm}"] for nm in cams}
            fovy = {c["name"]: float(c["fovy_deg"]) for c in calib}
            d = {"calib": calib, "cams": cams, "T": T, "pos": pos, "mat": mat, "fovy": fovy}
            _CACHE[path] = d
            return d
        except Exception:
            pass
    # No usable sidecar: compose extrinsics from the episode's OWN recorded body
    # poses (obs/pose/<body>) + the static rig calibration — pure NumPy, no
    # MuJoCo. This is what makes camera projection + clouds work on a model-less
    # host for ANY episode, including ones materialized/edited there.
    d = _extrinsics_from_poses(path)
    if d is not None:
        _CACHE[path] = d
        return d
    if _has_model():
        return _build_cache(path)
    return None


def _build_cache(path):
    """Compute calibration + per-frame camera extrinsics via MuJoCo and write the
    ``.camcache.npz`` sidecar. Only valid on a host with mujoco + the scene."""
    calib = _calibration_via_model(path)
    extr = _extrinsics_all_via_model(path)
    T = int(extr["T"])
    cams = list(extr["cams"].keys())
    pos = {nm: np.asarray(extr["cams"][nm]["pos"], np.float32).reshape(T, 3) for nm in cams}
    mat = {nm: np.asarray(extr["cams"][nm]["mat"], np.float32).reshape(T, 9) for nm in cams}
    try:
        np.savez(_cache_path(path),
                 calib_json=np.array(json.dumps(calib)),
                 cams=np.array(cams), T=np.array(T),
                 **{f"pos__{nm}": pos[nm] for nm in cams},
                 **{f"mat__{nm}": mat[nm] for nm in cams})
    except Exception:
        pass
    fovy = {c["name"]: float(c["fovy_deg"]) for c in calib}
    d = {"calib": calib, "cams": cams, "T": T, "pos": pos, "mat": mat, "fovy": fovy}
    _CACHE[path] = d
    return d


# ── pose-composition extrinsics (no MuJoCo; works for ANY episode) ────────────
# Cameras are rigidly mounted on tracked bodies (palms, head, chassis) whose world
# pose the recorder logs every frame as obs/pose/<body> = [x,y,z, qw,qx,qy,qz].
# Composing that with each camera's STATIC mount offset (from mabel_camera_rig.json,
# extracted once from the model) reproduces MuJoCo's cam_xpos/cam_xmat to <2 mm /
# <0.3°, with no model on the box. The torso cam (body_cam) is the one camera whose
# mount body isn't tracked, so it's simply omitted on a model-less host.
_RIG = None


def _rig():
    """Static per-robot camera calibration (intrinsics + mount offsets + pose_key),
    shipped as mabel_camera_rig.json next to this module. Same for every episode."""
    global _RIG
    if _RIG is None:
        rp = os.path.join(os.path.dirname(__file__), "mabel_camera_rig.json")
        try:
            with open(rp) as fh:
                _RIG = json.load(fh)
        except Exception:
            _RIG = []
    return _RIG


def _q2m(q):
    """3x3 rotation matrix from a MuJoCo (w,x,y,z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _extrinsics_from_poses(path):
    """Per-frame camera world poses composed from the episode's recorded mount-body
    poses and the static rig. Returns a cache-shaped dict (calib is the full rig;
    cams/pos/mat cover only cameras whose mount body was tracked) or None."""
    rig = _rig()
    if not rig:
        return None
    with h5pool.reading(path) as f:
        avail = {}
        for c in rig:
            pk = c.get("pose_key")
            if pk and pk not in avail and f"obs/pose/{pk}" in f:
                avail[pk] = np.asarray(f[f"obs/pose/{pk}"][:], dtype=np.float64)
    if not avail:
        return None
    T = int(len(next(iter(avail.values()))))
    pos, mat, cams, fovy = {}, {}, [], {}
    for c in rig:
        pk = c.get("pose_key")
        if not pk or pk not in avail:
            continue
        nm = c["name"]
        cpl = np.asarray(c["mount_pos"], dtype=np.float64)     # cam pos in body frame
        Rcl = _q2m(c["mount_quat"])                            # body <- cam rotation
        P = avail[pk]
        cp = np.empty((T, 3), np.float32)
        cm = np.empty((T, 9), np.float32)
        for i in range(T):
            Rb = _q2m(P[i, 3:7])                               # world <- body
            cp[i] = P[i, :3] + Rb @ cpl
            cm[i] = (Rb @ Rcl).reshape(-1)                     # world <- cam
        pos[nm] = cp; mat[nm] = cm; cams.append(nm); fovy[nm] = float(c["fovy_deg"])
    if not cams:
        return None
    return {"calib": rig, "cams": cams, "T": T, "pos": pos, "mat": mat, "fovy": fovy}


# ── public API (cache-first; model only as a fallback on dev hosts) ───────────
def camera_cloud(path, cam_name, frame, stride=4, max_points=60000):
    """Return a CLOUD (0x05) of world-frame, RGB-coloured points for one camera
    frame. Empty cloud if the camera has no depth, no cache/model, or anything is
    missing. Pure NumPy unprojection — needs only the cached pose + intrinsics."""
    c = _load_cache(path)
    if c is None or cam_name not in c["pos"]:
        return _empty()
    fovy = float(c["fovy"].get(cam_name, 0.0))
    if fovy <= 0.0:
        return _empty()

    dkey = f"obs/cam/{cam_name}/depth"
    rkey = f"obs/cam/{cam_name}/rgb"
    with h5pool.reading(path) as f:
        if dkey not in f:
            return _empty()
        n = f[dkey].shape[0]
        i = max(0, min(int(frame), n - 1))
        depth = np.asarray(f[dkey][i], dtype=np.uint16)
        rgb = None
        if rkey in f and cv2 is not None:
            rgb = cv2.imdecode(np.frombuffer(np.asarray(f[rkey][i], dtype=np.uint8), np.uint8),
                               cv2.IMREAD_COLOR)  # BGR

    j = max(0, min(int(frame), c["T"] - 1))
    cam_pos = np.asarray(c["pos"][cam_name][j], dtype=np.float64)            # world
    R = np.asarray(c["mat"][cam_name][j], dtype=np.float64).reshape(3, 3)    # world <- camera

    H, W = depth.shape
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy                                             # MuJoCo: square pixels
    cx, cy = W / 2.0, H / 2.0

    rr = np.arange(0, H, stride)
    cc = np.arange(0, W, stride)
    gc, gr = np.meshgrid(cc, rr)
    d = depth[gr, gc].astype(np.float32) / CFG.DEPTH_SCALE      # metres
    valid = (d > 0.05) & (d < CFG.DEPTH_MAX_M - 0.05)
    gr, gc, d = gr[valid], gc[valid], d[valid]
    if d.size == 0:
        return _empty()
    if d.size > max_points:
        sel = np.random.default_rng(0).choice(d.size, max_points, replace=False)
        gr, gc, d = gr[sel], gc[sel], d[sel]

    x_cam = (gc - cx) / fx * d
    y_cam = (cy - gr) / fy * d                          # image row down -> +Y up
    z_cam = -d                                          # camera looks down -Z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    pts_world = cam_pos + pts_cam @ R.T                 # (N,3)

    if rgb is not None:
        cols = rgb[gr, gc]                              # BGR
        rgb_u8 = cols[:, ::-1].astype(np.uint8)        # -> RGB
    else:
        t = (d - d.min()) / (np.ptp(d) + 1e-6)
        rgb_u8 = (np.stack([t, 0.5 * np.ones_like(t), 1 - t], 1) * 255).astype(np.uint8)
    return _pack(pts_world.astype(np.float32), rgb_u8)


# ── SLAM-style reconstruction: accumulated dense cloud + surface mesh ─────────
# A single depth camera (the head ZED or the base forward cam) is unprojected at
# several keyframes across the episode and fused into ONE world-frame model — the
# static map a visual-SLAM front-end (e.g. cuVSLAM / nvblox) would build. Two
# representations, both pure NumPy (no open3d): a voxel-downsampled dense POINT
# CLOUD, and a coloured triangle MESH from organized-depth triangulation (adjacent
# valid pixels stitched into quads, edges across depth discontinuities dropped).
def _resolve_recon_cam(c, source):
    """Map a logical source ('head'/'base') — or an explicit camera name — to one
    of the episode's depth cameras. Head → the ZED/stereo head cam; base → the
    forward body cam."""
    cams = [nm for nm in c["cams"] if float(c["fovy"].get(nm, 0.0)) > 0.0]
    if not cams:
        return None
    s = (source or "head").lower()

    def pick(subs):
        for sub in subs:
            for nm in cams:
                if sub in nm.lower():
                    return nm
        return None

    if s in ("head", "zed", "stereo"):
        return pick(["zed", "head", "stereo"]) or cams[0]
    if s in ("base", "chassis", "torso", "body", "front"):
        return pick(["body_front", "front", "body_cam", "body", "base", "chassis", "torso"]) or cams[0]
    return source if source in cams else (pick([s]) or cams[0])


def _frame_grid(path, c, cam_name, frame, ps):
    """Organized ``(Hs,Ws)`` world points + RGB colours + valid mask + depth (m)
    for one camera frame, pixel-strided by ``ps``. None if unavailable. Same
    intrinsic model + world unprojection as :func:`camera_cloud`, kept 2-D so the
    mesher can stitch neighbouring pixels."""
    fovy = float(c["fovy"].get(cam_name, 0.0))
    if fovy <= 0.0:
        return None
    dkey = f"obs/cam/{cam_name}/depth"
    rkey = f"obs/cam/{cam_name}/rgb"
    with h5pool.reading(path) as f:
        if dkey not in f:
            return None
        n = f[dkey].shape[0]
        i = max(0, min(int(frame), n - 1))
        depth = np.asarray(f[dkey][i], dtype=np.uint16)
        rgb = None
        if rkey in f and cv2 is not None:
            rgb = cv2.imdecode(np.frombuffer(np.asarray(f[rkey][i], dtype=np.uint8), np.uint8),
                               cv2.IMREAD_COLOR)  # BGR
    j = max(0, min(int(frame), c["T"] - 1))
    cam_pos = np.asarray(c["pos"][cam_name][j], dtype=np.float64)
    R = np.asarray(c["mat"][cam_name][j], dtype=np.float64).reshape(3, 3)

    H, W = depth.shape
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0
    ps = max(1, int(ps))
    rr = np.arange(0, H, ps)
    cc = np.arange(0, W, ps)
    gc, gr = np.meshgrid(cc, rr)                         # (Hs,Ws)
    d = depth[gr, gc].astype(np.float32) / CFG.DEPTH_SCALE
    valid = (d > 0.05) & (d < CFG.DEPTH_MAX_M - 0.05)
    x = (gc - cx) / fx * d
    y = (cy - gr) / fy * d                               # image row down -> +Y up
    z = -d                                               # camera looks down -Z
    pts_cam = np.stack([x, y, z], axis=-1)               # (Hs,Ws,3)
    P = cam_pos + pts_cam @ R.T                          # (Hs,Ws,3) world
    if rgb is not None:
        col = np.ascontiguousarray(rgb[gr, gc][:, :, ::-1]).astype(np.uint8)   # BGR->RGB
    else:
        t = np.clip((d - 0.2) / 2.0, 0.0, 1.0)
        col = (np.stack([t, 0.5 * np.ones_like(t), 1 - t], -1) * 255).astype(np.uint8)
    return P, col, valid, d


def _mesh_from_grid(P, col, valid, D, dz):
    """Triangulate an organized point grid. Each 2×2 cell whose four corners are
    valid AND depth-consistent (spread < ``dz`` + a small relative term) becomes
    two triangles. Returns ``(verts(V,3) f32, colours(V,3) u8, faces(F,3) i64)``."""
    Hs, Ws = valid.shape
    nv = int(valid.sum())
    if nv == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.int64))
    idx = np.full((Hs, Ws), -1, dtype=np.int64)
    idx[valid] = np.arange(nv)
    verts = P[valid].astype(np.float32)
    vcol = col[valid].astype(np.uint8)

    v00 = valid[:-1, :-1]; v01 = valid[:-1, 1:]; v10 = valid[1:, :-1]; v11 = valid[1:, 1:]
    quad = v00 & v01 & v10 & v11
    d00 = D[:-1, :-1]; d01 = D[:-1, 1:]; d10 = D[1:, :-1]; d11 = D[1:, 1:]
    dmax = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
    dmin = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11))
    quad = quad & ((dmax - dmin) < (dz + 0.03 * dmin))   # drop triangles spanning a depth edge
    rq, cq = np.nonzero(quad)
    if rq.size == 0:
        return verts, vcol, np.zeros((0, 3), np.int64)
    i00 = idx[rq, cq]; i01 = idx[rq, cq + 1]; i10 = idx[rq + 1, cq]; i11 = idx[rq + 1, cq + 1]
    t1 = np.stack([i00, i10, i11], axis=1)
    t2 = np.stack([i00, i11, i01], axis=1)
    faces = np.concatenate([t1, t2], axis=0)
    # drop vertices no triangle references (isolated far pixels) so the mesh bbox
    # is the actual surface, not inflated by invisible points
    used = np.unique(faces)
    remap = np.full(nv, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return verts[used], vcol[used], remap[faces]


def reconstruction(path, source="head", kind="cloud", frames=8, ps=0, voxel=0.02, dz=0.08):
    """Fused world-frame reconstruction from ONE camera over ``frames`` evenly
    spaced keyframes. ``kind='cloud'`` → voxel-downsampled CLOUD (0x05);
    ``kind='mesh'`` → coloured MESH (0x06)."""
    empty = _empty_mesh() if kind == "mesh" else _empty()
    c = _load_cache(path)
    if c is None:
        return empty
    cam = _resolve_recon_cam(c, source)
    if cam is None:
        return empty
    dkey = f"obs/cam/{cam}/depth"
    with h5pool.reading(path) as f:
        nd = f[dkey].shape[0] if dkey in f else 0
    if nd == 0:
        return empty
    N = min(int(c["T"]), int(nd))
    k = max(1, int(frames))
    idxs = np.unique(np.linspace(0, N - 1, k).astype(int))

    if kind == "mesh":
        pstride = int(ps) if ps else 4
        Vs, Cs, Fs, off = [], [], [], 0
        for fr in idxs:
            g = _frame_grid(path, c, cam, int(fr), pstride)
            if g is None:
                continue
            P, col, valid, D = g
            v, cc2, fa = _mesh_from_grid(P, col, valid, D, dz)
            if v.shape[0] == 0 or fa.shape[0] == 0:
                continue
            Vs.append(v); Cs.append(cc2); Fs.append(fa + off); off += v.shape[0]
        if not Vs:
            return _empty_mesh()
        V = np.concatenate(Vs); C = np.concatenate(Cs); F = np.concatenate(Fs)
        return _pack_mesh(V.astype(np.float32), C.astype(np.uint8), F.astype(np.uint32))

    # dense accumulated point cloud
    pstride = int(ps) if ps else 3
    Ps, Cs = [], []
    for fr in idxs:
        g = _frame_grid(path, c, cam, int(fr), pstride)
        if g is None:
            continue
        P, col, valid, _ = g
        Ps.append(P[valid]); Cs.append(col[valid])
    if not Ps:
        return _empty()
    P = np.concatenate(Ps).astype(np.float32)
    C = np.concatenate(Cs).astype(np.uint8)
    if voxel and voxel > 0:
        key = np.floor(P / voxel).astype(np.int64)
        _, ui = np.unique(key, axis=0, return_index=True)
        P = P[ui]; C = C[ui]
    if P.shape[0] > 220000:
        sel = np.random.default_rng(0).choice(P.shape[0], 220000, replace=False)
        P = P[sel]; C = C[sel]
    return _pack(P, C)


def cameras_calibration(path):
    """Static per-camera intrinsics + mount (Foxglove-style). Cache-first. Only
    cameras that actually have per-frame extrinsics for THIS episode are returned,
    so the client never builds a frustum it can't pose (e.g. the torso cam on a
    model-less host)."""
    c = _load_cache(path)
    if not c:
        return []
    have = set(c["cams"])
    return [x for x in c["calib"] if x["name"] in have]


def cameras_extrinsics(path, frame):
    """Per-frame WORLD pose of every recorded camera. Cache-first."""
    c = _load_cache(path)
    if not c:
        return {}
    i = max(0, min(int(frame), c["T"] - 1))
    return {nm: {"pos": c["pos"][nm][i].astype(float).tolist(),
                 "mat": c["mat"][nm][i].astype(float).tolist()} for nm in c["cams"]}


def cameras_extrinsics_all(path):
    """Camera world poses for EVERY frame in one shot, so the client indexes
    per-frame with zero network/compute during playback. Cache-first.
    Returns {"T": int, "cams": {name: {"pos": [T*3], "mat": [T*9]}}}."""
    c = _load_cache(path)
    if not c:
        return {"T": 0, "cams": {}}
    cams = {nm: {"pos": c["pos"][nm].reshape(-1).astype(float).tolist(),
                 "mat": c["mat"][nm].reshape(-1).astype(float).tolist()} for nm in c["cams"]}
    return {"T": c["T"], "cams": cams}


# ── model-backed builders (dev hosts; populate the cache) ─────────────────────
def _schema_cams(path):
    with h5pool.reading(path) as f:
        if "schema_json" in f.attrs:
            try:
                return json.loads(f.attrs["schema_json"]).get("cameras", [])
            except Exception:
                return []
    return []


def _calibration_via_model(path):
    """Per-camera intrinsics + mount from the model (the sim cameras are ideal
    pinhole — no lens distortion; real cams would carry k1..k3,p1,p2)."""
    import mujoco
    model = _model()
    out = []
    for c in _schema_cams(path):
        if not c.get("rgb"):
            continue
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c["name"])
        if cid < 0:
            continue
        w, h = int(c.get("w", 480)), int(c.get("h", 270))
        fovy = float(model.cam_fovy[cid])
        fy = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        fx = fy                                   # MuJoCo square pixels
        mount = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.cam_bodyid[cid]))
        # mount pose RELATIVE to its parent body (the extrinsic offset on the robot)
        mp = [float(v) for v in model.cam_pos[cid]]
        mq = [float(v) for v in model.cam_quat[cid]]   # wxyz, body<-cam
        out.append({
            "name": c["name"], "w": w, "h": h, "fovy_deg": round(fovy, 2),
            "fovx_deg": round(np.rad2deg(2 * np.arctan((w / 2.0) / fx)), 2),
            "fx": round(fx, 2), "fy": round(fy, 2), "cx": w / 2.0, "cy": h / 2.0,
            "has_depth": bool(c.get("depth")), "mount_body": mount,
            "mount_pos": mp, "mount_quat": mq,
            # Brown-Conrady [k1,k2,p1,p2,k3]. MuJoCo cameras are ideal pinhole, so
            # these are zero; the viewer applies the model regardless, so real
            # cameras with measured coeffs project (curve) correctly.
            "dist_model": "brown-conrady",
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "distortion": "none (ideal pinhole — sim)",
        })
    return out


def _extrinsics_all_via_model(path):
    """Camera world poses for every frame via FK on ``obs/qpos`` (model host)."""
    import mujoco
    model = _model()
    with h5pool.reading(path) as f:
        if "obs/qpos" not in f:
            return {"T": 0, "cams": {}}
        Q = np.asarray(f["obs/qpos"][:], dtype=np.float64)
    T = int(Q.shape[0])
    cids = {}
    for c in _schema_cams(path):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c["name"])
        if cid >= 0:
            cids[c["name"]] = cid
    data = mujoco.MjData(model)
    out = {nm: {"pos": [0.0] * (T * 3), "mat": [0.0] * (T * 9)} for nm in cids}
    for i in range(T):
        q = Q[i]
        if q.shape[0] == model.nq:
            data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)               # camera/light poses (cheaper than mj_forward)
        for nm, cid in cids.items():
            xp = data.cam_xpos[cid]; xm = data.cam_xmat[cid]
            o3 = i * 3; o9 = i * 9; po = out[nm]["pos"]; ma = out[nm]["mat"]
            po[o3] = float(xp[0]); po[o3 + 1] = float(xp[1]); po[o3 + 2] = float(xp[2])
            for k in range(9):
                ma[o9 + k] = float(xm[k])
    return {"T": T, "cams": out}


# ── cache builder CLI ─────────────────────────────────────────────────────────
def build_dataset_cache(dataset=None):
    """(Re)build the ``.camcache.npz`` sidecar for every episode (one dataset, or
    all). Requires mujoco + the scene model."""
    from . import catalog
    if not _has_model():
        raise SystemExit("need mujoco + the scene XML to build camera caches")
    cat = catalog.full_catalog()
    total = 0
    for ds in cat["datasets"]:
        if dataset and ds["name"] != dataset:
            continue
        for ep in ds["episodes"]:
            p = catalog.episode_path(ds["name"], ep["file"])
            _CACHE.pop(p, None)
            d = _build_cache(p)
            total += 1
            print(f"  cached {ds['name']}/{ep['file']}  T={d['T']}  cams={len(d['cams'])}")
    print(f"built {total} episode cache(s)")


def _pack(pts, cols):
    n = pts.shape[0]
    head = struct.pack("<BBHI", 0x05, 1, 0, n)
    buf = np.empty((n, 16), dtype=np.uint8)
    buf[:, 0:12] = pts.astype("<f4").view(np.uint8).reshape(n, 12)
    buf[:, 12:15] = cols
    buf[:, 15] = 255
    return head + buf.tobytes()


def _empty():
    return struct.pack("<BBHI", 0x05, 1, 0, 0)


def _pack_mesh(V, C, F):
    """MESH (0x06): 12-byte header ``<BBHII`` (tag,ver,pad,nverts,nfaces) then
    nverts×(f32 xyz) + nverts×(u8 rgba) + nfaces×(u32 v0v1v2). All sections stay
    4-byte aligned so the browser can view them as typed arrays."""
    nv = int(V.shape[0]); nf = int(F.shape[0])
    head = struct.pack("<BBHII", 0x06, 1, 0, nv, nf)
    ca = np.empty((nv, 4), dtype=np.uint8)
    ca[:, :3] = C
    ca[:, 3] = 255
    return head + V.astype("<f4").tobytes() + ca.tobytes() + F.astype("<u4").tobytes()


def _empty_mesh():
    return struct.pack("<BBHII", 0x06, 1, 0, 0, 0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build camera extrinsics caches.")
    ap.add_argument("--dataset", default=None, help="only this dataset (default: all)")
    build_dataset_cache(ap.parse_args().dataset)
