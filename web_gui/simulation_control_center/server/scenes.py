"""
scenes.py — the scene library (registry + thumbnails) and the safe-respawn
solver for "reset somewhere else".

Registry: simulation/mabel_mujoco/scenes/scenes.json
    [{"name","title","xml","thumb","description"}, ...]
(written by the scene tooling; falls back to scanning scenes/mabel_*.xml).

safe_spawn(): picks a collision-free, OPEN base pose in a scene —
  1. rasterise every environment geom's XY footprint (robot's standing z-band)
     onto a 10 cm grid, inflated by the robot's clearance radius;
  2. pick a random free cell (seeded RNG for tests), preferring cells far from
     obstacles and ≥ 1 m from the current pose;
  3. verify in MuJoCo: place the base free joint there, mj_forward, and reject
     on any robot↔environment penetration (floor excluded) — then return
     {x, y, yaw} with yaw facing the open interior.
"""
import json
import logging
import os
import random
import threading

import numpy as np

from . import config as CFG

log = logging.getLogger("scc.scenes")

_SKIP = {"_masstest_scene.xml", "test.xml", "swerve_test.xml", "mabel_bare.xml",
         "mabel_full_fake.xml", "mabel_sim.xml", "home_objects.xml"}

_models = {}                      # scene path -> (MjModel, MjData) cache
_models_lock = threading.Lock()
_reg_cache = {"key": None, "out": None}     # registry() result, mtime-keyed
_reg_lock = threading.Lock()

# picker thumbnails are shown at ~96x60: serve a downscaled JPEG (~10 KB)
# instead of the 640x400 render PNG (~150 KB) — 28 scenes over a slow link
# otherwise means megabytes on first open.
SMALL_W = 256
_SMALL_DIR = None                 # resolved lazily under thumbs/.small


def _registry_key():
    key = []
    for p in (CFG.SCENES_JSON, os.path.join(CFG.SCENES_DIR, "thumbs")):
        try:
            key.append(os.path.getmtime(p))
        except OSError:
            key.append(None)
    return tuple(key)


def registry():
    """List of scenes for the picker (cached on scenes.json / thumbs mtime)."""
    key = _registry_key()
    with _reg_lock:
        if _reg_cache["key"] == key and _reg_cache["out"] is not None:
            return _reg_cache["out"]
    out = _registry_scan()
    with _reg_lock:
        _reg_cache["key"] = key
        _reg_cache["out"] = out
    return out


def _registry_scan():
    entries = []
    if os.path.isfile(CFG.SCENES_JSON):
        try:
            with open(CFG.SCENES_JSON) as f:
                entries = json.load(f)
        except Exception as e:
            log.warning("bad scenes.json: %s", e)
    if not entries and os.path.isdir(CFG.SCENES_DIR):   # fallback: scan the folder
                                               # (absent on a web-only host)
        for fn in sorted(os.listdir(CFG.SCENES_DIR)):
            if not fn.endswith(".xml") or fn in _SKIP:
                continue
            # include-fragments, not scenes (clutter/env parts pulled in by
            # the real scene files)
            if fn.endswith(("_objects.xml", "_env.xml")):
                continue
            slug = fn.replace("mabel_", "").replace(".xml", "")
            entries.append({"name": slug, "title": slug.replace("_", " ").title(),
                            "xml": fn, "description": ""})
    out = []
    for e in entries:
        xml = e.get("xml", "")
        xml_abs = xml if os.path.isabs(xml) else os.path.join(CFG.SCENES_DIR, xml)
        if not os.path.isfile(xml_abs):
            continue
        thumb = e.get("thumb") or ""
        thumb_abs = (thumb if os.path.isabs(thumb)
                     else os.path.join(CFG.SCENES_DIR, thumb)) if thumb else ""
        out.append({**e, "xml": xml_abs,
                    "thumb": thumb_abs if os.path.isfile(thumb_abs) else None})
    return out


def by_name(name):
    for e in registry():
        if e["name"] == name:
            return e
    return None


def small_thumb(entry):
    """Path of the downscaled JPEG for a registry entry (built on demand,
    cached next to the renders); falls back to the full PNG."""
    src = entry.get("thumb")
    if not src:
        return None
    dst_dir = os.path.join(CFG.SCENES_DIR, "thumbs", ".small")
    dst = os.path.join(dst_dir, entry["name"] + ".jpg")
    try:
        if os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
    except OSError:
        pass
    try:
        from PIL import Image
        os.makedirs(dst_dir, exist_ok=True)
        im = Image.open(src)
        im.thumbnail((SMALL_W, SMALL_W))
        im.convert("RGB").save(dst, "JPEG", quality=82)
        return dst
    except Exception as e:
        log.debug("small thumb %s: %s", entry["name"], e)
        return src


def warm():
    """Pre-build everything the scene picker needs so the first click is
    instant: the registry cache + every downscaled thumbnail. Called from a
    daemon thread at app startup."""
    try:
        entries = registry()
        for e in entries:
            small_thumb(e)
        log.info("scene picker warm — %d scenes, thumbs cached", len(entries))
    except Exception as e:
        log.warning("scene warm-up failed: %s", e)


def current_scene_path(fallback=None):
    """What the BRIDGE actually loaded (its state-file sidecar)."""
    try:
        with open(CFG.STATE_FILE + ".scene") as f:
            p = f.readline().strip()
        if p and os.path.isfile(p):
            return p
    except OSError:
        pass
    return fallback or CFG.DEFAULT_SCENE


def name_for_path(path):
    base = os.path.splitext(os.path.basename(path or ""))[0]
    slug = base.replace("mabel_", "")
    for e in registry():
        if os.path.abspath(e["xml"]) == os.path.abspath(path or ""):
            return e["name"]
    return slug


# ── safe respawn ─────────────────────────────────────────────────────────────
ROBOT_CLEAR_R = 0.80        # chassis half-diagonal + arm swing + margin (m)
GRID_RES = 0.10
Z_BAND = (0.03, 1.90)       # obstacles overlapping the robot's standing height


def _load(scene_path):
    import mujoco
    with _models_lock:
        if scene_path not in _models:
            m = mujoco.MjModel.from_xml_path(scene_path)
            _models[scene_path] = (m, mujoco.MjData(m))
        return _models[scene_path]


def _robot_bodies(model):
    """ids of the robot subtree (descendants of the base_root free joint's body)."""
    import mujoco
    try:
        j = model.joint("base_root")
        root = int(model.jnt_bodyid[j.id])
    except Exception:
        return set()
    kids = {root}
    changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if b not in kids and int(model.body_parentid[b]) in kids:
                kids.add(b)
                changed = True
    return kids


def _geom_xy_aabb(model, data, g):
    """Conservative world-frame XY AABB + z-range for one geom."""
    import mujoco
    t = model.geom_type[g]
    pos = data.geom_xpos[g]
    R = data.geom_xmat[g].reshape(3, 3)
    size = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[sx, sy, sz] for sx in (-size[0], size[0])
                            for sy in (-size[1], size[1])
                            for sz in (-size[2], size[2])])
        w = corners @ R.T + pos
        return (w[:, 0].min(), w[:, 0].max(), w[:, 1].min(), w[:, 1].max(),
                w[:, 2].min(), w[:, 2].max())
    if t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER,
             mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        r = float(max(size[0], size[1] if size[1] > 0 else 0))
        h = float(size[1] if t == mujoco.mjtGeom.mjGEOM_SPHERE else
                  (size[1] if size[1] > 0 else size[0]))
        zr = float(np.abs(R[2]).max() * max(h, r) + r)
        return (pos[0] - r, pos[0] + r, pos[1] - r, pos[1] + r,
                pos[2] - zr, pos[2] + zr)
    r = float(model.geom_rbound[g]) or 0.05          # mesh/other: bounding sphere
    return (pos[0] - r, pos[0] + r, pos[1] - r, pos[1] + r,
            pos[2] - r, pos[2] + r)


def safe_spawn(scene_path, avoid_xy=None, min_dist=1.0, seed=None,
               n_candidates=40):
    """Return {"x","y","yaw"} for an open, collision-free base pose, or None."""
    import mujoco
    model, data = _load(scene_path)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    robot = _robot_bodies(model)
    plane_geoms, obstacles = set(), []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if body in robot:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            plane_geoms.add(g)
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue                                       # visual-only
        x0, x1, y0, y1, z0, z1 = _geom_xy_aabb(model, data, g)
        if z1 < Z_BAND[0] or z0 > Z_BAND[1]:
            continue                                       # under/over the robot
        obstacles.append((x0, x1, y0, y1))

    # working area: the floor plane's half-extents if finite, else env bounds
    bx = by = 6.0
    for g in plane_geoms:
        sx, sy = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        if sx > 0 and sy > 0:
            bx, by = min(bx, sx), min(by, sy)
    if obstacles and (bx >= 6.0 and by >= 6.0):
        arr = np.array(obstacles)
        bx = min(6.0, max(2.0, float(np.abs(arr[:, :2]).max())))
        by = min(6.0, max(2.0, float(np.abs(arr[:, 2:]).max())))

    nx = max(4, int((2 * bx) / GRID_RES))
    ny = max(4, int((2 * by) / GRID_RES))
    xs = np.linspace(-bx, bx, nx)
    ys = np.linspace(-by, by, ny)
    occ = np.zeros((ny, nx), dtype=bool)
    for (x0, x1, y0, y1) in obstacles:
        xi0 = np.searchsorted(xs, x0 - ROBOT_CLEAR_R)
        xi1 = np.searchsorted(xs, x1 + ROBOT_CLEAR_R)
        yi0 = np.searchsorted(ys, y0 - ROBOT_CLEAR_R)
        yi1 = np.searchsorted(ys, y1 + ROBOT_CLEAR_R)
        occ[yi0:yi1, xi0:xi1] = True
    # keep a wall margin even in wall-less scenes
    edge = max(1, int(ROBOT_CLEAR_R / GRID_RES))
    occ[:edge, :] = occ[-edge:, :] = True
    occ[:, :edge] = occ[:, -edge:] = True

    free_y, free_x = np.nonzero(~occ)
    if len(free_x) == 0:
        return None
    cand = np.stack([xs[free_x], ys[free_y]], axis=1)
    if avoid_xy is not None:
        d = np.hypot(cand[:, 0] - avoid_xy[0], cand[:, 1] - avoid_xy[1])
        far = cand[d >= min_dist]
        if len(far):
            cand = far

    rng = random.Random(seed)
    j = model.joint("base_root")
    adr = int(j.qposadr[0])
    z0 = float(model.qpos0[adr + 2])
    interior = cand.mean(axis=0)

    order = list(range(len(cand)))
    rng.shuffle(order)
    for idx in order[:n_candidates]:
        x, y = float(cand[idx][0]), float(cand[idx][1])
        yaw = float(np.arctan2(interior[1] - y, interior[0] - x))
        if not np.isfinite(yaw):
            yaw = rng.uniform(-np.pi, np.pi)
        mujoco.mj_resetData(model, data)
        data.qpos[adr:adr + 3] = (x, y, z0)
        data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        bad = False
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            r1, r2 = b1 in robot, b2 in robot
            if r1 == r2:                                   # env↔env or self
                continue
            envg = g2 if r1 else g1
            if envg in plane_geoms:                        # wheels on the floor
                continue
            if float(con.dist) < -1e-6:
                bad = True
                break
        if not bad:
            return {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4)}
    return None
