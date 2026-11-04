"""map_library.py — saved-room maps for the /map stream (the "known map").

exploration.launch.py (frontier_explorer) saves a finished room as:
    <name>.pgm + <name>.yaml   2D occupancy map (nav2 map_server format)
    <name>.ply                 nvblox 3D mesh (saved via /nvblox_node/save_ply)
    <name>.nvblx               nvblox serialized map (for relocalization, load_map)

save_map() writes that SAME layout from the live map on demand (the SAVE_MAP wire
message), so an operator-saved room and an exploration-saved one are byte-alike;
delete_map() removes the set.

This module lists those maps and turns a chosen one into the SAME wire frames the
live stream uses — GRID 0x04 (2D) + MESH 0x01 (3D point cloud from the .ply) — so
the client renders a loaded room exactly like a live one, with the live sensor
cloud (0x05) overlaid on top. Pure file IO + numpy; no ROS (the bridge calls it).

Map dir: $MABEL_MAP_DIR (default ~/mabel_maps, falling back to ~ so the existing
~/mabel_map.{pgm,yaml} is found). A "map" is any basename with a .yaml beside it.
"""
from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np

from map_stream import encode_occupancy, encode_mesh_chunks, MAP_CHUNK, MAP_MAX_POINTS

log = logging.getLogger(__name__)

# A map name becomes a filename in the map dir, so it is a strict allowlist rather
# than a traversal blocklist: '..', 'a/b', '' and every dotted form fail it. Every
# entry point that turns a client string into a path goes through _check_name().
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Every extension that belongs to one saved map — delete_map removes the whole set.
MAP_EXTS = (".pgm", ".yaml", ".ply", ".nvblx")


def _check_name(name: str) -> str:
    """Validate a client-supplied map name → the bare name. Raises ValueError."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            "invalid map name %r: use 1-64 chars of [A-Za-z0-9_-] only" % (name,))
    return name


def map_dirs() -> list[str]:
    """Directories searched for saved maps (first existing wins for a given name)."""
    env = os.getenv("MABEL_MAP_DIR", "")
    dirs = [env] if env else []
    dirs += [os.path.expanduser("~/mabel_maps"), os.path.expanduser("~")]
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d)) if d else d
        if d and d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out


def save_dir() -> str:
    """Where NEW maps are written: $MABEL_MAP_DIR, else ~/mabel_maps. Unlike
    map_dirs() this never falls back to ~ (we do not litter the home dir) and does
    not require the dir to exist yet — save_map creates it."""
    return os.path.abspath(os.path.expanduser(
        os.getenv("MABEL_MAP_DIR", "") or "~/mabel_maps"))


def list_maps() -> list[dict]:
    """All saved maps: {name, has_mesh, has_reloc, dir, saved_at, size_bytes}.
    A map = a <name>.yaml file with a <name>.pgm beside it. saved_at (epoch s) and
    size_bytes span the WHOLE set of <name>.* files, so a map with a mesh reports
    its true footprint. Newest first."""
    out, seen = [], set()
    for d in map_dirs():
        for y in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            name = os.path.splitext(os.path.basename(y))[0]
            if name in seen:
                continue
            # only count it as a map if a .pgm image sits beside the yaml
            if not os.path.exists(os.path.join(d, name + ".pgm")):
                continue
            seen.add(name)
            saved_at, size = 0.0, 0
            for ext in MAP_EXTS:
                try:
                    st = os.stat(os.path.join(d, name + ext))
                except OSError:
                    continue
                size += st.st_size
                saved_at = max(saved_at, st.st_mtime)
            out.append({
                "name": name,
                "has_mesh": os.path.exists(os.path.join(d, name + ".ply")),
                "has_reloc": os.path.exists(os.path.join(d, name + ".nvblx")),
                "dir": d,
                "saved_at": saved_at,
                "size_bytes": size,
            })
    out.sort(key=lambda m: m["saved_at"], reverse=True)
    return out


# ── saving the LIVE map ────────────────────────────────────────────────────────
def save_map(name: str, grid=None, mesh: bool = True) -> dict:
    """Serialize the CURRENT live map into the map dir as <name>.{pgm,yaml} (+ the
    nvblox <name>.{ply,nvblx} when available) → {name, dir, has_mesh, has_reloc,
    saved_at, size_bytes}.

    `grid` is the map source's live 2D occupancy cache — the tuple
    (w, h, res, ox, oy, oyaw, cells) that MapStream._grid_cache holds, with `cells`
    an HxW uint8 of 0..100 occupancy + 255 unknown, row 0 = the BOTTOM (y-up) row.
    The bridge passes src._grid_cache; keeping the ROS handling on that side leaves
    this module pure file IO + numpy, as the rest of it already is.

    The PGM/YAML encoding is nav2 map_saver's TRINARY form (free=254, occupied=0,
    unknown=205, rows flipped to top-first) — the SAME bytes frontier_explorer
    writes on exploration-complete, so a hand-saved map and an auto-saved one load
    back through load_grid_frame()/map_server identically.

    Raises ValueError on a bad name or with no live map to save."""
    _check_name(name)
    if grid is None:
        raise ValueError("no live map to save yet (the map source has no grid)")
    w, h, res, ox, oy, oyaw, cells = grid
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        raise ValueError("live map is empty (%dx%d)" % (w, h))

    g = np.asarray(cells).reshape(h, w)
    # nav2 trinary thresholds, matching the yaml we write below so the map reads
    # back as it was saved. free_thresh is 0.196, NOT the "nav2 default" 0.25:
    # unknown is stored as pixel 205, which decodes to occ=(255-205)/255=0.19608,
    # so any free_thresh above that reclassifies every UNKNOWN cell as FREE on the
    # way back in — the map loads with its unexplored space silently walkable.
    # 0.196 sits just under 0.19608, which is exactly why stock ROS map_saver
    # yamls carry that magic number. Verified by the roundtrip test.
    occth, freeth = 0.65, 0.196
    img = np.full((h, w), 205, np.uint8)               # unknown / in-between
    known = g <= 100                                    # 255 = unknown in the cache
    img[known & (g <= int(round(freeth * 100)))] = 254   # free     -> white
    img[known & (g >= int(round(occth * 100)))] = 0      # occupied -> black
    img = np.flipud(img)                                # grid y-up -> PGM top-first

    d = save_dir()
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, name)
    with open(base + ".pgm", "wb") as f:                # binary PGM (P5)
        f.write(("P5\n%d %d\n255\n" % (w, h)).encode("ascii"))
        f.write(img.tobytes())
    with open(base + ".yaml", "w") as f:
        f.write("image: %s\n"
                "mode: trinary\n"
                "resolution: %.6f\n"
                "origin: [%.6f, %.6f, %.6f]\n"
                "negate: 0\n"
                "occupied_thresh: %s\n"
                "free_thresh: %s\n"
                % (name + ".pgm", float(res), float(ox), float(oy),
                   float(oyaw), occth, freeth))
    log.info("saved map '%s' (%dx%d @ %.3f m/cell) -> %s.{pgm,yaml}",
             name, w, h, float(res), base)
    if mesh:
        _save_nvblox(base)
    for m in list_maps():
        if m["name"] == name:
            return m
    return {"name": name, "dir": d, "has_mesh": False, "has_reloc": False,
            "saved_at": 0.0, "size_bytes": 0}


def _save_nvblox(base: str) -> None:
    """Save the nvblox 3D reconstruction beside the 2D map: <base>.ply (mesh point
    cloud, rendered by load_mesh_chunks) + <base>.nvblx (serialized map, for
    relocalization). nvblox_msgs is not importable from this conda process, so we
    shell out through slam_env.sh exactly as frontier_explorer._save_mesh does.
    Best-effort: a missing nvblox just means the map has no mesh — never raises."""
    env = os.path.abspath(os.path.expanduser(os.getenv(
        "MABEL_NVBLOX_SAVE_ENV", "~/open_mabel/ros/scripts/slam_env.sh")))
    if not os.path.exists(env):
        log.info("3D-mesh save skipped: slam env script not found (%s)", env)
        return
    import subprocess
    for svc, ext in (("save_ply", ".ply"), ("save_map", ".nvblx")):
        target = base + ext
        cmd = ('source "%s" >/dev/null 2>&1 && '
               "ros2 service call /nvblox_node/%s nvblox_msgs/srv/FilePath "
               "\"{file_path: '%s'}\"" % (env, svc, target))
        try:
            r = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                               text=True, timeout=30.0)
            if r.returncode == 0 and "success=True" in (r.stdout or ""):
                log.info("nvblox %s -> %s", svc, target)
            else:
                log.warning("nvblox %s failed (rc=%s): %s", svc, r.returncode,
                            (r.stdout or r.stderr or "").strip()[:200])
        except Exception as e:                           # noqa: BLE001
            log.warning("nvblox %s call errored: %s", svc, e)


def delete_map(name: str) -> dict:
    """Remove the whole <name>.* set (pgm/yaml/ply/nvblx) from every map dir →
    {name, removed: [paths]}. Raises ValueError on a bad name or an unknown map."""
    _check_name(name)
    removed = []
    for d in map_dirs():
        for ext in MAP_EXTS:
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(p)
                except OSError as e:
                    log.warning("delete_map: cannot remove %s: %s", p, e)
    if not removed:
        raise ValueError("no such map: %s" % name)
    log.info("deleted map '%s' (%d files)", name, len(removed))
    return {"name": name, "removed": removed}


def _find(name: str, ext: str) -> str | None:
    base = os.path.basename(name)              # never traverse out of the map dirs
    base = os.path.splitext(base)[0]
    for d in map_dirs():
        p = os.path.join(d, base + ext)
        if os.path.exists(p):
            return p
    return None


# ── 2D occupancy (PGM + YAML) → GRID 0x04 ──────────────────────────────────────
def _read_yaml(path: str) -> dict:
    """Minimal nav2 map YAML reader (image, resolution, origin, thresholds)."""
    out: dict = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.startswith("["):
                out[k] = [float(x) for x in v.strip("[]").split(",") if x.strip()]
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v.strip().strip('"').strip("'")
    return out


def _read_pgm(path: str) -> "tuple[np.ndarray,int,int]":
    """Read a binary (P5) or ASCII (P2) PGM → (uint8 HxW, w, h). Rows top-first."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(b"P5") and not data.startswith(b"P2"):
        raise ValueError("not a PGM (P5/P2)")
    ascii_pgm = data.startswith(b"P2")
    # parse header tokens (magic, w, h, maxval), skipping # comments
    toks, i, n = [], 2, len(data)
    while len(toks) < 3 and i < n:
        while i < n and data[i:i+1].isspace():
            i += 1
        if data[i:i+1] == b"#":
            while i < n and data[i:i+1] not in (b"\n", b"\r"):
                i += 1
            continue
        j = i
        while j < n and not data[j:j+1].isspace():
            j += 1
        toks.append(data[i:j]); i = j
    w, h, maxv = int(toks[0]), int(toks[1]), int(toks[2])
    i += 1                                         # single whitespace after maxval
    if ascii_pgm:
        vals = np.array(data[i:].split()[: w * h], dtype=np.int32).astype(np.uint8)
    else:
        vals = np.frombuffer(data, dtype=np.uint8, count=w * h, offset=i)
    return vals.reshape(h, w), w, h


def load_grid_frame(name: str) -> "bytes | None":
    """Load <name>.pgm/.yaml → a GRID 0x04 frame in the `odom` frame (z-up). nav2
    stores the image rows TOP-first (y-down) with origin at the LOWER-left, so we
    flip rows to y-up. Pixel→occupancy: occ=(255-p)/255 → ≥occupied_thresh:100,
    ≤free_thresh:0, else unknown(255)."""
    y = _find(name, ".yaml")
    if not y:
        return None
    meta = _read_yaml(y)
    pgm = os.path.join(os.path.dirname(y), os.path.basename(str(meta.get("image", name + ".pgm"))))
    if not os.path.exists(pgm):
        pgm = _find(name, ".pgm")
        if not pgm:
            return None
    img, w, h = _read_pgm(pgm)
    res = float(meta.get("resolution", 0.05))
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])
    oyaw = float(origin[2]) if len(origin) > 2 else 0.0
    negate = int(meta.get("negate", 0))
    occth = float(meta.get("occupied_thresh", 0.65))
    freeth = float(meta.get("free_thresh", 0.25))
    p = img.astype(np.float32) / float(max(1, 255))
    occ = p if negate else (1.0 - p)               # nav2: occ = (255-pixel)/255
    cells = np.full((h, w), 255, np.uint8)         # unknown
    cells[occ >= occth] = 100                       # occupied
    cells[occ <= freeth] = 0                         # free
    cells = cells[::-1, :]                           # image y-down → grid y-up
    return encode_occupancy(w, h, res, ox, oy, 0.0, oyaw, cells.ravel())


# ── 3D mesh (.ply) → MESH 0x01 (point cloud) ───────────────────────────────────
def load_mesh_chunks(name: str) -> "list[bytes]":
    """Load <name>.ply vertices (x,y,z + optional r,g,b) → MESH 0x01 point-cloud
    chunks (the user picked point-cloud rendering). Returns [] if no .ply."""
    path = _find(name, ".ply")
    if not path:
        return []
    xyz, rgb = _read_ply_vertices(path)
    if xyz is None or xyz.shape[0] == 0:
        return []
    n = xyz.shape[0]
    # pack to the 16-byte MESH vertex layout, then reuse the chunker via raw bytes.
    out = np.empty((n, 16), np.uint8)
    out[:, 0:12] = np.ascontiguousarray(xyz, "<f4").view(np.uint8).reshape(n, 12)
    if rgb is not None:
        out[:, 12:15] = np.ascontiguousarray(rgb[:, :3], np.uint8)
    else:
        out[:, 12:15] = 150
    out[:, 15] = 255
    # encode_mesh_chunks expects packed-rgb f32; simpler to chunk here directly.
    import struct
    if MAP_MAX_POINTS and n > MAP_MAX_POINTS:
        stride = int(np.ceil(n / MAP_MAX_POINTS)); out = out[::stride]; n = out.shape[0]
    chunks, total = [], (n + MAP_CHUNK - 1) // MAP_CHUNK
    for i in range(total):
        seg = out[i * MAP_CHUNK:(i + 1) * MAP_CHUNK]
        final = 1 if i == total - 1 else 0
        chunks.append(struct.pack("<BBHI", 0x01, final, 0, seg.shape[0]) + seg.tobytes())
    return chunks


def _read_ply_vertices(path: str) -> "tuple[np.ndarray|None,np.ndarray|None]":
    """Parse a PLY (ascii or binary_little_endian) for vertex x,y,z (+ r,g,b)."""
    with open(path, "rb") as f:
        raw = f.read()
    nl = raw.find(b"end_header\n")
    if nl < 0 or not raw.startswith(b"ply"):
        return None, None
    header = raw[:nl].decode("ascii", "replace").splitlines()
    body = raw[nl + len(b"end_header\n"):]
    fmt = "ascii"
    vcount = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for ln in header:
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex:
                vcount = int(t[2])
        elif t[0] == "property" and in_vertex:
            props.append((t[1], t[2]))             # (type, name)  (ignore list props)
    names = [p[1] for p in props]
    if not all(a in names for a in ("x", "y", "z")) or vcount == 0:
        return None, None
    np_t = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "ushort": "<u2", "short": "<i2", "uint": "<u4", "int": "<i4"}
    if fmt.startswith("binary_little"):
        dt = np.dtype([(n, np_t.get(t, "<f4")) for t, n in props])
        v = np.frombuffer(body, dtype=dt, count=vcount)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        rgb = None
        if all(c in names for c in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
        return xyz, rgb
    # ascii
    rows = body.split(b"\n")[:vcount]
    arr = np.array([r.split() for r in rows if r.strip()], dtype=np.float64)
    idx = {n: i for i, (_, n) in enumerate(props)}
    xyz = arr[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    rgb = None
    if all(c in idx for c in ("red", "green", "blue")):
        rgb = arr[:, [idx["red"], idx["green"], idx["blue"]]].astype(np.uint8)
    return xyz, rgb
