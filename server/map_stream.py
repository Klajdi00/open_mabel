"""
map_stream.py — live SLAM map producer for the `/map` WebSocket.

The iOS app's Maps tab renders the robot's live nvblox mesh + odom pose. To keep
the heavy mesh off the 20 Hz teleop loop it rides a SEPARATE WebSocket path,
`/map`, multiplexed onto the SAME server port as `/teleop` (so the app reaches
it at the teleop host:9090). This module is the producer:

    ROS  /nvblox_node/mesh_pointcloud (PointCloud2)  ─┐
         /odom                        (Odometry)     ─┴─▶ MapRosSource
                                                            │ encode binary
                                                            ▼
                                                        MapStreamHub ──▶ /map clients

Wire protocol (binary, little-endian) — see server/docs/data_streams.md and the
iOS reader MABEL Mobile App/Services/MapStreamLink.swift. Both ends MUST agree:

    MESH (0x01):  u8 type, u8 flags(bit0=final chunk), u16 pad, u32 count,
                  count × { f32 x,y,z ; u8 r,g,b,a }   (ROS `odom` frame, z-up)
    ODOM (0x02):  u8 type, u8 pad×3, f32 x,y,z, f32 qx,qy,qz,qw (ROS x,y,z,w)
    PATH (0x03):  u8 type, u8 pad×3, u32 count, count × { f32 x,y,z }
                  the Nav2 planned path (`/plan`), transformed into the SAME
                  `odom` frame as the mesh so the app draws it on the live map
                  (the "live RViz" route). count=0 clears the client's path.

This socket is SEND-only. The nav GOAL travels the OTHER way on the always-on
teleop socket as the existing `NAV_GOAL` message ({"x","y","yaw"}) — the bridge
already forwards it to Nav2 (`/goal_pose`, NavFn A* + MPPI), whose `/plan` comes
back here as PATH. So: tap → teleop NAV_GOAL → Nav2 → /plan → /map PATH → draw.

The mesh is CHUNKED: iOS `URLSessionWebSocketTask` defaults to a 1 MiB max
message, so each frame carries ≤ MAP_CHUNK points (≈0.96 MB) and the final chunk
sets flags bit0=1. The client accumulates chunks and swaps the snapshot in on the
final one.

rclpy is imported lazily inside the source thread, so importing this module never
requires ROS — the bridge can `from map_stream import MAP_HUB` unconditionally,
and `start_map_stream()` simply returns None when ROS isn't available.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import threading
import time

import numpy as np

log = logging.getLogger("mabel.map")

# Points per WebSocket frame. 60000 × 16 + 8 ≈ 0.96 MB, under iOS's 1 MiB default.
MAP_CHUNK = int(os.getenv("MABEL_MAP_CHUNK", "60000"))
# Hard cap on a snapshot's total points (subsampled if exceeded) so a runaway map
# can't flood the link. 0 disables the cap.
MAP_MAX_POINTS = int(os.getenv("MABEL_MAP_MAX_POINTS", "400000"))
# Hard cap on a snapshot's TRIANGLES (surface mesh path). A triangle soup is ~20 B
# per vertex × 3 = 60 B/triangle, so 30k tris ≈ 1.8 MB — comparable to the point
# cloud but rendered as SHADED SURFACES. Subsampled (whole triangles) if exceeded.
# 15k tris ≈ 0.9 MB/mesh. Halved from 30k (2026-07-13): the mesh encode is pure
# Python over ROS Point objects (GIL-held), so its cost scales with triangle count
# and directly trades against the real-time state stream. The room surface reads
# fine at 15k; raise only if a client needs finer geometry and the box has headroom.
MAP_MAX_TRIS = int(os.getenv("MABEL_MAP_MAX_TRIS", "15000"))
# Throttle the (full-snapshot) mesh broadcast — the surface changes slowly and
# re-sending the whole mesh at the publisher's 2 Hz is what lagged the browser.
# 0.5 Hz: the SLAM surface changes slowly and each broadcast re-ships the whole
# mesh. Halved from 1 Hz (2026-07-13) to keep the GIL free for the real-time state
# stream; late joiners still get the cached mesh immediately on connect.
MAP_MESH_HZ = float(os.getenv("MABEL_MAP_MESH_HZ", "0.5"))
# Vertices converted per GIL slice in the mesh encoder. The conversion is pure
# Python (ROS Point objects), so it holds the GIL; we hand it back every slice so
# the asyncio network thread — which ships robot_state/telemetry to the apps — is
# never stalled for more than one slice. ~3k verts ≈ 10-20 ms. See _on_mesh_marker.
_MESH_SLICE_VERTS = int(os.getenv("MABEL_MAP_MESH_SLICE", "3000"))
# Odom is small but can be fast; throttle the broadcast to this rate.
MAP_ODOM_HZ = float(os.getenv("MABEL_MAP_ODOM_HZ", "30"))
# PLANAR ground projection of the robot pose (default ON): z clamped to 0 and
# roll/pitch stripped (yaw only) before broadcast. Visual-odometry z DRIFTS
# (measured −0.38 m after ~30 min of cuVSLAM) which sank the nav-view robot model
# below the floor mesh / 2D map plane — it looked like the model "disappeared".
# A wheeled base lives on the ground plane, exactly like RViz nav views assume.
MAP_ODOM_PLANAR = os.getenv("MABEL_MAP_ODOM_PLANAR", "1").strip() in ("1", "true", "yes", "on")
# Nav2 re-plans at up to 20 Hz; the path is tiny but redundant frame-to-frame, so
# throttle the broadcast (the route doesn't move fast).
MAP_PATH_HZ = float(os.getenv("MABEL_MAP_PATH_HZ", "5"))

_MESH_TYPE = 0x01
_ODOM_TYPE = 0x02
_PATH_TYPE = 0x03
_GRID_TYPE = 0x04          # 2D occupancy map (nav_msgs/OccupancyGrid)
_CLOUD_TYPE = 0x05         # live sensor pointcloud (robot's CURRENT view)
_SCAN_TYPE = 0x06          # raw LiDAR sweep (sensor_msgs/LaserScan, compact)
_AUDIO_TYPE = 0x07         # PCM audio frame (:9610, BOTH directions)

# Live cloud overlay: the dense camera cloud is the heaviest stream, so keep it at
# a LOW rate (well below the ~30 Hz sensor) + a tight point cap — it's a "current
# view" hint over the saved map, not the map itself. 2 Hz × 40k pts ≈ 1.3 MB/s
# (was 10 Hz × 80k ≈ 8 MB/s, which lagged the browser). Tune via the envs.
MAP_CLOUD_MAX_POINTS = int(os.getenv("MABEL_MAP_CLOUD_MAX_POINTS", "40000"))
MAP_CLOUD_HZ = float(os.getenv("MABEL_MAP_CLOUD_HZ", "2"))
MAP_GRID_HZ = float(os.getenv("MABEL_MAP_GRID_HZ", "2"))
# Raw LiDAR sweeps for the teleop "radar" overlay: ~1.5 KB/frame (720 × u16 mm),
# throttled well below the sensor's ~13 Hz — a live obstacle hint, not the map.
MAP_SCAN_HZ = float(os.getenv("MABEL_MAP_SCAN_HZ", "5"))


# ───────────────────────────────────────────────────────────────────────────
# Fan-out hub  (mirrors monitor_hub.MONITOR: a singleton the bridge multiplexes
# onto its WebSocket; broadcasts are scheduled onto the server's asyncio loop).
# ───────────────────────────────────────────────────────────────────────────
class MapStreamHub:
    # Per-socket outbound queue depth. A snapshot is ≤ ~7 chunks; odom is 30 Hz —
    # 256 frames is generous headroom before back-pressure drops the oldest.
    _QUEUE_MAX = 256

    def __init__(self) -> None:
        # ws -> asyncio.Queue. Each subscriber drains its OWN queue through a
        # single writer coroutine, so sends to one socket never overlap (the
        # `websockets` library forbids concurrent ws.send()).
        self._subs: dict = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        # Latest mesh as the list of chunk frames, replayed to a new subscriber so
        # it sees the map immediately (the nvblox publisher is latched, but a fresh
        # /map client connects to US, not to ROS).
        self._mesh_chunks: list[bytes] = []
        # Latest Nav2 path frame, likewise replayed so a client opening the Maps tab
        # mid-navigation immediately sees the active route (None = no active plan).
        self._path_frame: bytes | None = None
        # Latest 2D occupancy grid (0x04) + latest live cloud chunks (0x05), both
        # replayed to a new subscriber so it sees the map + the current view at once.
        self._grid_frame: bytes | None = None
        self._cloud_chunks: list[bytes] = []
        # Latest robot pose (0x02), replayed on join — without it a client had NO
        # robot model until the next odom message happened to arrive.
        self._odom_frame: bytes | None = None
        # Latest raw LiDAR sweep (0x06), replayed on join (instant radar view).
        self._scan_frame: bytes | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def enabled(self) -> bool:
        return bool(self._subs)

    # ── subscriber side (runs on the server's asyncio loop) ──────────────────
    async def add_subscriber(self, ws) -> None:
        """Register a /map client, replay the latest mesh, then hold the socket
        open until it closes. Clients are receive-only; any inbound frame is
        ignored (room for a future control channel)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAX)
        for fr in list(self._mesh_chunks):              # prime with the latest map
            self._offer(q, fr)
        if self._grid_frame is not None:                # …the 2D occupancy map
            self._offer(q, self._grid_frame)
        if self._odom_frame is not None:                # …the robot pose
            self._offer(q, self._odom_frame)
        for fr in list(self._cloud_chunks):             # …the latest live cloud
            self._offer(q, fr)
        if self._scan_frame is not None:                # …the latest LiDAR sweep
            self._offer(q, self._scan_frame)
        if self._path_frame is not None:                # …and the active route, if any
            self._offer(q, self._path_frame)
        self._subs[ws] = q
        writer = asyncio.create_task(self._writer(ws, q))
        try:
            async for _ in ws:           # keep the handler alive until close
                pass
        except Exception:
            pass
        finally:
            self._subs.pop(ws, None)
            writer.cancel()

    async def _writer(self, ws, q: asyncio.Queue) -> None:
        """Single sender per socket — serialises all frames to this client."""
        try:
            while True:
                frame = await q.get()
                await ws.send(frame)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ── producer side (called from the ROS thread) ──────────────────────────
    def publish_mesh(self, chunks: list[bytes]) -> None:
        self._mesh_chunks = chunks       # cache for late joiners (even with no subs)
        self._enqueue_all(chunks)

    def publish_odom(self, frame: bytes) -> None:
        self._odom_frame = frame          # cache so a new subscriber has a robot pose
        self._enqueue_all((frame,))       # IMMEDIATELY (not only at the next message)

    def publish_path(self, frame: bytes) -> None:
        self._path_frame = frame          # cache for late joiners (like the mesh)
        self._enqueue_all((frame,))

    def publish_grid(self, frame: bytes) -> None:
        self._grid_frame = frame          # cache the 2D map for late joiners
        self._enqueue_all((frame,))

    def publish_cloud(self, chunks: list[bytes]) -> None:
        self._cloud_chunks = chunks       # cache the latest live cloud snapshot
        self._enqueue_all(chunks)

    def publish_scan(self, frame: bytes) -> None:
        self._scan_frame = frame          # cache the latest sweep for late joiners
        self._enqueue_all((frame,))

    def publish_text(self, text: str) -> None:
        """Interleave a JSON TEXT frame on this (binary) stream — status/notice
        messages for the 96xx planes. Not cached: purely transient. Clients
        discriminate on the WebSocket frame type (text vs binary)."""
        self._enqueue_all((text,))

    def _enqueue_all(self, frames) -> None:
        loop = self.loop
        if loop is None or not self._subs:
            return
        # Hop to the loop thread once; enqueue every frame to every subscriber
        # there, so queue mutation is single-threaded (asyncio.Queue isn't
        # thread-safe) and chunk order is preserved per socket.
        loop.call_soon_threadsafe(self._enqueue_on_loop, list(frames))

    def _enqueue_on_loop(self, frames) -> None:
        for q in list(self._subs.values()):
            for fr in frames:
                self._offer(q, fr)

    @staticmethod
    def _offer(q: asyncio.Queue, frame: bytes) -> None:
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:        # slow client → drop oldest, keep latency bounded
            try:
                q.get_nowait()
                q.put_nowait(frame)
            except Exception:
                pass


# The singleton the bridge imports, exactly like monitor_hub.MONITOR.
MAP_HUB = MapStreamHub()

# Latest LiDAR sweep in compact JSON form (set by MapRosSource._on_scan at
# MAP_SCAN_HZ). The teleop bridge attaches it to `robot_state` so the lidar
# streams ALONGSIDE the joints on the always-open teleop socket — the app's
# drive view gets the radar without opening the /map socket. None until the
# first sweep arrives.
_latest_scan_json: "dict | None" = None


def latest_scan() -> "dict | None":
    return _latest_scan_json


# Latest per-joint motor health (torque/load + temperature), merged from every
# subsystem's <ns>/motor_health JSON topic (arms, body, both hands — published by
# the hardware bridge at ~2 Hz). The teleop bridge attaches it to `robot_state`
# as `motorHealth` so the apps get torque + temperature telemetry alongside the
# joints. {"t": <epoch>, "joints": {name: {...}}}; None until the first message.
_latest_health: "dict | None" = None
_health_parts: dict = {}


def latest_motor_health() -> "dict | None":
    return _latest_health

# DEDICATED LiDAR hub: fans ONLY 0x06 scan frames out to the raw-LiDAR WebSocket
# (:9093, sim_teleop_bridge LIDAR_PORT) so a minimal teleop device can render the
# live sweep without speaking (or paying for) the full map protocol. The same
# scan frames ALSO go to MAP_HUB above, so existing /map clients (iOS, Vision
# Pro, Control Center) receive them on their existing connection.
SCAN_HUB = MapStreamHub()


# ───────────────────────────────────────────────────────────────────────────
# Binary encoders
# ───────────────────────────────────────────────────────────────────────────
def encode_mesh_chunks(cloud_data: bytes, *, chunk: int = MAP_CHUNK,
                       max_points: int = MAP_MAX_POINTS) -> list[bytes]:
    """Encode a `/nvblox_node/mesh_pointcloud` PointCloud2 `.data` buffer
    (point_step=16: x,y,z f32 @0/4/8 + packed rgb f32 @12) into the wire mesh
    chunks. xyz bytes are copied verbatim (already the ROS `odom` frame the
    client expects); only the packed rgb is unpacked into r,g,b,a bytes."""
    n = len(cloud_data) // 16
    if n <= 0:
        return []
    arr = np.frombuffer(cloud_data, dtype=np.uint8)[: n * 16].reshape(n, 16)

    if max_points and n > max_points:    # uniform subsample to the cap
        stride = int(np.ceil(n / max_points))
        arr = arr[::stride]
        n = arr.shape[0]

    out = np.empty((n, 16), dtype=np.uint8)
    out[:, 0:12] = arr[:, 0:12]                                   # xyz f32, verbatim
    rgb = arr[:, 12:16].copy().view(np.uint32).ravel()            # packed (r<<16|g<<8|b)
    out[:, 12] = (rgb >> 16) & 0xFF
    out[:, 13] = (rgb >> 8) & 0xFF
    out[:, 14] = rgb & 0xFF
    out[:, 15] = 255

    chunks: list[bytes] = []
    total = (n + chunk - 1) // chunk
    for i in range(total):
        seg = out[i * chunk:(i + 1) * chunk]
        final = 1 if i == total - 1 else 0
        header = struct.pack("<BBHI", _MESH_TYPE, final, 0, seg.shape[0])
        chunks.append(header + seg.tobytes())
    return chunks


def encode_mesh_tris(soup_xyz: "np.ndarray", soup_rgb: "np.ndarray", *,
                     chunk_tris: int = MAP_CHUNK // 3,
                     max_tris: int = MAP_MAX_POINTS // 3) -> list[bytes]:
    """Encode a TRIANGLE-SOUP mesh (the real nvblox surface) into wire chunks WITH
    the index buffer the iOS reader renders as shaded faces (flags bit1 = mesh).

    `soup_xyz` is (3·T, 3) f32 in the ROS `odom` frame, `soup_rgb` is (3·T, 3) u8 —
    three consecutive rows per triangle (the same expansion nvblox's TRIANGLE_LIST
    marker uses). Each chunk is SELF-CONTAINED: its indices are local (0..3K-1), so
    the client's per-chunk rebasing stitches chunks back together. flags carries
    bit1 (mesh) and bit0 (final chunk)."""
    n = int(soup_xyz.shape[0]); n -= n % 3            # whole triangles only
    tcount = n // 3
    if tcount == 0:
        return []
    if max_tris and tcount > max_tris:                # subsample WHOLE triangles
        stride = int(np.ceil(tcount / max_tris))
        keep = np.repeat(np.arange(tcount) % stride == 0, 3)[:n]
        soup_xyz, soup_rgb = soup_xyz[:n][keep], soup_rgb[:n][keep]
        n = soup_xyz.shape[0] - soup_xyz.shape[0] % 3
        tcount = n // 3

    xyz_b = np.ascontiguousarray(soup_xyz[:n], dtype="<f4").view(np.uint8).reshape(n, 12)
    out = np.empty((n, 16), dtype=np.uint8)
    out[:, 0:12] = xyz_b
    out[:, 12:15] = np.ascontiguousarray(soup_rgb[:n, :3], dtype=np.uint8)
    out[:, 15] = 255

    chunks: list[bytes] = []
    total = (tcount + chunk_tris - 1) // chunk_tris
    for i in range(total):
        seg = out[i * chunk_tris * 3:(i + 1) * chunk_tris * 3]
        seg_n = seg.shape[0]
        flags = 0x02 | (1 if i == total - 1 else 0)   # bit1 = mesh, bit0 = final
        header = struct.pack("<BBHI", _MESH_TYPE, flags, 0, seg_n)
        idx = np.arange(seg_n, dtype="<u4").tobytes()  # local 0..seg_n-1 (triangle soup)
        chunks.append(header + seg.tobytes() + struct.pack("<I", seg_n) + idx)
    return chunks


def encode_odom(px: float, py: float, pz: float,
                qx: float, qy: float, qz: float, qw: float) -> bytes:
    return struct.pack("<B3x7f", _ODOM_TYPE, px, py, pz, qx, qy, qz, qw)


def encode_path(pts: "np.ndarray") -> bytes:
    """Encode a planned path as wire frame 0x03. `pts` is (N,3) f32 in the ROS
    `odom` frame (the same frame as the mesh + odom, z-up). An empty array
    encodes count=0 — the client clears its route. Tiny (12 B/point), so a path
    of a few hundred waypoints is a single sub-KB frame (no chunking needed)."""
    arr = np.ascontiguousarray(np.reshape(pts, (-1, 3)), dtype="<f4")
    return struct.pack("<B3xI", _PATH_TYPE, arr.shape[0]) + arr.tobytes()


def encode_occupancy(width: int, height: int, resolution: float,
                     ox: float, oy: float, oz: float, oyaw: float,
                     cells: "np.ndarray") -> bytes:
    """Encode a 2D occupancy grid as wire frame 0x04 (the RViz 2D map). `cells` is
    a row-major u8 array of width*height: 0=free … 100=occupied, 255=unknown. The
    origin (ox,oy,oz,oyaw) is the grid's bottom-left corner pose in the `odom`
    frame (same frame as the mesh). The client lays it as a flat plane at z=oz."""
    header = struct.pack("<BBHIIfffff", _GRID_TYPE, 0, 0,
                         int(width), int(height), float(resolution),
                         float(ox), float(oy), float(oz), float(oyaw))
    return header + np.ascontiguousarray(cells, dtype=np.uint8).tobytes()


def encode_scan(angle_min: float, angle_inc: float,
                range_min: float, range_max: float,
                mount_x: float, mount_y: float, mount_yaw: float,
                ranges_mm: "np.ndarray") -> bytes:
    """SCAN 0x06 — one raw LiDAR sweep, compact enough for phones/teleop pads.

    Layout (little-endian):
        u8  type = 0x06        u8 version = 0     u16 reserved = 0
        u32 n                                     number of rays
        f32 angle_min          f32 angle_increment    (rad, REP-103 CCW, ray i =
                                                       angle_min + i*increment in
                                                       the LIDAR frame)
        f32 range_min          f32 range_max          (m, validity gates)
        f32 mount_x  f32 mount_y  f32 mount_yaw       (lidar pose in base_link —
                                                       compose with the 0x02 odom
                                                       pose to place rays in the
                                                       world)
        n × u16 range_mm       0 = no return / out of range
    """
    header = struct.pack("<BBHIfffffff", _SCAN_TYPE, 0, 0, int(len(ranges_mm)),
                         angle_min, angle_inc, range_min, range_max,
                         mount_x, mount_y, mount_yaw)
    return header + ranges_mm.astype("<u2").tobytes()


# Codec ids for the AUDIO 0x07 header. PCM16 is the only one implemented; the
# field exists so Opus can land later WITHOUT a wire break (a client that only
# knows PCM16 must reject an unknown codec rather than play noise).
AUDIO_CODEC_PCM16 = 0
AUDIO_CODEC_OPUS = 1                       # reserved — not implemented


def encode_audio(pcm: bytes, *, seq: int, rate: int, channels: int,
                 codec: int = AUDIO_CODEC_PCM16) -> bytes:
    """AUDIO 0x07 — one frame of voice, in EITHER direction (:9610 is duplex).

    Layout (little-endian):
        u8  type = 0x07        u8 codec (0=PCM16, 1=Opus/reserved)
        u16 channels
        u32 seq                frame counter; the ONLY gap detector — audio is a
                               SEQUENCE, so a missing frame is an audible click,
                               not a stale render the client can ignore
        u32 rate               Hz  (carried per-frame, not assumed: the mic and
                               the browser can legitimately differ, and a client
                               that guesses plays chipmunks)
        u32 n_bytes            payload length
        ... n_bytes of payload (PCM16: interleaved S16_LE)

    16 bytes of header on a 640-byte frame is 2.5% — cheap insurance against the
    two failure modes that are silent and awful: a rate mismatch, and an
    undetectable drop.
    """
    if channels <= 0:
        raise ValueError(f"channels must be >= 1, got {channels}")
    return struct.pack("<BBHIII", _AUDIO_TYPE, int(codec), int(channels),
                       int(seq) & 0xFFFFFFFF, int(rate), len(pcm)) + pcm


def decode_audio(frame: "bytes | None"):
    """Inverse of encode_audio → (pcm, seq, rate, channels, codec), or None when
    the frame is not a well-formed 0x07 record (wrong tag, truncated header, or a
    length that disagrees with the payload — all of which mean "do not play this").
    """
    if not frame or len(frame) < 16 or frame[0] != _AUDIO_TYPE:
        return None
    _t, codec, channels, seq, rate, n = struct.unpack("<BBHIII", frame[:16])
    pcm = frame[16:16 + n]
    if len(pcm) != n:
        return None                        # truncated: never play a partial frame
    return pcm, int(seq), int(rate), int(channels), int(codec)


def decode_grid_cells(frame: "bytes | None"):
    """Inverse of encode_occupancy for reachability tests: a GRID 0x04 frame →
    (w, h, res, ox, oy, oyaw, cells2d) where cells2d is (h,w) u8 row-major
    (row 0 = grid y=0, the bottom). None if the frame is not a 0x04 grid."""
    if not frame or len(frame) < 32 or frame[0] != _GRID_TYPE:
        return None
    _t, _f, _p, w, h, res, ox, oy, _oz, oyaw = struct.unpack("<BBHIIfffff", frame[:32])
    cells = np.frombuffer(frame, dtype=np.uint8, count=w * h, offset=32)
    return int(w), int(h), float(res), float(ox), float(oy), float(oyaw), cells.reshape(h, w)


def encode_cloud_chunks(xyz: "np.ndarray", rgb: "np.ndarray", *,
                        chunk: int = MAP_CHUNK,
                        max_points: int = MAP_CLOUD_MAX_POINTS) -> list[bytes]:
    """Encode a LIVE sensor pointcloud (the robot's CURRENT view) as wire frame
    0x05 — same 16-byte vertex layout as MESH (x,y,z f32 + r,g,b,a u8), distinct
    type so the client renders it as a transient overlay it REPLACES each frame.
    `xyz` is (N,3) f32 already in the `odom` frame; `rgb` is (N,3) u8."""
    n = int(xyz.shape[0])
    if n <= 0:
        return [struct.pack("<BBHI", _CLOUD_TYPE, 1, 0, 0)]   # empty → clear overlay
    if max_points and n > max_points:                          # uniform subsample
        stride = int(np.ceil(n / max_points))
        xyz, rgb = xyz[::stride], rgb[::stride]
        n = xyz.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    out[:, 0:12] = np.ascontiguousarray(xyz, dtype="<f4").view(np.uint8).reshape(n, 12)
    out[:, 12:15] = np.ascontiguousarray(rgb[:, :3], dtype=np.uint8)
    out[:, 15] = 255
    chunks: list[bytes] = []
    total = (n + chunk - 1) // chunk
    for i in range(total):
        seg = out[i * chunk:(i + 1) * chunk]
        final = 1 if i == total - 1 else 0
        header = struct.pack("<BBHI", _CLOUD_TYPE, final, 0, seg.shape[0])
        chunks.append(header + seg.tobytes())
    return chunks


def _extract_xyzrgb(msg) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Extract (xyz Nx3 f32, rgb Nx3 u8) from a sensor_msgs/PointCloud2 by field
    offsets (robust to point_step 16 packed-rgb OR xyz-only). NaN/inf points are
    dropped. rgb defaults to light grey when the cloud has no colour field."""
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    if n <= 0 or step < 12:
        return None, None
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n = min(n, buf.size // step)
    if n <= 0:
        return None, None
    buf = buf[: n * step].reshape(n, step)
    off = {f.name: int(f.offset) for f in msg.fields}
    xo, yo, zo = off.get("x", 0), off.get("y", 4), off.get("z", 8)
    xyz = np.empty((n, 3), np.float32)
    xyz[:, 0] = buf[:, xo:xo + 4].copy().view("<f4").ravel()
    xyz[:, 1] = buf[:, yo:yo + 4].copy().view("<f4").ravel()
    xyz[:, 2] = buf[:, zo:zo + 4].copy().view("<f4").ravel()
    if "rgb" in off or "rgba" in off:
        ro = off.get("rgb", off.get("rgba"))
        packed = buf[:, ro:ro + 4].copy().view("<u4").ravel()
        rgb = np.empty((n, 3), np.uint8)
        rgb[:, 0] = (packed >> 16) & 0xFF
        rgb[:, 1] = (packed >> 8) & 0xFF
        rgb[:, 2] = packed & 0xFF
    else:
        rgb = np.full((n, 3), 200, np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def _quat_to_mat(x: float, y: float, z: float, w: float) -> "np.ndarray":
    """3×3 rotation matrix for a (x,y,z,w) quaternion (identity if degenerate)."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float32)


class CloudFramer:
    """Express a sensor cloud (camera OPTICAL frame, +z = FORWARD) in the odom
    frame — the frame every CLOUD 0x05 plane promises (docs/data_streams.md).

    Shipping an untransformed optical-frame cloud is NEVER acceptable: the
    clients map ROS→viewer as (x, z, -y), so the camera's forward axis (+z)
    lands on the viewer's UP axis and the cloud stands straight up out of the
    robot. Dropping a frame is strictly better than mis-orienting it.

    Fallback chain (each step weaker than the last):
      1. odom←sensor via TF        — exact; needs the DYNAMIC odom→base_link TF,
                                     absent while SLAM is down or initialising
                                     (thor with cuvslam:=false publishes no
                                     `odom` frame at all).
      2. base_link←sensor via TF   — the STATIC URDF chain robot_state_publisher
                                     always publishes, composed with the last
                                     pose from the odom TOPIC.
      3. None                      — the caller DROPS the frame.

    Owners feed it a tf2 `Buffer` and the latest odom pose; it is otherwise
    stateless and thread-confined to its owner's ROS callback thread.
    """

    def __init__(self, target_frame: str = "odom", label: str = "cloud") -> None:
        self.target_frame = target_frame
        self.label = label
        self.tf_buffer = None            # tf2_ros.Buffer, set by the owner
        self.last_odom_pose = None       # (t 3-vec, R 3x3) from the odom TOPIC
        self._state = None               # last logged path (log once per CHANGE)

    def set_odom_pose(self, px, py, pz, qx, qy, qz, qw) -> None:
        self.last_odom_pose = (np.array([px, py, pz], np.float32),
                               _quat_to_mat(qx, qy, qz, qw))

    def to_target(self, pts: "np.ndarray", src_frame: str):
        """odom-frame points, or None when no usable transform exists yet."""
        if not src_frame or src_frame == self.target_frame:
            return pts
        if self.tf_buffer is None:
            self._log("no tf2 buffer — cloud dropped, not mis-oriented")
            return None
        from rclpy.time import Time

        def _apply(tf, p):
            t, q = tf.transform.translation, tf.transform.rotation
            R = _quat_to_mat(q.x, q.y, q.z, q.w)
            return (p @ R.T) + np.array([t.x, t.y, t.z], np.float32)

        try:
            out = _apply(self.tf_buffer.lookup_transform(
                self.target_frame, src_frame, Time()), pts)
            self._log(f"{self.target_frame} TF")
            return out
        except Exception:
            pass
        try:
            out = _apply(self.tf_buffer.lookup_transform(
                "base_link", src_frame, Time()), pts)
            if self.last_odom_pose is not None:
                t, R = self.last_odom_pose
                out = (out @ R.T) + t
            self._log("base_link TF + last odom (odom TF not published yet)")
            return out
        except Exception as e:
            self._log(f"NO TF ({e}) — cloud dropped, not mis-oriented")
            return None

    def _log(self, state: str) -> None:
        """Log the transform path once per CHANGE (the odom TF appearing and
        vanishing is normal during SLAM init; per-frame logging would flood)."""
        if state != self._state:
            self._state = state
            log.info("%s: cloud → %s via %s", self.label, self.target_frame, state)


# ───────────────────────────────────────────────────────────────────────────
# ROS source — subscribes to the mesh + odom and feeds the hub.
# ───────────────────────────────────────────────────────────────────────────
class MapRosSource:
    def __init__(self, hub: MapStreamHub, mesh_topic: str, odom_topic: str,
                 plan_topic: str = "/plan", mesh_frame: str = "odom",
                 grid_topic: str = "/map",
                 cloud_topic: str = "/camera/front/depth/points",
                 scan_topic: str = "/scan") -> None:
        self.hub = hub
        self.mesh_topic = mesh_topic
        self.odom_topic = odom_topic
        self.plan_topic = plan_topic
        # Raw LiDAR sweep -> SCAN 0x06 (both MAP_HUB and the dedicated SCAN_HUB).
        self.scan_topic = scan_topic
        self._last_scan_t = 0.0
        self._scan_mount = None          # (x, y, yaw) of the lidar in base_link
        # 2D occupancy map (RViz "Map") + the live sensor cloud (robot's current
        # view). Empty string disables either.
        self.grid_topic = grid_topic
        self.cloud_topic = cloud_topic
        self._last_grid_t = 0.0
        self._last_cloud_t = 0.0
        # When a SAVED room is loaded, suppress the LIVE mesh (0x01) + grid (0x04)
        # so the saved map stays put; the live sensor cloud (0x05) + odom + path
        # keep flowing and overlay on top. None = live mode. (Set by the bridge's
        # LOAD_MAP handler via load_saved_map / resume_live below.)
        self._loaded_name: str | None = None
        # Latest 2D occupancy as (w,h,res,ox,oy,oyaw,cells2d), in the mesh/odom
        # frame, for goal_reachable() — set by _on_grid (live) + load_saved_map.
        self._grid_cache = None
        # The frame the mesh + odom are published in; the planned path (usually in
        # the global `map` frame) is transformed into THIS frame so it overlays the
        # mesh exactly on the client.
        self.mesh_frame = mesh_frame
        self._thread: threading.Thread | None = None
        self._cloud_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._owns_rclpy = False
        self._last_odom = 0.0
        self._last_path_t = 0.0
        self._tf_buffer = None
        self._tf_warned = False
        # Last odom pose (t 3-vec, R 3x3) from the odom TOPIC — the cloud fallback
        # placement when the dynamic odom TF is not (yet) published.
        self._last_odom_pose = None
        # optical→odom for the live sensor cloud; shared with the per-camera
        # :9601-9604 planes (cloud_streams.RosCloudSource) so both obey the same
        # "never ship an optical-frame cloud" rule.
        self._framer = CloudFramer(mesh_frame, "map stream")
        # Heartbeat state: last good ENCODED pose + when it was received, so the
        # 10 Hz timer can keep the client fed through source stalls.
        self._odom_last_frame: bytes | None = None
        self._odom_rx_t = 0.0
        self._mesh_n = 0
        self._mesh_log_t = 0.0
        self._last_mesh_t = 0.0
        # block (x,y,z) -> (xyz Nx3 f32, rgb Nx3 u8, tris (M*3,) i32 | None), the
        # accumulated nvblox mesh (incremental blocks), like mesh_to_pointcloud.
        self._blocks: dict = {}

    def start(self) -> "MapRosSource | None":
        try:
            import rclpy  # noqa: F401  (probe ROS availability before spawning)
        except Exception as e:
            log.warning("map stream: rclpy unavailable (%s) — /map will be idle. "
                        "Source a ROS env (the workspace / slam_env.sh) to stream "
                        "the live map.", e)
            return None
        self._thread = threading.Thread(target=self._run, name="mabel-map-ros", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cloud_thread is not None:
            self._cloud_thread.join(timeout=2.0)

    # runs on its own thread with a private executor
    def _run(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from nav_msgs.msg import Odometry, Path

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        node = rclpy.create_node("mabel_map_stream")

        # TF so the Nav2 plan (global `map` frame) can be expressed in the mesh's
        # `odom` frame. Optional: if tf2_ros is missing or no map→odom is published
        # (sim with no SLAM drift), _to_mesh_frame falls back to identity.
        try:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            TransformListener(self._tf_buffer, node, spin_thread=False)
        except Exception as e:
            log.info("map stream: tf2 unavailable (%s) — plan sent in its own frame "
                     "(fine when map≈odom).", e)
            self._tf_buffer = None

        # Mesh: match the latched RELIABLE/TRANSIENT_LOCAL publisher so a just-
        # connected source gets the current map immediately.
        mesh_qos = QoSProfile(depth=1)
        mesh_qos.reliability = QoSReliabilityPolicy.RELIABLE
        mesh_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        # Odom: MATCH the publisher's reliability. A BEST_EFFORT subscription on
        # this (busy) socket silently lost ~⅔ of the poses under UDP-buffer
        # pressure — RELIABLE retransmits recover them, and cuVSLAM / swerve
        # odometry both publish RELIABLE. BEST_EFFORT remains the fallback so a
        # best-effort publisher (some sims) still matches.
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        odom_qos.durability = QoSDurabilityPolicy.VOLATILE
        try:
            pubs = node.get_publishers_info_by_topic(self.odom_topic)
            if pubs and all(p.qos_profile.reliability == QoSReliabilityPolicy.RELIABLE
                            for p in pubs):
                odom_qos.reliability = QoSReliabilityPolicy.RELIABLE
                log.info("map stream: odom publisher is RELIABLE — matching (no "
                         "silent BEST_EFFORT drops).")
        except Exception:
            pass

        # Mesh source. DEFAULT = POINT CLOUD (/nvblox_node/mesh_pointcloud): a
        # triangle soup is ~6× the bytes of the vertex cloud (measured: 66k pts =
        # 1.06 MB vs 99k tris = 5.95 MB), so for a fast browser the cloud wins —
        # we just RATE-LIMIT it (MAP_MESH_HZ, below the publisher's 2 Hz) and cap
        # the point count. Opt into the shaded TRIANGLE surface mesh (heavier) with
        # MABEL_MAP_MESH_MODE=mesh — then we use the standard mesh_marker
        # (visualization_msgs/Marker, no nvblox_msgs needed), subsampled + throttled.
        mode = os.getenv("MABEL_MAP_MESH_MODE", "points").lower()
        if mode in ("mesh", "tris", "triangles", "surface"):
            from visualization_msgs.msg import Marker
            marker_topic = os.getenv("MABEL_MAP_MESH_MARKER", "/nvblox_node/mesh_marker")
            node.create_subscription(Marker, marker_topic, self._on_mesh_marker, mesh_qos)
            log.info("map stream: subscribed  mesh=%s (Marker TRIANGLE_LIST → surfaces, "
                     "≤%d tris @ %.1f Hz)  odom=%s", marker_topic, MAP_MAX_TRIS,
                     MAP_MESH_HZ, self.odom_topic)
        else:
            node.create_subscription(PointCloud2, self.mesh_topic, self._on_mesh, mesh_qos)
            log.info("map stream: subscribed  mesh=%s (PointCloud2 → cloud, ≤%d pts "
                     "@ %.1f Hz)  odom=%s", self.mesh_topic, MAP_MAX_POINTS,
                     MAP_MESH_HZ, self.odom_topic)
        node.create_subscription(Odometry, self.odom_topic, self._on_odom, odom_qos)
        # Keep the client's robot pose fed through source stalls / tracking loss.
        node.create_timer(0.1, self._odom_heartbeat)

        # Nav2 planned path (/plan, nav_msgs/Path). RELIABLE/VOLATILE matches the
        # planner_server publisher; each plan REPLACES the client's route.
        plan_qos = QoSProfile(depth=1)
        plan_qos.reliability = QoSReliabilityPolicy.RELIABLE
        plan_qos.durability = QoSDurabilityPolicy.VOLATILE
        node.create_subscription(Path, self.plan_topic, self._on_plan, plan_qos)
        log.info("map stream: subscribed  plan=%s → PATH (frame %s)",
                 self.plan_topic, self.mesh_frame)

        # 2D occupancy map (RViz "Map") → GRID 0x04. Latched, like the mesh.
        if self.grid_topic:
            from nav_msgs.msg import OccupancyGrid
            grid_qos = QoSProfile(depth=1)
            grid_qos.reliability = QoSReliabilityPolicy.RELIABLE
            grid_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            node.create_subscription(OccupancyGrid, self.grid_topic, self._on_grid, grid_qos)
            log.info("map stream: subscribed  grid=%s → GRID 0x04 @ %.1f Hz",
                     self.grid_topic, MAP_GRID_HZ)

        # Raw LiDAR sweep → SCAN 0x06 (both hubs: /map clients get it inline; the
        # dedicated LiDAR WebSocket serves it alone). ~1.5 KB/frame, throttled.
        if self.scan_topic:
            from sensor_msgs.msg import LaserScan
            from rclpy.qos import qos_profile_sensor_data
            node.create_subscription(LaserScan, self.scan_topic, self._on_scan,
                                     qos_profile_sensor_data)
            log.info("map stream: subscribed  scan=%s → SCAN 0x06 @ %.1f Hz "
                     "(+ dedicated LiDAR ws)", self.scan_topic, MAP_SCAN_HZ)

        # Motor health (torque/load + temps) from every bridge subsystem → merged
        # into robot_state.motorHealth (see latest_motor_health above). Tiny JSON
        # Strings at ~2 Hz each.
        from std_msgs.msg import String as _Str
        for topic in ("/arms/motor_health", "/body/motor_health",
                      "/left_hand/motor_health", "/right_hand/motor_health"):
            node.create_subscription(
                _Str, topic,
                (lambda t: lambda m: self._on_health(t, m))(topic), 5)
        log.info("map stream: subscribed  */motor_health → robot_state.motorHealth")

        # Live sensor pointcloud (robot's CURRENT view) → CLOUD 0x05. Runs on its
        # OWN DDS participant (own thread + rclpy Context = own UDP socket): the
        # 30 Hz × ~0.5 MB depth firehose previously landed on THIS node's socket and
        # its kernel buffer (1 MB) overflowed, dropping the mesh/grid fragments —
        # the RELIABLE ~0.5 MB mesh samples then livelocked in retransmission (first
        # mesh delivery measured 12 min after launch). Isolating the firehose keeps
        # this participant's socket quiet for the mesh/grid/odom/tf traffic.
        if self.cloud_topic:
            self._cloud_thread = threading.Thread(
                target=self._run_cloud, name="mabel-map-cloud", daemon=True)
            self._cloud_thread.start()

        ex = SingleThreadedExecutor()
        ex.add_node(node)
        try:
            while not self._stop.is_set() and rclpy.ok():
                ex.spin_once(timeout_sec=0.2)
        except Exception as e:
            log.warning("map stream: ROS spin stopped: %s", e)
        finally:
            ex.remove_node(node)
            node.destroy_node()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

    def _run_cloud(self) -> None:
        """Own the live-sensor-cloud subscription on a PRIVATE rclpy Context (own DDS
        participant + UDP socket) so the depth firehose can't starve the main node's
        socket (see the comment at the spawn site). Demand-driven: the subscription
        exists only while a /map client is connected — with no viewer the 15 MB/s
        stream is not even delivered to this process."""
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        from sensor_msgs.msg import PointCloud2

        ctx = Context()
        rclpy.init(args=None, context=ctx)
        node = rclpy.create_node("mabel_map_stream_cloud", context=ctx)
        cloud_qos = QoSProfile(depth=1)
        cloud_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        cloud_qos.durability = QoSDurabilityPolicy.VOLATILE
        log.info("map stream: cloud source on its own DDS participant  cloud=%s → "
                 "CLOUD 0x05 (≤%d pts @ %.1f Hz, subscribed only while a client is "
                 "connected)", self.cloud_topic, MAP_CLOUD_MAX_POINTS, MAP_CLOUD_HZ)
        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        sub = None
        try:
            while not self._stop.is_set() and ctx.ok():
                want = self.hub.enabled()
                if want and sub is None:
                    sub = node.create_subscription(
                        PointCloud2, self.cloud_topic, self._on_cloud, cloud_qos)
                elif not want and sub is not None:
                    node.destroy_subscription(sub)
                    sub = None
                ex.spin_once(timeout_sec=0.2)
        except Exception as e:
            log.warning("map stream: cloud source stopped: %s", e)
        finally:
            ex.remove_node(node)
            node.destroy_node()
            try:
                rclpy.shutdown(context=ctx)
            except Exception:
                pass

    def load_saved_map(self, name: str, grid_frame, mesh_chunks) -> None:
        """Switch to a SAVED room: cache + push its grid + mesh to clients and STOP
        streaming the live mesh/grid (the live cloud 0x05 + odom keep overlaying)."""
        self._loaded_name = name
        if mesh_chunks:
            self.hub.publish_mesh(mesh_chunks)
        if grid_frame:
            self.hub.publish_grid(grid_frame)
            self._grid_cache = decode_grid_cells(grid_frame)   # for goal_reachable
        log.info("map stream: LOADED saved room '%s' (live mesh/grid paused, "
                 "live cloud + odom overlay)", name)

    def resume_live(self) -> None:
        if self._loaded_name is not None:
            log.info("map stream: resuming LIVE map (was '%s')", self._loaded_name)
        self._loaded_name = None
        self._grid_cache = None        # live grid will repopulate it via _on_grid

    def goal_reachable(self, x: float, y: float) -> "bool | None":
        """Is world point (x,y) (mesh/odom frame) a navigable goal on the current
        2D map? True/False, or None if there's no map yet (can't decide → allow).
        Rejects goals OUTSIDE the grid bounds or on an OCCUPIED/UNKNOWN cell — the
        two cases the user wants flagged ("out of the map or un-reachable")."""
        g = self._grid_cache
        if not g:
            return None
        w, h, res, ox, oy, oyaw, cells = g
        dx, dy = x - ox, y - oy
        if oyaw:                                  # un-rotate into grid axes
            c, s = np.cos(-oyaw), np.sin(-oyaw)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        col = int(dx / res); row = int(dy / res)
        if col < 0 or row < 0 or col >= w or row >= h:
            return False                          # outside the mapped area
        # tolerate a 1-cell halo of unknown so goals at the frontier still take.
        cell = int(cells[row, col])
        if cell >= 90:                            # occupied
            return False
        if cell == 255:                           # unknown → check small neighborhood
            r0, r1 = max(0, row - 1), min(h, row + 2)
            c0, c1 = max(0, col - 1), min(w, col + 2)
            if not np.any(cells[r0:r1, c0:c1] == 0):
                return False                      # no free cell nearby → unreachable
        return True

    def _on_mesh(self, msg) -> None:
        if self._loaded_name is not None:           # a saved room is shown — keep it
            return
        # RATE-LIMIT: the publisher emits at ~2 Hz but the map changes slowly and
        # re-sending the whole cloud is what lagged the browser — throttle to
        # MAP_MESH_HZ (default 1 Hz). The live cloud overlays whatever map is loaded.
        now = time.monotonic()
        if MAP_MESH_HZ > 0 and (now - self._last_mesh_t) < (1.0 / MAP_MESH_HZ):
            return
        self._last_mesh_t = now
        data = msg.data if isinstance(msg.data, (bytes, bytearray)) else bytes(msg.data)
        t0 = time.perf_counter()
        try:
            chunks = encode_mesh_chunks(data)
        except Exception as e:
            log.warning("map stream: mesh encode failed: %s", e)
            return
        if chunks:
            self.hub.publish_mesh(chunks)
        # Latency budget the SERVER adds on top of the SLAM pipeline: encode + the
        # synchronous enqueue to every subscriber. Logged every ~2 s so "real-time"
        # is verifiable (transport on top is sub-ms on LAN / localhost).
        self._mesh_n += 1
        enc_ms = (time.perf_counter() - t0) * 1e3
        now = time.monotonic()
        if now - self._mesh_log_t >= 2.0:
            pts = len(data) // 16
            log.info("map mesh: %d pts, %d chunk(s), encode+enqueue %.1f ms, %d msgs",
                     pts, len(chunks), enc_ms, self._mesh_n)
            self._mesh_log_t = now

    def _on_nvblox_mesh(self, msg) -> None:
        # Accumulate nvblox's INCREMENTAL mesh blocks (each carries only changed
        # blocks + a `clear` flag), exactly like mesh_to_pointcloud, then expand the
        # block-local triangles into a coloured triangle soup and stream it.
        t0 = time.perf_counter()
        if getattr(msg, "clear", False):
            self._blocks.clear()
        for idx, blk in zip(msg.block_indices, msg.blocks):
            key = (idx.x, idx.y, idx.z)
            if not blk.vertices:                       # emptied block → deallocated
                self._blocks.pop(key, None)
                continue
            xyz = np.array([(p.x, p.y, p.z) for p in blk.vertices], np.float32)
            if blk.colors:
                rgb = np.array([(int(c.r * 255), int(c.g * 255), int(c.b * 255))
                                for c in blk.colors], np.uint8)
            else:
                rgb = np.tile(np.uint8([135, 135, 135]), (xyz.shape[0], 1))
            tris = np.asarray(blk.triangles, np.int32) if blk.triangles else None
            self._blocks[key] = (xyz, rgb, tris)

        # Expand each block's triangles (block-local indices) into a flat soup.
        sx, sc = [], []
        for xyz, rgb, tris in self._blocks.values():
            if tris is None or tris.size < 3:
                continue
            t = tris[: tris.size - tris.size % 3]
            sx.append(xyz[t]); sc.append(rgb[t])
        if not sx:
            return
        soup_xyz = np.concatenate(sx)
        soup_rgb = np.concatenate(sc)
        try:
            chunks = encode_mesh_tris(soup_xyz, soup_rgb)
        except Exception as e:
            log.warning("map stream: mesh encode failed: %s", e)
            return
        if chunks:
            self.hub.publish_mesh(chunks)
        self._mesh_n += 1
        now = time.monotonic()
        if now - self._mesh_log_t >= 2.0:
            log.info("map mesh: %d tris, %d chunk(s), encode+enqueue %.1f ms, %d msgs",
                     soup_xyz.shape[0] // 3, len(chunks),
                     (time.perf_counter() - t0) * 1e3, self._mesh_n)
            self._mesh_log_t = now

    def _on_mesh_marker(self, msg) -> None:
        # A visualization_msgs/Marker TRIANGLE_LIST (mesh_to_pointcloud re-emits the
        # nvblox mesh here): msg.points are 3 verts per triangle, msg.colors are
        # per-vertex. Stream as a shaded SURFACE mesh (encode_mesh_tris, flags bit1).
        # Rate-limit + subsample WHOLE triangles at extraction so a large map stays
        # small/fast on the browser (a full soup of ~100k tris would be ~6 MB).
        if self._loaded_name is not None:           # a saved room is shown — keep it
            return
        # NOBODY WATCHING => DO NOT ENCODE. This callback is the most expensive thing
        # in the gateway (see the GIL note below), and it used to run flat out even
        # with zero map subscribers. nvblox republishes the marker continuously, so a
        # client that connects later still gets a mesh within ~1 s.
        if not self.hub.enabled():
            return
        now = time.monotonic()
        if MAP_MESH_HZ > 0 and (now - self._last_mesh_t) < (1.0 / MAP_MESH_HZ):
            return
        self._last_mesh_t = now
        pts = msg.points
        n = len(pts) - (len(pts) % 3)
        if n == 0:
            return
        tcount = n // 3
        stride = int(np.ceil(tcount / MAP_MAX_TRIS)) if (MAP_MAX_TRIS and tcount > MAP_MAX_TRIS) else 1
        cols = msg.colors
        have_cols = len(cols) >= n
        t0 = time.perf_counter()
        # ── GIL DISCIPLINE (2026-07-13) ──────────────────────────────────────────
        # This runs on the map ROS thread, but the work below is PURE PYTHON
        # (attribute access on ~200k geometry_msgs/Point objects — they expose no
        # buffer protocol, so it cannot be vectorised away). It therefore HOLDS THE
        # GIL for 240-500 ms at a stretch, which starved the asyncio network thread
        # in this same process: `robot_state` on :9090 (the pose the iOS app and the
        # web control center render) collapsed from 100 Hz to 0.6 Hz with 2.6 s gaps,
        # and :9600 telemetry with it. The joints looked frozen while VIDEO stayed
        # smooth — video lives in another process, so it never paid this tax.
        # So convert in SLICES and hand the GIL back between them: time.sleep(0)
        # releases it, letting the network thread run. Same total CPU, but the
        # longest stall is now one slice (~10 ms) instead of the whole mesh.
        vidx = [b + k for ti in range(0, tcount, stride) for b in (ti * 3,) for k in (0, 1, 2)]
        nv = len(vidx)
        sx = np.empty((nv, 3), np.float32)
        sc = (np.empty((nv, 3), np.float32) if have_cols
              else np.full((nv, 3), 160, np.uint8))
        for s in range(0, nv, _MESH_SLICE_VERTS):
            e = min(s + _MESH_SLICE_VERTS, nv)
            sl = vidx[s:e]
            sx[s:e] = np.fromiter((c for i in sl for c in (pts[i].x, pts[i].y, pts[i].z)),
                                  dtype=np.float32, count=(e - s) * 3).reshape(-1, 3)
            if have_cols:
                sc[s:e] = np.fromiter(
                    (v for i in sl for v in (cols[i].r, cols[i].g, cols[i].b)),
                    dtype=np.float32, count=(e - s) * 3).reshape(-1, 3)
            time.sleep(0)                  # <- releases the GIL; the loop breathes
        if have_cols:
            sc = (np.clip(sc, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            chunks = encode_mesh_tris(sx, sc, max_tris=MAP_MAX_TRIS)
        except Exception as e:
            log.warning("map stream: marker mesh encode failed: %s", e)
            return
        if chunks:
            self.hub.publish_mesh(chunks)
        self._mesh_n += 1
        if now - self._mesh_log_t >= 2.0:
            log.info("map mesh: %d/%d tris (stride %d), %d chunk(s), %.1f ms, %d msgs",
                     len(vidx) // 3, tcount, stride, len(chunks),
                     (time.perf_counter() - t0) * 1e3, self._mesh_n)
            self._mesh_log_t = now

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vals = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        # Sanitize: a NaN/inf pose (SLAM re-init glitch) would put the client's
        # robot node at an invalid transform — SceneKit stops rendering it.
        if not all(math.isfinite(v) for v in vals):
            return
        qn = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if not (0.5 < qn < 2.0):
            return
        # Cache the pose BEFORE any gating: the cloud fallback placement (odom TF
        # not yet published) needs the latest odom even with no client connected.
        self._last_odom_pose = (np.array([p.x, p.y, p.z], np.float32),
                                _quat_to_mat(q.x / qn, q.y / qn, q.z / qn, q.w / qn))
        if MAP_ODOM_PLANAR:
            # Ground-plane projection: z=0 + yaw-only (see MAP_ODOM_PLANAR above).
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            frame = encode_odom(p.x, p.y, 0.0,
                                0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        else:
            frame = encode_odom(p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        self._odom_rx_t = time.monotonic()
        self._odom_last_frame = frame     # heartbeat re-sends this during stalls
        if not self.hub.enabled():
            return
        now = self._odom_rx_t
        if MAP_ODOM_HZ > 0 and (now - self._last_odom) < (1.0 / MAP_ODOM_HZ):
            return
        self._last_odom = now
        self.hub.publish_odom(frame)

    def _odom_heartbeat(self) -> None:
        """10 Hz timer: re-broadcast the last good pose while the SOURCE is silent
        (cuVSLAM stalls ~0.6 s during nvblox mesh updates and stops entirely on
        tracking loss). The client's robot model then never starves — worst case it
        holds its last pose instead of freezing/vanishing between real updates."""
        if self._odom_last_frame is None or not self.hub.enabled():
            return
        if time.monotonic() - self._odom_rx_t > 0.3:
            self.hub.publish_odom(self._odom_last_frame)

    def _on_plan(self, msg) -> None:
        # Nav2's planned route. Always cache (cheap) so a client opening the Maps
        # tab mid-nav sees it; throttle the broadcast since /plan can hit 20 Hz.
        now = time.monotonic()
        if MAP_PATH_HZ > 0 and (now - self._last_path_t) < (1.0 / MAP_PATH_HZ):
            return
        self._last_path_t = now
        n = len(msg.poses)
        if n == 0:                       # goal reached / canceled → clear the route
            self.hub.publish_path(encode_path(np.empty((0, 3), np.float32)))
            return
        pts = np.empty((n, 3), np.float32)
        for i, ps in enumerate(msg.poses):
            pos = ps.pose.position
            pts[i] = (pos.x, pos.y, pos.z)
        pts = self._to_mesh_frame(pts, msg.header.frame_id or self.mesh_frame)
        self.hub.publish_path(encode_path(pts))

    def _on_grid(self, msg) -> None:
        # nav_msgs/OccupancyGrid (the RViz 2D map). Throttle + transform the grid
        # origin (usually `map` frame) into the mesh `odom` frame so it overlays.
        if self._loaded_name is not None:           # a saved room's 2D map is shown
            return
        if not self.hub.enabled():
            return
        now = time.monotonic()
        if MAP_GRID_HZ > 0 and (now - self._last_grid_t) < (1.0 / MAP_GRID_HZ):
            return
        self._last_grid_t = now
        info = msg.info
        w, h = int(info.width), int(info.height)
        if w * h == 0:
            return
        data = np.asarray(msg.data, dtype=np.int16)
        cells = np.where(data < 0, 255, np.clip(data, 0, 100)).astype(np.uint8)
        op = info.origin.position
        oq = info.origin.orientation
        src = msg.header.frame_id or self.mesh_frame
        # origin + its +x axis tip, both transformed to odom → position + yaw.
        ux = _quat_to_mat(oq.x, oq.y, oq.z, oq.w) @ np.array([1, 0, 0], np.float32)
        pts = self._to_mesh_frame(
            np.array([[op.x, op.y, op.z], [op.x + ux[0], op.y + ux[1], op.z + ux[2]]],
                     np.float32), src)
        oyaw = float(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0]))
        try:
            frame = encode_occupancy(w, h, info.resolution,
                                     pts[0, 0], pts[0, 1], pts[0, 2], oyaw, cells)
        except Exception as e:
            log.warning("map stream: grid encode failed: %s", e)
            return
        self.hub.publish_grid(frame)
        self._grid_cache = (w, h, float(info.resolution),
                            float(pts[0, 0]), float(pts[0, 1]), oyaw,
                            cells.reshape(h, w))   # for goal_reachable

    def _on_scan(self, msg) -> None:
        """LaserScan → SCAN 0x06 to MAP_HUB (inline plane) + SCAN_HUB (:9093),
        plus the compact JSON form `latest_scan()` that the teleop bridge attaches
        to `robot_state` (so the lidar rides ALONGSIDE the joints on the always-
        open teleop socket). Runs even with zero hub subscribers — the teleop
        channel has its own clients and 5 Hz encode cost is negligible."""
        now = time.time()
        if MAP_SCAN_HZ > 0 and (now - self._last_scan_t) < (1.0 / MAP_SCAN_HZ):
            return
        self._last_scan_t = now
        # Lidar mount in base_link (static URDF chain), looked up once so clients
        # can compose sweep rays with the 0x02 robot pose.
        if self._scan_mount is None and self._tf_buffer is not None:
            try:
                import rclpy.time as _rt
                tr = self._tf_buffer.lookup_transform(
                    "base_link", msg.header.frame_id, _rt.Time())
                q = tr.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                self._scan_mount = (tr.transform.translation.x,
                                    tr.transform.translation.y, yaw)
            except Exception:
                pass                      # TF not up yet; retry next sweep
        mx, my, myaw = self._scan_mount or (0.0, 0.0, 0.0)
        r = np.asarray(msg.ranges, dtype=np.float32)
        mm = np.where(np.isfinite(r) & (r > 0.0), r * 1000.0, 0.0)
        mm = np.clip(mm, 0, 65535).astype(np.uint16)
        frame = encode_scan(float(msg.angle_min), float(msg.angle_increment),
                            float(msg.range_min), float(msg.range_max),
                            float(mx), float(my), float(myaw), mm)
        self.hub.publish_scan(frame)
        SCAN_HUB.publish_scan(frame)
        # Compact JSON twin for the teleop robot_state: 1° / centimetre precision
        # (min-pooled pairs so a thin obstacle never decimates away), ~1.8 KB.
        half = (len(mm) // 2) * 2
        pooled = mm[:half].reshape(-1, 2)
        nz = pooled.copy()
        nz[nz == 0] = 65535                      # 0 = no return: never wins the min
        cm = (np.min(nz, axis=1) // 10).astype(int)
        cm[np.max(pooled, axis=1) == 0] = 0      # both rays empty -> 0 (no return)
        global _latest_scan_json
        _latest_scan_json = {
            "t": now,
            "angleMin": float(msg.angle_min),
            "angleInc": float(msg.angle_increment) * 2.0,
            "rangeMaxM": float(msg.range_max),
            "mount": [round(float(mx), 4), round(float(my), 4),
                      round(float(myaw), 4)],
            "rangesCm": cm.tolist(),             # 0 = no return / invalid
        }

    def _on_health(self, topic, msg) -> None:
        """Merge one subsystem's motor-health JSON into the global snapshot."""
        global _latest_health
        try:
            import json as _json
            _health_parts[topic] = _json.loads(msg.data)
        except Exception:
            return
        merged: dict = {}
        for part in _health_parts.values():
            merged.update(part)
        _latest_health = {"t": time.time(), "joints": merged}

    def _on_cloud(self, msg) -> None:
        # Live sensor pointcloud (robot's CURRENT view, e.g. /camera/front/depth/
        # points). Throttle, extract xyz+rgb, transform optical→odom, downsample,
        # stream as CLOUD 0x05 (the client replaces its overlay each frame).
        if not self.hub.enabled():
            return
        now = time.monotonic()
        if MAP_CLOUD_HZ > 0 and (now - self._last_cloud_t) < (1.0 / MAP_CLOUD_HZ):
            return
        self._last_cloud_t = now
        xyz, rgb = _extract_xyzrgb(msg)
        if xyz is None or xyz.shape[0] == 0:
            return
        xyz = self._cloud_to_mesh_frame(xyz, msg.header.frame_id or self.mesh_frame)
        if xyz is None:                  # no usable transform yet — never ship a
            return                       # camera-optical-frame cloud ("pointing up")
        try:
            chunks = encode_cloud_chunks(xyz, rgb)
        except Exception as e:
            log.warning("map stream: cloud encode failed: %s", e)
            return
        if chunks:
            self.hub.publish_cloud(chunks)

    def _cloud_to_mesh_frame(self, pts: "np.ndarray", src_frame: str):
        """Express the live sensor cloud (camera OPTICAL frame, z-forward) in the
        mesh/odom frame — matching what RViz shows via TF. Unlike ``_to_mesh_frame``
        (map→odom for plan/grid, where identity is a fine fallback because map≈odom),
        identity is NEVER acceptable here (see CloudFramer): an untransformed
        optical-frame cloud renders rotated ("pointing up") on the client, so a
        frame with no usable transform is DROPPED."""
        self._framer.tf_buffer = self._tf_buffer
        self._framer.last_odom_pose = self._last_odom_pose
        return self._framer.to_target(pts, src_frame)

    def _to_mesh_frame(self, pts: "np.ndarray", src_frame: str) -> "np.ndarray":
        """Express map-frame path points in the mesh (`odom`) frame via the latest
        map→odom TF. Identity fallback (with a one-shot warning) when the frames
        already match, tf2 is absent, or no transform is published yet."""
        if not src_frame or src_frame == self.mesh_frame or self._tf_buffer is None:
            return pts
        try:
            from rclpy.time import Time
            tf = self._tf_buffer.lookup_transform(self.mesh_frame, src_frame, Time())
            t, q = tf.transform.translation, tf.transform.rotation
            R = _quat_to_mat(q.x, q.y, q.z, q.w)
            return (pts @ R.T) + np.array([t.x, t.y, t.z], np.float32)
        except Exception as e:
            if not self._tf_warned:
                log.warning("map stream: no %s←%s TF (%s) — drawing the plan in its "
                            "own frame (ok when map≈odom).", self.mesh_frame, src_frame, e)
                self._tf_warned = True
            return pts


def start_map_stream(hub: MapStreamHub = MAP_HUB) -> "MapRosSource | None":
    """Start the ROS→/map source if enabled and ROS is importable; else None.

    Controlled by env:
      MABEL_MAP_STREAM      "1"/"0"  (default on)
      MABEL_MAP_MESH_TOPIC  default /nvblox_node/mesh_pointcloud
      MABEL_MAP_ODOM_TOPIC  default /odom
      MABEL_MAP_PLAN_TOPIC  default /plan          (Nav2 planned path → PATH)
      MABEL_MAP_FRAME       default odom           (frame the plan is drawn in)
    """
    if os.getenv("MABEL_MAP_STREAM", "1").strip() not in ("1", "true", "yes", "on"):
        log.info("map stream: disabled (MABEL_MAP_STREAM=0)")
        return None
    mesh_topic = os.getenv("MABEL_MAP_MESH_TOPIC", "/nvblox_node/mesh_pointcloud")
    odom_topic = os.getenv("MABEL_MAP_ODOM_TOPIC", "/odom")
    plan_topic = os.getenv("MABEL_MAP_PLAN_TOPIC", "/plan")
    mesh_frame = os.getenv("MABEL_MAP_FRAME", "odom")
    # 2D occupancy map + live sensor cloud + raw LiDAR sweep (empty disables each).
    grid_topic = os.getenv("MABEL_MAP_GRID_TOPIC", "/map")
    cloud_topic = os.getenv("MABEL_MAP_CLOUD_TOPIC", "/camera/front/depth/points")
    scan_topic = os.getenv("MABEL_MAP_SCAN_TOPIC", "/scan")
    return MapRosSource(hub, mesh_topic, odom_topic, plan_topic, mesh_frame,
                        grid_topic, cloud_topic, scan_topic).start()
