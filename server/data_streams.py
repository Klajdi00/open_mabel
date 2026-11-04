"""
data_streams.py — the 96xx data plane: ports, gating, and the shared
dedicated-port WebSocket server helper.

One place defines every dedicated data-plane port so the bridge, both camera
servers, the docs table,
cite a single source of truth. The plane is ADDITIVE: nothing here touches the
:9090 control plane, and every listener is read-only broadcast — commands stay
on the teleop WebSocket.

  ── THE ONE EXCEPTION (2026-07-16): :9610 audio is DUPLEX ──
  The speaker downlink carries device→robot bytes on a 96xx port, which the
  paragraph above forbids as written. It is allowed, deliberately and narrowly,
  because the invariant that sentence protects is NO ACTUATION OFF THE CONTROL
  PLANE — and audio has no actuation authority: it cannot move a joint, a wheel,
  or a gripper, and it never reaches controller/'s safe_base_step gate. What it
  would break instead is LATENCY: PCM is a continuous ~32 kB/s firehose, and the
  :9090 socket is the 50 Hz control loop (the same reasoning that split
  perception_server.py out of the gateway). Its CONTROL messages — enable/mute,
  format negotiation — do stay on :9090 as `audio_config`, so the rule holds
  where it means something. Do not generalise this to a second duplex plane
  without the same argument: a command that MOVES the robot has no business here.
  (Precedent, smaller: the :9600 telemetry hub already reads inbound `ping`.)

Port block (9500–9999 was verified free across the repo):

  9600  telemetry WS   — typed JSON (proprio/control @50 Hz, status @5 Hz,
                         lidar @~5 Hz, collection/network @1 Hz)
  9601  pointcloud WS  — head        (CLOUD 0x05 records, docs/data_streams.md)
  9602  pointcloud WS  — wrist_left
  9603  pointcloud WS  — wrist_right
  9604  pointcloud WS  — front/base
  9605  SLAM + map WS  — MESH 0x01 + ODOM 0x02 + GRID 0x04 (+ JSON status)
  9610  audio WS       — AUDIO 0x07 records, DUPLEX (see the exception above):
                         mic up (robot→device) + speaker down (device→robot),
                         16 kHz mono S16_LE, 20 ms frames
  9620  video MJPEG    — head stereo side-by-side (the ONLY high-res stream)
  9621  video MJPEG    — wrist_left  (standard res, tee of the :8080 frames)
  9622  video MJPEG    — wrist_right (standard res, tee)
  9623  video MJPEG    — front/base  (standard res, tee)

WHICH PROCESS BINDS WHICH PORT DIFFERS BY WORLD (clients don't care):
  sim   — 9600/9605/9610 bridge; 9601-9604 + 9620-9623 sim_camera_server
  real  — 9600-9605 + 9610 bridge (rclpy lives there); 9620-9623 camera HAL
Audio stays in the GATEWAY, not perception_server: it is a memcpy (no numpy, no
PointCloud2 deserialisation), so it does not thrash the GIL the way the map/cloud
planes did — and the speaker downlink needs the plant handle next to it.
Every plane is LAN/Tailscale-direct: clients dial the robot host's address on
the port listed in the :9600 network message's portMap.

A source that is absent in the current world (sim SLAM, Mac-`--real` depth)
still BINDS its port and answers with a JSON text "notice" frame — an
explanation, never a refused connection. Binary planes interleave notice text
frames with binary records; clients discriminate on the WS frame type.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from urllib.parse import parse_qs, urlsplit

__all__ = [
    "TELEMETRY_PORT", "CLOUD_PORTS", "SLAM_STREAM_PORT", "AUDIO_PORT",
    "AUDIO_RATE", "AUDIO_CHANNELS", "AUDIO_FRAME_MS", "DEDICATED_VIDEO_PORTS",
    "ALL_STREAM_PORTS", "streams_enabled", "port_map", "notice_frame",
    "NoticeSource", "serve_hub_ws",
]

# ---------------------------------------------------------------------------
# Ports (env-overridable, but the defaults are the contract in the docs table)
# ---------------------------------------------------------------------------

TELEMETRY_PORT = int(os.getenv("MABEL_TELEMETRY_PORT", "9600"))

CLOUD_PORTS: dict[str, int] = {
    "head":        int(os.getenv("MABEL_CLOUD_PORT_HEAD",        "9601")),
    "wrist_left":  int(os.getenv("MABEL_CLOUD_PORT_WRIST_LEFT",  "9602")),
    "wrist_right": int(os.getenv("MABEL_CLOUD_PORT_WRIST_RIGHT", "9603")),
    "front":       int(os.getenv("MABEL_CLOUD_PORT_FRONT",       "9604")),
}

SLAM_STREAM_PORT = int(os.getenv("MABEL_SLAM_STREAM_PORT", "9605"))

# Audio: ONE duplex port (see the exception in the module docstring). One socket
# and not 9610-down/9611-up because a single connection is what makes a client's
# mic and ear share a lifetime — half a voice call is worse than none.
AUDIO_PORT = int(os.getenv("MABEL_AUDIO_PORT", "9610"))

# The audio wire format. MIRRORED in hardware_bridge/config.py AUDIO (frozen by
# its contract.json), the AUDIO 0x07 record header, and the apps' AudioWorklet —
# they cannot drift independently. PCM16 (not Opus) needs no encoder on the robot
# and no decoder in the browser: Web Audio consumes it natively, and 16 kHz mono
# is 256 kbit/s on a plane that is LAN/Tailscale-direct anyway.
AUDIO_RATE = int(os.getenv("MABEL_AUDIO_RATE", "16000"))     # Hz
AUDIO_CHANNELS = int(os.getenv("MABEL_AUDIO_CHANNELS", "1"))
AUDIO_FRAME_MS = int(os.getenv("MABEL_AUDIO_FRAME_MS", "20"))  # 320 samples = 640 B

DEDICATED_VIDEO_PORTS: dict[str, int] = {
    # head is the ONLY high-res stream: one side-by-side stereo MJPEG (matches
    # the real head cam's hardware-stitched SBS frame; sim composites the eyes).
    "head_sbs":    int(os.getenv("MABEL_VIDEO_PORT_HEAD_SBS",    "9620")),
    "wrist_left":  int(os.getenv("MABEL_VIDEO_PORT_WRIST_LEFT",  "9621")),
    "wrist_right": int(os.getenv("MABEL_VIDEO_PORT_WRIST_RIGHT", "9622")),
    "front":       int(os.getenv("MABEL_VIDEO_PORT_FRONT",       "9623")),
}

ALL_STREAM_PORTS: tuple[int, ...] = (
    TELEMETRY_PORT, *CLOUD_PORTS.values(), SLAM_STREAM_PORT, AUDIO_PORT,
    *DEDICATED_VIDEO_PORTS.values(),
)


def streams_enabled() -> bool:
    """Master gate for the whole 96xx plane (MABEL_DATA_STREAMS=0 disables)."""
    return os.getenv("MABEL_DATA_STREAMS", "1").strip() not in ("0", "false", "no")


def port_map() -> dict:
    """Full endpoint map, advertised in the teleop `hello` and the :9600
    network message so clients discover the plane instead of hardcoding it."""
    return {
        # legacy plane (unchanged, for completeness)
        "teleop":   int(os.getenv("MABEL_WS_PORT", "9090")),
        "camera":   int(os.getenv("MABEL_HTTP_PORT", "8080")),
        "udpVideo": int(os.getenv("MABEL_UDP_VIDEO_PORT", "9091")),
        "map":      int(os.getenv("MABEL_MAP_PORT", "9092")),
        # 96xx data plane
        "telemetry":       TELEMETRY_PORT,
        "cloudHead":       CLOUD_PORTS["head"],
        "cloudWristLeft":  CLOUD_PORTS["wrist_left"],
        "cloudWristRight": CLOUD_PORTS["wrist_right"],
        "cloudFront":      CLOUD_PORTS["front"],
        "slam":            SLAM_STREAM_PORT,
        "audio":           AUDIO_PORT,
        "videoHeadSbs":    DEDICATED_VIDEO_PORTS["head_sbs"],
        "videoWristLeft":  DEDICATED_VIDEO_PORTS["wrist_left"],
        "videoWristRight": DEDICATED_VIDEO_PORTS["wrist_right"],
        "videoFront":      DEDICATED_VIDEO_PORTS["front"],
    }


# ---------------------------------------------------------------------------
# Notice frames — "the port is alive but this world has no source"
# ---------------------------------------------------------------------------

def notice_frame(reason: str, port: int | None = None, **extra) -> str:
    """JSON text frame interleaved on any plane when its source is absent
    (e.g. SLAM in sim, depth on Mac --real). Text vs binary is the
    discriminator on the binary planes."""
    msg = {"type": "notice", "available": False, "reason": reason, "t": time.time()}
    if port is not None:
        msg["port"] = port
    msg.update(extra)
    return json.dumps(msg)


class NoticeSource:
    """Hub stand-in for a plane whose source doesn't exist in this world.

    Speaks the same `add_subscriber(ws)` surface as MapStreamHub/TelemetryHub,
    so `serve_hub_ws` doesn't care whether a real source is behind the port.
    Sends the notice immediately on connect, then re-sends every few seconds
    as a liveness heartbeat, until the client hangs up.
    """

    HEARTBEAT_S = 5.0

    def __init__(self, reason: str, port: int | None = None) -> None:
        self._frame = notice_frame(reason, port)

    def attach_loop(self, loop) -> None:   # hub-surface compatibility
        pass

    def enabled(self) -> bool:
        return False                        # never any live data to produce

    async def add_subscriber(self, ws) -> None:
        try:
            while True:
                await ws.send(self._frame)
                await asyncio.sleep(self.HEARTBEAT_S)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dedicated-port WebSocket server (the _map_handler template, generalised)
# ---------------------------------------------------------------------------

async def serve_hub_ws(host: str, port: int, hub, name: str, log=None,
                       compression="deflate"):
    """Bind `hub` (MapStreamHub / TelemetryHub / NoticeSource) on its own port.

    Token-gated exactly like the teleop planes: when MABEL_TOKEN is set the
    client must supply ?key=<token> (defence-in-depth — these ports are
    LAN-direct, but the same rule everywhere keeps the contract simple).
    Any path is accepted; the port IS the selector. Returns the server object
    (caller closes it on shutdown) or None when websockets is unavailable.
    """
    import websockets  # deferred so importing this module never needs it

    async def _handler(ws, path=None):
        req_path = path or getattr(ws, "path", None) or getattr(
            getattr(ws, "request", None), "path", "/")
        parts = urlsplit(req_path or "/")
        token = os.getenv("MABEL_TOKEN", "").strip()
        supplied = (parse_qs(parts.query).get("key", [""])[0]).strip() if token else ""
        if token and supplied != token:
            await ws.close(1008, "unauthorized")
            return
        peer = getattr(ws, "remote_address", None)
        if log:
            log.info("%s client connected    %s  (:%d)", name, peer, port)
        try:
            await hub.add_subscriber(ws)
        finally:
            if log:
                log.info("%s client disconnected %s", name, peer)

    # Small write buffer so a stalling subscriber back-pressures immediately and
    # the hub's latest-wins queues coalesce to FRESH data (same rationale as the
    # teleop socket's write_limit — snapshot streams must never replay history).
    # compression: None for small latency-critical planes (telemetry :9600), left
    # as deflate for the big cloud/mesh planes where it actually shrinks the wire.
    server = await websockets.serve(
        _handler, host, port, ping_interval=None, max_size=2**20,
        write_limit=2**16, compression=(None if compression is None else "deflate"))
    if log:
        log.info("%s up   ws://0.0.0.0:%d", name, port)
    return server
