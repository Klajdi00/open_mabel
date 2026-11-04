"""perception_server.py — the MAP + LOCALIZATION streaming, in its OWN process.

WHY THIS EXISTS (2026-07-13, measured). The teleop gateway (sim_teleop_bridge.py)
used to host EVERYTHING in one process: the real-time joint/proprio/control planes
AND the heavy map/SLAM/point-cloud planes. Each map/cloud/slam plane runs its own
`rclpy` executor thread, and those threads plus the asyncio network thread all
serialise through ONE GIL. Profiled with py-spy: ~10 threads at 5-7% CPU each, none
pegged — textbook GIL thrashing. The upshot was that `robot_state` (:9090) and the
`:9600` proprio plane — the joints the iOS app and the web control center render —
collapsed to ~0.5-1 Hz with multi-second gaps, while VIDEO (a separate process)
stayed smooth. The apps' joint pose lagged the real robot badly.

THE SPLIT (what the operator asked for):
  * GATEWAY keeps the REAL-TIME, latency-critical planes: robot_state (:9090),
    proprio/control/status telemetry (:9600), raw LiDAR (:9093), and video (:8080/
    the camera workers, already a separate process). Set MABEL_PERCEPTION_EXTERNAL=1
    on the gateway so it does NOT create the planes below.
  * THIS PROCESS owns MAP + LOCALIZATION: the SLAM mesh + 2D grid + ODOM (:9092),
    the per-camera point clouds (:9601-9604), and the SLAM stream (:9605). Odom is
    localization, so it rides with the map here (publish_odom → MAP_HUB), exactly as
    the operator specified. Its own process => its own GIL => it can spin its ROS
    executors as hard as it likes without ever touching the joint stream's latency.

Everything here REUSES the same hub + ROS-source classes the gateway used (MAP_HUB /
start_map_stream, SlamHub / start_slam_stream, CloudHubSet / RosCloudSource) — no
logic is duplicated, only the hosting moves. Wire contracts and ports are byte-
identical, so the apps connect exactly as before and cannot tell which process
answers.

Run:  python3 perception_server.py         (real robot; needs rclpy + the ROS graph)
Env:  MABEL_MAP_PORT (9092) · MABEL_SLAM_STREAM_PORT (9605) · MABEL_CLOUD_PORT_* ·
      MABEL_MAP_STREAM (1/0) · MABEL_DATA_STREAMS (1/0) · MABEL_TOKEN (shared gate).
"""
import asyncio
import logging
import os
from urllib.parse import parse_qs, urlsplit

import websockets

import data_streams
from map_stream import MAP_HUB, start_map_stream

log = logging.getLogger("perception")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d perception: %(message)s",
                    datefmt="%H:%M:%S")

MAP_PORT = int(os.getenv("MABEL_MAP_PORT", "9092"))
_HOST = os.getenv("MABEL_WS_HOST", "0.0.0.0")


def _authorised(ws) -> bool:
    """Same token gate the gateway uses: with MABEL_TOKEN set, a client must pass
    ?key=<token>; without it every connection is allowed (LAN/Tailscale trust)."""
    token = os.getenv("MABEL_TOKEN", "").strip()
    if not token:
        return True
    req_path = getattr(getattr(ws, "request", None), "path", None) or \
        getattr(ws, "path", "/")
    supplied = (parse_qs(urlsplit(req_path).query).get("key", [""])[0]).strip()
    return supplied == token


def _peer(ws) -> str:
    try:
        a = ws.remote_address
        return f"{a[0]}:{a[1]}" if a else "?"
    except Exception:
        return "?"


async def _serve() -> None:
    if not data_streams.streams_enabled():
        log.info("MABEL_DATA_STREAMS=0 — perception planes disabled; nothing to do.")
        await asyncio.Future()
        return

    loop = asyncio.get_event_loop()
    MAP_HUB.attach_loop(loop)
    servers = []
    sources = []

    # ── MAP + ODOM (:9092) — the SLAM mesh + 2D grid + localisation pose ─────────
    async def _map_handler(ws, path=None):
        if not _authorised(ws):
            await ws.close(1008, "unauthorized")
            return
        log.info("map client connected    %s  (:%d)", _peer(ws), MAP_PORT)
        await MAP_HUB.add_subscriber(ws)
        log.info("map client disconnected %s", _peer(ws))

    servers.append(await websockets.serve(
        _map_handler, _HOST, MAP_PORT, ping_interval=None, max_size=2**20))
    log.info("map up       ws://%s:%d  (SLAM mesh + 2D grid + odom)", _HOST, MAP_PORT)
    src = start_map_stream(MAP_HUB)
    if src is not None:
        sources.append(src)
    else:
        log.warning("map stream: no ROS source (rclpy/ROS unavailable) — :%d idle.",
                    MAP_PORT)

    # ── SLAM stream (:9605) — nvblox mesh + cuVSLAM pose + /map grid hub ─────────
    try:
        from slam_stream import SlamHub, start_slam_stream
        sh = SlamHub()
        sh.attach_loop(loop)
        if start_slam_stream(sh, log) is not None:
            servers.append(await data_streams.serve_hub_ws(
                _HOST, data_streams.SLAM_STREAM_PORT, sh, "slam", log))
        else:
            log.warning("slam stream: no ROS source — :%d idle.",
                        data_streams.SLAM_STREAM_PORT)
    except Exception as e:                                          # noqa: BLE001
        log.warning("slam stream unavailable: %s", e)

    # ── Point clouds (:9601-9604) — one RosCloudSource thread per camera ─────────
    try:
        from cloud_streams import CloudHubSet, RosCloudSource
        avail = {k: True for k in data_streams.CLOUD_PORTS}
        hubset = CloudHubSet(avail, log, host=_HOST)
        for key, hub in hubset.hubs.items():
            hub.attach_loop(loop)
            servers.append(await data_streams.serve_hub_ws(
                _HOST, hubset.ports[key], hub, f"cloud:{key}", log))
        if RosCloudSource(hubset, log).start() is None:
            log.warning("ros clouds: no source — cloud ports idle until ROS is up.")
    except Exception as e:                                          # noqa: BLE001
        log.warning("cloud planes unavailable: %s", e)

    log.info("perception server up — map/localization + clouds + slam isolated "
             "from the gateway's real-time joint planes.")
    try:
        await asyncio.Future()               # run until killed
    finally:
        for s in servers:
            s.close()
        for s in sources:
            try:
                s.stop()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
