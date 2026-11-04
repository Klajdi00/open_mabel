"""
main.py — Entry point for the MABEL Mac proxy server.

Starts two concurrent services:
  1. WebSocket server (port 9090) — receives TeleopFrames from the Vision Pro
  2. HTTP MJPEG server (port 8080) — streams webcam as mock robot cameras

Usage:
  cd server
  python main.py                    # default: webcam device 0, all interfaces
  python main.py --no-video         # disable camera (WS only)
  python main.py --cameras-only     # MJPEG cameras on 8080, NO WebSocket server.
                                    # Use with mujoco_pose_viz.py (which owns 9090):
                                    #   terminal 1: python main.py --cameras-only
                                    #   terminal 2: mjpython mujoco_pose_viz.py
  MABEL_VERBOSE=1 python main.py    # print full joint matrices

Environment variables (see config.py for full list):
  MABEL_WS_PORT       WebSocket port (default 9090)
  MABEL_HTTP_PORT     MJPEG HTTP port (default 8080)
  MABEL_CAM_DEVICE    OpenCV camera device index (default 0)
  MABEL_LOG_EVERY     Print pose summary every N frames (default 30)
  MABEL_VERBOSE       Set to 1 for full joint matrix dumps
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import signal
import sys

# ---- stdlib logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

import websockets

from config import config
from pose_logger import PoseLogger
from teleop_receiver import TeleopReceiver
from wire_protocol import TeleopFrame


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

pose_logger = PoseLogger(
    log_every_n=config.tracking.log_every_n_frames,
    verbose_matrices=config.tracking.verbose_matrices,
)
frame_count = 0


async def on_frame(frame: TeleopFrame) -> None:
    """Called for every TeleopFrame received from the Vision Pro."""
    global frame_count
    frame_count += 1
    pose_logger.log(frame)


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------

async def run_websocket_server(stop_event: asyncio.Event) -> None:
    receiver = TeleopReceiver(on_frame=on_frame)
    addr = (config.network.ws_host, config.network.ws_port)
    path = config.network.ws_path

    log.info("WebSocket server listening on  ws://%s:%d%s",
             config.network.ws_host, config.network.ws_port, path)

    async def _handler(ws):
        # ws.path = websockets ≤ 11 (legacy);  ws.request.path = websockets ≥ 12
        req_path = getattr(ws, "path", None) or getattr(
            getattr(ws, "request", None), "path", "/"
        )
        if req_path != path:
            log.warning("Rejected connection to unknown path: %s", req_path)
            await ws.close(1008, "wrong path")
            return
        await receiver.handler(ws)

    async with websockets.serve(_handler, *addr):
        log.info("  → WebSocket ready")
        await stop_event.wait()
        log.info("WebSocket server shutting down…")


async def run_mjpeg_server(stop_event: asyncio.Event) -> None:
    """Start per-camera captures and serve them all via one MJPEG HTTP server."""
    from aiohttp import web
    from mjpeg_server import MJPEGServer

    sources = config.video.cameras
    server = MJPEGServer(sources=sources, jpeg_quality=config.video.jpeg_quality)

    loop = asyncio.get_running_loop()
    server.start_captures(loop)

    log.info("MJPEG server listening on       http://%s:%d",
             config.network.http_host, config.network.http_port)
    for src in sources:
        log.info("  %-22s → http://%s:%d%s",
                 src.name, config.network.http_host, config.network.http_port, src.path)

    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, config.network.http_host, config.network.http_port)
    await site.start()

    await stop_event.wait()
    log.info("MJPEG server shutting down…")
    server.stop_captures()
    await runner.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(enable_video: bool = True, enable_ws: bool = True) -> None:
    stop_event = asyncio.Event()

    # Graceful shutdown on Ctrl-C / SIGTERM
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    ws_line   = f"║  WS   ws://mabel.local:{config.network.ws_port}{config.network.ws_path:<16}║" if enable_ws   else "║  WS   disabled (--cameras-only)              ║"
    http_line = f"║  HTTP http://mabel.local:{config.network.http_port}                   ║"       if enable_video else "║  HTTP disabled (--no-video)                  ║"

    print(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║        MABEL Mac Proxy Server                ║\n"
        "╠══════════════════════════════════════════════╣\n"
        f"{ws_line}\n"
        f"{http_line}\n"
        "║                                              ║\n"
        "║  Make sure 'mabel.local' resolves to this   ║\n"
        "║  Mac on your local network, or edit          ║\n"
        "║  Config.plist in Xcode to use the IP.        ║\n"
        "╚══════════════════════════════════════════════╝\n"
        "\nPress Ctrl-C to stop.\n"
    )

    tasks = []
    if enable_ws:
        tasks.append(asyncio.create_task(run_websocket_server(stop_event)))
    if enable_video:
        tasks.append(asyncio.create_task(run_mjpeg_server(stop_event)))

    if not tasks:
        log.error("Nothing to run — pass at least one of WS or video.")
        return

    await asyncio.gather(*tasks)
    log.info("All services stopped. Total frames received: %d", frame_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MABEL Mac proxy server")
    parser.add_argument(
        "--no-video", action="store_true",
        help="Disable MJPEG video server (WebSocket teleop only)"
    )
    parser.add_argument(
        "--cameras-only", action="store_true",
        help=(
            "Run MJPEG camera server only (port 8080), no WebSocket on 9090. "
            "Use alongside 'mjpython mujoco_pose_viz.py' which owns port 9090."
        )
    )
    args = parser.parse_args()

    enable_ws    = not args.cameras_only
    enable_video = not args.no_video

    try:
        asyncio.run(main(enable_video=enable_video, enable_ws=enable_ws))
    except KeyboardInterrupt:
        sys.exit(0)
