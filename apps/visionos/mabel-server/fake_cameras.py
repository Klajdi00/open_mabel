"""
Fake MJPEG camera server for smoke-testing the Vision Pro app without
a real robot. Serves three endpoints:

    /camera/main/stream.mjpg
    /camera/wrist_left/stream.mjpg
    /camera/wrist_right/stream.mjpg

Each one emits a JPEG every ~33 ms with a moving label and timestamp
so you can visually confirm frames are flowing and roughly measure
latency.

Run:
    pip install -r requirements.txt
    python fake_cameras.py --port 8080

Then point the Vision Pro app's Config.plist at your laptop's IP.
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("mabel.fake-cam")

BOUNDARY = "mjpegboundary"
FRAME_INTERVAL = 1.0 / 30.0   # 30 fps

ENDPOINTS = {
    "/camera/main/stream.mjpg":        ("MAIN",        (1280, 720), (30, 60, 120)),
    "/camera/wrist_left/stream.mjpg":  ("LEFT WRIST",  (640, 480),  (60, 120, 60)),
    "/camera/wrist_right/stream.mjpg": ("RIGHT WRIST", (640, 480),  (120, 60, 60)),
}


def render_frame(label: str, size: tuple[int, int], bg: tuple[int, int, int], t: float) -> bytes:
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # Label
    draw.text((24, 20), label, fill=(255, 255, 255), font=font)
    # Moving dot so you can see frames are fresh
    x = int((t * 200) % (size[0] - 60)) + 30
    draw.ellipse((x - 20, size[1] // 2 - 20, x + 20, size[1] // 2 + 20), fill=(255, 255, 255))
    # Wall clock
    draw.text((24, size[1] - 44), f"{time.strftime('%H:%M:%S')}  t={t:8.2f}s",
              fill=(255, 255, 255), font=small)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # quieter default
        log.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        if self.path not in ENDPOINTS:
            self.send_error(404, "not found")
            return
        label, size, bg = ENDPOINTS[self.path]
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        t0 = time.monotonic()
        try:
            while True:
                t = time.monotonic() - t0
                frame = render_frame(label, size, bg, t)
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(FRAME_INTERVAL)
        except (ConnectionResetError, BrokenPipeError):
            pass


def local_ip() -> str:
    """Best-effort LAN IP, for the startup banner."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake MJPEG camera server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), MJPEGHandler)

    ip = local_ip()
    log.info("serving 3 fake MJPEG streams:")
    for path, (label, size, _) in ENDPOINTS.items():
        log.info("  %s  [%s, %dx%d]  → http://%s:%d%s", label, label, *size, ip, args.port, path)
    log.info("point Config.plist `network.host` at %s", ip)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
