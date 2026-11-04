"""
Fake MJPEG camera server with Local Preview.
Serves three endpoints and displays a live composite window on the Mac.

Run:
    pip install opencv-python numpy Pillow
    python fake_camera.py --port 8080
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("mabel.fake-cam")

BOUNDARY = "mjpegboundary"
FRAME_INTERVAL = 1.0 / 30.0   # 30 fps

ENDPOINTS = {
    "/camera/main/stream.mjpg":        ("MAIN",        (1280, 720), (30, 60, 120)),
    "/camera/wrist_left/stream.mjpg":  ("LEFT WRIST",  (640, 480),  (60, 120, 60)),
    "/camera/wrist_right/stream.mjpg": ("RIGHT WRIST", (640, 480),  (120, 60, 60)),
}

# Shared state between the Preview Thread and the HTTP Server Thread
latest_jpegs: dict[str, bytes] = {}
frame_lock = threading.Lock()


def render_pil_image(label: str, size: tuple[int, int], bg: tuple[int, int, int], t: float) -> Image.Image:
    """Generates the base PIL Image with text and a moving dot."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        # Tries to load standard Mac fonts first
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.text((24, 20), label, fill=(255, 255, 255), font=font)
    
    # Moving dot so you can see frames are fresh
    x = int((t * 200) % (size[0] - 60)) + 30
    draw.ellipse((x - 20, size[1] // 2 - 20, x + 20, size[1] // 2 + 20), fill=(255, 255, 255))
    
    # Wall clock
    draw.text((24, size[1] - 44), f"{time.strftime('%H:%M:%S')}  t={t:8.2f}s",
              fill=(255, 255, 255), font=small)
    return img


def pil_to_jpeg(img: Image.Image) -> bytes:
    """Converts a PIL image to JPEG bytes for the HTTP stream."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass  # Keep the terminal clean

    def do_GET(self) -> None:
        if self.path not in ENDPOINTS:
            self.send_error(404, "not found")
            return
        
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    frame_data = latest_jpegs.get(self.path)
                
                if frame_data:
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame_data)}\r\n\r\n".encode())
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                
                time.sleep(FRAME_INTERVAL)
        except (ConnectionResetError, BrokenPipeError):
            pass


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake MJPEG camera server with Local Preview")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 1. Start the HTTP server in a background thread
    server = ThreadingHTTPServer((args.host, args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    ip = local_ip()
    log.info("serving 3 fake MJPEG streams:")
    for path, (label, size, _) in ENDPOINTS.items():
        log.info(f"  {label:<12} [{size[0]}x{size[1]}]  → http://{ip}:{args.port}{path}")
    log.info(f"point Config.plist `network.host` at {ip}")
    log.info("Press 'q' in the Preview Window or Ctrl+C in terminal to quit.")

    # 2. Run the Generation & Preview Loop on the Main Thread 
    # (OpenCV strictly requires running on the macOS Main Thread)
    t0 = time.monotonic()
    
    try:
        while True:
            t = time.monotonic() - t0
            
            # Generate the 3 PIL images
            img_main = render_pil_image(*ENDPOINTS["/camera/main/stream.mjpg"], t)
            img_lw = render_pil_image(*ENDPOINTS["/camera/wrist_left/stream.mjpg"], t)
            img_rw = render_pil_image(*ENDPOINTS["/camera/wrist_right/stream.mjpg"], t)
            
            # Save the JPEG bytes for the HTTP Server
            with frame_lock:
                latest_jpegs["/camera/main/stream.mjpg"] = pil_to_jpeg(img_main)
                latest_jpegs["/camera/wrist_left/stream.mjpg"] = pil_to_jpeg(img_lw)
                latest_jpegs["/camera/wrist_right/stream.mjpg"] = pil_to_jpeg(img_rw)

            # Convert to OpenCV format (BGR) for the local preview window
            # We scale them down so the window actually fits on your Macbook screen
            cv_main = cv2.cvtColor(np.array(img_main.resize((640, 360))), cv2.COLOR_RGB2BGR)
            cv_lw = cv2.cvtColor(np.array(img_lw.resize((320, 240))), cv2.COLOR_RGB2BGR)
            cv_rw = cv2.cvtColor(np.array(img_rw.resize((320, 240))), cv2.COLOR_RGB2BGR)
            
            # Composite them: Main on top, Wrists side-by-side on the bottom
            bottom_row = np.hstack((cv_lw, cv_rw))
            composite = np.vstack((cv_main, bottom_row))
            
            # Display the window
            cv2.imshow("MABEL - Fake Camera Preview", composite)
            
            # Wait ~33ms (30fps). If 'q' is pressed, break the loop.
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down servers...")
        cv2.destroyAllWindows()
        server.shutdown()

if __name__ == "__main__":
    main()