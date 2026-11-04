#!/usr/bin/env python3
"""
view.py — dead-simple peek at what the ZED-M is streaming.

It reads the LIVE MJPEG stream the camera server is already serving on :8080
(so it does NOT fight the server for the USB device). By default it saves a
snapshot you can open; pass --window for a live window if you have a display.

Usage:
    python3 view.py                  # save sensors/snapshot.jpg (left eye)
    python3 view.py --eye both       # save both eyes side-by-side
    python3 view.py --eye right      # right eye
    python3 view.py --window         # live window (needs a display, e.g. local desktop)
    python3 view.py --host 192.168.1.42   # view from another machine on the LAN

Run with the robot's python:
    ~/miniconda3/envs/base-module-env/bin/python view.py
"""
import argparse
import sys
import urllib.request

import cv2
import numpy as np

PATHS = {
    "left":  "/camera/head_left/stream.mjpg",
    "right": "/camera/head_right/stream.mjpg",
    "main":  "/camera/main/stream.mjpg",
}


def frames(url):
    """Yield decoded BGR frames from an MJPEG (multipart/x-mixed-replace) URL."""
    stream = urllib.request.urlopen(url, timeout=5)
    buf = b""
    while True:
        buf += stream.read(8192)
        a = buf.find(b"\xff\xd8")          # JPEG start
        b = buf.find(b"\xff\xd9", a + 2)   # JPEG end
        if a != -1 and b != -1:
            jpg = buf[a:b + 2]
            buf = buf[b + 2:]
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield img


def grab(host, port, eye):
    if eye == "both":
        left = next(frames(f"http://{host}:{port}{PATHS['left']}"))
        right = next(frames(f"http://{host}:{port}{PATHS['right']}"))
        h = min(left.shape[0], right.shape[0])
        return np.hstack([cv2.resize(left, (left.shape[1], h)),
                          cv2.resize(right, (right.shape[1], h))])
    return next(frames(f"http://{host}:{port}{PATHS[eye]}"))


def main():
    ap = argparse.ArgumentParser(description="peek at the ZED-M stream")
    ap.add_argument("--host", default="127.0.0.1", help="camera server host (default localhost)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--eye", choices=["left", "right", "main", "both"], default="left")
    ap.add_argument("--window", action="store_true", help="live window instead of a snapshot")
    ap.add_argument("--out", default="snapshot.jpg", help="snapshot file (default sensors/snapshot.jpg)")
    args = ap.parse_args()

    if args.window:
        url_eye = "left" if args.eye == "both" else args.eye
        url = f"http://{args.host}:{args.port}{PATHS[url_eye]}"
        print(f"live  {url}   (press q or Esc to quit)")
        try:
            for img in frames(url):
                cv2.imshow("ZED-M (q to quit)", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        except cv2.error:
            sys.exit("no display available — drop --window to save a snapshot instead")
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()
        return

    try:
        img = grab(args.host, args.port, args.eye)
    except Exception as e:                       # noqa: BLE001
        sys.exit(f"could not read stream from {args.host}:{args.port} — is the camera "
                 f"server running?  ({e})")
    cv2.imwrite(args.out, img)
    print(f"saved {args.out}  ({img.shape[1]}x{img.shape[0]})  — open it to see the {args.eye} view")


if __name__ == "__main__":
    main()
