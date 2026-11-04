"""
check_streams.py — Verify all MJPEG streams are delivering frames.

Run while main.py is running in another terminal:
  python check_streams.py

Connects to each camera path, pulls 5 frames, and reports:
  ✓ path  resolution  avg fps  frame size
  ✗ path  error message
"""
import sys
import time
import urllib.request

HOST  = "http://localhost:8080"
PATHS = [
    "/camera/main/stream.mjpg",
    "/camera/wrist_left/stream.mjpg",
    "/camera/wrist_right/stream.mjpg",
]
FRAMES_TO_GRAB = 5
TIMEOUT        = 8   # seconds per stream


def grab_frames(url: str, n: int, timeout: float) -> list[bytes]:
    """Pull n JPEG frames from a multipart MJPEG stream."""
    frames = []
    req = urllib.request.urlopen(url, timeout=timeout)

    buf = b""
    deadline = time.monotonic() + timeout

    while len(frames) < n and time.monotonic() < deadline:
        chunk = req.read(4096)
        if not chunk:
            break
        buf += chunk

        # Find complete JPEG frames by SOI (FFD8) and EOI (FFD9) markers
        while True:
            soi = buf.find(b"\xff\xd8")
            if soi == -1:
                buf = buf[-4:]   # keep tail in case marker spans chunks
                break
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi == -1:
                break
            frames.append(buf[soi : eoi + 2])
            buf = buf[eoi + 2:]

    req.close()
    return frames


def main():
    print(f"\nChecking MJPEG streams at {HOST} …\n")
    all_ok = True

    for path in PATHS:
        url = HOST + path
        t0 = time.monotonic()
        try:
            frames = grab_frames(url, FRAMES_TO_GRAB, TIMEOUT)
        except Exception as e:
            print(f"  ✗  {path}\n     {e}")
            all_ok = False
            continue

        elapsed = time.monotonic() - t0
        if not frames:
            print(f"  ✗  {path}  (connected but no JPEG frames within {TIMEOUT}s)")
            all_ok = False
            continue

        fps = (len(frames) - 1) / elapsed if elapsed > 0 else 0
        sizes = [len(f) for f in frames]
        avg_kb = sum(sizes) / len(sizes) / 1024

        # Decode first frame to get resolution
        import cv2, numpy as np
        arr = np.frombuffer(frames[0], dtype="uint8")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        res = f"{img.shape[1]}×{img.shape[0]}" if img is not None else "unknown"

        print(f"  ✓  {path}")
        print(f"     {len(frames)} frames  {res}  ~{fps:.1f} fps  avg {avg_kb:.1f} KB/frame")

    print()
    if all_ok:
        print("  All streams OK — ready for clients to connect.\n")
    else:
        print("  Some streams failed. Is main.py running?\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
