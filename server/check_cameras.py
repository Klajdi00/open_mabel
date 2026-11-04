"""
check_cameras.py — Scan for connected cameras and report their status.

Run this before starting main.py to confirm which device indices work:
  python check_cameras.py

On macOS the system may ask for camera permission the first time.
"""
import sys
import cv2

MAX_PROBE = 10          # probe device indices 0 … MAX_PROBE-1
FRAMES_TO_GRAB = 3      # grab a few frames to confirm the stream is live


def probe(index: int) -> dict:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return {"index": index, "ok": False}

    # Grab a few frames — some cameras open but yield black frames initially
    frames_ok = 0
    for _ in range(FRAMES_TO_GRAB):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            frames_ok += 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()
    cap.release()

    return {
        "index": index,
        "ok": frames_ok > 0,
        "resolution": f"{w}×{h}",
        "fps": fps,
        "backend": backend,
        "live_frames": frames_ok,
    }


def main():
    print("\nScanning cameras (this takes a few seconds)…\n")
    found = []
    for i in range(MAX_PROBE):
        result = probe(i)
        status = "✓ LIVE" if result["ok"] else ("⚠ opens but no frames" if result.get("resolution") else "✗ not found")
        if result["ok"] or result.get("resolution") not in (None, "0×0"):
            print(f"  [{i}]  {status:<24}  {result.get('resolution','—')}  "
                  f"{result.get('fps', 0):.0f}fps  {result.get('backend','')}")
            if result["ok"]:
                found.append(i)

    print()
    if not found:
        print("  No working cameras found.")
        print("  Troubleshooting:")
        print("    • macOS: System Settings → Privacy & Security → Camera → allow Terminal/Python")
        print("    • Try unplugging and replugging your USB camera")
        print("    • If using a capture card, check it shows up in QuickTime → New Movie Recording")
        sys.exit(1)
    else:
        print(f"  Working camera indices: {found}")
        print(f"  Set MABEL_CAM_DEVICE=<index> to choose one, e.g.:")
        for idx in found:
            print(f"    MABEL_CAM_DEVICE={idx} python main.py")
    print()


if __name__ == "__main__":
    main()
