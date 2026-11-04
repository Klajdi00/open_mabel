#!/usr/bin/env python3
"""bench.py — per-stage latency of the ZED stream pipeline (needs the camera free)."""
import statistics as st
import sys
import time

import cv2
import numpy as np

W, H = 2560, 720
FPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
Q = 85

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, float(FPS))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
afps = cap.get(cv2.CAP_PROP_FPS)
print(f"opened {aw}x{ah} @ requested {FPS} → driver reports {afps:.0f} fps")

read_ms, split_ms, enc_ms, sizes = [], [], [], []
for _ in range(120):
    t0 = time.monotonic(); ok, f = cap.read(); t1 = time.monotonic()
    if not ok:
        continue
    eye_w = f.shape[1] // 2
    t2 = time.monotonic()
    left = f[:, :eye_w]; right = f[:, eye_w:]
    t3 = time.monotonic()
    ok1, b1 = cv2.imencode(".jpg", left, [cv2.IMWRITE_JPEG_QUALITY, Q])
    ok2, b2 = cv2.imencode(".jpg", right, [cv2.IMWRITE_JPEG_QUALITY, Q])
    t4 = time.monotonic()
    read_ms.append((t1 - t0) * 1000)
    split_ms.append((t3 - t2) * 1000)
    enc_ms.append((t4 - t3) * 1000)
    sizes.append(len(b1) + len(b2))
cap.release()


def show(name, xs):
    xs = sorted(xs)
    print(f"  {name:<22} mean={st.mean(xs):6.2f}ms  p50={xs[len(xs)//2]:6.2f}  "
          f"p95={xs[int(len(xs)*0.95)]:6.2f}  max={max(xs):6.2f}")

print(f"\nper-stage over {len(read_ms)} frames (q{Q}):")
show("capture (cap.read)", read_ms)
show("split L/R", split_ms)
show("JPEG encode 2 eyes", enc_ms)
print(f"  avg encoded size/frame: {st.mean(sizes)/1024:.0f} KB (both eyes)")
budget = st.mean(read_ms) + st.mean(split_ms) + st.mean(enc_ms)
print(f"\n  server-side per-frame work ≈ {budget:.1f}ms  "
      f"(headroom at {FPS}fps = {1000/FPS:.1f}ms/frame)")
