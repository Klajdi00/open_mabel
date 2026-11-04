#!/usr/bin/env python3
"""video_latency_bench.py — rigorously measure the MABEL camera networking latency,
per step, and the on-wire cost of each live stream.

Three measurements:

  1. SERVER glass-to-glass (loopback, isolates encode + queue + localhost TCP):
     a timestamp is burned into a synthetic frame, pushed through a REAL VideoServer
     and read back over a REAL HTTP MJPEG client. The delta is the latency the server
     itself adds (everything except camera capture, WiFi, and the app's decoder).

  2. LIVE per-stream cost on the running :8080 server: fps, KB/frame, Mbps. This is
     what actually crosses the WiFi — the dominant driver of app-side lag.

  3. BACKLOG under a slow client: read a stream, pause 3 s, then count how many stale
     frames were buffered downstream. With latest-wins backpressure this is ~1; a big
     number means seconds of backlogged (laggy) video.

Usage:  python3 tools/video_latency_bench.py                 # all three
        CAM_HTTP_PORT=8080 python3 tools/video_latency_bench.py
"""
import io
import os
import sys
import threading
import time
import urllib.request

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cameras"))

PORT = int(os.getenv("CAM_HTTP_PORT", "8080"))
HOST = "127.0.0.1"


def _read_one_jpeg(url, cap_bytes=400000, timeout=5):
    data = urllib.request.urlopen(url, timeout=timeout).read(cap_bytes)
    a = data.find(b"\xff\xd8"); b = data.find(b"\xff\xd9", a + 2)
    return data[a:b + 2] if (a >= 0 and b > a) else None


def measure_server_g2g(n=120):
    """(1) Loopback glass-to-glass added by the VideoServer itself."""
    import cv2
    from video_server import VideoServer
    vs = VideoServer(HOST, 8096, net_width=640, jpeg_quality=65)
    vs.register("bench"); vs.start(); time.sleep(0.6)

    samples = []
    stop = threading.Event()
    # JPEG is lossy, so we can't burn exact bytes into pixels. Instead encode a frame
    # COUNTER (0..255) as the whole-frame gray level — JPEG preserves the mean within
    # ±1, so round(mean) recovers the counter. We stamp each counter's push time and
    # the client looks it up. (256-frame wrap ≈ 4 s at 60 fps; latencies are << that.)
    times = [0.0] * 256

    def producer():
        c = 0
        while not stop.is_set():
            if vs.has_clients("bench"):
                f = np.full((480, 640, 3), c, np.uint8)
                times[c] = time.monotonic()
                vs.push("bench", f)
                c = (c + 1) % 256
            time.sleep(1 / 60)

    threading.Thread(target=producer, daemon=True).start()
    r = urllib.request.urlopen(f"http://{HOST}:8096/camera/bench/stream.mjpg", timeout=5)
    buf = b""
    while len(samples) < n:
        buf += r.read(8192)
        while True:
            a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            jpg = buf[a:b + 2]; buf = buf[b + 2:]
            im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            c = int(round(float(im.mean())))
            if not 0 <= c <= 255 or times[c] == 0.0:
                continue
            dt = (time.monotonic() - times[c]) * 1000.0
            if 0 < dt < 1000:
                samples.append(dt)            # ms
    stop.set(); r.close()
    samples.sort()
    p50 = samples[len(samples) // 2]; p95 = samples[int(len(samples) * 0.95)]
    print(f"  server glass-to-glass (encode+queue+localhost TCP): "
          f"p50 {p50:.1f} ms  p95 {p95:.1f} ms  (n={len(samples)})")


def measure_live(secs=3.0):
    """(2) Per-stream fps / KB / Mbps on the running :8080 server."""
    try:
        idx = urllib.request.urlopen(f"http://{HOST}:{PORT}/", timeout=3).read().decode()
    except Exception as e:
        print(f"  :{PORT} unreachable ({e}) — is the bridge up?"); return
    import re
    names = sorted(set(re.findall(r"/camera/([A-Za-z0-9_]+)/stream\.mjpg", idx)))
    for nm in names:
        try:
            r = urllib.request.urlopen(f"http://{HOST}:{PORT}/camera/{nm}/stream.mjpg", timeout=4)
        except Exception as e:
            print(f"  {nm:12s}: FAIL ({e})"); continue
        buf = b""; t0 = time.monotonic(); nf = 0; total = 0
        while time.monotonic() - t0 < secs:
            c = r.read(65536)
            if not c:
                break
            buf += c
            while True:
                a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                if a < 0 or b < 0:
                    break
                buf = buf[b + 2:]; nf += 1; total += (b + 2 - a)
        r.close()
        dt = time.monotonic() - t0
        if nf:
            print(f"  {nm:12s}: {nf/dt:5.1f} fps  {total/nf/1024:6.1f} KB/frame  "
                  f"{total*8/dt/1e6:6.1f} Mbps")
        else:
            print(f"  {nm:12s}: no frames")


def measure_backlog(name="wrist_left", pause=3.0):
    """(3) Stale-frame backlog after a read pause (latency under a slow client)."""
    url = f"http://{HOST}:{PORT}/camera/{name}/stream.mjpg"
    try:
        r = urllib.request.urlopen(url, timeout=4)
    except Exception as e:
        print(f"  {name}: FAIL ({e})"); return
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        r.read(65536)
    time.sleep(pause)
    r.fp.raw._sock.setblocking(False)
    got = b""
    try:
        while True:
            c = r.fp.raw._sock.recv(65536)
            if not c:
                break
            got += c
    except Exception:
        pass
    r.close()
    nf = got.count(b"\xff\xd9")
    print(f"  {name}: after {pause:.0f}s pause, {nf} stale frame(s) buffered "
          f"(~{nf/30:.2f}s backlog) — lower is better; latest-wins should keep this ~1")


if __name__ == "__main__":
    print("── 1. server-added latency (loopback, real VideoServer + HTTP client) ──")
    try:
        measure_server_g2g()
    except Exception as e:
        print(f"  skipped ({e})")
    print(f"── 2. live per-stream cost on :{PORT} (what crosses the WiFi) ──────────")
    measure_live()
    print("── 3. backlog under a slow client ─────────────────────────────────────")
    for nm in ("wrist_left", "main"):
        try:
            measure_backlog(nm)
        except Exception as e:
            print(f"  {nm}: skipped ({e})")
