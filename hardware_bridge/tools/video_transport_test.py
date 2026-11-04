#!/usr/bin/env python3
"""video_transport_test.py — rigorously validate the bridge's video transports with
SYNTHETIC frames (no cameras needed). Stands up a real VideoServer + UdpVideoSink on
test ports, pushes a moving pattern, then for head + wrist:

  • H.264 / WebSocket (browser WebCodecs path) — receive access units, ffprobe the
    profile (MUST be 4:2:0 baseline, or HW decoders reject it), and full-decode them.
  • H.264 / UDP (iOS / Vision Pro path) — subscribe over the 'MV' protocol, reassemble,
    decode.
  • MJPEG / HTTP (fallback) — pull one frame and decode.

Run on Thor (needs ffmpeg with h264_nvenc + aiohttp). Exit 0 iff every path decodes.
"""
import asyncio
import os
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # hardware_bridge root (paths.py)
sys.path.insert(0, os.path.join(HERE, "..", "cameras"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))


def _grad(i, w, h):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    g = np.roll(np.tile(x, (h, 1)), i * 6, 1)
    return np.dstack([g, np.roll(g, 40, 1), np.roll(g, 80, 1)]).astype(np.uint8)


def _ffprobe(h264):
    return subprocess.run(
        ["ffprobe", "-loglevel", "error", "-show_entries",
         "stream=profile,pix_fmt,width,height", "-of", "default=nw=1", "pipe:0"],
        input=h264, capture_output=True).stdout.decode().replace("\n", " ").strip()


def _decode_count(h264, w, h):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "h264", "-i", "pipe:0",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                         input=h264, capture_output=True).stdout
    return len(out) // (w * h * 3)


def main() -> int:
    from udp_sink import UdpVideoSink
    from video_server import VideoServer
    import cv2

    sink = UdpVideoSink(fps=30, jpeg_port=9591, h264_port=9592, h264_bitrate="4M")
    vs = VideoServer("127.0.0.1", 8599, net_width=640, jpeg_quality=65, udp=sink,
                     h264_width=1280)
    vs.register("main", "wrist_left")
    vs.start()
    time.sleep(0.8)

    stop = threading.Event()

    def produce():
        i = 0
        while not stop.is_set():
            vs.push("main", _grad(i, 1280, 720))
            vs.push("wrist_left", _grad(i, 640, 480))
            i += 1
            time.sleep(1 / 30)
    threading.Thread(target=produce, daemon=True).start()

    ok = True

    async def ws_grab(name):
        import aiohttp
        aus, t0 = [], time.monotonic()
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"http://127.0.0.1:8599/camera/{name}/h264") as ws:
                while time.monotonic() - t0 < 6 and len(aus) < 90:
                    try:
                        m = await asyncio.wait_for(ws.receive(), timeout=4)
                    except asyncio.TimeoutError:
                        break
                    if m.type == aiohttp.WSMsgType.BINARY:
                        aus.append(m.data)
        return b"".join(aus), len(aus)

    print("── H.264 / WebSocket (browser WebCodecs) ──────────────────────")
    for name, (w, h) in (("main", (1280, 720)), ("wrist_left", (640, 480))):
        data, n = asyncio.run(ws_grab(name))
        info = _ffprobe(data)
        frames = _decode_count(data, w, h)
        baseline = "Baseline" in info and "yuv420p" in info
        good = n > 20 and frames > 20 and baseline
        ok &= good
        print(f"  {name:11s}: {n} AUs, decoded {frames} frames | {info} "
              f"| {'OK' if good else 'FAIL (must be 4:2:0 baseline)'}")

    print("── H.264 / UDP (iOS / Vision Pro) ─────────────────────────────")
    HEADER = struct.Struct("!2sBBHHHI")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0)); s.settimeout(0.5)

    def sub():
        import json
        while not stop.is_set():
            s.sendto(json.dumps({"sub": ["main"]}).encode(), ("127.0.0.1", 9592))
            time.sleep(1)
    threading.Thread(target=sub, daemon=True).start()
    parts, aus, t0 = {}, [], time.monotonic()
    while time.monotonic() - t0 < 6 and len(aus) < 60:
        try:
            pkt, _ = s.recvfrom(2048)
        except socket.timeout:
            continue
        if pkt[:2] != b"MV":
            continue
        _, _, sid, seq, idx, cnt, ts = HEADER.unpack(pkt[:14])
        d = parts.setdefault(seq, {}); d[idx] = pkt[14:]
        if len(d) == cnt:
            aus.append(b"".join(d[i] for i in range(cnt))); del parts[seq]
    frames = _decode_count(b"".join(aus), 1280, 720)
    good = frames > 20
    ok &= good
    print(f"  main       : {len(aus)} AUs, decoded {frames} frames | "
          f"{'OK' if good else 'FAIL'}")

    print("── MJPEG / HTTP (fallback) ────────────────────────────────────")
    import urllib.request
    d = urllib.request.urlopen("http://127.0.0.1:8599/camera/wrist_left/stream.mjpg",
                               timeout=4).read(200000)
    a = d.find(b"\xff\xd8"); b = d.find(b"\xff\xd9", a + 2)
    im = cv2.imdecode(np.frombuffer(d[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
    good = im is not None
    ok &= good
    print(f"  wrist_left : {im.shape[1]}x{im.shape[0]} {len(d[a:b+2])/1024:.1f} KB "
          f"| {'OK' if good else 'FAIL'}")

    stop.set(); vs.stop(); sink.stop()
    print("\n" + ("ALL TRANSPORTS PASS" if ok else "SOME TRANSPORTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
