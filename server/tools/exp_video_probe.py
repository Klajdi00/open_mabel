#!/usr/bin/env python3
"""exp_video_probe.py — rigorous check that every ADVERTISED video transport of a
RUNNING teleop server actually delivers frames, exactly as the apps consume them.

  1. hello        ws :9090 → read video.{udpPort, h264Port}; a port advertised
                  but not delivering is the failure mode that blacked out the
                  headset (hello said h264Port=9092 while the encoder was off).
  2. MJPEG        http :8080/camera/<name>/stream.mjpg — bytes + JPEG SOI count.
  3. UDP-JPEG     :9091 — subscribe {"sub": [...]} like UDPVideoReceiver, decode
                  the 14-byte 'MV' chunk headers, count complete frames/stream.
  4. H.264        :9092 — only when advertised; same 'MV' framing.

Run:  python3 server/tools/exp_video_probe.py     (server must be up; exits 0 on pass)
"""
import asyncio
import json
import socket
import struct
import sys
import time
import urllib.request

HOST = "127.0.0.1"
STREAMS = ["main", "wrist_left", "wrist_right"]


def get_hello():
    import websockets

    async def go():
        async with websockets.connect(f"ws://{HOST}:9090/teleop", open_timeout=5) as ws:
            return json.loads(await asyncio.wait_for(ws.recv(), 5))["payload"]
    return asyncio.run(go())


def probe_mjpeg(name, secs=3.0):
    url = f"http://{HOST}:8080/camera/{name}/stream.mjpg"
    t0 = time.time()
    n_bytes = 0
    frames = 0
    with urllib.request.urlopen(url, timeout=5) as r:
        while time.time() - t0 < secs:
            chunk = r.read(65536)
            if not chunk:
                break
            n_bytes += len(chunk)
            frames += chunk.count(b"\xff\xd8\xff")      # JPEG SOI markers
    return frames / secs, n_bytes


def probe_mv(port, secs=3.0):
    """Subscribe like the app's UDP receiver; count COMPLETE frames per stream."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    sock.bind(("0.0.0.0", 0))
    sub = json.dumps({"sub": STREAMS}).encode()
    sock.sendto(sub, (HOST, port))
    id_map = {}
    frames = {}                 # (stream_id, seq) -> set(chunk_idx)
    done = {}                   # stream_id -> complete frame count
    t0 = time.time()
    last_sub = t0
    while time.time() - t0 < secs:
        if time.time() - last_sub > 0.8:
            sock.sendto(sub, (HOST, port))
            last_sub = time.time()
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        if data[:1] == b"{":
            try:
                id_map = json.loads(data.decode())["map"]
            except Exception:
                pass
            continue
        if len(data) < 14 or data[:2] != b"MV":
            continue
        sid = data[3]
        seq, cidx, ctot = struct.unpack(">HHH", data[4:10])
        key = (sid, seq)
        got = frames.setdefault(key, set())
        got.add(cidx)
        if len(got) == ctot:
            done[sid] = done.get(sid, 0) + 1
            del frames[key]
    sock.close()
    rev = {v: k for k, v in id_map.items()}
    return {rev.get(sid, f"id{sid}"): n / secs for sid, n in done.items()}, id_map


def main():
    ok = True
    hello = get_hello()
    video = hello.get("video", {})
    udp_port, h264_port = int(video.get("udpPort", 0)), int(video.get("h264Port", 0))
    print(f"hello advertises: udpPort={udp_port} h264Port={h264_port}")

    # MJPEG
    for name in STREAMS:
        fps, nbytes = probe_mjpeg(name)
        good = fps >= 5.0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  MJPEG {name:12s} {fps:5.1f} fps "
              f"({nbytes/1024:.0f} KiB/3s)")

    # UDP-JPEG (the headset's LAN path when H.264 is off)
    if udp_port:
        rates, id_map = probe_mv(udp_port)
        for name in STREAMS:
            fps = rates.get(name, 0.0)
            good = fps >= 5.0
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  UDP-JPEG :{udp_port} {name:12s} {fps:5.1f} fps")
    else:
        print("  FAIL  hello advertises no udpPort")
        ok = False

    # H.264 — must deliver IF advertised (advertised-but-dead = headset blackout)
    if h264_port:
        rates, _ = probe_mv(h264_port)
        fps = rates.get("main", 0.0)
        good = fps >= 5.0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  H.264 :{h264_port} main {fps:5.1f} fps "
              f"(advertised → MUST deliver)")
    else:
        print("  PASS  H.264 not advertised (encoder off) → apps go straight to UDP-JPEG")

    print("VIDEO " + ("PASS — every advertised transport delivers" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
