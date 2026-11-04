#!/usr/bin/env python3
"""
exp_h264_video.py — measure the MABEL video transports head-to-head against a LIVE
teleop stack (sim via run.sh, or the real robot). It is a pure network client — no
MuJoCo — so it sees exactly what an app sees on the wire:

    H.264 / UDP   :9092   'MV' datagrams, Annex-B access units   (LAN, match-source)
    H.264 / WS    :8080   /camera/<name>/h264 binary access units (browser)
    JPEG  / UDP   :9091   'MV' datagrams, whole JPEG per frame    (LAN-JPEG fallback)
    MJPEG / HTTP  :8080   /camera/<name>/stream.mjpg              (universal fallback)

For each it reports delivered resolution, achieved fps, mean bitrate, bytes/frame,
decode success, and — for the two UDP paths that carry the sender's `ts_ms` clock —
the per-frame freshness delay (min-subtracted, so it isolates queueing/jitter ABOVE
the lightest frame; absolute glass-to-glass also needs the client display stage,
bounded in the doc). An optional induced-loss sweep shows H.264's GOP-recovery vs
JPEG's per-frame independence.

H.264 is decoded with PyAV (its bundled libav — independent of the dyld-broken
Homebrew ffmpeg); JPEG with OpenCV. Writes vector PDFs into docs/figures/ and a JSON
sidecar, prints a summary table, fixed seed. Run from server/ with the stack up:

    ./run.sh                                   # terminal 1 (or start_teleop.sh)
    ~/anaconda3/bin/python3 tools/exp_h264_video.py --host 127.0.0.1 --camera main
    ~/anaconda3/bin/python3 tools/exp_h264_video.py --secs 20 --loss 0,2,5,10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import struct
import time
from collections import defaultdict

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
FIGDIR = os.path.join(SERVER, "docs", "figures")

# 'MV' datagram header — must match udp_video.py exactly.
HEADER = struct.Struct("!2sBBHHHI")          # magic, ver, stream, seq, idx, count, ts_ms
MAGIC = b"MV"
AUD3 = b"\x00\x00\x01\x09"


# ─────────────────────────────────────────────────────────────────────────────
# Collectors: each returns a list of frames [{bytes, ts_ms|None, t_arrival}]
# ─────────────────────────────────────────────────────────────────────────────

def collect_mv_udp(host, port, stream, secs, loss=0.0, rng=None):
    """Subscribe to the 'MV' UDP video path, reassemble frames latest-wins, return one
    record per COMPLETE frame. `loss` (0..1) drops that fraction of inbound datagrams to
    emulate a lossy link. ts_ms is the sender's monotonic-ms clock."""
    rng = rng or np.random.default_rng(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.settimeout(0.5)
    sub = json.dumps({"sub": [stream]}).encode()
    sock.sendto(sub, (host, port))
    parts: dict[tuple, dict] = {}            # (sid,seq) -> {chunks{idx:bytes}, count, ts, t}
    frames = []
    t_end = time.monotonic() + secs
    last_sub = 0.0
    while time.monotonic() < t_end:
        now = time.monotonic()
        if now - last_sub > 1.0:             # keepalive (server TTL = 3 s)
            sock.sendto(sub, (host, port)); last_sub = now
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        if len(data) < HEADER.size or data[:2] != MAGIC:
            continue                         # the {"map":...} reply, etc.
        if loss and rng.random() < loss:
            continue                         # emulate datagram loss
        magic, ver, sid, seq, idx, count, ts = HEADER.unpack(data[:HEADER.size])
        payload = data[HEADER.size:]
        key = (sid, seq)
        p = parts.get(key)
        if p is None:
            # Drop partials older than the newest seq we've seen (wrap-aware).
            parts[key] = p = {"chunks": {}, "count": count, "ts": ts, "t": time.monotonic()}
        p["chunks"][idx] = payload
        if len(p["chunks"]) == p["count"]:
            buf = b"".join(p["chunks"][i] for i in range(p["count"]))
            frames.append({"bytes": buf, "ts_ms": ts, "t": time.monotonic()})
            del parts[key]
            # prune stale partials so memory stays bounded under loss
            if len(parts) > 64:
                for k in sorted(parts, key=lambda k: parts[k]["t"])[:32]:
                    parts.pop(k, None)
    sock.close()
    return frames


def collect_ws_h264(host, port, stream, secs):
    """Receive Annex-B access units (one per binary message) from /camera/<name>/h264."""
    frames = []

    async def _run():
        import aiohttp
        url = f"http://{host}:{port}/camera/{stream}/h264"
        timeout = aiohttp.ClientTimeout(total=secs + 5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.ws_connect(url, max_msg_size=0) as ws:
                    t_end = time.monotonic() + secs
                    while time.monotonic() < t_end:
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            frames.append({"bytes": msg.data, "ts_ms": None,
                                          "t": time.monotonic()})
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception as e:  # noqa: BLE001
            print(f"   [ws] {url} unavailable: {e}")

    asyncio.run(_run())
    return frames


def collect_mjpeg(host, port, stream, secs):
    """Pull the multipart MJPEG stream and split it into whole JPEGs (SOI..EOI)."""
    import urllib.request
    url = f"http://{host}:{port}/camera/{stream}/stream.mjpg"
    frames = []
    try:
        resp = urllib.request.urlopen(url, timeout=5)
    except Exception as e:  # noqa: BLE001
        print(f"   [mjpeg] {url} unavailable: {e}")
        return frames
    buf = b""
    t_end = time.monotonic() + secs
    resp.fp.raw._sock.settimeout(1.0) if hasattr(resp, "fp") else None
    while time.monotonic() < t_end:
        try:
            chunk = resp.read(65536)
        except Exception:  # noqa: BLE001
            break
        if not chunk:
            break
        buf += chunk
        while True:
            i = buf.find(b"\xff\xd8")        # SOI
            j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1   # EOI
            if i < 0 or j < 0:
                break
            frames.append({"bytes": buf[i:j + 2], "ts_ms": None, "t": time.monotonic()})
            buf = buf[j + 2:]
    try:
        resp.close()
    except Exception:  # noqa: BLE001
        pass
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Decode + metrics
# ─────────────────────────────────────────────────────────────────────────────

def decode_h264(frames):
    """Feed Annex-B access units to PyAV in order; return (n_decoded, (w,h))."""
    try:
        import av
    except Exception as e:  # noqa: BLE001
        print(f"   [decode] PyAV unavailable: {e}")
        return 0, None
    dec = av.codec.CodecContext.create("h264", "r")
    n, res = 0, None
    for fr in frames:
        try:
            for pkt in dec.parse(fr["bytes"]):
                for img in dec.decode(pkt):
                    n += 1
                    res = (img.width, img.height)
        except Exception:  # noqa: BLE001
            continue
    return n, res


def decode_jpeg(frames):
    n, res = 0, None
    for fr in frames:
        img = cv2.imdecode(np.frombuffer(fr["bytes"], np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            n += 1
            res = (img.shape[1], img.shape[0])
    return n, res


def summarize(name, frames, secs, codec):
    n_recv = len(frames)
    nbytes = sum(len(f["bytes"]) for f in frames)
    if codec == "h264":
        n_dec, res = decode_h264(frames)
    else:
        n_dec, res = decode_jpeg(frames)
    # freshness delay (only when ts_ms present): arrival - ts_ms, min-subtracted (ms).
    lat = None
    ts_frames = [f for f in frames if f["ts_ms"] is not None]
    if len(ts_frames) >= 5:
        d = np.array([(f["t"] * 1000.0) - f["ts_ms"] for f in ts_frames], float)
        lat = d - d.min()
    return {
        "name": name, "codec": codec,
        "recv_fps": n_recv / secs,
        "decoded_fps": n_dec / secs,
        "decode_ok": (n_dec / n_recv) if n_recv else 0.0,
        "mbps": nbytes * 8 / 1e6 / secs,
        "kb_per_frame": (nbytes / n_recv / 1024) if n_recv else 0.0,
        "res": res,
        "lat_med": float(np.median(lat)) if lat is not None else None,
        "lat_p95": float(np.percentile(lat, 95)) if lat is not None else None,
        "lat": lat,
    }


def print_table(rows):
    print("\n" + "=" * 96)
    print(f"{'transport':16} {'resolution':12} {'recv':>6} {'dec':>6} {'ok':>5} "
          f"{'Mbps':>7} {'KB/fr':>7} {'lat_med':>8} {'lat_p95':>8}")
    print(f"{'':16} {'(WxH)':12} {'fps':>6} {'fps':>6} {'%':>5} {'':>7} {'':>7} "
          f"{'ms*':>8} {'ms*':>8}")
    print("-" * 96)
    for r in rows:
        res = f"{r['res'][0]}x{r['res'][1]}" if r["res"] else "—"
        lm = f"{r['lat_med']:.1f}" if r["lat_med"] is not None else "—"
        lp = f"{r['lat_p95']:.1f}" if r["lat_p95"] is not None else "—"
        print(f"{r['name']:16} {res:12} {r['recv_fps']:6.1f} {r['decoded_fps']:6.1f} "
              f"{r['decode_ok']*100:5.0f} {r['mbps']:7.2f} {r['kb_per_frame']:7.1f} "
              f"{lm:>8} {lp:>8}")
    print("=" * 96)
    print("* lat = freshness delay above the lightest frame (min-subtracted, ms); "
          "UDP paths only (carry ts_ms).\n")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_bitrate_res(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["name"] for r in rows]
    mbps = [r["mbps"] for r in rows]
    colors = ["#2a7" if r["codec"] == "h264" else "#a55" for r in rows]
    bars = ax.bar(names, mbps, color=colors)
    for b, r in zip(bars, rows):
        res = f"{r['res'][0]}x{r['res'][1]}" if r["res"] else "—"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{res}\n{r['mbps']:.2f} Mbps", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mean bitrate (Mbps)")
    ax.set_title("Bitrate vs delivered resolution — H.264 (green) carries\n"
                 "more pixels per bit than JPEG (red)")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_latency(rows, path):
    have = [r for r in rows if r["lat"] is not None and len(r["lat"]) > 5]
    if not have:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([r["lat"] for r in have], showfliers=False)
    ax.set_xticks(range(1, len(have) + 1))
    ax.set_xticklabels([r["name"] for r in have])
    ax.set_ylabel("freshness delay above lightest frame (ms)")
    ax.set_title("Per-frame queueing/jitter delay (min-subtracted) — UDP paths")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_loss(loss_pts, h264_fps, jpeg_fps, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_pts, h264_fps, "o-", color="#2a7", label="H.264/UDP (decoded)")
    ax.plot(loss_pts, jpeg_fps, "s-", color="#a55", label="JPEG/UDP (decoded)")
    ax.set_xlabel("induced datagram loss (%)")
    ax.set_ylabel("decoded fps")
    ax.set_title("Loss robustness: JPEG is intra-only (per-frame); H.264 recovers\n"
                 "at the next IDR, so a lost chunk can stall a GOP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MABEL video transport comparison (live)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--camera", default="main")
    ap.add_argument("--secs", type=float, default=15.0, help="measurement window per transport")
    ap.add_argument("--h264-udp-port", type=int, default=9092)
    ap.add_argument("--jpeg-udp-port", type=int, default=9091)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--loss", default="", help="comma list of loss %% for the UDP sweep, e.g. 0,2,5,10")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"\nMABEL video transports — host={args.host} camera={args.camera} "
          f"{args.secs:.0f}s/transport")
    print("(bring the stack up first: ./run.sh)\n")

    rows = []
    print(" • H.264 / UDP :%d …" % args.h264_udp_port)
    rows.append(summarize("H.264/UDP", collect_mv_udp(
        args.host, args.h264_udp_port, args.camera, args.secs), args.secs, "h264"))
    print(" • H.264 / WS  :%d/camera/%s/h264 …" % (args.http_port, args.camera))
    rows.append(summarize("H.264/WS", collect_ws_h264(
        args.host, args.http_port, args.camera, args.secs), args.secs, "h264"))
    print(" • JPEG  / UDP :%d …" % args.jpeg_udp_port)
    rows.append(summarize("JPEG/UDP", collect_mv_udp(
        args.host, args.jpeg_udp_port, args.camera, args.secs), args.secs, "jpeg"))
    print(" • MJPEG / HTTP :%d …" % args.http_port)
    rows.append(summarize("MJPEG/HTTP", collect_mjpeg(
        args.host, args.http_port, args.camera, args.secs), args.secs, "jpeg"))

    print_table(rows)

    fig_bitrate_res(rows, os.path.join(FIGDIR, "h264_vs_jpeg_bitrate.pdf"))
    did_lat = fig_latency(rows, os.path.join(FIGDIR, "h264_vs_jpeg_latency.pdf"))
    print(f"wrote {FIGDIR}/h264_vs_jpeg_bitrate.pdf"
          + ("  +h264_vs_jpeg_latency.pdf" if did_lat else ""))

    loss_pts, h264_fps, jpeg_fps = [], [], []
    if args.loss:
        print("\nloss sweep (decoded fps vs induced UDP loss):")
        for s in args.loss.split(","):
            p = float(s) / 100.0
            hf = decode_h264(collect_mv_udp(args.host, args.h264_udp_port, args.camera,
                                            args.secs, loss=p, rng=rng))[0] / args.secs
            jf = decode_jpeg(collect_mv_udp(args.host, args.jpeg_udp_port, args.camera,
                                            args.secs, loss=p, rng=rng))[0] / args.secs
            loss_pts.append(float(s)); h264_fps.append(hf); jpeg_fps.append(jf)
            print(f"   loss {s:>4}%   H.264 {hf:5.1f} fps   JPEG {jf:5.1f} fps")
        fig_loss(loss_pts, h264_fps, jpeg_fps, os.path.join(FIGDIR, "h264_loss_recovery.pdf"))
        print(f"wrote {FIGDIR}/h264_loss_recovery.pdf")

    # JSON sidecar (drop the raw latency arrays).
    out = {"host": args.host, "camera": args.camera, "secs": args.secs,
           "transports": [{k: v for k, v in r.items() if k != "lat"} for r in rows],
           "loss_sweep": {"pct": loss_pts, "h264_fps": h264_fps, "jpeg_fps": jpeg_fps}}
    with open(os.path.join(FIGDIR, "h264_video.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {FIGDIR}/h264_video.json\n")


if __name__ == "__main__":
    main()
