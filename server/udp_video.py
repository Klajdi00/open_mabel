"""
udp_video.py — latest-frame-wins JPEG video over UDP.

Why UDP instead of the MJPEG/HTTP stream for the headset
────────────────────────────────────────────────────────
MJPEG rides on TCP: one lost packet stalls the WHOLE stream until the
retransmit lands (head-of-line blocking), and the kernel send buffer grows a
queue of stale frames whenever the link hiccups. For teleop video the freshest
frame is the only one worth showing, so the right transport drops instead of
queueing: each JPEG is chunked into datagrams and fired at subscribers; a lost
chunk costs exactly that frame and the next one renders on time. Glass-to-glass
latency is then just encode + wire + decode — no jitter buffer, no backlog.
JPEG (intra-only) keeps the codec pipeline depth at zero frames; over a good
LAN or a Tailscale/WireGuard tunnel (UDP passes through unchanged) this is the
minimum-latency configuration short of hardware H.264 with a custom RTP stack.

Wire protocol (port 9091 by default)
────────────────────────────────────
Client → server, every ~1 s (subscribe + keepalive; expires after 3 s):
    {"sub": ["main", "head_right", "wrist_left", "wrist_right"]}
Server → client immediately after each sub (stream-name → u8 id map):
    {"map": {"main": 0, "head_right": 1, ...}}
Server → client, per frame, one or more binary datagrams:

    0      2      3       4        6         8          10      14
    ┌──────┬──────┬───────┬────────┬─────────┬──────────┬───────┐
    │ "MV" │ ver  │stream │ seq    │ chunk   │ chunks   │ ts_ms │ payload…
    │ 2 B  │ u8=1 │ id u8 │ u16 BE │ idx u16 │ total u16│ u32   │ ≤1182 B
    └──────┴──────┴───────┴────────┴─────────┴──────────┴───────┘

Datagrams are ≤ 1196 B so they clear Tailscale's 1280-byte MTU without IP
fragmentation (a fragmented datagram is lost if ANY fragment is). `ts_ms` is
the sender's monotonic clock in ms (mod 2^32) — used by the latency harness.

Usage (both camera-rendering processes):
    sender = UDPVideoSender(port=9091)
    sender.start()
    sender.attach(path_caps)        # tees every PushCapture.push() to UDP
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time

log = logging.getLogger("udp_video")

MAGIC = b"MV"
VERSION = 1
HEADER = struct.Struct("!2sBBHHHI")          # magic, ver, stream, seq, idx, count, ts_ms
CHUNK_PAYLOAD = 1196 - HEADER.size           # 1182 B — datagram fits Tailscale MTU 1280
SUB_TTL = 3.0                                # seconds before a silent client is dropped


def stream_name(path: str) -> str:
    """'/camera/wrist_left/stream.mjpg' → 'wrist_left' (the wire stream key)."""
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "camera":
        return parts[1]
    return parts[-1] if parts else path


class UDPVideoSender:
    """One UDP socket serving every camera stream of this process.

    send() is called from the render thread; the datagram writes are
    non-blocking sendto()s (a full OS buffer drops the frame — latest wins,
    exactly the semantics we want). A tiny daemon thread handles subscribes.
    """

    def __init__(self, port: int = 9091, host: str = "0.0.0.0"):
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._ids: dict[str, int] = {}                    # stream name → u8 id
        self._seq: dict[str, int] = {}                    # stream name → u16 frame seq
        self._subs: dict[tuple, dict] = {}                # addr → {streams, t}
        self._caps: dict[str, object] = {}                # stream name → PushCapture
        self._running = False
        self._t0 = time.monotonic()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._sub_loop, name="udp-video-subs",
                         daemon=True).start()
        log.info("UDP video up   udp://0.0.0.0:%d (latest-frame-wins JPEG)", self.port)

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass

    def attach(self, path_caps: dict) -> None:
        """Tee every PushCapture in `path_caps` into this sender. The HTTP MJPEG
        path keeps working untouched — UDP is additive, MJPEG is the fallback."""
        for path, cap in path_caps.items():
            name = stream_name(path)
            with self._lock:
                self._ids.setdefault(name, len(self._ids))
                self._caps[name] = cap
            cap.on_push = (lambda jpeg, n=name: self.send(n, jpeg))
        log.info("UDP video streams: %s", ", ".join(sorted(self._ids)))

    # ── subscribe handling (daemon thread) ───────────────────────────────

    def _sub_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                msg = json.loads(data.decode("utf-8"))
                streams = set(map(str, msg.get("sub", [])))
            except (ValueError, UnicodeDecodeError, AttributeError):
                continue
            with self._lock:
                prev = self._subs.get(addr, {}).get("streams", set())
                fresh = addr not in self._subs
                self._subs[addr] = {"streams": streams, "t": time.monotonic()}
                for s in streams:                       # ids for names we may
                    self._ids.setdefault(s, len(self._ids))  # not render (yet)
                id_map = dict(self._ids)
                new_streams = streams - prev
            if fresh:
                log.info("UDP video client %s:%d ← %s", addr[0], addr[1],
                         ", ".join(sorted(streams)) or "(none)")
            try:
                self._sock.sendto(json.dumps({"map": id_map}).encode(), addr)
            except OSError:
                pass
            # Warm start: replay the newest cached frame of each NEWLY
            # subscribed stream to this client only, so slow streams (wrist
            # cams render at <1 fps under the weighted schedule) show a
            # picture immediately instead of after their next render slot.
            for s in new_streams:
                cap = self._caps.get(s)
                jpeg = getattr(cap, "_latest", None) if cap is not None else None
                if jpeg:
                    self._emit(s, jpeg, [addr])

    # ── subscriber query (render thread, demand-driven render) ────────────

    def has_subscriber(self, stream: str) -> bool:
        """True iff a live (un-expired) UDP client is subscribed to `stream`.
        Lets the camera render loop skip streams nobody is watching."""
        now = time.monotonic()
        with self._lock:
            return any(stream in s["streams"] and now - s["t"] <= SUB_TTL
                       for s in self._subs.values())

    # ── frame send (render thread) ───────────────────────────────────────

    def send(self, stream: str, jpeg: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            dead = [a for a, s in self._subs.items() if now - s["t"] > SUB_TTL]
            for a in dead:
                log.info("UDP video client %s:%d expired", a[0], a[1])
                del self._subs[a]
            targets = [a for a, s in self._subs.items() if stream in s["streams"]]
        if not targets:
            return
        self._emit(stream, jpeg, targets)

    def _emit(self, stream: str, jpeg: bytes, targets: list) -> None:
        """Chunk one JPEG and send it to `targets` (each frame gets a fresh
        sequence number, so a warm-start replay is superseded naturally)."""
        with self._lock:
            sid = self._ids.setdefault(stream, len(self._ids))
            seq = self._seq[stream] = (self._seq.get(stream, 0) + 1) & 0xFFFF
        ts_ms = int((time.monotonic() - self._t0) * 1000) & 0xFFFFFFFF
        nchunks = max(1, (len(jpeg) + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD)
        for i in range(nchunks):
            payload = jpeg[i * CHUNK_PAYLOAD:(i + 1) * CHUNK_PAYLOAD]
            pkt = HEADER.pack(MAGIC, VERSION, sid, seq, i, nchunks, ts_ms) + payload
            for addr in targets:
                try:
                    self._sock.sendto(pkt, addr)
                except (BlockingIOError, InterruptedError):
                    break                  # OS buffer full → drop frame remainder
                except OSError:
                    pass                   # unreachable client → keepalive expires it
