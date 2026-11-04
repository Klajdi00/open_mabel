"""video_pipe.py — the ROS-BYPASS video path: camera worker → video relay, directly.

The per-process camera split originally routed network video THROUGH ROS: each
worker published full-resolution rgb8 Images over DDS (ZED HD720 ≈ 2.7 MB × 30 Hz)
and one relay process decoded + downscaled + JPEG-encoded every stream under a
single GIL. That both loaded DDS with ~100 MB/s of loopback traffic and serialised
all encode work — the operator saw it as video latency.

This module restores the original direct-push design across the process boundary:

    worker (owns the pixels)                       video relay (owns :8080/:9091/:9092)
    ────────────────────────                       ────────────────────────────────────
    downscale to net_width      ── unix DGRAM ──▶  VideoServer.push_jpeg(name, jpeg)
    JPEG-encode at net_quality      (25 KB)        (fan-out only; no decode, no encode*)
    (parallel, in-process)

  * H.264/NVENC needs raw frames: the relay decodes the small JPEG back to BGR only
    while an H.264 client is attached (640-wide — a quality trade the low-latency
    JPEG path never pays).

One datagram per (stream, frame): [u8 name_len][name][f64 t_capture][jpeg]. Unix
SOCK_DGRAM is message-oriented and lossy-by-design here: if the relay stalls, sends
fail with ENOBUFS/EAGAIN and the frame is DROPPED (newest-wins) instead of queueing
into latency. t_capture rides every packet, so the relay continuously measures the
TRUE capture→fan-out latency per stream and logs p50/p90 every 30 s — the number
that regressed is now permanently instrumented.
"""

import os
import socket
import struct
import time

SOCK_PATH = os.getenv("MABEL_VIDEO_PIPE", "/tmp/mabel_video_pipe.sock")
_MAX_DGRAM = 200_000          # stay under the unix dgram limit; a net frame is ~25 KB


class PipeVideo:
    """Worker-side drop-in for the drivers' ``video`` hook (wants/push/register).

    Downscales + JPEG-encodes in the WORKER process (the encode parallelism the
    per-camera split was built for) and datagrams the result to the relay.
    Always-warm: ``wants()`` is True so latency is constant from a client's first
    frame; the cost is ~2-3 ms/frame per worker at net size."""

    def __init__(self, net_width=640, jpeg_quality=65):
        self.net_width = int(net_width)
        self.quality = int(jpeg_quality)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    # -- driver hook API ------------------------------------------------------
    def register(self, *names):                       # relay owns registration
        pass

    def set_fallback(self, *a, **k):                  # relay owns 'main' aliasing
        pass

    def wants(self, name):                            # always-warm
        return True

    def push(self, names, bgr, width=None):
        """``width`` overrides net_width for this push — the side-by-side STEREO
        frame ('head_stereo') carries two eyes, so it goes out at 2×net_width
        to keep each eye at the same width every other stream gets."""
        import cv2
        import numpy as np
        if isinstance(names, str):
            names = (names,)
        target = int(width) if width else self.net_width
        h, w = bgr.shape[:2]
        if target and w > target:
            nh = max(1, int(round(h * target / float(w))))
            bgr = cv2.resize(bgr, (target, nh),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr),
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        jpeg = buf.tobytes()
        if len(jpeg) > _MAX_DGRAM - 64:
            return                                    # never a valid net frame
        t = struct.pack("<d", time.time())
        for n in names:
            nb = n.encode()
            pkt = bytes([len(nb)]) + nb + t + jpeg
            try:
                self._sock.sendto(pkt, SOCK_PATH)
            except OSError:
                pass                                  # relay down/full: drop, never queue


class PipeReceiver:
    """Relay-side receiver: yields (name, t_capture, jpeg) per datagram."""

    def __init__(self, path=SOCK_PATH):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sock.bind(path)
        # Deep receive buffer: absorbs a burst from 5 workers without drops while
        # the fan-out thread is momentarily busy.
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:
            pass

    def recv(self):
        pkt, _ = self._sock.recvfrom(_MAX_DGRAM)
        nlen = pkt[0]
        name = pkt[1:1 + nlen].decode()
        (t,) = struct.unpack_from("<d", pkt, 1 + nlen)
        return name, t, pkt[9 + nlen:]
