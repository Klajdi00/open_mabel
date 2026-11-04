"""udp_sink.py — low-latency UDP video out for the bridge: H.264 (NVENC) + JPEG.

This is the SOTA transport that sits alongside the MJPEG/HTTP server (video_server.py).
UDP drops instead of queueing, so there is NO TCP head-of-line backlog — glass-to-glass
is just encode + wire + decode. Two sinks, both demand-driven (nothing runs until a UDP
client subscribes), both using the proven 'MV' datagram protocol in server/udp_video.py
so the SAME client works against the sim, the old sensors node, and this bridge:

  • H.264  udp :9092 — one ffmpeg `h264_nvenc` (hardware NVENC on Jetson Thor) per stream,
    fed raw BGR on a pipe, low-latency (-tune ll, baseline, IDR + in-band SPS/PPS every
    ~1 s so a late / lossy joiner recovers), AUD-delimited. We split the Annex-B stream on
    access-unit delimiters and send each access unit as ONE logical 'MV' message (chunked
    to fit the MTU). The client reassembles → VideoToolbox.
  • JPEG   udp :9091 — the already-encoded MJPEG frame, chunked over UDP (intra-only, zero
    codec latency). A simpler fallback for clients without an H.264 decoder.

Wire protocol + client subscribe/keepalive are documented in server/udp_video.py.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from mabel_hw import paths

# Reuse the SINGLE source of the UDP 'MV' protocol. APPEND server/ (low priority) — do
# NOT insert at position 0, or server/config.py would shadow the bridge's own config.py.
_SERVER_DIR = os.path.join(paths.mabel_root(), "server")
if _SERVER_DIR not in sys.path:
    sys.path.append(_SERVER_DIR)
import udp_video  # noqa: E402


# H.264 Annex-B access-unit delimiter (NAL type 9). NVENC emits one per access unit
# with `-aud 1`; we cut the byte stream on it so each datagram group = one frame.
_AUD = b"\x00\x00\x00\x01\x09"
_AUD3 = b"\x00\x00\x01\x09"


class _H264Fanout:
    """One ffmpeg h264_nvenc per stream, fed raw BGR, output chunked over UDP. Demand-
    driven: ffmpeg is only FED when a subscriber is present, so it idles at ~0 CPU."""

    def __init__(self, sender, fps, bitrate="2M", ffmpeg=None, log=None):
        self._udp = sender                       # a udp_video.UDPVideoSender (:9092)
        self._fps = int(fps)
        self._bitrate = bitrate
        self._ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self._log = log
        self._raw: dict[str, np.ndarray] = {}    # name -> latest BGR
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        # Optional second consumer of the SAME encoded access units: the VideoServer's
        # WebSocket endpoint (browser WebCodecs). One NVENC encoder feeds both UDP and
        # WS. set via set_ws_hook(); both are demand-driven and independent.
        self._on_au = None                       # (name, access_unit_bytes) -> None
        self._extra_wants = None                 # (name) -> bool

    def set_ws_hook(self, on_au, extra_wants):
        self._on_au = on_au
        self._extra_wants = extra_wants

    def _wanted(self, name):
        if self._udp.has_subscriber(name):
            return True
        return self._extra_wants is not None and bool(self._extra_wants(name))

    def has_any(self, name):
        return self._wanted(name)

    def feed(self, name, bgr):
        """Stash the latest BGR for `name` and lazily start its encoder."""
        with self._lock:
            self._raw[name] = bgr
            if name not in self._procs:
                self._start(name, int(bgr.shape[1]), int(bgr.shape[0]))

    def _start(self, name, w, h):
        # Low-latency hardware H.264: p1 (fastest) + ll tune + CBR, baseline (no B-frames
        # → simplest for VideoToolbox), forced IDR every `fps` frames (~1 s) and dump_extra
        # so SPS/PPS ride in-band with each IDR (a late/lossy UDP joiner can then start
        # decoding within a second). `-aud 1` delimits access units for our framing.
        # CRITICAL: -pix_fmt yuv420p + -profile:v baseline. The raw input is bgr24
        # (4:4:4); without forcing 4:2:0 NVENC picks a High-4:4:4 profile that ffmpeg's
        # SOFTWARE decoder accepts but iOS VideoToolbox and browser WebCodecs HARDWARE
        # decoders REJECT — silent black video on the real clients. baseline (no CABAC /
        # no B-frames) is the universally-decodable, lowest-latency teleop profile.
        cmd = [self._ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-framerate", str(self._fps), "-i", "pipe:0", "-an",
               "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
               "-preset", "p1", "-tune", "ll", "-rc", "cbr", "-b:v", self._bitrate,
               "-g", str(self._fps), "-forced-idr", "1", "-aud", "1",
               "-bsf:v", "dump_extra=freq=keyframe", "-f", "h264", "pipe:1"]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as e:                   # noqa: BLE001
            if self._log:
                self._log.warn(f"h264 ffmpeg start failed ({e}); H.264 off for {name}.")
            return
        self._procs[name] = p
        threading.Thread(target=self._feed_loop, args=(name, p), daemon=True).start()
        threading.Thread(target=self._read_loop, args=(name, p), daemon=True).start()

    def _feed_loop(self, name, p):
        period = 1.0 / self._fps
        while p.poll() is None:
            if not self._wanted(name):                 # demand-driven (UDP OR WS client)
                time.sleep(0.1)
                continue
            with self._lock:
                bgr = self._raw.get(name)
            if bgr is not None:
                try:
                    p.stdin.write(np.ascontiguousarray(bgr).tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    break
            time.sleep(period)

    def _read_loop(self, name, p):
        buf = b""
        while p.poll() is None:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:                                # split on AUD = access-unit edge
                i = buf.find(_AUD3)
                if i < 0:
                    break
                j = buf.find(_AUD3, i + 4)
                if j < 0:
                    break
                au = buf[i:j]                          # one complete access unit
                self._udp.send(name, au)               # → UDP subscribers ('MV' chunks)
                if self._on_au is not None:
                    try:
                        self._on_au(name, au)          # → WebSocket clients (browser)
                    except Exception:                  # noqa: BLE001
                        pass
                buf = buf[j:]

    def stop(self):
        for p in list(self._procs.values()):
            try:
                p.terminate()
            except Exception:                          # noqa: BLE001
                pass


class UdpVideoSink:
    """The bridge's UDP video out: JPEG (:9091) + H.264 (:9092), both 'MV' protocol."""

    def __init__(self, fps=30, jpeg_port=9091, h264_port=9092, h264_bitrate="2M",
                 enable_jpeg=True, enable_h264=True, ffmpeg=None, log=None):
        self.log = log
        self._jpeg = udp_video.UDPVideoSender(port=jpeg_port) if enable_jpeg else None
        self._h264sender = udp_video.UDPVideoSender(port=h264_port) if enable_h264 else None
        if self._jpeg:
            self._jpeg.start()
        self._h264 = None
        if self._h264sender:
            self._h264sender.start()
            self._h264 = _H264Fanout(self._h264sender, fps, h264_bitrate, ffmpeg, log)
        if log:
            log.info(f"UDP video: JPEG :{jpeg_port if enable_jpeg else 'off'} | "
                     f"H.264 :{h264_port if enable_h264 else 'off'} (NVENC)")

    # -- demand probes (let the capture loop skip unwatched cams) ----------
    def wants_jpeg(self, name):
        return self._jpeg is not None and self._jpeg.has_subscriber(name)

    def wants_h264(self, name):
        return self._h264 is not None and self._h264.has_any(name)

    def set_ws_hook(self, on_au, extra_wants):
        """Let the VideoServer's WebSocket H.264 endpoint consume the SAME encoded
        access units as the UDP path (one NVENC encoder, two transports)."""
        if self._h264 is not None:
            self._h264.set_ws_hook(on_au, extra_wants)

    def wants(self, name):
        return self.wants_jpeg(name) or self.wants_h264(name)

    # -- sinks (called from the VideoServer, which already downscaled the frame) --
    def send_jpeg(self, name, jpeg):
        if self._jpeg is not None:
            self._jpeg.send(name, jpeg)

    def feed_h264(self, name, bgr):
        if self._h264 is not None:
            self._h264.feed(name, bgr)

    def stop(self):
        if self._h264 is not None:
            self._h264.stop()
        for s in (self._jpeg, self._h264sender):
            if s is not None:
                try:
                    s.close()
                except Exception:                      # noqa: BLE001
                    pass
