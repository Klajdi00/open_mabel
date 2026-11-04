"""video_relay.py — the network-video process: camera-worker JPEGs → MJPEG/UDP.

NO ROS ANYWHERE on the video path (2026-07-10 latency fix). The per-camera workers
downscale + JPEG-encode their own frames and datagram them here over a unix socket
(cameras/video_pipe.py); this process only FANS OUT: the :8080 MJPEG server, UDP
JPEG :9091, and H.264/NVENC :9092 (raw frames recovered from the small JPEG only
while an H.264 client is attached). The previous design published full-resolution
rgb8 Images over DDS (ZED ≈ 100 MB/s of loopback traffic) into one GIL-shared
encoding loop — the operator saw it as video latency.

'main' (the apps' hard-coded primary view) is aliased PER-FRAME: the ZED worker
pushes ("main","head_left") itself while alive; when its packets go quiet (>1.5 s)
the front cam's frames are served as 'main' too, and it switches back mid-
connection the moment ZED packets resume — no client reconnect.

Every packet carries its capture timestamp, so this process CONTINUOUSLY measures
the true capture→fan-out latency per stream and logs p50/p90 every 30 s — the
metric that regressed is permanently instrumented. (grep the bridge journal for
"video latency".)
"""

import logging
import signal
import sys
import time

from mabel_hw import config

from .video_pipe import PipeReceiver
from .video_server import VideoServer

log = logging.getLogger("video_relay")
logging.basicConfig(level=logging.INFO,
                    format="[video-relay] %(message)s", stream=sys.stdout)

_ZED_ALIVE_S = 1.5        # 'main' falls back to front when ZED packets pause this long
_STATS_EVERY_S = 30.0

# 'head_stereo' = the ZED's rectified LEFT|RIGHT from ONE grab, side by side
# (2×net_width wide) — the headset's stereo source: both eyes from the same
# exposure, so the pair can never be out of step (zed_mini.py).
STREAMS = ("main", "front", "head_left", "wrist_left", "wrist_right", "head_stereo")


def main():
    cfg = config.CAMERAS
    udp = None
    u = cfg.get("udp", {})
    if u.get("enabled", True):
        from .udp_sink import UdpVideoSink
        udp = UdpVideoSink(
            fps=u.get("fps", 30),
            jpeg_port=u.get("jpeg_port", 9091),
            h264_port=u.get("h264_port", 9092),
            h264_bitrate=u.get("h264_bitrate", "2M"),
            enable_jpeg=u.get("jpeg", True),
            enable_h264=u.get("h264", True),
            log=log)
    video = VideoServer(cfg.get("http_host", "0.0.0.0"),
                        cfg.get("http_port", 8080),
                        net_width=cfg.get("net_width", 640),
                        jpeg_quality=cfg.get("net_quality", 65),
                        udp=udp,
                        h264_width=cfg.get("udp", {}).get("h264_width", 1280))
    video.register(*STREAMS)
    video.start()
    rx = PipeReceiver()
    log.info("up: worker JPEGs -> MJPEG :%d + UDP (NO ROS on the video path).",
             cfg.get("http_port", 8080))

    running = [True]
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__(0, False))

    last_zed = 0.0
    lat = {n: [] for n in STREAMS}                   # capture->here, seconds
    last_stats = time.monotonic()
    while running[0]:
        try:
            name, t_cap, jpeg = rx.recv()
        except OSError:
            continue
        now = time.time()
        if name in lat:
            lat[name].append(now - t_cap)
        if name == "head_left":
            last_zed = time.monotonic()
        video.push_jpeg(name, jpeg)
        # front carries 'main' while the ZED is silent (per-frame failover; the
        # ZED worker pushes 'main' itself while alive).
        if name == "front" and (time.monotonic() - last_zed) > _ZED_ALIVE_S:
            video.push_jpeg("main", jpeg)

        if time.monotonic() - last_stats >= _STATS_EVERY_S:
            last_stats = time.monotonic()
            parts = []
            for n, xs in lat.items():
                if xs:
                    xs.sort()
                    parts.append(f"{n} p50={xs[len(xs)//2]*1000:.0f}ms "
                                 f"p90={xs[int(len(xs)*0.9)]*1000:.0f}ms n={len(xs)}")
                lat[n] = []
            if parts:
                log.info("video latency (capture->fanout): " + " | ".join(parts))

    video.stop()
    if udp is not None:
        udp.stop()


if __name__ == "__main__":
    main()
