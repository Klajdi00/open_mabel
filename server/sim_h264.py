"""sim_h264.py — H.264 video out for the SIMULATOR, byte-for-byte compatible with
the real robot (hardware_bridge/cameras/udp_sink.py).

Why this exists
───────────────
The Jetson bridge encodes H.264 with `h264_nvenc` (hardware NVENC) and serves it on
UDP :9092 (the 'MV' datagram protocol in udp_video.py) AND a WebSocket
`/camera/<name>/h264`. The simulator runs on a Mac with no NVENC, so it never
emitted H.264 — the app's H.264 decoder got nothing from the sim and fell back to
MJPEG. This module gives the sim the SAME two H.264 outputs using Apple's hardware
`h264_videotoolbox` (software `libx264` / `libopenh264` fallback), so every teleop
client (iOS, Vision Pro, web) decodes ONE consistent codec whether it is talking to
the sim or the real robot.

It mirrors `_H264Fanout` (one ffmpeg per encode, fed raw BGR on a pipe, Annex-B split
on access-unit delimiters, demand-driven so an unwatched stream costs ~0 CPU) but adds:

  • a parameterised encoder backend (VideoToolbox runs on Apple's Media Engine, NOT
    the CPU cores the physics loop needs — so the old `libopenh264` CPU problem that
    forced sensors/ros_camera_node.py to default H.264 OFF does not recur here), and
  • TWO profiles per camera, because UDP and WebSocket clients have opposite needs:
      lan → match source resolution, auto bitrate, IDR every ~1 s   → UDP :9092
      ws  → downscaled, low fixed bitrate, IDR every ~0.5 s          → WebSocket
    Browsers cannot receive UDP, so their low-latency path is H.264-over-WebSocket
    (TCP); a tighter GOP recovers faster on a lossy link.

Wire contract (mandatory — the clients reject anything else)
────────────────────────────────────────────────────────────
baseline profile + yuv420p (High-4:4:4 → iOS VideoToolbox / WebCodecs render BLACK),
Annex-B, one AUD (NAL type 9) per access unit so we can split the stream, and SPS/PPS
in-band with every IDR (`dump_extra=freq=keyframe`) so a late / lossy joiner recovers
within one GOP. Confirmed against MABEL Mobile App/.../Services/H264Decoder.swift and
website/assets/h264-video.js.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

# H.264 Annex-B access-unit delimiter (NAL type 9). We cut the byte stream on it so
# each datagram group / WS message == exactly one access unit. Same split as udp_sink.py.
_AUD3 = b"\x00\x00\x01\x09"

LAN = "lan"
WS = "ws"


def _ffmpeg_runs(path: str) -> bool:
    """True iff `path` is an ffmpeg that actually executes (the Homebrew ffmpeg on this
    Mac is dyld-broken — missing libx265 — so `which ffmpeg` is NOT enough)."""
    if not path:
        return False
    try:
        r = subprocess.run([path, "-hide_banner", "-version"],
                           capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def resolve_ffmpeg() -> str | None:
    """A working ffmpeg binary: $MABEL_SIM_FFMPEG, then a runnable `which ffmpeg`, then
    imageio_ffmpeg's bundled static binary (PyAV's sibling; verified to run + expose
    h264_videotoolbox on this machine). None if nothing executes."""
    env = os.getenv("MABEL_SIM_FFMPEG")
    if env and _ffmpeg_runs(env):
        return env
    sys_ff = shutil.which("ffmpeg")
    if sys_ff and _ffmpeg_runs(sys_ff):
        return sys_ff
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if _ffmpeg_runs(ff):
            return ff
    except Exception:  # noqa: BLE001
        pass
    return None


_PREFERRED = ("h264_videotoolbox", "libx264", "libopenh264")


def resolve_encoder(ffmpeg: str) -> str | None:
    """Pick the best available H.264 encoder: hardware VideoToolbox if present, else
    libx264, else libopenh264. Honour $MABEL_SIM_H264_ENC to force one."""
    forced = os.getenv("MABEL_SIM_H264_ENC")
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                            capture_output=True, text=True, timeout=8).stdout
    except Exception:  # noqa: BLE001
        return forced
    available = {name for name in _PREFERRED if f" {name} " in out}
    if forced:
        return forced if forced in available else (forced or None)
    for name in _PREFERRED:
        if name in available:
            return name
    return None


def build_cmd(ffmpeg: str, encoder: str, w: int, h: int, fps: int,
              bitrate: str, gop: int) -> list[str]:
    """ffmpeg arg list reading raw bgr24 on stdin → Annex-B H.264 on stdout, in the
    universally-decodable teleop format. Encoder-specific low-latency flags; the
    bitstream-filter chain (inline SPS/PPS at every IDR + inserted AUDs) is identical
    across backends so the wire bytes match the Jetson's NVENC output."""
    head = [ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0", "-an",
            "-c:v", encoder, "-pix_fmt", "yuv420p", "-profile:v", "baseline"]
    if encoder == "h264_videotoolbox":
        # Hardware (Apple Media Engine): realtime, CBR-ish, no NVENC-only flags.
        head += ["-realtime", "1", "-b:v", bitrate]
    elif encoder == "libx264":
        head += ["-preset", "ultrafast", "-tune", "zerolatency", "-b:v", bitrate]
    else:  # libopenh264 / other software
        head += ["-rc_mode", "bitrate", "-b:v", bitrate]
    tail = ["-g", str(gop),
            "-force_key_frames", f"expr:gte(n,n_forced*{gop})",
            # dump_extra → SPS/PPS in-band at each IDR; h264_metadata → guarantee an
            # AUD delimits every access unit even on encoders that won't emit one.
            "-bsf:v", "dump_extra=freq=keyframe,h264_metadata=aud=insert",
            "-f", "h264", "pipe:1"]
    return head + tail


class SimH264Fanout:
    """One ffmpeg encode per (camera, profile), fed raw BGR, output split on AUDs and
    fanned to the matching transport. Demand-driven: an encode is only spawned and fed
    while a client is actually subscribed, so idle streams cost ~0 CPU.

      profile LAN   → udp_sender.send(name, au)   (UDP :9092, match-source res)
      profile WS → on_au(name, au)             (WebSocket, downscaled, tight GOP)
    """

    def __init__(self, udp_sender, fps: int = 30, lan_bitrate: str = "auto",
                 lan_bitrate_k: float = 0.12, ws_width: int = 640,
                 ws_bitrate: str = "2M", ffmpeg: str | None = None,
                 encoder: str | None = None, log=None):
        self._udp = udp_sender                    # udp_video.UDPVideoSender on :9092
        self._fps = int(fps)
        self._lan_bitrate = lan_bitrate           # "auto" or e.g. "6M"
        self._lan_k = float(lan_bitrate_k)        # bits / (pixel·frame) for auto
        self._ws_width = int(ws_width)
        self._ws_bitrate = ws_bitrate
        self._log = log
        self._ffmpeg = ffmpeg or resolve_ffmpeg()
        self._encoder = encoder or (resolve_encoder(self._ffmpeg) if self._ffmpeg else None)
        self._enabled = bool(self._ffmpeg and self._encoder)
        self._raw: dict[str, np.ndarray] = {}     # name -> latest BGR (shared by profiles)
        self._procs: dict[tuple, subprocess.Popen] = {}   # (name, profile) -> Popen
        self._lock = threading.Lock()
        # WebSocket hook, installed by the HTTP server (mjpeg_server.set_h264_hook).
        self._on_au = None                        # (name, au) -> None
        self._ws_wants = None                     # (name) -> bool
        if log:
            if self._enabled:
                log.info("sim H.264 ON  encoder=%s  ffmpeg=%s  (LAN udp + ws)",
                         self._encoder, os.path.basename(self._ffmpeg))
            else:
                log.warning("sim H.264 OFF — no working ffmpeg/encoder found "
                            "(set MABEL_SIM_FFMPEG / MABEL_SIM_H264_ENC). Clients use MJPEG.")

    # ── WebSocket hook ────────────────────────────────────────────────────────
    def set_ws_hook(self, on_au, ws_wants) -> None:
        """Let the WebSocket H.264 endpoint consume the ws encoder's access units
        and drive its demand (same contract as udp_sink.UdpVideoSink.set_ws_hook)."""
        self._on_au = on_au
        self._ws_wants = ws_wants

    # ── demand probes (drive demand-driven render in sim_camera_server) ────
    def wants_udp(self, name: str) -> bool:
        return self._enabled and self._udp is not None and self._udp.has_subscriber(name)

    def wants_ws(self, name: str) -> bool:
        return self._enabled and self._ws_wants is not None and bool(self._ws_wants(name))

    def wants(self, name: str) -> bool:
        return self.wants_udp(name) or self.wants_ws(name)

    def _profile_wanted(self, name: str, profile: str) -> bool:
        return self.wants_udp(name) if profile == LAN else self.wants_ws(name)

    # ── frame in (render thread) ──────────────────────────────────────────
    def feed(self, name: str, bgr: np.ndarray) -> None:
        """Stash the latest BGR for `name` and lazily start whichever profile encoders
        currently have a subscriber. Cheap: a copy of the latest frame + a dict write."""
        if not self._enabled:
            return
        with self._lock:
            self._raw[name] = bgr
            for profile in (LAN, WS):
                if (name, profile) not in self._procs and self._profile_wanted(name, profile):
                    self._start(name, profile, int(bgr.shape[1]), int(bgr.shape[0]))

    def _target_size(self, profile: str, w: int, h: int) -> tuple[int, int]:
        """LAN = match source; WS = downscale so the long edge <= ws_width (never
        upscale), rounded to even dimensions (yuv420p requires even W/H)."""
        if profile == LAN or self._ws_width <= 0 or max(w, h) <= self._ws_width:
            tw, th = w, h
        elif w >= h:
            tw = self._ws_width
            th = max(2, round(h * self._ws_width / w))
        else:
            th = self._ws_width
            tw = max(2, round(w * self._ws_width / h))
        return (tw - (tw & 1), th - (th & 1)) if (tw & 1 or th & 1) else (tw, th)

    def _bitrate_for(self, profile: str, w: int, h: int) -> str:
        if profile == WS:
            return self._ws_bitrate
        if self._lan_bitrate != "auto":
            return self._lan_bitrate
        bps = int(min(max(self._lan_k * w * h * self._fps, 1.5e6), 12e6))
        return str(bps)

    def _start(self, name: str, profile: str, w: int, h: int) -> None:
        tw, th = self._target_size(profile, w, h)
        gop = self._fps if profile == LAN else max(2, self._fps // 2)
        cmd = build_cmd(self._ffmpeg, self._encoder, tw, th, self._fps,
                        self._bitrate_for(profile, tw, th), gop)
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, bufsize=0)
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log.warning("sim H.264 start failed for %s/%s (%s)", name, profile, e)
            return
        self._procs[(name, profile)] = p
        threading.Thread(target=self._feed_loop, args=(name, profile, p, (tw, th)),
                         daemon=True, name=f"h264-feed-{name}-{profile}").start()
        threading.Thread(target=self._read_loop, args=(name, profile, p),
                         daemon=True, name=f"h264-read-{name}-{profile}").start()

    def _feed_loop(self, name: str, profile: str, p, size: tuple[int, int]) -> None:
        """Write the latest BGR to ffmpeg at the encode cadence, resizing for the ws
        profile. Idles (no writes) whenever this profile has no subscriber, then exits
        and reaps the encoder so a stream nobody watches stops costing CPU."""
        period = 1.0 / self._fps
        tw, th = size
        idle = 0.0
        while p.poll() is None:
            if not self._profile_wanted(name, profile):
                idle += 0.1
                if idle >= 4.0:                       # no demand for 4 s → reap encoder
                    break
                time.sleep(0.1)
                continue
            idle = 0.0
            with self._lock:
                bgr = self._raw.get(name)
            if bgr is not None:
                if bgr.shape[1] != tw or bgr.shape[0] != th:
                    bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    p.stdin.write(np.ascontiguousarray(bgr).tobytes())
                except (BrokenPipeError, ValueError, OSError):
                    break
            time.sleep(period)
        self._reap(name, profile, p)

    def _read_loop(self, name: str, profile: str, p) -> None:
        """Split the Annex-B output on AUDs; route each access unit to this profile's
        transport (LAN→UDP, WS→WebSocket hook)."""
        buf = b""
        while p.poll() is None:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(_AUD3)
                if i < 0:
                    break
                j = buf.find(_AUD3, i + 4)
                if j < 0:
                    break
                au = buf[i:j]                         # one complete access unit
                if profile == LAN:
                    self._udp.send(name, au)
                elif self._on_au is not None:
                    try:
                        self._on_au(name, au)
                    except Exception:                 # noqa: BLE001
                        pass
                buf = buf[j:]

    def _reap(self, name: str, profile: str, p) -> None:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            if self._procs.get((name, profile)) is p:
                del self._procs[(name, profile)]

    def stop(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
            self._procs.clear()
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
