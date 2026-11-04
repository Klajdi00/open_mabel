"""
config.py — Mac proxy server configuration.

Camera sources are defined as a list so you can add/remove cameras without
touching any other code. Each CameraSource maps a device (USB index or RTSP
URL) to the HTTP path the Vision Pro app will connect to.

Current layout (matches AppConfiguration.swift defaults):
  /camera/main/stream.mjpg          ← robot head camera (stereo later)
  /camera/wrist_left/stream.mjpg   ← left wrist camera
  /camera/wrist_right/stream.mjpg  ← right wrist camera

Future cameras just add more CameraSource entries.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Camera source definition
# ---------------------------------------------------------------------------

@dataclass
class CameraSource:
    """One physical camera → one MJPEG HTTP path."""
    name: str               # human label used in logs and the index page
    device: int | str       # OpenCV device index (int) or RTSP/HTTP URL (str)
    path: str               # URL path served by the MJPEG server
    width: int  = 0         # 0 = use camera default
    height: int = 0
    fps: int    = 30


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    ws_host: str  = "0.0.0.0"
    ws_port: int  = 9090
    ws_path: str  = "/teleop"
    http_host: str = "0.0.0.0"
    http_port: int = 8080


# ---------------------------------------------------------------------------
# Tracking / logging
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    log_every_n_frames: int  = 30
    verbose_matrices: bool   = False


# ---------------------------------------------------------------------------
# Video — global encode settings + per-camera sources
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    jpeg_quality: int = 80

    # -----------------------------------------------------------------------
    # Camera roster.
    # Each entry is one physical camera. Add entries here to support
    # additional cameras; no other file needs changing.
    #
    # Mac proxy defaults: all three paths served by the built-in/USB webcam.
    # On the Jetson replace device with RTSP URLs or ROS bridge endpoints.
    # -----------------------------------------------------------------------
    cameras: list[CameraSource] = field(default_factory=lambda: [
        # ── Head camera (main stereo view sent to the Vision Pro dome) ──
        # device=1 = USB camera @ 1920×1080 15fps (confirmed by check_cameras.py)
        CameraSource(
            name="Head (main)",
            device=int(os.getenv("MABEL_CAM_MAIN", "1")),
            path="/camera/main/stream.mjpg",
            fps=15,
        ),
        # ── Wrist cameras — share the head camera (device 1) until real wrist
        # cams are attached. Set MABEL_CAM_WRIST_LEFT/RIGHT env vars to override.
        CameraSource(
            name="Wrist Left",
            device=int(os.getenv("MABEL_CAM_WRIST_LEFT",  "1")),
            path="/camera/wrist_left/stream.mjpg",
            fps=15,
        ),
        CameraSource(
            name="Wrist Right",
            device=int(os.getenv("MABEL_CAM_WRIST_RIGHT", "1")),
            path="/camera/wrist_right/stream.mjpg",
            fps=15,
        ),
        # ── Uncomment when you have stereo head cameras ──────────────────
        # CameraSource(
        #     name="Head Left (stereo)",
        #     device=int(os.getenv("MABEL_CAM_HEAD_LEFT", "1")),
        #     path="/camera/head_left/stream.mjpg",
        # ),
        # CameraSource(
        #     name="Head Right (stereo)",
        #     device=int(os.getenv("MABEL_CAM_HEAD_RIGHT", "2")),
        #     path="/camera/head_right/stream.mjpg",
        # ),
    ])


# ---------------------------------------------------------------------------
# Root config object
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    network:  NetworkConfig  = field(default_factory=NetworkConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video:    VideoConfig    = field(default_factory=VideoConfig)


config = ServerConfig(
    network=NetworkConfig(
        ws_host=os.getenv("MABEL_WS_HOST",   "0.0.0.0"),
        ws_port=int(os.getenv("MABEL_WS_PORT",  "9090")),
        http_host=os.getenv("MABEL_HTTP_HOST", "0.0.0.0"),
        http_port=int(os.getenv("MABEL_HTTP_PORT", "8080")),
    ),
    tracking=TrackingConfig(
        log_every_n_frames=int(os.getenv("MABEL_LOG_EVERY", "30")),
        verbose_matrices=os.getenv("MABEL_VERBOSE", "0") == "1",
    ),
    video=VideoConfig(
        jpeg_quality=int(os.getenv("MABEL_JPEG_QUALITY", "80")),
    ),
)
