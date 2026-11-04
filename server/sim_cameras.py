"""
sim_cameras.py — the MuJoCo camera roster shared by the teleop bridge and the
stand-alone camera server.

Each entry maps a MuJoCo camera (defined in models/mabel_full.xml) to the MJPEG
HTTP path the Vision Pro app pulls. Multiple paths may map to one camera (e.g.
the mono `main` fallback reuses zed_left); that camera is rendered once and the
JPEG is pushed to every path that uses it. Keep the paths in sync with the Swift
CameraID paths in AppConfiguration.swift / MabelConfig.plist.

Why `weight`: on macOS, MuJoCo's off-screen `render()` is a FIXED per-frame
stall *regardless of resolution* (Apple's deprecated-OpenGL path) — ~41 ms with
shadows off, ~73 ms with them on (the shadow pass alone costs ~32 ms). The camera
server disables shadows by default, so the machine produces ~24 camera frames/s
total. Rendering every camera equally would split that into a ~8 fps slideshow
each; instead each camera gets a `weight` = its share of those render slots. The
head (the operator's primary view) gets the largest share (weight 4 → ~12 fps)
while the two wrists (weight 2 each → ~6 fps) stay responsive rather than the old
laggy ~1.4 fps. The unused second stereo eye and duplicate head paths get weight
0 (never rendered) because the head panel is a single mono WKWebView. Turn the
per-eye stereo head on with `sim_camera_server.py --stereo`, shadows back on with
`--shadows`.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

# Head render size. Render cost is resolution-INDEPENDENT (GPU readback
# dominates), so a larger buffer costs the same to render and only adds a little
# JPEG/network — worth it on a LAN for a sharper head view. Drop --head-res for a
# lossy/remote link (smaller frames = fewer UDP chunks to lose).
HEAD_W, HEAD_H = 960, 720   # per-eye, 4:3. This is the SHARP-FAST sweet spot: the macOS
                            # deprecated-OpenGL readback has a cliff just above 960 wide
                            # (measured tools/exp_video_latency.py: 960x720 ~55ms/18fps vs
                            # 1024x768 ~128ms/7.6fps — a 2.5x jump for 14% more pixels). Stay
                            # at/under 960 wide or the head stream goes laggy. Looks sharp now
                            # that CurvedStereoPanel samples trilinear (no more stale-mip shimmer).
WRIST_W, WRIST_H = 480, 360  # 4:3, doubled from 320x240 for finer manipulation detail (tiny → fast)


class Cam(NamedTuple):
    name: str                          # MuJoCo camera name (must exist in scene)
    path: str                          # served MJPEG path the Vision Pro pulls
    label: str                         # human label for logs / index page
    res: Optional[Tuple[int, int]]     # downscale before JPEG; None = native head size
    weight: int                        # share of render slots; 0 = disabled (never rendered)


CAMERAS = [
    # Render-slot shares of the ~24 frames/s budget (shadows off; see
    # sim_camera_server). The head is still the priority and gets the largest
    # share — but the wrists are no longer starved: at weight 4/2/2 the head runs
    # ~12 fps and each wrist ~6 fps (vs the old 8/1/1 → head ~11, wrists a laggy
    # ~1.4 fps). Tune with `--head-weight`, or `--head-only` for max head fps.
    Cam("zed_left",    "/camera/main/stream.mjpg",        "Head (mono)", None,               4),
    Cam("l_wrist_cam", "/camera/wrist_left/stream.mjpg",  "Wrist Left",  (WRIST_W, WRIST_H), 2),
    Cam("r_wrist_cam", "/camera/wrist_right/stream.mjpg", "Wrist Right", (WRIST_W, WRIST_H), 2),
    # Front base camera (the sim twin of the real robot's D435): SAME path as
    # the real camera server so the web/iOS camera roster is identical sim↔real.
    Cam("body_front_cam", "/camera/front/stream.mjpg",    "Front Base",  (WRIST_W, WRIST_H), 2),
    # Per-eye stereo head — off by default (mono WKWebView panel). --stereo enables
    # full-rate; head_left still RECEIVES the zed_left frames (path alias), so
    # probes of the honest head stream work in mono too.
    Cam("zed_left",    "/camera/head_left/stream.mjpg",   "Head Left",   None,               0),
    Cam("zed_right",   "/camera/head_right/stream.mjpg",  "Head Right",  None,               0),
]


def camera_plan(cameras=CAMERAS, *, head_weight=None, head_only=False, stereo=False):
    """Collapse the path-level roster into a per-MuJoCo-camera render plan.

    Returns (weights, paths, res):
      weights[cam] -> int  render-slot share (only cams with weight > 0 are rendered)
      paths[cam]   -> list[str]  every MJPEG path fed by that camera
      res[cam]     -> (w, h) | None  downscale applied before JPEG
    Each unique camera is rendered once per slot and pushed to all its paths.
    """
    weights: dict[str, int] = {}
    paths: dict[str, list[str]] = {}
    res: dict[str, Optional[Tuple[int, int]]] = {}
    for c in cameras:
        paths.setdefault(c.name, []).append(c.path)
        res[c.name] = c.res
        weights[c.name] = max(weights.get(c.name, 0), c.weight)

    if stereo:                                   # render both eyes at the head's rate
        eye = weights.get("zed_left", 4) or 4
        weights["zed_left"] = eye
        weights["zed_right"] = eye
    if head_weight is not None:
        weights["zed_left"] = head_weight
        if "zed_right" in weights and weights["zed_right"] > 0:
            weights["zed_right"] = head_weight
    if head_only:
        for cam in ("l_wrist_cam", "r_wrist_cam", "body_front_cam"):
            weights[cam] = 0
    return weights, paths, res


def weighted_schedule(weights: dict[str, int]) -> list[str]:
    """Smooth weighted round-robin (Nginx-style) over cameras with weight > 0.

    Spreads each camera's slots evenly instead of clumping them, so a head of
    weight 4 and two wrists of weight 1 yields e.g. head,wristL,head,head,wristR,head
    rather than head,head,head,head,wristL,wristR. Returns one full period.
    """
    items = [(cam, w) for cam, w in weights.items() if w > 0]
    total = sum(w for _, w in items)
    if total == 0:
        return []
    credit = {cam: 0 for cam, _ in items}
    sched: list[str] = []
    for _ in range(total):
        for cam, w in items:
            credit[cam] += w
        pick = max(items, key=lambda iw: credit[iw[0]])[0]
        credit[pick] -= total
        sched.append(pick)
    return sched
