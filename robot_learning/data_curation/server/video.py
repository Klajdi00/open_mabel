"""
Per-camera video: the stored JPEG blobs are served frame-by-frame (the primary,
frame-accurate scrub path — output frame index == timeline frame index, so the
3D rig, plots and video stay in lockstep with one integer), with an on-demand,
cached H.264 MP4 for smooth playback / download, plus depth-frame PNGs.

MP4 transcode uses the ffmpeg BINARY (not in-process PyAV) to dodge the av<->cv2
libav dylib clash, and caches per (episode mtime) under the studio workspace.
"""
import os
import subprocess

import numpy as np

from . import config as CFG, h5pool

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _cam_key(name, kind="rgb"):
    return f"obs/cam/{name}/{kind}"


def frame_jpeg(path, name, i):
    """Return raw JPEG bytes for one frame (the stored blob, zero re-encode)."""
    key = _cam_key(name, "rgb")
    with h5pool.reading(path) as f:
        if key not in f:
            return None
        n = f[key].shape[0]
        i = max(0, min(int(i), n - 1))
        return np.asarray(f[key][i], dtype=np.uint8).tobytes()


def cameras(path):
    import json
    with h5pool.reading(path) as f:
        schema = json.loads(f.attrs["schema_json"]) if "schema_json" in f.attrs else {}
        cams = []
        for c in schema.get("cameras", []):
            k = _cam_key(c["name"], "rgb")
            n = f[k].shape[0] if k in f else 0
            cams.append({"name": c["name"], "rgb": bool(c.get("rgb")),
                         "depth": bool(c.get("depth")), "w": c.get("w"),
                         "h": c.get("h"), "n_frames": int(n)})
        return cams


def depth_png(path, name, i, colorize=False, depth_scale=1000.0, max_m=8.0):
    """16-bit grayscale PNG (raw mm) or an 8-bit turbo preview of the depth frame."""
    if cv2 is None:
        return None
    key = _cam_key(name, "depth")
    with h5pool.reading(path) as f:
        if key not in f:
            return None
        n = f[key].shape[0]
        i = max(0, min(int(i), n - 1))
        d = np.asarray(f[key][i], dtype=np.uint16)
    if colorize:
        meters = d.astype(np.float32) / depth_scale
        norm = np.clip(meters / max_m, 0, 1)
        img8 = (norm * 255).astype(np.uint8)
        col = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
        col[d == 0] = (0, 0, 0)
        ok, buf = cv2.imencode(".png", col)
    else:
        ok, buf = cv2.imencode(".png", d)
    return buf.tobytes() if ok else None


def _cache_mp4_path(dataset, ep_file, name):
    sub = os.path.join(CFG.CACHE_DIR, dataset, ep_file)
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{name}.mp4")


def ensure_mp4(path, dataset, ep_file, name, fps=15.0):
    """Transcode the JPEG-blob sequence to a cached H.264 MP4 (faststart) and
    return its path. Re-transcodes if the source episode is newer than the cache."""
    out = _cache_mp4_path(dataset, ep_file, name)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    key = _cam_key(name, "rgb")
    with h5pool.reading(path) as f:
        if key not in f:
            return None
        blobs = [np.asarray(f[key][i], dtype=np.uint8).tobytes()
                 for i in range(f[key].shape[0])]
    tmp = out + ".tmp.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", tmp],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for b in blobs:
            proc.stdin.write(b)
        proc.stdin.close()
        if proc.wait(timeout=120) != 0 or not os.path.exists(tmp):
            return None
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return None
    os.replace(tmp, out)
    return out
