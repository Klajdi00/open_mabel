"""
One-shot camera frame proxy: pull a single JPEG out of the local MJPEG
multipart stream so the (same-origin) web app can composite the head-cam view
into canvases without CORS taint. Parser kept pure for tests.
"""
import urllib.request

from . import config as CFG

SOI = b"\xff\xd8"      # JPEG start-of-image
EOI = b"\xff\xd9"      # JPEG end-of-image


def extract_jpeg(chunks, max_bytes=4_000_000):
    """First complete JPEG from an iterable of byte chunks (or None)."""
    buf = b""
    for chunk in chunks:
        buf += chunk
        start = buf.find(SOI)
        if start >= 0:
            end = buf.find(EOI, start + 2)
            if end >= 0:
                return buf[start:end + 2]
        if len(buf) > max_bytes:
            # keep the tail so a JPEG spanning the cap is still found
            keep = buf.rfind(SOI)
            buf = buf[keep:] if keep > 0 else buf[-65536:]
    return None


def grab_frame(name, timeout=2.5):
    """One JPEG from the sim camera server's /camera/<name>/stream.mjpg."""
    url = (f"http://{CFG.UPSTREAM_HOST}:{CFG.VIDEO_PORT}"
           f"/camera/{name}/stream.mjpg")
    req = urllib.request.Request(url, headers={"User-Agent": "mabel-cc"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        def chunks():
            while True:
                b = r.read(16384)
                if not b:
                    return
                yield b
        return extract_jpeg(chunks())
