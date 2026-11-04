#!/usr/bin/env python3
"""exp_video_latency — where the Control Center's video delay actually comes from.

The studio's cameras were plain MJPEG <img> tags while the iOS app and the website
teleop page consumed the camera server's hardware-encoded H.264. This harness
measures the browser-side pipeline end-to-end, under a REAL H.264 stream paced at
real time, and A/Bs the decisions that matter:

  transport   : MJPEG <img>            (the old studio path)
              : H.264 / WebCodecs      (the new one, web/js/h264.js)
  decode      : prefer-hardware        (what the website's decoder pins)
              : prefer-software
  queueing    : queue-everything       (WebCodecs' default: never drops)
              : drop-don't-queue       (maxQueue=2, resync at the next IDR)
  streams     : 1 camera vs 4          (the studio shows a 4-tile strip)

Metric: LATENCY = bytes-arrive-in-the-browser -> pixels-on-the-canvas, per frame,
correlated through the decoder's own chunk timestamps (H264Stream._arrival). This
is the part of glass-to-glass the studio owns; capture + encode + network are
shared with iOS/web and measured in server/tools/exp_h264_video.py.

Run:
  python3 control_center/tools/exp_video_latency.py            # full sweep
  python3 control_center/tools/exp_video_latency.py --seconds 8
Outputs: a summary table + docs/figures/video_latency.pdf (vector).
"""
import argparse
import asyncio
import functools
import http.server
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # control_center/
WEB = os.path.join(ROOT, "web")
FIGS = os.path.join(ROOT, "docs", "figures")
CACHE = os.path.join(HERE, ".cache")

SEED = 7
FPS = 30
SIZE = "1280x720"        # the camera server encodes H.264 at h264_width=1280


# ── the stream under test: real H.264, the encoder's exact contract ───────────
def make_stream(seconds):
    """Baseline profile, yuv420p, Annex-B, SPS+PPS in-band at every IDR, ONE AUD
    per access unit, ONE slice per frame — the exact contract
    hardware_bridge/mabel_hw/cameras/video_server.py guarantees.

    The slice settings are not cosmetic: '-tune zerolatency' silently turns on
    x264 sliced-threads, which splits every frame into several slices. An AU
    splitter that closes at the first VCL NAL then tears frames apart and feeds
    the decoder fragments, which fails with a bogus 'EncodingError' and makes
    hardware decode look broken. Use testsrc2 (moving) so deltas carry real
    content instead of compressing to nothing.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"bench_{SIZE}_{seconds}s.h264")
    if os.path.exists(path):
        return path
    import imageio_ffmpeg                            # Homebrew ffmpeg is broken here
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ff, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={SIZE}:rate={FPS}",
        "-t", str(seconds),
        "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
        "-g", str(FPS),
        "-x264-params", "aud=1:sliced-threads=0:slices=1:bframes=0",
        "-bsf:v", "dump_extra=freq=keyframe", "-f", "h264", "-y", path,
    ], check=True)
    return path


def _nal_type(n):
    return n[4 if n[2] == 0 else 3] & 0x1f


def access_units(path):
    """Annex-B -> access units, split on the Access Unit Delimiter exactly like
    video_server.py. One AU == one WebSocket message == one frame."""
    data = open(path, "rb").read()
    offs, i = [], 0
    while i < len(data) - 3:
        if data[i] == 0 and data[i+1] == 0:
            if data[i+2] == 1:
                offs.append(i); i += 3; continue
            if data[i+2] == 0 and data[i+3] == 1:
                offs.append(i); i += 4; continue
        i += 1
    nals = [data[o:(offs[k+1] if k + 1 < len(offs) else len(data))]
            for k, o in enumerate(offs)]
    aus, cur = [], b""
    for n in nals:
        if _nal_type(n) == 9 and cur:        # AUD opens the next access unit
            aus.append(cur); cur = b""
        cur += n
    if cur:
        aus.append(cur)
    return aus


# ── fake camera server: the robot-side wire contract, paced at real time ───────
class CamServer:
    def __init__(self, aus, fps=FPS):
        self.aus, self.fps = aus, fps
        self.port = None
        self.bytes = 0            # what the transport actually costs on the wire
        self.frames = 0
        self._ready = threading.Event()

    def reset(self):
        self.bytes = self.frames = 0

    async def _handler(self, ws, path=None):
        try:
            i, t0 = 0, time.perf_counter()
            while True:
                au = self.aus[i % len(self.aus)]
                await ws.send(au)
                self.bytes += len(au)
                self.frames += 1
                i += 1
                # pace to real time; never burst (a burst would hide queueing)
                await asyncio.sleep(max(0.0, t0 + i / self.fps - time.perf_counter()))
        except Exception:
            pass

    def start(self):
        import websockets

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def main():
                async with websockets.serve(self._handler, "127.0.0.1", 0) as s:
                    self.port = s.sockets[0].getsockname()[1]
                    self._ready.set()
                    await asyncio.Future()
            loop.run_until_complete(main())

        threading.Thread(target=run, daemon=True).start()
        assert self._ready.wait(5)
        return self.port


# ── MJPEG server (the OLD studio path) — same frames, JPEG per frame ──────────
class MjpegServer:
    """multipart/x-mixed-replace, exactly like the camera server's /stream.mjpg."""

    def __init__(self, jpegs, fps=FPS):
        self.jpegs, self.fps = jpegs, fps
        self.bytes = 0
        self.frames = 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                i, t0 = 0, time.perf_counter()
                try:
                    while True:
                        j = outer.jpegs[i % len(outer.jpegs)]
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(j)).encode() +
                                         b"\r\n\r\n" + j + b"\r\n")
                        outer.bytes += len(j)
                        outer.frames += 1
                        i += 1
                        time.sleep(max(0.0, t0 + i / outer.fps - time.perf_counter()))
                except Exception:
                    pass

        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self._srv.server_address[1]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def reset(self):
        self.bytes = self.frames = 0


def make_jpegs(seconds):
    """Same source content as the H.264 stream, as individual JPEGs (what MJPEG
    ships every single frame — this is the bandwidth the studio used to pay)."""
    os.makedirs(CACHE, exist_ok=True)
    d = os.path.join(CACHE, f"jpg_{SIZE}")
    if not os.path.isdir(d):
        os.makedirs(d)
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ff, "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={SIZE}:rate={FPS}",
            "-t", "2", "-q:v", "4", os.path.join(d, "f_%03d.jpg"),
        ], check=True)
    return [open(os.path.join(d, f), "rb").read()
            for f in sorted(os.listdir(d)) if f.endswith(".jpg")]


# ── the browser: run the studio's REAL modules and read their own stats ───────
def serve_web():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


H264_PROBE = """
async ({ port, n, maxQueue, hw, seconds }) => {
  const { H264Stream } = await import('/js/h264.js');
  try { sessionStorage.removeItem('mabel.h264.hw'); } catch {}
  const streams = [];
  for (let i = 0; i < n; i++) {
    const c = document.createElement('canvas');
    document.body.appendChild(c);
    const s = new H264Stream({
      wsUrl: `ws://127.0.0.1:${port}/camera/cam${i}/h264`,
      canvas: c,
      maxQueue: maxQueue === null ? Infinity : maxQueue,
    });
    if (hw) s._hwIdx = H264Stream.HW_LADDER.indexOf(hw);   // pin the backend
    s.start();
    streams.push(s);
  }
  await new Promise(r => setTimeout(r, seconds * 1000));
  const st = streams.map(s => s.stats());
  streams.forEach(s => s.stop());
  return st;
}
"""

MJPEG_PROBE = """
async ({ port, n, seconds }) => {
  // The OLD studio path: a native MJPEG <img>. NOTE we deliberately do NOT report
  // a paint latency here — a native <img> on multipart/x-mixed-replace exposes no
  // per-frame hook (its 'load' event does not fire per part), so any number we
  // invented would be fiction. What MJPEG actually costs is BANDWIDTH, and that is
  // measured on the server side; we just hold the stream open for the run.
  const imgs = [];
  for (let i = 0; i < n; i++) {
    const img = document.createElement('img');
    document.body.appendChild(img);
    img.src = `http://127.0.0.1:${port}/camera/cam${i}/stream.mjpg?r=` + Math.random();
    imgs.push(img);
  }
  await new Promise(r => setTimeout(r, seconds * 1000));
  imgs.forEach(i => i.removeAttribute('src'));
  return [];
}
"""


def run_cases(seconds, cases):
    global H264_SRV, MJPEG_SRV
    from playwright.sync_api import sync_playwright

    aus = access_units(make_stream(max(4, int(seconds))))
    jpegs = make_jpegs(seconds)
    H264_SRV = CamServer(aus)
    h264_port = H264_SRV.start()
    MJPEG_SRV = MjpegServer(jpegs)
    mjpeg_port = MJPEG_SRV.port
    web_port = serve_web()

    results = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page()
        pg.goto(f"http://127.0.0.1:{web_port}/index.html", wait_until="domcontentloaded")
        for c in cases:
            if c["transport"] == "mjpeg":
                MJPEG_SRV.reset()
                pg.evaluate(MJPEG_PROBE, {"port": mjpeg_port, "n": c["n"],
                                          "seconds": seconds})
                results.append({
                    **c,
                    "fps": MJPEG_SRV.frames / c["n"] / seconds,
                    "mbps": MJPEG_SRV.bytes * 8 / seconds / 1e6,
                    "latP50": None, "latP95": None,     # no per-frame hook on an <img>
                    "dropped": 0, "queuePeak": 0, "backend": "img",
                })
            else:
                H264_SRV.reset()
                st = pg.evaluate(H264_PROBE, {
                    "port": h264_port, "n": c["n"], "maxQueue": c["maxQueue"],
                    "hw": c.get("hw"), "seconds": seconds})
                results.append({
                    **c,
                    "fps": sum(s["frames"] for s in st) / c["n"] / seconds,
                    "mbps": H264_SRV.bytes * 8 / seconds / 1e6,
                    "latP50": statistics.mean(s["latP50"] for s in st),
                    "latP95": statistics.mean(s["latP95"] for s in st),
                    "latMax": max(s["latMax"] for s in st),
                    "dropped": sum(s["dropped"] for s in st),
                    "queuePeak": max(s["queuePeak"] for s in st),
                    "backend": st[0]["backend"],
                })
            r = results[-1]
            lat = "   n/a" if r["latP50"] is None else f"{r['latP50']:6.1f}"
            print(f"  ran {c['label']:34} -> {r['fps']:5.1f} fps  "
                  f"p50 {lat} ms  {r['mbps']:6.1f} Mbps")
        b.close()
    return results


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGS, exist_ok=True)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.4))
    labels = [r["label"] for r in rows]
    y = list(range(len(rows)))
    colors = ["#8a8a8e" if r["transport"] == "mjpeg" else
              ("#d97706" if r["maxQueue"] is None else "#2563eb") for r in rows]

    # bandwidth first: it is the number that saturates a weak link
    ax0.barh(y, [r["mbps"] for r in rows], color=colors)
    ax0.set_yticks(y); ax0.set_yticklabels(labels, fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xscale("log")
    ax0.set_xlabel("wire bandwidth, Mbps (log)")
    ax0.set_title("Cost on the wire", fontsize=10)
    ax0.grid(axis="x", alpha=0.3)
    for i, r in enumerate(rows):
        ax0.text(r["mbps"], i, f" {r['mbps']:.0f}", va="center", fontsize=7)

    lat = [(r["latP50"] or 0) for r in rows]
    err = [max(0, (r["latP95"] or 0) - (r["latP50"] or 0)) for r in rows]
    ax1.barh(y, lat, color=colors, xerr=[[0] * len(rows), err], capsize=3)
    ax1.set_yticks(y); ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel("arrival $\\rightarrow$ painted, ms  (bar p50, whisker p95)")
    ax1.set_title("In-browser pipeline latency", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    for i, r in enumerate(rows):
        if r["latP50"] is None:
            ax1.text(0.02, i, " n/a (native <img>: no per-frame hook)",
                     va="center", fontsize=6.5, color="#555")

    ax2.barh(y, [r["fps"] for r in rows], color=colors)
    ax2.axvline(FPS, ls="--", c="k", lw=0.8, label=f"source {FPS} fps")
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("delivered fps")
    ax2.set_title("Throughput", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"MABEL Control Center — camera transport ({SIZE} @ {FPS} fps, "
                 f"Chrome/macOS)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGS, "video_latency.pdf")
    fig.savefig(out)
    print(f"\nfigure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()

    cases = [
        dict(label="MJPEG <img>  (old studio)", transport="mjpeg", n=1, maxQueue=0),
        dict(label="H.264 hw-pinned (website cfg)", transport="h264", n=1,
             maxQueue=2, hw="prefer-hardware"),
        dict(label="H.264 queue-everything", transport="h264", n=1, maxQueue=None),
        dict(label="H.264 drop-don't-queue", transport="h264", n=1, maxQueue=2),
        dict(label="MJPEG <img> x4 (strip)", transport="mjpeg", n=4, maxQueue=0),
        dict(label="H.264 queue-everything x4", transport="h264", n=4, maxQueue=None),
        dict(label="H.264 drop-don't-queue x4", transport="h264", n=4, maxQueue=2),
    ]
    print(f"video-latency sweep · {SIZE}@{FPS} · {a.seconds}s/case\n")
    rows = run_cases(a.seconds, cases)

    print("\n" + "=" * 104)
    print(f"{'case':34} {'fps':>6} {'Mbps':>7} {'p50 ms':>8} {'p95 ms':>8} "
          f"{'drop':>6} {'qpeak':>6}  backend")
    print("-" * 104)
    for r in rows:
        p50 = "    n/a" if r["latP50"] is None else f"{r['latP50']:8.1f}"
        p95 = "    n/a" if r["latP95"] is None else f"{r['latP95']:8.1f}"
        print(f"{r['label']:34} {r['fps']:6.1f} {r['mbps']:7.1f} {p50} {p95} "
              f"{r['dropped']:6d} {r['queuePeak']:6d}  {r['backend']}")
    print("=" * 104)
    print("p50/p95 = bytes-arrive -> pixels-on-canvas, in the browser. MJPEG has no\n"
          "per-frame hook on a native <img>, so it reports n/a rather than a fiction;\n"
          "its cost is the Mbps column — that is what saturates a weak link.")

    with open(os.path.join(CACHE, "video_latency.json"), "w") as f:
        json.dump(rows, f, indent=2)
    figure(rows)


if __name__ == "__main__":
    sys.exit(main())
