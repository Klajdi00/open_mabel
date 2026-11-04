#!/usr/bin/env python3
"""
exp_video_latency.py — per-stage latency budget of the MABEL head video pipeline.

Measures, on THIS machine, every server/network stage that a head frame passes
through, as a function of render resolution and JPEG quality, so we can pick the
sharpest configuration that still hits a target frame time. Stages:

    render   MuJoCo off-screen render() + GPU readback (the dominant server cost)
    encode   cv2.imencode('.jpg', q)        → bytes
    size     JPEG bytes  → UDP chunks (=ceil(size/1182))  → est. wire time
    decode   cv2.imdecode  (proxy for the headset's ImageIO/CGImageSource decode)

The client texture-upload + mipmap-generation stage runs on RealityKit/Metal and
cannot be timed from Python; it is discussed in the doc and bounded separately.
We also estimate the CPU cost of building the full mip chain (a stand-in upper
bound for "regenerate mips every frame") with successive INTER_AREA halvings.

Reproducible: fixed scene, fixed trial count, prints a summary table and writes
docs/figures/video_latency.pdf (vector). Run from server/:

    ~/anaconda3/bin/python3 tools/exp_video_latency.py
    ~/anaconda3/bin/python3 tools/exp_video_latency.py --trials 60 --camera zed_left
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
DEFAULT_SCENE = os.path.abspath(
    os.path.join(SERVER, "..", "simulation", "mabel_mujoco", "scenes", "mabel_home.xml"))
FIGDIR = os.path.join(SERVER, "docs", "figures")

CHUNK_PAYLOAD = 1196 - 14            # udp_video.py: 1182 B JPEG per datagram
# Resolutions to sweep (all 4:3 except the 16:9 reference), within offscreen 1920x1080.
RES = [(640, 480), (800, 600), (960, 720), (1024, 768), (1280, 720), (1280, 960)]
QUAL = [60, 75, 85, 95]


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, float), p))


def time_render(model, cam, w, h, trials, warmup=5, shadows=False):
    """Median render()+readback time (ms). Vary the scene each frame so nothing
    is cached away; readback dominates and is content-independent. Shadows OFF by
    default to match sim_camera_server.py's render path exactly."""
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=h, width=w)
    frames = []
    ts = []
    for i in range(trials + warmup):
        data.time = i * 0.01
        # a tiny pose wiggle so update_scene/readback can't be elided
        if model.nq:
            data.qpos[0] = 0.001 * (i % 7)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        r.update_scene(data, camera=cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
        t0 = time.perf_counter()
        img = r.render()
        t1 = time.perf_counter()
        if i >= warmup:
            ts.append((t1 - t0) * 1e3)
            frames.append(img)
    return ts, frames[-1]


def time_encode(bgr, q, trials):
    enc = [cv2.IMWRITE_JPEG_QUALITY, q]
    ts, sizes = [], []
    for _ in range(trials):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", bgr, enc)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3)
        sizes.append(len(buf))
    return ts, int(np.median(sizes))


def time_decode(jpeg, trials):
    arr = np.frombuffer(jpeg, np.uint8)
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        cv2.imdecode(arr, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3)
    return ts


def time_mipgen(bgr, trials):
    """CPU proxy for regenerating the full mip chain each frame (INTER_AREA
    halvings down to 1px). RealityKit does this on the GPU — this is an
    order-of-magnitude upper bound, not the real GPU number."""
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        lvl = bgr
        while min(lvl.shape[:2]) > 1:
            lvl = cv2.resize(lvl, (max(1, lvl.shape[1] // 2), max(1, lvl.shape[0] // 2)),
                             interpolation=cv2.INTER_AREA)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def concurrent_fps(model, layout, secs=4.0):
    """Replicate the server: one render thread PER camera, all rendering as fast
    as they can for `secs`. Returns achieved fps per stream. `layout` is a list
    of (name, cam, w, h) or (name, cam, w, h, cap_fps) — cap_fps throttles that
    thread so it stops stealing GPU from the others. This exposes the GPU-readback
    contention that the single-camera timings hide (the real reason the head —
    stereo, big — starves while a small wrist stays smooth)."""
    import threading
    counts = {}
    stop = threading.Event()

    def worker(name, cam, w, h, cap=0.0):
        data = mujoco.MjData(model)
        r = mujoco.Renderer(model, height=h, width=w)
        enc = [cv2.IMWRITE_JPEG_QUALITY, 85]
        interval = (1.0 / cap) if cap else 0.0
        n = 0
        i = 0
        last = 0.0
        while not stop.is_set():
            if interval:
                now = time.perf_counter()
                if now - last < interval:
                    time.sleep(0.001)
                    continue
                last = now
            i += 1
            if model.nq:
                data.qpos[0] = 0.001 * (i % 7)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            r.update_scene(data, camera=cam)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            img = r.render()
            cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), enc)
            n += 1
        counts[name] = n

    threads = [threading.Thread(target=worker, args=lay, daemon=True) for lay in layout]
    for t in threads:
        t.start()
    time.sleep(secs)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    return {k: v / secs for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--camera", default="zed_left")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--link-mbps", type=float, default=300.0,
                    help="assumed usable LAN/Wi-Fi throughput for the wire estimate")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    os.makedirs(FIGDIR, exist_ok=True)

    print(f"scene   {args.scene}")
    print(f"camera  {args.camera}   trials {args.trials}   link {args.link_mbps} Mbps")
    print(f"offscreen buffer {model.vis.global_.offwidth}x{model.vis.global_.offheight}\n")

    rows = []
    render_by_res = {}
    frame_by_res = {}
    for (w, h) in RES:
        rt, frame = time_render(model, args.camera, w, h, args.trials)
        render_by_res[(w, h)] = rt
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_by_res[(w, h)] = bgr
        for q in QUAL:
            et, size = time_encode(bgr, q, args.trials)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            dt = time_decode(buf.tobytes(), args.trials)
            chunks = max(1, (size + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD)
            wire_ms = size * 8 / (args.link_mbps * 1e6) * 1e3
            rows.append(dict(w=w, h=h, q=q,
                             render=pct(rt, 50), encode=pct(et, 50),
                             decode=pct(dt, 50), size=size, chunks=chunks,
                             wire=wire_ms))

    mip_ms = {res: pct(time_mipgen(frame_by_res[res], args.trials), 50) for res in RES}

    # ---- summary table ------------------------------------------------------
    hdr = (f"{'res':>9} {'q':>3} | {'render':>7} {'encode':>7} {'decode':>7} "
           f"{'wire':>6} | {'server':>7} {'srv_fps':>7} | {'size_kB':>7} {'chunks':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        server = r["render"] + r["encode"]      # what bounds the render-thread fps
        print(f"{r['w']}x{r['h']:>4} {r['q']:>3} | "
              f"{r['render']:7.1f} {r['encode']:7.2f} {r['decode']:7.2f} "
              f"{r['wire']:6.1f} | {server:7.1f} {1000/server:7.1f} | "
              f"{r['size']/1024:7.1f} {r['chunks']:6d}")
    print(f"\nfull mip-chain regen (CPU proxy, per frame):")
    for res in RES:
        print(f"  {res[0]}x{res[1]}: {mip_ms[res]:.2f} ms")

    # ---- figures ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    xs = [w for (w, h) in RES]
    xlab = [f"{w}x{h}" for (w, h) in RES]

    ax[0, 0].plot(xs, [pct(render_by_res[r], 50) for r in RES], "o-")
    ax[0, 0].set_title("Render + GPU readback (median)")
    ax[0, 0].set_ylabel("ms"); ax[0, 0].grid(True, alpha=.3)

    for q in QUAL:
        ax[0, 1].plot(xs, [next(r["encode"] for r in rows if r["w"] == w and r["q"] == q)
                           for (w, h) in RES], "o-", label=f"q{q}")
    ax[0, 1].set_title("JPEG encode"); ax[0, 1].set_ylabel("ms")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True, alpha=.3)

    for q in QUAL:
        ax[1, 0].plot(xs, [next(r["size"] for r in rows if r["w"] == w and r["q"] == q) / 1024
                           for (w, h) in RES], "o-", label=f"q{q}")
    ax[1, 0].set_title("JPEG size (→ UDP chunks)"); ax[1, 0].set_ylabel("kB")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, alpha=.3)

    for q in QUAL:
        ax[1, 1].plot(xs, [1000 / (next(r["render"] for r in rows if r["w"] == w and r["q"] == q)
                                   + next(r["encode"] for r in rows if r["w"] == w and r["q"] == q))
                           for (w, h) in RES], "o-", label=f"q{q}")
    ax[1, 1].set_title("Server frame rate (render+encode bound)")
    ax[1, 1].set_ylabel("fps"); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, alpha=.3)

    for a in ax.flat:
        a.set_xticks(xs); a.set_xticklabels(xlab, rotation=30, fontsize=8)
        a.set_xlabel("per-eye resolution")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "video_latency.pdf")
    fig.savefig(out)
    print(f"\nwrote {out}")

    # ---- concurrent contention (the real bottleneck) ------------------------
    print("\n=== CONCURRENT render fps (one thread/cam, like the server) ===")
    scenarios = {
        "stereo head 960 + 2 wrists 480 (current default)": [
            ("head_L", "zed_left", 960, 720), ("head_R", "zed_right", 960, 720),
            ("wrist_L", "l_wrist_cam", 480, 360), ("wrist_R", "r_wrist_cam", 480, 360)],
        "MONO head 960 + 2 wrists 480": [
            ("head", "zed_left", 960, 720),
            ("wrist_L", "l_wrist_cam", 480, 360), ("wrist_R", "r_wrist_cam", 480, 360)],
        "stereo head 640 + 2 wrists 480": [
            ("head_L", "zed_left", 640, 480), ("head_R", "zed_right", 640, 480),
            ("wrist_L", "l_wrist_cam", 480, 360), ("wrist_R", "r_wrist_cam", 480, 360)],
        "head ALONE 960 (no wrists)": [("head", "zed_left", 960, 720)],
        "stereo head ALONE 960 (no wrists)": [
            ("head_L", "zed_left", 960, 720), ("head_R", "zed_right", 960, 720)],
        "stereo head 960 + wrists CAPPED 8fps": [
            ("head_L", "zed_left", 960, 720), ("head_R", "zed_right", 960, 720),
            ("wrist_L", "l_wrist_cam", 480, 360, 8.0), ("wrist_R", "r_wrist_cam", 480, 360, 8.0)],
        "MONO head 960 + wrists CAPPED 8fps": [
            ("head", "zed_left", 960, 720),
            ("wrist_L", "l_wrist_cam", 480, 360, 8.0), ("wrist_R", "r_wrist_cam", 480, 360, 8.0)],
    }
    for label, lay in scenarios.items():
        fps = concurrent_fps(model, lay, secs=4.0)
        summary = "  ".join(f"{k}={v:.1f}" for k, v in fps.items())
        print(f"  {label}:\n      {summary}")


if __name__ == "__main__":
    main()
