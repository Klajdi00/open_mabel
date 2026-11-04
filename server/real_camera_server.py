#!/usr/bin/env python3
"""
real_camera_server.py — serve MABEL's REAL USB cameras as MJPEG, like the robot.

The hardware analog of sim_camera_server.py: instead of rendering MuJoCo cameras,
it reads the operator's actual USB cameras on the Mac and serves them on the SAME
MJPEG paths every client already pulls, so the Vision Pro / iOS / web apps need no
change — they just see real pixels:

    /camera/head_left/stream.mjpg    ← 3D USB Camera, LEFT half  (rectified)
    /camera/head_right/stream.mjpg   ← 3D USB Camera, RIGHT half (rectified)
    /camera/main/stream.mjpg         ← 3D USB Camera, LEFT (mono flat-panel fallback)
    /camera/wrist_left/stream.mjpg   ┐ Global Shutter Camera (one physical cam fills
    /camera/wrist_right/stream.mjpg  ┘ both wrist tiles until a 2nd wrist cam exists)

The head cam is a single hardware-stitched side-by-side stereo frame; we split it
down the middle and stereo-rectify each eye with the workspace calibration
(ros/.../config/*.yaml) so the headset gets a comfortable, row-aligned image.

Run (Mac, alongside the teleop bridge which owns :9090):

    # terminal 1 — teleop only, no in-process rendering:
    python sim_teleop_bridge.py --mock          # or a live headset session
    # terminal 2 — the REAL cameras (this process, owns :8080):
    python real_camera_server.py

    # verify the split + rectification without a headset (writes /tmp/realcam_*.jpg):
    python real_camera_server.py --selftest

Auto-detects the head (the stitched ≥3:1 wide device) and wrist (a global-shutter
cam) by resolution signature; override with --head-device / --wrist-device or the
env MABEL_CAM_HEAD / MABEL_CAM_WRIST.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time

import cv2

from config import CameraSource
from mjpeg_server import MJPEGServer, PushCapture
from dedicated_video import DedicatedVideoPlane
import data_streams
import real_cameras as rc

try:
    import console as con
    log = con.setup("real-cam")
    LAN_IP = con.lan_ip
except Exception:                       # console is optional
    import logging
    logging.basicConfig(level=logging.INFO, format="[real-cam] %(message)s")
    log = logging.getLogger("real-cam")
    def LAN_IP():                       # noqa: E306
        return "0.0.0.0"


def _start_http_thread(host, port, path_caps, ready, holder):
    """aiohttp MJPEG server on its own loop (mirrors sim_camera_server.py)."""
    from aiohttp import web

    async def _serve():
        server = MJPEGServer.from_captures(path_caps)
        console = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "website"))
        if server.add_console(console):
            log.info("web console  http://%s:%d/console", LAN_IP(), port)
        runner = web.AppRunner(server.app())
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        loop = asyncio.get_running_loop()
        server.start_captures(loop)
        holder["loop"] = loop
        log.info("real cameras up   http://%s:%d", LAN_IP(), port)
        for path, cap in path_caps.items():
            log.info("  %-32s %s", path, cap.source.name)
        ready.set()
        try:
            await asyncio.Future()
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_serve())
    except OSError as e:
        if getattr(e, "errno", None) in (48, 98):
            log.error("Port %d already in use — free it: lsof -ti :%d | xargs kill -9",
                      port, port)
        else:
            log.error("HTTP thread error: %s", e)
        holder["failed"] = True
        ready.set()


def build_cameras(args, devs: rc.Devices):
    """Build the head + wrist capture objects and the path→PushCapture map, and
    wire each camera's JPEG sinks to the matching PushCaptures."""
    path_caps: dict[str, PushCapture] = {}

    def cap_for(path, label):
        pc = PushCapture(CameraSource(name=label, device="usb", path=path))
        path_caps[path] = pc
        return pc

    cameras = []

    # ── Head (stitched stereo) → head_left / head_right / main ──
    if devs.head is not None and not args.no_head:
        sw, sh = (int(v) for v in args.head_mode.lower().split("x"))
        ow, oh = (0, 0) if args.head_out.lower() in ("native", "0", "0x0") else \
                 (int(v) for v in args.head_out.lower().split("x"))
        head = rc.StereoHeadCamera(
            devs.head, stitched_w=sw, stitched_h=sh, out_w=ow, out_h=oh,
            rectify=not args.no_rectify,
            fps=args.fps, jpeg_quality=args.jpeg_quality, flip180=args.flip_head)
        head.set_sink("left",  cap_for(rc.HEAD_LEFT_PATH,  "Head Left (real)").push)
        head.set_sink("right", cap_for(rc.HEAD_RIGHT_PATH, "Head Right (real)").push)
        head.set_sink("main",  cap_for(rc.HEAD_MAIN_PATH,  "Head Mono (real)").push)
        head.set_sink("stereo", cap_for(rc.HEAD_STEREO_PATH, "Head Stereo L|R (real, synced)").push)
        cameras.append(head)
    else:
        log.warning("head camera disabled or not found")

    # ── Wrists (mono) → each its OWN physical camera + path ──────────────────
    # left_wrist_cam.yaml isn't calibrated yet, so the left wrist falls back to
    # the right's calibration (identical camera model) — a close approximation.
    def calib_for(filename):
        return filename if os.path.exists(os.path.join(rc.CALIB_DIR, filename)) else "right_wrist_cam.yaml"

    if not args.no_wrist:
        if devs.wrist_right is not None:
            wr = rc.MonoCamera(
                devs.wrist_right, "right_wrist_cam.yaml", undistort=not args.no_undistort,
                fps=args.fps, jpeg_quality=args.jpeg_quality, flip180=args.flip_wrist,
                sink_keys=("w",), label="wrist-R")
            wr.set_sink("w", cap_for(rc.WRIST_RIGHT_PATH, "Wrist Right (real)").push)
            cameras.append(wr)
        else:
            log.warning("right wrist camera not found")

        if devs.wrist_left is not None and devs.wrist_left != devs.wrist_right:
            wl = rc.MonoCamera(
                devs.wrist_left, calib_for("left_wrist_cam.yaml"), undistort=not args.no_undistort,
                fps=args.fps, jpeg_quality=args.jpeg_quality, flip180=args.flip_wrist,
                sink_keys=("w",), label="wrist-L")
            wl.set_sink("w", cap_for(rc.WRIST_LEFT_PATH, "Wrist Left (real)").push)
            cameras.append(wl)
        else:
            # Only one wrist cam present — feed both tiles from it so neither is blank.
            log.warning("left wrist camera not found — mirroring the right wrist into "
                        "the left tile (attach a 2nd global-shutter cam for true L/R)")
            if devs.wrist_right is not None:
                cameras[-1].set_sink("w2", cap_for(rc.WRIST_LEFT_PATH, "Wrist Left (mirror)").push)
                cameras[-1].sink_keys = ("w", "w2")

    return cameras, path_caps


def selftest(args, devs: rc.Devices) -> int:
    """Grab one frame from each camera, run the real split/rectify path, and save
    debug JPEGs to /tmp so the operator can eyeball the result without a headset."""
    out = "/tmp"
    ok = True
    if devs.head is not None:
        sw, sh = (int(v) for v in args.head_mode.lower().split("x"))
        head = rc.StereoHeadCamera(devs.head, stitched_w=sw, stitched_h=sh,
                                   rectify=not args.no_rectify)
        cap = cv2.VideoCapture(devs.head)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, sw); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sh)
        got = None
        for _ in range(10):
            r, f = cap.read()
            if r and f is not None:
                got = f
        cap.release()
        if got is not None:
            cv2.imwrite(f"{out}/realcam_head_stitched.jpg", got)
            cv2.imwrite(f"{out}/realcam_head_left.jpg",  head._eye(got, "left"))
            cv2.imwrite(f"{out}/realcam_head_right.jpg", head._eye(got, "right"))
            log.info("head  %dx%d → %s/realcam_head_{stitched,left,right}.jpg",
                     got.shape[1], got.shape[0], out)
        else:
            log.error("head camera (device %s) gave no frame", devs.head); ok = False
    for side, dev, calib in (("right", devs.wrist_right, "right_wrist_cam.yaml"),
                             ("left",  devs.wrist_left,  "left_wrist_cam.yaml")):
        if dev is None:
            continue
        cf = calib if os.path.exists(os.path.join(rc.CALIB_DIR, calib)) else "right_wrist_cam.yaml"
        wrist = rc.MonoCamera(dev, cf, undistort=not args.no_undistort, sink_keys=())
        cap = cv2.VideoCapture(dev)
        got = None
        for _ in range(10):
            r, f = cap.read()
            if r and f is not None:
                got = f
        cap.release()
        if got is not None:
            und = got
            if (got.shape[1], got.shape[0]) != (wrist.out_w, wrist.out_h):
                und = cv2.resize(got, (wrist.out_w, wrist.out_h))
            if wrist._maps is not None:
                und = cv2.remap(und, wrist._maps[0], wrist._maps[1], cv2.INTER_LINEAR)
            cv2.imwrite(f"{out}/realcam_wrist_{side}_raw.jpg", got)
            cv2.imwrite(f"{out}/realcam_wrist_{side}.jpg", und)
            log.info("wrist-%s device=%s %dx%d → %s/realcam_wrist_%s{_raw,}.jpg",
                     side, dev, got.shape[1], got.shape[0], out, side)
        else:
            log.error("wrist-%s camera (device %s) gave no frame", side, dev); ok = False
    log.info("selftest %s", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="MABEL real USB MJPEG camera server")
    ap.add_argument("--head-device", default=os.getenv("MABEL_CAM_HEAD", "auto"),
                    help="OpenCV index of the 3D stereo head cam (default: auto-detect)")
    ap.add_argument("--wrist-right-device", default=os.getenv("MABEL_CAM_WRIST_RIGHT", "auto"),
                    help="OpenCV index of the RIGHT wrist global-shutter cam (default: auto-detect)")
    ap.add_argument("--wrist-left-device", default=os.getenv("MABEL_CAM_WRIST_LEFT", "auto"),
                    help="OpenCV index of the LEFT wrist global-shutter cam (default: auto-detect)")
    ap.add_argument("--swap-wrists", action="store_true",
                    help="swap which global-shutter cam is left vs right")
    ap.add_argument("--head-mode", default="3200x1200",
                    help="stitched stereo capture WxH; halves give the per-eye size. "
                         "Default 3200x1200 → 1600x1200 eyes: the camera's max 4:3 mode, "
                         "which MATCHES the calibration aspect so geometry/scale is correct "
                         "(16:9 modes like 2560x720 crop the vertical FOV). 1600x600 / "
                         "1280x480 are lighter 4:3 options.")
    ap.add_argument("--head-out", default="960x720",
                    help="per-eye STREAMED size (4:3). We capture+rectify at --head-mode "
                         "then area-average down to this (supersampling = band-limited, no "
                         "aliasing in the source); it is moiré-free now that the panel "
                         "samples trilinear. 960x720 is the low-latency default. Real cams "
                         "have NO MuJoCo render cliff (that only hits the sim), so 1280x960 "
                         "is a safe sharper option here, costing only a little more decode + "
                         "a few more UDP chunks. 'native' streams full --head-mode; drop "
                         "smaller for a lossy/remote link.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--no-rectify", action="store_true",
                    help="raw split, skip head stereo rectification (debug)")
    ap.add_argument("--no-undistort", action="store_true",
                    help="skip wrist undistortion (debug)")
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--flip-head", action="store_true", help="rotate head 180° (upside-down mount)")
    ap.add_argument("--flip-wrist", action="store_true", help="rotate wrist 180°")
    ap.add_argument("--udp-video-port", type=int,
                    default=int(os.getenv("MABEL_UDP_VIDEO_PORT", "9091")),
                    help="UDP latest-frame video tee (0 disables; MJPEG stays as fallback)")
    ap.add_argument("--http-host", default=os.getenv("MABEL_HTTP_HOST", "0.0.0.0"))
    ap.add_argument("--http-port", type=int, default=int(os.getenv("MABEL_HTTP_PORT", "8080")))
    ap.add_argument("--selftest", action="store_true",
                    help="grab one frame per camera, save split/rectified JPEGs to /tmp, exit")
    args = ap.parse_args()

    devs = rc.resolve_devices(args.head_device, args.wrist_right_device,
                              args.wrist_left_device, swap_wrists=args.swap_wrists)
    log.info("devices  head=%s  wrist_right=%s  wrist_left=%s  (calib dir: %s)",
             devs.head, devs.wrist_right, devs.wrist_left, rc.CALIB_DIR)
    fov = rc.head_fov()
    if fov:
        log.info("head FOV  %.1f° V × %.1f° H  (aspect %.3f) — the Vision Pro panel must "
                 "subtend this for real-life scale; pass to the bridge as "
                 "MABEL_HEAD_FOV=%.2f MABEL_HEAD_ASPECT=%.4f",
                 fov[0], fov[2], fov[1], fov[0], fov[1])
    if devs.head is None and devs.wrist_right is None and devs.wrist_left is None:
        sys.exit("[real-cam] no cameras found — check USB + macOS camera permission "
                 "(System Settings → Privacy & Security → Camera → Terminal).")

    if args.selftest:
        sys.exit(selftest(args, devs))

    cameras, path_caps = build_cameras(args, devs)
    if not path_caps:
        sys.exit("[real-cam] nothing to serve.")

    # 962x dedicated-port video plane (docs/data_streams.md). Head :9620 = the RAW
    # hardware-stitched side-by-side frame at native res (the only hi-res
    # stream); wrists tee their standard JPEGs; front binds as a 503 notice
    # (no front camera on the Mac --real rig).
    dedi = None
    if data_streams.streams_enabled():
        dedi = DedicatedVideoPlane({
            "head_sbs": devs.head is not None and not args.no_head,
            "head_sbs_reason": "head camera not found on this rig",
            "wrist_left": devs.wrist_left is not None or devs.wrist_right is not None,
            "wrist_left_reason": "no wrist camera found",
            "wrist_right": devs.wrist_right is not None,
            "wrist_right_reason": "right wrist camera not found",
            "front": False,
            "front_reason": "no front base camera on the Mac --real rig",
        }, log, host=args.http_host)
        if not dedi.start():
            dedi = None
    if dedi is not None:
        for cam in cameras:
            if isinstance(cam, rc.StereoHeadCamera):
                cam.set_sink("sbs", lambda j, d=dedi: d.push("head_sbs", j))
                cam.wants_sbs = (lambda d=dedi: d.wants("head_sbs"))

    # Low-latency UDP video tee (same as the sim path) — MJPEG stays as fallback.
    if args.udp_video_port:
        try:
            from udp_video import UDPVideoSender
            udp = UDPVideoSender(port=args.udp_video_port)
            udp.start(); udp.attach(path_caps)
            log.info("UDP video tee on :%d", args.udp_video_port)
        except OSError as e:
            log.warning("UDP video disabled — port %d unavailable (%s)",
                        args.udp_video_port, e)

    # Dedicated wrist tees (:9621/:9622): chain onto each capture's on_push,
    # PRESERVING the UDP tee installed just above (udp.attach SETS on_push, so
    # this wrapper must come after it). Demand-gated — no client on the
    # dedicated port, no extra work.
    if dedi is not None:
        def _chain_tee(cap, key):
            prev = cap.on_push
            def tee(jpeg):
                if prev is not None:
                    prev(jpeg)
                if dedi.wants(key):
                    dedi.push(key, jpeg)
            cap.on_push = tee
        for _path, _key in ((rc.WRIST_LEFT_PATH, "wrist_left"),
                            (rc.WRIST_RIGHT_PATH, "wrist_right")):
            if _path in path_caps:
                _chain_tee(path_caps[_path], _key)

    ready = threading.Event()
    holder: dict = {}
    threading.Thread(target=_start_http_thread,
                     args=(args.http_host, args.http_port, path_caps, ready, holder),
                     daemon=True).start()
    ready.wait(timeout=10.0)
    if holder.get("failed") or "loop" not in holder:
        sys.exit(f"[real-cam] HTTP server failed on {args.http_host}:{args.http_port}.")

    # Pointcloud ports (:9601-9604) on the Mac --real rig: the USB cameras have
    # no depth, so each port binds a WS notice — clients get an explanation,
    # never a refused connection. (In sim these ports serve real clouds from the
    # depth renderer; on thor the ROS bridge serves the RealSense/ZED clouds.)
    if data_streams.streams_enabled():
        loop = holder["loop"]
        for _ckey, _cport in data_streams.CLOUD_PORTS.items():
            asyncio.run_coroutine_threadsafe(
                data_streams.serve_hub_ws(
                    args.http_host, _cport,
                    data_streams.NoticeSource(
                        "no depth source on the Mac --real rig (USB color cameras only)",
                        _cport),
                    f"cloud:{_ckey}", log),
                loop)

    for c in cameras:
        c.start()
    log.info("streaming %d camera(s); Ctrl-C to stop", len(cameras))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("shutting down")
        for c in cameras:
            c.stop()


if __name__ == "__main__":
    main()
