#!/usr/bin/env python3
"""bringup.py -- the dedicated, ROS-free camera bring-up + live viewer for MABEL.

Connects to whatever depth cameras are plugged in (Intel RealSense D435i and/or
Stereolabs ZED-M), reads RGB + metric depth + an XYZRGB point cloud + IMU, and
shows them live:
    * a colourful terminal dashboard (per-camera FPS / intrinsics / depth / IMU)
    * an OpenCV window per camera: RGB next to a colour-mapped depth image
    * one Open3D window with the fused, live point cloud

    python3 bringup.py                 # all cameras, full viz
    python3 bringup.py --cameras realsense
    python3 bringup.py --no-window     # headless: dashboard only
    python3 bringup.py --no-cloud      # skip the 3D point-cloud window

Run it through sensors/bring_up_cameras.sh, which sets up the venv + SDKs.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

from camera_hal import CameraManager, RealSenseCamera, ZedCamera
from camera_hal.manager import CameraStatus

# -- optional deps: degrade gracefully if a viz lib is missing ----------------
try:
    import cv2
except Exception:
    cv2 = None
try:
    import open3d as o3d
except Exception:
    o3d = None
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except Exception:
    Console = None


# =============================================================================
# Pretty terminal dashboard (rich)
# =============================================================================
_STATE_STYLE = {"running": "bold green", "unavailable": "yellow",
                "error": "bold red", "init": "dim", "waiting": "cyan"}
_STATE_ICON = {"running": "●", "unavailable": "○", "error": "✗", "init": "…",
               "waiting": "◌"}


def _imu_str(frame) -> str:
    if frame is None or frame.imu is None:
        return "[dim]—[/dim]"
    ax, ay, az = frame.imu.accel
    gx, gy, gz = frame.imu.gyro
    return (f"a[{ax:+5.1f}{ay:+5.1f}{az:+5.1f}] "
            f"g[{gx:+5.2f}{gy:+5.2f}{gz:+5.2f}]")


def _build_table(statuses: list[CameraStatus]) -> "Table":
    t = Table(box=box.ROUNDED, expand=True, border_style="cyan",
              title="[bold cyan]MABEL · camera bring-up[/bold cyan]",
              caption="[dim]q in a window or Ctrl-C to quit[/dim]")
    t.add_column("camera", style="bold")
    t.add_column("state")
    t.add_column("fps", justify="right")
    t.add_column("stream / intrinsics", overflow="fold")
    t.add_column("depth m", justify="right")
    t.add_column("points", justify="right")
    t.add_column("imu  accel m/s²  gyro rad/s", overflow="fold")

    for s in statuses:
        style = _STATE_STYLE.get(s.state, "white")
        state = f"[{style}]{_STATE_ICON.get(s.state,'?')} {s.state}[/{style}]"
        f = s.latest
        if s.state == "running" and f is not None:
            lo, hi = f.depth_range_m
            depth = f"{lo:.2f}–{hi:.2f}"
            npts = f"{len(f.points_xyz):,}" if f.has_cloud else "0"
            intr = s.detail
            imu = _imu_str(f)
        else:
            depth, npts, intr, imu = "—", "—", s.detail, "[dim]—[/dim]"
        t.add_row(f"{s.model}", state, f"{s.fps:4.1f}", intr, depth, npts, imu)
    return t


class Dashboard:
    def __init__(self, manager: CameraManager):
        self.manager = manager
        self.console = Console() if Console else None

    def run(self, stop: threading.Event, fps: float = 4.0) -> None:
        if self.console is None:                  # rich missing -> plain lines
            while not stop.is_set():
                for s in self.manager.statuses():
                    print(f"[{s.state:11}] {s.model:24} {s.fps:4.1f}fps  {s.detail}")
                time.sleep(1.0)
            return
        with Live(_build_table(self.manager.statuses()), console=self.console,
                  refresh_per_second=fps, screen=False) as live:
            while not stop.is_set():
                live.update(_build_table(self.manager.statuses()))
                time.sleep(1.0 / fps)


# =============================================================================
# OpenCV RGB + depth windows
# =============================================================================
def _depth_colormap(depth_m: np.ndarray, near: float = 0.2, far: float = 6.0) -> np.ndarray:
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.clip((d - near) / max(far - near, 1e-3), 0.0, 1.0)
    img = (norm * 255).astype(np.uint8)
    col = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    col[d <= 0] = (0, 0, 0)                       # invalid -> black
    return col


def _rgbd_panel(frame, disp_h: int = 360) -> np.ndarray | None:
    """RGB beside a colour-mapped depth, at a small DISPLAY size.

    The depth colourmap (nan_to_num + applyColormap + invalid mask) on a full
    1280x720 frame costs ~27 ms; doing it on a downscaled image instead drops it
    to a few ms -- the single biggest viewer-latency win after the cloud fix.
    """
    if frame is None or frame.rgb is None:
        return None
    rgb = frame.rgb
    h, w = rgb.shape[:2]
    dw = max(1, int(w * disp_h / h))
    rgb_s = cv2.resize(rgb, (dw, disp_h), interpolation=cv2.INTER_AREA)
    if frame.depth_m is not None:
        # NEAREST keeps the 0=invalid sentinels intact through the resize
        d_s = cv2.resize(frame.depth_m, (dw, disp_h), interpolation=cv2.INTER_NEAREST)
        depth_s = _depth_colormap(d_s)
    else:
        depth_s = np.zeros_like(rgb_s)
    return np.hstack([rgb_s, depth_s])


# =============================================================================
# Open3D point cloud -- ONE window per camera (created once, never re-spawned)
# =============================================================================
class CloudWindow:
    def __init__(self, title: str, left: int = 0, top: int = 0):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=900, height=680,
                               left=left, top=top)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
        self.alive = True
        self._added = False
        self._last_stamp = -1.0

    def update(self, frame) -> None:
        # Re-upload to the GPU ONLY when a genuinely new frame arrived -- the
        # render loop spins faster than the camera, so re-pushing the same cloud
        # is wasted work. poll/render every call to keep the window responsive.
        new = frame is not None and frame.has_cloud and frame.stamp != self._last_stamp
        if new:
            self._last_stamp = frame.stamp
            # CRITICAL latency fix: Vector3dVector does an element-wise copy for
            # float32 (~110 ms / 340k pts) but a raw memcpy for CONTIGUOUS float64
            # (~0.5 ms). Always hand it contiguous float64.
            xyz = np.ascontiguousarray(frame.points_xyz, dtype=np.float64)
            if frame.points_rgb is not None and len(frame.points_rgb) == len(xyz):
                rgb = np.ascontiguousarray(frame.points_rgb[:, ::-1], dtype=np.float64) / 255.0
            else:
                rgb = np.full((len(xyz), 3), 0.7)
            self.pcd.points = o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = o3d.utility.Vector3dVector(rgb)
            if not self._added:
                self.vis.add_geometry(self.pcd, reset_bounding_box=True)  # frame view once
                self._added = True
            else:
                self.vis.update_geometry(self.pcd)
        self.alive = self.vis.poll_events() and (self.vis.update_renderer() or True)

    def close(self) -> None:
        try:
            self.vis.destroy_window()
        except Exception:
            pass


# =============================================================================
# main
# =============================================================================
def _select_cameras(args) -> list:
    want = args.cameras
    cams = []
    if want in ("all", "realsense"):
        cams.append(RealSenseCamera(width=args.width, height=args.height, fps=args.fps,
                                    enable_imu=not args.no_imu, cloud_stride=args.cloud_stride))
    if want in ("all", "zed"):
        cams.append(ZedCamera(resolution=args.zed_res, fps=args.fps,
                              cloud_stride=args.cloud_stride))
    return cams


def main() -> int:
    ap = argparse.ArgumentParser(description="ROS-free camera bring-up + live viewer")
    ap.add_argument("--cameras", choices=["all", "realsense", "zed"], default="all")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zed-res", default="HD720")
    ap.add_argument("--cloud-stride", type=int, default=2,
                    help="subsample factor for the point cloud (2 = 1/4 points)")
    ap.add_argument("--no-imu", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="headless: dashboard only")
    ap.add_argument("--no-cloud", action="store_true", help="skip the 3D cloud window")
    args = ap.parse_args()

    headless = args.no_window or not os.environ.get("DISPLAY")
    if args.no_window is False and not os.environ.get("DISPLAY"):
        print("[i] no DISPLAY -> dashboard-only (headless). Use --no-window to silence.")

    cams = _select_cameras(args)
    if not cams:
        print("no cameras selected", file=sys.stderr)
        return 2

    manager = CameraManager(cams)
    manager.start()
    time.sleep(0.5)                               # let workers attempt open()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    dash = Dashboard(manager)
    if headless:
        dash.run(stop)
        manager.stop()
        return 0

    if cv2 is None:
        print("[!] OpenCV missing -> dashboard only", file=sys.stderr)
        dash.run(stop)
        manager.stop()
        return 0

    # dashboard in the background, GUI loop owns the main thread
    dash_thread = threading.Thread(target=dash.run, args=(stop,), daemon=True)
    dash_thread.start()

    # Wait briefly for the cameras to come up, then snapshot the running set ONCE
    # so each camera gets exactly one RGB window + one cloud window -- created
    # here, never inside the render loop (that was spawning duplicate windows).
    deadline = time.monotonic() + 10.0           # ZED open()+calib can take ~4s
    while time.monotonic() < deadline:
        if all(s.state != "init" for s in manager.statuses()):
            break                                # every camera resolved (running/unavailable)
        time.sleep(0.2)
    running = manager.running_names()
    labels = {s.name: s.model for s in manager.statuses()}

    if not running:
        print("[!] no camera is streaming -> dashboard only "
              "(check sensors/usb_check.sh)", file=sys.stderr)
        dash.run(stop)
        manager.stop()
        return 0

    # one RGB|depth window per camera (top row), one cloud window per camera
    # (bottom row), tiled so they don't stack on top of each other.
    rgb_titles: dict[str, str] = {}
    clouds: dict[str, CloudWindow] = {}
    for i, name in enumerate(running):
        rt = f"{labels[name]}  —  RGB | depth"
        rgb_titles[name] = rt
        cv2.namedWindow(rt, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(rt, 900, 360)
        cv2.moveWindow(rt, 30 + i * 940, 30)
        if o3d is not None and not args.no_cloud:
            try:
                clouds[name] = CloudWindow(f"{labels[name]}  —  point cloud",
                                           left=30 + i * 940, top=440)
            except Exception as e:                # no GL
                print(f"[!] cloud window for {name} unavailable: {e}", file=sys.stderr)

    print(f"[i] showing {len(running)} camera(s): {', '.join(labels[n] for n in running)}"
          f"  — press q (or close a window) to quit")

    last_stamp = {name: -1.0 for name in running}
    try:
        while not stop.is_set():
            for name in running:
                f = manager.latest(name)
                if f is not None and f.stamp != last_stamp[name]:
                    last_stamp[name] = f.stamp     # only redraw on a NEW frame
                    panel = _rgbd_panel(f)
                    if panel is not None:
                        cv2.imshow(rgb_titles[name], panel)
                if name in clouds:
                    clouds[name].update(f)         # internally skips stale frames
            if clouds and all(not c.alive for c in clouds.values()):
                break                             # every cloud window closed
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        stop.set()
        for c in clouds.values():
            c.close()
        cv2.destroyAllWindows()
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
