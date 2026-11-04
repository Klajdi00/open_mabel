#!/usr/bin/env python3
"""
visualize_sensors.py -- live preview + verification of MABEL's simulated cameras and LiDAR.

Opens THREE windows:
  * the native MuJoCo viewer showing the robot in 3D (orbit/zoom with the mouse)
  * "MABEL Cameras"  -- all on-board cameras, laid out:

        +---------------------------------------------+
        |        HEAD ZED-L  |  HEAD ZED-R            |   <- head stereo (top)
        +---------------------------------------------+
        |  L WRIST   |    BODY CAM    |   R WRIST     |   <- wrists + body (middle)
        +---------------------------------------------+
        |     D435 FRONT     |     D435 BACK          |   <- mobile-base (bottom)
        +---------------------------------------------+

  * "MABEL LiDAR (top-down)" -- 2D plot of the 360-beam scan (robot front = up).

By default the robot is loaded into the furnished home scene (scenes/mabel_home.xml --
kitchen/dining room + real graspable objects) so the cameras and LiDAR have a
realistic environment to look at. Use --props for the simpler obstacle-box room.

Controls (focus a window first):
    q / ESC : quit          space : pause/resume physics
    s       : save snapshots of both image windows to PNG next to this script

PLATFORM:
    Linux:  python3 scripts/visualize_sensors.py            # one process, all windows
    macOS:  mjpython scripts/visualize_sensors.py            # native viewer + camera windows
    On macOS the native MuJoCo viewer (mjpython, main-thread Cocoa) and OpenCV's
    Cocoa windows cannot live in one process, so launching with mjpython runs the
    viewer in the parent and the camera/LiDAR windows in a spawned plain-python
    child (state shared via a memmap). Plain `python` on macOS still works but
    shows only the cv2 windows + a rendered 3D overview (no native viewer).

RENDER BACKEND (speed):
    Defaults to GPU rendering via glfw (fast, needs a display). Software rendering
    (osmesa) is ~10-30x slower -- only use it when there is no display:
        python3 visualize_sensors.py --backend osmesa
    Precedence: --backend flag > MUJOCO_GL env var > glfw default.

Usage (run from the repo root):
    python3 scripts/visualize_sensors.py                 # GPU + viewer + home scene
    python3 scripts/visualize_sensors.py --no-viewer     # skip the 3D robot window
    python3 scripts/visualize_sensors.py --props         # bare obstacle-box room
    python3 scripts/visualize_sensors.py --no-props      # bare model (floor only)
    python3 scripts/visualize_sensors.py --scene scenes/mabel_sim.xml
    python3 scripts/visualize_sensors.py --res 480x360   # per-tile render size
    python3 scripts/visualize_sensors.py --base depth    # show base depth instead of colour
"""
import os
import sys

# --- pick the GL backend BEFORE importing mujoco (must be set at import time) ---
def _pick_backend():
    backend = None
    for i, a in enumerate(sys.argv):
        if a == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
    return backend or os.environ.get("MUJOCO_GL") or "glfw"

os.environ["MUJOCO_GL"] = _pick_backend()

import argparse
import subprocess
import tempfile
import time

import numpy as np
import cv2
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))   # repo root (scripts/ -> ../)
MODEL = os.path.join(ROOT, "models", "mabel_full.xml")
HOME_SCENE = os.path.join(ROOT, "scenes", "mabel_home.xml")  # default: furnished home

IS_MAC = sys.platform == "darwin"


def _under_mjpython():
    """True when launched with `mjpython` (the macOS main-thread viewer host).

    On macOS the OpenCV (Cocoa) image windows and MuJoCo's native passive viewer
    both need the main-thread Cocoa GUI and cannot coexist: under mjpython the
    viewer claims it and cv2.namedWindow() raises "Unknown C++ exception from
    OpenCV code", while under plain python cv2 works but launch_passive() refuses
    to start. mujoco.viewer sets this attribute only under mjpython."""
    return getattr(mujoco.viewer, "_MJPYTHON", None) is not None

# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------
# A room of obstacle props so every directional camera sees something and the
# 360 deg LiDAR returns a recognisable rectangle.  Robot front = -X, back = +X.
PROPS = """
  <worldbody>
    <geom name="wall_front" type="box" pos="-2.0  0.0 0.6" size="0.05 2.0 0.6" rgba="0.85 0.30 0.30 1"/>
    <geom name="wall_back"  type="box" pos=" 2.4  0.0 0.6" size="0.05 2.0 0.6" rgba="0.30 0.55 0.85 1"/>
    <geom name="wall_left"  type="box" pos=" 0.0  2.0 0.6" size="2.2 0.05 0.6" rgba="0.35 0.80 0.40 1"/>
    <geom name="wall_right" type="box" pos=" 0.0 -2.0 0.6" size="2.2 0.05 0.6" rgba="0.90 0.75 0.25 1"/>
    <geom name="prop_fl"   type="cylinder" pos="-1.3  1.3 0.5" size="0.12 0.5" rgba="0.70 0.40 0.80 1"/>
    <geom name="prop_br"   type="box"      pos=" 1.5 -1.2 0.45" size="0.12 0.12 0.45" rgba="0.40 0.75 0.80 1"/>
    <geom name="prop_near" type="cylinder" pos="-1.1 -0.7 0.45" size="0.10 0.45" rgba="0.95 0.55 0.15 1"/>
    <geom name="prop_table" type="box" pos="-0.75 0.0 0.55" size="0.30 0.5 0.02" rgba="0.92 0.92 0.95 1"/>
  </worldbody>
"""


def build_model(args):
    """Return a compiled MjModel.

    Default: the furnished home scene (scenes/mabel_home.xml, robot + kitchen/dining
    room + real graspable objects from home_objects.xml). Override with --scene to
    load a different MJCF, --props for the bare obstacle-box room, or --no-props for
    the floor-only model."""
    if args.scene:
        path = args.scene if os.path.isabs(args.scene) else os.path.join(os.getcwd(), args.scene)
        return mujoco.MjModel.from_xml_path(path)
    if args.no_props:
        return mujoco.MjModel.from_xml_path(MODEL)
    if not args.props:
        return mujoco.MjModel.from_xml_path(HOME_SCENE)
    wrapper = f'<mujoco model="mabel_sensor_viz">\n  <include file="{MODEL}"/>\n{PROPS}\n</mujoco>\n'
    fd, tmp = tempfile.mkstemp(suffix=".xml", prefix="_viz_", dir=os.path.dirname(MODEL))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(wrapper)
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


# ---------------------------------------------------------------------------
# Camera grid
# ---------------------------------------------------------------------------
def tile(img_rgb, label, w, h):
    """RGB->BGR, resize to (w,h), draw a label banner."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != w or bgr.shape[0] != h:
        bgr = cv2.resize(bgr, (w, h))
    cv2.rectangle(bgr, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(bgr, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
    return bgr


def row(tiles, total_w):
    """hstack tiles, centre-pad to total_w with black."""
    r = np.hstack(tiles)
    if r.shape[1] < total_w:
        pad = total_w - r.shape[1]
        l = pad // 2
        r = cv2.copyMakeBorder(r, 0, 0, l, pad - l, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return r


def depth_to_gray(depth, far=5.0):
    """Float depth (m) -> 3ch uint8 (near = bright)."""
    d = np.clip(depth, 0, far) / far
    return (np.repeat((1 - d)[..., None], 3, axis=2) * 255).astype(np.uint8)


def set_render_flags(renderer, shadows=False, reflections=False):
    """Toggle expensive per-render effects. Shadow mapping is recomputed per camera
    per frame and is the dominant cost (~2x); reflections add another ~1.6x.
    MUST be called after update_scene() (which resets the scene flags)."""
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if reflections else 0


def compose_grid(renderer, data, w, h, base_mode, shadows=False):
    """Render every camera and stack them into the requested layout."""
    depth = base_mode == "depth"

    def render(cam):
        renderer.update_scene(data, camera=cam)
        set_render_flags(renderer, shadows)
        img = renderer.render()
        return depth_to_gray(img) if depth else img

    sfx = "(depth)" if depth else ""
    bsfx = "depth" if depth else "color"
    top = row([tile(render("zed_left"),  f"HEAD  ZED-L {sfx}", w, h),
               tile(render("zed_right"), f"HEAD  ZED-R {sfx}", w, h)], 3 * w)
    mid = row([tile(render("l_wrist_cam"), f"L WRIST {sfx}", w, h),
               tile(render("body_cam"),    f"BODY {sfx}",    w, h),
               tile(render("r_wrist_cam"), f"R WRIST {sfx}", w, h)], 3 * w)
    bot = row([tile(render("body_front_cam"), f"BASE FRONT D435 ({bsfx})", w, h),
               tile(render("body_back_cam"),  f"BASE BACK D435 ({bsfx})",  w, h)], 3 * w)
    return np.vstack([top, mid, bot])


# ---------------------------------------------------------------------------
# LiDAR top-down view
# ---------------------------------------------------------------------------
def lidar_beams(model):
    """Return (sensor_adr, site_id) for every rangefinder, plus the chassis body id."""
    beams = [(model.sensor_adr[i], model.sensor_objid[i])
             for i in range(model.nsensor)
             if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER]
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
    return beams, chassis


def auto_range(data, beams, cutoff):
    """Pick a tidy plot radius (even metres) that fits the current returns."""
    d = np.array([data.sensordata[a] for a, _ in beams])
    valid = d[(d >= 0) & (d <= cutoff)]
    if valid.size == 0:
        return 4.0
    even = int(np.ceil(float(np.percentile(valid, 98)) * 1.15 / 2.0)) * 2
    return float(min(max(even, 2), int(cutoff)))


def draw_lidar(data, beams, chassis, max_range, size=620):
    """Top-down scan in the chassis frame; robot front (-X) points up."""
    img = np.zeros((size, size, 3), np.uint8)
    cx = cy = size // 2
    margin = 30
    scale = (size // 2 - margin) / max_range

    step = 2 if max_range > 4 else 1
    for rng in range(step, int(max_range) + 1, step):
        cv2.circle(img, (cx, cy), int(rng * scale), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"{rng}m", (cx + int(rng * scale) - 18, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (cx, margin // 2), (cx, size - margin // 2), (30, 30, 30), 1)
    cv2.line(img, (margin // 2, cy), (size - margin // 2, cy), (30, 30, 30), 1)

    Rc = data.xmat[chassis].reshape(3, 3)
    pc = data.xpos[chassis]
    hits, nearest = 0, max_range
    for adr, sid in beams:
        dist = data.sensordata[adr]
        if dist < 0 or dist > max_range:
            continue
        hit_w = data.site_xpos[sid] + dist * data.site_xmat[sid].reshape(3, 3)[:, 2]
        loc = Rc.T @ (hit_w - pc)
        px = int(cx + loc[1] * scale)      # robot right (+Y) -> screen right
        py = int(cy + loc[0] * scale)      # front (-X) -> screen up  (cy - (-loc[0]))
        if 0 <= px < size and 0 <= py < size:
            t = min(dist / max_range, 1.0)
            col = (int(60 + 195 * t), int(255 * (1 - t) + 80), int(80 + 100 * (1 - t)))
            cv2.circle(img, (px, py), 2, col, -1)
            hits += 1
            nearest = min(nearest, dist)

    half = int(0.245 * scale)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (120, 120, 200), 1)
    tip = (cx, cy - half - 12)
    cv2.drawContours(img, [np.array([tip, (cx - 8, cy - half), (cx + 8, cy - half)])],
                     0, (120, 120, 255), -1)
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.putText(img, f"LiDAR  {hits}/{len(beams)} returns   nearest={nearest:0.2f} m   front=up",
                (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Reusable sensor capture -- import these from a ROS bridge node later.
#   rgb   = capture_rgb(renderer, data, "zed_left")              -> Image (rgb8)
#   depth = capture_depth(renderer_d, data, "body_front_cam")    -> Image (32FC1, m)
#   fx,fy,cx,cy = camera_intrinsics(model, "zed_left", W, H)     -> CameraInfo.K
#   angles,ranges,amin,ainc,rmax = lidar_scan(model,data,beams,chassis) -> LaserScan
# ---------------------------------------------------------------------------
CAMERA_NAMES = ["zed_left", "zed_right", "l_wrist_cam", "r_wrist_cam", "body_cam",
                "body_front_cam", "body_back_cam"]


def capture_rgb(renderer, data, cam, shadows=False):
    """RGB uint8 (H,W,3) for `cam`. Renderer must be in colour mode."""
    renderer.update_scene(data, camera=cam)
    set_render_flags(renderer, shadows)
    return renderer.render()


def capture_depth(renderer, data, cam):
    """Depth float32 (H,W) in metres. Renderer must have enable_depth_rendering()."""
    renderer.update_scene(data, camera=cam)
    return renderer.render().astype(np.float32)


def camera_intrinsics(model, cam, width, height):
    """Pinhole (fx, fy, cx, cy) from the camera's fovy at the given image size.
    K = [[fx,0,cx],[0,fy,cy],[0,0,1]] for sensor_msgs/CameraInfo."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fy = (height / 2.0) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2.0)
    return float(fy), float(fy), width / 2.0, height / 2.0


def lidar_scan(model, data, beams, chassis):
    """Planar scan as ROS sensor_msgs/LaserScan fields.
    Returns (angles, ranges, angle_min, angle_increment, range_max): angles ascending
    in rad about chassis +Z (0 = chassis +X), no-return beams -> +inf."""
    Rc = data.xmat[chassis].reshape(3, 3)
    pairs = []
    for adr, sid in beams:
        d = data.sensordata[adr]
        bdir = Rc.T @ data.site_xmat[sid].reshape(3, 3)[:, 2]
        pairs.append((float(np.arctan2(bdir[1], bdir[0])), d if d >= 0 else np.inf))
    pairs.sort()
    angles = np.array([a for a, _ in pairs])
    ranges = np.array([r for _, r in pairs])
    cutoff = max((model.sensor_cutoff[i] for i in range(model.nsensor)
                  if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER), default=12.0) or 12.0
    inc = float(np.mean(np.diff(angles))) if angles.size > 1 else 0.0
    return angles, ranges, (float(angles[0]) if angles.size else 0.0), inc, float(cutoff)


# ---------------------------------------------------------------------------
# Render helpers shared by every run mode
# ---------------------------------------------------------------------------
def _make_renderer(args, model):
    """Offscreen renderer + LiDAR beam table for the camera/LiDAR windows."""
    w, h = (int(v) for v in args.res.lower().split("x"))
    renderer = mujoco.Renderer(model, height=h, width=w)
    if args.base == "depth":
        renderer.enable_depth_rendering()
    beams, chassis = lidar_beams(model)
    cutoff = max((model.sensor_cutoff[i] for i in range(model.nsensor)
                  if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER),
                 default=12.0) or 12.0
    return w, h, renderer, beams, chassis, cutoff


def _draw_frame(renderer, data, args, w, h, beams, chassis, cutoff, fps_meas, status):
    """Render the camera grid + LiDAR scan and push both to their cv2 windows."""
    grid = compose_grid(renderer, data, w, h, args.base, shadows=args.shadows)
    plot_range = args.range if args.range > 0 else auto_range(data, beams, cutoff)
    scan = draw_lidar(data, beams, chassis, plot_range)
    cv2.putText(grid,
                f"cams {fps_meas:4.1f}/{args.cam_fps:.0f} FPS  [{os.environ['MUJOCO_GL']}]  {status}",
                (8, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imshow("MABEL Cameras", grid)
    cv2.imshow("MABEL LiDAR (top-down)", scan)
    return grid, scan


def _save_snapshots(grid, scan):
    ts = time.strftime("%Y%m%d_%H%M%S")
    cam_path = os.path.join(os.getcwd(), f"cameras_{ts}.png")
    lidar_path = os.path.join(os.getcwd(), f"lidar_{ts}.png")
    cv2.imwrite(cam_path, grid)
    cv2.imwrite(lidar_path, scan)
    print(f"[viz] saved {cam_path} and {lidar_path}")


# ---------------------------------------------------------------------------
# Single-process path -- Linux (native viewer + cv2 together) and the macOS
# plain-python fallback (cv2 windows + a rendered, auto-orbiting 3D overview).
# This is the original behaviour; nothing changes for Ubuntu users.
# ---------------------------------------------------------------------------
def run_single_process(args, allow_viewer=True):
    model = build_model(args)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    try:
        w, h, renderer, beams, chassis, cutoff = _make_renderer(args, model)
    except Exception as e:
        print(f"[viz] ERROR: could not create '{os.environ['MUJOCO_GL']}' renderer: {e}")
        print("[viz] If you have no display, retry with:  --backend osmesa")
        return

    viewer = None
    if allow_viewer and not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            # Camera markers live in geom group 3 (off by default, so the cameras
            # never render them). Turn the group on in the 3D viewer so the little
            # cyan camera boxes are visible. Toggle in-viewer with the '3' key.
            viewer.opt.geomgroup[3] = 1
            viewer.sync()
        except Exception as e:
            print(f"[viz] 3D viewer unavailable ({e}); continuing with image windows only")

    # When there is no native viewer, draw an auto-orbiting 3D overview via cv2.
    overview = ov_cam = ov_opt = None
    if viewer is None:
        try:
            overview = mujoco.Renderer(model, height=480, width=640)
            ov_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, ov_cam)
            ov_opt = mujoco.MjvOption()
            ov_opt.geomgroup[3] = 1
        except Exception as e:
            print(f"[viz] 3D overview unavailable ({e})")
            overview = None

    print(f"[viz] {model.ncam} cameras, {len(beams)} LiDAR beams | backend={os.environ['MUJOCO_GL']} | tile={w}x{h}")
    print("[viz] keys:  q/ESC quit   space pause   s save")

    cv2.namedWindow("MABEL Cameras", cv2.WINDOW_NORMAL)
    cv2.namedWindow("MABEL LiDAR (top-down)", cv2.WINDOW_NORMAL)
    if overview is not None:
        cv2.namedWindow("MABEL 3D overview", cv2.WINDOW_NORMAL)

    timestep = model.opt.timestep
    cam_period = 1.0 / max(args.cam_fps, 1.0)
    max_catchup = 0.05  # cap sim catch-up per loop (anti spiral-of-death)

    paused = False
    fps_meas, last_cam = 0.0, 0.0
    grid = scan = None
    prev = time.perf_counter()
    acc = 0.0
    while True:
        if viewer is not None and not viewer.is_running():
            break

        # ---- real-time physics: advance sim to match elapsed wall-clock ----
        now = time.perf_counter()
        if not paused:
            acc += min(now - prev, max_catchup)
            while acc >= timestep:
                mujoco.mj_step(model, data)
                acc -= timestep
        prev = now
        if viewer is not None:
            viewer.sync()                      # keep the 3D robot view live every loop

        # ---- camera grid + LiDAR, streamed at --cam-fps (default 30) ----
        if now - last_cam >= cam_period:
            fps_meas = 0.9 * fps_meas + 0.1 / max(now - last_cam, 1e-6)
            last_cam = now
            grid, scan = _draw_frame(renderer, data, args, w, h, beams, chassis, cutoff,
                                     fps_meas, f"phys {'PAUSED' if paused else 'real-time'}")
            if overview is not None:
                ov_cam.azimuth = (ov_cam.azimuth + 0.3) % 360   # slow auto-orbit
                overview.update_scene(data, camera=ov_cam, scene_option=ov_opt)
                cv2.imshow("MABEL 3D overview",
                           cv2.cvtColor(overview.render(), cv2.COLOR_RGB2BGR))

        k = cv2.waitKey(1) & 0xFF             # pumps GUI events + paces the loop
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("s") and grid is not None:
            _save_snapshots(grid, scan)

    if viewer is not None:
        viewer.close()
    if overview is not None:
        overview.close()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# macOS two-process path. cv2's Cocoa windows and MuJoCo's native viewer each
# need a main thread, which is impossible in one process under mjpython. So the
# mjpython PARENT owns physics + the native viewer and publishes the sim state
# to a memory-mapped file; a plain-python CHILD reads it and draws the camera
# grid + LiDAR with OpenCV. Shared float64 layout:
#   [0] sim time   [1] parent-alive   [2] child-quit   [3:] qpos
# ---------------------------------------------------------------------------
_STATE_HEADER = 3


def run_mac_parent(args):
    """mjpython parent: native 3D viewer + physics, spawn the cv2 camera child."""
    model = build_model(args)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    beams, _ = lidar_beams(model)

    n = _STATE_HEADER + model.nq
    fd, statepath = tempfile.mkstemp(suffix=".f64", prefix="mabel_viz_")
    os.close(fd)
    state = np.memmap(statepath, dtype=np.float64, mode="w+", shape=(n,))
    state[0] = data.time
    state[1] = 1.0                                   # parent alive
    state[2] = 0.0
    state[_STATE_HEADER:] = data.qpos

    # Spawn the camera/LiDAR window as a separate PLAIN-python process. Under
    # mjpython sys.executable is the regular python interpreter, so this gives a
    # process with its own main thread where cv2's Cocoa GUI works.
    child_cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:] + \
                ["--cam-worker", statepath]
    print(f"[viz] launching camera/LiDAR window: {os.path.basename(sys.executable)} (separate process)")
    child = subprocess.Popen(child_cmd, env=os.environ.copy())

    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.opt.geomgroup[3] = 1
        viewer.sync()
    except Exception as e:
        print(f"[viz] native viewer unavailable ({e}); camera window still runs")

    print(f"[viz] {model.ncam} cameras, {len(beams)} LiDAR beams | native viewer + camera process")

    timestep = model.opt.timestep
    max_catchup = 0.05
    prev = time.perf_counter()
    acc = 0.0
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if state[2] != 0.0:                      # child asked to quit
                break
            if child.poll() is not None:             # child window closed/crashed
                break
            now = time.perf_counter()
            acc += min(now - prev, max_catchup)
            while acc >= timestep:
                mujoco.mj_step(model, data)
                acc -= timestep
            prev = now
            state[0] = data.time                     # publish state for the child
            state[_STATE_HEADER:] = data.qpos
            if viewer is not None:
                viewer.sync()
            time.sleep(0.001)
    finally:
        state[1] = 0.0                               # tell the child to stop
        try:
            state.flush()
        except Exception:
            pass
        if viewer is not None:
            viewer.close()
        try:
            child.wait(timeout=2.0)
        except Exception:
            child.terminate()
        del state
        try:
            os.remove(statepath)
        except OSError:
            pass


def run_camera_worker(args):
    """Plain-python child: read qpos from the memmap and draw cv2 windows. Mirrors
    the parent's scene exactly (same args -> same model -> identical qpos layout)."""
    model = build_model(args)
    data = mujoco.MjData(model)
    if not os.path.exists(args.cam_worker):
        print("[viz][cam] parent state file missing; exiting")
        return
    n = _STATE_HEADER + model.nq
    state = np.memmap(args.cam_worker, dtype=np.float64, mode="r+", shape=(n,))

    try:
        w, h, renderer, beams, chassis, cutoff = _make_renderer(args, model)
    except Exception as e:
        print(f"[viz][cam] ERROR: could not create '{os.environ['MUJOCO_GL']}' renderer: {e}")
        state[2] = 1.0
        return

    cv2.namedWindow("MABEL Cameras", cv2.WINDOW_NORMAL)
    cv2.namedWindow("MABEL LiDAR (top-down)", cv2.WINDOW_NORMAL)
    print(f"[viz][cam] {model.ncam} cameras, {len(beams)} LiDAR beams | backend={os.environ['MUJOCO_GL']} | tile={w}x{h}")
    print("[viz][cam] keys (focus a camera window):  q/ESC quit   s save")

    cam_period = 1.0 / max(args.cam_fps, 1.0)
    fps_meas, last_cam = 0.0, 0.0
    grid = scan = None
    nq = model.nq
    try:
        while True:
            if state[1] == 0.0:                      # parent gone
                break
            now = time.perf_counter()
            if now - last_cam >= cam_period:
                fps_meas = 0.9 * fps_meas + 0.1 / max(now - last_cam, 1e-6)
                last_cam = now
                data.qpos[:] = state[_STATE_HEADER:_STATE_HEADER + nq]
                data.time = float(state[0])
                mujoco.mj_forward(model, data)
                grid, scan = _draw_frame(renderer, data, args, w, h, beams, chassis,
                                         cutoff, fps_meas, "live (viewer process)")
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                state[2] = 1.0                       # ask the parent to quit too
                break
            elif k == ord("s") and grid is not None:
                _save_snapshots(grid, scan)
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=os.environ["MUJOCO_GL"],
                    help="GL backend: glfw (GPU, default) | osmesa (software) | egl")
    ap.add_argument("--scene", help="load this MJCF scene (default: scenes/mabel_home.xml)")
    ap.add_argument("--props", action="store_true",
                    help="use the bare obstacle-box room instead of the home scene")
    ap.add_argument("--no-props", action="store_true", help="floor-only model, no scene at all")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the 3D robot viewer")
    ap.add_argument("--res", default="360x240", help="per-tile render size WxH (default 360x240)")
    ap.add_argument("--base", choices=["color", "depth"], default="color",
                    help="show base D435 colour or depth (default color)")
    ap.add_argument("--range", type=float, default=0.0,
                    help="LiDAR plot max range, m (default 0 = auto-fit to returns)")
    ap.add_argument("--cam-fps", type=float, default=30.0,
                    help="camera-grid + LiDAR stream rate, Hz (default 30; physics always real-time)")
    ap.add_argument("--shadows", action="store_true",
                    help="render camera shadows (prettier but ~2x slower; off by default for 30 FPS)")
    ap.add_argument("--cam-worker", default=None, help=argparse.SUPPRESS)  # internal (macOS child)
    args = ap.parse_args()

    # --- dispatch by platform / role ---------------------------------------
    if args.cam_worker:                         # macOS camera/LiDAR child process
        run_camera_worker(args)
        return

    if IS_MAC:
        # On macOS cv2's Cocoa windows and MuJoCo's native viewer can't share a
        # process. With mjpython we run the two-process mode (native viewer +
        # cv2 child); under plain python we fall back to cv2 windows + overview.
        if _under_mjpython():
            run_mac_parent(args)
        else:
            rel = os.path.relpath(os.path.abspath(__file__), os.getcwd())
            print("[viz] tip: launch with mjpython to also get the native MuJoCo viewer:")
            print(f"[viz]     mjpython {rel}")
            print("[viz] running camera/LiDAR windows + rendered 3D overview (no native viewer).")
            run_single_process(args, allow_viewer=False)
        return

    run_single_process(args)                    # Linux/other: native viewer + cv2 in one process


if __name__ == "__main__":
    main()
