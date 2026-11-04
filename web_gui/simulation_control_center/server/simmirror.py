"""
simmirror.py — MuJoCo state mirror: the pure-sim replacement for the ROS map
plane (and the LiDAR scan) on a Mac.

The bridge publishes its qpos to a shared memmap every physics substep (the
same file sim_camera_server.py and view_robot.py read). This mirror loads the
SAME scene, copies qpos in at MIRROR_HZ, runs mj_forward, and derives:

  * odom   — chassis world pose        → map-protocol 0x02 records
  * cloud  — head-camera depth render, back-projected + colored from the RGB
             render, in the world (= odom) frame → 0x05 records @ CLOUD_HZ
  * scan   — the model's 360 rangefinder ring (lidar_range000…359) → the
             telemetry `scan` field (the LiDAR layer in the viewer)

Records use the EXACT encoders from server/map_stream.py (loaded by file path,
so there is one wire-format source of truth), fanned out by MirrorMapHub to
the backend's same-origin /ws/map websocket. No SLAM mesh / occupancy / Nav2
path in pure sim — those chips simply stay empty, by design.
"""
import asyncio
import importlib.util
import json
import logging
import math
import os
import threading
import time

import numpy as np

from . import config as CFG

log = logging.getLogger("scc.mirror")


def load_map_stream():
    """server/map_stream.py by path (no sys.path pollution; rclpy stays lazy)."""
    path = os.path.join(CFG.SERVER_DIR, "map_stream.py")
    spec = importlib.util.spec_from_file_location("scc_map_stream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MirrorMapHub:
    """Latest-snapshot fan-out to FastAPI websockets (mirrors MapStreamHub's
    queue-per-subscriber pattern, but for starlette sockets)."""

    _QUEUE_MAX = 64

    def __init__(self):
        self.loop = None
        self._subs = {}
        self._odom = None            # latest 0x02 frame
        self._cloud = []             # latest 0x05 chunk list
        self._grid = None            # latest 0x04 frame (native Mac SLAM)
        self._path = None            # latest 0x03 frame (native Mac nav)

    def attach_loop(self, loop):
        self.loop = loop

    # ── producer side (mirror / nav threads) ─────────────────────────────────
    def publish_odom(self, frame):
        self._odom = frame
        self._enqueue((frame,))

    def publish_cloud(self, chunks):
        self._cloud = list(chunks)
        self._enqueue(chunks)

    def publish_grid(self, frame):
        self._grid = frame
        self._enqueue((frame,))

    def publish_path(self, frame):
        self._path = frame
        self._enqueue((frame,))

    def _enqueue(self, frames):
        loop, subs = self.loop, self._subs
        if loop is None or not subs:
            return
        frames = list(frames)

        def _on_loop():
            for q in list(subs.values()):
                for fr in frames:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    try:
                        q.put_nowait(fr)
                    except Exception:
                        pass
        try:
            loop.call_soon_threadsafe(_on_loop)
        except RuntimeError:
            pass

    # ── subscriber side (FastAPI /ws/map endpoint, already accepted) ─────────
    async def subscribe(self, ws):
        q = asyncio.Queue(maxsize=self._QUEUE_MAX)
        if self._odom is not None:
            q.put_nowait(self._odom)
        if self._grid is not None:
            q.put_nowait(self._grid)
        if self._path is not None:
            q.put_nowait(self._path)
        for fr in self._cloud:
            q.put_nowait(fr)
        key = id(ws)
        self._subs[key] = q

        async def _drain_uplink():
            while True:                       # clients are send-only; ignore input
                await ws.receive()

        async def _send():
            while True:
                await ws.send_bytes(await q.get())
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(_drain_uplink()),
                 asyncio.create_task(_send())],
                return_when=asyncio.FIRST_EXCEPTION)
            for t in pending:
                t.cancel()
        finally:
            self._subs.pop(key, None)


class SimMirror:
    """Background thread: memmap qpos → mj_forward → scan/odom/cloud."""

    def __init__(self, state, hub, stream_hub=None):
        self.state = state                  # TelemetryState (scan sink)
        self.hub = hub
        self.stream_hub = stream_hub        # streams.HUB: per-camera cloud sink
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="scc-sim-mirror")
        self.scene = None
        self.ok = False                     # attached to a live memmap
        self.error = None
        self._src_last = {}                 # per-source cloud last-render time

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    # ── attachment ───────────────────────────────────────────────────────────
    def _sidecar_scene(self):
        sidecar = CFG.STATE_FILE + ".scene"
        try:
            with open(sidecar) as f:
                path = f.readline().strip()
            return path or None
        except OSError:
            return None

    def _attach(self, mujoco):
        """(Re)load the scene the bridge recorded and open the memmap."""
        scene = self._sidecar_scene() or CFG.DEFAULT_SCENE
        model = mujoco.MjModel.from_xml_path(scene)
        data = mujoco.MjData(model)
        need = (1 + model.nq) * 8
        try:
            size = os.path.getsize(CFG.STATE_FILE) // 8
        except OSError:
            return None
        if size < 1 + model.nq:
            return None
        # v2 writers (simulation_bridge) append qvel+ctrl — map them when there
        full = 1 + model.nq + model.nv + model.nu
        has_vel = size >= full
        mm = np.memmap(CFG.STATE_FILE, dtype=np.float64, mode="r",
                       shape=(full if has_vel else 1 + model.nq,))
        # rangefinder ring (may be absent in reduced scenes)
        lidar = []
        i = 0
        while True:
            try:
                s = model.sensor(f"lidar_range{i:03d}")
            except Exception:
                break
            lidar.append(int(s.adr[0]))
            i += 1
        # chassis body for odom
        try:
            chassis = model.body("mabel_chassis").id
        except Exception:
            chassis = 0
        # depth-cloud cameras (head + front base; any that exist in the scene)
        cam_ids = []
        for cn in CFG.CLOUD_CAMERAS:
            try:
                cam_ids.append(model.camera(cn).id)
            except Exception:
                log.warning("cloud camera %r not in scene — skipped", cn)
        # per-source cloud cameras (head / front / wrist_left / wrist_right) for
        # the selectable /ws/stream toggles — any that exist in this scene
        from . import streams as _streams
        cam_by_source = {}
        for src, cn in _streams.SIM_CLOUD_CAMERAS.items():
            try:
                cam_by_source[src] = model.camera(cn).id
            except Exception:
                pass
        # scene-visualisation option with the LiDAR rangefinder rays OFF: the
        # default MjvOption has mjVIS_RANGEFINDER=1, which draws the 360 laser
        # beams as decorative geoms that land in the DEPTH buffer → a curtain of
        # phantom points along every ray. Turn it off (+ hide the group-3 ray
        # site) so the cloud sees only real geometry.
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        vopt.sitegroup[3] = 0
        # wheel + gyro addresses for the native nav/SLAM odometry
        wheels = None
        try:
            steer_q = [int(model.joint(f"{m} Steer").qposadr[0])
                       for m in ("FL", "FR", "B")]
            drive_d = [int(model.joint(f"{m} Drive").dofadr[0])
                       for m in ("FL", "FR", "B")]
            base_d = int(model.joint("base_root").dofadr[0])
            wheels = dict(steer_q=steer_q, drive_d=drive_d, wz_d=base_d + 5)
        except Exception:
            pass
        return dict(scene=scene, model=model, data=data, mm=mm,
                    lidar=np.asarray(lidar, dtype=int), chassis=chassis,
                    cam=(cam_ids or None), cam_by_source=cam_by_source,
                    vopt=vopt, rgb_r=None, dep_r=None,
                    has_vel=has_vel, wheels=wheels)

    # ── main loop ────────────────────────────────────────────────────────────
    def _run(self):
        try:
            import mujoco
        except Exception as e:                       # no MuJoCo → mirror off
            self.error = f"mujoco unavailable: {e}"
            log.warning(self.error)
            return
        ms = load_map_stream()
        ctx = None
        period = 1.0 / max(1.0, CFG.MIRROR_HZ)
        cloud_period = 1.0 / max(0.1, CFG.CLOUD_HZ)
        last_cloud = 0.0
        last_scene_check = 0.0
        t_frame = time.monotonic()
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if ctx is None:
                    ctx = self._attach(mujoco)
                    if ctx is None:
                        self.ok = False
                        self._stop.wait(1.0)
                        continue
                    self.scene = ctx["scene"]
                    self.ok = True
                    log.info("mirror attached — scene %s (nq=%d, lidar=%d rays)",
                             os.path.basename(ctx["scene"]),
                             ctx["model"].nq, len(ctx["lidar"]))
                # scene switched underneath us? (bridge restart writes sidecar)
                if t0 - last_scene_check > 1.0:
                    last_scene_check = t0
                    cur = self._sidecar_scene()
                    need = (1 + ctx["model"].nq) * 8
                    if ((cur and cur != ctx["scene"]) or not
                            (os.path.exists(CFG.STATE_FILE)
                             and os.path.getsize(CFG.STATE_FILE) >= need)):
                        log.info("scene changed → reattaching mirror")
                        self._close_renderers(ctx)
                        ctx = None
                        continue

                model, data, mm = ctx["model"], ctx["data"], ctx["mm"]
                data.qpos[:] = mm[1:1 + model.nq]
                data.time = float(mm[0])
                mujoco.mj_forward(model, data)

                # LiDAR scan → telemetry (ray i = i degrees around +x)
                if len(ctx["lidar"]):
                    rng = data.sensordata[ctx["lidar"]]
                    self.state.scan = [round(float(r), 3) if r >= 0 else None
                                       for r in rng]

                # wheel/gyro/ground-truth state → the native nav/SLAM runner
                w = ctx.get("wheels")
                if w is not None:
                    m = ctx["model"]
                    gyro = 0.0
                    drive = [0.0, 0.0, 0.0]
                    if ctx.get("has_vel"):
                        qv = mm[1 + m.nq:1 + m.nq + m.nv]
                        gyro = float(qv[w["wz_d"]])
                        drive = [float(qv[d]) for d in w["drive_d"]]
                    bq = data.xquat[ctx["chassis"]]
                    bp = data.xpos[ctx["chassis"]]
                    gt_yaw = math.atan2(2 * (bq[0] * bq[3] + bq[1] * bq[2]),
                                        1 - 2 * (bq[2] ** 2 + bq[3] ** 2))
                    self.base_state = {
                        "t": float(mm[0]),
                        "steer": [float(data.qpos[q]) for q in w["steer_q"]],
                        "drive": drive,
                        "gyro_wz": gyro,
                        "gt": (float(bp[0]), float(bp[1]), gt_yaw),
                    }

                # odom
                b_xpos = data.xpos[ctx["chassis"]]
                b_xquat = data.xquat[ctx["chassis"]]      # (w,x,y,z)
                self.hub.publish_odom(ms.encode_odom(
                    float(b_xpos[0]), float(b_xpos[1]), float(b_xpos[2]),
                    float(b_xquat[1]), float(b_xquat[2]), float(b_xquat[3]),
                    float(b_xquat[0])))

                # depth+rgb clouds — DEMAND-DRIVEN: rendering shares the GPU
                # with the camera server (macOS serializes GL across
                # processes), so only pay for it while a client is actually
                # watching. Two independent demands, one shared renderer:
                #   * merged 'live' cloud (head+front) → /ws/map subscribers
                #   * per-camera clouds (head/front/wrist_left/wrist_right) →
                #     /ws/stream subscribers, selectable per source
                # Renderers are torn down only when NEITHER wants a cloud.
                watching = bool(self.hub._subs)
                wanted = (self.stream_hub.sim_wanted_now()
                          if self.stream_hub is not None else [])
                if ctx["cam"] is not None and watching and \
                        t0 - last_cloud >= cloud_period:
                    last_cloud = t0
                    xyz, rgb = self._render_cloud(mujoco, ctx)
                    if xyz is not None:
                        self.hub.publish_cloud(ms.encode_cloud_chunks(xyz, rgb))
                if wanted:
                    self._render_source_clouds(mujoco, ctx, ms, wanted, t0,
                                               cloud_period)
                if not watching and not wanted and ctx.get("rgb_r") is not None:
                    self._close_renderers(ctx)
            except Exception as e:
                self.error = str(e)
                log.warning("mirror tick failed: %s", e)
                self._close_renderers(ctx)
                ctx = None
                self._stop.wait(1.0)
                continue
            dt = time.time() - t0
            self._stop.wait(max(0.0, period - dt))
        self._close_renderers(ctx)

    @staticmethod
    def _close_renderers(ctx):
        if not ctx:
            return
        for k in ("rgb_r", "dep_r"):
            r = ctx.get(k)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
                ctx[k] = None

    def _ensure_renderers(self, mujoco, ctx):
        if ctx["rgb_r"] is None:
            w, h = CFG.CLOUD_RES
            ctx["rgb_r"] = mujoco.Renderer(ctx["model"], h, w)
            ctx["dep_r"] = mujoco.Renderer(ctx["model"], h, w)
            ctx["dep_r"].enable_depth_rendering()

    def _cloud_one(self, ctx, cid):
        """World-frame (odom) cloud from ONE camera: depth back-projection
        coloured by the RGB render. The LiDAR rays are suppressed via
        ctx["vopt"] so they never contaminate the depth buffer."""
        w, h = CFG.CLOUD_RES
        data, vopt = ctx["data"], ctx["vopt"]
        ctx["dep_r"].update_scene(data, camera=cid, scene_option=vopt)
        depth = ctx["dep_r"].render()                    # (h,w) metres
        ctx["rgb_r"].update_scene(data, camera=cid, scene_option=vopt)
        color = ctx["rgb_r"].render()                    # (h,w,3) uint8

        fovy = math.radians(float(ctx["model"].cam_fovy[cid]))
        fy = (h / 2.0) / math.tan(fovy / 2.0)
        fx = fy
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        z = depth.astype(np.float32)
        valid = (z > 0.05) & (z < CFG.CLOUD_MAX_RANGE) & np.isfinite(z)
        if not valid.any():
            return None, None
        vs, us = np.nonzero(valid)
        zv = z[vs, us]
        # camera frame: x right, y up, looks along -z (MuJoCo convention)
        xc = (us.astype(np.float32) - cx) / fx * zv
        yc = -(vs.astype(np.float32) - cy) / fy * zv
        pc = np.stack([xc, yc, -zv], axis=1)             # (N,3) camera frame
        R = data.cam_xmat[cid].reshape(3, 3)
        p = data.cam_xpos[cid]
        return (pc @ R.T + p).astype(np.float32), color[vs, us].astype(np.uint8)

    def _render_cloud(self, mujoco, ctx):
        """Merged world-frame cloud from every configured depth camera
        (head + front base). ctx["cam"] is a list of camera ids."""
        self._ensure_renderers(mujoco, ctx)
        cams = ctx["cam"] if isinstance(ctx["cam"], (list, tuple)) else [ctx["cam"]]
        all_xyz, all_rgb = [], []
        for cid in cams:
            xyz, rgb = self._cloud_one(ctx, cid)
            if xyz is not None:
                all_xyz.append(xyz)
                all_rgb.append(rgb)
        if not all_xyz:
            return None, None
        return np.concatenate(all_xyz), np.concatenate(all_rgb)

    def _render_source_clouds(self, mujoco, ctx, ms, wanted, now, period):
        """Render each per-camera source a /ws/stream client wants, at CLOUD_HZ,
        and store it into the stream hub as cloud:<source>. Reuses the merged
        cloud's renderers (no extra GL context)."""
        by_source = ctx.get("cam_by_source") or {}
        due = [s for s in wanted if s in by_source
               and now - self._src_last.get(s, 0.0) >= period]
        if not due:
            return
        self._ensure_renderers(mujoco, ctx)
        for src in due:
            self._src_last[src] = now
            xyz, rgb = self._cloud_one(ctx, by_source[src])
            if xyz is None:
                continue
            # Per-camera clouds fit one 0x05 record (CLOUD_RES ≤ 25k pts <
            # MAP_CHUNK 60k → encode_cloud_chunks always returns one final=1
            # chunk); /ws/stream stores one blob per source key.
            chunks = ms.encode_cloud_chunks(xyz, rgb)
            if chunks:
                self.stream_hub.sim_store(src, chunks[0])


def _scene_bounds(model, data):
    """World-space (center, extent) of the whole scene after mj_forward.

    extent = the bounding-box diagonal over every geom's bounding sphere
    (planes excluded — they are infinite and would blow the bounds up). Used to
    open the free-orbit viewer's clip planes so nothing vanishes when zooming
    out, and to frame the whole world for the Fit button."""
    import mujoco
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo
    any_geom = False
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c = data.geom_xpos[g]
        r = float(model.geom_rbound[g]) or 0.05
        lo = np.minimum(lo, c - r)
        hi = np.maximum(hi, c + r)
        any_geom = True
    if not any_geom:
        return [0.0, 0.0, 0.6], 6.0
    center = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo))
    return [float(v) for v in center], max(extent, 2.0)


class SimViewSession(threading.Thread):
    """One browser MuJoCo-viewer window: renders the WHOLE simulation from a
    free orbit camera (the native viewer, in the browser). Per-connection
    thread — GL contexts are thread-bound — pacing at the client's fps and
    torn down with the socket, so idle windows cost nothing."""

    def __init__(self, loop, queue, state_file=None):
        super().__init__(daemon=True, name="scc-simview")
        self.loop = loop
        self.queue = queue
        # which WORLD to view: default = the live robot's state memmap; the
        # physics-preview instance passes its own (/tmp/mabel_sim_preview.f64)
        self.state_file = state_file or CFG.STATE_FILE
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # camera params (the view_robot.py defaults) + follow mode.
        # follow: "chassis" tracks the base, "head" tracks the head-cam frame,
        # None/False is a free orbit around `lookat`.
        self.params = {"az": 150.0, "el": -20.0, "dist": 3.2,
                       "lookat": [0.0, 0.0, 0.9], "follow": "chassis",
                       "w": 640, "h": 400, "fps": 15.0, "quality": 80}
        self._cmd = None                 # one-shot: "fit" | "recenter"
        self._meta_sent = False          # scene extent/center pushed to client?

    @staticmethod
    def _norm_follow(v):
        if v in (True, "chassis", "robot"):
            return "chassis"
        if v in ("head", "headcam", "head_cam"):
            return "head"
        return None                      # free orbit

    def update(self, p):
        with self._lock:
            for k in ("az", "el", "dist", "fps", "quality"):
                if k in p:
                    self.params[k] = float(p[k])
            if "lookat" in p and isinstance(p["lookat"], (list, tuple)):
                self.params["lookat"] = [float(v) for v in p["lookat"][:3]]
            if "follow" in p:
                self.params["follow"] = self._norm_follow(p["follow"])
            if "w" in p and "h" in p:
                self.params["w"] = max(160, min(1280, int(p["w"])))
                self.params["h"] = max(120, min(960, int(p["h"])))
            if p.get("cmd") in ("fit", "recenter"):
                self._cmd = p["cmd"]      # resolved in the render loop (needs data)

    def stop(self):
        self._stop.set()

    def _push(self, blob):
        def _on_loop():
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            try:
                self.queue.put_nowait(blob)
            except Exception:
                pass
        try:
            self.loop.call_soon_threadsafe(_on_loop)
        except RuntimeError:
            pass

    def _echo_cam(self, p, lookat):
        """After resolving a one-shot cmd (fit/recenter), tell the client the new
        camera state so its local orbit stays in sync (text frame)."""
        self._push(json.dumps({
            "type": "cam",
            "az": round(float(p["az"]), 2), "el": round(float(p["el"]), 2),
            "dist": round(float(p["dist"]), 3),
            "lookat": [round(float(v), 3) for v in lookat],
            "follow": p["follow"],
        }))

    def run(self):
        try:
            import cv2
            import mujoco
        except Exception as e:
            log.warning("simview unavailable: %s", e)
            return
        scene = None
        model = data = mm = renderer = None
        cam = None
        # hide the LiDAR rangefinder rays in the 3rd-person view (same reason as
        # the cloud/camera renders: the default option draws 360 laser beams)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        vopt.sitegroup[3] = 0
        size = (0, 0)
        chassis = 0
        while not self._stop.is_set():
            with self._lock:
                p = dict(self.params)
            try:
                # (re)attach on scene change, first run, or resize
                sidecar = self.state_file + ".scene"
                try:
                    with open(sidecar) as f:
                        cur = f.readline().strip()
                except OSError:
                    cur = None
                if model is None or (cur and cur != scene):
                    if renderer is not None:
                        renderer.close()
                        renderer = None
                    scene = cur or CFG.DEFAULT_SCENE
                    model = mujoco.MjModel.from_xml_path(scene)
                    data = mujoco.MjData(model)
                    mm = None
                    try:
                        chassis = model.body("mabel_chassis").id
                    except Exception:
                        chassis = 0
                    try:
                        head_body = model.body("head").id
                    except Exception:
                        head_body = chassis
                    # The robot MJCF PINS stat.extent small (stable wrist-camera
                    # clip planes); for this free-orbit WHOLE-SCENE viewer that
                    # makes the far clip = zfar*extent tiny, so distant geometry
                    # vanishes when you zoom out on a big scene. Recompute the
                    # real scene size (this is a throwaway model instance) and
                    # open the frustum so everything stays visible at any zoom.
                    mujoco.mj_forward(model, data)
                    scene_center, scene_ext = _scene_bounds(model, data)
                    model.stat.extent = scene_ext
                    model.stat.center[:] = scene_center
                    model.vis.map.znear = 0.004      # near = 0.004 * extent
                    model.vis.map.zfar = 24.0        # far  = 24 * extent (huge)
                    self._meta_sent = False
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                if mm is None:
                    need = (1 + model.nq) * 8
                    if not (os.path.exists(self.state_file)
                            and os.path.getsize(self.state_file) >= need):
                        self._stop.wait(0.5)
                        continue
                    mm = np.memmap(self.state_file, dtype=np.float64, mode="r",
                                   shape=(1 + model.nq,))
                if (p["w"], p["h"]) != size:
                    if renderer is not None:
                        renderer.close()
                    renderer = mujoco.Renderer(model, p["h"], p["w"])
                    size = (p["w"], p["h"])

                t_frame = time.monotonic()
                data.qpos[:] = mm[1:1 + model.nq]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)

                # push scene metadata once per scene so the client's Fit button
                # knows the world's extent + center
                if not self._meta_sent:
                    self._push(json.dumps({
                        "type": "scene",
                        "extent": round(float(model.stat.extent), 3),
                        "center": [round(float(v), 3) for v in model.stat.center],
                        "scene": os.path.splitext(os.path.basename(scene))[0]
                                 .replace("mabel_", ""),
                    }))
                    self._meta_sent = True

                # one-shot camera commands (need live data): resolve + echo back
                with self._lock:
                    cmd, self._cmd = self._cmd, None
                if cmd == "recenter":                # snap the orbit onto the robot
                    b = data.xpos[chassis]
                    p["follow"] = self.params["follow"] = "chassis"
                    p["dist"] = self.params["dist"] = 3.5
                    self._echo_cam(p, (b[0], b[1], b[2] + 0.55))
                elif cmd == "fit":                   # frame the ENTIRE scene
                    c = model.stat.center
                    fovy = math.radians(float(model.vis.global_.fovy) or 45.0)
                    d = float(model.stat.extent) / (2 * math.tan(fovy / 2)) * 1.15
                    p["follow"] = self.params["follow"] = None
                    p["lookat"] = self.params["lookat"] = [float(c[0]), float(c[1]), float(c[2])]
                    p["dist"] = self.params["dist"] = d
                    p["el"] = self.params["el"] = -28.0
                    self._echo_cam(p, p["lookat"])

                cam.azimuth = p["az"]
                cam.elevation = p["el"]
                cam.distance = p["dist"]
                if p["follow"] == "chassis":
                    b = data.xpos[chassis]
                    cam.lookat[:] = (b[0], b[1], b[2] + 0.55)
                elif p["follow"] == "head":
                    b = data.xpos[head_body]
                    cam.lookat[:] = (b[0], b[1], b[2])
                else:
                    cam.lookat[:] = p["lookat"]
                renderer.update_scene(data, camera=cam, scene_option=vopt)
                # streamed preview: shadows/reflections are pure cost at these
                # sizes (the camera-server made the same call) — big fps win.
                try:
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                except Exception:
                    pass
                rgb = renderer.render()
                ok, buf = cv2.imencode(
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, int(p["quality"])])
                if ok:
                    self._push(buf.tobytes())
            except Exception as e:
                log.debug("simview frame skipped: %s", e)
                model = None
                self._stop.wait(0.5)
                continue
            # pace to the requested fps ACCOUNTING for render+encode time —
            # the old unconditional wait made the ceiling 1/(work + period)
            # (~6-7 fps at any size); now the wait is only the remainder.
            elapsed = time.monotonic() - t_frame
            self._stop.wait(max(0.002, 1.0 / max(1.0, p["fps"]) - elapsed))
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
