"""
simview.py — the native MuJoCo viewer, in the browser, for the REAL robot.

SimViewSession is copied from simulation_control_center/server/simmirror.py
(repo rule: control centers copy, they do not cross-import). It renders the
canonical MJCF scene from a free orbit camera, posing it from the qpos memmap
at CFG.STATE_FILE. On the robot host the gateway (ROSPlant) mirrors the encoders into
that same memmap, so the "sim view" here is the MIRRORED REAL ROBOT — a
third-person 3D view no physical camera provides.

Degradation: if mujoco/cv2 are not importable the render thread exits after a
log line (the /ws/simview endpoint refuses the socket up front); if the memmap
is absent the thread idles at 2 Hz until it appears.
"""
import logging
import os
import threading

import numpy as np

from . import config as CFG

log = logging.getLogger("cc.simview")


def available():
    """Can this host render at all? (mujoco + cv2 importable)"""
    try:
        import cv2          # noqa: F401
        import mujoco       # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


class SimViewSession(threading.Thread):
    """One browser MuJoCo-viewer window: renders the WHOLE scene from a
    free orbit camera (the native viewer, in the browser). Per-connection
    thread — GL contexts are thread-bound — pacing at the client's fps and
    torn down with the socket, so idle windows cost nothing."""

    def __init__(self, loop, queue, state_file=None):
        super().__init__(daemon=True, name="cc-simview")
        self.loop = loop
        self.queue = queue
        # which WORLD to view: default = the live robot's state memmap; the
        # physics-preview instance passes its own (/tmp/mabel_sim_preview.f64)
        self.state_file = state_file or CFG.STATE_FILE
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # camera params (the view_robot.py defaults) + follow-the-robot mode
        self.params = {"az": 150.0, "el": -20.0, "dist": 3.2,
                       "lookat": [0.0, 0.0, 0.9], "follow": True,
                       "w": 640, "h": 400, "fps": 15.0, "quality": 80}

    def update(self, p):
        with self._lock:
            for k in ("az", "el", "dist", "fps", "quality"):
                if k in p:
                    self.params[k] = float(p[k])
            if "lookat" in p and isinstance(p["lookat"], (list, tuple)):
                self.params["lookat"] = [float(v) for v in p["lookat"][:3]]
            if "follow" in p:
                self.params["follow"] = bool(p["follow"])
            if "w" in p and "h" in p:
                self.params["w"] = max(160, min(1280, int(p["w"])))
                self.params["h"] = max(120, min(960, int(p["h"])))

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

                data.qpos[:] = mm[1:1 + model.nq]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                cam.azimuth = p["az"]
                cam.elevation = p["el"]
                cam.distance = p["dist"]
                if p["follow"]:
                    b = data.xpos[chassis]
                    cam.lookat[:] = (b[0], b[1], b[2] + 0.55)
                else:
                    cam.lookat[:] = p["lookat"]
                renderer.update_scene(data, camera=cam)
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
            self._stop.wait(1.0 / max(1.0, p["fps"]))
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
