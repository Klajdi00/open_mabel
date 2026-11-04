"""
supervisor.py — owns the sim teleop stack's lifecycle so the studio can switch
scenes / relaunch with the desktop viewer.

The stack itself is server/start_teleop.sh (camera server + bridge + optional
mjpython viewer + Bonjour) — exactly what server/run.sh launches by
hand. This module spawns it in its own process group, tears it down cleanly
(the script's own EXIT trap kills its children), and relaunches with a new
`--scene`. Every websocket client of the bridge auto-reconnects, so a scene
switch is a ~5 s blip in the GUI, not a page reload.

Attach mode: if a bridge already answers on :9090 when we boot (someone ran
server/run.sh themselves), we do NOT double-launch — the studio attaches and
scene switching is disabled with a clear message.
"""
import logging
import os
import signal
import socket
import subprocess
import threading
import time

from . import config as CFG

log = logging.getLogger("scc.sim")

def port_open(port, host="127.0.0.1", timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class SimSupervisor:
    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self.scene = CFG.DEFAULT_SCENE
        self.viewer = False
        self.managed = False          # we own the stack (vs attached)
        self.started_at = None
        self._log_path = "/tmp/mabel_scc_sim.log"
        self._extra_args = []

    # ── boot policy ──────────────────────────────────────────────────────────
    def autostart(self, scene=None, viewer=False, extra_args=None):
        """Called once at backend startup. Spawns the stack unless one is
        already up (attach mode) or MABEL_SCC_MANAGE_SIM=0."""
        self.viewer = bool(viewer)
        self._extra_args = list(extra_args or [])
        if scene:
            self.scene = scene
        mode = str(CFG.MANAGE_SIM).lower()
        if mode in ("0", "no", "false", "off"):
            log.info("sim management disabled (MABEL_SCC_MANAGE_SIM=0)")
            return self.status()
        if mode == "auto" and port_open(CFG.TELEOP_PORT):
            log.info("bridge already up on :%d — attaching (not spawning)",
                     CFG.TELEOP_PORT)
            return self.status()
        self.start()
        return self.status()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, scene=None, viewer=None):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self.status()
            if scene:
                self.scene = scene
            if viewer is not None:
                self.viewer = bool(viewer)
            if not os.path.isfile(CFG.SIM_LAUNCHER):
                raise FileNotFoundError(f"sim launcher missing: {CFG.SIM_LAUNCHER}")
            base = os.path.basename(CFG.SIM_LAUNCHER)
            args = [CFG.SIM_LAUNCHER, "--scene", self.scene]
            if base == "sim_ros_server.sh":
                # Ubuntu ROS stack: viewer defaults ON there; no monitor flag
                args.append("--viewer" if self.viewer else "--no-viewer")
            else:
                # start_teleop.sh (pure sim): the studio replaces the tk monitor
                args.append("--no-monitor")
                if self.viewer:
                    args.append("--viewer")
            args += self._extra_args
            env = dict(os.environ)
            env.setdefault("MABEL_EPISODES_DIR", CFG.EPISODES_DIR)
            logf = open(self._log_path, "ab", buffering=0)
            logf.write(f"\n===== launch {time.strftime('%F %T')}: {' '.join(args)}\n"
                       .encode())
            self._proc = subprocess.Popen(
                args, cwd=CFG.SERVER_DIR, env=env,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True)          # own pgid → clean teardown
            self.managed = True
            self.started_at = time.time()
            log.info("sim stack launched (pid %d) scene=%s viewer=%s",
                     self._proc.pid, os.path.basename(self.scene), self.viewer)
            return self.status()

    def stop(self, timeout=12.0):
        with self._lock:
            p, self._proc = self._proc, None
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)   # trap runs cleanup
        except ProcessLookupError:
            return
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.2)
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        # start_teleop's pre-run cleanup frees the ports on relaunch, but give
        # the sockets a beat so the new bind doesn't race the old FIN_WAIT.
        time.sleep(0.5)

    def can_switch(self):
        """True when the studio can (re)launch the sim into another scene —
        either it already owns the stack, or a launcher exists to ADOPT an
        externally-started one (start_teleop.sh pre-cleans the ports on boot,
        so a fresh managed launch evicts the squatter). Only mock mode — no
        real bridge to replace — disables switching."""
        if not os.path.isfile(CFG.SIM_LAUNCHER):
            return False
        if self.managed:
            return True
        mode = str(CFG.MANAGE_SIM).lower()
        if mode in ("0", "no", "false", "off"):
            # explicit attach (--gui-only) → adoptable iff a real sim is up;
            # mock (--mock) has no bridge on :9090, so this stays False.
            return port_open(CFG.TELEOP_PORT)
        return True                       # auto/managed: always switchable

    def restart(self, scene=None, viewer=None):
        # Adopt-on-restart: when we don't own the stack we still relaunch a
        # fresh MANAGED one — start_teleop.sh's pre-run cleanup frees the ports
        # and takes over the external bridge. This is exactly run.sh's own
        # "take over any stale one" policy, so scene switching works even after
        # the studio attached to an externally-started sim.
        if not self.managed and not os.path.isfile(CFG.SIM_LAUNCHER):
            raise RuntimeError(
                "no sim launcher available — relaunch via "
                "simulation_control_center/run.sh")
        if not self.managed:
            log.info("adopting external sim → relaunching managed (scene=%s)",
                     os.path.basename(scene or self.scene))
        self.stop()                        # no-op when we don't own it
        return self.start(scene=scene, viewer=viewer)

    # ── status ───────────────────────────────────────────────────────────────
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def status(self):
        bridge_up = port_open(CFG.TELEOP_PORT)
        cam_up = port_open(CFG.VIDEO_PORT)
        return {
            "managed": self.managed,
            "running": self.running() if self.managed else bridge_up,
            "pid": self._proc.pid if self.running() else None,
            "scene": self.scene,
            "sceneName": os.path.splitext(os.path.basename(self.scene))[0]
                         .replace("mabel_", ""),
            "viewer": self.viewer,
            "bridgeUp": bridge_up,
            "cameraUp": cam_up,
            "uptimeS": round(time.time() - self.started_at, 1)
                       if self.started_at and self.running() else None,
            "log": self._log_path,
        }


SUPERVISOR = SimSupervisor()
