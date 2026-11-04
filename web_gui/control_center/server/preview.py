"""
preview.py — PHYSICS-TRUE choreography preview in an isolated process.

Copied from simulation_control_center/server/preview.py (repo rule: copy, do
not cross-import) and hardened for the REAL robot host: pressing ▶ Preview on
the timeline must show what the robot would REALLY do — true MuJoCo physics,
not a kinematic ghost — WITHOUT moving the physical robot. The primitive is
the simulation_bridge's instance scoping: this service spawns an EPHEMERAL

    simulation_bridge --instance preview --scene <canonical scene>

with its own flock, its own state memmap (/tmp/mabel_sim_preview.f64), and its
own RPC port — the engine-level lock guarantees it can never touch a live sim
world, and it never talks to the hardware at all. The preview world is SEEDED
from the live state memmap: on the real robot the gateway (ROSPlant) mirrors
the encoders into that memmap, so long-horizon plans preview from the robot's
TRUE current pose.

Guarding: mujoco / simulation_bridge may be absent on the robot host (Jetson).
Imports are soft; PREVIEW.start() then raises with the reason and the API
answers {ok: false, error: ...} — the frontend degrades to a toast.

The timeline executor speaks the SAME wire dialect it uses live; wire() maps
the envelopes onto the preview engine's RPC:

    teleop_frame.navJoystick → set_twist   (inverse of the twist→stick map:
                                            vx = −ly·V, vy = −lx·V, w = rx·Ω)
    joint_command.joints      → set_targets (canonical ids → actuator names)
    reset / estop             → pass-through

Watch it in the browser MuJoCo viewer with /ws/simview?instance=preview.
Idle previews self-terminate after IDLE_TIMEOUT_S.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time

import numpy as np

from . import config as CFG

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(_REPO, "controller"), _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

# Soft imports — the real robot host may not ship the sim stack at all.
try:
    from simulation_bridge import config as SBCFG        # noqa: E402
    from simulation_bridge.rpc import SimBridgeClient    # noqa: E402
    from mabel import config as C                        # noqa: E402
    _IMPORT_ERROR = None
except Exception as _e:                                  # pragma: no cover
    SBCFG = SimBridgeClient = C = None
    _IMPORT_ERROR = f"physics preview unavailable on this host: {_e}"

log = logging.getLogger("cc.preview")

INSTANCE = "preview"
IDLE_TIMEOUT_S = 600.0


def _python():
    """Interpreter for the preview bridge: MABEL_PY, else the anaconda default
    (the Mac dev box), else whatever runs this server (the robot host)."""
    env = os.environ.get("MABEL_PY")
    if env:
        return env
    conda = os.path.expanduser("~/anaconda3/bin/python3")
    return conda if os.path.isfile(conda) else sys.executable


def _preview_state_file():
    return f"/tmp/mabel_sim_{INSTANCE}.f64"


class PreviewService:
    def __init__(self):
        self._proc = None
        self._rpc = None
        self._lock = threading.Lock()
        self.scene = None
        self.started_at = None
        self._last_activity = 0.0
        self._ios_to_mj = None
        self._log_path = "/tmp/mabel_cc_preview.log"
        threading.Thread(target=self._reaper, daemon=True,
                         name="cc-preview-reaper").start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self, scene=None, seed=True):
        if _IMPORT_ERROR:
            raise RuntimeError(_IMPORT_ERROR)
        with self._lock:
            live_scene = self._live_scene()
            scene = scene or live_scene
            if not os.path.isfile(scene):
                raise FileNotFoundError(f"scene not found: {scene}")
            if self.running() and self.scene == scene:
                self._last_activity = time.time()
                return self.status()
            self._stop_locked()
            env = dict(os.environ)
            env["MABEL_SIM_INSTANCE"] = INSTANCE
            env["MABEL_STATE_FILE"] = _preview_state_file()
            env.pop("MABEL_SIM_SENSORS", None)      # preview never needs lidar
            logf = open(self._log_path, "ab", buffering=0)
            logf.write(f"\n===== preview {time.strftime('%F %T')}: {scene}\n"
                       .encode())
            py = _python()
            if not os.path.isfile(py):
                raise FileNotFoundError(f"python not found: {py}")
            self._proc = subprocess.Popen(
                [py, "-u", "-m", "simulation_bridge.simulation_bridge",
                 "--instance", INSTANCE, "--scene", scene],
                cwd=_REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.scene = scene
            self.started_at = time.time()
            self._last_activity = time.time()
            # wait for the RPC to answer
            port = SBCFG.rpc_port(INSTANCE)
            self._rpc = SimBridgeClient(port=port, timeout=3.0)
            deadline = time.time() + 20.0
            hello = None
            while time.time() < deadline:
                try:
                    hello = self._rpc.call("hello")
                    break
                except Exception:
                    if self._proc.poll() is not None:
                        raise RuntimeError(
                            f"preview bridge died at launch — {self._log_path}")
                    time.sleep(0.3)
            if hello is None:
                self._stop_locked()
                raise RuntimeError("preview bridge never answered")
            seeded = False
            if seed and scene == live_scene:
                seeded = self._seed_from_live()
            log.info("preview up — %s (rpc :%d, seeded=%s)",
                     os.path.basename(scene), port, seeded)
            return self.status()

    def stop(self):
        with self._lock:
            self._stop_locked()
        return {"ok": True}

    def _stop_locked(self):
        if self._rpc is not None:
            try:
                self._rpc.close()
            except Exception:
                pass
            self._rpc = None
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self.scene = None

    def _reaper(self):
        while True:
            time.sleep(30)
            with self._lock:
                if (self.running() and
                        time.time() - self._last_activity > IDLE_TIMEOUT_S):
                    log.info("preview idle %.0f s — stopping", IDLE_TIMEOUT_S)
                    self._stop_locked()

    # ── seeding from the live robot ──────────────────────────────────────────
    def _live_scene(self):
        try:
            with open(CFG.STATE_FILE + ".scene") as f:
                return f.readline().strip() or CFG.DEFAULT_SCENE
        except OSError:
            return CFG.DEFAULT_SCENE

    def _seed_from_live(self):
        """Copy the LIVE robot's qpos into the preview world (works for sim
        AND for real — the gateway mirrors encoders into the same memmap)."""
        try:
            import mujoco
            m = mujoco.MjModel.from_xml_path(self.scene)
            need = 1 + m.nq
            if os.path.getsize(CFG.STATE_FILE) < need * 8:
                return False
            mm = np.memmap(CFG.STATE_FILE, dtype=np.float64, mode="r",
                           shape=(need,))
            qpos = [float(v) for v in mm[1:1 + m.nq]]
            del mm
            return bool(self._rpc.call("set_state", qpos=qpos))
        except Exception as e:
            log.warning("preview seed skipped: %s", e)
            return False

    def status(self):
        if _IMPORT_ERROR:
            return {"running": False, "error": _IMPORT_ERROR,
                    "instance": INSTANCE}
        return {
            "running": self.running(),
            "scene": self.scene,
            "instance": INSTANCE,
            "statePath": _preview_state_file(),
            "rpcPort": SBCFG.rpc_port(INSTANCE),
            "uptimeS": round(time.time() - self.started_at, 1)
            if self.started_at and self.running() else None,
            "idleTimeoutS": IDLE_TIMEOUT_S,
        }

    # ── the wire adapter (same envelopes the live executor sends) ───────────
    def _joint_map(self):
        if self._ios_to_mj is None:
            from mabel.wbc_runtime import _IOS_TO_MJ
            self._ios_to_mj = dict(_IOS_TO_MJ)
        return self._ios_to_mj

    def wire(self, msg_type, payload):
        """One wire envelope → the preview engine. Returns False when the
        preview isn't running (callers should surface that)."""
        rpc = self._rpc
        if rpc is None or not self.running():
            return False
        self._last_activity = time.time()
        p = payload or {}
        try:
            if msg_type == "teleop_frame":
                nj = p.get("navJoystick") or {}
                # inverse of the twist→stick mapping (front = −x body)
                vx = -float(nj.get("ly", 0.0)) * C.MAX_LIN_SPEED
                vy = -float(nj.get("lx", 0.0)) * C.MAX_LIN_SPEED
                w = float(nj.get("rx", 0.0)) * C.MAX_ANG_SPEED
                rpc.try_call("set_twist", vx=vx, vy=vy, wz=w)
            elif msg_type == "joint_command":
                jm = self._joint_map()
                targets = {}
                for jid, val in (p.get("joints") or {}).items():
                    name = jm.get(jid)
                    if name is not None:
                        targets[name] = float(val)
                if targets:
                    rpc.try_call("set_targets", targets=targets)
            elif msg_type == "reset":
                rpc.try_call("reset", pose=p.get("pose"))
            elif msg_type == "estop":
                rpc.try_call("set_estop", active=bool(p.get("active", True)))
            return True
        except Exception as e:
            log.warning("preview wire failed: %s", e)
            return False

    def snapshot(self):
        if self._rpc is None or not self.running():
            return None
        try:
            return self._rpc.call("state")
        except Exception:
            return None


PREVIEW = PreviewService()
