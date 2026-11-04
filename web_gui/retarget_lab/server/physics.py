"""
Physics: the retargeter's output through NOVA's WHOLE-BODY CONTROLLER, in MuJoCo.

With the studio's Physics toggle on, the lab drives NOVA's own simulated stack —
NOVA/sim_run.sh: simulation_bridge (SimWorld, the one MuJoCo owner) under NOVA's
gateway (server/teleop_bridge.py) — exactly as a Vision Pro would: it connects to
the gateway's /teleop as a headset, sends the same control_mode (method retarget,
the lab's scope), and forwards every frame the lab receives (live) or replays.

Nothing is re-implemented, so it is NOVA's chain, not a model of it:
  frame -> engine_retarget.EngineRetargeter (the lab's own engine, parity 2e-15 rad)
        -> NOVA's caps (<= 80 % slew, PD error <= 0.3 rated/kp vs MEASURED, limits,
           sphere contact) -> plant.control_tick -> SimWorld: the real setpoint
           shaper (drivers/shaper.py, the calibration tree's slews) -> MuJoCo
           position servos + gravity; base twist -> swerve IK on simulated wheels;
           fingers through NOVA's hand retargeter, neck through its neck retargeter.
The gateway's robot_state (measured joints, chassis pose) comes back here and the
viewer draws it under the kinematic robot; `lag()` measures the difference.

The sim stack runs on NOVA's sim ports (ws 9190, http 8180, rpc 8788) — never the
real robot's — and is only stopped by this module if this module started it.

FAST ON (operator: "it waits forever till it brings up the view"). A cold start is
~10 s (MuJoCo + gateway + NOVA's solver child), and a world reset to the keyframe
then made NOVA's torque-capped arms travel 6-9 s to the operator's posture. So:
  * the stack is PRE-STARTED when the lab starts (MABEL_RTL_PHYSICS_PREWARM=0 to
    disable) and KEPT RUNNING when Physics is toggled off — on is a reconnect;
  * the sim world is SEEDED with the kinematic robot's current pose (RPC set_state)
    instead of reset to the keyframe — physics starts where kinematics is;
  * `phase` + `t_phase` tell the UI what it is waiting for, with elapsed time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

log = logging.getLogger("retarget_lab.physics")

NOVA_ROOT = Path(os.environ.get("MABEL_RTL_NOVA", str(Path(__file__).resolve().parents[4] / "NOVA")))
WS_PORT = int(os.environ.get("NOVA_WS_PORT", "9190"))
URL = f"ws://127.0.0.1:{WS_PORT}/teleop?client=visionpro&sid=retarget-lab-physics"
ARM = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(1, 8)]
LAG_JOINTS = ARM + ["torso"]


def _listening(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


class PhysicsLink:
    def __init__(self, lab):
        self.lab = lab
        self.on = False
        self.status = "off"
        self.proc = None                 # the sim stack, only if WE started it
        self.latest = None               # {"q", "base", "t_rx"}
        self._loop = None
        self._ws = None
        self._out = None                 # newest frame to forward (latest wins)
        self._ctl = deque()              # ordered control messages
        self._thread = None
        self.hist = deque(maxlen=900)    # (t, kin q, phys q) for lag estimation (~18 s)
        self.n_tx = self.n_rx = 0
        self.phase, self.t_phase = "off", time.time()   # off | starting | connecting | seeding | live | error
        self._stack_lock = threading.Lock()
        self._model = None               # the scene, for name -> qpos (seeding)

    def _set_phase(self, ph, status=None):
        self.phase, self.t_phase = ph, time.time()
        if status is not None:
            self.status = status

    def prewarm(self):
        """Start NOVA's sim stack in the background now, so Physics turns on in ~1 s."""
        threading.Thread(target=self._ensure_stack, daemon=True, name="rtl-physics-prewarm").start()

    def _ensure_stack(self):
        with self._stack_lock:
            self._load_model()                             # for seeding: ~1-2 s, never on the click path
            if _listening(WS_PORT):
                return True
            return self._start_stack()

    def _load_model(self):
        if self._model is None:
            try:
                import mujoco
                from . import config as CFG
                self._model = mujoco.MjModel.from_xml_path(CFG.SCENE)
            except Exception as e:                        # noqa: BLE001
                log.warning("physics: scene for seeding unavailable (%s)", e)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def enable(self, on):
        on = bool(on)
        if on == self.on:
            return self.snapshot()
        self.on = on
        if on:
            self.latest = None
            self._set_phase("starting" if not _listening(WS_PORT) else "connecting",
                            "starting NOVA sim (SimWorld + gateway)…" if not _listening(WS_PORT) else "connecting…")
            self._thread = threading.Thread(target=self._run, daemon=True, name="rtl-physics")
            self._thread.start()
        else:
            self._set_phase("off", "off")
            self.latest = None
            self.hist.clear()                            # the stack keeps running: the next on is a reconnect
        return self.snapshot()

    def shutdown(self):
        self.on = False
        self._closing = True                              # a start still in flight stops itself
        self._stop_stack()

    def _start_stack(self):
        if _listening(WS_PORT):
            return True
        script = NOVA_ROOT / "sim_run.sh"
        if not script.exists():
            self.status = f"NOVA checkout not found ({NOVA_ROOT}); set MABEL_RTL_NOVA"
            return False
        if self.on:
            self._set_phase("starting", "starting NOVA sim (SimWorld + gateway)…")
        logf = open(Path(os.environ.get("TMPDIR", "/tmp")) / "rtl_nova_sim.log", "w")
        # --no-viewer: sim_run.sh opens the MuJoCo window by default since 2026-10-18; the lab draws the physics itself
        self.proc = subprocess.Popen([str(script), "--no-studio", "--no-browser", "--no-panel", "--no-viewer"], cwd=str(NOVA_ROOT),
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        if getattr(self, "_closing", False):              # the lab quit while we were launching
            self._stop_stack()
            return False
        t0 = time.time()
        while time.time() - t0 < 90:                         # prewarm runs with on == False
            if self.proc.poll() is not None:
                self.status = "NOVA sim exited — see $TMPDIR/rtl_nova_sim.log"
                return False
            if _listening(WS_PORT):
                return True
            if getattr(self, "_closing", False):
                self._stop_stack()
                return False
            time.sleep(0.5)
        return _listening(WS_PORT)

    def _stop_stack(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGINT)              # sim_run.sh's own Ctrl-C path stops its children
                p.wait(timeout=10)
            except Exception:                                 # noqa: BLE001
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except Exception:                             # noqa: BLE001
                    pass

    # ── the link ──────────────────────────────────────────────────────────
    def _run(self):
        if not self._ensure_stack():
            self._set_phase("error", self.status if "not found" in self.status or "exited" in self.status
                            else "NOVA sim did not come up — see $TMPDIR/rtl_nova_sim.log")
            self.on = False
            return
        self._set_phase("connecting", "connecting to NOVA's gateway…")
        asyncio.run(self._session())

    async def _session(self):
        import websockets
        self._loop = asyncio.get_running_loop()
        while self.on:
            try:
                async with websockets.connect(URL, open_timeout=10, max_size=None) as ws:
                    hello = json.loads(await ws.recv())
                    who = (hello.get("payload") or {})
                    self._who = f"{who.get('name', 'NOVA sim')} ({who.get('kind', '?')})"
                    self._set_phase("seeding", "seeding the sim with the kinematic pose…")
                    self._ws = ws
                    await self._handshake(ws, who)
                    rx = asyncio.ensure_future(self._rx(ws))
                    try:
                        while self.on:
                            while self._ctl:
                                m = self._ctl.popleft()
                                await ws.send(json.dumps({"type": m[0], "payload": m[1]}))
                            f, self._out = self._out, None
                            if f is not None:
                                await ws.send(json.dumps({"type": "teleop_frame", "payload": f}))
                                self.n_tx += 1
                            await asyncio.sleep(0.004)
                    finally:
                        rx.cancel()
                        self._ws = None
            except Exception as e:                            # noqa: BLE001
                if self.on:
                    self.status = f"reconnecting ({type(e).__name__})"
                    await asyncio.sleep(1.0)

    async def _handshake(self, ws, who):
        send = lambda t, p: ws.send(json.dumps({"type": t, "payload": p}))
        if who.get("kind") == "sim":
            # start WHERE THE KINEMATIC ROBOT IS: no 6-9 s torque-capped travel from the keyframe
            ok = await asyncio.get_running_loop().run_in_executor(None, self._seed_sim)
            if not ok:
                await send("reset", {})                       # fallback: the keyframe
            await asyncio.sleep(0.15)
        await send("estop", {"active": False})                # energise (sim: torque on)
        await send("control_mode", {"method": "retarget", "controlType": "impedance", "region": "arm",
                                    "stiffness": 0.6, "scope": self.lab.pool.settings.get("scope", "local")})
        await send("recalibrate", {})
        # no lab reset here: the sim was just SEEDED with the kinematic pose (base
        # included), so both sides already start together and NOVA's engine engages
        # from that pose. (A reset here would snap the kinematic robot home and
        # recreate the very travel the seeding removes.)

    async def _rx(self, ws):
        async for raw in ws:
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            if m.get("type") != "robot_state":
                continue
            p = m.get("payload") or {}
            q = p.get("jointPositions") or {}
            self.latest = {"q": q, "base": p.get("base"), "t_rx": time.time()}
            if q and self.phase != "live":
                self._set_phase("live", f"live — {getattr(self, '_who', 'NOVA sim')}")
            self.n_rx += 1
            kin = self._kin_q()
            if kin and q:
                self.hist.append((time.time(), np.array([kin.get(n, np.nan) for n in LAG_JOINTS], float),
                                  np.array([q.get(n, np.nan) if q.get(n) is not None else np.nan for n in LAG_JOINTS], float)))

    def _kin_q(self):
        w = self.lab.pool.workers.get(self.lab.mirror)
        lat = w and w.get("latest")
        return lat[1]["q"] if lat else None

    # ── called by the lab ─────────────────────────────────────────────────
    def forward(self, payload):
        if self.on and payload is not None:
            self._out = payload

    def control(self, mtype, payload):
        if self.on:
            self._ctl.append((mtype, payload))

    def _seed_sim(self):
        """RPC set_state: the sim robot := the mirror retargeter's current pose (joints +
        chassis). Returns False when there is no kinematic pose yet or the RPC fails."""
        try:
            import math
            import sys
            import mujoco
            from . import config as CFG
            w = self.lab.pool.workers.get(self.lab.mirror)
            lat = w and w.get("latest")
            if not lat:
                return False
            rec = lat[1]
            self._load_model()
            m = self._model
            d = mujoco.MjData(m)
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
            if k >= 0:
                mujoco.mj_resetDataKeyframe(m, d, k)
            for j in range(m.njnt):
                nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").strip().lower().replace(" ", "_")
                if nm in rec["q"] and m.jnt_type[j] in (2, 3):
                    d.qpos[m.jnt_qposadr[j]] = float(rec["q"][nm])
            b = rec.get("base") or {}
            fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_root")
            if fj >= 0 and b.get("x") is not None:
                a = int(m.jnt_qposadr[fj])
                mujoco.mj_kinematics(m, d)
                ch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
                quat = b.get("quat")
                if quat and len(quat) == 4:
                    d.qpos[a + 3:a + 7] = quat
                    mujoco.mj_kinematics(m, d)
                d.qpos[a] += float(b["x"]) - float(d.xpos[ch][0])
                d.qpos[a + 1] += float(b["y"]) - float(d.xpos[ch][1])
            root = str(NOVA_ROOT)
            if root not in sys.path:
                sys.path.insert(0, root)
            from simulation_bridge.rpc import SimBridgeClient
            from simulation_bridge import config as SC
            cl = SimBridgeClient(port=SC.rpc_port(), timeout=2.0)
            cl.call("set_state", qpos=[float(v) for v in d.qpos])
            cl.close()
            return True
        except Exception as e:                                # noqa: BLE001
            log.warning("physics: seeding the sim failed (%s) — falling back to its keyframe", e)
            return False

    def on_reset(self):
        """Engines re-anchored (re-anchor / seek / load / new source): NOVA's world and
        gateway too — the sim back to its keyframe (the kinematic base restarts at the
        origin on a reset) and the engine re-calibrated, so both start from one pose."""
        self.hist.clear()
        if getattr(self, "_quiet_reset", False):
            return
        self.control("reset", {})
        self.control("estop", {"active": False})
        self.control("control_mode", {"method": "retarget", "scope": self.lab.pool.settings.get("scope", "local")})
        self.control("recalibrate", {})

    def lag(self):
        """Controller lag, physics vs kinematic (the mirror retargeter): per-joint
        tracking error now (rad), and the TIME lag that best aligns the two over the
        last ~12 s (cross-correlation of arm-joint motion, ms)."""
        if len(self.hist) < 30:
            return None
        t = np.array([h[0] for h in self.hist])
        K = np.vstack([h[1] for h in self.hist]); P = np.vstack([h[2] for h in self.hist])
        ok = np.isfinite(K).all(0) & np.isfinite(P).all(0)
        if not ok.any():
            return None
        K, P = K[:, ok], P[:, ok]
        err = np.abs(K - P)
        names = [n for n, o in zip(LAG_JOINTS, ok) if o]
        worst = int(np.argmax(err[-1]))
        # time lag: shift P back by k samples to best match K (motion only)
        dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.02
        kmax = min(int(2.5 / max(dt, 1e-3)), len(t) // 2)
        best, best_k = np.inf, 0
        for k in range(0, kmax + 1):                  # physics(t + k) ~ kinematic(t)
            e = np.mean((K[: len(t) - k] - P[k:]) ** 2)
            if e < best:
                best, best_k = e, k
        return {"err_now_max": float(err[-1].max()), "err_now_joint": names[worst],
                "err_rms": float(np.sqrt(np.mean(err ** 2))), "err_p90": float(np.percentile(err.max(1), 90)),
                "time_lag_ms": best_k * dt * 1e3, "residual_rms": float(np.sqrt(best)),
                "lag_at_cap": best_k >= kmax, "window_s": float(t[-1] - t[0])}

    def snapshot(self):
        lat = self.latest
        return {"on": self.on, "status": self.status, "phase": self.phase,
                "phase_s": round(time.time() - self.t_phase, 1), "tx": self.n_tx, "rx": self.n_rx,
                "q": lat["q"] if lat else None, "base": lat["base"] if lat else None,
                "age": (time.time() - lat["t_rx"]) if lat else None, "lag": self.lag() if self.on else None,
                "started_stack": self.proc is not None}
