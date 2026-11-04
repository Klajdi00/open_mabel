#!/usr/bin/env python3
"""
sim_teleop_bridge.py — the unified MABEL teleop server, in one process.

ONE server for EVERY client — Vision Pro, iPhone app, web dashboard, and future
input devices (gamepads, joysticks, keyboards) — driving the MuJoCo simulation
(and, with the hardware path, the real robot) through one wire protocol:

  any client ──pose / sticks / joint cmds (WebSocket /teleop:9090)──▶ Retargeter ▶ WholeBodyIK ▶ MuJoCo
  any client ◀──stereo head + wrist cameras (MJPEG :8080)──────────── offscreen render of the sim

No webcams, no ZMQ publisher. Clients stream TeleopFrames to ws://host:9090/teleop
and pull MJPEG from http://host:8080/camera/* — this server sources those cameras
from the robot's own simulated ZED + wrist cameras and drives the simulated robot
with the operator's input.

Threading (macOS-safe)
----------------------
  • Main thread : MuJoCo physics + offscreen camera rendering + retarget/IK.
                  The GL context lives here (glfw wants the main thread on macOS).
  • Net  thread : asyncio loop hosting the WebSocket teleop server AND the MJPEG
                  HTTP server. The render loop pushes JPEGs across via PushCapture.

Cameras served (per-eye stereo head + wrists)
---------------------------------------------
  /camera/head_left/stream.mjpg   ← zed_left    (left eye)
  /camera/head_right/stream.mjpg  ← zed_right   (right eye)
  /camera/main/stream.mjpg        ← zed_left    (mono fallback for the flat panel)
  /camera/wrist_left/stream.mjpg  ← l_wrist_cam
  /camera/wrist_right/stream.mjpg ← r_wrist_cam

Usage
-----
  cd server
  python sim_teleop_bridge.py                 # live: wait for any teleop client
  python sim_teleop_bridge.py --mock          # no client — synthetic hands
  python sim_teleop_bridge.py --scene ../simulation/mabel_mujoco/scenes/mabel_home.xml
  python sim_teleop_bridge.py --head-res 960x540 --cam-fps 20

Calibration (clutch): the relative hand mapping anchors automatically on the first
tracked frame. Press ENTER in this terminal to re-anchor at the current pose with
no jump (equivalent to the 'C' key in teleop_vision_pro.py).
"""
from __future__ import annotations

import json
import os
import platform
# GL backend must be chosen before mujoco is imported. glfw = GPU offscreen.
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import asyncio
import atexit
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict as _asdict

# NOTE: cv2 is imported LAZILY inside the `if args.cameras:` block below — it is
# only needed to JPEG-encode the SIM's MuJoCo-rendered cameras. The real robot
# (MABEL_DISPATCH_TARGET=real:ros) never renders cameras here (real_camera_server
# / the camera_hal node own them), so the real bridge must NOT hard-depend on cv2
# — some conda envs ship an opencv with a numpy-ABI clash that would otherwise
# crash the bridge (and with it the /map mesh stream) before it ever serves.
import numpy as np

# ── Make the shared `mabel` controller package importable ────────────────────
# The controller lives in one place — ~/open_mabel/controller — and is shared by the
# sim viewer, the iOS app, and the Vision Pro app, so the control logic can
# never drift between clients again. This server sits at ~/open_mabel/server, so the
# controller is one directory up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONTROLLER = os.path.normpath(os.path.join(_HERE, "..", "controller"))
if _CONTROLLER not in sys.path:
    sys.path.insert(0, _CONTROLLER)
# The avp_teleop control stack (teleop.TeleopEngine — the apps' main retargeting)
# lives in the sim scripts dir; put it on the path so the bridge can drive it.
_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "simulation", "mabel_mujoco", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import mujoco

# The WHOLE control stack — retarget, whole-body IK, swerve, neck, the torque
# engine, the clutch and the tracking-loss fail-safe — lives behind this one
# facade. The bridge never touches the control internals: it decodes the wire,
# feeds the facade, and steps physics. Editing the controller never touches the
# bridge or the three apps. See controller/mabel/facade.py.
from mabel import config as C


# Pre-validated random-spawn POSE POOL: computed in a warm subprocess (one
# model load, several seeds) at boot + refilled after each use, so the reset
# button is INSTANT instead of waiting ~10-25 s for a cold picker. The picker
# stays subprocess-isolated (see _safe_spawn_fn).
_spawn_pool: list = []
_spawn_pool_busy = threading.Event()


def _refill_spawn_pool(scene_path, n=4):
    if _spawn_pool_busy.is_set():
        return
    _spawn_pool_busy.set()

    def _work():
        try:
            import subprocess
            _WCG = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "web_gui"))
            code = (
                "import sys, json\n"
                f"sys.path.insert(0, {_WCG!r})\n"
                "from simulation_control_center.server import scenes\n"
                "poses = []\n"
                f"for i in range({n}):\n"
                f"    p = scenes.safe_spawn({scene_path!r}, seed=None)\n"
                "    if p: poses.append(p)\n"
                "print(json.dumps(poses))\n"
            )
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True, timeout=90)
            if out.returncode == 0:
                poses = json.loads(out.stdout.strip().splitlines()[-1])
                _spawn_pool.extend(poses)
                del _spawn_pool[:-8]          # keep the freshest few
                log.info("spawn pool: %d validated poses ready", len(_spawn_pool))
            else:
                log.warning("spawn pool refill failed: %s", out.stderr.strip()[-160:])
        except Exception as e:                                  # noqa: BLE001
            log.warning("spawn pool refill failed: %s", e)
        finally:
            _spawn_pool_busy.clear()

    threading.Thread(target=_work, name="spawn-pool", daemon=True).start()


def _pop_spawn_pose(avoid_xy):
    """Pooled pose FARTHEST from the current base (fresh random feel without
    the wait); None when the pool is dry."""
    if not _spawn_pool:
        return None
    ax, ay = avoid_xy or (0.0, 0.0)
    best = max(_spawn_pool, key=lambda p: (p["x"] - ax) ** 2 + (p["y"] - ay) ** 2)
    _spawn_pool.remove(best)
    return best


def _safe_spawn_fn(scene_path, avoid_xy=None):
    """SCC's validated open-pose picker (web_gui/simulation_control_
    center/server/scenes.py) — run in an ISOLATED SUBPROCESS. In-process it
    shared MuJoCo's global error state with the bridge's own mj_ray lidar on
    the control thread; concurrent calls corrupted it ("mj_ray: vector length
    is too small") and took the whole control loop — and the sim — down
    (2026-07-11 random-reset crash). A subprocess owns its own interpreter +
    MuJoCo, so the picker can never touch the live sim."""
    import subprocess
    _WCG = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "web_gui"))
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {_WCG!r})\n"
        "from simulation_control_center.server import scenes\n"
        f"pose = scenes.safe_spawn({scene_path!r}, avoid_xy={tuple(avoid_xy) if avoid_xy else None!r})\n"
        "print(json.dumps(pose))\n"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"safe_spawn subprocess failed: {out.stderr.strip()[-200:]}")
    pose = json.loads(out.stdout.strip().splitlines()[-1])
    return pose
from mabel import MabelTeleop, OperatorPose, app_joystick
from curation_recorder import CurationRecorder
from record_pump import RecordPump
from data_collection import BagRecorder
from mabel.wbc_runtime import _IOS_TO_MJ    # ios id -> mj name, for the monitor tap
from mabel import mock_avp
# Demo-day animations: record a human demonstration as CONTROLLER INPUTS, replay
# it back through the same controller. Deliberately a sibling package, not part
# of the bridge: recording and replaying a clip is useful from a script with no
# websocket in sight (animation_demo/scripts/), and the format has to outlive
# this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from animation_demo import DemoPlayer, DemoRecorder, delete_clip  # noqa: E402
from animation_demo import list_clips as _list_demo_clips         # noqa: E402
from animation_demo import load_clip as _load_demo_clip           # noqa: E402
from animation_demo.clip import safe_name as _demo_safe_name      # noqa: E402

# ── Server-side networking ───────────────────────────────────────────────────
from config import CameraSource
from wire_protocol import WireCodec, WireMessageType, LEGACY_MODES
from mjpeg_server import MJPEGServer, PushCapture
from monitor_hub import MONITOR
from map_stream import (MAP_HUB, start_map_stream, latest_scan,
                        latest_motor_health, encode_odom, encode_cloud_chunks)
from robot_command import RobotCommand, BaseTwist
from plant import make_plant
import data_collection
import data_streams
from audio_streams import AUDIO
from telemetry_stream import TELEMETRY

import console
log = console.setup("bridge")

# WebSocket path a command-monitor (monitor_gui.py / browser) subscribes on. It
# shares the teleop port (:9090) so no extra listener or process is needed.
MONITOR_PATH = "/monitor"
MAP_PATH = "/map"          # live SLAM mesh + odom stream (see server/map_stream.py)
# Dedicated port for the SLAM map stream, ISOLATED from teleop (9090) and cameras
# (8080) so a multi-MB mesh never competes with the 20 Hz control loop or video.
# The browser map connects here; iOS can use either this or 9090/map. 0 disables.
MAP_PORT = int(os.getenv("MABEL_MAP_PORT", "9092"))
# Dedicated port for the RAW LiDAR sweep (SCAN 0x06 frames only, ~1.5 KB @ 5 Hz):
# a minimal teleop device renders the live 360° "radar" without speaking the full
# map protocol. The same frames also flow INLINE on the /map stream (9090/9092),
# so full clients (iOS / Vision Pro / Control Center) need no extra socket.
# 0 disables the extra listener.
LIDAR_PORT = int(os.getenv("MABEL_LIDAR_PORT", "9093"))

# Camera roster shared with sim_camera_server.py (the dedicated render process).
from sim_cameras import CAMERAS, HEAD_W, HEAD_H

HEAD_FOVY_DEFAULT = 83.3        # MMlove stereo module: 112° diagonal @4:3 (mabel_full.xml)

# ── Shared state between the net thread and the sim/render loop ──────────────
# Per-connected-client teleop state (keyed by id(ws)). The unified server serves
# many clients at once (iOS, a second copy of the app, Vision Pro, keyboard), so
# EACH client's frame AND control spec are tracked separately. The sim loop follows
# the single "active driver" — the client that most recently sent MEANINGFUL input
# (a stick deflection, a joint command, or a control-mode change). This is why an
# idle second app can neither thrash the base (frames) NOR clobber the controller
# (a stale `navigation` spec resetting the operator's `arms` mode).
_client_frames: dict = {}        # id(ws) -> (recv_time, TeleopFrame)
# Operator-overlay (retarget debug): clients opt in via the operator_points
# message; the control loop refreshes the record from the engine each pass.
_client_overlay: dict = {}       # id(ws) -> bool (send the VP tracking overlay)
_op_overlay_rec: dict = {"rec": None}   # newest engine.last_operator_points
_client_specs: dict = {}         # id(ws) -> control-spec dict
_client_cmd: dict = {}           # id(ws) -> wall time of last DISCRETE command
_client_stream: dict = {}        # id(ws) -> wall time of last hand-tracking frame
_DEFAULT_SPEC = {"method": "navigation", "type": "impedance", "region": "arm",
                 "stiffness": 0.0,
                 # mobile base OUT of the follow loop until asked (see scope)
                 "scope": "local",
                 # Which of the operator's body parts drive the robot. All on =
                 # the classic whole-body behaviour; a channel switched off is
                 # not computed and EASES to neutral (engine ChannelPark).
                 "tracking": {"neck": True, "arms": True, "hands": True,
                              "torso": True, "lift": True},
                 # Operator switches. NOTE the snake_case: this dict is splatted
                 # into MabelTeleop.set_control_spec, so its keys are that
                 # function's parameter names, not the wire's camelCase.
                 "gravity_comp": True,
                 "whole_body_retarget": False}

# Last control-mode spec PER DEVICE NAME, persisted across gateway restarts
# (/tmp file). Client specs are keyed by connection (id(ws)), so every gateway
# restart used to reset every app to `navigation` — and the Vision Pro sets its
# mode ONCE when the operator taps a tab, so after a restart its joint sliders
# were silently ignored until the operator re-toggled the tab (2026-07-29
# "cannot move the arms or torso from the VP"). On connect, a device's
# remembered spec is re-seeded; restoring a spec alone moves NOTHING (motion
# still needs the e-stop release + explicit commands).
_DEVICE_SPECS_PATH = "/tmp/mabel_client_specs.json"
try:
    with open(_DEVICE_SPECS_PATH) as _f:
        _device_specs: dict = {str(k): dict(v) for k, v in json.load(_f).items()
                               if isinstance(v, dict)}
except Exception:
    _device_specs = {}


def _remember_device_spec(device: dict, spec: dict) -> None:
    name = (device or {}).get("device")
    if not name:
        return
    _device_specs[str(name)] = dict(spec)
    try:
        with open(_DEVICE_SPECS_PATH, "w") as f:
            json.dump(_device_specs, f)
    except OSError:
        pass                                  # best-effort; never disturb teleop
CMD_HOLD_S = 4.0                 # a discrete command holds the driver token this long
STREAM_HOLD_S = 1.0              # a passive tracking stream (hands/head) must be FRESH


def _head_active(transform) -> bool:
    """True only for a REAL, non-identity head pose — actual head tracking. Every
    client streams a `head`, but an IDLE one sends the IDENTITY matrix (the web sends
    it literally; the iPhone sends it as trackingState 'notAvailable'). We must NOT let
    that count as 'driving', or an idle app forever owns the driver token. A genuinely
    tracked head has non-zero translation / off-diagonal rotation."""
    m = transform.get("matrix") if isinstance(transform, dict) else transform
    if not m or len(m) < 16:
        return False
    trans = abs(m[3]) + abs(m[7]) + abs(m[11])                       # last column
    rot = abs(m[1]) + abs(m[2]) + abs(m[4]) + abs(m[6]) + abs(m[8]) + abs(m[9])
    return trans > 0.02 or rot > 0.02


def _active_client():
    """id of the client ACTIVELY driving right now, or None (everyone is an OBSERVER).
    The frame + control spec are read from this ONE client, so a second app cannot
    fight it. A client earns the token ONLY by sending meaningful input — there is NO
    "any connected client drives" fallback: with no recent command or live tracking,
    NOBODY drives (this is the fix for "the GUI shows the idle/just-reconnected iPhone
    as driving").

    Copies the dicts first: this runs on the sim thread while the net thread adds/
    removes clients — iterating the live dict would 'change size during iteration'."""
    now = time.time()
    # STRONG claims: discrete commands (sticks, joint targets, Soft pushes, mode sets)
    # — hold the token CMD_HOLD_S past the last input.
    cmds = [(i, t) for i, t in list(_client_cmd.items()) if now - t < CMD_HOLD_S]
    if cmds:
        return max(cmds, key=lambda kv: kv[1])[0]
    # WEAK claims: live passive tracking (headset hands, phone head) — must be FRESH
    # (within STREAM_HOLD_S), so a client that STOPPED tracking releases the token and
    # becomes an observer instead of owning it forever.
    streams = [(i, t) for i, t in list(_client_stream.items()) if now - t < STREAM_HOLD_S]
    if streams:
        return max(streams, key=lambda kv: kv[1])[0]
    # SINGLE-CLIENT fallback: with exactly ONE connected teleop client there is no token
    # to fight over — let it drive. A centered stick already commands ZERO base, so a
    # truly idle lone client is harmless, while this fixes the iPhone "drives once then
    # dead until reconnect": the strict token otherwise dropped a lone client to OBSERVER
    # after CMD_HOLD_S with no >0.04 deflection, so pose was never read again.
    fids = list(_client_frames.keys())
    if len(fids) == 1:
        return fids[0]
    return None                                # 2+ clients, none active → all observers


_recalibrate = threading.Event()              # set by ENTER or first frame
_reset = threading.Event()                    # set by the headset's Reset button
_reset_pose = [None]          # optional {x,y,yaw} rider on RESET (relocate-reset)
_scene_path = [None]          # absolute scene XML (set in main; random reset needs it)
_estop = {"active": False}                     # operator emergency stop (any client)
# TORQUE POLICY. When MABEL_TORQUE_AUTO_ON=1 the operator directive applies: torque
# is ALWAYS ON unless the e-stop is engaged — intent = (NOT estop), evaluated
# continuously, energising at bring-up (hold-then-enable, no lurch) with no operator
# action. DEFAULT is OFF (intent None): the robot comes up LIMP and NEVER auto-
# energises — torque then only follows explicit e-stop edges from the app. This is the
# SAFE default so bringing the server up (e.g. just to let iOS connect) does not move
# the arms/hands. Flip the env on only when you want torque-follows-estop always-on.
_AUTO_TORQUE_ON = os.getenv("MABEL_TORQUE_AUTO_ON", "0").strip() in ("1", "true", "yes", "on")
_hand_torque_req = {"pending": False, "target": _AUTO_TORQUE_ON,
                    "intent": (True if _AUTO_TORQUE_ON else None), "t_reconcile": 0.0,
                    # last target actually APPLIED to the plant. Guards the hold-at-
                    # measured re-adopt against re-running on an already-energised
                    # robot (the shoulder ratchet — see the ESTOP handler).
                    "applied": None}
_clients = {"n": 0}                           # connected teleop-client count


def _resume_hold(plant, teleop):
    """RESUME (e-stop release / any torque re-enable): command = MEASURED pose.

    While e-stopped the joints are LIMP and get moved — by hand, by gravity, by a
    collision. `data.ctrl` + the app's joint_targets still hold the PRE-e-stop
    command, so re-energising used to SNAP every joint back to it at full servo
    authority: a violent lurch, and exactly what an e-stop must never do. (The
    hardware's own hold-then-enable can't prevent it — the bridge latches
    goal:=present, then our next command frame overwrites it with the stale target.)

    So: mirror the freshest encoders in, then adopt that measured pose as the
    command (facade.resume_hold reseeds BOTH the servo targets and the app's direct
    overrides, and re-anchors the clutch). Torque then engages on a command that
    equals reality → the robot holds still. Best-effort: a failure here must never
    block the e-stop path itself."""
    try:
        plant.read_state()                    # freshest measured joints (real: encoders)
    except Exception:
        pass
    try:
        teleop.resume_hold()
    except Exception as e:                    # noqa: BLE001
        log.warning("resume: hold-at-measured failed (%s) — re-enabling torque anyway; "
                    "the hardware's hold-then-enable still applies.", e)
# Per-subsystem torque-enable state, surfaced on the wire (:9090 status echo +
# the :9600 telemetry status). Hands follow the e-stop edge (set where the loop
# applies plant.set_hand_torque); arms/base are null until the real HAL's
# latched /​*/torque_state topics are plumbed through (P4).
_torque_state = {"left_hand": False, "right_hand": False, "arms": None,
                 "neck": None, "base": None}
# Has the 2 s torque reconcile ALREADY re-adopted the measured pose for this limp
# episode? _resume_hold must run once per LIMP->ENERGISED transition, never per
# poll — repeating it walks the target onto the servo error (the J1/J2/J4 ratchet).
# Cleared when the subsystem's latched torque_state agrees with intent again.
_reconcile_resumed = {"arms": False, "hands": False, "neck": False}
# Selected (NOT running) policy / animation — placeholder echo for the apps'
# Policies tab. A runner would plug into the RobotCommand seam; until then the
# server just remembers + broadcasts the selection so every client agrees.
_policy_state = {"selected": None, "running": False}
_animation_state = {"selected": None, "playing": False}
# ── Animation demos (animation_demo/) ──────────────────────────────────────
# The net thread only QUEUES; the CONTROL thread owns the recorder and the
# player, because both must be stepped in lockstep with the command state they
# read and write. Same discipline as _record_pending — a websocket handler that
# reached into `_joint_cmd` would race the loop that consumes it.
_demo_pending: list = []          # [("arm"|"stop"|"play"|"halt"|"delete", payload)]
# Mirrored into robot_state.animation so every client agrees on what is
# recording / playing without polling a second endpoint.
_demo_state = {"recording": False, "armed": False, "countdown": 0.0,
               "name": None, "frames": 0, "elapsed": 0.0,
               "playing": False, "phase": None, "rep": 0, "repeats": 0,
               "frac": 0.0, "clips": []}
# Mic mute intent, echoed in robot_state so every client agrees on who is hot.
# The PCM never passes through here — it rides :9610 (see audio_streams.py).
_audio_state = {"micEnabled": True}
# The plant, for the NETWORK thread. It is owned and built by the control thread
# (make_plant runs long after start_network_thread spawns), so the network side
# reaches it through this one-slot holder — the same shape as MODULES_LIVE. Do NOT
# close over `plant` in a network-thread handler: it is not in scope there, and
# the NameError takes the whole thread down, :9090 with it (2026-07-16).
_PLANT = [None]


def _motor_health_now():
    """Per-joint motor health for robot_state.motorHealth: the map plane's merged
    */motor_health cache when that plane runs in THIS process, else the ROS plant's own
    cache (server/plant.py::ROSPlant.motor_health). With perception EXTERNAL (thor since
    2026-07-13, the Mac always) the map plane is not here and the first source stays None
    forever — which is why the apps' motorHealth went missing (found 2026-10-01)."""
    mh = latest_motor_health()
    if mh is None:
        fn = getattr(_PLANT[0], "motor_health", None)
        if fn is not None:
            mh = fn()
    return mh

_telem_next = [0.0]                 # control-loop rate limiter for :9600 proprio/control


def _battery_state() -> dict:
    """Battery block for robot_state / telemetry. No battery source exists yet
    in EITHER world (sim has none by definition; the real robot has no BMS
    wired), so this is an honest placeholder: legacy keys keep old app builds'
    gauges working, `available:false` + `reason` tell new UIs to show n/a."""
    return {"percentage": 1.0, "voltage": 50.0, "isCharging": False,
            "available": False,
            "reason": "no battery monitor wired" if _IS_REAL else "not applicable (sim)"}


_ASSET_CACHE: dict[str, tuple[float, list]] = {}   # kind -> (scan time, names)


def _list_assets(kind: str, env: str, default_dir: str,
                 exts: tuple[str, ...]) -> list[str]:
    """Enumerate selectable policies/animations from a directory (cached 10 s —
    never scanned on the hot path). Missing dir → empty list, not an error."""
    now = time.time()
    hit = _ASSET_CACHE.get(kind)
    if hit and now - hit[0] < 10.0:
        return hit[1]
    root = os.path.expanduser(os.getenv(env, default_dir))
    names: list[str] = []
    try:
        for e in sorted(os.listdir(root)):
            p = os.path.join(root, e)
            if os.path.isdir(p) or e.lower().endswith(exts):
                names.append(e)
    except OSError:
        pass
    _ASSET_CACHE[kind] = (now, names)
    return names


def _list_policies() -> list[str]:
    return _list_assets("policy", "MABEL_POLICY_DIR", "~/Desktop/mabel_policies",
                        (".onnx", ".pth", ".pt", ".safetensors", ".ckpt"))


def _list_animations() -> list[str]:
    return _list_assets("animation", "MABEL_ANIMATION_DIR", "~/Desktop/mabel_animations",
                        (".yaml", ".yml", ".json", ".npz"))


# ── :9600 telemetry providers (run on the NETWORK loop — everything they read
# is a reference-swapped module dict, so no control-thread work and no locks).
def _telem_hello() -> dict:
    return {"robot": {"name": _robot_name[0], "kind": _hello_info.get("kind"),
                      "server": _hello_info.get("server"),
                      "version": _hello_info.get("version")},
            "schema": 1, "ports": data_streams.port_map()}


def _telem_status() -> dict:
    t = _robot_state.get("t", 0.0)
    return {"estop": _estop["active"],
            "controlMode": dict(_control_spec),
            "latency": {"cmdStalenessMs": round(_lat_ema[0], 1),
                        "stateAgeMs": round(max(0.0, (time.time() - t) * 1000.0), 1) if t else None},
            "battery": _battery_state(),
            "torque": _torque_report(),
            "policy": dict(_policy_state),
            "animation": dict(_animation_state),
            "driver": {"clients": _clients["n"]},
            "recording": _robot_state.get("recording", {}),
            "dataCollection": _robot_state.get("dataCollection", {})}


_bags_cache = {"t": 0.0, "d": None}    # 10 s TTL — a dir scan at 1 Hz is wasteful


def _telem_collection() -> dict:
    rec = _robot_state.get("recording", {}) or {}
    counts = rec.get("taskCounts") or {}
    now = time.time()
    dc = _robot_state.get("dataCollection", {}) or {}
    if _bags_cache["d"] is None or now - _bags_cache["t"] > 10.0 or dc.get("active"):
        _bags_cache.update(t=now, d=data_collection.list_sessions())
    bags = dict(_bags_cache["d"])
    bags["active"] = bool(dc.get("active"))
    return {"episodes": {"dir": rec.get("dir"),
                         "tasks": [{"name": t, "count": counts.get(t, 0)}
                                   for t in (rec.get("tasks") or [])],
                         "total": rec.get("total", 0)},
            "bags": bags}


def _telem_network(ws) -> dict:
    info = _srv_info_ref["d"] or {}
    net = _network_info(ws, "/telemetry")
    return {"hostname": info.get("host"),
            "ips": {"lan": info.get("lanIp"), "tailscale": info.get("tailscaleIp")},
            "transport": net.get("transport"),   # THIS client's classified route
            "secure": net.get("secure"),
            "portMap": data_streams.port_map(),
            "tokenGated": bool(info.get("tokenGated"))}


def _telem_lidar() -> dict | None:
    return _robot_state.get("lidar")
# Standardized-actuation tap (RobotCommand → monitor): throttle + sequence. The
# dispatch target names WHERE this command goes; the sim bridge drives MuJoCo, a
# hardware bridge would set "real:ros" (override via env for a real-robot run).
_act_seq = [0]
_last_act = [0.0]
_DISPATCH_TARGET = os.getenv("MABEL_DISPATCH_TARGET", "sim:mujoco")
# Is THIS bridge driving the REAL robot (MABEL_DISPATCH_TARGET=real:ros, as
# real_run.sh sets) or the MuJoCo sim? Clients read `kind` ("sim"/"real") and
# `robot` ("MABEL"; NOVA's gateway says "NOVA") from the hello below. Older app
# builds still fall back to whether `server` contains "sim", so the real bridge
# keeps an identity WITHOUT "sim" — otherwise those badge a real robot as "Sim".
_IS_REAL = _DISPATCH_TARGET.startswith("real")
# MABEL_PERCEPTION_EXTERNAL=1: a SEPARATE process (perception_server.py) owns the
# heavy map/localization/cloud/slam planes, so this gateway must NOT create them —
# it keeps ONLY the real-time joint planes (robot_state :9090, proprio telemetry
# :9600, raw LiDAR :9093, video). This is THE fix for the GIL-thrashing that
# starved the joint streams to ~0.5 Hz: with the map/cloud rclpy executors gone
# from this process, the state stream keeps the interpreter to itself. See
# perception_server.py for the full rationale.
_PERCEPTION_EXTERNAL = os.getenv("MABEL_PERCEPTION_EXTERNAL", "0").strip() in (
    "1", "true", "yes", "on")
_NS_ACTIVITY = None     # macOS App-Nap exemption handle (held for the process lifetime)
# Server identity + capabilities sent in the `hello`. The sim loop fills in the
# head-camera intrinsics (after any --head-fov override) BEFORE networking
# starts, so the headset can size its wrap-around panel from the real fov.
_hello_info = {"server": "mabel_real_bridge" if _IS_REAL else "mabel_sim_bridge",
               "kind": "real" if _IS_REAL else "sim",
               # WHICH robot, as data. NOVA speaks this same wire (and advertises the same
               # Bonjour type), so clients read `robot` + `kind` — never guess from `server`
               # or a renamable name. NOVA's gateway sends robot "NOVA" and its live `hands`.
               "robot": "MABEL",
               "robotId": "mabel-real" if _IS_REAL else "mabel-sim",
               "version": "2.2-handik",
               # Endpoint discovery for the 96xx data plane (additive; apps that
               # don't know the key ignore it). See docs/data_streams.md.
               "ports": data_streams.port_map()}
if not _IS_REAL:
    # The sim's model carries MABEL's own ORCA v1 hands. On the real robot the HAL picks
    # v1/v2 per side by adapter serial (mabel_hw/drivers/hands.py) and this gateway may
    # not import mabel_hw, so `hands` is omitted there and clients default MABEL to v1.
    _hello_info["hands"] = {"left": "v1", "right": "v1"}
# User-set robot display NAME — the same one the iPhone shows. Persisted so it
# survives restarts, carried in `hello` + every `robot_state` so all clients sync,
# and reflected in the monitor (serverInfo.name). Renamable from any client OR the
# command monitor. Separate file per mode so a real robot never inherits the sim's
# saved name (or vice-versa).
_NAME_FILE = os.path.expanduser("~/.mabel_real_name" if _IS_REAL else "~/.mabel_sim_name")
_robot_name = ["MABEL" if _IS_REAL else "MABEL Sim"]
_srv_info_ref = {"d": None}            # the live serverInfo dict (set in serve)


def _load_robot_name(default: str) -> str:
    try:
        with open(_NAME_FILE) as f:
            n = f.read().strip()
            return n or default
    except OSError:
        return default


def _apply_rename(name: str) -> None:
    """Rename from any client (set_robot_name) or the command monitor — persist,
    update hello + the monitor's serverInfo; clients sync via robot_state.server."""
    n = (name or "").strip()
    if not n or n == _robot_name[0]:
        return
    _robot_name[0] = n
    _hello_info["name"] = n
    try:
        with open(_NAME_FILE, "w") as f:
            f.write(n)
    except OSError as e:
        log.warning("could not persist robot name: %s", e)
    if _srv_info_ref["d"] is not None:
        _srv_info_ref["d"]["name"] = n
        MONITOR.set_server_info(_srv_info_ref["d"])
    log.info("renamed -> '%s' (synced to clients + monitor)", n)


# Operator mode + navigation joystick, set by the WebSocket `control` message
# (Phase 2). Default = manipulation, no joystick.
_control = {"joystick": None}
_DBG = {"t": 0.0}   # throttle for the temporary drive-path debug log
# Nav2 autonomous-with-takeover state. A web/app tap sends NAV_GOAL → the goal is
# queued here for the control thread (which owns `plant`) to forward on /goal_pose;
# while active, the loop feeds Nav2's /mabel_cmd (NavFn A* + MPPI) into the base
# until a human stick deflection takes over (then the goal is canceled).
_nav = {"active": False}
# Autonomous nav without an app goal: when MABEL_NAV_AUTO is set, the bridge keeps
# nav "active" so it follows Nav2's /mabel_cmd continuously (e.g. frontier
# exploration / autonomous data collection, where goals go straight to the Nav2
# action and never pass through this websocket). Operator stick still overrides.
_NAV_AUTO = os.getenv("MABEL_NAV_AUTO", "").lower() in ("1", "true", "yes")
_nav_goal_pending: list = []     # (x, y, yaw) to send, or None to cancel
_nav_preview_pending: list = []  # (x, y, yaw) staged goals to show on RViz (no drive)
# Input-device-agnostic commands (iOS app Body tab / future controllers):
# direct joint targets (iOS joint id -> radians) and the structured control spec.
_joint_cmd: dict = {}                          # latest {joint_id: value}
# The three orthogonal teleop axes (see wire_protocol). `method` selects the
# reference generator (navigation→base, retarget→IK, joint/wrist→WBC targets);
# `type`+`region` parameterise the torque controller; `stiffness` is impedance K.
_control_spec = {"method": "navigation", "type": "impedance",
                 "region": "arm", "stiffness": 0.0}
_ext_forces: dict = {}                         # {body_name: [fx,fy,fz]} operator push
# Episode recording start/stop requests from the app. The renderer touches the GL
# context, so the network thread only QUEUES (action, name); the sim thread acts.
_record_pending: list = []                      # [(True|False, name)]
# Data-collection (rosbag/MCAP) start/stop requests from the app. Like the
# episode recorder, the net thread only QUEUES (active, label); the sim thread
# (which owns the BagRecorder subprocess) acts. Separate from _record_pending:
# this records the full ROS bus to a bag, not the Mac-side demonstration frames.
_data_collect_pending: list = []                # [(True|False, label), ...]
_hold_request = threading.Event()              # snap-hold the pose (mode change)
# Per-operator body calibration: the net thread queues a (action, user) request here;
# the sim thread (which owns `teleop`) consumes it and drives the facade's calibrator.
_body_cal_pending: list = []                    # [("start"|"cancel", user), ...]
_last_cmd_recv = [0.0]                          # wall time the newest command arrived
_lat_ema = [0.0]                                # smoothed latency (ms), EMA
# Latest ACTUAL robot joint state (MuJoCo qpos → iOS joint ids), published back
# to every client so the app's 3D model mirrors the real robot, not its command
# echo. Reference-swapped wholesale → thread-safe to read from the net thread.
_robot_state = {"joints": {}, "base": {}}
# Per-module connection roster (module_status.py). Written by the sim/ROS source
# thread, read by every client's state loop; attached to robot_state only when it
# CHANGES, so the 50 Hz loop never re-ships 13 unchanged rows.
from module_status import ModuleTracker as _ModuleTracker      # noqa: E402
MODULES = _ModuleTracker()
# False until a module SOURCE actually starts. Gates the roster onto the wire: a
# tracker with no source reports every module "not connected", which would be a
# lie, not a status. No source ⇒ no `modules` key ⇒ the app shows WAITING.
MODULES_LIVE = [False]

# ── torque ⟷ module roster ────────────────────────────────────────────────────
# The rules live in module_status.py (TORQUE_MODULES / torque_report: absent modules
# report null torque, never False — see the rationale there). Only the roster cache
# is here, because it needs MODULES / MODULES_LIVE.
from module_status import (torque_module_present as _torque_module_present,   # noqa: E402
                           torque_report as _torque_report_pure, roster_ok as _roster_ok)
_modules_ok_cache = {"t": 0.0, "ok": {}}


def _modules_ok() -> dict:
    """{module id: connected?} from the roster, cached 0.5 s (snapshot() takes the
    tracker lock and the wire loops ask at 50 Hz). {} until a module source is live —
    treated as "unknown, assume present" so sim / early boot are unchanged."""
    now = time.time()
    if now - _modules_ok_cache["t"] > 0.5:
        _modules_ok_cache["t"] = now
        _modules_ok_cache["ok"] = _roster_ok(MODULES.snapshot()) if MODULES_LIVE[0] else {}
    return _modules_ok_cache["ok"]


def _torque_report() -> dict:
    """The wire's robot_state.torque: _torque_state with absent modules nulled."""
    return _torque_report_pure(_torque_state, _modules_ok())

# iOS joint id → MuJoCo joint name (qpos read). The cascaded lift exposes its
# lower stage; the upper stage mirrors it (one tendon drives both equally).
_TELEMETRY_JOINTS = {
    **{f"left_arm_{i}":  f"Left Arm {i}"  for i in range(1, 8)},
    **{f"right_arm_{i}": f"Right Arm {i}" for i in range(1, 8)},
    "torso": "Torso",
    "neck_yaw": "Neck Yaw", "neck_pitch": "Neck Pitch", "neck_roll": "Neck Roll",
    "lift_lower": "Lift Lower", "lift_upper": "Lift Upper",
    # ORCA hands: stream the wrists + all finger joints so the apps' 3D
    # models mirror the retargeted hands (previously command-echo only).
    "left_wrist": "Left Wrist", "right_wrist": "Right Wrist",
}
for _pre, _Side in (("left", "Left"), ("right", "Right")):
    for _f, _F in (("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"),
                   ("ring", "Ring"), ("pinky", "Pinky")):
        _segs = ("mcp", "abd", "pip", "dip") if _f == "thumb" else ("abd", "mcp", "pip")
        for _seg in _segs:
            _TELEMETRY_JOINTS[f"{_pre}_{_f}_{_seg}"] = f"{_Side} {_F} {_seg.upper().replace('ABD', 'Abd')}"


# MuJoCo qpos/qvel/body addresses, resolved ONCE per model and cached. The hot
# path (_read_robot_joints/_read_base_state) runs every physics substep on the
# control thread; the old per-call name lookups (data.joint("name") ×~50) were
# pure-Python work holding the GIL away from the 50 Hz network thread → telemetry
# jitter under load. Indexing data.qpos[adr] directly is ~free and GIL-light.
_JADDR: dict[int, dict] = {}        # id(model) → {ios_id: qpos address}
_BADDR: dict[int, dict] = {}        # id(model) → {steer/drive addresses, chassis body id}


def _telemetry_addr(model):
    c = _JADDR.get(id(model))
    if c is None:
        c = {}
        for ios_id, jname in _TELEMETRY_JOINTS.items():
            try:
                c[ios_id] = int(model.joint(jname).qposadr[0])
            except Exception:
                pass
        _JADDR[id(model)] = c
    return c


def _base_addr(model):
    c = _BADDR.get(id(model))
    if c is None:
        c = {"steer": [], "drive": [], "body": -1}
        for key in ("FL", "FR", "B"):
            try:
                c["steer"].append(int(model.joint(f"{key} Steer").qposadr[0]))
            except Exception:
                c["steer"].append(None)
            try:                                    # drive VELOCITY → qvel/dof address
                c["drive"].append(int(model.joint(f"{key} Drive").dofadr[0]))
            except Exception:
                c["drive"].append(None)
        try:
            c["body"] = int(model.body(C.CHASSIS_BODY).id)
        except Exception:
            c["body"] = -1
        _BADDR[id(model)] = c
    return c


def _read_robot_joints(model, data):
    """Sample the live MuJoCo qpos for every joint the app visualises (addresses
    cached once — see _telemetry_addr).

    Value convention = ABSOLUTE MuJoCo qpos. The apps' rig ships a per-joint ``ref``
    (= qpos0, stamped into rig.json by glb_to_rig.py) and renders each joint at
    ``qpos − ref``, because the GLB geometry is baked at the assembled (ref) pose.
    So the ref math lives CLIENT-side — the wire stays the raw absolute qpos it has
    always carried. (On MABEL only the 14 ORCA finger/wrist joints have a nonzero
    ref; every other joint has ref = 0, so raw == displacement there.)"""
    qpos = data.qpos
    out = {ios: float(qpos[adr]) for ios, adr in _telemetry_addr(model).items()}
    # If the upper lift stage isn't a separate joint, mirror the lower one.
    if "lift_upper" not in out and "lift_lower" in out:
        out["lift_upper"] = out["lift_lower"]
    return out


# ── OPTIONAL sparse joint dynamics (V2 telemetry extension) ─────────────────
# Per-joint velocity (qvel) and actuator torque (qfrc_actuator at the joint's
# DOF) for the apps' bar/line telemetry panels. Attached to robot_state ~10 Hz
# exactly like the lidar sweep (stamp-gated per client), so the 50 Hz stream
# pays the two ~50-float dicts ten times a second, not fifty. Additive and
# backwards-compatible: apps that don't know the fields ignore them, and the
# real:ros plant simply never produces them (apps then show "not streamed").
_JDOF: dict[int, dict] = {}             # id(model) → {ios_id: dof address}
_dyn_next = [0.0]                       # control-loop rate limiter (~10 Hz)


def _telemetry_dof(model):
    c = _JDOF.get(id(model))
    if c is None:
        c = {}
        for ios_id, jname in _TELEMETRY_JOINTS.items():
            try:
                c[ios_id] = int(model.joint(jname).dofadr[0])
            except Exception:
                pass
        _JDOF[id(model)] = c
    return c


def _read_joint_dynamics(model, data):
    qvel, qfrc = data.qvel, data.qfrc_actuator
    vel, tor = {}, {}
    for ios, adr in _telemetry_dof(model).items():
        vel[ios] = round(float(qvel[adr]), 4)
        tor[ios] = round(float(qfrc[adr]), 3)
    if "lift_upper" not in vel and "lift_lower" in vel:
        vel["lift_upper"] = vel["lift_lower"]
        tor["lift_upper"] = tor["lift_lower"]
    return {"stamp": time.time(), "vel": vel, "torque": tor}


# ── Planar LiDAR sweep for the apps' top-down tile ──────────────────────────
# The scene's 360 rangefinder SENSORS stay disabled on the Mac path (they ray-
# cast every mj_step and cost ~10 ms/step — see the sensor-disable block below);
# instead the control loop calls this ~5 Hz: one mj_multiRay fan from the
# `lidar` body, same geometry, ~0.1 ms. Contract matches ROS /scan: ray i points
# at angleMin + i·angleInc in the BASE frame (0 = +x/forward, CCW), range in
# metres, non-returns == rangeMax.
_LIDAR_NRAY = 360
_LIDAR_CUTOFF = 12.0                    # matches the scene rangefinder cutoff
_LIDAR_BID = [None]                     # resolved once ("lidar" body id, or -1)
_lidar_next = [0.0]                     # control-loop rate limiter (~5 Hz)
# World-frame lidar HIT points from the latest sweep (Nx3 f32), for the sim
# map-plane feed: the laser ring drawn in the apps' 3D Maps view. Written by
# _lidar_sweep on the control thread (reference swap), read on the net loop.
_lidar_pts = [None]


_LIDAR_R0 = 0.045                       # emitter radius — matches the ray sites


def _lidar_sweep(model, data):
    if _LIDAR_BID[0] is None:
        _LIDAR_BID[0] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar")
    bid = _LIDAR_BID[0]
    if bid < 0:
        return None
    ctr = np.asarray(data.xpos[bid], dtype=np.float64)
    R = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    a = np.arange(_LIDAR_NRAY) * (2.0 * np.pi / _LIDAR_NRAY)
    # World-frame ray directions AND per-ray origins offset to the emitter
    # radius, exactly like the scene's replicated <site> ring — a single shared
    # origin at the body CENTRE sits inside the lidar mast geom and every ray
    # self-hits at ~3 cm (measured), so each ray must start on the ring.
    dirs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1) @ R.T
    geomid = np.zeros(1, dtype=np.int32)
    ranges = np.empty(_LIDAR_NRAY)
    for i in range(_LIDAR_NRAY):
        pnt = ctr + dirs[i] * _LIDAR_R0
        d = mujoco.mj_ray(model, data, pnt, dirs[i], None, 1, bid, geomid)
        ranges[i] = d + _LIDAR_R0 if 0 <= d <= _LIDAR_CUTOFF else _LIDAR_CUTOFF
    # World-frame HIT points (non-returns dropped) for the map-plane laser ring
    # — the same sweep the apps' top-down tile gets, but drawable in their 3D
    # Maps view. Reference-swapped; ~free (one vectorized multiply).
    hit = ranges < _LIDAR_CUTOFF
    if hit.any():
        _lidar_pts[0] = (ctr + dirs[hit] * ranges[hit, None]).astype(np.float32)
    else:
        _lidar_pts[0] = None
    return {"angleMin": 0.0, "angleInc": 2.0 * np.pi / _LIDAR_NRAY,
            "rangeMax": _LIDAR_CUTOFF, "stamp": time.time(),
            "ranges": [round(float(r), 3) for r in ranges]}


def _read_base_state(model, data):
    """The canonical mobile-base state the apps mirror: the chassis world pose
    (x, y, z + full orientation quaternion, plus yaw for convenience) and each
    swerve module's STEER position and DRIVE velocity. The FULL quaternion lets
    the viewer show the base tipping / falling over, not just turning."""
    c = _base_addr(model)
    qpos, qvel = data.qpos, data.qvel
    steer = [float(qpos[a]) if a is not None else 0.0 for a in c["steer"]]
    drive = [float(qvel[a]) if a is not None else 0.0 for a in c["drive"]]
    bid = c["body"]
    if bid >= 0:
        R = data.xmat[bid].reshape(3, 3)
        x, y, z = float(data.xpos[bid][0]), float(data.xpos[bid][1]), float(data.xpos[bid][2])
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        quat = [float(q) for q in data.xquat[bid]]   # MuJoCo order [w, x, y, z]
    else:
        x = y = z = yaw = 0.0
        quat = [1.0, 0.0, 0.0, 0.0]
    return {"x": x, "y": y, "z": z, "yaw": yaw, "quat": quat,
            "steer": steer, "drive": drive}


def _read_body_xforms(data, nbody):
    """Flat [x,y,z, qw,qx,qy,qz] per body (MuJoCo body-id order) — the world
    pose the iOS mirror renders each body at, exactly like MuJoCo's viewer."""
    xp = data.xpos.reshape(-1)
    xq = data.xquat.reshape(-1)
    out = []
    for b in range(nbody):
        out.extend((float(xp[b * 3]), float(xp[b * 3 + 1]), float(xp[b * 3 + 2]),
                    float(xq[b * 4]), float(xq[b * 4 + 1]),
                    float(xq[b * 4 + 2]), float(xq[b * 4 + 3])))
    return out


# The direct-joint-override path (iOS Body tab / web sliders) now lives in the
# control facade as MabelTeleop._apply_joint_targets — applied AFTER the control
# law each substep, so the bridge no longer owns any actuator-mapping logic.


# ---------------------------------------------------------------------------
# Network thread: WebSocket teleop in + MJPEG cameras out
# ---------------------------------------------------------------------------

def _peer_str(ws) -> str:
    """`ip:port` for a websocket peer (any client device), or '?'."""
    addr = getattr(ws, "remote_address", None)
    if isinstance(addr, (tuple, list)) and len(addr) >= 2:
        return f"{addr[0]}:{addr[1]}"
    return str(addr) if addr else "?"


# Explicit ?client= tag → (device, app, platform). Clients self-identify so the
# monitor can name them exactly (iPhone vs Vision Pro vs Website).
_CLIENT_TAGS = {
    "visionpro": ("Vision Pro", "MabelTeleop", "visionOS"),
    "avp":       ("Vision Pro", "MabelTeleop", "visionOS"),
    "ios":       ("iPhone",     "MABEL Mobile App", "iOS"),
    "iphone":    ("iPhone",     "MABEL Mobile App", "iOS"),
    "ipad":      ("iPad",       "MABEL Mobile App", "iPadOS"),
    "web":       ("Website",    "teleop-web", "Browser"),
    "browser":   ("Website",    "teleop-web", "Browser"),
    "keyboard":  ("Keyboard",   "ROS node",   "ROS"),
    "gamepad":   ("Gamepad",    "—",          "—"),
    "linux":     ("Linux",      "client",     "Linux"),
    "desktop":   ("Linux",      "client",     "Linux"),
    # server/ble_bridge.py — the Bluetooth LE peripheral relaying a phone/headset's
    # GATT writes as ordinary wire messages (one more client, no second control path)
    "ble":       ("Bluetooth",  "ble_bridge", "BLE"),
}


def _classify_network(sock_ip: str) -> str:
    """Name the transport a client reached us over, from its socket IP: a
    Tailscale peer hits the host's 100.x directly; LAN peers are RFC-1918
    private."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(sock_ip)
    except ValueError:
        return "Unknown"
    if ip.is_loopback:
        return "Localhost"
    # Tailscale hands out CGNAT 100.64.0.0/10; treat that as the tailnet.
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return "Tailscale"
    if ip.is_private or ip.is_link_local:
        return "LAN"
    return "Internet"


def _network_info(ws, path: str) -> dict:
    """Per-connection networking detail for the monitor: the real client IP, the
    transport it came over (LAN / Tailscale / Localhost) and the addressed
    host / web origin."""
    peer = _peer_str(ws)
    sock_ip, _, port = peer.rpartition(":")
    sock_ip = sock_ip or peer

    headers = getattr(ws, "request_headers", None)

    def h(name: str) -> str:
        try:
            return (headers.get(name, "") or "") if headers is not None else ""
        except Exception:
            return ""

    transport = _classify_network(sock_ip)
    secure = transport == "Tailscale"
    return {
        "ip": sock_ip,
        "port": port,
        "transport": transport,
        "secure": secure,
        "host": h("Host"),
        "origin": h("Origin"),
        "path": path,
    }


def _parse_web_ua(ua: str) -> tuple[str, str]:
    """(browser, os) from a browser User-Agent, by token sniffing. Order matters:
    Edge/Opera/in-app browsers masquerade as Chrome, Chrome embeds 'Safari', so
    the more specific token must win."""
    low = ua.lower()
    if "windows nt" in low:       os_name = "Windows"
    elif "iphone" in low:         os_name = "iPhone"
    elif "ipad" in low:           os_name = "iPad"
    elif "android" in low:        os_name = "Android"
    elif "cros" in low:           os_name = "ChromeOS"
    elif "mac os x" in low or "macintosh" in low: os_name = "macOS"
    elif "linux" in low:          os_name = "Linux"
    else:                         os_name = "—"
    if "edg" in low:              br = "Edge"            # Edg/ Edga/ EdgiOS/
    elif "opr/" in low or "opera" in low: br = "Opera"
    elif "crios/" in low:         br = "Chrome"          # Chrome on iOS
    elif "fxios/" in low or "firefox/" in low: br = "Firefox"
    elif "samsungbrowser" in low: br = "Samsung"
    elif "chrome/" in low or "chromium/" in low: br = "Chrome"
    elif "safari/" in low:        br = "Safari"
    else:                         br = "Browser"
    return br, os_name


def _identify_device(query: str, headers) -> dict:
    """Best-effort client identity for the monitor, in priority order:
    1. an explicit `?client=` tag the app appends to its ws URL (exact);
    2. the WebSocket upgrade User-Agent / Origin (native URLSession apps send a
       bundle UA; browsers send Origin + a Mozilla UA) — works even before an
       app is rebuilt to send the tag;
    3. unknown → the hub falls back to a hands/rate heuristic.
    For a Website it additionally resolves the BROWSER + OS (Chrome · macOS, …)
    and a per-tab session id, so multiple browser openings are distinguishable.
    Returns {device, app, platform, os, browser, session, src}."""
    from urllib.parse import parse_qs
    q = parse_qs(query or "")
    tag = (q.get("client", [""])[0] or "").strip().lower()
    session = (q.get("sid", [""])[0] or "").strip()[:8]

    ua = origin = ""
    if headers is not None:
        try:
            ua = headers.get("User-Agent", "") or ""
            origin = headers.get("Origin", "") or ""
        except Exception:
            pass
    low = ua.lower()

    # Decide the device: tag wins, else sniff the UA.
    if tag in _CLIENT_TAGS:
        d = _CLIENT_TAGS[tag]
        info = {"device": d[0], "app": d[1], "platform": d[2], "src": "tag"}
    elif "mabelteleop" in low:
        info = {"device": "Vision Pro", "app": "MabelTeleop", "platform": "visionOS", "src": "ua"}
    elif "mabel" in low and "mobile" in low:
        info = {"device": "iPhone", "app": "MABEL Mobile App", "platform": "iOS", "src": "ua"}
    elif origin or "mozilla" in low:
        info = {"device": "Website", "app": "browser", "platform": "Browser", "src": "ua"}
    elif "linux" in low:                              # a non-browser Linux client
        info = {"device": "Linux", "app": "client", "platform": "Linux", "src": "ua"}
    else:
        info = {"device": None, "app": None, "platform": None, "src": "none"}

    info["session"] = session
    # For a browser, name the exact browser + OS so two tabs (or a Mac Chrome vs
    # an Android Safari) read distinctly.
    if info["device"] == "Website" and (ua or origin):
        br, os_name = _parse_web_ua(ua)
        info["browser"], info["os"] = br, os_name
        info["app"] = br
        info["platform"] = f"{br} · {os_name}" if os_name != "—" else br
    return info


# ── Server-side identity (shown in the monitor's Server card) ────────────────
def _local_ipv4s() -> list[str]:
    """Best-effort list of this host's non-loopback IPv4 addresses (LAN +
    Tailscale). Uses hostname resolution, the LAN-probe trick, and — if present
    — the `tailscale` CLI for the reliable 100.x."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except Exception:
        pass
    try:
        addrs.add(console.lan_ip())
    except Exception:
        pass
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                             text=True, timeout=2)
        for ln in out.stdout.split():
            if ln.strip():
                addrs.add(ln.strip())
    except Exception:
        pass
    return sorted(a for a in addrs if a and not a.startswith("127."))


def _joint_limits(model) -> dict:
    """Each joint's [lo, hi] range, so the monitor can draw bipolar bars scaled to
    the real travel. Limited joints use jnt_range; the rest fall back to the
    position-servo actuator's ctrlrange when it has one."""
    lim: dict[str, list] = {}
    try:
        for j in range(model.njnt):
            name = model.joint(j).name
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                lim[name] = [float(lo), float(hi)]
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jname = model.joint(model.actuator_trnid[a, 0]).name
            if jname not in lim:
                lo, hi = model.actuator_ctrlrange[a]
                if hi > lo:
                    lim[jname] = [float(lo), float(hi)]
    except Exception:
        pass
    return lim


# Arm end-stop safety margin (rad) the HAL keeps off each hard stop — mirror of
# a SIM-side slider margin (the HAL's own end-stop margin was removed 2026-07-16;
# the real robot's authority is /arms/joint_limits). Applied to the streamed limits so a
# teleop slider can never command an arm into the stall zone (the "gap" between the model
# range and the true safe limits). Only arm joints get it; a prismatic lift / continuous
# steer would be nonsense to pull in by 5°.
_ARM_LIMIT_MARGIN = float(os.getenv("MABEL_HW_ARMS_LIMIT_MARGIN_RAD", "0.087"))


def _urdfize(mj_name: str) -> str:
    """MuJoCo joint name -> URDF/wire id, the SAME convention jointPositions uses
    ('Left Arm 4' -> 'left_arm_4'), so jointLimits keys line up with the live pose."""
    return mj_name.strip().lower().replace(" ", "_")


def _commandable_joint_limits(model) -> dict:
    """Per-joint SAFE COMMANDABLE range for the teleop sliders, keyed by URDF id and in
    RADIANS (processed/model frame) — what gets streamed on robot_state.jointLimits.

    The model range (jnt_range / ctrlrange) pulled in on ARM joints by the end-stop safety
    margin, so a slider clamps to the true safe limits, not the raw model extremes. The app
    renders these as degrees; the encoder frame never leaves the server. On the REAL robot
    the calibration-derived authority is the HAL's commandable_ranges() (mabel_hw ArmHandle,
    also published on /arms/joint_limits; streamed via the
    plant when available) — this model-derived map is the SIM path and the fallback."""
    out = {}
    for mj, (lo, hi) in _joint_limits(model).items():
        name = _urdfize(mj)
        if "arm" in name and (hi - lo) > 2.0 * _ARM_LIMIT_MARGIN:
            lo, hi = lo + _ARM_LIMIT_MARGIN, hi - _ARM_LIMIT_MARGIN
        out[name] = [float(lo), float(hi)]
    return out


# The per-joint safe range streamed on robot_state.jointLimits (set once at bring-up).
_JOINT_LIMITS_STREAM: dict = {}


def _resolve_joint_limits(plant, model) -> dict:
    """The joint-limit map to stream: the REAL robot's calibration-derived ranges when the
    plant exposes them (mabel_hw ArmHandle.commandable_ranges), else the model-derived
    sim/fallback map. Keeping the authority in the plant means a recalibration flows out with
    no server edit; the sim path just reflects the model it is already simulating."""
    fn = getattr(plant, "joint_limits", None)
    if callable(fn):
        try:
            lims = fn()
            if lims:
                return {k: [float(v[0]), float(v[1])] for k, v in lims.items()}
        except Exception:
            pass
    return _commandable_joint_limits(model)


def _collect_server_info(args) -> dict:
    """The bridge's own connection info — host, OS, the IPs it's reachable on,
    and every endpoint it serves — so the monitor can show where to point a
    client and which transports are live."""
    import ipaddress
    addrs = _local_ipv4s()
    lan = tail = None
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
            tail = a
        elif ip.is_private and lan is None:
            lan = a
    if lan is None:
        try:
            lan = console.lan_ip()
        except Exception:
            lan = ""
    sysname = platform.system().replace("Darwin", "macOS")
    return {
        "host": socket.gethostname(),
        "os": sysname,
        "platform": f"{sysname} {platform.release()}".strip(),
        "python": platform.python_version(),
        "lanIp": lan or "",
        "tailscaleIp": tail,
        "addresses": addrs,
        "wsPort": args.ws_port,
        "wsPath": getattr(args, "ws_path", "/teleop"),
        "monitorPath": MONITOR_PATH,
        "mapPort": MAP_PORT,             # dedicated live-SLAM-map WebSocket port
        "lidarPort": LIDAR_PORT,         # dedicated raw-LiDAR-scan WebSocket port
        "httpPort": args.http_port,
        "udpPort": int(getattr(args, "udp_video_port", 0) or 0),
        "h264Port": (int(os.getenv("MABEL_SIM_H264_PORT", "9092"))
                     if os.getenv("MABEL_SIM_H264", "0").lower() not in ("0", "false", "no", "off")
                     else 0),
        "version": _hello_info.get("version", ""),
        "server": _hello_info.get("server", ""),   # identity (sim vs real badge)
        "kind": _hello_info.get("kind", ""),       # "real" | "sim"
        "name": _robot_name[0],          # the robot's display name (== the iPhone's)
        "tokenGated": bool(os.getenv("MABEL_TOKEN", "").strip()),
        "startedAt": time.time(),
    }


def start_network_thread(ws_host: str, ws_port: int, ws_path: str,
                         http_host: str, http_port: int,
                         path_caps: dict[str, PushCapture],
                         enable_ws: bool,
                         ready: threading.Event,
                         loop_holder: dict) -> None:
    import websockets
    from aiohttp import web

    async def _ws_handler(ws, path=None):
        # websockets ≤11 passes path positionally; ≥12 exposes ws.request.path.
        req_path = path or getattr(ws, "path", None) or getattr(
            getattr(ws, "request", None), "path", "/")
        # Optional access code: when MABEL_TOKEN is set, clients must append
        # ?key=<code> (…/teleop?key=CODE). Split it off before the path
        # comparison. Unset (the default) = open, as every LAN client expects.
        from urllib.parse import urlsplit, parse_qs
        parts = urlsplit(req_path)
        token = os.getenv("MABEL_TOKEN", "").strip()
        supplied = (parse_qs(parts.query).get("key", [""])[0]).strip() if token else ""
        # A command-monitor subscribes on /monitor (read-only fan-out of every
        # client's control stream) — handled entirely by the MonitorHub. It is
        # gated by the SAME token as teleop.
        if parts.path == MONITOR_PATH:
            if token and supplied != token:
                await ws.close(1008, "unauthorized")
                return
            log.info("monitor connected    %s", _peer_str(ws))
            await MONITOR.add_subscriber(ws)
            log.info("monitor disconnected %s", _peer_str(ws))
            return
        # The live SLAM map (nvblox mesh + odom) streams read-only on /map, gated
        # by the SAME token as teleop.
        if parts.path == MAP_PATH:
            if token and supplied != token:
                await ws.close(1008, "unauthorized")
                return
            log.info("map client connected    %s", _peer_str(ws))
            await MAP_HUB.add_subscriber(ws)
            log.info("map client disconnected %s", _peer_str(ws))
            return
        if parts.path != ws_path:
            await ws.close(1008, "wrong path")
            return
        if token and supplied != token:
            log.warning("rejected client %s — bad/missing token", _peer_str(ws))
            await ws.close(1008, "unauthorized")
            return
        _clients["n"] += 1
        # Disable Nagle on this connection: our 50 Hz state frames are small
        # (~3 KB) and latency-critical, but the legacy websockets server does NOT
        # set TCP_NODELAY, so Nagle + delayed-ACK batches them into periodic
        # ~40-200 ms stalls on real TCP/Wi-Fi (invisible on loopback). Send each
        # frame immediately.
        try:
            import socket as _socket
            _sock = ws.transport.get_extra_info("socket")
            if _sock is not None:
                _sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        except Exception:
            pass
        client_id = id(ws)                       # per-connection key for its frame
        peer = _peer_str(ws)
        device = _identify_device(parts.query, getattr(ws, "request_headers", None))
        net = _network_info(ws, parts.path)
        MONITOR.on_connect(client_id, peer, device, net)
        log.info("client connected     %s  %s via %s  (%d online)",
                 net["ip"], device.get("device") or "?", net["transport"], _clients["n"])
        # Re-seed this device's last control mode (see _device_specs): a reconnect —
        # including across a gateway restart — resumes e.g. the VP's Joint mode
        # instead of silently resetting to navigation.
        _dev_spec = _device_specs.get(str(device.get("device") or ""))
        if _dev_spec:
            _client_specs[client_id] = dict(_dev_spec)
            log.info("restored %s control mode: %s·%s·%s", device.get("device"),
                     _dev_spec.get("method"), _dev_spec.get("type"), _dev_spec.get("region"))
        try:
            await ws.send(WireCodec.encode(dict(_hello_info), WireMessageType.HELLO).decode())
        except Exception:
            pass

        async def _send_state():
            """Stream the ACTUAL robot joint state to this client at 100 Hz so the
            app's 3D model mirrors the simulation, not its own command echo.
            (10 ms; 20 Hz -> 50 Hz -> 100 Hz — the faster telemetry visibly lowers
            the displayed-robot lag for every client at negligible bandwidth.)

            NOTE the rate is 100 Hz, not the 50 Hz this docstring and the protocol
            docs claimed until 2026-07-19; measured 98.7 Hz over 594 frames by
            server/tools/exp_sim_stack.py (S1). Clients are latest-wins and pace
            their own render loop, so the bump is transparent to them — but the
            written contract had drifted from the code, so it is corrected here
            and in docs/protocol.md."""
            loop = asyncio.get_event_loop()
            period = 0.01                          # 100 Hz (VP retarget feel; states are ~KBs)
            next_t = loop.time()
            _last_scan_sent = 0.0                  # per-client: lidar dedup marker
            _last_health_sent = 0.0                # per-client: motorHealth dedup
            lidar_sent = None                      # stamp of the last SIM sweep sent
            dyn_sent = None                        # stamp of the last dynamics sample sent
            op_sent = None                         # stamp of the last operator overlay sent
            modules_sent = None                    # last module roster sent (change-gated)
            modules_sent_t = 0.0                   # + a 0.5 Hz heartbeat (latest-wins clients)
            limits_sent_t = 0.0                    # jointLimits heartbeat (rides the roster)
            try:
                while True:
                    # Phase-accurate pacing: schedule each tick at a fixed cadence
                    # and sleep only the REMAINDER after encode+send, so the stream
                    # holds ~50 Hz instead of drifting to ~45 Hz (the old
                    # sleep(0.02)-THEN-work pattern paid 20 ms + work every cycle).
                    next_t += period
                    delay = next_t - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        next_t = loop.time()       # fell behind → re-anchor, don't burst
                        await asyncio.sleep(0)
                    js = _robot_state["joints"]
                    if not js:
                        continue
                    # Bridge latency: how stale the newest command is when we
                    # publish state back, SMOOTHED with an EMA (~0.5 s window) so
                    # the displayed number is steady instead of jittering 0–50 ms.
                    # NOTE: this is command STALENESS (time since the last inbound
                    # message), not transport latency — the apps now display their
                    # own PING→PONG RTT instead. Kept for the monitor + old clients.
                    lat = 0.0
                    if _last_cmd_recv[0] > 0:
                        raw = max(0.0, min(999.0, (time.time() - _last_cmd_recv[0]) * 1000.0))
                        _lat_ema[0] = raw if _lat_ema[0] == 0.0 else 0.88 * _lat_ema[0] + 0.12 * raw
                        lat = _lat_ema[0]
                    # Honest freshness: age of the pose we're shipping, on the server
                    # clock only (no client-skew). ~1 tick on a healthy link; it grows
                    # ONLY when the control loop falls behind real-time — the true
                    # "the robot itself is lagging" signal, distinct from the network.
                    _st = _robot_state.get("t", 0.0)
                    age = max(0.0, (time.time() - _st) * 1000.0) if _st else 0.0
                    # Unified robot_state: joint positions + canonical base state
                    # (steer pos / drive vel / chassis pose) + global status.
                    # `driver` tells THIS client whether it is the active driver
                    # (arbitration: most recent meaningful input wins), so every
                    # UI can show DRIVING vs OBSERVING instead of fighting.
                    act = _active_client()
                    # Per-client server block: the shared robot identity/caps PLUS
                    # the transport THIS client actually reached us over (LAN /
                    # Tailscale / Localhost), classified server-side from its
                    # socket IP. Authoritative — the app no
                    # longer has to guess the route from the host IP it dialled.
                    _srv = dict(_hello_info)   # carries the robot name + caps
                    _srv["transport"] = net["transport"]
                    _srv["secure"] = net["secure"]
                    payload = {"jointPositions": js,
                               "base": _robot_state["base"],
                               "latencyMs": lat,
                               "stateAgeMs": age,
                               # Server wall-clock at send (additive, 2026-07).
                               # Same-host clients (simulator + loopback tests)
                               # subtract it from their receive time to measure
                               # TRUE one-way staleness — the direct probe for
                               # client-side receive-buffer buildup. Meaningless
                               # across hosts (clock skew): debug/QA only.
                               "sentAt": time.time(),
                               "battery": _battery_state(),
                               "recording": _robot_state.get("recording", {}),
                               "server": _srv,
                               "driver": {"you": act == client_id,
                                          "clients": _clients["n"]},
                               # Per-operator body calibration (null/omitted until used).
                               "calibrationStatus": _robot_state.get("calibrationStatus"),
                               "operatorProfile": _robot_state.get("operatorProfile"),
                               # ── Additive status echoes (2026-07 data-plane update).
                               # Server-authoritative so EVERY client can reflect them,
                               # not just the one that sent the command. Old app builds
                               # ignore unknown keys — nothing above changed shape.
                               "estop": _estop["active"],
                               "controlMode": dict(_control_spec),
                               "torque": _torque_report(),
                               "dataCollection": _robot_state.get("dataCollection", {}),
                               "policy": dict(_policy_state),
                               "animation": dict(_animation_state)}
                    # LiDAR alongside the joints: attach the latest sweep (compact
                    # 1°/cm JSON, ~1.8 KB, produced at MABEL_MAP_SCAN_HZ) only when
                    # it CHANGED since this client's last robot_state — the 50 Hz
                    # loop never repeats a sweep, so the channel stays light and a
                    # drive UI renders the radar with no extra socket.
                    _scan = latest_scan()
                    if _scan is not None and _scan["t"] != _last_scan_sent:
                        _last_scan_sent = _scan["t"]
                        payload["lidar"] = _scan
                    # SIM fallback: the Mac path fills _robot_state["lidar"] from
                    # mj_multiRay (~5 Hz) — attach it only when the map-stream scan
                    # pipeline above produced nothing (real robots use that one).
                    li = _robot_state.get("lidar")
                    if ("lidar" not in payload and li is not None
                            and li.get("stamp") != lidar_sent):
                        lidar_sent = li.get("stamp")
                        payload["lidar"] = li
                    # Per-module connection status (base/torso/lift/neck/arms/
                    # hands + the 4 cams and the lidar). Sent on CHANGE, plus a
                    # slow heartbeat — 13 rows every 20 ms would be pure waste,
                    # but change-only is NOT enough: the clients coalesce
                    # latest-wins (WebSocketLink keeps only the newest state and
                    # decodes it at 60 Hz), so the single frame carrying a
                    # rarely-changing roster is routinely dropped before it is
                    # ever read — the panel then never appears at all. The
                    # heartbeat costs ~0.5 Hz and guarantees delivery.
                    _now = time.time()
                    _mods = MODULES.snapshot() if MODULES_LIVE[0] else None
                    if _mods is not None and (_mods != modules_sent
                                              or (_now - modules_sent_t) > 2.0):
                        modules_sent = _mods
                        modules_sent_t = _now
                        payload["modules"] = _mods
                    # Calibrated slider limits ride the SAME low-freq cadence as the module
                    # roster (per the operator: stream them with the hardware connection).
                    # Idempotent full map on a ~0.5 Hz heartbeat so the latest-wins clients
                    # can't drop the one frame that carries it; the app persists it and
                    # clamps every joint slider to the true safe range (rendered as degrees).
                    if _JOINT_LIMITS_STREAM and (_now - limits_sent_t) > 2.0:
                        limits_sent_t = _now
                        payload["jointLimits"] = _JOINT_LIMITS_STREAM
                    # Motor health (torque/load + temperature per joint, merged
                    # from the bridge's */motor_health topics at ~2 Hz) rides
                    # robot_state the same way — attached only when NEW.
                    _mh = _motor_health_now()
                    if _mh is not None and _mh["t"] != _last_health_sent:
                        _last_health_sent = _mh["t"]
                        payload["motorHealth"] = _mh["joints"]
                    # Sparse joint dynamics (~10 Hz): velocities + actuator
                    # torques (+ temps when the HAL measures them — real only;
                    # sim has no thermals) for the apps' telemetry panels.
                    # Same stamp-gating as lidar.
                    # ECHO THE OPERATOR SWITCHES so a client that reconnects —
                    # or a second client watching the same robot — sees the LIVE
                    # state rather than its own last guess. A toggle whose UI can
                    # disagree with the robot is worse than no toggle. Cheap
                    # enough to ride every tick, and idempotent.
                    _aid = _active_client()      # (NOT `active_id` — that is a
                                                 # local of the control loop, a
                                                 # different scope entirely)
                    _sp = _client_specs.get(_aid) or _control_spec
                    payload["gravityComp"] = bool(_sp.get("gravity_comp", True))
                    payload["wholeBodyRetarget"] = bool(
                        _sp.get("whole_body_retarget", False))
                    dyn = _robot_state.get("dynamics")
                    if dyn is not None and dyn.get("stamp") != dyn_sent:
                        dyn_sent = dyn.get("stamp")
                        payload["jointVelocities"] = dyn["vel"]
                        payload["jointTorques"] = dyn["torque"]
                        if dyn.get("temp"):
                            payload["jointTemps"] = dyn["temp"]
                    await ws.send(WireCodec.encode(payload, WireMessageType.ROBOT_STATE).decode())
                    # Operator overlay (retarget debug, opt-in, frame-rate-deduped):
                    # the engine's own mapped wrist targets / elbow hints / hand
                    # skeleton in robot world — see WireMessageType.OPERATOR_POINTS.
                    if _client_overlay.get(client_id):
                        _rec = _op_overlay_rec["rec"]
                        if _rec is not None and _rec.get("t") != op_sent:
                            op_sent = _rec.get("t")
                            await ws.send(WireCodec.encode(
                                _rec, WireMessageType.OPERATOR_POINTS).decode())
            except Exception:
                pass

        sender = asyncio.create_task(_send_state())
        try:
            async for raw in ws:
                try:
                    mtype, payload = WireCodec.decode_envelope(raw)
                    _last_cmd_recv[0] = time.time()    # stamp every inbound command
                    now = time.time()
                    MONITOR.record(client_id, mtype.value, payload)
                    if mtype == WireMessageType.TELEOP_FRAME:
                        # Store the RAW payload; the control facade owns the wire
                        # dialect (OperatorPose.from_teleop_frame) — see mabel/wire.py.
                        _client_frames[client_id] = (now, payload)
                        # A real stick deflection counts as "driving" → become active.
                        nj = payload.get("navJoystick")
                        if nj is not None and max(abs(nj.get("lx", 0.0)), abs(nj.get("ly", 0.0)),
                                                  abs(nj.get("rx", 0.0)), abs(nj.get("ry", 0.0))) > 0.04:
                            _client_cmd[client_id] = now      # strong claim
                        # So do tracked hands: the headset streaming skeletons is
                        # the driver even while its sticks are idle (a second
                        # hands-free app must not starve the retargeting input).
                        if payload.get("leftHand") is not None or payload.get("rightHand") is not None:
                            _client_stream[client_id] = now   # weak (passive) claim
                        # ...and a LIVE (non-identity) head: an operator actively head-
                        # tracking is driving the neck even with idle sticks. An IDLE
                        # head (identity / trackingState "notAvailable" — what the web and
                        # a head-off phone send) does NOT claim, so they stay observers.
                        h = payload.get("head")
                        if (isinstance(h, dict) and h.get("trackingState", "normal") == "normal"
                                and _head_active(h.get("transform"))):
                            _client_stream[client_id] = now   # weak (passive) claim
                    elif mtype == WireMessageType.RESET:
                        # optional {"pose": {x, y, yaw}} → relocate-reset (the
                        # sim control center's "reset somewhere open" button);
                        # {"random": true} → the SERVER picks a validated open
                        # pose (SCC safe_spawn: 0.8 m clearance grid + MuJoCo
                        # contact check) off-thread, then applies the same
                        # relocate-reset. Falls back to a plain re-home if no
                        # open spot / the solver is unavailable.
                        p = payload or {}
                        # RESET RE-HOMES BY DEFAULT (2026-07-21 operator request):
                        # both apps' status-bar RESET taps send {"random": true},
                        # but the random safe-spawn read as disorienting — honor
                        # the same envelope with a plain re-home so NEITHER app
                        # changes. MABEL_RESET_RANDOM=1 restores the safe-spawn.
                        _allow_random = os.getenv(
                            "MABEL_RESET_RANDOM", "0").strip().lower() in ("1", "true", "yes")
                        if p.get("random") and _allow_random and not _IS_REAL:
                            def _random_reset():
                                base = _robot_state.get("base") or {}
                                avoid = (float(base.get("x", 0.0)),
                                         float(base.get("y", 0.0)))
                                # INSTANT path: a pre-validated pooled pose.
                                pose = _pop_spawn_pose(avoid)
                                if pose is None:
                                    try:                     # cold picker (isolated)
                                        pose = _safe_spawn_fn(_scene_path[0], avoid_xy=avoid)
                                    except Exception as e:   # noqa: BLE001
                                        log.warning("random reset: safe_spawn failed (%s) — plain re-home", e)
                                _reset_pose[0] = pose
                                _reset.set()
                                _refill_spawn_pool(_scene_path[0])
                            threading.Thread(target=_random_reset,
                                             name="random-reset", daemon=True).start()
                        else:
                            _reset_pose[0] = p.get("pose")
                            _reset.set()
                    elif mtype == WireMessageType.RECALIBRATE:
                        # Headset toggled body tracking / recenter: re-anchor the
                        # clutch at the operator's current pose, no jump.
                        _recalibrate.set()
                    elif mtype == WireMessageType.PING:
                        # Echo the client's ping timestamp straight back so it can
                        # compute a true round-trip latency (now − echoed t) with no
                        # dependence on server-vs-client clock skew. `srv` is the
                        # server clock for optional one-way estimates.
                        await ws.send(WireCodec.encode(
                            {"t": payload.get("t"), "srv": time.time()},
                            WireMessageType.PONG).decode())
                    elif mtype == WireMessageType.JOINT_COMMAND:
                        # Any input device can push direct joint targets (= active).
                        _joint_cmd.update(payload.get("joints", {}) or {})
                        _client_cmd[client_id] = now          # strong claim
                        # Env-gated arm-command trace. Off by default; set
                        # MABEL_LOG_ARMCMD=1 to verify the iOS green-ball plan/execute
                        # gate — planning must produce NO lines, Execute a short eased
                        # burst (server/tools/exp_arm_plan.py parses these).
                        if os.getenv("MABEL_LOG_ARMCMD"):
                            _ac = {k: round(v, 4) for k, v in (payload.get("joints") or {}).items()
                                   if "arm" in k}
                            if _ac:
                                log.info("ARMCMD %s", _ac)
                    elif mtype == WireMessageType.OPERATOR_POINTS:
                        # opt this client in/out of the retarget-debug overlay
                        _client_overlay[client_id] = bool(payload.get("on", True))
                    elif mtype == WireMessageType.CONTROL_MODE:
                        # Per-client structured spec {method, controlType, region,
                        # stiffness}; a legacy {mode: ...} payload maps onto the same
                        # axes. Setting a mode marks this client as the active driver.
                        p = payload
                        spec = dict(_client_specs.get(client_id, _DEFAULT_SPEC))
                        if any(k in p for k in ("method", "controlType", "region")):
                            if "method" in p:      spec["method"] = p["method"]
                            if "controlType" in p: spec["type"]   = p["controlType"]
                            if "region" in p:      spec["region"] = p["region"]
                        # RETARGET SCOPE (additive, orthogonal to region):
                        # "local" (default) keeps the mobile BASE out of the
                        # follow loop — the operator drives it on the joystick;
                        # "global" lets it follow past the comfort box. Absent or
                        # unrecognised stays LOCAL, so a stale or silent client
                        # can never obtain base authority by omission.
                        if "scope" in p:
                            spec["scope"] = p["scope"]
                        elif "mode" in p and p["mode"] in LEGACY_MODES:
                            meth, typ, reg = LEGACY_MODES[p["mode"]]
                            spec.update(method=meth.value, type=typ.value, region=reg.value)
                        # PER-CHANNEL TRACKING (additive). Merged, not replaced,
                        # so a client may send one key; absent = unchanged, which
                        # keeps every existing client on all-on.
                        if isinstance(p.get("tracking"), dict):
                            _tr = dict(spec.get("tracking") or {})
                            for _k in ("neck", "arms", "hands", "torso", "lift"):
                                if _k in p["tracking"]:
                                    _tr[_k] = bool(p["tracking"][_k])
                            spec["tracking"] = _tr
                        # OPERATOR SWITCHES (additive, merged): gravity
                        # compensation and which retargeting profile the arms
                        # run. Absent = unchanged, so an older client that knows
                        # neither key keeps whatever the device last chose.
                        for _wire, _key in (("gravityComp", "gravity_comp"),
                                            ("wholeBodyRetarget", "whole_body_retarget")):
                            if _wire in p:
                                spec[_key] = bool(p[_wire])
                        if "stiffness" in p:
                            try:
                                spec["stiffness"] = float(p["stiffness"])
                            except (TypeError, ValueError):
                                pass
                        _client_specs[client_id] = spec
                        _remember_device_spec(device, spec)   # survives reconnect/restart
                        if os.getenv("MABEL_LOG_ARMCMD"):     # region round-trip trace
                            log.info("CTLSPEC method=%s type=%s region=%s",
                                     spec.get("method"), spec.get("type"), spec.get("region"))
                        # Picking a mode makes this client the active driver so the
                        # spec engages IMMEDIATELY (e.g. toggling to SOFT must put the
                        # robot in gravity-comp/compliance at once, so a drag in the
                        # viewer/sim stays put — otherwise it idles in position-hold
                        # and snaps back). The real-robot single-driver LOCK
                        # (request_control) is separate and still gates motion
                        # authority; this only selects whose spec the controller uses.
                        _client_cmd[client_id] = now
                    elif mtype == WireMessageType.EXTERNAL_FORCE:
                        # Operator push anywhere on the arm (drag in the app's 3D
                        # model) → external wrench the WBC engine admits. A real
                        # push is meaningful input → this client becomes the driver.
                        _ext_forces.clear()
                        _ext_forces.update(payload.get("forces", {}) or {})
                        if payload.get("forces"):
                            _client_cmd[client_id] = now      # strong claim
                    elif mtype == WireMessageType.RECORD:
                        # Select a task folder, or start/stop a demonstration episode
                        # in it (queued for the sim thread, which owns the renderer).
                        p = payload
                        # Client tag routes the dataset root: Vision Pro episodes
                        # land in dataset/simulation/vision_os, iPhone in .../ios
                        # (CurationRecorder.set_source_client).
                        _ctag = (device.get("platform") or "").lower()
                        if "select" in p:
                            _record_pending.append(("select", p.get("select"), _ctag))
                        elif p.get("active"):
                            _record_pending.append(("start", p.get("task"), _ctag))
                        else:
                            _record_pending.append(("stop", None, _ctag))
                    elif mtype == WireMessageType.ESTOP:
                        # Operator emergency stop / release. Tracked as global
                        # state so it shows in the monitor and the control loop
                        # can gate motion; engaging is a strong driver claim.
                        _was = _estop["active"]
                        _estop["active"] = bool(payload.get("active", True))
                        _robot_state["estop"] = _estop["active"]
                        if _estop["active"]:
                            _client_cmd[client_id] = now      # strong claim
                        # Hand torque follows the e-stop: DISENGAGED (active=False) ->
                        # energise the hands so the operator can control them; ENGAGED ->
                        # release to free-drive. Apply in the control loop (it owns the
                        # plant); the bridge holds-in-place before energising (no lurch).
                        #
                        # ⚠ EDGE-TRIGGERED, and it MUST stay that way. Clients re-send
                        # estop{active:false} as a periodic STATE SYNC, not just on the
                        # release click. This used to raise `pending` on EVERY message,
                        # so the control loop re-ran _resume_hold() ~1×/s — and
                        # _resume_hold adopts the MEASURED pose as the command. Against
                        # an ALREADY-HOLDING arm the measured pose is not a hand-moved
                        # pose: it is the servo's steady-state error sitting slightly
                        # ABOVE the target (kp is hardware-capped, so any t_ff excess
                        # parks the joint high). Re-adopting it made that error the new
                        # target, which the arm again overshot → a RATCHET that walked
                        # the shoulder UP a step per message, forever, with nobody
                        # commanding anything (observed on j2, 2026-07-14).
                        # Fire only on a real change — or on the first e-stop message
                        # ever (intent is None), so bring-up still energises.
                        if _was != _estop["active"] or _hand_torque_req["intent"] is None:
                            _hand_torque_req["target"] = not _estop["active"]
                            _hand_torque_req["pending"] = True
                    elif mtype == WireMessageType.NAV_GOAL:
                        # Web/app tap → Nav2 goal (NavFn A* + MPPI). Queue for the
                        # control thread (it owns `plant` → /goal_pose). Setting a goal
                        # is a strong driver claim so this client owns the session.
                        try:
                            gx = float(payload.get("x", 0.0))
                            gy = float(payload.get("y", 0.0))
                            gyaw = float(payload.get("yaw", 0.0))
                            # Reject goals outside/blocked on the current 2D map and
                            # tell the client to pick another (NAV_RESULT unreachable).
                            src = loop_holder.get("map_source")
                            ok = src.goal_reachable(gx, gy) if src else None
                            if ok is False:
                                await ws.send(WireCodec.encode(
                                    {"status": "unreachable", "x": gx, "y": gy},
                                    WireMessageType.NAV_RESULT).decode())
                            else:
                                _nav_goal_pending.append((gx, gy, gyaw))
                                _client_cmd[client_id] = now
                        except (TypeError, ValueError):
                            pass
                    elif mtype == WireMessageType.NAV_GOAL_PREVIEW:
                        # Web/app placed a goal but hasn't pressed Follow yet. Mirror
                        # it to the operator's RViz (/goal_pose_preview) so they can
                        # confirm the goal communicated server-side, and run the SAME
                        # reachability check so an out-of-map tap is flagged on the
                        # spot — WITHOUT driving (that waits for NAV_GOAL on Follow).
                        try:
                            gx = float(payload.get("x", 0.0))
                            gy = float(payload.get("y", 0.0))
                            gyaw = float(payload.get("yaw", 0.0))
                            src = loop_holder.get("map_source")
                            ok = src.goal_reachable(gx, gy) if src else None
                            if ok is False:
                                await ws.send(WireCodec.encode(
                                    {"status": "unreachable", "x": gx, "y": gy},
                                    WireMessageType.NAV_RESULT).decode())
                            else:
                                _nav_preview_pending.append((gx, gy, gyaw))
                        except (TypeError, ValueError):
                            pass
                    elif mtype == WireMessageType.NAV_CANCEL:
                        _nav_goal_pending.append(None)     # control loop clears _nav["active"]
                    elif mtype == WireMessageType.LIST_MAPS:
                        # Saved-room picker: send the list of maps exploration saved.
                        try:
                            import map_library
                            src = loop_holder.get("map_source")
                            loaded = getattr(src, "_loaded_name", None) if src else None
                            await ws.send(WireCodec.encode(
                                {"maps": map_library.list_maps(), "loaded": loaded},
                                WireMessageType.MAP_LIST).decode())
                        except Exception as e:                       # noqa: BLE001
                            log.warning("list_maps failed: %s", e)
                    elif mtype == WireMessageType.LOAD_MAP:
                        # Load a saved room (its 3D mesh + 2D map) → the /map clients;
                        # the live sensor cloud + odom keep overlaying. "" / "live"
                        # returns to the live SLAM map.
                        try:
                            import map_library
                            name = str(payload.get("name") or "").strip()
                            src = loop_holder.get("map_source")
                            if name in ("", "live", "none"):
                                if src:
                                    src.resume_live()
                                loaded = None
                            else:
                                grid = map_library.load_grid_frame(name)
                                mesh = map_library.load_mesh_chunks(name)
                                if src:
                                    src.load_saved_map(name, grid, mesh)
                                else:                                # no ROS — push direct
                                    if mesh:
                                        MAP_HUB.publish_mesh(mesh)
                                    if grid:
                                        MAP_HUB.publish_grid(grid)
                                loaded = name
                            await ws.send(WireCodec.encode(
                                {"maps": map_library.list_maps(), "loaded": loaded},
                                WireMessageType.MAP_LIST).decode())
                        except Exception as e:                       # noqa: BLE001
                            log.warning("load_map failed: %s", e)
                    elif mtype == WireMessageType.SAVE_MAP:
                        # Serialize the LIVE map to disk under {"name"}. The map
                        # source owns the live 2D occupancy (_grid_cache), so hand
                        # it to map_library — which stays pure file IO + numpy.
                        try:
                            import map_library
                            name = str(payload.get("name") or "").strip()
                            src = loop_holder.get("map_source")
                            map_library.save_map(
                                name, getattr(src, "_grid_cache", None) if src else None)
                            loaded = getattr(src, "_loaded_name", None) if src else None
                            await ws.send(WireCodec.encode(
                                {"maps": map_library.list_maps(), "loaded": loaded},
                                WireMessageType.MAP_LIST).decode())
                        except Exception as e:                       # noqa: BLE001
                            log.warning("save_map failed: %s", e)
                    elif mtype == WireMessageType.DELETE_MAP:
                        # Delete a saved room's whole <name>.* set. If it was the
                        # one on screen, fall back to the live map so clients are
                        # never left rendering a room that no longer exists.
                        try:
                            import map_library
                            name = str(payload.get("name") or "").strip()
                            src = loop_holder.get("map_source")
                            map_library.delete_map(name)
                            if src and getattr(src, "_loaded_name", None) == name:
                                src.resume_live()
                            loaded = getattr(src, "_loaded_name", None) if src else None
                            await ws.send(WireCodec.encode(
                                {"maps": map_library.list_maps(), "loaded": loaded},
                                WireMessageType.MAP_LIST).decode())
                        except Exception as e:                       # noqa: BLE001
                            log.warning("delete_map failed: %s", e)
                    elif mtype == WireMessageType.DATA_COLLECT:
                        # Start/stop a rosbag (MCAP) data-collection session. The
                        # sim thread owns the recorder subprocess, so queue it.
                        _data_collect_pending.append((bool(payload.get("active", True)),
                                                      payload.get("label") or "session"))
                    elif mtype == WireMessageType.POLICY_SELECT:
                        # Select (not run) a learned policy. Selection-only echo:
                        # the server remembers + broadcasts so all clients agree;
                        # a runner plugs into the RobotCommand seam later.
                        name = payload.get("name") or None
                        _policy_state.update(selected=name, running=False,
                                             available=_list_policies())
                    elif mtype == WireMessageType.ANIMATION_SELECT:
                        name = payload.get("name") or None
                        _animation_state.update(selected=name, playing=False,
                                                available=_list_animations())
                    # ── Animation demos ──────────────────────────────────────
                    # QUEUE ONLY. The control thread owns the recorder + player;
                    # see _demo_pending.
                    elif mtype == WireMessageType.DEMO_RECORD:
                        if payload.get("active"):
                            _demo_pending.append(("arm", {
                                "name": payload.get("name") or "demo",
                                "countdown": float(payload.get("countdown", 3.0)),
                                "notes": payload.get("notes") or "",
                                "client": client_id,
                                "recordedBy": payload.get("recordedBy") or "",
                            }))
                        else:
                            _demo_pending.append(("stop", {}))
                    elif mtype == WireMessageType.DEMO_REPLAY:
                        if payload.get("active"):
                            _demo_pending.append(("play", {
                                "name": payload.get("name") or "",
                                "repeats": int(payload.get("repeats", 1) or 1),
                                "speed": float(payload.get("speed", 1.0) or 1.0),
                            }))
                        else:
                            _demo_pending.append(("halt", {}))
                    elif mtype == WireMessageType.DEMO_DELETE:
                        _demo_pending.append(("delete", {"name": payload.get("name") or ""}))
                    elif mtype == WireMessageType.DEMO_LIST:
                        # Cheap (manifests only) and read-only, so it is answered
                        # right here rather than queued behind the control loop —
                        # the library must populate the moment the sheet opens.
                        try:
                            _demo_state["clips"] = _list_demo_clips()
                            await ws.send(WireCodec.encode(
                                {"clips": _demo_state["clips"]},
                                WireMessageType.DEMO_CLIPS).decode())
                        except Exception as e:                       # noqa: BLE001
                            log.warning("demo_list failed: %s", e)
                    elif mtype == WireMessageType.AUDIO_CONFIG:
                        # CONTROL only — mute/unmute the ROBOT's mic. The voice
                        # PCM itself rides :9610; putting a 32 kB/s firehose on
                        # this 50 Hz socket is the starvation we split
                        # perception_server.py out to avoid.
                        _want = bool(payload.get("micEnabled", True))
                        _p = _PLANT[0]
                        if _p is not None and hasattr(_p, "set_mic_enabled"):
                            _p.set_mic_enabled(_want)
                        _audio_state["micEnabled"] = _want
                    elif mtype == WireMessageType.SET_ROBOT_NAME:
                        _apply_rename(payload.get("name") or "")
                    elif mtype == WireMessageType.BODY_CALIBRATION:
                        # Start/cancel the guided 3-pose anthropometry capture. The sim
                        # thread owns `teleop`, so queue the request for it; starting is
                        # a strong driver claim so this client owns the loop while the
                        # operator is measured (their pose frames feed the calibrator).
                        action = payload.get("action")
                        if action in ("start", "cancel", "apply"):
                            _body_cal_pending.append((action, payload.get("user") or "default"))
                            if action in ("start", "apply"):
                                _client_cmd[client_id] = now      # strong claim
                except Exception as e:                           # noqa: BLE001
                    # A malformed frame must not kill the socket — but it must
                    # not vanish either. This swallowed a broken `demo_record`
                    # handler silently (2026-07-29): the app sent, the server
                    # answered nothing, and the log was clean. Naming the message
                    # type turns "the feature does nothing" into a one-line fix.
                    log.warning("dropped %s: %s: %s",
                                getattr(mtype, "value", "?"), type(e).__name__, e)
        except Exception:
            pass
        finally:
            sender.cancel()
            _client_frames.pop(client_id, None)   # drop this client's stale state
            _client_specs.pop(client_id, None)
            _client_overlay.pop(client_id, None)
            _client_cmd.pop(client_id, None)
            _client_stream.pop(client_id, None)
            MONITOR.on_disconnect(client_id)
            _clients["n"] = max(0, _clients["n"] - 1)
            log.info("client disconnected  %s   (%d online)", peer, _clients["n"])

    async def _monitor_heartbeat():
        """Push the per-client status snapshot to monitors at ~5 Hz so their
        gauges (rate, driver, recording) stay live even between protocol events."""
        while True:
            await asyncio.sleep(0.2)
            if not MONITOR.enabled():       # no monitor subscribed → no work at all
                continue
            try:
                MONITOR.tick(_active_client(),
                             _robot_state.get("recording"),
                             _clients["n"])
            except Exception:
                pass

    async def _sim_map_feed():
        """SIM ONLY: feed the :9092 `/map` plane — the one the iOS + Vision Pro
        Maps tabs already render — with the robot pose (ODOM 0x02) and a live
        CLOUD 0x05: the front-camera pointcloud (relayed over localhost from the
        camera server's :9604 plane) merged with the lidar sweep drawn as a red
        ring. On the real robot MapRosSource feeds MAP_HUB from ROS instead;
        this task is only started when that source is absent. Fully
        demand-driven: with no /map subscriber nothing is built, and the :9604
        relay disconnects so the camera server's depth renderer idles."""
        import websockets
        front_verts = [None]           # latest full snapshot's 16-B vertex bytes
        relay_task = [None]

        async def _relay():
            url = f"ws://127.0.0.1:{data_streams.CLOUD_PORTS['front']}/"
            tok = os.getenv("MABEL_TOKEN", "").strip()
            if tok:
                url += f"?key={tok}"
            parts: list[bytes] = []
            while MAP_HUB.enabled():
                try:
                    async with websockets.connect(url, ping_interval=None,
                                                  max_size=2**23) as ws:
                        async for raw in ws:
                            if not MAP_HUB.enabled():
                                return          # nobody watching → free the render
                            if (isinstance(raw, str) or len(raw) < 8
                                    or raw[0] != 0x05):
                                continue
                            parts.append(raw[8:])
                            if raw[1] & 1:      # final chunk → snapshot complete
                                front_verts[0] = b"".join(parts)
                                parts = []
                except Exception:
                    front_verts[0] = None
                    parts = []
                    await asyncio.sleep(2.0)
            front_verts[0] = None

        red = np.array([255, 64, 48], np.uint8)
        sent = None
        while True:
            await asyncio.sleep(0.2)
            if not MAP_HUB.enabled():
                continue
            if data_streams.streams_enabled() and (
                    relay_task[0] is None or relay_task[0].done()):
                relay_task[0] = asyncio.create_task(_relay())
            base = _robot_state.get("base") or {}
            q = base.get("quat") or [1.0, 0.0, 0.0, 0.0]
            MAP_HUB.publish_odom(encode_odom(
                base.get("x", 0.0), base.get("y", 0.0), base.get("z", 0.0),
                q[1], q[2], q[3], q[0]))          # wire order qx qy qz qw
            xyz_l, rgb_l = [], []
            fv = front_verts[0]
            if fv:
                arr = np.frombuffer(fv, dtype=np.uint8).reshape(-1, 16)
                xyz_l.append(arr[:, :12].copy().view("<f4").reshape(-1, 3))
                rgb_l.append(arr[:, 12:15].copy())
            lp = _lidar_pts[0]
            if lp is not None and len(lp):
                xyz_l.append(lp)
                rgb_l.append(np.tile(red, (len(lp), 1)))
            if not xyz_l:
                continue
            xyz = np.concatenate(xyz_l)
            key = (xyz.shape[0], _robot_state.get("t"),
                   None if fv is None else len(fv))
            if key == sent:                       # nothing new → don't rebroadcast
                continue
            sent = key
            MAP_HUB.publish_cloud(encode_cloud_chunks(xyz, np.concatenate(rgb_l)))

    async def _serve():
        loop = asyncio.get_running_loop()
        loop_holder["loop"] = loop
        MONITOR.attach_loop(loop, _hello_info.get("version", ""))
        asyncio.create_task(_monitor_heartbeat())

        # Live SLAM map producer: a background ROS subscriber (/nvblox_node/
        # mesh_pointcloud + /odom) that fans binary mesh + odom frames out to
        # /map clients. No-op (returns None) when ROS isn't importable, so the
        # sim/Mac path without a sourced ROS env still runs — /map just stays idle.
        # ⚠ MUST honour MABEL_PERCEPTION_EXTERNAL: this was the ONE heavy plane
        # the flag forgot. With perception external the :9092 binding below is
        # (correctly) skipped — but this source still spun rclpy executors that
        # python-deserialised every nvblox MarkerArray + cloud INSIDE the gateway,
        # for a hub with no listener. On thor that starved the control loop's GIL
        # to 4 Hz idle / 0-1 Hz under retarget ("whole-body is slow af / nothing
        # tracks", 2026-07-19, py-spy: mabel-map-ros active in msg __init__).
        if _PERCEPTION_EXTERNAL:
            loop_holder["map_source"] = None    # perception_server.py owns /map
        else:
            MAP_HUB.attach_loop(loop)
            loop_holder["map_source"] = start_map_stream(MAP_HUB)
        # SIM: rclpy is absent on the Mac, so start_map_stream returns None and
        # the /map plane the apps' Maps tabs render would sit idle. Feed it
        # locally (pose + front-camera cloud + lidar ring) so pointclouds and
        # laser data show up in the Vision Pro / iOS apps with NO app change.
        # Skipped whenever a ROS source exists (real robot, sim:*+ros).
        if not _IS_REAL and loop_holder.get("map_source") is None:
            asyncio.create_task(_sim_map_feed())
            log.info("map feed     SIM-local: pose + front cloud + lidar ring → /map (:%d)", MAP_PORT)

        # MJPEG cameras — only when the bridge renders in-process (--cameras).
        runner = None
        if path_caps:
            mjpeg = MJPEGServer.from_captures(path_caps)
            mjpeg.start_captures(loop)                   # binds the loop to each PushCapture
            _console_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "website"))
            if mjpeg.add_console(_console_dir):
                log.info("web console  http://%s:%d/console", console.lan_ip(), http_port)
            runner = web.AppRunner(mjpeg.app())
            await runner.setup()
            await web.TCPSite(runner, http_host, http_port).start()
            log.info("cameras up   http://%s:%d", console.lan_ip(), http_port)
            for c in CAMERAS:
                if c.weight:
                    log.info(console.kv(c.label, f"→ {c.path}"))

        # WebSocket teleop
        ws_server = None
        if enable_ws:
            # write_limit caps the per-connection send buffer. The default 64 KB
            # holds ~22 of our ~3 KB state frames, so on a brief Wi-Fi/LAN hiccup
            # ws.send() returns instantly and queues up to ~440 ms of STALE frames
            # that then replay late (the "freeze then jump"). 8 KB (~3 frames) makes
            # send() apply backpressure almost immediately, so the 50 Hz loop blocks
            # and coalesces to the FRESHEST state instead of buffering history.
            # compression=None: DISABLE permessage-deflate on the teleop socket.
            # robot_state is a few KB and LATENCY-CRITICAL; deflating every frame
            # (measured 4.5% of the gateway in py-spy, on the GIL) bought nothing on
            # LAN/Tailscale but CPU + delay. The heavy planes (mesh/cloud) keep their
            # own sockets and can compress there if it ever helps.
            # max_queue=2: the library default (32) let a GIL-starved net thread
            # accumulate a backlog of inbound teleop frames, each still paying a
            # full json.loads (~900-float skeleton) before the newest-wins slot
            # dropped it. Two slots + TCP backpressure = the VP's own in-flight
            # cap (2) ends the queueing at the socket instead of after decode.
            ws_server = await websockets.serve(
                _ws_handler, ws_host, ws_port, ping_interval=None, max_size=2**20,
                write_limit=2**13, compression=None, max_queue=2)
            log.info("teleop up    ws://%s:%d%s", console.lan_ip(), ws_port, ws_path)

        # Dedicated SLAM-map WebSocket on its OWN port, so the heavy mesh stream is
        # fully isolated from teleop + cameras (the browser map connects here). Any
        # path is accepted; gated by the same token as teleop.
        map_server = None
        if enable_ws and MAP_PORT and not _PERCEPTION_EXTERNAL:
            async def _map_handler(ws, path=None):
                req_path = path or getattr(ws, "path", None) or getattr(
                    getattr(ws, "request", None), "path", "/")
                from urllib.parse import urlsplit, parse_qs
                parts = urlsplit(req_path)
                token = os.getenv("MABEL_TOKEN", "").strip()
                supplied = (parse_qs(parts.query).get("key", [""])[0]).strip() if token else ""
                if token and supplied != token:
                    await ws.close(1008, "unauthorized")
                    return
                log.info("map client connected    %s  (:%d)", _peer_str(ws), MAP_PORT)
                await MAP_HUB.add_subscriber(ws)
                log.info("map client disconnected %s", _peer_str(ws))
            map_server = await websockets.serve(
                _map_handler, ws_host, MAP_PORT, ping_interval=None, max_size=2**20)
            log.info("map up       ws://%s:%d  (live SLAM mesh + odom)", console.lan_ip(), MAP_PORT)

        # Dedicated raw-LiDAR WebSocket: SCAN 0x06 frames only (map_stream
        # SCAN_HUB). Same token gate as the map port.
        lidar_server = None
        if enable_ws and LIDAR_PORT:
            from map_stream import SCAN_HUB

            async def _lidar_handler(ws, path=None):
                req_path = path or getattr(ws, "path", None) or getattr(
                    getattr(ws, "request", None), "path", "/")
                from urllib.parse import urlsplit, parse_qs
                parts = urlsplit(req_path)
                token = os.getenv("MABEL_TOKEN", "").strip()
                supplied = (parse_qs(parts.query).get("key", [""])[0]).strip() if token else ""
                if token and supplied != token:
                    await ws.close(1008, "unauthorized")
                    return
                log.info("lidar client connected    %s  (:%d)", _peer_str(ws), LIDAR_PORT)
                await SCAN_HUB.add_subscriber(ws)
                log.info("lidar client disconnected %s", _peer_str(ws))
            SCAN_HUB.attach_loop(loop)
            lidar_server = await websockets.serve(
                _lidar_handler, ws_host, LIDAR_PORT, ping_interval=None, max_size=2**16)
            log.info("lidar up     ws://%s:%d  (raw LaserScan 0x06)", console.lan_ip(), LIDAR_PORT)

        # ── 96xx data plane (read-only broadcast; docs/data_streams.md) ────────────
        # :9600 telemetry always lives in the bridge. :9605 SLAM+map binds here
        # too: on the real robot a ROS source feeds it (nvblox mesh + cuVSLAM
        # pose + /map grid); in sim it serves an explicit notice — SLAM does not
        # run on the Mac. (Cloud :9601-9604 bind in the CAMERA server in sim;
        # on the real robot the bridge binds them — see _start_real_data_planes.)
        telem_server = slam_server = audio_server = None
        cloud_servers: list = []
        if enable_ws and data_streams.streams_enabled():
            TELEMETRY.set_providers(hello=_telem_hello, status=_telem_status,
                                    collection=_telem_collection,
                                    network=_telem_network, lidar=_telem_lidar)
            TELEMETRY.attach_loop(loop)
            telem_server = await data_streams.serve_hub_ws(
                ws_host, data_streams.TELEMETRY_PORT, TELEMETRY, "telemetry", log,
                compression=None)     # small + real-time proprio/control — no deflate

            # :9610 voice — the ONE duplex plane (data_streams.py documents why
            # that exception is safe). Hosted HERE, not in perception_server: it
            # is a memcpy with no numpy/PointCloud2 work, so it cannot thrash the
            # GIL the way the map planes did, and the speaker downlink needs the
            # plant handle that only this process holds.
            # Bind UNCONDITIONALLY: this thread starts long before make_plant(),
            # so there is no plant to ask here (asking one is what took the whole
            # network thread down with a NameError — and with it the :9090 teleop
            # socket every app connects to; both lines hit this independently,
            # 2026-07-16/18). The control thread wires the sinks in once the
            # plant exists; until then the hub answers with an honest notice
            # rather than a refused connection. This SUPERSEDES the deferred
            # _bind_audio_plane approach — the demand-driven mic wiring in
            # main() (AUDIO.set_demand_cb) is what keeps an idle voice plane
            # from dragging the 50 Hz proprio stream.
            AUDIO.attach_loop(loop)
            audio_server = await data_streams.serve_hub_ws(
                ws_host, data_streams.AUDIO_PORT, AUDIO, "audio", log,
                compression=None)     # PCM is incompressible + latency-critical
            # :9605 SLAM+map — real: nvblox mesh + cuVSLAM pose + /map grid via
            # ROS; sim (or ROS unavailable): an explicit notice, never a refusal.
            # Skipped entirely when perception_server.py owns the heavy planes.
            slam_hub = None
            if _IS_REAL and not _PERCEPTION_EXTERNAL:
                try:
                    from slam_stream import SlamHub, start_slam_stream
                    sh = SlamHub()
                    sh.attach_loop(loop)
                    if start_slam_stream(sh, log) is not None:
                        slam_hub = sh
                        loop_holder["slam_source"] = True
                except Exception as e:                             # noqa: BLE001
                    log.warning("slam stream unavailable: %s", e)
            # When perception_server.py owns :9605 we must NOT bind it here (port
            # conflict) — leave slam_server None. Otherwise serve the hub (or a
            # sim/no-ROS notice).
            if not _PERCEPTION_EXTERNAL:
                if slam_hub is None:
                    slam_hub = data_streams.NoticeSource(
                        "SLAM (nvblox/cuVSLAM) runs on the real robot only — no mesh in sim"
                        if not _IS_REAL else "SLAM stack not running (rclpy/ROS unavailable)",
                        data_streams.SLAM_STREAM_PORT)
                slam_server = await data_streams.serve_hub_ws(
                    ws_host, data_streams.SLAM_STREAM_PORT, slam_hub, "slam", log)
            # :9601-9604 pointclouds — REAL ONLY here (rclpy lives in this
            # process); in sim the camera server owns these ports (depth render).
            # Skipped when perception_server.py owns the cloud planes.
            if _IS_REAL and not _PERCEPTION_EXTERNAL:
                try:
                    from cloud_streams import CloudHubSet, RosCloudSource
                    avail = {k: True for k in data_streams.CLOUD_PORTS}
                    hubset = CloudHubSet(avail, log, host=ws_host)
                    for key, hub in hubset.hubs.items():
                        hub.attach_loop(loop)
                        srv = await data_streams.serve_hub_ws(
                            ws_host, hubset.ports[key], hub, f"cloud:{key}", log)
                        cloud_servers.append(srv)
                    if RosCloudSource(hubset, log).start() is None:
                        log.warning("ros clouds: no source — cloud ports serve "
                                    "no data until ROS is up")
                except Exception as e:                             # noqa: BLE001
                    log.warning("cloud planes unavailable: %s", e)

        # ── Per-module connection status → robot_state.modules ───────────────
        # Which limb/sensor is actually attached RIGHT NOW. On the real robot a
        # subsystem whose device is gone publishes NOTHING (arms_node bails on
        # `if not names: return`), so topic liveness is the honest signal; in sim
        # the MJCF joints are connected by definition, but the SENSORS are
        # measured for real (--head-only genuinely serves fewer cameras).
        try:
            from module_status import (ModuleTracker, RosModuleSource,
                                       SimModuleSource)
            _video_url = f"http://127.0.0.1:{http_port}/cameras.json"
            if _IS_REAL:
                _src = RosModuleSource(MODULES, _video_url, log).start()
                # No source ⇒ ship NO roster at all (MODULES_LIVE stays False).
                # Shipping one anyway would broadcast 13 "not connected" rows and
                # paint a perfectly healthy robot entirely red — worse than saying
                # nothing. The app then shows WAITING, which is the truth.
                if _src is None:
                    log.warning("module status: no ROS source — roster NOT sent "
                                "(the app shows 'waiting', not a false 'all down')")
                else:
                    MODULES_LIVE[0] = True
            else:
                # The joints must be read LAZILY: this thread starts before the
                # physics loop writes its first snapshot, so passing the list here
                # would freeze an empty {} and report every limb disconnected.
                SimModuleSource(MODULES,
                                lambda: list(_robot_state.get("joints") or {}),
                                _video_url,
                                lambda: _robot_state.get("lidar") or latest_scan(),
                                log).start()
                MODULES_LIVE[0] = True
                log.info("module status: sim source → robot_state.modules")
        except Exception as e:                                     # noqa: BLE001
            log.warning("module status unavailable: %s", e)

        ready.set()
        try:
            await asyncio.Future()       # run until the process exits
        finally:
            src = loop_holder.get("map_source")
            if src is not None:
                src.stop()
            if ws_server is not None:
                ws_server.close()
            if map_server is not None:
                map_server.close()
            if lidar_server is not None:
                lidar_server.close()
            for srv in (telem_server, slam_server, audio_server, *cloud_servers):
                if srv is not None:
                    srv.close()
            if runner is not None:
                await runner.cleanup()

    try:
        asyncio.run(_serve())
    except OSError as e:
        if getattr(e, "errno", None) in (48, 98):
            log.error("Port %d or %d already in use — a STALE bridge is "
                      "probably still running (old code!). Free it:  "
                      "lsof -ti :%d | xargs kill -9 ; lsof -ti :%d | xargs kill -9",
                      ws_port, http_port, ws_port, http_port)
        else:
            log.error("Network thread error: %s", e)
        loop_holder["failed"] = True
        ready.set()
    except Exception:                                              # noqa: BLE001
        # ANY startup error must surface as a refused run, not a silent 10 s
        # ready-timeout with a dead :9090 (how the audio-plane NameError hid).
        log.exception("Network thread crashed")
        loop_holder["failed"] = True
        ready.set()


# ---------------------------------------------------------------------------
# stdin watcher → ENTER recalibrates, "r" resets the sim
# ---------------------------------------------------------------------------

def _advertise_bonjour(ws_port: int) -> None:
    """Advertise _mabel-teleop._tcp on the LAN so clients AND the command monitor
    auto-discover this bridge — no hardcoded IP. Works on macOS (`dns-sd`) and on
    the Linux real robot (`avahi-publish-service`), so a physical robot shows up
    as a host in the monitor automatically, exactly like a sim does.

    mDNS is LAN-scoped; to reach a robot across networks (e.g. Tailscale) the
    monitor still takes an explicit --hosts / MABEL_HOSTS entry."""
    kind = "Robot" if _DISPATCH_TARGET.startswith("real") else "Sim"
    name = f"MABEL {kind} ({socket.gethostname().split('.')[0]})"
    # Pick the platform's mDNS responder: dns-sd (macOS) or Avahi (Linux).
    # TXT: the identity as data (NOVA's gateway sends robot=NOVA on the same service type).
    txt = ["robot=MABEL", f"kind={'real' if kind == 'Robot' else 'sim'}"]
    if sys.platform == "darwin" and shutil.which("dns-sd"):
        cmd = ["dns-sd", "-R", name, "_mabel-teleop._tcp", "local", str(ws_port), *txt]
    elif shutil.which("avahi-publish-service"):
        cmd = ["avahi-publish-service", name, "_mabel-teleop._tcp", str(ws_port), *txt]
    else:
        log.warning("No dns-sd/avahi-publish — install avahi-utils so this %s is "
                    "discoverable, or type its IP in the monitor (--hosts).",
                    kind.lower())
        return
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        atexit.register(proc.terminate)
        log.info("Bonjour: advertising '%s' _mabel-teleop._tcp on :%d "
                 "(auto-discovered by clients + the command monitor)", name, ws_port)
    except Exception as e:
        log.warning("Bonjour advertisement failed: %s", e)


def start_stdin_thread() -> None:
    def _watch():
        try:
            for line in iter(sys.stdin.readline, ""):
                if line.strip().lower() == "r":
                    _reset.set()
                else:
                    _recalibrate.set()
        except Exception:
            pass
    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------

_SIM_SCENES = os.path.normpath(
    os.path.join(_HERE, "..", "simulation", "mabel_mujoco", "scenes"))


def _resolve_scene(scene_arg: str | None) -> str:
    if scene_arg:
        return scene_arg if os.path.isabs(scene_arg) else os.path.abspath(scene_arg)
    # Default to the furnished home scene next to the simulation package. (The
    # controller package moved to ~/open_mabel/controller, so we resolve scenes from
    # the simulation dir directly rather than relative to the package.)
    return os.path.join(_SIM_SCENES, "mabel_home.xml")


def _open_state_publisher(path: str, scene: str, model):
    """Create a float64 memmap [time, qpos...] that view_robot.py mirrors, plus a
    sidecar text file naming the scene so the viewer loads an identical model, plus
    a perturbation memmap (nbody×6 xfrc) the viewer WRITES to — so a Ctrl+drag in
    the desktop window pushes the real sim (yields in compliance, resists in
    impedance). Returns (state, pert) memmaps (or (None, None) on failure)."""
    try:
        with open(path + ".scene", "w") as f:
            f.write(scene + "\n")
        state = np.memmap(path, dtype=np.float64, mode="w+", shape=(1 + model.nq,))
        state[:] = 0.0
        pert = np.memmap(path + ".pert", dtype=np.float64, mode="w+", shape=(model.nbody, 6))
        pert[:] = 0.0
        log.info("state file   %s  (desktop mirror: mjpython view_robot.py)", path)
        return state, pert
    except Exception as e:
        log.warning("Could not open state publisher (%s); desktop viewer disabled", e)
        return None, None


# ---------------------------------------------------------------------------
# Main sim + render loop (main thread)
# ---------------------------------------------------------------------------

def run(args) -> None:
    scene = _resolve_scene(args.scene)
    # Echo WHICH scene this sim is running in hello (additive) — the apps show
    # it and mark the active card in their scene library. Title comes from the
    # scene registry (scenes.json) when the entry exists.
    _scene_meta = {"xml": os.path.basename(scene),
                   "id": os.path.splitext(os.path.basename(scene))[0]}
    try:
        with open(os.path.join(os.path.dirname(scene), "scenes.json")) as _f:
            for _e in json.load(_f):
                if _e.get("xml") == _scene_meta["xml"]:
                    _scene_meta["id"] = _e.get("name", _scene_meta["id"])
                    _scene_meta["title"] = _e.get("title", "")
                    break
    except Exception:
        pass
    _hello_info["scene"] = _scene_meta
    _scene_path[0] = scene
    if not _IS_REAL:
        _refill_spawn_pool(scene)      # warm the random-reset pose pool
    log.info("MABEL unified teleop server %s", _hello_info["version"])
    log.info("scene        %s", scene)
    model = mujoco.MjModel.from_xml_path(scene)

    # Physics real-time tuning (teleop-only; the shared scene XML is untouched).
    # MEASURED (Jun 2026, this 108-DOF model w/ the ORCA hands): mj_step costs a
    # FIXED ~8.4 ms mean (p95 ~14) per step, REGARDLESS of timestep / solver iters
    # / cone / integrator (Newton exits early; collision is only 0.24 ms). The
    # control law — incl. hand-retargeting + arm IK in drive mode — is ~0.03–1 ms,
    # i.e. NOT the bottleneck. So the ONE lever for latency is the timestep: more
    # sim-time per (fixed-cost) step ⇒ higher real-time factor. At 2 ms the sim
    # runs 0.24x real-time (commands lag, jumpy); 8 ms ≈ 0.96x; 10 ms ≈ 1.19x.
    # Real-time factor must exceed 1 or the accumulator never drains and latency
    # pins at MAX_CATCHUP.
    #
    # STABILITY CEILING (measured Jun 2026 on mabel_home): the loose tabletop
    # objects in home_objects.xml (plate, mugs — free joints) sit on a stiff
    # island contact whose home |qacc| is already ~6.4e4. That contact is only
    # integrable at a FINE step: 4 ms is stable (5 s, no divergence); at ≥6 ms
    # MuJoCo trips "Nan/Inf in QACC" and silently resets `data` to qpos0 — which
    # presents as the robot "resetting" every fraction of a second and makes
    # teleop impossible. So the coarse DRIVE step that would buy >1x real-time is
    # NOT usable with this scene; default DRIVE == MANIP == 4 ms (the value the
    # bridge ran before the two-timestep change). Override MABEL_TIMESTEP_DRIVE
    # only with a scene whose manipulation objects are removed or better
    # conditioned. (base/lift/neck physics alone ARE fine coarse — the objects
    # are the limiter.)
    TS_MANIP = float(os.getenv("MABEL_TIMESTEP", "0.004"))
    TS_DRIVE = float(os.getenv("MABEL_TIMESTEP_DRIVE", "0.004"))
    model.opt.timestep = TS_MANIP

    # The scene's 360-beam rangefinder array (simulated LiDAR) ray-casts EVERY mj_step
    # and DOMINATES its cost: measured ~12.7 ms/step with sensors ON vs ~2.0 ms OFF on
    # this 108-DoF home scene — i.e. 0.31x vs 2.03x real-time at the 4 ms step. On the
    # plain Mac sim path NOTHING consumes that sensordata (the cameras render in a
    # SEPARATE process from the qpos memmap; SLAM/Nav/IMU run only on the "+ros" path),
    # so disable sensors here. This is THE fix for "the whole server runs at ~5-7 Hz /
    # the robot moves at half speed": with them off the loop runs faster than real-time
    # and the accumulator drains. The ROS sim path keeps sensors (its IMU/LiDAR feed
    # SLAM); the real-robot path has no mj_step at all.
    if _DISPATCH_TARGET in ("sim:mujoco", "sim:bridge"):
        # sim:bridge runs a kinematic MIRROR here (mj_forward per substep) —
        # position-stage rangefinders would ray-cast in the mirror too, so the
        # same disable applies; the simulation_bridge world makes its own call.
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
        log.info("sensors DISABLED on the Mac sim path (LiDAR unused here); "
                 "the +ros path keeps them for SLAM")

    # MEASURE mj_step on THIS host/scene (after the sensor decision above) so the
    # real-time factor we report is HONEST and self-diagnosing — if it ever drops
    # toward 1x the operator sees it right here. This replaces a fixed "[mj_step≈8.4
    # ms] ~0.48x" annotation that long outlived reality: with sensors off the real
    # step is ~0.4 ms (~10x real-time on an M1 Max), so the old line falsely accused
    # physics of being the bottleneck. Probe a scratch MjData so live state is untouched.
    _probe = mujoco.MjData(model)
    for _ in range(20):
        mujoco.mj_step(model, _probe)            # warm caches / JIT
    _t0 = time.perf_counter()
    for _ in range(60):
        mujoco.mj_step(model, _probe)
    _step_ms = (time.perf_counter() - _t0) / 60.0 * 1e3
    _rt = (TS_MANIP / (_step_ms * 1e-3)) if _step_ms > 0 else 0.0
    log.info("timestep: drive %.1f ms / manip %.1f ms  [mj_step≈%.2f ms → ~%.1fx real-time]",
             TS_DRIVE * 1e3, TS_MANIP * 1e3, _step_ms, _rt)
    if _rt < 1.5:
        log.warning("physics real-time margin is thin (~%.1fx) — joints may lag under "
                    "render/CPU contention; check sensors-off and camera-server priority", _rt)

    # Hold off macOS App Nap / timer coalescing. The bridge is a BACKGROUND (window-less)
    # process; the moment a FOREGROUND GUI runs — the desktop --viewer (mjpython) or the
    # Command Monitor — macOS throttles our timers to save power and the control loop's
    # `time.sleep` pacing collapses from ~240 Hz to ~5-12 Hz (measured), dragging the sim
    # to a fraction of real-time for EVERY client. Declaring a user-initiated,
    # latency-critical activity exempts us, so opening the viewer no longer starves teleop.
    global _NS_ACTIVITY
    try:
        from Foundation import NSProcessInfo
        # NSActivityUserInitiated (0x00FFFFFF) | NSActivityLatencyCritical (0xFF00000000)
        _NS_ACTIVITY = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
            0x00FFFFFF | 0xFF00000000, "MABEL teleop real-time control loop")
        log.info("App Nap / timer-coalescing disabled (stays real-time with the viewer open)")
    except Exception as e:
        log.info("App Nap guard unavailable (%s) — keep the desktop --viewer closed while teleoperating", e)

    # Optional head-camera FOV override (--head-fov / MABEL_HEAD_FOV): widen
    # the simulated ZED for a more wrap-around immersive view. The headset
    # learns the effective intrinsics from the hello and reshapes its panel.
    head_fovy = HEAD_FOVY_DEFAULT
    if args.head_fov is not None:
        for cam in ("zed_left", "zed_right"):
            try:
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                if cid >= 0:
                    model.cam_fovy[cid] = float(args.head_fov)
            except Exception:
                pass
        head_fovy = float(args.head_fov)
        log.info("Head camera fovy overridden → %.1f°", head_fovy)
    else:
        try:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_left")
            if cid >= 0:
                head_fovy = float(model.cam_fovy[cid])
        except Exception:
            pass
    # Head aspect for the immersive panel. This is the TANGENT-FOV ratio
    # (tan(hfov/2)/tan(fovy/2)), which equals the render pixel W/H for the sim's
    # square-pixel pinhole. When serving the REAL stereo camera (start_teleop.sh
    # --real), MABEL_HEAD_ASPECT carries the rectified camera's true ratio (from
    # its calibration) so the Vision Pro panel subtends the real FOV → life-size.
    _head_aspect = float(args.head_w) / float(args.head_h)
    if os.getenv("MABEL_HEAD_ASPECT"):
        try:
            _head_aspect = float(os.environ["MABEL_HEAD_ASPECT"])
            log.info("Head camera aspect overridden → %.4f (real stereo calibration)", _head_aspect)
        except ValueError:
            pass
    _hello_info["cameras"] = {"head": {"fovy": head_fovy, "aspect": _head_aspect}}
    # Advertise the low-latency UDP video port even in split mode — there the
    # port is bound by sim_camera_server.py on this same host (same default).
    if args.udp_video_port:
        # h264Port: the camera server (sim_camera_server.py / real_camera_server.py) binds
        # H.264 over UDP here (same default port across sim AND the real robot), so clients
        # subscribe to ONE codec everywhere. 0 ⇒ H.264 off → clients use UDP-JPEG / MJPEG.
        _h264p = (int(os.getenv("MABEL_SIM_H264_PORT", "9092"))
                  if os.getenv("MABEL_SIM_H264", "0").lower() not in ("0", "false", "no", "off")
                  else 0)
        _hello_info["video"] = {"udpPort": int(args.udp_video_port), "h264Port": _h264p}

    data = mujoco.MjData(model)
    # Seed the "home" keyframe before the first step. A bare MjData has
    # ctrl == 0 while the lift stages assemble at mid-travel, so the Lift
    # servo (kp 80000) would slam them to the bottom — in sim:mujoco mode
    # that impulse tips the robot over at launch (same fix as the
    # simulation_bridge engine boot; in sim:bridge mode this data is only
    # the kinematic mirror, where the seed is harmless).
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)

    # Verify the cameras exist before we promise to stream them.
    have = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)}
    needed = {c.name for c in CAMERAS if c.weight}
    missing = needed - have
    if missing:
        log.warning("Scene is missing cameras %s — those streams will be blank. "
                    "Present cameras: %s", sorted(missing), sorted(have))

    # The ONE shared control facade — retarget + whole-body IK + swerve + neck, the
    # torque engine (gravity-comp / compliance / joint impedance), the double-pinch
    # clutch, and the tracking-loss fail-safe — identical to the one the desktop sim
    # viewer uses. Every client (iOS, Vision Pro, web) drives THIS object; the bridge
    # just decodes the wire and steps physics. See controller/mabel/facade.py.
    teleop = MabelTeleop(model, data, base_follow=not args.no_base,
                         fail_safe=True, clutch=True)
    controller = teleop.controller        # telemetry/status reads (vx/vy/omega, status())
    wbc = teleop.wbc                       # monitor tap reads (managed/_q_des/active)
    # Share the net-thread command mailboxes with the facade IN PLACE: the WebSocket
    # handler keeps writing _joint_cmd / _ext_forces; the facade reads & re-homes the
    # same dicts (teleop.reset() clears them in place), so no extra plumbing is needed.
    teleop.joint_targets = _joint_cmd
    teleop.ext_forces = _ext_forces

    # JOINT SPEED LIMIT — REAL ROBOT ONLY (safety). The arm firmware closes
    # kp*(pos - q) itself, so the ONLY way this host can bound arm torque is to bound
    # the commanded ERROR, and the HAL deliberately carries no slew/stall/thermal
    # guard any more (it clamps t_ff to the motor's rated t_max and position to the
    # calibrated end-stops, nothing else — hardware_bridge/README.md). Without this a
    # slider dragged to an extreme is an instant multi-rad error: ~99 N.m at kp=30,
    # which parks the joint on a stop and latches the DaMiao's over-current trip (a
    # PHYSICAL power cycle to clear). With it, the joint slews at MABEL_ARM_MAX_SPEED
    # and the position term stays ~1 N.m. Sim keeps instantaneous posing (no motors to
    # cook, and the e2e tests assert the direct override lands verbatim).
    # ⚠ OFF unless MABEL_ARM_MAX_SPEED is set explicitly, and it MUST NOT be enabled
    # until the gravity feedforward is wired into /arms/command (JointState.effort).
    # WHY: the limiter caps the POSITION term (kp*error), and with t_ff = 0 that term
    # is the only thing fighting gravity. l_j2 needs ~16.7 N.m just to HOLD at
    # extension, while the limiter caps it at kp*v*dt = 30*4.0*0.02 = 2.4 N.m — so the
    # arm physically cannot lift itself and teleop/animations die (observed
    # 2026-07-16). Making max_speed large enough to hold j2 would need ~28 rad/s,
    # i.e. no limit at all. With t_ff carrying tau_g the position term only carries
    # the small residual, and a tight cap costs nothing — that is the ONLY
    # configuration where this belongs on.
    # 2026-07-29: ON BY DEFAULT — the gravity-FF precondition has been satisfied
    # since tau_g started riding /arms/command.effort, and with the limiter OFF the
    # green-ball task moves landed as VERBATIM step targets (the "super jitter" +
    # no interpolation report). JointGlideLimiter integrates a smooth setpoint from
    # the previous command (accel-limited, decelerates into the target; far jumps
    # blend at 1 rad/s → a 2-3 rad task move takes 2-3 s) while keeping the
    # measured-error torque bound as an explicit clamp. MABEL_ARM_MAX_SPEED=0
    # disables; any other value overrides the tracking speed.
    # 2026-07-30: this limiter is now the facade's UNIVERSAL WIRE GATE
    # (facade._glide_wire_gate): in retarget/navigation it bounds the FINAL
    # Arm/Torso/Wrist ctrl whoever wrote it — the retarget engine and the
    # engagement-supervisor ease used to bypass it entirely (arm keys are popped
    # from joint_targets in retarget), so whole-body engagement far from the
    # operator's pose sent unbounded kp*error and latched the J1/J2 over-current
    # trip. Same limiter instance = whole-body teleop gets the exact web
    # task-space glide feel. Tests: controller/tests/test_retarget_wire_gate.py.
    _max_speed = float(os.getenv("MABEL_ARM_MAX_SPEED") or "4.0")
    if _IS_REAL and _max_speed > 0:
        from mabel.arm_control import JointGlideLimiter
        teleop.arm_rate_limit = JointGlideLimiter(
            max_speed=_max_speed,
            blend_speed=float(os.getenv("MABEL_ARM_BLEND_SPEED") or "1.0"),
            err_max=float(os.getenv("MABEL_ARM_ERR_MAX") or "0.8"))
        _em = teleop.arm_rate_limit.err_max
        log.info("arm joint glide: tracking %.2f rad/s, blend %.2f rad/s, "
                 "err ceiling %s (thermal yield: >%.2f rad for %.1fs -> %.2f rad hold)",
                 teleop.arm_rate_limit.max_speed,
                 teleop.arm_rate_limit.blend_speed,
                 ("none — unlimited spring" if _em == float("inf")
                  else f"{_em:.2f} rad"),
                 teleop.arm_rate_limit.yield_err,
                 teleop.arm_rate_limit.yield_after,
                 teleop.arm_rate_limit.yield_rest)

    # Self-collision avoidance — ON by default (safety). Opt out explicitly with
    # --no-self-collision / MABEL_SELF_COLLISION=0; applies to both the kinematic
    # teleop (stop-at-margin) and the WBC compliance path (soft wall).
    teleop.set_self_collision(args.self_collision)
    log.info("Self-collision avoidance: %s", "ON" if args.self_collision else "OFF")

    # ── SINGLE control backend: the MabelTeleop facade ────────────────────────
    # The facade is the ONE entry point. Internally `teleop.controller` is the
    # scripts TeleopEngine (HandOptimizer fingers + cosine ArmRetargeter +
    # WholeBodyController + gravity-comp + fail-safe + clutch, heavy SLSQP paced once
    # per VR frame) — it replaced the dated TeleopController. The facade routes per
    # control_mode: navigation/retarget → the engine; the WBC torque modes (Body-tab
    # "joint"/"wrist" + compliance push) → its WBCEngine. No second backend, so the
    # base sign convention can never diverge again.
    controller.init_hold()      # engine: write rest targets onto the position servos
    controller.settle_done()    # engine: anchor the lift/base frame after settle

    # ── Cameras (in-process, OPTIONAL) ───────────────────────────────────────
    # By default the bridge does NOT render: rendering is a ~40 ms/cam blocking
    # GPU readback on macOS that competes with physics on this single thread.
    # Run the dedicated `sim_camera_server.py` instead (separate process/core),
    # which reads the qpos this bridge publishes and serves the MJPEG streams.
    # Pass --cameras to render in-process anyway (simpler to launch, slower sim).
    path_caps: dict[str, PushCapture] = {}
    cam_to_paths: dict[str, list[str]] = {}
    cam_to_res: dict[str, tuple[int, int] | None] = {}
    if args.cameras:
        for c in CAMERAS:
            if not c.weight:
                continue
            path_caps[c.path] = PushCapture(CameraSource(name=c.label, device="sim", path=c.path))
            cam_to_paths.setdefault(c.name, []).append(c.path)
            cam_to_res[c.name] = c.res
        if args.udp_video_port:
            try:
                from udp_video import UDPVideoSender
                udp_video = UDPVideoSender(port=args.udp_video_port)
                udp_video.start()
                udp_video.attach(path_caps)
            except OSError as e:
                log.warning("UDP video disabled — port %d unavailable (%s). "
                            "Clients will use the MJPEG fallback.",
                            args.udp_video_port, e)

    # ── Start networking ─────────────────────────────────────────────────────
    _srv_info = _collect_server_info(args)                # bridge's own IPs/ports
    _srv_info["jointLimits"] = _joint_limits(model)       # for the monitor's bars
    _srv_info_ref["d"] = _srv_info                        # so _apply_rename can update it
    MONITOR.set_server_info(_srv_info)
    MONITOR.set_rename_handler(_apply_rename)             # rename from the monitor
    ready = threading.Event()
    loop_holder: dict = {}
    net = threading.Thread(
        target=start_network_thread,
        args=(args.ws_host, args.ws_port, args.ws_path,
              args.http_host, args.http_port, path_caps,
              not args.mock, ready, loop_holder),
        daemon=True,
    )
    net.start()
    ready.wait(timeout=10.0)
    if loop_holder.get("failed"):
        log.error("Refusing to run without the teleop listener — exiting.")
        sys.exit(1)
    if not args.mock:
        start_stdin_thread()
        _advertise_bonjour(args.ws_port)

    # ── Offscreen renderer for the robot's cameras (only with --cameras) ─────
    renderer = None
    render_cams = []
    if args.cameras:
        try:
            renderer = mujoco.Renderer(model, height=args.head_h, width=args.head_w)
        except Exception as e:
            log.error("Could not create '%s' renderer: %s", os.environ["MUJOCO_GL"], e)
            log.error("On a headless box retry with: MUJOCO_GL=egl  (or osmesa)")
            return
        render_cams = [c for c in cam_to_paths if c in have]   # unique sim cameras
        import cv2          # lazy: only the sim camera-render path needs it
    encode = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if args.cameras else None

    def render_one(cam):
        """Render a single camera and push its JPEG to every path that uses it.
        macOS glfw offscreen render() is a ~40 ms fixed GPU-readback stall that
        blocks this thread, so we render ONE camera per loop tick (round-robin)
        instead of all four at once — that keeps the physics accumulator able to
        hold real-time. Each camera still refreshes at cam_fps."""
        import cv2          # already imported in main() when args.cameras is set
        renderer.update_scene(data, camera=cam)
        bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        res = cam_to_res[cam]
        if res is not None:
            bgr = cv2.resize(bgr, res)
        ok, buf = cv2.imencode(".jpg", bgr, encode)
        if not ok:
            return
        jpeg = buf.tobytes()
        for p in cam_to_paths[cam]:
            path_caps[p].push(jpeg)

    # Publish qpos for the optional desktop viewer (view_robot.py) — EXCEPT in
    # sim:bridge mode, where the SIMULATION BRIDGE owns physics and publishes
    # the (extended) state memmap itself; creating it here would zero the
    # bridge's live state. The viewer's Ctrl+drag .pert channel is likewise
    # bridge-owned there.
    if _DISPATCH_TARGET == "sim:bridge":
        state, pert = None, None
    else:
        state, pert = _open_state_publisher(args.state_file, scene, model)

    # The pluggable "advance the world" plant: MuJoCoPlant (sim:mujoco, the default Mac
    # path -- steps physics + publishes the qpos memmap) or ROSPlant (real:ros -- mirrors
    # encoders in and publishes the command contract; lazy rclpy import). The controller
    # loop below is identical for both. See server/plant.py.
    plant = make_plant(_DISPATCH_TARGET, model, data, state)
    _PLANT[0] = plant          # the network thread's only handle on it
    # Voice: hand the :9610 hub its plant seam now that the plant exists.
    # The network thread bound the port already — availability is wired HERE
    # because make_plant() runs after that thread starts.
    if getattr(plant, "audio_available", lambda: False)():
        AUDIO.set_speaker_sink(plant.send_speaker_audio)
        # DEMAND-DRIVEN, end to end: the mic subscription only exists while a
        # :9610 client is actually listening. Wiring the sink permanently
        # made the HAL see a listener forever — it held the card open and pumped
        # 50 Hz of PCM through THIS process's GIL for an empty room, dragging
        # proprio from 50 to 45 Hz. This gateway is where the joint stream lives;
        # the map/cloud planes were evicted to perception_server.py for exactly
        # that reason, and voice only earns its place here by costing nothing
        # when idle.
        AUDIO.set_demand_cb(
            lambda on: plant.set_mic_sink(AUDIO.publish_mic if on else None))
    else:
        AUDIO.set_unavailable(
            "no audio device on this rig (the hardware_bridge owns the sound "
            "card; check MABEL_HW_AUDIO and that the USB audio card is plugged in)")

    # Safe per-joint slider ranges to stream to the apps (calibration-derived on the real
    # robot when the plant exposes it, model-derived in sim). Resolved once — limits change
    # only on a recalibration, and the app persists the last map. See _resolve_joint_limits.
    global _JOINT_LIMITS_STREAM
    _JOINT_LIMITS_STREAM = _resolve_joint_limits(plant, model)
    log.info("joint slider limits: %d joints (e.g. left_arm_4=%s rad)",
             len(_JOINT_LIMITS_STREAM), _JOINT_LIMITS_STREAM.get("left_arm_4"))

    # ── Decoupled feedback (real:ros): refresh joints/pose at the STREAM rate ────
    # In real:ros the control loop only samples robot_state at its iteration rate,
    # which under load drops to ~18 Hz — so the 50 Hz app stream resends stale
    # repeats and the robot's joints/pose look laggy on the server + iOS. Mirror the
    # latest /joint_states + /odom into a SEPARATE MjData on a dedicated ~60 Hz
    # thread (mj_forward releases the GIL, ~0.5 ms), fully decoupled from the heavy
    # control law → minimum feedback latency regardless of control-loop load.
    if _DISPATCH_TARGET.startswith("real") and hasattr(plant, "mirror_into"):
        _fb_data = mujoco.MjData(model)
        _fb_hz = float(os.getenv("MABEL_FEEDBACK_HZ", "60"))
        _dyn_real_next = [0.0]           # ~10 Hz robot_state["dynamics"] limiter

        def _feedback_loop():
            period = 1.0 / _fb_hz
            while True:
                t0 = time.time()
                try:
                    plant.mirror_into(_fb_data)
                    _robot_state["joints"] = _read_robot_joints(model, _fb_data)
                    _robot_state["base"] = _read_base_state(model, _fb_data)
                    _robot_state["t"] = time.time()      # sample time → stateAgeMs
                    # Hardware-measured dynamics (vel/effort/temp) — shared by
                    # the :9600 telemetry plane AND robot_state's sparse keys.
                    # plant.joint_dynamics() sources /arms + /hands joint_states
                    # + hand temperature topics; absent values stay None —
                    # honest "not measured", never a guess. Keyed by ROS names;
                    # the wire uses the iOS ids (_TELEMETRY_JOINTS translates).
                    dyn = {}
                    try:
                        dyn = plant.joint_dynamics() or {}
                    except Exception:
                        pass
                    jd = dyn.get("joints", {})
                    _empty: dict = {}

                    # Sparse joint dynamics onto robot_state (~10 Hz): the SAME
                    # jointVelocities/jointTorques/jointTemps keys the SIM path
                    # emits, so the apps' telemetry graphs/bars light up
                    # identically on real hardware with zero app changes. Only
                    # joints the HAL actually measured are included — an
                    # unpowered/absent actuator publishes NOTHING for that
                    # joint (the app renders an honest empty stub, never a
                    # fake zero). motorHealth (the merged */motor_health
                    # topics) fills τ/temp for joints whose joint_states don't
                    # carry them (DaMiao MIT feedback, feetech hand temps).
                    if time.time() - _dyn_real_next[0] >= 0.1:
                        _dyn_real_next[0] = time.time()
                        _vel, _tor, _tmp = {}, {}, {}
                        for k in (_robot_state.get("joints") or {}):
                            # joint_dynamics is keyed by the names ON THE WIRE
                            # (/arms/joint_states etc = URDF snake_case, which IS
                            # the iOS id) — try that first, MuJoCo name second.
                            e = (jd.get(k)
                                 or jd.get(_TELEMETRY_JOINTS.get(k, k))
                                 or _empty)
                            if e.get("dq") is not None:
                                _vel[k] = round(float(e["dq"]), 4)
                            if e.get("tau") is not None:
                                _tor[k] = round(float(e["tau"]), 3)
                            if e.get("temp") is not None:
                                _tmp[k] = round(float(e["temp"]), 1)
                        _hj = (_motor_health_now() or {}).get("joints") or {}
                        for j, h in _hj.items():
                            if h.get("torqueNm") is not None and j not in _tor:
                                _tor[j] = h["torqueNm"]
                            if h.get("tempC") is not None and j not in _tmp:
                                _tmp[j] = h["tempC"]
                        _robot_state["dynamics"] = {"stamp": time.time(),
                                                    "vel": _vel, "torque": _tor,
                                                    "temp": _tmp}

                    # :9600 proprio (real): measured q from the mirror + the
                    # hardware's own vel/effort/temp from the shared read above.
                    if TELEMETRY.enabled():
                        # Same double-keyed lookup as robot_state["dynamics"]
                        # above: URDF/iOS id first, MuJoCo name second.
                        def _jde(k):
                            return (jd.get(k)
                                    or jd.get(_TELEMETRY_JOINTS.get(k, k))
                                    or _empty)
                        joints = {k: {"q": round(v, 4),
                                      "dq": _jde(k).get("dq"),
                                      "tau": _jde(k).get("tau"),
                                      "temp": _jde(k).get("temp")}
                                  for k, v in (_robot_state.get("joints") or {}).items()}
                        TELEMETRY.set_proprio({"stamp": _robot_state.get("t"),
                                               "joints": joints,
                                               "base": _robot_state.get("base", {})})
                        for side in ("left_hand", "right_hand"):
                            ts = dyn.get("torque", {}).get(side)
                            if ts is not None:
                                _torque_state[side] = bool(ts)
                except Exception:
                    pass
                time.sleep(max(0.0, period - (time.time() - t0)))

        threading.Thread(target=_feedback_loop, name="ros-feedback", daemon=True).start()
        log.info("feedback: decoupled joint/pose refresh @ %.0f Hz (real:ros)", _fb_hz)

    # Episode recorder + its sim sensor source (renderer is built lazily on the
    # first Record, on this main thread, so the GL context is valid).
    # Episodes are written in the DATA CURATION STUDIO's input format (one
    # episode_XXXX.h5 per episode under <root>/<folder>/) so headset-recorded
    # demonstrations flow straight into curation + training (curation_recorder.py).
    recorder = CurationRecorder(args.episodes_dir, fps=args.record_fps, model=model)
    pump = [None]                       # boxed RecordPump (threaded capture) for the loop
    _robot_state["recording"] = recorder.status()   # so the app shows counts at once

    # ── Animation demos ────────────────────────────────────────────────────
    # A SECOND recorder, and deliberately not a mode of the one above.
    # CurationRecorder writes the robot's realized state for policy training;
    # this one writes the operator's COMMANDS so they can be pushed back through
    # the same controller. Different payload, different consumer, different file.
    demo_rec = DemoRecorder(fps=args.demo_fps)
    demo_player = [None]                # boxed DemoPlayer, stepped by the loop
    _demo_clock = [time.monotonic()]    # wall clock for the player (see step call)
    # mj joint name → wire joint id, for turning the model's home pose into
    # something the player can command.
    _MJ_TO_IOS = {mj: ios for ios, mj in _IOS_TO_MJ.items()}

    def _episode_clip_name(rec, ctag, clips_dir) -> str:
        """`<folder>_episode_XXXX`, the H5's own index — unless that clip already
        exists. Episode folders are per CLIENT (ios/ vs vision_os/) but clips
        share ONE library, so the iPhone's and the Vision Pro's `pick/episode_0000`
        would otherwise overwrite each other's clip. On a collision the client
        tag joins the folder part (`pick_visionpro_episode_0000`), which the apps
        still group and label as an episode."""
        idx = rec.episode_index()
        name = f"{rec.task}_episode_{idx:04d}"
        if os.path.isdir(os.path.join(clips_dir, _demo_safe_name(name))):
            name = f"{rec.task}_{ctag or 'client'}_episode_{idx:04d}"
        return name

    def _demo_home() -> dict:
        """The MJCF's assembled pose (qpos0), in wire joint ids.

        This is the same `ref` every 3-D client renders against, so "home" here
        means the pose the robot is DRAWN at when every joint reads zero
        displacement — the one an audience recognises as neutral. Built fresh
        rather than cached: a scene switch rebuilds the model.
        """
        out = {}
        for j in range(model.njnt):
            try:
                mj = model.joint(j).name
            except Exception:
                continue
            ios = _MJ_TO_IOS.get(mj)
            if not ios:
                continue
            adr = model.jnt_qposadr[j]
            if 0 <= adr < model.nq:
                out[ios] = float(model.qpos0[adr])
        return out

    def _demo_hardware() -> dict:
        """Which subsystems were actually live for this take.

        Recorded per the operator's ask ("on what hardware was on"). In sim every
        joint the model defines is 'on' by construction, so the honest answer is
        the dispatch target plus what the plant reports — a clip that claims real
        hands when it was captured against MuJoCo is worse than one that says
        nothing.
        """
        hw = {"dispatch": _DISPATCH_TARGET, "sim": not _DISPATCH_TARGET.startswith("real")}
        for key in ("arms", "hands", "neck", "torso", "lift", "base"):
            hw[key] = bool((_torque_state or {}).get(key, hw["sim"]))
        hw["wbc"] = bool(wbc.active)
        return hw

    # Full-bus rosbag (MCAP) data collection — the iOS "Start collecting data"
    # button. Writes one bag per session to a LOCAL, git-ignored recordings dir;
    # convert later with tools/bag_to_rerun.py. Stopped on shutdown so a bag is
    # never left unfinalized.
    bag = BagRecorder()
    _robot_state["dataCollection"] = bag.status()
    atexit.register(bag.stop)

    def _control_snapshot():
        """The operator's current teleop command, logged with every frame."""
        j = _control.get("joystick")
        # Copies, not references: this runs on the control thread at the physics
        # snapshot instant and is written later by the record worker.
        spec = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _control_spec.items()}
        eng_mode = str(getattr(getattr(teleop, "controller", None), "mode", ""))
        return {**spec,
                "estop": bool(_estop["active"]),
                "engaged": spec.get("method") == "retarget" and eng_mode == "engaged",
                "navJoystick": list(j) if j else None,
                "joint_cmd": dict(_joint_cmd)}

    # Position-servo actuators (affine bias with a non-zero −kp term): their ctrl
    # IS a desired joint ANGLE. Velocity actuators (swerve wheels: ctrl is a rate)
    # and plain motors (torque) have no desired joint position, so they're excluded
    # from qpos_des — those joints keep their realized value there.
    _pos_servo_joint = {}
    for _a in range(model.nu):
        if model.actuator_trntype[_a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if (model.actuator_biastype[_a] == mujoco.mjtBias.mjBIAS_AFFINE
                and model.actuator_biasprm[_a, 1] != 0.0):
            _pos_servo_joint[model.joint(model.actuator_trnid[_a, 0]).name] = _a

    def _desired_joints():
        """The controller set-points (DESIRED joint pose), mj_joint_name → rad.

        Two regimes: WBC-managed arm/torso joints are torque-driven, so their
        desired comes from the impedance set-point (`wbc._q_des`); position-servo
        joints (neck, lift, grippers, steer) have their commanded actuator value
        `data.ctrl` as the desired angle. This is the command INTO the controller;
        qpos is the realized pose OUT of it."""
        des = {}
        managed = set(wbc.managed) if wbc.active else set()
        for jname, a in _pos_servo_joint.items():
            if jname not in managed:              # WBC may have neutralised this servo
                des[jname] = float(data.ctrl[a])
        if wbc.active:
            for ios, v in wbc._q_des.items():
                mj = _IOS_TO_MJ.get(ios)
                if mj:
                    des[mj] = float(v)
        return des

    def _pump_telemetry():
        """Build + reference-swap the :9600 proprio/control snapshots (control
        thread; only called while a telemetry client is subscribed). In real:ros
        the decoupled feedback thread owns proprio (freshest mirror); the
        control message is built here in BOTH worlds because this thread owns
        the command state. JSON encoding happens on the network loop."""
        if not _DISPATCH_TARGET.startswith("real"):
            dyn = _read_joint_dynamics(model, data)
            vel, tor = dyn["vel"], dyn["torque"]
            joints = {k: {"q": round(v, 4), "dq": vel.get(k),
                          "tau": tor.get(k), "temp": None}
                      for k, v in (_robot_state.get("joints") or {}).items()}
            TELEMETRY.set_proprio({"stamp": _robot_state.get("t"),
                                   "joints": joints,
                                   "base": _robot_state.get("base", {})})
        TELEMETRY.set_control({
            "mode": dict(_control_spec),
            "desired": {k: round(float(v), 4)
                        for k, v in _desired_joints().items()},
            "baseTwist": {"vx": round(controller.vx, 4),
                          "vy": round(controller.vy, 4),
                          "wz": round(controller.omega, 4)},
            "ctrl": [round(float(c), 4) for c in data.ctrl],
            "engaged": bool(wbc.active),
            "estop": _estop["active"]})

    banner(args)

    # ── Real-time control + render loop ──────────────────────────────────────
    # Round-robin: render one camera per tick so each of N cams hits cam_fps.
    n_cams = max(1, len(render_cams))
    tick_period = 1.0 / (max(args.cam_fps, 1.0) * n_cams)
    last_cam = 0.0
    cam_idx = 0
    last_log = 0.0
    loop_iters = 0        # outer-loop iterations since last status line
    src_t0 = time.time()
    prev = src_t0
    acc = 0.0
    # Hand→robot latency vs catch-up. Each outer iteration grabs ONE fresh hand
    # frame, then the inner loop simulates up to MAX_CATCHUP seconds of physics
    # chasing THAT frame. With MAX_CATCHUP=0.5 (the old value) the bridge would
    # batch ~0.5 s of sim per iteration — and since physics runs ~0.5x real-time,
    # that is ~1 s of wall-clock spent tracking a hand pose that is already 1 s
    # old, so the robot visibly lags your hand. Keeping it small makes the loop
    # come back for a fresh frame every few ms; physics that can't keep up just
    # advances in slight slow-motion instead of building a latency backlog.
    MAX_CATCHUP = float(os.getenv("MABEL_MAX_CATCHUP", "0.05"))  # holds 1.0x real-time down to a 20 Hz outer loop
    # REAL-ROBOT CONTROL RATE. In sim the inner loop is a PHYSICS accumulator: each
    # pass is one mj_step, so chasing model.opt.timestep (4 ms) is exactly right. On
    # the REAL robot there is NO physics to advance — the robot is the integrator —
    # yet the same accumulator still ran read_state() (mj_forward) + the entire
    # IK/WBC stack + a ROS publish 250x/second on this thread. That pinned a core,
    # and because it holds the GIL it STARVED the asyncio network thread: the apps'
    # joints/telemetry crawled at 0.5-1 Hz with ~2 s gaps while VIDEO (a separate
    # process) stayed smooth — the "state is laggy but video is fine" report.
    # The hardware contract only needs the control law at ~50 Hz, so pace it there
    # and give the rest of the process its CPU (and the GIL) back.
    CTRL_HZ = float(os.getenv("MABEL_CONTROL_HZ", "50"))

    # ── LIFT BRING-UP HOLD (real robot) ──────────────────────────────────────
    # The tendon `Lift` actuator is EXCLUDED from init_hold's rest-pose set (it is
    # not a joint servo), boots ENERGIZED on the Pico, and has no torque-enable edge
    # to trigger _resume_hold — so nothing seeds its command. It would default to
    # ctrl=0, and the loop below streams that verbatim as /lift/command=0, driving
    # the (worm-gear, non-back-drivable) column straight to the floor the instant the
    # first command goes out. Seed the command to the MEASURED height BEFORE any lift
    # command goes out, so the lift HOLDS where it is. Two guarantees make this
    # rigorous (the old one-shot 5 s wait gave up and streamed the unseeded 0 anyway
    # — the "lift dives on real_run start" bug):
    #   1. The plant's lift command channel starts GATED: until the seed succeeds it
    #      streams lift_pos=NaN — the contract's "no lift command" sentinel, dropped
    #      by swerve_node and ignored by the HAL — so an unseeded ctrl can NEVER
    #      reach the column, however late the Pico shows up.
    #   2. The seed is retried every control tick (cheap dict probe), so feedback
    #      that appears minutes later (Pico hot-join, HAL restart, robot USB rail
    #      powered after the gateway) still seeds-then-opens, and the first real
    #      command is always "hold where you are". Sim needs none of this (qpos0 ==
    #      the assembled pose); the jog /lift/velocity path is not gated.
    _lift_seed = {"ok": not _DISPATCH_TARGET.startswith("real")}

    def _try_seed_lift():
        """Seed Lift ctrl from measured feedback; open the command gate on success."""
        _js0 = _robot_state.get("joints") or {}
        if "lift_lower" not in _js0:
            return False
        # The joint mirror never forgets: after a lift disconnect, lift_lower keeps
        # its last value forever. Only a FRESH stream is a valid seed source.
        if hasattr(plant, "lift_feedback_age") and plant.lift_feedback_age() > 1.0:
            return False
        _lift_q = float(_js0.get("lift_lower", 0.0)) + float(_js0.get("lift_upper", 0.0))
        try:
            _laid = model.actuator("Lift").id
            _llo, _lhi = model.actuator_ctrlrange[_laid]
            data.actuator("Lift").ctrl[0] = float(min(max(_lift_q, _llo), _lhi))
        except Exception as e:                                      # noqa: BLE001
            log.warning("lift bring-up seed failed (%s) — lift commands stay gated.", e)
            return False
        _lift_seed["ok"] = True
        if hasattr(plant, "set_lift_command_enabled"):
            plant.set_lift_command_enabled(True)
        log.info("lift: seeded bring-up command to measured %.3f m "
                 "(holds in place — command gate OPEN).", _lift_q)
        return True

    if _DISPATCH_TARGET.startswith("real"):
        if hasattr(plant, "set_lift_command_enabled"):
            plant.set_lift_command_enabled(False)     # gate CLOSED until seeded
        _seed_dl = time.time() + float(os.getenv("MABEL_LIFT_SEED_WAIT", "5.0"))
        while time.time() < _seed_dl and not _try_seed_lift():
            time.sleep(0.1)
        if not _lift_seed["ok"]:
            log.warning("lift: no feedback yet — position commands GATED (NaN on the "
                        "wire); will keep retrying the measured-height seed in-loop.")

    # OperatorPose parse cache: the loop spins faster than the VP's 30 Hz frames,
    # so the same slot payload was re-parsed (~27 np reshapes + 48 matmuls) on
    # every extra pass. Key on (client, recv_time) — behavior-identical.
    _pose_cache = {"key": None, "pose": None}

    try:
        while True:
            loop_iters += 1     # one fresh hand frame is sampled per iteration

            # 0.0) episode record select / start / stop (queued by the net thread).
            # Capture runs on a dedicated thread (RecordPump) so rendering every
            # camera never stalls this control loop; the loop only triggers it.
            while _record_pending:
                action, task, ctag = _record_pending.pop(0)
                if action == "start":
                    recorder.set_source_client(ctag)
                    if pump[0] is None or not pump[0].active:
                        p = RecordPump(model, recorder,
                                       get_control=_control_snapshot, get_desired=_desired_joints,
                                       chassis_body=C.CHASSIS_BODY, fps=args.record_fps)
                        err = p.start(task, scene)
                        if err is not None:
                            log.error("recorder disabled — could not start capture: %s", err)
                            pump[0] = None
                        else:
                            pump[0] = p
                            # ONE take, two files: the same Record also captures
                            # the operator's COMMANDS as a replayable clip named
                            # after the episode (<folder>_episode_XXXX, the H5's
                            # own index), so every recorded episode can be
                            # replayed from the apps. The apps no longer have a
                            # separate "demo clip" recorder (operator, 2026-10-14).
                            _demo_pending.append(("arm", {
                                "name": _episode_clip_name(recorder, ctag, demo_rec.clips_dir),
                                "countdown": 0.0,
                                "notes": f"policy episode {recorder.task}/"
                                         f"episode_{recorder.episode_index():04d}.h5",
                                "client": ctag, "recordedBy": ctag,
                            }))
                elif action == "stop":
                    if pump[0] is not None:
                        # CurationRecorder writes curation-format .h5 directly —
                        # stop() queues the flush on its background worker.
                        pump[0].stop()
                        pump[0] = None
                    if demo_rec.active or demo_rec.armed:
                        _demo_pending.append(("stop", {}))   # the episode's clip
                elif action == "select":
                    recorder.set_source_client(ctag)
                    recorder.set_task(task)
                _robot_state["recording"] = recorder.status()

            # 0.05) data-collection (rosbag/MCAP) start/stop. Owns a `ros2 bag
            # record` subprocess; cheap to drive from here (just signals).
            while _data_collect_pending:
                active, label = _data_collect_pending.pop(0)
                if active:
                    err = bag.start(label)
                    if err:
                        log.error("data-collection could not start: %s", err)
                else:
                    bag.stop()
            _robot_state["dataCollection"] = bag.status()

            # 0.06) animation demos — record the operator's COMMANDS, and replay
            # them back through the same controller.
            #
            # Ordering inside this block is load-bearing:
            #   1. drain requests
            #   2. STEP THE PLAYER, writing into `_joint_cmd`
            #   3. sample the recorder
            # The player writes commands before the recorder reads them, so
            # re-recording a replay captures the replay (useful, and honest).
            # Both run BEFORE `teleop.update()` consumes `_joint_cmd` further
            # down, so a replayed frame is seen by the controller on the very
            # tick it was produced rather than one tick late.
            while _demo_pending:
                action, p = _demo_pending.pop(0)
                if action == "arm":
                    if demo_player[0] is not None:
                        log.info("demo: refusing to record while a replay is running")
                    else:
                        demo_rec.arm(p["name"], countdown=p["countdown"],
                                     notes=p["notes"], client=p["client"],
                                     recorded_by=p["recordedBy"])
                        log.info("demo: armed '%s' (%.1fs countdown)",
                                 p["name"], p["countdown"])
                elif action == "stop":
                    if demo_rec.active or demo_rec.armed:
                        clip = demo_rec.finish(
                            model=model, dispatch=_DISPATCH_TARGET,
                            scene=(scene or {}).get("id", "") if isinstance(scene, dict) else str(scene or ""),
                            model_path=getattr(args, "model", "") or "",
                            control_spec=dict(_control_spec),
                            wbc_active=bool(wbc.active),
                            wbc_managed=list(wbc.managed) if wbc.active else [],
                            hardware=_demo_hardware(), home=_demo_home())
                        if clip is None:
                            log.info("demo: stopped with no frames — nothing saved")
                        else:
                            log.info("demo: saved '%s' (%.2fs, %d frames, %d joints)",
                                     clip.manifest.name, clip.manifest.duration_s,
                                     clip.manifest.n_frames, len(clip.cmd_names))
                        _demo_state["clips"] = _list_demo_clips()
                elif action == "play":
                    if demo_rec.active or demo_rec.armed:
                        log.info("demo: refusing to replay while recording")
                    else:
                        try:
                            clip = _load_demo_clip(p["name"])
                        except Exception as e:                       # noqa: BLE001
                            log.warning("demo: cannot load '%s': %s", p["name"], e)
                            clip = None
                        if clip is not None:
                            # The ease-in starts from the CURRENT commanded pose,
                            # not the measured one: the controller is holding to a
                            # set-point, and ramping from the sagged/lagging
                            # measurement would command a step back to it first.
                            start = dict(_joint_cmd)
                            for mj, v in _desired_joints().items():
                                ios = _MJ_TO_IOS.get(mj)
                                if ios and ios not in start:
                                    start[ios] = v
                            demo_player[0] = DemoPlayer(
                                clip, repeats=p["repeats"], speed=p["speed"],
                                start_pose=start, home=_demo_home())
                            _demo_clock[0] = time.monotonic()   # zero the wall clock
                            log.info("demo: playing '%s' ×%d (%.2fs each)",
                                     clip.manifest.name, p["repeats"],
                                     clip.manifest.duration_s)
                elif action == "halt":
                    if demo_player[0] is not None:
                        demo_player[0].stop()      # winds down through return-home
                elif action == "delete":
                    if delete_clip(p["name"]):
                        log.info("demo: deleted '%s'", p["name"])
                    _demo_state["clips"] = _list_demo_clips()

            if demo_player[0] is not None:
                pl = demo_player[0]
                # An e-stop must not be ridden out by a ramp. abort() emits
                # nothing further; the controller's own fail-safe takes the robot.
                if _estop["active"]:
                    pl.abort()
                # WALL time, not the physics accumulator. A demo is a performance:
                # it has to take the number of seconds it took to record no matter
                # what the sim's real-time factor is doing that second.
                _now_demo = time.monotonic()
                _dt_demo = min(0.25, max(0.0, _now_demo - _demo_clock[0]))
                _demo_clock[0] = _now_demo
                cmd = pl.step(_dt_demo)
                if cmd:
                    # THE seam: the same dict `MabelTeleop.joint_targets` reads.
                    # Every gate, rate limit and WBC decision applies to a
                    # replayed frame exactly as to a human one.
                    _joint_cmd.update(cmd)
                _demo_state.update(pl.status())
                if pl.done:
                    log.info("demo: '%s' finished", pl.clip.manifest.name)
                    demo_player[0] = None
                    _demo_state.update(playing=False, phase=None, frac=0.0)
            elif _demo_state.get("playing"):
                _demo_state.update(playing=False, phase=None, frac=0.0)

            if demo_rec.active or demo_rec.armed:
                demo_rec.sample(_joint_cmd, _desired_joints(),
                                twist=_control.get("joystick"))
            _rs = demo_rec.status()
            if demo_player[0] is not None and not (demo_rec.active or demo_rec.armed):
                _rs.pop("name", None)      # a replay's `name` is the CLIP playing
            _demo_state.update(_rs)
            _animation_state.update(_demo_state)

            # 0) sim reset (headset Reset button / ENTER+R): home the robot. The
            # bridge owns the physics reset; the facade re-homes the control state
            # (hold pose, drop accumulated joint commands / pushes, re-anchor) so
            # nothing springs back to a pre-reset pose.
            if _reset.is_set():
                _reset.clear()
                # The PLANT owns the world reset: MuJoCoPlant homes its in-process
                # physics; ROSPlant relays to the sim node over /mabel/reset_sim (the
                # sim node owns the real physics). A bare local mj_reset here would only
                # touch the kinematic mirror in ROS mode, leaving the actual sim running.
                plant.reset()
                pose, _reset_pose[0] = _reset_pose[0], None
                if pose and hasattr(plant, "place_base"):
                    # relocate-reset: home the joints, then teleport the base to
                    # the (pre-validated) open pose BEFORE re-anchoring control.
                    try:
                        plant.place_base(float(pose.get("x", 0.0)),
                                         float(pose.get("y", 0.0)),
                                         float(pose.get("yaw", 0.0)))
                        log.info("reset relocated base → (%.2f, %.2f, yaw %.2f)",
                                 float(pose.get("x", 0.0)),
                                 float(pose.get("y", 0.0)),
                                 float(pose.get("yaw", 0.0)))
                    except Exception as e:
                        log.warning("relocate-reset ignored: %s", e)
                teleop.reset()          # re-homes the engine + clears _joint_cmd/_ext_forces in place
                controller.init_hold()  # rest targets back onto the position servos
                acc = 0.0
                log.info("simulation reset to home — re-anchoring on the next frame")

            # 0.5) hold-current request (e.g. switching into compliance): snap the
            # actuator targets to the live pose so nothing lurches on the change.
            if _hold_request.is_set():
                _hold_request.clear()
                teleop.controller.hold_current()

            # 0.55) the effective control spec = the ACTIVE driver's spec, pushed to
            # the facade (it (re)enters/exits the torque engine on change, on this
            # thread). Following ONE client means a second idle app can't reset the
            # operator's mode. _control_spec is kept only for the monitor/status line.
            active_id = _active_client()
            # PER-CLIENT spec: follow the ACTIVE driver's own spec (its method/type/
            # region), defaulting to navigation if that client hasn't set one. A client
            # that is connected but idle still wins via the frames[0] fallback in
            # _active_client(), so a single operator's SOFT mode persists while idle
            # WITHOUT pinning OTHER clients: the web/Vision Pro each drive with THEIR own
            # spec the moment they command. (An earlier "global sticky spec" made every
            # client inherit the last spec set by anyone — so the web, sending no
            # control_mode, got stuck in iOS's compliance and its commands were ignored.)
            if active_id is not None:
                # The ACTIVE driver's own spec (navigation default if it sent none — so
                # the web, which sends no control_mode, drives normally and is never
                # stuck in another client's mode).
                _control_spec.update(_client_specs.get(active_id, _DEFAULT_SPEC))
                # BASE-DRIVE PRIORITY: while the operator is actively pushing the nav
                # stick, FORCE navigation so the base ALWAYS responds. Otherwise an arm/
                # impedance control_mode (e.g. "joint·impedance·arm") routes through the
                # WBC path, which never applies the stick to the base — driving freezes
                # mid-move ("works 2s then stops"). "If teleop is driving, only listen to
                # teleop." (Harmless on a base-only robot; arms resume on stick release.)
                _af = _client_frames.get(active_id)
                _nj = (_af[1].get("navJoystick") if _af and _af[1] else None) or {}
                # ry (the LIFT) counts too: the apps' drive pad is now on screen in
                # Joint and Task space (2026-10-27), and the lift rate is serviced only
                # by the navigation path, so a lift push there went nowhere.
                if max(abs(_nj.get("lx", 0.0)), abs(_nj.get("ly", 0.0)),
                       abs(_nj.get("rx", 0.0)), abs(_nj.get("ry", 0.0))) > 0.06:
                    _control_spec["method"] = "navigation"
            # else: NOBODY is actively driving → KEEP the current spec (sticky). This is
            # what lets a SOFT/compliance session stay gravity-compensated while the
            # operator drags the PHYSICAL robot (sending no app input → no driver token);
            # the robot would otherwise stiffen back to navigation the moment they let go.
            # One backend: the facade routes per method internally (navigation/retarget
            # → the engine; "joint"/"wrist" → its WBC torque law, entering/exiting the
            # torque engine lurch-free on a spec change).
            teleop.set_control_spec(**_control_spec)

            # 1) newest operator input → OperatorPose + navigation joystick from the
            #    ACTIVE client. The facade owns the wire dialect (from_teleop_frame)
            #    and the stick sign convention (app_joystick) — see mabel/wire.py.
            if args.mock:
                pose = OperatorPose.from_avframe(mock_avp.mock_frame(time.time() - src_t0))
            else:
                entry = _client_frames.get(active_id) if active_id is not None else None
                fp = entry[1] if entry else None          # raw teleop_frame payload dict
                if fp is None:
                    pose = None
                else:
                    _pk = (active_id, entry[0])           # same client, same recv time
                    if _pose_cache["key"] != _pk:
                        _pose_cache["pose"] = OperatorPose.from_teleop_frame(fp)
                        _pose_cache["key"] = _pk
                    pose = _pose_cache["pose"]
                    _control["joystick"] = app_joystick(fp.get("navJoystick"))
            joystick = _control["joystick"]

            # ── Nav2 autonomous-with-takeover ───────────────────────────────────
            # Drain queued web goals → forward to Nav2 on /goal_pose (control thread
            # owns `plant`). Then, while a goal is active and Nav2 is freshly
            # commanding, drive the base from Nav2's twist fed through the SAME
            # navigation joystick path (NavFn A* + MPPI, not any local controller).
            # Any human stick deflection cancels the goal → teleop takes over.
            while _nav_goal_pending:
                g = _nav_goal_pending.pop(0)
                if g is None:
                    _nav["active"] = False
                    if hasattr(plant, "clear_goal_marker"):
                        try:
                            plant.clear_goal_marker()      # drop the green ball in RViz
                        except Exception:
                            pass
                elif hasattr(plant, "send_nav_goal"):
                    try:
                        plant.send_nav_goal(*g); _nav["active"] = True
                    except Exception:
                        _nav["active"] = False
            # Staged-goal previews → /goal_pose_preview (RViz only, no drive).
            while _nav_preview_pending:
                g = _nav_preview_pending.pop(0)
                if hasattr(plant, "send_nav_goal_preview"):
                    try:
                        plant.send_nav_goal_preview(*g)
                    except Exception:
                        pass
            if _NAV_AUTO and hasattr(plant, "send_nav_goal"):
                _nav["active"] = True       # follow Nav2 autonomously (no app goal)
            if _nav["active"] and hasattr(plant, "nav_twist"):
                human = joystick is not None and max(abs(joystick[0]), abs(joystick[1]),
                                                     abs(joystick[2])) > 0.06
                if human:
                    _nav["active"] = False                 # operator takeover
                else:
                    vx, vy, w, age = plant.nav_twist()
                    if age < 0.5:                          # Nav2 actively commanding
                        ml = getattr(C, "MAX_LIN_SPEED", 0.8) or 0.8
                        ma = getattr(C, "MAX_ANG_SPEED", 2.0) or 2.0
                        cl = lambda v: max(-1.0, min(1.0, v))
                        # _navigate: tvx=ly·ML, tvy=lx·ML, tw=rx·MA → invert to (lx,ly,rx,ry)
                        joystick = (cl(vy / ml), cl(vx / ml), cl(w / ma), 0.0)
                        _control_spec["method"] = "navigation"
                        teleop.set_control_spec(**_control_spec)

            # 2) explicit clutch re-anchor (headset Reset/recenter or ENTER). The
            #    first-frame auto-anchor is handled inside the facade.
            if _recalibrate.is_set():
                _recalibrate.clear()
                teleop.recalibrate()

            # 2.5) body-calibration start/cancel (queued by the net thread). The
            #      facade's update() then drives the 3-pose capture from the operator's
            #      pose stream, holds the robot still, and applies+saves the profile on
            #      completion. We just kick it off / cancel it here.
            while _body_cal_pending:
                action, user = _body_cal_pending.pop(0)
                if action == "start":
                    controller.start_calibration(user)   # engine 1-pose: extend arms forward
                elif action == "cancel":
                    controller.cancel_calibration()
                elif action == "apply":
                    controller.load_profile(user)        # re-apply a saved reach scale

            # 3) advance physics by real elapsed time; ONE facade.update() per substep
            #    runs the WHOLE control law (clutch, retarget + whole-body IK OR the
            #    torque engine, fail-safe, direct joint override). The bridge owns the
            #    accumulator + mj_step so the sim stays real-time even when the camera
            #    render below blocks this thread for tens of ms.
            # Mode-dependent timestep: DRIVE (navigation) runs coarse so the fixed
            # ~8.4 ms step cost stays real-time (~1.2x) — base/lift/neck physics are
            # fine that coarse; MANIPULATION runs finer for grasp accuracy. Full
            # physics either way (nothing dropped). Changing opt.timestep mid-run is
            # cheap and safe (read fresh each step).
            model.opt.timestep = TS_DRIVE if teleop.spec.get("method") == "navigation" else TS_MANIP
            now = time.time()
            acc += min(now - prev, MAX_CATCHUP)
            prev = now
            _stepped = False         # did physics advance (and refresh state) this iteration?
            # sim: one pass == one mj_step (physics timestep). real: no physics —
            # pace the control law at CTRL_HZ instead of the 250 Hz physics rate.
            _ctrl_dt = (max(model.opt.timestep, 1.0 / CTRL_HZ)
                        if _DISPATCH_TARGET.startswith("real") else model.opt.timestep)
            if _DISPATCH_TARGET.startswith("real"):
                # REAL: never burn catch-up substeps. The accumulator exists for
                # PHYSICS (sim must integrate every timestep of real time); the
                # real robot integrates itself, and the operator pose is read once
                # per OUTER pass — so catch-up substeps re-run the whole engine on
                # the SAME stale frame. On thor that burst (~3 substeps/pass) was
                # the arm jitter: measured 13-17 Hz outer / 39 Hz /arms/command in
                # bursts with 105 ms gaps, engine 3x over budget. One tick per
                # pass with the FRESHEST state turns the same compute into an
                # even ~30-45 Hz command stream (2026-07-21).
                acc = min(acc, _ctrl_dt)
            while acc >= _ctrl_dt:
                dt = _ctrl_dt
                # ros: mirror measured encoders/odom into the model (no physics);
                # sim: no-op (physics owns the state). While the KINEMATIC stack
                # (retarget/navigation) is active the mirror needs geometry only —
                # skip the full mj_forward (16% of gateway CPU, the "50 Hz
                # gap"); the WBC torque modes get real dynamics back automatically.
                _set_kin = getattr(plant, "set_mirror_kinematics_only", None)
                if _set_kin is not None:
                    _set_kin(not wbc.active)
                plant.read_state()
                # Compose the external wrench ONCE per substep, BEFORE the control law:
                # the desktop viewer's Ctrl+drag (pert memmap) is the base; the facade's
                # WBC then ADDS any app external_force push on top (apply_external). Doing
                # it here — not after teleop.update — is what lets the whole-body
                # compliance base-wrench estimate (read inside the WBC step) see the SAME
                # forces mj_step will apply, so the chassis glides under a viewer drag.
                if not _DISPATCH_TARGET.startswith("real"):
                    data.xfrc_applied[:] = pert if pert is not None else 0.0
                robot_cmd = teleop.update(pose, joystick, dt)   # the ONE backend (facade routes internally)
                # Refresh the operator-overlay record for opted-in web clients
                # (the engine stamps it with the VP frame ts; senders dedupe).
                _op_pts = getattr(teleop.controller, "last_operator_points", None)
                if _op_pts is not None:
                    _op_overlay_rec["rec"] = _op_pts
                if now - _DBG["t"] > 0.5:                       # TEMP drive-path debug
                    _DBG["t"] = now
                    _afd = _client_frames.get(active_id) if active_id is not None else None
                    _njd = (_afd[1].get("navJoystick") if _afd and _afd[1] else None)
                    log.info("DRIVEDBG act=%s pose=%s nclients=%s nav=%s joy=%s base=%s method=%s",
                             active_id, pose is not None, _clients.get("n"), _njd, joystick,
                             getattr(robot_cmd, "base", None), _control_spec.get("method"))
                # E-STOP: hard-stop the base server-side. engaged=0.0 -> /mabel_cmd
                # hardware_state=1 -> swerve_node emits a ZERO Twist + estop=True to the
                # HAL. The navigation path otherwise returns engaged=1.0, so without this
                # an operator e-stop would not stop the base through the command path.
                if _estop["active"] and robot_cmd is not None:
                    robot_cmd.engaged = 0.0
                # LIFT COMMAND GATE MAINTENANCE. Open: feedback can appear long
                # after bring-up (Pico hot-join / robot rail powered late) — until
                # it does the plant streams lift_pos=NaN, and the first successful
                # seed holds-in-place and opens the gate. Close: a DISCONNECTED
                # lift means the desired height must not change — the gate closes
                # (NaN on the wire) so slider/supervisor churn while the lift is
                # away commands nothing, and the return re-SEEDS ctrl from the
                # fresh measured height, discarding any phantom desired that
                # accumulated (the slider snaps back to reality).
                if _DISPATCH_TARGET.startswith("real"):
                    if not _lift_seed["ok"]:
                        _try_seed_lift()
                    elif (hasattr(plant, "lift_feedback_age")
                          and plant.lift_feedback_age() > 3.0):
                        _lift_seed["ok"] = False
                        if hasattr(plant, "set_lift_command_enabled"):
                            plant.set_lift_command_enabled(False)
                        log.warning("lift: feedback lost — command gate CLOSED; "
                                    "desired height frozen until the lift returns "
                                    "(will re-seed from measured).")
                # Advance the world: sim steps MuJoCo (+ desktop Ctrl+drag perturbation) and
                # publishes the qpos memmap the camera server reads; ros publishes the
                # controller's RobotCommand on the canonical hardware contract. Either way
                # the loop self-paces on the real-time accumulator below.
                # The app's stiffness slider + control type ride along so the real
                # plant can ship mode-appropriate PD gains to the HAL (/arms/gains):
                # STIFF = the arm.gains set; SOFT = the slider-scaled soft set.
                plant.spec_type = teleop.spec.get("type")
                plant.spec_stiffness = float(teleop.spec.get("stiffness") or 0.0)
                plant.advance(robot_cmd, pert, teleop.wbc.active)
                # LIFT JOG: the drive stick's up/down (right-stick Y = joystick[3]) rides
                # the firmware's VELOCITY mode on the real robot — move-while-held, stop +
                # release-torque on centre — instead of ramping a position setpoint that
                # floods the Pico's position loop. We publish the stick velocity (m/s, +up)
                # on /lift/velocity every tick; the HAL dedups + owns the torque, and 0.0
                # (centred stick or e-stop) means "stop and hold on the worm gear". Sim has
                # no such method (hasattr False) → it keeps the position-integration path.
                if hasattr(plant, "send_lift_velocity"):
                    _vjog = 0.0
                    if (not _estop["active"] and joystick is not None
                            and abs(joystick[3]) > 0.06):        # deadzone (matches takeover)
                        _vjog = float(joystick[3]) * getattr(C, "NAV_MAX_LIFT", 0.06)
                    try:
                        plant.send_lift_velocity(_vjog)
                    except Exception:
                        pass
                # Apply a pending e-stop->hand-torque edge (set by the ESTOP handler).
                # Done here because the control loop owns `plant`; real -> publishes
                # /<side>_hand/torque_enable, sim -> no-op. Edge-only, so the hands never
                # auto-energise on bring-up.
                if _hand_torque_req["pending"]:
                    _hand_torque_req["pending"] = False
                    _hand_torque_req["intent"] = _hand_torque_req["target"]
                    # Was this exact target already applied? Re-adopting the measured
                    # pose on an ALREADY-ENERGISED robot is a ratchet (see the e-stop
                    # handler) — the "measured" pose is then the servo's own steady-
                    # state error, not an operator/gravity displacement. So hold-at-
                    # measured runs ONLY on a genuine LIMP -> ENERGISED transition.
                    _already = (_hand_torque_req.get("applied") is
                                _hand_torque_req["target"])
                    _hand_torque_req["applied"] = _hand_torque_req["target"]
                    try:
                        if _hand_torque_req["target"] and not _already:
                            _resume_hold(plant, teleop)   # measured pose := commanded
                        plant.set_hand_torque(_hand_torque_req["target"])
                        # ARMS follow the SAME e-stop edge as the hands: teleop-start /
                        # e-stop release energises them so the operator can drive the
                        # arms (the bridge hold-then-enables → no lurch); engage frees
                        # them. Without this the arms stayed LIMP on start-teleop and
                        # ignored /arms/command — the "arm control does nothing" bug.
                        _set_arms = getattr(plant, "set_arms_torque", None)
                        if _set_arms is not None:
                            _set_arms(_hand_torque_req["target"])
                            _torque_state["arms"] = _hand_torque_req["target"]
                        # NECK follows the SAME e-stop edge as the arms + hands:
                        # teleop-start / e-stop release energises it so the operator
                        # can drive the neck (the _resume_hold above already reseeded
                        # the measured pose → no lurch); engage limps it. Without this
                        # the neck stayed LIMP forever and ignored /body/command AND
                        # no e-stop ever released its torque — the two reported bugs.
                        _set_neck = getattr(plant, "set_neck_torque", None)
                        if _set_neck is not None:
                            _set_neck(_hand_torque_req["target"])
                            _torque_state["neck"] = _hand_torque_req["target"]
                        # Echo the commanded state so every client's torque
                        # indicator agrees (server-authoritative, like estop).
                        _torque_state["left_hand"] = _hand_torque_req["target"]
                        _torque_state["right_hand"] = _hand_torque_req["target"]
                    except Exception:
                        pass
                # Torque-intent reconcile (real:ros only — sim plants have no
                # hand_torque_state). If an e-stop edge has ever set an intent,
                # re-assert it whenever a hand's latched torque_state disagrees:
                # a hardware-bridge restart boots the hands LIMP and the edge
                # alone never fires again. Intent None (no edge yet) reconciles
                # nothing, so bring-up still never auto-energises.
                _ti = _hand_torque_req["intent"]
                if _ti is not None and time.time() - _hand_torque_req["t_reconcile"] > 2.0:
                    _hand_torque_req["t_reconcile"] = time.time()
                    _mok = _modules_ok()          # absent modules: nothing to reconcile
                    _get_tq = getattr(plant, "hand_torque_state", None)
                    if _get_tq is not None:
                        try:
                            _stale = [(_s, _st) for _s, _st in _get_tq().items()
                                      if _st is not None and bool(_st) != _ti
                                      and _torque_module_present(f"{_s}_hand", _mok)]
                            # Re-energising here too must NOT snap: the hand was limp
                            # (bridge restart / hot-plug) and may have been moved.
                            # ⚠ ONCE per limp episode — see the arms note below.
                            if _stale and _ti and not _reconcile_resumed["hands"]:
                                _reconcile_resumed["hands"] = True
                                _resume_hold(plant, teleop)
                            if not _stale:
                                _reconcile_resumed["hands"] = False
                            for _s, _st in _stale:
                                plant.set_hand_torque(_ti, side=_s)
                                _torque_state[f"{_s}_hand"] = _ti
                        except Exception:
                            pass
                    # ARMS reconcile: same intent, across a bridge restart (arms boot
                    # limp). Re-assert torque if the latched /arms/torque_state
                    # disagrees with the operator's intent.
                    _get_at = getattr(plant, "arms_torque_state", None)
                    _set_at = getattr(plant, "set_arms_torque", None)
                    if _get_at is not None and _set_at is not None:
                        try:
                            _ast = _get_at()
                            if (_ast is not None and bool(_ast) != _ti
                                    and _torque_module_present("arms", _mok)):
                                # ⚠ RESUME ONCE PER LIMP EPISODE — NOT every 2 s.
                                # This poll repeats for as long as the bridge's latched
                                # /arms/torque_state disagrees with the intent (a bridge
                                # that refuses to energise, a wedged latch, a board that
                                # keeps resetting). _resume_hold adopts the MEASURED pose
                                # as the command, so running it per poll re-baselines the
                                # target onto the servo's own steady-state error every
                                # 2 s — the arm STEPS AWAY from the operator's slider,
                                # worst on the gravity-loaded joints (J1/J2/J4, biggest
                                # e = tau_g/kp). That is the ratchet the e-stop handler
                                # already guards against with `_already`; this path is
                                # the same bug reached by a different trigger (reported
                                # 2026-07-16: "J1 J2 J4 step away every 2 seconds").
                                # Resume only on a genuine LIMP -> ENERGISED transition;
                                # re-assert torque as often as we like (that is idempotent).
                                if _ti and not _reconcile_resumed["arms"]:
                                    _reconcile_resumed["arms"] = True
                                    _resume_hold(plant, teleop)
                                _set_at(_ti)
                                _torque_state["arms"] = _ti
                            elif _ast is not None:
                                # agreed again: a future limp episode may resume once more
                                _reconcile_resumed["arms"] = False
                        except Exception:
                            pass
                    # NECK reconcile: same intent, across a bridge restart (the neck
                    # boots torque-OFF, so the e-stop edge alone never re-energises it).
                    # Re-assert torque if the latched /body/torque_state disagrees with
                    # intent. RESUME ONCE PER LIMP EPISODE — never per poll (same ratchet
                    # guard as the arms; re-adopting measured every 2 s would walk the
                    # target onto the servo's steady-state error).
                    _get_nt = getattr(plant, "neck_torque_state", None)
                    _set_nt = getattr(plant, "set_neck_torque", None)
                    if _get_nt is not None and _set_nt is not None:
                        try:
                            _nst = _get_nt()
                            if (_nst is not None and bool(_nst) != _ti
                                    and _torque_module_present("neck", _mok)):
                                if _ti and not _reconcile_resumed["neck"]:
                                    _reconcile_resumed["neck"] = True
                                    _resume_hold(plant, teleop)
                                _set_nt(_ti)
                                _torque_state["neck"] = _ti
                            elif _nst is not None:
                                _reconcile_resumed["neck"] = False
                        except Exception:
                            pass
                acc -= _ctrl_dt              # sim: the physics step; real: 1/CTRL_HZ
                # High-rate feedback: refresh the app-facing joint/pose state EVERY
                # physics substep (cheap reads, same thread as mj_step → no race),
                # not just once per outer iteration. mj_step pacing makes the outer
                # loop run at ~10 Hz, which made joints/pose reach the server + iOS
                # choppy; the 50 Hz state stream now always has fresh data. (real:ros
                # has its own decoupled feedback thread, so skip it here.)
                if not _DISPATCH_TARGET.startswith("real"):
                    _robot_state["joints"] = _read_robot_joints(model, data)
                    _robot_state["base"] = _read_base_state(model, data)
                    _robot_state["t"] = time.time()      # sample time → stateAgeMs
                    _stepped = True
                    # :9600 telemetry producer, SUBSTEP-rate gated to ~50 Hz. It
                    # must live here (not only per outer iteration): under load
                    # the outer loop drops to ~35 Hz while substeps keep pace —
                    # exactly why _robot_state refreshes here too. Gated on a
                    # subscriber, so an unwatched robot pays nothing.
                    if TELEMETRY.enabled() and (_tn := time.time()) - _telem_next[0] >= 0.02:
                        _telem_next[0] = _tn
                        try:
                            _pump_telemetry()
                        except Exception:
                            pass

            # 3.4) REAL ROBOT: yield until the next control tick.
            # SIM MUST NOT DO THIS — there the loop is a physics engine and any idle
            # time is real-time it will have to catch up. On the REAL robot the robot
            # integrates itself, so once this tick's command is out there is genuinely
            # NOTHING to do until the next one. Without the sleep the outer loop
            # busy-spun, holding the GIL and starving the asyncio network thread that
            # ships robot_state/:9600 to the apps (measured: joints at ~0.8 Hz with 2 s
            # gaps while video, in its own process, was smooth). Sleeping here hands
            # the interpreter to the feedback + network threads, which is precisely
            # what makes the app-side joint stream real-time.
            if _DISPATCH_TARGET.startswith("real"):
                # Idle = what's left of THIS tick after compute, measured on the
                # wall clock from the pass start. The old `_ctrl_dt - acc` slept
                # a FULL tick whenever the accumulator drained (acc==0), i.e. it
                # stacked the sleep ON TOP of compute: 25 ms compute + 20 ms
                # sleep = 22 Hz ceiling no matter how fast the engine got
                # (measured on thor, 2026-07-21). When compute overruns the tick
                # there is nothing left to sleep and the loop runs back-to-back.
                _idle = _ctrl_dt - (time.time() - now)
                if _idle > 0:
                    time.sleep(min(_idle, _ctrl_dt))

            # 3.5) ROS state shim (sim:mujoco+ros only; no-op on the Mac sim path and the
            # real robot): publish /clock + /joint_states + /odom_ground_truth (+ IMU) from
            # the physics this bridge OWNS, so the sim sensor renderer / SLAM / Nav2 / RViz
            # compose on top exactly as they did over mujoco_node — same topics, same
            # convention — while the controller + motion model stay byte-identical to the Mac.
            plant.publish_ros_state()

            # 3.55) record a demonstration frame (sim-time throttled). Just hands a
            # cheap state snapshot to the capture thread; rendering happens there so
            # this control loop keeps running at full speed during recording.
            if pump[0] is not None and pump[0].active:
                pump[0].maybe_capture(data.time, data)

            # 3.6) publish the live joint state so connected apps render the ACTUAL
            # robot pose (not their command echo). In sim:mujoco the control loop is
            # the only state source. The inner loop already refreshed it on EVERY
            # substep, so only publish here as a FALLBACK when no substep ran this
            # iteration (acc < timestep) — re-reading the unchanged data otherwise was
            # pure duplicate work on the hot control thread. In real:ros the dedicated
            # ~60 Hz feedback thread owns joints/base, so skip it here entirely.
            if not _DISPATCH_TARGET.startswith("real") and not _stepped:
                _robot_state["joints"] = _read_robot_joints(model, data)
                # Canonical base state: chassis pose + steer positions + drive
                # velocities — the two motor quantities that reproduce base motion.
                _robot_state["base"] = _read_base_state(model, data)
                _robot_state["t"] = time.time()

            # 3.65) planar LiDAR sweep (~5 Hz) for the apps' top-down tile. Sim
            # paths ray-cast the live data HERE (this thread owns `data`, so no
            # race with mj_step); the real robot passes the RPLIDAR /scan through
            # the plant's subscriber. The state sender attaches a sweep to
            # robot_state only when its stamp changes, so the wire cost is one
            # ~360-float array per fresh sweep, not per 50 Hz state.
            if now - _lidar_next[0] >= 0.2:
                _lidar_next[0] = now
                try:
                    sc = (getattr(plant, "lidar_scan", lambda: None)()
                          if _DISPATCH_TARGET.startswith("real")
                          else _lidar_sweep(model, data))
                    if sc:
                        _robot_state["lidar"] = sc
                except Exception:
                    pass
            # 3.66) sparse joint dynamics (~10 Hz) — velocities + actuator torques
            # for the apps' telemetry panels (sim plants only; this thread owns
            # `data`, so no race with mj_step).
            if now - _dyn_next[0] >= 0.1 and not _DISPATCH_TARGET.startswith("real"):
                _dyn_next[0] = now
                try:
                    _robot_state["dynamics"] = _read_joint_dynamics(model, data)
                except Exception:
                    pass
            # 3.67) :9600 telemetry producer — FALLBACK cadence for iterations
            # with no physics substep (the substep block above is the primary
            # ~50 Hz producer; see _pump_telemetry defined before the loop).
            if TELEMETRY.enabled() and now - _telem_next[0] >= 0.02:
                _telem_next[0] = now
                try:
                    _pump_telemetry()
                except Exception:
                    pass
            # Body-calibration progress + result, so the app can show live pose prompts
            # and the measured profile (height, arm length, ...). status is None when
            # idle; profile persists once measured (apps display/save it).
            _robot_state["calibrationStatus"] = controller.calibration_status()
            if controller.profile is not None:
                _robot_state["operatorProfile"] = dict(controller.profile)   # already a dict
            # NOTE: full per-body transform streaming (for the MuJoCo mirror) is
            # disabled — it added ~581 floats/frame and slowed the loop.

            # 3b) STANDARDIZED actuation command → monitor. This is the resolved
            # control that drives the actuators (sim now, identical schema for the
            # real robot), surfaced so a human can verify it. Throttled to ~12 Hz
            # and handed to the hub by reference (the loop thread sets it; the
            # net-thread heartbeat broadcasts it — see monitor_hub.set_actuation).
            # Gated on a subscriber FIRST, so plain teleop pays nothing here — not
            # even the clock read or _desired_joints() — when no monitor is open.
            if MONITOR.enabled() and (now0 := time.time()) - _last_act[0] >= 0.08:
                _last_act[0] = now0
                _act_seq[0] += 1
                desired = _desired_joints()
                lift_m = sum(v for k, v in desired.items() if "lift" in k.lower())
                bs = _robot_state.get("base", {})
                steer, drive = bs.get("steer", []), bs.get("drive", [])
                wheels = {m: {"steer": steer[i] if i < len(steer) else 0.0,
                              "drive": drive[i] if i < len(drive) else 0.0}
                          for i, m in enumerate(("FL", "FR", "B"))}
                cmd = RobotCommand(
                    seq=_act_seq[0], stamp=now0, target=_DISPATCH_TARGET,
                    mode=dict(_control_spec), estop=_estop["active"],
                    base=BaseTwist(vx=round(controller.vx, 4), vy=round(controller.vy, 4),
                                   omega=round(controller.omega, 4), lift=round(lift_m, 4)),
                    joints=desired, wheels=wheels,
                    grippers={s: float(_joint_cmd[f"{s}_grip"])
                              for s in ("left", "right") if f"{s}_grip" in _joint_cmd},
                )
                MONITOR.set_actuation(cmd.to_dict())

            # 4) render ONE camera (round-robin) and push to the connected clients
            now = time.time()
            if render_cams and now - last_cam >= tick_period:
                last_cam = now
                render_one(render_cams[cam_idx])
                cam_idx = (cam_idx + 1) % n_cams

            # 5) periodic status line
            if now - last_log >= 2.0:
                hz = loop_iters / (now - last_log)     # fresh-frame sample rate
                loop_iters = 0
                last_log = now
                st = controller.status()
                stream = "mock input" if args.mock else (
                    f"{_clients['n']} client(s)" if _clients["n"] else "awaiting clients")
                ctl = (f"{_control_spec['method']}·{_control_spec['type']}·{_control_spec['region']}"
                       if wbc.active else _control_spec["method"])
                log.info("%-24s │ %-10s │ %-16s │ palm L/R %.3f/%.3f │ grasp L/R %.2f/%.2f │ %3.0f Hz",
                         ctl, "anchored" if st["calibrated"] else "no anchor",
                         stream, st["palm_err_l"], st["palm_err_r"],
                         st["grasp_l"], st["grasp_r"], hz)

            # Yield only when caught up — sleeping while behind real time was
            # costing ~6-10% of the loop budget at 30-50 Hz iteration rates.
            # SIM ONLY: on the real path acc is always drained to 0, so this
            # trailing 2 ms stacked on the CTRL_HZ pacing sleep every pass and
            # capped the loop at ~45 Hz (the last chunk of the "50 Hz gap").
            if not _IS_REAL and acc < model.opt.timestep:
                time.sleep(0.002)
    except KeyboardInterrupt:
        log.info("Shutting down.")


def banner(args) -> None:
    ip = console.lan_ip()
    mode = "MOCK — synthetic hands, no clients" if args.mock else "LIVE — waiting for clients"
    cams = (f"http://{ip}:{args.http_port}/camera/*  (in-process)" if args.cameras
            else f"http://{ip}:{args.http_port}  (via sim_camera_server.py)")
    console.banner(
        "M A B E L   T E L E O P   S E R V E R",
        "sim · real robot  ⇄  Vision Pro · iOS · web · gamepads",
        [("mode",    mode),
         ("teleop",  f"ws://{ip}:{args.ws_port}{args.ws_path}"),
         ("cameras", cams),
         ("scene",   os.path.basename(_resolve_scene(args.scene)))],
        footer=["ENTER re-anchor clutch    R reset sim    Ctrl-C quit"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MABEL unified teleop server — sim/real robot ⇄ any client "
                    "(Vision Pro, iOS, web, gamepads)")
    ap.add_argument("--mock", action="store_true",
                    help="drive with synthetic hands (no clients, no WebSocket)")
    ap.add_argument("--scene", default=None,
                    help="MJCF scene to load (default: mabel_sim.xml from config)")
    ap.add_argument("--self-collision", action=argparse.BooleanOptionalAction,
                    default=os.getenv("MABEL_SELF_COLLISION", "1") not in ("0", "", "false", "False"),
                    help="capsule self-collision avoidance across ALL modes (retarget/drive "
                         "kinematic projection + WBC torque barrier, same SelfCollisionGuard). "
                         "Default ON; disable with --no-self-collision or MABEL_SELF_COLLISION=0")
    ap.add_argument("--no-base", action="store_true",
                    help="disable the lazy swerve-base follower")
    ap.add_argument("--cameras", action="store_true",
                    help="render + serve the MJPEG cameras IN THIS PROCESS (slower; "
                         "default off — run sim_camera_server.py separately instead)")
    ap.add_argument("--head-res", default=f"{HEAD_W}x{HEAD_H}",
                    help="head/render resolution WxH (default 960x540)")
    ap.add_argument("--head-fov", type=float,
                    default=(float(os.getenv("MABEL_HEAD_FOV")) if os.getenv("MABEL_HEAD_FOV") else None),
                    help="override the simulated head-camera vertical FOV in degrees "
                         "(default: the model's MMlove 83.3° = 112° diagonal @4:3)")
    ap.add_argument("--cam-fps", type=float, default=20.0,
                    help="camera stream rate to clients, Hz (default 20)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--udp-video-port", type=int,
                    default=int(os.getenv("MABEL_UDP_VIDEO_PORT", "9091")),
                    help="UDP port for the latest-frame-wins JPEG video path "
                         "(udp_video.py; 0 disables). The headset subscribes here "
                         "for minimum-latency video and falls back to MJPEG/HTTP.")
    ap.add_argument("--state-file", default="/tmp/mabel_teleop_state.f64",
                    help="memmap path the desktop viewer (view_robot.py) reads")
    ap.add_argument("--ws-host", default=os.getenv("MABEL_WS_HOST", "0.0.0.0"))
    ap.add_argument("--ws-port", type=int, default=int(os.getenv("MABEL_WS_PORT", "9090")))
    ap.add_argument("--ws-path", default=os.getenv("MABEL_WS_PATH", "/teleop"))
    ap.add_argument("--http-host", default=os.getenv("MABEL_HTTP_HOST", "0.0.0.0"))
    ap.add_argument("--http-port", type=int, default=int(os.getenv("MABEL_HTTP_PORT", "8080")))
    ap.add_argument("--episodes-dir",
                    default=os.getenv("MABEL_EPISODES_DIR", os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dataset", "simulation", "ios")),
                    help="where teleop demonstration episodes are saved (the app's Record "
                         "button). Task folders live inside; each episode is saved as the "
                         "raw capture dir PLUS a curation-ready episode_XXXX.h5 "
                         "(robot_learning/data_curation ingests the folder directly).")
    ap.add_argument("--record-fps", type=float, default=float(os.getenv("MABEL_RECORD_FPS", "10")),
                    help="episode sampling rate in sim time (lower if recording slows the sim too much)")
    ap.add_argument("--demo-fps", type=float,
                    default=float(os.getenv("MABEL_DEMO_FPS", "30")),
                    help="animation-demo capture rate (Hz). Higher than the "
                         "episode recorder: a demo is a PERFORMANCE, and 10 Hz "
                         "makes a wave look like a slideshow.")
    ap.add_argument("--name", default=os.getenv("MABEL_ROBOT_NAME", ""),
                    help="robot display name (persisted; synced to clients + monitor)")
    args = ap.parse_args()

    # Robot display name: persisted file > --name/env > a hostname default.
    _default_prefix = "MABEL" if _IS_REAL else "MABEL Sim"
    _robot_name[0] = _load_robot_name(
        args.name or f"{_default_prefix} ({socket.gethostname().split('.')[0]})")
    _hello_info["name"] = _robot_name[0]

    try:
        args.head_w, args.head_h = (int(v) for v in args.head_res.lower().split("x"))
    except ValueError:
        ap.error("--head-res must look like 960x540")

    run(args)


if __name__ == "__main__":
    main()
