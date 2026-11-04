#!/usr/bin/env python3
"""
mujoco_pose_viz.py — Real-time MuJoCo visualization of Vision Pro pose data.

Connects to the MabelTeleop app via WebSocket (no separate companion app needed).
Renders head + both hands at 27 joints each.

Key design: ALL 27 joints are always rendered regardless of ARKit's per-joint
tracking confidence. ARKit always computes an estimated transform for every joint
even when occluded — the isTracked flag is a confidence signal, not a validity
gate. Filtering by isTracked caused joints to disappear during partial occlusion.
This matches how VisionProTeleop/avp_stream works: no isTracked filtering at all.

Usage:
    cd server
    mjpython mujoco_pose_viz.py [--host 0.0.0.0] [--port 9090]

App config (Config.plist):
    telemetryPort = 9090   telemetryPath = /teleop

Coordinate systems
------------------
ARKit (visionOS): right-handed, Y-up, -Z forward
MuJoCo:           right-handed, Z-up,  X forward
Conversion:  X_mj = X_arkit,  Y_mj = -Z_arkit,  Z_mj = Y_arkit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

from wire_protocol import WireCodec, WireMessageType, TeleopFrame, HandPose

# ── Joint ordering — matches HandJoint enum in PoseModels.swift ──────────────

JOINT_NAMES = [
    "wrist",                        # 0
    "thumb_knuckle",                # 1
    "thumb_intermediate_base",      # 2
    "thumb_intermediate_tip",       # 3
    "thumb_tip",                    # 4
    "index_metacarpal",             # 5
    "index_knuckle",                # 6
    "index_intermediate_base",      # 7
    "index_intermediate_tip",       # 8
    "index_tip",                    # 9
    "middle_metacarpal",            # 10
    "middle_knuckle",               # 11
    "middle_intermediate_base",     # 12
    "middle_intermediate_tip",      # 13
    "middle_tip",                   # 14
    "ring_metacarpal",              # 15
    "ring_knuckle",                 # 16
    "ring_intermediate_base",       # 17
    "ring_intermediate_tip",        # 18
    "ring_tip",                     # 19
    "little_metacarpal",            # 20
    "little_knuckle",               # 21
    "little_intermediate_base",     # 22
    "little_intermediate_tip",      # 23
    "little_tip",                   # 24
    "forearm_wrist",                # 25
    "forearm_arm",                  # 26
]
N_JOINTS = len(JOINT_NAMES)   # 27

# Skeleton edge list
HAND_CONNECTIONS = [
    # Forearm
    (26, 25), (25, 0),
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    # Middle
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    # Ring
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    # Little
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
    # Palm cross-links
    (5, 10), (10, 15), (15, 20),
]

# ── Coordinate conversion: ARKit Y-up → MuJoCo Z-up ─────────────────────────

_C = np.array([[1, 0, 0],
               [0, 0,-1],
               [0, 1, 0]], dtype=float)

def _arkit_pos_to_mj(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], -p[2], p[1]])

def _arkit_rot_to_mj(r: np.ndarray) -> np.ndarray:
    return _C @ r @ _C.T

# ── MuJoCo XML ───────────────────────────────────────────────────────────────

def _hand_bodies(prefix: str, rgba: str) -> str:
    return "\n".join(
        f'    <body name="{prefix}{i}" mocap="true">'
        f'<geom type="sphere" size="0.007" rgba="{rgba}"/>'
        f'<site name="{prefix}s{i}"/></body>'
        for i in range(N_JOINTS)
    )

def _hand_tendons(prefix: str, rgba: str) -> str:
    return "\n".join(
        f'  <spatial name="{prefix}l{idx}" rgba="{rgba}" width="0.003">'
        f'<site site="{prefix}s{a}"/><site site="{prefix}s{b}"/></spatial>'
        for idx, (a, b) in enumerate(HAND_CONNECTIONS)
    )

def build_xml() -> str:
    return f"""<mujoco model="visionpro">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 0"/>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1=".12 .18 .30" rgb2="0 0 0" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1=".07 .07 .07" rgb2=".11 .11 .11" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="12 12" reflectance="0.04"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse=".8 .8 .8" specular=".2 .2 .2"/>
    <geom name="floor" type="plane" size="6 6 .05" material="floor_mat"/>

    <body name="head" mocap="true">
      <geom type="box"      size="0.09 0.065 0.055" rgba=".35 .55 1 .55"/>
      <geom type="cylinder" pos="0.09 0 0" size="0.01 0.015" rgba=".35 .55 1 .9"/>
    </body>

{_hand_bodies("L", "0.10 0.80 0.90 1.0")}
{_hand_bodies("R", "0.10 0.90 0.30 1.0")}
  </worldbody>
  <tendon>
{_hand_tendons("L", "0.10 0.80 0.90 0.55")}
{_hand_tendons("R", "0.10 0.90 0.30 0.55")}
  </tendon>
</mujoco>"""

# ── World positions from HandPose ─────────────────────────────────────────────
#
# Critical fix: we do NOT filter by jp.isTracked.
#
# ARKit always computes an estimated anchorFromJointTransform for every joint,
# even when it is occluded. The isTracked flag is a confidence signal — it tells
# you whether ARKit directly observed that joint, not whether the transform is
# valid. Filtering on it caused joints to disappear whenever a finger was hidden.
#
# VisionProTeleop/avp_stream never exposes isTracked at all; it sends every
# joint matrix unconditionally. We replicate that behaviour here.

_HIDDEN = np.array([0.0, 0.0, -5.0])

def hand_world_positions(hand: HandPose) -> np.ndarray:
    """
    Returns (N_JOINTS, 3) world positions in MuJoCo frame.
    Uses ALL joints regardless of tracking confidence — no isTracked filter.
    """
    anchor = np.array(hand.anchorTransform.matrix, dtype=float).reshape(4, 4)
    positions = np.tile(_HIDDEN, (N_JOINTS, 1))

    # Build name → local-transform map for every joint (tracked or not)
    name_to_local: dict[str, np.ndarray] = {}
    for jp in hand.joints:
        # ← No isTracked check — ARKit always has a valid estimated transform
        local = np.array(jp.localTransform.matrix, dtype=float).reshape(4, 4)
        name_to_local[jp.joint] = local

    for i, name in enumerate(JOINT_NAMES):
        local = name_to_local.get(name)
        if local is not None:
            world = anchor @ local
            positions[i] = _arkit_pos_to_mj(world[:3, 3])

    return positions

# ── Shared latest-frame slot ─────────────────────────────────────────────────

_latest: deque[TeleopFrame] = deque(maxlen=1)

# ── WebSocket server (daemon thread) ─────────────────────────────────────────

async def _ws_handler(websocket, path=None):
    addr = getattr(websocket, "remote_address", "?")
    _log(f"Vision Pro connected: {addr}")
    try:
        await websocket.send(json.dumps({
            "type": "hello",
            "payload": {"server": "mabel_viz", "version": "1"}
        }))
    except Exception:
        pass
    try:
        async for raw in websocket:
            try:
                msg_type, payload = WireCodec.decode_envelope(raw)
                if msg_type == WireMessageType.TELEOP_FRAME:
                    _latest.append(WireCodec.decode_teleop_frame(payload))
                elif msg_type == WireMessageType.PING:
                    await websocket.send(WireCodec.encode_pong().decode())
            except Exception:
                pass
    except Exception:
        pass
    _log(f"Vision Pro disconnected: {addr}")


def _start_ws_thread(host: str, port: int) -> None:
    import websockets

    async def _serve():
        try:
            async with websockets.serve(_ws_handler, host, port,
                                        ping_interval=None, max_size=2**20):
                _log(f"WebSocket listening on ws://{host}:{port}")
                await asyncio.Future()
        except OSError as e:
            if e.errno in (48, 98):
                _log(
                    f"\033[1;31mERROR: Port {port} already in use.\033[0m\n"
                    f"  Kill it first:  lsof -ti :{port} | xargs kill -9\n"
                    f"  Or run cameras separately:  python main.py --cameras-only"
                )
            else:
                _log(f"WebSocket error: {e}")

    asyncio.run(_serve())

# ── Terminal HUD ─────────────────────────────────────────────────────────────

_hud_ready = False

def _hud(frame: TeleopFrame, fps: float) -> None:
    global _hud_ready
    if not _hud_ready:
        sys.stdout.write("\033[2J"); _hud_ready = True

    lh = frame.leftHand
    rh = frame.rightHand
    lw = lh.anchorTransform.position_xyz() if lh else None
    rw = rh.anchorTransform.position_xyz() if rh else None
    # Count joints that were actually in the payload (tracked or not)
    lj = len(lh.joints) if lh else 0
    rj = len(rh.joints) if rh else 0
    hx, hy, hz = frame.head.transform.position_xyz()

    out = [
        "\033[H",
        "\033[1;34m" + "═" * 58 + "\033[0m",
        f"  \033[1;32mMABEL VISION PRO VIZ\033[0m"
        f"  seq \033[33m{frame.sequence:<9}\033[0m"
        f"fps \033[32m{fps:>5.1f}\033[0m"
        f"  \033[35m{frame.mode.value}\033[0m",
        "  " + "─" * 54,
        f"  \033[1;36mHEAD \033[0m  X{hx:>+7.3f}  Y{hy:>+7.3f}  Z{hz:>+7.3f}"
        f"  [{frame.head.trackingState.value}]",
    ]

    if lw:
        x, y, z = lw
        out.append(
            f"  \033[1;36mL ARM\033[0m  X{x:>+7.3f}  Y{y:>+7.3f}  Z{z:>+7.3f}"
            f"  \033[32m{lj}/{N_JOINTS} joints\033[0m"
        )
    else:
        out.append("  \033[2mL ARM  not in scene\033[0m")

    if rw:
        x, y, z = rw
        out.append(
            f"  \033[1;32mR ARM\033[0m  X{x:>+7.3f}  Y{y:>+7.3f}  Z{z:>+7.3f}"
            f"  \033[32m{rj}/{N_JOINTS} joints\033[0m"
        )
    else:
        out.append("  \033[2mR ARM  not in scene\033[0m")

    out.append("\033[1;34m" + "═" * 58 + "\033[0m")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()

# ── Main ─────────────────────────────────────────────────────────────────────

def main(host: str, port: int) -> None:
    xml   = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    head_mid   = model.body("head").mocapid[0]
    left_mids  = [model.body(f"L{i}").mocapid[0] for i in range(N_JOINTS)]
    right_mids = [model.body(f"R{i}").mocapid[0] for i in range(N_JOINTS)]

    for mid in left_mids + right_mids:
        data.mocap_pos[mid] = _HIDDEN

    ws_thread = threading.Thread(
        target=_start_ws_thread, args=(host, port), daemon=True
    )
    ws_thread.start()

    _log("MuJoCo viewer starting — waiting for Vision Pro…")

    fps_frames = 0
    fps_t0     = time.monotonic()
    fps        = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 140
        viewer.cam.elevation = -20
        viewer.cam.distance  = 2.8
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        while viewer.is_running():
            t0 = time.monotonic()

            if _latest:
                frame: TeleopFrame = _latest[0]

                # Pre-compute outside the lock
                h_mat = np.array(frame.head.transform.matrix, dtype=float).reshape(4, 4)
                h_pos = _arkit_pos_to_mj(h_mat[:3, 3])
                h_rot = _arkit_rot_to_mj(h_mat[:3, :3]).flatten()

                # hand_world_positions() now uses ALL joints — no isTracked filter
                l_pos = hand_world_positions(frame.leftHand)  if (frame.leftHand  and frame.leftHand.isTracked)  else None
                r_pos = hand_world_positions(frame.rightHand) if (frame.rightHand and frame.rightHand.isTracked) else None

                with viewer.lock():
                    data.mocap_pos[head_mid] = h_pos
                    mujoco.mju_mat2Quat(data.mocap_quat[head_mid], h_rot)

                    if l_pos is not None:
                        for i, mid in enumerate(left_mids):
                            data.mocap_pos[mid] = l_pos[i]
                    else:
                        for mid in left_mids:
                            data.mocap_pos[mid] = _HIDDEN

                    if r_pos is not None:
                        for i, mid in enumerate(right_mids):
                            data.mocap_pos[mid] = r_pos[i]
                    else:
                        for mid in right_mids:
                            data.mocap_pos[mid] = _HIDDEN

                    # Propagate mocap_pos → xpos so the renderer sees the updates
                    mujoco.mj_forward(model, data)

                fps_frames += 1
                dt = time.monotonic() - fps_t0
                if dt >= 1.0:
                    fps = fps_frames / dt
                    fps_frames = 0
                    fps_t0 = time.monotonic()

                _hud(frame, fps)

            viewer.sync()

            elapsed = time.monotonic() - t0
            sleep   = 0.005 - elapsed
            if sleep > 0:
                time.sleep(sleep)


def _log(msg: str) -> None:
    print(f"\n  [viz] {msg}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MuJoCo visualization of Vision Pro pose tracking"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9090, type=int)
    args = parser.parse_args()
    main(args.host, args.port)
