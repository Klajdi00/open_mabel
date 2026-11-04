#!/usr/bin/env python3
"""
mock_visionpro_client.py — end-to-end test of the LIVE teleop bridge with a
synthetic Vision Pro client.

Launches sim_teleop_bridge.py (headless, no cameras), connects to
ws://localhost:9090/teleop exactly like the headset (same JSON envelopes,
same Codable field names), streams 20 Hz teleop_frames whose right hand
opens/closes and whose wrist sweeps — then verifies through the bridge's own
outputs that BOTH halves of the hand mapping work:

  * wrist tracking : the arm joints move (robot_state telemetry)
  * finger tracking: the ORCA finger joints follow the curl (read from the
                     /tmp qpos memmap the bridge publishes for view_robot.py)

Run:  python3 tools/mock_visionpro_client.py
Exits 0 on success; prints the failing stage otherwise.
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = os.path.normpath(os.path.join(_HERE, ".."))
_ROOT = os.path.normpath(os.path.join(_SERVER, ".."))
sys.path.insert(0, _SERVER)
sys.path.insert(0, os.path.join(_ROOT, "controller"))
sys.path.insert(0, os.path.join(_ROOT, "vision_os_apps", "docs", "tools"))

from exp_hand_retarget import synth_hand  # ARKit-style synthetic skeleton

WS_URL = "ws://127.0.0.1:9090/teleop"
STATE_FILE = "/tmp/mabel_teleop_state_e2e.f64"
IDENT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def frame_payload(seq, hand_pose=None, wrist_xyz=(0.0, 0.0, 0.0)):
    """Build the EXACT JSON the Swift app emits each tick."""
    payload = {
        "sequence": seq,
        "timestamp": time.time(),
        "head": {"transform": {"matrix": IDENT}, "trackingState": "normal"},
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
    }
    if hand_pose is not None:
        dx, dy, dz = wrist_xyz
        joints = []
        for name, T in hand_pose.items():
            m = np.asarray(T, dtype=float).copy()
            joints.append({"joint": name,
                           "localTransform": {"matrix": m.reshape(16).tolist()},
                           "isTracked": True})
        anchor = [1.0, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz, 0, 0, 0, 1]
        payload["rightHand"] = {"chirality": "right",
                                 "anchorTransform": {"matrix": anchor},
                                 "joints": joints, "isTracked": True}
    return {"type": "teleop_frame", "payload": payload}


async def run_client(result):
    import websockets
    async with websockets.connect(WS_URL, max_size=2**20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        result["hello"] = hello["payload"]

        # The headset's exact sequence: spec → recalibrate → frames.
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "retarget", "controlType": "impedance",
            "region": "whole_body", "stiffness": 0.0}}))
        await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))

        telemetry = {}

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "robot_state":
                    telemetry.update(msg["payload"].get("jointPositions", {}))
        rtask = asyncio.create_task(reader())

        seq = 0
        t0 = time.time()
        baseline = None
        # Phase 1 (2 s): open hand at rest → anchor.
        # Phase 2 (4 s): fist + wrist sweep 12 cm forward.
        while time.time() - t0 < 6.0:
            t = time.time() - t0
            if t < 2.0:
                pose, wrist = synth_hand(mcp=0.05, pip=0.05, thumb=0.1), (0, 0, 0)
            else:
                g = min(1.0, (t - 2.0) / 1.5)
                pose = synth_hand(mcp=1.4 * g, pip=1.6 * g, thumb=0.8 * g)
                wrist = (0.12 * g, 0.0, 0.0)
            seq += 1
            await ws.send(json.dumps(frame_payload(seq, pose, wrist)))
            if 1.5 < t < 2.0 and baseline is None and telemetry:
                baseline = dict(telemetry)
            await asyncio.sleep(0.05)

        rtask.cancel()
        result["baseline"] = baseline or {}
        result["final"] = dict(telemetry)


async def measure_realtime(result, secs=3.0):
    """Δsim-time / Δwall-time while a client streams (bridge memmap word 0)."""
    state = np.memmap(STATE_FILE, dtype=np.float64, mode="r", shape=(2,))
    t0_wall, t0_sim = time.time(), float(state[0])
    await asyncio.sleep(secs)
    result["rt_factor"] = (float(state[0]) - t0_sim) / (time.time() - t0_wall)


def read_finger_qpos(model_path):
    """Read the bridge's qpos memmap and extract the right-hand finger joints."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(model_path)
    state = np.memmap(STATE_FILE, dtype=np.float64, mode="r", shape=(1 + model.nq,))
    out = {}
    for f in ("Index", "Middle", "Thumb"):
        for seg in ("MCP", "PIP"):
            name = f"Right {f} {seg}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                out[name] = float(state[1 + model.jnt_qposadr[jid]])
    return out


def main():
    scene = os.path.join(_ROOT, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
    env = dict(os.environ)
    bridge = subprocess.Popen(
        [sys.executable, "sim_teleop_bridge.py", "--scene", scene,
         "--state-file", STATE_FILE],
        cwd=_SERVER, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(6.0)                       # model load + servers up
        if bridge.poll() is not None:
            print(bridge.stdout.read()[-3000:])
            print("FAIL: bridge exited early")
            return 1

        result = {}

        async def session():
            client = asyncio.create_task(run_client(result))
            await asyncio.sleep(2.5)              # mid-stream, fist phase
            await measure_realtime(result)
            await client
        asyncio.run(session())

        fingers = read_finger_qpos(scene)
        base = result["baseline"]
        final = result["final"]

        print(f"hello: server={result['hello'].get('server')} "
              f"cameras={result['hello'].get('cameras')}")
        rt = result.get("rt_factor")
        if rt is not None:
            print(f"sim real-time factor while streaming: {rt:.2f}x")

        # Wrist tracking: the right-arm joints must have moved with the 12 cm sweep.
        arm_delta = max(abs(final.get(f"right_arm_{i}", 0) - base.get(f"right_arm_{i}", 0))
                        for i in range(1, 8))
        print(f"wrist sweep → max right-arm joint delta: {np.degrees(arm_delta):.1f} deg")

        # Finger tracking: the fist must close the ORCA fingers in the SIM qpos.
        print("finger qpos after fist:",
              {k: f"{np.degrees(v):.0f}°" for k, v in fingers.items()})
        mcp_ok = fingers.get("Right Index MCP", 0) > 0.5 and fingers.get("Right Index PIP", 0) > 0.5

        # And the app must SEE it: finger joints in the streamed telemetry.
        tel_mcp = final.get("right_index_mcp", 0.0)
        print(f"telemetry right_index_mcp: {np.degrees(tel_mcp):.0f}°")

        ok = True
        if tel_mcp < 0.5:
            print("FAIL: finger telemetry missing/flat (app model would not show hands)")
            ok = False
        if arm_delta < 0.05:
            print("FAIL: arm did not follow the wrist (wrist mapping broken)")
            ok = False
        if not mcp_ok:
            print("FAIL: fingers did not close (hand mapping broken)")
            ok = False
        if ok:
            print("E2E PASS — live bridge maps Vision Pro hand pose to the robot hand")
        if not ok:
            bridge.send_signal(signal.SIGINT)
            try:
                out = bridge.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                bridge.kill()
                out = bridge.communicate()[0]
            print("---- bridge log tail ----")
            print("\n".join(out.splitlines()[-25:]))
        return 0 if ok else 1
    finally:
        if bridge.poll() is None:
            bridge.send_signal(signal.SIGINT)
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()


if __name__ == "__main__":
    sys.exit(main())
