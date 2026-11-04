#!/usr/bin/env python3
"""exp_vp_e2e_mock.py — RIGOROUS end-to-end test of the LIVE teleop bridge with
mock Vision Pro human-tracking data (the exact app JSON over ws://:9090/teleop).

Extends tools/mock_visionpro_client.py to cover the full 2026-07 app-path
controller rewiring: it launches the real sim_teleop_bridge.py, streams
synthetic BOTH-hand 27-joint ARKit skeletons (with metacarpals + forearm_arm),
and asserts through the bridge's own robot_state telemetry that every layer
works:

  P1  anchor            calibrate/anchor on neutral pose; telemetry flows
  P2  wrist orientation palm-pitch sweep → the ORCA wrist + arm joints track
  P3  fist / open       webcam-parity fingers close AND reopen (both directions)
  P4  torso turn        elbow-line body yaw (+30°) → chassis yaw follows
  P5  head-box cascade  within-box lean: base holds; overflow lean: base engages
  P6  nav joystick      stick drives the base through the tip-safe motion model
                        and the base STOPS on release
  RT  realtime          sim time advances ≈ wall time while streaming

Run:  python3 server/tools/exp_vp_e2e_mock.py               (spawns its own bridge)
      python3 server/tools/exp_vp_e2e_mock.py --attach      (tests an ALREADY-RUNNING
        production server on ws://:9090 — e.g. start_teleop.sh — using its default
        /tmp/mabel_teleop_state.f64 for the realtime check)
Exits 0 on success.
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
sys.path.insert(0, os.path.join(_ROOT, "simulation", "mabel_mujoco", "scripts", "tools"))

# Synthetic-operator builders shared with the offline harness (same mock data).
from exp_vp_app_teleop import hand_payload, palm_R, _rot, _mat16, _T  # noqa: E402

WS_URL = "ws://127.0.0.1:9090/teleop"
STATE_FILE = "/tmp/mabel_teleop_state_e2e_mock.f64"
SCENE = os.path.join(_ROOT, "simulation", "mabel_mujoco", "scenes", "mabel_sim.xml")
HEAD_P = np.array([0.0, 1.6, 0.0])


def operator_frame(seq, yaw=0.0, curl=0.0, palm_pitch=0.0, head_dz=0.0,
                   joystick=(0.0, 0.0, 0.0, 0.0)):
    """One teleop_frame payload for a full synthetic operator: both hands (27-joint
    skeletons incl. forearm_arm), head, nav stick. `yaw` turns the whole operator
    torso (hands + elbows) about the head's vertical axis."""
    Ry = _rot([0, 1, 0], yaw)
    hands = {}
    curls = {f: (1.3 * curl, 0.9 * curl, 0.5 * curl)
             for f in ("index", "middle", "ring", "little")} if curl else None
    for side, x in (("right", 0.25), ("left", -0.25)):
        w = Ry @ np.array([x, 1.20, -0.40])
        R_hand = Ry @ palm_R(side, pitch=palm_pitch)
        hands[side] = hand_payload(side, w, R_hand, curls=curls,
                                   thumb=(0.05, 0.15, 0.1))
    lx, ly, rx, ry = joystick
    return {"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(), "mode": "Arms & Hands",
        "navJoystick": {"lx": lx, "ly": ly, "rx": rx, "ry": ry},
        "head": {"transform": _mat16(_T(HEAD_P + [0, 0, head_dz])),
                 "trackingState": "normal"},
        "rightHand": hands["right"], "leftHand": hands["left"],
    }}


class Telemetry:
    def __init__(self):
        self.joints = {}
        self.base = {}
        self.n = 0

    def update(self, payload):
        self.joints.update(payload.get("jointPositions", {}))
        if payload.get("base"):
            self.base = payload["base"]
        self.n += 1

    def snap(self):
        return dict(self.joints), dict(self.base)


async def stream(ws, tel, secs, hz=30.0, **kw_fn):
    """Stream frames for `secs`, kwargs per-frame via callables or constants."""
    t0 = time.time()
    seq = int(t0 * 1000) % 10**6
    while (t := time.time() - t0) < secs:
        kw = {k: (v(t) if callable(v) else v) for k, v in kw_fn.items()}
        seq += 1
        await ws.send(json.dumps(operator_frame(seq, **kw)))
        await asyncio.sleep(1.0 / hz)
    return tel.snap()


def check(name, cond, detail):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    return bool(cond)


async def run_session(results):
    import websockets
    tel = Telemetry()
    async with websockets.connect(WS_URL, max_size=2**20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello"
        print(f"bridge hello: server={hello['payload'].get('server')} "
              f"version={hello['payload'].get('version', '?')}")

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "robot_state":
                    tel.update(msg["payload"])
        rtask = asyncio.create_task(reader())

        # The app's exact entry sequence: control spec → recalibrate → frames.
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "retarget", "controlType": "impedance",
            "region": "wholeBody", "stiffness": 0.0}}))
        await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))

        ok = True

        # P1 anchor ─ neutral pose, engage + calibrate
        j0, b0 = await stream(ws, tel, 3.0)
        ok &= check("P1 anchor/telemetry", tel.n > 30 and "yaw" in b0,
                    f"{tel.n} robot_state msgs, base yaw={b0.get('yaw', 0):+.3f}")

        # RT realtime factor mid-stream
        state = np.memmap(STATE_FILE, dtype=np.float64, mode="r", shape=(2,))
        rt0_wall, rt0_sim = time.time(), float(state[0])

        # P2 wrist orientation ─ palm pitch sinusoid ±45°
        wq = []

        async def sampler():
            while True:
                wq.append(float(tel.joints.get("right_wrist", 0.0)))
                await asyncio.sleep(0.05)
        stask = asyncio.create_task(sampler())
        await stream(ws, tel, 6.0,
                     palm_pitch=lambda t: np.deg2rad(45) * np.sin(2 * np.pi * t / 3.0))
        stask.cancel()
        rt = (float(state[0]) - rt0_sim) / (time.time() - rt0_wall)
        arm_span = 0.0
        j2, _ = tel.snap()
        for i in range(1, 8):
            k = f"right_arm_{i}"
            arm_span = max(arm_span, abs(j2.get(k, 0) - j0.get(k, 0)))
        wr_span = (max(wq) - min(wq)) if wq else 0.0
        ok &= check("P2 wrist orientation", wr_span > 0.3,
                    f"ORCA right_wrist span {wr_span:.2f} rad over ±45° palm pitch "
                    f"(arm joint Δ {np.degrees(arm_span):.0f}°)")
        ok &= check("RT realtime", rt > 0.85, f"sim/wall = {rt:.2f}x while streaming")

        # P3 fist → open ─ webcam-parity fingers, both directions
        jf, _ = await stream(ws, tel, 3.0, curl=lambda t: min(1.0, t / 2.0))
        mcp_fist = jf.get("right_index_mcp", 0.0)
        jo, _ = await stream(ws, tel, 3.0, curl=lambda t: max(0.0, 1.0 - t / 2.0))
        mcp_open = jo.get("right_index_mcp", 0.0)
        ok &= check("P3 fist closes", mcp_fist > 0.6,
                    f"right_index_mcp {np.degrees(mcp_fist):.0f}° at full curl")
        ok &= check("P3 hand reopens", mcp_open < 0.25,
                    f"right_index_mcp {np.degrees(mcp_open):.0f}° after reopen")

        # P4 torso turn ─ elbow-line body yaw → chassis follows
        _, b_pre = tel.snap()
        _, b_turn = await stream(ws, tel, 6.0,
                                 yaw=lambda t: np.deg2rad(30) * min(1.0, t / 1.5))
        dyaw = np.degrees((b_turn.get("yaw", 0) - b_pre.get("yaw", 0) + np.pi)
                          % (2 * np.pi) - np.pi)
        ok &= check("P4 torso-yaw follow", 18.0 < dyaw < 42.0,
                    f"operator +30° → chassis {dyaw:+.1f}°")

        # recenter for P5
        await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))
        _, b_c = await stream(ws, tel, 2.0, yaw=np.deg2rad(30))

        # P5a within-box lean: base must hold. (Head motion is RAMPED like a real
        # head — an instantaneous teleport is correctly treated as an ARKit
        # origin jump and re-anchored.)
        _, b_in = await stream(ws, tel, 3.0, yaw=np.deg2rad(30),
                               head_dz=lambda t: -0.10 * min(1.0, t / 0.8))
        d_in = float(np.hypot(b_in.get("x", 0) - b_c.get("x", 0),
                              b_in.get("y", 0) - b_c.get("y", 0)))
        ok &= check("P5a within-box lean holds base", d_in < 0.04,
                    f"base moved {d_in:.3f} m for a 0.10 m head lean")
        # P5b overflow lean: base must engage (ramp on from the -0.10 lean)
        _, b_ov = await stream(ws, tel, 4.0, yaw=np.deg2rad(30),
                               head_dz=lambda t: -0.10 - 0.35 * min(1.0, t / 1.0))
        d_ov = float(np.hypot(b_ov.get("x", 0) - b_c.get("x", 0),
                              b_ov.get("y", 0) - b_c.get("y", 0)))
        ok &= check("P5b overflow lean engages base", d_ov > 0.06,
                    f"base moved {d_ov:.3f} m for a 0.45 m head lean")

        # P6 nav joystick (Base Driving mode) ─ tip-safe drive, stop-and-hold on
        # release. In retarget mode the head-follow station-keeps the base (by
        # design), so the drone stop-on-release contract belongs to navigation.
        # ASSERTION LEVEL: the controller's contract is asserted at the ACTUATORS
        # (wheel drive speeds in base telemetry reach the tip-safe command and
        # return to zero on release). Chassis DISPLACEMENT is reported but not
        # asserted — the sim's wheel/ground contact currently caps realized base
        # speed at ~0.04 m/s regardless of command (a pre-existing MJCF contact
        # defect, verified independent of this controller; see the 2026-07-08
        # investigation notes). The real robot realizes the same twist through
        # the hardware swerve instead.
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "navigation", "controlType": "impedance",
            "region": "wholeBody", "stiffness": 0.0}}))
        _, b_j0 = await stream(ws, tel, 1.0)
        wheel = []

        async def wheel_sampler():
            while True:
                if tel.base.get("drive"):
                    wheel.append(max(abs(v) for v in tel.base["drive"]))
                await asyncio.sleep(0.05)
        wtask = asyncio.create_task(wheel_sampler())
        _, b_j1 = await stream(ws, tel, 2.5, joystick=(0.0, 1.0, 0.0, 0.0))
        wtask.cancel()
        d_drive = float(np.hypot(b_j1.get("x", 0) - b_j0.get("x", 0),
                                 b_j1.get("y", 0) - b_j0.get("y", 0)))
        w_peak = max(wheel) if wheel else 0.0
        # 0.8 m/s tip-safe cap / 0.0381 m wheel radius ≈ 21 rad/s at full stick
        ok &= check("P6 stick drives wheels", w_peak > 15.0,
                    f"peak wheel speed {w_peak:.1f} rad/s at full stick "
                    f"(chassis moved {d_drive:.2f} m — sim ground-contact limited)")
        # release: wheels must return to zero (drone stop-and-hold)
        await stream(ws, tel, 2.0)
        _, b_s1 = tel.snap()
        w_rest = max(abs(v) for v in b_s1.get("drive", [0.0]))
        ok &= check("P6 stops on release", w_rest < 1.5,
                    f"wheel speed {w_rest:.2f} rad/s after release")

        rtask.cancel()
        results["ok"] = ok


def main():
    global STATE_FILE
    if "--attach" in sys.argv:
        # Test a live production server (start_teleop.sh) instead of spawning one.
        idx = sys.argv.index("--attach")
        STATE_FILE = (sys.argv[idx + 1] if len(sys.argv) > idx + 1
                      and not sys.argv[idx + 1].startswith("-")
                      else "/tmp/mabel_teleop_state.f64")
        print(f"ATTACH mode: testing the running server on {WS_URL} "
              f"(state file {STATE_FILE})")
        results = {}
        asyncio.run(run_session(results))
        if results.get("ok"):
            print("\nE2E PASS — mock Vision Pro tracking drives retargeting + "
                  "whole-body control correctly through the live bridge")
            return 0
        print("\nE2E FAIL — see failing phase above")
        return 1

    subprocess.run("lsof -ti :9090 | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(1.0)
    bridge = subprocess.Popen(
        [sys.executable, "sim_teleop_bridge.py", "--scene", SCENE,
         "--state-file", STATE_FILE],
        cwd=_SERVER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(7.0)
        if bridge.poll() is not None:
            print(bridge.stdout.read()[-3000:])
            print("FAIL: bridge exited early")
            return 1
        results = {}
        asyncio.run(run_session(results))
        if results.get("ok"):
            print("\nE2E PASS — mock Vision Pro tracking drives retargeting + "
                  "whole-body control correctly through the live bridge")
            return 0
        print("\nE2E FAIL — see failing phase above")
        bridge.send_signal(signal.SIGINT)
        try:
            out = bridge.communicate(timeout=5)[0]
        except subprocess.TimeoutExpired:
            bridge.kill()
            out = bridge.communicate()[0]
        print("---- bridge log tail ----")
        print("\n".join(out.splitlines()[-25:]))
        return 1
    finally:
        if bridge.poll() is None:
            bridge.send_signal(signal.SIGINT)
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()


if __name__ == "__main__":
    sys.exit(main())
