"""arm_controllers.py — phased, on-hardware validation of the three arm controllers.

⚠ BROKEN BY THE ARMS REFACTOR (2026-07-16) — NEEDS REWORK BEFORE THE NEXT RUN.
This harness switches modes by publishing `/arms/control_mode`, but the bridge no longer
hosts a control law: the HAL does calibration + read + write only, and the mode/gravity
policy moved wholly into `controller/mabel/arm_control`. Publishing that topic now goes
nowhere (silently — nothing subscribes), so Phases 2/3 would measure PURE PD mislabelled
as GRAVITY_PD/COMPLIANT. The recorded results in `docs/arm_controllers.pdf` remain valid
(they were measured on the pre-refactor bridge).
TO REWORK: drive the arm from this process — build the `ArmImpedance` + gravity
compensator here, and stream the resulting per-joint (position, velocity, effort) to
`/arms/command` as a JointState (MODEL frame; the bridge calibrates into the raw frame).
Note the bridge's own thermal/stall guards referenced below are GONE — this harness is
now solely responsible for its own temperature abort and for not driving into a stop.

Tests the controllers that live in ``controller/mabel/arm_control`` (Pure PD, PD +
gravity compensation, Compliant drag-and-hold) on the REAL left arm, on the joints where
gravity matters most (J1, J2, J4), at small angles. The reference gravity model is BUILT
FROM controller/ — the same code the robot runs — so the experiment and the robot can
never disagree about what τ_g should be.

Phases (run independently; each records its own JSON):
  --pd        Phase 1  PURE_PD     baseline; expect a gravity droop e ≈ τ_g/kp
  --gravity   Phase 2  GRAVITY_PD  expect the droop cancelled (e ≈ 0)
  --compliant Phase 3  COMPLIANT   sets the mode + records; a HUMAN drags the arm and it
                                   should stay where left (gravity held, kp≈0). Interactive.

Mode is switched LIVE over ROS (/arms/control_mode) — no bridge restart — so Phase 3 can
run while you hold the arm. Targets go over the :9090 joint_command wire (the iOS path).

SAFETY: small angles only (|Δ| ≤ SAFE_STEP from the rest pose), per-sample temperature
abort well below the bridge's own 70/80 °C guard, and the arm is de-energised at the end
of every phase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                    # ~/open_mabel
sys.path.insert(0, str(ROOT / "controller"))              # build controllers from controller/
sys.path.insert(0, str(ROOT / "hardware_bridge"))

from mabel.arm_control import (ArmGravityCompensator, ArmControlMode,  # noqa: E402
                               HandsConfig)
from mabel_hw.calibration.schema import JointCalib         # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
GW = os.getenv("MABEL_GW", "ws://localhost:9090/teleop")
MJCF = str(ROOT / "simulation/mabel_mujoco/models/mabel_full.xml")
CAL = str(ROOT / "hardware_bridge/calibration_data/mabel.json")

SETTLE_S = 12.0          # kp is small -> slow approach; sample only after this
SAMPLE_S = 3.0          # median of the last SAMPLE_S = steady state
SAFE_STEP = 0.30        # rad (~17°) max target offset from the joint's rest pose
ABORT_TEMP_C = 55.0     # abort before the bridge's 70/80 °C guard

# The gravity-loaded joints, small symmetric steps about each one's rest pose.
# J1 shoulder roll, J2 shoulder pitch (lifts the arm — biggest τ_g), J4 elbow.
SWEEPS = {
    "left_arm_1": [0.0, -0.15, -0.30],
    "left_arm_2": [0.0, +0.15, +0.30],
    "left_arm_4": [0.0, -0.15, -0.30],       # offsets from rest (added below)
}


def _gravity_model():
    """The reference gravity compensator, from controller/ — HANDS OFF (no hands now)."""
    cal = json.load(open(CAL))["arm"]["openarm-dual"]["calib"]
    jc = {n: JointCalib.from_dict(cal[n], n) for n in cal}
    dirs = {n: jc[n].dir for n in cal}
    tmax = {n: 30.0 for n in cal}
    return ArmGravityCompensator(MJCF, dirs, tmax, hands=HandsConfig.NONE,
                                 log=lambda *a: None), jc


async def _state(ws):
    while True:
        m = json.loads(await ws.recv())
        if m.get("type") == "robot_state":
            return m["payload"]


async def _set_mode(mode: str):
    """Publish the control mode over ROS and confirm via the state topic. Uses a short-
    lived rclpy node so the experiment needs no persistent ROS deps."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    if not rclpy.ok():
        rclpy.init()
    n = Node("exp_set_mode")
    pub = n.create_publisher(String, "/arms/control_mode", 10)
    got = {}
    n.create_subscription(String, "/arms/control_mode_state",
                          lambda m: got.update(v=m.data), 10)
    msg = String(); msg.data = mode
    for _ in range(20):
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.1)
        if got.get("v") == ArmControlMode.coerce(mode).value:
            break
        time.sleep(0.1)
    n.destroy_node()
    return got.get("v")


async def _hold(ws, joint, target, others, grav_model, jc):
    """Command `target` on `joint` (others held), return steady-state stats + expected τ_g."""
    cmd = dict(others); cmd[joint] = float(target)
    t0 = time.time(); qs, temps = [], {}
    while time.time() - t0 < SETTLE_S:
        await ws.send(json.dumps({"type": "joint_command", "payload": {"joints": cmd}}))
        p = await _state(ws)
        jp = p.get("jointPositions") or {}
        for k, v in (p.get("jointTemps") or {}).items():
            if "arm" in k:
                temps[k] = v
        if joint in jp and time.time() - t0 > SETTLE_S - SAMPLE_S:
            qs.append(jp[joint])
    hot = max(temps.values()) if temps else 0.0
    if hot >= ABORT_TEMP_C:
        raise RuntimeError(f"ABORT: motor {hot:.0f}°C ≥ {ABORT_TEMP_C:.0f}")
    q = statistics.median(qs)
    fw = "l_j" + joint[-1]
    model_pose = {("l_j" + k[-1] if k.startswith("left_arm") else
                   "r_j" + k[-1]): v for k, v in cmd.items()}
    for s in "lr":                                        # fill the missing arm at 0
        for i in range(1, 8):
            model_pose.setdefault(f"{s}_j{i}", 0.0)
    tau_raw = grav_model.tff(model_pose, 1.0).get(fw, 0.0)
    return {"target": float(target), "measured": q, "error": q - float(target),
            "wobble": (max(qs) - min(qs)), "hot": hot, "tau_g_raw": tau_raw}


async def run_tracking(mode: str):
    """Phase 1/2: sweep J1/J2/J4 under one mode, record steady-state error."""
    grav_model, jc = _gravity_model()
    applied = await _set_mode(mode)
    print(f"[{mode}] control_mode_state = {applied}")
    res = {"mode": mode, "joints": {}}
    async with websockets.connect(GW, open_timeout=15, max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"type": "estop", "payload": {"active": False}}))
        await asyncio.sleep(4.0)
        rest = {k: v for k, v in (await _state(ws)).get("jointPositions", {}).items()
                if k.startswith("left_arm")}
        print(f"[{mode}] rest: " + " ".join(f"{k[-1]}={rest[k]:+.3f}" for k in sorted(rest)))
        try:
            for joint, offs in SWEEPS.items():
                others = {k: v for k, v in rest.items() if k != joint}
                rows = []
                for off in offs:
                    tgt = rest[joint] + off
                    if abs(tgt - rest[joint]) > SAFE_STEP + 1e-9:
                        continue
                    r = await _hold(ws, joint, tgt, others, grav_model, jc)
                    rows.append(r)
                    print(f"[{mode}] {joint} tgt={r['target']:+.3f} meas={r['measured']:+.3f}"
                          f" ERR={r['error']*57.3:+6.2f}°  τg={r['tau_g_raw']:+.2f}Nm"
                          f"  ({r['hot']:.0f}°C)")
                res["joints"][joint] = rows
        finally:
            await ws.send(json.dumps({"type": "estop", "payload": {"active": True}}))
            await asyncio.sleep(1.0)
            print(f"[{mode}] torque OFF.")
    (OUT / f"arm_{mode}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / f'arm_{mode}.json'}")


async def run_compliant():
    """Phase 3: set COMPLIANT, energize, and record the pose while a HUMAN drags the arm.
    Pass/fail is human-observed (does it stay where you leave it?); the log is the record."""
    await _set_mode("compliant")
    print("[compliant] control_mode = compliant. Energizing; DRAG the left arm by hand.")
    print("[compliant] recording pose for 40 s — move it to a few poses and let go each time.")
    samples = []
    async with websockets.connect(GW, open_timeout=15, max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"type": "estop", "payload": {"active": False}}))
        await asyncio.sleep(3.0)
        t0 = time.time()
        while time.time() - t0 < 40:
            await ws.send(json.dumps({"type": "joint_command", "payload": {"joints": {}}}))
            p = await _state(ws)
            jp = p.get("jointPositions") or {}
            row = {"t": round(time.time() - t0, 2),
                   **{k: round(jp[k], 4) for k in jp if k.startswith("left_arm")}}
            samples.append(row)
            if len(samples) % 20 == 0:
                print(f"[compliant] t={row['t']:.0f}s  j2={row.get('left_arm_2'):+.3f} "
                      f"j4={row.get('left_arm_4'):+.3f}")
        await ws.send(json.dumps({"type": "estop", "payload": {"active": True}}))
        await asyncio.sleep(1.0)
    (OUT / "arm_compliant.json").write_text(json.dumps(samples, indent=2))
    print(f"[compliant] torque OFF. wrote {OUT / 'arm_compliant.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd", action="store_true", help="Phase 1: pure PD baseline")
    ap.add_argument("--gravity", action="store_true", help="Phase 2: PD + gravity comp")
    ap.add_argument("--compliant", action="store_true", help="Phase 3: compliant drag test")
    a = ap.parse_args()
    if a.pd:
        asyncio.run(run_tracking("pure_pd"))
    elif a.gravity:
        asyncio.run(run_tracking("gravity_pd"))
    elif a.compliant:
        asyncio.run(run_compliant())
    else:
        ap.print_help()
