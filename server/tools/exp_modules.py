#!/usr/bin/env python3
"""
exp_modules.py — does `robot_state.modules` tell the TRUTH?

The module roster is only worth shipping if it MEASURES the robot instead of
printing a hardcoded "all good". These checks are built to catch exactly that
failure: each one tries to make the roster lie.

  E1  CONTRACT      all 13 modules, right ids/labels/groups, right order
  E2  ACTUATORS     base/torso/lift/neck/arms/hands report connected in sim,
                    with joint counts that match the MJCF (7-DOF arms, etc.)
  E3  CAMERAS       the roster agrees with what the video server ACTUALLY serves
                    (--expect-cams pins it: run the stack --head-only and the
                    wrist/front cams MUST go dark, or the status is fiction)
  E4  LIDAR         connected while sweeps flow
  E5  LIVENESS      kill the camera server → every camera flips to DOWN within
                    the TTL. This is the check that a hardcoded roster fails.
  E6  BANDWIDTH     the block is change-gated, not re-sent at 50 Hz

Usage:
  python3 tools/exp_modules.py                          # full stack
  python3 tools/exp_modules.py --expect-cams main       # stack started --head-only
  python3 tools/exp_modules.py --kill-cameras           # E5 (kills sim_camera_server)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets

EXPECT = [
    ("base", "Base", None), ("torso", "Torso", None), ("lift", "Lift", None),
    ("neck", "Neck", None), ("leftArm", "Left Arm", None),
    ("rightArm", "Right Arm", None), ("leftHand", "Left Hand", None),
    ("rightHand", "Right Hand", None), ("camHead", "Head Cam", "sensors"),
    ("camFront", "Front Cam", "sensors"), ("camWristLeft", "L Wrist Cam", "sensors"),
    ("camWristRight", "R Wrist Cam", "sensors"), ("lidar", "LiDAR", "sensors"),
]
CAM_IDS = {"main": "camHead", "front": "camFront",
           "wrist_left": "camWristLeft", "wrist_right": "camWristRight"}

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} {detail}")
    if not ok:
        fails.append(name)
    return ok


async def read_modules(host, port, timeout=20.0, want_change=False):
    """Return (modules, n_state_frames, n_module_frames) over one connection."""
    url = f"ws://{host}:{port}/teleop?client=exp_modules"
    tok = os.getenv("MABEL_TOKEN", "").strip()
    if tok:
        url += f"&key={tok}"
    mods, n_state, n_mod = None, 0, 0
    t0 = time.perf_counter()
    async with websockets.connect(url, ping_interval=None, open_timeout=8) as ws:
        while time.perf_counter() - t0 < timeout:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if isinstance(raw, bytes):
                continue
            m = json.loads(raw)
            if m.get("type") != "robot_state":
                continue
            n_state += 1
            p = m.get("payload") or {}
            if "modules" in p:
                mods = p["modules"]
                n_mod += 1
            # Enough frames to judge the change-gate, and we have a roster.
            if mods is not None and n_state > 150:
                break
    return mods, n_state, n_mod


def show(mods):
    print("\n  ── roster on the wire ──")
    for m in mods:
        grp = f"[{m['group']}]" if m.get("group") else "     "
        print(f"     {'●' if m['ok'] else '○'} {m['label']:<13} {grp:<10} {m.get('detail','')}")
    print()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--expect-cams", default="main,front,wrist_left,wrist_right",
                    help="camera stream names that SHOULD be connected")
    ap.add_argument("--kill-cameras", action="store_true",
                    help="E5: kill sim_camera_server and prove the cams go dark")
    a = ap.parse_args()
    want_cams = {c for c in a.expect_cams.split(",") if c}

    print(f"exp_modules — ws://{a.host}:{a.port}/teleop")
    mods, n_state, n_mod = await read_modules(a.host, a.port)
    if mods is None:
        check("E1 robot_state carries `modules`", False,
              f"no modules block in {n_state} state frames")
        return 1
    show(mods)

    # ── E1 contract ──────────────────────────────────────────────────────────
    got = [(m["id"], m["label"], m.get("group")) for m in mods]
    check("E1 all 13 modules, in order, ids+labels+groups", got == EXPECT,
          f"{len(got)} modules" if got != EXPECT else "exact match")

    by = {m["id"]: m for m in mods}

    # ── E2 actuators ─────────────────────────────────────────────────────────
    act = ["base", "torso", "lift", "neck", "leftArm", "rightArm",
           "leftHand", "rightHand"]
    down = [m for m in act if not by[m]["ok"]]
    check("E2 every actuator module connected (sim)", not down,
          "all 8 up" if not down else f"DOWN: {down}")
    check("E2 arms report 7 joints each",
          "7 joint" in by["leftArm"]["detail"] and "7 joint" in by["rightArm"]["detail"],
          f"L='{by['leftArm']['detail']}' R='{by['rightArm']['detail']}'")
    check("E2 hands report 17 joints each",
          "17 joint" in by["leftHand"]["detail"] and "17 joint" in by["rightHand"]["detail"],
          f"L='{by['leftHand']['detail']}' R='{by['rightHand']['detail']}'")

    # ── E3 cameras match what is actually served ─────────────────────────────
    for cam, mid in CAM_IDS.items():
        want = cam in want_cams
        got_ok = by[mid]["ok"]
        check(f"E3 {mid} {'connected' if want else 'DARK (not served)'}",
              got_ok == want, f"ok={got_ok} detail='{by[mid]['detail']}'")

    # ── E4 lidar ─────────────────────────────────────────────────────────────
    check("E4 lidar connected while sweeps flow", by["lidar"]["ok"],
          by["lidar"]["detail"])

    # ── E6 change-gated ──────────────────────────────────────────────────────
    check("E6 roster is change-gated (not re-sent every frame)",
          n_mod < max(4, n_state // 10),
          f"{n_mod} module blocks in {n_state} state frames")

    # ── E5 liveness: kill the cameras, the roster MUST notice ────────────────
    if a.kill_cameras:
        print("\n  ── E5  killing sim_camera_server …")
        subprocess.run(["pkill", "-f", "sim_camera_server"], check=False)
        await asyncio.sleep(6.0)          # > TTL (3 s) + poll period (1 s)
        mods2, _, _ = await read_modules(a.host, a.port, timeout=15)
        if mods2 is None:
            check("E5 roster still arrives after the camera server dies", False, "")
        else:
            show(mods2)
            by2 = {m["id"]: m for m in mods2}
            dark = [m for m in CAM_IDS.values() if not by2[m]["ok"]]
            check("E5 ALL cameras flip to DOWN when the server dies",
                  len(dark) == 4, f"{len(dark)}/4 dark — {by2['camHead']['detail']!r}")
            check("E5 actuators + lidar UNAFFECTED (no false alarm)",
                  all(by2[m]["ok"] for m in act) and by2["lidar"]["ok"],
                  "still up")

    print("\n" + "=" * 62)
    print(f"  {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}")
    return 0 if not fails else 1


sys.exit(asyncio.run(main()))
