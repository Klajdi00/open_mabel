#!/usr/bin/env python3
"""
view_robot.py — live desktop view of the teleoperated MABEL robot.

Opens a native MuJoCo viewer on the Mac and mirrors the robot that
sim_teleop_bridge.py is driving, by reading the qpos snapshot the bridge
publishes to a shared memory-mapped file. This is a *separate process* on
purpose: on macOS the native viewer (main-thread Cocoa, needs `mjpython`) and
the bridge's off-screen camera renderer cannot share one process's GL context,
so we keep them apart and sync through the memmap.

Usage
-----
  # terminal 1 — the teleop bridge (publishes state by default):
  python sim_teleop_bridge.py --scene ../simulation/mabel_mujoco/scenes/mabel_home.xml

  # terminal 2 — this viewer (macOS REQUIRES mjpython for the native window):
  mjpython view_robot.py
  # Linux:  python3 view_robot.py

It auto-loads the same scene the bridge did (recorded in <state-file>.scene).
Override with --scene if you ran the bridge with a non-default scene and want
to force a match.

The viewer never steps physics; it only copies qpos and calls mj_forward, so
what you see is exactly the robot state the bridge is simulating. BUT a Ctrl+drag
(double-click a body to select, then Ctrl+right-drag to push / Ctrl+left-drag to
twist) is forwarded back to the bridge through a second memmap: the bridge applies
it as an external wrench, so the arm yields in compliance ("Soft") and springs
back in impedance ("Hold"). Kinesthetic guiding, straight from the desktop window.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer


def _resolve_scene(state_file: str, scene_arg: str | None) -> str:
    if scene_arg:
        return scene_arg if os.path.isabs(scene_arg) else os.path.abspath(scene_arg)
    sidecar = state_file + ".scene"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            path = f.readline().strip()
        if path:
            return path
    sys.exit(f"No scene recorded at {sidecar}. Start the bridge first, or pass --scene.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live desktop viewer for MABEL teleop")
    ap.add_argument("--state-file", default="/tmp/mabel_teleop_state.f64",
                    help="memmap path written by sim_teleop_bridge.py")
    ap.add_argument("--scene", default=None,
                    help="force a scene (default: the one recorded by the bridge)")
    args = ap.parse_args()

    scene = _resolve_scene(args.state_file, args.scene)
    print(f"[viewer] scene: {scene}")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    nq = model.nq

    # Wait for the bridge's state file (it may start a moment later).
    expected = (1 + nq) * 8  # float64
    print(f"[viewer] waiting for state file {args.state_file} …")
    for _ in range(120):
        if os.path.exists(args.state_file) and os.path.getsize(args.state_file) >= expected:
            break
        time.sleep(0.5)
    if not (os.path.exists(args.state_file) and os.path.getsize(args.state_file) >= expected):
        sys.exit("[viewer] state file never appeared / wrong size — is the bridge running "
                 "with a matching --scene?")

    state = np.memmap(args.state_file, dtype=np.float64, mode="r", shape=(1 + nq,))

    # Perturbation channel back to the bridge: the bridge creates <state>.pert
    # (nbody×6 xfrc); we WRITE the Ctrl+drag force there each frame and it applies
    # it to the live sim. Best-effort — if it's missing, the viewer is read-only.
    pert_path = args.state_file + ".pert"
    pert = None
    if os.path.exists(pert_path) and os.path.getsize(pert_path) >= model.nbody * 6 * 8:
        try:
            pert = np.memmap(pert_path, dtype=np.float64, mode="r+", shape=(model.nbody, 6))
        except Exception as e:
            print(f"[viewer] perturbation channel unavailable ({e}); drag is read-only")
    print("[viewer] attached — mirroring the robot."
          + ("  Ctrl+drag a body to push it." if pert is not None else ""))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show the ROBOT ONLY — no sensor decoration. Three separate switches,
        # because MuJoCo's defaults hide some of this and show the rest:
        #   * geomgroup[3]: the cyan camera-marker boxes + orange lens cylinders
        #     (mabel_full.xml:375-378,474,569,761) and the front IMU site. Default
        #     is OFF; we used to force it ON here to show where the cams point.
        #   * sitegroup[3]: the 360 replicated `lidar_ray` sites. Default OFF
        #     already — pinned so a future default change can't leak them back.
        #   * mjVIS_RANGEFINDER: MuJoCo defaults this ON, and mabel_full.xml:1096
        #     declares a rangefinder per ray site, so a bare viewer draws all 360
        #     laser lines. This is the switch that actually kills the LiDAR fan.
        # Same suppression triple as every other renderer in the repo — see
        # sim_camera_server.py:492-494, cloud_streams.py:320-322 and
        # ros/src/mabel_sim/mabel_sim/mujoco_node.py:664-666.
        viewer.opt.geomgroup[3] = 0
        viewer.opt.sitegroup[3] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        # Open on a front three-quarter view: the robot's face, chest and both
        # arms all readable at once, looking slightly down so the base and the
        # floor around it stay in frame. Azimuth is relative to the robot, so
        # this framing carries across scenes. Override per-run with
        # MABEL_VIEWER_CAM="azimuth,elevation,distance,lookat_z".
        cam = os.getenv("MABEL_VIEWER_CAM", "40,-28,3.0,1.0").split(",")
        viewer.cam.azimuth = float(cam[0])
        viewer.cam.elevation = float(cam[1])
        viewer.cam.distance = float(cam[2])
        viewer.cam.lookat[:] = [0.0, 0.0, float(cam[3])]

        # This native window shares the GPU with sim_camera_server.py's off-screen
        # renders. Keep its footprint small: (1) the POSITION pipeline only
        # (kinematics+comPos+camlight) instead of mj_forward — pixel-identical for a
        # viewer but ~340× cheaper (no mass-matrix / collision / constraint solve a
        # viewer never needs); (2) skip the recompute when the robot pose is
        # unchanged; (3) cap to 30 fps so it leaves GPU headroom for the camera feeds.
        # (For lowest-latency web teleop, don't run --viewer at all.)
        # Every viewer.sync() is a full-scene GPU render that macOS serializes against
        # sim_camera_server.py's offscreen camera renders — so the desktop window steals
        # fps straight from the operator's feeds. Keep BOTH smooth with --viewer on:
        #   * RENDER-ON-CHANGE: re-render only when the robot pose actually moved (or the
        #     operator is dragging). A STATIC robot drops to a slow EVENT floor that just
        #     keeps the window/mouse alive, handing the WHOLE GPU back to the cameras.
        #   * CONFIGURABLE CAP: MABEL_VIEWER_FPS (default 20) bounds the viewer WHILE the
        #     robot is moving. Lower it (e.g. 10) for crisper camera feeds during teleop;
        #     raise it for a smoother desktop view when you're not streaming.
        # Default 12 fps (was 20): measured head+wrist camera feeds recover from
        # ~15 fps (p99 185 ms, choppy) at a 20 fps viewer to ~20 fps (p99 67 ms,
        # smooth) at 12 — and a 12 fps debug window is perfectly readable. Physics
        # stays real-time regardless (the App Nap guard in the bridge). Raise it
        # for a smoother desktop view when you're NOT streaming video.
        VIEW_FPS = float(os.getenv("MABEL_VIEWER_FPS", "12"))
        VIEW_DT = 1.0 / max(1.0, VIEW_FPS)
        EVENT_DT = 1.0 / 8.0                 # idle floor — window stays responsive
        last_t = -1.0
        last_sync = 0.0
        while viewer.is_running():
            t = float(state[0])
            moved = (t != last_t)
            if moved:
                data.qpos[:] = state[1:1 + nq]
                data.time = t
                mujoco.mj_kinematics(model, data)   # qpos → body/site xpos
                mujoco.mj_comPos(model, data)        # → xipos for perturb drag
                mujoco.mj_camlight(model, data)      # → correct lights/cameras
                last_t = t
            # Forward the operator's Ctrl+drag to the bridge as an external wrench.
            dragging = False
            if pert is not None:
                data.xfrc_applied[:] = 0.0
                if viewer.perturb.active and viewer.perturb.select > 0:
                    mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)
                    dragging = True
                pert[:] = data.xfrc_applied
            # Render at the cap while moving / dragging; otherwise only the slow event
            # floor — so a still robot never burns the GPU the camera feeds need.
            now = time.time()
            interval = VIEW_DT if (moved or dragging) else EVENT_DT
            if now - last_sync >= interval:
                viewer.sync()
                last_sync = now
            time.sleep(0.003)


if __name__ == "__main__":
    main()
