"""
MABEL Vision Pro teleop data collection — SIMULATION.

Drives the *same* whole-body TeleopController used by the live sim viewer
(simulation/.../teleop_vision_pro.py), and logs a full multimodal episode every
frame: all on-board cameras (RGB + depth), 360-ray LiDAR, every joint's
position/velocity/torque, the base twist, end-effector + head poses, AND the
target command for every actuator (data.ctrl) plus the operator's teleop intent.

The operator decides when to cut episodes: SPACE starts/stops-and-saves an
episode, X discards the in-progress one, R resets + re-randomizes the manipuland.

Usage
  # 1. Start the Vision Pro publisher (in dexterous_hand/vision_pro):
  #      python connect_full_avp.py --room <ROOM_ID>
  # 2. Then collect:
  python collect_sim.py --name pick_cube_v1            # live AVP stream
  python collect_sim.py --name pick_cube_v1 --mock     # synthetic source, no headset
  python collect_sim.py --name pick_cube_v1 --addr tcp://192.168.1.50:5556

Keys (focus is global via pynput, so the viewer window need not be focused):
  C      (re)calibrate the teleop clutch at the operator's current pose
  M      toggle MANIPULATION <-> NAVIGATION
  B      toggle lazy base-follow on/off
  SPACE  start a new episode  /  stop + save the current one
  X      discard the in-progress episode (no save)
  R      reset robot to home + re-randomize the manipuland
  Q/ESC  quit (auto-discards an unsaved in-progress episode)
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from data_collection import config as CFG
from data_collection.providers import SimStateProvider
from data_collection.recorder import EpisodeRecorder

from mabel import config as C
from mabel.teleop_controller import TeleopController
from mabel.avp_receiver import AVPReceiver
from mabel import mock_avp


_events = {"calibrate": False, "mode": False, "base": True,
           "rec_toggle": False, "discard": False, "reset": False, "quit": False}


def on_press(key):
    try:
        ch = key.char.lower()
        if ch == 'c': _events["calibrate"] = True
        elif ch == 'm': _events["mode"] = True
        elif ch == 'b': _events["base"] = not _events["base"]
        elif ch == 'x': _events["discard"] = True
        elif ch == 'r': _events["reset"] = True
        elif ch == 'q': _events["quit"] = True
    except AttributeError:
        if key == keyboard.Key.space: _events["rec_toggle"] = True
        elif key == keyboard.Key.esc: _events["quit"] = True


def reset_scene(model, data, controller, home_qpos, randomize=True):
    """Restore the home configuration and (optionally) re-randomize the manipuland
    so each episode starts from a fresh object pose."""
    data.qpos[:] = home_qpos
    data.qvel[:] = 0.0
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    if randomize and jid >= 0:
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 3] = [np.random.uniform(0.52, 0.82),
                                  np.random.uniform(-0.30, 0.30), 0.75]
        theta = np.random.uniform(0, 2 * np.pi)
        data.qpos[adr + 3:adr + 7] = [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
    mujoco.mj_forward(model, data)
    controller.hold_current()        # actuator targets <- fresh pose (no snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset folder name under datasets/")
    ap.add_argument("--task", default="", help="free-text task label stored per episode")
    ap.add_argument("--mock", action="store_true", help="use a synthetic AVP source")
    ap.add_argument("--addr", default=C.AVP_ZMQ_ADDR, help="AVP ZMQ publisher address")
    ap.add_argument("--no-randomize", action="store_true", help="fixed manipuland pose")
    args = ap.parse_args()

    keyboard.Listener(on_press=on_press).start()

    model = mujoco.MjModel.from_xml_path(CFG.SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    home_qpos = data.qpos.copy()

    controller = TeleopController(model, data, base_follow=_events["base"])
    provider = SimStateProvider(model, data, controller)
    dataset_dir = os.path.join(CFG.DATASET_ROOT, args.name)
    recorder = EpisodeRecorder(dataset_dir, provider.schema_meta())
    recv = None if args.mock else AVPReceiver(args.addr)

    log_dt = 1.0 / CFG.LOG_FPS
    next_log = time.time()
    src = "MOCK" if args.mock else args.addr
    print(f"📁 dataset: {dataset_dir}  (resuming at episode {recorder.ep_count})")
    print(f"🎥 cameras: {[c['name'] for c, _ in provider._cams]}  | log {CFG.LOG_FPS:.0f} Hz")
    print(f"🕹️  source: {src}   — SPACE start/stop, X discard, R reset, C calibrate, Q quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and not _events["quit"]:
            frame = mock_avp.mock_frame(time.time() - t0) if args.mock else recv.poll()

            # ── operator events ──────────────────────────────────────────────
            if _events["calibrate"]:
                _events["calibrate"] = False; controller.calibrate(frame)
            if _events["mode"]:
                _events["mode"] = False; controller.toggle_mode()
            controller.base_follow = _events["base"]
            if _events["reset"]:
                _events["reset"] = False
                reset_scene(model, data, controller, home_qpos, not args.no_randomize)
            if _events["discard"]:
                _events["discard"] = False
                if recorder.recording:
                    recorder.finish(discard=True); print("[rec] discarded.")
            if _events["rec_toggle"]:
                _events["rec_toggle"] = False
                if recorder.recording:
                    path = recorder.finish(task=args.task)
                    print(f"[rec] saved episode -> {os.path.basename(path) if path else 'empty'}")
                else:
                    reset_scene(model, data, controller, home_qpos, not args.no_randomize)
                    recorder.start(); print(f"[rec] ● recording episode {recorder.ep_count}")

            # ── physics catch-up (same scheme as teleop_vision_pro.py) ───────
            target = time.time() - t0
            if target - data.time > 0.1:
                t0 = time.time() - data.time; target = data.time
            joystick = None
            while data.time < target:
                controller.step(frame, joystick, model.opt.timestep)
                mujoco.mj_step(model, data)

            # ── log a frame at LOG_FPS ───────────────────────────────────────
            now = time.time()
            if recorder.recording and now >= next_log:
                obs = provider.observation(record_images=True)
                act = provider.action()
                recorder.append(obs, act, provider.sim_time(), now)
                next_log = max(next_log + log_dt, now)   # don't accumulate backlog

            viewer.sync()
            time.sleep(0.002)

    print("\n[shutdown] flushing pending saves…")
    recorder.close()
    if recv is not None:
        recv.close()
    print("done.")


if __name__ == "__main__":
    main()
