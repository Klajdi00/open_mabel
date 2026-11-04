"""
MABEL Vision Pro teleop data collection — REAL ROBOT.

Same episode schema and operator workflow as ``collect_sim.py``; only the state
source changes. On hardware the operator still teleops through the shared
``TeleopController`` (Vision Pro -> retarget -> whole-body IK), and we log the
identical key layout via ``RealStateProvider`` so a policy can move between sim
and real without remapping its observation/action spaces.

This file is intentionally a thin, well-marked harness: MABEL's real I/O drivers
(motor bus, ZED + RealSense cameras, RPLIDAR A3, IMU) are not in this repo, so
fill in the ``# TODO`` hooks below and inside ``providers.RealStateProvider``.

Keys are identical to collect_sim.py:
  SPACE start/stop+save · X discard · R reset · C calibrate · M mode · Q quit
"""
import argparse
import os
import sys
import time

from pynput import keyboard

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from data_collection import config as CFG          # noqa: E402
from data_collection.providers import RealStateProvider  # noqa: E402
from data_collection.recorder import EpisodeRecorder      # noqa: E402

from mabel import config as C                       # noqa: E402
from mabel.teleop_controller import TeleopController  # noqa: E402
from mabel.avp_receiver import AVPReceiver           # noqa: E402

_events = {"calibrate": False, "mode": False, "rec_toggle": False,
           "discard": False, "reset": False, "quit": False}


def on_press(key):
    try:
        ch = key.char.lower()
        if ch == 'c': _events["calibrate"] = True
        elif ch == 'm': _events["mode"] = True
        elif ch == 'x': _events["discard"] = True
        elif ch == 'r': _events["reset"] = True
        elif ch == 'q': _events["quit"] = True
    except AttributeError:
        if key == keyboard.Key.space: _events["rec_toggle"] = True
        elif key == keyboard.Key.esc: _events["quit"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="dataset folder name under datasets/")
    ap.add_argument("--task", default="", help="free-text task label stored per episode")
    ap.add_argument("--addr", default=C.AVP_ZMQ_ADDR, help="AVP ZMQ publisher address")
    ap.add_argument("--rate", type=float, default=CFG.LOG_FPS, help="logging rate (Hz)")
    args = ap.parse_args()

    keyboard.Listener(on_press=on_press).start()

    # ── TODO: bring up the real robot ─────────────────────────────────────────
    #   hw = MabelHardware(...)        # motor bus + cameras + lidar + IMU
    #   hw.enable(); hw.home()
    #   The TeleopController needs a kinematics model+state to solve IK. On real
    #   MABEL run the controller against a MuJoCo *shadow* model whose qpos you
    #   keep synced to the encoders each tick (mirror of teleop_vision_pro.py),
    #   then stream controller targets (data.ctrl) to the motors. Reuse that same
    #   model/data here and pass them in.
    hw = None
    controller = None  # = TeleopController(shadow_model, shadow_data)
    if controller is None:
        sys.exit("collect_real.py is a hardware harness: wire up `hw` + `controller` "
                 "and RealStateProvider (see the module + providers.py docstrings).")

    recv = AVPReceiver(args.addr)
    provider = RealStateProvider(controller=controller, hw=hw)
    dataset_dir = os.path.join(CFG.DATASET_ROOT, args.name)
    recorder = EpisodeRecorder(dataset_dir, provider.schema_meta())

    log_dt = 1.0 / args.rate
    next_log = time.time()
    print(f"📁 dataset: {dataset_dir}  (resuming at episode {recorder.ep_count})")
    print("🕹️  SPACE start/stop · X discard · R reset · C calibrate · Q quit")

    try:
        while not _events["quit"]:
            frame = recv.poll()

            if _events["calibrate"]:
                _events["calibrate"] = False; controller.calibrate(frame)
            if _events["mode"]:
                _events["mode"] = False; controller.toggle_mode()
            if _events["reset"]:
                _events["reset"] = False
                # TODO: hw.home() / move to a safe start pose
            if _events["discard"]:
                _events["discard"] = False
                if recorder.recording:
                    recorder.finish(discard=True); print("[rec] discarded.")
            if _events["rec_toggle"]:
                _events["rec_toggle"] = False
                if recorder.recording:
                    path = recorder.finish(task=args.task)
                    print(f"[rec] saved -> {os.path.basename(path) if path else 'empty'}")
                else:
                    recorder.start(); print(f"[rec] ● recording episode {recorder.ep_count}")

            # ── one teleop tick on hardware ──────────────────────────────────
            #   controller.step(frame, joystick, dt)
            #   hw.sync_state_into(shadow_data)          # encoders -> qpos/qvel
            #   hw.send_targets(shadow_data.ctrl)        # ctrl -> motor commands

            now = time.time()
            if recorder.recording and now >= next_log:
                obs = provider.observation(record_images=True)
                act = provider.action()
                recorder.append(obs, act, provider.sim_time(), now)
                next_log = max(next_log + log_dt, now)
            time.sleep(0.001)
    finally:
        print("\n[shutdown] flushing pending saves…")
        recorder.close()
        recv.close()
        # TODO: hw.disable()


if __name__ == "__main__":
    main()
