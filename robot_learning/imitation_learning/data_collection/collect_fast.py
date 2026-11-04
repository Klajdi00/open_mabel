"""
collect_fast.py — the FAST path: drive the MABEL sim and write Data-Studio
episodes (``episode_XXXX.h5``) *directly*, with no ROS / rosbag round-trip.

The Studio only consumes ``.h5``, so the quickest way to fill a dataset is to log
straight to HDF5 (the native ``EpisodeRecorder`` format) instead of recording a
rosbag and re-processing it. This script reuses the exact same collection stack
as ``collect_sim.py`` / ``sampledata.py`` (MuJoCo home scene → shared
``TeleopController`` driven by synthetic-AVP motion → ``SimStateProvider`` →
``EpisodeRecorder``) but is tuned for throughput:

  • lean cameras: only the two wrist views (RGB, no depth);
  • a coarser physics step (default 0.004 s, ~2.4× fewer steps than the 500 Hz
    default) — benchmarked as the fastest step that stays numerically stable on
    the contact-heavy home scene;
  • background-thread HDF5 saves that overlap the next episode's compute.

This produces ~50 episodes in a few minutes. (The rosbag pipeline —
``collect_ros.py`` + the Studio's drag-and-drop importer — still exists for when
you specifically want ROS bags; this is just the fast route to a dataset.)

Run in the Studio env (mujoco): it has mujoco + h5py + cv2 + the mabel pkg.

    python collect_fast.py                          # 5 tasks × 10 eps -> robot_learning/processed_dataset/simulated
    python collect_fast.py --tasks 1 --episodes 2   # quick test
    python collect_fast.py --frames 90 --dt 0.004 --out /path/to/dataset_root

Episodes land at ``<out>/<task_slug>/episode_XXXX.h5`` (one dataset per task).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")          # headless GPU render

_THIS = os.path.dirname(os.path.abspath(__file__))
_IL = os.path.dirname(_THIS)
for _p in (_THIS, _IL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_collection import config as CFG           # noqa: E402 (bootstraps `mabel`)

DEFAULT_SCENE = os.path.abspath(os.path.join(
    _THIS, "..", "..", "..", "simulation", "mabel_mujoco", "scenes", "mabel_home.xml"))
DEFAULT_OUT = os.path.abspath(os.path.join(
    _THIS, "..", "..", "processed_dataset", "simulated"))

# Home-scene-appropriate task labels (text metadata; motion is scripted).
TASKS = [
    "pick the red mug",
    "place the bowl on the counter",
    "open the cabinet",
    "wipe the kitchen island",
    "hand over the bottle",
]
WRIST_CAMS = ("l_wrist_cam", "r_wrist_cam")


def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_") or "task"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--tasks", type=int, default=len(TASKS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--frames", type=int, default=90, help="log frames per episode")
    ap.add_argument("--dt", type=float, default=0.004,
                    help="physics timestep (s); 0.004 = fastest stable on home scene")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe each task folder first (don't resume numbering)")
    args = ap.parse_args()

    import mujoco
    from data_collection.providers import SimStateProvider
    from data_collection.recorder import EpisodeRecorder
    from mabel.teleop_controller import TeleopController
    from mabel import mock_avp

    # lean cameras BEFORE the provider builds renderers
    CFG.CAMERAS = [dict(c, depth=False) for c in CFG.CAMERAS if c["name"] in WRIST_CAMS]

    print(f"[collect_fast] scene: {args.scene}")
    t_build = time.time()
    model = mujoco.MjModel.from_xml_path(args.scene)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctrl = TeleopController(model, data)
    prov = SimStateProvider(model, data, ctrl)
    substeps = max(1, int(round((1.0 / CFG.LOG_FPS) / args.dt)))
    schema = prov.schema_meta()
    print(f"[collect_fast] built in {time.time()-t_build:.1f}s  nu={model.nu} "
          f"cams={[c['name'] for c in schema['cameras']]} dt={args.dt} substeps={substeps} "
          f"frames/ep={args.frames}")

    out_root = os.path.abspath(args.out)
    n_tasks = max(1, min(args.tasks, len(TASKS)))
    total = 0
    t0 = time.time()
    for ti in range(n_tasks):
        task = TASKS[ti]
        ds_dir = os.path.join(out_root, _slug(task))
        if args.fresh and os.path.isdir(ds_dir):
            import shutil; shutil.rmtree(ds_dir)
        rec = EpisodeRecorder(ds_dir, dict(schema, task=task, fps=CFG.LOG_FPS,
                                           scene=os.path.basename(args.scene)))
        for ei in range(args.episodes):
            mujoco.mj_resetData(model, data)
            mujoco.mj_step(model, data)
            phase = 7.3 * (ti * 13 + ei + 1)        # distinct trajectory per (task, ep)
            ep_t0 = time.time()
            rec.start()
            for fi in range(args.frames):
                fr = mock_avp.mock_frame(phase + fi / CFG.LOG_FPS)
                if fi == 0:
                    ctrl.calibrate(fr)
                for _ in range(substeps):
                    ctrl.step(fr, None, args.dt)
                    mujoco.mj_step(model, data)
                obs = prov.observation(record_images=True)
                act = prov.action()
                rec.append(obs, act, prov.sim_time(), time.time())
            path = rec.finish(task=task)
            total += 1
            print(f"[collect_fast] [{total}] {task!r} -> {os.path.basename(path)} "
                  f"({args.frames}f, {time.time()-ep_t0:.1f}s)", flush=True)
        rec.close()                                 # flush this task's save queue
    print(f"[collect_fast] DONE: {total} episodes in {time.time()-t0:.1f}s -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
