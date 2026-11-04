"""
Generate REAL-format sample episodes for the Data Studio, headlessly.

`datasets/` ships empty, so the Studio has nothing to show out of the box. This
reuses the *exact* headless collection stack that ``tools/bench_collection.py``
drives (MuJoCo scene -> TeleopController -> SimStateProvider -> EpisodeRecorder,
fed by the synthetic Vision Pro source) so the .h5 it writes is byte-for-byte the
same schema as production collection. Nothing about the format is faked.

Per-episode variety: each episode runs the mock-AVP motion from a different time
offset, so the operator trajectories differ episode-to-episode — which makes the
timeline editor's cut/merge/append demo meaningful instead of three identical
clips.

CLI (no server needed):
    python -m data_studio.server.sampledata --name studio_sample --episodes 3 --frames 120

Or call generate() from the server's /api/dev/generate-sample job.
"""
import argparse
import os
import sys
import time

# ── make the sibling data_collection package and the mabel controller importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDIO = os.path.dirname(_HERE)                       # data_studio/
_IL = os.path.dirname(_STUDIO)                         # imitation_learning/
if _IL not in sys.path:
    sys.path.insert(0, _IL)

import mujoco  # noqa: E402
from data_collection import config as CFG                   # noqa: E402  (also bootstraps `mabel`)
from data_collection.providers import SimStateProvider      # noqa: E402
from data_collection.recorder import EpisodeRecorder        # noqa: E402
from mabel.teleop_controller import TeleopController        # noqa: E402
from mabel import mock_avp                                  # noqa: E402

TASKS = [
    "pick the blue cube",
    "place cube on the shelf",
    "hand the bottle over",
    "open the drawer",
    "wipe the table",
]


def _build_sim():
    model = mujoco.MjModel.from_xml_path(CFG.SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctrl = TeleopController(model, data)
    prov = SimStateProvider(model, data, ctrl)
    return model, data, ctrl, prov


def generate(name="studio_sample", episodes=3, frames=120, substeps=None,
             progress=None):
    """Write `episodes` real-format .h5 files into datasets/<name>/.

    `progress(done_frames, total_frames, ep_index)` is an optional callback the
    server's job runner uses to stream a progress bar.
    """
    model, data, ctrl, prov = _build_sim()
    if substeps is None:
        # one log frame per 1/LOG_FPS s of sim, matching the real collector
        substeps = max(1, int(round((1.0 / CFG.LOG_FPS) / model.opt.timestep)))

    out_dir = os.path.join(CFG.DATASET_ROOT, name)
    rec = EpisodeRecorder(out_dir, prov.schema_meta())

    total = episodes * frames
    done = 0
    for ep in range(episodes):
        # reset to a clean state for each episode and use a per-episode time
        # offset so the mock trajectory differs between episodes.
        mujoco.mj_resetData(model, data)
        mujoco.mj_step(model, data)
        t_off = 7.3 * (ep + 1)            # arbitrary phase shift per episode
        rec.start()
        for i in range(frames):
            tm = t_off + i / CFG.LOG_FPS
            frame = mock_avp.mock_frame(tm)
            if i == 0:
                ctrl.calibrate(frame)
            for _ in range(substeps):
                ctrl.step(frame, None, model.opt.timestep)
                mujoco.mj_step(model, data)
            obs = prov.observation(record_images=True)
            act = prov.action()
            rec.append(obs, act, prov.sim_time(), time.time())
            done += 1
            if progress and (i % 10 == 0 or i == frames - 1):
                progress(done, total, ep)
        path = rec.finish(task=TASKS[ep % len(TASKS)])
        print(f"[sampledata] queued {os.path.basename(path) if path else '?'} "
              f"({frames} frames, task={TASKS[ep % len(TASKS)]!r})")
    rec.close()                          # flush the background save thread
    print(f"[sampledata] wrote {episodes} episodes -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Generate sample MABEL episodes for the Data Studio.")
    ap.add_argument("--name", default="studio_sample")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--cameras", default=None,
                    help="comma list to restrict cameras (faster); default = all 7")
    args = ap.parse_args()
    if args.cameras:
        want = set(args.cameras.split(","))
        CFG.CAMERAS = [c for c in CFG.CAMERAS if c["name"] in want]
    t0 = time.time()
    generate(args.name, args.episodes, args.frames)
    print(f"[sampledata] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
