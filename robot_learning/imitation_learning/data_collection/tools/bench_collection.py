"""
Reproducible benchmark / figure generator for the MABEL data-collection pipeline.

Runs the collection stack headlessly with the synthetic (mock) Vision Pro source
— no headset, no GUI — so the whole pipeline (teleop controller -> full-state
observation render -> HDF5 episode) is exercised end to end and measured. It:

  1. records N frames into a temporary episode,
  2. reports a metrics table: achieved log rate, per-frame render time, episode
     size, and the byte breakdown per modality (RGB / depth / lidar / state),
  3. regenerates the diagnostic PDFs into docs/figures/ via visualize_dataset.

Everything the data_collection.tex doc cites is produced here, from scratch:

    cd data_collection && python tools/bench_collection.py --frames 60

Fix the seed for repeatability; nothing is hand-entered into the doc.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if os.path.dirname(_PKG) not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG))

import mujoco  # noqa: E402
from data_collection import config as CFG          # noqa: E402
from data_collection.providers import SimStateProvider  # noqa: E402
from data_collection.recorder import EpisodeRecorder      # noqa: E402
from mabel.teleop_controller import TeleopController      # noqa: E402
from mabel import mock_avp                                # noqa: E402


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def modality_bytes(path):
    """Sum stored bytes per modality group inside the HDF5 file. Variable-length
    RGB (JPEG) blobs live in the file's global heap, which get_storage_size()
    does not report, so they are summed from the blob lengths directly."""
    groups = {"rgb": 0, "depth": 0, "lidar": 0, "joint/state": 0,
              "pose/twist": 0, "teleop": 0, "time": 0}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.endswith("/rgb"):
                groups["rgb"] += int(sum(len(b) for b in obj[:]))
                return
            nb = obj.id.get_storage_size()
            if name.endswith("/depth"):         groups["depth"] += nb
            elif name == "obs/lidar":           groups["lidar"] += nb
            elif name.startswith("obs/joint") or name.startswith("obs/qpos") \
                    or name.startswith("obs/qvel") or name == "action/ctrl":
                groups["joint/state"] += nb
            elif name.startswith("obs/pose") or name == "obs/base_twist":
                groups["pose/twist"] += nb
            elif name.startswith("teleop"):     groups["teleop"] += nb
            elif name.startswith("time"):       groups["time"] += nb
        f.visititems(visit)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--substeps", type=int, default=33, help="physics steps per log frame (~LOG_FPS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the benchmark episode")
    ap.add_argument("--cameras", default="", help="comma-separated camera subset (default: all)")
    args = ap.parse_args()
    np.random.seed(args.seed)

    if args.cameras:
        want = set(args.cameras.split(","))
        CFG.CAMERAS = [c for c in CFG.CAMERAS if c["name"] in want]

    model = mujoco.MjModel.from_xml_path(CFG.SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctrl = TeleopController(model, data)
    prov = SimStateProvider(model, data, ctrl)

    out_dir = os.path.join(CFG.DATASET_ROOT, "_bench")
    rec = EpisodeRecorder(out_dir, prov.schema_meta())
    # start numbering fresh for a clean benchmark file
    rec.ep_count = 0

    t0 = time.time()
    rec.start()
    render_ms, append_ms = [], []
    for i in range(args.frames):
        frame = mock_avp.mock_frame(time.time() - t0)
        if i == 0:
            ctrl.calibrate(frame)
        for _ in range(args.substeps):
            ctrl.step(frame, None, model.opt.timestep)
            mujoco.mj_step(model, data)
        ta = time.time()
        obs = prov.observation(record_images=True)
        act = prov.action()
        render_ms.append((time.time() - ta) * 1e3)
        tb = time.time()
        rec.append(obs, act, prov.sim_time(), time.time())
        append_ms.append((time.time() - tb) * 1e3)
    wall = time.time() - t0
    path = rec.finish(task="benchmark")
    rec.close()

    size = os.path.getsize(path)
    mods = modality_bytes(path)
    n_cam_rgb = sum(c["rgb"] for c in CFG.CAMERAS)
    n_cam_depth = sum(c["depth"] for c in CFG.CAMERAS)

    print("\n" + "=" * 62)
    print(" MABEL data-collection benchmark (mock AVP, headless)")
    print("=" * 62)
    print(f" frames logged          : {args.frames}")
    print(f" cameras (rgb / depth)  : {n_cam_rgb} / {n_cam_depth} @ {CFG.RGB_W}x{CFG.RGB_H}")
    print(f" wall time              : {wall:.2f} s")
    print(f" per-frame render+encode: {np.mean(render_ms):.1f} +- {np.std(render_ms):.1f} ms")
    print(f" per-frame append       : {np.mean(append_ms):.2f} +- {np.std(append_ms):.2f} ms")
    print(f" sustainable log rate   : {1000.0/np.mean(render_ms):.1f} Hz (render-bound)")
    print(f" episode size           : {human(size)}  ({human(size/args.frames)}/frame)")
    print(" per-modality stored bytes:")
    for k, v in sorted(mods.items(), key=lambda kv: -kv[1]):
        print(f"   {k:14s} {human(v):>10s}  ({100*v/size:4.1f}%)")
    print("=" * 62)

    # regenerate the doc figures from this episode
    import subprocess
    viz = os.path.join(_PKG, "visualize_dataset.py")
    subprocess.run([sys.executable, viz, path], check=False)
    # rename figures to a stable stem the doc references
    fig_dir = os.path.join(_PKG, "docs", "figures")
    stem = os.path.splitext(os.path.basename(path))[0]
    for kind in ("joints", "ee", "base"):
        src = os.path.join(fig_dir, f"{stem}_{kind}.pdf")
        dst = os.path.join(fig_dir, f"bench_{kind}.pdf")
        if os.path.exists(src):
            os.replace(src, dst)
    print(f"figures -> {fig_dir}/bench_{{joints,ee,base}}.pdf")

    if not args.keep:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
