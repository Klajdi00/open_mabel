"""
Reproducible Data Studio benchmark — measures the read/serve path against the
real sample episodes and regenerates the doc figure. No hand-entered numbers.

    cd robot_learning/imitation_learning
    python -m data_studio.server.bench_studio --dataset studio_sample

Reports, per episode: dense-bundle size + decode-relevant breakdown, and the
wall time to produce each payload (dense bundle, one video frame, one lidar
frame, the accumulated scene, and a single materialize round-trip). Writes
docs/figures/studio_bench.pdf.
"""
import argparse
import os
import time

import numpy as np

from . import catalog, episode, scene, video, edl as EDL, materialize

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")


def _timeit(fn, n=5):
    best = []
    for _ in range(n):
        t = time.perf_counter(); fn(); best.append((time.perf_counter() - t) * 1e3)
    return float(np.mean(best)), float(np.std(best))


def run(dataset):
    eps = catalog.list_episodes(dataset)
    if not eps:
        print(f"no episodes in {dataset}"); return
    rows = []
    for ep in eps:
        p = catalog.episode_path(dataset, ep["file"])
        cams = video.cameras(p)
        cam0 = cams[0]["name"] if cams else None
        bundle = episode.dense_bundle(p)
        m_bundle = _timeit(lambda: episode.dense_bundle(p))
        m_frame = _timeit(lambda: episode.frame_state(p, ep["num_frames"] // 2))
        m_jpg = _timeit(lambda: video.frame_jpeg(p, cam0, ep["num_frames"] // 2)) if cam0 else (0, 0)
        m_lidar = _timeit(lambda: episode.lidar_bin(p))
        m_scene = _timeit(lambda: scene.accumulated_scene(p), n=2)
        rows.append({
            "ep": ep["file"], "frames": ep["num_frames"], "size_mb": ep["size_bytes"] / 1e6,
            "bundle_kb": len(bundle) / 1e3,
            "t_bundle": m_bundle[0], "t_frame": m_frame[0], "t_jpg": m_jpg[0],
            "t_lidar": m_lidar[0], "t_scene": m_scene[0],
        })

    # materialize round-trip: trim+concat first two episodes
    t_mat = mat_frames = None
    if len(eps) >= 2:
        e = {"fps": eps[0]["fps"] or 15, "segments": [
            {"dataset": dataset, "episode": eps[0]["file"], "start": 0, "end": eps[0]["num_frames"] // 2},
            {"dataset": dataset, "episode": eps[1]["file"], "start": 0, "end": eps[1]["num_frames"] // 2}]}
        t = time.perf_counter()
        res = materialize.materialize(e, "_bench_studio", task="bench")
        t_mat = (time.perf_counter() - t) * 1e3; mat_frames = res["num_frames"]
        import shutil
        shutil.rmtree(os.path.join(catalog.CFG.DATASET_ROOT, "_bench_studio"), ignore_errors=True)

    # table
    print(f"\n=== Data Studio serve benchmark · dataset '{dataset}' ===")
    hdr = ["episode", "frames", "h5 MB", "bundle KB", "bundle ms", "frame ms", "jpg ms", "lidar ms", "scene ms"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print("  ".join([f"{r['ep']:>10}", f"{r['frames']:>10}", f"{r['size_mb']:>10.1f}",
              f"{r['bundle_kb']:>10.1f}", f"{r['t_bundle']:>10.1f}", f"{r['t_frame']:>10.2f}",
              f"{r['t_jpg']:>10.2f}", f"{r['t_lidar']:>10.1f}", f"{r['t_scene']:>10.1f}"]))
    if t_mat is not None:
        print(f"\n  materialize (trim+concat -> {mat_frames} frames): {t_mat:.0f} ms")

    _plot(rows, t_mat, mat_frames)
    return rows


def _plot(rows, t_mat, mat_frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    labels = ["dense\nbundle", "1 frame\nstate", "1 video\nframe", "lidar\nblock", "accum.\nscene"]
    means = [np.mean([r[k] for r in rows]) for k in ("t_bundle", "t_frame", "t_jpg", "t_lidar", "t_scene")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(labels, means, color="#C25B2A")
    ax1.set_ylabel("serve time (ms, mean)"); ax1.set_title("Per-payload serve time")
    ax1.grid(axis="y", alpha=0.3)
    fr = [r["frames"] for r in rows]; bk = [r["bundle_kb"] for r in rows]
    ax2.scatter(fr, bk, color="#8B5A3C")
    ax2.set_xlabel("episode frames"); ax2.set_ylabel("dense bundle (KB)")
    ax2.set_title("Dense numeric payload scales linearly"); ax2.grid(alpha=0.3)
    fig.suptitle(f"MABEL Data Studio — serve benchmark (materialize {mat_frames}f in {t_mat:.0f} ms)"
                 if t_mat else "MABEL Data Studio — serve benchmark")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "studio_bench.pdf")
    fig.savefig(out); plt.close(fig)
    print(f"\n[bench] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="studio_sample")
    main_args = ap.parse_args()
    run(main_args.dataset)


if __name__ == "__main__":
    main()
