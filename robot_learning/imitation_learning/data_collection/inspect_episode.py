"""
Inspect / replay a recorded MABEL episode HDF5.

Two modes:
  --summary   print the schema, every dataset's shape/dtype, and timing stats
              (mean/std logging rate) — a quick integrity check (default).
  --play      synchronized video playback of all RGB camera streams tiled in a
              grid, with a frame/time HUD (q quits, space pauses).

Usage
  python inspect_episode.py datasets/pick_cube_v1/episode_0000.h5
  python inspect_episode.py datasets/pick_cube_v1/episode_0000.h5 --play --fps 15
"""
import argparse
import json

import h5py
import numpy as np


def summarize(path):
    with h5py.File(path, "r") as f:
        print(f"\n=== {path} ===")
        for k in ("episode_index", "num_frames", "created", "task"):
            if k in f.attrs:
                print(f"  {k}: {f.attrs[k]}")
        if "schema_json" in f.attrs:
            schema = json.loads(f.attrs["schema_json"])
            print(f"  source: {schema.get('source')}  "
                  f"cameras: {[c['name'] for c in schema.get('cameras', [])]}")

        print("\n  datasets:")
        keys = sorted(f.keys())

        def _walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                shp = obj.shape
                extra = ""
                if name.endswith("/rgb"):
                    extra = " (JPEG blobs)"
                print(f"    {name:38s} {str(shp):14s} {obj.dtype}{extra}")
        f.visititems(_walk)

        # timing
        if "time/sim" in f:
            ts = f["time/sim"][:]
            tw = f["time/wall"][:]
            if len(ts) > 1:
                dt = np.diff(tw)
                print(f"\n  duration: {ts[-1]-ts[0]:.2f}s sim / {tw[-1]-tw[0]:.2f}s wall")
                print(f"  log rate: {1.0/dt.mean():.1f} Hz (mean) "
                      f"min {1.0/dt.max():.1f} / max {1.0/dt.min():.1f}")
        keys  # noqa


def play(path, fps):
    import cv2
    with h5py.File(path, "r") as f:
        cam_keys = sorted(k for k in _all_keys(f) if k.endswith("/rgb"))
        if not cam_keys:
            print("no RGB streams to play."); return
        streams = {k: f[k][:] for k in cam_keys}
        T = min(len(v) for v in streams.values())
        names = [k.split("/")[2] for k in cam_keys]
        print(f"playing {T} frames from {len(cam_keys)} cameras: {names}")

        delay = int(1000.0 / fps)
        t, paused = 0, False
        while t < T:
            if not paused:
                imgs = []
                for k in cam_keys:
                    img = cv2.imdecode(np.frombuffer(streams[k][t], np.uint8), cv2.IMREAD_COLOR)
                    cv2.putText(img, k.split("/")[2], (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    imgs.append(img)
                grid = _tile(imgs)
                cv2.putText(grid, f"frame {t}/{T}", (8, grid.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("MABEL episode", grid)
                t += 1
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        cv2.destroyAllWindows()


def _all_keys(f):
    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return out


def _tile(imgs):
    import cv2
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = imgs[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--play", action="store_true", help="video playback instead of summary")
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    if args.play:
        play(args.file, args.fps)
    else:
        summarize(args.file)


if __name__ == "__main__":
    main()
