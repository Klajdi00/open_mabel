"""
Merge several MABEL episode folders into one, renumbering episodes contiguously.

Useful for pooling collection sessions (e.g. different operators / days) into a
single training set. Copies (default) or moves the .h5 files and rewrites the
``episode_XXXX.h5`` numbering; carries the first dataset_meta.json through.

Usage
  python merge_dataset.py --out datasets/pick_cube_all \
      datasets/pick_cube_v1 datasets/pick_cube_v2
  python merge_dataset.py --out datasets/all --move datasets/sess_*
"""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("sources", nargs="+", help="source dataset folders")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    op = shutil.move if args.move else shutil.copy2

    idx = 0
    for src in args.sources:
        files = sorted(glob.glob(os.path.join(src, "*.h5")))
        if not files:
            print(f"[merge] {src}: no episodes, skipping")
            continue
        meta = os.path.join(src, "dataset_meta.json")
        if idx == 0 and os.path.exists(meta):
            shutil.copy2(meta, os.path.join(args.out, "dataset_meta.json"))
        for fpath in files:
            dst = os.path.join(args.out, f"episode_{idx:04d}.h5")
            op(fpath, dst)
            idx += 1
        print(f"[merge] {src}: {len(files)} episodes")
    print(f"[merge] total {idx} episodes -> {args.out}")


if __name__ == "__main__":
    main()
