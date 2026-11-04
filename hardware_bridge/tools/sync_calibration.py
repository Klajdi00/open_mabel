#!/usr/bin/env python3
"""sync_calibration.py — pull the legacy calibration sources into
calibration_data/ (the bridge's read-first mirror).

The studios dual-write into calibration_data on every save, so normally this
tool has nothing to do. It exists for the fallback cases: a upstream checkout
whose calib server predates the MABEL_CALIB_MIRRORS hook, a hand calibrated by
an old GUI build, or a fresh robot checkout being brought up from legacy files.

    python3 tools/sync_calibration.py            # one shot: copy newer sources in
    python3 tools/sync_calibration.py --watch    # keep polling (2 s) and copying
    python3 tools/sync_calibration.py --check    # exit 1 if the mirror is stale

Only copies when the source is NEWER than the mirror (never clobbers a fresher
studio-written mirror with a stale legacy file).
"""
import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.dirname(_HERE)
sys.path.insert(0, _BRIDGE)

from mabel_hw import paths                                       # noqa: E402

MIRROR = os.path.join(_BRIDGE, "calibration_data")


def _pairs():
    root = paths.mabel_root()
    yield (os.path.join(root, "upstream", "MABEL", "modules", "calib",
                        "config", "mabel.json"),
           os.path.join(MIRROR, "mabel.json"))
    for side in ("left", "right"):
        yield (os.path.join(root, "firmware", "orca_hand", "feetech",
                            "orcahand", "configs", f"{side}_calibration.json"),
               os.path.join(MIRROR, "hands", f"{side}_calibration.json"))


def _mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None


def sync(verbose=True):
    copied = []
    for src, dst in _pairs():
        st, dt = _mtime(src), _mtime(dst)
        if st is None:
            continue                       # legacy source absent: nothing to pull
        if dt is None or st > dt + 1e-6:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            copied.append(dst)
            if verbose:
                print(f"synced {os.path.relpath(src, paths.mabel_root())} "
                      f"-> {os.path.relpath(dst, _BRIDGE)}")
    return copied


def check():
    stale = [dst for src, dst in _pairs()
             if _mtime(src) is not None
             and (_mtime(dst) is None or _mtime(src) > _mtime(dst) + 1e-6)]
    for d in stale:
        print(f"STALE mirror: {d}")
    return 1 if stale else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--watch", action="store_true",
                    help="poll every 2 s and keep the mirror fresh")
    ap.add_argument("--check", action="store_true",
                    help="report staleness only (exit 1 if stale)")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    if not args.watch:
        copied = sync()
        print(f"{len(copied)} file(s) updated." if copied else "mirror already fresh.")
        return 0
    print("watching legacy calibration sources (Ctrl-C to stop)...")
    try:
        while True:
            sync()
            time.sleep(2.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
