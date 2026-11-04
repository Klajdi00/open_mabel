"""curate_export.py — run the Data-Curation studio's quality engine over a set of
collected episodes and materialize a LeRobot v2.1 dataset from them.

This is the headless equivalent of what the studio (:8093) does interactively:
score each episode for learning-relevant defects (control-lag, frozen/timing/
no-torque/NaN), then export the corpus to LeRobot for the Training Studio.

Run from the `robot_learning/` dir (so `data_curation.server` imports resolve):
    ~/miniconda3/envs/mabel/bin/python3 curate_export.py \
        --sessions '/home/mabel/open_mabel/real/vision_os/pick_place_*' \
        --out-name pick_place --realm real
Output: MABEL/curated_dataset/<realm>/<out-name>_<NNN>_<YYYYMMDD>/  (LeRobot v2.1)
        + curation_report.json (quality scan of every source episode)
"""
from __future__ import annotations
import argparse, glob, json, os, re, time
from datetime import datetime

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from data_curation.server import quality, lerobot_export

_MABEL = os.path.dirname(os.path.abspath(__file__)).rsplit("/learning", 1)[0]


def _next_number(parent, task):
    pat = re.compile(rf"^{re.escape(task)}_(\d+)_")
    best = 0
    if os.path.isdir(parent):
        for n in os.listdir(parent):
            m = pat.match(n)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True,
                    help="glob of session folders OR .h5 files (e.g. '.../pick_place_*')")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--realm", default="real", choices=["real", "simulation"])
    ap.add_argument("--curated-root",
                    default=os.path.join(_MABEL, "curated_dataset"))
    args = ap.parse_args()

    # resolve h5 episode files
    hits = sorted(glob.glob(args.sessions))
    h5s = []
    for h in hits:
        if h.endswith(".h5"):
            h5s.append(h)
        elif os.path.isdir(h):
            h5s += sorted(glob.glob(os.path.join(h, "*.h5")))
    h5s = [h for h in h5s if os.path.getsize(h) > 0]
    if not h5s:
        raise SystemExit(f"no .h5 episodes matched {args.sessions!r}")
    print(f"found {len(h5s)} episodes")

    # ── 1) curation quality scan (studio's engine) ───────────────────────────
    scans, ok = [], 0
    for h in h5s:
        try:
            q = quality.analyze(h)
            score = q.get("quality_score")
            defects = q.get("defects", [])
            scans.append({"episode": os.path.relpath(h, _MABEL),
                          "frames": q.get("T"), "fps": q.get("fps"),
                          "quality_score": score, "frac_bad": q.get("frac_bad"),
                          "severity": q.get("severity"),
                          "n_defects": len(defects), "flags": q.get("flags")})
            ok += 1
            print(f"  scanned {os.path.basename(os.path.dirname(h))}: "
                  f"score={score} frac_bad={q.get('frac_bad')} defects={len(defects)}")
        except Exception as e:                            # never abort the corpus
            scans.append({"episode": os.path.relpath(h, _MABEL), "error": str(e)})
            print(f"  scan FAILED {h}: {e}")

    # ── 2) materialize LeRobot v2.1 ──────────────────────────────────────────
    parent = os.path.join(args.curated_root, args.realm)
    os.makedirs(parent, exist_ok=True)
    num = _next_number(parent, args.out_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(parent, f"{args.out_name}_{num:03d}_{stamp}")
    t0 = time.time()
    res = lerobot_export.to_lerobot_dataset(
        h5s, out_dir, repo_id=f"mabel/{args.out_name}")
    res["export_seconds"] = round(time.time() - t0, 1)
    print("LeRobot export:", json.dumps(res, indent=2))

    report = {"created": stamp, "realm": args.realm, "source_episodes": len(h5s),
              "quality_ok": ok, "lerobot": res, "scans": scans,
              "out_dir": out_dir}
    json.dump(report, open(os.path.join(out_dir, "curation_report.json"), "w"), indent=2)
    print("\nCURATED DATASET:", out_dir)
    print("report:", os.path.join(out_dir, "curation_report.json"))


if __name__ == "__main__":
    main()
