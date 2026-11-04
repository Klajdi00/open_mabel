#!/usr/bin/env python3
"""
exp_record_episode.py — end-to-end proof that an app-triggered recording lands
as a valid DATA CURATION STUDIO episode.

Acts as the Vision Pro app on the wire: selects a recording folder, starts a
recording, teleops the base for a few seconds (nav joystick sine), stops, then
validates the saved episode with the CURATION STUDIO'S OWN loader
(robot_learning/data_curation/server/catalog.py) — the definitive format check.

    server/start_teleop.sh
    python3 server/tools/exp_record_episode.py --folder smoke_test
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

URI = "ws://127.0.0.1:9090/teleop?client=visionpro"   # tag routes to .../vision_os
SEND_HZ = 60.0


def frame(seq, t):
    return json.dumps({"type": "teleop_frame", "payload": {
        "sequence": seq, "timestamp": time.time(), "mode": "Base Driving",
        "head": {"transform": {"matrix": [1,0,0,0, 0,1,0,1.5, 0,0,1,0, 0,0,0,1]},
                 "trackingState": "notAvailable"},
        "navJoystick": {"lx": 0.4 * math.sin(0.8 * t), "ly": 0.5 * math.sin(0.5 * t),
                        "rx": 0.0, "ry": 0.0},
    }})


async def record(folder, secs):
    status = {}
    async with websockets.connect(URI, max_size=None) as ws:
        stop = asyncio.Event()

        async def rx():
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if d.get("type") == "robot_state":
                    rec = d.get("payload", {}).get("recording")
                    if rec:
                        status.update(rec)

        rxt = asyncio.create_task(rx())
        await ws.send(json.dumps({"type": "record", "payload": {"select": folder}}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "record", "payload": {"active": True, "task": folder}}))
        t0 = time.time()
        seq = 0
        while time.time() - t0 < secs:
            seq += 1
            await ws.send(frame(seq, time.time() - t0))
            await asyncio.sleep(1.0 / SEND_HZ)
        await ws.send(json.dumps({"type": "record", "payload": {"active": False, "task": folder}}))
        await asyncio.sleep(2.0)                       # let the save worker flush
        stop.set(); await asyncio.sleep(0.2); rxt.cancel()
    return status


def validate(folder):
    root = os.path.join(_ROOT, "dataset", "simulation", "vision_os", folder)
    eps = sorted(glob.glob(os.path.join(root, "episode_*.h5")))
    metaf = os.path.join(root, "dataset_meta.json")
    print(f"\n  dataset dir   {root}")
    print(f"  episodes      {[os.path.basename(e) for e in eps]}")
    print(f"  dataset_meta  {'present' if os.path.exists(metaf) else 'MISSING'}")
    if not eps:
        sys.exit("  FAIL: no episode written")

    # THE definitive check: the curation studio's own catalog loader.
    sys.path.insert(0, os.path.join(_ROOT, "robot_learning", "data_curation"))
    os.environ.setdefault("MABEL_DATASET_ROOT",
                          os.path.join(_ROOT, "dataset", "simulation", "vision_os"))
    from server import catalog                                     # noqa: E402
    card = catalog._episode_card(eps[-1])
    print(f"\n  curation-studio episode card for {os.path.basename(eps[-1])}:")
    for k in ("num_frames", "duration_s", "fps", "task", "modalities", "cams"):
        if k in card:
            print(f"    {k:12s} {card[k]}")
    need = {"rgb", "joint"}
    have = set(card.get("modalities", []))
    ok = card.get("num_frames", 0) > 0 and need.issubset(have)
    print(f"\n  verdict: {'CURATION-READY ✓' if ok else 'FAIL ✗ (missing ' + str(need - have) + ')'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="smoke_test")
    ap.add_argument("--secs", type=float, default=6.0)
    args = ap.parse_args()
    st = asyncio.run(record(args.folder, args.secs))
    print(f"  wire recording status after stop: task={st.get('task')} "
          f"tasks={st.get('tasks')} counts={st.get('taskCounts')} dir={st.get('dir')}")
    sys.exit(0 if validate(args.folder) else 1)
