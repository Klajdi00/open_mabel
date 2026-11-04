#!/usr/bin/env python3
"""
exp_record_replay.py — ONE take is BOTH a training episode and a replayable clip.

The apps have a single Record flow (2026-10-14): one `record` message, and the
bridge saves the curation H5 AND a command clip named `<folder>_episode_XXXX`
(same index as the H5). This acts as an app on the wire and checks every link:

  1. record N takes into one folder (nav-stick sine while recording)
  2. each take produces  <root>/<folder>/episode_XXXX.h5   (curation data)
                    and  a clip `<folder>_episode_XXXX`     (in demo_list)
     with consecutive indices and matching names
  3. replaying the newest clip sets robot_state.animation.playing with the
     CLIP's name (not the last recorded one), and stopping it ends playback
  4. deleting the clips leaves the H5s alone

    server/start_teleop.sh --no-tui
    python3 server/tools/exp_record_replay.py --folder rr_smoke --takes 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

URI = "ws://127.0.0.1:9090/teleop?client=ios"


def env(t, p):
    return json.dumps({"type": t, "payload": p})


def nav(seq, t):
    return env("teleop_frame", {
        "sequence": seq, "timestamp": time.time(), "mode": "Base Driving",
        "head": {"transform": {"matrix": [1, 0, 0, 0, 0, 1, 0, 1.5, 0, 0, 1, 0, 0, 0, 0, 1]},
                 "trackingState": "notAvailable"},
        "navJoystick": {"lx": 0.3 * math.sin(0.8 * t), "ly": 0.0, "rx": 0.0, "ry": 0.0}})


async def main(folder, takes, secs):
    latest = {}
    clips_msg = {}
    async with websockets.connect(URI, max_size=None) as ws:
        done = asyncio.Event()

        async def rx():
            while not done.is_set():
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
                t, p = m.get("type"), m.get("payload") or {}
                if t == "robot_state":
                    latest.update(p)
                elif t == "demo_clips":
                    clips_msg["clips"] = p.get("clips", [])

        rtask = asyncio.create_task(rx())
        await asyncio.sleep(1.0)

        async def wait_for(pred, timeout, what):
            t0 = time.time()
            while time.time() - t0 < timeout:
                if pred():
                    return time.time() - t0
                await asyncio.sleep(0.05)
            raise SystemExit(f"FAIL: timed out waiting for {what}")

        rec = lambda: latest.get("recording") or {}
        anim = lambda: latest.get("animation") or {}

        await ws.send(env("record", {"select": folder}))
        await wait_for(lambda: rec().get("task") == folder, 5, "folder select")
        start_count = rec().get("episodeCount", 0)
        out_dir = rec().get("dir", "")
        print(f"folder={folder} dir={out_dir} existing={start_count}")

        rows = []
        for k in range(takes):
            await ws.send(env("record", {"active": True, "task": folder}))
            await wait_for(lambda: anim().get("recording"), 5, "clip capture to start")
            idx = rec().get("episodeIndex")
            name = anim().get("name")
            seq, t0 = 0, time.time()
            while time.time() - t0 < secs:
                await ws.send(nav(seq, time.time() - t0)); seq += 1
                await asyncio.sleep(1 / 30)
            await ws.send(env("record", {"active": False, "task": folder}))
            await wait_for(lambda: not anim().get("recording"), 5, "clip capture to stop")
            rows.append((k, idx, name))
            print(f"  take {k}: episodeIndex={idx} clip={name}")
            await asyncio.sleep(0.6)

        # Every take: consecutive index, clip named after it, H5 on disk.
        await asyncio.sleep(1.5)                      # H5 save is queued off-thread
        await ws.send(env("demo_list", {}))
        await wait_for(lambda: "clips" in clips_msg, 5, "demo_clips")
        names = {c["name"]: c for c in clips_msg["clips"]}
        ok = True
        for k, idx, name in rows:
            exp = f"{folder}_episode_{start_count + k:04d}"
            h5 = os.path.join(out_dir, folder, f"episode_{start_count + k:04d}.h5")
            c = names.get(exp)
            good = idx == start_count + k and name == exp and c is not None and os.path.isfile(h5)
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  {exp}: index={idx} clip_listed={c is not None}"
                  f" dur={c['duration'] if c else '-'}s frames={c['frames'] if c else '-'}"
                  f" h5={os.path.isfile(h5)}")

        # Replay the newest: the echo must carry the CLIP's name.
        newest = f"{folder}_episode_{start_count + takes - 1:04d}"
        first = f"{folder}_episode_{start_count:04d}"
        await ws.send(env("demo_replay", {"active": True, "name": first, "repeats": 1}))
        lat = await wait_for(lambda: anim().get("playing"), 5, "replay to start")
        playing_name = anim().get("name")
        good = playing_name == first
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  replay started in {lat*1000:.0f} ms, "
              f"echo name={playing_name} (want {first}; newest recorded was {newest})")
        await asyncio.sleep(1.0)
        await ws.send(env("demo_replay", {"active": False}))
        lat = await wait_for(lambda: not anim().get("playing"), 8, "replay to stop")
        print(f"  PASS  replay stopped (incl. ease-home) in {lat:.2f} s")

        # Clean up the clips; the H5s are training data and must survive.
        for k, _, _ in rows:
            await ws.send(env("demo_delete", {"name": f"{folder}_episode_{start_count + k:04d}"}))
        await asyncio.sleep(1.0)
        h5_left = all(os.path.isfile(os.path.join(out_dir, folder, f"episode_{start_count + k:04d}.h5"))
                      for k in range(takes))
        ok &= h5_left
        print(f"  {'PASS' if h5_left else 'FAIL'}  clips deleted, H5 episodes kept")
        done.set()
        await rtask
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="rr_smoke")
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.folder, a.takes, a.secs)))
