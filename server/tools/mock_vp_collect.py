"""mock_vp_collect.py — simulate the Vision Pro / iOS "Start collecting data"
button driving the real-robot collector (real_curation_collector.py).

It speaks the SAME wire shape the app's WebSocketLink sends for `record`
({"active":bool,"task":str,"client":str}) plus the dataset folder pick/create
messages, over the collector's control WebSocket. Use it to exercise the whole
start → collect → stop signal path with zero hardware motion.

Examples:
    # one 60 s episode into a freshly-named vision_os dataset folder
    python3 tools/mock_vp_collect.py --client visionos --task pick_place --duration 60

    # ten 60 s samples (numbered dataset folders) — the autonomous test
    python3 tools/mock_vp_collect.py --client visionos --task pick_place \
        --duration 60 --count 10

    # just list / create folders (folder-picker mock)
    python3 tools/mock_vp_collect.py --list
    python3 tools/mock_vp_collect.py --create --task pour_water
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time


async def _rpc(ws, msg):
    await ws.send(json.dumps(msg))
    return json.loads(await ws.recv())


async def run(args):
    import websockets
    url = f"ws://{args.host}:{args.port}"
    async with websockets.connect(url, open_timeout=8) as ws:
        # folder-picker mocks
        if args.list:
            print("[mock-VP] list datasets:", json.dumps(await _rpc(ws, {"type": "list"}), indent=2))
            return 0
        if args.create:
            print("[mock-VP] create/select folder:",
                  await _rpc(ws, {"type": "create", "task": args.task}))
            return 0

        st = await _rpc(ws, {"type": "status"})
        print(f"[mock-VP] connected {url}  client={st.get('client')}  "
              f"existing datasets={st.get('datasets')}")

        ok_eps = 0
        for i in range(args.count):
            # ── START (exactly what the VP button sends) ──────────────────────
            start = await _rpc(ws, {"type": "record", "active": True,
                                    "task": args.task, "client": args.client})
            if not start.get("ok"):
                print(f"[mock-VP] START rejected: {start}"); return 1
            print(f"[mock-VP] ▶ episode {i+1}/{args.count}  START → {start['dir']}")

            # ── collect (robot is NOT commanded; a separate mock_control node
            #    injects synthetic action targets on /*/command) ───────────────
            t0 = time.time()
            while time.time() - t0 < args.duration:
                await asyncio.sleep(2.0)
                s = await _rpc(ws, {"type": "status"})
                print(f"    …{s.get('elapsed'):5.1f}s  frames={s.get('frames')}  "
                      f"seen={len(s.get('seen', []))} channels", end="\r", flush=True)

            # ── STOP ──────────────────────────────────────────────────────────
            stop = await _rpc(ws, {"type": "record", "active": False})
            print(f"\n[mock-VP] ■ STOP  frames={stop.get('frames')}  "
                  f"episode={stop.get('episode')}")
            print(f"    channels seen: {stop.get('seen')}")
            if stop.get("ok") and stop.get("frames", 0) > 0:
                ok_eps += 1
            if i + 1 < args.count:
                await asyncio.sleep(args.gap)

        print(f"[mock-VP] done — {ok_eps}/{args.count} episodes recorded with frames.")
        return 0 if ok_eps == args.count else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9095)
    ap.add_argument("--client", default="visionos", help="visionos|ios|real")
    ap.add_argument("--task", default="mock_task")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per episode")
    ap.add_argument("--count", type=int, default=1, help="number of episodes")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between episodes")
    ap.add_argument("--list", action="store_true", help="list dataset folders and exit")
    ap.add_argument("--create", action="store_true", help="create/select a folder and exit")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
