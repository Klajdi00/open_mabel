#!/usr/bin/env python3
"""map_latency_probe.py — measure the live-map (/map port) throughput + latency.

Run while the ROS server is up (sim_ros_server.sh / real_ros_server.sh). It connects
to the dedicated map WebSocket and reports, for both record types:

  * ODOM (0x02): rate (Hz) + inter-arrival p50/p95 — the real-time pose channel.
  * MESH (0x01): snapshot rate (Hz, counting final chunks), per-snapshot point count,
    chunk sizes, and total bandwidth.

Compare the rates to the ROS source to confirm the server adds no lag:
    ros2 topic hz /odom
    ros2 topic hz /nvblox_node/mesh_pointcloud
The bridge logs its own encode+enqueue time ("map mesh: … encode+enqueue X ms").

    python map_latency_probe.py --url ws://127.0.0.1:9092 --secs 12
"""
import argparse
import asyncio
import statistics
import struct
import time

import websockets


def _stats(ts):
    if len(ts) > 1:
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
    else:
        rate = 0.0
    if len(ts) >= 3:
        d = sorted((ts[i] - ts[i - 1]) * 1e3 for i in range(1, len(ts)))
        p50 = statistics.median(d)
        p95 = d[min(len(d) - 1, int(len(d) * 0.95))]
    else:
        p50 = p95 = 0.0
    return rate, p50, p95


async def main(url, secs):
    odom_t, mesh_final_t, chunk_bytes, snap_pts = [], [], [], []
    print(f"connecting {url} for {secs}s …")
    async with websockets.connect(url, max_size=2 ** 24) as ws:
        start = time.perf_counter()
        while time.perf_counter() - start < secs:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=secs)
            except asyncio.TimeoutError:
                break
            now = time.perf_counter()
            if not isinstance(msg, (bytes, bytearray)) or not msg:
                continue
            tag = msg[0]
            if tag == 0x01:
                chunk_bytes.append(len(msg))
                snap_pts.append(struct.unpack_from("<I", msg, 4)[0])
                if msg[1] & 0x01:                 # final chunk → one full snapshot
                    mesh_final_t.append(now)
            elif tag == 0x02:
                odom_t.append(now)

    r, p50, p95 = _stats(odom_t)
    print(f"\nODOM  {len(odom_t):4d} msgs   {r:6.1f} Hz   inter-arrival p50/p95 {p50:5.1f}/{p95:5.1f} ms")
    r, p50, p95 = _stats(mesh_final_t)
    print(f"MESH  {len(mesh_final_t):4d} snaps  {r:6.2f} Hz   inter-arrival p50/p95 {p50:5.0f}/{p95:5.0f} ms")
    if chunk_bytes:
        mb = sum(chunk_bytes) / 1e6
        print(f"      {len(chunk_bytes)} chunks, {mb:.1f} MB total ({mb/max(secs,1e-9):.2f} MB/s), "
              f"max chunk {max(chunk_bytes)/1e3:.0f} KB, up to {max(snap_pts)} pts/snapshot")
    else:
        print("      (no mesh yet — nvblox only emits geometry as the robot drives & sees new surface)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:9092")
    ap.add_argument("--secs", type=float, default=12)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.secs))
