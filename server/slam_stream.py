"""
slam_stream.py — the :9605 SLAM + 2D-map plane (real robot).

Streams what the Isaac ROS SLAM stack produces, on its own dedicated port:

  MESH 0x01 — the nvblox 3D reconstruction (/nvblox_node/mesh_pointcloud)
  ODOM 0x02 — the cuVSLAM tracking pose (/visual_slam/tracking/odometry),
              i.e. the VISUAL localization estimate — distinct from the fused
              /odom the :9092 map plane ships
  GRID 0x04 — the 2D occupancy map (/map: nvblox slice or slam_toolbox)
  PATH 0x03 — the active Nav2 plan
  + a 1 Hz JSON text status frame {"type":"status", vslam, nvblox, grid, ...}

Wire records are docs/data_streams.md verbatim — the existing /map client
decoders work unchanged; only the port (and the odom SOURCE) differ.

Implementation: a thin wrapper over map_stream.MapRosSource, which already
owns every hard-won lesson on this path (QoS matching, TF-to-odom, nvblox
block accumulation, heartbeats, rate caps). This module just points a second
instance at the cuVSLAM odometry, disables its cloud channel (clouds have
their own :9601-9604 planes), and adds the status ticker. No-op (returns
None) when rclpy is unavailable — the bridge then serves the sim notice.
"""
from __future__ import annotations

import json
import os
import threading
import time

from map_stream import MapStreamHub, MapRosSource

__all__ = ["SlamHub", "start_slam_stream"]


class SlamHub(MapStreamHub):
    """Dedicated hub for :9605 (same record types as MAP_HUB, own subscribers)."""


def start_slam_stream(hub: SlamHub, log) -> "MapRosSource | None":
    """Start the ROS source feeding `hub`, or None when ROS is unavailable."""
    src = MapRosSource(
        hub,
        mesh_topic=os.getenv("MABEL_SLAM_MESH_TOPIC", "/nvblox_node/mesh_pointcloud"),
        odom_topic=os.getenv("MABEL_SLAM_ODOM_TOPIC", "/visual_slam/tracking/odometry"),
        plan_topic=os.getenv("MABEL_SLAM_PLAN_TOPIC", "/plan"),
        grid_topic=os.getenv("MABEL_SLAM_GRID_TOPIC", "/map"),
        cloud_topic="",                    # per-camera clouds live on :9601-9604
    )
    if src.start() is None:
        return None
    threading.Thread(target=_status_loop, args=(hub, src), name="slam-status",
                     daemon=True).start()
    log.info("slam stream: nvblox mesh + cuVSLAM pose + /map grid → :9605 hub")
    return src


def _status_loop(hub: SlamHub, src: MapRosSource) -> None:
    """1 Hz JSON status so a client can tell 'no mesh yet' from 'SLAM is down'."""
    while True:
        time.sleep(1.0)
        if not hub.enabled():
            continue
        now = time.time()
        try:
            hub.publish_text(json.dumps({
                "type": "status", "t": now,
                # cuVSLAM alive = a tracking pose arrived within the last 2 s.
                "vslam": (now - getattr(src, "_odom_rx_t", 0.0)) < 2.0,
                "nvblox": int(getattr(src, "_mesh_n", 0)) > 0,
                "grid": getattr(src, "_grid_cache", None) is not None,
                "meshPoints": int(getattr(src, "_mesh_n", 0)),
            }))
        except Exception:
            pass
