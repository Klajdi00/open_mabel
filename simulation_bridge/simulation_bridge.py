#!/usr/bin/env python3
"""
simulation_bridge.py — the MABEL SIMULATION HAL: the one process that owns the
MuJoCo world, exactly as hardware_bridge.py is the one process that owns the
physical devices.

    real robot                                  simulation
    hardware_bridge  (owns Teensy/CAN/serial)   simulation_bridge (owns mj_step)
         ▲  DDS /arms/command …                      ▲  DDS /sim/arms/command …
         │  RPC hand_bridge :8779                    │  RPC :8788 (newline-JSON)
         │                                           │  + state memmap [t,qpos,qvel,ctrl]
    the SAME teleop server / ROS stack / GUIs consume either — only the source
    differs, and clients are told kind = "sim".

Single-owner rule: ONE bridge per simulated robot. On launch this process
TAKES OVER its instance — any previous bridge holding the instance lock is
terminated (sims are cheap; a stale owner would freeze the world). Multiple
simulated robots on one computer = multiple instances (`--instance sim1` →
its own lock, state memmap, RPC port, and /sim1 ROS namespace).

ROS (rclpy) is OPTIONAL: on a Mac the bridge runs ROS-free (RPC + memmap); on
Ubuntu it additionally mirrors the full hardware_bridge DDS contract under the
instance namespace (default /sim) — same topics, same message types, same
rates, sim_-distinguished by namespace.

    python3 -m simulation_bridge                 # home scene, instance sim0
    python3 -m simulation_bridge --scene S.xml --instance sim1
"""
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import signal
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from simulation_bridge import config as CFG               # noqa: E402
from simulation_bridge.engine import SimWorld             # noqa: E402
from simulation_bridge.rpc import SimBridgeServer, SimBridgeService  # noqa: E402

log = logging.getLogger("simbridge")


def _acquire_lock(path, takeover=True):
    """Takeover semantics per instance. Unlike hardware_bridge (which
    refuses), the SIM bridge TAKES OVER: kill the stale owner, wait for the
    flock to free, then RELEASE our probe — SimWorld itself holds the real
    lock for the lifetime of the world ('when launched, kill all the other
    bridges of this robot')."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    for attempt in range(40):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # lock is free — hand it to SimWorld (tiny race window is fine:
            # we're the only launcher this instant)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            return None
        except BlockingIOError:
            if not takeover:
                sys.exit(f"simulation_bridge: another bridge owns "
                         f"{path} — refusing (use --takeover)")
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                pid = int(os.read(fd, 32).decode().strip() or "0")
            except Exception:
                pid = 0
            if pid and attempt == 0:
                log.warning("taking over: terminating previous bridge pid %d",
                            pid)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif pid and attempt == 20:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.25)
    sys.exit("simulation_bridge: could not take the instance lock")


def _maybe_ros(world, ns):
    """Ubuntu: the DDS mirror of the hardware contract under `ns`. Fully
    optional — the Mac path never imports rclpy."""
    mode = str(CFG.ROS).lower()
    if mode in ("0", "false", "no", "off"):
        return None
    try:
        from simulation_bridge.ros_io import start_ros
    except Exception as e:
        if mode in ("1", "true", "yes", "on"):
            raise
        log.info("ROS layer off (%s)", e)
        return None
    try:
        return start_ros(world, ns)
    except Exception as e:
        if mode in ("1", "true", "yes", "on"):
            raise
        log.info("ROS layer unavailable — running ROS-free (%s)", e)
        return None


def main():
    ap = argparse.ArgumentParser(description="MABEL simulation bridge (MuJoCo HAL)")
    ap.add_argument("--scene", default=None, help="scene xml (default: home)")
    ap.add_argument("--instance", default=CFG.INSTANCE,
                    help="robot instance name (lock/state/port/namespace scope)")
    ap.add_argument("--rpc-port", type=int, default=None)
    ap.add_argument("--ns", default=None, help="ROS namespace (default /sim)")
    ap.add_argument("--no-takeover", action="store_true",
                    help="refuse instead of killing a previous owner")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname).1s %(message)s",
                        datefmt="%H:%M:%S")

    _acquire_lock(CFG.lock_file(args.instance), takeover=not args.no_takeover)

    world = SimWorld(scene=args.scene, instance=args.instance)
    server = SimBridgeServer(SimBridgeService(world),
                             port=args.rpc_port or CFG.rpc_port(args.instance))
    server.start()
    ros = _maybe_ros(world, args.ns or CFG.ROS_NS)

    def _bye(*_a):
        world.stop()
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    log.info("simulation_bridge up — instance %s · scene %s · rpc :%d · %s",
             args.instance, os.path.basename(world.scene), server.port,
             "DDS mirror on" if ros else "ROS-free")
    try:
        world.run()                       # blocks; the physics loop
    finally:
        server.stop()
        if ros:
            try:
                ros.stop()
            except Exception:
                pass
        log.info("simulation_bridge down")


if __name__ == "__main__":
    main()
