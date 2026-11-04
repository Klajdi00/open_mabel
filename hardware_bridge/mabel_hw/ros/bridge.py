"""bridge.py — the HAL supervisor: one process, every subsystem, one facade.

Moved from the top-level hardware_bridge.py (which remains the systemd/script
entry point and re-exports this module's surface). Flow, unchanged in shape:

    flock singleton  ->  rclpy.init  ->  MabelHardware.open()  ->  one rclpy
    node per enabled subsystem (build(cfg, hw=hw))  ->  MultiThreadedExecutor

New vs the pre-mabel_hw bridge: the supervisor now owns a single
:class:`mabel_hw.MabelHardware` and hands it to every node — so the DDS
adapters, the write leases, and any in-process consumer all share ONE set of
device objects. Enable/disable + tune each subsystem in mabel_hw/config.py
(or via MABEL_HW_* env vars). Every subsystem still degrades gracefully: a
failed node/device never sinks the HAL.
"""

import importlib
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

from .. import config
from ..api import MabelHardware
from ..singleton import _acquire_singleton

# (config attr, node module) for each subsystem. Keys are the MABEL_HW_<NAME>
# enable flags; order is bring-up order.
_REGISTRY = [
    ("BASE", "mabel_hw.ros.base_node"),
    ("ARMS", "mabel_hw.ros.arms_node"),
    ("BODY", "mabel_hw.ros.body_node"),
    ("HANDS", "mabel_hw.ros.hands_node"),
    ("LIFT", "mabel_hw.ros.lift_node"),
    ("LIDAR", "mabel_hw.ros.lidar_node"),
    ("AUDIO", "mabel_hw.ros.audio_node"),
    ("CAMERAS", "mabel_hw.cameras.manager"),
]

# Facade subsystems the supervisor brings up, keyed by THEIR config dicts. Each
# _REGISTRY node above is a thin DDS adapter over these — e.g. the ARMS node
# serves /arms/* from the left_arm + right_arm handles. hw opens only what its
# own config enables; the legacy BODY node self-builds when its new-taxonomy
# group (torso) is disabled.
_HW_SUBSYSTEMS = ("base", "left_arm", "right_arm", "neck", "torso",
                  "hands", "lift", "lidar", "cameras")


def _hw_enabled(name: str) -> bool:
    cfg = getattr(config, name.upper(), None) or {}
    return bool(cfg.get("enabled", False))


# Subsystems spun on their OWN SingleThreadedExecutor thread instead of the
# shared pool. Measured 2026-07-27 on the live robot: with the full bridge up
# (~15 nodes, camera retry-timer churn), the shared MultiThreadedExecutor
# NEVER serviced the body node's /body/torque_enable subscription — CycloneDDS
# had the sample ACKed in the reader cache (verified by wire trace) and the
# node's own timers ran at a clean 10 Hz, but the callback never fired, on
# every restart, while the SAME code serviced fine in a small-process repro.
# A dedicated executor sidesteps that rclpy wait-set starvation entirely for
# the safety-critical torque/command path. ARMS added 2026-07-29: its
# /arms/command callback shared the same starved pool (the teleop latency
# audit) — the two subsystems that move the most mass now each own a thread.
_DEDICATED_SPIN = {"BODY", "ARMS"}


def _build_enabled(logger, hw=None):
    nodes = []
    for cfg_name, module_path in _REGISTRY:
        cfg = getattr(config, cfg_name)
        if not cfg.get("enabled", False):
            logger.info(f"[{cfg_name.lower()}] disabled; skipping.")
            continue
        try:
            mod = importlib.import_module(module_path)
            nodes.append((cfg_name, mod.build(cfg, hw=hw)))
            logger.info(f"[{cfg_name.lower()}] brought up.")
        except Exception as e:                        # one subsystem must not sink the HAL
            logger.error(f"[{cfg_name.lower()}] failed to start ({e}); continuing.")
    return nodes


def main(args=None):
    _acquire_singleton()        # refuse to start if another bridge owns the hardware
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger("hardware_bridge")
    logger.info("MABEL hardware bridge (single HAL) starting...")

    # The one facade every node shares. The flock is already held above, so the
    # facade must not re-acquire; subsystems= mirrors the config enable flags.
    hw = MabelHardware(config, acquire_singleton=False,
                       subsystems=[n for n in _HW_SUBSYSTEMS if _hw_enabled(n)],
                       logger=lambda m: logger.info(m))
    try:
        hw.open()
    except Exception as e:                            # never a startup wall
        logger.error(f"MabelHardware.open failed ({e}); nodes fall back to "
                     "standalone drivers.")

    built = _build_enabled(logger, hw=hw)
    if not built:
        logger.warn("no subsystems enabled; nothing to do. Enable some in config.py.")
        hw.close()
        rclpy.try_shutdown()
        return
    nodes = [n for _, n in built]

    executor = MultiThreadedExecutor()
    side_threads = []
    for cfg_name, n in built:
        if cfg_name in _DEDICATED_SPIN:
            ded = SingleThreadedExecutor()
            ded.add_node(n)
            t = threading.Thread(target=ded.spin, daemon=True,
                                 name=f"spin-{cfg_name.lower()}")
            t.start()
            side_threads.append((ded, t))
            logger.info(f"[{cfg_name.lower()}] on a dedicated executor thread.")
        else:
            executor.add_node(n)

    logger.info(f"hardware bridge online: {len(nodes)} subsystem(s) hosted.")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for ded, _t in side_threads:
            try:
                ded.shutdown(timeout_sec=1.0)
            except Exception:
                pass
        for n in nodes:
            try:
                n.destroy_node()
            except Exception:
                pass
        try:
            hw.close()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
