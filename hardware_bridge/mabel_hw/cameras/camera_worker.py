"""camera_worker.py — ONE camera per PROCESS (the bridge's GIL fix).

All cameras used to share the single bridge process: every driver's per-frame work
(librealsense copies, ZED depth grab, JPEG decode, ROS publish) contended for ONE
GIL, so adding a camera stole frame rate from the others (measured repeatedly:
front D435 30 -> ~15 Hz whenever a second/third stream was active). The camera
manager now SPAWNS one of these workers per physical camera; each worker is its own
OS process with its own rclpy context, so every camera runs at its native rate on
its own core.

    python3 -m mabel_hw.cameras.camera_worker realsense     front D435i  -> /camera/front/*
    python3 -m mabel_hw.cameras.camera_worker zed           head ZED-M   -> /camera/zed/*
    python3 -m mabel_hw.cameras.camera_worker wrist_left    D405 or UVC  -> /camera/wrist_left/*
    python3 -m mabel_hw.cameras.camera_worker wrist_right   UVC or D405  -> /camera/wrist_right/*

Workers publish ROS topics ONLY (video=None): the network video (:8080 MJPEG +
UDP JPEG/H.264) is served by the separate mabel_hw.cameras.video_relay process, which
subscribes to these topics on demand. Static mount TFs stay in the parent manager
(published once for the whole camera suite). Config comes from the bridge's
config.py — the worker inherits the parent's environment, so all MABEL_HW_CAM_*
env overrides apply identically.

Each worker exits on SIGINT/SIGTERM; the manager restarts a dead worker with
backoff (a crashing camera never takes the others down).
"""

import signal
import sys

import rclpy
from rclpy.node import Node

from mabel_hw import config


def _build(node, kind):
    cams = config.CAMERAS
    # Network video BYPASSES ROS entirely: this worker downscales + JPEG-encodes
    # its own frames and datagrams them to mabel_hw.cameras.video_relay (video_pipe.py) —
    # the drivers' existing `video` hook, now crossing the process boundary. The
    # ROS image topics remain for RViz/SLAM; they are no longer the video path.
    from .video_pipe import PipeVideo
    pipe = PipeVideo(net_width=cams.get("net_width", 640),
                     jpeg_quality=cams.get("net_quality", 65))
    if kind == "realsense":
        from .realsense_d435 import RealSenseD435
        return RealSenseD435(node, cams["realsense"], video=pipe)
    if kind == "zed":
        from .zed_mini import ZedMini
        return ZedMini(node, cams["zed"], video=pipe)
    if kind in ("wrist_left", "wrist_right"):
        side = kind.split("_", 1)[1]
        wrist = cams["wrist"]
        if str(wrist.get(f"{side}_kind", "uvc")).lower() == "d405":
            from .wrist_d405 import WristD405
            return WristD405(node, side, wrist, video=pipe)
        from .wrist_cams import _WristSide
        return _WristSide(node, side, wrist[f"{side}_device"],
                          wrist[f"{side}_fallback"], wrist, video=pipe)
    raise SystemExit(f"unknown camera kind {kind!r}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m mabel_hw.cameras.camera_worker "
                         "<realsense|zed|wrist_left|wrist_right>")
    kind = sys.argv[1]
    rclpy.init()
    node = Node(f"mabel_hw_cam_{kind}")
    driver = _build(node, kind)
    node.get_logger().info(f"camera worker '{kind}' up (own process, ROS-only).")

    def _bail(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _bail)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            driver.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
