"""manager.py — camera SUPERVISOR: one process per camera + the video relay.

The manager no longer hosts any capture loop. All cameras used to share the ONE
bridge process and its ONE GIL, so every added stream stole frame rate from the
others (front D435 measured 30 -> ~15 Hz repeatedly). Now:

    manager (this node, in the bridge process)
      ├─ static mount TFs for the whole camera suite (published once, latched)
      ├─ spawns  python3 -m mabel_hw.cameras.camera_worker realsense    (own process)
      ├─ spawns  python3 -m mabel_hw.cameras.camera_worker zed          (own process)
      ├─ spawns  python3 -m mabel_hw.cameras.camera_worker wrist_left   (own process)
      ├─ spawns  python3 -m mabel_hw.cameras.camera_worker wrist_right  (own process)
      └─ spawns  python3 -m mabel_hw.cameras.video_relay                (own process:
                 VideoServer :8080 MJPEG + UDP :9091/:9092, fed by the workers over
                 an AF_UNIX datagram socket — NOT by subscribing to their ROS topics;
                 see video_pipe.py. The workers ALSO publish ROS image topics, but
                 those exist for RViz / cuVSLAM / nvblox, and are not on the path to
                 a device. Fixed 2026-07-10; this line used to claim the ROS path.)

Each camera runs at its native rate on its own core; the encode/fan-out CPU lives
in the relay. The bridge remains the SINGLE HAL boundary — the workers are its
children (same lifetime, same env/config, same systemd unit); nothing outside the
bridge touches camera hardware. A dead worker is restarted with backoff and never
takes the others down.

TF layout (unchanged):
    base_link -> front_link -> front_{color,infra1}_optical_frame        (D435)
    head -> zed_camera_link -> zed_{left,right}_camera_optical_frame     (ZED-M)
    left_palm/right_palm -> wrist_<side>_camera_link -> ..._optical_frame (wrist)
"""

import math
import os
import subprocess
import sys
import time

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

# REP-103 body(x-fwd,y-left,z-up) -> optical(x-right,y-down,z-fwd) rotation.
_OPTICAL_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)

_RESTART_BACKOFF_S = 3.0
# <bridge>/mabel_hw/cameras/manager.py -> the bridge dir is two up; workers
# spawn with cwd there so mabel_hw resolves without a pip install.
_BRIDGE_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def _quat(roll, pitch, yaw):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (sr * cp * cy - cr * sp * sy,     # x
            cr * sp * cy + sr * cp * sy,     # y
            cr * cp * sy - sr * sp * cy,     # z
            cr * cp * cy + sr * sp * sy)     # w


def build(cfg, **_):
    return CameraManager(cfg)


class CameraManager(Node):
    def __init__(self, cfg):
        super().__init__("mabel_hw_cameras")
        self.cfg = cfg
        self._tf = StaticTransformBroadcaster(self)
        self._tfs = []
        self._procs = {}                 # label -> Popen
        self._died_at = {}               # label -> monotonic time of last death

        rs = cfg["realsense"]
        if rs["enabled"]:
            m = rs["mount"]
            self._static(rs["base_frame"], rs["camera_frame"],
                         m["x"], m["y"], m["z"], m["roll"], m["pitch"], m["yaw"])
            self._static(rs["camera_frame"], "front_color_optical_frame",
                         0, 0, 0, *_OPTICAL_RPY)
            # Rectified IR stereo (the cuVSLAM input): the LEFT imager (infra1) sits
            # at the front cam mount, same optical rotation as color. The driver adds
            # infra1 -> infra2 at the CALIBRATED baseline once the camera reports it.
            self._static(rs["camera_frame"], "front_infra1_optical_frame",
                         0, 0, 0, *_OPTICAL_RPY)
            self._spawn("realsense")

        zed = cfg["zed"]
        if zed["enabled"]:
            m = zed["mount"]
            self._static(zed["mount_frame"], zed["camera_frame"],
                         m["x"], m["y"], m["z"], m["roll"], m["pitch"], m["yaw"])
            self._static(zed["camera_frame"], "zed_left_camera_optical_frame",
                         0, 0, 0, *_OPTICAL_RPY)
            # Right eye = left eye + the ZED-Mini 63 mm stereo baseline along optical
            # +X (REP-145, x-right) — a rectified rig cuVSLAM can triangulate.
            self._static("zed_left_camera_optical_frame",
                         "zed_right_camera_optical_frame",
                         0.063, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._spawn("zed")

        # Wrist cams (per-side uvc Arducam or RealSense D405 — see wrist_cams.py).
        # Mounts are DERIVED FROM THE MJCF (l_wrist_cam / r_wrist_cam on the palm
        # bodies) so the D405 cloud renders exactly like the sim's wrist camera.
        wrist = cfg.get("wrist")
        if wrist and wrist["enabled"]:
            for side in ("left", "right"):
                m = wrist.get(f"{side}_mount")
                parent = wrist.get(f"{side}_mount_frame")
                if m and parent:
                    cam_link = f"wrist_{side}_camera_link"
                    self._static(parent, cam_link, m["x"], m["y"], m["z"],
                                 m["roll"], m["pitch"], m["yaw"])
                    self._static(cam_link, f"wrist_{side}_camera_optical_frame",
                                 0, 0, 0, *_OPTICAL_RPY)
                self._spawn(f"wrist_{side}")

        if cfg.get("serve", True):
            self._spawn("video_relay", module="mabel_hw.cameras.video_relay", arg=None)

        if self._tfs:
            self._tf.sendTransform(self._tfs)
        self.create_timer(2.0, self._watch)
        self.get_logger().info(
            "cameras online (per-process): "
            + ", ".join(sorted(self._procs)) + ".")

    # ---- worker processes ----------------------------------------------------
    def _spawn(self, label, module="mabel_hw.cameras.camera_worker", arg="__label__"):
        cmd = [sys.executable, "-u", "-m", module]
        if arg is not None:
            cmd.append(label if arg == "__label__" else arg)
        try:
            self._procs[label] = subprocess.Popen(cmd, cwd=_BRIDGE_ROOT)
        except Exception as e:
            self.get_logger().error(f"camera worker '{label}' failed to spawn ({e}).")

    def _watch(self):
        """Restart dead workers with backoff; a crashing camera never takes the
        suite down (mirrors every driver's own idle-retry philosophy)."""
        now = time.monotonic()
        for label, proc in list(self._procs.items()):
            if proc.poll() is None:
                continue
            last = self._died_at.get(label, 0.0)
            if now - last < _RESTART_BACKOFF_S:
                continue
            self._died_at[label] = now
            self.get_logger().warn(
                f"camera worker '{label}' exited (rc={proc.returncode}); restarting.")
            module = "mabel_hw.cameras.video_relay" if label == "video_relay" \
                else "mabel_hw.cameras.camera_worker"
            self._spawn(label, module=module,
                        arg=None if label == "video_relay" else "__label__")

    def _static(self, parent, child, x, y, z, roll, pitch, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        qx, qy, qz, qw = _quat(roll, pitch, yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._tfs.append(t)

    def destroy_node(self):
        try:
            for p in self._procs.values():
                if p.poll() is None:
                    p.send_signal(2)                 # SIGINT: clean rclpy shutdown
            deadline = time.monotonic() + 3.0
            for p in self._procs.values():
                while p.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if p.poll() is None:
                    p.kill()
        finally:
            super().destroy_node()
