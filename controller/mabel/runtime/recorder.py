"""
recorder.py — teleop episode recorder (sim AND real robot).

Source-agnostic: a *sensor source* (SimSensorSource for MuJoCo, or a hardware/ROS
source on the real robot) produces a `sample` dict each tick; this writer lays it
out on disk in a standard imitation-learning episode layout. The SAME recorder
runs in sim and on the robot — only the source differs.

A `sample` dict (every field optional — missing channels are simply skipped):

    {
      "t":      float,                       # source time (sim time or ROS stamp)
      "rgb":    {cam_name: HxWx3 uint8},     # colour frames
      "depth":  {cam_name: HxW float32 (m)}, # depth maps (metres)
      "lidar":  np.ndarray (N,) float32,     # planar-LiDAR ranges (metres)
      "qpos":     np.ndarray (nq,) float,    # REALIZED generalized position (state after the controller)
      "qpos_des": np.ndarray (nq,) float,    # DESIRED joint pose (controller set-point, before the plant responds)
      "qvel":   np.ndarray (nv,) float,      # generalized velocity
      "ctrl":   np.ndarray (nu,) float,      # actuator commands (torques for impedance joints, targets for servos)
      "base":   (x, y, z, qw, qx, qy, qz),   # odometry (world pose of the chassis)
      "control": {...},                      # the operator's teleop command (json-able)
    }

On disk (`<out_dir>/<name>_<YYYYmmdd_HHMMSS>/`):

    meta.json                  # channels, joint names, rate, counts, start/end, source
    timestamps.npy             # (frames,)            source time per frame
    qpos.npy / qvel.npy / ctrl.npy   # (frames, n*)  realized robot state + actuator commands
    qpos_des.npy               # (frames, nq)         desired joint pose (controller set-point)
    lidar.npy                  # (frames, N)          planar LiDAR
    base.csv                   # frame,t,x,y,z,qw,qx,qy,qz   odometry
    control.csv                # frame,t,<json>       teleop command per frame
    cam/<name>/frameNNNNNN.jpg # colour
    depth/<name>/frameNNNNNN.png   # 16-bit millimetres
"""
from __future__ import annotations

import os
import csv
import json
import time
import logging
from datetime import datetime

import numpy as np

log = logging.getLogger("recorder")

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                       # pragma: no cover - cv2 should be present
    _HAVE_CV2 = False


class EpisodeRecorder:
    """Buffers per-frame samples and writes a complete episode to disk.

    `fps` is the *source-time* sampling rate (frames are taken when the source's
    time has advanced 1/fps), so an episode stays temporally consistent even if
    the sim runs slower than real time while rendering all the cameras."""

    def __init__(self, out_dir: str, fps: float = 10.0):
        self.out_dir = os.path.expanduser(out_dir)
        self.fps = max(1.0, float(fps))
        self.active = False
        self.ep_dir = None
        self.task = "default"          # current task folder (episodes live inside it)
        self._frame = 0
        self._last_t = -1e9
        self._reset_buffers()

    # ── task folders / status ──────────────────────────────────────────────────
    @staticmethod
    def _safe(name: str | None, fallback: str) -> str:
        s = "".join(c for c in (name or "") if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        return s or fallback

    def set_task(self, task: str | None):
        """Select the task folder episodes are saved into (created on demand)."""
        self.task = self._safe(task, "default")

    def count_episodes(self, task: str | None = None) -> int:
        d = os.path.join(self.out_dir, self._safe(task or self.task, "default"))
        if not os.path.isdir(d):
            return 0
        return sum(1 for n in os.listdir(d)
                   if n.startswith("episode_") and os.path.isdir(os.path.join(d, n)))

    def list_tasks(self) -> list:
        if not os.path.isdir(self.out_dir):
            return []
        return sorted(n for n in os.listdir(self.out_dir)
                      if os.path.isdir(os.path.join(self.out_dir, n)) and not n.startswith("."))

    def status(self) -> dict:
        tasks = self.list_tasks()
        counts = {t: self.count_episodes(t) for t in tasks}
        return {"task": self.task, "episodeCount": self.count_episodes(),
                "isRecording": self.active, "tasks": tasks,
                "taskCounts": counts, "total": sum(counts.values()),
                "dir": self.out_dir}

    def _reset_buffers(self):
        self._ts = []
        self._qpos, self._qvel, self._ctrl, self._lidar = [], [], [], []
        self._qpos_des = []
        self._base_rows, self._control_rows = [], []
        self._meta = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self, task: str | None, meta: dict):
        """Begin a new episode inside the task folder. The episode dir is uniquely
        named `episode_<NNNN>_<timestamp>` — ordered AND timestamped, so nothing is
        ever overwritten even across machines/clock changes."""
        if self.active:
            self.stop()
        self.set_task(task)
        task_dir = os.path.join(self.out_dir, self.task)
        os.makedirs(task_dir, exist_ok=True)
        idx = self.count_episodes()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep = f"episode_{idx:04d}_{stamp}"
        while os.path.exists(os.path.join(task_dir, ep)):      # never overwrite
            idx += 1
            ep = f"episode_{idx:04d}_{stamp}"
        self.ep_dir = os.path.join(task_dir, ep)
        os.makedirs(self.ep_dir)
        for cam in meta.get("rgb_cameras", []):
            os.makedirs(os.path.join(self.ep_dir, "cam", cam), exist_ok=True)
        for cam in meta.get("depth_cameras", []):
            os.makedirs(os.path.join(self.ep_dir, "depth", cam), exist_ok=True)
        self._reset_buffers()
        self._frame = 0
        self._last_t = -1e9
        self._meta = dict(meta)
        self._meta.update(task=self.task, episode=ep, index=idx, fps=self.fps, start_wall=time.time())
        self.active = True
        log.info("REC ▶ %s / %s → %s  (%.0f fps)", self.task, ep, self.ep_dir, self.fps)

    def maybe_record(self, t: float, sample_fn):
        """Capture a frame if 1/fps of source-time has elapsed. `sample_fn()` is
        called lazily (it renders cameras) only when a frame is actually due."""
        if not self.active or (t - self._last_t) < (1.0 / self.fps):
            return
        self._last_t = t
        try:
            self._write_frame(t, sample_fn())
        except Exception as e:               # never let recording crash teleop
            log.warning("REC frame %d skipped: %s", self._frame, e)

    def _write_frame(self, t: float, s: dict):
        i = self._frame
        for cam, img in (s.get("rgb") or {}).items():
            self._save_jpg(os.path.join(self.ep_dir, "cam", cam, f"frame{i:06d}.jpg"), img)
        for cam, dep in (s.get("depth") or {}).items():
            self._save_depth(os.path.join(self.ep_dir, "depth", cam, f"frame{i:06d}.png"), dep)
        if s.get("lidar") is not None:  self._lidar.append(np.asarray(s["lidar"], dtype=np.float32))
        if s.get("qpos")  is not None:  self._qpos.append(np.asarray(s["qpos"], dtype=np.float32))
        if s.get("qpos_des") is not None:  self._qpos_des.append(np.asarray(s["qpos_des"], dtype=np.float32))
        if s.get("qvel")  is not None:  self._qvel.append(np.asarray(s["qvel"], dtype=np.float32))
        if s.get("ctrl")  is not None:  self._ctrl.append(np.asarray(s["ctrl"], dtype=np.float32))
        self._ts.append(float(t))
        if s.get("base") is not None:
            self._base_rows.append([i, t, *[float(v) for v in s["base"]]])
        self._control_rows.append([i, t, json.dumps(s.get("control", {}), separators=(",", ":"))])
        self._frame += 1

    def stop(self) -> str | None:
        """Finalise the episode: flush arrays + CSVs + meta.json. Returns the dir."""
        if not self.active:
            return None
        d = self.ep_dir
        try:
            np.save(os.path.join(d, "timestamps.npy"), np.asarray(self._ts, dtype=np.float64))
            for nm, buf in (("qpos", self._qpos), ("qpos_des", self._qpos_des),
                            ("qvel", self._qvel), ("ctrl", self._ctrl), ("lidar", self._lidar)):
                if buf:
                    np.save(os.path.join(d, f"{nm}.npy"), np.stack(buf))
            if self._base_rows:
                self._write_csv(os.path.join(d, "base.csv"),
                                ["frame", "t", "x", "y", "z", "qw", "qx", "qy", "qz"], self._base_rows)
            if self._control_rows:
                self._write_csv(os.path.join(d, "control.csv"),
                                ["frame", "t", "control_json"], self._control_rows)
            self._meta.update(frames=self._frame, end_wall=time.time())
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(self._meta, f, indent=2)
            log.info("REC ■ episode saved: %d frames → %s", self._frame, d)
        except Exception as e:
            log.error("REC failed to finalise episode: %s", e)
        finally:
            self.active = False
        return d

    # ── encoders ──────────────────────────────────────────────────────────────
    @staticmethod
    def _save_jpg(path: str, rgb: np.ndarray):
        if _HAVE_CV2:
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            np.save(path.replace(".jpg", ".npy"), rgb)

    @staticmethod
    def _save_depth(path: str, depth_m: np.ndarray):
        # 16-bit millimetres: compact, lossless enough for IL (0…65.5 m).
        mm = np.clip(np.asarray(depth_m) * 1000.0, 0, 65535).astype(np.uint16)
        if _HAVE_CV2:
            cv2.imwrite(path, mm)
        else:
            np.save(path.replace(".png", ".npy"), mm)

    @staticmethod
    def _write_csv(path: str, header: list, rows: list):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
