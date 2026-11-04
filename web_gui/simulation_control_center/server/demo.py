"""
Offline DEMO mode — replay a recorded episode so the Sim Control Center is a
full, self-contained demo with NO live sim/bridge and NO MuJoCo on the host
(e.g. a web-only demo host). One recorded episode gives all three planes the user
asked for offline:

  • robot MOTION  — obs/joint_pos replayed at 15 Hz (real MuJoCo-generated
                    trajectory, so the robot actually moves like the sim);
  • VIDEO         — the 7 recorded camera JPEG streams, looped as MJPEG, so the
                    tiles / FPV / 3D projections show real footage;
  • MAP           — obs/lidar replayed into the 360° scan the mirror renders.

The episode's actuator_names are MuJoCo-style ("Left Arm 1", "Lift"); the client
keys jointPositions by ios-style names ("left_arm_1", "lift_lower/upper"), so we
normalize names and split Lift → lift_lower+lift_upper (each = Lift/2), matching
controller/mabel/facade.py.
"""
import glob
import json
import os
import threading
import time

import numpy as np

from . import config as CFG

# Sim-CC camera slot  →  recorded episode camera name.
CAM_MAP = {"main": "zed_left", "head": "zed_left", "head_left": "zed_left",
           "front": "body_front_cam", "wrist_left": "l_wrist_cam",
           "wrist_right": "r_wrist_cam"}


def _norm(name):
    return name.strip().lower().replace(" ", "_")


def find_demo_episode():
    """Locate a demo episode .h5: explicit env first, then the curator datasets."""
    env = os.environ.get("MABEL_SCC_DEMO_EPISODE")
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.expanduser(env)
    cands = []
    for base in (getattr(CFG, "EPISODES_DIR", ""),
                 os.path.join(getattr(CFG, "_REPO", os.path.expanduser("~")),
                              "robot_learning", "imitation_learning", "data_collection",
                              "datasets", "studio_sample")):
        if base and os.path.isdir(base):
            cands += sorted(glob.glob(os.path.join(base, "**", "episode_*.h5"),
                                      recursive=True))
    return cands[0] if cands else None


class DemoEpisode:
    """Lazy, thread-safe reader over one recorded episode (frames read on demand
    so we never hold the whole thing in RAM — important on a small host)."""

    def __init__(self, path):
        import h5py
        self.path = path
        self._lock = threading.Lock()
        self._f = h5py.File(path, "r")
        self.T = int(self._f.attrs.get("num_frames",
                                       self._f["obs/joint_pos"].shape[0]))
        sj = json.loads(self._f.attrs.get("schema_json", "{}"))
        self._acts = sj.get("actuator_names", [])
        self._cams = list(self._f["obs/cam"].keys()) if "obs/cam" in self._f else []
        # PRELOAD the tiny numeric arrays into RAM (120×59 floats ≈ 200 KB) so the
        # per-frame stream never touches h5 — h5py reads are NOT safe across the
        # DemoSource thread AND the asyncio event loop, and cross-thread reads were
        # returning stale frame-0 data (the robot looked frozen). Cameras stay lazy.
        self._pos = np.asarray(self._f["obs/joint_pos"][:], np.float64)
        self._vel = (np.asarray(self._f["obs/joint_vel"][:], np.float64)
                     if "obs/joint_vel" in self._f else np.zeros_like(self._pos))
        self._eff = (np.asarray(self._f["obs/joint_torque"][:], np.float64)
                     if "obs/joint_torque" in self._f else np.zeros_like(self._pos))
        self._lidar = (np.asarray(self._f["obs/lidar"][:], np.float64)
                       if "obs/lidar" in self._f else None)

    def state_frame(self, i):
        """(names, pos, vel, eff) in ios-name space for update_joints, + scan."""
        i = int(i) % max(1, self.T)
        pos, vel, eff = self._pos[i], self._vel[i], self._eff[i]
        scan = self._lidar[i].tolist() if self._lidar is not None else None
        names, P, V, E = [], [], [], []
        for k, mj in enumerate(self._acts):
            if k >= pos.shape[0]:
                break
            if mj == "Lift":                       # single actuator → two rig joints
                for suff in ("lift_lower", "lift_upper"):
                    names.append(suff); P.append(pos[k] / 2.0)
                    V.append(vel[k] / 2.0); E.append(eff[k])
            else:
                names.append(_norm(mj)); P.append(pos[k]); V.append(vel[k]); E.append(eff[k])
        return names, P, V, E, scan

    def cam_jpeg(self, sim_name, i):
        """Raw JPEG bytes for a sim-CC camera slot at frame i (looped), or None."""
        cam = CAM_MAP.get(sim_name, sim_name)
        if cam not in self._cams:
            cam = self._cams[0] if self._cams else None
        if not cam:
            return None
        key = f"obs/cam/{cam}/rgb"
        with self._lock:
            if key not in self._f:
                return None
            blob = np.asarray(self._f[key][int(i) % max(1, self.T)], np.uint8)
        return blob.tobytes() if blob.size else None


class DemoSource:
    """Telemetry source that REPLAYS the episode's joints + lidar at `fps`,
    looping — a drop-in for MockSource so the whole GUI runs offline."""

    def __init__(self, state, episode, fps=15.0):
        self.state = state
        self.ep = episode
        self.fps = fps
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.state.source = "demo"
        self.state.torque = {"arms": True, "handL": True, "handR": True}
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            try:
                names, pos, vel, eff, scan = self.ep.state_frame(i)
                self.state.update_joints(names, pos, vel, eff)
                self.state.update_targets(names, pos)       # replay = perfectly tracked
                hands = [n for n in names if "hand" in n or "thumb" in n
                         or "index" in n or "middle" in n or "ring" in n or "pinky" in n]
                if hands:
                    self.state.update_temps(hands, [44.0] * len(hands))
                if scan:
                    self.state.scan = scan
            except Exception:
                pass
            i += 1
            self._stop.wait(1.0 / self.fps)
