"""data_collection.py — rosbag2 (MCAP) session recorder for MABEL demonstrations.

The iOS app's "Start collecting data" button triggers a `data_collect` wire
message; the teleop bridge (`sim_teleop_bridge.py`) drives a `BagRecorder`, which
spawns `ros2 bag record` on a curated topic set and writes one MCAP bag per
session under a LOCAL recordings directory (git-ignored — never pushed).

Why MCAP / rosbag2 rather than a bespoke format:
  * it is the canonical ROS 2 dataset container — self-describing (every message
    type + QoS is embedded), so a bag replays/introspects with zero side files;
  * it captures EVERYTHING on the wire (joints, commands, all cameras, depth,
    point clouds, the nvblox mesh, the 2-D map, tf, odom, control mode, …) with
    correct timestamps in one stream — exactly what long-horizon nav+manip
    learning needs;
  * `tools/bag_to_rerun.py` converts a bag → a Rerun `.rrd` for visualization, so
    the recording truth (MCAP) and the viewer (Rerun) stay cleanly separated.

This module owns ONLY the recording lifecycle (start/stop/status + the topic
list). It shells out to the `ros2 bag record` CLI (rather than the rosbag2_py
writer) so that all message types — including ones whose Python typesupport is
not importable in this process (e.g. nvblox_msgs) — are captured byte-for-byte,
and bag splitting / compression come for free.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("data_collection")

# ── Curated topic set ────────────────────────────────────────────────────────
# Grouped by purpose so it is obvious what each block buys you and easy to trim.
# These are the canonical sim topics (mabel_sim interface_config.py /
# sensor_config.py) plus the SLAM/nav stack. Topics that are not currently
# published (e.g. the nvblox mesh when SLAM is off) are simply never recorded —
# rosbag2 subscribes to them if/when they appear, and ignores them otherwise.

PROPRIOCEPTION = [
    "/joint_states",            # full-robot measured pose (sensor_msgs/JointState)
    "/arms/joint_states",       # 14-DOF arms measured
    "/body/joint_states",       # torso + 3-DOF neck + lift measured
    "/left_hand/joint_states",  # ORCA left hand measured
    "/right_hand/joint_states", # ORCA right hand measured
    "/odom",                    # wheel odometry (nav_msgs/Odometry)
    "/odom_ground_truth",       # MuJoCo ground-truth base pose (sim only)
    "/camera/front/imu",        # front D435i IMU (sensor_msgs/Imu, 200 Hz)
]

# The human teleop COMMANDS — the supervision signal for imitation learning.
COMMANDS = [
    "/mabel_cmd",               # swerve base + lift target (mabel_interfaces/MabelSwerveCmd)
    "/arms/command",            # arm joint targets (sensor_msgs/JointState)
    "/body/command",            # torso/neck targets
    "/left_hand/command",       # ORCA left targets
    "/right_hand/command",      # ORCA right targets
    "/goal_pose",               # Nav2 goal taps (geometry_msgs/PoseStamped)
    "/cmd_vel",                 # Nav2 base twist (autonomous segments)
]

CONTROL_STATUS = [
    "/mabel/control_mode",      # std_msgs/String: navigation|arms|wrist|compliance…
    "/mabel/body_ready",        # std_msgs/Bool: arms tucked / ready for nav
]

# Frames — without /tf the whole dataset is spatially incoherent, so always on.
TRANSFORMS = [
    "/tf",
    "/tf_static",
    "/clock",                   # sim-time authority (rosgraph_msgs/Clock)
]

# Head + front-base RGB-D (the user explicitly wants depth here) + the SLAM IR
# stereo pair, plus their intrinsics so depth back-projects in 3-D.
CAMERAS_PRIMARY = [
    "/camera/front/color/image_raw",       "/camera/front/color/camera_info",
    "/camera/front/depth/image_rect_raw",  "/camera/front/depth/camera_info",
    "/camera/front/depth/points",          # colored cloud (nvblox input)
    "/camera/front/infra1/image_rect_raw", "/camera/front/infra1/camera_info",
    "/camera/front/infra2/image_rect_raw", "/camera/front/infra2/camera_info",
    "/camera/zed/left/image_raw",          "/camera/zed/left/camera_info",
    "/camera/zed/right/image_raw",         "/camera/zed/right/camera_info",
    "/camera/zed/depth/image_rect_raw",    "/camera/zed/depth/camera_info",
    "/camera/zed/depth/points",            # head stereo cloud
]

# Manipulation + situational-awareness views (wrist eyes, chest, rear).
CAMERAS_AUX = [
    "/camera/left_wrist/image_raw",        "/camera/left_wrist/camera_info",
    "/camera/right_wrist/image_raw",       "/camera/right_wrist/camera_info",
    "/camera/body/image_raw",              "/camera/body/camera_info",
    "/camera/back/color/image_raw",        "/camera/back/color/camera_info",
    "/camera/back/depth/image_rect_raw",   "/camera/back/depth/camera_info",
]

# SLAM / mapping products (Isaac nvblox + slam_toolbox + Nav2).
MAPPING = [
    "/scan",                               # 360° LiDAR (sensor_msgs/LaserScan)
    "/map",                                # 2-D occupancy grid (nav_msgs/OccupancyGrid)
    "/map_costmap",                        # inflated nav cost field (visual SLAM)
    "/nvblox_node/mesh_pointcloud",        # 3-D env mesh as colored points
    "/nvblox_node/mesh_marker",            # 3-D env mesh as a triangle list
    "/plan",                               # planned path (nav_msgs/Path)
]

# The full default set. Override entirely with MABEL_BAG_TOPICS (space/comma sep),
# or set MABEL_BAG_ALL=1 to record every topic on the graph (`ros2 bag record -a`).
DEFAULT_TOPICS: list[str] = (
    TRANSFORMS + PROPRIOCEPTION + COMMANDS + CONTROL_STATUS
    + CAMERAS_PRIMARY + CAMERAS_AUX + MAPPING
)


def resolve_topics() -> tuple[list[str], bool]:
    """(topics, record_all). Honors MABEL_BAG_ALL / MABEL_BAG_TOPICS overrides."""
    if os.getenv("MABEL_BAG_ALL", "").strip() in ("1", "true", "yes"):
        return [], True
    override = os.getenv("MABEL_BAG_TOPICS", "").strip()
    if override:
        toks = [t for t in re.split(r"[\s,]+", override) if t]
        return toks, False
    # de-dup while preserving order
    seen, out = set(), []
    for t in DEFAULT_TOPICS:
        if t not in seen:
            seen.add(t); out.append(t)
    return out, False


def default_recordings_dir() -> Path:
    """Local, git-ignored. Override with MABEL_RECORDINGS_DIR."""
    env = os.getenv("MABEL_RECORDINGS_DIR")
    if env:
        return Path(env).expanduser()
    # repo root = two levels up from this file (…/MABEL/server/data_collection.py)
    return Path(__file__).resolve().parent.parent / "recordings"


def _slugify(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (label or "").strip()).strip("_")
    return s[:48] or "session"


def list_sessions(root: Path | None = None, limit: int = 50) -> dict:
    """Enumerate recorded bag sessions for the :9600 collection message.

    Cheap by design (called ~1 Hz when a telemetry client is watching):
    one directory scan; size = the session dir's top-level files only (the
    MCAP chunks live there — no deep walk). Newest first, capped at `limit`.
    """
    root = Path(root or default_recordings_dir())
    sessions: list[dict] = []
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.name, reverse=True)
    except OSError:
        dirs = []
    for d in dirs[:limit]:
        meta = {}
        try:
            mf = d / "session.json"
            if mf.exists():
                meta = json.loads(mf.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        try:
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        except OSError:
            size = 0
        sessions.append({
            "name": d.name,
            "label": meta.get("label"),
            "startedIso": meta.get("started_iso"),
            "durationSec": meta.get("duration_sec"),
            "sizeBytes": size,
        })
    return {"dir": str(root), "count": len(sessions), "sessions": sessions}


class BagRecorder:
    """Owns at most one live `ros2 bag record` child. Start/stop are idempotent.

    Driven entirely from the bridge's sim thread; no locking needed because every
    method is called from that one thread. `status()` returns a small dict the
    bridge folds into the robot-state broadcast so the app can show a REC badge.
    """

    def __init__(self, recordings_dir: Path | None = None):
        self.root = Path(recordings_dir or default_recordings_dir())
        self._proc: subprocess.Popen | None = None
        self._cur_dir: Path | None = None
        self._label: str = ""
        self._t_start: float = 0.0
        self._error: str | None = None
        self._sessions = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, label: str = "session") -> str | None:
        """Begin a recording. Returns None on success, or an error string."""
        if self.active:
            log.info("data-collect: already recording → ignoring start")
            return None
        if shutil.which("ros2") is None:
            self._error = "ros2 CLI not found on PATH (is the ROS env sourced?)"
            log.error("data-collect: %s", self._error)
            return self._error

        topics, record_all = resolve_topics()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(label)
        out_dir = self.root / f"{stamp}_{slug}"
        try:
            out_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self._error = f"cannot create recordings dir: {e}"
            log.error("data-collect: %s", self._error)
            return self._error

        cmd = ["ros2", "bag", "record",
               "-o", str(out_dir),
               "--storage", "mcap",
               # zstd chunk compression: ~3-5x smaller bags, negligible CPU.
               "--storage-preset-profile", "zstd_fast"]
        cmd += ["-a"] if record_all else topics

        log.info("data-collect: starting bag → %s (%s)",
                 out_dir, "ALL topics" if record_all else f"{len(topics)} topics")
        try:
            # New process group so we can SIGINT the whole `ros2 bag record` tree
            # (it spawns the recorder as a child) for a clean MCAP finalize.
            self._proc = subprocess.Popen(cmd, start_new_session=True)
        except OSError as e:
            self._error = f"failed to launch ros2 bag record: {e}"
            log.error("data-collect: %s", self._error)
            self._proc = None
            return self._error

        self._cur_dir = out_dir
        self._label = slug
        self._t_start = time.time()
        self._error = None
        # Detect an immediate failure (bad topic spec, mcap plugin missing, …)
        # so the app gets an error instead of a silent no-op. `ros2 bag record`
        # creates `out_dir` itself; we must NOT pre-create it.
        time.sleep(0.4)
        if self._proc.poll() is not None:
            self._error = f"ros2 bag record exited immediately (code {self._proc.returncode})"
            log.error("data-collect: %s", self._error)
            self._proc = None
            self._cur_dir = None
            return self._error
        self._write_session_meta(out_dir, slug, topics, record_all, cmd)
        return None

    def stop(self) -> None:
        """End the live recording, finalizing the MCAP + metadata.yaml cleanly."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            log.info("data-collect: stopping bag (%s)", self._cur_dir)
            try:
                # SIGINT the process group → rosbag2 flushes chunks + writes
                # metadata.yaml. terminate() would orphan a partial bag.
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                log.warning("data-collect: bag did not exit in 10s → terminating")
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                except Exception:
                    self._proc.terminate()
        if self._cur_dir is not None:
            self._finalize_session_meta(self._cur_dir)
            self._sessions += 1
            log.info("data-collect: saved session → %s", self._cur_dir)
        self._proc = None
        self._cur_dir = None

    def toggle(self, active: bool, label: str = "session") -> str | None:
        return self.start(label) if active else (self.stop() or None)

    # ── status / metadata ────────────────────────────────────────────────────
    def status(self) -> dict:
        # Detect a child that died on its own (bad topic spec, disk full, …).
        if self._proc is not None and self._proc.poll() is not None and self._proc.returncode not in (0, -2):
            self._error = self._error or f"ros2 bag record exited ({self._proc.returncode})"
            self._proc = None
            self._cur_dir = None
        return {
            "active": self.active,
            "label": self._label if self.active else None,
            "dir": str(self._cur_dir) if (self.active and self._cur_dir) else None,
            "elapsedSec": round(time.time() - self._t_start, 1) if self.active else 0.0,
            "sessions": self._sessions,
            "error": self._error,
        }

    def _write_session_meta(self, d: Path, label, topics, record_all, cmd) -> None:
        meta = {
            "label": label,
            "started_iso": datetime.now().isoformat(timespec="seconds"),
            "started_unix": self._t_start,
            "record_all": record_all,
            "topics": topics,
            "ros_distro": os.getenv("ROS_DISTRO"),
            "git_commit": _git_sha(),
            "command": cmd,
            "host": os.uname().nodename,
        }
        try:
            # ros2 bag record OWNS `d` (it errors if `d` pre-exists), so write the
            # meta beside it as a sidecar; stop() folds it into `d` at the end.
            sidecar = d.parent / f"{d.name}.session.json"
            sidecar.write_text(json.dumps(meta, indent=2))
        except OSError as e:
            log.warning("data-collect: could not write session meta: %s", e)

    def _finalize_session_meta(self, d: Path) -> None:
        sidecar = d.parent / f"{d.name}.session.json"
        try:
            meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta["ended_iso"] = datetime.now().isoformat(timespec="seconds")
        meta["duration_sec"] = round(time.time() - self._t_start, 1)
        try:
            # Move the sidecar INTO the finished bag dir for a tidy self-contained
            # session folder (the bag dir exists by now).
            target = (d / "session.json") if d.is_dir() else sidecar
            target.write_text(json.dumps(meta, indent=2))
            if target != sidecar and sidecar.exists():
                sidecar.unlink()
        except OSError as e:
            log.warning("data-collect: could not finalize session meta: %s", e)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


# ── Standalone CLI: test recording without the app/bridge ────────────────────
# Usage:  python3 data_collection.py [label]   → records until Ctrl-C.
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rec = BagRecorder()
    lbl = sys.argv[1] if len(sys.argv) > 1 else "manual"
    err = rec.start(lbl)
    if err:
        print("ERROR:", err); sys.exit(1)
    print(f"recording '{lbl}' → {rec.status()['dir']}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1.0)
            s = rec.status()
            print(f"\r  {s['elapsedSec']:6.1f}s  active={s['active']}", end="", flush=True)
            if not s["active"]:
                print("\nrecorder exited:", s["error"]); break
    except KeyboardInterrupt:
        print("\nstopping…")
        rec.stop()
        print("saved.")
