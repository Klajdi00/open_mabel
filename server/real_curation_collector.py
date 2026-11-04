"""real_curation_collector.py — REAL-robot data collection in the Data Curation
Studio's exact episode format, straight off the live ROS 2 graph.

This is the real-robot counterpart to ``curation_recorder.py`` (which renders the
SIM cameras). It is a standalone ``rclpy`` node — it does NOT need the teleop
bridge or a MuJoCo model. It latches the newest message on every sensor / command
topic and, on a fixed clock (default 15 Hz, index-aligned exactly like the studio
expects), appends one frame to a curation-format ``episode_XXXX.h5``:

    <root>/<client>/<TASK>_<NNN>_<YYYYMMDD>_<HHMMSS>/episode_XXXX.h5   (+ dataset_meta.json)

with  <root> = <repo>/real  (or --root),  <client> ∈ {ios, vision_os, real}.
That leaf folder IS a Data-Curation dataset — drag it into the studio unchanged.

Design goals (in priority order):
  1. ROBUST TO OFFLINE COMPONENTS. Every topic is optional. A missing camera,
     a dead hand, no lidar, no SLAM — the episode still records with whatever is
     live. Nothing here raises out of a callback or the sample timer.
  2. SMALL ON DISK. RGB → JPEG at reduced resolution; depth → uint16 millimetres
     at reduced resolution (gzip'd); NO raw point clouds (the studio rebuilds
     clouds from depth + intrinsics — we store the intrinsics in the schema).
     Target ≈ 100–500 MB per minute for the full camera set (see DATA_COLLECTION).
  3. STUDIO-BYTE-COMPATIBLE. Frames are written by the SAME HDF5 writer the IL /
     curation pipeline uses (robot_learning/imitation_learning/data_collection/recorder.py),
     so the keys/attrs match ``curation_recorder.py`` exactly.

Control is over a tiny WebSocket that speaks the app's ``record`` wire shape, so a
mock client (tools/mock_vp_collect.py) drives it exactly like the Vision Pro /
iOS "Start collecting data" button:  {"type":"record","active":true,"task":..,
"client":"visionos"}.  Also: {"type":"select"|"create","task":..} and
{"type":"list"} for dataset-folder pick/create.

Run:
    source /opt/ros/jazzy/setup.bash && source ros/install/setup.bash
    ~/miniconda3/envs/mabel/bin/python3 server/real_curation_collector.py \
        --root ~/open_mabel/real --client vision_os --control-port 9095
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger("real_collect")

# ── curation config + HDF5 writer (loaded by FILE PATH, same trick as
#    curation_recorder.py: the name `data_collection` is taken by our BagRecorder). ─
_HERE = os.path.dirname(os.path.abspath(__file__))
_IL = os.path.normpath(os.path.join(_HERE, "..", "robot_learning", "imitation_learning"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DC = _load("_mabel_dc_config", os.path.join(_IL, "data_collection", "config.py"))
_H5Writer = _load("_mabel_dc_recorder",
                  os.path.join(_IL, "data_collection", "recorder.py")).EpisodeRecorder

# ── Canonical 59-actuator order (the studio's observation.state / action order) ──
ACTUATOR_GROUPS = [
    ("base",       list(_DC.BASE_ACTUATORS)),   # 6
    ("lift_torso", list(_DC.LIFT_TORSO)),        # 2
    ("neck",       list(_DC.NECK)),              # 3
    ("left_arm",   list(_DC.LEFT_ARM)),          # 8
    ("right_arm",  list(_DC.RIGHT_ARM)),         # 8
    ("left_hand",  list(_DC.LEFT_HAND)),         # 16
    ("right_hand", list(_DC.RIGHT_HAND)),        # 16
]
ACTUATOR_NAMES: list[str] = [n for _, names in ACTUATOR_GROUPS for n in names]
NU = len(ACTUATOR_NAMES)                          # 59


def _norm(name: str) -> str:
    """Normalize a joint name for cross-naming matching: lower, strip spaces/_/-."""
    return re.sub(r"[\s_\-]+", "", str(name).strip().lower())


# normalized canonical name -> canonical index (auto-matches urdf names like
# "left_arm_1" <-> "Left Arm 1", "fl_steer" <-> "FL Steer", "torso" <-> "Torso").
_CANON_IDX = {_norm(n): i for i, n in enumerate(ACTUATOR_NAMES)}

# ORCA hand joints publish short physical names; map them onto the canonical
# finger names (a SIDE prefix — "Left "/"Right " — is added from the topic).
# VERIFY against live /left_hand/joint_states .name once the HAL is up; unmatched
# names just land as NaN in that slot (robust) and are logged once.
_HAND_ALIAS = {
    "thumb_mcp": "Thumb MCP", "thumb_mp": "Thumb MCP", "thumb_abd": "Thumb Abd",
    "thumb_pip": "Thumb PIP", "thumb_ip": "Thumb PIP", "thumb_dip": "Thumb DIP",
    "index_abd": "Index Abd", "index_mcp": "Index MCP", "index_pip": "Index PIP",
    "middle_abd": "Middle Abd", "middle_mcp": "Middle MCP", "middle_pip": "Middle PIP",
    "ring_abd": "Ring Abd", "ring_mcp": "Ring MCP", "ring_pip": "Ring PIP",
    "pinky_abd": "Pinky Abd", "pinky_mcp": "Pinky MCP", "pinky_pip": "Pinky PIP",
}
_HAND_ALIAS_NORM = {_norm(k): v for k, v in _HAND_ALIAS.items()}


_LIFT_IDX = _CANON_IDX[_norm("Lift")]
# The URDF splits the single "Lift" actuator into two prismatic stages
# (lift_lower + lift_upper, each = Lift/2); fold both back onto the Lift slot.
_LIFT_STAGES = {"liftlower", "liftupper"}


def _canon_index(name: str, side: str | None = None) -> int:
    """Canonical actuator index for a raw ROS joint name, or -1 if unmatched.
    ``side`` ('left'/'right') is prefixed for bare finger names."""
    n = _norm(name)
    if n in _LIFT_STAGES:
        return _LIFT_IDX
    if n in _CANON_IDX:
        return _CANON_IDX[n]
    if side and n in _HAND_ALIAS_NORM:
        return _CANON_IDX.get(_norm(f"{side} {_HAND_ALIAS_NORM[n]}"), -1)
    # already-sided finger name e.g. "left thumb mcp"
    return _CANON_IDX.get(n, -1)


# ── ROS camera topic -> curation camera name (matches the sim CAMERAS names so a
#    sim-trained policy transfers). Each entry: color topic, depth topic (opt). ──
CAMERA_TOPICS = [
    # curation name,   color image topic,                     depth image topic (opt)
    ("body_front_cam", "/camera/front/color/image_raw",   "/camera/front/depth/image_rect_raw"),
    ("zed_left",       "/camera/zed/left/image_raw",       "/camera/zed/depth/image_rect_raw"),
    ("zed_right",      "/camera/zed/right/image_raw",      None),
    ("l_wrist_cam",    "/camera/wrist_left/image_raw",     "/camera/wrist_left/depth/image_rect_raw"),
    ("r_wrist_cam",    "/camera/wrist_right/image_raw",    "/camera/wrist_right/depth/image_rect_raw"),
    # (body / back cams are not present on this robot — kept out; the collector
    #  tolerates missing cameras, so add them back here if a build gains them.)
]

# per-subsystem measured + command joint topics (all optional)
JOINT_STATE_TOPICS = [
    ("/arms/joint_states", None), ("/body/joint_states", None),
    ("/base/joint_states", None),
    ("/left_hand/joint_states", "left"), ("/right_hand/joint_states", "right"),
    ("/joint_states", None),          # aggregate (urdf-named) — fills any gaps
]
COMMAND_TOPICS = [
    ("/arms/command", None), ("/body/command", None),
    ("/left_hand/command", "left"), ("/right_hand/command", "right"),
]

_CLIENT_SUBDIR = {"visionos": "vision_os", "vision_os": "vision_os",
                  "ios": "ios", "ipados": "ios", "iphone": "ios"}


def _safe(name: str, fallback="default") -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    return s or fallback


class RealCurationCollector:
    """rclpy node: latch newest topic messages, sample at a fixed clock, write
    curation-format episodes. Thread-safe start/stop from the control server."""

    def __init__(self, node, args):
        self.node = node
        self.args = args
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        # ONE clock for the whole dataset — images, proprio AND action all land on
        # this grid (default 15 Hz). Do not record action faster than you deploy.
        self.fps = max(1.0, float(args.fps))
        # Aspect-preserving max long-side (quality-preserving, still compact); never
        # upscales. RGB ~640, depth ~320 by default.
        self.rgb_max = int(args.rgb_max)
        self.depth_max = int(args.depth_max)
        self.jpeg_q = int(args.jpeg_quality)
        self.hr_sidestream = bool(getattr(args, "hr_sidestream", True))

        # dataset root: <root>/<client>
        self.client = _CLIENT_SUBDIR.get((args.client or "").lower(), "real")
        self.root = os.path.abspath(os.path.expanduser(args.root))
        self.out_dir = os.path.join(self.root, self.client)
        os.makedirs(self.out_dir, exist_ok=True)

        # ── shared latched state (guarded by _lk; RLock because start() holds it
        #    while _schema() re-acquires it) ───────────────────────────────────
        self._lk = threading.RLock()
        self._rgb_msg: dict[str, object] = {}       # cam -> latest raw ROS Image (decoded at sample time)
        self._depth_msg: dict[str, object] = {}     # cam -> latest raw depth Image
        self._lidar: np.ndarray | None = None
        self._intr: dict[str, dict] = {}            # cam -> {fx,fy,cx,cy,w,h}
        self._jpos = np.full(NU, np.nan, np.float32)
        self._jvel = np.full(NU, np.nan, np.float32)
        self._jtau = np.full(NU, np.nan, np.float32)
        self._ctrl = np.full(NU, np.nan, np.float32)
        self._pose = {}                             # key -> [7] pos+quat(wxyz)
        self._btwist = np.zeros(3, np.float32)
        self._warned: set[str] = set()

        # High-rate (native-rate) proprio + action side-stream, timestamped — an
        # analysis-only archive; the TRAINING frames stay on the one 15 Hz clock.
        self._hr_prop_t: list = []
        self._hr_prop: list = []          # each: [jpos(59)|jvel(59)|jtau(59)] concat
        self._hr_act_t: list = []
        self._hr_act: list = []           # each: ctrl(59)

        # ── recording lifecycle ──────────────────────────────────────────────
        self.task = "default"
        self._recording = False
        self._writer: _H5Writer | None = None
        self._session_dir: str | None = None
        self._n = 0
        self._t_start = 0.0
        self._dropped = 0                 # grid ticks dropped because encoders lagged
        self._seen_topics: set[str] = set()

        # Decouple the fixed-rate SAMPLER (exact grid timing) from the ENCODER
        # (JPEG/depth, parallelized per-camera), so a heavy encode never skews the
        # clock. The sampler snapshots latest frames onto the grid and enqueues;
        # a single encoder thread drains in order (append order preserved) and
        # fans the per-camera JPEG work across a thread pool (cv2 releases the GIL).
        import queue as _queue
        from concurrent.futures import ThreadPoolExecutor
        self._q: _queue.Queue = _queue.Queue(maxsize=90)
        self._encpool = ThreadPoolExecutor(max_workers=int(getattr(args, "encoder_threads", 4)))
        self._shutdown = threading.Event()
        self._ep_done = threading.Event()
        self._sampler = threading.Thread(target=self._sample_loop, name="sampler", daemon=True)
        self._encoder = threading.Thread(target=self._encode_loop, name="encoder", daemon=True)

        self._subscribe()
        self._sampler.start()
        self._encoder.start()
        log.info("collector ready — root=%s client=%s fps=%.0f rgb<=%d depth<=%d hr=%s",
                 self.out_dir, self.client, self.fps, self.rgb_max, self.depth_max,
                 self.hr_sidestream)

    # ── subscriptions (best-effort sensor QoS for streams; reliable for state) ──
    def _subscribe(self):
        from sensor_msgs.msg import Image, LaserScan, JointState, CameraInfo
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
        state_qos = QoSProfile(depth=10)
        state_qos.reliability = ReliabilityPolicy.RELIABLE

        for cam, color, depth in CAMERA_TOPICS:
            self.node.create_subscription(
                Image, color, self._mk_rgb_cb(cam), qos_profile_sensor_data)
            info = color.rsplit("/", 1)[0] + "/camera_info"
            self.node.create_subscription(
                CameraInfo, info, self._mk_info_cb(cam), qos_profile_sensor_data)
            if depth:
                self.node.create_subscription(
                    Image, depth, self._mk_depth_cb(cam), qos_profile_sensor_data)
        self.node.create_subscription(LaserScan, "/scan", self._scan_cb,
                                      qos_profile_sensor_data)
        for topic, side in JOINT_STATE_TOPICS:
            self.node.create_subscription(
                JointState, topic, self._mk_js_cb(side), state_qos)
        for topic, side in COMMAND_TOPICS:
            self.node.create_subscription(
                JointState, topic, self._mk_cmd_cb(side), state_qos)
        for topic in ("/odom_laser", "/visual_slam/tracking/odometry", "/odom"):
            self.node.create_subscription(Odometry, topic, self._odom_cb, state_qos)

    def _note(self, topic):
        self._seen_topics.add(topic)

    def _mk_rgb_cb(self, cam):
        # HOT PATH: only stash the raw ROS message (no cv_bridge decode here) so the
        # light joint/command callbacks are never GIL-starved by image decoding. The
        # decode happens once per 15 Hz sample in the encoder, not per 30 Hz message.
        def cb(msg):
            self._rgb_msg[cam] = msg
            self._note(cam + ":rgb")
        return cb

    def _mk_depth_cb(self, cam):
        def cb(msg):
            self._depth_msg[cam] = msg
            self._note(cam + ":depth")
        return cb

    def _mk_info_cb(self, cam):
        def cb(msg):
            try:
                k = msg.k
                with self._lk:
                    self._intr[cam] = {"fx": float(k[0]), "fy": float(k[4]),
                                       "cx": float(k[2]), "cy": float(k[5]),
                                       "w": int(msg.width), "h": int(msg.height)}
            except Exception as e:
                self._warn(f"info {cam}", e)
        return cb

    def _scan_cb(self, msg):
        try:
            r = np.asarray(msg.ranges, np.float32)
            r[~np.isfinite(r)] = float(msg.range_max)    # no-hit = max range
            with self._lk:
                self._lidar = r
            self._note("/scan")
        except Exception as e:
            self._warn("scan", e)

    def _scatter(self, dst, names, values, side):
        lift_reset = False
        for nm, v in zip(names, values):
            idx = _canon_index(nm, side)
            if idx < 0:
                self._warned.add("js:" + nm)
                continue
            # lift_lower + lift_upper SUM into the single Lift slot. Reset the slot
            # on the first stage seen IN THIS message so it never grows across
            # callbacks; every other joint is a straight (idempotent) assignment.
            if idx == _LIFT_IDX and _norm(nm) in _LIFT_STAGES:
                if not lift_reset:
                    dst[idx] = 0.0
                    lift_reset = True
                dst[idx] += float(v)
            else:
                dst[idx] = float(v)

    def _mk_js_cb(self, side):
        def cb(msg):
            with self._lk:
                if msg.position:
                    self._scatter(self._jpos, msg.name, msg.position, side)
                if msg.velocity:
                    self._scatter(self._jvel, msg.name, msg.velocity, side)
                if msg.effort:
                    self._scatter(self._jtau, msg.name, msg.effort, side)
                # native-rate proprio side-stream (timestamped, merged 59-vector)
                if self._recording and self.hr_sidestream:
                    self._hr_prop_t.append(time.time())
                    self._hr_prop.append(np.concatenate([
                        np.nan_to_num(self._jpos), np.nan_to_num(self._jvel),
                        np.nan_to_num(self._jtau)]).astype(np.float32))
            self._note("js")
        return cb

    def _mk_cmd_cb(self, side):
        def cb(msg):
            with self._lk:
                if msg.position:
                    self._scatter(self._ctrl, msg.name, msg.position, side)
                if self._recording and self.hr_sidestream:
                    self._hr_act_t.append(time.time())
                    self._hr_act.append(np.nan_to_num(self._ctrl).astype(np.float32))
            self._note("cmd")
        return cb

    def _odom_cb(self, msg):
        try:
            p, o = msg.pose.pose.position, msg.pose.pose.orientation
            tw = msg.twist.twist
            with self._lk:
                self._pose["chassis"] = np.array(
                    [p.x, p.y, p.z, o.w, o.x, o.y, o.z], np.float32)
                self._btwist = np.array([tw.linear.x, tw.linear.y, tw.angular.z],
                                        np.float32)
            self._note("odom")
        except Exception as e:
            self._warn("odom", e)

    def _warn(self, what, e):
        key = f"{what}:{type(e).__name__}"
        if key not in self._warned:
            self._warned.add(key)
            log.warning("channel %s: %s (further identical warnings suppressed)", what, e)

    # ── dataset folders (pick / create / list) ───────────────────────────────
    def list_datasets(self):
        try:
            return sorted(n for n in os.listdir(self.out_dir)
                          if os.path.isdir(os.path.join(self.out_dir, n)))
        except OSError:
            return []

    def _next_number(self, task):
        pat = re.compile(rf"^{re.escape(task)}_(\d+)_")
        best = 0
        for n in self.list_datasets():
            m = pat.match(n)
            if m:
                best = max(best, int(m.group(1)))
        return best + 1

    def _new_session_dir(self, task):
        task = _safe(task)
        num = self._next_number(task)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(self.out_dir, f"{task}_{num:03d}_{stamp}")
        os.makedirs(d, exist_ok=True)
        return d

    # ── recording control (called from the WS control thread) ─────────────────
    def start(self, task=None, client=None):
        with self._lk:
            if self._recording:
                return {"ok": False, "error": "already recording"}
            if client:
                sub = _CLIENT_SUBDIR.get(client.lower())
                if sub and sub != self.client:
                    self.client = sub
                    self.out_dir = os.path.join(self.root, sub)
                    os.makedirs(self.out_dir, exist_ok=True)
            self.task = _safe(task or self.task)
            self._session_dir = self._new_session_dir(self.task)
            self._writer = _H5Writer(self._session_dir, self._schema())
            self._writer.start()
            self._n = 0
            self._dropped = 0
            self._hr_prop_t.clear(); self._hr_prop.clear()
            self._hr_act_t.clear(); self._hr_act.clear()
            self._ep_done.clear()
            self._t_start = time.time()
            self._t0 = time.monotonic()          # the episode's grid origin
            self._k = 0
            self._recording = True
        log.info("REC ▶ %s → %s  (%.0f Hz, rgb<=%d)", self.task, self._session_dir,
                 self.fps, self.rgb_max)
        return {"ok": True, "dir": self._session_dir}

    def stop(self):
        with self._lk:
            if not self._recording:
                return {"ok": False, "error": "not recording"}
            self._recording = False
            hr = (list(self._hr_prop_t), list(self._hr_prop),
                  list(self._hr_act_t), list(self._hr_act))
        # flush the encoder queue (drain every sampled frame in order) before finishing
        self._q.put(self._SENTINEL)
        self._ep_done.wait(timeout=30.0)
        with self._lk:
            path = self._writer.finish(task=self.task) if self._writer else None
            n, sdir, drops = self._n, self._session_dir, self._dropped
        hr_path = self._write_hr_sidecar(path, hr) if (path and self.hr_sidestream) else None
        log.info("REC ■ %s (%d frames, %d dropped) → %s", self.task, n, drops, path)
        return {"ok": True, "episode": path, "frames": n, "dropped": drops, "dir": sdir,
                "hr_sidecar": hr_path, "fps": self.fps, "seen": sorted(self._seen_topics)}

    def _write_hr_sidecar(self, ep_path, hr):
        """Native-rate proprio+action archive next to episode_XXXX.h5 (analysis only,
        NOT read by training). Two independent timestamped streams."""
        try:
            pt, pv, at, av = hr
            side = ep_path[:-3] + "_hr.npz" if ep_path.endswith(".h5") else ep_path + "_hr.npz"
            np.savez_compressed(
                side,
                proprio_time=np.asarray(pt, np.float64),
                proprio=np.asarray(pv, np.float32) if pv else np.zeros((0, 3 * NU), np.float32),
                proprio_layout="[joint_pos(59) | joint_vel(59) | joint_torque(59)]",
                action_time=np.asarray(at, np.float64),
                action=np.asarray(av, np.float32) if av else np.zeros((0, NU), np.float32),
                actuator_names=np.asarray(ACTUATOR_NAMES))
            return side
        except Exception as e:
            self._warn("hr_sidecar", e)
            return None

    @staticmethod
    def _resize_max(img, max_side, interp):
        """Aspect-preserving downscale so the longer side == max_side. Never upscales."""
        h, w = img.shape[:2]
        s = max_side / float(max(h, w))
        if s >= 1.0:
            return img
        return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)

    def _schema(self):
        with self._lk:
            cams = []
            for cam, color, depth in CAMERA_TOPICS:
                intr = self._intr.get(cam)
                # report the aspect-preserving encoded size when we know the native one
                w = h = self.rgb_max
                if intr and intr.get("w") and intr.get("h"):
                    s = self.rgb_max / float(max(intr["w"], intr["h"]))
                    s = min(1.0, s)
                    w, h = round(intr["w"] * s), round(intr["h"] * s)
                cams.append({"name": cam, "rgb": True, "depth": bool(depth),
                             "w": int(w), "h": int(h), "intrinsics": intr})
        groups = {g: names for g, names in ACTUATOR_GROUPS}
        return {
            "source": "real",
            "recorded_by": f"real_curation_collector/{self.client}",
            "clock_hz": self.fps, "rgb_max_side": self.rgb_max, "depth_max_side": self.depth_max,
            "actuator_names": ACTUATOR_NAMES,
            "actuator_groups": groups,
            "cameras": cams,
            "lidar": {"n_rays": int(getattr(_DC, "LIDAR_N_RAYS", 360)),
                      "angle_res_deg": getattr(_DC, "LIDAR_ANGLE_RES_DEG", 1.0),
                      "units": "metres", "cutoff_is_no_hit": True},
            "depth_encoding": {"dtype": "uint16", "scale": _DC.DEPTH_SCALE,
                               "max_m": _DC.DEPTH_MAX_M, "units": "millimetres"},
            "hr_sidestream": self.hr_sidestream, "nq": 0, "nv": 0, "nu": NU,
        }

    # ── decoupled fixed-clock sampler + parallel encoder ─────────────────────
    _SENTINEL = ("__end__",)

    def _snapshot(self):
        # cheap: reference the latest raw image msgs (immutable once published) +
        # copy the tiny state vectors. NO image decode under the lock.
        with self._lk:
            return {
                "rgb_msg": dict(self._rgb_msg),
                "depth_msg": dict(self._depth_msg),
                "lidar": None if self._lidar is None else self._lidar.copy(),
                "jpos": self._jpos.copy(), "jvel": self._jvel.copy(),
                "jtau": self._jtau.copy(), "ctrl": self._ctrl.copy(),
                "poses": {k: v.copy() for k, v in self._pose.items()},
                "btwist": self._btwist.copy()}

    def _sample_loop(self):
        """Exact-grid sampler: one snapshot per 1/fps tick, independent of encode
        cost, so the whole dataset sits on ONE clock. Enqueues for the encoder."""
        while not self._shutdown.is_set():
            if not self._recording:
                time.sleep(0.005)
                continue
            target = self._t0 + self._k / self.fps
            dt = target - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            if not self._recording:
                continue
            snap = self._snapshot()
            try:
                self._q.put_nowait((self._k, time.time(), snap))
            except Exception:                              # queue full → encoder lagging
                self._dropped += 1
            self._k += 1

    def _encode_frame(self, snap):
        obs, act = {}, {}
        def _enc_rgb(cam_msg):
            cam, msg = cam_msg
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception:
                return (cam, None)
            small = self._resize_max(img, self.rgb_max, cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            return (cam, jpg.flatten()) if ok else (cam, None)
        for cam, blob in self._encpool.map(_enc_rgb, list(snap["rgb_msg"].items())):
            if blob is not None:
                obs[f"obs/cam/{cam}/rgb"] = blob
        for cam, msg in snap["depth_msg"].items():
            try:
                d = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
            except Exception:
                continue
            m = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)
            small = self._resize_max(m, self.depth_max, cv2.INTER_NEAREST)
            mm = np.clip(np.nan_to_num(small, nan=0.0, posinf=_DC.DEPTH_MAX_M, neginf=0.0),
                         0.0, _DC.DEPTH_MAX_M) * _DC.DEPTH_SCALE
            obs[f"obs/cam/{cam}/depth"] = mm.astype(np.uint16)
        if snap["lidar"] is not None:
            obs["obs/lidar"] = snap["lidar"].astype(np.float32)
        obs["obs/joint_pos"] = np.nan_to_num(snap["jpos"], nan=0.0)
        obs["obs/joint_vel"] = np.nan_to_num(snap["jvel"], nan=0.0)
        obs["obs/joint_torque"] = np.nan_to_num(snap["jtau"], nan=0.0)
        for key, pose in snap["poses"].items():
            obs[f"obs/pose/{key}"] = pose.astype(np.float32)
        obs["obs/base_twist"] = snap["btwist"].astype(np.float32)
        act["action/ctrl"] = np.nan_to_num(snap["ctrl"], nan=0.0)
        return obs, act

    def _encode_loop(self):
        """Single ordered consumer (append order == grid order); per-frame camera
        JPEGs fan out across the thread pool (cv2 releases the GIL)."""
        while not self._shutdown.is_set():
            item = self._q.get()
            if item is self._SENTINEL:
                self._ep_done.set()
                continue
            k, t, snap = item
            try:
                obs, act = self._encode_frame(snap)
                with self._lk:
                    if self._writer is not None:
                        self._writer.append(obs, act, sim_time=float(t), wall_time=t)
                        self._n += 1
            except Exception as e:
                self._warn("encode", e)

    def status(self):
        with self._lk:
            return {"recording": self._recording, "task": self.task,
                    "frames": self._n, "dir": self._session_dir,
                    "client": self.client, "datasets": self.list_datasets(),
                    "elapsed": round(time.time() - self._t_start, 1) if self._recording else 0.0,
                    "seen": sorted(self._seen_topics)}


# ── WebSocket control server: speaks the app's `record` wire shape ────────────
async def _control_server(collector, host, port, stop_evt):
    import websockets

    async def handler(ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send(json.dumps({"ok": False, "error": "bad json"}))
                continue
            t = msg.get("type")
            if t == "record":
                res = (collector.start(msg.get("task"), msg.get("client"))
                       if msg.get("active", True) else collector.stop())
            elif t in ("select", "create"):
                collector.task = _safe(msg.get("task") or collector.task)
                if t == "create":
                    os.makedirs(collector.out_dir, exist_ok=True)
                res = {"ok": True, "task": collector.task}
            elif t in ("list", "status"):
                res = {"ok": True, **collector.status()}
            else:
                res = {"ok": False, "error": f"unknown type {t!r}"}
            await ws.send(json.dumps(res))

    async with websockets.serve(handler, host, port):
        log.info("control WS on ws://%s:%d  (record/select/create/list/status)", host, port)
        await stop_evt.wait()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The MABEL HAL/ROS stack runs on CycloneDDS; FastDDS (rclpy's default) will NOT
    # interoperate — the collector would silently receive no joint_states/commands.
    # Default to Cyclone to match, unless the caller pinned something else.
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("MABEL_REAL_ROOT",
                    os.path.join(os.path.dirname(_HERE), "real")))
    ap.add_argument("--client", default=os.getenv("MABEL_COLLECT_CLIENT", "real"),
                    help="visionos|ios|real — routes into <root>/{vision_os,ios,real}")
    ap.add_argument("--fps", type=float, default=float(os.getenv("MABEL_COLLECT_FPS", "15")),
                    help="THE dataset clock (Hz) — images, proprio AND action all land here")
    ap.add_argument("--rgb-max", type=int, default=int(os.getenv("MABEL_RGB_MAX", "640")),
                    help="aspect-preserving RGB max long-side (never upscales)")
    ap.add_argument("--depth-max", type=int, default=int(os.getenv("MABEL_DEPTH_MAX", "320")))
    ap.add_argument("--jpeg-quality", type=int, default=int(os.getenv("MABEL_JPEG_Q", "88")))
    ap.add_argument("--hr-sidestream", action=argparse.BooleanOptionalAction,
                    default=os.getenv("MABEL_HR_SIDESTREAM", "1") not in ("0", "false", ""),
                    help="also log native-rate timestamped proprio+action to episode_XXXX_hr.npz")
    ap.add_argument("--encoder-threads", type=int, default=int(os.getenv("MABEL_ENC_THREADS", "4")))
    ap.add_argument("--control-host", default="127.0.0.1")
    ap.add_argument("--control-port", type=int, default=int(os.getenv("MABEL_COLLECT_PORT", "9095")))
    args = ap.parse_args()

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init()
    node = rclpy.create_node("real_curation_collector")
    collector = RealCurationCollector(node, args)
    exe = MultiThreadedExecutor()
    exe.add_node(node)

    spin = threading.Thread(target=exe.spin, name="rclpy-spin", daemon=True)
    spin.start()

    stop_evt = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_evt = asyncio.Event()
    try:
        loop.run_until_complete(_control_server(
            collector, args.control_host, args.control_port, stop_evt))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if collector._recording:
                collector.stop()
        finally:
            if collector._writer:
                collector._writer.close()
            exe.shutdown()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
