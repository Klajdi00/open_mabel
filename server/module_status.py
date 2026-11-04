"""
module_status.py — "which part of MABEL is actually connected right now?"

The apps used to show battery + latency and nothing else, so a half-plugged robot
looked identical to a healthy one: the arm just didn't move. This builds the
per-module connection roster that rides `robot_state.modules`, so the operator can
see — before driving — that the right arm is up and the left hand is not.

THE MODULES (fixed order; the app renders them in this order):

    base · torso · lift · neck · left/right arm · left/right hand
    sensors: head cam · front cam · wrist-L cam · wrist-R cam · lidar

WHAT "CONNECTED" MEANS, PER WORLD — this is the whole design problem:

  REAL  A subsystem node publishes NOTHING when its device is absent:
        `arms_node._publish()` bails on `if not names: return` (mabel_hw/ros/
        arms_node.py), and `MabelHardware.read()` returns None for a subsystem
        that is disabled, unplugged, or degraded. So TOPIC LIVENESS is a true
        connectivity signal, not a guess: a module is connected iff one of its
        joints arrived on its subsystem's /joint_states within TTL. No new
        hardware plumbing is needed — the topics already exist.

  SIM   There is no device to unplug: every joint in the MJCF is, by
        definition, connected. Reporting "sim" honestly beats faking a link
        state. What IS genuinely variable in sim is the SENSORS — `--head-only`
        serves one camera, the scene may lack a camera body, the lidar only
        sweeps if mj_multiRay runs — so those are measured, not assumed.

  BOTH  Cameras come from the video server's /cameras.json roster (one endpoint,
        shared by sim_camera_server and real_camera_server — both build on
        mjpeg_server.MJPEGServer). A camera is DOWN when it is not served at all,
        or when it errors, or when it has a viewer but produces no frames.
        A camera nobody is watching is IDLE, not down: the sim renders on demand,
        so "no frames" for an unwatched camera is correct behaviour.

The block is attached to robot_state only when it CHANGES (like lidar/motorHealth),
so the 50 Hz loop stays light.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

__all__ = ["MODULES", "ModuleTracker", "SimModuleSource", "RosModuleSource",
           "camera_roster"]

# ── the roster (id, label, group). Order is the app's render order. ──────────
MODULES: list[dict] = [
    {"id": "base",         "label": "Base",        "group": None},
    {"id": "torso",        "label": "Torso",       "group": None},
    {"id": "lift",         "label": "Lift",        "group": None},
    {"id": "neck",         "label": "Neck",        "group": None},
    {"id": "leftArm",      "label": "Left Arm",    "group": None},
    {"id": "rightArm",     "label": "Right Arm",   "group": None},
    {"id": "leftHand",     "label": "Left Hand",   "group": None},
    {"id": "rightHand",    "label": "Right Hand",  "group": None},
    {"id": "camHead",      "label": "Head Cam",    "group": "sensors"},
    {"id": "camFront",     "label": "Front Cam",   "group": "sensors"},
    {"id": "camWristLeft", "label": "L Wrist Cam", "group": "sensors"},
    {"id": "camWristRight", "label": "R Wrist Cam", "group": "sensors"},
    {"id": "lidar",        "label": "LiDAR",       "group": "sensors"},
    # AUDIO, not "sensors": the speaker is an OUTPUT — grouping it under sensors
    # made "6/7 sensors up" count a thing that emits with the things that observe.
    # One physical device (the USB sound card) but TWO rows: an operator needs
    # to know which HALF is dark. The speaker can be perfectly healthy while a
    # pulled mic plug (or, on the old bar, its mute button) makes the mic send
    # digital silence — one row would report that as "audio down" and send
    # someone hunting a software bug.
    {"id": "speaker",      "label": "Speaker",     "group": "audio"},
    {"id": "mic",          "label": "Microphone",  "group": "audio"},
]
_ORDER = [m["id"] for m in MODULES]
_META = {m["id"]: m for m in MODULES}

# Camera stream name (as served on :8080 and in /cameras.json) → module id.
CAM_TO_MODULE = {
    "main": "camHead", "front": "camFront",
    "wrist_left": "camWristLeft", "wrist_right": "camWristRight",
}

# REAL robot: which joints belong to which module, and which topic carries them
# (mabel_hw/config.py — LEFT_ARM/RIGHT_ARM/NECK/TORSO joints + topic_joint_states).
# A module is connected iff ANY of its joints arrived within TTL.
JOINT_TO_MODULE: dict[str, str] = {}
for _j in ("l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7"):
    JOINT_TO_MODULE[_j] = "leftArm"
for _j in ("r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7"):
    JOINT_TO_MODULE[_j] = "rightArm"
for _j in ("motor_L1", "motor_L2", "motor_head"):
    JOINT_TO_MODULE[_j] = "neck"
JOINT_TO_MODULE["torso"] = "torso"

# topic → the module(s) it can prove. Joint-name lookup refines the shared ones.
REAL_TOPICS: dict[str, tuple[str, ...]] = {
    os.getenv("MABEL_TOPIC_ARMS", "/arms/joint_states"): ("leftArm", "rightArm"),
    os.getenv("MABEL_TOPIC_BODY", "/body/joint_states"): ("neck", "torso"),
    os.getenv("MABEL_TOPIC_BASE", "/mabel/hw/base/joint_states"): ("base",),
    os.getenv("MABEL_TOPIC_LIFT", "/lift/joint_states"): ("lift",),
    os.getenv("MABEL_TOPIC_LHAND", "/left_hand/joint_states"): ("leftHand",),
    os.getenv("MABEL_TOPIC_RHAND", "/right_hand/joint_states"): ("rightHand",),
}

TTL = float(os.getenv("MABEL_MODULE_TTL", "3.0"))       # s since last evidence
HZ = float(os.getenv("MABEL_MODULE_HZ", "1.0"))         # refresh rate

_HAND_PARTS = ("thumb", "index", "middle", "ring", "pinky", "wrist")


def module_of(joint: str) -> "str | None":
    """Which module owns this joint — for ANY of the three naming schemes in play.

      driver / bus names   l_j1, r_j7, motor_head          (mabel_hw config)
      DDS + wire names     left_arm_1, left_thumb_mcp      (what the ROS nodes and
                                                            robot_state actually publish)
      MJCF names           "Left Arm 1", "FL Steer"        (the model)

    This matters more than it looks: the ROS nodes TRANSLATE driver names to DDS
    names before publishing (arms_node's drv_to_dds), so thor's /arms/joint_states
    carries `left_arm_1`, NOT `l_j1`. Resolving only the bus table would have
    reported both arms DOWN on a perfectly healthy robot. Normalising "_" to " "
    lets one set of rules cover the DDS and MJCF forms; the bus names go through
    the explicit table first.
    """
    mid = JOINT_TO_MODULE.get(joint)
    if mid:
        return mid
    j = joint.lower().replace("_", " ")
    if "steer" in j or "drive" in j or j == "base root":   return "base"
    if "lift" in j:                                        return "lift"
    if j == "torso":                                       return "torso"
    if j.startswith("neck"):                               return "neck"
    # Order matters: "left wrist" belongs to the HAND (the ORCA wrist roll), while
    # "left arm 7" is the arm — so test the hand parts before "arm".
    if j.startswith("left "):
        if any(p in j for p in _HAND_PARTS):   return "leftHand"
        if "arm" in j:                         return "leftArm"
    if j.startswith("right "):
        if any(p in j for p in _HAND_PARTS):   return "rightHand"
        if "arm" in j:                         return "rightArm"
    return None


class ModuleTracker:
    """Last-seen evidence per module → an ordered, JSON-ready snapshot.

    `mark()` records evidence; `snapshot()` expires anything older than TTL, so a
    subsystem that DIES goes stale on its own — nobody has to notice and clear it.
    Thread-safe: ROS callbacks and the poll thread both write.
    """

    def __init__(self, ttl: float = TTL) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._seen: dict[str, tuple[float, str]] = {}    # id -> (t, detail)
        self._down: dict[str, str] = {}                  # id -> why (sticky)

    def mark(self, mid: str, detail: str = "") -> None:
        """This module just proved it is alive."""
        with self._lock:
            self._seen[mid] = (time.time(), detail)
            self._down.pop(mid, None)

    def mark_joint(self, joint: str, detail: str = "") -> None:
        mid = JOINT_TO_MODULE.get(joint)
        if mid:
            self.mark(mid, detail)

    def mark_down(self, mid: str, why: str) -> None:
        """This module is known ABSENT (not merely quiet) — e.g. a camera the
        video server does not serve at all. Distinguishing 'absent' from 'stale'
        is what lets the app say *why* something is dark."""
        with self._lock:
            self._seen.pop(mid, None)
            self._down[mid] = why

    def snapshot(self) -> list[dict]:
        now = time.time()
        out = []
        with self._lock:
            for mid in _ORDER:
                meta = _META[mid]
                seen = self._seen.get(mid)
                if seen is not None and (now - seen[0]) <= self.ttl:
                    ok, detail = True, seen[1]
                elif seen is not None:
                    ok = False
                    detail = f"no data {int(now - seen[0])}s"
                else:
                    ok, detail = False, self._down.get(mid, "not connected")
                out.append({"id": mid, "label": meta["label"], "group": meta["group"],
                            "ok": ok, "detail": detail})
        return out



# ── torque ⟷ roster ──────────────────────────────────────────────────────────
# robot_state.torque is per-SUBSYSTEM; the roster is per-MODULE. A torque entry only
# means something when the hardware behind it is actually connected: the studio's
# rule is "energised ⇔ every REPORTED module is on" (control_center app.js torqueOn),
# so an ABSENT right hand that kept reporting False pinned the whole robot LIMP in the
# studio while the one connected (left) hand was holding torque — "the Torque ON
# button does nothing" with only a hand plugged into the Mac (2026-10-01). Absent ⇒
# null (unknown), never False. The same map gates the gateway's 2 s torque reconcile,
# so intent is not re-asserted forever at a module that is not there.
TORQUE_MODULES: dict[str, tuple[str, ...]] = {
    "left_hand": ("leftHand",), "right_hand": ("rightHand",),
    "arms": ("leftArm", "rightArm"), "neck": ("neck",), "base": ("base",),
}


def torque_module_present(key: str, ok: dict) -> bool:
    """False only when the roster is live (`ok` non-empty) AND every module behind
    torque key `key` is down. An empty `ok` (no module source yet / sim) means
    "unknown — assume present", so early boot and sim behave exactly as before."""
    if not ok:
        return True
    return any(ok.get(m, False) for m in TORQUE_MODULES.get(key, ()))


def torque_report(torque_state: dict, ok: dict) -> dict:
    """The wire's robot_state.torque: `torque_state` with absent modules nulled."""
    return {k: (v if torque_module_present(k, ok) else None)
            for k, v in torque_state.items()}


def roster_ok(snapshot: list) -> dict:
    """{module id: connected?} from ModuleTracker.snapshot()."""
    return {m["id"]: bool(m.get("ok")) for m in snapshot}

# ── cameras: the shared roster endpoint (sim + real video servers) ───────────

def camera_roster(url: str, timeout: float = 1.0) -> list[dict] | None:
    """GET /cameras.json. None when the video server is down/unreachable — which
    is itself meaningful: every camera is then unknown, not 'fine'."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("cameras", [])
    except Exception:                                          # noqa: BLE001
        return None


def _apply_cameras(tr: ModuleTracker, cams: list[dict] | None) -> None:
    if cams is None:
        for mid in CAM_TO_MODULE.values():
            tr.mark_down(mid, "video server down")
        return
    served = set()
    for c in cams:
        mid = CAM_TO_MODULE.get(c.get("name", ""))
        if not mid:
            continue                       # head_left/head_right aliases etc.
        served.add(mid)
        err = c.get("error")
        frames = int(c.get("frames") or 0)
        streaming = bool(c.get("streaming"))
        if err and not (frames == 0 and not streaming):
            tr.mark_down(mid, str(err))            # a real fault
        elif streaming and frames == 0:
            tr.mark_down(mid, "no frames")         # watched but producing nothing
        elif frames > 0:
            age = c.get("ageMs")
            tr.mark(mid, "live" if streaming else
                    (f"idle {int(age/1000)}s" if isinstance(age, int) else "idle"))
        else:
            # Served, healthy, nobody watching yet: the sim renders on demand, so
            # zero frames here is CORRECT — the camera is ready, not broken.
            tr.mark(mid, "ready")
    for mid in set(CAM_TO_MODULE.values()) - served:
        tr.mark_down(mid, "not served")


class SimModuleSource:
    """SIM: the actuator modules are whatever the MJCF has (nothing to unplug —
    reported honestly as simulated); the SENSORS are measured for real, because
    they genuinely vary (`--head-only`, a scene with no camera body, lidar off)."""

    def __init__(self, tracker: ModuleTracker, joints_getter, video_url: str,
                 lidar_getter, log=None) -> None:
        self.tr = tracker
        # LAZY on purpose: this source starts on the network thread, which comes up
        # BEFORE the physics loop has written its first joint snapshot. Reading the
        # names once at construction would see {} and report every limb dark.
        self.joints_getter = (joints_getter if callable(joints_getter)
                              else (lambda _j=list(joints_getter or []): _j))
        self.video_url = video_url
        self.lidar_getter = lidar_getter
        self.log = log
        self._sim_mods: dict[str, str] = {}
        self._stop = threading.Event()

    def _resolve(self) -> dict[str, str]:
        """Group the model's joints into modules. In sim these never drop out, so
        resolve once (when the joints first appear) and re-mark every tick to stay
        inside TTL."""
        if self._sim_mods:
            return self._sim_mods
        try:
            names = list(self.joints_getter() or [])
        except Exception:                                          # noqa: BLE001
            names = []
        if not names:
            return {}
        counts: dict[str, int] = {}
        for j in names:
            mid = module_of(j)
            if mid:
                counts[mid] = counts.get(mid, 0) + 1
        mods = {mid: f"sim · {n} joint{'s' if n != 1 else ''}"
                for mid, n in counts.items()}
        mods.setdefault("base", "sim · swerve")   # the base is a free joint, unnamed
        self._sim_mods = mods
        if self.log:
            self.log.info("module status: sim modules %s",
                          ", ".join(sorted(mods)))
        return mods

    # THREE naming schemes describe the same robot, and this has to handle all of
    # them. The MJCF is human-readable ("Left Arm 1", "Right Thumb MCP", "FL
    # Steer"); the WIRE snake-cases it ("left_arm_1", "left_thumb_mcp"); the real
    # robot publishes hardware bus names ("l_j1", "motor_head") — those go through
    # the explicit JOINT_TO_MODULE table, which is consulted first. Normalising
    # "_" to " " lets ONE set of rules cover the first two. Verified against all
    # 61 MJCF joints and all 54 wire joints.
    _HAND_PARTS = ("thumb", "index", "middle", "ring", "pinky", "wrist")

    @staticmethod
    def _infer(joint: str) -> str | None:
        j = joint.lower().replace("_", " ")
        if "steer" in j or "drive" in j or j == "base root":   return "base"
        if "lift" in j:                                        return "lift"
        if j == "torso":                                       return "torso"
        if j.startswith("neck"):                               return "neck"
        # Order matters: "left wrist" belongs to the HAND (the ORCA wrist roll),
        # while "left arm 7" is the arm — so test the hand parts before "arm".
        if j.startswith("left "):
            if any(p in j for p in SimModuleSource._HAND_PARTS):  return "leftHand"
            if "arm" in j:                                        return "leftArm"
        if j.startswith("right "):
            if any(p in j for p in SimModuleSource._HAND_PARTS):  return "rightHand"
            if "arm" in j:                                        return "rightArm"
        return None

    def start(self) -> "SimModuleSource":
        threading.Thread(target=self._run, name="module-status",
                         daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        period = 1.0 / max(HZ, 0.2)
        while not self._stop.is_set():
            for mid, detail in self._resolve().items():
                self.tr.mark(mid, detail)
            _apply_cameras(self.tr, camera_roster(self.video_url))
            scan = None
            try:
                scan = self.lidar_getter() if self.lidar_getter else None
            except Exception:                                  # noqa: BLE001
                scan = None
            if scan and (time.time() - float(scan.get("stamp", 0))) < TTL:
                n = len(scan.get("ranges") or ())
                self.tr.mark("lidar", f"{n} rays")
            else:
                self.tr.mark_down("lidar", "no sweeps")
            # No sound card behind the sim plant. mark_down (not silence) so the
            # app says WHY the rows are dark instead of implying a broken device.
            for _aid in ("speaker", "mic"):
                self.tr.mark_down(_aid, "no audio device in sim")
            self._stop.wait(period)


class RosModuleSource:
    """REAL: subscribe to each subsystem's /joint_states (+ /scan) and let TOPIC
    LIVENESS answer the question. A subsystem whose device is gone publishes
    nothing at all, so silence past TTL == disconnected — which is exactly what
    the operator wants to see. Runs on its OWN rclpy Context (the map-stream
    lesson: never share a DDS participant with a firehose)."""

    def __init__(self, tracker: ModuleTracker, video_url: str, log=None) -> None:
        self.tr = tracker
        self.video_url = video_url
        self.log = log
        self._stop = threading.Event()

    def start(self) -> "RosModuleSource | None":
        try:
            import rclpy  # noqa: F401
        except Exception as e:                                 # noqa: BLE001
            if self.log:
                self.log.warning("module status: rclpy unavailable (%s)", e)
            return None
        threading.Thread(target=self._run, name="module-status-ros",
                         daemon=True).start()
        threading.Thread(target=self._run_cams, name="module-status-cams",
                         daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run_cams(self) -> None:
        period = 1.0 / max(HZ, 0.2)
        while not self._stop.is_set():
            _apply_cameras(self.tr, camera_roster(self.video_url))
            self._stop.wait(period)

    def _on_audio_health(self, msg) -> None:
        """One bar, two rows: report the speaker and the mic separately.

        The failure that makes this worth the extra row: the bar's own mute
        button makes the mic emit digital silence while the speaker is perfect.
        Collapsing both into one "audio" row would show that as a dead device
        and send someone hunting a software bug that isn't there.
        """
        try:
            h = json.loads(msg.data) or {}
        except Exception:                                  # noqa: BLE001
            return
        if not h.get("connected"):
            why = "no sound card"
            self.tr.mark_down("speaker", why)
            self.tr.mark_down("mic", why)
            return
        spk = h.get("speaker") or {}
        mic = h.get("mic") or {}
        vol = spk.get("volumePct")
        # A speaker with nothing to say is READY, not down — the same idle-vs-down
        # rule the cameras follow ("a camera nobody is watching is IDLE").
        self.tr.mark("speaker", "playing" if spk.get("open")
                     else (f"ready ({vol}%)" if vol is not None else "ready"))
        if mic.get("muted"):
            self.tr.mark("mic", "muted")
        elif not mic.get("open"):
            self.tr.mark("mic", "ready")               # demand-driven: nobody listening
        else:
            rms = float(mic.get("rms") or 0.0)
            # Distinguish "capturing silence" from "capturing": on this bar,
            # digital silence with everything else green means its PHYSICAL mute
            # button is on — the one thing software cannot see or fix.
            self.tr.mark("mic", f"live ({rms:.3f})" if rms > 0 else "live (silent)")

    def _run(self) -> None:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        from sensor_msgs.msg import JointState, LaserScan
        from std_msgs.msg import String

        ctx = Context()
        rclpy.init(args=None, context=ctx)
        node = rclpy.create_node("mabel_module_status", context=ctx)
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = QoSDurabilityPolicy.VOLATILE

        def on_joints(msg, mods: tuple[str, ...]) -> None:
            names = list(msg.name or ())
            hit = False
            for n in names:
                mid = module_of(n)
                if mid:
                    n_j = sum(1 for x in names if module_of(x) == mid)
                    self.tr.mark(mid, f"{n_j} joint{'s' if n_j != 1 else ''}")
                    hit = True
            if not hit and names and len(mods) == 1:
                # Topic whose joint names we don't map (base wheels, hand fingers):
                # the topic itself only publishes when the device is alive, so its
                # arrival IS the evidence.
                self.tr.mark(mods[0], f"{len(names)} joints")

        for topic, mods in REAL_TOPICS.items():
            node.create_subscription(
                JointState, topic,
                (lambda m, _mods=mods: on_joints(m, _mods)), qos)
        node.create_subscription(
            LaserScan, os.getenv("MABEL_MAP_SCAN_TOPIC", "/scan"),
            lambda m: self.tr.mark("lidar", f"{len(m.ranges)} rays"), qos)
        # Audio liveness comes from device_health (@2 Hz), NOT from /audio/mic:
        # mic capture is DEMAND-DRIVEN, so a silent mic topic is the normal idle
        # state and would report a healthy bar as down. device_health publishes
        # whenever the subsystem is up, whoever is or isn't listening.
        node.create_subscription(
            String, os.getenv("MABEL_TOPIC_AUDIO_HEALTH", "/audio/device_health"),
            self._on_audio_health, 5)
        if self.log:
            self.log.info("module status: %d subsystem topics + /scan + audio "
                          "→ robot_state.modules", len(REAL_TOPICS))

        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        try:
            while not self._stop.is_set() and ctx.ok():
                ex.spin_once(timeout_sec=0.2)
        except Exception as e:                                 # noqa: BLE001
            if self.log:
                self.log.warning("module status: source stopped: %s", e)
        finally:
            ex.remove_node(node)
            node.destroy_node()
            try:
                rclpy.shutdown(context=ctx)
            except Exception:
                pass
