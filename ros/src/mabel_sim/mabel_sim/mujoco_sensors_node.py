#!/usr/bin/env python3
"""MuJoCo -> ROS 2 sensor bridge for MABEL (cameras + LiDAR).

This is a SECOND, render-only node that runs alongside the physics bridge
(``mujoco_node.py``).  Keeping it separate means:

  * the physics node keeps its GLFW on-screen viewer, while this node renders
    offscreen with EGL -- no GL-context fight inside one process;
  * the performance-tuned physics loop is left untouched.

How it stays in sync without touching the physics ``MjData``: it loads its OWN
``MjModel`` from the same scene file (identical qpos layout) and each render tick
reconstructs the pose from the topics the physics node already publishes --
``/joint_states`` (every hinge/slide joint, URDF-named) and ``/odom`` (the base
free joint) -- then calls ``mj_forward``.  That single call updates camera poses,
LiDAR ray poses, and rangefinder ``sensordata``, which we render and publish.

PARALLEL RENDER WORKERS (why the front cloud is no longer ~2 Hz)
---------------------------------------------------------------
The first version rendered ALL 11 cameras *and* did all the CPU post-processing
(depth noise, point-cloud packing, message serialisation, the 360-beam LiDAR
loop) SERIALLY in one GIL-bound thread.  The SLAM-critical front point cloud was
the last thing in that tick, so it surfaced in RViz at only ~2 Hz even though
each individual render is 4-6 ms.

Benchmarking on the 4090 (EGL, see ``scripts/bench_sensors.py``) showed:
  * a single front-D435 full pipeline sustains ~59 Hz, a full 11-camera serial
    tick ~17 Hz -- so GPU render is NOT the cap; the serial CPU work was;
  * adding more EGL contexts does NOT raise GPU throughput (the driver
    serialises contexts ~99 fps for 640x360) -- so one-thread-per-camera is
    pointless; a FEW workers at decoupled rates is the right shape.

So rendering now runs in THREE independent worker threads, each owning its own
GL context + ``MjData`` + rate, so the front camera never waits on the others
and CPU post-processing on one worker overlaps GPU render on another:

  * FRONT  (~30 Hz): front D435i color+depth+cloud + IR stereo   -- SLAM / nav
  * AUX    (~12 Hz): back D435i + body/wrist/ZED mono cameras     -- viewing
  * LIDAR  (~15 Hz): the 360-beam RPLIDAR A3 scan (vectorised)    -- nav / SLAM

Each worker BUILDS its renderers on its own thread (a MuJoCo Renderer's GL
context is bound to the thread that creates it and can only be made current
there).  The executor only services the GL-free pose callbacks, which write a
shared, lock-protected qpos snapshot the workers copy each tick.

Topics (sim namespaces, see ``sensor_config.py``):
  Front D435i : /camera/front/{color/image_raw, depth/image_rect_raw,
                depth/points, infra1/image_rect_raw, infra2/image_rect_raw}
  Back  D435i : /camera/back/{color/image_raw, depth/image_rect_raw, depth/points}
  ZED head    : /camera/zed/{left,right}/image_raw  (rgb)
                /camera/zed/{depth/image_rect_raw, depth/points}  (head stereo cloud)
  Wrists/body : /camera/{left_wrist,right_wrist,body}/image_raw
  LiDAR       : /scan
plus matching */camera_info and a base_link->*_optical_frame TF per camera.
"""

import array
import os
import threading
import time

# MUJOCO_GL MUST be chosen BEFORE `import mujoco`: mujoco binds its offscreen GL
# backend at import, so setting it afterwards silently falls back to SOFTWARE EGL
# (llvmpipe), ~100x slower per frame, while still reporting "egl". sim.launch.py
# sets this in the child env (authoritative); this setdefault only covers a bare
# `ros2 run` with nothing set, defaulting to headless-GPU EGL.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField, LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from mabel_sim import interface_config as cfg
from mabel_sim import sensor_config as scfg
from mabel_sim import sensor_render as sr


# GL CONTEXT CREATION / TEARDOWN IS NOT THREAD-SAFE.  Each render worker builds
# its own thread-bound GL context (EGL on Thor/4090, GLFW with a display, osmesa
# in software), but the underlying context init touches PROCESS-GLOBAL state --
# notably GLFW's X11 error handler (`_glfwGrabErrorHandlerX11`) and EGL's display
# bring-up.  Two workers constructing `mujoco.Renderer` (or freeing one) at the
# SAME instant race on that global state and abort the process -- the
# "Assertion `_glfw.x11.errorHandler == NULL' failed" / SIGSEGV we saw at start-up
# when the front + aux workers built concurrently.  Serialising ONLY the
# create/destroy with this one process-wide lock fixes it on every backend
# (Jetson included); the per-tick RENDER calls stay fully parallel, so throughput
# is unchanged.  Reproduced + verified: 4 threads building renderers crash without
# the lock, all succeed with it.
_GL_CTX_LOCK = threading.Lock()


class _RenderGroup(threading.Thread):
    """One render worker: owns a GL context, an ``MjData`` and a publish rate.

    Builds its renderers on its OWN thread (GL contexts are thread-bound), then
    loops ``tick_fn`` at ``rate`` Hz, pacing itself to wall time.  Each tick it
    pulls the latest pose snapshot from the node, runs ``mj_forward`` on its own
    ``MjData`` (so workers never share mutable physics state), and renders.
    """

    def __init__(self, node, name, rate, color_sizes, depth_sizes, tick_fn):
        super().__init__(daemon=True, name=f"render-{name}")
        self.node = node
        self.label = name
        self.rate = max(float(rate), 0.1)
        self._color_sizes = set(color_sizes)
        self._depth_sizes = set(depth_sizes)
        self._tick_fn = tick_fn
        self.color = {}                 # (w,h) -> color Renderer
        self.depth = {}                 # (w,h) -> depth Renderer
        self.data = None                # this worker's private MjData
        self.rng = np.random.default_rng()   # per-worker (np Generator isn't thread-safe)
        self.last_dt = 0.0              # last tick wall time (s); for telemetry
        # NOT `_stop`: threading.Thread has an internal _stop() method that join()
        # calls, and shadowing it with a bool crashes shutdown ('bool' not callable).
        self._stopping = False

    def _build(self):
        self.data = mujoco.MjData(self.node.model)
        mujoco.mj_forward(self.node.model, self.data)
        # Serialise GL-context CREATION across workers (see _GL_CTX_LOCK): building
        # a Renderer on two threads at once races on global GLFW/EGL state and
        # aborts the process. The lock is held only for the cheap constructor; the
        # render loop below runs unlocked and fully parallel.
        with _GL_CTX_LOCK:
            for (w, h) in self._color_sizes:
                self.color[(w, h)] = mujoco.Renderer(
                    self.node.model, height=h, width=w)
            for (w, h) in self._depth_sizes:
                r = mujoco.Renderer(self.node.model, height=h, width=w)
                r.enable_depth_rendering()
                self.depth[(w, h)] = r

    def _selftest_ms(self):
        """Time a few warm renders so a software-EGL fallback (10x slow) is obvious."""
        if not self.color:
            return 0.0
        r = next(iter(self.color.values()))
        for _ in range(3):                          # warm
            r.update_scene(self.data, camera=0, scene_option=self.node._vopt)
            r.render()
        t0 = time.perf_counter()
        for _ in range(5):
            r.update_scene(self.data, camera=0, scene_option=self.node._vopt)
            r.render()
        return (time.perf_counter() - t0) / 5 * 1000.0

    def run(self):
        try:
            self._build()
        except Exception as e:           # noqa: BLE001
            self.node.get_logger().error(
                f"[{self.label}] could not create '{self.node._backend}' renderer: {e}. "
                f"Try render_backend:=glfw (with a display) or osmesa (software).")
            return
        ms = self._selftest_ms()
        msg = f"[{self.label}] render worker online @ {self.rate:.0f} Hz ({ms:.1f} ms/render)"
        # A single small render should be ~1-2 ms on a GPU; >8 ms means EGL fell
        # back to CPU software rendering (see _check_gpu_egl) -- flag it once.
        if self.node._backend == "egl" and ms > 8.0:
            self.node.get_logger().warn(
                msg + " -- SLOW: likely Mesa SOFTWARE rendering, not the GPU; "
                "check __EGL_VENDOR_LIBRARY_DIRS.")
        else:
            self.node.get_logger().info(msg)
        period = 1.0 / self.rate
        while rclpy.ok() and not self._stopping:
            t0 = time.monotonic()
            # Wait for the first pose (and thus sim /clock) before publishing, so
            # we never emit a frame at the default pose with a zero timestamp.
            if not self.node._have_pose:
                time.sleep(period)
                continue
            try:
                self.node._apply_snapshot(self.data)
                self._tick_fn(self)
            except Exception as e:       # noqa: BLE001
                self.node.get_logger().warn(f"[{self.label}] render tick failed: {e}")
            self.last_dt = time.monotonic() - t0
            time.sleep(max(0.0, period - self.last_dt))

    def stop(self):
        self._stopping = True

    def close(self):
        # GL-context DESTRUCTION races on the same global state as creation, so it
        # takes the same lock (a concurrent free across workers segfaults at exit).
        with _GL_CTX_LOCK:
            for r in list(self.color.values()) + list(self.depth.values()):
                try:
                    r.close()
                except Exception:        # noqa: BLE001
                    pass


class MujocoSensorBridge(Node):

    def __init__(self):
        super().__init__("mabel_mujoco_sensors")

        # ---- parameters ----------------------------------------------------
        self.declare_parameter("scene", "")
        # Per-group rates. sensor_rate (legacy) overrides ALL three when >0.
        self.declare_parameter("sensor_rate", scfg.DEFAULT_RATE)      # 0 = use per-group
        self.declare_parameter("front_rate", scfg.DEFAULT_FRONT_RATE)
        self.declare_parameter("aux_rate", scfg.DEFAULT_AUX_RATE)
        self.declare_parameter("lidar_rate", scfg.DEFAULT_LIDAR_RATE)
        self.declare_parameter("render_backend", scfg.DEFAULT_BACKEND)
        self.declare_parameter("pointcloud_stride", 2)
        self.declare_parameter("publish_color", True)
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("publish_ir", True)
        self.declare_parameter("publish_lidar", True)
        # Render ONLY the front D435i (color+depth+IR) + LiDAR -- skips the ZED,
        # wrist, body and back cameras (the whole AUX worker never starts). Cuts
        # GPU/unified memory + render time, which keeps SLAM runs from OOM-killing
        # the sim on the Jetson, and matches "SLAM via the front RealSense only".
        self.declare_parameter("front_only", False)
        # Sensor noise, modelled on the real hardware (see sensor_render.py).
        # RPLIDAR A3: datasheet %-of-range accuracy; scale 1.0 = datasheet.
        self.declare_parameter("lidar_noise_scale", 1.0)
        # Footprint SELF-FILTER: the planar LiDAR sits ON the robot, so its rays
        # hit MABEL's own body/arms/lift (returns from ~0.13 m, well inside the
        # ~0.40 m footprint). Those are not real obstacles -- they'd otherwise be
        # mapped as a blob by SLAM and permanently trip Nav2's collision monitor.
        # Drop every return closer than this (m) to a no-return, exactly as a real
        # deployment masks the robot's own profile. Nothing external can be inside
        # the footprint anyway. Set 0 to disable.
        self.declare_parameter("lidar_min_range", 0.45)
        # D435 axial depth: std = base + k*z^2 (Ahn 2019 empirical ~4 cm @ 2 m).
        self.declare_parameter("depth_noise_base", 0.001)     # m, constant term
        self.declare_parameter("depth_noise_k", 0.005)        # m per m^2, quadratic
        self.declare_parameter("depth_noise_lat", 0.0005)     # lateral std = lat*z
        # Gross errors / outliers (robustness testing -- real failure modes).
        self.declare_parameter("lidar_dropout_prob", 0.02)    # beams -> no return
        self.declare_parameter("lidar_outlier_prob", 0.01)    # beams -> random range
        self.declare_parameter("depth_hole_prob", 0.03)       # pixels -> invalid
        self.declare_parameter("depth_outlier_prob", 0.005)   # pixels -> flying point

        scene_path = self.get_parameter("scene").value or cfg.DEFAULT_SCENE
        scene_path = os.path.expanduser(scene_path)
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"MuJoCo scene not found: {scene_path}")

        # Effective per-group rates: a >0 legacy sensor_rate pins all three.
        legacy = float(self.get_parameter("sensor_rate").value)
        self.front_rate = legacy if legacy > 0 else float(self.get_parameter("front_rate").value)
        self.aux_rate = legacy if legacy > 0 else float(self.get_parameter("aux_rate").value)
        self.lidar_rate = legacy if legacy > 0 else float(self.get_parameter("lidar_rate").value)

        self.pc_stride = int(self.get_parameter("pointcloud_stride").value)
        self.pub_color = bool(self.get_parameter("publish_color").value)
        self.pub_depth = bool(self.get_parameter("publish_depth").value)
        self.pub_ir = bool(self.get_parameter("publish_ir").value)
        self.pub_lidar = bool(self.get_parameter("publish_lidar").value)
        self.lidar_noise_scale = float(self.get_parameter("lidar_noise_scale").value)
        self.depth_noise_base = float(self.get_parameter("depth_noise_base").value)
        self.depth_noise_k = float(self.get_parameter("depth_noise_k").value)
        self.depth_noise_lat = float(self.get_parameter("depth_noise_lat").value)
        self.lidar_dropout_prob = float(self.get_parameter("lidar_dropout_prob").value)
        self.lidar_outlier_prob = float(self.get_parameter("lidar_outlier_prob").value)
        self.lidar_min_range = float(self.get_parameter("lidar_min_range").value)
        self.depth_hole_prob = float(self.get_parameter("depth_hole_prob").value)
        self.depth_outlier_prob = float(self.get_parameter("depth_outlier_prob").value)

        # Camera sets. front_only -> render only the front D435i + IR (+ LiDAR);
        # the AUX worker (back + mono) never starts.
        self.front_only = bool(self.get_parameter("front_only").value)
        # D435i RGB-D cams + the ZED head stereo-depth cloud (both flow through the
        # generic _do_depth path; the ZED carries publish_color/emit_tf flags).
        all_depth = list(scfg.DEPTH_CAMERAS) + list(scfg.STEREO_DEPTH_CAMERAS)
        self._mono_cams = [] if self.front_only else list(scfg.MONO_CAMERAS)
        self._ir_cams = list(scfg.IR_CAMERAS)        # already front-only
        if self.front_only:
            all_depth = [c for c in all_depth if c["ns"] == "front"]
        self._depth_cams = all_depth
        self._front_depth = [c for c in all_depth if c["ns"] == "front"]
        self._aux_depth = [c for c in all_depth if c["ns"] != "front"]

        # The GL backend is bound at `import mujoco` (top of file) from MUJOCO_GL in
        # the process env -- it CANNOT be changed now (a late os.environ write lands
        # rendering on slow software EGL). So trust the env that was actually used;
        # the render_backend param is just the fallback when MUJOCO_GL wasn't set.
        # sim.launch.py sets MUJOCO_GL (+ the NVIDIA EGL vendor/device) before exec.
        self._backend = os.environ.get(
            "MUJOCO_GL", str(self.get_parameter("render_backend").value))
        self._check_gpu_egl()

        self.get_logger().info(
            f"Loading sensor model: {scene_path}  (backend={self._backend})")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)      # template; pose source of truth
        mujoco.mj_forward(self.model, self.data)

        # ---- state reconstruction tables -----------------------------------
        self._build_state_maps()
        self._precompute_intrinsics()

        # cache static handles for the base + lidar frames
        self._chassis_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
        self._lidar_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, scfg.LIDAR_BODY)
        beams, lidar_chassis = sr.lidar_beams(self.model)
        # Precompute the static LiDAR beam table ONCE (angles/order are constant
        # because the LiDAR is rigid on the chassis) -- the live scan is then a
        # single vectorised gather instead of a 360-beam Python loop per tick.
        self._lidar_adr, self._lidar_amin, self._lidar_inc, self._lidar_cutoff = \
            sr.lidar_beam_table(self.model, self.data, beams, lidar_chassis)

        # ---- stereo baseline for the IR pair --------------------------------
        # cuVSLAM recovers METRIC scale from the right camera's projection term
        # P[3] = Tx = -fx*baseline; with Tx=0 it sees a zero baseline and its
        # odometry scale is wrong. Measure the baseline straight from the MJCF
        # camera placement (single source of truth) so it can't drift.
        self._ir_bx = self._compute_ir_baselines()

        # latest pose, as a lock-protected qpos snapshot the workers copy.
        self._qpos = self.data.qpos.copy()
        self._pose_lock = threading.Lock()
        self._have_pose = False

        # Realistic camera rendering: no rangefinder ray lines, no group-3
        # debug markers/sites bleeding into the images.
        self._vopt = sr.clean_render_option(3)

        # ---- ROS interfaces -------------------------------------------------
        self._setup_publishers()
        self.tf_broadcaster = TransformBroadcaster(self)
        # Pose subscriptions live in a REENTRANT group so they keep running while
        # renders are in flight, keeping the pose baked into each frame fresh.
        self._pose_cbg = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states",
                                 self._on_joint_states, qos_profile_sensor_data,
                                 callback_group=self._pose_cbg)
        # Render at the robot's GROUND-TRUTH pose, NOT the odometry estimate. A
        # simulated camera must see where the robot ACTUALLY is; rendering from the
        # estimate would make the view track wheel-odom drift and give cuVSLAM a
        # world consistent with that drift, so it could never correct it.
        self.create_subscription(Odometry, "/odom_ground_truth",
                                 self._on_odom, qos_profile_sensor_data,
                                 callback_group=self._pose_cbg)

        # ---- render workers -------------------------------------------------
        self._groups = self._build_groups()
        for g in self._groups:
            g.start()
        self.get_logger().info(
            f"Sensor bridge: {self.model.ncam} cameras, {len(self._lidar_adr)} "
            f"LiDAR beams across {len(self._groups)} workers "
            f"(front={self.front_rate:.0f} aux={self.aux_rate:.0f} "
            f"lidar={self.lidar_rate:.0f} Hz)")
        # Periodic telemetry: report each worker's ACHIEVED vs configured rate so
        # it's obvious when a rate is set higher than the hardware can sustain.
        self.create_timer(5.0, self._log_rates, callback_group=self._pose_cbg)

    # ------------------------------------------------------------------ groups
    def _build_groups(self):
        """Partition the sensors into the FRONT / AUX / LIDAR render workers."""
        groups = []

        # FRONT worker: front D435i (color+depth+cloud) + IR stereo.
        front_csizes, front_dsizes = set(), set()
        for c in self._front_depth:
            front_csizes.add((c["w"], c["h"]))
            if self.pub_depth:
                front_dsizes.add((c["w"], c["h"]))
        if self.pub_ir:
            for _, _, _, w, h in self._ir_cams:
                front_csizes.add((w, h))
        if front_csizes or front_dsizes:
            groups.append(_RenderGroup(self, "front", self.front_rate,
                                       front_csizes, front_dsizes, self._tick_front))

        # AUX worker: back D435i + body/wrist/ZED mono cameras.
        aux_csizes, aux_dsizes = set(), set()
        for c in self._aux_depth:
            aux_csizes.add((c["w"], c["h"]))
            if self.pub_depth:
                aux_dsizes.add((c["w"], c["h"]))
        if self.pub_color:
            for _, _, _, w, h in self._mono_cams:
                aux_csizes.add((w, h))
        if aux_csizes or aux_dsizes:
            groups.append(_RenderGroup(self, "aux", self.aux_rate,
                                       aux_csizes, aux_dsizes, self._tick_aux))

        # LIDAR worker: no GL needed (pure sensordata), but still its own MjData.
        if self.pub_lidar:
            groups.append(_RenderGroup(self, "lidar", self.lidar_rate,
                                       set(), set(), self._tick_lidar))
        return groups

    def _tick_front(self, g):
        now = self.get_clock().now().to_msg()
        bp = self.data_base_pos(g)
        bpos, bmat = bp
        tfs = []
        for c in self._front_depth:
            self._do_depth(g, c, now, bpos, bmat, tfs)
        if self.pub_ir:
            for i, (mj, ns, frame, w, h) in enumerate(self._ir_cams):
                baseline = self._ir_bx[i] if i < len(self._ir_bx) else 0.0
                self._do_mono(g, mj, f"/camera/{ns}/image_rect_raw",
                              f"/camera/{ns}/camera_info", frame, w, h, now,
                              bpos, bmat, tfs, baseline=baseline)
        if tfs:
            self.tf_broadcaster.sendTransform(tfs)

    def _tick_aux(self, g):
        now = self.get_clock().now().to_msg()
        bpos, bmat = self.data_base_pos(g)
        tfs = []
        if self.pub_color:
            for mj, ns, frame, w, h in self._mono_cams:
                self._do_color(g, mj, f"/camera/{ns}/image_raw",
                               f"/camera/{ns}/camera_info", frame, w, h, now,
                               bpos, bmat, tfs)
        for c in self._aux_depth:
            self._do_depth(g, c, now, bpos, bmat, tfs)
        if tfs:
            self.tf_broadcaster.sendTransform(tfs)

    def _tick_lidar(self, g):
        now = self.get_clock().now().to_msg()
        bpos, bmat = self.data_base_pos(g)
        tfs = []
        self._do_lidar(g, now, bpos, bmat, tfs)
        if tfs:
            self.tf_broadcaster.sendTransform(tfs)

    @staticmethod
    def data_base_pos(g):
        return g.data.xpos[g.node._chassis_bid], g.data.xmat[g.node._chassis_bid]

    def _log_rates(self):
        parts = []
        for g in self._groups:
            ach = (1.0 / g.last_dt) if g.last_dt > 0 else 0.0
            parts.append(f"{g.label} {ach:4.1f}/{g.rate:.0f}Hz")
        self.get_logger().info("render rates (achieved/target): " + "  ".join(parts))

    def _check_gpu_egl(self):
        """Warn (loudly) if ``egl`` is about to render on Mesa software, not the GPU.

        EGL resolves its driver from libglvnd's vendor ICDs in the dir named by
        ``__EGL_VENDOR_LIBRARY_DIRS``, AT PROCESS START. Conda activation exports a
        dir that has ONLY the Mesa software ICD (no NVIDIA), so ``MUJOCO_GL=egl``
        silently falls back to llvmpipe (CPU): 0% GPU, ~9x slower per frame
        (measured 11.6 ms vs 1.3 ms for a 640x360 render here). The real fix is the
        ``__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d`` that
        ``sim.launch.py`` sets in the process env -- we can't fix it from here
        because ``import mujoco`` (top of file) has already bound the vendor. This
        is just a diagnostic so a misconfigured run is obvious instead of slow."""
        if self._backend != "egl":
            return
        cur = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", "")
        if cur and not os.path.exists(os.path.join(cur, "10_nvidia.json")):
            self.get_logger().warn(
                f"__EGL_VENDOR_LIBRARY_DIRS='{cur}' has no NVIDIA ICD -> EGL may be "
                f"rendering on Mesa SOFTWARE (slow). Launch via sim.launch.py, or "
                f"export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d "
                f"before starting the node.")

    # ------------------------------------------------------------------ setup
    def _build_state_maps(self):
        """URDF joint name -> qpos address, plus the base free-joint address."""
        m = self.model
        self._joint_qadr = {}
        self._base_qadr = None
        for j in range(m.njnt):
            jtype = m.jnt_type[j]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            if jtype == mujoco.mjtJoint.mjJNT_FREE and name == cfg.BASE_FREE_JOINT:
                self._base_qadr = m.jnt_qposadr[j]
            elif jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                self._joint_qadr[cfg.urdfize(name)] = m.jnt_qposadr[j]

    def _precompute_intrinsics(self):
        """Pinhole intrinsics are static (fovy + render size) -- compute once."""
        self._intr = {}
        for mj, _, _, w, h in self._mono_cams:
            self._intr[mj] = sr.camera_intrinsics(self.model, mj, w, h)
        for mj, _, _, w, h in self._ir_cams:
            self._intr[mj] = sr.camera_intrinsics(self.model, mj, w, h)
        for c in self._depth_cams:
            self._intr[c["mj"]] = sr.camera_intrinsics(
                self.model, c["mj"], c["w"], c["h"])

    def _compute_ir_baselines(self):
        """Optical-frame x offset (m) of each IR camera from the first (reference).

        The IR cameras are rigid on the chassis, so the relative offset is constant
        regardless of base pose -- measure it once from the model. Index 0 (the
        reference / left camera) is 0.0 and the right camera is its +baseline. Used
        to set camera_info P[3] = -fx*bx so cuVSLAM gets the true stereo baseline."""
        if not self._ir_cams:
            return []
        rid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._ir_cams[0][0])
        ref_pos = self.data.cam_xpos[rid].copy()
        opt_x = (self.data.cam_xmat[rid].reshape(3, 3) @ sr._MJ_TO_OPTICAL)[:, 0]
        bx = []
        for mj, *_ in self._ir_cams:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, mj)
            bx.append(float(opt_x @ (self.data.cam_xpos[cid] - ref_pos)))
        return bx

    def _setup_publishers(self):
        qos = qos_profile_sensor_data
        self._img_pubs = {}      # topic -> publisher
        self._info_pubs = {}
        self._pc_pubs = {}

        def img(topic):
            self._img_pubs[topic] = self.create_publisher(Image, topic, qos)

        def info(topic):
            self._info_pubs[topic] = self.create_publisher(CameraInfo, topic, qos)

        if self.pub_color:
            for _, ns, _, _, _ in self._mono_cams:
                img(f"/camera/{ns}/image_raw")
                info(f"/camera/{ns}/camera_info")
            for c in self._depth_cams:
                # The ZED depth cam sets publish_color=False -- its RGB already
                # comes from the zed_left/zed_right mono cams, so skip the pub.
                if c.get("publish_color", True):
                    img(f"/camera/{c['ns']}/color/image_raw")
                    info(f"/camera/{c['ns']}/color/camera_info")
        if self.pub_depth:
            for c in self._depth_cams:
                img(f"/camera/{c['ns']}/depth/image_rect_raw")
                info(f"/camera/{c['ns']}/depth/camera_info")
                self._pc_pubs[c["ns"]] = self.create_publisher(
                    PointCloud2, f"/camera/{c['ns']}/depth/points", qos)
        if self.pub_ir:
            for _, ns, _, _, _ in self._ir_cams:
                img(f"/camera/{ns}/image_rect_raw")
                info(f"/camera/{ns}/camera_info")
        if self.pub_lidar:
            self._scan_pub = self.create_publisher(
                LaserScan, scfg.LIDAR_TOPIC, qos)

    # -------------------------------------------------------------- callbacks
    def _on_joint_states(self, msg: JointState):
        q0 = self.model.qpos0
        with self._pose_lock:
            for name, pos in zip(msg.name, msg.position):
                adr = self._joint_qadr.get(name)
                if adr is not None:
                    # /joint_states carries URDF-convention angles (qpos - qpos0, see
                    # mujoco_node._publish_joint_states). MuJoCo data.qpos needs the
                    # ABSOLUTE qpos, so add qpos0 back. Without this the rendered hand
                    # (thumb/abductions/wrists have qpos0 != 0) is posed wrong and the
                    # depth cloud no longer lands on the now-correct robot model.
                    self._qpos[adr] = pos + q0[adr]
        self._have_pose = True

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if self._base_qadr is None:
            return
        b = self._base_qadr
        with self._pose_lock:
            self._qpos[b:b + 7] = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
        self._have_pose = True

    def _apply_snapshot(self, data):
        """Copy the latest pose snapshot into a worker's private MjData + forward."""
        with self._pose_lock:
            data.qpos[:] = self._qpos
        mujoco.mj_forward(self.model, data)

    # ----------------------------------------------------------- per-sensor
    def _cam_tf(self, data, mj, frame, now, base_xpos, base_xmat):
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, mj)
        t, q = sr.rel_optical_tf(data.cam_xpos[cid], data.cam_xmat[cid],
                                 base_xpos, base_xmat)
        return self._tf_msg(now, scfg.BASE_FRAME, frame, t, q)

    def _do_color(self, g, mj, img_topic, info_topic, frame, w, h, now,
                  base_xpos, base_xmat, tfs):
        rgb = sr.capture_rgb(g.color[(w, h)], g.data, mj, self._vopt)
        self._img_pubs[img_topic].publish(
            self._image_msg(now, frame, rgb, "rgb8", w * 3))
        fx, fy, cx, cy = self._intr[mj]
        self._info_pubs[info_topic].publish(
            self._info_msg(now, frame, w, h, fx, fy, cx, cy))
        tfs.append(self._cam_tf(g.data, mj, frame, now, base_xpos, base_xmat))

    def _do_mono(self, g, mj, img_topic, info_topic, frame, w, h, now,
                 base_xpos, base_xmat, tfs, baseline=0.0):
        rgb = sr.capture_rgb(g.color[(w, h)], g.data, mj, self._vopt)
        mono = sr.rgb_to_mono(rgb)
        self._img_pubs[img_topic].publish(
            self._image_msg(now, frame, mono, "mono8", w))
        fx, fy, cx, cy = self._intr[mj]
        # baseline>0 (the right IR camera) -> Tx=-fx*baseline encodes the stereo baseline.
        self._info_pubs[info_topic].publish(
            self._info_msg(now, frame, w, h, fx, fy, cx, cy, tx=-fx * baseline))
        tfs.append(self._cam_tf(g.data, mj, frame, now, base_xpos, base_xmat))

    def _do_depth(self, g, c, now, base_xpos, base_xmat, tfs):
        mj, ns = c["mj"], c["ns"]
        w, h = c["w"], c["h"]
        fx, fy, cx, cy = self._intr[mj]
        # Per-camera flags (D435i cams default both True; the ZED head cloud sets
        # publish_color=False so it never duplicates the ZED's mono RGB topics).
        emit_color = c.get("publish_color", True)
        emit_tf = c.get("emit_tf", True)
        # ONE color render serves BOTH the color image and the cloud's colour
        # (the old node rendered colour twice for depth cams).
        rgb = sr.capture_rgb(g.color[(w, h)], g.data, mj, self._vopt)
        if self.pub_color and emit_color:
            self._img_pubs[f"/camera/{ns}/color/image_raw"].publish(
                self._image_msg(now, c["color_frame"], rgb, "rgb8", w * 3))
            self._info_pubs[f"/camera/{ns}/color/camera_info"].publish(
                self._info_msg(now, c["color_frame"], w, h, fx, fy, cx, cy))
            if emit_tf:
                tfs.append(self._cam_tf(g.data, mj, c["color_frame"], now,
                                        base_xpos, base_xmat))

        depth = sr.capture_depth(g.depth[(w, h)], g.data, mj, self._vopt)
        # Inject range-dependent axial depth noise once, so the depth image and
        # the point cloud derived from it stay consistent.
        depth = sr.add_depth_noise(depth, self.depth_noise_base,
                                   self.depth_noise_k, g.rng)
        depth = sr.add_depth_outliers(depth, self.depth_hole_prob,
                                      self.depth_outlier_prob, 0.3,
                                      scfg.DEPTH_MAX_M, g.rng)
        self._img_pubs[f"/camera/{ns}/depth/image_rect_raw"].publish(
            self._image_msg(now, c["depth_frame"], depth, "32FC1", w * 4))
        self._info_pubs[f"/camera/{ns}/depth/camera_info"].publish(
            self._info_msg(now, c["depth_frame"], w, h, fx, fy, cx, cy))
        data, n = sr.depth_to_xyzrgb_bytes(depth, rgb, fx, fy, cx, cy,
                                           scfg.DEPTH_MAX_M, self.pc_stride,
                                           self.depth_noise_lat, g.rng)
        self._pc_pubs[ns].publish(self._pointcloud_msg(now, c["depth_frame"], data, n))
        if emit_tf:
            tfs.append(self._cam_tf(g.data, mj, c["depth_frame"], now,
                                    base_xpos, base_xmat))

    def _do_lidar(self, g, now, base_xpos, base_xmat, tfs):
        ranges = sr.lidar_scan_fast(g.data, self._lidar_adr, self._lidar_cutoff)
        rmax = self._lidar_cutoff
        # Footprint self-filter: rays returning from inside the robot's own profile
        # are its body/arms, not obstacles -> drop to no-return (+inf). See the
        # lidar_min_range param. Done BEFORE noise so the published scan is clean.
        if self.lidar_min_range > 0.0:
            ranges = np.where(ranges < self.lidar_min_range, np.inf, ranges)
        ranges = sr.add_range_noise(ranges, g.rng, self.lidar_noise_scale)
        ranges = sr.add_lidar_outliers(ranges, self.lidar_dropout_prob,
                                       self.lidar_outlier_prob, 0.0, rmax, g.rng)
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = scfg.LIDAR_FRAME
        scan.angle_min = self._lidar_amin
        scan.angle_increment = self._lidar_inc
        scan.angle_max = self._lidar_amin + self._lidar_inc * (len(ranges) - 1) \
            if len(ranges) else self._lidar_amin
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / max(self.lidar_rate, 1.0)
        scan.range_min = self.lidar_min_range
        scan.range_max = rmax
        scan.ranges = ranges.tolist()
        self._scan_pub.publish(scan)
        t, q = sr.rel_body_tf(g.data.xpos[self._lidar_bid],
                              g.data.xmat[self._lidar_bid], base_xpos, base_xmat)
        tfs.append(self._tf_msg(now, scfg.BASE_FRAME, scfg.LIDAR_FRAME, t, q))

    # -------------------------------------------------------- message helpers
    @staticmethod
    def _image_msg(stamp, frame, arr, encoding, step):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height = int(arr.shape[0])
        msg.width = int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(step)
        # Assign the uint8[] payload as an array.array('B'), NOT bytes/list:
        # rclpy validates a bytes/list field element-by-element (~40 ms for a
        # 640x360 image!), but fast-paths an array.array via a single memcpy
        # (~0.3 ms). This one change is what lets the workers actually hit their
        # rates -- with bytes, ~6 messages/tick x ~40 ms pinned the front tick at
        # ~0.5 s (the real cause of the ~2 Hz front cloud, alongside software EGL).
        msg.data = array.array("B", np.ascontiguousarray(arr).tobytes())
        return msg

    @staticmethod
    def _info_msg(stamp, frame, w, h, fx, fy, cx, cy, tx=0.0):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.width = int(w)
        msg.height = int(h)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # P[3]=Tx encodes the stereo baseline of the RIGHT camera (Tx=-fx*baseline);
        # 0 for monocular / the left reference camera.
        msg.p = [fx, 0.0, cx, float(tx), 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    @staticmethod
    def _pointcloud_msg(stamp, frame, data, n):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        # array.array('B') for the same rclpy fast-path reason as _image_msg.
        msg.data = array.array("B", data)
        msg.is_dense = True
        return msg

    @staticmethod
    def _tf_msg(stamp, parent, child, t, q):
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = parent
        tf.child_frame_id = child
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        return tf

    # ----------------------------------------------------------------- shutdown
    def stop(self):
        for g in self._groups:
            g.stop()
        for g in self._groups:
            g.join(timeout=2.0)

    def destroy_node(self):
        # Close GL contexts explicitly so EGL teardown doesn't error during GC.
        for g in self._groups:
            g.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSensorBridge()
    # MultiThreadedExecutor so the reentrant pose callbacks update while the
    # render workers run -- keeps each frame's pose in sync with its timestamp.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
