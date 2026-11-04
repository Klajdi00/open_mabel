"""realsense_d435.py — native pyrealsense2 driver for the D435 front camera.

Publishes the existing sim/ROS contract so SLAM/nvblox runs unchanged:
    /camera/front/color/image_raw        (+ .../color/camera_info)   rgb8
    /camera/front/depth/image_rect_raw   (+ .../depth/camera_info)   16UC1 (mm)
    /camera/front/depth/points           sensor_msgs/PointCloud2 (XYZRGB)
    /camera/front/infra1/image_rect_raw  (+ .../infra1/camera_info)  mono8 (left IR)
    /camera/front/infra2/image_rect_raw  (+ .../infra2/camera_info)  mono8 (right IR)

Depth is aligned to color, so color + depth + cloud all share ``front_color_optical_frame``
(the manager publishes base_link -> front_link -> front_color_optical_frame). The IR
pair is the factory-RECTIFIED stereo cuVSLAM tracks with (visual_slam navigation env):
infrared,1 is the LEFT imager (== depth origin) on ``front_infra1_optical_frame`` (the
manager parents it at the cam mount); this driver publishes the infra1 -> infra2 static
TF at the calibrated baseline, and infra2's CameraInfo carries P[3] = -fx*baseline (the
rectified-stereo convention cuVSLAM triangulates with). Runs its own capture thread; a
missing camera => idle + periodic reconnect, never fatal; a device that refuses the IR
streams falls back to color+depth only (SLAM idles, video/teleop unaffected).

Visual-SLAM-critical details (measured on thor, 2026-07):
  * EMITTER (cfg ``emitter``, env CAM_RS_EMITTER, default "off"): the D435's IR
    projector paints a dense speckle dot pattern into BOTH IR imagers. Those dots
    are rigid to the CAMERA, not the world, so cuVSLAM latches onto them and
    tracking drifts wildly under handheld/robot motion (observed: +12 m in z).
    NVIDIA's reference D435i launch sets ``depth_module.emitter_enabled: 0``.
    "off" costs some passive-depth density (measured 80% -> 65% valid pixels in a
    textured room) — an acceptable trade for usable VIO. Set "on" only when depth
    density matters more than cuVSLAM (e.g. wheel-odom-only nvblox mapping).
  * STAMPS: images are stamped with the SENSOR's hardware timestamp (librealsense
    global-time domain == host epoch) instead of publish-time, so a GIL stall in
    another camera thread shows up as harmless latency, not timestamp jitter.
    Falls back to node-clock if the device clock is unsynchronised (>1 s off).
  * IMU (cfg ``imu``, env CAM_RS_IMU, default on = try): D435i accel+gyro opened
    SENSOR-LEVEL on the SAME device handle the video pipeline holds (single-owner
    rule: nothing outside this bridge process may ever touch the camera — an
    external RSUSB IMU grabber took the whole camera HAL down on 2026-07-01).
    Publishes /camera/front/imu (gyro rate, accel linearly interpolated to gyro
    stamps — realsense-ros unite_imu_method=2) on ``front_imu_optical_frame`` +
    static TF from infra1 (SDK extrinsics). NOTE: needs an RSUSB-backend
    pyrealsense2: the pip wheel uses the V4L2/IIO backend and the Tegra kernel
    ships no hid-sensor IIO modules, so the motion module does not enumerate
    ("HID Motion Sensor Failure") and this driver logs one warning + continues
    without IMU. A python binding built from the RSUSB tree
    (~/librealsense_rsusb, BUILD_PYTHON_BINDINGS=ON, verified: motion module
    enumerates, gyro 200 Hz clean) is wired OPT-IN via run_bridge.sh
    MABEL_RS_RSUSB=1 — it switches the WHOLE bridge (video too) onto the RSUSB
    backend, so flip it during a maintenance window and verify video first.
"""

import threading
import time

import numpy as np

try:
    import pyrealsense2 as rs
    _RS_OK, _RS_ERR = True, None
except Exception as _e:                              # pragma: no cover
    rs = None
    _RS_OK, _RS_ERR = False, _e

from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, Imu, PointCloud2

from . import ros_msgs, rs_util

OPTICAL_FRAME = "front_color_optical_frame"
IR1_FRAME = "front_infra1_optical_frame"      # left imager (manager parents at the mount)
IR2_FRAME = "front_infra2_optical_frame"      # right imager (this driver: infra1 + baseline)
IMU_FRAME = "front_imu_optical_frame"         # D435i motion module (this driver: infra1 + SDK extrinsics)
_CLOUD_STRIDE = 2     # subsample the cloud (1/stride^2 points) to bound bandwidth
_CLOUD_HZ = 6.0       # viewing/overlay consumers only (RViz, app map stream @2 Hz):
                      # full-rate ~36 MB/s pressured the 1 MB kernel UDP buffer and
                      # made every RViz cloud flicker. nvblox consumes depth IMAGES,
                      # not this topic — 6 Hz costs nothing downstream.


class RealSenseD435:
    def __init__(self, node, cfg, video=None):
        self.node = node
        self.cfg = cfg
        self.video = video                # bridge VideoServer (or None): serves "front"
        self.stream = "front"             # /camera/front/stream.mjpg
        self.log = node.get_logger()
        self._stop = threading.Event()

        self.pub_color = node.create_publisher(Image, "/camera/front/color/image_raw", 5)
        self.pub_color_info = node.create_publisher(
            CameraInfo, "/camera/front/color/camera_info", 5)
        self.pub_depth = node.create_publisher(
            Image, "/camera/front/depth/image_rect_raw", 5)
        self.pub_depth_info = node.create_publisher(
            CameraInfo, "/camera/front/depth/camera_info", 5)
        self.pub_cloud = node.create_publisher(PointCloud2, "/camera/front/depth/points", 5)
        # IR pair on best-effort sensor QoS (matches the cuVSLAM/NITROS subscriber
        # and the old dev node): RELIABLE retransmits stall the 30 fps stereo stream,
        # which cuVSLAM logs as ">500 ms delta between frames" tracking gaps.
        q = qos_profile_sensor_data
        self.pub_ir1 = node.create_publisher(
            Image, "/camera/front/infra1/image_rect_raw", q)
        self.pub_ir1_info = node.create_publisher(
            CameraInfo, "/camera/front/infra1/camera_info", q)
        self.pub_ir2 = node.create_publisher(
            Image, "/camera/front/infra2/image_rect_raw", q)
        self.pub_ir2_info = node.create_publisher(
            CameraInfo, "/camera/front/infra2/camera_info", q)
        self.pub_imu = node.create_publisher(Imu, "/camera/front/imu", q)
        self._ir_tf_done = False        # infra1 -> infra2 static TF (needs the baseline)
        self._imu_tf_done = False       # infra1 -> imu static TF (needs SDK extrinsics)
        self._imu_warned = False        # warn ONCE when the motion module is absent
        self._accel = []                # last two (t_ms, (x,y,z)) accel samples
        self._stamp_warned = False      # warn ONCE on unsynchronised device clock
        self._last_cloud = 0.0          # cloud publish throttle (_CLOUD_HZ)

        if not _RS_OK:
            self.log.warn(f"pyrealsense2 unavailable ({_RS_ERR}); front camera IDLE.")
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._capture_loop()
            except Exception as e:                    # camera unplugged / SDK error
                self.log.warn(f"D435 error ({e}); retrying in 3 s.",
                              throttle_duration_sec=10.0)
                # A failed pipeline.start() leaves USB handles alive inside the
                # pyrealsense2 objects, pinned by the traceback's reference cycle
                # until the CYCLIC gc runs — under a retry loop those fds pile up
                # (same EMFILE family as the rs_util context leak). Collect now
                # so each retry starts clean.
                import gc
                gc.collect()
                time.sleep(3.0)

    def _resolve_serial(self):
        """Serial of the front camera. A serial-less rs.config grabs WHICHEVER
        RealSense enumerates first — with the D405 wrist cam plugged in, the front
        pipeline actually hijacked it (observed: /camera/front/* streaming with the
        D405's 18 mm baseline / 0.1 mm depth scale). Pin to cfg['serial'] when set
        (used verbatim — start just fails while it's absent); otherwise take the
        first device that is NOT a D405 (D405s are wrist cams, wrist_d405.py owns
        them). Live scans throw while ANY RealSense is busy streaming, so fall back
        to the registry cached from earlier successful scans (see rs_util). None =>
        unknown serial — caller may still blind-start if sysfs shows a front cam."""
        want = str(self.cfg.get("serial", "") or "")
        if want:
            return want
        try:
            devices = rs_util.scan()
        except Exception:
            devices = rs_util.registry()
        for serial, name in devices.items():
            if "D405" not in name:
                return serial
        return None

    def _capture_loop(self):
        c = self.cfg
        serial = self._resolve_serial()
        if serial is None:
            # Serial unknowable (device hot-plugged while another camera streams
            # blocks live scans and it was never scanned before). sysfs product
            # names stay readable: if a non-D405 RealSense is physically present,
            # claim it with a serial-less start — resolution skips busy devices —
            # and VERIFY the claim below (a D405 is never ours). Otherwise idle.
            if not any("RealSense" in p and "405" not in p
                       for p in rs_util.sysfs_products()):
                time.sleep(2.0)                      # front cam absent; keep polling
                return
        pipeline = rs.pipeline(rs_util.context())
        config = rs.config()
        if serial is not None:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, c["width"], c["height"],
                             rs.format.rgb8, c["fps"])
        config.enable_stream(rs.stream.depth, c["width"], c["height"],
                             rs.format.z16, c["fps"])
        # Rectified IR stereo pair for cuVSLAM. infrared,1 = LEFT imager (the depth
        # origin), infrared,2 = right. If the device refuses the extra streams
        # (USB-2 fallback bandwidth), retry color+depth only — video/teleop keep
        # working, SLAM idles until the camera lands on a proper USB-3 port.
        with_ir = True
        config.enable_stream(rs.stream.infrared, 1, c["width"], c["height"],
                             rs.format.y8, c["fps"])
        config.enable_stream(rs.stream.infrared, 2, c["width"], c["height"],
                             rs.format.y8, c["fps"])
        try:
            profile = pipeline.start(config)
        except Exception as e:
            self.log.warn(f"D435 refused the IR stereo streams ({e}); retrying "
                          "color+depth only — cuVSLAM has NO input (check USB-3).")
            with_ir = False
            config = rs.config()
            if serial is not None:
                config.enable_device(serial)
            config.enable_stream(rs.stream.color, c["width"], c["height"],
                                 rs.format.rgb8, c["fps"])
            config.enable_stream(rs.stream.depth, c["width"], c["height"],
                                 rs.format.z16, c["fps"])
            profile = pipeline.start(config)
        dev = profile.get_device()
        dev_name = dev.get_info(rs.camera_info.name)
        rs_util.remember(dev.get_info(rs.camera_info.serial_number), dev_name)
        if serial is None and "D405" in dev_name:
            # Blind start grabbed the wrist cam (it happened to be idle) — release
            # it for wrist_d405.py and wait for an actual front camera.
            pipeline.stop()
            time.sleep(2.0)
            return
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self._apply_emitter(depth_sensor)
        ir_intr, ir_baseline = None, 0.0
        if with_ir:
            try:
                ir1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
                ir2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
                ir_intr = ir1.get_intrinsics()      # shared rectified pinhole (both imagers)
                ir_baseline = float(abs(ir1.get_extrinsics_to(ir2).translation[0]))
                self._publish_ir_tf(ir_baseline)
            except Exception as e:
                self.log.warn(f"D435 IR intrinsics unavailable ({e}); IR not published.")
                with_ir = False
        align = rs.align(rs.stream.color)
        imu_sensor = self._start_imu(profile, with_ir)
        self.log.info(
            f"D435 streaming {c['width']}x{c['height']}@{c['fps']} "
            f"(depth_scale={depth_scale:.5f}, "
            f"ir={'on baseline=%.1fmm' % (ir_baseline * 1000) if with_ir else 'OFF'}, "
            f"imu={'on' if imu_sensor else 'OFF'}).")
        try:
            while not self._stop.is_set():
                raw = pipeline.wait_for_frames()
                # Pull the IR pair from the RAW frameset, BEFORE align: align
                # reprojects depth into the colour frame, and the IR imagers must
                # stay in their own rectified frame so the stereo geometry is
                # preserved for cuVSLAM.
                ir1 = ir2 = None
                if with_ir:
                    ir1 = raw.get_infrared_frame(1)
                    ir2 = raw.get_infrared_frame(2)
                frames = align.process(raw)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    continue
                self._publish(color, depth, depth_scale, ir1, ir2, ir_intr, ir_baseline,
                              stamp=self._frame_stamp(raw))
        finally:
            if imu_sensor is not None:
                try:
                    imu_sensor.stop()
                    imu_sensor.close()
                except Exception:
                    pass
            pipeline.stop()

    def _apply_emitter(self, depth_sensor):
        """cfg['emitter'] ('off' default / 'on'): the projector's speckle dots are
        rigid to the CAMERA, so cuVSLAM tracks them instead of the world and the
        odometry drifts wildly under motion. Off = clean IR for VIO, sparser
        passive depth (measured 80% -> 65% valid px in a textured room)."""
        mode = str(self.cfg.get("emitter", "off")).lower()
        on = mode not in ("off", "0", "false", "no")
        try:
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if on else 0.0)
                self.log.info(f"D435 IR emitter {'ON (dense depth; speckle in IR — bad '
                              'for cuVSLAM)' if on else 'OFF (clean IR for visual SLAM)'}.")
        except Exception as e:
            self.log.warn(f"D435 emitter option failed ({e}); leaving device default.")

    # ---------------- IMU (D435i motion module) ----------------

    def _start_imu(self, video_profile, with_ir):
        """Motion sensor of the SAME already-open device -> /camera/front/imu.

        Single-owner rule: the bridge is the only process that may touch the
        camera, so the motion module is opened SENSOR-LEVEL on the device handle
        the video pipeline already holds (dev.query_sensors) — no second
        pipeline, no re-enumeration, no separate process (a standalone RSUSB
        publisher grabbing the device from outside took the whole camera HAL
        down on 2026-07-01 — never do that again).

        Publishes gyro-rate sensor_msgs/Imu with accel LINEARLY INTERPOLATED
        onto the gyro stamp (realsense-ros unite_imu_method=2, what Isaac VSLAM
        expects) and latches the infra1 -> imu static TF from the SDK
        extrinsics so cuVSLAM's ``imu_frame`` resolves. Returns the started
        motion sensor (caller stops/closes it), or None when the motion module
        does not enumerate (V4L2-backend pyrealsense2 on Tegra: no hid-sensor
        IIO kernel modules -> 'HID Motion Sensor Failure'; needs the
        RSUSB-backend pyrealsense2 — see module docstring + run_bridge.sh)."""
        if not self.cfg.get("imu", True):
            return None
        try:
            dev = video_profile.get_device()
            motion = None
            for s in dev.query_sensors():
                try:
                    if s.is_motion_sensor():
                        motion = s
                        break
                except Exception:
                    continue
            if motion is None:
                raise RuntimeError("no motion sensor enumerated")

            profs = motion.get_stream_profiles()

            def pick(stype, want_fps):
                c = [p for p in profs if p.stream_type() == stype
                     and p.format() == rs.format.motion_xyz32f]
                return min(c, key=lambda p: abs(p.fps() - want_fps)) if c else None

            gyro_p = pick(rs.stream.gyro, 200)
            accel_p = pick(rs.stream.accel, 250)
            if gyro_p is None or accel_p is None:
                raise RuntimeError("motion stream profiles unavailable")
            motion.open([accel_p, gyro_p])
            motion.start(self._imu_cb)
            if with_ir and not self._imu_tf_done:
                try:
                    ir1 = video_profile.get_stream(rs.stream.infrared, 1)
                    self._publish_imu_tf(ir1.get_extrinsics_to(gyro_p))
                except Exception as e:
                    self.log.warn(f"D435i IMU extrinsics unavailable ({e}); "
                                  f"infra1 -> {IMU_FRAME} TF not published.")
            self.log.info(f"D435i IMU on (gyro {gyro_p.fps()} Hz + accel "
                          f"{accel_p.fps()} Hz -> /camera/front/imu).")
            return motion
        except Exception as e:
            if not self._imu_warned:
                self._imu_warned = True
                self.log.warn(
                    f"D435i motion module unavailable ({e}); /camera/front/imu OFF. "
                    "cuVSLAM runs stereo-only (drifts under fast handheld motion). "
                    "Fix: RSUSB-backend pyrealsense2 (see realsense_d435.py docstring).")
            return None

    def _imu_cb(self, frame):
        """librealsense callback thread: accumulate accel, publish on gyro."""
        try:
            if frame.is_frameset():           # syncer may batch accel+gyro together
                for f in rs.composite_frame(frame):
                    self._imu_cb(f)
                return
            prof = frame.get_profile()
            # Motion frames arrive on the device clock until librealsense's
            # global-time sync locks (~1 s after start); those stamps are in a
            # different epoch and would feed cuVSLAM garbage dt — drop them.
            if (frame.get_frame_timestamp_domain() != rs.timestamp_domain.global_time):
                return
            t_ms = frame.get_timestamp()
            if abs(t_ms / 1000.0 - time.time()) > 1.0:
                return
            data = frame.as_motion_frame().get_motion_data()
            if prof.stream_type() == rs.stream.accel:
                self._accel = (self._accel + [(t_ms, (data.x, data.y, data.z))])[-2:]
                return
            if prof.stream_type() != rs.stream.gyro or not self._accel:
                return
            # linear interpolation (extrapolation-clamped) of accel at the gyro stamp
            (t0, a0) = self._accel[0]
            (t1, a1) = self._accel[-1]
            if t1 > t0:
                w = min(max((t_ms - t0) / (t1 - t0), 0.0), 2.0)
                ax = [a0[i] + w * (a1[i] - a0[i]) for i in range(3)]
            else:
                ax = list(a1)
            m = Imu()
            m.header.stamp = self._ms_to_stamp(t_ms)
            m.header.frame_id = IMU_FRAME
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = \
                float(data.x), float(data.y), float(data.z)
            m.linear_acceleration.x, m.linear_acceleration.y, \
                m.linear_acceleration.z = float(ax[0]), float(ax[1]), float(ax[2])
            m.orientation_covariance[0] = -1.0        # no orientation estimate
            self.pub_imu.publish(m)
        except Exception:
            pass                                       # never kill the SDK thread

    def _publish_imu_tf(self, ext):
        """infra1 -> imu static TF from librealsense extrinsics (row-major R + t)."""
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster
        r = ext.rotation                               # column-major 3x3 per SDK docs
        # quaternion from rotation matrix (transpose: SDK stores column-major)
        m00, m10, m20, m01, m11, m21, m02, m12, m22 = r
        tr = m00 + m11 + m22
        if tr > 0:
            s = (tr + 1.0) ** 0.5 * 2
            qw, qx, qy, qz = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = (1.0 + m00 - m11 - m22) ** 0.5 * 2
            qw, qx, qy, qz = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
        elif m11 > m22:
            s = (1.0 + m11 - m00 - m22) ** 0.5 * 2
            qw, qx, qy, qz = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
        else:
            s = (1.0 + m22 - m00 - m11) ** 0.5 * 2
            qw, qx, qy, qz = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
        self._imu_statictf = StaticTransformBroadcaster(self.node)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = IR1_FRAME
        t.child_frame_id = IMU_FRAME
        t.transform.translation.x = float(ext.translation[0])
        t.transform.translation.y = float(ext.translation[1])
        t.transform.translation.z = float(ext.translation[2])
        t.transform.rotation.x, t.transform.rotation.y = float(qx), float(qy)
        t.transform.rotation.z, t.transform.rotation.w = float(qz), float(qw)
        self._imu_statictf.sendTransform([t])
        self._imu_tf_done = True

    # ---------------- timestamps ----------------

    def _ms_to_stamp(self, t_ms):
        sec = int(t_ms // 1000)
        return TimeMsg(sec=sec, nanosec=int((t_ms - sec * 1000.0) * 1e6))

    def _frame_stamp(self, frame):
        """SENSOR-side (host-clock) stamp when available, else node clock.

        Publish-time stamping turns any GIL stall elsewhere in the bridge into
        TIMESTAMP jitter, which cuVSLAM integrates as false motion. Both
        global_time (RSUSB, hw-clock synced to host) and system_time (V4L2 wheel
        without metadata kernel patches: stamped at SDK dequeue by a NATIVE,
        GIL-immune thread) are host-epoch and keep inter-frame deltas honest
        even when python-side publication is delayed."""
        try:
            if frame.get_frame_timestamp_domain() in (
                    rs.timestamp_domain.global_time,
                    rs.timestamp_domain.system_time):
                t_ms = frame.get_timestamp()
                if abs(t_ms / 1000.0 - time.time()) < 1.0:
                    return self._ms_to_stamp(t_ms)
        except Exception:
            pass
        if not self._stamp_warned:
            self._stamp_warned = True
            self.log.warn("D435 global-time stamps unavailable; falling back to "
                          "node-clock (publish-time) stamps.")
        return self.node.get_clock().now().to_msg()

    def _publish_ir_tf(self, baseline_m):
        # infra1 -> infra2: the right imager offset by the calibrated baseline along
        # the optical +X axis (REP-145 x-right), identity rotation — an exact
        # rectified-stereo rig matching the camera_info P matrices. front_link ->
        # infra1 is the manager's (static, at the cam mount). Latched once.
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster
        if self._ir_tf_done:
            return
        self._ir_statictf = StaticTransformBroadcaster(self.node)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = IR1_FRAME
        t.child_frame_id = IR2_FRAME
        t.transform.translation.x = float(baseline_m)
        t.transform.rotation.w = 1.0
        self._ir_statictf.sendTransform([t])
        self._ir_tf_done = True

    def _publish(self, color, depth, depth_scale, ir1=None, ir2=None,
                 ir_intr=None, ir_baseline=0.0, stamp=None):
        if stamp is None:
            stamp = self.node.get_clock().now().to_msg()
        intr = color.profile.as_video_stream_profile().intrinsics
        K = (intr.fx, intr.fy, intr.ppx, intr.ppy)

        color_np = np.asanyarray(color.get_data())            # (H,W,3) rgb8
        depth_np = np.asanyarray(depth.get_data())            # (H,W) uint16, depth units

        # network sink (no ROS hop) — push raw BGR; the VideoServer downscales and
        # fans out to MJPEG / UDP-JPEG / UDP-H.264, only if a client is watching.
        if self.video is not None and self.video.wants(self.stream):
            bgr = color_np[:, :, ::-1]                        # rgb8 -> BGR
            if self.cfg.get("net_rotate180"):
                bgr = bgr[::-1, ::-1]     # camera mounted upside-down: upright for humans
            self.video.push(self.stream, np.ascontiguousarray(bgr))

        self.pub_color.publish(ros_msgs.image_msg(color_np, "rgb8", OPTICAL_FRAME, stamp))
        # depth as 16UC1 in millimetres (the ROS realsense convention)
        depth_mm = (depth_np.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)
        self.pub_depth.publish(ros_msgs.image_msg(depth_mm, "16UC1", OPTICAL_FRAME, stamp))

        info = ros_msgs.camera_info_msg(intr.width, intr.height, K, OPTICAL_FRAME, stamp)
        self.pub_color_info.publish(info)
        self.pub_depth_info.publish(info)

        # Rectified IR stereo -> cuVSLAM. Shared rectified intrinsics; the RIGHT
        # imager's P[3] = -fx*baseline (rectified-stereo convention).
        if ir1 and ir2 and ir_intr is not None:
            irK = (ir_intr.fx, ir_intr.fy, ir_intr.ppx, ir_intr.ppy)
            self.pub_ir1.publish(ros_msgs.image_msg(
                np.asanyarray(ir1.get_data()), "mono8", IR1_FRAME, stamp))
            self.pub_ir2.publish(ros_msgs.image_msg(
                np.asanyarray(ir2.get_data()), "mono8", IR2_FRAME, stamp))
            self.pub_ir1_info.publish(ros_msgs.camera_info_msg(
                ir_intr.width, ir_intr.height, irK, IR1_FRAME, stamp))
            self.pub_ir2_info.publish(ros_msgs.camera_info_msg(
                ir_intr.width, ir_intr.height, irK, IR2_FRAME, stamp,
                tx=-ir_intr.fx * ir_baseline))

        if (self.cfg["pointcloud"] and self.pub_cloud.get_subscription_count() > 0
                and (time.monotonic() - self._last_cloud) >= 1.0 / _CLOUD_HZ):
            self._last_cloud = time.monotonic()
            self._publish_cloud(color_np, depth_np, depth_scale, K, stamp)

    def _publish_cloud(self, color_np, depth_np, depth_scale, K, stamp):
        fx, fy, cx, cy = K
        s = _CLOUD_STRIDE
        z = depth_np[::s, ::s].astype(np.float32) * depth_scale     # metres
        h, w = z.shape
        us, vs = np.meshgrid(np.arange(w) * s, np.arange(h) * s)
        x = (us - cx) / fx * z
        y = (vs - cy) / fy * z
        valid = z > 0.0
        xyz = np.stack((x[valid], y[valid], z[valid]), axis=-1)
        rgb = color_np[::s, ::s][valid]
        self.pub_cloud.publish(ros_msgs.pointcloud2_msg(xyz, OPTICAL_FRAME, stamp, rgb=rgb))

    def stop(self):
        self._stop.set()
