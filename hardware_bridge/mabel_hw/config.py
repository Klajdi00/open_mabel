"""config.py — central tunables for the MABEL hardware bridge (the single HAL).

One place to enable/disable subsystems and set device parameters. Every value can be
overridden by an environment variable (``MABEL_HW_<NAME>``) so the systemd unit / a
shell can retune without editing code. The bridge owns ALL hardware; a subsystem with
``enabled=False`` is simply not brought up this run.
"""

import json
import os
from pathlib import Path


def _control_params() -> dict:
    """The shared arm control-law parameters (controller/control_param.json). mabel_hw reads it
    as DATA — it does not import controller code, so the HAL stays below the control layer — but
    it honours the SAME per-joint gains the controller uses, so the real robot and the setup
    studio can't run different kp/kd (they did: the HAL defaulted the shoulders to kp=30, above
    the damping ceiling, while the derived value is 14.9). Missing file -> {} -> built-in
    fallbacks below. $MABEL_CONTROL_PARAM overrides the path."""
    env = os.environ.get("MABEL_CONTROL_PARAM")
    # config.py is hardware_bridge/mabel_hw/config.py, so the repo root is parents[2].
    p = Path(env) if env else Path(__file__).resolve().parents[2] / "controller" / "control_param.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


_CONTROL_PARAMS = _control_params()


def _envf(name, default):
    try:
        return float(os.environ[f"MABEL_HW_{name}"])
    except (KeyError, ValueError):
        return default


def _envi(name, default):
    try:
        return int(os.environ[f"MABEL_HW_{name}"])
    except (KeyError, ValueError):
        return default


def _envs(name, default):
    return os.environ.get(f"MABEL_HW_{name}", default)


def _envb(name, default):
    v = os.environ.get(f"MABEL_HW_{name}")
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Mobile base (MAXSwerve Teensy) — the validated ps5_teleop path.
# ---------------------------------------------------------------------------
BASE = dict(
    enabled=_envb("BASE", True),
    teensy_ip=_envs("BASE_TEENSY_IP", "192.168.124.10"),
    port=_envi("BASE_PORT", 5800),
    stream_hz=_envi("BASE_STREAM_HZ", 200),
    stream_rate=_envf("BASE_STREAM_RATE", 50.0),   # Hz the bridge re-sends setpoints
    cmd_timeout=_envf("BASE_CMD_TIMEOUT", 0.5),    # s; zero drive if cmd_vel goes stale
    wheel_radius=_envf("BASE_WHEEL_RADIUS", 0.0381),  # REV 3in MAXSwerve
    # Host-side calibration lives in the bridge (decided): apply ps5_teleop's
    # STEER_OFFSET / DRIVE_DIR before sending. Set False only if the Teensy firmware
    # applies calibration itself (double-applying would mis-steer the wheels).
    apply_host_calib=_envb("BASE_APPLY_HOST_CALIB", True),
    # drive_gain scales the velocity SETPOINT (command only, not the odometry feedback).
    # WARNING — measured on the real base (drive_diag.py): the SPARK drive velocity loop
    # under-reaches to ~50% of commanded (the kF=0 steady-state error), AND it COLLAPSES
    # above ~1.5 m/s commanded (2 of 3 wheels drop to ~0 at 17-21 A). So drive_gain > 1 is
    # NOT a safe workaround here — pushing the setpoint up drives it into the collapse.
    # Leave at 1.0; the real fix is the SPARK drive kFF + current limit in REV Hardware
    # Client. (Kept as a knob for AFTER the SPARK is tuned.)
    drive_gain=_envf("BASE_DRIVE_GAIN", 1.0),
    joint_state_rate=_envf("BASE_JOINT_STATE_RATE", 50.0),
    # Direct PS5 pad teleop owned by the bridge (no ROS in the loop). Off by default;
    # arbitration in the base subsystem is estop > pad (deadman held) > ROS cmd_vel.
    joystick=_envb("BASE_JOYSTICK", False),
    joystick_dev=_envs("BASE_JOYSTICK_DEV", "/dev/input/js0"),
    vmax=_envf("BASE_VMAX", 1.0),     # m/s at full pad translation
    wmax=_envf("BASE_WMAX", 2.0),     # rad/s at full pad yaw
    # DDS contract (generic; mabel_interfaces-free).
    topic_cmd_vel="/mabel/hw/base/cmd_vel",
    topic_estop="/mabel/hw/base/estop",
    topic_joint_states="/mabel/hw/base/joint_states",
)

# ---------------------------------------------------------------------------
# Arms — the DDS ADAPTER's config only (topics + publish rate). The hardware
# itself is configured by LEFT_ARM / RIGHT_ARM below: they pick the backend, the
# link, the joints, and the calibration slice, and the facade builds them.
#
# Torque defaults OFF (limp) on bring-up; energising via /arms/torque_enable
# holds-in-place first (no lurch) — the hands' rule.
# ---------------------------------------------------------------------------
ARMS = dict(
    enabled=_envb("ARMS", True),
    auto_enable_torque=_envb("ARMS_AUTO_TORQUE", False),
    publish_rate=_envf("ARMS_RATE", 50.0),
    topic_command="/arms/command",
    topic_joint_states="/arms/joint_states",
    topic_torque="/arms/torque_enable",
    topic_torque_state="/arms/torque_state",
    # ADDITIVE (2026-07-30): PD-gain overrides (String JSON {"kp":{...},"kd":{...}})
    # — the app stiffness slider / soft-vs-stiff gain sets, computed gateway-side.
    topic_gains="/arms/gains",
)

# ---------------------------------------------------------------------------
# Body — LEGACY combined fallback (all-DaMiao, one CAN DamiaoGroup): torso +
# 3-DOF DaMiao neck, MIT mode, same protocol as the arms (subsystems/damiao.py).
# ⚠ This is NOT the as-built body. The as-built body is TWO SEPARATE buses,
# owned by NeckSubsystem + TorsoSubsystem (see the NECK / TORSO dicts below):
#   • neck  = SERIAL DYNAMIXEL (pitch/roll on XC330-T181, /dev/ttyUSB0) — plus an
#             OPTIONAL DaMiao yaw at the base; backend `dynamixel`, MABEL_HW_NECK.
#   • torso = DaMiao CAN (torso:can0:1); backend `can`, MABEL_HW_TORSO.
# body_node uses those two facade handles when they are up and only falls back to
# THIS single DamiaoGroup (wire-identical old node) when neither handle exists.
# Contract joints (either path): torso, neck_yaw, neck_pitch, neck_roll (rad).
# ---------------------------------------------------------------------------
BODY = dict(
    enabled=_envb("BODY", False),
    # "torso:can0:1, neck_yaw:can0:2, neck_pitch:can0:3, neck_roll:can0:4"
    joint_motors=_envs("BODY_JOINT_MOTORS", ""),
    kp_default=_envf("BODY_KP", 30.0),
    kd_default=_envf("BODY_KD", 1.0),
    p_max=_envf("BODY_PMAX", 12.5),     # DMJ4310 MIT position range (rad); tune per motor
    v_max=_envf("BODY_VMAX", 30.0),
    t_max=_envf("BODY_TMAX", 10.0),
    publish_rate=_envf("BODY_RATE", 50.0),
    topic_command="/body/command",
    topic_joint_states="/body/joint_states",
    topic_torque="/body/torque_enable",
    topic_torque_state="/body/torque_state",
)

# ---------------------------------------------------------------------------
# Hands (ORCA hand on Feetech STS/SCS over TTL serial) — the vendored HAL in
# hands/orcahand (ported verbatim from firmware/orca_hand/feetech). The bridge OWNS
# the serial bus and the joint<->tick map. End-stop calibration is NOT a ROS topic — it
# is done directly on this bus (firmware GUI, Control Center → Hands → Calibration, or
# `python3 -m mabel_hw.drivers.orcahand.calibrate`) with this subsystem stopped. The calibration
# SOURCE OF TRUTH is the firmware GUI's config folder
# (firmware/orca_hand/feetech/orcahand/configs/<side>_calibration.json): the bridge
# prefers it on every (re)connect when it exists, and the bridge's calibrate CLI
# writes back to that same file — the vendored copy in hands/orcahand/configs is only
# a fallback for checkouts without firmware/. One entry per physical hand; port "none"
# keeps that
# side idle. Port "auto" (the default) resolves the adapter SIDE-AWARELY through the
# adapter_serials registry below — never first-come-first-served across sides.
# ---------------------------------------------------------------------------
HANDS = dict(
    enabled=_envb("HANDS", True),
    # Port per side: an explicit device path pins it; "none" keeps the side idle;
    # "auto" (default) resolves through adapter_serials below — SIDE-AWARE, so with one
    # adapter plugged in the connected hand comes up as the correct side, and a side
    # whose adapter is absent stays idle and hot-plugs when it appears (2 s watchdog).
    left_port=_envs("HANDS_LEFT_PORT", "auto"),
    right_port=_envs("HANDS_RIGHT_PORT", "auto"),
    # Adapter registry: USB serial number -> side. The two hand adapters are identical
    # CH340s, so the USB serial number is the only stable discriminator. Find yours with
    # `ls -l /dev/serial/by-id/` (usb-1a86_USB_Single_Serial_<SERIAL>-if00) and set
    # MABEL_HW_HANDS_LEFT_SERIAL / MABEL_HW_HANDS_RIGHT_SERIAL. The defaults are placeholders.
    adapter_serials=dict(
        left=_envs("HANDS_LEFT_SERIAL", "0000000001"),
        right=_envs("HANDS_RIGHT_SERIAL", "0000000002"),
        # NOVA's ORCA v2 hands, swappable onto MABEL (NOVA servos/orca_hands.py export-v2 writes their
        # wiring + calibration here: <side>_v2_hand.yaml, hand/orca-v2-<side>). A side whose own v1
        # adapter is absent and whose v2 adapter is present comes up as that v2 hand.
        left_v2=_envs("HANDS_LEFT_V2_SERIAL", "0000000003"),
        right_v2=_envs("HANDS_RIGHT_V2_SERIAL", "0000000004"),
        # RESERVED pseudo-sides: serials that are NEVER a hand adapter. Any serial
        # registered to a non-matching "side" is excluded from the auto-claim in
        # detect_side_port. The RPLIDAR kit's CP2102 is a USB_VENDOR_IDS match, and
        # on 2026-07-08 the left-hand auto-claim actually grabbed it and sprayed
        # Feetech packets at the lidar (wedging it). Keep this in sync with
        # LIDAR.port below.
        _reserved_lidar=_envs("LIDAR_ADAPTER_SERIAL",
                              "00000000000000000000000000000005"),
    ),
    baudrate=_envi("HANDS_BAUD", 1000000),
    publish_rate=_envf("HANDS_RATE", 50.0),
    # Default FREE-DRIVE (torque OFF) on bring-up: a launch must NEVER command/energise
    # the hand. Energising a Feetech servo makes it snap to its (stale) goal-position
    # register, which looks like an unasked-for command. Torque is engaged explicitly,
    # later, via /<side>_hand/torque_enable (control center / teleop) — and even then the
    # controller holds-in-place first (no snap). Set MABEL_HW_HANDS_AUTO_TORQUE=true only
    # if you really want torque on at startup.
    auto_enable_torque=_envb("HANDS_AUTO_TORQUE", False),
    reconnect_interval=_envf("HANDS_RECONNECT", 2.0),   # s between idle-side retries
    # config_dir OVERRIDE: a dir holding <side>_hand.yaml (joint<->motor map). Empty
    # (default) => use the vendored packaged configs in hands/orcahand/configs. Note
    # calibration still resolves firmware-GUI-first regardless of this override; the
    # override dir's calibration file is only the fallback.
    config_dir=_envs("HANDS_CONFIG_DIR", ""),
)

# ---------------------------------------------------------------------------
# Lift (vertical column) — a BTS7960-driven DC motor + quadrature encoder on a
# Pico, closed-loop on the Pico, driven over USB serial. The bridge uses the
# proven host API at firmware/lift/host (lift.py -> Lift, metres). With no Pico on
# the port the subsystem advertises the /lift contract and stays idle.
# PORT: default "auto" resolves the Pico by USB vendor id (RP2040/RP2350 = 0x2E8A)
# instead of a fixed ttyACMx — ttyACM numbering is arrival-order, so the Pico is
# NOT reliably ACM0 (on the current hub ACM0 is the base Teensy). See the driver's
# _resolve_port(); override with LIFT_PORT=/dev/serial/by-id/... to pin one board.
# ---------------------------------------------------------------------------
LIFT = dict(
    enabled=_envb("LIFT", False),
    port=_envs("LIFT_PORT", "auto"),             # "auto" = find Pico by USB VID 0x2E8A
    baud=_envi("LIFT_BAUD", 115200),
    max_pos=_envf("LIFT_MAX_POS", 0.5),          # m; updated by on-Pico calibration
    auto_home=_envb("LIFT_AUTO_HOME", False),    # stall-seek the floor on startup
    joint_name=_envs("LIFT_JOINT", "lift"),
    publish_rate=_envf("LIFT_RATE", 50.0),
    topic_command="/lift/command",               # std_msgs/Float64 target height (m)
    topic_joint_states="/lift/joint_states",
    topic_calibrate="/lift/calibrate",           # std_msgs/Bool  (true = stall-seek floor+ceiling)
    topic_limits="/lift/limits",                 # std_msgs/String {min,max,calibrated} (latched)
)

# ---------------------------------------------------------------------------
# LiDAR — Slamtec RPLIDAR A2M12 (12 m, 360°) on its kit CP2102 USB-UART adapter,
# 256000 baud. The bridge is the SOLE owner of the serial port (replaces the old
# apt sllidar_ros2 node) and publishes the same /scan + lidar_link contract the
# sim / slam_toolbox / Nav2 already consume. ⚠ POWER: the adapter feeds the whole
# lidar from USB 5 V and the motor start surges ~1.5 A — it needs a powered hub /
# root port; on a weak port the lidar browns out the moment the motor engages
# (serial goes dark, only a replug revives it — the subsystem logs this pattern).
# ---------------------------------------------------------------------------
LIDAR = dict(
    enabled=_envb("LIDAR", True),
    port=_envs("LIDAR_PORT",
               "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_"
               "00000000000000000000000000000005-if00-port0"),
    # (no blind /dev/ttyUSB0 fallback: the subsystem falls back to ANY CP210x by
    # USB VID instead — a bare ttyUSB index can be a hand's CH340 adapter.)
    baud=_envi("LIDAR_BAUD", 256000),            # A2M12 (A1 = 115200)
    motor_pwm=_envi("LIDAR_MOTOR_PWM", 660),     # Slamtec default duty
    topic=_envs("LIDAR_TOPIC", "/scan"),
    frame=_envs("LIDAR_FRAME", "lidar_link"),    # sim + slam_toolbox contract
    urdf_frame="lidar",                          # URDF link; identity lidar->lidar_link TF
    bins=_envi("LIDAR_BINS", 720),               # 0.5° uniform grid per revolution
    range_min=_envf("LIDAR_RANGE_MIN", 0.15),
    range_max=_envf("LIDAR_RANGE_MAX", 12.0),    # A2M12 spec
    # How the unit is bolted on: REP-103 CCW yaw of the sensor's 0° mark in the
    # lidar frame. Re-measure against the URDF/sim if the mount changes.
    yaw_offset=_envf("LIDAR_YAW_OFFSET", 0.0),
)

# ---------------------------------------------------------------------------
# Audio — USB sound-card dongle + 8 Ω/5 W speaker pair + mic (JMTek/Solid State
# System "USB PnP Audio Device", VID:PID 0c76:1203 — replaced the MZ-631M bar
# 2026-07-27) on the USB hub. One duplex UAC1 device: ALSA card "Device" (the
# kernel's derivation from that generic product string — install/99-mabel-audio.rules
# renames it "MabelAudio"), hw:X,0 for BOTH playback and capture. The bridge is
# the SOLE owner of the sound card; the ROS node is a thin DDS relay
# (/audio/speaker in, /audio/mic out) exactly like the lidar's.
#
# ⚠ KEEP THE DEVICE PATH `plughw:`. This card DOES advertise 16 kHz natively
# (8k…48k S16_LE — /proc/asound/card*/stream0), but both directions are 2ch and
# our wire is MONO, so the plug layer's channel conversion is still load-bearing
# (and it keeps a device swap free: the old bar was 48 kHz/2ch-ONLY and a bare
# `hw:` at our format could not open at all).
#
# ⚠ PIN BY CARD NAME + VID:PID, *NEVER* BY SERIAL. This dongle reports the
# hard-coded serial "000000000000" (the old bar: "000000010000") — not unique, a
# second identical dongle would collide. `plughw:CARD=…` also survives card
# RENUMBERING (the USB card is 2 today, behind Thor's HDA + APE onboard cards,
# but that is arrival-order and a bare hw:2,0 silently becomes the wrong device
# when a card appears/disappears).
#
# ⚠ ASSERT THE MIXER ON CONNECT. The old bar booted its 'PCM' control at 0% —
# aplay then "succeeded" and played SILENCE, with no error anywhere. This card's
# playback control is 'Speaker' (verified audible at 62% boot level, 2026-07-27),
# but the assertion stays: do not assume a working device means an audible one.
# (Its 'Mic' control also has a PLAYBACK side — the sidetone loop to the speaker;
# leave that off or the mic feeds back. We only assert its CAPTURE side.)
# ---------------------------------------------------------------------------
AUDIO = dict(
    enabled=_envb("AUDIO", True),
    # ALSA device strings. CARD= is the card's *id* (`cat /proc/asound/cards`),
    # stable across renumbering; plughw = the converting plug layer (see above).
    # "Device" = the auto-derived id; the udev rule renames it "MabelAudio", and
    # detection retargets these strings onto whichever id the VID:PID resolves to.
    device_playback=_envs("AUDIO_DEVICE_PLAYBACK", "plughw:CARD=Device,DEV=0"),
    device_capture=_envs("AUDIO_DEVICE_CAPTURE", "plughw:CARD=Device,DEV=0"),
    card=_envs("AUDIO_CARD", "Device"),          # /proc/asound/cards id + amixer -c
    usb_id=_envs("AUDIO_USB_ID", "0c76:1203"),   # lsusb VID:PID (presence probe)
    # Wire format. 16 kHz mono S16_LE = 32 kB/s — voice-band, browser-native
    # (Web Audio decodes PCM16 with no codec), and the plug layer downmixes to
    # the hardware's stereo for free. Keep rate/channels in sync with
    # server/data_streams.py AUDIO_* and the apps' AudioWorklet.
    rate=_envi("AUDIO_RATE", 16000),             # Hz, wire + ALSA plug rate
    channels=_envi("AUDIO_CHANNELS", 1),         # mono both ways
    sample_bytes=2,                              # S16_LE — not env-tunable (wire contract)
    frame_ms=_envi("AUDIO_FRAME_MS", 20),        # 20 ms = 320 samples = 640 B/frame
    # Seconds of silence before the speaker PCM is released back to the card, so
    # a calibration studio / speaker-test can grab it between phrases.
    playback_idle_s=_envf("AUDIO_PLAYBACK_IDLE_S", 2.0),
    # Mixer assertion on connect (the 0%-volume scar above). None/-1 = leave alone.
    mixer_playback=_envs("AUDIO_MIXER_PLAYBACK", "Speaker"),  # amixer simple control
    mixer_capture=_envs("AUDIO_MIXER_CAPTURE", "Mic"),
    volume_playback=_envi("AUDIO_VOLUME_PLAYBACK", 70),    # % — 0 is silent-but-"working"
    volume_capture=_envi("AUDIO_VOLUME_CAPTURE", 100),     # %
    topic_speaker=_envs("AUDIO_TOPIC_SPEAKER", "/audio/speaker"),
    topic_mic=_envs("AUDIO_TOPIC_MIC", "/audio/mic"),
    topic_enable=_envs("AUDIO_TOPIC_ENABLE", "/audio/enable"),
    topic_state=_envs("AUDIO_TOPIC_STATE", "/audio/state"),
    topic_health=_envs("AUDIO_TOPIC_HEALTH", "/audio/device_health"),
)

# ---------------------------------------------------------------------------
# Cameras (native SDK ownership): D435 front + ZED-M head, republishing the
# existing /camera/front/* + /camera/zed/* contract so SLAM/nvblox runs unchanged.
# Mount extrinsics are copied from mabel_sensors/launch/{realsense,zed}.launch.py.
# ---------------------------------------------------------------------------
CAMERAS = dict(
    enabled=_envb("CAMERAS", True),
    # Bridge-hosted network video server: every bridge-owned camera streams to BOTH
    # ROS and this MJPEG server, so the GUI / apps see them on the same
    # /camera/<name>/stream.mjpg contract as the simulator (the network path bypasses
    # ROS to stay fast). MIRRORS THE SIM SERVER's port handling exactly: the SAME env
    # vars + default as server/config.py (MABEL_HTTP_HOST / MABEL_HTTP_PORT, default
    # :8080) — so a client (GUI, iOS, VP) hits the identical host:port whether the sim
    # server or the real bridge is the camera owner. The bridge is the SOLE camera
    # server (replaces the manual sensors/ros_camera_node dev node on :8080).
    serve=_envb("CAM_SERVE", True),
    http_host=os.getenv("MABEL_HTTP_HOST", "0.0.0.0"),
    http_port=int(os.getenv("MABEL_HTTP_PORT", "8080")),
    # Network-stream sizing (the dominant MJPEG latency lever). Every camera's frame
    # is downscaled to net_width (long edge) and JPEG'd at net_quality BEFORE the wire,
    # decoupled from capture/ROS resolution. 640 @ q65 ≈ 15-25 KB/frame → a phone keeps
    # up in real time on LAN. Raise net_width for sharper (more bandwidth/latency).
    net_width=_envi("CAM_NET_WIDTH", 640),
    net_quality=_envi("CAM_NET_QUALITY", 65),
    # 962x dedicated-port video plane (server/docs/data_streams.md is the contract; port
    # numbers mirror server/data_streams.py and are IDENTICAL sim vs real). One extra
    # VideoServer per port, one stream each — no mux contention with :8080. Only the
    # head SBS stream is high-res (native width, its own JPEG quality); wrist/front
    # keep the standard net_width. LAN/Tailscale direct only.
    dedicated=dict(
        enabled=_envb("CAM_DEDICATED", True),
        head_sbs_port=_envi("CAM_DEDI_HEAD_SBS_PORT", 9620),
        wrist_left_port=_envi("CAM_DEDI_WRIST_LEFT_PORT", 9621),
        wrist_right_port=_envi("CAM_DEDI_WRIST_RIGHT_PORT", 9622),
        front_port=_envi("CAM_DEDI_FRONT_PORT", 9623),
        sbs_quality=_envi("CAM_DEDI_SBS_QUALITY", 85),
    ),
    # Low-latency UDP video out (SOTA), additive to MJPEG and demand-driven. UDP drops
    # instead of queueing, so there's no TCP head-of-line backlog. H.264 = hardware NVENC
    # (Jetson Thor) on :9092; JPEG on :9091. Same 'MV' wire protocol as the sim/Vision Pro
    # (server/udp_video.py), so the iOS/VP clients work against sim OR real unchanged. Use
    # over LAN/Tailscale (UDP passes WireGuard); TCP-only paths fall back to MJPEG.
    udp=dict(
        enabled=_envb("CAM_UDP", True),
        jpeg=_envb("CAM_UDP_JPEG", True),
        h264=_envb("CAM_UDP_H264", True),
        jpeg_port=_envi("CAM_UDP_JPEG_PORT", 9091),
        h264_port=_envi("CAM_UDP_H264_PORT", 9092),
        # H.264 runs HIGH-RES (its own width, larger than the MJPEG net_width fallback):
        # inter-frame compression makes 720p cheap, so browser/headset H.264 clients get
        # a crisp image at low latency. Bitrate is CBR; bump for sharper at high motion.
        h264_width=_envi("CAM_H264_WIDTH", 1280),
        h264_bitrate=_envs("CAM_UDP_H264_BITRATE", "4M"),
        fps=_envi("CAM_UDP_FPS", 30),
    ),
    realsense=dict(
        enabled=_envb("CAM_REALSENSE", True),
        width=_envi("CAM_RS_WIDTH", 640),
        height=_envi("CAM_RS_HEIGHT", 480),
        fps=_envi("CAM_RS_FPS", 30),
        pointcloud=_envb("CAM_RS_POINTCLOUD", True),
        # IR projector: "off" (default) = clean IR stereo for cuVSLAM (the speckle
        # dot pattern is rigid to the CAMERA and wrecks visual odometry — measured
        # +12 m z-drift handheld); "on" = denser depth (80% vs 65% valid px) for
        # wheel-odom-only nvblox mapping where cuVSLAM is not used.
        emitter=_envs("CAM_RS_EMITTER", "off"),
        # Pin the FRONT camera to a serial (default '' = first non-D405 RealSense).
        # A serial-less rs.config grabs whichever device enumerates first — the D405
        # WRIST cam once hijacked the front pipeline this way (18 mm baseline on
        # /camera/front/*). D405s are always wrist cams (wrist_d405.py owns them).
        serial=_envs("CAM_RS_SERIAL", ""),
        # D435i accel+gyro -> /camera/front/imu for cuVSLAM VIO fusion. Publishes
        # only when the motion module enumerates (needs RSUSB-backend pyrealsense2
        # on Tegra — see realsense_d435.py docstring); harmless to leave on.
        imu=_envb("CAM_RS_IMU", True),
        base_frame="base_link",
        camera_frame="front_link",
        # base_link -> front_link mount (realsense.launch.py defaults).
        mount=dict(x=-0.25, y=0.0, z=0.13, roll=0.0, pitch=0.0, yaw=3.14159265),
        # Rotate the HUMAN video plane (MJPEG/UDP/H.264) 180° when the D435 is
        # physically mounted upside-down. The camera is now mounted RIGHT-SIDE-UP
        # (remounted 2026-07-10; verified against the raw stream + on-board IMU),
        # so this stays OFF — with it on, every app/GUI view is upside-down while
        # RViz (raw topics) looks fine. If the camera ever goes back in a flipped
        # bracket: set this True AND put roll=pi in `mount` above, else the
        # pointcloud/TF renders inverted in RViz (raw topics are never rotated).
        net_rotate180=_envb("CAM_RS_NET_ROTATE180", False),
    ),
    zed=dict(
        enabled=_envb("CAM_ZED", True),
        resolution=_envs("CAM_ZED_RES", "HD720"),
        fps=_envi("CAM_ZED_FPS", 30),
        depth=_envb("CAM_ZED_DEPTH", True),
        mount_frame="head",
        camera_frame="zed_camera_link",
        # head -> zed_camera_link mount. The `head` link frame is heavily rotated
        # (neck euler chain), so the OLD rpy (-pi/2,0,-pi/2) -- which assumed a body-
        # aligned parent -- left the optical z-forward pointing at base -Z (straight
        # DOWN: the cloud rendered under the ground). These angles make the NET
        # head -> zed_left_camera_optical_frame match camera_hal's MJCF-derived value
        # (quat 1,0,0,0 about head): optical z-forward = robot front (base -X),
        # y-down = base -Z. Derived as R_mount = R_target @ R_optical^-1 with
        # R_optical = _OPTICAL_RPY = (-pi/2,0,-pi/2)  ->  rpy = (-3pi/4, pi/2, -pi/4).
        mount=dict(x=0.0, y=0.006, z=-0.109,
                   roll=-2.3561945, pitch=1.5707963, yaw=-0.7853982),
    ),
    # Wrist cams. Per-side driver kind:
    #   "d405" — Intel RealSense D405 (pyrealsense2, wrist_d405.py): colour + depth +
    #            cloud on /camera/wrist_<side>/*, picked by device NAME containing
    #            "D405" (+ optional per-side serial pin for a future two-D405 rig).
    #   "uvc"  — Arducam Global Shutter (USB 32e4:0144, wrist_cams.py). Those two are
    #            ELECTRONICALLY IDENTICAL (same VID:PID AND serial 01.00.00), so they
    #            CANNOT be told apart by /dev/videoN or by-id — only by the physical
    #            USB hub port: 1-4.1.1 = RIGHT, 1-4.1.2 = LEFT (99-mabel-wrist-cams.udev
    #            pins those to /dev/wrist_{right,left}; by-path nodes are the fallback).
    # Current rig (2026-07-10, user-confirmed): BOTH wrists = D405 (the old Arducam
    # right-wrist pin is gone). With two D405s, pin CAM_WRIST_{LEFT,RIGHT}_SERIAL for
    # deterministic sides — REQUIRED under the per-process camera workers (each
    # worker scans in its own process, so an in-process "unclaimed" dedupe cannot
    # arbitrate between them; unpinned sides race for the first D405 by name).
    # SERIAL GOTCHA (2026-07-11): a D405 wears TWO serials — the pin below must be
    # the LIBREALSENSE serial (rs.camera_info.serial_number, what wrist_d405 logs
    # as "streaming"), NOT the sysfs/lsusb USB-descriptor iSerial. The RIGHT unit
    # is rs 128422271517 = sysfs 125423070861 (same camera, two numbers). An
    # unpinned side auto-resolves any D405 EXCEPT its sibling's pin (wrist_d405
    # exclude logic), so the left worker can never steal the right camera; pin
    # the left rs serial here as soon as the left unit enumerates and it's read
    # off the "wrist_left D405 (serial ...) streaming" log line. RSUSB quirk:
    # enable_device() on a D405 already claimed by the sibling worker fails with
    # the misleading text "No device connected" (it IS connected — just busy).
    wrist=dict(
        enabled=_envb("CAM_WRIST", True),
        left_kind=_envs("CAM_WRIST_LEFT_KIND", "d405"),
        right_kind=_envs("CAM_WRIST_RIGHT_KIND", "d405"),
        # D405 options: per-side serial pin ('' = first D405 by name minus the other
        # side's pin), and whether to stream depth (ROS depth image + cloud publish
        # only while subscribed). Pins are LIBREALSENSE serials — see gotcha above.
        # PHYSICAL UNIT REGISTRY (2026-07-11, both read off live hardware) — the
        # side assignment is BY SERIAL, so any USB port / hub / cable works:
        #   LEFT  wrist unit: rs 218622275378 (usb-descriptor iSerial 133323070711)
        #   RIGHT wrist unit: rs 128422271517 (usb-descriptor iSerial 125423070861)
        # If a unit is ever swapped/replaced, read the new rs serial from the
        # "wrist_<side> D405 (serial ...) streaming" bridge log line (or a
        # pyrealsense2 query while that unit is idle) and update it here.
        left_serial=_envs("CAM_WRIST_LEFT_SERIAL", "218622275378"),
        right_serial=_envs("CAM_WRIST_RIGHT_SERIAL", "128422271517"),
        # Depth OFF by default (2026-07-11): rs enable_stream is NOT lazy — depth
        # halves onto the USB wire even with zero cloud subscribers, and with BOTH
        # wrists on the expansion-board hub (2-3.2.4.x, shared uplink with the
        # front D435i) the extra ~300 Mbit stalled all three RealSenses: wrist
        # workers hit "Frame didn't arrive within 5000" ~1/min (frozen app tiles)
        # and the D435i logged 100+ uvcvideo -71 errors (2 before). Color-only
        # restores solid 30 fps wrist video. Set CAM_WRIST_D405_DEPTH=1 to get
        # wrist depth/clouds back (e.g. for grasp work) — ideally after moving one
        # camera off the shared hub (Thor root port 2-1 is free).
        d405_depth=_envb("CAM_WRIST_D405_DEPTH", False),
        # URDF link each camera rides on (the manager publishes
        # <mount_frame> -> wrist_<side>_camera_link -> wrist_<side>_camera_optical_frame
        # so the depth cloud renders attached to the moving arm in RViz). Values are
        # DERIVED FROM THE CANONICAL MJCF (mabel_full.xml l_wrist_cam / r_wrist_cam,
        # parent body = the PALM, not the wrist link):
        #     pos="0 0.048 -0.035"  xyaxes="-1 0 0  0 0.9438 0.3307"
        # MuJoCo cam (x-right, y-up, looks -z) -> ROS optical (x-right, y-down, z-fwd),
        # then rpy solved so R_mount @ R(_OPTICAL_RPY) == that camera orientation
        # (net view = palm +Z distal, tilted toward palm -Y — same alignment as the
        # sim's left_wrist camera). Not yet hand-eye calibrated against the REAL
        # D405 mount; re-derive if the MJCF camera pose changes.
        left_mount_frame="left_palm",
        right_mount_frame="right_palm",
        left_mount=dict(x=0.0, y=0.048, z=-0.035,
                        roll=0.0, pitch=-1.2337723, yaw=-1.5707963),
        right_mount=dict(x=0.0, y=0.048, z=-0.035,
                         roll=0.0, pitch=-1.2337723, yaw=-1.5707963),
        # 640x480 (NOT 1280x720): the two cams share ONE USB hub, and 720p MJPG x2
        # starved it to ~1-2 fps (measured). 480p restores full 30 fps, shrinks the
        # per-frame ROS image 4x, and is already at/under the net stream width — so the
        # app gets smooth, low-latency wrist video. Bump only if the hub has headroom.
        width=_envi("CAM_WRIST_WIDTH", 640),
        height=_envi("CAM_WRIST_HEIGHT", 480),
        fps=_envi("CAM_WRIST_FPS", 30),       # MJPG supports up to 60; 30 bounds shared-hub bw
        # Exposure: AUTO (aperture-priority) collapses to ~2 fps in dim light (the cam
        # stretches exposure to ~0.5 s). Force MANUAL (auto_exposure=1) + a short exposure
        # so the framerate holds; gain compensates the brightness. Set CAM_WRIST_AUTO_EXPOSURE=3
        # to restore auto (only do so under consistently bright light). exposure unit = x100us.
        auto_exposure=_envi("CAM_WRIST_AUTO_EXPOSURE", 1),   # 1=manual, 3=aperture-priority auto
        exposure=_envi("CAM_WRIST_EXPOSURE", 120),           # x100us -> 12 ms (leaves 30 fps headroom)
        gain=_envi("CAM_WRIST_GAIN", 180),                   # 16..255; raise if the image is too dark
        left_device=_envs(
            "CAM_WRIST_LEFT_DEV", "/dev/wrist_left"),
        right_device=_envs(
            "CAM_WRIST_RIGHT_DEV", "/dev/wrist_right"),
        # by-path fallbacks (used automatically if the /dev/wrist_* symlinks are absent).
        left_fallback=_envs(
            "CAM_WRIST_LEFT_FALLBACK",
            "/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2:1.0-video-index0"),
        right_fallback=_envs(
            "CAM_WRIST_RIGHT_FALLBACK",
            "/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.1:1.0-video-index0"),
    ),
)

# ---------------------------------------------------------------------------
# NEW-TAXONOMY subsystems (the mabel_hw API surface: left_arm / right_arm /
# neck / torso as separate handles). These drive DRIVER construction for the
# MabelHardware facade. The legacy ARMS/BODY dicts above keep driving the DDS
# adapters' topics/rates.
#
# backend: "openarm" — the openarm-dual protobuf firmware ACTUALLY FLASHED on the
#                      arms Teensy (drivers/openarm_link.py). Does NOT energize on
#                      connect: nothing is streamed until set_torque(True), which
#                      is why the arms may default ENABLED.
#          "teensy"  — upstream robot-host UDP protobuf link (the in-repo ArmLink
#                      gateway; self-describing RobotInfo; NOT yet flashed).
#                      ⚠ ENABLES motors on connect (damping hold) — any subsystem
#                      on this backend must default enabled=False.
#          "can"     — direct socketcan DaMiao MIT (CanLink; bench path; needs the
#                      joint_motors map string)
# ---------------------------------------------------------------------------
LINKS = dict(
    # The DEPLOYED arms firmware, on the direct NIC lane shared with the base.
    arms_openarm=dict(ip=_envs("ARMS_TEENSY_IP", "192.168.124.11"),
                      port=_envi("ARMS_PORT", 5800),
                      stream_hz=_envi("ARMS_STREAM_HZ", 100)),
    # The in-repo ArmLink gateway (not yet flashed).
    arms_teensy=dict(ip=_envs("ARMS_TEENSY_IP", "192.168.125.10"),
                     port=_envi("ARMS_TEENSY_PORT", 5800),
                     stream_hz=_envi("ARMS_TEENSY_STREAM_HZ", 1000)),
    neck_teensy=dict(ip=_envs("NECK_TEENSY_IP", "192.168.124.12"),
                     port=_envi("NECK_TEENSY_PORT", 5800),
                     stream_hz=_envi("NECK_STREAM_HZ", 1000)),
)

# ---------------------------------------------------------------------------
# ARM GAINS — per joint, because one number for all seven is not physical.
#
# J1/J2 swing the entire arm; J5/J6/J7 are the wrist. With NO END EFFECTOR fitted
# the wrist's inertia is orders of magnitude smaller than the shoulder's, and the
# closed loop does not care what we intended:
#
#     natural frequency   w  ~ sqrt(kp / J)          -> small J, MUCH faster
#     damping ratio       z  ~ kd / (2*sqrt(kp*J))   -> small J, MUCH less damped
#
# So the same kp=30 that is calm on the shoulder makes the unloaded wrist fast AND
# underdamped; it rings against the 50 Hz command rate and the link latency, and
# the arm buzzes. That is the "J5/J6/J7 gains are too high with no hand" report
# (2026-07-16). Lowering kp on the wrist is also the SAFE direction — less torque,
# not more.
#
# ⚠ THESE WRIST NUMBERS ARE A CONSERVATIVE STARTING POINT, NOT A MEASUREMENT.
# Scaled ~3.5x down from the shoulder's 30 and damped to keep z in the same
# ballpark. They are deliberately soft: a soft wrist tracks a little lazily, a
# stiff one oscillates and cooks the motor. TUNE from here — raise kp until the
# tracking is crisp, back off at the first sign of buzz, and re-tune once an end
# effector IS fitted (a hand adds inertia, so the wrist can take more kp again;
# that is what MABEL_HW_LEFT_ARM_KP_J5.. are for).
# ---------------------------------------------------------------------------
def _arm_gains(side: str) -> tuple:
    """(kp, kd) per-joint maps for one arm.

    Priority per joint: env override  >  controller/control_param.json  >  built-in fallback.
    The control-param file is the shared source of truth (see _control_params), so by default
    every joint gets the derived per-joint gain the studio and the controller use — a heavy
    shoulder (kp≈15, capped by kd≤5 damping) and a light wrist (kp≈8) instead of a flat kp=30
    that the shoulders cannot damp. A joint absent from the file keeps its historical fallback.
    MABEL_HW_<SIDE>_ARM_KP/_KD still set the whole-arm default and MABEL_HW_<SIDE>_ARM_KP_J5 etc.
    still pin one joint, so an operator can always override the file from the shell."""
    j = side[0].lower()                       # "LEFT" -> "l"
    file_gains = (_CONTROL_PARAMS.get("arm") or {}).get("gains") or {}

    def _gain(joint: int, field: str, fallback: float) -> float:
        # env override wins; else the control-param file; else the historical hardcoded value.
        env = os.environ.get(f"MABEL_HW_{side}_ARM_{field.upper()}_J{joint}")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                pass
        fg = file_gains.get(f"{j}_j{joint}")
        if isinstance(fg, dict) and field in fg:
            return float(fg[field])
        return fallback

    # Historical fallbacks: flat 30/1.0 for j1..j4, soft wrist triple for j5..j7.
    FALLBACK = {1: (30.0, 1.0), 2: (30.0, 1.0), 3: (30.0, 1.0), 4: (30.0, 1.0),
                5: (8.0, 0.4), 6: (8.0, 0.4), 7: (6.0, 0.3)}
    kp = {"default": _envf(f"{side}_ARM_KP", 30.0)}
    kd = {"default": _envf(f"{side}_ARM_KD", 1.0)}
    for joint, (fkp, fkd) in FALLBACK.items():
        kp[f"{j}_j{joint}"] = _gain(joint, "kp", fkp)
        kd[f"{j}_j{joint}"] = _gain(joint, "kd", fkd)
    return kp, kd


_LEFT_KP, _LEFT_KD = _arm_gains("LEFT")
_RIGHT_KP, _RIGHT_KD = _arm_gains("RIGHT")

LEFT_ARM = dict(
    enabled=_envb("LEFT_ARM", True),
    backend=_envs("LEFT_ARM_BACKEND", "openarm"),
    link=_envs("LEFT_ARM_LINK", "arms_openarm"),         # backend=openarm|teensy
    joints=["l_j1", "l_j2", "l_j3", "l_j4", "l_j5", "l_j6", "l_j7"],
    joint_motors=_envs("LEFT_ARM_JOINT_MOTORS", ""),     # backend=can
    kp=_LEFT_KP, kd=_LEFT_KD,           # per-joint maps — see _arm_gains above
    p_max=_envf("LEFT_ARM_PMAX", 12.5), v_max=_envf("LEFT_ARM_VMAX", 30.0),
    t_max=_envf("LEFT_ARM_TMAX", 10.0),
    calib_slice="arm/openarm-dual",                      # CalibrationStore path
    # END-STOP MARGIN (rad) held off each CALIBRATED hard stop when clamping a
    # target. The calibrated window IS the measured mechanical stop, so clamping a
    # command exactly TO it parks the motor against the stop at full kp*error — it
    # stalls, overheats, and trips the DaMiao's power-cycle over-current latch. This
    # is part of the POSITION clamp, not a torque guard: the clamp is only correct
    # if it stops SHORT of the stop. Measured 2026-07-16: l_j4's MJCF ctrlrange
    # equals its hard stop EXACTLY (margin -0.0000) and the joint RESTS 1.3 deg from
    # that stop, so every top-of-range command drove it into the stop — the "J4
    # always hits the limit and stalls / overheats" report. l_j7 could be commanded
    # 0.26 rad BEYOND its stop. clamp_raw drops the margin on the side a joint is
    # already past, so a joint resting inside the margin is HELD, never shoved in.
    end_stop_margin=_envf("LEFT_ARM_END_STOP_MARGIN", 0.05),
)

RIGHT_ARM = dict(
    enabled=_envb("RIGHT_ARM", True),
    backend=_envs("RIGHT_ARM_BACKEND", "openarm"),
    link=_envs("RIGHT_ARM_LINK", "arms_openarm"),
    joints=["r_j1", "r_j2", "r_j3", "r_j4", "r_j5", "r_j6", "r_j7"],
    joint_motors=_envs("RIGHT_ARM_JOINT_MOTORS", ""),
    kp=_RIGHT_KP, kd=_RIGHT_KD,         # per-joint maps — see _arm_gains above
    p_max=_envf("RIGHT_ARM_PMAX", 12.5), v_max=_envf("RIGHT_ARM_VMAX", 30.0),
    t_max=_envf("RIGHT_ARM_TMAX", 10.0),
    calib_slice="arm/openarm-dual",
    # END-STOP MARGIN (rad) held off each CALIBRATED hard stop when clamping a
    # target. The calibrated window IS the measured mechanical stop, so clamping a
    # command exactly TO it parks the motor against the stop at full kp*error — it
    # stalls, overheats, and trips the DaMiao's power-cycle over-current latch. This
    # is part of the POSITION clamp, not a torque guard: the clamp is only correct
    # if it stops SHORT of the stop. Measured 2026-07-16: l_j4's MJCF ctrlrange
    # equals its hard stop EXACTLY (margin -0.0000) and the joint RESTS 1.3 deg from
    # that stop, so every top-of-range command drove it into the stop — the "J4
    # always hits the limit and stalls / overheats" report. l_j7 could be commanded
    # 0.26 rad BEYOND its stop. clamp_raw drops the margin on the side a joint is
    # already past, so a joint resting inside the margin is HELD, never shoved in.
    end_stop_margin=_envf("RIGHT_ARM_END_STOP_MARGIN", 0.05),
)

# NECK — the AS-BUILT serial neck: a 3-link SERIAL chain, yaw (DaMiao DMJ4310, at
# the base) -> pitch (Dynamixel) -> roll (Dynamixel, carries the camera). It is a
# DIRECT-DRIVE serial chain, so task space == motor space: pitch commands the pitch
# motor 1:1 (no linkage IK) — see subsystems/neck.py::NeckSubsystem.
#
# WIRING SINCE 2026-09-30 (backend `openarm`, the default): ALL THREE neck motors
# hang on the ARMS TEENSY. Yaw is a DM-J4310 on its torso CAN bus; pitch/roll are
# one ROBOTIS 2XL430-W250-T (two axes, ids 1/2) on the Teensy's half-duplex serial
# port (Serial3, 1 Mbaud — upstream firmware `dxl_actuator_transport`, 2026-08-26).
# The firmware advertises them as ordinary indexed joints `neck_pitch`/`neck_roll`
# with p_max/v_max/t_max = 0 BY DESIGN (a Dynamixel has no MIT scale), reports RAW
# servo radians (tick 2048 = 0 rad), and translates the MIT command as: kp > 0 =
# hold the goal position, kp == 0 = RELEASE torque; kd/vel/t_ff are ignored (the
# servo's own position loop runs). So the whole neck is ONE OpenArmGroup on the
# shared link — nothing energizes on connect; set_torque holds at the measured pose.
# Calibration: slice `neck/teensy` (center-anchored, seeded from the upstream
# 2026-08-26 capture of this robot — verify with Setup Studio ▸ Neck ▸ Set Level).
#
# NECK_BACKEND=dynamixel reverts to the OLD wiring: XC330-T181 servos on the FT232H
# USB adapter (/dev/ttyUSB0) + yaw via `yaw_backend` (slice `neck/serial`).
NECK = dict(
    enabled=_envb("NECK", True),
    backend=_envs("NECK_BACKEND", "openarm"),
    link="arms_openarm",                                 # the 192.168.124.11 Teensy
    # Yaw DEFAULT ON since 2026-07-27 (DM-J4310, ESC 2 / MST 0x12 on the Teensy's
    # torso bus). MABEL_HW_NECK_YAW=false drops it from the group.
    joints=(["neck_yaw"] if _envb("NECK_YAW", True) else []) +
           ["neck_pitch", "neck_roll"],
    # openarm backend: per-joint MIT gains. Yaw is a real MIT loop (soft — camera
    # mass only, see yaw_kp below). For the Dynamixels the wire gains are NOT a
    # control law: any kp > 0 means "hold" (the servo's internal Position P gain,
    # register 84, does the holding), so 1.0 is a flag, not a stiffness.
    kp={"neck_yaw": _envf("NECK_YAW_KP", 6.0),
        "neck_pitch": _envf("NECK_DXL_KP", 1.0), "neck_roll": _envf("NECK_DXL_KP", 1.0)},
    kd={"neck_yaw": _envf("NECK_YAW_KD", 0.5), "neck_pitch": 0.0, "neck_roll": 0.0},
    # Torque-enable guard (openarm.py): only yaw has a DaMiao flash T_MAX to check
    # the SD manifest against; the Dynamixels advertise 0 by design, so no entry.
    wire_t_max={"neck_yaw": _envf("NECK_YAW_WIRE_TMAX", 10.0)},
    calib_slice=("neck/teensy" if _envs("NECK_BACKEND", "openarm") == "openarm"
                 else "neck/serial"),
    # ── legacy `dynamixel` backend (USB FT232H adapter) ──────────────────────
    # servo id map on the shared TTL bus. pitch is the middle joint, roll the last
    # (camera). Swap here (or MABEL_HW_NECK_DXL_IDS) if the physical wiring differs.
    dxl_ids=_envs("NECK_DXL_IDS", "neck_pitch:1, neck_roll:2"),
    device=_envs("NECK_DEVICE", "/dev/ttyUSB0"),
    device_serial=_envs("NECK_DEVICE_SERIAL", "FTB8HR1Y"),  # by-id pin (replug-safe)
    baud=_envi("NECK_BAUD", 1000000),
    # Yaw = the DaMiao DM-J4310 at the base of the chain. Since 2026-07-28 it
    # hangs on the ARMS TEENSY's torso CAN bus (advertised as `neck_yaw` in the
    # SD-card robot.json manifest — default backend "openarm", shared link, no
    # energize on connect). NECK_YAW_BACKEND=fdcan reverts to the DM-USB2FDCAN
    # dongle (USB 34b7:6877 — drivers/damiao_fdcan.py). Commissioned ESC id 2 /
    # MST 0x12 (torso owns 1/0x11) — the SD manifest's can_id must match.
    # Gains deliberately soft — direct drive, camera mass only.
    yaw_backend=_envs("NECK_YAW_BACKEND", "openarm"),
    yaw_can_id=_envi("NECK_YAW_CAN_ID", 2),
    # Softened 2026-07-29 (was 20/1.0): the head is ≤0.5 kg, so J_eff about yaw is a
    # few mΩ·m² — kp 20 made it rigid. kp 6/kd 0.5 tracks the gaze stream and yields
    # to a hand; pitch/roll get the matching treatment via the Dynamixels' internal
    # Position P Gain (drivers/dynamixel.py softens reg 84 to MABEL_HW_NECK_DXL_P_GAIN,
    # default 300 vs the ≈900 factory value).
    yaw_kp=_envf("NECK_YAW_KP", 6.0), yaw_kd=_envf("NECK_YAW_KD", 0.5),
    yaw_p_max=_envf("NECK_YAW_PMAX", 12.5), yaw_v_max=_envf("NECK_YAW_VMAX", 30.0),
    yaw_t_max=_envf("NECK_YAW_TMAX", 10.0),
    # The DM-J4310's flash-register T_MAX — the openarm torque-enable guard
    # refuses to energize while the SD manifest advertises anything else.
    yaw_wire_t_max=_envf("NECK_YAW_WIRE_TMAX", 10.0),
    yaw_slew=_envf("NECK_YAW_SLEW", 1.5),        # adapter glide speed (rad/s)
)

TORSO = dict(
    # Ships ENABLED: the default `openarm` backend rides the arms Teensy's shared
    # UDP link (drivers/openarm_link.py) and does NOT energize on connect —
    # nothing streams until set_torque. Physical wiring since 2026-07-28: the
    # torso DM-J10422P hangs on the arms Teensy's THIRD CAN bus (bus 3, CAN id
    # 1 / MST 0x11 — no collision with the arms, which mirror ids on buses 1+2),
    # so the Teensy's SD-card /robot.json must advertise a `torso` actuator or
    # the group stays UNRESOLVED (hot-joins when the manifest gains it).
    # TORSO_BACKEND=serial reverts to the HDSC USB-CAN dongle (drivers/solo.py
    # over drivers/damiao_serial.py) if the torso is ever rewired back. The
    # legacy `can` (socketcan) and `teensy` backends DO energize on connect —
    # re-disable if you switch to one of those.
    enabled=_envb("TORSO", True),
    backend=_envs("TORSO_BACKEND", "openarm"),           # via the arms Teensy CAN3
    link="arms_openarm",                                 # the flashed 192.168.124.11 link
    joints=["torso"],
    joint_motors=_envs("TORSO_JOINT_MOTORS", "torso:can0:1"),
    # MIT gains. History: 2026-07-26 measured ~7-10 N·m stiction (22:1 gear) and,
    # with NO gravity feedforward, kp 400-500 was the only way to move/hold — but
    # through the HAL that felt "rigid and jittery / unsafe" (2026-07-27), and the
    # single-sinusoid t_ff that let kp drop to 150 was fitted at ONE arm pose, so
    # kp 150 was still covering the arm-extended FF error ("too stiff",
    # 2026-07-29). Now the gateway streams an ARM-POSE-AWARE t_ff (the shared
    # arm+torso MJCF evaluation, live-trimmed — controller/mabel/torso_control
    # posture_tau) plus TARGET VELOCITY with every /body/command, so PD only
    # fights friction + tracks. Measured live 2026-07-29 (0.25 rad/s profile,
    # settled-hold worst / moving max): kp 60 → 2.1°/4.0°, kp 90 → 1.2°/3.2°,
    # kp 120 → 0.7°/2.7° — all glass-smooth (zero hold jitter, ≤2 velocity sign
    # flips per sweep); the residual error is the gear's ±2 N·m rest-friction
    # band over kp, so dial TORSO_KP for the accuracy/compliance trade. NO
    # integral trim: torque ramped against the ~7-10 N·m breakaway just
    # limit-cycles (tested). Mirror changes in control_param.json torso.hardware.
    kp=_envf("TORSO_KP", 90.0), kd=_envf("TORSO_KD", 4.0),
    # Adapter-side glide speed (rad/s) — the MIT wire has no profile generator.
    slew=_envf("TORSO_SLEW", 0.8),
    p_max=_envf("TORSO_PMAX", 12.5), v_max=_envf("TORSO_VMAX", 30.0),
    t_max=_envf("TORSO_TMAX", 10.0),
    # The DM-J10422P's flash-register T_MAX (500 — studio torso_calib.py). The
    # openarm backend REFUSES torque enable while the Teensy's SD manifest
    # advertises a different limit (a smaller manifest limit would multiply
    # every commanded torque at the motor — seen 5x with the stale SD's 100).
    wire_t_max=_envf("TORSO_WIRE_TMAX", 500.0),
    # DaMiao USB-CAN serial dongle (drivers/damiao_serial.py) — the physical wiring
    # today: one DM-J10422P on the HDSC dongle (USB VID 0x2E88), CAN id 1 / MST 0x11.
    # "auto" resolves the dongle by USB vendor id, replug-safe (the lift's Pico pattern).
    # The setup studio's torso calibration + slider control talk this port directly.
    port=_envs("TORSO_PORT", "auto"),
    serial_baud=_envi("TORSO_SERIAL_BAUD", 921600),
    can_id=_envi("TORSO_CAN_ID", 1),
    calib_slice="torso/damiao",                          # setup-studio manual wizard
    # Never PARK the stiff (kp 450) MIT hold against a measured stop: clamped
    # targets stay this far inside the calibrated window (the studio slider
    # learned the same lesson — "it is hitting the max limit", 2026-07-26; seen
    # live 2026-07-27 when a stale over-window target drove the torso onto
    # raw_max and held it there).
    end_stop_margin=_envf("TORSO_END_STOP_MARGIN", 0.03),
)

# CalibrationStore knobs (root "" => <bridge>/calibration_data).
CALIBRATION = dict(
    root=_envs("CALIB_ROOT", ""),
    watch=_envb("CALIB_WATCH", True),
    watch_period=_envf("CALIB_WATCH_S", 2.0),
)
