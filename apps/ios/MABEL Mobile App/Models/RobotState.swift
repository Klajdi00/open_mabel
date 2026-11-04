//
//  RobotState.swift
//  MABEL Mobile App
//
//  Snapshot of what the robot reports back to the operator each tick:
//  pose, power, motion and a diagnostics roll-up.
//

import Foundation

struct RobotState: Equatable {
    var timestamp: TimeInterval = 0

    // Power
    var batteryLevel: Double = 0.78      // 0...1
    var batteryVoltage: Double = 48.4
    var charging: Bool = false
    /// False when the robot has no battery monitor (sim, or real rig without a
    /// BMS): the gauge shows n/a + `batteryReason` instead of a fake constant.
    var batteryAvailable: Bool = true
    var batteryReason: String = ""

    // Motion (base, in robot frame)
    var linearSpeed: Double = 0          // m/s
    var angularSpeed: Double = 0         // rad/s
    var heading: Double = 0              // rad, world frame (yaw)
    /// Full chassis orientation as the MuJoCo world quaternion [w, x, y, z].
    /// Identity = upright; a non-level value tips the viewer so a fall is visible.
    var baseQuat: [Double] = [1, 0, 0, 0]
    /// Base odom position (ROS frame, metres) — robot-pose fallback for the
    /// 3D viewer when the dedicated map/odom stream is disabled.
    var baseX: Double = 0
    var baseY: Double = 0

    // Data-collection status, reported by the server (recorder).
    var recordTask: String = ""          // current task folder
    var episodeCount: Int = 0            // episodes already saved in that task
    var taskList: [String] = []          // existing task folders on the server
    var taskCounts: [String: Int] = [:]  // episodes per task folder
    var episodesTotal: Int = 0           // episodes across all tasks
    var episodesDir: String = ""         // where episodes are written on the server
    var serverRecording: Bool = false    // the recorder is actively writing

    // Health
    var cpuTemp: Double = 46             // °C
    var wifiRSSI: Int = -52              // dBm
    /// TELEMETRY THE LINK ITSELF CANNOT CARRY. Bluetooth moves pose and safety flags
    /// and nothing else — no battery gauge, no CPU thermal, no round-trip to measure. The
    /// panel used to render this struct's DEFAULTS there (78 %, 46 °C, 18 ms), which is a
    /// robot inventing its own vitals. False means "show —", not "zero".
    var vitalsAvailable = true
    var latencyMs: Double = 18           // measured WebSocket round-trip (see WebSocketLink.ping)
    var stateAgeMs: Double = 0           // server-side freshness of the pose (control-loop lag)

    // Pose — every joint we expose, in radians/metres.
    var joints: [String: Double] = RobotModel.homePose
    /// Which joints the robot ACTUALLY reported in the newest frame. `joints` above is
    /// pre-filled with `RobotModel.homePose` so the 3D model always has a pose to draw, and
    /// that makes "the robot told us" indistinguishable from "nobody has said yet" — a
    /// drawn default reads exactly like a measurement. Anything that must not act on a
    /// guess (the lift: see `RobotController.liftAdopted`) asks this instead.
    var reportedJoints: Set<String> = []
    /// Operator switches ECHOED by the server, so the UI shows what the robot is
    /// actually doing rather than what this client last asked for. They differ
    /// after a reconnect, or when a second client drives the same robot — and a
    /// toggle whose UI can disagree with the robot is worse than no toggle.
    /// OPTIONAL on purpose: `nil` means "this state carried no opinion", which
    /// is what an older server — or the on-device demo, which has no control law
    /// at all — reports. Defaulting them to a concrete value made the echo
    /// overwrite the operator's own tap on the very next state tick, so the
    /// toggle sprang back and nothing appeared to happen. Matches the wire's
    /// additive semantics: absent = unchanged.
    var gravityComp: Bool?
    var wholeBodyRetarget: Bool?

    /// Flat per-body world transforms from the sim, [x,y,z, qw,qx,qy,qz] × nbody
    /// in MuJoCo body-id order — drives the faithful MuJoCo mirror view.
    var bodyXforms: [Double] = []

    var diagnostics: [Diagnostic] = Diagnostic.nominal

    /// Latest planar LiDAR sweep, attached by the bridge only when fresh (~5 Hz).
    /// A range ≥ rangeMax is a non-return and should be treated as "no obstacle".
    /// Same contract for the sim's ray-cast sweep and the real RPLIDAR via /scan.
    var lidar: LidarScan? = nil

    /// Sparse joint dynamics (~10 Hz): streamed velocities + actuator torques,
    /// attached only when a fresh sample exists (same stamp-gated contract as
    /// `lidar`) — the controller latches the last one. Keyed by the same iOS
    /// joint ids as `joints`.
    var jointVelocities: [String: Double]? = nil
    var jointTorques: [String: Double]? = nil
    /// Per-joint actuator temperature (°C) when the robot measures it (real
    /// HAL); absent = honestly "not measured", never a guess.
    var jointTemps: [String: Double]? = nil
    /// `motorHealth` fallback (older real-robot bridges): per-joint torque /
    /// temperature from the merged */motor_health topics (~2 Hz, often a
    /// PARTIAL frame). The controller MERGES these into the same latches the
    /// sparse dynamics fill, so the telemetry panels work either way.
    var healthTorques: [String: Double]? = nil
    var healthTemps: [String: Double]? = nil

    // ── Server-authoritative status echoes (2026-07 data-plane update). All
    // additive: nil / empty when talking to an older server or the demo.
    /// E-stop state as the SERVER sees it — reflects every client's estop,
    /// not just ours. nil = not reported.
    var estop: Bool? = nil
    /// Per-subsystem torque enable (left_hand / right_hand / arms / base).
    /// A subsystem absent from the dict = state unknown (server sent null).
    var torqueState: [String: Bool] = [:]
    var moduleTorque: [String: Bool] = [:]
    var stopUnconfirmed: [String]? = nil
    /// Controller-reported state only; this does not measure physical motor power.
    var controllerStopAcknowledged: Bool {
        guard estop == true else { return false }
        let flags = moduleTorque.isEmpty ? torqueState.filter { $0.key != "any" } : moduleTorque
        let nonlift = flags.filter { !$0.key.lowercased().contains("lift") }
        return nonlift.isEmpty ? torqueState["any"] == false : !nonlift.values.contains(true)
    }
    var controllerStopConfirmed: Bool {
        controllerStopAcknowledged && stopUnconfirmed?.isEmpty != false
    }
    /// Active control-spec method echoed by the server ("navigation",
    /// "retarget", "joint", …) — what the robot is ACTUALLY running.
    var controlMethod: String = ""
    var dataCollection: DataCollection? = nil
    var policy: PolicyEcho? = nil
    var animation: AnimationEcho? = nil

    /// Per-module connection roster (`robot_state.modules`, server/module_status.py):
    /// which limb / sensor is actually attached RIGHT NOW. Ordered by the SERVER —
    /// render it in the order it arrives, don't re-sort. Empty against an older
    /// bridge or in demo mode, which is why the panel section hides when it is.
    ///
    /// On the real robot a subsystem whose device is gone publishes nothing at all,
    /// so `ok == false` is measured, not guessed; in sim the actuators are
    /// "connected" by definition (nothing to unplug) while the cameras and lidar are
    /// genuinely measured.
    var modules: [Module] = []

    struct Module: Equatable, Identifiable {
        var id: String            // "leftArm", "camHead", …
        var label: String         // server-owned display name ("Left Arm")
        var group: String?        // "sensors" for the 4 cams + lidar; nil otherwise
        var ok: Bool
        var detail: String        // "7/7", "J3, J6", "not served", "link down"

        // ── PER-ACTUATOR HEALTH (server/module_status.py) ────────────────────
        // A module is not one device: an arm is seven motors on a bus, a hand is
        // seventeen. "Left Arm: DOWN" is true but useless — it does not say whether
        // the adapter is unplugged or one elbow connector is loose. So the roster
        // carries the COUNT (the headline: 7/7 vs 5/7) and one entry PER MOTOR, and
        // the panel names the ones that are silent.
        //
        // n == 0 means this build has NO DRIVER for the module (NOVA's hands and
        // swerve base today). That is not a fault, and it must not be painted as one.
        // OPTIONAL, and the distinction matters: `nil` means the server never mentioned
        // motor counts (an older bridge — MABEL's roster), which is NOT the same claim as
        // `0`, which means "this build has no driver for it". Conflating them would paint
        // every module on an older robot grey as if it were unfitted.
        var up: Int?
        var n: Int?
        var actuators: [Actuator] = []

        /// One physical motor: `right_arm_3` shown as "J3", `fl_steer` as "FL Steer".
        struct Actuator: Equatable, Identifiable {
            var id: String
            var label: String
            var ok: Bool
            var detail: String    // "38 °C" when live, "no reply" when silent
        }

        /// "5/7" — the headline verdict, empty when there are no motors to count (or the
        /// server does not report them, in which case the row falls back to OK / DOWN).
        var countText: String {
            guard let n, let up, n > 0 else { return "" }
            return "\(up)/\(n)"
        }
        /// False ONLY when the server explicitly says this module has no motors. A server
        /// that says nothing is assumed to be fitted — silence is not a claim.
        var hasDriver: Bool { n.map { $0 > 0 } ?? true }

        /// The three states the panel distinguishes. "Waiting" is NOT "down": before
        /// the first roster arrives (or against a bridge too old to send one) we do
        /// not know anything, and painting a healthy robot's limbs red would be a lie.
        enum Health { case up, down, waiting }

        /// Shown until the first roster arrives, so the panel is NEVER blank and the
        /// operator sees the shape of the robot immediately. Mirrors the server's
        /// order and labels (server/module_status.py MODULES) — if that list changes,
        /// the real roster simply replaces this one on the next heartbeat.
        static let placeholder: [Module] = [
            ("base", "Base", nil), ("torso", "Torso", nil), ("lift", "Lift", nil),
            ("neck", "Neck", nil), ("leftArm", "Left Arm", nil),
            ("rightArm", "Right Arm", nil), ("leftHand", "Left Hand", nil),
            ("rightHand", "Right Hand", nil),
            ("camHead", "Head Cam", "sensors"), ("camFront", "Front Cam", "sensors"),
            ("camWristLeft", "L Wrist Cam", "sensors"),
            ("camWristRight", "R Wrist Cam", "sensors"), ("lidar", "LiDAR", "sensors"),
            // The USB bar's two halves — AUDIO, not sensors: the speaker EMITS.
            // Kept in step with server/module_status.py MODULES (15 rows).
            ("speaker", "Speaker", "audio"), ("mic", "Microphone", "audio"),
        ].map { Module(id: $0.0, label: $0.1, group: $0.2, ok: false, detail: "waiting") }
    }

    struct DataCollection: Equatable {
        var active = false
        var label: String? = nil
        var elapsedSec: Double = 0
        var error: String? = nil
    }
    struct PolicyEcho: Equatable {
        var selected: String? = nil
        var running = false
    }
    struct AnimationEcho: Equatable {
        var selected: String? = nil
        var playing = false

        // ── Animation demos (animation_demo/) ────────────────────────────────
        // Recording and replay are BOTH reported here, because the console shows
        // one control for them and the operator needs a single truth for "is the
        // robot busy". Every field is server-authored: the countdown in
        // particular must not be run locally, or the "recording starts now" the
        // operator sees is off by the link latency.
        var recording = false
        /// Armed but still counting down — the operator's three seconds to get
        /// into the start position.
        var armed = false
        var countdown: Double = 0
        /// Clip being recorded or played.
        var name: String? = nil
        var frames = 0
        var elapsed: Double = 0
        /// "lead_in" | "play" | "return" while replaying.
        var phase: String? = nil
        var rep = 0
        var repeats = 0
        var frac: Double = 0
        var clips: [DemoClipInfo] = []

        var busy: Bool { recording || armed || playing }
    }

    /// One saved demo clip, as the server lists it.
    struct DemoClipInfo: Equatable, Identifiable, Hashable {
        var name: String
        var duration: Double
        var frames: Int
        var created: String
        var recordedBy: String
        var joints: Int
        var id: String { name }
    }

    struct LidarScan: Equatable {
        var angleMin: Double = 0
        var angleInc: Double = 0
        var rangeMax: Double = 12
        var ranges: [Double] = []
        var stamp: TimeInterval = 0      // server clock, for staleness display
    }

    struct Diagnostic: Identifiable, Equatable {
        enum Level: Equatable { case ok, warn, fault }
        var id: String { subsystem }
        var subsystem: String
        var message: String
        var level: Level

        static let nominal: [Diagnostic] = [
            .init(subsystem: "Swerve Drive", message: "3/3 modules nominal", level: .ok),
            .init(subsystem: "Lift Column",  message: "Within travel limits", level: .ok),
            .init(subsystem: "Left Arm",     message: "Nominal",              level: .ok),
            .init(subsystem: "Right Arm",    message: "Joint 3 near limit",   level: .warn),
            .init(subsystem: "Vision",       message: "RGB-D streaming 30 fps", level: .ok),
            .init(subsystem: "Compute",      message: "Jetson nominal",       level: .ok),
        ]
    }
}
