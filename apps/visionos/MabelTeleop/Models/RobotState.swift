//
//  RobotState.swift
//  MABEL Mobile App
//
//  Snapshot of what the robot reports back to the operator each tick:
//  pose, power, motion and a diagnostics roll-up.
//

import Foundation

struct RobotState: Equatable {

    // MARK: - Animation demos (animation_demo/)
    //
    // iPhone parity. Every field is SERVER-authored — the countdown especially:
    // run locally it would be off by whatever the link latency was, and the
    // operator's "recording starts now" has to be true.
    var demo: DemoEcho? = nil

    struct DemoEcho: Equatable {
        var recording = false
        /// Armed but still counting down — the operator's seconds to get into
        /// the start position.
        var armed = false
        var countdown: Double = 0
        var playing = false
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

    struct DemoClipInfo: Equatable, Identifiable, Hashable {
        var name: String
        var duration: Double
        var frames: Int
        var created: String
        var recordedBy: String
        var joints: Int
        var id: String { name }
    }
    var timestamp: TimeInterval = 0

    // Power
    var batteryLevel: Double = 0.78      // 0...1
    var batteryVoltage: Double = 48.4
    var charging: Bool = false

    // Motion (base, in robot frame)
    var linearSpeed: Double = 0          // m/s
    var angularSpeed: Double = 0         // rad/s
    var heading: Double = 0              // rad, world frame (yaw)
    /// Full chassis orientation as the MuJoCo world quaternion [w, x, y, z].
    /// Identity = upright; a non-level value tips the viewer so a fall is visible.
    var baseQuat: [Double] = [1, 0, 0, 0]
    /// Chassis world position (odom frame, metres) — robot_state.base x/y.
    /// Drives the sensor-world centering when the /map odom stream is idle
    /// (the bare Mac sim path).
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
    /// True once a link has reported the recorder at all — the Rec button then follows the ROBOT, not the tap.
    var recordingReported = false
    var recordElapsed: Double = 0        // seconds into the take, robot clock
    /// What the robot said went wrong (a refused take, a camera that never delivered, a refused replay).
    var recordError: String? = nil
    /// Every camera stream of the take and its frame count, so the operator can see they are all writing.
    var recordCameras: [RecordCamera] = []

    struct RecordCamera: Equatable, Identifiable {
        var name: String
        var frames: Int
        var live: Bool
        var error: Bool
        var id: String { name }
    }

    // Per-operator body calibration (controller/mabel/calibration.py):
    //  progress of the guided capture (nil when idle) + the measured anthropometry
    //  (nil until a profile exists). Drive the calibration flow + Settings profiles.
    var calibration: CalibrationProgress? = nil
    var operatorMetrics: OperatorMetrics? = nil

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
    /// after a reconnect, or when a second client drives the same robot.
    /// OPTIONAL on purpose: `nil` means "this state carried no opinion", which
    /// is what an older server — or the on-device demo, which has no control law
    /// at all — reports. Defaulting them to a concrete value made the echo
    /// overwrite the operator's own tap on the very next state tick, so the
    /// toggle sprang back and nothing appeared to happen. Matches the wire's
    /// additive semantics: absent = unchanged.
    var gravityComp: Bool?
    var wholeBodyRetarget: Bool?
    /// E-stop as the SERVER sees it — ONE flag shared by every client (iPhone,
    /// headset, studio). nil = not reported (demo / older bridge).
    var estop: Bool? = nil
    /// What the robot's last torque-on left LIMP, {module: reason} (robot_state.torqueDegraded, 2026-10-25).
    /// Torque-on degrades instead of refusing: a missing hand or a silent motor stays off, the rest comes up.
    var torqueDegraded: [String: String] = [:]

    /// OPTIONAL per-joint velocities (rad/s · m/s) and torques (Nm) — attached
    /// only by bridges that stream them (sim bridge ≥ the V2 extension, sparse
    /// ~10 Hz). Empty when the wire doesn't carry them; the graphs then derive
    /// velocity numerically and state torque as "not streamed" (never faked).
    var jointVelocities: [String: Double] = [:]
    var jointTorques: [String: Double] = [:]

    /// Flat per-body world transforms from the sim, [x,y,z, qw,qx,qy,qz] × nbody
    /// in MuJoCo body-id order — drives the faithful MuJoCo mirror view.
    var bodyXforms: [Double] = []

    var diagnostics: [Diagnostic] = Diagnostic.nominal

    /// Latest planar LiDAR sweep, attached by the bridge only when fresh (~5 Hz).
    /// A range ≥ rangeMax is a non-return and should be treated as "no obstacle".
    /// Same contract for the sim's ray-cast sweep and the real RPLIDAR via /scan.
    var lidar: LidarScan? = nil

    /// Per-module connection roster (robot_state.modules, server/module_status.py):
    /// which limb / sensor is actually attached. Ordered by the server; empty against
    /// an older bridge / demo. Ported from iOS v2.
    var modules: [Module] = []

    struct Module: Equatable, Identifiable {
        var id: String
        var label: String
        var group: String?        // "sensors" for the 4 cams + lidar; nil otherwise
        var ok: Bool
        var detail: String

        // ── PER-ACTUATOR HEALTH (server/module_roster.py) ────────────────────
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

        enum Health { case up, down, waiting }
        /// Shown until the first roster arrives, so the panel is never blank.
        static let placeholder: [Module] = [
            ("base", "Base", nil), ("torso", "Torso", nil), ("lift", "Lift", nil),
            ("neck", "Neck", nil), ("leftArm", "Left Arm", nil),
            ("rightArm", "Right Arm", nil), ("leftHand", "Left Hand", nil),
            ("rightHand", "Right Hand", nil),
            ("camHead", "Head Cam", "sensors"), ("camFront", "Front Cam", "sensors"),
            ("camWristLeft", "L Wrist Cam", "sensors"),
            ("camWristRight", "R Wrist Cam", "sensors"), ("lidar", "LiDAR", "sensors"),
        ].map { Module(id: $0.0, label: $0.1, group: $0.2, ok: false, detail: "waiting") }
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
