//
//  RobotLink.swift
//  MabelTeleop
//
//  Transport abstraction between the app and MABEL — the same protocol the
//  iOS app uses, plus the headset-only `recalibrate` clutch re-anchor.
//

import Foundation

/// A command the operator issues. Encoded to the wire by a real link.
enum RobotCommand {
    /// Base twist: forward (m/s), strafe (m/s), yaw (rad/s).
    case twist(forward: Double, strafe: Double, yaw: Double)
    /// Absolute joint target in the joint's native units.
    case jointTarget(id: String, value: Double)
    /// Lift column height fraction 0...1 (maps both lift stages).
    case liftFraction(Double)
    /// Lift velocity command from a joystick, −1 (down) … +1 (up).
    case liftRate(Double)
    /// Operator head pose for active perception — a 4×4 row-major transform
    /// (16 floats). Used by the demo link; the WebSocket link streams the real
    /// ARKit head pose inside each TeleopFrame instead.
    case headPose([Double])
    /// Structured control spec — the orthogonal teleop axes.
    case controlSpec(tracking: [String: Bool] = [:],
                     method: String, type: String, region: String,
                     stiffness: Double, scope: String? = nil)
    /// Upper-body gravity feed-forward on/off. Orthogonal to `controlSpec`'s
    /// axes: it does not select a control space, it decides whether the shared
    /// compensation layer feeds anything at all. Default ON — OFF lets the arm
    /// sag on servo stiffness alone (diagnostics, or showing what it buys).
    case gravityComp(Bool)
    /// Which retargeting profile the arms run.
    /// `true`  — the ORIGINAL whole-body retargeting: follows the operator's
    ///           whole arm, elbow direction included.
    /// `false` — (default) the front-workspace tabular profile, which spends the
    ///           arm's redundancy on putting the PALM where the operator points
    ///           it rather than on matching their elbow.
    case wholeBodyRetarget(Bool)
    /// Operator push applied to robot bodies (drag on the 3D model).
    case externalForce([String: [Double]])
    /// Run a named policy / behaviour.
    case dispatchPolicy(id: UUID)
    /// Engage / release the software emergency stop.
    case estop(Bool)
    /// Mute/unmute the ROBOT'S mic — a CONTROL message. The voice PCM
    /// itself rides the dedicated :9610 duplex plane; a 32 kB/s firehose
    /// on this 50 Hz socket is exactly the starvation the 96xx split exists
    /// to avoid.
    case setMicEnabled(Bool)
    /// Return all joints to the home pose.
    case goHome
    /// Reset the simulation to its initial start environment (sim only).
    case resetSim
    /// Sim only: the SERVER relocates the robot to a validated random open pose
    /// (clearance-checked against the scene) and re-homes the joints there.
    case resetSimRandom
    /// Re-anchor the retargeting clutch at the operator's current pose.
    case recalibrate
    /// Start (true) / end (false) recording a teleop episode into a task folder.
    case record(active: Bool, task: String)
    /// Select a task folder (so the server reports its episode count) without recording.
    case selectTask(String)

    // MARK: Animation demos (animation_demo/)
    //
    // Separate from `record`, not a mode of it: `record` drives CurationRecorder
    // and stores the robot's REALIZED state for training; a demo clip stores the
    // operator's COMMANDS so the server can push them back through the same
    // controller. Different payload, different consumer, different folder.
    case demoRecord(active: Bool, name: String, countdown: Double)
    case demoReplay(active: Bool, name: String, repeats: Int)
    case demoDelete(String)
    case demoList
    /// Rename the connected robot — the bridge persists it and broadcasts the
    /// new name to every client (iOS / web / headset), so names stay in sync.
    case setRobotName(String)
    /// Adjust the robot's full-stick speed caps (m/s, rad/s). Server clamps to its
    /// hard ceiling. Used by the Settings speed sliders.
    case setSpeedCaps(vmax: Double, wmax: Double)
    /// Ask the real robot for the single-driver lock (only the holder can drive).
    /// `name` identifies us to observers. Reply rides `onControl`.
    case requestControl(name: String)
    /// Give up the driver lock (the link sends back the grant key it is holding).
    case releaseControl
    /// Start the guided 3-pose body calibration for the named operator. The bridge
    /// runs the capture from the streamed pose frames and reports progress + the
    /// measured profile back in `robot_state`.
    case startBodyCalibration(user: String)
    /// Cancel an in-progress body calibration.
    case cancelBodyCalibration
    /// Re-apply a previously saved profile by operator name (the bridge loads it).
    case applyBodyCalibration(user: String)
    /// Nav2 navigation goal in the odom/map frame (metres + heading rad) — the
    /// tapped point on the live map. The planned route streams back on /map.
    case navGoal(x: Double, y: Double, yaw: Double)
    /// Ask the server for its saved-map catalogue → reply rides `onMapList`.
    case listMaps
    /// Load a saved map by name ("live" / "" returns to live SLAM).
    case loadMap(name: String)
    /// Drop the current SLAM map and start a fresh one at the robot's pose.
    case resetSlam
}

/// A saved map advertised by the server (list_maps → map_list).
struct MapInfo: Identifiable, Equatable {
    var name: String
    var hasMesh: Bool
    var hasReloc: Bool
    var id: String { name }
}

protocol RobotLink: AnyObject {
    var status: ConnectionStatus { get }
    /// Called on the main actor whenever fresh telemetry arrives.
    var onState: ((RobotState) -> Void)? { get set }
    var onStatus: ((ConnectionStatus) -> Void)? { get set }
    /// Called when the link learns what it's driving — the server identity
    /// string from `hello` (e.g. "mabel_sim_bridge"). nil/absent means unknown.
    var onServerInfo: ((String?) -> Void)? { get set }
    /// The server's full identity (name + kind + id), from `hello` and every
    /// `robot_state.server` — drives the server-synced robot name.
    var onServerMeta: ((ServerMeta) -> Void)? { get set }
    /// Driver-lock reply from the real robot: state ("granted"/"denied"/
    /// "released"), the grant key (when granted), and the current holder's name.
    var onControl: ((_ state: String, _ key: String?, _ holder: String) -> Void)? { get set }
    /// Saved-map catalogue reply (`map_list`): the maps + the loaded one
    /// (nil = live SLAM).
    var onMapList: ((_ maps: [MapInfo], _ loaded: String?) -> Void)? { get set }
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)? { get set }

    func connect()
    func disconnect()
    func send(_ command: RobotCommand)
}
