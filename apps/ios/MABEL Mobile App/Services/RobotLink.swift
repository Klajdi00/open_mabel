//
//  RobotLink.swift
//  MABEL Mobile App
//
//  Transport abstraction between the app and MABEL. The app talks to a
//  `RobotLink`; today that's the on-device `DemoLink` simulator, but a
//  `WebSocketLink` (mirroring server's wire protocol) can be
//  dropped in without touching the UI.
//

import Foundation

/// A command the operator issues. Encoded to the wire by a real link.
enum RobotCommand {
    /// Base twist: forward (m/s), strafe (m/s), yaw (rad/s).
    case twist(forward: Double, strafe: Double, yaw: Double)
    /// Nav2 navigation goal in the map/`odom` (mesh) frame: x,y metres, yaw rad.
    /// The server forwards it to /goal_pose (NavFn A* + MPPI); the planned route
    /// returns on the /map stream as a PATH frame for the live map to draw.
    case navGoal(x: Double, y: Double, yaw: Double)
    /// Absolute joint target in the joint's native units.
    case jointTarget(id: String, value: Double)
    /// Lift column height fraction 0...1 (maps both lift stages).
    case liftFraction(Double)
    /// Lift velocity command from a joystick, −1 (down) … +1 (up).
    case liftRate(Double)
    /// Operator head pose for active perception — a 4×4 row-major transform
    /// (16 floats). The robot's neck looks where this points.
    case headPose([Double])
    /// Active-perception head tracking on/off. Reported on the wire (the frame's
    /// head `trackingState`) so the server / command monitor sees the toggle.
    case headTracking(Bool)
    /// Structured control spec — the three orthogonal teleop axes:
    /// teleop method, controller type (impedance/compliance), region
    /// (arm/upper_body/whole_body), plus the impedance stiffness K (Nm/rad).
    case controlSpec(method: String, type: String, region: String, stiffness: Double)
    /// Upper-body gravity feed-forward on/off. Orthogonal to `controlSpec`'s
    /// three axes: it does not select a control space, it decides whether the
    /// shared compensation layer feeds anything at all. Default ON — OFF lets the
    /// arm sag on servo stiffness alone (diagnostics, or showing what it buys).
    case gravityComp(Bool)
    /// Which retargeting profile the arms run.
    /// `true`  — the ORIGINAL whole-body retargeting: the robot follows the
    ///           operator's whole arm, elbow direction included.
    /// `false` — (default) the front-workspace tabular profile, which spends the
    ///           arm's redundancy on putting the PALM where the operator points
    ///           it rather than on matching their elbow.
    case wholeBodyRetarget(Bool)
    /// Operator push applied to robot bodies (drag anywhere on the 3D model):
    /// MuJoCo body name → world-frame force [fx, fy, fz] in newtons.
    case externalForce([String: [Double]])
    /// Run a named policy / behaviour.
    case dispatchPolicy(id: UUID)
    /// Engage / release the software emergency stop.
    case estop(Bool)
    /// Mute / unmute the ROBOT's microphone. Control only — the voice PCM itself
    /// rides the duplex `:9610` audio plane, never this 20 Hz socket.
    case setMicEnabled(Bool)
    /// Return all joints to the home pose.
    case goHome
    /// Reset the simulation to its initial start environment (sim only).
    case resetSim

    /// Sim only: the SERVER relocates the robot to a validated random open pose
    /// (clearance-checked against the scene) and re-homes the joints there.
    case resetSimRandom
    /// Restart SLAM server-side: drop the map, start a FRESH one from the
    /// robot's current pose (localization re-anchors there too).
    case resetSlam
    /// Start (true) / end (false) recording a teleop episode into a task folder.
    case record(active: Bool, task: String)
    /// Select a task folder (so the server reports its episode count) without recording.
    case selectTask(String)

    // MARK: Animation demos (animation_demo/)
    //
    // A SEPARATE pair of commands from `record`, not a mode of it. `record`
    // drives CurationRecorder and writes a policy-training episode of the
    // robot's REALIZED state; a demo clip stores the operator's COMMANDS so the
    // server can push them back through the same controller. Different payload,
    // different consumer, different folder — the app's Record menu picks.
    /// Arm (true) / stop (false) a demo recording. `countdown` seconds of
    /// pre-roll are timed by the SERVER: an app-side countdown drifts by
    /// whatever the link latency was that second.
    case demoRecord(active: Bool, name: String, countdown: Double)
    /// Play (true) / wind down (false) a saved clip, `repeats` times.
    case demoReplay(active: Bool, name: String, repeats: Int)
    /// Delete a saved clip.
    case demoDelete(String)
    /// Ask for the clip library (answered with `demo_clips`).
    case demoList
    /// Adjust the robot's full-stick speed caps (m/s, rad/s). Server clamps to its
    /// hard ceiling. Used by the Settings speed sliders.
    case setSpeedCaps(vmax: Double, wmax: Double)
    /// Rename this robot. The server persists it and syncs the name to all clients.
    case setRobotName(String)
    /// Ask the real robot for the single-driver lock (only the holder can drive).
    /// `name` identifies us to observers. Reply rides `onControl`.
    case requestControl(name: String)
    /// Give up the driver lock (the link sends back the grant key it is holding).
    case releaseControl
    /// Start (true) / stop (false) a full-bus rosbag (MCAP) data-collection
    /// session on the server. `label` names the bag dir. The server records all
    /// joint/command/camera/depth/cloud/mesh/map/odom/tf topics to a local bag
    /// (never pushed) for later long-horizon nav+manip use — see
    /// server/docs/data_collection.md.
    case dataCollect(active: Bool, label: String)
    /// Register this device's APNs push token so the server can notify the
    /// operator (connection lost, fault, task complete…) while the app is asleep.
    case registerPush(token: String)
    /// Ask the server for the list of saved maps (mirrors the web console's
    /// `list_maps`). Reply rides `onMapList`.
    case listMaps
    /// Load a saved map by name so the robot localizes against it; the loaded
    /// map then streams on `/map`. `""` / `"live"` / `"none"` return to live SLAM.
    case loadMap(name: String)
}

/// One saved map advertised by the server's `map_list` (mirrors the web console).
struct MapInfo: Identifiable, Equatable, Hashable {
    let name: String
    var hasMesh: Bool = false
    var hasReloc: Bool = false
    var id: String { name }
}

protocol RobotLink: AnyObject {
    var status: ConnectionStatus { get }
    /// Called on the main actor whenever fresh telemetry arrives.
    var onState: ((RobotState) -> Void)? { get set }
    var onStatus: ((ConnectionStatus) -> Void)? { get set }
    /// Called when the link learns what it's driving — the server identity
    /// string from `hello` (e.g. "mabel_sim_bridge"), or a synthetic name for
    /// the on-device demo. nil/absent means unknown.
    var onServerInfo: ((String?) -> Void)? { get set }
    /// Richer server identity + live speed caps from `hello`/`robot_state` — used
    /// to sync the robot's display name and reflect its caps in Settings.
    var onServerMeta: ((ServerMeta) -> Void)? { get set }
    /// Driver-lock reply from the real robot: state ("granted"/"denied"/
    /// "released"), the grant key (when granted), and the current holder's name.
    var onControl: ((_ state: String, _ key: String?, _ holder: String) -> Void)? { get set }
    /// Saved-map catalogue from the server's `map_list`: the available maps and
    /// the currently loaded one (nil = live SLAM).
    var onMapList: ((_ maps: [MapInfo], _ loaded: String?) -> Void)? { get set }
    /// The demo-clip library, in reply to `.demoList`.
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)? { get set }

    func connect()
    func disconnect()
    func send(_ command: RobotCommand)
}
