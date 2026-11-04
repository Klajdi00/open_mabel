//
//  RobotController.swift
//  MabelTeleop
//
//  The app's single source of truth — a direct port of the iOS app's
//  controller (same intents, same wire protocol) with the headset's input
//  stack swapped in: ARKit head tracking replaces CoreMotion, and ARKit
//  hand skeletons stream to the server's retargeter for direct arm + finger
//  teleoperation (the decoupled wrist/hand mapping lives server-side).
//
//  Performance note: high-rate values (telemetry state, joint echoes, the
//  speed governor) are PLAIN vars — publishing them at 20–30 Hz re-rendered
//  the whole window per tick and made the governor slider lag. A single
//  `uiTick` heartbeat (~8 Hz) drives SwiftUI refresh instead; 3D stages poll
//  at their own 30 Hz. Low-rate flags stay @Published.
//

import SwiftUI
import UIKit
import Combine
import Network
import QuartzCore

/// Live 3D pose, decoupled from SwiftUI — the same pattern as MABEL iOS v2.
/// The network thread writes the newest merged pose here CHEAPLY (no
/// `@Published`, so the 50 Hz stream never invalidates the SwiftUI tree), and
/// the SceneKit viewer reads it from a `CADisplayLink` render loop and applies
/// it straight to the nodes. `tick` bumps on every write so a viewer applies
/// only when the pose actually changed. Main-thread confined — no locking.
final class LivePose {
    private(set) var joints: [String: Double] = [:]
    private(set) var heading: Double = 0
    private(set) var baseQuat: [Double] = [1, 0, 0, 0]
    /// Chassis world position (odom, metres) from robot_state.base.
    private(set) var baseX: Double = 0
    private(set) var baseY: Double = 0
    private(set) var tick: UInt64 = 0
    func set(_ j: [String: Double], heading h: Double, quat q: [Double],
             x: Double = 0, y: Double = 0) {
        joints = j; heading = h; baseQuat = q; baseX = x; baseY = y; tick &+= 1
    }
}

@MainActor
final class RobotController: ObservableObject {

    // ── UI refresh heartbeat (see header note) ────────────────────────────
    /// Bumped ~8 Hz; any view observing the controller refreshes then.
    @Published private(set) var uiTick = 0
    private var tickTimer: Timer?

    /// Decoupled 3D pose for the SceneKit viewer (see `LivePose`). Updated every
    /// telemetry frame WITHOUT touching `@Published`.
    let livePose = LivePose()
    private var lastTelemetry: [String: Double] = [:]   // newest streamed joint subset
    private var lastHeading: Double = 0
    private var lastQuat: [Double] = [1, 0, 0, 0]
    private var lastBaseX: Double = 0
    private var lastBaseY: Double = 0

    /// Latched planar LiDAR sweep (the bridge attaches one to robot_state only
    /// when fresh, ~5 Hz — held here so no sparse frame is ever dropped).
    @Published private(set) var lidarScan: RobotState.LidarScan? = nil
    /// Latched per-module connection roster (change-gated on the wire).
    private(set) var moduleRoster: [RobotState.Module] = []

    /// Latched sparse joint dynamics (~10 Hz sim-bridge V2 extension): the wire
    /// attaches jointVelocities/jointTorques only on fresh ticks, so the last
    /// non-empty sample is held here for the telemetry windows (plain vars —
    /// the windows sample on their own timers, never observe).
    private(set) var jointVelocities: [String: Double] = [:]
    private(set) var jointTorques: [String: Double] = [:]

    // High-rate values: plain vars, surfaced to SwiftUI via uiTick.
    private(set) var state = RobotState()
    /// Local echo of joint targets so sliders feel instant even before
    /// telemetry round-trips back.
    var jointTargets: [String: Double] = RobotModel.homePose
    /// Operator-facing speed scaling (0.1...1) — a safety governor.
    var speedScale: Double = 1.0

    // Low-rate state: published (cheap, infrequent).
    @Published private(set) var connection: ConnectionStatus = .disconnected
    @Published private(set) var target: ConnectionTarget = .simulation
    /// BOOT E-STOPPED (operator directive, 2026-07-18): the app starts with the
    /// e-stop engaged so the robot is presumed LIMP until the operator presses
    /// TORQUE ON — matching the server's safe bring-up (torque intent None, the
    /// robot never auto-energises). This also makes the very first TORQUE ON
    /// press the estop{active:false} edge that energises arms + hands.
    @Published var estopEngaged = true {
        didSet { wireFlags.estop = estopEngaged }
    }
    /// TORQUE ON was sent; waiting for the server's `estop:false` echo. The server
    /// can REFUSE (hardware preflight) — so the button only turns red once it agrees.
    @Published private(set) var torqueOnRequested = false
    private var torqueRequestAt: TimeInterval?
    /// Our own E-STOP, until the server reports `estop:true` (resent meanwhile).
    private var stopRequestedAt: TimeInterval?
    private var stopSentAt: TimeInterval = 0
    /// Why the last torque request failed (the gateway's own words). nil = fine.
    @Published var safetyNotice: String?
    /// "Torque on · limp: Left Hand" — what the robot's last torque-on left OFF (robot_state.torqueDegraded).
    /// Torque-on degrades instead of refusing (2026-10-25): it is a WARNING, never a stop. nil = all came up.
    var torqueDegradedLine: String? {
        guard !estopEngaged, !state.torqueDegraded.isEmpty else { return nil }
        let names = state.torqueDegraded.keys.sorted().map { $0.replacingOccurrences(of: "_", with: " ").capitalized }
        return "Torque on · limp: " + names.joined(separator: ", ")
    }

    @Published private(set) var videoBase: URL? = nil
    /// Robot host for the low-latency UDP video path (nil in demo mode);
    /// VideoFeed subscribes here first and falls back to MJPEG on `videoBase`.
    @Published private(set) var videoHost: String? = nil
    /// UDP video port for the LAN path. Defaults to the H.264 port (9092 — the convention
    /// across sim AND the real robot); the hello's video.h264Port/udpPort refines it.
    private(set) var videoUDPPort: UInt16 = 9092
    /// Codec on the LAN UDP path: H.264 (:9092) preferred; JPEG (:9091) on an old bridge.
    private(set) var videoCodec: VideoCodec = .h264
    /// Head camera intrinsics reported by the server's `hello` (fovy °, aspect).
    @Published private(set) var headCamera: (fovy: Double, aspect: Double)? = nil

    /// Full MJPEG URL for a camera stream. Single source of truth for every
    /// video tile and the head wrap.
    func videoURL(_ stream: String) -> URL? {
        // A dropped/reconnecting link keeps `videoBase` (so a reconnect resumes
        // instantly), but no consumer may STREAM while control is down — an
        // "Offline" pill with live video reads as a phantom connection.
        guard connection == .connected, let base = videoBase else { return nil }
        return base.appendingPathComponent("camera/\(stream)/stream.mjpg")
    }

    /// HERO MJPEG URL for the full-screen POV crossfade (identical to `videoURL`).
    func videoMainURL(_ stream: String) -> URL? { videoURL(stream) }

    /// MJPEG URL for a thumbnail tile (wrist cams, Cam 3D projected planes).
    func videoThumbURL(_ stream: String) -> URL? { videoURL(stream) }

    /// Joints the bridge streams as live telemetry: arms/torso/neck/lift +
    /// (since bridge 2.2) the ORCA wrists and finger joints, so the model
    /// mirrors hand retargeting live instead of only the command echo.
    /// False until this link has told us where the column ACTUALLY is.
    ///
    /// `jointTargets` starts at `RobotModel.homePose`, and homePose puts the two lift
    /// stages at their ref — mid-stroke — because that is the right pose to DRAW an
    /// unknown robot in. It is not a pose to COMMAND. Switching transports reset the
    /// targets and the next thing that streamed them drove the column to mid-stroke,
    /// unasked: a heavy axis moving on its own is the one thing a transport switch must
    /// never do.
    ///
    /// So no lift command leaves this app until telemetry has said where the column is and
    /// `jointTargets` has adopted it. After that the targets already equal the measurement,
    /// so an incidental re-send is a no-op and only a deliberate move — the slider, the
    /// jog, Home — changes anything.
    private var liftAdopted = false

    /// Lift ids, in the one place that has to agree with itself.
    static let liftJoints = ["lift_lower", "lift_upper"]

    static let telemetryJoints: Set<String> = {
        var s = Set<String>(["torso", "neck_yaw", "neck_pitch", "neck_roll",
                             "lift_lower", "lift_upper", "fl_steer", "fr_steer", "b_steer",
                             "left_wrist", "right_wrist"])
        for i in 1...7 { s.insert("left_arm_\(i)"); s.insert("right_arm_\(i)") }
        for side in ["left", "right"] {
            s.insert("\(side)_gripper")                  // OpenArm gripper opening (m), when mounted
            for f in ["thumb", "index", "middle", "ring", "pinky"] {
                for seg in (f == "thumb" ? ["mcp", "abd", "pip", "dip"] : ["abd", "mcp", "pip"]) {
                    s.insert("\(side)_\(f)_\(seg)")
                }
            }
        }
        return s
    }()

    /// Stage pose: ACTUAL robot state for arms/torso/lift/neck merged with
    /// the COMMANDED finger joints (telemetry doesn't stream the fingers).
    var liveModelJoints: [String: Double] {
        var j = jointTargets
        for k in Self.telemetryJoints { if let v = state.joints[k] { j[k] = v } }
        return j
    }

    /// Rebuild the decoupled `livePose` from the newest telemetry merged with the
    /// commanded targets (fingers/grips) — the SAME merge as `liveModelJoints`, but
    /// pushed into the lock-free holder the 3D render loop reads. Called every frame
    /// from `onState` and on any direct joint/grip command so the model is instant.
    func refreshLivePose() {
        var j = jointTargets
        // During a Home glide the COMMANDED pose is authoritative for the body — skip
        // the telemetry overlay so the model shows the arms gliding down even on a sim
        // that doesn't echo joint commands. The base pose (below) still tracks live.
        if !homing {
            for (k, v) in lastTelemetry { j[k] = v }
        }
        livePose.set(j, heading: lastHeading, quat: lastQuat, x: lastBaseX, y: lastBaseY)
        // The demo navigator's speed cap is the tip-over model at the CURRENT lift
        // height (MPPINavigator) — keep it in step with the commanded column.
        liveMap.demoLiftHeight = (j["lift_lower"] ?? 0) + (j["lift_upper"] ?? 0)
    }

    /// The robot's configured max base speeds (used to scale joysticks).
    let maxLinear = 1.2          // m/s
    let maxAngular = 1.8         // rad/s

    // ── Auto-connect (race the active robot's routes) ─────────────────────────
    /// Races a robot's OWN LAN / Tailscale routes — the SAME orchestration as the
    /// iOS app and web console.
    let planner = ConnectionPlanner()
    /// The transport the live link is using — drives the Settings network card.
    @Published private(set) var activeRoute: ConnectionRoute? = nil
    /// The full ordered plan from the last race (fastest first).
    @Published private(set) var routePlan: [ConnectionRoute] = []
    /// True while connectAuto is racing candidates / cycling the plan.
    @Published private(set) var autoConnecting = false

    // ── Is there a network at all? ──────────────────────────────────────────
    /// What the OS says about the path off this device. The transport picker
    /// needs it to answer one question honestly: is switching off Bluetooth even
    /// an option right now? Offering a Wi-Fi row that cannot work is worse than
    /// showing why it cannot.
    enum NetworkPath: Equatable {
        case wifi, cellular, wired, none

        var isOnline: Bool { self != .none }
        var icon: String {
            switch self {
            case .wifi:     return "wifi"
            case .cellular: return "antenna.radiowaves.left.and.right"
            case .wired:    return "cable.connector"
            case .none:     return "wifi.slash"
            }
        }
        var label: String {
            switch self {
            case .wifi:     return "Wi-Fi"
            case .cellular: return "Cellular"
            case .wired:    return "Wired"
            case .none:     return "No network"
            }
        }
    }
    @Published private(set) var networkPath: NetworkPath = .none
    private let pathMonitor = NWPathMonitor()

    private func startNetworkMonitor() {
        pathMonitor.pathUpdateHandler = { [weak self] path in
            let next: NetworkPath
            if path.status != .satisfied { next = .none }
            else if path.usesInterfaceType(.wifi) { next = .wifi }
            else if path.usesInterfaceType(.wiredEthernet) { next = .wired }
            else if path.usesInterfaceType(.cellular) { next = .cellular }
            else { next = .wifi }
            Task { @MainActor in
                guard let self, self.networkPath != next else { return }
                self.networkPath = next
            }
        }
        pathMonitor.start(queue: DispatchQueue(label: "mabel.path"))
    }

    /// The remembered set of robots (sim + real), persisted and synced —
    /// the SAME model as the iOS app, so the switcher + settings read identically.
    let robotStore = RobotStore()

    /// Live SLAM-map stream (nvblox mesh + odom + grid + cloud + path) for the
    /// Robot-space viewer. Rides its own WebSocket, pointed at whichever robot
    /// we're driving; the viewer toggles `active` so it only streams on screen.
    let liveMap = LiveMapStore()
    /// Live server identity from the connected robot (the server-owned name +
    /// kind). nil until a real link reports it. Drives the synced robot name.
    @Published private(set) var serverMeta: ServerMeta? = nil

    private var routeIdx = 0
    private var failoverTimer: Timer?
    private var autoWanted = false              // true between connect(to:) and disconnect
    private var lastDiscovered: [String] = []
    /// The robot we're racing toward — its identity + its OWN routes (so the sim
    /// and the real rig never share a host pool) + the secret access code carried
    /// on every route for a gated (real) robot.
    private var connectTarget: (id: String, hosts: [String], accessCode: String?)?
    private var storeObserver: AnyCancellable?
    private var liveMapObserver: AnyCancellable?
    private var voiceObserver: AnyCancellable?

    private var link: RobotLink

    // MARK: - Live SLAM map endpoint

    /// The live-map WebSocket for the robot we're driving, or nil when there's no
    /// real link (on-device demo) — derived from the active route so it tracks
    /// LAN / Tailscale exactly like the camera + teleop URLs.
    var mapStreamURL: URL? {
        guard let route = activeRoute else { return nil }
        let code = robotStore.active?.kind == .real ? SecureStore.realRobotCode : nil
        return NetworkConfig.directMapURL(host: route.host, accessCode: code)
    }

    /// Re-point the live-map stream after any route change (connect / failover /
    /// disconnect). Cheap and idempotent — the store ignores an unchanged URL.
    private func refreshMapEndpoint() {
        liveMap.setEndpoint(mapStreamURL)
        refreshAudioEndpoint()
    }

    // MARK: - Voice (the :9610 duplex plane)

    /// Talk through the robot's speaker / hear its room. The one DUPLEX plane in
    /// the 96xx family; everything else there is read-only telemetry.
    let voice = AudioStreamLink()

    /// Where the voice plane lives for the current route. Shares the map
    /// endpoint's lifecycle because both are derived from `activeRoute`, and both
    /// can be asked for before it exists.
    var audioStreamURL: URL? {
        guard connection == .connected, let route = activeRoute else { return nil }
        return URL(string: "ws://\(route.host):\(NetworkConfig.audioPort)")
    }

    func refreshAudioEndpoint() { voice.setEndpoint(audioStreamURL) }

    /// Is there a voice path at all? Drives the rail buttons' DIM — a control
    /// that looks live but has nothing behind it is worse than a disabled one.
    var audioAvailable: Bool { audioStreamURL != nil }

    /// TALK: the operator's mic → the robot's speaker. A SWITCH, per the
    /// operator's spec ("make them like a switch — switch on to talk"), not the
    /// hold-to-talk the iOS console uses. A latched mic is a real hazard — it is
    /// a hot mic in a room — so the button stays loud about being on (filled,
    /// animated waveform) and ANY loss of the voice path drops it.
    @Published private(set) var talking = false

    func setTalking(_ on: Bool) {
        guard on != talking else { return }
        guard !on || audioAvailable else { return }
        talking = on
        if on { voice.startTalking() } else { voice.stopTalking() }
    }

    /// LISTEN: the robot's room → the operator. This mutes at the SOURCE (the
    /// robot stops sending) rather than locally, so an unwanted downlink costs
    /// no bandwidth at all — and it is the same `audio_config` control the iOS
    /// app and the web console send, so all three agree on one robot-side state.
    @Published private(set) var micEnabled = true

    func setMicEnabled(_ on: Bool) {
        guard on != micEnabled else { return }
        micEnabled = on
        // BOTH ends. Source-mute (the wire message) is the efficient half — the
        // robot stops sending. LOCAL mute is the half that is guaranteed: it takes
        // effect within one 20 ms frame and does not depend on the robot agreeing.
        // Without it, a robot that ignores `audio_config` leaves the app playing
        // under a button that says "Muted" (reported 2026-07-29).
        voice.muted = !on
        link.send(.setMicEnabled(on))
    }

    /// Voice follows the link: losing the robot must not leave a hot mic open.
    func stopVoice() {
        if talking { talking = false; voice.stopTalking() }
    }

    // MARK: - Headset tracking (ARKit)

    let handService = HandTrackingService()
    let headService = HeadTrackingService()
    private let latestHands = LatestHands()
    private var handTask: Task<Void, Never>?

    /// Lock-protected mirror of the five flags that gate the outbound
    /// teleop_frame — written here (MainActor) on every change, read by the
    /// frame source ON THE WIRE QUEUE. This is what lets the entire
    /// sample→encode→send path run off the main thread: the old frame source
    /// read `self.bodyTracking` etc. directly, which chained the whole
    /// outbound pipeline to main-actor availability.
    private let wireFlags = WireFlags()

    /// Head active perception: the robot's neck follows the operator's head.
    @Published var headEnabled = true {
        didSet { wireFlags.headEnabled = headEnabled }
    }
    /// Body tracking: stream the hand skeletons → server-side arm + finger
    /// retargeting. Toggleable so the operator can "let go" safely.
    @Published var bodyTracking = false {
        didSet {
            wireFlags.bodyTracking = bodyTracking
            applyTeleopSpec()
        }
    }
    /// True while the immersive teleop space is open (hand tracking needs it).
    /// The space opens in MIXED style ("Start Teleop": windows stay visible,
    /// hands track); `fullImmersion` switches it to the 180° wrap.
    @Published var immersiveActive = false {
        didSet { wireFlags.immersiveActive = immersiveActive }
    }
    @Published var fullImmersion = false {
        didSet {
            if fullImmersion { trackingOnlyMode = false }
            // A change while the space is open RE-CREATES it (`.immersionStyle`).
            // Remember that, so the teardown half of the re-creation knows it is
            // a restart and carries a live session across — in BOTH directions.
            if immersiveActive, fullImmersion != oldValue { styleFlipInFlight = true }
        }
    }
    /// Set while an `.immersionStyle` re-creation is in flight (see fullImmersion).
    private var styleFlipInFlight = false
    /// The NEXT space teardown is the app's own doing (Stop / Exit / leaving the
    /// tab): end the session. Any OTHER teardown while mirroring — the Digital
    /// Crown, the system closing the space — must not end teleop (operator,
    /// 2026-10-16: "do not stop teleop till I explicitly press"): the robot
    /// freezes and the window re-opens the tracking space to resume.
    private var immersiveExitIntended = false
    /// Raised when the system closed the space under a live session; the window
    /// observes it, re-opens the (invisible) tracking space and clears it.
    @Published var resumeTrackingRequested = false

    /// The app is about to dismiss the immersive space ON PURPOSE.
    func noteIntendedImmersiveExit() { immersiveExitIntended = true }

    /// Leave FULL immersion WITHOUT ending teleop. Mirroring continues in the
    /// window's 3rd-person view: the space stays open as an invisible tracking
    /// session (`trackingOnlyMode`) — ARKit hand tracking only exists while a
    /// space is open, so closing it is what used to stop the robot. Not
    /// mirroring → nothing to keep, so the caller just dismisses.
    /// Returns true when the space should stay OPEN (mixed, hidden).
    @discardableResult
    func leaveFullImmersionKeepingTeleop() -> Bool {
        guard bodyTracking else { return false }
        deferBodyEngage = false
        trackingOnlyMode = true
        fullImmersion = false          // style flip → mixed; engagement carried
        return true
    }
    /// TRUE when the immersive space is open ONLY as a background ARKit
    /// tracking session (window task-space teleop): every visible entity in
    /// the space stays hidden — no sticks/console popping into the room.
    @Published var trackingOnlyMode = false
    /// The MIRROR IMAGE of `trackingOnlyMode`: the space is open ONLY to float
    /// the two spatial drive sticks in the operator's room — the windowed
    /// 1st-person cockpit (operator, 2026-07-31: "I need the joystick to be
    /// floating around the user, not on the window"). NO ARKit session is
    /// claimed: nothing tracks, nothing streams, and `immersiveActive` stays
    /// FALSE, so every teleop guard that keys on it behaves exactly as it does
    /// with no space open at all. Two entities and the pinch routing, nothing
    /// more.
    /// DORMANT since 2026-10-16: the 3-D sticks were replaced by the 2-D
    /// `DrivePadDock`, so nothing sets this any more. Kept because
    /// `needsImmersiveOpen()` and the space's promotion path still honour it.
    @Published var stickSpaceOnly = false
    /// Set by the Full Immersive button BEFORE opening the dome: the tracking
    /// services start, but `bodyTracking` does NOT auto-engage — the robot must
    /// not mirror the operator until they explicitly press Start Teleop
    /// (safety confirm → engage). One-shot; cleared on the next space open.
    var deferBodyEngage = false
    /// The connected gateway is the Mac RETARGET LAB (hello.caps ∋ "stream_always"):
    /// stream head + both hands continuously while connected — no engage, no torque,
    /// no calibration card — so the lab records human data and runs its retargeters.
    @Published private(set) var streamAlways = false {
        didSet { wireFlags.streamAlways = streamAlways }
    }
    /// Open the invisible tracking session for the lab. Returns true when the CALLER
    /// must `openImmersiveSpace` (a view owns that action). Body tracking stays
    /// DEFERRED: the lab needs the stream, not the engage edge (which would send
    /// control_mode + recalibrate on every hand reacquire).
    func prepareLabTracking() -> Bool {
        guard streamAlways, !immersiveActive else { return false }
        trackingOnlyMode = true
        deferBodyEngage = true
        return needsImmersiveOpen()
    }
    /// Live hand-tracking status for the console indicators.
    @Published private(set) var leftHandTracked = false
    @Published private(set) var rightHandTracked = false
    /// Why tracking isn't running (authorization denied, ARKit error) — shown
    /// in the immersive console so failures are never silent.
    @Published private(set) var trackingNotice: String? = nil

    // ── Per-operator body calibration ─────────────────────────────────────────
    /// On-device store of named calibration profiles (persisted, like RobotStore).
    let calibrationStore = CalibrationStore()
    /// Live progress of an in-flight guided capture (nil when idle).
    @Published private(set) var calibration: CalibrationProgress? = nil {
        didSet { wireFlags.calibrating = calibration?.active ?? false }
    }
    /// The most recent measured anthropometry the server reported (drives the
    /// completion summary). nil until a profile exists this session.
    @Published private(set) var liveMetrics: OperatorMetrics? = nil
    /// The name the operator gave the capture currently running — used to save the
    /// result exactly once on completion. nil when no capture is awaiting a result.
    private var calibratingName: String? = nil
    private var calibrationObserver: AnyCancellable?
    /// When the in-flight capture started, and whether the SERVER has said
    /// anything about it since (a non-nil `calibrationStatus`). Together they
    /// drive `calibrationStall` — the way out of a capture that never ends.
    private(set) var calibrationStartedAt: Date? = nil
    @Published private(set) var calibrationServerHeard = false
    /// The saved profile the operator chose instead of measuring this session
    /// (shown on the countdown). nil after a fresh capture.
    @Published private(set) var appliedSavedProfile: CalibrationProfile? = nil

    /// Live coaching for the calibration card (the maths lives in `CalibrationCoach`,
    /// which is pure so it can be tested without a headset).
    @Published private(set) var calibrationLive = CalibrationLive()
    private var calCoach = CalibrationCoach()
    private var calLiveTimer: Timer?

    /// Saved profiles for the Settings list.
    var calibrationProfiles: [CalibrationProfile] { calibrationStore.profiles }
    /// The name of the profile currently applied on the robot (nil = none chosen).
    var activeProfileName: String? { calibrationStore.activeName }
    /// First-run / no profile chosen yet → prompt the operator to pick or calibrate
    /// when they start teleop.
    var needsProfileChoice: Bool { calibrationStore.activeName == nil }
    /// True when a capture can run: connected (tracking is started on demand by the
    /// calibration flow, so being in the Teleop tab is NOT required).
    var canCalibrate: Bool { connection.isLive }

    /// Wait until ARKit hand tracking is streaming (the immersive space was just
    /// opened), up to `timeout`. Lets the calibration flow start a capture from
    /// anywhere — it opens the space, awaits this, then begins.
    /// TRUE once the ARKit services are LIVE in the open space — hand data can
    /// flow (a calibration capture) even while `bodyTracking` (the MIRRORING
    /// switch) is still deferred. `handTask` starts only after authorization +
    /// provider start succeed, so this can't go true on a denied permission.
    var trackingReady: Bool {
        immersiveActive && HandTrackingService.isSupported
            && trackingNotice == nil && handTask != nil
    }

    func waitForTracking(timeout: TimeInterval = 5) async -> Bool {
        let start = Date()
        while Date().timeIntervalSince(start) < timeout {
            if bodyTracking || trackingReady { return true }
            try? await Task.sleep(for: .milliseconds(120))
        }
        return bodyTracking || trackingReady
    }

    /// The calibration countdown's engage: flip the MIRRORING switch on — the
    /// retarget spec + clutch re-anchor ride bodyTracking's didSet, so the
    /// mapping starts from wherever the operator's hands are at "GO".
    func engageWholeBody() {
        guard immersiveActive, !bodyTracking else { return }
        Haptics.rigid()
        // Head mirroring comes BACK with the session. `stopWholeBodyTeleop()`
        // freezes it on Stop/Home, and without restoring it here the second
        // engage would drive the arms with a neck that never moved again.
        headEnabled = true
        recenterHead()
        bodyTracking = true
    }

    /// Saved poses + calibrated nav moves — robot-side features gated on hello caps.
    let skills = RobotSkills()

    init(link: RobotLink? = nil) {
        self.link = link ?? DemoLink()
        wire(self.link)
        startNetworkMonitor()
        skills.torqueOff = { [weak self] in self?.engageEStop() }    // human demo = Torque OFF, exactly
        skills.bind(live: $connection.map(\.isLive).eraseToAnyPublisher(),
                    bluetooth: $overBluetooth.eraseToAnyPublisher(),
                    estop: $estopEngaged.eraseToAnyPublisher())
        // The robot list is a nested ObservableObject; forward its changes so
        // views observing the controller refresh when robots are found/renamed.
        storeObserver = robotStore.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        // The calibration store is a nested ObservableObject too — forward its
        // changes so the Settings profiles list refreshes when one is saved/removed.
        calibrationObserver = calibrationStore.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        // Same for the live-map stream's stats (connection / point count) so the
        // viewer overlays refresh through the controller's environment object.
        liveMapObserver = liveMap.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        // The MIC LATCH follows the LINK, not the button press. `startTalking()`
        // may resolve asynchronously (an undetermined permission raises a system
        // prompt) or refuse outright (denied, no input device), and it drops the
        // latch itself on link loss. Mirroring `voice.talking` here is what keeps
        // the Talk switch from sitting lit over a mic that never opened — the
        // hot-mic hazard this control is deliberately loud about, inverted.
        voiceObserver = voice.$talking.sink { [weak self] on in
            guard let self, self.talking != on else { return }
            self.talking = on
        }
        // A scene switch replaces the world under the robot, so it stops
        // whole-body teleop for the same reason Home and Reset do (see
        // `resetSim`). The scene UI only ever holds the studio, so the studio
        // calls back rather than the sheets reaching for the controller.
        simStudio.onWillSwitchScene = { [weak self] in
            guard let self, self.bodyTracking || self.immersiveActive else { return }
            self.stopWholeBodyTeleop()
        }
        let t = Timer(timeInterval: 0.12, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.uiTick &+= 1 }
        }
        RunLoop.main.add(t, forMode: .common)
        tickTimer = t
    }

    /// Identity of the CURRENT tracking session. Flipping `.immersionStyle`
    /// RE-CREATES the immersive space, and the NEW space's `.task` (start) can
    /// run BEFORE the OLD space's `.onDisappear` (stop) — start awaits ARKit
    /// authorization + provider startup, so the stale teardown lands mid-start:
    /// it cleared `immersiveActive`, stopped the just-started providers and
    /// dropped the server to navigation, after which the engage tail silently
    /// no-oped (`applyTeleopSpec` guards on `immersiveActive`). The dome opened
    /// with video but arms + hands dead, and no head-follow base drive (the
    /// 2026-07-29 report). Each space instance therefore carries the token its
    /// `.task` claimed; a stop with a stale token belongs to a REPLACED space
    /// and must not touch the live session.
    private var trackingGen = 0

    /// Claim a new tracking session — SYNCHRONOUS, so the session identity is
    /// taken before any teardown can interleave. The immersive view calls this
    /// first thing in its `.task`, keeps the token, and hands it back from
    /// `.onDisappear`.
    func beginTrackingSession() -> Int {
        trackingGen += 1
        immersiveActive = true
        trackingNotice = nil
        return trackingGen
    }

    /// Make sure the immersive space is open for a REAL tracking session —
    /// returns TRUE when the caller still has to open it.
    ///
    /// Every opener used to test `!immersiveActive` directly, which is wrong the
    /// moment the space can already be open as the cockpit's stick host: that
    /// session is deliberately ARKit-less, so `immersiveActive` is false while
    /// the space very much exists, and `openImmersiveSpace` on an open space
    /// fails. PROMOTE it in place instead — dropping `stickSpaceOnly` is what
    /// tells the live space to claim its tracking session (and bring back its
    /// console, dome and wrist cams).
    func needsImmersiveOpen() -> Bool {
        if immersiveActive { return false }
        if stickSpaceOnly { stickSpaceOnly = false; return false }
        return true
    }

    /// Start ARKit tracking for session `gen` (the immersive space just
    /// opened). Body tracking auto-engages: entering teleop IS starting hand
    /// retargeting — the console toggle is for letting go, not for opting in.
    /// Every await re-checks the token: a superseded or torn-down session must
    /// not keep starting providers or flip the mirroring switch.
    func startImmersiveTracking(session gen: Int) async {
        if HandTrackingService.isSupported {
            // Explicit authorization first — a denied prompt must be visible,
            // not a silent dead toggle.
            let allowed = await handService.requestAuthorization()
            guard gen == trackingGen else { return }
            if !allowed {
                trackingNotice = "Hand tracking permission denied — enable it in Settings › Privacy."
            }
        } else {
            trackingNotice = "Hand tracking unavailable on this device (Simulator?) — joysticks only."
        }

        // ARKit providers can fail transiently right as the space opens; retry once.
        for attempt in 0..<2 {
            do {
                if HandTrackingService.isSupported, trackingNotice == nil {
                    try await handService.start()
                    guard gen == trackingGen else { return }
                }
                try await headService.start()
                break
            } catch {
                NSLog("[MABEL] ARKit start failed (attempt \(attempt + 1)): \(error.localizedDescription)")
                if attempt == 1 {
                    trackingNotice = "ARKit failed to start: \(error.localizedDescription)"
                } else {
                    try? await Task.sleep(for: .seconds(1))
                }
            }
            guard gen == trackingGen else { return }
        }
        guard gen == trackingGen else { return }

        handTask?.cancel()
        let box = latestHands
        let service = handService
        handTask = Task.detached {
            // EDGE-DETECT before hopping to the main actor: ARKit delivers ~90 Hz
            // snapshots, and a per-snapshot MainActor Task assigning @Published
            // tracked flags was a ~90 Hz objectWillChange storm invalidating the
            // ENTIRE SwiftUI tree — starving the run-loop send timer in exactly
            // the whole-body mode that needs it ("hands super slow / jittery").
            var lastL = false, lastR = false
            for await snap in service.snapshots() {
                if Task.isCancelled { break }
                box.store(snap)
                let l = snap.left?.isTracked ?? false
                let r = snap.right?.isTracked ?? false
                if l != lastL || r != lastR {
                    lastL = l; lastR = r
                    Task { @MainActor [weak self] in self?.handsTrackedChanged(left: l, right: r) }
                }
            }
        }

        trackingServicesLive(gen, handsAvailable: HandTrackingService.isSupported)
    }

    /// Tracking services for session `gen` are LIVE — decide the mirroring
    /// switch. Hands drive the robot from the moment teleop starts — EXCEPT
    /// when the space was opened by the Full Immersive button: there the
    /// operator must explicitly press Start Teleop before the robot mirrors
    /// them. Split from the ARKit plumbing (and `handsAvailable` injected)
    /// because the simulator has no hand tracking, so no UI-level harness can
    /// reach this branch — ImmersiveLifecycleTests drives the start/stop
    /// interleavings directly.
    func trackingServicesLive(_ gen: Int, handsAvailable: Bool) {
        guard gen == trackingGen else { return }
        if handsAvailable, trackingNotice == nil {
            if deferBodyEngage && !carryEngage {
                deferBodyEngage = false   // services live; engage awaits Start Teleop
            } else {
                deferBodyEngage = false
                headEnabled = true        // see engageWholeBody: restore with the session
                bodyTracking = true       // didSet → retarget spec + recalibrate
            }
            carryEngage = false
            styleFlipInFlight = false
        } else {
            applyTeleopSpec()
        }
    }

    /// Rising edge of "a hand is tracked" while body tracking is on → re-anchor
    /// the server clutch so the mapping starts from where the hands ARE. (The
    /// server also late-anchors; this is belt and braces against races.)
    private func handsTrackedChanged(left: Bool, right: Bool) {
        let wasAny = leftHandTracked || rightHandTracked
        leftHandTracked = left
        rightHandTracked = right
        if bodyTracking && !wasAny && (left || right) {
            link.send(.recalibrate)
        }
    }

    /// TRUE while the immersive space is being re-created for a style change
    /// (mixed <-> full). The space genuinely stops and restarts, so the engaged
    /// state has to be carried across or whole-body teleop silently dies on
    /// entering full immersion.
    private var carryEngage = false

    /// Stop ARKit tracking (immersive space dismissed). `gen` is the token the
    /// dismissed space claimed in its `.task`: when it is stale, a NEWER space
    /// has already claimed the session (the `.immersionStyle` flip re-creation)
    /// and this teardown belongs to the replaced instance — touching the live
    /// session here is exactly the arms-dead-in-full-immersion race.
    func stopImmersiveTracking(session gen: Int) {
        guard gen == trackingGen else { return }
        trackingGen += 1          // invalidate any in-flight start of this session
        // If the space is being RE-CREATED (a style flip either way, or the dome
        // still wanted), this is a RESTART, not an exit — carry a live session.
        // Reading `fullImmersion` alone missed full→mixed: it is already false
        // by the time the old space tears down, so leaving the dome killed
        // teleop (2026-10-16).
        let wasTracking = bodyTracking
        // `styleFlipInFlight` ALONE decides "restart": every genuine re-creation
        // passes through an `.immersionStyle` change (fullImmersion's didSet
        // raises it). `fullImmersion` is NOT evidence — a Digital-Crown close of
        // the dome tears down with it still true, and reading it as "restart"
        // left a stale carried session and no resume.
        carryEngage = wasTracking && styleFlipInFlight
        // A teardown that is neither a restart nor the app's own Stop/Exit was
        // the SYSTEM (Digital Crown…): keep the session's intent and ask the
        // window to re-open the tracking space. The robot holds meanwhile
        // (navigation spec below = hold the pose; no hands stream).
        if wasTracking && !carryEngage && !immersiveExitIntended {
            resumeTrackingRequested = true
            carryEngage = true             // the re-opened space re-engages
        }
        styleFlipInFlight = false
        immersiveExitIntended = false
        immersiveActive = false
        fullImmersion = false
        handTask?.cancel(); handTask = nil
        handService.stop()
        headService.stop()
        latestHands.clear()
        leftHandTracked = false
        rightHandTracked = false
        bodyTracking = false
        setNavigation()
    }

    /// The frame the link streams: real head pose + (optionally) hands.
    /// THREAD-SAFE BY CONSTRUCTION (2026-07-29): the closure is called on the
    /// link's WIRE QUEUE, never the main thread, so it may only touch the lock
    /// boxes (`wireFlags`, `latestHands`) and the nonisolated ARKit reads —
    /// never `self`'s MainActor state. Sampling ARKit + copying two 27-joint
    /// skeletons on main was part of the load that made the finger overlay lag
    /// the moment whole-body engaged.
    private func makeFrameSource() -> () -> HeadsetFrame? {
        let head = headService
        let hands = latestHands            // head freeze only — see the hands block
        let flags = wireFlags
        // The hand sampler's published output. Lock-protected and nonisolated,
        // so the wire queue may read it directly.
        let bcast = handService.broadcaster
        // Prediction lead: ARKit extrapolates anchors ~30 ms ahead, cancelling
        // most of the wire + retarget latency instead of stacking onto it. The
        // hands get the SAME lead, applied by the sampler.
        let lead: TimeInterval = HandTrackingService.predictLead
        return {
            guard let pose = head.currentPose(at: CACurrentMediaTime() + lead) else { return nil }
            var frame = HeadsetFrame(head: pose)
            let f = flags.load()
            if !f.immersiveActive {
                // WINDOWED app: the operator looking around their room must
                // NOT drive the robot's neck. A frozen head with state
                // "normal" was NOT enough — the engine's relative mapping
                // re-asserts the anchor pose every frame and STOMPS
                // joint-space neck commands ("the neck joystick does
                // nothing"). iOS parity: mark the head notAvailable; the
                // wire gates head on trackingState == normal, so the engine
                // leaves the neck to joint_command entirely.
                frame.head = hands.frozenHead ?? pose
                frame.head.trackingState = .notAvailable
            } else if !f.headEnabled && !f.streamAlways {
                // Freeze the head at the last anchor: the server's relative
                // mapping then holds the neck still.
                // NOT for a Retarget Lab (2026-10-24): Stop/Home clear
                // headEnabled and the lab's deferred engage never restores it,
                // so the lab got one stuck head ("normal") for a whole take —
                // whole body (base, lean, lift, arm body frame) all derive
                // from the head, and the base never moved. The lab drives no
                // robot: it always gets the live head.
                frame.head = hands.frozenHead ?? pose
            } else {
                hands.frozenHead = pose
            }
            // Hands stream while body tracking is live UNLESS e-stopped —
            // EXCEPT during a calibration capture: measuring reach moves
            // nothing (the server holds still + zeroes the base while its
            // _cal is active), and the app BOOTS E-STOPPED, so gating
            // hands on torque silently starved the capture ("never says
            // finished") for anyone calibrating before TORQUE ON.
            // An ACTIVE capture streams hands even while `bodyTracking` is
            // still deferred (2026-07-20: the whole-body flow measures
            // FIRST and mirrors only after the countdown) — the server
            // holds the robot still while its _cal is active, so hand data
            // during a capture can never move anything.
            // A RETARGET LAB (hello.caps ∋ "stream_always", the Mac app at
            // web_gui/retarget_lab) records and retargets offline — it drives no
            // robot, so it gets the hands whenever ARKit sees them, e-stop or not.
            // Only a server that ASKS for it gets this; every robot gateway still
            // receives hands only while body tracking is engaged and torque is on.
            if f.calibrating || f.streamAlways || (f.bodyTracking && !f.estop) {
                // PREDICT the hands to the same instant as the head. The head
                // has always been sampled at now + `lead` (visionOS extrapolates
                // anchors, so asking ahead CANCELS transport latency instead of
                // adding to it) while the hands took the last BROADCAST pose —
                // itself up to 11 ms old at the 90 Hz anchor rate. That is a
                // 30 ms lead against a ~5-11 ms lag: the hands ran ~35-40 ms
                // behind the head, on the one channel the operator watches
                // most, and it also SKEWED the two against each other (the neck
                // led the arms through every motion).
                //
                // `predictedPair` was written for this and had never been
                // called. It costs ~54 transform builds per tick — but this runs
                // on the WIRE QUEUE, never main (that cost is only a problem on
                // the main actor, which is where it used to be).
                //
                // ONE ARKit READER (2026-08-01). This used to call
                // `predictedPair` itself, so the provider had two consumers at
                // two cadences — this 60 Hz tick and the event stream feeding
                // the overlay — which is what made the hands burst and the dot
                // overlay blink. The sampler in `HandTrackingService` now polls
                // at 90 Hz and publishes; every consumer reads what it
                // published, and nothing else touches ARKit.
                //
                // The lead is already baked in by the sampler, so what lands
                // here is a pose predicted to now + 30 ms and at most one poll
                // period (11 ms) old — fresher than the old path, which mixed a
                // predicted read with a stream pose of unbounded age.
                //
                // Expiry still happens at READ time as well as in the sampler:
                // this tick can outlive a stalled sampler (ARKit hiccup, session
                // teardown), and a hand must never be driven from a pose whose
                // age nobody checked.
                let (l, r) = bcast.latest()?
                    .fresh(after: CACurrentMediaTime() - HandTrackingService.sideStaleTimeout)
                    ?? (nil, nil)
                frame.leftHand = l
                frame.rightHand = r
            }
            return frame
        }
    }

    /// Toggle head active perception.
    func toggleHeadTracking() {
        Haptics.select()
        headEnabled.toggle()
        if headEnabled { recenterHead() }
    }

    /// Re-anchor the retargeting clutch (head AND hands) at the current pose.
    func recenterHead() {
        Haptics.rigid()
        link.send(.recalibrate)
        link.send(.headPose([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]))   // demo neck home
    }

    // MARK: - Wiring

    private var latEMA = 0.0
    private func wire(_ l: RobotLink) {
        skills.attach(l as? SkillsTransport, bluetooth: l is BLELink)
        // VP tracking overlay: latch the newest operator_points for the 3D
        // viewer's render tick (main-actor; the overlay hides itself on stale).
        (l as? WebSocketLink)?.onOperatorPoints = { p in
            OperatorPointsLatch.store(p)
        }
        (l as? WebSocketLink)?.onSafetyError = { [weak self] message in
            guard let self else { return }
            self.safetyNotice = message
            self.torqueOnRequested = false; self.torqueRequestAt = nil
            self.estopEngaged = true
        }
        l.onState = { [weak self] s in
            guard let self else { return }
            self.ingestSafety(s)
            var s = s
            self.latEMA = self.latEMA == 0 ? s.latencyMs : 0.85 * self.latEMA + 0.15 * s.latencyMs
            s.latencyMs = self.latEMA
            // Latch the sparse (~5 Hz) LiDAR sweep so no sweep is ever lost.
            if let scan = s.lidar { self.lidarScan = scan }
            // Latch the sparse (~10 Hz) dynamics sample the same way.
            if !s.jointVelocities.isEmpty { self.jointVelocities = s.jointVelocities }
            if !s.jointTorques.isEmpty { self.jointTorques = s.jointTorques }
            // Latch the change-gated module roster so the status panel never blinks.
            if !s.modules.isEmpty { self.moduleRoster = s.modules }
            s.modules = self.moduleRoster
            // THE REC BUTTON FOLLOWS THE ROBOT once it has had time to answer the tap: a take the robot refused
            // (a camera missing, the disk full) or one another client stopped must not sit on "REC".
            if s.recordingReported, s.serverRecording != self.recording,
               Date().timeIntervalSince(self.recordToggledAt) > 3 {
                self.recording = s.serverRecording
            }
            // HIGH-RATE PATH: push the newest pose into the lock-free `livePose`
            // the SceneKit render loop reads. HOLD-LAST: update only the joints
            // present in THIS frame so a partial frame never snaps a joint home.
            for k in Self.telemetryJoints { if let v = s.joints[k] { self.lastTelemetry[k] = v } }
            // ADOPT THE MEASURED COLUMN, once per link. Silently: this is learning where
            // the lift is, not commanding it.
            if !self.liftAdopted, s.reportedJoints.contains(where: Self.liftJoints.contains) {
                for k in Self.liftJoints where s.reportedJoints.contains(k) {
                    self.jointTargets[k] = s.joints[k]
                }
                self.liftAdopted = true
            }
            self.lastHeading = s.heading
            self.lastQuat = s.baseQuat
            self.lastBaseX = s.baseX
            self.lastBaseY = s.baseY
            self.refreshLivePose()
            self.state = s          // plain var — uiTick surfaces it to SwiftUI
            self.adoptSwitches(from: s)          // keep the toggles honest
            self.ingestCalibration(s)
        }
        l.onStatus = { [weak self] st in
            guard let self else { return }
            let wasLive = self.connection.isLive
            withAnimation(Theme.Motion.smooth) { self.connection = st }
            // WE ARE NO LONGER TALKING TO A HOST → show nothing rather than
            // something stale. Covers every way a link dies (socket close, error,
            // the heartbeat watchdog) AND leaving `.demo`; the link then retries
            // its endpoint on its own. Fresh data repaints everything on reconnect.
            // Joint/task sliders keep driving the on-screen model offline because
            // `jointTargets` (the operator's command) is untouched — only ROBOT
            // data is cleared. `.demo` is live, so the on-device sim is never wiped.
            if wasLive && !st.isLive {
                self.resetTelemetry()
                if self.torqueOnRequested {        // never claim a torque-on we did not see land
                    self.torqueOnRequested = false; self.torqueRequestAt = nil
                    self.estopEngaged = true
                    self.safetyNotice = "Torque request unconfirmed after connection loss"
                }
            }
            if st == .connected {
                self.sendControlSpec()
                self.link.send(.selectTask(self.recordTask))
                // VP tracking overlay: opt into the operator_points stream (the
                // gateway only actually streams while whole-body retargeting
                // runs, so a standing opt-in costs nothing when idle).
                (self.link as? WebSocketLink)?.setOperatorOverlay(true)
                // A route answered — stop cycling the plan (the link now
                // auto-reconnects on THIS route by itself).
                self.failoverTimer?.invalidate(); self.failoverTimer = nil
                self.autoConnecting = false
                // Single-driver lock: claim it on connect so this operator drives.
                // If another client already holds it the robot replies "denied" and
                // we stay an observer (their input wins until they release it).
                if self.robotStore.active?.kind == .real { self.requestControl() }
            }
        }
        l.onServerInfo = { [weak self] server in
            // A first guess from the bare server string; the full meta that follows (hello.robot +
            // hello.hands + hello.kind) is authoritative and settles both the rig and the badge.
            RobotRig.select(robot: nil, server: server, hands: nil)
            guard let self else { return }
            let t: ConnectionTarget
            if let s = server?.lowercased() {
                t = s.contains("sim") ? .simulation : .realRobot
            } else {
                t = .unknown
            }
            withAnimation(Theme.Motion.smooth) { self.target = t }
            self.syncSimStudio()     // scene library / physics viewer availability
        }
        l.onServerMeta = { [weak self] meta in
            guard let self else { return }
            self.serverMeta = meta
            if let id = self.connectTarget?.id { self.robotStore.noteReached(id: id) }
            self.streamAlways = meta.caps.contains("stream_always")
            // WHICH robot + which hands, from the gateway's own words (hello.robot / .hands):
            // NOVA and MABEL share the wire, and the ORCA hands swap between them.
            RobotRig.select(robot: meta.robot, server: meta.server, hands: meta.hands)
            // SIM vs REAL from `kind`, which every gateway sends. The `server` string only
            // says "sim" for MABEL's sim — NOVA's sim is "nova_teleop_bridge" and was badged REAL.
            let kt: ConnectionTarget? = meta.kind == .sim ? .simulation : meta.kind == .real ? .realRobot : nil
            if let kt, kt != self.target { withAnimation(Theme.Motion.smooth) { self.target = kt } }
            // Mirror the real robot's safety posture for the Settings/status UI.
            self.securityEnforced = meta.authRequired || meta.armRequired
            self.armed = meta.driver.armed
            self.driverHolder = meta.driver.holder
            if let me = meta.you, let holder = meta.driver.holderId {
                self.isDriver = (holder == me)
            } else if !meta.driver.locked {
                self.isDriver = false
            }
            // Adopt the server-owned NAME + kind for the active robot so all
            // clients converge on the same name, and remember it. A rename on
            // any client (iOS / web / headset) lands here and updates the row.
            self.robotStore.adoptServerMeta(id: self.connectTarget?.id, serverID: meta.id,
                                            name: meta.name, kind: meta.kind)
        }
        l.onControl = { [weak self] state, _, holder in
            guard let self else { return }
            switch state {
            case "granted":  self.isDriver = true;  self.driverHolder = holder; self.controlNotice = ""
            case "denied":   self.isDriver = false; self.driverHolder = holder
                             self.controlNotice = "Another operator is driving (\(holder))."
            case "released": self.isDriver = false; self.controlNotice = ""
            default: break
            }
        }
        // The clip library. WITHOUT this the `demo_list` reply was decoded and
        // then dropped — no subscriber — so the Replay picker only ever saw the
        // `robot_state.animation` mirror, which is empty until the server has a
        // reason to refresh it. The picker looked broken on a healthy link.
        l.onDemoClips = { [weak self] clips in
            guard let self else { return }
            self.demoClipList = clips
        }
        l.onMapList = { [weak self] maps, loaded in
            guard let self else { return }
            withAnimation(Theme.Motion.smooth) {
                self.savedMaps = maps
                self.loadedMap = loaded
            }
        }
        if let ble = l as? BLELink {
            ble.frameSource = makeFrameSource()          // hands over Bluetooth too (HAND_POSE)
        }
        if let ws = l as? WebSocketLink {
            ws.frameSource = makeFrameSource()
            ws.onCameraInfo = { [weak self] fovy, aspect in
                self?.headCamera = (fovy, aspect)
            }
            ws.onVideoInfo = { [weak self] udpPort, h264Port in
                guard let self else { return }
                if h264Port > 0 {                       // H.264 on :9092 (preferred)
                    self.videoUDPPort = UInt16(clamping: h264Port); self.videoCodec = .h264
                } else if udpPort > 0 {                 // old bridge → UDP-JPEG on :9091
                    self.videoUDPPort = UInt16(clamping: udpPort); self.videoCodec = .jpeg
                }
            }
        }
    }

    /// The transport actually in use. Server-authoritative when the robot reports
    /// it (the bridge knows from our socket IP whether we came over LAN or
    /// Tailscale); otherwise we fall back to
    /// the app's host-IP guess from the route that won the connect race.
    var connectionMethod: RouteKind? {
        if let t = serverMeta?.transport, let k = RouteKind(serverTransport: t) { return k }
        return activeRoute?.kind
    }

    /// True when a Reset-to-start-environment is meaningful (simulation, live).
    var canReset: Bool { target.isSimulation && connection.isLive }

    /// True only when a REAL server link is live. The on-device demo connects
    /// instantly at launch, so plain `connection == .connected` must never
    /// satisfy auto-connect guards — that blocked the remote (Tailscale)
    /// fallback from ever firing.
    var connectedToRealServer: Bool { connection == .connected && !(link is DemoLink) }

    /// Reset the simulation to its initial start environment.
    func resetSim() {
        guard canReset else { return }
        Haptics.rigid()
        // A RESET RELOCATES THE ROBOT OUT FROM UNDER THE OPERATOR, so teleop
        // must stop with it — the operator's hands stay exactly where they were
        // while the robot is teleported home, and a live retargeter spends the
        // next frame dragging it straight back out of the pose the reset just
        // established. Same reasoning that already made Home stop it
        // (`goRestPose`), and the same remedy: mirroring resumes only on an
        // explicit Start Teleop (operator, 2026-07-31).
        if bodyTracking || immersiveActive { stopWholeBodyTeleop() }
        snapModelToHome()
        link.send(.resetSim)
    }

    /// The Simulation Control Center client (scene library / switch / MuJoCo
    /// viewer) — available only when the SCC backend runs beside a sim on a
    /// direct link. Reconfigured whenever the connection target changes.
    let simStudio = SimStudio()

    func syncSimStudio() {
        simStudio.configure(host: videoBase?.host, isSimulation: target.isSimulation)
    }

    /// TRUE while any window-space pop-up presented OUTSIDE RobotSpaceView
    /// (the Worlds sheet, …) is up: the protruding 3D stage must hide — it
    /// renders IN FRONT of window modals and its invisible tap targets
    /// (goal_floor, link grabs) EAT the pop-up's pinches otherwise.
    @Published var modalPopupVisible = false
    /// The proximity Connect card is up (centred over the window). Its own flag, not
    /// `modalPopupVisible`: two writers on one Bool clobber each other (closing the Worlds
    /// sheet would bring the stage back over a card still on screen).
    @Published var connectCardVisible = false

    // MARK: - Physics view (MuJoCo takeover) — state lives HERE so the Robot
    // space view re-creation (tab overlays / POV rebuilds) can't drop the mode.
    @Published var physicsViewOn = false
    let physicsViewCam = PhysicsCam()
    /// Bumped on every sim reset: the viewers re-frame onto the robot (the
    /// stage recenters, the physics cam refits its orbit).
    @Published private(set) var refitToken = 0

    /// Sim only: relocate-reset to a server-validated random open pose.
    func resetSimRandom() {
        guard canReset else { return }
        Haptics.rigid()
        if bodyTracking || immersiveActive { stopWholeBodyTeleop() }   // see resetSim
        snapModelToHome()
        link.send(.resetSimRandom)
        physicsViewCam.refit()
        refitToken += 1              // viewers re-frame onto the robot
    }

    /// Reset EVERY desired pose locally and snap the 3D model to home AT ONCE —
    /// don't wait for the server's home telemetry to round-trip. Clearing
    /// `lastTelemetry` means `refreshLivePose` renders the commanded home
    /// immediately; fresh (home) telemetry then confirms it.
    private func snapModelToHome() {
        jointTargets = RobotModel.homePose
        lastTelemetry.removeAll(keepingCapacity: true)
        liftAdopted = false        // unknown column again: no lift command until it re-adopts
        lastHeading = 0; lastQuat = [1, 0, 0, 0]
        refreshLivePose()
    }

    // MARK: - Lifecycle

    /// Demo connection (on-device simulator), used at launch.
    func connect() {
        guard connection == .disconnected else { return }
        link.connect()
    }

    // MARK: - Saved maps (mirrors the web console: list_maps / load_map)

    /// Saved maps advertised by the server, and the currently loaded one
    /// (nil = live SLAM). In the on-device demo these are the procedural rooms.
    @Published var savedMaps: [MapInfo] = []
    @Published var loadedMap: String? = nil

    /// Ask the server for its saved-map catalogue (call when the viewer appears).
    func requestMapList() { link.send(.listMaps) }

    /// Load a saved map so the robot localizes against it; it then streams on /map.
    func loadMap(_ name: String) { link.send(.loadMap(name: name)) }

    /// Return to building a fresh map live (drops any loaded saved map).
    func goLiveSLAM() { link.send(.loadMap(name: "live")) }

    /// Restart SLAM: the server drops its map and begins a FRESH one anchored
    /// at the robot's current pose. The app wipes its rendered map immediately.
    func resetSlam() {
        Haptics.rigid()
        link.send(.resetSlam)
        liveMap.restartMapping()
    }

    /// Send a Nav2 navigation goal (the tapped point on the live map, in the
    /// odom/mesh frame — metres + heading rad). The planned route streams back
    /// on the /map socket as a PATH frame.
    func sendNavGoal(x: Double, y: Double, yaw: Double) {
        guard !estopEngaged else { return }
        // Passive Soft: driving the base to a goal is the opposite of letting the
        // operator move the robot by hand.
        guard !passiveCompliance else { return }
        link.send(.navGoal(x: x, y: y, yaw: yaw))
    }

    // MARK: - Robots: switching, naming

    /// Connect to a specific remembered robot: race ITS routes (LAN/Tailscale)
    /// fastest-first.
    /// This is what the status-bar switcher + Settings rows call.
    /// - Parameter explicit: true when a HUMAN chose this rig (status-bar
    ///   picker, Settings → Connect). An explicit pick PINS the rig: if it can't
    ///   be reached we keep retrying it and surface the failure, rather than
    ///   quietly binding to the other robot. `connectAuto` passes false, so
    ///   launch-time auto-connect keeps its real↔sim alternation.
    ///
    ///   This matters far more than it looks. Failover used to cross rigs on ANY
    ///   exhausted plan, including one the operator had just selected by hand —
    ///   so picking "Simulation" while the sim was down silently connected to
    ///   the PHYSICAL ROBOT instead, with the UI showing a healthy green pill.
    ///   Believing you are in sim while commanding hardware is the single most
    ///   dangerous state this app can be in. Caught by `debugSwitchRig` on
    ///   2026-07-19: three consecutive "connect to sim" calls all landed on
    ///   thor-swerve.
    /// The rig the OPERATOR explicitly selected, if any. Non-nil disables the
    /// cross-rig failover in `failoverTick`.
    private var pinnedRigID: String? = nil
    /// The real rig that started the current auto-failover sweep, so the sweep visits every
    /// real robot once before it falls back to the sim.
    private var firstRealThisCycle: String? = nil

    func connect(to robot: Robot, explicit: Bool = true) {
        startAutoBluetooth()
        pinnedRigID = explicit ? robot.id : nil
        if robot.kind == .real, explicit || firstRealThisCycle == nil { firstRealThisCycle = robot.id }
        // The REAL robot is gated by a secret access code (Keychain) carried on
        // every route; the sim is open.
        let code = robot.kind == .real ? SecureStore.realRobotCode : nil
        connectTarget = (robot.id, robot.routes, code)
        robotStore.setActive(id: robot.id)
        resetDriverState()
        // Switching robots (sim ↔ real) must not carry the PREVIOUS robot's data
        // into the new one: `resetDriverState()` clears the joints but not the odom
        // pose, heading, or the map — so the sim's mesh + base position used to
        // linger over the real robot until it happened to overwrite them.
        resetTelemetry()
        startRace()
    }

    /// Rename a robot. Updates locally + (if it's the one we're driving) pushes
    /// the name to its server, which persists and syncs it to EVERY client.
    func renameRobot(id: String, to name: String) {
        if robotStore.rename(id: id, to: name), connection == .connected {
            link.send(.setRobotName(name))
        }
    }

    // MARK: - Driver lock + arm + access code (real robot safety)

    /// Whether WE hold the single-driver lock on the connected robot.
    @Published private(set) var isDriver = false
    /// Name of whoever holds the lock (may be another operator), "" if free.
    @Published private(set) var driverHolder = ""
    /// The real robot's live arm posture, mirrored from `serverMeta`.
    @Published private(set) var armed = true
    /// True once a real robot has told us it requires an access code / local arm.
    @Published private(set) var securityEnforced = false
    /// Last driver-lock reply we should surface ("denied" when another holds it).
    @Published private(set) var controlNotice = ""

    /// The real robot needs a code but this device hasn't stored one yet → the UI
    /// prompts the operator to enter it (Settings).
    var needsAccessCode: Bool {
        (robotStore.active?.kind == .real) && !SecureStore.hasRealRobotCode
    }
    var hasAccessCode: Bool { SecureStore.hasRealRobotCode }

    /// Store / replace the secret real-robot access code (Keychain) and, if we're
    /// connected to the real robot, reconnect so it takes effect.
    func setAccessCode(_ code: String) {
        let c = code.trimmingCharacters(in: .whitespacesAndNewlines)
        SecureStore.realRobotCode = c
        if let r = robotStore.active, r.kind == .real { connect(to: r) }
        objectWillChange.send()
    }

    /// Ask the real robot for the driver lock (only the holder can drive).
    func requestControl() {
        let who = UIDevice.current.name
        link.send(.requestControl(name: who))
    }
    /// Give up the driver lock so another operator can take it.
    func releaseControl() {
        link.send(.releaseControl)
        isDriver = false
    }

    private func resetDriverState() {
        isDriver = false; driverHolder = ""; armed = true
        securityEnforced = false; controlNotice = ""
        lidarScan = nil                 // stale sweep must not survive a reconnect
        moduleRoster = []               // don't claim modules on a dead link
        jointVelocities = [:]; jointTorques = [:]   // ditto the dynamics latch
    }

    /// NO HOST → NO DATA. Drop everything the robot ever told us, so the headset
    /// falls back to its default (home) pose instead of freezing on the last frame.
    /// Mirror of the iOS twin (see MABEL_iOS_v2 OfflineStateTests): the old
    /// `onStatus` cleared ONLY `lastTelemetry`, so the odom pose, the LiDAR sweep
    /// and the SLAM/pointcloud all lingered after the link died — and only an
    /// explicit disconnect went through even that. A real drop (socket close /
    /// heartbeat watchdog) retries the same endpoint forever and cleared nothing.
    ///
    /// Deliberately does NOT touch routing (`activeRoute`/`videoBase`/`serverMeta`):
    /// the link reconnects on its own endpoint, so clearing those would strand
    /// video + the map socket after a transient blip. Video needs no clearing — its
    /// URL builders are gated on `connection == .connected`. `jointTargets` is the
    /// OPERATOR's command (not robot data), so it is left alone → joint/task sliders
    /// keep driving the on-screen model offline.
    private func resetTelemetry() {
        state = RobotState()            // battery/base/joints/latency → default
        lidarScan = nil
        jointVelocities = [:]; jointTorques = [:]
        moduleRoster = []
        lastTelemetry = [:]
        liftAdopted = false        // unknown column again: no lift command until it re-adopts
        lastHeading = 0; lastQuat = [1, 0, 0, 0]
        lastBaseX = 0; lastBaseY = 0    // refreshLivePose reads these — rig un-parks
        refreshLivePose()               // model falls back to the commanded/default pose
        liveMap.goOffline()             // mesh, occupancy grid, route, pointcloud
        stopVoice()                     // never leave a hot mic open on a dead link
    }

    // MARK: - Speed caps (server-synced, live per robot)

    /// The connected robot's live speed caps + hard ceilings (for the sliders).
    var speedCaps: (vmax: Double, wmax: Double, vmaxCeil: Double, wmaxCeil: Double)? {
        guard let m = serverMeta, let v = m.vmax, let w = m.wmax else { return nil }
        return (v, w, m.vmaxCeil ?? 1.0, m.wmaxCeil ?? 2.0)
    }

    /// Push new full-stick speed caps to the robot (Settings sliders). The server
    /// clamps to its ceiling and echoes them back in `serverMeta`.
    func setSpeedCaps(vmax: Double, wmax: Double) {
        link.send(.setSpeedCaps(vmax: vmax, wmax: wmax))
    }

    /// Connect to a specific host typed/added manually (LAN or Tailscale IP).
    /// Empty / "demo" → the on-device sim. Sanitized so a stale `%en2`-scoped or
    /// link-local address falls back to demo instead of looping forever.
    func connect(host: String) {
        cancelAuto()
        overBluetooth = false
        link.disconnect()
        serverMeta = nil
        let raw = host.trimmingCharacters(in: .whitespaces)
        let trimmed = raw.lowercased() == "demo" ? "" : (ServerDiscovery.sanitized(raw) ?? "")
        if trimmed.isEmpty {
            videoBase = nil; videoHost = nil
            headCamera = nil; activeRoute = nil
            robotStore.setActive(id: nil)
            target = .simulation
            let l = DemoLink(); link = l; wire(l); l.connect()
        } else {
            let l = WebSocketLink(host: trimmed)
            videoBase = l.videoBase
            videoHost = trimmed            // direct IP → UDP video available
            activeRoute = ConnectionRoute(kind: NetworkConfig.classify(trimmed), host: trimmed)
            robotStore.setActive(id: trimmed)
            target = .unknown
            link = l; wire(l); l.connect()
        }
        refreshMapEndpoint()
    }

    // MARK: - Bluetooth (teleop with no network at all)

    /// True while the live link is Bluetooth. The UI uses it to hide what BLE genuinely
    /// cannot carry — camera feeds, maps, recordings — instead of showing controls that
    /// would silently do nothing.
    @Published private(set) var overBluetooth = false
    /// Automatic Bluetooth (the extension at the end of this file).
    private var bleAutoTimer: Timer?
    private var bleAutoOff = false
    private var bleNotLiveSince: TimeInterval?
    private var bleDownSince: TimeInterval?

    /// Connect to a robot found by BLE proximity (the connect panel's one button).
    ///
    /// `done` reports whether the robot's GATT service actually came up, so the panel can
    /// show success or offer a retry — a BLE connect that stalls gives no error of its own,
    /// it simply never completes, so the timeout below is the only thing that can say no.
    /// Connect to a robot REMEMBERED from a previous session: no scan, no advertisement,
    /// nothing on the air. CoreBluetooth resolves the saved identifier directly, which is
    /// what lets Settings offer a saved robot a Connect button that works the moment the
    /// app opens — the list used to be empty until a scan happened to catch a broadcast.
    func connectBluetooth(saved: KnownRobot, done: @escaping (Bool) -> Void) {
        connectBluetooth(id: saved.id, name: saved.advertisedName, done: done)
    }

    func connectBluetooth(_ robot: NearbyRobot, done: @escaping (Bool) -> Void) {
        connectBluetooth(id: robot.id, name: robot.name, done: done)
    }

    private func connectBluetooth(id: UUID, name: String, done: @escaping (Bool) -> Void) {
        cancelAuto()
        link.disconnect()
        serverMeta = nil
        videoBase = nil; videoHost = nil   // no video on this link
        headCamera = nil
        activeRoute = ConnectionRoute(kind: .bluetooth, host: name)
        target = .realRobot

        let l = BLELink(id: id, name: name, central: BLECentral.shared.centralManager)
        l.robotHint = BLECentral.shared.nearby.first { $0.id == id }.map { $0.beacon.isNova ? "NOVA" : "MABEL" }
        link = l
        wire(l)
        overBluetooth = true

        // wire() installs its own onStatus; chain rather than replace it, or the rest of
        // the app stops hearing about this link's connection state.
        let wired = l.onStatus
        var answered = false
        l.onStatus = { status in
            wired?(status)
            guard !answered, status == .connected else { return }
            answered = true
            // Remembered only on a REAL connection, so a failed attempt never leaves a
            // robot in Settings that was never reachable.
            KnownRobots.shared.remember(id: id, advertisedName: name)
            done(true)
        }
        l.connect()

        DispatchQueue.main.asyncAfter(deadline: .now() + 12) {
            guard !answered else { return }
            answered = true
            done(false)
        }
        refreshMapEndpoint()
    }

    func disconnect() {
        bleAutoOff = true             // an explicit Disconnect: no automatic Bluetooth until the next connect
        cancelAuto()
        connectTarget = nil
        link.disconnect()
        videoBase = nil; videoHost = nil
        headCamera = nil; activeRoute = nil; serverMeta = nil
        resetDriverState()
        resetTelemetry()            // pose / LiDAR / pointcloud / map → default
        robotStore.setActive(id: nil)
        pinnedRigID = nil           // an explicit disconnect releases the pin
        refreshMapEndpoint()
    }

    // MARK: - Auto-connect: race the active robot's routes

    /// Connect automatically at launch — to the last-used robot if we have one,
    /// else the simulation (LAN + Tailscale).
    func connectAuto(discovered: [RobotStore.DiscoveredRobot] = []) {
        startAutoBluetooth()
        if !discovered.isEmpty {
            robotStore.mergeDiscovered(named: discovered); lastDiscovered = discovered.map(\.host)
        }
        // A robot answering on THIS Wi-Fi right now beats any remembered one.
        // Otherwise (iOS parity) the REAL robot is tried FIRST; the sim is the
        // fallback tier (failoverTick alternates real → sim → real … until
        // one binds). A stale "active = sim" from a previous session must not
        // stop the default from reaching the physical robot.
        let robot = robotStore.discoveredOnline(discovered).first
            ?? robotStore.lastReached
            ?? robotStore.robots.first { $0.kind == .real }
            ?? robotStore.active
            ?? robotStore.robots.first { $0.kind == .sim }
            ?? robotStore.robots.first
        // explicit:false — launch auto-connect keeps the real↔sim alternation.
        if let robot {
            connect(to: robot, explicit: false)
        } else if !connection.isLive {
            // No robot on the network at all — Bluetooth before the on-device demo, since a
            // real robot within arm's reach beats a simulated one every time.
            considerBluetoothFallback(hadNetwork: false)
            if !overBluetooth { fallbackToDemo() }
        }
    }

    /// Fresh Bonjour hits — fold them into the right robot, and if we're mid-race
    /// for a robot those hosts belong to (and not yet connected on the fastest
    /// tier), re-race so a just-appeared LAN host can win over Tailscale.
    func noteDiscovered(_ found: [RobotStore.DiscoveredRobot]) {
        let hosts = found.map(\.host)
        upgradeFromBluetoothIfPossible(hosts)
        lastDiscovered = hosts
        robotStore.mergeDiscovered(named: found)
        guard autoWanted, let t = connectTarget else { return }
        if connection == .connected, activeRoute?.kind == .lan { return }
        // The rig we are racing is NOT answering, and a robot on this Wi-Fi just
        // proved it is reachable (ServerDiscovery only lists hosts that took a TCP
        // connect): go to it. Without this, launch auto-connect raced thor
        // → sim → thor … forever while NOVA sat discovered and never tried. Never
        // when the operator picked the rig (pinned), never away from a live link.
        if pinnedRigID == nil, connection != .connected,
           let here = robotStore.discoveredOnline(found).first, here.id != t.id {
            connect(to: here, explicit: false)
            return
        }
        // Same robot, new routes: re-race it — keeping its pin state (this used to
        // pass explicit:true and silently PIN an auto-connected rig).
        if let robot = robotStore.robot(id: t.id), robot.routes != t.hosts {
            connect(to: robot, explicit: pinnedRigID == robot.id)
        }
    }

    /// Race the current target robot's routes; bind the fastest.
    private func startRace() {
        autoWanted = true
        guard let t = connectTarget else { return }
        autoConnecting = true
        serverMeta = nil
        failoverTimer?.invalidate(); failoverTimer = nil
        Task { @MainActor in
            let routes = await planner.plan(hosts: t.hosts)
            guard autoWanted, connectTarget?.id == t.id else { return }   // switched mid-race
            routePlan = routes
            routeIdx = 0
            guard !routes.isEmpty else {                // nothing reachable
                autoConnecting = false
                if !connection.isLive { fallbackToDemo() }
                return
            }
            applyRoute(routes[0])
            // Cycle the plan until one answers; the link self-heals once connected.
            failoverTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.failoverTick() }
            }
        }
    }

    /// Bring up the link for a chosen route (direct WebSocket).
    private func applyRoute(_ route: ConnectionRoute) {
        activeRoute = route
        overBluetooth = false
        serverMeta = nil
        link.disconnect()
        target = .unknown
        headCamera = nil
        let l = WebSocketLink(host: route.host, accessCode: connectTarget?.accessCode)
        videoHost = route.host                    // direct IP → low-latency UDP video
        videoBase = l.videoBase
        link = l; wire(l); l.connect()
        refreshMapEndpoint()
    }

    /// Still not connected after a cycle → advance to the next route; when the
    /// plan is exhausted, re-race the SAME robot (the network may have changed).
    private func failoverTick() {
        guard autoWanted else { return }
        if connection == .connected { failoverTimer?.invalidate(); failoverTimer = nil; return }
        guard !routePlan.isEmpty else { return }
        routeIdx += 1
        if routeIdx >= routePlan.count {
            failoverTimer?.invalidate(); failoverTimer = nil
            // iOS parity: a REAL robot whose whole plan failed falls back to
            // the SIM robot; an exhausted sim goes back to trying the real
            // one (the network may have changed) — alternating until bound.
            let current = connectTarget.flatMap { t in robotStore.robot(id: t.id) }
            // An OPERATOR-CHOSEN rig is never swapped out from under them: keep
            // re-racing the rig they picked so the status pill tells the truth
            // about the robot they think they are driving.
            // Real rigs are all tried before the sim: NOVA and thor both seed as .real, and the
            // alternation used to go thor → sim → thor and never reach NOVA.
            if pinnedRigID != nil {
                startRace()
            } else if current?.kind == .real, let other = robotStore.nextReal(after: current?.id),
                      other.id != firstRealThisCycle {
                connect(to: other, explicit: false)
            } else if current?.kind == .real,
               let sim = robotStore.robots.first(where: { $0.kind == .sim }) {
                firstRealThisCycle = nil
                connect(to: sim, explicit: false)
            } else if current?.kind == .sim,
                      let real = robotStore.lastReached.flatMap({ $0.kind == .real ? $0 : nil })
                                 ?? robotStore.robots.first(where: { $0.kind == .real }) {
                connect(to: real, explicit: false)
            } else {
                startRace()
            }
            return
        }
        applyRoute(routePlan[routeIdx])
    }

    /// Stop auto-connecting (manual connect / disconnect).
    private func cancelAuto() {
        autoWanted = false
        autoConnecting = false
        failoverTimer?.invalidate(); failoverTimer = nil
    }

    /// No real route was reachable — keep the app usable on the on-device sim.
    // MARK: - Transport handover (the policy lives in LinkPriority)

    /// Set when a NETWORK link drops and a robot is in Bluetooth range. The UI asks before
    /// taking it: moving DOWN a tier silently removes video and most of the protocol, and
    /// the headset is usually still nominally on Wi-Fi when it is the ROBOT that has gone.
    /// Moving UP needs no such ceremony, which is why there is no matching flag for Wi-Fi.
    @Published var bluetoothFallback: NearbyRobot?

    /// Nothing on the network. Bluetooth is the last rung: take it outright on a FIRST
    /// connection (there is nothing to lose), ask first if we are falling back from a
    /// network link that just died.
    func considerBluetoothFallback(hadNetwork: Bool) {
        // Automatic (2026-10-25): any robot heard is taken outright — the operator asked for no prompts.
        guard !connection.isLive, let ble = bluetoothRobotInRange ?? BLECentral.shared.promoted else { return }
        bluetoothFallback = nil
        connectBluetooth(ble) { _ in }
    }


    func acceptBluetoothFallback() {
        guard let ble = bluetoothFallback else { return }
        bluetoothFallback = nil
        connectBluetooth(ble) { _ in }
    }

    func declineBluetoothFallback() { bluetoothFallback = nil }

    /// A network route appeared while we are on Bluetooth — move UP to it. Automatic by
    /// design: it strictly gains capability, and it is what "prioritise the local Wi-Fi"
    /// means when the robot has been on Bluetooth since before the network came back.
    private func upgradeFromBluetoothIfPossible(_ hosts: [String]) {
        guard overBluetooth, connection.isLive, !hosts.isEmpty else { return }
        guard LinkPriority.isUpgrade(from: .bluetooth, to: .lan) else { return }
        robotStore.mergeDiscovered(hosts)
        let robot = robotStore.robots.first { $0.kind == .real } ?? robotStore.active
        if let robot { connect(to: robot, explicit: false) }
    }

    private func fallbackToDemo() {
        videoBase = nil; videoHost = nil
        overBluetooth = false
        headCamera = nil; activeRoute = nil; serverMeta = nil
        target = .simulation
        let l = DemoLink(); link = l; wire(l); l.connect()
        refreshMapEndpoint()
    }

    // MARK: - Episode recording (data collection)

    @Published private(set) var recording = false
    @Published var recordTask = "default"

    var episodeCount: Int { state.episodeCount }
    var serverTasks: [String] { state.taskList }
    var taskCounts: [String: Int] { state.taskCounts }
    var episodesTotal: Int { state.episodesTotal }
    var episodesDir: String { state.episodesDir }

    func selectTask(_ task: String) {
        let t = task.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        recordTask = t
        link.send(.selectTask(t))
    }

    func refreshCollectionStatus() {
        guard connection.isLive else { return }
        link.send(.selectTask(recordTask))
    }

    func startEpisode() { recording = true; recordToggledAt = Date(); link.send(.record(active: true, task: recordTask)) }
    func endEpisode()   { recording = false; recordToggledAt = Date(); link.send(.record(active: false, task: recordTask)) }
    private var recordToggledAt = Date.distantPast

    // MARK: - Animation demos (animation_demo/)
    //
    // Read back from `state.demo`, never latched locally: a local flag disagrees
    // with the robot the moment a countdown expires, a clip is refused, or a
    // second operator stops the take.
    var demo: RobotState.DemoEcho { state.demo ?? .init() }
    var demoClips: [RobotState.DemoClipInfo] {
        demoClipList.isEmpty ? demo.clips : demoClipList
    }
    var demoBusy: Bool { demo.busy }

    func startDemoRecording(_ name: String, countdown: Double = 3.0) {
        link.send(.demoRecord(active: true, name: name, countdown: countdown))
    }
    func stopDemoRecording() {
        link.send(.demoRecord(active: false, name: "", countdown: 0))
        refreshDemoClips()
    }
    func playDemo(_ name: String, repeats: Int) {
        link.send(.demoReplay(active: true, name: name, repeats: max(1, repeats)))
    }
    /// Winds the player down through its return-home ramp — not a freeze.
    func stopDemo() { link.send(.demoReplay(active: false, name: "", repeats: 1)) }
    func deleteDemo(_ name: String) { link.send(.demoDelete(name)); refreshDemoClips() }
    /// Clips from the `demo_clips` REPLY. Kept separate from the
    /// `robot_state.animation` mirror because the two refresh on different
    /// events: the reply lands the moment the picker asks, the mirror only when
    /// the server has a reason to restate it. Whichever is fresher wins.
    @Published private(set) var demoClipList: [RobotState.DemoClipInfo] = []

    func refreshDemoClips() { link.send(.demoList) }

    // MARK: - Base driving

    /// Drive from a normalized joystick vector (dx right+, dy down+) and yaw.
    /// SOFT (compliance) is a PASSIVE mode: the robot floats under gravity
    /// compensation and the OPERATOR moves it with their hands. The app must not
    /// command it — a joint/base/grip target here is the app fighting the human
    /// holding the arm.
    ///
    /// Gated HERE, at the one choke point every view already funnels through,
    /// rather than by hiding controls per-view: the Soft toggle is independent of
    /// `space`, so Joint space kept its sliders (and Drive its sticks) live in
    /// Soft and they still streamed joint_command. One missed call site is a robot
    /// that resists the person pulling it. The server assumes this too — it holds
    /// the compliance spec sticky precisely because a SOFT session "sends no app
    /// input" (sim_teleop_bridge.py).
    ///
    /// NOT gated: `setControl` (that is how compliance is DECLARED), and
    /// `applyExternalForce` — in the sim that IS the operator's hand, the stand-in
    /// for physically grabbing the real robot.
    var passiveCompliance: Bool { controllerType == .compliance }

    func drive(translate: CGVector, yaw: Double) {
        guard !estopEngaged, !passiveCompliance else { return }
        let f = -translate.dy * maxLinear * speedScale
        let s = translate.dx * maxLinear * speedScale
        let w = yaw * maxAngular * speedScale
        link.send(.twist(forward: f, strafe: s, yaw: w))
    }

    /// Not gated on `passiveCompliance`: zeroing the base is how we make the robot
    /// SAFE, and entering Soft calls it. A stop must always get through.
    func stopBase() {
        link.send(.twist(forward: 0, strafe: 0, yaw: 0))
    }

    /// Physical body twist (m/s, m/s, rad/s) for the map's path follower.
    func driveTwist(forward: Double, strafe: Double, yaw: Double) {
        guard !estopEngaged, !passiveCompliance else { return }
        link.send(.twist(forward: forward * speedScale,
                          strafe: strafe * speedScale,
                          yaw: yaw * speedScale))
    }

    /// A twist the robot must reproduce EXACTLY (a choreography block, not a thumb) — the iPhone app's
    /// driveTwistExact, same reason: m/s as a stick against 1.2 m/s, then the gateway's 0.06 deadzone,
    /// left Open Fridge's 0.11 m/s back-off at ~0.02 m/s (2026-10-26). Solved against the robot's caps
    /// with the deadzone pre-compensated (NOVA server/nav_axes.undeadzone); no operator speed scale.
    func driveTwistExact(forward: Double, strafe: Double, yaw: Double) {
        guard !estopEngaged, !passiveCompliance else { return }
        guard connection == .connected else { return driveTwist(forward: forward, strafe: strafe, yaw: yaw) }
        let caps = speedCaps
        let vmax = max(caps?.vmax ?? maxLinear, 1e-3), wmax = max(caps?.wmax ?? maxAngular, 1e-3)
        link.send(.twist(forward: Self.exactStick(forward, cap: vmax) * maxLinear,
                          strafe: Self.exactStick(strafe, cap: vmax) * maxLinear,
                          yaw: Self.exactStick(yaw, cap: wmax) * maxAngular))
    }

    /// The raw stick that the gateway turns back into exactly `v` (nav_axes.undeadzone, then ÷ cap).
    nonisolated static func exactStick(_ v: Double, cap: Double) -> Double {
        let u = min(1, abs(v) / cap)
        return u < 1e-4 ? 0 : (u * (1 - navDeadzone) + navDeadzone) * (v < 0 ? -1 : 1)
    }

    /// The gateway's stick deadzone (NOVA server/nav_axes.DEADZONE; MABEL carries the same 0.06).
    nonisolated static let navDeadzone = 0.06

    // MARK: - Joint control

    /// A lift command may only leave this app once we know where the column IS — see
    /// `liftAdopted`. Before that the target is still the drawn default (mid-stroke) and
    /// sending it would move a heavy axis nobody asked to move. Dropped, not queued: the
    /// operator's next deliberate touch sends the real value.
    private func liftAllowed(_ id: String) -> Bool {
        guard Self.liftJoints.contains(id) else { return true }
        // …EXCEPT Home, which is the operator deliberately asking for a known pose. The
        // gate is there to stop an INCIDENTAL stream turning the drawn default into a
        // commanded height; Home is the opposite of incidental, and mid-stroke is a
        // defined, bounded place to send the column whether or not we have heard from it
        // yet. Blocking it made the Home button silently do nothing to the lift — which
        // is exactly what the comment above already said it should not do ("only a
        // deliberate move — the slider, the jog, Home — changes anything").
        return liftAdopted || homing
    }

    func setJoint(_ id: String, _ value: Double) {
        guard !passiveCompliance else { return }
        var jt = jointTargets
        jt[id] = value
        Self.echoGripper(&jt, id, value)
        jointTargets = jt
        refreshLivePose()                                  // instant model echo
        guard liftAllowed(id) else { return }
        link.send(.jointTarget(id: id, value: value))
    }

    /// BATCHED joint command for HIGH-RATE multi-joint sources — grab-IK
    /// (≈60 Hz × the whole arm chain) and the arm joysticks (30 Hz × 6 joints).
    /// ONE `jointTargets` mutation + ONE `refreshLivePose()` for the whole batch
    /// (per-joint setJoint() in those loops fired an objectWillChange storm that
    /// saturated the main thread — the iOS v2 latency lesson).
    func setJoints(_ updates: [String: Double]) {
        guard !updates.isEmpty, !passiveCompliance else { return }
        var jt = jointTargets
        for (id, v) in updates { jt[id] = v; Self.echoGripper(&jt, id, v) }
        jointTargets = jt                                  // ONE invalidation for the batch
        refreshLivePose()                                  // ONE rebuild
        for (id, v) in updates where liftAllowed(id) { link.send(.jointTarget(id: id, value: v)) }
    }

    /// Open/close a whole hand (0 = open, 1 = closed fist). Batched into ONE
    /// mutation — a per-joint loop fired objectWillChange 11× per slider tick.
    static let flexSuffixes = ["thumb_pip", "thumb_dip", "index_mcp", "index_pip",
                               "middle_mcp", "middle_pip", "ring_mcp", "ring_pip",
                               "pinky_mcp", "pinky_pip"]
    func setGrip(_ side: String, _ g: Double) {
        guard !passiveCompliance else { return }
        var jt = jointTargets
        for suf in Self.flexSuffixes {
            let id = "\(side)_\(suf)"
            guard let spec = RobotModel.spec(id) else { continue }
            // Limit-proof flexion: interpolate HOME → the joint's true "closed"
            // extreme (the larger-magnitude bound — some rigs flex negative),
            // clamped inside [lower, upper] so no finger ever exceeds its range.
            let home = RobotModel.homePose[id] ?? 0
            let closed = abs(spec.upper) >= abs(spec.lower) ? spec.upper : spec.lower
            let v = home + g * (closed - home)
            jt[id] = min(max(v, min(spec.lower, spec.upper)), max(spec.lower, spec.upper))
        }
        jt["\(side)_grip"] = g
        Self.echoGripper(&jt, "\(side)_grip", g)          // a gripper side: the opening the gateway will drive
        jointTargets = jt                                  // ONE invalidation
        refreshLivePose()                                  // instant hand echo
        link.send(.jointTarget(id: "\(side)_grip", value: g))
    }

    /// On a side wearing the OpenArm gripper the 0…1 grip and the opening (m) are one control: the
    /// gateway drives opening = 0.044·(1 − grip). Keep the two echoes consistent so the model and both
    /// sliders agree before telemetry returns. No-op on an ORCA side (it has no `<side>_gripper`).
    nonisolated static func echoGripper(_ jt: inout [String: Double], _ id: String, _ value: Double,
                            isGripper: (String) -> Bool = { RobotRig.handGen($0) == "gripper" }) {
        if id.hasSuffix("_gripper") {
            let side = String(id.dropLast("_gripper".count))
            jt["\(side)_grip"] = 1 - min(1, max(0, value / RobotModel.gripperMax))
        } else if id.hasSuffix("_grip") {
            let side = String(id.dropLast("_grip".count))
            if isGripper(side) { jt["\(side)_gripper"] = RobotModel.gripperOpening(grip: value) }
        }
    }

    /// The three orthogonal control axes (mirrors the bridge's wire protocol).
    @Published private(set) var teleopMethod: TeleopMethod = .navigation
    @Published private(set) var controllerType: ControllerType = .impedance
    @Published private(set) var region: ControlRegion = .arm
    @Published var stiffness: Double = 0.5            // 0 = limp … 1 = rigid

    /// Upper-body gravity feed-forward. Defaults ON: the robot holding its own
    /// arms up is the normal state; OFF exists to show what the layer buys (or
    /// to hunt droop), not as a mode anyone drives in.
    @Published private(set) var gravityComp = true
    /// Which retargeting the arms run. `false` (default) is the front-workspace
    /// profile that prioritises WRIST position and palm direction; `true` is the
    /// original whole-body map that follows the operator's whole arm.
    @Published private(set) var wholeBodyRetarget = false
    static let maxStiffness = 45.0                    // Nm/rad at slider = 1

    /// One-line summary for status displays, e.g. "Wrist · Hold · Arm".
    var controlSummary: String {
        teleopMethod == .navigation ? "Navigation"
            : "\(teleopMethod.rawValue.capitalized) · \(controllerType.title) · \(region.title)"
    }

    /// Toggle the upper-body gravity feed-forward.
    func setGravityComp(_ on: Bool) {
        guard on != gravityComp else { return }
        gravityComp = on
        link.send(.gravityComp(on))
    }

    /// Switch between the original whole-body retargeting (`true`) and the
    /// wrist-focused front-workspace profile (`false`).
    func setWholeBodyRetarget(_ on: Bool) {
        guard on != wholeBodyRetarget else { return }
        wholeBodyRetarget = on
        link.send(.wholeBodyRetarget(on))
    }

    /// Adopt the server's echoed switch state — a no-op unless the robot
    /// disagrees, so it corrects a stale UI (reconnect, second driver) without
    /// ever fighting the operator mid-tap.
    func adoptSwitches(from st: RobotState) {
        // Only adopt what the robot actually reported. A `nil` means this state
        // carried no opinion (older server, or the offline demo), and adopting a
        // default there would fight the operator: their tap flipped the toggle,
        // the next tick reverted it, and the switch looked broken.
        if let g = st.gravityComp, g != gravityComp { gravityComp = g }
        if let w = st.wholeBodyRetarget, w != wholeBodyRetarget { wholeBodyRetarget = w }
    }

    func setControl(method: TeleopMethod, type: ControllerType, region: ControlRegion) {
        teleopMethod = method; controllerType = type; self.region = region
        sendControlSpec()
    }

    /// Macro driving (base + lift) — the persistent nav joystick's home spec.
    func setNavigation() { setControl(method: .navigation, type: .impedance, region: .wholeBody) }

    /// The REGION the immersive retargeting should own — set from the app's region
    /// toggle. arm → the retargeter tracks only the wrists (arms); upper_body → +
    /// torso; whole_body → + the mobile base (global body position). Defaults to
    /// whole body (the original always-whole behaviour).
    @Published var teleopRegion: ControlRegion = .wholeBody

    /// LOCAL (default) vs GLOBAL retargeting — is the mobile BASE in the follow
    /// loop? LOCAL keeps it out: arms, hands, torso, lift and neck mirror you
    /// while the base stays where YOUR joystick left it. GLOBAL lets it follow
    /// once you leave the comfort box. Orthogonal to `teleopRegion`.
    ///
    /// Defaults LOCAL deliberately: a base that moves on its own is the
    /// consequential state, so it is asked for, never inherited. The server
    /// applies the same rule to an absent or unrecognised value.
    @Published var retargetScope: RetargetScope = .local {
        didSet {
            guard retargetScope != oldValue else { return }
            // Leaving GLOBAL parks the base rather than letting it coast on the
            // last follow command — the switch means "stop following", and the
            // engine only zeroes the base when something overwrites it.
            if retargetScope == .local { stopBase() }
            // UPPER BODY (local) = lift AND base frozen; WHOLE BODY (global) =
            // both follow. One switch, so the two can never disagree.
            trackLift = (retargetScope == .global)
            sendControlSpec()
        }
    }

    /// Unified teleop: body tracking ON engages retargeting (scoped by
    /// `teleopRegion`) from the streamed hand skeletons; OFF falls back to pure
    /// navigation. The nav joystick keeps working in BOTH (the bridge honours it in
    /// manipulation mode too).
    func applyTeleopSpec() {
        guard immersiveActive else { return }
        if bodyTracking {
            setControl(method: .retarget, type: .impedance, region: teleopRegion)
            link.send(.recalibrate)        // anchor the clutch where the hands are NOW
        } else {
            setNavigation()
        }
    }

    func setStiffness(_ k: Double) {
        stiffness = min(1, max(0, k))
        sendControlSpec()
    }

    private func sendControlSpec() {
        link.send(.controlSpec(tracking: ["neck": trackNeck, "arms": trackArms,
                                          "hands": trackHands, "torso": trackTorso,
                                          "lift": trackLift],
                               method: teleopMethod.rawValue, type: controllerType.rawValue,
                               region: region.rawValue,
                               stiffness: stiffness * Self.maxStiffness,
                               scope: retargetScope.rawValue))
    }

    // MARK: - Per-channel tracking (which of MY body parts drive the robot)

    /// Three independent channels, all on by default (the classic whole-body
    /// behaviour). Switching one off is a SERVER-side change, not a client
    /// filter: the engine stops computing that channel and EASES its joints to
    /// a neutral pose over ~2.5 s. That matters — position actuators hold, so a
    /// channel that merely stopped being written would freeze the limb where
    /// the operator left it, and a frozen neck points the cameras at the floor.
    ///
    /// Orthogonal to `region` — which joints the torque engine owns.
    ///
    /// NECK OFF = FROZEN, server-side only (2026-10-16 ablation switches): the
    /// engine parks the neck level and forward. The app KEEPS streaming the live
    /// head — it used to stop (`headEnabled = trackNeck`), which froze the head
    /// pose for EVERYTHING the engine derives from it (arm body frame, torso
    /// lean, lift and base following), so a "neck" ablation silently changed
    /// every other channel too.
    @Published var trackNeck = true {
        didSet {
            guard trackNeck != oldValue else { return }
            sendControlSpec()
        }
    }
    /// TORSO OFF = FROZEN upright at 0 rad (engine `park["torso"]`); the same
    /// retargeter still runs. An ablation switch.
    @Published var trackTorso = true {
        didSet {
            guard trackTorso != oldValue else { return }
            sendControlSpec()
        }
    }
    /// LIFT OFF = FROZEN at its home height (engine `park["lift"]`). Not a
    /// switch of its own on screen: it is half of UPPER BODY (see
    /// `retargetScope`), which freezes the lift AND takes the base out of the
    /// loop. Driving (the drive pad) still moves the lift — the park applies to
    /// retargeting only.
    /// Starts FALSE to match the default scope (local = UPPER BODY): a moving
    /// base is asked for, never inherited, and the lift travels with it.
    @Published private(set) var trackLift = false
    /// The ARM channel — arms, wrists, and the two big reach extenders the
    /// operator groups with them: the torso and the lift. Off → they ease to the
    /// gravity-equilibrium rest pose, arms hanging at the sides.
    @Published var trackArms = true {
        didSet {
            guard trackArms != oldValue else { return }
            sendControlSpec()
        }
    }
    @Published var trackHands = true {
        didSet {
            guard trackHands != oldValue else { return }
            sendControlSpec()
        }
    }

    /// THE BODY DATUM. The operator looks the way they want "straight ahead" to
    /// mean and presses; the server re-derives the body facing from that frame
    /// (`recalibrate` → `TeleopEngine.set_forward`) instead of estimating it.
    ///
    /// Estimating an operator's torso heading from a head and two wrists is the
    /// dominant open problem in `docs/retargeting_modes.tex`, and it does not
    /// have to be solved when the operator can simply declare it. It also
    /// RECOVERS the case that made this button necessary: turn while the hands
    /// are out of view and the estimator holds its old heading through the
    /// dropout and never recovers — measured at 81.5° of arm displacement,
    /// restored to 10.0° by one press (`server/tools/exp_tracking_channels.py`).
    func setForward() {
        Haptics.rigid()
        link.send(.recalibrate)
    }

    // MARK: - Home / rest pose (arms hanging down = minimal holding torque)

    /// The minimal-holding-torque REST pose: the arms hang at the sides with the
    /// elbows EXTENDED; torso + neck neutral; the wrists at their MJCF ref (so the
    /// hand is level); the swerve wheels facing forward (steer 0).
    ///
    /// Re-derived 2026-09 by `controller/experiments/exp_rest_pose.py`, which
    /// minimises the gravity holding torque Στ² (mj_inverse, qvel = qacc = 0) over
    /// the symmetric (shoulder-roll, elbow) pair with the shoulder pitch pinned at 0:
    ///
    ///     qpos zero      Σ|τ| = 38.587 Nm   elbow bend 91.3° — a 90° BENT elbow
    ///     restBodyPose   Σ|τ| =  1.638 Nm   elbow bend 15.1° — the linkage's floor
    ///
    /// Two things changed in that re-derivation. It is now SYMMETRIC (the old
    /// numbers came from a per-arm solve and were 0.002–0.008 rad apart, so the two
    /// arms were never commanded quite the same pose). And the old left elbow,
    /// 1.5795, was OUTSIDE that joint's MJCF limit of 1.542 — the elbows are not
    /// mirrored in the model, so the mirror of the right arm's 1.670 is not a legal
    /// left-arm value. MuJoCo clamped it silently; the real arm would have pushed
    /// into its stop. ±1.52 keeps 0.022 rad of margin on both sides.
    ///
    /// Byte-identical to `REST_POSE` in the studio's homepose.js and to `REST` in
    /// motions.js — ONE rest pose across every client. Values are ABSOLUTE joint
    /// units (what the wire commands use): the arms/torso/neck/steer joints have
    /// ref 0, so those numbers are also the neutral displacement, but the two WRISTS
    /// carry ref ±0.15, so a level hand is commanded by the ref value, not 0 —
    /// sending 0 would twist the wrist by −ref. Fingers relax via `homeGrip`.
    ///
    /// PER ROBOT for the two wrists only: that ±0.15 is MABEL's ORCA v1 ref, while NOVA's
    /// v2 hands speak the URDF frame where a level wrist IS 0 — `RobotModel.handNeutral`
    /// carries the right pair once the hello has picked the rig, so Home never twists
    /// NOVA's wrists by 0.15 rad. Every other joint is the same body on both robots. A side
    /// wearing the OpenArm gripper has no wrist joint, so Home sends it none.
    static var restBodyPose: [String: Double] {
        var p = restBodyPoseV1
        for side in ["left", "right"] {
            // A gripper side has no ORCA wrist joint at all: command none.
            p["\(side)_wrist"] = RobotRig.handGen(side) == "gripper" ? nil : (RobotModel.handNeutral["\(side)_wrist"] ?? 0)
        }
        return p
    }
    private static let restBodyPoseV1: [String: Double] = [
        "left_arm_1": 0, "left_arm_2": -0.1711, "left_arm_3": 0, "left_arm_4": 1.52,
        "left_arm_5": 0, "left_arm_6": 0, "left_arm_7": 0,
        "right_arm_1": 0, "right_arm_2": 0.1711, "right_arm_3": 0, "right_arm_4": -1.52,
        "right_arm_5": 0, "right_arm_6": 0, "right_arm_7": 0,
        "left_wrist": 0.15, "right_wrist": -0.15,          // MJCF ref → level (neutral) hand
        "torso": 0, "neck_yaw": 0, "neck_pitch": 0, "neck_roll": 0,
        "fl_steer": 0, "fr_steer": 0, "b_steer": 0,
        // THE COLUMN, at mid-stroke. It was missing here while the studio's REST_POSE
        // carried it, so "one rest pose across every client" was not true and Home on the
        // phone left the lift wherever it happened to be — the operator presses Home and
        // the robot does not go home. Stage metres, like every other lift value on the
        // wire: 0.15875 is half the 0.3175 m stage travel (= the column's 635 mm mid).
        "lift_lower": 0.15875, "lift_upper": 0.15875,
    ]
    /// A gentle, human-like relaxed curl for the hands at rest (0 = open, 1 = fist).
    /// Driven through `setGrip` (→ the mirror-aware `apply_grasp` on the robot) rather
    /// than per-finger targets: streaming raw finger angles fought apply_grasp and
    /// twisted the RIGHT hand the wrong way (its flexion mirrors the left).
    static let homeGrip = 0.2
    /// Peak per-joint speed cap (rad/s) and the default minimum glide duration (s).
    static let homeMaxJointSpeed = 1.0
    static let homeMinDuration = 3.0
    /// The lift is the one joint in the rest pose that is not an ANGLE, and the cap above
    /// is rad/s. Weighting a stage-metre as if it were a radian budgets the column at
    /// ~150 mm/s — four times what it can do — so the glide would command a ramp the
    /// firmware answers with bang-bang and overshoot. The weight makes the same cap mean
    /// 28 mm/s of column: 1/0.028 per stage-metre, doubled because the two stages extend
    /// together so a stage metre IS two column metres. Byte-identical to the studio's
    /// homepose.js LIFT_RAD_PER_M / LIFT_WEIGHT — one budget across every client.
    static let liftRadPerM = 1 / 0.028
    static let liftWeight = 2 * liftRadPerM
    static func homeWeight(_ id: String) -> Double {
        (id == "lift_lower" || id == "lift_upper") ? liftWeight : 1
    }

    /// True while the Home glide streams: the model shows the COMMANDED rest pose
    /// (telemetry would otherwise mask it, e.g. on a sim that doesn't echo joint
    /// commands). Cleared when the glide finishes/cancels.
    @Published private(set) var homing = false
    private var homeTimer: Timer?

    /// Glide the whole body to the minimal-torque REST pose (arms down · torso/neck
    /// neutral · wheels forward · hands relaxed) over ~3 s, eased and speed-capped.
    /// Works on sim + real: it declares joint control, then streams the SAME
    /// smoothstep trajectory the operator watches — nothing snaps. Re-pressing
    /// restarts the glide from wherever the arms currently are. (Named `goRestPose`,
    /// not `goHome`, because VP's `goHome` is the autonomy nav-home used by RobotHub.)
    /// End whole-body mirroring — HANDS AND HEAD.
    ///
    /// Clearing `bodyTracking` alone stops the hands; the frame source keeps
    /// streaming a LIVE head pose whenever the immersive space is open, so the
    /// robot's neck went on following the operator's head after they pressed
    /// Stop Teleop. From the operator's seat that is "the tracking is still
    /// running" — they cannot see which limb the wire thinks it is driving
    /// (reported 2026-07-30, full immersion).
    ///
    /// `headEnabled` is a real feature on its own (active perception without the
    /// arms), so this does not couple the two flags permanently — it turns head
    /// mirroring off at the moments the operator has explicitly said "stop", and
    /// engaging teleop again turns it back on.
    func stopWholeBodyTeleop() {
        bodyTracking = false          // didSet → spec + wire flag
        headEnabled = false           // freeze the neck at the last anchor
    }

    func goRestPose() {
        // HOME MUST WIN. The rest-pose glide writes joint targets, and a live
        // retargeter re-asserts the operator's arms on the very next frame — so
        // pressing Home mid-teleop produced a robot fighting itself, twitching
        // toward home and being dragged back, with no indication why. Stopping
        // the mirroring first is what makes Home a command rather than a
        // suggestion (reported 2026-07-30).
        if bodyTracking || immersiveActive { stopWholeBodyTeleop() }
        homeTimer?.invalidate()
        // Enable joint control so the pose is honoured (a Home pressed outside teleop
        // would otherwise be ignored while the robot is in navigation mode).
        setControl(method: .joint, type: .impedance, region: .wholeBody)
        // THE COLUMN IS NOT GLIDED. It is a BANG-BANG servo whose only speeds are ~37 mm/s
        // and zero, so a streamed trajectory asking for anything slower is tracked by
        // stop-starting — 54 stops on a full-stroke Home, each a fresh full-duty pulse,
        // which is what "the lift moves in jittery motion" is (measured in NOVA's
        // lift simulation: 124 stops streamed vs 4 one-shot). It gets ONE
        // absolute target and the firmware runs its own profile and taper. The arms are
        // PD-tracked, so for them a streamed trajectory is exactly right.
        let lift = Self.restBodyPose.filter { Self.liftJoints.contains($0.key) }
        let target = Self.restBodyPose.filter { !Self.liftJoints.contains($0.key) }
        let live = liveModelJoints
        var start: [String: Double] = [:]
        // An unknown start falls back to the TARGET ("already there", so it contributes
        // no motion), never to 0 — on the lift, 0 is the fully RETRACTED column, so a
        // fabricated zero start is a 317 mm sweep down and back that nobody asked for.
        for id in target.keys { start[id] = live[id] ?? jointTargets[id] ?? target[id] }
        // The hands relax through the grip (→ mirror-aware apply_grasp), eased from
        // wherever each hand is now to the gentle rest curl.
        let gripStart = ["left": jointTargets["left_grip"] ?? 0,
                         "right": jointTargets["right_grip"] ?? 0]
        // Duration: default, stretched so no joint's PEAK speed (smoothstep peaks at
        // 1.5× the average) exceeds the cap — a big swing simply glides for longer.
        let maxDelta = target.map {
            abs($0.value - (start[$0.key] ?? $0.value)) * Self.homeWeight($0.key)
        }.max() ?? 0
        let dur = max(Self.homeMinDuration, 1.5 * maxDelta / Self.homeMaxJointSpeed)
        homing = true
        Haptics.rigid()
        if !lift.isEmpty { setJoints(lift) }           // once, before the glide starts
        let t0 = Date()
        homeTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] tm in
            guard let self else { tm.invalidate(); return }
            let raw = min(1.0, Date().timeIntervalSince(t0) / dur)
            let e = raw * raw * (3 - 2 * raw)          // smoothstep ease
            var cfg: [String: Double] = [:]
            for (id, t) in target {
                let s = start[id] ?? t
                cfg[id] = s + (t - s) * e
            }
            self.setJoints(cfg)                        // body: ONE batched stream per tick
            // Hands: eased grip (the finger echo + the apply_grasp signal); NEVER raw
            // per-finger targets — those fight apply_grasp and mis-mirror the R hand.
            for side in ["left", "right"] {
                let g = (gripStart[side] ?? 0) + (Self.homeGrip - (gripStart[side] ?? 0)) * e
                self.setGrip(side, g)
            }
            if raw >= 1 {
                tm.invalidate(); self.homeTimer = nil
                self.homing = false
                Haptics.tap()
            }
        }
    }

    /// Abort an in-flight Home glide — leaves the arms wherever they are and hands the
    /// model back to live telemetry.
    func cancelRestPose() {
        guard homeTimer != nil else { return }
        homeTimer?.invalidate(); homeTimer = nil
        homing = false
    }

    /// Push a robot body in a world-frame direction (drag on the 3D model).
    func applyExternalForce(_ forces: [String: [Double]]) {
        link.send(.externalForce(forces))
    }
    func clearExternalForce() { link.send(.externalForce([:])) }

    /// Arm-joint group → the MuJoCo body that receives the push.
    static func pushBody(forJoint id: String) -> String? {
        let side = id.hasPrefix("left") ? "l" : (id.hasPrefix("right") ? "r" : nil)
        guard let s = side else { return nil }
        if id.contains("hand") || id.contains("grip") || id.contains("palm") { return "\(s)_hand_mount" }
        guard let n = id.split(separator: "_").last.flatMap({ Int($0) }) else { return "\(s)_forearm" }
        switch n {
        case 1, 2, 3: return "\(s)_arm_mount"
        case 4, 5:    return "\(s)_forearm"
        default:      return "\(s)_wrist"
        }
    }

    func setLift(fraction: Double) {
        jointTargets["lift_lower"] = fraction * 0.3175
        jointTargets["lift_upper"] = fraction * 0.3175
        link.send(.liftFraction(fraction))
    }

    static let liftStageMax = 0.3175
    static let liftMax = liftStageMax * 2          // 0.635 m total travel
    var liftHeight: Double { (jointTargets["lift_lower"] ?? 0) + (jointTargets["lift_upper"] ?? 0) }

    /// TOTAL column travel in metres, split evenly across the two SYNC'd stages
    /// (the MJCF ties them with an equality, so they are one axis in every
    /// command path). Routed through `setJoints` so it inherits the ONE Soft
    /// choke point and the instant model echo, like every other joint command.
    func setLiftHeight(_ meters: Double) {
        let half = min(Self.liftStageMax, max(0, meters / 2))
        // A DELIBERATE move — the slider, the jog, Home. It is also the moment the
        // operator takes ownership of the column, so it ends the adoption wait even if no
        // telemetry ever named the lift.
        liftAdopted = true
        setJoints(["lift_lower": half, "lift_upper": half])
    }

    func setLiftRate(_ rate: Double) {
        guard !estopEngaged else { return }
        // Passive Soft: a lift rate is a command. Zero still gets through so
        // entering Soft can park the column rather than leaving it jogging.
        guard !passiveCompliance || rate == 0 else { return }
        // NOT scaled by speedScale: the Speed Governor is for the mobile base only, the lift always
        // runs full speed (the robot maps any push to full duty anyway — NOVA server/plant.py).
        link.send(.liftRate(rate))
    }

    var liftFraction: Double {
        let total = (state.joints["lift_lower"] ?? 0) + (state.joints["lift_upper"] ?? 0)
        return min(1, max(0, total / (0.3175 * 2)))
    }

    func goHome() {
        Haptics.success()
        snapModelToHome()
        link.send(.goHome)
    }

    // MARK: - Policies

    func dispatch(_ policy: Policy) {
        Haptics.rigid()
        link.send(.dispatchPolicy(id: policy.id))
    }

    // MARK: - Safety

    func engageEStop() {
        estopEngaged = true
        torqueOnRequested = false; torqueRequestAt = nil
        safetyNotice = nil
        stopBase()
        let now = Date().timeIntervalSince1970
        stopRequestedAt = now; stopSentAt = now
        link.send(.estop(true))
    }

    func releaseEStop() {
        if link is DemoLink {                       // the on-device sim has no server to confirm
            estopEngaged = false
            link.send(.estop(false))
            return
        }
        guard connection.isLive else {
            estopEngaged = true
            safetyNotice = "No robot connection. Reconnect and tap Torque On again."
            return
        }
        safetyNotice = nil
        stopRequestedAt = nil
        torqueOnRequested = true; torqueRequestAt = Date().timeIntervalSince1970
        estopEngaged = true                         // wait for the server's release echo
        link.send(.estop(false))
    }

    /// Follow the SERVER's e-stop — one flag for every client — so a TORQUE ON or
    /// E-STOP from any device (iPhone, studio, another headset) shows here too.
    /// Our own just-sent E-STOP wins until the server confirms it.
    private func ingestSafety(_ s: RobotState) {
        let now = Date().timeIntervalSince1970
        guard let serverStop = s.estop else {
            // Older bridge / demo: no echo to wait for — keep the old optimistic path.
            if torqueOnRequested { torqueOnRequested = false; torqueRequestAt = nil; estopEngaged = false }
            return
        }
        if serverStop {
            stopRequestedAt = nil                   // our stop (if any) has landed
            if torqueOnRequested {
                if now - (torqueRequestAt ?? now) > 4 {
                    torqueOnRequested = false; torqueRequestAt = nil
                    safetyNotice = "Torque request unconfirmed"
                }
            } else if !estopEngaged {
                estopEngaged = true
            }
        } else if stopRequestedAt != nil {
            estopEngaged = true                     // not landed yet: keep it, resend
            if now - stopSentAt >= 0.75 { stopSentAt = now; link.send(.estop(true)) }
        } else if estopEngaged || torqueOnRequested {
            torqueOnRequested = false; torqueRequestAt = nil
            safetyNotice = nil
            estopEngaged = false
        }
    }

    // MARK: - Body calibration

    /// Begin the guided 3-pose capture for `name`. The bridge runs the capture
    /// from our streamed pose frames; progress + the result arrive in telemetry,
    /// and the finished profile is saved on-device under `name` automatically.
    func startBodyCalibration(name: String) {
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let user = n.isEmpty ? "operator" : n
        calibratingName = user
        liveMetrics = nil
        calibrationStartedAt = Date()
        calibrationServerHeard = false
        appliedSavedProfile = nil
        // Optimistic prompt so the UI advances instantly; telemetry refines it.
        calibration = CalibrationProgress(active: true, pose: "arms_forward",
                                          index: 0, total: 1, done: false, reach: nil)
        startCalibrationLive()
        Haptics.rigid()
        link.send(.startBodyCalibration(user: user))
    }

    /// Abort an in-progress capture.
    func cancelBodyCalibration() {
        calibratingName = nil
        calibration = nil
        calibrationStartedAt = nil
        calibrationServerHeard = false
        stopCalibrationLive()
        link.send(.cancelBodyCalibration)
    }

    // MARK: - Live calibration coaching (see `CalibrationLive`)

    /// Sample head→wrist at 20 Hz while a capture is in flight. Cheap: two anchor
    /// reads and a distance, no allocation storm — it must never compete with the
    /// 60 Hz frame source that is actually feeding the server's measurement.
    private func startCalibrationLive() {
        stopCalibrationLive()
        let t = Timer(timeInterval: 1.0 / 20.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sampleCalibrationLive() }
        }
        RunLoop.main.add(t, forMode: .common)
        calLiveTimer = t
    }

    private func stopCalibrationLive() {
        calLiveTimer?.invalidate(); calLiveTimer = nil
        calCoach.reset()
        calibrationLive = CalibrationLive()
    }

    /// One 20 Hz sample: read the ARKit anchors and hand them to the pure coach.
    /// Cheap by design — two anchor reads and a distance. It must never compete
    /// with the frame source that feeds the server's actual measurement.
    private func sampleCalibrationLive() {
        func origin(_ m: simd_float4x4) -> SIMD3<Float> {
            SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        }
        let now = CACurrentMediaTime()
        let head = headService.currentPose(at: now).map { origin($0.transform.simd) }
        let (l, r) = latestHands.load()
        calibrationLive = calCoach.observe(
            head: head,
            leftWrist: (l?.isTracked ?? false) ? origin(l!.anchorTransform.simd) : nil,
            rightWrist: (r?.isTracked ?? false) ? origin(r!.anchorTransform.simd) : nil,
            now: now)
    }

    /// Re-apply a saved profile on the robot and mark it active locally.
    func applyProfile(_ profile: CalibrationProfile) {
        Haptics.success()
        calibrationStore.setActive(profile.name)
        link.send(.applyBodyCalibration(user: profile.name))
    }

    /// The saved calibration "Use Saved Calibration" would apply for `name`.
    func preferredSavedProfile(forName name: String) -> CalibrationProfile? {
        calibrationStore.preferred(forName: name)
    }

    /// Where an in-flight capture stands against the stall watchdog.
    func calibrationStall(now: Date = Date()) -> CalibrationStall {
        guard let t0 = calibrationStartedAt, calibratingName != nil else { return .waiting }
        return CalibrationStall.verdict(elapsed: now.timeIntervalSince(t0),
                                        serverHeard: calibrationServerHeard,
                                        done: calibration?.done ?? false)
    }

    /// "USE MY SAVED NUMBERS" (operator 2026-10-12): skip the capture and drive
    /// with a profile measured earlier. Abandons any in-flight capture, marks
    /// the profile active, and asks the server to load it.
    ///
    /// HONEST LIMIT: the wire's apply is `body_calibration {action: "apply",
    /// user}` — the NAME only. The MABEL bridge then loads ITS OWN
    /// `profiles/<user>.json` (engine.load_profile), i.e. the reach saved on
    /// that server when this operator last calibrated there; the number in
    /// this app is not transmitted. A server with no file for the name (or one
    /// that does not implement body_calibration, like NOVA's gateway) keeps its
    /// default REACH_SCALE. Sending the reach itself needs an additive wire
    /// field + a bridge change.
    func useSavedCalibration(_ profile: CalibrationProfile) {
        if calibratingName != nil || (calibration?.active ?? false) {
            cancelBodyCalibration()
        }
        applyProfile(profile)
        appliedSavedProfile = profile
    }

    func deleteProfile(_ profile: CalibrationProfile) {
        Haptics.soft()
        calibrationStore.remove(profile)
    }

    func isActiveProfile(_ profile: CalibrationProfile) -> Bool {
        calibrationStore.isActive(profile)
    }

    /// Fold the calibration telemetry into published state, and save the profile
    /// exactly once when a capture we started completes.
    private func ingestCalibration(_ s: RobotState) {
        if let m = s.operatorMetrics { liveMetrics = m }
        // Adopt only a NON-NIL server status. robot_state frames that raced the
        // body_calibration start carry calibrationStatus = nil, and adopting that
        // wiped the optimistic `active` — which closed the hands-during-capture
        // e-stop exception (`!estopEngaged || calibrating`), so the server never
        // received the very hands it needed to start measuring (chicken-and-egg).
        // A stale done=true from a PRIOR capture is still kept out once we've
        // finished saving (calibratingName nil + server active false → no adopt).
        if let sc = s.calibration, calibratingName != nil || sc.active {
            calibration = sc
            if calibratingName != nil, !calibrationServerHeard { calibrationServerHeard = true }
        }
        guard let cal = s.calibration, cal.done,
              let name = calibratingName, let m = s.operatorMetrics else { return }
        calibrationStore.upsert(CalibrationProfile(name: name, metrics: m))
        calibrationStore.setActive(name)
        calibratingName = nil
        calibrationStartedAt = nil
        stopCalibrationLive()      // the capture is over; stop sampling
        Haptics.success()
    }
}

// MARK: - Thread-safe wire-gating flags

/// Lock-protected mirror of the MainActor flags that gate the outbound
/// teleop_frame, so the frame source can run entirely on the wire queue.
/// Written from RobotController property didSets; read once per frame tick.
/// Defaults MUST match the controller's boot state (estop engaged).
final class WireFlags: @unchecked Sendable {
    struct Values {
        var immersiveActive = false
        var headEnabled = true
        var bodyTracking = false
        var estop = true          // the app boots e-stopped
        var calibrating = false
        var streamAlways = false  // the gateway is a data/retarget lab (hello.caps)
    }
    private let lock = NSLock()
    private var values = Values()

    var immersiveActive: Bool {
        get { lock.withLock { values.immersiveActive } }
        set { lock.withLock { values.immersiveActive = newValue } }
    }
    var headEnabled: Bool {
        get { lock.withLock { values.headEnabled } }
        set { lock.withLock { values.headEnabled = newValue } }
    }
    var bodyTracking: Bool {
        get { lock.withLock { values.bodyTracking } }
        set { lock.withLock { values.bodyTracking = newValue } }
    }
    var estop: Bool {
        get { lock.withLock { values.estop } }
        set { lock.withLock { values.estop = newValue } }
    }
    var calibrating: Bool {
        get { lock.withLock { values.calibrating } }
        set { lock.withLock { values.calibrating = newValue } }
    }
    var streamAlways: Bool {
        get { lock.withLock { values.streamAlways } }
        set { lock.withLock { values.streamAlways = newValue } }
    }
    /// One lock round-trip for the whole per-frame read.
    func load() -> Values { lock.withLock { values } }
}

// MARK: - Thread-safe hand snapshot box

/// Lock-protected single-slot for the most recent hand snapshot, written by
/// the ARKit broadcast task and read at frame-send time. Also stashes the
/// frozen head pose used while head tracking is toggled off.
private final class LatestHands: @unchecked Sendable {
    private let lock = NSLock()
    private var _left: HandPose?
    private var _right: HandPose?
    private var _frozenHead: HeadPose?

    var frozenHead: HeadPose? {
        get { lock.withLock { _frozenHead } }
        set { lock.withLock { _frozenHead = newValue } }
    }

    // Per-side ARKit refresh stamps — a side is only as fresh as its OWN last
    // anchor update, never as fresh as the snapshot that carried it.
    private var _leftStamp: TimeInterval = 0
    private var _rightStamp: TimeInterval = 0

    func store(_ s: HandTrackingService.Snapshot) {
        lock.withLock {
            _left = s.left; _right = s.right
            _leftStamp = s.leftStamp; _rightStamp = s.rightStamp
        }
    }
    /// Sides refreshed within `timeout`. Expiring at READ time (the 60 Hz wire
    /// tick) is what covers the case the producer cannot: when BOTH hands leave
    /// the field of view no anchor update arrives at all, so nothing downstream
    /// would ever clear them and the robot would keep both arms pinned to the
    /// last poses it saw.
    func load(timeout: TimeInterval = HandTrackingService.sideStaleTimeout)
        -> (HandPose?, HandPose?) {
        let cutoff = CACurrentMediaTime() - timeout
        return lock.withLock {
            (_leftStamp >= cutoff ? _left : nil, _rightStamp >= cutoff ? _right : nil)
        }
    }
    func clear() {
        lock.withLock { _left = nil; _right = nil; _leftStamp = 0; _rightStamp = 0 }
    }
}

// MARK: - Automatic Bluetooth (2026-10-25: "super automatic — if the connection drops it should reconnect")
//
// Every 2 s: when nothing better is live and the robot is HEARD over Bluetooth, take it — no card, no button.
// The network gets the first 6 s (Wi-Fi and Tailscale outrank Bluetooth: they carry video); an
// automatic fall-back to the SIMULATION loses to a real robot in radio range at once. A Bluetooth link that
// drops re-connects by itself (BLELink.didDisconnectNote); after 20 s without it the network gets a turn again,
// and the tick comes back to Bluetooth if that finds nothing. Bonjour still moves a Bluetooth link UP to the
// Wi-Fi (upgradeFromBluetoothIfPossible). An explicit Disconnect switches all of this off until the next connect.
// Connecting never energises anything: torque stays the operator's button.
extension RobotController {
    func startAutoBluetooth() {
        bleAutoOff = false
        guard bleAutoTimer == nil else { return }
        bleAutoTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.autoBluetoothTick() }
        }
    }

    /// The robot heard over Bluetooth right now, strongest first, at ANY distance: the proximity card's
    /// arm's-reach threshold is for offering a connection, not for keeping one.
    var bluetoothRobotInRange: NearbyRobot? {
        BLECentral.shared.nearby.first { Date().timeIntervalSince($0.lastSeen) < 6 }
    }

    private func autoBluetoothTick() {
        guard !bleAutoOff else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if overBluetooth {
            bleNotLiveSince = nil
            if connection == .connected { bleDownSince = nil; return }
            if bleDownSince == nil { bleDownSince = now }
            if now - (bleDownSince ?? now) > 20, let real = robotStore.robots.first(where: { $0.kind == .real }) {
                bleDownSince = nil
                connect(to: real, explicit: false)
                bleAutoOff = false
            }
            return
        }
        bleDownSince = nil
        let onAutoSim = connection == .connected && target.isSimulation && pinnedRigID == nil
        if connection == .connected && !onAutoSim { bleNotLiveSince = nil; return }
        if bleNotLiveSince == nil { bleNotLiveSince = now }
        guard onAutoSim || now - (bleNotLiveSince ?? now) >= 6, let ble = bluetoothRobotInRange else { return }
        bleNotLiveSince = nil
        connectBluetooth(ble) { _ in }
    }
}
