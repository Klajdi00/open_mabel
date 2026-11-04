//
//  RobotController.swift
//  MABEL Mobile App
//
//  The app's single source of truth. Owns the transport link, publishes
//  live telemetry to SwiftUI, and exposes intent-level methods the views
//  call ("nudge this joint", "drive with this twist", "run this policy").
//

import SwiftUI
import Combine
import Network
import simd

/// Live 3D pose, decoupled from SwiftUI — the iOS analogue of the browser teleop's
/// `sim.state.q` dict. The network thread writes the newest merged pose here CHEAPLY
/// (no `@Published`, so the 50 Hz stream never invalidates the SwiftUI tree), and the
/// 3D viewers read it from a `CADisplayLink` render loop (the rAF equivalent) and apply
/// it straight to the SceneKit nodes. `tick` bumps on every write so a viewer applies
/// only when the pose actually changed. Main-thread confined (writes from `onState`
/// on the main delegate queue; reads from the main-thread display link) — no locking.
final class LivePose {
    private(set) var joints: [String: Double] = [:]
    private(set) var heading: Double = 0
    private(set) var baseQuat: [Double] = [1, 0, 0, 0]
    /// Base odom position (ROS frame, metres) from robot_state @~100 Hz —
    /// the viewer's ROBOT-POSE FALLBACK when the map stream is off
    /// (hello.ports.map = 0): without it the rendered robot stays at the
    /// origin while world-frame clouds draw at the TRUE pose (the
    /// "pointclouds vanished after relocate" desync).
    private(set) var baseX: Double = 0
    private(set) var baseY: Double = 0
    private(set) var tick: UInt64 = 0
    func set(_ j: [String: Double], heading h: Double, quat q: [Double],
             x: Double = 0, y: Double = 0) {
        joints = j; heading = h; baseQuat = q; baseX = x; baseY = y; tick &+= 1
    }
}

/// Task-space wrist-drive rates — written by the teleop tick (30 Hz), consumed
/// by the viewer's display-link IK (no @Published churn, like LivePose).
/// BASE-frame semantics, m/s: x = forward (the robot's nose), y = LEFT,
/// z = up. The viewer resolves these into world space against the live rig.
final class WristDrive {
    var left = SIMD3<Float>.zero
    var right = SIMD3<Float>.zero
    var active: Bool { simd_length(left) > 1e-4 || simd_length(right) > 1e-4 }
    func stop() { left = .zero; right = .zero }
}

@MainActor
final class RobotController: ObservableObject {

    /// Decoupled 3D pose for the SceneKit viewers (see `LivePose`). Updated every
    /// telemetry frame WITHOUT touching `@Published`, so the 3D model stays real-time
    /// while the heavy SwiftUI invalidation runs at a throttled rate.
    let livePose = LivePose()
    let wristDrive = WristDrive()
    private var lastTelemetry: [String: Double] = [:]   // newest streamed joint subset
    private var lastHeading: Double = 0
    private var lastQuat: [Double] = [1, 0, 0, 0]
    private var lastStatePublish: CFTimeInterval = 0     // throttles the @Published state

    // Published, drives the entire UI.
    @Published private(set) var state = RobotState()
    /// Latched planar LiDAR sweep (the bridge attaches one to robot_state only
    /// when fresh, ~5 Hz — held here so the throttled `state` publish can't
    /// drop the sparse frames it happens to skip).
    @Published private(set) var lidarScan: RobotState.LidarScan? = nil
    /// Latched sparse joint dynamics (attached to robot_state ~10 Hz, same
    /// contract as lidar). NOT @Published — the console telemetry panels poll
    /// these at their own sample cadence, so a 10 Hz latch never churns SwiftUI.
    private(set) var jointVel: [String: Double] = [:]
    private(set) var jointTorque: [String: Double] = [:]
    private(set) var jointTemp: [String: Double] = [:]
    /// Latched per-module connection roster (change-gated on the wire — see the
    /// latch in the state handler). Re-stamped onto every published state so the
    /// status panel always has the CURRENT roster, not just the frame it arrived on.
    private(set) var moduleRoster: [RobotState.Module] = []
    @Published private(set) var connection: ConnectionStatus = .disconnected
    /// What we're actually driving — a simulation or real hardware. Drives the
    /// status chip and gates the Reset button (sim only).
    @Published private(set) var target: ConnectionTarget = .simulation
    @Published var mode: TeleopMode = .drive
    @Published var estopEngaged = false
    @Published private(set) var stopState = RobotStopState()
    @Published private(set) var safetyNotice: String?
    @Published private(set) var torqueOnRequested = false
    private var torqueRequestAt: TimeInterval?
    private var stopRetryTimer: Timer?
    /// Presents the full-screen Data Collection view. Held here (not in a tab's
    /// @State) so it survives the view-tree swap on device rotation.
    @Published var collectingData = false
    /// 3rd ↔ 1st person viewer mode — held here for the same reason: rotation
    /// swaps the Robot space's view tree, and per-view @State snapped an
    /// operator in 1st person back to 3rd on every portrait↔landscape flip.
    /// DEBUG: `debugPov` boots straight into 1st (argv is dropped on prewarmed
    /// launches — the defaults channel is the reliable one).
    @Published var povMode: PovMode = {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-pov"), i + 1 < args.count,
           let m = PovMode(rawValue: args[i + 1]) { return m }
        if let s = UserDefaults.standard.string(forKey: "debugPov"),
           let m = PovMode(rawValue: s) { return m }
        #endif
        return .third
    }()

    /// 1st-person video gates. Held HERE, not in RobotSpaceView's @State, for the
    /// same reason as povMode: rotation swaps ContentView's portrait/landscape view
    /// trees, so per-view @State is destroyed — and the operator, mid-teleop in 1st
    /// person, was dropped back onto the raw 3D render until a timer re-raised the
    /// video. `fpvVisible` = the camera flight has landed; `fpvLive` = the head
    /// stream is actually delivering frames.
    @Published var fpvVisible = false
    @Published var fpvLive = false

    /// Operator-facing speed scaling (0.1...1) — a safety governor. Defaults to
    /// full so the base drives at the SAME ceiling as scripts/teleop_keyboard.py
    /// (full stick → MAX_LIN_SPEED). Lower it for finer control.
    @Published var speedScale: Double = 1.0

    /// Local echo of joint targets so sliders feel instant even before
    /// telemetry round-trips back.
    @Published var jointTargets: [String: Double] = RobotModel.homePose

    /// What the 3D model should render: the ACTUAL robot state, with the grips
    /// taken from the command echo (the bridge doesn't stream grip telemetry,
    /// and the rig has no finger sensors anyway).
    var modelJoints: [String: Double] {
        var j = state.joints
        j["left_grip"] = jointTargets["left_grip"] ?? j["left_grip"]
        j["right_grip"] = jointTargets["right_grip"] ?? j["right_grip"]
        return j
    }

    /// Joints the bridge streams as live telemetry — overlaid onto the command echo
    /// so the 3D model mirrors the ACTUAL robot (MuJoCo qpos), not what we asked
    /// for. The bridge now streams the wrists AND every finger joint (realized
    /// hand pose), so we include them here — otherwise the hands/wrists render the
    /// command echo and visibly diverge from MuJoCo. Only `left_grip`/`right_grip`
    /// (synthetic slider values, not rig joints) stay command-only.
    static let telemetryJoints: Set<String> = {
        var s = Set<String>(["torso", "neck_yaw", "neck_pitch", "neck_roll",
                             "lift_lower", "lift_upper", "fl_steer", "fr_steer", "b_steer"])
        for i in 1...7 { s.insert("left_arm_\(i)"); s.insert("right_arm_\(i)") }
        for side in ["left", "right"] {
            s.insert("\(side)_wrist")
            s.insert("\(side)_gripper")                  // OpenArm gripper opening (m), when mounted
            for f in ["thumb", "index", "middle", "ring", "pinky"] {
                let segs = f == "thumb" ? ["mcp", "abd", "pip", "dip"] : ["abd", "mcp", "pip"]
                for seg in segs { s.insert("\(side)_\(f)_\(seg)") }
            }
        }
        return s
    }()

    /// Body-tab pose: the ACTUAL robot state for the arms/torso/lift/neck (so
    /// soft-mode yielding and impedance tracking are visible on the model) merged
    /// with the COMMANDED finger joints (telemetry doesn't stream the fingers, so
    /// the hands keep curling from `jointTargets`).
    var liveModelJoints: [String: Double] {
        var j = jointTargets                                  // commanded base (incl. fingers)
        for k in Self.telemetryJoints { if let v = state.joints[k] { j[k] = v } }
        return j
    }

    /// Align the local command echo with the robot's CURRENT streamed pose —
    /// entering a direct-control surface (Joint/Task space, a module switch)
    /// must start from where the robot actually IS: the drive-mode lift jog,
    /// another client, or the WBC all move joints without touching
    /// `jointTargets`, and a stale target makes the first slider touch JUMP
    /// the joint to wherever it was last commanded.
    /// False until this link has told us where the column ACTUALLY is.
    ///
    /// `jointTargets` is reset to `RobotModel.homePose` on a robot switch, and homePose
    /// puts the two lift stages at their ref — mid-stroke — because that is the right pose
    /// to DRAW an unknown robot in. It is not a pose to COMMAND. Switching from Wi-Fi to
    /// Bluetooth reset the targets and the very next thing that streamed them drove the
    /// column to mid-stroke, unasked: a heavy axis moving on its own is the one thing a
    /// transport switch must never do.
    ///
    /// So no lift command leaves this app until a telemetry frame has said where the
    /// column is and `jointTargets` has adopted it. After that the targets already equal
    /// the measurement, so an incidental re-send is a no-op and only a deliberate move —
    /// the slider, the jog, Home — changes anything.
    private var liftAdopted = false

    /// Lift ids, in the one place that has to agree with itself.
    static let liftJoints = ["lift_lower", "lift_upper"]

    func syncTargetsToState() {
        for k in Self.telemetryJoints {
            if let v = state.joints[k] { jointTargets[k] = v }
        }
        refreshLivePose()
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
        livePose.set(j, heading: lastHeading, quat: lastQuat,
                     x: state.baseX, y: state.baseY)
        // The demo navigator's speed cap is the tip-over model at the CURRENT lift
        // height (MPPINavigator) — keep it in step with the commanded column.
        liveMap.demoLiftHeight = (j["lift_lower"] ?? 0) + (j["lift_upper"] ?? 0)
    }

    /// Teleop model pose: the operator-commanded joints (arms/fingers/lift) plus
    /// the live swerve steer angles from telemetry, so the model reflects what's
    /// commanded AND turns/steers with the real base. Pair with `state.heading`.
    var teleopModelJoints: [String: Double] {
        var j = jointTargets
        for s in ["fl_steer", "fr_steer", "b_steer"] where state.joints[s] != nil {
            j[s] = state.joints[s]
        }
        return j
    }

    /// The robot's configured max base speeds (used to scale joysticks).
    let maxLinear = 1.2          // m/s
    let maxAngular = 1.8         // rad/s

    /// Base URL for the live MJPEG camera streams when connected to a real
    /// host (nil in demo mode → the cockpit shows the synthetic feed).
    @Published private(set) var videoBase: URL? = nil

    /// Video streams ONLY while the teleop link is actually CONNECTED. The WS
    /// (:9090) and the MJPEG server (:8080) are independent transports, and
    /// `videoBase` is set optimistically at route-apply time (before the WS
    /// answers) and re-set by every failover cycle — so without this gate a
    /// dead/racing teleop link left every camera tile + Cam3D plane pulling
    /// real video from the host while the app read "Offline". `.demo` doesn't
    /// count: the on-device sim has no host to stream from.
    private var videoAllowed: Bool { connection == .connected }

    /// Full MJPEG URL for a camera stream. Single source of truth for every
    /// video tile, so URL construction lives in one place.
    func videoURL(_ stream: String) -> URL? {
        guard videoAllowed, let base = videoBase else { return nil }
        return base.appendingPathComponent("camera/\(stream)/stream.mjpg")
    }

    // ── Auto-connect (fastest network first) ─────────────────────────────────
    /// Races LAN / Tailscale candidates.
    let planner = ConnectionPlanner()
    /// The transport the live link is using — drives the Settings network card.
    /// Only set once the link actually reports connected (see `pendingRoute`), so
    /// the displayed method is never an optimistic guess from a route we're still
    /// trying (e.g. it stays correct through a LAN→Tailscale failover).
    @Published private(set) var activeRoute: ConnectionRoute? = nil
    /// The route whose link we are CURRENTLY bringing up; promoted to `activeRoute`
    /// the moment that link connects.
    private var pendingRoute: ConnectionRoute? = nil
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

    /// The remembered set of robots (sim + real), persisted and synced.
    let robotStore = RobotStore()

    /// Live SLAM-map stream (nvblox mesh + odom) for the Maps tab. Rides its own
    /// WebSocket (ws://host:9090/map), pointed at whichever robot we're driving;
    /// the Maps tab toggles `active` so it only streams while on screen.
    let liveMap = LiveMapStore()
    /// The `:9610` duplex voice plane (talk through the robot / hear its room).
    let voice = AudioStreamLink()
    /// Live server identity + speed caps from the connected robot (drives the
    /// Settings name + the speed sliders). nil until a real link reports it.
    @Published private(set) var serverMeta: ServerMeta? = nil

    private var routeIdx = 0
    private var failoverTimer: Timer?
    /// Armed when an ALREADY-connected link drops. The per-link reconnect only
    /// retries the SAME endpoint, so if that endpoint vanished (e.g. Wi-Fi turned
    /// off and the phone moved to another network) it would loop forever. This timer
    /// gives the same-route reconnect a short grace window to recover a transient
    /// blip, then re-races ALL routes so a live one (e.g. Tailscale) wins.
    private var recoveryTimer: Timer?
    private var autoWanted = false              // true between connect(to:) and disconnect
    private var lastDiscovered: [String] = []
    /// The robot the OPERATOR picked (robot list / status bar). Discovery never
    /// switches away from it; nil while auto-connect is choosing.
    private var pinnedRigID: String?
    /// The robot we're racing toward — its identity + its OWN routes (so the sim
    /// and the real rig never share a host pool) + the secret access code (real
    /// robot only; gates every transport).
    private var connectTarget: (id: String, hosts: [String], accessCode: String?)?

    private var link: RobotLink

    /// iPhone attitude → robot head active perception.
    let poseTracker = DevicePoseTracker()
    @Published private(set) var headTracking = false

    /// How strongly the phone's tilt maps to the robot's head (0.3…1.2).
    @Published var headSensitivity: Double = 0.85 {
        didSet { poseTracker.motionScale = headSensitivity }
    }

    private var storeObserver: AnyCancellable?
    private var liveMapObserver: AnyCancellable?

    init(link: RobotLink? = nil) {
        self.link = link ?? DemoLink()
        wire(self.link)
        startNetworkMonitor()
        poseTracker.onUpdate = { [weak self] m in self?.link.send(.headPose(m)) }
        // The robot list is a nested ObservableObject; forward its changes so
        // views observing the controller refresh when robots are found/renamed.
        storeObserver = robotStore.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        // Same for the live-map stream's stats (connection / point count) so the
        // Maps overlay refreshes through the controller's environment object.
        liveMapObserver = liveMap.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }

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

    /// Point the voice plane at the robot. Shares the map endpoint's lifecycle —
    /// both are derived from `activeRoute` + `hello.ports`, and BOTH can be asked
    /// for before either exists (applyRoute runs while activeRoute is still nil,
    /// and `hello.ports` can arrive after the connection flips), so this is called
    /// again on serverMeta changes.
    func refreshAudioEndpoint() {
        let url = audioStreamURLs().first
        #if DEBUG
        // Voice failing to come up looks exactly like voice being broken, and the
        // three reasons it resolves to nil (not connected / no `ports.audio` in
        // hello / no route) are invisible from the UI — the section just sits
        // dimmed. Say which one it is.
        print("[MABEL-AUDIO] refresh -> \(url?.absoluteString ?? "nil") "
              + "conn=\(connection) port=\(serverMeta?.ports["audio"].map(String.init) ?? "nil") "
              + "route=\(activeRoute?.host ?? "nil")")
        #endif
        voice.setEndpoint(url)
    }

    /// Per-camera pointcloud WS candidates (96xx data plane) for a video
    /// stream key. On a connected route they dial that host; otherwise they
    /// fall back to the robot's known direct hosts (LAN, Tailscale — the
    /// store cycles the candidates until one answers). Only while connected
    /// and only when the server advertises the port in `hello.ports` —
    /// empty otherwise, so the console stays honest.
    func cameraCloudURLs(_ stream: String) -> [URL] {
        guard connection == .connected else { return [] }
        let key: String
        switch stream {
        case "main":        key = "cloudHead"
        case "front":       key = "cloudFront"
        case "wrist_left":  key = "cloudWristLeft"
        case "wrist_right": key = "cloudWristRight"
        default: return []
        }
        guard let port = serverMeta?.ports[key] else { return [] }
        let hosts = activeRoute.map { [$0.host] } ?? (robotStore.active?.routes ?? [])
        return hosts.compactMap { URL(string: "ws://\($0):\(port)") }
    }
    /// Convenience: whether ANY direct path to this stream's cloud exists.
    func cameraCloudURL(_ stream: String) -> URL? { cameraCloudURLs(stream).first }

    /// Voice plane (`:9610`) candidates — the SAME 96xx rule as the clouds above:
    /// DIRECT LAN/Tailscale. Empty when the server doesn't advertise the
    /// port, so the UI can DIM the talk button and say why rather than offering a
    /// live-looking control down which no voice can flow.
    func audioStreamURLs() -> [URL] {
        guard connection == .connected, let port = serverMeta?.ports["audio"] else { return [] }
        // DIRECT LAN/Tailscale.
        let hosts = activeRoute.map { [$0.host] } ?? (robotStore.active?.routes ?? [])
        return hosts.compactMap { URL(string: "ws://\($0):\(port)") }
    }
    /// Is a voice path available at all? (Drives the dim + the note.)
    var audioAvailable: Bool { !audioStreamURLs().isEmpty }

    // MARK: - Head active perception

    /// True when the phone can supply attitude (real device); false in the
    /// Simulator, where the drag-to-look fallback is used instead.
    var motionAvailable: Bool { poseTracker.isAvailable }

    private var gazeYaw = 0.0, gazePitch = 0.0
    private let headScale = 0.85          // robot head moves 15% less

    /// Operator intent: head active perception on/off (toggle button).
    @Published var headEnabled = false

    func toggleHeadTracking() {
        Haptics.select()
        headEnabled.toggle()
        if headEnabled { startHeadTracking() } else { stopHeadTracking() }
    }

    func startHeadTracking() {
        guard headEnabled else { return }
        headTracking = true
        link.send(.headTracking(true))      // tell the server head tracking is ON
        if motionAvailable { poseTracker.start() } else { sendGaze() }
    }
    func stopHeadTracking() {
        let (y0, p0) = poseTracker.current
        poseTracker.stop()
        headTracking = false
        gazeYaw = 0; gazePitch = 0
        // GLIDE the head back to neutral (~1.2 s ease-out) instead of snapping,
        // then release tracking server-side.
        Task { @MainActor [weak self] in
            let steps = 24
            for i in 1...steps {
                guard let self, !self.headTracking else { return }
                let k = 1 - Double(i) / Double(steps)
                self.link.send(.headPose(DevicePoseTracker.headMatrix(
                    yaw: y0 * k * k, pitch: p0 * k * k)))
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
            guard let self, !self.headTracking else { return }
            self.link.send(.headPose([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]))
            self.link.send(.headTracking(false))
        }
    }
    /// Re-center the neck to "straight ahead": zero the gaze, re-zero the device
    /// reference (so the CURRENT phone/headset pose becomes forward), and snap the
    /// neck to center NOW — so a drifted iPhone/Vision Pro view recovers instantly.
    func recenterHead() { Haptics.rigid(); centerNeck() }

    /// Snap the neck to center (no haptic). Shared by recenter + reset.
    private func centerNeck() {
        gazeYaw = 0; gazePitch = 0
        poseTracker.recenter()
        link.send(.headPose([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]))
    }

    /// Drag-to-look fallback (Simulator / no gyro): set the gaze angles.
    func setGaze(yaw: Double, pitch: Double) {
        gazeYaw = max(-1.4, min(1.4, yaw))
        gazePitch = max(-0.9, min(0.9, pitch))
        sendGaze()
    }

    private func sendGaze() {
        // Match the device-motion path's pitch sense (tilt up → look up).
        let yaw = gazeYaw * headScale, pitch = -gazePitch * headScale   // 15% reduction
        let cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch)
        // Ry(yaw) · Rx(pitch), row-major 4×4 head transform.
        let m: [Double] = [ cy,  sy*sp,  sy*cp, 0,
                            0,   cp,    -sp,    0,
                           -sy,  cy*sp,  cy*cp, 0,
                            0,   0,      0,     1]
        link.send(.headPose(m))
    }

    private var latEMA = 0.0
    private func wire(_ l: RobotLink) {
        let safetyError: (String) -> Void = { [weak self] message in
            guard let self else { return }
            self.safetyNotice = message
            self.estopEngaged = true
            self.torqueOnRequested = false; self.torqueRequestAt = nil
            if self.stopState.requested { self.stopState.failed(message) }
        }
        (l as? WebSocketLink)?.onSafetyError = safetyError
        (l as? BLELink)?.onSafetyError = safetyError
        // VP tracking overlay: latch the newest operator_points for the 3D
        // viewer's render tick (main-actor; the overlay hides itself on stale).
        (l as? WebSocketLink)?.onOperatorPoints = { p in
            OperatorPointsLatch.store(p)
        }
        l.onState = { [weak self] s in
            guard let self else { return }
            var s = s
            // Smooth latency so the readout is steady, not jittering each frame.
            self.latEMA = self.latEMA == 0 ? s.latencyMs : 0.85 * self.latEMA + 0.15 * s.latencyMs
            s.latencyMs = self.latEMA

            // Latch the sparse (~5 Hz) LiDAR sweep OUTSIDE the throttled state
            // publish, so no sweep is lost to the decimation below.
            if let scan = s.lidar { self.lidarScan = scan }
            // Same for the ~10 Hz joint dynamics (vel / torque / temps). The
            // sparse keys REPLACE the latch (full snapshot); the motorHealth
            // fallback MERGES (older bridges send partial ~2 Hz frames).
            if let v = s.jointVelocities { self.jointVel = v }
            if let tq = s.jointTorques { self.jointTorque = tq }
            if let tm = s.jointTemps { self.jointTemp = tm }
            if let ht = s.healthTorques { for (k, v) in ht { self.jointTorque[k] = v } }
            if let hm = s.healthTemps { for (k, v) in hm { self.jointTemp[k] = v } }
            // The module roster is CHANGE-GATED on the wire (it only rides a state
            // when something plugs in or falls off), so latch it: an empty decode
            // means "unchanged", never "everything disconnected". Without this the
            // panel would flash dark on every frame that omits it.
            if !s.modules.isEmpty { self.moduleRoster = s.modules }
            s.modules = self.moduleRoster

            // Server-authoritative e-stop (2026-07 echo): reflect the state the
            // ROBOT is in, including a stop pressed by another client. Local
            // taps still flip immediately; the echo reconciles.
            self.ingestSafety(s)

            // 1) HIGH-RATE PATH (every frame, cheap): push the newest pose into the
            //    lock-free `livePose` the 3D render loop reads — the iOS analogue of
            //    the browser's `onmessage → sim.state.q`. No `@Published`, so the
            //    50 Hz stream never invalidates SwiftUI (that backlog was what made
            //    the model lag and the ping RTT ramp without bound).
            // HOLD-LAST: update only the joints present in THIS frame; keep the
            // previous value for any joint missing from a partial/dropped frame.
            // Clearing + rebuilding every frame made a lost hand joint fall back to the
            // home/commanded pose (refreshLivePose overlays lastTelemetry onto
            // jointTargets) for one frame, then jump back — the visible hand "snap to
            // home then return". Holding the last streamed value tracks ROS smoothly
            // through dropouts. Hold-last is only safe because `resetTelemetry()` wipes
            // it the moment the link actually dies (`onStatus`, and `disconnect()`), so
            // a stale pose can never survive a disconnect or carry across a reconnect.
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
            self.refreshLivePose()

            // 2) LOW-RATE PATH (throttled ≈20 Hz): publish chrome (latency, battery,
            //    recording, base steer) to SwiftUI. The 3D model does NOT depend on
            //    this anymore, so a steady cadence here is plenty for the status UI.
            let now = CACurrentMediaTime()
            if now - self.lastStatePublish >= 0.05 {
                self.lastStatePublish = now
                self.state = s
                self.adoptSwitches(from: s)   // keep the toggles honest
            }
        }
        l.onStatus = { [weak self] st in
            guard let self else { return }
            let wasConnected = self.connection == .connected   // a real network link, not demo
            let wasLive = self.connection.isLive
            withAnimation(Theme.Motion.smooth) { self.connection = st }
            if !st.isLive {
                self.safetyConnectionLost()
            }
            // The displayed transport reflects the route that ACTUALLY connected —
            // not the one we optimistically tried — so Tailscale never reads as Wi-Fi.
            if st == .connected, let pr = self.pendingRoute {
                self.activeRoute = pr
                // The live-map endpoint is derived from `activeRoute` (mapStreamURL),
                // which is nil until THIS promotion. applyRoute() called
                // refreshMapEndpoint() earlier while activeRoute was still nil, so the
                // map socket was pointed at nothing and fell back to the demo room.
                // Re-point it now that the real route is live so the nvblox mesh
                // actually streams (ws://host:9092/map).
                self.refreshMapEndpoint()
            }
            // Re-assert the active controller spec whenever a link comes up, so the
            // bridge (sim or real robot) always knows which controller to run — even
            // after a reconnect that reset it to the default.
            if st == .connected && !wasLive {
                if self.stopState.requested { self.retryStopIfNeeded() }
                self.sendControlSpec()
                // VP tracking overlay: opt into the operator_points stream (the
                // gateway only streams while whole-body retargeting runs).
                (self.link as? WebSocketLink)?.setOperatorOverlay(true)
                // Re-assert the chosen task so the server reports its episode count.
                self.link.send(.selectTask(self.recordTask))
                // A route answered — stop cycling the plan (the link now
                // auto-reconnects on THIS route by itself).
                self.failoverTimer?.invalidate(); self.failoverTimer = nil
                self.recoveryTimer?.invalidate(); self.recoveryTimer = nil
                self.simFallbackTimer?.invalidate(); self.simFallbackTimer = nil
                self.autoConnecting = false
                // Single-driver lock: claim it on connect so this operator drives.
                // If another client already holds it the robot replies "denied" and
                // we stay an observer (their input wins until they release it).
                if self.robotStore.active?.kind == .real { self.requestControl() }
            }
            // WE ARE NO LONGER TALKING TO A HOST → show nothing rather than something
            // stale. `dropped()` in the link reports `.connecting` and then retries the
            // same endpoint forever, so without this the app sat in a reconnect loop
            // still rendering the last pose, the last LiDAR sweep and the last
            // pointcloud — data the operator could easily read as live. Covers every
            // way a link dies (socket close, error, the 3.5 s heartbeat watchdog) and
            // leaving `.demo` too. Fresh data repaints it all on reconnect.
            if wasLive && !st.isLive { self.resetTelemetry() }

            // A previously-CONNECTED link just went down. The link itself is already
            // retrying the SAME endpoint — fine for a transient blip, useless if the
            // route vanished (Wi-Fi off). Arm recovery so that, if the
            // same-route reconnect hasn't restored us shortly, we re-race ALL routes
            // and fail over to whichever still answers. (Ignore demo, and drops we caused via
            // disconnect/cancel — those clear autoWanted.)
            if wasConnected && !st.isLive && self.autoWanted { self.scheduleRecovery() }
        }
        l.onServerInfo = { [weak self] server in
            // A first guess from the bare server string; the full meta that follows (hello.robot +
            // hello.hands + hello.kind) is authoritative and settles both the rig and the badge.
            RobotRig.select(robot: nil, server: server, hands: nil)
            // "…sim…" (incl. the on-device demo) → simulation; anything else
            // that greeted us is treated as real hardware.
            let t: ConnectionTarget
            if let s = server?.lowercased() {
                t = s.contains("sim") ? .simulation : .realRobot
            } else {
                t = .unknown
            }
            withAnimation(Theme.Motion.smooth) { self?.target = t }
        }
        l.onServerMeta = { [weak self] meta in
            guard let self else { return }
            self.serverMeta = meta
            // WHICH robot + which hands, from the gateway's own words (hello.robot / .hands):
            // NOVA and MABEL share the wire, and the ORCA hands swap between them.
            RobotRig.select(robot: meta.robot, server: meta.server, hands: meta.hands)
            // SIM vs REAL from `kind`, which every gateway sends. The `server` string only
            // says "sim" for MABEL's sim — NOVA's sim is "nova_teleop_bridge" and was badged REAL.
            let kt: ConnectionTarget? = meta.kind == .sim ? .simulation : meta.kind == .real ? .realRobot : nil
            if let kt, kt != self.target { withAnimation(Theme.Motion.smooth) { self.target = kt } }
            // Mirror the real robot's safety posture for the cockpit/status UI.
            self.securityEnforced = meta.authRequired || meta.armRequired
            self.armed = meta.driver.armed
            self.driverHolder = meta.driver.holder
            if let me = meta.you, let holder = meta.driver.holderId {
                self.isDriver = (holder == me)
            } else if !meta.driver.locked {
                self.isDriver = false
            }
            // Adopt the server-owned name + kind for the active robot so all
            // clients converge on the same name, and remember it.
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
    }

    /// Detach a superseded link's callbacks so its late/async events (a delayed
    /// open, a final error) can never write stale state into the controller.
    private func unwire(_ l: RobotLink) {
        l.onState = nil; l.onStatus = nil; l.onServerInfo = nil
        l.onServerMeta = nil; l.onControl = nil; l.onMapList = nil
        (l as? WebSocketLink)?.onSafetyError = nil
        (l as? BLELink)?.onSafetyError = nil
    }

    // MARK: - Saved maps (mirrors the web console: list_maps / load_map)

    /// Saved maps advertised by the server, and the currently loaded one
    /// (nil = live SLAM). In the on-device demo these are the procedural rooms.
    @Published var savedMaps: [MapInfo] = []
    @Published var loadedMap: String? = nil

    /// Ask the server for its saved-map catalogue (call when the Maps tab appears).
    func requestMapList() { link.send(.listMaps) }

    /// Load a saved map so the robot localizes against it; it then streams on /map.
    func loadMap(_ name: String) { link.send(.loadMap(name: name)) }

    /// Return to building a fresh map live (drops any loaded saved map).
    func goLiveSLAM() { link.send(.loadMap(name: "live")) }

    // MARK: - Robots: switching, naming, speed caps

    /// Connect to a specific remembered robot: race ITS routes (LAN/Tailscale)
    /// fastest-first.
    /// `explicit` is an operator pick (the UI's default); auto-connect passes false
    /// so a robot found on this Wi-Fi may still take over while this one is silent.
    func connect(to robot: Robot, explicit: Bool = true) {
        pinnedRigID = explicit ? robot.id : nil
        if explicit { simFallbackTimer?.invalidate() }
        // The REAL robot is gated by a secret access code (Keychain) carried on
        // every route; the sim is open.
        let code = robot.kind == .real ? SecureStore.realRobotCode : nil
        connectTarget = (robot.id, robot.routes, code)
        robotStore.setActive(id: robot.id)
        resetDriverState()
        // Switching robots (sim ↔ real) must not carry the PREVIOUS robot's data into
        // the new one. `resetDriverState()` clears the joints but not the odom pose,
        // heading, or the map — so the sim's mesh and its base position used to linger
        // over the real robot until it happened to overwrite them.
        resetTelemetry()
        startRace()
    }

    // MARK: - Driver lock + arm (real robot safety)

    /// Whether WE hold the single-driver lock on the connected robot.
    @Published private(set) var isDriver = false
    /// Name of whoever holds the lock (may be another operator), "" if free.
    @Published private(set) var driverHolder = ""
    /// The real robot's live driver/arm posture, mirrored from `serverMeta`.
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
        if let r = robotStore.active, r.kind == .real { connect(to: r, explicit: pinnedRigID == r.id) }
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
        jointVel = [:]; jointTorque = [:]; jointTemp = [:]   // ditto the dynamics latch
        // A latched roster on a DEAD link would claim the arms are still
        // attached to a robot we cannot even reach. Offline = unknown, not stale.
        moduleRoster = []
        // Same for the video gates: a stale `fpvLive` on a dead link would keep the
        // 1st-person overlay up over a frozen last frame.
        fpvLive = false; fpvVisible = false
        // A robot switch must not carry the previous robot's pose: hold-last
        // telemetry would keep the OLD robot's joints on screen until the new
        // stream happens to cover them. Back to home; the new robot repaints.
        lastTelemetry = [:]
        liftAdopted = false        // unknown column again: no lift command until it re-adopts
        jointTargets = RobotModel.homePose
        refreshLivePose()
    }

    /// NO HOST → NO DATA. Drop everything the robot ever told us, so the app falls back
    /// to its default (home) state instead of freezing on the last frame it received.
    ///
    /// This is the function `onState`'s hold-last comment always claimed existed — it did
    /// not, which is why a disconnect left a stale pose, a stale LiDAR sweep, and a stale
    /// pointcloud on screen. Hold-last is right *through* a dropout (a joint missing from
    /// one frame keeps its value instead of snapping home) and wrong the moment the link
    /// is actually gone; this is the other half of that contract.
    ///
    /// Deliberately does NOT touch routing (`activeRoute`, `pendingRoute`, `videoBase`,
    /// `serverMeta`): the link reconnects on its own endpoint and
    /// restores `activeRoute` from `pendingRoute`, so clearing those here would strand
    /// video and the map socket after a transient blip. Video needs no clearing anyway —
    /// every URL builder is gated on `videoAllowed` (`connection == .connected`), so the
    /// tiles and the Cam3D planes go to their placeholder on their own.
    /// `jointTargets` is the OPERATOR's command, not robot data, so it is left alone.
    private func resetTelemetry() {
        // `refreshLivePose()` reads `state.baseX/baseY`, so the rig stays parked at the
        // last odom pose unless `state` itself goes back to its default.
        state = RobotState()
        // The `state` publish is throttled to ~20 Hz; without clearing the throttle the
        // FIRST frame after a reconnect can be swallowed, holding the default pose on
        // screen for an extra beat. A repaint after going offline is never "too soon".
        lastStatePublish = 0
        lidarScan = nil
        jointVel = [:]; jointTorque = [:]; jointTemp = [:]
        moduleRoster = []
        lastTelemetry = [:]
        liftAdopted = false        // unknown column again: no lift command until it re-adopts
        lastHeading = 0; lastQuat = [1, 0, 0, 0]
        refreshLivePose()               // rig falls back to the commanded/default pose
        liveMap.goOffline()             // mesh, occupancy grid, route, LiDAR + cam clouds
    }

    /// Rename a robot. Updates locally + (if it's the one we're driving) pushes the
    /// name to its server, which persists and syncs it to every client.
    func renameRobot(id: String, to name: String) {
        if robotStore.rename(id: id, to: name), connection == .connected {
            link.send(.setRobotName(name))
        }
    }

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

    /// The Simulation Control Center client (scene library / switch / MuJoCo
    /// viewer) — available only when the SCC backend runs beside a sim on a
    /// direct link. Reconfigured whenever the connection target changes.
    let simStudio = SimStudio()

    func syncSimStudio() {
        simStudio.configure(host: videoBase?.host, isSimulation: target.isSimulation)
    }

    // MARK: - Physics view (MuJoCo takeover) — state lives HERE because the
    // Robot space view is re-created on rotation (different landscape/portrait
    // hosts) and @State would drop the mode mid-flip.
    @Published var physicsViewOn = false
    let physicsViewCam = PhysicsCam()
    /// Bumped on every sim reset: the viewers re-frame onto the robot (3rd
    /// person recenters, the physics cam refits its orbit).
    @Published private(set) var refitToken = 0

    /// Sim only: relocate-reset to a server-validated random open pose.
    func resetSimRandom() {
        guard canReset else { return }
        Haptics.rigid()
        snapModelToHome()
        centerNeck()
        link.send(.resetSimRandom)
        physicsViewCam.refit()
        refitToken += 1              // viewers re-frame onto the robot
    }

    /// Reset the simulation to its initial start environment.
    /// Restart SLAM: the server drops its map and begins a FRESH one anchored
    /// at the robot's current pose (localization resets there too). The app
    /// wipes its rendered map immediately; the new one streams in.
    func resetSlam() {
        Haptics.rigid()
        link.send(.resetSlam)
        liveMap.restartMapping()
    }

    func resetSim() {
        guard canReset else { return }
        Haptics.rigid()
        snapModelToHome()
        centerNeck()                 // neck back to straight-ahead on reset
        link.send(.resetSim)
        physicsViewCam.refit()
        refitToken += 1              // viewers re-frame onto the robot
    }

    /// Reset EVERY desired pose locally and snap the 3D model to home AT ONCE — don't
    /// wait for the server's home telemetry to round-trip (that lag, plus the stale
    /// pre-reset telemetry still in `lastTelemetry`, is what made the model "reset then
    /// snap back to the old pose"). Clearing `lastTelemetry` means `refreshLivePose`
    /// renders the commanded home immediately; fresh (home) telemetry then confirms it.
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

    /// Connect to a real MABEL teleop server (start_teleop.sh on the Mac).
    /// An empty / "demo" host falls back to the on-device simulator. Manual —
    /// cancels any in-flight auto-connect race.
    func connect(host: String) {
        cancelAuto()
        overBluetooth = false
        unwire(link); link.disconnect()
        let trimmed = host.trimmingCharacters(in: .whitespaces)
        serverMeta = nil
        if trimmed.isEmpty || trimmed.lowercased() == "demo" {
            videoBase = nil; activeRoute = nil; pendingRoute = nil
            robotStore.setActive(id: nil)
            target = .simulation
            let l = DemoLink(); link = l; wire(l); l.connect()
        } else {
            // If this host belongs to the gated real rig, carry its access code.
            let realHost = robotStore.robots.first { $0.kind == .real }?.routes.contains(trimmed) ?? false
            let code = realHost ? SecureStore.realRobotCode : nil
            let l = WebSocketLink(host: trimmed, accessCode: code)
            videoBase = l.videoBase
            // Promoted to activeRoute when the link connects.
            activeRoute = nil
            pendingRoute = ConnectionRoute(kind: NetworkConfig.classify(trimmed), host: trimmed)
            robotStore.setActive(id: trimmed)
            target = .unknown          // learned from the server's `hello`
            link = l; wire(l); l.connect()
        }
        refreshMapEndpoint()
    }

    // MARK: - Bluetooth (teleop with no network at all)

    /// True while the live link is Bluetooth. The UI uses it to hide what BLE genuinely
    /// cannot carry — camera feeds, maps, recordings — instead of showing controls that
    /// would silently do nothing.
    @Published private(set) var overBluetooth = false

    /// Connect to a robot found by BLE proximity (the connect card's one button).
    ///
    /// `done` reports whether the robot's GATT service actually came up, so the card can
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
        unwire(link); link.disconnect()
        serverMeta = nil
        videoBase = nil      // no video on this link, by design
        activeRoute = nil
        pendingRoute = ConnectionRoute(kind: .bluetooth, host: name)
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
        l.onStatus = { [weak self] status in
            wired?(status)
            guard !answered, status == .connected else { return }
            answered = true
            // Remembered only on a REAL connection, so a failed attempt never leaves a
            // robot in Settings that was never reachable.
            KnownRobots.shared.remember(id: id, advertisedName: name)
            self?.activeRoute = ConnectionRoute(kind: .bluetooth, host: name)
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
        cancelAuto()
        connectTarget = nil
        pinnedRigID = nil
        unwire(link); link.disconnect()
        videoBase = nil; activeRoute = nil; pendingRoute = nil
        robotStore.setActive(id: nil); serverMeta = nil
        // We unwired the link, so its status callback won't fire — reflect the
        // disconnect ourselves so the status bar shows the red "offline" icon and
        // the Dynamic Island (gated on `.connected`) tears down. No demo fallback.
        withAnimation(Theme.Motion.smooth) { connection = .disconnected }
        safetyConnectionLost()
        target = .unknown
        resetDriverState()
        // We unwired the link, so `onStatus` (which normally does this on a drop)
        // will NOT fire — clear the robot's data here too, or an explicit disconnect
        // leaves the pose/LiDAR/pointcloud frozen on screen.
        resetTelemetry()
        refreshMapEndpoint()
    }

    // MARK: - Auto-connect: race the active robot's own routes

    /// Connect automatically at launch — the REAL robot is always tried FIRST
    /// (it's the primary rig when it's on the network); if it hasn't answered
    /// after a grace window, the race retargets the simulation. Demo remains
    /// the terminal fallback while nothing is reachable.
    func connectAuto(discovered: [RobotStore.DiscoveredRobot] = []) {
        if !discovered.isEmpty {
            robotStore.mergeDiscovered(named: discovered); lastDiscovered = discovered.map(\.host)
        }
        let sim = robotStore.robots.first { $0.kind == .sim }
        // A robot answering on THIS Wi-Fi right now (NOVA, a lab rig) beats any
        // remembered one; it already took a Bonjour resolve, so no sim fallback.
        if let here = robotStore.discoveredOnline(discovered).first {
            connect(to: here, explicit: false)
            return
        }
        let real = robotStore.robots.first { $0.kind == .real }
        let robot = real ?? robotStore.active ?? sim ?? robotStore.robots.first
        if let robot {
            connect(to: robot, explicit: false)
            if robot.kind == .real, let sim { scheduleSimFallback(sim) }
        } else if !connection.isLive {
            // No robot on the network at all — Bluetooth before the on-device demo, since a
            // real robot within arm's reach beats a simulated one every time.
            considerBluetoothFallback(hadNetwork: false)
            if !overBluetooth { fallbackToDemo() }
        }
    }

    /// The real rig didn't answer in time → aim the race at the sim.
    /// Cancelled the moment any link actually connects (or on manual action).
    /// One GRACE extension while the race is still actively cycling routes —
    /// the LAN probes time out first and the later routes start last, so
    /// firing at exactly 8 s used to kill a handshake mid-flight.
    private func scheduleSimFallback(_ sim: Robot, graceLeft: Int = 1) {
        simFallbackTimer?.invalidate()
        simFallbackTimer = Timer.scheduledTimer(withTimeInterval: 8, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.autoWanted, self.connection != .connected else { return }
                if graceLeft > 0, self.autoConnecting {
                    self.scheduleSimFallback(sim, graceLeft: graceLeft - 1)
                } else {
                    self.connect(to: sim, explicit: false)
                }
            }
        }
    }
    private var simFallbackTimer: Timer?

    /// Fresh Bonjour hits — fold them into the right robot, and if we're mid-race
    /// for a robot those hosts belong to (and not yet connected), re-race so a
    /// just-appeared LAN host can win.
    func noteDiscovered(_ found: [RobotStore.DiscoveredRobot]) {
        let hosts = found.map(\.host)
        lastDiscovered = hosts
        upgradeFromBluetoothIfPossible(hosts)
        robotStore.mergeDiscovered(named: found)
        guard autoWanted else { return }
        // Already on a real link — leave it (a LAN route is the best we can do).
        if connection.isLive { return }
        // The rig auto-connect is racing is NOT answering, and a robot on this
        // Wi-Fi just resolved: go to it. Discovery used to fold every unknown host
        // (NOVA) into the simulation, so launch raced thor → sim forever while
        // NOVA sat on the same network. Never away from an operator's pick.
        if pinnedRigID == nil, connection != .connected,
           let here = robotStore.discoveredOnline(found).first, here.id != connectTarget?.id {
            simFallbackTimer?.invalidate()
            connect(to: here, explicit: false)
            return
        }
        // We're on the on-device demo (the live map shows the synthetic room): a
        // real server just appeared on the LAN, so RACE it now. This is the fix for
        // "stuck on demo forever" — the initial race finished before Bonjour
        // resolved the host, and nothing re-raced it. Skip while a race is already
        // in flight so we don't thrash the link.
        guard !autoConnecting, !hosts.isEmpty else { return }
        let robot = connectTarget.flatMap { robotStore.robot(id: $0.id) }
            ?? robotStore.robots.first { $0.kind == .sim }
            ?? robotStore.robots.first
        if let robot { connect(to: robot, explicit: pinnedRigID == robot.id) }
    }

    /// Race the current target robot's routes; bind the fastest.
    private func startRace() {
        autoWanted = true
        guard let t = connectTarget else { return }
        autoConnecting = true
        serverMeta = nil
        failoverTimer?.invalidate(); failoverTimer = nil
        recoveryTimer?.invalidate(); recoveryTimer = nil
        Task { @MainActor in
            let routes = await planner.plan(hosts: t.hosts)
            guard autoWanted, connectTarget?.id == t.id else { return }
            routePlan = routes
            routeIdx = 0
            guard !routes.isEmpty else {                // nothing reachable
                autoConnecting = false
                if !connection.isLive { fallbackToDemo() }
                return
            }
            applyRoute(routes[0])
            failoverTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
                Task { @MainActor in self?.failoverTick() }
            }
        }
    }

    /// Bring up the link for a chosen route (direct WebSocket).
    private func applyRoute(_ route: ConnectionRoute) {
        pendingRoute = route            // promoted to activeRoute when this link connects
        overBluetooth = false
        serverMeta = nil
        unwire(link)                    // a superseded link must not report a stale route
        link.disconnect()
        target = .unknown
        let l = WebSocketLink(host: route.host, accessCode: connectTarget?.accessCode)
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
            startRace()
            return
        }
        applyRoute(routePlan[routeIdx])
    }

    /// A live link dropped — arm a one-shot re-race. The grace window lets the
    /// link's own same-route reconnect ride out a transient blip (a tunnel flap
    /// recovers in well under a second); only a route that's genuinely gone still
    /// has us disconnected when this fires, and then re-racing picks a live route.
    // MARK: - Transport handover (the policy lives in LinkPriority)

    /// Set when a NETWORK link drops and a robot is in Bluetooth range. The UI asks before
    /// taking it: moving DOWN a tier silently removes video and most of the protocol, and
    /// the phone is usually still nominally on Wi-Fi when it is the ROBOT that has gone.
    /// Moving UP needs no such ceremony, which is why there is no matching flag for Wi-Fi.
    @Published var bluetoothFallback: NearbyRobot?

    /// Nothing on the network. Bluetooth is the last rung: take it outright on a FIRST
    /// connection (there is nothing to lose), ask first if we are falling back from a
    /// network link that just died.
    func considerBluetoothFallback(hadNetwork: Bool) {
        guard !connection.isLive, bluetoothFallback == nil,
              let ble = BLECentral.shared.promoted else { return }
        if hadNetwork {
            bluetoothFallback = ble
        } else {
            connectBluetooth(ble) { _ in }
        }
    }

    /// The operator said yes to the Bluetooth card.
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
        if let robot { connect(to: robot, explicit: pinnedRigID == robot.id) }     // races LAN → Tailscale, best first
    }

    private func scheduleRecovery() {
        guard recoveryTimer == nil else { return }      // one in flight is enough
        recoveryTimer = Timer.scheduledTimer(withTimeInterval: 4, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.recoveryTick() }
        }
    }

    /// Grace window elapsed. If the same-route reconnect already brought us back,
    /// do nothing; otherwise re-race every route (the dead one fails its probe, so
    /// a live one wins) — this is what carries a Wi-Fi drop over to Tailscale.
    private func recoveryTick() {
        recoveryTimer?.invalidate(); recoveryTimer = nil
        guard autoWanted, connection != .connected else { return }
        startRace()
        // If the race finds nothing either, Bluetooth is what is left. Ask, because this is
        // a fall BACK from a network link the operator was already using.
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            self?.considerBluetoothFallback(hadNetwork: true)
        }
    }

    /// Stop auto-connecting (manual connect / disconnect).
    private func cancelAuto() {
        autoWanted = false
        autoConnecting = false
        failoverTimer?.invalidate(); failoverTimer = nil
        recoveryTimer?.invalidate(); recoveryTimer = nil
        simFallbackTimer?.invalidate(); simFallbackTimer = nil
    }

    /// No real route was reachable — keep the app usable on the on-device sim.
    private func fallbackToDemo() {
        videoBase = nil; activeRoute = nil; pendingRoute = nil
        overBluetooth = false
        unwire(link)
        target = .simulation
        let l = DemoLink(); link = l; wire(l); l.connect()
        refreshMapEndpoint()
    }

    // MARK: - Episode recording (data collection)

    @Published private(set) var recording = false
    /// Task folder the operator is collecting into. Episodes are saved under
    /// `<episodes_dir>/<recordTask>/episode_NNNN_<timestamp>/` (server-side,
    /// never overwritten). Defaults until the operator picks/creates a task.
    @Published var recordTask = "default"

    /// Episodes already on the server for the current task (live from telemetry).
    var episodeCount: Int { state.episodeCount }
    /// Existing task folders on the server, for the picker.
    var serverTasks: [String] { state.taskList }
    /// Episodes per task folder, for the Settings collection tracker.
    var taskCounts: [String: Int] { state.taskCounts }
    /// Total episodes collected across all tasks.
    var episodesTotal: Int { state.episodesTotal }
    /// Where episodes are written on the server (sim/robot host).
    var episodesDir: String { state.episodesDir }

    /// Choose / create a task folder. Sends a no-op `select` so the server starts
    /// reporting that task's episode count immediately (before any recording).
    func selectTask(_ task: String) {
        let t = task.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        recordTask = t
        link.send(.selectTask(t))
    }

    /// Ask the server to recompute + rebroadcast its recording status (per-task
    /// counts), e.g. when opening Settings. Re-selecting the current task is a
    /// no-op for collection but forces a fresh status.
    func refreshCollectionStatus() {
        guard connection.isLive else { return }
        link.send(.selectTask(recordTask))
    }

    func startEpisode() { recording = true; link.send(.record(active: true, task: recordTask)) }
    func endEpisode()   { recording = false; link.send(.record(active: false, task: recordTask)) }

    // MARK: - Animation demos (animation_demo/)
    //
    // Every piece of state here is READ BACK from `state.animation`, never
    // latched locally. A local "isRecording" flag would disagree with the robot
    // the moment a countdown expires, a clip is refused, or a second operator
    // stops the take — and on demo day the screen has to be right.

    /// `state.animation` is OPTIONAL on the wire (absent until the server sends
    /// its first block), and every call site here wants "the current truth, or
    /// nothing happening". Defaulting once keeps that `?? .init()` out of the
    /// views, where a missed unwrap would read as "not recording" anyway.
    var demo: RobotState.AnimationEcho { state.animation ?? .init() }
    var demoClips: [RobotState.DemoClipInfo] {
        demoClipList.isEmpty ? demo.clips : demoClipList
    }
    /// Anything demo-related in flight. Gates the menu so a replay cannot be
    /// started on top of a recording (the server refuses too; this stops the
    /// operator asking).
    var demoBusy: Bool { demo.busy }

    /// Arm a demo take. Capture begins `countdown` seconds later, on the SERVER's
    /// clock — the operator's time to get into the start position.
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
    /// Winds the player down through its return-home ramp. NOT a freeze: leaving
    /// the arms wherever the clip happened to be is not a safe resting state.
    func stopDemo() { link.send(.demoReplay(active: false, name: "", repeats: 1)) }
    func deleteDemo(_ name: String) { link.send(.demoDelete(name)); refreshDemoClips() }
    /// Clips from the `demo_clips` REPLY. Kept separate from the
    /// `robot_state.animation` mirror because the two refresh on different
    /// events: the reply lands the moment the picker asks, the mirror only when
    /// the server has a reason to restate it. Whichever is fresher wins.
    @Published private(set) var demoClipList: [RobotState.DemoClipInfo] = []

    func refreshDemoClips() { link.send(.demoList) }

    // MARK: - Full-bus data collection (rosbag → Rerun)
    /// Toggle a server-side rosbag (MCAP) recording of the WHOLE ROS bus — every
    /// joint/command/camera/depth/cloud/mesh/map/odom/tf topic — for later
    /// long-horizon nav+manip use. Bags are saved locally on the server host
    /// (never pushed) under its `recordings/` dir; convert with
    /// `server/bag_to_rerun.py`. Independent of the per-episode `recording` above.
    func startDataCollection(label: String? = nil) {
        guard connection.isLive else { return }
        collectingData = true
        link.send(.dataCollect(active: true, label: label ?? recordTask))
    }
    func stopDataCollection() {
        collectingData = false
        link.send(.dataCollect(active: false, label: recordTask))
    }
    func toggleDataCollection() {
        collectingData ? stopDataCollection() : startDataCollection()
    }

    // MARK: - Base driving

    /// Drive from two normalized joystick vectors, each component in [-1, 1].
    /// Left stick → translation, right stick x → yaw.
    /// One 30 Hz drive tick (RobotSpaceView's tick loop): send the RAW stick
    /// targets. The motion model lives SERVER-SIDE and is already the exact
    /// keyboard-teleop tip-safe model — `TeleopEngine._navigate` does stick ×
    /// ceiling → `mabel.config.safe_base_step` (per-axis speed/accel caps that
    /// SHRINK with lift height + the ZMP support-polygon clamp) → swerve IK.
    /// Shaping client-side too would double-smooth and mask the safety model,
    /// so the app stays a thin stick: targets in, the robot ramps tip-safely.
    func driveTick(translate: CGVector, yaw: Double, dt: Double) {
        guard !estopEngaged else { return }
        link.send(.twist(forward: -translate.dy * maxLinear * speedScale,
                          strafe: translate.dx * maxLinear * speedScale,
                          yaw: yaw * maxAngular * speedScale))
    }

    /// Stop EVERY axis the nav frame carries — including the lift.
    ///
    /// `.twist` writes `nav.lx/ly/rx` only. The lift rides `nav.ry`, and `nav` is
    /// PERSISTENT link state streamed at 20–60 Hz, so an axis nobody rewrites
    /// keeps commanding its last value indefinitely. A lift nudge followed by
    /// "stop" therefore left the column rising on its own, with the sticks
    /// centred and nothing on screen suggesting a command was still in flight
    /// (reported 2026-07-29: "the lift will move up by itself").
    func stopBase() {
        link.send(.twist(forward: 0, strafe: 0, yaw: 0))
        link.send(.liftRate(0))
    }

    /// Drive the base from a physical body twist (m/s, m/s, rad/s). Used by
    /// the map's path follower so autonomous navigation moves the sim/real
    /// robot over the same bridge the Drive joysticks use. `forward` is +front,
    /// `strafe` is +right, `yaw` is +clockwise — matching `drive(translate:yaw:)`.
    func driveTwist(forward: Double, strafe: Double, yaw: Double) {
        guard !estopEngaged else { return }
        link.send(.twist(forward: forward * speedScale,
                          strafe: strafe * speedScale,
                          yaw: yaw * speedScale))
    }

    /// Send a Nav2 navigation goal (the tapped point on the LIVE map, in the
    /// `odom`/mesh frame — metres + heading rad). The server forwards it to ROS
    /// `/goal_pose` (NavFn A* + MPPI); Nav2 drives the base via `/mabel_cmd`
    /// (operator stick overrides), and the planned route streams back on the /map
    /// socket as a PATH frame, drawn on the live map. The robot's TRUE path
    /// planning + execution happen in ROS — the app is the live RViz, not a
    /// local planner.
    func sendNavGoal(x: Double, y: Double, yaw: Double) {
        guard !estopEngaged else { return }
        link.send(.navGoal(x: x, y: y, yaw: yaw))
    }

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
        var jt = jointTargets
        jt[id] = value
        Self.echoGripper(&jt, id, value)
        jointTargets = jt
        refreshLivePose()                                  // instant model echo
        guard liftAllowed(id) else { return }
        link.send(.jointTarget(id: id, value: value))
    }


    /// BATCHED joint command for HIGH-RATE multi-joint sources — the green-ball IK
    /// (≈60 Hz × the whole arm chain) and the arm joysticks (30 Hz × 6 joints).
    /// Critically this is ONE `jointTargets` mutation + ONE `refreshLivePose()` for
    /// the whole batch, NOT one per joint. Per-joint `setJoint()` in those loops
    /// fired an `objectWillChange` storm (240–420/s) that saturated the main thread
    /// shared with the WebSocket send/receive + pong path → the "super severe" lag.
    /// Same fix as `setGrip`. The visual echo rides the decoupled `livePose`; the
    /// targets stream as `joint_command`, coalesced on the wire (latest-wins), so the
    /// app stays a lightweight command device and the robot's WBC does the control.
    func setJoints(_ updates: [String: Double]) {
        guard !updates.isEmpty else { return }
        var jt = jointTargets
        for (id, v) in updates { jt[id] = v; Self.echoGripper(&jt, id, v) }
        jointTargets = jt                                  // ONE invalidation for the batch
        refreshLivePose()                                  // ONE rebuild
        for (id, v) in updates where liftAllowed(id) { link.send(.jointTarget(id: id, value: v)) }
    }

    /// Open/close a whole hand (0 = open, 1 = closed fist): curls the flexion
    /// joints on the model AND sends the grip to the robot (it runs apply_grasp).
    static let flexSuffixes = ["thumb_pip", "thumb_dip", "index_mcp", "index_pip",
                               "middle_mcp", "middle_pip", "ring_mcp", "ring_pip",
                               "pinky_mcp", "pinky_pip"]
    func setGrip(_ side: String, _ g: Double) {
        // Batch ALL the finger flexions into ONE @Published mutation. Assigning
        // jointTargets[id] in a loop fired objectWillChange 11x per slider tick → a
        // SwiftUI re-render storm on the main thread that starved the ping/pong path
        // and spiked the latency readout while dragging the grip. One assignment = one
        // invalidation; the visual hand echo rides the decoupled livePose, not SwiftUI.
        var jt = jointTargets
        for suf in Self.flexSuffixes {
            let id = "\(side)_\(suf)"
            if let spec = RobotModel.spec(id) { jt[id] = g * spec.upper }   // 0=extended, upper=flexed
        }
        jt["\(side)_grip"] = g
        Self.echoGripper(&jt, "\(side)_grip", g)          // a gripper side: the opening the gateway will drive
        jointTargets = jt                                  // ONE invalidation
        refreshLivePose()                                  // instant hand echo (no SwiftUI)
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

    /// The three orthogonal control axes (mirrors the bridge's wire protocol):
    /// teleop method, controller type, and region. The robot models live
    /// server-side; the app only declares intent.
    @Published private(set) var teleopMethod: TeleopMethod = .navigation
    @Published private(set) var controllerType: ControllerType = .impedance
    @Published private(set) var region: ControlRegion = .arm
    @Published var stiffness: Double = 0.5            // 0 = limp … 1 = rigid
    static let maxStiffness = 45.0                    // Nm/rad at slider = 1

    /// Upper-body gravity feed-forward. Defaults ON: the robot holding its own
    /// arms up is the normal state, and OFF exists to show what the layer buys
    /// (or to hunt droop), not as a mode anyone drives in.
    @Published private(set) var gravityComp = true
    /// Which retargeting the arms run. `false` (default) is the front-workspace
    /// profile that prioritises WRIST position and palm direction; `true` is the
    /// original whole-body map that follows the operator's whole arm.
    @Published private(set) var wholeBodyRetarget = false

    /// One-line summary for status displays, e.g. "Wrist · Hold · Arm".
    var controlSummary: String {
        teleopMethod == .navigation ? "Navigation"
            : "\(teleopMethod.rawValue.capitalized) · \(controllerType.title) · \(region.title)"
    }

    /// Declare the full control spec to the bridge (method × type × region).
    func setControl(method: TeleopMethod, type: ControllerType, region: ControlRegion) {
        teleopMethod = method; controllerType = type; self.region = region
        sendControlSpec()
    }

    /// Convenience: macro driving (base + lift), used by the Drive cockpit.
    func setNavigation() { setControl(method: .navigation, type: .impedance, region: .wholeBody) }

    /// Re-send the current spec with an updated impedance stiffness.
    func setStiffness(_ k: Double) {
        stiffness = min(1, max(0, k))
        sendControlSpec()
    }

    /// Adopt the server's echoed switch state. Called on every state tick; a
    /// no-op unless the robot disagrees with us, so it never fights the operator
    /// mid-tap — it only corrects a stale UI (reconnect, or a second driver).
    func adoptSwitches(from st: RobotState) {
        // Only adopt what the robot actually reported. A `nil` means this state
        // carried no opinion (older server, or the offline demo), and adopting a
        // default there would fight the operator: their tap flipped the toggle,
        // the next tick reverted it, and the switch looked broken.
        if let g = st.gravityComp, g != gravityComp { gravityComp = g }
        if let w = st.wholeBodyRetarget, w != wholeBodyRetarget { wholeBodyRetarget = w }
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

    private func sendControlSpec() {
        link.send(.controlSpec(method: teleopMethod.rawValue, type: controllerType.rawValue,
                               region: region.rawValue, stiffness: stiffness * Self.maxStiffness))
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
    /// The glide runs `homeMinDuration` unless a joint would exceed the cap, in which
    /// case it stretches so the motion always stays gentle.
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
    /// restarts the glide from wherever the arms currently are.
    func goHome() {
        homeTimer?.invalidate()
        // Enable joint control so the pose is honoured (a Home pressed outside teleop
        // would otherwise be ignored while the robot is in navigation mode).
        setControl(method: .joint, type: .impedance, region: .wholeBody)
        // THE COLUMN IS NOT GLIDED. It is a BANG-BANG servo whose only speeds are ~37 mm/s
        // and zero, so a streamed trajectory asking for anything slower is tracked by
        // stop-starting — 54 stops on a full-stroke Home, each a fresh full-duty pulse,
        // which is what "the lift moves in jittery motion" is (measured in NOVA's
        // harness/lift_home_sim.py: 124 stops streamed vs 4 one-shot). It gets ONE
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

    /// Abort an in-flight Home glide (e.g. the operator grabs a control) — leaves the
    /// arms wherever they are and hands the model back to live telemetry.
    func cancelHome() {
        guard homeTimer != nil else { return }
        homeTimer?.invalidate(); homeTimer = nil
        homing = false
    }

    /// Push a robot body in a world-frame direction (drag anywhere on the 3D
    /// model). Empty dict = release. The bridge's WBC engine admits the wrench.
    func applyExternalForce(_ forces: [String: [Double]]) {
        link.send(.externalForce(forces))
    }
    func clearExternalForce() { link.send(.externalForce([:])) }

    /// iOS arm-joint group → the MuJoCo body that receives the push. Coarser
    /// than the 7 joints (the model lumps the arm into mount/forearm/wrist/hand).
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

    /// Absolute lift POSITION from a 0…1 slider.
    ///
    /// Sends joint TARGETS, not `.liftFraction`. That command maps the fraction
    /// onto the right stick's `ry` axis (`nav.ry = frac * 2 − 1`), and `ry` is
    /// integrated by the controller as a RATE
    /// (`teleop_controller._drive_lift(ry * NAV_MAX_LIFT, dt)`). So a slider
    /// parked anywhere except exactly 0.5 commanded the lift to keep moving in
    /// that direction FOREVER — a position control wired to a velocity input.
    /// Full up meant "rise at full speed, indefinitely".
    func setLift(fraction: Double) {
        setLiftHeight(min(1, max(0, fraction)) * Self.liftMax)
    }

    /// Total lift-column height in metres (both cascaded stages summed).
    static let liftStageMax = 0.3175
    static let liftMax = liftStageMax * 2          // 0.635 m total travel
    var liftHeight: Double { (jointTargets["lift_lower"] ?? 0) + (jointTargets["lift_upper"] ?? 0) }

    /// Command the lift as ONE actuator: split the requested total height evenly
    /// across the two cascaded prismatic stages so they move together (mirrors
    /// the website's `_setLift`, which sets each stage to v/2). Sent as joint
    /// targets the bridge sums onto the single `Lift` actuator.
    func setLiftHeight(_ meters: Double) {
        let half = min(Self.liftStageMax, max(0, meters / 2))
        jointTargets["lift_lower"] = half
        jointTargets["lift_upper"] = half
        // A DELIBERATE move — the slider, the jog, Home. It is also the moment the
        // operator takes ownership of the column, so it ends the adoption wait even if no
        // telemetry ever named the lift (a plant that reports no column cannot be moved by
        // a stale default either way).
        liftAdopted = true
        link.send(.jointTarget(id: "lift_lower", value: half))
        link.send(.jointTarget(id: "lift_upper", value: half))
    }

    /// Lift velocity from the right stick's Y axis, −1 (down) … +1 (up).
    func setLiftRate(_ rate: Double) {
        guard !estopEngaged else { return }
        // NOT scaled by speedScale: the Speed Governor is for the mobile base only, the lift always
        // runs full speed (the robot maps any push to full duty anyway — NOVA server/plant.py).
        link.send(.liftRate(rate))
    }

    var liftFraction: Double {
        let total = (state.joints["lift_lower"] ?? 0) + (state.joints["lift_upper"] ?? 0)
        return min(1, max(0, total / (0.3175 * 2)))
    }

    // MARK: - Policies

    /// The policy currently dispatched to the robot (drives the Policies UI and
    /// the Dynamic Island). nil = nothing running. Lifted here from the view so
    /// the Live Activity can see it.
    @Published var runningPolicy: Policy? = nil
    /// When the running policy was dispatched — drives the Dynamic Island ETA
    /// (estimated against the policy's time horizon).
    @Published private(set) var runningPolicySince: Date? = nil

    func dispatch(_ policy: Policy) {
        Haptics.rigid()
        runningPolicy = policy
        runningPolicySince = Date()
        link.send(.dispatchPolicy(id: policy.id))
    }

    /// Stop / clear the running policy.
    func stopPolicy() {
        runningPolicy = nil
        runningPolicySince = nil
        Haptics.soft()
    }

    // MARK: - Autonomous navigation progress (Dynamic Island hook)

    /// Live autonomous-navigation progress (0…1) + goal label, surfaced on the
    /// Dynamic Island. Set by a path follower when one is running; nil = not
    /// navigating. No placeholder values — stays nil until a real follower reports.
    @Published private(set) var navProgress: Double? = nil
    @Published private(set) var navGoal: String = ""

    func setNavProgress(_ progress: Double?, goal: String = "") {
        navProgress = progress.map { min(1, max(0, $0)) }
        navGoal = goal
    }

    // MARK: - Push registration

    /// Hand this device's APNs token to the connected server so it can notify the
    /// operator (faults, task completion, lost link) while the app is asleep.
    func registerPushToken(_ token: String) {
        link.send(.registerPush(token: token))
    }

    // MARK: - Safety

    private func safetyConnectionLost() {
        stopState.disconnected()
        stopRetryTimer?.invalidate(); stopRetryTimer = nil
        if torqueOnRequested {
            torqueOnRequested = false; torqueRequestAt = nil
            estopEngaged = true
            safetyNotice = "Torque request unconfirmed after connection loss"
        }
    }

    func engageEStop() {
        estopEngaged = true
        torqueOnRequested = false; torqueRequestAt = nil
        stopState.request(now: Date().timeIntervalSince1970)
        safetyNotice = nil
        retryStopIfNeeded(force: true)
        startStopRetry()
    }

    func releaseEStop() {
        guard connection.isLive else {
            estopEngaged = true
            torqueOnRequested = false; torqueRequestAt = nil
            safetyNotice = "No robot connection. Reconnect and tap Turn torque on again."
            return
        }
        stopState.resume()
        safetyNotice = nil
        stopRetryTimer?.invalidate(); stopRetryTimer = nil
        torqueOnRequested = true; torqueRequestAt = Date().timeIntervalSince1970
        estopEngaged = true             // wait for the controller's release echo
        link.send(.estop(false))
    }

    var stopStatusLabel: String {
        if !connection.isLive { return stopState.requested ? "Stop unconfirmed" : "No robot connection" }
        if torqueOnRequested {
            return Date().timeIntervalSince1970 - (torqueRequestAt ?? 0) < 1.5
                ? "Torque request pending" : "Torque request unconfirmed"
        }
        if safetyNotice != nil && !stopState.requested { return "Request unconfirmed" }
        if stopState.requested || stopState.confirmed { return stopState.label(now: Date().timeIntervalSince1970) }
        if state.estop == false {
            if state.torqueState["any"] == true { return "Controller torque on" }
            if state.torqueState["any"] == false { return "Controller torque off" }
        }
        return "Controller state unconfirmed"
    }

    private func ingestSafety(_ telemetry: RobotState) {
        guard connection.isLive else { stopState.disconnected(); return }
        stopState.observe(telemetry)
        if stopState.requested { estopEngaged = true }
        else if torqueOnRequested, telemetry.estop == false {
            torqueOnRequested = false; torqueRequestAt = nil; estopEngaged = false
        } else if telemetry.estop == true { estopEngaged = true }
        if stopState.requested && !stopState.acknowledged {
            startStopRetry(); retryStopIfNeeded()
        } else {
            stopRetryTimer?.invalidate(); stopRetryTimer = nil
            if stopState.acknowledged { safetyNotice = nil }
        }
    }

    private func startStopRetry() {
        guard stopRetryTimer == nil else { return }
        stopRetryTimer = Timer.scheduledTimer(withTimeInterval: RobotStopState.retryInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.retryStopIfNeeded() }
        }
    }

    private func retryStopIfNeeded(force: Bool = false) {
        guard force || connection.isLive else { return }
        let now = Date().timeIntervalSince1970
        guard force || stopState.shouldRetry(now: now) else { return }
        stopState.sent(now: now)
        link.send(.estop(true))
    }

    // MARK: - Voice (the robot's speaker + mic)

    /// Is the ROBOT's microphone open — i.e. can the operator hear the room the
    /// robot is standing in? Control only; the PCM rides the `:9610` plane.
    ///
    /// Optimistic locally (the gateway keeps `_audio_state` but does not echo it
    /// in `robot_state`), so the AUTHORITY on what the hardware is doing is the
    /// `modules` roster: it distinguishes `muted` from `live (silent)` — the bar's
    /// own physical mute button makes a perfectly healthy mic send digital zeros,
    /// and reporting that as "muted" would send someone hunting a software bug.
    @Published private(set) var micEnabled = true

    func setMicEnabled(_ on: Bool) {
        guard on != micEnabled else { return }
        micEnabled = on
        Haptics.select()
        // BOTH ends. Source-mute (the wire message) is the efficient half — the
        // robot stops sending and the link goes quiet. LOCAL mute is the half
        // that is guaranteed: it takes effect within one 20 ms frame and does not
        // depend on the robot agreeing. Without it, a robot that ignores
        // `audio_config` leaves the app playing under a button that says "Muted".
        voice.muted = !on
        link.send(.setMicEnabled(on))
    }

    /// The live `mic` / `speaker` rows from the server's roster, when it ships one.
    func audioModule(_ id: String) -> RobotState.Module? {
        state.modules.first { $0.id == id }
    }
}
