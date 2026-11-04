//
//  WebSocketLink.swift
//  MabelTeleop
//
//  Resilient live connection to the MABEL teleop server — the SAME wire
//  protocol as the iOS app (JSON envelopes on ws://<host>:9090/teleop,
//  `robot_state` back, MJPEG video on :8080), with one headset upgrade:
//  every 30 Hz `teleop_frame` carries the REAL ARKit head transform and,
//  when body tracking is enabled, the full 27-joint hand skeletons. The
//  server's retargeter + the new per-finger ORCA hand retargeting consume
//  them directly.
//
//  Stability mirrors the iOS link: tolerant sends, ping keepalive, silent-
//  link heartbeat, and auto-reconnect until the operator disconnects.
//

import Foundation

/// What the headset contributes to each outbound frame. Pulled at send time
/// so ARKit → network never queues stale poses.
struct HeadsetFrame {
    var head: HeadPose
    var leftHand: HandPose?
    var rightHand: HandPose?
}

final class WebSocketLink: NSObject, RobotLink, URLSessionWebSocketDelegate {

    private(set) var status: ConnectionStatus = .disconnected { didSet { onStatus?(status) } }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((_ state: String, _ key: String?, _ holder: String) -> Void)?
    var onMapList: ((_ maps: [MapInfo], _ loaded: String?) -> Void)?
    /// The demo-clip library, in reply to `.demoList`. Also mirrored inside
    /// robot_state so a reconnecting client still fills its picker.
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    /// VP tracking overlay payload (`operator_points`, main-actor) — the raw
    /// headset points + the engine's mapped IK targets for the 3D viewer.
    var onOperatorPoints: (([String: Any]) -> Void)?
    /// Saved poses + calibrated nav moves (`RobotSkills`): the messages this link does
    /// not consume itself, and the robot_state keys that layer reads.
    var onSkill: ((String, [String: Any]) -> Void)?
    /// The gateway refused a request (`{type:"error"}`) — e.g. a torque-on whose
    /// hardware preflight failed. Main queue. iOS parity.
    var onSafetyError: ((String) -> Void)?

    /// Opt this client in/out of the gateway's operator_points stream.
    func setOperatorOverlay(_ on: Bool) {
        sendEnvelope(type: "operator_points", payload: ["on": on])
    }
    /// De-dupe: `robot_state.server` repeats at 20 Hz; only emit on a real change.
    private var lastMeta: ServerMeta?
    /// Our connection id on this robot (from `hello.you`) and the driver-lock grant
    /// key (held while we are the driver), so the UI can tell whether WE hold it.
    private var myConnId: String?
    private var controlKey: String?
    /// The kind reported by this robot, used to stamp every meta consistently.
    private var serverKind: RobotKind = .unknown
    /// Head camera intrinsics from the server `hello` (fovy °, aspect w/h).
    var onCameraInfo: ((Double, Double) -> Void)?
    /// Server advertises its low-latency UDP video ports in the hello:
    /// (udpPort = UDP-JPEG :9091, h264Port = H.264 :9092; 0 ⇒ that codec is off).
    var onVideoInfo: ((_ udpPort: Int, _ h264Port: Int) -> Void)?

    /// Sampled when building each TeleopFrame — ON THE WIRE QUEUE, never the
    /// main thread, so the closure must be thread-safe (RobotController builds
    /// it over lock boxes + nonisolated ARKit reads). nil → identity head,
    /// no hands (the iOS-style fallback).
    var frameSource: (() -> HeadsetFrame?)?

    let host: String
    let wsPort: Int
    /// Secret real-robot access code carried as `?key=…` on the link (LAN /
    /// Tailscale) so the bridge authorizes it. nil for an open robot.
    private let accessCode: String?

    var videoBase: URL? { URL(string: "http://\(host):8080") }

    private var task: URLSessionWebSocketTask? {
        didSet { wireLock.lock(); wireTask = task; wireLock.unlock() }
    }
    /// Mirror of `task` for the wire queue (the frame clock sends on it).
    /// `task` itself stays main-confined for lifecycle.
    private var wireTask: URLSessionWebSocketTask?
    private var session: URLSession!
    /// 60 Hz outbound frame clock — a DispatchSourceTimer ON THE WIRE QUEUE,
    /// deliberately not a main-RunLoop Timer: the old main timer fired late and
    /// bursty exactly when whole-body teleop loaded the main thread, and bursty
    /// frames are what trips the server's 0.4 s engagement watchdog. A dispatch
    /// timer on a dedicated serial queue holds cadence regardless of UI load.
    private var frameClock: DispatchSourceTimer?
    private var pingTimer: Timer?
    /// 60 Hz inbound flush clock — decodes the newest robot_state ON THE
    /// DECODE QUEUE (was a main Timer + main-thread JSONSerialization parse of
    /// the full ~KB payload: real, constant main-thread load that competed
    /// with the hand overlay and SwiftUI).
    private var stateClock: DispatchSourceTimer?
    // Newest un-rendered robot_state JSON. The receive callback only STORES it (cheap)
    // so the socket drains fast; `stateClock` decodes+delivers the latest at a fixed
    // rate. Keeps every consumer from falling behind the 100 Hz stream — the backlog
    // that made the displayed latency climb without bound.
    private var latestStateText: String?
    /// Serial wire queue — the ENTIRE outbound path lives here: the 60 Hz
    /// frame clock, the ARKit sample, the skeleton encode and the socket
    /// writes, in order. The main thread never touches an outbound frame.
    private static let wireQueue = DispatchQueue(label: "mabel.wire.send", qos: .userInteractive)
    /// teleop_frame sends currently in flight on the socket (touched ONLY on
    /// `wireQueue`). Caps the outbound queue at 2 so a slow uplink drops frames
    /// (latest-wins) instead of building a standing seconds-deep FIFO.
    private var framesInFlight = 0
    /// Serial RECEIVE queue: URLSession delivers every receive completion here,
    /// so the socket drains at wire speed no matter how busy the main thread is.
    /// Draining on `.main` capped the drain at roughly one message per runloop
    /// pass — under UI load that sat BELOW the bridge's 100 Hz robot_state rate,
    /// so TCP queued the overflow CLIENT-SIDE into a standing multi-second FIFO
    /// ("the 3D model lags the sim by seconds"; video rides separate sockets,
    /// which is why it stayed live). Measured: a 40 Hz drain of the 100 Hz
    /// stream reaches 24 s of staleness in 45 s; wire-speed drain holds 0.7 ms.
    private static let recvQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "mabel.ws.recv"
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInteractive
        return q
    }()
    /// Serial DECODE queue: robot_state JSON parse runs here (fed by
    /// `stateClock`), delivering a ready RobotState struct to main. Separate
    /// from `wireQueue` so a large state parse can never delay a frame tick.
    private static let decodeQueue = DispatchQueue(label: "mabel.wire.decode", qos: .userInitiated)
    /// Guards `latestStateText`, `lastRecv` + `rttMs` (crossing recv/decode/main).
    private let recvLock = NSLock()
    private var watchdog: Timer?
    private var heartbeat: Timer?            // detects a silently-dead link
    private var lastRecv = Date()
    /// Wire-queue confined (bumped once per frame tick).
    private var seq: UInt64 = 0

    private var wantConnection = false       // false only after explicit disconnect()
    private var reconnectAttempts = 0
    private var reconnecting = false

    /// Guards the outbound command state below — written by UI intents on
    /// main, read once per tick on the wire queue.
    private let wireLock = NSLock()
    private var nav = (lx: 0.0, ly: 0.0, rx: 0.0, ry: 0.0)
    private var estopped = false
    private var head: [Double] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    private var joints: [String: Double] = [:]
    private var jointsDirty = false
    private let maxLin = 1.2, maxAng = 1.8

    private let encoder = JSONEncoder()
    /// Wire-queue confined encoder for the 60 Hz frame envelope.
    private static let frameEncoder = JSONEncoder()
    /// Frame cadence. 60 Hz (v1's rate): halves the cadence-quantization delay
    /// vs the previous 30 Hz. The 30 Hz choice existed because the encode +
    /// timer lived on the MAIN thread — off-main on a dispatch timer, the
    /// steadiness argument inverts and the extra rate is free where it matters
    /// (the wire queue). The ≤2-in-flight drop below still self-throttles slow
    /// uplinks, and the server retargets per NEW frame either way.
    private static let frameInterval = 1.0 / 60.0

    init(host: String, wsPort: Int = 9090, accessCode: String? = nil) {
        self.host = host
        self.wsPort = wsPort
        self.accessCode = accessCode
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: Self.recvQueue)
    }

    // MARK: - RobotLink

    func connect() {
        wantConnection = true
        reconnectAttempts = 0
        openSocket()
    }

    func disconnect() {
        wantConnection = false
        teardown()
        status = .disconnected
    }

    func send(_ command: RobotCommand) {
        switch command {
        case let .twist(f, s, y):
            // Canonical sim wire convention (matches iOS TwistSigns + the web client,
            // verified through the bridge's app_joystick → engine path): forward push
            // → ly = +fwd, right push → lx = -strafe, turn-right → rx = -yaw. The old
            // (-f, +s) mapping drove the base BACKWARD and to the LEFT (inverted).
            wireLock.lock()
            if estopped { nav = (0, 0, 0, 0) } else {
                nav.ly = clamp(f / maxLin)
                nav.lx = clamp(-s / maxLin)
                nav.rx = clamp(-y / maxAng)
            }
            wireLock.unlock()
        case let .liftRate(r):
            wireLock.lock(); nav.ry = clamp(r); wireLock.unlock()   // right stick Y = lift up/down
        case let .headPose(m):
            // fallback when no ARKit frame source
            if m.count == 16 { wireLock.lock(); head = m; wireLock.unlock() }
        case let .jointTarget(id, value):
            wireLock.lock(); joints[id] = value; jointsDirty = true; wireLock.unlock()
        case let .controlSpec(tracking, method, type, region, stiffness, scope):
            var p: [String: Any] = [
                "method": method, "controlType": type, "region": region,
                "stiffness": stiffness]
            // ADDITIVE, and orthogonal to `region`: is the mobile base in the
            // retargeting follow loop? Absent stays LOCAL server-side, so an
            // older build can never hand the base authority it never asked for.
            if let scope { p["scope"] = scope }
            // ADDITIVE: which operator body parts drive the robot. Merged
            // server-side, so omitting it leaves every channel as it was.
            if !tracking.isEmpty { p["tracking"] = tracking }
            sendEnvelope(type: "control_mode", payload: p)
        // Both switches ride control_mode as SINGLE keys. The bridge MERGES the
        // payload into the client's spec, so sending one never disturbs the
        // other — nor the method/type/region axes.
        case let .gravityComp(on):
            sendEnvelope(type: "control_mode", payload: ["gravityComp": on])
        case let .wholeBodyRetarget(on):
            sendEnvelope(type: "control_mode", payload: ["wholeBodyRetarget": on])
        case let .externalForce(forces):
            sendEnvelope(type: "external_force", payload: ["forces": forces])
        case let .liftFraction(frac):
            wireLock.lock(); nav.ry = clamp(frac * 2 - 1); wireLock.unlock()
        case .goHome:
            sendEnvelope(type: "reset", payload: [:])
        case .resetSim:
            sendEnvelope(type: "reset", payload: [:])   // bridge homes the sim env
        case .resetSimRandom:
            // Server-side safe respawn (SCC safe_spawn): random open pose with
            // scene clearance, joints re-homed there.
            sendEnvelope(type: "reset", payload: ["random": true])
        case .recalibrate:
            sendEnvelope(type: "recalibrate", payload: [:])
        case let .record(active, task):
            sendEnvelope(type: "record", payload: ["active": active, "task": task])
        case let .demoRecord(active, name, countdown):
            sendEnvelope(type: "demo_record",
                         payload: ["active": active, "name": name,
                                   "countdown": countdown, "recordedBy": "visionos"])
        case let .demoReplay(active, name, repeats):
            sendEnvelope(type: "demo_replay",
                         payload: ["active": active, "name": name, "repeats": repeats])
        case let .demoDelete(name):
            sendEnvelope(type: "demo_delete", payload: ["name": name])
        case .demoList:
            sendEnvelope(type: "demo_list", payload: [:])
        case let .selectTask(task):
            sendEnvelope(type: "record", payload: ["select": task])
        case let .setRobotName(name):
            sendEnvelope(type: "set_robot_name", payload: ["name": name])
        case let .setSpeedCaps(vmax, wmax):
            sendEnvelope(type: "set_speed_caps", payload: ["vmax": vmax, "wmax": wmax])
        case let .requestControl(name):
            sendEnvelope(type: "request_control", payload: ["name": name])
        case .releaseControl:
            sendEnvelope(type: "release_control", payload: ["key": controlKey ?? ""])
        case let .startBodyCalibration(user):
            sendEnvelope(type: "body_calibration", payload: ["action": "start", "user": user])
        case .cancelBodyCalibration:
            sendEnvelope(type: "body_calibration", payload: ["action": "cancel"])
        case let .applyBodyCalibration(user):
            sendEnvelope(type: "body_calibration", payload: ["action": "apply", "user": user])
        case let .estop(on):
            wireLock.lock()
            estopped = on
            if on { nav = (0, 0, 0, 0) }
            wireLock.unlock()
            // Tell the bridge too (iOS parity): the server's e-stop handler is what
            // cuts/restores MOTOR TORQUE (arms + hands) and hard-stops the base.
            // Without this envelope the button only zeroed the local twist — the
            // real robot never energised ("TORQUE ON does nothing") and E-STOP
            // never reached the server's safety path.
            sendEnvelope(type: "estop", payload: ["active": on])
        case let .setMicEnabled(on):
            // Sent IMMEDIATELY, like estop — a rare control message, not
            // per-frame state, so it must not wait on the pushFrame batch.
            sendEnvelope(type: "audio_config", payload: ["micEnabled": on])
        case let .navGoal(x, y, yaw):
            sendEnvelope(type: "nav_goal", payload: ["x": x, "y": y, "yaw": yaw])
        case .listMaps:
            sendEnvelope(type: "list_maps", payload: [:])
        case let .loadMap(name):
            sendEnvelope(type: "load_map", payload: ["name": name])
        case .resetSlam:
            sendEnvelope(type: "reset_slam", payload: [:])
        case .dispatchPolicy:
            break
        }
    }

    /// Emit server identity only when it actually changes (the `server` block
    /// repeats in every 20 Hz `robot_state`).
    private func emitMeta(_ m: ServerMeta) {
        var m = m
        if m.kind != .unknown { serverKind = m.kind }   // pick the mapping for this robot
        m.you = myConnId                                 // so the UI knows if WE hold the lock
        guard m != lastMeta else { return }
        lastMeta = m
        onServerMeta?(m)
    }

    // MARK: - Socket lifecycle

    private func openSocket() {
        guard wantConnection,
              let base = URL(string: "ws://\(host):\(wsPort)/teleop") else {
            status = .disconnected; return
        }
        // `?client=visionpro` lets the server's command monitor name this device
        // exactly (Vision Pro). Append it alongside any existing query rather
        // than overwriting it.
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        if !items.contains(where: { $0.name == "client" }) {
            items.append(URLQueryItem(name: "client", value: "visionpro"))
        }
        // A direct link to a GATED robot (real rig over LAN/Tailscale) carries the
        // secret access code as ?key=…, so the bridge authorizes it. The sim
        // passes nil and stays open.
        if let code = accessCode, !code.isEmpty, !items.contains(where: { $0.name == "key" }) {
            items.append(URLQueryItem(name: "key", value: code))
        }
        comps?.queryItems = items
        let url = comps?.url ?? base
        teardown()
        reconnecting = false
        status = .connecting
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        receive()

        recvLock.lock(); lastRecv = Date(); recvLock.unlock()
        // 60 Hz command stream on a DISPATCH TIMER pinned to the wire queue.
        // The whole outbound path (ARKit sample → skeleton encode → socket
        // send) runs there, so the cadence is immune to main-thread load —
        // steady AND fast, where the old main-RunLoop timer had to trade one
        // for the other (the engagement supervisor treats >0.4 s gaps as
        // "headset off", so bursts are worse than a slower steady rate).
        let fc = DispatchSource.makeTimerSource(queue: Self.wireQueue)
        fc.schedule(deadline: .now() + Self.frameInterval,
                    repeating: Self.frameInterval, leeway: .milliseconds(2))
        fc.setEventHandler { [weak self] in self?.pushFrame() }
        fc.resume()
        frameClock = fc
        pingTimer = scheduled(1.0) { [weak self] in self?.ping() }   // 1 Hz: keepalive + live latency probe
        // Decode + deliver the NEWEST robot_state at a fixed 60 Hz cadence, ON
        // THE DECODE QUEUE — the parse (JSONSerialization over the full ~KB
        // payload) used to run on main, where it was a standing tax competing
        // with the hand overlay + SwiftUI. Main now only receives the finished
        // RobotState struct. Intermediate frames are dropped undecoded
        // (latest-wins), so the arrival rate can never outrun the consumer.
        // Fixed-rate flush: an event-driven flush (flag set from the socket
        // thread) could wedge and silently degrade — the timer can't.
        let sc = DispatchSource.makeTimerSource(queue: Self.decodeQueue)
        sc.schedule(deadline: .now() + 1.0 / 60.0,
                    repeating: 1.0 / 60.0, leeway: .milliseconds(2))
        sc.setEventHandler { [weak self] in self?.flushLatestState() }
        sc.resume()
        stateClock = sc
        heartbeat = scheduled(1.5) { [weak self] in self?.checkAlive() }
        watchdog = Timer.scheduledTimer(withTimeInterval: 6, repeats: false) { [weak self] _ in
            guard let self, self.status == .connecting else { return }
            NSLog("[MABEL-WS] watchdog: handshake didn't complete in 6s → reconnect")
            self.dropped()
        }
    }

    private func checkAlive() {
        recvLock.lock(); let last = lastRecv; recvLock.unlock()
        guard status == .connected, Date().timeIntervalSince(last) > 3.5 else { return }
        NSLog("[MABEL-WS] no data for >3.5s → link is dead, reconnect")
        dropped()
    }

    private func teardown() {
        frameClock?.cancel(); frameClock = nil
        pingTimer?.invalidate(); pingTimer = nil
        stateClock?.cancel(); stateClock = nil
        recvLock.lock(); latestStateText = nil; recvLock.unlock()
        watchdog?.invalidate(); watchdog = nil
        heartbeat?.invalidate(); heartbeat = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
    }

    private func dropped() {
        guard !reconnecting else { return }
        reconnecting = true
        teardown()
        guard wantConnection else { status = .disconnected; return }
        status = .connecting
        reconnectAttempts += 1
        let delay = min(3.0, 0.3 * Double(reconnectAttempts))
        NSLog("[MABEL-WS] reconnecting (attempt \(reconnectAttempts)) in \(delay)s")
        Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            self?.openSocket()
        }
    }

    /// Keepalive AND latency probe — the APPLICATION-level ping the bridge
    /// understands (sim_teleop_bridge.py echoes a `pong` carrying our `t`), the SAME
    /// mechanism the web console uses. We stamp the send time and the matching pong
    /// (handled in `handle`) turns it into an RTT. This is deliberately NOT
    /// URLSession's protocol-level `sendPing`: that completion proved unreliable
    /// across the bridge (it often never fired), and when it DID fire under
    /// load its completion sat behind the inbound backlog on the main queue — so the
    /// "latency" climbed without bound. The app-level pong flows through the same
    /// receive path as `robot_state` (handled inline, cheaply), so the number is a
    /// true end-to-end RTT that stays bounded. A missing pong just leaves the last
    /// RTT in place — it never tears the session down (drops surface on receive()).
    private func ping() {
        guard task != nil else { return }
        pingSentAt = Date()
        sendEnvelope(type: "ping", payload: ["t": pingSentAt!.timeIntervalSince1970])
    }
    /// When the outstanding ping left (nil ⇒ none in flight); the matching pong
    /// converts it to `rttMs`. Single in-flight, like the web console's `_pingSent`.
    private var pingSentAt: Date?
    /// Smoothed application round-trip latency in ms (0 until the first pong).
    /// Guarded by `recvLock`: written on main (onPong), read on the decode queue.
    private var rttMs: Double = 0

    /// A `pong` came back — elapsed since our ping is the true round-trip latency.
    private func onPong() {
        guard let sent = pingSentAt else { return }
        pingSentAt = nil
        let rtt = Date().timeIntervalSince(sent) * 1000.0
        #if DEBUG
        NSLog("[MABEL-RTT] rtt=%.3f t=%.4f", rtt, sent.timeIntervalSince1970 * 1000)   // per-pong, for teleop_latency
        #endif
        recvLock.lock()
        rttMs = rttMs == 0 ? rtt : 0.8 * rttMs + 0.2 * rtt
        recvLock.unlock()
    }

    // MARK: - Framing

    /// Single-pass encodable envelope for the 60 Hz frame — one JSONEncoder
    /// pass over the Codable pose models replaces the old three
    /// encode→parse round-trips + a final JSONSerialization (~3× the CPU for
    /// the identical bytes; the Codable field names ARE the wire contract).
    private struct TeleopFrameEnvelope: Encodable {
        var type = "teleop_frame"
        var payload: TeleopFrame
    }

    /// Internal so WireCodecTests pins the wire shape against the manual-dict
    /// encoding this replaced.
    static func encodeFrameEnvelope(_ frame: TeleopFrame) throws -> Data {
        try frameEncoder.encode(TeleopFrameEnvelope(payload: frame))
    }

    /// Build + send one teleop_frame. Runs ON THE WIRE QUEUE (the frame
    /// clock's event handler): the ARKit sample, the skeleton encode and the
    /// socket send never touch the main thread — the ENTIRE outbound tracking
    /// path is main-thread-free (2026-07-29; the finger-overlay-lag cluster).
    private func pushFrame() {
        wireLock.lock()
        let task = wireTask
        let nav = self.nav
        let headFallback = self.head
        let dirtyJoints: [String: Double]? = jointsDirty ? joints : nil
        jointsDirty = false
        wireLock.unlock()
        guard let task else { return }
        seq += 1

        // Stream direct joint targets when they change (Body tab / WBC
        // control) — already on the wire queue, so serialize + send inline.
        if let dirtyJoints {
            let env: [String: Any] = ["type": "joint_command",
                                      "payload": ["joints": dirtyJoints]]
            if let data = try? JSONSerialization.data(withJSONObject: env),
               let str = String(data: data, encoding: .utf8) {
                task.send(.string(str)) { _ in }
            }
        }

        // DROP, don't queue: URLSession's socket send has no back-pressure,
        // so on a slow uplink (weak Wi-Fi / Tailscale) queued skeleton frames
        // build a standing client-side FIFO and the robot mirrors seconds-old
        // hands — the OUTBOUND twin of the robot_state drain bug. Skip this
        // tick when two sends are already in flight; the next tick pulls a
        // FRESHER pose anyway (latest-wins, exactly like the video path).
        guard framesInFlight < 2 else { return }

        let hf = frameSource?()   // thread-safe by contract (see property doc)
        let headPose: HeadPose
        if let hf {
            headPose = hf.head
        } else {
            // No ARKit (simulator / tracking down): the head must NOT read
            // as live — "normal" made the engine hold the neck at this
            // identity pose every frame, stomping joint-space neck commands
            // (iOS sends notAvailable here too).
            headPose = HeadPose(transform: Transform4x4(matrix: headFallback.map(Float.init)),
                                trackingState: .notAvailable)
        }
        var frame = TeleopFrame(sequence: seq,
                                timestamp: Date().timeIntervalSinceReferenceDate,
                                head: headPose,
                                leftHand: hf?.leftHand,
                                rightHand: hf?.rightHand)
        frame.navJoystick = NavJoystick(lx: Float(nav.lx), ly: Float(nav.ly),
                                        rx: Float(nav.rx), ry: Float(nav.ry))

        framesInFlight += 1
        guard let data = try? Self.encodeFrameEnvelope(frame),
              let str = String(data: data, encoding: .utf8) else {
            framesInFlight -= 1
            return
        }
        task.send(.string(str)) { _ in
            Self.wireQueue.async { self.framesInFlight -= 1 }
        }
    }

    /// Encodable → JSON object (the Codable field names are the wire contract).
    private func jsonObject<T: Encodable>(_ value: T) -> Any {
        guard let data = try? encoder.encode(value),
              let obj = try? JSONSerialization.jsonObject(with: data) else { return [:] }
        return obj
    }

    /// One decoder for both arrival paths — the `demo_clips` reply and the
    /// `robot_state.animation` mirror. Two copies would drift.
    static func parseClips(_ raw: Any?) -> [RobotState.DemoClipInfo] {
        ((raw as? [[String: Any]]) ?? []).compactMap {
            guard let name = $0["name"] as? String else { return nil }
            return RobotState.DemoClipInfo(
                name: name,
                duration: ($0["duration"] as? Double) ?? 0,
                frames: ($0["frames"] as? Int) ?? 0,
                created: ($0["created"] as? String) ?? "",
                recordedBy: ($0["recordedBy"] as? String) ?? "",
                joints: ($0["joints"] as? Int) ?? 0)
        }
    }

    private func sendEnvelope(type: String, payload: [String: Any]) {
        guard let task else { return }
        let env: [String: Any] = ["type": type, "payload": payload]
        Self.wireQueue.async {
            guard let data = try? JSONSerialization.data(withJSONObject: env),
                  let str = String(data: data, encoding: .utf8) else { return }
            task.send(.string(str)) { _ in }
        }
    }

    private func receive() {
        task?.receive { [weak self] result in
            guard let self else { return }
            // Runs on `recvQueue` — NEVER the main thread. The work here must stay
            // cheap (classify + store) so the drain rate always exceeds the 100 Hz
            // arrival rate and no client-side TCP backlog can ever form.
            switch result {
            case .failure(let err):
                NSLog("[MABEL-WS] receive failure: \(err.localizedDescription)")
                DispatchQueue.main.async { self.dropped() }
            case .success(let msg):
                self.receive()       // re-arm on the recv queue — never waits on main
                var text: String?
                if case let .string(t) = msg { text = t }
                else if case let .data(d) = msg { text = String(data: d, encoding: .utf8) }
                guard let text else { return }
                self.recvLock.lock()
                self.lastRecv = Date()
                // Coalesce the high-rate robot_state flood to the NEWEST frame (decoded
                // by stateClock on the decode queue); hand rarer control messages (hello/
                // control/pong/map_list) to main immediately so none are ever dropped.
                if text.contains("\"robot_state\"") {
                    self.latestStateText = text
                    self.recvLock.unlock()
                } else {
                    self.recvLock.unlock()
                    DispatchQueue.main.async { self.handle(text) }
                }
            }
        }
    }

    /// Decode + deliver the newest buffered robot_state (called by `stateClock`
    /// at a fixed cadence, ON THE DECODE QUEUE). Intermediate frames are
    /// dropped undecoded, and the JSON parse itself runs here — the main
    /// thread only receives the finished RobotState struct, so a 100 Hz state
    /// stream costs main nothing but the delivery closure.
    private func flushLatestState() {
        recvLock.lock()
        let t = latestStateText
        latestStateText = nil
        let rtt = rttMs
        recvLock.unlock()
        guard let t, let data = t.data(using: .utf8),
              let env = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        guard env["type"] as? String == "robot_state",
              let p = env["payload"] as? [String: Any] else {
            // The receive() router keys on a substring — a rare non-state
            // message mentioning "robot_state" lands here; hand it to the
            // normal main-thread path instead of dropping it.
            DispatchQueue.main.async { self.handle(t) }
            return
        }

        #if DEBUG
        // Same-host QA probe (simulator + loopback bridge share a clock): true
        // one-way staleness of the state we're about to render, logged ~1/s as
        // [MABEL-LAG]. Guards against the receive-buffer-buildup regression.
        // Cross-host the value includes clock skew — ignore it there.
        // Counters are decode-queue confined.
        if let sent = p["sentAt"] as? Double {
            logStaleness(Date().timeIntervalSince1970 - sent)
        }
        #endif

        // Pure construction off-main; main-side delivery below.
        let meta = (p["server"] as? [String: Any]).map { ServerMeta(from: $0) }
        let st = Self.parseRobotState(p, rttMs: rtt)
        let skills = SkillMessages.stateExtras(p)
        DispatchQueue.main.async {
            if let skills { self.onSkill?("robot_state", skills) }
            if self.status != .connected {
                self.status = .connected; self.reconnectAttempts = 0; self.watchdog?.invalidate()
            }
            if let meta { self.emitMeta(meta) }
            self.onState?(st)
        }
    }

    private func handle(_ text: String) {
        recvLock.lock(); lastRecv = Date(); recvLock.unlock()
        if status != .connected { status = .connected; reconnectAttempts = 0; watchdog?.invalidate() }
        guard let data = text.data(using: .utf8),
              let env = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        let type = env["type"] as? String

        if type == "error" {
            let p = env["payload"] as? [String: Any] ?? [:]
            onSkill?("error", p)
            if SkillMessages.ownsError(p) { return }     // a pose / nav_move refusal, not a safety event
            onSafetyError?((p["message"] as? String) ?? (p["msg"] as? String) ?? "Robot rejected request")
            return
        }

        // Latency probe reply — convert our outstanding ping into an RTT.
        if type == "pong" { onPong(); return }

        // VP tracking overlay (retarget debug): raw + engine-mapped operator
        // points in robot/odom world — see OperatorOverlay.swift.
        if type == "operator_points" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            DispatchQueue.main.async { self.onOperatorPoints?(p) }
            return
        }

        if type == "hello" {
            let payload = env["payload"] as? [String: Any]
            myConnId = payload?["you"] as? String   // remember OUR id for the lock UI
            onServerInfo?(payload?["server"] as? String)
            if let payload { emitMeta(ServerMeta(from: payload)) }
            onSkill?("hello", payload ?? [:])
            if let head = ((payload?["cameras"] as? [String: Any])?["head"] as? [String: Any]),
               let fovy = head["fovy"] as? Double {
                onCameraInfo?(fovy, head["aspect"] as? Double ?? 16.0 / 9.0)
            }
            if let video = payload?["video"] as? [String: Any] {
                let udpPort = video["udpPort"] as? Int ?? 0
                let h264Port = video["h264Port"] as? Int ?? 0
                if udpPort > 0 || h264Port > 0 { onVideoInfo?(udpPort, h264Port) }
            }
            return
        }

        // Saved-map catalogue (list_maps reply, and pushed after a load_map).
        if type == "demo_clips" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            onDemoClips?(Self.parseClips(p["clips"]))
            return
        }
        if type == "map_list" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            let maps = ((p["maps"] as? [[String: Any]]) ?? []).compactMap { m -> MapInfo? in
                guard let name = m["name"] as? String else { return nil }
                return MapInfo(name: name,
                               hasMesh: m["has_mesh"] as? Bool ?? false,
                               hasReloc: m["has_reloc"] as? Bool ?? false)
            }
            let loaded = (p["loaded"] as? String).flatMap { $0.isEmpty || $0 == "live" ? nil : $0 }
            onMapList?(maps, loaded)
            return
        }

        // Driver-lock reply (real robot): granted carries the key we hold; denied
        // means someone else has it; released frees ours.
        if type == "control" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            let state = (p["state"] as? String) ?? ""
            let key = p["key"] as? String
            if state == "granted" { controlKey = key }
            else if state == "released" || state == "denied" { controlKey = nil }
            onControl?(state, key, (p["holder"] as? String) ?? "")
            return
        }

        if let type, SkillMessages.types.contains(type) {
            onSkill?(type, (env["payload"] as? [String: Any]) ?? [:])
            return
        }

        guard type == "robot_state",
              let p = env["payload"] as? [String: Any] else { return }
        if let x = SkillMessages.stateExtras(p) { onSkill?("robot_state", x) }
        // Normally unreachable — receive() routes robot_state to the decode
        // queue. Kept for the DemoLink-free direct-call paths and safety.
        if let srv = p["server"] as? [String: Any] { emitMeta(ServerMeta(from: srv)) }
        recvLock.lock(); let rtt = rttMs; recvLock.unlock()
        onState?(Self.parseRobotState(p, rttMs: rtt))
    }

    /// Field-by-field robot_state extraction — PURE (no self), so it can run
    /// on the decode queue. The dict keys are the wire contract.
    private static func parseRobotState(_ p: [String: Any], rttMs: Double) -> RobotState {
        var st = RobotState()
        if let bat = p["battery"] as? [String: Any] {
            st.batteryLevel = (bat["percentage"] as? Double) ?? st.batteryLevel
            st.batteryVoltage = (bat["voltage"] as? Double) ?? st.batteryVoltage
        }
        // Server-echoed operator switches (absent on older servers → unchanged).
        if let g = p["gravityComp"] as? Bool { st.gravityComp = g }
        if let w = p["wholeBodyRetarget"] as? Bool { st.wholeBodyRetarget = w }
        st.estop = p["estop"] as? Bool
        st.torqueDegraded = (p["torqueDegraded"] as? [String: Any])?.mapValues { "\($0)" } ?? [:]
        if let joints = p["jointPositions"] as? [String: Double] {
            for (k, v) in joints { st.joints[k] = v }
            st.reportedJoints.formUnion(joints.keys)
        }
        // Optional (sparse ~10 Hz, sim-bridge V2 extension): true qvel/actuator
        // torque per joint. Absent on older bridges / the real robot — the UI
        // must degrade to derived velocity + "torque not streamed".
        if let vel = p["jointVelocities"] as? [String: Double] { st.jointVelocities = vel }
        if let tor = p["jointTorques"] as? [String: Double] { st.jointTorques = tor }
        // Prefer the locally-measured WebSocket round-trip (true link latency) over
        // the server's latencyMs (command staleness). Fall back to the server value
        // only until the first pong lands.
        if rttMs > 0 { st.latencyMs = rttMs }
        else if let lat = p["latencyMs"] as? Double { st.latencyMs = lat }
        if let age = p["stateAgeMs"] as? Double { st.stateAgeMs = age }
        if let an = p["animation"] as? [String: Any] {
            st.demo = RobotState.DemoEcho(
                recording: (an["recording"] as? Bool) ?? false,
                armed: (an["armed"] as? Bool) ?? false,
                countdown: (an["countdown"] as? Double) ?? 0,
                playing: (an["playing"] as? Bool) ?? false,
                name: an["name"] as? String,
                frames: (an["frames"] as? Int) ?? 0,
                elapsed: (an["elapsed"] as? Double) ?? 0,
                phase: an["phase"] as? String,
                rep: (an["rep"] as? Int) ?? 0,
                repeats: (an["repeats"] as? Int) ?? 0,
                frac: (an["frac"] as? Double) ?? 0,
                clips: parseClips(an["clips"]))
        }
        if let rec = p["recording"] as? [String: Any] {
            st.recordTask = rec["task"] as? String ?? ""
            st.episodeCount = rec["episodeCount"] as? Int ?? 0
            st.taskList = rec["tasks"] as? [String] ?? []
            st.taskCounts = rec["taskCounts"] as? [String: Int] ?? [:]
            st.episodesTotal = rec["total"] as? Int ?? 0
            st.episodesDir = rec["dir"] as? String ?? ""
            st.serverRecording = rec["isRecording"] as? Bool ?? false
            st.recordingReported = rec["isRecording"] is Bool   // an older server without it keeps the tap's state
            st.recordElapsed = rec["elapsed"] as? Double ?? 0
            st.recordError = rec["error"] as? String
            let frames = ((rec["counts"] as? [String: Any])?["frames"] as? [String: Int]) ?? [:]
            let health = ((rec["quality"] as? [String: Any])?["cameras"] as? [String: [String: Any]]) ?? [:]
            st.recordCameras = frames.keys.sorted().map { n in
                RobotState.RecordCamera(name: n, frames: frames[n] ?? 0, live: (frames[n] ?? 0) > 0,
                                        error: (frames[n] ?? 0) == 0 && health[n]?["error"] != nil)
            }
        }
        if let base = p["base"] as? [String: Any] {
            if let yaw = base["yaw"] as? Double { st.heading = yaw }
            if let x = base["x"] as? Double { st.baseX = x }
            if let y = base["y"] as? Double { st.baseY = y }
            if let quat = base["quat"] as? [Double], quat.count == 4 { st.baseQuat = quat }
            if let steer = base["steer"] as? [Double], steer.count == 3 {
                st.joints["fl_steer"] = steer[0]
                st.joints["fr_steer"] = steer[1]
                st.joints["b_steer"] = steer[2]
            }
        }
        // Per-operator body calibration: live capture progress + measured profile.
        if let cal = p["calibrationStatus"] as? [String: Any] { st.calibration = CalibrationProgress(wire: cal) }
        if let prof = p["operatorProfile"] as? [String: Any] { st.operatorMetrics = OperatorMetrics(wire: prof) }
        // Sparse planar LiDAR sweep — attached by the bridge only when fresh (~5 Hz).
        if let lidar = p["lidar"] as? [String: Any],
           let ranges = lidar["ranges"] as? [Double], !ranges.isEmpty {
            st.lidar = RobotState.LidarScan(
                angleMin: lidar["angleMin"] as? Double ?? 0,
                angleInc: lidar["angleInc"] as? Double ?? (2 * .pi / Double(ranges.count)),
                rangeMax: lidar["rangeMax"] as? Double ?? 12,
                ranges: ranges,
                stamp: lidar["stamp"] as? Double ?? 0)
        }
        // Per-module connection roster (sparse/change-gated → latched in RobotController).
        if let mods = p["modules"] as? [[String: Any]] {
            st.modules = mods.compactMap { m in
                guard let id = m["id"] as? String, let ok = m["ok"] as? Bool else { return nil }
                // up / n / actuators are NEWER than the roster itself: a server that
                // sends neither still decodes, it just has no per-motor detail to show.
                let acts = (m["actuators"] as? [[String: Any]] ?? []).compactMap { a -> RobotState.Module.Actuator? in
                    guard let aid = a["id"] as? String, let aok = a["ok"] as? Bool else { return nil }
                    return RobotState.Module.Actuator(id: aid,
                                                      label: (a["label"] as? String) ?? aid,
                                                      ok: aok,
                                                      detail: (a["detail"] as? String) ?? "")
                }
                return RobotState.Module(id: id, label: (m["label"] as? String) ?? id,
                                         group: m["group"] as? String, ok: ok,
                                         detail: (m["detail"] as? String) ?? "",
                                         up: (m["up"] as? Int) ?? (acts.isEmpty ? nil : acts.filter(\.ok).count),
                                         n: (m["n"] as? Int) ?? (acts.isEmpty ? nil : acts.count),
                                         actuators: acts)
            }
        }
        return st
    }

    #if DEBUG
    private var staleSumMs = 0.0, staleMaxMs = 0.0
    private var staleN = 0
    private var staleLogAt = Date()
    private func logStaleness(_ s: TimeInterval) {
        let ms = s * 1000
        staleSumMs += ms; staleMaxMs = max(staleMaxMs, ms); staleN += 1
        guard Date().timeIntervalSince(staleLogAt) >= 1.0 else { return }
        NSLog("[MABEL-LAG] robot_state staleness avg=%.1fms max=%.1fms n=%d",
              staleSumMs / Double(max(staleN, 1)), staleMaxMs, staleN)
        staleSumMs = 0; staleMaxMs = 0; staleN = 0; staleLogAt = Date()
    }
    #endif

    private func clamp(_ v: Double) -> Double { max(-1, min(1, v)) }

    private func scheduled(_ interval: TimeInterval, _ block: @escaping () -> Void) -> Timer {
        let t = Timer(timeInterval: interval, repeats: true) { _ in block() }
        RunLoop.main.add(t, forMode: .common)
        return t
    }

    // MARK: - URLSessionWebSocketDelegate

    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol proto: String?) {
        NSLog("[MABEL-WS] didOpen")
        DispatchQueue.main.async { self.status = .connected; self.reconnectAttempts = 0; self.watchdog?.invalidate() }
    }
    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith code: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        NSLog("[MABEL-WS] didClose code=\(code.rawValue)")
        DispatchQueue.main.async { self.dropped() }
    }
    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        NSLog("[MABEL-WS] task completed error: \(error.localizedDescription) → reconnect")
        DispatchQueue.main.async { self.dropped() }
    }
}

// MARK: - RobotSkills transport

extension WebSocketLink: SkillsTransport {
    func sendSkill(_ type: String, _ payload: [String: Any]) {
        sendEnvelope(type: type, payload: payload)
    }
}
