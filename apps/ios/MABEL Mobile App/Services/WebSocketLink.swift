//
//  WebSocketLink.swift
//  MABEL Mobile App
//
//  Resilient live connection to the MABEL teleop server (server,
//  started with MABEL/server/run.sh). It speaks the SAME wire protocol as
//  the Vision Pro app: JSON `teleop_frame` envelopes on
//  ws://<host>:9090/teleop, decoding `robot_state` back. Base driving rides
//  as a `navJoystick` (the server runs the swerve IK). Video is MJPEG (see
//  MJPEGView).
//
//  Stability: transient send errors are tolerated; the link keeps a 20 Hz
//  command stream and a periodic ping, and AUTO-RECONNECTS on any drop until
//  the operator explicitly disconnects.
//

import Foundation

final class WebSocketLink: NSObject, RobotLink, URLSessionWebSocketDelegate {

    private(set) var status: ConnectionStatus = .disconnected { didSet { onStatus?(status) } }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((_ state: String, _ key: String?, _ holder: String) -> Void)?
    var onMapList: ((_ maps: [MapInfo], _ loaded: String?) -> Void)?
    /// The demo-clip library, in reply to `.demoList`. Also arrives inside
    /// robot_state, so a client that missed this reply still fills its picker.
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    /// VP tracking overlay payload (`operator_points`, main-actor) — the raw
    /// headset points + the engine's mapped IK targets for the 3D viewer.
    var onOperatorPoints: (([String: Any]) -> Void)?
    var onSafetyError: ((String) -> Void)?

    /// Opt this client in/out of the gateway's operator_points stream.
    func setOperatorOverlay(_ on: Bool) {
        sendEnvelope(type: "operator_points", payload: ["on": on])
    }
    private var lastMeta: ServerMeta?
    /// Learned from `hello.kind` — selects the per-robot stick→twist mapping.
    private var serverKind: RobotKind = .unknown
    /// Our connection id on this robot (from `hello.you`) and the driver-lock grant
    /// key (when we hold it) — used to drive the lock UI and to release on demand.
    private var myConnId: String?
    private var controlKey: String?

    let host: String
    let wsPort: Int
    /// Secret access code appended as `?key=…` on a DIRECT link to a gated robot
    /// (the real rig over LAN/Tailscale). nil for the open sim bridge.
    private let accessCode: String?

    var videoBase: URL? { URL(string: "http://\(host):8080") }

    private var task: URLSessionWebSocketTask?
    private var session: URLSession!
    private var sendTimer: Timer?
    private var pingTimer: Timer?
    private var stateTimer: Timer?           // throttles inbound robot_state decode+render
    // Newest un-rendered robot_state JSON. The receive callback only STORES it
    // (cheap) so the socket drains at wire speed; `stateTimer` decodes + delivers
    // the latest at a fixed cadence on main. Latest-wins: intermediate frames are
    // dropped undecoded, so the arrival rate can never outrun the main thread.
    private var latestStateText: String?
    private var latestStateTask: URLSessionWebSocketTask?
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
    /// Guards `latestStateText` + `lastRecv` (written on recvQueue, read on main).
    private let recvLock = NSLock()
    private var watchdog: Timer?
    private var heartbeat: Timer?            // detects a silently-dead link
    private var lastRecv = Date()            // last time ANY message arrived
    private var seq: UInt64 = 0

    private var wantConnection = false       // false only after explicit disconnect()
    private var reconnectAttempts = 0
    private var reconnecting = false
    private var reconnectTimer: Timer?

    private var nav = (lx: 0.0, ly: 0.0, rx: 0.0, ry: 0.0)
    private var estopped = false
    private var head: [Double] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    private var headOn = false          // active-perception head tracking on/off
    // Direct joint targets (Body tab / gravity-comp / compliance) and the
    // active control mode — streamed to the bridge so the iOS app is a pure
    // input device and the robot models live server-side.
    private var joints: [String: Double] = [:]
    private var jointsDirty = false
    private let maxLin = 1.2, maxAng = 1.8

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
            guard !estopped else { nav = (0, 0, 0, 0); return }
            // Stick → nav signs are PER ROBOT KIND: the sim and the physical rig
            // have independent base frames, so each gets its own mapping (learned
            // from `hello.kind`). Sim keeps its historical mapping; the real rig
            // is tuned separately.
            let sg = TwistSigns.forKind(serverKind)
            nav.ly = clamp(sg.fwd * f / maxLin)
            nav.lx = clamp(sg.strafe * s / maxLin)
            nav.rx = clamp(sg.yaw * y / maxAng)
        case let .navGoal(x, y, yaw):
            // Nav2 goal → the bridge's NAV_GOAL handler → /goal_pose (NavFn A*+MPPI).
            sendEnvelope(type: "nav_goal", payload: ["x": x, "y": y, "yaw": yaw])
        case let .liftRate(r):
            nav.ry = estopped ? 0 : clamp(r)
        case let .headPose(m):
            if m.count == 16 { head = m }     // streamed each tick in the frame
        case let .headTracking(on):
            headOn = on                       // surfaces as the frame's trackingState
        case let .jointTarget(id, value):
            if !estopped { joints[id] = value; jointsDirty = true }
        case let .controlSpec(method, type, region, stiffness):
            sendEnvelope(type: "control_mode", payload: [
                "method": method, "controlType": type, "region": region, "stiffness": stiffness])
        // Both switches ride control_mode as SINGLE keys. The bridge MERGES the
        // payload into the client's spec, so sending one never disturbs the
        // other — and never disturbs the method/type/region axes either.
        case let .gravityComp(on):
            sendEnvelope(type: "control_mode", payload: ["gravityComp": on])
        case let .wholeBodyRetarget(on):
            sendEnvelope(type: "control_mode", payload: ["wholeBodyRetarget": on])
        case let .externalForce(forces):
            sendEnvelope(type: "external_force", payload: ["forces": forces])
        case let .liftFraction(frac):
            nav.ry = estopped ? 0 : clamp(frac * 2 - 1)
        case .goHome:
            sendEnvelope(type: "reset", payload: [:])
        case .resetSim:
            sendEnvelope(type: "reset", payload: [:])   // bridge homes the sim env
        case .resetSimRandom:
            // Server-side safe respawn (SCC safe_spawn): random open pose with
            // 0.8 m clearance + MuJoCo contact verification, then re-home there.
            sendEnvelope(type: "reset", payload: ["random": true])
        case .resetSlam:
            // Server drops the map and starts SLAM fresh from the current pose
            // (ROS relaunches mapping; localization re-anchors at "here").
            sendEnvelope(type: "reset_slam", payload: [:])
        case let .demoRecord(active, name, countdown):
            sendEnvelope(type: "demo_record",
                         payload: ["active": active, "name": name,
                                   "countdown": countdown, "recordedBy": "ios"])
        case let .demoReplay(active, name, repeats):
            sendEnvelope(type: "demo_replay",
                         payload: ["active": active, "name": name, "repeats": repeats])
        case let .demoDelete(name):
            sendEnvelope(type: "demo_delete", payload: ["name": name])
        case .demoList:
            sendEnvelope(type: "demo_list", payload: [:])
        case let .record(active, task):
            sendEnvelope(type: "record", payload: ["active": active, "task": task])
        case let .selectTask(task):
            sendEnvelope(type: "record", payload: ["select": task])
        case let .setSpeedCaps(vmax, wmax):
            sendEnvelope(type: "set_speed_caps", payload: ["vmax": vmax, "wmax": wmax])
        case let .setRobotName(name):
            sendEnvelope(type: "set_robot_name", payload: ["name": name])
        case let .requestControl(name):
            sendEnvelope(type: "request_control", payload: ["name": name])
        case .releaseControl:
            sendEnvelope(type: "release_control", payload: ["key": controlKey ?? ""])
            controlKey = nil
        case let .estop(on):
            estopped = on
            if on { nav = (0, 0, 0, 0); joints.removeAll(); jointsDirty = false }
            // Tell the bridge too, so the server (and the command monitor) sees
            // the e-stop — not just the local nav clamp.
            sendEnvelope(type: "estop", payload: ["active": on], safety: true)
        case let .setMicEnabled(on):
            // Sent IMMEDIATELY, like estop — a rare control message, not per-frame
            // state, so it must not wait on the 20 Hz pushFrame dirty-flag batch.
            sendEnvelope(type: "audio_config", payload: ["micEnabled": on])
        case .dispatchPolicy:
            break
        case let .dataCollect(active, label):
            sendEnvelope(type: "data_collect", payload: ["active": active, "label": label])
        case let .registerPush(token):
            sendEnvelope(type: "register_push", payload: ["token": token, "platform": "apns"])
        case .listMaps:
            sendEnvelope(type: "list_maps", payload: [:])
        case let .loadMap(name):
            sendEnvelope(type: "load_map", payload: ["name": name])
        }
    }

    // MARK: - Socket lifecycle

    private func openSocket() {
        // A direct link uses ws://host:9090/teleop, with `client=ios` appended so the
        // server's command monitor names this device exactly (iPhone).
        guard wantConnection,
              let base = URL(string: "ws://\(host):\(wsPort)/teleop") else {
            status = .disconnected; return
        }
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)
        var items = comps?.queryItems ?? []
        if !items.contains(where: { $0.name == "client" }) {
            items.append(URLQueryItem(name: "client", value: "ios"))
        }
        // A direct link to a GATED robot (real rig over LAN/Tailscale) carries the
        // secret access code as ?key=…, so the bridge authorizes it on every
        // transport. The sim passes nil and stays open.
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
        // 60 Hz uplink — matches the web console (TX_HZ = 60): command staleness ~16 ms
        // instead of ~50 ms. Frames are tiny fire-and-forget JSON (no awaited send,
        // latest-wins), so the higher rate adds no backlog.
        sendTimer = scheduled(1.0 / 60.0) { [weak self] in self?.pushFrame() }
        pingTimer = scheduled(1.0) { [weak self] in self?.ping() }   // 1 Hz: keepalive + live latency probe
        // Fixed-rate flush: decode + deliver the NEWEST robot_state at 60 Hz on
        // main. The receive queue only stores text (latest-wins), so the render
        // rate — never the arrival rate — bounds the main-thread cost.
        stateTimer = scheduled(1.0 / 60.0) { [weak self] in self?.flushLatestState() }
        // Heartbeat: the bridge streams robot_state at 20 Hz, so a multi-second
        // silence means the link is dead (e.g. the link-local address vanished) —
        // reconnect even if no FIN/RST ever arrives.
        heartbeat = scheduled(1.5) { [weak self] in self?.checkAlive() }
        // If the handshake never completes, retry rather than hang.
        watchdog = Timer.scheduledTimer(withTimeInterval: 6, repeats: false) { [weak self] _ in
            guard let self, self.status == .connecting else { return }
            NSLog("[MABEL-WS] watchdog: handshake didn't complete in 6s → reconnect")
            self.dropped()
        }
    }

    /// Drop a link that's gone silent (no message for >3.5 s while connected).
    private func checkAlive() {
        recvLock.lock(); let last = lastRecv; recvLock.unlock()
        guard status == .connected, Date().timeIntervalSince(last) > 3.5 else { return }
        NSLog("[MABEL-WS] no data for >3.5s → link is dead, reconnect")
        dropped()
    }

    private func teardown() {
        sendTimer?.invalidate(); sendTimer = nil
        pingTimer?.invalidate(); pingTimer = nil
        stateTimer?.invalidate(); stateTimer = nil
        recvLock.lock(); latestStateText = nil; latestStateTask = nil; recvLock.unlock()
        reconnectTimer?.invalidate(); reconnectTimer = nil
        watchdog?.invalidate(); watchdog = nil
        heartbeat?.invalidate(); heartbeat = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
    }

    /// An unexpected drop — reconnect (once) if the operator still wants a link.
    private func dropped() {
        guard !reconnecting else { return }       // collapse simultaneous drop signals
        reconnecting = true
        teardown()
        guard wantConnection else { status = .disconnected; return }
        status = .connecting
        reconnectAttempts += 1
        // Quick first retries, capped — keep trying FOREVER while the operator
        // still wants the link (wantConnection stays true until disconnect()).
        let delay = min(3.0, 0.3 * Double(reconnectAttempts))
        NSLog("[MABEL-WS] reconnecting (attempt \(reconnectAttempts)) in \(delay)s")
        reconnectTimer = Timer(timeInterval: delay, repeats: false) { [weak self] _ in
            self?.openSocket()
        }
        RunLoop.main.add(reconnectTimer!, forMode: .common)
    }

    /// Keepalive AND latency probe — the APPLICATION-level ping the bridge
    /// understands (sim_teleop_bridge.py echoes a `pong` carrying our `t`), the
    /// SAME mechanism the web console uses. We stamp the send time and the matching
    /// pong (handled in `handle`) turns it into an RTT. This is deliberately NOT
    /// URLSession's protocol-level `sendPing`: that completion proved unreliable
    /// across the bridge (it often never fired), and when it DID fire under
    /// load its completion sat behind the inbound backlog on the main queue — so the
    /// "latency" climbed without bound (the 42 s runaway). The app-level pong flows
    /// through the same receive path as `robot_state` (handled inline, cheaply), so
    /// the number is a true end-to-end RTT that stays bounded. A missing pong just
    /// leaves the last RTT in place — it never tears the session down (real drops
    /// surface on receive() / the heartbeat).
    private func ping() {
        guard task != nil else { return }
        pingSentAt = Date()
        sendEnvelope(type: "ping", payload: ["t": pingSentAt!.timeIntervalSince1970])
    }
    /// When the outstanding ping left (nil ⇒ none in flight); the matching pong
    /// converts it to `rttMs`. Single in-flight, like the web console's `_pingSent`.
    private var pingSentAt: Date?
    /// Smoothed application round-trip latency in ms (0 until the first pong).
    private var rttMs: Double = 0

    /// A `pong` came back — elapsed since our ping is the true round-trip latency.
    private func onPong() {
        guard let sent = pingSentAt else { return }
        pingSentAt = nil
        let rtt = Date().timeIntervalSince(sent) * 1000.0
        rttMs = rttMs == 0 ? rtt : 0.8 * rttMs + 0.2 * rtt
    }

    // MARK: - Framing

    private func pushFrame() {
        guard task != nil else { return }
        seq += 1
        let payload: [String: Any] = [
            "sequence": seq,
            "timestamp": Date().timeIntervalSinceReferenceDate,
            "head": ["transform": ["matrix": head],
                     "trackingState": headOn ? "normal" : "notAvailable"],
            "mode": "Base Driving",
            "navJoystick": ["lx": nav.lx, "ly": nav.ly, "rx": nav.rx, "ry": nav.ry],
        ]
        sendEnvelope(type: "teleop_frame", payload: payload)

        // Stream direct joint targets when they change (Body / WBC control).
        if !estopped && jointsDirty {
            jointsDirty = false
            sendEnvelope(type: "joint_command", payload: ["joints": joints])
            // Send only what CHANGED. Left to accumulate, every joint ever targeted rode along on
            // every later command — after one motion clip that was all sixteen finger joints, and
            // the bridge lets explicit finger targets beat the grip, so the grip went dead (2026-10-13).
            joints.removeAll(keepingCapacity: true)
        }
    }

    /// One decoder for both arrival paths — the `demo_clips` reply and the
    /// `robot_state.animation` mirror. Two copies would drift, and the mirror is
    /// the one that keeps a reconnecting client's picker populated.
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

    private func sendEnvelope(type: String, payload: [String: Any], safety: Bool = false) {
        let env: [String: Any] = ["type": type, "payload": payload]
        guard let data = try? JSONSerialization.data(withJSONObject: env),
              let str = String(data: data, encoding: .utf8) else { return }
        guard let socket = task else {
            if safety { onSafetyError?("No robot connection") }
            return
        }
        // Tolerate transient stream errors — the ping / receive paths detect a
        // real drop. A single failed frame must not kill the session.
        socket.send(.string(str)) { [weak self] error in
            if safety, let error {
                DispatchQueue.main.async {
                    guard let self, self.task === socket else { return }
                    self.onSafetyError?(error.localizedDescription)
                }
            }
        }
    }

    private func receive(_ current: URLSessionWebSocketTask? = nil) {
        guard let socket = current ?? task, task === socket else { return }
        socket.receive { [weak self] result in
            guard let self, self.task === socket else { return }
            // Runs on `recvQueue` — NEVER the main thread. The work here must stay
            // cheap (classify + store) so the drain rate always exceeds the 100 Hz
            // arrival rate and no client-side TCP backlog can ever form. (Decoding
            // every message inline on main — the old model — fell behind under UI
            // load and let TCP queue a multi-second FIFO of stale states.)
            switch result {
            case .failure(let err):
                NSLog("[MABEL-WS] receive failure: \(err.localizedDescription)")
                DispatchQueue.main.async {
                    guard self.task === socket, self.wantConnection else { return }
                    self.dropped()
                }
            case .success(let msg):
                self.receive(socket) // never arm a replacement socket from an old completion
                var text: String?
                if case let .string(t) = msg { text = t }
                else if case let .data(d) = msg { text = String(data: d, encoding: .utf8) }
                guard let text else { return }
                self.recvLock.lock()
                self.lastRecv = Date()
                // Coalesce the high-rate robot_state flood to the NEWEST frame
                // (decoded by stateTimer on main); hand rarer control messages
                // (hello/control/pong/map_list) to main immediately so none are
                // ever dropped.
                if text.contains("\"robot_state\"") {
                    self.latestStateText = text
                    self.latestStateTask = socket
                    self.recvLock.unlock()
                } else {
                    self.recvLock.unlock()
                    DispatchQueue.main.async {
                        guard self.task === socket, self.wantConnection else { return }
                        self.handle(text)
                    }
                }
            }
        }
    }

    /// Decode + deliver the newest buffered robot_state (called by `stateTimer` at a
    /// fixed cadence, on main). Intermediate frames are dropped undecoded — the
    /// SceneKit display-link reads the resulting `livePose`, so the model always
    /// renders the robot's CURRENT pose, RViz-like.
    private func flushLatestState() {
        recvLock.lock()
        let t = latestStateText
        let socket = latestStateTask
        latestStateText = nil
        latestStateTask = nil
        recvLock.unlock()
        guard let t, wantConnection, task === socket else { return }
        handle(t)
    }

    private func handle(_ text: String) {
        recvLock.lock(); lastRecv = Date(); recvLock.unlock()   // heartbeat: the link is alive
        if status != .connected { status = .connected; reconnectAttempts = 0; watchdog?.invalidate() }
        guard let data = text.data(using: .utf8),
              let env = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        let type = env["type"] as? String
        if type == "error" {
            let p = env["payload"] as? [String: Any] ?? [:]
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

        // The server greets with `hello` carrying its identity, which tells us
        // whether we're driving the MuJoCo sim bridge or real hardware.
        if type == "hello" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            myConnId = p["you"] as? String          // remember OUR id for the lock UI
            onServerInfo?(p["server"] as? String)
            emitMeta(ServerMeta(from: p))
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

        // Saved-map catalogue (mirrors the web console's `map_list`).
        if type == "demo_clips" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            onDemoClips?(Self.parseClips(p["clips"]))
            return
        }
        if type == "map_list" {
            let p = (env["payload"] as? [String: Any]) ?? [:]
            let maps: [MapInfo] = ((p["maps"] as? [[String: Any]]) ?? []).compactMap {
                guard let name = $0["name"] as? String else { return nil }
                return MapInfo(name: name,
                               hasMesh: ($0["has_mesh"] as? Bool) ?? false,
                               hasReloc: ($0["has_reloc"] as? Bool) ?? false)
            }
            onMapList?(maps, p["loaded"] as? String)
            return
        }

        guard type == "robot_state",
              let p = env["payload"] as? [String: Any] else { return }

        #if DEBUG
        // Same-host QA probe (simulator + loopback bridge share a clock): true
        // one-way staleness of the state we're about to render, logged ~1/s as
        // [MABEL-LAG]. Guards against the receive-buffer-buildup regression.
        // Cross-host the value includes clock skew — ignore it there.
        if let sent = p["sentAt"] as? Double {
            logStaleness(Date().timeIntervalSince1970 - sent)
        }
        #endif

        // The bridge restamps its identity + live speed caps in every state, so a
        // rename / cap change on any client converges here.
        if let srv = p["server"] as? [String: Any] { emitMeta(ServerMeta(from: srv)) }

        var st = RobotState()
        if let bat = p["battery"] as? [String: Any] {
            st.batteryLevel = (bat["percentage"] as? Double) ?? st.batteryLevel
            st.batteryVoltage = (bat["voltage"] as? Double) ?? st.batteryVoltage
            st.charging = (bat["isCharging"] as? Bool) ?? st.charging
            // 2026-07: the server says whether a battery monitor EXISTS —
            // available:false → the gauge shows n/a + reason, not a fake 100%.
            if let avail = bat["available"] as? Bool {
                st.batteryAvailable = avail
                st.batteryReason = (bat["reason"] as? String) ?? ""
            }
        }
        // ── Server-authoritative status echoes (additive keys, 2026-07).
        Self.applySafety(p, to: &st)
        if let cm = p["controlMode"] as? [String: Any] {
            st.controlMethod = (cm["method"] as? String) ?? ""
        }
        if let dc = p["dataCollection"] as? [String: Any], !dc.isEmpty {
            st.dataCollection = RobotState.DataCollection(
                active: (dc["active"] as? Bool) ?? false,
                label: dc["label"] as? String,
                elapsedSec: (dc["elapsedSec"] as? Double) ?? 0,
                error: dc["error"] as? String)
        }
        if let po = p["policy"] as? [String: Any] {
            st.policy = RobotState.PolicyEcho(selected: po["selected"] as? String,
                                              running: (po["running"] as? Bool) ?? false)
        }
        if let an = p["animation"] as? [String: Any] {
            st.animation = RobotState.AnimationEcho(
                selected: an["selected"] as? String,
                playing: (an["playing"] as? Bool) ?? false,
                recording: (an["recording"] as? Bool) ?? false,
                armed: (an["armed"] as? Bool) ?? false,
                countdown: (an["countdown"] as? Double) ?? 0,
                name: an["name"] as? String,
                frames: (an["frames"] as? Int) ?? 0,
                elapsed: (an["elapsed"] as? Double) ?? 0,
                phase: an["phase"] as? String,
                rep: (an["rep"] as? Int) ?? 0,
                repeats: (an["repeats"] as? Int) ?? 0,
                frac: (an["frac"] as? Double) ?? 0,
                clips: Self.parseClips(an["clips"]))
        }
        // Per-module connection roster. SPARSE on the wire — the server only
        // attaches it when it CHANGES (13 rows at 50 Hz would be pure waste), so
        // it is LATCHED in RobotController like lidar / motorHealth. Decoding it
        // as "absent ⇒ empty" here would blink the panel dark between updates.
        // The server owns the order; preserve it (a Swift dict would not).
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
                return RobotState.Module(id: id,
                                         label: (m["label"] as? String) ?? id,
                                         group: m["group"] as? String,
                                         ok: ok,
                                         detail: (m["detail"] as? String) ?? "",
                                         up: (m["up"] as? Int) ?? (acts.isEmpty ? nil : acts.filter(\.ok).count),
                                         n: (m["n"] as? Int) ?? (acts.isEmpty ? nil : acts.count),
                                         actuators: acts)
            }
        }
        if let joints = p["jointPositions"] as? [String: Double] {
            for (k, v) in joints { st.joints[k] = v }
            st.reportedJoints.formUnion(joints.keys)
        }
        // Server-echoed operator switches (absent on older servers → unchanged).
        if let g = p["gravityComp"] as? Bool { st.gravityComp = g }
        if let w = p["wholeBodyRetarget"] as? Bool { st.wholeBodyRetarget = w }
        // Prefer the locally-measured WebSocket round-trip (true link latency) over
        // the server's latencyMs (command staleness). Fall back to the server value
        // only until the first pong lands.
        if rttMs > 0 { st.latencyMs = rttMs }
        else if let lat = p["latencyMs"] as? Double { st.latencyMs = lat }
        if let age = p["stateAgeMs"] as? Double { st.stateAgeMs = age }
        if let rec = p["recording"] as? [String: Any] {
            st.recordTask = rec["task"] as? String ?? ""
            st.episodeCount = rec["episodeCount"] as? Int ?? 0
            st.taskList = rec["tasks"] as? [String] ?? []
            st.taskCounts = rec["taskCounts"] as? [String: Int] ?? [:]
            st.episodesTotal = rec["total"] as? Int ?? 0
            st.episodesDir = rec["dir"] as? String ?? ""
            st.serverRecording = rec["isRecording"] as? Bool ?? false
        }
        // Sparse LiDAR: the bridge attaches a sweep only when a fresh one exists
        // (~5 Hz), so most states carry none; the controller latches the last
        // one. Ranges may arrive as Double or Int (JSON), normalize both.
        if let li = p["lidar"] as? [String: Any],
           let ranges = li["ranges"] as? [Any] {
            var scan = RobotState.LidarScan()
            scan.angleMin = li["angleMin"] as? Double ?? 0
            scan.angleInc = li["angleInc"] as? Double ?? 0
            scan.rangeMax = li["rangeMax"] as? Double ?? 12
            scan.stamp = li["stamp"] as? Double ?? 0
            scan.ranges = ranges.map { ($0 as? Double) ?? Double($0 as? Int ?? 0) }
            if !scan.ranges.isEmpty, scan.angleInc != 0 { st.lidar = scan }
        }
        // Sparse joint dynamics (~10 Hz, stamp-gated server-side like lidar):
        // streamed velocities + actuator torques (+ temps when the real HAL
        // measures them). Most states carry none; the controller latches.
        if let v = p["jointVelocities"] as? [String: Double] { st.jointVelocities = v }
        if let tq = p["jointTorques"] as? [String: Double] { st.jointTorques = tq }
        if let tm = p["jointTemps"] as? [String: Double] { st.jointTemps = tm }
        // motorHealth fallback (older bridges): per-joint {torqueNm, tempC}
        // (arms, DaMiao MIT feedback) / {loadPct, tempC, voltage} (hands).
        // Torque only when it's actually N·m; temps from either shape.
        if let mh = p["motorHealth"] as? [String: Any] {
            var tor: [String: Double] = [:], tmp: [String: Double] = [:]
            for (joint, v) in mh {
                guard let h = v as? [String: Any] else { continue }
                if let tau = h["torqueNm"] as? Double { tor[joint] = tau }
                if let t = h["tempC"] as? Double { tmp[joint] = t }
            }
            if !tor.isEmpty { st.healthTorques = tor }
            if !tmp.isEmpty { st.healthTemps = tmp }
        }
        // Canonical base state: chassis heading (odom) + swerve steer angles, so
        // the model turns/steers with the real robot.
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
        onState?(st)
    }

    /// Surface server identity/caps only when they actually change (the meta
    /// rides every 20 Hz state, so dedupe to avoid spamming the UI).
    private func emitMeta(_ m: ServerMeta) {
        var m = m
        if m.kind != .unknown { serverKind = m.kind }   // pick the mapping for this robot
        m.you = myConnId                                 // so the UI knows if WE hold the lock
        guard m != lastMeta else { return }
        lastMeta = m
        onServerMeta?(m)
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

    static func applySafety(_ p: [String: Any], to state: inout RobotState) {
        state.estop = p["estop"] as? Bool
        state.stopUnconfirmed = p["stopUnconfirmed"] as? [String]
        if let torque = p["torque"] as? [String: Any] {
            state.torqueState = torque.compactMapValues { $0 as? Bool }
            if let modules = torque["modules"] as? [String: [String: Any]] {
                state.moduleTorque = modules.compactMapValues { $0["torque"] as? Bool }
            }
        }
    }

    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol proto: String?) {
        NSLog("[MABEL-WS] didOpen")
        DispatchQueue.main.async {
            guard self.wantConnection, self.task === webSocketTask else { return }
            self.status = .connected; self.reconnectAttempts = 0; self.watchdog?.invalidate()
        }
    }
    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith code: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        NSLog("[MABEL-WS] didClose code=\(code.rawValue) reason=\(reason.flatMap { String(data: $0, encoding: .utf8) } ?? "-")")
        DispatchQueue.main.async {
            guard self.wantConnection, self.task === webSocketTask else { return }
            self.dropped()
        }
    }
    // Lower-level URLSession task failures (TLS / connection resets / the server's
    // "Can't assign requested address" on a flaky link-local address). These don't
    // always surface as a receive() failure, so reconnect from here too.
    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        NSLog("[MABEL-WS] task completed error: \(error.localizedDescription) → reconnect")
        DispatchQueue.main.async {
            guard self.wantConnection, self.task === task else { return }
            self.dropped()
        }
    }
}
