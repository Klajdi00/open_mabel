//
//  BLELink.swift
//  MabelTeleop
//
//  A `RobotLink` that drives the robot over Bluetooth LE instead of Wi-Fi — teleoperation
//  with no network at all: no router, no hotspot, no internet. Same app, same controls.
//
//  It is a TRANSPORT SWAP, nothing more. Commands are packed by `BLEProtocol` and written
//  to the robot's `command` characteristic, where NOVA's ble_bridge decodes them into the
//  exact wire messages the WebSocket carries and hands them to the same gateway. So the
//  arms, the base, the lift, the torso, the neck and e-stop all behave identically — there
//  is one control path on the robot, and this is another way onto it.
//
//  WHAT BLUETOOTH CANNOT CARRY: video. iOS negotiates ~15-30 ms connection intervals at
//  ATT_MTU ~185, so roughly 5-20 kB/s; the head camera wants ~5 Mbit/s, about 300× more.
//  The UI therefore shows no camera on a BLE link, by design, and says so rather than
//  spinning on a feed that can never arrive.
//
//  WRITE-WITHOUT-RESPONSE is used for the streaming pose and twist: a 20 Hz pose does not
//  need acknowledging, and waiting for one would serialise every tick onto the connection
//  interval. E-stop is the exception — it goes out WITH response, on its own characteristic,
//  because the one message that must not be silently dropped is the one that stops the robot.
//

import Foundation
import CoreBluetooth

/// NOT `@MainActor`, deliberately: `RobotLink` is a non-isolated protocol, and both other
/// links (WebSocketLink, DemoLink) are plain classes. Every callback here already arrives on
/// the main queue because `BLECentral` creates its `CBCentralManager` with `queue: .main`,
/// so isolation would buy nothing and break the conformance.
final class BLELink: NSObject, RobotLink {

    // MARK: RobotLink

    private(set) var status: ConnectionStatus = .disconnected {
        didSet { if status != oldValue { onStatus?(status) } }
    }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((String, String?, String) -> Void)?
    var onMapList: (([MapInfo], String?) -> Void)?
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?

    /// Commands Bluetooth cannot express (maps, policy dispatch…). Surfaced so the UI can
    /// explain the gap instead of appearing broken. Recording, replay and Home DO ride the link.
    private(set) var lastUnsupported: String?

    // MARK: State

    let robotID: UUID
    let robotName: String

    private let central: CBCentralManager
    private var peripheral: CBPeripheral?
    private var cmdChar: CBCharacteristic?
    private var estopChar: CBCharacteristic?
    private var stateChar: CBCharacteristic?
    private var infoChar: CBCharacteristic?

    /// Seeded with what this link CANNOT report. The packed frame is pose + estop +
    /// torque; battery, CPU and latency have no source here, so they are marked absent
    /// rather than left at RobotState's defaults for the panel to render as readings.
    private(set) var state: RobotState = {
        var s = RobotState()
        s.vitalsAvailable = false
        return s
    }()
    /// Readable (not writable) outside this type so a test can assert WHAT would go on the
    /// wire. The lift's units were wrong here for the life of this link and nothing could
    /// see it; `LiftUnitsOverBluetoothTests` now can.
    private(set) var joints: [String: Double] = [:]
    private var jointsDirty = false
    private var twist = (vx: 0.0, vy: 0.0, wz: 0.0)
    private var twistDirty = false
    /// The lift channel of the drive joystick, -1…1. Rides the teleop frame; the gateway
    /// expires it after 0.5 s of silence, which is the deadman.
    private(set) var liftStick: Double = 0
    private var estopped = false
    /// The app's own memory of the stiffness toggle: the robot echoes `gravityComp` back
    /// but not stiffness, so CONTROL_MODE has to be rebuilt from what we last sent.
    private var softMode = false
    /// The commanded column, in METRES PER STAGE — the units `joints` carries for every
    /// other joint and the units the Wi-Fi link sends. `nil` until a state frame has said
    /// where the column actually is.
    ///
    /// THIS WAS A 0…1 FRACTION written straight into a metres field, starting at 0.5. One
    /// stick touch therefore asked for 0.5 m PER STAGE — a metre of column against 0.635 m
    /// of travel — so the lift ran to its ceiling and sat there, and the rate integrated at
    /// 0.6 fraction/s ≈ 381 mm/s against a column that does 35. It moved nothing while the
    /// gateway was still dropping lift joint commands; the moment the gateway started
    /// honouring them it surfaced as "switching to Bluetooth drives the lift up".
    private var liftPerStage: Double?
    /// Half the column's travel: the two stages are commanded together and their targets SUM.
    private static let liftStageMax = 0.3175
    /// 35 mm/s of column is the measured plant limit, so 17.5 mm/s per stage. The old
    /// integrator asked for eleven times that.
    private static let liftMaxVelPerStage = 0.0175
    /// Matches `RealPlant._LIFT_DEADBAND` on the robot: below this a stick is at rest.
    private static let liftRateDeadband = 0.05
    /// Matches the gateway's 0.5 s twist deadman: a stick that stops reporting is released.
    private static let liftRateStale = 0.5
    private var liftMoving = false
    private var lastLiftRateAt = 0.0
    /// Has the robot actually told us a pose yet? `RobotState.joints` is PRE-FILLED with
    /// `RobotModel.homePose`, so "nothing has arrived" and "the column is at mid-stroke"
    /// read identically off `state` — seeding from it unchecked would adopt a drawn default
    /// as if it were a measurement, which is the very mistake this file is fixing.
    private var sawState = false
    private var pump: Timer?

    /// 20 Hz uplink. Fast enough that a joystick feels direct, slow enough to sit inside
    /// iOS's connection interval without queueing — the queue IS the latency on BLE.
    private let uplinkHz = 20.0
    /// The headset's head + hands, pulled at send time like the Wi-Fi link does (RobotController
    /// sets it). One hand packet per tick, hands alternating, after the joints/twist packet — so
    /// with both hands tracked each streams at 10 Hz, one hand at 20 Hz; the robot smooths.
    var frameSource: (() -> HeadsetFrame?)?
    private var handTurn = 0
    /// Whole-body teleop over Bluetooth (2026-10-13): the control spec's method, carried in CONTROL_MODE,
    /// and the headset's uptime clock for the WRIST_POSE timestamp (the robot's engine dedupes frames on it).
    private var method = "navigation"
    private let clockT0 = ProcessInfo.processInfo.systemUptime
    /// CLIPS pages of the current DEMO_LIST reply, collected until every page is in.
    private var clipPages: [Int: [RobotState.DemoClipInfo]] = [:]

    convenience init(robot: NearbyRobot, central: CBCentralManager) {
        self.init(id: robot.id, name: robot.name, central: central)
    }

    /// Connect from an identifier alone — a robot REMEMBERED from a previous session, with
    /// nothing on the air yet. `connect()` below resolves it with
    /// `retrievePeripherals(withIdentifiers:)`, which needs no scan and no advertisement:
    /// CoreBluetooth already knows this peripheral because this app connected to it before.
    /// That is the whole reason Settings can offer a saved robot a Connect button.
    /// WHICH robot, from its advert's robot byte (BLEProtocol.Beacon.isNova) — set by the
    /// caller, which can see the scan. nil (a saved robot not on the air yet) ⇒ the info
    /// characteristic's `robot` decides once it is read.
    var robotHint: String?
    private var bleServer: String { robotHint?.uppercased() == "MABEL" ? "mabel_ble_bridge" : "nova_ble_bridge" }

    init(id: UUID, name: String, central: CBCentralManager) {
        self.robotID = id
        self.robotName = name
        self.central = central
        super.init()
    }

    // MARK: Lifecycle

    func connect() {
        guard let p = central.retrievePeripherals(withIdentifiers: [robotID]).first else {
            status = .disconnected
            return
        }
        peripheral = p
        p.delegate = self
        status = .connecting
        central.connect(p, options: nil)
        NotificationCenter.default.addObserver(
            self, selector: #selector(didConnectNote(_:)), name: .bleDidConnect, object: nil)
        NotificationCenter.default.addObserver(
            self, selector: #selector(didDisconnectNote(_:)), name: .bleDidDisconnect, object: nil)
    }

    func disconnect() {
        pump?.invalidate(); pump = nil
        NotificationCenter.default.removeObserver(self)
        if let p = peripheral { central.cancelPeripheralConnection(p) }
        peripheral = nil; cmdChar = nil; estopChar = nil; stateChar = nil; infoChar = nil
        status = .disconnected
    }

    @objc private func didConnectNote(_ note: Notification) {
        guard let p = note.object as? CBPeripheral, p.identifier == robotID else { return }
        peripheral = p
        p.delegate = self
        p.discoverServices([CBUUID(string: BLEProtocol.UUIDs.service)])
    }

    @objc private func didDisconnectNote(_ note: Notification) {
        guard let p = note.object as? CBPeripheral, p.identifier == robotID else { return }
        pump?.invalidate(); pump = nil
        cmdChar = nil; estopChar = nil; stateChar = nil
        liftMoving = false
        // RECONNECT ON ITS OWN (2026-10-24, "the signal keeps cutting off"): a drop that we did not ask for
        // (disconnect() clears `peripheral` and stops observing first) re-issues the connect. CoreBluetooth
        // keeps a pending connect alive with no timeout and completes it the moment the robot is heard
        // again; didConnectNote then rediscovers the service and the pump restarts. Before, one dip in the
        // signal left the link dead until the operator reconnected by hand. The robot holds still while no
        // frames arrive (the gateway's deadmen), exactly as during a Wi-Fi reconnect.
        if let held = peripheral, held.identifier == p.identifier {
            status = .connecting
            // half a second first: a failed connect posts this same note, and must not spin
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                guard let self, let q = self.peripheral, q.identifier == p.identifier, q.state != .connected else { return }
                self.central.connect(q, options: nil)
            }
        } else {
            status = .disconnected
        }
    }

    private func startPump() {
        pump?.invalidate()
        // Scheduled on the main run loop, so it fires on the same queue every callback
        // here arrives on — no hop needed.
        pump = Timer.scheduledTimer(withTimeInterval: 1.0 / uplinkHz, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    /// One uplink tick. Sends ONLY what changed: on a link this narrow, re-sending an
    /// unchanged pose 20×/s would spend the whole budget saying nothing.
    private func tick() {
        guard let p = peripheral, let c = cmdChar, p.state == .connected else { return }
        // The gateway stops the column when the stick stops reporting for 0.5 s. Nothing
        // enforced that on this link, so a stream that simply ENDED — the view torn down
        // mid-press, the app backgrounded — left the last target latched and the column
        // travelling.
        if liftMoving, ProcessInfo.processInfo.systemUptime - lastLiftRateAt > Self.liftRateStale,
           let measured = measuredPerStage() {
            stopLift(at: measured)
        }
        // Respect the link's own backpressure. Queueing past this is what turns a BLE
        // control link from "direct" into "half a second behind".
        guard p.canSendWriteWithoutResponse else { return }

        if jointsDirty, let frame = BLEProtocol.encodeJointCommand(joints) {
            p.writeValue(frame, for: c, type: .withoutResponse)
            jointsDirty = false
            return                                  // one packet per tick; twist takes the next
        }
        if twistDirty {
            let frame = BLEProtocol.encodeTeleopFrame(vx: twist.vx, vy: twist.vy,
                                                      wz: twist.wz, lift: liftStick)
            p.writeValue(frame, for: c, type: .withoutResponse)
            // Keep streaming a non-zero twist OR a held lift stick: the gateway expires
            // both after 0.5 s as a deadman, so "still driving" has to keep being said.
            twistDirty = twist.vx != 0 || twist.vy != 0 || twist.wz != 0 || liftStick != 0
        }
        // The headset's head, wrists and hands ride the same link: WRIST_POSE (108 bytes: head pose, both
        // wrists' world poses, elbows, timestamp) every tick anything is tracked — the robot's whole-body arm
        // retargeting and its neck run on it exactly as on a Wi-Fi frame — then one tracked hand per tick
        // (HAND_POSE, 129 bytes) for the fingers. ~5 kB/s at 20 Hz, inside the link's budget.
        guard !estopped, p.canSendWriteWithoutResponse, let hf = frameSource?() else { return }
        let ms = UInt32(truncatingIfNeeded: Int((ProcessInfo.processInfo.systemUptime - clockT0) * 1000))
        if let frame = BLEProtocol.encodeWristPose(hf, timestampMs: ms) {
            p.writeValue(frame, for: c, type: .withoutResponse)
        } else if hf.head.trackingState == .normal, let frame = BLEProtocol.encodeHeadPose(hf.head) {
            p.writeValue(frame, for: c, type: .withoutResponse)
        }
        guard p.canSendWriteWithoutResponse else { return }
        let hands = [hf.leftHand, hf.rightHand].compactMap { $0 }.filter { $0.isTracked && !$0.joints.isEmpty }
        guard !hands.isEmpty else { return }
        handTurn = (handTurn + 1) % hands.count
        if let frame = BLEProtocol.encodeHandPose(hands[handTurn]) {
            p.writeValue(frame, for: c, type: .withoutResponse)
        }
    }

    /// Take a decoded state frame. Split out of the characteristic callback so this path can
    /// be exercised without a live radio — the lift's units were wrong HERE, on the only side
    /// of the link a unit test could never reach. visionOS's RobotState carries no
    /// estop/torque fields (the controller owns those), so they are accepted and ignored.
    func applyStateFrame(positions: [String: Double], estop: Bool, torque: Bool) {
        state.timestamp = Date().timeIntervalSince1970
        for (k, v) in positions { state.joints[k] = v }
        state.reportedJoints.formUnion(positions.keys)
        sawState = true
        if liftPerStage == nil { liftPerStage = measuredPerStage() }
    }

    /// A RECORDING notify → the same RobotState fields the Wi-Fi link fills from robot_state.recording / .animation.
    func applyRecording(_ r: BLEProtocol.Recording) {
        state.serverRecording = r.isRecording
        state.recordingReported = true
        state.recordTask = r.task
        state.episodeCount = r.episodeCount
        state.episodesTotal = r.total
        state.recordElapsed = r.elapsed
        state.recordError = r.error
        state.recordCameras = r.cameras
        var demo = state.demo ?? .init()
        demo.recording = r.isRecording
        demo.playing = r.replayPlaying
        demo.name = r.replayPlaying ? r.replayName : (r.isRecording ? r.task : nil)
        demo.elapsed = r.elapsed
        demo.phase = r.phase.isEmpty ? nil : r.phase
        demo.rep = r.rep; demo.repeats = r.repeats; demo.frac = r.frac
        state.demo = demo
    }

    /// The lift stick, integrated to a height — and integrated to mean EXACTLY what the
    /// Wi-Fi link means by the same stick.
    ///
    /// On Wi-Fi the app sends the raw stick and the ROBOT integrates it
    /// (`RealPlant.set_lift_rate`). Bluetooth has no rate on the wire — the packed teleop
    /// frame carries a twist only — so the integration has to happen here, and it has to
    /// reproduce the robot's rules or the same control means two different things depending
    /// on how the phone happens to be connected:
    ///
    ///   • `|r| >= 0.05` advances the target at the column's own maximum speed, seeded from
    ///     the MEASURED height on the first tick of a press so it never steps.
    ///   • `|r| < 0.05` is a **stop**, not "no change": it commands where the column actually
    ///     is, which is what the robot does with a released stick. `stopBase()` sends exactly
    ///     this when teleop is switched off — treating it as a no-op is what let an
    ///     already-latched target keep driving the column into its ceiling after the operator
    ///     had stopped. On Wi-Fi the robot halted; on Bluetooth nothing did.
    ///   • no rate for 0.5 s while moving is also a stop — the gateway's own deadman, which
    ///     otherwise has no equivalent on this link (see `tick()`).
    private func applyLiftRate(_ r: Double) {
        lastLiftRateAt = ProcessInfo.processInfo.systemUptime
        guard let measured = measuredPerStage() else { return }   // nowhere safe to command
        guard abs(r) >= Self.liftRateDeadband else { return stopLift(at: measured) }
        let base = liftMoving ? (liftPerStage ?? measured) : measured
        liftMoving = true
        let v = min(Self.liftStageMax, max(0, base + r * Self.liftMaxVelPerStage / uplinkHz))
        liftPerStage = v
        joints["lift_lower"] = v; joints["lift_upper"] = v
        jointsDirty = true
    }

    /// Age the lift stream past its deadman, so the stop path can be exercised without
    /// waiting half a second of wall clock in a test.
    func expireLiftRateForTesting() {
        lastLiftRateAt -= Self.liftRateStale * 2
        if liftMoving, let m = measuredPerStage() { stopLift(at: m) }
    }

    /// Halt the column where it is by commanding its measured height — the robot's `stop()`,
    /// expressed in the only language this link speaks (positions).
    private func stopLift(at measured: Double) {
        guard liftMoving else { return }
        liftMoving = false
        liftPerStage = measured
        joints["lift_lower"] = measured; joints["lift_upper"] = measured
        jointsDirty = true
    }

    /// The column as the robot last reported it, in metres per stage. Both stages carry the
    /// same height, so either one answers.
    private func measuredPerStage() -> Double? {
        guard sawState else { return nil }
        return state.joints["lift_lower"] ?? state.joints["lift_upper"]
    }

    // MARK: Commands

    func send(_ command: RobotCommand) {
        switch command {
        case let .twist(f, s, y):
            guard !estopped else { twist = (0, 0, 0); twistDirty = true; return }
            twist = (f, s, y); twistDirty = true

        case let .jointTarget(id, value):
            guard !estopped else { return }
            joints[id] = value; jointsDirty = true

        case let .liftFraction(frac):
            // The lift's two stages share the column travel, as on the Wi-Fi link — and, as
            // on the Wi-Fi link, they are commanded in METRES, so a fraction is scaled here.
            let v = min(Self.liftStageMax, max(0, frac * Self.liftStageMax))
            liftPerStage = v
            joints["lift_lower"] = v; joints["lift_upper"] = v; jointsDirty = true

        case let .liftRate(r):
            // THE STICK GOES AS A RATE, exactly as it does over Wi-Fi. It used to be
            // integrated here into a position, because the packed format carried positions
            // only — and a position is the one thing the column cannot be asked for slowly:
            // it is a bang-bang servo, so a slowly-advancing target is answered with full
            // duty, an overshoot, a cut and a repeat. That is the jitter felt on a
            // half-pushed stick. The robot now turns the rate into open-loop duty
            // (server/plant.py jog_effort), which is smooth at any deflection.
            //
            // It also retires a whole class of bug for free: a RATE cannot carry the
            // column to a remembered absolute height, so there is nothing for a transport
            // switch or a stale target to step.
            guard !estopped else { liftStick = 0; twistDirty = true; return }
            liftStick = max(-1, min(1, r))
            twistDirty = true

        case let .estop(on):
            estopped = on
            if on {
                twist = (0, 0, 0); liftStick = 0
                joints.removeAll(); jointsDirty = false
                twistDirty = true            // say the zero once, do not just stop talking
            }
            writeEstop(on)

        case let .controlSpec(_, m, type, _, stiffness, _):
            // The packed CONTROL_MODE carries the axes that change how the robot FEELS — soft (compliant,
            // floating) vs stiff (position-holding), gravity compensation — and, since 2026-10-13, the
            // METHOD: "retarget" is what makes the robot's arms follow the wrists (Engage whole body).
            softMode = type == "compliance" || stiffness < 0.5
            method = m
            writeMode(soft: softMode, gravityComp: state.gravityComp ?? true)

        case let .startBodyCalibration(user):
            writeWithResponse(BLEProtocol.encodeBodyCalibration(action: "start", user: user))
        case .cancelBodyCalibration:
            writeWithResponse(BLEProtocol.encodeBodyCalibration(action: "cancel", user: ""))
        case let .applyBodyCalibration(user):
            writeWithResponse(BLEProtocol.encodeBodyCalibration(action: "apply", user: user))
        case .recalibrate:
            writeWithResponse(BLEProtocol.encodeRecalibrate())

        case let .gravityComp(on):
            state.gravityComp = on
            writeMode(soft: softMode, gravityComp: on)

        case .goHome:
            // The robot's own Home glide (the wire's `reset`), exactly as over Wi-Fi — not a pose step from here.
            writeWithResponse(BLEProtocol.encodeHome())

        // DATA COLLECTION with no network: the robot records every camera and tick on its own disk; this link
        // carries the button, and the RECORDING notify carries the status back.
        case let .record(active, task):
            writeWithResponse(BLEProtocol.encodeRecord(action: active ? "start" : "stop", task: task))
        case let .selectTask(task):
            writeWithResponse(BLEProtocol.encodeRecord(action: "select", task: task))
        case let .demoRecord(active, name, _):
            // NOVA has no separate clip recorder: a demo take is a dataset episode named after the clip
            writeWithResponse(BLEProtocol.encodeRecord(action: active ? "start" : "stop", task: name))
        case let .demoReplay(active, name, repeats):
            writeWithResponse(BLEProtocol.encodeDemoReplay(active: active, name: name, repeats: repeats))
        case .demoList:
            clipPages = [:]
            writeWithResponse(BLEProtocol.encodeDemoList())

        // (visionOS routes head tracking through `controlSpec(tracking:)`, not a
        // case of its own — the iOS enum has `headTracking`, this one does not.)
        case .headPose:
            // Head retargeting streams a 4x4 transform per tick on Wi-Fi. That is 64 bytes
            // of float on a link whose whole budget is one 102-byte packet per tick, and it
            // would crowd out the arms. Drive the neck with jointTarget on Bluetooth.
            lastUnsupported = "head tracking"

        default:
            lastUnsupported = "\(command)"
        }
    }

    private func writeEstop(_ on: Bool) {
        guard let p = peripheral, let c = estopChar ?? cmdChar else { return }
        // WITH response, always: this is the message that must not be quietly dropped.
        p.writeValue(BLEProtocol.encodeEstop(on), for: c, type: .withResponse)
    }

    private func writeMode(soft: Bool, gravityComp: Bool) {
        guard let p = peripheral, let c = cmdChar else { return }
        p.writeValue(BLEProtocol.encodeControlMode(soft: soft, gravityComp: gravityComp, method: method),
                     for: c, type: .withResponse)
    }

    /// The one-shot control messages (calibration, recentre): WITH response, like the mode — these must land.
    private func writeWithResponse(_ frame: Data?) {
        guard let frame = frame, let p = peripheral, let c = cmdChar else { return }
        p.writeValue(frame, for: c, type: .withResponse)
    }
}

// MARK: - Peripheral delegate

extension BLELink: CBPeripheralDelegate {

    func peripheral(_ p: CBPeripheral, didDiscoverServices error: Error?) {
        guard let svc = p.services?.first(where: {
            $0.uuid == CBUUID(string: BLEProtocol.UUIDs.service) }) else { return }
        p.discoverCharacteristics(nil, for: svc)
    }

    func peripheral(_ p: CBPeripheral, didDiscoverCharacteristicsFor service: CBService,
                                error: Error?) {
        let chars = service.characteristics ?? []
        do {
            for c in chars {
                switch c.uuid.uuidString.uppercased() {
                case BLEProtocol.UUIDs.command: cmdChar = c
                case BLEProtocol.UUIDs.estop:   estopChar = c
                case BLEProtocol.UUIDs.info:    infoChar = c; p.readValue(for: c)
                case BLEProtocol.UUIDs.state:
                    stateChar = c
                    p.setNotifyValue(true, for: c)
                default: break
                }
            }
            guard cmdChar != nil else { return }
            status = .connected
            onServerInfo?(bleServer)
            var meta = ServerMeta()
            meta.server = bleServer
            meta.robot = robotHint
            meta.kind = .real
            meta.name = robotName
            meta.transport = "Bluetooth"
            onServerMeta?(meta)
            startPump()
        }
    }

    func peripheral(_ p: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic,
                                error: Error?) {
        guard error == nil, let data = characteristic.value else { return }
        let uuid = characteristic.uuid.uuidString.uppercased()
        do {
            if uuid == BLEProtocol.UUIDs.state {
                // TWO frame types share this characteristic, told apart by their first
                // byte: the pose (hot path) and, once a second, the module roster. The
                // roster is what lets the status panel say "5/7, J3 and J6 silent" over
                // Bluetooth instead of sitting on "no data yet" — which is what it did
                // for the whole life of this link.
                if data.first == BLEProtocol.MessageType.health.rawValue {
                    guard let mods = BLEProtocol.decodeHealth(data) else { return }
                    state.modules = mods
                    onState?(state)
                    return
                }
                // The calibration card's two fields, sent when they change — the same
                // CalibrationProgress / OperatorMetrics the Wi-Fi robot_state carries.
                // The recorder + replay status: the Rec button, the episode count and every camera's frames.
                if data.first == BLEProtocol.MessageType.recording.rawValue {
                    guard let r = BLEProtocol.decodeRecording(data) else { return }
                    applyRecording(r)
                    onState?(state)
                    return
                }
                if data.first == BLEProtocol.MessageType.clips.rawValue {
                    guard let c = BLEProtocol.decodeClips(data) else { return }
                    clipPages[c.page] = c.clips
                    if clipPages.count >= c.pages {
                        onDemoClips?((0..<c.pages).flatMap { clipPages[$0] ?? [] })
                        clipPages = [:]
                    }
                    return
                }
                if data.first == BLEProtocol.MessageType.calibration.rawValue {
                    guard let cal = BLEProtocol.decodeCalibration(data) else { return }
                    state.calibration = cal.0
                    if let m = cal.1 { state.operatorMetrics = m }
                    onState?(state)
                    return
                }
                guard let s = BLEProtocol.decodeState(data) else { return }
                applyStateFrame(positions: s.positions, estop: s.estop, torque: s.torque)
                // visionOS RobotState has no estop/torque fields; the controller owns
                // that state on this platform, so the pose is all we apply.
                // No camera, no lidar, no battery telemetry on this link — the packed
                // frame carries pose and safety flags only, which is the whole point.
                onState?(state)
            } else if uuid == BLEProtocol.UUIDs.info {
                guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                else { return }
                var meta = ServerMeta()
                // Identity from the bridge's info JSON (robot + mounted hands), not from its
                // NAME: a renamed robot used to lose the "nova" match and draw the wrong rig.
                if let r = obj["robot"] as? String { robotHint = r }
                meta.server = bleServer
                meta.robot = robotHint
                meta.hands = obj["hands"] as? [String: String]
                meta.kind = .real
                meta.name = obj["name"] as? String ?? robotName
                meta.transport = "Bluetooth"
                onServerMeta?(meta)
            }
        }
    }

    func peripheralIsReady(toSendWriteWithoutResponse p: CBPeripheral) {
        tick()
    }
}
