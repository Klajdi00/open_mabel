//
//  BLEProtocol.swift
//  MabelTeleop
//
//  The compact binary teleop frame the app exchanges with the robot over Bluetooth LE.
//  EXACT mirror of NOVA/server/ble_protocol.py — change both together, never one.
//
//  JSON is what every other transport speaks, but it does not fit here. iOS negotiates an
//  ATT_MTU of ~185 bytes, and `{"left_arm_1": -0.123, …}` for 20 joints is ~400 bytes: two
//  or three fragmented writes per control tick, which is exactly the latency Bluetooth can
//  least afford. Packed, a whole-body pose is 102 bytes — one write, every tick.
//
//      byte 0        message type
//      0x01  JOINT_COMMAND   u8 count, then count × (u8 jointIndex, f32 radians)
//      0x02  TELEOP_FRAME    f32 vx, f32 vy, f32 wz      (base twist, m/s + rad/s)
//      0x03  ESTOP           u8 active (1 = engage, 0 = release)
//      0x04  CONTROL_MODE    u8 flags (bit0 soft, bit1 gravityComp) [, u8 method: 0 navigation, 1 joint, 2 retarget]
//      0x07  WRIST_POSE      whole-body teleop (2026-10-13): u8 flags (bit0 left wrist, bit1 right wrist, bit2 head tracked),
//                            u32 timestamp ms, head / left wrist / right wrist each f32 x,y,z + 9 × i16 rotation (row-major,
//                            ×1e-4), then each elbow (ARKit forearm_arm) as 3 × i16 relative to its wrist in 0.02 mm — 108 bytes.
//                            The robot joins it with the latest HAND_POSE per side into the same teleop_frame Wi-Fi sends.
//      0x08  BODY_CALIBRATION u8 action (0 start, 1 cancel, 2 apply), then the operator name (utf-8)
//      0x09  RECALIBRATE     (no body)
//      0x0A  RECORD          u8 action (0 stop, 1 start, 2 select), then the task name (utf-8) — the robot records locally
//      0x0B  DEMO_REPLAY     u8 active, u8 repeats, then the clip name (utf-8)
//      0x0C  DEMO_LIST       (no body) → CLIPS notifications
//      0x0D  HOME            (no body) the robot glides to its rest pose (the wire's `reset`)
//      0x81  STATE (notify)  u8 flags, u8 count, count × (u8 jointIndex, f32 radians)
//      0x83  CALIBRATION (notify) u8 flags (bit0 active, bit1 done), f32 reach, f32 humanReach, f32 robotReach, operator name
//      0x84  RECORDING (notify) u8 flags (bit0 recording, bit1 finalizing, bit2 error, bit3 replay playing), u16 episodeCount,
//                            u16 total, f32 elapsed, u16 dropped, u8 rep, u8 repeats, u8 frac×200, u8 phase, u8 n + n × (u8 camera,
//                            u8 flags (bit0 frames arriving, bit1 error), u32 frames), then u8-length strings: task, replay name, error
//      0x85  CLIPS (notify)  u8 page, u8 pages, u8 n, n × (u16 duration×10, u8 takes, u8 len, name)
//
//  All floats are little-endian f32. Joint INDEX, not name: an index costs 1 byte where
//  "left_arm_1" costs 10, which is the whole reason this format exists.
//

import Foundation

enum BLEProtocol {

    /// CANONICAL JOINT ORDER — the index IS the wire identity. APPEND ONLY.
    /// Reordering or renumbering this makes an older build of the app silently drive the
    /// WRONG joint on the robot, with torque on. Never do it; add to the end instead.
    static let joints: [String] = [
        "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
        "left_arm_5", "left_arm_6", "left_arm_7",
        "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4",
        "right_arm_5", "right_arm_6", "right_arm_7",
        "torso",
        "neck_yaw", "neck_pitch", "neck_roll",
        "lift_lower", "lift_upper",
        // 20+: the hands (2026-09-22) — grips (0..1, expanded by the robot), wrists, 16 finger joints per hand.
        // Command-only over Bluetooth: the robot's STATE keeps to the first 20 (an ATT_MTU cannot carry 54 joints).
        "left_grip", "right_grip", "left_wrist", "right_wrist",
        "left_thumb_mcp", "left_thumb_abd", "left_thumb_pip", "left_thumb_dip",
        "left_index_abd", "left_index_mcp", "left_index_pip", "left_middle_abd",
        "left_middle_mcp", "left_middle_pip", "left_ring_abd", "left_ring_mcp",
        "left_ring_pip", "left_pinky_abd", "left_pinky_mcp", "left_pinky_pip",
        "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
        "right_index_abd", "right_index_mcp", "right_index_pip", "right_middle_abd",
        "right_middle_mcp", "right_middle_pip", "right_ring_abd", "right_ring_mcp",
        "right_ring_pip", "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
    ]

    static let index: [String: UInt8] = {
        var m = [String: UInt8]()
        for (i, n) in joints.enumerated() { m[n] = UInt8(i) }
        return m
    }()

    enum MessageType: UInt8 {
        case jointCommand = 0x01
        case handPose = 0x05
        case headPose = 0x06
        case teleopFrame  = 0x02
        case estop        = 0x03
        case controlMode  = 0x04
        case wristPose    = 0x07
        case bodyCalibration = 0x08
        case recalibrate  = 0x09
        case state        = 0x81
        case health       = 0x82
        case calibration  = 0x83
        case record       = 0x0A
        case demoReplay   = 0x0B
        case demoList     = 0x0C
        case home         = 0x0D
        case recording    = 0x84
        case clips        = 0x85
    }

    // MARK: - Module health (server/ble_protocol.py encode_health)

    /// CANONICAL MODULE ORDER — the index IS the wire identity, exactly like `joints`.
    /// APPEND ONLY, and keep in step with `module_roster.MODULES` on the robot.
    static let moduleIds = ["base", "torso", "lift", "neck", "leftArm", "rightArm",
                            "leftHand", "rightHand", "camHead", "camWristLeft", "camWristRight"]

    /// Display names for the rows. Over Wi-Fi the server owns these; BLE cannot afford to
    /// send them, so the app supplies the same words from the same table.
    static let moduleLabels: [String: String] = [
        "base": "Base", "torso": "Torso", "lift": "Lift", "neck": "Neck",
        "leftArm": "Left Arm", "rightArm": "Right Arm",
        "leftHand": "Left Hand", "rightHand": "Right Hand", "camHead": "Head Cam",
        "camWristLeft": "L Wrist Cam", "camWristRight": "R Wrist Cam",
    ]

    /// MOTOR NAMES, DERIVED. The health frame carries counts and one bit per motor — not
    /// names, which is the trade that makes it fit one BLE write. These are the same names
    /// `module_roster.actuator_label()` produces, reconstructed from the module and the
    /// count, so the panel reads identically on either transport. The hand is the one
    /// exception: its seventeen joints have no derivable order here, so they are numbered.
    static func actuatorLabels(module: String, count: Int) -> [String] {
        guard count > 0 else { return [] }
        switch module {
        case "leftArm", "rightArm": return (1...count).map { "J\($0)" }
        case "torso":               return ["Torso"]
        case "lift":                return ["Column"]
        case "camHead":             return ["Stereo"]
        case "camWristLeft", "camWristRight": return ["D405"]
        case "neck":                return Array(["Yaw", "Pitch", "Roll"].prefix(count))
        case "base":
            return (0..<count).map { i in
                let corner = ["FL", "FR", "RL", "RR"][min(i / 2, 3)]
                return "\(corner) \(i % 2 == 0 ? "Steer" : "Drive")"
            }
        default:                    return (1...count).map { "M\($0)" }
        }
    }

    /// Parse a HEALTH notification into the roster the status panel renders. nil on ANY
    /// malformation: a truncated frame would otherwise report motors as dead.
    static func decodeHealth(_ data: Data) -> [RobotState.Module]? {
        let b = [UInt8](data)
        guard b.count >= 2, b[0] == MessageType.health.rawValue else { return nil }
        var out: [RobotState.Module] = []
        var o = 2
        for _ in 0..<Int(b[1]) {
            guard o + 3 <= b.count else { return nil }
            let idx = Int(b[o]), n = Int(b[o + 1]), flags = b[o + 2]
            let bytes = (n + 7) / 8
            o += 3
            guard o + bytes <= b.count else { return nil }
            let bm = Array(b[o..<(o + bytes)])
            o += bytes
            guard idx < moduleIds.count else { continue }   // a module this build doesn't know
            let id = moduleIds[idx]
            let labels = actuatorLabels(module: id, count: n)
            var acts: [RobotState.Module.Actuator] = []
            acts.reserveCapacity(n)
            for k in 0..<n {
                let live = bm[k / 8] >> (k % 8) & 1 == 1
                acts.append(.init(id: "\(id).\(k)",
                                  label: k < labels.count ? labels[k] : "M\(k + 1)",
                                  ok: live,
                                  detail: live ? "" : "no reply"))
            }
            let up = acts.filter(\.ok).count
            let hasDriver = flags & 0x01 != 0
            let down = acts.filter { !$0.ok }.map(\.label)
            // Same words as the Wi-Fi roster, so the panel does not change vocabulary
            // when the transport does.
            let detail: String = !hasDriver ? "no driver"
                : (up == n ? "\(up)/\(n)"
                   : (up == 0 ? "link down" : down.prefix(2).joined(separator: ", ")
                      + (down.count > 2 ? " +\(down.count - 2)" : "")))
            out.append(RobotState.Module(id: id,
                                         label: moduleLabels[id] ?? id,
                                         group: id.hasPrefix("cam") ? "sensors" : nil,
                                         ok: flags & 0x04 != 0,
                                         detail: detail,
                                         up: up, n: n, actuators: acts))
        }
        return out
    }

    /// Keep one write inside a single ATT_MTU on iOS (~185 bytes).
    static let maxPayload = 180

    // MARK: - Encoding (app → robot)

    /// `[joint: radians]` → packed. Unknown joint names are skipped, never guessed.
    /// Returns nil when nothing encodable was passed, so the caller does not write an
    /// empty frame the robot would have to reject.
    static func encodeJointCommand(_ targets: [String: Double]) -> Data? {
        var items: [(UInt8, Float)] = targets.compactMap { name, value in
            guard let i = index[name], value.isFinite else { return nil }
            return (i, Float(value))
        }
        guard !items.isEmpty else { return nil }
        // Deterministic order so identical poses produce identical bytes (dedupe upstream).
        items.sort { $0.0 < $1.0 }
        if 2 + 5 * items.count > maxPayload { items = Array(items.prefix((maxPayload - 2) / 5)) }

        var out = Data([MessageType.jointCommand.rawValue, UInt8(items.count)])
        for (i, v) in items { out.append(i); out.append(le(v)) }
        return out
    }


    // MARK: - Hand pose (headset → robot), 2026-09-22

    /// `0x05` HAND_POSE: `u8 side` (0 left, 1 right), `u8 flags` (bit0 tracked), then 21 × (`i16 x, y, z`)
    /// — the ARKit joint POSITIONS relative to the wrist in 0.02 mm, `handJoints` order (129 bytes, one
    /// write). The robot's retargeter reads positions only, so this IS the HandPose a Wi-Fi
    /// teleop_frame carries, 30× smaller than its JSON. Same list on the robot (ble_protocol.HAND_JOINTS).
    static let handJoints: [String] = [
        "wrist", "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip",
        "thumb_tip", "index_knuckle", "index_intermediate_base", "index_intermediate_tip",
        "index_tip", "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip",
        "middle_tip", "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip",
        "ring_tip", "little_knuckle", "little_intermediate_base", "little_intermediate_tip",
        "little_tip",
    ]

    /// `0x06` HEAD_POSE: `u8 flags` (bit0 tracked), then the head's 3×3 rotation as 9 × `i16` (row-major,
    /// ×1e-4) — 19 bytes. The robot turns it into neck yaw / pitch / roll like a Wi-Fi frame's head.
    static func encodeHeadPose(_ head: HeadPose) -> Data? {
        let m = head.transform.matrix
        guard m.count == 16 else { return nil }
        var out = Data([0x06, head.trackingState == .normal ? 0x01 : 0x00])
        for r in 0..<3 { for c in 0..<3 {
            let q = Int16(clamping: Int((m[4 * r + c] * 10_000).rounded()))
            out.append(UInt8(truncatingIfNeeded: Int(q) & 0xFF)); out.append(UInt8(truncatingIfNeeded: (Int(q) >> 8) & 0xFF))
        } }
        return out
    }

    static func encodeHandPose(_ hand: HandPose) -> Data? {
        var pos: [String: (Float, Float, Float)] = [:]
        for j in hand.joints where j.localTransform.matrix.count == 16 {
            let m = j.localTransform.matrix
            pos[j.joint.rawValue] = (m[3], m[7], m[11])        // row-major: the translation column
        }
        guard let w = pos["wrist"] else { return nil }
        var out = Data([0x05, hand.chirality == .left ? 0 : 1, hand.isTracked ? 0x01 : 0x00])
        for n in handJoints {
            let p = pos[n] ?? w
            for v in [p.0 - w.0, p.1 - w.1, p.2 - w.2] {
                let q = Int16(clamping: Int((v * 50_000).rounded()))   // 0.02 mm (±0.65 m)
                out.append(UInt8(truncatingIfNeeded: Int(q) & 0xFF)); out.append(UInt8(truncatingIfNeeded: (Int(q) >> 8) & 0xFF))
            }
        }
        return out
    }

    /// Base twist plus the LIFT STICK. The lift rides this frame rather than a message
    /// of its own because the gateway reads the whole joystick from one frame — a
    /// lift-only frame would arrive with the base axes absent and zero the twist the
    /// previous frame had just set. 16 bytes; the robot still accepts the old 12.
    static func encodeTeleopFrame(vx: Double, vy: Double, wz: Double, lift: Double = 0) -> Data {
        var out = Data([MessageType.teleopFrame.rawValue])
        out.append(le(Float(vx))); out.append(le(Float(vy))); out.append(le(Float(wz)))
        out.append(le(Float(lift)))
        return out
    }

    static func encodeEstop(_ active: Bool) -> Data {
        Data([MessageType.estop.rawValue, active ? 1 : 0])
    }

    /// `method` rides as an optional third byte (0 navigation, 1 joint, 2 retarget); nil keeps the robot's
    /// current method, which is also what an older two-byte frame means to it.
    static func encodeControlMode(soft: Bool, gravityComp: Bool, method: String? = nil) -> Data {
        let flags: UInt8 = (soft ? 0x01 : 0) | (gravityComp ? 0x02 : 0)
        var out = Data([MessageType.controlMode.rawValue, flags])
        if let m = method, let i = ["navigation", "joint", "retarget"].firstIndex(of: m) { out.append(UInt8(i)) }
        return out
    }

    // MARK: - Whole-body teleop over Bluetooth (2026-10-13)

    /// `0x07` WRIST_POSE: the wrists' WORLD poses, the head's pose and the elbows — what the robot's arm
    /// retargeting needs and HAND_POSE (wrist-relative fingers) does not carry. 108 bytes, one write per tick.
    /// The robot merges it with the latest HAND_POSE of each side into one Wi-Fi-shaped teleop_frame.
    static func encodeWristPose(_ hf: HeadsetFrame, timestampMs: UInt32) -> Data? {
        func rot(_ m: [Float]) -> [Int16] {
            var r: [Int16] = []
            for i in 0..<3 { for j in 0..<3 { r.append(Int16(clamping: Int((m[4 * i + j] * 10_000).rounded()))) } }
            return r
        }
        func pose(_ m: [Float]?) -> Data {
            var d = Data()
            guard let m = m, m.count == 16 else {                       // untracked slot: identity, zero translation
                d.append(le(Float(0))); d.append(le(Float(0))); d.append(le(Float(0)))
                for v in [1, 0, 0, 0, 1, 0, 0, 0, 1] as [Int16] { d.append(le16(v * 10_000)) }
                return d
            }
            d.append(le(m[3])); d.append(le(m[7])); d.append(le(m[11]))  // row-major: translation column
            for v in rot(m) { d.append(le16(v)) }
            return d
        }
        func elbow(_ hand: HandPose?) -> Data {
            var d = Data()
            guard let h = hand, h.isTracked, h.anchorTransform.matrix.count == 16,
                  let fa = h.joints.first(where: { $0.joint.rawValue == "forearm_arm" }), fa.localTransform.matrix.count == 16 else {
                for _ in 0..<3 { d.append(le16(0)) }
                return d
            }
            // forearm_arm is in the hand anchor's local frame; the wire wants the WORLD delta from the wrist: R·local
            let a = h.anchorTransform.matrix, l = fa.localTransform.matrix
            for i in 0..<3 {
                let w = a[4 * i] * l[3] + a[4 * i + 1] * l[7] + a[4 * i + 2] * l[11]
                d.append(le16(Int16(clamping: Int((w * 50_000).rounded()))))     // 0.02 mm
            }
            return d
        }
        let left = hf.leftHand.flatMap { $0.isTracked ? $0 : nil }
        let right = hf.rightHand.flatMap { $0.isTracked ? $0 : nil }
        let head = hf.head.trackingState == .normal
        guard left != nil || right != nil || head else { return nil }
        var flags: UInt8 = 0
        if left != nil { flags |= 0x01 }; if right != nil { flags |= 0x02 }; if head { flags |= 0x04 }
        var out = Data([MessageType.wristPose.rawValue, flags])
        out.append(withUnsafeBytes(of: timestampMs.littleEndian) { Data($0) })
        out.append(pose(head ? hf.head.transform.matrix : nil))
        out.append(pose(left?.anchorTransform.matrix))
        out.append(pose(right?.anchorTransform.matrix))
        out.append(elbow(left)); out.append(elbow(right))
        return out                                                         // 1 + 1 + 4 + 3×30 + 2×6 = 108
    }

    /// `0x08` BODY_CALIBRATION: start / cancel / apply + the operator name.
    static func encodeBodyCalibration(action: String, user: String) -> Data? {
        guard let i = ["start", "cancel", "apply"].firstIndex(of: action) else { return nil }
        var out = Data([MessageType.bodyCalibration.rawValue, UInt8(i)])
        out.append(Data(user.utf8.prefix(maxPayload - 2)))
        return out
    }

    static func encodeRecalibrate() -> Data { Data([MessageType.recalibrate.rawValue]) }

    // MARK: - Data collection + replay over Bluetooth (2026-10-24)
    //
    // The robot records every camera and tick on its OWN disk; Bluetooth only carries the button and the status
    // back, so data collection works with no network at all.

    /// `0x0A` RECORD: "stop" / "start" / "select" + the task folder (empty = keep the robot's current one).
    static func encodeRecord(action: String, task: String) -> Data? {
        guard let i = ["stop", "start", "select"].firstIndex(of: action) else { return nil }
        var out = Data([MessageType.record.rawValue, UInt8(i)])
        out.append(Data(task.utf8.prefix(maxPayload - 2)))
        return out
    }

    static func encodeDemoReplay(active: Bool, name: String, repeats: Int) -> Data {
        var out = Data([MessageType.demoReplay.rawValue, active ? 1 : 0, UInt8(clamping: max(1, repeats))])
        out.append(Data(name.utf8.prefix(maxPayload - 3)))
        return out
    }

    static func encodeDemoList() -> Data { Data([MessageType.demoList.rawValue]) }
    static func encodeHome() -> Data { Data([MessageType.home.rawValue]) }

    /// The recorder's camera streams — index IS the wire identity (ble_protocol.CAMERAS). APPEND ONLY.
    static let cameras = ["head_left", "head_right", "wrist_left", "wrist_right",
                          "wrist_left_depth", "wrist_right_depth", "head_left_depth"]

    struct Recording: Equatable {
        var isRecording = false
        var finalizing = false
        var error: String?
        var episodeCount = 0
        var total = 0
        var elapsed: Double = 0
        var dropped = 0
        var task = ""
        var cameras: [RobotState.RecordCamera] = []
        var replayPlaying = false
        var replayName: String?
        var rep = 0
        var repeats = 0
        var frac: Double = 0
        var phase = ""
    }

    /// `0x84` RECORDING notify → the Rec pill, the episode count, every camera's frame count and the robot's
    /// refusal (if any). nil on any malformation.
    static func decodeRecording(_ data: Data) -> Recording? {
        let b = [UInt8](data)
        guard b.count >= 18, b[0] == MessageType.recording.rawValue else { return nil }
        func u16(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 | Int(b[o + 2]) << 16 | Int(b[o + 3]) << 24 }
        let f = b[1]
        var r = Recording()
        r.isRecording = f & 0x01 != 0; r.finalizing = f & 0x02 != 0; r.replayPlaying = f & 0x08 != 0
        r.episodeCount = u16(2); r.total = u16(4)
        let e = Float(bitPattern: UInt32(u32(6))); r.elapsed = e.isFinite ? Double(e) : 0
        r.dropped = u16(10)
        r.rep = Int(b[12]); r.repeats = Int(b[13]); r.frac = Double(b[14]) / 200
        r.phase = ["", "lead_in", "play", "return"][min(Int(b[15]), 3)]
        var o = 17
        for _ in 0..<Int(b[16]) {
            guard o + 6 <= b.count else { return nil }
            let i = Int(b[o]), cf = b[o + 1], n = u32(o + 2)
            o += 6
            if i < cameras.count {
                r.cameras.append(.init(name: cameras[i], frames: n, live: cf & 0x01 != 0, error: cf & 0x02 != 0))
            }
        }
        func string() -> String? {
            guard o < b.count else { return nil }
            let n = Int(b[o]); let end = min(b.count, o + 1 + n)
            defer { o = end }
            return String(decoding: b[(o + 1)..<end], as: UTF8.self)
        }
        r.task = string() ?? ""
        let name = string() ?? ""
        r.replayName = name.isEmpty ? nil : name
        let err = string() ?? ""
        r.error = f & 0x04 != 0 && !err.isEmpty ? err : nil
        return r
    }

    /// `0x85` CLIPS notify → (page, pages, clips). The app collects pages until it has all of them.
    static func decodeClips(_ data: Data) -> (page: Int, pages: Int, clips: [RobotState.DemoClipInfo])? {
        let b = [UInt8](data)
        guard b.count >= 4, b[0] == MessageType.clips.rawValue else { return nil }
        var o = 4, out: [RobotState.DemoClipInfo] = []
        for _ in 0..<Int(b[3]) {
            guard o + 4 <= b.count else { return nil }
            let d = Double(Int(b[o]) | Int(b[o + 1]) << 8) / 10, n = Int(b[o + 3])
            guard o + 4 + n <= b.count else { return nil }
            let name = String(decoding: b[(o + 4)..<(o + 4 + n)], as: UTF8.self)
            out.append(.init(name: name, duration: d, frames: 0, created: "", recordedBy: "NOVA", joints: 0))
            o += 4 + n
        }
        return (Int(b[1]), Int(b[2]), out)
    }

    /// `0x83` CALIBRATION notify → the two robot_state fields the calibration card reads over Wi-Fi.
    /// nil status = idle (flags 0); nil metrics until a reach > 0.05 has been measured.
    static func decodeCalibration(_ data: Data) -> (CalibrationProgress?, OperatorMetrics?)? {
        let b = [UInt8](data)
        guard b.count >= 14, b[0] == MessageType.calibration.rawValue else { return nil }
        func f32(_ o: Int) -> Double {
            let bits = UInt32(b[o]) | UInt32(b[o+1]) << 8 | UInt32(b[o+2]) << 16 | UInt32(b[o+3]) << 24
            let v = Float(bitPattern: bits); return v.isFinite ? Double(v) : 0
        }
        let flags = b[1], reach = f32(2), human = f32(6), robot = f32(10)
        let user = String(decoding: b[14...], as: UTF8.self)
        let status: CalibrationProgress? = flags == 0 ? nil
            : CalibrationProgress(active: flags & 0x01 != 0, pose: "arms_forward", index: flags & 0x02 != 0 ? 1 : 0,
                                  total: 1, done: flags & 0x02 != 0, reach: reach > 0 ? reach : nil)
        var metrics: OperatorMetrics? = nil
        if reach > 0.05 {
            var m = OperatorMetrics(); m.user = user.isEmpty ? "operator" : user
            m.reach = reach; m.humanReach = human; m.robotReach = robot; metrics = m
        }
        return (status, metrics)
    }

    // MARK: - Decoding (robot → app)

    struct State {
        var positions: [String: Double] = [:]
        var estop = false
        var torque = false
    }

    /// Parse a STATE notification. Returns nil on ANY malformation — a short read on a
    /// Bluetooth link is normal, and half a pose is worse than no pose.
    static func decodeState(_ data: Data) -> State? {
        let b = [UInt8](data)
        guard b.count >= 3, b[0] == MessageType.state.rawValue else { return nil }
        let flags = b[1], count = Int(b[2])
        guard b.count == 3 + 5 * count else { return nil }

        var s = State(positions: [:], estop: flags & 0x01 != 0, torque: flags & 0x02 != 0)
        s.positions.reserveCapacity(count)
        for k in 0..<count {
            let o = 3 + 5 * k
            let idx = Int(b[o])
            guard idx < joints.count else { continue }      // a joint this build doesn't know
            let bits = UInt32(b[o+1]) | UInt32(b[o+2]) << 8 | UInt32(b[o+3]) << 16 | UInt32(b[o+4]) << 24
            let v = Float(bitPattern: bits)
            guard v.isFinite else { continue }
            s.positions[joints[idx]] = Double(v)
        }
        return s
    }

    /// The robot's passive-scan advertisement payload (manufacturer id 0xFFFF):
    /// `[version, kind, flags]`. Read WITHOUT connecting, so the proximity card can show
    /// live status the moment the robot is in range.
    struct Beacon {
        var version: UInt8 = 0
        var isNova = false
        var estop = false
        var torque = false
        /// The robot's own gateway link is up — it can actually accept commands.
        var gatewayUp = false
    }

    static let companyID: UInt16 = 0xFFFF

    static func decodeBeacon(_ data: Data) -> Beacon? {
        let b = [UInt8](data)
        guard b.count >= 3 else { return nil }
        return Beacon(version: b[0], isNova: b[1] == 0x01,
                      estop: b[2] & 0x01 != 0,
                      torque: b[2] & 0x02 != 0,
                      gatewayUp: b[2] & 0x04 != 0)
    }

    // MARK: - Service / characteristic identity

    enum UUIDs {
        static let service = "6D696C6F-0001-4A55-9E00-000000000000"
        static let command = "6D696C6F-0002-4A55-9E00-000000000000"
        static let estop   = "6D696C6F-0003-4A55-9E00-000000000000"
        static let state   = "6D696C6F-0004-4A55-9E00-000000000000"
        static let info    = "6D696C6F-0005-4A55-9E00-000000000000"
    }

    // MARK: - Private

    private static func le(_ f: Float) -> Data {
        withUnsafeBytes(of: f.bitPattern.littleEndian) { Data($0) }
    }

    private static func le16(_ v: Int16) -> Data {
        withUnsafeBytes(of: v.littleEndian) { Data($0) }
    }
}
