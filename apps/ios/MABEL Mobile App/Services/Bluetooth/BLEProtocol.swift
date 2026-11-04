//
//  BLEProtocol.swift
//  MABEL Mobile App
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
//      0x04  CONTROL_MODE    u8 flags (bit0 soft, bit1 gravityComp)
//      0x81  STATE (notify)  u8 flags, u8 count, count × (u8 jointIndex, f32 radians)
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
        case teleopFrame  = 0x02
        case estop        = 0x03
        case controlMode  = 0x04
        case state        = 0x81
        case health       = 0x82
    }

    // MARK: - Module health (server/ble_protocol.py encode_health)

    /// CANONICAL MODULE ORDER — the index IS the wire identity, exactly like `joints`.
    /// APPEND ONLY, and keep in step with `ble_protocol.MODULES` on the robot.
    static let moduleIds = ["base", "torso", "lift", "neck", "leftArm", "rightArm",
                            "leftHand", "rightHand", "camHead"]

    /// Display names for the rows. Over Wi-Fi the server owns these; BLE cannot afford to
    /// send them, so the app supplies the same words from the same table.
    static let moduleLabels: [String: String] = [
        "base": "Base", "torso": "Torso", "lift": "Lift", "neck": "Neck",
        "leftArm": "Left Arm", "rightArm": "Right Arm",
        "leftHand": "Left Hand", "rightHand": "Right Hand", "camHead": "Head Cam",
    ]

    /// MOTOR NAMES, DERIVED. The health frame carries counts and one bit per motor — not
    /// names, which is the trade that makes it fit one BLE write. These are the same names
    /// the robot's actuator labeller produces, reconstructed from the module and the
    /// count, so the panel reads identically on either transport. The hand is the one
    /// exception: its seventeen joints have no derivable order here, so they are numbered.
    static func actuatorLabels(module: String, count: Int) -> [String] {
        guard count > 0 else { return [] }
        switch module {
        case "leftArm", "rightArm": return (1...count).map { "J\($0)" }
        case "torso":               return ["Torso"]
        case "lift":                return ["Column"]
        case "camHead":             return ["Stereo"]
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
                                         group: id == "camHead" ? "sensors" : nil,
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

    // encodeHandPose lives in the visionOS app only: this app has no hand tracking (and no HandPose type).

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

    static func encodeControlMode(soft: Bool, gravityComp: Bool) -> Data {
        let flags: UInt8 = (soft ? 0x01 : 0) | (gravityComp ? 0x02 : 0)
        return Data([MessageType.controlMode.rawValue, flags])
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
}
