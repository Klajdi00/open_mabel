//
//  PoseModels.swift
//  MabelTeleop
//
//  Value types that describe what the headset sends downstream.
//  Keep these pure, `Codable`, and side-effect free — they cross
//  the network boundary and show up in unit tests.
//

import Foundation
import simd

// MARK: - Transforms

/// A 4x4 rigid transform serialized in row-major order.
/// Using a flat array keeps the JSON/MessagePack payload stable across
/// language boundaries (Python / C++ deserializers on the robot side).
public struct Transform4x4: Codable, Equatable, Sendable {
    public var matrix: [Float]   // length 16, row-major

    public init(_ m: simd_float4x4) {
        self.matrix = [
            m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x,
            m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y,
            m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z,
            m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w
        ]
    }

    /// From a flat row-major array (the wire form). Anything but 16 elements
    /// falls back to identity — never trap on wire-shaped input.
    public init(matrix: [Float]) {
        self.matrix = matrix.count == 16
            ? matrix
            : [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
    }

    public var simd: simd_float4x4 {
        simd_float4x4(rows: [
            SIMD4(matrix[0],  matrix[1],  matrix[2],  matrix[3]),
            SIMD4(matrix[4],  matrix[5],  matrix[6],  matrix[7]),
            SIMD4(matrix[8],  matrix[9],  matrix[10], matrix[11]),
            SIMD4(matrix[12], matrix[13], matrix[14], matrix[15])
        ])
    }
}

// MARK: - Head

public struct HeadPose: Codable, Equatable, Sendable {
    /// Device-to-world transform (head/headset origin).
    public var transform: Transform4x4
    /// ARKit world tracking confidence / state as a simple enum.
    public var trackingState: TrackingState

    public enum TrackingState: String, Codable, Sendable {
        case normal, limited, notAvailable
    }
}

// MARK: - Hands

/// Mirrors the 27 joints exposed by `HandSkeleton.JointName` on visionOS.
/// Kept as a string-backed enum so the JSON is self-describing on the robot side.
public enum HandJoint: String, Codable, CaseIterable, Sendable {
    case wrist
    case thumbKnuckle          = "thumb_knuckle"
    case thumbIntermediateBase = "thumb_intermediate_base"
    case thumbIntermediateTip  = "thumb_intermediate_tip"
    case thumbTip              = "thumb_tip"
    case indexFingerMetacarpal   = "index_metacarpal"
    case indexFingerKnuckle      = "index_knuckle"
    case indexFingerIntermediateBase = "index_intermediate_base"
    case indexFingerIntermediateTip  = "index_intermediate_tip"
    case indexFingerTip          = "index_tip"
    case middleFingerMetacarpal   = "middle_metacarpal"
    case middleFingerKnuckle      = "middle_knuckle"
    case middleFingerIntermediateBase = "middle_intermediate_base"
    case middleFingerIntermediateTip  = "middle_intermediate_tip"
    case middleFingerTip          = "middle_tip"
    case ringFingerMetacarpal   = "ring_metacarpal"
    case ringFingerKnuckle      = "ring_knuckle"
    case ringFingerIntermediateBase = "ring_intermediate_base"
    case ringFingerIntermediateTip  = "ring_intermediate_tip"
    case ringFingerTip          = "ring_tip"
    case littleFingerMetacarpal   = "little_metacarpal"
    case littleFingerKnuckle      = "little_knuckle"
    case littleFingerIntermediateBase = "little_intermediate_base"
    case littleFingerIntermediateTip  = "little_intermediate_tip"
    case littleFingerTip          = "little_tip"
    case forearmWrist = "forearm_wrist"
    case forearmArm   = "forearm_arm"
}

public enum Chirality: String, Codable, Sendable {
    case left, right
}

public struct HandJointPose: Codable, Equatable, Sendable {
    public var joint: HandJoint
    /// Joint transform in the anchor's local frame.
    public var localTransform: Transform4x4
    /// Whether ARKit considers this joint tracked this frame.
    public var isTracked: Bool
}

public struct HandPose: Codable, Equatable, Sendable {
    public var chirality: Chirality
    /// Hand-anchor-to-world transform (the "root" of this hand).
    public var anchorTransform: Transform4x4
    /// Per-joint local transforms. Empty if tracking is lost this frame.
    public var joints: [HandJointPose]
    /// True if ARKit reports this anchor as tracked this frame.
    public var isTracked: Bool
}

// MARK: - Teleop Mode

public enum TeleopMode: String, Codable, CaseIterable, Sendable {
    case manipulation = "Arms & Hands"
    case baseNavigation = "Base Driving"
}

// MARK: - Navigation joysticks

/// Drone-style virtual joysticks for base-navigation mode, each axis in [-1, 1].
/// Left stick = translate (x = strafe, y = forward/back); right stick = yaw (x)
/// + lift up/down (y). Zero in manipulation mode.
public struct NavJoystick: Codable, Equatable, Sendable {
    public var lx: Float = 0
    public var ly: Float = 0
    public var rx: Float = 0
    public var ry: Float = 0

    public static let zero = NavJoystick()
    public var isActive: Bool { lx != 0 || ly != 0 || rx != 0 || ry != 0 }
}

// MARK: - Outbound packet

/// Everything the headset emits each tick. Timestamp is the
/// headset's monotonic clock in seconds — the robot clocks should
/// resync using their own receive-time if they need absolute alignment.
public struct TeleopFrame: Codable, Equatable, Sendable {
    public var sequence: UInt64
    public var timestamp: TimeInterval
    public var head: HeadPose
    public var leftHand: HandPose?
    public var rightHand: HandPose?
    public var mode: TeleopMode = .manipulation
    public var navJoystick: NavJoystick? = nil
}
