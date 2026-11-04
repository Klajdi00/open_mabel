//
//  WireCodecTests.swift
//  MabelTeleopTests
//

import XCTest
@testable import MabelTeleop
import simd

final class WireCodecTests: XCTestCase {

    func test_teleopFrame_roundTrip_preservesAllFields() throws {
        let frame = Self.makeFrame()

        let encoded = try WireCodec.encode(frame, as: .teleopFrame)
        let envelope = try WireCodec.decodeEnvelope(encoded)

        XCTAssertEqual(envelope.type, .teleopFrame)

        let decoded = try WireCodec.decodePayload(TeleopFrame.self, from: envelope)
        XCTAssertEqual(decoded, frame)
    }

    // NOTE: the revision-1 Codable RobotState round-trip test was removed —
    // the app now uses the iOS-style RobotState, decoded field-by-field from
    // the `robot_state` JSON in WebSocketLink (not via WireCodec).

    // MARK: - Frame-envelope wire shape (the 60 Hz outbound path)

    /// WebSocketLink.encodeFrameEnvelope replaced the manual-dict encode
    /// (3 × JSONEncoder→JSONSerialization round trips per frame) with ONE
    /// Codable pass. These pins hold the emitted wire shape to the contract
    /// the server parses — if a Codable rename ever changes a key, this
    /// fails before the robot goes still.
    func test_frameEnvelope_wireShape_fullFrame() throws {
        var frame = Self.makeFrame()
        frame.navJoystick = NavJoystick(lx: 0.25, ly: -1, rx: 0.5, ry: 0)

        let data = try WebSocketLink.encodeFrameEnvelope(frame)
        let env = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(env["type"] as? String, "teleop_frame")
        let p = try XCTUnwrap(env["payload"] as? [String: Any])
        XCTAssertEqual(p["sequence"] as? UInt64, 42)
        XCTAssertEqual(p["timestamp"] as? Double, 9.9)
        XCTAssertEqual(p["mode"] as? String, "Arms & Hands")

        let nav = try XCTUnwrap(p["navJoystick"] as? [String: Any])
        XCTAssertEqual(nav["lx"] as? Double ?? -9, 0.25, accuracy: 1e-6)
        XCTAssertEqual(nav["ly"] as? Double ?? -9, -1, accuracy: 1e-6)

        let head = try XCTUnwrap(p["head"] as? [String: Any])
        XCTAssertEqual(head["trackingState"] as? String, "normal")
        let headT = try XCTUnwrap(head["transform"] as? [String: Any])
        XCTAssertEqual((headT["matrix"] as? [Any])?.count, 16)

        for key in ["leftHand", "rightHand"] {
            let hand = try XCTUnwrap(p[key] as? [String: Any], key)
            XCTAssertEqual(hand["isTracked"] as? Bool, true)
            let joints = try XCTUnwrap(hand["joints"] as? [[String: Any]])
            XCTAssertEqual(joints.count, HandJoint.allCases.count)
            let j0 = try XCTUnwrap(joints.first)
            XCTAssertNotNil(j0["joint"] as? String)
            XCTAssertNotNil((j0["localTransform"] as? [String: Any])?["matrix"])
            XCTAssertNotNil(j0["isTracked"] as? Bool)
        }
        XCTAssertEqual(p["leftHand"] as? [String: Any] != nil
                       ? (p["leftHand"] as? [String: Any])?["chirality"] as? String : nil, "left")
    }

    /// The no-ARKit fallback (simulator / tracking down): hands OMITTED (not
    /// null — the server treats a present key as data), head notAvailable.
    func test_frameEnvelope_wireShape_noHands() throws {
        let identity = Transform4x4(matrix_identity_float4x4)
        var frame = TeleopFrame(
            sequence: 1, timestamp: 0,
            head: HeadPose(transform: identity, trackingState: .notAvailable),
            leftHand: nil, rightHand: nil)
        frame.navJoystick = NavJoystick.zero

        let data = try WebSocketLink.encodeFrameEnvelope(frame)
        let env = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let p = try XCTUnwrap(env["payload"] as? [String: Any])
        XCTAssertNil(p["leftHand"])
        XCTAssertNil(p["rightHand"])
        XCTAssertEqual((p["head"] as? [String: Any])?["trackingState"] as? String, "notAvailable")
    }

    /// Transform4x4(matrix:) is the head-fallback constructor on the wire
    /// path — wrong-length input must yield identity, never trap mid-teleop.
    func test_transform4x4_arrayInit() {
        let t = Transform4x4(matrix: [Float](repeating: 2, count: 16))
        XCTAssertEqual(t.matrix, [Float](repeating: 2, count: 16))
        let bad = Transform4x4(matrix: [1, 2, 3])
        XCTAssertEqual(bad.simd, matrix_identity_float4x4)
    }

    func test_transform4x4_roundTrip_preservesMatrix() {
        let m = simd_float4x4(
            SIMD4<Float>(1, 2, 3, 4),
            SIMD4<Float>(5, 6, 7, 8),
            SIMD4<Float>(9, 10, 11, 12),
            SIMD4<Float>(13, 14, 15, 16)
        )
        let t = Transform4x4(m)
        XCTAssertEqual(t.simd, m)
    }

    // MARK: - Factory

    private static func makeFrame() -> TeleopFrame {
        let identity = Transform4x4(matrix_identity_float4x4)
        let joints = HandJoint.allCases.map {
            HandJointPose(joint: $0, localTransform: identity, isTracked: true)
        }
        return TeleopFrame(
            sequence: 42,
            timestamp: 9.9,
            head: HeadPose(transform: identity, trackingState: .normal),
            leftHand:  HandPose(chirality: .left,  anchorTransform: identity, joints: joints, isTracked: true),
            rightHand: HandPose(chirality: .right, anchorTransform: identity, joints: joints, isTracked: true)
        )
    }
}
