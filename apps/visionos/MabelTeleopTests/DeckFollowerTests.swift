//
//  DeckFollowerTests.swift
//  MabelTeleopTests
//
//  Full immersion's control deck + status bar. `.follow` (the default, operator
//  2026-10-17: "follow the human gaze too like the two wrist cams") tracks heading
//  AND pitch but ignores head micro-motion while a control is pinched; `.steady`
//  stays put while the operator glances around or looks down at it, and comes
//  back in front only on a real turn. The Simulator cannot turn a head, so these
//  are the harness.
//

import XCTest
import simd
@testable import MabelTeleop

final class DeckFollowerTests: XCTestCase {

    private func head(yaw: Float, pitch: Float = 0, at p: SIMD3<Float> = SIMD3(0, 1.4, 0)) -> simd_float4x4 {
        var m = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
                              * simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0)))
        m.columns.3 = SIMD4(p, 1)
        return m
    }

    private func run(_ f: inout DeckFollower, yaw: Float, pitch: Float = 0, seconds: Float,
                     at p: SIMD3<Float> = SIMD3(0, 1.4, 0)) {
        let h = head(yaw: yaw, pitch: pitch, at: p)
        for _ in 0..<Int((seconds * 90).rounded()) {
            _ = f.step(headPos: p, headYaw: DeckFollower.yaw(of: h),
                       headPitch: DeckFollower.pitch(of: h), dt: 1 / 90)
        }
    }

    private func steady() -> DeckFollower {
        var f = DeckFollower(); f.mode = .steady; return f
    }

    func test_yawOf_readsHeadingNotPitch() {
        XCTAssertEqual(DeckFollower.yaw(of: head(yaw: 0.5)), 0.5, accuracy: 1e-5)
        XCTAssertEqual(DeckFollower.yaw(of: head(yaw: 0.5, pitch: -0.9)), 0.5, accuracy: 1e-5,
                       "looking DOWN at the deck must not read as a turn")
    }

    /// STEADY: a glance (below the hysteresis) never moves the deck.
    func test_glanceDoesNotMoveTheDeck() {
        var f = steady()
        run(&f, yaw: 0, seconds: 0.5)
        run(&f, yaw: 20 * .pi / 180, pitch: -0.7, seconds: 3)
        XCTAssertEqual(f.yaw, 0, accuracy: 1e-6)
        XCTAssertFalse(f.recentering)
    }

    /// STEADY: a real turn brings it back in front, then it stops.
    func test_turnRecentersAndSettles() {
        var f = steady()
        run(&f, yaw: 0, seconds: 0.5)
        run(&f, yaw: 90 * .pi / 180, seconds: 3)
        XCTAssertEqual(f.yaw, Float(90 * Double.pi / 180), accuracy: Float(2.1 * Double.pi / 180))
        XCTAssertFalse(f.recentering, "it must stop, not keep creeping")
    }

    /// Standing up / stepping: the deck travels with the head position.
    func test_followsHeadPosition() {
        var f = DeckFollower()
        run(&f, yaw: 0, seconds: 0.5)
        run(&f, yaw: 0, seconds: 3, at: SIMD3(0.4, 1.7, -0.3))
        XCTAssertEqual(f.position.y, 1.7, accuracy: 0.01)
        XCTAssertEqual(f.position.x, 0.4, accuracy: 0.01)
    }

    /// Wrap-around: turning through ±180° takes the SHORT way.
    func test_wrapsTheShortWay() {
        var f = steady()
        run(&f, yaw: 170 * .pi / 180, seconds: 0.5)
        run(&f, yaw: -170 * .pi / 180, seconds: 0.05)
        XCTAssertFalse(f.recentering, "20° across the seam is a glance, not a 340° turn")
    }

    // MARK: - FOLLOW (the default)

    func test_defaultIsFollow() {
        XCTAssertEqual(DeckFollower().mode, .follow)
    }

    func test_pitchOf_readsElevation() {
        XCTAssertEqual(DeckFollower.pitch(of: head(yaw: 0.7, pitch: -0.4)), -0.4, accuracy: 1e-5)
        XCTAssertEqual(DeckFollower.pitch(of: head(yaw: 0, pitch: 0.3)), 0.3, accuracy: 1e-5)
    }

    /// A turn AND a nod: the deck ends up framed with the head-locked window (within the
    /// deadband), like the wrist cams — fast (well under a second).
    func test_followsTurnAndNod() {
        var f = DeckFollower()
        run(&f, yaw: 0, seconds: 0.5)
        run(&f, yaw: 60 * .pi / 180, pitch: -30 * .pi / 180, seconds: 0.8)
        // ALIGNED under the window, not parked at the deadband's edge.
        XCTAssertEqual(f.yaw, 60 * .pi / 180, accuracy: f.alignAt + 1e-4)
        XCTAssertEqual(f.pitch, -30 * .pi / 180, accuracy: f.alignAt + 1e-4)
        XCTAssertFalse(f.followingYaw || f.followingPitch, "it must stop, not keep creeping")
    }

    /// Head micro-motion while the eyes drop to a control and the hand pinches must not
    /// move the target under the finger.
    func test_microMotionInsideDeadbandHoldsStill() {
        var f = DeckFollower()
        run(&f, yaw: 0, seconds: 0.5)
        for i in 0..<180 {                                   // 2 s of ±2° tremor, both axes
            let a = 2 * .pi / 180 * sin(Float(i) * 0.7)
            run(&f, yaw: a, pitch: -a, seconds: 1 / 90)
        }
        XCTAssertEqual(f.yaw, 0, accuracy: 1e-6)
        XCTAssertEqual(f.pitch, 0, accuracy: 1e-6)
    }

    /// Crossing the deadband starts a glide, not a jump.
    func test_followIsContinuousAtTheDeadband() {
        var f = DeckFollower()
        run(&f, yaw: 0, seconds: 0.5)
        var prev = f.yaw
        var worst: Float = 0
        for i in 1...90 {                                    // a slow 0 → 20° turn over 1 s
            run(&f, yaw: Float(i) / 90 * 20 * .pi / 180, seconds: 1 / 90)
            worst = max(worst, abs(f.yaw - prev)); prev = f.yaw
        }
        XCTAssertLessThan(worst, 0.5 * .pi / 180, "per-frame step must stay small")
    }

    /// Looking straight down never flips the panel over the top.
    func test_pitchIsClamped() {
        var f = DeckFollower()
        run(&f, yaw: 0, pitch: -85 * .pi / 180, seconds: 2)
        XCTAssertGreaterThanOrEqual(f.pitch, -f.maxPitch - 1e-4)
    }

    /// Switching to STEADY levels the panel and stops tracking pitch.
    func test_switchToSteadyLevelsOut() {
        var f = DeckFollower()
        run(&f, yaw: 0, pitch: -30 * .pi / 180, seconds: 1)
        f.mode = .steady
        run(&f, yaw: 0, pitch: -30 * .pi / 180, seconds: 2)
        XCTAssertEqual(f.pitch, 0, accuracy: 0.01)
    }

    // MARK: - The CONTROL DECK never follows pitch (operator, 2026-10-23: "when I
    // move the head down the bar moves away … do not have that again").

    private func deck() -> DeckFollower {
        var f = DeckFollower(); f.followsPitch = false; return f
    }

    /// Looking down at the deck must not move it: its pitch stays level, in BOTH modes.
    func test_deckPitchStaysLevelWhenLookingDown() {
        for mode in DeckFollower.Mode.allCases {
            var f = deck(); f.mode = mode
            run(&f, yaw: 0, seconds: 0.5)
            run(&f, yaw: 0, pitch: -40 * .pi / 180, seconds: 3)
            XCTAssertEqual(f.pitch, 0, accuracy: 1e-6, "mode \(mode)")
        }
    }

    /// The point of it: nodding DOWN brings the deck TOWARD the centre of the view.
    /// (Seated at `DeckLayout.deckCentreAngle` below the rig's horizon, its angle off
    /// the gaze is that minus the nod — it must shrink as the head goes down.)
    func test_nodBringsTheDeckIntoView() {
        let c = DeckLayout.deckCentreAngle(height: 0.3)
        var offAxis: [Float] = []
        for nodDeg: Float in [0, 15, 30] {
            var f = deck()
            run(&f, yaw: 0, pitch: -nodDeg * .pi / 180, seconds: 2)
            offAxis.append(abs(c + f.pitch - nodDeg * .pi / 180))
        }
        XCTAssertGreaterThan(offAxis[0], offAxis[1])
        XCTAssertGreaterThan(offAxis[1], offAxis[2])
    }

    /// Heading still follows: turn and the deck comes round in front.
    func test_deckStillFollowsHeading() {
        var f = deck()
        run(&f, yaw: 0, seconds: 0.5)
        run(&f, yaw: 0.8, pitch: -0.4, seconds: 2)
        XCTAssertEqual(f.yaw, 0.8, accuracy: 0.01)
        XCTAssertEqual(f.pitch, 0, accuracy: 1e-6)
    }
}
