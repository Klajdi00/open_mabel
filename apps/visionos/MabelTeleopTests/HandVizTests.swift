//
//  HandVizTests.swift
//  MabelTeleopTests
//
//  Pins the render-loop hand-dot overlay (HandDotOverlay) + the broadcaster's
//  latest-snapshot slot — the 2026-07-29 rework that replaced the async
//  MainActor overlay task (which the main actor starved into visible lag the
//  moment whole-body teleop engaged) with a synchronous per-render-frame
//  apply. The simulator has no hand tracking, so synthetic snapshots are the
//  ONLY way this path is ever exercised by CI.
//

import XCTest
import RealityKit
import simd
@testable import MabelTeleop

@MainActor
final class HandVizTests: XCTestCase {

    // MARK: - Helpers

    private func pose(chirality: Chirality, tracked: Bool = true,
                      anchorX: Float = 0, jointY: Float = 0) -> HandPose {
        var anchor = matrix_identity_float4x4
        anchor.columns.3.x = anchorX
        var local = matrix_identity_float4x4
        local.columns.3.y = jointY
        let joints = HandJoint.allCases.map {
            HandJointPose(joint: $0, localTransform: Transform4x4(local), isTracked: tracked)
        }
        return HandPose(chirality: chirality,
                        anchorTransform: Transform4x4(anchor),
                        joints: tracked ? joints : [],
                        isTracked: tracked)
    }

    private func snap(left: HandPose?, right: HandPose?, t: TimeInterval)
        -> HandTrackingService.Snapshot {
        HandTrackingService.Snapshot(left: left, right: right, timestamp: t)
    }

    // MARK: - Overlay

    func test_freshSnapshot_positionsAndEnablesDots() {
        let viz = HandDotOverlay()
        let s = snap(left: pose(chirality: .left, anchorX: 0.3, jointY: 0.1),
                     right: nil, t: 100)
        viz.tick(s, now: 100.01)

        for (_, e) in viz.left {
            XCTAssertTrue(e.isEnabled, "tracked left joints must show")
            XCTAssertEqual(e.transform.translation.x, 0.3, accuracy: 1e-5)
            XCTAssertEqual(e.transform.translation.y, 0.1, accuracy: 1e-5)
        }
        for (_, e) in viz.right {
            XCTAssertFalse(e.isEnabled, "missing right hand must hide")
        }
    }

    func test_staleSnapshot_hidesEverything() {
        let viz = HandDotOverlay()
        viz.tick(snap(left: pose(chirality: .left), right: pose(chirality: .right), t: 100),
                 now: 100.01)
        XCTAssertTrue(viz.left.values.contains { $0.isEnabled })

        // Same snapshot, but the clock moved past the staleness window
        // (tracking stopped / space parked): a frozen skeleton must vanish.
        viz.tick(snap(left: pose(chirality: .left), right: pose(chirality: .right), t: 100),
                 now: 100 + HandDotOverlay.staleAfter + 0.01)
        XCTAssertFalse(viz.left.values.contains { $0.isEnabled })
        XCTAssertFalse(viz.right.values.contains { $0.isEnabled })
    }

    func test_untrackedHand_hidesItsDots() {
        let viz = HandDotOverlay()
        viz.tick(snap(left: pose(chirality: .left), right: pose(chirality: .right), t: 1),
                 now: 1.0)
        viz.tick(snap(left: pose(chirality: .left, tracked: false),
                      right: pose(chirality: .right), t: 2),
                 now: 2.0)
        XCTAssertFalse(viz.left.values.contains { $0.isEnabled })
        XCTAssertTrue(viz.right.values.allSatisfy { $0.isEnabled })
    }

    func test_repeatTimestamp_isDeduped() {
        let viz = HandDotOverlay()
        viz.tick(snap(left: pose(chirality: .left, anchorX: 0.1), right: nil, t: 5), now: 5.0)
        let before = viz.left[.wrist]!.transform.translation.x

        // Same stamp with DIFFERENT data must be ignored (one apply per
        // ARKit snapshot — the render loop runs faster than new data arrives).
        viz.tick(snap(left: pose(chirality: .left, anchorX: 0.9), right: nil, t: 5), now: 5.01)
        XCTAssertEqual(viz.left[.wrist]!.transform.translation.x, before, accuracy: 1e-6)

        // A NEW stamp applies.
        viz.tick(snap(left: pose(chirality: .left, anchorX: 0.9), right: nil, t: 6), now: 6.0)
        XCTAssertEqual(viz.left[.wrist]!.transform.translation.x, 0.9, accuracy: 1e-5)
    }

    func test_nilSnapshot_hidesEverything() {
        let viz = HandDotOverlay()
        viz.tick(snap(left: pose(chirality: .left), right: nil, t: 1), now: 1.0)
        viz.tick(nil, now: 2.0)
        XCTAssertFalse(viz.left.values.contains { $0.isEnabled })
    }

    // MARK: - The HAND is the unit, not the joint
    //
    // "The point overlay disappears here and there" (operator, 2026-08-01).
    // ARKit flags individual joints untracked constantly — a curled finger, a
    // thumb behind a palm — and gating each dot on its own `isTracked` made the
    // overlay sparkle on a hand that was being tracked perfectly well.

    /// A tracked hand whose joints carry MIXED per-joint tracking flags — the
    /// ordinary case, not an edge case.
    private func mixedPose(chirality: Chirality) -> HandPose {
        var local = matrix_identity_float4x4
        local.columns.3.y = 0.2
        let joints = HandJoint.allCases.enumerated().map { i, j in
            HandJointPose(joint: j, localTransform: Transform4x4(local),
                          isTracked: i % 3 != 0)          // every third is "untracked"
        }
        return HandPose(chirality: chirality,
                        anchorTransform: Transform4x4(matrix_identity_float4x4),
                        joints: joints, isTracked: true)
    }

    func test_trackedHand_drawsEveryJointDespiteMixedJointFlags() {
        let viz = HandDotOverlay()
        viz.tick(snap(left: mixedPose(chirality: .left), right: nil, t: 100), now: 100.01)
        for (joint, e) in viz.left {
            XCTAssertTrue(e.isEnabled,
                          "\(joint) winked out on a TRACKED hand — that is the sparkle")
            XCTAssertEqual(e.transform.translation.y, 0.2, accuracy: 1e-5,
                           "every dot must still be positioned, not left stale")
        }
    }

    func test_untrackedHand_stillHidesTheWholeHand() {
        // The hand-level flag is the one that matters, and it must still hide.
        let viz = HandDotOverlay()
        viz.tick(snap(left: mixedPose(chirality: .left), right: nil, t: 100), now: 100.01)
        XCTAssertTrue(viz.left.values.contains { $0.isEnabled })
        viz.tick(snap(left: pose(chirality: .left, tracked: false), right: nil, t: 101),
                 now: 101.01)
        XCTAssertFalse(viz.left.values.contains { $0.isEnabled },
                       "an untracked HAND draws nothing")
    }

    // MARK: - Broadcaster latest slot

    func test_broadcaster_latestSlot() {
        let b = SnapshotBroadcaster()
        XCTAssertNil(b.latest())
        let s = snap(left: pose(chirality: .left), right: nil, t: 7)
        b.send(s)
        XCTAssertEqual(b.latest()?.timestamp, 7)
        b.clearLatest()
        XCTAssertNil(b.latest())
    }
}
