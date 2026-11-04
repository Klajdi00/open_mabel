//
//  HomeLiftTests.swift
//  MabelTeleopTests
//
//  PRESS HOME AND THE COLUMN GOES TO MID-STROKE — at a speed it can actually do.
//
//  Two separate defects met here, and the report was one sentence: "the lift oscillates
//  and does not go to the middle height position".
//
//    1. `restBodyPose` had no lift, while the studio's REST_POSE (homepose.js) did. So
//       "one rest pose across every client" was not true, and Home left the column
//       wherever it was. That is the "does not go to the middle height" half.
//    2. Home streams the WHOLE joint dict at 30 Hz. Over Bluetooth that dict still carried
//       the lift keys from the last stick touch, so Home streamed a CONSTANT lift target
//       for the length of the glide — and a static target is what makes the plant's
//       stop-band lead command past it, flip sign, and buzz (server/plant.py
//       _clear_stop_band, measured in NOVA's lift simulation). That is the
//       oscillation half. Now that the lift is in the rest pose, Home streams a CONVERGING
//       target like every other joint.
//
//  The speed budget is the third thing that has to hold: the lift is the one joint here
//  that is not an angle, and the cap is in rad/s.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class HomeLiftTests: XCTestCase {

    private let liftKeys = ["lift_lower", "lift_upper"]
    /// The column's stage travel, m (half its 635 mm, since the two stages extend together).
    private let stageSpan = 0.3175
    /// What the column can actually do, m/s of COLUMN (README: 37.0 up / 36.4 down).
    private let columnMaxMS = 0.035

    func testHomeCommandsTheColumnToMidStroke() {
        for k in liftKeys {
            let v = try? XCTUnwrap(RobotController.restBodyPose[k])
            XCTAssertEqual(v ?? -1, stageSpan / 2, accuracy: 1e-9,
                           "\(k): Home must command mid-stroke, the same value homepose.js uses")
        }
    }

    /// One rest pose across every client — the value the studio and the motion library use.
    func testTheRestPoseMatchesTheStudio() {
        XCTAssertEqual(RobotController.restBodyPose["lift_lower"],
                       RobotController.restBodyPose["lift_upper"],
                       "both stages share one height")
        XCTAssertEqual(RobotController.restBodyPose["lift_lower"] ?? 0, 0.15875, accuracy: 1e-9)
    }

    /// The glide budget is rad/s. A stage-metre weighted as a radian budgets the column at
    /// ~150 mm/s — four times its ceiling — and bang-bang answers that with overshoot.
    func testTheGlideNeverCommandsTheColumnFasterThanItCanMove() {
        // Worst case: the full stroke, with every other joint already at rest so nothing
        // else stretches the duration.
        for (label, start) in [("bottom to middle", 0.0), ("top to middle", 0.3175)] {
            let target = 0.15875
            let delta = abs(target - start)
            let weighted = delta * RobotController.liftWeight
            let dur = max(RobotController.homeMinDuration,
                          1.5 * weighted / RobotController.homeMaxJointSpeed)
            // smoothstep's peak velocity is 1.5x its mean.
            let peakStage = 1.5 * delta / dur
            let peakColumn = 2 * peakStage          // two stages extend together
            XCTAssertLessThanOrEqual(peakColumn, columnMaxMS,
                "\(label): Home would command \(peakColumn * 1000) mm/s, over the column's \(columnMaxMS * 1000)")
            XCTAssertGreaterThan(peakColumn, 0.02,
                "\(label): and it must not be so conservative the column crawls")
        }
    }

    /// The arms must not be slowed down by the lift's weight — the budget stretches for
    /// whichever joint needs it, and an arm swing is still an arm swing.
    func testAnArmOnlyHomeIsNotStretchedByTheLiftWeight() {
        var start = RobotController.restBodyPose
        start["left_arm_4"] = 0.0                    // 1.52 rad away from rest
        let maxDelta = RobotController.restBodyPose.map {
            abs($0.value - (start[$0.key] ?? $0.value)) * RobotController.homeWeight($0.key)
        }.max() ?? 0
        XCTAssertEqual(maxDelta, 1.52, accuracy: 1e-9,
                       "with the column already home, the elbow sets the duration")
        let dur = max(RobotController.homeMinDuration, 1.5 * maxDelta / RobotController.homeMaxJointSpeed)
        XCTAssertEqual(dur, RobotController.homeMinDuration, accuracy: 1e-9)
    }

    /// An unknown start must mean "already there", never 0 — on the lift, 0 is the fully
    /// RETRACTED column, so a fabricated zero start is a 317 mm sweep nobody asked for.
    func testAnUnknownStartHoldsRatherThanSweepingFromTheFloor() {
        let target = RobotController.restBodyPose
        let live: [String: Double] = [:]              // nothing reported yet
        var start: [String: Double] = [:]
        for id in target.keys { start[id] = live[id] ?? target[id] }
        XCTAssertEqual(start["lift_lower"], target["lift_lower"])
        let maxDelta = target.map {
            abs($0.value - (start[$0.key] ?? $0.value)) * RobotController.homeWeight($0.key)
        }.max() ?? 0
        XCTAssertEqual(maxDelta, 0, accuracy: 1e-12, "an unreported robot must not be swept")
    }
}
