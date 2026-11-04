//
//  CalibrationTests.swift
//  MabelTeleopTests
//
//  The calibration capture is the one flow the SIMULATOR CANNOT EXERCISE — it has
//  no hand or head tracking, so the card there can only ever show "bring your hands
//  into view". These tests stand in for the headset: they drive `CalibrationCoach`
//  (the pure maths behind the live card) with synthetic head/wrist positions and
//  assert the operator sees the truth.
//
//  The coach mirrors the server's capture rule (engine.py `_calibrate_step` /
//  hyperparams CAL_*: head→wrist mean, window MEDIAN captured once the window
//  spans ≥0.5 s with n≥3 and std ≤2 cm, held 0.25 s — redesigned 2026-07-19 to be
//  rate-independent and tremor-robust), so these also pin the app's on-screen
//  progress to what the bridge is about to do.
//

import XCTest
import simd
@testable import MabelTeleop

final class CalibrationTests: XCTestCase {

    private let head = SIMD3<Float>(0, 1.6, 0)

    /// A wrist `d` metres in front of the head (−Z is forward in ARKit world).
    private func wrist(_ d: Float) -> SIMD3<Float> { SIMD3(0, 1.6, -d) }

    /// Feed a steady pose for `duration` at 20 Hz starting at t0; returns the last live.
    @discardableResult
    private func hold(_ coach: inout CalibrationCoach,
                      reach: Float,
                      duration: TimeInterval,
                      from t0: TimeInterval = 100,
                      bothHands: Bool = true,
                      jitter: Float = 0) -> CalibrationLive {
        var live = CalibrationLive()
        var t = t0
        var i = 0
        while t <= t0 + duration {
            let j = jitter == 0 ? 0 : (i % 2 == 0 ? jitter : -jitter)
            let w = wrist(reach + j)
            live = coach.observe(head: head,
                                 leftWrist: bothHands ? w : nil,
                                 rightWrist: w,
                                 now: t)
            t += 1.0 / 20.0
            i += 1
        }
        return live
    }

    // MARK: - The failure modes the operator could not previously see

    func testNoHeadReportsNoHead() {
        var coach = CalibrationCoach()
        let live = coach.observe(head: nil, leftWrist: wrist(0.6), rightWrist: wrist(0.6), now: 1)
        XCTAssertFalse(live.hasHead)
        XCTAssertEqual(live.hands, 0)
        XCTAssertFalse(live.steady)
    }

    func testNoHandsReportsZeroTracked() {
        var coach = CalibrationCoach()
        let live = coach.observe(head: head, leftWrist: nil, rightWrist: nil, now: 1)
        XCTAssertTrue(live.hasHead)
        XCTAssertEqual(live.hands, 0)
        XCTAssertEqual(live.reach, 0)
        XCTAssertFalse(live.steady)
    }

    func testOneHandIsReportedAndStillMeasures() {
        var coach = CalibrationCoach()
        let live = coach.observe(head: head, leftWrist: nil, rightWrist: wrist(0.62), now: 1)
        XCTAssertEqual(live.hands, 1)
        XCTAssertEqual(live.reach, 0.62, accuracy: 1e-4)
    }

    /// The card must not claim a measurement for hands tucked at the chest — the
    /// server discards those frames outright (`hr < 0.1`).
    func testReachBelowServerFloorIsRejected() {
        var coach = CalibrationCoach()
        let live = hold(&coach, reach: 0.05, duration: 1.0)
        XCTAssertFalse(live.steady)
        XCTAssertEqual(live.hold, 0)
    }

    // MARK: - The measurement itself

    /// The number on screen must BE the server's quantity: mean head→wrist.
    func testReachIsMeanHeadToWristDistance() {
        var coach = CalibrationCoach()
        let live = coach.observe(head: head,
                                 leftWrist: wrist(0.50),
                                 rightWrist: wrist(0.70),
                                 now: 1)
        XCTAssertEqual(live.hands, 2)
        XCTAssertEqual(live.reach, 0.60, accuracy: 1e-4)   // mean of 0.50 / 0.70
    }

    // MARK: - The hold the operator watches fill

    func testSteadyHoldFillsTheRingAndCompletes() {
        var coach = CalibrationCoach()
        // Needs the window to span before it can call "still", then the hold to run.
        let live = hold(&coach, reach: 0.65, duration: 1.2)
        XCTAssertTrue(live.steady)
        XCTAssertEqual(live.hold, 1.0, accuracy: 1e-6, "a sustained hold must reach 100%")
        XCTAssertEqual(live.reach, 0.65, accuracy: 1e-4)
    }

    func testHoldIsPartwayThroughBeforeItCompletes() {
        var coach = CalibrationCoach()
        // Span minSpan (0.5 s) so `still` latches, then only ~0.15 s of the 0.25 s hold.
        let live = hold(&coach, reach: 0.65, duration: 0.65)
        XCTAssertTrue(live.steady)
        XCTAssertGreaterThan(live.hold, 0)
        XCTAssertLessThan(live.hold, 1.0)
    }

    /// Waving around must NOT read as measuring — the server would not capture it.
    func testJitterBeyondServerSpreadNeverGoesSteady() {
        var coach = CalibrationCoach()
        // ±4 cm alternating ⇒ reach std = 4 cm, past the server's 2 cm stillness rule.
        let live = hold(&coach, reach: 0.65, duration: 1.2, jitter: 0.04)
        XCTAssertFalse(live.steady, "±4 cm of movement must not count as holding still")
        XCTAssertEqual(live.hold, 0)
    }

    /// Tiny tremor is normal and must still capture — a rule that only accepts a
    /// perfectly rigid arm would hang forever on a real person.
    func testHumanTremorWithinSpreadStillCaptures() {
        var coach = CalibrationCoach()
        let live = hold(&coach, reach: 0.65, duration: 1.2, jitter: 0.01)  // ±1 cm
        XCTAssertTrue(live.steady)
        XCTAssertEqual(live.hold, 1.0, accuracy: 1e-6)
    }

    /// Moving mid-hold must reset progress, not creep toward a capture.
    func testMovingMidHoldResetsProgress() {
        var coach = CalibrationCoach()
        _ = hold(&coach, reach: 0.65, duration: 0.9, from: 100)
        // A big jump: 0.65 → 0.45 blows the 2 cm std stillness rule instantly.
        let live = coach.observe(head: head, leftWrist: wrist(0.45), rightWrist: wrist(0.45),
                                 now: 100.95)
        XCTAssertFalse(live.steady)
        XCTAssertEqual(live.hold, 0)
    }

    /// Losing a hand mid-hold must drop the progress — the server stops measuring too.
    func testLosingHandsMidHoldClearsProgress() {
        var coach = CalibrationCoach()
        _ = hold(&coach, reach: 0.65, duration: 1.2)
        let live = coach.observe(head: head, leftWrist: nil, rightWrist: nil, now: 101.3)
        XCTAssertEqual(live.hands, 0)
        XCTAssertFalse(live.steady)
        XCTAssertEqual(live.hold, 0)
    }

    /// After a reset the next hold must start from zero, not resume the old one.
    func testResetClearsTheHold() {
        var coach = CalibrationCoach()
        _ = hold(&coach, reach: 0.65, duration: 1.2)
        coach.reset()
        let live = coach.observe(head: head, leftWrist: wrist(0.65), rightWrist: wrist(0.65),
                                 now: 200)
        XCTAssertFalse(live.steady, "a fresh capture must re-earn the hold")
        XCTAssertEqual(live.hold, 0)
    }

    // MARK: - The app/server contract

    /// The coach exists to echo the bridge. If engine.py's thresholds move and these
    /// don't, the ring fills at a different moment than the capture fires.
    func testThresholdsMatchTheServerCaptureRule() {
        // REDESIGNED 2026-07-19 with the server (rate-independent + tremor-robust:
        // "the arm-length number stabilised" == success). hyperparams.py CAL_*.
        XCTAssertEqual(CalibrationCoach.window, 1.0, accuracy: 1e-9)    // CAL_WINDOW
        XCTAssertEqual(CalibrationCoach.minSpan, 0.5, accuracy: 1e-9)   // CAL_MIN_SPAN
        XCTAssertEqual(CalibrationCoach.stdMax, 0.02, accuracy: 1e-9)   // CAL_STD
        XCTAssertEqual(CalibrationCoach.hold, 0.25, accuracy: 1e-9)     // CAL_HOLD
        XCTAssertEqual(CalibrationCoach.minReach, 0.1, accuracy: 1e-9)  // engine.py: hr < 0.1
    }

    /// The 1-pose wire shape the engine actually emits (total 1 + a `reach` field) —
    /// the facade's older 3-pose status has neither, so a regression there would
    /// silently strand the card on "capturing" forever.
    func testWireProgressDecodesTheEngineOnePoseShape() {
        let p = CalibrationProgress(wire: ["active": true, "pose": "arms_forward",
                                           "index": 1, "total": 1, "done": true,
                                           "reach": 1.24])
        XCTAssertEqual(p?.total, 1)
        XCTAssertEqual(p?.pose, "arms_forward")
        XCTAssertTrue(p?.done ?? false)
        XCTAssertEqual(p?.reach ?? 0, 1.24, accuracy: 1e-9)
    }

    /// A measurement that never happened must not become a profile.
    func testUnmeasuredProfileIsRejected() {
        XCTAssertNil(OperatorMetrics(wire: ["reach": 0.0]), "reach 0 is not a measurement")
        XCTAssertNotNil(OperatorMetrics(wire: ["reach": 1.2, "human_reach": 0.62,
                                               "robot_reach": 0.74]))
    }
}
