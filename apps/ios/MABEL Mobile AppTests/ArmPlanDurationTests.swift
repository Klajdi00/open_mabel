//
//  ArmPlanDurationTests.swift
//  MABEL Mobile AppTests
//
//  The velocity budget for task-space plan playback (Services/ArmPlan.swift).
//
//  Execute used to play every plan over a hard-coded 1.4 s — a duration with no
//  reference to how far the arm actually had to travel. A plan swinging a joint
//  2.5 rad therefore ran at ~2.7 rad/s peak, roughly 3× the 1.0 rad/s budget the
//  Home glide, the web studio's timeline blends and the wrist plan all respect,
//  while a 5° nudge crawled for the same 1.4 s.
//
//  These tests measure the PEAK velocity of the sampled trajectory rather than
//  trusting the formula, so swapping `ArmPlan.ease` for a curve with a different
//  peak/average ratio fails here instead of silently weakening the cap — the same
//  trap the web's `EASE_PEAK` comment warns about (raised cosine π/2 vs
//  smoothstep 1.5).
//

import Testing
import Foundation
@testable import MABEL_Mobile_App

struct ArmPlanDurationTests {

    // a real 7-DOF arm chain
    private static let chain = (1...7).map { "right_arm_\($0)" }
    private static func start() -> [String: Double] {
        Dictionary(uniqueKeysWithValues: chain.map { ($0, 0.0) })
    }

    /// Sample the planned trajectory at `hz` and return the worst per-joint speed.
    private static func peakSpeed(from a: [String: Double], to b: [String: Double],
                                  dur: Double, hz: Double = 60) -> Double {
        let n = Swift.max(2, Int((dur * hz).rounded(.up)))
        var peak = 0.0
        var prev = ArmPlan.sample(from: a, to: b, chain: chain, t: 0)
        for i in 1...n {
            let t = Double(i) / Double(n)
            let cur = ArmPlan.sample(from: a, to: b, chain: chain, t: t)
            let dt = dur / Double(n)
            for (id, v) in cur { peak = Swift.max(peak, abs(v - (prev[id] ?? v)) / dt) }
            prev = cur
        }
        return peak
    }

    @Test("a big swing plays slowly enough to stay under the velocity cap")
    func bigSwingIsCapped() {
        var target = Self.start()
        target["right_arm_2"] = -2.5                       // a 2.5 rad swing
        let dur = ArmPlan.duration(from: Self.start(), to: target, chain: Self.chain)
        #expect(dur > 1.4, "a 2.5 rad swing must play for LONGER than the old 1.4 s")
        let peak = Self.peakSpeed(from: Self.start(), to: target, dur: dur)
        #expect(peak <= ArmPlan.safeBlendVel * 1.05,
                "peak \(peak) rad/s exceeds the \(ArmPlan.safeBlendVel) rad/s budget")
    }

    @Test("the old fixed 1.4 s would have blown the budget — the regression this locks out")
    func fixedDurationWasUnsafe() {
        var target = Self.start()
        target["right_arm_2"] = -2.5
        let peak = Self.peakSpeed(from: Self.start(), to: target, dur: 1.4)
        #expect(peak > ArmPlan.safeBlendVel * 2,
                "expected the old constant to be well over budget, got \(peak)")
    }

    @Test("the cap holds across a sweep of plan sizes")
    func capHoldsAcrossSweep() {
        for delta in stride(from: 0.2, through: 3.0, by: 0.2) {
            var target = Self.start()
            target["right_arm_1"] = -delta                 // within the joint's range
            let dur = ArmPlan.duration(from: Self.start(), to: target, chain: Self.chain)
            let peak = Self.peakSpeed(from: Self.start(), to: target, dur: dur)
            #expect(peak <= ArmPlan.safeBlendVel * 1.05,
                    "delta \(delta): peak \(peak) rad/s over budget (dur \(dur) s)")
        }
    }

    @Test("EASE_PEAK matches the easing actually used")
    func easePeakMatchesEase() {
        // measure peak/average slope of ArmPlan.ease over [0,1]
        let n = 20_000
        var peak = 0.0
        var prev = ArmPlan.ease(0)
        for i in 1...n {
            let e = ArmPlan.ease(Double(i) / Double(n))
            peak = Swift.max(peak, (e - prev) * Double(n))   // d(ease)/dt, average slope = 1
            prev = e
        }
        #expect(abs(peak - ArmPlan.easePeak) < 0.01,
                "ease peaks at \(peak)× average but easePeak says \(ArmPlan.easePeak)")
    }

    @Test("an OUT-OF-RANGE target still respects the budget")
    func outOfRangeTargetStaysInBudget() {
        // IK solutions land outside joint limits routinely. `duration` sizes on
        // the CLAMPED delta, so if `sample` interpolated toward the unclamped
        // value the joint would race to its limit and pin there — measured
        // 2.15 rad/s against a 1.0 cap before `sample` clamped its endpoints.
        var target = Self.start()
        target["right_arm_1"] = -12.0                  // far beyond the limit
        let dur = ArmPlan.duration(from: Self.start(), to: target, chain: Self.chain)
        let peak = Self.peakSpeed(from: Self.start(), to: target, dur: dur)
        #expect(peak <= ArmPlan.safeBlendVel * 1.05,
                "out-of-range target peaks at \(peak) rad/s, over budget")
        // and it must actually stop at the reachable limit, not overshoot
        let end = ArmPlan.sample(from: Self.start(), to: target, chain: Self.chain, t: 1)
        let lo = RobotModel.spec("right_arm_1")?.lower ?? 0
        #expect(abs((end["right_arm_1"] ?? 0) - lo) < 1e-9,
                "should land exactly on the joint limit")
    }

    @Test("a benign nudge is sent directly rather than ramped")
    func subThresholdSkipsTheRamp() {
        var target = Self.start()
        target["right_arm_1"] = 0.05                       // well under blendThresh
        #expect(ArmPlan.duration(from: Self.start(), to: target, chain: Self.chain) == 0)
    }

    @Test("duration is floored and ceilinged")
    func durationIsBounded() {
        var small = Self.start(); small["right_arm_1"] = 0.16   // just over threshold
        let d1 = ArmPlan.duration(from: Self.start(), to: small, chain: Self.chain)
        #expect(d1 >= ArmPlan.minDuration)

        var huge = Self.start(); huge["right_arm_1"] = -3.4     // near the joint limit
        let d2 = ArmPlan.duration(from: Self.start(), to: huge, chain: Self.chain)
        #expect(d2 <= ArmPlan.maxDuration)
    }
}
