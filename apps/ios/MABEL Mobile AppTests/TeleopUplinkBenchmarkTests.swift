//
//  TeleopUplinkBenchmarkTests.swift
//  MABEL Mobile AppTests
//
//  Headless micro-benchmark that isolates the iOS teleop latency ROOT CAUSE and
//  proves the fix — no bridge or UI dragging required, so it runs in CI / the
//  simulator deterministically.
//
//  Root cause: `setJoint(id, value)` mutates the @Published `jointTargets` and
//  rebuilds `livePose` ON EVERY CALL. The green-ball IK (≈60 Hz over the whole arm
//  chain) and the arm joysticks (30 Hz × 6 joints) called it PER JOINT, firing an
//  `objectWillChange` storm that saturated the main thread the WebSocket send/recv
//  + pong path shares → the "super severe" lag (the same trap the in-code `setGrip`
//  comment documents). The web console has none of this: it writes a plain buffer
//  and the 3D reads it in the render loop.
//
//  Fix: batch a whole chain into ONE `setJoints(_:)` → one @Published mutation +
//  one livePose rebuild for the batch. This test quantifies the reduction.
//

import Testing
import Foundation
import Combine
@testable import MABEL_Mobile_App

@MainActor
struct TeleopUplinkBenchmarkTests {

    /// A representative 7-DOF arm chain and a 1-second drag at the 60 Hz IK rate.
    static let chain = ["left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4",
                        "left_arm_5", "left_arm_6", "left_arm_7"]
    static let solves = 60

    @Test func batchedUplinkSlashesMainThreadChurn() async {
        // DemoLink (no network): we measure CONTROLLER-side churn. The same link
        // backs both runs, so any background emissions cancel out in the comparison.
        let c = RobotController()

        func measure(_ body: (Int) -> Void) -> (emissions: Int, ms: Double) {
            var n = 0
            let sub = c.objectWillChange.sink { _ in n += 1 }
            let t0 = Date()
            for i in 0..<Self.solves { body(i) }
            let ms = Date().timeIntervalSince(t0) * 1000.0
            sub.cancel()
            return (n, ms)
        }

        // OLD behaviour — one setJoint() per chain joint per solve (the storm).
        let old = measure { i in
            for (k, id) in Self.chain.enumerated() {
                c.setJoint(id, 0.001 * Double((i + k) % 40))
            }
        }
        // NEW behaviour — one batched setJoints() per solve.
        let new = measure { i in
            var batch: [String: Double] = [:]
            for (k, id) in Self.chain.enumerated() {
                batch[id] = 0.0011 * Double((i + k) % 40)
            }
            c.setJoints(batch)
        }

        let emitX = Double(old.emissions) / Double(max(1, new.emissions))
        let timeX = old.ms / max(0.0001, new.ms)
        print("════════ iOS UPLINK BENCH — 1 s arm drag (\(Self.solves) solves · \(Self.chain.count)-joint chain) ════════")
        print("  OLD  per-joint setJoint : \(old.emissions) objectWillChange   \(String(format: "%6.1f", old.ms)) ms main-thread")
        print("  NEW  batched  setJoints : \(new.emissions) objectWillChange   \(String(format: "%6.1f", new.ms)) ms main-thread")
        print("  →   \(String(format: "%.1f", emitX))× fewer SwiftUI invalidations, \(String(format: "%.1f", timeX))× less main-thread time")
        print("══════════════════════════════════════════════════════════════════════════")

        // The batched path must fire ONE invalidation per solve, not one per joint…
        #expect(new.emissions <= Self.solves + 2)
        // …while the per-joint path fires ~chainLength× as many…
        #expect(old.emissions >= Self.solves * (Self.chain.count - 1))
        // …i.e. a large, unambiguous reduction.
        #expect(Double(new.emissions) * 3.0 < Double(old.emissions))
    }
}
