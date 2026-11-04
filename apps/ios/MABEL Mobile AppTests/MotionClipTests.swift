//
//  MotionClipTests.swift
//  MABEL Mobile AppTests
//
//  The app's motion playback is a PORT of the web studio's registry, and a port
//  is only worth having if something fails when it drifts. These are that
//  something.
//
//  Three classes of assertion:
//    1. PARITY — the constants and the sampled motion match the values
//       `gen_motion_clips.py` recorded straight out of `motions.js`. A number
//       typed from memory into `MotionMath` fails here.
//    2. BUDGET — no clip, and no blend the safety guard sizes, asks a joint to
//       move faster than the velocity budget every MABEL client shares.
//    3. SEMANTICS — hold/clamp behaviour, and that a pose clip's zeros are
//       COMMANDS (pose home is nothing but zeros; dropping them would make
//       "go home" a no-op).
//

import XCTest
@testable import MABEL_Mobile_App

@MainActor
final class MotionClipTests: XCTestCase {

    private var table: MotionClips { MotionClips.shared }

    // MARK: - the table loaded at all

    func testBakedTableIsPresentAndComplete() {
        XCTAssertFalse(table.clips.isEmpty,
            "motion_clips.json missing from the bundle — run simulation/tools/gen_motion_clips.py")
        XCTAssertGreaterThan(table.clips.count, 40, "the studio ships 44 clips")
        // Everything the catalogue offers must be playable or explicitly a goal.
        let catalog = ActionCatalog()
        for spec in catalog.actions where spec.available {
            if spec.id == "say" || spec.id == "say_wait" { continue }   // app-only speech blocks
            XCTAssertNotNil(table.clips[spec.id],
                "\(spec.id) is offered in the library but has no motion in the table")
        }
    }

    // MARK: - 1. parity with the studio

    func testConstantsMatchTheStudio() {
        let c = table.constants
        XCTAssertEqual(MotionMath.navSpeed, c["navSpeed"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.navSpeedStrafe, c["navSpeedStrafe"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.navYawRate, c["navYawRate"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.safeBlendVel, c["safeBlendVel"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.easePeak, c["easePeak"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.blendThresh, c["blendThresh"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.blendMinDur, c["blendMinDur"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.maxJointVel, c["maxJointVel"] ?? 0, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.gripScale, c["gripScale"] ?? 0, accuracy: 1e-9)
    }

    func testOneVelocityBudgetAcrossTheApp() {
        // `ArmPlan` (task-space execute) and the timeline must not disagree about
        // how fast commanded motion may go.
        XCTAssertEqual(MotionMath.safeBlendVel, ArmPlan.safeBlendVel, accuracy: 1e-9)
        XCTAssertEqual(MotionMath.blendThresh, ArmPlan.blendThresh, accuracy: 1e-9)
    }

    func testEasePeakIsDerivedFromTheEaseItDescribes() {
        // Measure the peak/average ratio of `ease` numerically instead of
        // trusting the literal: change the curve and this fails.
        let n = 20_000
        var peak = 0.0
        for i in 0..<n {
            let a = MotionMath.ease(Double(i) / Double(n))
            let b = MotionMath.ease(Double(i + 1) / Double(n))
            peak = max(peak, abs(b - a) * Double(n))
        }
        XCTAssertEqual(peak, MotionMath.easePeak, accuracy: 0.01,
            "easePeak must be the peak/average velocity ratio of `ease` itself")
    }

    func testNavProfileMatchesTheStudioGolden() {
        let golden = table.navProfileGolden
        XCTAssertEqual(golden.count, 9)
        for (i, want) in golden.enumerated() {
            let t = 4.0 * Double(i) / 8.0
            XCTAssertEqual(MotionMath.navProfile(t, 4), want, accuracy: 1e-3,
                "navProfile diverged from the studio at t=\(t)")
        }
    }

    func testNavPrimitivesMatchTheStudioGolden() {
        for (id, golden) in table.navGolden {
            guard MotionMath.navPrimitives[id] != nil else { continue }
            let d = table.duration(id, params: defaults(id))
            for (i, row) in golden.enumerated() {
                let t = d * Double(i) / 8.0
                let s = table.nav(id, at: t, params: defaults(id)) ?? .zero
                XCTAssertEqual(s.lx, row[0], accuracy: 2e-3, "\(id) lx @ \(t)")
                XCTAssertEqual(s.ly, row[1], accuracy: 2e-3, "\(id) ly @ \(t)")
                XCTAssertEqual(s.rx, row[2], accuracy: 2e-3, "\(id) rx @ \(t)")
            }
        }
    }

    func testBakedNavClipsMatchTheirGolden() {
        // The clips that sway the base while they dance are table-driven, not
        // analytic — check those against the same golden rows.
        for (id, golden) in table.navGolden {
            guard MotionMath.navPrimitives[id] == nil, id != "nav_hold",
                  let clip = table.clips[id], !clip.navTracks.isEmpty else { continue }
            let d = clip.duration
            for (i, row) in golden.enumerated() {
                let s = table.nav(id, at: d * Double(i) / 8.0) ?? .zero
                XCTAssertEqual(s.lx, row[0], accuracy: 2e-3, "\(id) lx frame \(i)")
                XCTAssertEqual(s.ly, row[1], accuracy: 2e-3, "\(id) ly frame \(i)")
                XCTAssertEqual(s.rx, row[2], accuracy: 2e-3, "\(id) rx frame \(i)")
            }
        }
    }

    func testNavDurationsFollowTheirParameter() {
        // 0.5 m at half of 0.8 m/s, plus the ramp allowance: 0.5/0.4 + 0.6.
        XCTAssertEqual(table.duration("nav_forward", params: ["dist": 0.5]), 1.85, accuracy: 1e-6)
        XCTAssertEqual(table.duration("nav_forward", params: ["dist": 2.0]), 5.6, accuracy: 1e-6)
        // 90° at half of 1.5 rad/s.
        XCTAssertEqual(table.duration("nav_turn_left", params: ["angle": 90]),
                       .pi / 2 / 0.75 + 0.6, accuracy: 1e-6)
        XCTAssertEqual(table.duration("nav_hold", params: ["secs": 7]), 7, accuracy: 1e-9)
    }

    // MARK: - 2. the velocity budget

    func testNoClipExceedsTheJointVelocityBudget() {
        // Sample each baked clip at the executor's own rate — the rate at which
        // the studio's numbers were measured — and check the worst step.
        //
        // RAW joint values, deliberately: `MAX_JOINT_VEL` is the studio's budget
        // on the actuator command stream, where the lift's unit is metres and
        // not a radian-equivalent. `blendScale` exists to weigh a POSE DISTANCE
        // when sizing a blend, which is a different question — applying it here
        // would fail `happy` at a nominal 7.3 "rad/s" that is really 0.73 m/s of
        // lift travel.
        for (id, clip) in table.clips where !clip.tracks.isEmpty {
            var peak = 0.0
            var worst = ""
            let dt = 1.0 / table.rate
            var t = 0.0
            var previous = table.joints(id, at: 0) ?? [:]
            while t < clip.duration {
                t += dt
                guard let now = table.joints(id, at: t) else { break }
                for (j, v) in now {
                    let speed = abs(v - (previous[j] ?? v)) / dt
                    if speed > peak { peak = speed; worst = j }
                }
                previous = now
            }
            XCTAssertLessThanOrEqual(peak, MotionMath.maxJointVel + 1e-6,
                "\(id) drives \(worst) at \(peak) rad/s, over the \(MotionMath.maxJointVel) budget")
        }
    }

    func testABlendIsAlwaysLongEnoughToStayInBudget() {
        // The guard's whole job: a blend sized for a jump must keep the eased
        // interpolation's PEAK speed under safeBlendVel. Check the real sampled
        // motion, not the formula that produced the duration.
        for delta in [0.16, 0.4, 1.0, 2.5, 3.5] {
            let d = MotionMath.blendDuration(for: delta)
            let from = ["left_arm_1": 0.0]
            let to = ["left_arm_1": delta]
            var peak = 0.0
            let dt = 1.0 / 200.0
            var t = 0.0
            var previous = MotionMath.blendPose(from: from, to: to, at: 0, duration: d)
            while t < d {
                t += dt
                let now = MotionMath.blendPose(from: from, to: to, at: t, duration: d)
                peak = max(peak, abs((now["left_arm_1"] ?? 0) - (previous["left_arm_1"] ?? 0)) / dt)
                previous = now
            }
            XCTAssertLessThanOrEqual(peak, MotionMath.safeBlendVel + 0.02,
                "a \(delta) rad jump blended over \(d) s peaks at \(peak) rad/s")
            XCTAssertGreaterThanOrEqual(d, MotionMath.blendMinDur,
                "a blend must never be shorter than its floor")
        }
    }

    func testBlendDurationNeverRoundsDown() {
        // A 1.0 rad jump needs π/2 s exactly; rounding to a centisecond must go UP.
        let d = MotionMath.blendDuration(for: 1.0)
        XCTAssertGreaterThanOrEqual(d, MotionMath.easePeak * 1.0 / MotionMath.safeBlendVel)
    }

    func testGripsAndTheLiftAreWeighedOnTheJointScale() {
        // A full open→fist is 1.0 of grip but 1.9 rad of finger travel; the whole
        // lift is 0.318 m but a whole-body move. Weighing them as 1.0 would size
        // their blends far too short.
        let grip = MotionMath.poseDelta(from: ["left_grip": 0], to: ["left_grip": 1])
        XCTAssertEqual(grip.delta, MotionMath.gripScale, accuracy: 1e-9)
        let lift = MotionMath.poseDelta(from: ["lift_lower": 0], to: ["lift_lower": 0.3])
        XCTAssertEqual(lift.delta, 3.0, accuracy: 1e-9)
    }

    // MARK: - 3. semantics

    func testAPoseClipsHeldJointsAreCommands() {
        // `pose_home` IS the rest pose: nothing in it moves, so every one of its
        // rows collapses into `holds`. If the table dropped flat rows instead of
        // recording their value, "go home" would command nothing at all — and
        // when the flat value stopped being zero (the elbows rest at ±1.52, the
        // lift at 0.15875, the wrists at ±0.15), a `zeros` list of NAMES could
        // not have carried it either.
        let pose = table.joints("pose_home", at: 1.0)
        XCTAssertNotNil(pose)
        XCTAssertGreaterThan(pose?.count ?? 0, 15, "home must command the whole body")
        for (j, v) in pose ?? [:] {
            XCTAssertEqual(v, table.rest[j] ?? 0, accuracy: 1e-9,
                           "\(j) should be commanded at its rest value")
        }
        XCTAssertEqual(pose?["left_arm_4"] ?? .nan, 1.52, accuracy: 1e-6,
                       "the elbow rests EXTENDED, not at the 90\u{00B0} bend of qpos zero")
    }

    func testTheAppsHomePoseIsTheStudiosRestPose() {
        // One rest pose across every client. The generator records the studio's
        // REST in the table; this is the gate that the app's own copy agrees.
        for (j, v) in table.rest {
            guard let mine = RobotModel.homePose[j] else {
                continue        // a joint the app's model does not carry
            }
            XCTAssertEqual(mine, v, accuracy: 1e-4,
                           "RobotModel.homePose[\(j)] = \(mine) but the studio rests at \(v)")
        }
        XCTAssertFalse(table.rest.isEmpty, "the table must carry the rest pose")
    }

    func testSamplingClampsPastTheEnds() {
        let end = table.joints("wave", at: 5.0)
        let past = table.joints("wave", at: 50.0)
        XCTAssertEqual(end?.count, past?.count)
        for (j, v) in end ?? [:] {
            XCTAssertEqual(past?[j] ?? .nan, v, accuracy: 1e-9,
                "sampling past a clip's end must hold its final pose")
        }
        let before = table.joints("wave", at: -3)
        for (j, v) in table.joints("wave", at: 0) ?? [:] {
            XCTAssertEqual(before?[j] ?? .nan, v, accuracy: 1e-9)
        }
    }

    func testEveryClipStartsAtHomeSoNothingJumpsIntoIt() {
        // The half of the studio's convention that actually holds: every clip
        // RAMPS UP from rest, so dropping one after anything is safe at its
        // leading edge. (The other half — "and ends there too" — does NOT hold:
        // `pickup` finishes far from rest, holding what it picked up. That is
        // why the pose-jump guard exists, and it is measured below.)
        //
        // "From rest" is now literal: the studio plays every clip inside a REST
        // anchor, so this is exact rather than approximate. It used to be
        // measured against ZERO, which happened to work only because the clips
        // were authored in the qpos-zero frame — the frame in which the robot
        // stands with both elbows bent 90°.
        for (id, clip) in table.clips where clip.hasJoints {
            guard let first = table.joints(id, at: 0) else { continue }
            let worst = first.map {
                abs($0.value - (table.rest[$0.key] ?? 0)) * (MotionMath.blendScale[$0.key] ?? 1)
            }.max() ?? 0
            XCTAssertLessThan(worst, MotionMath.blendThresh,
                "\(id) starts \(worst) rad from rest — it would step on its first tick")
        }
    }

    func testTheClipsThatDoNotReturnHomeAreTheOnesWeThinkTheyAre() {
        // A property of the REGISTRY worth pinning: if this set changes, some
        // clip started or stopped leaving the robot holding a pose, and every
        // timeline that follows it gains or loses a safety blend.
        var stranded: [String: Double] = [:]
        for (id, clip) in table.clips where clip.hasJoints {
            guard let last = table.joints(id, at: clip.duration) else { continue }
            let worst = last.map {
                abs($0.value - (table.rest[$0.key] ?? 0)) * (MotionMath.blendScale[$0.key] ?? 1)
            }.max() ?? 0
            if worst >= MotionMath.blendThresh { stranded[id] = worst }
        }
        // Poses and the lift HOLD by design — that is what a pose is.
        let expressive = stranded.keys.filter {
            !$0.hasPrefix("pose_") && !$0.hasPrefix("lift_")
        }.sorted()
        // `pickup` ends carrying what it picked up; `sleep` is `hold` and stays
        // asleep. Both are deliberate, and both are why the guard exists.
        XCTAssertEqual(expressive, ["pickup", "sleep"],
            "clips that end away from rest: \(stranded)")
    }

    func testBlendHoldsJointsTheTargetDoesNotName() {
        let pose = MotionMath.blendPose(from: ["a": 1.0, "b": 0.5], to: ["a": 0.0],
                                        at: 0.2, duration: 0.4)
        XCTAssertEqual(pose["b"] ?? .nan, 0.5, accuracy: 1e-9,
            "a blend must not invent motion on a joint the next block does not command")
        XCTAssertEqual(pose["a"] ?? .nan, 0.5, accuracy: 1e-6, "halfway through a raised cosine")
    }

    // MARK: - helpers

    private func defaults(_ id: String) -> [String: Double] {
        ActionCatalog().spec(id)?.defaults ?? [:]
    }
}
