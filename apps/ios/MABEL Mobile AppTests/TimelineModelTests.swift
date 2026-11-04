//
//  TimelineModelTests.swift
//  MABEL Mobile AppTests
//
//  The choreography model's rules, pinned. These are the invariants a timeline
//  UI leans on and cannot check for itself: two motions can never occupy one
//  resource at one instant, a trim can never produce a block too short to be a
//  motion, a cut half plays ITS half of the clip, and — the one that matters on
//  hardware — no edit may leave a pose discontinuity the executor would command
//  in a single 50 ms tick.
//

import XCTest
@testable import MABEL_Mobile_App

@MainActor
final class TimelineModelTests: XCTestCase {

    private func makeDoc() -> (TimelineDocument, ActionCatalog) {
        let cat = ActionCatalog()
        return (TimelineDocument(catalog: cat), cat)
    }

    private func spec(_ cat: ActionCatalog, _ id: String) throws -> ActionSpec {
        try XCTUnwrap(cat.spec(id), "\(id) missing from the registry")
    }

    private func block(_ doc: TimelineDocument, _ id: UUID) throws -> TimelineBlock {
        try XCTUnwrap(doc.timeline.blocks.first { $0.id == id })
    }

    // MARK: - the registry

    func testRegistryCarriesTheStudioCatalogue() {
        let cat = ActionCatalog()
        XCTAssertGreaterThan(cat.actions.count, 40,
            "the generated catalogue should carry the studio's clips")
        for lane in ActionLane.tracks {
            XCTAssertTrue(cat.actions.contains { $0.lane == lane || $0.lane == .both },
                          "no actions for the \(lane.title) track")
        }
        XCTAssertTrue(cat.actions.contains { $0.id == "say" && $0.available },
                      "the Voice track needs a speech block")
    }

    // MARK: - placement

    func testAddingTwoClipsToOneRowNeverOverlaps() throws {
        let (doc, cat) = makeDoc()
        let wave = try spec(cat, "wave")
        doc.add(wave, at: 0)
        doc.add(wave, at: 0)                       // asked for the SAME instant
        let lane = doc.timeline.blocks(in: .manipulation, row: 0).filter { !$0.isBlend }
        XCTAssertEqual(lane.count, 2)
        XCTAssertGreaterThanOrEqual(lane[1].start, lane[0].end - 1e-6,
            "a second clip on one arm must be pushed clear, not stacked")
    }

    func testStackedRowsMayOverlapDeliberately() throws {
        let (doc, cat) = makeDoc()
        let wave = try spec(cat, "wave")
        doc.addRow(.manipulation)
        XCTAssertEqual(doc.timeline.rowCount(.manipulation), 2)
        let a = try XCTUnwrap(doc.add(wave, at: 0, row: 0).first)
        let b = try XCTUnwrap(doc.add(wave, at: 0, row: 1).first)
        XCTAssertEqual(try block(doc, a).start, 0, accuracy: 1e-9)
        XCTAssertEqual(try block(doc, b).start, 0, accuracy: 1e-9,
            "stacking is the whole point of a second row — it must not push")
        XCTAssertNotEqual(try block(doc, a).row, try block(doc, b).row)
    }

    func testARowWithBlocksOnItCannotBeRemoved() throws {
        let (doc, cat) = makeDoc()
        doc.addRow(.navigation)
        doc.add(try spec(cat, "nav_forward"), at: 0, row: 1)
        doc.removeEmptyRow(.navigation)
        XCTAssertEqual(doc.timeline.rowCount(.navigation), 2, "that row is in use")
    }

    func testBothLaneActionDropsALinkedPair() throws {
        let (doc, cat) = makeDoc()
        let groove = try spec(cat, "groove")
        XCTAssertEqual(groove.lane, .both)
        let ids = doc.add(groove, at: 0)
        XCTAssertEqual(ids.count, 2, "a `both` action occupies two tracks")
        let links = Set(doc.timeline.blocks.compactMap(\.linkID))
        XCTAssertEqual(links.count, 1, "the pair must share one link id")
    }

    func testMovingOneHalfMovesItsTwin() throws {
        let (doc, cat) = makeDoc()
        let ids = doc.add(try spec(cat, "groove"), at: 0)
        doc.move(try XCTUnwrap(ids.first), to: 5)
        let starts = Set(doc.timeline.blocks.filter { !$0.isBlend }.map(\.start))
        XCTAssertEqual(starts, [5], "linked halves move together")
    }

    // MARK: - snapping

    func testSnapPrefersABlockEdgeThenTheGrid() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 2).first)
        let end = try block(doc, id).end
        // Just inside the tolerance of the block's tail → snaps to the tail.
        let near = doc.snap(end + 0.05, excluding: UUID())
        XCTAssertEqual(near.time, end, accuracy: 1e-6)
        XCTAssertTrue(near.magnetic)
        // Well clear of everything → the deterministic 0.1 s grid.
        let far = doc.snap(13.37, excluding: UUID())
        XCTAssertEqual(far.time, 13.4, accuracy: 1e-6)
        XCTAssertFalse(far.magnetic)
    }

    func testSnapCatchesThePlayhead() throws {
        let (doc, _) = makeDoc()
        let s = doc.snap(4.04, playhead: 4.0)
        XCTAssertEqual(s.time, 4.0, accuracy: 1e-6)
    }

    // MARK: - trim

    func testTrimCannotProduceAGlitchLengthBlock() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 0).first)
        doc.trim(id, edge: .trailing, to: 0.05)     // far below the floor
        XCTAssertGreaterThanOrEqual(try block(doc, id).duration, kMinBlockDuration - 1e-9,
            "a block shorter than the blend floor asks the robot for a step change")
    }

    func testFixedLengthClipCannotBeStretchedButAHoldCan() throws {
        let (doc, cat) = makeDoc()
        let wave = try spec(cat, "wave")
        XCTAssertFalse(wave.holdable)
        let fixed = try XCTUnwrap(doc.add(wave, at: 0).first)
        doc.trim(fixed, edge: .trailing, to: 99)
        XCTAssertEqual(try block(doc, fixed).duration, wave.duration, accuracy: 1e-6,
            "a clip with fixed motion math cannot be stretched past its length")

        let hold = try spec(cat, "nav_hold")
        XCTAssertTrue(hold.holdable)
        let held = try XCTUnwrap(doc.add(hold, at: 20).first)
        doc.trim(held, edge: .trailing, to: 20 + 30)
        XCTAssertEqual(try block(doc, held).duration, 30, accuracy: 1e-6, "a hold stretches")
    }

    func testTrimmingTheLeadingEdgeKeepsTheTailPutAndWalksIntoTheClip() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 2).first)
        let end = try block(doc, id).end
        doc.trim(id, edge: .leading, to: 3)
        let b = try block(doc, id)
        XCTAssertEqual(b.start, 3, accuracy: 1e-6)
        XCTAssertEqual(b.end, end, accuracy: 1e-6, "the tail must not move")
        XCTAssertEqual(b.inPoint, 1, accuracy: 1e-6,
            "trimming a second off the head must skip the clip's first second")
    }

    // MARK: - split

    func testSplitProducesTwoPlayableHalvesThatPartitionTheClip() throws {
        let (doc, cat) = makeDoc()
        let wave = try spec(cat, "wave")
        let id = try XCTUnwrap(doc.add(wave, at: 0).first)
        doc.split(id, at: 2)
        let halves = doc.timeline.blocks(in: .manipulation, row: 0).filter { !$0.isBlend }
        XCTAssertEqual(halves.count, 2)
        XCTAssertEqual(halves[0].duration + halves[1].duration, wave.duration, accuracy: 1e-6,
            "a cut conserves the clip's length")
        XCTAssertEqual(halves[1].start, halves[0].end, accuracy: 1e-6, "no gap at the cut")
        XCTAssertEqual(halves[0].outPoint, halves[1].inPoint, accuracy: 1e-6,
            "the halves must partition the clip, not both replay its head")
        for h in halves { XCTAssertGreaterThanOrEqual(h.duration, kMinBlockDuration - 1e-9) }
    }

    func testTheCutIsInvisibleWhenTheHalvesPlayBackToBack() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 0).first)
        let before = stride(from: 0.0, through: 5.0, by: 0.1).map { doc.pose(at: $0) }
        doc.split(id, at: 2)
        let after = stride(from: 0.0, through: 5.0, by: 0.1).map { doc.pose(at: $0) }
        for (i, pair) in zip(before, after).enumerated() {
            guard let a = pair.0, let b = pair.1 else { continue }
            for (j, v) in a {
                XCTAssertEqual(b[j] ?? .nan, v, accuracy: 1e-6,
                    "cutting changed the motion at sample \(i), joint \(j)")
            }
        }
    }

    func testSplitTooCloseToAnEdgeIsRefused() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 0).first)
        doc.split(id, at: 0.05)
        XCTAssertEqual(doc.timeline.blocks.filter { !$0.isBlend }.count, 1,
                       "that cut would leave a glitch")
    }

    // MARK: - the pose-jump guard

    /// The load-bearing test. A pose clip steps the whole body to its target on
    /// its first tick; dropped after an animation, the executor would command
    /// the entire delta in one 50 ms frame.
    func testAPoseJumpIsBridgedAutomatically() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)      // ends 1.71 rad from home
        doc.add(try spec(cat, "wave"), at: 12)        // starts AT home
        let blends = doc.timeline.blocks.filter(\.isBlend)
        XCTAssertEqual(blends.count, 1,
            "the robot is still holding what it picked up — that step must be bridged")
        let b = try XCTUnwrap(blends.first)
        XCTAssertEqual(b.end, 12, accuracy: 1e-6, "a blend lands exactly on the jump")
        XCTAssertGreaterThanOrEqual(b.duration, MotionMath.blendMinDur)
    }

    func testTheBridgedTimelineStaysInsideTheVelocityBudget() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)
        doc.add(try spec(cat, "pose_bow"), at: 20)

        // Walk the timeline at the EXECUTOR's rate and measure what it would
        // command, exactly as the robot would receive it.
        let dt = 1.0 / 20.0
        var peak = 0.0, worst = "", when = 0.0
        var previous = doc.pose(at: 0) ?? [:]
        var t = 0.0
        while t < doc.timeline.duration {
            t += dt
            guard let now = doc.pose(at: t) else { continue }
            for (j, v) in now {
                guard let p = previous[j] else { continue }
                let speed = abs(v - p) / dt * (MotionMath.blendScale[j] ?? 1)
                if speed > peak { peak = speed; worst = j; when = t }
            }
            previous = now
        }
        XCTAssertLessThanOrEqual(peak, MotionMath.maxJointVel + 1e-6,
            "auto-blend left \(worst) stepping at \(peak) rad/s at t=\(when)")
    }

    /// Measure what the executor would actually command, at its own rate.
    private func peakCommanded(_ doc: TimelineDocument,
                               through end: TimeInterval)
        -> (peak: Double, joint: String, at: TimeInterval) {
        let dt = 1.0 / 20.0
        var peak = 0.0, worst = "", when = 0.0
        var previous = doc.pose(at: 0) ?? [:]
        var t = 0.0
        while t < end {
            t += dt
            guard let now = doc.pose(at: t) else { continue }
            for (j, v) in now {
                guard let p = previous[j] else { continue }
                let speed = abs(v - p) / dt * (MotionMath.blendScale[j] ?? 1)
                if speed > peak { peak = speed; worst = j; when = t }
            }
            previous = now
        }
        return (peak, worst, when)
    }

    /// THE REGRESSION THAT MATTERS. A block ENDING is a boundary too: when a
    /// layered clip stops, whatever is still running underneath takes the joints
    /// back mid-motion. This app's own demo document does exactly that —
    /// `pose_tall` over a running `pickup` — and the guard, which looked only at
    /// block STARTS, reported no jump while the executor commanded 31.75 rad/s.
    func testABlockEndingMidMotionIsBridged() throws {
        let (doc, cat) = makeDoc()
        doc.autoBlend = false
        doc.add(try spec(cat, "pickup"), at: 0)         // 8 s, runs underneath
        doc.addRow(.manipulation)
        doc.add(try spec(cat, "pose_tall"), at: 2.5, row: 1)
        let before = peakCommanded(doc, through: 12)
        XCTAssertGreaterThan(before.peak, 10,
            "the unguarded handover should be violent, measured \(before.peak)")

        doc.autoBlend = true
        let tail = doc.timeline.blocks.first { $0.blend?.edge == .tail }
        XCTAssertNotNil(tail, "a trailing blend belongs where the layered pose ends")
        XCTAssertEqual(try XCTUnwrap(tail).start,
                       try XCTUnwrap(doc.timeline.blocks.first {
                           $0.actionID == "pose_tall" }).end,
                       accuracy: 1e-6, "it starts exactly at the handover")
        let after = peakCommanded(doc, through: 12)
        XCTAssertLessThanOrEqual(after.peak, MotionMath.maxJointVel + 1e-2,
            "guarded run still commands \(after.peak) rad/s on \(after.joint) at \(after.at)")
    }

    /// A blend samples both endpoints at its OWN edges, so it is continuous
    /// wherever it sits and may pre-empt the tail of the clip it leaves. Two
    /// blocks with no gap between them are therefore still safe.
    func testBackToBackBlocksAreStillBridged() throws {
        let (doc, cat) = makeDoc()
        let pickup = try spec(cat, "pickup")
        doc.add(pickup, at: 0)
        doc.add(try spec(cat, "wave"), at: pickup.duration)
        let b = try XCTUnwrap(doc.timeline.blocks.first(where: \.isBlend))
        XCTAssertLessThan(b.start, pickup.duration - 1e-6,
            "with no gap the blend must overlap the outgoing clip")
        let p = peakCommanded(doc, through: doc.timeline.duration + 1)
        XCTAssertLessThanOrEqual(p.peak, MotionMath.maxJointVel + 1e-2,
            "back-to-back run commands \(p.peak) rad/s on \(p.joint) at \(p.at)")
    }

    /// THE RULE OF THUMB, as a test: whatever you compose, the pose the executor
    /// commands is continuous. Not "the library is safe" (that is per clip) and
    /// not "this junction is safe" (that is one case) — every junction, at the
    /// rate the robot receives.
    ///
    /// Cutting is what exposes it. A clip cut in half ends wherever it happened
    /// to be mid-motion, so the guard cannot lean on "clips start and end at
    /// rest": both the outgoing pose and the gap are arbitrary.
    func testAnyCutFollowedByAnythingStaysContinuous() throws {
        let from = ["wave", "pickup", "happy", "pose_tuck", "lift_up", "tai_chi"]
        let onto = ["wave", "pose_home", "celebrate", "angry", "pose_tuck", "lift_down"]
        for k in from {
            for j in onto {
                for gap in [0.0, 0.3, 1.0, 3.0] {
                    let (doc, cat) = makeDoc()
                    guard let a = try? spec(cat, k), let b = try? spec(cat, j) else { continue }
                    doc.add(a, at: 0)
                    guard let first = doc.timeline.blocks.first else { continue }
                    let half = first.end / 2
                    doc.split(first.id, at: half)
                    // Drop the right half: the left one now ends mid-motion.
                    if let right = doc.timeline.blocks
                        .first(where: { !$0.isBlend && $0.start > half - 1e-6 }) {
                        doc.remove([right.id])
                    }
                    doc.add(b, at: half + gap)
                    let p = peakCommanded(doc, through: doc.timeline.duration + 2)
                    XCTAssertLessThanOrEqual(p.peak, MotionMath.maxJointVel + 1e-9,
                        "\(k) cut, then \(j) after \(gap)s: \(p.peak) rad/s on \(p.joint) at \(p.at)")
                }
            }
        }
    }

    /// A bridge must OWN the pose while it runs. `lift_up` is lane `.navigation`
    /// but commands joints; while it kept the lift underneath a bridge, the
    /// bridge never actually moved the lift, and the tick the lift block ended
    /// the pose underneath was revealed — measured 19.9 rad/s.
    func testABridgeIsNotOverriddenByAnotherLane() throws {
        let (doc, cat) = makeDoc()
        let lift = try spec(cat, "lift_up")
        doc.add(lift, at: 0)
        doc.add(try spec(cat, "wave"), at: lift.duration + 1)
        let p = peakCommanded(doc, through: doc.timeline.duration + 2)
        XCTAssertLessThanOrEqual(p.peak, MotionMath.maxJointVel + 1e-9,
            "lift_up -> wave: \(p.peak) rad/s on \(p.joint) at \(p.at)")
    }

    /// The benign-jump threshold is DERIVED from the budget: one tick of it must
    /// still be inside the budget. 0.15 rad in one 50 ms tick is 3.0 rad/s, and
    /// a jump sitting exactly on that old threshold was waved through.
    func testTheBenignThresholdIsOneTickOfTheBudget() {
        XCTAssertEqual(MotionMath.blendThresh,
                       MotionMath.maxJointVel / MotionMath.execRate, accuracy: 1e-12)
        XCTAssertLessThanOrEqual(MotionMath.blendThresh * MotionMath.execRate,
                                 MotionMath.maxJointVel + 1e-9)
    }

    func testTheGuardIsIdempotent() throws {
        // The studio's bug: a blend hid the jump that justified it, the next
        // reconcile deleted the blend, and the timeline oscillated. Validating
        // twice must be the same as validating once.
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)
        let first = doc.timeline.blocks.filter(\.isBlend)
        doc.validate()
        doc.validate()
        let second = doc.timeline.blocks.filter(\.isBlend)
        XCTAssertEqual(first.count, second.count, "the reconcile must converge")
        XCTAssertEqual(first.first?.start ?? -1, second.first?.start ?? -2, accuracy: 1e-9)
    }

    func testABlendDisappearsWithTheJumpThatJustifiedIt() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        let follower = try XCTUnwrap(doc.add(try spec(cat, "wave"), at: 12).first)
        XCTAssertEqual(doc.timeline.blocks.filter(\.isBlend).count, 1)
        doc.remove([follower])
        XCTAssertTrue(doc.timeline.blocks.filter(\.isBlend).isEmpty,
            "no jump, no guard — a stale blend is motion nobody asked for")
    }

    func testABlendCannotBeBladed() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)
        let blend = try XCTUnwrap(doc.timeline.blocks.first(where: \.isBlend))
        let mid = blend.start + blend.duration / 2
        doc.split(blend.id, at: mid)
        XCTAssertEqual(doc.timeline.blocks.filter(\.isBlend).count, 1,
            "half a blend is a new pose jump")
    }

    func testABlendTrimsOnlyItsLeadingEdgeAndNeverBelowItsFloor() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)
        let blend = try XCTUnwrap(doc.timeline.blocks.first(where: \.isBlend))
        let floor = try XCTUnwrap(blend.blend?.minDuration)

        doc.trim(blend.id, edge: .trailing, to: 20)
        XCTAssertEqual(try block(doc, blend.id).end, 12, accuracy: 1e-6,
            "the trailing edge is anchored to the jump it guards")

        doc.trim(blend.id, edge: .leading, to: 11.99)    // ask for ~nothing
        XCTAssertGreaterThanOrEqual(try block(doc, blend.id).duration, floor - 1e-9)

        doc.trim(blend.id, edge: .leading, to: 8.0)      // a longer, gentler bridge
        XCTAssertEqual(try block(doc, blend.id).duration, 4.0, accuracy: 1e-6)
    }

    func testTurningAutoBlendOffSurfacesTheJumpInstead() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)
        doc.autoBlend = false
        XCTAssertTrue(doc.timeline.blocks.filter(\.isBlend).isEmpty)
        XCTAssertEqual(doc.poseJumps.count, 1, "an unguarded jump must be visible")
        let jump = try XCTUnwrap(doc.poseJumps.first)
        XCTAssertGreaterThan(jump.delta, MotionMath.blendThresh)
        doc.insertBlend(for: jump)
        XCTAssertEqual(doc.timeline.blocks.filter(\.isBlend).count, 1)
        XCTAssertTrue(doc.poseJumps.isEmpty, "the marker clears once it is bridged")
    }

    func testABenignNudgeIsNotBlended() throws {
        // Ramping a 0.1 rad difference only adds latency to a move the operator
        // experiences as instant.
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "wave"), at: 0)
        doc.add(try spec(cat, "wave"), at: 6)           // home → home
        XCTAssertTrue(doc.timeline.blocks.filter(\.isBlend).isEmpty)
        // And a pose clip after an animation is NOT a jump either: every clip in
        // the registry ramps up FROM home, so its first tick is already there.
        doc.add(try spec(cat, "pose_tpose"), at: 12)
        XCTAssertTrue(doc.timeline.blocks.filter(\.isBlend).isEmpty,
            "a clip that ramps from home needs no bridge into it")
    }

    // MARK: - location gaps

    func testAnActionSomewhereElseRaisesAGapAndAutoFixClearsIt() throws {
        let (doc, cat) = makeDoc()
        let coffee = try spec(cat, "make_coffee")
        XCTAssertNotNil(coffee.location, "make coffee happens in a kitchen")
        // Every location-carrying clip in the registry is still `coming_soon`,
        // so the library refuses to place one and the validator has nothing to
        // chew on yet. Insert the block directly — this is exactly the timeline
        // the day that policy's checkpoint deploys, and the validator has to be
        // right before then, not after.
        XCTAssertFalse(coffee.available, "if this ever deploys, use doc.add here")
        doc.startLocation = ActionLocation(x: 0, y: 0, label: "home")
        doc.applyEdit { tl in
            tl.blocks.append(TimelineBlock(actionID: coffee.id, lane: .manipulation,
                                           start: 10, duration: 60))
        }
        XCTAssertEqual(doc.gaps.count, 1, "the robot is never driven to the kitchen")
        let gap = try XCTUnwrap(doc.gaps.first)
        doc.fixGap(gap)
        XCTAssertTrue(doc.gaps.isEmpty, "the inserted move closes the gap")
        XCTAssertTrue(doc.timeline.blocks.contains { $0.actionID == "nav_goto" })
    }

    // MARK: - parameters re-time their block

    func testChangingANavDistanceRelengthsTheBlock() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "nav_forward"), at: 0).first)
        let before = try block(doc, id).duration
        doc.setParam(id, "dist", 2.0)
        let after = try block(doc, id).duration
        XCTAssertGreaterThan(after, before)
        XCTAssertEqual(after, MotionClips.shared.duration("nav_forward", params: ["dist": 2.0]),
                       accuracy: 1e-6,
            "the block must be as long as the drive it now asks for")
    }

    // MARK: - annotations

    func testAnnotationsRoundTrip() throws {
        let (doc, _) = makeDoc()
        let id = doc.addNote("say hello", at: 3)
        XCTAssertEqual(doc.timeline.notes.count, 1)
        doc.moveNote(id, to: 5)
        doc.retimeNote(id, duration: 4)
        doc.setNoteText(id, "wave at the guests")
        let n = try XCTUnwrap(doc.timeline.notes.first)
        XCTAssertEqual(n.start, 5, accuracy: 1e-9)
        XCTAssertEqual(n.duration, 4, accuracy: 1e-9)
        XCTAssertEqual(n.text, "wave at the guests")
        XCTAssertEqual(doc.timeline.duration, 9, accuracy: 1e-9,
            "a note at the end extends the timeline — it is content")
        doc.removeNote(id)
        XCTAssertTrue(doc.timeline.notes.isEmpty)
    }

    // MARK: - keyframes

    func testKeyframesInterpolateAndBlocksWinOverThem() throws {
        let (doc, cat) = makeDoc()
        doc.addKey(at: 0, pose: ["left_arm_1": 0])
        doc.addKey(at: 2, pose: ["left_arm_1": 1.0])
        XCTAssertEqual(doc.keyPose(at: 1)?["left_arm_1"] ?? .nan, 0.5, accuracy: 1e-9)
        XCTAssertEqual(doc.keyPose(at: 9)?["left_arm_1"] ?? .nan, 1.0, accuracy: 1e-9,
            "past the last key the pose holds")

        // A block over a key drives the joints it names; the key still drives
        // the rest, exactly as the studio's executor merges them.
        doc.add(try spec(cat, "pose_tpose"), at: 0)
        let merged = try XCTUnwrap(doc.pose(at: 1.9))
        XCTAssertNotEqual(merged["left_arm_2"] ?? 0, 0, "the block commands the T-pose")
    }

    // MARK: - history

    func testUndoRestoresTheWholeEdit() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "wave"), at: 0)
        XCTAssertEqual(doc.timeline.blocks.count, 1)
        doc.undo()
        XCTAssertTrue(doc.timeline.blocks.isEmpty)
        doc.redo()
        XCTAssertEqual(doc.timeline.blocks.count, 1)
    }

    // MARK: - tidy

    func testCompactClosesGapsPerRow() throws {
        let (doc, cat) = makeDoc()
        let wave = try spec(cat, "wave")
        doc.add(wave, at: 0)
        doc.add(wave, at: 40)
        doc.compact()
        let lane = doc.timeline.blocks(in: .manipulation, row: 0).filter { !$0.isBlend }
        XCTAssertEqual(lane[0].start, 0, accuracy: 1e-6)
        XCTAssertEqual(lane[1].start, lane[0].end, accuracy: 1e-6)
    }

    // MARK: - what the base actually does

    func testANavBlockMovesTheBaseTheDistanceItAdvertises() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "nav_forward"), at: 0)     // default 0.5 m
        let path = doc.integrateBasePath(through: doc.timeline.duration + 0.5)
        let travelled = hypot(path.last?.dx ?? 0, path.last?.dy ?? 0)
        XCTAssertEqual(travelled, 0.5, accuracy: 0.08,
            "a 'drive 0.5 m' block must integrate to ~0.5 m, or the calibration drifted")
    }

    func testATurnBlockTurnsTheAngleItAdvertises() throws {
        let (doc, cat) = makeDoc()
        let id = try XCTUnwrap(doc.add(try spec(cat, "nav_turn_left"), at: 0).first)
        doc.setParam(id, "angle", 90)
        let path = doc.integrateBasePath(through: doc.timeline.duration + 0.5)
        XCTAssertEqual(abs(path.last?.yaw ?? 0), .pi / 2, accuracy: 0.12)
    }

    // MARK: - persistence

    func testADocumentRoundTripsThroughJSON() throws {
        let (doc, cat) = makeDoc()
        doc.add(try spec(cat, "pickup"), at: 0)
        doc.add(try spec(cat, "wave"), at: 12)           // drops a blend in too
        doc.addNote("hello", at: 1)
        doc.addKey(at: 2, pose: ["left_arm_1": 0.4])
        doc.addRow(.manipulation)

        let data = try JSONEncoder().encode(doc.timeline)
        let back = try JSONDecoder().decode(Timeline.self, from: data)
        XCTAssertEqual(back.blocks.count, doc.timeline.blocks.count)
        XCTAssertEqual(back.notes.count, 1)
        XCTAssertEqual(back.keys.count, 1)
        XCTAssertEqual(back.rowCount(.manipulation), 2)
        XCTAssertEqual(back.blocks.first(where: \.isBlend)?.blend?.to.count,
                       doc.timeline.blocks.first(where: \.isBlend)?.blend?.to.count,
                       "a blend's endpoint poses must survive a save")
    }

    // MARK: - what the viewer is actually showing

    /// Scrubbing must POSE the robot, not just move a line. This is the claim
    /// the 3-D viewer rests on and the one a screenshot cannot make: the pose
    /// the viewer renders is the pose the timeline commands at the playhead.
    func testScrubbingPosesTheViewersRobot() throws {
        let (doc, cat) = makeDoc()
        let catalog = cat
        let player = TimelinePlayer(doc: doc, catalog: catalog)
        doc.add(try spec(cat, "wave"), at: 0)
        player.refreshBasePath()

        player.scrub(to: 0)
        let atStart = player.previewPose.joints
        player.scrub(to: 2.0)
        let midWave = player.previewPose.joints

        XCTAssertFalse(atStart.isEmpty, "the viewer must be posed even at t = 0")
        let moved = midWave.contains { key, v in abs(v - (atStart[key] ?? 0)) > 0.2 }
        XCTAssertTrue(moved, "two seconds into a wave the arm must be somewhere else")

        // And it matches what the timeline says is commanded there.
        let commanded = try XCTUnwrap(doc.pose(at: 2.0))
        for (j, v) in commanded {
            XCTAssertEqual(midWave[j] ?? .nan, v, accuracy: 1e-9,
                "the viewer is rendering something other than the commanded pose")
        }
    }

    /// A nav block has to MOVE the previewed base, or "drive forward" looks
    /// identical to "do nothing".
    func testTheViewersBaseDrivesForANavBlock() throws {
        let (doc, cat) = makeDoc()
        let player = TimelinePlayer(doc: doc, catalog: cat)
        doc.add(try spec(cat, "nav_forward"), at: 0)
        player.refreshBasePath()

        player.scrub(to: 0)
        XCTAssertEqual(player.previewPose.baseX, 0, accuracy: 1e-9)
        player.scrub(to: doc.timeline.duration)
        let travelled = hypot(player.previewPose.baseX, player.previewPose.baseY)
        XCTAssertEqual(travelled, 0.5, accuracy: 0.08,
            "the previewed base must drive the distance the block advertises")
    }

    // MARK: - the run profile

    func testNavProfileEasesFromRestAndBackToRest() {
        XCTAssertEqual(MotionMath.navProfile(0, 4), 0, accuracy: 1e-6,
            "a block that starts at full speed is a step command to the base")
        XCTAssertEqual(MotionMath.navProfile(4, 4), 0, accuracy: 1e-6)
        XCTAssertEqual(MotionMath.navProfile(2, 4), 1, accuracy: 1e-6, "cruises in the middle")
    }
}
