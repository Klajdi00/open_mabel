//
//  LiftAdoptionTests.swift
//  MabelTeleopTests
//
//  A TRANSPORT SWITCH MUST NOT MOVE THE COLUMN.
//
//  Switching from Wi-Fi to Bluetooth drove the lift to mid-stroke on its own. The cause was
//  not the transport: `jointTargets` resets to `RobotModel.homePose` whenever the robot or
//  link changes, and homePose parks the two lift stages at their ref — mid-stroke — because
//  that is the right pose to DRAW a robot we know nothing about. The moment anything
//  streamed those targets, that drawn default became a commanded height, and the heaviest
//  axis on the robot moved with nobody touching a control.
//
//  The rule these tests pin is the operator's: the lift does not move unless the slider,
//  the jog or Home moved it. A new link adopts the MEASURED height first, and until it has,
//  no lift command leaves the app at all.
//

import XCTest
import QuartzCore
@testable import MabelTeleop

/// Records what the controller actually put on the wire. The shared `FakeLink` in this
/// target does not keep the sends, and "was anything sent" is the whole question here.
private final class LiftSpyLink: RobotLink {
    var status: ConnectionStatus = .disconnected { didSet { onStatus?(status) } }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((String, String?, String) -> Void)?
    var onMapList: (([MapInfo], String?) -> Void)?
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    private(set) var sent: [RobotCommand] = []
    /// Observe commands as they are sent, for tests that need WHEN as well as WHAT.
    var onSend: ((RobotCommand) -> Void)?
    func connect() { status = .connected }
    func disconnect() { status = .disconnected }
    func send(_ c: RobotCommand) { sent.append(c); onSend?(c) }
}

@MainActor
final class LiftAdoptionTests: XCTestCase {

    private func lifts(_ sent: [RobotCommand]) -> [(String, Double)] {
        sent.compactMap {
            if case let .jointTarget(id, value) = $0, id.hasPrefix("lift_") { return (id, value) }
            return nil
        }
    }

    /// The exact reported bug: connect, stream the default pose, and nothing must reach the
    /// column.
    func testAFreshLinkNeverCommandsTheLift() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        // Whatever the UI echoes on arrival — here the untouched home default.
        c.setJoints(["lift_lower": 0.15875, "lift_upper": 0.15875, "left_arm_4": 1.52])

        XCTAssertTrue(lifts(link.sent).isEmpty,
                      "a lift target may not be sent before the app knows where the column is")
        XCTAssertFalse(link.sent.isEmpty, "…while other joints still command normally")
    }

    /// And after telemetry arrives, the app's own target IS the measured height — so the
    /// next incidental re-send asks for exactly where the column already is.
    func testTelemetryIsAdoptedAsTheTargetWithoutCommandingAnything() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        var s = RobotState()
        s.joints["lift_lower"] = 0.05; s.joints["lift_upper"] = 0.05   // column low, not at home
        s.reportedJoints = ["lift_lower", "lift_upper"]                // the robot SAID so
        link.onState?(s)

        XCTAssertEqual(c.jointTargets["lift_lower"] ?? -1, 0.05, accuracy: 1e-9)
        XCTAssertEqual(c.jointTargets["lift_upper"] ?? -1, 0.05, accuracy: 1e-9)
        XCTAssertTrue(lifts(link.sent).isEmpty, "adopting is learning, never commanding")

        // The re-send now asks for where it already is: no motion.
        c.setJoints(["lift_lower": c.jointTargets["lift_lower"]!,
                     "lift_upper": c.jointTargets["lift_upper"]!])
        for (_, v) in lifts(link.sent) { XCTAssertEqual(v, 0.05, accuracy: 1e-9) }
    }

    /// A deliberate move still works, immediately, and is the only thing that does.
    func testTheSliderStillMovesIt() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        c.setLiftHeight(0.40)                                   // the operator dragged it

        let sent = lifts(link.sent)
        XCTAssertEqual(sent.count, 2, "both stages are commanded together")
        for (_, v) in sent { XCTAssertEqual(v, 0.20, accuracy: 1e-9, "0.40 m of column = 0.20 per stage") }
    }

    /// Switching transports again must not inherit the previous link's adoption.
    func testSwitchingLinksRequiresAdoptingAgain() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()
        var s = RobotState()
        s.joints["lift_lower"] = 0.05; s.joints["lift_upper"] = 0.05
        s.reportedJoints = ["lift_lower", "lift_upper"]
        link.onState?(s)
        c.setLiftHeight(0.40)                                   // owned, and moving

        c.disconnect()                                          // …then the link changes
        let before = link.sent.count
        c.setJoints(["lift_lower": 0.15875, "lift_upper": 0.15875])

        XCTAssertTrue(lifts(Array(link.sent.dropFirst(before))).isEmpty,
                      "a new link starts unknown again — the default must not command it")
    }

    /// A robot that never mentions the lift must never unlock it. `RobotState.joints` is
    /// pre-filled with homePose, so before `reportedJoints` existed this frame looked exactly
    /// like "the column is at mid-stroke" and adoption fired on a value nobody had measured.
    func testAFrameThatNeverMentionsTheLiftDoesNotUnlockIt() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        var s = RobotState()
        s.reportedJoints = ["left_arm_1"]                // arms only; no column on this robot
        link.onState?(s)

        c.setJoints(["lift_lower": 0.15875, "lift_upper": 0.15875])
        XCTAssertTrue(lifts(link.sent).isEmpty, "a default is not a measurement")
    }

    // MARK: - The Home button actually moves the column

    /// THE REPORTED BUG, end to end: press Home and the column must go to mid-stroke.
    /// (On visionOS the status-bar Home button calls `goRestPose()` — `goHome()` is the
    /// one-shot `.goHome` wire command, a different path.)
    /// This drives the real `goHome()` — its 30 Hz timer, its glide, its gate — and reads
    /// what reached the wire, because every earlier test asserted on the POSE CONSTANT and
    /// so could not see that the command was being dropped one layer lower.
    func testPressingHomeCommandsTheColumnToMidStroke() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        // The robot says where the column is: low, and nowhere near home.
        var s = RobotState()
        s.joints["lift_lower"] = 0.05; s.joints["lift_upper"] = 0.05
        s.reportedJoints = ["lift_lower", "lift_upper"]
        link.onState?(s)

        c.goRestPose()
        XCTAssertTrue(c.homing, "the glide has to actually be running")
        spin(until: { !c.homing }, timeout: 25)

        let sent = lifts(link.sent)
        XCTAssertFalse(sent.isEmpty, "Home sent NOTHING to the lift — the button does not work")
        for k in ["lift_lower", "lift_upper"] {
            let last = sent.last { $0.0 == k }?.1
            XCTAssertEqual(try! XCTUnwrap(last), 0.15875, accuracy: 1e-6,
                           "\(k) must finish at mid-stroke")
        }
    }

    /// …and it goes ONCE. The column is bang-bang: streaming it a trajectory slower than
    /// its single speed makes it stop-start the whole way (measured: 124 stops across four
    /// presses streamed, 4 one-shot). So Home must send exactly one target per stage and
    /// then leave the column alone while the arms glide.
    func testTheColumnIsCommandedOnceNotStreamed() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()
        var s = RobotState()
        s.joints["lift_lower"] = 0.05; s.joints["lift_upper"] = 0.05
        s.reportedJoints = ["lift_lower", "lift_upper"]
        link.onState?(s)

        c.goRestPose()
        spin(until: { !c.homing }, timeout: 25)

        for k in ["lift_lower", "lift_upper"] {
            let n = lifts(link.sent).filter { $0.0 == k }.count
            XCTAssertEqual(n, 1, "\(k) was sent \(n) times — the column must not be streamed")
        }
        // The arms, by contrast, ARE streamed: a PD-tracked joint wants a trajectory.
        let armFrames = link.sent.filter {
            if case let .jointTarget(id, _) = $0 { return id == "left_arm_4" }
            return false
        }.count
        XCTAssertGreaterThan(armFrames, 10, "the arms still glide")
    }

    /// The column's one target must be sent BEFORE the glide, so it starts moving at once
    /// rather than after the arms have finished.
    func testTheColumnStartsMovingAtTheTopOfTheGlide() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()
        var s = RobotState()
        s.joints["lift_lower"] = 0.05; s.joints["lift_upper"] = 0.05
        s.reportedJoints = ["lift_lower", "lift_upper"]
        link.onState?(s)

        c.goRestPose()
        let firstLiftIndex = link.sent.firstIndex {
            if case let .jointTarget(id, _) = $0 { return id.hasPrefix("lift_") }
            return false
        }
        XCTAssertNotNil(firstLiftIndex, "the column must be commanded when Home is pressed")
        spin(until: { !c.homing }, timeout: 25)
    }

    /// Dropping the column from the glide must not change how long the ARMS take.
    func testTheGlideIsNotStretchedByTheColumnAnyMore() {
        var start = RobotController.restBodyPose
        start["lift_lower"] = 0.0; start["lift_upper"] = 0.0   // column at the bottom
        let streamed = RobotController.restBodyPose.filter {
            !RobotController.liftJoints.contains($0.key)
        }
        let maxDelta = streamed.map {
            abs($0.value - (start[$0.key] ?? $0.value)) * RobotController.homeWeight($0.key)
        }.max() ?? 0
        let dur = max(RobotController.homeMinDuration, 1.5 * maxDelta / RobotController.homeMaxJointSpeed)
        XCTAssertEqual(dur, RobotController.homeMinDuration, accuracy: 1e-9,
                       "a column 158 mm from home must not stretch the arm glide to 17 s")
    }

    /// And the gate still does its job: an INCIDENTAL stream before adoption is dropped.
    /// Home is deliberate, a re-sent default is not, and only the second one is a surprise.
    func testHomeIsAllowedBeforeAdoptionButAnIncidentalStreamIsNot() {
        let link = LiftSpyLink()
        let c = RobotController(link: link)
        link.connect()

        c.setJoints(["lift_lower": 0.15875, "lift_upper": 0.15875])
        XCTAssertTrue(lifts(link.sent).isEmpty, "an unasked-for stream must not reach the column")

        c.goRestPose()
        spin(until: { !c.homing }, timeout: 25)
        XCTAssertFalse(lifts(link.sent).isEmpty,
                       "Home is the operator asking; it must reach the column")
    }

    /// Pump the main run loop until `done`, so the 30 Hz Timer inside goHome() actually fires.
    private func spin(until done: () -> Bool, timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while !done(), Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
    }

}
