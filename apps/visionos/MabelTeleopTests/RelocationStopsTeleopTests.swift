//
//  RelocationStopsTeleopTests.swift
//  MabelTeleopTests
//
//  ANYTHING THAT RELOCATES THE ROBOT MUST STOP WHOLE-BODY TELEOP.
//
//  Home, Reset and a scene switch all move the robot out from under the
//  operator. The operator's hands stay exactly where they were, so a live
//  retargeter spends the very next frame dragging the robot back out of the
//  pose the command just established — the robot fights itself, twitching
//  toward home and being pulled back, with nothing on screen explaining why.
//
//  Home already did this (2026-07-30). Reset and the scene switch did not
//  (operator, 2026-07-31), which is the gap these tests close. Mirroring must
//  then resume only on an explicit Start Teleop — never implicitly.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class RelocationStopsTeleopTests: XCTestCase {

    private final class SpyLink: RobotLink {
        var status: ConnectionStatus = .connected
        var onState: ((RobotState) -> Void)?
        var onStatus: ((ConnectionStatus) -> Void)?
        var onServerInfo: ((String?) -> Void)?
        var onServerMeta: ((ServerMeta) -> Void)?
        var onControl: ((String, String?, String) -> Void)?
        var onMapList: (([MapInfo], String?) -> Void)?
        var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
        var sent: [RobotCommand] = []
        func connect() {}
        func disconnect() {}
        func send(_ c: RobotCommand) { sent.append(c) }
    }

    /// A controller with whole-body teleop RUNNING, as after Start Teleop.
    private func engaged() -> (RobotController, SpyLink) {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.torqueOnForTest(link)
        // Reset is refused unless it can actually happen (`canReset` =
        // simulation AND a live link) — and that is correct: a refused reset
        // must not stop teleop either. Make it real so the rule is under test
        // rather than the guard.
        link.onStatus?(.connected)
        XCTAssertTrue(c.canReset, "precondition: the reset is actually available")
        let gen = c.beginTrackingSession()
        c.trackingServicesLive(gen, handsAvailable: true)
        XCTAssertTrue(c.bodyTracking, "precondition: teleop is running")
        return (c, link)
    }

    // MARK: - The rule

    func testResetStopsTeleop() {
        let (c, _) = engaged()
        c.resetSim()
        XCTAssertFalse(c.bodyTracking,
                       "a scene reset must stop mirroring — the robot is being moved")
    }

    func testRandomResetStopsTeleop() {
        let (c, _) = engaged()
        c.resetSimRandom()
        XCTAssertFalse(c.bodyTracking)
    }

    func testHomeStopsTeleop() {
        let (c, _) = engaged()
        c.goRestPose()
        XCTAssertFalse(c.bodyTracking, "Home must win over a live retargeter")
    }

    func testSceneSwitchHookStopsTeleop() {
        // The scene UI holds only the studio, so the controller installs a hook.
        // Verify the hook is actually wired, not merely defined.
        let (c, _) = engaged()
        XCTAssertNotNil(c.simStudio.onWillSwitchScene,
                        "the controller must install the scene-switch hook")
        c.simStudio.onWillSwitchScene?()
        XCTAssertFalse(c.bodyTracking)
    }

    // MARK: - …and it must take an EXPLICIT restart to come back

    func testTeleopDoesNotResumeByItselfAfterReset() {
        let (c, _) = engaged()
        c.resetSim()
        XCTAssertFalse(c.bodyTracking)
        // Nothing about being connected, in the space, or ticking may re-engage.
        c.applyTeleopSpec()
        XCTAssertFalse(c.bodyTracking, "mirroring resumed without Start Teleop")
    }

    func testStartTeleopReEngagesAfterReset() {
        let (c, _) = engaged()
        c.resetSim()
        c.engageWholeBody()
        XCTAssertTrue(c.bodyTracking, "Start Teleop must bring it back")
        // …with a LIVE neck. `stopWholeBodyTeleop` freezes head mirroring, so a
        // re-engage that did not restore it would drive the arms behind a neck
        // that never moved again.
        XCTAssertTrue(c.headEnabled, "the neck must come back with the session")
    }

    func testResetIsHarmlessWhenTeleopWasNeverRunning() {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.torqueOnForTest(link)
        link.onStatus?(.connected)
        XCTAssertFalse(c.bodyTracking)
        c.resetSim()
        XCTAssertFalse(c.bodyTracking)
    }
}
