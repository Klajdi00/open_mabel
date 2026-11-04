//
//  TeleopPersistenceTests.swift
//  MabelTeleopTests
//
//  "Do not stop teleop till I explicitly press Stop" (operator, 2026-10-16),
//  and the body-part ablation switches on the wire.
//
//  Leaving full immersion RE-CREATES the immersive space (an `.immersionStyle`
//  flip) or closes it outright, and ARKit hand tracking only exists while a
//  space is open — so every exit used to end whole-body teleop. These pin:
//    • leaving the dome keeps mirroring, under BOTH teardown orders;
//    • a SYSTEM close (Digital Crown) freezes the robot and asks the window to
//      resume, and the resumed space re-engages;
//    • the app's own Stop/Exit does not resume;
//    • Stop freezes (mirroring off AND head frozen, spec → navigation = hold);
//    • the body switches reach the wire with the right keys, and freezing the
//      neck no longer freezes the head pose every other channel reads.
//  The Simulator has no hand tracking and cannot press the Crown, so the token
//  state machine is driven directly (same harness as ImmersiveLifecycleTests).
//

import XCTest
@testable import MabelTeleop

private final class WireSpy: RobotLink {
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

    var lastSpec: (tracking: [String: Bool], method: String, scope: String?)? {
        for c in sent.reversed() {
            if case let .controlSpec(tracking, method, _, _, _, scope) = c {
                return (tracking, method, scope)
            }
        }
        return nil
    }
}

@MainActor
final class TeleopPersistenceTests: XCTestCase {

    private func engagedInDome() -> (RobotController, WireSpy, Int) {
        let link = WireSpy()
        let c = RobotController(link: link)
        c.fullImmersion = true
        c.deferBodyEngage = true                    // Full Immersive opens parked…
        let gen = c.beginTrackingSession()
        c.trackingServicesLive(gen, handsAvailable: true)
        c.engageWholeBody()                         // …then Start Teleop
        XCTAssertTrue(c.bodyTracking)
        return (c, link, gen)
    }

    // MARK: - leaving the dome keeps mirroring

    func test_leaveDome_keepsTeleop_whenOldSpaceTearsDownFirst() {
        let (c, _, gen) = engagedInDome()
        XCTAssertTrue(c.leaveFullImmersionKeepingTeleop())
        XCTAssertFalse(c.fullImmersion)
        XCTAssertTrue(c.trackingOnlyMode, "the space stays open, hidden, for ARKit")
        c.stopImmersiveTracking(session: gen)       // old space's onDisappear FIRST
        let gen2 = c.beginTrackingSession()         // re-created mixed space
        c.trackingServicesLive(gen2, handsAvailable: true)
        XCTAssertTrue(c.bodyTracking, "leaving full immersion must not end teleop")
        XCTAssertFalse(c.resumeTrackingRequested, "a style flip is not a system exit")
    }

    func test_leaveDome_keepsTeleop_whenNewSpaceStartsFirst() {
        let (c, _, gen) = engagedInDome()
        XCTAssertTrue(c.leaveFullImmersionKeepingTeleop())
        let gen2 = c.beginTrackingSession()         // new space's .task FIRST
        c.trackingServicesLive(gen2, handsAvailable: true)
        c.stopImmersiveTracking(session: gen)       // stale teardown → inert
        XCTAssertTrue(c.bodyTracking)
        XCTAssertTrue(c.immersiveActive)
    }

    func test_leaveDome_whenNotMirroring_isAPlainExit() {
        let link = WireSpy()
        let c = RobotController(link: link)
        c.fullImmersion = true
        c.deferBodyEngage = true
        let gen = c.beginTrackingSession()
        c.trackingServicesLive(gen, handsAvailable: true)
        XCTAssertFalse(c.bodyTracking)
        XCTAssertFalse(c.leaveFullImmersionKeepingTeleop(), "nothing to keep → caller dismisses")
    }

    // MARK: - a SYSTEM close freezes, then resumes

    func test_systemClose_freezesAndRequestsResume_thenReEngages() {
        let (c, link, gen) = engagedInDome()
        c.stopImmersiveTracking(session: gen)       // Digital Crown: not intended
        XCTAssertTrue(c.resumeTrackingRequested, "the window must re-open the space")
        XCTAssertFalse(c.bodyTracking, "no hands without a space — the robot holds")
        XCTAssertEqual(link.lastSpec?.method, "navigation", "navigation = hold the pose")
        // The window re-opens the tracking space (ContentView observer).
        c.resumeTrackingRequested = false
        c.trackingOnlyMode = true
        let gen2 = c.beginTrackingSession()
        c.trackingServicesLive(gen2, handsAvailable: true)
        XCTAssertTrue(c.bodyTracking, "the resumed session re-engages without a Start press")
        XCTAssertEqual(link.lastSpec?.method, "retarget")
    }

    func test_intendedExit_doesNotResume() {
        let (c, _, gen) = engagedInDome()
        c.noteIntendedImmersiveExit()
        c.stopImmersiveTracking(session: gen)
        XCTAssertFalse(c.resumeTrackingRequested)
        let gen2 = c.beginTrackingSession()
        c.trackingServicesLive(gen2, handsAvailable: true)
        // A later open is a fresh start: nothing carried, so Full Immersive's
        // defer applies as before (here no defer set → engages normally; the
        // point is that the old session's intent did not leak).
        XCTAssertFalse(c.resumeTrackingRequested)
    }

    // MARK: - Stop = freeze

    func test_stop_freezesEverything() {
        let (c, link, _) = engagedInDome()
        c.stopWholeBodyTeleop()
        XCTAssertFalse(c.bodyTracking)
        XCTAssertFalse(c.headEnabled, "the neck must not keep following the head after Stop")
        XCTAssertEqual(link.lastSpec?.method, "navigation")
    }

    // MARK: - body switches on the wire

    func test_upperVsWholeBody_drivesScopeAndLiftTogether() {
        let (c, link, _) = engagedInDome()
        c.retargetScope = .global
        XCTAssertEqual(link.lastSpec?.scope, "global")
        XCTAssertEqual(link.lastSpec?.tracking["lift"], true)
        c.retargetScope = .local
        XCTAssertEqual(link.lastSpec?.scope, "local")
        XCTAssertEqual(link.lastSpec?.tracking["lift"], false, "UPPER BODY freezes the lift")
    }

    func test_neckAndTorsoSwitches_onTheWire() {
        let (c, link, _) = engagedInDome()
        c.trackTorso = false
        XCTAssertEqual(link.lastSpec?.tracking["torso"], false)
        c.trackNeck = false
        XCTAssertEqual(link.lastSpec?.tracking["neck"], false)
        XCTAssertTrue(c.headEnabled,
                      "freezing the NECK must keep the live head streaming — torso, lift, base and the arm frame all read it")
        c.trackNeck = true; c.trackTorso = true
        XCTAssertEqual(link.lastSpec?.tracking["neck"], true)
        XCTAssertEqual(link.lastSpec?.tracking["torso"], true)
    }

    func test_defaultIsUpperBody_withLiftFrozen() {
        let c = RobotController(link: WireSpy())
        XCTAssertEqual(c.retargetScope, .local, "a moving base is asked for, never inherited")
        XCTAssertFalse(c.trackLift)
        XCTAssertEqual(RetargetScope.local.title, "Upper body")
        XCTAssertEqual(RetargetScope.global.title, "Whole body")
    }
}
