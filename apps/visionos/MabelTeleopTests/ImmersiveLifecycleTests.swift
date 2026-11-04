//
//  ImmersiveLifecycleTests.swift
//  MabelTeleopTests
//
//  The immersive-space tracking lifecycle, under every start/stop interleaving.
//
//  Flipping `.immersionStyle` RE-CREATES the immersive space, and SwiftUI gives
//  no ordering guarantee between the NEW space's `.task` (start) and the OLD
//  space's `.onDisappear` (stop). `startImmersiveTracking` awaits ARKit
//  authorization + provider startup, so the stale teardown can land mid-start:
//  before session tokens it cleared `immersiveActive`, stopped the just-started
//  providers and dropped the server to navigation — the engage tail then
//  silently no-oped and the dome opened with arms + hands dead ("full immersive
//  mode: the arm and hand tracking does not work", operator 2026-07-29). With
//  the robot never in retarget, walking didn't drive the base either, which
//  read as "stuck in local tracking" (the local mode itself was removed
//  2026-07-28 — the live stack is GLOBAL only).
//
//  The simulator has no hand tracking, so no UI-level harness can reach the
//  engage branch — these tests drive the token state machine directly:
//    beginTrackingSession()            the `.task` claiming a session
//    trackingServicesLive(_:handsAvailable:)  services up → mirroring decision
//    stopImmersiveTracking(session:)   the `.onDisappear` teardown
//

import XCTest
@testable import MabelTeleop

/// Records everything the app puts on the wire.
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

    /// The methods of every control spec sent, in order.
    var specMethods: [String] {
        sent.compactMap { (c: RobotCommand) -> String? in
            if case let .controlSpec(_, method, _, _, _, _) = c { return method }
            return nil
        }
    }
}

@MainActor
final class ImmersiveLifecycleTests: XCTestCase {

    private func makeController() -> (RobotController, SpyLink) {
        let link = SpyLink()
        let c = RobotController(link: link)
        return (c, link)
    }

    /// Open the dome the way the Full Immersive / console buttons do (defer
    /// from a cold start, carry a live session), then bring services up.
    /// Returns the session token the space instance would hold in `@State`.
    private func openSpace(_ c: RobotController, deferFromButton: Bool = true) -> Int {
        if deferFromButton { c.deferBodyEngage = !c.bodyTracking }
        let gen = c.beginTrackingSession()
        c.trackingServicesLive(gen, handsAvailable: true)
        return gen
    }

    // MARK: - Cold start: the dome must open PARKED, then engage on Start Teleop

    func testColdFullImmersiveOpensParked() {
        let (c, _) = makeController()
        c.fullImmersion = true
        let gen = openSpace(c)

        XCTAssertTrue(c.immersiveActive)
        XCTAssertFalse(c.bodyTracking, "opening the dome must not silently start mirroring")
        XCTAssertFalse(c.deferBodyEngage, "the deferral is one-shot")
        XCTAssertGreaterThan(gen, 0)
    }

    func testStartTeleopEngagesParkedDome() {
        let (c, link) = makeController()
        c.fullImmersion = true
        _ = openSpace(c)
        link.sent.removeAll()

        c.engageWholeBody()

        XCTAssertTrue(c.bodyTracking)
        XCTAssertEqual(link.specMethods.last, "retarget",
                       "Start Teleop must put the server in retarget")
        XCTAssertTrue(link.sent.contains { if case .recalibrate = $0 { return true }; return false },
                      "engage re-anchors the clutch at the operator's current pose")
    }

    // MARK: - Style flip, order A: old space tears down BEFORE the new one starts

    func testStyleFlipStopThenStart_keepsMirroring() {
        let (c, link) = makeController()
        let gen1 = openSpace(c, deferFromButton: false)   // classic auto-engage entry
        XCTAssertTrue(c.bodyTracking)

        // Window button: defer per the live session, dial full, space re-creates.
        c.deferBodyEngage = !c.bodyTracking
        c.fullImmersion = true
        c.stopImmersiveTracking(session: gen1)            // old space .onDisappear
        XCTAssertFalse(c.bodyTracking)                    // torn down mid-flip
        _ = openSpace(c, deferFromButton: false)          // new space .task

        XCTAssertTrue(c.immersiveActive)
        XCTAssertTrue(c.bodyTracking, "a live session must survive the style flip")
        XCTAssertEqual(link.specMethods.last, "retarget")
    }

    // MARK: - Style flip, order B: the NEW space starts FIRST — the replaced
    // space's teardown must be inert (the 2026-07-29 arms-dead regression)

    func testStyleFlipStartThenStop_replacedTeardownIsInert() {
        let (c, link) = makeController()
        let gen1 = openSpace(c, deferFromButton: false)
        XCTAssertTrue(c.bodyTracking)
        link.sent.removeAll()

        c.deferBodyEngage = !c.bodyTracking               // window button
        c.fullImmersion = true
        let gen2 = c.beginTrackingSession()               // new space .task (first!)
        c.stopImmersiveTracking(session: gen1)            // old space .onDisappear (late)

        XCTAssertTrue(c.immersiveActive,
                      "the replaced space's teardown must not clear the live session")
        XCTAssertTrue(c.bodyTracking)

        c.trackingServicesLive(gen2, handsAvailable: true)

        XCTAssertTrue(c.bodyTracking, "mirroring continues across the flip")
        XCTAssertEqual(link.specMethods.last, "retarget")
        XCTAssertFalse(link.specMethods.contains("navigation"),
                       "the stale teardown must not drop the server to navigation mid-session")
    }

    func testStartTeleopStillWorksAfterOrderBFlip() {
        // The report's exact shape: enter the dome, then Start Teleop does nothing.
        // After an order-B flip of a PARKED dome, engage must still work — i.e.
        // immersiveActive survived the stale teardown.
        let (c, link) = makeController()
        c.fullImmersion = true
        let gen1 = openSpace(c)                           // parked (deferred)
        XCTAssertFalse(c.bodyTracking)

        c.deferBodyEngage = !c.bodyTracking               // console toggle contract
        c.fullImmersion = false
        let gen2 = c.beginTrackingSession()               // order B again
        c.stopImmersiveTracking(session: gen1)
        c.trackingServicesLive(gen2, handsAvailable: true)
        XCTAssertFalse(c.bodyTracking, "a parked dome stays parked across the flip")

        c.engageWholeBody()
        XCTAssertTrue(c.bodyTracking, "Start Teleop must engage after any flip order")
        XCTAssertEqual(link.specMethods.last, "retarget")
    }

    // MARK: - Genuine exits

    func testStopMidStartDoesNotEngage() {
        // The space is dismissed while startImmersiveTracking is still awaiting
        // ARKit: the teardown invalidates the in-flight session, so the late
        // services-live tail must not flip the mirroring switch back on.
        let (c, _) = makeController()
        let gen = c.beginTrackingSession()
        c.stopImmersiveTracking(session: gen)             // user exits immediately
        c.trackingServicesLive(gen, handsAvailable: true) // in-flight start finishing late

        XCTAssertFalse(c.immersiveActive)
        XCTAssertFalse(c.bodyTracking)
    }

    func testRealExitTearsDown() {
        let (c, link) = makeController()
        let gen = openSpace(c, deferFromButton: false)
        XCTAssertTrue(c.bodyTracking)

        c.stopImmersiveTracking(session: gen)

        XCTAssertFalse(c.immersiveActive)
        XCTAssertFalse(c.bodyTracking)
        XCTAssertEqual(link.specMethods.last, "navigation",
                       "closing the space falls back to plain navigation")
    }

    // MARK: - The console Window↔Immersive toggle must not auto-engage a parked dome

    func testConsoleFlipParkedDomeStaysParked() {
        let (c, _) = makeController()
        c.fullImmersion = true
        let gen1 = openSpace(c)                           // parked (deferred entry)
        XCTAssertFalse(c.bodyTracking)

        // Console toggle: defer per the (not-)live session, then flip (order A).
        c.deferBodyEngage = !c.bodyTracking
        c.fullImmersion = false
        c.stopImmersiveTracking(session: gen1)
        _ = openSpace(c, deferFromButton: false)          // restart consumes the deferral

        XCTAssertTrue(c.immersiveActive)
        XCTAssertFalse(c.bodyTracking,
                       "a Window↔Immersive flip of a parked dome must not start mirroring by itself")
    }
}
