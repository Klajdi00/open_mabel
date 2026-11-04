//
//  CockpitStickSpaceTests.swift
//  MabelTeleopTests
//
//  (2026-10-16: the floating `Joystick3D` sticks — and the cockpit's stick
//  host that opened this space for them — are GONE; driving is the 2-D
//  `DrivePadDock` on the window's bottom ornament. Nothing sets
//  `stickSpaceOnly` any more, but `needsImmersiveOpen()` is still the single
//  opener gate, so its contract stays pinned here.)
//
//  HISTORY: THE COCKPIT'S DRIVE STICKS FLOATED IN THE ROOM (operator,
//  2026-07-31: "I need the joystick to be floating around the user").
//
//  1st person opens the immersive space as a pure STICK HOST — `stickSpaceOnly`,
//  which claims no ARKit session, so `immersiveActive` stays false and every
//  teleop guard that keys on it behaves exactly as it does with no space open.
//  That creates ONE new state the app never had before: the space is open while
//  `immersiveActive` is false. Every opener used to test `!immersiveActive` and
//  would therefore try to open an already-open space — `openImmersiveSpace`
//  fails, and whole-body teleop / the dome / a calibration capture would have
//  come up with no session at all.
//
//  `needsImmersiveOpen()` is the one place that decides, so it is the one place
//  worth pinning. The Simulator has no hand tracking and cannot be gazed at or
//  pinched, so this contract has no UI-level harness — only these tests and the
//  DEBUG `[MABEL-STICKS]` log.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class CockpitStickSpaceTests: XCTestCase {

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

    private func controller() -> RobotController { RobotController(link: SpyLink()) }

    /// Nothing is open: the caller must open the space itself, and nothing is
    /// silently claimed on its behalf.
    func test_noSpace_callerMustOpen() {
        let c = controller()
        XCTAssertFalse(c.immersiveActive)
        XCTAssertFalse(c.stickSpaceOnly)
        XCTAssertTrue(c.needsImmersiveOpen())
        XCTAssertFalse(c.stickSpaceOnly)
    }

    /// The cockpit's stick host is up: PROMOTE it in place. Re-opening would
    /// fail (one immersive space at a time), and dropping `stickSpaceOnly` is
    /// exactly what tells the live space to claim its tracking session.
    func test_stickHost_isPromotedInPlace_notReopened() {
        let c = controller()
        c.stickSpaceOnly = true
        XCTAssertFalse(c.needsImmersiveOpen(), "must not re-open a space that is already up")
        XCTAssertFalse(c.stickSpaceOnly, "promotion is the signal that starts the session")
    }

    /// A real session already owns the space: no open, and the cockpit's flag
    /// is not touched (it is already false, and clearing state that belongs to
    /// someone else is how the promotion signal would fire spuriously).
    func test_liveSession_noOpenAndNoPromotion() {
        let c = controller()
        _ = c.beginTrackingSession()
        XCTAssertTrue(c.immersiveActive)
        XCTAssertFalse(c.needsImmersiveOpen())
        XCTAssertFalse(c.stickSpaceOnly)
    }

    /// Promotion must be ONE-SHOT. A second caller (Full Immersive pressed
    /// while the whole-body flow is still starting its session) must be told to
    /// open nothing — the session start is already in flight, and a second
    /// `openImmersiveSpace` is the failure this whole path exists to avoid.
    func test_promotionIsOneShot() {
        let c = controller()
        c.stickSpaceOnly = true
        XCTAssertFalse(c.needsImmersiveOpen())
        // The space's onChange has not run yet, so `immersiveActive` is still
        // false — the naive `!immersiveActive` test would say "open it".
        XCTAssertFalse(c.immersiveActive)
        XCTAssertTrue(c.needsImmersiveOpen(),
                      "with nothing left to promote the caller does own the open")
    }
}
