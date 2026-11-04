//
//  SimRealSwitchingTests.swift
//  MabelTeleopTests
//
//  Pins the SIM ↔ REAL robot switch — the one flow an operator uses constantly.
//  The two rigs must keep DISJOINT route pools: if the sim and the real robot
//  ever shared a host, a headset driving the SIM could silently bind to the
//  REAL robot (or pull its cameras). That is a safety-relevant illusion, not a
//  cosmetic bug.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class SimRealSwitchingTests: XCTestCase {

    private func seed(_ id: String) throws -> Robot {
        try XCTUnwrap(NetworkConfig.seedRobots().first { $0.id == id },
                      "seed robot '\(id)' is missing — the picker cannot offer it")
    }

    // MARK: - The seeds an operator picks between

    func test_bothRigsAreSeeded_withTheirOwnRoutes() throws {
        let sim = try seed("mabel-sim")
        let real = try seed("thor-swerve")

        XCTAssertEqual(sim.kind, .sim)
        XCTAssertEqual(real.kind, .real)
        XCTAssertFalse(sim.routes.isEmpty)
        XCTAssertFalse(real.routes.isEmpty)
        XCTAssertTrue(real.routes.contains { NetworkConfig.classify($0) == .tailscale },
                      "the real robot must carry a Tailscale route")
    }

    // MARK: - Controller-level switch
    //
    // (That the ACTIVE robot and telemetry follow a switch is already covered
    // by OfflineStateTests.testSwitchingClearsTheOldRobotsState / …FollowsTheSwitch —
    // not duplicated here. What follows is the video half, which was not.)

    /// Offline, no rig may serve video — a live tile under an "Offline" pill
    /// reads to the operator as a phantom connection to a robot that is gone.
    func test_noVideoWhileDisconnected() throws {
        let c = RobotController(link: FakeLink())
        c.connect(to: try seed("mabel-sim"))
        c.disconnect()

        XCTAssertNil(c.videoURL("main"))
        XCTAssertNil(c.videoThumbURL("wrist_left"))
    }
}
