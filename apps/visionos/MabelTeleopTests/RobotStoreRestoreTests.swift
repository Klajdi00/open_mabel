//
//  RobotStoreRestoreTests.swift
//  MabelTeleopTests
//
//  "Forget host" used to be a ONE-WAY DOOR for the built-in robots. `forget(id:)`
//  records the id in a `forgotten` set that `seed()` skips forever, so the sim and
//  thor never came back — and re-adding the host by hand (Add Robot) builds a
//  DEGRADED entry: one route. The robot then works on the LAN and is unreachable
//  over Tailscale, which is exactly the reported
//  "the default simulation robot is gone from the host options".
//
//  These tests pin the failure and the cure (ported 1:1 from the iOS app). They
//  touch UserDefaults, so each one runs against scratch keys and cleans up.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class RobotStoreRestoreTests: XCTestCase {

    // MUST track RobotStore's persist key, or setUp/tearDown scrub a key nothing
    // uses and the tests leak fixture robots into the real app's defaults.
    private let storeKey = "mabel.robots.v6"
    private let forgottenKey = "mabel.robots.forgotten.v1"

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removeObject(forKey: storeKey)
        UserDefaults.standard.removeObject(forKey: forgottenKey)
    }
    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: storeKey)
        UserDefaults.standard.removeObject(forKey: forgottenKey)
        super.tearDown()
    }

    /// The seeds must actually carry every way in — LAN and Tailscale.
    func testSeedsCarryLanAndTailscale() {
        let store = RobotStore()

        guard let real = store.robot(id: "thor-swerve") else {
            return XCTFail("thor missing from the seeds")
        }
        XCTAssertTrue(real.routes.contains("192.168.1.42"), "thor LAN route missing")
        XCTAssertTrue(real.routes.contains("100.64.0.2"), "thor Tailscale route missing")

        guard let sim = store.robot(id: "mabel-sim") else {
            return XCTFail("sim missing from the seeds")
        }
        // The Mac's Tailscale IP (NOT thor's) is the sim's off-LAN route.
        XCTAssertTrue(sim.routes.contains("100.64.0.5"))
        XCTAssertFalse(sim.routes.isEmpty)
    }

    /// The bug: forgetting a built-in kills it permanently, and the hand re-add
    /// that follows loses Tailscale.
    func testForgettingBuiltInIsStickyAndReAddIsDegraded() {
        let store = RobotStore()
        store.forget(id: "mabel-sim")
        XCTAssertNil(store.robot(id: "mabel-sim"))

        // A fresh store (app relaunch) must NOT bring it back — that's the trap.
        let relaunched = RobotStore()
        XCTAssertNil(relaunched.robot(id: "mabel-sim"),
                     "forgotten built-ins stay forgotten across launches")
        XCTAssertTrue(relaunched.hasForgottenBuiltIns)

        // Re-adding a sim IP by hand gives a LAN-only stub: no Tailscale.
        let readded = relaunched.add(host: "192.168.1.188", name: "sim", ssh: nil)
        XCTAssertEqual(readded.routes, ["192.168.1.188"])
        XCTAssertFalse(readded.routes.contains("100.64.0.5"))
    }

    /// The cure: restore brings the built-ins back WHOLE and drops the crippled
    /// hand-added duplicate, so the robot isn't listed twice.
    func testRestoreDefaultsBringsBackFullRoutesAndDropsTheStub() {
        let store = RobotStore()
        store.forget(id: "thor-swerve")
        store.forget(id: "mabel-sim")
        store.add(host: "192.168.1.42", name: "thor (typed)", ssh: nil)  // the LAN-only stub
        XCTAssertTrue(store.hasForgottenBuiltIns)

        store.restoreDefaults()

        XCTAssertFalse(store.hasForgottenBuiltIns)
        guard let real = store.robot(id: "thor-swerve") else {
            return XCTFail("restore did not bring thor back")
        }
        XCTAssertTrue(real.routes.contains("100.64.0.2"))
        XCTAssertTrue(real.routes.contains("192.168.1.42"))
        XCTAssertNotNil(store.robot(id: "mabel-sim"))

        // The hand-typed stub is subsumed by thor's routes → gone, not duplicated.
        XCTAssertNil(store.robot(id: "192.168.1.42"),
                     "the crippled duplicate must be dropped")
        XCTAssertEqual(store.robots.filter { $0.routes.contains("192.168.1.42") }.count, 1)

        // And it survives a relaunch (the forgotten set was cleared + persisted).
        let relaunched = RobotStore()
        XCTAssertNotNil(relaunched.robot(id: "mabel-sim"))
        XCTAssertTrue(relaunched.robot(id: "mabel-sim")?.routes.contains("100.64.0.5") ?? false)
    }

    /// Restore must not touch a genuinely custom robot (a second sim elsewhere).
    func testRestoreKeepsCustomRobots() {
        let store = RobotStore()
        store.add(host: "192.168.7.42", name: "Alienware sim", ssh: nil)
        store.forget(id: "mabel-sim")

        store.restoreDefaults()

        XCTAssertNotNil(store.robot(id: "192.168.7.42"), "custom robots must survive")
        XCTAssertNotNil(store.robot(id: "mabel-sim"))
    }
}
