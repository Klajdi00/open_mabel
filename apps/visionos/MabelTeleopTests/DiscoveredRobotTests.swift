//
//  DiscoveredRobotTests.swift
//  MabelTeleopTests
//
//  "It won't connect to NOVA even on the same Wi-Fi" (2026-10-12). NOVA advertises
//  the same `_mabel-teleop._tcp` Bonjour service as MABEL, and the store folded EVERY
//  host it did not already know into the built-in "Simulation" entry. So NOVA never
//  appeared in the robot list: it was a hidden extra route of the simulation, raced
//  against the Mac's sim (first TCP answer won), and launch auto-connect only ever
//  cycled thor → sim. These tests pin the cure: a discovered robot is ITS
//  OWN entry, keyed by its advertised name, and the controller goes to it when the
//  default rig is not answering.
//
//  NOVA itself folds into the BUILT-IN "nova-orin" entry (it carries the Tailscale
//  route too). Other robots still get their own entry.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class DiscoveredRobotTests: XCTestCase {

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

    private let nova = RobotStore.DiscoveredRobot(name: "NOVA Robot (mabel-orin)", host: "192.168.2.44")
    private let other = RobotStore.DiscoveredRobot(name: "Lab Bot (box)", host: "192.168.2.44")
    private let macSim = RobotStore.DiscoveredRobot(name: "MABEL Sim (dev-mac)", host: "192.168.2.39")

    func testOtherRobotBecomesItsOwnRealRobot() {
        let store = RobotStore()
        store.mergeDiscovered(named: [other])
        let r = store.robot(id: RobotStore.discoveredID(other.name))
        XCTAssertNotNil(r, "a discovered robot must be in the list, not a hidden route")
        XCTAssertEqual(r?.name, "Lab Bot")
        XCTAssertEqual(r?.kind, .real)
        XCTAssertEqual(r?.routes, ["192.168.2.44"])
        XCTAssertEqual(r?.online, true)
        XCTAssertFalse(store.robot(id: "mabel-sim")!.routes.contains(other.host),
                       "its host must NOT be folded into the simulation")
    }

    func testNovaFoldsIntoTheBuiltInEntry() {
        let store = RobotStore()
        store.mergeDiscovered(named: [nova])
        XCTAssertNil(store.robot(id: RobotStore.discoveredID(nova.name)), "no one-route duplicate")
        let r = store.robot(id: "nova-orin")!
        XCTAssertEqual(r.routes.first, nova.host, "the address seen on this Wi-Fi goes first")
        XCTAssertFalse(store.robot(id: "mabel-sim")!.routes.contains(nova.host))
    }

    func testNovaSimAdvertIsNotFoldedIntoTheRealRobot() {
        let store = RobotStore()
        let sim = RobotStore.DiscoveredRobot(name: "NOVA Sim (dev-mac)", host: "192.168.2.60")
        store.mergeDiscovered(named: [sim])
        XCTAssertFalse(store.robot(id: "nova-orin")!.routes.contains(sim.host), "a NOVA SIM must never become a route of the real robot")
    }

    func testOldDiscoveredNovaEntryIsMigrated() throws {
        var stale = NetworkConfig.seedRobots().first { $0.id == "nova-orin" }!
        stale.routes = ["192.168.1.43"]
        let dup = Robot(id: RobotStore.discoveredID(nova.name), name: "NOVA Robot", kind: .real,
                        routes: ["10.0.0.9"], saved: true)
        UserDefaults.standard.set(try JSONEncoder().encode([stale, dup]), forKey: storeKey)
        let store = RobotStore()
        XCTAssertNil(store.robot(id: dup.id))
        let r = store.robot(id: "nova-orin")!
        XCTAssertTrue(r.routes.contains("10.0.0.9"))
        XCTAssertTrue(r.routes.contains("100.64.0.3"), "seeded routes are refreshed from this build")
    }

    func testMabelSimAdvertStillFoldsIntoTheSimulation() {
        let store = RobotStore()
        store.mergeDiscovered(named: [macSim])
        XCTAssertTrue(store.robot(id: "mabel-sim")!.routes.contains(macSim.host))
        XCTAssertNil(store.robot(id: RobotStore.discoveredID(macSim.name)))
    }

    func testSameRobotOnANewNetworkKeepsOneEntryNewestAddressFirst() {
        let store = RobotStore()
        store.mergeDiscovered(named: [other])
        store.mergeDiscovered(named: [.init(name: other.name, host: "10.0.0.7")])
        let entries = store.robots.filter { $0.name == "Lab Bot" }
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.routes, ["10.0.0.7", "192.168.2.44"])
    }

    func testDiscoveredRobotSurvivesRelaunch() {
        RobotStore().mergeDiscovered(named: [other])
        XCTAssertNotNil(RobotStore().robot(id: RobotStore.discoveredID(other.name)))
    }

    func testForgottenDiscoveredRobotStaysForgotten() {
        let store = RobotStore()
        store.mergeDiscovered(named: [other])
        store.forget(id: RobotStore.discoveredID(other.name))
        store.mergeDiscovered(named: [other])
        XCTAssertNil(store.robot(id: RobotStore.discoveredID(other.name)))
    }

    func testRealRobotOnThisWifiRanksAboveTheSim() {
        let store = RobotStore()
        let found = [macSim, nova]
        store.mergeDiscovered(named: found)
        XCTAssertEqual(store.discoveredOnline(found).first?.id, "nova-orin")
    }

    func testDisplayNameStripsOnlyTheMachineSuffix() {
        XCTAssertEqual(RobotStore.displayName("NOVA Robot (Operator's MacBook Pro)"), "NOVA Robot")
        XCTAssertEqual(RobotStore.displayName("NOVA"), "NOVA")
        XCTAssertEqual(RobotStore.displayName("(x)"), "(x)")
    }
}
