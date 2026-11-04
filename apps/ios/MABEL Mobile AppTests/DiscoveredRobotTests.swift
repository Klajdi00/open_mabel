//
//  DiscoveredRobotTests.swift
//  MABEL Mobile AppTests
//
//  "It won't connect to NOVA even on the same Wi-Fi" (2026-10-12). NOVA advertises
//  the same `_mabel-teleop._tcp` Bonjour service as MABEL, and the store folded EVERY
//  host it did not already know into the built-in "Simulation" entry, so NOVA never
//  appeared in the robot list and auto-connect never tried it. Same cure as the
//  Vision Pro app (its DiscoveredRobotTests): a discovered robot is ITS OWN entry,
//  keyed by its advertised name, and auto-connect goes to it — never away from a
//  robot the operator picked.
//

import Testing
import Foundation
@testable import MABEL_Mobile_App

@MainActor
@Suite("Bonjour-discovered robots", .serialized)
struct DiscoveredRobotTests {

    private static let keys = ["mabel.robots.v6", "mabel.robots.forgotten.v1"]
    init() { Self.keys.forEach { UserDefaults.standard.removeObject(forKey: $0) } }

    private let nova = RobotStore.DiscoveredRobot(name: "NOVA Robot (mabel-orin)", host: "192.168.2.44")
    private let macSim = RobotStore.DiscoveredRobot(name: "MABEL Sim (dev-mac)", host: "192.168.2.39")

    @Test("NOVA becomes its own real robot, not a hidden route of the sim")
    func novaIsItsOwnRobot() {
        let store = RobotStore()
        store.mergeDiscovered(named: [nova])
        let r = store.robot(id: RobotStore.discoveredID(nova.name))
        #expect(r?.name == "NOVA Robot")
        #expect(r?.kind == .real)
        #expect(r?.routes == ["192.168.2.44"])
        #expect(r?.online == true)
        #expect(store.robot(id: "mabel-sim")?.routes.contains(nova.host) == false)
    }

    @Test("a MABEL Sim advert still folds into the simulation")
    func simAdvertFoldsIntoSim() {
        let store = RobotStore()
        store.mergeDiscovered(named: [macSim])
        #expect(store.robot(id: "mabel-sim")?.routes.contains(macSim.host) == true)
        #expect(store.robot(id: RobotStore.discoveredID(macSim.name)) == nil)
    }

    @Test("same robot on a new network: one entry, newest address first")
    func newNetworkOneEntry() {
        let store = RobotStore()
        store.mergeDiscovered(named: [nova])
        store.mergeDiscovered(named: [.init(name: nova.name, host: "10.0.0.7")])
        let entries = store.robots.filter { $0.name == "NOVA Robot" }
        #expect(entries.count == 1)
        #expect(entries.first?.routes == ["10.0.0.7", "192.168.2.44"])
    }

    @Test("a discovered robot survives relaunch; a forgotten one stays forgotten")
    func persistenceAndForget() {
        RobotStore().mergeDiscovered(named: [nova])
        let store = RobotStore()
        #expect(store.robot(id: RobotStore.discoveredID(nova.name)) != nil)
        store.forget(id: RobotStore.discoveredID(nova.name))
        store.mergeDiscovered(named: [nova])
        #expect(store.robot(id: RobotStore.discoveredID(nova.name)) == nil)
    }

    @Test("a real robot on this Wi-Fi ranks above a sim")
    func realRanksAboveSim() {
        let store = RobotStore()
        store.mergeDiscovered(named: [macSim, nova])
        #expect(store.discoveredOnline([macSim, nova]).first?.id == RobotStore.discoveredID(nova.name))
    }

    @Test("launch auto-connect goes to the robot found on this Wi-Fi")
    func autoConnectPrefersDiscovered() {
        let c = RobotController(link: FakeLink())
        c.connectAuto(discovered: [nova])
        #expect(c.robotStore.active?.id == RobotStore.discoveredID(nova.name))
    }

    @Test("discovery moves a silent auto-connect to NOVA")
    func discoveryRescuesSilentRace() {
        let c = RobotController(link: FakeLink())
        c.connectAuto()                         // nothing found yet → the seeded rig
        #expect(c.robotStore.active?.id != RobotStore.discoveredID(nova.name))
        c.noteDiscovered([nova])
        #expect(c.robotStore.active?.id == RobotStore.discoveredID(nova.name))
    }

    @Test("discovery never overrides the robot the operator picked")
    func operatorPickIsPinned() {
        let c = RobotController(link: FakeLink())
        guard let sim = c.robotStore.robots.first(where: { $0.kind == .sim }) else {
            Issue.record("no sim seeded"); return
        }
        c.connect(to: sim)                      // the UI's call: an explicit pick
        c.noteDiscovered([nova])
        #expect(c.robotStore.active?.id == sim.id)
    }
}
