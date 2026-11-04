//
//  LinkPriorityTests.swift
//  MabelTeleopTests
//
//  THE THREE SITUATIONS, pinned.
//
//    1. Both the network and Bluetooth are there   → take the network, every time.
//    2. Bluetooth only (the robot has no network)  → take Bluetooth, without being asked.
//    3. Network only, too far for Bluetooth        → take Wi-Fi, or Tailscale from away.
//
//  …plus the transitions between them, which is where a transport policy actually earns its
//  keep: sitting on Bluetooth with the robot plainly on the same Wi-Fi, and losing Wi-Fi
//  with the robot an arm's length away.
//

import XCTest
@testable import MabelTeleop

final class LinkPriorityTests: XCTestCase {

    private func route(_ kind: RouteKind, _ host: String = "10.0.0.5") -> ConnectionRoute {
        ConnectionRoute(kind: kind, host: host)
    }

    // MARK: The order itself

    func testTheOrderIsWiFiThenTailscaleThenBluetooth() {
        let ranked: [RouteKind] = [.lan, .tailscale, .bluetooth]
        let ranks = ranked.map(LinkPriority.rank)
        XCTAssertEqual(ranks, ranks.sorted(), "priority must be strictly best-first")
        XCTAssertTrue(LinkPriority.isUpgrade(from: .bluetooth, to: .tailscale))
        XCTAssertTrue(LinkPriority.isUpgrade(from: .tailscale, to: .lan))
        XCTAssertFalse(LinkPriority.isUpgrade(from: .lan, to: .tailscale), "never downgrade on purpose")
        XCTAssertFalse(LinkPriority.isUpgrade(from: .lan, to: .lan))
    }

    // MARK: 1 — both work

    func testBothAvailableTakesTheNetwork() {
        let d = LinkPriority.decide(current: nil, live: false,
                                    network: [route(.lan)], bluetoothInRange: true,
                                    hadNetwork: false)
        XCTAssertEqual(d, .connect(route(.lan)), "Wi-Fi wins outright when both are there")
    }

    func testOnBluetoothWithWiFiPresentUpgradesAutomatically() {
        let d = LinkPriority.decide(current: .bluetooth, live: true,
                                    network: [route(.lan)], bluetoothInRange: true,
                                    hadNetwork: false)
        XCTAssertEqual(d, .connect(route(.lan)),
                       "this is the reported case: connected over Bluetooth on the same Wi-Fi")
    }

    func testAlreadyOnWiFiIsLeftAlone() {
        let d = LinkPriority.decide(current: .lan, live: true,
                                    network: [route(.lan)], bluetoothInRange: true,
                                    hadNetwork: true)
        XCTAssertEqual(d, .stay, "re-connecting a working link to shave milliseconds drops frames")
    }

    // MARK: 2 — Bluetooth only

    func testBluetoothOnlyIsTakenWithoutAsking() {
        let d = LinkPriority.decide(current: nil, live: false,
                                    network: [], bluetoothInRange: true, hadNetwork: false)
        XCTAssertEqual(d, .connectBluetooth, "a first connection has nothing to lose")
    }

    func testStaysOnBluetoothWhileNothingBetterExists() {
        let d = LinkPriority.decide(current: .bluetooth, live: true,
                                    network: [], bluetoothInRange: true, hadNetwork: false)
        XCTAssertEqual(d, .stay)
    }

    // MARK: 3 — network only

    func testTailscaleIsUsedWhenTheLanIsNotReachable() {
        let d = LinkPriority.decide(current: nil, live: false,
                                    network: [route(.tailscale, "100.64.0.3")],
                                    bluetoothInRange: false, hadNetwork: false)
        XCTAssertEqual(d, .connect(route(.tailscale, "100.64.0.3")),
                       "out of Bluetooth range, Tailscale is the whole point")
    }

    func testTheLanBeatsTailscaleWhenTheRaceReturnsBoth() {
        // ConnectionPlanner returns fastest-first; the policy must not reorder it.
        let d = LinkPriority.decide(current: nil, live: false,
                                    network: [route(.lan), route(.tailscale, "100.64.0.3")],
                                    bluetoothInRange: false, hadNetwork: false)
        XCTAssertEqual(d, .connect(route(.lan)))
    }

    func testNothingReachableSaysSo() {
        let d = LinkPriority.decide(current: nil, live: false,
                                    network: [], bluetoothInRange: false, hadNetwork: false)
        XCTAssertEqual(d, .nothingReachable)
    }

    // MARK: The transition that has to ask

    func testLosingTheNetworkWithBluetoothInRangeASKS() {
        let d = LinkPriority.decide(current: .lan, live: false,
                                    network: [], bluetoothInRange: true, hadNetwork: true)
        XCTAssertEqual(d, .offerBluetooth,
                       "dropping a tier removes video — it must never happen silently")
    }

    func testLosingTheNetworkWithNoBluetoothIsJustLost() {
        let d = LinkPriority.decide(current: .lan, live: false,
                                    network: [], bluetoothInRange: false, hadNetwork: true)
        XCTAssertEqual(d, .nothingReachable)
    }
}
