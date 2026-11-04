//
//  KnownRobotsTests.swift
//  MABEL Mobile AppTests
//
//  Settings used to list only what was on the air RIGHT NOW. Walk in with the app already
//  open and there was nothing to tap until a scan happened to catch an advertisement — even
//  for a robot this phone had driven ten minutes earlier. CoreBluetooth can reconnect from a
//  saved identifier with no scan at all, so the missing piece was only ever the memory.
//
//  These tests pin that memory: what gets remembered (and when), that a rename is the
//  operator's label and never hides the robot's own name, and that everything survives a
//  relaunch. They run against a scratch UserDefaults suite and clean up after themselves.
//

import XCTest
@testable import MABEL_Mobile_App

@MainActor
final class KnownRobotsTests: XCTestCase {

    private var suite: String!
    private var defaults: UserDefaults!

    override func setUp() {
        super.setUp()
        suite = "KnownRobotsTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suite)
        super.tearDown()
    }

    func testRemembersARobotAndSurvivesRelaunch() {
        let id = UUID()
        KnownRobots(defaults: defaults).remember(id: id, advertisedName: "NOVA")

        let reopened = KnownRobots(defaults: defaults)          // a fresh launch
        XCTAssertEqual(reopened.robots.count, 1)
        XCTAssertEqual(reopened.robot(id: id)?.advertisedName, "NOVA")
        XCTAssertEqual(reopened.robot(id: id)?.displayName, "NOVA")
        XCTAssertFalse(reopened.robot(id: id)!.isRenamed)
    }

    func testReconnectingUpdatesRatherThanDuplicates() {
        let store = KnownRobots(defaults: defaults)
        let id = UUID()
        store.remember(id: id, advertisedName: "NOVA")
        store.remember(id: id, advertisedName: "NOVA Sim")      // the robot renamed itself

        XCTAssertEqual(store.robots.count, 1, "one entry per peripheral, not one per connection")
        XCTAssertEqual(store.robot(id: id)?.advertisedName, "NOVA Sim")
    }

    func testMostRecentlyConnectedIsListedFirst() {
        let store = KnownRobots(defaults: defaults)
        let older = UUID(), newer = UUID()
        store.remember(id: older, advertisedName: "Lab")
        store.remember(id: newer, advertisedName: "Bench")

        XCTAssertEqual(store.robots.first?.id, newer)
        store.remember(id: older, advertisedName: "Lab")        // reconnect the first one
        XCTAssertEqual(store.robots.first?.id, older, "reconnecting should float it back to the top")
    }

    func testRenameIsALocalLabelAndKeepsTheAdvertisedName() {
        let store = KnownRobots(defaults: defaults)
        let id = UUID()
        store.remember(id: id, advertisedName: "NOVA")
        store.rename(id: id, to: "Lab robot")

        let r = store.robot(id: id)!
        XCTAssertEqual(r.displayName, "Lab robot")
        XCTAssertEqual(r.advertisedName, "NOVA", "a local label must not overwrite what the robot calls itself")
        XCTAssertTrue(r.isRenamed, "the UI shows both names when they differ")

        // …and it is still the operator's label after a relaunch
        XCTAssertEqual(KnownRobots(defaults: defaults).robot(id: id)?.displayName, "Lab robot")
    }

    func testBlankRenameFallsBackToTheAdvertisedName() {
        let store = KnownRobots(defaults: defaults)
        let id = UUID()
        store.remember(id: id, advertisedName: "NOVA")
        store.rename(id: id, to: "Lab robot")
        store.rename(id: id, to: "   ")                          // cleared

        XCTAssertEqual(store.robot(id: id)?.displayName, "NOVA")
        XCTAssertNil(store.robot(id: id)?.nickname, "whitespace is not a name")
    }

    func testForgetRemovesItEverywhere() {
        let store = KnownRobots(defaults: defaults)
        let id = UUID()
        store.remember(id: id, advertisedName: "NOVA")
        store.forget(id: id)

        XCTAssertTrue(store.isEmpty)
        XCTAssertTrue(KnownRobots(defaults: defaults).isEmpty, "forget must persist too")
    }
}
