//
//  SavedCalibrationTests.swift
//  MabelTeleopTests
//
//  "Calibration can never be completed again — add an option to skip
//  calibration and just use my saved numbers" (operator, 2026-10-12).
//
//  The card only advances on the SERVER's `calibrationStatus.done`, and NOVA's
//  gateway does not implement `body_calibration` at all, so there the card sat
//  on "Extend Both Arms Forward" forever. Two things fix it, both pinned here
//  because the Simulator has no hand tracking and cannot run a capture:
//    1. the saved profile — which one "Use Saved Calibration" picks, that it
//       survives a relaunch, and what it sends;
//    2. the stall watchdog — when the card stops waiting and offers a way out.
//

import XCTest
@testable import MabelTeleop

@MainActor
final class SavedCalibrationTests: XCTestCase {

    // MARK: - Persistence

    private var suiteName = ""
    private var defaults: UserDefaults!

    override func setUp() {
        super.setUp()
        suiteName = "SavedCalibrationTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        super.tearDown()
    }

    private func metrics(_ reach: Double, human: Double = 0.62, robot: Double = 0.74) -> OperatorMetrics {
        var m = OperatorMetrics()
        m.reach = reach; m.humanReach = human; m.robotReach = robot
        return m
    }

    private func profile(_ name: String, _ reach: Double, ageDays: Double) -> CalibrationProfile {
        CalibrationProfile(name: name, metrics: metrics(reach),
                           createdAt: Date(timeIntervalSinceNow: -ageDays * 86_400))
    }

    /// A successful capture's numbers outlive the app: a fresh store (= the next
    /// launch) reads back the same reach, arm length and active operator.
    func test_savedProfile_survivesRelaunch() {
        let a = CalibrationStore(defaults: defaults)
        a.upsert(CalibrationProfile(name: "Operator", metrics: metrics(1.18, human: 0.63, robot: 0.74)))
        a.setActive("Operator")

        let b = CalibrationStore(defaults: defaults)
        XCTAssertEqual(b.profiles.count, 1)
        XCTAssertEqual(b.profiles.first?.name, "Operator")
        XCTAssertEqual(b.profiles.first?.reach ?? 0, 1.18, accuracy: 1e-9)
        XCTAssertEqual(b.profiles.first?.humanReach ?? 0, 0.63, accuracy: 1e-9)
        XCTAssertEqual(b.activeName, "Operator")
        XCTAssertEqual(b.preferred(forName: "")?.reach ?? 0, 1.18, accuracy: 1e-9)
    }

    /// Re-calibrating the same name replaces its numbers in place (one row per
    /// operator), case-insensitively.
    func test_recalibratingSameName_updatesInPlace() {
        let s = CalibrationStore(defaults: defaults)
        s.upsert(CalibrationProfile(name: "Operator", metrics: metrics(1.10)))
        s.upsert(CalibrationProfile(name: "operator", metrics: metrics(1.25)))
        XCTAssertEqual(s.profiles.count, 1)
        XCTAssertEqual(s.profiles.first?.reach ?? 0, 1.25, accuracy: 1e-9)
    }

    /// Which saved numbers: the typed name, else the active operator, else the
    /// newest — and nothing at all when nothing was ever saved.
    func test_preferredProfile_priority() {
        let operator = profile("Operator", 1.18, ageDays: 10)
        let ana = profile("Ana", 1.31, ageDays: 1)
        let old = profile("Guest", 1.02, ageDays: 30)
        let all = [old, operator, ana]

        XCTAssertEqual(CalibrationStore.preferred(in: all, name: " operator ", activeName: "Ana")?.name,
                       "Operator", "the name on the card wins")
        XCTAssertEqual(CalibrationStore.preferred(in: all, name: "", activeName: "Guest")?.name,
                       "Guest", "no name typed → the operator last applied")
        XCTAssertEqual(CalibrationStore.preferred(in: all, name: "Nobody", activeName: nil)?.name,
                       "Ana", "unknown name, no active → the newest measurement")
        XCTAssertNil(CalibrationStore.preferred(in: [], name: "Operator", activeName: "Operator"))
    }

    /// A corrupt / unmeasured row is never offered as "your saved numbers".
    func test_preferredProfile_ignoresNonsenseReach() {
        let bad = profile("Operator", 0.0, ageDays: 0)
        let good = profile("Ana", 1.2, ageDays: 5)
        XCTAssertEqual(CalibrationStore.preferred(in: [bad, good], name: "Operator", activeName: nil)?.name,
                       "Ana")
    }

    // MARK: - Stall watchdog (pure)

    func test_stall_serverSilent_after10s() {
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 3, serverHeard: false, done: false), .waiting)
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 9.9, serverHeard: false, done: false), .waiting)
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 10, serverHeard: false, done: false), .serverSilent)
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 300, serverHeard: false, done: false), .serverSilent)
    }

    /// A server that IS measuring gets far longer than one that is silent — the
    /// measured capture takes 0.8–2.6 s, so 10 s of "measuring" is not a stall.
    func test_stall_serverMeasuring_getsLonger() {
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 12, serverHeard: true, done: false), .waiting)
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 25, serverHeard: true, done: false), .notConverging)
        XCTAssertLessThan(CalibrationStall.silentAfter, CalibrationStall.slowAfter)
    }

    func test_stall_doneNeverStalls() {
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 999, serverHeard: false, done: true), .waiting)
        XCTAssertEqual(CalibrationStall.verdict(elapsed: 999, serverHeard: true, done: true), .waiting)
    }

    // MARK: - Controller wiring

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

    /// NOVA's case: a capture starts, the server never says a word. The card's
    /// watchdog must fire at 10 s — the old card waited forever.
    func test_controller_silentServer_isFlagged() {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.startBodyCalibration(name: "Operator")
        let t0 = c.calibrationStartedAt ?? Date()
        XCTAssertEqual(c.calibrationStall(now: t0.addingTimeInterval(5)), .waiting)
        XCTAssertEqual(c.calibrationStall(now: t0.addingTimeInterval(11)), .serverSilent)
        c.cancelBodyCalibration()
        XCTAssertEqual(c.calibrationStall(now: t0.addingTimeInterval(60)), .waiting,
                       "no capture in flight → nothing to offer")
    }

    /// The MABEL bridge DOES answer: a `calibrationStatus` arrives, so 11 s is
    /// still "measuring", not "silent".
    func test_controller_serverStatus_isHeard() {
        let link = SpyLink()
        let c = RobotController(link: link)
        c.startBodyCalibration(name: "Operator")
        XCTAssertFalse(c.calibrationServerHeard, "the optimistic local status is not the server")
        var s = RobotState()
        s.calibration = CalibrationProgress(active: true, pose: "arms_forward",
                                            index: 0, total: 1, done: false, reach: nil)
        link.onState?(s)
        XCTAssertTrue(c.calibrationServerHeard)
        let t0 = c.calibrationStartedAt ?? Date()
        XCTAssertEqual(c.calibrationStall(now: t0.addingTimeInterval(11)), .waiting)
        XCTAssertEqual(c.calibrationStall(now: t0.addingTimeInterval(26)), .notConverging)
        c.cancelBodyCalibration()
    }

    /// "Use Saved Calibration" mid-capture: the capture is abandoned (cancel on
    /// the wire), the saved profile is applied by NAME (the only apply the wire
    /// has), and it becomes the active operator.
    func test_controller_useSaved_cancelsCaptureAndApplies() {
        let link = SpyLink()
        let c = RobotController(link: link)
        let previousActive = c.calibrationStore.activeName
        defer { c.calibrationStore.setActive(previousActive) }   // don't leak into the app's defaults

        c.startBodyCalibration(name: "Operator")
        link.sent.removeAll()
        let p = CalibrationProfile(name: "Operator", metrics: metrics(1.18))
        c.useSavedCalibration(p)

        XCTAssertNil(c.calibration, "the capture card state is cleared")
        XCTAssertEqual(c.calibrationStall(now: Date().addingTimeInterval(60)), .waiting)
        XCTAssertEqual(c.appliedSavedProfile?.name, "Operator")
        XCTAssertEqual(c.activeProfileName, "Operator")
        let cancelled = link.sent.contains { if case .cancelBodyCalibration = $0 { return true }; return false }
        let applied = link.sent.compactMap { cmd -> String? in
            if case let .applyBodyCalibration(user) = cmd { return user }; return nil
        }
        XCTAssertTrue(cancelled, "the in-flight capture is cancelled on the server")
        XCTAssertEqual(applied, ["Operator"], "the saved profile is applied exactly once, by name")
    }
}
