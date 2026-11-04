//
//  OfflineStateTests.swift
//  MabelTeleopTests (visionOS)
//
//  NO HOST → NO DATA. Mirror of the iOS OfflineStateTests.
//
//  The headset used to freeze on the last frame it received: disconnect the robot
//  and the pose, the LiDAR sweep and the pointcloud stayed on screen, indistinct
//  from live data. `onState` holds the last value for any joint missing from a frame
//  (right, THROUGH a dropout); nothing undid that when the link actually died —
//  `onStatus` cleared only `lastTelemetry`, leaving the odom pose, the LiDAR sweep,
//  the SLAM mesh and the pointcloud intact. And only the EXPLICIT disconnect went
//  through even that; a real drop (socket close / heartbeat watchdog → `.connecting`,
//  retries forever) cleared nothing — the case an operator actually hits.
//

import XCTest
import SceneKit
@testable import MabelTeleop

/// A link we drive by hand: emit telemetry, then kill it the way a real socket dies.
final class FakeLink: RobotLink {
    var status: ConnectionStatus = .disconnected { didSet { onStatus?(status) } }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((String, String?, String) -> Void)?
    var onMapList: (([MapInfo], String?) -> Void)?
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?

    func connect() { status = .connected }
    func disconnect() { status = .disconnected }
    func send(_ command: RobotCommand) {}

    /// A robot that is alive and telling us where it is. NB `right_arm_1` — only
    /// joints in `RobotController.telemetryJoints` are taken from the wire.
    func emitLiveTelemetry() {
        var s = RobotState()
        s.baseX = 3.5; s.baseY = -2.25
        s.heading = 1.1
        s.baseQuat = [0.8, 0, 0.6, 0]
        s.joints = ["right_arm_1": 0.77, "lift_lower": 0.2]
        s.lidar = RobotState.LidarScan(angleMin: -.pi, angleInc: 0.01,
                                       rangeMax: 12, ranges: [1.0, 2.0, 3.0])
        onState?(s)
    }

    /// How a link ACTUALLY dies in the field: socket error / heartbeat watchdog →
    /// `dropped()` → `.connecting`, then it retries the same endpoint forever.
    func dropUnexpectedly() { status = .connecting }
}

@MainActor
final class OfflineStateTests: XCTestCase {

    private func connectedController() -> (RobotController, FakeLink) {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()
        link.emitLiveTelemetry()
        return (c, link)
    }

    func testTelemetryLands() {
        let (c, _) = connectedController()
        XCTAssertEqual(c.connection, .connected)
        XCTAssertEqual(c.state.baseX, 3.5)
        XCTAssertNotNil(c.lidarScan)
        XCTAssertEqual(c.livePose.joints["right_arm_1"], 0.77)
    }

    func testUnexpectedDropClearsTelemetry() {
        let (c, link) = connectedController()
        link.dropUnexpectedly()

        XCTAssertFalse(c.connection.isLive, "a dropped link must not read as live")
        XCTAssertEqual(c.state.baseX, 0, "stale odom X survived the drop")
        XCTAssertEqual(c.state.baseY, 0, "stale odom Y survived the drop")
        XCTAssertEqual(c.livePose.heading, 0, "stale heading survived the drop")
        XCTAssertEqual(c.livePose.baseX, 0, "the rig is still parked at the last odom pose")
        XCTAssertNil(c.lidarScan, "the LiDAR sweep is still on screen")
        XCTAssertNotEqual(c.livePose.joints["right_arm_1"], 0.77,
                          "hold-last telemetry survived the drop — the pose is frozen, not default")
    }

    func testExplicitDisconnectClearsTelemetry() {
        let (c, _) = connectedController()
        c.disconnect()
        XCTAssertEqual(c.connection, .disconnected)
        XCTAssertEqual(c.state.baseX, 0)
        XCTAssertNil(c.lidarScan)
        XCTAssertEqual(c.livePose.baseX, 0)
    }

    func testReconnectDoesNotResurrectStaleData() {
        let (c, link) = connectedController()
        link.dropUnexpectedly()
        XCTAssertEqual(c.state.baseX, 0)

        link.connect()
        XCTAssertEqual(c.connection, .connected)
        XCTAssertEqual(c.state.baseX, 0, "nothing should reappear until the robot says so")

        link.emitLiveTelemetry()
        XCTAssertEqual(c.state.baseX, 3.5)
        XCTAssertNotNil(c.lidarScan)
    }

    func testMapGoesOffline() {
        let store = LiveMapStore()
        var mesh: MapMesh?, cloud: MapMesh?, grid: MapGrid?, path: MapPath?
        store.onMesh = { mesh = $0 }
        store.onCloud = { cloud = $0 }
        store.onMap = { grid = $0 }
        store.onPath = { path = $0 }

        store.goOffline()

        // An EMPTY payload is this app's "clear the overlay" signal all the way down
        // to the SceneKit/RealityKit containers and the minimap raster.
        XCTAssertEqual(cloud?.count, 0, "the live sensor pointcloud is still rendered")
        XCTAssertEqual(mesh?.count, 0, "the SLAM mesh is still rendered")
        XCTAssertEqual(grid?.cells.isEmpty, true, "the occupancy grid is still rendered")
        XCTAssertEqual(path?.points.isEmpty, true, "the planned route is still drawn")
        XCTAssertEqual(store.vertexCount, 0)
        XCTAssertFalse(store.hasPose)
    }

    func testDropClearsTheMap() {
        let link = FakeLink()
        let c = RobotController(link: link)
        var cloud: MapMesh?
        c.liveMap.onCloud = { cloud = $0 }
        link.connect()
        link.emitLiveTelemetry()

        link.dropUnexpectedly()
        XCTAssertEqual(cloud?.count, 0,
                       "the pointcloud outlived the robot — exactly the reported bug")
    }

    func testDemoStaysLive() {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()
        link.emitLiveTelemetry()

        link.status = .demo             // on-device sim: live, just not a host
        XCTAssertTrue(c.connection.isLive)
        XCTAssertEqual(c.state.baseX, 3.5, "demo is live data; it must not be wiped")
    }
}

@MainActor
final class RobotSwitchingTests: XCTestCase {

    func testBothRobotsExist() {
        let store = RobotStore()
        XCTAssertNotNil(store.robots.first { $0.kind == .real }, "real robot missing from seeds")
        XCTAssertNotNil(store.robots.first { $0.kind == .sim }, "simulator missing from seeds")
        XCTAssertEqual(store.robots.first { $0.kind == .real }?.routes.contains { NetworkConfig.classify($0) == .tailscale }, true,
                       "the real robot must be reachable over Tailscale")
    }

    func testSwitchingClearsTheOldRobotsState() {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()
        link.emitLiveTelemetry()
        XCTAssertEqual(c.state.baseX, 3.5)

        guard let real = c.robotStore.robots.first(where: { $0.kind == .real }) else {
            return XCTFail("no real robot seeded")
        }
        c.connect(to: real)

        XCTAssertEqual(c.state.baseX, 0, "the previous robot's odom carried into the new one")
        XCTAssertNil(c.lidarScan, "the previous robot's LiDAR sweep carried over")
        XCTAssertEqual(c.livePose.baseX, 0, "the rig is still parked at the old robot's pose")
    }

    func testActiveRobotFollowsTheSwitch() {
        let c = RobotController(link: FakeLink())
        guard let sim = c.robotStore.robots.first(where: { $0.kind == .sim }),
              let real = c.robotStore.robots.first(where: { $0.kind == .real }) else {
            return XCTFail("seeds missing")
        }
        c.connect(to: sim)
        XCTAssertEqual(c.robotStore.active?.id, sim.id)
        c.connect(to: real)
        XCTAssertEqual(c.robotStore.active?.id, real.id, "switching to the real robot didn't take")
        c.disconnect()
        XCTAssertNil(c.robotStore.active, "disconnect must leave no active robot")
    }
}
