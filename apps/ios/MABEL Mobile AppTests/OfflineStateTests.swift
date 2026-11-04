//
//  OfflineStateTests.swift
//  MABEL Mobile AppTests
//
//  NO HOST → NO DATA.
//
//  The app used to freeze on the last frame it ever received: disconnect the robot
//  and the pose, the LiDAR sweep and the pointcloud all stayed on screen, looking
//  exactly like live data. The cause was structural — `onState` holds the last value
//  for any joint missing from a frame (right, through a dropout) and NOTHING undid
//  that when the link actually died. Its own comment claimed a `resetTelemetry()`
//  cleared it on disconnect; that function did not exist.
//
//  Worse, only the EXPLICIT `disconnect()` cleaned anything up. A real drop — robot
//  powered off, Wi-Fi lost, the 3.5 s heartbeat watchdog firing — goes through
//  `WebSocketLink.dropped()`, which reports `.connecting` and retries the same
//  endpoint forever. That path cleared nothing at all, which is the case an operator
//  actually hits.
//
//  These tests drive a fake link so both paths are exercised deterministically.
//

import Testing
import Foundation
import SceneKit
@testable import MABEL_Mobile_App

/// A link we drive by hand: emit telemetry, then kill it the way a real socket dies.
final class FakeLink: RobotLink {
    var status: ConnectionStatus = .disconnected { didSet { onStatus?(status) } }
    var onState: ((RobotState) -> Void)?
    var onStatus: ((ConnectionStatus) -> Void)?
    var onServerInfo: ((String?) -> Void)?
    var onServerMeta: ((ServerMeta) -> Void)?
    var onControl: ((String, String?, String) -> Void)?
    var onMapList: (([MapInfo], String?) -> Void)?
    /// Added 2026-08-23: the protocol grew this callback and the double was
    /// never updated, so the whole iOS test target had stopped compiling.
    var onDemoClips: (([RobotState.DemoClipInfo]) -> Void)?
    private(set) var sent: [RobotCommand] = []

    func connect() { status = .connected }
    func disconnect() { status = .disconnected }
    /// Observe commands as they are sent, for tests that need WHEN as well as WHAT
    /// (the Home glide's speed is a rate, and a list of values cannot show a rate).
    var onSend: ((RobotCommand) -> Void)?
    func send(_ command: RobotCommand) { sent.append(command); onSend?(command) }

    /// A robot that is alive and telling us where it is.
    /// NB `right_arm_1` (not a prettier name) — only joints in
    /// `RobotController.telemetryJoints` are taken from the wire.
    func emitLiveTelemetry() {
        var s = RobotState()
        s.baseX = 3.5; s.baseY = -2.25            // driven well away from the origin
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
struct OfflineStateTests {

    private func connectedController() -> (RobotController, FakeLink) {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()                       // → .connected
        link.emitLiveTelemetry()
        return (c, link)
    }

    // MARK: - The bug: an UNEXPECTED drop must clear everything

    @Test("live telemetry actually lands (guards the tests below)")
    func telemetryLands() {
        let (c, _) = connectedController()
        #expect(c.connection == .connected)
        #expect(c.state.baseX == 3.5)
        #expect(c.lidarScan != nil)
        #expect(c.livePose.joints["right_arm_1"] == 0.77)
    }

    @Test("an unexpected drop clears the pose, the LiDAR sweep and the odom")
    func unexpectedDropClearsTelemetry() {
        let (c, link) = connectedController()
        link.dropUnexpectedly()              // socket died; the link is now retrying

        #expect(!c.connection.isLive, "a dropped link must not read as live")
        // The rig must fall back to its default, not hang at the last streamed pose.
        #expect(c.state.baseX == 0, "stale odom X survived the drop")
        #expect(c.state.baseY == 0, "stale odom Y survived the drop")
        #expect(c.livePose.heading == 0, "stale heading survived the drop")
        #expect(c.livePose.baseX == 0, "the rig is still parked at the last odom pose")
        #expect(c.lidarScan == nil, "the LiDAR sweep is still on screen")
        #expect(c.livePose.joints["right_arm_1"] != 0.77,
                "hold-last telemetry survived the drop — the pose is frozen, not default")
    }

    @Test("an explicit disconnect clears the same state")
    func explicitDisconnectClearsTelemetry() {
        let (c, _) = connectedController()
        c.disconnect()

        #expect(c.connection == .disconnected)
        #expect(c.state.baseX == 0)
        #expect(c.lidarScan == nil)
        #expect(c.livePose.baseX == 0)
    }

    @Test("reconnecting repaints from the new robot, not the old one's cache")
    func reconnectDoesNotResurrectStaleData() {
        let (c, link) = connectedController()
        link.dropUnexpectedly()
        #expect(c.state.baseX == 0)

        link.connect()                       // same endpoint comes back
        #expect(c.connection == .connected)
        #expect(c.state.baseX == 0, "nothing should reappear until the robot says so")

        link.emitLiveTelemetry()             // fresh data → repaint
        #expect(c.state.baseX == 3.5)
        #expect(c.lidarScan != nil)
    }

    // MARK: - The map / pointcloud must go with it

    @Test("goOffline() clears every rendered map layer, including the pointcloud")
    func mapGoesOffline() {
        let store = LiveMapStore()
        var mesh: MapMesh?
        var cloud: MapMesh?
        var grid: MapGrid?
        var path: MapPath?
        store.onMesh = { mesh = $0 }
        store.onCloud = { cloud = $0 }
        store.onMap = { grid = $0 }
        store.onPath = { path = $0 }

        store.goOffline()

        // An EMPTY payload is this app's "clear the overlay" signal all the way down
        // to the SceneKit containers (LiveMapRig.apply) and the 2-D minimap raster.
        #expect(cloud?.count == 0, "the live sensor pointcloud is still rendered")
        #expect(mesh?.count == 0, "the SLAM mesh is still rendered")
        #expect(grid?.cells.isEmpty == true, "the occupancy grid is still rendered")
        #expect(path?.points.isEmpty == true, "the planned route is still drawn")
        #expect(store.vertexCount == 0)
        #expect(store.hasPose == false)
        #expect(store.lastDataAt == nil, "the SLAM-live chip still reads streaming")
    }

    @Test("losing the robot takes the map down with it")
    func dropClearsTheMap() {
        let link = FakeLink()
        let c = RobotController(link: link)
        var cloud: MapMesh?
        c.liveMap.onCloud = { cloud = $0 }
        link.connect()
        link.emitLiveTelemetry()

        link.dropUnexpectedly()
        #expect(cloud?.count == 0,
                "the pointcloud outlived the robot — exactly the reported bug")
    }

    // MARK: - Demo is NOT a disconnect

    @Test("demo mode keeps streaming — the reset must not kill the on-device sim")
    func demoStaysLive() {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()
        link.emitLiveTelemetry()

        link.status = .demo                  // on-device sim: live, just not a host
        #expect(c.connection.isLive)
        #expect(c.state.baseX == 3.5, "demo is live data; it must not be wiped")
    }
}
