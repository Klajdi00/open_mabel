//
//  MapStage.swift
//  MabelTeleop
//
//  The Navigate tab's 3D diorama: a SLAM room point cloud + the live MABEL
//  miniature, A* goal planning on a pinch, and the same pure-pursuit +
//  swerve-IK follower as the iOS app — driving the sim/real base over the
//  bridge with the identical twist convention.
//

import SwiftUI
import Combine
import Spatial
import RealityKit
import simd

@MainActor
final class MapStage: ObservableObject {

    @Published var hasGoal = false
    @Published var pathLen: Double = 0
    @Published var following = false        // pure-pursuit path following (Go)
    @Published var tracking = false         // 3rd-person FOLLOW: keep the robot centred

    weak var controller: RobotController?

    /// Gesture pivot (rotate/scale the whole diorama).
    let root = Entity()
    /// Room frame: 1 unit = 1 metre, origin at the room centre.
    let roomFrame = Entity()
    let rig = RobotRigEntity()
    private(set) var nav: RoomNav?
    private var rigLoaded = false

    // ── Live SLAM overlay (server /map: nvblox mesh + 2D map + sensor cloud) ──
    /// True while streaming the real robot's live map instead of a synthetic room.
    @Published private(set) var isLive = false
    @Published private(set) var liveConnected = false
    private let mapLink = MapStreamLink()
    private let liveRoot = Entity()
    private let liveMeshC = Entity()    // accumulated nvblox map (as colour points)
    private let liveGridC = Entity()    // 2D occupancy map (flat textured quad)
    private let liveCloudC = Entity()   // robot's CURRENT sensor view
    private let livePathC = Entity()    // live Nav2 route

    // Follower state (persisted so a re-plan continues from where it stopped).
    private var plannedPath: [CGPoint] = []
    private var pursuit = PurePursuit()
    private var swerve = SwerveDrive()
    private var followTask: Task<Void, Never>?
    private var curX = 0.0, curZ = 0.0, curYaw = 0.0

    /// The model's front is −X (sim chassis frame), 90° from +Z.
    private let frontYawOffset = Float.pi / 2

    /// Extra heading correction applied ONLY while streaming live SLAM. The odom
    /// pose's yaw frame is rotated 90° from the diorama, so without it the robot
    /// faces 90° off and the live cloud lands on its right. −π/2 turns the robot
    /// clockwise (top-down) so it faces the generated point cloud.
    private let liveYawOffset = -Float.pi / 2

    /// Swerve joints are driven by the local follower, not telemetry.
    private static let swerveJoints: Set<String> = [
        "fl_steer", "fl_drive", "fr_steer", "fr_drive", "b_steer", "b_drive"
    ]

    init() {
        root.addChild(roomFrame)
        roomFrame.addChild(rig.root)
        [liveMeshC, liveGridC, liveCloudC, livePathC].forEach { liveRoot.addChild($0) }
        roomFrame.addChild(liveRoot)
    }

    // MARK: - Live SLAM stream

    /// Stream the real robot's live map (mesh + 2D occupancy + sensor cloud + pose
    /// + route) from the bridge's /map socket, overlaid in the diorama. The robot
    /// is parked at the SLAM odom pose. Idempotent on the same host.
    func startLive(host: String, accessCode: String? = nil) {
        guard let url = NetworkConfig.directMapURL(host: host, accessCode: accessCode) else { return }
        isLive = true
        nav?.cloud.isEnabled = false                 // hide the synthetic room
        mapLink.onMesh  = { [weak self] c in self?.applyLivePoints(c, into: self?.liveMeshC, size: 0.016) }
        mapLink.onCloud = { [weak self] c in self?.applyLivePoints(c, into: self?.liveCloudC, size: 0.026) }
        mapLink.onGrid  = { [weak self] g in self?.applyLiveGrid(g) }
        mapLink.onPose  = { [weak self] p in self?.applyLivePose(p) }
        mapLink.onPath  = { [weak self] pts in
            guard let self else { return }
            PointCloudEntity.renderLivePath(pts, into: self.livePathC)
        }
        mapLink.connect(url)
        liveConnected = mapLink.connected
    }

    func stopLive() {
        isLive = false
        mapLink.disconnect()
        liveConnected = false
        [liveMeshC, liveGridC, liveCloudC, livePathC].forEach { $0.children.removeAll() }
        nav?.cloud.isEnabled = true
    }

    private func applyLivePoints(_ c: LiveCloud, into container: Entity?, size: Float) {
        guard let container else { return }
        container.children.removeAll()
        guard !c.points.isEmpty else { return }
        container.addChild(PointCloudEntity.coloredCloud(points: c.points, colors: c.colors, size: size))
        liveConnected = true
    }

    private func applyLiveGrid(_ g: LiveGrid) {
        liveGridC.children.removeAll()
        if let plane = PointCloudEntity.occupancyPlane(width: g.width, height: g.height,
                                                       resolution: g.resolution, origin: g.origin,
                                                       yaw: g.yaw, cells: g.cells) {
            liveGridC.addChild(plane)
        }
    }

    private func applyLivePose(_ p: MapOdomPose) {
        // Park the robot at the live SLAM odom pose (overrides the local follower).
        curX = Double(p.position.x); curZ = Double(p.position.z); curYaw = Double(p.yaw)
        poseRobot()
    }

    func loadRigIfNeeded() async {
        guard !rigLoaded else { return }
        rigLoaded = true
        await rig.load()
        poseRobot()
    }

    /// Latest known view extents in scene METRES (set by the hosting view).
    private var viewExtent: SIMD2<Float> = SIMD2(0.6, 0.45)

    /// Swap in a room: rebuild the cloud; the diorama refits to the view.
    func setRoom(_ room: SLAMRoom) {
        stopFollowing()
        nav?.cloud.removeFromParent()
        let built = PointCloudEntity.make(for: room)
        roomFrame.addChild(built.cloud)
        nav = built
        hasGoal = false; pathLen = 0
        curX = 0; curZ = 0; curYaw = 0
        swerve = SwerveDrive()
        plannedPath = []
        poseRobot()
        refit()
    }

    /// Fit the room diorama to the hosting view (windowed RealityViews map
    /// ~1360 points per metre, so metre-guessed scales overflow and clip).
    func setViewExtent(width: Float, height: Float) {
        viewExtent = SIMD2(width, height)
        refit()
    }

    private func refit() {
        guard let nav else { return }
        let extent = Float(max(nav.bounds.width, nav.bounds.height))
        let s = 0.8 * min(viewExtent.x, viewExtent.y) / max(extent, 1)
        roomFrame.scale = SIMD3(repeating: s)
        // Slightly below centre and TOWARD the user, clear of the window
        // plane (windows clip content behind it).
        roomFrame.position = SIMD3(0, -viewExtent.y * 0.18,
                                   max(0.06, 0.55 * s * extent / 2))
    }

    // MARK: - Goal + planning (pinch on the floor)

    /// `point` is in the room frame (metres). Plans from the robot's pose.
    func setGoal(_ point: SIMD3<Float>) {
        guard let nav else { return }
        var goal = CGPoint(x: CGFloat(point.x), y: CGFloat(point.z))
        let hw = nav.bounds.width / 2, hh = nav.bounds.height / 2
        goal.x = min(hw - 0.1, max(-hw + 0.1, goal.x))
        goal.y = min(hh - 0.1, max(-hh + 0.1, goal.y))

        let start = CGPoint(x: curX, y: curZ)
        var path = PathPlanner.plan(from: start, to: goal,
                                    bounds: nav.bounds, obstacles: nav.obstacles)
        if path.isEmpty { path = [start, goal] }
        PointCloudEntity.renderPath(path, goal: goal, into: nav.pathContainer)

        stopFollowing()
        plannedPath = path
        poseRobot()

        var len: Double = 0
        for i in 1..<path.count { len += Double(hypot(path[i].x - path[i-1].x, path[i].y - path[i-1].y)) }
        Haptics.rigid()
        hasGoal = true
        pathLen = len
    }

    // MARK: - Pure-pursuit following

    func startFollowing() {
        guard plannedPath.count > 1 else { return }
        pursuit.start(path: plannedPath, from: CGPoint(x: curX, y: curZ), yaw: curYaw)
        following = true
        followTask?.cancel()
        followTask = Task { [weak self] in
            var last = Date()
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(33))
                guard let self else { return }
                let now = Date()
                let dt = min(max(now.timeIntervalSince(last), 1.0 / 120), 1.0 / 20)
                last = now
                self.tick(dt: dt)
                if self.pursuit.finished {
                    Haptics.success()
                    self.stopFollowing()
                    return
                }
            }
        }
    }

    func stopFollowing() {
        followTask?.cancel(); followTask = nil
        controller?.stopBase()           // halt the bridge base
        following = false
    }

    private func tick(dt: Double) {
        pursuit.step(dt: dt)
        curX = pursuit.x; curZ = pursuit.z; curYaw = pursuit.yaw
        swerve.update(vx: pursuit.bodyVx, vy: pursuit.bodyVy, omega: pursuit.omega, dt: dt)
        poseRobot()
        // Drive the sim/real base with the SAME twist, in the drive convention:
        // forward = +front, strafe = +right (−left), yaw = +clockwise (−ccw).
        controller?.driveTwist(forward: pursuit.bodyVx, strafe: -pursuit.bodyVy, yaw: -pursuit.omega)
    }

    // MARK: - Robot posing

    func poseRobot() {
        rig.root.position = SIMD3(Float(curX), 0, Float(curZ))
        let heading = Float(curYaw) + frontYawOffset + (isLive ? liveYawOffset : 0)
        rig.root.orientation = simd_quatf(angle: heading, axis: SIMD3(0, 1, 0))
        let map = [("fl", "fl_steer", "fl_drive"), ("fr", "fr_steer", "fr_drive"), ("b", "b_steer", "b_drive")]
        for (key, steerID, driveID) in map {
            guard let m = swerve.state[key] else { continue }
            if let j = rig.joints[steerID] {
                j.node.orientation = simd_quatf(angle: Float(m.steer), axis: simd_normalize(j.axis))
            }
            if let j = rig.joints[driveID] {
                j.node.orientation = simd_quatf(angle: Float(m.wheel), axis: simd_normalize(j.axis))
            }
        }
        if tracking { applyTracking() }     // keep the robot centred while following
    }

    // MARK: - Follow (3rd-person: keep the robot centred in the diorama)

    /// Toggle FOLLOW. While on, the diorama re-centres on the robot every pose, so
    /// it stays put as it drives. Orbit (drag) + zoom (pinch) then rotate/zoom
    /// AROUND the robot (it sits at the pivot); look from above by moving your head.
    func setTracking(_ on: Bool) {
        tracking = on
        if on { applyTracking() }
        else { root.position = .zero; refit() }   // restore room-centred framing
    }

    /// Place the robot at the diorama pivot (so orbit/zoom revolve around it) and
    /// push the framing depth/height onto `root` so windows don't clip it.
    private func applyTracking() {
        guard let nav else { return }
        let s = roomFrame.scale.x
        let extent = Float(max(nav.bounds.width, nav.bounds.height))
        root.position = SIMD3(0, -viewExtent.y * 0.18, max(0.06, 0.55 * s * extent / 2))
        roomFrame.position = SIMD3(-Float(curX) * s, 0, -Float(curZ) * s)
    }

    /// Pose every articulated joint from the shared robot state (arms, lift,
    /// neck, torso, grips) — swerve joints are locally driven, base heading
    /// comes from `curYaw`.
    func applyJoints(_ j: [String: Double]) {
        for (id, joint) in rig.joints where !Self.swerveJoints.contains(id) {
            // Displacement from the ASSEMBLED pose = absolute qpos - MuJoCo ref, with a
            // joint missing from telemetry falling back to its own ref (the default
            // pose), never 0. See RigJoint.ref: the 14 ORCA hand joints and the 2
            // half-raised lift stages carry ref != 0.
            let value = Float(RobotRig.jointValue(id, mimic: joint.mimic, in: j) ?? Double(joint.ref)) - joint.ref
            if joint.type == "prismatic" {
                joint.node.position = joint.axis * value
            } else {
                joint.node.orientation = simd_quatf(angle: value, axis: simd_normalize(joint.axis))
            }
        }
    }

    // MARK: - Volume / view controls

    /// Reset orientation + refit to the current view.
    func recenter() {
        root.orientation = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        root.scale = .one
        if tracking { applyTracking() } else { root.position = .zero; refit() }
    }

    /// Re-fit the diorama to the view (reset zoom, keep the current orientation).
    func fitView() {
        root.scale = .one
        if tracking { applyTracking() } else { refit() }
    }

    /// True 1:1 scale with the room floor on `floorY` (volumetric window).
    /// Rooms are metres across, so expect the volume to crop — that's the
    /// point of life size.
    func lifeSize(floorY: Float) {
        var t = roomFrame.transform
        t.scale = .one
        t.translation = SIMD3(0, floorY, 0)
        roomFrame.move(to: t, relativeTo: roomFrame.parent, duration: 0.45, timingFunction: .easeInOut)
    }

    /// Keep the current scale but sit the room floor on `floorY`.
    func placeOnFloor(y: Float) {
        var t = roomFrame.transform
        t.translation = SIMD3(t.translation.x, y, t.translation.z)
        roomFrame.move(to: t, relativeTo: roomFrame.parent, duration: 0.35, timingFunction: .easeInOut)
    }

    /// Fit back inside the given extents (volume interior).
    func fitTo(width: Float, height: Float) {
        setViewExtent(width: width, height: height)
    }

    // MARK: - Diorama gestures

    private var yaw0: Float?
    private var scale0: Float?

    func orbit(dx: Float) {
        if yaw0 == nil { yaw0 = root.orientation.angle * (root.orientation.axis.y < 0 ? -1 : 1) }
        root.orientation = simd_quatf(angle: (yaw0 ?? 0) + dx * 2.4, axis: SIMD3(0, 1, 0))
    }
    func orbitEnded() { yaw0 = nil }

    func zoom(_ m: Float) {
        if scale0 == nil { scale0 = root.scale.x }
        root.scale = SIMD3(repeating: max(0.4, min(3.0, (scale0 ?? 1) * m)))
    }
    func zoomEnded() { scale0 = nil }
}

// MARK: - RealityView host

struct MapStageView: View {
    @EnvironmentObject private var controller: RobotController
    @ObservedObject var map: MapStage

    private let tick = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    @State private var fittedSize: CGSize = .zero

    var body: some View {
        GeometryReader { geo in
            realityBody(size: geo.size)
        }
    }

    private func realityBody(size: CGSize) -> some View {
        RealityView { content in
            map.root.position = .zero
            content.add(map.root)
            fit(content, size: size)
            await map.loadRigIfNeeded()
        } update: { content in
            if size != fittedSize { fit(content, size: size) }
        }
        .gesture(
            // Look + pinch on the floor → drop a navigation goal.
            SpatialTapGesture()
                .targetedToAnyEntity()
                .onEnded { v in
                    guard v.entity.name == "floor" else { return }
                    let p = v.convert(v.location3D, from: .local, to: map.roomFrame)
                    map.setGoal(p)
                }
        )
        .simultaneousGesture(
            DragGesture(minimumDistance: 6)
                .onChanged { v in map.orbit(dx: Float(v.translation.width) / 600) }
                .onEnded { _ in map.orbitEnded() }
        )
        .simultaneousGesture(
            MagnifyGesture()
                .onChanged { v in map.zoom(Float(v.magnification)) }
                .onEnded { _ in map.zoomEnded() }
        )
        .onReceive(tick) { _ in
            map.applyJoints(controller.liveModelJoints)
        }
        // Stream the real robot's live map whenever we have a direct host (LAN /
        // Tailscale); fall back to the synthetic room when there's none.
        .task(id: controller.videoHost) {
            if let h = controller.videoHost { map.startLive(host: h) }
            else { map.stopLive() }
        }
        .onDisappear { map.stopLive() }
    }

    /// Convert the view's point size into scene metres and refit the diorama.
    private func fit(_ content: RealityViewContent, size: CGSize) {
        guard size.width > 1, size.height > 1 else { return }
        let o = content.convert(Point3D(x: 0, y: 0, z: 0), from: .local, to: .scene)
        let c = content.convert(Point3D(x: size.width, y: size.height, z: 0), from: .local, to: .scene)
        map.setViewExtent(width: abs(Float(c.x - o.x)), height: abs(Float(c.y - o.y)))
        Task { @MainActor in fittedSize = size }
    }
}
