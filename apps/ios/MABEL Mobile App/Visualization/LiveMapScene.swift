//
//  LiveMapScene.swift
//  MABEL Mobile App
//
//  The LIVE SLAM map: a SceneKit scene fed by the robot's nvblox mesh and odom
//  pose over MapStreamLink. Unlike PointCloudScene (procedural stand-in rooms),
//  nothing here is synthesised — the point cloud is swapped in wholesale each
//  time a fresh mesh snapshot arrives, and the articulated MABEL (same rig as
//  every other tab) is parked at the robot's live pose in the map frame.
//
//  Coordinates arrive already rebased into SceneKit space (y-up) by the link, so
//  this file only places geometry; the ROS→SceneKit conversion lives in
//  MapStreamLink.
//

import SceneKit
import simd
import UIKit

/// Where the 3rd-person orbit pivots, measured up from the robot's base. Chest
/// height — the same point `frameCamera` looks at and the physics arc orbits, so
/// every camera path in the app turns around the SAME spot on the robot.
extension LiveMapScene {
    static let orbitEyeHeight: Float = 0.55
}

/// Live handles the view coordinator drives each time the stream ticks.
final class LiveMapRig {
    let root = SCNNode()
    let meshContainer = SCNNode()        // the accumulated nvblox map (mesh / cloud)
    let gridContainer = SCNNode()        // the 2D occupancy map (flat textured quad)
    let cloudContainer = SCNNode()       // the robot's LIVE sensor cloud (current view)
    let camCloudContainer = SCNNode()    // per-CAMERA clouds (96xx data plane), one node per stream
    let pathContainer = SCNNode()        // the Nav2 planned route (green polyline)
    let goalContainer = SCNNode()        // the active goal marker (orange pillar)
    let robot: SCNNode                   // articulated MABEL at the live pose
    let robotJoints: [String: RigJoint]
    private var handles: [String: SCNNode] = [:]
    let camera = SCNNode()
    /// 1st-person camera — parented to the HEAD link at the zed_left mount pose
    /// (from the MJCF camera rig), so it rides every neck/odom motion. The main
    /// viewer camera flies into this pose on the 3rd→1st transition, then hands
    /// the view over to this node.
    let povCamera = SCNNode()
    /// zed_left mount orientation in the head frame — the base the 1st-person
    /// look-around offsets rotate from.
    var povMountQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    // MARK: The STAGE — the ground the robot stands on (see GroundPlane.swift)
    //
    // Two separate layers with two separate chips, because they answer two
    // different questions. The SURFACE answers "where is the floor" (console
    // "Floor"); the GRID answers "how far is that" (console "Grid"). They used to
    // be one faint 16 m square of lines doing neither job well, wired to a chip
    // called "Ground" — while the chip called "2D Grid" drove the SLAM occupancy
    // plane, which on a robot without nvblox running has no data at all. That is
    // the whole reason both buttons read as broken.
    //
    // `stage` carries both and SNAPS to the robot's grid cell each pose update, so
    // the ground never runs out and the lines still read as fixed world.
    let stage = SCNNode()
    var floorSurface: SCNNode?
    var floorGrid: SCNNode?
    /// The grounding pool under the robot. Parented to `robot`, not `stage` — it
    /// must track continuously or it visibly lags its own robot on the 1 m snap.
    var contactShadow: SCNNode?

    /// Every node of the ROBOT's own mesh, captured BEFORE anything else is
    /// parented into the robot subtree. That ordering is the point: the LiDAR ring,
    /// the TF triads, the Cam 3D video planes and the POV camera all hang off joint
    /// nodes, so `robot.isHidden = true` would take the sensor layers down with the
    /// model. Hiding exactly this list hides the robot and nothing else.
    private var robotGeometry: [SCNNode] = []

    /// No extra heading offset: the model's own front (local −X) already coincides
    /// with the robot's sensor-forward axis, so the raw `odom` orientation aims the
    /// nose straight at the live sensor cloud / mapped surface the robot is looking
    /// at. Verified against thor's live odom + cloud stream — a +90°-about-Y offset
    /// (the old value) put the nose 90° off the cloud, which read as "the robot is
    /// facing sideways into empty space."
    private var hasFramed = false

    /// `parts` defaults to the connected robot's rig; tests pass an explicit pick.
    init(parts: (SCNNode, [String: RigJoint], [String: SCNNode]) = RealRobotScene.makeRobotNodeWithHandles()) {
        let (robotNode, joints, palmHandles) = parts
        robot = robotNode
        robotJoints = joints
        handles = palmHandles
        // Deterministic from frame one. They are built hidden, but relying on that
        // alone left the state implicit — and setHandlesVisible could not correct it
        // either way, because it was looking for the nodes in the wrong place.
        setHandlesVisible(false)
        // BEFORE any sensor layer is attached — see `robotGeometry`.
        robotNode.enumerateHierarchy { n, _ in
            if n.geometry != nil { robotGeometry.append(n) }
        }
        buildStage()
        root.addChildNode(stage)
        root.addChildNode(meshContainer)
        root.addChildNode(gridContainer)
        root.addChildNode(cloudContainer)
        root.addChildNode(camCloudContainer)
        root.addChildNode(pathContainer)
        root.addChildNode(goalContainer)
        root.addChildNode(robot)
        robot.addChildNode(lidarContainer)   // the LiDAR ring rides the base pose
        mountPovCamera()
    }

    /// Assemble the ground: wash + reference grid under `stage`, contact pool under
    /// the robot. All three are cheap, static geometry built once.
    private func buildStage() {
        stage.name = "stage"
        let s = GroundPlane.surface()
        let g = GroundPlane.grid()
        stage.addChildNode(s)
        stage.addChildNode(g)
        floorSurface = s
        floorGrid = g
        let sh = GroundPlane.contactShadow()
        robot.addChildNode(sh)
        contactShadow = sh
    }

    /// Keep the ground under the robot. Called on every pose update (live AND the
    /// demo plant) — the SNAP is what lets the floor follow without the grid lines
    /// appearing glued to the robot. See GroundPlane's header.
    func syncStage() {
        stage.simdPosition = GroundPlane.snap(to: robot.simdPosition)
    }

    /// Attach the 1st-person camera to the head link at the zed_left mount pose
    /// (mabel_camera_rig.json, MJCF head-body frame). Both MuJoCo and SceneKit
    /// cameras look along −Z with +Y up, so the mount transform drops straight in.
    /// The head link's articulation node is keyed by its driving joint (neck_roll).
    private func mountPovCamera() {
        guard let head = robotJoints["neck_roll"]?.node else { return }
        let cam = SCNCamera()
        cam.projectionDirection = .vertical
        cam.fieldOfView = 83.3               // zed_left true vertical FOV
        cam.zNear = 0.12                     // just past the head shell
        cam.zFar = 400
        cam.wantsHDR = true
        povCamera.camera = cam
        povCamera.name = "povCamera"
        povCamera.position = SCNVector3(-0.0317, 0.006, -0.1089)
        // mount_quat (w,x,y,z) → SCN (x,y,z,w); zed_left is ~identity in the head frame.
        povCamera.orientation = SCNQuaternion(-1.6e-07, 0.0004, 0.0004, 0.99999984)
        povMountQuat = povCamera.simdOrientation
        head.addChildNode(povCamera)
    }

    /// Replace the whole map with a fresh snapshot. A real nvblox snapshot carries
    /// triangles → render shaded SURFACES (the RViz mesh); a vertices-only snapshot
    /// falls back to a point cloud.
    func apply(mesh: MapMesh) {
        meshContainer.childNodes.forEach { $0.removeFromParentNode() }
        guard mesh.count > 0 else { return }
        let node = mesh.isMesh
            ? PointCloudScene.meshNode(points: mesh.points, colors: mesh.colors, indices: mesh.indices)
            : PointCloudScene.pointNode(points: mesh.points, colors: mesh.colors)
        meshContainer.addChildNode(node)
        if !hasFramed { frameCamera(on: robot.simdPosition); hasFramed = true }
    }

    /// Live-cloud decimation: render every Nth point (1 = full density). Set from
    /// the pull-up console's density control; applied on the next cloud frame.
    var cloudStride = 1

    /// The robot's LIVE sensor cloud (its CURRENT view) — replaced every frame.
    /// Rendered as points (slightly larger than the map) so it reads as "what the
    /// robot sees right now" overlaid on the accumulated map.
    func apply(cloud: MapMesh) {
        cloudContainer.childNodes.forEach { $0.removeFromParentNode() }
        guard cloud.count > 0 else { return }
        if cloudStride > 1 {
            let pts = stride(from: 0, to: cloud.points.count, by: cloudStride).map { cloud.points[$0] }
            let cols = stride(from: 0, to: cloud.colors.count, by: cloudStride).map { cloud.colors[$0] }
            cloudContainer.addChildNode(PointCloudScene.pointNode(points: pts, colors: cols))
        } else {
            cloudContainer.addChildNode(
                PointCloudScene.pointNode(points: cloud.points, colors: cloud.colors))
        }
    }

    /// Per-CAMERA pointcloud (96xx data plane, one WS per camera): replaced per
    /// stream on every frame, independent of the SLAM `cloudContainer`. Same
    /// stride decimation as the live cloud. Empty cloud clears that stream.
    private var camCloudNodes: [String: SCNNode] = [:]
    /// Drop any per-camera cloud whose toggle is OFF — view-driven truth, so
    /// a toggle always clears the points even if a frame raced the close.
    func pruneCameraClouds(keep: Set<String>) {
        for (stream, node) in camCloudNodes where !keep.contains(stream) {
            node.removeFromParentNode()
            camCloudNodes[stream] = nil
        }
    }
    func apply(cameraCloud stream: String, cloud: MapMesh) {
        camCloudNodes[stream]?.removeFromParentNode()
        camCloudNodes[stream] = nil
        guard cloud.count > 0 else { return }
        let node: SCNNode
        if cloudStride > 1 {
            let pts = stride(from: 0, to: cloud.points.count, by: cloudStride).map { cloud.points[$0] }
            let cols = stride(from: 0, to: cloud.colors.count, by: cloudStride).map { cloud.colors[$0] }
            node = PointCloudScene.pointNode(points: pts, colors: cols)
        } else {
            node = PointCloudScene.pointNode(points: cloud.points, colors: cloud.colors)
        }
        camCloudNodes[stream] = node
        camCloudContainer.addChildNode(node)
    }

    /// The 2D occupancy map (RViz "Map") — a flat textured quad laid under the
    /// mesh. Replaced whenever a fresh grid arrives. Empty grid clears it.
    func apply(grid: MapGrid) {
        gridContainer.childNodes.forEach { $0.removeFromParentNode() }
        guard grid.count > 0, let node = LiveMapScene.occupancyNode(grid) else { return }
        gridContainer.addChildNode(node)
        if !hasFramed { frameCamera(on: robot.simdPosition); hasFramed = true }
    }

    /// Show / hide the optional overlays (the inspector exposes toggles).
    func setGridHidden(_ hidden: Bool)  { gridContainer.isHidden = hidden }
    func setCloudHidden(_ hidden: Bool) { cloudContainer.isHidden = hidden }
    /// The console's Cloud chip is the MASTER switch over every pointcloud, so it
    /// must also blank the per-camera planes — `pruneCameraClouds` + closing the
    /// sockets already does that, but a frame in flight when the toggle lands
    /// would otherwise repaint one node before the prune. Hiding the container is
    /// the belt to that braces: OFF means OFF, with no one-frame flicker.
    func setCameraCloudsHidden(_ hidden: Bool) { camCloudContainer.isHidden = hidden }
    func setMeshHidden(_ hidden: Bool)  { meshContainer.isHidden = hidden }
    func setLidarHidden(_ hidden: Bool) { lidarContainer.isHidden = hidden }

    /// "Floor" — the ground surface AND its contact pool. They are one idea (the
    /// robot is standing on something) and must never be half-on: a shadow with no
    /// floor under it is a smudge in mid-air.
    func setFloorHidden(_ hidden: Bool) {
        floorSurface?.isHidden = hidden
        contactShadow?.isHidden = hidden
    }
    /// "Grid" — the 1 m / 5 m reference lines only. Independent of the floor so the
    /// operator can keep the metric cadence over a fully mapped room without the
    /// wash brightening the SLAM mesh underneath.
    func setFloorGridHidden(_ hidden: Bool) { floorGrid?.isHidden = hidden }
    /// "Robot" — MABEL's own mesh, and ONLY that. The LiDAR ring, TF triads and
    /// Cam 3D planes live in the same subtree and keep drawing (see
    /// `robotGeometry`), so hiding the model reveals the sensor picture instead of
    /// blanking it.
    func setRobotHidden(_ hidden: Bool) {
        for n in robotGeometry { n.isHidden = hidden }
    }

    // MARK: - LiDAR ring + TF tree (Vision Pro parity, ported 2026-07-28)

    /// LiDAR-derived point ring — its OWN layer here (the Vision Pro couples it to
    /// the cloud master; iOS gives it a chip, see ViewerSettings.laserOn). The planar
    /// sweep on `robot_state` streams ~5 Hz from EVERY plant (sim mj_ray, real
    /// RPLIDAR, demo ray-cast), so this layer has live geometry even on the Mac sim
    /// path where the /map CLOUD (nvblox depth) stream is idle.
    ///
    /// Parented to the ROBOT node so it rides odom; points are rebased from the
    /// MuJoCo base frame (x fwd, y left, z up) into the rig's SceneKit frame
    /// (nose = local −X): (x, y, z)_base → (−x, z, y)_scn.
    let lidarContainer = SCNNode()
    private static let lidarHeight: Float = 0.18     // emitter height on the mast

    func apply(lidar scan: (angleMin: Double, angleInc: Double, rangeMax: Double,
                            ranges: [Double]), stride strideN: Int) {
        lidarContainer.childNodes.forEach { $0.removeFromParentNode() }
        guard !scan.ranges.isEmpty else { return }
        var pts: [SCNVector3] = []
        var cols: [SCNVector3] = []
        pts.reserveCapacity(scan.ranges.count)
        let n = scan.ranges.count
        let step = max(1, strideN)
        var i = 0
        while i < n {
            let r = scan.ranges[i]
            defer { i += step }
            guard r < scan.rangeMax - 1e-3 else { continue }     // non-return
            let a = scan.angleMin + Double(i) * scan.angleInc
            let x = Float(r * cos(a)), y = Float(r * sin(a))
            pts.append(SCNVector3(-x, Self.lidarHeight, y))
            // Range-graded colour: near = warm orange → far = control-center blue.
            let t = Float(min(1, r / scan.rangeMax))
            cols.append(SCNVector3(0.95 - 0.65 * t, 0.45 + 0.17 * t, 0.20 + 0.80 * t))
        }
        guard !pts.isEmpty else { return }
        lidarContainer.addChildNode(PointCloudScene.pointNode(points: pts, colors: cols))
    }

    /// RViz-style TF-tree layer: an axis triad (x red · y green · z blue) parented
    /// to the base link and every articulated joint node — the triads ride the
    /// live FK for free, so the frames animate with the robot. Built LAZILY on the
    /// first toggle (~40 triads × 4 nodes is not something to pay for on every
    /// launch when the layer defaults off); hidden thereafter by name.
    private var tfBuilt = false
    func setTFHidden(_ hidden: Bool) {
        if !hidden && !tfBuilt {
            tfBuilt = true
            robot.addChildNode(Self.tfTriad(scale: 0.11))           // base_link frame
            for (_, j) in robotJoints { j.node.addChildNode(Self.tfTriad(scale: 0.055)) }
        }
        guard tfBuilt else { return }
        robot.enumerateHierarchy { n, _ in if n.name == "tf_axes" { n.isHidden = hidden } }
    }

    /// One constant-lit axis triad (thin boxes along +x/+y/+z with a white hub).
    private static func tfTriad(scale s: Float) -> SCNNode {
        let triad = SCNNode()
        triad.name = "tf_axes"
        func axis(_ color: UIColor, _ rot: SCNVector3) -> SCNNode {
            let g = SCNBox(width: CGFloat(s), height: CGFloat(s) * 0.09,
                           length: CGFloat(s) * 0.09, chamferRadius: 0)
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = color
            m.emission.contents = color
            m.readsFromDepthBuffer = true
            g.materials = [m]
            let n = SCNNode(geometry: g)
            n.position = SCNVector3(s / 2, 0, 0)     // grow along +x, then rotate
            let holder = SCNNode()
            holder.eulerAngles = rot
            holder.addChildNode(n)
            return holder
        }
        triad.addChildNode(axis(UIColor(rgb: 0xE5484D), SCNVector3(0, 0, 0)))            // +x
        triad.addChildNode(axis(UIColor(rgb: 0x30A46C), SCNVector3(0, 0, Float.pi / 2))) // +y
        triad.addChildNode(axis(UIColor(rgb: 0x4D9DFF), SCNVector3(0, -Float.pi / 2, 0))) // +z
        let hub = SCNNode(geometry: SCNSphere(radius: CGFloat(s) * 0.07))
        hub.geometry!.firstMaterial!.lightingModel = .constant
        hub.geometry!.firstMaterial!.diffuse.contents = UIColor.white
        triad.addChildNode(hub)
        return triad
    }

    /// Park the robot at its live pose (already in SceneKit space). The orientation
    /// is applied raw — see `frontOffset` note above for why there's no yaw fudge.
    func apply(pose: MapPose) {
        robot.simdPosition = simd_float3(pose.position.x, pose.position.y, pose.position.z)
        robot.simdOrientation = pose.orientation
        syncStage()
    }

    /// Draw the live Nav2 route as a green polyline raised just above the floor.
    /// An empty path (goal reached / canceled) clears BOTH the route and the goal.
    func apply(path: MapPath) {
        pathContainer.childNodes.forEach { $0.removeFromParentNode() }
        let pts = path.points
        guard pts.count >= 2 else {
            if pts.isEmpty {
                goalContainer.childNodes.forEach { $0.removeFromParentNode() }
                goalMarker = nil
            }
            return
        }
        let raised = pts.map { SCNVector3($0.x, $0.y + 0.05, $0.z) }
        let src = SCNGeometrySource(vertices: raised)
        var idx = [Int32]()
        for i in 0..<(raised.count - 1) { idx.append(Int32(i)); idx.append(Int32(i + 1)) }
        let elem = SCNGeometryElement(indices: idx, primitiveType: .line)
        let geo = SCNGeometry(sources: [src], elements: [elem])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = UIColor(rgb: 0x34D058)
        m.emission.contents = UIColor(rgb: 0x34D058)
        geo.materials = [m]
        pathContainer.addChildNode(SCNNode(geometry: geo))
        // Anchor the marker at the route's end — but never stomp the operator's
        // placed marker (with its yaw ring) while it exists; Nav2's route ends at
        // the same goal anyway.
        if goalMarker == nil { apply(goal: raised.last!, yaw: nil) }
    }

    /// The active goal marker (green pillar + heading arrow + draggable yaw ring).
    private(set) var goalMarker: SCNNode?

    /// Plant the goal marker at a SceneKit ground point, aimed at `yaw` (SceneKit
    /// rotation about +Y; nil keeps the arrow at the marker's default). Shown the
    /// instant the operator taps, before Nav2's route returns. The RING is a drag
    /// handle — the view coordinator rotates the marker and re-sends the goal.
    func apply(goal p: SCNVector3, yaw: Float?) {
        goalContainer.childNodes.forEach { $0.removeFromParentNode() }
        let green = UIColor(rgb: 0x34D058)
        func glow(_ g: SCNGeometry) {
            let m = g.firstMaterial!
            m.lightingModel = .constant
            m.diffuse.contents = green
            m.emission.contents = green
        }

        let marker = SCNNode()
        marker.name = "goal_marker"
        marker.simdPosition = simd_float3(p.x, 0, p.z)
        if let yaw { marker.eulerAngles.y = yaw }

        // SIZES (2026-07-16): scaled to ~55% of the original, in step with the
        // Vision Pro app (both read this same marker). The old marker was a 0.8 m-
        // wide, 0.7 m-tall green tower that dominated the scene and hid the floor
        // it was pointing at. It is a map annotation, not a monument — the ring
        // stays a comfortable touch/drag target at 0.44 m across.
        let pillar = SCNNode(geometry: SCNCylinder(radius: 0.025, height: 0.38))
        glow(pillar.geometry!)
        pillar.name = "goal_pillar"
        pillar.position = SCNVector3(0, 0.19, 0)
        marker.addChildNode(pillar)

        // Heading arrow — a cone lying along the marker's +X (the goal yaw).
        let cone = SCNCone(topRadius: 0, bottomRadius: 0.042, height: 0.14)
        glow(cone)
        let arrow = SCNNode(geometry: cone)
        arrow.name = "goal_arrow"
        arrow.position = SCNVector3(0.29, 0.02, 0)
        arrow.eulerAngles.z = -.pi / 2          // cone +Y → world +X
        marker.addChildNode(arrow)

        // Yaw ring — the drag donut around the pillar. Kept proportionally the
        // widest part so it stays an easy target after the size cut.
        let ring = SCNNode(geometry: SCNTube(innerRadius: 0.155, outerRadius: 0.22, height: 0.014))
        glow(ring.geometry!)
        ring.geometry!.firstMaterial!.transparency = 0.85
        ring.name = "goal_ring"
        ring.position = SCNVector3(0, 0.012, 0)
        ring.runAction(.repeatForever(.sequence([
            .scale(to: 1.12, duration: 0.8), .scale(to: 1.0, duration: 0.8)])))
        marker.addChildNode(ring)

        goalContainer.addChildNode(marker)
        goalMarker = marker
    }

    /// Rotate the placed marker (ring drag feedback).
    func setGoalYaw(_ yaw: Float) { goalMarker?.eulerAngles.y = yaw }

    /// Drop the placed goal marker — the operator cancelled a goal that was staged
    /// but never dispatched, so nothing was sent and nothing needs stopping.
    func clearGoal() {
        goalContainer.childNodes.forEach { $0.removeFromParentNode() }
        goalMarker = nil
    }

    /// Pose every articulated joint (arms / lift / neck / fingers) from the shared
    /// robot state so the live map's MABEL matches the model on the other tabs.
    /// Each joint moves about its OWN axis (mirrors RealRobotView.apply()).
    /// `except` freezes joints under an active grab-IK drag — they show the local
    /// prediction for lag-free finger feedback instead of fighting telemetry.
    func apply(joints j: [String: Double], except frozen: Set<String> = []) {
        for (id, joint) in robotJoints where !frozen.contains(id) {
            // Displacement from the assembled pose: streamed (absolute) qpos − MuJoCo
            // ref. Only the ORCA finger/wrist joints have ref ≠ 0; without this the
            // hands rendered offset (splayed thumb) — the hand-viz bug. Same as VP.
            // Missing joint -> fall back to its ref (qpos0), not 0; see RealRobotView.
            let value = Float(RobotRig.jointValue(id, mimic: joint.mimic, in: j) ?? Double(joint.ref)) - joint.ref
            if joint.type == "prismatic" {
                joint.node.simdPosition = joint.axis * value
            } else {
                joint.node.simdOrientation = simd_quatf(angle: value, axis: simd_normalize(joint.axis))
            }
        }
    }

    /// The green palm balls + orientation rings (built hidden by the rig loader) —
    /// shown only while TASK-space manipulation is active.
    func setHandlesVisible(_ visible: Bool) {
        // Drive the nodes the rig actually built. This used to look for a child named
        // "handle" under `<side>_arm_7`, but the balls hang off `l_hand_mount` /
        // `r_hand_mount` — and `childNodes` is direct children only — so the guard
        // always failed and the function silently did nothing in BOTH directions:
        // the task-space handles could never appear, and nothing could hide them.
        for (_, ball) in handles { ball.isHidden = !visible }
    }

    /// The palm balls (each carrying its orientation rings as children), keyed by side.
    /// Shown only while TASK-space manipulation is armed.

    func handle(side: String) -> SCNNode? { handles[side] }

    /// Frame the orbit camera on a target point (the robot, once we have a map).
    /// Default = FRONT-ISOMETRIC of the robot (heading-aware: nose = local −X),
    /// slightly elevated and close enough that the robot reads clearly with a
    /// band of map/point-cloud context around it. Re-center returns here too.
    func frameCamera(on target: simd_float3) {
        let up = simd_float3(0, 1, 0)
        let rw = robot.simdWorldTransform
        var front = -simd_make_float3(rw.columns.0); front.y = 0
        front = simd_length(front) < 1e-5 ? simd_float3(0, 0, 1) : simd_normalize(front)
        var lateral = simd_make_float3(rw.columns.2); lateral.y = 0
        lateral = simd_length(lateral) < 1e-5 ? simd_float3(1, 0, 0) : simd_normalize(lateral)
        let dir = simd_normalize(front + 0.55 * lateral + 0.62 * up)
        let center = target + simd_float3(0, 0.55, 0)
        camera.simdPosition = center + dir * 3.1
        // Explicit up vector — the single-arg look(at:) inherits the camera's
        // CURRENT roll (e.g. after a POV return), tilting the floor.
        camera.look(at: SCNVector3(center), up: SCNVector3(0, 1, 0), localFront: SCNVector3(0, 0, -1))
    }

    /// FIT the whole map (mesh + occupancy + robot) to ~`fraction` of the view,
    /// keeping the iso viewing direction. No FOV math — it converges on the
    /// screen-projected bounds, so it's correct at any aspect ratio.
    func fitCamera(view: SCNView, fraction: CGFloat) {
        var lo = simd_float3(repeating: Float.greatestFiniteMagnitude)
        var hi = simd_float3(repeating: -Float.greatestFiniteMagnitude)
        func accumulate(_ node: SCNNode) {
            node.enumerateHierarchy { n, _ in
                guard n.geometry != nil, !n.isHidden else { return }
                let (a, b) = n.boundingBox
                guard a.x <= b.x else { return }
                for sx in [a.x, b.x] { for sy in [a.y, b.y] { for sz in [a.z, b.z] {
                    let w = n.simdConvertPosition(simd_float3(sx, sy, sz), to: nil)
                    lo = simd_min(lo, w); hi = simd_max(hi, w)
                } } }
            }
        }
        accumulate(meshContainer); accumulate(gridContainer); accumulate(robot)
        guard lo.x <= hi.x else { frameCamera(on: robot.simdPosition); return }
        let center = (lo + hi) / 2
        var corners: [simd_float3] = []
        for sx in [lo.x, hi.x] { for sy in [lo.y, hi.y] { for sz in [lo.z, hi.z] {
            corners.append(simd_float3(sx, sy, sz))
        } } }

        let dir = simd_normalize(simd_float3(4.5, 4.0, 4.5))
        var d = max(2, simd_length(hi - lo))
        let bw = max(1, view.bounds.width), bh = max(1, view.bounds.height)
        SCNTransaction.begin(); SCNTransaction.animationDuration = 0
        for _ in 0..<7 {
            camera.simdPosition = center + dir * d
            // Explicit up — the single-arg look(at:) inherits the camera's
            // current ROLL (from orbiting), which tilted the fitted horizon.
            camera.look(at: SCNVector3(center), up: SCNVector3(0, 1, 0),
                        localFront: SCNVector3(0, 0, -1))
            var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
            var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude
            for c in corners {
                let s = view.projectPoint(SCNVector3(c))
                minX = min(minX, CGFloat(s.x)); maxX = max(maxX, CGFloat(s.x))
                minY = min(minY, CGFloat(s.y)); maxY = max(maxY, CGFloat(s.y))
            }
            let frac = max((maxX - minX) / bw, (maxY - minY) / bh)
            guard frac.isFinite, frac > 0.0001 else { break }
            d = max(1.5, min(80, d * Float(frac / fraction)))
        }
        camera.simdPosition = center + dir * d
        camera.look(at: SCNVector3(center), up: SCNVector3(0, 1, 0),
                    localFront: SCNVector3(0, 0, -1))
        SCNTransaction.commit()
    }

}

/// Smooth 3rd-person chase camera shared by the live + demo maps. It tracks a
/// target node's position and HEADING (yaw only — so the horizon stays perfectly
/// level even if the robot pitches or rolls), critically damps toward it every
/// render frame (so a 2 Hz /odom stream still looks smooth, never jittery), and
/// applies a user-controlled zoom `distance` while ALWAYS keeping it centred.
final class FollowCamera {
    var distance: Float = 4.5
    private let minD: Float = 1.5, maxD: Float = 22
    // User ORBIT offsets (pan): azimuth relative to "behind the robot" + elevation.
    // Lets the operator swing around to face the robot, or tilt to a top-down view,
    // all while the camera keeps tracking + centring the robot.
    private var yawOffset: Float = 0
    private var pitch: Float = 0.42
    private let minPitch: Float = 0.05, maxPitch: Float = 1.50   // ~86° ≈ top-down

    private var pos: simd_float3?
    private var look: simd_float3?
    private var yaw: Float?

    func reset() { pos = nil; look = nil; yaw = nil; yawOffset = 0; pitch = 0.42 }

    /// Pinch handler: gesture scale > 1 zooms IN (camera moves closer).
    func zoom(by scale: Float) {
        guard scale > 0 else { return }
        distance = max(minD, min(maxD, distance / scale))
    }

    /// Pan-to-orbit: drag horizontally to swing around the robot (e.g. round to its
    /// front to face it), drag up to tilt toward a top-down view.
    func orbit(dx: Float, dy: Float) {
        yawOffset -= dx * 0.006
        pitch = max(minPitch, min(maxPitch, pitch - dy * 0.006))
    }

    /// Advance one render frame. `m` = target world transform; `dt` seconds.
    /// Returns (position, look-at, up) — `up` keeps the horizon level and eases to
    /// the robot's forward near top-down so look(at:) never goes degenerate.
    func step(target m: simd_float4x4, dt: Float) -> (pos: simd_float3, look: simd_float3, up: simd_float3) {
        let p = simd_float3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
        // Robot nose = local −X taken to world, flattened to the ground (yaw only).
        var f = simd_float3(-m.columns.0.x, 0, -m.columns.0.z)
        if simd_length(f) < 1e-4 { f = simd_float3(0, 0, 1) }
        f = simd_normalize(f)
        let yTarget = atan2(f.x, f.z)
        let yNow = smoothAngle(yaw ?? yTarget, yTarget, rate: 7, dt: dt); yaw = yNow
        let sf = simd_float3(sin(yNow), 0, cos(yNow))             // robot forward (level)
        let azim = yNow + .pi + yawOffset                        // behind the robot + user offset
        let cp = cos(pitch), sp = sin(pitch)
        let dir = simd_float3(cp * sin(azim), sp, cp * cos(azim))
        let center = p + simd_float3(0, 0.45, 0)
        let dPos = center + dir * distance
        let nP = smoothVec(pos ?? dPos, dPos, rate: 9, dt: dt); pos = nP
        let nL = smoothVec(look ?? center, center, rate: 12, dt: dt); look = nL
        // Level horizon when low; ease the up-vector to the robot's forward as we
        // approach top-down (so the robot points "up" and there's no gimbal flip).
        let blend = max(0, min(1, (pitch - 1.0) / 0.45))
        let up = simd_normalize(simd_float3(0, 1, 0) * (1 - blend) + sf * blend)
        return (nP, nL, up)
    }

    private func smoothVec(_ a: simd_float3, _ b: simd_float3, rate: Float, dt: Float) -> simd_float3 {
        a + (b - a) * (1 - exp(-rate * dt))
    }
    private func smoothAngle(_ a: Float, _ b: Float, rate: Float, dt: Float) -> Float {
        var d = b - a
        while d >  .pi { d -= 2 * .pi }
        while d < -.pi { d += 2 * .pi }
        return a + d * (1 - exp(-rate * dt))
    }
}

enum LiveMapScene {

    static func make() -> (SCNScene, LiveMapRig) {
        let scene = SCNScene()
        scene.background.contents = UIColor(rgb: 0x0A0908)

        let rig = LiveMapRig()
        scene.rootNode.addChildNode(rig.root)

        // The ground lives on the RIG's `stage` node now (built in LiveMapRig.init,
        // geometry in GroundPlane.swift) — it has to follow the robot, and a node
        // parked on the scene root cannot. What used to be here was a single
        // 16 × 16 m square of 6 %-white lines: no surface, a hard edge, and fixed
        // at the odom origin, so driving away left the robot standing on nothing.

        // Constant-shaded points carry their own colour; a soft ambient keeps the
        // PBR robot readable, a key light gives it form (the points ignore both).
        let amb = SCNLight(); amb.type = .ambient; amb.intensity = 650
        let an = SCNNode(); an.light = amb; scene.rootNode.addChildNode(an)
        let key = SCNLight(); key.type = .directional; key.intensity = 750; key.temperature = 6500
        let kn = SCNNode(); kn.light = key; kn.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 5, 0)
        scene.rootNode.addChildNode(kn)

        let cam = SCNCamera()
        cam.fieldOfView = 52
        cam.zNear = 0.05; cam.zFar = 400
        cam.wantsHDR = true; cam.bloomIntensity = 0.1
        rig.camera.camera = cam
        rig.frameCamera(on: simd_float3(0, 0, 0))
        scene.rootNode.addChildNode(rig.camera)

        return (scene, rig)
    }

    /// The 2D occupancy map as a flat, textured quad anchored at the grid origin,
    /// spanning odom +x (SceneKit +x) and odom +y (SceneKit −z). Cell (0,0) =
    /// origin corner → texture top-left.
    static func occupancyNode(_ grid: MapGrid) -> SCNNode? {
        let w = grid.width, h = grid.height
        guard w > 0, h > 0, grid.cells.count >= w * h,
              let tex = occupancyTexture(grid) else { return nil }
        let ww = Float(w) * grid.resolution
        let hh = Float(h) * grid.resolution
        let verts: [SCNVector3] = [
            SCNVector3(0, 0, 0),      // cell (0,0)       UV (0,0)
            SCNVector3(ww, 0, 0),     // cell (W-1,0)     UV (1,0)
            SCNVector3(0, 0, -hh),    // cell (0,H-1)     UV (0,1)
            SCNVector3(ww, 0, -hh),   // cell (W-1,H-1)   UV (1,1)
        ]
        let uvs: [CGPoint] = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0),
                              CGPoint(x: 0, y: 1), CGPoint(x: 1, y: 1)]
        let idx: [Int32] = [0, 2, 3, 0, 3, 1]            // two upward-facing triangles
        let vSrc = SCNGeometrySource(vertices: verts)
        let tSrc = SCNGeometrySource(textureCoordinates: uvs)
        let elem = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        let geo = SCNGeometry(sources: [vSrc, tSrc], elements: [elem])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = tex
        m.diffuse.magnificationFilter = .nearest         // crisp cells, no blur
        m.diffuse.minificationFilter = .nearest
        m.isDoubleSided = true
        m.transparencyMode = .aOne                       // honour the alpha channel
        geo.materials = [m]
        let node = SCNNode(geometry: geo)
        // origin corner pose (SceneKit). +1 cm so the map floats just over the floor.
        node.simdPosition = simd_float3(grid.origin.x, grid.origin.y + 0.01, grid.origin.z)
        node.simdOrientation = simd_quatf(angle: grid.yaw, axis: simd_float3(0, 1, 0))
        node.name = "occupancy"
        return node
    }

    /// Rasterise the occupancy cells to an RGBA image: free → light, occupied →
    /// dark (both ~67% alpha so the 3D mesh shows through), unknown → transparent.
    /// Premultiplied alpha (CGContext requirement). NOTE: if the map appears
    /// vertically mirrored on device, reverse the row order here (one-line flip).
    private static func occupancyTexture(_ grid: MapGrid) -> UIImage? {
        let w = grid.width, h = grid.height
        let a = 170                                       // ~0.67 alpha
        var px = [UInt8](repeating: 0, count: w * h * 4)  // RGBA, premultiplied
        for r in 0..<h {
            for c in 0..<w {
                let val = Int(grid.cells[r * w + c])
                let o = (r * w + c) * 4
                if val == 255 { continue }                // unknown → all zero (clear)
                let g = 255 - min(val, 100) * 255 / 100   // free=255 light, occ=0 dark
                let pg = UInt8(g * a / 255)               // premultiply
                px[o] = pg; px[o + 1] = pg; px[o + 2] = pg; px[o + 3] = UInt8(a)
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs, bitmapInfo: info),
              let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }

    // The old `gridNode(half:step:)` lived here. Replaced by `GroundPlane` — see
    // that file's header for what was wrong with a bare square of lines.
}
