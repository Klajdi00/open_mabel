//
//  SensorLayers.swift
//  MabelTeleop
//
//  The RealityKit SENSOR WORLD around an articulated rig — shared by the
//  in-window true-3D Robot stage and the volumetric Robot space:
//
//    · Cam 3D — camera-projection planes with LIVE MJPEG textures at the true
//      mount/lens geometry (same rig json + math as the SceneKit CameraProjector)
//    · 2D occupancy grid + live map point cloud + accumulated map MESH
//      (odom frame, counter-translated so the robot stays centred)
//    · Laser — the planar LiDAR sweep as a point ring riding the base
//
//  Attach once to a RobotRigEntity; call start()/stop() around the hosting
//  view's lifetime. All data is PULLED (timers + map-store observers) — the
//  hosting view never observes the controller.
//

import RealityKit
import SceneKit      // LiveMapStore's MapGrid/MapMesh carry SCNVector3 fields
import SwiftUI
import QuartzCore

@MainActor
final class SensorLayers {

    private let world = Entity()            // odom-frame content, robot-centred
    private let gridHolder = Entity()
    private let cloudHolder = Entity()
    private let meshHolder = Entity()
    private let pathHolder = Entity()       // planned route + goal marker
    private let laserHolder = Entity()      // child of rig.robot (base frame)
    private var camGroups: [Entity] = []
    private var camGroupByStream: [String: Entity] = [:]
    private var camPlaneWidth: [String: Float] = [:]   // physical plane width (m)
    /// Per-stream lens geometry, so the image-size slider can re-solve the
    /// plane WITHOUT tearing the video streamers down and reconnecting.
    private var camPlaneGeom: [String: (aspect: Float, fyOverH: Float)] = [:]
    /// Image height of the projected planes, metres (Cam 3D ▸ image size).
    private var camProjHeight: Float = 0.3
    /// Render every Nth cloud point (Cloud ▸ density).
    private var cloudStride = 1
    private var camMaster = true
    private var camEnabled: Set<String> = ["main"]   // default: head cam only
    // Virtual floor disc under the robot (rides the base).
    private let floorHolder = Entity()
    private var floorBuilt = false

    private weak var controller: RobotController?
    private weak var rig: RobotRigEntity?
    private var observerID: UUID?
    private var streamers: [MJPEGTextureStreamer] = []
    private var camMaterialsBusy: [String: Bool] = [:]
    private var camPlanes: [String: ModelEntity] = [:]
    private var camsAttached = false
    private var lastLidarStamp: TimeInterval = -1
    private var tickTimer: Timer?
    // TF-frame triads (web control-center style): built lazily once the rig's
    // joints exist, parented to the articulation entities so they ride the FK.
    private var camsNextTry: TimeInterval = 0
    /// Video route the planes were attached to — a change (sim ↔ real robot
    /// switch) tears the streamers down so they re-attach to the NEW robot.
    private var camRoute = ""

    private var tfWanted = false
    private var tfBuilt = false
    private var tfNodes: [Entity] = []

    /// FOLLOW: keep the robot centred (the world counter-translates under it).
    /// OFF freezes the world where it is and lets the ROBOT roam the stage
    /// from its live odom pose.
    var followRobot = true {
        didSet {
            guard followRobot != oldValue else { return }
            roamOrigin = followRobot ? nil : world.position
            applyCentering()
        }
    }
    private var odomScene = SIMD3<Float>(0, 0, 0)   // robot pose, map/scene coords
    private var odomQuat: simd_quatf? = nil         // SLAM base orientation (scene frame)
    private var odomFreshAt: CFTimeInterval = 0     // last /map odom arrival
    private var roamOrigin: SIMD3<Float>? = nil     // frozen world offset (follow off)

    /// TRUE while a StagePlanner whole-body plan owns rig.robot.position (the
    /// previewed/executing base walk) — odom must not snap the robot back.
    /// The world/map layers keep following live odom regardless.
    var baseFrozen = false

    private func applyCentering() {
        guard let rig else { return }
        // THE ODOM PLANE IS THE GROUND, and the rig's origin is not.
        // `glb_to_rig.py` parks `base_link` at y = 0 — the chassis origin, with
        // the wheels 7.8 cm BELOW it — while everything in `world` (the SLAM
        // cloud, the mesh, the occupancy quad, the route, the tap-catcher and
        // the virtual floor) arrives in odom, whose z = 0 IS the floor. Leaving
        // the world container at y = 0 therefore floated every ground-referenced
        // overlay 7.8 cm above the wheels — the same class of defect the web
        // viewer's `run_frames` check exists to catch (`cloudFloorWorldY −
        // rigFeetWorldY` must be 0). Dropping the container by the measured
        // wheel-contact height makes odom z = 0 land exactly on the ground the
        // robot stands on, for every layer at once.
        let g = groundY()
        if followRobot {
            world.position = SIMD3(-odomScene.x, g, -odomScene.z)
            if !baseFrozen { rig.robot.position = .zero }
        } else {
            let frozen = roamOrigin ?? -odomScene
            world.position = SIMD3(frozen.x, g, frozen.z)
            if !baseFrozen { rig.robot.position = odomScene + SIMD3(frozen.x, 0, frozen.z) }
        }
        // iOS-map-view parity: while SLAM odom flows it owns the chassis
        // orientation too — the robot_state heading is odom-frame and reads
        // rotated against the map layers on the real robot.
        if let q = odomQuat {
            rig.baseOrientationExternal = true
            rig.robot.orientation = q
        }
    }

    private static let lidarHeight: Float = 0.18

    init(controller: RobotController, rig: RobotRigEntity) {
        self.controller = controller
        self.rig = rig
        rig.root.addChild(world)
        world.addChild(gridHolder)
        world.addChild(cloudHolder)
        world.addChild(meshHolder)
        world.addChild(pathHolder)
        rig.robot.addChild(laserHolder)
        // Tap-catcher GROUND: a near-invisible 30 m plane in the odom frame,
        // ALWAYS present — look at the floor + pinch places a nav goal even
        // with the 2D Grid layer off (the occupancy texture used to be the
        // only tap target). Horizontal at y≈0, so it never occludes gaze on
        // the robot's links above it.
        var mat = UnlitMaterial(color: UIColor(white: 1.0, alpha: 0.015))
        mat.blending = .transparent(opacity: 0.015)
        let ground = ModelEntity(mesh: .generatePlane(width: 30, depth: 30), materials: [mat])
        ground.name = "goal_floor"
        ground.position = SIMD3(0, 0.0005, 0)
        ground.components.set(CollisionComponent(shapes: [
            .generateBox(width: 30, height: 0.002, depth: 30)
        ]))
        ground.components.set(InputTargetComponent())
        world.addChild(ground)
        // The virtual floor belongs to the WORLD (odom), not to `rig.robot`.
        // Parented to the base it inherited the chassis' yaw and X/Z drift, so
        // the "ground" swung and slid with the robot. The web control center
        // makes the same call — its ground disc lives in the scene root, never
        // under the rig (`viewport3d.js:107`). The world container itself is
        // seated on the measured wheel contact by `applyCentering`, so this and
        // every other odom overlay share one ground plane.
        world.addChild(floorHolder)
    }

    func start() {
        guard let controller else { return }
        controller.liveMap.retainStream()
        observerID = controller.liveMap.addObserver(.init(
            mesh: { [weak self] m in self?.apply(mapMesh: m) },
            pose: { [weak self] p in
                guard let self else { return }
                self.odomScene = SIMD3(Float(p.position.x), 0, Float(p.position.z))
                self.odomQuat = p.orientation
                self.odomFreshAt = CACurrentMediaTime()
                self.applyCentering()
            },
            path: { [weak self] path in self?.apply(path: path) },
            grid: { [weak self] g in self?.apply(grid: g) },
            cloud: { [weak self] c in self?.apply(cloud: c) }
        ))
        let t = Timer(timeInterval: 1.0 / 20.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        tickTimer = t
    }

    func stop() {
        tickTimer?.invalidate(); tickTimer = nil
        if let id = observerID { controller?.liveMap.removeObserver(id) }
        observerID = nil
        controller?.liveMap.releaseStream()
        streamers.forEach { $0.stop() }
        streamers = []
        camGroups.forEach { $0.removeFromParent() }
        camGroups = []
        camGroupByStream = [:]
        camPlanes = [:]
        camsAttached = false
        tfNodes.forEach { $0.removeFromParent() }
        tfNodes = []
        tfBuilt = false
        gridPlane = nil
        world.removeFromParent()
        laserHolder.removeFromParent()
    }

    func setLayers(cam: Bool, grid: Bool, cloud: Bool, laser: Bool, mesh: Bool,
                   tf: Bool = false, floor: Bool = false) {
        camMaster = cam
        applyCamVisibility()
        gridHolder.isEnabled = grid
        cloudHolder.isEnabled = cloud
        meshHolder.isEnabled = mesh
        laserHolder.isEnabled = laser
        tfWanted = tf
        tfNodes.forEach { $0.isEnabled = tf }
        if floor, !floorBuilt { buildFloor() }
        floorHolder.isEnabled = floor
    }

    /// The HEAD camera's projected-image plane centre in rig-root space — the
    /// POV transition dives the stage to THIS TF so the image fills the view
    /// before the 2D video fades in (nil offline: no plane attached).
    func headCamAnchorLocal() -> SIMD3<Float>? {
        headCamFrameLocal()?.pos
    }

    /// Position, orientation AND physical width (rig-metres) of the head-image
    /// plane in rig-root space — the dive rotates the whole rig so the plane
    /// faces the viewer squarely and scales it to an EXACT on-screen width.
    func headCamFrameLocal() -> (pos: SIMD3<Float>, rot: simd_quatf, width: Float)? {
        guard let rig, let grp = camGroupByStream["main"],
              let plane = grp.children.first else { return nil }
        return (plane.position(relativeTo: rig.root),
                plane.orientation(relativeTo: rig.root),
                camPlaneWidth["main"] ?? 0.4)
    }

    /// Which camera streams project (the rail's Cam 3D ▸ per-camera toggles).
    func setCameraStreams(_ streams: Set<String>) {
        camEnabled = streams
        applyCamVisibility()
    }

    /// Image height of the projected planes, metres. Re-solves the existing
    /// planes IN PLACE — rebuilding them would tear down the MJPEG streamers
    /// and blank every projection for a second on each slider tick.
    func setCamProjSize(_ height: Float) {
        let h = max(0.05, min(2.0, height))
        guard abs(h - camProjHeight) > 1e-4 else { return }
        camProjHeight = h
        for (stream, plane) in camPlanes {
            guard let g = camPlaneGeom[stream] else { continue }
            camPlaneWidth[stream] = h * g.aspect
            plane.model?.mesh = .generatePlane(width: h * g.aspect, height: h)
            plane.position = SIMD3(0, 0, -h * g.fyOverH)
        }
    }

    /// Render every Nth cloud point. Applies to the NEXT frame (2 Hz), which is
    /// soon enough that the segment reads as instant.
    func setCloudStride(_ n: Int) {
        cloudStride = max(1, n)
    }

    private func applyCamVisibility() {
        for (stream, grp) in camGroupByStream {
            grp.isEnabled = camMaster && camEnabled.contains(stream)
        }
    }

    /// The rig's WHEEL-CONTACT plane in rig-root space.
    ///
    /// `glb_to_rig.py` strips the GLB's centering bake, so the exported rig
    /// parks **`base_link` at y = 0** — which is the chassis origin, NOT the
    /// ground. Measured over every PLY in `Resources/MabelRig` (each part
    /// transformed by its accumulated group matrix), the lowest geometry is
    /// the three wheels at **y = −0.0781 m**; the head tops out at +0.9825 m
    /// for a 1.0606 m overall height.
    ///
    /// The virtual floor used to sit at `+0.003` in the **base** frame, i.e.
    /// 8.1 cm ABOVE the wheel contacts — so it cut through the wheels and
    /// into the mobile base (operator report, 2026-07-19). Same class of bug
    /// as the web control center's ground disc, which is likewise world-frame
    /// and pushed just BELOW zero (`viewport3d.js:109`, `g.position.y =
    /// -0.004`) so it passes under the contact patches instead of through them.
    ///
    /// Runtime measurement refines this once the async rig load lands; the
    /// constant is the fallback for a floor enabled before the meshes arrive.
    private static let measuredGroundY: Float = -0.0781

    /// Wheel-contact height, measured from the loaded rig when possible.
    ///
    /// It MUST be measured over the rig's PLY parts (`rig.groundY`), never over
    /// `rig.robot.visualBounds`: that subtree also carries the force/guide
    /// arrows, and RealityKit's `visualBounds` counts DISABLED entities by
    /// default. The idle guide arrow is a 1 m cylinder centred on the robot
    /// origin, so the "wheel contact" came back at −0.50 m and the virtual
    /// floor sat ~42 cm BELOW the ground the robot stands on — the operator's
    /// "floor is much lower than the robot" report (2026-07-28). The [MABEL-FLOOR]
    /// check below did not catch it because it measured the same wrong bounds
    /// on both sides of the comparison; it now measures the geometry too.
    private func groundY() -> Float {
        if let cached = groundCached { return cached }
        guard let rig, let y = rig.groundY(relativeTo: rig.robot) else {
            return Self.measuredGroundY            // meshes not in yet — retry next tick
        }
        // Sanity envelope: the wheels are ~8 cm below base_link and the chassis
        // never leaves that band, so anything outside it is a measurement bug,
        // not a robot. Fall back rather than draw a floor in the wrong place.
        let g = (y > -0.30 && y < 0.10) ? y : Self.measuredGroundY
        groundCached = g
        return g
    }
    /// Measured once the meshes land; `nil` = still on the fallback constant.
    /// Relative to `rig.robot`, so it is pose-independent (the wheels do not
    /// move in the chassis frame) and a lift/arm pose can never shift the floor.
    private var groundCached: Float? = nil

    /// A soft, system-consistent virtual floor: a translucent disc of faint
    /// 0.5 m guides that dissolves radially into the room. Deliberately NOT
    /// MuJoCo's blue-grey checkerboard — that reads as a foreign object over
    /// visionOS glass. This conveys "ground" through a grounding shadow-like
    /// wash plus grid cadence, while staying quiet enough to sit under the
    /// robot without competing with it.
    private func buildFloor() {
        floorBuilt = true
        let S = 1024
        let span: Float = 8                      // metres across the plane
        let cells = 16                           // 16 × 0.5 m
        let cell = CGFloat(S) / CGFloat(cells)
        let fmt = UIGraphicsImageRendererFormat.default()
        fmt.opaque = false
        fmt.scale = 1
        let img = UIGraphicsImageRenderer(size: CGSize(width: S, height: S), format: fmt).image { ctx in
            let c = ctx.cgContext
            // A soft CONTACT-SHADOW wash — the way Apple grounds an object in
            // space. Deliberately dark rather than light: the viewer's room
            // shows through visionOS glass, and a white floor vanished against
            // a bright room. A gentle darkening reads as ground on a light
            // background, while the light grid lines below carry it on a dark
            // one, so the layer never disappears either way.
            c.setFillColor(UIColor(white: 0.0, alpha: 0.13).cgColor)
            c.fill(CGRect(x: 0, y: 0, width: S, height: S))
            // 0.5 m guides for scale. Minor lines hairline, majors (2 m) a
            // touch brighter, both far below the old checker's contrast.
            for i in 0...cells {
                let major = i % 4 == 0
                c.setStrokeColor(UIColor(white: 1.0, alpha: major ? 0.26 : 0.14).cgColor)
                c.setLineWidth(major ? 2.5 : 1.5)
                let p = CGFloat(i) * cell
                c.move(to: CGPoint(x: p, y: 0)); c.addLine(to: CGPoint(x: p, y: CGFloat(S)))
                c.move(to: CGPoint(x: 0, y: p)); c.addLine(to: CGPoint(x: CGFloat(S), y: p))
                c.strokePath()
            }
            // Radial alpha falloff — the plane fades out well before its edge,
            // so there is never a hard square rim floating in the room.
            c.setBlendMode(.destinationIn)
            let mid = CGFloat(S) / 2
            if let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceGray(),
                                     colors: [UIColor(white: 1, alpha: 1.0).cgColor,
                                              UIColor(white: 1, alpha: 0.85).cgColor,
                                              UIColor(white: 1, alpha: 0.0).cgColor] as CFArray,
                                     locations: [0.0, 0.42, 1.0]) {
                c.drawRadialGradient(grad,
                                     startCenter: CGPoint(x: mid, y: mid), startRadius: 0,
                                     endCenter: CGPoint(x: mid, y: mid), endRadius: mid,
                                     options: [])
            }
        }
        var mat = UnlitMaterial()
        // Unlit: the floor must read identically against a bright room and a
        // dark one, like the rest of the sensor layers.
        //
        // The texture is a nicety, NOT a prerequisite: `TextureResource(image:)`
        // failing used to `return` before the plane was ever built, so the
        // "ground" silently disappeared and, worse, the [MABEL-FLOOR] check
        // never ran — the one measurement that would have caught the height bug
        // reported nothing at all. Fall back to the flat wash instead.
        if let cg = img.cgImage,
           let tex = try? TextureResource(image: cg, options: .init(semantic: .color)) {
            mat.color = .init(tint: .white.withAlphaComponent(0.85), texture: .init(tex))
        } else {
            mat.color = .init(tint: UIColor(white: 0.0, alpha: 0.13))
        }
        mat.blending = .transparent(opacity: 0.85)
        mat.faceCulling = .none
        let plane = ModelEntity(mesh: .generatePlane(width: span, depth: span), materials: [mat])
        // The world container already sits ON the ground (see applyCentering),
        // so the disc is 1 mm proud of odom z = 0 — enough to clear z-fighting
        // with the wheels and with the occupancy quad (+0.005) above it.
        plane.position = SIMD3(0, 0.001, 0)
        floorHolder.addChild(plane)
        #if DEBUG
        logFloorGeometry(plane)
        #endif
    }

    #if DEBUG
    /// The Swift-side counterpart to the web's `run_frames` check
    /// (`ui_layout_playwright.py`), which asserts
    /// `|cloudFloorWorldY − rigFeetWorldY| < 1 cm`. There was no equivalent
    /// here, which is exactly why the floor could ride 8 cm high unnoticed —
    /// the defect is invisible from most camera angles and only a human
    /// staring at a screenshot ever caught it. Measure it instead.
    private func logFloorGeometry(_ plane: Entity) {
        guard let rig else { return }
        // WAIT for the geometry, don't sample once and give up. The rig parses
        // ~14 MB of PLY off the main actor, so a single 3 s deferral raced it
        // and the guard below returned — the check that exists to catch a
        // silent defect was itself silent. Poll instead, and say so if the
        // geometry never arrives.
        Task { @MainActor in
            var feet: Float? = nil
            for _ in 0..<40 {                       // ≤ 10 s
                try? await Task.sleep(for: .milliseconds(250))
                // GEOMETRY bounds — `robot.visualBounds` includes the idle guide
                // arrow, which is what let the floor ride 42 cm low while this
                // very check reported PASS.
                //
                // Measured in RIG-ROOT space, not scene space: the numbers are
                // then robot metres (directly comparable to the −0.0781 m wheel
                // contact) and do not depend on the rig being attached to a
                // Scene.
                if let f = rig.groundY(relativeTo: rig.root) { feet = f; break }
            }
            guard let feet else {
                print("[MABEL-FLOOR] no rig geometry after 10 s — check NOT run")
                return
            }
            let floorY = plane.position(relativeTo: rig.root).y
            let gap = (floorY - feet) * 100
            print("""
            [MABEL-FLOOR] feet=\(String(format: "%+.4f", feet)) m \
            floor=\(String(format: "%+.4f", floorY)) m \
            gap=\(String(format: "%+.2f", gap)) cm \
            \(abs(gap) < 1.0 ? "PASS" : "FAIL (floor must sit within 1 cm of wheel contact)")
            """)
        }
    }
    #endif

    /// RGB axis triads on the base + every articulated joint frame — the same
    /// TF view the web control center draws. Built once the async rig load has
    /// produced the joint entities (see tick()).
    private func buildTF() {
        guard let rig, !rig.joints.isEmpty else { return }
        tfBuilt = true
        func triad(_ scale: Float) -> Entity {
            let t = Entity()
            func axis(_ color: UIColor, _ rot: simd_quatf) -> Entity {
                let m = ModelEntity(mesh: .generateBox(size: SIMD3(scale, scale * 0.09, scale * 0.09)),
                                    materials: [UnlitMaterial(color: color)])
                m.position = SIMD3(scale / 2, 0, 0)
                let holder = Entity()
                holder.orientation = rot
                holder.addChild(m)
                return holder
            }
            t.addChild(axis(UIColor(red: 0.90, green: 0.28, blue: 0.30, alpha: 1),
                            simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))))                 // +x
            t.addChild(axis(UIColor(red: 0.19, green: 0.64, blue: 0.42, alpha: 1),
                            simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))))           // +y
            t.addChild(axis(UIColor(red: 0.30, green: 0.62, blue: 1.00, alpha: 1),
                            simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 1, 0))))          // +z
            let hub = ModelEntity(mesh: .generateSphere(radius: scale * 0.08),
                                  materials: [UnlitMaterial(color: .white)])
            t.addChild(hub)
            return t
        }
        let base = triad(0.18)
        rig.robot.addChild(base)
        tfNodes.append(base)
        let finger = ["thumb", "index", "middle", "ring", "pinky", "grip"]
        for (id, j) in rig.joints {
            let small = finger.contains { id.lowercased().contains($0) }
            let t = triad(small ? 0.035 : 0.10)
            j.node.addChild(t)
            tfNodes.append(t)
        }
        tfNodes.forEach { $0.isEnabled = tfWanted }
    }

    /// Swap a holder's content atomically: add the fresh entity, then drop the
    /// old ones — the render pipeline never sees an empty frame (no flicker).
    private func swapContent(of holder: Entity, with fresh: Entity?) {
        let old = Array(holder.children)
        if let fresh { holder.addChild(fresh) }
        old.forEach { $0.removeFromParent() }
    }

    private func tick() {
        guard let controller else { return }
        // The rig loads async, so the first frames run on the fallback ground
        // constant. Re-seat the odom plane the moment the real geometry exists.
        if groundCached == nil, rig?.parts.isEmpty == false {
            _ = groundY()
            applyCentering()
        }
        // Odom fallback: the bare Mac sim serves no /map odom — robot_state.base
        // x/y (50 Hz) keeps the world-frame clouds registered to the robot
        // (odom → scene: x = x, z = −y). /map odom OWNS the pose while it flows:
        // on the real robot SLAM odom is sparse (~0.3 Hz idle) and lives in the
        // MAP frame, so a short-fuse fallback to the odom-frame teleop pose kept
        // snapping the robot away from the occupancy grid. Only take over when
        // SLAM has never delivered, or has been silent long past its cadence.
        let slamAge = CACurrentMediaTime() - odomFreshAt
        if odomFreshAt == 0 || slamAge > 5.0 {
            if odomQuat != nil {         // SLAM died: heading back to robot_state
                odomQuat = nil
                rig?.baseOrientationExternal = false
            }
            let lp = controller.livePose
            let p = SIMD3(Float(lp.baseX), 0, Float(-lp.baseY))
            if simd_distance(p, odomScene) > 1e-4 {
                odomScene = p
                applyCentering()
            }
        }
        if let scan = controller.lidarScan, scan.stamp != lastLidarStamp {
            lastLidarStamp = scan.stamp
            apply(lidar: scan)
        }
        if CACurrentMediaTime() > camsNextTry {
            camsNextTry = CACurrentMediaTime() + 1.0   // rate-limit the URL probe
            let route = controller.videoThumbURL("main")?.absoluteString ?? "offline"
            if camsAttached, route != camRoute {
                detachCameras()                        // robot switched — drop OLD streams
            }
            if !camsAttached { attachCameras() }
        }
        if tfWanted, !tfBuilt { buildTF() }
    }

    // ── Cam 3D: projection planes with live MJPEG at the true mounts ─────────

    private func detachCameras() {
        streamers.forEach { $0.stop() }
        streamers = []
        camGroups.forEach { $0.removeFromParent() }
        camGroups = []
        camGroupByStream = [:]
        camPlanes = [:]
        camsAttached = false
    }

    private func attachCameras() {
        guard let controller, let rig else { return }
        camRoute = controller.videoThumbURL("main")?.absoluteString ?? "offline"
        let specs = CameraRigFile.load()
        guard !specs.isEmpty else { camsAttached = true; return }
        var attachedAny = false
        for (stream, camName) in CameraProjector.streamToCam {
            guard let c = specs.first(where: { $0.name == camName }),
                  c.mount_pos.count == 3, c.mount_quat.count == 4,
                  let nodeName = CameraProjector.mountToNode[c.mount_body],
                  let mount = rig.robot.findEntity(named: nodeName) else { continue }
            // No stream = no plane. Offline the URL builders return nil, so the
            // viewer never shows empty black slabs; tick() retries after connect.
            guard let url = controller.videoThumbURL(stream) else { continue }
            attachedAny = true

            let grp = Entity()
            grp.name = "camviz_\(camName)"
            grp.position = SIMD3(Float(c.mount_pos[0]), Float(c.mount_pos[1]), Float(c.mount_pos[2]))
            let q = c.mount_quat
            grp.orientation = simd_quatf(ix: Float(q[1]), iy: Float(q[2]),
                                         iz: Float(q[3]), r: Float(q[0]))

            // Rectified pinhole plane at the depth where the requested image
            // height subtends the true vertical FOV (CameraProjector math).
            let H = camProjHeight
            let aspect = Float(c.w) / Float(max(1, c.h))
            camPlaneWidth[stream] = H * aspect
            camPlaneGeom[stream] = (aspect: aspect, fyOverH: Float(c.fy) / Float(max(1, c.h)))
            let D = H * camPlaneGeom[stream]!.fyOverH
            var mat = UnlitMaterial(color: UIColor(white: 0.06, alpha: 0.9))
            mat.faceCulling = .none
            let plane = ModelEntity(mesh: .generatePlane(width: H * aspect, height: H),
                                    materials: [mat])
            plane.position = SIMD3(0, 0, -D)
            grp.addChild(plane)
            camPlanes[stream] = plane

            grp.isEnabled = camMaster && camEnabled.contains(stream)
            mount.addChild(grp)
            camGroups.append(grp)
            camGroupByStream[stream] = grp

            camMaterialsBusy[stream] = false
            let streamer = MJPEGTextureStreamer(url: url, label: "layers_\(stream)") { [weak self] cg in
                Task { @MainActor [weak self] in self?.applyFrame(cg, stream: stream) }
            }
            streamer.start()
            streamers.append(streamer)
        }
        camsAttached = attachedAny   // keep retrying until the rig + link are up
    }

    private func applyFrame(_ image: CGImage, stream: String) {
        guard camMaterialsBusy[stream] == false, camPlanes[stream] != nil else { return }
        camMaterialsBusy[stream] = true
        Task { @MainActor [weak self] in
            defer { self?.camMaterialsBusy[stream] = false }
            guard let self, let plane = self.camPlanes[stream] else { return }
            let opts = TextureResource.CreateOptions(semantic: .color, mipmapsMode: .none)
            guard let tex = try? await TextureResource(image: image, options: opts) else { return }
            var mat = UnlitMaterial()
            mat.color = .init(texture: .init(tex))
            mat.faceCulling = .none
            plane.model?.materials = [mat]
        }
    }

    // ── Laser: LiDAR sweep ring in the base frame ────────────────────────────

    private func apply(lidar scan: RobotState.LidarScan) {
        guard !scan.ranges.isEmpty else { swapContent(of: laserHolder, with: nil); return }
        var pts: [SIMD3<Float>] = []
        var cols: [SIMD3<Float>] = []
        let n = scan.ranges.count
        var i = 0
        while i < n {
            let r = scan.ranges[i]
            defer { i += 2 }
            guard r < scan.rangeMax - 1e-3 else { continue }
            let a = scan.angleMin + Double(i) * scan.angleInc
            // Base frame (x fwd, y left) → rig space (nose = −X): (−x, h, y).
            pts.append(SIMD3(-Float(r * cos(a)), Self.lidarHeight, Float(r * sin(a))))
            let t = Float(min(1, r / scan.rangeMax))
            cols.append(SIMD3(0.95 - 0.65 * t, 0.45 + 0.17 * t, 0.20 + 0.80 * t))
        }
        guard !pts.isEmpty else { swapContent(of: laserHolder, with: nil); return }
        swapContent(of: laserHolder,
                    with: PointCloudEntity.coloredCloud(points: pts, colors: cols, size: 0.012))
    }

    // ── Map layers (odom frame, robot-centred) ───────────────────────────────

    /// ONE persistent plane entity, updated IN PLACE. Replacing the entity per
    /// grid frame (the old atomic swap) destroyed the gaze/tap target ~3×/s —
    /// pinches resolved against a dead entity and goal taps NEVER landed.
    private var gridPlane: ModelEntity? = nil

    private func apply(grid g: MapGrid) {
        guard g.count > 0 else {
            gridPlane?.removeFromParent(); gridPlane = nil
            return
        }
        #if DEBUG
        NSLog("[MABEL-GRID] origin=(%.2f, %.2f, %.2f) yaw=%.2f span=%.1fx%.1f odomScene=(%.2f, %.2f) world=(%.2f, %.2f) robot=(%.2f, %.2f)",
              g.origin.x, g.origin.y, g.origin.z, g.yaw,
              Float(g.width) * g.resolution, Float(g.height) * g.resolution,
              odomScene.x, odomScene.z, world.position.x, world.position.z,
              rig?.robot.position.x ?? -999, rig?.robot.position.z ?? -999)
        #endif
        guard let fresh = PointCloudEntity.occupancyPlane(
            width: g.width, height: g.height, resolution: g.resolution,
            origin: SIMD3(g.origin.x, g.origin.y, g.origin.z),
            yaw: g.yaw, cells: g.cells) as? ModelEntity else { return }
        if let plane = gridPlane {
            plane.model = fresh.model
            plane.transform = fresh.transform
            plane.generateCollisionShapes(recursive: false)
        } else {
            // Tap-to-goal: the occupancy plane is the tappable floor.
            fresh.name = "goal_floor"
            fresh.generateCollisionShapes(recursive: false)
            fresh.components.set(InputTargetComponent())
            gridHolder.addChild(fresh)
            gridPlane = fresh
        }
    }

    /// Planned route (PATH frames) + a small goal pillar at its end.
    private func apply(path: MapPath) {
        let pts = path.points.map { SIMD3(Float($0.x), Float($0.y) + 0.03, Float($0.z)) }
        guard pts.count >= 2 else { swapContent(of: pathHolder, with: nil); return }
        let fresh = Entity()
        PointCloudEntity.renderLivePath(pts, into: fresh)
        let pillar = ModelEntity(mesh: .generateCylinder(height: 0.5, radius: 0.03),
                                 materials: [UnlitMaterial(color: UIColor(red: 0.2, green: 0.82, blue: 0.35, alpha: 0.95))])
        pillar.position = SIMD3(pts.last!.x, 0.25, pts.last!.z)
        fresh.addChild(pillar)
        swapContent(of: pathHolder, with: fresh)
    }

    /// Scene-space point → odom coordinates (the world container's local frame):
    /// used by the stage's tap-to-goal to send `nav_goal` in the map frame.
    func odomPoint(fromScene p: SIMD3<Float>) -> SIMD3<Float> {
        world.convert(position: p, from: nil)
    }

    private func apply(cloud c: MapMesh) {
        guard c.count > 0 else { swapContent(of: cloudHolder, with: nil); return }
        let n = cloudStride
        let idx = n > 1 ? Array(stride(from: 0, to: c.points.count, by: n)) : nil
        let pts = (idx?.map { c.points[$0] } ?? c.points)
            .map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) }
        let cols = (idx.map { i in i.compactMap { $0 < c.colors.count ? c.colors[$0] : nil } }
                    ?? c.colors)
            .map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) }
        // POINTS, not shards: tiny splats read as dots (triangles are reserved
        // for the SLAM mesh layer).
        swapContent(of: cloudHolder,
                    with: PointCloudEntity.coloredCloud(points: pts, colors: cols, size: 0.008))
    }

    /// The accumulated map: triangles when the snapshot carries indices, else a
    /// point cloud (same fallback as the SceneKit viewer).
    private func apply(mapMesh m: MapMesh) {
        guard m.count > 0 else { swapContent(of: meshHolder, with: nil); return }
        let pts = m.points.map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) }
        let cols = m.colors.map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) }
        if m.isMesh, !m.indices.isEmpty {
            var desc = MeshDescriptor()
            desc.positions = MeshBuffer(pts)
            desc.primitives = .triangles(m.indices.map { UInt32($0) })
            if let mesh = try? MeshResource.generate(from: [desc]) {
                // Average tint — per-vertex colour needs a custom material; the
                // average keeps the surface readable and cheap.
                let avg = cols.reduce(SIMD3<Float>.zero, +) / Float(max(1, cols.count))
                var mat = PhysicallyBasedMaterial()
                mat.baseColor = .init(tint: UIColor(red: CGFloat(avg.x), green: CGFloat(avg.y),
                                                    blue: CGFloat(avg.z), alpha: 1))
                mat.roughness = 0.9
                mat.faceCulling = .none
                swapContent(of: meshHolder, with: ModelEntity(mesh: mesh, materials: [mat]))
                return
            }
        }
        // Vertices-only snapshot → dots (points), never triangle shards.
        swapContent(of: meshHolder,
                    with: PointCloudEntity.coloredCloud(points: pts, colors: cols, size: 0.008))
    }
}
