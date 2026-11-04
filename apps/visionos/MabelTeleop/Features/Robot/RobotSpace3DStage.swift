//
//  RobotSpace3DStage.swift
//  MabelTeleop
//
//  The TRUE-3D third-person Robot space — RealityKit content with real depth
//  inside the plain window (the original Vision Pro feel), replacing the flat
//  SceneKit projection for 3rd person. Composed from the proven pieces:
//
//    · RobotStageView (v1) — the articulated live robot with gaze-highlight,
//      pinch-drag guiding/pushing, orbit + zoom, auto window fit
//    · SensorLayers — Cam 3D projection planes (live MJPEG), occupancy grid,
//      map cloud + mesh, laser ring (all robot-centred)
//
//  1st person keeps the SceneKit path (camera flight + FPV crossfade); this
//  stage owns 3rd person only. The 2D minimap keeps working through its own
//  map-store observer.
//

import SwiftUI
import RealityKit
import SceneKit      // MapGrid/MapPath types
import Combine

struct RobotSpace3DStage: View {
    let controller: RobotController
    @ObservedObject var navState: NavMapState
    var camProjections: Bool
    /// Which streams project (per-camera Layers toggles).
    var camStreams: Set<String> = ["main"]
    /// Cam 3D image height in metres, and cloud decimation — the rail's
    /// Cam 3D ▸ / Cloud ▸ sub-options. Both were SceneKit-only until now, so
    /// on the default (RealityKit) stage they were controls that did nothing.
    var camProjSize: Double = 0.3
    var cloudStride: Int = 1
    var floorOn: Bool = false
    var meshOn: Bool
    var cloudOn: Bool
    var gridOn: Bool
    var laserOn: Bool
    var tfOn: Bool = false
    var recenterToken: Int
    var fitToken: Int
    /// Keep the robot centred (world moves) vs let it roam the stage.
    var follow: Bool = true
    /// Content-framing preset: since the user's head IS the camera, "moving the
    /// camera" means re-framing the CONTENT — scale + recentre the rig so the
    /// requested subject fills the stage.
    enum StagePreset: Equatable {
        case iso          // whole robot, isometric (Task space / default)
        case isoFar       // wide isometric, ~5 m of scene context (Drive space)
        case module(ModuleFocus)   // zoom to one body part (Joint space rail)
        case head                  // dive INTO the head — the 3rd→1st lead-in
        case headImage             // fly to the head-cam IMAGE plane (fills the view)
        case returnFromHead        // volumetric pull-out back to the pre-dive pose
    }
    var preset: StagePreset = .iso
    var presetToken: Int = 0
    /// TRUE = apply the preset with NO animation (parking the hidden stage at
    /// the head-image pose before the reverse transition reveals it).
    var presetInstant: Bool = false
    /// Window size in points (RobotSpaceView's geometry) — the head-image dive
    /// solves its zoom from this so the image ALWAYS lands at the exact
    /// 1st-person viewport width (visionOS windows: ~1360 pt per metre).
    var viewSize: CGSize = .zero
    /// Points of `viewSize` covered by the floating chrome (top control rows,
    /// bottom pill rows). Module framing fits INSIDE what is left, and centres
    /// there — the window's geometric centre is under the top bar.
    var safeInsets = EdgeInsets()
    /// Tap-to-goal: odom-frame goal (x, y metres, yaw rad), like the SceneKit
    /// viewer's floor tap. nil = view-only.
    var onGoal: ((Double, Double, Double) -> Void)? = nil
    /// PHYSICS VIEW: the shared MuJoCo spherical camera (az/el/dist/lookat)
    /// that also drives the SCC's server-side render. While `physicsCam.active`
    /// the stage re-frames the RIG every tick to show exactly that camera's
    /// view (rotation from az/el, zoom from dist at MuJoCo's 45° fovy) — the
    /// engage runs the iOS cinematic arc (+120° sweep, settle to the classic
    /// three-quarter view over 4.5 s), then drifts ~4°/s until the overlay
    /// crossfades, then only TRACKS (overlay gestures own the values).
    var physicsCam: PhysicsCam? = nil
    /// Bumped when the physics view ENGAGES (captures the start pose).
    var physicsToken: Int = 0
    /// Task · Wrist green-ball PLANNER on the stage (StagePlanner): TRUE arms
    /// the handles/rings; drags stage a plan, Execute streams it. The stage —
    /// not the SceneKit takeover — is the Task viewer (operator 2026-07-18).
    var planMode: Bool = false
    var planRegion: ControlRegion = .arm
    var executePlanToken: Int = 0
    var cancelPlanToken: Int = 0
    /// Headless harness: bump to stage a canned plan (debugPlanExecute).
    var debugStageToken: Int = 0
    var debugStageSide: String = "left"
    var onPlanChange: ((ArmPlanPreview?) -> Void)? = nil
    var onJointsChange: (([String: Double]) -> Void)? = nil
    var onBaseGoal: ((Double, Double, Double) -> Void)? = nil

    @StateObject private var stage: RobotStage
    @State private var layers: SensorLayers? = nil
    @State private var planner: StagePlanner? = nil
    @State private var navObserver: UUID? = nil
    @State private var lastRecenter = 0
    @State private var lastFit = 0
    @State private var lastPreset = 0
    @State private var lastExecute = 0
    @State private var lastCancel = 0
    @State private var lastDebugStage = 0
    /// The stage's FITTED transform (RobotStageView sizes the rig to the real
    /// window — ~1360 pt/metre). All presets derive from THIS, never from the
    /// volume-space homeSeat/baseScale (using those teleported the rig a metre
    /// below the window: "Start Teleop made the viewer disappear").
    @State private var homeTransform: Transform? = nil
    /// Framing the operator was in before the head-image dive — the reverse
    /// transition pulls back to EXACTLY this pose ("the original camera").
    @State private var preDive: Transform? = nil

    init(controller: RobotController, navState: NavMapState,
         camProjections: Bool, camStreams: Set<String> = ["main"],
         camProjSize: Double = 0.3, cloudStride: Int = 1, floorOn: Bool = false,
         meshOn: Bool, cloudOn: Bool, gridOn: Bool, laserOn: Bool,
         tfOn: Bool = false,
         recenterToken: Int, fitToken: Int,
         follow: Bool = true,
         preset: StagePreset = .iso, presetToken: Int = 0, presetInstant: Bool = false,
         viewSize: CGSize = .zero, safeInsets: EdgeInsets = EdgeInsets(),
         onGoal: ((Double, Double, Double) -> Void)? = nil,
         physicsCam: PhysicsCam? = nil, physicsToken: Int = 0,
         planMode: Bool = false, planRegion: ControlRegion = .arm,
         executePlanToken: Int = 0, cancelPlanToken: Int = 0,
         debugStageToken: Int = 0, debugStageSide: String = "left",
         onPlanChange: ((ArmPlanPreview?) -> Void)? = nil,
         onJointsChange: (([String: Double]) -> Void)? = nil,
         onBaseGoal: ((Double, Double, Double) -> Void)? = nil) {
        self.controller = controller
        self.navState = navState
        self.camProjections = camProjections
        self.camStreams = camStreams
        self.camProjSize = camProjSize
        self.cloudStride = cloudStride
        self.floorOn = floorOn
        self.meshOn = meshOn
        self.cloudOn = cloudOn
        self.gridOn = gridOn
        self.laserOn = laserOn
        self.tfOn = tfOn
        self.recenterToken = recenterToken
        self.fitToken = fitToken
        self.follow = follow
        self.preset = preset
        self.presetToken = presetToken
        self.presetInstant = presetInstant
        self.viewSize = viewSize
        self.safeInsets = safeInsets
        self.onGoal = onGoal
        self.physicsCam = physicsCam
        self.physicsToken = physicsToken
        self.planMode = planMode
        self.planRegion = planRegion
        self.executePlanToken = executePlanToken
        self.cancelPlanToken = cancelPlanToken
        self.debugStageToken = debugStageToken
        self.debugStageSide = debugStageSide
        self.onPlanChange = onPlanChange
        self.onJointsChange = onJointsChange
        self.onBaseGoal = onBaseGoal
        _stage = StateObject(wrappedValue: RobotStage(scale: 0.4, interactive: true))
    }

    var body: some View {
        // Split from `stageCore`: the single modifier chain outgrew the type
        // checker ("unable to type-check this expression in reasonable time").
        // NOTE: keep new modifiers on THIS chain, not on `stageCore` — that one
        // is already at the type checker's limit ("unable to type-check this
        // expression in reasonable time" the moment two more land on it).
        stageCore
            .onChange(of: camProjSize) { _, v in layers?.setCamProjSize(Float(v)) }
            .onChange(of: cloudStride) { _, v in layers?.setCloudStride(v) }
            .onChange(of: planMode) { _, on in planner?.setActive(on) }
            .onChange(of: planRegion) { _, r in planner?.planRegion = r }
            .onChange(of: executePlanToken) { _, t in
                guard t != lastExecute else { return }
                lastExecute = t
                planner?.executePlan()
            }
            .onChange(of: cancelPlanToken) { _, t in
                guard t != lastCancel else { return }
                lastCancel = t
                planner?.cancelPlan()
            }
            .onChange(of: debugStageToken) { _, t in
                guard t != lastDebugStage, t != 0 else { return }
                lastDebugStage = t
                stageDebugPlan()
            }
    }

    /// The Task-space planner: create once, re-wire to the fresh SensorLayers
    /// on every appear (layers are torn down in onDisappear).
    private func wirePlanner(to l: SensorLayers) {
        let pl = planner ?? StagePlanner(rig: stage.rig)
        pl.joints = { [weak controller] in controller?.liveModelJoints ?? [:] }
        pl.heading = { [weak controller] in controller?.state.heading ?? 0 }
        pl.odomFromScene = { [weak l] p in l?.odomPoint(fromScene: p) ?? p }
        pl.onBaseFreeze = { [weak l] on in l?.baseFrozen = on }
        pl.onPlanChange = onPlanChange
        pl.onJointsChange = onJointsChange
        pl.onBaseGoal = onBaseGoal
        pl.planRegion = planRegion
        planner = pl
        stage.manip = pl        // the CLASS-side tick reference (see RobotStage.manip)
        pl.setActive(planMode)
    }

    /// debugPlanExecute harness: stage a canned plan OFF the update pass
    /// (gotcha #19 — staging ends in onPlanChange, which writes host @State).
    private func stageDebugPlan() {
        let side = debugStageSide
        DispatchQueue.main.async { [planner] in
            planner?.stagePlanForTest(side: side,
                                      deltas: ["\(side)_arm_1": 0.35,
                                               "\(side)_arm_2": 0.30,
                                               "\(side)_arm_4": 0.40])
        }
    }

    private var stageCore: some View {
        RobotStageView(stage: stage, interaction: .links, manip: planner, floorRaise: 0.13)
            // Tap the floor → nav goal (odom frame), same contract as the
            // SceneKit viewer's tap; heading faces the goal from the robot.
            // simultaneousGesture: the stage's own link-drag gestures must not
            // preempt the tap (a pinch is a zero-length drag to them).
            .simultaneousGesture(SpatialTapGesture().targetedToAnyEntity().onEnded { v in
                guard let onGoal, v.entity.name == "goal_floor", let layers else { return }
                let scenePos = v.convert(v.location3D, from: .local, to: .scene)
                let odom = layers.odomPoint(fromScene: SIMD3(Float(scenePos.x), Float(scenePos.y), Float(scenePos.z)))
                let gx = Double(odom.x), gy = Double(-odom.z)
                let robot = navState.robot
                let yaw = atan2(gy - (-Double(robot.z)), gx - Double(robot.x))
                Haptics.rigid()
                onGoal(gx, gy, yaw)
            })
            .onAppear {
                let l = SensorLayers(controller: controller, rig: stage.rig)
                l.start()
                l.setCameraStreams(camStreams)
                l.setCamProjSize(Float(camProjSize))
                l.setCloudStride(cloudStride)
                l.setLayers(cam: camProjections, grid: gridOn, cloud: cloudOn,
                            laser: laserOn, mesh: meshOn, tf: tfOn, floor: floorOn)
                l.followRobot = follow
                layers = l
                wirePlanner(to: l)
                navState.beginLive()
                var poseTick = 0
                navObserver = controller.liveMap.addObserver(.init(
                    mesh: { [weak navState] m in navState?.ingest(mesh: m) },
                    pose: { [weak navState] p in
                        poseTick += 1
                        guard poseTick % 3 == 0 else { return }
                        navState?.setRobot(x: Double(p.position.x), z: Double(p.position.z),
                                           yaw: Double(p.yaw))
                    },
                    path: { [weak navState] path in
                        navState?.setLivePath(path.points.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.z)) })
                    },
                    grid: { [weak navState] g in navState?.ingest(grid: g) }
                ))
            }
            .onDisappear {
                layers?.stop(); layers = nil
                if let id = navObserver { controller.liveMap.removeObserver(id) }
                navObserver = nil
            }
            .onChange(of: camProjections) { _, v in applyLayers() }
            .onChange(of: meshOn) { _, _ in applyLayers() }
            .onChange(of: cloudOn) { _, _ in applyLayers() }
            .onChange(of: gridOn) { _, _ in applyLayers() }
            .onChange(of: laserOn) { _, _ in applyLayers() }
            .onChange(of: tfOn) { _, _ in applyLayers() }
            .onChange(of: floorOn) { _, _ in applyLayers() }
            .onChange(of: camStreams) { _, v in layers?.setCameraStreams(v) }
            .onChange(of: follow) { _, f in layers?.followRobot = f }
            .onChange(of: presetToken) { _, t in
                guard t != lastPreset else { return }
                lastPreset = t
                apply(preset: preset)
            }
            .onChange(of: stage.loaded) { _, ok in
                // The planner's handles need rig joints — arm it (again) once
                // the async rig load lands.
                if ok, planMode { planner?.setActive(true) }
                // A preset present at MOUNT (returning from 1st person rebuilds
                // this view, so onChange(presetToken) never fires) or requested
                // before the async rig load — apply once loaded.
                #if DEBUG
                if ok { runModuleTourIfRequested() }
                #endif
                guard ok, presetToken != 0 else { return }
                lastPreset = presetToken
                apply(preset: preset)
            }
            .onChange(of: recenterToken) { _, t in
                guard t != lastRecenter else { return }
                lastRecenter = t
                frame(zoom: 1.0, focusLocal: nil)
            }
            .onChange(of: fitToken) { _, t in
                guard t != lastFit else { return }
                lastFit = t
                frame(zoom: 0.62, focusLocal: nil)   // pull back for map context
            }
            .onChange(of: physicsToken) { _, t in
                guard t != lastPhysics else { return }
                lastPhysics = t
                engagePhysics()
            }
            .onReceive(physTick) { _ in
                guard let cam = physicsCam, cam.active else { return }
                stepPhysics(cam)
            }
    }

    // ── Physics view (the MuJoCo takeover's stage side) ───────────────────────

    @State private var lastPhysics = 0
    @State private var physT0 = Date.distantPast
    @State private var physFrom: (az: Double, el: Double, dist: Double) = (150, -20, 3.2)
    /// @State so uiTick re-renders can't recreate it before it fires.
    @State private var physTick = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    /// POV-flight-like engage (operator feedback: the 4.5 s cinematic read as
    /// "nothing happened" — the swap should feel like the 1st↔3rd transition).
    private static let physDuration: TimeInterval = 1.6
    /// MuJoCo's default free-camera vertical FOV — the server render's lens.
    private static let mjFovyDeg = 45.0
    private static let ptPerM: Double = 1360

    /// Engage: seed the shared spherical camera from the CURRENT rig framing
    /// (exact inversion of `applyPhysics`, so the arc starts seamlessly), then
    /// the tick evolves it. Resuming (already faded, view rebuilt) locks
    /// straight onto the shared pose instead.
    private func engagePhysics() {
        guard let cam = physicsCam, stage.loaded else { return }
        let rig = stage.rig
        cam.lookatScn = rig.robot.position(relativeTo: rig.root) + SIMD3<Float>(0, 0.55, 0)
        if homeTransform == nil { homeTransform = rig.root.transform }
        guard let home = homeTransform else { return }
        let t = rig.root.transform
        let s0 = Double(home.scale.x)
        let zoom = Double(t.scale.x) / max(1e-6, s0)
        // Current view direction (rig-local vector that faces the viewer).
        let dirLocal = t.rotation.inverse.act(SIMD3<Float>(0, 0, 1))
        let fs = -simd_normalize(dirLocal)                        // camera → lookat
        let fmj = SIMD3<Double>(Double(fs.x), Double(-fs.z), Double(fs.y))
        let el = asin(max(-1, min(1, fmj.z))) * 180 / .pi
        let az = atan2(fmj.y, fmj.x) * 180 / .pi
        // Invert the zoom↔distance relation (45° fovy at the lookat plane).
        let viewH = Double(viewSize.height > 0 ? viewSize.height : 800)
        let halfTan = tan(Self.mjFovyDeg * .pi / 360)
        let d = (viewH / Self.ptPerM) / max(1e-6, s0 * zoom * 2 * halfTan)
        let resuming = cam.faded
        if !resuming {
            cam.az = az; cam.el = el; cam.dist = max(1.2, d)
            cam.arcDone = false
        }
        physFrom = (cam.az, cam.el, cam.dist)
        physT0 = resuming ? Date(timeIntervalSinceNow: -Self.physDuration) : Date()
        if resuming { applyPhysics(cam) }
    }

    /// Quintic smootherstep: zero velocity AND acceleration at both ends.
    private static func smootherstep(_ s: Double) -> Double {
        let x = min(max(s, 0), 1)
        return x * x * x * (x * (6 * x - 15) + 10)
    }

    private func stepPhysics(_ cam: PhysicsCam) {
        guard stage.loaded else { return }
        let t = Date().timeIntervalSince(physT0)
        if !cam.faded {
            if t < Self.physDuration {
                // Short engage sweep (+45° azimuth), settle to the classic
                // MuJoCo three-quarter view — one eased motion, POV-flight feel.
                let e = Self.smootherstep(t / Self.physDuration)
                cam.az = physFrom.az + 45 * e
                cam.el = physFrom.el + (-18 - physFrom.el) * e
                cam.dist = physFrom.dist + (min(4.2, max(2.8, physFrom.dist)) - physFrom.dist) * e
            } else {
                cam.arcDone = true
                cam.az += 4.0 / 60.0        // slow drift (~4°/s) doubles as loading
            }
        }
        // Track the robot if it drives away (lookat follows the chest).
        cam.lookatScn = stage.rig.robot.position(relativeTo: stage.rig.root) + SIMD3<Float>(0, 0.55, 0)
        applyPhysics(cam)
    }

    /// Re-frame the RIG so the window shows exactly the MuJoCo camera's view:
    /// rotation puts the camera direction on stage +Z, zoom matches the 45°
    /// fovy vertical field at the lookat distance, and the LOOKAT lands at the
    /// TRUE window centre (the server render centres its lookat — frame()'s
    /// home-fit anchor sits lower, which read as a vertical jump at the fade).
    /// Centre + scale match — the same contract as the head-image dive.
    private func applyPhysics(_ cam: PhysicsCam) {
        guard stage.loaded else { return }
        if homeTransform == nil { homeTransform = stage.rig.root.transform }
        guard let home = homeTransform else { return }
        let azr = cam.az * .pi / 180, elr = cam.el * .pi / 180
        let fmj = SIMD3<Double>(cos(elr) * cos(azr), cos(elr) * sin(azr), sin(elr))
        let fs = SIMD3<Float>(Float(fmj.x), Float(fmj.z), Float(-fmj.y))   // mj→rig (y-up)
        let dir = -fs                                                       // lookat → camera
        let rot = Self.lookRotation(dir)
        let s0 = Double(home.scale.x)
        let viewH = Double(viewSize.height > 0 ? viewSize.height : 800)
        let halfTan = tan(Self.mjFovyDeg * .pi / 360)
        let zoom = Float((viewH / Self.ptPerM) / max(1e-6, s0 * 2 * cam.dist * halfTan))
        // Land the lookat at window centre (x=0, y=0) at the fitted depth.
        let s0f = home.scale.x
        let mid = SIMD3<Float>(0, 0.8, 0)
        let centre = home.translation + home.rotation.act(mid * s0f)
        let landing = SIMD3<Float>(0, 0, centre.z)
        var t = home
        t.scale = SIMD3(repeating: s0f * zoom)
        t.rotation = rot
        t.translation = landing - rot.act(cam.lookatScn * (s0f * zoom))
        stage.rig.root.transform = t
    }

    private func applyLayers() {
        layers?.setLayers(cam: camProjections, grid: gridOn, cloud: cloudOn,
                          laser: laserOn, mesh: meshOn, tf: tfOn, floor: floorOn)
    }

    // ── Content framing (the stage's "camera flights") ───────────────────────

    private func apply(preset: StagePreset) {
        // Capture the fitted home BEFORE anything reads it. `frame` captures it
        // too, but lazily — and `moduleVerticalOffset()` is evaluated as an
        // ARGUMENT to `frame`, so on the very first flight after launch it saw
        // `homeTransform == nil`, returned 0, and the module landed 120 pt low
        // (measured: the tour's first module was the only one to report FAIL).
        if homeTransform == nil, stage.loaded { homeTransform = stage.rig.root.transform }
        let isoTurn = simd_quatf(angle: -0.55, axis: SIMD3(0, 1, 0))
        switch preset {
        case .iso:
            frame(zoom: 1.0, focusLocal: nil, extraYaw: isoTurn)
        case .isoFar:
            // Wide shot: the robot at ~1/3 size reads as ~5 m of standoff, with
            // the map/cloud context around it.
            frame(zoom: 0.35, focusLocal: nil, extraYaw: isoTurn)
        case .module(let m):
            guard let c = moduleCenterLocal(m) else { return }
            if let f = moduleFraming(m, center: c) {
                // Per-module view direction (palm-on hands, front-on lift, …)
                // ported from iOS v2's flyToModule — see moduleFraming.
                frame(zoom: f.zoom, focusLocal: c, absoluteRotation: f.rot,
                      verticalOffset: moduleVerticalOffset())
                #if DEBUG
                verifyFraming(m)
                #endif
            } else {
                frame(zoom: 2.1, focusLocal: c,
                      extraYaw: simd_quatf(angle: m.side == "left" ? 0.45 : -0.45,
                                           axis: SIMD3(0, 1, 0)),
                      verticalOffset: moduleVerticalOffset())
            }
        case .head:
            // Gentle push-in toward the head: enough motion to read as a dive,
            // small enough that the robot stays inside the window's content box
            // until the crossfade takes over (4.6× clipped the whole rig out).
            let c = moduleCenterLocal(.neck).map { $0 + SIMD3(0, 0.14, 0) }
                ?? SIMD3(0, 1.35, 0)
            frame(zoom: 2.3, focusLocal: c)
        case .headImage:
            // VOLUMETRIC dive to the head-cam's projected-image plane: the rig
            // rotates so the plane faces the viewer SQUARELY and scales until
            // the image fills the window width (matching the video's
            // aspect-fill crop) — the crossfade then swaps identical pictures.
            // Offline (no plane) falls back to a shallow neck dive.
            if preDive == nil { preDive = stage.rig.root.transform }
            if let f = layers?.headCamFrameLocal(), let home = homeTransform ?? (stage.loaded ? stage.rig.root.transform : nil) {
                // Exact fill: on-screen plane width == the 1st-person viewport
                // width (window minus the FPV insets). ~1360 pt per metre.
                let ptPerM: Float = 1360
                let targetW = viewSize.width > 0
                    ? Float(viewSize.width - 2 * RobotSpaceView.povInsetSide) / ptPerM
                    : 1.1
                let zoom = max(2.0, targetW / max(0.05, f.width * home.scale.x))
                frame(zoom: zoom, focusLocal: f.pos, absoluteRotation: f.rot.inverse,
                      verticalOffset: -Float(RobotSpaceView.povInsetTop - RobotSpaceView.povInsetBottom) / (2 * ptPerM),
                      duration: presetInstant ? 0 : 1.6,
                      // The dive's whole job is to face the head-cam's image
                      // plane SQUARELY so the crossfade swaps identical
                      // pictures. Levelling it would tilt the picture.
                      levelled: false)
            } else {
                let c = moduleCenterLocal(.neck).map { $0 + SIMD3(0, 0.14, 0) }
                    ?? SIMD3(0, 1.35, 0)
                frame(zoom: 2.6, focusLocal: c,
                      duration: presetInstant ? 0 : 1.6)
            }
        case .returnFromHead:
            let target = preDive ?? homeTransform
            preDive = nil
            if let t = target {
                stage.rig.root.move(to: t, relativeTo: stage.rig.root.parent,
                                    duration: 1.6, timingFunction: .easeInOut)
            } else {
                frame(zoom: 1.0, focusLocal: nil)
            }
        }
    }

    /// Animate the rig relative to its FITTED home: `zoom` scales it, an
    /// optional rig-local focus point is brought to the stage centre, and
    /// `extraYaw` composes an isometric turn onto the home orientation.
    private func frame(zoom: Float, focusLocal: SIMD3<Float>?,
                       extraYaw: simd_quatf = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)),
                       absoluteRotation: simd_quatf? = nil,
                       verticalOffset: Float = 0,
                       duration: TimeInterval = 0.8,
                       levelled: Bool = true) {
        if homeTransform == nil, stage.loaded {
            homeTransform = stage.rig.root.transform
        }
        guard let home = homeTransform else { return }
        let rig = stage.rig
        let s0 = home.scale.x
        let r = absoluteRotation ?? (home.rotation * extraYaw)
        var t = home
        t.scale = SIMD3(repeating: s0 * zoom)
        t.rotation = r
        // Keep the robot's mid-height point fixed at the stage centre so zoom
        // pivots around the robot, not the rig origin (its feet).
        let mid = SIMD3<Float>(0, 0.8, 0)
        let centre = home.translation + home.rotation.act(mid * s0)
        let anchor = focusLocal ?? mid
        t.translation = centre - r.act(anchor * (s0 * zoom)) + SIMD3(0, verticalOffset, 0)
        // Hand the framing to the stage so a subsequent drag/pinch composes ON
        // it and pivots on `anchor`, instead of resetting the rig to a level yaw
        // about its own origin (which threw away every module framing on the
        // first drag — operator, 2026-07-28).
        stage.adoptFraming(t, anchorLocal: anchor, levelled: levelled)
        if duration <= 0 {
            rig.root.transform = t
        } else {
            rig.root.move(to: t, relativeTo: rig.root.parent, duration: duration,
                          timingFunction: .easeInOut)
        }
    }

    /// Per-module viewing recipe, ported from iOS v2's `flyToModule` (which
    /// mirrors the firmware GUI's robot3d flights): lift = straight-on FRONT
    /// view directly at the column, hands = PALM-ON (looking into the palm
    /// along the wrist-cam axis), arms = per-side isometric, neck / torso =
    /// front isometrics. The stage has no camera — the RIG rotates so the
    /// desired view direction points at the viewer (stage +Z) — so the recipe
    /// returns that absolute rotation plus a zoom fitted to the module extent.
    private func moduleFraming(_ m: ModuleFocus, center: SIMD3<Float>) -> (rot: simd_quatf, zoom: Float)? {
        let rig = stage.rig
        guard stage.loaded else { return nil }
        func joint(_ name: String) -> SIMD3<Float>? {
            rig.joints[name].map { $0.node.position(relativeTo: rig.root) }
        }
        let up = SIMD3<Float>(0, 1, 0)
        // Robot frame axes on the ground plane, rig-root space (nose = base −X —
        // the same convention as the SceneKit port).
        let rw = rig.robot.transformMatrix(relativeTo: rig.root)
        var front = -SIMD3(rw.columns.0.x, rw.columns.0.y, rw.columns.0.z); front.y = 0
        front = simd_length(front) < 1e-5 ? SIMD3(0, 0, 1) : simd_normalize(front)
        var lateral = SIMD3(rw.columns.2.x, rw.columns.2.y, rw.columns.2.z); lateral.y = 0
        lateral = simd_length(lateral) < 1e-5 ? SIMD3(1, 0, 0) : simd_normalize(lateral)

        var dir: SIMD3<Float>
        var radius: Float
        switch m {
        case .armLeft, .armRight:
            let side = m.side ?? "left"
            radius = moduleRadius(m, center: center)
                ?? ((1...7).compactMap { joint("\(side)_arm_\($0)") }
                        .map { simd_length($0 - center) }.max() ?? 0.4) + 0.15
            // INNER (medial) side view — operator feedback: seeing the arm from
            // the body side reads the elbow/wrist pose better than the outer iso.
            let sideSign: Float = side == "left" ? -1 : 1
            dir = simd_normalize(front + 0.55 * sideSign * lateral + 0.65 * up)
        case .handLeft, .handRight:
            let side = m.side ?? "left"
            guard let palm = rig.robot.findEntity(named: RobotRig.palmName(side)) else { return nil }
            // Palm normal ≈ the wrist camera's viewing axis (its mount looks out
            // of the palm): rotate the mount's −Z into rig space via the palm.
            // l/r_wrist_cam mount_quat (w,x,y,z) = (0, 0, 0.98584, 0.16772).
            // (a gripper: across the jaws — RobotRig.palmViewAxis)
            let palmRot = palm.orientation(relativeTo: rig.root)
            dir = simd_normalize(palmRot.act(RobotRig.palmViewAxis(side)))
            // Anti-occlusion: tilt horizontally away from the arm mass.
            let armPts = (1...7).flatMap { ["left_arm_\($0)", "right_arm_\($0)"] }
                .compactMap { joint($0) }
            if !armPts.isEmpty {
                var away = center - armPts.reduce(.zero, +) / Float(armPts.count)
                away.y = 0
                if simd_length(away) > 1e-5 {
                    if RobotRig.handGen(side) == "gripper", simd_dot(dir, away) < 0 { dir = -dir }
                    dir = simd_normalize(dir + 0.35 * simd_normalize(away))
                }
            }
            radius = moduleRadius(m, center: center) ?? 0.16
        case .neck:
            dir = simd_normalize(front + 0.5 * lateral + 0.45 * up)
            radius = moduleRadius(m, center: center) ?? 0.28
        case .lift:
            radius = moduleRadius(m, center: center)
                ?? ((["lift_lower", "lift_upper", "torso"].compactMap { joint($0) }
                        .map { simd_length($0 - center) }.max() ?? 0.4) + 0.25)
            dir = simd_normalize(front + 0.12 * up)      // straight-on front view
        case .torso:
            dir = simd_normalize(front + 0.4 * lateral + 0.5 * up)
            radius = moduleRadius(m, center: center) ?? 0.45
        }
        return (Self.lookRotation(dir), moduleZoom(radius: radius))
    }

    /// The module's bounding-sphere radius about `center`, in rig metres,
    /// measured from the PLY geometry (not joint positions: a hand's fingers,
    /// an arm's shells and the lift's column all extend well past their joint
    /// frames, and framing to the frames is what cropped them). View-rotation
    /// independent by construction, so ONE number bounds every viewing angle.
    /// nil until the async rig load lands.
    private func moduleRadius(_ m: ModuleFocus, center: SIMD3<Float>) -> Float? {
        guard let b = moduleBox(m) else { return nil }
        // Sphere about the framing anchor, so the fit holds at any orbit angle.
        let r = Self.corners(b).map { simd_length($0 - center) }.max() ?? 0
        return r > 0.02 ? r : nil
    }

    /// A module's PLY geometry bounds in rig-root space, or nil before load.
    private func moduleBox(_ m: ModuleFocus) -> BoundingBox? {
        let rig = stage.rig
        guard !rig.parts.isEmpty else { return nil }
        func under(_ names: [String], stop: Set<String> = []) -> BoundingBox? {
            var out: BoundingBox? = nil
            for n in names {
                guard let node = rig.robot.findEntity(named: n),
                      let b = rig.geometryBounds(under: node, relativeTo: rig.root,
                                                 stoppingAt: stop)
                else { continue }
                out = out.map { $0.union(b) } ?? b
            }
            return out
        }
        // The rig is ONE tree, so a module has to be pruned as well as rooted:
        // `lift_base`'s subtree continues through the torso into both arms and
        // every finger. Measured extents (scratch harness over the shipped
        // rig.json + PLYs, home pose): arms 0.570/0.578 m, hands 0.211 m, neck
        // 0.164 m, lift 0.353 m, torso 0.437 m of bounding-sphere radius.
        let arms: Set<String> = ["r_arm_mount", "l_arm_mount"]
        let box: BoundingBox?
        switch m {
        case .armLeft:   box = under(["left_arm_1"])       // arm chain + its hand
        case .armRight:  box = under(["right_arm_1"])
        case .handLeft:  box = under([RobotRig.palmName("left")])    // palm + fingers, or the gripper
        case .handRight: box = under([RobotRig.palmName("right")])
        case .neck:      box = under(["neck_yaw"])         // neck links + head
        case .lift:      box = under(["lift_base"], stop: ["body_mount"])   // the column
        case .torso:     box = under(["torso"], stop: arms)                 // torso shell + head
        }
        guard let b = box, b.extents.x.isFinite else { return nil }
        return b
    }

    static func corners(_ b: BoundingBox) -> [SIMD3<Float>] {
        [SIMD3(b.min.x, b.min.y, b.min.z), SIMD3(b.max.x, b.min.y, b.min.z),
         SIMD3(b.min.x, b.max.y, b.min.z), SIMD3(b.max.x, b.max.y, b.min.z),
         SIMD3(b.min.x, b.min.y, b.max.z), SIMD3(b.max.x, b.min.y, b.max.z),
         SIMD3(b.min.x, b.max.y, b.max.z), SIMD3(b.max.x, b.max.y, b.max.z)]
    }

    /// SOLVE the zoom instead of guessing it: a module of bounding-sphere radius
    /// `radius` must land inside the USABLE viewport (the window minus the
    /// control chrome) at `Self.fillFraction` of its smaller side.
    ///
    /// On-screen diameter, in points, of a rig-metre sphere at `zoom`:
    ///     d_pt = 2 · radius · s0 · zoom / scenePerPoint
    /// so requiring `d_pt ≤ fill · usableMin` inverts to the expression below.
    /// The old code used a hand-tuned constant (`1.37 / (radius·margin)`,
    /// clamped to 5.5) that knew nothing about the window OR the chrome, so the
    /// hand and neck framings clipped: "the camera moves so close the viewer
    /// cuts the robot off" (operator, 2026-07-28). `StageFramingTests` pins the
    /// inversion and checks every module's coverage.
    private func moduleZoom(radius: Float) -> Float {
        Self.fitZoom(radius: radius,
                     homeScale: homeTransform?.scale.x ?? stage.baseScale,
                     scenePerPoint: stage.scenePerPoint,
                     usable: Self.usableViewport(viewSize, insets: safeInsets))
    }

    /// The inversion, as a pure function so `StageFramingTests` can check it
    /// without a rig or a window. Falls back to the old hand-tuned curve ONLY
    /// when a measurement is missing (before the first window fit).
    static func fitZoom(radius: Float, homeScale: Float, scenePerPoint: Float,
                        usable: CGFloat) -> Float {
        guard homeScale > 0, scenePerPoint > 0, radius > 0.02, usable > 1 else {
            return max(minModuleZoom, min(maxModuleZoom, 1.37 / max(0.08, radius * 1.3)))
        }
        let fit = Float(usable) * scenePerPoint * fillFraction / (2 * radius * homeScale)
        return max(minModuleZoom, min(maxModuleZoom, fit))
    }

    /// On-screen DIAMETER in points that a given zoom actually produces — what
    /// the test asserts on (the zoom number itself means nothing on its own).
    static func onScreenDiameter(radius: Float, zoom: Float, homeScale: Float,
                                 scenePerPoint: Float) -> CGFloat {
        guard scenePerPoint > 0 else { return 0 }
        return CGFloat(2 * radius * homeScale * zoom / scenePerPoint)
    }

    /// Land the module at the centre of the CHROME-FREE box.
    ///
    /// `frame`'s default anchor sits ~11% of the view height LOW (it is built to
    /// stand a whole robot on the floor of the window), and the window's own
    /// centre is under a three-row top bar. Either one puts a tightly-framed
    /// module behind chrome; the offset below puts it in the middle of the space
    /// the operator can actually see.
    private func moduleVerticalOffset() -> Float {
        let mpp = stage.scenePerPoint
        guard mpp > 0, viewSize.height > 1, let home = homeTransform else { return 0 }
        let s0 = home.scale.x
        let anchorY = (home.translation + home.rotation.act(SIMD3<Float>(0, 0.8, 0) * s0)).y
        let targetY = -Float(safeInsets.top - safeInsets.bottom) / 2 * mpp
        return targetY - anchorY
    }

    /// Smaller side of the viewport actually free of chrome, in points.
    static func usableViewport(_ size: CGSize, insets: EdgeInsets) -> CGFloat {
        let w = size.width - insets.leading - insets.trailing
        let h = size.height - insets.top - insets.bottom
        return max(0, min(w, h))
    }

    /// How much of the usable box a framed module fills. 0.82 leaves a visible
    /// margin all round, which is what makes the framing read as deliberate
    /// rather than cropped; higher looks tight the moment the module moves.
    static let fillFraction: Float = 0.82
    /// Bounds on the solved zoom. The FLOOR is deliberately below 1: the solve
    /// only ever asks to zoom out for a module that is genuinely large (a fully
    /// extended lift), and refusing that request is how a clamp puts the subject
    /// back outside the window — the exact failure being fixed here. The CEILING
    /// stops a small module (the neck) from filling the view with one knuckle.
    static let minModuleZoom: Float = 0.6
    static let maxModuleZoom: Float = 4.0

    // ── Framing verification (measure it, never eyeball it) ───────────────────
    //
    // The floor bug taught this: a framing defect is invisible from most angles
    // and only a human staring at a screenshot ever catches it. So the flight
    // reports what it actually put on screen. `debugModuleTour YES` walks every
    // module in turn, which is how all seven get checked in one simulator run.

    #if DEBUG
    /// Project the module's geometry through the settled rig transform and
    /// report its on-screen box against the chrome-free viewport.
    private func verifyFraming(_ m: ModuleFocus, after delay: TimeInterval = 1.1) {
        let mpp = stage.scenePerPoint
        guard mpp > 0, viewSize.width > 1, let box = moduleBox(m) else { return }
        let seq = stage.framingSeq
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(delay))
            // A later flight superseded this one — measuring now would sample a
            // rig in mid-air and report a false FAIL (it did, on the tour's
            // first module).
            guard stage.framingSeq == seq else { return }
            let t = stage.rig.root.transform
            var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
            var hi = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
            for c in Self.corners(box) {
                let p = t.translation + t.rotation.act(c * t.scale.x)
                let pt = SIMD2(p.x / mpp, -p.y / mpp)      // scene metres → points, y down
                lo = simd_min(lo, pt); hi = simd_max(hi, pt)
            }
            let W = Float(viewSize.width), H = Float(viewSize.height)
            let l = -W / 2 + Float(safeInsets.leading), r = W / 2 - Float(safeInsets.trailing)
            let tp = -H / 2 + Float(safeInsets.top), b = H / 2 - Float(safeInsets.bottom)
            let inside = lo.x >= l && hi.x <= r && lo.y >= tp && hi.y <= b
            let covW = (hi.x - lo.x) / max(1, r - l), covH = (hi.y - lo.y) / max(1, b - tp)
            // Too SMALL is a defect too — a module framing that shows a speck is
            // as useless as one that crops.
            let big = max(covW, covH) >= 0.30
            print("""
            [MABEL-FRAME] \(m.title)\(m.side.map { " (\($0))" } ?? "") \
            box=[\(Int(lo.x)),\(Int(lo.y))]..[\(Int(hi.x)),\(Int(hi.y))] pt \
            usable=[\(Int(l)),\(Int(tp))]..[\(Int(r)),\(Int(b))] pt \
            cover=\(Int(covW * 100))%x\(Int(covH * 100))% \
            r=\(String(format: "%.3f", moduleRadius(m, center: moduleCenterLocal(m) ?? .zero) ?? -1)) \
            parts=\(stage.rig.parts.count) \
            zoom=\(String(format: "%.2f", t.scale.x / max(1e-6, homeTransform?.scale.x ?? 1))) \
            \(inside && big ? "PASS" : (inside ? "FAIL (too small to read)" : "FAIL (clipped)"))
            """)
        }
    }

    /// `debugModuleTour YES` — walk every module, 2.2 s apart, verifying each.
    /// The one way to check all seven framings without seven manual taps.
    func runModuleTourIfRequested() {
        guard UserDefaults.standard.bool(forKey: "debugModuleTour") else { return }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            for (i, m) in ModuleFocus.allCases.enumerated() {
                apply(preset: .module(m))      // verifies as part of the flight
                if i < ModuleFocus.allCases.count - 1 {
                    try? await Task.sleep(for: .seconds(2.2))
                }
            }
        }
    }
    #endif

    /// Rotation taking `dir` (rig-local) to stage +Z (out of the window, toward
    /// the viewer) with the horizon kept level — the subject is seen ALONG `dir`.
    static func lookRotation(_ dir: SIMD3<Float>) -> simd_quatf {
        let up = SIMD3<Float>(0, 1, 0)
        var z = simd_normalize(dir)
        var x = simd_cross(up, z)
        if simd_length(x) < 1e-4 { x = SIMD3(1, 0, 0); z = simd_normalize(z) } else { x = simd_normalize(x) }
        let y = simd_cross(z, x)
        return simd_quatf(simd_float3x3(columns: (x, y, z))).inverse
    }

    /// A module's centre in the rig root's coordinate space (live FK).
    private func moduleCenterLocal(_ m: ModuleFocus) -> SIMD3<Float>? {
        let rig = stage.rig
        func joint(_ name: String) -> SIMD3<Float>? {
            rig.joints[name].map { $0.node.position(relativeTo: rig.root) }
        }
        func average(_ names: [String]) -> SIMD3<Float>? {
            let pts = names.compactMap { joint($0) }
            return pts.isEmpty ? nil : pts.reduce(.zero, +) / Float(pts.count)
        }
        switch m {
        case .armLeft:   return average((1...7).map { "left_arm_\($0)" })
        case .armRight:  return average((1...7).map { "right_arm_\($0)" })
        case .handLeft:  return rig.robot.findEntity(named: RobotRig.palmName("left"))?.position(relativeTo: rig.root)
        case .handRight: return rig.robot.findEntity(named: RobotRig.palmName("right"))?.position(relativeTo: rig.root)
        case .neck:      return average(["neck_yaw", "neck_pitch", "neck_roll"])
        case .lift:      return average(["lift_lower", "lift_upper", "torso"])
        case .torso:     return joint("torso")
        }
    }
}
