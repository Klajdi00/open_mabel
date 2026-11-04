//
//  LiveMapView.swift
//  MABEL Mobile App
//
//  SceneKit viewport for the LIVE SLAM map. Subscribes to the LiveMapStore's
//  mesh / pose callbacks and mutates the scene nodes directly — the heavy point
//  cloud never round-trips through SwiftUI state. The articulated MABEL is posed
//  from the shared robot joints (so arm/lift poses set on other tabs show here),
//  while its base pose comes from the live odom stream.
//

import SwiftUI
import SceneKit
import UIKit

/// Viewer camera mode: 3rd person (orbit / chase over the map) or 1st person
/// (the robot's own head camera). Switching flies the viewer camera between the
/// two poses — the same choreography as the web control center's POV toggle.
enum PovMode: String, CaseIterable, Identifiable {
    case third, first
    var id: String { rawValue }
    var title: String { self == .third ? "3rd" : "1st" }
    var icon: String { self == .third ? "video.fill" : "eye.fill" }
}

/// A staged (not-yet-sent) arm plan, surfaced to the UI for the Execute pill —
/// which arm(s), how big the move, and whether it looks like a self-collision.
/// Ported from iOS v2.
struct ArmPlanPreview: Equatable {
    var side: String          // "left" | "right" | "both"
    var deltaRad: Double
    var collision: Bool
    var reason: String
}

/// Joint-space module focus — picking one flies the viewer camera to that
/// module's ideal framing (ported from the firmware GUI's robot3d flights):
/// lift = front view, arms = per-side isometric, hands = palm-on (facing the
/// palm inside), neck = head isometric, torso = torso TF.
enum ModuleFocus: String, CaseIterable, Identifiable {
    case armLeft, armRight, neck, handLeft, handRight, lift, torso
    var id: String { rawValue }
    var title: String {
        switch self {
        case .armLeft: return "L Arm"
        case .armRight: return "R Arm"
        case .neck: return "Neck"
        case .handLeft: return RobotRig.handGen("left") == "gripper" ? "L Gripper" : "L Hand"
        case .handRight: return RobotRig.handGen("right") == "gripper" ? "R Gripper" : "R Hand"
        case .lift: return "Lift"
        case .torso: return "Torso"
        }
    }
    var icon: String {
        switch self {
        case .armLeft, .armRight: return "figure.arms.open"
        case .neck: return "face.smiling"
        case .handLeft, .handRight: return RobotRig.handGen(side ?? "left") == "gripper" ? "hand.pinch.fill" : "hand.raised.fill"
        case .lift: return "arrow.up.and.down"
        case .torso: return "figure.stand"
        }
    }
    var side: String? {
        switch self {
        case .armLeft, .handLeft: return "left"
        case .armRight, .handRight: return "right"
        default: return nil
        }
    }
}

struct LiveMapView: UIViewRepresentable {
    let store: LiveMapStore
    /// Decoupled live pose: a CADisplayLink render loop applies the newest
    /// joints straight to the SceneKit nodes (v1's RealRobotView pattern), so
    /// the articulated robot animates WITHOUT any SwiftUI body churn — the
    /// glass chrome above never re-renders for telemetry.
    let pose: LivePose
    /// 2D-map mirror — fed the SAME grid/odom/path/mesh so the floating minimap is
    /// a true top-down RViz panel of the live stream.
    var navState: NavMapState? = nil
    var recenterToken: Int              // bump to re-frame the camera on the robot
    var fitToken: Int = 0               // bump to FIT the whole map to ~70% of view
    /// 3rd-person chase cam locked behind the robot (orbit disabled while on).
    var follow: Bool = false
    /// Viewer camera mode; changing it runs the 3rd↔1st person flight.
    var pov: PovMode = .third
    /// TRUE when this view mounts mid POV-transition (the 3D stage just dove
    /// into the head): start the camera just behind the head and settle in a
    /// SHORT flight, instead of the full 2 s flight from the wide orbit pose.
    var mountNearHead: Bool = false
    /// TRUE while the POV transition PARKS this hidden view at the head pose —
    /// the pov change applies near-instantly instead of the 2 s flight.
    var instantPov: Bool = false
    /// How long the 3rd↔1st camera flight takes — RobotSpaceView times the live
    /// FPV video crossfade to land exactly when the flight does. V2 (visionOS)
    /// uses a slower, super-smooth 2 s glide (iOS uses 1.4 s).
    static let povFlightDuration: TimeInterval = 2.0

    /// Vertical FOV (degrees) that makes the 3D first-person render PIXEL-
    /// CONSISTENT with the FPV video's aspect-FILL crop at the given view
    /// aspect — the control center's crossfade trick (true fx/fy + canvas
    /// aspect). zed_left rig intrinsics: 480×270 @ f = 151.79.
    static func povDisplayFov(viewAspect a: CGFloat) -> CGFloat {
        let srcW = 480.0, srcH = 270.0, f = 151.79
        let srcAspect = srcW / srcH
        guard Double(a) > srcAspect else {
            // View narrower than the source: HEIGHT fills → native vertical FOV.
            return CGFloat(2 * atan(srcH / 2 / f) * 180 / .pi)
        }
        // View wider than the source: WIDTH fills, the vertical is cropped —
        // the visible vertical FOV shrinks with the view aspect.
        let fovxFull = 2 * atan(srcW / 2 / f)
        let fovy = 2 * atan(tan(fovxFull / 2) / Double(a))
        return CGFloat(fovy * 180 / .pi)
    }
    /// Control-center-style "Cam 3D": live camera frustum + video projections
    /// hung on the robot. `camVideoURL` maps a wire stream key → MJPEG URL.
    /// `camRouteToken` must change whenever those URLs change (connect /
    /// LAN↔Tailscale switch) — it re-attaches the projections so the streams follow.
    var camProjections: Bool = false
    var camVideoURL: ((String) -> URL?)? = nil
    var camRouteToken: String = ""

    // Layer visibility (the RViz-style toggles in the pull-up console).
    var meshHidden: Bool = false        // accumulated 3D SLAM mesh map
    var cloudHidden: Bool = false       // live 3D sensor point cloud
    var gridHidden: Bool = false        // 2D occupancy plane (the SLAM floorplan)
    var robotHidden: Bool = false       // MABEL's own mesh (sensor layers stay)
    var floorHidden: Bool = false       // ground surface + contact shadow
    var floorGridHidden: Bool = false   // 1 m / 5 m reference grid on the floor
    var tfHidden: Bool = true           // TF-tree axis triads on every joint frame
    /// Latest planar LiDAR sweep — rendered as a 3D point ring in the CLOUD
    /// layer (rides the robot node). Applied when `stamp` changes.
    var lidarScan: RobotState.LidarScan? = nil
    var cloudStride: Int = 1            // live-cloud density (1 = full, N = every Nth)
    var camProjSize: Double = 0.3       // Cam 3D image height (m)
    var camFlat: Bool = true            // Cam 3D: rectified plane vs curved lens shell

    /// TASK-space direct manipulation: grab an arm link / palm ball → damped-
    /// least-squares IK; drag a wrist ring → that wrist axis. `forceMode` (Soft)
    /// turns a grab into an external PUSH instead (the robot is back-drivable).
    var manipulate: Bool = false
    var forceMode: Bool = false
    /// MoveIt-style safety gate for the green-ball grab (Stiff + Task). When true a
    /// drag PLANS + previews instead of streaming IK to the robot; nothing is sent
    /// until Execute. Ported from iOS v2.
    var planMode: Bool = false
    /// Control REGION for the plan (arm / upper body / whole body). Widens the IK
    /// chain the green-ball drag solves over: `arm` = the arm only; `upperBody` also
    /// lets the TORSO and LIFT reach for a far target; `wholeBody` additionally walks
    /// the mobile BASE toward it. Proximal DOFs are heavily damped, so they engage
    /// ONLY when the arm can't reach ("unless torso/lift/base not moving"). Execute
    /// streams arm+torso+lift as a joint_command burst and the base as a nav goal.
    var planRegion: ControlRegion = .arm
    var onPlanChange: ((ArmPlanPreview?) -> Void)? = nil
    /// Whole-body Execute: drive the mobile base to the planned pose (odom x, y, yaw)
    /// via a nav goal — the base half of a whole-body plan.
    var onBaseGoal: ((Double, Double, Double) -> Void)? = nil
    var executePlanToken: Int = 0
    var cancelPlanToken: Int = 0
    /// Bump to drop a PLACED (never dispatched) goal marker.
    var clearGoalToken: Int = 0
    var debugStageToken: Int = 0
    var debugStageSide: String = "left"
    var onJointsChange: (([String: Double]) -> Void)? = nil
    var onJointChange: ((String, Double) -> Void)? = nil
    var onPartForce: ((String, [Double]) -> Void)? = nil

    /// Links the operator may grab-drag: the 7-DOF arms + the torso.
    static let grabbableJoints: Set<String> = {
        var s: Set<String> = ["torso"]
        for side in ["left", "right"] { for i in 1...7 { s.insert("\(side)_arm_\(i)") } }
        return s
    }()
    /// The proximal DOFs an upper-body / whole-body plan adds to the wrist IK, in
    /// proximal→distal order: the torso pitch and the two synced lift stages. They
    /// share the target's residual with the arm, but only when it can't reach alone.
    static let upperBodyDofs = ["torso", "lift_lower", "lift_upper"]
    /// The arm must fall SHORT by more than this (metres) before a proximal DOF
    /// (torso/lift/base) is allowed to help. A reachable target keeps the arm's
    /// residual under it for the whole drag, so those DOFs never move ("unless not
    /// moving"); once the arm saturates and can't close the gap they engage. Weight
    /// alone isn't enough — a small weight still creeps the base over a long drag.
    static let proximalEngageDist: Float = 0.08
    /// Tap-to-navigate: a tapped floor point → a Nav2 goal in the ROS `odom`/mesh
    /// frame (x, y metres, yaw rad facing the goal). nil disables tapping (demo /
    /// not connected), so the Live map is view-only until a real robot is up.
    var onGoal: ((_ x: Double, _ y: Double, _ yaw: Double) -> Void)? = nil
    /// SAFETY margin around the floating UI: taps inside these edge bands never
    /// set a goal — a near-miss on a button must not send the robot somewhere.
    var goalExclusion: UIEdgeInsets = .zero
    /// Joint-space module focus: bump `focusToken` to fly the camera to
    /// `focusModule`'s framing (computed live from the rig's FK state).
    var focusModule: ModuleFocus? = nil
    var focusToken: Int = 0
    /// While Follow is ON and Joint space has a module selected, the chase cam
    /// tracks THAT module (closer, centred on its live FK anchor) — flying the
    /// free camera would just be stomped by the follow display link.
    var followModule: ModuleFocus? = nil

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.backgroundColor = .clear
        v.antialiasingMode = .multisampling2X
        v.allowsCameraControl = true
        v.defaultCameraController.interactionMode = .orbitTurntable
        v.defaultCameraController.inertiaEnabled = true

        let (scene, rig) = LiveMapScene.make()
        v.scene = scene
        v.pointOfView = rig.camera

        let c = context.coordinator
        c.view = v
        c.rig = rig
        c.store = store
        c.onGoal = onGoal
        c.navState = navState
        c.lastRecenter = recenterToken
        c.lastFit = fitToken
        c.pose = pose
        rig.apply(joints: pose.joints)
        // rAF-style joint mirror: apply the newest LivePose each display frame,
        // decoupled from SwiftUI (cheap no-op while the pose tick is unchanged).
        let jointLink = CADisplayLink(target: c, selector: #selector(Coordinator.poseTickStep))
        jointLink.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 120, preferred: 120)
        jointLink.add(to: .main, forMode: .common)
        c.jointLink = jointLink
        navState?.beginLive()

        // A discrete tap sets a nav goal; it coexists with the orbit camera
        // (which uses pan / pinch), so driving the view and tapping don't fight.
        let tap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.onTap(_:)))
        v.addGestureRecognizer(tap)
        // Pinch zoom that works WHILE following (orbit is off then). Enabled only
        // during follow, so it never fights the default camera controller.
        let pinch = UIPinchGestureRecognizer(target: c, action: #selector(Coordinator.onFollowPinch(_:)))
        pinch.isEnabled = false
        v.addGestureRecognizer(pinch); c.followPinch = pinch
        // One-finger pan orbits the follow cam (two-finger stays as pinch zoom).
        let pan = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.onFollowPan(_:)))
        pan.maximumNumberOfTouches = 1; pan.isEnabled = false
        v.addGestureRecognizer(pan); c.followPan = pan
        // TASK-space manipulation drag: begins ONLY over a grabbable part (the
        // delegate hit-tests), so empty-space pans still orbit the default camera.
        let manip = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.onManipPan(_:)))
        manip.maximumNumberOfTouches = 1
        manip.delegate = c
        manip.isEnabled = manipulate
        v.addGestureRecognizer(manip); c.manipPan = manip
        // Goal-ring drag: re-aim the placed Nav2 goal's yaw. Begins only over the
        // goal marker (delegate hit-tests), commits the new heading on release.
        let goalPan = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.onGoalPan(_:)))
        goalPan.maximumNumberOfTouches = 1
        goalPan.delegate = c
        v.addGestureRecognizer(goalPan); c.goalPan = goalPan
        if follow { c.setFollowing(true, view: v) }

        // Stream → scene. Hops are already on the main actor (store dispatches
        // its callbacks there), so direct node mutation is safe. The chase cam is
        // driven by a display link (not here), so it stays smooth between poses.
        store.onMesh = { [weak c] mesh in c?.rig?.apply(mesh: mesh); c?.navState?.ingest(mesh: mesh) }
        store.onPose = { [weak c] pose in c?.ingestMapPose(pose) }
        store.onPath = { [weak c] path in c?.rig?.apply(path: path); c?.mirrorPath(path) }
        store.onMap = { [weak c] grid in c?.rig?.apply(grid: grid); c?.navState?.ingest(grid: grid) }
        store.onCloud = { [weak c] cloud in c?.rig?.apply(cloud: cloud) }
        return v
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        let c = context.coordinator
        c.onGoal = onGoal               // track connection state (nil → view-only)
        c.goalExclusion = goalExclusion
        c.onJointsChange = onJointsChange
        c.onJointChange = onJointChange
        c.onPartForce = onPartForce
        c.forceMode = forceMode
        c.planMode = planMode
        c.planRegion = planRegion
        c.onPlanChange = onPlanChange
        c.onBaseGoal = onBaseGoal
        // OFF the update pass, all three: each ends in `onPlanChange`, which writes
        // @State on the host view. Mutating state from inside updateUIView is
        // undefined behaviour and SwiftUI silently DROPS it — pressing ✕ tore the
        // plan down in the scene but left the Execute/Cancel pills on screen
        // forever, because `armPlan = nil` never landed. (The drag path was always
        // fine: it runs from a UIKit gesture callback, outside the update.)
        if !planMode && c.hasPlan { DispatchQueue.main.async { [weak c] in c?.cancelPlan() } }
        if executePlanToken != c.executePlanToken {
            c.executePlanToken = executePlanToken
            DispatchQueue.main.async { [weak c] in c?.executePlan() }
        }
        if cancelPlanToken != c.cancelPlanToken {
            c.cancelPlanToken = cancelPlanToken
            DispatchQueue.main.async { [weak c] in c?.cancelPlan() }
        }
        if clearGoalToken != c.clearGoalToken {
            c.clearGoalToken = clearGoalToken
            c.rig?.clearGoal()
            c.lastGoalScene = nil
            c.lastGoalOdom = nil
        }
        if debugStageToken != c.debugStageToken {
            c.debugStageToken = debugStageToken
            let side = debugStageSide
            // OFF the update pass. stagePlanForTest ends in onPlanChange, which
            // assigns @State on the host view; mutating state from inside
            // updateUIView is undefined behaviour and SwiftUI silently drops it
            // — the plan staged and the ghost previewed, but the Execute pill
            // never appeared, so the harness "passed" while testing nothing.
            // The real path (onManipPan) is a UIKit gesture callback and runs
            // outside the update, which is why only the harness was affected.
            DispatchQueue.main.async { [weak c] in
                c?.stagePlanForTest(side: side,
                                    deltas: ["\(side)_arm_1": 0.35,
                                             "\(side)_arm_2": 0.30,
                                             "\(side)_arm_4": 0.40])
            }
        }
        c.rig?.setMeshHidden(meshHidden)
        c.rig?.setCloudHidden(cloudHidden)
        c.rig?.setGridHidden(gridHidden)
        c.rig?.setRobotHidden(robotHidden)
        c.rig?.setFloorHidden(floorHidden)
        c.rig?.setFloorGridHidden(floorGridHidden)
        c.rig?.setTFHidden(tfHidden)
        c.rig?.cloudStride = cloudStride
        if let scan = lidarScan, scan.stamp != c.lastLidarStamp {
            c.lastLidarStamp = scan.stamp
            c.rig?.apply(lidar: (scan.angleMin, scan.angleInc, scan.rangeMax, scan.ranges),
                         stride: cloudStride)
        }
        c.projector.setProjection(size: camProjSize, flat: camFlat)
        if manipulate != c.manipulating { c.setManipulating(manipulate, view: uiView) }
        if recenterToken != c.lastRecenter {
            c.lastRecenter = recenterToken
            if c.currentPov == .third {
                if c.following {
                    c.resetFollowCam()          // snap behind the robot, default zoom — still following
                } else if let rig = c.rig {
                    rig.frameCamera(on: rig.robot.simdPosition); uiView.pointOfView = rig.camera
                }
            }
        }
        if fitToken != c.lastFit {
            c.lastFit = fitToken
            if c.currentPov == .third {
                if c.following {
                    c.fitFollowCam()            // wide chase view — still following
                } else if let rig = c.rig {
                    rig.fitCamera(view: uiView, fraction: 0.70); uiView.pointOfView = rig.camera
                }
            }
        }
        if follow != c.following {
            c.setFollowing(follow, view: uiView)
        }
        if mountNearHead, !c.didNearHeadMount, pov == .first, c.currentPov == .third,
           let rig = c.rig {
            c.didNearHeadMount = true
            let n = SCNNode()
            n.transform = rig.povCamera.worldTransform
            n.simdLocalTranslate(by: SIMD3(0, 0.18, 1.1))   // just up-and-behind the head
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0
            rig.camera.transform = n.worldTransform
            SCNTransaction.commit()
            uiView.pointOfView = rig.camera
            c.flightDuration = 0.9                          // short settle, not a re-flight
        }
        if pov != c.currentPov {
            if instantPov { c.flightDuration = 0.08 }
            c.setPov(pov, view: uiView)
        }
        if focusToken != c.lastFocusToken {
            c.lastFocusToken = focusToken
            // Free camera only — while following, the chase cam handles module
            // focus itself (see setFollowModule); a flight would be overwritten
            // on the very next follow display-link frame.
            if let m = focusModule, c.currentPov == .third, !c.following {
                c.flyToModule(m, view: uiView)
            }
        }
        c.setFollowModule(followModule)
        if camProjections != c.projector.attached
            || (camProjections && c.lastCamRoute != camRouteToken) {
            if camProjections, let rig = c.rig {
                c.lastCamRoute = camRouteToken
                c.projector.attach(to: rig.robot, urlProvider: { camVideoURL?($0) })
            } else {
                c.projector.detach()
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    /// Leaving the tab: drop the stream callbacks so a torn-down view never holds
    /// the store's closures (the store keeps streaming only while `active`).
    static func dismantleUIView(_ uiView: SCNView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var flightDuration: TimeInterval = LiveMapView.povFlightDuration
        var didNearHeadMount = false
        weak var view: SCNView?
        weak var store: LiveMapStore?
        var rig: LiveMapRig?
        var onGoal: ((Double, Double, Double) -> Void)?
        var goalExclusion: UIEdgeInsets = .zero
        var navState: NavMapState?
        // Manipulation wiring (task space).
        var manipPan: UIPanGestureRecognizer?
        var manipulating = false
        var forceMode = false
        var currentJoints: [String: Double] = [:]
        var frozenJoints: Set<String> = []
        // Decoupled joint mirror (LivePose → nodes, per display frame).
        var pose: LivePose?
        var jointLink: CADisplayLink?
        private var lastPoseTick: UInt64 = .max

        @objc func poseTickStep() {
            guard let pose, let rig else { return }
            guard pose.tick != lastPoseTick else { return }
            lastPoseTick = pose.tick
            currentJoints = pose.joints
            rig.apply(joints: pose.joints, except: frozenJoints)
        }
        var onJointsChange: (([String: Double]) -> Void)?
        var onJointChange: ((String, Double) -> Void)?
        var onPartForce: ((String, [Double]) -> Void)?
        private var poseTick = 0
        var lastRecenter = 0
        var lastFit = 0
        var lastLidarStamp: TimeInterval = -1
        var following = false
        var currentPov: PovMode = .third
        // Saved 3rd-person pose so leaving 1st person returns exactly where we were.
        private var savedCamTransform: SCNMatrix4?
        private var savedFov: CGFloat = 52
        /// "Cam 3D" frustum + live-video projections on the robot.
        let projector = CameraProjector()
        var lastCamRoute = ""

        /// SLAM odom frame arrived. A staged whole-body plan owns the base — hold the
        /// root where the preview parked it — else track the live pose.
        func ingestMapPose(_ pose: MapPose) {
            if !baseFrozen { rig?.apply(pose: pose) }
            mirrorPose(pose)
        }

        /// Mirror the odom pose into the 2D map (throttled — odom can run 50 Hz; the
        /// minimap only needs ~15 Hz). Ground = SceneKit (x, z); yaw about +Y.
        func mirrorPose(_ pose: MapPose) {
            poseTick += 1
            guard poseTick % 3 == 0 else { return }
            navState?.setRobot(x: Double(pose.position.x), z: Double(pose.position.z),
                               yaw: Double(pose.yaw))
        }
        /// Mirror the Nav2 route into the 2D map (SceneKit ground (x, z)).
        func mirrorPath(_ path: MapPath) {
            navState?.setLivePath(path.points.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.z)) })
        }

        // Smooth chase cam: a display link damps the camera toward the robot every
        // frame, and a pinch (enabled only while following) zooms it.
        private let follow = FollowCamera()
        private var followLink: CADisplayLink?
        private var lastFollowTS: CFTimeInterval = 0
        var followPinch: UIPinchGestureRecognizer?
        var followPan: UIPanGestureRecognizer?

        func setFollowing(_ on: Bool, view: SCNView) {
            following = on
            view.allowsCameraControl = !on            // our orbit replaces the default one
            followPinch?.isEnabled = on
            followPan?.isEnabled = on
            if on {
                follow.reset()
                view.pointOfView = rig?.camera
                followLink?.invalidate(); lastFollowTS = 0
                let l = CADisplayLink(target: self, selector: #selector(followStep(_:)))
                l.add(to: .main, forMode: .common)
                followLink = l
            } else {
                followLink?.invalidate(); followLink = nil
            }
        }

        @objc private func followStep(_ link: CADisplayLink) {
            guard following, let rig else { return }
            let dt: Float = lastFollowTS == 0 ? 1.0 / 60
                : Float(min(max(link.timestamp - lastFollowTS, 1.0 / 120), 1.0 / 15))
            lastFollowTS = link.timestamp
            // Chassis by default; the focused module's live FK centre when one
            // is selected (rotation stays the robot's, so "behind" still means
            // behind the robot's heading).
            var target = rig.robot.simdWorldTransform
            if let m = currentFollowModule, let c = moduleCenter(m) {
                target.columns.3 = simd_float4(c.x, c.y, c.z, 1)
            }
            let (p, l, u) = follow.step(target: target, dt: dt)
            rig.camera.simdPosition = p
            rig.camera.look(at: SCNVector3(l), up: SCNVector3(u), localFront: SCNVector3(0, 0, -1))
        }

        // Follow-the-module state: swap the chase distance/lift in and out so
        // leaving Joint space restores the operator's chassis-follow zoom.
        private var currentFollowModule: ModuleFocus? = nil
        private var savedFollowDistance: Float? = nil

        func setFollowModule(_ m: ModuleFocus?) {
            guard m != currentFollowModule else { return }
            currentFollowModule = m
            if m != nil {
                if savedFollowDistance == nil { savedFollowDistance = follow.distance }
                follow.distance = 1.7
                follow.lift = 0
            } else {
                if let d = savedFollowDistance { follow.distance = d }
                savedFollowDistance = nil
                follow.lift = 0.45
            }
        }

        /// Live FK centre of a module (same anchors as flyToModule).
        private func moduleCenter(_ m: ModuleFocus) -> simd_float3? {
            guard let rig else { return nil }
            switch m {
            case .armLeft, .armRight:
                let pts = anchors((1...7).map { "\(m.side!)_arm_\($0)" })
                return pts.isEmpty ? nil : pts.reduce(.zero, +) / Float(pts.count)
            case .handLeft, .handRight:
                guard let palm = rig.robot.childNode(withName: RobotRig.palmName(m.side!), recursively: true)
                else { return nil }
                return simd_make_float3(palm.simdWorldTransform.columns.3)
            case .neck:
                let pts = anchors(["neck_yaw", "neck_pitch", "neck_roll"])
                return pts.isEmpty ? nil : pts.reduce(.zero, +) / Float(pts.count)
            case .lift:
                let pts = anchors(["lift_lower", "lift_upper", "torso"])
                return pts.isEmpty ? nil : pts.reduce(.zero, +) / Float(pts.count)
            case .torso:
                return anchor("torso")
            }
        }

        /// Fly the viewer camera between 3rd person (orbit/chase) and 1st person
        /// (the robot's head camera). To 1st: save the orbit pose, glide the free
        /// camera into the head-camera pose while widening the FOV to the zed's,
        /// then hand pointOfView to the head-parented povCamera so the view rides
        /// the robot. To 3rd: take over at the head pose and glide back out.
        func setPov(_ mode: PovMode, view: SCNView) {
            guard mode != currentPov, let rig else { return }
            currentPov = mode
            let duration = flightDuration
            flightDuration = LiveMapView.povFlightDuration   // one-shot override
            if mode == .first {
                // 1st person: our flight owns the camera — stop orbit + follow.
                if following { setFollowing(false, view: view) }
                view.allowsCameraControl = false
                // Take over from the CURRENT view camera — the default orbit
                // controller works on its own point-of-view node, so the user's
                // actual view is there, not on rig.camera. The flight starts
                // exactly where the user left the camera (and returns there).
                let current = view.pointOfView ?? rig.camera
                savedCamTransform = current.worldTransform
                savedFov = current.camera?.fieldOfView ?? 52
                SCNTransaction.begin(); SCNTransaction.animationDuration = 0
                rig.camera.transform = current.worldTransform
                rig.camera.camera?.fieldOfView = savedFov
                SCNTransaction.commit()
                view.pointOfView = rig.camera
                // FOV target = what the FPV's aspect-fill crop will actually show
                // at THIS view's aspect — the handover is then pixel-consistent
                // (no zoom jump when the video fades in).
                let fov = LiveMapView.povDisplayFov(
                    viewAspect: view.bounds.width / max(1, view.bounds.height))
                rig.povCamera.camera?.fieldOfView = fov
                SCNTransaction.begin()
                SCNTransaction.animationDuration = duration
                SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                rig.camera.setWorldTransform(rig.povCamera.worldTransform)
                rig.camera.camera?.fieldOfView = fov
                SCNTransaction.completionBlock = { [weak self, weak view] in
                    guard let self, let view, self.currentPov == .first else { return }
                    view.pointOfView = rig.povCamera
                }
                SCNTransaction.commit()
            } else {
                // 3rd person: take over exactly at the head pose, then pull back
                // out to the saved orbit pose.
                rig.camera.setWorldTransform(rig.povCamera.worldTransform)
                rig.camera.camera?.fieldOfView = rig.povCamera.camera?.fieldOfView ?? 83.3
                view.pointOfView = rig.camera
                let target = savedCamTransform
                SCNTransaction.begin()
                SCNTransaction.animationDuration = duration
                SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                if let target { rig.camera.transform = target }
                else { rig.frameCamera(on: rig.robot.simdPosition) }
                rig.camera.camera?.fieldOfView = savedFov
                SCNTransaction.completionBlock = { [weak self, weak view] in
                    guard let self, let view, self.currentPov == .third else { return }
                    view.allowsCameraControl = !self.following
                }
                SCNTransaction.commit()
            }
        }

        // MARK: - Module focus flights (port of the firmware GUI's robot3d views)

        var lastFocusToken = 0

        /// World position of a rig joint's articulation node.
        private func anchor(_ name: String) -> simd_float3? {
            rig?.robotJoints[name].map { simd_float3($0.node.simdWorldPosition) }
        }
        private func anchors(_ names: [String]) -> [simd_float3] {
            names.compactMap { anchor($0) }
        }

        /// Fit distance so a bounding radius fills the frame (robot3d._fit).
        private func fit(_ radius: Float, fovDeg: CGFloat, margin: Float = 1.3) -> Float {
            let fov = Float(max(0.2, fovDeg * .pi / 180))
            return max(0.35, radius * margin / tan(fov / 2))
        }

        /// Fly the viewer camera to a module's framing over ~1.2 s. Computed from
        /// the CURRENT FK state (nodes ride the live joints), same recipes as the
        /// firmware GUI: lift front-on, arms per-side isometric, hands palm-on,
        /// neck head-isometric, torso at its TF.
        func flyToModule(_ m: ModuleFocus, view: SCNView) {
            guard let rig else { return }
            let up = simd_float3(0, 1, 0)
            // Robot frame axes on the ground: nose = local −X taken to world.
            let rw = rig.robot.simdWorldTransform
            var front = -simd_make_float3(rw.columns.0); front.y = 0
            front = simd_length(front) < 1e-5 ? simd_float3(0, 0, 1) : simd_normalize(front)
            var lateral = simd_make_float3(rw.columns.2); lateral.y = 0   // robot local +Z on the ground
            lateral = simd_length(lateral) < 1e-5 ? simd_float3(1, 0, 0) : simd_normalize(lateral)

            let fovDeg = (view.pointOfView?.camera ?? rig.camera.camera)?.fieldOfView ?? 52
            var center = simd_float3(rig.robot.simdWorldPosition) + simd_float3(0, 0.6, 0)
            var dir = simd_normalize(front + 0.55 * lateral + 0.65 * up)
            var dist: Float = 2.4

            switch m {
            case .armLeft, .armRight:
                let side = m.side!
                let pts = anchors((1...7).map { "\(side)_arm_\($0)" })
                guard !pts.isEmpty else { return }
                center = pts.reduce(.zero, +) / Float(pts.count)
                let r = pts.map { simd_length($0 - center) }.max() ?? 0.4
                // Isometric from the robot's front, tilted toward THAT side's free space.
                let sideSign: Float = side == "left" ? 1 : -1
                dir = simd_normalize(front + 0.55 * sideSign * lateral + 0.65 * up)
                dist = fit(r + 0.15, fovDeg: fovDeg)
            case .handLeft, .handRight:
                let side = m.side!
                guard let palmNode = rig.robot.childNode(withName: RobotRig.palmName(side), recursively: true)
                else { return }
                let pw = palmNode.simdWorldTransform
                center = simd_make_float3(pw.columns.3)
                // Palm normal ≈ the palm camera's viewing axis (its mount looks out
                // of the palm): rotate the mount's −Z into world via the palm node.
                // l/r_wrist_cam mount_quat (w,x,y,z) = (0, 0, 0.98584, 0.16772).
                // (a gripper: across the jaws — RobotRig.palmViewAxis)
                let palmRot = simd_quatf(pw)
                var normal = simd_normalize(palmRot.act(RobotRig.palmViewAxis(side)))
                // Anti-occlusion: tilt horizontally away from the arm mass.
                let armPts = anchors((1...7).flatMap { ["left_arm_\($0)", "right_arm_\($0)"] })
                if !armPts.isEmpty {
                    var away = center - armPts.reduce(.zero, +) / Float(armPts.count)
                    away.y = 0
                    if simd_length(away) > 1e-5 {
                        if RobotRig.handGen(side) == "gripper", simd_dot(normal, away) < 0 { normal = -normal }
                        normal = simd_normalize(normal + 0.35 * simd_normalize(away))
                    }
                }
                dir = normal
                dist = fit(0.16, fovDeg: fovDeg, margin: 1.5)
            case .neck:
                let pts = anchors(["neck_yaw", "neck_pitch", "neck_roll"])
                guard !pts.isEmpty else { return }
                center = pts.reduce(.zero, +) / Float(pts.count)
                dir = simd_normalize(front + 0.5 * lateral + 0.45 * up)
                dist = fit(0.28, fovDeg: fovDeg, margin: 1.5)
            case .lift:
                let pts = anchors(["lift_lower", "lift_upper", "torso"])
                guard !pts.isEmpty else { return }
                center = pts.reduce(.zero, +) / Float(pts.count)
                let r = (pts.map { simd_length($0 - center) }.max() ?? 0.4) + 0.25
                dir = simd_normalize(front + 0.12 * up)      // straight-on front view
                dist = fit(r, fovDeg: fovDeg, margin: 1.25)
            case .torso:
                guard let t = anchor("torso") else { return }
                center = t
                dir = simd_normalize(front + 0.4 * lateral + 0.5 * up)
                dist = fit(0.45, fovDeg: fovDeg, margin: 1.4)
            }

            // Build the destination pose and glide there (same takeover pattern as
            // the POV flight: capture the CURRENT view camera first).
            if following { setFollowing(false, view: view) }
            let current = view.pointOfView ?? rig.camera
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0
            rig.camera.transform = current.worldTransform
            rig.camera.camera?.fieldOfView = current.camera?.fieldOfView ?? 52
            SCNTransaction.commit()
            view.pointOfView = rig.camera
            view.allowsCameraControl = false

            let tmp = SCNNode()
            tmp.simdPosition = center + dir * dist
            tmp.look(at: SCNVector3(center), up: SCNVector3(up), localFront: SCNVector3(0, 0, -1))
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 1.2
            SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            rig.camera.transform = tmp.transform
            SCNTransaction.completionBlock = { [weak self, weak view] in
                guard let self, let view, self.currentPov == .third else { return }
                view.allowsCameraControl = !self.following
            }
            SCNTransaction.commit()
            Haptics.soft()
        }

        // MARK: - Task-space manipulation (port of RealRobotView's grab IK)

        func setManipulating(_ on: Bool, view: SCNView) {
            manipulating = on
            manipPan?.isEnabled = on
            rig?.setHandlesVisible(on)
            if !on { endGrab() }
        }

        /// Begin ONLY over the intended 3D handles — empty space falls through to
        /// the default orbit camera.
        func gestureRecognizerShouldBegin(_ g: UIGestureRecognizer) -> Bool {
            guard let view else { return true }
            if g === goalPan {
                guard onGoal != nil, currentPov == .third, rig?.goalMarker != nil else { return false }
                return goalHit(at: g.location(in: view), view: view)
            }
            if g === manipPan {
                guard manipulating, currentPov == .third, rig != nil else { return false }
                let p = g.location(in: view)
                if ringHit(at: p, view: view) != nil { return true }
                return grabTarget(at: p, view: view) != nil
            }
            return true
        }

        /// Is the touch on the goal marker (ring / pillar / arrow)?
        private func goalHit(at p: CGPoint, view: SCNView) -> Bool {
            let hits = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.all.rawValue])
            return hits.contains { ($0.node.name ?? "").hasPrefix("goal_") }
        }

        // Goal yaw-ring drag state: the placed goal in scene + odom coordinates.
        var goalPan: UIPanGestureRecognizer?
        var lastGoalScene: simd_float3?
        var lastGoalOdom: (x: Double, y: Double, yaw: Double)?
        private var goalRingLast: CGFloat = 0

        @objc func onGoalPan(_ g: UIPanGestureRecognizer) {
            guard let view, let rig, let marker = rig.goalMarker,
                  var odom = lastGoalOdom else { return }
            let p = g.location(in: view)
            let center = view.projectPoint(SCNVector3(marker.simdWorldPosition))
            switch g.state {
            case .began:
                view.allowsCameraControl = false
                goalRingLast = angleAround(center: center, p: p)
                Haptics.soft()
            case .changed:
                let a = angleAround(center: center, p: p)
                var delta = a - goalRingLast
                if delta > .pi { delta -= 2 * .pi }; if delta < -.pi { delta += 2 * .pi }
                goalRingLast = a
                // Screen angle grows clockwise (y down); floor yaw grows counter-
                // clockwise viewed from above → subtract.
                odom.yaw -= Double(delta)
                lastGoalOdom = odom
                rig.setGoalYaw(Float(odom.yaw))
            case .ended, .cancelled, .failed:
                view.allowsCameraControl = !following && currentPov == .third
                Haptics.rigid()
                onGoal?(odom.x, odom.y, odom.yaw)     // commit the re-aimed goal
            default: break
            }
        }

        private enum DragKind { case none, ring(joint: String), grab }
        private var dragKind: DragKind = .none
        private var grabChain: [String] = []
        private var grabNode: SCNNode?
        private var grabLocal: simd_float3 = .zero
        private var dragDepth: Float = 0.5
        // Palm-ball grab: the ball is a FREE target (not the IK end-effector), so it
        // can be pulled clear of the body and the arm extends after it. Ported from iOS.
        private weak var dragBall: SCNNode?
        private var dragBallRest: simd_float3 = .zero
        private var grabAngles: [String: Float] = [:]
        private var grabJoint: String = ""
        private var grabStartScreen: CGPoint = .zero
        private var ringLastAngle: CGFloat = 0
        private var ringDragAngle: Float = 0
        private var lastIKSolve: TimeInterval = 0
        private weak var hlNode: SCNNode?
        private var hlScale = SCNVector3(1, 1, 1)
        private var hlEmission: Any?

        // ── MoveIt-style plan / preview / execute (green-ball grab, Stiff+Task) ──
        // Ported from iOS v2. Dragging the ball PLANS a target the operator previews
        // (an animated ghost of current→target), and NOTHING reaches the robot until
        // Execute. Accumulates across drags so both arms can be planned + run together.
        var planMode = false
        var planRegion: ControlRegion = .arm
        var executePlanToken = 0
        var cancelPlanToken = 0
        var clearGoalToken = 0
        var debugStageToken = 0
        var onPlanChange: ((ArmPlanPreview?) -> Void)?
        var onBaseGoal: ((Double, Double, Double) -> Void)?
        private var planStart: [String: Double] = [:]
        private var planTarget: [String: Double] = [:]
        private var planChain: [String] = []
        private var planSides: Set<String> = []
        private var pendingTarget: [String: Double] = [:]
        private var pendingChain: [String] = []
        private var pendingSide = "left"
        // Whole-body base half of the plan: the rig-root scene XZ where the base
        // started and where the drag walked it to (nil ⇒ the base never moved).
        // Interpolated in preview/execute alongside the joints, sent as an odom nav
        // goal on Execute.
        private var planBaseStart: SIMD2<Float>? = nil
        private var planBaseTarget: SIMD2<Float>? = nil
        private var pendingBaseTarget: SIMD2<Float>? = nil
        /// While a whole-body plan owns the base, the odom stream must NOT snap the
        /// rig root back — the preview drives it. Cleared when the plan resolves.
        private var baseFrozen = false
        // Region DOFs appended to the wrist IK for the in-progress drag (torso/lift),
        // plus whether the base is a live DOF and where it started (rig-root scene XZ).
        private var grabExtra: [String] = []
        private var grabBase = false
        private var grabBaseStartScene = SIMD2<Float>(0, 0)
        private var grabBaseScene = SIMD2<Float>(0, 0)
        private var previewLink: CADisplayLink?
        private var previewT0: TimeInterval = 0
        private var executeLink: CADisplayLink?
        private var executeT0: TimeInterval = 0
        /// Velocity-capped playback duration for the running Execute.
        private var executeDur: TimeInterval = 0
        var hasPlan: Bool { !planChain.isEmpty || planBaseTarget != nil }

        @objc func onManipPan(_ g: UIPanGestureRecognizer) {
            guard let view, let rig else { return }
            let p = g.location(in: view)
            switch g.state {
            case .began:
                view.allowsCameraControl = false      // the drag owns the camera arbitration
                if let (joint, side) = ringHit(at: p, view: view) {
                    dragKind = .ring(joint: joint)
                    ringDragAngle = Float(currentJoints[joint] ?? 0)
                    if let ball = rig.handle(side: side) {
                        ringLastAngle = angleAround(center: view.projectPoint(SCNVector3(ball.simdWorldPosition)), p: p)
                    }
                    frozenJoints = [joint]
                    let axis = joint.hasSuffix("_5") ? "roll" : (joint.hasSuffix("_6") ? "pitch" : "yaw")
                    setSelected(view.scene?.rootNode.childNode(withName: "ring_\(axis)_\(side)", recursively: true))
                } else if let target = grabTarget(at: p, view: view) {
                    dragKind = .grab
                    startGrab(target, at: p, view: view)
                    let side = grabJoint.hasPrefix("left") ? "left" : "right"
                    setSelected(rig.handle(side: side))
                } else {
                    dragKind = .none
                }
            case .changed:
                switch dragKind {
                case .ring(let joint): stepRing(joint: joint, at: p, view: view, rig: rig)
                case .grab: stepGrab(finger: p, view: view, rig: rig)
                case .none: break
                }
            case .ended, .cancelled, .failed:
                if case .grab = dragKind, forceMode {
                    onPartForce?(grabJoint, [0, 0, 0])       // release the compliant push
                }
                // PLAN MODE: a ball drag OR a wrist-ring twist stages a target but
                // sends nothing — hand it to the preview loop.
                let staged: Bool = { if case .grab = dragKind { return true }
                                     if case .ring = dragKind { return true }; return false }()
                if planMode, !forceMode, !pendingTarget.isEmpty, staged {
                    Haptics.tap()
                    finalizePlan()
                    clearSelected(); restoreDragBall(); grabNode = nil; dragKind = .none
                    view.allowsCameraControl = !following && currentPov == .third
                    return
                }
                if case .grab = dragKind { Haptics.tap() }
                endGrab()
                view.allowsCameraControl = !following && currentPov == .third
            default: break
            }
        }

        private func endGrab() {
            clearSelected()
            restoreDragBall()
            grabChain = []; grabNode = nil
            frozenJoints = []
            dragKind = .none
        }

        /// Snap the free-dragged palm ball back onto the wrist (its rest offset).
        private func restoreDragBall() {
            if let b = dragBall { b.simdPosition = dragBallRest }
            dragBall = nil
        }

        // MARK: - Plan / preview / execute (green-ball safety gate, ported from iOS)

        func stagePlanForTest(side: String, deltas: [String: Double]) {
            pendingSide = side
            pendingChain = deltas.keys.sorted()
            for i in 1...7 { frozenJoints.insert("\(side)_arm_\(i)") }
            pendingTarget = [:]
            for (id, d) in deltas { pendingTarget[id] = ArmPlan.clamp(id, (currentJoints[id] ?? 0) + d) }
            finalizePlan()
        }

        /// Merge the just-ended drag into the accumulated plan (both arms can stack).
        /// The pill/preview only appear once a drag actually MOVED the target past the
        /// micro-drag floor. `currentJoints` holds live telemetry even while frozen.
        private func finalizePlan() {
            let chain = pendingChain
            let target = pendingTarget
            let side = pendingSide
            let baseTarget = pendingBaseTarget
            pendingChain = []; pendingTarget = [:]; pendingBaseTarget = nil
            guard !chain.isEmpty || baseTarget != nil else { return }
            // How far the drag moved the arm, and how far it walked the base — a plan
            // needs a meaningful move in AT LEAST one of them.
            var start: [String: Double] = [:]
            for id in chain { start[id] = currentJoints[id] ?? 0 }
            let jointMove = ArmPlan.maxDelta(from: start, to: target, chain: chain)
            let baseMove = baseTarget.map { simd_length($0 - grabBaseStartScene) } ?? 0
            guard jointMove > 0.02 || baseMove > 0.03 else {
                if baseFrozen, planBaseTarget == nil { baseFrozen = false }
                return
            }
            for id in chain {
                if !planChain.contains(id) { planChain.append(id) }
                planStart[id] = start[id]
                planTarget[id] = ArmPlan.clamp(id, target[id] ?? start[id] ?? 0)
                frozenJoints.insert(id)
            }
            if !chain.isEmpty { planSides.insert(side) }
            // Whole-body base half: keep the ORIGINAL pre-plan base as the start.
            if let bt = baseTarget {
                if planBaseStart == nil { planBaseStart = grabBaseStartScene }
                planBaseTarget = bt
                baseFrozen = true
            }
            startPreviewLoop()
            emitPlanPreview()
            Haptics.rigid()
        }

        private func emitPlanPreview() {
            let delta = ArmPlan.maxDelta(from: planStart, to: planTarget, chain: planChain)
            let baseMove: Float = (planBaseStart != nil && planBaseTarget != nil)
                ? simd_length(planBaseTarget! - planBaseStart!) : 0
            let reason = baseMove > 0.03 ? String(format: "+ base %.1f m", baseMove) : ""
            onPlanChange?(ArmPlanPreview(side: planSides.count > 1 ? "both"
                                        : (planSides.first ?? "left"),
                                        deltaRad: delta, collision: false, reason: reason))
        }

        /// Slide the previewed base along its planned path at phase `t` (same ease as
        /// the joints), so a whole-body preview walks the robot toward the target.
        private func applyBasePreview(_ t: Double, rig: LiveMapRig) {
            guard let s = planBaseStart, let tg = planBaseTarget else { return }
            grabBaseScene = s + (tg - s) * Float(ArmPlan.ease(t))
            applyBaseScene(rig)
        }

        private func startPreviewLoop() {
            for side in planSides { setArmGhost(side, true) }
            setPlanPartsGhost(true)                 // torso / lift / base go translucent too
            previewT0 = 0
            previewLink?.invalidate()
            let l = CADisplayLink(target: self, selector: #selector(previewStep(_:)))
            l.add(to: .main, forMode: .common)
            previewLink = l
        }

        @objc private func previewStep(_ link: CADisplayLink) {
            guard let rig, hasPlan else { return }
            if previewT0 == 0 { previewT0 = link.timestamp }
            let period = 2.8
            let phase = (link.timestamp - previewT0).truncatingRemainder(dividingBy: period)
            let t: Double
            switch phase {
            case ..<1.4: t = phase / 1.4
            case ..<1.7: t = 1
            case ..<2.6: t = 1 - (phase - 1.7) / 0.9
            default:     t = 0
            }
            applyArm(ArmPlan.sample(from: planStart, to: planTarget, chain: planChain, t: t))
            applyBasePreview(t, rig: rig)
        }

        func executePlan() {
            guard hasPlan, rig != nil else { return }
            previewLink?.invalidate(); previewLink = nil
            for side in planSides { setArmGhost(side, false) }
            setPlanPartsGhost(false)                // commit: parts go solid as they run
            // Whole-body base half: drive the base to the planned pose ONCE (a nav
            // goal, not a stream). Scene→odom: odom.x = scene.x, odom.y = −scene.z.
            if let tg = planBaseTarget {
                onBaseGoal?(Double(tg.x), Double(-tg.y), Double(pose?.heading ?? 0))
            }
            executeT0 = 0
            executeLink?.invalidate()
            let l = CADisplayLink(target: self, selector: #selector(executeStep(_:)))
            l.add(to: .main, forMode: .common)
            executeLink = l
            Haptics.rigid()
        }

        @objc private func executeStep(_ link: CADisplayLink) {
            guard hasPlan, let rig else { link.invalidate(); executeLink = nil; return }
            if executeT0 == 0 {
                executeT0 = link.timestamp
                // Size the playback to the MOTION, not to a constant — see
                // ArmPlan.duration. 1.4 s ran a 2.5 rad swing at ~2.7 rad/s peak,
                // ≈3× the velocity budget the rest of the stack respects.
                executeDur = ArmPlan.duration(from: planStart, to: planTarget,
                                              chain: planChain)
            }
            let t = min(1, (link.timestamp - executeT0) / max(1e-3, executeDur))
            let cfg = ArmPlan.sample(from: planStart, to: planTarget, chain: planChain, t: t)
            applyArm(cfg)
            applyBasePreview(t, rig: rig)
            if !cfg.isEmpty { onJointsChange?(cfg) }   // the ONLY joint send in the flow
            if t >= 1 { link.invalidate(); executeLink = nil; clearPlan(unfreeze: true); Haptics.tap() }
        }

        func cancelPlan() {
            previewLink?.invalidate(); previewLink = nil
            executeLink?.invalidate(); executeLink = nil
            if hasPlan {
                for side in planSides { setArmGhost(side, false) }
                setPlanPartsGhost(false)
            }
            clearPlan(unfreeze: true)
        }

        private func clearPlan(unfreeze: Bool) {
            planChain = []; planTarget = [:]; planStart = [:]; planSides = []
            pendingChain = []; pendingTarget = [:]
            planBaseStart = nil; planBaseTarget = nil; pendingBaseTarget = nil
            if unfreeze { frozenJoints = []; baseFrozen = false }
            onPlanChange?(nil)
        }

        /// Apply a planned config to the live rig. Revolute joints rotate; the LIFT's
        /// prismatic stages slide along their axis (an upper/whole-body plan drives
        /// them). Plan joints (arm/torso/lift) all have MuJoCo ref = 0, so no ref
        /// subtraction here (unlike the telemetry `apply(joints:)` path).
        private func applyArm(_ cfg: [String: Double]) {
            guard let rig else { return }
            for (id, v) in cfg {
                guard let j = rig.robotJoints[id] else { continue }
                if j.type == "prismatic" {
                    j.node.simdPosition = j.axis * Float(v)
                } else {
                    j.node.simdOrientation = simd_quatf(angle: Float(v), axis: simd_normalize(j.axis))
                }
            }
        }

        private func setArmGhost(_ side: String, _ ghost: Bool) {
            guard let rig else { return }
            let o: CGFloat = ghost ? 0.45 : 1.0
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.25
            for i in 1...7 { rig.robotJoints["\(side)_arm_\(i)"]?.node.opacity = o }
            rig.robot.childNode(withName: RobotRig.palmName(side), recursively: true)?.opacity = o
            rig.handle(side: side)?.opacity = ghost ? 0.7 : 1.0
            SCNTransaction.commit()
        }

        /// Ghost a link's OWN mesh (its direct geometry children) WITHOUT touching its
        /// subtree — so ghosting the torso / base doesn't dim the arms or head riding
        /// on it. Same technique as the head-shell dissolve.
        private func setLinkShellGhost(_ node: SCNNode?, _ ghost: Bool) {
            guard let node else { return }
            let o: CGFloat = ghost ? 0.45 : 1.0
            for child in node.childNodes where child.geometry != nil { child.opacity = o }
        }

        /// Ghost the PROXIMAL parts a region plan actually moves — the torso and lift
        /// (upper body) and the base (whole body) — so they read as translucent
        /// preview just like the dragged arm. Only the parts IN the plan ghost.
        private func setPlanPartsGhost(_ ghost: Bool) {
            guard let rig else { return }
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.25
            if planChain.contains("torso") {
                setLinkShellGhost(rig.robotJoints["torso"]?.node, ghost)
            }
            if planChain.contains("lift_lower") || planChain.contains("lift_upper") {
                setLinkShellGhost(rig.robotJoints["lift_lower"]?.node, ghost)
                setLinkShellGhost(rig.robotJoints["lift_upper"]?.node, ghost)
            }
            if planBaseTarget != nil {
                setLinkShellGhost(rig.robot.childNode(withName: "mabel_chassis", recursively: true), ghost)
                setLinkShellGhost(rig.robot.childNode(withName: "lift_base", recursively: true), ghost)
            }
            SCNTransaction.commit()
        }

        /// What's under the finger: the palm ball (wrist grab) or a grabbable link.
        private func grabTarget(at p: CGPoint, view: SCNView)
            -> (node: SCNNode, world: simd_float3, chain: [String], joint: String)? {
            guard let rig else { return nil }
            let hits = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.all.rawValue])
            for h in hits where h.node.name == "handle" {
                let side = (h.node === rig.handle(side: "left")) ? "left" : "right"
                return (h.node, simd_float3(h.node.simdWorldPosition),
                        (1...4).map { "\(side)_arm_\($0)" }, "\(side)_hand")
            }
            guard let closest = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.closest.rawValue]).first
            else { return nil }
            var node: SCNNode? = closest.node
            while let n = node {
                if let name = n.name, LiveMapView.grabbableJoints.contains(name),
                   let art = rig.robotJoints[name]?.node {
                    let chain: [String]
                    if name == "torso" { chain = ["torso"] }
                    else {
                        let side = name.hasPrefix("left") ? "left" : "right"
                        let k = Int(name.split(separator: "_").last ?? "1") ?? 1
                        chain = (1...max(1, min(k, 4))).map { "\(side)_arm_\($0)" }
                    }
                    return (art, simd_float3(closest.worldCoordinates), chain, name)
                }
                node = n.parent
            }
            return nil
        }

        private func startGrab(_ t: (node: SCNNode, world: simd_float3, chain: [String], joint: String),
                               at p: CGPoint, view: SCNView) {
            grabChain = t.chain
            grabJoint = t.joint
            grabStartScreen = p
            dragBall = nil
            // PALM-BALL grab: decouple the free target (ball) from the end-effector
            // (wrist) — the arm reaches toward wherever the ball is dragged.
            if t.node.name == "handle", let parent = t.node.parent {
                dragBall = t.node
                dragBallRest = t.node.simdPosition
                grabNode = parent
                grabLocal = t.node.simdPosition
            } else {
                grabNode = t.node
                grabLocal = t.node.simdConvertPosition(t.world, from: nil)
            }
            dragDepth = view.projectPoint(SCNVector3(t.world)).z
            grabAngles = [:]
            for id in t.chain { grabAngles[id] = Float(currentJoints[id] ?? 0) }
            // Freeze the whole dragged arm so telemetry doesn't fight the prediction.
            frozenJoints = Set(t.chain)
            if let side = t.joint.split(separator: "_").first, t.joint != "torso" {
                for i in 1...7 { frozenJoints.insert("\(side)_arm_\(i)") }
            }

            // REGION DOFs: widen the wrist IK past the arm when planning the upper /
            // whole body. Only a palm-ball grab drives torso/lift/base — a link grab
            // stays local. Off in Soft (a compliant push, not IK).
            grabExtra = []
            grabBase = false
            if planMode, !forceMode, t.node.name == "handle" {
                switch planRegion {
                case .arm: break
                case .upperBody:
                    grabExtra = LiveMapView.upperBodyDofs
                case .wholeBody:
                    grabExtra = LiveMapView.upperBodyDofs
                    grabBase = true
                }
                for id in grabExtra {
                    grabAngles[id] = Float(currentJoints[id] ?? 0)
                    frozenJoints.insert(id)
                }
                if grabBase, let rig {
                    // Keep the base origin fixed across re-drags: once a base plan
                    // exists, its start IS the true pre-plan pose (the rig root is
                    // mid-preview now), so re-anchor to it rather than the live node.
                    grabBaseStartScene = planBaseStart
                        ?? SIMD2(rig.robot.simdPosition.x, rig.robot.simdPosition.z)
                    grabBaseScene = grabBaseStartScene
                    baseFrozen = true
                }
            }
            Haptics.soft()
        }

        private func stepGrab(finger p: CGPoint, view: SCNView, rig: LiveMapRig) {
            guard let node = grabNode, !grabChain.isEmpty else { return }

            // SOFT (compliance): the drag is an external PUSH in camera axes.
            if forceMode {
                guard let cam = view.pointOfView?.simdWorldTransform else { return }
                let right = simd_normalize(simd_make_float3(cam.columns.0))
                let up    = simd_normalize(simd_make_float3(cam.columns.1))
                let k: Float = 2.0
                let f = right * Float(p.x - grabStartScreen.x) * k
                      - up    * Float(p.y - grabStartScreen.y) * k
                onPartForce?(grabJoint, [Double(f.x), Double(f.y), Double(f.z)])
                return
            }

            // Throttle the DLS solve to ~60 Hz (main thread is shared with the WS path).
            let now = CACurrentMediaTime()
            if now - lastIKSolve < 1.0 / 60.0 { return }
            lastIKSolve = now

            let target = simd_float3(view.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), dragDepth)))
            dragBall?.simdWorldPosition = target      // free ball follows the finger

            // Build the IK DOF list: the arm chain (always), then the region's
            // proximal DOFs (torso + lift for upper/whole body), then the mobile base
            // (whole body). Proximal DOFs carry a small WEIGHT so the well-damped
            // solve spends the arm first and only recruits them once the arm can't
            // close the error — which is exactly "torso/lift/base move only if needed".
            let dofs = buildIKDofs(rig: rig)
            guard !dofs.isEmpty else { return }
            let lambda: Float = 0.14
            for _ in 0..<4 {
                let ee = node.simdConvertPosition(grabLocal, to: nil)
                let err = target - ee
                let errLen = simd_length(err)
                if errLen < 0.0015 { break }
                // REGION GATE: the proximal DOFs join the solve ONLY while the arm is
                // still short by more than `proximalEngageDist`. A reachable target
                // keeps the residual under it, so torso/lift/base never move; once the
                // arm saturates they engage to extend the reach. (Weight alone lets a
                // small base contribution creep in over a long reachable drag.)
                let useProximal = errLen > LiveMapView.proximalEngageDist
                // Weighted DLS: scaling column i by √wᵢ and recovering δθᵢ = wᵢ·(cᵢ·y)
                // keeps J·δθ ≈ err while distributing effort by weight (arm eager,
                // torso/lift/base reluctant). With every weight = 1 (arm-only) this is
                // byte-identical to the original solve.
                var cols = [simd_float3](repeating: .zero, count: dofs.count)
                var jjt = matrix_float3x3(diagonal: SIMD3<Float>(repeating: lambda * lambda))
                for (i, d) in dofs.enumerated() {
                    if d.proximal && !useProximal { continue }   // column stays zero → inert
                    let c = ikColumn(d, ee: ee)
                    cols[i] = c
                    let cw = c * sqrt(d.weight)
                    jjt += matrix_float3x3(rows: [cw * cw.x, cw * cw.y, cw * cw.z])
                }
                let y = jjt.inverse * err
                for (i, d) in dofs.enumerated() {
                    if d.proximal && !useProximal { continue }
                    let dv = max(-d.iterClamp, min(d.iterClamp, d.weight * simd_dot(cols[i], y)))
                    applyIKStep(d, delta: dv, rig: rig)
                }
            }

            // ONE batched command per solve: the arm is always staged as a unit; a
            // region DOF is staged ONLY if it actually moved past a floor, so a target
            // the arm reached on its own leaves "torso/lift not moving" out of the plan.
            var batch: [String: Double] = [:]
            for id in grabChain {
                var a = grabAngles[id] ?? 0
                if let s = RobotModel.spec(id) { a = max(Float(s.lower), min(Float(s.upper), a)) }
                batch[id] = Double(a)
            }
            for id in grabExtra {
                let a = Double(grabAngles[id] ?? 0)
                let floor = (id == "torso") ? 0.02 : 0.005
                if abs(a - (currentJoints[id] ?? 0)) > floor { batch[id] = ArmPlan.clamp(id, a) }
            }
            // PLAN MODE (the safety gate): the drag has already advanced the rig for
            // a live preview, but we STAGE the target instead of sending it. Nothing
            // reaches the robot until Execute. Legacy mode streams as before.
            if planMode {
                pendingTarget = batch
                pendingChain = Array(batch.keys)
                pendingSide = grabJoint.hasPrefix("left") ? "left"
                    : (grabJoint.hasPrefix("right") ? "right" : pendingSide)
                if grabBase, simd_length(grabBaseScene - grabBaseStartScene) > 0.03 {
                    pendingBaseTarget = grabBaseScene
                } else {
                    pendingBaseTarget = nil
                }
            } else {
                onJointsChange?(batch)
            }
        }

        // MARK: - Region-aware IK DOFs (arm · torso · lift · base)

        /// One degree of freedom the green-ball IK may move. Revolute/prismatic wrap a
        /// rig joint; base X/Z translate the whole rig root in the scene ground plane.
        private struct IKDof {
            enum Kind { case revolute, prismatic, baseX, baseZ }
            let kind: Kind
            let id: String          // joint id ("" for base)
            let node: SCNNode?      // nil for base
            let axis: simd_float3   // joint LOCAL axis, or the base world unit vector
            let weight: Float       // eagerness in the weighted DLS (arm 1, proximal small)
            let iterClamp: Float    // max step per iteration (rad or m)
            let lower: Float
            let upper: Float
            let proximal: Bool      // torso/lift/base — gated on the arm saturating
        }

        private func buildIKDofs(rig: LiveMapRig) -> [IKDof] {
            var dofs: [IKDof] = []
            for id in grabChain {
                guard let j = rig.robotJoints[id] else { continue }
                let s = RobotModel.spec(id)
                dofs.append(IKDof(kind: .revolute, id: id, node: j.node, axis: j.axis,
                                  weight: 1.0, iterClamp: 0.10,
                                  lower: Float(s?.lower ?? -.pi), upper: Float(s?.upper ?? .pi),
                                  proximal: false))
            }
            for id in grabExtra {
                guard let j = rig.robotJoints[id] else { continue }
                let s = RobotModel.spec(id)
                let prism = (j.type == "prismatic")
                dofs.append(IKDof(kind: prism ? .prismatic : .revolute, id: id, node: j.node, axis: j.axis,
                                  weight: id == "torso" ? 0.20 : 0.15,
                                  iterClamp: prism ? 0.012 : 0.10,
                                  lower: Float(s?.lower ?? (prism ? 0 : -.pi)),
                                  upper: Float(s?.upper ?? (prism ? 0.3175 : .pi)),
                                  proximal: true))
            }
            if grabBase {
                let big = Float.greatestFiniteMagnitude
                dofs.append(IKDof(kind: .baseX, id: "", node: nil, axis: SIMD3(1, 0, 0),
                                  weight: 0.05, iterClamp: 0.025, lower: -big, upper: big, proximal: true))
                dofs.append(IKDof(kind: .baseZ, id: "", node: nil, axis: SIMD3(0, 0, 1),
                                  weight: 0.05, iterClamp: 0.025, lower: -big, upper: big, proximal: true))
            }
            return dofs
        }

        /// The DOF's Jacobian column: how the end-effector moves per unit of this DOF.
        /// Revolute → ω×r about the world axis; prismatic → the world axis direction;
        /// base → the world unit vector (translating the root shifts the ee by it).
        private func ikColumn(_ d: IKDof, ee: simd_float3) -> simd_float3 {
            switch d.kind {
            case .revolute:
                let wt = d.node!.simdWorldTransform
                let rot = simd_float3x3(simd_make_float3(wt.columns.0),
                                        simd_make_float3(wt.columns.1),
                                        simd_make_float3(wt.columns.2))
                return simd_cross(simd_normalize(rot * d.axis), ee - simd_make_float3(wt.columns.3))
            case .prismatic:
                let wt = d.node!.simdWorldTransform
                let rot = simd_float3x3(simd_make_float3(wt.columns.0),
                                        simd_make_float3(wt.columns.1),
                                        simd_make_float3(wt.columns.2))
                return simd_normalize(rot * d.axis)
            case .baseX: return SIMD3(1, 0, 0)
            case .baseZ: return SIMD3(0, 0, 1)
            }
        }

        /// Advance one DOF by `delta`, clamped to its limit, and reflect it on the rig
        /// (rotate / slide the joint, or walk the rig root for the base).
        private func applyIKStep(_ d: IKDof, delta: Float, rig: LiveMapRig) {
            switch d.kind {
            case .revolute:
                let a = max(d.lower, min(d.upper, (grabAngles[d.id] ?? 0) + delta))
                grabAngles[d.id] = a
                d.node!.simdOrientation = simd_quatf(angle: a, axis: simd_normalize(d.axis))
            case .prismatic:
                let a = max(d.lower, min(d.upper, (grabAngles[d.id] ?? 0) + delta))
                grabAngles[d.id] = a
                d.node!.simdPosition = d.axis * a
            case .baseX:
                grabBaseScene.x += delta; applyBaseScene(rig)
            case .baseZ:
                grabBaseScene.y += delta; applyBaseScene(rig)
            }
        }

        /// Park the rig root at the previewed base position (scene ground plane; keep
        /// the deck height). `baseFrozen` keeps the odom stream from snapping it back.
        private func applyBaseScene(_ rig: LiveMapRig) {
            var pos = rig.robot.simdPosition
            pos.x = grabBaseScene.x
            pos.z = grabBaseScene.y
            rig.robot.simdPosition = pos
        }

        private func ringHit(at p: CGPoint, view: SCNView) -> (joint: String, side: String)? {
            let hits = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.all.rawValue])
            for h in hits {
                guard let name = h.node.name, name.hasPrefix("ring_"), !h.node.isHidden else { continue }
                let parts = name.split(separator: "_")   // ring_<axis>_<side>
                guard parts.count == 3 else { continue }
                let side = String(parts[2])
                switch parts[1] {
                case "roll":  return ("\(side)_arm_5", side)
                case "pitch": return ("\(side)_arm_6", side)
                default:      return ("\(side)_arm_7", side)
                }
            }
            return nil
        }

        private func angleAround(center: SCNVector3, p: CGPoint) -> CGFloat {
            atan2(p.y - CGFloat(center.y), p.x - CGFloat(center.x))
        }

        private func stepRing(joint: String, at p: CGPoint, view: SCNView, rig: LiveMapRig) {
            let side = joint.hasPrefix("left") ? "left" : "right"
            guard let ball = rig.handle(side: side), let j = rig.robotJoints[joint] else { return }
            let center = view.projectPoint(SCNVector3(ball.simdWorldPosition))
            let a = angleAround(center: center, p: p)
            var delta = a - ringLastAngle
            if delta > .pi { delta -= 2 * .pi }; if delta < -.pi { delta += 2 * .pi }
            ringLastAngle = a
            var ang = ringDragAngle + Float(delta)
            if let s = RobotModel.spec(joint) { ang = max(Float(s.lower), min(Float(s.upper), ang)) }
            ringDragAngle = ang
            j.node.simdOrientation = simd_quatf(angle: ang, axis: simd_normalize(j.axis))
            // PLAN MODE: a wrist-ring twist stages into the plan (previewed, executed
            // on the button) like the ball drag — nothing moves until Execute.
            if planMode {
                pendingTarget = [joint: Double(ang)]
                pendingChain = [joint]
                pendingSide = side
            } else {
                onJointChange?(joint, Double(ang))
            }
        }

        /// Light a node as SELECTED (white emission + scale pulse); nil clears.
        private func setSelected(_ node: SCNNode?) {
            if hlNode !== node { clearSelected() }
            guard let node, hlNode == nil else { return }
            hlNode = node
            hlScale = node.scale
            let mat = node.geometry?.firstMaterial
            hlEmission = mat?.emission.contents
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.12
            node.scale = SCNVector3(hlScale.x * 1.4, hlScale.y * 1.4, hlScale.z * 1.4)
            mat?.emission.contents = UIColor.white
            SCNTransaction.commit()
            Haptics.rigid()
        }

        private func clearSelected() {
            guard let node = hlNode else { return }
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.12
            node.scale = hlScale
            node.geometry?.firstMaterial?.emission.contents = hlEmission
            SCNTransaction.commit()
            hlNode = nil
        }

        @objc func onFollowPinch(_ g: UIPinchGestureRecognizer) {
            guard following else { return }
            follow.zoom(by: Float(g.scale)); g.scale = 1
        }

        /// Re-center while following: reset orbit offsets + zoom (stays locked on).
        func resetFollowCam() { follow.reset() }
        /// Fit while following: pull the chase cam out wide (stays locked on).
        func fitFollowCam() { follow.distance = 9.5 }

        /// One-finger drag while following → orbit around the robot (face it / go
        /// top-down). Two-finger pinch still zooms; both keep the robot centred.
        @objc func onFollowPan(_ g: UIPanGestureRecognizer) {
            guard following else { return }
            let t = g.translation(in: g.view)
            follow.orbit(dx: Float(t.x), dy: Float(t.y))
            g.setTranslation(.zero, in: g.view)
        }

        /// Tap → ground point → Nav2 goal. The mesh + odom live in the ROS `odom`
        /// frame, rebased to SceneKit as (x, z, −y); we invert that to send the
        /// goal back in `odom`. Heading faces from the robot toward the goal so the
        /// front-facing MPPI drives there nose-first.
        @objc func onTap(_ g: UITapGestureRecognizer) {
            guard let cb = onGoal, let v = view, let rig = rig else { return }
            let p = g.location(in: v)
            // Safety band around the floating chrome: a tap that just misses a
            // button (top control rows, bottom pill/joystick clusters, the nav
            // rail) must NOT plant a Nav2 goal behind it.
            let b = v.bounds
            guard p.y > goalExclusion.top,
                  p.y < b.height - goalExclusion.bottom,
                  p.x > goalExclusion.left,
                  p.x < b.width - goalExclusion.right else { return }
            // Intersect the tap ray with the floor plane (SceneKit y = 0).
            let near = v.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), 0))
            let far  = v.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), 1))
            let dy = far.y - near.y
            guard abs(dy) > 1e-5 else { return }
            let t = -near.y / dy
            guard t > 0 else { return }
            let gx = near.x + (far.x - near.x) * t
            let gz = near.z + (far.z - near.z) * t

            // SceneKit ground (gx, 0, gz) → ROS odom: x = gx, y = −gz.
            let goalX = Double(gx), goalY = Double(-gz)
            let rp = rig.robot.simdPosition
            let robX = Double(rp.x), robY = Double(-rp.z)
            let yaw = atan2(goalY - robY, goalX - robX)

            // Remember the placement so the yaw ring can re-aim + re-send it.
            lastGoalScene = simd_float3(gx, 0, gz)
            lastGoalOdom = (goalX, goalY, yaw)
            rig.apply(goal: SCNVector3(gx, 0, gz), yaw: Float(yaw))   // immediate feedback
            navState?.setGoal(CGPoint(x: CGFloat(gx), y: CGFloat(gz)))   // 2D map echo
            Haptics.rigid()
            cb(goalX, goalY, yaw)
        }

        // The store is owned elsewhere (RobotController); release our hooks so a
        // torn-down rig is never driven by a stray callback.
        func detach() {
            projector.detach()
            jointLink?.invalidate(); jointLink = nil
            followLink?.invalidate(); followLink = nil
            store?.onMesh = nil
            store?.onPose = nil
            store?.onPath = nil
            store?.onMap = nil
            store?.onCloud = nil
            rig = nil
            navState = nil
        }
    }
}
