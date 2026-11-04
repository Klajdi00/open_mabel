//
//  RobotSpaceView.swift
//  MABEL Mobile App
//
//  The unified Robot space — v2's replacement for the separate Pilot /
//  Manipulate / Navigate tabs. ONE full-bleed 3D viewer is both the visualizer
//  and the controller (Tesla / Roborock style: the hero live view + a few
//  floating pills + a pull-up console for everything else):
//
//    • The live SLAM map (mesh + occupancy + Nav2 route) with the articulated
//      MABEL at its live odom pose — tap the floor to send a Nav2 goal.
//    • 3rd ↔ 1st person: the viewer camera flies from WHEREVER you orbited it
//      into the robot's head camera (and back); live FPV crossfades in.
//    • Teleop: Start Teleop + a Drive · Joint · Task space picker. Drive
//      (base joysticks) is landscape-only; Joint (arm joysticks) and Task
//      (grab the robot's arms / palm balls / wrist rings in the viewer) work
//      in BOTH orientations. Stiff = IK commands; Soft = compliant pushes.
//    • The pull-up CONSOLE (bottom grabber, like the camera sheet): status,
//      layer toggles (mesh / cloud / grid / minimap / Cam 3D), fine viz
//      controls (projection size, flat vs lens-curved, cloud density) and
//      the live camera tiles.
//
//  The chrome (status bar, nav rail / bottom bar) is owned by ContentView;
//  everything here floats over the scene.
//

import SwiftUI
import Combine
import UIKit
import Charts

struct RobotSpaceView: View {
    /// 1st-person viewport insets: the FPV image sits BELOW the top controls
    /// and inside side margins, and the head-image dive lands at exactly this
    /// rect — chrome never overlaps the picture, and the fade sizes match.
    static let povInsetTop: CGFloat = 148
    static let povInsetSide: CGFloat = 72
    static let povInsetBottom: CGFloat = 34
    /// Landscape is fully immersive (no nav rail) — this leaves the Robot space.
    var onBack: (() -> Void)? = nil
    /// App navigation (Robot · Autonomy · Settings) — rendered as the LEFT-MOST
    /// column of this tab's leading ornament, beside the viewer buttons.
    var navSelection: Binding<AppSection>? = nil
    /// TRUE while the Robot tab is frontmost. The view stays mounted on other
    /// tabs (warm rig + SceneKit scene + streams — zero reload on return), but
    /// its chrome and ornaments hide.
    private var isActiveTab: Bool { navSelection?.wrappedValue != nil ? navSelection!.wrappedValue == .robot : true }
    @EnvironmentObject private var controller: RobotController
    @Environment(\.bottomBarInset) private var bottomBarInset
    @Environment(\.railLeadingInset) private var railLeadingInset
    @Environment(\.isLandscape) private var isLandscape
    /// visionOS: opens aux windows / the volumetric robot space.
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openImmersiveSpace) private var openImmersive
    @Environment(\.dismissImmersiveSpace) private var dismissImmersive
    /// Mini live rig floating bottom-right in 1st person — the SAME RealityKit
    /// stage as the v1 pilot column, gaze-highlightable, sharing live telemetry.
    @StateObject private var miniStage = RobotStage(scale: 0.13)

    // Map source: live SLAM or a saved map streamed back from the robot.
    @State private var selected: SLAMRoom = .live
    @State private var showSavedMaps = false

    // Viewport camera controls.
    @State private var recenterToken = 0
    @State private var fitToken = 0
    @State private var following = false

    // 3rd ↔ 1st person. DEBUG: `debugPov` user default boots straight into 1st
    // (launch argv is dropped on prewarmed launches — defaults are reliable).
    @State private var pov: PovMode = {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-pov"), i + 1 < args.count,
           let m = PovMode(rawValue: args[i + 1]) { return m }
        if let s = UserDefaults.standard.string(forKey: "debugPov"),
           let m = PovMode(rawValue: s) { return m }
        #endif
        return .third
    }()
    @State private var fpvVisible = false     // video layer opacity gate
    /// What the SceneKit path renders during POV transitions: it mounts wide
    /// (.third) and FLIES to .first so 3rd→1st reads as one continuous zoom.
    @State private var scenePov: PovMode = .first
    @State private var povAnimating = false
    @State private var stagePresetInstant = false
    @State private var sceneInstantPov = false
    /// Whole-body safety gate: Start Teleop in Task space (or Start Hand
    /// Tracking) first confirms the operator is clear to be tracked.
    @State private var showWholeBodyConfirm = false
    /// The guided arm-extension calibration pop-up (CalibrationFlowView).
    @State private var showCalibrationFlow = false
    @State private var viewSize: CGSize = .zero
    // Quick-action side panels — ornaments EXTENDING the window to the right
    // (cameras rail · joint bars top-right · 2-min trends bottom-right), so
    // they never overlap the 3D viewer.
    @State private var showCamsPanel = false
    @State private var showBarsPanel = false
    @State private var showSeriesPanel = false
    /// The two CONTROL toggles on the trailing rail (2026-10-27, operator: "those joysticks
    /// should always be persistent … even under joint and task space … all the controls
    /// should be clear and be toggled"). The drive pad's key is the dock's own, so the rail
    /// toggle, the dock's Hide and the dome's copy are one switch.
    @AppStorage(DrivePadDock.visibleKey) private var drivePadShown = true
    @AppStorage("jointPanelsShown") private var jointPanelsShown = true
    @State private var fpvLive = false        // head stream actually delivering frames

    // Teleop. Drive (base) is LANDSCAPE-ONLY; Joint / Task work everywhere.
    @State private var teleopActive = false
    @State private var space: TeleopSpace = .drive
    @State private var feel: ControllerType = .impedance   // Stiff | Soft
    /// Control-tuning sheet (gravity comp · retargeting profile). A SHEET rather
    /// than more toolbar chrome: this row already carries the space picker, the
    /// primary verb and the Stiff|Soft law, and on visionOS a protruding 3D stage
    /// makes crowded window-space rows especially easy to occlude.
    @State private var translate: CGVector = .zero
    @State private var yaw: Double = 0
    @State private var joyL: CGVector = .zero
    @State private var joyR: CGVector = .zero
    @State private var pitchL: Double = 0
    @State private var pitchR: Double = 0
    // TRUE-3D pads (DJI-stick metaphor): 3-axis normalized rates.
    @State private var padC: SIMD3<Float> = .zero    // neck / torso pad
    @State private var padL: SIMD3<Float> = .zero    // task: left wrist
    @State private var padR: SIMD3<Float> = .zero    // task: right wrist
    @State private var gripL: Double = 0
    @State private var gripR: Double = 0
    // Joint-space module focus: which module the controls (and the camera) are on.
    @State private var module: ModuleFocus = .armRight
    /// Let the BODY MODULE / CONTROL REGION selection highlights SLIDE between rows.
    @Namespace private var moduleNS
    @Namespace private var regionNS
    @State private var focusToken = 0
    /// Control REGION for task space + Soft mode: arm / upper body / whole body.
    /// The controller + server already implement all three (arm→wrist retarget,
    /// upper→+torso+lift, whole→+base). The app just declares which one.
    @State private var region: ControlRegion = .arm
    /// WHICH teleop method Task space uses (operator's spec, 2026-07-16). Both
    /// are 3rd-person controls; they differ in what moves the robot:
    ///   .wrist     — drag the GREEN PALM BALL, the arm stages a PLAN (ghosted
    ///                preview loop), Execute commits. Nothing reaches the robot
    ///                until Execute, so the plan gate IS the safety gate and no
    ///                Start Teleop / body tracking is needed. iOS v2 parity.
    ///   .wholeBody — ARKit head + arms + fingers retarget live through the
    ///                immersive space (the Vision Pro superpower); gated behind
    ///                Start Teleop → safety confirm → arm-extension calibration.
    @State private var taskMode: TaskMode = .wrist
    /// A staged green-ball arm plan (nil = none) — drives the Execute/Cancel pills.
    @State private var armPlan: ArmPlanPreview? = nil
    @State private var executePlanToken = 0
    @State private var cancelPlanToken = 0
    /// Bumped to drop a placed-but-never-dispatched nav goal marker.
    @State private var clearGoalToken = 0
    @State private var debugStageToken = 0
    @State private var debugStageSide = "left"
    // 3D-stage content framing (drive = wide iso, task = iso, joint = module).
    @State private var stagePreset: RobotSpace3DStage.StagePreset = .iso
    @State private var presetToken = 0
    @State private var rollDir: Double = 0     // neck roll hold-button direction
    private let tick = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()
    private let armRate = 1.4

    // Visualization state — driven by the left rail's layer buttons.
    // The offline standby card is TRANSIENT — it announces, then gets out of
    // the way (map selection stays one tap away on the pill / in the console).
    @State private var standbyDismissed = false
    @State private var meshOn = true
    @State private var cloudOn = true
    @State private var gridOn = true
    @State private var minimapOn = true
    // Per-camera Cam-3D toggles (Layers popover). Default: head cam only.
    @State private var camHeadOn = true
    @State private var camWristLOn = false
    @State private var camWristROn = false
    @State private var camBaseOn = false
    // Virtual floor ON by default (operator 2026-07-18): without it the robot
    // reads as hovering, and the occupancy map was the only "ground". Drives BOTH
    // grounds — the immersive RealityKit stage's (SensorLayers.buildFloor) and the
    // windowed SceneKit viewer's (GroundPlane.surface) — so the chip means the same
    // thing wherever the operator is looking from.
    @State private var floorOn = true
    /// The 1 m / 5 m reference lines ON that floor. Its own layer, because the far
    /// field is the grid's job and the near field is the wash's: over a fully
    /// mapped room you want the cadence without the wash lifting the SLAM mesh.
    @State private var floorGridOn = true
    /// MABEL's own mesh. Sensor layers in the same subtree keep drawing, so turning
    /// the robot off reveals the cloud / LiDAR / TF picture instead of blanking it.
    @State private var robotOn = true
    private var camStreamSet: Set<String> {
        var set = Set<String>()
        if camHeadOn { set.insert("main") }
        if camWristLOn { set.insert("wrist_left") }
        if camWristROn { set.insert("wrist_right") }
        if camBaseOn { set.insert("front") }
        return set
    }
    private var camProjections: Bool { !camStreamSet.isEmpty }
    @State private var tfOn = false
    @State private var laserOn = true
    @State private var camProjSize = 0.3
    @State private var camFlat = true
    @State private var cloudDensity: CloudDensity = .full

    /// Teleop control space (which controls the joysticks / gestures drive).
    enum TeleopSpace: String, CaseIterable, Identifiable {
        case drive, joint, task
        var id: String { rawValue }
        var title: String {
            switch self {
            case .drive: return "Drive"
            case .joint: return "Joint"
            case .task:  return "Task"
            }
        }
        var icon: String {
            switch self {
            case .drive: return "dot.arrowtriangles.up.right.down.left.circle"
            case .joint: return "slider.horizontal.3"
            case .task:  return "hand.draw.fill"
            }
        }
    }

    /// The two Task-space teleop methods (see `taskMode`).
    enum TaskMode: String, CaseIterable, Identifiable {
        case wrist, wholeBody
        var id: String { rawValue }
        var title: String { self == .wrist ? "Wrist" : "Whole Body" }
        var icon: String { self == .wrist ? "hand.point.up.left.fill" : "figure.arms.open" }
    }

    /// Cam 3D image surface: rectified pinhole plane vs through-the-lens shell.
    enum ProjSurface: String, CaseIterable, Identifiable {
        case rectified, lens
        var id: String { rawValue }
        var title: String { self == .rectified ? "Rectified" : "Lens Curve" }
        var icon: String { self == .rectified ? "rectangle" : "circle.and.line.horizontal" }
    }

    /// Live-cloud render density (every Nth point).
    enum CloudDensity: Int, CaseIterable, Identifiable {
        case full = 1, half = 2, quarter = 4, tenth = 10
        var id: Int { rawValue }
        var title: String {
            switch self {
            case .full: return "Full"
            case .half: return "1/2"
            case .quarter: return "1/4"
            case .tenth: return "1/10"
            }
        }
    }

    private var store: LiveMapStore { controller.liveMap }

    // MARK: - Which renderer owns 3rd person
    //
    // 3rd person draws the TRUE-3D RealityKit stage — INCLUDING Task space
    // (operator directive 2026-07-18: "Task must be 3D like Drive/Joint", not
    // the flat SceneKit window). The green-ball planner now lives ON the stage
    // (StagePlanner: handles + rings + weighted-DLS IK + ghost preview +
    // Execute), so the wrist takeover that swapped in the 2D SceneKit viewer
    // for Task·Wrist is GONE. The SceneKit path still takes over for SOFT
    // compliant pushes (its grab-force flow was never ported) and still owns
    // 1st person (camera flight + FPV crossfade).

    /// THE RULE (iOS parity, 2026-07-17): torque OFF → this is a VIEWER; torque ON
    /// → a controller. With the e-stop engaged the motors are dead, so every
    /// control surface — the space/feel toggles, Start Teleop, joysticks, the
    /// module/region panels, GO, Execute — is a button that cannot do the thing it
    /// depicts. They hide until TORQUE ON; what stays is everything that OBSERVES
    /// (SLAM/map, cameras, layers, POV, telemetry, the volumetric windows).
    private var torqueOn: Bool { !controller.estopEngaged }

    /// Joint space's module + slider panels. SOFT suppresses them: there the robot
    /// floats under gravity compensation and `RobotController` refuses every joint
    /// command, so the sliders would be inert. Torque off suppresses them too.
    private var jointPanelsVisible: Bool {
        torqueOn && space == .joint && feel == .impedance && jointPanelsShown
    }

    /// CONTROL REGION rides the trailing ornament wherever a region-scoped control
    /// is live: Task space (both methods) and Soft.
    private var regionPanelVisible: Bool {
        torqueOn && pov == .third && (space == .task || feel == .compliance)
    }

    /// Task · Wrist · Stiff · 3rd person — the green-ball PLANNER is live:
    /// drag a palm ball, the arm stages a plan and animates a ghosted preview,
    /// Execute commits. Needs no Start Teleop: nothing reaches the robot until
    /// Execute, so the plan gate is the safety gate. Torque off disables it — the
    /// robot can't move, so there is nothing to plan.
    private var wristPlanning: Bool {
        torqueOn && pov == .third && space == .task && taskMode == .wrist && feel == .impedance
    }
    /// Soft: the same grabs become COMPLIANT PUSHES (no plan) — still SceneKit
    /// (the compliant grab-force flow lives only there).
    /// TASK SPACE ONLY. `space == .task` is the clause that used to be missing:
    /// without it the green palm balls and wrist rings appeared in Drive and Joint
    /// space too, any time the feel toggle was Soft. That is three grabbable
    /// objects floating over a robot whose sticks are driving something else — the
    /// operator reads them as live controls, and in Drive they are simply in the
    /// way. iPhone parity (its `manipulate:` gate had the same defect).
    ///
    /// This also governs which renderer owns 3rd person, so the SceneKit grab path
    /// no longer takes over the stage in Drive+Soft.
    private var softPushing: Bool {
        pov == .third && feel == .compliance && teleopActive && space == .task
    }
    /// The SceneKit viewer is the visible 3rd-person renderer — SOFT ONLY now
    /// (Task·Wrist plans on the RealityKit stage). Suspended during a POV
    /// transition so the head-image dive always plays on the RealityKit stage
    /// it was designed for, then the swap resumes on the far side.
    private var sceneOwnsThird: Bool { !povAnimating && softPushing }
    /// Goal flow: tapping the floor PLACES the marker (aim it with the ring),
    /// the GO button dispatches it (Nav2 or the demo robot), and GO becomes
    /// STOP while the route is active. With no robot and no demo map loaded,
    /// the map is view-only.
    private enum GoalState { case none, placed, navigating }
    @State private var goalState: GoalState = .none
    @State private var pendingGoal: (x: Double, y: Double, yaw: Double)? = nil

    private var goalHandler: ((Double, Double, Double) -> Void)? {
        // Demo, or ANY robot endpoint: nav_goal rides the :9090 control socket,
        // so a down /map stream must not disable goal placement.
        guard store.demo || store.hasEndpoint || controller.connection == .connected
        else { return nil }
        return { x, y, yaw in
            // ONE staged intent at a time (2026-07-16): a nav goal and an arm plan
            // both put a green commit button in the same spot, and two of them side
            // by side is a coin flip over which one "GO" means. Placing a goal
            // discards a staged arm plan.
            if armPlan != nil { cancelPlanToken += 1 }
            pendingGoal = (x, y, yaw)
            withAnimation(Theme.Motion.snappy) { goalState = .placed }
        }
    }

    private func dispatchGoal() {
        guard let g = pendingGoal else { return }
        if store.demo { store.setDemoGoal(x: g.x, y: g.y, yaw: g.yaw) }
        else { controller.sendNavGoal(x: g.x, y: g.y, yaw: g.yaw) }
        withAnimation(Theme.Motion.snappy) { goalState = .navigating }
        Haptics.rigid()
    }

    private func stopNav() {
        if store.demo { store.cancelDemoGoal() }
        else if let o = store.lastOdom {
            // Pragmatic cancel on the wire: re-target the robot's own pose —
            // Nav2 replans to "here" and stops.
            controller.sendNavGoal(x: o.x, y: o.y, yaw: o.yaw)
        }
        pendingGoal = nil
        withAnimation(Theme.Motion.snappy) { goalState = .none }
        Haptics.rigid()
    }

    /// GO + ✕ (placed) / STOP (navigating) — floats just above the camera pills.
    ///
    /// A PLACED goal is a staged intent, exactly like a staged arm plan, so it gets
    /// the same two-button shape: commit or discard. Before this the only way out of
    /// a mis-tapped goal was to dispatch it and then stop the robot — the marker had
    /// no cancel at all.
    @ViewBuilder private var goalButton: some View {
        if goalState != .none {
            HStack(spacing: 8) {
                if goalState == .placed {
                    Button { clearGoal() } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 13, weight: .heavy)).foregroundStyle(.white)
                            .frame(width: 34, height: 34)
                            .background(Theme.Palette.danger.gradient, in: Circle())
                            .overlay { Circle().strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                            .contentShape(Circle())
                    }
                    .buttonStyle(PressableStyle())
                    .contentShape(.hoverEffect, Circle())
                    .hoverEffect()
                }
                Button {
                    if goalState == .placed { dispatchGoal() } else { stopNav() }
                } label: {
                    HStack(spacing: 7) {
                        Image(systemName: goalState == .placed ? "play.fill" : "stop.fill")
                            .font(.system(size: 14, weight: .bold))
                        Text(goalState == .placed ? "GO" : "STOP NAV")
                            .font(.system(size: 14, weight: .heavy)).tracking(0.5)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 22).frame(height: 34)
                    .background((goalState == .placed ? Theme.Palette.live : Theme.Palette.danger).gradient,
                                in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.3), lineWidth: 1) }
                    .shadow(color: (goalState == .placed ? Theme.Palette.live : Theme.Palette.danger).opacity(0.5),
                            radius: 10, y: 4)
                    .contentShape(Capsule(style: .continuous))
                }
                .buttonStyle(PressableStyle())
                .contentShape(.hoverEffect, Capsule(style: .continuous))
                .hoverEffect()
            }
            .transition(.scale.combined(with: .opacity))
        }
    }

    /// Discard a PLACED (not yet dispatched) goal: nothing was ever sent, so this
    /// only drops the marker and the pending pose.
    private func clearGoal() {
        Haptics.tap()
        pendingGoal = nil
        if store.demo { store.cancelDemoGoal() }   // clears the demo's own marker/route
        clearGoalToken += 1                        // drop the SceneKit marker
        navState.setGoal(nil)                      // and the 2D minimap's
        withAnimation(Theme.Motion.snappy) { goalState = .none }
    }

    /// SAFETY bands around the floating chrome — a tap that just misses a button
    /// must never plant a nav goal on the map behind it.
    private var goalExclusion: UIEdgeInsets {
        UIEdgeInsets(top: isLandscape ? 130 : 170,
                     left: isLandscape ? max(railLeadingInset, 90) : 12,
                     bottom: (isLandscape ? 100 : 170) + (teleopActive ? 140 : 0),
                     right: isLandscape ? 90 : 12)
    }

    /// Connected: ONLY the maps the robot actually advertises (`list_maps` reads
    /// the ROS workspace's saved SLAM maps) — an empty list stays EMPTY, never
    /// padded with fakes. Offline: the local dummy maps, labeled as such.
    private var connectedToRobot: Bool { store.hasEndpoint || controller.connection.isLive }
    private var savedRooms: [SLAMRoom] {
        if connectedToRobot {
            return controller.savedMaps.map { SLAMRoom.streamed(name: $0.name) }
        }
        return SLAMRoom.library
    }
    private var mapSectionTitle: String { connectedToRobot ? "SAVED MAPS · ROBOT" : "LOCAL DUMMY MAPS" }

    var body: some View {
        ZStack(alignment: .top) {
            // THE unified viewer. 3rd person = TRUE-3D RealityKit stage (real
            // depth in the window, v1 feel); 1st person = the SceneKit path
            // (2 s camera flight into the head + live FPV crossfade).
            // BOTH stay mounted permanently — a POV flip only crossfades
            // opacity between two already-rendering scenes. Recreating the
            // hidden one on each flip cost a black "loading" beat (SceneKit
            // scene build / RealityKit rig load) right after the flight.
            RobotSpace3DStage(controller: controller,
                                  navState: navState,
                                  camProjections: camProjections,
                                  camStreams: camStreamSet,
                                  camProjSize: camProjSize,
                                  cloudStride: cloudDensity.rawValue,
                                  floorOn: floorOn,
                                  meshOn: meshOn,
                                  cloudOn: cloudOn,
                                  gridOn: gridOn,
                                  laserOn: laserOn,
                                  tfOn: tfOn,
                                  recenterToken: recenterToken,
                                  fitToken: fitToken,
                                  follow: following,
                                  preset: stagePreset,
                                  presetToken: presetToken,
                                  presetInstant: stagePresetInstant,
                                  viewSize: viewSize,
                                  safeInsets: stageSafeInsets,
                                  onGoal: goalHandler,
                                  physicsCam: controller.physicsViewCam,
                                  physicsToken: physToken,
                                  // Task · Wrist · Stiff: the green-ball planner
                                  // ON the stage (StagePlanner) — Task keeps the
                                  // true-3D viewer like Drive/Joint.
                                  planMode: wristPlanning && !povAnimating,
                                  planRegion: region,
                                  executePlanToken: executePlanToken,
                                  cancelPlanToken: cancelPlanToken,
                                  debugStageToken: debugStageToken,
                                  debugStageSide: debugStageSide,
                                  onPlanChange: { p in
                                      // Mirror of goalHandler: ONE staged intent at
                                      // a time — staging an arm plan drops a PLACED
                                      // (never dispatched) nav goal.
                                      if p != nil, goalState == .placed { clearGoal() }
                                      withAnimation(.snappy) { armPlan = p }
                                  },
                                  onJointsChange: { controller.setJoints($0) },
                                  onBaseGoal: { x, y, yaw in
                                      controller.sendNavGoal(x: x, y: y, yaw: yaw)
                                  })
                .ignoresSafeArea()
                // The true-3D rig pops OUT of the window plane and would sit in
                // front of any pop-up (scene switch, calibration, confirm,
                // pickers) — hide it while a modal is up. It also yields to the
                // SceneKit viewer whenever that owns 3rd person (see above).
                .opacity(pov == .third && !sceneOwnsThird && !modalUp ? 1 : 0)
                .animation(.easeInOut(duration: 0.22), value: modalUp)
                .animation(.easeInOut(duration: 0.25), value: sceneOwnsThird)
                .allowsHitTesting(pov == .third && !sceneOwnsThird && !physicsFaded && !modalUp)
            // PHYSICS VIEW takeover: the MuJoCo viewer replaces the 3rd-person
            // visualizer. Mounted (connecting, invisible) the moment the toggle
            // engages; the crossfade fires once the stage's cinematic arc
            // completes AND server frames are flowing — landing on a
            // camera-matched view (shared PhysicsCam, both sides).
            if controller.physicsViewOn, pov == .third,
               let ws = controller.simStudio.viewerWS {
                PhysicsViewerOverlay(url: ws, cam: controller.physicsViewCam, faded: $physicsFaded)
                    .opacity(physicsFaded ? 1 : 0)
                    .allowsHitTesting(physicsFaded)
                    .transition(.opacity)
            }
            // Contrast scrim for the SceneKit viewer. That scene is tuned for
            // iOS's DARK canvas (ambient 650 + key 750 + HDR bloom on a white
            // PBR robot); over visionOS glass a bright room bleeds straight
            // through and the robot blows out to an unreadable white blob —
            // which is fatal for a mode whose whole job is placing a ball at
            // the wrist. Only while the SceneKit path owns 3rd person: the POV
            // dive REQUIRES a clear background to crossfade the two renderers,
            // and sceneOwnsThird is already false throughout that transition.
            if sceneOwnsThird {
                Color.black.opacity(0.62)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
            LiveMapView(store: store,
                        pose: controller.livePose,
                        navState: minimapOn ? navState : nil,
                        recenterToken: recenterToken,
                        fitToken: fitToken,
                        follow: following,
                        pov: scenePov,
                        // Park the camera with no flight when the POV dive is
                        // staging this view, or when the wrist planner brings
                        // it up in place. (This was declared but never passed —
                        // the dive's "park the hidden view NOW" silently ran
                        // the full 2 s flight instead.)
                        instantPov: sceneInstantPov,
                        camProjections: camProjections,
                        camVideoURL: { controller.videoThumbURL($0) },
                        camRouteToken: controller.videoThumbURL("main")?.absoluteString ?? "offline",
                        meshHidden: !meshOn,
                        cloudHidden: !cloudOn,
                        gridHidden: !gridOn,
                        robotHidden: !robotOn,
                        floorHidden: !floorOn,
                        floorGridHidden: !floorGridOn,
                        tfHidden: !tfOn,
                        lidarScan: laserOn ? controller.lidarScan : nil,
                        cloudStride: cloudDensity.rawValue,
                        camProjSize: camProjSize,
                        camFlat: camFlat,
                        // Handles (green palm balls + wrist rings) appear exactly
                        // where they can be grabbed — the SceneKit path in 3rd
                        // person. The grab recognizer already refuses unless
                        // currentPov == .third, so arming them anywhere else
                        // only ever showed balls that could not be dragged.
                        manipulate: sceneOwnsThird,
                        forceMode: feel == .compliance,
                        // The SceneKit path serves SOFT pushes only now — the
                        // Task·Wrist plan/execute flow moved to the stage's
                        // StagePlanner (Task stays true-3D). planMode off keeps
                        // its Coordinator from double-staging plans.
                        planMode: false,
                        planRegion: region,
                        onPlanChange: { _ in },
                        onBaseGoal: { x, y, yaw in controller.sendNavGoal(x: x, y: y, yaw: yaw) },
                        executePlanToken: 0,
                        cancelPlanToken: 0,
                        clearGoalToken: clearGoalToken,
                        debugStageToken: 0,
                        debugStageSide: debugStageSide,
                        onJointsChange: { controller.setJoints($0) },
                        onJointChange: { controller.setJoint($0, $1) },
                        onPartForce: { joint, f in
                            if f == [0, 0, 0] { controller.clearExternalForce() }
                            else { controller.applyExternalForce([joint: f]) }
                        },
                        onGoal: goalHandler,
                        goalExclusion: goalExclusion,
                        focusModule: module,
                        focusToken: focusToken,
                        followModule: space == .joint ? module : nil)
                .rebuildsOnRobotRigChange()   // new robot / swapped hands → rebuild the scene
                .ignoresSafeArea()
                .padding(scenePov == .first
                         ? EdgeInsets(top: Self.povInsetTop, leading: Self.povInsetSide,
                                      bottom: Self.povInsetBottom, trailing: Self.povInsetSide)
                         : EdgeInsets())
                .clipShape(RoundedRectangle(cornerRadius: scenePov == .first ? 26 : 0,
                                            style: .continuous))
                // Visible in 1st person (the FPV dive) AND whenever it owns
                // 3rd person (the wrist planner / Soft pushes). Modals hide it
                // for the same reason they hide the stage.
                .opacity((pov == .first || sceneOwnsThird) && !modalUp ? 1 : 0)
                .animation(.easeInOut(duration: 0.25), value: sceneOwnsThird)
                .allowsHitTesting((pov == .first || sceneOwnsThird) && !modalUp)

            // 1st-person LIVE video — the robot's real head camera, faded in over
            // the matching 3D view once the camera flight lands. In demo (or with
            // the stream dead) the pure 3D first-person render stays visible.
            if pov == .first, controller.connection.isLive,
               let fpvURL = controller.videoMainURL("main") {
                MJPEGView(url: fpvURL, onLive: { live in fpvLive = live })
                    .id(fpvURL)     // robot switch (sim ↔ real) restarts the stream
                    .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
                    .padding(EdgeInsets(top: Self.povInsetTop, leading: Self.povInsetSide,
                                        bottom: Self.povInsetBottom, trailing: Self.povInsetSide))
                    .opacity(fpvVisible && fpvLive ? 1 : 0)
                    .allowsHitTesting(fpvVisible && fpvLive)
                    .animation(.easeInOut(duration: 0.45), value: fpvVisible && fpvLive)
                // Wrist cams pinned to the left/right edges while the live
                // head video fills the screen.
                if fpvVisible && fpvLive { fpvChrome }
            }
            // The small live 3D robot rides the BOTTOM-RIGHT corner whenever
            // 1st person is active (v1 Vision Pro pilot layout) — live or demo.
            // The live 3D model, bottom-right. It STAYS: in the pilot view the
            // camera shows what the robot sees but not what the robot IS, and
            // this is the only place the operator can watch their own body being
            // mirrored while flying the head cam (operator, 2026-07-28 —
            // restored after I removed it reading an earlier note as being about
            // this window rather than the buttons above it).
            if pov == .first { miniRigCorner }

            // Centred standby card — only while the scene is truly EMPTY and
            // we're overlooking the map in 3rd person. TRANSIENT: it fades out
            // after a few seconds so it never squats over the robot.
            if pov == .third, !store.demo, !store.hasEndpoint || !store.connected,
               store.vertexCount == 0, !store.hasPose, !standbyDismissed {
                standby
                    .task {
                        try? await Task.sleep(nanoseconds: 6_000_000_000)
                        withAnimation(.easeOut(duration: 0.5)) { standbyDismissed = true }
                    }
                    .transition(.opacity)
            }

            VStack(spacing: 8) {
                topControls
                if !isLandscape && (!teleopActive || space == .drive) && goalState == .none { rotateHint }
                // TRANSIENT (2026-07-28): coaching copy earns two seconds, not a
                // permanent slot over the viewer. `hintToken` re-arms it on each
                // fresh entry into the space so it still greets a new session.
                if space == .task && (wristPlanning || teleopActive), hintVisible {
                    taskHint
                        .transition(.opacity)
                        .task(id: hintToken) {
                            try? await Task.sleep(nanoseconds: 2_000_000_000)
                            withAnimation(.easeOut(duration: 0.35)) { hintVisible = false }
                        }
                }
                if store.demo { demoBanner }
                // Floating 2D map — top-right, below the controls.
                if minimapOn {
                    HStack(spacing: 0) {
                        Spacer(minLength: 0)
                        MiniMap2D(state: navState)
                            // Landscape: smaller so the right-edge control panel
                            // never has to share space with it.
                            .frame(width: isLandscape ? 112 : nil,
                                   height: isLandscape ? 112 : nil)
                            .transition(.opacity)
                    }
                    .padding(.trailing, 6)
                }
                Spacer(minLength: 0)
            }
            // Immersive landscape reserves the status bar via a safe-area inset
            // (ContentView), so no extra clearance is needed in either orientation.
            .padding(.leading, isLandscape ? 4 : 0)
        }
        // Record / Replay cards — small centred windows over everything (the
        // STOP button itself lives in the top bar, where Record was).
        .overlay { recordCardOverlay }
        .animation(Theme.Motion.snappy, value: recordCard)
        .overlay {
            if showWholeBodyConfirm {
                ZStack {
                    Color.black.opacity(0.25).ignoresSafeArea()
                        .onTapGesture { showWholeBodyConfirm = false }
                    wholeBodyConfirmCard
                }
                .transition(.opacity.combined(with: .scale(scale: 0.96)))
            }
        }
        .animation(Theme.Motion.snappy, value: showWholeBodyConfirm)
        // Teleop clusters flank the centred pills. ALWAYS live for the selected
        // space (operator feedback: Drive/Joint controls appear the moment the
        // space is picked — no Start gate; Task shows its whole-body card, and
        // only IT needs the explicit Start Teleop).
        .overlay(alignment: .bottom) {
            // Joysticks only in Stiff. Soft is PASSIVE (arms float on gravity comp
            // and the controller drops every command), so the whole stick cluster
            // disappears — the operator moves the robot by hand. Gated on `feel`
            // ALONE: keyed on `teleopActive` too, Soft-without-Start still showed
            // live-looking sticks that the controller silently ignored.
            // …and never with the motors dead: a stick you can push that moves
            // nothing is the worst control in the app. (See `torqueOn`.)
            if torqueOn, feel == .impedance {
                GeometryReader { geo in
                    teleopControls(height: geo.size.height)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        // (CONTROL REGION moved to the TRAILING ornament — see `regionPanel`. As a
        // floating top-left rail it collided with the Record button and used a
        // style nothing else in the app shared.)
        // Viewer controls live on a LEADING scene ornament (see leftRail) —
        // the 3D stage pops out of the window plane, so in-window chrome at the
        // bottom/edges gets physically occluded. Ornaments never do.
        .ornament(attachmentAnchor: .scene(.leading), contentAlignment: .trailing) {
            // TWO vertical columns on the leading edge: the app nav LEFT-MOST,
            // the viewer controls beside it as INDIVIDUAL glass buttons.
            if isActiveTab {
                HStack(alignment: .center, spacing: 10) {
                    if let nav = navSelection { LeftNavRail(selection: nav) }
                    // ALWAYS mounted — `leftRail` decides what it contains. This
                    // used to be gated on 3rd person, which took Talk and Listen
                    // down with the stage controls and left first person with no
                    // way to speak at all (2026-07-29).
                    leftRail
                }
                .padding(.trailing, 12)
            }
        }
        // The DRIVE PAD hangs under the window (see `drivePadOrnament`).
        .ornament(attachmentAnchor: .scene(.bottom), contentAlignment: .top) {
            drivePadOrnament
        }
        .ornament(attachmentAnchor: .scene(.trailing), contentAlignment: .leading) {
            // ONE right-edge extension, read left→right: the PANEL TOGGLES rail
            // (vertically centred, mirroring the leading viewer rail), then the
            // panels they open. Joint space's MODULE + slider panels appear
            // automatically; CONTROL REGION rides Task/Soft.
            // Joint panels are hidden in SOFT: the robot is passive there and the
            // controller drops every joint command, so the sliders would move and
            // snap back — a dead control is worse than no control.
            if isActiveTab {
                HStack(alignment: .center, spacing: 10) {
                    // Toggles first (nearest the window), vertically centred like
                    // the left rail — they are view controls, not robot commands.
                    if pov == .third { quickPanelToggles }
                    panelStack
                }
                .padding(.leading, 12)
            }
        }
        .onChange(of: store.navRouteActive) { _, active in
            // Route emptied server-side (goal reached / cancelled) → reset.
            if !active && goalState == .navigating {
                withAnimation(Theme.Motion.snappy) { goalState = .none }
            }
        }
        // Torque dropped → teleop is OVER, not paused (iOS parity). Without this,
        // teleopActive survives the e-stop with stale state behind the hidden
        // controls, and TORQUE ON would restore an abandoned session — mid-drag,
        // in whatever mode the operator panicked out of.
        .onChange(of: controller.estopEngaged) { _, engaged in
            guard engaged else { return }
            if teleopActive { stopTeleop() }
            if goalState == .placed { withAnimation(Theme.Motion.snappy) { goalState = .none } }
        }
        // Scene-switch loading pop-up (SCC milestone card: physics → telemetry
        // → cameras) — same style as the iOS app's overlay.
        .overlay {
            SceneLoadingOverlay(studio: controller.simStudio)
                .animation(Theme.Motion.smooth, value: controller.simStudio.switching != nil)
        }
        .onChange(of: controller.refitToken) { _, _ in
            // Sim reset / random respawn → re-frame the viewers on the robot. The
            // respawn lands a moment LATER (async reset, then the new odom pose), so
            // re-frame NOW and again after it settles — else the robot leaves view.
            recenterToken += 1
            for delay in [0.5, 1.1] {
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) { recenterToken += 1 }
            }
        }
        .background {
            GeometryReader { geo in
                Color.clear
                    .onAppear { viewSize = geo.size }
                    .onChange(of: geo.size) { _, sz in viewSize = sz }
            }
        }
        .onReceive(tick) { _ in onTick() }
        .onChange(of: controller.connection) { _, c in
            // The on-appear request fires before the socket is up and is lost —
            // re-ask for the saved-map catalogue the moment the link goes live.
            if c == .connected { controller.requestMapList() }
        }
        // Guided arm-extension calibration — a floating glass CARD in the same
        // visual language as the whole-body confirm that immediately precedes it
        // (scrim + rounded glass; never a full sheet), and the 3D stage hides
        // behind it (modalUp). Real visionOS glass via `.glassBackgroundEffect`,
        // not a clipped `.ultraThinMaterial` rectangle — that read as a flat iOS
        // sheet pasted into the room, and stacked a second material on top of the
        // one the sheet presentations already provide.
        .overlay {
            if showCalibrationFlow {
                ZStack {
                    Color.black.opacity(0.55).ignoresSafeArea()
                    // SAME shell as wholeBodyConfirmCard (operator 2026-07-18:
                    // "make it smaller + consistent with the clear-area card"):
                    // 380 pt glass card, radius 24, content-hugging height.
                    CalibrationFlowView(presetName: controller.activeProfileName ?? "",
                                        onClose: { showCalibrationFlow = false },
                                        // Whole-body flow: capture → 3·2·1 →
                                        // engage. (Settings' sheet never engages.)
                                        engageOnFinish: true)
                        .environmentObject(controller)
                        .frame(width: 380)
                        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                        .overlay { RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .strokeBorder(.white.opacity(0.16), lineWidth: 1) }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.97)))
            }
        }
        .animation(Theme.Motion.smooth, value: showCalibrationFlow)
        .background(Theme.Palette.canvas)
        .onAppear {
            controller.requestMapList()
            controller.liveMap.active = true
            // Offline / on-device demo: auto-load the LOCAL demo map so the Mesh
            // and 2D Grid layers have data the moment the viewer opens (it stops
            // by itself once a real robot endpoint comes up; the map picker can
            // still switch rooms or go live).
            if !store.hasEndpoint, !store.demoRequested { store.startLocalDemo() }
            // RE-ADOPT a live whole-body session. Full immersion unmounts this
            // view, so its @State is FRESH on the way back — but the session
            // (`bodyTracking`) lives on the controller and must keep driving no
            // matter which mode the window last showed. Without this the window
            // came back in Drive with the robot still mirroring and no Stop
            // verb anywhere in sight.
            if controller.bodyTracking {
                space = .task
                taskMode = .wholeBody
                teleopActive = true
            }
            #if DEBUG
            // The app BOOTS E-STOPPED (2026-07-18) and every driving control is
            // gated on TORQUE ON — a harness that drives must "press" it first,
            // exactly like the operator would (sends estop{active:false}).
            if UserDefaults.standard.bool(forKey: "debugTeleop")
                || UserDefaults.standard.string(forKey: "debugSpace") != nil
                || UserDefaults.standard.string(forKey: "debugPlanExecute") != nil
                || UserDefaults.standard.bool(forKey: "debugFullImmersive") {
                controller.releaseEStop()
            }
            if UserDefaults.standard.bool(forKey: "debugTeleop"), !teleopActive {
                startTeleop()
            }
            if let s = UserDefaults.standard.string(forKey: "debugSpace"),
               let sp = TeleopSpace(rawValue: s) {
                space = sp
                if teleopActive { applySpaceSpec() }
                applyStagePreset(for: sp)   // faithful boot: run the space's camera flight too
            }
            // debugCalibration=YES — open the guided calibration card straight away.
            // The card is otherwise 3 gated taps deep (Start Teleop → safety confirm
            // → tracking), and the SIMULATOR HAS NO HAND OR HEAD TRACKING, so this is
            // the only way to review its layout without a headset. It will sit on
            // "bring your hands into view" there — that is the honest empty state.
            if UserDefaults.standard.bool(forKey: "debugCalibration") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { showCalibrationFlow = true }
            }
            if UserDefaults.standard.bool(forKey: "debugTF") { tfOn = true }
            // debugFreeze=neck,torso,upper — set the body switches without taps
            // (the Simulator cannot synthesise them) for screenshots/checks.
            if let fz = UserDefaults.standard.string(forKey: "debugFreeze") {
                controller.trackNeck = !fz.contains("neck")
                controller.trackTorso = !fz.contains("torso")
                controller.retargetScope = fz.contains("upper") ? .local : .global
            }
            // debugRecordCard=record|replay|recording — open a recording card (or
            // start a real take into `ui_check`) for screenshots: the Simulator
            // cannot synthesise the taps. Delete the key after the run.
            if let rc = UserDefaults.standard.string(forKey: "debugRecordCard") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                    switch rc {
                    case "replay":    openReplayCard()
                    case "recording": startRecording(into: "ui_check",
                                                      episode: controller.taskCounts["ui_check"] ?? 0)
                    default:          recordFolder = "ui_check"; openRecordCard()
                    }
                }
            }
            if UserDefaults.standard.bool(forKey: "debugFloor") { floorOn = true }
            // debugLayers=<csv of mesh,cloud,grid,laser,minimap,cam> — ONLY
            // these stay on. A headless check of the robot itself (green balls,
            // plan ghosting) needs the map layers off: the demo room's cloud is
            // a dense shell AROUND the robot, so the framed camera sits inside
            // it and the robot is buried — and the head-cam projection plane is
            // a blank WHITE slab across the torso when no video is streaming.
            // `debugLayers=` (empty) = bare robot.
            if let l = UserDefaults.standard.string(forKey: "debugLayers") {
                let want = Set(l.split(separator: ",").map(String.init))
                meshOn = want.contains("mesh"); cloudOn = want.contains("cloud")
                gridOn = want.contains("grid"); laserOn = want.contains("laser")
                minimapOn = want.contains("minimap")
                // Stage layers (iPhone-harness parity): `grid` is the SLAM
                // occupancy floorplan, `floorgrid` the reference lines on the
                // floor. `ground` is accepted as the retired spelling.
                robotOn = want.contains("robot"); floorOn = want.contains("floor")
                floorGridOn = want.contains("floorgrid") || want.contains("ground")
                camHeadOn = want.contains("cam")
                camWristLOn = false; camWristROn = false; camBaseOn = false
            }
            if let pnl = UserDefaults.standard.string(forKey: "debugPanels") {
                showCamsPanel = pnl.contains("cams")
                showBarsPanel = pnl.contains("bars")
                showSeriesPanel = pnl.contains("series")
            }
            if UserDefaults.standard.bool(forKey: "debugFullImmersive") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    if !teleopActive { startTeleop() }
                    Task {
                        controller.fullImmersion = true
                        if controller.needsImmersiveOpen() {
                            _ = await openImmersive(id: WindowID.immersive.rawValue)
                        }
                    }
                }
            }
            if let g = UserDefaults.standard.string(forKey: "debugGoal") {
                let c = g.split(separator: ",").compactMap { Double($0) }
                if c.count >= 2 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 8.0) {
                        goalHandler?(c[0], c[1], 0)
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { dispatchGoal() }
                    }
                }
            }
            if UserDefaults.standard.bool(forKey: "debugPovFlip") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 6.0) { setPov(.first) }
                DispatchQueue.main.asyncAfter(deadline: .now() + 13.0) { setPov(.third) }
            }
            // Plan/execute harness (server/tools/exp_arm_plan.py parity): Task+Stiff,
            // STAGE a plan (no send), hold, then EXECUTE. debugPlanExecute=<left|right>,
            // debugRegion=<arm|upper_body|whole_body>.
            if let s = UserDefaults.standard.string(forKey: "debugPlanExecute") {
                debugStageSide = s == "right" ? "right" : "left"
                if let rg = UserDefaults.standard.string(forKey: "debugRegion"),
                   let r = ControlRegion(rawValue: rg) { region = r }
                DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                    scenePov = .third; pov = .third      // the green ball is a 3rd-person control
                    space = .task; feel = .impedance
                    if !teleopActive { startTeleop() } else { applySpaceSpec() }
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 6)  { debugStageToken += 1 }
                DispatchQueue.main.asyncAfter(deadline: .now() + 12) { executePlanToken += 1 }
            }
            if UserDefaults.standard.bool(forKey: "debugLocalDemo") { store.startLocalDemo() }
            // Physics-view harness (iOS parity): engage the MuJoCo takeover
            // after N seconds — lets a headless run capture the arc + crossfade.
            let pdelay = UserDefaults.standard.double(forKey: "debugPhysics")
            if pdelay > 0 {
                DispatchQueue.main.asyncAfter(deadline: .now() + pdelay) { togglePhysics() }
            }
            // Scene-switch harness: kick a real SCC scene switch after 8 s so a
            // headless run exercises the loading overlay + milestone polling.
            if let sw = UserDefaults.standard.string(forKey: "debugSceneSwitch"), !sw.isEmpty {
                DispatchQueue.main.asyncAfter(deadline: .now() + 8.0) {
                    if let sc = controller.simStudio.scenes.first(where: { $0.name == sw }) {
                        controller.simStudio.switchScene(sc)
                    }
                }
            }
            // Auto-open an aux window for screenshot-driven testing:
            // bars | series | map2d | camera | volume
            if let w = UserDefaults.standard.string(forKey: "debugOpenWindow") {
                switch w {
                case "bars":   openWindow(id: WindowID.jointBars.rawValue)
                case "series": openWindow(id: WindowID.timeSeries.rawValue)
                case "map2d":  openWindow(id: WindowID.map2D.rawValue)
                case "sensors": openWindow(id: WindowID.auxSensors.rawValue)
                case "volume": openWindow(id: WindowID.robotVolume.rawValue)
                default: break
                }
            }
            #endif
            if pov == .first {
                following = false
                DispatchQueue.main.asyncAfter(deadline: .now() + LiveMapView.povFlightDuration) {
                    if pov == .first { fpvVisible = true }
                }
            }
        }
        .onChange(of: selected) { _, _ in controller.liveMap.active = true }
        // Whole-body teleop can engage from INSIDE the space (the dome's Start
        // Teleop / console) while this window sits in any mode — the body must
        // drive regardless, so the window adopts the session rather than the
        // session asking the window's permission. Idempotent for the normal
        // Task-flow engage (those states are already set).
        .onChange(of: controller.bodyTracking) { _, on in
            // BOTH directions. This used to return early when tracking went OFF,
            // so a session ended anywhere but this view's own Stop button —
            // Home, an e-stop, the immersive space closing — left `teleopActive`
            // true and the console still showing a Stop verb for a session that
            // was already over.
            guard on else {
                if teleopActive { withAnimation(Theme.Motion.snappy) { teleopActive = false } }
                return
            }
            space = .task
            taskMode = .wholeBody
            teleopActive = true
        }
        // The SceneKit viewer is becoming the visible 3rd-person renderer
        // (wrist planner / Soft). It boots parked at the head (scenePov ==
        // .first) so the POV dive can crossfade into it, so park it WIDE with
        // no flight — otherwise it fades in mid-glide out of the robot's skull.
        .onChange(of: sceneOwnsThird) { _, owns in
            guard owns, scenePov != .third else { return }
            sceneInstantPov = true
            scenePov = .third
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                sceneInstantPov = false
                // Re-frame only AFTER the park has landed. LiveMapView reads
                // recenterToken BEFORE it applies pov in the same update pass,
                // so a same-pass bump is dropped while the coordinator still
                // reads 1st person — leaving the camera parked in the robot's
                // skull, filling the view with map mesh at point-blank range.
                if sceneOwnsThird { recenterToken += 1 }
            }
        }
        .onDisappear {
            controller.liveMap.active = false
            // Full immersion UNMOUNTS this view (ContentView swaps the window
            // content for the placeholder) — the session is moving INTO the
            // dome, not ending. Stopping teleop here sent `navigation` and
            // dropped `bodyTracking` in a race against the dome's own engage:
            // "enter full immersive and the arms + hands are dead". Whole-body
            // teleop must survive the hop regardless of the window's mode.
            if teleopActive, !controller.fullImmersion { stopTeleop() }
        }
    }

    /// Top-down 2D map mirror — plain @State (NOT @StateObject) so its 20 Hz
    /// pose updates invalidate only the minimap canvas.
    @State private var navState = NavMapState()

    // MARK: - Top controls

    /// Top controls — THREE FUNCTIONAL COLUMNS (2026-07-16 operator layout):
    ///   LEFT   = WORLD SOURCE: [3D|Physics / Full Immersive · map chip] over [Record].
    ///   CENTRE = CONTROL + VIEW: [Drive·Joint·Task · 3rd|1st] over the verbs
    ///            [Start Whole Body Tracking · GO · Execute]. The POV toggle sits
    ///            beside the space it applies to; the verbs sit under the mode.
    ///   RIGHT  = [Stiff|Soft] — the robot's FEEL, alone in the corner.
    /// The Cams/Joints/Trends panel toggles are no longer here at all: they are a
    /// vertical rail on the TRAILING ornament, beside the panels they open (see
    /// `quickPanelToggles`).
    /// Does the top bar's LEFT column carry a second row (Back / Record)?
    private var topSecondRowVisible: Bool { true }   // Record always occupies it
    /// Height of the whole top-controls stack, from the SAME constants it is built
    /// from: 6 pt top pad + N rows of the 36 pt control standard + 8 pt gutters.
    /// Anything overlaid below the bar (the region rail) must clear this — hard-
    /// coding a number is how the rail ended up sitting on the Record button.
    private var topControlsHeight: CGFloat {
        // Row 1 space/POV · row 2 verbs+Record · row 3 the retarget controls.
        var rows: CGFloat = topSecondRowVisible ? 2 : 1
        if torqueOn, wholeBodyEngaged { rows += 1 }
        return 6 + rows * 36 + (rows - 1) * 8
    }

    /// Points of the window the FLOATING CHROME covers, derived from the same
    /// constants the chrome is built from. The 3D stage frames its module
    /// flights inside what is left — a framing computed against the raw window
    /// puts the subject under the top bar or behind the pill rows.
    /// Left/right are 0: the rail and the panels are ORNAMENTS, outside the
    /// window, so they never cover the viewer.
    private var stageSafeInsets: EdgeInsets {
        // Bottom: whatever `teleopControls` puts IN the window for this space —
        // joysticks in Drive, grip triggers in Task·Wrist, the whole-body card
        // in Task·WholeBody. Joint space puts its controls on the trailing
        // ORNAMENT, so its window floor is clear. Heights follow the same
        // `jSize`/`gripH` expressions the controls are laid out from.
        let bar = (isLandscape ? 126 : bottomBarInset + 52) as CGFloat
        let cluster: CGFloat = {
            switch space {
            // Drive puts NOTHING in the window — its pad is on the BOTTOM
            // ORNAMENT (see `drivePadOrnament`), outside the window rect — so
            // there is no cluster to clear and the stage gets the whole floor.
            case .drive: return 0
            case .task:  return taskMode == .wrist
                ? min(122, max(90, viewSize.height * 0.34)) : 120
            case .joint: return 0
            }
        }()
        // The DRIVE PAD is on the bottom ornament in every space (2026-10-27). It is
        // outside the window, but the rig stands IN FRONT of the window plane, and a
        // module flight zooms it further out — so in perspective the lower body
        // reaches past the window's bottom edge, over the pad's header. Reserving a
        // strip keeps every framing above it (measured on the Simulator: R Arm's
        // base landed ~30 pt over the header without it).
        let pad: CGFloat = (drivePadWanted && drivePadShown) ? 90 : 0
        return EdgeInsets(top: topControlsHeight + 10,
                          leading: 0,
                          bottom: (cluster > 0 ? bar + cluster : 16) + 10 + pad,
                          trailing: 0)
    }

    private var topControls: some View {
        HStack(alignment: .top, spacing: 12) {
            // ── LEFT · world source ───────────────────────────────────────
            VStack(alignment: .leading, spacing: 8) {
                // The map is what the view is OF, so it rides beside the
                // renderer switch. (The 3rd|1st toggle moved to the CENTRE
                // column, beside the control space — 2026-07-16.)
                HStack(spacing: 8) {
                    if pov == .third {
                        stageModeToggle          // 3D | Physics (sim + SCC)
                        mapSourceMenu            // Live SLAM · saved maps
                    }
                }
                // Session/data actions sit underneath, well clear of Start/Stop.
                let showBack = isLandscape && onBack != nil && pov == .third
                // ALWAYS — Record is the left column's second row now that the
                // status bar no longer carries one (operator, 2026-07-28).
                let showRec = true
                // Same predicate as `topSecondRowVisible`, which the region rail
                // clears — keep them in step.
                if topSecondRowVisible {
                    HStack(spacing: 8) {
                        if showBack, let onBack {
                            Button {
                                Haptics.select(); onBack()
                            } label: {
                                Image(systemName: "chevron.left").font(.system(size: 16, weight: .bold))
                                    .foregroundStyle(.white).frame(width: 40, height: 40)
                                    .background(.black.opacity(0.4), in: Circle())
                                    .contentShape(Circle())
                            }
                            .buttonStyle(PressableStyle())
                        }
                        if showRec { recButton }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // ── CENTRE · control + view ───────────────────────────────────
            VStack(spacing: 8) {
                HStack(spacing: 8) {
                    // The space picker is a CONTROL — hidden with the motors
                    // dead, and hidden once WHOLE-BODY TELEOP IS RUNNING: your
                    // body is the controller then, and switching to Drive or
                    // Joint mid-session is not something you do, it is
                    // something you'd do by accident. What is left is the two
                    // things you DO want mid-session — which view you are in,
                    // and how to stop (operator, 2026-07-28).
                    if torqueOn, pov == .third, feel == .impedance,
                       !leanTeleopBar { spaceToggle }
                    topPovToggle          // the view of the space beside it
                }
                if torqueOn {
                    HStack(spacing: 8) {
                        // ONE Start Teleop button everywhere (2026-07-16): the SAME
                        // orange `StartTeleopButton` as 1st person. In Task it also
                        // selects the whole-body METHOD (Wrist is the resting state,
                        // so starting teleop IS choosing whole-body).
                        if space == .task || pov == .first || controller.fullImmersion
                            || feel == .compliance {
                            StartTeleopButton(active: teleopActive, compact: !isLandscape && teleopActive) {
                                if teleopActive {
                                    stopTeleop()
                                    if space == .task { setTaskMode(.wrist) }   // back to the planner
                                } else {
                                    if space == .task { setTaskMode(.wholeBody) }
                                    showWholeBodyConfirm = true
                                }
                            }
                        }
                        if leanTeleopBar {
                            // WHOLE-BODY, RUNNING: two verbs, full width — Stop
                            // and go inside (operator, 2026-07-28). Everything
                            // else that used to sit here is either impossible
                            // now (a green-ball plan cannot be staged while the
                            // body is driving the arm) or belongs to a different
                            // job (a nav goal). Recalibrate and Set Forward do
                            // survive, as ICONS behind a divider: they are the
                            // two things that go wrong mid-session — the reach
                            // scale drifting and the forward datum sliding — and
                            // dropping them outright would leave a 3rd-person
                            // operator with no way to fix either without first
                            // entering full immersion.
                            if !controller.fullImmersion { fullImmersiveCTA }
                        } else {
                            // The other thing you reach for mid-session: go inside
                            // it — offered in 1st person, where it is the pilot's
                            // one move.
                            if !controller.fullImmersion, pov == .first { fullImmersiveCTA }
                            if pov == .third { goalButton }
                            // EXECUTE / CANCEL for a staged green-ball arm plan.
                            planButtons
                        }
                    }
                }
                // ── ROW 3 · BODY (2026-10-16): which parts the retargeter drives,
                // plus its two verbs — every control in words, one place. A REAL
                // row of the bar (counted by `topControlsHeight`), so whatever
                // sits below the bar moves down for it — as an offset overlay
                // the demo/hint banners drew on top of it.
                if torqueOn, wholeBodyEngaged {
                    BodyControls(onRecalibrate: canRecalibrate ? { showCalibrationFlow = true } : nil)
                        .transition(.opacity)
                }
            }
            .fixedSize()

            // ── RIGHT · the robot's FEEL (a control law — dead motors, no toggle) ─
            VStack(alignment: .trailing, spacing: 8) {
                if torqueOn { tuningButton }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.horizontal, 14)
        .padding(.top, 6)
    }

    /// A pop-up is on screen: the protruding 3D stage must yield (it renders in
    /// FRONT of window-space modals otherwise — "the 3D model blocks the pop-up").
    private var modalUp: Bool {
        controller.simStudio.switching != nil || showCalibrationFlow
            || showWholeBodyConfirm || showSavedMaps
            // The recording cards belong here for the same reason every other
            // entry does: the true-3D rig pops OUT of the window plane, so it
            // renders IN FRONT of in-window chrome and swallows the taps meant
            // for it (the 2026-07-29 "Record and Replay look dead" report).
            || recordCard != nil
            || controller.modalPopupVisible          // e.g. the Worlds sheet
            || controller.connectCardVisible         // the proximity Connect card
    }

    // MARK: - Physics view (MuJoCo takeover)

    @State private var physicsFaded = false
    @State private var physToken = 0

    /// The stage's render source: the app's 3D visualizer or the simulator's
    /// MuJoCo pixels.
    enum StageMode: String, CaseIterable, Identifiable, Hashable {
        case visualizer, physics
        var id: String { rawValue }
        var title: String { self == .visualizer ? "3D" : "Physics" }
        var icon: String { self == .visualizer ? "cube.transparent" : "atom" }
    }

    /// PROPER TOGGLE (same GlassSegmented language as 3rd|1st), riding right
    /// next to the POV segment — sim + SCC studio, 3rd person only.
    @ViewBuilder private var stageModeToggle: some View {
        if controller.simStudio.available {
            GlassSegmented(options: StageMode.allCases,
                           selection: Binding(
                               get: { controller.physicsViewOn ? StageMode.physics : .visualizer },
                               set: { mode in
                                   let wantPhysics = (mode == .physics)
                                   if wantPhysics != controller.physicsViewOn { togglePhysics() }
                               }),
                           title: { isLandscape ? $0.title : "" }, icon: { $0.icon },
                           prominent: false)
                .fixedSize()
        }
    }

    /// Engage: the stage flies its cinematic arc to the shared spherical pose
    /// while the MuJoCo stream connects (invisible); the overlay crossfades in
    /// once the arc lands and frames flow. Disengage: fade back to the (still
    /// tracking, camera-matched) stage underneath.
    private func togglePhysics() {
        Haptics.rigid()
        if controller.physicsViewOn {
            withAnimation(.easeInOut(duration: 0.5)) { physicsFaded = false }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.55) {
                controller.physicsViewOn = false
                controller.physicsViewCam.active = false
            }
        } else {
            physicsFaded = false
            controller.physicsViewCam.active = true
            controller.physicsViewOn = true
            physToken += 1                    // stage captures + starts the arc
        }
    }

    // MARK: - Recording (one take = a training episode AND a replayable clip)

    /// Which recording card is up — a small centred glass card over a scrim
    /// (RecordingControls.swift, BYTE-IDENTICAL with the iPhone app).
    enum RecordCard: Equatable { case record, replay }
    @State private var recordCard: RecordCard?
    @State private var recordFolder = ""
    @State private var replayRepeats = 1
    /// The take in progress, as the operator started it (folder + 0-based
    /// episode index + start time). Drives the big STOP RECORDING button.
    @State private var take: (folder: String, episode: Int, since: Date)?

    private var recordMode: RecordTopControl.Mode {
        let d = controller.demo
        if controller.recording {
            // `take` is nil only if this view was rebuilt mid-take (full
            // immersion unmounts it); fall back to what the server reports.
            let t = take ?? (controller.recordTask, max(0, controller.episodeCount),
                             Date().addingTimeInterval(-d.elapsed))
            return .recording(folder: t.folder, episode: t.episode, since: t.since)
        }
        if d.playing { return .replaying(name: d.name ?? "", rep: d.rep, repeats: d.repeats) }
        return .idle
    }

    private var recButton: some View {
        RecordTopControl(mode: recordMode,
                         replayEnabled: !controller.recording,
                         onRecord: openRecordCard,
                         onReplay: openReplayCard,
                         onStop: stopWhateverIsRunning)
    }

    private func openRecordCard() {
        Haptics.select()
        controller.refreshCollectionStatus()
        // Keep going in the folder you were in: the next take is Episode N+1.
        if recordFolder.isEmpty, controller.recordTask != "default" {
            recordFolder = controller.recordTask
        }
        withAnimation(Theme.Motion.snappy) { recordCard = .record }
    }

    private func openReplayCard() {
        Haptics.select()
        controller.refreshDemoClips()
        withAnimation(Theme.Motion.snappy) { recordCard = .replay }
    }

    private func closeRecordCard() {
        withAnimation(Theme.Motion.snappy) { recordCard = nil }
    }

    /// Folder chips: the one in use first, then the rest A→Z.
    private var recordFolders: [String] {
        let all = controller.serverTasks.filter { $0 != "default" || (controller.taskCounts[$0] ?? 0) > 0 }
        return all.sorted { a, b in
            if a == controller.recordTask { return true }
            if b == controller.recordTask { return false }
            return a < b
        }
    }

    @ViewBuilder private var recordCardOverlay: some View {
        switch recordCard {
        case .record:
            RecordStartCard(folder: $recordFolder, folders: recordFolders,
                            counts: controller.taskCounts,
                            onCancel: closeRecordCard,
                            onStart: { folder, episode in startRecording(into: folder, episode: episode) })
        case .replay:
            ReplayCard(clips: controller.demoClips, live: controller.connection.isLive,
                       repeats: $replayRepeats,
                       onPlay: { name in
                           controller.playDemo(name, repeats: replayRepeats)
                           closeRecordCard()
                           Haptics.rigid()
                       },
                       onDelete: { name in controller.deleteDemo(name) },
                       onClose: closeRecordCard)
                .onAppear { controller.refreshDemoClips() }
        case nil:
            EmptyView()
        }
    }

    private func startRecording(into folder: String, episode: Int) {
        controller.selectTask(folder)
        controller.startEpisode()
        recordFolder = folder
        take = (folder, episode, Date())
        closeRecordCard()
        Haptics.rigid()
    }

    private func stopWhateverIsRunning() {
        Haptics.rigid()
        if controller.recording {
            controller.endEpisode()
            take = nil
        } else if controller.demo.playing {
            controller.stopDemo()
        }
    }

    /// Drive · Joint · Task — all available in both orientations (landscape is
    /// just the roomier drive cockpit). Prominent = the same large metrics as
    /// every other primary control.
    private var spaceToggle: some View {
        GlassSegmented(options: TeleopSpace.allCases,
                       selection: Binding(get: { space }, set: { setSpace($0) }),
                       title: { isLandscape ? $0.title : "" }, icon: { $0.icon },
                       prominent: false)
            .fixedSize()
    }

    /// Switch Task-space method. The two are mutually exclusive controllers, so
    /// the outgoing one is fully torn down: a staged plan is discarded (it
    /// belongs to the wrist planner and must never survive into live
    /// retargeting) and body tracking is stopped before the planner takes over.
    private func setTaskMode(_ m: TaskMode) {
        guard m != taskMode else { return }
        if armPlan != nil { cancelPlanToken += 1 }
        if m == .wrist, teleopActive { stopTeleop() }
        withAnimation(Theme.Motion.snappy) { taskMode = m }
        applySpaceSpec()
        Haptics.select()
    }

    /// TRUE where whole-body retargeting is the live (or imminent) controller:
    /// Task·WholeBody in 3rd person, or any 1st-person / fully-immersive view,
    /// where whole body IS the mode. Gates the Local/Global switch, which is
    /// meaningless under the wrist planner or plain drive.
    /// Recalibrate needs a live tracking session to measure with, and is a
    /// 3rd-person job (the pilot view is the picture).
    private var canRecalibrate: Bool {
        controller.immersiveActive && teleopActive && pov != .first
    }

    private var wholeBodyEngaged: Bool {
        (space == .task && taskMode == .wholeBody) || pov == .first
            || controller.fullImmersion || teleopActive
    }

    /// The control-law dropdown: FEEL (Stiff | Soft) plus the two law switches.
    /// A native `Menu`, not a sheet — three quick switches did not deserve a
    /// half-screen modal with explanatory paragraphs. Rows are a noun and a
    /// state. The dot marks any non-default so the operator can see the robot is
    /// not in its normal configuration without opening it.
    private var tuningButton: some View {
        Menu {
            Picker("Feel", selection: Binding(get: { feel }, set: { setFeel($0) })) {
                Label("Stiff", systemImage: "bolt.horizontal.fill").tag(ControllerType.impedance)
                Label("Soft", systemImage: "wind").tag(ControllerType.compliance)
            }
            .pickerStyle(.inline)
            Divider()
            Toggle(isOn: Binding(get: { controller.gravityComp },
                                 set: { controller.setGravityComp($0) })) {
                Label("Gravity Comp", systemImage: "arrow.down.circle")
            }
            Divider()
            // The arm retargeting PROFILE, named for what each one tracks rather
            // than as "whole-body on/off" — the interesting mode was the OFF state
            // of a negative toggle, which is a poor way to offer a choice.
            // Maps to controller/mabel/teleop_engine/arm.py PROFILES:
            //   Posture = "whole_body" — follows your arm, elbow included.
            //   Wrist   = "wrist"     — palm goes where you point it; the posture
            //                           table resolves the elbow (palm error p90
            //                           4.6 deg -> 1.4 deg, no unconverged flips).
            Picker("Retargeting",
                   selection: Binding(get: { controller.wholeBodyRetarget },
                                      set: { controller.setWholeBodyRetarget($0) })) {
                Label("Posture", systemImage: "figure.stand").tag(true)
                Label("Wrist", systemImage: "hand.point.up.left.fill").tag(false)
            }
            .pickerStyle(.inline)
        } label: {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 44, height: 44)
                if !controller.gravityComp || controller.wholeBodyRetarget || feel != .impedance {
                    Circle().fill(Theme.Palette.warn)
                        .frame(width: 9, height: 9)
                        .offset(x: -2, y: 2)
                }
            }
            .contentShape(Circle())
        }
        .menuOrder(.fixed)
        .accessibilityLabel("Control tuning")
        .accessibilityIdentifier("btn.controlTuning")
    }

    // (Torque button removed — on the wire, motor torque follows the e-stop,
    // so the status bar's STOP button already owns that state.)

    /// "Add Window" — spawn control-center-style aux windows the operator can
    /// place anywhere in the room: single camera streams, the joint bar graph,
    /// the 2-minute time series, the 2D occupancy map, and the volumetric
    /// 3D mesh map.
    @State private var showAddWindow = false
    /// The task hint auto-dismisses; the token re-arms it on a fresh entry.
    @State private var hintVisible = true
    @State private var hintToken = 0

    /// Cameras · Joints · Trends — a VERTICAL rail on the trailing scene ornament,
    /// vertically centred, mirroring the leading viewer rail.
    ///
    /// They now open REAL WINDOWS, not ornament panels (2026-07-16). As panels they
    /// extended the main window rightward, and with three open the app became an
    /// enormous slab that dominated the room — the data crowded out the viewer it
    /// was supposed to annotate. Windows are what visionOS has FOR this: the app
    /// stays exactly viewer-sized, the operator puts the cameras above the robot and
    /// the trends off to one side, and every panel stays available. The WindowGroups
    /// already existed (`AuxWindows.swift`); Add Window opened them by hand.
    ///
    /// `openWindow` on an already-open id is a no-op, so the rail tracks its own
    /// idea of open/closed to drive the toggle state and the dismiss.
    private var quickPanelToggles: some View {
        VStack(spacing: 8) {
            // CONTROLS first — they move the robot; the data panels below only observe.
            // Each appears only where its control can act (torque on, Stiff), so a lit
            // toggle never stands for a dead control.
            if torqueOn, feel == .impedance {
                railToggle("Drive", "steeringwheel", active: drivePadShown) {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { drivePadShown.toggle() }
                }
                .accessibilityIdentifier("rail.drivePad")
                if space == .joint {
                    railToggle("Sliders", "slider.horizontal.3", active: jointPanelsShown) {
                        Haptics.select()
                        // A neck pad held over the fold must not keep the head turning.
                        if jointPanelsShown { padC = .zero; rollDir = 0 }
                        withAnimation(Theme.Motion.snappy) { jointPanelsShown.toggle() }
                    }
                    .accessibilityIdentifier("rail.jointPanels")
                }
                Rectangle().fill(Theme.Palette.hairline).frame(width: 36, height: 1)
                    .padding(.vertical, 2)
            }
            railToggle("Cams", "video.fill", active: showCamsPanel) {
                toggleAux(.auxSensors, $showCamsPanel)
            }
            railToggle("Joints", "chart.bar.fill", active: showBarsPanel) {
                toggleAux(.jointBars, $showBarsPanel)
            }
            railToggle("Trends", "waveform.path.ecg", active: showSeriesPanel) {
                toggleAux(.timeSeries, $showSeriesPanel)
            }
        }
    }

    private func toggleAux(_ id: WindowID, _ flag: Binding<Bool>) {
        Haptics.select()
        if flag.wrappedValue { dismissWindow(id: id.rawValue) }
        else { openWindow(id: id.rawValue) }
        withAnimation(Theme.Motion.snappy) { flag.wrappedValue.toggle() }
    }

    /// Open (or focus) the placeable camera-detail window for one stream key —
    /// value-parameterized, so each camera gets its own window.
    private func openCameraDetail(_ key: String) {
        openWindow(id: WindowID.cameraDetail.rawValue, value: key)
    }

    /// What stays ATTACHED to the trailing ornament: the small PICKERS only —
    /// Joint space's MODULE + slider panels and the Task/Soft CONTROL REGION.
    ///
    /// These are controls you reach for while looking at the robot, so they belong
    /// on the window. The DATA panels (cameras, joint bars, trends) are windows now
    /// — hanging them here too made the app a slab (see `quickPanelToggles`).
    @ViewBuilder private var panelStack: some View {
        HStack(alignment: .top, spacing: 10) {
            if jointPanelsVisible {
                modulePanel
                jointControlsPanel
            }
            if regionPanelVisible { regionPanel }
        }
    }

    /// Vertical camera rail for the trailing ornament: head + wrists + base.
    /// Tap any tile → its own placeable camera-detail window (zoom + pan).
    private var camerasSidePanel: some View {
        VStack(spacing: 8) {
            Text("CAMERAS").font(.system(size: 10, weight: .heavy)).tracking(1.6)
                .foregroundStyle(Theme.Palette.textTertiary)
            CameraFeed(name: "HEAD", compact: true,
                       streamURL: controller.videoThumbURL("main"),
                       stream: "main",
                       onExpand: { openCameraDetail("main") })
                .frame(width: 240, height: 150)
            CameraFeed(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                       streamURL: controller.videoThumbURL("wrist_left"),
                       stream: "wrist_left",
                       onExpand: { openCameraDetail("wrist_left") })
                .frame(width: 240, height: 150)
            CameraFeed(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                       streamURL: controller.videoThumbURL("wrist_right"),
                       stream: "wrist_right",
                       onExpand: { openCameraDetail("wrist_right") })
                .frame(width: 240, height: 150)
            CameraFeed(name: "BASE", compact: true, tint: Theme.Palette.accent,
                       streamURL: controller.videoThumbURL("front"),
                       stream: "front",
                       onExpand: { openCameraDetail("front") })
                .frame(width: 240, height: 150)
        }
        .padding(12)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 22, style: .continuous))
    }

    private var addWindowMenu: some View {
        Button {
            Haptics.select(); showAddWindow.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "plus.square.on.square").font(.system(size: 13, weight: .bold))
                Text("Add Window").font(.system(size: 13, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 13).padding(.vertical, 9)
            .contentShape(Capsule(style: .continuous))
            .glassBackgroundEffect(in: Capsule(style: .continuous))
            .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.18), lineWidth: 1) }
            .contentShape(.hoverEffect, Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .hoverEffect()
        .popover(isPresented: $showAddWindow, arrowEdge: .top) {
            addWindowPicker.presentationCompactAdaptation(.popover)
        }
    }

    /// Stable popover list (the system Menu flickered under the 8 Hz uiTick
    /// re-renders) — same chrome as the saved-maps picker.
    private var addWindowPicker: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("ADD WINDOW").font(.system(size: 9, weight: .heavy)).tracking(1.2)
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
            addWindowRow("Cameras & Sensors", "camera.on.rectangle.fill") {
                openWindow(id: WindowID.auxSensors.rawValue)
            }
            addWindowRow("Joint Bars · Pos/Vel/Torque", "chart.bar.fill") {
                openWindow(id: WindowID.jointBars.rawValue)
            }
            addWindowRow("Time Series · 2 min", "chart.xyaxis.line") {
                openWindow(id: WindowID.timeSeries.rawValue)
            }
            addWindowRow("2D Occupancy Map", "grid") {
                openWindow(id: WindowID.map2D.rawValue)
            }
            addWindowRow("3D Mesh Map (Volume)", "cube.transparent") {
                openWindow(id: WindowID.mapVolume.rawValue)
            }
        }
        .frame(width: 250)
        .padding(.bottom, 6)
        .presentationBackground(.ultraThinMaterial)
    }

    private func addWindowRow(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select(); showAddWindow = false; action()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Palette.accent)
                    .frame(width: 22)
                Text(title).font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                Spacer()
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .hoverEffect()
    }

    /// Map-source selection    /// Map-source selection — connected: live SLAM / robot-saved maps; offline:
    /// the LOCAL demo map (nvblox-style mesh + grid, goal-driven robot).
    private func selectLive() {
        Haptics.select()
        if store.hasEndpoint { controller.goLiveSLAM() } else { store.stopLocalDemo() }
        withAnimation(Theme.Motion.snappy) { selected = .live }
    }

    private func selectRoom(_ room: SLAMRoom) {
        Haptics.select()
        if store.hasEndpoint { controller.loadMap(room.name) } else { store.restartLocalDemo() }
        withAnimation(Theme.Motion.snappy) { selected = room }
    }

    /// ONE map-source chip (iOS parity, 2026-07-11 streamline): shows the
    /// ACTIVE source ("Live SLAM" while streaming, "Offline Map" when not, or
    /// the loaded map's name) and drops down to the live row + every saved
    /// map/demo room + Restart SLAM — the old two-pill Live/Saved pair is gone.
    private var mapSourceMenu: some View {
        Button {
            Haptics.select()
            // The catalogue only exists once we ask — list_maps on every open,
            // so a freshly saved map appears without reconnecting.
            if connectedToRobot { controller.requestMapList() }
            showSavedMaps.toggle()
        } label: {
            pill(isLandscape ? mapSourceTitle : nil,
                 selected.isLive ? "dot.radiowaves.left.and.right" : "square.stack.3d.up.fill",
                 active: selected.isLive && slamLive, chevron: true)
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        .popover(isPresented: $showSavedMaps, arrowEdge: .top) {
            savedMapsPicker.presentationCompactAdaptation(.popover)
        }
        #if DEBUG
        .task {
            // Screenshot harness: `defaults write <bundle> debugPopup -string maps`.
            guard UserDefaults.standard.string(forKey: "debugPopup") == "maps" else { return }
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            showSavedMaps = true
        }
        #endif
    }

    private var mapSourceTitle: String {
        selected.isLive ? (slamLive ? "Live SLAM" : "Offline Map") : selected.name
    }

    /// Is map data actually flowing right now? (Real endpoint: the map stream
    /// is connected; offline: the local demo is running.)
    private var slamLive: Bool {
        store.hasEndpoint ? store.connected : store.demo
    }

    /// The map-source dropdown, built from the SAME rows as the status bar's
    /// connection picker (`MabelMenuSection` / `MabelMenuRow`, 272 pt, ultra-thin
    /// material) so the three top-bar dropdowns read as one family (2026-10-25).
    private var savedMapsPicker: some View {
        VStack(alignment: .leading, spacing: 0) {
            MabelMenuSection(title: "Live")
            // Live SLAM first — with its actual streaming status.
            MabelMenuRow(icon: "dot.radiowaves.left.and.right", title: "Live SLAM",
                         detail: slamLive ? "streaming" : "offline — no map data",
                         live: selected.isLive,
                         tint: slamLive ? Theme.Palette.live : nil) {
                selectLive(); showSavedMaps = false
            }
            .accessibilityIdentifier("maps.live")
            MabelMenuDivider()
            MabelMenuSection(title: mapSectionTitle)
            if savedRooms.isEmpty {
                MabelMenuNote(text: "No saved maps on the robot — run SLAM and save one.")
            } else {
                ForEach(savedRooms) { room in savedRow(room) }
            }
            MabelMenuDivider()
            // Restart SLAM: fresh map + localization from the CURRENT pose.
            MabelMenuRow(icon: "arrow.counterclockwise.circle.fill", title: "Restart SLAM",
                         detail: "New map + localization from here", tint: Theme.Palette.warn) {
                Haptics.rigid()
                controller.resetSlam()
                withAnimation(Theme.Motion.snappy) { selected = .live }
                showSavedMaps = false
            }
            .accessibilityIdentifier("maps.restart")
        }
        .mabelMenuPanel()
    }

    private func savedRow(_ room: SLAMRoom) -> some View {
        let active = !selected.isLive && selected.name == room.name
        return MabelMenuRow(icon: room.icon, title: room.name,
                            detail: active ? "loaded — localizing against it" : "",
                            live: active) {
            selectRoom(room)
            showSavedMaps = false
        }
        .accessibilityIdentifier("maps.room.\(room.name)")
    }

    /// The 180° wrap-around view — opens the immersive teleop space (hand
    /// tracking + curved stereo head wrap) and dials it to full immersion.
    /// Starts teleop too if it isn't already running (v1 behaviour).
    /// PROMINENT variant — 1st person's primary action. In the pilot view the
    /// one thing the operator is reaching for is "put me inside it", so it gets
    /// the accent fill and the centre of the bar rather than sitting as one more
    /// grey capsule among the view controls (operator, 2026-07-28: "the 1st
    /// person view full immersive mode should be front and center").
    private var fullImmersiveCTA: some View {
        Button {
            Haptics.rigid()
            // DEFER ONLY FROM A COLD START. `deferBodyEngage` exists so opening
            // the dome does not silently start mirroring — but if teleop is
            // ALREADY running this is a VIEW change, not a new session, and the
            // engagement has to survive it. It did not: flipping
            // `.immersionStyle` RE-CREATES the immersive space, so
            // `stopImmersiveTracking` (which clears `bodyTracking`) ran and the
            // restart then saw the deferral and never re-engaged. The head
            // streams unconditionally while the hands are gated on
            // `bodyTracking`, so the robot kept following the operator's head
            // with its arms frozen — exactly the report (operator, 2026-07-28).
            controller.deferBodyEngage = !controller.bodyTracking
            Task {
                controller.fullImmersion = true
                // `needsImmersiveOpen`, not `!immersiveActive`: the cockpit may
                // already have this space open as its stick host, which claims
                // no session — that one is PROMOTED in place, never re-opened.
                if controller.needsImmersiveOpen() {
                    _ = await openImmersive(id: WindowID.immersive.rawValue)
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "visionpro").font(.system(size: 16, weight: .bold))
                Text("Enter Full Immersive")
                    .font(.system(size: 15, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 22)
            .frame(height: 44)
            .background(Theme.accentGradient, in: Capsule(style: .continuous))
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }

    private var fullImmersiveButton: some View {
        Button {
            Haptics.rigid()
            // The dome opens WITHOUT engaging body tracking: the operator must
            // press Start Teleop (visible in full immersion) to begin full-body
            // mirroring — safety confirm + arm-extension calibration first.
            // DEFER ONLY FROM A COLD START. `deferBodyEngage` exists so opening
            // the dome does not silently start mirroring — but if teleop is
            // ALREADY running this is a VIEW change, not a new session, and the
            // engagement has to survive it. It did not: flipping
            // `.immersionStyle` RE-CREATES the immersive space, so
            // `stopImmersiveTracking` (which clears `bodyTracking`) ran and the
            // restart then saw the deferral and never re-engaged. The head
            // streams unconditionally while the hands are gated on
            // `bodyTracking`, so the robot kept following the operator's head
            // with its arms frozen — exactly the report (operator, 2026-07-28).
            controller.deferBodyEngage = !controller.bodyTracking
            Task {
                // Set the style FIRST so the space OPENS directly in full
                // immersion — switching styles mid-open left the system black
                // (the device bug); v1 only ever switched an already-open space.
                controller.fullImmersion = true
                // `needsImmersiveOpen`, not `!immersiveActive`: the cockpit may
                // already have this space open as its stick host, which claims
                // no session — that one is PROMOTED in place, never re-opened.
                if controller.needsImmersiveOpen() {
                    _ = await openImmersive(id: WindowID.immersive.rawValue)
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "visionpro").font(.system(size: 14, weight: .bold))
                Text("Full Immersive").font(.system(size: 13, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .frame(height: 36)   // top-bar control standard
            .contentShape(Capsule(style: .continuous))
            .glassBackgroundEffect(in: Capsule(style: .continuous))
            .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.18), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }

    // MARK: - Left control rail (scene ornament)

    /// The viewer control column — INDIVIDUAL floating glass buttons (not one
    /// continuous strip), stacked vertically beside the app nav.
    ///
    /// TWO GROUPS, separated by a hairline, exactly like the web control
    /// center's `viewrail.js` (which was itself written as the web twin of this
    /// rail — they have now converged):
    ///
    ///   VIEW    Center · Fit · Follow      the three camera verbs — where you look
    ///   SCENE   Floor · Grid · Cloud▸ · …  what is drawn
    ///
    /// Every layer is its OWN button, not a row buried in one "Overlays" popover
    /// (operator, 2026-07-28: "expand overlays for me — keep it 3 buttons for
    /// the camera viewer, then all the buttons for the visualization toggles").
    /// A layer with sub-options carries a small caret and EXPANDS RIGHTWARD into
    /// a flyout, so the sub-options never steal a row from the rail itself —
    /// the web's `rail-flyout`, and the reason Cam 3D and Cloud can each own a
    /// per-camera list without the rail growing.
    ///
    /// Tapping the BUTTON toggles its layer; tapping the CARET opens the
    /// flyout. That is the web's two-hit-zone rule, and it matters: a caret that
    /// also flipped the layer would make "show me the wrist cloud" turn the
    /// whole cloud layer off on the way in.
    /// VOICE SURVIVES THE POV FLIP; nothing else does (2026-07-29).
    ///
    /// Every other control on this rail acts on the 3rd-person STAGE — Center,
    /// Fit and Follow move a camera that first person does not have, and the
    /// layer toggles show or hide world overlays the operator is now standing
    /// inside and cannot see. Offering them from inside the head is offering
    /// controls for a view that is not on screen.
    ///
    /// Talking to the person in front of the robot is not a 3rd-person activity.
    /// It is the one thing an operator is MOST likely to need while looking
    /// through the robot's eyes, and the whole rail used to vanish on the flip —
    /// so first person had no way to speak at all without flipping back.
    private var leftRail: some View {
        VStack(spacing: 8) {
            if pov == .third {
                railButton("Center", "scope") { recenterToken += 1 }
                railButton("Fit", "arrow.up.left.and.arrow.down.right") { fitToken += 1 }
                railToggle("Follow", "video.fill", active: following) {
                    Haptics.select(); following.toggle()
                }
                railDivider
                // ONE button, not ten. The ten individual layer buttons made the
                // rail taller than the robot and put a wall of toggles between the
                // operator and the three verbs they use constantly (Center, Fit,
                // Follow) — the iPhone console already solved this by folding the
                // same set behind a pull-up. Same idea, sideways: the rail keeps
                // the verbs, and the visualisation toggles live one tap away.
                railToggle("Layers", "square.3.layers.3d",
                           active: showLayersPanel, caret: true) {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { showLayersPanel.toggle() }
                }
                .popover(isPresented: $showLayersPanel, arrowEdge: .leading) {
                    layersPanel.presentationCompactAdaptation(.popover)
                }
                railDivider
            }
            voiceRail
        }
        // The rail SHRINKS to two buttons on the flip rather than emptying and
        // refilling — same position, same size, so the operator's hand does not
        // have to re-find Talk.
        .animation(.easeInOut(duration: 0.22), value: pov)
    }

    @State private var showLayersPanel = false

    /// Every visualisation toggle, in ONE panel — the iPhone console's LAYERS
    /// section, laid out for a popover.
    ///
    /// Grouped the way the iPhone groups them, because the split is what makes
    /// the list readable: WORLD layers are stream-fed and can legitimately be
    /// empty, STAGE layers are generated locally and always draw something. The
    /// per-source sub-options (which cameras get a projection, which get a cloud)
    /// stay INLINE here rather than behind their own flyouts — a popover inside a
    /// popover is a place taps go to die on visionOS, and it was the only reason
    /// those two layers needed a caret in the first place.
    @ViewBuilder private var layersPanel: some View {
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                consoleHeader("WORLD")
                ForEach([RailLayer.mesh, .grid, .cloud, .laser, .cam]) { l in
                    layerRow(l.title, l.icon, railLayerBinding(l))
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                consoleHeader("STAGE")
                ForEach([RailLayer.robot, .floor, .floorGrid, .minimap, .tf]) { l in
                    layerRow(l.title, l.icon, railLayerBinding(l))
                }
            }
            // The two layers that used to own their own flyout bring their
            // sub-options INLINE, revealed only when the layer is on. Dropping
            // them when the flyouts went would have quietly removed cloud
            // density and the Cam 3D image size — controls with no other home.
            if camProjections {
                VStack(alignment: .leading, spacing: 6) {
                    consoleHeader("CAM 3D · CAMERAS")
                    layerRow("Head", "camera.metering.matrix", $camHeadOn)
                    layerRow("L Wrist", "hand.point.left.fill", $camWristLOn)
                    layerRow("R Wrist", "hand.point.right.fill", $camWristROn)
                    layerRow("Base", "car.rear.road.lane", $camBaseOn)
                    consoleHeader("IMAGE SIZE")
                    HStack(spacing: 8) {
                        Slider(value: $camProjSize, in: 0.1...1.0)
                        Text(String(format: "%.2fm", camProjSize))
                            .font(.mabelMonoSmall)
                            .foregroundStyle(Theme.Palette.textSecondary)
                            .frame(width: 46, alignment: .trailing)
                    }
                    GlassSegmented(options: ProjSurface.allCases,
                                   selection: Binding(get: { camFlat ? ProjSurface.rectified : .lens },
                                                      set: { camFlat = ($0 == .rectified); Haptics.select() }),
                                   title: { $0.title }, icon: { $0.icon })
                }
            }
            if cloudOn {
                VStack(alignment: .leading, spacing: 6) {
                    consoleHeader("CLOUD · CAMERAS")
                    ForEach(LiveMapStore.CloudSource.allCases) { src in
                        layerRow(src.title, src.icon,
                                 Binding(get: { store.cloudSourceOn(src) },
                                         set: { store.setCloudSource(src, $0)
                                                if $0 { cloudOn = true } }))
                    }
                    Text("Depth planes are direct-link only (Wi-Fi / Tailscale).")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                    consoleHeader("DENSITY")
                    GlassSegmented(options: CloudDensity.allCases, selection: $cloudDensity,
                                   title: { $0.title }, icon: { _ in nil })
                }
            }
        }
        .padding(14)
        }
        .frame(width: 268)
        // The panel grows with the sub-sections, so it SCROLLS rather than
        // running off the bottom of a popover that cannot be dragged.
        .frame(maxHeight: 620)
        .presentationBackground(.ultraThinMaterial)
    }

    /// VOICE — the third rail group, mirroring the web console's (`viewrail.js`
    /// `RAIL_GROUPS[2]`): talk THROUGH the robot, and listen TO its room.
    ///
    /// Both are SWITCHES (operator, 2026-07-28: "make them like a switch — switch
    /// on to talk; and mute un-mute"), where the web and the iOS console use
    /// push-to-talk. A latched mic is a hot mic in a real room, so Talk states it
    /// loudly — filled, and the glyph animates while it is open — and any loss of
    /// the voice path drops it (`RobotController.stopVoice`).
    ///
    /// Both DIM when there is no voice path, with the reason on hover, rather
    /// than offering a live-looking control down which nothing can flow.
    @ViewBuilder private var voiceRail: some View {
        let live = controller.audioAvailable
        railToggle(controller.talking ? "Talking" : "Talk",
                   controller.talking ? "waveform" : "mic.fill",
                   active: controller.talking) {
            guard live else { return }
            Haptics.rigid()
            controller.setTalking(!controller.talking)
        }
        .disabled(!live)
        .opacity(live ? 1 : 0.4)
        .help(live ? "Talk through the robot's speaker"
                   : "No voice path to this robot (:9610)")
        .symbolEffect(.variableColor, isActive: controller.talking)

        railToggle(controller.micEnabled ? "Listen" : "Muted",
                   controller.micEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill",
                   active: controller.micEnabled) {
            guard live else { return }
            Haptics.select()
            controller.setMicEnabled(!controller.micEnabled)
        }
        .disabled(!live)
        .opacity(live ? 1 : 0.4)
        .help(live ? "Hear the robot's room (mutes at the robot, not locally)"
                   : "No voice path to this robot (:9610)")
    }

    /// One scene layer. Rendered as a row in `layersPanel`.
    enum RailLayer: String, CaseIterable, Identifiable {
        // World (stream-fed) first, then the STAGE — the robot and the ground it
        // stands on, which are always drawable. iPhone console parity, 2026-07-29.
        case mesh, grid, cloud, laser, cam, robot, floor, floorGrid, tf, minimap
        var id: String { rawValue }
        var title: String {
            switch self {
            case .floor:    return "Floor"
            case .floorGrid: return "Grid"
            case .grid:     return "2D Map"
            case .cloud:    return "Cloud"
            case .laser:    return "LiDAR"
            case .mesh:     return "Mesh"
            case .cam:      return "Cam 3D"
            case .robot:    return "Robot"
            case .tf:       return "TF"
            case .minimap:  return "Minimap"
            }
        }
        var icon: String {
            switch self {
            case .floor:    return "circle.dashed"
            case .floorGrid: return "grid"
            case .grid:     return "squareshape.split.3x3"
            case .cloud:    return "aqi.medium"
            case .laser:    return "rays"
            case .mesh:     return "square.3.layers.3d"
            case .cam:      return "camera.metering.matrix"
            case .robot:    return "figure.stand"
            case .tf:       return "move.3d"
            case .minimap:  return "map"
            }
        }
    }

    private func railLayerBinding(_ l: RailLayer) -> Binding<Bool> {
        switch l {
        case .floor:    return $floorOn
        case .floorGrid: return $floorGridOn
        case .grid:     return $gridOn
        case .cloud:    return $cloudOn
        case .laser:    return $laserOn
        case .mesh:     return $meshOn
        case .robot:    return $robotOn
        case .tf:       return $tfOn
        case .minimap:  return $minimapOn
        case .cam:
            // Cam 3D has no bool of its own — it IS "any camera selected", so
            // the master toggle remembers the last selection and restores it.
            return Binding(get: { camProjections },
                           set: { on in
                               if on {
                                   if camStreamSet.isEmpty { camHeadOn = true }
                               } else {
                                   camHeadOn = false; camWristLOn = false
                                   camWristROn = false; camBaseOn = false
                               }
                           })
        }
    }

    // `railLayerButton` and `railFlyoutBody` lived here — one rail button per
    // layer, each expanding rightward into its own popover. Replaced by the
    // single `Layers` button + `layersPanel` (2026-07-30): ten buttons made the
    // rail taller than the robot, and the per-layer popovers are why Cam 3D and
    // Cloud needed carets. Their contents live inline in the panel now.

    /// Which rail layer's flyout is open (at most one, like the web rail's
    /// single `openId` — two open panels beside a 3D view is a wall).

    private func consoleHeader(_ t: String) -> some View {
        Text(t).font(.system(size: 10, weight: .heavy)).tracking(1.4)
            .foregroundStyle(Theme.Palette.textTertiary)
    }

    /// One full-width toggle row inside a flyout (icon · name · state dot).
    private func layerRow(_ title: String, _ icon: String, _ on: Binding<Bool>) -> some View {
        Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { on.wrappedValue.toggle() }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 13, weight: .semibold))
                    .frame(width: 22)
                Text(title).font(.system(size: 13, weight: .semibold))
                Spacer(minLength: 0)
                Circle().fill(on.wrappedValue ? Theme.Palette.live : Theme.Palette.surface)
                    .frame(width: 8, height: 8)
            }
            .foregroundStyle(on.wrappedValue ? .white : Theme.Palette.textSecondary)
            .padding(.horizontal, 11).padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(on.wrappedValue ? Theme.Palette.accent.opacity(0.28)
                                          : Theme.Palette.surface.opacity(0.5))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 10, style: .continuous))
        .hoverEffect()
    }

    /// The 3rd/1st view toggle — the SAME GlassSegmented as Drive·Joint·Task
    /// (identical size and style, one shared control vocabulary).
    private var topPovToggle: some View {
        GlassSegmented(options: [PovMode.third, PovMode.first],
                       selection: Binding(get: { pov }, set: { setPov($0) }),
                       title: { $0 == .third ? "3rd" : "1st" },
                       icon: { $0 == .third ? "video.fill" : "eye.fill" })
            .fixedSize()
    }

    /// Thin group separator: view toggle · camera controls · mapping.
    private var railDivider: some View {
        Rectangle().fill(.white.opacity(0.22)).frame(width: 34, height: 1)
            .padding(.vertical, 2)
    }

    /// One free-standing glass button (its own capsule chrome).
    private func railButton(_ title: String, _ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 15, weight: .semibold))
                Text(title).font(.system(size: 9, weight: .semibold)).lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(Theme.Palette.textPrimary)
            .frame(width: 56, height: 50)
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 14, style: .continuous))
        .hoverEffect()
    }

    /// A rail toggle — ONE 56 pt button, always, with the glyph and label
    /// centred exactly like every other button on the rail.
    ///
    /// `caret` marks it as EXPANDABLE, and on visionOS that means the WHOLE
    /// button opens the flyout; the chevron is a corner affordance, not a
    /// control. The web splits the button into two hit zones because a mouse has
    /// pixel precision — gaze-and-pinch does not, and a 14 pt sliver is far
    /// under the ~44 pt target visionOS wants. Split that way it was unhittable
    /// AND it shrank the glyph column to 42 pt, so the two expandable buttons
    /// sat visibly off-centre against their neighbours (operator, 2026-07-28:
    /// "I cannot click and open the individual cameras… the button looks offset
    /// compared to the other buttons"). The layer's own on/off moved INTO the
    /// flyout as its first row, so nothing is lost.
    private func railToggle(_ title: String, _ icon: String, active: Bool,
                            caret: Bool = false,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 15, weight: .semibold))
                Text(title).font(.system(size: 9, weight: .bold)).lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(active ? .white : Theme.Palette.textPrimary)
            .frame(width: 56, height: 50)
            .overlay(alignment: .topTrailing) {
                if caret {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 8, weight: .heavy))
                        .foregroundStyle(active ? .white.opacity(0.85) : Theme.Palette.textTertiary)
                        .padding(.top, 5).padding(.trailing, 5)
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .background {
                if active {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Theme.accentGradient)
                }
            }
            .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(active ? 0.25 : 0.12), lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 14, style: .continuous))
        .hoverEffect()
    }

    // MARK: - Teleop
    // MARK: - Teleop

    private func startTeleop() {
        applySpaceSpec()
        withAnimation(Theme.Motion.snappy) { teleopActive = true }
        applyStagePreset(for: space)
        syncTrackingSpace(for: space)
        Haptics.rigid()
    }

    /// Task space IS whole-body teleop: it needs the MIXED immersive space open
    /// in the background (ARKit hands/head only track inside a space). Other
    /// spaces close it again — a live ARKit head would stomp the neck sliders.
    private func syncTrackingSpace(for s: TeleopSpace) {
        if s == .task && teleopActive {
            // NO auto-open: ARKit hands need our own ImmersiveSpace, and any
            // app space EVICTS a system Environment (Moon…) and takes the
            // Digital Crown — a platform rule. Task space therefore leaves
            // Apple's side untouched; the operator opts into tracking with the
            // button on the whole-body card (and gets Moon back on leaving).
        } else if controller.immersiveActive && !controller.fullImmersion {
            controller.trackingOnlyMode = false
            controller.noteIntendedImmersiveExit()
            Task { await dismissImmersive() }         // restores the system env
        }
    }

    // MARK: - The drive pad

    /// Is the flat 2-D DRIVE PAD (`DrivePadDock`: TRANSLATE · LIFT · ROTATE)
    /// on the window's bottom ornament? In EVERY space and either POV
    /// (operator, 2026-10-27: "merge drive with joint and task … those joysticks
    /// should always be persistent … so that even under joint and task space I
    /// can still teleop the robot"). Both gateways take a nav stick in any
    /// control mode: NOVA's `_body_follow` applies it whatever `method` is, and
    /// the MABEL bridge forces `navigation` while a stick (now incl. the lift)
    /// is deflected — and a stick beats whole-body follow on both. Same gate as every
    /// stick before it: motors live, Stiff (Soft is passive), this tab on
    /// screen. Folding it is the operator's call (`DrivePadDock`'s Hide).
    ///
    /// It REPLACED the floating `Joystick3D` pair and the immersive "stick
    /// host" that opened the space just to float them (operator, 2026-10-16:
    /// "do not use the 3d joystick — keep things 2D and easy to toggle and
    /// similar to the ios"). Full immersion unmounts this view; the dome
    /// mounts its own `DrivePadDock` attachment.
    private var drivePadWanted: Bool {
        isActiveTab && torqueOn && feel == .impedance
    }

    /// The bottom ornament's content. An ORNAMENT, never in-window chrome: the
    /// true-3D stage protrudes through the window plane and occludes anything
    /// drawn in the window body (why the Joint panels moved to the trailing
    /// ornament) — the pad must stay reachable over a robot filling the view.
    @ViewBuilder private var drivePadOrnament: some View {
        if drivePadWanted {
            DrivePadDock()
                .padding(.top, 18)
                .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    /// Explicit opt-in: SAFETY CONFIRM first, then the invisible tracking
    /// session (hands + neck) + the arm-extension calibration.
    private func startHandTracking() {
        guard !controller.immersiveActive else { return }
        showWholeBodyConfirm = true
    }

    /// Operator said YES: teleop on, tracking space open, then the GUIDED
    /// arm-extension calibration POP-UP (the Settings/v1 CalibrationFlowView —
    /// intro → arms-forward capture → measured summary). A floating sheet,
    /// never bottom-left card text (operator feedback).
    private func confirmWholeBody() {
        Haptics.rigid()
        showWholeBodyConfirm = false
        if !teleopActive { startTeleop() }
        if !controller.immersiveActive {
            controller.trackingOnlyMode = true
            // MEASURE FIRST, MIRROR AFTER (operator 2026-07-20): the space opens
            // with tracking services live but the robot NOT mirroring — the
            // calibration card's 3·2·1 countdown is now the ONE engage point
            // (engageWholeBody), so the robot never moves mid-capture.
            controller.deferBodyEngage = true
            // Both flags are set BEFORE `needsImmersiveOpen`, because promoting
            // the cockpit's stick host starts its tracking session immediately —
            // and that start is what consumes `deferBodyEngage`.
            if controller.needsImmersiveOpen() {
                Task { _ = await openImmersive(id: WindowID.immersive.rawValue) }
            }
        }
        // Space already open (full-immersive dome, deferred): nothing engages
        // here either — the calibration flow's countdown owns the engage.
        // Let the confirm card animate out, then hand over to the pop-up flow
        // (it waits for tracking itself before starting the capture).
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
            showCalibrationFlow = true
        }
    }

    /// Safety gate card — same glass vocabulary as every other panel.
    private var wholeBodyConfirmCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "figure.arms.open")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(Theme.Palette.warn)
                Text("Whole-Body Tracking")
                    .font(.system(size: 16, weight: .bold)).foregroundStyle(.white)
            }
            Text("The robot is about to mirror your head, arms and fingers in real time.")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.white.opacity(0.85))
            VStack(alignment: .leading, spacing: 6) {
                confirmRow("figure.stand", "Stand in an OPEN space — arm's reach clear on every side.")
                confirmRow("eye.fill", "Keep the robot's workspace clear of people and obstacles.")
                confirmRow("hand.raised.fingers.spread.fill", "Calibration first: EXTEND BOTH ARMS straight forward — or use your saved calibration.")
                confirmRow("exclamationmark.octagon.fill", "STOP (top bar) freezes the robot at any moment.")
            }
            HStack(spacing: 10) {
                Button {
                    Haptics.select(); showWholeBodyConfirm = false
                } label: {
                    Text("Not Yet").font(.system(size: 13, weight: .bold))
                        .foregroundStyle(Theme.Palette.textPrimary)
                        .padding(.horizontal, 18).frame(height: 36)
                        .contentShape(Capsule(style: .continuous))
                        .glassBackgroundEffect(in: Capsule(style: .continuous))
                }
                .buttonStyle(PressableStyle())
                .contentShape(.hoverEffect, Capsule(style: .continuous))
                .hoverEffect()
                Button {
                    confirmWholeBody()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill").font(.system(size: 13, weight: .bold))
                        Text("Yes — I'm Clear").font(.system(size: 13, weight: .bold))
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 18).frame(height: 36)
                    .background(Theme.accentGradient, in: Capsule(style: .continuous))
                    .contentShape(Capsule(style: .continuous))
                }
                .buttonStyle(PressableStyle())
                .contentShape(.hoverEffect, Capsule(style: .continuous))
                .hoverEffect()
            }
        }
        .padding(18)
        .frame(width: 380)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 24, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 24, style: .continuous)
            .strokeBorder(.white.opacity(0.16), lineWidth: 1) }
    }

    private func confirmRow(_ icon: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon).font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.Palette.accent).frame(width: 18)
            Text(text).font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Whole-body teleop is RUNNING in 3rd person: the bar strips down to the
    /// two verbs that matter mid-session. Your body is the controller now, so a
    /// Drive/Joint/Task picker is not something you reach for — it is something
    /// you hit by accident (operator, 2026-07-28).
    private var leanTeleopBar: Bool {
        space == .task && teleopActive && pov == .third
    }

    private func stopTeleop() {
        translate = .zero; yaw = 0
        joyL = .zero; joyR = .zero; pitchL = 0; pitchR = 0
        padC = .zero; padL = .zero; padR = .zero
        controller.stopBase()
        controller.setNavigation()
        withAnimation(Theme.Motion.snappy) { teleopActive = false }
        if controller.immersiveActive && !controller.fullImmersion {
            controller.noteIntendedImmersiveExit()
            Task { await dismissImmersive() }   // tracking space dies with teleop
        } else if controller.fullImmersion {
            // Full immersion: Stop ends the MIRRORING but keeps the dome —
            // the operator stays in the view and can Start Teleop again.
            // HANDS AND HEAD: clearing `bodyTracking` alone left the head pose
            // streaming, so the neck kept following the operator after Stop.
            controller.stopWholeBodyTeleop()
        }
        Haptics.rigid()
    }

    private func setSpace(_ s: TeleopSpace) {
        guard s != space else { return }
        if space == .drive { translate = .zero; yaw = 0; controller.stopBase() }
        // Leaving Task space ends whole-body teleop (teleopActive is Task-only
        // now — Drive/Joint controls live directly off the selected space).
        if space == .task, teleopActive { stopTeleop() }
        withAnimation(Theme.Motion.snappy) { space = s }
        hintVisible = true; hintToken += 1        // a fresh entry re-arms the hint
        applySpaceSpec()
        if s == .joint { focusToken += 1 }     // fly to the focused module
        applyStagePreset(for: s)
        syncTrackingSpace(for: s)
        Haptics.select()
    }

    /// 3D-stage framing per control space: Drive = wide isometric (~5 m of
    /// scene), Task = whole-robot isometric, Joint = zoom on the focused module.
    private func applyStagePreset(for s: TeleopSpace) {
        switch s {
        case .drive: stagePreset = .isoFar
        case .task:  stagePreset = .iso
        case .joint: stagePreset = .module(module)
        }
        presetToken += 1
    }

    /// Module picker (Joint space) — a LABELED panel on the trailing ornament,
    /// shown AUTOMATICALLY whenever Joint space is selected. Selecting flies
    /// the camera to that module's framing (front-on lift, per-side arm
    /// isometric, palm-on hands, …).
    private var modulePanel: some View {
        railPanel("BODY MODULE") {
            ForEach(ModuleFocus.allCases) { m in moduleRow(m) }
        }
    }

    /// One module row.
    private func moduleRow(_ m: ModuleFocus) -> some View {
        railRow(title: m.title, icon: m.icon,
                badge: m.side.map { $0 == "left" ? "L" : "R" },
                on: module == m, highlightID: "module.sel", ns: moduleNS) {
            selectModule(m)
        }
    }

    /// The shared trailing-ornament picker row — BODY MODULE and CONTROL REGION are
    /// the same control in two places, so they are literally the same row.
    ///
    /// The gaze highlight used to read as a small, wrong-shaped chip because THREE
    /// things disagreed about the row's shape: the 128×36 frame, the accent fill's
    /// 10 pt rounded rect, and — the actual culprit — `PressableStyle`, which stamps
    /// its own `.contentShape(Capsule())` + `.hoverEffect()` onto every label it
    /// styles. That capsule became the hover shape (a pill on a rounded-rect row)
    /// and nested a SECOND hover effect inside the one the call site added. Here the
    /// fill, the hit area and the hover shape are one rounded rect on a row with
    /// real gaze area, and the selection highlight slides between rows.
    private func railRow(title: String, icon: String, badge: String? = nil,
                         on: Bool, highlightID: String, ns: Namespace.ID,
                         action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 24)
                Text(title).font(.system(size: 14, weight: .semibold))
                Spacer(minLength: 0)
                if let badge {
                    Text(badge)
                        .font(.system(size: 10, weight: .heavy))
                        .foregroundStyle(on ? .white.opacity(0.9) : Theme.Palette.textTertiary)
                }
            }
            .foregroundStyle(on ? .white : Color(hex: 0xF4F1EA).opacity(0.85))
            .padding(.horizontal, 12)
            .frame(width: 172, height: 48, alignment: .leading)
            .background {
                if on {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Theme.accentGradient)
                        .matchedGeometryEffect(id: highlightID, in: ns)
                }
            }
            // Inactive rows have no background — without an explicit shape the
            // hit area (and the highlight with it) collapses to the glyphs.
            .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .contentShape(.hoverEffect, RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .buttonStyle(.plain)      // see above: PressableStyle would re-shape the hover
        .hoverEffect()
    }

    /// Shared glass chrome for the trailing-ornament picker panels.
    private func railPanel<C: View>(_ title: String, @ViewBuilder _ rows: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 10, weight: .heavy)).tracking(1.3)
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 10).padding(.bottom, 4)
            rows()
        }
        .padding(8)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 20, style: .continuous)
            .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
    }

    /// CONTROL REGION picker — Task space's Arm / Upper / Whole. Lives on the
    /// TRAILING ornament in the same panel language as BODY MODULE (2026-07-16
    /// operator request): it is the same kind of choice, so it gets the same
    /// control. The old version was a cramped 42 pt icon rail floating over the
    /// top-left of the viewer, in a style nothing else in the app used.
    private var regionPanel: some View {
        railPanel("CONTROL REGION") {
            ForEach(ControlRegion.allCases) { r in
                railRow(title: r.title, icon: r.icon, on: region == r,
                        highlightID: "region.sel", ns: regionNS) {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { region = r }
                    if teleopActive || wristPlanning { applySpaceSpec() }
                }
            }
        }
    }

    /// Focus a body module: zero any live nudges, then fly the stage camera to
    /// that module's framing.
    private func selectModule(_ m: ModuleFocus) {
        Haptics.select()
        joyL = .zero; joyR = .zero; rollDir = 0
        padC = .zero
        controller.setLiftRate(0)
        withAnimation(Theme.Motion.snappy) { module = m }
        focusToken += 1
        stagePreset = .module(m)          // 3D stage: zoom to the part
        presetToken += 1
    }

    private func setFeel(_ f: ControllerType) {
        guard f != feel else { return }
        // Soft removes the whole stick cluster (it is PASSIVE — the controller
        // drops every command), so a deflected stick vanishes with it. Zero the
        // base on the way out for the same reason as `setPov`: nothing else
        // will, and `stopBase` is deliberately NOT gated on passiveCompliance.
        if f == .compliance {
            translate = .zero; yaw = 0
            controller.stopBase()
            controller.setLiftRate(0)
        }
        // (Soft takes the drive pad down too — `drivePadWanted` keys on `feel`,
        // and the dock's onDisappear stops the base + lift on the way out.)
        withAnimation(Theme.Motion.snappy) { feel = f }
        applySpaceSpec()
        Haptics.select()
    }

    /// (The old 42 pt icon rail that floated over the top-left of the viewer is
    /// gone — it is `regionPanel` on the trailing ornament now, in the shared
    /// BODY MODULE panel language.)

    /// EXECUTE / CANCEL for a staged green-ball arm plan (ported from iOS). Shown
    /// while a plan is staged and the arm is animating the preview loop; nothing has
    /// moved the robot yet.
    @ViewBuilder private var planButtons: some View {
        if let plan = armPlan {
            HStack(spacing: 8) {
                Button { cancelPlanToken += 1; Haptics.tap() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .heavy)).foregroundStyle(.white)
                        .frame(width: 34, height: 34)
                        .background(Theme.Palette.danger.gradient, in: Circle())
                        .overlay { Circle().strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                Button { executePlanToken += 1; Haptics.rigid() } label: {
                    HStack(spacing: 6) {
                        Image(systemName: plan.collision ? "exclamationmark.triangle.fill" : "play.fill")
                            .font(.system(size: 13, weight: .bold))
                        Text("Execute").font(.system(size: 13, weight: .heavy)).tracking(0.4)
                        Text(plan.collision ? plan.reason
                             : (plan.side == "both" ? "both arms"
                                : plan.side == "right" ? "R arm" : "L arm"))
                            .font(.system(size: 9.5, weight: .semibold)).opacity(0.85)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14).frame(minHeight: 34)
                    .background((plan.collision ? Theme.Palette.warn : Theme.Palette.accent).gradient,
                                in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
            }
            .transition(.scale.combined(with: .opacity))
        }
    }

    /// Push the control spec matching the current space + feel + REGION. The
    /// region (arm / upper_body / whole_body) scopes what the controller owns —
    /// for the immersive retargeter (arm = wrist tracking only, upper = +torso,
    /// whole = +base/global position) AND for the green-ball / Soft drag.
    private func applySpaceSpec() {
        controller.teleopRegion = region      // the immersive retargeter reads this
        // Soft (compliance) is a float/drag mode over the selected region.
        if feel == .compliance {
            controller.setControl(method: .joint, type: .compliance, region: region)
            return
        }
        switch space {
        case .drive: controller.setNavigation()
        case .joint:
            controller.setControl(method: .joint, type: feel, region: .arm)
        case .task:
            // Whole Body → the immersive retarget law owns the arms, scoped by
            // the region (applyTeleopSpec sends retarget·<region>). Wrist → the
            // green-ball plan/execute drives joint·<region>. Gate on the METHOD,
            // not just on immersiveActive: the full-immersive dome can be open
            // (as a viewer) while the operator plans with the balls, and that
            // must not hand the arms to the retargeter.
            if taskMode == .wholeBody, controller.immersiveActive {
                controller.applyTeleopSpec()
            } else {
                controller.setControl(method: .joint, type: feel, region: region)
            }
        }
    }

    @ViewBuilder private func teleopControls(height: CGFloat) -> some View {
        let gripH = min(122, max(90, height * 0.34))
        Group {
            if pov == .first {
                // ── FPV COCKPIT ────────────────────────────────────────────
                // The DRIVE PAD is available in EVERY space here, not only in
                // Drive (operator, 2026-07-31: "add the two drive joysticks so
                // I can control the drive from 1st person as well") — on the
                // BOTTOM ORNAMENT, not in this window (`drivePadOrnament`).
                // The floating `Joystick3D` pair it replaced is gone
                // (operator, 2026-10-16: "keep things 2D and easy to toggle").
                //
                // Nothing downstream needed changing: `drive()` is gated only on
                // e-stop and Soft, never on `space`, and the engine's
                // `_apply_stick` already lets a deflected nav stick override the
                // head-follow while retargeting runs — driving and whole-body
                // mirroring are designed to coexist.
                //
                // The GRIPS stay IN the window: the green ball is a 3rd-person
                // control, so in the cockpit they are the only hand control
                // there is.
                HStack(alignment: .bottom, spacing: 8) {
                    if space == .task, taskMode == .wrist {
                        gripTrigger(side: "left", h: gripH)
                        Spacer(minLength: 0)
                        gripTrigger(side: "right", h: gripH)
                    }
                }
            } else {
            switch space {
            case .drive:
                // NOTHING IN THE WINDOW. Drive's TRANSLATE · LIFT · ROTATE pad
                // hangs on the bottom ornament (`drivePadOrnament`): in-window
                // chrome is occluded by the protruding 3-D stage.
                EmptyView()
            case .task where taskMode == .wrist:
                // WRIST method (iOS v2 parity): the green ball plans the wrist
                // POSE, so the only stick left is the GRIP — one per hand,
                // flanking the view like the iOS cockpit.
                HStack(alignment: .bottom, spacing: 6) {
                    gripTrigger(side: "left", h: gripH)
                    Spacer(minLength: 0)
                    gripTrigger(side: "right", h: gripH)
                }
            case .task:
                // WHOLE-BODY teleop: the mixed tracking space runs behind the
                // window, so head + arms + fingers retarget to the robot while
                // the operator watches the 3rd-person stage. No wrist balls, and
                // NO status card — its state (live / reach) is in the status bar
                // and its two controls are the top bar's third row now. A
                // floating card in the corner of a 3D view is one more thing
                // between the operator and the robot (operator, 2026-07-28).
                EmptyView()
            case .joint:
                // Joint-space controls live on the TRAILING ORNAMENT beside the
                // module picker — NOT in the window. The module fly-to zooms the
                // true-3D rig out through the window plane, which physically
                // occluded any in-window panel ~1 s after entering Joint space
                // (exactly the flight's duration: "sliders show then vanish").
                // Ornaments never get occluded.
                EmptyView()
            }
            }
        }
        // SYMMETRIC margins in landscape for the sticks. EVERY cluster clears
        // the two-line POV + pills stack, and the 1st-person right cluster
        // shifts clear of the bottom-right mini rig (230x205 at +10/+16) —
        // still in EVERY space: the inset costs nothing when the cockpit's row
        // is empty (the drive sticks are in the room now), and the Task·Wrist
        // grip would otherwise land on the mini rig.
        .padding(.horizontal, isLandscape ? 64 : 10)
        .padding(.trailing, pov == .first ? 220 : 0)
        .padding(.bottom, isLandscape ? 126 : bottomBarInset + 52)
    }

    /// Joint-space controls for the focused module — labelled sliders or 2D
    /// pads on the TRAILING ORNAMENT beside the module picker (iOS v2's
    /// right-hand joint column). In-window placement is impossible here: the
    /// module fly-to zooms the true-3D stage out through the window plane and
    /// occludes window chrome (ornaments never occlude).
    @ViewBuilder private var jointControlsPanel: some View {
        switch module {
        case .armLeft:   jointSliderPanel(specs(prefix: "left", in: .arms), title: "LEFT ARM")
        case .armRight:  jointSliderPanel(specs(prefix: "right", in: .arms), title: "RIGHT ARM")
        case .handLeft:  jointSliderPanel(specs(prefix: "left", in: .hands), title: RobotRig.handGen("left") == "gripper" ? "LEFT GRIPPER" : "LEFT HAND")
        case .handRight: jointSliderPanel(specs(prefix: "right", in: .hands), title: RobotRig.handGen("right") == "gripper" ? "RIGHT GRIPPER" : "RIGHT HAND")
        case .neck:
            // FLAT 2D pad (operator feedback: the 3D ball's push/pull pitch
            // axis was nearly unusable in a window) — iOS v2 mapping: stick =
            // yaw/pitch, roll on hold buttons.
            leftPanel("NECK") {
                HStack(alignment: .center, spacing: 16) {
                    Joystick(label: "← → YAW · ↕ PITCH",
                             tint: Theme.Palette.clay, size: 128) { v in
                        padC = SIMD3(Float(v.dx), 0, Float(v.dy))
                    }
                    VStack(spacing: 8) {
                        rollHold("arrow.counterclockwise", dir: 1)
                        Text("ROLL")
                            .font(.system(size: 8, weight: .heavy)).tracking(1)
                            .foregroundStyle(.white.opacity(0.6))
                        rollHold("arrow.clockwise", dir: -1)
                    }
                }
            }
        case .lift:
            // ONE slider for the whole column (operator, 2026-07-28). The lift is
            // a SYNC pair — `lift_lower` and `lift_upper` are tied by a MuJoCo
            // equality and every command path (`setLift`, `setLiftHeight`, the
            // drive stick, Home) drives them together — so exposing two sliders
            // offered a split the robot cannot take and made "raise the lift"
            // two gestures. Direct setpoint in metres of TOTAL travel; the rate
            // joystick is still gone (it rode the nav channel, which joint mode
            // does not service).
            leftPanel("LIFT") {
                VStack(spacing: 2) {
                    JointSlider(spec: Self.liftColumnSpec,
                                value: Binding(get: { controller.liftHeight },
                                               set: { controller.setLiftHeight($0) }))
                }
                .frame(width: 268)
            }
        case .torso:
            jointSliderPanel(RobotModel.specs(in: .body).filter { $0.id == "torso" },
                             title: "TORSO")
        }
    }

    /// The lift as the ONE axis it physically is: total column travel in metres
    /// (both sync'd stages summed). Not in `RobotModel.joints`, which is the
    /// wire vocabulary — the robot really does have two lift joints and the
    /// telemetry/bar/trend views must keep showing both.
    static let liftColumnSpec = JointSpec(id: "lift", label: "Lift Height",
                                          group: .body, kind: .prismatic,
                                          lower: 0, upper: RobotController.liftMax)

    /// The focused side's joints from the shared vocabulary (limits included).
    private func specs(prefix: String, in group: JointGroup) -> [JointSpec] {
        RobotModel.specs(in: group).filter { $0.id.hasPrefix(prefix) && !RobotModel.isRedundantGrip($0.id) }
    }

    /// Shared LEFT glass-panel chrome for the joint-space controls.
    private func leftPanel<Content: View>(_ title: String,
                                          @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.system(size: 10, weight: .heavy)).tracking(1.3)
                .foregroundStyle(Theme.Palette.textTertiary)
            content()
        }
        .padding(14)
        // Real visionOS glass, matching the BODY MODULE panel it sits beside
        // (2026-07-16). The old OPAQUE 0x14171B slab dated from when this panel
        // lived IN THE WINDOW, where a sheer `.ultraThinMaterial` let the bright
        // 3D stage bleed through and the sliders became unreadable. It lives on
        // the trailing ORNAMENT now — outside the window, over passthrough, where
        // system glass is legible — so the slab only made the two neighbouring
        // panels look like they came from different apps.
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 20, style: .continuous)
            .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
    }

    /// Single-column labelled sliders (the app-wide JointSlider rows) — direct
    /// setpoints; scrolls for the 17-DOF hands.
    private func jointSliderPanel(_ list: [JointSpec], title: String) -> some View {
        leftPanel(title) {
            ScrollView(showsIndicators: false) {
                VStack(spacing: 2) {
                    ForEach(list) { spec in
                        JointSlider(spec: spec,
                                    value: Binding(get: { controller.jointTargets[spec.id] ?? 0 },
                                                   set: { controller.setJoint(spec.id, $0) }))
                    }
                }
            }
            .frame(width: 268)
            .frame(maxHeight: 400)
        }
    }

    /// Press-and-hold neck-roll nudge (30 Hz integration while held).
    private func rollHold(_ icon: String, dir: Double) -> some View {
        Image(systemName: icon)
            .font(.system(size: 15, weight: .bold)).foregroundStyle(.white)
            .frame(width: 46, height: 38)
            .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(.white.opacity(0.16), lineWidth: 1) }
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in if rollDir != dir { rollDir = dir; Haptics.soft() } }
                .onEnded { _ in rollDir = 0 })
    }

    /// Press-and-hold lift rate button (release → rate 0).
    private func liftHold(_ icon: String, rate: Double) -> some View {
        Image(systemName: icon)
            .font(.system(size: 18, weight: .bold)).foregroundStyle(.white)
            .frame(width: 64, height: 46)
            .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.16), lineWidth: 1) }
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in controller.setLiftRate(rate) }
                .onEnded { _ in controller.setLiftRate(0) })
    }

    // (No in-window drive stick: driving is the 2-D `DrivePadDock` on the
    //  bottom ornament — see `drivePadOrnament`. Joint space uses the flat
    //  `Joystick` neck pad and `JointSlider`, which are different controls.)

    private func armGroup(side: String, mirrored: Bool, jSize: CGFloat, gripH: CGFloat) -> some View {
        // Task space is driven by DRAGGING THE GREEN BALL (plan → Execute) — far more
        // intuitive than the wrist pad — so the 3-axis pad is gone. Only the GRIP
        // slider stays (the ball plans the wrist pose; the grip opens/closes the hand).
        gripTrigger(side: side, h: gripH)
    }

    private func holdButton(_ icon: String, side: String, dir: Double) -> some View {
        Image(systemName: icon)
            .font(.system(size: 14, weight: .bold)).foregroundStyle(.white)
            .frame(width: isLandscape ? 46 : 40, height: 32)
            .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(.white.opacity(0.15), lineWidth: 1) }
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    if side == "left" { if pitchL != dir { pitchL = dir; Haptics.soft() } }
                    else { if pitchR != dir { pitchR = dir; Haptics.soft() } }
                }
                .onEnded { _ in if side == "left" { pitchL = 0 } else { pitchR = 0 } })
    }

    private func gripTrigger(side: String, h: CGFloat) -> some View {
        let g = side == "left" ? gripL : gripR
        return VStack(spacing: 4) {
            Text("\(Int(g * 100))%").font(.mabelMonoSmall).foregroundStyle(.white)
            ZStack(alignment: .bottom) {
                Capsule().fill(.black.opacity(0.4)).frame(width: 42, height: h)
                    .overlay { Capsule().strokeBorder(.white.opacity(0.15), lineWidth: 1) }
                Capsule().fill(Theme.accentGradient).frame(width: 42, height: max(14, h * CGFloat(g)))
                Image(systemName: "hand.raised.fill").font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white).padding(.bottom, 6)
            }
            .frame(width: 42, height: h)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                let val = Double(1 - min(1, max(0, v.location.y / h)))
                if side == "left" { gripL = val } else { gripR = val }
                controller.setGrip(side, val)
            })
            Text(side == "left" ? "L GRIP" : "R GRIP")
                .font(.system(size: 7.5, weight: .bold)).tracking(1).foregroundStyle(.white.opacity(0.7))
        }
    }

    /// 30 Hz joint-nudge integration — one BATCHED command per tick. Joint space
    /// nudges the focused module; Task space runs the classic dual wrist sticks.
    private func onTick() {
        // Joint controls are live whenever Joint space is selected (no Start
        // gate); nudges only exist here — the other spaces are direct-drive.
        guard space == .joint else { return }
        let dt = 1.0 / 30.0
        var batch: [String: Double] = [:]
        switch module {
        case .neck:
            // iOS v2 signs: stick right → yaw right, stick up → pitch up,
            // roll on the hold buttons (2D pad stores dx→padC.x, dy→padC.z).
            nudge(&batch, "neck_yaw", -Double(padC.x) * armRate * dt)
            nudge(&batch, "neck_pitch", -Double(padC.z) * armRate * dt)
            nudge(&batch, "neck_roll", rollDir * armRate * dt)
        case .armLeft, .armRight, .handLeft, .handRight, .lift, .torso:
            break   // sliders send direct setpoints
        }
        if !batch.isEmpty { controller.setJoints(batch) }
    }

    private func nudge(_ batch: inout [String: Double], _ id: String, _ delta: Double) {
        guard abs(delta) > 1e-6 else { return }
        let cur = batch[id] ?? controller.jointTargets[id] ?? 0
        let spec = RobotModel.spec(id)
        let v = cur + delta
        batch[id] = spec.map { min($0.upper, max($0.lower, v)) } ?? v
    }

    // MARK: - POV

    /// POV switch, MIRRORING iOS v2: ALL visible camera motion is the single
    /// SceneKit flight (2 s, `LiveMapView.povFlightDuration`) — the renderer
    /// swap happens only at the WIDE end, between two static whole-robot
    /// framings, where a 0.35 s crossfade is invisible. Never crossfade two
    /// moving cameras (the "very bad" transition).
    private func setPov(_ mode: PovMode) {
        guard mode != pov, !povAnimating else { return }
        Haptics.select()
        // A STICK THAT DISAPPEARS MUST NOT LEAVE THE BASE RUNNING. The cockpit
        // drive sticks now exist in every space in 1st person, so leaving 1st
        // person can remove a DEFLECTED stick from the screen — and the base
        // holds its last twist until something overwrites it (the engine only
        // zeroes on release, and a released stick that no longer exists never
        // sends one). Same rule `setSpace` applies when leaving Drive.
        // (They are the SPACE's floating sticks now, not window pads — leaving
        // 1st person tears the stick host down, which removes them just as
        // abruptly, so the rule is unchanged.)
        if pov == .first {
            translate = .zero; yaw = 0
            controller.stopBase()
            controller.setLiftRate(0)
        }
        // (`pov` itself is assigned deep inside the POV flight's timers below,
        // so the stick host follows it from `.onChange(of: pov)` — the sticks
        // then appear exactly when the cockpit does, not 1.6 s before it.)
        // (A take survives a POV / space switch: Record — and its STOP — is in
        // the top bar in every view now, so nothing records invisibly.)
        // Physics view is 3rd-person-only: drop the takeover instantly so the
        // POV flight runs on the normal stage (iOS parity).
        if controller.physicsViewOn {
            physicsFaded = false
            controller.physicsViewOn = false
            controller.physicsViewCam.active = false
        }
        povAnimating = true
        if mode == .first {
            // THE VOLUMETRIC DIVE (operator's spec): the TRUE-3D stage flies
            // to the head-cam's projected-image plane until the image fills
            // the view (real depth the whole way); only THEN does the flat
            // handoff happen, between two copies of the SAME picture.
            following = false
            fpvVisible = false
            stagePresetInstant = false
            stagePreset = .headImage; presetToken += 1      // 1.6 s dive
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                sceneInstantPov = true
                scenePov = .first    // park the hidden flat view NOW — long settled
            }                        // (and inset) by the time the crossfade runs
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.7) {
                withAnimation(.easeInOut(duration: 0.3)) { pov = .first }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    fpvVisible = true              // 2D image over the same image
                    sceneInstantPov = false
                    povAnimating = false
                }
            }
        } else {
            // REVERSE: the image fades out to the 3D projected view (the
            // hidden stage is PARKED at the head-image pose, instantly), the
            // renderers swap between identical pictures, then the stage pulls
            // back — with real depth — to the ORIGINAL camera framing.
            fpvVisible = false
            stagePresetInstant = true
            stagePreset = .headImage; presetToken += 1      // instant hidden park
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                withAnimation(.easeInOut(duration: 0.3)) { pov = .third }
                stagePresetInstant = false
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    stagePreset = .returnFromHead; presetToken += 1   // 1.6 s pull-out
                    scenePov = .third            // hidden flat view glides home
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.7) {
                        if teleopActive { applyStagePreset(for: space) }
                        povAnimating = false
                    }
                }
            }
        }
    }

    // MARK: - Overlays

    /// 1st-person cockpit chrome — the two wrist cams pinned to the LEFT and
    /// RIGHT edges while the live head video fills the screen (v1 VP parity).
    /// The TILES are tappable (tap → placeable camera-detail window); the rest
    /// of the stack is Spacers, so FPV interactions still pass through.
    private var fpvChrome: some View {
        // Wrist cams ride the TOP corners of the FPV viewport — left wrist top-
        // left, right wrist top-right — below the top bar, clear of every
        // control (mini rig owns bottom-right).
        VStack {
            HStack(alignment: .top, spacing: 0) {
                CameraFeed(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                           streamURL: controller.videoThumbURL("wrist_left"),
                           stream: "wrist_left",
                           onExpand: { openCameraDetail("wrist_left") })
                    .frame(width: 180, height: 120)
                Spacer(minLength: 0)
                CameraFeed(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                           streamURL: controller.videoThumbURL("wrist_right"),
                           stream: "wrist_right",
                           onExpand: { openCameraDetail("wrist_right") })
                    .frame(width: 180, height: 120)
            }
            Spacer(minLength: 0)
        }
        .padding(.top, Self.povInsetTop + 10)
        .padding(.horizontal, Self.povInsetSide + 10)
        .transition(.opacity)
    }

    /// The small live articulated 3D robot in the BOTTOM-RIGHT corner during
    /// 1st person — interactive (gaze-highlight, pinch-drag to guide), same
    /// spot as the original Vision Pro pilot column.
    private var miniRigCorner: some View {
        VStack {
            Spacer(minLength: 0)
            HStack {
                Spacer(minLength: 0)
                RobotStageView(stage: miniStage, interaction: .links)
                    .frame(width: 230, height: 205)
            }
        }
        .padding(.trailing, 10)
        .padding(.bottom, 16)
        .transition(.opacity)
    }

    /// Floating glass prompt — driving works everywhere; landscape is roomier.
    private var rotateHint: some View {
        HintPill(systemImage: "rotate.right.fill", title: "Landscape Recommended",
                 subtitle: "Driving works here too — rotate for the full cockpit",
                 iconTint: Theme.Palette.accent, pulse: true)
    }

    /// Task-space coaching (shown while task manipulation is armed). The wrist
    /// method's hint says the ball PLANS — the operator must not be surprised
    /// that the robot didn't move.
    private var taskHint: some View {
        if feel == .compliance {
            return HintPill(systemImage: "wind", title: "Soft — move it by hand",
                            subtitle: "Floating on gravity comp. The app sends no commands.",
                            iconTint: Theme.Palette.live)
        }
        if taskMode == .wrist {
            return HintPill(systemImage: "hand.draw.fill", title: "Drag the green ball",
                            subtitle: "Staging a plan — the robot moves on Execute, not before",
                            iconTint: Theme.Palette.live)
        }
        return HintPill(systemImage: "figure.arms.open", title: "Whole-body teleop",
                        subtitle: "Start Teleop — your head, arms and fingers drive the robot",
                        iconTint: Theme.Palette.live)
    }

    /// Honest, concise notice while the LOCAL demo map is loaded (opt-in).
    private var demoBanner: some View {
        HintPill(systemImage: "cpu",
                 title: "Local demo map",
                 subtitle: "Tap the floor — goals drive the demo robot",
                 iconTint: Theme.Palette.clay)
    }

    /// Centred standby card while there's nothing (yet) to show.
    private var standby: some View {
        let fg = Color(hex: 0xF4F1EA)
        return VStack(spacing: 8) {
            Image(systemName: store.hasEndpoint ? "point.3.connected.trianglepath.dotted" : "globe.badge.chevron.backward")
                .font(.system(size: 30, weight: .light)).foregroundStyle(fg.opacity(0.7))
            Text(store.hasEndpoint ? "Waiting for the robot's map…" : "No map data streamed")
                .font(.system(size: 13, weight: .semibold)).foregroundStyle(fg)
            Text(store.hasEndpoint
                 ? "Run visual SLAM on the sim or robot — the mesh and odom appear here live."
                 : "Connect to a robot (Settings), or load a local demo map.")
                .font(.system(size: 11, weight: .medium)).foregroundStyle(fg.opacity(0.7))
                .multilineTextAlignment(.center).frame(maxWidth: 270)
            Button {
                Haptics.select()
                controller.requestMapList()
                showSavedMaps = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "square.stack.3d.up.fill").font(.system(size: 11, weight: .semibold))
                    Text("Choose Map").font(.system(size: 12, weight: .bold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(Theme.accentGradient, in: Capsule(style: .continuous))
            }
            .buttonStyle(PressableStyle())
            .padding(.top, 4)
        }
        .padding(20)
        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(.white.opacity(0.1), lineWidth: 1) }
        .frame(maxHeight: .infinity, alignment: .center)
        .padding(.bottom, 40)
    }

    // The map viewport is always dark, so the glass pills use fixed light ink.
    // `title: nil` → compact icon-only pill (portrait, where width is scarce).
    private func pill(_ title: String?, _ icon: String, active: Bool, chevron: Bool = false) -> some View {
        let content = HStack(spacing: 6) {
            Image(systemName: icon).font(.system(size: 12, weight: .semibold))
            if let title { Text(title).font(.system(size: 13, weight: .semibold)).lineLimit(1) }
            if chevron { Image(systemName: "chevron.down").font(.system(size: 9, weight: .bold)).opacity(0.85) }
        }
        .foregroundStyle(active ? .white : Color(hex: 0xF4F1EA))
        .padding(.horizontal, title == nil ? 11 : 14)
        .frame(height: 36)   // top-bar control standard

        return Group {
            if active {
                content.background(Theme.accentGradient, in: Capsule(style: .continuous))
            } else {
                content.glassBackgroundEffect(in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.16), lineWidth: 1) }
            }
        }
        .shadow(color: .black.opacity(0.28), radius: 10, y: 4)
    }
}
