//
//  RobotStageView.swift
//  MabelTeleop
//
//  The live articulated MABEL as an interactive 3D stage. Used in-window
//  (Teleop mini model, Body tab) and inside the volumetric model window.
//
//  Sync: the robot poses itself from the shared controller state on EVERY
//  RealityKit frame (SceneEvents.Update subscription), so the model always
//  mirrors the sim/real robot regardless of SwiftUI re-render cadence.
//
//  Two interaction modes:
//    .links — look at an arm link (gaze highlight) and pinch-drag to guide/
//             push it; drag empty space rotates, pinch zooms. (Body tab.)
//    .move  — the whole robot is grabbable: pinch-drag MOVES it, a two-hand
//             twist ROTATES it about its vertical axis, pinch zooms.
//             (Volumetric window.)
//

import SwiftUI
import Combine
import Spatial
import QuartzCore     // CACurrentMediaTime — the planner tick's clock
import RealityKit
import simd

/// Critically-damped display smoothing for the live robot model.
///
/// Telemetry is a 20 Hz robot_state stream that BUNCHES on jittery links
/// (Tailscale WAN: several messages arrive back-to-back, then a gap), so
/// posing the rig directly from each message makes the model judder/shake.
/// Every render frame (90 Hz) glides the displayed values toward the newest
/// telemetry instead: α = 0.25/frame ≈ 30 ms settling — imperceptible lag,
/// no snapping. A jump > 0.5 rad is a real discontinuity (sim reset), so it
/// snaps rather than animating through it.
@MainActor
final class PoseSmoother {
    private var joints: [String: Double] = [:]
    private var heading: Double? = nil

    func step(toward target: [String: Double]) -> [String: Double] {
        for (k, v) in target {
            let s = joints[k] ?? v
            // FIXED α = 0.6 @ ~90 fps (τ ≈ 17 ms): real-time feel with just
            // enough filtering to hide telemetry jitter. Error-adaptive α
            // trailed at slow speeds (the gain sank with the error — the
            // model crept behind the robot).
            joints[k] = abs(v - s) > 0.5 ? v : s + 0.6 * (v - s)
        }
        return joints
    }

    /// Wrap-aware (yaw crosses ±π).
    func stepHeading(toward target: Double) -> Double {
        guard let s = heading else { heading = target; return target }
        var d = (target - s).truncatingRemainder(dividingBy: 2 * .pi)
        if d > .pi { d -= 2 * .pi } else if d < -.pi { d += 2 * .pi }
        let out = abs(d) > 0.5 ? target : s + 0.6 * d
        heading = out
        return out
    }
}

/// Owns the rig + view-local camera state across SwiftUI updates.
@MainActor
final class RobotStage: ObservableObject {
    enum Interaction { case none, links, move }

    let rig = RobotRigEntity()
    /// Pivot the gestures act on (yaw + scale + position).
    var yaw: Float = 0.6
    var baseScale: Float
    let interactive: Bool
    @Published var loaded = false
    var updateSub: EventSubscription?
    /// Live planner reference for the per-frame tick. The SceneEvents.Update
    /// closure captures the RobotStageView STRUCT by value at subscribe time —
    /// RobotSpace3DStage wires the planner in .onAppear, AFTER that first body,
    /// so the captured copy's `manip` is nil FOREVER (staging still worked —
    /// gestures rebuild per body — but the ghost preview never looped and
    /// Execute armed without ever streaming: "click Execute, nothing moves").
    /// Routing the tick through this CLASS property keeps it always-current.
    weak var manip: StagePlanner?

    private var dragYaw0: Float?
    private var dragPitch0: Float?
    private var zoom0: Float?

    // MARK: - Framing (what orbit + zoom compose ON TOP OF)
    //
    // The stage has no camera, so a "camera flight" is a transform on the rig:
    // an absolute rotation plus a translation that puts one rig-local ANCHOR
    // point on screen (RobotSpace3DStage.frame). The gestures used to ignore all
    // of that — `orbit` overwrote the orientation with a level yaw about the rig
    // ORIGIN, so the first drag inside a Joint-space module framing threw away
    // its elevation AND its centring: "the 3D model snaps back to the horizontal
    // frame" (operator, 2026-07-28). Zoom had the twin defect, scaling about the
    // rig origin so a pinch flung the framed module out of the window.
    //
    // So the framing is state. Orbit composes stage-axis yaw/pitch onto
    // `orbitBase` and both gestures re-solve the translation to keep
    // `orbitAnchorLocal` pinned at `orbitAnchorScene`, i.e. they orbit and zoom
    // AROUND THE SUBJECT. With no framing adopted (`orbitAnchorLocal == nil`,
    // identity base, zero pitch) the maths reduces EXACTLY to the old
    // `orientation = quat(yaw, Y)` behaviour, which is what the volumetric
    // window and the Body tab still get.

    /// Rotation the orbit deltas compose on top of.
    private(set) var orbitBase = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    /// Rig-local point kept pinned on screen while orbiting/zooming (nil = none).
    private(set) var orbitAnchorLocal: SIMD3<Float>? = nil
    private(set) var orbitAnchorScene = SIMD3<Float>(repeating: 0)
    /// Elevation of the orbit, radians. Positive = the viewer is ABOVE, looking
    /// down at the ground plane.
    private(set) var orbitPitch: Float = 0
    /// Elevation limits. The floor is a real plane in this scene, so the camera
    /// is kept just above it — dropping under the ground shows the underside of
    /// the floor disc and an upside-down world, and there is no visual cue for
    /// how to get back. Up is capped short of the pole, where yaw and roll
    /// degenerate into each other.
    static let orbitPitchMin: Float = -0.12       // ~7° below the horizon
    static let orbitPitchMax: Float = 1.30        // ~74°, short of gimbal flip
    /// Kept for call sites that only need the symmetric magnitude.
    static let orbitPitchLimit: Float = 1.30

    /// Bumped by every adopted framing — a deferred measurement of "where did
    /// that flight land" can tell whether ANOTHER flight has since superseded it.
    private(set) var framingSeq = 0

    /// Adopt `t` as the current framing: `anchorLocal` is the rig-local point
    /// `t` placed on screen, and orbit/zoom will pivot on it from here.
    ///
    /// THE HORIZON STAYS LEVEL. `levelled` (the default) DECOMPOSES the framing's
    /// rotation into an azimuth + an elevation about the rig's own up axis and
    /// rebuilds it as `Rx(elev)·Ry(−azim)` — the same view direction, but
    /// expressed as a TURNTABLE. Everything after it is then `Rx(pitch)·Ry(yaw)`,
    /// which maps the rig's up into the screen's YZ plane for ANY pitch and yaw,
    /// so the ground can never tip: "I want the camera view to lock the floor —
    /// right now the entire view rotates" (operator, 2026-07-28). Composing a
    /// stage-axis pitch on top of an arbitrary base rotation does not have that
    /// property, and a module framing's base is arbitrary.
    ///
    /// `levelled: false` is for the head-image dive, whose whole job is to face
    /// a tilted image plane squarely — forcing it level would break the fade.
    func adoptFraming(_ t: Transform, anchorLocal: SIMD3<Float>, levelled: Bool = true) {
        framingSeq += 1
        orbitAnchorLocal = anchorLocal
        orbitAnchorScene = t.translation + t.rotation.act(anchorLocal * t.scale.x)
        dragYaw0 = nil; dragPitch0 = nil
        guard levelled else {
            orbitBase = t.rotation
            yaw = 0; orbitPitch = 0
            return
        }
        // The rig-local direction that faces the viewer: R·dir = +Z ⇒ dir = R⁻¹·+Z.
        let dir = t.rotation.inverse.act(SIMD3<Float>(0, 0, 1))
        let e = asin(max(-1, min(1, dir.y)))
        let a = atan2(dir.x, dir.z)
        orbitBase = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        yaw = -a
        orbitPitch = min(Self.orbitPitchMax, max(Self.orbitPitchMin, e))
    }

    /// Re-measure where the anchor sits after something else moved the rig (a
    /// window resize re-fits it), so the next drag doesn't jump.
    func reseatFraming() {
        guard let a = orbitAnchorLocal else { return }
        let t = rig.root.transform
        orbitAnchorScene = t.translation + t.rotation.act(a * t.scale.x)
    }

    /// Forget the framing — gestures fall back to plain yaw about the origin.
    func clearFraming() {
        orbitBase = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        orbitAnchorLocal = nil
        orbitPitch = 0
        dragYaw0 = nil; dragPitch0 = nil
    }

    /// Compose the live orbit onto the framing and re-pin the anchor.
    private func applyOrbit() {
        var t = rig.root.transform
        t.rotation = simd_quatf(angle: orbitPitch, axis: SIMD3(1, 0, 0))
            * simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)) * orbitBase
        if let a = orbitAnchorLocal {
            t.translation = orbitAnchorScene - t.rotation.act(a * t.scale.x)
        }
        rig.root.transform = t
    }

    /// Display smoothing for the live model — see PoseSmoother below.
    let smoother = PoseSmoother()

    // Measured volume geometry (volumeFit) — buttons and clamps use the REAL
    // box, not assumed constants (visionOS restores persisted window sizes,
    // so a "2.2 m" volume may actually be much smaller).
    var volumeFloor: Float = -0.9
    var homeSeat = SIMD3<Float>(0, -0.9, 0.2)
    var measuredBounds: (min: SIMD3<Float>, max: SIMD3<Float>)? = nil
    var measuredExtents: SIMD3<Float>? = nil

    init(scale: Float, interactive: Bool = false) {
        self.baseScale = scale
        self.interactive = interactive
        rig.root.scale = SIMD3(repeating: scale)
        rig.root.orientation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
    }

    /// Current uniform scale of the stage (for converting scene → robot metres).
    var currentScale: Float { rig.root.scale.x }

    /// Scene METRES per view POINT, measured by `fitToView` from the hosting
    /// window (windowed RealityViews are ~1/1360 m per point, but that is a
    /// platform detail, not a promise). Framing maths that has to guarantee
    /// something about ON-SCREEN size solves through this instead of a magic
    /// constant. 0 until the first fit.
    private(set) var scenePerPoint: Float = 0
    func setScenePerPoint(_ v: Float) { if v > 0 { scenePerPoint = v } }

    func load() async {
        guard !loaded else { return }
        await rig.load(interactive: interactive)
        loaded = true
    }

    /// Whole-robot grab proxy: one collision box over the body so pinch-drag
    /// anywhere on the robot moves/rotates it. The box is TIGHT around the
    /// robot's real visual bounds — InputTarget colliders occlude gaze for
    /// UI behind them, so a loose box of empty air was blocking the volume's
    /// ornament buttons from being looked at.
    func installGrabProxy() {
        guard rig.robot.components[CollisionComponent.self] == nil else { return }
        let bounds = rig.robot.visualBounds(relativeTo: rig.robot)
        let extents = bounds.extents
        if extents.x.isFinite, extents.x > 0.05, extents.y > 0.05 {
            rig.robot.components.set(CollisionComponent(shapes: [
                .generateBox(size: extents * 1.04).offsetBy(translation: bounds.center)
            ]))
        } else {
            rig.robot.components.set(CollisionComponent(shapes: [
                .generateBox(width: 0.6, height: 1.7, depth: 0.6)
                    .offsetBy(translation: SIMD3(0, 0.85, 0))
            ]))
        }
        rig.robot.components.set(InputTargetComponent())
        rig.robot.components.set(HoverEffectComponent())
    }

    // MARK: - Gestures

    /// Drag to orbit: horizontal = yaw, vertical = elevation. Direct
    /// manipulation — dragging right swings the robot's front to the right,
    /// dragging down tips its top toward the viewer.
    func orbit(dx: Float, dy: Float = 0) {
        if dragYaw0 == nil { dragYaw0 = yaw; dragPitch0 = orbitPitch }
        yaw = (dragYaw0 ?? 0) + dx * 2.6
        orbitPitch = min(Self.orbitPitchMax,
                         max(Self.orbitPitchMin, (dragPitch0 ?? 0) + dy * 1.8))
        applyOrbit()
    }
    func orbitEnded() { dragYaw0 = nil; dragPitch0 = nil }

    /// Absolute yaw delta (two-hand twist in `.move` mode).
    func twist(_ delta: Float) {
        if dragYaw0 == nil { dragYaw0 = yaw; dragPitch0 = orbitPitch }
        yaw = (dragYaw0 ?? 0) + delta
        applyOrbit()
    }
    func twistEnded() { dragYaw0 = nil; dragPitch0 = nil }

    /// Pinch to zoom — about the framed SUBJECT when a framing is adopted, so a
    /// zoomed-in module stays where the operator is looking.
    func zoom(_ magnification: Float) {
        if zoom0 == nil { zoom0 = rig.root.scale.x }
        let s = max(0.05, min(2.5, (zoom0 ?? baseScale) * magnification))
        var t = rig.root.transform
        t.scale = SIMD3(repeating: s)
        if let a = orbitAnchorLocal {
            t.translation = orbitAnchorScene - t.rotation.act(a * s)
        }
        rig.root.transform = t
    }
    func zoomEnded() { zoom0 = nil }

    /// Reset the view: home orientation + the stage's natural scale.
    func center(position: SIMD3<Float>? = nil) {
        yaw = 0.6
        clearFraming()
        zoom0 = nil
        var t = rig.root.transform
        t.rotation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
        t.scale = SIMD3(repeating: baseScale)
        if let position { t.translation = position }
        rig.root.move(to: t, relativeTo: rig.root.parent, duration: 0.35, timingFunction: .easeInOut)
    }

    /// TRUE life size: scale 1:1 AND feet on the given floor height, in ONE
    /// transform animation. (Two separate move(to:) calls raced — the second
    /// captured the pre-animation scale and cancelled the first.)
    func lifeSize(floorY: Float) {
        zoom0 = nil; dragYaw0 = nil
        var t = rig.root.transform
        t.scale = .one
        t.translation = SIMD3(0, floorY, t.translation.z)
        rig.root.move(to: t, relativeTo: rig.root.parent, duration: 0.45, timingFunction: .easeInOut)
    }

    /// Life size INSIDE the measured volume: 1:1 when the box allows it,
    /// otherwise the largest scale that shows the WHOLE robot — never
    /// sliced by a wall. Centred laterally, feet on the real floor, body
    /// kept clear of the front and back walls.
    func lifeSizeMeasured() {
        guard let e = measuredExtents else {
            lifeSize(floorY: volumeFloor)
            return
        }
        zoom0 = nil; dragYaw0 = nil; yaw = 0
        // Robot envelope at scale 1 (home pose): 1.6 m tall, ~1.3 m span.
        // True 1:1 when the box allows; otherwise the largest whole-robot
        // scale. Face forward (axis-aligned) so the caps are valid.
        let s = min(1, (e.y * 0.96) / 1.6, (e.x * 0.92) / 1.3, (e.z * 0.92) / 1.3)
        var t = rig.root.transform
        t.rotation = simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))
        t.scale = SIMD3(repeating: s)
        t.translation = SIMD3(0, volumeFloor, 0)
        rig.root.move(to: t, relativeTo: rig.root.parent, duration: 0.45, timingFunction: .easeInOut)
    }

    func setScale(_ s: Float) {
        var t = rig.root.transform
        t.scale = SIMD3(repeating: s)
        rig.root.move(to: t, relativeTo: rig.root.parent, duration: 0.35, timingFunction: .easeInOut)
    }

    /// Move the robot's feet to a given height (keeps scale + yaw).
    func placeFeet(atY y: Float) {
        // The rig origin is NOT the feet (base_link sits above the wheel
        // contact), and after Life Size / a carry the root's y is usually
        // already at the floor value — so snapping the ORIGIN to y read as a
        // dead button. Measure the rig's CURRENT visual bounds instead and
        // translate so its lowest point lands exactly on the floor; works at
        // any scale and any carried pose.
        let bounds = rig.root.visualBounds(relativeTo: rig.root.parent)
        var t = rig.root.transform
        t.translation.y += y - bounds.min.y
        rig.root.move(to: t, relativeTo: rig.root.parent, duration: 0.35, timingFunction: .easeInOut)
    }
}

struct RobotStageView: View {
    @EnvironmentObject private var controller: RobotController
    @ObservedObject var stage: RobotStage
    /// Authoritative points→metres conversion for the hosting scene — the
    /// documented way to learn a volumetric window's REAL dimensions.
    @Environment(\.physicalMetrics) private var physicalMetrics
    /// Fit the robot to the view automatically (window contexts). Windowed
    /// RealityViews map ~1360 points per metre, so metre-guessed scales
    /// overflow and get clipped — fitting converts the REAL view size into
    /// scene metres and sizes the robot to it. Volumes (true metres) pass
    /// false and use floorY/depth/scale directly.
    var fit: Bool = true
    /// Vertical offset of the robot's feet from the stage origin (metres).
    /// Used only when `fit == false`.
    var floorY: Float = -0.35
    /// Depth offset (metres). Used only when `fit == false`.
    var depth: Float = 0
    /// Interaction mode (see header).
    var interaction: RobotStage.Interaction = .links
    /// Scene-space bounds for `.move` drags (volume interior, metres) — the
    /// robot can never be carried into a wall and half-clipped. When
    /// `volumeFit` is on, the MEASURED bounds take precedence.
    var moveBounds: (min: SIMD3<Float>, max: SIMD3<Float>)? = nil
    /// Measure the hosting volume and fit the robot inside it (scale, floor,
    /// seat, carry clamps all derived from the real box).
    var volumeFit: Bool = false
    /// Control bar pinned to the volume's front-bottom (nearest the user).
    var volumeControls: (() -> AnyView)? = nil
    /// Link drag callbacks (`.links` mode): (link, dragged entity, scene-
    /// space delta in metres since drag start, scene-space contact point).
    var onLinkDrag: (RigLink, Entity, SIMD3<Float>, SIMD3<Float>) -> Void = { _, _, _, _ in }
    var onLinkDragEnd: () -> Void = {}
    /// Task-space green-ball planner (Robot space only). When a drag lands on
    /// one of its handles/rings the planner owns the drag — link-guide and
    /// orbit stand aside — and it ticks every render frame after pose().
    var manip: StagePlanner? = nil
    /// Where the window fit seats the feet, as a fraction of the window height above
    /// its bottom edge. The Robot space raises it because its DRIVE PAD hangs on the
    /// bottom ornament in every space now (2026-10-27): the rig sits IN FRONT of the
    /// window plane, so at the old 6 % its base projected over the pad's header.
    var floorRaise: Float = 0.06

    @State private var linkDragging = false
    @State private var manipDragging = false
    @State private var orbitDragging = false
    @State private var dragStartScene: SIMD3<Float>? = nil
    @State private var moveStart: SIMD3<Float>? = nil
    @State private var fittedSize: CGSize = .zero

    /// Rig height at scale 1 (home pose, metres) for fit computations.
    private static let robotHeight: Float = 1.6

    var body: some View {
        if volumeFit {
            GeometryReader3D { geo in
                volumeRealityBody(size: geo.size)
            }
        } else {
            GeometryReader { geo in
                windowRealityBody(size: geo.size)
            }
        }
    }

    @ViewBuilder private func windowRealityBody(size: CGSize) -> some View {
        RealityView { content in
            if !fit { stage.rig.root.position = SIMD3(0, floorY, depth) }
            content.add(stage.rig.root)
            // Frame-locked sync: pose from the live shared state EVERY frame.
            stage.updateSub = content.subscribe(to: SceneEvents.Update.self) { _ in
                // SYNCHRONOUS apply (the iOS display-link pattern): a Task per
                // frame backlogs the main actor — the model falls behind and
                // eventually reads as frozen.
                if Thread.isMainThread { MainActor.assumeIsolated { pose() } }
                else { Task { @MainActor in pose() } }
            }
            await stage.load()
            if interaction == .move { stage.installGrabProxy() }
            if fit { fitToView(content, size: size) }
            pose()
        } update: { content in
            if fit, size != fittedSize {
                fitToView(content, size: size)
            }
        }
        .gesture(linkDragGesture)
        .gesture(moveDragGesture)
        .simultaneousGesture(orbitGesture)
        .simultaneousGesture(zoomGesture)
        .simultaneousGesture(twistGesture)
        .overlay {
            if !stage.loaded {
                ProgressView().controlSize(.large)
            }
        }
    }

    @ViewBuilder private func volumeRealityBody(size: Size3D) -> some View {
        RealityView { content, attachments in
            content.add(stage.rig.root)
            stage.updateSub = content.subscribe(to: SceneEvents.Update.self) { _ in
                if Thread.isMainThread { MainActor.assumeIsolated { pose() } }
                else { Task { @MainActor in pose() } }
            }
            await stage.load()
            if interaction == .move { stage.installGrabProxy() }
            fitToVolume(size: size)
            if let controls = attachments.entity(for: "volumeControls") {
                content.add(controls)
                placeControls(controls)
            }
            pose()
        } update: { content, attachments in
            fitToVolume(size: size)
            if let controls = attachments.entity(for: "volumeControls") {
                placeControls(controls)
            }
        } attachments: {
            if let make = volumeControls {
                Attachment(id: "volumeControls") { make() }
            }
        }
        .gesture(linkDragGesture)
        .gesture(moveDragGesture)
        .simultaneousGesture(orbitGesture)
        .simultaneousGesture(zoomGesture)
        .simultaneousGesture(twistGesture)
        .overlay {
            if !stage.loaded {
                ProgressView().controlSize(.large)
            }
        }
    }

    /// Pin the control bar to the volume's FRONT-BOTTOM — right above the
    /// system window bar, nearest the user — tilted up at eye line. (Scene
    /// `.bottom` ornaments sit at CENTRE depth on volumes, a metre behind
    /// the front face and buried near the content: ungazeable.)
    private func placeControls(_ controls: Entity) {
        let e = stage.measuredExtents ?? SIMD3(1.0, 1.2, 1.0)
        controls.position = SIMD3(0, -e.y / 2 + 0.11, e.z / 2 - 0.02)
        controls.orientation = simd_quatf(angle: -0.32, axis: SIMD3(1, 0, 0))
    }

    /// Measure the volume's REAL interior via physicalMetrics (authoritative
    /// points→metres) and derive everything: robot scale, floor height, home
    /// seat, and carry clamps — with margins so no wall can slice the model.
    /// visionOS restores persisted window sizes, so assumed dimensions are
    /// never trusted.
    private func fitToVolume(size: Size3D) {
        let m = physicalMetrics.convert(size, to: .meters)
        let ex = Float(m.width), ey = Float(m.height)
        let ez = max(Float(m.depth), 0.3)
        guard ex > 0.05, ey > 0.05 else { return }

        // Re-derive only when the box meaningfully changed.
        if let prev = stage.measuredExtents,
           abs(prev.x - ex) < 0.01, abs(prev.y - ey) < 0.01, abs(prev.z - ez) < 0.01 {
            return
        }
        stage.measuredExtents = SIMD3(ex, ey, ez)

        // Robot at home pose: ~1.6 m tall, ~1.3 m span with arms.
        let s = min(ey * 0.8 / 1.6, ex * 0.7 / 1.3, ez * 0.7 / 1.3)
        let floor = -ey / 2 + ey * 0.02
        let seat = SIMD3<Float>(0, floor, 0)
        stage.baseScale = s
        stage.volumeFloor = floor
        stage.homeSeat = seat
        stage.measuredBounds = (min: SIMD3(-ex * 0.5 + 0.7 * s, floor, -ez * 0.5 + 0.7 * s),
                                max: SIMD3(ex * 0.5 - 0.7 * s, ey * 0.2, ez * 0.5 - 0.7 * s))
        stage.rig.root.scale = SIMD3(repeating: s)
        stage.rig.root.position = seat
    }

    /// Convert the view's point size into scene metres and size/place the
    /// robot to sit fully inside it (feet near the bottom, slight depth
    /// recess so it never pokes through the window plane).
    private func fitToView(_ content: RealityViewContent, size: CGSize) {
        guard size.width > 1, size.height > 1 else { return }
        let o = content.convert(Point3D(x: 0, y: 0, z: 0), from: .local, to: .scene)
        let c = content.convert(Point3D(x: size.width, y: size.height, z: 0), from: .local, to: .scene)
        let h = abs(c.y - o.y)
        guard h > 0.001 else { return }
        let s = Float(h) * 0.66 / Self.robotHeight
        stage.baseScale = s
        stage.setScenePerPoint(Float(h) / Float(size.height))
        stage.rig.root.scale = SIMD3(repeating: s)
        // Bring the model TOWARD the user, clear of the window plane: windows
        // clip content BEHIND the plane, which is what kept slicing the model.
        stage.rig.root.position = SIMD3(0, Float(-h / 2) + Float(h) * floorRaise,
                                        max(0.06, 0.6 * s))
        // The re-fit just moved the rig out from under any adopted framing —
        // re-measure the anchor so the next drag continues instead of jumping.
        stage.reseatFraming()
        Task { @MainActor in fittedSize = size }
    }

    private func pose() {
        // Glide toward the newest telemetry instead of snapping per message:
        // over jittery links the 20 Hz robot_state stream arrives in bursts,
        // and unsmoothed posing makes the model visibly shake.
        stage.rig.pose(joints: stage.smoother.step(toward: controller.liveModelJoints),
                       heading: stage.smoother.stepHeading(toward: controller.state.heading),
                       baseQuat: controller.state.baseQuat)
        // The planner's ghost-preview / execute step, AFTER telemetry posing so
        // the previewed configuration always wins the frame on its own joints.
        // Resolved through the stage CLASS (stage.manip), never the struct's
        // `manip` copy: this method runs inside the SceneEvents.Update closure,
        // which captured the view value from BEFORE the planner was wired.
        (stage.manip ?? manip)?.tick(now: CACurrentMediaTime())
    }

    // MARK: - .links mode: pinch-drag a highlighted arm link

    private var linkDragGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .targetedToAnyEntity()
            .onChanged { v in
                guard interaction == .links else { return }
                let p = v.convert(v.location3D, from: .local, to: .scene)
                // Planner handles/rings take the drag whole (targetedToAnyEntity
                // keeps v.entity fixed for the drag, so claim once at start).
                if manipDragging {
                    manip?.updateDrag(scenePoint: p)
                    return
                }
                if !linkDragging, dragStartScene == nil,
                   let m = manip, m.claims(v.entity) {
                    manipDragging = true
                    m.beginDrag(entity: v.entity, scenePoint: p)
                    return
                }
                guard let link = stage.rig.link(of: v.entity) else { return }
                if dragStartScene == nil { dragStartScene = p }
                linkDragging = true
                onLinkDrag(link, v.entity, p - (dragStartScene ?? p), dragStartScene ?? p)
            }
            .onEnded { _ in
                if manipDragging { manip?.endDrag() }
                manipDragging = false
                if linkDragging { onLinkDragEnd() }
                linkDragging = false
                dragStartScene = nil
            }
    }

    // MARK: - .move mode: grab the robot and carry it

    private var moveDragGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .targetedToAnyEntity()
            .onChanged { v in
                guard interaction == .move else { return }
                let p = v.convert(v.location3D, from: .local, to: .scene)
                let rootScene = stage.rig.root.position(relativeTo: nil)
                if moveStart == nil { moveStart = rootScene - p }
                var target = (moveStart ?? .zero) + p
                if let b = stage.measuredBounds ?? moveBounds {
                    target = simd_clamp(target, b.min, b.max)
                }
                stage.rig.root.setPosition(target, relativeTo: nil)
            }
            .onEnded { _ in moveStart = nil }
    }

    /// Two-hand twist rotates the model about its vertical axis (.move mode).
    private var twistGesture: some Gesture {
        RotateGesture3D()
            .onChanged { v in
                guard interaction == .move else { return }
                let q = v.rotation.quaternion
                let yaw = Float(atan2(2 * (q.real * q.imag.y + q.imag.x * q.imag.z),
                                      1 - 2 * (q.imag.y * q.imag.y + q.imag.z * q.imag.z)))
                stage.twist(yaw)
            }
            .onEnded { _ in stage.twistEnded() }
    }

    /// Drag empty space to rotate the model — available in every mode except
    /// .move (where dragging carries the robot instead). Viewers rotate too.
    private var orbitGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { v in
                guard interaction != .move, !linkDragging, !manipDragging else { return }
                stage.orbit(dx: Float(v.translation.width) / 600,
                            dy: Float(v.translation.height) / 600)
            }
            .onEnded { _ in
                if interaction != .move { stage.orbitEnded() }
            }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { v in stage.zoom(Float(v.magnification)) }
            .onEnded { _ in stage.zoomEnded() }
    }
}
