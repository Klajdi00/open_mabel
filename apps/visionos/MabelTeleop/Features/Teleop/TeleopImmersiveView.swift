//
//  TeleopImmersiveView.swift
//  MabelTeleop
//
//  The ONE unified teleop space (no Drive/Arms split like the phone):
//
//   • FULL immersion shows the robot's stereo head cameras in a flat,
//     HEAD-LOCKED window the size of the 1st-person window view (2026-10-30;
//     was true scale, 83° tall, which the headset's view cut off), left eye ←
//     left camera, right eye ← right camera, per-eye shift. Looking down, it
//     rises just enough to stay above the level control deck (windowLift). The two wrist cams (upper corners) and
//     the live mini robot (lower right) ride the same head anchor, so all
//     three are always in frame. (Replaced the 360° wrap, 2026-10-16.)
//   • Body tracking (toggleable): ARKit hand skeletons stream to the
//     server's retargeter — wrists via the SE(3) clutch mapping, fingers via
//     per-joint ORCA retargeting. Head motion drives the robot's neck.
//     Live L/R indicators show whether ARKit currently tracks each hand.
//   • Navigation (FULL immersion only — the window hides): the flat 2-D
//     drive pad (`DrivePadDock`, TRANSLATE · LIFT · ROTATE), the SAME control
//     the window carries on its bottom ornament. The floating `Joystick3D`
//     pair it replaced is gone (operator, 2026-10-16: "do not use the 3d
//     joystick — keep things 2D and easy to toggle").
//

import SwiftUI
import Combine
import RealityKit
import simd

struct TeleopImmersiveView: View {
    @EnvironmentObject private var controller: RobotController

    @State private var panel: StereoWindowPanel?
    /// Everything that follows the operator's head in full immersion.
    @State private var headRig: AnchorEntity? = nil
    /// The CONTROL DECK: drive pad · Start/Stop · body switches · console ·
    /// status, around the camera window, following the operator's head
    /// (DeckFollower `.follow`: heading + pitch, with a deadband so it can be
    /// looked at and pinched) or, in `.steady`, the heading only and lazily.
    @State private var deckRig: Entity? = nil
    /// The STATUS bar's own root: heading AND pitch (it sits inside the window's
    /// top edge, so it moves with the window). The deck's root follows HEADING
    /// only — pitch level with the room, so a nod brings it into view (2026-10-23).
    @State private var statusRig: Entity? = nil
    @State private var deckEntity: Entity? = nil
    @State private var statusEntity: Entity? = nil
    @State private var deckSub: EventSubscription? = nil
    @State private var follower = DeckFollower()
    @State private var deckFollower: DeckFollower = {
        var f = DeckFollower(); f.followsPitch = false; return f
    }()
    @AppStorage("headWindowScaleFPV") private var windowScale: Double = 1
    @AppStorage("deckFollowMode") private var deckMode: String = DeckFollower.Mode.follow.rawValue

    // First head-video frame arrived — hides the "no head video" notice and
    // lets the head window show (it stays hidden before that: a frameless window
    // in full immersion is a pure black screen).
    @State private var panelLive = false
    // Tracking-session token this space instance claimed (see `.task`).
    @State private var trackingSession = 0
    @State private var noVideoEntity: Entity? = nil
    @State private var voidStage: Entity? = nil

    // Live mini robot, head-locked in the lower-right of the view.
    @State private var miniRig: RobotRigEntity? = nil
    @State private var miniPoseSub: EventSubscription? = nil
    @State private var miniSmoother = PoseSmoother()

    // Hand pose-estimation overlay: one dot per ARKit joint (cyan = left,
    // magenta = right) — instant visual confirmation of WHAT the robot is
    // being driven with. Applied SYNCHRONOUSLY from the render loop (see
    // HandDotOverlay), never from an async MainActor task: under whole-body
    // load the main actor's task queue backs up and an awaiting overlay loop
    // is its lowest-priority customer — the 2026-07-29 "finger overlay goes
    // super laggy the moment whole-body engages" report. nil until the make
    // closure builds it (an @State initial value re-evaluates on every struct
    // re-init — 54 entities per uiTick, all discarded).
    @State private var handViz: HandDotOverlay? = nil
    @State private var handVizSub: EventSubscription? = nil

    var body: some View {
        RealityView { content, attachments in
            // ── HEAD RIG: everything that follows the operator's head ─────────
            // `.continuous` = rigidly locked to the head every frame. The window's
            // true-scale geometry is defined RELATIVE TO THE EYES, so it must not
            // lag them (a lazy-follow window would be the wrong size/angle while
            // it catches up). Display-only content lives here — nothing on it
            // is pinch-targeted; the interactive panels stay in the room.
            let head = AnchorEntity(.head, trackingMode: .continuous)
            head.name = "headRig"
            head.isEnabled = controller.fullImmersion
            content.add(head)
            headRig = head

            let cam = controller.headCamera
            let stereo = StereoWindowPanel(
                fovyDeg: Float(cam?.fovy ?? Double(StereoWindowPanel.defaultFovyDeg)),
                aspect: Float(cam?.aspect ?? Double(StereoWindowPanel.defaultAspect)),
                distance: StereoWindowGeometry.fpvDistance,
                scale: DeckLayout.windowScale(Float(windowScale)),
                matchedWidth: StereoWindowGeometry.fpvWidth)
            stereo.wantVisible = controller.fullImmersion
            stereo.onFirstFrame = { panelLive = true }
            head.addChild(stereo.entity)
            panel = stereo
            syncPanelFeeds(stereo)
            #if DEBUG
            let g = stereo.geometry
            NSLog("[MABEL-WINDOW] fovy=\(g.fovyDeg)° aspect=\(g.aspect) hfov=\(g.hfovRad * 180 / .pi)° D=\(g.distance)m W=\(g.width)m H=\(g.height)m eyeShiftU=\(g.eyeShiftU) (cancels \(g.unshiftedVergenceError * 180 / .pi)° vergence)")
            #endif

            // ── CONTROL DECK low in the view: heading follows, pitch stays LEVEL ──
            // (DeckFollower.followsPitch = false) so looking down brings it up into
            // view — 2026-10-23, "when I move the head down the bar moves away".
            let deckRoot = Entity()
            deckRoot.name = "deckRig"
            content.add(deckRoot)
            deckRig = deckRoot
            if let deck = attachments.entity(for: "deck") {
                deck.scale = SIMD3(repeating: DeckLayout.scale)
                deckRoot.addChild(deck)
                deck.isEnabled = consoleWanted
                deckEntity = deck
            }
            // Status bar at the TOP (operator, 2026-10-17: "the status and e stop should
            // always be at the top") — INSIDE the window's top edge since 2026-10-23:
            // above it, it sat 36–40° up, at the headset's field-of-view edge. Own
            // root, heading + pitch, so it stays framed with the window.
            let statusRoot = Entity()
            statusRoot.name = "statusRig"
            content.add(statusRoot)
            statusRig = statusRoot
            if let bar = attachments.entity(for: "status") {
                bar.scale = SIMD3(repeating: DeckLayout.scale)
                statusRoot.addChild(bar)
                bar.isEnabled = statusWanted
                statusEntity = bar
            }
            let headSvc = controller.headService
            deckSub = content.subscribe(to: SceneEvents.Update.self) { ev in
                let step = {
                    guard let rig = deckRig, let sRig = statusRig else { return }
                    // Head pose from the session's own ARKit provider; none yet
                    // (first frames) → a seated/standing default straight ahead.
                    let head = headSvc.currentPose()?.transform.simd
                        ?? { var m = matrix_identity_float4x4; m.columns.3 = SIMD4(0, 1.4, 0, 1); return m }()
                    let pos = SIMD3(head.columns.3.x, head.columns.3.y, head.columns.3.z)
                    let mode = DeckFollower.Mode(rawValue: deckMode) ?? .follow
                    let dt = Float(min(max(ev.deltaTime, 1.0 / 240), 0.1))
                    let yaw = DeckFollower.yaw(of: head), pitch = DeckFollower.pitch(of: head)
                    follower.mode = mode
                    deckFollower.mode = mode
                    sRig.transform = Transform(matrix: follower.step(
                        headPos: pos, headYaw: yaw, headPitch: pitch, dt: dt))
                    rig.transform = Transform(matrix: deckFollower.step(
                        headPos: pos, headYaw: yaw, headPitch: pitch, dt: dt))
                    if let deck = deckEntity { DeckLayout.placeDeck(deck) }
                    if let p = panel {
                        // Looking down: the window rises just enough to stay above the deck.
                        let lift = consoleWanted ? DeckLayout.windowLift(pitchDown: -pitch, window: p.geometry) : 0
                        p.entity.orientation = simd_quatf(angle: lift, axis: SIMD3(1, 0, 0))
                        if let bar = statusEntity { DeckLayout.place(bar, insideTopOf: p.geometry, lift: lift) }
                    }
                }
                if Thread.isMainThread { MainActor.assumeIsolated { step() } }
                else { Task { @MainActor in step() } }
            }

            // ── Live mini robot: head-locked, LOWER RIGHT of the view ─────────
            // Same corner as the window's 1st-person mini rig. In front of the
            // video window (1.2 m vs 2 m) so it draws over it. Display-only:
            // no InputTarget — a collider on head-locked content would sit in
            // the gaze path of everything behind it, wherever the operator looks.
            let rig = RobotRigEntity()
            rig.root.scale = SIMD3(repeating: HeadLayout.miniScale)
            rig.root.position = HeadLayout.miniPosition
            rig.root.orientation = simd_quatf(angle: -0.45, axis: SIMD3(0, 1, 0))
            head.addChild(rig.root)
            miniRig = rig
            Task { await rig.load() }
            // Full immersion has no environment light, so the PBR rig renders
            // near-black. A soft key light riding the head (above, behind the
            // eye line) lights it the same way whichever way the operator
            // faces. Unlit content (video, cards) ignores it.
            let key = DirectionalLight()
            key.light.intensity = 2200
            key.look(at: HeadLayout.miniPosition, from: SIMD3(-0.4, 1.0, 0.6), relativeTo: nil)
            head.addChild(key)
            miniPoseSub = content.subscribe(to: SceneEvents.Update.self) { _ in
                // SYNCHRONOUS apply — a Task per frame backlogs the main actor.
                let apply = {
                    rig.pose(joints: miniSmoother.step(toward: controller.liveModelJoints),
                             heading: miniSmoother.stepHeading(toward: controller.state.heading),
                             baseQuat: controller.state.baseQuat)
                }
                if Thread.isMainThread { MainActor.assumeIsolated { apply() } }
                else { Task { @MainActor in apply() } }
            }

            // ── Void stage: UNLIT grid floor + horizon markers so FULL
            //    immersion is a dark studio, never a black screen (glass and
            //    PBR content is invisible in a lightless void — unlit isn't). ──
            let stage = Self.makeVoidStage()
            stage.isEnabled = controller.fullImmersion
            content.add(stage)
            voidStage = stage

            if let card = attachments.entity(for: "novideo") {
                card.position = SIMD3(0, 0, -1.5)        // head-locked, dead centre
                head.addChild(card)
                noVideoEntity = card
            }
            // ── Hand pose dots (the v1 overlay) — ALWAYS ON in this space ────
            // Gating them off in tracking-only mode (2026-07-18 operator bug:
            // "the hand tracking point overlay is gone") removed the ONE live
            // confirmation that ARKit sees your hands, in exactly the mode
            // where whole-body tracking is driven from the window. The dots
            // sit ON the operator's physical hands, so they never occlude the
            // window UI — unlike the sticks/console, they have no reason to hide.
            // RENDER-LOOP driven (2026-07-29): the update closure captures the
            // overlay CLASS + the broadcaster (both reference types, gotcha
            // #23) and applies the newest ARKit snapshot synchronously each
            // frame — zero async hops, so main-actor congestion (whole-body
            // engaged: robot_state decode + SwiftUI invalidations) can never
            // starve it the way it starved the old await-loop.
            let viz = HandDotOverlay()
            content.add(viz.root)
            handViz = viz
            let handBroadcast = controller.handService.broadcaster
            handVizSub = content.subscribe(to: SceneEvents.Update.self) { _ in
                viz.tick(handBroadcast.latest(), now: CACurrentMediaTime())
            }

            // ── Wrist cams: head-locked, UPPER LEFT / UPPER RIGHT of the view ──
            // The window cockpit's FPV layout (wrists in the top corners, mini
            // robot bottom-right), so the two views read the same. Toed in and
            // tilted to face the eye square-on.
            for (id, side) in [("wrist_left", Float(-1)), ("wrist_right", Float(1))] {
                guard let cam = attachments.entity(for: id) else { continue }
                let p = HeadLayout.wristPosition(side: side)
                cam.position = p
                cam.scale = SIMD3(repeating: HeadLayout.wristScale)
                cam.orientation = simd_quatf(angle: atan2(-p.x, -p.z), axis: SIMD3(0, 1, 0))
                    * simd_quatf(angle: atan2(p.y, -p.z), axis: SIMD3(1, 0, 0))
                head.addChild(cam)
            }
        } attachments: {
            Attachment(id: "deck") { ImmersiveControlDeck() }
            Attachment(id: "status") { GlobalStatusBar() }
            Attachment(id: "novideo") { if !panelLive { noVideoCard } }
            Attachment(id: "wrist_left") { wristCam("L WRIST", "wrist_left") }
            Attachment(id: "wrist_right") { wristCam("R WRIST", "wrist_right") }
        }
        .task {
            // STICK-ONLY: the cockpit opened this space purely to float the two
            // drive sticks in the room. Claim NOTHING — no session token, no
            // ARKit providers, no `immersiveActive` — so a windowed operator
            // who just wants to reach for a joystick neither hands the robot's
            // neck to their head nor raises a hand-tracking prompt. The session
            // starts later if this space is ever promoted (see the onChange).
            guard !controller.stickSpaceOnly else { return }
            await startTrackingSession()
        }
        .onChange(of: controller.stickSpaceOnly) { _, only in
            // PROMOTION (see `RobotController.needsImmersiveOpen`): something
            // asked for a real tracking session while this space was up as the
            // cockpit's stick host. The space is already open, so there is no
            // second `openImmersiveSpace` to run — claim the session here.
            applyFurniture()
            syncPanelFeeds()
            guard !only, trackingSession == 0 else { return }
            Task { await startTrackingSession() }
        }
        .onChange(of: windowScale) { _, s in panel?.setScale(Float(s)) }
        .onChange(of: controller.headCamera?.fovy) { _, fovy in
            if let fovy, let aspect = controller.headCamera?.aspect {
                panel?.updateIntrinsics(fovyDeg: Float(fovy), aspect: Float(aspect))
            }
        }
        .onChange(of: controller.videoBase) { _, _ in
            // The space can open BEFORE the server connection completes
            // (remote fallback takes ~8 s), in which case the make closure
            // saw videoBase == nil and the head window never started — restart
            // the feeds whenever the video endpoints appear or change.
            syncPanelFeeds()
        }
        .onChange(of: controller.trackingOnlyMode) { _, _ in
            // Flipping into Full Immersive from a tracking-only session brings
            // the space's UI back; entering tracking-only hides the sticks and
            // console (the window has those) — but NEVER the hand dots, which
            // are the live is-ARKit-seeing-me confirmation in both modes.
            applyFurniture()
        }
        .onChange(of: controller.fullImmersion) { _, full in
            panel?.wantVisible = full
            headRig?.isEnabled = full          // window + wrist cams + mini robot
            voidStage?.isEnabled = full
            applyFurniture()
        }
        .onChange(of: controller.estopEngaged) { _, _ in applyFurniture() }
        .onChange(of: controller.passiveCompliance) { _, _ in applyFurniture() }
        .onDisappear {
            deckSub?.cancel(); deckSub = nil
            handVizSub?.cancel()
            handVizSub = nil
            panel?.stop()
            controller.stopBase()
            controller.stickSpaceOnly = false
            // A stick-only space never claimed a session (token 0), and token 0
            // is never a live generation — but say so explicitly rather than
            // leaning on `stopImmersiveTracking`'s guard, because that call also
            // clears `immersiveActive`/`bodyTracking` for whoever DOES own them.
            if trackingSession != 0 {
                controller.stopImmersiveTracking(session: trackingSession)
            }
        }
    }

    /// Claim the session identity SYNCHRONOUSLY, then start ARKit. The token
    /// stays with THIS space instance (@State): on an `.immersionStyle` flip
    /// visionOS re-creates the space, and the replaced instance's `.onDisappear`
    /// must not tear down the session the new instance has already claimed.
    private func startTrackingSession() async {
        let gen = controller.beginTrackingSession()
        trackingSession = gen
        await controller.startImmersiveTracking(session: gen)
    }

    /// The console + Start/Stop CTA belong to a REAL teleop session in this
    /// space: hidden both when the window owns the session (tracking-only) and
    /// when there is no session at all (stick-only).
    private var consoleWanted: Bool {
        !controller.trackingOnlyMode && !controller.stickSpaceOnly
    }

    /// The status bar is FULL IMMERSION's top bar. In mixed immersion the window
    /// is still up and carries its own status bar, so a second one floating in the
    /// room is a duplicate (operator, 2026-10-24: "the full immersive top bar is
    /// floating even not in the full immersive mode").
    private var statusWanted: Bool {
        controller.fullImmersion && !controller.stickSpaceOnly
    }

    /// Re-apply every purpose-dependent visibility in one place. Split out
    /// because several flags (`trackingOnlyMode`, `stickSpaceOnly`,
    /// `fullImmersion`, e-stop, Soft) decide it, and onChange handlers each
    /// re-deriving their own subset is how they drift apart.
    private func applyFurniture() {
        deckEntity?.isEnabled = consoleWanted
        statusEntity?.isEnabled = statusWanted
        #if DEBUG
        NSLog("[MABEL-DECK] deck enabled=\(consoleWanted) full=\(controller.fullImmersion) estop=\(controller.estopEngaged) soft=\(controller.passiveCompliance)")
        #endif
    }

    /// Start the head window's feeds unless this space is only hosting the
    /// cockpit's sticks — there the operator is watching the SAME head stream
    /// in the window, and a second decoder of it is pure cost on the one
    /// pipeline the whole app's glass-to-glass latency rides on. Re-runs on
    /// promotion, so a space that becomes a real session gets its window.
    /// `stereo` is passed explicitly from the make closure: `panel` is @State
    /// being assigned in that same closure, and a read-back there is not
    /// guaranteed to see the write.
    private func syncPanelFeeds(_ stereo: StereoWindowPanel? = nil) {
        guard !controller.stickSpaceOnly, let p = stereo ?? panel else { return }
        startPanelFeeds(p)
    }

    /// (Re)start the head-window video: UDP primary, MJPEG fallback. LEFT eye
    /// uses /camera/main — served in EVERY camera-server config (head_left is
    /// weight-0 unless --stereo, which used to leave this panel black);
    /// head_right upgrades to true stereo when the server renders it.
    /// No-op while disconnected (videoBase nil) — onChange retries later.
    private func startPanelFeeds(_ stereo: StereoWindowPanel) {
        guard let left = controller.videoURL("main"),
              let right = controller.videoURL("head_right") else { return }
        let host = controller.videoHost
        let port = controller.videoUDPPort
        let codec = controller.videoCodec
        // PRIMARY source: "head_stereo" — both eyes from ONE capture, side by
        // side (synced by construction). The eye streams are the fallback.
        let sources = HeadVideoSources(
            stereoURL: controller.videoURL("head_stereo"),
            leftURL: left, rightURL: right,
            udpHost: host, udpPort: port, codec: codec)
        Task { await stereo.start(sources: sources) }
    }

    /// A self-luminous stage for full immersion: 14×14 m grid floor (drawn
    /// into one texture — a single entity), a soft centre marker, and four
    /// cardinal pillars for orientation. Unlit materials render identically
    /// with or without scene lighting.
    private static func makeVoidStage() -> Entity {
        let root = Entity()
        root.name = "voidStage"
        let size: CGFloat = 1024
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: size, height: size))
        let img = renderer.image { ctx in
            let c = ctx.cgContext
            c.setFillColor(UIColor(white: 0.07, alpha: 1).cgColor)
            c.fill(CGRect(x: 0, y: 0, width: size, height: size))
            c.setStrokeColor(UIColor(white: 0.16, alpha: 1).cgColor)
            c.setLineWidth(1.5)
            let step = size / 14      // one cell = 1 m
            for i in 0...14 {
                let p = CGFloat(i) * step
                c.move(to: CGPoint(x: p, y: 0)); c.addLine(to: CGPoint(x: p, y: size))
                c.move(to: CGPoint(x: 0, y: p)); c.addLine(to: CGPoint(x: size, y: p))
            }
            c.strokePath()
            c.setStrokeColor(UIColor(red: 1.0, green: 0.55, blue: 0.25, alpha: 0.55).cgColor)
            c.setLineWidth(3)
            c.strokeEllipse(in: CGRect(x: size/2 - 40, y: size/2 - 40, width: 80, height: 80))
        }
        if let cg = img.cgImage,
           let tex = try? TextureResource(image: cg, options: .init(semantic: .color)) {
            var mat = UnlitMaterial()
            mat.color = .init(tint: .white, texture: .init(tex))
            let floor = ModelEntity(mesh: .generatePlane(width: 14, depth: 14), materials: [mat])
            floor.position = SIMD3(0, 0.001, 0)
            root.addChild(floor)
        }
        for (dx, dz) in [(Float(6), Float(0)), (-6, 0), (0, 6), (0, -6)] {
            let pillar = ModelEntity(mesh: .generateBox(size: SIMD3(0.05, 1.2, 0.05)),
                                     materials: [UnlitMaterial(color: UIColor(white: 0.22, alpha: 1))])
            pillar.position = SIMD3(dx, 0.6, dz)
            root.addChild(pillar)
        }
        return root
    }

    /// Honest full-immersion state while no head video streams (offline/demo,
    /// or the stream is still connecting): the drive pad, console and mini robot
    /// stay live — this card says WHY there's no head view.
    private var noVideoCard: some View {
        VStack(spacing: 8) {
            Image(systemName: "video.slash.fill")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(Theme.Palette.warn)
            Text("No head video").font(.system(size: 16, weight: .bold))
                .foregroundStyle(.white)
            Text("Connect to the robot or sim for the head-camera view.\nDrive pad, console and the mini robot stay live.")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
        }
        .padding(18)
        .frame(width: 320)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private func wristCam(_ name: String, _ stream: String) -> some View {
        CameraFeedView(name: name, tint: Theme.Palette.live,
                       streamURL: controller.videoURL(stream),
                       udpHost: controller.videoHost,
                       udpPort: controller.videoUDPPort,
                       codec: controller.videoCodec,
                       stream: stream)
            // Bigger (operator, 2026-07-28) — 380x240 was small enough at the
            // panel's 1.35 m standoff that the wrist view was a thumbnail you
            // squinted at rather than a feed you worked from.
            .frame(width: CGFloat(HeadLayout.wristSizePt.x), height: CGFloat(HeadLayout.wristSizePt.y))
            .glassBackgroundEffect()
    }
}

// MARK: - Hand pose overlay (render-loop driven)

/// The finger-tracking dot overlay: one unlit sphere per ARKit hand joint
/// (cyan = left, magenta = right), applied SYNCHRONOUSLY from a
/// `SceneEvents.Update` subscription reading `SnapshotBroadcaster.latest()`.
///
/// A CLASS on purpose (gotcha #23): the render-update closure captures the
/// view STRUCT by value, so anything it must see live has to ride a reference
/// type. And deliberately NOT an async stream consumer: an awaiting MainActor
/// task is scheduled BEHIND every other main-actor item, so the overlay
/// visibly froze exactly when whole-body teleop loaded the main thread. A
/// synchronous apply inside the render callback costs ~54 transform writes and
/// cannot be starved independently of the frame rate itself.
///
/// Pure entity math — no ARKit — so `HandVizTests` drives it with synthetic
/// snapshots on the simulator.
final class HandDotOverlay {
    let root = Entity()
    private(set) var left: [HandJoint: ModelEntity] = [:]
    private(set) var right: [HandJoint: ModelEntity] = [:]
    private var lastStamp: TimeInterval = -1
    /// Hide the dots once the newest snapshot is older than this — tracking
    /// stopped (or the space is parked) and a frozen skeleton reads as live.
    static let staleAfter: TimeInterval = 0.5

    init() {
        for joint in HandJoint.allCases {
            let lo = Self.dot(.cyan), ro = Self.dot(.magenta)
            root.addChild(lo); root.addChild(ro)
            left[joint] = lo; right[joint] = ro
        }
    }

    private static func dot(_ color: UIColor) -> ModelEntity {
        let e = ModelEntity(mesh: .generateSphere(radius: 0.007),
                            materials: [UnlitMaterial(color: color.withAlphaComponent(0.9))])
        e.isEnabled = false
        return e
    }

    /// One render-frame tick. De-dups on the snapshot timestamp (ARKit ~90 Hz
    /// ≈ render ~90 Hz, so most frames apply exactly one fresh snapshot).
    func tick(_ snap: HandTrackingService.Snapshot?, now: TimeInterval) {
        guard let snap, now - snap.timestamp < Self.staleAfter else {
            hideAll(); return
        }
        guard snap.timestamp != lastStamp else { return }
        lastStamp = snap.timestamp
        Self.apply(hand: snap.left, into: left)
        Self.apply(hand: snap.right, into: right)
    }

    private func hideAll() {
        for e in left.values where e.isEnabled { e.isEnabled = false }
        for e in right.values where e.isEnabled { e.isEnabled = false }
    }

    /// Allocation-free apply: transforms are written every fresh snapshot;
    /// `isEnabled` only on EDGES (an enable flip is a scene-graph change,
    /// a transform write is a component update — the former is the pricier).
    ///
    /// THE HAND IS THE UNIT, NOT THE JOINT (operator, 2026-08-01: "the point
    /// overlay disappears here and there"). This used to gate each dot on its
    /// own `jp.isTracked`, and ARKit flags individual joints untracked
    /// constantly — a curled finger, a thumb behind a palm, anything briefly
    /// occluded — so the overlay SPARKLED: dots winking out and back
    /// independently, several times a second, on a hand that was being tracked
    /// perfectly well. ARKit still reports a usable (estimated) transform for
    /// those joints, so the honest and far calmer rendering is: if the HAND is
    /// tracked, draw all of it; if it is not, draw none of it. That also drops
    /// the per-joint enable churn, which was the expensive half of this loop.
    static func apply(hand: HandPose?, into map: [HandJoint: ModelEntity]) {
        guard let hand, hand.isTracked else {
            for e in map.values where e.isEnabled { e.isEnabled = false }
            return
        }
        let anchor = hand.anchorTransform.simd
        for jp in hand.joints {
            guard let e = map[jp.joint] else { continue }
            e.transform = Transform(matrix: anchor * jp.localTransform.simd)
            if !e.isEnabled { e.isEnabled = true }
        }
    }
}

// MARK: - Start / Stop Teleop (top centre)

/// The primary action of the immersive view, on its own so the attachment
/// builder stays inside the type checker's budget.
/// Head-locked layout for full immersion (metres, head frame: −Z ahead,
/// +Y up). Everything sits INSIDE the video window's angular extent (camera
/// hfov ≈ 100° × vfov 83°) and inside the headset's comfortable gaze, in
/// front of the window (≈1.2 m vs the window's 2 m) so it draws over it.
/// The centre ±15° is left clear: that is where the operator looks, and where
/// the room-fixed status bar / Start Teleop sit when facing forward.
enum HeadLayout {
    /// Wrist-cam centres: ±27.3° across at eye level, 1.2 m out. 0.55 → 0.62 m (2026-10-30) so
    /// their inner edges (≈21.7°) clear the head window at its largest size (±18.6°); pinned by
    /// FullImmersionWindowTests.
    static func wristPosition(side: Float) -> SIMD3<Float> { SIMD3(side * 0.62, 0.04, -1.2) }
    /// 520 × 330 pt panels at ~1360 pt/m × this scale → 0.287 × 0.182 m.
    static let wristScale: Float = 0.75
    static let wristSizePt = SIMD2<Float>(520, 330)
    /// Mini robot (feet origin): 27° right (was 18°, inside the new window's corner), its
    /// 0.37 m height spanning ≈6°–23° below.
    static let miniPosition = SIMD3<Float>(0.62, -0.50, -1.2)
    static let miniScale: Float = 0.22
}

/// Where the control deck and the status bar sit, in their follower rigs' frames
/// (eye height, facing −Z), tilted to face the eyes.
///
/// 2026-10-23 (operator: "make sure the bottom bar can be viewed … at one point it
/// was so low that I cannot even see — when I move the head down the bar moves
/// away … do not have that again"). The camera window at true scale is ~83° tall,
/// i.e. as tall as the headset's whole vertical field of view (~90°, ±45°), so the
/// old rule — status bar ABOVE the window's top edge, deck BELOW its bottom edge —
/// put both at 36°+ off-axis, at or past the edge of the view. And the deck
/// followed head pitch, so nodding toward it pushed it further away. Now:
///   • DECK: a FIXED angle below the horizon (`deckTop` for its top edge), on a
///     follower whose pitch stays LEVEL. Looking straight ahead its top is in the
///     lower part of the view; looking down brings the whole deck up to the centre.
///     It draws in front of the window's lower edge (0.8 m vs 2 m).
///   • STATUS: INSIDE the window's top edge, capped at `statusTopMax` so a large
///     (true-scale) window cannot push it out of view. It rides the window.
enum DeckLayout {
    #if DEBUG
    /// debugDeckPreview: the Simulator's fixed camera cannot look up or down, so
    /// for a LAYOUT screenshot the WINDOW is shrunk to `previewScale`; the deck and
    /// status bar use the same rules.
    static let previewFlag = UserDefaults.standard.bool(forKey: "debugDeckPreview")
    static let previewScale: Float = 0.27
    nonisolated(unsafe) static var loggedReal = false
    #endif
    static let distance: Float = 0.80          // m from the eyes
    static let scale: Float = 0.75             // SwiftUI points → metres
    static let gap: Float = 1.0 * .pi / 180    // status bar inset from the window's top edge
    /// The deck's TOP edge, below the horizon. Well inside the lower field of view
    /// (Vision Pro ≈ ±45° vertical; `viewHalfV` is a conservative 40°).
    /// 24° → 34° (operator, 2026-10-24: "the bottom control panel is too high —
    /// bring the control panel way lower"): its top edge still shows in the lower
    /// view looking ahead, and the level-pitch rig means a nod always brings the
    /// whole deck to the centre.
    static let deckTop: Float = 34 * .pi / 180
    /// Highest the status bar's TOP edge may sit above eye level. 30° → 18°
    /// (operator, 2026-10-24: "the top bar is way too high") — a glance up, not a
    /// look up, while still inside the window's top edge.
    static let statusTopMax: Float = 18 * .pi / 180
    /// Conservative half vertical field of view of the headset the layout must fit.
    static let viewHalfV: Float = 40 * .pi / 180

    /// The head window's scale: the operator's setting (debugDeckPreview: `previewScale`).
    static func windowScale(_ s: Float) -> Float {
        #if DEBUG
        if previewFlag { return previewScale }
        #endif
        return s
    }

    private static func halfAngle(_ h: Float) -> Float {
        atan(((h.isFinite && h > 0.01) ? h : 0.3) / 2 / distance)
    }

    /// Deck centre (rad, + = below the horizon) for a deck `h` m tall.
    static func deckCentreAngle(height h: Float) -> Float { deckTop + halfAngle(h) }

    /// Status-bar centre (rad, NEGATIVE = above eye level) for a bar `h` m tall,
    /// its top edge `gap` inside the window's top edge, never above `statusTopMax`.
    ///
    /// 2026-10-30: the window is now the 1st-person window's size (~24° tall), so the bar sits
    /// just ABOVE its top edge — inside, it covered the top of the picture. A window too tall for
    /// that (true scale) still gets the old inside-the-top placement under `statusTopMax`.
    static func statusCentreAngle(height h: Float, window g: StereoWindowGeometry, lift: Float = 0) -> Float {
        let above = g.halfHeightAngle + lift + gap + 2 * halfAngle(h)
        // Lifted (looking down) the bar may go higher in the view, still inside it.
        let limit = lift > 0 ? viewHalfV - 2 * .pi / 180 : statusTopMax
        let top = above <= limit ? above : min(g.halfHeightAngle + lift - gap, limit)
        return -(top - halfAngle(h))
    }

    /// How far (rad, up) the head-locked window rises in the view so the deck never covers it.
    /// The deck's rig stays LEVEL (`deckTop` below the horizon, so a nod brings it up), while the
    /// window rides the head: looking down `pitchDown` rad puts the deck's top edge
    /// `deckTop − pitchDown` below the view axis. The window keeps its bottom edge `gap` above
    /// that, and never rises so far that the bar above it leaves the view (`viewHalfV`, keeping
    /// 7° for the gap + a ≤ 3° bar + 2° margin). The cap binds only far down: the window is fully
    /// clear to ≈42° at 100 %, ≈38° at 120 %, 45° at 60 % (FullImmersionWindowTests); past that
    /// the operator is looking AT the deck and its top edge reaches the window's bottom.
    static func windowLift(pitchDown: Float, window g: StereoWindowGeometry) -> Float {
        let need = g.halfHeightAngle + gap - (deckTop - pitchDown)
        let cap = max(0, viewHalfV - g.halfHeightAngle - 7 * .pi / 180)
        return min(max(0, need), cap)
    }

    /// Deck: fixed angle below the horizon (its rig keeps pitch level).
    static func placeDeck(_ deck: Entity) {
        let h = deck.visualBounds(relativeTo: deck.parent).extents.y
        let theta = deckCentreAngle(height: h)
        #if DEBUG
        if !loggedReal {
            loggedReal = true
            NSLog("[MABEL-DECK] deck centre \(theta * 180 / .pi)° below horizon, top \(deckTop * 180 / .pi)°, height \(h) m at \(distance) m")
        }
        #endif
        seat(deck, theta)
    }

    /// Status bar: inside the window's top edge.
    static func place(_ bar: Entity, insideTopOf g: StereoWindowGeometry, lift: Float = 0) {
        seat(bar, statusCentreAngle(height: bar.visualBounds(relativeTo: bar.parent).extents.y,
                                    window: g, lift: lift))
    }

    /// Seat at `theta` below eye level (negative = above), facing the eye.
    private static func seat(_ e: Entity, _ theta: Float) {
        e.position = SIMD3(0, -distance * sin(theta), -distance * cos(theta))
        e.orientation = simd_quatf(angle: -theta, axis: SIMD3(1, 0, 0))
    }
}

/// THE CONTROL DECK — the full-immersion controls low in the view, top to
/// bottom: drive pad, Start/Stop + body switches, console. (The status bar sits
/// inside the window's top edge: `DeckLayout.place(_:insideTopOf:)`.)
private struct ImmersiveControlDeck: View {
    @EnvironmentObject private var controller: RobotController

    /// Same gate as the window's drive pad: full immersion, motors live, Stiff.
    private var padWanted: Bool {
        controller.fullImmersion && !controller.estopEngaged && !controller.passiveCompliance
    }

    var body: some View {
        VStack(spacing: 12) {
            // fixedSize: hug its own controls and CENTRE (it otherwise stretched
            // to the console's width with the controls pushed left).
            if padWanted { DrivePadDock(stickSize: 130).fixedSize() }
            HStack(spacing: 12) {
                TeleopStartStop()
                BodyControls(onRecalibrate: nil)
            }
            TeleopConsole()
        }
        .animation(Theme.Motion.snappy, value: padWanted)
    }
}

private struct TeleopStartStop: View {
    @EnvironmentObject private var controller: RobotController

    /// START / STOP TELEOP — the primary action of the immersive view, at the
    /// top and prominent. Red while live (stopping is the urgent one), accent
    /// when idle.
    var body: some View {
        Button {
            Haptics.rigid()
            // STOP = FREEZE: mirroring off AND the head frozen, so nothing —
            // neck included — keeps following; the robot holds where it is.
            if controller.bodyTracking { controller.stopWholeBodyTeleop() }
            else { controller.engageWholeBody() }
        } label: {
            HStack(spacing: 9) {
                Image(systemName: controller.bodyTracking
                      ? "stop.circle.fill" : "figure.arms.open")
                    .font(.system(size: 19, weight: .bold))
                Text(controller.bodyTracking ? "Stop Teleop" : "Start Teleop")
                    .font(.system(size: 18, weight: .bold)).lineLimit(1).fixedSize()
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 26)
            .frame(height: 52)
            .background {
                Capsule(style: .continuous)
                    .fill(controller.bodyTracking
                          ? AnyShapeStyle(Theme.Palette.danger.gradient)
                          : AnyShapeStyle(Theme.accentGradient))
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        .animation(Theme.Motion.snappy, value: controller.bodyTracking)
    }

}

// MARK: - Floating console

/// Glass control bar under the main panel: record/task, body-tracking with
/// live L/R hand indicators, head toggle, recenter, STOP, and exit.
private struct TeleopConsole: View {
    @EnvironmentObject private var controller: RobotController
    @AppStorage("headWindowScaleFPV") private var windowScale: Double = 1
    @AppStorage("deckFollowMode") private var deckMode: String = DeckFollower.Mode.follow.rawValue
    @Environment(\.dismissImmersiveSpace) private var dismissImmersive

    @State private var elapsed = 0
    @State private var timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    @State private var showTaskPicker = false
    @State private var newTaskName = ""

    var body: some View {
        VStack(spacing: 8) {
            if let notice = controller.trackingNotice {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 12, weight: .bold))
                    Text(notice).font(.system(size: 12, weight: .semibold))
                }
                .foregroundStyle(Theme.Palette.warn)
            }
            // WHAT THE ROBOT IS WRITING — on Wi-Fi and on Bluetooth alike: every camera's stream must be
            // delivering, and a refused take says why, instead of a Rec button that only looks like it worked.
            let cams = controller.state.recordCameras
            if controller.recording, !cams.isEmpty {
                let silent = cams.filter { !$0.live }.map(\.name)
                HStack(spacing: 6) {
                    Image(systemName: silent.isEmpty ? "video.fill" : "video.slash.fill")
                        .font(.system(size: 12, weight: .bold))
                    Text("\(cams.count - silent.count)/\(cams.count) cameras writing"
                         + (silent.isEmpty ? "" : " — no frames: " + silent.joined(separator: ", ")))
                        .font(.system(size: 12, weight: .semibold)).lineLimit(1)
                }
                .foregroundStyle(silent.isEmpty ? Theme.Palette.live : Theme.Palette.danger)
            }
            if let err = controller.state.recordError, !err.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.octagon.fill").font(.system(size: 12, weight: .bold))
                    Text(err).font(.system(size: 12, weight: .semibold)).lineLimit(2)
                }
                .foregroundStyle(Theme.Palette.warn)
            }
            consoleRow
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .glassBackgroundEffect()
        .onReceive(timer) { _ in if controller.recording { elapsed += 1 } }
        .popover(isPresented: $showTaskPicker) { taskPicker.padding(4) }
    }

    private var consoleRow: some View {
        HStack(spacing: 6) {
            // ── data collection ──────────────────────────────────────────────
            ConsoleButton(icon: controller.recording ? "stop.fill" : "record.circle",
                          label: controller.recording ? "End · \(timeString)" : "Record",
                          isOn: controller.recording,
                          tint: controller.recording ? Theme.Palette.danger : Theme.Palette.live) {
                Haptics.rigid()
                if controller.recording { controller.endEpisode() }
                else { elapsed = 0; controller.startEpisode() }
            }

            ConsoleButton(icon: "folder", label: "Task · \(controller.episodeCount)") {
                newTaskName = ""; showTaskPicker = true
            }
            .disabled(controller.recording)
            .opacity(controller.recording ? 0.4 : 1)

            Divider().frame(height: 44)

            // ── WHOLE-BODY TRACKING: what of me drives the robot ─────────────
            //
            // The five controls of whole-body teleop, together, because in full
            // immersion the window's chrome is unreachable and this is exactly
            // where an operator discovers they need them. Each of the three
            // channel switches is a SERVER-side change: the channel stops being
            // retargeted and its joints ease to neutral over ~2.5 s, so nothing
            // ever freezes mid-pose (see RobotController.trackNeck).
            // (ENGAGE lives at the TOP now — see `teleopCTA`. It is the primary
            // action of this whole view and the console sits at y = 0.52, below
            // the operator's eye line; a Start button you have to look DOWN for
            // is one you cannot find when you need it.)
            VStack(spacing: 2) {
                ConsoleButton(icon: "hand.raised.fingers.spread", label: "Hands",
                              isOn: controller.trackHands) {
                    Haptics.select()
                    controller.trackHands.toggle()
                }
                handDots
            }
            .disabled(!controller.bodyTracking)
            .opacity(controller.bodyTracking ? 1 : 0.4)

            ConsoleButton(icon: "figure.arms.open", label: "Arms",
                          isOn: controller.trackArms) {
                Haptics.select()
                controller.trackArms.toggle()
            }
            .disabled(!controller.bodyTracking)
            .opacity(controller.bodyTracking ? 1 : 0.4)

            // (NECK, UPPER/WHOLE BODY, TORSO and RE-ANCHOR live in the deck's
            // BodyControls row — the same component as the window's top bar, so
            // there is exactly one of each and they always read the same.)

            // Sim only: re-home the scene — props back to their start positions,
            // robot to the home pose, operator mapping re-anchored (bridge `reset`).
            if controller.canReset {
                ConsoleButton(icon: "arrow.counterclockwise", label: "Reset") {
                    controller.resetSim()
                }
            }

            Divider().frame(height: 44)

            // ── view · safety · exit ─────────────────────────────────────────
            // WINDOW SIZE: 60–120 %, where 100 % is the 1st-person window's size
            // (operator, 2026-10-30; it was true scale, which the headset cut off).
            ConsoleButton(icon: "minus.magnifyingglass", label: "Smaller") {
                Haptics.select()
                windowScale = max(Double(StereoWindowGeometry.fpvScaleRange.lowerBound), windowScale - 0.05)
            }
            ConsoleButton(icon: "plus.magnifyingglass",
                          label: abs(windowScale - 1) < 0.005 ? "1st-person size" : "View \(Int((windowScale * 100).rounded()))%",
                          isOn: abs(windowScale - 1) < 0.005) {
                Haptics.select()
                windowScale = min(Double(StereoWindowGeometry.fpvScaleRange.upperBound), windowScale + 0.05)
            }
            // DOCK (operator, 2026-10-17: the dock + status bar follow the gaze like the
            // wrist cams). Follow = heading + pitch; Steady = the old lazy heading-only
            // follow, for when a control is easier to reach with the panel holding still.
            ConsoleButton(icon: deckMode == DeckFollower.Mode.follow.rawValue
                                ? "dot.viewfinder" : "pin.fill",
                          label: deckMode == DeckFollower.Mode.follow.rawValue
                                ? "Dock · Follow" : "Dock · Steady",
                          isOn: deckMode == DeckFollower.Mode.follow.rawValue) {
                Haptics.select()
                deckMode = deckMode == DeckFollower.Mode.follow.rawValue
                    ? DeckFollower.Mode.steady.rawValue : DeckFollower.Mode.follow.rawValue
            }

            ConsoleButton(icon: "visionpro", label: controller.fullImmersion ? "Window" : "Immersive",
                          isOn: controller.fullImmersion) {
                // DEFER ON A COLD FLIP, like every other style-change entry
                // point: flipping `.immersionStyle` re-creates the space, and a
                // restart without the deferral AUTO-ENGAGES mirroring — a
                // parked (not-mirroring) dome silently started driving the
                // robot on a Window↔Immersive flip. A live session keeps
                // mirroring (defer false), same contract as the window buttons.
                controller.deferBodyEngage = !controller.bodyTracking
                if controller.fullImmersion {
                    // Leaving the dome mid-session: keep mirroring, space hidden.
                    if !controller.leaveFullImmersionKeepingTeleop() {
                        controller.fullImmersion = false
                    }
                } else {
                    controller.fullImmersion = true
                }
            }

            ConsoleButton(icon: controller.estopEngaged ? "play.fill" : "exclamationmark.octagon.fill",
                          label: controller.estopEngaged ? "Resume" : "STOP",
                          isOn: controller.estopEngaged ? false : nil,
                          tint: Theme.Palette.danger) {
                if controller.estopEngaged { controller.releaseEStop() }
                else { controller.engageEStop() }
            }

            // EXIT never ends a running session (operator, 2026-10-16: "do not
            // stop teleop till I explicitly press"). Mirroring → leave the dome
            // and keep driving from the window (the space stays open, hidden,
            // for ARKit). Only Stop Teleop ends teleop.
            ConsoleButton(icon: "arrow.down.right.and.arrow.up.left",
                          label: controller.bodyTracking ? "Exit · keep teleop" : "Exit") {
                if !controller.leaveFullImmersionKeepingTeleop() {
                    controller.noteIntendedImmersiveExit()
                    Task { await dismissImmersive() }
                }
            }
        }
    }

    /// L/R dots: green = ARKit is tracking that hand right now.
    private var handDots: some View {
        HStack(spacing: 3) {
            ForEach(["L", "R"], id: \.self) { side in
                let tracked = side == "L" ? controller.leftHandTracked : controller.rightHandTracked
                Text(side)
                    .font(.system(size: 9, weight: .heavy))
                    .foregroundStyle(tracked ? .black : Theme.Palette.textTertiary)
                    .frame(width: 16, height: 16)
                    .background(tracked ? Theme.Palette.live : Theme.Palette.surface, in: Circle())
            }
        }
    }




    private var taskPicker: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("COLLECTION TASK")
                .font(.system(size: 11, weight: .heavy)).tracking(0.8)
                .foregroundStyle(.secondary)
            HStack(spacing: 8) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 14, weight: .semibold)).foregroundStyle(Theme.Palette.live)
                TextField("new task name", text: $newTaskName)
                    .font(.system(size: 14, weight: .medium))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onSubmit { commitNewTask() }
                Button("Use") { commitNewTask() }
                    .disabled(newTaskName.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            if !controller.serverTasks.isEmpty {
                ScrollView {
                    VStack(spacing: 2) {
                        ForEach(controller.serverTasks, id: \.self) { task in
                            Button {
                                controller.selectTask(task); showTaskPicker = false
                            } label: {
                                HStack(spacing: 8) {
                                    Image(systemName: task == controller.recordTask ? "checkmark.circle.fill" : "folder")
                                        .foregroundStyle(task == controller.recordTask ? Theme.Palette.live : .secondary)
                                    Text(task).font(.system(size: 14, weight: .medium))
                                    Spacer()
                                }
                                .contentShape(Rectangle())
                                .padding(.horizontal, 8).padding(.vertical, 7)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .frame(maxHeight: 180)
            }
        }
        .padding(16)
        .frame(width: 300)
    }

    private func commitNewTask() {
        let t = newTaskName.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        controller.selectTask(t)
        showTaskPicker = false
    }

    private var timeString: String { String(format: "%d:%02d", elapsed / 60, elapsed % 60) }
}
