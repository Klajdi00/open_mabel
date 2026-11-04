//
//  RealRobotView.swift
//  MABEL Mobile App
//
//  Renders the real articulated MABEL and poses every joint from live state.
//  A manual orbit camera lets the operator inspect from any angle.
//
//  DIRECT MANIPULATION (gravityComp = true): grab ANY upper-body part — an arm
//  link, the torso, or the green wrist ball — and drag it. The grabbed point
//  follows your finger while damped-least-squares IK over the proximal joints
//  moves the whole arm NATURALLY (multi-joint), as if you applied a force. The
//  solved joint targets are held (gravity compensation), so the arm stays where
//  you leave it — it never springs back. The roll/pitch/yaw rings around the ball
//  set the wrist orientation. Orbit still works on empty space.
//

import SwiftUI
import SceneKit
import simd

struct RealRobotView: UIViewRepresentable {
    /// Decoupled live pose (see `LivePose`). The view reads it from a `CADisplayLink`
    /// render loop — the rAF equivalent the browser teleop uses — and applies it
    /// straight to the SceneKit nodes, so the model stays real-time regardless of how
    /// often (or seldom) SwiftUI re-evaluates this struct. NOT a per-frame SwiftUI
    /// binding, so a 50 Hz stream never invalidates the view tree.
    var pose: LivePose
    /// Enable direct manipulation: show the wrist balls + rings and let the operator
    /// grab-drag any link. Off → a passive viewer (orbit only), e.g. the cockpit.
    var gravityComp: Bool = false
    /// Bump to recenter the orbit camera to its default framing.
    var recenterToken: Int = 0
    /// Bump to FIT the whole robot to ~70% of the view (frames the bounding box).
    var fitToken: Int = 0
    /// Place the rig at the pose's PLANAR odom position as well as its heading.
    /// Off everywhere the base's world placement is owned by the scene around
    /// the robot (the live map); on for the choreography preview, where the
    /// viewer IS the world and a "drive forward" block has to look like driving
    /// forward rather than like nothing happening.
    var basePlanar: Bool = false
    /// The choreography preview's planned base path: draws the GLOBAL frame (grid + origin axes)
    /// and the path itself, so how far the robot moved and turned is visible (TimelineWorldFrame).
    var plannedPath: PreviewPath? = nil
    /// The robot's measured wheel odometry in the same world frame (LiveOdomTrack): when set, the viewer
    /// draws the actual-path trail and places the planned path at `track.planAnchor`.
    var odomTrack: LiveOdomTrack? = nil
    /// `pose` is the LIVE robot (its base x/y/heading are gateway odom → world through `odomTrack`), not a
    /// plan-frame preview (placed at `odomTrack.planAnchor`).
    var poseIsOdom: Bool = false
    /// While the player is idle the planned path starts wherever the live robot is.
    var anchorPlanToLive: Bool = true
    /// Push a joint's solved angle (grab IK + the wrist rings).
    var onJointChange: (String, Double) -> Void = { _, _ in }
    /// Batched variant for the IK chain — one command for the whole arm per solve.
    var onJointsChange: ([String: Double]) -> Void = { _ in }
    /// SOFT (compliance) mode: a grab is an external PUSH, not IK. The grabbed
    /// part is reported by joint id; the force is a camera-relative world vector.
    var forceMode: Bool = false
    var onPartForce: (_ joint: String, _ worldForce: [Double]) -> Void = { _, _ in }

    /// Links the operator may grab-drag: the 7-DOF arms + the torso.
    static let grabbableJoints: Set<String> = {
        var s: Set<String> = ["torso"]
        for side in ["left", "right"] { for i in 1...7 { s.insert("\(side)_arm_\(i)") } }
        return s
    }()

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        let (scene, rig) = RealRobotScene.make()
        v.scene = scene
        v.backgroundColor = .clear
        v.antialiasingMode = .multisampling4X
        v.allowsCameraControl = false
        v.pointOfView = rig.camera
        // SceneKit publishes EVERY node as an accessibility element, so the rig
        // (about two hundred baked mesh parts) floods the tree: VoiceOver reads
        // out "mabel_chassis__v0" and a UI test's snapshot of the screen takes
        // long enough that a sheet presented over it times out looking for its
        // own buttons. The robot is a picture; it is described by the chrome
        // around it, not by its geometry.
        v.isAccessibilityElement = false
        v.accessibilityElementsHidden = true
        context.coordinator.rig = rig
        context.coordinator.view = v

        v.addGestureRecognizer(UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.onPan(_:))))
        v.addGestureRecognizer(UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.onPinch(_:))))

        context.coordinator.apply(pose.joints, heading: pose.heading, baseQuat: pose.baseQuat, gravityComp: gravityComp)
        // rAF-style render loop: apply the newest pose each display frame, decoupled
        // from SwiftUI. Self-throttling — if a frame's apply runs long the link just
        // fires less often; it never queues a backlog (the cause of the RTT ramp).
        let link = CADisplayLink(target: context.coordinator, selector: #selector(Coordinator.renderTick))
        // Render at the display refresh (≤120 Hz ProMotion), like the web console's
        // requestAnimationFrame — so the model mirrors the robot's CURRENT streamed pose
        // with no render-cadence lag. renderTick is a cheap no-op when the pose hasn't
        // changed (it gates on LivePose.tick), so a high refresh costs ~nothing while
        // telemetry is static; the old 30 fps cap added up to ~33 ms of avoidable lag.
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 120, preferred: 120)
        link.add(to: .main, forMode: .common)
        context.coordinator.displayLink = link
        #if DEBUG
        context.coordinator.runIKSelfTestOnce()
        #endif
        return v
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        context.coordinator.parent = self
        // Structural props (gravityComp handles, recenter/fit) change rarely and may
        // arrive while telemetry is static — force the next render tick to re-apply.
        context.coordinator.forceReapply()
        context.coordinator.syncRecenter(recenterToken)
        context.coordinator.syncFit(fitToken)
    }

    static func dismantleUIView(_ uiView: SCNView, coordinator: Coordinator) {
        coordinator.displayLink?.invalidate()
        coordinator.displayLink = nil
    }

    final class Coordinator: NSObject {
        var parent: RealRobotView
        var rig: RealRig?
        weak var view: SCNView?
        var displayLink: CADisplayLink?
        private var lastAppliedTick: UInt64 = .max   // force first apply
        private var lastPan: CGPoint = .zero
        private var lastRecenter = 0
        private var lastFit = 0

        /// Render loop: apply the newest `LivePose` only when it changed (or when a
        /// structural prop forced a re-apply). Runs on the main thread at the display
        /// refresh; cheap no-op when the pose is unchanged.
        @objc func renderTick() {
            // VP tracking overlay refreshes every display frame (it hides
            // itself when the operator_points stream goes stale) — NOT gated
            // on the pose tick, which only advances with robot telemetry.
            rig?.opOverlay?.refresh()
            let p = parent.pose
            guard p.tick != lastAppliedTick else { return }
            lastAppliedTick = p.tick
            guard parent.basePlanar else {
                apply(p.joints, heading: p.heading, baseQuat: p.baseQuat, gravityComp: parent.gravityComp)
                syncPlannedPath()
                return
            }
            // the world pose: live odom re-zeroed by the track, or a plan-frame sample at the plan anchor
            var w = BasePose(dx: p.baseX, dy: p.baseY, yaw: p.heading)
            if let track = parent.odomTrack {
                w = parent.poseIsOdom
                    ? track.ingest(x: p.baseX, y: p.baseY, yaw: p.heading, followPlan: parent.anchorPlanToLive)
                    : LiveOdomTrack.compose(track.planAnchor, w)
            }
            // NOVA streams no chassis tilt; the live quat is level, so heading alone orients the rig.
            // Drop `base_heading`: every pose carries it (homePose zero-fills all joints) and it OUTRANKS the
            // heading in apply(), which pinned the rig at 0° — the viewer could never show a turn.
            var joints = p.joints; joints["base_heading"] = nil
            apply(joints, heading: w.yaw, baseQuat: [1, 0, 0, 0], gravityComp: parent.gravityComp)
            // the start body frame (forward, left) → the rig, through the ONE mapping the
            // world frame and the drawn path use too (TimelineWorldFrame.scene)
            rig?.root.position = TimelineWorldFrame.scene(w.dx, w.dy, lift: 0.078)
            syncPlannedPath()
            syncOdomTrail()
        }

        private var trailNode: SCNNode?
        private var trailVersion = -1
        private var trailBuiltAt: CFTimeInterval = 0
        private var anchorVersion = -1

        /// The measured trail, rebuilt at most 10×/s (a 50 Hz odom stream would otherwise rebuild
        /// a few-thousand-vertex strip every frame), and the plan placed at its anchor.
        func syncOdomTrail() {
            guard let track = parent.odomTrack, let root = view?.scene?.rootNode else { return }
            if track.anchorVersion != anchorVersion, let pathNode {
                anchorVersion = track.anchorVersion
                TimelineWorldFrame.place(pathNode, at: track.planAnchor)
            }
            let now = CACurrentMediaTime()
            guard track.trailVersion != trailVersion, now - trailBuiltAt > 0.1 else { return }
            trailVersion = track.trailVersion; trailBuiltAt = now
            trailNode?.removeFromParentNode()
            let n = TimelineWorldFrame.makeTrail(track.trail)
            root.addChildNode(n); trailNode = n
        }

        private var worldFrame: SCNNode?
        private var pathNode: SCNNode?
        private var pathVersion = -1

        /// Grid + axes once, the path whenever the timeline changes it (a version bump).
        func syncPlannedPath() {
            guard let path = parent.plannedPath, let root = view?.scene?.rootNode else { return }
            if worldFrame == nil {
                let f = TimelineWorldFrame.makeFrame()
                root.addChildNode(f); worldFrame = f
                if let scene = view?.scene, let rig { TimelineWorldFrame.adoptRobotTabStage(scene: scene, robot: rig.root) }
            }
            guard path.version != pathVersion else { return }
            pathVersion = path.version
            pathNode?.removeFromParentNode()
            let n = TimelineWorldFrame.makePath(path.points)
            if let track = parent.odomTrack { TimelineWorldFrame.place(n, at: track.planAnchor) }
            root.addChildNode(n); pathNode = n
        }

        /// Make the next `renderTick` re-apply even if the pose tick is unchanged
        /// (e.g. a gravityComp/handle toggle while telemetry is paused).
        func forceReapply() { lastAppliedTick = .max }

        // Drag dispatch (decided at .began).
        private enum DragKind { case orbit, ring(joint: String), grab }
        private var dragKind: DragKind = .orbit

        // Grab-IK state: the chain we solve, the node the grab point is fixed to,
        // that point in the node's local frame, the screen depth to hold, and the
        // live joint angles we're driving.
        private var grabChain: [String] = []
        private var grabNode: SCNNode?
        private var grabLocal: simd_float3 = .zero
        private var dragDepth: Float = 0.5
        private var grabAngles: [String: Float] = [:]
        // SOFT-mode push: which part is grabbed + the finger location at grab start.
        private var grabJoint: String = ""
        private var grabStartScreen: CGPoint = .zero

        // Ring drag state.
        private var ringLastAngle: CGFloat = 0
        private var ringDragAngle: Float = 0

        // IK solve throttle (cap the per-gesture DLS to ~60 Hz so it can't flood main).
        private var lastIKSolve: TimeInterval = 0

        // Selection highlight: the node currently lit as "grabbed" + its originals.
        private weak var hlNode: SCNNode?
        private var hlScale = SCNVector3(1, 1, 1)
        private var hlEmission: Any?

        init(_ parent: RealRobotView) { self.parent = parent }

        /// Light a node as SELECTED — bright white emission + a scale-up pulse — so it's
        /// unmistakable which handle/ring the operator grabbed. `nil` clears the current.
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

        func apply(_ j: [String: Double], heading: Double, baseQuat: [Double], gravityComp: Bool) {
            guard let rig else { return }
            // Snap to the new pose — no implicit easing — so a reset (home jump) or a
            // fresh telemetry frame appears immediately instead of lazily animating.
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0; defer { SCNTransaction.commit() }
            let isLevel = baseQuat.count == 4 && baseQuat[0] > 0.9999
                && abs(baseQuat[1]) < 1e-4 && abs(baseQuat[2]) < 1e-4 && abs(baseQuat[3]) < 1e-4
            if !isLevel, baseQuat.count == 4 {
                let qm = simd_quatf(ix: Float(baseQuat[1]), iy: Float(baseQuat[2]),
                                    iz: Float(baseQuat[3]), r: Float(baseQuat[0]))
                let c = simd_quatf(angle: -.pi / 2, axis: simd_float3(1, 0, 0))
                rig.root.simdOrientation = c * qm * c.inverse
            } else {
                rig.root.simdOrientation = simd_quatf(angle: Float(j["base_heading"] ?? heading),
                                                      axis: simd_float3(0, 1, 0))
            }
            // Telemetry mirror (RViz) — EXCEPT the joints under an active drag, which
            // show the local IK prediction (set by stepGrab/stepRing) for immediate,
            // lag-free finger feedback. Freeze the whole dragged arm so telemetry doesn't
            // fight the prediction mid-drag. Not dragging → grabChain is empty → pure
            // telemetry. On release the freeze clears and the real-time robot has caught
            // up, so there's no snap.
            var frozen = Set(grabChain)
            var seed = grabChain.first
            if case .ring(let rj) = dragKind { frozen.insert(rj); if seed == nil { seed = rj } }
            if let s = seed, s.contains("arm"), let side = s.split(separator: "_").first {
                for i in 1...7 { frozen.insert("\(side)_arm_\(i)") }
            }
            for (id, joint) in rig.joints where !frozen.contains(id) {
                // Missing joint -> its ref (qpos0) = the assembled pose, not 0. With the
                // lift assembled half-raised a 0 fallback drops it 31.75 cm to the floor.
                let value = Float(RobotRig.jointValue(id, mimic: joint.mimic, in: j) ?? Double(joint.ref))
                if joint.type == "prismatic" {
                    joint.node.simdPosition = joint.axis * (value - joint.ref)
                } else {
                    joint.node.simdOrientation = simd_quatf(angle: value - joint.ref,
                                                            axis: simd_normalize(joint.axis))
                }
            }
            rig.handles["left"]?.isHidden = !gravityComp
            rig.handles["right"]?.isHidden = !gravityComp
        }

        func syncRecenter(_ token: Int) {
            guard token != lastRecenter, let rig else { return }
            lastRecenter = token
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0.35
            rig.orbit.eulerAngles = SCNVector3(-0.16, 0.5 + .pi, 0)
            rig.orbit.position = SCNVector3(0, 0.42, 0)
            rig.camera.position = SCNVector3(0, 0, rig.distance)
            SCNTransaction.commit()
        }

        func syncFit(_ token: Int) {
            guard token != lastFit, let rig, let view else { return }
            lastFit = token
            frameToFit(rig: rig, view: view, fraction: 0.70)
        }

        /// FIT the whole robot to ~`fraction` of the smaller screen dimension.
        /// Keeps the current viewing angle: it only re-centres the orbit pivot on
        /// the model's world bounding box and zooms the camera so the box fills the
        /// target fraction. Uses the screen-projected box bounds (no FOV / aspect
        /// assumptions) and snaps so the projection math always reads the final pose.
        private func frameToFit(rig: RealRig, view: SCNView, fraction: CGFloat) {
            var lo = simd_float3(repeating: .greatestFiniteMagnitude)
            var hi = simd_float3(repeating: -.greatestFiniteMagnitude)
            rig.root.enumerateHierarchy { node, _ in
                let nm = node.name ?? ""
                guard node.geometry != nil, !node.isHidden,
                      nm != "handle", !nm.hasPrefix("ring_") else { return }
                let (a, b) = node.boundingBox
                guard a.x <= b.x else { return }
                for sx in [a.x, b.x] { for sy in [a.y, b.y] { for sz in [a.z, b.z] {
                    let w = node.simdConvertPosition(simd_float3(sx, sy, sz), to: nil)
                    lo = simd_min(lo, w); hi = simd_max(hi, w)
                } } }
            }
            guard lo.x <= hi.x else { return }
            let center = (lo + hi) / 2
            var corners: [simd_float3] = []
            for sx in [lo.x, hi.x] { for sy in [lo.y, hi.y] { for sz in [lo.z, hi.z] {
                corners.append(simd_float3(sx, sy, sz))
            } } }

            let bw = max(1, view.bounds.width), bh = max(1, view.bounds.height)
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0
            rig.orbit.simdPosition = center
            for _ in 0..<6 {
                var minX = CGFloat.greatestFiniteMagnitude, minY = CGFloat.greatestFiniteMagnitude
                var maxX = -CGFloat.greatestFiniteMagnitude, maxY = -CGFloat.greatestFiniteMagnitude
                for c in corners {
                    let s = view.projectPoint(SCNVector3(c))
                    minX = min(minX, CGFloat(s.x)); maxX = max(maxX, CGFloat(s.x))
                    minY = min(minY, CGFloat(s.y)); maxY = max(maxY, CGFloat(s.y))
                }
                let frac = max((maxX - minX) / bw, (maxY - minY) / bh)
                guard frac.isFinite, frac > 0.0001 else { break }
                rig.camera.position.z = max(0.9, min(8, rig.camera.position.z * Float(frac / fraction)))
            }
            SCNTransaction.commit()
        }

        // MARK: - Gestures

        @objc func onPan(_ g: UIPanGestureRecognizer) {
            guard let rig, let view else { return }
            let p = g.location(in: view)
            switch g.state {
            case .began:
                lastPan = g.translation(in: view)
                if parent.gravityComp, let (joint, side) = ringHit(at: p, view: view) {
                    dragKind = .ring(joint: joint)
                    ringDragAngle = Float(parent.pose.joints[joint] ?? 0)
                    if let ball = rig.handles[side] {
                        ringLastAngle = angleAround(center: view.projectPoint(SCNVector3(ball.simdWorldPosition)), p: p)
                    }
                    // Light the grabbed ring so the operator SEES which axis is selected.
                    let axis = joint.hasSuffix("_5") ? "roll" : (joint.hasSuffix("_6") ? "pitch" : "yaw")
                    setSelected(view.scene?.rootNode.childNode(withName: "ring_\(axis)_\(side)", recursively: true))
                } else if parent.gravityComp, beginGrab(at: p, view: view, rig: rig) {
                    dragKind = .grab
                    let side = grabJoint.hasPrefix("left") ? "left" : "right"
                    setSelected(rig.handles[side])      // light the green ball = SELECTED
                } else {
                    dragKind = .orbit
                }
            case .changed:
                switch dragKind {
                case .ring(let joint): stepRing(joint: joint, at: p, view: view, rig: rig)
                case .grab: stepGrab(finger: p, view: view)
                case .orbit:
                    let t = g.translation(in: view)
                    let dx = Float(t.x - lastPan.x), dy = Float(t.y - lastPan.y)
                    rig.orbit.eulerAngles.y -= dx * 0.0085
                    rig.orbit.eulerAngles.x = max(-1.35, min(1.2, rig.orbit.eulerAngles.x - dy * 0.0085))
                    lastPan = t
                }
            case .ended, .cancelled:
                if case .grab = dragKind {
                    Haptics.tap()
                    if parent.forceMode { parent.onPartForce(grabJoint, [0, 0, 0]) }  // release the push
                }
                clearSelected()                  // un-highlight the ball / ring
                grabChain = []; grabNode = nil
                dragKind = .orbit
            default: break
            }
        }

        @objc func onPinch(_ g: UIPinchGestureRecognizer) {
            guard let rig else { return }
            if g.state == .changed {
                rig.camera.position.z = max(0.9, min(6, rig.camera.position.z / Float(g.scale)))
                g.scale = 1
            }
        }

        // MARK: - Grab IK (drag any part → natural multi-joint move, then hold)

        /// Resolve what was grabbed → the IK chain + the fixed grab point. Returns
        /// false if nothing grabbable is under the finger (→ orbit).
        private func beginGrab(at p: CGPoint, view: SCNView, rig: RealRig) -> Bool {
            let hits = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.all.rawValue])
            // The green ball = grab the wrist (shoulder+elbow reach to it).
            for h in hits where h.node.name == "handle" {
                let side = (h.node === rig.handles["left"]) ? "left" : "right"
                return startGrab(node: h.node, world: simd_float3(h.node.simdWorldPosition),
                                 chain: (1...4).map { "\(side)_arm_\($0)" }, joint: "\(side)_hand",
                                 at: p, view: view)
            }
            // Otherwise walk up from the closest mesh to a grabbable link.
            guard let closest = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.closest.rawValue]).first
            else { return false }
            var node: SCNNode? = closest.node
            while let n = node {
                if let name = n.name, RealRobotView.grabbableJoints.contains(name), let art = rig.joints[name]?.node {
                    let chain: [String]
                    if name == "torso" {
                        chain = ["torso"]
                    } else {
                        let side = name.hasPrefix("left") ? "left" : "right"
                        let k = Int(name.split(separator: "_").last ?? "1") ?? 1
                        chain = (1...max(1, min(k, 4))).map { "\(side)_arm_\($0)" }   // shoulder+elbow reach
                    }
                    return startGrab(node: art, world: simd_float3(closest.worldCoordinates), chain: chain,
                                     joint: name, at: p, view: view)
                }
                node = n.parent
            }
            return false
        }

        private func startGrab(node: SCNNode, world: simd_float3, chain: [String], joint: String,
                               at p: CGPoint, view: SCNView) -> Bool {
            guard !chain.isEmpty else { return false }
            grabChain = chain
            grabNode = node
            grabJoint = joint
            grabStartScreen = p
            grabLocal = node.simdConvertPosition(world, from: nil)
            dragDepth = view.projectPoint(SCNVector3(world)).z
            grabAngles = [:]
            for id in chain { grabAngles[id] = Float(parent.pose.joints[id] ?? 0) }
            Haptics.soft()
            return true
        }

        private func stepGrab(finger p: CGPoint, view: SCNView) {
            guard let rig, let node = grabNode, !grabChain.isEmpty else { return }

            // SOFT (compliance): the drag is an external PUSH on the grabbed body,
            // proportional to the screen displacement, in camera-relative world axes.
            // The bridge's gravity-comp compliance controller admits it (back-drivable).
            if parent.forceMode {
                let cam = rig.camera.simdWorldTransform
                let right = simd_normalize(simd_make_float3(cam.columns.0))
                let up    = simd_normalize(simd_make_float3(cam.columns.1))
                let k: Float = 2.0     // N per screen-point of drag
                let f = right * Float(p.x - grabStartScreen.x) * k
                      - up    * Float(p.y - grabStartScreen.y) * k   // screen y is down
                parent.onPartForce(grabJoint, [Double(f.x), Double(f.y), Double(f.z)])
                return
            }

            // Throttle the solve to ~60 Hz. The pan gesture fires up to ~120 Hz; running
            // an 8-iteration DLS IK on EVERY event floods the main thread it shares with
            // the WebSocket receive/pong path and the video display → the latency spike
            // (40→400 ms) and the stuttery video. 60 Hz is silky for a drag and halves
            // the load.
            let now = CACurrentMediaTime()
            if now - lastIKSolve < 1.0 / 60.0 { return }
            lastIKSolve = now

            let target = simd_float3(view.unprojectPoint(SCNVector3(Float(p.x), Float(p.y), dragDepth)))
            let chain = grabChain.compactMap { id in rig.joints[id].map { (id, $0) } }
            guard !chain.isEmpty else { return }

            // Local damped-least-squares IK → the dragged arm tracks the finger.
            // STABILITY: matches the web console's gentle, well-damped convergence.
            // The previous solve (8 iterations, λ=0.06, ±0.25 rad/iter ⇒ up to ~110°
            // per frame) overshot and oscillated ("back and forth"). We now take a few
            // heavily-damped, small-clamped steps per frame so the arm eases onto the
            // target over successive frames — rock-steady once the finger stops, like
            // the web (where the target pose is set once and the model holds it).
            for _ in 0..<4 {
                let ee = node.simdConvertPosition(grabLocal, to: nil)
                let err = target - ee
                if simd_length(err) < 0.0015 { break }
                var cols: [simd_float3] = []
                cols.reserveCapacity(chain.count)
                for (_, joint) in chain {
                    let wt = joint.node.simdWorldTransform
                    let rot = simd_float3x3(simd_make_float3(wt.columns.0), simd_make_float3(wt.columns.1), simd_make_float3(wt.columns.2))
                    cols.append(simd_cross(simd_normalize(rot * joint.axis), ee - simd_make_float3(wt.columns.3)))
                }
                let lambda: Float = 0.14                  // more damping → no overshoot
                var jjt = matrix_float3x3(diagonal: SIMD3<Float>(repeating: lambda * lambda))
                for c in cols { jjt += matrix_float3x3(rows: [c * c.x, c * c.y, c * c.z]) }
                let y = jjt.inverse * err
                for (i, (id, joint)) in chain.enumerated() {
                    let dTheta = max(-0.10, min(0.10, simd_dot(cols[i], y)))   // gentle per-step clamp
                    let a = (grabAngles[id] ?? 0) + dTheta
                    grabAngles[id] = a
                    joint.node.simdOrientation = simd_quatf(angle: a, axis: simd_normalize(joint.axis))
                }
            }
            // ONE batched command for the whole arm chain — never one send/invalidation
            // per joint (that was the main-thread storm behind the latency spike).
            var batch: [String: Double] = [:]
            for (id, _) in chain {
                var a = grabAngles[id] ?? 0
                if let s = RobotModel.spec(id) { a = max(Float(s.lower), min(Float(s.upper), a)) }
                batch[id] = Double(a)
            }
            parent.onJointsChange(batch)
        }

        // MARK: - Orientation-ring drag (wrist roll/pitch/yaw)

        private func ringHit(at p: CGPoint, view: SCNView) -> (joint: String, side: String)? {
            let hits = view.hitTest(p, options: [SCNHitTestOption.searchMode: SCNHitTestSearchMode.all.rawValue])
            for h in hits {
                guard let name = h.node.name, name.hasPrefix("ring_") else { continue }
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

        private func stepRing(joint: String, at p: CGPoint, view: SCNView, rig: RealRig) {
            let side = joint.hasPrefix("left") ? "left" : "right"
            guard let ball = rig.handles[side], let j = rig.joints[joint] else { return }
            let center = view.projectPoint(SCNVector3(ball.simdWorldPosition))
            let a = angleAround(center: center, p: p)
            var delta = a - ringLastAngle
            if delta > .pi { delta -= 2 * .pi }; if delta < -.pi { delta += 2 * .pi }
            ringLastAngle = a
            var ang = ringDragAngle + Float(delta)
            if let s = RobotModel.spec(joint) { ang = max(Float(s.lower), min(Float(s.upper), ang)) }
            ringDragAngle = ang
            // Predict the wrist locally (immediate feedback; the joint is frozen from
            // telemetry by apply() during the drag) AND send the command.
            j.node.simdOrientation = simd_quatf(angle: ang, axis: simd_normalize(j.axis))
            parent.onJointChange(joint, Double(ang))
        }

        #if DEBUG
        private static var didSelfTest = false
        /// Rigor check: from rest, command world targets and verify the grab IK
        /// lands the wrist within tolerance. Prints residuals once.
        func runIKSelfTestOnce() {
            guard !Self.didSelfTest, let rig else { return }
            Self.didSelfTest = true
            let side = "right"
            guard let ball = rig.handles[side] else { return }
            let chain = (1...4).map { "\(side)_arm_\($0)" }.compactMap { id in rig.joints[id].map { (id, $0) } }
            guard !chain.isEmpty else { return }
            let rest = ball.simdWorldPosition
            let offsets: [simd_float3] = [.init(0.06,0,0), .init(-0.06,0.05,0), .init(0,-0.08,0.04), .init(0.04,0.04,-0.04)]
            var saved: [String: Float] = [:]
            for (id, _) in chain { saved[id] = Float(parent.pose.joints[id] ?? 0) }
            NSLog("[MABEL-IK] grab self-test (%@ arm, %d DOF)", side, chain.count)
            for off in offsets {
                for (id, _) in chain { grabAngles[id] = saved[id] }
                let target = rest + off
                for _ in 0..<40 {
                    let ee = ball.simdWorldPosition
                    let err = target - ee
                    if simd_length(err) < 0.0015 { break }
                    var cols: [simd_float3] = []
                    for (_, joint) in chain {
                        let wt = joint.node.simdWorldTransform
                        let rot = simd_float3x3(simd_make_float3(wt.columns.0), simd_make_float3(wt.columns.1), simd_make_float3(wt.columns.2))
                        cols.append(simd_cross(simd_normalize(rot * joint.axis), ee - simd_make_float3(wt.columns.3)))
                    }
                    var jjt = matrix_float3x3(diagonal: SIMD3<Float>(repeating: 0.0036))
                    for c in cols { jjt += matrix_float3x3(rows: [c * c.x, c * c.y, c * c.z]) }
                    let y = jjt.inverse * err
                    for (i, (id, joint)) in chain.enumerated() {
                        let a = (grabAngles[id] ?? 0) + max(-0.25, min(0.25, simd_dot(cols[i], y)))
                        grabAngles[id] = a
                        joint.node.simdOrientation = simd_quatf(angle: a, axis: simd_normalize(joint.axis))
                    }
                }
                NSLog("[MABEL-IK]   target Δ=(%.0f,%.0f,%.0f)mm → residual %.2f mm",
                      off.x*1000, off.y*1000, off.z*1000, simd_length(target - ball.simdWorldPosition) * 1000)
            }
            for (id, joint) in chain { joint.node.simdOrientation = simd_quatf(angle: saved[id] ?? 0, axis: simd_normalize(joint.axis)) }
        }
        #endif
    }
}
