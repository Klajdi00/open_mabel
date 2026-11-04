//
//  StagePlanner.swift
//  MabelTeleop
//
//  The MoveIt-style green-ball plan / preview / execute planner ON THE
//  REALITYKIT STAGE — the port of LiveMapView's SceneKit manipulation
//  Coordinator (operator directive 2026-07-18: Task space must stay the
//  TRUE-3D protruding viewer, like Drive and Joint — not the flat SceneKit
//  window the wrist takeover swapped in).
//
//  Same contract as the SceneKit path, byte-compatible where it matters:
//    · green palm ball under <side>_arm_7 at the palm centre; drag it and a
//      weighted-DLS IK reaches the arm (joints 1–4) toward the pinch point,
//      recruiting torso/lift (upper body) and the mobile base (whole body)
//      only once the arm saturates (proximal gate).
//    · roll/pitch/yaw rings around the ball twist wrist joints 5/6/7 — here
//      the drag's TRUE 3D rotation about the joint axis (the 2D screen-angle
//      trick was SceneKit's substitute for not having spatial input).
//    · releasing STAGES the target: the planned parts go ghost (0.45) and
//      loop an eased current→target preview. NOTHING reaches the robot until
//      Execute streams the SAME trajectory (ArmPlan is the shared pure core).
//
//  Coordinates: all IK runs in the RIG-ROOT frame — real robot metres,
//  independent of the stage's window fit (~1360 pt/m ⇒ root scale ≈ 0.5).
//  Scene-space pinch points convert in via rig.root.convert(position:from:).
//
//  The planner is TICK-driven (RobotStageView's per-frame pose() calls
//  tick(now:)), never CADisplayLink — so StagePlannerTests can step preview
//  and execute deterministically with synthetic timestamps.
//

import Foundation
import RealityKit
import UIKit
import simd

@MainActor
final class StagePlanner {

    // ── Wiring (RobotSpace3DStage) ────────────────────────────────────────────
    /// Staged-plan summary for the Execute pill (nil = no plan). Fired OUTSIDE
    /// any SwiftUI update pass (gesture callbacks / explicit calls only).
    var onPlanChange: ((ArmPlanPreview?) -> Void)?
    /// Execute's ONLY joint send — one batched config per streamed step.
    var onJointsChange: (([String: Double]) -> Void)?
    /// Whole-body base half on Execute: ONE odom nav goal (x, y, yaw).
    var onBaseGoal: ((Double, Double, Double) -> Void)?
    /// Live telemetry joints (raw qpos — the same dict the rig poses from).
    var joints: () -> [String: Double] = { [:] }
    /// Robot heading (odom) for the base goal's yaw.
    var heading: () -> Double = { 0 }
    /// Scene-space point → odom frame (SensorLayers.odomPoint(fromScene:)).
    var odomFromScene: ((SIMD3<Float>) -> SIMD3<Float>)?
    /// Freeze/unfreeze SensorLayers' odom repositioning of the robot while a
    /// base plan owns rig.robot.position (mirror of the SceneKit baseFrozen).
    var onBaseFreeze: ((Bool) -> Void)?

    var planRegion: ControlRegion = .arm

    // STRONG on purpose: the rig never references the planner (no cycle), and
    // `unowned` here abort-trapped the moment a caller outlived its rig
    // (swift_abortRetainUnowned in installIfNeeded — caught by
    // StagePlannerTests when a test dropped the rig binding).
    private let rig: RobotRigEntity
    private(set) var active = false
    private var installed = false
    private var handles: [String: Entity] = [:]          // side → ball

    init(rig: RobotRigEntity) { self.rig = rig }

    // ── Grab state ────────────────────────────────────────────────────────────
    private enum DragKind { case none, ball(side: String), ring(joint: String, side: String) }
    private var dragKind: DragKind = .none
    private var grabChain: [String] = []
    private var grabJoint = ""
    private var grabAngles: [String: Float] = [:]
    private var grabExtra: [String] = []
    private var grabBase = false
    private var grabBaseStart = SIMD2<Float>(0, 0)       // rig.robot root-frame XZ
    private var grabBaseNow = SIMD2<Float>(0, 0)
    private var ballRest: SIMD3<Float>? = nil            // dragged ball's rest local pos
    private var lastIKSolve: TimeInterval = 0
    // Ring drag: rotation about the joint's WORLD axis, plane captured at begin.
    private var ringAxisWorld = SIMD3<Float>(0, 1, 0)
    private var ringCentre = SIMD3<Float>(0, 0, 0)
    private var ringPrev = SIMD3<Float>(1, 0, 0)
    private var ringAngle: Float = 0

    // ── Plan state (accumulates across drags; both arms can stack) ───────────
    private var planStart: [String: Double] = [:]
    private var planTarget: [String: Double] = [:]
    private var planChain: [String] = []
    private var planSides: Set<String> = []
    private var pendingTarget: [String: Double] = [:]
    private var pendingChain: [String] = []
    private var pendingSide = "left"
    private var planBaseStart: SIMD2<Float>? = nil
    private var planBaseTarget: SIMD2<Float>? = nil
    private var pendingBaseTarget: SIMD2<Float>? = nil
    private var previewT0: TimeInterval = 0
    private var executeT0: TimeInterval? = nil
    /// Velocity-capped playback duration for the running Execute.
    private var executeDur: TimeInterval = 0
    var hasPlan: Bool { !planChain.isEmpty || planBaseTarget != nil }

    static let upperBodyDofs = ["torso", "lift_lower", "lift_upper"]
    static let proximalEngageDist: Float = 0.08

    // ── Activation / handles ─────────────────────────────────────────────────

    /// Arm/disarm the planner: build (lazily) + show the palm balls and rings.
    /// Deactivating cancels any staged plan — leaving Task space must never
    /// carry a plan into another controller.
    func setActive(_ on: Bool) {
        if on { installIfNeeded() }
        guard on != active else { return }
        active = on
        for (_, ball) in handles { ball.isEnabled = on }
        if !on {
            endDrag()
            cancelPlan()
        }
    }

    private func installIfNeeded() {
        guard !installed else { return }
        var built = false
        for side in ["left", "right"] {
            guard let mount = rig.joints["\(side)_arm_7"]?.node else { continue }
            let ball = Self.makeHandle()
            ball.name = "handle_\(side)"
            // Palm centre, not the wrist mount (0.255 m forward along local Y —
            // the SceneKit rig uses the identical offset).
            ball.position = RobotRig.handlePoint(side)          // palm, or the gripper TCP
            Self.addOrientationRings(to: ball, side: side)
            // The rig can finish loading AFTER setActive(true) (async PLY parse)
            // — a late install must come up in the CURRENT armed state.
            ball.isEnabled = active
            mount.addChild(ball)
            handles[side] = ball
            built = true
        }
        installed = built           // rig not loaded yet → retry on next setActive
    }

    /// The green control knob — the app's `live` green, PBR like the robot's
    /// own meshes (deliberately not an emissive blob).
    private static func makeHandle() -> Entity {
        var mat = PhysicallyBasedMaterial()
        let live = UIColor(red: 0x2E / 255.0, green: 0x9E / 255.0, blue: 0x52 / 255.0, alpha: 1)
        mat.baseColor = .init(tint: live)
        mat.roughness = .init(floatLiteral: 0.38)
        mat.metallic = .init(floatLiteral: 0.0)
        mat.emissiveColor = .init(color: live)
        mat.emissiveIntensity = 0.10
        let ball = ModelEntity(mesh: .generateSphere(radius: 0.06), materials: [mat])
        // Grab target slightly larger than the visual so the pinch lands.
        ball.components.set(CollisionComponent(shapes: [.generateSphere(radius: 0.09)]))
        ball.components.set(InputTargetComponent())
        ball.components.set(HoverEffectComponent())
        return ball
    }

    /// Roll/Pitch/Yaw rings around the ball — same radii/colours/naming as the
    /// SceneKit gizmo ("ring_<axis>_<side>"). Collision is a ring of small
    /// boxes around the annulus so the rings are pinchable WITHOUT covering
    /// the ball's own grab sphere in the middle.
    private static func addOrientationRings(to ball: Entity, side: String) {
        func ring(_ axisName: String, rgb: UInt32, orient: simd_quatf) -> Entity {
            let color = UIColor(red: CGFloat((rgb >> 16) & 0xFF) / 255,
                                green: CGFloat((rgb >> 8) & 0xFF) / 255,
                                blue: CGFloat(rgb & 0xFF) / 255, alpha: 0.85)
            let mesh = torusMesh(ringRadius: 0.10, pipeRadius: 0.009)
            let e = ModelEntity(mesh: mesh, materials: [UnlitMaterial(color: color)])
            e.name = "ring_\(axisName)_\(side)"
            e.orientation = orient
            var shapes: [ShapeResource] = []
            for k in 0..<12 {
                let a = Float(k) * (2 * .pi / 12)
                shapes.append(ShapeResource
                    .generateBox(size: SIMD3(0.055, 0.024, 0.03))
                    .offsetBy(rotation: simd_quatf(angle: -a, axis: SIMD3(0, 1, 0)),
                              translation: SIMD3(cos(a) * 0.10, 0, sin(a) * 0.10)))
            }
            e.components.set(CollisionComponent(shapes: shapes))
            e.components.set(InputTargetComponent())
            e.components.set(HoverEffectComponent())
            return e
        }
        // Torus lies in the X-Z plane (axis = Y), like SCNTorus.
        ball.addChild(ring("roll", rgb: 0x4F9DFF, orient: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0))))
        ball.addChild(ring("pitch", rgb: 0xE7913A, orient: simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))))
        ball.addChild(ring("yaw", rgb: 0xE0503D, orient: simd_quatf(angle: .pi / 2, axis: SIMD3(0, 0, 1))))
    }

    /// RealityKit has no torus primitive — a small parametric one (48×12).
    private static func torusMesh(ringRadius R: Float, pipeRadius r: Float) -> MeshResource {
        let nu = 48, nv = 12
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        for i in 0...nu {
            let u = Float(i) / Float(nu) * 2 * .pi
            let cu = cos(u), su = sin(u)
            for j in 0...nv {
                let v = Float(j) / Float(nv) * 2 * .pi
                let cv = cos(v), sv = sin(v)
                positions.append(SIMD3((R + r * cv) * cu, r * sv, (R + r * cv) * su))
                normals.append(SIMD3(cv * cu, sv, cv * su))
            }
        }
        let row = UInt32(nv + 1)
        for i in 0..<UInt32(nu) {
            for j in 0..<UInt32(nv) {
                let a = i * row + j, b = (i + 1) * row + j
                indices.append(contentsOf: [a, b, a + 1, b, b + 1, a + 1])
            }
        }
        var d = MeshDescriptor(name: "torus")
        d.positions = MeshBuffer(positions)
        d.normals = MeshBuffer(normals)
        d.primitives = .triangles(indices)
        return (try? MeshResource.generate(from: [d]))
            ?? .generateSphere(radius: R)      // never crash the stage over a gizmo
    }

    // ── Drag routing (RobotStageView's gesture) ──────────────────────────────

    /// TRUE if this entity (or an ancestor) is one of ours — the stage's link
    /// drag / orbit must stand aside for the duration of the drag.
    func claims(_ entity: Entity) -> Bool {
        guard active else { return false }
        return resolve(entity) != nil
    }

    private func resolve(_ entity: Entity) -> (kind: DragKind, entity: Entity)? {
        var e: Entity? = entity
        while let cur = e {
            if cur.name.hasPrefix("handle_") {
                return (.ball(side: String(cur.name.dropFirst("handle_".count))), cur)
            }
            if cur.name.hasPrefix("ring_") {
                let parts = cur.name.split(separator: "_")     // ring_<axis>_<side>
                if parts.count == 3 {
                    let side = String(parts[2])
                    let joint: String
                    switch parts[1] {
                    case "roll":  joint = "\(side)_arm_5"
                    case "pitch": joint = "\(side)_arm_6"
                    default:      joint = "\(side)_arm_7"
                    }
                    return (.ring(joint: joint, side: side), cur)
                }
            }
            e = cur.parent
        }
        return nil
    }

    func beginDrag(entity: Entity, scenePoint p: SIMD3<Float>) {
        guard active, let hit = resolve(entity) else { return }
        dragKind = hit.kind
        switch hit.kind {
        case let .ball(side):
            grabChain = (1...4).map { "\(side)_arm_\($0)" }
            grabJoint = "\(side)_hand"
            grabAngles = [:]
            let live = joints()
            for id in grabChain { grabAngles[id] = Float(live[id] ?? 0) }
            // Freeze the whole dragged arm so telemetry doesn't fight the local
            // IK prediction (the SceneKit rule, verbatim).
            var frozen = Set(grabChain)
            for i in 1...7 { frozen.insert("\(side)_arm_\(i)") }
            // Region DOFs: torso/lift for upper body, + the mobile base for
            // whole body — recruited by the IK only once the arm saturates.
            grabExtra = []
            grabBase = false
            switch planRegion {
            case .arm: break
            case .upperBody:
                grabExtra = Self.upperBodyDofs
            case .wholeBody:
                grabExtra = Self.upperBodyDofs
                grabBase = true
            }
            for id in grabExtra {
                grabAngles[id] = Float(live[id] ?? 0)
                frozen.insert(id)
            }
            if grabBase {
                // Once a base plan exists its start IS the true pre-plan pose
                // (the robot node is mid-preview now) — re-anchor to it.
                grabBaseStart = planBaseStart
                    ?? SIMD2(rig.robot.position.x, rig.robot.position.z)
                grabBaseNow = grabBaseStart
                onBaseFreeze?(true)
            }
            rig.frozenJoints.formUnion(frozen)
            if let ball = handles[side] { ballRest = ball.position }
            Haptics.soft()
        case let .ring(joint, side):
            guard let j = rig.joints[joint], let ball = handles[side] else {
                dragKind = .none; return
            }
            ringAngle = Float(joints()[joint] ?? 0)
            grabJoint = joint
            // Plane captured at drag START (world joint axis + ball centre):
            // rotating about a live axis would feed back into itself.
            let wt = j.node.transformMatrix(relativeTo: nil)
            let rot = simd_float3x3(simd_make_float3(wt.columns.0),
                                    simd_make_float3(wt.columns.1),
                                    simd_make_float3(wt.columns.2))
            ringAxisWorld = simd_normalize(rot * j.axis)
            ringCentre = ball.position(relativeTo: nil)
            let v0 = (p - ringCentre) - ringAxisWorld * simd_dot(p - ringCentre, ringAxisWorld)
            ringPrev = simd_length(v0) > 1e-5 ? simd_normalize(v0) : SIMD3(1, 0, 0)
            // Freeze the wrist + everything distal against telemetry.
            let sideJoints = (5...7).map { "\(side)_arm_\($0)" }
            rig.frozenJoints.formUnion(sideJoints)
            Haptics.soft()
        case .none:
            break
        }
    }

    func updateDrag(scenePoint p: SIMD3<Float>) {
        switch dragKind {
        case let .ball(side): stepBall(side: side, scenePoint: p)
        case let .ring(joint, _): stepRing(joint: joint, scenePoint: p)
        case .none: break
        }
    }

    func endDrag() {
        switch dragKind {
        case .ball(let side):
            // Snap the free-dragged ball back onto its palm seat.
            if let rest = ballRest { handles[side]?.position = rest }
            ballRest = nil
            if !pendingTarget.isEmpty {
                Haptics.tap()
                finalizePlan()
            } else {
                unfreezeUnplanned()
            }
        case .ring:
            if !pendingTarget.isEmpty {
                Haptics.tap()
                finalizePlan()
            } else {
                unfreezeUnplanned()
            }
        case .none:
            break
        }
        dragKind = .none
        grabChain = []
        grabExtra = []
        grabBase = false
    }

    /// A drag that never staged anything (micro-drag) releases its freezes so
    /// telemetry repaints — but never the joints an EXISTING plan still owns.
    private func unfreezeUnplanned() {
        let keep = Set(planChain)
        rig.frozenJoints = rig.frozenJoints.filter { keep.contains($0) }
        if planBaseTarget == nil { onBaseFreeze?(false) }
    }

    // ── Ball drag: weighted-DLS IK (port of the SceneKit solve) ──────────────

    private func stepBall(side: String, scenePoint: SIMD3<Float>) {
        guard let ball = handles[side] else { return }
        let now = CACurrentMediaTime()
        if now - lastIKSolve < 1.0 / 60.0 { return }
        lastIKSolve = now

        // Free ball follows the pinch — the arm reaches toward it.
        ball.setPosition(scenePoint, relativeTo: nil)
        // IK in the RIG-ROOT frame: real robot metres at any window fit.
        let target = rig.root.convert(position: scenePoint, from: nil)

        let dofs = buildIKDofs()
        guard !dofs.isEmpty, let rest = ballRest,
              let mount = rig.joints["\(side)_arm_7"]?.node else { return }
        let lambda: Float = 0.14
        for _ in 0..<4 {
            // End-effector = the ball's REST seat (palm centre), which follows
            // the wrist as the IK advances the chain.
            let ee = mount.convert(position: rest, to: rig.root)
            let err = target - ee
            let errLen = simd_length(err)
            if errLen < 0.0015 { break }
            // REGION GATE: proximal DOFs (torso/lift/base) join only while the
            // arm is still short by more than proximalEngageDist.
            let useProximal = errLen > Self.proximalEngageDist
            var cols = [SIMD3<Float>](repeating: .zero, count: dofs.count)
            var jjt = matrix_float3x3(diagonal: SIMD3<Float>(repeating: lambda * lambda))
            for (i, d) in dofs.enumerated() {
                if d.proximal && !useProximal { continue }
                let c = ikColumn(d, ee: ee)
                cols[i] = c
                let cw = c * sqrt(d.weight)
                jjt += matrix_float3x3(rows: [cw * cw.x, cw * cw.y, cw * cw.z])
            }
            let y = jjt.inverse * err
            for (i, d) in dofs.enumerated() {
                if d.proximal && !useProximal { continue }
                let dv = max(-d.iterClamp, min(d.iterClamp, d.weight * simd_dot(cols[i], y)))
                applyIKStep(d, delta: dv)
            }
        }

        // Stage ONE batched target: the arm as a unit; region DOFs only if the
        // solve actually moved them past a floor.
        var batch: [String: Double] = [:]
        for id in grabChain {
            var a = grabAngles[id] ?? 0
            if let s = RobotModel.spec(id) { a = max(Float(s.lower), min(Float(s.upper), a)) }
            batch[id] = Double(a)
        }
        let live = joints()
        for id in grabExtra {
            let a = Double(grabAngles[id] ?? 0)
            let floor = (id == "torso") ? 0.02 : 0.005
            if abs(a - (live[id] ?? 0)) > floor { batch[id] = ArmPlan.clamp(id, a) }
        }
        pendingTarget = batch
        pendingChain = Array(batch.keys)
        pendingSide = side
        if grabBase, simd_length(grabBaseNow - grabBaseStart) > 0.03 {
            pendingBaseTarget = grabBaseNow
        } else {
            pendingBaseTarget = nil
        }
    }

    private struct IKDof {
        enum Kind { case revolute, prismatic, baseX, baseZ }
        let kind: Kind
        let id: String
        let node: Entity?
        let axis: SIMD3<Float>
        let weight: Float
        let iterClamp: Float
        let lower: Float
        let upper: Float
        let proximal: Bool
    }

    private func buildIKDofs() -> [IKDof] {
        var dofs: [IKDof] = []
        for id in grabChain {
            guard let j = rig.joints[id] else { continue }
            let s = RobotModel.spec(id)
            dofs.append(IKDof(kind: .revolute, id: id, node: j.node, axis: j.axis,
                              weight: 1.0, iterClamp: 0.10,
                              lower: Float(s?.lower ?? -.pi), upper: Float(s?.upper ?? .pi),
                              proximal: false))
        }
        for id in grabExtra {
            guard let j = rig.joints[id] else { continue }
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

    /// Jacobian column in the RIG-ROOT frame: revolute → ω×r, prismatic → the
    /// world axis, base → the root-frame ground unit vector.
    private func ikColumn(_ d: IKDof, ee: SIMD3<Float>) -> SIMD3<Float> {
        switch d.kind {
        case .revolute:
            let wt = d.node!.transformMatrix(relativeTo: rig.root)
            let rot = simd_float3x3(simd_normalize(simd_make_float3(wt.columns.0)),
                                    simd_normalize(simd_make_float3(wt.columns.1)),
                                    simd_normalize(simd_make_float3(wt.columns.2)))
            return simd_cross(simd_normalize(rot * d.axis), ee - simd_make_float3(wt.columns.3))
        case .prismatic:
            let wt = d.node!.transformMatrix(relativeTo: rig.root)
            let rot = simd_float3x3(simd_normalize(simd_make_float3(wt.columns.0)),
                                    simd_normalize(simd_make_float3(wt.columns.1)),
                                    simd_normalize(simd_make_float3(wt.columns.2)))
            return simd_normalize(rot * d.axis)
        case .baseX: return SIMD3(1, 0, 0)
        case .baseZ: return SIMD3(0, 0, 1)
        }
    }

    private func applyIKStep(_ d: IKDof, delta: Float) {
        switch d.kind {
        case .revolute:
            let a = max(d.lower, min(d.upper, (grabAngles[d.id] ?? 0) + delta))
            grabAngles[d.id] = a
            // Plan joints (arm/torso) all carry MuJoCo ref = 0, but subtract it
            // anyway so this can never silently diverge from pose()'s maths.
            let ref = rig.joints[d.id]?.ref ?? 0
            d.node!.orientation = simd_quatf(angle: a - ref, axis: simd_normalize(d.axis))
        case .prismatic:
            let a = max(d.lower, min(d.upper, (grabAngles[d.id] ?? 0) + delta))
            grabAngles[d.id] = a
            let ref = rig.joints[d.id]?.ref ?? 0
            d.node!.position = d.axis * (a - ref)
        case .baseX:
            grabBaseNow.x += delta; applyBase()
        case .baseZ:
            grabBaseNow.y += delta; applyBase()
        }
    }

    /// Park the robot at the previewed base position (root-frame ground plane).
    /// SensorLayers' odom centering is frozen (onBaseFreeze) so it can't snap
    /// the robot back mid-preview.
    private func applyBase() {
        var pos = rig.robot.position
        pos.x = grabBaseNow.x
        pos.z = grabBaseNow.y
        rig.robot.position = pos
    }

    // ── Ring drag: wrist joints 5/6/7 about the TRUE joint axis ─────────────

    private func stepRing(joint: String, scenePoint p: SIMD3<Float>) {
        guard let j = rig.joints[joint] else { return }
        let v = (p - ringCentre) - ringAxisWorld * simd_dot(p - ringCentre, ringAxisWorld)
        guard simd_length(v) > 1e-4 else { return }
        let v1 = simd_normalize(v)
        // Signed angle prev→now about the captured axis; incremental so the
        // twist can wind past ±π toward the joint limits.
        let delta = atan2(simd_dot(simd_cross(ringPrev, v1), ringAxisWorld),
                          simd_dot(ringPrev, v1))
        ringPrev = v1
        var ang = ringAngle + delta
        if let s = RobotModel.spec(joint) { ang = max(Float(s.lower), min(Float(s.upper), ang)) }
        ringAngle = ang
        let ref = j.ref
        j.node.orientation = simd_quatf(angle: ang - ref, axis: simd_normalize(j.axis))
        pendingTarget = [joint: Double(ang)]
        pendingChain = [joint]
        pendingSide = joint.hasPrefix("left") ? "left" : "right"
    }

    // ── Plan / preview / execute (ArmPlan is the shared pure core) ───────────

    /// Headless staging for the debug harness + StagePlannerTests — identical
    /// to the SceneKit Coordinator's hook.
    func stagePlanForTest(side: String, deltas: [String: Double]) {
        let live = joints()
        pendingSide = side
        pendingChain = deltas.keys.sorted()
        for i in 1...7 { rig.frozenJoints.insert("\(side)_arm_\(i)") }
        pendingTarget = [:]
        for (id, d) in deltas { pendingTarget[id] = ArmPlan.clamp(id, (live[id] ?? 0) + d) }
        finalizePlan()
    }

    /// Merge the just-ended drag into the accumulated plan. Only a drag that
    /// MOVED the target past the micro-drag floor stages (arm ≥ 0.02 rad or
    /// base ≥ 0.03 m).
    private func finalizePlan() {
        let chain = pendingChain
        let target = pendingTarget
        let side = pendingSide
        let baseTarget = pendingBaseTarget
        pendingChain = []; pendingTarget = [:]; pendingBaseTarget = nil
        guard !chain.isEmpty || baseTarget != nil else { return }
        let live = joints()
        var start: [String: Double] = [:]
        for id in chain { start[id] = live[id] ?? 0 }
        let jointMove = ArmPlan.maxDelta(from: start, to: target, chain: chain)
        let baseMove = baseTarget.map { simd_length($0 - grabBaseStart) } ?? 0
        guard jointMove > 0.02 || baseMove > 0.03 else {
            unfreezeUnplanned()
            return
        }
        for id in chain {
            if !planChain.contains(id) { planChain.append(id) }
            planStart[id] = start[id]
            planTarget[id] = ArmPlan.clamp(id, target[id] ?? start[id] ?? 0)
            rig.frozenJoints.insert(id)
        }
        if !chain.isEmpty { planSides.insert(side) }
        if let bt = baseTarget {
            if planBaseStart == nil { planBaseStart = grabBaseStart }
            planBaseTarget = bt
            onBaseFreeze?(true)
        }
        for s in planSides { setArmGhost(s, true) }
        setPlanPartsGhost(true)
        previewT0 = 0
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

    /// Per-render-frame step (RobotStageView calls this right after pose()).
    /// Runs the looping ghost preview while a plan is staged, and the one-shot
    /// eased send while executing. Timestamp-injected so tests can drive it.
    func tick(now: TimeInterval) {
        if var t0 = executeT0 {
            if t0 < 0 {
                t0 = now; executeT0 = now                 // clock starts on the first tick
                // Size the playback to the MOTION, not to a constant — see
                // ArmPlan.duration. 1.4 s ran a 2.5 rad swing at ~2.7 rad/s peak,
                // ≈3× the velocity budget the rest of the stack respects.
                executeDur = ArmPlan.duration(from: planStart, to: planTarget,
                                              chain: planChain)
            }
            let t = min(1, (now - t0) / max(1e-3, executeDur))
            let cfg = ArmPlan.sample(from: planStart, to: planTarget, chain: planChain, t: t)
            applyConfig(cfg)
            applyBasePreview(t)
            if !cfg.isEmpty { onJointsChange?(cfg) }   // the ONLY joint send in the flow
            if t >= 1 {
                executeT0 = nil
                clearPlan(unfreeze: true)
                Haptics.tap()
            }
            return
        }
        guard hasPlan, active else { return }
        if previewT0 == 0 { previewT0 = now }
        let period = 2.8
        let phase = (now - previewT0).truncatingRemainder(dividingBy: period)
        let t: Double
        switch phase {
        case ..<1.4: t = phase / 1.4
        case ..<1.7: t = 1
        case ..<2.6: t = 1 - (phase - 1.7) / 0.9
        default:     t = 0
        }
        applyConfig(ArmPlan.sample(from: planStart, to: planTarget, chain: planChain, t: t))
        applyBasePreview(t)
    }

    /// Commit: parts go solid, the base half goes out ONCE as an odom nav
    /// goal, and the joint half streams from the next ticks.
    func executePlan() {
        guard hasPlan else { return }
        for s in planSides { setArmGhost(s, false) }
        setPlanPartsGhost(false)
        if let tg = planBaseTarget {
            // Root frame → scene → odom (the same conversion as the goal tap).
            let scenePt = rig.root.convert(position: SIMD3(tg.x, 0, tg.y), to: nil)
            let odom = odomFromScene?(scenePt) ?? scenePt
            onBaseGoal?(Double(odom.x), Double(-odom.z), heading())
        }
        Haptics.rigid()
        executeT0 = -1                          // armed; the next tick seeds the clock
    }

    func cancelPlan() {
        executeT0 = nil
        if hasPlan {
            for s in planSides { setArmGhost(s, false) }
            setPlanPartsGhost(false)
        }
        clearPlan(unfreeze: true)
    }

    private func clearPlan(unfreeze: Bool) {
        planChain = []; planTarget = [:]; planStart = [:]; planSides = []
        pendingChain = []; pendingTarget = [:]
        planBaseStart = nil; planBaseTarget = nil; pendingBaseTarget = nil
        previewT0 = 0
        if unfreeze {
            rig.frozenJoints = []
            onBaseFreeze?(false)
        }
        onPlanChange?(nil)
    }

    /// Apply a planned config to the rig (revolute rotate, prismatic slide) —
    /// the preview/execute pose path, same maths as pose() minus telemetry.
    private func applyConfig(_ cfg: [String: Double]) {
        for (id, v) in cfg {
            guard let j = rig.joints[id] else { continue }
            let value = Float(v) - j.ref
            if j.type == "prismatic" {
                j.node.position = j.axis * value
            } else {
                j.node.orientation = simd_quatf(angle: value, axis: simd_normalize(j.axis))
            }
        }
    }

    private func applyBasePreview(_ t: Double) {
        guard let s = planBaseStart, let tg = planBaseTarget else { return }
        grabBaseNow = s + (tg - s) * Float(ArmPlan.ease(t))
        applyBase()
    }

    // ── Ghosting (OpacityComponent — RealityKit's whole-subtree opacity) ─────

    private func setArmGhost(_ side: String, _ ghost: Bool) {
        guard let armRoot = rig.joints["\(side)_arm_1"]?.node else { return }
        if ghost {
            armRoot.components.set(OpacityComponent(opacity: 0.45))
        } else {
            armRoot.components.remove(OpacityComponent.self)
        }
    }

    /// Ghost a link's OWN meshes without dimming the subtree riding on it
    /// (the torso carries the arms and head).
    private func setShellGhost(_ node: Entity?, _ ghost: Bool) {
        guard let node else { return }
        for child in node.children where child is ModelEntity {
            if ghost { child.components.set(OpacityComponent(opacity: 0.45)) }
            else { child.components.remove(OpacityComponent.self) }
        }
    }

    private func setPlanPartsGhost(_ ghost: Bool) {
        if planChain.contains("torso") {
            setShellGhost(rig.joints["torso"]?.node, ghost)
        }
        if planChain.contains("lift_lower") || planChain.contains("lift_upper") {
            setShellGhost(rig.joints["lift_lower"]?.node, ghost)
            setShellGhost(rig.joints["lift_upper"]?.node, ghost)
        }
        if planBaseTarget != nil {
            setShellGhost(rig.robot.findEntity(named: "mabel_chassis"), ghost)
            setShellGhost(rig.robot.findEntity(named: "lift_base"), ghost)
        }
    }
}
