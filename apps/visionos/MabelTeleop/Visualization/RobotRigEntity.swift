//
//  RobotRigEntity.swift
//  MabelTeleop
//
//  Loads the REAL MABEL — the same rig the iOS app, website and MuJoCo sim
//  use (Resources/MabelRig: rig_<node>_<colour>.ply + rig.json with the
//  kinematic hierarchy) — as an articulated RealityKit entity. Every joint
//  rotates/slides about its own local axis, exactly like the iOS SceneKit
//  viewer, so all clients render the identical robot.
//
//  Interaction: every ARM link is a first-class input target — look at it
//  and it highlights (system gaze hover effect); pinch-drag it to guide
//  (Hold) or push (Soft) that part of the body. A force arrow visualizes
//  the applied push: position, direction and magnitude.
//

import Foundation
import RealityKit
import SwiftUI
import simd

struct RigJoint {
    let node: Entity
    let type: String                 // "revolute" | "prismatic"
    var axis: SIMD3<Float> = SIMD3(0, 0, 1)
    /// MuJoCo qpos0 — geometry is baked at qpos == ref; apply (value − ref).
    var ref: Float = 0
    /// Leader joint of a MIMIC joint (the OpenArm gripper's second finger): drawn at the leader's
    /// value, multiplier 1 — see `RobotRig.jointValue(_:mimic:in:)`.
    var mimic: String? = nil
}

/// What a draggable link resolves to.
struct RigLink {
    let side: String                 // "left" | "right"
    let pushBody: String             // MuJoCo body name for external_force
}

/// An articulated MABEL. `root` goes into the scene; `pose(...)` drives it.
@MainActor
final class RobotRigEntity {

    /// Yaw/scale pivot the caller owns (gestures, life-size etc.).
    let root = Entity()
    /// The robot itself (base orientation applied here).
    let robot = Entity()
    private(set) var joints: [String: RigJoint] = [:]
    /// Every PLY part model, in FK order — the robot's actual GEOMETRY.
    ///
    /// `robot`'s subtree is not only the robot: the force/guide arrows hang off
    /// it, `SensorLayers` parents the camera-projection planes to the mount
    /// entities and the TF triads to the joint nodes, and the laser ring rides
    /// the base. `visualBounds` counts all of them (and, by default, DISABLED
    /// ones too), so it is not a measurement of the machine. Anything that
    /// needs the robot's real extent — the virtual floor's wheel-contact
    /// height, a module's framing — measures over THIS list instead.
    private(set) var parts: [ModelEntity] = []
    /// link-entity name → what dragging it means. Keys are the `link:`-prefixed
    /// entity names installed on each interactive part.
    private var links: [String: RigLink] = [:]
    /// Force visualization (position + direction + magnitude of the push).
    private let forceArrow = ForceArrow(color: UIColor(red: 0.2, green: 0.85, blue: 0.4, alpha: 0.95))
    /// Drag-direction indicator for FIRM guiding: a red line whose length is
    /// the drag magnitude and whose direction is the drag direction.
    private let guideArrow = ForceArrow(color: UIColor(red: 0.92, green: 0.22, blue: 0.18, alpha: 0.95))
    private(set) var interactive = false
    /// What `load` built, so a robot / hand change can rebuild IN PLACE (`reloadIfRigChanged`):
    /// the top-level rest entities, each group's articulated entity, and every entity the rig
    /// itself created — anything else hanging off a joint (SensorLayers' camera planes, TF
    /// triads) is the caller's and is carried over to the rebuilt joint of the same name.
    private var builtRoots: [Entity] = []
    private var builtNodes: [String: Entity] = [:]
    private var builtSet = Set<ObjectIdentifier>()
    private var loadedSignature: String?
    /// The build in flight (or last finished). Every load/rebuild chains onto it, so only ONE
    /// build ever runs: a `.robotRigChanged` that arrived while the first load was still parsing
    /// PLYs used to start a second build beside it (nothing built yet ⇒ nothing removed), and
    /// both attached a full rig — two robots drawn on top of each other (seen connecting to the
    /// Retarget Lab with NOVA's rig saved from the last session).
    private var buildChain: Task<Void, Never>?
    /// What is drawn now — `RobotRig.signature`; a seam so tests can stage a swap without flipping
    /// the app-wide robot pick.
    var currentSignature: () -> String = { RobotRig.signature }
    /// Which rig to build (robot manifest + per-side end effector) — the app-wide `RobotRig` pick;
    /// a seam so tests can build NOVA-with-grippers without flipping it.
    var rigPick: () -> (manifest: String, gen: (String) -> String) = { (RobotRig.manifest, { RobotRig.handGen($0) }) }
    private var rigObserver: NSObjectProtocol?

    init() {
        rigObserver = NotificationCenter.default.addObserver(forName: .robotRigChanged, object: nil,
                                                             queue: .main) { [weak self] _ in
            Task { @MainActor in await self?.reloadIfRigChanged() }
        }
        root.addChild(robot)
        // Named so a bounds walk can recognise (and skip) them — see `parts`.
        forceArrow.entity.name = "decor:force"
        guideArrow.entity.name = "decor:guide"
        robot.addChild(forceArrow.entity)
        forceArrow.entity.isEnabled = false
        robot.addChild(guideArrow.entity)
        guideArrow.entity.isEnabled = false
    }

    /// Parse + build the rig. Mesh parsing is heavy (~14 MB of PLY), so the
    /// geometry is parsed off the main actor; entities are assembled here.
    /// - Parameter interactive: install collision + hover + input targets on
    ///   the arm links (the Teleop stage); pure visualizers skip it.
    func load(interactive: Bool = false) async {
        self.interactive = interactive
        await serialized { await self.build() }
    }

    /// Run `work` after every build queued before it; returns when it (and so the latest rig
    /// requested so far) is done.
    private func serialized(_ work: @escaping @MainActor () async -> Void) async {
        let previous = buildChain
        let task = Task { @MainActor in
            await previous?.value
            await work()
        }
        buildChain = task
        await task.value
    }

    private func build() async {
        loadedSignature = currentSignature()
        // The CONNECTED robot's rig (MABEL or NOVA, see RobotRig) — this loaded MABEL's "rig"
        // unconditionally, so every RealityKit view drew MABEL while connected to NOVA.
        let pick = rigPick()
        guard let url = Bundle.main.url(forResource: pick.manifest, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else {
            let box = ModelEntity(mesh: .generateBox(size: SIMD3(0.4, 1, 0.4)),
                                  materials: [SimpleMaterial(color: .gray, isMetallic: false)])
            box.position.y = 0.5
            robot.addChild(box)
            builtRoots.append(box)
            builtSet.insert(ObjectIdentifier(box))
            return
        }

        // Parse all PLYs off the main thread.
        let groups = RobotRig.withMountedHands(manifest.groups, name: { $0.name }, parent: { $0.parent },
                                               nova: pick.manifest == "nova_rig", gen: pick.gen)
        let parsed: [String: PLYMesh.Geometry] = await Task.detached(priority: .userInitiated) {
            var out: [String: PLYMesh.Geometry] = [:]
            for grp in groups {
                for part in grp.parts {
                    let name = (part.file as NSString).deletingPathExtension
                    guard out[name] == nil,
                          let u = Bundle.main.url(forResource: name, withExtension: "ply"),
                          let g = PLYMesh.parse(url: u) else { continue }
                    out[name] = g
                }
            }
            return out
        }.value

        var restNodes = [String: Entity]()
        var jointNodes = [String: Entity]()

        for grp in groups {
            let rest = Entity()
            rest.transform = Transform(matrix: grp.matrix)
            let art = Entity()
            art.name = grp.joint ?? grp.name
            rest.addChild(art)

            let linkKey = "link:\(grp.name)"
            let draggable = interactive && Self.pushBody(forGroup: grp) != nil

            for part in grp.parts {
                let name = (part.file as NSString).deletingPathExtension
                guard let g = parsed[name], let mesh = try? PLYMesh.meshResource(from: g) else { continue }
                let lum = Double(part.rgb.reduce(0, +)) / 3.0 / 255.0
                var mat = PhysicallyBasedMaterial()
                mat.baseColor = .init(tint: UIColor(red: CGFloat(part.rgb[0]) / 255,
                                                    green: CGFloat(part.rgb[1]) / 255,
                                                    blue: CGFloat(part.rgb[2]) / 255, alpha: 1))
                mat.metallic = .init(floatLiteral: lum < 0.4 ? 0.6 : 0.12)
                mat.roughness = .init(floatLiteral: lum < 0.4 ? 0.42 : 0.6)
                let model = ModelEntity(mesh: mesh, materials: [mat])
                if interactive {
                    // EVERY part is gaze-targetable: draggable links carry a
                    // link: name (pinch-drag guides/pushes); the rest resolve
                    // to no link, which the stage treats as ORBIT — so
                    // looking anywhere on the robot and dragging rotates it.
                    if draggable { model.name = linkKey }
                    let bounds = model.model?.mesh.bounds ?? .init(min: .zero, max: .zero)
                    let extents = simd_max(bounds.extents, SIMD3(repeating: 0.015))
                    model.components.set(CollisionComponent(shapes: [
                        .generateBox(size: extents).offsetBy(translation: bounds.center)
                    ]))
                    model.components.set(InputTargetComponent())
                    model.components.set(HoverEffectComponent())
                }
                art.addChild(model)
                parts.append(model)
                builtSet.insert(ObjectIdentifier(model))
            }

            if draggable, let body = Self.pushBody(forGroup: grp) {
                links[linkKey] = RigLink(side: body.hasPrefix("l_") ? "left" : "right",
                                         pushBody: body)
            }

            restNodes[grp.name] = rest
            jointNodes[grp.name] = art
            builtSet.insert(ObjectIdentifier(rest)); builtSet.insert(ObjectIdentifier(art))
            if let joint = grp.joint {
                let ax = grp.axis.flatMap { $0.count == 3 ? SIMD3<Float>($0[0], $0[1], $0[2]) : nil }
                    ?? SIMD3<Float>(0, 0, 1)
                joints[joint] = RigJoint(node: art, type: grp.type, axis: ax, ref: grp.ref ?? 0, mimic: grp.mimic)
            }
        }
        for grp in groups {
            guard let rest = restNodes[grp.name] else { continue }
            if let parent = grp.parent, let pArt = jointNodes[parent] {
                pArt.addChild(rest)
            } else {
                robot.addChild(rest)
                builtRoots.append(rest)
            }
        }
        builtNodes = jointNodes
    }

    /// Rebuild the geometry when the connected robot or its mounted hands changed since `load`
    /// (RobotRig posts `.robotRigChanged`): same `root`/`robot`, so the caller's transforms,
    /// gestures and attached layers survive. A layer hung off a link that no longer exists
    /// (a v2 finger, after a swap to v1) is detached.
    func reloadIfRigChanged() async {
        // Queued behind any build in flight; the signature is compared once that one is done, so
        // a change that landed mid-load rebuilds exactly once, and a no-op change not at all.
        await serialized { await self.rebuildIfStale() }
    }

    private func rebuildIfStale() async {
        guard let sig = loadedSignature, sig != currentSignature() else { return }
        var foreign: [(String, Entity)] = []
        for (name, art) in builtNodes {
            for c in art.children where !builtSet.contains(ObjectIdentifier(c)) { foreign.append((name, c)) }
        }
        for (_, c) in foreign { c.removeFromParent() }
        for r in builtRoots { r.removeFromParent() }
        joints = [:]; parts = []; links = [:]
        builtRoots = []; builtNodes = [:]; builtSet = []
        await build()
        for (name, c) in foreign { builtNodes[name]?.addChild(c) }
    }

    /// MuJoCo push body for a rig group, or nil if the link isn't draggable.
    /// Mirrors RobotController.pushBody: the arm chain takes a push, and ALL
    /// hand geometry (palm, fingers — whose rig groups carry joint: None and
    /// "__vN"/"__anchor" suffixed names, and the OpenArm gripper's base +
    /// fingers) maps to the hand mount, so looking
    /// at the robot's hand and pinch-dragging works.
    private static func pushBody(forGroup grp: Group) -> String? {
        // Prefer the joint id (e.g. "left_arm_4") → the iOS mapping.
        if let joint = grp.joint, let body = RobotController.pushBody(forJoint: joint),
           joint.hasPrefix("left") || joint.hasPrefix("right") {
            return body
        }
        // Group names that ARE MuJoCo arm bodies (hand mounts, wrists, forearms).
        let armBodies: Set<String> = ["l_hand_mount", "r_hand_mount", "l_wrist", "r_wrist",
                                      "l_forearm", "r_forearm", "l_arm_mount", "r_arm_mount"]
        if armBodies.contains(grp.name) { return grp.name }
        // Hand geometry: normalize "left_index_mp__v38" → side + hand keyword.
        let base = grp.name.replacingOccurrences(of: #"__(v\d+|anchor)$"#,
                                                 with: "", options: .regularExpression)
        let side: String? = base.hasPrefix("left_") || base.hasPrefix("l_") ? "l"
            : (base.hasPrefix("right_") || base.hasPrefix("r_") ? "r" : nil)
        guard let side else { return nil }
        let handParts = ["palm", "hand", "thumb", "index", "middle", "ring", "pinky", "wrist", "gripper"]
        return handParts.contains(where: { base.contains($0) }) ? "\(side)_hand_mount" : nil
    }

    // MARK: - Geometry measurement (the robot, not its decorations)

    /// Union of the PLY parts' bounds in `ref`'s space — the robot's TRUE
    /// extent at the current pose. nil until the async load has produced parts.
    ///
    /// Use this, never `robot.visualBounds`: that also counts the force/guide
    /// arrows, the camera-projection planes, the TF triads and the laser ring
    /// (and RealityKit includes DISABLED entities unless asked not to), so it
    /// answers a different question than "how big is the robot".
    func geometryBounds(relativeTo ref: Entity?, where include: ((ModelEntity) -> Bool)? = nil)
        -> BoundingBox? {
        var out: BoundingBox? = nil
        for p in parts where include?(p) ?? true {
            let b = p.visualBounds(recursive: false, relativeTo: ref)
            guard b.extents.x.isFinite, b.extents.x > 0 || b.extents.y > 0 else { continue }
            out = out.map { $0.union(b) } ?? b
        }
        return out
    }

    /// Bounds of the geometry under `node` (a joint/mount entity), in `ref`'s
    /// space — a module's real extent, decorations excluded.
    ///
    /// `stoppingAt` prunes named sub-chains, because the rig is one tree: the
    /// lift column's subtree runs on through the torso into both arms and every
    /// finger, so an unpruned walk answers "the whole upper body" when the
    /// question was "the lift".
    func geometryBounds(under node: Entity, relativeTo ref: Entity?,
                        stoppingAt stop: Set<String> = []) -> BoundingBox? {
        geometryBounds(relativeTo: ref) { Self.isDescendant($0, of: node, stoppingAt: stop) }
    }

    private static func isDescendant(_ e: Entity, of node: Entity,
                                     stoppingAt stop: Set<String> = []) -> Bool {
        var cur: Entity? = e
        while let c = cur {
            if c === node { return true }
            if !stop.isEmpty, stop.contains(c.name) { return false }
            cur = c.parent
        }
        return false
    }

    /// The wheel-contact plane in `ref`'s space: the lowest point of the robot's
    /// own geometry. nil until the meshes load.
    func groundY(relativeTo ref: Entity?) -> Float? {
        geometryBounds(relativeTo: ref)?.min.y
    }

    /// Resolve a hit entity to its link, walking up the tree.
    func link(of entity: Entity) -> RigLink? {
        var e: Entity? = entity
        while let cur = e {
            if cur.name.hasPrefix("link:"), let info = links[cur.name] { return info }
            e = cur.parent
        }
        return nil
    }

    // MARK: - Force visualization

    /// Show the push arrow at a scene-space point on the robot, pointing along
    /// a scene-space direction; `magnitude` in newtons sets length and label.
    func showForce(at scenePos: SIMD3<Float>, direction: SIMD3<Float>, magnitude: Float) {
        let localPos = robot.convert(position: scenePos, from: nil)
        let localDir = robot.convert(direction: direction, from: nil)
        forceArrow.update(at: localPos, direction: localDir, magnitude: magnitude)
        forceArrow.entity.isEnabled = true
    }

    func hideForce() {
        forceArrow.entity.isEnabled = false
    }

    /// Red drag indicator (FIRM guiding). The anchor point is captured ON
    /// the dragged link at drag start and FOLLOWS the link as the arm moves —
    /// the arrow base always stays on the robot.
    private var guideAnchor: (entity: Entity, local: SIMD3<Float>)?

    func showGuide(on entity: Entity, contactScene: SIMD3<Float>,
                   direction: SIMD3<Float>, magnitude: Float) {
        if guideAnchor?.entity !== entity {
            guideAnchor = (entity, entity.convert(position: contactScene, from: nil))
        }
        guard let anchor = guideAnchor else { return }
        let scenePoint = anchor.entity.convert(position: anchor.local, to: nil)
        let localPos = robot.convert(position: scenePoint, from: nil)
        let localDir = robot.convert(direction: direction, from: nil)
        guideArrow.update(at: localPos, direction: localDir, magnitude: magnitude)
        guideArrow.entity.isEnabled = true
    }

    func hideGuide() {
        guideArrow.entity.isEnabled = false
        guideAnchor = nil
    }

    // MARK: - Posing

    /// TRUE while the SLAM /map odom owns the chassis orientation (SensorLayers
    /// applies the map-frame pose, like the iOS map view). robot_state's heading
    /// is in the bridge's ODOM frame — on the real robot that differs from the
    /// SLAM map frame by the localization transform, so letting the 50 Hz pose()
    /// stomp it put the robot's nose (and body) out of register with the
    /// occupancy grid / mesh / clouds.
    var baseOrientationExternal = false

    /// Joints under an active StagePlanner grab / staged plan: telemetry must
    /// not repaint them (the local IK prediction / ghost preview owns them) —
    /// the RealityKit mirror of LiveMapScene.apply(joints:except:).
    var frozenJoints: Set<String> = []

    /// Pose every joint from the shared joint dictionary (radians / metres),
    /// plus the chassis orientation — the same maths as the iOS viewer.
    func pose(joints j: [String: Double], heading: Double, baseQuat: [Double]) {
        // MuJoCo is Z-up, RealityKit Y-up: rebase by Rx(−90°).
        let isLevel = baseQuat.count == 4 && baseQuat[0] > 0.9999
            && abs(baseQuat[1]) < 1e-4 && abs(baseQuat[2]) < 1e-4 && abs(baseQuat[3]) < 1e-4
        if baseOrientationExternal {
            // SLAM owns the chassis — joints only.
        } else if !isLevel, baseQuat.count == 4 {
            let qm = simd_quatf(ix: Float(baseQuat[1]), iy: Float(baseQuat[2]),
                                iz: Float(baseQuat[3]), r: Float(baseQuat[0]))
            let c = simd_quatf(angle: -.pi / 2, axis: SIMD3(1, 0, 0))
            robot.orientation = c * qm * c.inverse
        } else {
            robot.orientation = simd_quatf(angle: Float(j["base_heading"] ?? heading),
                                           axis: SIMD3(0, 1, 0))
        }
        for (id, joint) in joints where !frozenJoints.contains(id) {
            // Missing joint -> fall back to its ref (qpos0), not 0; a 0 fallback drops
            // the half-raised lift 31.75 cm to the floor.
            let value = Float(RobotRig.jointValue(id, mimic: joint.mimic, in: j) ?? Double(joint.ref)) - joint.ref
            if joint.type == "prismatic" {
                joint.node.position = joint.axis * value
            } else {
                joint.node.orientation = simd_quatf(angle: value, axis: simd_normalize(joint.axis))
            }
        }
    }

    // MARK: - Manifest (identical schema to the iOS loader)

    private struct Manifest: Decodable { let groups: [Group] }
    fileprivate struct Group: Decodable {
        let name: String
        let parent: String?
        let joint: String?
        let type: String
        let parts: [Part]
        let m: [Float]
        let axis: [Float]?
        let ref: Float?
        let mimic: String?
        var matrix: simd_float4x4 {
            simd_float4x4(columns: (
                SIMD4(m[0], m[1], m[2], m[3]),
                SIMD4(m[4], m[5], m[6], m[7]),
                SIMD4(m[8], m[9], m[10], m[11]),
                SIMD4(m[12], m[13], m[14], m[15])
            ))
        }
    }
    fileprivate struct Part: Decodable { let file: String; let rgb: [Int] }
}

// MARK: - Force arrow

/// A shaft + cone arrow showing where the operator's push lands, which way
/// it points, and how strong it is (length and colour scale with magnitude).
@MainActor
private final class ForceArrow {
    let entity = Entity()
    private let shaft: ModelEntity
    private let tip: ModelEntity
    private let ring: ModelEntity

    init(color: UIColor) {
        shaft = ModelEntity(mesh: .generateCylinder(height: 1, radius: 0.008),
                            materials: [UnlitMaterial(color: color)])
        tip = ModelEntity(mesh: .generateCone(height: 0.05, radius: 0.02),
                          materials: [UnlitMaterial(color: color)])
        ring = ModelEntity(mesh: .generateCylinder(height: 0.004, radius: 0.035),
                           materials: [UnlitMaterial(color: color.withAlphaComponent(0.5))])
        entity.addChild(shaft)
        entity.addChild(tip)
        entity.addChild(ring)
    }

    /// `position` in the robot's frame; arrow points along `direction`.
    func update(at position: SIMD3<Float>, direction: SIMD3<Float>, magnitude: Float) {
        let len = min(0.5, max(0.06, magnitude / 120))    // N → metres of arrow
        let dir = simd_length(direction) > 1e-6 ? simd_normalize(direction) : SIMD3(0, 1, 0)
        entity.position = position
        entity.orientation = simd_quatf(from: SIMD3(0, 1, 0), to: dir)
        shaft.scale = SIMD3(1, len, 1)
        shaft.position = SIMD3(0, len / 2, 0)
        tip.position = SIMD3(0, len + 0.025, 0)
        // Anchor ring marks the contact point.
        ring.position = .zero
        ring.orientation = simd_quatf(from: SIMD3(0, 1, 0), to: SIMD3(0, 1, 0))
    }
}
