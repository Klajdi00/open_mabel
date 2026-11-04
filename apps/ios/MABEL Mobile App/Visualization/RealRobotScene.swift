//
//  RealRobotScene.swift
//  MABEL Mobile App
//
//  Loads the REAL MABEL — the same rig the website and MuJoCo sim use —
//  as an articulated SceneKit model. The GLB was re-exported grouped by
//  rig node, each group split into solid-colour parts (Resources/MabelRig:
//  rig_<node>_<colour>.ply + rig.json with the kinematic hierarchy). We
//  rebuild that hierarchy so every joint moves live, exactly like the
//  website's viewer, keeping the real geometry and URDF colours.
//
//  Every MABEL joint rotates/slides about its own local Z axis (per
//  mabel_joints.json). The model also carries two green wrist handles for
//  gravity-compensation dragging, mirroring the web viewer.
//

import SceneKit
import SwiftUI
import Combine
import ModelIO
import SceneKit.ModelIO
import UIKit
import simd

struct RigJoint {
    let node: SCNNode
    let type: String   // "revolute" | "prismatic"
    var axis: simd_float3 = simd_float3(0, 0, 1)   // rotation/slide axis (local)
    /// MuJoCo qpos0 (the joint's ref). The rig geometry is baked at the ASSEMBLED
    /// pose (qpos == ref), and the wire streams the ABSOLUTE qpos, so a joint is
    /// rendered at the DISPLACEMENT `qpos − ref`. On MABEL only the 14 ORCA finger/
    /// wrist joints carry a nonzero ref (thumb Abd/PIP/DIP up to −0.73 rad); ignoring
    /// it rendered the hands splayed/twisted while the arms (ref = 0) looked right.
    var ref: Float = 0
    /// A MIMIC joint (the OpenArm gripper's second finger, `<side>_gripper_mirror`) has no wire
    /// value of its own: it is drawn at its LEADER's value (multiplier 1 — its own axis already
    /// points the other way). Read the pose through `RobotRig.jointValue(_:mimic:in:)`.
    var mimic: String? = nil
}

/// Everything the view needs to drive the scene.
final class RealRig {
    var joints: [String: RigJoint] = [:]
    var root = SCNNode()
    let orbit = SCNNode()       // azimuth/elevation pivot at the look-at point
    let camera = SCNNode()      // child of orbit, offset by distance
    var handles: [String: SCNNode] = [:]   // "left"/"right" green palm balls
    var opOverlay: OperatorOverlayNode?    // VP tracking overlay (retarget debug)
    var distance: Float = 2.9
}

enum RealRobotScene {

    /// Build just the articulated robot node + its joint handles, for
    /// embedding elsewhere (e.g. the navigation map's miniature MABEL).
    static func makeRobotNode() -> (SCNNode, [String: RigJoint]) {
        let (root, joints, _) = loadRig()
        return (root, joints)
    }

    /// Same rig, but keeping the palm handles. LiveMapScene needs them: it is the
    /// only surface that shows the task-space grab targets, and it previously had no
    /// reference to the nodes at all, so it guessed at a parent that does not hold
    /// them and its show/hide silently did nothing. Added alongside rather than
    /// widening `makeRobotNode`, whose four other callers want the rig and nothing
    /// else and should not have to destructure a value they will discard.
    static func makeRobotNodeWithHandles() -> (SCNNode, [String: RigJoint], [String: SCNNode]) {
        loadRig()
    }

    /// The rig for an EXPLICIT pick (robot manifest + per-side end effector) instead of the app-wide
    /// `RobotRig` one — the seam the tests build NOVA-with-grippers through without flipping the pick.
    static func makeRobotNodeWithHandles(manifest: String, gen: @escaping (String) -> String)
        -> (SCNNode, [String: RigJoint], [String: SCNNode]) {
        loadRig(manifest: manifest, gen: gen)
    }

    static func make() -> (SCNScene, RealRig) {
        let scene = SCNScene()
        scene.background.contents = UIColor.clear
        let rig = RealRig()

        let (robotRoot, joints, handles) = loadRig()
        rig.root = robotRoot
        rig.joints = joints
        rig.handles = handles
        // rig.json's base_link is the odom/free-joint frame (no baked website
        // centering any more): lift it to the true base height so the feet
        // stand on the studio floor, and pivot the orbit on the body centre.
        robotRoot.position = SCNVector3(0, 0.078, 0)
        scene.rootNode.addChildNode(robotRoot)

        // VP tracking overlay: parented to the rig's base_link node, whose rest
        // transform is the ONE Z-up→Y-up rotation that places the robot from
        // odom — the streamed odom-frame points are frame-exact as raw local
        // positions (mirrors the web CC's rig.baseQ0 parenting).
        if let baseLink = robotRoot.childNode(withName: "base_link", recursively: true) {
            let overlay = OperatorOverlayNode()
            baseLink.addChildNode(overlay)
            rig.opOverlay = overlay
        }

        installStudio(scene)

        // Manual orbit camera rig so we can mix orbit + handle dragging.
        let cam = SCNCamera()
        cam.fieldOfView = 38; cam.wantsHDR = true
        // No bloom: the only emissive things were the wrist handle + rings, and bloom is
        // exactly the "glow" that read as unprofessional. HDR tone-mapping stays on.
        cam.bloomIntensity = 0.0; cam.zNear = 0.02
        rig.camera.camera = cam
        rig.distance = 3.2                                  // slight zoom-out so it fits
        rig.camera.position = SCNVector3(0, 0, rig.distance)
        // Look lower on the robot so the WHOLE robot (incl. the base) renders in the
        // upper part of the frame — clear of the floating bottom control panel.
        // x = the body's bbox centre (was baked into the rig root pre-fix).
        rig.orbit.position = SCNVector3(-0.23, 0.42, 0)
        // FRONT isometric view: azimuth turned ~180° from the old back-facing angle
        // so we look at the robot's FRONT, with a slight downward tilt for a clean
        // 3/4 isometric framing.
        rig.orbit.eulerAngles = SCNVector3(-0.16, 0.5 + .pi, 0)
        rig.orbit.addChildNode(rig.camera)
        scene.rootNode.addChildNode(rig.orbit)
        return (scene, rig)
    }

    // MARK: - Rig

    private static func loadRig(manifest manifestName: String = RobotRig.manifest,
                                gen: @escaping (String) -> String = { RobotRig.handGen($0) })
        -> (SCNNode, [String: RigJoint], [String: SCNNode]) {
        let root = SCNNode(); root.name = "mabel"
        var joints = [String: RigJoint]()
        var handles = [String: SCNNode]()

        guard let url = Bundle.main.url(forResource: manifestName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else {
            let n = SCNNode(geometry: SCNBox(width: 0.4, height: 1, length: 0.4, chamferRadius: 0.05))
            n.position = SCNVector3(0, 0.5, 0); root.addChildNode(n)
            return (root, joints, handles)
        }

        let groups = RobotRig.withMountedHands(manifest.groups, name: { $0.name }, parent: { $0.parent },
                                               nova: manifestName == "nova_rig", gen: gen)
        var restNodes = [String: SCNNode]()
        var jointNodes = [String: SCNNode]()
        for grp in groups {
            let rest = SCNNode()
            rest.simdTransform = grp.matrix
            let art = SCNNode()
            rest.addChildNode(art)
            for part in grp.parts {
                if let geo = loadMesh(named: (part.file as NSString).deletingPathExtension, rgb: part.rgb) {
                    art.addChildNode(SCNNode(geometry: geo))
                }
            }
            art.name = grp.joint ?? grp.name   // so a drag can identify this link
            restNodes[grp.name] = rest
            jointNodes[grp.name] = art
            if let joint = grp.joint {
                let ax = grp.axis.flatMap { $0.count == 3 ? simd_float3($0[0], $0[1], $0[2]) : nil } ?? simd_float3(0, 0, 1)
                joints[joint] = RigJoint(node: art, type: grp.type, axis: ax, ref: grp.ref ?? 0, mimic: grp.mimic)
            }
        }
        for grp in groups {
            guard let rest = restNodes[grp.name] else { continue }
            if let parent = grp.parent, let pArt = jointNodes[parent] {
                pArt.addChildNode(rest)
            } else {
                root.addChildNode(rest)
            }
        }

        // Green draggable balls at each PALM (the wrist mount is ±0.255 m back
        // along local Y from the palm/end-effector; place the marker at the palm).
        // Each ball also carries a roll/pitch/yaw ORIENTATION RING gizmo (like the
        // web teleop nav): drag a ring to spin that wrist axis.
        for (side, node) in [("left", "l_hand_mount"), ("right", "r_hand_mount")] {
            guard let art = jointNodes[node] else { continue }
            let ball = makeHandle()
            // Place the marker at the PALM CENTRE, not the wrist mount: the palm/end-
            // effector is 0.255 m forward of the mount along local Y (see header). The
            // old 0.135 left it back at the wrist, off the hand.
            ball.position = SCNVector3(RobotRig.handlePoint(side, gen: gen(side)))   // palm, or the gripper TCP
            addOrientationRings(to: ball, side: side)
            art.addChildNode(ball)
            handles[side] = ball
        }
        return (root, joints, handles)
    }

    /// Roll/Pitch/Yaw rings around a wrist ball. Each ring is named
    /// "ring_<axis>_<side>" so a drag can map it to that wrist joint.
    private static func addOrientationRings(to ball: SCNNode, side: String) {
        func ring(_ axisName: String, color: UInt32, euler: SCNVector3) -> SCNNode {
            // A thin, crisp gizmo ring in the axis colour — flat-shaded, NO additive blend
            // and NO bloom halo, so it reads as a precise control ring (consistent with the
            // refined handle) rather than a neon planetary glow.
            let torus = SCNTorus(ringRadius: 0.10, pipeRadius: 0.009)
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = UIColor(rgb: color)
            m.emission.contents = UIColor(rgb: color)
            m.writesToDepthBuffer = false
            torus.materials = [m]
            let n = SCNNode(geometry: torus)
            n.name = "ring_\(axisName)_\(side)"
            n.eulerAngles = euler
            n.opacity = 0.85
            return n
        }
        // Default SCNTorus lies in the X-Z plane (axis = Y).
        ball.addChildNode(ring("roll",  color: 0x4F9DFF, euler: SCNVector3(0, 0, 0)))                  // XZ plane
        ball.addChildNode(ring("pitch", color: 0xE7913A, euler: SCNVector3(Float.pi / 2, 0, 0)))       // XY plane
        ball.addChildNode(ring("yaw",   color: 0xE0503D, euler: SCNVector3(0, 0, Float.pi / 2)))       // YZ plane
    }

    private static func makeHandle() -> SCNNode {
        // A clean, physically-shaded control knob in the app's green (Theme.Palette.live
        // = 0x2E9E52), lit like the robot's own PBR meshes — deliberately NOT a bright
        // emissive blob with an additive bloom halo. It reads as a precise, professional
        // handle, consistent with the rest of the UI's restrained material language.
        let live: UInt32 = 0x2E9E52
        let s = SCNSphere(radius: 0.06); s.segmentCount = 48
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = UIColor(rgb: live)
        m.roughness.contents = 0.38
        m.metalness.contents = 0.0
        // A whisper of self-illumination so it stays legible in shadow, kept well below
        // the camera's bloomThreshold (0.95) so it never blooms.
        m.emission.contents = UIColor(rgb: live)
        m.emission.intensity = 0.10
        s.materials = [m]
        let n = SCNNode(geometry: s)
        n.name = "handle"
        n.isHidden = true   // shown only in gravity-comp mode (toggled in apply)
        // A restrained breath (≤5 %, slow) — in line with the app's subtle motion, and
        // a far cry from the old 16 % pulsing glow.
        let pulse = SCNAction.sequence([
            .scale(to: 1.05, duration: 0.9), .scale(to: 1.0, duration: 0.9)
        ])
        pulse.timingMode = .easeInEaseOut
        n.runAction(.repeatForever(pulse))
        return n
    }

    // Parsing ~100 PLY meshes is the dominant cost of building the rig, and it
    // happened on every visit to a 3D screen. Cache the finished geometry by file
    // (geometry is immutable + safely shareable across nodes/scenes) so only the
    // FIRST load pays for it. Guarded by a lock because the widget pose renderer
    // builds the rig off the main thread while the app builds it on main.
    private static var meshCache: [String: SCNGeometry] = [:]
    private static let cacheLock = NSLock()

    private static func loadMesh(named: String, rgb: [Int]) -> SCNGeometry? {
        cacheLock.lock(); let cached = meshCache[named]; cacheLock.unlock()
        if let cached { return cached }

        guard let url = Bundle.main.url(forResource: named, withExtension: "ply") else { return nil }
        let asset = MDLAsset(url: url)
        guard let mdl = asset.object(at: 0) as? MDLMesh else { return nil }
        let geo = SCNGeometry(mdlMesh: mdl)
        let lum = Double(rgb.reduce(0, +)) / 3.0 / 255.0
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = UIColor(red: CGFloat(rgb[0]) / 255, green: CGFloat(rgb[1]) / 255,
                                     blue: CGFloat(rgb[2]) / 255, alpha: 1)
        m.metalness.contents = lum < 0.4 ? 0.6 : 0.12
        m.roughness.contents = lum < 0.4 ? 0.42 : 0.6
        geo.materials = [m]

        cacheLock.lock(); meshCache[named] = geo; cacheLock.unlock()
        return geo
    }

    // MARK: - Studio

    private static func installStudio(_ scene: SCNScene) {
        let key = SCNLight(); key.type = .directional; key.intensity = 1000; key.temperature = 6500
        key.castsShadow = true; key.shadowMode = .deferred; key.shadowRadius = 9
        key.shadowColor = UIColor(white: 0, alpha: 0.38)
        let kn = SCNNode(); kn.light = key; kn.eulerAngles = SCNVector3(-Float.pi/3.2, Float.pi/5, 0)
        scene.rootNode.addChildNode(kn)
        let fill = SCNLight(); fill.type = .directional; fill.intensity = 360; fill.temperature = 7400
        let fn = SCNNode(); fn.light = fill; fn.eulerAngles = SCNVector3(-Float.pi/6, -Float.pi/2.4, 0)
        scene.rootNode.addChildNode(fn)
        let amb = SCNLight(); amb.type = .ambient; amb.intensity = 480; amb.color = UIColor(white: 0.32, alpha: 1)
        let an = SCNNode(); an.light = amb; scene.rootNode.addChildNode(an)
        let floor = SCNFloor(); floor.reflectivity = 0.04
        floor.firstMaterial?.diffuse.contents = UIColor(white: 0.05, alpha: 1)
        floor.firstMaterial?.lightingModel = .physicallyBased
        floor.firstMaterial?.roughness.contents = 0.96
        scene.rootNode.addChildNode(SCNNode(geometry: floor))
    }

    // MARK: - Manifest

    private struct Manifest: Decodable { let groups: [Group] }
    private struct Group: Decodable {
        let name: String
        let parent: String?
        let joint: String?
        let type: String
        let parts: [Part]
        let m: [Float]
        let axis: [Float]?
        let ref: Float?              // MuJoCo qpos0 (see RigJoint.ref); optional
        let mimic: String?           // leader joint (see RigJoint.mimic); optional
        var matrix: simd_float4x4 {
            simd_float4x4(columns: (
                SIMD4(m[0], m[1], m[2], m[3]),
                SIMD4(m[4], m[5], m[6], m[7]),
                SIMD4(m[8], m[9], m[10], m[11]),
                SIMD4(m[12], m[13], m[14], m[15])
            ))
        }
    }
    private struct Part: Decodable { let file: String; let rgb: [Int] }
}

// MARK: - Which robot's rig, and which hands are on it
//
// Two robots share this app. MABEL ships Resources/MabelRig (rig.json + rig_*.ply, ORCA hand v1) and is
// the default; NOVA ships Resources/NovaRig (nova_rig.json + nova_rig_*.ply, ORCA hand v2 on the same
// arm datum — generated by NOVA/ros_ws/src/nova_description/urdf_to_glb.py --apps).
//
// The ORCA hands SWAP between the robots (a hand is known by its USB adapter serial), so NOVA may be
// wearing MABEL's v1 hands on either side. The gateway's hello says which: `robot` ("MABEL"/"NOVA") and
// `hands` ({left, right}: "v1"/"v2", what its HAL actually drives). For a NOVA side whose hand is v1 the
// loaders splice Resources/NovaRig/nova_hands_v1_<side>.json (the v1 subtree under that side's hand
// mount, refs = qpos0 like MABEL's; built by NOVA's build_hand_variants.py --apps) over the v2 subtree.
// A side may also wear the original OpenArm v1 parallel GRIPPER (`hands` side = "gripper"): the loaders
// splice nova_gripper_v1_<side>.json the same way — one prismatic `<side>_gripper` (one finger's
// opening, 0…0.044 m, ref 0) plus a second finger that MIMICS it — and that side has no ORCA joints.
// MABEL can wear the gripper too (the Retarget Lab's End effector toggle): Resources/MabelRig/
// mabel_gripper_v1_<side>.json replaces everything under MABEL's J7 node (ORCA adapter + hand) with
// OpenArm's link7 + the gripper, placed exactly as the lab draws it (simulation/tools/export_gripper_rig.py).
// An older gateway without `robot` is told apart by its `server` string ("nova_*"); without `hands` the
// robot's own hands are assumed. The choice persists, so the next launch starts on the right robot, and
// a change posts `.robotRigChanged` so every open 3D view rebuilds (`.rebuildsOnRobotRigChange()`).
enum RobotRig {
    static let key = "robotRigManifest"
    static let handsKey = "robotRigHands"
    private static var _manifest = UserDefaults.standard.string(forKey: key) ?? "rig"
    private static var _hands = (UserDefaults.standard.dictionary(forKey: handsKey) as? [String: String]) ?? [:]
    /// The gateway the current pick came from (hello.server, this launch): a DIFFERENT gateway that says
    /// nothing about its hands (MABEL real) must not inherit the last one's gripper.
    private static var _server: String?
    static var manifest: String { _manifest }
    static var isNova: Bool { _manifest == "nova_rig" }
    /// What is mounted on this side: "v1" (MABEL's ORCA hand), "v2" (NOVA's) or "gripper" (OpenArm v1).
    static func handGen(_ side: String) -> String { handGen(side, manifest: _manifest, hands: _hands) }
    /// The per-side values a gateway may report in `hello.hands`.
    static let endEffectors: Set<String> = ["v1", "v2", "gripper"]
    /// What is drawn; when it changes, every 3D view rebuilds.
    static var signature: String { signature(manifest: _manifest, hands: _hands) }
    /// `signature` for an arbitrary pick (pure): "<manifest>|<left>|<right>", e.g. "nova_rig|gripper|v1".
    static func signature(manifest: String, hands: [String: String]) -> String {
        "\(manifest)|\(handGen("left", manifest: manifest, hands: hands))|\(handGen("right", manifest: manifest, hands: hands))"
    }

    /// - Parameters:
    ///   - robot: hello.robot ("MABEL" / "NOVA"); nil from an older gateway.
    ///   - server: hello.server, the fallback ("nova_teleop_bridge" ⇒ NOVA).
    ///   - hands: hello.hands per side; nil ⇒ unchanged for the same robot (the bare `server`
    ///     string arrives a moment before the full meta), else the new robot's own hands.
    static func select(robot: String?, server: String?, hands: [String: String]?) {
        let changed = server != nil && _server != nil && server != _server
        if server != nil { _server = server }
        let (want, wantHands) = resolve(robot: robot, server: server, hands: hands,
                                        manifest: _manifest, current: _hands, gatewayChanged: changed)
        guard want != _manifest || wantHands != _hands else { return }
        _manifest = want; _hands = wantHands
        UserDefaults.standard.set(want, forKey: key)
        UserDefaults.standard.set(wantHands, forKey: handsKey)
        NotificationCenter.default.post(name: .robotRigChanged, object: signature)
    }

    /// The pick itself, pure (no globals — what the tests drive): (manifest, hands per side).
    static func resolve(robot: String?, server: String?, hands: [String: String]?,
                        manifest: String, current: [String: String],
                        gatewayChanged: Bool = false) -> (String, [String: String]) {
        let nova = robot.map { $0.uppercased() == "NOVA" } ?? (server?.lowercased().contains("nova") ?? false)
        let want = nova ? "nova_rig" : "rig"
        // NOVA's rig carries every overlay (v1 hands, gripper). MABEL's carries only the gripper:
        // MABEL wearing NOVA's v2 hands still draws its own v1 rig (no MABEL v2 overlay baked yet).
        let reported = hands.map { $0.filter { nova ? endEffectors.contains($0.value) : $0.value == "gripper" } }
        let wantHands = reported ?? (want == manifest && !gatewayChanged ? current : [:])
        return (want, wantHands)
    }
    /// `handGen` for an arbitrary pick (same rule, no globals).
    static func handGen(_ side: String, manifest: String, hands: [String: String]) -> String {
        if let g = hands[side], endEffectors.contains(g) { return g }
        return manifest == "nova_rig" ? "v2" : "v1"
    }

    /// The value a rig joint is drawn at, from a pose dictionary: a mimic joint reads its leader, and a
    /// gripper with no opening in the pose (a timeline preview, a ghost, a command echo that only knows
    /// the 0…1 grip) follows its side's `<side>_grip` the way the gateway maps it. nil ⇒ draw at ref.
    static func jointValue(_ id: String, mimic: String? = nil, in j: [String: Double]) -> Double? {
        let src = mimic ?? id
        if let v = j[src] { return v }
        if src.hasSuffix("_gripper"), let g = j[String(src.dropLast("_gripper".count)) + "_grip"] {
            return RobotModel.gripperOpening(grip: g)
        }
        return nil
    }

    /// The node a hand module frames (and dims): the ORCA palm, or the gripper's hand body — named
    /// `<side>_gripper_base` in both NOVA's and MABEL's gripper overlays.
    static func palmName(_ side: String, gen: String? = nil) -> String {
        (gen ?? handGen(side)) == "gripper" ? "\(side)_gripper_base" : "\(side)_palm"
    }
    /// The green task-space ball, in the J7 (`<l|r>_hand_mount`) frame — also the point the drag IK
    /// reaches with. ORCA: the palm centre, 0.255 m out along the mount's ±Y. Gripper: the TCP between the
    /// fingertips, as OpenArm and the Retarget Lab define it — the wrist centre (20 mm down the J7 axis,
    /// z −0.02) plus 0.1801 m along the approach (±Y); NOVA and MABEL share the arm datum.
    static func handlePoint(_ side: String, gen: String? = nil) -> SIMD3<Float> {
        let s: Float = side == "left" ? -1 : 1
        return (gen ?? handGen(side)) == "gripper" ? SIMD3(0, s * 0.1801, -0.02) : SIMD3(0, s * 0.255, -0.01)
    }
    /// The hand module's camera axis in the `palmName` node's frame (the view looks back ALONG it): ORCA =
    /// out of the palm (the wrist camera's mount −Z, mount_quat (w,x,y,z) = (0, 0, 0.98584, 0.16772));
    /// gripper = across the jaws (the hand's ±X, normal to both the finger travel Y and the approach Z) —
    /// callers flip it away from the arm mass.
    static func palmViewAxis(_ side: String, gen: String? = nil) -> SIMD3<Float> {
        if (gen ?? handGen(side)) == "gripper" { return SIMD3(1, 0, 0) }
        return simd_quatf(ix: 0, iy: 0.98583542, iz: 0.16771562, r: 0).act(SIMD3(0, 0, -1))
    }

    /// The rig groups with each side's MOUNTED hand: for a NOVA side that reports v1 (or the gripper),
    /// the v2 subtree under the hand mount is replaced by the v1 (gripper) overlay; for a MABEL side that
    /// reports the gripper, everything under its J7 node (also `<l|r>_hand_mount`). Groups are
    /// parent-first (both exporters emit depth-first), so one pass finds every descendant of the mount.
    static func withMountedHands<G: Decodable>(_ groups: [G], name: (G) -> String, parent: (G) -> String?,
                                               nova: Bool = RobotRig.isNova,
                                               gen: (String) -> String = { RobotRig.handGen($0) }) -> [G] {
        var out = groups
        for (side, mount) in [("left", "l_hand_mount"), ("right", "r_hand_mount")] {
            let file: String
            switch (nova, gen(side)) {
            case (true, "v1"):       file = "nova_hands_v1_\(side)"
            case (true, "gripper"):  file = "nova_gripper_v1_\(side)"
            case (false, "gripper"): file = "mabel_gripper_v1_\(side)"
            default:                 continue
            }
            guard let url = Bundle.main.url(forResource: file, withExtension: "json"),
                  let data = try? Data(contentsOf: url),
                  let overlay = try? JSONDecoder().decode(RigGroupFile<G>.self, from: data) else { continue }
            var drop = Set<String>()
            for g in out {
                if let p = parent(g), p == mount || drop.contains(p) { drop.insert(name(g)) }
            }
            out.removeAll { drop.contains(name($0)) }
            // An overlay may RESTATE a group the rig already has — NOVA's gripper restates the mount itself, without
            // the ORCA arm-to-hand adapter — which then takes the original's place (parents stay ahead of children).
            let restated = Dictionary(overlay.groups.map { (name($0), $0) }, uniquingKeysWith: { a, _ in a })
            let kept = Set(out.map(name))
            out = out.map { restated[name($0)] ?? $0 }
            out += overlay.groups.filter { !kept.contains(name($0)) }
        }
        return out
    }
}
private struct RigGroupFile<G: Decodable>: Decodable { let groups: [G] }
extension Notification.Name { static let robotRigChanged = Notification.Name("robotRigChanged") }

/// Rebuild a 3D view when the connected robot or its mounted hands change (the scene is built once,
/// from `RobotRig`, when the view is created).
private struct RebuildOnRobotRigChange: ViewModifier {
    @State private var signature = RobotRig.signature
    func body(content: Content) -> some View {
        content
            .id(signature)
            .onReceive(NotificationCenter.default.publisher(for: .robotRigChanged).receive(on: RunLoop.main)) { _ in
                signature = RobotRig.signature
            }
    }
}
extension View {
    func rebuildsOnRobotRigChange() -> some View { modifier(RebuildOnRobotRigChange()) }
}
