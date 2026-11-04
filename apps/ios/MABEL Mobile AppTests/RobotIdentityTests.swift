//
//  RobotIdentityTests.swift
//  MABEL Mobile AppTests
//
//  "The app should auto-detect which hand is attached — NOVA has the v1 ORCA hands on but
//  I see the v2 visualization" (2026-10-13). The hands swap between MABEL and NOVA, so the
//  robot does not decide the hand: the gateway's `hello.hands` does, per side. These pin
//  (1) the identity read from hello (robot / hands, with the old fallbacks), (2) the per-side
//  joint limits + neutral pose that follow the mounted hand, and (3) the rig splice that
//  swaps NOVA's v2 subtree for MABEL's v1 one under each hand mount.
//
//  A NOVA side may also wear the original OpenArm v1 parallel GRIPPER (`hands` side = "gripper",
//  2026-10-18): one prismatic `<side>_gripper` (one finger's opening, 0…0.044 m) plus a second
//  finger that MIMICS it. The gripper cases pin the per-side pick, the joint list (no ORCA joints
//  on that side), the overlay splice, and that the rig really moves BOTH fingers.
//
//  Everything here goes through the PURE forms (`RobotRig.resolve`, `RobotModel.joints(handGen:)`,
//  `withMountedHands(nova:gen:)`): the live pick is app-wide state, and flipping it from a test
//  would swap the clip table under XCTest suites running alongside.
//

import Testing
import Foundation
import simd
import SceneKit
@testable import MABEL_Mobile_App

@Suite("Robot + mounted-hand identity")
struct RobotIdentityTests {

    private func gens(_ h: [String: String], nova: Bool = true) -> (String) -> String {
        { RobotRig.handGen($0, manifest: nova ? "nova_rig" : "rig", hands: h) }
    }

    @Test func helloRobotAndHandsAreParsed() {
        let m = ServerMeta(from: ["server": "nova_teleop_bridge", "kind": "sim", "robot": "NOVA",
                                  "hands": ["left": "v1", "right": "v2"]])
        #expect(m.robot == "NOVA")
        #expect(m.hands == ["left": "v1", "right": "v2"])
        #expect(m.kind == .sim)          // NOVA's sim: badged by kind, its server string has no "sim"
        // NOVA's first gateway sent one string for both sides.
        let legacy = ServerMeta(from: ["server": "nova_teleop_bridge", "hands": "orca-v2"])
        #expect(legacy.hands == ["left": "v2", "right": "v2"])
        // MABEL's gateway on the real robot omits hands.
        #expect(ServerMeta(from: ["server": "mabel_real_bridge", "robot": "MABEL"]).hands == nil)
    }

    @Test func robotComesFromHelloRobotThenServerString() {
        // older NOVA gateway: no `robot`, no `hands` → NOVA by its server string, its own v2 hands
        let (m1, h1) = RobotRig.resolve(robot: nil, server: "nova_teleop_bridge", hands: nil, manifest: "rig", current: [:])
        #expect(m1 == "nova_rig" && h1.isEmpty)
        #expect(RobotRig.handGen("left", manifest: m1, hands: h1) == "v2")
        // hello.robot wins over a misleading server string
        #expect(RobotRig.resolve(robot: "MABEL", server: "nova_x", hands: nil, manifest: "rig", current: [:]).0 == "rig")
        // MABEL has no v2 overlay: it keeps drawing its own (v1) hands whatever it reports
        let (m2, h2) = RobotRig.resolve(robot: "MABEL", server: "mabel_sim_bridge",
                                        hands: ["left": "v2", "right": "v2"], manifest: "nova_rig", current: [:])
        #expect(m2 == "rig" && RobotRig.handGen("left", manifest: m2, hands: h2) == "v1")
    }

    @Test func bareServerStringKeepsTheReportedHands() {
        // a reconnect's bare `server` arrives before its meta — no flicker back to v2
        let cur = ["left": "v1", "right": "v1"]
        let (m, h) = RobotRig.resolve(robot: nil, server: "nova_teleop_bridge", hands: nil, manifest: "nova_rig", current: cur)
        #expect(m == "nova_rig" && h == cur)
        // …but a DIFFERENT robot does not inherit them
        #expect(RobotRig.resolve(robot: "NOVA", server: nil, hands: nil, manifest: "rig", current: cur).1.isEmpty)
    }

    @Test func novaWearingV1HandsLimitsAndRestsAsV1() {
        let g = gens(["left": "v1", "right": "v1"])
        let specs = Dictionary(uniqueKeysWithValues: RobotModel.joints(handGen: g).map { ($0.id, $0) })
        // v1 = MABEL's hand: MJCF limits and the qpos0 neutral, on either robot
        #expect(specs["left_thumb_abd"]?.lower == -1.0821)
        #expect(RobotModel.handNeutral(handGen: g)["right_thumb_abd"] == -0.733)
        #expect(RobotModel.handNeutral(handGen: g)["lift_lower"] == 0.15875)
    }

    @Test func handsArePerSide() {
        let g = gens(["left": "v1", "right": "v2"])
        let all = RobotModel.joints(handGen: g)
        let specs = Dictionary(uniqueKeysWithValues: all.map { ($0.id, $0) })
        #expect(specs.count == all.count)                                  // no duplicate ids
        #expect(specs["left_thumb_abd"]?.lower == -1.0821)                 // v1 (MABEL's)
        #expect(specs["right_thumb_abd"]?.lower == -0.314159)              // v2 (NOVA's)
        let n = RobotModel.handNeutral(handGen: g)
        #expect(n["left_thumb_abd"] == -0.733)
        #expect(n["right_thumb_abd"] == nil)                               // v2: a level hand IS zero
        #expect(all.filter { $0.group == .hands && !$0.id.hasSuffix("grip") }.count == 34)
    }

    @Test func mabelIsUnchanged() {
        let g = gens([:], nova: false)
        #expect(Array(RobotModel.joints(handGen: g).suffix(34)).map(\.id) == RobotModel.fingerJoints.map(\.id))
        #expect(RobotModel.handNeutral(handGen: g)["left_wrist"] == 0.15)
    }

    private struct G: Decodable {
        let name: String; let parent: String?; let joint: String?
        var type: String? = nil; var mimic: String? = nil
        var parts: [P]? = nil
        struct P: Decodable { let file: String }
    }
    private struct F: Decodable { let groups: [G] }

    @Test func v1OverlayReplacesTheV2SubtreeUnderEachMount() throws {
        let url = try #require(Bundle.main.url(forResource: "nova_rig", withExtension: "json"))
        let base = try JSONDecoder().decode(F.self, from: Data(contentsOf: url)).groups
        let out = RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                            nova: true, gen: gens(["left": "v1", "right": "v2"]))
        let byName = Dictionary(out.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(byName.count == out.count)                                 // the v2 subtree is gone, not doubled
        func under(_ m: String, _ gs: [G], _ idx: [String: G]) -> [G] {
            gs.filter { g in
                var p = g.parent
                while let q = p { if q == m { return true }; p = idx[q]?.parent }
                return false
            }
        }
        let left = under("l_hand_mount", out, byName), right = under("r_hand_mount", out, byName)
        #expect(left.compactMap(\.joint).count == 17)                     // a whole hand per side
        #expect(right.compactMap(\.joint).count == 17)
        let ovURL = try #require(Bundle.main.url(forResource: "nova_hands_v1_left", withExtension: "json"))
        let ov = try JSONDecoder().decode(F.self, from: Data(contentsOf: ovURL)).groups.map(\.name)
        #expect(Set(left.map(\.name)) == Set(ov))                          // left = the v1 overlay
        let baseIdx = Dictionary(base.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(Set(right.map(\.name)) == Set(under("r_hand_mount", base, baseIdx).map(\.name)))  // right untouched
        #expect(out.contains { $0.name == "l_hand_mount" })               // the mount itself stays
        // MABEL / v2-v2: untouched
        #expect(RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                          nova: true, gen: gens([:])).count == base.count)
    }
    // MARK: - OpenArm v1 parallel gripper

    @Test func gripperIsAPerSideEndEffector() {
        let (m, h) = RobotRig.resolve(robot: "NOVA", server: nil, hands: ["left": "gripper", "right": "v1"],
                                      manifest: "rig", current: [:])
        #expect(m == "nova_rig" && h == ["left": "gripper", "right": "v1"])
        #expect(RobotRig.handGen("left", manifest: m, hands: h) == "gripper")
        #expect(RobotRig.handGen("right", manifest: m, hands: h) == "v1")
        // an unknown value is no evidence: dropped, so the side keeps its default
        #expect(RobotRig.resolve(robot: "NOVA", server: nil, hands: ["left": "claw"], manifest: "nova_rig",
                                 current: [:]).1.isEmpty)
        // MABEL carries the gripper too (the Retarget Lab's End effector toggle) — but not NOVA's v2 hands
        #expect(RobotRig.resolve(robot: "MABEL", server: nil, hands: ["left": "gripper", "right": "v1"], manifest: "rig",
                                 current: [:]) == ("rig", ["left": "gripper"]))
        #expect(RobotRig.resolve(robot: "MABEL", server: nil, hands: ["left": "v2"], manifest: "rig",
                                 current: [:]).1.isEmpty)
        // the lab toggling back to the ORCA hand ("v1") clears it
        #expect(RobotRig.resolve(robot: "MABEL", server: "retarget_lab", hands: ["left": "v1", "right": "v1"],
                                 manifest: "rig", current: ["left": "gripper"]).1.isEmpty)
        // a DIFFERENT gateway that says nothing about hands (MABEL real) does not inherit the lab's gripper,
        // while the same gateway's bare `server` string keeps it
        #expect(RobotRig.resolve(robot: "MABEL", server: "mabel_teleop_bridge", hands: nil, manifest: "rig",
                                 current: ["left": "gripper"], gatewayChanged: true).1.isEmpty)
        #expect(RobotRig.resolve(robot: nil, server: "retarget_lab", hands: nil, manifest: "rig",
                                 current: ["left": "gripper"]).1 == ["left": "gripper"])
        // a reconnect's bare `server` keeps the gripper
        #expect(RobotRig.resolve(robot: nil, server: "nova_teleop_bridge", hands: nil, manifest: "nova_rig",
                                 current: h).1 == h)
        // hello carries it as-is
        let meta = ServerMeta(from: ["robot": "NOVA", "hands": ["left": "gripper", "right": "gripper"]])
        #expect(meta.hands == ["left": "gripper", "right": "gripper"])
    }

    @Test func gripperChangesTheSignature() {
        let v1 = RobotRig.signature(manifest: "nova_rig", hands: ["left": "v1", "right": "v1"])
        let g = RobotRig.signature(manifest: "nova_rig", hands: ["left": "gripper", "right": "v1"])
        #expect(g == "nova_rig|gripper|v1")
        #expect(g != v1)                                                    // every 3D view rebuilds
        #expect(RobotRig.signature(manifest: "nova_rig", hands: ["left": "gripper", "right": "gripper"])
                == "nova_rig|gripper|gripper")
    }

    @Test func gripperSideHasOneJointAndNoFingers() throws {
        let g = gens(["left": "gripper", "right": "v2"])
        let all = RobotModel.joints(handGen: g)
        let ids = Set(all.map(\.id))
        #expect(ids.count == all.count)                                    // no duplicate ids
        let spec = try #require(all.first { $0.id == "left_gripper" })
        #expect(spec.kind == .prismatic && spec.group == .hands && spec.label == "L Gripper")
        #expect(spec.lower == 0 && spec.upper == 0.044 && spec.unit == "m")
        // no ORCA finger or wrist joint on the gripper side; the 0…1 grip stays
        let orca = ["wrist", "thumb", "index", "middle", "ring", "pinky"]
        #expect(!all.contains { s in s.id.hasPrefix("left_") && orca.contains { s.id.contains($0) } })
        #expect(ids.contains("left_grip"))
        // the other side keeps its own hand
        #expect(!ids.contains("right_gripper"))
        #expect(all.filter { $0.id.hasPrefix("right_") && $0.group == .hands && !$0.id.hasSuffix("grip") }.count == 17)
        // neutral: nothing for the gripper side (ref 0), v1 still injects its own refs
        #expect(!RobotModel.handNeutral(handGen: g).keys.contains { $0.hasPrefix("left_") })
        let n = RobotModel.handNeutral(handGen: gens(["left": "gripper", "right": "v1"]))
        #expect(n["right_thumb_abd"] == -0.733 && n["left_thumb_abd"] == nil && n["left_wrist"] == nil)
        // both sides gripper: zero finger joints anywhere
        let both = RobotModel.joints(handGen: gens(["left": "gripper", "right": "gripper"]))
        #expect(both.filter { $0.group == .hands }.map(\.id).sorted()
                == ["left_grip", "left_gripper", "right_grip", "right_gripper"])
    }

    @Test func gripperSideShowsOneSlider() {
        // the slider panels hide the 0…1 grip on a gripper side: ONE control, closed (0) left … open right
        let g = gens(["left": "gripper", "right": "v2"])
        #expect(RobotModel.isRedundantGrip("left_grip", handGen: g))
        #expect(!RobotModel.isRedundantGrip("right_grip", handGen: g))           // an ORCA side keeps its grip
        #expect(!RobotModel.isRedundantGrip("left_gripper", handGen: g))
        let left = RobotModel.joints(handGen: g)
            .filter { $0.group == .hands && $0.id.hasPrefix("left") && !RobotModel.isRedundantGrip($0.id, handGen: g) }
        #expect(left.map(\.id) == ["left_gripper"])
        #expect(left[0].lower == 0 && left[0].upper == RobotModel.gripperMax)   // 0 = closed = slider's left end
    }

    @Test func gripperOpeningFollowsTheGatewaysGripMap() {
        // opening = 0.044·(1 − grip): 0 = open, 1 = closed, clamped
        #expect(RobotModel.gripperOpening(grip: 0) == 0.044)
        #expect(RobotModel.gripperOpening(grip: 1) == 0)
        #expect(abs(RobotModel.gripperOpening(grip: 0.25) - 0.033) < 1e-12)
        #expect(RobotModel.gripperOpening(grip: 2) == 0 && RobotModel.gripperOpening(grip: -1) == 0.044)
        // the echo keeps grip and opening consistent in both directions (ORCA sides untouched)
        var jt: [String: Double] = [:]
        RobotController.echoGripper(&jt, "left_grip", 0.5, isGripper: { $0 == "left" })
        #expect(abs((jt["left_gripper"] ?? -1) - 0.022) < 1e-12)
        RobotController.echoGripper(&jt, "right_grip", 0.5, isGripper: { $0 == "left" })
        #expect(jt["right_gripper"] == nil)                                 // an ORCA side: no opening
        RobotController.echoGripper(&jt, "left_gripper", 0.011)
        #expect(abs((jt["left_grip"] ?? -1) - 0.75) < 1e-12)
    }

    @Test func gripperDrawsFromTheWireThenFromTheGrip() {
        #expect(RobotRig.jointValue("left_gripper", in: ["left_gripper": 0.02]) == 0.02)
        // the mimic finger reads its LEADER's value — it has none of its own on the wire
        #expect(RobotRig.jointValue("left_gripper_mirror", mimic: "left_gripper", in: ["left_gripper": 0.02]) == 0.02)
        #expect(RobotRig.jointValue("left_gripper_mirror", in: ["left_gripper": 0.02]) == nil)
        // no opening in the pose (timeline preview / ghost): the side's grip drives it…
        #expect(RobotRig.jointValue("left_gripper_mirror", mimic: "left_gripper", in: ["left_grip": 1]) == 0)
        // …but a streamed opening always wins
        #expect(RobotRig.jointValue("left_gripper", in: ["left_grip": 0, "left_gripper": 0.01]) == 0.01)
        #expect(RobotRig.jointValue("left_gripper", in: [:]) == nil)
        #expect(RobotRig.jointValue("left_arm_1", in: ["left_grip": 0.5]) == nil)
    }

    @Test func gripperOverlayReplacesTheHandUnderTheMount() throws {
        let url = try #require(Bundle.main.url(forResource: "nova_rig", withExtension: "json"))
        let base = try JSONDecoder().decode(F.self, from: Data(contentsOf: url)).groups
        let out = RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                            nova: true, gen: gens(["left": "gripper", "right": "v2"]))
        let byName = Dictionary(out.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(byName.count == out.count)
        func under(_ m: String, _ gs: [G], _ idx: [String: G]) -> [G] {
            gs.filter { g in
                var p = g.parent
                while let q = p { if q == m { return true }; p = idx[q]?.parent }
                return false
            }
        }
        let left = under("l_hand_mount", out, byName)
        let ovURL = try #require(Bundle.main.url(forResource: "nova_gripper_v1_left", withExtension: "json"))
        let ov = try JSONDecoder().decode(F.self, from: Data(contentsOf: ovURL)).groups.map(\.name)
        #expect(Set(left.map(\.name)) == Set(ov).subtracting(["l_hand_mount"]))   // left = the gripper overlay
        // the overlay restates the mount without the ORCA arm-to-hand adapter (OpenArm's link7 replaces it),
        // in place: still the J7 joint, still ahead of its children
        let mount = try #require(byName["l_hand_mount"])
        #expect(mount.joint == "left_arm_7" && (mount.parts ?? []).isEmpty)
        #expect(out.firstIndex { $0.name == "l_hand_mount" }! < out.firstIndex { $0.joint == "left_gripper" }!)
        #expect(byName["left_gripper_base"]?.parts?.count == 4)            // link7 housing + motor, hand x2
        #expect(left.compactMap(\.joint).sorted() == ["left_gripper", "left_gripper_mirror"])
        let f1 = try #require(left.first { $0.joint == "left_gripper" })
        let f2 = try #require(left.first { $0.joint == "left_gripper_mirror" })
        #expect(f1.type == "prismatic" && f1.mimic == nil)
        #expect(f2.type == "prismatic" && f2.mimic == "left_gripper")      // the mimic field parsed
        // no ORCA geometry left on that side
        let orca = ["palm", "tower", "thumb", "index", "middle", "ring", "pinky"]
        #expect(!left.contains { g in orca.contains { g.name.contains($0) } })
        let baseIdx = Dictionary(base.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(Set(under("r_hand_mount", out, byName).map(\.name))
                == Set(under("r_hand_mount", base, baseIdx).map(\.name)))   // right untouched
        // both sides: two fingers each, and both overlays ship in the bundle
        let both = RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                             nova: true, gen: gens(["left": "gripper", "right": "gripper"]))
        #expect(both.filter { $0.mimic != nil }.map(\.joint) == ["left_gripper_mirror", "right_gripper_mirror"])
        #expect(both.compactMap(\.joint).filter { $0.contains("thumb") }.isEmpty)
    }

    @Test func clipHandStateDrivesTheGripperDof() {
        let g: (String) -> Bool = { $0 == "left" }
        // fingers fully curled -> closed; the ORCA joints of that side are gone, the other side untouched
        var a: [String: Double] = ["left_index_mcp": 1.6581, "left_index_pip": 1.885, "left_wrist": 0.3,
                                   "left_arm_1": 0.2, "right_index_mcp": 1.0, "right_grip": 0.5]
        RobotModel.mapHandToGripper(&a, isGripper: g)
        #expect(a["left_gripper"] == 0 && a["left_index_mcp"] == nil && a["left_wrist"] == nil)
        #expect(a["left_arm_1"] == 0.2 && a["right_index_mcp"] == 1.0 && a["right_gripper"] == nil)
        // open fingers -> fully open; half curl -> half open
        var b: [String: Double] = ["left_index_mcp": 0, "left_middle_mcp": 0]
        RobotModel.mapHandToGripper(&b, isGripper: g)
        #expect(b["left_gripper"] == RobotModel.gripperMax)
        var c: [String: Double] = ["left_index_mcp": 1.6581 / 2]
        RobotModel.mapHandToGripper(&c, isGripper: g)
        #expect(abs((c["left_gripper"] ?? -1) - RobotModel.gripperMax / 2) < 1e-12)
        // an explicit grip wins when it closes more than the fingers
        var d: [String: Double] = ["left_grip": 1, "left_index_mcp": 0]
        RobotModel.mapHandToGripper(&d, isGripper: g)
        #expect(d["left_gripper"] == 0)
        // an explicit opening is kept; no hand state -> nothing invented
        var e: [String: Double] = ["left_gripper": 0.01, "left_index_mcp": 1.6]
        RobotModel.mapHandToGripper(&e, isGripper: g)
        #expect(e["left_gripper"] == 0.01)
        var f: [String: Double] = ["left_arm_1": 0.1]
        RobotModel.mapHandToGripper(&f, isGripper: g)
        #expect(f == ["left_arm_1": 0.1])
    }

    @Test func greenBallSitsOnThePalmOrTheGripperTcp() {
        #expect(RobotRig.handlePoint("left", gen: "v1") == SIMD3<Float>(0, -0.255, -0.01))
        #expect(RobotRig.handlePoint("right", gen: "gripper") == SIMD3<Float>(0, 0.1801, -0.02))
        #expect(RobotRig.handlePoint("left", gen: "gripper") == SIMD3<Float>(0, -0.1801, -0.02))
        #expect(RobotRig.palmName("left", gen: "gripper") == "left_gripper_base")
        #expect(RobotRig.palmName("right", gen: "v2") == "right_palm")
    }

    @Test func mabelWearsTheLabsGripperOnJ7() throws {
        // the Retarget Lab's End effector = gripper: MABEL's J7 node loses the ORCA adapter + hand, gains
        // OpenArm's link7 (housing + gripper motor), the gripper hand and two fingers (export_gripper_rig.py)
        let url = try #require(Bundle.main.url(forResource: "rig", withExtension: "json"))
        let base = try JSONDecoder().decode(F.self, from: Data(contentsOf: url)).groups
        let out = RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                            nova: false, gen: gens(["left": "gripper"], nova: false))
        let idx = Dictionary(out.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(idx.count == out.count)
        func under(_ m: String, _ gs: [G], _ ix: [String: G]) -> [String] {
            gs.filter { g in
                var p = g.parent
                while let q = p { if q == m { return true }; p = ix[q]?.parent }
                return false
            }.map(\.name)
        }
        let left = under("l_hand_mount", out, idx)
        #expect(Set(left) == ["left_openarm_link7", "left_gripper_base", "left_gripper_finger1", "left_gripper_finger2"])
        #expect(out.compactMap(\.joint).filter { $0.hasPrefix("left_") && $0.contains("gripper") }.sorted()
                == ["left_gripper", "left_gripper_mirror"])
        #expect(idx["left_gripper_finger2"]?.mimic == "left_gripper")
        #expect(idx["l_hand_mount"]?.joint == "left_arm_7")                // J7 itself stays
        let bIdx = Dictionary(base.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        #expect(Set(under("r_hand_mount", out, idx)) == Set(under("r_hand_mount", base, bIdx)))   // right untouched
        // every part the overlay names ships in the bundle
        for g in out where g.name.contains("gripper") || g.name.contains("openarm") {
            for p in g.parts ?? [] {
                #expect(Bundle.main.url(forResource: String(p.file.dropLast(4)), withExtension: "ply") != nil)
            }
        }
        // ORCA (no gripper reported) is the plain rig
        #expect(RobotRig.withMountedHands(base, name: { $0.name }, parent: { $0.parent },
                                          nova: false, gen: gens([:], nova: false)).count == base.count)
    }

    @Test func sceneKitRigMovesTheMimicFingerWithItsLeader() throws {
        let rig = LiveMapRig(parts: RealRobotScene.makeRobotNodeWithHandles(manifest: "nova_rig", gen: { _ in "gripper" }))
        let f1 = try #require(rig.robotJoints["left_gripper"])
        let f2 = try #require(rig.robotJoints["left_gripper_mirror"])
        let r2 = try #require(rig.robotJoints["right_gripper_mirror"])
        #expect(f2.mimic == "left_gripper" && f1.mimic == nil)
        #expect(rig.robotJoints["left_thumb_abd"] == nil && rig.robotJoints["right_wrist"] == nil)
        #expect(f1.node.childNodes.contains { $0.geometry != nil })           // the finger meshes loaded
        #expect(f2.node.childNodes.contains { $0.geometry != nil })
        rig.apply(joints: ["left_gripper": 0.03, "right_gripper": 0.01])
        // finger 1 slides along −y, finger 2 (no wire value) along its own +y by the SAME amount
        #expect(simd_distance(f1.node.simdPosition, simd_float3(0, -0.03, 0)) < 1e-6)
        #expect(simd_distance(f2.node.simdPosition, simd_float3(0, 0.03, 0)) < 1e-6)
        #expect(simd_distance(r2.node.simdPosition, simd_float3(0, 0.01, 0)) < 1e-6)
        // a pose with only the 0…1 grip (preview / echo) still opens both fingers
        rig.apply(joints: ["left_grip": 0.5])
        #expect(abs(f1.node.simdPosition.y + 0.022) < 1e-6 && abs(f2.node.simdPosition.y - 0.022) < 1e-6)
    }
}
