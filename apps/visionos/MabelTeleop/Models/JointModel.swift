//
//  JointModel.swift
//  MABEL Mobile App
//
//  MABEL's kinematic vocabulary. These definitions mirror
//  web_gui/shared_assets/mabel_joints.json — the same joint names, groups and
//  limits the web visualizer and MuJoCo model use — so sim, web and
//  mobile stay aligned.
//

import Foundation

/// Functional group a joint belongs to. Drives grouping in the UI.
enum JointGroup: String, CaseIterable, Identifiable, Codable {
    case base, body, arms, hands
    var id: String { rawValue }

    var label: String {
        switch self {
        case .base:  return "Swerve Base"
        case .body:  return "Lift & Neck"
        case .arms:  return "Arms"
        case .hands: return "Hands"
        }
    }
}

enum JointKind: String, Codable {
    case revolute, prismatic, continuous
}

/// Static definition of a controllable degree of freedom.
struct JointSpec: Identifiable, Hashable {
    let id: String          // canonical joint name
    let label: String       // human-friendly label
    let group: JointGroup
    let kind: JointKind
    let lower: Double
    let upper: Double
    var unitOverride: String? = nil

    var range: ClosedRange<Double> { lower...upper }
    /// Display units for the slider read-out.
    var unit: String { unitOverride ?? (kind == .prismatic ? "m" : "rad") }
}

/// The subset of MABEL's DOFs we expose for direct manipulation on the
/// phone. The full robot has 59 joints (incl. fingers); the mobile app
/// focuses on the meaningful operator controls.
enum RobotModel {
    private static let bodyJoints: [JointSpec] = [
        // Lift column + torso (the cinematic height/tilt axes)
        JointSpec(id: "lift_lower", label: "Lower Lift",  group: .body, kind: .prismatic, lower: 0,      upper: 0.3175),
        JointSpec(id: "lift_upper", label: "Upper Lift",  group: .body, kind: .prismatic, lower: 0,      upper: 0.3175),
        JointSpec(id: "torso",      label: "Torso Pitch", group: .body, kind: .revolute,  lower: -1.0472, upper: 0.2618),
        // Neck (camera head)
        JointSpec(id: "neck_yaw",   label: "Neck Yaw",    group: .body, kind: .revolute,  lower: -1.5708, upper: 1.5708),
        JointSpec(id: "neck_pitch", label: "Neck Pitch",  group: .body, kind: .revolute,  lower: -0.7854, upper: 0.7854),
        JointSpec(id: "neck_roll",  label: "Neck Roll",   group: .body, kind: .revolute,  lower: -0.7854, upper: 0.7854),
        // Left arm (7-DOF, shoulder → wrist)
        JointSpec(id: "left_arm_1", label: "L · Shoulder Pitch", group: .arms, kind: .revolute, lower: -3.49066, upper: 1.39626),
        JointSpec(id: "left_arm_2", label: "L · Shoulder Roll",  group: .arms, kind: .revolute, lower: -0.1745,  upper: 3.3161),
        JointSpec(id: "left_arm_3", label: "L · Upper Yaw",      group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        JointSpec(id: "left_arm_4", label: "L · Elbow",          group: .arms, kind: .revolute, lower: -0.943,   upper: 1.542),
        JointSpec(id: "left_arm_5", label: "L · Forearm Yaw",    group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        JointSpec(id: "left_arm_6", label: "L · Wrist Pitch",    group: .arms, kind: .revolute, lower: -0.7854,  upper: 0.7854),
        JointSpec(id: "left_arm_7", label: "L · Wrist Roll",     group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        // Right arm (7-DOF, mirrored)
        JointSpec(id: "right_arm_1", label: "R · Shoulder Pitch", group: .arms, kind: .revolute, lower: -1.39626, upper: 3.49066),
        JointSpec(id: "right_arm_2", label: "R · Shoulder Roll",  group: .arms, kind: .revolute, lower: -3.3161,  upper: 0.1745),
        JointSpec(id: "right_arm_3", label: "R · Upper Yaw",      group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        JointSpec(id: "right_arm_4", label: "R · Elbow",          group: .arms, kind: .revolute, lower: -1.670,   upper: 0.773),
        JointSpec(id: "right_arm_5", label: "R · Forearm Yaw",    group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        JointSpec(id: "right_arm_6", label: "R · Wrist Pitch",    group: .arms, kind: .revolute, lower: -0.7854,  upper: 0.7854),
        JointSpec(id: "right_arm_7", label: "R · Wrist Roll",     group: .arms, kind: .revolute, lower: -1.5708,  upper: 1.5708),
        // Swerve base
        JointSpec(id: "base_heading", label: "Base Heading", group: .base, kind: .revolute, lower: -3.1416, upper: 3.1416, unitOverride: "rad"),
        // Hands — quick parallel grip (0 = open, 1 = closed fist).
        JointSpec(id: "left_grip",  label: "L Grip",  group: .hands, kind: .prismatic, lower: 0, upper: 1, unitOverride: ""),
        JointSpec(id: "right_grip", label: "R Grip", group: .hands, kind: .prismatic, lower: 0, upper: 1, unitOverride: ""),
    ]

    /// Body + arms + base + grips + the fingers of the hand MOUNTED on each side
    /// (`RobotRig.handGen`). The ORCA hands swap between MABEL and NOVA, and a v1 hand
    /// speaks MABEL's frame (MuJoCo qpos, MJCF limits) whichever robot wears it. A side
    /// wearing the OpenArm gripper has no finger joints — just `<side>_gripper`.
    static var joints: [JointSpec] {
        let sig = RobotRig.signature
        jointsLock.lock(); defer { jointsLock.unlock() }
        if let c = jointsCache, c.sig == sig { return c.specs }
        let all = joints(handGen: { RobotRig.handGen($0) })
        jointsCache = (sig, all)
        return all
    }
    /// The joint list for an arbitrary per-side hand pick (pure — what the tests drive).
    static func joints(handGen: (String) -> String) -> [JointSpec] {
        bodyJoints + ["left", "right"].flatMap { side -> [JointSpec] in
            let set: [JointSpec]
            switch handGen(side) {
            case "v1":      set = fingerJoints
            case "gripper": set = gripperJoints
            default:        set = novaFingerJoints
            }
            return set.filter { $0.id.hasPrefix(side + "_") }
        }
    }
    private static var jointsCache: (sig: String, specs: [JointSpec])?
    private static let jointsLock = NSLock()

    /// The 17-DOF Orca hand per side (16 finger joints + wrist), limits from
    /// the URDF (web_gui/shared_assets/mabel_joints.json). Generated so both hands stay
    /// in sync. Side mirroring only flips the wrist range.
    static let fingerJoints: [JointSpec] = {
        // (suffix, label, lower, upper)
        // Limits = the hand actuators' ctrlrange (models/*_hand.xml) — the real
        // control source, so the sliders span exactly what the robot accepts.
        let dofs: [(String, String, Double, Double)] = [
            ("thumb_mcp", "Thumb MCP", -0.8727, 0.8727), ("thumb_abd", "Thumb Abd", -1.0821, 0.0),
            ("thumb_pip", "Thumb PIP", -0.7944, 1.23), ("thumb_dip", "Thumb DIP", -0.8538, 1.45),
            ("index_abd", "Index Abd", -1.0458, 0.2458), ("index_mcp", "Index MCP", -0.3491, 1.6581),
            ("index_pip", "Index PIP", -0.3491, 1.885),
            ("middle_abd", "Middle Abd", -0.6458, 0.6458), ("middle_mcp", "Middle MCP", -0.3491, 1.5882),
            ("middle_pip", "Middle PIP", -0.3491, 1.8675),
            ("ring_abd", "Ring Abd", -0.4758, 0.8058), ("ring_mcp", "Ring MCP", -0.3491, 1.5882),
            ("ring_pip", "Ring PIP", -0.3491, 1.8675),
            ("pinky_abd", "Pinky Abd", -0.1224, 1.1691), ("pinky_mcp", "Pinky MCP", -0.3491, 1.7104),
            ("pinky_pip", "Pinky PIP", -0.3491, 1.885),
        ]
        var out: [JointSpec] = []
        for (pre, P) in [("left", "L"), ("right", "R")] {
            let wl = pre == "left" ? (-0.8972, 0.6736) : (-0.6736, 0.8972)
            out.append(JointSpec(id: "\(pre)_wrist", label: "\(P) Wrist", group: .hands, kind: .revolute, lower: wl.0, upper: wl.1))
            for (suf, lbl, lo, hi) in dofs {
                out.append(JointSpec(id: "\(pre)_\(suf)", label: "\(P) \(lbl)", group: .hands, kind: .revolute, lower: lo, upper: hi))
            }
        }
        return out
    }()

    /// NOVA's ORCA hand v2: the joint limits of the upstream v2 URDF (orcahand_description v2,
    /// identical on both sides, wrist included — no mirroring), in the frame the NOVA gateway
    /// speaks (URDF angles: no MJCF ref). Same names, same thumb map (thumb_mcp = ORCA cmc,
    /// thumb_pip = middle flex, thumb_dip = distal).
    static let novaFingerJoints: [JointSpec] = {
        let dofs: [(String, String, Double, Double)] = [
            ("wrist", "Wrist", -1.134464, 0.610865),
            ("thumb_mcp", "Thumb MCP", -0.785398, 0.575959), ("thumb_abd", "Thumb Abd", -0.314159, 0.959931),
            ("thumb_pip", "Thumb PIP", -0.436332, 1.745329), ("thumb_dip", "Thumb DIP", -0.261799, 1.867502),
            ("index_abd", "Index Abd", -0.436332, 0.523599), ("index_mcp", "Index MCP", -0.436332, 1.745329),
            ("index_pip", "Index PIP", -0.261799, 1.867502),
            ("middle_abd", "Middle Abd", -0.471239, 0.471239), ("middle_mcp", "Middle MCP", -0.436332, 1.745329),
            ("middle_pip", "Middle PIP", -0.261799, 1.867502),
            ("ring_abd", "Ring Abd", -0.471239, 0.471239), ("ring_mcp", "Ring MCP", -0.436332, 1.745329),
            ("ring_pip", "Ring PIP", -0.261799, 1.867502),
            ("pinky_abd", "Pinky Abd", -0.523599, 0.523599), ("pinky_mcp", "Pinky MCP", -0.436332, 1.745329),
            ("pinky_pip", "Pinky PIP", -0.261799, 1.867502),
        ]
        var out: [JointSpec] = []
        for (pre, P) in [("left", "L"), ("right", "R")] {
            for (suf, lbl, lo, hi) in dofs {
                out.append(JointSpec(id: "\(pre)_\(suf)", label: "\(P) \(lbl)", group: .hands, kind: .revolute, lower: lo, upper: hi))
            }
        }
        return out
    }()

    /// The OpenArm v1 parallel gripper (NOVA, either side, `hands` side = "gripper"): ONE finger's
    /// opening in metres, 0 = closed … `gripperMax` = fully open — the URDF prismatic joint frame
    /// (ref 0, drawn raw). The second finger mimics it, so it is not a joint of its own here.
    static let gripperMax = 0.044
    static let gripperJoints: [JointSpec] = [
        JointSpec(id: "left_gripper",  label: "L Gripper", group: .hands, kind: .prismatic, lower: 0, upper: gripperMax),
        JointSpec(id: "right_gripper", label: "R Gripper", group: .hands, kind: .prismatic, lower: 0, upper: gripperMax),
    ]
    /// The gateway's map from the 0…1 `<side>_grip` (0 open, 1 closed) onto the gripper opening.
    static func gripperOpening(grip: Double) -> Double { gripperMax * (1 - min(1, max(0, grip))) }

    /// The ORCA finger joints whose flexion reads as hand CLOSURE (positive = curling in), with the upper limit
    /// that counts as fully curled: the MCP + PIP of the four fingers and the thumb's PIP + DIP.
    static let closureJoints: [(String, Double)] = [
        ("index_mcp", 1.6581), ("index_pip", 1.885), ("middle_mcp", 1.5882), ("middle_pip", 1.8675),
        ("ring_mcp", 1.5882), ("ring_pip", 1.8675), ("pinky_mcp", 1.7104), ("pinky_pip", 1.885),
        ("thumb_pip", 1.23), ("thumb_dip", 1.45),
    ]
    /// A hand pose (an animation clip's ORCA fingers and/or its 0…1 `<side>_grip`) on a side that wears the
    /// OpenArm gripper becomes that gripper's ONE DOF: closure = the larger of the explicit grip and the mean
    /// finger curl (each joint's flexion over its full-curl limit, clamped 0…1), written as `<side>_gripper`
    /// (one finger's opening, `gripperOpening`), and the side's ORCA finger + wrist joints are dropped — the
    /// gripper has none, and the gateway takes an explicit `<side>_gripper` over the grip. A side whose pose
    /// already carries `<side>_gripper`, or that carries no hand state at all, is left alone.
    static func mapHandToGripper(_ j: inout [String: Double], isGripper: (String) -> Bool) {
        for side in ["left", "right"] where isGripper(side) {
            guard j["\(side)_gripper"] == nil else { continue }
            let curls = closureJoints.compactMap { (n, hi) in j["\(side)_\(n)"].map { min(1, max(0, $0 / hi)) } }
            let grip = j["\(side)_grip"]
            guard grip != nil || !curls.isEmpty else { continue }
            let curl = curls.isEmpty ? 0 : curls.reduce(0, +) / Double(curls.count)
            let closure = max(min(1, max(0, grip ?? 0)), curl)
            for k in Array(j.keys) where k.hasPrefix(side + "_") {
                let s = k.dropFirst(side.count + 1)
                if s == "wrist" || ["thumb_", "index_", "middle_", "ring_", "pinky_"].contains(where: { s.hasPrefix($0) }) {
                    j[k] = nil
                }
            }
            j["\(side)_gripper"] = gripperOpening(grip: closure)
        }
    }

    static func specs(in group: JointGroup) -> [JointSpec] {
        joints.filter { $0.group == group }
    }

    /// True for a `<side>_grip` whose side wears the OpenArm gripper. There the 0…1 grip and
    /// `<side>_gripper` are the SAME DOF (`echoGripper` mirrors them), so a slider panel shows only
    /// the gripper — one slider, left = closed (0) … right = open (`gripperMax`). The grip stays a
    /// joint: the wire, the clips and Home still speak it.
    static func isRedundantGrip(_ id: String, handGen: (String) -> String = { RobotRig.handGen($0) }) -> Bool {
        id.hasSuffix("_grip") && handGen(String(id.dropLast("_grip".count))) == "gripper"
    }

    static func spec(_ id: String) -> JointSpec? {
        joints.first { $0.id == id }
    }

    /// MJCF ref (qpos0) for the joints whose assembled/NEUTRAL pose is not 0 — exactly
    /// the 14 ORCA finger/wrist joints. The rig renders `qpos − ref`, so a level,
    /// neutral hand needs these ABSOLUTE values; using 0 renders the thumb splayed
    /// (~42°) and the wrist twisted. (Only the wrist differs L/R; the finger refs
    /// mirror.) Kept in sync with the MJCF — see the repo-root model pipeline.
    static var handNeutral: [String: Double] { handNeutral(handGen: { RobotRig.handGen($0) }) }
    static func handNeutral(handGen: (String) -> String) -> [String: Double] {
        var n = novaNeutral                              // the lift (both robots)
        for side in ["left", "right"] where handGen(side) == "v1" {
            for (k, v) in mabelNeutral where k.hasPrefix(side + "_") { n[k] = v }
        }
        return n
    }
    /// NOVA's v2 hand: the URDF frame has no assembled-pose offsets — a level hand IS zero — so
    /// only the lift's half-raised ref remains. A v1 hand (MABEL's, on either robot) adds its
    /// side's MJCF qpos0 refs from `mabelNeutral`. The gripper's frame has no offset either (ref 0).
    private static let novaNeutral: [String: Double] = ["lift_lower": 0.15875, "lift_upper": 0.15875]
    private static let mabelNeutral: [String: Double] = [
        "left_wrist": 0.15, "right_wrist": -0.15,
        "left_thumb_abd": -0.733, "right_thumb_abd": -0.733,
        "left_thumb_pip": -0.585, "right_thumb_pip": -0.585,
        "left_thumb_dip": -0.5048, "right_thumb_dip": -0.5048,
        "left_index_abd": -0.40, "right_index_abd": -0.40,
        "left_ring_abd": 0.17, "right_ring_abd": 0.17,
        "left_pinky_abd": 0.5233, "right_pinky_abd": 0.5233,
        // The LIFT is assembled half-raised in the MJCF (ref 0.15875 m per stage =
        // 0.3175 m of the 0.635 m stroke, matching SAFE_HOME_LIFT). It is a
        // ref-carrying joint exactly like the hand joints above, so home must be its
        // ref -- seeding 0 parks the robot bottomed-out, which is what the offline /
        // demo model used to show.
        "lift_lower": 0.15875, "lift_upper": 0.15875,
    ]

    /// The ARMS at rest, in ABSOLUTE joint units. Identical to
    /// `RobotController.restBodyPose`, to `REST` in the studio's motions.js and to
    /// `REST_POSE` in its homepose.js — ONE rest pose across every client.
    ///
    /// This is the fix for the thing everyone kept seeing and nobody could name:
    /// qpos ZERO is not the robot at rest. At qpos zero the upper arm hangs but the
    /// forearm juts straight out in front — a 90° BENT ELBOW (measured 91.3°),
    /// costing 38.6 Nm of holding torque. The arm only hangs when the elbow is
    /// EXTENDED, which is ±1.52, not 0. Every viewer that started from a
    /// zero-filled dictionary drew the bent-elbow robot.
    static let armsAtRest: [String: Double] = [
        "left_arm_1": 0, "left_arm_2": -0.1711, "left_arm_3": 0, "left_arm_4": 1.52,
        "left_arm_5": 0, "left_arm_6": 0, "left_arm_7": 0,
        "right_arm_1": 0, "right_arm_2": 0.1711, "right_arm_3": 0, "right_arm_4": -1.52,
        "right_arm_5": 0, "right_arm_6": 0, "right_arm_7": 0,
        "left_grip": 0.2, "right_grip": 0.2,           // a hand at rest is not splayed
    ]

    /// The robot's rest pose in ABSOLUTE joint units: `armsAtRest` for the arms and
    /// hands, the ref-carrying joints at their ref (see `handNeutral`: the 14 ORCA
    /// hand joints plus the 2 lift stages), everything else at 0. Snapping the model
    /// here on reset must land a NEUTRAL hand rather than a contorted one, a
    /// HALF-RAISED lift rather than a bottomed one, and an EXTENDED elbow rather
    /// than a bent one.
    static var homePose: [String: Double] {
        var p = [String: Double]()
        for j in joints { p[j.id] = 0 }
        for (k, v) in handNeutral { p[k] = v }
        for (k, v) in armsAtRest { p[k] = v }
        // A gripper rests where its side's rest grip puts it (the gateway's map), not shut.
        for side in ["left", "right"] where p["\(side)_gripper"] != nil {
            p["\(side)_gripper"] = gripperOpening(grip: p["\(side)_grip"] ?? 0)
        }
        return p
    }
}
