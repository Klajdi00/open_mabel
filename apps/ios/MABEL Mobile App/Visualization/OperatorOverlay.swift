//
//  OperatorOverlay.swift
//  MABEL Mobile App
//
//  VP tracking overlay (retarget debugging) — renders the operator's Vision
//  Pro tracking beside the robot in the 3D viewer, from the gateway's
//  `operator_points` message:
//   · RAW (amber): head + both full hand skeletons through nothing but the
//     fixed ARKit→robot axis relabel — true metric scale, floating at the
//     ARKit session origin, exactly the points the headset itself tracks.
//   · MAPPED (magenta = right, cyan = left): the retargeting engine's own
//     wrist target, elbow hint, and the hand skeleton carried rigidly around
//     the target — the byte-exact quantities the arm IK consumed. When
//     tracking is perfect the target ball sits ON the robot's wrist; any gap
//     is the true retargeting error.
//
//  FRAME-EXACT BY CONSTRUCTION: this node must be parented to the rig's
//  `base_link` node (rig.json's base_link carries the ONE Z-up→Y-up rotation
//  that places the robot from odom, and nothing else). Points then keep their
//  raw odom coordinates as LOCAL positions — no new frame math exists here to
//  get wrong. Mirrors the web Control Center's implementation (viewport3d.js
//  onOperatorPoints), verified by controller/tests/test_operator_overlay.py.
//

import SceneKit
import QuartzCore
import UIKit

/// Main-thread latch for the newest `operator_points` payload (the link
/// callback writes it; the render tick reads it). Stale data hides the
/// overlay rather than freezing a ghost skeleton on screen.
enum OperatorPointsLatch {
    static var payload: [String: Any]?
    static var stamp: CFTimeInterval = 0

    static func store(_ p: [String: Any]) {
        payload = p
        stamp = CACurrentMediaTime()
    }
}

final class OperatorOverlayNode: SCNNode {
    static let maxHandJoints = 25
    private static let staleAfter: CFTimeInterval = 0.7

    private struct SideNodes {
        let target: SCNNode
        let elbow: SCNNode
        let hand: [SCNNode]
    }

    private var sides: [String: SideNodes] = [:]
    private var rawHands: [String: [SCNNode]] = [:]
    private var rawHead = SCNNode()

    override init() {
        super.init()
        name = "operator_overlay"
        isHidden = true

        func ball(_ radius: CGFloat, _ rgb: UInt32, _ alpha: CGFloat) -> SCNNode {
            let g = SCNSphere(radius: radius)
            g.segmentCount = 10
            let m = SCNMaterial()
            m.diffuse.contents = UIColor(rgb: rgb).withAlphaComponent(alpha)
            m.lightingModel = .constant
            m.readsFromDepthBuffer = false
            g.materials = [m]
            let n = SCNNode(geometry: g)
            n.renderingOrder = 900
            n.isHidden = true
            addChildNode(n)
            return n
        }

        for (side, hue, pale) in [("right", UInt32(0xFF4DD2), UInt32(0xFFB3EA)),
                                  ("left", UInt32(0x4DD2FF), UInt32(0xB3ECFF))] {
            sides[side] = SideNodes(
                target: ball(0.024, hue, 0.85),
                elbow: ball(0.014, hue, 0.85),
                hand: (0..<Self.maxHandJoints).map { _ in ball(0.007, pale, 0.9) })
            rawHands[side] = (0..<Self.maxHandJoints).map { _ in ball(0.006, 0xF0B68B, 0.8) }
        }
        rawHead = ball(0.05, 0xF0B68B, 0.6)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) unsupported") }

    /// Call from the render tick: applies the latched payload, hides on stale.
    func refresh(now: CFTimeInterval = CACurrentMediaTime()) {
        guard let p = OperatorPointsLatch.payload,
              now - OperatorPointsLatch.stamp < Self.staleAfter else {
            isHidden = true
            return
        }
        isHidden = false
        apply(p)
        OperatorPointsLatch.payload = nil          // applied — wait for the next frame
    }

    private static func vec(_ any: Any?) -> SCNVector3? {
        guard let l = any as? [Any], l.count >= 3,
              let x = (l[0] as? NSNumber)?.floatValue,
              let y = (l[1] as? NSNumber)?.floatValue,
              let z = (l[2] as? NSNumber)?.floatValue else { return nil }
        return SCNVector3(x, y, z)                 // raw odom coords — the parent
    }                                              // (base_link) rotates them

    private func place(_ nodes: [SCNNode], _ pts: [Any]?) {
        let list = pts ?? []
        for (i, n) in nodes.enumerated() {
            if i < list.count, let v = Self.vec(list[i]) {
                n.position = v
                n.isHidden = false
            } else {
                n.isHidden = true
            }
        }
    }

    private func apply(_ p: [String: Any]) {
        let sidesP = p["sides"] as? [String: [String: Any]] ?? [:]
        for (side, n) in sides {
            guard let d = sidesP[side] else {
                n.target.isHidden = true
                n.elbow.isHidden = true
                n.hand.forEach { $0.isHidden = true }
                continue
            }
            if let t = Self.vec(d["target"]) { n.target.position = t; n.target.isHidden = false }
            else { n.target.isHidden = true }
            if let e = Self.vec(d["elbow"]) { n.elbow.position = e; n.elbow.isHidden = false }
            else { n.elbow.isHidden = true }
            place(n.hand, d["hand"] as? [Any])
        }
        let raw = p["raw"] as? [String: Any] ?? [:]
        if let h = Self.vec(raw["head"]) { rawHead.position = h; rawHead.isHidden = false }
        else { rawHead.isHidden = true }
        for (side, nodes) in rawHands {
            place(nodes, raw[side] as? [Any])
        }
    }
}
