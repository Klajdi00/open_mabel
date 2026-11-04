//
//  PairingRobotView.swift
//  MabelTeleop
//
//  The robot, turning slowly on its own, for the proximity connect card — the way an
//  AirPods case tumbles in Apple's pairing sheet.
//
//  It reuses `RealRobotScene.makeRobotNode()`, so this is the SAME articulated rig the
//  teleop viewer drives, not a stand-in render. Nothing here is a picture of the robot;
//  it is the robot's own model, which is why it can adopt the live pose the moment a link
//  comes up.
//
//  The rotation is a SceneKit action on the model's own node rather than a SwiftUI
//  animation: it must keep turning smoothly while SwiftUI re-lays-out the card around it,
//  and it must not cost a view-tree invalidation per frame.
//

import SwiftUI
import SceneKit
import simd

struct PairingRobotView: UIViewRepresentable {

    /// Seconds per full turn. Slow — this is ambient motion, not a spinner.
    var period: Double = 14
    /// Live joint angles once connected; nil keeps the rest pose.
    var pose: [String: Double]?
    /// Pauses rotation for Reduce Motion.
    var animated: Bool = true

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        let scene = SCNScene()
        view.scene = scene
        view.backgroundColor = .clear
        view.isOpaque = false
        view.antialiasingMode = .multisampling4X
        view.allowsCameraControl = false
        view.rendersContinuously = true

        let (robot, joints) = RealRobotScene.makeRobotNode()
        context.coordinator.joints = joints

        // A spin pivot ABOVE the model root, so the robot turns about its own vertical
        // axis instead of orbiting whatever offset the rig root carries.
        //
        // The offset is MEASURED, not typed. It used to be a hand-tuned
        // (0.23, -0.95, 0), which framed the torso and cropped the mobile base
        // off the bottom of the card — and any change to the rig (the lift's
        // rest height moved 0.16 m in 2026-09) silently re-cropped it somewhere
        // else. Centring on the model's own bounding box cannot go stale.
        let box = Self.bounds(of: robot)
        let centre = SCNVector3((box.min.x + box.max.x) / 2,
                                (box.min.y + box.max.y) / 2,
                                (box.min.z + box.max.z) / 2)
        let spin = SCNNode()
        spin.addChildNode(robot)
        robot.position = SCNVector3(-centre.x, -centre.y, -centre.z)
        scene.rootNode.addChildNode(spin)
        context.coordinator.spin = spin

        if animated {
            spin.runAction(.repeatForever(
                .rotateBy(x: 0, y: .pi * 2, z: 0, duration: period)))
        }
        // Start turned to a three-quarter view: straight-on reads flat, and the
        // three-quarter is the angle the robot's shoulders actually read from.
        spin.eulerAngles.y = -0.6

        installLights(scene)

        // Frame the WHOLE robot, head to wheels, whatever the card's aspect ratio
        // is. `projectionDirection = .vertical` is the load-bearing line: the
        // card is wide and short, and SceneKit's default fits the LARGER
        // dimension, so a 30-degree field of view spent itself sideways and let
        // the base fall out of the bottom. Pinning the field of view to the
        // vertical makes the visible height exactly 2*d*tan(fov/2) — a number we
        // can solve for the model we measured.
        let height = Double(box.max.y - box.min.y)
        let distance = 3.35
        let margin = 1.12                       // a little air above the head
        let camera = SCNCamera()
        camera.projectionDirection = .vertical
        camera.fieldOfView = CGFloat(2 * atan((height * margin / 2) / distance) * 180 / .pi)
        camera.wantsHDR = true
        camera.bloomIntensity = 0
        camera.zNear = 0.01
        let camNode = SCNNode()
        camNode.camera = camera
        camNode.position = SCNVector3(0, 0, Float(distance))
        scene.rootNode.addChildNode(camNode)
        view.pointOfView = camNode

        return view
    }

    /// Union of every geometry bounding box in the subtree, in `root`'s space.
    /// `SCNNode.boundingBox` alone describes a node's OWN geometry, and the rig's
    /// root carries none — asking it returns an empty box.
    private static func bounds(of root: SCNNode) -> (min: SCNVector3, max: SCNVector3) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var found = false
        func walk(_ n: SCNNode) {
            if n.geometry != nil {
                let (a, b) = n.boundingBox
                for corner in [SCNVector3(a.x, a.y, a.z), SCNVector3(b.x, a.y, a.z),
                               SCNVector3(a.x, b.y, a.z), SCNVector3(b.x, b.y, a.z),
                               SCNVector3(a.x, a.y, b.z), SCNVector3(b.x, a.y, b.z),
                               SCNVector3(a.x, b.y, b.z), SCNVector3(b.x, b.y, b.z)] {
                    let p = root.convertPosition(corner, from: n)
                    let v = SIMD3<Float>(Float(p.x), Float(p.y), Float(p.z))
                    lo = simd_min(lo, v); hi = simd_max(hi, v)
                    found = true
                }
            }
            n.childNodes.forEach(walk)
        }
        walk(root)
        guard found else { return (SCNVector3(-0.4, 0, -0.4), SCNVector3(0.4, 1.6, 0.4)) }
        return (SCNVector3(lo.x, lo.y, lo.z), SCNVector3(hi.x, hi.y, hi.z))
    }

    func updateUIView(_ view: SCNView, context: Context) {
        guard let pose else { return }
        // Pose the rig without animating each value: the card may adopt the live pose
        // mid-spin, and SceneKit's implicit transaction would tween every joint at a
        // different rate and read as a twitch.
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.45
        for (name, joint) in context.coordinator.joints {
            // Through jointValue: the gripper's second finger mimics its leader.
            guard let value = RobotRig.jointValue(name, mimic: joint.mimic, in: pose) else { continue }
            let angle = Float(value) - joint.ref
            if joint.type == "prismatic" {
                joint.node.simdPosition = joint.axis * angle
            } else {
                joint.node.simdOrientation = simd_quatf(angle: angle,
                                                        axis: simd_normalize(joint.axis))
            }
        }
        SCNTransaction.commit()
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var spin: SCNNode?
        var joints: [String: SCNRigJoint] = [:]
    }

    /// Three-point studio lighting. The card sits on a blurred material of unknown
    /// brightness, so the model carries its own light and never depends on the backdrop.
    private func installLights(_ scene: SCNScene) {
        func light(_ type: SCNLight.LightType, _ intensity: CGFloat, _ temp: CGFloat,
                   at pos: SCNVector3) -> SCNNode {
            let l = SCNLight()
            l.type = type
            l.intensity = intensity
            l.temperature = temp
            if type == .directional { l.castsShadow = false }
            let n = SCNNode(); n.light = l; n.position = pos
            n.look(at: SCNVector3Zero)
            return n
        }
        scene.rootNode.addChildNode(light(.directional, 780, 6200, at: SCNVector3(2, 3, 3)))   // key
        scene.rootNode.addChildNode(light(.directional, 340, 7400, at: SCNVector3(-3, 1, 2)))  // fill
        scene.rootNode.addChildNode(light(.directional, 260, 5200, at: SCNVector3(0, 2, -4)))  // rim
        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 260
        ambient.temperature = 6500
        let a = SCNNode(); a.light = ambient
        scene.rootNode.addChildNode(a)
    }
}
