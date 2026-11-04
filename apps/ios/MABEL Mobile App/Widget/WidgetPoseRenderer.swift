//
//  WidgetPoseRenderer.swift
//  MABEL Mobile App
//
//  Renders the live MABEL pose to a still PNG for the Home/Lock-Screen widget.
//  WidgetKit can't run SceneKit, so the app renders the rig off-screen and drops
//  a picture in the shared App Group container; the widget just shows it.
//
//  Performance matters here — this used to do a full-quality snapshot (shadows,
//  HDR, bloom, a reflective floor, 4× MSAA, 560px) on the MAIN thread every few
//  seconds, which janked the app. This version:
//    • builds a LIGHTWEIGHT scene (just the robot + flat lights, no floor/shadow),
//    • renders on a background queue (the scene is confined to that queue),
//    • caches by a pose hash so an unchanged pose is never re-rendered.
//  The maths mirror RealRobotView.Coordinator.apply() so the still matches the
//  in-app cockpit.
//

import SceneKit
import Metal
import UIKit
import simd

final class WidgetPoseRenderer {
    static let shared = WidgetPoseRenderer()

    /// SceneKit graph mutation isn't thread-safe; everything here is confined to
    /// this one serial queue, so the @MainActor publisher never touches the scene.
    private let queue = DispatchQueue(label: "com.mabel.widget.pose", qos: .utility)

    private var renderer: SCNRenderer?
    private var robot: SCNNode?
    private var joints: [String: RigJoint] = [:]
    private var lastHash: Int = 0
    private var hasRendered = false

    static let fileName = "pose.png"

    /// Render `joints/heading/baseQuat` to `pose.png` (off the main thread). Skips
    /// the work when the pose is unchanged unless `force` is set. `completion`
    /// (called on `queue`) gets the file name on success, nil otherwise.
    func render(joints j: [String: Double],
                heading: Double,
                baseQuat: [Double],
                force: Bool = false,
                size: CGFloat = 440,
                completion: @escaping (String?) -> Void) {
        queue.async { [weak self] in
            guard let self, let (renderer, robot, rigJoints) = self.ensureBuilt() else {
                completion(nil); return
            }
            let key = Self.hash(j, heading, baseQuat)
            if !force, self.hasRendered, key == self.lastHash {
                completion(Self.fileName); return        // pose unchanged — reuse the PNG
            }

            Self.apply(j, heading: heading, baseQuat: baseQuat, root: robot, joints: rigJoints)

            let image = renderer.snapshot(atTime: 0,
                                          with: CGSize(width: size, height: size),
                                          antialiasingMode: .multisampling2X)
            guard let data = image.pngData(),
                  let url = MabelWidgetShared.fileURL(Self.fileName) else {
                completion(nil); return
            }
            do {
                try data.write(to: url, options: .atomic)
                self.lastHash = key; self.hasRendered = true
                completion(Self.fileName)
            } catch { completion(nil) }
        }
    }

    // MARK: - Lightweight scene (built once, on `queue`)

    private func ensureBuilt() -> (SCNRenderer, SCNNode, [String: RigJoint])? {
        if let renderer, let robot { return (renderer, robot, joints) }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }

        let scene = SCNScene()
        // Bake a subtle viewport gradient so the PNG reads as a "screen", not a
        // floating cut-out (transparent SCNRenderer snapshots are unreliable).
        scene.background.contents = UIColor(red: 0.10, green: 0.085, blue: 0.067, alpha: 1)

        let (node, rigJoints) = RealRobotScene.makeRobotNode()   // robot only — no floor/shadows
        scene.rootNode.addChildNode(node)

        // Flat, cheap studio lighting (no shadow maps, no HDR/bloom).
        let amb = SCNLight(); amb.type = .ambient; amb.intensity = 720
        amb.color = UIColor(white: 0.55, alpha: 1)
        scene.rootNode.addChildNode(node: amb, euler: SCNVector3Zero)
        let key = SCNLight(); key.type = .directional; key.intensity = 850; key.temperature = 6500
        scene.rootNode.addChildNode(node: key, euler: SCNVector3(-Float.pi / 3.2, Float.pi / 5, 0))
        let fill = SCNLight(); fill.type = .directional; fill.intensity = 300; fill.temperature = 7400
        scene.rootNode.addChildNode(node: fill, euler: SCNVector3(-Float.pi / 6, -Float.pi / 2.4, 0))

        // Orbit camera matching the cockpit's default framing.
        let orbit = SCNNode()
        orbit.position = SCNVector3(0, 0.62, 0)
        orbit.eulerAngles = SCNVector3(-0.14, 0.6, 0)
        let camNode = SCNNode()
        let cam = SCNCamera(); cam.fieldOfView = 36; cam.zNear = 0.02
        camNode.camera = cam
        camNode.position = SCNVector3(0, 0, 2.7)
        orbit.addChildNode(camNode)
        scene.rootNode.addChildNode(orbit)

        let r = SCNRenderer(device: device, options: nil)
        r.scene = scene
        r.pointOfView = camNode

        self.renderer = r
        self.robot = node
        self.joints = rigJoints
        return (r, node, rigJoints)
    }

    // MARK: - Pose maths (mirror RealRobotView.Coordinator.apply)

    private static func apply(_ j: [String: Double], heading: Double, baseQuat: [Double],
                              root: SCNNode, joints: [String: RigJoint]) {
        let isLevel = baseQuat.count == 4 && baseQuat[0] > 0.9999
            && abs(baseQuat[1]) < 1e-4 && abs(baseQuat[2]) < 1e-4 && abs(baseQuat[3]) < 1e-4
        if !isLevel, baseQuat.count == 4 {
            let qm = simd_quatf(ix: Float(baseQuat[1]), iy: Float(baseQuat[2]),
                                iz: Float(baseQuat[3]), r: Float(baseQuat[0]))
            let c = simd_quatf(angle: -.pi / 2, axis: simd_float3(1, 0, 0))
            root.simdOrientation = c * qm * c.inverse
        } else {
            root.simdOrientation = simd_quatf(angle: Float(j["base_heading"] ?? heading),
                                              axis: simd_float3(0, 1, 0))
        }
        for (id, joint) in joints {
            // Displacement from the ASSEMBLED pose = absolute qpos - MuJoCo ref, with a
            // joint missing from telemetry falling back to its own ref (the default
            // pose), never 0. See RigJoint.ref: the 14 ORCA hand joints and the 2
            // half-raised lift stages carry ref != 0.
            let value = Float(RobotRig.jointValue(id, mimic: joint.mimic, in: j) ?? Double(joint.ref)) - joint.ref
            if joint.type == "prismatic" {
                joint.node.simdPosition = joint.axis * value
            } else {
                joint.node.simdOrientation = simd_quatf(angle: value, axis: simd_normalize(joint.axis))
            }
        }
    }

    /// Quantised pose fingerprint — micro-jitter (< ~0.3°) doesn't trigger a re-render.
    private static func hash(_ j: [String: Double], _ heading: Double, _ quat: [Double]) -> Int {
        var h = Hasher()
        for key in j.keys.sorted() {
            h.combine(key)
            h.combine(Int(((j[key] ?? 0) * 200).rounded()))
        }
        h.combine(Int((heading * 200).rounded()))
        for q in quat { h.combine(Int((q * 200).rounded())) }
        return h.finalize()
    }
}

// MARK: - Small SceneKit helper

private extension SCNNode {
    func addChildNode(node light: SCNLight, euler: SCNVector3) {
        let n = SCNNode(); n.light = light; n.eulerAngles = euler
        addChildNode(n)
    }
}
