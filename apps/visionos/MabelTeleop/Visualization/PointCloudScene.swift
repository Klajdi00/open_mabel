//
//  PointCloudScene.swift
//  MABEL Mobile App
//
//  Builds an inspectable 3D point cloud for a SLAM room. Points are
//  generated procedurally per room (floor, walls, furniture clusters)
//  with warm palette colours, then packed into a single SCNGeometry of
//  `.point` primitives for fast rendering.
//

import SceneKit
import UIKit

/// Navigation handles for a rendered room: where to hit-test goals, where
/// to attach the planned path, and the obstacle footprints A* must avoid.
struct SCNRoomNav {
    let floor: SCNNode
    let pathContainer: SCNNode
    let robot: SCNNode                       // the real MABEL that follows the path
    let robotJoints: [String: SCNRigJoint]      // incl. swerve steer/drive joints
    let obstacles: [CGRect]                  // XZ footprints, metres, room frame
    let bounds: CGSize                       // XZ extent
    let camera: SCNNode                      // the viewport camera (recenter/fit/follow)
    let cloud: SCNNode                       // the room point cloud (for fit bounds)
    let camHome: SCNVector3                  // default camera position (recenter target)
    let camLook: SCNVector3                  // default look-at point
}

enum PointCloudScene {

    static func make(for room: SLAMRoom) -> (SCNScene, SCNRoomNav) {
        let scene = SCNScene()
        scene.background.contents = UIColor(rgb: 0x0A0908)

        var rng = SplitMix(seed: room.seed)
        var pts = [SCNVector3]()
        var cols = [SCNVector3]()
        generate(room: room, rng: &rng, pts: &pts, cols: &cols)

        let node = pointNode(points: pts, colors: cols)
        node.position = SCNVector3(0, 0, 0)
        node.name = "cloud"
        scene.rootNode.addChildNode(node)

        // Soft ambient so the points carry the image (they're constant-shaded).
        let amb = SCNLight(); amb.type = .ambient; amb.intensity = 650
        let an = SCNNode(); an.light = amb; scene.rootNode.addChildNode(an)

        // A key light so the PBR robot reads with form (the points ignore it).
        let key = SCNLight(); key.type = .directional; key.intensity = 750; key.temperature = 6500
        let kn = SCNNode(); kn.light = key; kn.eulerAngles = SCNVector3(-Float.pi/3, Float.pi/5, 0)
        scene.rootNode.addChildNode(kn)

        // Invisible floor for tap hit-testing (oversized so edge taps land).
        let s = room.size
        let floorGeo = SCNPlane(width: CGFloat(s.x) + 8, height: CGFloat(s.z) + 8)
        floorGeo.firstMaterial?.diffuse.contents = UIColor.clear
        floorGeo.firstMaterial?.isDoubleSided = true
        let floor = SCNNode(geometry: floorGeo)
        floor.eulerAngles.x = -.pi / 2
        floor.name = "floor"
        scene.rootNode.addChildNode(floor)

        // The real articulated MABEL (same URDF rig as the rest of the app)
        // at the room origin — this is the robot that drives the path.
        let (robot, robotJoints) = RealRobotScene.makeRobotNode()
        scene.rootNode.addChildNode(robot)

        let pathContainer = SCNNode(); pathContainer.name = "path"
        scene.rootNode.addChildNode(pathContainer)

        let cam = SCNCamera()
        cam.fieldOfView = 50
        cam.zNear = 0.05; cam.zFar = 100
        cam.wantsHDR = true; cam.bloomIntensity = 0.12
        let camNode = SCNNode(); camNode.camera = cam
        camNode.name = "viewportCamera"
        let camHome = SCNVector3(s.x * 0.9, s.y * 1.2, s.z * 1.1)
        let camLook = SCNVector3(0, s.y * 0.2, 0)
        camNode.position = camHome
        camNode.look(at: camLook)
        scene.rootNode.addChildNode(camNode)

        let nav = SCNRoomNav(floor: floor, pathContainer: pathContainer, robot: robot,
                          robotJoints: robotJoints,
                          obstacles: footprints(for: room.kind, size: s),
                          bounds: CGSize(width: CGFloat(s.x), height: CGFloat(s.z)),
                          camera: camNode, cloud: node, camHome: camHome, camLook: camLook)
        return (scene, nav)
    }

    /// Footprints (XZ rects) of the furniture clusters, for the planner.
    private static func footprints(for kind: SLAMRoom.Kind, size s: SIMD3<Float>) -> [CGRect] {
        furniture(for: kind, size: s).map { b in
            CGRect(x: CGFloat(b.center.x - b.extent.x), y: CGFloat(b.center.z - b.extent.z),
                   width: CGFloat(b.extent.x * 2), height: CGFloat(b.extent.z * 2))
        }
    }


    /// Render an A* path (room-frame XZ waypoints) into the container.
    static func renderPath(_ waypoints: [CGPoint], goal: CGPoint, into container: SCNNode) {
        container.childNodes.forEach { $0.removeFromParentNode() }
        guard !waypoints.isEmpty else { return }

        // Path dots (green), raised slightly so they read above the cloud.
        for (i, w) in waypoints.enumerated() {
            let dot = SCNNode(geometry: SCNSphere(radius: 0.07))
            let mat = dot.geometry!.firstMaterial!
            mat.lightingModel = .constant
            mat.diffuse.contents = UIColor(rgb: 0x34D058)
            mat.emission.contents = UIColor(rgb: 0x34D058)
            dot.position = SCNVector3(Float(w.x), 0.08, Float(w.y))
            dot.opacity = (i == 0) ? 0.5 : 1
            container.addChildNode(dot)
        }

        // Goal: a bright pillar + ground ring so it's unmistakable.
        let pillar = SCNNode(geometry: SCNCylinder(radius: 0.04, height: 0.7))
        let pm = pillar.geometry!.firstMaterial!
        pm.lightingModel = .constant
        pm.diffuse.contents = UIColor(rgb: 0xE0691F)
        pm.emission.contents = UIColor(rgb: 0xE0691F)
        pillar.position = SCNVector3(Float(goal.x), 0.35, Float(goal.y))
        container.addChildNode(pillar)

        let ring = SCNNode(geometry: SCNTube(innerRadius: 0.18, outerRadius: 0.26, height: 0.02))
        let rm = ring.geometry!.firstMaterial!
        rm.lightingModel = .constant
        rm.diffuse.contents = UIColor(rgb: 0xE0691F)
        rm.emission.contents = UIColor(rgb: 0xE0691F)
        ring.position = SCNVector3(Float(goal.x), 0.02, Float(goal.y))
        container.addChildNode(ring)

        // Pulse the goal so it draws the eye.
        ring.runAction(.repeatForever(.sequence([
            .scale(to: 1.3, duration: 0.8), .scale(to: 1.0, duration: 0.8)
        ])))
    }

    // MARK: - Generation

    private static func generate(room: SLAMRoom, rng: inout SplitMix,
                                 pts: inout [SCNVector3], cols: inout [SCNVector3]) {
        let s = room.size
        let n = min(room.pointCount, 90_000)   // cap for the phone
        let bone   = SCNVector3(0.93, 0.90, 0.84)
        let ash    = SCNVector3(0.55, 0.52, 0.47)
        let rust   = SCNVector3(0.76, 0.36, 0.16)
        let clay   = SCNVector3(0.55, 0.35, 0.24)
        let dark   = SCNVector3(0.22, 0.20, 0.18)

        func jitter(_ a: Float) -> Float { (rng.unit() - 0.5) * a }
        func shade(_ c: SCNVector3, _ k: Float) -> SCNVector3 {
            SCNVector3(c.x * k, c.y * k, c.z * k)
        }

        // Floor
        for _ in 0..<(n / 3) {
            let x = (rng.unit() - 0.5) * s.x
            let z = (rng.unit() - 0.5) * s.z
            pts.append(SCNVector3(x, jitter(0.015), z))
            cols.append(shade(ash, 0.7 + rng.unit() * 0.5))
        }
        // Walls (4 planes) + ceiling hints
        for _ in 0..<(n / 3) {
            let wall = Int(rng.unit() * 4)
            let y = rng.unit() * s.y
            var x: Float = 0, z: Float = 0
            switch wall {
            case 0: x = -s.x/2 + jitter(0.02); z = (rng.unit() - 0.5) * s.z
            case 1: x =  s.x/2 + jitter(0.02); z = (rng.unit() - 0.5) * s.z
            case 2: z = -s.z/2 + jitter(0.02); x = (rng.unit() - 0.5) * s.x
            default: z = s.z/2 + jitter(0.02); x = (rng.unit() - 0.5) * s.x
            }
            pts.append(SCNVector3(x, y, z))
            cols.append(shade(bone, 0.7 + rng.unit() * 0.4))
        }
        // Furniture clusters (room-specific accents)
        let clusters = furniture(for: room.kind, size: s)
        let perCluster = (n / 3) / max(1, clusters.count)
        for (i, c) in clusters.enumerated() {
            let col = (i % 3 == 0) ? rust : (i % 3 == 1 ? clay : dark)
            for _ in 0..<perCluster {
                let p = SCNVector3(
                    c.center.x + jitter(c.extent.x),
                    c.center.y + jitter(c.extent.y),
                    c.center.z + jitter(c.extent.z)
                )
                pts.append(p)
                cols.append(shade(col, 0.75 + rng.unit() * 0.4))
            }
        }
    }

    private struct Box { var center: SCNVector3; var extent: SCNVector3 }

    private static func furniture(for kind: SLAMRoom.Kind, size s: SIMD3<Float>) -> [Box] {
        switch kind {
        case .studio:
            return [Box(center: .init(-s.x*0.25, 0.5, 0), extent: .init(0.6, 0.5, 1.2)),
                    Box(center: .init(s.x*0.3, 0.9, s.z*0.2), extent: .init(0.3, 0.9, 0.3)),
                    Box(center: .init(0, 0.2, -s.z*0.3), extent: .init(1.4, 0.2, 0.5))]
        case .livingRoom:
            return [Box(center: .init(-s.x*0.2, 0.4, -s.z*0.2), extent: .init(1.0, 0.4, 0.4)),
                    Box(center: .init(s.x*0.2, 0.25, 0), extent: .init(0.6, 0.25, 0.5)),
                    Box(center: .init(0, 0.6, s.z*0.35), extent: .init(0.8, 0.6, 0.2))]
        case .workshop:
            return [Box(center: .init(-s.x*0.3, 0.45, 0), extent: .init(0.4, 0.45, 1.6)),
                    Box(center: .init(s.x*0.25, 0.45, -s.z*0.2), extent: .init(0.5, 0.45, 0.5)),
                    Box(center: .init(s.x*0.3, 0.8, s.z*0.3), extent: .init(0.3, 0.5, 0.3))]
        case .corridor:
            return [Box(center: .init(-s.x*0.3, 0.5, 0), extent: .init(0.2, 0.5, 0.8)),
                    Box(center: .init(s.x*0.3, 0.5, 0), extent: .init(0.2, 0.5, 0.8))]
        }
    }

    // MARK: - Geometry packing

    /// Pack parallel position/colour arrays into a single `.point`-primitive
    /// SCNGeometry node. Shared with the live SLAM map (LiveMapScene), which
    /// streams its points in rather than synthesising them.
    static func pointNode(points: [SCNVector3], colors: [SCNVector3]) -> SCNNode {
        // SceneKit requires the vertex and colour sources to have equal counts; a
        // truncated/garbled mesh frame must never crash the renderer.
        guard !points.isEmpty, points.count == colors.count else { return SCNNode() }
        let vData = points.withUnsafeBufferPointer { Data(buffer: $0) }
        let vSource = SCNGeometrySource(
            data: vData, semantic: .vertex, vectorCount: points.count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SCNVector3>.stride
        )
        let cData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
        let cSource = SCNGeometrySource(
            data: cData, semantic: .color, vectorCount: colors.count,
            usesFloatComponents: true, componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<SCNVector3>.stride
        )

        let indices = [Int32](0..<Int32(points.count))
        let iData = indices.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(
            data: iData, primitiveType: .point,
            primitiveCount: points.count, bytesPerIndex: MemoryLayout<Int32>.size
        )
        element.pointSize = 4
        element.minimumPointScreenSpaceRadius = 1.6
        element.maximumPointScreenSpaceRadius = 5

        let geo = SCNGeometry(sources: [vSource, cSource], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        geo.materials = [mat]
        return SCNNode(geometry: geo)
    }

    /// Pack a triangle MESH (vertices + per-vertex colour + triangle indices) into
    /// a single shaded `.triangles` node — the true nvblox surface, not a cloud.
    /// Per-vertex normals are computed from the faces so the surfaces light up.
    static func meshNode(points: [SCNVector3], colors: [SCNVector3], indices: [Int32]) -> SCNNode {
        guard !points.isEmpty, points.count == colors.count, indices.count >= 3 else { return SCNNode() }
        let n = points.count

        // Area-weighted vertex normals (accumulate face normals at each corner).
        func s3(_ v: SCNVector3) -> simd_float3 { simd_float3(Float(v.x), Float(v.y), Float(v.z)) }
        var nrm = [simd_float3](repeating: .zero, count: n)
        var t = 0
        while t + 2 < indices.count {
            let a = Int(indices[t]), b = Int(indices[t + 1]), c = Int(indices[t + 2]); t += 3
            guard a >= 0, b >= 0, c >= 0, a < n, b < n, c < n else { continue }
            let va = s3(points[a]), vb = s3(points[b]), vc = s3(points[c])
            let fn = simd_cross(vb - va, vc - va)   // unnormalised → area-weighted
            nrm[a] += fn; nrm[b] += fn; nrm[c] += fn
        }
        let normals: [SCNVector3] = nrm.map { v in
            let len = simd_length(v)
            let u = len > 1e-6 ? v / len : simd_float3(0, 1, 0)
            return SCNVector3(u.x, u.y, u.z)
        }

        let vData = points.withUnsafeBufferPointer { Data(buffer: $0) }
        let vSource = SCNGeometrySource(
            data: vData, semantic: .vertex, vectorCount: n, usesFloatComponents: true,
            componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0, dataStride: MemoryLayout<SCNVector3>.stride)
        let nData = normals.withUnsafeBufferPointer { Data(buffer: $0) }
        let nSource = SCNGeometrySource(
            data: nData, semantic: .normal, vectorCount: n, usesFloatComponents: true,
            componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0, dataStride: MemoryLayout<SCNVector3>.stride)
        let cData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
        let cSource = SCNGeometrySource(
            data: cData, semantic: .color, vectorCount: n, usesFloatComponents: true,
            componentsPerVector: 3, bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0, dataStride: MemoryLayout<SCNVector3>.stride)

        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geo = SCNGeometry(sources: [vSource, nSource, cSource], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .blinn          // shaded surfaces (this is the RViz mesh)
        mat.isDoubleSided = true            // nvblox faces aren't consistently wound
        mat.diffuse.contents = UIColor.white   // modulated by the per-vertex colour
        geo.materials = [mat]
        return SCNNode(geometry: geo)
    }
}

/// Tiny deterministic RNG so each room cloud is stable across launches.
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed &* 0x9E3779B97F4A7C15 &+ 1 }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
}
