//
//  PointCloudEntity.swift
//  MabelTeleop
//
//  Builds a SLAM room's dense point cloud as RealityKit geometry, plus the
//  furniture footprints the A* planner routes around — the visionOS
//  counterpart of the iOS app's PointCloudScene. Clouds are procedurally
//  synthesized stand-ins for real RTAB-Map / ORB-SLAM exports (same as iOS).
//
//  RealityKit has no point primitive, so points are batched into a handful
//  of colour-bucketed meshes of tiny camera-agnostic triangles — one draw
//  call per bucket, tens of thousands of points at 90 fps.
//

import Foundation
import RealityKit
import UIKit
import simd

/// Everything the Navigate view needs to plan and drive in a room.
@MainActor
struct RoomNav {
    let cloud: Entity                 // the point cloud (+floor) container
    let bounds: CGSize                // room footprint (metres)
    let obstacles: [CGRect]           // furniture footprints on the XZ plane
    let pathContainer: Entity         // route segments + goal marker live here
}

enum PointCloudEntity {

    // Deterministic per-room randomness.
    private struct SplitMix: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    /// Build the room: point cloud + invisible floor collider for pinch-to-goal.
    @MainActor
    static func make(for room: SLAMRoom) -> RoomNav {
        let container = Entity()
        var rng = SplitMix(state: room.seed)

        let w = CGFloat(room.size.x), h = CGFloat(room.size.y), d = CGFloat(room.size.z)
        let obstacles = furniture(for: room, rng: &rng)

        // ── Sample points: floor, walls, furniture volumes ───────────────────
        var pts: [SIMD3<Float>] = []
        let budget = 24_000

        let floorN = budget * 38 / 100
        for _ in 0..<floorN {
            let x = Float.random(in: Float(-w/2)...Float(w/2), using: &rng)
            let z = Float.random(in: Float(-d/2)...Float(d/2), using: &rng)
            pts.append(SIMD3(x, Float.random(in: 0...0.015, using: &rng), z))
        }
        let wallN = budget * 22 / 100
        for _ in 0..<wallN {
            let y = Float.random(in: 0...Float(h), using: &rng)
            let t = Float.random(in: 0...1, using: &rng)
            switch Int.random(in: 0..<4, using: &rng) {
            case 0: pts.append(SIMD3(Float(-w/2), y, Float(d) * (t - 0.5)))
            case 1: pts.append(SIMD3(Float(w/2),  y, Float(d) * (t - 0.5)))
            case 2: pts.append(SIMD3(Float(w) * (t - 0.5), y, Float(-d/2)))
            default: pts.append(SIMD3(Float(w) * (t - 0.5), y, Float(d/2)))
            }
        }
        let furnN = budget - floorN - wallN
        if !obstacles.isEmpty {
            for _ in 0..<furnN {
                let box = obstacles[Int.random(in: 0..<obstacles.count, using: &rng)]
                let bh = Float.random(in: 0.4...1.4, using: &rng)
                // Sample the box's surface shell so furniture reads as solid.
                let x = Float.random(in: Float(box.minX)...Float(box.maxX), using: &rng)
                let z = Float.random(in: Float(box.minY)...Float(box.maxY), using: &rng)
                pts.append(SIMD3(x, Float.random(in: 0...bh, using: &rng), z))
            }
        }

        // ── Colour-bucket by height (warm SLAM-viz ramp), one mesh per bucket ─
        let ramp: [UIColor] = [
            UIColor(red: 0.42, green: 0.36, blue: 0.30, alpha: 1),   // floor
            UIColor(red: 0.76, green: 0.45, blue: 0.24, alpha: 1),   // low
            UIColor(red: 0.85, green: 0.62, blue: 0.34, alpha: 1),   // mid
            UIColor(red: 0.62, green: 0.70, blue: 0.78, alpha: 1),   // high
            UIColor(red: 0.80, green: 0.82, blue: 0.85, alpha: 1),   // ceiling-ish
        ]
        var buckets: [[SIMD3<Float>]] = Array(repeating: [], count: ramp.count)
        for p in pts {
            let f = max(0, min(0.999, p.y / Float(h)))
            let idx = p.y < 0.03 ? 0 : 1 + Int(f * Float(ramp.count - 1 - 1))
            buckets[min(idx, ramp.count - 1)].append(p)
        }
        for (i, bucket) in buckets.enumerated() where !bucket.isEmpty {
            if let mesh = pointMesh(bucket, size: 0.014, rng: &rng) {
                var mat = UnlitMaterial(color: ramp[i])
                mat.faceCulling = .none
                container.addChild(ModelEntity(mesh: mesh, materials: [mat]))
            }
        }

        // ── Invisible floor plane: the pinch target for goal setting ─────────
        let floor = ModelEntity(mesh: .generatePlane(width: Float(w), depth: Float(d)),
                                materials: [UnlitMaterial(color: .clear)])
        floor.name = "floor"
        floor.components.set(InputTargetComponent())
        floor.components.set(CollisionComponent(shapes: [
            .generateBox(width: Float(w), height: 0.02, depth: Float(d))
        ]))
        container.addChild(floor)

        let pathContainer = Entity()
        container.addChild(pathContainer)

        return RoomNav(cloud: container, bounds: CGSize(width: w, height: d),
                       obstacles: obstacles, pathContainer: pathContainer)
    }

    /// One tiny triangle per point, batched into a single mesh.
    private static func pointMesh(_ points: [SIMD3<Float>], size: Float,
                                  rng: inout SplitMix) -> MeshResource? {
        var positions = [SIMD3<Float>]()
        positions.reserveCapacity(points.count * 6)
        var indices = [UInt32]()
        indices.reserveCapacity(points.count * 24)
        let r = size / 2
        // Tiny OCTAHEDRON per point (6 verts / 8 faces): reads as a round 3D
        // ball from every angle — a flat splat reads as a triangle up close.
        let oct: [SIMD3<Float>] = [SIMD3(r, 0, 0), SIMD3(-r, 0, 0),
                                   SIMD3(0, r, 0), SIMD3(0, -r, 0),
                                   SIMD3(0, 0, r), SIMD3(0, 0, -r)]
        let faces: [UInt32] = [0, 2, 4,  2, 1, 4,  1, 3, 4,  3, 0, 4,
                               2, 0, 5,  1, 2, 5,  3, 1, 5,  0, 3, 5]
        for p in points {
            let base = UInt32(positions.count)
            for v in oct { positions.append(p + v) }
            for i in faces { indices.append(base + i) }
        }
        _ = rng
        var desc = MeshDescriptor()
        desc.positions = MeshBuffer(positions)
        desc.primitives = .triangles(indices)
        return try? MeshResource.generate(from: [desc])
    }

    /// Furniture footprints per room kind — these become A* obstacles.
    private static func furniture(for room: SLAMRoom, rng: inout SplitMix) -> [CGRect] {
        let w = CGFloat(room.size.x), d = CGFloat(room.size.z)
        var boxes: [CGRect] = []
        let count: Int
        switch room.kind {
        case .studio:     count = 5
        case .livingRoom: count = 4
        case .workshop:   count = 6
        case .corridor:   count = 3
        }
        for _ in 0..<count {
            let bw = CGFloat(Float.random(in: 0.5...1.6, using: &rng))
            let bd = CGFloat(Float.random(in: 0.5...1.4, using: &rng))
            let x = CGFloat(Float.random(in: Float(-w/2 + 0.6)...Float(w/2 - 0.6), using: &rng))
            let z = CGFloat(Float.random(in: Float(-d/2 + 0.6)...Float(d/2 - 0.6), using: &rng))
            let box = CGRect(x: x - bw/2, y: z - bd/2, width: bw, height: bd)
            // Keep the spawn point (origin) clear so the robot never starts inside.
            if box.insetBy(dx: -0.4, dy: -0.4).contains(CGPoint.zero) { continue }
            boxes.append(box)
        }
        return boxes
    }

    // MARK: - Route rendering

    /// Draw the planned route + goal marker into `container`.
    @MainActor
    static func renderPath(_ path: [CGPoint], goal: CGPoint, into container: Entity) {
        container.children.removeAll()
        guard path.count > 1 else { return }

        let mat = UnlitMaterial(color: UIColor(red: 0.25, green: 0.78, blue: 0.40, alpha: 0.95))
        for i in 1..<path.count {
            let a = SIMD3<Float>(Float(path[i-1].x), 0.025, Float(path[i-1].y))
            let b = SIMD3<Float>(Float(path[i].x), 0.025, Float(path[i].y))
            let mid = (a + b) / 2
            let len = simd_length(b - a)
            guard len > 1e-4 else { continue }
            let seg = ModelEntity(mesh: .generateBox(width: 0.03, height: 0.006, depth: len),
                                  materials: [mat])
            seg.position = mid
            seg.orientation = simd_quatf(from: SIMD3(0, 0, 1), to: (b - a) / len)
            container.addChild(seg)
        }

        let ring = ModelEntity(mesh: .generateCylinder(height: 0.01, radius: 0.16),
                               materials: [UnlitMaterial(color: UIColor(red: 0.25, green: 0.78, blue: 0.40, alpha: 0.5))])
        ring.position = SIMD3(Float(goal.x), 0.02, Float(goal.y))
        container.addChild(ring)
        let post = ModelEntity(mesh: .generateCylinder(height: 0.5, radius: 0.012),
                               materials: [mat])
        post.position = SIMD3(Float(goal.x), 0.27, Float(goal.y))
        container.addChild(post)
    }

    // MARK: - Live SLAM stream geometry (map_stream /map: mesh · cloud · grid · path)

    /// A colour-bucketed point cloud from arbitrary per-point RGB (the live nvblox
    /// mesh — rendered as points like the web client — or the live sensor cloud).
    /// RealityKit has no per-vertex colour, so quantise to a small palette → one
    /// solid-colour splat mesh per bucket (a handful of draw calls).
    @MainActor
    static func coloredCloud(points: [SIMD3<Float>], colors: [SIMD3<Float>],
                             size: Float = 0.02) -> Entity {
        let container = Entity()
        guard !points.isEmpty, points.count == colors.count else { return container }
        @inline(__always) func q(_ v: Float) -> Int { min(4, max(0, Int(v * 5))) }
        var buckets: [Int: [SIMD3<Float>]] = [:]
        for i in 0..<points.count {
            let c = colors[i]
            buckets[(q(c.x) << 8) | (q(c.y) << 4) | q(c.z), default: []].append(points[i])
        }
        var rng = SplitMix(state: 0xC1011D)
        for (key, pts) in buckets {
            guard let mesh = pointMesh(pts, size: size, rng: &rng) else { continue }
            var mat = UnlitMaterial(color: UIColor(red: CGFloat((key >> 8) & 0xF) / 4.0,
                                                   green: CGFloat((key >> 4) & 0xF) / 4.0,
                                                   blue: CGFloat(key & 0xF) / 4.0, alpha: 1))
            mat.faceCulling = .none
            container.addChild(ModelEntity(mesh: mesh, materials: [mat]))
        }
        return container
    }

    /// The 2D occupancy map as a flat textured quad anchored at the grid origin,
    /// spanning odom +x (RealityKit +x) and odom +y (RealityKit −z).
    @MainActor
    static func occupancyPlane(width: Int, height: Int, resolution: Float,
                               origin: SIMD3<Float>, yaw: Float, cells: [UInt8]) -> Entity? {
        guard width > 0, height > 0, cells.count >= width * height,
              let tex = occupancyTexture(width: width, height: height, cells: cells)
        else { return nil }
        let ww = Float(width) * resolution, hh = Float(height) * resolution
        var desc = MeshDescriptor()
        desc.positions = MeshBuffer([SIMD3(0, 0, 0), SIMD3(ww, 0, 0),
                                     SIMD3(0, 0, -hh), SIMD3(ww, 0, -hh)])
        desc.textureCoordinates = MeshBuffer([SIMD2(0, 0), SIMD2(1, 0),
                                              SIMD2(0, 1), SIMD2(1, 1)])
        desc.primitives = .triangles([0, 2, 3, 0, 3, 1])
        guard let mesh = try? MeshResource.generate(from: [desc]) else { return nil }
        var mat = UnlitMaterial()
        mat.color = .init(tint: .white.withAlphaComponent(0.78), texture: .init(tex))
        mat.faceCulling = .none
        mat.blending = .transparent(opacity: 1.0)
        let e = ModelEntity(mesh: mesh, materials: [mat])
        // FLOOR-CLAMPED (operator 2026-07-18): the occupancy map is a picture
        // of the FLOOR, but the wire origin carries the map FRAME's height —
        // on maps built from the LiDAR that's the sensor plane (~0.18 m), so
        // the quad hovered and visually collided with the mobile base. Render
        // it just above y = 0 (the odom ground the wheels/goal_floor sit on)
        // regardless of the frame's z; +0.005 clears the virtual floor plane.
        e.position = SIMD3(origin.x, 0.005, origin.z)
        e.orientation = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0))
        return e
    }

    /// Rasterise occupancy cells → a texture: free light, occupied dark, unknown
    /// transparent. NOTE: if the map appears vertically mirrored on device, reverse
    /// the row order here (one-line flip).
    @MainActor
    private static func occupancyTexture(width w: Int, height h: Int,
                                         cells: [UInt8]) -> TextureResource? {
        let a = 200
        var px = [UInt8](repeating: 0, count: w * h * 4)            // RGBA, premultiplied
        for r in 0..<h { for c in 0..<w {
            let val = Int(cells[r * w + c]); let o = (r * w + c) * 4
            if val == 255 { continue }                             // unknown → clear
            let pg = UInt8((255 - min(val, 100) * 255 / 100) * a / 255)
            px[o] = pg; px[o + 1] = pg; px[o + 2] = pg; px[o + 3] = UInt8(a)
        } }
        let info = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info),
              let cg = ctx.makeImage(),
              let tex = try? TextureResource.generate(from: cg, options: .init(semantic: .color))
        else { return nil }
        return tex
    }

    /// Draw the live Nav2 route (RealityKit-space points) + a goal ring.
    @MainActor
    static func renderLivePath(_ pts: [SIMD3<Float>], into container: Entity) {
        container.children.removeAll()
        guard pts.count > 1 else { return }
        let mat = UnlitMaterial(color: UIColor(red: 0.25, green: 0.78, blue: 0.40, alpha: 0.95))
        for i in 1..<pts.count {
            let a = SIMD3<Float>(pts[i - 1].x, 0.03, pts[i - 1].z)
            let b = SIMD3<Float>(pts[i].x, 0.03, pts[i].z)
            let len = simd_length(b - a)
            guard len > 1e-4 else { continue }
            let seg = ModelEntity(mesh: .generateBox(width: 0.03, height: 0.006, depth: len),
                                  materials: [mat])
            seg.position = (a + b) / 2
            seg.orientation = simd_quatf(from: SIMD3(0, 0, 1), to: (b - a) / len)
            container.addChild(seg)
        }
        if let g = pts.last {
            let ring = ModelEntity(mesh: .generateCylinder(height: 0.01, radius: 0.16),
                                   materials: [UnlitMaterial(color: UIColor(red: 0.25, green: 0.78,
                                                                            blue: 0.40, alpha: 0.5))])
            ring.position = SIMD3(g.x, 0.02, g.z)
            container.addChild(ring)
        }
    }
}
