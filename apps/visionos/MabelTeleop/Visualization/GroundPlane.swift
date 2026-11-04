//
//  GroundPlane.swift
//  MabelTeleop
//
//  THE FLOOR. A ground the operator can actually read, instead of the eight faint
//  hairlines that used to stand in for one.
//
//  Shared verbatim with the iPhone app (`MABEL Mobile App/Visualization/
//  GroundPlane.swift`). visionOS note: this is the WINDOWED SceneKit viewer's
//  floor; the immersive RealityKit stage has its own
//  (`Features/Robot/SensorLayers.buildFloor`), which sits over passthrough where a
//  light lattice competes with the operator's real room, so it grounds with a DARK
//  wash instead. Same chip, same intent, deliberately different value. Both viewers are the same SceneKit scene graph, and the
//  floor is the one element the operator sees in EVERY mode — 3rd person, physics
//  view, an empty map with no SLAM data at all — so a divergence here reads as two
//  different products. Change one, change the other.
//
//  WHAT WAS WRONG WITH THE OLD ONE
//
//  `gridNode(half: 8, step: 1)`: a 16 × 16 m square of 1 m lines at 6 % white,
//  pinned to the odom origin, and nothing else. Three faults, each a rule here:
//
//  · A GRID IS NOT A SURFACE. One cadence of hairlines reads as a wireframe in a
//    void; there is no moment where the eye says "that is the ground". The floor
//    needs a FINE near-field texture you can read the surface off, which is what
//    `surface()` draws, plus the contact shadow below.
//  · A HARD EDGE IS A PLATE, NOT A FLOOR. The old square stopped dead at ±8 m,
//    announcing its own size. Everything here fades radially to nothing.
//  · IT DID NOT FOLLOW THE ROBOT. Drive 10 m and MABEL walked off the edge of the
//    world. `stage` now follows, snapped to whole cells (see `snap`) so the lines
//    still land on the same world positions and the robot's motion stays legible.
//    Following continuously would glue the grid to the robot and it would look
//    motionless.
//
//  WHY EVERYTHING IS LINES AND NOT A LIT PLANE
//
//  The obvious design — a soft radial wash on a big quad — was built and measured
//  three times (iPhone 17 Pro, live robot, 2026-07-29) and failed three times:
//
//  · 8-bit texture, no dither → concentric contour RINGS, and rainbow ones,
//    because premultiplied RGB crosses its quantisation thresholds at different
//    radii than alpha does.
//  · 8-bit texture, dithered → no rings, but a 1-LSB dither is ~10 % of a signal
//    this faint, so the floor looked like film grain.
//  · 16-bit texture → THE SAME RINGS. That is the diagnostic result. The banding
//    was never in the texture: blending alpha 0.075 over a near-black background
//    (0x0A0908) puts the entire ramp inside ~13 FRAMEBUFFER levels, and each
//    channel steps at a slightly different radius. No source precision can fix a
//    destination that coarse — only a shader-side dither, which is a great deal
//    of machinery for a floor.
//
//  Lines cannot band. Each one carries a single opacity, and the radial fade is
//  carried by having DIFFERENT lines at different opacities rather than by a
//  gradient across one surface. It is also the more Apple answer: a fine lattice
//  plus a contact shadow is how the system's own 3-D tools ground an object.
//
//  Everything is `.constant`-shaded and depth-write-free: the floor must never
//  z-fight the occupancy plane (+1 cm), the path polyline (+5 cm) or the clouds,
//  and it must never be lit — a directional key sweeping a 30 m plane is a
//  headlight, not a room.
//

import SceneKit
import UIKit
import simd

enum GroundPlane {

    /// Radius, in metres, at which the reference GRID has faded to nothing. 15 m is
    /// about four rooms across — big enough that the operator never sees the ground
    /// end, small enough that the whole lattice stays a few thousand vertices.
    static let radius: Float = 15

    /// How far the fine near-field texture extends. Deliberately small: its job is
    /// "there is ground right HERE, under the robot". Past this the metric lattice
    /// takes over, and does the far field better than any wash could.
    static let surfaceRadius: Float = 4

    /// Reference cadence. 1 m minor, every 5th line major — the same cadence as the
    /// occupancy grid's metric ticks, so the two layers agree when both are on.
    static let step: Float = 1
    static let majorEvery = 5

    /// The fine texture's cadence, and the reason `skip` exists below: every 4th
    /// fine line coincides with a 1 m reference line, which `grid()` already draws
    /// brighter. Drawing both would double the opacity on exactly those lines and
    /// pick out a 1 m plaid.
    static let fineStep: Float = 0.25

    // MARK: - The two layers

    /// FLOOR — the fine near-field lattice. Chip "Floor", together with the
    /// contact shadow.
    static func surface() -> SCNNode {
        let node = lattice(step: fineStep, radius: surfaceRadius, alpha: 0.055,
                           skip: Int((step / fineStep).rounded()), plateau: 0.1)
        node.name = "groundSurface"
        node.renderingOrder = -30                  // first, under everything
        return node
    }

    /// GRID — the metric reference lattice: 1 m minor, 5 m major, brighter axes.
    /// Chip "Grid".
    static func grid() -> SCNNode {
        let node = lattice(step: step, radius: radius, alpha: 0.075,
                           skip: nil, plateau: 0.35,
                           majorEvery: majorEvery, majorAlpha: 0.16, axisAlpha: 0.30)
        node.name = "groundGrid"
        node.renderingOrder = -20                  // above the fine texture
        return node
    }

    // MARK: - Lattice builder

    /// A disc of XZ lines, faded radially, at up to three opacities.
    ///
    /// Segments are bucketed BY OPACITY into one node each rather than carrying
    /// per-vertex alpha: SceneKit's vertex-colour path is a 3-component `.color`
    /// semantic, so an alpha ramp there is not portable. Bucketing costs a handful
    /// of extra draw calls (all trivial line geometry) and is exact.
    ///
    /// - Parameters:
    ///   - skip: omit lines whose index is a multiple of this — used by the fine
    ///     lattice so it does not double up on the reference lines.
    ///   - plateau: fraction of `radius` held at full opacity before the fade
    ///     starts. The working area around the robot keeps its full cadence.
    private static func lattice(step: Float, radius R: Float, alpha minorA: Float,
                                skip: Int?, plateau: Float,
                                majorEvery: Int = 0, majorAlpha: Float = 0,
                                axisAlpha: Float = 0) -> SCNNode {
        let container = SCNNode()

        // 0.5 m segments give a fade that reads as continuous; 6 buckets is the
        // point past which the steps are below 8-bit alpha resolution anyway.
        let seg: Float = 0.5
        let buckets: Float = 6
        var byAlpha: [Int: [SCNVector3]] = [:]

        func fade(_ r: Float) -> Float {
            let t = (r / R - plateau) / (1 - plateau)
            if t <= 0 { return 1 }
            if t >= 1 { return 0 }
            return 1 - t * t * (3 - 2 * t)                  // smoothstep
        }

        let n = Int((R / step).rounded())
        for i in -n...n {
            let p = Float(i) * step
            guard abs(p) <= R else { continue }
            if let skip, i % skip == 0 { continue }         // the coarser tier owns it

            let base: Float
            if i == 0, axisAlpha > 0 { base = axisAlpha }
            else if majorEvery > 0, i % majorEvery == 0, majorAlpha > 0 { base = majorAlpha }
            else { base = minorA }

            // Half-chord of the disc at this offset — what clips the square lattice
            // into a circle, so there are no corners to announce the edge.
            let halfChord = (R * R - p * p).squareRoot()
            guard halfChord > seg else { continue }

            var t = -halfChord
            while t < halfChord {
                let t2 = min(t + seg, halfChord)
                let mid = (t + t2) / 2
                let a = base * ((fade((p * p + mid * mid).squareRoot()) * buckets).rounded() / buckets)
                let key = Int((a * 1000).rounded())
                if key > 0 {
                    byAlpha[key, default: []].append(contentsOf: [
                        SCNVector3(p, 0, t), SCNVector3(p, 0, t2),   // ∥ Z
                        SCNVector3(t, 0, p), SCNVector3(t2, 0, p),   // ∥ X
                    ])
                }
                t = t2
            }
        }

        for (key, verts) in byAlpha where !verts.isEmpty {
            let src = SCNGeometrySource(vertices: verts)
            let elem = SCNGeometryElement(indices: [Int32](0..<Int32(verts.count)),
                                          primitiveType: .line)
            let geo = SCNGeometry(sources: [src], elements: [elem])
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = UIColor(white: 1, alpha: CGFloat(key) / 1000)
            m.writesToDepthBuffer = false
            m.readsFromDepthBuffer = true
            m.blendMode = .alpha
            geo.materials = [m]
            let node = SCNNode(geometry: geo)
            node.castsShadow = false
            container.addChildNode(node)
        }
        return container
    }

    // MARK: - Contact shadow

    /// The dark pool under the robot — the cue that turns "a model near a plane"
    /// into "a robot standing on the floor". The single strongest grounding signal
    /// in the scene, and the reason the fine lattice can stay as quiet as it does.
    ///
    /// This one IS a gradient, and that is fine: its peak alpha (0.46) spans ~117
    /// framebuffer levels, so unlike the abandoned wash it has ample room to be
    /// smooth.
    ///
    /// Parented to the ROBOT (not the stage) so it tracks continuously; the stage's
    /// 1 m snapping would make a shadow visibly lag its own robot.
    ///
    /// `y` is in the robot's local frame, where +Y is up (the rig's base_link keeps
    /// only the Z-up→Y-up rotation) and the wheels sit ~8 cm below base_link.
    static func contactShadow(radius r: Float = 0.9, y: Float = -0.074) -> SCNNode {
        let plane = SCNPlane(width: CGFloat(r * 2), height: CGFloat(r * 2))
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = shadowTexture()
        m.isDoubleSided = true
        m.writesToDepthBuffer = false
        m.readsFromDepthBuffer = true
        m.blendMode = .alpha
        plane.materials = [m]
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)   // XY plane → XZ ground
        node.simdPosition = simd_float3(0, y, 0)
        node.renderingOrder = -10                  // over the lattices, under data
        node.name = "contactShadow"
        node.castsShadow = false
        return node
    }

    /// Dense core, long soft tail — an ambient-occlusion pool, not a hard cast
    /// shadow. There is no single light in this scene to cast one from, and a crisp
    /// ellipse would claim there is.
    ///
    /// Written pixel by pixel, NOT with `CGGradient`: Core Graphics dithers
    /// gradients, which is invisible in a UI element and, on a texture stretched
    /// over metres and viewed at a grazing angle, magnifies into soft blobs.
    private static func shadowTexture() -> UIImage? {
        let S = 256
        let peak: Float = 0.46
        var px = [UInt8](repeating: 0, count: S * S * 4)   // RGBA, premultiplied
        let half = Float(S) / 2
        for y in 0..<S {
            let dy = (Float(y) + 0.5 - half) / half
            for x in 0..<S {
                let dx = (Float(x) + 0.5 - half) / half
                let t = (dx * dx + dy * dy).squareRoot()
                guard t < 1 else { continue }              // outside the disc: clear
                // Zero slope at the rim, so the plane's edge can never show.
                let a = peak * pow(1 - t, 1.9)
                // Black, premultiplied ⇒ RGB stays 0; only alpha carries the pool.
                px[(y * S + x) * 4 + 3] = UInt8(max(0, min(255, (a * 255).rounded())))
            }
        }
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
                 | CGBitmapInfo.byteOrder32Big.rawValue
        return px.withUnsafeMutableBytes { raw -> UIImage? in
            guard let ctx = CGContext(data: raw.baseAddress, width: S, height: S,
                                      bitsPerComponent: 8, bytesPerRow: S * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: info),
                  let cg = ctx.makeImage() else { return nil }
            return UIImage(cgImage: cg)
        }
    }

    // MARK: - Follow

    /// Where the stage should sit so the ground stays under the robot: the robot's
    /// position snapped to whole grid cells, with Y pinned to the floor.
    ///
    /// The snap is the whole point — see the file header.
    static func snap(to p: simd_float3, y: Float = 0) -> simd_float3 {
        simd_float3((p.x / step).rounded() * step, y, (p.z / step).rounded() * step)
    }
}
