//
//  NavMapState.swift
//  MABEL Mobile App
//
//  The data model behind the floating top-down 2D map — the app's "mobile RViz"
//  mini-view. It mirrors, in a lightweight 2D form, the SAME data ROS streams the
//  app: the 2D occupancy map (nav_msgs/OccupancyGrid → GRID 0x04), the robot's
//  odom pose (ODOM 0x02), the Nav2 route (PATH 0x03), and — when no occupancy grid
//  is published — a 2D floorplan EXTRACTED from the live SLAM mesh (MESH 0x01),
//  i.e. the same "flatten the 3D reconstruction" RViz does. Offline (no robot) it
//  falls back to the on-device A* planner's footprint + route.
//
//  It is fed by the SceneKit coordinators (LiveMapView for the live stream,
//  PointCloudView for the procedural demo) and read only by `MiniMap2D`. MapsView
//  holds it as plain @State (NOT @StateObject), so the 20–50 Hz odom updates
//  invalidate ONLY the small minimap canvas — never the whole screen / SceneKit host.
//

import SwiftUI
import Combine
import SceneKit
import CoreGraphics

/// The 2D-map model. Geometry is in the GROUND plane (SceneKit XZ, metres). For the
/// live stream that is the ROS `odom` frame rebased to SceneKit (x, z); for the
/// procedural demo it is the room frame — either way a top-down XZ projection.
final class NavMapState: ObservableObject {
    struct Pose: Equatable { var x = 0.0; var z = 0.0; var yaw = 0.0 }
    enum Source { case none, procedural, live }

    @Published var source: Source = .none

    // ── Procedural (on-device A*) ───────────────────────────────────────────
    @Published var bounds: CGSize = .zero        // room XZ extent (m)
    @Published var obstacles: [CGRect] = []      // furniture footprints (m)

    // ── Live (ROS) 2D map ───────────────────────────────────────────────────
    /// Rasterised top-down occupancy: from the real OccupancyGrid when present,
    /// else extracted from the SLAM mesh. Pixel (col,row) → world (minX+col·res,
    /// minZ+row·res); row 0 is minZ (top of the canvas).
    @Published var liveRaster: CGImage? = nil
    /// World rect the raster covers, SceneKit XZ: (x:minX, y:minZ, w, h).
    @Published var liveExtent: CGRect = .zero
    /// Grid pose yaw (SceneKit +Y, radians) — the raster is drawn rotated by
    /// this about its origin corner so it aligns with the 3D map.
    @Published var liveYaw: CGFloat = 0
    /// True once a *real* OccupancyGrid arrived (so a later mesh frame won't
    /// overwrite the authoritative 2D map with the cruder mesh projection).
    private var haveRealGrid = false

    // ── Common ──────────────────────────────────────────────────────────────
    @Published var path: [CGPoint] = []          // ground waypoints (x, z)
    @Published var goal: CGPoint? = nil          // active goal (x, z)
    @Published var robot = Pose()
    @Published var following = false

    var hasMap: Bool { source != .none }

    // MARK: - Procedural feed (PointCloudView)

    func setProceduralStatic(bounds: CGSize, obstacles: [CGRect]) {
        if source != .procedural { resetLive() }
        source = .procedural
        if self.bounds != bounds { self.bounds = bounds }
        self.obstacles = obstacles
    }

    // MARK: - Live feed (LiveMapView)

    /// Switch into live mode (Live SLAM / streamed map). Clears procedural state.
    func beginLive() {
        if source != .live { bounds = .zero; obstacles = []; path = []; goal = nil }
        source = .live
    }

    /// The real ROS OccupancyGrid → the authoritative 2D map. Wins over the mesh.
    func ingest(grid: MapGrid) {
        guard grid.width > 0, grid.height > 0, grid.cells.count >= grid.width * grid.height else { return }
        source = .live
        let res = CGFloat(grid.resolution > 0 ? grid.resolution : 0.05)
        // Assume an axis-aligned grid (yaw ≈ 0, the common case): origin is the
        // SceneKit pos of cell (0,0); cells advance +x (col) and +z… but the ROS
        // grid's +y (row) maps to SceneKit −y of odom, already folded into origin.
        let ox = CGFloat(grid.origin.x), oz = CGFloat(grid.origin.z)
        let w = CGFloat(grid.width) * res, h = CGFloat(grid.height) * res
        // Rows advance toward −z from the origin corner (same convention as the
        // 3D occupancy quad) — the extent spans [oz − h, oz], NOT [oz, oz + h].
        // The old +z extent mirrored the map to the wrong side of the robot.
        liveExtent = CGRect(x: ox, y: oz - h, width: w, height: h)
        liveYaw = CGFloat(grid.yaw)
        liveRaster = Self.rasterizeGrid(grid)
        haveRealGrid = true
    }

    /// No occupancy grid published → extract a 2D floorplan from the SLAM mesh by
    /// flattening everything between floor and ceiling onto the ground plane.
    func ingest(mesh: MapMesh) {
        guard !haveRealGrid, mesh.points.count > 2 else { return }
        source = .live
        guard let (img, ext) = Self.rasterizeMesh(mesh) else { return }
        liveRaster = img
        liveExtent = ext
    }

    func setLivePath(_ pts: [CGPoint]) { source = .live; path = pts }
    func setRobot(x: Double, z: Double, yaw: Double) { robot = Pose(x: x, z: z, yaw: yaw) }
    func setGoal(_ g: CGPoint?) { goal = g }

    private func resetLive() {
        liveRaster = nil; liveExtent = .zero; haveRealGrid = false
    }

    // MARK: - Procedural mutations (shared)

    func setPlan(path: [CGPoint], goal: CGPoint?) { self.path = path; self.goal = goal }
    func clearPlan() { path = []; goal = nil }

    // MARK: - Rasterisers (build a small CGImage once per map update)

    /// Walls/objects = light; free = faint; unknown = clear. Drawn on the dark card,
    /// so occupied cells read as bright floorplan lines.
    private static func rasterizeGrid(_ g: MapGrid) -> CGImage? {
        let w = g.width, h = g.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        for r in 0..<h {
            for c in 0..<w {
                let v = g.cells[r * w + c]
                // Row 0 sits at the +z edge of the extent (rows advance −z), and
                // the canvas draws image row 0 at the rect's −z edge → flip rows.
                let i = ((h - 1 - r) * w + c) * 4
                if v == 255 { continue }                 // unknown → clear
                if v >= 65 { px[i] = 222; px[i+1] = 224; px[i+2] = 230; px[i+3] = 235 }  // occupied
                else        { px[i] = 96;  px[i+1] = 98;  px[i+2] = 104; px[i+3] = 34  }  // free
            }
        }
        return image(from: &px, w: w, h: h)
    }

    /// Flatten the mesh: any vertex between floor and ceiling becomes an obstacle
    /// cell — the 2D footprint of the reconstructed environment.
    private static func rasterizeMesh(_ mesh: MapMesh) -> (CGImage, CGRect)? {
        var minX = Float.greatestFiniteMagnitude, minZ = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude, maxZ = -Float.greatestFiniteMagnitude
        for p in mesh.points {
            let y = Float(p.y)
            guard y > 0.12, y < 2.4 else { continue }    // skip floor + ceiling
            minX = min(minX, Float(p.x)); maxX = max(maxX, Float(p.x))
            minZ = min(minZ, Float(p.z)); maxZ = max(maxZ, Float(p.z))
        }
        guard minX <= maxX, minZ <= maxZ else { return nil }
        let res: Float = 0.08
        let cols = min(360, max(2, Int((maxX - minX) / res) + 1))
        let rows = min(360, max(2, Int((maxZ - minZ) / res) + 1))
        let sx = Float(cols - 1) / max(1e-3, maxX - minX)
        let sz = Float(rows - 1) / max(1e-3, maxZ - minZ)
        var hits = [UInt16](repeating: 0, count: cols * rows)
        for p in mesh.points {
            let y = Float(p.y)
            guard y > 0.12, y < 2.4 else { continue }
            let c = Int((Float(p.x) - minX) * sx), r = Int((Float(p.z) - minZ) * sz)
            if c >= 0, c < cols, r >= 0, r < rows { hits[r * cols + c] &+= 1 }
        }
        var px = [UInt8](repeating: 0, count: cols * rows * 4)
        for idx in 0..<(cols * rows) where hits[idx] > 0 {
            let i = idx * 4
            px[i] = 222; px[i+1] = 224; px[i+2] = 230
            px[i+3] = hits[idx] >= 2 ? 235 : 150         // denser → more opaque
        }
        guard let img = image(from: &px, w: cols, h: rows) else { return nil }
        return (img, CGRect(x: CGFloat(minX), y: CGFloat(minZ),
                            width: CGFloat(maxX - minX), height: CGFloat(maxZ - minZ)))
    }

    private static func image(from px: inout [UInt8], w: Int, h: Int) -> CGImage? {
        guard w > 0, h > 0 else { return nil }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let bmp = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        return px.withUnsafeMutableBytes { raw -> CGImage? in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: cs, bitmapInfo: bmp.rawValue) else { return nil }
            return ctx.makeImage()
        }
    }
}

/// A floating top-down 2D map — the schematic counterpart to the 3D scene, like a
/// mobile RViz map panel. Live: the real ROS occupancy grid (or a mesh-extracted
/// floorplan) with the odom robot + Nav2 route. Offline: the A* planner's footprint.
///
/// It has NO hard card / border — a soft radial backing and a feathered edge mask
/// make it fade ("disappear") into the scene, so it never reads as a panel that
/// obstructs the controls. It also ignores hit-testing, so it can never block a tap.
struct MiniMap2D: View {
    @ObservedObject var state: NavMapState
    var size: CGFloat = 138

    private let accent = Theme.Palette.accent
    private let live   = Theme.Palette.live
    private let clay   = Theme.Palette.clay

    /// Edge feather — opaque center, fully transparent rim, so the map dissolves
    /// into the surrounding scene instead of ending at a hard border.
    private var fadeMask: RadialGradient {
        RadialGradient(gradient: Gradient(stops: [
            .init(color: .black, location: 0.0),
            .init(color: .black, location: 0.66),
            .init(color: .black.opacity(0.25), location: 0.9),
            .init(color: .clear, location: 1.0),
        ]), center: .center, startRadius: 0, endRadius: size * 0.62)
    }

    var body: some View {
        Canvas { ctx, sz in draw(ctx, size: sz) }
            .frame(width: size, height: size)
            // Soft dark vignette that itself fades to nothing at the rim.
            .background(
                RadialGradient(gradient: Gradient(stops: [
                    .init(color: Color(hex: 0x0A0908).opacity(0.5), location: 0.0),
                    .init(color: Color(hex: 0x0A0908).opacity(0.32), location: 0.6),
                    .init(color: .clear, location: 1.0),
                ]), center: .center, startRadius: 0, endRadius: size * 0.6)
            )
            .mask(fadeMask)                 // feathered, disappearing edges
            .allowsHitTesting(false)        // purely a viz — never blocks a map tap
    }

    // MARK: - Drawing

    private func draw(_ ctx: GraphicsContext, size sz: CGSize) {
        switch state.source {
        case .live:       drawLive(ctx, size: sz)
        case .procedural: drawProcedural(ctx, size: sz)
        case .none:       break
        }
    }

    /// Live: ROS occupancy grid / mesh floorplan + odom robot + Nav2 route.
    private func drawLive(_ ctx: GraphicsContext, size sz: CGSize) {
        // World extent: the raster's, expanded to include the robot + path so they
        // never clip off-card while the map is still small.
        var minX = state.liveExtent.minX, maxX = state.liveExtent.maxX
        var minZ = state.liveExtent.minY, maxZ = state.liveExtent.maxY
        func grow(_ x: CGFloat, _ z: CGFloat) {
            minX = min(minX, x); maxX = max(maxX, x); minZ = min(minZ, z); maxZ = max(maxZ, z)
        }
        if state.liveExtent.isEmpty { minX = .greatestFiniteMagnitude; maxX = -.greatestFiniteMagnitude
                                      minZ = .greatestFiniteMagnitude; maxZ = -.greatestFiniteMagnitude }
        grow(CGFloat(state.robot.x), CGFloat(state.robot.z))
        for p in state.path { grow(p.x, p.y) }
        if let g = state.goal { grow(g.x, g.y) }
        guard minX <= maxX, minZ <= maxZ else {
            placeholder(ctx, size: sz); return
        }
        // Pad and keep a little breathing room.
        let padW = max(0.5, (maxX - minX) * 0.06), padZ = max(0.5, (maxZ - minZ) * 0.06)
        minX -= padW; maxX += padW; minZ -= padZ; maxZ += padZ
        let w = max(0.001, maxX - minX), h = max(0.001, maxZ - minZ)
        let pad: CGFloat = 9
        let scale = min((sz.width - pad * 2) / w, (sz.height - pad * 2) / h)
        let mid = CGPoint(x: sz.width / 2, y: sz.height / 2)
        let cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2
        func P(_ x: CGFloat, _ z: CGFloat) -> CGPoint {
            CGPoint(x: mid.x + (x - cx) * scale, y: mid.y + (z - cz) * scale)
        }

        // Occupancy raster — drawn rotated by the grid pose's yaw about its
        // ORIGIN corner (x: minX, z: maxY), matching the 3D occupancy quad.
        if let img = state.liveRaster, !state.liveExtent.isEmpty {
            let e = state.liveExtent
            var inner = ctx
            let o = P(e.minX, e.maxY)
            inner.translateBy(x: o.x, y: o.y)
            inner.rotate(by: Angle(radians: -Double(state.liveYaw)))
            let r = CGRect(x: 0, y: -e.height * scale,
                           width: e.width * scale, height: e.height * scale)
            inner.draw(Image(decorative: img, scale: 1, orientation: .up), in: r)
        }
        drawRoute(ctx, P: P)
        drawRobot(ctx, P: P, scale: scale)
    }

    /// Offline: the A* planner's footprint + obstacles + route.
    private func drawProcedural(_ ctx: GraphicsContext, size sz: CGSize) {
        let w = max(0.001, state.bounds.width), h = max(0.001, state.bounds.height)
        let pad: CGFloat = 9
        let scale = min((sz.width - pad * 2) / w, (sz.height - pad * 2) / h)
        let mid = CGPoint(x: sz.width / 2, y: sz.height / 2)
        func P(_ x: CGFloat, _ z: CGFloat) -> CGPoint { CGPoint(x: mid.x + x * scale, y: mid.y + z * scale) }

        let floor = CGRect(x: P(-w/2, -h/2).x, y: P(-w/2, -h/2).y, width: w * scale, height: h * scale)
        ctx.fill(Path(roundedRect: floor, cornerRadius: 4), with: .color(.white.opacity(0.05)))
        ctx.stroke(Path(roundedRect: floor, cornerRadius: 4), with: .color(.white.opacity(0.18)), lineWidth: 1)
        for box in state.obstacles {
            let r = CGRect(x: P(box.minX, box.minY).x, y: P(box.minX, box.minY).y,
                           width: box.width * scale, height: box.height * scale)
            ctx.fill(Path(roundedRect: r, cornerRadius: 2.5), with: .color(clay.opacity(0.55)))
            ctx.stroke(Path(roundedRect: r, cornerRadius: 2.5), with: .color(clay.opacity(0.9)), lineWidth: 0.75)
        }
        drawRoute(ctx, P: P)
        drawRobot(ctx, P: P, scale: scale)
    }

    private func placeholder(_ ctx: GraphicsContext, size sz: CGSize) {
        let r = CGRect(x: 10, y: 10, width: sz.width - 20, height: sz.height - 20)
        ctx.stroke(Path(roundedRect: r, cornerRadius: 6),
                   with: .color(.white.opacity(0.12)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
    }

    private func drawRoute(_ ctx: GraphicsContext, P: (CGFloat, CGFloat) -> CGPoint) {
        guard state.path.count > 1 else {
            if let g = state.goal { goalMark(ctx, at: P(g.x, g.y)) }
            return
        }
        var line = Path()
        line.move(to: P(state.path[0].x, state.path[0].y))
        for p in state.path.dropFirst() { line.addLine(to: P(p.x, p.y)) }
        ctx.stroke(line, with: .color(live.opacity(0.95)),
                   style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        if let g = state.goal { goalMark(ctx, at: P(g.x, g.y)) }
        else if let last = state.path.last { goalMark(ctx, at: P(last.x, last.y)) }
    }

    private func goalMark(_ ctx: GraphicsContext, at c: CGPoint) {
        ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 5, y: c.y - 5, width: 10, height: 10)),
                   with: .color(clay), lineWidth: 1.5)
        ctx.fill(Path(ellipseIn: CGRect(x: c.x - 2, y: c.y - 2, width: 4, height: 4)), with: .color(clay))
    }

    /// Robot heading-triangle. Point along the route when one exists (robust to the
    /// yaw convention); otherwise use the reported yaw.
    private func drawRobot(_ ctx: GraphicsContext, P: (CGFloat, CGFloat) -> CGPoint, scale: CGFloat) {
        let rp = P(CGFloat(state.robot.x), CGFloat(state.robot.z))
        var dir = CGVector(dx: 0, dy: -1)
        if let nxt = nextWaypoint() {
            let np = P(nxt.x, nxt.y)
            let d = hypot(np.x - rp.x, np.y - rp.y)
            if d > 0.5 { dir = CGVector(dx: (np.x - rp.x) / d, dy: (np.y - rp.y) / d) }
        } else {
            let y = CGFloat(state.robot.yaw)
            dir = CGVector(dx: sin(y), dy: cos(y))
        }
        let perp = CGVector(dx: -dir.dy, dy: dir.dx)
        let tip = CGPoint(x: rp.x + dir.dx * 9, y: rp.y + dir.dy * 9)
        let bl  = CGPoint(x: rp.x - dir.dx * 5 + perp.dx * 6, y: rp.y - dir.dy * 5 + perp.dy * 6)
        let br  = CGPoint(x: rp.x - dir.dx * 5 - perp.dx * 6, y: rp.y - dir.dy * 5 - perp.dy * 6)
        var tri = Path(); tri.move(to: tip); tri.addLine(to: bl); tri.addLine(to: br); tri.closeSubpath()
        ctx.fill(Path(ellipseIn: CGRect(x: rp.x - 8, y: rp.y - 8, width: 16, height: 16)),
                 with: .color(accent.opacity(0.18)))
        ctx.fill(tri, with: .color(accent))
        ctx.stroke(tri, with: .color(.white.opacity(0.9)), lineWidth: 1)
    }

    /// The path point a little ahead of the robot, for heading.
    private func nextWaypoint() -> CGPoint? {
        guard state.path.count > 1 else { return nil }
        let r = CGPoint(x: state.robot.x, y: state.robot.z)
        var best: CGPoint? = nil; var bestD = CGFloat.greatestFiniteMagnitude
        for p in state.path {
            let d = hypot(p.x - r.x, p.y - r.y)
            if d > 0.2, d < bestD { bestD = d; best = p }
        }
        return best ?? state.path.last
    }
}
