//
//  MPPINavigator.swift
//  MabelTeleop (visionOS)
//
//  Byte-for-byte the iOS twin (MABEL_iOS_v2) — pure simd/CoreGraphics, no platform
//  deps — so both apps navigate the offline demo with IDENTICAL, unit-tested code.
//
//  ON-DEVICE NAVIGATION for demo / offline mode.
//
//  On a real robot the app is RViz, not a planner: a tapped goal goes to ROS
//  (`/goal_pose` → NavFn A* + MPPI) and the route comes back on the /map stream.
//  With no robot there is no ROS, and the demo used to "navigate" by sliding the
//  robot along a STRAIGHT LINE to the goal — straight through walls and furniture
//  ("the demo has no planner", MapStreamLink.setDemoGoal).
//
//  This is the planner. It mirrors the robot's real navigation stack in miniature,
//  so the offline demo behaves like the machine rather than a cartoon:
//
//    GLOBAL   A* over the demo occupancy grid, inflated by the robot's radius
//             (the NavFn layer's job).
//    LOCAL    MPPI — sample K control sequences, roll each one out through the base
//             model, score it, and take the softmax-weighted mean (the controller
//             layer's job). Obstacle cost is a hard penalty, so the rollout that
//             clips a table simply loses.
//
//  The base is holonomic (swerve), so the control is a world-frame velocity
//  (vx, vz); speed is capped by the SAME tip-over motion model the real robot
//  uses (`MotionModel.maxSpeedDir`), so raising the lift slows the demo robot
//  exactly like it slows the machine.
//
//  Deterministic: the sampler is a seeded SplitMix64, so a given (grid, start,
//  goal, seed) always produces the same trajectory — which is what makes the
//  navigation tests in MABEL_Mobile_AppTests meaningful rather than flaky.
//

import CoreGraphics
import Foundation

// MARK: - Occupancy

/// A read-only view of an occupancy grid, in the SAME convention the map stream
/// uses (`MapGrid`): cell (0,0) is the origin corner, +columns → +x, +rows → −z;
/// 0 = free, 100 = occupied, 255 = unknown.
struct NavGrid {
    var cells: [UInt8]
    var width: Int
    var height: Int
    var resolution: Float
    /// World position of the (0,0) cell corner — `MapGrid.origin` (x, _, z).
    var originX: Float
    var originZ: Float

    static let free: UInt8 = 0
    static let occupied: UInt8 = 100
    static let unknown: UInt8 = 255

    func cell(x: Float, z: Float) -> (c: Int, r: Int)? {
        let c = Int((x - originX) / resolution)
        let r = Int((originZ - z) / resolution)
        guard c >= 0, c < width, r >= 0, r < height else { return nil }
        return (c, r)
    }

    func world(c: Int, r: Int) -> CGPoint {
        CGPoint(x: CGFloat(originX + (Float(c) + 0.5) * resolution),
                y: CGFloat(originZ - (Float(r) + 0.5) * resolution))    // .y carries z
    }

    func value(c: Int, r: Int) -> UInt8 {
        guard c >= 0, c < width, r >= 0, r < height else { return Self.unknown }
        return cells[r * width + c]
    }
}

/// Cells the robot may not occupy: anything marked occupied, grown by the robot's
/// radius so the PLAN keeps the whole chassis clear rather than just its centre.
/// Unknown space is treated as FREE — the demo map is revealed as the robot
/// explores, and refusing to enter unknown cells would strand it in the corner it
/// starts in (the real Nav2 layer makes the same choice for the frontier).
struct NavCostmap {
    private(set) var blocked: [Bool]
    let grid: NavGrid

    init(grid: NavGrid, robotRadius: Float = 0.33) {
        self.grid = grid
        var b = [Bool](repeating: false, count: grid.cells.count)
        let pad = max(1, Int((robotRadius / grid.resolution).rounded(.up)))
        for r in 0..<grid.height {
            for c in 0..<grid.width where grid.value(c: c, r: r) == NavGrid.occupied {
                for dr in -pad...pad {
                    for dc in -pad...pad where dr * dr + dc * dc <= pad * pad {
                        let rr = r + dr, cc = c + dc
                        guard rr >= 0, rr < grid.height, cc >= 0, cc < grid.width else { continue }
                        b[rr * grid.width + cc] = true
                    }
                }
            }
        }
        blocked = b
    }

    func isBlocked(c: Int, r: Int) -> Bool {
        guard c >= 0, c < grid.width, r >= 0, r < grid.height else { return true }  // off-map = wall
        return blocked[r * grid.width + c]
    }

    /// World-frame (x, z) query — the one the rollouts use.
    func isBlocked(x: Float, z: Float) -> Bool {
        guard let (c, r) = grid.cell(x: x, z: z) else { return true }
        return blocked[r * grid.width + c]
    }
}

// MARK: - Global plan (A*)

enum NavPlanner {

    /// 8-connected A* from `start` to `goal` (world x,z metres) over the inflated
    /// costmap. Returns world-frame waypoints INCLUDING the goal, or [] if the goal
    /// is unreachable (e.g. tapped inside a wall) — the caller must not drive then.
    static func plan(from start: CGPoint, to goal: CGPoint, costmap: NavCostmap) -> [CGPoint] {
        let g = costmap.grid
        guard let s = g.cell(x: Float(start.x), z: Float(start.y)),
              var t = g.cell(x: Float(goal.x), z: Float(goal.y)) else { return [] }

        // A goal a few cm inside the inflation skirt is the common case (tapping the
        // floor right next to a table). Nudge to the closest free cell rather than
        // declaring failure; a goal deep inside geometry still fails, as it should.
        if costmap.isBlocked(c: t.c, r: t.r) {
            guard let near = nearestFree(to: t, costmap: costmap, maxRadius: 8) else { return [] }
            t = near
        }
        if costmap.isBlocked(c: s.c, r: s.r) {
            // Robot itself is inside the skirt (started against a wall) — plan from
            // the nearest free cell so it can escape instead of refusing to move.
            guard let near = nearestFree(to: s, costmap: costmap, maxRadius: 8) else { return [] }
            return plan(from: g.world(c: near.c, r: near.r), to: goal, costmap: costmap)
        }

        let n = g.width * g.height
        let sIdx = s.r * g.width + s.c, tIdx = t.r * g.width + t.c
        var gScore = [Float](repeating: .infinity, count: n)
        var came = [Int](repeating: -1, count: n)
        var closed = [Bool](repeating: false, count: n)
        gScore[sIdx] = 0

        func h(_ i: Int) -> Float {
            let c = i % g.width, r = i / g.width
            let dc = Float(c - t.c), dr = Float(r - t.r)
            return (dc * dc + dr * dr).squareRoot()
        }

        // Binary heap keyed on f = g + h.
        var heap: [(f: Float, i: Int)] = [(h(sIdx), sIdx)]
        func push(_ e: (f: Float, i: Int)) {
            heap.append(e)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                if heap[p].f <= heap[i].f { break }
                heap.swapAt(p, i); i = p
            }
        }
        func pop() -> (f: Float, i: Int)? {
            guard !heap.isEmpty else { return nil }
            heap.swapAt(0, heap.count - 1)
            let out = heap.removeLast()
            var i = 0
            while true {
                let l = 2 * i + 1, r = 2 * i + 2
                var m = i
                if l < heap.count, heap[l].f < heap[m].f { m = l }
                if r < heap.count, heap[r].f < heap[m].f { m = r }
                if m == i { break }
                heap.swapAt(i, m); i = m
            }
            return out
        }

        let neighbours: [(Int, Int, Float)] = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142),
        ]

        while let cur = pop() {
            if closed[cur.i] { continue }
            closed[cur.i] = true
            if cur.i == tIdx { break }
            let c = cur.i % g.width, r = cur.i / g.width
            for (dc, dr, step) in neighbours {
                let nc = c + dc, nr = r + dr
                guard !costmap.isBlocked(c: nc, r: nr) else { continue }
                // No corner-cutting between two diagonal obstacles.
                if dc != 0, dr != 0,
                   costmap.isBlocked(c: c + dc, r: r) || costmap.isBlocked(c: c, r: r + dr) { continue }
                let ni = nr * g.width + nc
                let tentative = gScore[cur.i] + step
                if tentative < gScore[ni] {
                    gScore[ni] = tentative
                    came[ni] = cur.i
                    push((tentative + h(ni), ni))
                }
            }
        }
        guard gScore[tIdx] < .infinity else { return [] }        // unreachable

        var path: [CGPoint] = []
        var i = tIdx
        while i != -1 {
            path.append(g.world(c: i % g.width, r: i / g.width))
            if i == sIdx { break }
            i = came[i]
        }
        return path.reversed()
    }

    private static func nearestFree(to cell: (c: Int, r: Int), costmap: NavCostmap,
                                    maxRadius: Int) -> (c: Int, r: Int)? {
        for rad in 1...maxRadius {
            for dr in -rad...rad {
                for dc in -rad...rad where max(abs(dr), abs(dc)) == rad {
                    let c = cell.c + dc, r = cell.r + dr
                    if !costmap.isBlocked(c: c, r: r) { return (c, r) }
                }
            }
        }
        return nil
    }
}

// MARK: - Local controller (MPPI)

/// Model-Predictive Path Integral controller for the holonomic (swerve) base.
///
/// Each step: perturb the previous control sequence K ways, roll each out through
/// the base model, score it (obstacle / path / goal / effort), then take the
/// softmax-weighted mean and emit its first control. That is the whole method —
/// the value is that obstacle cost is evaluated on the ROLLOUT, so the robot slows
/// and arcs around furniture instead of discovering it by driving through it.
struct MPPINavigator {

    // Tuning. Horizon × dt = 1.2 s of lookahead at the demo's 20 Hz.
    var samples = 96
    var horizon = 24
    var dt: Float = 0.05
    var lambda: Float = 0.6            // softmax temperature (lower = greedier)
    var sigma: Float = 0.45            // exploration, m/s
    var wObstacle: Float = 1000        // hard: a colliding rollout must always lose
    var wPath: Float = 6               // track the A* corridor
    var wGoal: Float = 2.5             // make progress to the goal
    var wEffort: Float = 0.08          // prefer smooth, low-effort motion
    var lookahead: Float = 0.9         // m along the path to aim at

    /// Previous optimal sequence (warm start — MPPI's shift-and-reuse).
    private var uPrev: [SIMD2<Float>]
    private var rng: SplitMix64

    init(seed: UInt64 = 0x5EED) {
        rng = SplitMix64(seed: seed)
        uPrev = [SIMD2<Float>](repeating: .zero, count: horizon)
    }

    /// One control step.
    /// - Returns: world-frame velocity (vx, vz) in m/s. `.zero` means "stay put"
    ///   (arrived, or every rollout collides).
    mutating func step(x: Float, z: Float, path: [CGPoint], goal: CGPoint,
                       costmap: NavCostmap, vMax: Float) -> SIMD2<Float> {
        if uPrev.count != horizon { uPrev = [SIMD2<Float>](repeating: .zero, count: horizon) }
        let p0 = SIMD2<Float>(x, z)
        let goalV = SIMD2<Float>(Float(goal.x), Float(goal.y))

        // Aim at a point `lookahead` metres along the remaining path (pure-pursuit
        // style), so MPPI tracks the A* corridor instead of cutting to the goal
        // through whatever happens to lie between.
        let aim = aimPoint(from: p0, path: path, goal: goalV)

        var costs = [Float](repeating: 0, count: samples)
        var noise = [[SIMD2<Float>]](repeating: [], count: samples)

        for k in 0..<samples {
            var eps = [SIMD2<Float>](); eps.reserveCapacity(horizon)
            var p = p0
            var cost: Float = 0
            var dead = false
            for t in 0..<horizon {
                let e = SIMD2<Float>(rng.gaussian() * sigma, rng.gaussian() * sigma)
                eps.append(e)
                if dead { continue }
                var u = uPrev[t] + e
                let sp = (u * u).sum().squareRoot()
                if sp > vMax { u *= vMax / sp }                     // respect the speed cap
                p += u * dt
                if costmap.isBlocked(x: p.x, z: p.y) {
                    cost += wObstacle
                    dead = true                                     // rollout is over; it lost
                    continue
                }
                cost += wPath * dist(p, aim) * dt
                cost += wEffort * (u * u).sum() * dt
            }
            if !dead { cost += wGoal * dist(p, goalV) }             // terminal: close the gap
            costs[k] = cost
            noise[k] = eps
        }

        // Softmax weights over the sampled costs.
        let minC = costs.min() ?? 0
        var w = [Float](repeating: 0, count: samples)
        var wsum: Float = 0
        for k in 0..<samples {
            w[k] = exp(-(costs[k] - minC) / lambda)
            wsum += w[k]
        }
        guard wsum > 0, wsum.isFinite else { return .zero }

        for t in 0..<horizon {
            var acc = SIMD2<Float>.zero
            for k in 0..<samples { acc += w[k] * noise[k][t] }
            uPrev[t] += acc / wsum
            let sp = (uPrev[t] * uPrev[t]).sum().squareRoot()
            if sp > vMax { uPrev[t] *= vMax / sp }
        }

        let u0 = uPrev[0]
        // Shift the sequence (warm start for the next tick).
        uPrev.removeFirst()
        uPrev.append(.zero)

        // If the best we can do is drive into something, hold position — the caller
        // shows the goal as unreachable rather than grinding along a wall.
        if costs.min() ?? 0 >= wObstacle { return .zero }
        return u0
    }

    /// Reset the warm start (new goal → don't inherit the old plan's momentum).
    mutating func reset() {
        uPrev = [SIMD2<Float>](repeating: .zero, count: horizon)
    }

    private func aimPoint(from p: SIMD2<Float>, path: [CGPoint],
                          goal: SIMD2<Float>) -> SIMD2<Float> {
        guard !path.isEmpty else { return goal }
        // closest point on the path, then walk forward `lookahead` metres
        var best = 0
        var bestD = Float.greatestFiniteMagnitude
        for (i, wp) in path.enumerated() {
            let d = dist(p, SIMD2<Float>(Float(wp.x), Float(wp.y)))
            if d < bestD { bestD = d; best = i }
        }
        var travelled: Float = 0
        var i = best
        while i + 1 < path.count, travelled < lookahead {
            let a = SIMD2<Float>(Float(path[i].x), Float(path[i].y))
            let b = SIMD2<Float>(Float(path[i + 1].x), Float(path[i + 1].y))
            travelled += dist(a, b)
            i += 1
        }
        return SIMD2<Float>(Float(path[i].x), Float(path[i].y))
    }

    private func dist(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let d = a - b
        return (d * d).sum().squareRoot()
    }
}

// MARK: - Deterministic sampling

/// SplitMix64 + Box–Muller. Seeded so every rollout sweep is reproducible (the
/// nav tests assert on the resulting trajectory, which a system RNG would make flaky).
struct SplitMix64 {
    private var s: UInt64
    private var spare: Float?

    init(seed: UInt64) { s = seed }

    mutating func next() -> UInt64 {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    mutating func unit() -> Float {
        Float((next() >> 40) &+ 1) / Float(1 << 24)
    }

    /// Standard normal.
    mutating func gaussian() -> Float {
        if let g = spare { spare = nil; return g }
        let u1 = unit(), u2 = unit()
        let r = (-2 * log(u1)).squareRoot()
        let theta = 2 * Float.pi * u2
        spare = r * sin(theta)
        return r * cos(theta)
    }
}
