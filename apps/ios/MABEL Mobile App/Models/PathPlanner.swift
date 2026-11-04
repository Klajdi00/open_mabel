//
//  PathPlanner.swift
//  MABEL Mobile App
//
//  A* global path planning over a 2D occupancy grid derived from a room's
//  furniture footprints. Used to route the base from its current pose to a
//  goal the operator drops on the map.
//

import Foundation
import CoreGraphics

enum PathPlanner {

    /// Plan on the XZ plane. `obstacles` are footprints (metres, room frame).
    /// Returns world-frame waypoints (x, z) including start and goal, or an
    /// empty array if no path exists.
    static func plan(from start: CGPoint, to goal: CGPoint,
                     bounds: CGSize, obstacles: [CGRect],
                     cell: CGFloat = 0.18, inflate: CGFloat = 0.28) -> [CGPoint] {

        let cols = max(1, Int(bounds.width / cell))
        let rows = max(1, Int(bounds.height / cell))
        let originX = -bounds.width / 2, originZ = -bounds.height / 2

        func toCell(_ p: CGPoint) -> (Int, Int) {
            (min(cols - 1, max(0, Int((p.x - originX) / cell))),
             min(rows - 1, max(0, Int((p.y - originZ) / cell))))
        }
        func toWorld(_ c: Int, _ r: Int) -> CGPoint {
            CGPoint(x: originX + (CGFloat(c) + 0.5) * cell, y: originZ + (CGFloat(r) + 0.5) * cell)
        }

        // Build blocked grid (obstacles inflated by robot radius).
        var blocked = [[Bool]](repeating: [Bool](repeating: false, count: rows), count: cols)
        for box in obstacles {
            let infl = box.insetBy(dx: -inflate, dy: -inflate)
            let (c0, r0) = toCell(CGPoint(x: infl.minX, y: infl.minY))
            let (c1, r1) = toCell(CGPoint(x: infl.maxX, y: infl.maxY))
            for c in min(c0, c1)...max(c0, c1) where c >= 0 && c < cols {
                for r in min(r0, r1)...max(r0, r1) where r >= 0 && r < rows {
                    blocked[c][r] = true
                }
            }
        }

        let s = toCell(start), g = toCell(goal)
        blocked[s.0][s.1] = false; blocked[g.0][g.1] = false

        // A* with 8-connectivity.
        struct Node { let c, r: Int }
        func h(_ a: (Int, Int), _ b: (Int, Int)) -> CGFloat {
            let dx = CGFloat(abs(a.0 - b.0)), dy = CGFloat(abs(a.1 - b.1))
            return hypot(dx, dy)
        }
        let startKey = s.0 * rows + s.1
        let goalKey = g.0 * rows + g.1
        var gScore: [Int: CGFloat] = [startKey: 0]
        var came = [Int: Int]()
        var open: Set<Int> = [startKey]
        var fScore: [Int: CGFloat] = [startKey: h(s, g)]

        let neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while !open.isEmpty {
            let current = open.min { (fScore[$0] ?? .infinity) < (fScore[$1] ?? .infinity) }!
            if current == goalKey {
                // Reconstruct
                var path = [Int]([current])
                var k = current
                while let p = came[k] { path.append(p); k = p }
                return path.reversed().map { key in
                    toWorld(key / rows, key % rows)
                }
            }
            open.remove(current)
            let cc = current / rows, cr = current % rows
            for (dc, dr) in neighbors {
                let nc = cc + dc, nr = cr + dr
                guard nc >= 0, nc < cols, nr >= 0, nr < rows, !blocked[nc][nr] else { continue }
                // Disallow diagonal corner-cutting.
                if dc != 0 && dr != 0, blocked[cc + dc][cr] || blocked[cc][cr + dr] { continue }
                let nKey = nc * rows + nr
                let tentative = (gScore[current] ?? .infinity) + hypot(CGFloat(dc), CGFloat(dr))
                if tentative < (gScore[nKey] ?? .infinity) {
                    came[nKey] = current
                    gScore[nKey] = tentative
                    fScore[nKey] = tentative + h((nc, nr), g)
                    open.insert(nKey)
                }
            }
        }
        return []   // no path
    }
}
