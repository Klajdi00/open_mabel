//
//  NavigationTests.swift
//  MabelTeleopTests (visionOS)
//
//  On-device navigation for demo/offline mode (Models/MPPINavigator.swift), shared
//  byte-for-byte with the iOS app. Before this the headset demo "navigated" with a
//  waypoint pursuit ("MPPI feel"); it is real MPPI now — A* over the room's
//  occupancy (inflated by the robot radius) + an MPPI local controller that scores
//  rolled-out trajectories, so obstacles are avoided by the controller rather than
//  discovered by collision. The sampler is a seeded SplitMix64, so trajectories are
//  reproducible (a system RNG would make these tests flaky).
//

import XCTest
import CoreGraphics
@testable import MabelTeleop

final class NavigationTests: XCTestCase {

    // A 6×6 m room at 0.1 m/cell. Cell (0,0) corner is (-3, +3); +cols → +x, +rows → −z.
    private static let res: Float = 0.1
    private static let n = 60

    /// Empty room, optionally with a wall blocking world x∈[x0,x1], z∈[z0,z1].
    private func room(wall: (x0: Float, x1: Float, z0: Float, z1: Float)? = nil) -> NavGrid {
        var cells = [UInt8](repeating: NavGrid.free, count: Self.n * Self.n)
        var g = NavGrid(cells: cells, width: Self.n, height: Self.n,
                        resolution: Self.res, originX: -3, originZ: 3)
        if let w = wall {
            for r in 0..<Self.n {
                for c in 0..<Self.n {
                    let p = g.world(c: c, r: r)
                    let x = Float(p.x), z = Float(p.y)
                    if x >= w.x0, x <= w.x1, z >= w.z0, z <= w.z1 {
                        cells[r * Self.n + c] = NavGrid.occupied
                    }
                }
            }
            g.cells = cells
        }
        return g
    }

    // MARK: - Global plan (A*)

    func testPlanInEmptyRoom() {
        let cm = NavCostmap(grid: room())
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: CGPoint(x: 2, y: 2),
                                   costmap: cm)
        XCTAssertFalse(path.isEmpty, "no route in an empty room")
        let end = path.last!
        XCTAssertLessThan(hypot(end.x - 2, end.y - 2), 0.2, "route does not end at the goal")
    }

    func testPlanAvoidsTheWall() {
        // A wall across the middle with a gap on the right (x > 1.5).
        let g = room(wall: (x0: -3, x1: 1.5, z0: -0.15, z1: 0.15))
        let cm = NavCostmap(grid: g)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: 2), to: CGPoint(x: -2, y: -2),
                                   costmap: cm)
        XCTAssertFalse(path.isEmpty, "the gap makes this reachable — the planner missed it")
        for p in path {
            XCTAssertFalse(cm.isBlocked(x: Float(p.x), z: Float(p.y)),
                           "the route drives through the wall at \(p)")
        }
        XCTAssertTrue(path.contains { $0.x > 1.4 }, "the route teleported through the wall")
    }

    func testUnreachableGoalReturnsNoRoute() {
        let g = room(wall: (x0: 0.5, x1: 2.5, z0: -1, z1: 1))
        let cm = NavCostmap(grid: g)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: CGPoint(x: 1.5, y: 0),
                                   costmap: cm)
        XCTAssertTrue(path.isEmpty, "the planner invented a route into solid geometry")
    }

    func testObstaclesAreInflated() {
        let g = room(wall: (x0: 0, x1: 0.2, z0: 0, z1: 0.2))
        let cm = NavCostmap(grid: g, robotRadius: 0.33)
        XCTAssertTrue(cm.isBlocked(x: 0.1, z: 0.1), "the obstacle itself is not blocked")
        XCTAssertTrue(cm.isBlocked(x: -0.2, z: 0.1), "the obstacle was not inflated")
        XCTAssertFalse(cm.isBlocked(x: -1.5, z: -1.5), "free space was inflated away")
    }

    // MARK: - Local controller (MPPI)

    private func drive(from start: CGPoint, to goal: CGPoint, costmap: NavCostmap,
                       path: [CGPoint], seconds: Double = 30,
                       vMax: Float = 0.8) -> (end: CGPoint, collided: Bool, arrived: Bool) {
        var mppi = MPPINavigator(seed: 42)
        let dt: Float = 0.05
        var x = Float(start.x), z = Float(start.y)
        var collided = false, arrived = false
        for _ in 0..<Int(seconds / Double(dt)) {
            if hypot(Double(x) - goal.x, Double(z) - goal.y) < 0.12 { arrived = true; break }
            let u = mppi.step(x: x, z: z, path: path, goal: goal, costmap: costmap, vMax: vMax)
            x += u.x * dt; z += u.y * dt
            if costmap.isBlocked(x: x, z: z) { collided = true; break }
        }
        return (CGPoint(x: CGFloat(x), y: CGFloat(z)), collided, arrived)
    }

    func testMPPIReachesTheGoal() {
        let cm = NavCostmap(grid: room())
        let goal = CGPoint(x: 2, y: 2)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: goal, costmap: cm)
        let r = drive(from: CGPoint(x: -2, y: -2), to: goal, costmap: cm, path: path)
        XCTAssertTrue(r.arrived, "MPPI never reached the goal (ended at \(r.end))")
        XCTAssertFalse(r.collided)
    }

    func testMPPIAvoidsTheWall() {
        let g = room(wall: (x0: -3, x1: 1.5, z0: -0.15, z1: 0.15))
        let cm = NavCostmap(grid: g)
        let start = CGPoint(x: -2, y: 2), goal = CGPoint(x: -2, y: -2)
        let path = NavPlanner.plan(from: start, to: goal, costmap: cm)
        let r = drive(from: start, to: goal, costmap: cm, path: path, seconds: 60)
        XCTAssertFalse(r.collided, "the robot drove into the wall at \(r.end)")
        XCTAssertTrue(r.arrived, "the robot never got round the wall (ended at \(r.end))")
    }

    func testMPPIRespectsTheSpeedCap() {
        let cm = NavCostmap(grid: room())
        var mppi = MPPINavigator(seed: 7)
        let goal = CGPoint(x: 2.5, y: 2.5)
        let path = NavPlanner.plan(from: CGPoint(x: -2.5, y: -2.5), to: goal, costmap: cm)
        let vMax: Float = 0.4
        var x: Float = -2.5, z: Float = -2.5
        for _ in 0..<200 {
            let u = mppi.step(x: x, z: z, path: path, goal: goal, costmap: cm, vMax: vMax)
            let speed = (u.x * u.x + u.y * u.y).squareRoot()
            XCTAssertLessThanOrEqual(speed, vMax + 1e-3, "commanded \(speed) over a \(vMax) cap")
            x += u.x * 0.05; z += u.y * 0.05
        }
    }

    func testLiftHeightThrottlesTheNavigator() {
        let down = MotionModel.maxSpeedDir(0)
        let up = MotionModel.maxSpeedDir(0.60)
        XCTAssertLessThan(up.0, down.0, "raising the lift must reduce the forward speed cap")
        XCTAssertLessThan(up.2, down.2, "raising the lift must reduce the lateral speed cap")
    }

    func testMPPIIsReproducible() {
        let cm = NavCostmap(grid: room())
        let goal = CGPoint(x: 1.5, y: 1.5)
        let path = NavPlanner.plan(from: .zero, to: goal, costmap: cm)
        func firstControl(seed: UInt64) -> SIMD2<Float> {
            var m = MPPINavigator(seed: seed)
            return m.step(x: 0, z: 0, path: path, goal: goal, costmap: cm, vMax: 0.8)
        }
        XCTAssertEqual(firstControl(seed: 99), firstControl(seed: 99),
                       "the sampler is not reproducible — the nav tests would be flaky")
    }
}
