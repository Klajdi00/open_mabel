//
//  NavigationTests.swift
//  MABEL Mobile AppTests
//
//  On-device navigation for demo/offline mode (Models/MPPINavigator.swift).
//
//  Before this, the demo "navigated" by sliding the robot along a straight line to
//  the tapped goal — straight through walls and furniture (`setDemoGoal`: "the demo
//  has no planner"). Now it mirrors the real stack: A* over the room's occupancy
//  (inflated by the robot radius) for the global route, and an MPPI local controller
//  that rolls the base model forward and scores each rollout, so obstacles are
//  avoided by the CONTROLLER rather than discovered by collision.
//
//  The sampler is a seeded SplitMix64, so these trajectories are reproducible.
//

import Testing
import CoreGraphics
import Foundation
@testable import MABEL_Mobile_App

struct NavigationTests {

    // A 6×6 m room at 0.1 m/cell. Cell (0,0) corner is (-3, +3); +cols → +x, +rows → −z.
    private static let res: Float = 0.1
    private static let n = 60

    /// Empty room, optionally with a wall. `wall` blocks cells in world x ∈ [x0,x1],
    /// z ∈ [z0,z1].
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

    @Test("a clear room plans a route that reaches the goal")
    func planInEmptyRoom() {
        let cm = NavCostmap(grid: room())
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: CGPoint(x: 2, y: 2),
                                   costmap: cm)
        #expect(!path.isEmpty, "no route in an empty room")
        let end = path.last!
        #expect(hypot(end.x - 2, end.y - 2) < 0.2, "route does not end at the goal")
    }

    @Test("the planned route never crosses an obstacle")
    func planAvoidsTheWall() {
        // A wall across the middle with a gap on the right (x > 1.5).
        let g = room(wall: (x0: -3, x1: 1.5, z0: -0.15, z1: 0.15))
        let cm = NavCostmap(grid: g)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: 2), to: CGPoint(x: -2, y: -2),
                                   costmap: cm)
        #expect(!path.isEmpty, "the gap makes this reachable — the planner missed it")
        for p in path {
            #expect(!cm.isBlocked(x: Float(p.x), z: Float(p.y)),
                    "the route drives through the wall at \(p)")
        }
        // It must go AROUND: any route through the gap has to swing right of x = 1.4.
        #expect(path.contains { $0.x > 1.4 }, "the route teleported through the wall")
    }

    @Test("a goal sealed inside geometry is refused, not faked")
    func unreachableGoalReturnsNoRoute() {
        // Solid block; a goal in its middle cannot be reached.
        let g = room(wall: (x0: 0.5, x1: 2.5, z0: -1, z1: 1))
        let cm = NavCostmap(grid: g)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: CGPoint(x: 1.5, y: 0),
                                   costmap: cm)
        #expect(path.isEmpty, "the planner invented a route into solid geometry")
    }

    @Test("the costmap inflates obstacles by the robot radius")
    func obstaclesAreInflated() {
        let g = room(wall: (x0: 0, x1: 0.2, z0: 0, z1: 0.2))
        let cm = NavCostmap(grid: g, robotRadius: 0.33)
        #expect(cm.isBlocked(x: 0.1, z: 0.1), "the obstacle itself is not blocked")
        // A point 0.2 m from the obstacle is closer than the robot's radius: the
        // CENTRE may not go there, or the chassis clips the corner.
        #expect(cm.isBlocked(x: -0.2, z: 0.1), "the obstacle was not inflated")
        #expect(!cm.isBlocked(x: -1.5, z: -1.5), "free space was inflated away")
    }

    // MARK: - Local controller (MPPI)

    /// Drive the base with MPPI, exactly as the demo loop does, and report where it
    /// ends up plus whether it ever entered a blocked cell.
    private func drive(from start: CGPoint, to goal: CGPoint, costmap: NavCostmap,
                       path: [CGPoint], seconds: Double = 30,
                       vMax: Float = 0.8) -> (end: CGPoint, collided: Bool, arrived: Bool) {
        var mppi = MPPINavigator(seed: 42)
        let dt: Float = 0.05                     // the demo runs at 20 Hz
        var x = Float(start.x), z = Float(start.y)
        var collided = false
        var arrived = false
        for _ in 0..<Int(seconds / Double(dt)) {
            let d = hypot(Double(x) - goal.x, Double(z) - goal.y)
            if d < 0.12 { arrived = true; break }
            let u = mppi.step(x: x, z: z, path: path, goal: goal,
                              costmap: costmap, vMax: vMax)
            x += u.x * dt
            z += u.y * dt
            if costmap.isBlocked(x: x, z: z) { collided = true; break }
        }
        return (CGPoint(x: CGFloat(x), y: CGFloat(z)), collided, arrived)
    }

    @Test("MPPI drives to the goal across an empty room")
    func mppiReachesTheGoal() {
        let cm = NavCostmap(grid: room())
        let goal = CGPoint(x: 2, y: 2)
        let path = NavPlanner.plan(from: CGPoint(x: -2, y: -2), to: goal, costmap: cm)
        let r = drive(from: CGPoint(x: -2, y: -2), to: goal, costmap: cm, path: path)
        #expect(r.arrived, "MPPI never reached the goal (ended at \(r.end))")
        #expect(!r.collided)
    }

    @Test("MPPI follows the route around a wall instead of through it")
    func mppiAvoidsTheWall() {
        let g = room(wall: (x0: -3, x1: 1.5, z0: -0.15, z1: 0.15))
        let cm = NavCostmap(grid: g)
        let start = CGPoint(x: -2, y: 2), goal = CGPoint(x: -2, y: -2)
        let path = NavPlanner.plan(from: start, to: goal, costmap: cm)
        let r = drive(from: start, to: goal, costmap: cm, path: path, seconds: 60)
        #expect(!r.collided, "the robot drove into the wall at \(r.end)")
        #expect(r.arrived, "the robot never got round the wall (ended at \(r.end))")
    }

    @Test("MPPI never exceeds the commanded speed cap")
    func mppiRespectsTheSpeedCap() {
        let cm = NavCostmap(grid: room())
        var mppi = MPPINavigator(seed: 7)
        let goal = CGPoint(x: 2.5, y: 2.5)
        let path = NavPlanner.plan(from: CGPoint(x: -2.5, y: -2.5), to: goal, costmap: cm)
        let vMax: Float = 0.4
        var x: Float = -2.5, z: Float = -2.5
        for _ in 0..<200 {
            let u = mppi.step(x: x, z: z, path: path, goal: goal, costmap: cm, vMax: vMax)
            let speed = (u.x * u.x + u.y * u.y).squareRoot()
            #expect(speed <= vMax + 1e-3, "commanded \(speed) m/s over a \(vMax) cap")
            x += u.x * 0.05; z += u.y * 0.05
        }
    }

    @Test("the tip-over motion model slows the robot when the lift is raised")
    func liftHeightThrottlesTheNavigator() {
        // The demo's speed cap is the REAL robot's tip-over model at the current lift
        // height — the same MotionModel the whole stack uses.
        let down = MotionModel.maxSpeedDir(0)
        let up = MotionModel.maxSpeedDir(0.60)
        #expect(up.0 < down.0, "raising the lift must reduce the forward speed cap")
        #expect(up.2 < down.2, "raising the lift must reduce the lateral speed cap")
    }

    @Test("MPPI is deterministic — same seed, same trajectory")
    func mppiIsReproducible() {
        let cm = NavCostmap(grid: room())
        let goal = CGPoint(x: 1.5, y: 1.5)
        let path = NavPlanner.plan(from: .zero, to: goal, costmap: cm)
        func firstControl(seed: UInt64) -> SIMD2<Float> {
            var m = MPPINavigator(seed: seed)
            return m.step(x: 0, z: 0, path: path, goal: goal, costmap: cm, vMax: 0.8)
        }
        let a = firstControl(seed: 99), b = firstControl(seed: 99)
        #expect(a == b, "the sampler is not reproducible — the nav tests would be flaky")
    }
}

// MARK: - Sim ↔ real switching

@MainActor
struct RobotSwitchingTests {

    @Test("both robots are seeded and reachable every way in")
    func bothRobotsExist() {
        let store = RobotStore()
        let real = store.robots.first { $0.kind == .real }
        let sim = store.robots.first { $0.kind == .sim }
        #expect(real != nil, "the real robot is missing from the seeds")
        #expect(sim != nil, "the simulator is missing from the seeds")
        #expect(real?.routes.contains { NetworkConfig.classify($0) == .tailscale } == true,
                "the real robot must be reachable over Tailscale")
    }

    @Test("switching sim → real does not carry the previous robot's data across")
    func switchingClearsTheOldRobotsState() {
        let link = FakeLink()
        let c = RobotController(link: link)
        link.connect()
        link.emitLiveTelemetry()
        #expect(c.state.baseX == 3.5)

        guard let real = c.robotStore.robots.first(where: { $0.kind == .real }) else {
            Issue.record("no real robot seeded"); return
        }
        c.connect(to: real)

        // The new robot has said nothing yet — showing the OLD robot's pose here is
        // exactly the class of bug this suite exists to prevent.
        #expect(c.state.baseX == 0, "the previous robot's odom carried into the new one")
        #expect(c.lidarScan == nil, "the previous robot's LiDAR sweep carried over")
        #expect(c.livePose.baseX == 0, "the rig is still parked at the old robot's pose")
    }

    @Test("the active robot follows the switch")
    func activeRobotFollowsTheSwitch() {
        let c = RobotController(link: FakeLink())
        guard let sim = c.robotStore.robots.first(where: { $0.kind == .sim }),
              let real = c.robotStore.robots.first(where: { $0.kind == .real }) else {
            Issue.record("seeds missing"); return
        }
        c.connect(to: sim)
        #expect(c.robotStore.active?.id == sim.id)
        c.connect(to: real)
        #expect(c.robotStore.active?.id == real.id, "switching to the real robot didn't take")
        c.disconnect()
        #expect(c.robotStore.active == nil, "disconnect must leave no active robot")
    }
}
