//
//  PurePursuit.swift
//  MABEL Mobile App
//
//  Path follower for the map's MABEL. The base is holonomic, so we
//  translate straight toward the look-ahead point; separately we rotate the
//  chassis so its FRONT faces the velocity (so the operator can read where
//  it's going), rate-limited because steering the swerve modules is slow.
//
//  Yaw is tracked as the SceneKit node yaw directly, so planning a route
//  never changes the robot's orientation — only driving does. The body-frame
//  twist (vx, vy, ω) feeds the swerve IK (SwerveDrive), matching the sim's
//  motion-model limits.
//

import CoreGraphics
import Foundation

struct PurePursuit {
    static let maxLinSpeed: Double = 0.8      // m/s   (sim MAX_LIN_SPEED)
    static let maxLinAccel: Double = 1.5      // m/s^2
    static let omegaMax:    Double = 1.5      // rad/s — rate-limited turn
    static let omegaGain:   Double = 2.5      // how hard it tries to face velocity
    static let lookahead:   Double = 0.55     // m
    static let goalRadius:  Double = 0.12     // m

    // World pose in the room frame.
    var x: Double = 0
    var z: Double = 0
    var yaw: Double = 0        // SceneKit node yaw (front = +Z rotated by yaw)
    var speed: Double = 0
    // Body-frame twist (for the swerve IK).
    private(set) var bodyVx: Double = 0
    private(set) var bodyVy: Double = 0
    private(set) var omega: Double = 0

    private(set) var path: [CGPoint] = []
    private var seg = 0
    private(set) var finished = false

    /// Start from the robot's current pose (so a re-plan continues from
    /// where it is, keeping its current orientation — no snap).
    mutating func start(path: [CGPoint], from start: CGPoint, yaw: Double) {
        self.path = path
        self.seg = 0
        self.finished = path.count < 2
        self.speed = 0; self.omega = 0; self.bodyVx = 0; self.bodyVy = 0
        x = Double(start.x); z = Double(start.y)
        self.yaw = yaw
    }

    mutating func step(dt: Double) {
        guard !finished, let goal = path.last else { return }
        let distToGoal = hypot(Double(goal.x) - x, Double(goal.y) - z)
        if distToGoal < Self.goalRadius {
            finished = true; speed = 0; omega = 0; bodyVx = 0; bodyVy = 0; return
        }

        // Look-ahead target.
        while seg < path.count - 1 {
            let p = path[seg]
            if hypot(Double(p.x) - x, Double(p.y) - z) < Self.lookahead { seg += 1 } else { break }
        }
        let target = path[min(seg, path.count - 1)]
        let ddx = Double(target.x) - x, ddz = Double(target.y) - z
        let d = max(1e-6, hypot(ddx, ddz))
        let dirX = ddx / d, dirZ = ddz / d

        // Speed: ramp up, ease down near the goal.
        let targetSpeed = min(Self.maxLinSpeed, max(0.12, distToGoal))
        if speed < targetSpeed { speed = min(targetSpeed, speed + Self.maxLinAccel * dt) }
        else { speed = max(targetSpeed, speed - Self.maxLinAccel * dt) }

        // Holonomic translation toward the look-ahead (world frame).
        let wvx = dirX * speed, wvz = dirZ * speed
        x += wvx * dt; z += wvz * dt

        // Turn the FRONT (+Z at yaw 0) toward the velocity direction, rate-limited.
        let desiredYaw = atan2(wvx, wvz)
        var e = desiredYaw - yaw
        while e >  .pi { e -= 2 * .pi }
        while e < -.pi { e += 2 * .pi }
        omega = max(-Self.omegaMax, min(Self.omegaMax, Self.omegaGain * e))
        yaw += omega * dt

        // World velocity expressed in the chassis (front / left) frame.
        let sy = sin(yaw), cy = cos(yaw)
        bodyVx =  wvx * sy + wvz * cy     // forward (+Z)
        bodyVy =  wvx * cy - wvz * sy     // left
    }
}
