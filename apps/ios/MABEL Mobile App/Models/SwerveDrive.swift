//
//  SwerveDrive.swift
//  MABEL Mobile App
//
//  Swerve-drive inverse kinematics, ported from the simulation's
//  legacy/motion_model.py `swerve_ik` (3 modules: fl, fr, b). Given a body
//  twist (vx, vy, ω) it steers each module and spins its wheel — with the
//  same anti-slip + shortest-steer logic — so the map's MABEL rolls
//  realistically.
//

import Foundation

struct SwerveDrive {
    static let wheelRadius = 0.05
    /// Module positions (x forward, y left) relative to chassis centre.
    static let modules: [(key: String, x: Double, y: Double)] = [
        ("fl", 0.2, 0.2), ("fr", 0.2, -0.2), ("b", -0.2, 0.0)
    ]

    struct Module { var steer = 0.0; var wheel = 0.0 }
    var state: [String: Module] = ["fl": Module(), "fr": Module(), "b": Module()]

    /// Keep a steering column from spinning past ±90° (reverse the wheel instead).
    private func optimizeSteering(_ target: Double, _ current: Double) -> (Double, Double) {
        var t = target
        var dir = 1.0
        var diff = (t - current + .pi).truncatingRemainder(dividingBy: 2 * .pi) - .pi
        if diff > .pi / 2 { t -= .pi; dir = -1 }
        else if diff < -.pi / 2 { t += .pi; dir = -1 }
        diff = (t - current + .pi).truncatingRemainder(dividingBy: 2 * .pi) - .pi
        return (current + diff, dir)
    }

    /// Integrate one timestep of the given body twist.
    mutating func update(vx: Double, vy: Double, omega: Double, dt: Double) {
        for m in Self.modules {
            var mod = state[m.key] ?? Module()
            let vix = vx - omega * m.y
            let viy = vy + omega * m.x
            let vmag = (vix * vix + viy * viy).squareRoot()
            if vmag > 1e-3 {
                let raw = atan2(viy, vix)
                let (optimal, dir) = optimizeSteering(raw, mod.steer)
                let angleErr = abs(optimal - mod.steer)
                var driveScale = max(0.0, cos(angleErr))
                if angleErr > 0.45 { driveScale = 0 }
                let wheelOmega = (vmag / Self.wheelRadius) * dir * driveScale
                mod.steer = optimal
                mod.wheel += wheelOmega * dt
            }
            state[m.key] = mod
        }
    }

    func reset() {}
}
