//
//  MotionModel.swift
//  MABEL Mobile App
//
//  A faithful Swift port of the simulation's base motion model
//  (controller/mabel/config.py — driven by scripts/teleop_keyboard.py). It
//  is the SINGLE source of base dynamics for the app's on-device sim: the
//  tip-safe `safeBaseStep` smooths a target twist into the achievable one,
//  with per-axis speed/acceleration caps that SHRINK as the lift rises
//  (higher CoM → easier to tip), plus the coupled ZMP/support-polygon clamp.
//
//  Conventions match the sim exactly:  forward = −vx, left = −vy, front = −X.
//

import Foundation

enum MotionModel {

    // Ceilings (the tip model lowers them with lift).
    static let maxLinSpeed = 0.8        // m/s
    static let maxAngSpeed = 2.0        // rad/s
    static let maxLinAccel = 2.0        // m/s^2
    static let maxAngAccel = 5.0        // rad/s^2
    static let liftRange: ClosedRange<Double> = 0.0...0.635
    static let liftSpeed = 0.20         // m/s manual jog (NAV_MAX_LIFT)

    // Navigation joystick → twist scaling.
    static let navMaxVx = 0.8, navMaxVy = 0.8, navMaxOmega = 1.2

    // Tip-over geometry (from mabel_sim.xml).
    private static let g = 9.81
    private static let leverFwd = 0.094, leverLat = 0.103, leverBack = 0.225
    private static let sfFwd = 2.0, sfAgile = 1.2
    private static let stopFwd = 0.80, stopAgile = 0.55
    private static let comZ0 = 0.443, comDzDLift = 0.432
    private static let vSpeedFloor = 0.15, vFwdCeil = 0.8, vAgileCeil = 1.50

    private static let supportPts: [(Double, Double)] = [(-0.161, -0.174), (-0.147, 0.166), (0.165, -0.017)]
    private static let comXY = (-0.060, -0.005)

    static func comHeight(_ zLift: Double) -> Double {
        comZ0 + comDzDLift * min(max(zLift, liftRange.lowerBound), liftRange.upperBound)
    }
    private static func axisDecel(_ lever: Double, _ sf: Double, _ z: Double) -> Double {
        g * lever / (sf * comHeight(z))
    }
    /// Tip-safe |accel| caps (fwd, back, lat).
    static func maxDecelDir(_ z: Double) -> (Double, Double, Double) {
        (axisDecel(leverFwd, sfFwd, z), axisDecel(leverBack, sfAgile, z), axisDecel(leverLat, sfAgile, z))
    }
    /// Tip-safe speed caps (fwd, back, lat).
    static func maxSpeedDir(_ z: Double) -> (Double, Double, Double) {
        let (af, ab, al) = maxDecelDir(z)
        func clamp(_ v: Double, _ hi: Double) -> Double { min(max(v, vSpeedFloor), hi) }
        return (clamp(af * stopFwd, vFwdCeil), clamp(ab * stopAgile, vAgileCeil), clamp(al * stopAgile, vAgileCeil))
    }

    // Support polygon edges (outward normal, lever d, safety factor).
    private static let edges: [(nx: Double, ny: Double, d: Double, sf: Double)] = {
        let pts = supportPts
        let ctr = (pts.reduce(0){$0+$1.0}/3, pts.reduce(0){$0+$1.1}/3)
        var out: [(Double, Double, Double, Double)] = []
        for i in 0..<3 {
            let a = pts[i], b = pts[(i + 1) % 3]
            let e = (b.0 - a.0, b.1 - a.1)
            var n = (e.1, -e.0)
            let nl = (n.0*n.0 + n.1*n.1).squareRoot(); n = (n.0/nl, n.1/nl)
            if n.0*(a.0 - ctr.0) + n.1*(a.1 - ctr.1) < 0 { n = (-n.0, -n.1) }
            let d = n.0*(a.0 - comXY.0) + n.1*(a.1 - comXY.1)
            let sf = (n.0 * -1.0) > 0.5 ? sfFwd : sfAgile   // front = −X
            out.append((n.0, n.1, d, sf))
        }
        return out
    }()

    private static func comAccel(_ vx: Double, _ vy: Double, _ w: Double, _ ax: Double, _ ay: Double, _ aw: Double) -> (Double, Double) {
        let rx = comXY.0, ry = comXY.1
        return ((ax - w*vy) - aw*ry - w*w*rx, (ay + w*vx) + aw*rx - w*w*ry)
    }
    private static func tipUsage(_ vx: Double, _ vy: Double, _ w: Double, _ ax: Double, _ ay: Double, _ aw: Double, _ z: Double) -> Double {
        let (acx, acy) = comAccel(vx, vy, w, ax, ay, aw)
        let h = comHeight(z); var u = 0.0
        for e in edges {
            let toward = -(e.nx*acx + e.ny*acy)
            if toward > 0 { u = max(u, toward / (g * e.d / (e.sf * h))) }
        }
        return u
    }
    private static func accelSafeScale(_ vx: Double, _ vy: Double, _ w: Double, _ ax: Double, _ ay: Double, _ aw: Double, _ z: Double) -> Double {
        let u0 = tipUsage(vx, vy, w, 0, 0, 0, z); let cap = max(1.0, u0)
        if tipUsage(vx, vy, w, ax, ay, aw, z) <= cap { return 1 }
        var lo = 0.0, hi = 1.0
        for _ in 0..<12 {
            let mid = 0.5 * (lo + hi)
            if tipUsage(vx, vy, w, mid*ax, mid*ay, mid*aw, z) <= cap { lo = mid } else { hi = mid }
        }
        return lo
    }
    private static func tipSpeedScale(_ vx: Double, _ vy: Double, _ w: Double, _ z: Double) -> Double {
        let u = tipUsage(vx, vy, w, 0, 0, 0, z)
        return u <= 1 ? 1 : pow(u, -0.5)
    }

    /// One tip-safe smoothing step. Returns the new (vx, vy, omega).
    static func safeBaseStep(_ vx: Double, _ vy: Double, _ omega: Double,
                             _ tvx: Double, _ tvy: Double, _ tw: Double,
                             dt: Double, zLift: Double) -> (Double, Double, Double) {
        let (aFwd, aBack, aLat) = maxDecelDir(zLift)
        let (vFwd, vBack, vLat) = maxSpeedDir(zLift)
        var txv = max(-vFwd, min(vBack, tvx))   // forward = −vx, so −vFwd … +vBack
        var tyv = max(-vLat, min(vLat, tvy))
        var twv = tw
        let s = tipSpeedScale(txv, tyv, twv, zLift)
        txv *= s; tyv *= s; twv *= s
        let axCap = vx < txv ? aFwd : aBack
        let cvx = vx < txv ? min(txv, vx + axCap*dt) : max(txv, vx - axCap*dt)
        let cvy = vy < tyv ? min(tyv, vy + aLat*dt)  : max(tyv, vy - aLat*dt)
        let cw  = omega < twv ? min(twv, omega + maxAngAccel*dt) : max(twv, omega - maxAngAccel*dt)
        let al = accelSafeScale(vx, vy, omega, (cvx-vx)/dt, (cvy-vy)/dt, (cw-omega)/dt, zLift)
        return (vx + al*(cvx-vx), vy + al*(cvy-vy), omega + al*(cw-omega))
    }
}
