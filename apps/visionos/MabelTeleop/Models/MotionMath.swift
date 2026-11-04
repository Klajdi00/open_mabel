//
//  MotionMath.swift
//  MabelTeleop (visionOS)
//
//  The parts of the studio's motion math that are PARAMETRIC, and so cannot be
//  baked into a table: the nav primitives (amplitude and length are functions of
//  a distance or angle the operator sets on the block) and the safety blend (a
//  function of the two poses it bridges). Everything else is sampled straight
//  out of `motions.js` — see `MotionClips.swift`.
//
//  Every constant here is CHECKED against the registry rather than remembered:
//  `motion_clips.json` carries the studio's own values under `consts`, and
//  `MotionClipTests` fails if these drift from them. Get a number wrong in a
//  safety budget and nothing looks broken until the robot moves.
//

import Foundation

/// The nav wire's four stick axes, in the studio's units (see motions.js:
/// `ly` = +forward, `lx` = +LEFT strafe, `rx` = +CCW yaw, `ry` = lift rate).
struct NavStick: Equatable {
    var lx: Double = 0
    var ly: Double = 0
    var rx: Double = 0
    var ry: Double = 0

    static let zero = NavStick()
    var isZero: Bool { abs(lx) < 1e-9 && abs(ly) < 1e-9 && abs(rx) < 1e-9 && abs(ry) < 1e-9 }
}

enum MotionMath {

    // MARK: - calibration (measured live; motions.js header)

    /// m/s of base travel per unit of forward stick.
    static let navSpeed = 0.8
    /// m/s per unit of strafe stick — the swerve base is slightly faster sideways.
    static let navSpeedStrafe = 0.87
    /// rad/s per unit of yaw stick.
    static let navYawRate = 1.5

    // MARK: - the velocity budget

    /// Peak per-joint speed a commanded blend may reach, rad/s. The SAME number
    /// as `ArmPlan.safeBlendVel` and the studio's `SAFE_BLEND_VEL` — one budget
    /// for commanded motion across every MABEL client (pinned by a test).
    static let safeBlendVel = 1.0
    /// Peak/average velocity ratio of `ease` BELOW. A raised cosine peaks at
    /// exactly π/2 × its average. `ArmPlan` eases with a smoothstep and so uses
    /// 1.5 — different curve, same budget. Change `ease` and re-derive this, or
    /// the cap silently weakens.
    static let easePeak = Double.pi / 2
    /// The clip library's own budget at the executor's sample rate.
    static let maxJointVel = 2.5
    /// The executor's command rate. It matters to the guard, not just to the
    /// baker: the robot does not receive a raised cosine, it receives 20 samples
    /// a second OF one, and the biggest step between two of those is what a
    /// servo answers.
    /// MIRRORS the Control Center's EXEC_RATE (control_center/web/js/motions.js), raised from 20
    /// to 50 on 2026-09-24. Everything derived below gets MORE conservative when it rises —
    /// blendThresh shrinks and easedPeakRate is sampled finer — so this is safe by construction.
    static let execRate = 50.0
    /// Pose jumps under this are benign: one tick of them is still inside the
    /// budget, so the servo absorbs it and a bridge would only add latency.
    /// DERIVED, not chosen — `maxJointVel / execRate`. It used to be 0.15, and
    /// those two numbers disagreed: 0.15 rad in one 50 ms tick is 3.0 rad/s, so
    /// a jump sitting exactly on the threshold was waved through at 1.2x the
    /// budget it was meant to protect. `pose_tuck` moves a shoulder by exactly
    /// 0.15 rad and found it.
    static let blendThresh = maxJointVel / execRate
    /// The timeline's snap grid, and the step the guard grows a bridge by.
    static let snapGrid: TimeInterval = 0.1
    /// Floor on a blend's length. A shorter "blend" is a step change wearing a
    /// costume — never round down to it.
    static let blendMinDur: TimeInterval = 0.4
    /// Ceiling, so one pathological jump cannot stall a timeline for a minute.
    static let blendMaxDur: TimeInterval = 8.0

    /// Radians of finger travel per unit of whole-hand grip, so a grip change is
    /// weighed against joint angles on one scale.
    static let gripScale = 1.9
    /// The lift is metres, not radians. 0.318 m of travel is a whole-body move,
    /// so it is weighed like a 3.18 rad joint.
    static let liftRadPerM = 10.0

    /// Per-joint weights that put grips and the lift onto the joint-angle scale.
    static let blendScale: [String: Double] = [
        "left_grip": gripScale, "right_grip": gripScale,
        "lift_lower": liftRadPerM, "lift_upper": liftRadPerM,
    ]

    // MARK: - easing

    /// Raised cosine, 0→1. The studio's `ease`; every clip and every blend in the
    /// registry is shaped by it, so the app must ease identically or a blend
    /// sized here plays at a different peak speed there.
    static func ease(_ u: Double) -> Double {
        let x = min(1, max(0, u))
        return 0.5 - 0.5 * cos(.pi * x)
    }

    /// Peak per-tick rate of the eased interpolation, per unit of pose distance,
    /// AS SAMPLED at the executor's rate. For a long bridge this tends to
    /// `easePeak / dur`; for a short one the discrete peak is meaningfully
    /// higher, and it is the discrete one the servo sees.
    static func easedPeakRate(_ dur: TimeInterval) -> Double {
        let n = max(1, Int((dur * execRate).rounded()))
        var worst = 0.0
        for i in 0..<n {
            let d = ease(Double(i + 1) / Double(n)) - ease(Double(i) / Double(n))
            if d > worst { worst = d }
        }
        return worst * execRate
    }


    /// Eased 0→1 ramp across [t0, t1].
    static func seg(_ t: Double, _ t0: Double, _ t1: Double) -> Double {
        ease((t - t0) / max(1e-6, t1 - t0))
    }

    /// Eased 0→1→0 window: up over [t0, t0+r], down over [t1−r, t1].
    static func win(_ t: Double, _ t0: Double, _ t1: Double, _ r: Double = 0.5) -> Double {
        guard t > t0, t < t1 else { return 0 }
        return min(seg(t, t0, t0 + r), 1 - seg(t, t1 - r, t1))
    }

    // MARK: - nav primitives

    /// The studio's `navProfile`: ease up, cruise, ease down, unit peak, over a
    /// block of length `d`. A block that starts at full speed is a step command
    /// to the base.
    static func navProfile(_ t: Double, _ d: Double) -> Double {
        win(t, 0, d, min(0.6, d * 0.3))
    }

    /// Length of a nav block that should cover `ideal` seconds of cruising.
    /// `win`'s two half-ramps lose ≈ r seconds of area between them, so the
    /// block is that much longer than the naive distance ÷ speed.
    static func navDur(_ ideal: Double) -> TimeInterval {
        max(0.8, ideal + 0.6)
    }

    /// One parametric nav primitive: which axis it drives, with what sign, and
    /// how its length follows its parameter. Amplitude is 0.5 of a full stick —
    /// the studio drives these at half speed so a choreography is rehearsable.
    struct NavPrimitive {
        enum Axis { case lx, ly, rx }
        let axis: Axis
        let sign: Double
        /// Parameter this clip is sized by, and the divisor that turns it into
        /// seconds of cruising at half stick.
        let param: String
        let unitPerSecond: Double
        /// `angle` parameters arrive in degrees.
        let degrees: Bool

        func duration(_ params: [String: Double], default def: Double) -> TimeInterval {
            var v = params[param] ?? def
            if degrees { v = v * .pi / 180 }
            return navDur(v / unitPerSecond)
        }

        func stick(at t: TimeInterval, duration d: TimeInterval) -> NavStick {
            let a = sign * 0.5 * navProfile(t, d)
            switch axis {
            case .lx: return NavStick(lx: a)
            case .ly: return NavStick(ly: a)
            case .rx: return NavStick(rx: a)
            }
        }
    }

    /// The seven parametric nav clips, exactly as `motions.js` declares them.
    /// `nav_hold` has no primitive (it streams zeros for its `secs`); `nav_goto`
    /// is a goal, not a stream.
    static let navPrimitives: [String: NavPrimitive] = [
        "nav_forward":      .init(axis: .ly, sign:  1, param: "dist",  unitPerSecond: navSpeed * 0.5,   degrees: false),
        "nav_back":         .init(axis: .ly, sign: -1, param: "dist",  unitPerSecond: navSpeed * 0.5,   degrees: false),
        "nav_strafe_left":  .init(axis: .lx, sign:  1, param: "dist",  unitPerSecond: navSpeed * 0.5,   degrees: false),
        "nav_strafe_right": .init(axis: .lx, sign: -1, param: "dist",  unitPerSecond: navSpeed * 0.5,   degrees: false),
        "nav_turn_left":    .init(axis: .rx, sign:  1, param: "angle", unitPerSecond: navYawRate * 0.5, degrees: true),
        "nav_turn_right":   .init(axis: .rx, sign: -1, param: "angle", unitPerSecond: navYawRate * 0.5, degrees: true),
    ]

    // MARK: - the safety blend

    /// Worst-joint pose distance in radian-equivalents (grips and the lift are
    /// weighed by `blendScale`). A joint the start pose does not name cannot
    /// jump — we do not know where it was.
    static func poseDelta(from: [String: Double], to: [String: Double]) -> (delta: Double, joint: String?) {
        var delta = 0.0
        var joint: String?
        for (k, v) in to {
            guard let a = from[k] else { continue }
            let d = abs(v - a) * (blendScale[k] ?? 1)
            if d > delta { delta = d; joint = k }
        }
        return (delta, joint)
    }

    /// Shortest blend that keeps the PEAK velocity of the eased interpolation
    /// (easePeak × the average) under `safeBlendVel`. Rounded UP to a
    /// centisecond: a safety floor must never round down.
    static func blendDuration(for delta: Double) -> TimeInterval {
        max(blendMinDur, min(blendMaxDur, (100 * easePeak * delta / safeBlendVel).rounded(.up) / 100))
    }

    /// The blend's pose at phase `t` of a `d`-second bridge. Joints present only
    /// in `from` HOLD — a blend must never invent motion on a joint the next
    /// block does not command.
    static func blendPose(from: [String: Double], to: [String: Double],
                          at t: TimeInterval, duration d: TimeInterval) -> [String: Double] {
        let u = ease(t / max(blendMinDur, d))
        var out = from
        for (k, b) in to {
            out[k] = from[k].map { $0 + (b - $0) * u } ?? b
        }
        return out
    }

    /// Linear pose interpolation — the keyframe lane's rule (`sequence.js`).
    static func lerpPose(_ a: [String: Double], _ b: [String: Double], _ u: Double) -> [String: Double] {
        var out = a
        for (k, vb) in b {
            out[k] = a[k].map { $0 + (vb - $0) * u } ?? vb
        }
        return out
    }
}
