//
//  ArmPlan.swift
//  MABEL Mobile App
//
//  The MoveIt-style plan/preview/execute core for the green-ball wrist control.
//
//  WHY THIS EXISTS: dragging the green palm ball used to run damped-least-squares
//  IK and STREAM the result to the robot on every solve (~60 Hz). That is unstable
//  — a fast or jittery drag whips the arm, and with nothing checking geometry the
//  arm can drive into itself. The fix is a safety gate: dragging now PLANS a target
//  the operator previews (an animated ghost of current→target, like MoveIt on
//  RViz), and NOTHING reaches the robot until they press Execute. Execute then
//  streams the SAME eased trajectory that was previewed, so what you saw is what
//  the robot does.
//
//  This file is the pure, side-effect-free core: trajectory interpolation, joint-
//  limit clamping, and a self-collision proximity heuristic. It touches no SceneKit
//  and no network, so it is unit-testable headlessly (ArmPlanTests) — the IK and the
//  ghost animation live in LiveMapView, the send path in RobotController.
//

import Foundation
import simd

enum ArmPlan {

    /// smoothstep ease (0→1). A joint-space linear blend under this profile gives
    /// the arm a soft accel/decel — the same shape MoveIt plays back — instead of a
    /// velocity step at the endpoints.
    static func ease(_ t: Double) -> Double {
        let x = max(0, min(1, t))
        return x * x * (3 - 2 * x)
    }

    /// One interpolated arm configuration at phase `t` in [0,1]. Only `chain` joints
    /// move; every value is clamped to its RobotModel limit so a plan can never ask
    /// the robot for an out-of-range angle. Joints absent from `target` hold `start`.
    /// Clamp the ENDPOINT, then interpolate — not the other way round.
    ///
    /// Interpolating toward an out-of-range target and clamping the *result*
    /// makes the joint race toward the unreachable value and pin at its limit
    /// partway through the playback. `maxDelta` (and so `duration`) measure the
    /// CLAMPED distance, so the two disagreed and an out-of-range plan executed
    /// over-budget: a −3.0 rad request on a joint whose limit is −1.396 was
    /// sized for 1.396 rad but driven at the 3.0 rad rate — measured 2.15 rad/s
    /// against a 1.0 rad/s cap, and it then sat pinned for the second half.
    /// IK solutions land out of range routinely, so this was reachable, not
    /// theoretical. Clamping the endpoint gives a clean ease onto the nearest
    /// reachable pose at exactly the intended speed.
    static func sample(from start: [String: Double], to target: [String: Double],
                       chain: [String], t: Double) -> [String: Double] {
        let e = ease(t)
        var out: [String: Double] = [:]
        for id in chain {
            let a = clamp(id, start[id] ?? 0)
            let b = clamp(id, target[id] ?? (start[id] ?? 0))
            out[id] = a + (b - a) * e
        }
        return out
    }

    /// Clamp a joint value to its model limits (identity if the joint is unknown).
    static func clamp(_ id: String, _ v: Double) -> Double {
        guard let s = RobotModel.spec(id) else { return v }
        return Swift.max(s.lower, Swift.min(s.upper, v))
    }

    /// The full previewed / executed trajectory: `steps`+1 configurations from start
    /// (t=0) to the clamped target (t=1). Step count scales with the biggest joint
    /// move so a large motion is sampled finely and a tiny nudge stays cheap —
    /// bounded so the streaming loop can never explode.
    static func waypoints(from start: [String: Double], to target: [String: Double],
                          chain: [String], stepsPerRad: Double = 40,
                          minSteps: Int = 12, maxSteps: Int = 80) -> [[String: Double]] {
        let span = chain.map { abs((target[$0] ?? start[$0] ?? 0) - (start[$0] ?? 0)) }.max() ?? 0
        let n = Swift.max(minSteps, Swift.min(maxSteps, Int((span * stepsPerRad).rounded(.up))))
        return (0...n).map { sample(from: start, to: target, chain: chain, t: Double($0) / Double(n)) }
    }

    /// How far the target moves each joint, radians. Drives the plan's ETA + the
    /// "big move" warning (a plan that swings a joint most of its range deserves a
    /// second look even without a collision).
    static func maxDelta(from start: [String: Double], to target: [String: Double],
                         chain: [String]) -> Double {
        chain.map { abs(clamp($0, target[$0] ?? start[$0] ?? 0) - (start[$0] ?? 0)) }.max() ?? 0
    }

    // MARK: - Execute duration (the velocity budget)

    /// Peak per-joint speed a plan playback may reach, rad/s. This is the SAME
    /// number as `RobotController.homeMaxJointSpeed` and as the web studio's
    /// `motions.js` `SAFE_BLEND_VEL` — one velocity budget for commanded motion
    /// across every MABEL client.
    static let safeBlendVel = 1.0
    /// Peak/average velocity ratio of `ease` ABOVE. Smoothstep (3x²−2x³) peaks at
    /// exactly 1.5× its average. The web's `motions.js` eases with a RAISED COSINE
    /// and so uses π/2 ≈ 1.571 — different curve, same budget. If `ease` is ever
    /// changed here, re-derive this or the cap silently weakens.
    static let easePeak = 1.5
    /// Jumps under this are benign — playing a small nudge over a ramp just adds
    /// latency to a move the operator experiences as instant. DERIVED from the
    /// budget rather than chosen, and shared with the timeline guard
    /// (`MotionMath.blendThresh`, pinned by a test): a jump is benign exactly
    /// when ONE command tick of it is still inside the budget.
    static let blendThresh = MotionMath.blendThresh
    static let minDuration = 0.4
    static let maxDuration = 8.0

    /// How long Execute should take, so no joint's PEAK speed exceeds
    /// `safeBlendVel`. A big swing simply plays for longer.
    ///
    /// This replaced a hard-coded 1.4 s, which was a duration with no reference to
    /// the motion: a plan swinging a joint 2.5 rad ran at ~2.7 rad/s peak — nearly
    /// 3× the budget every other commanded path on this robot respects — while a
    /// 5° nudge crawled for the same 1.4 s. Returns 0 for a delta below
    /// `blendThresh`, meaning "send it directly, no ramp needed".
    static func duration(from start: [String: Double], to target: [String: Double],
                         chain: [String]) -> Double {
        let d = maxDelta(from: start, to: target, chain: chain)
        guard d > blendThresh else { return 0 }
        return Swift.min(maxDuration, Swift.max(minDuration, easePeak * d / safeBlendVel))
    }

    // MARK: - Self-collision heuristic

    /// A coarse, cheap self-collision check on END-EFFECTOR/link world positions —
    /// NOT a full mesh collision solver. It exists to catch the obvious, dangerous
    /// cases the operator asked about ("sometimes it collides itself"): the planned
    /// palm ending up inside the torso column, or the two palms driven into each
    /// other. It is ADVISORY: it colours the plan and warns, it does not block
    /// Execute (the operator can always see the previewed motion and decide).
    ///
    /// Distances are centre-to-centre in metres; the thresholds are padded radii for
    /// MABEL's ~9 cm palm and ~12 cm torso column.
    struct CollisionReport: Equatable {
        var risk: Bool
        var reason: String
    }

    static let palmClearance: Float = 0.12      // palm↔palm centre distance floor
    static let torsoClearance: Float = 0.18     // palm↔torso-axis distance floor
    static let floorClearance: Float = 0.05     // palm height floor (above the base deck)

    /// - Parameters (all in the ODOM/world frame the rig renders in):
    ///   - palm: the planned palm centre of the dragged arm.
    ///   - otherPalm: the OTHER arm's current palm centre (nil if unknown).
    ///   - torsoAxis: a point on the torso's vertical column at palm height.
    ///   - floorY: the base deck height; a palm below it is driving into the robot.
    static func collision(palm: simd_float3, otherPalm: simd_float3?,
                          torsoAxis: simd_float3, floorY: Float) -> CollisionReport {
        if let o = otherPalm, simd_distance(palm, o) < palmClearance {
            return .init(risk: true, reason: "palms would collide")
        }
        // Horizontal distance to the torso column (ignore vertical — the column is tall).
        let dx = palm.x - torsoAxis.x, dz = palm.z - torsoAxis.z
        if sqrt(dx * dx + dz * dz) < torsoClearance {
            return .init(risk: true, reason: "palm would hit the torso")
        }
        if palm.y < floorY - floorClearance {
            return .init(risk: true, reason: "palm would hit the base")
        }
        return .init(risk: false, reason: "")
    }
}
