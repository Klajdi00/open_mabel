//
//  DeckFollower.swift
//  MabelTeleop
//
//  Where full immersion's CONTROL DECK (drive pad · Start/Stop · body switches ·
//  console) and the STATUS bar sit: around the head-locked camera window,
//  wherever the operator looks. Two modes (`Mode`, the console's Dock toggle):
//
//   • `.follow` (DEFAULT — operator, 2026-10-17: "make the bottom dock and the top
//     status bar follow the human gaze too, like the two wrist cams and the 3D
//     model"). Heading AND pitch follow the head, so the deck and status bar stay
//     framed with the window and the wrist cams on a turn or a nod. NOT rigidly
//     locked like those display-only panels: a `deadband` swallows the head's
//     micro-motion while the eyes drop to a control and the hand pinches (a
//     rigid lock jitters the target under the finger). Once an axis leaves the
//     deadband it glides in with `tauFollow` — tight enough to read as attached —
//     all the way to within `alignAt`, so the deck re-centres EXACTLY under the
//     window (following only the excess over the band parked it 3° off-axis after
//     every turn). Roll is dropped: the controls stay level.
//   • `.steady` (the earlier behaviour). YAW follows LAZILY with hysteresis:
//     nothing happens until the operator has turned more than `recenterAt` away,
//     then the deck glides back in front (`tauYaw`) and stops within `settleAt`.
//     Head PITCH is ignored, so looking down at the deck never moves it.
//
//  In both, POSITION follows the head's position and height smoothly (`tauPos`).
//  Pure value type → `DeckFollowerTests`.
//

import Foundation
import simd

struct DeckFollower {
    enum Mode: String, CaseIterable { case follow, steady }

    var mode: Mode = .follow
    /// FALSE for the CONTROL DECK (operator, 2026-10-23: "when I move the head down
    /// the bar moves away … do not have that again"). A deck that follows head PITCH
    /// can never be looked at: every nod toward it pushes it further down, out of the
    /// headset's view. Off, the deck's pitch stays level with the room, so looking
    /// down brings it INTO view. The status bar keeps following (it rides the window).
    var followsPitch = true
    // .follow
    var deadband: Float = 3 * .pi / 180
    var alignAt: Float = 0.15 * .pi / 180
    var tauFollow: Float = 0.12
    var maxPitch: Float = 60 * .pi / 180
    // .steady
    var recenterAt: Float = 25 * .pi / 180
    var settleAt: Float = 2 * .pi / 180
    var tauYaw: Float = 0.25
    // both
    var tauPos: Float = 0.35

    private(set) var yaw: Float = 0
    /// + = looking UP (rotation about the rig's +X).
    private(set) var pitch: Float = 0
    private(set) var position = SIMD3<Float>(0, 1.4, 0)
    private(set) var recentering = false
    /// `.follow`, per axis: catching up after leaving the deadband (until aligned).
    private(set) var followingYaw = false
    private(set) var followingPitch = false
    private var seeded = false

    /// Advance one frame. `headPos` in the space's frame, `headYaw` = heading of
    /// the head's forward (−Z) about +Y, `headPitch` its elevation (+ = up).
    /// Returns the deck root transform.
    mutating func step(headPos: SIMD3<Float>, headYaw: Float, headPitch: Float = 0,
                       dt: Float) -> simd_float4x4 {
        let wantPitch = (mode == .follow && followsPitch) ? min(max(headPitch, -maxPitch), maxPitch) : 0
        if !seeded {
            yaw = headYaw; pitch = wantPitch; position = headPos; seeded = true
        } else {
            switch mode {
            case .follow:
                let k = 1 - exp(-dt / tauFollow)
                let dy = Self.wrap(headYaw - yaw), dp = wantPitch - pitch
                followingYaw = Self.engaged(followingYaw, error: dy, band: deadband, settle: alignAt)
                followingPitch = Self.engaged(followingPitch, error: dp, band: deadband, settle: alignAt)
                if followingYaw { yaw = Self.wrap(yaw + dy * k) }
                if followingPitch { pitch += dp * k }
                recentering = false
            case .steady:
                pitch += (0 - pitch) * (1 - exp(-dt / tauYaw))     // a switch from .follow levels out
                let d = Self.wrap(headYaw - yaw)
                if !recentering && abs(d) > recenterAt { recentering = true }
                if recentering {
                    yaw = Self.wrap(yaw + d * (1 - exp(-dt / tauYaw)))
                    if abs(Self.wrap(headYaw - yaw)) < settleAt { recentering = false }
                }
            }
            position += (headPos - position) * (1 - exp(-dt / tauPos))
        }
        let q = simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)) * simd_quatf(angle: pitch, axis: SIMD3(1, 0, 0))
        var m = simd_float4x4(q)
        m.columns.3 = SIMD4(position, 1)
        return m
    }

    /// Hysteresis: start following when the error leaves ±`band`, stop once inside ±`settle`.
    static func engaged(_ on: Bool, error: Float, band: Float, settle: Float) -> Bool {
        on ? abs(error) > settle : abs(error) > band
    }

    /// Heading of a head transform: the yaw of its forward (−Z) axis about +Y,
    /// pitch and roll discarded.
    static func yaw(of head: simd_float4x4) -> Float {
        let f = -SIMD3(head.columns.2.x, head.columns.2.y, head.columns.2.z)
        return atan2(-f.x, -f.z)
    }

    /// Elevation of a head transform's forward (−Z) axis (+ = looking up), roll discarded.
    static func pitch(of head: simd_float4x4) -> Float {
        let f = -SIMD3(head.columns.2.x, head.columns.2.y, head.columns.2.z)
        return asin(min(max(f.y / max(simd_length(f), 1e-6), -1), 1))
    }

    static func wrap(_ a: Float) -> Float {
        var x = a.truncatingRemainder(dividingBy: 2 * .pi)
        if x > .pi { x -= 2 * .pi }
        if x < -.pi { x += 2 * .pi }
        return x
    }
}
