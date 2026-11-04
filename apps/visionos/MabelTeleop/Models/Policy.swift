//
//  Policy.swift
//  MABEL Mobile App
//
//  A learned or scripted behaviour the operator can dispatch from the
//  Policies library (e.g. a CVAE visuomotor policy or a nav routine).
//

import SwiftUI

struct Policy: Identifiable, Equatable {
    enum Kind: String { case imitation = "Imitation", navigation = "Navigation", scripted = "Scripted" }

    let id = UUID()
    var name: String
    var kind: Kind
    var summary: String
    var icon: String
    var successRate: Double      // 0...1, from offline eval
    var horizonSec: Int
    var episodes: Int            // training demos (0 for scripted)
    var freqHz: Int              // control rate
    var lastRun: String
    var clip: String             // looping task-overview clip (Resources/Clips)
    var tint: Color

    /// Longer detail copy for the review sheet.
    var detail: String {
        switch kind {
        case .imitation:
            return "Trained from \(episodes) teleoperated demonstrations. The policy maps RGB-D + proprioception to a \(horizonSec)s action horizon at \(freqHz) Hz, evaluated offline at \(Int(successRate * 100))% task success."
        case .navigation:
            return "Global plan + swerve trajectory tracking. Runs the full route at \(freqHz) Hz with \(Int(successRate * 100))% arrival reliability over recorded trials."
        case .scripted:
            return "Deterministic skill running at \(freqHz) Hz. Used as a reliable building block between learned behaviours."
        }
    }

    var kindTint: Color {
        switch kind {
        case .imitation:  return Color(hex: 0xB18CFF)
        case .navigation: return Theme.Palette.accent
        case .scripted:   return Theme.Palette.live
        }
    }

    static let library: [Policy] = [
        .init(name: "Home → Set Dressing", kind: .navigation,
              summary: "Autonomous swerve navigation from dock to the staging mark.",
              icon: "map.circle.fill", successRate: 0.97, horizonSec: 42,
              episodes: 0, freqHz: 20, lastRun: "12m ago", clip: "home_nav", tint: Theme.Palette.accent),
        .init(name: "Cinematic Crane Reach", kind: .imitation,
              summary: "CVAE visuomotor policy — lift + dual-arm reach to a marked prop.",
              icon: "camera.aperture", successRate: 0.88, horizonSec: 15,
              episodes: 240, freqHz: 30, lastRun: "1h ago", clip: "wbc_reach", tint: Color(hex: 0xB18CFF)),
        .init(name: "Bimanual Lift", kind: .imitation,
              summary: "Coordinated two-arm grasp and torso lift of a payload.",
              icon: "arrow.up.and.person.rectangle.portrait", successRate: 0.82, horizonSec: 18,
              episodes: 180, freqHz: 30, lastRun: "Yesterday", clip: "lift", tint: Color(hex: 0x6FB8FF)),
        .init(name: "Track Subject", kind: .scripted,
              summary: "Neck servoing keeps the head camera framed on a moving subject.",
              icon: "person.and.background.dotted", successRate: 0.94, horizonSec: 0,
              episodes: 0, freqHz: 60, lastRun: "3h ago", clip: "swerve", tint: Theme.Palette.live),
        .init(name: "Return to Dock", kind: .navigation,
              summary: "Retrace the last trajectory back to the charging dock.",
              icon: "bolt.fill.batteryblock.fill", successRate: 0.99, horizonSec: 55,
              episodes: 0, freqHz: 20, lastRun: "5h ago", clip: "home_nav", tint: Theme.Palette.warn),
        .init(name: "Pour & Place", kind: .imitation,
              summary: "Single-arm reach, grasp a vessel and pour into a target bowl.",
              icon: "drop.fill", successRate: 0.79, horizonSec: 22,
              episodes: 320, freqHz: 30, lastRun: "2d ago", clip: "wbc_reach", tint: Color(hex: 0xB18CFF)),
        .init(name: "Stow Arms", kind: .scripted,
              summary: "Fold both arms into the transit pose and lower the lift.",
              icon: "arrow.down.right.and.arrow.up.left", successRate: 1.0, horizonSec: 6,
              episodes: 0, freqHz: 100, lastRun: "8m ago", clip: "lift", tint: Theme.Palette.live),
    ]
}
