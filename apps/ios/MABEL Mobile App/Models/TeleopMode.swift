//
//  TeleopMode.swift
//  MABEL Mobile App
//
//  The control surface modes the operator can switch between while
//  teleoperating MABEL. Mirrors the Vision Pro app's model but adds a
//  dedicated whole-body mode for the phone form factor.
//

import Foundation

enum TeleopMode: String, CaseIterable, Identifiable, Codable {
    case drive      = "Drive"
    case arm        = "Arm"
    case wholeBody  = "Whole Body"

    var id: String { rawValue }

    var subtitle: String {
        switch self {
        case .drive:     return "Swerve base navigation"
        case .arm:       return "Cartesian end-effector"
        case .wholeBody: return "Coordinated base + arms"
        }
    }

    var icon: String {
        switch self {
        case .drive:     return "steeringwheel"
        case .arm:       return "hand.point.up.braille.fill"
        case .wholeBody: return "figure.walk.motion"
        }
    }
}
