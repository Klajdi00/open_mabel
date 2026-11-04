//
//  AppSection.swift
//  MABEL Mobile App
//
//  Primary destinations. v2 unifies the old Pilot / Manipulate / Navigate tabs
//  into ONE Robot space — a single 3D viewer that is both the visualizer and
//  the controller (drive, manipulate, nav goals, cameras). Autonomy and
//  Settings carry over unchanged from v1.
//

import Foundation

enum AppSection: String, CaseIterable, Identifiable {
    case robot     = "robot"
    case policies  = "policies"
    case settings  = "settings"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .robot:     return "Robot"        // unified 3D viewer: drive + manipulate + navigate
        case .policies:  return "Autonomy"     // high-level policies, talk-to-robot, monitor
        case .settings:  return "Settings"     // app + robot settings
        }
    }

    var icon: String {
        switch self {
        // Fallback only. The tab bar draws MabelBodyGlyph (RobotGlyph.swift) for
        // this section — `move.3d` is a 3-axis gizmo, which describes a viewport
        // rather than the robot. Anywhere that still wants a String gets the
        // closest system symbol.
        case .robot:     return "figure.stand"
        case .policies:  return "brain.head.profile"
        case .settings:  return "gearshape.fill"
        }
    }
}
