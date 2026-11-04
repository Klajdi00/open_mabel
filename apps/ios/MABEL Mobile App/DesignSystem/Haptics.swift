//
//  Haptics.swift
//  MABEL Mobile App
//
//  Thin wrapper over UIFeedbackGenerator so call sites stay tidy.
//  Thoughtful haptics are a big part of an Apple-grade feel.
//

import UIKit

enum Haptics {
    /// Honors the Settings › Preferences "Haptics" toggle. The switch writes the
    /// `haptics.enabled` default (via `@AppStorage`); every call site gates on it
    /// here so turning it off actually silences feedback app-wide. Defaults on.
    static var enabled: Bool {
        UserDefaults.standard.object(forKey: "haptics.enabled") as? Bool ?? true
    }

    static func tap() {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    static func select() {
        guard enabled else { return }
        UISelectionFeedbackGenerator().selectionChanged()
    }

    static func soft() {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .soft).impactOccurred(intensity: 0.6)
    }

    static func rigid() {
        guard enabled else { return }
        UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
    }

    static func success() {
        guard enabled else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    static func warning() {
        guard enabled else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.warning)
    }

    static func error() {
        guard enabled else { return }
        UINotificationFeedbackGenerator().notificationOccurred(.error)
    }
}
