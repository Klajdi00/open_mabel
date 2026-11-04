//
//  SectionLabel.swift
//  MabelTeleop
//
//  Shared micro-label + status-row primitives — the SAME definition as the iOS
//  app, replacing the copy-pasted `cardTitle(_:)` / `statusRow(...)` helpers so
//  the tracked all-caps label and label/value rows render identically across
//  both apps.
//

import SwiftUI

extension Text {
    /// The tracked micro all-caps label that heads a group of cards / a section.
    func sectionLabel() -> some View {
        self.font(.mabelCaption)
            .foregroundStyle(Theme.Palette.textTertiary)
            .tracking(Theme.Tracking.microCaps)
    }
}

/// Convenience view for `Text(...).sectionLabel()`.
struct SectionLabel: View {
    private let text: String
    init(_ text: String) { self.text = text }
    var body: some View { Text(text).sectionLabel() }
}

/// A label / value line with a tinted leading icon — the shared `statusRow`.
struct StatusRow: View {
    private let label: String
    private let value: String
    private let tint: Color
    private let icon: String

    init(_ label: String, _ value: String, tint: Color, icon: String) {
        self.label = label
        self.value = value
        self.tint = tint
        self.icon = icon
    }

    var body: some View {
        HStack(spacing: 10) {
            // TransportGlyph, not Image: a route icon here must draw the Bluetooth
            // rune. Every other symbol name passes straight through unchanged.
            TransportGlyph(icon: icon, size: 13)
                .foregroundStyle(tint)
                .frame(width: 22)
            Text(label)
                .font(.mabelBody)
                .foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(value)
                .font(.mabelLabel.weight(.semibold))
                .foregroundStyle(tint)
                .multilineTextAlignment(.trailing)
        }
    }
}
