//
//  GlassCard.swift
//  MabelTeleop
//
//  The workhorse container — same as the iOS app's GlassCard, but on
//  visionOS the material reads as true liquid glass.
//

import SwiftUI

struct GlassCard<Content: View>: View {
    var padding: CGFloat = Theme.Space.md
    var cornerRadius: CGFloat = Theme.Radius.lg
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(padding)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            }
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

/// A subtle frosted capsule used for inline badges and toolbars.
struct FrostedCapsule<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.sm)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
    }
}
