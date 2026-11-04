//
//  GlassCard.swift
//  MABEL Mobile App
//
//  The workhorse container: a soft, frosted surface with a hairline
//  border. Used everywhere to group content into Apple-style cards.
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
            .mabelShadow(Theme.Shadow.card)
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
            .overlay {
                Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            }
    }
}
