//
//  SegmentedPicker.swift
//  MABEL Mobile App
//
//  One compact pill-style segmented control used everywhere a choice is made
//  (Teleop · Feel · Reach · joint regions) so every toggle looks identical.
//

import SwiftUI

struct SegmentedPicker<T: Hashable>: View {
    let options: [T]
    let label: (T) -> String
    @Binding var selection: T
    var onPick: (T) -> Void = { _ in }

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(options.enumerated()), id: \.offset) { _, value in
                let active = value == selection
                Button {
                    withAnimation(Theme.Motion.snappy) { selection = value }
                    onPick(value)
                } label: {
                    Text(label(value))
                        .font(.system(size: 11, weight: .semibold))
                        .lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity).padding(.vertical, 6)
                        .foregroundStyle(active ? .white : Theme.Palette.textSecondary)
                        .background(active ? AnyShapeStyle(Theme.accentGradient) : AnyShapeStyle(Color.clear))
                }
                .buttonStyle(.plain)
            }
        }
        .clipShape(Capsule())
        .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        .background(Theme.Palette.surface, in: Capsule())
    }
}

/// **Liquid-Glass** segmented control that speaks the same language as the bottom
/// nav bar: one glass capsule with a soft accent-tinted selection that glides
/// between segments (matchedGeometry) and tints the selected label. Used for the
/// Manipulate tab's mode pickers so they read like the nav / control bars.
///
/// `prominent` is the hero switch (Stiff/Soft): bigger type, content-width segments
/// (so the bar centres itself); otherwise compact segments fill the available width.
struct GlassSegmented<T: Identifiable & Hashable>: View {
    let options: [T]
    @Binding var selection: T
    var title: (T) -> String
    var icon: (T) -> String? = { _ in nil }
    var prominent: Bool = false
    var onPick: (T) -> Void = { _ in }

    @Namespace private var ns

    var body: some View {
        HStack(spacing: 2) {
            ForEach(options) { opt in segment(opt) }
        }
        .padding(3)
        .glassEffect(.regular, in: Capsule(style: .continuous))
        .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        .shadow(color: .black.opacity(prominent ? 0.22 : 0.14), radius: prominent ? 8 : 6, y: 3)
    }

    @ViewBuilder private func segment(_ opt: T) -> some View {
        let active = opt == selection
        Button {
            withAnimation(.spring(response: 0.36, dampingFraction: 0.82)) { selection = opt }
            onPick(opt); Haptics.select()
        } label: {
            HStack(spacing: 5) {
                if let ic = icon(opt) {
                    Image(systemName: ic).font(.system(size: prominent ? 14 : 13, weight: .bold))
                        // SF Symbols differ in intrinsic glyph height (e.g. "atom"
                        // vs "video.fill") — pin the icon row so every segmented
                        // control renders the SAME height side by side.
                        .frame(height: prominent ? 18 : 16)
                }
                Text(title(opt)).font(.system(size: prominent ? 15 : 14, weight: .bold))
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
            .frame(height: prominent ? 20 : 18)
            // Adaptive inactive colour (crisp on the light-mode glass panel; light on
            // the dark-mode / over-camera glass) — never the old hard-coded white that
            // vanished in Light Mode.
            .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textPrimary.opacity(0.78))
            .padding(.vertical, prominent ? 9 : 7.5)
            .padding(.horizontal, prominent ? 22 : 7)
            .frame(maxWidth: prominent ? nil : .infinity)
            .background {
                if active {
                    Capsule(style: .continuous)
                        .fill(Theme.Palette.accent.opacity(0.18))
                        .matchedGeometryEffect(id: "glass.seg", in: ns)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
