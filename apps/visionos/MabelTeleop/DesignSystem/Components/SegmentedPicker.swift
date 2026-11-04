//
//  SegmentedPicker.swift
//  MabelTeleop
//
//  One compact pill-style segmented control used everywhere a choice is made
//  (Teleop · Feel · Reach · joint regions) — identical to the iOS app.
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
                        .font(.system(size: 13, weight: .semibold))
                        .lineLimit(1).minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                        .foregroundStyle(active ? .white : Theme.Palette.textSecondary)
                        .background(active ? AnyShapeStyle(Theme.accentGradient) : AnyShapeStyle(Color.clear))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .clipShape(Capsule())
        .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
        .background(Theme.Palette.surface, in: Capsule())
    }
}
