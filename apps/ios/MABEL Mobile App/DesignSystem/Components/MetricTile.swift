//
//  MetricTile.swift
//  MABEL Mobile App
//
//  Compact KPI tile: icon, value, label. Used on the dashboard and
//  the status screen.
//

import SwiftUI

struct MetricTile: View {
    let icon: String
    let value: String
    let label: String
    var tint: Color = Theme.Palette.accent

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 34, height: 34)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            Spacer(minLength: 0)

            Text(value)
                .font(.mabel(22, weight: .semibold))
                .foregroundStyle(Theme.Palette.textPrimary)
                .contentTransition(.numericText())

            Text(label)
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textTertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(height: 116)
        .padding(Theme.Space.md)
        .background {
            RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous)
                .fill(Theme.Palette.surfaceHi)
        }
        .overlay {
            RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        }
    }
}

/// Section title + optional trailing accessory, used to head a group of cards.
struct SectionHeader<Trailing: View>: View {
    let title: String
    var subtitle: String? = nil
    @ViewBuilder var trailing: () -> Trailing

    init(_ title: String, subtitle: String? = nil, @ViewBuilder trailing: @escaping () -> Trailing = { EmptyView() }) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.mabelTitle)
                    .foregroundStyle(Theme.Palette.textPrimary)
                if let subtitle {
                    Text(subtitle)
                        .font(.mabelCaption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            Spacer()
            trailing()
        }
    }
}
