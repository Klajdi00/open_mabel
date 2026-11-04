//
//  StatChip.swift
//  MABEL Mobile App
//
//  Dense, glanceable telemetry chip for the cockpit status strip:
//  icon · value · tiny unit. Designed to pack many readings in a row.
//

import SwiftUI

struct StatChip: View {
    let icon: String
    let value: String
    var unit: String = ""
    var tint: Color = Theme.Palette.textPrimary

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(tint)
            HStack(alignment: .firstTextBaseline, spacing: 1) {
                Text(value)
                    .font(.mabelMonoSmall)
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .contentTransition(.numericText())
                if !unit.isEmpty {
                    Text(unit)
                        .font(.system(size: 8.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(Theme.Palette.surface.opacity(0.75), in: Capsule())
        .overlay { Capsule().strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }
}
