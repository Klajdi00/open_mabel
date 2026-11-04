//
//  StageActionButton.swift
//  MabelTeleop
//
//  Right-edge stage action — a labelled glass capsule used identically by
//  the Body and Maps 3D stages (Center / Volume / Recenter), styled to sit
//  with the right-hand toggles.
//

import SwiftUI

struct StageActionButton: View {
    let icon: String
    let label: String
    /// Toggle on-state (e.g. Follow) — tints the icon + label accent and fills
    /// the capsule with a soft accent wash.
    var active: Bool = false
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.accent)
                Text(label)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textPrimary)
            }
            .frame(width: 104, alignment: .leading)
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(active ? Theme.Palette.accent.opacity(0.18) : Color.clear, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(PressableStyle())
        .glassBackgroundEffect(in: Capsule())
    }
}

/// A floating instruction pill — the visionOS analogue of the iOS `HintPill`
/// (glass capsule, concise copy), so a hint like "Live SLAM not connected" reads
/// the same as everywhere else.
struct HintPill: View {
    let systemImage: String
    let title: String
    var subtitle: String? = nil
    var iconTint: Color = Theme.Palette.accent
    var pulse: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage).font(.system(size: 13, weight: .bold))
                .foregroundStyle(iconTint)
                .symbolEffect(.pulse, isActive: pulse)
            if let subtitle {
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 13, weight: .bold)).foregroundStyle(Theme.Palette.textPrimary)
                    Text(subtitle).font(.system(size: 11, weight: .medium)).foregroundStyle(Theme.Palette.textSecondary)
                }
            } else {
                Text(title).font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.Palette.textPrimary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 9)
        .glassBackgroundEffect(in: Capsule())
    }
}
