//
//  StatusPill.swift
//  MabelTeleop
//
//  Small live-status capsule (connection state etc.), same as iOS.
//

import SwiftUI

struct StatusPill: View {
    let title: String
    let color: Color
    var pulses: Bool = false

    @State private var pulse = false

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
                .opacity(pulses && pulse ? 0.35 : 1)
                .animation(pulses ? .easeInOut(duration: 0.9).repeatForever(autoreverses: true) : .default,
                           value: pulse)
            Text(title)
                .font(.mabelCaption)
                .foregroundStyle(Theme.Palette.textPrimary)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(color.opacity(0.12), in: Capsule())
        .overlay { Capsule().strokeBorder(color.opacity(0.3), lineWidth: 1) }
        .onAppear { pulse = pulses }
    }
}
