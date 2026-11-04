//
//  Buttons.swift
//  MabelTeleop
//
//  Shared button styles, ported from the iOS app so both apps read
//  identically: filled accent capsule, quiet secondary, spring press.
//

import SwiftUI

/// Filled, accent-gradient capsule button.
struct PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    var action: () -> Void

    var body: some View {
        Button {
            Haptics.tap()
            action()
        } label: {
            HStack(spacing: 8) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title)
            }
            .font(.mabelHeadline)
            .foregroundStyle(.black.opacity(0.85))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Theme.accentGradient, in: Capsule())
        }
        .buttonStyle(PressableStyle())
    }
}

/// Outlined secondary button that reads as a quieter peer to PrimaryButton.
struct SecondaryButton: View {
    let title: String
    var systemImage: String? = nil
    var action: () -> Void

    var body: some View {
        Button {
            Haptics.soft()
            action()
        } label: {
            HStack(spacing: 8) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title)
            }
            .font(.mabelHeadline)
            .foregroundStyle(Theme.Palette.textPrimary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
    }
}

/// A spring-scale press effect applied to interactive controls.
struct PressableStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .opacity(configuration.isPressed ? 0.9 : 1)
            .animation(Theme.Motion.snappy, value: configuration.isPressed)
            .contentShape(Capsule())
            .hoverEffect()
    }
}
