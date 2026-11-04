//
//  PortedControls.swift
//  MabelTeleop
//
//  Controls ported from MABEL iOS v2 for the unified Robot space — the
//  segmented glass toggle, viewport pills, the Start-Teleop gate, and the
//  E-stop. iOS 26's `glassEffect` is replaced by visionOS's
//  `glassBackgroundEffect`, and Liquid-Glass button styles by the app's
//  PressableStyle idiom; everything else is 1:1 so the two apps read the same.
//

import SwiftUI

/// Segmented control that speaks the same language as the bottom nav bar: one
/// glass capsule with a soft accent-tinted selection that glides between
/// segments and tints the selected label.
///
/// `prominent` is the hero switch (Stiff/Soft): bigger type, content-width
/// segments; otherwise compact segments fill the available width.
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
        .glassBackgroundEffect(in: Capsule(style: .continuous))
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
                }
                Text(title(opt)).font(.system(size: prominent ? 15 : 14, weight: .bold))
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
            .foregroundStyle(active ? Theme.Palette.accent : Theme.Palette.textPrimary.opacity(0.78))
            .padding(.horizontal, prominent ? 22 : 7)
            .frame(maxWidth: prominent ? nil : .infinity)
            .frame(height: 30)   // + the 3 pt chrome = the 36 pt control standard
            .background {
                if active {
                    Capsule(style: .continuous)
                        .fill(Theme.Palette.accent.opacity(0.18))
                        .matchedGeometryEffect(id: "glass.seg", in: ns)
                }
            }
            .contentShape(Rectangle())
            .contentShape(.hoverEffect, Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }
}

/// Canonical tinted glass pill button (ported from iOS v2's Liquid-Glass
/// GlassButton, rendered with visionOS glass).
struct GlassButton: View {
    let title: String
    var systemImage: String? = nil
    var tint: Color = Theme.Palette.accent
    var prominent: Bool = false
    var fullWidth: Bool = false
    var size: ControlSize = .regular
    var role: ButtonRole? = nil
    var action: () -> Void

    var body: some View {
        Button(role: role) {
            Haptics.tap()
            action()
        } label: {
            HStack(spacing: 6) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title)
            }
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(prominent ? .white : tint)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .padding(.horizontal, 14)
            .padding(.vertical, size == .large ? 12 : 8)
            .background {
                if prominent {
                    Capsule().fill(tint.gradient)
                } else {
                    Capsule().fill(tint.opacity(0.12))
                }
            }
            .overlay { Capsule().strokeBorder(tint.opacity(prominent ? 0.35 : 0.3), lineWidth: 1) }
            .contentShape(.hoverEffect, Capsule(style: .continuous))
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }
}

/// A frosted glass container for an inline text field, so an editable line
/// clearly reads as a tappable field rather than bare text.
struct GlassField<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .glassBackgroundEffect(in: Capsule())
    }
}

/// A compact action pill (icon + label) for 3D-viewport controls — re-center,
/// fit-to-view, follow, POV. `active` (for toggles like Follow) fills it with
/// the brand accent so its on-state is unmistakable.
struct ViewportButton: View {
    let title: String
    let systemImage: String
    var active: Bool = false
    var compact: Bool = false
    var foreground: Color = .white
    var action: () -> Void

    var body: some View {
        Button { Haptics.tap(); action() } label: {
            HStack(spacing: compact ? 5 : 6) {
                Image(systemName: systemImage).font(.system(size: compact ? 13 : 14, weight: .bold))
                if !title.isEmpty {
                    Text(title).font(.system(size: compact ? 13 : 14, weight: .bold)).lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(active ? .white : foreground)
            .padding(.horizontal, compact ? 12 : 14).padding(.vertical, compact ? 8.5 : 9)
            .frame(minWidth: title.isEmpty ? 46 : 0, minHeight: 34)
            .contentShape(Capsule(style: .continuous))
            .contentShape(.hoverEffect, Capsule(style: .continuous))
            .modifier(ViewportPillChrome(active: active))
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
    }
}

/// Active = filled accent capsule (toggle on); inactive = clear glass with a
/// hairline. Kept separate so the active/inactive backgrounds compose cleanly.
private struct ViewportPillChrome: ViewModifier {
    let active: Bool
    @ViewBuilder func body(content: Content) -> some View {
        if active {
            content
                .background(Theme.accentGradient, in: Capsule())
                .overlay { Capsule().strokeBorder(.white.opacity(0.25), lineWidth: 1) }
                .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
        } else {
            content
                .glassBackgroundEffect(in: Capsule())
                .overlay { Capsule().strokeBorder(.white.opacity(0.18), lineWidth: 1) }
                .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
        }
    }
}

/// The teleop engage gate — top-center of the Robot space. While disengaged
/// it's deliberately INVITING: the play glyph pulses and a soft accent glow
/// breathes around the capsule. Engaged, it turns into a calm red "Stop Teleop".
struct StartTeleopButton: View {
    let active: Bool
    var compact: Bool = false
    let toggle: () -> Void
    @State private var glow = false

    private var tint: Color { active ? Theme.Palette.danger : Theme.Palette.accent }

    var body: some View {
        Button(action: toggle) {
            HStack(spacing: 7) {
                Image(systemName: active ? "stop.circle.fill" : "play.circle.fill")
                    .font(.system(size: 15, weight: .bold))
                    .symbolEffect(.pulse, isActive: !active)
                if !compact {
                    // Named for what it DOES (operator, 2026-07-18): this is the
                    // ONE whole-body entry — head + arms + fingers retargeting.
                    Text(active ? "Stop Body Tracking" : "Start Whole Body Tracking")
                        .font(.system(size: 13.5, weight: .heavy)).lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(.white)
            .padding(.horizontal, compact ? 12 : 16)
            .frame(height: 36)   // top-bar control standard
            .background(tint.gradient, in: Capsule())
            .overlay { Capsule().strokeBorder(.white.opacity(active ? 0.25 : 0.4), lineWidth: 1) }
            .shadow(color: tint.opacity(active ? 0.4 : (glow ? 0.85 : 0.45)),
                    radius: active ? 8 : (glow ? 16 : 9), y: 3)
            .scaleEffect(!active && glow ? 1.035 : 1.0)
        }
        .buttonStyle(PressableStyle())
        .contentShape(.hoverEffect, Capsule(style: .continuous))
        .hoverEffect()
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) { glow = true }
        }
    }
}

/// The unmistakable emergency stop (full-width row variant used in panels).
struct EStopButton: View {
    @Binding var engaged: Bool
    var onEngage: () -> Void
    var onRelease: () -> Void

    @State private var ring = false

    var body: some View {
        Button {
            if engaged {
                Haptics.success()
                engaged = false
                onRelease()
            } else {
                Haptics.error()
                engaged = true
                onEngage()
            }
        } label: {
            HStack(spacing: 10) {
                Image(systemName: engaged ? "play.fill" : "exclamationmark.octagon.fill")
                    .font(.system(size: 18, weight: .bold))
                Text(engaged ? "RELEASE E-STOP" : "EMERGENCY STOP")
                    .font(.system(size: 15, weight: .heavy))
                    .tracking(0.5)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 17)
            .background((engaged ? Theme.Palette.live : Theme.Palette.danger).gradient, in: Capsule())
            .overlay {
                Capsule()
                    .strokeBorder(.white.opacity(0.25), lineWidth: ring ? 6 : 1)
                    .opacity(engaged ? 0 : 1)
            }
            .shadow(color: (engaged ? Theme.Palette.live : Theme.Palette.danger).opacity(0.5),
                    radius: 16, y: 6)
        }
        .buttonStyle(PressableStyle())
        .onAppear {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                ring = true
            }
        }
    }
}
