//
//  Buttons.swift
//  MABEL Mobile App
//
//  Shared button styles: a filled accent capsule and the always-present
//  emergency-stop control.
//

import SwiftUI

/// The app's canonical **Liquid Glass** button (iOS 26). Tinted by role:
/// `prominent` fills the glass with the tint for primary actions; otherwise a
/// clear glass pill with a tinted label for secondary / inline actions. Used
/// EVERYWHERE so every tappable control reads the same — obviously a button.
struct GlassButton: View {
    let title: String
    var systemImage: String? = nil
    var tint: Color = Theme.Palette.accent
    var prominent: Bool = false
    var fullWidth: Bool = false
    var size: ControlSize = .regular        // .large only for big primary actions
    var role: ButtonRole? = nil
    var action: () -> Void

    private var base: some View {
        Button(role: role) {
            Haptics.tap()
            action()
        } label: {
            HStack(spacing: 6) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title)
            }
            .font(.system(size: 13, weight: .semibold))
            .frame(maxWidth: fullWidth ? .infinity : nil)
        }
        .controlSize(size)
        .tint(tint)
        .buttonBorderShape(.capsule)
    }

    @ViewBuilder var body: some View {
        if prominent { base.buttonStyle(.glassProminent) }
        else { base.buttonStyle(.glass) }
    }
}

/// Icon-only sibling of `GlassButton` — a small Liquid Glass pill with a clear
/// border, for toolbar-style actions (mute, expand, add a routine) that must
/// still obviously read as buttons rather than bare glyphs.
struct GlassIconButton: View {
    let systemImage: String
    var tint: Color = Theme.Palette.accent
    var prominent: Bool = false
    var action: () -> Void

    private var base: some View {
        Button {
            Haptics.tap()
            action()
        } label: {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .semibold))
                .frame(width: 30, height: 30)
        }
        .controlSize(.small)
        .tint(tint)
        .buttonBorderShape(.circle)
    }

    @ViewBuilder var body: some View {
        if prominent { base.buttonStyle(.glassProminent) }
        else { base.buttonStyle(.glass) }
    }
}

/// A frosted **Liquid Glass** container for an inline text field, so an editable
/// line clearly reads as a tappable field rather than bare text.
struct GlassField<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        content()
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .glassEffect(.regular, in: Capsule())
    }
}

/// Filled primary action — Liquid Glass, tinted with the brand accent.
struct PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    var tint: Color = Theme.Palette.accent
    var action: () -> Void

    var body: some View {
        GlassButton(title: title, systemImage: systemImage, tint: tint,
                    prominent: true, fullWidth: true, size: .large, action: action)
    }
}

/// Quieter secondary peer to PrimaryButton — clear Liquid Glass.
struct SecondaryButton: View {
    let title: String
    var systemImage: String? = nil
    var tint: Color = Theme.Palette.textPrimary
    var action: () -> Void

    var body: some View {
        GlassButton(title: title, systemImage: systemImage, tint: tint,
                    prominent: false, fullWidth: true, size: .large, action: action)
    }
}

/// A compact **Liquid-Glass** action pill (icon + label) for 3D-viewport controls
/// — re-center, fit-to-view, follow. Shared by the Manipulate (Body) and Navigate
/// (Map) tabs so every viewport control reads identically. `active` (for toggles
/// like Follow) fills it with the brand accent so its on-state is unmistakable.
struct ViewportButton: View {
    let title: String
    let systemImage: String
    var active: Bool = false
    /// Compact = tighter padding/type so several pills fit on one line (e.g. the
    /// map's Re-center · Fit · Follow · Go row on a phone).
    var compact: Bool = false
    /// Inactive label colour. Defaults to white for the always-dark map viewports;
    /// pass an adaptive colour (e.g. `Theme.Palette.textPrimary`) when the pill sits
    /// on a glass panel that turns light in Light Mode (the Manipulate column).
    var foreground: Color = .white
    var action: () -> Void

    var body: some View {
        Button { Haptics.tap(); action() } label: {
            HStack(spacing: compact ? 5 : 6) {
                Image(systemName: systemImage).font(.system(size: compact ? 13 : 14, weight: .bold))
                // Icon-only pills must not lay out an empty Text (its spacing
                // pushed the icon off-centre).
                if !title.isEmpty {
                    Text(title).font(.system(size: compact ? 13 : 14, weight: .bold)).lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(active ? .white : foreground)
            .padding(.horizontal, compact ? 12 : 14).padding(.vertical, compact ? 8.5 : 9)
            // Keep a comfortable ≥44 pt hit target even when icon-only.
            .frame(minWidth: title.isEmpty ? 46 : 0, minHeight: 34)
            .contentShape(Capsule(style: .continuous))
            .modifier(ViewportPillChrome(active: active))
        }
        .buttonStyle(PressableStyle())
    }
}

/// Active = filled accent capsule (toggle on); inactive = clear Liquid Glass with
/// a hairline. Kept separate so the active/inactive backgrounds compose cleanly.
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
                .glassEffect(.regular, in: Capsule())
                .overlay { Capsule().strokeBorder(.white.opacity(0.18), lineWidth: 1) }
                .shadow(color: .black.opacity(0.28), radius: 8, y: 3)
        }
    }
}

/// The app's ONE floating instruction-pill chrome — used everywhere a hint floats
/// over a live viewport (Pilot's "Rotate to Teleop", the Navigate SLAM notice, the
/// Manipulate mode tips) so they all read identically: a clear Liquid-Glass capsule
/// with a white hairline (never a coloured border). Keep the copy concise.
struct HintPill: View {
    let systemImage: String
    let title: String
    var subtitle: String? = nil
    var iconTint: Color = .white
    var pulse: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: systemImage).font(.system(size: 13, weight: .bold))
                .foregroundStyle(iconTint)
                .symbolEffect(.pulse, isActive: pulse)
            if let subtitle {
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 12.5, weight: .bold)).foregroundStyle(.white)
                    Text(subtitle).font(.system(size: 10, weight: .medium)).foregroundStyle(.white.opacity(0.75))
                }
            } else {
                Text(title).font(.system(size: 12, weight: .semibold)).foregroundStyle(.white)
                    .lineLimit(1).minimumScaleFactor(0.8)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 9)
        .glassEffect(.regular, in: Capsule())
        .overlay { Capsule().strokeBorder(.white.opacity(0.2), lineWidth: 1) }
        .shadow(color: .black.opacity(0.3), radius: 10, y: 4)
    }
}

/// The teleop engage gate — shared by the Pilot cockpit and the landscape Navigate
/// console. While disengaged it's deliberately INVITING: the play glyph pulses and
/// a soft accent glow breathes around the capsule, drawing the eye without a modal
/// hint. Engaged, it turns into a calm red "Stop Teleop".
struct StartTeleopButton: View {
    let active: Bool
    /// Icon-only (portrait, where the row is width-starved) — the pulsing
    /// play/stop glyph carries the meaning.
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
                    Text(active ? "Stop Teleop" : "Start Teleop")
                        .font(.system(size: 13.5, weight: .heavy)).lineLimit(1).fixedSize()
                }
            }
            .foregroundStyle(.white)
            .padding(.horizontal, compact ? 12 : 16).padding(.vertical, 10)
            .background(tint.gradient, in: Capsule())
            .overlay { Capsule().strokeBorder(.white.opacity(active ? 0.25 : 0.4), lineWidth: 1) }
            .shadow(color: tint.opacity(active ? 0.4 : (glow ? 0.85 : 0.45)),
                    radius: active ? 8 : (glow ? 16 : 9), y: 3)
            .scaleEffect(!active && glow ? 1.035 : 1.0)
        }
        .buttonStyle(PressableStyle())
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) { glow = true }
        }
    }
}

/// Press-and-HOLD, reported as edges: `onHold(true)` on press, `onHold(false)` on
/// release or cancel.
///
/// Use this instead of `DragGesture(minimumDistance: 0)` for hold controls that
/// live inside a ScrollView or a sheet. A zero-distance drag gesture CLAIMS every
/// touch that lands on the view, so it eats the scroll/resize drag and fires its
/// action on a gesture the operator meant for the container. A Button's press
/// state cancels the moment the finger travels, so the container gets its drag —
/// and the hold only fires when the touch really stayed put.
struct HoldButtonStyle: ButtonStyle {
    let onHold: (Bool) -> Void
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? 0.92 : 1)
            .onChange(of: configuration.isPressed) { _, down in onHold(down) }
    }
}

/// A spring-scale press effect applied to interactive controls. Uses the
/// critically-damped `press` spring (no overshoot) so taps read crisp and
/// Apple-like rather than bouncy.
struct PressableStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .opacity(configuration.isPressed ? 0.9 : 1)
            .animation(Theme.Motion.press, value: configuration.isPressed)
    }
}

/// The unmistakable emergency stop. Held briefly to engage to prevent
/// accidental taps; a single tap when already engaged releases.
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
                    .font(.mabel(15, weight: .heavy))
                    .tracking(0.5)
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 17)
            .background(
                (engaged ? Theme.Palette.live : Theme.Palette.danger)
                    .gradient,
                in: Capsule()
            )
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
