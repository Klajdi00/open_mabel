//
//  Theme.swift
//  MABEL Mobile App
//
//  Design tokens. The visual language matches the Vision Pro MabelTeleop
//  app: system materials for surfaces, SF system type, and Apple semantic
//  status colours (cyan accent · green live · orange warn · red stop) on a
//  neutral dark canvas.
//

import SwiftUI
import UIKit

enum Theme {

    /// A colour that adapts to light / dark mode (resolves against the active
    /// trait collection, which `.preferredColorScheme` drives).
    static func dyn(_ light: UInt32, _ dark: UInt32, lightA: Double = 1, darkA: Double = 1) -> Color {
        Color(UIColor { tc in
            let hex = tc.userInterfaceStyle == .dark ? dark : light
            let a = tc.userInterfaceStyle == .dark ? darkA : lightA
            return UIColor(red: CGFloat((hex >> 16) & 0xFF) / 255,
                           green: CGFloat((hex >> 8) & 0xFF) / 255,
                           blue: CGFloat(hex & 0xFF) / 255, alpha: a)
        })
    }

    // MARK: - Color  (Apple system semantics, like the Vision Pro app)

    enum Palette {
        // Brand accent — the website's rust/orange. Reads on light & dark.
        static let accent      = Color(hex: 0xC25B2A)   // --rust
        static let accentDeep  = Color(hex: 0x8B3F1D)   // --rust-dim
        static let clay        = Color(hex: 0x9A6A3C)   // warm secondary (no blue)

        static let live        = Color(hex: 0x2E9E52)
        static let warn        = Color(hex: 0xCC8A1E)
        static let danger      = Color(hex: 0xD23B2A)

        // Adaptive surfaces & text — warm, editorial, high-contrast.
        static let canvas      = dyn(0xF3F0E9, 0x0C0A08)   // warm paper / warm ink
        static let ink         = dyn(0xFBF9F4, 0x161412)
        static let surface     = dyn(0x161412, 0xF2EEE6, lightA: 0.045, darkA: 0.06)
        static let surfaceHi   = dyn(0x161412, 0xF2EEE6, lightA: 0.075, darkA: 0.10)
        static let hairline    = dyn(0x161412, 0xF2EEE6, lightA: 0.14, darkA: 0.12)
        static let hairlineSoft = dyn(0x161412, 0xF2EEE6, lightA: 0.07, darkA: 0.07)

        static let textPrimary   = dyn(0x1A1714, 0xF2EEE6)
        static let textSecondary = dyn(0x1A1714, 0xF2EEE6, lightA: 0.62, darkA: 0.64)
        static let textTertiary  = dyn(0x1A1714, 0xF2EEE6, lightA: 0.45, darkA: 0.42)

        // Fixed inks for the ALWAYS-DARK 3D viewport (NOT adaptive): light text
        // that sits over the live map / camera scene, and the near-black scene
        // fills. Named here so feature code stops hardcoding these hexes.
        static let onViewport  = Color(hex: 0xF4F1EA)   // light ink over the dark viewer
        static let viewportInk = Color(hex: 0x0A0908)   // camera / scene backdrop
        static let violet      = Color(hex: 0xB18CFF)   // sleep / demos accent

        /// One source of truth for chart + status hues (module telemetry). Indices
        /// 0…6 map to the joint/module palette; status bars reference live/warn/danger
        /// so charts and the status system can never drift to slightly-different hex.
        static let dataHues: [Color] = [
            accent,                 // 0 rust
            live,                   // 1 green
            Color(hex: 0x4D9DFF),   // 2 blue
            warn,                   // 3 amber
            Color(hex: 0x8E6FD8),   // 4 violet
            clay,                   // 5 clay
            Color(hex: 0x2AA8A0),   // 6 teal
        ]
    }

    static var accentGradient: LinearGradient {
        LinearGradient(colors: [Color(hex: 0xD66A33), Palette.accent],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    static var canvasGradient: RadialGradient {
        RadialGradient(colors: [dyn(0xFAF7F1, 0x211D19), Palette.canvas],
                       center: .init(x: 0.5, y: 0.32), startRadius: 40, endRadius: 720)
    }

    // MARK: - Geometry

    enum Radius {
        static let xs: CGFloat = 6      // tiny chips / inner tiles
        static let sm: CGFloat = 10
        static let chip: CGFloat = 12    // pills, small controls
        static let md: CGFloat = 15
        static let lg: CGFloat = 20      // cards / sheets
        static let pill: CGFloat = 999
    }

    enum Space {
        static let xs: CGFloat = 5
        static let sm: CGFloat = 9
        static let md: CGFloat = 14
        static let lg: CGFloat = 20
        static let xl: CGFloat = 30
    }

    // MARK: - Motion

    enum Motion {
        static let snappy = Animation.spring(response: 0.32, dampingFraction: 0.85)
        static let smooth = Animation.spring(response: 0.5,  dampingFraction: 0.9)
        static let gentle = Animation.easeInOut(duration: 0.22)

        // Named springs matching Apple's interaction guidance: critically
        // damped for direct UI, under-damped only when a gesture carried momentum.
        // Prefer these over fixed-duration easing for anything interactive/interruptible.
        static let press    = Animation.spring(response: 0.30, dampingFraction: 1.0)
        static let move     = Animation.spring(response: 0.40, dampingFraction: 1.0)
        static let sheet    = Animation.spring(response: 0.30, dampingFraction: 0.8)
        static let momentum = Animation.spring(response: 0.35, dampingFraction: 0.8)
        static let rotate   = Animation.spring(response: 0.40, dampingFraction: 0.8)

        /// Motion that respects Reduce Motion: cross-fade instead of spring/slide.
        static func adaptive(_ base: Animation, reduceMotion: Bool) -> Animation {
            reduceMotion ? .easeInOut(duration: 0.2) : base
        }
    }

    // MARK: - Elevation

    /// Depth tokens — replace ad-hoc `.shadow(...)` call sites with
    /// `.mabelShadow(Theme.Shadow.card)`. Bigger surfaces cast deeper shadows.
    enum Shadow {
        static let card     = MabelShadow(color: .black.opacity(0.30), radius: 14, x: 0, y: 8)
        static let pill     = MabelShadow(color: .black.opacity(0.28), radius: 10, x: 0, y: 4)
        static let floating = MabelShadow(color: .black.opacity(0.28), radius: 8,  x: 0, y: 3)
        static let overlay  = MabelShadow(color: .black.opacity(0.40), radius: 24, x: 0, y: 12)
    }

    // MARK: - Typography tracking (size-specific letter-spacing)

    /// Apple tightens large text and gives small all-caps labels air. Point-based
    /// tracking keyed to type role — pair with the `Font.mabel*` scale.
    enum Tracking {
        static let display: CGFloat   = -0.5   // large titles
        static let title: CGFloat     = -0.3
        static let body: CGFloat      = 0
        static let label: CGFloat     = 0.2
        static let microCaps: CGFloat = 1.4    // tracked all-caps micro labels
    }
}

// MARK: - Shadow token

/// A reusable drop-shadow spec (named `MabelShadow` to avoid colliding with
/// SwiftUI's own `ShadowStyle`). Apply via `.mabelShadow(_:)`.
struct MabelShadow {
    let color: Color
    let radius: CGFloat
    let x: CGFloat
    let y: CGFloat
}

extension View {
    /// Apply a `Theme.Shadow` depth token.
    func mabelShadow(_ s: MabelShadow) -> some View {
        shadow(color: s.color, radius: s.radius, x: s.x, y: s.y)
    }
}

// MARK: - Typography (SF system, like the Vision Pro app)

extension Font {
    static func mabel(_ size: CGFloat, weight: Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }

    static let mabelLargeTitle = Font.system(size: 30, weight: .bold)
    /// The standard screen/section title — consistent across every view.
    static let mabelViewTitle  = Font.system(size: 23, weight: .bold)
    static let mabelTitle      = Font.system(size: 20, weight: .semibold)
    static let mabelHeadline   = Font.system(size: 16, weight: .semibold)
    static let mabelBody       = Font.system(size: 14, weight: .regular)
    static let mabelCaption    = Font.system(size: 11, weight: .medium)
    static let mabelMono       = Font.system(size: 13, weight: .medium, design: .monospaced)
    static let mabelMonoSmall  = Font.system(size: 11, weight: .medium, design: .monospaced)

    // Extended scale — named roles the ad-hoc inline sizes collapse onto.
    static let mabelDisplay   = Font.system(size: 34, weight: .bold)      // hero numerals / splash
    static let mabelValue     = Font.system(size: 22, weight: .semibold)  // KPI values
    static let mabelLabel     = Font.system(size: 13, weight: .medium)    // row labels / controls
    static let mabelMicro     = Font.system(size: 9.5, weight: .semibold) // tracked micro labels
    static let mabelMonoTiny  = Font.system(size: 10, weight: .medium, design: .monospaced) // endpoints
}

// MARK: - Icon helper (SF Symbol OR a custom symbol asset)

extension Image {
    /// Render an SF Symbol by name, or a custom symbol from the asset catalog when
    /// the name is prefixed with "custom:" (e.g. the robot glyph SF Symbols lacks).
    /// Custom symbols scale with `.font` and tint with `.foregroundStyle`, exactly
    /// like system symbols, so call sites stay identical.
    init(icon name: String) {
        if name.hasPrefix("custom:") {
            self.init(String(name.dropFirst("custom:".count)))
        } else {
            self.init(systemName: name)
        }
    }
}

// MARK: - Color hex helper

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(.sRGB,
                  red:   Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8)  & 0xFF) / 255,
                  blue:  Double(hex & 0xFF) / 255,
                  opacity: alpha)
    }
}
