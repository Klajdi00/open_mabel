//
//  Theme.swift
//  MabelTeleop
//
//  Design tokens — the SAME visual language as the MABEL iOS app (rust/orange
//  brand accent, warm editorial surfaces, Apple semantic status colours), tuned
//  for visionOS where every window already sits on liquid glass. Surfaces are
//  translucent overlays so the glass shows through.
//

import SwiftUI

enum Theme {

    // MARK: - Color (matches MABEL Mobile App Theme.Palette)

    enum Palette {
        // Brand accent — the website's rust/orange.
        static let accent      = Color(hex: 0xD66A33)   // brighter rust for dark glass
        static let accentDeep  = Color(hex: 0x8B3F1D)
        static let clay        = Color(hex: 0xB98B55)

        static let live        = Color(hex: 0x3FB565)
        static let warn        = Color(hex: 0xDFA02E)
        static let danger      = Color(hex: 0xE0503D)

        // On visionOS the canvas is glass — surfaces are soft white overlays.
        static let canvas       = Color.black.opacity(0.15)
        static let ink          = Color.black.opacity(0.25)
        static let surface      = Color.white.opacity(0.06)
        static let surfaceHi    = Color.white.opacity(0.11)
        static let hairline     = Color.white.opacity(0.16)
        static let hairlineSoft = Color.white.opacity(0.09)

        static let textPrimary   = Color.white.opacity(0.96)
        static let textSecondary = Color.white.opacity(0.65)
        static let textTertiary  = Color.white.opacity(0.42)

        // Fixed inks for the ALWAYS-DARK 3D viewport: light text that sits over the
        // live map / camera scene, and the near-black scene fills. Named here so
        // feature code stops hardcoding these hexes. (Parallel to the iOS app.)
        static let onViewport  = Color(hex: 0xF4F1EA)   // light ink over the dark viewer
        static let viewportInk = Color(hex: 0x0A0908)   // camera / scene backdrop
        static let violet      = Color(hex: 0xB18CFF)   // sleep / demos accent

        /// One source of truth for chart + status hues (module telemetry). Indices
        /// 0…6 map to the joint/module palette; status bars reference live/warn/danger.
        static let dataHues: [Color] = [
            accent,                 // 0 rust
            live,                   // 1 green
            Color(hex: 0x5AA9FF),   // 2 blue  (brighter for dark glass)
            warn,                   // 3 amber
            Color(hex: 0x9C7FE6),   // 4 violet
            clay,                   // 5 clay
            Color(hex: 0x37B7AE),   // 6 teal
        ]
    }

    static var accentGradient: LinearGradient {
        LinearGradient(colors: [Color(hex: 0xE2793F), Palette.accent],
                       startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    // MARK: - Geometry

    enum Radius {
        static let xs: CGFloat = 6      // tiny chips / inner tiles
        static let sm: CGFloat = 10
        static let chip: CGFloat = 12    // pills, small controls
        static let md: CGFloat = 15
        static let lg: CGFloat = 22      // cards / sheets (a touch larger on VP)
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

        // Named springs matching the Apple interaction values (SKILL.md): critically
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
    /// `.mabelShadow(Theme.Shadow.card)`. Softer than the iOS twin because the
    /// glass already reads as depth over passthrough.
    enum Shadow {
        static let card     = MabelShadow(color: .black.opacity(0.18), radius: 10, x: 0, y: 4)
        static let pill     = MabelShadow(color: .black.opacity(0.16), radius: 8,  x: 0, y: 3)
        static let floating = MabelShadow(color: .black.opacity(0.14), radius: 6,  x: 0, y: 2)
        static let overlay  = MabelShadow(color: .black.opacity(0.30), radius: 20, x: 0, y: 10)
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

// MARK: - Typography (SF system, same scale as the iOS app)

extension Font {
    static func mabel(_ size: CGFloat, weight: Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }

    static let mabelLargeTitle = Font.system(size: 32, weight: .bold)
    /// The standard screen/section title — consistent across every view.
    static let mabelViewTitle  = Font.system(size: 26, weight: .bold)
    static let mabelTitle      = Font.system(size: 21, weight: .semibold)
    static let mabelHeadline   = Font.system(size: 17, weight: .semibold)
    static let mabelBody       = Font.system(size: 15, weight: .regular)
    static let mabelCaption    = Font.system(size: 12, weight: .medium)
    static let mabelMono       = Font.system(size: 14, weight: .medium, design: .monospaced)
    static let mabelMonoSmall  = Font.system(size: 12, weight: .medium, design: .monospaced)

    // Extended scale — named roles the ad-hoc inline sizes collapse onto
    // (a touch larger than the iOS twin for headset viewing distance).
    static let mabelDisplay   = Font.system(size: 36, weight: .bold)      // hero numerals / splash
    static let mabelValue     = Font.system(size: 24, weight: .semibold)  // KPI values
    static let mabelLabel     = Font.system(size: 14, weight: .medium)    // row labels / controls
    static let mabelMicro     = Font.system(size: 11, weight: .semibold)  // tracked micro labels
    static let mabelMonoTiny  = Font.system(size: 11, weight: .medium, design: .monospaced) // endpoints
}

// MARK: - Surfaces

extension View {
    /// A translucent frosted "page" behind a PURE-2D view (Settings, Policies):
    /// the native visionOS look where grouped cards sit on a clean glass surface
    /// instead of floating on raw passthrough. Reads as more SOLID than the
    /// floating chrome bars (which stay light `glassBackgroundEffect`), matching
    /// the intended hierarchy — transparent chrome, translucent content.
    ///
    /// Do NOT place behind a RealityView holding recessed (negative-z) 3D: a
    /// window-plane material renders in FRONT of it ("phantom window"). For
    /// 3D-bearing views, back only their flat side panels.
    func consoleSurface(_ cornerRadius: CGFloat = 30) -> some View {
        self
            .background {
                // Frosted glass page + a faint dark wash: keeps white text legible
                // over bright passthrough AND lets the (un-washed) regularMaterial
                // cards read as raised sections, for clean Apple-like depth.
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Theme.Palette.canvas)
                    .background(.regularMaterial,
                                in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            }
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1)
            }
    }
}

// MARK: - Icon helper (SF Symbol OR a custom symbol asset)

extension Image {
    /// Render an SF Symbol by name, or a custom symbol from the asset catalog when
    /// the name is prefixed with "custom:" (e.g. the robot glyph SF Symbols lacks).
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
