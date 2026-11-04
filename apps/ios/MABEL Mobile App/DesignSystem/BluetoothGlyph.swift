//
//  BluetoothGlyph.swift
//  MABEL Mobile App
//
//  THE BLUETOOTH RUNE, DRAWN.
//
//  SF Symbols has no Bluetooth glyph — Apple does not ship the trademark — so every
//  Bluetooth surface in this app borrowed `dot.radiowaves.left.and.right`, which is the
//  generic "broadcasting" glyph it also uses for AirPlay, Wi-Fi strength and a lidar sweep.
//  At a status-bar size that is indistinguishable from the Wi-Fi icon sitting next to it,
//  which is exactly the confusion reported: you could not tell which transport was live.
//
//  The rune (ᛒ, Harald "Bluetooth" Gormsson's bind-rune) is six line segments, so it is
//  drawn rather than shipped as an asset — it stays sharp at any size, takes the current
//  foreground colour like a template, and needs no asset catalogue entry.
//

import SwiftUI

/// The rune as a single continuous stroke: bottom-left → upper-right vertex → top of the
/// stem → down the stem → lower-right vertex → upper-left. That one polyline IS the logo.
struct BluetoothRune: Shape {
    func path(in rect: CGRect) -> Path {
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            CGPoint(x: rect.minX + x * rect.width, y: rect.minY + y * rect.height)
        }
        var path = Path()
        path.move(to:    p(0.08, 0.72))
        path.addLine(to: p(0.92, 0.28))
        path.addLine(to: p(0.50, 0.02))
        path.addLine(to: p(0.50, 0.98))
        path.addLine(to: p(0.92, 0.72))
        path.addLine(to: p(0.08, 0.28))
        return path
    }
}

/// Sized to sit inline with `Image(systemName:)` at the same point size. The rune is taller
/// than it is wide, so the frame is keyed to height and the width follows the logo's ratio.
struct BluetoothGlyph: View {
    var size: CGFloat = 14
    /// Matches the weight of the SF Symbols beside it (`.semibold` ≈ 0.14 × size).
    var lineWidth: CGFloat? = nil

    var body: some View {
        BluetoothRune()
            .stroke(style: StrokeStyle(lineWidth: lineWidth ?? max(1.2, size * 0.13),
                                       lineCap: .round, lineJoin: .round))
            .frame(width: size * 0.60, height: size)
            // Optically centred in the space an SF Symbol of the same point size occupies.
            .frame(width: size, alignment: .center)
            .accessibilityLabel("Bluetooth")
    }
}

/// One glyph for any transport: the drawn rune for Bluetooth, the SF Symbol for the rest.
/// EVERY transport icon goes through this — the status-bar chip, the rows in its dropdown,
/// the Settings readouts and the connect card — so the two can never be confused again.
/// Any other symbol name passes straight through to `Image(icon:)`, so this is a drop-in
/// replacement for an `Image` at a call site that MIGHT be showing a route.
struct TransportGlyph: View {
    /// What to draw when this is not Bluetooth.
    private let symbol: String
    private let isBluetooth: Bool
    private let size: CGFloat
    private let weight: Font.Weight
    /// Pulses while a link is being established — the equivalent of `.symbolEffect(.pulse)`,
    /// which cannot reach the rune because the rune is a Shape, not a symbol.
    private let pulsing: Bool

    @State private var pulsePhase = false

    /// From a route. `nil` — nothing is carrying commands — draws `fallback`.
    init(kind: RouteKind?, size: CGFloat = 14, weight: Font.Weight = .semibold,
         pulsing: Bool = false, fallback: String = "wifi.slash") {
        self.isBluetooth = (kind == .bluetooth)
        self.symbol = kind?.icon ?? fallback
        self.size = size
        self.weight = weight
        self.pulsing = pulsing
    }

    /// From a bare symbol name, for the call sites that only carry the string.
    init(icon: String, size: CGFloat = 14, weight: Font.Weight = .semibold,
         pulsing: Bool = false) {
        self.isBluetooth = (icon == RouteKind.bluetooth.icon)
        self.symbol = icon
        self.size = size
        self.weight = weight
        self.pulsing = pulsing
    }

    var body: some View {
        if isBluetooth {
            BluetoothGlyph(size: size, lineWidth: max(1.2, size * strokeScale))
                .opacity(pulsePhase ? 0.35 : 1)
                .task(id: pulsing) {
                    pulsePhase = false
                    guard pulsing else { return }
                    withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) {
                        pulsePhase = true
                    }
                }
        } else {
            Image(icon: symbol)
                .font(.system(size: size, weight: weight))
                .symbolEffect(.pulse, isActive: pulsing)
        }
    }

    /// Keep the rune's stroke as heavy as the SF Symbols sitting beside it.
    private var strokeScale: CGFloat {
        switch weight {
        case .black, .heavy:            return 0.17
        case .bold:                     return 0.15
        case .ultraLight, .thin, .light: return 0.10
        default:                        return 0.13
        }
    }
}
