//
//  RobotGlyph.swift
//  MABEL Mobile App
//
//  THE ROBOT, DRAWN.
//
//  Same reasoning as BluetoothGlyph: SF Symbols ships no robot, so the app was
//  borrowing `gearshape.2.fill` for the real machine. Two cogs read as "settings"
//  everywhere else in iOS — and it sits inches from an actual settings gear in the
//  same bar — so the one control that says WHICH ROBOT YOU ARE DRIVING looked like
//  a preferences button.
//
//  Drawn rather than shipped as an asset, for the reasons the Bluetooth rune is:
//  it stays sharp at any size, takes the current foreground colour like a template
//  symbol, needs no asset-catalogue entry, and cannot fall out of step between the
//  iPhone and Vision Pro targets.
//
//  The mark is MABEL's own silhouette reduced to what survives at 12 pt: a rounded
//  head, two eyes, a stubby antenna, and the shoulder line of the torso below. No
//  mouth, no detail — anything finer turns to mud at status-bar size.
//

import SwiftUI

/// A small robot mark that behaves like an SF Symbol: size it with `size:`, colour
/// it with `.foregroundStyle`. Aspect is fixed so it never distorts.
struct RobotGlyph: View {
    /// Cap height, matching the `.font(.system(size:))` you would give a symbol.
    var size: CGFloat = 12
    /// Filled reads as "connected"; outline as "not yet".
    var filled: Bool = true

    private var w: CGFloat { size * 1.10 }
    private var h: CGFloat { size }

    var body: some View {
        Canvas { ctx, rect in
            let s = min(rect.width / 1.10, rect.height)
            let ox = (rect.width - s * 1.10) / 2
            let oy = (rect.height - s) / 2
            func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
                CGPoint(x: ox + x * s * 1.10, y: oy + y * s)
            }

            // Antenna: a stem and a dot, centred over the head.
            var stem = Path()
            stem.move(to: p(0.50, 0.14))
            stem.addLine(to: p(0.50, 0.02))
            ctx.stroke(stem, with: .style(.foreground),
                       style: StrokeStyle(lineWidth: s * 0.10, lineCap: .round))
            ctx.fill(Path(ellipseIn: CGRect(origin: p(0.43, -0.06),
                                            size: CGSize(width: s * 0.155, height: s * 0.155))),
                     with: .style(.foreground))

            // Head with the eyes cut straight out of it, as ONE path filled even-odd.
            //
            // The eyes were knocked out with `.destinationOut` before. Inside a Canvas
            // that punches through to transparency, so at 12 pt the eyes became two
            // pinholes onto the frosted capsule behind — invisible once the glyph went
            // green on connect, which is exactly when you most want to see a face.
            // An even-odd fill makes them real holes in the shape itself: they always
            // read, whatever sits behind, at any tint.
            let head = CGRect(origin: p(0.12, 0.14),
                              size: CGSize(width: s * 0.84, height: s * 0.54))
            let eyeD = s * 0.175
            let eyes = [p(0.285, 0.325), p(0.575, 0.325)]

            if filled {
                var mask = Path(roundedRect: head, cornerRadius: s * 0.20, style: .continuous)
                for e in eyes {
                    mask.addEllipse(in: CGRect(origin: e,
                                               size: CGSize(width: eyeD, height: eyeD)))
                }
                ctx.fill(mask, with: .style(.foreground), style: FillStyle(eoFill: true))
            } else {
                // Not connected: an outlined head with solid eyes. Same silhouette,
                // clearly "not live" — and the eyes stay visible either way.
                ctx.stroke(Path(roundedRect: head, cornerRadius: s * 0.20, style: .continuous),
                           with: .style(.foreground), lineWidth: s * 0.105)
                for e in eyes {
                    ctx.fill(Path(ellipseIn: CGRect(origin: e,
                                                    size: CGSize(width: eyeD, height: eyeD))),
                             with: .style(.foreground))
                }
            }

            // Shoulders: one short bar under the head, so the mark is a robot and
            // not just a face. Deliberately detached — a neck line closes up at size.
            let sh = CGRect(origin: p(0.24, 0.78),
                            size: CGSize(width: s * 0.60, height: s * 0.17))
            ctx.fill(Path(roundedRect: sh, cornerRadius: s * 0.075, style: .continuous),
                     with: .style(.foreground))
        }
        .frame(width: w, height: h * 1.12)   // headroom for the antenna dot
        .accessibilityHidden(true)
    }
}

/// MABEL head to wheels — the tab-bar mark.
///
/// The Robot tab used `move.3d`, a 3-axis gizmo: correct for a viewport, wrong for
/// the destination that IS the robot. This draws the machine's actual silhouette —
/// **mobile bimanual**: a head over a torso with an arm hanging either side, on a
/// lift column above a wheeled base. Those four masses are what distinguish MABEL
/// from any other robot, and they are the most that survives at 15 pt.
///
/// Style matches the rest of the bar: solid masses like `gearshape.fill`, one
/// knocked-out visor for the face. No outlines — a 1 pt stroke at tab size turns to
/// grey mush next to the filled symbols beside it.
struct MabelBodyGlyph: View {
    /// Cap height, as you would pass to `.font(.system(size:))`.
    var size: CGFloat = 15

    var body: some View {
        Canvas { ctx, rect in
            let h = min(rect.height, rect.width / 0.92)
            let w = h * 0.92
            let ox = (rect.width - w) / 2
            let oy = (rect.height - h) / 2
            func r(_ x: CGFloat, _ y: CGFloat, _ x2: CGFloat, _ y2: CGFloat,
                   _ rad: CGFloat) -> Path {
                Path(roundedRect: CGRect(x: ox + x * w, y: oy + y * h,
                                         width: (x2 - x) * w, height: (y2 - y) * h),
                     cornerRadius: rad * h, style: .continuous)
            }
            func dot(_ cx: CGFloat, _ cy: CGFloat, _ rad: CGFloat) -> Path {
                Path(ellipseIn: CGRect(x: ox + (cx - rad) * w, y: oy + (cy - rad) * h,
                                       width: 2 * rad * w, height: 2 * rad * h))
            }
            let fg = GraphicsContext.Shading.style(.foreground)

            // Antenna — the one thing that says "robot" before any detail resolves.
            ctx.fill(dot(0.50, 0.055, 0.055), with: fg)
            var stem = Path()
            stem.move(to: CGPoint(x: ox + 0.50 * w, y: oy + 0.10 * h))
            stem.addLine(to: CGPoint(x: ox + 0.50 * w, y: oy + 0.15 * h))
            ctx.stroke(stem, with: fg, style: StrokeStyle(lineWidth: h * 0.05, lineCap: .round))

            // Head with the visor cut out, even-odd — not `.destinationOut`, which
            // punches to transparency and vanishes against a tinted background.
            var headMask = r(0.31, 0.14, 0.69, 0.33, 0.07)
            headMask.addPath(r(0.37, 0.205, 0.63, 0.262, 0.028))
            ctx.fill(headMask, with: fg, style: FillStyle(eoFill: true))

            // Torso, and an arm either side — the bimanual signature.
            ctx.fill(r(0.30, 0.37, 0.70, 0.60, 0.075), with: fg)
            ctx.fill(r(0.10, 0.375, 0.235, 0.615, 0.062), with: fg)
            ctx.fill(r(0.765, 0.375, 0.90, 0.615, 0.062), with: fg)

            // Lift column, then the mobile base and its wheels.
            ctx.fill(r(0.435, 0.60, 0.565, 0.70, 0.02), with: fg)
            ctx.fill(r(0.22, 0.70, 0.78, 0.83, 0.05), with: fg)
            ctx.fill(dot(0.315, 0.895, 0.083), with: fg)
            ctx.fill(dot(0.685, 0.895, 0.083), with: fg)
        }
        .frame(width: size * 0.92, height: size)
        .accessibilityHidden(true)
    }
}

#if DEBUG
#Preview("RobotGlyph") {
    VStack(spacing: 18) {
        HStack(spacing: 14) {
            RobotGlyph(size: 12)
            RobotGlyph(size: 18)
            RobotGlyph(size: 28)
            RobotGlyph(size: 44)
        }
        .foregroundStyle(Theme.Palette.accent)

        HStack(spacing: 14) {
            RobotGlyph(size: 12, filled: false)
            RobotGlyph(size: 18, filled: false)
            RobotGlyph(size: 28, filled: false)
        }
        .foregroundStyle(Theme.Palette.textSecondary)

        // The full body at tab size, beside the symbols it sits next to.
        HStack(spacing: 16) {
            MabelBodyGlyph(size: 15)
            Image(systemName: "brain.head.profile").font(.system(size: 15, weight: .semibold))
            Image(systemName: "gearshape.fill").font(.system(size: 15, weight: .semibold))
        }
        .foregroundStyle(Theme.Palette.accent)
        HStack(spacing: 16) {
            MabelBodyGlyph(size: 22)
            MabelBodyGlyph(size: 34)
            MabelBodyGlyph(size: 56)
        }
        .foregroundStyle(Theme.Palette.textPrimary)

        // Next to the gear it replaced, to check it no longer reads as settings.
        HStack(spacing: 14) {
            RobotGlyph(size: 16)
            Image(systemName: "gearshape.2.fill").font(.system(size: 16))
            Image(systemName: "gearshape.fill").font(.system(size: 16))
        }
        .foregroundStyle(Theme.Palette.textPrimary)
    }
    .padding(30)
    .background(Theme.Palette.canvas)
}
#endif
