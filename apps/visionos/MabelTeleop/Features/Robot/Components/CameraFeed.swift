//
//  CameraFeed.swift
//  MABEL Mobile App
//
//  A camera-stream tile. Until a real MJPEG/WebRTC stream is wired in
//  (see RobotLink), it renders a convincing synthetic "view" — a receding
//  perspective grid, horizon and vignette — with a live badge, label and
//  timecode overlay, so the cockpit reads like a real teleop feed.
//

import SwiftUI

struct CameraFeed: View {
    let name: String
    var compact: Bool = false
    /// Main feed gets framing reticles; wrist feeds stay clean.
    var isMain: Bool = false
    var tint: Color = Theme.Palette.accent
    /// When connected to a real server, the live MJPEG stream URL.
    var streamURL: URL? = nil
    /// The wire stream key ("main" | "wrist_left" | …).
    var stream: String? = nil
    /// Tap-to-expand: when set, tapping the tile opens the camera-detail window
    /// for this stream (zoom + pan). The tile shows a small expand glyph and a
    /// gaze hover highlight so the affordance is discoverable. nil = inert tile.
    var onExpand: (() -> Void)? = nil

    /// True once the MJPEG stream delivers frames; false on a 404 / error (a
    /// camera that isn't wired) -> show "NOT CONNECTED". Optimistic so a healthy
    /// feed never flashes the badge while the first frame is in flight.
    @State private var live = true

    var body: some View {
        // The tap gesture is attached ONLY when expandable, so inert tiles
        // never swallow touches meant for what's underneath them.
        if let onExpand {
            let shape = RoundedRectangle(cornerRadius: compact ? Theme.Radius.sm : Theme.Radius.md,
                                         style: .continuous)
            tile.contentShape(shape)
                .contentShape(.hoverEffect, shape)
                .hoverEffect()
                .onTapGesture { Haptics.select(); onExpand() }
        } else {
            tile
        }
    }

    private var tile: some View {
        ZStack {
            if let streamURL {                              // MJPEG
                MJPEGView(url: streamURL, onLive: { live = $0 })
            } else {
                TimelineView(.animation(minimumInterval: 1.0 / 12.0)) { tl in
                    Canvas { ctx, size in draw(ctx, size, time: tl.date.timeIntervalSinceReferenceDate) }
                }
            }
            overlay
            if streamURL != nil && !live { notConnected }
        }
        .background(Color(hex: 0x07060A))
        .clipShape(RoundedRectangle(cornerRadius: compact ? Theme.Radius.sm : Theme.Radius.md, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: compact ? Theme.Radius.sm : Theme.Radius.md, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        }
        .overlay(alignment: .bottomTrailing) {
            if onExpand != nil {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: compact ? 7 : 10, weight: .bold))
                    .foregroundStyle(.white.opacity(0.65))
                    .padding(compact ? 3 : 5)
                    .background(.black.opacity(0.4), in: Circle())
                    .padding(compact ? 4 : 7)
                    .allowsHitTesting(false)
            }
        }
    }

    // MARK: - Not-connected badge

    /// Shown over a tile whose camera stream is dead (404 / no frames) — e.g. a
    /// wrist cam that isn't wired on the robot — so it reads "NOT CONNECTED"
    /// instead of a frozen or blank frame.
    private var notConnected: some View {
        ZStack {
            Color.black.opacity(0.82)
            VStack(spacing: compact ? 2 : 5) {
                Image(systemName: "video.slash.fill")
                    .font(.system(size: compact ? 11 : 20, weight: .regular))
                Text("NOT CONNECTED")
                    .font(.system(size: compact ? 7 : 11, weight: .bold, design: .monospaced))
                    .tracking(1.4)
            }
            .foregroundStyle(.white.opacity(0.55))
        }
        .allowsHitTesting(false)
    }

    // MARK: - Synthetic scene

    private func draw(_ ctx: GraphicsContext, _ size: CGSize, time: TimeInterval) {
        let w = size.width, h = size.height
        let horizon = h * 0.46

        // Sky / floor wash
        ctx.fill(Path(CGRect(x: 0, y: 0, width: w, height: horizon)),
                 with: .linearGradient(.init(colors: [Color(hex: 0x14110E), Color(hex: 0x0B0A09)]),
                                       startPoint: .zero, endPoint: CGPoint(x: 0, y: horizon)))
        ctx.fill(Path(CGRect(x: 0, y: horizon, width: w, height: h - horizon)),
                 with: .linearGradient(.init(colors: [Color(hex: 0x1A1612), Color(hex: 0x0E0C0B)]),
                                       startPoint: CGPoint(x: 0, y: horizon), endPoint: CGPoint(x: 0, y: h)))

        // Perspective floor grid receding to a vanishing point.
        let vp = CGPoint(x: w / 2, y: horizon)
        var grid = Path()
        let drift = CGFloat((time * 0.25).truncatingRemainder(dividingBy: 1))
        // Horizontal depth lines
        for i in 0..<8 {
            let t = (CGFloat(i) + drift) / 8
            let y = horizon + (h - horizon) * t * t
            grid.move(to: CGPoint(x: 0, y: y)); grid.addLine(to: CGPoint(x: w, y: y))
        }
        // Vertical converging lines
        for i in -6...6 {
            let x = vp.x + CGFloat(i) * (w / 6)
            grid.move(to: CGPoint(x: vp.x + CGFloat(i) * 14, y: horizon))
            grid.addLine(to: CGPoint(x: x, y: h))
        }
        ctx.stroke(grid, with: .color(tint.opacity(0.16)), lineWidth: 1)

        // Soft horizon glow
        var glow = Path()
        glow.addRect(CGRect(x: 0, y: horizon - 1.5, width: w, height: 3))
        ctx.fill(glow, with: .color(tint.opacity(0.28)))

        // Vignette
        ctx.fill(Path(CGRect(origin: .zero, size: size)),
                 with: .radialGradient(.init(colors: [.clear, .black.opacity(0.55)]),
                                       center: CGPoint(x: w/2, y: h/2),
                                       startRadius: min(w, h) * 0.2, endRadius: max(w, h) * 0.7))
    }

    // MARK: - HUD overlay

    private var overlay: some View {
        VStack {
            HStack(alignment: .top) {
                // Only the small wrist thumbnails carry a label (L / R); the main
                // POV stays clean — no "HEAD · POV" or resolution text.
                if compact {
                    HStack(spacing: 4) {
                        Circle().fill(Theme.Palette.danger).frame(width: 5, height: 5)
                        Text(name).font(.system(size: 8, weight: .bold, design: .rounded))
                            .foregroundStyle(Theme.Palette.textPrimary)
                    }
                    .padding(.horizontal, 5).padding(.vertical, 3)
                    .background(.black.opacity(0.4), in: Capsule())
                }
                Spacer()
            }
            Spacer()
        }
        .padding(compact ? 5 : 9)
        .overlay {
            if isMain {
                // Center framing reticle
                Image(systemName: "viewfinder")
                    .font(.system(size: 26, weight: .ultraLight))
                    .foregroundStyle(.white.opacity(0.25))
            }
        }
    }
}
