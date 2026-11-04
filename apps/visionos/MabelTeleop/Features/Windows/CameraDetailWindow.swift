//
//  CameraDetailWindow.swift
//  MabelTeleop
//
//  A placeable visionOS window showing ONE camera stream big — opened by
//  tapping any camera tile (console camera rail, Sensors window, Teleop tab,
//  FPV wrist chrome). Value-parameterized by the wire stream key, so each
//  camera gets its OWN window the operator can pin anywhere in the room.
//
//  Video rides the same transport ladder as the Teleop hero (LAN UDP
//  H.264/JPEG → MJPEG fallback) at HERO quality
//  (`videoMainURL`, not the thumbnail transcode), with pinch-to-zoom,
//  drag-to-pan, double-tap zoom and ± buttons for gaze-and-pinch users.
//

import SwiftUI
import UIKit

struct CameraDetailWindow: View {
    // Plain reference, NOT @EnvironmentObject — same rule as every aux window:
    // observing the controller re-renders at uiTick rate for no benefit here.
    let controller: RobotController
    let streamKey: String

    @State private var image: UIImage?
    @State private var feed: VideoFeed?

    // Committed zoom/pan. Gesture deltas apply on top of a base frozen at
    // gesture start (MagnifyGesture/DragGesture report totals, not deltas).
    @State private var scale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var pinchBase: (scale: CGFloat, offset: CGSize)? = nil
    @State private var dragBase: CGSize? = nil

    private static let minScale: CGFloat = 1
    private static let maxScale: CGFloat = 8

    private var title: String {
        switch streamKey {
        case "main":        return "HEAD · POV"
        case "wrist_left":  return "L WRIST"
        case "wrist_right": return "R WRIST"
        case "front":       return "BASE"
        default:            return streamKey.replacingOccurrences(of: "_", with: " ").uppercased()
        }
    }

    private var tint: Color {
        switch streamKey {
        case "wrist_left":  return Theme.Palette.live
        case "wrist_right": return Theme.Palette.clay
        default:            return Theme.Palette.accent
        }
    }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: geo.size.width, height: geo.size.height)
                        .scaleEffect(scale)
                        .offset(offset)
                } else {
                    placeholder
                }
                chrome(in: geo.size)
            }
            .contentShape(Rectangle())
            .gesture(zoomGestures(in: geo.size))
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        }
        .padding(10)
        .onAppear { start() }
        .onDisappear { stop() }
    }

    // MARK: - Video plumbing (CameraFeedView's ladder, hero quality)

    private func start() {
        guard feed == nil else { return }
        let f = VideoFeed(host: controller.videoHost, udpPort: controller.videoUDPPort,
                          codec: controller.videoCodec, stream: streamKey,
                          mjpegURL: controller.videoMainURL(streamKey),
                          label: title) { cg in
            Task { @MainActor in image = UIImage(cgImage: cg) }
        }
        f.start()
        feed = f
    }

    private func stop() {
        feed?.stop()
        feed = nil
        image = nil
    }

    // MARK: - Gestures (pinch anchored at the hands, pan clamped to frame)

    private func zoomGestures(in size: CGSize) -> some Gesture {
        let pinch = MagnifyGesture()
            .onChanged { v in
                let base = pinchBase ?? (scale, offset)
                pinchBase = base
                let s = min(max(base.scale * v.magnification, Self.minScale), Self.maxScale)
                // Keep the pinch's start point fixed on screen while scaling.
                let anchor = CGPoint(x: (v.startAnchor.x - 0.5) * size.width,
                                     y: (v.startAnchor.y - 0.5) * size.height)
                let k = s / base.scale
                scale = s
                offset = Self.clamp(CGSize(width: anchor.x - (anchor.x - base.offset.width) * k,
                                           height: anchor.y - (anchor.y - base.offset.height) * k),
                                    scale: s, in: size)
            }
            .onEnded { _ in
                pinchBase = nil
                withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) {
                    if scale < 1.02 { scale = 1; offset = .zero }
                }
            }
        let pan = DragGesture()
            .onChanged { v in
                guard scale > 1.01 else { return }        // nothing to pan at 1×
                let base = dragBase ?? offset
                dragBase = base
                offset = Self.clamp(CGSize(width: base.width + v.translation.width,
                                           height: base.height + v.translation.height),
                                    scale: scale, in: size)
            }
            .onEnded { _ in dragBase = nil }
        let doubleTap = SpatialTapGesture(count: 2)
            .onEnded { v in
                Haptics.select()
                withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                    if scale > 1.5 {
                        scale = 1; offset = .zero
                    } else {
                        let s: CGFloat = 2.5
                        let anchor = CGPoint(x: v.location.x - size.width / 2,
                                             y: v.location.y - size.height / 2)
                        scale = s
                        offset = Self.clamp(CGSize(width: -anchor.x * (s - 1),
                                                   height: -anchor.y * (s - 1)),
                                            scale: s, in: size)
                    }
                }
            }
        return doubleTap.simultaneously(with: pinch).simultaneously(with: pan)
    }

    /// Center the zoom on `size` at a given scale, keeping the current pan
    /// direction — used by the ± buttons (two-hand magnify is awkward mid-teleop).
    private func setZoom(_ s: CGFloat, in size: CGSize) {
        let clamped = min(max(s, Self.minScale), Self.maxScale)
        withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) {
            let k = clamped / scale
            scale = clamped
            offset = Self.clamp(CGSize(width: offset.width * k, height: offset.height * k),
                                scale: clamped, in: size)
        }
    }

    /// Pan bound: the scaled frame may shift at most (scale−1)/2 of the
    /// viewport per axis, so the video can never be dragged fully off screen.
    private static func clamp(_ o: CGSize, scale s: CGFloat, in size: CGSize) -> CGSize {
        let maxX = size.width * (s - 1) / 2
        let maxY = size.height * (s - 1) / 2
        return CGSize(width: min(max(o.width, -maxX), maxX),
                      height: min(max(o.height, -maxY), maxY))
    }

    // MARK: - Chrome

    private func chrome(in size: CGSize) -> some View {
        VStack {
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Circle().fill(image == nil ? Theme.Palette.warn : Theme.Palette.live)
                        .frame(width: 6, height: 6)
                    Text(title)
                        .font(.system(size: 13, weight: .heavy, design: .rounded)).tracking(1.2)
                        .foregroundStyle(.white.opacity(0.9))
                    Text(image == nil ? "WAITING" : "LIVE")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(image == nil ? Theme.Palette.warn : Theme.Palette.live)
                }
                .padding(.horizontal, 13).padding(.vertical, 8)
                .background(.black.opacity(0.45), in: Capsule())
                Spacer()
                if scale > 1.02 {
                    Text(String(format: "%.1f×", scale))
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.9))
                        .padding(.horizontal, 10).padding(.vertical, 8)
                        .background(.black.opacity(0.45), in: Capsule())
                }
                chromeButton("minus.magnifyingglass") { setZoom(scale / 1.5, in: size) }
                chromeButton("plus.magnifyingglass") { setZoom(scale * 1.5, in: size) }
                if scale > 1.02 {
                    chromeButton("arrow.counterclockwise") { setZoom(1, in: size) }
                }
            }
            Spacer()
            Text("Pinch to zoom · drag to pan · double-tap")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.35))
                .allowsHitTesting(false)
        }
        .padding(14)
    }

    private func chromeButton(_ symbol: String, action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select(); action()
        } label: {
            Image(systemName: symbol)
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(.white.opacity(0.9))
                .frame(width: 38, height: 38)
                .background(.black.opacity(0.45), in: Circle())
                .contentShape(Circle())
                .contentShape(.hoverEffect, Circle())
        }
        .buttonStyle(.plain)
        .hoverEffect()
    }

    private var placeholder: some View {
        ZStack {
            LinearGradient(colors: [Color(hex: 0x2A2622), Color(hex: 0x14110E)],
                           startPoint: .top, endPoint: .bottom)
            VStack(spacing: 8) {
                Image(systemName: "video.slash")
                    .font(.system(size: 30, weight: .light))
                    .foregroundStyle(tint.opacity(0.5))
                Text(controller.videoMainURL(streamKey) == nil
                     ? "NO VIDEO ROUTE — connect to a robot"
                     : "WAITING FOR FRAMES…")
                    .font(.system(size: 11, weight: .bold, design: .monospaced)).tracking(1.2)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
    }
}
