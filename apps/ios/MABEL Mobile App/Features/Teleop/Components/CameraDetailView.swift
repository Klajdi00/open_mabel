//
//  CameraDetailView.swift
//  MABEL Mobile App
//
//  Full-screen live inspector for ONE camera stream — presented when the
//  operator taps any camera tile (console monitor, FPV wrist thumbnails).
//  Shows the FULL-QUALITY live feed (the `videoURL` MJPEG transport) with pinch-to-zoom, drag-to-pan and double-tap
//  zoom, so fine detail (a grasp, a label, a far obstacle) can be inspected
//  without leaving the cockpit.
//

import SwiftUI

/// A camera tile the operator expanded — identity for the full-screen cover.
struct ExpandedCamera: Identifiable, Equatable {
    let key: String          // wire stream key ("main" | "wrist_left" | …)
    let name: String         // display label ("HEAD · POV")
    let tint: Color
    var id: String { key }
}

struct CameraDetailView: View {
    let camera: ExpandedCamera
    /// Plain reference, NOT observed: the URL is re-resolved each body eval and
    /// `.id(url)` restarts the stream if the route fails over mid-inspection.
    let controller: RobotController
    @Environment(\.dismiss) private var dismiss

    /// Optimistic like the tiles: never flash OFFLINE while the first frame
    /// is in flight.
    @State private var live = true

    // Committed zoom/pan. Gesture deltas apply on top of a base frozen at
    // gesture start (MagnifyGesture/DragGesture report totals, not deltas).
    @State private var scale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var pinchBase: (scale: CGFloat, offset: CGSize)? = nil
    @State private var dragBase: CGSize? = nil

    private static let minScale: CGFloat = 1
    private static let maxScale: CGFloat = 8

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.ignoresSafeArea()
                if let url = controller.videoURL(camera.key) {
                    MJPEGView(url: url, onLive: { live = $0 },
                              streamKey: camera.key, contentMode: .scaleAspectFit)
                        .id(url)                    // route failover restarts the stream
                        .frame(width: geo.size.width, height: geo.size.height)
                        .scaleEffect(scale)
                        .offset(offset)
                        .clipped()
                    if !live { offlineBadge }
                } else {
                    noRoute
                }
                chrome(in: geo.size)
            }
            .contentShape(Rectangle())
            .gesture(zoomGestures(in: geo.size))
        }
        // The backdrop bleeds full-screen; the video + chrome stay inside the
        // safe area (the header pill must not ride under the status bar).
        .background(Color.black.ignoresSafeArea())
        #if DEBUG
        // Screenshot harness: `defaults write … debugCameraZoom -float 2.5`
        // boots at that magnification — pinch/drag can't be driven by
        // synthetic events on the Simulator.
        .onAppear {
            let z = UserDefaults.standard.double(forKey: "debugCameraZoom")
            if z > 1 { scale = min(CGFloat(z), Self.maxScale) }
        }
        #endif
    }

    // MARK: - Gestures (pinch anchored at the fingers, pan clamped to frame)

    private func zoomGestures(in size: CGSize) -> some Gesture {
        let pinch = MagnifyGesture()
            .onChanged { v in
                let base = pinchBase ?? (scale, offset)
                pinchBase = base
                let s = min(max(base.scale * v.magnification, Self.minScale), Self.maxScale)
                // Keep the pinch's start point fixed on screen while scaling:
                // q = (P − offset)/scale must land back on P at the new scale.
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
                    Circle().fill(live ? Theme.Palette.live : Theme.Palette.danger)
                        .frame(width: 6, height: 6)
                    Text(camera.name)
                        .font(.system(size: 12, weight: .heavy, design: .rounded)).tracking(1.2)
                        .foregroundStyle(Theme.Palette.textPrimary)
                    Text(live ? "LIVE" : "OFFLINE")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(live ? Theme.Palette.live : Theme.Palette.danger)
                }
                .padding(.horizontal, 12).padding(.vertical, 7)
                .background(.black.opacity(0.45), in: Capsule())
                Spacer()
                if scale > 1.02 {
                    Text(String(format: "%.1f×", scale))
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textPrimary)
                        .padding(.horizontal, 10).padding(.vertical, 7)
                        .background(.black.opacity(0.45), in: Capsule())
                    chromeButton("arrow.counterclockwise") {
                        withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                            scale = 1; offset = .zero
                        }
                    }
                }
                chromeButton("xmark") { dismiss() }
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            Spacer()
            Text("Pinch to zoom · drag to pan · double-tap")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white.opacity(0.35))
                .padding(.bottom, 14)
                .allowsHitTesting(false)
        }
    }

    private func chromeButton(_ symbol: String, action: @escaping () -> Void) -> some View {
        Button {
            Haptics.select(); action()
        } label: {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(.white.opacity(0.9))
                .frame(width: 34, height: 34)
                .background(.black.opacity(0.45), in: Circle())
                .contentShape(Circle())
        }
    }

    // MARK: - Fault states

    private var offlineBadge: some View {
        VStack(spacing: 6) {
            Image(systemName: "video.slash.fill")
                .font(.system(size: 26, weight: .regular))
            Text("NOT CONNECTED")
                .font(.system(size: 12, weight: .bold, design: .monospaced)).tracking(1.4)
        }
        .foregroundStyle(.white.opacity(0.55))
        .allowsHitTesting(false)
    }

    private var noRoute: some View {
        VStack(spacing: 6) {
            Image(systemName: "video.slash.fill")
                .font(.system(size: 26, weight: .regular))
            Text("NO VIDEO ROUTE")
                .font(.system(size: 12, weight: .bold, design: .monospaced)).tracking(1.4)
            Text("Connect to a robot to stream this camera.")
                .font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(.white.opacity(0.55))
    }
}
