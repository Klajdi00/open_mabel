//
//  CameraFeedView.swift
//  MabelTeleop
//
//  A labelled live MJPEG camera tile — the visionOS twin of the iOS app's
//  CameraFeed. Shows a synthetic placeholder grid when no stream URL is
//  available (demo mode), so the layout always reads correctly.
//

import SwiftUI
import UIKit

struct CameraFeedView: View {
    let name: String
    var compact: Bool = false
    var tint: Color = Theme.Palette.accent
    var streamURL: URL?
    /// Low-latency UDP video (preferred when set): robot host + stream key
    /// ("main", "wrist_left", …). `streamURL` stays the MJPEG fallback.
    var udpHost: String?
    var udpPort: UInt16 = 9091
    /// LAN UDP codec (H.264 :9092 / JPEG :9091).
    var codec: VideoCodec = .jpeg
    var stream: String?
    /// Tap-to-expand: when set, tapping the tile opens the camera-detail window
    /// for this stream (zoom + pan). Shows an expand glyph + gaze hover so the
    /// affordance is discoverable. nil = inert tile.
    var onExpand: (() -> Void)? = nil

    @State private var image: UIImage?
    @State private var feed: VideoFeed?

    var body: some View {
        // Tap gesture attached ONLY when expandable, so inert tiles never
        // swallow touches meant for what's underneath them.
        if let onExpand {
            let shape = RoundedRectangle(cornerRadius: compact ? 10 : Theme.Radius.md,
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
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                placeholder
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: compact ? 10 : Theme.Radius.md, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: compact ? 10 : Theme.Radius.md, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        }
        .overlay(alignment: .topLeading) { label }
        .overlay(alignment: .bottomTrailing) {
            if onExpand != nil {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: compact ? 7 : 10, weight: .bold))
                    .foregroundStyle(.white.opacity(0.65))
                    .padding(compact ? 3 : 5)
                    .background(.black.opacity(0.4), in: Circle())
                    .padding(compact ? 4 : 8)
                    .allowsHitTesting(false)
            }
        }
        .onAppear { start() }
        .onDisappear { stop() }
        .onChange(of: streamURL) { _, _ in stop(); start() }
    }

    private var label: some View {
        HStack(spacing: 4) {
            Circle().fill(image == nil ? Theme.Palette.warn : Theme.Palette.live)
                .frame(width: 5, height: 5)
            Text(name)
                .font(.system(size: compact ? 8 : 10, weight: .bold)).tracking(0.8)
                .foregroundStyle(.white.opacity(0.85))
        }
        .padding(.horizontal, compact ? 6 : 8).padding(.vertical, compact ? 3 : 5)
        .background(.black.opacity(0.45), in: Capsule())
        .padding(compact ? 4 : 8)
    }

    private var placeholder: some View {
        ZStack {
            LinearGradient(colors: [Color(hex: 0x2A2622), Color(hex: 0x14110E)],
                           startPoint: .top, endPoint: .bottom)
            Image(systemName: "video.slash")
                .font(.system(size: compact ? 14 : 26, weight: .light))
                .foregroundStyle(tint.opacity(0.5))
        }
    }

    private func start() {
        let canStream = streamURL != nil
            || (stream != nil && udpHost != nil)
        guard feed == nil, canStream else { return }
        let f = VideoFeed(host: stream != nil ? udpHost : nil, udpPort: udpPort,
                          codec: codec, stream: stream ?? "", mjpegURL: streamURL,
                          label: name) { cg in
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
}
