//
//  UDPCameraView.swift
//  MABEL Mobile App
//
//  Low-latency camera tile: H.264 over UDP (VideoToolbox) with automatic MJPEG
//  fallback. Prefers the bridge's :9092 H.264/UDP stream; if no datagrams arrive
//  within 2.5 s (e.g. a UDP-blocked network) it falls back to the
//  existing MJPEG/HTTP stream so video ALWAYS works — UDP is purely an upgrade.
//
//  Drop-in replacement for MJPEGView when a LAN/Tailscale `udpHost` is known. Both
//  paths produce a CGImage rendered into one UIImageView (drop-to-latest), so the
//  switch is invisible to the rest of the UI.
//

import SwiftUI
import UIKit
import CoreGraphics
import ImageIO

struct UDPCameraView: UIViewRepresentable {
    var udpHost: String? = nil   // robot host for the UDP H.264 port (LAN/Tailscale only)
    var udpPort: UInt16 = 9092   // 9092 = H.264
    let stream: String           // "main" | "wrist_left" | …
    let mjpegURL: URL?           // fallback (and only path if the primary is unreachable)
    var onLive: ((Bool) -> Void)? = nil

    func makeUIView(context: Context) -> UIImageView {
        let v = UIImageView()
        v.contentMode = .scaleAspectFill
        v.clipsToBounds = true
        v.backgroundColor = .black
        // The DECODED frame's resolution must NOT drive layout — otherwise a tile
        // sized by SwiftUI (e.g. an 80×50 wrist thumbnail, or the full-bleed POV)
        // blows up to the native H.264 size and overflows the window. Drop the
        // intrinsic content-size priorities to minimum so the SwiftUI `.frame()`
        // wins — same fix MJPEGView already carries.
        v.setContentHuggingPriority(.init(1), for: .horizontal)
        v.setContentHuggingPriority(.init(1), for: .vertical)
        v.setContentCompressionResistancePriority(.init(1), for: .horizontal)
        v.setContentCompressionResistancePriority(.init(1), for: .vertical)
        context.coordinator.attach(to: v)
        return v
    }

    func updateUIView(_ uiView: UIImageView, context: Context) {}

    /// Accept the size SwiftUI proposes (ignore the decoded image's intrinsic
    /// resolution) so `.frame()` actually constrains the tile — mirrors MJPEGView.
    func sizeThatFits(_ proposal: ProposedViewSize, uiView: UIImageView, context: Context) -> CGSize? {
        proposal.replacingUnspecifiedDimensions()
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(udpHost: udpHost, udpPort: udpPort, stream: stream,
                    mjpegURL: mjpegURL, onLive: onLive)
    }

    static func dismantleUIView(_ uiView: UIImageView, coordinator: Coordinator) {
        coordinator.stop()
    }

    // MARK: - Coordinator: owns the receiver + decoder + MJPEG fallback

    final class Coordinator: NSObject, URLSessionDataDelegate {
        private let udpHost: String?
        private let udpPort: UInt16
        private let stream: String
        private let mjpegURL: URL?
        private let onLive: ((Bool) -> Void)?

        private weak var imageView: UIImageView?
        private var receiver: UDPVideoReceiver?
        private var token: UUID?
        private var decoder: H264Decoder?

        private var lastLive = Date.distantPast
        private var watchdog: Timer?
        private var mjpegTask: URLSessionDataTask?
        private var mjpegSession: URLSession?
        private var mjpegBuffer = Data()
        private var live = false

        init(udpHost: String?, udpPort: UInt16, stream: String,
             mjpegURL: URL?, onLive: ((Bool) -> Void)?) {
            self.udpHost = udpHost; self.udpPort = udpPort
            self.stream = stream; self.mjpegURL = mjpegURL
            self.onLive = onLive
            super.init()
        }

        func attach(to view: UIImageView) {
            imageView = view
            let dec = H264Decoder { [weak self] cg in self?.show(cg, fromLive: true) }
            decoder = dec
            if let host = udpHost {                       // LAN/Tailscale: H.264 over UDP
                let r = UDPVideoReceiver.shared(host: host, port: udpPort)
                receiver = r
                token = r.addHandler(stream: stream) { [weak self] au, lost in
                    guard let dec = self?.decoder else { return }
                    if lost { dec.noteLoss() }            // resync on the next IDR, no smear
                    dec.decode(accessUnit: au)
                }
            }
            // Watchdog: start/keep MJPEG fallback whenever the UDP primary is stale.
            let t = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
                self?.checkFallback()
            }
            watchdog = t
            // Give the primary a head start; if nothing in 2.5 s the watchdog brings up MJPEG.
        }

        func stop() {
            watchdog?.invalidate(); watchdog = nil
            if let r = receiver, let token { r.removeHandler(stream: stream, token: token) }
            receiver = nil; token = nil; decoder = nil
            stopMJPEG()
        }

        // MARK: render (drop-to-latest, main thread)

        private func show(_ cg: CGImage, fromLive: Bool) {
            if fromLive { lastLive = Date() }
            let img = UIImage(cgImage: cg)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.imageView?.image = img
                if !self.live { self.live = true; self.onLive?(true) }
            }
        }

        // MARK: MJPEG fallback

        private func checkFallback() {
            let primaryFresh = Date().timeIntervalSince(lastLive) < 2.5
            if primaryFresh {
                if mjpegTask != nil { stopMJPEG() }          // primary recovered → drop MJPEG
            } else if mjpegTask == nil {
                startMJPEG()                                  // no primary frames → fall back
            }
            if !primaryFresh && mjpegTask == nil && live {
                live = false; onLive?(false)
            }
        }

        private func startMJPEG() {
            guard let url = mjpegURL else { return }
            mjpegBuffer.removeAll(keepingCapacity: true)
            let cfg = URLSessionConfiguration.default
            cfg.timeoutIntervalForRequest = 10
            let s = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
            mjpegSession = s
            mjpegTask = s.dataTask(with: url)
            mjpegTask?.resume()
        }

        private func stopMJPEG() {
            mjpegTask?.cancel(); mjpegTask = nil
            mjpegSession?.invalidateAndCancel(); mjpegSession = nil
            mjpegBuffer.removeAll(keepingCapacity: false)
        }

        /// The fallback stream ended (10 s timeout on a dead stream, server restart,
        /// network blip). Clear `mjpegTask` so the 1.5 s watchdog can REOPEN it —
        /// before this, the dead task stayed assigned, `checkFallback` saw
        /// "fallback already running" forever, and video never came back
        /// after a single drop until the tile was rebuilt.
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        didCompleteWithError error: Error?) {
            DispatchQueue.main.async { [weak self] in
                guard let self, task === self.mjpegTask else { return }
                self.stopMJPEG()
            }
        }

        // multipart/x-mixed-replace → scan SOI..EOI, keep only the latest frame.
        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive data: Data) {
            mjpegBuffer.append(data)
            var latest: CGImage?
            while true {
                guard let soi = mjpegBuffer.range(of: Data([0xFF, 0xD8])),
                      let eoi = mjpegBuffer.range(of: Data([0xFF, 0xD9]),
                                                  in: soi.upperBound..<mjpegBuffer.endIndex)
                else { break }
                let jpeg = mjpegBuffer.subdata(in: soi.lowerBound..<eoi.upperBound)
                mjpegBuffer.removeSubrange(mjpegBuffer.startIndex..<eoi.upperBound)
                if let src = CGImageSourceCreateWithData(jpeg as CFData, nil),
                   let img = CGImageSourceCreateImageAtIndex(src, 0, nil) { latest = img }
            }
            if mjpegBuffer.count > 4_000_000 { mjpegBuffer.removeAll(keepingCapacity: true) }
            if let latest { show(latest, fromLive: false) }
        }
    }
}
