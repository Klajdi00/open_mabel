//
//  MJPEGTextureStreamer.swift
//  MabelTeleop
//
//  Streams one MJPEG (multipart/x-mixed-replace) HTTP feed and decodes each
//  JPEG frame to a CGImage, delivered via a callback.
//
//  Why this exists (and why it's different from CameraView):
//  ────────────────────────────────────────────────────────
//  CameraView shows a feed in a WKWebView — a flat SwiftUI surface drawn the
//  SAME to both eyes, so it can't carry stereo. For true per-eye stereo we must
//  get the pixels INTO RealityKit as a TextureResource that a ShaderGraphMaterial
//  samples per camera index. That means decoding frames ourselves. This streamer
//  is the decode half; StereoHeadEntity is the RealityKit half (two streamers,
//  one per eye, feeding one stereo quad).
//
//  The parser is a minimal multipart/x-mixed-replace reader: it scans the byte
//  stream for JPEG SOI (FF D8) … EOI (FF D9) markers and emits each complete
//  frame. It does not rely on the multipart boundary string, so it tolerates
//  the slightly different headers different MJPEG servers emit.
//

import Foundation
import CoreGraphics
import ImageIO
import OSLog

/// Streams an MJPEG feed, emitting one decoded CGImage per frame.
/// Reconnects automatically with a fixed backoff when the stream drops.
final class MJPEGTextureStreamer: NSObject, @unchecked Sendable {

    // MARK: - Configuration

    private let label: String
    private let reconnectDelay: TimeInterval
    private let maxBuffer = 8 * 1024 * 1024     // hard cap to avoid runaway growth

    /// Called on a background queue for every decoded frame.
    private let onImage: @Sendable (CGImage) -> Void

    // MARK: - State

    private var url: URL
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var buffer = Data()
    private var running = false
    /// Guards url/session/task/buffer/running. `buffer` is appended on the
    /// session's delegate queue but cleared by stop() (main) and connect()
    /// (reconnect queue) — and every reconnect brings a NEW session with its
    /// own delegate queue while the old one's callbacks can still be in
    /// flight. Unlocked, two queues drained the same Data at once and
    /// `removeSubrange` trapped on a range the other had already removed
    /// (headset crash, 2026-10-14).
    private let lock = NSLock()

    private static let SOI = Data([0xFF, 0xD8])   // JPEG start-of-image
    private static let EOI = Data([0xFF, 0xD9])   // JPEG end-of-image

    private let log = Logger(subsystem: "com.mabel.teleop", category: "MJPEGTextureStreamer")

    // MARK: - Init

    init(url: URL,
         label: String,
         reconnectDelay: TimeInterval = 1.5,
         onImage: @escaping @Sendable (CGImage) -> Void) {
        self.url = url
        self.label = label
        self.reconnectDelay = reconnectDelay
        self.onImage = onImage
        super.init()
    }

    // MARK: - Control

    func start() {
        lock.lock(); defer { lock.unlock() }
        guard !running else { return }
        running = true
        connect()
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        running = false
        task?.cancel()
        task = nil
        session?.invalidateAndCancel()
        session = nil
        buffer.removeAll(keepingCapacity: false)
    }

    /// Point the streamer at a new URL (e.g. the user changed host in Settings).
    func updateURL(_ newURL: URL) {
        lock.lock()
        let changed = newURL != url
        url = newURL
        let wasRunning = running
        lock.unlock()
        if changed && wasRunning {
            stop()
            start()
        }
    }

    // MARK: - Connection

    /// Caller holds `lock`.
    private func connect() {
        // A session PER reconnect used to LEAK: URLSession strongly retains its
        // delegate (us), so every retry pinned a ghost session that kept pulling
        // and decoding frames until process exit — the same defect the iOS app
        // fixed in MJPEGView. Tear the old one down before building the next.
        session?.invalidateAndCancel()
        task?.cancel()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        config.waitsForConnectivity = true
        config.urlCache = nil
        let session = URLSession(configuration: config, delegate: self,
                                 delegateQueue: nil)
        self.session = session

        var request = URLRequest(url: url)
        request.setValue("multipart/x-mixed-replace", forHTTPHeaderField: "Accept")
        let task = session.dataTask(with: request)
        self.task = task
        buffer.removeAll(keepingCapacity: true)
        log.info("MJPEG \(self.label, privacy: .public) connecting → \(self.url.absoluteString, privacy: .public)")
        task.resume()
    }

    private func scheduleReconnect() {
        lock.lock()
        let shouldReconnect = running
        lock.unlock()
        guard shouldReconnect else { return }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + reconnectDelay) { [weak self] in
            guard let self else { return }
            self.lock.lock(); defer { self.lock.unlock() }
            if self.running { self.connect() }
        }
    }

    // MARK: - Frame extraction

    /// Pull complete JPEGs out of `buffer` and emit only the NEWEST one.
    ///
    /// This used to decode and deliver EVERY frame it found. Under any backlog
    /// — a slow decode, a stalled main actor, a burst after a hiccup — that
    /// plays the queue out late instead of skipping it, so latency accumulates
    /// monotonically and never recovers. Video is only ever useful live: a
    /// frame that arrived while we were busy is already worthless, so drop it.
    /// (This is drop-to-latest, the same policy the iOS MJPEG view and the
    /// H.264/JPEG UDP receiver already use. Safe here precisely BECAUSE MJPEG
    /// frames are independent — never do this to H.264 P-frames, which would
    /// corrupt the GOP.)
    /// Caller holds `lock`; returns the newest complete JPEG (decode it AFTER
    /// unlocking — decoding under the lock would stall stop()/connect()).
    private func drainFrames() -> Data? {
        var newest: Data?
        var dropped = 0
        while true {
            guard let start = buffer.range(of: Self.SOI) else {
                // No SOI yet — keep only a tail in case a marker is split.
                if buffer.count > maxBuffer { buffer.removeAll(keepingCapacity: true) }
                break
            }
            // Search for EOI strictly after the SOI.
            let searchFrom = start.lowerBound
            guard let end = buffer.range(of: Self.EOI,
                                         in: start.upperBound..<buffer.endIndex) else {
                // Incomplete frame — drop anything before SOI, wait for more bytes.
                if start.lowerBound != buffer.startIndex {
                    buffer.removeSubrange(buffer.startIndex..<searchFrom)
                }
                if buffer.count > maxBuffer { buffer.removeAll(keepingCapacity: true) }
                break
            }
            if newest != nil { dropped += 1 }
            newest = buffer.subdata(in: searchFrom..<end.upperBound)
            buffer.removeSubrange(buffer.startIndex..<end.upperBound)
        }
        if dropped > 0 {
            log.debug("MJPEG \(self.label, privacy: .public) skipped \(dropped) stale frame(s)")
        }
        return newest
    }

    private static func decode(_ jpeg: Data) -> CGImage? {
        // kCGImageSourceShouldCacheImmediately forces the JPEG to be turned into
        // a bitmap HERE, on this background delegate queue. Without it CGImage
        // is lazy and the decode lands at DRAW time — i.e. on the main thread,
        // for every frame, on every consumer of this streamer.
        let opts: [CFString: Any] = [
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldCache: true,
        ]
        guard let src = CGImageSourceCreateWithData(jpeg as CFData, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, opts as CFDictionary) else {
            return nil
        }
        return img
    }
}

// MARK: - URLSessionDataDelegate

extension MJPEGTextureStreamer: URLSessionDataDelegate {

    /// Reject non-200 before a single byte is parsed. An unwired camera answers
    /// 404 with an HTML body, which the SOI/EOI scanner simply never matches —
    /// so without this the stream "connects", yields no frames, and the tile
    /// sits on WAITING forever with nothing in the log to say why.
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.allow); return
        }
        guard http.statusCode == 200 else {
            log.warning("MJPEG \(self.label, privacy: .public) HTTP \(http.statusCode) → \(self.url.absoluteString, privacy: .public)")
            completionHandler(.cancel)      // didCompleteWithError → scheduleReconnect
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        lock.lock()
        // Bytes from a session we already replaced belong to a dead stream.
        guard running, session === self.session else { lock.unlock(); return }
        buffer.append(data)
        let jpeg = drainFrames()
        lock.unlock()
        if let jpeg, let image = Self.decode(jpeg) { onImage(image) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        lock.lock(); let current = session === self.session; lock.unlock()
        guard current else { return }       // the replaced session's farewell
        if let error, (error as NSError).code != NSURLErrorCancelled {
            log.warning("MJPEG \(self.label, privacy: .public) dropped: \(error.localizedDescription, privacy: .public)")
        }
        scheduleReconnect()
    }
}
