//
//  MJPEGView.swift
//  MABEL Mobile App
//
//  Streams an MJPEG (multipart/x-mixed-replace) HTTP feed from the MABEL
//  camera server (http://<host>:8080/camera/<name>/stream.mjpg) and shows
//  the decoded frames — the same streams the Vision Pro app pulls.
//

import SwiftUI

struct MJPEGView: UIViewRepresentable {
    let url: URL
    /// Reports live/dead: true once real JPEG frames arrive, false on a 404 /
    /// stream error (e.g. a camera that isn't wired) so the tile can show
    /// "NOT CONNECTED" instead of a frozen/blank image.
    var onLive: ((Bool) -> Void)? = nil
    /// Wire stream key ("main", …): when set, decoded frame sizes are published
    /// to VideoStreamAspect so 3D geometry can match the actual stream aspect.
    var streamKey: String? = nil
    /// Tiles crop-to-fill; the full-screen inspector letterboxes (.scaleAspectFit)
    /// so the whole frame is visible before zooming.
    var contentMode: UIView.ContentMode = .scaleAspectFill

    func makeUIView(context: Context) -> UIImageView {
        let v = UIImageView()
        v.contentMode = contentMode
        v.clipsToBounds = true
        v.backgroundColor = UIColor(white: 0.03, alpha: 1)
        // The image's resolution must NOT drive layout — otherwise a small wrist
        // tile (.frame 40×26) blows up to the 640×480 stream size. Let the SwiftUI
        // frame win by dropping the intrinsic content-size priorities to minimum.
        v.setContentHuggingPriority(.init(1), for: .horizontal)
        v.setContentHuggingPriority(.init(1), for: .vertical)
        v.setContentCompressionResistancePriority(.init(1), for: .horizontal)
        v.setContentCompressionResistancePriority(.init(1), for: .vertical)
        context.coordinator.imageView = v
        context.coordinator.streamKey = streamKey
        context.coordinator.start(url: url)
        return v
    }

    func updateUIView(_ uiView: UIImageView, context: Context) {
        uiView.contentMode = contentMode
        context.coordinator.onLive = onLive
        context.coordinator.streamKey = streamKey
        if context.coordinator.url != url { context.coordinator.start(url: url) }
    }

    /// Accept whatever size the SwiftUI frame proposes (ignore the image's
    /// intrinsic resolution), so .frame() actually constrains the tile.
    func sizeThatFits(_ proposal: ProposedViewSize, uiView: UIImageView, context: Context) -> CGSize? {
        proposal.replacingUnspecifiedDimensions()
    }

    func makeCoordinator() -> Coordinator { Coordinator(onLive: onLive) }

    /// SwiftUI recreates this view's identity on layout changes (rotation, tab
    /// switches, LAN↔Tailscale route changes). Without an explicit stop, the coordinator's
    /// URLSession — which retains its delegate (the coordinator) strongly — kept the
    /// stream alive FOREVER after teardown: a ghost connection per rotation, each
    /// still pulling + JPEG-decoding every frame. Enough of them saturated the
    /// downlink/CPU and the visible tiles starved ("video worked, then died").
    static func dismantleUIView(_ uiView: UIImageView, coordinator: Coordinator) {
        coordinator.stop()
    }

    final class Coordinator: NSObject, URLSessionDataDelegate {
        weak var imageView: UIImageView?
        private(set) var url: URL?
        /// ONE session for the coordinator's lifetime (invalidated in `stop()`).
        /// A session PER restart leaked: URLSession strongly retains its delegate,
        /// so every retry pinned the coordinator + its socket until process exit.
        private lazy var session: URLSession = {
            let cfg = URLSessionConfiguration.default
            cfg.timeoutIntervalForRequest = 10         // MJPEG: fires after 10 s of NO data
            cfg.waitsForConnectivity = true            // ride out brief network gaps
            return URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
        }()
        private var task: URLSessionDataTask?
        private var buffer = Data()
        var onLive: ((Bool) -> Void)?
        var streamKey: String?
        private var lastLive: Bool? = nil

        init(onLive: ((Bool) -> Void)?) { self.onLive = onLive; super.init() }

        /// Report live/dead to SwiftUI on the main actor, de-duped.
        private func setLive(_ v: Bool) {
            guard lastLive != v else { return }
            lastLive = v
            DispatchQueue.main.async { [weak self] in self?.onLive?(v) }
        }

        /// Reject a non-200 (e.g. 404 for an unwired camera) -> mark dead.
        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse,
                        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                setLive(false)
                completionHandler(.cancel)
                return
            }
            completionHandler(.allow)
        }

        // JPEG SOI / EOI markers.
        private let soi = Data([0xFF, 0xD8])
        private let eoi = Data([0xFF, 0xD9])

        private var stopped = false

        func start(url: URL) {
            self.url = url
            stopped = false
            task?.cancel(); buffer.removeAll(keepingCapacity: true)
            let t = session.dataTask(with: url)
            task = t
            t.resume()
        }

        /// Terminal: invalidating the session releases its strong delegate ref
        /// (self), so the coordinator can actually deinit after dismantle.
        func stop() {
            stopped = true
            task?.cancel(); task = nil
            session.invalidateAndCancel()
        }

        // If the stream ends or errors (network blip, server restart), restart
        // it automatically so video recovers on its own under any network.
        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            // ONLY the current task may drive the retry loop. Without this guard,
            // a task cancelled by start() (e.g. the stream URL changed on a
            // LAN→Tailscale failover) ALSO scheduled a restart — which cancelled the
            // healthy new task, whose cancellation scheduled another restart…
            // a self-sustaining 1 Hz kill/reconnect storm that flashed
            // "NOT CONNECTED" and hammered the camera server indefinitely.
            guard task === self.task, !stopped, let url else { return }
            setLive(false)                              // stream dropped -> not connected
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                guard let self, !self.stopped, self.url == url else { return }
                self.start(url: url)
            }
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            guard dataTask === task else { return }     // stale (cancelled) task data
            buffer.append(data)
            // DROP-TO-LATEST: for live video only the freshest frame matters.
            // If we ever receive faster than we can display (the server runs at
            // 60 fps; the phone may not), decoding every buffered JPEG just piles
            // up latency. Walk all complete frames but keep only the LAST, decode
            // that one, and discard the rest — so a momentary backlog is skipped
            // instead of played out late.
            var latest: Data? = nil
            while let start = buffer.range(of: soi),
                  let end = buffer.range(of: eoi, in: start.upperBound..<buffer.endIndex) {
                latest = buffer.subdata(in: start.lowerBound..<end.upperBound)
                buffer.removeSubrange(buffer.startIndex..<end.upperBound)
            }
            if let frame = latest, let raw = UIImage(data: frame) {
                setLive(true)                           // real frames -> connected
                if let k = streamKey { VideoStreamAspect.set(k, raw.size) }
                // Force the JPEG DECODE here, on this background delegate queue —
                // `UIImage(data:)` is lazy and would otherwise decode on the MAIN thread
                // at display time. At 20+ fps of 960×540 frames that saturated main,
                // backing up the WebSocket receive/pong path (latency spiked to ~400 ms)
                // AND making the video itself stutter. preparingForDisplay() returns an
                // already-bitmapped image so the main thread only does a cheap assign.
                let img = raw.preparingForDisplay() ?? raw
                DispatchQueue.main.async { self.imageView?.image = img }
            }
            // Guard against unbounded growth if markers desync.
            if buffer.count > 4_000_000 { buffer.removeAll(keepingCapacity: true) }
        }

        deinit { task?.cancel() }
    }
}
