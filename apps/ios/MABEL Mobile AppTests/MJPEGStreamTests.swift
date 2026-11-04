//
//  MJPEGStreamTests.swift
//  MABEL Mobile AppTests
//
//  Live-network regression tests for the MJPEG camera client (MJPEGView.Coordinator)
//  against a running bridge (server/start_teleop.sh → camera server on :8080). The
//  iOS Simulator shares the Mac's network, so 127.0.0.1 reaches the sim bridge.
//
//  These pin the 2026-07-08 "video works on connect, later drops" fixes:
//    1. STORM: a stream-URL change (a LAN→Tailscale failover rewrites the URL) used to
//       schedule a retry from the CANCELLED task's completion, which killed the
//       healthy replacement task every ~1 s forever — video flapped live/dead.
//    2. GHOST: URLSession strongly retains its delegate, and nothing invalidated
//       the session — a dismantled tile kept streaming + decoding forever.
//
//  Like LiveConnectionTests, every test SKIPS cleanly when no server is running.
//

import Testing
import Foundation
import UIKit
@testable import MABEL_Mobile_App

@MainActor
struct MJPEGStreamTests {

    static let base = "http://127.0.0.1:8080"

    /// True iff the local camera server answers — gates the live assertions.
    private func serverUp() async -> Bool {
        guard let url = URL(string: Self.base + "/") else { return false }
        var req = URLRequest(url: url); req.timeoutInterval = 2
        return (try? await URLSession.shared.data(for: req)) != nil
    }

    /// Collects onLive transitions with timestamps; the coordinator de-dupes, so
    /// every entry is a real live↔dead flip.
    private final class LiveLog: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [(Date, Bool)] = []
        func append(_ v: Bool) { lock.lock(); events.append((Date(), v)); lock.unlock() }
        var all: [(Date, Bool)] { lock.lock(); defer { lock.unlock() }; return events }
    }

    /// A healthy stream goes live quickly and STAYS live — no dead/live flapping
    /// (the storm bug's signature) over a sustained window.
    @Test func streamStaysLive() async throws {
        guard await serverUp() else { return }                    // no server → skip
        let log = LiveLog()
        let view = UIImageView()
        let coord = MJPEGView.Coordinator(onLive: { log.append($0) })
        coord.imageView = view
        coord.start(url: URL(string: Self.base + "/camera/main/stream.mjpg")!)
        defer { coord.stop() }

        try await Task.sleep(for: .seconds(5))                    // connect + first frames
        #expect(log.all.first?.1 == true, "stream should go live")
        var frames = 0
        var last: UIImage? = view.image
        for _ in 0..<20 {                                         // 10 s @ 0.5 s polls
            try await Task.sleep(for: .milliseconds(500))
            if view.image !== last { frames += 1; last = view.image }
        }
        #expect(frames >= 5, "frames should keep arriving (got \(frames) changes)")
        let deads = log.all.filter { !$0.1 }
        #expect(deads.isEmpty, "healthy stream must never flap dead: \(deads)")
    }

    /// STORM regression: changing the URL mid-stream (what a route failover does)
    /// must hand over cleanly — no live→dead flap, frames keep flowing. The old
    /// code entered a permanent 1 Hz cancel/reconnect storm here.
    @Test func urlChangeHandsOverWithoutStorm() async throws {
        guard await serverUp() else { return }
        let log = LiveLog()
        let view = UIImageView()
        let coord = MJPEGView.Coordinator(onLive: { log.append($0) })
        coord.imageView = view
        coord.start(url: URL(string: Self.base + "/camera/main/stream.mjpg")!)
        defer { coord.stop() }
        try await Task.sleep(for: .seconds(4))
        guard log.all.first?.1 == true else { return }            // stream never came up → skip

        let switchTime = Date()
        // Same stream, different query — the shape of a failover URL rewrite.
        coord.start(url: URL(string: Self.base + "/camera/main/stream.mjpg?q=50")!)
        try await Task.sleep(for: .seconds(12))

        let flapsAfterSwitch = log.all.filter { $0.0 > switchTime.addingTimeInterval(3) && !$0.1 }
        #expect(flapsAfterSwitch.isEmpty,
                "URL handover must not flap dead (storm signature): \(flapsAfterSwitch)")
        var frames = 0
        var last: UIImage? = view.image
        for _ in 0..<10 {
            try await Task.sleep(for: .milliseconds(500))
            if view.image !== last { frames += 1; last = view.image }
        }
        #expect(frames >= 3, "frames should still flow after the handover (got \(frames))")
    }

    /// GHOST regression: after stop() the coordinator must actually deallocate.
    /// The old code never invalidated the URLSession, whose strong delegate ref
    /// pinned the coordinator (and its socket + per-frame JPEG decode) forever.
    @Test func coordinatorDeallocatesAfterStop() async throws {
        guard await serverUp() else { return }
        weak var leaked: MJPEGView.Coordinator?
        do {
            let coord = MJPEGView.Coordinator(onLive: nil)
            leaked = coord
            coord.start(url: URL(string: Self.base + "/camera/main/stream.mjpg")!)
            try await Task.sleep(for: .seconds(3))                // stream for a bit
            coord.stop()
        }
        // Session invalidation releases the delegate asynchronously.
        for _ in 0..<20 where leaked != nil {
            try await Task.sleep(for: .milliseconds(250))
        }
        #expect(leaked == nil, "stopped coordinator must deallocate (ghost-stream leak)")
    }
}
