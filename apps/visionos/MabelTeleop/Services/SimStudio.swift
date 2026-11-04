//
//  SimStudio.swift
//  MabelTeleop (ported from MABEL Mobile App v2)
//
//  Client for the Simulation Control Center backend (:8097) — the process that
//  OWNS scene switching (a scene switch is a supervised full-stack relaunch,
//  so it can't be a teleop-bridge message). Exposes:
//    · the scene library (categories + thumbnails, /api/scenes)
//    · scene switching with the SCC's loading-progress model (/api/sim/scene
//      + /api/sim/status polling: physics → telemetry → cameras)
//    · the browser MuJoCo viewer endpoint (ws /ws/simview, JPEG-over-WS)
//  Only meaningful against a SIMULATION on a direct link; `available` stays
//  false otherwise and every sim-only control hides.
//

import Combine
import Foundation
import SwiftUI
import UIKit          // UIImage (thumbnail preload cache)

@MainActor
final class SimStudio: ObservableObject {

    struct SceneInfo: Identifiable, Decodable, Equatable {
        let name: String
        let title: String?
        var xml: String?
        var description: String?
        var category: String?
        var dynamic: Bool?
        var thumb: Bool?
        var id: String { name }
        var displayTitle: String { (title?.isEmpty == false ? title! : name).capitalized }
    }

    /// Live progress of a scene switch — drives the loading overlay. Mirrors
    /// the SCC's milestone model (stop → load → physics → telemetry → cameras).
    struct SwitchProgress: Equatable {
        var scene: SceneInfo
        var stage = "Stopping the old world"
        var pct: Double = 0.08
        var bridge = false          // physics engine answering
        var telemetry = false       // teleop websocket accepting
        var camera = false          // camera server up
        var ready = false
        var failed: String? = nil
    }

    @Published private(set) var scenes: [SceneInfo] = []
    @Published private(set) var currentScene: String?
    @Published private(set) var available = false
    @Published var switching: SwitchProgress? = nil

    /// PRELOADED scene thumbnails, keyed by scene name. Filled the moment the
    /// scene list arrives — long before the Worlds sheet is opened — so cards show
    /// instantly instead of each card firing its own AsyncImage fetch the first
    /// time it scrolls into view (the "thumbnails load really slowly" the operator
    /// saw: 35 cold requests racing as the sheet appears). The SceneLibrary card
    /// reads this first and only falls back to AsyncImage for a straggler.
    @Published private(set) var thumbnails: [String: UIImage] = [:]

    private(set) var base: URL?
    private var pollTask: Task<Void, Never>?
    private var prefetchTask: Task<Void, Never>?

    /// Point the client at the connected robot's host (nil → tear down). The
    /// studio answers only when the SCC backend runs beside the sim; a quick
    /// probe of /api/scenes decides availability.
    func configure(host: String?, isSimulation: Bool) {
        guard let host, isSimulation else {
            base = nil; available = false; scenes = []; currentScene = nil
            return
        }
        let url = URL(string: "http://\(host):8097")
        guard url != base || !available else { return }     // already configured
        base = url
        thumbnails = [:]                 // a different sim — drop the old sim's cache
        Task { await refresh() }
    }

    /// Re-probe the scene backend NOW — recovers a late SCC / transient probe miss.
    func reprobe() {
        guard base != nil else { return }
        Task { await refresh() }
    }

    func refresh() async {
        guard let base else { return }
        var req = URLRequest(url: base.appendingPathComponent("api/scenes"))
        req.timeoutInterval = 3
        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            struct Reply: Decodable {
                let scenes: [SceneInfo]
                let current: String?
            }
            let r = try JSONDecoder().decode(Reply.self, from: data)
            withAnimation(Theme.Motion.smooth) {
                scenes = r.scenes
                currentScene = r.current
                available = !r.scenes.isEmpty
            }
            prefetchThumbnails(r.scenes)      // warm the cache now, not on sheet-open
        } catch {
            available = false
        }
    }

    func thumbURL(_ name: String) -> URL? {
        base?.appendingPathComponent("api/scenes/thumb/\(name)")
    }

    /// Preload every scene thumbnail into memory as soon as the list is known.
    /// Bounded concurrency (6): 35 simultaneous fetches at sheet-open was the
    /// slowness — spread them out ahead of time instead. Already-cached names are
    /// skipped, so a reprobe is cheap. Decoding runs off the main actor; only the
    /// tiny dictionary write is hopped back.
    private func prefetchThumbnails(_ scenes: [SceneInfo]) {
        prefetchTask?.cancel()
        let want = scenes.filter { $0.thumb == true }
            .compactMap { s in thumbURL(s.name).map { (s.name, $0) } }
            .filter { thumbnails[$0.0] == nil }
        guard !want.isEmpty else { return }
        prefetchTask = Task { [weak self] in
            await withTaskGroup(of: SimStudio.ThumbResult.self) { group in
                var it = want.makeIterator()
                var inFlight = 0
                func pump() {
                    while inFlight < 6, let (name, url) = it.next() {
                        inFlight += 1
                        group.addTask { await SimStudio.fetchThumb(name: name, url: url) }
                    }
                }
                pump()
                while let r = await group.next() {
                    inFlight -= 1
                    if let img = r.image { self?.thumbnails[r.name] = img }   // @MainActor write
                    if Task.isCancelled { break }
                    pump()
                }
            }
        }
    }

    private struct ThumbResult { let name: String; let image: UIImage? }

    /// One thumbnail fetch + decode, off the main actor.
    nonisolated private static func fetchThumb(name: String, url: URL) async -> ThumbResult {
        var req = URLRequest(url: url)
        req.timeoutInterval = 6
        guard let (d, _) = try? await URLSession.shared.data(for: req),
              let img = UIImage(data: d) else { return ThumbResult(name: name, image: nil) }
        return ThumbResult(name: name, image: img)
    }

    /// The MuJoCo viewer stream (JPEG frames over a WebSocket).
    var viewerWS: URL? {
        guard available, let host = base?.host else { return nil }
        return URL(string: "ws://\(host):8097/ws/simview")
    }

    // MARK: - Scene switch (POST + status polling, SCC milestone parity)

    /// Installed by `RobotController`: a scene switch relocates the robot into a
    /// whole new world (and relaunches the sim stack under it), so whole-body
    /// teleop has to stop with it — same rule as Home and Reset. The hook exists
    /// because the scene UI holds only this studio, never the controller, and
    /// plumbing one through every sheet to say one thing is worse than a hook.
    var onWillSwitchScene: (() -> Void)?

    func switchScene(_ scene: SceneInfo) {
        guard let base, switching == nil else { return }
        onWillSwitchScene?()
        switching = SwitchProgress(scene: scene)
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            guard let self else { return }
            var req = URLRequest(url: base.appendingPathComponent("api/sim/scene"))
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: ["name": scene.name])
            req.timeoutInterval = 10
            do { _ = try await URLSession.shared.data(for: req) }
            catch {
                self.switching?.failed = "The studio refused the switch — is the sim managed by run.sh?"
                return
            }
            await self.pollStatus(deadline: Date().addingTimeInterval(60))
        }
    }

    private func pollStatus(deadline: Date) async {
        guard let base else { return }
        var sawDown = false
        var stableHits = 0
        let t0 = Date()
        while !Task.isCancelled, Date() < deadline {
            try? await Task.sleep(nanoseconds: 700_000_000)
            var req = URLRequest(url: base.appendingPathComponent("api/sim/status"))
            req.timeoutInterval = 2
            guard var p = switching else { return }
            do {
                let (data, _) = try await URLSession.shared.data(for: req)
                let d = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
                let bridge = d["bridgeUp"] as? Bool ?? false
                let telem = d["bridgeConnected"] as? Bool ?? false
                let cam = d["cameraUp"] as? Bool ?? false
                let elapsed = Date().timeIntervalSince(t0)
                if !bridge && !sawDown { sawDown = true }
                if !sawDown && elapsed > 2.5 { sawDown = true }   // old stack gone
                p.bridge = bridge && sawDown
                p.telemetry = p.bridge && telem
                p.camera = p.telemetry && cam
                // Milestones (SCC parity): the bar never over-claims.
                if p.camera { p.pct = max(p.pct, 0.92); p.stage = "Cameras online" }
                else if p.telemetry { p.pct = max(p.pct, 0.75); p.stage = "Reconnecting telemetry" }
                else if p.bridge { p.pct = max(p.pct, 0.55); p.stage = "Physics engine up" }
                else if sawDown { p.pct = max(p.pct, 0.22); p.stage = "Loading the MuJoCo world" }
                let coreReady = p.bridge && p.telemetry
                let full = coreReady && p.camera
                if full || (coreReady && elapsed > 22) {
                    stableHits += 1
                    if stableHits >= 2 {
                        p.pct = 1.0; p.ready = true; p.stage = "World ready"
                        switching = p
                        try? await Task.sleep(nanoseconds: 1_400_000_000)
                        withAnimation(Theme.Motion.smooth) { switching = nil }
                        await refresh()
                        return
                    }
                } else {
                    stableHits = 0
                }
                switching = p
            } catch {
                // backend briefly unreachable while the stack bounces — expected
                if var p2 = switching, !sawDown {
                    sawDown = true
                    p2.pct = max(p2.pct, 0.22); p2.stage = "Loading the MuJoCo world"
                    switching = p2
                }
            }
        }
        if var p = switching, !p.ready {
            p.failed = "Timed out waiting for the new world — check the studio."
            switching = p
        }
    }

    func dismissSwitch() {
        pollTask?.cancel()
        withAnimation(Theme.Motion.smooth) { switching = nil }
    }
}
