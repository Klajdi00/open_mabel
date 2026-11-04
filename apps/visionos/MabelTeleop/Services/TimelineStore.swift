//
//  TimelineStore.swift
//  MabelTeleop (visionOS)
//
//  Where choreographies live. A timeline is a DOCUMENT — you should be able to
//  start one on a train, put the phone away, and find it again — so the editor
//  autosaves the open one on every edit and keeps a named library beside it.
//
//  The studio persists to `localStorage` under `mabel.cc.choreo.v1`; this is the
//  same idea against Application Support, with the whole `Timeline` (blocks,
//  annotations, keyframes, rows) as one JSON file per document.
//

import Combine
import Foundation

@MainActor
final class TimelineStore: ObservableObject {
    /// Saved documents, newest first.
    @Published private(set) var saved: [Timeline] = []

    private let fm = FileManager.default

    private var root: URL {
        let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = base.appendingPathComponent("Choreography", isDirectory: true)
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }
    private var openFile: URL { root.appendingPathComponent("open.json") }
    private func file(_ id: UUID) -> URL { root.appendingPathComponent("\(id.uuidString).json") }

    init() { reload() }

    // MARK: - the open document

    /// The timeline the editor had open last, or a fresh one.
    func restoreOpen() -> Timeline {
        guard let data = try? Data(contentsOf: openFile),
              let tl = try? JSONDecoder().decode(Timeline.self, from: data) else {
            return Timeline(name: "Untitled")
        }
        return tl
    }

    /// Autosave. Cheap enough to call on every edit (one small JSON write), and
    /// debounced by the caller's `onChange`.
    func persistOpen(_ timeline: Timeline) {
        guard let data = try? JSONEncoder().encode(timeline) else { return }
        try? data.write(to: openFile, options: .atomic)
    }

    // MARK: - the library

    func reload() {
        let urls = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        saved = urls
            .filter { $0.lastPathComponent != "open.json" && $0.pathExtension == "json" }
            .compactMap { try? JSONDecoder().decode(Timeline.self, from: Data(contentsOf: $0)) }
            .sorted { $0.modified > $1.modified }
    }

    /// Save under a name. Saving twice under one id overwrites — "Save" on an
    /// open document, not a second copy of it.
    func save(_ timeline: Timeline, as name: String) {
        var copy = timeline
        copy.name = name.isEmpty ? "Untitled" : name
        copy.modified = Date()
        guard let data = try? JSONEncoder().encode(copy) else { return }
        try? data.write(to: file(copy.id), options: .atomic)
        reload()
    }

    func delete(_ timeline: Timeline) {
        try? fm.removeItem(at: file(timeline.id))
        reload()
    }

    /// A copy with fresh ids, so opening a saved timeline and editing it does
    /// not silently rewrite the original until you save it again.
    func duplicate(_ timeline: Timeline) -> Timeline {
        var copy = timeline
        copy.id = UUID()
        copy.name = timeline.name + " copy"
        copy.modified = Date()
        return copy
    }
}
