//
//  MabelStatusWidget.swift
//  MabelWidgetExtension
//
//  The widget definition + its timeline provider. The provider reads the
//  snapshot the app wrote into the shared App Group container (telemetry JSON +
//  a rendered pose PNG) and hands it to the views. The app calls
//  WidgetCenter.reloadAllTimelines() whenever state changes, so the widget is
//  as live as iOS allows; the periodic refresh is just a safety net.
//

import WidgetKit
import SwiftUI

// MARK: - Timeline entry

struct MabelEntry: TimelineEntry {
    let date: Date
    let snapshot: MabelWidgetSnapshot
    let poseImage: UIImage?
}

// MARK: - Provider

struct MabelProvider: TimelineProvider {
    func placeholder(in context: Context) -> MabelEntry {
        MabelEntry(date: .now, snapshot: .placeholder, poseImage: nil)
    }

    func getSnapshot(in context: Context, completion: @escaping (MabelEntry) -> Void) {
        completion(currentEntry(galleryFallback: context.isPreview))
    }

    func getTimeline(in context: Context, completion: @escaping (Timeline<MabelEntry>) -> Void) {
        let entry = currentEntry(galleryFallback: false)
        // The app pushes reloads on change; refresh every 15 min as a backstop so
        // the "updated Xm ago" stamp keeps advancing even if the app never runs.
        let next = Calendar.current.date(byAdding: .minute, value: 15, to: .now) ?? .now.addingTimeInterval(900)
        completion(Timeline(entries: [entry], policy: .after(next)))
    }

    private func currentEntry(galleryFallback: Bool) -> MabelEntry {
        guard let snap = MabelWidgetShared.load() else {
            // No data yet: pretty sample in the gallery, an honest "open app to
            // connect" state on the actual Home Screen.
            return MabelEntry(date: .now,
                              snapshot: galleryFallback ? .placeholder : .coldStart,
                              poseImage: nil)
        }
        var image: UIImage? = nil
        if let name = snap.poseImage, let url = MabelWidgetShared.fileURL(name) {
            image = UIImage(contentsOfFile: url.path)
        }
        return MabelEntry(date: .now, snapshot: snap, poseImage: image)
    }
}

// MARK: - Widget

struct MabelStatusWidget: Widget {
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: MabelWidgetShared.kind, provider: MabelProvider()) { entry in
            MabelWidgetEntryView(entry: entry)
                .containerBackground(WidgetTheme.canvas, for: .widget)
        }
        .configurationDisplayName("MABEL Status")
        .description("Live 3D pose, connection, battery and control mode for your MABEL robot.")
        .supportedFamilies([
            .systemSmall, .systemMedium, .systemLarge,
            .accessoryRectangular, .accessoryCircular, .accessoryInline,
        ])
    }
}
