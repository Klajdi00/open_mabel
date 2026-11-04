//
//  WeeklyScheduleView.swift
//  MabelTeleop
//
//  A Calendar-style weekly timeline of the robot's programmed routines: a 7-day
//  grid over a 24-hour axis with each automation in its day/hour cell, a pinned
//  day-label header, press-and-hold-then-drag to reschedule, and a long-press
//  menu (Edit / Duplicate / Copy / Delete). Tap an empty cell to add or paste.
//

import SwiftUI

struct WeeklyScheduleView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var store: TaskScheduleStore

    @State private var editing: ScheduledTask? = nil
    @State private var clipboard: ScheduledTask? = nil
    @State private var addSlot: Slot? = nil
    @State private var colW: CGFloat = 60

    @State private var dragKey: String? = nil
    @State private var dragTranslation: CGSize = .zero

    private struct Slot: Identifiable { let day: Int; let hour: Int; var id: String { "\(day)-\(hour)" } }

    private let hours = Array(0..<24)
    private let rowHeight: CGFloat = 56
    private let gutter: CGFloat = 50
    private let days: [(n: Int, label: String)] =
        [(1, "Sun"), (2, "Mon"), (3, "Tue"), (4, "Wed"), (5, "Thu"), (6, "Fri"), (7, "Sat")]

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: false) {
                    LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                        Section {
                            ForEach(hours, id: \.self) { h in row(h).id("h\(h)") }
                        } header: {
                            dayHeader
                        }
                    }
                }
                .onAppear { proxy.scrollTo("h6", anchor: .top) }
            }
            .background(Theme.Palette.canvas)
            .navigationTitle("Weekly Schedule")
            .tint(Theme.Palette.accent)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button { addSlot = Slot(day: 2, hour: 8) } label: { Image(systemName: "plus") }
                }
            }
            .sheet(item: $editing) { t in AutomationEditorView(store: store, task: t, isNew: false) }
            .sheet(item: $addSlot) { slot in
                AutomationEditorView(
                    store: store,
                    task: clipboard.map { var c = $0; c.id = UUID().uuidString; c.hour = slot.hour; c.weekdays = [slot.day]; return c }
                        ?? ScheduledTask(title: "", icon: "sparkles", hour: slot.hour, minute: 0, weekdays: [slot.day]),
                    isNew: true)
            }
        }
        .frame(width: 760, height: 760)
    }

    private var dayHeader: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: gutter)
            ForEach(days, id: \.n) { d in
                Text(d.label)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(.vertical, 8)
        .background(Theme.Palette.ink)
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.Palette.hairline).frame(height: 1) }
    }

    private func row(_ h: Int) -> some View {
        HStack(spacing: 0) {
            Text(hourLabel(h))
                .font(.system(size: 11, weight: .medium)).monospacedDigit()
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.trailing, 6)
                .frame(width: gutter, height: rowHeight, alignment: .topTrailing)
                .offset(y: -7)
            ForEach(days, id: \.n) { d in cell(day: d.n, hour: h) }
        }
        .frame(height: rowHeight)
    }

    private func cell(day: Int, hour: Int) -> some View {
        ZStack {
            if let task = taskAt(day: day, hour: hour) { blockView(task, day: day) }
        }
        .frame(maxWidth: .infinity, minHeight: rowHeight, maxHeight: rowHeight)
        .overlay(alignment: .top) { Rectangle().fill(Theme.Palette.hairlineSoft).frame(height: 1) }
        .overlay(alignment: .leading) { Rectangle().fill(Theme.Palette.hairlineSoft).frame(width: 1) }
        .contentShape(Rectangle())
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { colW = $0 }
        .onTapGesture { if taskAt(day: day, hour: hour) == nil { addSlot = Slot(day: day, hour: hour) } }
    }

    private func taskAt(day: Int, hour: Int) -> ScheduledTask? {
        store.tasks.first { $0.weekdays.contains(day) && $0.hour == hour }
    }

    private func hourLabel(_ h: Int) -> String {
        let ampm = h < 12 ? "AM" : "PM"
        let h12 = h % 12 == 0 ? 12 : h % 12
        return "\(h12)\(ampm)"
    }

    private func blockView(_ task: ScheduledTask, day: Int) -> some View {
        let key = "\(task.id)-\(day)"
        let isDragging = dragKey == key
        let tint = task.enabled ? Theme.Palette.accent : Theme.Palette.textTertiary

        return VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 4) {
                Image(systemName: task.icon).font(.system(size: 10, weight: .bold))
                Text(task.title.isEmpty ? "Task" : task.title)
                    .font(.system(size: 11, weight: .bold)).lineLimit(1)
            }
            Text(task.timeString).font(.system(size: 10, weight: .medium)).opacity(0.9).lineLimit(1)
        }
        .foregroundStyle(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.horizontal, 6).padding(.vertical, 4)
        .background(tint.gradient, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 8).strokeBorder(.white.opacity(0.25), lineWidth: 0.5) }
        .opacity(task.enabled ? 1 : 0.55)
        .padding(2)
        .scaleEffect(isDragging ? 1.05 : 1)
        .offset(isDragging ? dragTranslation : .zero)
        .zIndex(isDragging ? 10 : 1)
        .animation(.spring(response: 0.28, dampingFraction: 0.8), value: isDragging)
        .gesture(
            LongPressGesture(minimumDuration: 0.2)
                .sequenced(before: DragGesture(minimumDistance: 0))
                .onChanged { value in
                    if case .second(true, let drag) = value {
                        if dragKey != key { dragKey = key; Haptics.rigid() }
                        dragTranslation = drag?.translation ?? .zero
                    }
                }
                .onEnded { value in
                    if case .second(true, let drag?) = value {
                        commitDrag(task, weekday: day, translation: drag.translation)
                    }
                    dragKey = nil; dragTranslation = .zero
                }
        )
        .onTapGesture { editing = task }
        .contextMenu {
            Button { editing = task } label: { Label("Edit", systemImage: "pencil") }
            Button { duplicate(task, weekday: day) } label: { Label("Duplicate", systemImage: "plus.square.on.square") }
            Button { clipboard = task } label: { Label("Copy", systemImage: "doc.on.doc") }
            Button { store.toggle(task) } label: {
                Label(task.enabled ? "Disable" : "Enable", systemImage: task.enabled ? "pause" : "play")
            }
            Button(role: .destructive) { store.remove(task) } label: { Label("Delete", systemImage: "trash") }
        }
    }

    private func commitDrag(_ task: ScheduledTask, weekday: Int, translation: CGSize) {
        let dHour = Int((translation.height / rowHeight).rounded())
        let dCol = colW > 0 ? Int((translation.width / colW).rounded()) : 0
        var t = task
        t.hour = min(23, max(0, task.hour + dHour))
        t.minute = 0
        if task.weekdays.count == 1, dCol != 0 { t.weekdays = [min(7, max(1, weekday + dCol))] }
        store.update(t)
        Haptics.success()
    }

    private func duplicate(_ task: ScheduledTask, weekday: Int) {
        var copy = task
        copy.id = UUID().uuidString
        copy.hour = min(23, task.hour + 1)
        copy.weekdays = [weekday]
        store.add(copy)
        Haptics.success()
    }
}
