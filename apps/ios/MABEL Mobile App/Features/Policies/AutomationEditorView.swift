//
//  AutomationEditorView.swift
//  MABEL Mobile App
//
//  Apple-Shortcuts-style editor for a programmed robot routine: name it, give it
//  an icon, pick a time, choose which days it repeats, and optionally bind it to
//  a learned skill. Standard Form sheet so it reads exactly like iOS automations.
//

import SwiftUI

struct AutomationEditorView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var store: TaskScheduleStore
    @State var task: ScheduledTask
    let isNew: Bool

    @State private var time: Date = Date()

    private let icons = ["cup.and.saucer.fill", "washer.fill", "house.fill",
                         "shield.lefthalf.filled", "sparkles", "trash.fill",
                         "fanblades.fill", "lightbulb.fill", "figure.walk", "camera.fill"]
    private let days: [(Int, String)] = [(1, "S"), (2, "M"), (3, "T"), (4, "W"), (5, "T"), (6, "F"), (7, "S")]

    var body: some View {
        NavigationStack {
            Form {
                Section("Task") {
                    TextField("Name (e.g. Make coffee)", text: $task.title)
                    iconPicker
                }
                Section("Time") {
                    DatePicker("Run at", selection: $time, displayedComponents: .hourAndMinute)
                }
                Section("Repeat") { weekdayPicker }
                Section("Run skill") { skillPicker }
                if !isNew {
                    Section {
                        Button(role: .destructive) {
                            store.remove(task); dismiss()
                        } label: { Label("Delete Automation", systemImage: "trash") }
                    }
                }
            }
            .navigationTitle(isNew ? "New Automation" : "Edit Automation")
            .navigationBarTitleDisplayMode(.inline)
            .tint(Theme.Palette.accent)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .disabled(task.title.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .onAppear {
                var c = DateComponents(); c.hour = task.hour; c.minute = task.minute
                time = Calendar.current.date(from: c) ?? Date()
            }
        }
        .presentationDetents([.large])
    }

    private var iconPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(icons, id: \.self) { icon in
                    let on = task.icon == icon
                    Button { task.icon = icon } label: {
                        Image(systemName: icon)
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(on ? .white : Theme.Palette.accent)
                            .frame(width: 40, height: 40)
                            .background(on ? AnyShapeStyle(Theme.Palette.accent)
                                           : AnyShapeStyle(Theme.Palette.accent.opacity(0.14)),
                                        in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var weekdayPicker: some View {
        HStack(spacing: 6) {
            ForEach(days, id: \.0) { (n, label) in
                let on = task.weekdays.contains(n)
                Button {
                    if on { task.weekdays.remove(n) } else { task.weekdays.insert(n) }
                } label: {
                    Text(label)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(on ? .white : Theme.Palette.textSecondary)
                        .frame(width: 36, height: 36)
                        .background(on ? AnyShapeStyle(Theme.Palette.accent)
                                       : AnyShapeStyle(Theme.Palette.surfaceHi), in: Circle())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var skillPicker: some View {
        Picker("Skill", selection: Binding(
            get: { task.policyName ?? "" },
            set: { task.policyName = $0.isEmpty ? nil : $0 }
        )) {
            Text("None").tag("")
            ForEach(Policy.library) { p in Text(p.name).tag(p.name) }
        }
    }

    private func save() {
        let c = Calendar.current.dateComponents([.hour, .minute], from: time)
        task.hour = c.hour ?? task.hour
        task.minute = c.minute ?? task.minute
        if isNew { store.add(task) } else { store.update(task) }
        Haptics.success()
        dismiss()
    }
}
