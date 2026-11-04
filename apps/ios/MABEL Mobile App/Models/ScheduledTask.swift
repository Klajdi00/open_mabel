//
//  ScheduledTask.swift
//  MABEL Mobile App
//
//  A user-programmed robot routine — an Apple-Home/Shortcuts-style automation:
//  a task the robot runs at a time of day, optionally repeating on chosen
//  weekdays (e.g. "Make coffee · 7:00 AM · Daily"). Persisted across launches.
//  The actual time-triggered dispatch lives server-side; this is the interface
//  + schedule the operator programs.
//

import SwiftUI
import Combine

struct ScheduledTask: Identifiable, Codable, Equatable {
    var id: String = UUID().uuidString
    var title: String
    var icon: String
    var hour: Int                 // 0…23
    var minute: Int               // 0…59
    /// 1 = Sunday … 7 = Saturday (Calendar convention). Empty = run once (no repeat).
    var weekdays: Set<Int>
    var enabled: Bool = true
    /// Optional link to a Policy/skill to dispatch by name; nil = a named chore.
    var policyName: String? = nil

    /// 12-hour clock string, e.g. "7:00 AM".
    var timeString: String {
        let ampm = hour < 12 ? "AM" : "PM"
        let h12 = hour % 12 == 0 ? 12 : hour % 12
        return String(format: "%d:%02d %@", h12, minute, ampm)
    }

    /// Friendly repeat summary, e.g. "Daily", "Weekdays", "Mon · Wed", "Once".
    var repeatString: String {
        if weekdays.isEmpty { return "Once" }
        if weekdays.count == 7 { return "Daily" }
        if weekdays == [2, 3, 4, 5, 6] { return "Weekdays" }
        if weekdays == [1, 7] { return "Weekends" }
        let names = ["", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        return weekdays.sorted().map { names[$0] }.joined(separator: " · ")
    }

    static let everyDay: Set<Int> = [1, 2, 3, 4, 5, 6, 7]
}

/// Persistent store for the operator's programmed routines.
@MainActor
final class TaskScheduleStore: ObservableObject {
    @Published var tasks: [ScheduledTask] = [] {
        didSet { save() }
    }

    private let key = "robot.scheduledTasks.v1"

    init() {
        load()
        if tasks.isEmpty { tasks = Self.seed }
    }

    func add(_ task: ScheduledTask) { tasks.append(task); sort() }

    func update(_ task: ScheduledTask) {
        if let i = tasks.firstIndex(where: { $0.id == task.id }) { tasks[i] = task; sort() }
    }

    func remove(_ task: ScheduledTask) { tasks.removeAll { $0.id == task.id } }

    func toggle(_ task: ScheduledTask) {
        if let i = tasks.firstIndex(where: { $0.id == task.id }) { tasks[i].enabled.toggle() }
    }

    private func sort() {
        tasks.sort { $0.hour != $1.hour ? $0.hour < $1.hour : $0.minute < $1.minute }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(tasks) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode([ScheduledTask].self, from: data) else { return }
        tasks = decoded
    }

    /// First-run examples that match how operators think about home routines.
    static let seed: [ScheduledTask] = [
        .init(title: "Make coffee", icon: "cup.and.saucer.fill", hour: 7, minute: 0,
              weekdays: ScheduledTask.everyDay),
        .init(title: "Do laundry", icon: "washer.fill", hour: 9, minute: 0,
              weekdays: [2, 3, 4, 5, 6]),
        .init(title: "Evening patrol", icon: "shield.lefthalf.filled", hour: 22, minute: 30,
              weekdays: ScheduledTask.everyDay, enabled: false),
    ]
}
