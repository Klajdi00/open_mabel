//
//  AuxWindows.swift
//  MabelTeleop
//
//  Control-center-style ADD-ON WINDOWS, opened from the Robot space's
//  "Add Window" menu — real visionOS windows the operator can place anywhere:
//
//    · AuxCameraWindow — any single camera stream (main / front / wrists)
//    · JointBarsWindow — live bar graph of the current joint values with a
//      Position / Velocity / Torque toggle
//    · TimeSeriesWindow — rolling 2-minute position + velocity + torque line
//      charts with per-module add/remove
//    · Map2DWindow — top-down 2D occupancy map (grid + odom + route + goal)
//
//  Data honesty: velocity prefers the wire's `jointVelocities` (sim-bridge V2
//  extension) and falls back to numeric differentiation; torque is shown ONLY
//  when the wire carries `jointTorques` (the demo synthesizes its own spring
//  model) — otherwise the panel says "not streamed", never fakes.
//

import SwiftUI
import Charts
import Combine
import SceneKit    // MapPose/MapPath carry SCNVector3 components

/// Which quantity a telemetry panel displays.
enum TelemetryQuantity: String, CaseIterable, Identifiable {
    case position, velocity, torque
    var id: String { rawValue }
    var title: String {
        switch self {
        case .position: return "Pos"
        case .velocity: return "Vel"
        case .torque:   return "Torque"
        }
    }
    var icon: String {
        switch self {
        case .position: return "angle"
        case .velocity: return "gauge.with.needle"
        case .torque:   return "tornado"
        }
    }
    var unit: String {
        switch self {
        case .position: return "rad · m"
        case .velocity: return "rad/s · m/s"
        case .torque:   return "N·m"
        }
    }
}

/// The joints a module contributes to the graphs (same slicing as the console's
/// JointGraphs: hands capped at 8 joints so legends stay readable).
func moduleSpecs(_ m: ModuleFocus) -> [JointSpec] {
    switch m {
    case .armLeft:   return RobotModel.specs(in: .arms).filter { $0.id.hasPrefix("left") }
    case .armRight:  return RobotModel.specs(in: .arms).filter { $0.id.hasPrefix("right") }
    case .handLeft:  return Array(RobotModel.specs(in: .hands).filter { $0.id.hasPrefix("left") }.prefix(8))
    case .handRight: return Array(RobotModel.specs(in: .hands).filter { $0.id.hasPrefix("right") }.prefix(8))
    case .neck:      return RobotModel.specs(in: .body).filter { $0.id.hasPrefix("neck") }
    case .lift:      return RobotModel.specs(in: .body).filter { $0.id.hasPrefix("lift") }
    case .torso:     return RobotModel.specs(in: .body).filter { $0.id == "torso" }
    }
}

// MARK: - Cameras & Sensors window (all video + sensor feeds together)

struct SensorsWindow: View {
    // Plain reference, NOT @EnvironmentObject: observing the controller re-renders
    // this window at uiTick rate (~8 Hz), recreating the sampling timers before
    // they ever fire (the iOS JointGraphs lesson). Windows read live values on
    // their own cadence instead.
    let controller: RobotController
    /// Tap any tile → its own placeable camera-detail window (zoom + pan).
    @Environment(\.openWindow) private var openWindow
    /// LiDAR tile refresh — the sweep is latched on the controller (~5 Hz);
    /// polled here so the window never observes the controller.
    @State private var scan: RobotState.LidarScan? = nil
    @State private var sampler = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("CAMERAS & SENSORS").font(.system(size: 11, weight: .heavy)).tracking(1.6)
                .foregroundStyle(Theme.Palette.textTertiary)
            CameraFeed(name: "HEAD · POV", isMain: true,
                       streamURL: controller.videoMainURL("main"),
                       stream: "main",
                       onExpand: { openWindow(id: WindowID.cameraDetail.rawValue, value: "main") })
                .frame(maxHeight: .infinity)
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10),
                                GridItem(.flexible(), spacing: 10), GridItem(.flexible())], spacing: 10) {
                CameraFeed(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                           streamURL: controller.videoThumbURL("wrist_left"),
                           stream: "wrist_left",
                           onExpand: { openWindow(id: WindowID.cameraDetail.rawValue, value: "wrist_left") })
                    .frame(height: 150)
                CameraFeed(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                           streamURL: controller.videoThumbURL("wrist_right"),
                           stream: "wrist_right",
                           onExpand: { openWindow(id: WindowID.cameraDetail.rawValue, value: "wrist_right") })
                    .frame(height: 150)
                CameraFeed(name: "BASE", compact: true, tint: Theme.Palette.accent,
                           streamURL: controller.videoThumbURL("front"),
                           stream: "front",
                           onExpand: { openWindow(id: WindowID.cameraDetail.rawValue, value: "front") })
                    .frame(height: 150)
                LidarView(scan: scan)
                    .frame(height: 150)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous)
                            .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                    }
            }
        }
        .padding(14)
        .onReceive(sampler) { _ in scan = controller.lidarScan }
    }
}

// MARK: - Joint bar graph window

struct JointBarsWindow: View {
    let controller: RobotController
    @State private var module: ModuleFocus = .armRight
    @State private var quantity: TelemetryQuantity = .position
    @State private var values: [String: Double] = [:]
    @State private var torqueOnWire = false
    @State private var lastPos: [String: Double] = [:]
    @State private var lastPosAt: Date? = nil
    // @State so the publisher SURVIVES view-struct recreation: the App re-evaluates
    // its Scene body on every controller objectWillChange (~8 Hz uiTick), which
    // re-inits this struct — a plain `let` publisher would be replaced (and its
    // countdown reset) faster than it can ever fire.
    @State private var sampler = Timer.publish(every: 0.15, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text("JOINT BARS").font(.system(size: 11, weight: .heavy)).tracking(1.6)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Spacer(minLength: 0)
                GlassSegmented(options: TelemetryQuantity.allCases,
                               selection: $quantity,
                               title: { $0.title }, icon: { $0.icon })
                    .fixedSize()
            }
            modulePicker($module)
            if quantity == .torque && !torqueOnWire {
                ContentUnavailableView {
                    Label("Torque not streamed", systemImage: "tornado")
                } description: {
                    Text("This bridge's robot_state carries positions only — connect to a sim bridge with the V2 telemetry extension (or Demo Mode) for torque.")
                }
            } else {
                chart
            }
            Text(quantity.unit)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(Theme.Palette.textTertiary)
        }
        .padding(16)
        .onReceive(sampler) { _ in sample() }
    }

    private var chart: some View {
        Chart {
            ForEach(moduleSpecs(module)) { spec in
                BarMark(x: .value("Joint", String(spec.label.dropFirst(2))),
                        y: .value("Value", values[spec.id] ?? 0))
                    .foregroundStyle(by: .value("Joint", String(spec.label.dropFirst(2))))
                    .cornerRadius(3)
            }
        }
        .chartLegend(.hidden)
        .chartXAxis {
            AxisMarks { AxisValueLabel().font(.system(size: 9, weight: .semibold)) }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 6)) {
                AxisGridLine(); AxisValueLabel().font(.system(size: 9))
            }
        }
        .animation(.linear(duration: 0.12), value: values)
    }

    private func sample() {
        let now = Date()
        let pos = controller.livePose.joints
        switch quantity {
        case .position:
            values = pos
        case .velocity:
            let wire = controller.jointVelocities
            if !wire.isEmpty {
                values = wire
            } else if let t0 = lastPosAt {
                // No velocities on the wire → numeric derivative of positions.
                let dt = max(0.05, now.timeIntervalSince(t0))
                values = Dictionary(uniqueKeysWithValues: pos.map { (k, v) in
                    (k, (v - (lastPos[k] ?? v)) / dt)
                })
            }
        case .torque:
            values = controller.jointTorques
        }
        torqueOnWire = !controller.jointTorques.isEmpty
        lastPos = pos
        lastPosAt = now
    }
}

/// Shared horizontal module chip row (single-select).
@ViewBuilder
func modulePicker(_ selection: Binding<ModuleFocus>) -> some View {
    ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 6) {
            ForEach(ModuleFocus.allCases) { m in
                let active = selection.wrappedValue == m
                Button {
                    Haptics.select(); selection.wrappedValue = m
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: m.icon).font(.system(size: 11, weight: .semibold))
                        Text(m.title).font(.system(size: 12, weight: .semibold))
                    }
                    .foregroundStyle(active ? .white : Theme.Palette.textSecondary)
                    .padding(.horizontal, 11).padding(.vertical, 7)
                    .background {
                        if active { Capsule().fill(Theme.accentGradient) }
                        else { Capsule().fill(Theme.Palette.surface) }
                    }
                }
                .buttonStyle(.plain)
                .hoverEffect()
            }
        }
    }
}

// MARK: - Time-series window

struct TimeSeriesWindow: View {
    let controller: RobotController
    /// Modules shown on the charts — add/remove via the chips (multi-select).
    @State private var modules: Set<ModuleFocus> = [.armRight]
    @State private var samples: [Sample] = []
    // @State so the publisher SURVIVES view-struct recreation: the App re-evaluates
    // its Scene body on every controller objectWillChange (~8 Hz uiTick), which
    // re-inits this struct — a plain `let` publisher would be replaced (and its
    // countdown reset) faster than it can ever fire.
    @State private var sampler = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    struct Sample {
        let t: Date
        let pos: [String: Double]
        let vel: [String: Double]
        let torque: [String: Double]
    }

    private var specsList: [JointSpec] {
        ModuleFocus.allCases.filter { modules.contains($0) }.flatMap { moduleSpecs($0) }
    }
    private var torqueOnWire: Bool { !(samples.last?.torque.isEmpty ?? true) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("TIME SERIES · 2 MIN").font(.system(size: 11, weight: .heavy)).tracking(1.6)
                    .foregroundStyle(Theme.Palette.textTertiary)
                moduleChips
                chart(title: "POSITION (rad · m)") { s, id in s.pos[id] }
                chart(title: "VELOCITY (rad/s · m/s)") { s, id in s.vel[id] }
                if torqueOnWire {
                    chart(title: "TORQUE (N·m)") { s, id in s.torque[id] }
                } else {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("TORQUE (N·m)").font(.system(size: 9, weight: .heavy)).tracking(1)
                            .foregroundStyle(Theme.Palette.textTertiary)
                        Text("Not streamed by this bridge — positions only on the wire.")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }
            }
            .padding(16)
        }
        .onReceive(sampler) { _ in
            let now = Date()
            let pos = controller.livePose.joints
            let wireVel = controller.jointVelocities
            let vel: [String: Double]
            if !wireVel.isEmpty {
                vel = wireVel
            } else if let prev = samples.last {
                let dt = max(0.05, now.timeIntervalSince(prev.t))
                vel = Dictionary(uniqueKeysWithValues: pos.map { (k, v) in
                    (k, (v - (prev.pos[k] ?? v)) / dt)
                })
            } else {
                vel = [:]
            }
            samples.append(Sample(t: now, pos: pos, vel: vel,
                                  torque: controller.jointTorques))
            let cutoff = now.addingTimeInterval(-120)
            if let first = samples.first, first.t < cutoff {
                samples.removeAll { $0.t < cutoff }
            }
        }
    }

    /// Multi-select module chips — tap to add/remove a module from the charts.
    private var moduleChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(ModuleFocus.allCases) { m in
                    let active = modules.contains(m)
                    Button {
                        Haptics.select()
                        if active {
                            if modules.count > 1 { modules.remove(m) }   // keep ≥ 1
                        } else {
                            modules.insert(m)
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: active ? "checkmark" : "plus")
                                .font(.system(size: 9, weight: .bold))
                            Text(m.title).font(.system(size: 12, weight: .semibold))
                        }
                        .foregroundStyle(active ? .white : Theme.Palette.textSecondary)
                        .padding(.horizontal, 11).padding(.vertical, 7)
                        .background {
                            if active { Capsule().fill(Theme.accentGradient) }
                            else { Capsule().fill(Theme.Palette.surface) }
                        }
                    }
                    .buttonStyle(.plain)
                    .hoverEffect()
                }
            }
        }
    }

    @ViewBuilder private func chart(title: String,
                                    value: @escaping (Sample, String) -> Double?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.system(size: 9, weight: .heavy)).tracking(1)
                .foregroundStyle(Theme.Palette.textTertiary)
            Chart {
                ForEach(specsList) { spec in
                    ForEach(Array(samples.enumerated()), id: \.offset) { _, s in
                        if let v = value(s, spec.id) {
                            LineMark(x: .value("t", s.t), y: .value("v", v))
                                .foregroundStyle(by: .value("Joint", spec.label))
                                .lineStyle(StrokeStyle(lineWidth: 1.2))
                        }
                    }
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) {
                    AxisGridLine(); AxisTick()
                    AxisValueLabel(format: .dateTime.minute().second(), centered: false)
                        .font(.system(size: 8))
                }
            }
            .chartYAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) {
                    AxisGridLine(); AxisValueLabel().font(.system(size: 8))
                }
            }
            .chartLegend(position: .bottom, spacing: 4)
            .frame(height: 190)
        }
    }
}

// MARK: - 2D occupancy map window

struct Map2DWindow: View {
    let controller: RobotController
    @StateObject private var navState = NavMapState()
    @State private var observerID: UUID? = nil

    var body: some View {
        GeometryReader { geo in
            VStack(spacing: 8) {
                Text("2D OCCUPANCY MAP").font(.system(size: 11, weight: .heavy)).tracking(1.6)
                    .foregroundStyle(Theme.Palette.textTertiary)
                MiniMap2D(state: navState, size: min(geo.size.width, geo.size.height) - 70)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .padding(14)
        }
        .onAppear {
            navState.beginLive()
            controller.liveMap.retainStream()
            var poseTick = 0
            observerID = controller.liveMap.addObserver(.init(
                mesh: { [weak navState] m in navState?.ingest(mesh: m) },
                pose: { [weak navState] p in
                    poseTick += 1
                    guard poseTick % 3 == 0 else { return }
                    navState?.setRobot(x: Double(p.position.x), z: Double(p.position.z),
                                       yaw: Double(p.yaw))
                },
                path: { [weak navState] path in
                    navState?.setLivePath(path.points.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.z)) })
                },
                grid: { [weak navState] g in navState?.ingest(grid: g) }
            ))
        }
        .onDisappear {
            if let id = observerID { controller.liveMap.removeObserver(id) }
            observerID = nil
            controller.liveMap.releaseStream()
        }
    }
}
