//
//  RobotSpaceView.swift
//  MABEL Mobile App
//
//  The unified Robot space — v2's replacement for the separate Pilot /
//  Manipulate / Navigate tabs. ONE full-bleed 3D viewer is both the visualizer
//  and the controller (Tesla / Roborock style: the hero live view + a few
//  floating pills + a pull-up console for everything else):
//
//    • The live SLAM map (mesh + occupancy + Nav2 route) with the articulated
//      MABEL at its live odom pose — tap the floor to send a Nav2 goal.
//    • 3rd ↔ 1st person: the viewer camera flies from WHEREVER you orbited it
//      into the robot's head camera (and back); live FPV crossfades in.
//    • Teleop: Start Teleop + a Drive · Joint · Task space picker. Drive
//      (base joysticks) is landscape-only; Joint (arm joysticks) and Task
//      (grab the robot's arms / palm balls / wrist rings in the viewer) work
//      in BOTH orientations. Stiff = IK commands; Soft = compliant pushes.
//    • The pull-up CONSOLE (bottom grabber, like the camera sheet): status,
//      layer toggles (mesh / cloud / grid / minimap / Cam 3D), fine viz
//      controls (projection size, flat vs lens-curved, cloud density) and
//      the live camera tiles.
//
//  The chrome (status bar, nav rail / bottom bar) is owned by ContentView;
//  everything here floats over the scene.
//

import SwiftUI
import Combine
import UIKit
import Charts

/// Telemetry metric shared by the live bar panel and the history chart —
/// the web control center's header seg, ported.
private enum TelemetryMetric: String, CaseIterable, Identifiable {
    case pos, vel, torque
    var id: String { rawValue }
    var title: String {
        switch self { case .pos: return "Pos"; case .vel: return "Vel"; case .torque: return "Torque" }
    }
    var unit: String {
        switch self { case .pos: return "rad · m"; case .vel: return "rad/s"; case .torque: return "N·m" }
    }
}

/// Console telemetry — the web control center's joint panel, ported. ONE
/// metric seg (Pos · Vel · Torque) drives BOTH plots:
///   • JOINTS · LIVE — a centre-line bar per joint, grouped by module, with
///     the per-joint temperature strip underneath (green / amber / red; faint
///     = not measured — honest, never guessed). Tap a bar to pin its full
///     readout. Two rows portrait, one row landscape.
///   • HISTORY · 2 MIN — one rolling line chart; the module chips are
///     multi-select OVERLAYS (chip colour = curve colour), not a radio.
/// Velocity/torque ride the wire's sparse ~10 Hz jointVelocities /
/// jointTorques (latched by the controller); velocity falls back to the
/// numerical derivative of position until the stream provides a joint.
private struct JointGraphs: View {
    let controller: RobotController
    @Environment(\.isLandscape) private var isLandscape
    @State private var metric: TelemetryMetric = {
        #if DEBUG
        // Screenshot harness: boot straight into a metric (`debugTelemetryMetric
        // = pos|vel|torque`) — synthetic taps don't land on the Simulator.
        if let s = UserDefaults.standard.string(forKey: "debugTelemetryMetric"),
           let m = TelemetryMetric(rawValue: s) { return m }
        #endif
        return .pos
    }()
    @State private var enabled: Set<ModuleFocus> = [.armLeft, .armRight]
    /// The HISTORY chart has its own metric — bars and history can show
    /// different quantities side by side (e.g. live torque bars + pos trends).
    @State private var historyMetric: TelemetryMetric = {
        #if DEBUG
        if let s = UserDefaults.standard.string(forKey: "debugTelemetryMetric"),
           let m = TelemetryMetric(rawValue: s) { return m }
        #endif
        return .pos
    }()
    @State private var pinned: String? = nil
    @State private var samples: [Sample] = []
    /// STATIC so the publisher survives view re-inits: the console rebuilds this
    /// struct on every @Published telemetry tick (~10 Hz), and an instance-let
    /// Timer.publish would be a NEW publisher each time — onReceive resubscribes,
    /// the 0.5 s tick never fires, and the history chart stays empty forever.
    private static let sampler = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()
    /// The BARS run faster than the 2 Hz history sampler — they are a live
    /// readout, not a trend — and, more importantly, they run off an explicit
    /// tick.
    ///
    /// They used to be a `Canvas` inside `TimelineView(.animation)` reading
    /// `controller.livePose` directly. `livePose` is deliberately NOT `@Published`
    /// (it is written ~100 Hz and publishing it would invalidate the whole tree at
    /// the state rate), so the Canvas had no changing input at all: SwiftUI saw a
    /// view whose every parameter was identical frame to frame and had no reason
    /// to re-invoke the draw closure. The line chart kept working because it hangs
    /// off `@State samples`, which really does change. Result: identical data
    /// sources, one live view and one frozen one (reported 2026-07-29).
    private static let barTicker = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    /// One snapshot, passed DOWN. Making the bars a pure function of their inputs
    /// is what guarantees they redraw — and that they agree with the chart, since
    /// both now read the same fields at a tick rather than whenever they happen to
    /// be evaluated.
    @State private var live = LiveJointSnapshot()

    struct Sample {
        let t: Date
        let pos: [String: Double]
        let vel: [String: Double]
        let tor: [String: Double]
    }

    /// The joints of one module. The history chart trims each hand to its
    /// first 8 DOFs (line-count sanity); the bar panel shows the full set.
    static func specs(for m: ModuleFocus, full: Bool = false) -> [JointSpec] {
        switch m {
        case .armLeft:   return RobotModel.specs(in: .arms).filter { $0.id.hasPrefix("left") }
        case .armRight:  return RobotModel.specs(in: .arms).filter { $0.id.hasPrefix("right") }
        case .handLeft:
            let s = RobotModel.specs(in: .hands).filter { $0.id.hasPrefix("left") }
            return full ? s : Array(s.prefix(8))
        case .handRight:
            let s = RobotModel.specs(in: .hands).filter { $0.id.hasPrefix("right") }
            return full ? s : Array(s.prefix(8))
        case .neck:      return RobotModel.specs(in: .body).filter { $0.id.hasPrefix("neck") }
        case .lift:      return RobotModel.specs(in: .body).filter { $0.id.hasPrefix("lift") }
        case .torso:     return RobotModel.specs(in: .body).filter { $0.id == "torso" }
        }
    }

    /// One stable hue per module — the chips, the curves and the bars all
    /// share it, so the chips double as the chart legend.
    static func hue(_ m: ModuleFocus) -> Color {
        switch m {
        case .armLeft:   return Color(hex: 0xC25B2A)   // rust
        case .armRight:  return Color(hex: 0x2E9E52)   // green
        case .neck:      return Color(hex: 0x4D9DFF)   // blue
        case .torso:     return Color(hex: 0xCC8A1E)   // amber
        case .lift:      return Color(hex: 0x8E6FD8)   // violet
        case .handLeft:  return Color(hex: 0x9A6A3C)   // clay
        case .handRight: return Color(hex: 0x2AA8A0)   // teal
        }
    }

    private var barRows: [[ModuleFocus]] {
        isLandscape
            ? [[.armLeft, .armRight, .neck, .torso, .lift, .handLeft, .handRight]]
            : [[.armLeft, .armRight, .neck, .torso, .lift], [.handLeft, .handRight]]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("TELEMETRY").font(.system(size: 10, weight: .heavy)).tracking(1.4)
                .foregroundStyle(Theme.Palette.textTertiary)

            HStack {
                Text("JOINTS · LIVE (\(metric.unit))")
                    .font(.system(size: 8.5, weight: .heavy)).tracking(1)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Spacer()
                metricSeg($metric)
            }
            JointBars(live: live, metric: metric, rows: barRows, pinned: $pinned)
            if let id = pinned { readout(id) }

            HStack {
                Text("HISTORY · 2 MIN (\(historyMetric.unit))")
                    .font(.system(size: 8.5, weight: .heavy)).tracking(1)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Spacer()
                metricSeg($historyMetric)
            }
            .padding(.top, 2)
            moduleChips
            historyChart
            if historyMetric == .torque, !samples.contains(where: { !$0.tor.isEmpty }) {
                Text("Waiting for the torque stream — this robot hasn't sent jointTorques yet.")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .onReceive(Self.barTicker) { _ in
            live = LiveJointSnapshot(pos: controller.livePose.joints,
                                     vel: controller.jointVel,
                                     tor: controller.jointTorque,
                                     temp: controller.jointTemp)
        }
        .onReceive(Self.sampler) { _ in
            let now = Date()
            samples.append(Sample(t: now,
                                  pos: controller.livePose.joints,
                                  vel: controller.jointVel,
                                  tor: controller.jointTorque))
            let cutoff = now.addingTimeInterval(-120)
            if let first = samples.first, first.t < cutoff {
                samples.removeAll { $0.t < cutoff }
            }
        }
    }

    // MARK: - Controls

    private func metricSeg(_ sel: Binding<TelemetryMetric>) -> some View {
        HStack(spacing: 2) {
            ForEach(TelemetryMetric.allCases) { m in
                Button {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { sel.wrappedValue = m }
                } label: {
                    Text(m.title).font(.system(size: 10.5, weight: .bold))
                        .foregroundStyle(sel.wrappedValue == m ? .white : Theme.Palette.textSecondary)
                        .padding(.horizontal, 9).padding(.vertical, 5)
                        .background { if sel.wrappedValue == m { Capsule().fill(Theme.accentGradient) } }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(Capsule().fill(Theme.Palette.surface))
    }

    /// Multi-select module overlays. At least one stays on (an empty chart
    /// reads as broken, not as "everything hidden").
    private var moduleChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(ModuleFocus.allCases) { m in
                    let on = enabled.contains(m)
                    Button {
                        Haptics.select()
                        withAnimation(Theme.Motion.snappy) {
                            if on { if enabled.count > 1 { enabled.remove(m) } }
                            else { enabled.insert(m) }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Circle().fill(Self.hue(m)).frame(width: 6, height: 6)
                                .opacity(on ? 1 : 0.4)
                            Text(m.title).font(.system(size: 11, weight: .semibold))
                        }
                        .foregroundStyle(on ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
                        .padding(.horizontal, 9).padding(.vertical, 6)
                        .background { Capsule().fill(on ? Self.hue(m).opacity(0.16) : Theme.Palette.surface) }
                        .overlay {
                            Capsule().strokeBorder(on ? Self.hue(m).opacity(0.55) : Theme.Palette.hairline,
                                                   lineWidth: 1)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - History chart

    private struct PlotPoint: Identifiable {
        let id: Int
        let t: Date
        let v: Double
        let joint: String
        let color: Color
    }

    /// Flattened, pre-colored plot data — computed OUTSIDE the Chart builder so
    /// the result-builder type-checks in reasonable time. Same hue per module
    /// with a brightness ramp per joint: the chip is the legend, individual
    /// joints stay distinguishable.
    private var plotPoints: [PlotPoint] {
        var out: [PlotPoint] = []
        var n = 0
        for m in ModuleFocus.allCases where enabled.contains(m) {
            let specs = Self.specs(for: m)
            for (i, spec) in specs.enumerated() {
                let fade = specs.count > 1 ? 0.55 * Double(i) / Double(specs.count - 1) : 0
                let col = Self.hue(m).opacity(1 - fade)
                for p in series(spec.id) {
                    out.append(PlotPoint(id: n, t: p.0, v: p.1, joint: spec.id, color: col))
                    n += 1
                }
            }
        }
        return out
    }

    private var historyChart: some View {
        Chart(plotPoints) { p in
            LineMark(x: .value("t", p.t), y: .value("v", p.v),
                     series: .value("Joint", p.joint))
                .foregroundStyle(p.color)
                .lineStyle(StrokeStyle(lineWidth: 1.1))
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) {
                AxisGridLine(); AxisTick()
                AxisValueLabel(format: .dateTime.minute().second(), centered: false)
                    .font(.system(size: 7))
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) {
                AxisGridLine(); AxisValueLabel().font(.system(size: 7))
            }
        }
        .chartLegend(.hidden)
        .frame(height: 150)
    }

    /// One joint's series for the current metric. Velocity prefers the
    /// streamed samples and falls back to the pos derivative; torque is
    /// streamed-only (absent → empty, surfaced by the note below the chart).
    private func series(_ id: String) -> [(Date, Double)] {
        switch historyMetric {
        case .pos:
            return samples.map { ($0.t, $0.pos[id] ?? 0) }
        case .vel:
            let streamed = samples.compactMap { s in s.vel[id].map { (s.t, $0) } }
            if streamed.count >= 2 { return streamed }
            let pos = samples.map { ($0.t, $0.pos[id] ?? 0) }
            guard pos.count > 1 else { return [] }
            return (1..<pos.count).map { i in
                let dt = max(0.05, pos[i].0.timeIntervalSince(pos[i - 1].0))
                return (pos[i].0, (pos[i].1 - pos[i - 1].1) / dt)
            }
        case .torque:
            return samples.compactMap { s in s.tor[id].map { (s.t, $0) } }
        }
    }

    // MARK: - Pinned readout

    @ViewBuilder private func readout(_ id: String) -> some View {
        if let spec = ModuleFocus.allCases.flatMap({ Self.specs(for: $0, full: true) })
            .first(where: { $0.id == id }) {
            let q = controller.livePose.joints[spec.id]
            let v = controller.jointVel[spec.id]
            let tau = controller.jointTorque[spec.id]
            let t = controller.jointTemp[spec.id]
            let deg: String = q.map { spec.kind == .prismatic ? String(format: "%.3f m", $0)
                                                              : String(format: "%.1f°", $0 * 180 / .pi) } ?? "—"
            let velS: String = v.map { String(format: "%.2f", $0) } ?? "—"
            let tauS: String = tau.map { String(format: "%.2f", $0) } ?? "—"
            let tmpS: String = t.map { String(format: "%.0f °C", $0) } ?? "temp —"
            let line = "\(spec.label)  ·  \(deg)  ·  vel \(velS)/s  ·  τ \(tauS)  ·  \(tmpS)"
            Text(line)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(1).minimumScaleFactor(0.7)
        }
    }
}

/// The MuJoCo viewer console section — renders only when the SCC studio is
/// reachable beside a sim (SimStudio.available).
private struct SimViewerSection: View {
    @ObservedObject var studio: SimStudio

    var body: some View {
        if let ws = studio.viewerWS {
            VStack(alignment: .leading, spacing: 8) {
                Text("SIM VIEWER · MUJOCO")
                    .font(.system(size: 10, weight: .heavy)).tracking(1.4)
                    .foregroundStyle(Theme.Palette.textTertiary)
                SimViewerTile(url: ws)
                    .frame(height: 200)
            }
        }
    }
}

/// The live joint-bar panel — a SwiftUI-Canvas port of the web control
/// center's `jointpanel.js`: one vertical bar per joint about a centre line
/// (position is normalized to each joint's own range; vel/torque to fixed
/// spans), module groups with dividers + labels, and the per-joint
/// temperature strip underneath (web `grading.js` thresholds: ≥55 °C amber,
/// ≥70 °C red; faint = not measured). Tap a bar to pin it.
/// A frozen read of the live joint telemetry. `Equatable` so SwiftUI can tell
/// when the bars genuinely need redrawing — and, crucially, tell when they DO.
struct LiveJointSnapshot: Equatable {
    var pos: [String: Double] = [:]
    var vel: [String: Double] = [:]
    var tor: [String: Double] = [:]
    var temp: [String: Double] = [:]
}

private struct JointBars: View {
    let live: LiveJointSnapshot
    let metric: TelemetryMetric
    let rows: [[ModuleFocus]]
    @Binding var pinned: String?

    private static let statusCols: [Color] =
        [Color(hex: 0x3ECF8E), Color(hex: 0xE7C24B), Color(hex: 0xD9463A)]

    var body: some View {
        VStack(spacing: 4) {
            ForEach(rows.indices, id: \.self) { r in
                rowView(rows[r]).frame(height: 84)
            }
        }
    }

    private func rowView(_ modules: [ModuleFocus]) -> some View {
        GeometryReader { geo in
            // No TimelineView. The snapshot IS the clock: `live` changes at 10 Hz
            // and `Canvas` redraws because its input did, which is both cheaper
            // than an animation-driven timeline and — unlike one — guaranteed.
            Canvas { ctx, size in draw(modules, ctx: &ctx, size: size) }
            .contentShape(Rectangle())
            .onTapGesture { p in
                let bars = layout(modules, width: geo.size.width)
                if let hit = bars.first(where: { p.x >= $0.x0 && p.x < $0.x1 }) {
                    Haptics.select()
                    pinned = (pinned == hit.spec.id) ? nil : hit.spec.id
                }
            }
        }
    }

    private func layout(_ modules: [ModuleFocus], width: CGFloat)
        -> [(spec: JointSpec, module: ModuleFocus, x0: CGFloat, x1: CGFloat)] {
        let padL: CGFloat = 2, padR: CGFloat = 2, gap: CGFloat = 12
        let groups = modules.map { ($0, JointGraphs.specs(for: $0, full: true)) }
        let n = groups.reduce(0) { $0 + $1.1.count }
        guard n > 0, width > 20 else { return [] }
        let innerW = width - padL - padR - gap * CGFloat(groups.count - 1)
        let bw = innerW / CGFloat(n)
        var out: [(JointSpec, ModuleFocus, CGFloat, CGFloat)] = []
        var x = padL
        for (gi, (m, specs)) in groups.enumerated() {
            for s in specs { out.append((s, m, x, x + bw)); x += bw }
            if gi < groups.count - 1 { x += gap }
        }
        return out.map { (spec: $0.0, module: $0.1, x0: $0.2, x1: $0.3) }
    }

    private func norm(_ v: Double, spec: JointSpec) -> Double {
        switch metric {
        case .pos:
            let span = spec.upper - spec.lower
            guard span > 1e-9 else { return 0 }
            return min(1, max(-1, 2 * (v - spec.lower) / span - 1))
        case .vel:    return min(1, max(-1, v / 3.0))     // web normVal span
        case .torque: return min(1, max(-1, v / 5.0))
        }
    }

    private func draw(_ modules: [ModuleFocus], ctx: inout GraphicsContext, size: CGSize) {
        let bars = layout(modules, width: size.width)
        guard !bars.isEmpty else { return }
        let labH: CGFloat = 11, tempH: CGFloat = 4
        let plotH = size.height - labH - tempH - 2
        let midY = plotH * 0.5 + 2
        let amp = plotH * 0.5 - 4

        var mid = Path()
        mid.move(to: CGPoint(x: 0, y: midY)); mid.addLine(to: CGPoint(x: size.width, y: midY))
        ctx.stroke(mid, with: .color(Theme.Palette.hairline), lineWidth: 1)

        let joints = live.pos
        let vel = live.vel, tor = live.tor, temp = live.temp
        var groupX: [ModuleFocus: (CGFloat, CGFloat)] = [:]

        for b in bars {
            let bx = b.x0 + (b.x1 - b.x0) * 0.15
            let bws = max(1.5, (b.x1 - b.x0) * 0.7)
            let value: Double?
            switch metric {
            case .pos:    value = joints[b.spec.id]
            case .vel:    value = vel[b.spec.id]
            case .torque: value = tor[b.spec.id]
            }
            if let v = value {
                let bh = CGFloat(norm(v, spec: b.spec)) * amp
                let rect = bh >= 0
                    ? CGRect(x: bx, y: midY - bh, width: bws, height: max(1, bh))
                    : CGRect(x: bx, y: midY, width: bws, height: max(1, -bh))
                ctx.fill(Path(rect), with: .color(JointGraphs.hue(b.module).opacity(0.9)))
            } else {
                // no data for this metric yet — a faint stub, never a fake bar
                ctx.fill(Path(CGRect(x: bx, y: midY - 1, width: bws, height: 2)),
                         with: .color(Theme.Palette.textTertiary.opacity(0.25)))
            }
            // temperature cell (discrete grade, web parity; faint = unmeasured)
            let t = temp[b.spec.id]
            let tCol: Color = t == nil ? Theme.Palette.textTertiary.opacity(0.12)
                : t! >= 70 ? Self.statusCols[2] : t! >= 55 ? Self.statusCols[1] : Self.statusCols[0]
            ctx.fill(Path(CGRect(x: bx, y: size.height - labH - tempH, width: bws, height: tempH - 1)),
                     with: .color(tCol))
            if pinned == b.spec.id {
                ctx.stroke(Path(CGRect(x: b.x0 + 0.5, y: 2,
                                       width: (b.x1 - b.x0) - 1,
                                       height: size.height - labH - 4)),
                           with: .color(Color(hex: 0xF0B68B).opacity(0.9)), lineWidth: 1)
            }
            if var g = groupX[b.module] { g.1 = b.x1; groupX[b.module] = g }
            else { groupX[b.module] = (b.x0, b.x1) }
        }

        for m in modules {
            guard let (x0, x1) = groupX[m] else { continue }
            ctx.draw(Text(m.title.uppercased())
                        .font(.system(size: 7, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textTertiary),
                     at: CGPoint(x: (x0 + x1) / 2, y: size.height - labH / 2 - 1),
                     anchor: .center)
        }
    }
}

struct RobotSpaceView: View {
    /// Landscape is fully immersive (no nav rail) — this leaves the Robot space.
    var onBack: (() -> Void)? = nil
    @EnvironmentObject private var controller: RobotController
    @Environment(\.bottomBarInset) private var bottomBarInset
    @Environment(\.railLeadingInset) private var railLeadingInset
    @Environment(\.isLandscape) private var isLandscape

    // Map source: live SLAM or a saved map streamed back from the robot.
    @State private var selected: SLAMRoom = .live
    @State private var showSavedMaps = false

    // Viewport camera controls.
    @State private var recenterToken = 0
    @State private var fitToken = 0
    @State private var following = false

    // 3rd ↔ 1st person. HELD ON THE CONTROLLER (not @State): rotation swaps the
    // portrait/landscape view trees, and per-view state would snap an operator
    // in 1st person back to 3rd on every flip. Same pattern as collectingData.
    private var pov: PovMode {
        get { controller.povMode }
        nonmutating set { controller.povMode = newValue }
    }
    // Video gates live in the controller (survive rotation) — see RobotController.
    private var fpvVisible: Bool {
        get { controller.fpvVisible }
        nonmutating set { controller.fpvVisible = newValue }
    }
    private var fpvLive: Bool {
        get { controller.fpvLive }
        nonmutating set { controller.fpvLive = newValue }
    }
    @State private var povLooking = false     // 1st-person look-around off-axis
    /// Bumped on every teleop-link recovery: forces the Cam 3D projections to
    /// re-attach (fresh MJPEG sources) and bounces the camera-cloud links —
    /// after a scene switch the sim stack is a NEW process on the SAME URLs,
    /// and half-dead sockets from the old one otherwise linger (frozen video
    /// texture / missing pointclouds).
    @State private var connectionEpoch = 0
    @State private var lastStreamBounce = Date.distantPast
    /// Physics-view takeover (MuJoCo viewer replaces the 3rd-person render).
    /// Mode + camera live on the CONTROLLER (rotation re-creates this view);
    /// only the fade animation state is local, re-seeded on appear.
    private var physicsOn: Bool { controller.physicsViewOn }
    private var physicsCam: PhysicsCam { controller.physicsViewCam }
    @State private var physicsFaded = false

    // Teleop. Drive (base) is LANDSCAPE-ONLY; Joint / Task work everywhere.
    @State private var teleopActive = false
    @State private var space: TeleopSpace = .drive
    @State private var feel: ControllerType = .impedance   // Stiff | Soft
    @State private var translate: CGVector = .zero
    @State private var yaw: Double = 0
    @State private var joyL: CGVector = .zero
    @State private var joyR: CGVector = .zero
    @State private var pitchL: Double = 0
    @State private var pitchR: Double = 0
    @State private var gripL: Double = 0
    @State private var gripR: Double = 0
    // Joint-space module focus: which module the controls (and the camera) are on.
    @State private var module: ModuleFocus = .armRight
    @State private var focusToken = 0
    /// Control REGION for task space + Soft mode: arm only / upper body (+torso+lift)
    /// / whole body (+mobile base). The controller + server already implement all
    /// three; this picks which. Also scopes what the plan preview animates.
    @State private var region: ControlRegion = .arm
    /// Control-tuning sheet (gravity comp · retargeting profile). A SHEET rather
    /// than more toolbar chrome: this row already carries the space picker, the
    /// primary Start/Stop verb and the Stiff|Soft law, and two more controls
    /// there would collide with the centred button in portrait.
    /// WHICH modal is up. This is ONE piece of state driving ONE `.sheet(item:)`
    /// at the root of the view, replacing four separate `.sheet(isPresented:)`
    /// modifiers that were chained onto `recButton`. That was broken two ways:
    ///   * `recButton` is conditional (`if controller.connection.isLive`), so with
    ///     the robot offline the modifiers were not in the hierarchy at ALL — the
    ///     control-tuning button set its flag and nothing ever presented, which
    ///     reads as "the button does nothing".
    ///   * four `.sheet` on one view: SwiftUI honours one, so even live the
    ///     tuning sheet lost to the ones applied after it.
    /// A single source of truth cannot regress into either failure.
    enum ActiveSheet: String, Identifiable {
        case console
        var id: String { rawValue }
    }
    @State private var activeSheet: ActiveSheet?
    /// THE RULE (2026-07-17): torque OFF → this is a VIEWER. torque ON → it is a
    /// controller.
    ///
    /// With the e-stop engaged the motors are dead, so every control surface —
    /// Start Teleop, Stiff/Soft, Drive·Joint·Task, the joysticks, the module and
    /// region rails, GO, Execute — is a button that cannot do the thing it depicts.
    /// Showing them anyway teaches the operator that this app's controls sometimes
    /// lie. What stays is everything that OBSERVES: SLAM and the map, the camera
    /// views and layers, telemetry, the minimap, the POV toggle, recording. Press
    /// TORQUE ON and the controls come back.
    private var torqueOn: Bool { !controller.estopEngaged }

    /// PRESS to talk — press once to open the mic, press again to close it
    /// (2026-07-28, operator request: holding a button while you talk to someone
    /// standing in front of the robot costs you the hand you were driving with).
    /// The mic is therefore latched, so it is closed for you on every path where
    /// the operator cannot see the button any more: link loss, voice plane going
    /// away, and the view leaving the screen (see `stopTalkingIfNeeded`).
    @State private var talking = false
    /// The `:9610` voice plane (owned by the controller so it survives the view-tree
    /// swap on rotation, like every other link).
    private var voice: AudioStreamLink { controller.voice }
    /// Teleop-space auto-framing (Start Teleop / space switch) — see frameForSpace().
    @State private var spaceFrame: SpaceFrame? = nil
    @State private var spaceFrameToken = 0
    /// A staged green-ball arm plan (nil = none). Drives the Execute / Cancel pills;
    /// the viewer coordinator owns the actual trajectory + preview animation.
    @State private var armPlan: ArmPlanPreview? = nil
    @State private var executePlanToken = 0
    @State private var cancelPlanToken = 0
    /// Bumped to drop a placed-but-never-dispatched nav goal marker.
    @State private var clearGoalToken = 0
    @State private var debugStageToken = 0
    @State private var debugStageSide = "left"
    private let tick = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    // Pull-up console + the visualization state it controls.
    @State private var showConsole = false
    /// Portrait console detent — a selection binding so the DEBUG telemetry
    /// harness can boot the sheet fully open (synthetic drags don't land on
    /// the Simulator).
    @State private var consoleDetent: PresentationDetent = {
        #if DEBUG
        if UserDefaults.standard.bool(forKey: "debugTelemetryOnly") { return consoleFullDetent }
        // debugConsoleDetent=mid|full — boot the console sheet taller.
        switch UserDefaults.standard.string(forKey: "debugConsoleDetent") {
        case "mid":  return .fraction(0.7)
        case "full": return consoleFullDetent
        default: break
        }
        #endif
        return .fraction(0.28)
    }()
    @State private var consoleExpanded = false     // landscape panel: 60% ↔ full
    /// Tap-to-expand camera inspector: the tile the operator tapped (full-screen
    /// live view with pinch-zoom + pan), and whether the portrait console sheet
    /// was auto-closed to make room for it (restored when the inspector closes).
    @State private var expandedCamera: ExpandedCamera? = nil
    @State private var restoreConsoleAfterCamera = false
    // The offline standby card is TRANSIENT — it announces, then gets out of
    // the way (map selection stays one tap away on the pill / in the console).
    @State private var standbyDismissed = false
    // What the viewer DRAWS lives in a shared, persisted store — NOT in @State.
    // Rotating the phone gives this view a new identity (ContentView swaps its
    // portrait/landscape branches), and @State would reset every layer toggle,
    // density and projection setting at exactly the wrong moment: you rotate to
    // landscape because you are about to drive. See ViewerSettings.
    @ObservedObject private var viz = ViewerSettings.shared

    /// The per-camera streams the console can toggle. Projections list every
    /// camera; the POINTCLOUD row is head + base-front only (operator picture —
    /// the wrist clouds are manipulation detail, not navigation context).
    private static let camStreamOptions: [(key: String, label: String)] = [
        ("main", "Head"), ("front", "Front"),
        ("wrist_left", "L Wrist"), ("wrist_right", "R Wrist"),
    ]
    private static let camCloudOptions: [(key: String, label: String)] = [
        ("main", "Head"), ("front", "Front"),
    ]

    /// The density segmented control speaks `CloudDensity`; the persisted store
    /// keeps the plain Int stride (an enum would need custom coding for no gain).
    /// This bridges the two, so the choice survives a rotation like every other
    /// viewer setting.
    private var cloudDensityBinding: Binding<CloudDensity> {
        Binding(get: { CloudDensity(rawValue: self.viz.cloudStride) ?? .full },
                set: { self.viz.cloudStride = $0.rawValue })
    }

    /// Does this robot advertise the dedicated per-camera cloud planes (96xx)?
    ///
    /// This decides who draws the front camera. The /map plane's "live sensor
    /// cloud" (CLOUD 0x05 on :9092) IS the front camera's cloud on BOTH rigs —
    /// the real bridge subscribes `/camera/front/depth/points` for it
    /// (MABEL_MAP_CLOUD_TOPIC), and the sim bridge forwards the :9604 plane into
    /// it. So whenever the dedicated planes exist we draw the cameras from
    /// THOSE (Head / Front chips) and hide the map cloud — drawing both painted
    /// the front camera TWICE, which is what made the Cloud layer look like a
    /// third, redundant "front base" cloud. The map cloud stays as the FALLBACK
    /// for a rig with no 96xx planes (demo mode, an older bridge), so the layer
    /// is never dead.
    private var hasCamCloudPlanes: Bool {
        Self.camCloudOptions.contains { controller.cameraCloudURL($0.key) != nil }
    }

    /// Teleop control space (which controls the joysticks / gestures drive).
    enum TeleopSpace: String, CaseIterable, Identifiable {
        case drive, joint, task
        var id: String { rawValue }
        var title: String {
            switch self {
            case .drive: return "Drive"
            case .joint: return "Joint"
            case .task:  return "Task"
            }
        }
        var icon: String {
            switch self {
            case .drive: return "dot.arrowtriangles.up.right.down.left.circle"
            case .joint: return "slider.horizontal.3"
            case .task:  return "hand.draw.fill"
            }
        }
    }

    /// Cam 3D image surface: rectified pinhole plane vs through-the-lens shell.
    enum ProjSurface: String, CaseIterable, Identifiable {
        case rectified, lens
        var id: String { rawValue }
        var title: String { self == .rectified ? "Rectified" : "Lens Curve" }
        var icon: String { self == .rectified ? "rectangle" : "circle.and.line.horizontal" }
    }

    /// Live-cloud render density (every Nth point).
    enum CloudDensity: Int, CaseIterable, Identifiable {
        case full = 1, half = 2, quarter = 4, tenth = 10
        var id: Int { rawValue }
        var title: String {
            switch self {
            case .full: return "Full"
            case .half: return "1/2"
            case .quarter: return "1/4"
            case .tenth: return "1/10"
            }
        }
    }

    private var store: LiveMapStore { controller.liveMap }
    /// Goal flow: tapping the floor PLACES the marker (aim it with the ring),
    /// the GO button dispatches it (Nav2 or the demo robot), and GO becomes
    /// STOP while the route is active. With no robot and no demo map loaded,
    /// the map is view-only.
    private enum GoalState { case none, placed, navigating }
    @State private var goalState: GoalState = .none
    @State private var pendingGoal: (x: Double, y: Double, yaw: Double)? = nil

    private var goalHandler: ((Double, Double, Double) -> Void)? {
        guard store.demo || (store.connected && store.hasEndpoint) else { return nil }
        return { x, y, yaw in
            pendingGoal = (x, y, yaw)
            withAnimation(Theme.Motion.snappy) { goalState = .placed }
        }
    }

    private func dispatchGoal() {
        guard let g = pendingGoal else { return }
        if store.demo { store.setDemoGoal(x: g.x, y: g.y, yaw: g.yaw) }
        else { controller.sendNavGoal(x: g.x, y: g.y, yaw: g.yaw) }
        withAnimation(Theme.Motion.snappy) { goalState = .navigating }
        Haptics.rigid()
    }

    private func stopNav() {
        if store.demo { store.cancelDemoGoal() }
        else if let o = store.lastOdom {
            // Pragmatic cancel on the wire: re-target the robot's own pose —
            // Nav2 replans to "here" and stops.
            controller.sendNavGoal(x: o.x, y: o.y, yaw: o.yaw)
        }
        pendingGoal = nil
        withAnimation(Theme.Motion.snappy) { goalState = .none }
        Haptics.rigid()
    }

    /// GO (placed) / STOP (navigating) — a compact pill riding WITH the camera
    /// pills at the bottom (same metrics as ViewportButton), not a hero button
    /// floating mid-screen. Colour carries the state: green GO / red STOP.
    @ViewBuilder private var goalButton: some View {
        if goalState != .none {
            HStack(spacing: 6) {
                // A PLACED goal is a staged intent, exactly like a staged arm plan,
                // so it gets the same two-button shape: commit or discard. Before
                // this the only way out of a mis-tapped goal was to dispatch it and
                // then stop the robot — the marker had no cancel at all.
                // A goal already GOING keeps only STOP: that is live motion, not a
                // staged intent.
                if goalState == .placed {
                    Button { clearGoal() } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .heavy))
                            .foregroundStyle(.white)
                            .frame(width: 34, height: 34)
                            .background(Theme.Palette.danger.gradient, in: Circle())
                            .overlay { Circle().strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                            .contentShape(Circle())
                    }
                    .buttonStyle(PressableStyle())
                }
                Button {
                    if goalState == .placed { dispatchGoal() } else { stopNav() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: goalState == .placed ? "play.fill" : "stop.fill")
                            .font(.system(size: 12, weight: .bold))
                        Text(goalState == .placed ? "GO" : "STOP NAV")
                            .font(.system(size: 12, weight: .heavy)).tracking(0.4)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .frame(minHeight: 34)
                    .background((goalState == .placed ? Theme.Palette.live : Theme.Palette.danger).gradient,
                                in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                    .shadow(color: (goalState == .placed ? Theme.Palette.live : Theme.Palette.danger).opacity(0.4),
                            radius: 6, y: 2)
                    .contentShape(Capsule())
                }
                .buttonStyle(PressableStyle())
            }
            .transition(.scale.combined(with: .opacity))
        }
    }

    /// Discard a PLACED (never dispatched) goal: nothing was sent, so this only
    /// drops the marker and the pending pose.
    private func clearGoal() {
        Haptics.tap()
        pendingGoal = nil
        if controller.liveMap.demo { controller.liveMap.cancelDemoGoal() }
        clearGoalToken += 1                        // drop the SceneKit marker
        navState.setGoal(nil)                      // and the 2D minimap's
        withAnimation(Theme.Motion.snappy) { goalState = .none }
    }

    /// The green-ball plan review: EXECUTE (commit the previewed motion) + CANCEL.
    /// Mirrors the nav GO/STOP pill. Shown only while a plan is staged; the arm is
    /// meanwhile animating the preview loop in the viewer. Red when the self-
    /// collision heuristic flagged the plan — you can still execute, but you've been
    /// told. This is the whole point of the feature: nothing moved the robot yet.
    @ViewBuilder private var planButtons: some View {
        if let plan = armPlan {
            HStack(spacing: 8) {
                Button { cancelPlanToken += 1; Haptics.tap() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .heavy))
                        .foregroundStyle(.white)
                        .frame(width: 34, height: 34)
                        .background(Theme.Palette.danger.gradient, in: Circle())
                        .overlay { Circle().strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                        .contentShape(Circle())
                }
                .buttonStyle(PressableStyle())

                Button { executePlanToken += 1; Haptics.rigid() } label: {
                    HStack(spacing: 6) {
                        Image(systemName: plan.collision ? "exclamationmark.triangle.fill" : "play.fill")
                            .font(.system(size: 13, weight: .bold))
                        Text("Execute")
                            .font(.system(size: 13, weight: .heavy)).tracking(0.4)
                        Text(plan.collision ? plan.reason
                             : (plan.side == "both" ? "both arms"
                                : plan.side == "right" ? "R arm" : "L arm"))
                            .font(.system(size: 9.5, weight: .semibold))
                            .opacity(0.85)
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .frame(minHeight: 34)
                    .background((plan.collision ? Theme.Palette.warn : Theme.Palette.live).gradient,
                                in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.28), lineWidth: 1) }
                    .shadow(color: (plan.collision ? Theme.Palette.warn : Theme.Palette.live).opacity(0.4),
                            radius: 6, y: 2)
                    .contentShape(Capsule())
                }
                .buttonStyle(PressableStyle())
            }
            .transition(.scale.combined(with: .opacity))
        }
    }

    /// SAFETY bands around the floating chrome — a tap that just misses a button
    /// must never plant a nav goal on the map behind it.
    private var goalExclusion: UIEdgeInsets {
        UIEdgeInsets(top: isLandscape ? 130 : 170,
                     left: isLandscape ? max(railLeadingInset, 90) : 12,
                     bottom: (isLandscape ? 100 : 170) + (teleopActive ? 140 : 0),
                     right: isLandscape ? 90 : 12)
    }

    /// Connected: ONLY the maps the robot actually advertises (`list_maps` reads
    /// the ROS workspace's saved SLAM maps) — an empty list stays EMPTY, never
    /// padded with fakes. Offline: the local dummy maps, labeled as such.
    private var connectedToRobot: Bool { store.hasEndpoint || controller.connection.isLive }
    private var savedRooms: [SLAMRoom] {
        if connectedToRobot {
            return controller.savedMaps.map { SLAMRoom.streamed(name: $0.name) }
        }
        return SLAMRoom.library
    }
    private var mapSectionTitle: String { connectedToRobot ? "SAVED MAPS · ROBOT" : "LOCAL DUMMY MAPS" }

    var body: some View {
        ZStack(alignment: .top) {
            // THE unified viewer — everything else floats over it.
            LiveMapView(store: store,
                        pose: controller.livePose,
                        navState: viz.minimapOn ? navState : nil,
                        recenterToken: recenterToken,
                        fitToken: fitToken,
                        follow: following,
                        pov: pov,
                        camProjections: viz.camProjections,
                        camStreams: viz.camStreams,
                        camClouds: viz.cloudOn ? viz.camClouds : [],
                        // Head plane rides the SAME stream as the 1st-person FPV
                        // (main quality) so the POV handover is byte-identical;
                        // the small wrist/front planes stay on cheap thumbs.
                        camVideoURL: { controller.videoURL($0) },
                        camRouteToken: "\(controller.videoURL("main")?.absoluteString ?? "offline")#\(connectionEpoch)",
                        meshHidden: !viz.meshOn,
                        cloudHidden: !viz.cloudOn || hasCamCloudPlanes,
                        gridHidden: !viz.gridOn,
                        robotHidden: !viz.robotOn,
                        floorHidden: !viz.floorOn,
                        floorGridHidden: !viz.floorGridOn,
                        renderFps: viewerFps,
                        tfHidden: !viz.tfOn,
                        // The sweep is its own layer here (VP couples it to Cloud):
                        // it is the one sensor every plant streams, so it stays
                        // available when the depth cloud is off or idle.
                        lidarScan: viz.laserOn ? controller.lidarScan : nil,
                        cloudStride: viz.cloudStride,
                        camProjSize: viz.camProjSize,
                        camFlat: viz.camFlat,
                        // Soft (compliance) is a FLOAT mode: the space controls +
                        // joysticks disappear and the ONLY control is dragging the
                        // back-drivable arms directly — so manipulation is armed in
                        // Soft regardless of space, and always in Task under Stiff.
                        // TASK SPACE ONLY. The green palm balls and wrist rings are
                        // the task-space control surface; the `|| feel == .compliance`
                        // that used to be here put them on screen in Drive and Joint
                        // space too, any time the feel toggle was Soft. That is three
                        // grabbable objects floating over a robot whose sticks are
                        // driving something else entirely — the operator reads them as
                        // live controls, and in Drive they are simply in the way.
                        // Soft-mode pushes go with the surface that offers them.
                        manipulate: teleopActive && space == .task,
                        forceMode: feel == .compliance,
                        // Plan/preview/execute is the green-ball safety gate: ON in
                        // Task + Stiff + 3rd person (where you drag the ball). In Soft
                        // the grab is a compliant push, not an IK target, so no plan.
                        planMode: teleopActive && space == .task
                                  && feel == .impedance && pov == .third,
                        planRegion: region,
                        onPlanChange: { plan in
                            withAnimation(Theme.Motion.snappy) { armPlan = plan }
                        },
                        onBaseGoal: { x, y, yaw in controller.sendNavGoal(x: x, y: y, yaw: yaw) },
                        executePlanToken: executePlanToken,
                        cancelPlanToken: cancelPlanToken,
                        clearGoalToken: clearGoalToken,
                        debugStageToken: debugStageToken,
                        debugStageSide: debugStageSide,
                        onJointsChange: { controller.setJoints($0) },
                        onJointChange: { controller.setJoint($0, $1) },
                        onPartForce: { joint, f in
                            if f == [0, 0, 0] { controller.clearExternalForce() }
                            else { controller.applyExternalForce([joint: f]) }
                        },
                        wristDrive: controller.wristDrive,
                        onPovLook: { povLooking = $0 },
                        physicsActive: physicsOn,
                        physicsCam: physicsCam,
                        onGoal: goalHandler,
                        goalExclusion: goalExclusion,
                        focusModule: module,
                        focusToken: focusToken,
                        spaceFrame: spaceFrame,
                        spaceFrameToken: spaceFrameToken)
                .rebuildsOnRobotRigChange()   // new robot / swapped hands → rebuild the scene
                .ignoresSafeArea()

            // PHYSICS VIEW takeover: the MuJoCo viewer replaces the 3rd-person
            // visualizer. Mounted (connecting, invisible) the moment the toggle
            // engages so the stream warms up DURING the cinematic pan; the
            // crossfade fires from the overlay once the arc completes and
            // frames are fresh. Teleop chrome stays above and keeps working.
            if physicsOn, pov == .third, let ws = controller.simStudio.viewerWS {
                PhysicsViewerOverlay(url: ws, cam: physicsCam, faded: $physicsFaded)
                    .opacity(physicsFaded ? 1 : 0)
                    .allowsHitTesting(physicsFaded)
                    .transition(.opacity)
            }

            // 1st-person LIVE video — the robot's real head camera, faded in over
            // the matching 3D view once the camera flight lands. In demo (or with
            // the stream dead) the pure 3D first-person render stays visible.
            if pov == .first, controller.connection.isLive,
               let fpvURL = controller.videoURL("main") {
                MJPEGView(url: fpvURL, onLive: { live in
                        withAnimation(.easeInOut(duration: 0.45)) { fpvLive = live }
                    }, streamKey: "main")
                    .ignoresSafeArea()
                    .opacity(fpvVisible && fpvLive && !povLooking ? 1 : 0)
                    .allowsHitTesting(false)   // gestures reach the 3D view (look-around)
                    .animation(.easeInOut(duration: 0.45), value: fpvVisible && fpvLive && !povLooking)
                // The v1 cockpit's glance chrome over the FPV: the two wrist
                // cams top-left. (The 3D twin rides the right-hand column with
                // the minimap — see below — so the two never overlap.)
                if fpvChromeOn { fpvChrome }
            }

            // 1st person, connected, flight landed — but NO live head image. The
            // app used to just sit on the 3D render, which is indistinguishable
            // from "the transition didn't work": you cannot tell a dead camera from
            // a broken app. Say which it is. (Demo has no video by design and is
            // excluded — `connection.isLive` is false there.)
            if pov == .first, controller.connection.isLive, fpvVisible, !povLooking,
               controller.videoURL("main") == nil || !fpvLive {
                headCamOfflinePill
                    .transition(.opacity)
            }

            // Centred standby card — only while the scene is truly EMPTY and
            // we're overlooking the map in 3rd person. TRANSIENT: it fades out
            // after a few seconds so it never squats over the robot.
            if pov == .third, !store.demo, !store.hasEndpoint || !store.connected,
               store.vertexCount == 0, !store.hasPose, !standbyDismissed {
                standby
                    .task {
                        try? await Task.sleep(nanoseconds: 6_000_000_000)
                        withAnimation(.easeOut(duration: 0.5)) { standbyDismissed = true }
                    }
                    .transition(.opacity)
            }

            VStack(spacing: 8) {
                topControls
                // GO / STOP for the placed nav goal — dead centre, right under
                // the Start Teleop button. Both of these COMMIT motion, so they
                // follow the torque rule (see `torqueOn`).
                if torqueOn { goalButton }
                // EXECUTE / CANCEL for a staged green-ball arm plan (same slot).
                if torqueOn { planButtons }
                // Recording is not a property of the CAMERA. It was gated on
                // 1st person (the old app's cockpit Record), which meant a demo
                // could not be started or stopped from the 3rd-person view the
                // operator actually stages a performance in — and the STOP
                // button vanished if they flipped POV mid-take.
                if controller.connection.isLive { recButton }
                // "Driving works here too" — only where driving is a thing. With
                // torque off the robot cannot be driven in ANY orientation.
                if torqueOn, !isLandscape, !teleopActive || space == .drive,
                   goalState == .none { rotateHint }
                // Task-space coaching, OR the Soft-mode float hint (Soft has no space,
                // so it shows regardless of which space was last selected).
                if torqueOn, teleopActive, space == .task || feel == .compliance { taskHint }
                if store.demo { demoBanner }
                // Floating 2D map — top-right, below the controls. Over live FPV
                // video an EMPTY minimap is just a dark vignette blob — skip it
                // until the SLAM grid actually has content.
                if viz.minimapOn && !(fpvChromeOn && navState.liveRaster == nil) {
                    HStack(spacing: 0) {
                        Spacer(minLength: 0)
                        MiniMap2D(state: navState)
                            // Landscape: smaller so the right-edge control panel
                            // never has to share space with it.
                            .frame(width: isLandscape ? 112 : nil,
                                   height: isLandscape ? 112 : nil)
                            .transition(.opacity)
                    }
                    .padding(.trailing, 6)
                }
                // FPV: the live 3D digital twin (v1 cockpit parity) — stacked
                // UNDER the minimap in the same right column so they can never
                // overlap. Pure monitor chrome; touches fall through.
                if fpvChromeOn {
                    HStack(spacing: 0) {
                        Spacer(minLength: 0)
                        fpvModelTile
                    }
                    .padding(.trailing, 6)
                    .transition(.opacity)
                }
                Spacer(minLength: 0)
            }
            // Immersive landscape reserves the status bar via a safe-area inset
            // (ContentView), so no extra clearance is needed in either orientation.
            .padding(.leading, isLandscape ? 4 : 0)
        }
        // Compact left rail — pinned UPPER-left so it clears the bottom panel.
        // Joint space → module picker; Task/Soft → control-region picker.
        .overlay(alignment: .topLeading) {
            Group {
                if torqueOn, teleopActive, space == .joint {
                    moduleRail
                } else if torqueOn, teleopActive, space == .task || feel == .compliance {
                    regionRail
                }
            }
            .padding(.leading, 8)
            .padding(.top, isLandscape ? 64 : 138)
            .transition(.move(edge: .leading).combined(with: .opacity))
        }
        // Teleop clusters flank the centred pills (Task = the classic dual wrist
        // sticks + up/down + grips, on top of in-viewer grab-IK).
        .overlay(alignment: .bottom) {
            // Joysticks only in Stiff. Soft is drag-only (the arms float), so the
            // whole stick cluster disappears — the operator moves the robot by hand.
            // And never with the motors dead: a stick you can push that moves
            // nothing is the worst control in the app. (See `torqueOn`.) This also
            // takes a whole GeometryReader + the live stick gestures out of the
            // tree while the robot is parked.
            if torqueOn, teleopActive, feel == .impedance {
                GeometryReader { geo in
                    teleopControls(height: geo.size.height)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        // The camera-pill row goes LAST → topmost, so the Console grabber, the
        // POV toggle and the GO/STOP pill (it rides in this stack) are always
        // tappable above every other overlay.
        .overlay(alignment: .bottom) { viewportPills }
        // Torque dropped → teleop is OVER, not paused. Without this, teleopActive
        // survives the e-stop with stale space/stick state behind the hidden
        // controls, and TORQUE ON would restore a session the operator had already
        // abandoned — mid-drag, in whatever mode they panicked out of.
        .onChange(of: controller.estopEngaged) { _, engaged in
            guard engaged else { return }
            if teleopActive { stopTeleop() }
            if goalState == .placed { clearGoal() }     // a staged goal is void too
        }
        .onChange(of: store.navRouteActive) { _, active in
            // Route emptied server-side (goal reached / cancelled) → reset.
            if !active && goalState == .navigating {
                withAnimation(Theme.Motion.snappy) { goalState = .none }
            }
        }
        // Per-camera clouds follow the toggles AND the link (connect / failover
        // re-resolves the 96xx URLs; disconnect closes them).
        .onChange(of: viz.camClouds) { _, _ in syncCameraClouds() }
        .onChange(of: viz.cloudOn) { _, _ in syncCameraClouds() }
        .onChange(of: controller.refitToken) { _, _ in
            // Sim reset: re-frame every viewer on the robot (3rd person recenter;
            // 1st person look-reset; physics cam already refit). The RANDOM respawn
            // lands a moment LATER (the bridge processes reset async, then the new
            // odom pose streams in), so re-frame NOW and again after it settles —
            // otherwise the camera frames the OLD spot and the robot leaves the view.
            recenterToken += 1
            for delay in [0.5, 1.1] {
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) { recenterToken += 1 }
            }
        }
        .onChange(of: controller.connection.isLive) { _, live in
            guard live else { return }
            // DEBOUNCE: a heavily-loaded sim (huge scene → slow control loop)
            // can flap liveness every few seconds; bouncing the camera links +
            // re-attaching Cam 3D on every flap machine-guns the streams into
            // a permanent connect/disconnect churn. Only bounce when the link
            // has actually been quiet for a while.
            let now = Date()
            guard now.timeIntervalSince(lastStreamBounce) > 8 else { return }
            lastStreamBounce = now
            connectionEpoch += 1
            syncCameraClouds()
            store.bounceCameraClouds()
        }
        .onChange(of: controller.connection) { _, _ in
            syncCameraClouds()
            // A LATCHED mic (press-to-talk) must not survive the link that carries
            // it. Losing the link takes the lit button off screen with it, and a
            // capture nobody can see is the failure mode hold-to-talk existed to
            // prevent — so drop it here rather than trusting the operator to notice.
            if !controller.connection.isLive || !controller.audioAvailable {
                stopTalkingIfNeeded()
            }
        }
        // hello.ports can arrive AFTER the connection flips (or refresh on a
        // failover) — re-resolve the cloud URLs whenever the meta changes.
        .onChange(of: controller.serverMeta) { _, _ in
            syncCameraClouds()
            // `hello.ports` (and therefore ports["audio"]) can arrive AFTER the
            // connection flips — re-resolve or the talk button stays dark on a
            // perfectly good link.
            controller.refreshAudioEndpoint()
        }
        // The link is the TRUTH about the mic, and it can now change WITHOUT a
        // press: the first-ever Talk raises the system permission prompt and the
        // capture only opens when the answer comes back, and any failure path
        // (denied, dead capture, heal that could not recover) drops the latch on
        // its own. Mirroring it here is what stops the button from claiming
        // "Talking…" over a mic that is not open — the exact fakery the scope was
        // added to expose.
        .onChange(of: voice.talking) { _, isTalking in talking = isTalking }
        .onReceive(tick) { _ in onTick() }
        // Reaching PAST the console gets rid of it (2026-07-17). The sheet allows
        // background interaction below 70%, so a tap on the viewer or on Start
        // Teleop / TORQUE ON already works THROUGH it — but the console then sat
        // there covering the robot you just told to move. Touching anything
        // underneath means you are done with the console, so it leaves.
        //
        // simultaneousGesture, so it never swallows the tap it rides along with:
        // the button still fires, the goal is still placed. TapGesture needs zero
        // movement, so orbiting/dragging the scene doesn't trigger it either.
        // PORTRAIT ONLY. The portrait console is a separate SHEET presentation, so
        // this gesture only ever sees taps on the view UNDERNEATH it — exactly the
        // "I'm reaching past the console" signal. The LANDSCAPE console is an
        // in-tree overlay, so the same gesture would fire when you press its own
        // chips and close it under your finger.
        .simultaneousGesture(
            TapGesture().onEnded {
                if showConsole, !isLandscape {
                    withAnimation(Theme.Motion.snappy) { showConsole = false }
                }
            }
        )
        // ONE presenter for every sheet in this view. The portrait console rides
        // the same enum as the tuning / recording sheets: SwiftUI honours a single
        // `.sheet` per view, so adding a second modifier here would silently kill
        // one of them (which is exactly how the control-tuning sheet died).
        .sheet(item: Binding(get: { showConsole && !isLandscape ? .console : activeSheet },
                             set: { newValue in
                                 if newValue == nil, showConsole, !isLandscape { showConsole = false }
                                 activeSheet = (newValue == .console) ? nil : newValue
                             })) { which in
            switch which {
            case .console:      consoleSheet
            }
        }
        // Full-screen live inspector for a tapped camera tile. ONE presenter at
        // the root: `expandCamera` closes the portrait console sheet first (two
        // sibling presentations can't coexist) and restores it on dismissal.
        .fullScreenCover(item: $expandedCamera, onDismiss: {
            if restoreConsoleAfterCamera {
                restoreConsoleAfterCamera = false
                withAnimation(Theme.Motion.snappy) { showConsole = true }
            }
        }) { cam in
            CameraDetailView(camera: cam, controller: controller)
        }
        // Landscape console: an in-view 60%-height panel (sheets go full-screen
        // in compact height, which would bury the 3D viewer).
        .overlay {
            if showConsole && isLandscape { landscapeConsole }
        }
        // Record / Replay cards — small centred windows over EVERYTHING (the
        // STOP button itself lives in the top bar, where Record was).
        .overlay { recordCardOverlay }
        .animation(Theme.Motion.snappy, value: recordCard)
        .background(Theme.Palette.canvas)
        .onAppear { onRobotSpaceAppear() }
        .onChange(of: selected) { _, _ in controller.liveMap.active = true }
        .onDisappear {
            controller.liveMap.active = false
            if teleopActive { stopTeleop() }
            stopTalkingIfNeeded()      // a latched mic never outlives the Robot space
        }
    }

    /// Top-down 2D map mirror — plain @State (NOT @StateObject) so its 20 Hz
    /// pose updates invalidate only the minimap canvas.
    @State private var navState = NavMapState()

    // MARK: - Top controls

    /// Top controls, organized by FUNCTION:
    ///   LEFT  = the WORLD (leave the space · map source)
    ///   CENTER = the ACTION (Start/Stop Teleop — the primary verb, in words)
    ///   RIGHT = the ROBOT'S BEHAVIOUR (Stiff | Soft; e-stop above covers torque)
    /// The control-space picker gets its own centred row while teleop is live.
    private var topControls: some View {
        ZStack {
            HStack(spacing: 8) {
                if isLandscape, let onBack {
                    Button {
                        Haptics.select(); onBack()
                    } label: {
                        Image(systemName: "chevron.left").font(.system(size: 16, weight: .bold))
                            .foregroundStyle(.white).frame(width: 40, height: 40)
                            .background(.black.opacity(0.4), in: Circle())
                            .contentShape(Circle())
                    }
                    .buttonStyle(PressableStyle())
                }
                if teleopActive, torqueOn {
                    // Control space rides LEFT of the centred Stop button — but ONLY
                    // in Stiff. Soft is a float/drag mode with no Drive·Joint·Task
                    // choice, so the picker disappears (a small "Soft · drag" tag
                    // stands in so the row isn't empty).
                    if feel == .impedance { spaceToggle } else { softFloatTag }
                } else {
                    mapSourceMenu
                }
                Spacer(minLength: 0)
                // Stiff | Soft is a CONTROL LAW — meaningless with the motors dead.
                // Tuning stacks UNDER it rather than beside it: this row centres
                // the primary Start/Stop verb in a ZStack, so anything that widens
                // a flanking cluster grows INWARD and slides under that button.
                // Measured in the simulator — adding the tuning button in-line
                // pushed Stiff|Soft clean behind the "Start Teleop" pill.
                // ONE control on this side, on the SAME line as the centred verb and
                // the left cluster. Stiff|Soft used to sit here too — stacked above the
                // tuning button, which gave the row three baselines; in-line it slid
                // under the "Start Teleop" pill in portrait. It now lives INSIDE this
                // menu, so the row is one line and nothing can collide with the verb.
                if torqueOn { tuningButton }
            }
            // The primary action — dead centre, always in words (compact label
            // in portrait so the flanking clusters never collide with it).
            if torqueOn {
                StartTeleopButton(active: teleopActive, compact: !isLandscape && teleopActive) {
                    if teleopActive { stopTeleop() } else { startTeleop() }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 6)
    }

    /// Drive · Joint · Task — all available in both orientations (landscape is
    /// just the roomier drive cockpit). NON-prominent in both: the prominent
    /// metrics made the landscape seg wide enough to slide under the centred
    /// Stop Teleop button; the compact form always clears it.
    private var spaceToggle: some View {
        GlassSegmented(options: TeleopSpace.allCases,
                       selection: Binding(get: { space }, set: { setSpace($0) }),
                       title: { isLandscape ? $0.title : "" }, icon: { $0.icon })
            .fixedSize()
    }

    /// Stands in for the space picker while Soft: names the mode so the row reads
    /// "the robot is floating, drag it" rather than looking like a missing control.
    private var softFloatTag: some View {
        HStack(spacing: 5) {
            Image(systemName: "hand.draw.fill").font(.system(size: 11, weight: .semibold))
            if isLandscape {
                Text("SOFT · DRAG").font(.system(size: 11, weight: .heavy)).tracking(0.4)
            }
        }
        .foregroundStyle(Theme.Palette.warn)
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(Theme.Palette.surface, in: Capsule(style: .continuous))
        .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.warn.opacity(0.4), lineWidth: 1) }
        .fixedSize()
    }

    /// The control-law dropdown: FEEL (Stiff | Soft) plus the two law switches.
    /// A native `Menu`, not a sheet — these are three quick switches, and a
    /// half-screen modal with explanatory paragraphs buried them. Rows are a
    /// noun and a state, nothing else; the icon-only 40 pt button carries a dot
    /// when anything is off its default so the operator can see at a glance that
    /// the robot is not in its normal configuration without opening it.
    ///
    /// Feel lives here rather than beside the button: as an in-line segmented
    /// control it collided with the centred Start/Stop verb in portrait.
    private var tuningButton: some View {
        Menu {
            Picker("Feel", selection: Binding(get: { feel }, set: { setFeel($0) })) {
                Label("Stiff", systemImage: "bolt.horizontal.fill").tag(ControllerType.impedance)
                Label("Soft", systemImage: "wind").tag(ControllerType.compliance)
            }
            .pickerStyle(.inline)
            Divider()
            Toggle(isOn: Binding(get: { controller.gravityComp },
                                 set: { controller.setGravityComp($0) })) {
                Label("Gravity Comp", systemImage: "arrow.down.circle")
            }
            Divider()
            // The arm retargeting PROFILE, named for what each one tracks rather
            // than as "whole-body on/off" — the interesting mode was the OFF state
            // of a negative toggle, which is a poor way to offer a choice.
            // Maps to controller/mabel/teleop_engine/arm.py PROFILES:
            //   Posture = "whole_body" — follows your arm, elbow included.
            //   Wrist   = "wrist"     — palm goes where you point it; the posture
            //                           table resolves the elbow (palm error p90
            //                           4.6 deg -> 1.4 deg, no unconverged flips).
            Picker("Retargeting",
                   selection: Binding(get: { controller.wholeBodyRetarget },
                                      set: { controller.setWholeBodyRetarget($0) })) {
                Label("Posture", systemImage: "figure.stand").tag(true)
                Label("Wrist", systemImage: "hand.point.up.left.fill").tag(false)
            }
            .pickerStyle(.inline)
        } label: {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 40, height: 40)
                    .background(.black.opacity(0.4), in: Circle())
                if !controller.gravityComp || controller.wholeBodyRetarget || feel != .impedance {
                    Circle().fill(Theme.Palette.warn)
                        .frame(width: 9, height: 9)
                        .overlay { Circle().strokeBorder(.black.opacity(0.5), lineWidth: 1) }
                        .offset(x: 1, y: -1)
                }
            }
            .contentShape(Circle())
        }
        .menuOrder(.fixed)
        .accessibilityLabel("Control tuning")
        .accessibilityIdentifier("btn.controlTuning")
    }

    /// Map-source selection — connected: live SLAM / robot-saved maps; offline:
    /// the LOCAL demo map (nvblox-style mesh + grid, goal-driven robot).
    private func selectLive() {
        Haptics.select()
        if store.hasEndpoint { controller.goLiveSLAM() } else { store.stopLocalDemo() }
        withAnimation(Theme.Motion.snappy) { selected = .live }
    }

    private func selectRoom(_ room: SLAMRoom) {
        Haptics.select()
        if store.hasEndpoint { controller.loadMap(room.name) } else { store.startLocalDemo() }
        withAnimation(Theme.Motion.snappy) { selected = room }
    }

    /// Is SLAM actually streaming map data right now? (real records only —
    /// the local demo never counts.)
    private var slamLive: Bool {
        guard let t = store.lastDataAt else { return false }
        return Date().timeIntervalSince(t) < 5
    }

    /// ONE map-source chip: shows the ACTIVE source ("Live SLAM" while SLAM
    /// streams, "Offline Map" when it isn't, or the loaded saved map's name)
    /// and drops down to the live row + every saved map + Restart SLAM.
    private var mapSourceMenu: some View {
        Button {
            Haptics.select(); showSavedMaps.toggle()
        } label: {
            pill(isLandscape ? mapSourceTitle : nil,
                 selected.isLive ? "dot.radiowaves.left.and.right" : "square.stack.3d.up.fill",
                 active: selected.isLive && slamLive, chevron: true)
        }
        .buttonStyle(PressableStyle())
        .popover(isPresented: $showSavedMaps, arrowEdge: .top) {
            savedMapsPicker.presentationCompactAdaptation(.popover)
        }
    }

    private var mapSourceTitle: String {
        selected.isLive ? (slamLive ? "Live SLAM" : "Offline Map") : selected.name
    }

    private var savedMapsPicker: some View {
        VStack(alignment: .leading, spacing: 2) {
            // Live SLAM first — with its actual streaming status.
            Button {
                selectLive(); showSavedMaps = false
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "dot.radiowaves.left.and.right")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(slamLive ? Theme.Palette.live : Theme.Palette.textSecondary)
                        .frame(width: 22)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Live SLAM").font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(Theme.Palette.textPrimary)
                        Text(slamLive ? "streaming" : "offline — no map data")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(slamLive ? Theme.Palette.live : Theme.Palette.textTertiary)
                    }
                    Spacer()
                    if selected.isLive { Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live) }
                }
                .padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Divider().padding(.vertical, 2)
            Text(mapSectionTitle).font(.system(size: 9, weight: .heavy)).tracking(1.2)
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 12).padding(.top, 4).padding(.bottom, 4)
            if savedRooms.isEmpty {
                Text("No saved maps on the robot — run SLAM and save one.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 12).padding(.vertical, 8)
            } else {
                ForEach(savedRooms) { room in savedRow(room) }
            }
            Divider().padding(.vertical, 2)
            // Restart SLAM: fresh map + localization from the CURRENT pose.
            Button {
                Haptics.rigid()
                controller.resetSlam()
                withAnimation(Theme.Motion.snappy) { selected = .live }
                showSavedMaps = false
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.counterclockwise.circle.fill")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.Palette.warn)
                        .frame(width: 22)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Restart SLAM").font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(Theme.Palette.textPrimary)
                        Text("New map + localization from here")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                    Spacer()
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .frame(width: 240)
        .padding(.bottom, 6)
        .presentationBackground(.ultraThinMaterial)
    }

    private func savedRow(_ room: SLAMRoom) -> some View {
        let active = !selected.isLive && selected.name == room.name
        return Button {
            selectRoom(room)
            showSavedMaps = false
        } label: {
            HStack(spacing: 10) {
                Image(systemName: room.icon).font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(active ? Theme.Palette.live : Theme.Palette.textSecondary)
                    .frame(width: 22)
                Text(room.name).font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textPrimary).lineLimit(1)
                Spacer()
                if active { Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.Palette.live) }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(active ? Theme.Palette.accent.opacity(0.10) : .clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Bottom pill row (camera controls + console grabber)

    private var viewportPills: some View {
        // Portrait is width-starved — the pills go icon-only (labels in landscape),
        // but stay LARGE (≥44 pt targets) and consistent with the top controls.
        let t = { (s: String) in isLandscape ? s : "" }
        // Compact pills — visibly SMALLER than the tab bar below, but the hit
        // targets stay ≥44 pt (component min frame + contentShape). The POV
        // toggle rides its own line ABOVE the pills in portrait (compact,
        // icon-only); landscape keeps one row.
        let povSeg = GlassSegmented(options: PovMode.allCases,
                                    selection: Binding(get: { pov }, set: { setPov($0) }),
                                    title: { isLandscape ? $0.title : "" }, icon: { $0.icon })
            .fixedSize()
        // Visualizer ↔ Physics segmented toggle, RIGHT NEXT to the POV segment
        // (sim only, 3rd person only) — same segmented language as 3rd|1st.
        let physicsBtn = Group {
            if pov == .third, controller.simStudio.available {
                GlassSegmented(options: ViewerKind.allCases,
                               selection: Binding(get: { physicsOn ? ViewerKind.physics : .visualizer },
                                                  set: { kind in
                                                      if (kind == .physics) != physicsOn { togglePhysics() }
                                                  }),
                               title: { isLandscape ? $0.title : "" }, icon: { $0.icon })
                    .fixedSize()
            }
        }
        return VStack(spacing: 7) {
            if !isLandscape { HStack(spacing: 7) { povSeg; physicsBtn } }
            HStack(spacing: 7) {
                if pov == .first {
                    // 1st person: the pill is the HEAD-TRACKING toggle — the
                    // phone's attitude drives the robot's neck (incremental
                    // from engage; off → the head glides back to neutral).
                    // Engaging also recenters the drag look-around, so the
                    // view and the real head agree.
                    ViewportButton(title: t("Head Track"), systemImage: "faceid",
                                   active: controller.headTracking, compact: true) {
                        recenterToken += 1          // drag look-around back on-axis
                        controller.toggleHeadTracking()
                    }
                } else {
                    ViewportButton(title: t("Re-center"), systemImage: "scope", compact: true) { recenterToken += 1 }
                    ViewportButton(title: t("Fit"), systemImage: "arrow.up.left.and.arrow.down.right", compact: true) { fitToken += 1 }
                    ViewportButton(title: t("Follow"), systemImage: "video.fill", active: following, compact: true) {
                        following.toggle()
                    }
                }
                if isLandscape { povSeg; physicsBtn }
                // Console grabber — pull up the robot's control center.
                ViewportButton(title: t("Console"), systemImage: "chevron.up", compact: true) {
                    Haptics.select(); showConsole = true
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 10)
        // Portrait: hug the tab bar — close, but its glass top edge sits at
        // ~(bottomBarInset − 12), so −4 keeps a safe 8 pt above it (the old
        // −26 overlap is what used to eat the taps).
        .padding(.bottom, isLandscape ? 14 : bottomBarInset - 4)
    }

    // MARK: - Recording (one take = a training episode AND a replayable clip)

    /// Which recording card is up. NOT a sheet: a small centred card over a
    /// scrim (RecordingControls.swift — the SAME component the Vision Pro uses).
    enum RecordCard: Equatable { case record, replay }
    @State private var recordCard: RecordCard?
    @State private var recordFolder = ""
    @State private var replayRepeats = 1
    /// The take in progress, as the operator started it (folder + 0-based
    /// episode index + start time). Drives the big STOP RECORDING button.
    @State private var take: (folder: String, episode: Int, since: Date)?

    private var recordMode: RecordTopControl.Mode {
        let d = controller.demo
        if controller.recording {
            // `take` is nil only if this view was rebuilt mid-take; fall back to
            // what the server reports (its clip clock, the folder's count).
            let t = take ?? (controller.recordTask, max(0, controller.episodeCount),
                             Date().addingTimeInterval(-d.elapsed))
            return .recording(folder: t.folder, episode: t.episode, since: t.since)
        }
        if d.playing { return .replaying(name: d.name ?? "", rep: d.rep, repeats: d.repeats) }
        return .idle
    }

    private var recButton: some View {
        RecordTopControl(mode: recordMode,
                         replayEnabled: !controller.recording,
                         onRecord: openRecordCard,
                         onReplay: openReplayCard,
                         onStop: stopWhateverIsRunning)
    }

    private func openRecordCard() {
        Haptics.select()
        controller.refreshCollectionStatus()
        // Keep going in the folder you were in: the next take is Episode N+1.
        if recordFolder.isEmpty, controller.recordTask != "default" {
            recordFolder = controller.recordTask
        }
        withAnimation(Theme.Motion.snappy) { recordCard = .record }
    }

    private func openReplayCard() {
        Haptics.select()
        controller.refreshDemoClips()
        withAnimation(Theme.Motion.snappy) { recordCard = .replay }
    }

    private func closeRecordCard() {
        withAnimation(Theme.Motion.snappy) { recordCard = nil }
    }

    /// Folder chips: the one in use first, then the rest A→Z.
    private var recordFolders: [String] {
        let all = controller.serverTasks.filter { $0 != "default" || (controller.taskCounts[$0] ?? 0) > 0 }
        return all.sorted { a, b in
            if a == controller.recordTask { return true }
            if b == controller.recordTask { return false }
            return a < b
        }
    }

    @ViewBuilder private var recordCardOverlay: some View {
        switch recordCard {
        case .record:
            RecordStartCard(folder: $recordFolder, folders: recordFolders,
                            counts: controller.taskCounts,
                            onCancel: closeRecordCard,
                            onStart: { folder, episode in startRecording(into: folder, episode: episode) })
        case .replay:
            ReplayCard(clips: controller.demoClips, live: controller.connection.isLive,
                       repeats: $replayRepeats,
                       onPlay: { name in
                           controller.playDemo(name, repeats: replayRepeats)
                           closeRecordCard()
                           Haptics.rigid()
                       },
                       onDelete: { name in controller.deleteDemo(name) },
                       onClose: closeRecordCard)
                .onAppear { controller.refreshDemoClips() }
        case nil:
            EmptyView()
        }
    }

    private func startRecording(into folder: String, episode: Int) {
        controller.selectTask(folder)
        controller.startEpisode()
        recordFolder = folder
        take = (folder, episode, Date())
        closeRecordCard()
        Haptics.rigid()
    }

    private func stopWhateverIsRunning() {
        Haptics.rigid()
        if controller.recording {
            controller.endEpisode()
            take = nil
        } else if controller.demo.playing {
            controller.stopDemo()
        }
    }

    // MARK: - Teleop

    private func startTeleop() {
        applySpaceSpec()
        controller.syncTargetsToState()        // controls start at the live pose
        withAnimation(Theme.Motion.snappy) { teleopActive = true }
        frameForSpace()                        // put the operator at a useful angle
        Haptics.rigid()
    }

    /// Move the 3D viewer to the right vantage for the space we're about to drive
    /// in: DRIVE wants the whole robot (what am I about to hit?), TASK wants the
    /// wrists (what am I about to grab?). JOINT already flies to the picked module,
    /// so leave it alone. Only meaningful in 3rd person — the 1st-person camera IS
    /// the head, and yanking it would be disorienting.
    private func frameForSpace() {
        guard pov == .third else { return }
        switch space {
        case .drive: spaceFrame = .robot
        case .task:  spaceFrame = .wrists
        case .joint: focusToken += 1; return    // per-module framing owns this space
        }
        spaceFrameToken += 1
    }

    private func stopTeleop() {
        translate = .zero; yaw = 0
        joyL = .zero; joyR = .zero; pitchL = 0; pitchR = 0
        controller.wristDrive.stop()
        controller.stopBase()
        controller.setNavigation()
        withAnimation(Theme.Motion.snappy) { teleopActive = false }
        Haptics.rigid()
    }

    private func setSpace(_ s: TeleopSpace) {
        guard s != space else { return }
        if space == .drive { translate = .zero; yaw = 0; controller.stopBase() }
        controller.wristDrive.stop()
        withAnimation(Theme.Motion.snappy) { space = s }
        applySpaceSpec()
        // Re-frame for the space we just entered (joint → its module, drive → the
        // whole robot, task → the wrists). Only while actually teleoperating: a
        // space tap outside teleop shouldn't hijack the camera the operator set.
        if teleopActive { frameForSpace() } else if s == .joint { focusToken += 1 }
        // Direct control starts from the robot's LIVE pose — no first-touch jump.
        if s == .joint || s == .task { controller.syncTargetsToState() }
        Haptics.select()
    }

    /// Module picker (Joint space) — a COMPACT vertical rail on the left.
    /// Selecting flies the camera to that module's framing (front-on lift,
    /// per-side arm isometric, palm-on hands, …).
    private var moduleRail: some View {
        VStack(spacing: 4) {
            ForEach(ModuleFocus.allCases) { m in
                Button {
                    Haptics.select()
                    joyL = .zero; joyR = .zero
                    withAnimation(Theme.Motion.snappy) { module = m }
                    focusToken += 1
                } label: {
                    ZStack(alignment: .topTrailing) {
                        Image(systemName: m.icon)
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 40, height: 34)
                        if let side = m.side {
                            Text(side == "left" ? "L" : "R")
                                .font(.system(size: 8, weight: .heavy))
                                .padding(.top, 2).padding(.trailing, 4)
                        }
                    }
                    .foregroundStyle(module == m ? .white : Color(hex: 0xF4F1EA).opacity(0.8))
                    .background {
                        if module == m {
                            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.accentGradient)
                        }
                    }
                    // Inactive chips have no background — without an explicit
                    // shape the hit area collapses to the glyph strokes.
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableStyle())
            }
        }
        .padding(4)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 13, style: .continuous).strokeBorder(.white.opacity(0.14), lineWidth: 1) }
        .shadow(color: .black.opacity(0.25), radius: 8, x: 1, y: 3)
    }

    /// Control-region rail (Task space + Soft mode) — a compact vertical picker on
    /// the LEFT, mirroring the joint-space module rail. Arm / Upper / Whole: each
    /// widens what the controller owns (and what the plan preview animates).
    private var regionRail: some View {
        VStack(spacing: 4) {
            ForEach(ControlRegion.allCases) { r in
                Button {
                    Haptics.select()
                    withAnimation(Theme.Motion.snappy) { region = r }
                    if teleopActive { applySpaceSpec() }   // re-declare the spec live
                } label: {
                    VStack(spacing: 2) {
                        Image(systemName: r.icon).font(.system(size: 13, weight: .semibold))
                        Text(r.title).font(.system(size: 7.5, weight: .heavy)).tracking(0.2)
                    }
                    .frame(width: 40, height: 38)
                    .foregroundStyle(region == r ? .white : Color(hex: 0xF4F1EA).opacity(0.8))
                    .background {
                        if region == r {
                            RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.accentGradient)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableStyle())
            }
        }
        .padding(4)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 13, style: .continuous).strokeBorder(.white.opacity(0.14), lineWidth: 1) }
        .shadow(color: .black.opacity(0.25), radius: 8, x: 1, y: 3)
    }

    private func setFeel(_ f: ControllerType) {
        guard f != feel else { return }
        withAnimation(Theme.Motion.snappy) { feel = f }
        applySpaceSpec()
        Haptics.select()
    }

    /// Push the control spec matching the current space + feel + REGION. The
    /// controller/server already implement all three regions (arm / upper_body /
    /// whole_body → distinct managed joint sets + base admittance in whole body);
    /// the app just declares which one. Joint space stays arm-scoped (its rail
    /// picks a single module); task and Soft honour the region rail.
    private func applySpaceSpec() {
        // Soft (compliance) is a float/drag mode with NO space — it always runs the
        // torque law over the selected region, so the operator drags whatever the
        // region covers (arm / +torso+lift / +base).
        if feel == .compliance {
            controller.setControl(method: .joint, type: .compliance, region: region)
            return
        }
        switch space {
        case .drive: controller.setNavigation()
        case .joint:
            controller.setControl(method: .joint, type: feel, region: .arm)
        case .task:
            controller.setControl(method: .joint, type: feel, region: region)
        }
    }

    @ViewBuilder private func teleopControls(height: CGFloat) -> some View {
        let jSize = min(isLandscape ? 116 : 92, max(80, height * 0.28))
        let gripH = min(122, max(90, height * 0.34))
        Group {
            switch space {
            case .drive:
                // TRANSLATE · LIFT · ROTATE as three separate controls — the
                // lift slider sits in the MIDDLE (see DriveControls.swift).
                HStack(alignment: .bottom, spacing: 0) {
                    driveTranslateStick(jSize: jSize)
                    Spacer(minLength: 0)
                    driveLiftSlider(length: jSize + 8)
                    Spacer(minLength: 0)
                    driveRotateStick(length: jSize + 8)
                }
            case .task:
                // The classic cockpit arm cluster: per-side wrist sticks, up/down
                // shoulder nudges and grips — alongside in-viewer grab-IK.
                HStack(alignment: .bottom, spacing: 6) {
                    armGroup(side: "left", mirrored: false, jSize: jSize, gripH: gripH)
                    Spacer(minLength: 0)
                    armGroup(side: "right", mirrored: true, jSize: jSize, gripH: gripH)
                }
            case .joint:
                moduleControls(jSize: jSize, gripH: gripH)
            }
        }
        // SYMMETRIC margins in landscape for the sticks; the Joint slider panel
        // instead hugs the right edge. EVERY cluster clears the bottom pill row.
        // Portrait JOINT space clears MORE: the POV (1st/3rd) toggle rides its
        // own line above the pills there (stack top ≈ bottomBarInset + 82), and
        // joint controls are centred — +96 keeps them clear of the toggle.
        .padding(.horizontal, isLandscape ? (space == .joint ? 12 : 64) : 10)
        .padding(.bottom, isLandscape ? 60 : bottomBarInset + (space == .joint ? 96 : 52))
    }

    /// Joint-space controls for the FOCUSED module only — the camera is already
    /// framing it, the controls match it. EVERYTHING here sends DIRECT setpoints
    /// (arms + hands + torso = linear sliders; the neck = a yaw dial + pitch/roll
    /// sliders) — no module streams a rate through `onTick`, so nothing jitters.
    @ViewBuilder private func moduleControls(jSize: CGFloat, gripH: CGFloat) -> some View {
        switch module {
        case .armLeft:
            jointSliderPanel(specs(prefix: "left", in: .arms), title: "LEFT ARM", trailing: false)
        case .armRight:
            jointSliderPanel(specs(prefix: "right", in: .arms), title: "RIGHT ARM", trailing: true)
        case .handLeft:
            jointSliderPanel(specs(prefix: "left", in: .hands), title: RobotRig.handGen("left") == "gripper" ? "LEFT GRIPPER" : "LEFT HAND", trailing: false)
        case .handRight:
            jointSliderPanel(specs(prefix: "right", in: .hands), title: RobotRig.handGen("right") == "gripper" ? "RIGHT GRIPPER" : "RIGHT HAND", trailing: true)
        case .neck:
            neckPanel(dialSize: min(jSize + 4, 122))
        case .lift:
            // One slider drives BOTH stages (the robot's Sync Lift equality).
            liftSliderPanel
        case .torso:
            jointSliderPanel(RobotModel.specs(in: .body).filter { $0.id == "torso" },
                             title: "TORSO", trailing: true)
        }
    }

    /// Single-slider LIFT panel: both prismatic stages are mechanically synced
    /// (Sync Lift equality on the wire), so one height setpoint writes both.
    private var liftSliderPanel: some View {
        let range = RobotModel.spec("lift_lower").map { $0.lower...$0.upper } ?? 0...0.3175
        return HStack {
            Spacer(minLength: 0)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text("LIFT · HEIGHT").font(.system(size: 8, weight: .heavy)).tracking(1.2)
                        .foregroundStyle(Color(hex: 0xF4F1EA).opacity(0.7))
                    Spacer(minLength: 2)
                    Text(String(format: "%.3f m", controller.jointTargets["lift_lower"] ?? 0))
                        .font(.system(size: 8, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.Palette.accent)
                }
                Slider(value: Binding(get: { controller.jointTargets["lift_lower"] ?? 0 },
                                      set: { v in
                                          controller.setJoint("lift_lower", v)
                                          controller.setJoint("lift_upper", v)
                                      }),
                       in: range)
                    .tint(Theme.Palette.accent)
            }
            .padding(.horizontal, 9).padding(.vertical, 6)
            .frame(maxWidth: 186)
            .background(.black.opacity(0.42), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.1), lineWidth: 1) }
            if !isLandscape { Spacer(minLength: 0) }
        }
    }

    /// NECK panel — a circular YAW dial over PITCH / ROLL linear sliders. Every
    /// control writes an ABSOLUTE setpoint on drag (`setJoint`), exactly like the
    /// arm sliders and the web control center, so the neck no longer jitters from
    /// streamed 30 Hz rate deltas. Each axis shows its live angle and full travel,
    /// so the operator can see the continuous range and the joint's hard limits.
    /// Same right-hug (landscape) / centred (portrait) placement as the sliders.
    private func neckPanel(dialSize: CGFloat) -> some View {
        let yaw   = RobotModel.spec("neck_yaw")
        let pitch = RobotModel.specs(in: .body).filter { $0.id == "neck_pitch" }
        let roll  = RobotModel.specs(in: .body).filter { $0.id == "neck_roll" }
        return HStack {
            Spacer(minLength: 0)
            VStack(alignment: .leading, spacing: 8) {
                Text("NECK").font(.system(size: 8, weight: .heavy)).tracking(1.2)
                    .foregroundStyle(Color(hex: 0xF4F1EA).opacity(0.7))
                // YAW — circular dial (knob sits AT the joint angle; left tip =
                // look left, right tip = look right, mirroring the head).
                HStack {
                    Spacer(minLength: 0)
                    AngleDial(label: "YAW",
                              value: Binding(get: { controller.jointTargets["neck_yaw"] ?? 0 },
                                             set: { controller.setJoint("neck_yaw", $0) }),
                              range: yaw.map { $0.lower...$0.upper } ?? -1.5708...1.5708,
                              leftTag: "LEFT", rightTag: "RIGHT",
                              tint: Theme.Palette.clay, size: dialSize, invert: true)
                    Spacer(minLength: 0)
                }
                // PITCH / ROLL — linear sliders (reuse the shared compact slider).
                VStack(spacing: 1) {
                    ForEach(pitch + roll) { compactSlider($0) }
                }
            }
            .padding(.horizontal, 9).padding(.vertical, 6)
            .frame(maxWidth: 186)
            .background(.black.opacity(0.42), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.1), lineWidth: 1) }
            if !isLandscape { Spacer(minLength: 0) }
        }
    }

    /// The focused side's joints from the shared vocabulary (limits included).
    private func specs(prefix: String, in group: JointGroup) -> [JointSpec] {
        RobotModel.specs(in: group).filter { $0.id.hasPrefix(prefix) && !RobotModel.isRedundantGrip($0.id) }
    }

    /// Per-joint slider panel (arms / hands) — COMPACT setpoint controls.
    /// Landscape: pinned to the RIGHT (the rail owns the left, robot stays
    /// visible). Portrait: CENTRED, riding above the POV toggle line like the
    /// other module clusters; two columns when a module has many joints.
    private func jointSliderPanel(_ list: [JointSpec], title: String, trailing: Bool) -> some View {
        let twoCol = list.count > 5
        return HStack {
            Spacer(minLength: 0)   // landscape: right-aligned · portrait: centred
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 8, weight: .heavy)).tracking(1.2)
                    .foregroundStyle(Color(hex: 0xF4F1EA).opacity(0.7))
                if twoCol {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible())],
                                  spacing: 1) {
                            ForEach(list) { compactSlider($0) }
                        }
                    }
                    .frame(maxHeight: isLandscape ? 168 : 170)
                } else {
                    // Few joints (torso: one) — hug the content, no scroll box.
                    VStack(spacing: 1) { ForEach(list) { compactSlider($0) } }
                }
            }
            .padding(.horizontal, 9).padding(.vertical, 6)
            .frame(maxWidth: twoCol ? 324 : 186)
            .background(.black.opacity(0.42), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(.white.opacity(0.1), lineWidth: 1) }
            if !isLandscape { Spacer(minLength: 0) }   // portrait: centre it
        }
    }

    /// One tiny labelled slider (label + value above, track below) — a fraction
    /// of JointSlider's footprint so panels stay out of the 3D view.
    private func compactSlider(_ spec: JointSpec) -> some View {
        // Sided labels ("L · Elbow") drop the prefix — the panel title carries
        // the side; body joints ("Torso Pitch") keep their full label.
        let sided = spec.label.hasPrefix("L ") || spec.label.hasPrefix("R ")
        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 4) {
                Text(sided ? String(spec.label.dropFirst(2)) : spec.label)
                    .font(.system(size: 8.5, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.75)).lineLimit(1)
                Spacer(minLength: 2)
                Text(spec.kind == .prismatic
                        ? String(format: "%.3f m", controller.jointTargets[spec.id] ?? 0)
                        : String(format: "%.0f°", (controller.jointTargets[spec.id] ?? 0) * 180 / .pi))
                    .font(.system(size: 8, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.Palette.accent)
            }
            Slider(value: Binding(get: { controller.jointTargets[spec.id] ?? 0 },
                                  set: { controller.setJoint(spec.id, $0) }),
                   in: spec.range)
                .tint(Theme.Palette.accent)
                .controlSize(.mini)
        }
    }

    // (liftHold removed — the Lift module is a height SLIDER now, one setpoint
    // driving both synced stages; Drive's centre LIFT slider jogs it by rate.)

    // Drive = three SEPARATE controls (2026-10-16): the old right stick put
    // yaw on x and lift on y, so a turn leaked lift and a lift nudge leaked
    // turn. Sticks only set the TARGET twist — the 30 Hz tick sends it and the
    // server's tip-safe motion model ramps it (see `driveTick`).
    private func driveTranslateStick(jSize: CGFloat) -> some View {
        Joystick(label: "TRANSLATE", tint: Theme.Palette.accent, size: jSize) { v in
            translate = v
        }
    }

    /// 1-D yaw stick — horizontal only, springs back to zero.
    private func driveRotateStick(length: CGFloat) -> some View {
        AxisStick(label: "ROTATE", axis: .horizontal, tint: Theme.Palette.clay,
                  length: length, thickness: 52,
                  negativeIcon: "arrow.turn.up.left", positiveIcon: "arrow.turn.up.right") { v in
            yaw = v
        }
        .frame(height: length, alignment: .bottom)
    }

    /// Vertical LIFT slider in the middle — spring-loaded RATE (up = raise,
    /// release = hold), same `liftRate` → nav.ry path the old stick used, so
    /// the server's tip-safe drone lift is still the only thing moving the
    /// column. The fill behind the thumb is the live height.
    private func driveLiftSlider(length: CGFloat) -> some View {
        let h = (controller.state.joints["lift_lower"] ?? 0)
        return AxisStick(label: "LIFT", axis: .vertical, tint: Color(hex: 0x8E6FD8),
                         length: length, thickness: 46,
                         negativeIcon: "arrow.down", positiveIcon: "arrow.up",
                         gauge: controller.liftFraction,
                         readout: String(format: "%.2f m", h)) { v in
            controller.setLiftRate(v)
        }
    }

    private func armGroup(side: String, mirrored: Bool, jSize: CGFloat, gripH: CGFloat) -> some View {
        // Task space is now driven by DRAGGING THE GREEN BALL (plan → Execute), which
        // is far more intuitive than the wrist joystick + up/down nudges — so those
        // are gone. Only the GRIP slider stays (the ball plans the wrist POSE; the
        // grip opens/closes the hand). One control per side.
        gripTrigger(side: side, h: gripH)
    }

    private func holdButton(_ icon: String, side: String, dir: Double) -> some View {
        Image(systemName: icon)
            .font(.system(size: 14, weight: .bold)).foregroundStyle(.white)
            .frame(width: isLandscape ? 46 : 40, height: 32)
            .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(.white.opacity(0.15), lineWidth: 1) }
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    if side == "left" { if pitchL != dir { pitchL = dir; Haptics.soft() } }
                    else { if pitchR != dir { pitchR = dir; Haptics.soft() } }
                }
                .onEnded { _ in if side == "left" { pitchL = 0 } else { pitchR = 0 } })
    }

    private func gripTrigger(side: String, h: CGFloat) -> some View {
        let g = side == "left" ? gripL : gripR
        return VStack(spacing: 4) {
            Text("\(Int(g * 100))%").font(.mabelMonoSmall).foregroundStyle(.white)
            ZStack(alignment: .bottom) {
                Capsule().fill(.black.opacity(0.4)).frame(width: 42, height: h)
                    .overlay { Capsule().strokeBorder(.white.opacity(0.15), lineWidth: 1) }
                Capsule().fill(Theme.accentGradient).frame(width: 42, height: max(14, h * CGFloat(g)))
                Image(systemName: "hand.raised.fill").font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white).padding(.bottom, 6)
            }
            .frame(width: 42, height: h)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                let val = Double(1 - min(1, max(0, v.location.y / h)))
                if side == "left" { gripL = val } else { gripR = val }
                controller.setGrip(side, val)
            })
            Text(side == "left" ? "L GRIP" : "R GRIP")
                .font(.system(size: 7.5, weight: .bold)).tracking(1).foregroundStyle(.white.opacity(0.7))
        }
    }

    /// 30 Hz tick — Task space runs the dual wrist sticks (Cartesian rate) and
    /// Drive ramps the base twist. Joint space sends NOTHING here: every module
    /// (arms/hands/torso sliders, the neck yaw dial + pitch/roll sliders, the
    /// lift slider) writes ABSOLUTE setpoints directly on drag via `setJoint`, so
    /// there is no streamed rate to jitter the robot.
    private func onTick() {
        guard teleopActive else { return }
        let dt = 1.0 / 30.0
        if space == .task {
            // CARTESIAN wrist-target drive in the ROBOT BASE frame: stick up =
            // toward the nose, stick right = the robot's right, hold buttons =
            // raise/lower. The viewer's display-link IK (LiveMapView
            // stepWristDrive) chases the moving target with the grab-IK solver.
            let rate: Float = 0.25                     // m/s at full deflection
            controller.wristDrive.left = SIMD3(Float(-joyL.dy),   // forward
                                               Float(-joyL.dx),   // left (+) / right (−)
                                               Float(pitchL)) * rate
            controller.wristDrive.right = SIMD3(Float(-joyR.dy),
                                                Float(-joyR.dx),
                                                Float(pitchR)) * rate
        } else if space == .drive {
            // Ramped base twist from the current stick targets (see driveTick).
            controller.driveTick(translate: CGVector(dx: translate.dx, dy: translate.dy),
                                 yaw: yaw, dt: dt)
        }
    }

    // MARK: - POV

    /// Engage/exit the physics view. Engage starts the cinematic arc (the
    /// coordinator owns it via the physicsActive prop edge); exit fades back
    /// to the SceneKit render — which has been TRACKING the same camera, so
    /// the reverse cut is seamless.
    /// What renders the 3rd-person viewport: the SceneKit visualizer (map +
    /// rig) or the simulator's own MuJoCo render.
    enum ViewerKind: String, CaseIterable, Identifiable {
        case visualizer, physics
        var id: String { rawValue }
        var title: String { self == .visualizer ? "Visual" : "Physics" }
        var icon: String { self == .visualizer ? "cube.transparent" : "atom" }
    }

    private func togglePhysics() {
        Haptics.rigid()
        if physicsOn {
            withAnimation(.easeInOut(duration: 0.5)) { physicsFaded = false }
            physicsCam.faded = false
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.55) {
                controller.physicsViewOn = false
                physicsCam.active = false
            }
        } else {
            physicsFaded = false
            physicsCam.active = true
            controller.physicsViewOn = true
        }
    }

    private func setPov(_ mode: PovMode) {
        guard mode != pov else { return }
        Haptics.select()
        if physicsOn {
            physicsFaded = false
            physicsCam.faded = false
            controller.physicsViewOn = false
            physicsCam.active = false
        }
        if mode == .third, controller.headTracking { controller.toggleHeadTracking() }
        withAnimation(Theme.Motion.snappy) { pov = mode }
        if mode == .first {
            following = false            // the flight owns the camera
            fpvVisible = false
            DispatchQueue.main.asyncAfter(deadline: .now() + LiveMapView.povFlightDuration) {
                if pov == .first { fpvVisible = true }
            }
        } else {
            fpvVisible = false
        }
    }

    /// Open the full-screen live inspector for a tapped camera tile. The
    /// portrait console is a SHEET — a root-level fullScreenCover can't present
    /// while it's up, so close it first (and restore it when the inspector
    /// dismisses); every other host (landscape panel, FPV chrome) presents
    /// immediately.
    private func expandCamera(_ key: String, _ name: String, _ tint: Color) {
        let cam = ExpandedCamera(key: key, name: name, tint: tint)
        if showConsole && !isLandscape {
            restoreConsoleAfterCamera = true
            withAnimation(Theme.Motion.snappy) { showConsole = false }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { expandedCamera = cam }
        } else {
            expandedCamera = cam
        }
    }

    // MARK: - Pull-up console (the robot's control center)

    /// Shared console body — VIEWER toggles first (the low heights show only
    /// this block, so flipping layers reads live on the 3D scene).
    private var consoleContent: some View {
        // Margins + rhythm (2026-07-17): 14 pt all round put the section headers
        // hard against the sheet edge and left nothing under the drag indicator,
        // so the whole console read as one crowded block. 20 pt side margins is
        // the platform's content inset; 20 pt between sections lets each header
        // group with its OWN controls instead of floating between two.
        //
        // LAZY (2026-07-28) — this is the single biggest reason the app went
        // sluggish the moment you dragged the console up. A plain VStack inside a
        // ScrollView materialises EVERY child immediately, so opening the sheet at
        // the 0.28 detent — where only VOICE and LAYERS are on screen — also built
        // the camera section (up to five CameraFeed tiles, each opening its own
        // MJPEG socket and decoding frames), the MuJoCo sim-viewer WebSocket, and
        // the telemetry canvases. All of it streaming and redrawing behind a blur,
        // none of it visible. LazyVStack builds each section as it scrolls in and
        // tears the offscreen ones down, so the cost now matches what you can see.
        LazyVStack(alignment: .leading, spacing: 20) {
            #if DEBUG
            // Deterministic telemetry screenshots: `defaults write … debugTelemetryOnly
            // -bool YES` renders JUST the telemetry section (sheet scrolling can't be
            // driven reliably by synthetic events on the Simulator).
            if UserDefaults.standard.bool(forKey: "debugTelemetryOnly") {
                JointGraphs(controller: controller)
            } else {
                consoleVoice          // TOP: talk is a reach-for-it-now control
                consoleLayers
                if viz.camProjections { consoleCamProj }
                if viz.cloudOn { consoleCloudDensity }
                if consoleShowsHeavy {
                    consoleCameras
                    JointGraphs(controller: controller)
                } else { consoleHeavyHint }
            }
            #else
            consoleVoice              // keep BOTH branches in step — see below
            consoleLayers
            if viz.camProjections { consoleCamProj }
            if viz.cloudOn { consoleCloudDensity }
            if consoleShowsHeavy {
                consoleCameras
                JointGraphs(controller: controller)
            } else { consoleHeavyHint }
            #endif
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)        // clear of the sheet's drag indicator
        .padding(.bottom, 28)     // the last row must not sit on the home indicator
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Do the EXPENSIVE console sections exist right now?
    ///
    /// The camera section opens up to five live MJPEG sockets and the MuJoCo
    /// sim-viewer WebSocket; the telemetry section runs three display-linked
    /// canvases. At the 0.28 detent none of that is on screen — it is hundreds of
    /// points below the fold — and it was ALL running anyway, which is what made
    /// dragging the console up feel like the app fell over. Measured against the
    /// live robot on the LAN: opening the console at 0.28 took the app from 1 video
    /// stream to 5. LazyVStack alone does not fix this (SwiftUI materialises the
    /// sheet's content at its presented height and keeps what it built), so the
    /// gate is explicit.
    ///
    /// Landscape has no detents — its panel is 60% of the height and the operator
    /// opened it deliberately — so everything exists there, as before.
    private var consoleShowsHeavy: Bool {
        isLandscape || consoleDetent != .fraction(0.28)
    }

    /// What sits where the heavy sections will be, so they stay discoverable.
    private var consoleHeavyHint: some View {
        HStack(spacing: 6) {
            Image(systemName: "chevron.up").font(.system(size: 10, weight: .heavy))
            Text("Pull up for cameras, LiDAR and telemetry")
                .font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(Theme.Palette.textTertiary)
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 2)
    }

    /// Portrait: a translucent SHEET (~28% / 70% / full detents; scene stays
    /// interactive below 70%). Landscape uses `landscapeConsole` instead —
    /// compact-height iPhone sheets ignore detents and would go full-screen.
    private var consoleSheet: some View {
        ScrollView { consoleContent }
        .presentationBackground { consoleGlass }
        .presentationDetents([.fraction(0.28), .fraction(0.7), Self.consoleFullDetent],
                             selection: $consoleDetent)
        .presentationDragIndicator(.visible)
        .presentationBackgroundInteraction(.enabled(upThrough: .fraction(0.7)))
    }

    /// The console's glass (2026-07-28). `.ultraThinMaterial` alone reads as a solid
    /// slab over a dark 3D scene, and the console's whole job is to change what that
    /// scene DRAWS — you flip a layer and the thing you are judging is behind the
    /// panel. So the material is knocked back to `consoleGlassOpacity` and the scene
    /// shows through.
    ///
    /// The thin wash on top is not decoration: with the material this sheer, a bright
    /// pointcloud or the white MuJoCo render sliding underneath would take the
    /// contrast out from under the 8.5 pt chip labels. It is `canvas` — warm paper in
    /// light, warm ink in dark — so it pushes AWAY from the text colour in both
    /// appearances. A flat black wash would have been right in dark mode and wrong in
    /// light, where the console's own text is dark.
    /// Just the material, knocked back — NO tint on top. A tint was tried and
    /// removed: `Theme.Palette.canvas` is warm PAPER in light appearance, so over a
    /// dark 3D scene it lightened the panel back up (measured 173 vs the 163 of the
    /// untinted glass) — i.e. it cancelled the very transparency it was added to
    /// make safe. Legibility rides on the material's blur, which is what it is for.
    ///
    /// Measured over a black scene, light appearance: solid `.ultraThinMaterial`
    /// read 196, this reads ~150. The backdrop now MODULATES the panel in the right
    /// direction in both appearances — darker over the empty scene, brighter when
    /// the robot or a pointcloud passes behind it.
    private static let consoleGlassOpacity = 0.55

    /// The TOP detent — 0.94, not `.large` (2026-07-29).
    ///
    /// The operator asked for the fully-raised console to stay glassy enough to
    /// see the 3D model through. The first attempt made the material sheerer at
    /// `.large` and changed NOTHING: measured panel luminance 249.4 → 251.5 with
    /// the fill cut from 0.55 to 0.34. That null result is the useful one — at a
    /// FULL-HEIGHT detent SwiftUI hands `.presentationBackground` an opaque
    /// backdrop, because there is nothing left on screen behind the sheet to
    /// sample. A sheerer material just blurs opaque white harder.
    ///
    /// Stopping 6% short leaves a live strip of scene above the panel AND, more
    /// importantly, keeps the presenting view as the material's backdrop — so the
    /// glass samples the robot and the map again, which is what "transparent"
    /// actually meant here. The 6% is a real sliver on a 2622 pt phone (~55 pt),
    /// enough for the material to have something to work with.
    static let consoleFullDetent: PresentationDetent = .fraction(0.94)

    private var consoleGlass: some View {
        Rectangle()
            .fill(.ultraThinMaterial)
            .opacity(Self.consoleGlassOpacity)
    }

    /// Landscape console: an in-view translucent PANEL — 60% height by default
    /// (viewer toggles visible, scene above stays live), expandable, closable.
    private var landscapeConsole: some View {
        GeometryReader { geo in
            VStack(spacing: 0) {
                HStack {
                    Capsule().fill(.white.opacity(0.35)).frame(width: 40, height: 4)
                        .frame(maxWidth: .infinity)
                }
                .overlay(alignment: .trailing) {
                    HStack(spacing: 10) {
                        Button {
                            Haptics.select()
                            withAnimation(Theme.Motion.snappy) { consoleExpanded.toggle() }
                        } label: {
                            Image(systemName: consoleExpanded ? "chevron.down" : "chevron.up")
                                .font(.system(size: 13, weight: .bold)).foregroundStyle(.white.opacity(0.85))
                                .frame(width: 34, height: 28).contentShape(Rectangle())
                        }
                        Button {
                            Haptics.select(); showConsole = false
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 13, weight: .bold)).foregroundStyle(.white.opacity(0.85))
                                .frame(width: 34, height: 28).contentShape(Rectangle())
                        }
                    }
                    .padding(.trailing, 8)
                }
                .padding(.top, 8)
                ScrollView { consoleContent }
            }
            .frame(height: geo.size.height * (consoleExpanded ? 0.94 : 0.6))
            .frame(maxWidth: 620)
            // Same sheer glass as the portrait sheet — see `consoleGlass`.
            .background {
                consoleGlass
                    .clipShape(UnevenRoundedRectangle(
                        topLeadingRadius: 18, bottomLeadingRadius: 0,
                        bottomTrailingRadius: 0, topTrailingRadius: 18, style: .continuous))
            }
            .overlay {
                UnevenRoundedRectangle(topLeadingRadius: 18, bottomLeadingRadius: 0,
                                       bottomTrailingRadius: 0, topTrailingRadius: 18, style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func consoleHeader(_ t: String) -> some View {
        Text(t).font(.system(size: 10, weight: .heavy)).tracking(1.4)
            .foregroundStyle(Theme.Palette.textTertiary)
    }

    /// Frame budget handed to the 3D viewer (see `LiveMapView.renderFps`).
    ///
    /// Uncapped while the viewer is the screen. The console changes that: a sheer
    /// glass panel over a live SceneKit scene costs the SCENE plus a backdrop blur
    /// of it on every frame, and at the tall detents most or all of that scene is
    /// behind the panel. 30 fps still reads as live motion through the glass at a
    /// quarter of the work; the covered case drops to 10, just enough that the
    /// reveal is never a stale frame.
    ///
    /// TELEOP IS EXEMPT from the deep cut. The same display link steps the
    /// task-space wrist IK, and its dt is clamped at 1/20 s — starving it would
    /// silently slow a commanded wrist velocity, which is a control change, not a
    /// rendering one. Driving with the console open therefore never goes below 30.
    private var viewerFps: Int {
        guard showConsole else { return 60 }        // 60 = SCNView's own default
        // 30, not 10. The 10 fps floor was for a sheet that COVERED the scene —
        // nobody can see a render they cannot see. The top detent now stops short
        // of full height so the scene shows THROUGH the glass, and a 10 fps robot
        // behind it reads as a laggy robot.
        if !isLandscape, consoleDetent == Self.consoleFullDetent, !teleopActive { return 30 }
        return 30
    }

    // MARK: - Voice (talk through the robot / hear its room)

    /// TALK + LISTEN, at the TOP of the console.
    ///
    /// Placement is deliberate: the 0.28 detent shows roughly only the first block,
    /// and a talk button you must drag a sheet open to reach is a button you won't
    /// use in the moment you need it (someone is standing in front of the robot).
    ///
    /// The whole section DIMS when no voice path exists (the server does not
    /// advertise `:9610`). A mute button that looks live where no voice can
    /// physically flow would be misleading; it says why instead.
    private var consoleVoice: some View {
        let live = controller.audioAvailable
        return VStack(alignment: .leading, spacing: 8) {
            consoleHeader("VOICE")
            HStack(spacing: 8) {
                // Live scope — LEFT of the two buttons (operator request). Makes the
                // two SILENT failures visible: a flat DOWN lane while UP moves is
                // the robot bar's physical mute (peak=0); a flat UP lane while Talk
                // is latched on is a dead / denied capture. Observes the LINK
                // directly so its 14 Hz refresh invalidates only this view.
                //
                // It takes ALL the width the two fixed 78 pt buttons don't (2026-07-28):
                // at 60 pt the 128-sample window was ~0.5 pt/sample and the trace read
                // as noise. Talk and Listen are deliberately the SAME size so every
                // point the row saves goes to the graph.
                AudioScope(link: voice)
                    .frame(maxWidth: .infinity, minHeight: 44, maxHeight: 44)
                pushToTalkButton(enabled: live)
                micToggleChip(enabled: live)
            }
            .opacity(live ? 1 : 0.45)
            .allowsHitTesting(live)

            if !live { voiceUnavailableNote }
            else if let n = voice.notice {
                voiceNote(n)                          // the link's own verdict (e.g. the port is unreachable)
            }
            else if let mic = controller.audioModule("mic"), !mic.ok, !mic.detail.isEmpty {
                // The server distinguishes muted / live (silent) / no sound card —
                // pass its own words through rather than inventing a verdict. The
                // bar's own physical mute makes a healthy mic send digital zeros,
                // and calling that "muted" sends someone hunting a software bug.
                voiceNote("Robot mic: \(mic.detail)")
            }
        }
    }

    /// PRESS to talk — a LATCH, not a hold (2026-07-28). Press once and keep
    /// talking with both hands free; press again to close the mic.
    ///
    /// The old hold-to-talk was there because an open mic you forgot about is worse
    /// than a button you have to mean — that argument is answered by making the open
    /// state loud instead of making it tiring: while latched the button is the live
    /// gradient with an animating waveform and the label reads "Talking…", the scope's
    /// UP lane is moving right next to it, and `stopTalkingIfNeeded()` closes the mic
    /// on every path that takes the button off screen or kills the voice plane.
    ///
    /// A plain `Button` action, NOT a `DragGesture(minimumDistance: 0)`. This row
    /// sits at the TOP of the console, i.e. exactly where you grab to drag the sheet,
    /// and a zero-distance DragGesture claims any touch that lands on it — that ate
    /// the sheet's resize drag (the console "froze") and spun an AVAudioEngine up and
    /// down on the main actor on every attempt (the "very slow"). A Button press
    /// cancels when the finger travels, so the sheet still gets its drag.
    ///
    /// Same 78 pt footprint as the Listen chip, so the pair is symmetric and the
    /// scope gets everything else.
    private func pushToTalkButton(enabled: Bool) -> some View {
        Button {
            guard enabled else { return }
            toggleTalking()
        } label: {
            VStack(spacing: 3) {
                Image(systemName: talking ? "waveform" : "mic.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .symbolEffect(.variableColor, isActive: talking)
                Text(talking ? "Talking…" : "Talk")
                    .font(.system(size: 8.5, weight: .semibold)).lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(talking ? .white : Theme.Palette.textPrimary)
            .frame(width: 78).frame(height: 44)
            .background {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(talking ? AnyShapeStyle(Theme.Palette.live.gradient)
                                  : AnyShapeStyle(Theme.Palette.surface))
            }
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .animation(Theme.Motion.snappy, value: talking)
        .accessibilityLabel(talking ? "Stop talking" : "Press to talk")
    }

    /// The latch. `voice.talking` is the TRUTH — `startTalking()` reports a denied or
    /// dead capture by leaving it false — so the button mirrors the link rather than
    /// its own optimism; a lit button over a mic that never opened is the exact
    /// silent failure the scope was added to expose.
    private func toggleTalking() {
        if talking {
            voice.stopTalking()
            talking = false
            Haptics.soft()
        } else {
            voice.startTalking()
            talking = voice.talking
            if talking { Haptics.rigid() } else { Haptics.warning() }
        }
    }

    /// Close a latched mic. Called wherever the operator loses sight of the button:
    /// the voice plane going away, the link dropping, the Robot space disappearing.
    private func stopTalkingIfNeeded() {
        guard talking else { return }
        talking = false
        voice.stopTalking()
    }

    /// Mute / unmute what the operator HEARS from the robot's room.
    /// Same 78 pt footprint as Talk — the two are a matched pair, and every point
    /// they give up goes to the scope.
    private func micToggleChip(enabled: Bool) -> some View {
        Button {
            controller.setMicEnabled(!controller.micEnabled)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: controller.micEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill")
                    .font(.system(size: 13, weight: .semibold))
                Text(controller.micEnabled ? "Listening" : "Muted")
                    .font(.system(size: 8.5, weight: .semibold)).lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(controller.micEnabled ? .white : Theme.Palette.textSecondary)
            .frame(width: 78).frame(height: 44)
            .background {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(controller.micEnabled ? AnyShapeStyle(Theme.accentGradient)
                                                : AnyShapeStyle(Theme.Palette.surface))
            }
            .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }

    private var voiceUnavailableNote: some View {
        voiceNote(controller.connection == .connected
                  ? "Voice needs a Wi‑Fi or Tailscale link and a robot that advertises it."
                  : "Connect to MABEL to talk through it.")
    }

    private func voiceNote(_ s: String) -> some View {
        Text(s)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(Theme.Palette.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// Robot vitals at a glance (Tesla-style chip row).
    private var consoleStatus: some View {
        VStack(alignment: .leading, spacing: 8) {
            consoleHeader("STATUS")
            HStack(spacing: 8) {
                statusChip("battery.75",
                           "\(Int(controller.state.batteryLevel * 100))%",
                           tint: Theme.Palette.live)
                statusChip("point.3.connected.trianglepath.dotted",
                           controller.connection.isLive ? "Live" : "Demo",
                           tint: controller.connection.isLive ? Theme.Palette.live : Theme.Palette.clay)
                statusChip("bolt.horizontal",
                           "\(Int(controller.state.latencyMs)) ms",
                           tint: Theme.Palette.clay)
                statusChip(controller.estopEngaged ? "bolt.slash.fill" : "bolt.fill",
                           controller.estopEngaged ? "Limp" : "Torque",
                           tint: controller.estopEngaged ? Theme.Palette.danger : Theme.Palette.live)
                Spacer(minLength: 0)
            }
        }
    }

    private func statusChip(_ icon: String, _ label: String, tint: Color) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.system(size: 11, weight: .bold)).foregroundStyle(tint)
            Text(label).font(.system(size: 12, weight: .bold))
                .foregroundStyle(Theme.Palette.textPrimary)
        }
        .padding(.horizontal, 10).padding(.vertical, 7)
        .background(Theme.Palette.surface, in: Capsule(style: .continuous))
        .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
    }

    /// The RViz layer toggles — ONE line, five equal chips (icon over a tiny
    /// label), so the low console detent shows them all with the scene visible.
    private var consoleLayers: some View {
        // 14 pt BETWEEN the sub-groups, 8 pt INSIDE each one. Flat 8 pt everywhere
        // put "CAM 3D · CAMERAS" exactly as far from its own chips as from the row
        // above it, so it read as floating between the two instead of titling the
        // one below. Proximity is what makes a header belong to something.
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 8) {
                consoleHeader("LAYERS")
                // TWO rows, grouped by WHAT THEY DRAW rather than packed to fit:
                //   row 1 = the WORLD the robot has sensed (all stream-fed)
                //   row 2 = the STAGE — the robot and the ground it stands on
                //           (all locally generated, so these ALWAYS do something)
                //
                // That split is the fix for "the mesh and 2D grid buttons don't
                // work" (2026-07-29). They worked; they had nothing to draw. On a
                // robot running without nvblox the SLAM mesh and the occupancy
                // floorplan are both empty, so both chips lit up and changed
                // nothing on screen, while the only visible ground was wired to a
                // third chip called "Ground". Row 1 now DIMS when its stream is
                // silent and says so, and the ground has its own honest chips.
                //
                // Five across a 390 pt sheet is ~67 pt each — comfortably over
                // Apple's 44 pt, unlike the eight-across row this replaced.
                HStack(spacing: 6) {
                    layerChip("Mesh", "square.3.layers.3d", $viz.meshOn, live: store.hasMesh)
                    layerChip("2D Map", "squareshape.split.3x3", $viz.gridOn, live: store.hasGrid)
                    layerChip("Cloud", "aqi.medium", $viz.cloudOn)
                    layerChip("LiDAR", "rays", $viz.laserOn, live: controller.lidarScan != nil)
                    layerChip("Cam 3D", "camera.metering.matrix", $viz.camProjections)
                }
                HStack(spacing: 6) {
                    layerChip("Robot", "figure.stand", $viz.robotOn)
                    layerChip("Floor", "circle.dashed", $viz.floorOn)
                    layerChip("Grid", "grid", $viz.floorGridOn)
                    layerChip("Minimap", "map", $viz.minimapOn)
                    layerChip("TF", "move.3d", $viz.tfOn)
                }
                if !store.hasMesh || !store.hasGrid {
                    Text(mapLayerHint)
                        .font(.system(size: 9.5, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            // Per-camera Cam 3D projections (default: head only).
            if viz.camProjections {
                camStreamRow("CAM 3D · CAMERAS", options: Self.camStreamOptions, set: $viz.camStreams)
            }
            // The Cloud chip is the MASTER switch and draws NO cloud of its own:
            // off → every pointcloud in the scene goes (per-camera planes pruned
            // + their sockets closed, the /map fallback cloud hidden). The only
            // per-cloud switches are these two — head + base-front (the wrist
            // clouds aren't part of the operator picture). Chips dim (with the
            // note) until a direct link advertises the 96xx ports.
            if viz.cloudOn {
                camStreamRow("CAMERA CLOUDS", options: Self.camCloudOptions, set: $viz.camClouds,
                             enabledFor: { controller.cameraCloudURL($0) != nil })
                if !Self.camCloudOptions.contains(where: { controller.cameraCloudURL($0.key) != nil }) {
                    Text("Camera clouds need a direct LAN link to a robot advertising the 96xx stream ports.")
                        .font(.system(size: 9.5, weight: .medium))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
        }
    }

    /// One row of per-camera multi-select chips (projection / cloud toggles).
    private func camStreamRow(_ title: String, options: [(key: String, label: String)],
                              set: Binding<Set<String>>,
                              enabledFor: ((String) -> Bool)? = nil) -> some View {
        // 6 pt: a sub-header hugs its own chips TIGHTER than the 14 pt that
        // separates it from the group above — that difference is the hierarchy.
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.system(size: 8, weight: .heavy)).tracking(1)
                .foregroundStyle(Theme.Palette.textTertiary)
            HStack(spacing: 6) {
                ForEach(options, id: \.key) { opt in
                    let available = enabledFor?(opt.key) ?? true
                    let on = set.wrappedValue.contains(opt.key) && available
                    Button {
                        Haptics.select()
                        withAnimation(Theme.Motion.snappy) {
                            if set.wrappedValue.contains(opt.key) { set.wrappedValue.remove(opt.key) }
                            else { set.wrappedValue.insert(opt.key) }
                        }
                    } label: {
                        Text(opt.label)
                            .font(.system(size: 10.5, weight: .semibold)).lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .foregroundStyle(on ? .white
                                : available ? Theme.Palette.textSecondary
                                            : Theme.Palette.textTertiary.opacity(0.6))
                            .frame(maxWidth: .infinity)
                            .frame(height: 30)
                            .background {
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(on ? AnyShapeStyle(Theme.accentGradient)
                                             : AnyShapeStyle(Theme.Palette.surface))
                            }
                            .overlay { RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(PressableStyle())
                    .disabled(!available)
                }
            }
        }
    }

    /// Reconcile the per-camera cloud links to the toggles + current route.
    /// Idempotent — called on toggle, connect/route change, and appear.
    private func syncCameraClouds() {
        var urls: [String: [URL]] = [:]
        for (key, _) in Self.camCloudOptions where viz.cloudOn && viz.camClouds.contains(key) {
            let candidates = controller.cameraCloudURLs(key)
            if !candidates.isEmpty { urls[key] = candidates }
        }
        store.setCameraClouds(urls)
    }

    /// Which stream-fed layers have nothing to show, phrased as one sentence.
    /// Only ever visible when at least one of them is empty.
    private var mapLayerHint: String {
        let missing = [store.hasMesh ? nil : "3D mesh",
                       store.hasGrid ? nil : "2D map"].compactMap { $0 }
        return "No \(missing.joined(separator: " or ")) from SLAM yet — those layers "
            + "stay empty until the robot is brought up with mapping (nvblox / cuVSLAM)."
    }

    /// One layer toggle.
    ///
    /// `live: false` means "this layer is ON but its stream has never delivered
    /// anything". It stays TAPPABLE — the operator's choice is still recorded and
    /// takes effect the moment data arrives — but renders dimmed and outlined
    /// instead of filled, so an empty layer never masquerades as a lit one. The
    /// alternative (a chip that lights up and changes nothing on screen) is what
    /// made Mesh and 2D Map read as broken buttons.
    private func layerChip(_ title: String, _ icon: String, _ on: Binding<Bool>,
                           live: Bool = true) -> some View {
        let lit = on.wrappedValue && live
        return Button {
            Haptics.select()
            withAnimation(Theme.Motion.snappy) { on.wrappedValue.toggle() }
        } label: {
            VStack(spacing: 3) {
                Image(systemName: icon).font(.system(size: 13, weight: .semibold))
                Text(title).font(.system(size: 8.5, weight: .semibold)).lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .foregroundStyle(lit ? .white
                                 : on.wrappedValue ? Theme.Palette.textSecondary
                                                   : Theme.Palette.textTertiary)
            .frame(maxWidth: .infinity)
            .frame(height: 44)
            .background {
                if lit {
                    RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.accentGradient)
                } else {
                    RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.Palette.surface)
                }
            }
            .overlay {
                // ON-but-empty gets the accent as an OUTLINE: the operator's choice
                // is visibly recorded without claiming anything is being drawn.
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(on.wrappedValue && !live
                                    ? AnyShapeStyle(Theme.Palette.accent.opacity(0.55))
                                    : AnyShapeStyle(Theme.Palette.hairline),
                                  lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableStyle())
    }

    /// Map source: build live SLAM now, or localize against a saved map (the
    /// portrait top row hands this to the console to stay lean).
    private var consoleMapSource: some View {
        VStack(alignment: .leading, spacing: 8) {
            consoleHeader("MAP SOURCE · \(mapSectionTitle)")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    Button { selectLive() } label: {
                        sourceChip("Live SLAM", "dot.radiowaves.left.and.right", active: selected.isLive)
                    }
                    .buttonStyle(PressableStyle())
                    ForEach(savedRooms) { room in
                        Button { selectRoom(room) } label: {
                            sourceChip(room.name, room.icon,
                                       active: !selected.isLive && selected.name == room.name)
                        }
                        .buttonStyle(PressableStyle())
                    }
                    if connectedToRobot && savedRooms.isEmpty {
                        Text("No saved maps on the robot")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .padding(.vertical, 9)
                    }
                }
            }
        }
    }

    private func sourceChip(_ title: String, _ icon: String, active: Bool) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon).font(.system(size: 12, weight: .semibold))
            Text(title).font(.system(size: 12, weight: .semibold)).lineLimit(1)
        }
        .foregroundStyle(active ? .white : Theme.Palette.textSecondary)
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background {
            if active { Capsule(style: .continuous).fill(Theme.accentGradient) }
            else { Capsule(style: .continuous).fill(Theme.Palette.surface) }
        }
        .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
    }

    /// Cam 3D fine controls — size slider + rectified/lens toggle on ONE line.
    private var consoleCamProj: some View {
        VStack(alignment: .leading, spacing: 8) {
            consoleHeader("CAM 3D PROJECTION")
            HStack(spacing: 10) {
                Slider(value: $viz.camProjSize, in: 0.1...1.0)
                Text(String(format: "%.1fm", viz.camProjSize))
                    .font(.mabelMonoSmall).foregroundStyle(Theme.Palette.textSecondary)
                    .frame(width: 36, alignment: .trailing)
                GlassSegmented(options: ProjSurface.allCases,
                               selection: Binding(get: { viz.camFlat ? ProjSurface.rectified : .lens },
                                                  set: { viz.camFlat = ($0 == .rectified); Haptics.select() }),
                               title: { _ in "" }, icon: { $0.icon })
                    .fixedSize()
            }
        }
    }

    /// Live point-cloud render density.
    private var consoleCloudDensity: some View {
        VStack(alignment: .leading, spacing: 8) {
            consoleHeader("POINT CLOUD DENSITY")
            GlassSegmented(options: CloudDensity.allCases,
                           selection: cloudDensityBinding,
                           title: { $0.title }, icon: { _ in "aqi.medium" })
                .fixedSize()
        }
    }

    /// The camera + sensor monitor: head hero, then ALL streams and the LiDAR
    /// sweep in a 2×2 grid.
    private var consoleCameras: some View {
        VStack(alignment: .leading, spacing: 8) {
            // The MuJoCo viewer, right above the camera tiles (sim only): the
            // simulator's own render of the live physics, orbit/pinch to look
            // around — all teleop controls keep working while it's open.
            SimViewerSection(studio: controller.simStudio)
            consoleHeader("CAMERAS & LIDAR")
            CameraFeed(name: "HEAD · POV", isMain: true,
                       streamURL: controller.videoURL("main"),
                       stream: "main",
                       onExpand: { expandCamera("main", "HEAD · POV", Theme.Palette.accent) })
                .frame(height: 200)
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible())], spacing: 8) {
                CameraFeed(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                           streamURL: controller.videoURL("wrist_left"),
                           stream: "wrist_left",
                           onExpand: { expandCamera("wrist_left", "L WRIST", Theme.Palette.live) })
                    .frame(height: 120)
                CameraFeed(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                           streamURL: controller.videoURL("wrist_right"),
                           stream: "wrist_right",
                           onExpand: { expandCamera("wrist_right", "R WRIST", Theme.Palette.clay) })
                    .frame(height: 120)
                CameraFeed(name: "BASE", compact: true, tint: Theme.Palette.accent,
                           streamURL: controller.videoURL("front"),
                           stream: "front",
                           onExpand: { expandCamera("front", "BASE", Theme.Palette.accent) })
                    .frame(height: 120)
                LidarView(scan: controller.lidarScan)
                    .frame(height: 120)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: Theme.Radius.sm, style: .continuous)
                            .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                    }
            }
        }
    }

    // MARK: - Overlays

    /// True while the live head video is on screen in 1st person — gates the
    /// cockpit glance chrome (wrist cams + the floating 3D twin).
    private var fpvChromeOn: Bool {
        pov == .first && controller.connection.isLive && fpvVisible && fpvLive && !povLooking
    }

    /// Shown in 1st person when the head camera is NOT delivering: the view you are
    /// looking at is the 3D twin, not the robot's eyes, and that difference matters
    /// (the twin cannot show you an obstacle the model does not know about). Names
    /// the actual fault so it is fixable, instead of leaving the operator to wonder
    /// why the POV transition "didn't work".
    private var headCamOfflinePill: some View {
        VStack {
            Spacer()
            HStack(spacing: 6) {
                Image(systemName: "video.slash.fill")
                    .font(.system(size: 10, weight: .bold))
                Text(controller.videoURL("main") == nil
                     ? "NO HEAD CAMERA ROUTE · 3D VIEW"
                     : "HEAD CAMERA OFFLINE · 3D VIEW")
                    .font(.system(size: 10, weight: .heavy)).tracking(0.4)
            }
            .foregroundStyle(Theme.Palette.warn)
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(.ultraThinMaterial, in: Capsule(style: .continuous))
            .overlay { Capsule(style: .continuous).strokeBorder(Theme.Palette.warn.opacity(0.5), lineWidth: 1) }
            .padding(.bottom, 96)
        }
        .allowsHitTesting(false)
    }

    /// 1st-person cockpit chrome (v1 parity): the two wrist cams glanceable at
    /// the TOP-LEFT while the head video fills the screen. The TILES are
    /// tappable (tap → full-screen zoomable inspector); everywhere else the
    /// stack is Spacers, so look-around drags on the FPV still pass through.
    /// Slides right when the Joint-space module rail owns the left edge.
    private var fpvChrome: some View {
        VStack {
            HStack(spacing: 6) {
                CameraFeed(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                           streamURL: controller.videoURL("wrist_left"),
                           stream: "wrist_left",
                           onExpand: { expandCamera("wrist_left", "L WRIST", Theme.Palette.live) })
                    .frame(width: 106, height: 66)
                CameraFeed(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                           streamURL: controller.videoURL("wrist_right"),
                           stream: "wrist_right",
                           onExpand: { expandCamera("wrist_right", "R WRIST", Theme.Palette.clay) })
                    .frame(width: 106, height: 66)
                Spacer(minLength: 0)
            }
            .padding(.leading, teleopActive && space == .joint ? 56 : 12)
            .padding(.trailing, 12)
            .padding(.top, isLandscape ? 58 : 112)
            Spacer(minLength: 0)
        }
        .transition(.opacity)
    }

    /// The live articulated 3D model, v1 cockpit style (same card treatment as
    /// CockpitView.modelView) — clipped so the studio floor's square corners
    /// never poke past the rounded card over bright video.
    private var fpvModelTile: some View {
        RealRobotView(pose: controller.livePose)
            .rebuildsOnRobotRigChange()   // new robot / swapped hands → rebuild the scene
            .frame(width: isLandscape ? 132 : 116, height: isLandscape ? 100 : 88)
            .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(.white.opacity(0.14), lineWidth: 1) }
            .allowsHitTesting(false)
    }

    /// Floating glass prompt — driving works everywhere; landscape is roomier.
    private var rotateHint: some View {
        HintPill(systemImage: "rotate.right.fill", title: "Landscape Recommended",
                 subtitle: "Driving works here too — rotate for the full cockpit",
                 iconTint: Theme.Palette.accent, pulse: true)
    }

    /// Task-space coaching (shown while task manipulation is armed). In Stiff the
    /// green ball now PLANS — the hint says so, so the operator isn't surprised the
    /// robot doesn't move until Execute. While a plan is staged the Execute pill
    /// carries the instruction, so the hint steps aside.
    @ViewBuilder private var taskHint: some View {
        if armPlan == nil {
            HintPill(systemImage: "hand.draw.fill",
                     title: feel == .impedance ? "Drag to plan" : "Soft · drag to move",
                     subtitle: feel == .impedance
                        ? "Drag the ball, then Execute"
                        : "Arms yield to your drag",
                     iconTint: Theme.Palette.live)
        }
    }

    /// Honest, concise notice while the LOCAL demo map is loaded (opt-in).
    private var demoBanner: some View {
        HintPill(systemImage: "cpu",
                 title: "Local demo map",
                 subtitle: "Tap the floor — goals drive the demo robot",
                 iconTint: Theme.Palette.clay)
    }

    /// Centred standby card while there's nothing (yet) to show.
    private var standby: some View {
        let fg = Color(hex: 0xF4F1EA)
        return VStack(spacing: 8) {
            Image(systemName: store.hasEndpoint ? "point.3.connected.trianglepath.dotted" : "globe.badge.chevron.backward")
                .font(.system(size: 30, weight: .light)).foregroundStyle(fg.opacity(0.7))
            Text(store.hasEndpoint ? "Waiting for the robot's map…" : "No map data streamed")
                .font(.system(size: 13, weight: .semibold)).foregroundStyle(fg)
            Text(store.hasEndpoint
                 ? "Run visual SLAM on the sim or robot — the mesh and odom appear here live."
                 : "Connect to a robot (Settings), or load a local demo map.")
                .font(.system(size: 11, weight: .medium)).foregroundStyle(fg.opacity(0.7))
                .multilineTextAlignment(.center).frame(maxWidth: 270)
            Button {
                Haptics.select(); showConsole = true
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "square.stack.3d.up.fill").font(.system(size: 11, weight: .semibold))
                    Text("Choose Map").font(.system(size: 12, weight: .bold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(Theme.accentGradient, in: Capsule(style: .continuous))
            }
            .buttonStyle(PressableStyle())
            .padding(.top, 4)
        }
        .padding(20)
        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(.white.opacity(0.1), lineWidth: 1) }
        .frame(maxHeight: .infinity, alignment: .center)
        .padding(.bottom, 40)
    }

    // The map viewport is always dark, so the glass pills use fixed light ink.
    // `title: nil` → compact icon-only pill (portrait, where width is scarce).
    private func pill(_ title: String?, _ icon: String, active: Bool, chevron: Bool = false) -> some View {
        let content = HStack(spacing: 6) {
            Image(systemName: icon).font(.system(size: 12, weight: .semibold))
            if let title { Text(title).font(.system(size: 13, weight: .semibold)).lineLimit(1) }
            if chevron { Image(systemName: "chevron.down").font(.system(size: 9, weight: .bold)).opacity(0.85) }
        }
        .foregroundStyle(active ? .white : Color(hex: 0xF4F1EA))
        .padding(.horizontal, title == nil ? 11 : 14).padding(.vertical, 9)

        return Group {
            if active {
                content.background(Theme.accentGradient, in: Capsule(style: .continuous))
            } else {
                content.glassEffect(.regular, in: Capsule(style: .continuous))
                    .overlay { Capsule(style: .continuous).strokeBorder(.white.opacity(0.16), lineWidth: 1) }
            }
        }
        .shadow(color: .black.opacity(0.28), radius: 10, y: 4)
    }

    /// Everything the Robot space does on appear.
    ///
    /// EXTRACTED from the `.onAppear` closure: inline, this had grown into a
    /// ~60-line body inside a view builder, and the Swift type-checker gave up on
    /// it ("unable to type-check this expression in reasonable time"). A plain
    /// method is checked once, on its own, instead of as part of the surrounding
    /// view expression.
    private func onRobotSpaceAppear() {
        controller.requestMapList()
        controller.liveMap.active = true
        syncCameraClouds()
        // Rotation re-created this view: resume physics view WITHOUT the
        // cinematic (the shared cam still holds the faded pose).
        if controller.physicsViewOn, controller.physicsViewCam.faded {
            physicsFaded = true
        }
        #if DEBUG
        if UserDefaults.standard.bool(forKey: "debugTeleop"), !teleopActive {
            startTeleop()
        }
        if let s = UserDefaults.standard.string(forKey: "debugSpace"),
           let sp = TeleopSpace(rawValue: s) { space = sp; if teleopActive { applySpaceSpec() } }
        // debugDrivePad="ly=-0.8:3;rot=0.8:3;lift=1:3" — script the Drive pad
        // (see DebugDriveScript). Writes the SAME state the pad's controls do.
        if let spec = UserDefaults.standard.string(forKey: "debugDrivePad") {
            DebugDriveScript(spec).play(apply: { st in
                switch st.axis {
                case "lx":  translate.dx = st.value
                case "ly":  translate.dy = st.value
                case "rot": yaw = st.value
                default:    controller.setLiftRate(st.value)
                }
            }, release: {
                translate = .zero; yaw = 0
                controller.setLiftRate(0)
            })
        }
        if let m = UserDefaults.standard.string(forKey: "debugModule"),
           let mf = ModuleFocus(rawValue: m) { module = mf; focusToken += 1 }
        // Screenshot harness: show the GO pill without a (synthetic) tap.
        if UserDefaults.standard.bool(forKey: "debugGoal") { goalState = .placed }
        // Boot with camera-cloud toggles on ("main,front") — taps don't land.
        if let cc = UserDefaults.standard.string(forKey: "debugCamClouds") {
            viz.camClouds = Set(cc.split(separator: ",").map(String.init))
        }
        // debugLayers=mesh,cloud,laser,grid,robot,floor,floorgrid,minimap,cam,tf —
        // set the layer set EXACTLY (anything absent goes off), so a screenshot can
        // isolate one layer without ten taps the Simulator can't synthesise. VP
        // parity (its own `debugLayers`), plus `debugTF` for the common case.
        // `grid` is the SLAM occupancy floorplan ("2D Map"); the reference lines on
        // the floor are `floorgrid`. `ground` is accepted as the retired spelling
        // of `floorgrid` so older harness invocations keep working.
        if let ls = UserDefaults.standard.string(forKey: "debugLayers") {
            let want = Set(ls.split(separator: ",").map(String.init))
            viz.meshOn = want.contains("mesh");     viz.cloudOn = want.contains("cloud")
            viz.laserOn = want.contains("laser");   viz.gridOn = want.contains("grid")
            viz.robotOn = want.contains("robot");   viz.floorOn = want.contains("floor")
            viz.floorGridOn = want.contains("floorgrid") || want.contains("ground")
            viz.minimapOn = want.contains("minimap")
            viz.camProjections = want.contains("cam"); viz.tfOn = want.contains("tf")
        }
        if UserDefaults.standard.bool(forKey: "debugTF") { viz.tfOn = true }
        if UserDefaults.standard.bool(forKey: "debugConsole") { showConsole = true }
        // debugRecordCard=record|replay|recording — open a recording card (or
        // start a real take into `ui_check`) for screenshots: the Simulator
        // cannot synthesise the taps. Delete the key after the run.
        if let rc = UserDefaults.standard.string(forKey: "debugRecordCard") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                switch rc {
                case "replay":    openReplayCard()
                case "recording": startRecording(into: "ui_check",
                                                  episode: controller.taskCounts["ui_check"] ?? 0)
                default:          recordFolder = "ui_check"; openRecordCard()
                }
            }
        }
        // Screenshot harness: boot straight into the camera inspector for a
        // stream key ("main" | "wrist_left" | …) — tile taps can't be driven
        // by synthetic events on the Simulator.
        if let dk = UserDefaults.standard.string(forKey: "debugCameraDetail") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                expandCamera(dk, dk == "main" ? "HEAD · POV" : dk.uppercased(),
                             Theme.Palette.accent)
            }
        }
        // debugTalk=<seconds> — latch the mic open for N s with no finger. Exercises
        // the uplink headlessly (capture → 16 kHz → 0x07 → :9610 → the robot's
        // speaker) without needing the Simulator to deliver a tap.
        let talkFor = UserDefaults.standard.double(forKey: "debugTalk")
        if talkFor > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 6) {
                toggleTalking()
                DispatchQueue.main.asyncAfter(deadline: .now() + talkFor) {
                    stopTalkingIfNeeded()
                }
            }
        }
        if UserDefaults.standard.bool(forKey: "debugLocalDemo") { store.startLocalDemo() }
        // Physics-view harness: engage the MuJoCo takeover after N seconds.
        let pdelay = UserDefaults.standard.double(forKey: "debugPhysics")
        if pdelay > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + pdelay) { togglePhysics() }
            // round trip: back to the visualizer after the fade settles
            if UserDefaults.standard.bool(forKey: "debugPhysicsRoundtrip") {
                DispatchQueue.main.asyncAfter(deadline: .now() + pdelay + 14) { togglePhysics() }
            }
        }
        // Flight harness: fly 3rd→1st after N seconds, back at 2N+3 — lets
        // the POV transition be captured on video without a (synthetic) tap.
        let fdelay = UserDefaults.standard.double(forKey: "debugPovFlight")
        if fdelay > 0 {
            DispatchQueue.main.asyncAfter(deadline: .now() + fdelay) { setPov(.first) }
            DispatchQueue.main.asyncAfter(deadline: .now() + 2 * fdelay + 3) { setPov(.third) }
        }
        // Plan/execute harness: start teleop in Task+Stiff,
        // STAGE a plan (no send), hold so the wire can confirm the arm is still,
        // then EXECUTE. `debugPlanExecute=<left|right>`.
        // Full harness: stage (no send), hold, then execute (streams).
        // `debugPlanStage` stops after staging so the Execute pill + preview can
        // be inspected without the auto-execute.
        let planSideArg = UserDefaults.standard.string(forKey: "debugPlanExecute")
            ?? UserDefaults.standard.string(forKey: "debugPlanStage")
        if let side = planSideArg {
            let stageOnly = UserDefaults.standard.string(forKey: "debugPlanStage") != nil
            debugStageSide = side == "right" ? "right" : "left"
            if let rg = UserDefaults.standard.string(forKey: "debugRegion"),
               let r = ControlRegion(rawValue: rg) { region = r }
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                if !teleopActive { startTeleop() }
                space = .task; feel = .impedance; applySpaceSpec()
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 6)  { debugStageToken += 1 }   // stage (no send)
            if !stageOnly {
                DispatchQueue.main.asyncAfter(deadline: .now() + 12) { executePlanToken += 1 }  // execute
            }
        }
        #endif
        if pov == .first {
            following = false
            DispatchQueue.main.asyncAfter(deadline: .now() + LiveMapView.povFlightDuration) {
                if pov == .first { fpvVisible = true }
            }
        }
    }
}
