//
//  MabelTeleopApp.swift
//  MabelTeleop
//
//  App entry point.
//
//  Scenes:
//    main-window      — the five-tab operator console (Teleop · Body ·
//                       Maps · Policies · Robot) with the global status bar
//                       and bottom navigation as glass ornaments
//    model-volume     — volumetric MABEL model (drag / zoom / life size)
//    immersive-teleop — the ONE unified teleop space: wrap-around stereo
//                       head view, hand retargeting, persistent spatial
//                       joysticks, head → neck tracking
//

import SwiftUI

@main
struct MabelTeleopApp: App {

    @StateObject private var controller = RobotController()
    @StateObject private var discovery = ServerDiscovery()

    var body: some Scene {
        // Primary operator console
        WindowGroup(id: WindowID.main.rawValue) {
            ContentView()
                .environmentObject(controller)
                .environmentObject(discovery)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.plain)
        .defaultSize(width: 1280, height: 820)

        // Volumetric robot model — walk around MABEL, snap to life size.
        WindowGroup(id: WindowID.modelVolume.rawValue) {
            ModelVolumeView()
                .environmentObject(controller)
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 2.2, height: 2.4, depth: 2.2, in: .meters)

        // Volumetric SLAM map — the room diorama with the live robot inside.
        WindowGroup(id: WindowID.mapVolume.rawValue) {
            MapVolumeView()
                .environmentObject(controller)
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 2.2, height: 2.4, depth: 2.2, in: .meters)

        // Volumetric Robot space — the live robot + Cam 3D projections + grid +
        // cloud + laser as a walk-around diorama (right-edge button in the viewer).
        WindowGroup(id: WindowID.robotVolume.rawValue) {
            RobotVolumeView(controller: controller)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.volumetric)
        .defaultSize(width: 1.6, height: 1.3, depth: 1.6, in: .meters)
        .restorationBehavior(.disabled)
        .defaultWindowPlacement { _, context in
            // Spawn BESIDE the console, never on top of it — co-located with an
            // empty (still-loading) volume it read as "my window disappeared".
            if let main = context.windows.first(where: { $0.id == WindowID.main.rawValue }) {
                return WindowPlacement(.trailing(main))
            }
            return WindowPlacement()
        }

        // ── Add-on windows (control-center style, opened from the Robot space) ──

        // ALL video + sensor feeds together in one placeable window (head hero,
        // wrists, base cam, LiDAR sweep). The aux windows take a PLAIN controller
        // reference (not environmentObject) so the 8 Hz uiTick never re-renders
        // them — their sampling timers must survive re-renders.
        WindowGroup(id: WindowID.auxSensors.rawValue) {
            SensorsWindow(controller: controller)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 860, height: 820)
        .restorationBehavior(.disabled)

        // Live joint bar graph — current positions, Pos / Vel / Torque toggle.
        WindowGroup(id: WindowID.jointBars.rawValue) {
            JointBarsWindow(controller: controller)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 760, height: 540)
        .restorationBehavior(.disabled)

        // Rolling 2-minute time series — position, velocity and torque charts
        // with per-module add/remove.
        WindowGroup(id: WindowID.timeSeries.rawValue) {
            TimeSeriesWindow(controller: controller)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 860, height: 900)
        .restorationBehavior(.disabled)

        // Top-down 2D occupancy map (grid + odom trail + route + goal).
        WindowGroup(id: WindowID.map2D.rawValue) {
            Map2DWindow(controller: controller)
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 640, height: 640)
        .restorationBehavior(.disabled)

        // Live inspector for ONE camera stream — opened by tapping any camera
        // tile. Value-parameterized by the wire stream key, so each camera gets
        // its own placeable window (hero-quality video, pinch-zoom + pan).
        WindowGroup(id: WindowID.cameraDetail.rawValue, for: String.self) { $key in
            CameraDetailWindow(controller: controller, streamKey: key ?? "main")
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1080, height: 700)
        .restorationBehavior(.disabled)

        // Unified teleoperation space. Opens MIXED ("Start Teleop": the room
        // and windows stay visible while hands/head track and the spatial
        // joysticks float); switches to FULL for the 180° wrap-around view.
        ImmersiveSpace(id: WindowID.immersive.rawValue) {
            TeleopImmersiveView()
                .environmentObject(controller)
        }
        .immersionStyle(selection: Binding(
            get: {
                // Full mode is PROGRESSIVE: fully immersed by default, and
                // the Digital Crown dials the room back in — no app toggle.
                controller.fullImmersion
                    ? .progressive(0.05...1.0, initialAmount: 1.0)
                    : .mixed
            },
            set: { controller.fullImmersion = !($0 is MixedImmersionStyle) }
        ), in: .mixed, .progressive)
    }
}

/// Centralized window / immersive-space identifiers.
enum WindowID: String {
    case main        = "main-window"
    case modelVolume = "model-volume"
    case mapVolume   = "map-volume"
    case immersive   = "immersive-teleop"
    // Add-on windows (control-center style).
    case robotVolume = "robot-volume"
    case auxSensors  = "aux-sensors"
    case jointBars   = "aux-joint-bars"
    case timeSeries  = "aux-time-series"
    case map2D       = "aux-map-2d"
    case cameraDetail = "camera-detail"
}
