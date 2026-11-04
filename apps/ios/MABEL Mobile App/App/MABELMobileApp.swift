//
//  MABELMobileApp.swift
//  MABEL Mobile App
//
//  Application entry point. Owns the single RobotController and injects
//  it into the environment for every screen.
//
//

import SwiftUI

@main
struct MABELMobileApp: App {
    @UIApplicationDelegateAdaptor(MabelAppDelegate.self) private var appDelegate
    @StateObject private var controller = RobotController()

    @State private var showLaunch = true
    /// Mirrors live telemetry into the Home/Lock-Screen widget's shared container.
    @State private var widget = MabelWidgetPublisher()
    /// Drives the Dynamic Island / Live Activity while connected to a host.
    @State private var liveActivity = MabelLiveActivity()
    /// Turns robot state transitions into (local) notifications.
    @State private var watcher = RobotEventWatcher()
    @AppStorage("notif.asked") private var askedForNotifications = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ZStack {
                ContentView()
                    .environmentObject(controller)
                if showLaunch {
                    LaunchView()
                        .transition(.opacity)
                        .zIndex(1)
                        .task {
                            try? await Task.sleep(nanoseconds: 2_400_000_000)
                            withAnimation(.easeInOut(duration: 0.55)) { showLaunch = false }
                        }
                }
            }
            .task {
                widget.start(controller)
                liveActivity.start(controller)
                MabelNotifier.shared.start()
                watcher.start(controller)
                // Ask once, gently, on first launch; Settings can re-trigger later.
                if !askedForNotifications {
                    askedForNotifications = true
                    await MabelNotifier.shared.requestAuthorization()
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            // Capture one last fresh snapshot on the way out so the widget keeps the
            // most recent pose + telemetry while the app is suspended.
            if phase == .background || phase == .inactive { widget.flush() }
        }
    }
}
