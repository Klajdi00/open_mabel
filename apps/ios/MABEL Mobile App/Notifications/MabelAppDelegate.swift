//
//  MabelAppDelegate.swift
//  MABEL Mobile App
//
//  Minimal UIKit delegate the SwiftUI app adopts, purely to bridge the things
//  SwiftUI can't: APNs device-token registration and notification taps/foreground
//  presentation. The actual policy lives in MabelNotifier.
//

import UIKit
import UserNotifications
import WidgetKit

final class MabelAppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        Task { @MainActor in MabelNotifier.shared.registerCategories() }
        return true
    }

    // MARK: APNs token

    func application(_ application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let hex = deviceToken.map { String(format: "%02x", $0) }.joined()
        Task { @MainActor in MabelNotifier.shared.setAPNSToken(hex) }
    }

    func application(_ application: UIApplication,
                     didFailToRegisterForRemoteNotificationsWithError error: Error) {
        // No remote push (e.g. Simulator / no entitlement) — local notifications
        // still work while the app is alive.
    }

    // MARK: Silent pushes (content-available) → refresh the widget + island

    func application(_ application: UIApplication,
                     didReceiveRemoteNotification userInfo: [AnyHashable: Any],
                     fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        WidgetCenter.shared.reloadAllTimelines()
        completionHandler(.newData)
    }

    // MARK: Foreground presentation + taps

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                willPresent notification: UNNotification) async
        -> UNNotificationPresentationOptions {
        // A push that arrived while the app is open is server-deemed important —
        // surface it as a banner. (Local events are already suppressed in
        // foreground by MabelNotifier.)
        [.banner, .sound, .list]
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter,
                                didReceive response: UNNotificationResponse) async {
        // Tapping any MABEL notification routes into the live cockpit via the
        // app's existing deep-link handler.
        let info = response.notification.request.content.userInfo
        let link = (info["deeplink"] as? String).flatMap(URL.init) ?? URL(string: "mabelv2://island")
        if let link {
            await MainActor.run { UIApplication.shared.open(link) }
        }
    }
}
