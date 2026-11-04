//
//  Layout.swift
//  MABEL Mobile App
//
//  Orientation flag pushed down the environment so each screen can pick a
//  landscape (side-by-side) or portrait (stacked) layout. Drive is the only
//  landscape-only screen; everything else adapts.
//

import SwiftUI

private struct LandscapeKey: EnvironmentKey {
    static let defaultValue = true
}

/// How much vertical space the floating bottom nav bar occupies, pushed down the
/// environment so each screen can keep its INTERACTIVE bottom elements (the last
/// scroll item, the map's Go bar) clear of the bar — while its decorative content
/// still bleeds underneath the glass. Zero in landscape (the nav rail is on the
/// side, so nothing floats over the bottom).
private struct BottomBarInsetKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

/// Leading space the FLOATING landscape nav rail occupies, pushed down the
/// environment so a screen's text/interactive content clears the glass rail while
/// its full-bleed background still flows edge-to-edge UNDER it (like the portrait
/// bottom bar uses `bottomBarInset`). Zero in portrait.
private struct RailLeadingInsetKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

/// Shared geometry for the immersive Navigate console.
enum NavLayout {
    /// Vertical space the floating status bar + top control row occupy in the
    /// landscape Navigate console — the viewport's tap hints drop below this so
    /// they never collide with the status bar, E-stop, or Start-Teleop row.
    static let topChromeInset: CGFloat = 78
}

extension EnvironmentValues {
    var isLandscape: Bool {
        get { self[LandscapeKey.self] }
        set { self[LandscapeKey.self] = newValue }
    }

    var bottomBarInset: CGFloat {
        get { self[BottomBarInsetKey.self] }
        set { self[BottomBarInsetKey.self] = newValue }
    }

    var railLeadingInset: CGFloat {
        get { self[RailLeadingInsetKey.self] }
        set { self[RailLeadingInsetKey.self] = newValue }
    }
}
