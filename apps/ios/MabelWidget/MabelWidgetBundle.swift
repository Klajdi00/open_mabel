//
//  MabelWidgetBundle.swift
//  MabelWidgetExtension
//
//  Entry point for the widget extension: the Home/Lock-Screen widget plus the
//  Dynamic Island / Live Activity. Add more entries here to ship more widgets.
//

import WidgetKit
import SwiftUI

@main
struct MabelWidgetBundle: WidgetBundle {
    var body: some Widget {
        MabelStatusWidget()
        MabelLiveActivityWidget()
    }
}
