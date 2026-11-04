//
//  ModelVolumeView.swift
//  MabelTeleop
//
//  The robot in a volumetric window: a true 3D MABEL you can grab and carry
//  (pinch-drag), spin about its vertical axis (two-hand twist), and zoom
//  (pinch). The control bar is pinned INSIDE the volume at its front-bottom
//  edge — right above the system window bar, nearest the user — because
//  scene-bottom ornaments on volumes sit at centre depth, buried behind the
//  content and out of comfortable gaze reach.
//

import SwiftUI

struct ModelVolumeView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openWindow) private var openWindow
    @StateObject private var stage = RobotStage(scale: 0.9)

    var body: some View {
        // volumeFit MEASURES the real interior (visionOS restores persisted
        // window sizes, so assumed dimensions sliced the robot) and derives
        // scale, floor, seat and carry clamps from it.
        RobotStageView(stage: stage, fit: false, interaction: .move, volumeFit: true,
                       volumeControls: { AnyView(controlsBar) })
    }

    private var controlsBar: some View {
        HStack(spacing: 10) {
            control("rectangle.compress.vertical", "Refit") {
                stage.setScale(stage.baseScale)
                stage.center(position: stage.homeSeat)
            }
            control("figure.stand", "Life Size") {
                stage.lifeSizeMeasured()   // 1:1 when the box allows; never sliced
            }
            control("arrow.down.to.line", "Place on Floor") {
                stage.placeFeet(atY: stage.volumeFloor)
            }
            control("scope", "Center") {
                stage.center(position: stage.homeSeat)
            }
            Divider().frame(height: 22)
            control("rectangle.on.rectangle", "Console") {
                openWindow(id: WindowID.main.rawValue)
            }
            if let e = stage.measuredExtents {
                Text(String(format: "box %.1f×%.1f×%.1f m · scale %.2f",
                            e.x, e.y, e.z, stage.currentScale))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .glassBackgroundEffect()
    }

    private func control(_ icon: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon).font(.system(size: 13, weight: .bold))
                Text(label).font(.system(size: 13, weight: .semibold))
            }
        }
        .buttonStyle(.bordered)
        .tint(Theme.Palette.accent)
    }
}
