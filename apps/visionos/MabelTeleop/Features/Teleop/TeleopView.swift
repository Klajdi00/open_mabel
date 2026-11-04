//
//  TeleopView.swift
//  MabelTeleop
//
//  The Teleop tab — clean and camera-first: the head POV is the hero in the
//  centre, the LEFT wrist cam sits to its left and the RIGHT wrist cam to
//  its right (matching the robot's arms), with the small live 3D robot
//  tucked under the right wrist cam. The deck is just Start Teleop and
//  Full Immersive — recording and the task picker live on the floating
//  console that appears the moment teleop starts.
//

import SwiftUI
import Combine

struct TeleopView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openImmersiveSpace) private var openImmersive
    @Environment(\.dismissImmersiveSpace) private var dismissImmersive
    /// Tap any camera tile → its own placeable camera-detail window (zoom + pan).
    @Environment(\.openWindow) private var openWindow

    @StateObject private var miniStage = RobotStage(scale: 0.13)

    // Calibration phase at EVERY teleop entry: starting teleop prompts the picker,
    // which can apply a saved profile, launch the guided arms-forward calibration,
    // or skip explicitly. (It used to appear only on first run — the calibration
    // phase silently disappeared once any profile had been chosen.)
    private enum TeleopSheet: Int, Identifiable { case picker, calibrate; var id: Int { rawValue } }
    @State private var sheet: TeleopSheet?
    @State private var wantsFullImmersion = false

    var body: some View {
        VStack(spacing: 12) {
            // ── L wrist · head POV hero · R wrist (+ mini robot below) ───────
            HStack(spacing: 12) {
                VStack(spacing: 12) {
                    CameraFeedView(name: "L WRIST", compact: true, tint: Theme.Palette.live,
                                   streamURL: streamURL("wrist_left", thumb: true),
                                   udpHost: controller.videoHost,
                                   udpPort: controller.videoUDPPort,
                                   codec: controller.videoCodec,
                                   stream: "wrist_left",
                                   onExpand: { openWindow(id: WindowID.cameraDetail.rawValue,
                                                          value: "wrist_left") })
                        .aspectRatio(4 / 3, contentMode: .fit)
                    Spacer(minLength: 0)
                }
                .frame(width: 230)

                CameraFeedView(name: "HEAD · POV", streamURL: streamURL("main"),
                               udpHost: controller.videoHost,
                               udpPort: controller.videoUDPPort,
                               codec: controller.videoCodec,
                               stream: "main",
                               onExpand: { openWindow(id: WindowID.cameraDetail.rawValue,
                                                      value: "main") })
                    .aspectRatio(16 / 9, contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                VStack(spacing: 12) {
                    CameraFeedView(name: "R WRIST", compact: true, tint: Theme.Palette.clay,
                                   streamURL: streamURL("wrist_right", thumb: true),
                                   udpHost: controller.videoHost,
                                   udpPort: controller.videoUDPPort,
                                   codec: controller.videoCodec,
                                   stream: "wrist_right",
                                   onExpand: { openWindow(id: WindowID.cameraDetail.rawValue,
                                                          value: "wrist_right") })
                        .aspectRatio(4 / 3, contentMode: .fit)
                    Spacer(minLength: 0)
                    // Small live robot pinned to the BOTTOM of the column,
                    // right above the Full Immersive button. Interactive like
                    // the Maps diorama: look at it, drag to rotate, pinch to
                    // zoom. (No backing plate — a translucent card at the
                    // window plane floats in front of recessed 3D content.)
                    RobotStageView(stage: miniStage, interaction: .links)
                        .frame(height: 310)
                }
                .frame(width: 230)
            }
            .padding(.horizontal, 18)
            .padding(.top, 52)

            // ── One simple deck, centred ─────────────────────────────────────
            HStack(spacing: 14) {
                Spacer()
                startButton
                fullImmersiveButton
                Spacer()
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 14)
        }
        .sheet(item: $sheet) { which in
            switch which {
            case .picker:
                ProfilePickerView(
                    onSelect: { profile in
                        controller.applyProfile(profile)
                        sheet = nil
                        Task { await enterTeleop() }
                    },
                    onNew: { sheet = .calibrate },          // swap straight to the guided flow
                    onSkip: {
                        sheet = nil
                        Task { await enterTeleop() }
                    }
                )
                .environmentObject(controller)
            case .calibrate:
                // The flow opens the immersive space + starts tracking itself, so the
                // operator lands in teleop already calibrated.
                CalibrationFlowView().environmentObject(controller)
            }
        }
    }

    /// Open the immersive teleop space (the operator is already past the profile gate).
    private func enterTeleop() async {
        guard !controller.immersiveActive else { return }
        controller.fullImmersion = wantsFullImmersion
        await openImmersive(id: WindowID.immersive.rawValue)
    }

    /// Start/stop teleoperating RIGHT HERE in the window: opens the space in
    /// MIXED style (the room and this window stay visible) so the hands track
    /// and retargeting drives the robot immediately.
    private var startButton: some View {
        Button {
            Haptics.rigid()
            Task {
                if controller.immersiveActive {
                    controller.noteIntendedImmersiveExit()
                    await dismissImmersive()
                } else {
                    // Calibration phase: every entry asks who's driving (apply a
                    // saved reach profile, run the arms-forward capture, or skip).
                    wantsFullImmersion = false
                    sheet = .picker
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: controller.immersiveActive ? "stop.circle.fill" : "play.fill")
                    .font(.system(size: 15, weight: .bold))
                Text(controller.immersiveActive ? "Stop Teleop" : "Start Teleop")
                    .font(.system(size: 15, weight: .bold))
            }
            .foregroundStyle(.black.opacity(0.85))
            .padding(.horizontal, 18).padding(.vertical, 11)
            .background(Theme.accentGradient, in: Capsule())
        }
        .buttonStyle(PressableStyle())
    }

    /// The 180° wrap-around view (full immersion). Starts teleop too if it
    /// isn't already running.
    private var fullImmersiveButton: some View {
        Button {
            Haptics.rigid()
            Task {
                if controller.immersiveActive {
                    controller.fullImmersion = true
                } else {
                    // Same calibration phase as Start Teleop (this button used to
                    // bypass the profile/calibration gate entirely).
                    wantsFullImmersion = true
                    sheet = .picker
                }
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "visionpro")
                    .font(.system(size: 15, weight: .bold))
                Text("Full Immersive")
                    .font(.system(size: 15, weight: .bold))
            }
            .foregroundStyle(Theme.Palette.textPrimary)
            .padding(.horizontal, 18).padding(.vertical, 11)
            .background(Theme.Palette.surfaceHi, in: Capsule())
            .overlay { Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1) }
        }
        .buttonStyle(PressableStyle())
    }





    /// - Parameter thumb: true for the 230 pt wrist tiles. They are rendered at
    ///   a fraction of source size.
    private func streamURL(_ name: String, thumb: Bool = false) -> URL? {
        thumb ? controller.videoThumbURL(name)   // thumbnail tile
              : controller.videoMainURL(name)    // hero quality
    }
}
