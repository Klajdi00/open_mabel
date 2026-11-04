//
//  BodyView.swift
//  MabelTeleop
//
//  The manipulation console — same structure as the iOS Whole Body tab,
//  laid out like the Maps tab: a compact liquid-glass panel on the LEFT and
//  the 3D stage filling the rest. The Firm/Soft toggle floats in a glass
//  capsule at the TOP of the stage, flat on the window plane (no foreground
//  pop), using the same segmented control as every other choice in the app.
//
//    FIRM (impedance): panel + stage. Wrist tracking lets you pinch-drag
//      the arms DIRECTLY on the model (gaze highlight, red drag indicator);
//      Joints shows the sliders instead.
//    SOFT (compliance): full-bleed visualizer; the only choice is the
//      compliant region. Push the robot in the Mac sim or by hand on the
//      real hardware and watch it yield here.
//

import SwiftUI
import Combine
import RealityKit
import simd

struct BodyView: View {
    @EnvironmentObject private var controller: RobotController
    @Environment(\.openWindow) private var openWindow

    // The three axes (same vocabulary as iOS).
    @State private var method: TeleopMethod = .wrist     // Wrist | Joints
    @State private var feel: ControllerType = .impedance // Firm | Soft
    @State private var region: ControlRegion = .arm
    @State private var dragStartJoints: [String: Double]? = nil

    @StateObject private var stage = RobotStage(scale: 0.4, interactive: true)

    var body: some View {
        HStack(spacing: 12) {
            if feel == .impedance {
                leftPanel
                    .frame(width: 270)
            }
            stageArea
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(.horizontal, 14)
        .animation(Theme.Motion.snappy, value: feel)
        .onAppear { applySpec() }
        .onDisappear {
            controller.setNavigation()        // restore driving on exit
            controller.clearExternalForce()
        }
    }

    /// Push the current spec to the bridge (same semantics as iOS: Soft pins
    /// the method to wrist just to engage the compliance controller).
    private func applySpec() {
        controller.setControl(method: feel == .compliance ? .wrist : method,
                              type: feel, region: region)
    }

    /// Direct model dragging is FIRM + wrist-tracking only.
    private var dragEnabled: Bool { feel == .impedance && method == .wrist }

    // MARK: - Stage (Firm/Soft toggle on top, flat on the window plane)

    private var stageArea: some View {
        ZStack(alignment: .top) {
            RobotStageView(
                stage: stage,
                interaction: dragEnabled ? .links : .none,
                onLinkDrag: { link, _, delta, _ in guideDrag(link, delta) },
                onLinkDragEnd: {
                    dragStartJoints = nil
                    Haptics.tap()
                }
            )

            // Firm/Soft (+ region in Soft) — one glass capsule at the top.
            VStack(spacing: 8) {
                HStack(spacing: 10) {
                    SegmentedPicker(options: ControllerType.allCases, label: { $0.title },
                                    selection: $feel) { _ in applySpec(); Haptics.rigid() }
                        .frame(width: 200)
                    if feel == .compliance {
                        SegmentedPicker(options: ControlRegion.allCases, label: { $0.title },
                                        selection: $region) { _ in applySpec(); Haptics.select() }
                            .frame(width: 220)
                    }
                }
                .padding(.horizontal, 12).padding(.vertical, 8)
                .glassBackgroundEffect(in: Capsule())

                if feel == .compliance { softHint }
            }
            .padding(.top, 56)

            // Stage view controls — right-hand side, like the toggles.
            HStack { Spacer(); stageControls.padding(.trailing, 12) }
                .frame(maxHeight: .infinity, alignment: .center)
        }
    }

    private var softHint: some View {
        HStack(spacing: 6) {
            Image(systemName: "hand.draw.fill").font(.system(size: 11, weight: .semibold))
            Text("Compliant — push the robot by hand or Ctrl-drag the Mac sim; it yields live here.")
                .font(.system(size: 12, weight: .semibold)).lineLimit(1)
        }
        .foregroundStyle(Theme.Palette.live)
        .padding(.horizontal, 12).padding(.vertical, 7)
        .glassBackgroundEffect(in: Capsule())
    }

    /// Firm + wrist tracking: pinch-drag a link → the arm's impedance target
    /// follows (the gaze highlight on the link is the affordance; the old
    /// red drag line was dropped — it lagged the interaction).
    private func guideDrag(_ link: RigLink, _ deltaScene: SIMD3<Float>) {
        let d = deltaScene / max(stage.currentScale, 0.01)   // robot-frame metres
        if dragStartJoints == nil {
            dragStartJoints = controller.liveModelJoints
            Haptics.soft()
        }
        guard let s = dragStartJoints else { return }
        let gain = 2.5                                       // rad per metre of drag
        let down = Double(-d.y) * gain
        let dx = Double(d.x) * gain
        let side = link.side
        set("\(side)_arm_1", (s["\(side)_arm_1"] ?? 0) + down)
        set("\(side)_arm_3", (s["\(side)_arm_3"] ?? 0) + (side == "left" ? -dx : dx))
    }

    private func set(_ id: String, _ value: Double) {
        if let spec = RobotModel.spec(id) {
            controller.setJoint(id, min(spec.upper, max(spec.lower, value)))
        } else {
            controller.setJoint(id, value)
        }
    }

    private var stageControls: some View {
        VStack(spacing: 10) {
            StageActionButton(icon: "scope", label: "Center") { stage.center() }
            StageActionButton(icon: "cube.transparent", label: "Volume") {
                openWindow(id: WindowID.modelVolume.rawValue)
            }
        }
    }

    // MARK: - Left panel (FIRM only) — liquid glass, Maps-style

    private var leftPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Manipulate").font(.mabelTitle).foregroundStyle(Theme.Palette.textPrimary)
                    Text("impedance manipulation")
                        .font(.system(size: 10, weight: .medium)).foregroundStyle(Theme.Palette.textTertiary)
                }
                .padding(.top, 4).padding(.bottom, 2)

                SegmentedPicker(options: [TeleopMethod.wrist, .joint],
                                label: { $0 == .wrist ? "Wrist" : "Joints" },
                                selection: $method) { _ in applySpec(); Haptics.select() }

                // The Arm/Upper/Whole region only parameterises COMPLIANCE —
                // it lives with the Soft toggle on the stage. Firm keeps just
                // the stiffness.
                stiffnessControl

                if method == .wrist {
                    wristHint
                } else {
                    JointSections()
                }
            }
            .padding(10)
        }
        .scrollIndicators(.hidden)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.Radius.lg, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: Theme.Radius.lg, style: .continuous)
                .strokeBorder(.white.opacity(0.12), lineWidth: 1)
        }
        .padding(.vertical, 52)
    }

    private var wristHint: some View {
        HStack(spacing: 8) {
            Image(systemName: "eye").font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.Palette.live)
            Text("Look at an arm or hand — it highlights. Pinch and drag directly on the model; the arm springs to follow.")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.Palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(10)
        .background(Theme.Palette.live.opacity(0.08), in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.live.opacity(0.2), lineWidth: 1) }
    }

    private var stiffnessControl: some View {
        HStack(spacing: 8) {
            Image(systemName: "lock.fill").font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary).frame(width: 20)
            Text("Stiffness").font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.Palette.textSecondary).frame(width: 58, alignment: .leading)
            Slider(value: Binding(get: { controller.stiffness },
                                  set: { controller.setStiffness($0) }), in: 0...1)
                .tint(Theme.Palette.accent).controlSize(.mini)
            Text("\(Int(controller.stiffness * 100))%").font(.mabelMonoSmall).foregroundStyle(Theme.Palette.accent).frame(width: 38, alignment: .trailing)
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(Theme.Palette.surface, in: RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: Theme.Radius.md, style: .continuous).strokeBorder(Theme.Palette.hairlineSoft, lineWidth: 1) }
    }
}
