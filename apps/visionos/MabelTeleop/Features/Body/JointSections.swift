//
//  JointSections.swift
//  MabelTeleop
//
//  Individual joint control, grouped into Body · Arms · Hands. The Hands
//  group is simplified: the controls you actually use — open/close (grip)
//  and the wrist for each hand — sit at the top; the 16 individual finger
//  joints per hand are tucked behind a disclosure.
//

import SwiftUI

struct JointSections: View {
    @EnvironmentObject private var controller: RobotController
    @State private var group: JointGroup = .arms
    @State private var showFingers = false

    private let order: [JointGroup] = [.body, .arms, .hands]

    /// Synthetic single-actuator lift: one slider for the whole column.
    private static let liftSpec = JointSpec(
        id: "lift", label: "Lift Column", group: .body, kind: .prismatic,
        lower: 0, upper: RobotController.liftMax)

    var body: some View {
        VStack(spacing: Theme.Space.sm) {
            SegmentedPicker(options: order, label: { $0.label }, selection: $group)

            VStack(spacing: 2) {
                if group == .body {
                    JointSlider(spec: Self.liftSpec, value: Binding(
                        get: { controller.liftHeight },
                        set: { controller.setLiftHeight($0) }
                    ))
                }

                if group == .hands {
                    handsSection
                } else {
                    ForEach(visibleSpecs) { spec in
                        slider(for: spec)
                    }
                }

                if group == .body {
                    Text("Neck follows the headset's head tracking (Teleop).")
                        .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 2)
                }
                SecondaryButton(title: "Reset Region", systemImage: "arrow.counterclockwise") {
                    resetRegion()
                }
                .padding(.top, Theme.Space.sm)
            }
        }
        .animation(Theme.Motion.gentle, value: group)
    }

    // MARK: - Hands: the useful four first, fingers behind a disclosure

    /// Grip (open/close) + wrist for each hand on top; everything else later.
    private var handsSection: some View {
        VStack(spacing: 2) {
            ForEach(primaryHandSpecs) { spec in
                slider(for: spec)
            }
            Text("0 = open · 1 = closed fist")
                .font(.system(size: 11)).foregroundStyle(Theme.Palette.textTertiary)
                .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 2)

            DisclosureGroup(isExpanded: $showFingers) {
                VStack(spacing: 2) {
                    ForEach(fingerSpecs) { spec in
                        slider(for: spec)
                    }
                }
                .padding(.top, 6)
            } label: {
                Label("All Finger Joints", systemImage: "hand.raised.fingers.spread")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
            .tint(Theme.Palette.accent)
            .padding(.top, 8)
        }
    }

    /// Grips and wrists, L/R interleaved — what the operator reaches for. A side wearing the
    /// OpenArm gripper shows its ONE slider (`<side>_gripper`, closed left … open right) instead.
    private var primaryHandIDs: [String] {
        ["left", "right"].map { RobotModel.isRedundantGrip("\($0)_grip") ? "\($0)_gripper" : "\($0)_grip" }
            + ["left_wrist", "right_wrist"]
    }

    private var primaryHandSpecs: [JointSpec] {
        primaryHandIDs.compactMap { RobotModel.spec($0) }
    }

    /// The remaining per-finger joints.
    private var fingerSpecs: [JointSpec] {
        let primary = Set(primaryHandIDs)
        return RobotModel.specs(in: .hands).filter { !primary.contains($0.id) && !RobotModel.isRedundantGrip($0.id) }
    }

    private func slider(for spec: JointSpec) -> some View {
        JointSlider(spec: spec, value: Binding(
            get: { controller.jointTargets[spec.id] ?? 0 },
            set: { v in
                if spec.id.hasSuffix("_grip") {
                    controller.setGrip(spec.id.hasPrefix("left") ? "left" : "right", v)
                } else { controller.setJoint(spec.id, v) }
            }
        ))
    }

    /// Body group hides the raw lift stages (single lift control) and the neck.
    private var visibleSpecs: [JointSpec] {
        RobotModel.specs(in: group).filter { spec in
            guard group == .body else { return true }
            return !spec.id.hasPrefix("lift") && !spec.id.hasPrefix("neck")
        }
    }

    private func resetRegion() {
        if group == .body { controller.setLiftHeight(0) }
        let specs = group == .hands ? primaryHandSpecs + fingerSpecs : visibleSpecs
        for spec in specs {
            controller.setJoint(spec.id, RobotModel.homePose[spec.id] ?? 0)
        }
        Haptics.soft()
    }
}
