//
//  DrivePadDock.swift
//  MabelTeleop
//
//  The Vision Pro's base-drive control: the SAME flat 2-D cluster as the
//  iPhone (TRANSLATE stick · LIFT slider · ROTATE stick, `DriveControls.swift`)
//  on a glass card that folds down to one pill (operator, 2026-10-16: "do not
//  use the 3d joystick — keep things 2D and easy to toggle and similar to the
//  ios; separate the lift control from the joystick").
//
//  It replaced the floating `Joystick3D` pair. Where it is mounted matters
//  more than what it draws:
//    • the window hosts it on a BOTTOM SCENE ORNAMENT, never in the window
//      body — the 3-D stage pops out through the window plane and physically
//      occludes in-window chrome (the same reason the Joint panels live on the
//      trailing ornament). Ornaments are never occluded.
//    • full immersion hides the window, so the immersive space mounts the same
//      dock as a RealityView attachment.
//  The show/hide state is ONE `@AppStorage` key, so folding it in the window
//  folds it in the dome too.
//

import SwiftUI

struct DrivePadDock: View {
    @EnvironmentObject private var controller: RobotController
    @AppStorage(Self.visibleKey) private var visible = true
    @State private var translate: CGVector = .zero
    @State private var yaw: Double = 0

    static let visibleKey = "drivePadVisible"
    var stickSize: CGFloat = 150

    var body: some View {
        Group {
            if visible { card } else { foldedPill }
        }
        .animation(Theme.Motion.snappy, value: visible)
        // The window's rail toggle flips the same key from outside the dock.
        .onChange(of: visible) { _, on in if !on { release() } }
        // A control that disappears must not leave the robot moving: the base
        // and the lift hold their last command until something overwrites it,
        // and a deflected stick that is no longer on screen never sends its
        // release. (`stopBase` on visionOS does not zero the lift — do both.)
        .onDisappear { release() }
        #if DEBUG
        .onAppear { runDebugScript() }
        #endif
    }

    // The pad's three outputs — the ONLY way this dock moves the robot.
    private func setTranslate(_ v: CGVector) {
        translate = v
        controller.drive(translate: translate, yaw: yaw)
    }
    private func setRotate(_ v: Double) {
        yaw = v
        controller.drive(translate: translate, yaw: yaw)
    }
    private func setLift(_ v: Double) { controller.setLiftRate(v) }

    #if DEBUG
    /// debugDrivePad="ly=-0.8:3;rot=0.8:3;lift=1:3" (see DebugDriveScript):
    /// drives the SAME three outputs the controls do. Once per process — the
    /// dock re-appears on every fold/unfold and space switch.
    private static var debugScriptPlayed = false
    private func runDebugScript() {
        guard !Self.debugScriptPlayed,
              let spec = UserDefaults.standard.string(forKey: "debugDrivePad") else { return }
        Self.debugScriptPlayed = true
        DebugDriveScript(spec).play(apply: { st in
            switch st.axis {
            case "lx":  setTranslate(CGVector(dx: st.value, dy: translate.dy))
            case "ly":  setTranslate(CGVector(dx: translate.dx, dy: st.value))
            case "rot": setRotate(st.value)
            default:    setLift(st.value)
            }
        }, release: { release() })
    }
    #endif

    private var card: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "steeringwheel")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Theme.Palette.accent)
                Text("DRIVE")
                    .font(.system(size: 12, weight: .heavy)).tracking(1.4)
                    .foregroundStyle(.white.opacity(0.85))
                Spacer(minLength: 16)
                Button { setVisible(false) } label: {
                    Label("Hide", systemImage: "chevron.down")
                        .font(.system(size: 12, weight: .semibold))
                }
                .buttonStyle(.bordered)
                .buttonBorderShape(.capsule)
                .accessibilityIdentifier("drivePad.hide")
            }
            DrivePad(stickSize: stickSize,
                     liftHeight: controller.state.joints["lift_lower"] ?? 0,
                     spacing: 36,
                     onTranslate: setTranslate, onRotate: setRotate, onLift: setLift)
        }
        .padding(.horizontal, 22).padding(.vertical, 16)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 30, style: .continuous))
        .accessibilityIdentifier("drivePad.card")
    }

    private var foldedPill: some View {
        Button { setVisible(true) } label: {
            Label("Drive Pad", systemImage: "steeringwheel")
                .font(.system(size: 14, weight: .semibold))
                .padding(.horizontal, 6)
        }
        .buttonBorderShape(.capsule)
        .glassBackgroundEffect(in: Capsule(style: .continuous))
        .accessibilityIdentifier("drivePad.show")
    }

    private func setVisible(_ on: Bool) {
        if !on { release() }
        Haptics.select()
        withAnimation(Theme.Motion.snappy) { visible = on }
    }

    private func release() {
        translate = .zero; yaw = 0
        controller.stopBase()
        controller.setLiftRate(0)
    }
}
